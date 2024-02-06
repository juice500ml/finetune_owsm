from argparse import ArgumentParser
from collections import defaultdict
from datetime import datetime
from itertools import product
from os import getcwd
from pathlib import Path

import pandas as pd
import torch
from espnet2.bin.s2t_inference import Speech2Text
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torchmetrics.text import WordErrorRate, CharErrorRate, SacreBLEUScore

from utils import model_add_new_tokens, converter_tokenizer_add_new_tokens, beam_search_add_new_tokens
from dataset import FieldworkDataModule


class FinetuneOWSM(LightningModule):
    def __init__(
        self,
        model_name: str,
        valid_ds_names: list[str],
        test_ds_names: list[str],
        unseen_langs: set[str],
        lr: float,
        new_tokens: list[str],
        new_tokens_initialize: int = None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.s2t = Speech2Text.from_pretrained(model_name)
        self.model = self.s2t.s2t_model

        # Add new tokens
        model_add_new_tokens(self.model, new_tokens, initialize=new_tokens_initialize)
        self.s2t.converter, self.s2t.tokenizer = converter_tokenizer_add_new_tokens(self.s2t.s2t_train_args, new_tokens)
        beam_search_add_new_tokens(self.s2t.beam_search, self.model, new_tokens)

        self._valid_ds_names = valid_ds_names
        self._test_ds_names = test_ds_names

        self._val_outputs = []
        self._test_outputs = []

        self.unseen_langs = unseen_langs
        self.metrics = {
            "wer": WordErrorRate(),
            "cer": CharErrorRate(),
            "sacrebleu": SacreBLEUScore(),
        }

    def _log(self, split, key, value, verbose=False, **kwargs):
        if not self.trainer.sanity_checking:
            self.log(f"{split}/{key}", value, **kwargs)
            if verbose:
                print(f"{split}/{key}", value)

    def _log_items(self, split, df, verbose=False):
        metrics = sorted(set(df.keys()) - set(["task", "lang", "ref", "hyp"]))

        metric_values = defaultdict(list)
        for _, row in df.iterrows():
            seen = "seen" if row["lang"] in self.unseen_langs else "unseen"
            for metric in metrics:
                for t, l in product([row["task"], "full"], [row["lang"], seen, "all"]):
                    metric_values[f"{metric}_{t}_{l}"].append(row[metric])

        results = list()
        for key, values in metric_values.items():
            v = sum(values) / len(values)
            self._log(split, key, v, verbose=verbose, sync_dist=True)
            results.append({"name": f"{split}/{key}", "value": v})
        return pd.DataFrame(results)

    def training_step(self, batch, batch_idx):
        uids, batch = batch
        device = next(self.model.parameters()).device
        loss, output, _ = self.model(**{k: v.to(device) for k, v in batch.items()})
        for key, value in output.items():
            if value is not None:
                self._log("train", key, value.item(), on_step=True)
        return loss

    def on_validation_epoch_start(self):
        self._val_outputs.clear()

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        uids, batch = batch
        task, lang = self._valid_ds_names[dataloader_idx].split("_")

        device = next(self.model.parameters()).device
        _, outputs, _ = self.model(**{k: v.to(device) for k, v in batch.items()})
        outputs = {k: v.item() for k, v in outputs.items() if v is not None}
        outputs.update({"task": task, "lang": lang})
        self._val_outputs.append(outputs)

    def _gather(self, outputs):
        if self.trainer.num_devices > 1:
            outputs = self.all_gather(outputs)
            flattened_outputs = []
            for output in outputs:
                metrics = set(output.keys()) - set(["task", "lang"])
                values = [output[m].detach().tolist() for m in metrics]
                for vs in zip(*values):
                    flattened_outputs.append({k: v for k, v in zip(metrics, vs)})
                    flattened_outputs[-1].update({
                        "task": output["task"], "lang": output["lang"],
                    })
            return flattened_outputs
        else:
            return outputs

    def on_validation_epoch_end(self):
        _val_outputs = self._gather(self._val_outputs)
        self._log_items("val", pd.DataFrame(_val_outputs))

    def _ids2text(self, text_ids):
        text_ids = [i for i in text_ids if i >= 0]
        text_toks = self.s2t.converter.ids2tokens(text_ids)
        text_toks = [t for t in text_toks if not (t[0] == "<" and t[-1] == ">")]
        return self.s2t.tokenizer.tokens2text(text_toks)

    def on_test_epoch_start(self):
        self._test_outputs.clear()

    def test_step(self, batch, batch_idx, dataloader_idx):
        uids, batch = batch
        assert len(uids) == 1

        lang_id, task_id, *text_ids = batch["text"][0]
        lang_tok = self.model.token_list[lang_id]
        task_tok = self.model.token_list[task_id]
        ref = self._ids2text(text_ids)

        device = next(self.model.parameters()).device
        self.s2t.beam_search.to(device)
        self.s2t.device = device
        hyp = self.s2t(
            batch["speech"][0].to(device),
            lang_sym=lang_tok,
            task_sym=task_tok,
        )[0][3]

        task, lang = self._test_ds_names[dataloader_idx].split("_")
        outputs = {
            metric_name: metric([hyp], [ref]).item()
            for metric_name, metric in self.metrics.items()
        }
        outputs.update({"task": task, "lang": lang, "ref": ref, "hyp": hyp})
        self._test_outputs.append(outputs)

    def on_test_epoch_end(self):
        assert self.trainer.num_devices < 2
        df = pd.DataFrame(self._test_outputs)
        df.to_csv(Path(self.logger.log_dir) / "test_inference.csv")
        self._log_items("test", df, verbose=True).to_csv(Path(self.logger.log_dir) / "test_metrics.csv")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)


def main(args):
    # Add new tokens
    new_tokens = ["<st_gloss>", "<st_underlying>"]
    new_tokens += [f"<{p.name}>" for p in (args["dataset_path"] / "data").glob("*")]

    # Dataset & Model
    datamodule = FieldworkDataModule(new_tokens=new_tokens, **args)
    unseen_langs = set(args["langs"]) - set([k.split("_")[1] for k in datamodule.valid_ds.keys()])
    model = FinetuneOWSM(
        new_tokens=new_tokens,
        new_tokens_initialize=None,
        valid_ds_names=list(datamodule.valid_ds.keys()),
        test_ds_names=list(datamodule.test_ds.keys()),
        unseen_langs=unseen_langs,
        **args,
    )

    # Make it deterministic
    seed_everything(seed=42, workers=True)

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val/acc_full_all", save_top_k=1, save_last=True, mode="max")
    early_stop_callback = EarlyStopping(
        monitor="val/acc_full_all", min_delta=0.00, patience=3, verbose=False, mode="max")

    trainer = Trainer(
        accelerator="gpu" if args["devices"] > 0 else "cpu",
        devices=args["devices"],
        fast_dev_run=args["fast_dev_run"],
        max_epochs=args["max_epochs"],
        callbacks=[checkpoint_callback, early_stop_callback],
        deterministic="warn",
        default_root_dir=f"{getcwd()}/exps/{args['exp_name']}_{datetime.today().isoformat()}"
    )
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    parser = ArgumentParser()

    # Dataset & Model
    parser.add_argument("--dataset_path", default=Path("fieldwork"), type=Path, help="Path to dataset folder")
    parser.add_argument("--model_name", default="espnet/owsm_v3.1_ebf_base", help="Huggingface espnet model name")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--tasks", type=str, nargs="+", default=["transcription", "underlying", "gloss", "translation"])
    parser.add_argument("--langs", type=str, nargs="+", default=["taba1259", "tond1251", "kach1280", "arta1239", "vera1241", "sanz1248", "sumb1241", "nort2641", "kara1499", "mand1415", "tehr1242", "taul1251", "ainu1240", "even1259", "dolg1241", "kama1378", "selk1253", "komn1238", "sout2856", "apah1238", "teop1238", "jeju1234", "ruul1235", "sumi1235", "beja1238", "kaka1265", "goro1270", "savo1255", "texi1237", "pnar1238", "nngg1234", "arap1274", "port1286", "trin1278", "bora1263", "slav1254", "balk1252"])

    # Trainer
    parser.add_argument("--exp_name", type=str, default="test", help="Experiment name (Folder to store the results)")
    parser.add_argument("--devices", type=int, default=1, help="# of gpus")
    parser.add_argument("--fast_dev_run", action="store_true", help="Flag for debug mode")
    parser.add_argument("--max_epochs", type=int, default=10, help="Maximum epochs to train")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for AdamW")

    args = parser.parse_args()
    print(args)
    main(vars(args))
