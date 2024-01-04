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

from utils import model_add_new_tokens
from dataset import FieldworkDataModule


class FinetuneOWSM(LightningModule):
    def __init__(
        self,
        model_name: str,
        valid_ds_names: list[str],
        test_ds_names: list[str],
        new_tokens: list[str],
        new_tokens_initialize: int = None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.s2t = Speech2Text.from_pretrained(model_name, category_sym="<eng>")
        self.model = self.s2t.s2t_model

        model_add_new_tokens(self.model, new_tokens, initialize=new_tokens_initialize)

        self._validation_outputs = defaultdict(list)
        self._valid_ds_names = valid_ds_names
        self._test_outputs = defaultdict(list)
        self._test_ds_names = test_ds_names

    def _log(self, split, key, value):
        if not self.trainer.sanity_checking:
            self.log(f"{split}/{key}", value)

    def training_step(self, batch, batch_idx):
        uids, batch = batch
        loss, output, _ = self.model(**{k: v.to(0) for k, v in batch.items()})
        for key, value in output.items():
            if value is not None:
                self._log("train", key, value.item())
        return loss

    @staticmethod
    def _aggregate_metrics(df):
        metrics = set(df.keys()) - set(["task", "lang"])
        tasks = df["task"].unique()
        langs = df["lang"].unique()
        for metric in metrics:
            _df = df[metric]
            for task in tasks:
                yield f"{metric}_{task}_all", _df[df.task == task].mean()
            for lang in langs:
                yield f"{metric}_full_{lang}", _df[df.lang == lang].mean()
            for task, lang in product(tasks, langs):
                yield f"{metric}_{task}_{lang}", _df[(df.task == task) & (df.lang == lang)].mean()

    def on_validation_epoch_start(self):
        self._validation_outputs.clear()

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        uids, batch = batch
        loss, output, _ = self.model(**{k: v.to(0) for k, v in batch.items()})
        for key, value in output.items():
            self._validation_outputs[key].append(value.item())

        task, lang = self._valid_ds_names[dataloader_idx].split("_")
        self._validation_outputs["task"].append(task)
        self._validation_outputs["lang"].append(lang)

    def on_validation_epoch_end(self):
        df = pd.DataFrame(self._validation_outputs)
        for key, value in self._aggregate_metrics(df):
            self._log("val", key, value)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters())


def main(args):
    # Add new tokens
    new_tokens = ["<st_gloss>", "<st_underlying>"]
    new_tokens += [f"<{p.name}>" for p in (args["dataset_path"] / "data").glob("*")]

    # Dataset & Model
    datamodule = FieldworkDataModule(new_tokens=new_tokens, **args)
    model = FinetuneOWSM(
        new_tokens=new_tokens,
        new_tokens_initialize=datamodule.unk_id,
        valid_ds_names=list(datamodule.valid_ds.keys()),
        test_ds_names=list(datamodule.test_ds.keys()),
        **args,
    )

    # Make it deterministic
    seed_everything(seed=42, workers=True)

    # Callbacks
    checkpoint_callback = ModelCheckpoint(monitor="val/acc_full_all")
    early_stop_callback = EarlyStopping(
        monitor="val/acc_full_all", min_delta=0.00, patience=3, verbose=False, mode="max")

    trainer = Trainer(
        accelerator=args["accelerator"],
        devices=args["devices"],
        fast_dev_run=args["fast_dev_run"],
        max_epochs=args["max_epochs"],
        callbacks=[checkpoint_callback, early_stop_callback],
        deterministic="warn",
        default_root_dir=f"{getcwd()}/exps/{args['exp_name']}_{datetime.today().isoformat()}"
    )
    trainer.fit(model, datamodule=datamodule)
    # trainer.test(ckpt_path="best", datamodule=datamodule)


if __name__ == "__main__":
    parser = ArgumentParser()

    # Dataset & Model
    parser.add_argument("--dataset_path", default=Path("fieldwork"), type=Path, help="Path to dataset folder")
    parser.add_argument("--model_name", default="espnet/owsm_v3", help="Huggingface espnet model name")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--tasks", type=str, nargs="+", default=["transcription", "underlying", "gloss", "translation"])
    parser.add_argument("--langs", type=str, nargs="+", default=["dolg1241", "kama1378", "ainu1240"])

    # Trainer
    parser.add_argument("--exp_name", type=str, default="test", help="Experiment name (Folder to store the results)")
    parser.add_argument("--accelerator",  default="gpu", help="gpu or cpu")
    parser.add_argument("--devices", default=1, help="# of gpus")
    parser.add_argument("--fast_dev_run", type=bool, help="True if debug mode", default=False)
    parser.add_argument("--max_epochs", type=int, default=10, help="Maximum epochs to train")

    args = parser.parse_args()
    print(args)
    main(vars(args))
