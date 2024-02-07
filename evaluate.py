from argparse import ArgumentParser
from pathlib import Path

from pytorch_lightning import Trainer

from finetune import FinetuneOWSM
from dataset import FieldworkDataModule


def main(args):
    # Add new tokens
    new_tokens = ["<st_gloss>", "<st_underlying>"]
    new_tokens += [f"<{p.name}>" for p in (args["dataset_path"] / "data").glob("*")]

    # Dataset & Model
    datamodule = FieldworkDataModule(new_tokens=new_tokens, batch_size=1, **args)
    model = FinetuneOWSM.load_from_checkpoint(
        args["checkpoint_path"], batch_size=1, devices=args["devices"],
    )
    trainer = Trainer(
        accelerator="gpu" if args["devices"] > 0 else "cpu",
        devices=args["devices"],
        default_root_dir=Path(args["checkpoint_path"]).parent.parent,
    )
    trainer.test(model=model, datamodule=datamodule)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--dataset_path", default=Path("fieldwork"), type=Path, help="Path to dataset folder")
    parser.add_argument("--model_name", default="espnet/owsm_v3.1_ebf_base", help="Huggingface espnet model name")
    parser.add_argument("--tasks", type=str, nargs="+", default=["transcription", "underlying", "gloss", "translation"])
    parser.add_argument("--langs", type=str, nargs="+", default=["taba1259", "tond1251", "kach1280", "arta1239", "vera1241", "sanz1248", "sumb1241", "nort2641", "kara1499", "mand1415", "tehr1242", "taul1251", "ainu1240", "even1259", "dolg1241", "kama1378", "selk1253", "komn1238", "sout2856", "apah1238", "teop1238", "jeju1234", "ruul1235", "sumi1235", "beja1238", "kaka1265", "goro1270", "savo1255", "texi1237", "pnar1238", "nngg1234", "arap1274", "port1286", "trin1278", "bora1263", "slav1254", "balk1252"])
    parser.add_argument("--checkpoint_path", type=Path, help="Path to checkpoint")
    parser.add_argument("--devices", type=int, default=0, choices=(0, 1), help="Number of GPUs to use")

    args = parser.parse_args()
    print(args)
    main(vars(args))
