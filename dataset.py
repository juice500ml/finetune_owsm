import json
from itertools import chain, product
from pathlib import Path

import librosa
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader

from espnet2.bin.s2t_inference import Speech2Text
from espnet2.tasks.s2t import S2TTask

from utils import preprocessor_add_new_tokens


class FieldworkDataset(Dataset):
    def __init__(self, root, split, tasks, langs, preprocessor):
        assert split in ("train", "dev", "test")
        assert all((t in ("transcription", "underlying", "gloss", "translation")) for t in tasks)

        self.split = split
        self.data = self._parse_data(root, tasks, langs)
        self.preprocessor = preprocessor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data[index]
        row["speech"], _ = librosa.load(row["speech"], mono=True, sr=16_000)
        uid = f"{index:08d}"
        return uid, self.preprocessor(uid=uid, data=row)

    def _parse_data(self, root, tasks, langs):
        # paths = list(root.glob(f"data/*/{self.split}.json"))
        paths = [root / f"data/{lang}/{self.split}.json" for lang in langs]
        return list(chain.from_iterable(
            self._parse_single_data(path, task)
            for path, task in product(paths, tasks)
            if path.exists()
        ))

    def _parse_single_data(self, path, task):
        with open(path) as f:
            meta = json.load(f)

        data = []
        for key, value in meta.items():
            speech = path.parent / "audio" / path.stem / key
            if (self.split != "test") and (librosa.get_duration(filename=speech) > 30.0):
                pass

            text_ctc = value["transcription"]
            text_prev = "<na>"

            lang = path.parent.name
            task_id = {
                "transcription": "asr",
                "gloss": "st_gloss",
                "underlying": "st_underlying",
                "translation": "st_eng",
            }
            text = f"<{lang}><{task_id[task]}>{value[task]}"

            data.append({
                "speech": speech,
                "text": text,
                "text_prev": text_prev,
                "text_ctc": text_ctc,
            })
        return data


class FieldworkDataModule(LightningDataModule):
    def __init__(
            self,
            dataset_path: Path,
            model_name: str,
            batch_size: int,
            new_tokens: list[str],
            tasks: list[str],
            langs: list[str],
            **kwargs,
        ):
        super().__init__()
        self.save_hyperparameters()

        s2t = Speech2Text.from_pretrained(model_name)

        preprocessor_train = S2TTask.build_preprocess_fn(s2t.s2t_train_args, train=True)
        preprocessor_test = S2TTask.build_preprocess_fn(s2t.s2t_train_args, train=False)
        preprocessor_add_new_tokens(preprocessor_train, new_tokens)
        preprocessor_add_new_tokens(preprocessor_test, new_tokens)

        self.collator_train = S2TTask.build_collate_fn(s2t.s2t_train_args, train=True)
        self.collator_test = S2TTask.build_collate_fn(s2t.s2t_train_args, train=False)

        self.unk_id = preprocessor_train.token_id_converter.tokens2ids(["<unk>"])[0]

        self.train_ds = FieldworkDataset(
            dataset_path, "train", tasks, langs, preprocessor_train,
        )

        self.valid_ds = {
            f"{task}_{lang}": FieldworkDataset(
                dataset_path, "dev", [task], [lang], preprocessor_test)
            for task, lang in product(tasks, langs)
        }
        self.valid_ds = {k: v for k, v in self.valid_ds.items() if len(v) > 0}

        self.test_ds = {
            f"{task}_{lang}": FieldworkDataset(
                dataset_path, "test", [task], [lang], preprocessor_test)
            for task, lang in product(tasks, langs)
        }
        self.test_ds = {k: v for k, v in self.test_ds.items() if len(v) > 0}

        del s2t

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=self.collator_train,
        )

    def val_dataloader(self):
        return [
            DataLoader(
                ds,
                batch_size=self.hparams.batch_size,
                shuffle=False,
                num_workers=4,
                collate_fn=self.collator_test,
            )
            for ds in self.valid_ds.values()
        ]

    def test_dataloader(self):
        return [
            DataLoader(
                ds,
                batch_size=1,
                shuffle=False,
                num_workers=4,
                collate_fn=self.collator_test,
            )
            for ds in self.test_ds.values()
        ]
