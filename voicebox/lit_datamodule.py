import os
from pathlib import Path

from lightning import LightningDataModule
from torch.utils.data import DataLoader
from voicebox.data.audiotext_dataset import AudioTextDataset

from .data.bucketsampler import DistributedBucketSampler


class AudioTextDataModule(LightningDataModule):
    def __init__(
        self,
        trainset={
            "root": "/data/Users/chenhaitao/Codes/leetcode/Voicebox/examples/ljspeech/data",
            "meta": "train.txt",
            "phonesets": "phonesets.txt",
            "melspec_dir": "melspec",
            "text_path": "lab.pt",
        },
        valset={
            "root": "/data/Users/chenhaitao/Codes/leetcode/Voicebox/examples/ljspeech/data",
            "meta": "eval.txt",
            "phonesets": "phonesets.txt",
            "melspec_dir": "melspec",
            "text_path": "lab.pt",
        },
        batch_size=4,
    ):
        super().__init__()

        self.train_set = trainset
        self.val_set = valset

        self.batch_size = batch_size

    def prepare_data(self):
        pass

    def setup(self, stage=None, output_logs=False):
       
        self._train_dataset = AudioTextDataset(
            metadata_path=Path(self.train_set["root"]) / self.train_set['meta'],
            phonesets_path=Path(self.train_set["root"]) / self.train_set['phonesets'],
            melspec_dir=Path(self.train_set["root"]) / self.train_set["melspec_dir"],
            text_path=Path(self.train_set["root"]) / self.train_set["text_path"],
            max_sample=self.train_set["max_eval_sample"],
        )
        self._val_dataset = AudioTextDataset(
            metadata_path=Path(self.val_set["root"]) / self.val_set['meta'],
            phonesets_path=Path(self.val_set["root"]) / self.val_set['phonesets'],
            melspec_dir=os.path.join(self.val_set["root"], self.val_set["melspec_dir"]),
            text_path=os.path.join(self.val_set["root"], self.val_set["text_path"]),
            max_sample=self.val_set["max_eval_sample"],
        )

    def train_dataloader(self):
        batch_size = self.batch_size
        sampler = DistributedBucketSampler(self._train_dataset, batch_size=batch_size)
        return DataLoader(
            self._train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=self._train_dataset.collate,
        )

    def val_dataloader(self):
        return DataLoader(
            self._val_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=self._val_dataset.collate,
        )

    def test_dataloader(self):
        return DataLoader(
            self._test_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=self._test_dataset.collate,
        )
