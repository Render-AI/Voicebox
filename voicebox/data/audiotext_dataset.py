import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset


class AudioTextDataset(Dataset):
    """dataset class for text tokens to semantic model training."""

    def __init__(
        self,
        metadata_path: Union[str, List[str]],
        phonesets_path,
        melspec_dir: Union[str, List[str]],
        text_path: Union[str, List[str]],
        max_sample=None,
    ) -> None:
        super().__init__()

        self.dataframe: pd.DataFrame = pd.read_csv(metadata_path, sep="|", header=None)
        self.melspec_dir = Path(melspec_dir)
        self.text = torch.load(text_path)

        self.phone2id = {}
        with open(phonesets_path, "r") as fin:
            for idx, line in enumerate(fin):
                phone = line.strip()
                self.phone2id[phone] = idx

        if max_sample:
            self.dataframe = self.dataframe[:max_sample]

        self.PAD: int = 511

        # a little diff with paper
        self.mel_mean = -5.365301
        self.mel_std = 2.0389206

    def __len__(self) -> int:
        return len(self.dataframe)

    def get_aligned_phone_id(self, phone_info):
        phones = []
        ids = []
        for phone, (start, end) in phone_info:
            phones = phones + [phone] * int((end + 0.001 - start) * 100)
            ids = ids + [self.phone2id[phone]] * int((end + 0.001 - start) * 100)
        return phones, ids

    def __getitem__(self, idx: int) -> Dict:
        row = self.dataframe.iloc[idx]
        wav_stem = Path(row[0]).stem

        # melspec input
        melspec = torch.load((self.melspec_dir / wav_stem).with_suffix(".pt"))  # 1 * 128 * T

        # aliged phones input
        phone_info = self.text[wav_stem]
        phones, phones_ids = self.get_aligned_phone_id(phone_info)
        phones_ids = torch.tensor(phones_ids, dtype=torch.long)

        assert abs(melspec.shape[-1] - phones_ids.shape[0]) <= 6, f"Melspec shape: {melspec.shape}, phones_ids shape: {phones_ids.shape}"

        min_length = min(melspec.shape[-1], phones_ids.shape[0])

        melspech = melspec[:, :, :min_length]
        phones_ids = phones_ids[:min_length]

        length = melspec.shape[-1]

        return {
            "id": idx,
            "length": length,
            "melspec": melspec[0],             # 128 * T
            "aligned_phones_ids": phones_ids,
        }

    def get_sample_length(self, idx: int):
        row = self.dataframe.iloc[idx]
        wav_stem = Path(row[0]).stem
        melspec = torch.load((self.melspec_dir / wav_stem).with_suffix(".pt"))
        sec = 1.0 * melspec.shape[-1] / 100
        return sec

    def collate(self, features: List[Dict]) -> Dict:
        sample_index: List[int] = []
        phones_ids: List[torch.LongTensor] = []
        melspec: List[torch.FloatTensor] = []
        length: List[int] = []

        for feature in features:
            sample_index.append(feature["id"])
            phones_ids.append(feature["aligned_phones_ids"])
            melspec.append(feature["melspec"])
            length.append(feature["length"])

        batch_size: int = len(sample_index)
        max_length: int = max(length)

        length_t: torch.Tensor = torch.tensor(length, dtype=torch.long)

        # collate melspec
        melspec_t: torch.Tensor = torch.zeros(
            (batch_size, 128, max_length), dtype=torch.float
        )
        for i, frame_seq in enumerate(melspec):
            melspec_t[i, :, : frame_seq.shape[-1]] = (frame_seq - self.mel_mean) / self.mel_std


        # collate text
        phones_ids_t: torch.Tensor = (
            torch.zeros((batch_size, max_length), dtype=torch.long) + self.PAD
        )
        for i, frame_seq in enumerate(phones_ids):
            phones_ids_t[i, : frame_seq.shape[0]] = frame_seq


        return {
            "ids": sample_index,                    # List[int]
            "lengths": length_t,                    # bs * max_semantic_ids_length
            "melspec": melspec_t.transpose(1,2),    # bs * T * 128
            "aligned_phones_ids": phones_ids_t,     # bs * T
        }

    def _read_semantic_tokens(self, semantic_token_path: str) -> List[List[int]]:
        semantic_tokens: List[List[int]] = []
        with open(semantic_token_path, "r") as f:
            for line in f:
                semantic_tokens.append([int(x) for x in line.split(" ")])
        return semantic_tokens


if __name__ == "__main__":
    # dataset = Semantic2AcousticDataset(metadata_path='data/ljspeech_prod',
    #                                    semantic_token_path='data/ljspeech_prod/km_semtok/ljspeech_0_1.km')

    dataset = Semantic2AcousticDataset(
        metadata_path=[
            "data/libritts_train_clean_100",
            "data/libritts_train_clean_360",
            "data/libritts_train_other_500",
        ],
        semantic_token_path=[
            "data/libritts_train_clean_100/km_semtok/libritts_train_clean_100.km",
            "data/libritts_train_clean_360/km_semtok/libritts_train_clean_360.km",
            "data/libritts_train_other_500/km_semtok/libritts_train_other_500.km",
        ],
    )
    sample1 = dataset.__getitem__(0)
    sample2 = dataset.__getitem__(6)
    sample = dataset.collate([sample1, sample2])

    print(sample["semantic_ids"].size())
    print(sample["semantic_ids_len"])
    print(sample["acoustic_ids"].size())
    print(sample["acoustic_ids_len"])
    print("found", len(dataset), "samples")
