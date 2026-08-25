import numpy as np
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.nn.functional as F

import kagglehub
from kagglehub import KaggleDatasetAdapter

from transformers import get_cosine_schedule_with_warmup

from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter

from pathlib import Path
import shutil

MAX_SEQ_LEN = 265


class BufferedSampler:
    def __init__(self, probs: torch.Tensor, buffer_size: int = 100_000):
        self.probs = probs
        self.buffer_size = buffer_size
        self._buf = None
        self._pos = 0

    def _refill(self):
        self._buf = torch.multinomial(
            self.probs, self.buffer_size, replacement=True
        )
        self._pos = 0

    @torch.no_grad()
    def sample(self, n: int) -> torch.Tensor:
        if self._buf is None or self._pos + n > self._buf.numel():
            self._refill()
            if n > self._buf.numel():
                return torch.multinomial(self.probs, n, replacement=True)

        out = self._buf[self._pos : self._pos + n]
        self._pos += n
        return out


class PseudoFastSampler:
    def __init__(self, probs: torch.Tensor, buffer_size: int = 100_000):
        self.probs = probs
        self.buffer_size = buffer_size
        self._buf = torch.multinomial(
            self.probs, self.buffer_size, replacement=True
        )
        self._pos = 0

    def _refill(self):
        self._pos = 0

    @torch.no_grad()
    def sample(self, n: int) -> torch.Tensor:
        if self._buf is None or self._pos + n > self._buf.numel():
            self._refill()
            if n > self._buf.numel():
                return torch.multinomial(self.probs, n, replacement=True)

        out = self._buf[self._pos : self._pos + n]
        self._pos += n
        return out


class TrainDataset(Dataset):
    NEGATIVE_SAMPLE_SIZE = 5

    def _calc_popularity(self, data):
        hour = 24  # [0.5, 1, 2, 3, 6, 12, 24]
        decay = 1.0
        tau = decay ** (1 / (60 * 60) / (hour / 24))

        popularity = (
            data.select(
                "item_ind",
                (tau ** (data["timestamp"].max() - pl.col("timestamp"))).alias(
                    "value"
                ),
            )
            .group_by("item_ind")
            .agg(pl.col("value").sum().alias("popularity"))
        )

        popularity = popularity.with_columns(
            (pl.col("popularity") / pl.col("popularity").sum()).alias(
                "popularity"
            )
        )

        self.items = popularity["item_ind"].to_torch()
        self.popolarities_values = popularity["popularity"].to_torch()

        return popularity

    def __init__(self, likes, dislikes, negatives_count):
        self.negatives_count = negatives_count
        self._calc_popularity(likes)

        likes = (
            likes.sort(["uid", "timestamp"])
            .group_by("uid")
            .agg(pl.col("item_ind").alias("item_inds"))
        )
        likes = likes.with_columns(
            pl.col("item_inds")
            .list.unique(maintain_order=True)
            .alias("item_inds")
        )

        print(likes.select(pl.col("item_inds").list.len().max()).item())

        likes = likes.with_columns(
            pl.col("item_inds").list.tail(MAX_SEQ_LEN - 1).alias("item_inds")
        )

        dislikes = (
            dislikes.sort(["uid", "timestamp"])
            .group_by("uid")
            .agg(pl.col("item_ind").alias("item_inds"))
        )
        dislikes = dislikes.with_columns(
            pl.col("item_inds")
            .list.unique(maintain_order=True)
            .alias("item_inds")
        )

        data = likes.join(
            dislikes.rename({"item_inds": "dislike_item_inds"}),
            on="uid",
            how="left",
        )

        data = data.with_columns(
            pl.col("dislike_item_inds").fill_null(pl.lit([]))
        )
        self.data = data

        self.likes = [x.to_torch() for x in self.data["item_inds"]]
        self.dislikes = [x.to_torch() for x in self.data["dislike_item_inds"]]
        self.sampler = BufferedSampler(self.popolarities_values, 500_000)
        # self.sampler = PseudoFastSampler(self.popolarities_values, 500_000)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input = torch.cat([torch.tensor([1]), self.likes[idx][:-1]])
        target = self.likes[idx]

        T = len(target)
        negatives = self.dislikes[idx]

        #         while len(negatives) < T * self.negatives_count:
        #             idx = self.sampler.sample(T * self.negatives_count - len(negatives) + 20)

        #             sampled_items = self.items[idx]

        #             mask = ~torch.isin(sampled_items, negatives)
        #             negatives = torch.cat([negatives, sampled_items[mask]])

        #             break

        idx = self.sampler.sample(
            T * self.negatives_count - len(negatives) + 20
        )
        sampled_items = self.items[idx]

        mask = ~torch.isin(sampled_items, negatives)
        negatives = torch.cat([negatives, sampled_items[mask]])

        negatives = negatives[: T * self.negatives_count]
        negatives = F.pad(
            negatives,
            (0, T * self.negatives_count - negatives.numel()),
            value=0,
        )

        negatives[:] = negatives[
            torch.randperm(negatives.size(0), device=negatives.device)
        ]
        negatives = negatives.reshape(T, self.negatives_count)

        return {"input": input, "target": target, "negatives": negatives}


class ValDataset(Dataset):
    def __init__(self, train_data, val_data):
        train_data = (
            train_data.sort(["uid", "timestamp"])
            .group_by("uid")
            .agg(pl.col("item_ind").alias("item_inds"))
        )
        train_data = train_data.with_columns(
            pl.col("item_inds")
            .list.unique(maintain_order=True)
            .alias("item_inds")
        )

        train_data = train_data.with_columns(
            pl.col("item_inds").list.tail(MAX_SEQ_LEN - 1).alias("item_inds")
        )

        val_data = (
            val_data.sort(["uid", "timestamp"])
            .group_by("uid")
            .agg(pl.col("item_ind").alias("item_inds"))
        )
        val_data = val_data.with_columns(
            pl.col("item_inds")
            .list.unique(maintain_order=True)
            .alias("item_inds")
        )

        data = val_data.join(
            train_data.rename({"item_inds": "history_item_inds"}),
            on="uid",
            how="left",
        )

        missing = data.select(
            pl.col("history_item_inds").is_null().sum()
        ).item()
        assert (
            missing == 0
        ), f"{missing} users in val_data have no history in train_data"

        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "history": self.data["history_item_inds"][idx].to_torch(),
            "target": self.data["item_inds"][idx].to_torch(),
        }


def collate_fn(batch):
    result = {}

    for name in ["history", "input", "target"]:
        if name in batch[0]:
            ls = [x[name] for x in batch]
            result[name] = pad_sequence(ls, batch_first=True, padding_value=0)

    if "negatives" in batch[0]:
        neg_list = [x["negatives"] for x in batch]
        K = neg_list[0].size(1)
        B, T_max = result["input"].shape

        negatives = torch.full((B, T_max, K), 0, dtype=torch.long)
        for i, n in enumerate(neg_list):
            T_i = n.size(0)
            negatives[i, :T_i, :] = n
        result["negatives"] = negatives

    return result
