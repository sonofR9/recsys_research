import functools
import heapq
import logging
import os
import pathlib
import random
from dataclasses import dataclass, field
from typing import Any

import click
import numpy as np
import polars as pl
import torch

from .data import Data, preprocess

logger = logging.getLogger(__name__)


@dataclass(order=True)
class CheckpointEntry:
    score: float
    epoch: int = field(compare=False)
    path: str = field(compare=False)
    state_dict: dict[str, Any] = field(compare=False, repr=False)


class TopKCheckpoints:
    def __init__(self, k: int, checkpoint_dir: str, exp_name: str):
        self._k = k
        self._checkpoint_dir = checkpoint_dir
        self._exp_name = exp_name
        self._heap: list[CheckpointEntry] = []

    def update(
        self, score: float, epoch: int, state_dict: dict[str, Any]
    ) -> bool:
        path = f"{self._checkpoint_dir}/{self._exp_name}_epoch{epoch}.pth"
        entry = CheckpointEntry(
            score=score, epoch=epoch, path=path, state_dict=state_dict
        )

        if len(self._heap) < self._k:
            heapq.heappush(self._heap, entry)
            torch.save(state_dict, path)
            logger.debug(
                f"Saved checkpoint epoch {epoch} with recall@100={score:.4f}"
            )
            return True

        if score > self._heap[0].score:
            removed = heapq.heapreplace(self._heap, entry)
            if os.path.exists(removed.path):
                os.remove(removed.path)
            torch.save(state_dict, path)
            logger.debug(
                f"Saved checkpoint epoch {epoch} with recall@100={score:.4f}, removed epoch {removed.epoch}"
            )
            return True

        return False

    def best(self) -> CheckpointEntry | None:
        if not self._heap:
            return None
        return max(self._heap, key=lambda e: e.score)


def setup_environment(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")

    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(False)


class MultiOptimizer:
    def __init__(self, optimizers: list[torch.optim.Optimizer]):
        self.optimizers = optimizers

    def zero_grad(self):
        for opt in self.optimizers:
            opt.zero_grad()

    def step(self):
        for opt in self.optimizers:
            opt.step()

    def state_dict(self):
        return [opt.state_dict() for opt in self.optimizers]

    def load_state_dict(self, state_dicts: list[dict]):
        for opt, state_dict in zip(self.optimizers, state_dicts):
            opt.load_state_dict(state_dict)


def load_data(
    data_dir: str,
    size: str,
    interaction: str,
    max_seq_len: int,
    train_days: int | None,
    full_train: bool,
) -> tuple[Data, pl.DataFrame]:
    data_path = pathlib.Path(data_dir) / "sequential" / size / interaction
    df = pl.scan_parquet(data_path.with_suffix(".parquet"))

    embeddings_path = pathlib.Path(data_dir) / "embeddings.parquet"
    pretrained_embeddings = (
        pl.scan_parquet(embeddings_path)
        if embeddings_path.exists()
        else None
    )

    data = preprocess(
        df,
        pretrained_embeddings,
        interaction,
        max_seq_len=max_seq_len,
        train_days=train_days,
        full_train=full_train,
    )
    train_df = data.train.collect(engine="streaming")
    return data, train_df


def prepare_eval_df(
    train_df: pl.DataFrame, val_df: pl.DataFrame
) -> pl.DataFrame:
    return train_df.join(val_df, on="uid", how="inner", suffix="_valid").select(
        pl.col("uid"),
        pl.col("item_id").alias("item_id_train"),
        pl.col("item_id_valid"),
    )


def common_options(func):
    @click.option(
        "--data_dir",
        required=True,
        type=str,
        default="../../data/",
        show_default=True,
    )
    @click.option(
        "--size",
        required=True,
        type=click.Choice(["50m", "500m", "5b"]),
        default="50m",
        show_default=True,
    )
    @click.option(
        "--batch_size", required=True, type=int, default=256, show_default=True
    )
    @click.option(
        "--max_seq_len",
        required=False,
        type=int,
        default=200,
        show_default=True,
    )
    @click.option(
        "--train_days",
        required=False,
        type=int,
        default=None,
        help="Train on first N days, validate on day N+1",
    )
    @click.option(
        "--full_train",
        is_flag=True,
        default=False,
        help="Train on full dataset without time split",
    )
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper
