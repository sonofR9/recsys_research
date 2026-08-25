from __future__ import annotations

import hashlib
import logging
import os
from abc import abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dcn.config.experiment import Experiment
from dcn.config.settings import DataloaderConfig
from dcn.data import BucketShuffleSampler, SequenceDataset, collate_sequence_batch
from dcn.nn.ple import PiecewiseLinearEncoder
from dcn.training import EpochTrainer, register_stable_optimizer_groups
from utils.global_config import config as global_config
from neuralrec.run.callbacks import (
    CheckpointCallback,
    ValidationCallback,
)

logger = logging.getLogger(__name__)

_COUNTER_BINS = 32
_COUNTER_FIT_ROWS = 1_000_000


@dataclass
class SequenceExperiment(Experiment):
    """An experiment whose batches are user sequences rather than loose events."""

    num_epochs: int = 3
    lr_schedule_horizon_epochs: int | None = None
    max_seq_len: int = 100
    min_seq_len: int = 2
    window: Literal["whole", "sliding", "next_item"] = "sliding"
    stride: float = 1.0
    validation_days: int = 1
    validation_interval_seconds: int | None = None
    _prepared_train_iterator: Iterator | None = field(
        default=None, init=False, repr=False
    )

    embedding_learning_rate: float = 1e-2
    deep_learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            self.lr_schedule_horizon_epochs is not None
            and self.lr_schedule_horizon_epochs < 1
        ):
            raise ValueError("lr_schedule_horizon_epochs must be positive")

    def settings_defaults(self) -> dict[str, Any]:
        return {
            **super().settings_defaults(),
            "dataloader": DataloaderConfig(
                batch_size=64, val_batch_size=64, num_workers=8, prefetch_factor=4
            ),
        }

    @property
    @abstractmethod
    def sequence_columns(self) -> list[str]: ...

    @property
    def emit_user_column(self) -> bool:
        """:class:`SequenceDataset` groups by user without emitting the id, so
        a model or metric that keys on it has to ask."""
        return False

    @property
    def row_filter(self) -> pl.Expr | None:
        """Events to keep. Applied before histories are built, so a dropped
        event is absent from the history too."""
        return None

    @cached_property
    def train_and_validation_days(self) -> tuple[list[int], list[int]]:
        start_day, end_day = self.training_day_bounds
        available = [
            day
            for day in self.dataset_manager.get_available_days()
            if start_day <= day <= end_day
        ]
        if self.validation_interval_seconds is not None:
            cutoff = self.validation_cutoff_timestamp
            train = [
                day
                for day in available
                if day in self._day_timestamp_bounds
                and self._day_timestamp_bounds[day][0] < cutoff
            ]
            validation = [
                day
                for day in available
                if day in self._day_timestamp_bounds
                and self._day_timestamp_bounds[day][1] >= cutoff
            ]
            return train, validation

        held_out = min(self.validation_days, max(len(available) - 1, 0))
        if held_out == 0:
            logger.warning(
                "Only %s day(s) available; validating on train", len(available)
            )
            return available, available
        return available[:-held_out], available[-held_out:]

    @cached_property
    def _day_timestamp_bounds(self) -> dict[int, tuple[int, int]]:
        columns = [self.artifacts.timestamp_column]
        if self.row_filter is not None:
            columns.extend(self.row_filter.meta.root_names())
        columns = list(dict.fromkeys(columns))
        bounds = {}
        for day, path in self.dataset_manager.day_to_path.items():
            frame = pl.read_parquet(path, columns=columns)
            if self.row_filter is not None:
                frame = frame.filter(self.row_filter)
            if frame.height:
                timestamp = frame[self.artifacts.timestamp_column]
                bounds[day] = (int(timestamp.min()), int(timestamp.max()))
        return bounds

    @cached_property
    def validation_cutoff_timestamp(self) -> int:
        interval = self.validation_interval_seconds
        if interval is None or interval <= 0:
            raise ValueError("validation_interval_seconds must be positive")
        start_day, end_day = self.training_day_bounds
        maxima = [
            maximum
            for day, (_, maximum) in self._day_timestamp_bounds.items()
            if start_day <= day <= end_day
        ]
        if not maxima:
            raise ValueError("no events remain for the validation interval split")
        return max(maxima) - interval

    def row_filter_for_split(self, split: str) -> pl.Expr | None:
        row_filter = self.row_filter
        if self.validation_interval_seconds is None:
            return row_filter
        timestamp = pl.col(self.artifacts.timestamp_column)
        if split in {"train", "true_metric_query"}:
            boundary = timestamp < self.validation_cutoff_timestamp
        elif split in {"val", "validation"}:
            boundary = timestamp >= self.validation_cutoff_timestamp
        else:
            return row_filter
        return boundary if row_filter is None else row_filter & boundary

    def _sequence_cache_dir(
        self,
        split: str,
        window: str,
        days: list[int],
        row_filter: pl.Expr | None,
    ) -> Path:
        shape = "\n".join(
            [
                *self.sequence_columns,
                str(row_filter),
                str(self.emit_user_column),
                f"{window}:{self.max_seq_len}:{self.min_seq_len}:{self.stride}",
                ",".join(map(str, days)),
            ]
        )
        digest = hashlib.sha1(shape.encode()).hexdigest()[:10]
        return self.dataset_cache_dir / "sequences" / f"{split}_{digest}"

    def make_sequence_loader(
        self,
        days: list[int],
        *,
        split: str,
        batch_size: int,
        shuffle: bool,
        window: Literal["whole", "sliding", "next_item"] | None = None,
        num_workers: int | None = None,
        pin_memory: bool | None = None,
    ) -> DataLoader:
        window = window or self.window
        row_filter = self.row_filter_for_split(split)
        dataset = SequenceDataset(
            [self.dataset_manager.day_to_path[day] for day in days],
            self.sequence_columns,
            self._sequence_cache_dir(split, window, days, row_filter),
            user_column=self.user_column,
            timestamp_column=self.artifacts.timestamp_column,
            max_seq_len=self.max_seq_len,
            min_seq_len=self.min_seq_len,
            window=window,
            stride=self.stride,
            emit_user_column=self.emit_user_column,
            row_filter=row_filter,
            invalidate_cache=self.invalidate_cache,
        )
        # Sequences are stored bucket-major and read a bucket at a time, so a
        # global shuffle would thrash the reads.
        sampler = BucketShuffleSampler(dataset) if shuffle else None
        workers = self.dataloader.num_workers if num_workers is None else num_workers
        return self.dataset_manager.create_dataloader(
            dataset=dataset,
            collate_fn=collate_sequence_batch,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=workers,
            prefetch_factor=self.dataloader.prefetch_factor,
            pin_memory=(
                os.environ.get("DCN_GPU_LOCK_SLOT") is None
                if pin_memory is None
                else pin_memory
            ),
        )

    def make_cutoff_query_loader(self, train_days: list[int]) -> DataLoader:
        """One row per user: their state at the cutoff. The training loader
        slides, and scoring a mid-history state as if it were the cutoff
        measures a different question."""
        return self.make_sequence_loader(
            train_days,
            split="true_metric_query",
            batch_size=self.dataloader.val_batch_size,
            shuffle=False,
            window="whole",
            num_workers=0,
        )

    def fit_counter_encoder(
        self, counter_columns: Sequence[str]
    ) -> PiecewiseLinearEncoder | None:
        """Fit the piecewise-linear encoder a model puts its counters through.

        Bin edges are quantiles of the *training* days, which is why this lives
        on the experiment rather than in the model: only a run that holds days
        out knows which ones it may look at. Each training day contributes an
        equal share of the sample, because counters warm up over the run and the
        first days alone would put every later value in the last bin.
        """
        if not counter_columns:
            return None

        training_days, _ = self.train_and_validation_days
        rows_per_day = max(1, _COUNTER_FIT_ROWS // len(training_days))
        counters = pl.concat(
            [
                self._sample_counters(day, counter_columns, rows_per_day)
                for day in training_days
            ]
        )
        return PiecewiseLinearEncoder.from_dataset(counters, n_bins=_COUNTER_BINS)

    def _sample_counters(
        self, day: int, counter_columns: Sequence[str], rows: int
    ) -> pl.DataFrame:
        # Sampled rather than taken off the top: nothing orders a day file, so
        # its first rows can be one contiguous slice of the user base.
        counters = pl.read_parquet(
            self.dataset_manager.day_to_path[day], columns=list(counter_columns)
        )
        return counters.sample(min(rows, counters.height), seed=self.seed)

    @property
    def user_column(self) -> str:
        return self.artifacts.user_column

    @property
    def item_id_column(self) -> str:
        return self.artifacts.item_id_column

    def setup(self) -> None:
        super().setup()
        # flash-attn is CUDA-only; a sequence model has to fall back on CPU.
        global_config.set_cpu_attention(not torch.cuda.is_available())

    def create_checkpoint_callback(
        self, prefix: str, **kwargs: Any
    ) -> CheckpointCallback:
        return CheckpointCallback(
            checkpoint_dir=str(global_config.checkpoints_path),
            run_name=self.run_name,
            prefix=prefix,
            **kwargs,
        )

    def create_validation_callback(self) -> ValidationCallback:
        return ValidationCallback(prediction_config=None, cache_batches=True)

    @property
    def trainer_runs_validation(self) -> bool:
        return True

    def extra_callbacks(self, train_days: list[int], val_days: list[int]) -> list[Any]:
        return []

    @cached_property
    def sequence_train_loader(self) -> DataLoader:
        train_days, _ = self.train_and_validation_days
        return self.make_sequence_loader(
            train_days,
            split="train",
            batch_size=self.dataloader.batch_size,
            shuffle=True,
        )

    @cached_property
    def sequence_val_loader(self) -> DataLoader:
        _, val_days = self.train_and_validation_days
        return self.make_sequence_loader(
            val_days,
            split="val",
            batch_size=self.dataloader.val_batch_size,
            shuffle=False,
            num_workers=0,
        )

    @cached_property
    def sequence_callbacks(self) -> list[Any]:
        train_days, val_days = self.train_and_validation_days
        return self.extra_callbacks(train_days, val_days)

    def prebuild_runner_components(self) -> None:
        started = perf_counter()
        super().prebuild_runner_components()
        logger.info("Prepared runner objects in %.3fs", perf_counter() - started)
        self.callbacks.validation.val_loader = self.sequence_val_loader
        started = perf_counter()
        for callback in [
            self.callbacks.validation,
            *self.sequence_callbacks,
            *self.callbacks.all,
        ]:
            callback.prepare()
        logger.info("Prepared runner callbacks in %.3fs", perf_counter() - started)
        started = perf_counter()
        self._prepared_train_iterator = EpochTrainer.prepare_train_iterator(
            self.sequence_train_loader
        )
        logger.info("Prepared first training batch in %.3fs", perf_counter() - started)

    def create_trainer(self, model: nn.Module, optimizer: Any) -> EpochTrainer:
        train_days, val_days = self.train_and_validation_days
        logger.info(
            "Sequence split: %s training day(s), %s validation day(s)",
            len(train_days),
            len(val_days),
        )

        prepared_train_iterator = self._prepared_train_iterator
        self._prepared_train_iterator = None
        return EpochTrainer(
            model=model,
            optimizer=optimizer,
            train_loader=self.sequence_train_loader,
            prepared_train_iterator=prepared_train_iterator,
            val_loader=self.sequence_val_loader,
            num_epochs=self.num_epochs,
            lr_schedule_horizon_epochs=self.lr_schedule_horizon_epochs,
            val_callback=self.callbacks.validation,
            callbacks=[
                *self.sequence_callbacks,
                *self.callbacks.all,
            ],
            gradient_accumulation_steps=(
                self.dataloader.gradient_accumulation_steps
            ),
        )

    def create_optimizers(self) -> torch.optim.Optimizer:
        embedding_params, deep_params = self.split_parameters(
            self.base_model, self.embedding_types
        )
        groups = [
            {
                "params": embedding_params,
                "lr": self.embedding_learning_rate,
                "schedule_group": "embedding",
            },
            {
                "params": deep_params,
                "lr": self.deep_learning_rate,
                "schedule_group": "deep",
            },
        ]
        optimizer = torch.optim.Adam(
            # Adam rejects an empty parameter group, and a model may have no
            # table of its own.
            [group for group in groups if group["params"]],
            weight_decay=self.weight_decay,
            # Only CUDA float parameters can take the fused kernel.
            fused=self.runner_build_device.type == "cuda",
        )
        return register_stable_optimizer_groups(optimizer)
