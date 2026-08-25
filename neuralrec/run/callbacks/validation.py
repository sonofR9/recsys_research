from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path
from time import perf_counter
from typing import Any

import polars as pl
import torch

from neuralrec.data.dataloader import DataLoader
from neuralrec.run.callbacks.base import Callback
from neuralrec.run.train import TrainRunner
from neuralrec.utils import DeferredScalars, LOSS_DENOMINATOR, add_metrics


@dataclass
class PredictionConfig:
    """Which columns of a validation batch and its output land in the dump."""

    predictions_dir: str | Path = "predictions"
    int_columns: dict[str, str] = field(default_factory=dict)
    float_columns: dict[str, str] = field(default_factory=dict)


class ValidationCallback(Callback):
    def __init__(
        self,
        val_loader: DataLoader[object] | None = None,
        prediction_config: PredictionConfig | None = None,
        cache_batches: bool = False,
    ) -> None:
        self.val_loader = val_loader
        self.prediction_config = prediction_config
        self.cache_batches = cache_batches
        self._prepared_iterator: Iterator | None = None
        self._prepared_batches: list | None = None

    def prepare(self) -> None:
        if (
            self.val_loader is None
            or self._prepared_iterator is not None
            or self._prepared_batches is not None
        ):
            return
        if self.cache_batches:
            self._prepared_batches = list(self.val_loader)
            return
        iterator = iter(self.val_loader)
        try:
            first_batch = next(iterator)
        except StopIteration:
            self._prepared_iterator = iter(())
        else:
            self._prepared_iterator = chain((first_batch,), iterator)

    def on_epoch_end(
        self,
        state: dict[str, Any],
    ) -> None:
        if self.val_loader is None:
            return

        runner: TrainRunner = state["train_runner"]

        runner.model.eval()

        deferred_metrics = DeferredScalars()
        n_batches = 0

        predictions_data: dict[str, list[torch.Tensor]] = defaultdict(list)

        start_time = perf_counter()
        first_batch_time = None
        first_batch_load_time = None
        first_batch_start = None
        first_batch_end = None
        if next(runner.model.parameters()).device.type == "cuda":
            first_batch_start = torch.cuda.Event(enable_timing=True)
            first_batch_end = torch.cuda.Event(enable_timing=True)
        batches = (
            self._prepared_batches
            if self._prepared_batches is not None
            else (
                self._prepared_iterator
                if self._prepared_iterator is not None
                else self.val_loader
            )
        )
        self._prepared_iterator = None

        with torch.inference_mode():
            for batch in batches:
                is_first_batch = n_batches == 0
                if is_first_batch:
                    first_batch_load_time = perf_counter() - start_time
                    if first_batch_start is not None:
                        first_batch_start.record()
                batch_out = runner._model_forward(batch)
                deferred_metrics.add(
                    n_batches,
                    {
                        name: value
                        for name, value in batch_out.items()
                        if name != LOSS_DENOMINATOR
                    },
                )
                n_batches += 1

                if is_first_batch:
                    if first_batch_end is None:
                        first_batch_time = perf_counter() - start_time
                    else:
                        first_batch_end.record()

                if self.prediction_config:
                    self._collect_predictions(batch, batch_out, predictions_data)

        metrics = DeferredScalars.means(deferred_metrics.drain())
        inference_time = perf_counter() - start_time
        if first_batch_start is not None and first_batch_load_time is not None:
            assert first_batch_end is not None
            first_batch_time = first_batch_load_time + (
                first_batch_start.elapsed_time(first_batch_end) / 1000
            )

        add_metrics(state, "epoch/val", metrics)

        save_time = 0.0
        if self.prediction_config and predictions_data:
            save_start = perf_counter()
            self._save_predictions(predictions_data, state)
            save_time = perf_counter() - save_start

        add_metrics(
            state,
            "timing",
            {
                "val_inference_time": inference_time,
                "val_first_batch_time": first_batch_time or 0.0,
                "val_save_time": save_time,
            },
        )

    def _collect_predictions(
        self,
        batch: dict[str, Any],
        batch_out: dict[str, Any],
        predictions_data: dict[str, list[torch.Tensor]],
    ) -> None:
        assert self.prediction_config is not None
        cfg = self.prediction_config

        combined_columns = chain(
            ((col_name, key, batch) for col_name, key in cfg.int_columns.items()),
            (
                (
                    col_name,
                    key,
                    batch_out if key in batch_out or "." not in key else batch,
                )
                for col_name, key in cfg.float_columns.items()
            ),
        )

        for col_name, key, source in combined_columns:
            try:
                value = self._extract_value(source, key)
                predictions_data[col_name].append(value.cpu())
            except (KeyError, TypeError):
                pass

    def _extract_value(self, source: dict[str, Any], key: str) -> torch.Tensor:
        parts = key.split(".")
        value = source
        for part in parts:
            value = value[part]

        if not isinstance(value, torch.Tensor) and hasattr(value, "dense"):
            value = value.dense()

        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Expected tensor, got {type(value)}")

        if value.dtype in (torch.bfloat16, torch.float16):
            value = value.float()
        return value

    def _save_predictions(
        self,
        predictions_data: dict[str, list[torch.Tensor]],
        state: dict[str, Any],
    ) -> None:
        cfg = self.prediction_config
        assert cfg is not None, "Prediction config is not set"
        df_data = {}

        int_cols = set(cfg.int_columns.keys())

        for col_name, tensors in predictions_data.items():
            if not tensors:
                continue

            concatenated = torch.cat(tensors)
            if concatenated.ndim > 1:
                concatenated = concatenated.squeeze()

            if col_name in int_cols:
                df_data[col_name] = concatenated.to(torch.int64).numpy()
            else:
                df_data[col_name] = concatenated.float().numpy()

        if not df_data:
            return

        df = pl.DataFrame(df_data)

        runner: TrainRunner = state["train_runner"]
        epoch = runner.current_epoch

        filename = f"predictions_epoch_{epoch}.parquet"

        predictions_dir = Path(cfg.predictions_dir)
        predictions_dir.mkdir(parents=True, exist_ok=True)
        pred_path = predictions_dir / filename
        df.write_parquet(pred_path)
