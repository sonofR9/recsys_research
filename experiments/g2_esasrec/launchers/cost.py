from __future__ import annotations

import json
import math
from pathlib import Path
import statistics
from time import perf_counter
from types import MethodType
from typing import Any

from neuralrec.run.callbacks.base import Callback
from neuralrec.utils import EXTRA_METRICS, to_float


class CostEvidenceCallback(Callback):
    def __init__(self, experiment: Any) -> None:
        self.experiment = experiment
        self.epoch_seconds: list[float] = []
        self.resources: list[dict[str, float]] = []
        self.wall_started: float | None = None

    def on_train_begin(self, state: dict[str, Any]) -> None:
        if self.wall_started is not None:
            raise RuntimeError("G2 cost timer started twice")
        self.wall_started = perf_counter()

    def on_epoch_end(self, state: dict[str, Any]) -> None:
        metrics = state.get(EXTRA_METRICS, {})
        timing = to_float(metrics.get("timing", {}).get("train_epoch_time"))
        resources = {
            name: value
            for name in (
                "params_total",
                "params_trainable",
                "params_embedding",
                "params_deep",
                "peak_memory_gb",
            )
            if (value := to_float(metrics.get("resources", {}).get(name))) is not None
        }
        if timing is None or timing <= 0:
            raise RuntimeError("G2 cost evidence requires a positive epoch time")
        required = {
            "params_total",
            "params_trainable",
            "params_embedding",
            "params_deep",
        }
        if not required <= resources.keys():
            raise RuntimeError("G2 cost evidence requires resource metrics")
        if self.experiment.device.type == "cuda" and "peak_memory_gb" not in resources:
            raise RuntimeError("G2 CUDA cost evidence requires peak memory")
        self.epoch_seconds.append(timing)
        self.resources.append(resources)

    def on_train_end(self, state: dict[str, Any]) -> None:
        if not self.epoch_seconds or not self.resources:
            raise RuntimeError("G2 cost evidence contains no completed epoch")
        if self.wall_started is None:
            raise RuntimeError("G2 cost timer did not start")
        latest = self.resources[-1]
        peak_memory = max(row.get("peak_memory_gb", 0.0) for row in self.resources)
        median_epoch = statistics.median(self.epoch_seconds)
        targets_per_epoch = float(self.experiment.training_targets_per_epoch)
        best_epoch = self.experiment.callbacks.best_weights.best_epoch
        if best_epoch is None:
            raise RuntimeError("G2 cost evidence requires a selected best epoch")
        document = {
            **{name: latest[name] for name in latest if name != "peak_memory_gb"},
            "peak_memory_gb": peak_memory,
            "training_seconds": math.fsum(self.epoch_seconds),
            "wall_seconds": perf_counter() - self.wall_started,
            "median_train_epoch_seconds": median_epoch,
            "targets_per_second": targets_per_epoch / median_epoch,
            "best_epoch": best_epoch + 1,
            "epochs_timed": len(self.epoch_seconds),
        }
        destination = (
            Path(self.experiment.base_path)
            / "logs"
            / self.experiment.run_name
            / "cost_metrics.json"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")


def attach_cost_evidence(experiment: Any) -> None:
    recorder = CostEvidenceCallback(experiment)
    original = experiment.create_callbacks

    def create_callbacks(owner: Any):
        callbacks = original()
        callbacks.all.append(recorder)
        return callbacks

    experiment.create_callbacks = MethodType(create_callbacks, experiment)
