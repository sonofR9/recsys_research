from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import os
from pathlib import Path
import runpy
from typing import Iterator

from dcn.config import MuTransferGenerationExperiment
from experiments.g1_aggregate_dataset_size.protocol.fixed26_calibration import (
    Fixed26CalibrationJob,
    Fixed26CalibrationManifest,
)


_SOURCE_CONFIG = Path(__file__).with_name("aggregate_variant.py")


def build_fixed26_experiment(
    job: Fixed26CalibrationJob,
    manifest: Fixed26CalibrationManifest,
) -> MuTransferGenerationExperiment:
    if job not in manifest.jobs:
        raise ValueError("job is outside the fixed-26 calibration manifest")
    source = _load_source_control(manifest.source_run_name)
    if not isinstance(source, MuTransferGenerationExperiment):
        raise TypeError("fixed-26 source control is not a MuTransfer experiment")
    if (
        source.size != manifest.dataset_size
        or source.dataloader.batch_size != manifest.batch_size
        or source.embedding_learning_rate != manifest.embedding_learning_rate
        or source.deep_learning_rate != manifest.deep_learning_rate
    ):
        raise ValueError("fixed-26 manifest differs from its selected source control")
    return replace(
        source,
        run_name=job.run_name,
        seed=job.seed,
        num_epochs=manifest.num_epochs,
        lr_schedule_horizon_epochs=None,
        eval_every_n_epochs=1,
        early_stopping_patience=None,
        adaptive_schedule_early_stopping=False,
        restore_best_weights=manifest.restore_best_weights,
    )


def _load_source_control(run_name: str) -> object:
    with _g1_environment(
        {
            "G1_AGGREGATE_RUN": run_name,
            "G1_DATASET_SIZE": "50m",
        }
    ):
        return runpy.run_path(str(_SOURCE_CONFIG))["experiment"]


@contextmanager
def _g1_environment(updates: dict[str, str]) -> Iterator[None]:
    previous = {
        name: value for name, value in os.environ.items() if name.startswith("G1_")
    }
    try:
        for name in tuple(os.environ):
            if name.startswith("G1_"):
                os.environ.pop(name)
        os.environ.update(updates)
        yield
    finally:
        for name in tuple(os.environ):
            if name.startswith("G1_"):
                os.environ.pop(name)
        os.environ.update(previous)
