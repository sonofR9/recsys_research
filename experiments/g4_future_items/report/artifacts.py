from __future__ import annotations

import json
import math
from pathlib import Path
import re
from typing import Any

from .selection import RecommenderTrial


_EPOCH = re.compile(r"\bepoch (\d+) finished\b")
_METRIC = re.compile(r"(?:^|\s)([^\s=]+)=([^\s]+)")


def _document(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _epoch_metrics(path: Path) -> dict[int, dict[str, float]]:
    result: dict[int, dict[str, float]] = {}
    for line in path.read_text(errors="replace").splitlines():
        epoch_match = _EPOCH.search(line)
        if epoch_match is None:
            continue
        metrics = {}
        for name, raw in _METRIC.findall(line[epoch_match.end() :]):
            try:
                metrics[name] = float(raw)
            except ValueError:
                continue
        result[int(epoch_match.group(1))] = metrics
    return result


def read_recommender_trial(run_directory: Path) -> RecommenderTrial:
    contract = _document(run_directory / "g4_job.json")
    metadata = _document(run_directory / "training_metadata.json")
    job = contract.get("job")
    if not isinstance(job, dict):
        raise ValueError("G4 job contract has no resolved job")
    dataloader = job.get("dataloader")
    if not isinstance(dataloader, dict) or dataloader.get("batch_size") != 512:
        raise ValueError("G4 job contract batch must be 512")
    contract_horizon = job.get("lr_schedule_horizon_epochs")
    if type(contract_horizon) is not int or contract_horizon < 1:
        raise ValueError("G4 job contract has no valid schedule horizon")
    if any(
        metadata.get(name) != contract_horizon
        for name in (
            "lr_schedule_horizon_epochs",
            "num_epochs",
            "max_epochs",
        )
    ):
        raise ValueError("G4 contract and metadata horizon differ")
    expected_metadata = {
        "batch_size": 512,
        "embedding_learning_rate": job.get("embedding_learning_rate"),
        "deep_learning_rate": job.get("deep_learning_rate"),
    }
    if any(metadata.get(name) != value for name, value in expected_metadata.items()):
        raise ValueError("G4 contract and metadata runtime parameters differ")
    parameters = {
        "batch_size": dataloader["batch_size"],
        "embedding_learning_rate": job["embedding_learning_rate"],
        "deep_learning_rate": job["deep_learning_rate"],
    }
    objective = job.get("objective")
    if isinstance(objective, dict) and "period_count" in objective:
        parameters["period_count"] = objective["period_count"]

    horizon = contract_horizon
    epochs = metadata.get("epochs_trained")
    resolved = (
        type(horizon) is int
        and type(epochs) is int
        and metadata.get("lr_horizon_complete") is True
        and metadata.get("selection_resolved") is True
        and epochs == horizon
    )
    recall = math.nan
    loss = math.nan
    if resolved:
        best_epoch = metadata.get("best_epoch")
        if type(best_epoch) is not int or not 1 <= best_epoch <= horizon:
            raise ValueError("resolved G4 run has no valid best epoch")
        metrics = _epoch_metrics(run_directory / "sweep.log").get(best_epoch - 1)
        if metrics is None:
            raise ValueError("best G4 epoch is absent from the training log")
        recall = metrics.get("epoch/val_true.recall@100", math.nan)
        loss = metrics.get("epoch/val.loss", math.nan)

    return RecommenderTrial(
        row_id=str(contract["row_id"]),
        run_name=str(job["run_name"]),
        parameters=parameters,
        validation_recall_at_100=float(recall),
        validation_loss=float(loss),
        epochs_trained=int(epochs) if type(epochs) is int else -1,
        horizon_epochs=int(horizon) if type(horizon) is int else -1,
    )
