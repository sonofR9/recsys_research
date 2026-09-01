from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch

from dcn.config import GenerationExperiment
from dcn.config.settings import (
    DataloaderConfig,
    DayRangeConfig,
    LrScheduleConfig,
    RuntimeConfig,
    TransformerConfig,
)
from utils.global_config import config as global_config


_CONTROL_MANIFEST_SHA256 = (
    "ceccb6d6e73d082dea9502fa64f1e2af88c3460788dffdb4656e5aaf6aebd459"
)


def _load_control_manifest() -> dict[str, Any]:
    path = Path(__file__).resolve().parents[1] / "protocol/control_manifest.json"

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        pairs_seen = set()
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in pairs_seen:
                raise ValueError(f"duplicate control-manifest key {key!r}")
            pairs_seen.add(key)
            result[key] = value
        return result

    document = json.loads(path.read_text(), object_pairs_hook=reject_duplicates)
    canonical = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    if hashlib.sha256(canonical).hexdigest() != _CONTROL_MANIFEST_SHA256:
        raise ValueError("control manifest differs from its approved canonical hash")
    return document


def _ranking_snapshot_document(
    *,
    prepared: Any,
    rankings: dict[int, tuple[int, ...]],
    exclude_seen: bool,
    max_k: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "catalog_sha256": hashlib.sha256(
            json.dumps(prepared.item_id_list, separators=(",", ":")).encode()
        ).hexdigest(),
        "catalog_size": len(prepared.item_id_list),
        "exclude_seen": exclude_seen,
        "max_k": max_k,
        "rankings": [
            {"user_id": user_id, "item_ids": list(rankings[user_id])}
            for user_id in sorted(rankings)
        ],
    }


@dataclass
class G4GenerationExperiment(GenerationExperiment):
    final_ranking_evidence_group: str | None = "g4-native50m"

    def _requires_final_top_item_rankings(self) -> bool:
        return True

    def _write_final_top_item_rankings(
        self, rankings: dict[int, tuple[int, ...]]
    ) -> None:
        prepared = self.true_metric._prepared.get(None)
        if prepared is None:
            raise RuntimeError("full-population ranking was not prepared")
        max_k = min(100, len(prepared.item_id_list))
        document = _ranking_snapshot_document(
            prepared=prepared,
            rankings=rankings,
            exclude_seen=self.exclude_seen_from_evaluation,
            max_k=max_k,
        )
        destination = global_config.logs_path / self.run_name / "top_item_rankings.json"
        content = json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
        if destination.exists():
            if destination.read_bytes() != content:
                raise RuntimeError(f"top-item rankings changed: {destination}")
            return
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_bytes(content)
        temporary.replace(destination)


def _positive_rate(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return result


def _positive_batch(value: int) -> int:
    if value != 512 or isinstance(value, bool):
        raise ValueError("G4 batch_size must be 512")
    return value


def _schedule_horizon(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("lr_schedule_horizon_epochs must be a positive integer")
    return value


def build_control(
    *,
    run_name: str,
    batch_size: int,
    embedding_learning_rate: float,
    deep_learning_rate: float,
    lr_schedule_horizon_epochs: int,
    seed: int = 42,
) -> G4GenerationExperiment:
    manifest = _load_control_manifest()
    fixed = manifest["fixed"]
    data = fixed["data"]
    evaluation = fixed["evaluation"]
    loss = fixed["loss"]
    model = fixed["model"]
    training = fixed["training"]
    transformer = TransformerConfig(**model["transformer"])
    day_range = DayRangeConfig(**data["day_range"])
    dataloader = DataloaderConfig(
        batch_size=_positive_batch(batch_size),
        val_batch_size=training["val_batch_size"],
        num_workers=training["num_workers"],
        prefetch_factor=training["prefetch_factor"],
        gradient_accumulation_steps=training["gradient_accumulation_steps"],
    )
    runtime = RuntimeConfig(
        compile=training["compile"],
        dtype=torch.bfloat16,
        gradient_clip_norm=training["gradient_clip_norm"],
    )
    direct = {
        **{name: value for name, value in data.items() if name not in {"day_range"}},
        **{
            name: value
            for name, value in evaluation.items()
            if name
            not in {
                "catalog",
                "exclude_seen",
                "ks",
                "selection_max_users",
            }
        },
        **loss,
        **{name: value for name, value in model.items() if name != "transformer"},
        **{
            name: value
            for name, value in training.items()
            if name
            not in {
                "compile",
                "dtype",
                "gradient_clip_norm",
                "num_workers",
                "prefetch_factor",
                "val_batch_size",
                "gradient_accumulation_steps",
                "lr_schedule",
                "training_semantics_revision",
            }
        },
    }
    horizon = _schedule_horizon(lr_schedule_horizon_epochs)
    return G4GenerationExperiment(
        run_name=run_name,
        seed=seed,
        day_range=day_range,
        dataloader=dataloader,
        runtime=runtime,
        lr_schedule=LrScheduleConfig(**training["lr_schedule"]),
        transformer=transformer,
        evaluation_catalog=evaluation["catalog"],
        exclude_seen_from_evaluation=evaluation["exclude_seen"],
        eval_ks=tuple(evaluation["ks"]),
        eval_max_users=evaluation["selection_max_users"],
        embedding_learning_rate=_positive_rate(
            "embedding_learning_rate", embedding_learning_rate
        ),
        deep_learning_rate=_positive_rate("deep_learning_rate", deep_learning_rate),
        lr_schedule_horizon_epochs=horizon,
        num_epochs=horizon,
        **direct,
    )


def build_anchor_control(
    *, run_name: str = "g4_control_anchor_native50m", seed: int = 42
) -> G4GenerationExperiment:
    anchor = _load_control_manifest()["anchor"]
    return build_control(
        run_name=run_name,
        seed=seed,
        batch_size=anchor["batch_size"],
        embedding_learning_rate=anchor["embedding_learning_rate"],
        deep_learning_rate=anchor["deep_learning_rate"],
        lr_schedule_horizon_epochs=anchor["lr_schedule_horizon_epochs"],
    )


def control_runtime_projection(experiment: GenerationExperiment) -> dict[str, Any]:
    template = _load_control_manifest()
    fixed = template["fixed"]
    projected = {
        "data": {
            **{
                name: getattr(experiment, name)
                for name in fixed["data"]
                if name != "day_range"
            },
            "day_range": asdict(experiment.day_range),
        },
        "evaluation": {
            "catalog": experiment.evaluation_catalog,
            "eval_every_n_epochs": experiment.eval_every_n_epochs,
            "exclude_seen": experiment.exclude_seen_from_evaluation,
            "ks": list(experiment.eval_ks),
            "selection_k": experiment.selection_k,
            "selection_max_users": experiment.eval_max_users,
        },
        "loss": {name: getattr(experiment, name) for name in fixed["loss"]},
        "model": {
            **{
                name: getattr(experiment, name)
                for name in fixed["model"]
                if name != "transformer"
            },
            "transformer": {
                name: getattr(experiment.transformer, name)
                for name in fixed["model"]["transformer"]
            },
        },
        "training": {
            "adaptive_schedule_early_stopping": (
                experiment.adaptive_schedule_early_stopping
            ),
            "compile": experiment.runtime.compile,
            "dtype": str(experiment.runtime.dtype).removeprefix("torch."),
            "early_stopping_min_delta": experiment.early_stopping_min_delta,
            "early_stopping_patience": experiment.early_stopping_patience,
            "gradient_accumulation_steps": (
                experiment.dataloader.gradient_accumulation_steps
            ),
            "gradient_clip_norm": experiment.runtime.gradient_clip_norm,
            "initializer_std": experiment.initializer_std,
            "lr_schedule": {
                name: getattr(experiment.lr_schedule, name)
                for name in fixed["training"]["lr_schedule"]
            },
            "lr_schedule_horizon_epochs": experiment.lr_schedule_horizon_epochs,
            "num_epochs": experiment.num_epochs,
            "num_workers": experiment.dataloader.num_workers,
            "prefetch_factor": experiment.dataloader.prefetch_factor,
            "restore_best_weights": experiment.restore_best_weights,
            "training_semantics_revision": fixed["training"][
                "training_semantics_revision"
            ],
            "val_batch_size": experiment.dataloader.val_batch_size,
            "weight_decay": experiment.weight_decay,
        },
    }
    for name in ("lr_schedule_horizon_epochs", "num_epochs"):
        if name not in fixed["training"]:
            projected["training"].pop(name)
    return {
        "fixed": projected,
        "selected": {
            "batch_size": experiment.dataloader.batch_size,
            "embedding_learning_rate": experiment.embedding_learning_rate,
            "deep_learning_rate": experiment.deep_learning_rate,
            "lr_schedule_horizon_epochs": experiment.lr_schedule_horizon_epochs,
        },
    }
