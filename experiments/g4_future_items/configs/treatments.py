from __future__ import annotations

from dataclasses import dataclass, fields
from functools import cached_property
from pathlib import Path
from typing import Any, Literal

import polars as pl

from experiments.g4_future_items.configs.control import (
    G4GenerationExperiment,
    build_control,
)
from experiments.g4_future_items.targets import (
    OCCURRENCE_POSITION_COLUMN,
    FutureEventIndex,
    FuturePositiveTwoTowerLoss,
    FutureWindowTargets,
    ObjectiveId,
    PeriodArtifactTargets,
)

MaskMode = Literal[
    "next_24h_unique",
    "next_10_unique",
    "selected_period_union_unique",
    "all_positive_probability_periods_unique",
]
_PERIOD_ARTIFACT_ROOT = (
    Path(__file__).resolve().parents[3] / "generated" / "g4_selector_artifacts"
)


@dataclass
class G4FutureExperiment(G4GenerationExperiment):
    objective_id: ObjectiveId = "rq1_24h"
    objective_window_seconds: int | None = 86400
    objective_event_lookahead: int | None = None
    valid_positive_mask_mode: MaskMode = "next_24h_unique"
    selector_artifact_sha256: str | None = None
    objective_period_count: int | None = None

    @property
    def sequence_columns(self) -> list[str]:
        return [*super().sequence_columns, OCCURRENCE_POSITION_COLUMN]

    @cached_property
    def future_event_index(self) -> FutureEventIndex:
        train_days, _ = self.train_and_validation_days
        frame = (
            pl.scan_parquet(
                [self.dataset_manager.day_to_path[day] for day in train_days]
            )
            .filter(self.row_filter_for_split("train"))
            .select(
                self.user_column,
                self.artifacts.timestamp_column,
                self.item_id_column,
            )
            .collect(engine="streaming")
        )
        return FutureEventIndex.from_columns(
            user_ids=frame[self.user_column].to_list(),
            timestamps=frame[self.artifacts.timestamp_column].to_list(),
            item_ids=frame[self.item_id_column].to_list(),
        )

    @cached_property
    def period_artifact(self) -> Any:
        if self.selector_artifact_sha256 is None:
            raise ValueError("RQ3 requires selector_artifact_sha256")
        from experiments.g4_future_items.protocol.materialization import PeriodArtifact

        return PeriodArtifact.open(
            _PERIOD_ARTIFACT_ROOT,
            expected_sha256=self.selector_artifact_sha256,
        )

    def create_targets(self) -> FutureWindowTargets | PeriodArtifactTargets:
        if self.objective_id.startswith("rq3_"):
            if self.objective_period_count is None:
                raise ValueError("RQ3 requires objective period_count")
            artifact = self.period_artifact
            expected_selector_kind = (
                "deterministic"
                if self.objective_id == "rq3_deterministic_hard"
                else "learned"
            )
            if artifact.manifest.get("selector_kind") != expected_selector_kind:
                raise ValueError("RQ3 objective and selector artifact kind differ")
            return PeriodArtifactTargets(
                objective_id=self.objective_id,
                training_seed=self.seed,
                artifact=artifact,
                period_count=self.objective_period_count,
                training_cutoff_timestamp=self.validation_cutoff_timestamp,
            )
        return FutureWindowTargets(
            objective_id=self.objective_id,
            training_seed=self.seed,
            event_index=self.future_event_index,
            window_seconds=self.objective_window_seconds,
            event_lookahead=self.objective_event_lookahead,
            training_cutoff_timestamp=self.validation_cutoff_timestamp,
        )

    def create_criterion(self) -> FuturePositiveTwoTowerLoss:
        base = super().create_criterion()
        return FuturePositiveTwoTowerLoss(
            base.model,
            base.loss,
            targets=base.targets,
            user_id_column=self.user_column,
        )

    def generation_architecture_metadata(self) -> dict[str, object]:
        return {
            **super().generation_architecture_metadata(),
            "g4_objective_id": self.objective_id,
            "g4_objective_window_seconds": self.objective_window_seconds,
            "g4_objective_event_lookahead": self.objective_event_lookahead,
            "g4_selector_artifact_sha256": self.selector_artifact_sha256,
            "g4_objective_period_count": self.objective_period_count,
            "g4_valid_positive_mask_mode": self.valid_positive_mask_mode,
            "g4_target_seed_revision": "g4-target-v1",
        }


def build_treatment(
    *,
    objective: dict[str, Any],
    valid_positive_mask_mode: str,
    run_name: str,
    batch_size: int,
    embedding_learning_rate: float,
    deep_learning_rate: float,
    lr_schedule_horizon_epochs: int,
    seed: int,
) -> G4FutureExperiment:
    objective_id = objective.get("id")
    expected = {
        "rq1_24h": (
            {"id", "window_seconds"},
            "next_24h_unique",
        ),
        "rq2_next10": (
            {"id", "event_lookahead"},
            "next_10_unique",
        ),
        "rq3_deterministic_hard": (
            {"id", "selector_artifact_sha256", "period_count"},
            "selected_period_union_unique",
        ),
        "rq3_learned_hard": (
            {"id", "selector_artifact_sha256", "period_count"},
            "selected_period_union_unique",
        ),
        "rq3_learned_proportional": (
            {"id", "selector_artifact_sha256", "period_count"},
            "all_positive_probability_periods_unique",
        ),
    }
    if objective_id not in expected:
        raise ValueError(f"objective {objective_id!r} has no implemented G4 target")
    expected_keys, expected_mask = expected[objective_id]
    if set(objective) != expected_keys or valid_positive_mask_mode != expected_mask:
        raise ValueError("objective fields or valid-positive mask differ from protocol")
    control = build_control(
        run_name=run_name,
        batch_size=batch_size,
        embedding_learning_rate=embedding_learning_rate,
        deep_learning_rate=deep_learning_rate,
        lr_schedule_horizon_epochs=lr_schedule_horizon_epochs,
        seed=seed,
    )
    values = {
        field.name: getattr(control, field.name)
        for field in fields(control)
        if field.init
    }
    values.update(
        {
            "objective_id": objective_id,
            "objective_window_seconds": objective.get("window_seconds"),
            "objective_event_lookahead": objective.get("event_lookahead"),
            "selector_artifact_sha256": objective.get("selector_artifact_sha256"),
            "objective_period_count": objective.get("period_count"),
            "valid_positive_mask_mode": valid_positive_mask_mode,
        }
    )
    return G4FutureExperiment(**values)
