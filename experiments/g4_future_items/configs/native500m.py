from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cached_property
from pathlib import Path
from typing import Any, Literal

import polars as pl
import torch

from dcn.config import GenerationExperiment, MuTransferGenerationExperiment
from dcn.config.settings import DataloaderConfig, LrScheduleConfig, RuntimeConfig
from experiments.generation_protocol import generation_protocol
from experiments.g4_future_items.native500m_targets import (
    Native500MFuturePositiveTwoTowerLoss,
    Native500MFutureWindowTargets,
    Native500MPeriodArtifactTargets,
)
from experiments.g4_future_items.targets import (
    OCCURRENCE_POSITION_COLUMN,
    FutureEventIndex,
    ObjectiveId,
)


EMBEDDING_LEARNING_RATE = 0.0468526465053628
DEEP_LEARNING_RATE_ANCHOR = 0.032703745675187676
TRAINING_HORIZON_EPOCHS = 15
BATCH_SIZE = 512

MaskMode = Literal[
    "next_item_unique",
    "next_24h_unique",
    "next_10_unique",
    "selected_period_union_unique",
    "all_positive_probability_periods_unique",
]

_PERIOD_ARTIFACT_ROOT = (
    Path(__file__).resolve().parents[3] / "generated/g4_native500m/selector_artifacts"
)


@dataclass
class G4Native500MExperiment(MuTransferGenerationExperiment):
    final_ranking_evidence_group: str | None = "g4-native500m"
    objective_id: ObjectiveId | Literal["control_next_item"] = "control_next_item"
    objective_window_seconds: int | None = None
    objective_event_lookahead: int | None = None
    valid_positive_mask_mode: MaskMode = "next_item_unique"
    selector_artifact_sha256: str | None = None
    objective_period_count: int | None = None

    def __post_init__(self) -> None:
        cls_token_mode = self.cls_token_mode
        if self.objective_id != "control_next_item":
            self.cls_token_mode = "none"
        try:
            super().__post_init__()
        finally:
            self.cls_token_mode = cls_token_mode

    @property
    def sequence_columns(self) -> list[str]:
        if self.objective_id == "control_next_item":
            return super().sequence_columns
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

    def create_targets(self) -> Any:
        if self.objective_id == "control_next_item":
            return super().create_targets()
        if self.objective_id.startswith("rq3_"):
            if self.objective_period_count != 1:
                raise ValueError("native-500M RQ3 period_count must be 1")
            artifact = self.period_artifact
            expected_selector_kind = (
                "deterministic"
                if self.objective_id == "rq3_deterministic_hard"
                else "learned"
            )
            if artifact.manifest.get("selector_kind") != expected_selector_kind:
                raise ValueError("RQ3 objective and selector artifact kind differ")
            return Native500MPeriodArtifactTargets(
                objective_id=self.objective_id,
                training_seed=self.seed,
                artifact=artifact,
                period_count=1,
                training_cutoff_timestamp=self.validation_cutoff_timestamp,
            )
        return Native500MFutureWindowTargets(
            objective_id=self.objective_id,
            training_seed=self.seed,
            event_index=self.future_event_index,
            window_seconds=self.objective_window_seconds,
            event_lookahead=self.objective_event_lookahead,
            training_cutoff_timestamp=self.validation_cutoff_timestamp,
        )

    def create_criterion(self) -> Any:
        base = super().create_criterion()
        if self.objective_id == "control_next_item":
            return base
        return Native500MFuturePositiveTwoTowerLoss(
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
            "g4_dataset_lineage": "native500m-v1",
        }


def build_native500m_control(
    *, run_name: str, deep_learning_rate: float, seed: int = 42
) -> G4Native500MExperiment:
    if (
        not isinstance(deep_learning_rate, (int, float))
        or isinstance(deep_learning_rate, bool)
        or deep_learning_rate <= 0
    ):
        raise ValueError("deep_learning_rate must be positive")
    transformer = replace(
        GenerationExperiment.transformer,
        dim=64,
        num_layers=2,
        nhead=2,
        num_kv_heads=1,
        ffn="swiglu",
        ffn_intermediate_dim=192,
        gated_ffn_dropout=True,
        dropout=0.1,
        input_dropout=0.1,
        ffn_dropout=0.1,
        norm="layer",
        norm_place="post",
        input_norm="rms",
        final_norm="rms",
        alibi=True,
        rope="timestamp_reverse",
        learned_positions=("forward", "reverse"),
        learned_position_fusion="concat",
        learned_position_fusion_residual="rezero",
        learned_position_reverse_correction="bounded_tanh",
        learned_position_reverse_max_scale=0.025,
        learned_position_reverse_initializer_rng_nonadvancing=True,
        attention_window=None,
    )
    return G4Native500MExperiment(
        run_name=run_name,
        seed=seed,
        **generation_protocol(
            event_type_filter="like",
            window="next_item",
            size="500m",
            user_sample=None,
        ),
        dataloader=DataloaderConfig(
            batch_size=BATCH_SIZE,
            val_batch_size=8192,
            num_workers=4,
            prefetch_factor=4,
            gradient_accumulation_steps=1,
        ),
        runtime=RuntimeConfig(
            dtype=torch.bfloat16,
            compile=False,
            gradient_clip_norm=None,
        ),
        num_epochs=TRAINING_HORIZON_EPOCHS,
        lr_schedule_horizon_epochs=TRAINING_HORIZON_EPOCHS,
        lr_schedule=LrScheduleConfig(
            "cosine",
            warmup_fraction=0.05,
            cycles=1,
            optimizer_group_scope="deep_only",
        ),
        eval_every_n_epochs=1,
        restore_best_weights=True,
        adaptive_schedule_early_stopping=False,
        early_stopping_patience=3,
        early_stopping_min_delta=0.0,
        transformer=transformer,
        max_seq_len=100,
        item_embedding_dim=64,
        bos=True,
        cls_token=False,
        cls_token_mode="end_only",
        timestamp_delta="bins",
        timestamp_combination="add",
        timestamp_num_bins=32,
        negative_sampling="random_offline_logq",
        logq_correction="yi2019",
        correct_positive_logq=True,
        mask_false_negatives=False,
        exclude_own_group_negatives=False,
        num_in_batch_negatives=2048,
        dense_random_negative_scores=True,
        random_negative_fraction=0.5,
        initializer_std=0.02,
        embedding_learning_rate=EMBEDDING_LEARNING_RATE,
        deep_learning_rate=float(deep_learning_rate),
        weight_decay=0.0,
        mup_base_dim=16,
        mup_delta_dim=32,
    )


def build_native500m_treatment(
    *,
    run_name: str,
    deep_learning_rate: float,
    objective: dict[str, Any],
    valid_positive_mask_mode: str,
    seed: int = 42,
) -> G4Native500MExperiment:
    expected: dict[str, tuple[set[str], str]] = {
        "rq1_24h": ({"id", "window_seconds"}, "next_24h_unique"),
        "rq2_next10": ({"id", "event_lookahead"}, "next_10_unique"),
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
    objective_id = objective.get("id")
    if objective_id not in expected:
        raise ValueError(f"objective {objective_id!r} is not approved")
    expected_fields, expected_mask = expected[objective_id]
    if set(objective) != expected_fields or valid_positive_mask_mode != expected_mask:
        raise ValueError("objective fields or valid-positive mask differ from protocol")
    if objective_id.startswith("rq3_") and objective["period_count"] != 1:
        raise ValueError("native-500M RQ3 period_count must be 1")
    control = build_native500m_control(
        run_name=run_name,
        deep_learning_rate=deep_learning_rate,
        seed=seed,
    )
    return replace(
        control,
        objective_id=objective_id,
        objective_window_seconds=objective.get("window_seconds"),
        objective_event_lookahead=objective.get("event_lookahead"),
        selector_artifact_sha256=objective.get("selector_artifact_sha256"),
        objective_period_count=objective.get("period_count"),
        valid_positive_mask_mode=valid_positive_mask_mode,
    )
