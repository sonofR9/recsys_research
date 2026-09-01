from __future__ import annotations

from dataclasses import replace
import math

import torch

from dcn.config import (
    GenerationExperiment,
    MuTransferGenerationExperiment,
    SemanticHistoryExperiment,
    SemanticIdConfig,
)
from dcn.config.settings import DataloaderConfig, LrScheduleConfig, RuntimeConfig
from experiments.g6_rqkmeans_history.protocol.manifest import (
    CONTROL_BATCHES,
    NUM_CODES,
    NUM_LEVELS,
    RANKING_EVIDENCE_GROUP,
    REPRESENTATIONS,
    REPRESENTATION_WIDTHS,
    Backbone,
    Representation,
)
from experiments.generation_protocol import generation_protocol


_DATA_PROTOCOL = generation_protocol(
    event_type_filter="like",
    window="next_item",
    size="50m",
)
_ORIGINAL_TRANSFORMER = replace(
    GenerationExperiment.transformer,
    dim=64,
    num_layers=2,
    nhead=2,
    num_kv_heads=2,
    ffn_intermediate_dim=256,
    ffn="gelu",
    norm="layer",
    norm_place="pre",
    input_norm=None,
    final_norm="layer",
    alibi=False,
    rope=None,
    learned_positions="forward",
    learned_position_fusion="add",
    learned_position_fusion_normalization=None,
    learned_position_fusion_residual=None,
    learned_position_initialization="default",
    learned_position_reverse_correction=None,
    learned_position_reverse_max_scale=0.1,
    learned_position_reverse_initializer_rng_nonadvancing=False,
    attention_window=None,
    input_dropout=0.1,
    ffn_dropout=0.1,
    gated_ffn_dropout=False,
)
_BEST_TRANSFORMER = replace(
    _ORIGINAL_TRANSFORMER,
    num_layers=4,
    num_kv_heads=1,
    ffn_intermediate_dim=192,
    ffn="swiglu",
    gated_ffn_dropout=True,
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
)


def _positive_rate(name: str, value: float) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive finite")
    return value


def _common(
    backbone: Backbone,
    *,
    batch_size: int,
    validation_batch_size: int,
    embedding_learning_rate: float,
    deep_learning_rate: float,
    run_name: str,
) -> dict[str, object]:
    if backbone not in {"original_g1", "best_g1"}:
        raise ValueError(f"unknown backbone {backbone!r}")
    if batch_size not in CONTROL_BATCHES:
        raise ValueError(f"batch_size must be one of {CONTROL_BATCHES}")
    if (
        not isinstance(validation_batch_size, int)
        or isinstance(validation_batch_size, bool)
        or validation_batch_size < 1
    ):
        raise ValueError("validation_batch_size must be a positive integer")
    best = backbone == "best_g1"
    return {
        "run_name": run_name,
        **_DATA_PROTOCOL,
        "dataloader": DataloaderConfig(
            batch_size=batch_size,
            val_batch_size=validation_batch_size,
            num_workers=4,
            prefetch_factor=4,
            gradient_accumulation_steps=1,
        ),
        "num_epochs": 15 if best else 40,
        "lr_schedule_horizon_epochs": 15 if best else 20,
        "eval_every_n_epochs": 1,
        "restore_best_weights": True,
        "early_stopping_patience": 3,
        "early_stopping_min_delta": 0.0,
        "adaptive_schedule_early_stopping": False,
        "transformer": _BEST_TRANSFORMER if best else _ORIGINAL_TRANSFORMER,
        "max_seq_len": 100,
        "bos": best,
        "cls_token": False,
        "cls_token_mode": "end_only" if best else "none",
        "lr_schedule": (
            LrScheduleConfig(
                "cosine",
                warmup_fraction=0.05,
                cycles=1,
                optimizer_group_scope="deep_only",
            )
            if best
            else LrScheduleConfig()
        ),
        "timestamp_delta": "bins" if best else None,
        "timestamp_combination": "add",
        "timestamp_num_bins": 32,
        "negative_sampling": "random_offline_logq" if best else "offline_logq",
        "num_in_batch_negatives": 2048 if best else 512,
        "logq_correction": "yi2019" if best else "baseline",
        "correct_positive_logq": best,
        "mask_false_negatives": False,
        "exclude_own_group_negatives": False,
        "dense_random_negative_scores": best,
        "initializer_std": 0.02,
        "item_embedding_dim": 64,
        "embedding_learning_rate": _positive_rate(
            "embedding_learning_rate", embedding_learning_rate
        ),
        "deep_learning_rate": _positive_rate("deep_learning_rate", deep_learning_rate),
        "weight_decay": 0.0,
        "runtime": RuntimeConfig(
            dtype=torch.bfloat16,
            compile=False,
            gradient_clip_norm=None,
        ),
        "mup_base_dim": 16,
        "mup_delta_dim": 32,
        "final_ranking_evidence_group": RANKING_EVIDENCE_GROUP,
    }


def build_control(
    backbone: Backbone,
    *,
    batch_size: int,
    validation_batch_size: int,
    embedding_learning_rate: float,
    deep_learning_rate: float,
    run_name: str,
) -> MuTransferGenerationExperiment:
    return MuTransferGenerationExperiment(
        **_common(
            backbone,
            batch_size=batch_size,
            validation_batch_size=validation_batch_size,
            embedding_learning_rate=embedding_learning_rate,
            deep_learning_rate=deep_learning_rate,
            run_name=run_name,
        )
    )


def build_semantic_treatment(
    representation: Representation,
    *,
    backbone: Backbone,
    batch_size: int,
    validation_batch_size: int,
    embedding_learning_rate: float,
    deep_learning_rate: float,
    num_levels: int,
    num_codes: int,
    representation_width: int,
    run_name: str,
) -> SemanticHistoryExperiment:
    if representation not in REPRESENTATIONS:
        raise ValueError(f"unknown representation {representation!r}")
    if num_levels not in NUM_LEVELS:
        raise ValueError(f"num_levels must be one of {NUM_LEVELS}")
    if num_codes not in NUM_CODES:
        raise ValueError(f"num_codes must be one of {NUM_CODES}")
    if representation_width not in REPRESENTATION_WIDTHS:
        raise ValueError(f"representation_width must be one of {REPRESENTATION_WIDTHS}")
    return SemanticHistoryExperiment(
        **_common(
            backbone,
            batch_size=batch_size,
            validation_batch_size=validation_batch_size,
            embedding_learning_rate=embedding_learning_rate,
            deep_learning_rate=deep_learning_rate,
            run_name=run_name,
        ),
        history_representation=representation,
        representation_width=representation_width,
        semantic=SemanticIdConfig(
            quantizer="kmeans",
            num_levels=num_levels,
            num_codes=num_codes,
            kmeans_iterations=20,
            seed=42,
        ),
    )


def build_learned_sid_residual_remediation(
    *,
    backbone: Backbone,
    batch_size: int,
    validation_batch_size: int,
    embedding_learning_rate: float,
    deep_learning_rate: float,
    num_levels: int,
    num_codes: int,
    representation_width: int,
    frozen_event_width: int,
    run_name: str,
    learned_residual_max_scale: float | None = None,
) -> SemanticHistoryExperiment:
    if num_levels != 3:
        raise ValueError("remediation fixes num_levels=3")
    if num_codes != 512:
        raise ValueError("remediation fixes num_codes=512")
    if representation_width not in REPRESENTATION_WIDTHS:
        raise ValueError(f"representation_width must be one of {REPRESENTATION_WIDTHS}")
    if frozen_event_width != 128:
        raise ValueError("remediation fixes frozen_event_width=128")
    return SemanticHistoryExperiment(
        **_common(
            backbone,
            batch_size=batch_size,
            validation_batch_size=validation_batch_size,
            embedding_learning_rate=embedding_learning_rate,
            deep_learning_rate=deep_learning_rate,
            run_name=run_name,
        ),
        history_representation="item_frozen_sid_learned_residual_event",
        representation_width=representation_width,
        frozen_event_width=frozen_event_width,
        learned_residual_max_scale=learned_residual_max_scale,
        semantic=SemanticIdConfig(
            quantizer="kmeans",
            num_levels=num_levels,
            num_codes=num_codes,
            kmeans_iterations=20,
            seed=42,
        ),
    )
