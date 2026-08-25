from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Literal, get_args

import torch

from neuralrec.run.callbacks.lr_schedule import OptimizerGroupScope, ScheduleShape


@dataclass
class DayRangeConfig:
    start_day: int = 0
    end_day: int = 10**9


@dataclass
class DataloaderConfig:
    batch_size: int = 512
    val_batch_size: int = 1024
    num_workers: int = 16
    prefetch_factor: int | None = 4
    gradient_accumulation_steps: int = 1

    def __post_init__(self) -> None:
        if (
            not isinstance(self.gradient_accumulation_steps, int)
            or isinstance(self.gradient_accumulation_steps, bool)
            or self.gradient_accumulation_steps < 1
        ):
            raise ValueError("gradient_accumulation_steps must be positive")

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps


@dataclass
class PretrainConfig:
    days: int = 0
    num_epochs: int = 1
    shuffle_days: bool = True


@dataclass
class EmbeddingConfig:
    num_embeddings: int = 131072
    dim: int = 64
    sparse: bool = True


PositionOrder = Literal["forward", "reverse"]
LearnedPositionFusion = Literal["add", "concat"]
LearnedPositionFusionNormalization = Literal["input_rms"]
LearnedPositionFusionResidual = Literal["rezero"]
LearnedPositionInitialization = Literal["default", "zero_reverse"]
LearnedPositionReverseCorrection = Literal["bounded_tanh"]
RopeKind = Literal[
    "forward",
    "reverse",
    "timestamp",
    "timestamp_reverse",
    "timestamp_log",
    "timestamp_log_reverse",
]
FFNKind = Literal["relu", "gelu", "silu", "reglu", "geglu", "swiglu"]
NormKind = Literal["rms", "layer", "batch"]
NormPlace = Literal["pre", "post"]


@dataclass(frozen=True)
class TransformerConfig:
    """The SASRec-shaped stack every sequence variant builds. Defaults are the
    shared ones; a variant that differs passes ``replace(TRANSFORMER, ...)``."""

    dim: int = 128
    num_layers: int = 2
    nhead: int = 4
    num_kv_heads: int = 2
    ffn_intermediate_dim: int = 256
    dropout: float = 0.1
    input_dropout: float = 0.0
    ffn_dropout: float = 0.0
    gated_ffn_dropout: bool = False
    ffn: FFNKind = "swiglu"
    norm: NormKind = "rms"
    norm_place: NormPlace = "pre"
    input_norm: NormKind | None = "layer"
    final_norm: NormKind | None = "layer"
    alibi: bool = True
    rope: RopeKind | None = None
    rope_base: float = 10000.0
    learned_positions: PositionOrder | tuple[PositionOrder, ...] | None = "reverse"
    learned_position_fusion: LearnedPositionFusion = "add"
    learned_position_fusion_normalization: LearnedPositionFusionNormalization | None = (
        None
    )
    learned_position_fusion_residual: LearnedPositionFusionResidual | None = None
    learned_position_initialization: LearnedPositionInitialization = "default"
    learned_position_reverse_correction: LearnedPositionReverseCorrection | None = None
    learned_position_reverse_max_scale: float = 0.1
    learned_position_reverse_initializer_rng_nonadvancing: bool = False
    attention_window: int | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.rope_base) or self.rope_base <= 0:
            raise ValueError("rope_base must be positive finite")
        if self.learned_position_fusion not in get_args(LearnedPositionFusion):
            raise ValueError(
                f"unknown learned position fusion {self.learned_position_fusion!r}"
            )
        if (
            self.learned_position_fusion_normalization is not None
            and self.learned_position_fusion_normalization
            not in get_args(LearnedPositionFusionNormalization)
        ):
            raise ValueError(
                "unknown position fusion normalization "
                f"{self.learned_position_fusion_normalization!r}"
            )
        if (
            self.learned_position_fusion_residual is not None
            and self.learned_position_fusion_residual
            not in get_args(LearnedPositionFusionResidual)
        ):
            raise ValueError(
                "unknown position fusion residual "
                f"{self.learned_position_fusion_residual!r}"
            )
        if (
            self.learned_position_fusion_normalization is not None
            and self.learned_position_fusion_residual is not None
        ):
            raise ValueError("position fusion normalization and residual are exclusive")
        if self.learned_position_fusion == "concat" and not self.learned_positions:
            raise ValueError("concatenated position fusion requires learned positions")
        if self.learned_position_fusion != "concat" and (
            self.learned_position_fusion_normalization is not None
            or self.learned_position_fusion_residual is not None
        ):
            raise ValueError("position fusion options require concatenation")
        if self.learned_position_initialization not in get_args(
            LearnedPositionInitialization
        ):
            raise ValueError(
                "unknown learned position initialization "
                f"{self.learned_position_initialization!r}"
            )
        if (
            self.learned_position_initialization == "zero_reverse"
            and self.learned_positions != ("forward", "reverse")
        ):
            raise ValueError(
                "zero-reverse initialization requires forward and reverse positions"
            )
        if (
            self.learned_position_reverse_correction is not None
            and self.learned_position_reverse_correction
            not in get_args(LearnedPositionReverseCorrection)
        ):
            raise ValueError(
                "unknown learned position reverse correction "
                f"{self.learned_position_reverse_correction!r}"
            )
        if self.learned_position_reverse_correction is not None and (
            self.learned_positions != ("forward", "reverse")
            or self.learned_position_initialization != "default"
            or (
                self.learned_position_fusion == "concat"
                and self.learned_position_fusion_residual != "rezero"
            )
        ):
            raise ValueError(
                "bounded reverse correction requires forward and reverse positions, "
                "default initialization, and concat r3 semantics when concatenated"
            )
        if (
            not isinstance(self.learned_position_reverse_max_scale, (int, float))
            or isinstance(self.learned_position_reverse_max_scale, bool)
            or not math.isfinite(self.learned_position_reverse_max_scale)
            or self.learned_position_reverse_max_scale <= 0
        ):
            raise ValueError(
                "learned position reverse max scale must be positive finite"
            )
        if (
            self.learned_position_reverse_correction is None
            and self.learned_position_reverse_max_scale != 0.1
        ):
            raise ValueError("reverse max scale requires bounded reverse correction")
        if (
            self.learned_position_reverse_initializer_rng_nonadvancing
            and self.learned_position_reverse_correction is None
        ):
            raise ValueError(
                "reverse initializer RNG isolation requires bounded reverse correction"
            )


def transformer_metadata(transformer: TransformerConfig) -> dict[str, Any]:
    metadata = asdict(transformer)
    if transformer.rope_base == 10000.0:
        metadata.pop("rope_base")
    if transformer.learned_position_fusion == "add":
        metadata.pop("learned_position_fusion")
        metadata.pop("learned_position_fusion_normalization")
        metadata.pop("learned_position_fusion_residual")
    else:
        if transformer.learned_position_fusion_residual == "rezero":
            metadata.pop("learned_position_fusion_normalization")
            metadata["learned_position_fusion_semantics_revision"] = 3
        else:
            metadata.pop("learned_position_fusion_residual")
            metadata["learned_position_fusion_semantics_revision"] = (
                2
                if transformer.learned_position_fusion_normalization is not None
                else 1
            )
    if transformer.learned_position_initialization == "default":
        metadata.pop("learned_position_initialization")
    else:
        metadata["learned_position_initialization_semantics_revision"] = 1
    if transformer.learned_position_reverse_correction is None:
        metadata.pop("learned_position_reverse_correction")
        metadata.pop("learned_position_reverse_max_scale")
    else:
        metadata["learned_position_reverse_correction_semantics_revision"] = 1
    if transformer.learned_position_reverse_initializer_rng_nonadvancing:
        metadata["learned_position_reverse_initializer_semantics_revision"] = 1
    else:
        metadata.pop("learned_position_reverse_initializer_rng_nonadvancing")
    return metadata


TRANSFORMER = TransformerConfig()


@dataclass(frozen=True)
class LrScheduleConfig:
    """Finite-horizon decay shapes and warmup need a known run length."""

    shape: ScheduleShape = "constant"
    warmup_fraction: float = 0.0
    min_lr_fraction: float = 0.0
    cycles: int = 1
    timescale_steps: int | None = None
    timescale_fraction: float | None = None
    power_exponent: float = -0.51
    power_transition_tokens: int | None = None
    optimizer_group_scope: OptimizerGroupScope = "both"

    @property
    def requires_horizon(self) -> bool:
        return (
            self.warmup_fraction > 0
            or self.timescale_fraction is not None
            or self.shape not in ("constant", "inverse_sqrt", "power")
        )

    @property
    def anneals_over_horizon(self) -> bool:
        """Whether the horizon, not early stopping, decides how long to train.

        These shapes decay against the whole horizon, so a run that stops before
        spending it measures a schedule it never finished applying.
        """
        return self.shape not in ("constant", "inverse_sqrt", "power")


@dataclass
class CheckpointConfig:
    # Off by default: a checkpoint is the model and both Adam moments, and a
    # run short enough to repeat is cheaper to repeat than to write to disk.
    # `BestWeights` keeps the best epoch in memory either way, so a run that
    # saves nothing is still reported on its best epoch.
    enabled: bool = False
    best_strategy: str = "last_n"
    best_n_checkpoints: int = 3
    best_metric_name: str = "val_loss"
    best_metric_mode: str = "min"
    best_metric_prefix: str = "epoch/val"
    load_checkpoint: bool = False
    last_n_checkpoints: int = 1

    def __post_init__(self) -> None:
        if self.best_metric_mode not in {"min", "max"}:
            raise ValueError(
                f"best_metric_mode must be 'min' or 'max', got {self.best_metric_mode!r}"
            )


@dataclass
class LoggingConfig:
    log_interval: int = 100
    enable_predictions: bool = True
    wandb_project: str = "ysda_recsys"
    prediction_int_columns: dict[str, str] = field(default_factory=dict)
    prediction_float_columns: dict[str, str] = field(default_factory=dict)


@dataclass
class RuntimeConfig:
    dtype: torch.dtype = torch.bfloat16
    compile: bool = False
    gradient_clip_norm: float | None = 1.0
