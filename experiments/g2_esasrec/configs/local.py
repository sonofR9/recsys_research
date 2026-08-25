from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import ClassVar, Literal, get_args

import torch
from torch import nn

from dcn.config import GenerationExperiment
from dcn.config.settings import DataloaderConfig, LrScheduleConfig, RuntimeConfig
from dcn.models import TwoTowerLoss
from dcn.nn.sampled_softmax import RandomCatalogNegatives
from dcn.nn.transformer import (
    ReverseRelativePositionInput,
    TransformerEncoder,
)
from dcn.nn.types import ModuleWithDim
from experiments.generation_protocol import generation_protocol

LayerFamily = Literal["sasrec", "ligr"]
LossKind = Literal["sampled_softmax", "gbce"]
ComponentMethod = Literal[
    "standard_sampled_softmax",
    "standard_gbce",
    "matched_standard_sampled_softmax",
    "matched_standard_gbce",
    "ligr_sampled_softmax",
    "ligr_gbce",
]

COMPONENT_METHODS: tuple[ComponentMethod, ...] = get_args(ComponentMethod)
CONTROL_BATCHES = (128, 256, 512, 1024, 1280)
EMBEDDING_LR_BOUNDS = (1e-4, 0.256)
DEEP_LR_BOUNDS = (1e-4, 0.128)
GBCE_T_BOUNDS = (0.25, 1.0)
MIXED_UNIFORM_FRACTION_BOUNDS = (0.2, 0.8)
LIGR_WIDTHS = {2: 512, 4: 1024, 6: 1536}
MATCHED_STANDARD_WIDTHS = {2: 1024, 4: 1792, 6: 2560}
_RETOOLS_LAYER_NORM_EPS = 1e-8

_DATA_PROTOCOL = generation_protocol(
    event_type_filter="like",
    window="next_item",
    size="50m",
)
_CONTROL_TRANSFORMER = replace(
    GenerationExperiment.transformer,
    dim=64,
    num_layers=2,
    nhead=2,
    num_kv_heads=1,
    ffn_intermediate_dim=171,
    dropout=0.1,
    input_dropout=0.1,
    ffn_dropout=0.1,
    gated_ffn_dropout=False,
    ffn="swiglu",
    norm="layer",
    norm_place="pre",
    input_norm="rms",
    final_norm="layer",
    alibi=False,
    rope=None,
    learned_positions="forward",
    attention_window=50,
)
_COMPONENT_TRANSFORMER = replace(
    GenerationExperiment.transformer,
    dim=256,
    num_layers=2,
    nhead=4,
    num_kv_heads=4,
    ffn_intermediate_dim=1024,
    dropout=0.2,
    input_dropout=0.2,
    ffn_dropout=0.2,
    gated_ffn_dropout=True,
    ffn="swiglu",
    norm="layer",
    norm_place="pre",
    input_norm=None,
    final_norm=None,
    alibi=False,
    rope=None,
    learned_positions="reverse",
    attention_window=None,
)


@dataclass
class LocalG2Experiment(GenerationExperiment):
    training_reverse_position_offset: ClassVar[int] = 1
    layer_family: LayerFamily = "ligr"
    loss_kind: LossKind = "sampled_softmax"
    gbce_t: float | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.layer_family not in get_args(LayerFamily):
            raise ValueError(f"unknown layer_family {self.layer_family!r}")
        if self.loss_kind not in get_args(LossKind):
            raise ValueError(f"unknown loss_kind {self.loss_kind!r}")
        if self.layer_family == "ligr" and self.transformer.ffn_intermediate_dim % 32:
            raise ValueError("LiGR intermediate width must be divisible by 32")
        if self.loss_kind == "gbce":
            if self.gbce_t is None or not (
                GBCE_T_BOUNDS[0] <= self.gbce_t <= GBCE_T_BOUNDS[1]
            ):
                raise ValueError("gbce_t must be in [0.25, 1.0]")
        elif self.gbce_t is not None:
            raise ValueError("only gBCE methods accept gbce_t")

    def create_sequence_model(self, tokens_per_event: int) -> ModuleWithDim:
        from dcn.nn.esasrec import LiGRBlock, SASRecBlock

        block_class = SASRecBlock if self.layer_family == "sasrec" else LiGRBlock
        blocks = [
            block_class(
                dim=self.transformer.dim,
                nhead=self.transformer.nhead,
                intermediate_dim=self.transformer.ffn_intermediate_dim,
                dropout=self.transformer.dropout,
            )
            for _ in range(self.transformer.num_layers)
        ]
        events_per_sequence = self.max_seq_len + (self.window == "next_item")
        cls_tokens = (
            events_per_sequence
            if self.effective_cls_token_mode == "interleaved"
            else int(self.effective_cls_token_mode == "end_only")
        )
        max_sequence_length = (
            events_per_sequence * tokens_per_event + self.bos + cls_tokens
        )
        return TransformerEncoder(
            blocks=blocks,
            final_norm=(
                nn.LayerNorm(self.model_dim, eps=_RETOOLS_LAYER_NORM_EPS)
                if self.layer_family == "sasrec"
                else nn.Identity()
            ),
            input_norm=nn.Identity(),
            position_inputs=[
                ReverseRelativePositionInput(self.model_dim, max_sequence_length)
            ],
            position_dropout=self.transformer.input_dropout,
            max_seqlen=max_sequence_length,
        )

    def _create_model(self) -> nn.Module:
        model = super()._create_model()
        model.training_reverse_position_offset = self.training_reverse_position_offset
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, (nn.LayerNorm, nn.RMSNorm)):
                    module.weight.fill_(1.0)
        return model

    def create_criterion(self) -> nn.Module:
        if self.loss_kind == "sampled_softmax":
            return super().create_criterion()

        from dcn.nn.sampled_softmax import GeneralizedBCELoss

        random_negatives = RandomCatalogNegatives(
            catalog_size=self.catalog_size,
            first_item_id=1,
            num_negatives=self.num_in_batch_negatives,
            item_encoder=self.base_model.item_embedding,
            dense_scores=self.dense_random_negative_scores,
        )
        loss = GeneralizedBCELoss(
            catalog_size=self.num_items,
            num_in_batch_negatives=0,
            t=self.gbce_t,
            random_negatives=random_negatives,
            mask_false_negatives=self.mask_false_negatives,
            exclude_own_group=self.exclude_own_group_negatives,
        )
        return TwoTowerLoss(self.base_model, loss, targets=self.create_targets())

    def _report_training_metadata(self, runner) -> None:
        super()._report_training_metadata(runner)
        destination = (
            Path(self.base_path) / "logs" / self.run_name / "training_metadata.json"
        )
        metadata = json.loads(destination.read_text())
        metadata["g2_recipe"] = {
            "layer_family": self.layer_family,
            "loss_kind": self.loss_kind,
            "gbce_t": self.gbce_t,
        }
        destination.write_text(json.dumps(metadata, indent=2, sort_keys=True))


def _dataloader(batch_size: int) -> DataloaderConfig:
    if (
        not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size < 1
    ):
        raise ValueError("batch_size must be a positive integer")
    return DataloaderConfig(
        batch_size=batch_size,
        val_batch_size=8192,
        num_workers=4,
        prefetch_factor=4,
    )


def _positive_rate(name: str, value: float) -> float:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive finite")
    return value


def build_control(
    *,
    batch_size: int = 512,
    embedding_learning_rate: float = 0.032,
    deep_learning_rate: float = 0.012,
    run_name: str = "g2_control_native50m",
) -> GenerationExperiment:
    return GenerationExperiment(
        run_name=run_name,
        **_DATA_PROTOCOL,
        dataloader=_dataloader(batch_size),
        num_epochs=20,
        lr_schedule=LrScheduleConfig("linear"),
        lr_schedule_horizon_epochs=20,
        eval_every_n_epochs=1,
        restore_best_weights=True,
        early_stopping_patience=3,
        early_stopping_min_delta=0.0,
        transformer=_CONTROL_TRANSFORMER,
        max_seq_len=128,
        timestamp_delta="bins",
        timestamp_combination="add",
        timestamp_num_bins=16,
        negative_sampling="random",
        num_in_batch_negatives=512,
        logq_correction="yi2019",
        correct_positive_logq=False,
        mask_false_negatives=False,
        exclude_own_group_negatives=False,
        dense_random_negative_scores=True,
        initializer_std=0.02,
        embedding_learning_rate=_positive_rate(
            "embedding_learning_rate", embedding_learning_rate
        ),
        deep_learning_rate=_positive_rate("deep_learning_rate", deep_learning_rate),
        weight_decay=0.0,
        runtime=RuntimeConfig(
            dtype=torch.bfloat16,
            compile=False,
            gradient_clip_norm=None,
        ),
    )


def build_component(
    method: ComponentMethod,
    *,
    batch_size: int = 128,
    embedding_learning_rate: float = 0.001,
    deep_learning_rate: float = 0.001,
    ligr_multiplier: int = 4,
    gbce_t: float | None = None,
    run_name: str | None = None,
) -> LocalG2Experiment:
    if method not in COMPONENT_METHODS:
        raise ValueError(f"unknown component method {method!r}")
    if ligr_multiplier not in LIGR_WIDTHS:
        raise ValueError(f"ligr_multiplier must be one of {tuple(LIGR_WIDTHS)}")

    matched = method.startswith("matched_standard_")
    layer_family: LayerFamily = "ligr" if method.startswith("ligr_") else "sasrec"
    loss_kind: LossKind = "gbce" if method.endswith("_gbce") else "sampled_softmax"
    if loss_kind == "gbce":
        gbce_t = 0.75 if gbce_t is None else gbce_t
    elif gbce_t is not None:
        raise ValueError("only gBCE methods accept gbce_t")

    if layer_family == "ligr":
        width = LIGR_WIDTHS[ligr_multiplier]
        ffn = "swiglu"
        gated_ffn_dropout = True
    else:
        width = MATCHED_STANDARD_WIDTHS[ligr_multiplier] if matched else 256
        ffn = "relu"
        gated_ffn_dropout = False

    transformer = replace(
        _COMPONENT_TRANSFORMER,
        ffn=ffn,
        ffn_intermediate_dim=width,
        gated_ffn_dropout=gated_ffn_dropout,
        final_norm="layer" if layer_family == "sasrec" else None,
    )
    return LocalG2Experiment(
        run_name=run_name or f"g2_{method}_m{ligr_multiplier}_native50m",
        **_DATA_PROTOCOL,
        dataloader=_dataloader(batch_size),
        num_epochs=100,
        lr_schedule=LrScheduleConfig("constant"),
        eval_every_n_epochs=1,
        restore_best_weights=True,
        early_stopping_patience=10,
        early_stopping_min_delta=0.0,
        transformer=transformer,
        max_seq_len=100,
        negative_sampling="random",
        num_in_batch_negatives=256,
        logq_correction="none",
        correct_positive_logq=False,
        mask_false_negatives=False,
        exclude_own_group_negatives=False,
        dense_random_negative_scores=False,
        initializer_std=0.02,
        embedding_learning_rate=_positive_rate(
            "embedding_learning_rate", embedding_learning_rate
        ),
        deep_learning_rate=_positive_rate("deep_learning_rate", deep_learning_rate),
        weight_decay=0.0,
        runtime=RuntimeConfig(
            dtype=torch.bfloat16,
            compile=False,
            gradient_clip_norm=None,
        ),
        layer_family=layer_family,
        loss_kind=loss_kind,
        gbce_t=gbce_t,
    )


def build_mixed_sampler(
    *,
    uniform_fraction: float,
    logq_correction: Literal["none", "yi2019"],
    batch_size: int = 128,
    embedding_learning_rate: float = 0.001,
    deep_learning_rate: float = 0.001,
    ligr_multiplier: int = 4,
    run_name: str | None = None,
) -> LocalG2Experiment:
    if (
        not math.isfinite(uniform_fraction)
        or not MIXED_UNIFORM_FRACTION_BOUNDS[0]
        <= uniform_fraction
        <= MIXED_UNIFORM_FRACTION_BOUNDS[1]
    ):
        raise ValueError("uniform_fraction must be in [0.2, 0.8]")
    if logq_correction not in {"none", "yi2019"}:
        raise ValueError("logq_correction must be 'none' or 'yi2019'")
    base = build_component(
        "ligr_sampled_softmax",
        batch_size=batch_size,
        embedding_learning_rate=embedding_learning_rate,
        deep_learning_rate=deep_learning_rate,
        ligr_multiplier=ligr_multiplier,
    )
    fraction_tag = str(uniform_fraction).replace(".", "p")
    return replace(
        base,
        run_name=(
            run_name
            or (
                f"g2_mixed_u{fraction_tag}_{logq_correction}"
                f"_m{ligr_multiplier}_native50m"
            )
        ),
        negative_sampling="mixed_online_global_q",
        random_negative_fraction=uniform_fraction,
        logq_correction=logq_correction,
        correct_positive_logq=logq_correction == "yi2019",
    )
