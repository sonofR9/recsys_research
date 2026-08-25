from __future__ import annotations

from dataclasses import fields, replace
import math
import os
from pathlib import Path
import re
import runpy
from typing import get_args

from dcn.config import GenerationExperiment, MuTransferGenerationExperiment
from dcn.nn.sampled_softmax import Correction
from dcn.training_metadata import GENERATION_TRAINING_SEMANTICS_REVISION


def _positive_integer(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default))
    if not raw.isdigit() or int(raw) < 1 or raw != str(int(raw)):
        raise ValueError(f"{name} must be a canonical positive integer")
    return int(raw)


def _positive_float(name: str, default: float) -> float:
    raw = os.environ.get(name, str(default))
    value = float(raw)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive finite")
    return value


def _boolean(name: str, default: bool) -> bool:
    raw = os.environ.get(name, str(int(default)))
    if raw not in {"0", "1"}:
        raise ValueError(f"{name} must be 0 or 1")
    return raw == "1"


def _fields(name: str, allowed: set[str]) -> tuple[str, ...]:
    raw = os.environ.get(name, "")
    selected = tuple(field for field in raw.split(",") if field)
    unknown = set(selected) - allowed
    if unknown:
        kind = "transformer" if "TRANSFORMER" in name else "experiment"
        raise ValueError(f"unknown {kind} field(s): {sorted(unknown)}")
    return selected


run = os.environ["G1_TUNE_RUN"]
if re.fullmatch(r"[a-z0-9_]+", run) is None:
    raise ValueError("G1_TUNE_RUN must contain only lowercase letters, digits, and _")
num_epochs = _positive_integer("G1_TUNE_EPOCHS", 20)
if num_epochs < 20:
    raise ValueError("G1_TUNE_EPOCHS must be at least 20")
revision = _positive_integer("G1_TUNE_RUN_REVISION", 2)
expected_suffix = f"_ts{GENERATION_TRAINING_SEMANTICS_REVISION}_r{revision}"
if num_epochs != 20:
    expected_suffix = f"_cap{num_epochs}{expected_suffix}"
if not run.endswith(expected_suffix):
    raise ValueError(f"G1_TUNE_RUN must end with {expected_suffix}")

os.environ["G1_VARIANT"] = "selected_quality_b1280"
os.environ.pop("G1_TRAIN_BATCH_SIZE", None)
os.environ.pop("G1_VAL_BATCH_SIZE", None)
namespace = runpy.run_path(str(Path(__file__).with_name("variant.py")))
variants = namespace["VARIANTS"]
base = variants["selected_quality_b1280"]
source_name = os.environ["G1_TUNE_SOURCE_VARIANT"]
if source_name not in variants:
    raise ValueError(f"unknown G1_TUNE_SOURCE_VARIANT {source_name!r}")
source = variants[source_name]

transformer_fields = _fields(
    "G1_TUNE_TRANSFORMER_FIELDS",
    {
        "dim",
        "num_layers",
        "nhead",
        "num_kv_heads",
        "ffn_intermediate_dim",
        "ffn",
        "norm",
        "norm_place",
        "input_norm",
        "final_norm",
        "alibi",
        "rope",
        "learned_positions",
        "dropout",
        "input_dropout",
        "ffn_dropout",
        "gated_ffn_dropout",
        "attention_window",
    },
)
experiment_fields = _fields(
    "G1_TUNE_EXPERIMENT_FIELDS",
    {
        "negative_sampling",
        "logq_correction",
        "correct_positive_logq",
        "mask_false_negatives",
        "exclude_own_group_negatives",
        "max_seq_len",
        "bos",
        "cls_token",
        "cls_token_mode",
        "lr_schedule",
        "timestamp_delta",
        "timestamp_combination",
        "timestamp_num_bins",
        "per_layer_item_embeddings",
    },
)
transformer = replace(
    base.transformer,
    **{name: getattr(source.transformer, name) for name in transformer_fields},
)
if "G1_TUNE_FFN_DIM" in os.environ:
    transformer = replace(
        transformer,
        ffn_intermediate_dim=_positive_integer("G1_TUNE_FFN_DIM", 1),
    )
if "G1_TUNE_NUM_LAYERS" in os.environ:
    transformer = replace(
        transformer,
        num_layers=_positive_integer("G1_TUNE_NUM_LAYERS", 1),
    )

overrides = {
    name: getattr(source, name)
    for name in experiment_fields
}
overrides.update(
    run_name=f"g1_rqtune_{run}_{base.size}",
    num_epochs=num_epochs,
    eval_every_n_epochs=1,
    restore_best_weights=True,
    early_stopping_patience=3,
    early_stopping_min_delta=0.0,
    transformer=transformer,
    dataloader=replace(
        base.dataloader,
        batch_size=_positive_integer("G1_TUNE_BATCH_SIZE", 1280),
        gradient_accumulation_steps=_positive_integer(
            "G1_TUNE_GRADIENT_ACCUMULATION_STEPS", 1
        ),
        num_workers=_positive_integer(
            "G1_TUNE_NUM_WORKERS", base.dataloader.num_workers
        ),
    ),
    embedding_learning_rate=_positive_float(
        "G1_TUNE_EMBEDDING_LR", base.embedding_learning_rate
    ),
    deep_learning_rate=_positive_float(
        "G1_TUNE_DEEP_LR", base.deep_learning_rate
    ),
    num_in_batch_negatives=_positive_integer("G1_TUNE_NUM_NEGATIVES", 512),
    logq_alpha=_positive_float("G1_TUNE_LOGQ_ALPHA", base.logq_alpha),
    correct_positive_logq=_boolean(
        "G1_TUNE_CORRECT_POSITIVE_LOGQ",
        overrides.get("correct_positive_logq", base.correct_positive_logq),
    ),
    mask_false_negatives=_boolean(
        "G1_TUNE_MASK_FALSE_NEGATIVES",
        overrides.get("mask_false_negatives", base.mask_false_negatives),
    ),
    exclude_own_group_negatives=_boolean(
        "G1_TUNE_EXCLUDE_OWN_GROUP_NEGATIVES",
        overrides.get(
            "exclude_own_group_negatives", base.exclude_own_group_negatives
        ),
    ),
)
if "G1_TUNE_LOGQ_CORRECTION" in os.environ:
    correction = os.environ["G1_TUNE_LOGQ_CORRECTION"]
    if correction not in get_args(Correction):
        raise ValueError(
            f"G1_TUNE_LOGQ_CORRECTION must be one of {get_args(Correction)}"
        )
    overrides["logq_correction"] = correction
if "G1_TUNE_RANDOM_FRACTION" in os.environ:
    fraction = float(os.environ["G1_TUNE_RANDOM_FRACTION"])
    if not math.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError("G1_TUNE_RANDOM_FRACTION must be in (0, 1)")
    overrides["random_negative_fraction"] = fraction

common = {
    field.name: getattr(base, field.name)
    for field in fields(GenerationExperiment)
    if field.init
}
common.update(overrides)
common["item_embedding_dim"] = 64
experiment = MuTransferGenerationExperiment(
    **common,
    mup_base_dim=16,
    mup_delta_dim=32,
)
