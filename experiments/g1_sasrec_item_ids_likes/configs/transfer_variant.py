from __future__ import annotations

from dataclasses import fields, replace
import math
import os
from pathlib import Path
import re
import runpy

from dcn.config import GenerationExperiment, MuTransferGenerationExperiment
from dcn.config.settings import LrScheduleConfig
from dcn.training_metadata import GENERATION_TRAINING_SEMANTICS_REVISION


def _positive_integer(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default))
    if not raw.isdigit() or int(raw) < 1 or raw != str(int(raw)):
        raise ValueError(f"{name} must be a canonical positive integer")
    return int(raw)


def _positive_float(name: str) -> float:
    raw = os.environ[name]
    value = float(raw)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive finite")
    return value


run = os.environ["G1_TRANSFER_RUN"]
if re.fullmatch(r"[a-z0-9_]+", run) is None:
    raise ValueError(
        "G1_TRANSFER_RUN must contain only lowercase letters, digits, and _"
    )
epochs = _positive_integer("G1_TRANSFER_EPOCHS", 20)
if epochs < 20:
    raise ValueError("G1_TRANSFER_EPOCHS must be at least the 20-epoch safety cap")
run_revision = _positive_integer("G1_TRANSFER_RUN_REVISION", 2)
cap = f"_cap{epochs}" if epochs != 20 else ""
expected_identity = (
    f"{cap}_ts{GENERATION_TRAINING_SEMANTICS_REVISION}_r{run_revision}"
)
explicit_initial_identity = (
    f"_cap20_ts{GENERATION_TRAINING_SEMANTICS_REVISION}_r{run_revision}"
)
if not run.endswith(expected_identity) and not (
    epochs == 20 and run.endswith(explicit_initial_identity)
):
    raise ValueError(f"G1_TRANSFER_RUN must end with {expected_identity}")
batch_size = _positive_integer("G1_TRANSFER_BATCH_SIZE", 1280)
embedding_learning_rate = _positive_float("G1_TRANSFER_EMBEDDING_LR")
deep_learning_rate = _positive_float("G1_TRANSFER_DEEP_LR")
parameterization = os.environ.get("G1_TRANSFER_PARAMETERIZATION", "conventional")
if parameterization not in {"conventional", "mup"}:
    raise ValueError("G1_TRANSFER_PARAMETERIZATION must be conventional or mup")
source_variant = os.environ.get(
    "G1_TRANSFER_SOURCE_VARIANT", "selected_quality_b1280"
)
if source_variant not in {
    "homework_fixed_leave_one_out",
    "selected_quality_b1280",
}:
    raise ValueError("G1_TRANSFER_SOURCE_VARIANT must be an approved source variant")

os.environ["G1_VARIANT"] = source_variant
os.environ.pop("G1_TRAIN_BATCH_SIZE", None)
os.environ.pop("G1_VAL_BATCH_SIZE", None)
variant_path = Path(__file__).with_name("variant.py")
base = runpy.run_path(str(variant_path))["VARIANTS"][source_variant]
dimension = _positive_integer("G1_TRANSFER_DIM", base.transformer.dim)
power_tokens = (
    _positive_integer("G1_TRANSFER_POWER_TOKENS", 1)
    if "G1_TRANSFER_POWER_TOKENS" in os.environ
    else None
)
mup_base_ffn_dim = (
    _positive_integer("G1_TRANSFER_MUP_FFN_BASE", 1)
    if "G1_TRANSFER_MUP_FFN_BASE" in os.environ
    else None
)
if mup_base_ffn_dim is not None and parameterization != "mup":
    raise ValueError(
        "G1_TRANSFER_MUP_FFN_BASE needs G1_TRANSFER_PARAMETERIZATION=mup"
    )
transformer = replace(
    base.transformer,
    dim=dimension,
    ffn_intermediate_dim=(
        _positive_integer("G1_TRANSFER_FFN_DIM", 1)
        if "G1_TRANSFER_FFN_DIM" in os.environ
        else round(
            base.transformer.ffn_intermediate_dim * dimension / base.transformer.dim
        )
    ),
)
overrides = {
    "run_name": f"g1_transfer_{run}_{base.size}",
    "num_epochs": epochs,
    "eval_every_n_epochs": 1,
    "restore_best_weights": True,
    "early_stopping_patience": 3,
    "early_stopping_min_delta": 0.0,
    "dataloader": replace(base.dataloader, batch_size=batch_size),
    "embedding_learning_rate": embedding_learning_rate,
    "deep_learning_rate": deep_learning_rate,
    "transformer": transformer,
}
if power_tokens is not None:
    overrides["lr_schedule"] = LrScheduleConfig(
        "power", power_exponent=-0.51, power_transition_tokens=power_tokens
    )

if parameterization == "conventional":
    experiment = replace(base, **overrides)
else:
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
        mup_base_ffn_dim=mup_base_ffn_dim,
        mup_delta_ffn_dim=(
            None if mup_base_ffn_dim is None else 2 * mup_base_ffn_dim
        ),
    )
