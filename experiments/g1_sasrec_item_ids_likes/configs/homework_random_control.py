from __future__ import annotations

from dataclasses import replace
import math
import os
from pathlib import Path
import re
import runpy

from dcn.training_metadata import GENERATION_TRAINING_SEMANTICS_REVISION


def _positive_integer(name: str) -> int:
    raw = os.environ[name]
    if not raw.isdigit() or int(raw) < 1 or raw != str(int(raw)):
        raise ValueError(f"{name} must be a canonical positive integer")
    return int(raw)


def _positive_float(name: str) -> float:
    value = float(os.environ[name])
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive finite")
    return value


run = os.environ["G1_HOMEWORK_RANDOM_RUN"]
if re.fullmatch(r"[a-z0-9_]+", run) is None:
    raise ValueError(
        "G1_HOMEWORK_RANDOM_RUN must contain lowercase letters, digits, and _"
    )
epochs = _positive_integer("G1_HOMEWORK_RANDOM_EPOCHS")
if epochs < 20:
    raise ValueError("G1_HOMEWORK_RANDOM_EPOCHS must be at least 20")
revision = _positive_integer("G1_HOMEWORK_RANDOM_RUN_REVISION")
suffix = f"_ts{GENERATION_TRAINING_SEMANTICS_REVISION}_r{revision}"
if epochs != 20:
    suffix = f"_cap{epochs}{suffix}"
if not run.endswith(suffix):
    raise ValueError(f"G1_HOMEWORK_RANDOM_RUN must end with {suffix}")

dataset_size = os.environ.get("G1_HOMEWORK_RANDOM_DATASET_SIZE", "50m")
if dataset_size not in {"50m", "500m"}:
    raise ValueError("G1_HOMEWORK_RANDOM_DATASET_SIZE must be 50m or 500m")
embedding_lr = _positive_float("G1_HOMEWORK_RANDOM_EMBEDDING_LR")
deep_lr = _positive_float("G1_HOMEWORK_RANDOM_DEEP_LR")
environment_names = (
    "G1_DATASET_SIZE",
    "G1_MAX_EPOCHS",
    "G1_VARIANT",
    "G1_MAX_USERS",
    "G1_VAL_BATCH_SIZE",
    "G1_TRAIN_BATCH_SIZE",
)
previous_environment = {name: os.environ.get(name) for name in environment_names}
try:
    os.environ["G1_DATASET_SIZE"] = dataset_size
    os.environ["G1_MAX_EPOCHS"] = str(epochs)
    os.environ["G1_VARIANT"] = "homework_fixed_leave_one_out"
    os.environ.pop("G1_MAX_USERS", None)
    os.environ.pop("G1_VAL_BATCH_SIZE", None)
    os.environ.pop("G1_TRAIN_BATCH_SIZE", None)
    namespace = runpy.run_path(str(Path(__file__).with_name("variant.py")))
finally:
    for name, value in previous_environment.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
control = namespace["VARIANTS"]["homework_baseline_native500_r3"]

experiment = replace(
    control,
    run_name=f"g1_homework_random_{run}_{dataset_size}",
    negative_sampling="random",
    embedding_learning_rate=embedding_lr,
    deep_learning_rate=deep_lr,
)
