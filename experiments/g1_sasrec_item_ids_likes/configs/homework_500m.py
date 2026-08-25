"""Exact fixed leave-one-out homework baseline on Yambda-500M."""

import os
from dataclasses import replace
import re

os.environ["G1_VARIANT"] = "homework_fixed_leave_one_out"
os.environ["G1_DATASET_SIZE"] = "500m"
os.environ.pop("G1_MAX_USERS", None)
os.environ.pop("G1_VAL_BATCH_SIZE", None)
os.environ.pop("G1_TRAIN_BATCH_SIZE", None)

from experiments.g1_sasrec_item_ids_likes.configs.variant import VARIANTS  # noqa: E402


homework = VARIANTS["homework_fixed_leave_one_out"]
batch_size = os.environ.get("G1_HOMEWORK_BATCH_SIZE")
run_tag = os.environ.get("G1_HOMEWORK_RUN_TAG")
if run_tag is not None and re.fullmatch(r"[a-z0-9_]+", run_tag) is None:
    raise ValueError(
        "G1_HOMEWORK_RUN_TAG must contain lowercase letters, digits, and _"
    )
if batch_size is not None:
    if (
        not batch_size.isdigit()
        or int(batch_size) < 1
        or batch_size != str(int(batch_size))
    ):
        raise ValueError("G1_HOMEWORK_BATCH_SIZE must be a canonical positive integer")
    base_run_name = homework.run_name.removesuffix("_500m")
    homework = replace(
        homework,
        run_name=f"{base_run_name}_b{batch_size}_500m",
        dataloader=replace(homework.dataloader, batch_size=int(batch_size)),
    )
if run_tag is not None:
    base_run_name = homework.run_name.removesuffix("_500m")
    homework = replace(homework, run_name=f"{base_run_name}_{run_tag}_500m")
seed = os.environ.get("G1_SEED")
experiment = (
    replace(
        homework,
        seed=int(seed),
        run_name=f"{homework.run_name}_s{seed}",
    )
    if seed is not None
    else homework
)
