from __future__ import annotations

from dataclasses import fields, replace
import os
from pathlib import Path
import runpy

from dcn.config import GenerationExperiment, MuTransferGenerationExperiment
from dcn.config.settings import LrScheduleConfig
from experiments.g1_sasrec_item_ids_likes.analysis.rq8_reinvestigation_candidates import (
    candidate_by_run,
)


candidate = candidate_by_run(os.environ["G1_RQ8_RUN"])
os.environ["G1_DATASET_SIZE"] = "500m"
os.environ["G1_MAX_EPOCHS"] = str(candidate.cap_epochs)
os.environ["G1_VARIANT"] = "selected_quality_b1280"
namespace = runpy.run_path(str(Path(__file__).with_name("variant.py")))
base = namespace["VARIANTS"]["selected_quality_b1280"]

if candidate.position_method == "learned_forward":
    position = {"alibi": False, "rope": None, "learned_positions": "forward"}
elif candidate.position_method == "alibi":
    position = {"alibi": True, "rope": None, "learned_positions": None}
else:
    position = {"alibi": True, "rope": "reverse", "learned_positions": None}

cls_token_mode = (
    candidate.query_method if candidate.study == "query" else "none"
)
if cls_token_mode == "standard":
    cls_token_mode = "none"
attention_window = (
    {
        "standard": 50,
        "end_only": 51,
        "interleaved": 100,
    }[candidate.query_method]
    if candidate.study == "query"
    else None
)

overrides = {
    "run_name": candidate.run_name,
    "seed": candidate.seed,
    "num_epochs": candidate.cap_epochs,
    "lr_schedule_horizon_epochs": candidate.cap_epochs,
    "lr_schedule": LrScheduleConfig("linear"),
    "adaptive_schedule_early_stopping": False,
    "eval_every_n_epochs": 1,
    "restore_best_weights": True,
    "early_stopping_patience": 3,
    "early_stopping_min_delta": 0.0,
    "transformer": replace(
        base.transformer,
        attention_window=attention_window,
        **position,
    ),
    "max_seq_len": candidate.max_seq_len,
    "cls_token": False,
    "cls_token_mode": cls_token_mode,
    "dataloader": replace(
        base.dataloader,
        batch_size=candidate.batch_size,
        gradient_accumulation_steps=1,
    ),
    "embedding_learning_rate": candidate.embedding_lr,
    "deep_learning_rate": candidate.deep_lr,
    "dense_random_negative_scores": False,
}
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
