from __future__ import annotations

from dataclasses import fields, replace
import os
from pathlib import Path
import runpy

from dcn.config import GenerationExperiment, MuTransferGenerationExperiment
from dcn.config.settings import LrScheduleConfig
from experiments.g1_sasrec_item_ids_likes.analysis.rq10_reinvestigation_candidates import (
    candidate_by_run,
)


candidate = candidate_by_run(os.environ["G1_RQ10_RUN"])
os.environ["G1_DATASET_SIZE"] = candidate.dataset_size
os.environ["G1_MAX_EPOCHS"] = str(candidate.horizon_epochs)
os.environ["G1_VARIANT"] = "selected_quality_b1280"
namespace = runpy.run_path(str(Path(__file__).with_name("variant.py")))
base = namespace["VARIANTS"]["selected_quality_b1280"]

overrides = {
    "run_name": candidate.run_name,
    "seed": candidate.seed,
    "num_epochs": candidate.horizon_epochs,
    "lr_schedule_horizon_epochs": candidate.horizon_epochs,
    "lr_schedule": LrScheduleConfig("linear"),
    "adaptive_schedule_early_stopping": False,
    "eval_every_n_epochs": 1,
    "restore_best_weights": True,
    "transformer": replace(base.transformer, num_layers=candidate.num_layers),
    "max_seq_len": 128,
    "bos": False,
    "cls_token": False,
    "cls_token_mode": "none",
    "dataloader": replace(
        base.dataloader,
        batch_size=candidate.batch_size,
        gradient_accumulation_steps=1,
    ),
    "embedding_learning_rate": candidate.embedding_lr,
    "deep_learning_rate": candidate.deep_lr,
    "per_layer_item_embeddings": False,
    "per_layer_item_features": (
        "none" if candidate.family == "input_output_only" else candidate.family
    ),
    "per_layer_item_feature_dim": (
        candidate.feature_width
        if candidate.family in {"concat_residual", "gemma_ple"}
        else None
    ),
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
