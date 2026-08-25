from __future__ import annotations

from dataclasses import fields, replace
import os
from pathlib import Path
import runpy

from dcn.config import GenerationExperiment, MuTransferGenerationExperiment
from dcn.config.settings import LrScheduleConfig
from experiments.g1_sasrec_item_ids_likes.analysis.rq11_mixed_streaming_candidates import (
    candidate_by_run,
)


candidate = candidate_by_run(os.environ["G1_RQ11_RUN"])
os.environ["G1_DATASET_SIZE"] = "500m"
os.environ["G1_MAX_EPOCHS"] = str(candidate.horizon_epochs)
os.environ["G1_VARIANT"] = "selected_quality_b1280"
namespace = runpy.run_path(str(Path(__file__).with_name("variant.py")))
base = namespace["VARIANTS"]["selected_quality_b1280"]

negative_sampling = {
    "uniform_catalog": "random",
    "streaming_global_q": "online_logq",
    "popularity_global_q": "random_offline_logq",
    "aggregate_uniform_streaming_global_q": "mixed_online_global_q",
    "aggregate_uniform_streaming_global_q_negative_only": (
        "mixed_online_global_q_negative_only"
    ),
}[candidate.family]

overrides = {
    "run_name": candidate.run_name,
    "seed": candidate.seed,
    "num_epochs": candidate.horizon_epochs,
    "lr_schedule_horizon_epochs": candidate.horizon_epochs,
    "lr_schedule": LrScheduleConfig("linear"),
    "adaptive_schedule_early_stopping": False,
    "early_stopping_patience": None,
    "eval_every_n_epochs": 1,
    "restore_best_weights": True,
    "dataloader": replace(
        base.dataloader,
        batch_size=candidate.batch_size,
        gradient_accumulation_steps=1,
    ),
    "embedding_learning_rate": candidate.embedding_lr,
    "deep_learning_rate": candidate.deep_lr,
    "num_in_batch_negatives": candidate.negative_count,
    "logq_alpha": 0.01 if candidate.alpha is None else candidate.alpha,
    "random_negative_fraction": (
        0.5 if candidate.uniform_fraction is None else candidate.uniform_fraction
    ),
    "negative_sampling": negative_sampling,
    "logq_correction": "yi2019",
    "correct_positive_logq": (
        candidate.family != "aggregate_uniform_streaming_global_q_negative_only"
    ),
    "mask_false_negatives": False,
    "exclude_own_group_negatives": False,
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
