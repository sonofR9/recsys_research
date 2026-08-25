from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import runpy

from dcn.config.settings import LrScheduleConfig
from experiments.g1_sasrec_item_ids_likes.analysis.rq5_scheduler_candidates import (
    candidate_by_run,
)


candidate = candidate_by_run(os.environ["G1_RQ5_RUN"])
os.environ["G1_DATASET_SIZE"] = "500m"
os.environ["G1_MAX_EPOCHS"] = str(max(20, candidate.cap_epochs))
os.environ["G1_VARIANT"] = "selected_quality_b1280"
namespace = runpy.run_path(str(Path(__file__).with_name("variant.py")))
base = namespace["VARIANTS"]["selected_quality_b1280"]
schedule = LrScheduleConfig(
    candidate.shape,
    warmup_fraction=candidate.warmup_fraction,
    cycles=candidate.cycles,
    timescale_fraction=candidate.timescale_fraction,
    optimizer_group_scope=candidate.scope,
)
experiment = replace(
    base,
    run_name=candidate.run_name,
    seed=candidate.seed,
    num_epochs=candidate.cap_epochs,
    lr_schedule_horizon_epochs=candidate.horizon_epochs,
    lr_schedule=schedule,
    adaptive_schedule_early_stopping=True,
    early_stopping_patience=3,
    early_stopping_min_delta=0.0,
    embedding_learning_rate=candidate.embedding_lr,
    deep_learning_rate=candidate.deep_lr,
    dataloader=replace(
        base.dataloader,
        batch_size=1280,
        gradient_accumulation_steps=1,
    ),
)
