from __future__ import annotations

from dataclasses import fields, replace
import os
from pathlib import Path
import runpy

from dcn.config import GenerationExperiment, MuTransferGenerationExperiment
from dcn.config.settings import LrScheduleConfig
from experiments.g1_sasrec_item_ids_likes.analysis.rq7_reinvestigation_candidates import (
    candidate_by_run,
)


candidate = candidate_by_run(os.environ["G1_RQ7_RUN"])
os.environ["G1_DATASET_SIZE"] = candidate.dataset_size
os.environ["G1_MAX_EPOCHS"] = str(candidate.horizon_epochs)
os.environ["G1_VARIANT"] = "selected_quality_b1280"
namespace = runpy.run_path(str(Path(__file__).with_name("variant.py")))
base = namespace["VARIANTS"]["selected_quality_b1280"]
position = candidate.position

overrides = {
    "run_name": candidate.run_name,
    "seed": candidate.seed,
    "num_epochs": candidate.horizon_epochs,
    "lr_schedule_horizon_epochs": candidate.horizon_epochs,
    "lr_schedule": LrScheduleConfig("linear"),
    "adaptive_schedule_early_stopping": False,
    "eval_every_n_epochs": 1,
    "restore_best_weights": True,
    "transformer": replace(
        base.transformer,
        learned_positions=position.learned_positions,
        learned_position_fusion=position.learned_position_fusion,
        learned_position_fusion_normalization=(
            "input_rms" if candidate.implementation_revision == 2 else None
        ),
        learned_position_fusion_residual=(
            "rezero"
            if position.learned_position_fusion == "concat"
            and candidate.implementation_revision in (3, 4, 5, 6, 7)
            else None
        ),
        learned_position_initialization=(
            "zero_reverse" if candidate.implementation_revision == 4 else "default"
        ),
        learned_position_reverse_correction=(
            "bounded_tanh" if candidate.implementation_revision in (5, 6, 7) else None
        ),
        learned_position_reverse_max_scale=(
            0.025 if candidate.implementation_revision in (6, 7) else 0.1
        ),
        learned_position_reverse_initializer_rng_nonadvancing=(
            candidate.implementation_revision == 7
        ),
        rope=position.rope,
        rope_base=position.rope_base,
        alibi=position.alibi,
    ),
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
