from __future__ import annotations

from dataclasses import fields, replace
import os
from pathlib import Path
import runpy

from dcn.config import GenerationExperiment
from dcn.config.query_retrieval_training import (
    MuTransferRq14PretrainedCrossAttentionGenerationExperiment,
)
from dcn.config.settings import LrScheduleConfig
from experiments.g1_sasrec_item_ids_likes.analysis.rq14_pretrained_candidates import (
    candidate_by_run,
)
from experiments.g1_sasrec_item_ids_likes.analysis.rq15_training_candidates import (
    RQ15_SOURCE_CHECKPOINT_NAME,
    source_candidate_by_run,
    source_checkpoint_metadata,
)


candidate = candidate_by_run(os.environ["G1_RQ14_PRETRAINED_RUN"])
source_run = os.environ.get("G1_RQ15_SOURCE_RUN")
if source_run is None:
    raise ValueError("pretrained RQ14 run requires G1_RQ15_SOURCE_RUN")
source_candidate = source_candidate_by_run(source_run)

os.environ["G1_DATASET_SIZE"] = candidate.dataset_size
os.environ["G1_MAX_EPOCHS"] = str(candidate.horizon_epochs)
os.environ["G1_VARIANT"] = "selected_quality_b1280"
namespace = runpy.run_path(str(Path(__file__).with_name("variant.py")))
base = namespace["VARIANTS"]["selected_quality_b1280"]
checkpoint = Path(
    os.environ.get(
        "G1_RQ15_FIRST_STAGE_CHECKPOINT",
        str(
            Path(base.base_path)
            / "logs"
            / source_candidate.run_name
            / RQ15_SOURCE_CHECKPOINT_NAME
        ),
    )
)

history_transformer = replace(base.transformer, attention_window=54)
retrieval_decoder = replace(
    base.transformer,
    num_layers=1,
    ffn="swiglu",
    ffn_intermediate_dim=128,
    attention_window=None,
)
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
    "transformer": history_transformer,
    "retrieval_decoder": retrieval_decoder,
    "max_seq_len": 128,
    "window": "bounded_prefix",
    "prefix_length_rule": "truncated",
    "prefix_cap": 1,
    "query_architecture": "decoder_decoder",
    "query_slots_shared": candidate.query_slots_shared,
    "include_history_memory": candidate.include_history_memory,
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
experiment = MuTransferRq14PretrainedCrossAttentionGenerationExperiment(
    **common,
    mup_base_dim=16,
    mup_delta_dim=32,
    first_stage_checkpoint=checkpoint,
    first_stage_checkpoint_metadata=source_checkpoint_metadata(source_candidate),
)
