from __future__ import annotations

from dataclasses import fields, replace
import os
from pathlib import Path
import runpy

from dcn.config import (
    GenerationExperiment,
    MuTransferCrossAttentionGenerationExperiment,
)
from dcn.config.settings import LrScheduleConfig
from experiments.g1_sasrec_item_ids_likes.analysis.rq13_rq14_query_candidates import (
    candidate_by_run,
)


candidate = candidate_by_run(os.environ["G1_QUERY_RUN"])
os.environ["G1_DATASET_SIZE"] = candidate.dataset_size
os.environ["G1_MAX_EPOCHS"] = str(candidate.horizon_epochs)
os.environ["G1_VARIANT"] = "selected_quality_b1280"
namespace = runpy.run_path(str(Path(__file__).with_name("variant.py")))
base = namespace["VARIANTS"]["selected_quality_b1280"]

rq13 = candidate.study == "rq13"
if candidate.treatment == "one_example":
    prefix_length_rule = "truncated"
    prefix_cap = 1
elif candidate.treatment.startswith("truncated_"):
    prefix_length_rule = "truncated"
    prefix_cap = int(candidate.treatment.rsplit("_", 1)[1])
elif candidate.treatment.startswith("required_"):
    prefix_length_rule = "required"
    prefix_cap = int(candidate.treatment.rsplit("_", 1)[1])
elif candidate.treatment.startswith("selected_cap_"):
    prefix_length_rule = "truncated"
    prefix_cap = int(candidate.treatment.rsplit("_", 1)[1])
else:
    prefix_length_rule = "truncated"
    prefix_cap = 1

query_slots_shared = candidate.treatment.startswith("shared_")
include_history_memory = candidate.treatment.endswith("_history")
history_transformer = replace(
    base.transformer,
    attention_window=None if rq13 else 54,
)
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
    "prefix_length_rule": prefix_length_rule,
    "prefix_cap": prefix_cap,
    "query_architecture": "encoder_decoder" if rq13 else "decoder_decoder",
    "query_slots_shared": query_slots_shared,
    "include_history_memory": include_history_memory,
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
experiment = MuTransferCrossAttentionGenerationExperiment(
    **common,
    mup_base_dim=16,
    mup_delta_dim=32,
)
