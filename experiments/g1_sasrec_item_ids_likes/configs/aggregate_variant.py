from __future__ import annotations

from dataclasses import fields, replace
import os
from pathlib import Path
import runpy

from dcn.config import GenerationExperiment, MuTransferGenerationExperiment
from dcn.config.settings import LrScheduleConfig
from experiments.g1_sasrec_item_ids_likes.analysis.aggregate_candidates import (
    AggregateCandidate,
    Member,
    candidate_by_run,
)


candidate = candidate_by_run(os.environ["G1_AGGREGATE_RUN"])
os.environ["G1_DATASET_SIZE"] = "500m"
os.environ["G1_MAX_EPOCHS"] = "20"
os.environ["G1_VARIANT"] = "baseline"
namespace = runpy.run_path(str(Path(__file__).with_name("variant.py")))
base = namespace["VARIANTS"]["baseline"]


def _with_member(
    experiment: GenerationExperiment, member: Member
) -> GenerationExperiment:
    transformer = experiment.transformer
    updates: dict[str, object] = {}
    if member == "swiglu":
        transformer = replace(
            transformer,
            ffn="swiglu",
            ffn_intermediate_dim=192,
            gated_ffn_dropout=True,
            ffn_dropout=0.1,
        )
    elif member == "scheduler":
        updates["lr_schedule"] = LrScheduleConfig(
            "cosine",
            warmup_fraction=0.05,
            cycles=1,
            optimizer_group_scope="deep_only",
        )
    elif member == "position":
        transformer = replace(
            transformer,
            alibi=True,
            learned_positions=("forward", "reverse"),
            learned_position_fusion="concat",
            learned_position_fusion_residual="rezero",
            learned_position_reverse_correction="bounded_tanh",
            learned_position_reverse_max_scale=0.025,
            learned_position_reverse_initializer_rng_nonadvancing=True,
        )
    elif member == "post_norm":
        transformer = replace(transformer, norm_place="post")
    elif member == "input_final_rms":
        transformer = replace(transformer, input_norm="rms", final_norm="rms")
    elif member == "cls":
        updates["cls_token_mode"] = "end_only"
    elif member == "time":
        transformer = replace(transformer, rope="timestamp_reverse")
        updates.update(
            timestamp_delta="bins",
            timestamp_combination="add",
            timestamp_num_bins=32,
        )
    elif member == "popularity":
        updates.update(
            negative_sampling="random_offline_logq",
            logq_correction="yi2019",
            correct_positive_logq=True,
            num_in_batch_negatives=2048,
            dense_random_negative_scores=True,
        )
    elif member == "gqa":
        transformer = replace(transformer, num_kv_heads=1)
    elif member == "bos":
        updates["bos"] = True
    elif member == "depth":
        transformer = replace(transformer, num_layers=candidate.num_layers)
    else:
        raise ValueError(f"unknown aggregate member {member!r}")
    return replace(experiment, transformer=transformer, **updates)


baseline = replace(
    base,
    run_name=candidate.run_name,
    seed=42,
    num_epochs=candidate.num_epochs,
    lr_schedule_horizon_epochs=candidate.horizon_epochs,
    lr_schedule=LrScheduleConfig(),
    adaptive_schedule_early_stopping=False,
    eval_every_n_epochs=1,
    restore_best_weights=True,
    early_stopping_patience=3,
    early_stopping_min_delta=0.0,
    dataloader=replace(
        base.dataloader,
        batch_size=1280,
        gradient_accumulation_steps=1,
    ),
    embedding_learning_rate=candidate.embedding_lr,
    deep_learning_rate=candidate.deep_lr,
    max_seq_len=100,
    bos=False,
    cls_token=False,
    cls_token_mode="none",
    timestamp_delta=None,
    timestamp_combination="add",
    timestamp_num_bins=32,
    negative_sampling="offline_logq",
    logq_correction="baseline",
    correct_positive_logq=False,
    mask_false_negatives=False,
    exclude_own_group_negatives=False,
    num_in_batch_negatives=512,
    dense_random_negative_scores=False,
    transformer=replace(
        base.transformer,
        dim=64,
        num_layers=2,
        nhead=2,
        num_kv_heads=2,
        ffn="gelu",
        ffn_intermediate_dim=256,
        gated_ffn_dropout=False,
        norm="layer",
        norm_place="pre",
        input_norm=None,
        final_norm="layer",
        alibi=False,
        rope=None,
        learned_positions="forward",
        learned_position_fusion="add",
        learned_position_fusion_normalization=None,
        learned_position_fusion_residual=None,
        learned_position_initialization="default",
        learned_position_reverse_correction=None,
        learned_position_reverse_max_scale=0.1,
        learned_position_reverse_initializer_rng_nonadvancing=False,
        attention_window=None,
    ),
)

treatment = baseline
if candidate.family == "bridge":
    assert candidate.member is not None
    treatment = _with_member(treatment, candidate.member)
elif candidate.family == "aggregate":
    for member in (
        "swiglu",
        "scheduler",
        "position",
        "post_norm",
        "input_final_rms",
        "cls",
        "time",
        "popularity",
        "gqa",
        "bos",
    ):
        treatment = _with_member(treatment, member)
    treatment = replace(
        treatment,
        transformer=replace(treatment.transformer, num_layers=candidate.num_layers),
    )

common = {
    field.name: getattr(treatment, field.name)
    for field in fields(GenerationExperiment)
    if field.init
}
common["item_embedding_dim"] = 64
experiment = MuTransferGenerationExperiment(
    **common,
    mup_base_dim=16,
    mup_delta_dim=32,
)
