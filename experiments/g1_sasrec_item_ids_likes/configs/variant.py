"""1.1: one transformer variant, named by the G1_VARIANT environment variable.

One factor at a time around the baseline. The axes are not independent, so each
is answered before the useful changes are crossed.

Run the lot with ``launchers/core/sweep.sh``; a single variant with

    G1_VARIANT=dim_128 python -m dcn.main -s experiments/g1_sasrec_item_ids_likes/configs/variant.py

``G1_SEED`` re-runs one under a different seed, into its own run name.
"""

import os
from dataclasses import replace
from typing import cast, get_args

import torch

from dcn.config import GenerationExperiment, MuTransferGenerationExperiment
from dcn.config.generation import NegativeSampling
from dcn.models import ClsTokenMode
from dcn.nn.sampled_softmax import Correction
from dcn.config.settings import DataloaderConfig, LrScheduleConfig, RuntimeConfig
from dcn.datasets.yambda import UserSample, YambdaSize
from experiments.generation_protocol import (
    FINAL_YAMBDA_SIZE,
    generation_protocol,
)
from dcn.training_metadata import GENERATION_TRAINING_SEMANTICS_REVISION

_BASELINE_TRANSFORMER = replace(
    GenerationExperiment.transformer,
    dim=64,
    nhead=2,
    num_kv_heads=2,
    ffn_intermediate_dim=256,
    ffn="gelu",
    norm="layer",
    input_norm=None,
    final_norm="layer",
    alibi=False,
    learned_positions="forward",
    input_dropout=0.1,
    ffn_dropout=0.1,
)
COSINE_WARMUP = LrScheduleConfig("cosine", warmup_fraction=0.05)
_DATASET_SIZE = cast(YambdaSize, os.environ.get("G1_DATASET_SIZE", FINAL_YAMBDA_SIZE))
if _DATASET_SIZE not in get_args(YambdaSize):
    raise ValueError(
        f"G1_DATASET_SIZE must be one of {get_args(YambdaSize)}, got {_DATASET_SIZE!r}"
    )
_MAX_USERS = os.environ.get("G1_MAX_USERS")
if _MAX_USERS is not None and (
    not _MAX_USERS.isdigit()
    or int(_MAX_USERS) < 1
    or _MAX_USERS != str(int(_MAX_USERS))
):
    raise ValueError("G1_MAX_USERS must be a canonical positive integer")
_USER_SAMPLE = UserSample(max_users=int(_MAX_USERS)) if _MAX_USERS else None
_VAL_BATCH_OVERRIDE = os.environ.get("G1_VAL_BATCH_SIZE")
if _VAL_BATCH_OVERRIDE is not None and (
    not _VAL_BATCH_OVERRIDE.isdigit()
    or int(_VAL_BATCH_OVERRIDE) < 1
    or _VAL_BATCH_OVERRIDE != str(int(_VAL_BATCH_OVERRIDE))
):
    raise ValueError("G1_VAL_BATCH_SIZE must be a canonical positive integer")
_VAL_BATCH_SIZE = int(_VAL_BATCH_OVERRIDE or 8192)
_TRAIN_BATCH_OVERRIDE = os.environ.get("G1_TRAIN_BATCH_SIZE")
if _TRAIN_BATCH_OVERRIDE is not None and (
    not _TRAIN_BATCH_OVERRIDE.isdigit()
    or int(_TRAIN_BATCH_OVERRIDE) < 1
    or _TRAIN_BATCH_OVERRIDE != str(int(_TRAIN_BATCH_OVERRIDE))
):
    raise ValueError("G1_TRAIN_BATCH_SIZE must be a canonical positive integer")
_TRAIN_BATCH_SIZE = (
    None if _TRAIN_BATCH_OVERRIDE is None else int(_TRAIN_BATCH_OVERRIDE)
)
_MAX_EPOCHS_RAW = os.environ.get("G1_MAX_EPOCHS", "20")
if (
    not _MAX_EPOCHS_RAW.isdigit()
    or int(_MAX_EPOCHS_RAW) < 20
    or _MAX_EPOCHS_RAW != str(int(_MAX_EPOCHS_RAW))
):
    raise ValueError("G1_MAX_EPOCHS must be a canonical integer of at least 20")
_MAX_EPOCHS = int(_MAX_EPOCHS_RAW)
_DATA_PROTOCOL = generation_protocol(
    event_type_filter="like",
    window="next_item",
    size=_DATASET_SIZE,
    user_sample=_USER_SAMPLE,
)


def _run_name(name: str) -> str:
    cap = f"_cap{_MAX_EPOCHS}" if _MAX_EPOCHS != 20 else ""
    sample = f"_{_USER_SAMPLE.name}" if _USER_SAMPLE is not None else ""
    validation = f"_val{_VAL_BATCH_SIZE}" if _VAL_BATCH_OVERRIDE else ""
    semantics = GENERATION_TRAINING_SEMANTICS_REVISION
    return f"g1_calibrated_{name}{cap}_ts{semantics}_{_DATASET_SIZE}{sample}{validation}"


def _variant(
    name: str,
    *,
    max_seq_len: int = 100,
    bos: bool = False,
    cls_token: bool = False,
    cls_token_mode: ClsTokenMode = "none",
    lr_schedule: LrScheduleConfig | None = None,
    timestamp_delta: str | None = None,
    timestamp_combination: str = "add",
    timestamp_num_bins: int = 32,
    per_layer_item_embeddings: bool = False,
    negative_sampling: NegativeSampling = "offline_logq",
    logq_correction: Correction = "yi2019",
    correct_positive_logq: bool = False,
    mask_false_negatives: bool = False,
    exclude_own_group_negatives: bool = False,
    dense_random_negative_scores: bool = False,
    batch_size: int = 128,
    embedding_learning_rate: float = 1e-3,
    deep_learning_rate: float = 1e-3,
    **transformer,
):
    return GenerationExperiment(
        run_name=_run_name(name),
        **_DATA_PROTOCOL,
        dataloader=DataloaderConfig(
            batch_size=_TRAIN_BATCH_SIZE or batch_size,
            val_batch_size=_VAL_BATCH_SIZE,
            num_workers=4,
            prefetch_factor=4,
        ),
        num_epochs=_MAX_EPOCHS,
        lr_schedule_horizon_epochs=20,
        eval_every_n_epochs=1,
        restore_best_weights=True,
        early_stopping_patience=3,
        early_stopping_min_delta=0.0,
        transformer=replace(_BASELINE_TRANSFORMER, **transformer),
        max_seq_len=max_seq_len,
        bos=bos,
        cls_token=cls_token,
        cls_token_mode=cls_token_mode,
        lr_schedule=lr_schedule,
        timestamp_delta=timestamp_delta,
        timestamp_combination=timestamp_combination,
        timestamp_num_bins=timestamp_num_bins,
        per_layer_item_embeddings=per_layer_item_embeddings,
        negative_sampling=negative_sampling,
        logq_correction=logq_correction,
        dense_random_negative_scores=dense_random_negative_scores,
        correct_positive_logq=correct_positive_logq,
        mask_false_negatives=mask_false_negatives,
        exclude_own_group_negatives=exclude_own_group_negatives,
        initializer_std=0.02,
        embedding_learning_rate=embedding_learning_rate,
        deep_learning_rate=deep_learning_rate,
        weight_decay=0.0,
        runtime=RuntimeConfig(
            dtype=torch.bfloat16,
            compile=False,
            gradient_clip_norm=None,
        ),
    )


def _mup_variant(
    name: str, *, dim: int, deep_learning_rate: float
) -> MuTransferGenerationExperiment:
    return MuTransferGenerationExperiment(
        run_name=_run_name(name),
        **_DATA_PROTOCOL,
        dataloader=DataloaderConfig(
            batch_size=_TRAIN_BATCH_SIZE or 128,
            val_batch_size=_VAL_BATCH_SIZE,
            num_workers=4,
            prefetch_factor=4,
        ),
        num_epochs=_MAX_EPOCHS,
        lr_schedule_horizon_epochs=20,
        eval_every_n_epochs=1,
        restore_best_weights=True,
        early_stopping_patience=3,
        early_stopping_min_delta=0.0,
        transformer=replace(
            _BASELINE_TRANSFORMER,
            dim=dim,
            ffn_intermediate_dim=4 * dim,
        ),
        lr_schedule=COSINE_WARMUP,
        negative_sampling="offline_logq",
        correct_positive_logq=False,
        mask_false_negatives=False,
        exclude_own_group_negatives=False,
        initializer_std=0.02,
        embedding_learning_rate=1e-3,
        deep_learning_rate=deep_learning_rate,
        weight_decay=0.0,
        runtime=RuntimeConfig(
            dtype=torch.bfloat16,
            compile=False,
            gradient_clip_norm=None,
        ),
    )


_BASELINE = _variant("baseline")
VARIANTS = {
    "baseline": _BASELINE,
    "homework_reproduction": replace(
        _BASELINE,
        run_name=_run_name("homework_reproduction"),
    ),
    "homework_fixed_leave_one_out": replace(
        _BASELINE,
        run_name=_run_name("homework_baseline"),
        logq_correction="baseline",
    ),
    **{
        f"neg_{name}": _variant(
            f"neg_{name}",
            negative_sampling=negative_sampling,
        )
        for name, negative_sampling in (
            ("online_logq", "online_logq"),
            ("random", "random"),
            ("random_offline_logq", "random_offline_logq"),
            ("in_batch_no_logq", "in_batch_no_logq"),
            ("mixed_online_logq", "mixed_online_logq"),
            ("mixed_offline_logq", "mixed_offline_logq"),
        )
    },
    "neg_fixed_inbatch_global_q_yi2019": _variant(
        "neg_fixed_inbatch_global_q_yi2019",
        negative_sampling="offline_logq",
        logq_correction="yi2019",
        correct_positive_logq=True,
    ),
    "neg_fixed_inbatch_leave_one_out": _variant(
        "neg_fixed_inbatch_leave_one_out",
        negative_sampling="offline_logq",
        logq_correction="baseline",
    ),
    "neg_streaming_inbatch_global_q_yi2019": _variant(
        "neg_streaming_inbatch_global_q_yi2019",
        negative_sampling="online_logq",
        logq_correction="yi2019",
        correct_positive_logq=True,
    ),
    "neg_popularity_random_global_q_yi2019": _variant(
        "neg_popularity_random_global_q_yi2019",
        negative_sampling="random_offline_logq",
        logq_correction="yi2019",
        correct_positive_logq=True,
    ),
    "neg_mixed_streaming_logq_negative_only": _variant(
        "neg_mixed_streaming_logq_negative_only",
        negative_sampling="mixed_online_logq",
        logq_correction="yi2019",
    ),
    "neg_mixed_fixed_logq_negative_only": _variant(
        "neg_mixed_fixed_logq_negative_only",
        negative_sampling="mixed_offline_logq",
        logq_correction="yi2019",
    ),
    "neg_aggregate_uniform_streaming_global_q_yi2019": _variant(
        "neg_aggregate_uniform_streaming_global_q_yi2019",
        negative_sampling="mixed_online_global_q",
        logq_correction="yi2019",
        correct_positive_logq=True,
    ),
    "neg_aggregate_uniform_streaming_global_q_negative_only": _variant(
        "neg_aggregate_uniform_streaming_global_q_negative_only",
        negative_sampling="mixed_online_global_q_negative_only",
        logq_correction="yi2019",
    ),
    **{
        f"mup_dim{dim}_lr{rate_name}": _mup_variant(
            f"mup_dim{dim}_lr{rate_name}", dim=dim, deep_learning_rate=rate
        )
        for dim in (32, 128)
        for rate_name, rate in (
            ("5e4", 5e-4),
            ("1e3", 1e-3),
            ("2e3", 2e-3),
            ("3e3", 3e-3),
            ("5e3", 5e-3),
            ("1e2", 1e-2),
            ("2e2", 2e-2),
            ("3e2", 3e-2),
            ("5e2", 5e-2),
            ("1e1", 1e-1),
            ("2e1", 2e-1),
        )
    },
    "dim_16": _variant("dim_16", dim=16, ffn_intermediate_dim=64),
    "dim_32": _variant("dim_32", dim=32, ffn_intermediate_dim=128),
    "dim_128": _variant("dim_128", dim=128, ffn_intermediate_dim=512),
    "dim_256": _variant("dim_256", dim=256, ffn_intermediate_dim=1024),
    "depth_1": _variant("depth_1", num_layers=1),
    "depth_4": _variant("depth_4", num_layers=4),
    "dropout_30": _variant(
        "dropout_30", dropout=0.3, input_dropout=0.3, ffn_dropout=0.3
    ),
    "dropout_50": _variant(
        "dropout_50", dropout=0.5, input_dropout=0.5, ffn_dropout=0.5
    ),
    "dropout_0": _variant("dropout_0", dropout=0.0, input_dropout=0.0, ffn_dropout=0.0),
    "dropout_5": _variant(
        "dropout_5", dropout=0.05, input_dropout=0.05, ffn_dropout=0.05
    ),
    "dropout_20": _variant(
        "dropout_20", dropout=0.2, input_dropout=0.2, ffn_dropout=0.2
    ),
    "heads_1": _variant("heads_1", nhead=1, num_kv_heads=1),
    "heads_4": _variant("heads_4", nhead=4, num_kv_heads=4),
    "heads_8": _variant("heads_8", nhead=8, num_kv_heads=8),
    "heads_gqa": _variant("heads_gqa", nhead=2, num_kv_heads=1),
    "ffn_128": _variant("ffn_128", ffn_intermediate_dim=128),
    "ffn_512": _variant("ffn_512", ffn_intermediate_dim=512),
    "seq_12": _variant("seq_12", max_seq_len=12),
    "seq_25": _variant("seq_25", max_seq_len=25),
    "seq_50": _variant("seq_50", max_seq_len=50),
    "seq_128": _variant("seq_128", max_seq_len=128),
    "seq_200": _variant("seq_200", max_seq_len=200),
    "seq_256": _variant("seq_256", max_seq_len=256),
    "seq_512": _variant("seq_512", max_seq_len=512),
    "ffn_swiglu": _variant("ffn_swiglu", ffn="swiglu"),
    "ffn_relu": _variant("ffn_relu", ffn="relu"),
    "ffn_gelu": _variant("ffn_gelu", ffn="gelu"),
    "ffn_silu": _variant("ffn_silu", ffn="silu"),
    "ffn_reglu": _variant(
        "ffn_reglu", ffn="reglu", gated_ffn_dropout=True
    ),
    "ffn_geglu": _variant(
        "ffn_geglu", ffn="geglu", gated_ffn_dropout=True
    ),
    "ffn_swiglu_dropout": _variant(
        "ffn_swiglu_dropout", ffn="swiglu", gated_ffn_dropout=True
    ),
    "ffn_swiglu_matched": _variant(
        "ffn_swiglu_matched", ffn="swiglu", ffn_intermediate_dim=171
    ),
    "pos_none": _variant("pos_none", alibi=False, learned_positions=None),
    "pos_alibi": _variant("pos_alibi", alibi=True, learned_positions=None),
    "pos_learned_reverse": _variant(
        "pos_learned_reverse", alibi=False, learned_positions="reverse"
    ),
    "pos_learned_forward_reverse": _variant(
        "pos_learned_forward_reverse",
        alibi=False,
        learned_positions=("forward", "reverse"),
    ),
    "pos_learned_forward_concat": _variant(
        "pos_learned_forward_concat",
        alibi=False,
        learned_positions="forward",
        learned_position_fusion="concat",
    ),
    "pos_learned_forward_reverse_concat": _variant(
        "pos_learned_forward_reverse_concat",
        alibi=False,
        learned_positions=("forward", "reverse"),
        learned_position_fusion="concat",
    ),
    "pos_learned_forward_concat_alibi": _variant(
        "pos_learned_forward_concat_alibi",
        alibi=True,
        learned_positions="forward",
        learned_position_fusion="concat",
    ),
    "pos_learned_forward_reverse_concat_alibi": _variant(
        "pos_learned_forward_reverse_concat_alibi",
        alibi=True,
        learned_positions=("forward", "reverse"),
        learned_position_fusion="concat",
    ),
    "pos_rope": _variant(
        "pos_rope", alibi=False, learned_positions=None, rope="forward"
    ),
    "pos_rope_base100": _variant(
        "pos_rope_base100",
        alibi=False,
        learned_positions=None,
        rope="forward",
        rope_base=100.0,
    ),
    "pos_rope_base1000": _variant(
        "pos_rope_base1000",
        alibi=False,
        learned_positions=None,
        rope="forward",
        rope_base=1000.0,
    ),
    "pos_rope_reverse": _variant(
        "pos_rope_reverse", alibi=False, learned_positions=None, rope="reverse"
    ),
    "pos_rope_alibi": _variant(
        "pos_rope_alibi", alibi=True, learned_positions=None, rope="forward"
    ),
    "pos_rope_learned": _variant("pos_rope_learned", alibi=False, rope="forward"),
    "pos_rope_learned_reverse": _variant(
        "pos_rope_learned_reverse",
        alibi=False,
        rope="forward",
        learned_positions="reverse",
    ),
    "pos_rope_reverse_learned_reverse": _variant(
        "pos_rope_reverse_learned_reverse",
        alibi=False,
        rope="reverse",
        learned_positions="reverse",
    ),
    "pos_rope_reverse_learned": _variant(
        "pos_rope_reverse_learned",
        alibi=False,
        rope="reverse",
        learned_positions="forward",
    ),
    "pos_rope_reverse_alibi": _variant(
        "pos_rope_reverse_alibi",
        alibi=True,
        rope="reverse",
        learned_positions=None,
    ),
    "pos_learned_alibi": _variant(
        "pos_learned_alibi",
        alibi=True,
        learned_positions="forward",
    ),
    "pos_learned_reverse_alibi": _variant(
        "pos_learned_reverse_alibi",
        alibi=True,
        learned_positions="reverse",
    ),
    "pos_rope_learned_reverse_alibi": _variant(
        "pos_rope_learned_reverse_alibi",
        alibi=True,
        rope="forward",
        learned_positions="reverse",
    ),
    "pos_rope_reverse_learned_alibi": _variant(
        "pos_rope_reverse_learned_alibi",
        alibi=True,
        rope="reverse",
        learned_positions="forward",
    ),
    "pos_all": _variant("pos_all", alibi=True, rope="forward"),
    "pos_reverse_all": _variant(
        "pos_reverse_all",
        alibi=True,
        rope="reverse",
        learned_positions="reverse",
    ),
    "norm_rms": _variant("norm_rms", norm="rms"),
    "norm_batch": _variant("norm_batch", norm="batch"),
    "norm_all_rms": _variant("norm_all_rms", input_norm="rms", final_norm="rms"),
    "norm_input_layer": _variant("norm_input_layer", input_norm="layer"),
    "norm_input_rms": _variant("norm_input_rms", input_norm="rms"),
    "norm_no_final": _variant("norm_no_final", final_norm=None),
    "norm_post": _variant("norm_post", norm_place="post"),
    "bos": _variant("bos", bos=True),
    "cls": _variant("cls", cls_token=True),
    "cls_interleaved": _variant(
        "cls_interleaved", cls_token_mode="interleaved"
    ),
    # rq5 and rq6. The optimizer ran at a fixed rate until now, so "which
    # schedule" and "does warmup help" are one grid read two ways.
    "lr_cosine": _variant("lr_cosine", lr_schedule=LrScheduleConfig("cosine")),
    "lr_linear": _variant("lr_linear", lr_schedule=LrScheduleConfig("linear")),
    "lr_step": _variant("lr_step", lr_schedule=LrScheduleConfig("step")),
    "lr_exponential": _variant(
        "lr_exponential", lr_schedule=LrScheduleConfig("exponential")
    ),
    "lr_polynomial": _variant(
        "lr_polynomial", lr_schedule=LrScheduleConfig("polynomial")
    ),
    "lr_wsd": _variant(
        "lr_wsd",
        lr_schedule=LrScheduleConfig("warmup_stable_decay", warmup_fraction=0.05),
    ),
    # The two inverse-sqrt rows decay on the same 5%-of-run timescale and differ
    # only in the ramp.
    "lr_inverse_sqrt": _variant(
        "lr_inverse_sqrt",
        lr_schedule=LrScheduleConfig("inverse_sqrt", timescale_fraction=0.05),
    ),
    "lr_warmup": _variant(
        "lr_warmup", lr_schedule=LrScheduleConfig(warmup_fraction=0.05)
    ),
    "lr_cosine_warmup": _variant(
        "lr_cosine_warmup",
        lr_schedule=COSINE_WARMUP,
    ),
    "lr_cosine_cycles2": _variant(
        "lr_cosine_cycles2",
        lr_schedule=LrScheduleConfig("cosine", warmup_fraction=0.05, cycles=2),
    ),
    "lr_cosine_cycles4": _variant(
        "lr_cosine_cycles4",
        lr_schedule=LrScheduleConfig("cosine", warmup_fraction=0.05, cycles=4),
    ),
    # Its usual form: inverse-sqrt is defined to peak at the end of a warmup,
    # and without one it decays from the first step.
    "lr_inverse_sqrt_warmup": _variant(
        "lr_inverse_sqrt_warmup",
        lr_schedule=LrScheduleConfig("inverse_sqrt", warmup_fraction=0.05),
    ),
    "embedding_lr_5e4": _variant(
        "embedding_lr_5e4",
        embedding_learning_rate=5e-4,
        lr_schedule=COSINE_WARMUP,
    ),
    "embedding_lr_1e4": _variant(
        "embedding_lr_1e4",
        embedding_learning_rate=1e-4,
        lr_schedule=COSINE_WARMUP,
    ),
    "embedding_lr_2e4": _variant(
        "embedding_lr_2e4",
        embedding_learning_rate=2e-4,
        lr_schedule=COSINE_WARMUP,
    ),
    "embedding_lr_2e3": _variant(
        "embedding_lr_2e3",
        embedding_learning_rate=2e-3,
        lr_schedule=COSINE_WARMUP,
    ),
    "embedding_lr_3e3": _variant(
        "embedding_lr_3e3",
        embedding_learning_rate=3e-3,
        lr_schedule=COSINE_WARMUP,
    ),
    "embedding_lr_5e3": _variant(
        "embedding_lr_5e3",
        embedding_learning_rate=5e-3,
        lr_schedule=COSINE_WARMUP,
    ),
    "deep_lr_5e4": _variant(
        "deep_lr_5e4", deep_learning_rate=5e-4, lr_schedule=COSINE_WARMUP
    ),
    "deep_lr_2e3": _variant(
        "deep_lr_2e3", deep_learning_rate=2e-3, lr_schedule=COSINE_WARMUP
    ),
    "deep_lr_3e3": _variant(
        "deep_lr_3e3", deep_learning_rate=3e-3, lr_schedule=COSINE_WARMUP
    ),
    "deep_lr_5e3": _variant(
        "deep_lr_5e3", deep_learning_rate=5e-3, lr_schedule=COSINE_WARMUP
    ),
    # rq2 and rq3. The three axes that cleared the noise floor, crossed.
    "combo_dim32_lr": _variant(
        "combo_dim32_lr",
        dim=32,
        ffn_intermediate_dim=128,
        lr_schedule=COSINE_WARMUP,
    ),
    "combo_all": _variant(
        "combo_all",
        dim=32,
        ffn_intermediate_dim=128,
        max_seq_len=200,
        lr_schedule=COSINE_WARMUP,
    ),
    "combo_lr_rates": _variant(
        "combo_lr_rates",
        embedding_learning_rate=2e-3,
        deep_learning_rate=5e-4,
        lr_schedule=COSINE_WARMUP,
    ),
    "combo_embedding_time": _variant(
        "combo_embedding_time",
        embedding_learning_rate=2e-3,
        timestamp_delta="bins",
        rope="timestamp_log",
        lr_schedule=COSINE_WARMUP,
    ),
    "combo_embedding_position": _variant(
        "combo_embedding_position",
        embedding_learning_rate=2e-3,
        rope="reverse",
        learned_positions="reverse",
        lr_schedule=COSINE_WARMUP,
    ),
    "selected_quality": _variant(
        "selected_quality",
        max_seq_len=128,
        lr_schedule=LrScheduleConfig("linear"),
        timestamp_delta="bins",
        timestamp_num_bins=16,
        negative_sampling="random",
        dense_random_negative_scores=True,
        deep_learning_rate=3e-3,
        ffn="swiglu",
        ffn_intermediate_dim=171,
        nhead=2,
        num_kv_heads=1,
        attention_window=50,
    ),
    "selected_balanced": _variant(
        "selected_balanced",
        max_seq_len=128,
        lr_schedule=LrScheduleConfig("linear"),
        timestamp_delta="bins",
        timestamp_num_bins=16,
        deep_learning_rate=3e-3,
        ffn="swiglu",
        ffn_intermediate_dim=171,
        nhead=2,
        num_kv_heads=1,
        attention_window=50,
    ),
    "window_10": _variant(
        "window_10", attention_window=10, lr_schedule=COSINE_WARMUP
    ),
    "window_none": _variant(
        "window_none", attention_window=None, lr_schedule=COSINE_WARMUP
    ),
    "window_25": _variant("window_25", attention_window=25, lr_schedule=COSINE_WARMUP),
    "window_50": _variant("window_50", attention_window=50, lr_schedule=COSINE_WARMUP),
    "window_75": _variant(
        "window_75", attention_window=75, lr_schedule=COSINE_WARMUP
    ),
    "window_100": _variant(
        "window_100", attention_window=100, lr_schedule=COSINE_WARMUP
    ),
    "per_layer_embeddings": _variant(
        "per_layer_embeddings",
        per_layer_item_embeddings=True,
        lr_schedule=COSINE_WARMUP,
    ),
    "time_rope": _variant("time_rope", rope="timestamp", lr_schedule=COSINE_WARMUP),
    "time_rope_reverse": _variant(
        "time_rope_reverse", rope="timestamp_reverse", lr_schedule=COSINE_WARMUP
    ),
    "time_log_rope": _variant(
        "time_log_rope", rope="timestamp_log", lr_schedule=COSINE_WARMUP
    ),
    "time_log_rope_reverse": _variant(
        "time_log_rope_reverse",
        rope="timestamp_log_reverse",
        lr_schedule=COSINE_WARMUP,
    ),
    "time_plain_add": _variant(
        "time_plain_add", timestamp_delta="plain", lr_schedule=COSINE_WARMUP
    ),
    "time_log_add": _variant(
        "time_log_add", timestamp_delta="log", lr_schedule=COSINE_WARMUP
    ),
    "time_bins_add": _variant(
        "time_bins_add", timestamp_delta="bins", lr_schedule=COSINE_WARMUP
    ),
    "time_bins_8": _variant(
        "time_bins_8",
        timestamp_delta="bins",
        timestamp_num_bins=8,
        lr_schedule=COSINE_WARMUP,
    ),
    "time_bins_16": _variant(
        "time_bins_16",
        timestamp_delta="bins",
        timestamp_num_bins=16,
        lr_schedule=COSINE_WARMUP,
    ),
    "time_bins_64": _variant(
        "time_bins_64",
        timestamp_delta="bins",
        timestamp_num_bins=64,
        lr_schedule=COSINE_WARMUP,
    ),
    "time_bins_reverse_rope": _variant(
        "time_bins_reverse_rope",
        timestamp_delta="bins",
        rope="timestamp_reverse",
        lr_schedule=COSINE_WARMUP,
    ),
    "time_log_concat": _variant(
        "time_log_concat",
        timestamp_delta="log",
        timestamp_combination="concat",
        lr_schedule=COSINE_WARMUP,
    ),
    "time_bins_concat": _variant(
        "time_bins_concat",
        timestamp_delta="bins",
        timestamp_combination="concat",
        lr_schedule=COSINE_WARMUP,
    ),
    "time_bins_log_rope": _variant(
        "time_bins_log_rope",
        timestamp_delta="bins",
        rope="timestamp_log",
        lr_schedule=COSINE_WARMUP,
    ),
}

VARIANTS["selected_quality_b1024"] = replace(
    VARIANTS["selected_quality"],
    run_name=_run_name("selected_quality_b1024"),
    dataloader=replace(VARIANTS["selected_quality"].dataloader, batch_size=1024),
)

VARIANTS["selected_quality_b1280"] = replace(
    VARIANTS["selected_quality"],
    run_name=_run_name("selected_quality_b1280"),
    dataloader=replace(VARIANTS["selected_quality"].dataloader, batch_size=1280),
)

VARIANTS["homework_baseline_native500_r3"] = replace(
    VARIANTS["homework_fixed_leave_one_out"],
    run_name=_run_name("homework_baseline_native500_r3"),
    dataloader=replace(
        VARIANTS["homework_fixed_leave_one_out"].dataloader,
        batch_size=1280,
    ),
    embedding_learning_rate=0.001,
    deep_learning_rate=0.002,
)

VARIANTS["selected_quality_b512"] = replace(
    VARIANTS["selected_quality"],
    run_name=_run_name("selected_quality_b512"),
    dataloader=replace(VARIANTS["selected_quality"].dataloader, batch_size=512),
)

VARIANTS["future_baseline"] = replace(
    VARIANTS["selected_quality_b512"],
    run_name=_run_name("future_baseline"),
    transformer=replace(
        VARIANTS["selected_quality_b512"].transformer,
        input_norm="rms",
    ),
    embedding_learning_rate=32e-3,
    deep_learning_rate=12e-3,
)

for embedding_name, embedding_rate in (("e4e3", 4e-3), ("e8e3", 8e-3)):
    for deep_name, deep_rate in (("d6e3", 6e-3), ("d12e3", 12e-3)):
        name = f"selected_quality_b512_{embedding_name}_{deep_name}"
        VARIANTS[name] = replace(
            VARIANTS["selected_quality_b512"],
            run_name=_run_name(name),
            embedding_learning_rate=embedding_rate,
            deep_learning_rate=deep_rate,
        )

for embedding_name, embedding_rate in (
    ("e5e4", 5e-4),
    ("e1e3", 1e-3),
    ("e2e3", 2e-3),
):
    for deep_name, deep_rate in (
        ("d15e4", 1.5e-3),
        ("d3e3", 3e-3),
        ("d6e3", 6e-3),
    ):
        name = f"selected_quality_b1024_{embedding_name}_{deep_name}"
        VARIANTS[name] = replace(
            VARIANTS["selected_quality_b1024"],
            run_name=_run_name(name),
            embedding_learning_rate=embedding_rate,
            deep_learning_rate=deep_rate,
        )

for embedding_name, embedding_rate in (
    ("e4e3", 4e-3),
    ("e8e3", 8e-3),
    ("e12e3", 12e-3),
):
    for deep_name, deep_rate in (
        ("d12e3", 12e-3),
        ("d24e3", 24e-3),
        ("d36e3", 36e-3),
    ):
        name = f"selected_quality_b1024_{embedding_name}_{deep_name}"
        VARIANTS[name] = replace(
            VARIANTS["selected_quality_b1024"],
            run_name=_run_name(name),
            embedding_learning_rate=embedding_rate,
            deep_learning_rate=deep_rate,
        )

ARCHITECTURE_VARIANTS = (
    "dim_16",
    "dim_32",
    "dim_128",
    "dim_256",
    "depth_1",
    "depth_4",
    "dropout_0",
    "dropout_5",
    "dropout_20",
    "dropout_30",
    "dropout_50",
    "heads_1",
    "heads_4",
    "heads_8",
    "heads_gqa",
    "ffn_128",
    "ffn_512",
    "seq_50",
    "seq_128",
    "seq_200",
    "seq_256",
    "seq_512",
    "ffn_swiglu",
    "ffn_swiglu_matched",
    "pos_none",
    "pos_alibi",
    "pos_learned_reverse",
    "pos_learned_forward_reverse",
    "pos_rope",
    "pos_rope_reverse",
    "pos_rope_alibi",
    "pos_rope_learned",
    "pos_rope_learned_reverse",
    "pos_rope_reverse_learned_reverse",
    "pos_rope_reverse_learned",
    "pos_rope_reverse_alibi",
    "pos_learned_alibi",
    "pos_learned_reverse_alibi",
    "pos_rope_learned_reverse_alibi",
    "pos_rope_reverse_learned_alibi",
    "pos_all",
    "pos_reverse_all",
    "norm_rms",
    "norm_batch",
    "norm_all_rms",
    "norm_input_layer",
    "norm_input_rms",
    "norm_no_final",
    "norm_post",
    "bos",
    "cls",
    "window_none",
    "window_10",
    "window_25",
    "window_50",
    "window_75",
    "window_100",
)

VARIANTS.update(
    {
        f"cosine_{name}": replace(
            VARIANTS[name],
            run_name=_run_name(f"cosine_{name}"),
            lr_schedule=COSINE_WARMUP,
        )
        for name in ARCHITECTURE_VARIANTS
    }
)

experiment = VARIANTS[os.environ["G1_VARIANT"]]

if seed := os.environ.get("G1_SEED"):
    experiment = replace(
        experiment, seed=int(seed), run_name=f"{experiment.run_name}_s{seed}"
    )
