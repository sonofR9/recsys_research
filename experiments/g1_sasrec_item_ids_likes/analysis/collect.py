"""Generate metadata-backed G1 tuning and reader-report draft tables.

The tuning ledger contains usable completed results. Scratchpad reports contain
the RQ-specific table layer intended for ``README.md`` and are generated only
after the shared ten-baseline empirical bands exist.
"""

import argparse
import json
import math
import re
import statistics
import subprocess
import sys
import textwrap
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from functools import cache
from pathlib import Path

from dcn.training_metadata import (
    GENERATION_TRAINING_SEMANTICS_REVISION,
    TIMESTAMP_BIN_SEMANTICS_REVISION,
    has_current_generation_semantics,
)
from experiments.g1_sasrec_item_ids_likes.analysis import baseline_spread
from experiments.g1_sasrec_item_ids_likes.analysis import native_500m_provenance
from experiments.g1_sasrec_item_ids_likes.analysis import reporting
from experiments.g1_sasrec_item_ids_likes.analysis import rq1_width_transfer
from experiments.g1_sasrec_item_ids_likes.analysis import rq4_activation_depth
from experiments.g1_sasrec_item_ids_likes.analysis import rq5_scheduler_report
from experiments.g1_sasrec_item_ids_likes.analysis import rq8_reinvestigation_report
from experiments.g1_sasrec_item_ids_likes.analysis import rq10_reinvestigation_report
from experiments.g1_sasrec_item_ids_likes.analysis import rq11_mixed_streaming_report
from experiments.g1_sasrec_item_ids_likes.analysis import select_native_500m
from experiments.g1_sasrec_item_ids_likes.launchers import verify_artifact
from utils.report_file_facts import report_file_facts

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
GENERATED = HERE.parents[2] / "generated"
ARCHIVED_50M_QUESTIONS = EXPERIMENT / "archive/50m/research_questions.md"
CURATED_500M_QUESTIONS = EXPERIMENT / "evidence/research_questions_500m.md"
READER_REPORT = EXPERIMENT / "README.md"
BASELINE_SPREAD = EXPERIMENT / "scratchpad/baseline_spread_500m.json"
RQ7_REINVESTIGATION_READER = (
    EXPERIMENT / "scratchpad/rq7_reinvestigation_reader_500m.md"
)
BASELINE = "baseline"
MARKER = "<!-- RESULTS TABLES -->"
QUESTION_MARKER = "<!-- QUESTION TABLES -->"
RUN_PREFIX = "g1_calibrated_"
PROVENANCE_MARKER = f"<!-- run-prefix: {RUN_PREFIX} -->"
HOMEWORK_RECALL_MIN = 0.1235
HOMEWORK_RECALL_MAX = 0.13


@cache
def _current_native_selection() -> dict:
    return select_native_500m.select_native_500m(GENERATED)


def homework_recall_in_calibration_range(recall_at_100: float) -> bool:
    return HOMEWORK_RECALL_MIN <= recall_at_100 <= HOMEWORK_RECALL_MAX


def validate_homework_reproduction_runs(runs: list["ReportRun"]) -> None:
    for run in runs:
        recall = _metric_value(run, "recall@100")
        if recall is None or not homework_recall_in_calibration_range(recall):
            raise ValueError(
                f"{run.name}: homework recall@100 must be in "
                f"[{HOMEWORK_RECALL_MIN}, {HOMEWORK_RECALL_MAX}]"
            )

_ARCHITECTURE_SPECS = [
    (
        "Embedding and model dimension",
        "dim=64",
        ["dim_16", "dim_32", "dim_128", "dim_256"],
    ),
    ("Depth", "depth=2", ["depth_1", "depth_4"]),
    (
        "Number of attention heads",
        "heads=2, kv_heads=2",
        ["heads_1", "heads_4", "heads_8"],
    ),
    ("Grouped-query attention", "MHA: heads=2, kv_heads=2", ["heads_gqa"]),
    ("FFN ratio", "ffn_dim=256 (4x model dim)", ["ffn_128", "ffn_512"]),
    (
        "Sequence length",
        "max_seq_len=100",
        ["seq_50", "seq_128", "seq_200", "seq_256", "seq_512"],
    ),
    (
        "Dropout",
        "dropout=0.1",
        ["dropout_0", "dropout_5", "dropout_20", "dropout_30", "dropout_50"],
    ),
    (
        "Feedforward kind",
        "GELU, ffn_dim=256",
        ["ffn_swiglu", "ffn_swiglu_matched"],
    ),
    (
        "Position encoding",
        "learned forward positions",
        [
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
        ],
    ),
    (
        "Block normalization kind",
        "LayerNorm",
        ["norm_rms", "norm_batch"],
    ),
    (
        "Residual normalization place",
        "pre-norm",
        ["norm_post"],
    ),
    (
        "Input and final normalization",
        "no input norm + final LayerNorm",
        ["norm_all_rms", "norm_input_layer", "norm_input_rms", "norm_no_final"],
    ),
    ("BOS token", "no BOS token", ["bos"]),
]

ARCHITECTURE_BASE_TITLES = {
    title: f"{title} (baseline: {baseline})"
    for title, baseline, _ in _ARCHITECTURE_SPECS
}
ARCHITECTURE_COSINE_TITLES = {
    title: f"{title} under cosine warmup (baseline: {baseline})"
    for title, baseline, _ in _ARCHITECTURE_SPECS
}
ARCHITECTURE_AXES = [
    (ARCHITECTURE_BASE_TITLES[title], names) for title, _, names in _ARCHITECTURE_SPECS
]
COSINE_AXES = [
    (ARCHITECTURE_COSINE_TITLES[title], [f"cosine_{name}" for name in names])
    for title, _, names in _ARCHITECTURE_SPECS
]

EMBEDDING_LR_TITLE = (
    "Embedding learning rate under cosine warmup "
    "(baseline: embedding LR=0.001; deep LR fixed at 0.001)"
)
DEEP_LR_TITLE = (
    "Deep learning rate under cosine warmup "
    "(baseline: deep LR=0.001; embedding LR fixed at 0.001)"
)
FINAL_COMBINATIONS_TITLE = (
    "Final combinations "
    "(reference: cosine warmup, embedding LR=0.001, deep LR=0.001)"
)
SHARED_WINDOW_TITLE = (
    "Shared attention window under cosine warmup (baseline: full attention)"
)
TIMESTAMP_TITLE = (
    "Timestamp delta and timestamp RoPE under cosine warmup "
    "(baseline: no timestamp-delta feature)"
)
PER_LAYER_TITLE = (
    "Per-layer item embeddings under cosine warmup "
    "(baseline: one shared item embedding)"
)
MUTRANSFER_TITLE = (
    "muTransfer rate transfer across width (reference: standard width=64)"
)
MUTRANSFER_FINAL_VARIANTS = {
    "mup_dim32_lr5e2",
    "mup_dim32_lr1e1",
    "mup_dim128_lr5e2",
    "mup_dim128_lr1e1",
}
NEGATIVE_SAMPLING_TITLE = (
    "Negative sampling and logQ " "(baseline: 512 in-batch negatives with offline logQ)"
)
FEEDFORWARD_COSINE_TITLE = ARCHITECTURE_COSINE_TITLES["Feedforward kind"]
POSITION_BASE_TITLE = ARCHITECTURE_BASE_TITLES["Position encoding"]
POSITION_COSINE_TITLE = ARCHITECTURE_COSINE_TITLES["Position encoding"]
_LEGACY_TABLE_TITLES = {
    **{
        ARCHITECTURE_BASE_TITLES[title]: "Bos token" if title == "BOS token" else title
        for title, _, _ in _ARCHITECTURE_SPECS
    },
    **{
        ARCHITECTURE_COSINE_TITLES[title]: (
            "Bos token under cosine warmup"
            if title == "BOS token"
            else f"{title} under cosine warmup"
        )
        for title, _, _ in _ARCHITECTURE_SPECS
    },
    EMBEDDING_LR_TITLE: "Embedding and deep learning rates",
    DEEP_LR_TITLE: "Embedding and deep learning rates",
    FINAL_COMBINATIONS_TITLE: "Final combinations",
    SHARED_WINDOW_TITLE: "Shared attention window",
    TIMESTAMP_TITLE: "Timestamp delta and timestamp RoPE",
    PER_LAYER_TITLE: "Per-layer item embeddings",
    MUTRANSFER_TITLE: "muTransfer rate transfer across width",
}

BASE_AXES = [
    *ARCHITECTURE_AXES,
    (
        "Learning-rate schedule and warmup",
        [
            "lr_cosine",
            "lr_linear",
            "lr_inverse_sqrt",
            "lr_inverse_sqrt_warmup",
            "lr_warmup",
            "lr_cosine_warmup",
            "lr_cosine_cycles2",
            "lr_cosine_cycles4",
            "lr_step",
            "lr_exponential",
            "lr_polynomial",
            "lr_wsd",
        ],
    ),
    (
        EMBEDDING_LR_TITLE,
        [
            "embedding_lr_1e4",
            "embedding_lr_2e4",
            "embedding_lr_5e4",
            "embedding_lr_2e3",
            "embedding_lr_3e3",
            "embedding_lr_5e3",
        ],
    ),
    (
        DEEP_LR_TITLE,
        [
            "deep_lr_5e4",
            "deep_lr_2e3",
            "deep_lr_3e3",
            "deep_lr_5e3",
        ],
    ),
    (
        NEGATIVE_SAMPLING_TITLE,
        [
            "neg_online_logq",
            "neg_random",
            "neg_random_offline_logq",
            "neg_in_batch_no_logq",
            "neg_mixed_online_logq",
            "neg_mixed_offline_logq",
        ],
    ),
    (
        FINAL_COMBINATIONS_TITLE,
        [
            "combo_dim32_lr",
            "combo_all",
            "combo_lr_rates",
            "combo_embedding_time",
            "combo_embedding_position",
            "selected_quality",
            "selected_balanced",
        ],
    ),
    (
        SHARED_WINDOW_TITLE,
        [
            "window_none",
            "window_10",
            "window_25",
            "window_50",
            "window_75",
            "window_100",
        ],
    ),
    (
        TIMESTAMP_TITLE,
        [
            "time_rope",
            "time_rope_reverse",
            "time_log_rope",
            "time_log_rope_reverse",
            "time_plain_add",
            "time_log_add",
            "time_bins_add",
            "time_bins_8",
            "time_bins_16",
            "time_bins_64",
            "time_bins_reverse_rope",
            "time_log_concat",
            "time_bins_concat",
            "time_bins_log_rope",
        ],
    ),
    (PER_LAYER_TITLE, ["per_layer_embeddings"]),
    (
        MUTRANSFER_TITLE,
        [
            "cosine_dim_32",
            "lr_cosine_warmup",
            "cosine_dim_128",
            "mup_dim32_lr5e2",
            "mup_dim32_lr1e1",
            "mup_dim128_lr5e2",
            "mup_dim128_lr1e1",
        ],
    ),
]
COSINE_AXIS_TITLES = [title for title, _ in COSINE_AXES]
AXES = BASE_AXES + COSINE_AXES

_BY_AXIS = dict(AXES)
_AXIS_VARIANTS = {name for _, names in AXES for name in names} | {BASELINE}
_SCHEDULED_VARIANTS = {
    name for _, names in BASE_AXES[len(ARCHITECTURE_AXES) + 1 :] for name in names
} | {name for _, names in COSINE_AXES for name in names}
_TABLE_REFERENCES = {
    "Learning-rate schedule and warmup": BASELINE,
    "Quality per unit of cost": "lr_cosine_warmup",
    "Schedule shape, no warmup": BASELINE,
    "Warmup-stable-decay": BASELINE,
    "Constant": BASELINE,
    "Cosine": "lr_cosine",
    "Cosine restarts (warmup=5%)": "lr_cosine_warmup",
    "Inverse sqrt": "lr_inverse_sqrt",
    NEGATIVE_SAMPLING_TITLE: BASELINE,
}
_REFERENCE_DETAILS = {
    **{
        ARCHITECTURE_BASE_TITLES[title]: baseline
        for title, baseline, _ in _ARCHITECTURE_SPECS
    },
    **{
        ARCHITECTURE_COSINE_TITLES[title]: f"{baseline}; cosine, warmup=5%"
        for title, baseline, _ in _ARCHITECTURE_SPECS
    },
    "Learning-rate schedule and warmup": "constant; warmup=0%; LR=0.001",
    EMBEDDING_LR_TITLE: "embedding LR=0.001; deep LR=0.001",
    DEEP_LR_TITLE: "deep LR=0.001; embedding LR=0.001",
    NEGATIVE_SAMPLING_TITLE: "offline logQ; 512 in-batch negatives",
    FINAL_COMBINATIONS_TITLE: "embedding LR=0.001; deep LR=0.001; cosine, warmup=5%",
    SHARED_WINDOW_TITLE: "full attention; cosine, warmup=5%",
    TIMESTAMP_TITLE: "no time feature; cosine, warmup=5%",
    PER_LAYER_TITLE: "one shared item embedding; cosine, warmup=5%",
    MUTRANSFER_TITLE: "standard width=64; item embedding dim=64; cosine, warmup=5%",
    "Quality per unit of cost": "width=64; cosine, warmup=5%",
    "Schedule shape, no warmup": "constant; warmup=0%; LR=0.001",
    "Warmup-stable-decay": "constant; warmup=0%; LR=0.001",
    "Constant": "constant; warmup=0%; LR=0.001",
    "Cosine": "cosine; warmup=0%; LR=0.001",
    "Cosine restarts (warmup=5%)": "cosine; cycles=1; warmup=5%; LR=0.001",
    "Inverse sqrt": "inverse sqrt; warmup=0%; timescale=5%; LR=0.001",
    "Final metric candidates": "corrected baseline; cosine, warmup=5%",
    "Ungrouped": "corrected baseline",
}


@dataclass(frozen=True)
class Question:
    """A table is an AXES title or a titled variant list of its own.

    ``costs`` adds the time, memory and parameter columns, which only rq3 is
    asking about.
    """

    title: str
    note: str
    tables: list = field(default_factory=list)
    costs: bool = False


QUESTIONS = [
    Question(
        "rq1 — does µTransfer work?",
        "Hypothesis: with a fixed 64-dimensional item table, learned input and "
        "μP readout projections, μP initialization, MuAdam, and width-aware "
        "attention scaling, the deep learning rate selected at width 32 should "
        "transfer to width 128. Result: yes at the stable deep LR 0.05. Recall "
        "rises from 0.1256 at width 32 to 0.1303 at width 128 with lower "
        "four-seed spread, while LR 0.1 crosses the width-128 stability boundary. "
        "The fixed item table and projections are necessary: width changes only "
        "the transformer, as required by μP.",
        [MUTRANSFER_TITLE],
    ),
    Question(
        "rq3 — best metrics/performance balance",
        "Hypothesis: moderate model width and separately tuned embedding and deep "
        "learning rates should retain most quality while reducing epoch time, "
        "memory, and parameter count. Result: width 64 is the balance point; "
        "embedding rates below 0.001 hurt, while deep LR 0.003 is best among "
        "the tested values. The combined balanced candidate reaches 0.1407 "
        "recall@100 in 18.0 seconds/epoch at 13.3 GB, versus 0.1439, 20.8 "
        "seconds/epoch, and 21.5 GB for the quality candidate.",
        [
            (
                "Quality per unit of cost",
                [
                    "lr_cosine_warmup",
                    "cosine_dim_16",
                    "cosine_dim_32",
                    "cosine_dim_128",
                    "cosine_dim_256",
                    "selected_quality",
                    "selected_balanced",
                ],
            ),
            EMBEDDING_LR_TITLE,
            DEEP_LR_TITLE,
        ],
        costs=True,
    ),
    Question(
        "rq4 — does SwiGLU help?",
        "Hypothesis: SwiGLU should improve ranking quality over GELU once both "
        "implementations and their parameter-matched comparison are rerun from "
        "the corrected baseline. "
        "`ffn_swiglu` changes only the GELU MLP to SwiGLU; "
        "`ffn_swiglu_matched` also reduces the hidden width to keep the "
        "feedforward parameter count approximately fixed. Result: SwiGLU gains "
        "1.2%, and the parameter-matched arm gains 1.7%; use the matched arm.",
        [FEEDFORWARD_COSINE_TITLE],
    ),
    Question(
        "rq5 — which lr scheduler works best?",
        "Hypothesis: decaying schedules should outperform a constant learning "
        "rate, with smooth cosine or linear decay likely to be strongest. Two- "
        "and four-cycle cosine restarts test whether repeated exploration helps. "
        "Result: linear led the accepted schedule screen; two and four cosine "
        "cycles are tied with one cycle and add no benefit.",
        [
            (
                "Schedule shape, no warmup",
                [
                    "lr_cosine",
                    "lr_linear",
                    "lr_inverse_sqrt",
                    "lr_step",
                    "lr_exponential",
                    "lr_polynomial",
                ],
            ),
            (
                "Cosine restarts (warmup=5%)",
                ["lr_cosine_cycles2", "lr_cosine_cycles4"],
            ),
            ("Warmup-stable-decay", ["lr_wsd"]),
        ],
    ),
    Question(
        "rq6 — does lr warmup help?",
        "Hypothesis: a 5% warmup should stabilize early optimization and improve "
        "constant, cosine, and inverse-sqrt schedules. Each table changes only "
        "warmup within one schedule shape. Result: all three changes are inside "
        "four-seed noise, so warmup is not independently beneficial.",
        [
            ("Constant", ["lr_warmup"]),
            ("Cosine", ["lr_cosine", "lr_cosine_warmup"]),
            ("Inverse sqrt", ["lr_inverse_sqrt", "lr_inverse_sqrt_warmup"]),
        ],
    ),
    Question(
        "rq7 — rope / alibi / position embeddings, and from-the-end variants",
        "Hypothesis: relative encodings should generalize better than learned "
        "absolute positions, while counting positions from the sequence end may "
        "better align histories of different lengths. The completed grid crosses "
        "forward/reverse RoPE, forward/reverse learned positions, and ALiBi, "
        "including pure, pairwise, and three-way combinations. All rows use the "
        "same cosine-warmup reference so the comparison does not mix scheduler "
        "families.",
        [POSITION_COSINE_TITLE],
    ),
    Question(
        "rq8 — scaling",
        "Hypothesis: quality should improve with width, depth, sequence length, "
        "attention capacity, and FFN capacity until regularization or compute "
        "cost dominates. The follow-up specifically checks zero dropout, sequence "
        "length 128, GQA throughput, post-norm stability, and a window of 50. "
        "Because FlashAttention consumes ragged sequences, max_seq_len=128 does "
        "not itself create a Tensor-Core alignment benefit. Each dependence is "
        "reported in its own table. Result: keep width 64, two layers, pre-norm, "
        "dropout 0.1 and learned forward positions; adopt sequence length 128, "
        "two-query/one-KV-head GQA, and attention window 50 for throughput.",
        [
            SHARED_WINDOW_TITLE,
            *[
                ARCHITECTURE_COSINE_TITLES[title]
                for title in (
                    "Embedding and model dimension",
                    "Depth",
                    "Sequence length",
                    "Number of attention heads",
                    "Grouped-query attention",
                    "FFN ratio",
                    "Dropout",
                    "Block normalization kind",
                    "Residual normalization place",
                    "Input and final normalization",
                    "BOS token",
                )
            ],
        ],
    ),
    Question(
        "rq9 — timestamp-delta embeddings",
        "Hypothesis: learned binned time deltas and timestamp-aware RoPE should "
        "help distinguish short-term intent from older interactions. The follow-up "
        "crosses 32-bin addition with reverse timestamp RoPE and tests 8, 16, 32, "
        "and 64 logarithmically spaced delta bins. Result: additive 16-bin deltas "
        "are best at 0.1322; reverse timestamp RoPE adds cost without quality.",
        [TIMESTAMP_TITLE],
    ),
    Question(
        "rq10 — per-layer embeddings (Gemma-style)",
        "Hypothesis: fresh item embeddings at each transformer layer should add "
        "capacity and improve ranking quality, at increased parameter and memory "
        "cost. Result: recall is flat while embedding parameters triple, so do "
        "not use per-layer item tables.",
        [PER_LAYER_TITLE],
    ),
    Question(
        "rq11 — online/offline logQ, random, random+logQ, or uncorrected in-batch?",
        "Hypothesis: offline logQ should be the most accurate correction because "
        "it uses the exact positive-item distribution of the cached windows. The "
        "seven-arm comparison keeps 512 total negatives, architecture, "
        "optimization, data, and seeds fixed. It compares online/offline logQ on "
        "in-batch negatives, raw uniform random negatives, popularity-sampled "
        "random negatives with offline logQ, raw in-batch negatives, and 256+256 "
        "mixed random/in-batch arms with online or offline logQ applied only to "
        "the in-batch half.",
        [NEGATIVE_SAMPLING_TITLE],
    ),
    Question(
        "rq2 — best combination for metrics",
        "Hypothesis: combining the independently selected parameters should beat "
        "every single-axis corrected-baseline variant. Result: the quality "
        "combination reaches recall@100 0.1439 ±0.0005 and "
        "ndcg@100 0.0556 ±0.0004. It uses dim 64, depth 2, sequence length 128, "
        "2 query heads/1 KV head, window 50, parameter-matched SwiGLU width 171, "
        "pre-LayerNorm, no input norm, final LayerNorm, learned forward positions, "
        "dropout 0.1, 16 additive time-delta bins, uniform random negatives, "
        "embedding LR 0.001, deep LR 0.003, and linear decay without warmup. "
        "Training selects the best validation checkpoint within a 20-epoch "
        "safety cap at batch 128 in bf16; weight decay, gradient "
        "clipping, BOS, per-layer item tables, RoPE, and ALiBi are disabled.",
        [
            (
                "Final metric candidates",
                [
                    "selected_quality",
                    "selected_balanced",
                ],
            )
        ],
    ),
]

QUALITY_COLUMNS = [
    ("recall@100", "{:.3f}"),
    ("ndcg@100", "{:.3f}"),
    ("recall@10", "{:.3f}"),
    ("ndcg@10", "{:.3f}"),
    ("coverage@100", "{:.3f}"),
]
COST_COLUMNS = [
    ("epoch_time", "{:.1f}"),
    ("peak_memory_gb", "{:.1f}"),
    ("params_deep", "{:.3f}M"),
    ("params_embedding", "{:.1f}M"),
    ("best_epoch", "{:.0f}"),
]

REPORT_QUESTION_TITLES = {
    1: "Does μTransfer work?",
    2: "What is the best transformer combination for metrics?",
    3: "What is the best balance between metrics and performance?",
    4: "Does SwiGLU help?",
    5: "Which learning-rate scheduler works best?",
    6: "Does learning-rate warmup help?",
    7: "Which position encoding works best: RoPE, ALiBi, learned positions, or combinations?",
    8: "How do scaling and architecture choices affect metrics?",
    9: "Does a timestamp-delta representation improve metrics?",
    10: "Do separate item embeddings at every transformer layer help?",
    11: "How do online logQ, offline logQ, random, mixed, and uncorrected negatives compare?",
}

REPORT_RQ_ORDER = (1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 2)
GENERIC_REPORT_RQS = {
    "50m": frozenset(REPORT_RQ_ORDER) - {5, 11},
    "500m": frozenset(REPORT_RQ_ORDER) - {5},
}
CURRENT_COMPONENT_RQS = tuple(range(4, 12))
REPORT_METRICS = (
    "recall@100",
    "ndcg@100",
    "recall@10",
    "ndcg@10",
    "coverage@100",
)

_TUNING_FIELDS = [
    ("training semantics revision", ("training_semantics_revision",)),
    ("experiment class", ("transfer_invariants", "experiment_class")),
    ("muP base dim", ("transfer_invariants", "mup_base_dim")),
    ("muP delta dim", ("transfer_invariants", "mup_delta_dim")),
    ("embedding learning rate", ("embedding_learning_rate",)),
    ("deep learning rate", ("deep_learning_rate",)),
    ("batch size", ("batch_size",)),
    ("model dim", ("model_dim",)),
    ("item embedding dim", ("item_embedding_dim",)),
    ("max sequence length", ("transfer_invariants", "max_seq_len")),
    ("BOS", ("transfer_invariants", "bos")),
    ("CLS", ("transfer_invariants", "cls_token")),
    ("timestamp delta", ("transfer_invariants", "timestamp_delta")),
    (
        "timestamp combination",
        ("transfer_invariants", "timestamp_combination"),
    ),
    ("timestamp bins", ("transfer_invariants", "timestamp_num_bins")),
    (
        "per-layer item embeddings",
        ("transfer_invariants", "per_layer_item_embeddings"),
    ),
    (
        "dense random negative scores",
        ("transfer_invariants", "dense_random_negative_scores"),
    ),
    (
        "exclude seen from evaluation",
        ("transfer_invariants", "exclude_seen_from_evaluation"),
    ),
    ("negative sampling", ("transfer_invariants", "negative_sampling")),
    (
        "negative count",
        ("transfer_invariants", "num_in_batch_negatives"),
    ),
    (
        "random negative fraction",
        ("transfer_invariants", "random_negative_fraction"),
    ),
    ("logQ alpha", ("transfer_invariants", "logq_alpha")),
    ("logQ correction", ("transfer_invariants", "logq_correction")),
    (
        "correct positive logQ",
        ("transfer_invariants", "correct_positive_logq"),
    ),
    (
        "mask false negatives",
        ("transfer_invariants", "mask_false_negatives"),
    ),
    (
        "exclude own-group negatives",
        ("transfer_invariants", "exclude_own_group_negatives"),
    ),
    ("schedule", ("transfer_invariants", "lr_schedule", "shape")),
    (
        "warmup fraction",
        ("transfer_invariants", "lr_schedule", "warmup_fraction"),
    ),
    ("schedule cycles", ("transfer_invariants", "lr_schedule", "cycles")),
    (
        "power exponent",
        ("transfer_invariants", "lr_schedule", "power_exponent"),
    ),
    (
        "power transition tokens",
        ("transfer_invariants", "lr_schedule", "power_transition_tokens"),
    ),
    (
        "attention window",
        ("transfer_invariants", "transformer", "attention_window"),
    ),
    (
        "num layers",
        ("transfer_invariants", "transformer", "num_layers"),
    ),
    ("dropout", ("transfer_invariants", "transformer", "dropout")),
    ("FFN", ("transfer_invariants", "transformer", "ffn")),
    (
        "FFN intermediate dim",
        ("transfer_invariants", "transformer", "ffn_intermediate_dim"),
    ),
    ("attention heads", ("transfer_invariants", "transformer", "nhead")),
    ("KV heads", ("transfer_invariants", "transformer", "num_kv_heads")),
    (
        "learned positions",
        ("transfer_invariants", "transformer", "learned_positions"),
    ),
    ("RoPE", ("transfer_invariants", "transformer", "rope")),
    ("ALiBi", ("transfer_invariants", "transformer", "alibi")),
    ("normalization", ("transfer_invariants", "transformer", "norm")),
    ("norm place", ("transfer_invariants", "transformer", "norm_place")),
    ("input norm", ("transfer_invariants", "transformer", "input_norm")),
    ("final norm", ("transfer_invariants", "transformer", "final_norm")),
]


@dataclass(frozen=True)
class ReportRun:
    name: str
    configuration: str
    dataset_size: str
    research_question: int
    method: str
    status: str
    metrics: dict
    metadata: dict


_NEGATIVE_METHODS = {
    "fixed_inbatch_global_q_yi2019": "fixed in-batch global-q Yi-2019",
    "fixed_inbatch_leave_one_out": "fixed in-batch leave-one-out logQ",
    "streaming_inbatch_global_q_yi2019": "streaming in-batch global-q Yi-2019",
    "uniform_random": "uniform random",
    "popularity_random_global_q_yi2019": "popularity random global-q Yi-2019",
    "uncorrected_inbatch": "uncorrected in-batch",
    "uniform_random_plus_streaming_logq_negative_only": (
        "uniform random + streaming logQ on in-batch negatives"
    ),
    "uniform_random_plus_fixed_logq_negative_only": (
        "uniform random + fixed logQ on in-batch negatives"
    ),
    "offline_logq": "fixed in-batch global-q Yi-2019",
    "online_logq": "streaming in-batch global-q Yi-2019",
    "random": "uniform random",
    "random_offline_logq": "popularity random global-q Yi-2019",
    "in_batch_no_logq": "uncorrected in-batch",
    "mixed_online_logq": "uniform random + streaming logQ on in-batch negatives",
    "mixed_offline_logq": "uniform random + fixed logQ on in-batch negatives",
}

# Yi et al. 2019 give alpha to the streaming frequency estimator only; a cached
# proposal distribution has no such step size, so the report must not print the
# config default as if it were a tuned correction strength.
_STREAMING_LOGQ_METHODS = frozenset(
    {
        "streaming in-batch global-q Yi-2019",
        "uniform random + streaming logQ on in-batch negatives",
    }
)

_CONTROL_ALIAS_BASES = (
    "combination_baseline",
    "performance_baseline",
    "ffn_swiglu171",
    "position_learned_forward",
    "dimension_64",
    "depth_2",
    "heads_gqa2q1kv",
    "normalization_layer_pre",
    "sequence_128",
    "window_50",
    "dropout_10",
    "bos_off",
    "cls_off",
    "item_embeddings_shared",
)
_TIMESTAMP_BIN_BASES = frozenset(
    {
        "time_bins8_add",
        "time_bins16_add",
        "time_bins32_add",
        "time_bins64_add",
        "time_bins32_add_raw_rope_reverse",
        "time_bins32_concat",
        "time_bins32_add_log_rope_forward",
    }
)
# The metric the run is selected and ranked on, as the trainer logs it. Chosen
# in `GenerationExperiment` from `selection_k`; a variant that moves that has to
# move this too, which nothing here can check.
SELECTION_KEY = "epoch/val_true.recall@100"
_EPOCH_LINE = re.compile(r"epoch (\d+) finished ")


def _epoch_values(log: Path, key: str) -> list[float]:
    if not log.exists():
        return []
    pattern = re.compile(rf"{re.escape(key)}=([0-9.]+)")
    return [float(match) for match in pattern.findall(log.read_text())]


def _best_epoch(log: Path) -> float | None:
    """The epoch the run is reported on -- the first to reach its best score,
    which is the one the trainer keeps. Read back off the epoch lines rather
    than from a checkpoint, because a run that saves nothing still has one, and
    paired with the epoch each line names rather than counted, because a resumed
    run's log does not start at zero."""
    if not log.exists():
        return None
    scored = []
    for line in log.read_text().splitlines():
        epoch = _EPOCH_LINE.search(line)
        score = re.search(rf"{re.escape(SELECTION_KEY)}=([0-9.]+)", line)
        if epoch and score:
            scored.append((float(score.group(1)), float(epoch.group(1))))
    return max(scored, key=lambda pair: pair[0])[1] if scored else None


def _run(run_name: str) -> dict | None:
    report = GENERATED / "logs" / run_name / "final_metrics.json"
    if not report.exists():
        return None

    log = GENERATED / "logs" / run_name / "sweep.log"
    epoch_times = _epoch_values(log, "timing.train_epoch_time")
    memory = _epoch_values(log, "resources.peak_memory_gb")

    def millions(key: str) -> float | None:
        counted = _epoch_values(log, key)
        return counted[-1] / 1e6 if counted else None

    return json.loads(report.read_text()) | {
        "params_deep": millions("resources.params_deep"),
        "params_embedding": millions("resources.params_embedding"),
        # The first epoch also pays for compilation.
        "epoch_time": statistics.median(epoch_times) if epoch_times else None,
        "peak_memory_gb": max(memory) if memory else None,
        "best_epoch": _best_epoch(log),
    }


def _collect(dataset_size: str = "500m") -> dict[str, list[dict]]:
    variants: dict[str, dict[int, dict]] = defaultdict(dict)
    for path in sorted((GENERATED / "logs").glob(f"{RUN_PREFIX}*")):
        name = path.name.removeprefix(RUN_PREFIX)
        if name.startswith("perf_"):
            continue
        if dataset_size == "500m":
            identity = re.fullmatch(r"(.+?)_500m(?:_s(\d+))?", name)
        else:
            if re.search(r"_(?:50m|500m|5b)(?:_|$)", name):
                continue
            identity = re.fullmatch(r"(.+?)(?:_s(\d+))?", name)
        if identity is None:
            continue
        variant, seed_text = identity.groups()
        if variant == "homework_reproduction":
            continue
        if variant.startswith("mup_dim") and variant not in MUTRANSFER_FINAL_VARIANTS:
            continue
        seed = int(seed_text) if seed_text is not None else 42
        if seed not in {0, 1, 2, 3, 42}:
            continue
        if (
            dataset_size == "500m"
            and seed_text is None
            and path.with_name(f"{path.name}_s0").exists()
        ):
            continue
        if (
            dataset_size != "500m"
            and seed == 0
            and path.with_name(path.name[:-3]).exists()
        ):
            continue
        if run := _run(path.name):
            if seed in variants[variant]:
                raise ValueError(f"duplicate seed {seed} for {variant}")
            variants[variant][seed] = run
    return {
        variant: [runs[seed] for seed in sorted(runs)]
        for variant, runs in variants.items()
    }


def _values(runs: list[dict], name: str) -> list[float]:
    return [run[name] for run in runs if run.get(name) is not None]


def _cell(runs: list[dict], name: str, template: str) -> str:
    values = _values(runs, name)
    if not values:
        return "—"
    if len(values) == 1:
        return template.format(values[0])
    mean, spread = statistics.fmean(values), statistics.stdev(values)
    return f"{template.format(mean)} ±{template.format(spread)}"


def _quality_cell(runs, name, template, baseline: list[dict]) -> str:
    base = _values(baseline, name)
    if not base or not (reference := statistics.fmean(base)):
        return _cell(runs, name, template)
    change = 100 * (statistics.fmean(_values(runs, name)) - reference) / reference
    return f"{change:+.0f}% ({_cell(runs, name, template)})"


def _table(
    rows: list[tuple[str, list[dict]]],
    reference_name: str,
    reference: list[dict],
    costs: bool = True,
    title: str | None = None,
) -> str:
    cost_columns = COST_COLUMNS if costs else []
    columns = QUALITY_COLUMNS + cost_columns
    lines = [
        "| variant | reference configuration | runs | "
        + " | ".join(name for name, _ in columns)
        + " |",
        "| --- | --- | --- | " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for variant, runs in rows:
        cells = [
            variant,
            (
                _REFERENCE_DETAILS.get(title, reference_name)
                if variant == reference_name
                else "—"
            ),
            str(len(runs)),
        ]
        cells += [
            (
                _cell(runs, name, template)
                if variant == reference_name
                else _quality_cell(runs, name, template, reference)
            )
            for name, template in QUALITY_COLUMNS
        ]
        cells += [_cell(runs, name, template) for name, template in cost_columns]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _table_header(costs: bool) -> str:
    columns = QUALITY_COLUMNS + (COST_COLUMNS if costs else [])
    return (
        "| variant | reference configuration | runs | "
        + " | ".join(name for name, _ in columns)
        + " |"
    )


def _by_score(
    title: str, names: list[str], variants: dict[str, list[dict]]
) -> tuple[list[tuple[str, list[dict]]], str, list[dict]]:
    reference_name = _TABLE_REFERENCES.get(
        title,
        (
            "lr_cosine_warmup"
            if names and all(name in _SCHEDULED_VARIANTS for name in names)
            else BASELINE
        ),
    )
    reference = variants.get(reference_name, [])
    rows = [(name, variants[name]) for name in names if name in variants]
    if not rows:
        return [], reference_name, reference
    ranked = sorted(
        (
            [(reference_name, reference), *rows]
            if reference and reference_name not in names
            else rows
        ),
        key=lambda row: -statistics.fmean(run["recall@100"] for run in row[1]),
    )
    return ranked, reference_name, reference


def render(existing: str = "") -> str:
    raise RuntimeError(
        "legacy raw-result renderer is disabled; use render_compact_report"
    )


def render_questions(existing: str = "") -> str:
    raise RuntimeError(
        "legacy raw-result renderer is disabled; use render_compact_report"
    )


def _calibrated_existing(existing: str, variants: dict[str, list[dict]]) -> str:
    if PROVENANCE_MARKER in existing:
        return existing
    if set(variants) <= {BASELINE}:
        sys.exit("no calibrated comparison runs; refusing to erase archived tables")
    return ""


def _resolve_table(table: str | tuple[str, list[str]]) -> tuple[str, list[str]]:
    if isinstance(table, str):
        return table, _BY_AXIS[table]
    title, names = table
    # `_by_score` skips a name with no runs, which is how a variant that has not
    # been run yet stays out of the table -- and how a typo would too.
    if unknown := [name for name in names if name not in _AXIS_VARIANTS]:
        raise KeyError(f"{title!r} names no variant of any axis: {unknown}")
    return title, names


def _sections(text: str, prefix: str) -> dict[str, str]:
    pattern = re.compile(
        rf"(?ms)^{re.escape(prefix)}([^\n]+)\n.*?(?=^{re.escape(prefix)}|\Z)"
    )
    return {match.group(1): match.group(0).rstrip() for match in pattern.finditer(text)}


def _table_rows(
    text: str, cell_count: int | None = None, header: str | None = None
) -> dict[str, str]:
    rows = {}
    active = header is None
    for line in text.splitlines():
        if line.startswith("| variant |"):
            active = header is None or line == header
            continue
        if not line.startswith("| "):
            active = header is None
            continue
        if not active or line.startswith("| ---"):
            continue
        cells = line.split("|")[1:-1]
        if cell_count is None or len(cells) == cell_count:
            rows[cells[0].strip()] = line
    return rows


def _merge_table_rows(
    previous: str,
    current: str,
    reference_name: str | None = None,
    fallback_reference_row: str | None = None,
    allowed_rows: set[str] | None = None,
) -> str:
    if not previous and fallback_reference_row is None:
        return current

    header = next(
        line for line in current.splitlines() if line.startswith("| variant |")
    )
    cell_count = len(header.split("|")[1:-1])
    current_rows = _table_rows(current, cell_count, header)
    merged = _table_rows(previous, cell_count, header) | current_rows
    if allowed_rows is not None:
        merged = {name: row for name, row in merged.items() if name in allowed_rows}
    if reference_name == "lr_cosine_warmup":
        merged.pop(BASELINE, None)
    if (
        reference_name not in merged
        and fallback_reference_row is not None
        and len(fallback_reference_row.split("|")[1:-1]) == cell_count
    ):
        merged[reference_name] = fallback_reference_row
    if reference_name not in merged:
        reference_name = None
    reference = (
        [_mean_from_table_cell(cell) for cell in merged[reference_name].split("|")[4:9]]
        if reference_name
        else []
    )

    def relative(line: str) -> str:
        cells = [cell.strip() for cell in line.split("|")[1:-1]]
        if reference_name and cells[0] == reference_name:
            for index in range(3, 8):
                cells[index] = _absolute_table_cell(cells[index])
        elif reference_name:
            for index in range(3, 8):
                absolute = _absolute_table_cell(cells[index])
                value = _mean_from_table_cell(absolute)
                base = reference[index - 3]
                if value is not None and base:
                    change = 100 * (value - base) / base
                    cells[index] = f"{change:+.0f}% ({absolute})"
        return "| " + " | ".join(cells) + " |"

    ranked = sorted(
        (relative(line) for line in merged.values()),
        key=lambda line: -(_mean_from_table_cell(line.split("|")[4]) or 0),
    )
    return "\n".join([*current.splitlines()[:4], *ranked])


def _mean_from_table_cell(cell: str) -> float | None:
    match = re.search(r"(?:\(|^)\s*(-?\d+\.\d+)", cell.strip())
    return float(match.group(1)) if match else None


def _absolute_table_cell(cell: str) -> str:
    return cell[cell.index("(") + 1 : -1] if "%" in cell else cell


def _merge_question_tables(
    previous: str,
    current: str,
    references: dict[str, str],
    reference_rows: dict[str, dict[str, str]] | None = None,
    allowed_rows: dict[str, set[str]] | None = None,
) -> str:
    if not previous:
        return current
    pattern = re.compile(r"(?ms)^\*\*([^\n]+)\*\*\n\n.*?(?=^\*\*|^## |\Z)")
    old = _question_tables(previous)

    def merge(match: re.Match) -> str:
        title = match.group(1)
        return (
            _merge_table_rows(
                old.get(title, ""),
                match.group(0).rstrip(),
                references[title],
                (reference_rows or {}).get(title, {}).get(references[title]),
                (allowed_rows or {}).get(title),
            )
            + "\n\n"
        )

    return pattern.sub(merge, current).rstrip()


def _question_tables(text: str) -> dict[str, str]:
    pattern = re.compile(r"(?ms)^\*\*([^\n]+)\*\*\n\n.*?(?=^\*\*|^## |\Z)")
    tables = {
        match.group(1): match.group(0).rstrip() for match in pattern.finditer(text)
    }
    tables.update(
        {
            title: tables[legacy]
            for title, legacy in _LEGACY_TABLE_TITLES.items()
            if title not in tables and legacy in tables
        }
    )
    return tables


def _previous_table(previous: dict[str, str], title: str) -> str:
    if title in previous:
        return previous[title]
    return previous.get(_LEGACY_TABLE_TITLES.get(title, ""), "")


def _generated_block(path: Path, marker: str, ends_at: str | None) -> str:
    text = path.read_text()
    start = text.index(marker) + len(marker)
    end = text.index(ends_at, start) if ends_at else len(text)
    return text[start:end].strip()


def _replace_between(path: Path, marker: str, tables: str, ends_at: str | None) -> None:
    """``ends_at`` is what follows the generated block; None means the block runs
    to the end of the file."""
    text = path.read_text()
    start = text.index(marker)
    end = text.index(ends_at, start) if ends_at else len(text)
    result = f"{text[:start]}{marker}\n\n{tables}\n{text[end:]}"
    path.write_text(result.rstrip() + "\n")
    print(f"wrote {path}")


def _negative_method(configuration: str) -> str | None:
    if not configuration.startswith("neg_"):
        return None
    negative_name = configuration.removeprefix("neg_")
    for prefix in sorted(_NEGATIVE_METHODS, key=len, reverse=True):
        if negative_name == prefix or negative_name.startswith(f"{prefix}_"):
            return _NEGATIVE_METHODS[prefix]
    return None


def _without_rate_suffix(configuration: str) -> str:
    return re.split(
        r"_e\d+(?:p\d+|em\d+|e\d+)?_?d\d+(?:p\d+|em\d+|e\d+)?",
        configuration,
        maxsplit=1,
    )[0]


def _manifest_base(configuration: str) -> str:
    base = re.split(
        r"_e\d+e\d+_?d\d+e\d+",
        _without_rate_suffix(configuration),
        maxsplit=1,
    )[0]
    base = re.sub(
        r"_(?:initial|secondary|local|r\d+|[a-z0-9]+ext)$"
        # A cap continuation names the stage it continues, sometimes only by
        # carrying its cap marker into the stage token.
        r"|_(?:initial|secondary|local)?cap(?:cont(?:inue)?|resolve)?\d*$",
        "",
        base,
    )
    base = re.sub(r"_cap\d+", "", base)
    return re.sub(r"_ts\d+", "", base)


def _manifest_identity(configuration: str) -> tuple[int, str] | None:
    method = _manifest_base(configuration)

    if method == "homework_baseline":
        return 2, "homework-compatible baseline repeats"
    if method.startswith("homework_baseline_b"):
        return 2, "homework-compatible selected-batch calibration"
    if negative_method := _negative_method(configuration):
        return 11, negative_method
    if method.startswith("ffn_"):
        family = re.sub(r"(ffn_(?:gelu|swiglu))\d+$", r"\1", method)
        return 4, family.replace("_", " ")
    if method.startswith("position_"):
        return 7, method.replace("_", " ")
    if method.startswith("schedule_"):
        warmup_method = _warmup_method(method)
        return None if warmup_method is None else (6, warmup_method)
    if method.startswith("warmup_"):
        return 6, method.replace("_", " ")
    if method.startswith("time_"):
        return 9, method.replace("_", " ")
    if method.startswith(("item_embeddings_", "per_layer_")):
        return 10, method.replace("_", " ")
    if method.startswith(("combination_", "final_metrics_", "quality_")):
        return 2, method.replace("_", " ")
    if method.startswith(("combo_", "selected_quality", "selected_balanced")):
        return 2, method.replace("_", " ")
    if method.startswith(("performance_", "balance_", "cost_")):
        return 3, method.replace("_", " ")
    if method.startswith("cosine_dim_"):
        return 3, "width/quality balance"
    if match := re.fullmatch(r"mup_dim\d+_(lr.+)", method):
        return 1, f"muP width transfer at {match.group(1)}"
    if method.startswith(
        (
            "architecture_",
            "control_",
            "dimension_",
            "depth_",
            "heads_",
            "normalization_",
            "sequence_",
            "dropout_",
            "window_",
            "bos_",
            "cls_",
        )
    ):
        return 8, method.replace("_", " ")
    return None


def _report_identity(
    run_name: str, metadata: dict | None = None
) -> tuple[int, str, str] | None:
    name = re.sub(r"_s\d+$", "", run_name)
    name = re.sub(r"_(?:50m|500m)$", "", name)

    for family, method in (
        ("random", "homework-matched uniform random"),
        ("logq", "homework-matched fixed leave-one-out logQ"),
    ):
        prefix = f"g1_homework_{family}_"
        if name.startswith(prefix):
            configuration = f"homework_{family}_{name.removeprefix(prefix)}"
            return 11, method, configuration

    if name.startswith("g1_transfer_direct_"):
        configuration = name.removeprefix("g1_transfer_")
        return 1, "direct 500M LR oracle", configuration
    if name.startswith("g1_transfer_fitted_"):
        configuration = name.removeprefix("g1_transfer_")
        return 1, "fitted token-horizon prediction", configuration
    if name.startswith("g1_transfer_power_"):
        configuration = name.removeprefix("g1_transfer_")
        return 1, "Power Scheduler", configuration
    if match := re.match(r"g1_transfer_batchscale_b(?P<batch>\d+)_", name):
        configuration = name.removeprefix("g1_transfer_")
        return 1, f"batch-scaling proxy, batch {match.group('batch')}", configuration
    if name.startswith("g1_transfer_batchscaled_calibration_"):
        configuration = name.removeprefix("g1_transfer_")
        return 1, "batch-scaled 500M calibration", configuration
    if name.startswith("g1_transfer_"):
        configuration = name.removeprefix("g1_transfer_")
        schedule = _nested_value(
            metadata or {}, ("transfer_invariants", "lr_schedule", "shape")
        )
        method = (
            "Power Scheduler"
            if schedule == "power"
            else "token-horizon response surface"
        )
        return 1, method, configuration
    if name.startswith("g1_rqtune_"):
        configuration = name.removeprefix("g1_rqtune_").removeprefix("rqfinal_")
    elif name.startswith("g1_rqfinal_"):
        configuration = name.removeprefix("g1_rqfinal_")
    elif name.startswith("g1_calibrated_"):
        configuration = name.removeprefix("g1_calibrated_")
    elif name.startswith("g1_"):
        configuration = name.removeprefix("g1_")
    else:
        return None
    identity = _manifest_identity(configuration)
    if identity is None:
        return None
    research_question, method = identity
    return research_question, method, configuration


_ARCHITECTURE_MANIFEST_CACHE: (
    dict[str, tuple[str, str, str, tuple[str, ...]]] | None
) = None


def _architecture_manifest() -> dict[str, tuple[str, str, str, tuple[str, ...]]]:
    global _ARCHITECTURE_MANIFEST_CACHE
    if _ARCHITECTURE_MANIFEST_CACHE is not None:
        return _ARCHITECTURE_MANIFEST_CACHE
    manifest = EXPERIMENT / "launchers/architecture/manifest.sh"
    script = r"""
source "$1"
while IFS='|' read -r axis treatment source transformer experiment overrides alias; do
    stem=$(g1_run_stem "$axis/$treatment")
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$stem" "$source" "$transformer" "$experiment" "$overrides" "$alias"
done < <(g1_manifest_rows)
"""
    result = subprocess.run(
        ["bash", "-c", script, "_", str(manifest)],
        check=True,
        capture_output=True,
        text=True,
    )
    parsed = {}
    for line in result.stdout.splitlines():
        stem, source, transformer, experiment, overrides, alias = line.split("\t")
        parsed[stem] = (
            source or "selected_quality_b1280",
            transformer,
            experiment,
            tuple(value for value in overrides.split() if value),
        )
    _ARCHITECTURE_MANIFEST_CACHE = parsed
    return parsed


def _configuration_rates(configuration: str) -> tuple[float, float] | None:
    match = re.search(
        r"_e(?P<embedding>\d+(?:p\d+|em\d+|e\d+)?)"
        r"_?d(?P<deep>\d+(?:p\d+|em\d+|e\d+)?)",
        configuration,
    )
    if match is None:
        return None
    return (
        float(_decode_rate_slug(match.group("embedding"))),
        float(_decode_rate_slug(match.group("deep"))),
    )


def _slug_number(
    configuration: str, patterns: tuple[str, ...], default: float
) -> float:
    for pattern in patterns:
        if match := re.search(pattern, configuration):
            return float(_decode_slug(match.group(1)))
    return default


def _tuning_assignments(
    directory: Path, configuration: str, dataset_size: str
) -> list[str] | None:
    rates = _configuration_rates(configuration)
    if rates is None:
        return None
    raw_run = directory.name.removeprefix("g1_rqtune_")
    suffix = f"_{dataset_size}"
    if not raw_run.endswith(suffix):
        return None
    raw_run = raw_run.removesuffix(suffix)
    base = _manifest_base(configuration)
    manifest = _architecture_manifest().get(base)
    if manifest is not None:
        source, transformer_fields, experiment_fields, overrides = manifest
        assignments = list(overrides)
    else:
        negative_sources = {
            "fixed_inbatch_global_q_yi2019": "neg_fixed_inbatch_global_q_yi2019",
            "fixed_inbatch_leave_one_out": "neg_fixed_inbatch_leave_one_out",
            "streaming_inbatch_global_q_yi2019": "neg_streaming_inbatch_global_q_yi2019",
            "uniform_random": "neg_random",
            "popularity_random_global_q_yi2019": "neg_popularity_random_global_q_yi2019",
            "uncorrected_inbatch": "neg_in_batch_no_logq",
            "uniform_random_plus_streaming_logq_negative_only": (
                "neg_mixed_streaming_logq_negative_only"
            ),
            "uniform_random_plus_fixed_logq_negative_only": (
                "neg_mixed_fixed_logq_negative_only"
            ),
        }
        family = next(
            (
                family
                for family in sorted(negative_sources, key=len, reverse=True)
                if configuration.startswith(f"neg_{family}_")
            ),
            None,
        )
        if family is None:
            return None
        source = negative_sources[family]
        transformer_fields = ""
        experiment_fields = (
            "negative_sampling,logq_correction,correct_positive_logq,"
            "mask_false_negatives,exclude_own_group_negatives"
        )
        count = int(_slug_number(configuration, (r"_n(\d+)(?:_|$)",), 512))
        alpha = _slug_number(
            configuration,
            (r"_alpha([^_]+)(?:_|$)", r"_a([^_]+)(?:_|$)"),
            0.01,
        )
        random_fraction = _slug_number(
            configuration,
            (r"_random([^_]+)(?:_|$)", r"_a[^_]+_r([^_]+)(?:_|$)"),
            0.5,
        )
        assignments = [
            f"G1_TUNE_NUM_NEGATIVES={count}",
            f"G1_TUNE_LOGQ_ALPHA={alpha}",
        ]
        if family.startswith("uniform_random_plus_"):
            assignments.append(f"G1_TUNE_RANDOM_FRACTION={random_fraction}")
    batch_match = re.search(r"_b(?P<batch>\d+)(?:_|$)", configuration)
    effective_batch_size = (
        int(batch_match.group("batch")) if batch_match else _INITIAL_BATCH_SIZE
    )
    physical_batch_match = re.search(r"_pb(?P<batch>\d+)(?:_|$)", configuration)
    accumulation_match = re.search(r"_ga(?P<steps>\d+)(?:_|$)", configuration)
    if (physical_batch_match is None) != (accumulation_match is None):
        raise ValueError(
            "accumulated run name must include both physical batch and steps"
        )
    physical_batch_size = effective_batch_size
    accumulation_steps = 1
    if physical_batch_match is not None and accumulation_match is not None:
        physical_batch_size = int(physical_batch_match.group("batch"))
        accumulation_steps = int(accumulation_match.group("steps"))
        if physical_batch_size * accumulation_steps != effective_batch_size:
            raise ValueError(
                "physical batch times accumulation must equal batch size"
            )
    provenance_match = re.search(
        r"(?:_cap(?P<epochs>\d+))?_ts(?P<semantics>\d+)_r(?P<revision>\d+)$",
        configuration,
    )
    if (
        provenance_match is None
        or int(provenance_match.group("semantics"))
        != GENERATION_TRAINING_SEMANTICS_REVISION
    ):
        return None
    epochs = int(provenance_match.group("epochs") or 20)
    revision = int(provenance_match.group("revision"))
    return [
        f"G1_TUNE_RUN={raw_run}",
        f"G1_TUNE_RUN_REVISION={revision}",
        f"G1_TUNE_EPOCHS={epochs}",
        f"G1_TUNE_SOURCE_VARIANT={source}",
        f"G1_TUNE_TRANSFORMER_FIELDS={transformer_fields}",
        f"G1_TUNE_EXPERIMENT_FIELDS={experiment_fields}",
        f"G1_TUNE_EMBEDDING_LR={rates[0]}",
        f"G1_TUNE_DEEP_LR={rates[1]}",
        f"G1_TUNE_BATCH_SIZE={physical_batch_size}",
        f"G1_TUNE_GRADIENT_ACCUMULATION_STEPS={accumulation_steps}",
        *assignments,
    ]


def _homework_negative_assignments(
    directory: Path, configuration: str, dataset_size: str, family: str
) -> list[str] | None:
    prefix = f"homework_{family}_"
    if not configuration.startswith(prefix):
        return None
    run = configuration.removeprefix(prefix)
    rates = _configuration_rates(configuration)
    provenance = re.search(
        rf"(?:_cap(?P<epochs>\d+))?"
        rf"_ts{GENERATION_TRAINING_SEMANTICS_REVISION}"
        r"_r(?P<revision>[1-9]\d*)$",
        configuration,
    )
    expected_name = f"g1_homework_{family}_{run}_{dataset_size}"
    if rates is None or provenance is None or directory.name != expected_name:
        return None
    variable = f"G1_HOMEWORK_{family.upper()}"
    assignments = [
        f"{variable}_RUN={run}",
        f"{variable}_EPOCHS={provenance.group('epochs') or 20}",
        f"{variable}_RUN_REVISION={provenance.group('revision')}",
        f"{variable}_EMBEDDING_LR={rates[0]}",
        f"{variable}_DEEP_LR={rates[1]}",
    ]
    if dataset_size == "500m":
        assignments.append(f"{variable}_DATASET_SIZE=500m")
    return assignments


def _transfer_assignments(
    directory: Path, configuration: str, dataset_size: str
) -> list[str] | None:
    rates = _configuration_rates(configuration)
    if rates is None:
        return None
    provenance_match = re.search(
        rf"(?:_cap(?P<epochs>\d+))?"
        rf"_ts{GENERATION_TRAINING_SEMANTICS_REVISION}"
        r"_r(?P<revision>[1-9]\d*)$",
        configuration,
    )
    if provenance_match is None:
        return None
    epochs = int(provenance_match.group("epochs") or 20)
    run_revision = int(provenance_match.group("revision"))
    if epochs < 20:
        return None
    batch_size = 1280
    source_variant = "selected_quality_b1280"
    if configuration.startswith("power_"):
        pass
    elif match := re.match(r"batchscale_b(?P<batch>\d+)_", configuration):
        batch_size = int(match.group("batch"))
        source_variant = "homework_fixed_leave_one_out"
    elif configuration.startswith("selected_native50_"):
        source_variant = "homework_fixed_leave_one_out"
    else:
        return None
    power_tokens = None
    if configuration.startswith("power_"):
        if match := re.search(r"_t(?P<value>\d+)(?P<unit>k)?(?:_|$)", configuration):
            power_tokens = int(match.group("value"))
            if match.group("unit") == "k":
                power_tokens *= 1000
        else:
            return None
    raw_run = directory.name.removeprefix("g1_transfer_")
    suffix = f"_{dataset_size}"
    if not raw_run.endswith(suffix):
        return None
    raw_run = raw_run.removesuffix(suffix)
    assignments = [
        f"G1_DATASET_SIZE={dataset_size}",
        f"G1_TRANSFER_RUN={raw_run}",
        f"G1_TRANSFER_RUN_REVISION={run_revision}",
        f"G1_TRANSFER_EPOCHS={epochs}",
        f"G1_TRANSFER_EMBEDDING_LR={rates[0]}",
        f"G1_TRANSFER_DEEP_LR={rates[1]}",
        "G1_TRANSFER_PARAMETERIZATION=conventional",
        f"G1_TRANSFER_BATCH_SIZE={batch_size}",
        "G1_TRANSFER_DIM=64",
        f"G1_TRANSFER_SOURCE_VARIANT={source_variant}",
    ]
    if power_tokens is not None:
        assignments.append(f"G1_TRANSFER_POWER_TOKENS={power_tokens}")
    return assignments


@cache
def _cached_exact_artifact_matches(
    directory: Path,
    configuration: str,
    dataset_size: str,
    artifact_revision: tuple[
        tuple[int, int] | None,
        tuple[int, int] | None,
        tuple[int, int] | None,
    ],
) -> bool:
    try:
        for family in ("random", "logq"):
            if directory.name.startswith(f"g1_homework_{family}_"):
                assignments = _homework_negative_assignments(
                    directory, configuration, dataset_size, family
                )
                config = EXPERIMENT / f"configs/homework_{family}_control.py"
                return assignments is not None and verify_artifact.verify_config(
                    directory, config, assignments
                )
        if directory.name.startswith("g1_rqtune_"):
            assignments = _tuning_assignments(directory, configuration, dataset_size)
            return assignments is not None and verify_artifact.verify(
                directory, dataset_size, assignments
            )
        if directory.name.startswith("g1_transfer_"):
            assignments = _transfer_assignments(directory, configuration, dataset_size)
            config = EXPERIMENT / "configs/transfer_variant.py"
            return assignments is not None and verify_artifact.verify_config(
                directory, config, assignments
            )
        if directory.name.startswith("g1_calibrated_"):
            seed_match = re.search(r"_s(?P<seed>\d+)$", directory.name)
            name = re.sub(r"_s\d+$", "", directory.name)
            name_match = re.fullmatch(
                rf"g1_calibrated_(?P<identity>.+)_{re.escape(dataset_size)}"
                r"(?P<sample>_\d+users_seed42)?(?P<validation>_val\d+)?",
                name,
            )
            if name_match is None:
                return False
            identity_match = re.fullmatch(
                r"(?P<variant>.+?)(?:_cap(?P<epochs>\d+))?"
                rf"_ts(?P<semantics>{GENERATION_TRAINING_SEMANTICS_REVISION})",
                name_match.group("identity"),
            )
            if identity_match is None:
                return False
            assignments = [
                f"G1_DATASET_SIZE={dataset_size}",
                f"G1_VARIANT={identity_match.group('variant')}",
                f"G1_MAX_EPOCHS={identity_match.group('epochs') or 20}",
            ]
            if sample := name_match.group("sample"):
                users = sample[1:].removesuffix("users_seed42")
                assignments.append(f"G1_MAX_USERS={users}")
            if validation := name_match.group("validation"):
                assignments.append(f"G1_VAL_BATCH_SIZE={validation.removeprefix('_val')}")
            if seed_match:
                assignments.append(f"G1_SEED={seed_match.group('seed')}")
            config = EXPERIMENT / "configs/variant.py"
            return verify_artifact.verify_config(directory, config, assignments)
    except (KeyError, OSError, RuntimeError, subprocess.SubprocessError, ValueError):
        return False
    return False


def _artifact_revision(
    directory: Path,
) -> tuple[
    tuple[int, int] | None,
    tuple[int, int] | None,
    tuple[int, int] | None,
]:
    def modified(path: Path) -> tuple[int, int] | None:
        try:
            status = path.stat()
            return status.st_mtime_ns, status.st_size
        except OSError:
            return None

    return (
        modified(directory),
        modified(directory / "training_metadata.json"),
        modified(directory / "final_metrics.json"),
    )


def _exact_artifact_matches(
    directory: Path, configuration: str, dataset_size: str
) -> bool:
    return _cached_exact_artifact_matches(
        directory,
        configuration,
        dataset_size,
        _artifact_revision(directory),
    )


_HORIZON_FREE_SHAPES = frozenset({"constant", "inverse_sqrt", "power"})

# Fraction of the decay phase a shape has to reach before its rate moves at
# all. The smooth shapes decay from their first post-warmup step.
_DECAY_ONSET = {"step": 0.5, "warmup_stable_decay": 0.8}


def _schedule_horizon_epochs(metadata: dict) -> int | None:
    horizon = metadata.get("lr_schedule_horizon_epochs")
    if isinstance(horizon, int) and not isinstance(horizon, bool):
        return horizon
    num_epochs = metadata.get("num_epochs")
    return num_epochs if isinstance(num_epochs, int) else None


def _decay_progress(metadata: dict) -> float | None:
    """How far into its decay phase the run's last epoch was."""
    schedule = _nested_value(metadata, ("transfer_invariants", "lr_schedule"))
    horizon = _schedule_horizon_epochs(metadata)
    epochs_trained = metadata.get("epochs_trained")
    if not isinstance(schedule, dict) or not horizon or not epochs_trained:
        return None
    warmup_epochs = horizon * (schedule.get("warmup_fraction") or 0.0)
    decay_epochs = horizon - warmup_epochs
    if decay_epochs <= 0:
        return None
    return (epochs_trained - warmup_epochs) / decay_epochs


def _completed_an_annealed_horizon(metadata: dict) -> bool:
    """Whether a schedule that anneals over its horizon spent all of it.

    Such a shape declares its own length, so the epoch it anneals into is the
    result the schedule was set up to produce: the run has nothing left to
    train and needs no early stop to have chosen where to end.
    """
    if metadata.get("lr_horizon_complete"):
        return True
    schedule = _nested_value(metadata, ("transfer_invariants", "lr_schedule"))
    if not isinstance(schedule, dict):
        return False
    if schedule.get("shape") in _HORIZON_FREE_SHAPES:
        return False
    progress = _decay_progress(metadata)
    return progress is not None and progress >= 1


def _resolved_its_selection(metadata: dict) -> bool:
    if _completed_an_annealed_horizon(metadata):
        horizon = _schedule_horizon_epochs(metadata)
        return horizon is None or metadata.get("best_epoch") <= horizon
    return (
        metadata.get("stopped_epoch") < metadata.get("max_epochs")
        and metadata.get("early_stopped") is True
        and metadata.get("best_epoch_at_cap") is False
        and metadata.get("selection_resolved") is True
    )


def _declared_decay_engaged(metadata: dict) -> bool:
    """Whether a decaying shape ever left its opening rate.

    A ``step`` or warmup-stable-decay run that stopped before its first drop is
    numerically the constant schedule, so it is no evidence for its own shape.
    """
    schedule = _nested_value(metadata, ("transfer_invariants", "lr_schedule"))
    if not isinstance(schedule, dict):
        return True
    onset = _DECAY_ONSET.get(schedule.get("shape"))
    if onset is None:
        return True
    progress = _decay_progress(metadata)
    return progress is None or progress >= onset


def _uses_validation_selected_training(metadata: dict) -> bool:
    invariants = metadata.get("transfer_invariants")
    if not isinstance(invariants, dict):
        return False
    patience = invariants.get("early_stopping_patience")
    min_delta = invariants.get("early_stopping_min_delta")
    max_epochs = metadata.get("max_epochs")
    epochs_trained = metadata.get("epochs_trained")
    best_epoch = metadata.get("best_epoch")
    stopped_epoch = metadata.get("stopped_epoch")
    return (
        isinstance(max_epochs, int)
        and not isinstance(max_epochs, bool)
        and max_epochs >= 20
        and metadata.get("num_epochs") == max_epochs
        and isinstance(epochs_trained, int)
        and not isinstance(epochs_trained, bool)
        and 1 <= epochs_trained <= max_epochs
        and invariants.get("eval_every_n_epochs") == 1
        and invariants.get("restore_best_weights") is True
        and patience == 3
        and min_delta == 0.0
        and invariants.get("early_stopping_metric") == "recall@100"
        and invariants.get("early_stopping_metric_prefix") == "epoch/val_true"
        and isinstance(best_epoch, int)
        and not isinstance(best_epoch, bool)
        and best_epoch > 0
        and isinstance(stopped_epoch, int)
        and not isinstance(stopped_epoch, bool)
        and stopped_epoch >= best_epoch
        and stopped_epoch == epochs_trained
        and _resolved_its_selection(metadata)
        and _declared_decay_engaged(metadata)
    )


def _run_status(
    directory: Path,
    metrics: dict,
    metadata: dict,
    dataset_size: str,
    research_question: int,
    configuration: str,
) -> str:
    if metrics:
        metrics_path = directory / "final_metrics.json"
        metadata_path = directory / "training_metadata.json"
        if not metadata or not metadata_path.exists():
            return "unusable"
        if (
            metrics_path.exists()
            and metrics_path.stat().st_mtime_ns < metadata_path.stat().st_mtime_ns
        ):
            return "incomplete"
        if not has_current_generation_semantics(metadata):
            return "unusable"
        base = _manifest_base(configuration)
        homework_control = base.startswith("homework_baseline")
        homework_negative_control = any(
            base == prefix or base.startswith(f"{prefix}_")
            for prefix in ("homework_random", "homework_logq")
        )
        if homework_negative_control and not _exact_artifact_matches(
            directory, configuration, dataset_size
        ):
            return "unusable"
        if not _uses_validation_selected_training(metadata):
            return "unusable"
        if (
            _manifest_base(configuration) in _TIMESTAMP_BIN_BASES
            and _nested_value(
                metadata,
                ("transfer_invariants", "timestamp_bin_semantics_revision"),
            )
            != TIMESTAMP_BIN_SEMANTICS_REVISION
        ):
            return "unusable"
        experiment_class = _nested_value(
            metadata, ("transfer_invariants", "experiment_class")
        )
        if homework_control:
            homework_base = _manifest_base(configuration)
            batch_match = re.fullmatch(
                r"homework_baseline(?:_b(?P<batch>\d+))?",
                homework_base,
            )
            seed_match = re.search(r"_s(?P<seed>\d+)$", directory.name)
            if homework_base == "homework_baseline_native500_r3":
                batch_size = 1280
            elif batch_match is not None:
                batch_size = int(batch_match.group("batch") or 1280)
            else:
                return "unusable"
            expected_seed = (
                int(seed_match.group("seed")) if seed_match is not None else 42
            )
            if baseline_spread.homework_metadata_errors(
                metadata,
                batch_size=batch_size,
                seed=expected_seed,
            ):
                return "unusable"
        elif not homework_negative_control and not _exact_artifact_matches(
            directory, configuration, dataset_size
        ):
            return "unusable"
        if (
            2 <= research_question <= 11
            and not homework_control
            and not homework_negative_control
            and experiment_class != "MuTransferGenerationExperiment"
        ):
            return "unusable"
        return "completed"
    log = directory / "sweep.log"
    if log.exists() and re.search(
        r"(?m)(Traceback|Error:|\bFAILED\b|\bFATAL\b)",
        log.read_text(errors="replace"),
    ):
        return "failed"
    return "incomplete"


def _dataset_from_directory(directory: Path, metadata: dict) -> str | None:
    if re.search(r"(?:^|_)\d+users(?:_|$)", directory.name):
        return None
    if metadata.get("dataset_size") in {"50m", "500m"}:
        return str(metadata["dataset_size"])
    match = re.search(r"_(50m|500m)(?:_s\d+)?$", directory.name)
    if match:
        return match.group(1)
    return "50m" if _report_identity(directory.name, metadata) is not None else None


def _status_rank(status: str) -> int:
    return {"completed": 3, "unusable": 2, "failed": 1, "incomplete": 0}[status]


def _load_run_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _materialize_control_aliases(runs: list[ReportRun]) -> None:
    controls = [
        run
        for run in runs
        if run.research_question == 8
        and _without_rate_suffix(run.configuration)
        in {"architecture_control", "control_control"}
    ]
    for control in controls:
        source_base = _without_rate_suffix(control.configuration)
        suffix = control.configuration.removeprefix(source_base)
        for base in _CONTROL_ALIAS_BASES:
            configuration = f"{base}{suffix}"
            identity = _manifest_identity(configuration)
            if identity is None:
                raise ValueError(
                    f"control alias has no report identity: {configuration}"
                )
            research_question, method = identity
            alias = replace(
                control,
                configuration=configuration,
                research_question=research_question,
                method=method,
            )
            runs.append(alias)


def _materialize_rq1_mup_aliases(runs: list[ReportRun]) -> None:
    aliases = []
    for run in runs:
        base = _without_rate_suffix(run.configuration)
        if run.research_question != 8 or not base.startswith("dimension_"):
            continue
        width = base.removeprefix("dimension_")
        suffix = run.configuration.removeprefix(base)
        aliases.append(
            replace(
                run,
                configuration=f"mup_dim{width}{suffix}",
                research_question=1,
                method="μP model-width transfer",
            )
        )
    runs.extend(aliases)


def _materialize_homework_logq_aliases(runs: list[ReportRun]) -> None:
    runs.extend(
        replace(
            run,
            configuration=f"homework_logq_{run.configuration}",
            research_question=11,
            method="homework-matched fixed leave-one-out logQ",
        )
        for run in list(runs)
        if run.research_question == 1
        and run.configuration.startswith("selected_native50_")
    )


def _is_rq1_alias_source(configuration: str) -> bool:
    base = _without_rate_suffix(configuration)
    return base.startswith("dimension_") or base in {
        "architecture_control",
        "control_control",
    }


def load_report_runs(
    dataset_size: str,
    *,
    research_question: int | None = None,
    directories: Iterable[Path] | None = None,
    configuration_base_filter: Callable[[str], bool] | None = None,
) -> list[ReportRun]:
    runs: list[ReportRun] = []
    run_directories = (
        (GENERATED / "logs").glob("g1_*")
        if directories is None
        else directories
    )
    for directory in sorted(run_directories):
        if not directory.is_dir():
            continue
        name_identity = _report_identity(directory.name)
        if name_identity is None:
            continue
        if (
            configuration_base_filter is not None
            and not configuration_base_filter(_manifest_base(name_identity[2]))
        ):
            continue
        metadata_path = directory / "training_metadata.json"
        metrics_path = directory / "final_metrics.json"
        metadata = _load_run_json(metadata_path)
        if _dataset_from_directory(directory, metadata) != dataset_size:
            continue
        identity = _report_identity(directory.name, metadata)
        if identity is None:
            continue
        if research_question is not None and identity[0] != research_question:
            rq1_alias = research_question == 1 and _is_rq1_alias_source(identity[2])
            rq11_alias = research_question == 11 and identity[0] == 1 and identity[
                2
            ].startswith("selected_native50_")
            if not (rq1_alias or rq11_alias):
                continue
        metrics = _load_run_json(metrics_path)
        if metrics:
            metrics.update(
                {
                    name: value
                    for name, value in (_run(directory.name) or {}).items()
                    if name in {column for column, _ in COST_COLUMNS}
                }
            )
        run_research_question, method, configuration = identity
        run = ReportRun(
            name=directory.name,
            configuration=configuration,
            dataset_size=dataset_size,
            research_question=run_research_question,
            method=method,
            status=_run_status(
                directory,
                metrics,
                metadata,
                dataset_size,
                run_research_question,
                configuration,
            ),
            metrics=metrics,
            metadata=metadata,
        )
        runs.append(run)

        base = _without_rate_suffix(configuration)
        if run_research_question == 2 and base.startswith(
            ("selected_quality", "selected_balanced")
        ):
            runs.append(
                replace(
                    run,
                    research_question=3,
                    method="selected quality/performance candidate",
                )
            )
        if run_research_question == 2 and base.startswith("homework_baseline"):
            runs.append(
                replace(
                    run,
                    research_question=3,
                    method=run.method,
                )
            )

        if run_research_question == 6:
            warmup_method = _warmup_method(configuration)
            if warmup_method is not None:
                run = replace(run, method=warmup_method)
                runs[-1] = run
    _materialize_control_aliases(runs)
    _materialize_rq1_mup_aliases(runs)
    _materialize_homework_logq_aliases(runs)
    if research_question is not None:
        runs = [run for run in runs if run.research_question == research_question]
    return sorted(
        runs,
        key=lambda run: (
            run.research_question,
            run.method,
            run.configuration,
            run.name,
        ),
    )


def _warmup_method(configuration: str) -> str | None:
    treatment = _without_rate_suffix(configuration).removeprefix("schedule_")
    if treatment in {"constant", "constant_warmup5"}:
        return "constant warmup"
    if treatment in {"cosine", "cosine_warmup5_cycles1"}:
        return "cosine warmup"
    if treatment in {"inverse_sqrt", "inverse_sqrt_warmup5"}:
        return "inverse sqrt warmup"
    return None


def _nested_value(mapping: dict, path: tuple[str, ...]) -> object:
    value: object = mapping
    for name in path:
        if not isinstance(value, dict) or name not in value:
            return None
        value = value[name]
    return value


def _format_report_value(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _decode_slug(value: str) -> str:
    return value.replace("em", "e-").replace("p", ".").replace("m", "-")


def _decode_rate_slug(value: str) -> str:
    if "p" in value or "m" in value:
        return _decode_slug(value)
    if match := re.fullmatch(r"(\d+)e(\d+)", value):
        return f"{match.group(1)}e-{match.group(2)}"
    return f"{int(value) / 1000:g}"


def _tuned_value(run: ReportRun, label: str, path: tuple[str, ...]) -> object:
    value = _nested_value(run.metadata, path)
    if value is not None:
        return value
    rates = re.search(
        r"_e(?P<embedding>\d+(?:p\d+|em\d+|e\d+)?)"
        r"_?d(?P<deep>\d+(?:p\d+|em\d+|e\d+)?)",
        run.configuration,
    )
    if rates is not None and label in {
        "embedding learning rate",
        "deep learning rate",
    }:
        group = "embedding" if label == "embedding learning rate" else "deep"
        return _decode_rate_slug(rates.group(group))
    patterns = {
        "batch size": r"_b(\d+)(?:_|$)",
        "negative count": r"_n(\d+)(?:_|$)",
        "logQ alpha": r"_a([^_]+)(?:_|$)",
        "random negative fraction": r"_r([^_]+)(?:_|$)",
    }
    pattern = patterns.get(label)
    if pattern is None:
        return None
    match = re.search(pattern, run.configuration)
    return _decode_slug(match.group(1)) if match else None


def _tuned_columns(runs: list[ReportRun]) -> list[tuple[str, tuple[str, ...]]]:
    required = {
        "embedding learning rate",
        "deep learning rate",
    }
    if runs and runs[0].research_question == 1:
        required.update(
            {
                "model dim",
                "item embedding dim",
                "schedule",
                "power exponent",
                "power transition tokens",
            }
        )
    if runs and runs[0].research_question == 4:
        required.update({"FFN", "FFN intermediate dim"})
    if runs and runs[0].research_question in {5, 6}:
        required.update({"schedule", "warmup fraction", "schedule cycles"})
    if runs and runs[0].research_question == 7:
        required.update({"learned positions", "RoPE", "ALiBi"})
    if runs and runs[0].research_question == 8:
        required.update(
            {
                "model dim",
                "max sequence length",
                "BOS",
                "CLS",
                "attention window",
                "num layers",
                "dropout",
                "FFN",
                "FFN intermediate dim",
                "attention heads",
                "KV heads",
                "normalization",
                "norm place",
                "input norm",
                "final norm",
            }
        )
    if runs and runs[0].research_question == 9:
        required.update(
            {"timestamp delta", "timestamp combination", "timestamp bins", "RoPE"}
        )
    if runs and runs[0].research_question == 10:
        required.add("per-layer item embeddings")
    if runs and runs[0].research_question == 11:
        required.update(
            {
                "negative sampling",
                "negative count",
                "random negative fraction",
                "logQ alpha",
                "logQ correction",
                "correct positive logQ",
                "mask false negatives",
                "exclude own-group negatives",
                "dense random negative scores",
            }
        )
    selected = []
    for label, path in _TUNING_FIELDS:
        values = {
            _format_report_value(_tuned_value(run, label, path)) for run in runs
        } - {"—"}
        if label in required or len(values) > 1:
            selected.append((label, path))
    return selected


def _metric_value(run: ReportRun, name: str) -> float | None:
    value = run.metrics.get(name)
    return float(value) if value is not None else None


def _validated_baseline_summary(*, required: bool = False) -> dict | None:
    if not BASELINE_SPREAD.exists():
        if required:
            raise FileNotFoundError(
                f"shared empirical bands are unavailable: {BASELINE_SPREAD}"
            )
        return None
    try:
        summary = json.loads(BASELINE_SPREAD.read_text())
        prefix = summary.get("run_prefix")
        if not isinstance(prefix, str) or not prefix:
            raise ValueError("shared empirical bands omit their run prefix")
        recomputed = baseline_spread.summarize(prefix)
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        if required:
            raise
        return None
    if summary != recomputed:
        if required:
            raise ValueError(
                "shared empirical bands do not match the current ten homework artifacts"
            )
        return None
    if summary.get("n") != 10 or summary.get("seeds") != list(range(10)):
        raise ValueError("shared empirical bands require exactly seeds 0 through 9")
    return summary


def _metric_bands(*, required: bool = False) -> dict[str, float]:
    summary = _validated_baseline_summary(required=required)
    if summary is None:
        return {}
    bands = {}
    for metric, values in summary.get("metrics", {}).items():
        band = float(values["absolute_band"])
        if not math.isfinite(band) or band < 0:
            raise ValueError(f"invalid empirical band for {metric}: {band}")
        bands[metric] = band
    if required:
        missing = set(REPORT_METRICS) - set(bands)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"shared empirical bands omit reported metrics: {names}")
    return bands




# A width whose 50M-local optimum is the common rate has one artifact, not two, so
# any delta printed for it is the run against itself.
SELF_COMPARISON = "same run"


def _absolute_metric_cell(run: ReportRun, metric: str, bands: dict[str, float]) -> str:
    value = _metric_value(run, metric)
    if value is None:
        return "—"
    return reporting.absolute(value)


def _relative_metric_cell(
    run: ReportRun,
    control: ReportRun | None,
    metric: str,
    bands: dict[str, float],
) -> str:
    absolute = _absolute_metric_cell(run, metric, bands)
    value = _metric_value(run, metric)
    reference = _metric_value(control, metric) if control is not None else None
    if value is None or reference is None or reference == 0 or run is control:
        return absolute
    difference = value - reference
    percent = 100 * difference / reference
    rendered_percent = "0%" if round(percent) == 0 else f"{percent:+.0f}%"
    return reporting.colored(f"{rendered_percent} ({absolute})", metric, difference)


def _relative_percent_cell(
    run: ReportRun,
    control: ReportRun | None,
    metric: str,
    bands: dict[str, float],
) -> str:
    value = _metric_value(run, metric)
    reference = _metric_value(control, metric) if control is not None else None
    if value is None or reference is None or reference == 0 or run == control:
        return "—"
    difference = value - reference
    percent = 100 * difference / reference
    cell = "0%" if round(percent) == 0 else f"{percent:+.0f}%"
    return reporting.colored(cell, metric, difference)


def _completed_best(runs: list[ReportRun]) -> ReportRun | None:
    completed = [
        run
        for run in runs
        if run.status == "completed" and _metric_value(run, "recall@100") is not None
    ]
    return (
        min(
            completed,
            key=lambda run: (
                -(_metric_value(run, "recall@100") or float("-inf")),
                -(_metric_value(run, "ndcg@100") or float("-inf")),
                run.name,
            ),
        )
        if completed
        else None
    )


def _aggregate_homework_baseline(runs: list[ReportRun]) -> ReportRun | None:
    repeats = [
        run
        for run in runs
        if run.method == "homework-compatible baseline repeats"
        and run.status == "completed"
    ]
    if len(repeats) != 10:
        return None
    summary = _validated_baseline_summary()
    if summary is None:
        return None
    prefix = str(summary.get("run_prefix", ""))
    expected_names = {f"{prefix}{seed}" for seed in range(10)}
    if {run.name for run in repeats} != expected_names:
        return None
    metrics = {
        metric: float(values["mean"])
        for metric, values in summary.get("metrics", {}).items()
    }
    return replace(
        repeats[0],
        name=f"aggregate:{prefix}0-9",
        configuration="homework_baseline",
        metrics=metrics,
    )


def _control_patterns_for_method(
    research_question: int, method: str
) -> tuple[str, ...]:
    if research_question == 5:
        return ("schedule_constant",)
    if research_question == 6:
        return {
            "constant warmup": ("schedule_constant",),
            "cosine warmup": ("schedule_cosine",),
            "inverse sqrt warmup": ("schedule_inverse_sqrt",),
        }.get(method, ())
    if research_question == 8:
        for prefix, patterns in (
            ("FFN capacity", ("ffn_swiglu171",)),
            ("dimension ", ("dimension_64",)),
            ("depth ", ("depth_2",)),
            ("heads ", ("heads_mha2",)),
            ("normalization ", ("normalization_layer_pre",)),
            ("sequence ", ("sequence_128",)),
            ("window ", ("window_50",)),
            ("dropout ", ("dropout_0.1", "dropout_10")),
            ("bos ", ("bos_off",)),
            ("cls ", ("cls_off",)),
        ):
            if method.startswith(prefix):
                return patterns
    return _CONTROL_CONFIGURATION_PATTERNS.get(research_question, ())


def _ledger_control_run(
    research_question: int,
    method: str,
    available: list[ReportRun],
) -> ReportRun | None:
    if research_question == 1:
        if method == "μP model-width transfer":
            return _completed_best(
                [
                    run
                    for run in available
                    if run.method == method
                    and _manifest_base(run.configuration).startswith("mup_dim32_")
                ]
            )
        direct = [run for run in available if run.method == "direct 500M LR oracle"]
        return _completed_best(direct)
    if research_question == 4:
        return _completed_best([run for run in available if run.method == "ffn gelu"])
    if research_question in {2, 3}:
        aggregate = _aggregate_homework_baseline(available)
        if aggregate is not None:
            return aggregate
    if research_question == 11:
        return _completed_best(
            [
                run
                for run in available
                if run.method == "fixed in-batch leave-one-out logQ"
            ]
        )
    patterns = _control_patterns_for_method(research_question, method)
    return _completed_best(
        [
            run
            for run in available
            if any(pattern == _manifest_base(run.configuration) for pattern in patterns)
        ]
    )


def _performance_cell(run: ReportRun, name: str, template: str) -> str:
    value = _metric_value(run, name)
    return template.format(value) if value is not None else "—"


_PROTOCOL_IDENTITY_FIELDS = (
    "training_semantics_revision",
    "dataset_size",
    "seed",
    "batch_size",
    "embedding_learning_rate",
    "deep_learning_rate",
    "num_epochs",
    "max_epochs",
    "transfer_invariants",
)


# Wall clock and peak memory depend on the host the job landed on, so two launches
# of the same computation agree on everything except these.
_HOST_MEASURED_METRICS = frozenset({"epoch_time", "peak_memory_gb"})


_CAP_PROVENANCE = re.compile(
    r"(?:_cap(?P<epochs>\d+))?"
    rf"_ts{GENERATION_TRAINING_SEMANTICS_REVISION}"
    r"_r(?P<revision>[1-9]\d*)$"
)


def _continued_to_its_longest_cap(runs: list[ReportRun]) -> ReportRun | None:
    """The last stage of one configuration's cap-continuation chain.

    Artifacts of the same configuration form a chain when each launch extends
    the cap of the one before it. ``None`` says they do not: two launches at the
    same cap, or a later revision that shortened it, are separate results rather
    than one lineage, and the caller has to say so.
    """
    stages = []
    for run in runs:
        provenance = _CAP_PROVENANCE.search(run.configuration)
        if provenance is None:
            return None
        stages.append(
            (
                int(provenance.group("epochs") or 20),
                int(provenance.group("revision")),
                _manifest_base(run.configuration),
                json.dumps(
                    {
                        name: run.metadata.get(name)
                        for name in (
                            "training_semantics_revision",
                            "dataset_size",
                            "seed",
                            "batch_size",
                            "embedding_learning_rate",
                            "deep_learning_rate",
                            "transfer_invariants",
                        )
                    },
                    sort_keys=True,
                ),
                run,
            )
        )
    stages.sort(key=lambda stage: stage[:2])
    epochs = [stage[0] for stage in stages]
    revisions = [stage[1] for stage in stages]
    if (
        not stages
        or len({stage[2] for stage in stages}) != 1
        or len({stage[3] for stage in stages}) != 1
        or epochs != sorted(set(epochs))
        # A cap continuation keeps the revision it continues, so the chain has
        # to grow in cap without ever going back a revision.
        or revisions != sorted(revisions)
    ):
        return None
    return stages[-1][4]


def _repeated_launch_identity(run: ReportRun) -> str:
    return json.dumps(
        {
            "base": _manifest_base(run.configuration),
            "protocol": {
                name: run.metadata.get(name) for name in _PROTOCOL_IDENTITY_FIELDS
            },
            "metrics": {
                name: value
                for name, value in run.metrics.items()
                if name not in _HOST_MEASURED_METRICS
            },
        },
        sort_keys=True,
    )


def _without_repeated_launches(
    runs: list[ReportRun], control: ReportRun | None
) -> list[ReportRun]:
    """Collapse artifacts that ran one configuration twice under different labels.

    Same protocol and the same reported metrics means the second launch recomputed
    the first instead of extending it, so the two are one result for the reader
    rather than a lineage of stages.
    """
    by_identity: dict[str, list[ReportRun]] = defaultdict(list)
    for run in runs:
        by_identity[_repeated_launch_identity(run)].append(run)
    kept = set()
    for group in by_identity.values():
        if control is not None and control in group:
            kept.add(control.name)
        else:
            kept.add(max(run.name for run in group))
    return [run for run in runs if run.name in kept]


def _report_table(
    runs: list[ReportRun],
    *,
    compact: bool,
    research_question: int,
    control: ReportRun | None,
) -> str:
    if compact:
        raise ValueError("compact tables require an explicit research question")
    completed_runs = [run for run in runs if run.status == "completed"]
    tuning_columns = _tuned_columns(completed_runs)
    grouped_by_configuration: dict[tuple[str, ...], list[ReportRun]] = defaultdict(
        list
    )
    for run in completed_runs:
        visible_configuration = tuple(
            (
                run.method,
                *(
                    _format_report_value(_tuned_value(run, label, path))
                    for label, path in tuning_columns
                ),
            )
        )
        grouped_by_configuration[visible_configuration].append(run)
    report_runs = []
    control_is_grouped = False
    for visible_configuration, configuration_runs in grouped_by_configuration.items():
        configuration_runs = _without_repeated_launches(configuration_runs, control)
        grouped_control = control is not None and control in configuration_runs
        by_name = {run.name: run for run in configuration_runs}
        if len(by_name) != 1:
            candidates = list(by_name.values())
            alias_protocols = {
                json.dumps(
                    {
                        name: run.metadata.get(name)
                        for name in (
                            "training_semantics_revision",
                            "dataset_size",
                            "seed",
                            "batch_size",
                            "embedding_learning_rate",
                            "deep_learning_rate",
                            "num_epochs",
                            "max_epochs",
                            "transfer_invariants",
                        )
                    },
                    sort_keys=True,
                )
                for run in candidates
            }
            alias_stages = {
                _without_rate_suffix(run.configuration).rsplit("_", 1)[-1]
                for run in candidates
            }
            stage_precedence = {
                "initial": 0,
                "boundary": 0,
                "upperboundary": 0,
                "secondary": 1,
                "local": 2,
            }
            alias_provenance_matches = [
                re.search(
                    r"(?:_cap(?P<epochs>\d+))?"
                    rf"_ts{GENERATION_TRAINING_SEMANTICS_REVISION}"
                    r"_r(?P<revision>[1-9]\d*)$",
                    run.configuration,
                )
                for run in candidates
            ]
            alias_provenance = {
                (int(match.group("epochs") or 20), int(match.group("revision")))
                for match in alias_provenance_matches
                if match is not None
            }
            exact_alias = (
                len({_manifest_base(run.configuration) for run in candidates}) == 1
                and len(alias_protocols) == 1
                and all(match is not None for match in alias_provenance_matches)
                and len(alias_provenance) == 1
                and len(alias_stages) > 1
                and alias_stages <= set(stage_precedence)
            )
            if exact_alias:
                highest_precedence = max(stage_precedence[stage] for stage in alias_stages)
                preferred = [
                    run
                    for run in candidates
                    if stage_precedence[
                        _without_rate_suffix(run.configuration).rsplit("_", 1)[-1]
                    ]
                    == highest_precedence
                ]
                if len(preferred) != 1:
                    exact_alias = False
                else:
                    configuration_runs = preferred
            if exact_alias:
                by_name = {configuration_runs[0].name: configuration_runs[0]}
                if grouped_control:
                    control = configuration_runs[0]
                candidates = []
            if candidates:
                continued = _continued_to_its_longest_cap(candidates)
                if continued is None:
                    artifacts = ", ".join(sorted(by_name))
                    rendered = "/".join(visible_configuration)
                    raise ValueError(
                        f"ambiguous completed artifacts for reader configuration "
                        f"{rendered}: {artifacts}"
                    )
                configuration_runs = [continued]
                by_name = {continued.name: continued}
                if grouped_control:
                    control = continued
        if control is not None and control in configuration_runs:
            report_runs.append(control)
            control_is_grouped = True
            continue
        report_runs.append(next(iter(by_name.values())))
    if (
        control is not None
        and control.status == "completed"
        and not control_is_grouped
    ):
        report_runs.append(control)
    ranked = sorted(
        report_runs,
        key=lambda run: (
            run != control,
            -(_metric_value(run, "recall@100") or float("-inf")),
            run.configuration,
            run.name,
        ),
    )
    best = (
        None
        if report_runs
        and all(
            run.method == "homework-compatible baseline repeats"
            for run in report_runs
        )
        else _completed_best(report_runs)
    )
    bands = _metric_bands()
    cost_columns = COST_COLUMNS if research_question == 3 else []
    columns = [
        *(label for label, _ in tuning_columns),
        *(
            column
            for metric in REPORT_METRICS
            for column in (metric, f"{metric} vs control")
        ),
        *(label for label, _ in cost_columns),
    ]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for run in ranked:
        values = [
            _format_report_value(_tuned_value(run, label, path))
            for label, path in tuning_columns
        ]
        for metric in REPORT_METRICS:
            values.extend(
                [
                    _absolute_metric_cell(run, metric, bands),
                    _relative_percent_cell(run, control, metric, bands),
                ]
            )
        values.extend(
            _performance_cell(run, name, template) for name, template in cost_columns
        )
        if best is not None and run == best:
            values = [f"**{value}**" for value in values]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def render_full_tuning_report(dataset_size: str = "50m") -> str:
    runs = [
        run for run in load_report_runs(dataset_size) if run.status == "completed"
    ]
    grouped: dict[int, dict[str, list[ReportRun]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for run in runs:
        grouped[run.research_question][run.method].append(run)
    if dataset_size == "50m":
        grouped.pop(11, None)

    sections = [
        f"# G1 — full hyperparameter tuning on Yambda-{dataset_size.upper()}",
        "",
        "Generated from usable completed run directories and available metadata. "
        "The highest recall@100 displayed row in every method table is bold.",
    ]
    order = [rq for rq in REPORT_RQ_ORDER if rq in grouped]
    order.extend(sorted(set(grouped) - set(order)))
    for research_question in order:
        research_question_runs = [
            run
            for method_runs in grouped[research_question].values()
            for run in method_runs
        ]
        sections += [
            "",
            f"## RQ{research_question} — {REPORT_QUESTION_TITLES[research_question]}",
        ]
        for method, method_runs in sorted(grouped[research_question].items()):
            control = _ledger_control_run(
                research_question, method, research_question_runs
            )
            sections += [
                "",
                f"### {method}",
                "",
                _report_table(
                    method_runs,
                    compact=False,
                    research_question=research_question,
                    control=control,
                ),
            ]
    return "\n".join(sections).rstrip() + "\n"


def _archived_50m_tables() -> dict[int, str]:
    if not ARCHIVED_50M_QUESTIONS.exists():
        return {}
    tables = {}
    for line in ARCHIVED_50M_QUESTIONS.read_text().splitlines():
        cells = [cell.strip() for cell in line.split("|")[1:-1]]
        if len(cells) != 7 or not re.fullmatch(r"rq\d+", cells[0]):
            continue
        research_question = int(cells[0].removeprefix("rq"))
        tables[research_question] = "\n".join(
            [
                "| best configuration | recall@100 | vs baseline |",
                "| --- | --- | --- |",
                f"| {cells[3]} | {cells[4]} | {cells[5]} |",
            ]
        )
    return tables


def _curated_500m_tables() -> dict[int, str]:
    tables = {}
    for path in (CURATED_500M_QUESTIONS, READER_REPORT):
        if not path.exists():
            continue
        for title, section in _sections(path.read_text(), "## ").items():
            match = re.match(r"rq(\d+)\b", title, re.IGNORECASE)
            if match is None:
                continue
            research_question = int(match.group(1))
            blocks = re.findall(r"(?ms)^\| .+?\n\| ---.*?(?=\n\n|\Z)", section)
            if blocks and research_question not in tables:
                tables[research_question] = "\n\n".join(
                    _compact_markdown_table(block.rstrip()) for block in blocks
                )
    return tables


def _compact_markdown_table(table: str) -> str:
    rows = []
    keep = None
    for line in table.splitlines():
        cells = [cell.strip() for cell in line.split("|")[1:-1]]
        if keep is None:
            keep = [index for index, cell in enumerate(cells) if cell != "runs"]
        rows.append("| " + " | ".join(cells[index] for index in keep) + " |")
    return "\n".join(rows)


_CONTROL_CONFIGURATION_PATTERNS = {
    2: (
        "homework_baseline",
        "combination_baseline",
        "final_metrics_baseline",
        "baseline",
    ),
    3: ("homework_baseline", "performance_baseline", "balance_baseline", "baseline"),
    5: ("schedule_constant",),
    7: ("position_none",),
    8: ("architecture_control", "control_control"),
    9: ("time_none",),
    10: ("item_embeddings_shared",),
    11: ("neg_fixed_inbatch_leave_one_out",),
}

_RQ8_AXIS_SPECS = (
    (
        "Dimension",
        ("dimension_64",),
        (),
        (
            "dimension_16",
            "dimension_32",
            "dimension_64",
            "dimension_128",
            "dimension_256",
        ),
    ),
    ("Depth", ("depth_2",), (), ("depth_1", "depth_2", "depth_4")),
    (
        "MHA head count",
        ("heads_mha2",),
        (),
        ("heads_mha1", "heads_mha2", "heads_mha4", "heads_mha8"),
    ),
    (
        "Attention grouping",
        ("heads_mha2",),
        (),
        ("heads_mha2", "heads_gqa2q1kv"),
    ),
    (
        "Block normalization kind",
        ("normalization_layer_pre",),
        (),
        (
            "normalization_layer_pre",
            "normalization_rms",
            "normalization_batch",
        ),
    ),
    (
        "Residual normalization placement",
        ("normalization_layer_pre",),
        (),
        ("normalization_layer_pre", "normalization_post"),
    ),
    (
        "Input and final normalization",
        ("normalization_layer_pre",),
        (),
        (
            "normalization_layer_pre",
            "normalization_all_rms",
            "normalization_input_layer",
            "normalization_input_rms",
            "normalization_no_final",
        ),
    ),
    (
        "Attention window",
        ("window_50",),
        (),
        (
            "window_none",
            "window_10",
            "window_25",
            "window_50",
            "window_75",
            "window_100",
        ),
    ),
    (
        "Dropout",
        ("dropout_10",),
        (),
        (
            "dropout_0",
            "dropout_5",
            "dropout_10",
            "dropout_20",
            "dropout_30",
            "dropout_50",
        ),
    ),
    ("BOS", ("bos_off",), (), ("bos_off", "bos_on")),
)

_REQUIRED_COMPLETED_BASES = {
    2: {"combination_baseline", "selected_quality", "selected_balanced"},
    3: {"performance_baseline", "selected_quality", "selected_balanced"},
    4: {
        "ffn_gelu128",
        "ffn_gelu171",
        "ffn_gelu256",
        "ffn_gelu384",
        "ffn_swiglu16",
        "ffn_swiglu32",
        "ffn_swiglu64",
        "ffn_swiglu96",
        "ffn_swiglu128",
        "ffn_swiglu171",
        "ffn_swiglu224",
    },
    6: {
        "schedule_constant",
        "schedule_constant_warmup5",
        "schedule_cosine",
        "schedule_cosine_warmup5_cycles1",
        "schedule_inverse_sqrt",
        "schedule_inverse_sqrt_warmup5",
    },
    7: {
        "position_none",
        "position_learned_forward",
        "position_learned_reverse",
        "position_learned_forward_reverse",
        "position_rope",
        "position_rope_reverse",
        "position_alibi",
        "position_rope_alibi",
        "position_rope_reverse_alibi",
        "position_learned_alibi",
        "position_learned_reverse_alibi",
        "position_rope_learned",
        "position_rope_learned_reverse",
        "position_rope_reverse_learned",
        "position_rope_reverse_learned_reverse",
        "position_all",
        "position_rope_learned_reverse_alibi",
        "position_rope_reverse_learned_alibi",
        "position_reverse_all",
    },
    9: {
        "time_none",
        "time_raw_rope_forward",
        "time_raw_rope_reverse",
        "time_log_rope_forward",
        "time_log_rope_reverse",
        "time_plain_delta_add",
        "time_log_delta_add",
        "time_bins8_add",
        "time_bins16_add",
        "time_bins32_add",
        "time_bins64_add",
        "time_bins32_add_raw_rope_reverse",
        "time_log_delta_concat",
        "time_bins32_concat",
        "time_bins32_add_log_rope_forward",
    },
    10: {"item_embeddings_shared", "item_embeddings_per_layer"},
}

_RQ9_BASE_ORDER = (
    "time_none",
    "time_bins8_add",
    "time_bins16_add",
    "time_bins32_add",
    "time_bins64_add",
    "time_bins32_add_raw_rope_reverse",
    "time_bins32_add_log_rope_forward",
    "time_bins32_concat",
    "time_plain_delta_add",
    "time_log_delta_add",
    "time_log_delta_concat",
    "time_raw_rope_forward",
    "time_raw_rope_reverse",
    "time_log_rope_forward",
    "time_log_rope_reverse",
)

_REQUIRED_RQ11_METHODS = frozenset(_NEGATIVE_METHODS.values())
_HOMEWORK_RQ11_METHODS = frozenset(
    {
        "homework-matched fixed leave-one-out logQ",
        "homework-matched uniform random",
    }
)

_INITIAL_EMBEDDING_LRS = (0.008, 0.016, 0.032)
_INITIAL_DEEP_LRS = (0.003, 0.006, 0.012)
_COMMON_EMBEDDING_LR = 0.032
_INITIAL_BATCH_SIZE = 1280
_DEFAULT_NEGATIVE_COUNT = 512


def _rate_key(value: object) -> float | None:
    return round(float(value), 12) if value is not None else None


def _run_rates(run: ReportRun) -> tuple[float | None, float | None]:
    return (
        _rate_key(
            _tuned_value(run, "embedding learning rate", ("embedding_learning_rate",))
        ),
        _rate_key(_tuned_value(run, "deep learning rate", ("deep_learning_rate",))),
    )


def _run_batch_size(run: ReportRun) -> int | None:
    value = _nested_value(run.metadata, ("effective_batch_size",))
    if value is None:
        value = _nested_value(
            run.metadata, ("transfer_invariants", "effective_batch_size")
        )
    if value is None:
        value = _tuned_value(run, "batch size", ("batch_size",))
    return int(value) if value is not None else None


def _completed_runs(runs: list[ReportRun]) -> list[ReportRun]:
    return [
        run
        for run in runs
        if run.status == "completed" and _metric_value(run, "recall@100") is not None
    ]


def _require_rate_grid(
    label: str,
    runs: list[ReportRun],
    *,
    batch_size: int,
    embedding_lrs: tuple[float, ...] = _INITIAL_EMBEDDING_LRS,
    deep_lrs: tuple[float, ...] = _INITIAL_DEEP_LRS,
) -> None:
    present = {
        _run_rates(run)
        for run in _completed_runs(runs)
        if _run_batch_size(run) == batch_size
    }
    expected = {
        (_rate_key(embedding_lr), _rate_key(deep_lr))
        for embedding_lr in embedding_lrs
        for deep_lr in deep_lrs
    }
    missing = expected - present
    if missing:
        points = ", ".join(f"{embedding}/{deep}" for embedding, deep in sorted(missing))
        raise ValueError(f"{label}: missing batch-{batch_size} LR points {points}")


def _require_tuned_deep_line(
    label: str,
    runs: list[ReportRun],
    *,
    batch_size: int,
    embedding_lr: float = _COMMON_EMBEDDING_LR,
    deep_lrs: tuple[float, ...] = _INITIAL_DEEP_LRS,
) -> None:
    """An architecture treatment searches the deep rate at the common embedding rate.

    The embedding rate is held at the value that wins across the other arms, so
    treatments differ in the axis under study rather than in how far their own
    search happened to reach.
    """
    present = {
        deep
        for run in _completed_runs(runs)
        if _run_batch_size(run) == batch_size
        and (rates := _run_rates(run))[0] == _rate_key(embedding_lr)
        and (deep := rates[1]) is not None
    }
    missing = {_rate_key(rate) for rate in deep_lrs} - present
    if missing:
        points = ", ".join(f"{embedding_lr:g}/{rate:g}" for rate in sorted(missing))
        raise ValueError(f"{label}: missing batch-{batch_size} LR points {points}")


def _best_run(runs: list[ReportRun]) -> ReportRun:
    winner = _completed_best(runs)
    if winner is None:
        raise ValueError("no completed run with recall@100")
    return winner


def _brackets_its_winner(tested: set[float], winner: float) -> bool:
    """An axis held at one value by protocol has no boundary of its own to close."""
    return tested == {winner} or (
        any(rate < winner for rate in tested) and any(rate > winner for rate in tested)
    )


def _require_closed_lr_boundary(
    label: str,
    runs: list[ReportRun],
    *,
    axis_aligned: bool = False,
) -> ReportRun:
    completed = _completed_runs(runs)
    winner = _best_run(completed)
    embedding_lr, deep_lr = _run_rates(winner)
    if embedding_lr is None or deep_lr is None:
        raise ValueError(f"{label}: best run has no complete LR pair")
    if not axis_aligned:
        embedding_values = sorted(
            {rate for run in completed if (rate := _run_rates(run)[0]) is not None}
        )
        deep_values = sorted(
            {rate for run in completed if (rate := _run_rates(run)[1]) is not None}
        )
        if embedding_lr in {embedding_values[0], embedding_values[-1]} or deep_lr in {
            deep_values[0],
            deep_values[-1],
        }:
            raise ValueError(
                f"{label}: best LR {embedding_lr}/{deep_lr} remains on a tested boundary"
            )
        return winner
    rate_pairs = {_run_rates(run) for run in completed}
    embedding_neighbors = {
        candidate_embedding
        for candidate_embedding, candidate_deep in rate_pairs
        if candidate_deep == deep_lr and candidate_embedding is not None
    }
    deep_neighbors = {
        candidate_deep
        for candidate_embedding, candidate_deep in rate_pairs
        if candidate_embedding == embedding_lr and candidate_deep is not None
    }
    if not (
        _brackets_its_winner(embedding_neighbors, embedding_lr)
        and _brackets_its_winner(deep_neighbors, deep_lr)
    ):
        raise ValueError(
            f"{label}: best LR {embedding_lr}/{deep_lr} lacks an axis-aligned "
            "lower or upper neighbor"
        )
    return winner


def _control_proxy_winner(label: str, runs: list[ReportRun]) -> ReportRun:
    completed = _completed_runs(runs)
    initial = [run for run in completed if _run_batch_size(run) == _INITIAL_BATCH_SIZE]
    _require_rate_grid(label, initial, batch_size=_INITIAL_BATCH_SIZE)
    return _require_closed_lr_boundary(
        f"{label} initial grid", initial, axis_aligned=True
    )


def _architecture_bases() -> set[str]:
    bases = set().union(
        *(
            values
            for research_question, values in _REQUIRED_COMPLETED_BASES.items()
            if 4 <= research_question <= 10
        ),
        *(set(values) for _, _, _, values in _RQ8_AXIS_SPECS),
    )
    return bases


def _treatment_proxy_winner(
    label: str, runs: list[ReportRun], batch_size: int
) -> ReportRun:
    fixed_batch_runs = [
        run for run in _completed_runs(runs) if _run_batch_size(run) == batch_size
    ]
    _require_tuned_deep_line(label, fixed_batch_runs, batch_size=batch_size)
    return _require_closed_lr_boundary(
        f"{label} fixed-batch LR grid", fixed_batch_runs, axis_aligned=True
    )


def _architecture_proxy_selections(
    proxy_runs: list[ReportRun], bases: set[str]
) -> dict[str, ReportRun]:
    architecture_control_base = "sequence_128"
    control_bases = {architecture_control_base}
    control_winners = {}
    for control_base in control_bases:
        control_runs = [
            run
            for run in proxy_runs
            if _manifest_base(run.configuration) == control_base
        ]
        control_winners[control_base] = _control_proxy_winner(
            f"{control_base} batch control", control_runs
        )

    selections = {
        base: winner for base, winner in control_winners.items() if base in bases
    }
    for base in sorted(bases):
        control_base = architecture_control_base
        if base == control_base:
            continue
        selected_batch = _run_batch_size(control_winners[control_base])
        if selected_batch is None:
            raise ValueError(f"{control_base}: selected control has no batch size")
        treatment_runs = [
            run
            for run in proxy_runs
            if _manifest_base(run.configuration) == base
        ]
        selections[base] = _treatment_proxy_winner(
            base, treatment_runs, selected_batch
        )
    return selections


def select_architecture_report_runs(
    dataset_size: str,
    report_runs: list[ReportRun],
    proxy_runs: list[ReportRun],
    *,
    bases: set[str] | None = None,
) -> list[ReportRun]:
    selected_bases = _architecture_bases() if bases is None else bases
    proxy_selections = _architecture_proxy_selections(proxy_runs, selected_bases)
    selected_names = {}
    for base, proxy_winner in proxy_selections.items():
        if dataset_size == "50m":
            selected_names[base] = proxy_winner.name
            continue
        matches = [
            run
            for run in report_runs
            if run.status == "completed"
            and _manifest_base(run.configuration) == base
            and _run_rates(run) == _run_rates(proxy_winner)
            and _run_batch_size(run) == _run_batch_size(proxy_winner)
        ]
        by_name = {run.name: run for run in matches}
        confirmation = (
            next(iter(by_name.values()))
            if len(by_name) == 1
            else _continued_to_its_longest_cap(list(by_name.values()))
        )
        if confirmation is None:
            raise ValueError(
                f"{base}: expected one 500M confirmation of proxy winner "
                f"LR={_run_rates(proxy_winner)}, "
                f"control-selected batch={_run_batch_size(proxy_winner)}"
            )
        selected_names[base] = confirmation.name
    return [
        run
        for run in report_runs
        if run.status == "completed"
        and selected_names.get(_manifest_base(run.configuration)) == run.name
    ]


_RQ4_FFN_FAMILIES = {
    "GELU": ("ffn_gelu128", "ffn_gelu171", "ffn_gelu256", "ffn_gelu384"),
    "SwiGLU": (
        "ffn_swiglu16",
        "ffn_swiglu32",
        "ffn_swiglu64",
        "ffn_swiglu96",
        "ffn_swiglu128",
        "ffn_swiglu171",
        "ffn_swiglu224",
    ),
}


def _ffn_width(base: str, family: str) -> int | None:
    match = re.fullmatch(f"ffn_{family.lower()}(?P<width>\\d+)", base)
    return int(match.group("width")) if match is not None else None


def _rq4_proxy_selections(proxy_runs: list[ReportRun]) -> dict[str, ReportRun]:
    selected: dict[str, ReportRun] = {}
    for label, required_bases in _RQ4_FFN_FAMILIES.items():
        family = label.lower()
        observed_bases = set(required_bases) | {
            base
            for run in proxy_runs
            if run.status == "completed"
            if (base := _manifest_base(run.configuration))
            and _ffn_width(base, family) is not None
        }
        width_selections = _architecture_proxy_selections(
            proxy_runs, observed_bases
        )
        winner = _best_run(list(width_selections.values()))
        winner_base = _manifest_base(winner.configuration)
        winner_width = _ffn_width(winner_base, family)
        widths = sorted(
            width
            for base in observed_bases
            if (width := _ffn_width(base, family)) is not None
        )
        if winner_width in {widths[0], widths[-1]}:
            raise ValueError(
                f"RQ4 {label} width winner {winner_width} remains on the finite "
                "boundary; extend and close the family width search"
            )
        selected[winner_base] = winner
    return selected


def _rq4_bases() -> set[str]:
    return set().union(*(set(bases) for bases in _RQ4_FFN_FAMILIES.values()))


def select_rq4_report_runs(
    dataset_size: str,
    report_runs: list[ReportRun],
    proxy_runs: list[ReportRun],
) -> list[ReportRun]:
    proxy_selections = _rq4_proxy_selections(proxy_runs)
    if dataset_size == "50m":
        selected_names = {
            base: winner.name for base, winner in proxy_selections.items()
        }
    else:
        selected_names = {}
        for base, proxy_winner in proxy_selections.items():
            matches = [
                run
                for run in report_runs
                if run.status == "completed"
                and _manifest_base(run.configuration) == base
                and _run_rates(run) == _run_rates(proxy_winner)
                and _run_batch_size(run) == _run_batch_size(proxy_winner)
            ]
            by_name = {run.name: run for run in matches}
            confirmation = (
                next(iter(by_name.values()))
                if len(by_name) == 1
                else _continued_to_its_longest_cap(list(by_name.values()))
            )
            if confirmation is None:
                raise ValueError(
                    f"RQ4 {base}: expected one exact 500M confirmation of the "
                    "closed native-50M family winner"
                )
            selected_names[base] = confirmation.name
    return [
        run
        for run in report_runs
        if run.status == "completed"
        and selected_names.get(_manifest_base(run.configuration)) == run.name
    ]


def validate_architecture_final_selections(
    proxy_runs: list[ReportRun],
    requested_rates: dict[str, tuple[float, float]],
    *,
    exploratory_bases: frozenset[str] = frozenset(),
) -> dict[str, ReportRun]:
    """Resolve each requested base to the closed native-50M winner it transfers.

    An exploratory base still has to name its own closed winner, but it is exempt
    from the rule that an FFN transfer carries a family winner: it answers whether
    the proxy ranked the widths correctly, which is a question the winners alone
    cannot settle. The report selects only family winners, so such a run is
    evidence beside RQ4 rather than part of it.
    """
    requested_bases = set(requested_rates)
    unknown_exploratory = exploratory_bases - requested_bases
    if unknown_exploratory:
        raise ValueError(
            "exploratory bases without a selection: "
            + ", ".join(sorted(unknown_exploratory))
        )
    requested_ffn = {
        base
        for base in requested_bases - exploratory_bases
        if re.fullmatch(r"ffn_(?:gelu|swiglu)\d+", base)
    }
    selected: dict[str, ReportRun] = {}
    if requested_ffn:
        rq4_selections = _rq4_proxy_selections(proxy_runs)
        if requested_ffn != set(rq4_selections):
            raise ValueError(
                "RQ4 final selections must be exactly the two closed family winners: "
                + ", ".join(sorted(rq4_selections))
            )
        selected.update(rq4_selections)
    remaining = requested_bases - requested_ffn
    if remaining:
        selected.update(_architecture_proxy_selections(proxy_runs, remaining))
    for base, winner in selected.items():
        expected = tuple(_rate_key(value) for value in requested_rates[base])
        if _run_rates(winner) != expected:
            raise ValueError(
                f"{base}: selected LR {expected[0]}/{expected[1]} does not match "
                f"closed native-50M winner {_run_rates(winner)[0]}/"
                f"{_run_rates(winner)[1]} from {winner.name}"
            )
    return selected


def _validate_architecture_tuning(
    dataset_size: str,
    report_runs: list[ReportRun],
    proxy_runs: list[ReportRun],
) -> None:
    architecture_bases = _architecture_bases()
    if dataset_size == "500m":
        architecture_bases -= _rq4_bases()
    select_architecture_report_runs(
        dataset_size,
        report_runs,
        proxy_runs,
        bases=architecture_bases,
    )
    select_rq4_report_runs(dataset_size, report_runs, proxy_runs)


def _negative_semantics(
    run: ReportRun,
) -> tuple[int | None, int | None, float | None, float | None]:
    count = _tuned_value(
        run,
        "negative count",
        ("transfer_invariants", "num_in_batch_negatives"),
    )
    alpha = _tuned_value(
        run,
        "logQ alpha",
        ("transfer_invariants", "logq_alpha"),
    )
    random_fraction = _tuned_value(
        run,
        "random negative fraction",
        ("transfer_invariants", "random_negative_fraction"),
    )
    return (
        _run_batch_size(run),
        int(count) if count is not None else None,
        _rate_key(alpha),
        _rate_key(random_fraction),
    )


def _unique_negative_configurations(runs: list[ReportRun]) -> list[ReportRun]:
    grouped: dict[
        tuple[
            tuple[float | None, float | None],
            tuple[int | None, int | None, float | None, float | None],
        ],
        list[ReportRun],
    ] = defaultdict(list)
    for run in runs:
        grouped[(_run_rates(run), _negative_semantics(run))].append(run)
    return [min(group, key=lambda run: run.name) for group in grouped.values()]


def _negative_initial_lr_winner(
    label: str, method_runs: list[ReportRun], batch_size: int
) -> ReportRun:
    completed = _unique_negative_configurations(_completed_runs(method_runs))
    defaults = (
        batch_size,
        _DEFAULT_NEGATIVE_COUNT,
        _rate_key(0.01),
        _rate_key(0.5),
    )
    default_runs = [run for run in completed if _negative_semantics(run) == defaults]
    _require_rate_grid(
        f"{label} initial",
        default_runs,
        batch_size=batch_size,
        embedding_lrs=_INITIAL_EMBEDDING_LRS,
        deep_lrs=_INITIAL_DEEP_LRS,
    )
    return _require_closed_lr_boundary(f"{label} initial", default_runs)


def _require_axis_points(
    label: str,
    runs: list[ReportRun],
    expected: set[tuple[int, int, float, float]],
) -> None:
    present = {
        semantics
        for run in _completed_runs(runs)
        if None not in (semantics := _negative_semantics(run))
    }
    missing = expected - present
    if missing:
        points = ", ".join("/".join(map(str, point)) for point in sorted(missing))
        raise ValueError(f"{label}: missing secondary points {points}")


def _negative_proxy_winner(
    label: str, method_runs: list[ReportRun], batch_size: int
) -> ReportRun:
    completed = _unique_negative_configurations(_completed_runs(method_runs))
    initial_winner = _negative_initial_lr_winner(label, completed, batch_size)
    selected_rates = _run_rates(initial_winner)
    at_selected_lr = [run for run in completed if _run_rates(run) == selected_rates]

    expected = {
        (batch_size, count, _rate_key(0.01), _rate_key(0.5))
        for count in (512, 1024, 2048)
    }
    streaming = "streaming" in label
    mixed = label.startswith("uniform random +")
    if streaming:
        expected |= {
            (
                batch_size,
                _DEFAULT_NEGATIVE_COUNT,
                _rate_key(alpha),
                _rate_key(0.5),
            )
            for alpha in (0.0025, 0.005, 0.01, 0.02, 0.04)
        }
    if mixed:
        expected |= {
            (
                batch_size,
                _DEFAULT_NEGATIVE_COUNT,
                _rate_key(0.01),
                _rate_key(fraction),
            )
            for fraction in (0.125, 0.25, 0.5, 0.75, 0.875)
        }
    _require_axis_points(label, at_selected_lr, expected)
    secondary = [run for run in at_selected_lr if _negative_semantics(run) in expected]
    secondary_winner = _best_run(secondary)
    if _negative_semantics(secondary_winner)[1] == 2048:
        extension_semantics = (
            batch_size,
            4096,
            _rate_key(0.01),
            _rate_key(0.5),
        )
        _require_axis_points(label, at_selected_lr, {extension_semantics})
        secondary = [
            *secondary,
            *(run for run in at_selected_lr if _negative_semantics(run) == extension_semantics),
        ]
        secondary_winner = _best_run(secondary)
    selected_semantics = _negative_semantics(secondary_winner)
    local = [run for run in completed if _negative_semantics(run) == selected_semantics]
    selected_batch = selected_semantics[0]
    if selected_batch is None:
        raise ValueError(f"{label}: selected secondary configuration has no batch size")
    _require_rate_grid(f"{label} local", local, batch_size=selected_batch)
    return _require_closed_lr_boundary(f"{label} local", local)


def select_negative_report_runs(
    dataset_size: str,
    report_runs: list[ReportRun],
    proxy_runs: list[ReportRun],
    *,
    global_batch_size: int | None = None,
) -> list[ReportRun]:
    if global_batch_size is None:
        control_winner = _architecture_proxy_selections(
            proxy_runs, {"sequence_128"}
        )["sequence_128"]
        global_batch_size = _run_batch_size(control_winner)
    if global_batch_size is None:
        raise ValueError("global batch control has no batch size")
    selected_names = {}
    for method in sorted(_REQUIRED_RQ11_METHODS):
        proxy_method_runs = [
            run
            for run in proxy_runs
            if run.research_question == 11 and run.method == method
        ]
        winner = _negative_proxy_winner(
            method, proxy_method_runs, global_batch_size
        )
        if dataset_size == "50m":
            selected_names[method] = winner.name
            continue
        selected = [
            run
            for run in report_runs
            if run.research_question == 11
            and run.method == method
            and run.status == "completed"
            and _run_rates(run) == _run_rates(winner)
            and _negative_semantics(run) == _negative_semantics(winner)
        ]
        by_name = {run.name: run for run in selected}
        confirmation = (
            next(iter(by_name.values()))
            if len(by_name) == 1
            else _continued_to_its_longest_cap(list(by_name.values()))
        )
        if confirmation is None:
            raise ValueError(
                f"{method}: expected one 500M confirmation of the exact proxy winner"
            )
        selected_names[method] = confirmation.name
    return [
        run
        for run in report_runs
        if run.status == "completed"
        and run.research_question == 11
        and selected_names.get(run.method) == run.name
    ]


def _homework_continuation(run: ReportRun) -> tuple[int, int, str] | None:
    match = re.search(
        r"(?:_cap(?P<epochs>\d+))?"
        rf"_ts{GENERATION_TRAINING_SEMANTICS_REVISION}"
        r"_r(?P<revision>[1-9]\d*)$",
        run.configuration,
    )
    if match is None:
        return None
    return (
        int(match.group("epochs") or 20),
        int(match.group("revision")),
        _manifest_base(run.configuration),
    )


def _latest_homework_configuration(
    label: str, runs: list[ReportRun]
) -> ReportRun:
    lineages = [(run, _homework_continuation(run)) for run in runs]
    if any(lineage is None for _, lineage in lineages):
        raise ValueError(f"{label}: ambiguous artifacts")
    ordered = sorted(lineages, key=lambda item: item[1][:2])
    epochs = [lineage[0] for _, lineage in ordered]
    revisions = [lineage[1] for _, lineage in ordered]
    tags = {lineage[2] for _, lineage in ordered}
    normalized_tags = {
        tag.replace("_capcontinue", "") for tag in tags
    }
    expected_epochs = [20]
    while expected_epochs[-1] < epochs[-1]:
        expected_epochs.append(expected_epochs[-1] * 2)
    expected_revisions = list(range(1, len(expected_epochs) + 1))
    predecessors = [run for run, _ in ordered[:-1]]
    latest = ordered[-1][0]
    if (
        epochs != expected_epochs
        or revisions != expected_revisions
        or len(normalized_tags) != 1
        or latest.status != "completed"
        or any(
            run.status != "unusable"
            or run.metadata.get("selection_resolved") is not False
            or run.metadata.get("stopped_epoch") != run.metadata.get("max_epochs")
            for run in predecessors
        )
    ):
        artifacts = ", ".join(sorted(run.name for run in runs))
        raise ValueError(f"{label}: invalid cap lineage: {artifacts}")
    return latest


def _homework_proxy_winner(method: str, runs: list[ReportRun]) -> ReportRun:
    candidates = [
        run
        for run in runs
        if run.research_question == 11
        and run.method == method
        and _run_batch_size(run) == _INITIAL_BATCH_SIZE
    ]
    by_rates: dict[tuple[float | None, float | None], list[ReportRun]] = defaultdict(
        list
    )
    for run in candidates:
        by_rates[_run_rates(run)].append(run)
    resolved = [
        _latest_homework_configuration(f"{method} at {rates}", candidates)
        for rates, candidates in by_rates.items()
    ]
    _require_rate_grid(
        method,
        resolved,
        batch_size=_INITIAL_BATCH_SIZE,
        embedding_lrs=(0.0005, 0.001, 0.002),
        deep_lrs=(0.001, 0.002, 0.004),
    )
    return _require_closed_lr_boundary(method, resolved, axis_aligned=True)


def select_homework_negative_controls(
    dataset_size: str,
    report_runs: list[ReportRun],
    proxy_runs: list[ReportRun],
) -> list[ReportRun]:
    selected = []
    for method in sorted(_HOMEWORK_RQ11_METHODS):
        winner = _homework_proxy_winner(method, proxy_runs)
        if dataset_size == "50m":
            selected.append(winner)
            continue
        confirmations = [
            run
            for run in report_runs
            if run.research_question == 11
            and run.method == method
            and run.status == "completed"
            and _run_rates(run) == _run_rates(winner)
            and _run_batch_size(run) == _run_batch_size(winner)
        ]
        names = {run.name for run in confirmations}
        if len(names) != 1:
            raise ValueError(
                f"{method}: expected one native 500M confirmation of the "
                "exact proxy winner"
            )
        selected.append(confirmations[0])
    return selected


def _validate_negative_tuning(
    dataset_size: str,
    report_runs: list[ReportRun],
    proxy_runs: list[ReportRun],
) -> None:
    select_negative_report_runs(dataset_size, report_runs, proxy_runs)


def _rq1_width_evidence(
    proxy_runs: list[ReportRun],
) -> tuple[tuple[float | None, float | None], list[tuple[int, ReportRun, ReportRun]]]:
    width_runs = [
        run
        for run in proxy_runs
        if run.method == "μP model-width transfer"
        and run.status == "completed"
        and _run_batch_size(run) == _INITIAL_BATCH_SIZE
    ]
    by_width: dict[int, list[ReportRun]] = defaultdict(list)
    for run in width_runs:
        match = re.fullmatch(
            r"mup_dim(?P<width>\d+)", _manifest_base(run.configuration)
        )
        if match is not None:
            width = int(match.group("width"))
            expected_metadata = {
                ("transfer_invariants", "experiment_class"): (
                    "MuTransferGenerationExperiment"
                ),
                ("transfer_invariants", "mup_base_dim"): 16,
                ("transfer_invariants", "mup_delta_dim"): 32,
                ("item_embedding_dim",): 64,
                ("model_dim",): width,
            }
            mismatches = [
                ".".join(path)
                for path, expected in expected_metadata.items()
                if _nested_value(run.metadata, path) != expected
            ]
            if mismatches:
                raise ValueError(
                    f"{run.name}: invalid RQ1 width-transfer metadata: "
                    + ", ".join(mismatches)
                )
            by_width[width].append(run)
    expected_widths = (16, 32, 64, 128, 256)
    if set(by_width) != set(expected_widths):
        missing = ", ".join(
            str(width) for width in sorted(set(expected_widths) - set(by_width))
        )
        raise ValueError(f"RQ1 width transfer is missing dimensions: {missing}")
    oracles = {}
    for width in expected_widths:
        width_label = f"RQ1 width {width}"
        _require_rate_grid(width_label, by_width[width], batch_size=_INITIAL_BATCH_SIZE)
        oracles[width] = _require_closed_lr_boundary(
            width_label,
            by_width[width],
            axis_aligned=True,
        )
    width32_oracle = oracles[32]
    selected_rates = _run_rates(width32_oracle)
    comparisons = []
    for width in expected_widths:
        selected = {
            run.name: run
            for run in by_width[width]
            if _run_rates(run) == selected_rates
        }
        unchanged = (
            next(iter(selected.values()))
            if len(selected) == 1
            else _continued_to_its_longest_cap(list(selected.values()))
        )
        if unchanged is None:
            raise ValueError(
                f"RQ1 width {width} does not have exactly one unchanged selected-LR run"
            )
        comparisons.append((width, unchanged, oracles[width]))
    return selected_rates, comparisons


def _transfer_delta_cell(
    common: ReportRun, local: ReportRun, metric: str, bands: dict[str, float]
) -> str:
    if common.name == local.name:
        return SELF_COMPARISON
    return _relative_metric_cell(common, local, metric, bands)


def _rq1_width_table(
    proxy_runs: list[ReportRun], target_runs: list[ReportRun] | None = None
) -> str:
    selected_rates, comparisons = _rq1_width_evidence(proxy_runs)
    bands = _metric_bands(required=True)
    if target_runs is not None:
        confirmations = []
        for width, proxy_selected, proxy_oracle in comparisons:
            width_runs = [
                run
                for run in target_runs
                if run.status == "completed"
                and run.method == "μP model-width transfer"
                and _manifest_base(run.configuration) == f"mup_dim{width}"
                and _run_batch_size(run) == _run_batch_size(proxy_selected)
            ]

            def exact_confirmation(
                rates: tuple[float | None, float | None], label: str
            ) -> ReportRun:
                matches = {
                    run.name: run
                    for run in width_runs
                    if _run_rates(run) == rates
                }
                confirmation = (
                    next(iter(matches.values()))
                    if len(matches) == 1
                    else _continued_to_its_longest_cap(list(matches.values()))
                )
                if confirmation is None:
                    raise ValueError(
                        f"RQ1 width {width} requires exactly one native 500M "
                        f"confirmation of the {label} rates"
                    )
                return confirmation

            common = exact_confirmation(selected_rates, "common")
            local_rates = _run_rates(proxy_oracle)
            local = exact_confirmation(local_rates, "50M-local")
            confirmations.append((width, common, local_rates, local))
        columns = [
            "width",
            "common LR",
            "50M-local LR",
            "recall@100 vs local",
            "local recall@100",
            "ndcg@100 vs local",
            "local ndcg@100",
        ]
        lines = [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join("---" for _ in columns) + " |",
        ]
        for width, common, local_rates, local in confirmations:
            cells = [
                str(width),
                f"{selected_rates[0]}/{selected_rates[1]}",
                f"{local_rates[0]}/{local_rates[1]}",
                _transfer_delta_cell(common, local, "recall@100", bands),
                _absolute_metric_cell(local, "recall@100", bands),
                _transfer_delta_cell(common, local, "ndcg@100", bands),
                _absolute_metric_cell(local, "ndcg@100", bands),
            ]
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)
    columns = [
        "width",
        "common LR",
        "50M-local LR",
        "common recall@100",
        "local recall@100",
        "recall regret",
    ]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for width, selected, oracle in comparisons:
        selected_recall = _metric_value(selected, "recall@100")
        oracle_recall = _metric_value(oracle, "recall@100")
        if selected.name == oracle.name:
            regret = SELF_COMPARISON
        elif selected_recall is None or oracle_recall in {None, 0}:
            regret = "—"
        else:
            regret = f"{100 * (oracle_recall - selected_recall) / oracle_recall:.0f}%"
        local_rates = _run_rates(oracle)
        cells = [
            str(width),
            f"{selected_rates[0]}/{selected_rates[1]}",
            f"{local_rates[0]}/{local_rates[1]}",
            _absolute_metric_cell(selected, "recall@100", bands),
            _absolute_metric_cell(oracle, "recall@100", bands),
            regret,
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _rq1_native_dataset_size_evidence(
    proxy_runs: list[ReportRun], target_runs: list[ReportRun]
) -> tuple[str, ReportRun, ReportRun] | None:
    target_candidates = [
        run
        for run in target_runs
        if re.match(r"selected_native50_[0-9a-f]{12}_", run.configuration)
    ]
    if not target_candidates:
        return None
    completed_targets = [
        run for run in target_candidates if run.status == "completed"
    ]
    if len({run.name for run in completed_targets}) != 1:
        raise ValueError(
            "RQ1 dataset-size transfer requires exactly one completed native "
            "500M confirmation"
        )
    target = completed_targets[0]
    match = re.match(
        r"selected_native50_(?P<source>[0-9a-f]{12})_", target.configuration
    )
    if match is None:
        raise AssertionError("selected native target has no source id")
    selection = _current_native_selection()
    source_id = selection["source_id"]
    winner_run = selection["winner_run"]
    if match.group("source") != source_id:
        raise ValueError(
            f"{target.name}: selected native source id does not match the "
            "current approved selector result"
        )
    rates = _run_rates(target)
    batch_size = _run_batch_size(target)
    if batch_size is None or any(rate is None for rate in rates):
        raise ValueError(f"{target.name}: invalid RQ1 rates or batch size")
    selected_rates = (
        float(selection["embedding_lr"]),
        float(selection["deep_lr"]),
    )
    if rates != selected_rates:
        raise ValueError(f"{target.name}: rates do not match the native selector")
    native_500m_provenance.validate(
        GENERATED / "logs" / target.name,
        selection=selection,
    )
    proxies = [
        run
        for run in proxy_runs
        if run.status == "completed"
        and run.name == winner_run
        and _run_rates(run) == rates
        and _run_batch_size(run) == batch_size
    ]
    if len({run.name for run in proxies}) != 1:
        raise ValueError(
            "RQ1 dataset-size transfer requires exactly one completed native "
            "50M source at the confirmed rates"
        )
    proxy = proxies[0]
    for run in (proxy, target):
        if any(
            _metric_value(run, metric) is None
            for metric in ("recall@100", "ndcg@100")
        ):
            raise ValueError(f"{run.name}: incomplete RQ1 quality metrics")
        best_epoch = _nested_value(run.metadata, ("best_epoch",))
        stopped_epoch = _nested_value(run.metadata, ("stopped_epoch",))
        max_epochs = _nested_value(run.metadata, ("max_epochs",))
        if not (
            isinstance(best_epoch, int)
            and not isinstance(best_epoch, bool)
            and isinstance(stopped_epoch, int)
            and not isinstance(stopped_epoch, bool)
            and isinstance(max_epochs, int)
            and not isinstance(max_epochs, bool)
            and 1 <= best_epoch <= stopped_epoch < max_epochs
        ):
            raise ValueError(f"{run.name}: invalid RQ1 stopping metadata")
    return selection["source_digest"], proxy, target


def _rq1_native_dataset_size_table(
    proxy_runs: list[ReportRun], target_runs: list[ReportRun]
) -> str | None:
    evidence = _rq1_native_dataset_size_evidence(proxy_runs, target_runs)
    if evidence is None:
        return None
    _, proxy, target = evidence
    validate_homework_reproduction_runs([target])
    bands = _metric_bands()
    columns = [
        "dataset",
        "batch size",
        "embedding LR",
        "deep LR",
        "best/stopped epoch",
        "epoch cap",
        "recall@100",
        "ndcg@100",
    ]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for run in (proxy, target):
        embedding_lr, deep_lr = _run_rates(run)
        values = [
            run.dataset_size.upper(),
            str(_run_batch_size(run)),
            _format_report_value(embedding_lr),
            _format_report_value(deep_lr),
            f"{run.metadata['best_epoch']}/{run.metadata['stopped_epoch']}",
            str(run.metadata["max_epochs"]),
            _absolute_metric_cell(run, "recall@100", bands),
            _absolute_metric_cell(run, "ndcg@100", bands),
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _has_complete_rq1_width_evidence(runs: list[ReportRun]) -> bool:
    widths = set()
    for run in runs:
        if run.status != "completed" or run.method != "μP model-width transfer":
            continue
        match = re.fullmatch(
            r"mup_dim(?P<width>\d+)", _manifest_base(run.configuration)
        )
        if match is not None:
            widths.add(int(match.group("width")))
    return widths == {16, 32, 64, 128, 256}


def _rq1_compact_tables(dataset_size: str, runs: list[ReportRun]) -> str:
    proxy_runs = (
        runs
        if dataset_size == "50m"
        else load_report_runs("50m", research_question=1)
    )
    target_runs = (
        runs
        if dataset_size == "500m"
        else load_report_runs("500m", research_question=1)
    )
    sections: list[str] = []
    if table := _rq1_native_dataset_size_table(proxy_runs, target_runs):
        sections.append(table)
    if _has_complete_rq1_width_evidence(proxy_runs):
        if sections:
            sections.append("")
        sections.append(
            _rq1_width_table(
                proxy_runs,
                target_runs if dataset_size == "500m" else None,
            )
        )
    if not sections:
        raise ValueError("RQ1 has no completed dataset-size or model-width evidence")
    return "\n".join(sections)


def _method_winners(research_question: int, runs: list[ReportRun]) -> list[ReportRun]:
    grouped: dict[str, list[ReportRun]] = defaultdict(list)
    for run in runs:
        key = run.method
        if research_question in {3, 5}:
            key = _manifest_base(run.configuration)
        if research_question == 8 and _without_rate_suffix(
            run.configuration
        ).startswith("ffn_"):
            key = _without_rate_suffix(run.configuration)
        if research_question == 6:
            key += ":" + _without_rate_suffix(run.configuration)
        grouped[key].append(run)
    winners = []
    for method_runs in grouped.values():
        winner = _aggregate_homework_baseline(method_runs)
        if winner is None:
            winner = _completed_best(method_runs)
        if winner is None:
            winner = max(
                method_runs,
                key=lambda run: (_status_rank(run.status), run.configuration),
            )
        winners.append(winner)
    return sorted(winners, key=lambda run: (run.method, run.configuration))


def _control_run(
    research_question: int,
    winners: list[ReportRun],
    patterns: tuple[str, ...] | None = None,
) -> ReportRun | None:
    completed = [run for run in winners if run.status == "completed"]
    if research_question == 1:
        direct = [run for run in completed if run.method == "direct 500M LR oracle"]
        return _completed_best(direct)
    if research_question == 4:
        gelu = [
            run
            for run in completed
            if _manifest_base(run.configuration)
            in {"ffn_gelu128", "ffn_gelu171", "ffn_gelu256", "ffn_gelu384"}
        ]
        return _completed_best(gelu)
    if research_question == 11:
        return _completed_best(
            [
                run
                for run in completed
                if run.method == "fixed in-batch leave-one-out logQ"
            ]
        )
    selected_patterns = (
        patterns
        if patterns is not None
        else _CONTROL_CONFIGURATION_PATTERNS.get(research_question, ())
    )
    for pattern in selected_patterns:
        matches = [
            run for run in completed if pattern == _manifest_base(run.configuration)
        ]
        if matches:
            return _completed_best(matches)
    explicit = [run for run in completed if "control" in run.method]
    return _completed_best(explicit)


def _selected_lr(run: ReportRun) -> str:
    embedding, deep = _run_rates(run)
    if embedding is None or deep is None:
        return "—"
    return f"{embedding:g}/{deep:g}"


def _axis_value(configuration: str, axis_label: str) -> str:
    base = _manifest_base(configuration)
    if axis_label == "dimension":
        return base.removeprefix("dimension_")
    if axis_label == "depth":
        return base.removeprefix("depth_") + " layers"
    if axis_label == "sequence length":
        return base.removeprefix("sequence_")
    if axis_label == "attention window":
        return base.removeprefix("window_").replace("none", "full")
    if axis_label == "dropout":
        return {
            "dropout_0": "0.0",
            "dropout_5": "0.05",
            "dropout_10": "0.1",
            "dropout_20": "0.2",
            "dropout_30": "0.3",
            "dropout_50": "0.5",
        }.get(base, base)
    if axis_label == "mha head count":
        count = base.removeprefix("heads_mha")
        return f"{count}Q/{count}KV"
    if axis_label == "attention grouping":
        return {"heads_mha2": "MHA 2Q/2KV", "heads_gqa2q1kv": "GQA 2Q/1KV"}.get(
            base, base
        )
    if axis_label in {"ffn capacity", "configuration"} and base.startswith("ffn_"):
        return (
            base.removeprefix("ffn_")
            .replace("gelu", "GELU-")
            .replace("swiglu", "SwiGLU-")
        )
    if axis_label == "block normalization kind":
        return {
            "normalization_layer_pre": "LayerNorm",
            "normalization_rms": "RMSNorm",
            "normalization_batch": "BatchNorm",
        }.get(base, base)
    if axis_label == "residual normalization placement":
        return {
            "normalization_layer_pre": "pre-LayerNorm",
            "normalization_post": "post-LayerNorm",
        }.get(base, base)
    if axis_label == "input and final normalization":
        return {
            "normalization_layer_pre": "no input + final LayerNorm",
            "normalization_all_rms": "input + final RMSNorm",
            "normalization_input_layer": "input LayerNorm + final LayerNorm",
            "normalization_input_rms": "input RMSNorm + final LayerNorm",
            "normalization_no_final": "no input or final norm",
        }.get(base, base)
    if axis_label in {"bos", "cls query"}:
        return "enabled" if base.endswith("_on") else "disabled"
    if axis_label == "item embeddings":
        return {
            "item_embeddings_shared": "shared table",
            "item_embeddings_per_layer": "per-layer tables",
        }.get(base, base)
    if axis_label == "time representation":
        return {
            "time_none": "no time feature",
            "time_raw_rope_forward": "raw elapsed-time RoPE, forward",
            "time_raw_rope_reverse": "raw elapsed-time RoPE, reverse",
            "time_log_rope_forward": "log elapsed-time RoPE, forward",
            "time_log_rope_reverse": "log elapsed-time RoPE, reverse",
            "time_plain_delta_add": "clipped linear delta, add",
            "time_log_delta_add": "log delta, add",
            "time_bins8_add": "8 log-spaced bins, add",
            "time_bins16_add": "16 log-spaced bins, add",
            "time_bins32_add": "32 log-spaced bins, add",
            "time_bins64_add": "64 log-spaced bins, add",
            "time_bins32_add_raw_rope_reverse": "32 bins + raw reverse RoPE",
            "time_log_delta_concat": "log delta, concatenate-and-project",
            "time_bins32_concat": "32 bins, concatenate-and-project",
            "time_bins32_add_log_rope_forward": "32 bins + log forward RoPE",
        }.get(base, base)
    if axis_label == "scheduler":
        return base.removeprefix("schedule_").replace("_", " ")
    if axis_label.startswith("cosine cycles"):
        match = re.search(r"cycles(\d+)", base)
        return match.group(1) if match else base
    return base.replace("_", " ")


def _reader_report_table(
    research_question: int,
    runs: list[ReportRun],
    *,
    control_patterns: tuple[str, ...] | None = None,
    axis_label: str | None = None,
    metrics: tuple[str, ...] = REPORT_METRICS,
    show_selected_lr: bool = False,
    show_selected_batch: bool = False,
    configuration_order: tuple[str, ...] = (),
    cost_columns: tuple[tuple[str, str], ...] = (),
) -> str:
    bands = _metric_bands(required=True)
    winners = _method_winners(research_question, runs)
    unresolved = [run.method for run in winners if run.status != "completed"]
    if unresolved:
        methods = ", ".join(sorted(unresolved))
        raise ValueError(
            f"RQ{research_question} has no completed evidence for: {methods}"
        )
    control = _control_run(research_question, winners, control_patterns)
    if control is None:
        raise ValueError(f"RQ{research_question} has no explicit control")
    control_names = {control.name}
    overall = _completed_best(winners)
    columns = [
        axis_label or "method",
        *(("selected LR",) if show_selected_lr else ()),
        *(("selected batch",) if show_selected_batch else ()),
        *metrics,
        *(label for label, _ in cost_columns),
    ]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    if configuration_order:
        order = {
            configuration: index
            for index, configuration in enumerate(configuration_order)
        }
        ordered = sorted(
            winners,
            key=lambda run: (
                order.get(_manifest_base(run.configuration), len(order)),
                run.method,
                run.configuration,
            ),
        )
    else:
        ordered = sorted(
            winners,
            key=lambda run: (
                run.name not in control_names,
                run.method,
                run.configuration,
            ),
        )
    for run in ordered:
        display = (
            _axis_value(run.configuration, axis_label)
            if axis_label is not None
            else run.method
        )
        if overall is not None and run.name == overall.name:
            display = f"**{display}**"
        cells = [display]
        if show_selected_lr:
            cells.append(_selected_lr(run))
        if show_selected_batch:
            cells.append(_format_report_value(_run_batch_size(run)))
        cells.extend(
            _relative_metric_cell(run, control, metric, bands) for metric in metrics
        )
        cells.extend(
            _performance_cell(run, name, template) for name, template in cost_columns
        )
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _ffn_family_width(base: str) -> tuple[str, int]:
    for family in ("GELU", "SwiGLU"):
        width = _ffn_width(base, family.lower())
        if width is not None:
            return family, width
    raise ValueError(f"invalid FFN capacity base: {base}")


def _rq4_table(
    runs: list[ReportRun], candidates: list[ReportRun] | None = None
) -> str:
    winners = _method_winners(4, runs)
    by_family = {
        _ffn_family_width(_manifest_base(run.configuration))[0]: run
        for run in winners
        if run.status == "completed"
    }
    if set(by_family) != {"GELU", "SwiGLU"}:
        raise ValueError("RQ4 requires one completed selected width per FFN family")
    control = by_family["GELU"]
    overall = _completed_best(list(by_family.values()))
    bands = _metric_bands(required=True)
    lines = [
        "| proxy-selected FFN family | selected width | recall@100 | ndcg@100 "
        "| coverage@100 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for family in ("GELU", "SwiGLU"):
        run = by_family[family]
        width = str(_ffn_family_width(_manifest_base(run.configuration))[1])
        display = (
            f"**{family}**"
            if overall is not None and run.name == overall.name
            else family
        )
        cells = [display, width]
        cells.extend(
            _relative_metric_cell(run, control, metric, bands)
            for metric in ("recall@100", "ndcg@100", "coverage@100")
        )
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _rq4_tables(
    dataset_size: str,
    runs: list[ReportRun],
    candidates: list[ReportRun] | None = None,
) -> str:
    if dataset_size == "50m":
        return _rq4_table(runs, candidates)
    activation_depth = rq4_activation_depth.load_runs(
        GENERATED, report_file_facts(GENERATED)
    )
    if not activation_depth:
        raise ValueError("RQ4 activation/depth surface is absent")
    return rq4_activation_depth.reader_tables(activation_depth)


def _rq5_tables(runs: list[ReportRun]) -> str:
    main_bases = {
        "schedule_constant",
        "schedule_linear",
        "schedule_cosine",
        "schedule_polynomial",
        "schedule_exponential",
        "schedule_wsd_warmup5",
        "schedule_step",
        "schedule_inverse_sqrt",
    }
    cycle_bases = {
        "schedule_cosine_warmup5_cycles1",
        "schedule_cosine_warmup5_cycles2",
        "schedule_cosine_warmup5_cycles4",
    }
    main = [run for run in runs if _manifest_base(run.configuration) in main_bases]
    cycles = [run for run in runs if _manifest_base(run.configuration) in cycle_bases]
    return "\n\n".join(
        (
            _reader_report_table(
                5,
                main,
                control_patterns=("schedule_constant",),
                axis_label="scheduler",
                metrics=("recall@100", "ndcg@100"),
            ),
            _reader_report_table(
                5,
                cycles,
                control_patterns=("schedule_cosine_warmup5_cycles1",),
                axis_label="cosine cycles, warmup 5%",
                metrics=("recall@100", "ndcg@100"),
            ),
        )
    )


def _rq6_table(runs: list[ReportRun]) -> str:
    winners = _method_winners(6, runs)
    by_base = {_manifest_base(run.configuration): run for run in winners}
    pairs = (
        ("constant", "schedule_constant", "schedule_constant_warmup5"),
        ("cosine", "schedule_cosine", "schedule_cosine_warmup5_cycles1"),
        ("inverse sqrt", "schedule_inverse_sqrt", "schedule_inverse_sqrt_warmup5"),
    )
    bands = _metric_bands(required=True)
    lines = [
        "| schedule | no-warmup LR | warmup LR | no-warmup recall@100 | warmup=5% recall@100 | no-warmup ndcg@100 | warmup=5% ndcg@100 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    best_warmup = _completed_best(
        [by_base[warmup_base] for _, _, warmup_base in pairs if warmup_base in by_base]
    )
    for label, control_base, warmup_base in pairs:
        try:
            control, warmup = by_base[control_base], by_base[warmup_base]
        except KeyError as error:
            raise ValueError(f"RQ6 is incomplete: {error.args[0]}") from error
        cells = [
            label,
            _selected_lr(control),
            _selected_lr(warmup),
            _absolute_metric_cell(control, "recall@100", bands),
            _relative_metric_cell(warmup, control, "recall@100", bands),
            _absolute_metric_cell(control, "ndcg@100", bands),
            _relative_metric_cell(warmup, control, "ndcg@100", bands),
        ]
        if best_warmup is not None and warmup.name == best_warmup.name:
            cells = [f"**{cell}**" for cell in cells]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _rq10_table(runs: list[ReportRun]) -> str:
    return _reader_report_table(
        10,
        runs,
        axis_label="item embeddings",
        metrics=("recall@100", "ndcg@100"),
    )


def _rq9_table(runs: list[ReportRun]) -> str:
    current = [
        run
        for run in runs
        if _manifest_base(run.configuration) in _REQUIRED_COMPLETED_BASES[9]
    ]
    return _reader_report_table(
        9,
        current,
        axis_label="time representation",
        configuration_order=_RQ9_BASE_ORDER,
    )


def _rq11_table(runs: list[ReportRun]) -> str:
    family_runs = [run for run in runs if run.method in _REQUIRED_RQ11_METHODS]
    winners = _method_winners(11, family_runs)
    unresolved = [run.method for run in winners if run.status != "completed"]
    if unresolved:
        raise ValueError("RQ11 has no completed evidence for: " + ", ".join(unresolved))
    control = _control_run(11, winners)
    if control is None:
        raise ValueError("RQ11 has no explicit control")
    bands = _metric_bands(required=True)
    overall = _completed_best(winners)
    columns = [
        "negative sampling",
        "negatives",
        "logQ alpha",
        "random fraction",
        *REPORT_METRICS,
    ]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for run in sorted(winners, key=lambda candidate: candidate.name != control.name):
        label = run.method
        if overall is not None and run.name == overall.name:
            label = f"**{label}**"
        cells = [
            label,
            _format_report_value(
                _tuned_value(
                    run,
                    "negative count",
                    ("transfer_invariants", "num_in_batch_negatives"),
                )
            ),
            (
                _format_report_value(
                    _tuned_value(
                        run,
                        "logQ alpha",
                        ("transfer_invariants", "logq_alpha"),
                    )
                )
                if run.method in _STREAMING_LOGQ_METHODS
                else "—"
            ),
            (
                _format_report_value(
                    _tuned_value(
                        run,
                        "random negative fraction",
                        ("transfer_invariants", "random_negative_fraction"),
                    )
                )
                if run.method.startswith("uniform random +")
                else "—"
            ),
        ]
        cells.extend(
            _relative_metric_cell(run, control, metric, bands)
            for metric in REPORT_METRICS
        )
        lines.append("| " + " | ".join(cells) + " |")
    family_table = "\n".join(lines)
    homework_runs = [run for run in runs if run.method in _HOMEWORK_RQ11_METHODS]
    if not homework_runs:
        return family_table
    if {run.method for run in homework_runs} != set(_HOMEWORK_RQ11_METHODS):
        raise ValueError("RQ11 homework-matched control is incomplete")
    homework_control = next(
        run
        for run in homework_runs
        if run.method == "homework-matched fixed leave-one-out logQ"
    )
    homework_best = _completed_best(homework_runs)
    homework_columns = ["homework-matched objective", *REPORT_METRICS]
    homework_lines = [
        "| " + " | ".join(homework_columns) + " |",
        "| " + " | ".join("---" for _ in homework_columns) + " |",
    ]
    for run in sorted(homework_runs, key=lambda item: item.name != homework_control.name):
        label = run.method.removeprefix("homework-matched ")
        if homework_best is not None and run.name == homework_best.name:
            label = f"**{label}**"
        cells = [label]
        cells.extend(
            _relative_metric_cell(run, homework_control, metric, bands)
            for metric in REPORT_METRICS
        )
        homework_lines.append("| " + " | ".join(cells) + " |")
    return f"{family_table}\n\n" + "\n".join(homework_lines)


def _rq8_compact_tables(runs: list[ReportRun]) -> str:
    tables = []
    for title, controls, prefixes, bases in _RQ8_AXIS_SPECS:
        axis_runs = [
            run
            for run in runs
            if _manifest_base(run.configuration) in bases
        ]
        if not axis_runs:
            raise ValueError(f"RQ8 has no generated evidence for {title}")
        completed_bases = {
            _manifest_base(run.configuration)
            for run in axis_runs
            if run.status == "completed"
        }
        missing = set(bases) - completed_bases
        if missing:
            labels = ", ".join(sorted(missing))
            raise ValueError(f"RQ8 {title} is incomplete: {labels}")
        tables.append(
            _reader_report_table(
                8,
                axis_runs,
                control_patterns=controls,
                axis_label=title.lower(),
                metrics=("recall@100", "ndcg@100"),
            )
        )
    return "\n\n".join(tables)


_RQ2_RQ3_CANDIDATES = (
    "performance_baseline",
    "selected_quality",
    "selected_balanced",
)

_SELECTED_PARAMETER_PATHS = (
    ("dataset", ("dataset_size",)),
    ("seed", ("seed",)),
    ("experiment class", ("transfer_invariants", "experiment_class")),
    ("μP base dim", ("transfer_invariants", "mup_base_dim")),
    ("μP delta dim", ("transfer_invariants", "mup_delta_dim")),
    ("invariant dataset", ("transfer_invariants", "dataset_size")),
    ("user sample", ("transfer_invariants", "user_sample")),
    ("event filter", ("transfer_invariants", "event_type_filter")),
    (
        "minimum item interactions",
        ("transfer_invariants", "min_item_interactions_per_item"),
    ),
    ("drop unmapped items", ("transfer_invariants", "drop_unmapped_items")),
    (
        "validation interval seconds",
        ("transfer_invariants", "validation_interval_seconds"),
    ),
    ("day range", ("transfer_invariants", "day_range")),
    ("model dim", ("model_dim",)),
    ("item embedding dim", ("item_embedding_dim",)),
    ("layers", ("transfer_invariants", "transformer", "num_layers")),
    ("query heads", ("transfer_invariants", "transformer", "nhead")),
    ("KV heads", ("transfer_invariants", "transformer", "num_kv_heads")),
    ("FFN", ("transfer_invariants", "transformer", "ffn")),
    ("FFN width", ("transfer_invariants", "transformer", "ffn_intermediate_dim")),
    ("normalization", ("transfer_invariants", "transformer", "norm")),
    ("norm place", ("transfer_invariants", "transformer", "norm_place")),
    ("input norm", ("transfer_invariants", "transformer", "input_norm")),
    ("final norm", ("transfer_invariants", "transformer", "final_norm")),
    ("learned positions", ("transfer_invariants", "transformer", "learned_positions")),
    ("RoPE", ("transfer_invariants", "transformer", "rope")),
    ("ALiBi", ("transfer_invariants", "transformer", "alibi")),
    ("attention window", ("transfer_invariants", "transformer", "attention_window")),
    ("dropout", ("transfer_invariants", "transformer", "dropout")),
    ("input dropout", ("transfer_invariants", "transformer", "input_dropout")),
    ("FFN dropout", ("transfer_invariants", "transformer", "ffn_dropout")),
    ("maximum sequence length", ("transfer_invariants", "max_seq_len")),
    ("training window", ("transfer_invariants", "window")),
    ("BOS", ("transfer_invariants", "bos")),
    ("CLS", ("transfer_invariants", "cls_token")),
    ("timestamp delta", ("transfer_invariants", "timestamp_delta")),
    ("timestamp combination", ("transfer_invariants", "timestamp_combination")),
    ("timestamp bins", ("transfer_invariants", "timestamp_num_bins")),
    ("per-layer item embeddings", ("transfer_invariants", "per_layer_item_embeddings")),
    ("negative sampling", ("transfer_invariants", "negative_sampling")),
    ("negative count", ("transfer_invariants", "num_in_batch_negatives")),
    ("logQ correction", ("transfer_invariants", "logq_correction")),
    ("logQ alpha", ("transfer_invariants", "logq_alpha")),
    ("random negative fraction", ("transfer_invariants", "random_negative_fraction")),
    ("correct positive logQ", ("transfer_invariants", "correct_positive_logq")),
    ("mask false negatives", ("transfer_invariants", "mask_false_negatives")),
    (
        "exclude own-group negatives",
        ("transfer_invariants", "exclude_own_group_negatives"),
    ),
    (
        "dense random negative scores",
        ("transfer_invariants", "dense_random_negative_scores"),
    ),
    ("embedding LR", ("embedding_learning_rate",)),
    ("deep LR", ("deep_learning_rate",)),
    ("batch size", ("batch_size",)),
    ("validation batch size", ("val_batch_size",)),
    ("data workers", ("num_workers",)),
    ("prefetch factor", ("prefetch_factor",)),
    ("epochs", ("num_epochs",)),
    ("schedule", ("transfer_invariants", "lr_schedule", "shape")),
    ("warmup fraction", ("transfer_invariants", "lr_schedule", "warmup_fraction")),
    ("schedule cycles", ("transfer_invariants", "lr_schedule", "cycles")),
    ("minimum LR fraction", ("transfer_invariants", "lr_schedule", "min_lr_fraction")),
    (
        "schedule power exponent",
        ("transfer_invariants", "lr_schedule", "power_exponent"),
    ),
    (
        "schedule power transition tokens",
        ("transfer_invariants", "lr_schedule", "power_transition_tokens"),
    ),
    (
        "schedule timescale fraction",
        ("transfer_invariants", "lr_schedule", "timescale_fraction"),
    ),
    (
        "schedule timescale steps",
        ("transfer_invariants", "lr_schedule", "timescale_steps"),
    ),
    ("weight decay", ("weight_decay",)),
    ("gradient clipping", ("gradient_clip_norm",)),
    ("initializer std", ("initializer_std",)),
    ("runtime dtype", ("runtime_dtype",)),
    ("runtime compile", ("runtime_compile",)),
    ("evaluation ks", ("transfer_invariants", "eval_ks")),
    ("evaluation max users", ("transfer_invariants", "eval_max_users")),
    ("evaluation cadence", ("transfer_invariants", "eval_every_n_epochs")),
    ("selection k", ("transfer_invariants", "selection_k")),
    ("evaluation catalog", ("transfer_invariants", "evaluation_catalog")),
    (
        "exclude seen in evaluation",
        ("transfer_invariants", "exclude_seen_from_evaluation"),
    ),
    ("restore best weights", ("transfer_invariants", "restore_best_weights")),
)


def _metadata_has_path(metadata: dict, path: tuple[str, ...]) -> bool:
    value: object = metadata
    for name in path:
        if not isinstance(value, dict) or name not in value:
            return False
        value = value[name]
    return True


def _inventory_value(metadata: dict, path: tuple[str, ...]) -> str:
    value = _nested_value(metadata, path)
    return "none" if value is None else _format_report_value(value)


def _single_completed_base(runs: list[ReportRun], base: str) -> ReportRun:
    matching = [
        run
        for run in runs
        if run.status == "completed" and _manifest_base(run.configuration) == base
    ]
    names = {run.name for run in matching}
    if len(names) != 1:
        raise ValueError(f"{base}: expected exactly one completed candidate artifact")
    return matching[0]


def _rq2_rq3_evidence(
    runs: list[ReportRun],
) -> tuple[list[ReportRun], set[str], ReportRun, ReportRun]:
    candidates = [_single_completed_base(runs, base) for base in _RQ2_RQ3_CANDIDATES]
    for run in candidates:
        for field in ("recall@100", "epoch_time", "peak_memory_gb"):
            value = _metric_value(run, field)
            if value is None or not math.isfinite(value):
                raise ValueError(f"{run.name}: RQ2/RQ3 candidate lacks {field}")
    quality = _single_completed_base(runs, "selected_quality")
    balanced = _single_completed_base(runs, "selected_balanced")
    for run in (quality, balanced):
        missing = [
            label
            for label, path in _SELECTED_PARAMETER_PATHS
            if not _metadata_has_path(run.metadata, path)
        ]
        if missing:
            raise ValueError(
                f"{run.name}: selected-configuration inventory is missing "
                + ", ".join(missing)
            )
    quality_winner = _best_run(candidates)
    if quality.name != quality_winner.name:
        raise ValueError("selected_quality is not the maximum-recall candidate")
    _metric_bands(required=True)
    threshold = reporting.difference_threshold("recall@100")
    maximum_recall = _metric_value(quality_winner, "recall@100")
    assert maximum_recall is not None
    eligible = [
        run
        for run in candidates
        if (_metric_value(run, "recall@100") or float("-inf"))
        >= maximum_recall - threshold
    ]
    balance_winner = min(
        eligible,
        key=lambda run: (
            _metric_value(run, "epoch_time"),
            _metric_value(run, "peak_memory_gb"),
            run.name,
        ),
    )
    if balanced.name != balance_winner.name:
        raise ValueError(
            "selected_balanced is not the fastest candidate within one shared "
            "recall band of selected_quality"
        )
    pareto = {
        run.name
        for run in candidates
        if not any(
            other.name != run.name
            and (_metric_value(other, "recall@100") or float("-inf"))
            >= (_metric_value(run, "recall@100") or float("-inf"))
            and (_metric_value(other, "epoch_time") or float("inf"))
            <= (_metric_value(run, "epoch_time") or float("inf"))
            and (
                (_metric_value(other, "recall@100") or float("-inf"))
                > (_metric_value(run, "recall@100") or float("-inf"))
                or (_metric_value(other, "epoch_time") or float("inf"))
                < (_metric_value(run, "epoch_time") or float("inf"))
            )
            for other in candidates
        )
    }
    if balanced.name not in pareto:
        raise ValueError("selected_balanced is not Pareto-nondominated")
    return candidates, pareto, quality, balanced


def _reader_configuration(run: ReportRun) -> str:
    fields = (
        ("dim", ("model_dim",)),
        ("depth", ("transfer_invariants", "transformer", "num_layers")),
        ("seq", ("transfer_invariants", "max_seq_len")),
        ("Q heads", ("transfer_invariants", "transformer", "nhead")),
        ("KV heads", ("transfer_invariants", "transformer", "num_kv_heads")),
        ("FFN", ("transfer_invariants", "transformer", "ffn")),
        ("FFN width", ("transfer_invariants", "transformer", "ffn_intermediate_dim")),
        ("window", ("transfer_invariants", "transformer", "attention_window")),
        ("dropout", ("transfer_invariants", "transformer", "dropout")),
        ("norm", ("transfer_invariants", "transformer", "norm")),
        ("norm place", ("transfer_invariants", "transformer", "norm_place")),
        ("input norm", ("transfer_invariants", "transformer", "input_norm")),
        ("final norm", ("transfer_invariants", "transformer", "final_norm")),
        ("positions", ("transfer_invariants", "transformer", "learned_positions")),
        ("RoPE", ("transfer_invariants", "transformer", "rope")),
        ("ALiBi", ("transfer_invariants", "transformer", "alibi")),
        ("time", ("transfer_invariants", "timestamp_combination")),
        ("time bins", ("transfer_invariants", "timestamp_num_bins")),
        ("negatives", ("transfer_invariants", "negative_sampling")),
        ("negative count", ("transfer_invariants", "num_in_batch_negatives")),
        ("batch", ("batch_size",)),
        ("schedule", ("transfer_invariants", "lr_schedule", "shape")),
        ("warmup", ("transfer_invariants", "lr_schedule", "warmup_fraction")),
    )
    values = [
        f"{label}={_format_report_value(value)}"
        for label, path in fields
        if (value := _nested_value(run.metadata, path)) is not None
    ]
    values.append(f"LR={_selected_lr(run)}")
    return "; ".join(values)


def _rq2_table(runs: list[ReportRun]) -> str:
    candidates, _, _, _ = _rq2_rq3_evidence(runs)
    bands = _metric_bands(required=True)
    metric_lines = [
        "| variant | configuration | " + " | ".join(REPORT_METRICS) + " |",
        "| --- | --- | " + " | ".join("---" for _ in REPORT_METRICS) + " |",
    ]
    reference = _single_completed_base(runs, "performance_baseline")
    for run in candidates:
        metric_lines.append(
            "| "
            + " | ".join(
                [
                    _manifest_base(run.configuration).replace("_", " "),
                    _reader_configuration(run),
                    *(
                        _relative_metric_cell(run, reference, metric, bands)
                        for metric in REPORT_METRICS
                    ),
                ]
            )
            + " |"
        )
    return "\n".join(metric_lines)


def _rq3_table(runs: list[ReportRun]) -> str:
    candidates, pareto, _, _ = _rq2_rq3_evidence(runs)
    bands = _metric_bands(required=True)
    reference = _single_completed_base(runs, "performance_baseline")
    lines = [
        "| variant | configuration | Pareto-nondominated | recall@100 | ndcg@100 | epoch time | peak memory |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for run in candidates:
        lines.append(
            "| "
            + " | ".join(
                (
                    _manifest_base(run.configuration).replace("_", " "),
                    _reader_configuration(run),
                    "yes" if run.name in pareto else "no",
                    _relative_metric_cell(run, reference, "recall@100", bands),
                    _relative_metric_cell(run, reference, "ndcg@100", bands),
                    _performance_cell(run, "epoch_time", "{:.1f}s"),
                    _performance_cell(run, "peak_memory_gb", "{:.1f} GiB"),
                )
            )
            + " |"
        )
    return "\n".join(lines)


def _validate_compact_coverage(
    dataset_size: str, grouped: dict[int, list[ReportRun]]
) -> None:
    report_runs = [run for runs in grouped.values() for run in runs]
    proxy_runs = report_runs if dataset_size == "50m" else load_report_runs("50m")
    _rq1_width_evidence(proxy_runs)
    _validate_architecture_tuning(dataset_size, report_runs, proxy_runs)
    _rq2_rq3_evidence(grouped.get(2, []) + grouped.get(3, []))

    for research_question, required in _REQUIRED_COMPLETED_BASES.items():
        if dataset_size == "500m" and research_question == 4:
            required = set(_rq4_proxy_selections(proxy_runs))
        if dataset_size == "500m" and research_question == 8:
            required = (
                set(required) - _rq4_bases() | set(_rq4_proxy_selections(proxy_runs))
            )
        completed = {
            _manifest_base(run.configuration)
            for run in grouped.get(research_question, [])
            if run.status == "completed"
        }
        missing = required - completed
        if missing:
            labels = ", ".join(sorted(missing))
            raise ValueError(f"RQ{research_question} is incomplete: {labels}")

    if dataset_size != "500m":
        return
    aggregate = _aggregate_homework_baseline(grouped.get(2, []))
    if aggregate is None:
        raise ValueError("500M report requires ten current homework-baseline repeats")
    validate_homework_reproduction_runs([aggregate])


def render_compact_report(dataset_size: str) -> str:
    _metric_bands(required=True)
    runs = [
        run
        for run in load_report_runs(dataset_size)
        if run.status == "completed"
    ]
    grouped: dict[int, list[ReportRun]] = defaultdict(list)
    for run in runs:
        grouped[run.research_question].append(run)
    if dataset_size == "50m":
        grouped.pop(5, None)
        grouped.pop(11, None)

    rq5_bundle = _load_rq5_report_bundle() if dataset_size == "500m" else None
    rq5_draft = (
        _require_rq5_reader_draft(rq5_bundle) if rq5_bundle is not None else None
    )
    rq8_reader_tables = (
        _require_rq8_reader_tables(_load_rq8_report_bundle())
        if dataset_size == "500m" and 8 in grouped
        else None
    )
    rq7_reader_tables = (
        _load_rq7_reinvestigation_reader()
        if dataset_size == "500m" and 7 in grouped
        else None
    )
    rq11_reader_table = (
        _require_rq11_reader_table(_load_rq11_report_bundle())
        if dataset_size == "500m" and 11 in grouped
        else None
    )
    rq10_reader_tables = (
        _load_rq10_report_bundle().reader_markdown.strip()
        if dataset_size == "500m"
        else None
    )

    expected_generic = GENERIC_REPORT_RQS[dataset_size]
    if expected_generic <= set(grouped):
        _validate_compact_coverage(dataset_size, grouped)
    proxy_runs = runs if dataset_size == "50m" else load_report_runs("50m")
    available = set(grouped)
    if rq5_bundle is not None:
        available.add(5)
    if rq10_reader_tables is not None:
        available.add(10)
    architecture_questions = available & set(range(4, 11))
    architecture_bases = set().union(
        *(
            _component_bases(research_question)
            for research_question in architecture_questions
            if research_question not in {4, 10}
        )
    )
    selected_architecture_runs = (
        select_architecture_report_runs(
            dataset_size,
            runs,
            proxy_runs,
            bases=(
                architecture_bases - _rq4_bases()
                if dataset_size == "500m"
                else architecture_bases
            ),
        )
        if architecture_bases
        else []
    )
    selected_architecture_grouped: dict[int, list[ReportRun]] = defaultdict(list)
    for run in selected_architecture_runs:
        selected_architecture_grouped[run.research_question].append(run)
    selected_rq4_runs = (
        select_rq4_report_runs(dataset_size, runs, proxy_runs)
        if 4 in architecture_questions
        else []
    )
    selected_architecture_grouped[4] = [
        run for run in selected_rq4_runs if run.research_question == 4
    ]
    selected_negative_runs = []
    if dataset_size == "500m" and 11 in available:
        selected_negative_runs = select_negative_report_runs(
            dataset_size, runs, proxy_runs
        )
        selected_negative_runs.extend(
            select_homework_negative_controls(dataset_size, runs, proxy_runs)
        )
    sections = [f"# G1 — Yambda-{dataset_size.upper()} results"]
    order = [rq for rq in REPORT_RQ_ORDER if rq in available]
    order.extend(sorted(available - set(order)))
    for research_question in order:
        if research_question == 5 and rq5_bundle is not None:
            assert rq5_draft is not None
            sections += ["", rq5_draft]
            continue
        table_runs = (
            selected_architecture_grouped[research_question]
            if 4 <= research_question <= 10
            else grouped[research_question]
        )
        if research_question == 1 and grouped.get(research_question):
            table = _rq1_compact_tables(dataset_size, grouped[research_question])
        elif research_question == 4 and table_runs:
            table = _rq4_tables(dataset_size, table_runs, grouped.get(4, []))
        elif research_question == 6 and table_runs:
            table = _rq6_table(table_runs)
        elif research_question == 8 and table_runs:
            table = _rq8_compact_tables(table_runs)
            if rq8_reader_tables is not None:
                table = f"{table}\n\n{rq8_reader_tables}"
        elif research_question == 9 and table_runs:
            table = _rq9_table(table_runs)
        elif research_question == 2 and grouped.get(research_question):
            table = _rq2_table(grouped.get(2, []) + grouped.get(3, []))
        elif research_question == 3 and grouped.get(research_question):
            table = _rq3_table(grouped.get(2, []) + grouped.get(3, []))
        elif research_question == 10 and rq10_reader_tables is not None:
            table = rq10_reader_tables
        elif research_question == 10 and table_runs:
            table = _rq10_table(table_runs)
        elif research_question == 11 and rq11_reader_table is not None:
            table = _combined_rq11_reader_tables(
                _rq11_table(selected_negative_runs), rq11_reader_table
            )
        elif research_question == 7 and table_runs:
            table = _reader_report_table(
                research_question, table_runs, axis_label="encoding"
            )
            if rq7_reader_tables is not None:
                table = _combined_rq7_reader_tables(table, rq7_reader_tables)
        elif table_runs:
            table = _reader_report_table(research_question, table_runs)
        else:
            raise AssertionError(f"RQ{research_question} has no generated evidence")
        sections += [
            "",
            f"## RQ{research_question} — {REPORT_QUESTION_TITLES[research_question]}",
            "",
            table,
        ]
    return "\n".join(sections).rstrip() + "\n"


def _load_rq5_report_bundle() -> rq5_scheduler_report.Rq5ReportBundle:
    return rq5_scheduler_report.collect_report_bundle(
        GENERATED / "logs",
        EXPERIMENT / "scratchpad/rq5_scheduler_candidate_manifest.json",
    )


def _load_rq8_report_bundle() -> rq8_reinvestigation_report.Rq8ReportBundle:
    return rq8_reinvestigation_report.collect_report_bundle(GENERATED / "logs")


def _load_rq10_report_bundle() -> rq10_reinvestigation_report.Rq10ReportBundle:
    return rq10_reinvestigation_report.collect_report_bundle(GENERATED / "logs")


def _load_rq11_report_bundle() -> rq11_mixed_streaming_report.Rq11ReportBundle:
    return rq11_mixed_streaming_report.collect_report_bundle(GENERATED / "logs")


def _load_rq7_reinvestigation_reader() -> str:
    reader = RQ7_REINVESTIGATION_READER.read_text().strip()
    headings = (
        "### Learned-position fusion comparisons",
        "### RoPE / ALiBi comparison",
    )
    if any(reader.count(heading) != 1 for heading in headings):
        raise ValueError("dedicated native-500M RQ7 reader tables are malformed")
    return reader


def _combined_rq7_reader_tables(historical: str, current: str) -> str:
    return (
        "### Earlier broad position-encoding comparison\n\n"
        + historical.strip()
        + "\n\n"
        + current.strip()
    )


def _combined_rq11_reader_tables(historical: str, current: str) -> str:
    return (
        "### Earlier broad negative-sampling comparison\n\n"
        + historical.strip()
        + "\n\n### Corrected uniform/streaming mixture comparison\n\n"
        + current.strip()
    )


def _require_rq11_reader_table(
    bundle: rq11_mixed_streaming_report.Rq11ReportBundle,
) -> str:
    if bundle.evidence.get("claims_status") != "ready":
        raise ValueError("native-500M RQ11 evidence is not selection-complete")
    table = bundle.reader_markdown.strip()
    if table.count("\n|") != 5 or "| negative sampling |" not in table:
        raise ValueError("native-500M RQ11 reader table is malformed")
    return table


def _require_rq8_reader_tables(
    bundle: rq8_reinvestigation_report.Rq8ReportBundle,
) -> str:
    return _require_rq8_reader_markdown(bundle.reader_markdown)


def _require_rq8_reader_markdown(reader_markdown: str) -> str:
    draft = reader_markdown.rstrip()
    blocks = draft.split("\n\n")
    expected_headers = (
        "| query objective |",
        "| causal ALiBi retained history length |",
        "| reverse-RoPE + ALiBi retained history length |",
    )
    expected_lengths = [12, 25, 50, 100, 128, 200, 256, 512]
    if (
        len(blocks) != 4
        or not blocks[0].startswith("## RQ8 — ")
        or draft.count("## ") != 1
        or any(
            not block.startswith(header)
            for block, header in zip(blocks[1:], expected_headers)
        )
        or any(
            any(not line.startswith("|") for line in block.splitlines())
            for block in blocks[1:]
        )
        or len(blocks[1].splitlines()) != 5
        or any(len(block.splitlines()) != 10 for block in blocks[2:])
        or any(
            _rq8_sequence_lengths(block) != expected_lengths for block in blocks[2:]
        )
        or re.search(r"g1_rq8_sequence_(?!fullcausal)", draft) is not None
    ):
        raise ValueError("dedicated native-500M RQ8 three-table draft is malformed")
    return "\n\n".join(blocks[1:])


def _rq8_sequence_lengths(table: str) -> list[int]:
    lengths = []
    for line in table.splitlines()[2:]:
        label = line.split("|", 2)[1].strip().strip("*")
        if not label.isdigit():
            return []
        lengths.append(int(label))
    return lengths


def _require_rq5_reader_draft(
    bundle: rq5_scheduler_report.Rq5ReportBundle,
) -> str:
    draft = bundle.reader_markdown.rstrip()
    if (
        not draft.startswith("## RQ5 — ")
        or draft.count("## ") != 1
        or "\n| " not in draft
    ):
        raise ValueError("dedicated native-500M RQ5 reader draft is malformed")
    return draft


def _component_bases(research_question: int) -> set[str]:
    if research_question == 8:
        return set().union(*(set(spec[3]) for spec in _RQ8_AXIS_SPECS))
    return set(_REQUIRED_COMPLETED_BASES.get(research_question, set()))


def _render_current_component_question(
    dataset_size: str,
    research_question: int,
    runs: list[ReportRun],
    proxy_runs: list[ReportRun],
) -> str:
    if research_question == 5:
        if dataset_size != "500m":
            raise ValueError("RQ5 has no Yambda-50M report stage")
        return _require_rq5_reader_draft(_load_rq5_report_bundle())
    if research_question == 11:
        table_runs = select_negative_report_runs(dataset_size, runs, proxy_runs)
        table_runs.extend(
            select_homework_negative_controls(dataset_size, runs, proxy_runs)
        )
    elif research_question == 10 and dataset_size == "500m":
        table_runs = []
    elif research_question == 4:
        table_runs = [
            run
            for run in select_rq4_report_runs(dataset_size, runs, proxy_runs)
            if run.research_question == 4
        ]
    elif research_question == 8:
        architecture_bases = _component_bases(research_question)
        if dataset_size == "500m":
            architecture_bases -= _rq4_bases()
        selected = select_architecture_report_runs(
            dataset_size,
            runs,
            proxy_runs,
            bases=architecture_bases,
        )
        if dataset_size == "500m":
            selected.extend(select_rq4_report_runs(dataset_size, runs, proxy_runs))
        table_runs = [
            run for run in selected if run.research_question == research_question
        ]
    else:
        selected = select_architecture_report_runs(
            dataset_size,
            runs,
            proxy_runs,
            bases=_component_bases(research_question),
        )
        table_runs = [
            run for run in selected if run.research_question == research_question
        ]
    if research_question == 4:
        table = _rq4_tables(dataset_size, table_runs, runs)
    elif research_question == 6:
        table = _rq6_table(table_runs)
    elif research_question == 8:
        table = _rq8_compact_tables(table_runs)
        if dataset_size == "500m":
            table = f"{table}\n\n{_require_rq8_reader_tables(_load_rq8_report_bundle())}"
    elif research_question == 9:
        table = _rq9_table(table_runs)
    elif research_question == 10:
        table = (
            _load_rq10_report_bundle().reader_markdown.strip()
            if dataset_size == "500m"
            else _rq10_table(table_runs)
        )
    elif research_question == 11:
        if dataset_size != "500m":
            raise ValueError("RQ11 reader evidence is native-500M only")
        table = _combined_rq11_reader_tables(
            _rq11_table(table_runs),
            _require_rq11_reader_table(_load_rq11_report_bundle()),
        )
    elif research_question == 7:
        table = _reader_report_table(
            research_question, table_runs, axis_label="encoding"
        )
        if dataset_size == "500m":
            table = _combined_rq7_reader_tables(
                table, _load_rq7_reinvestigation_reader()
            )
    else:
        table = _reader_report_table(research_question, table_runs)
    return "\n".join(
        (
            f"## RQ{research_question} — "
            f"{REPORT_QUESTION_TITLES[research_question]}",
            "",
            table,
        )
    )


def render_current_component_questions(dataset_size: str) -> dict[int, str]:
    _metric_bands(required=True)
    runs = load_report_runs(dataset_size)
    proxy_runs = runs if dataset_size == "50m" else load_report_runs("50m")
    return {
        research_question: _render_current_component_question(
            dataset_size, research_question, runs, proxy_runs
        )
        for research_question in CURRENT_COMPONENT_RQS
        if dataset_size == "500m" or research_question not in {5, 11}
    }


def write_automated_reports(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "research_questions_50m.md": render_compact_report("50m"),
        "research_questions_500m.md": render_compact_report("500m"),
        "hyperparameter_tuning_50m.md": render_full_tuning_report("50m"),
    }
    activation_depth = rq4_activation_depth.load_runs(
        GENERATED, report_file_facts(GENERATED)
    )
    if activation_depth:
        artifacts["rq4_activation_depth_tuning_500m.md"] = (
            rq4_activation_depth.tuning_report(activation_depth)
        )
    for name, contents in artifacts.items():
        path = output / name
        path.write_text(contents)
        print(f"wrote {path}")


def write_current_component_reports(output: Path) -> None:
    _metric_bands(required=True)
    blockers = []
    for dataset_size in ("50m", "500m"):
        runs = load_report_runs(dataset_size)
        proxy_runs = runs if dataset_size == "50m" else load_report_runs("50m")
        path = output / f"research_questions_{dataset_size}.md"
        for research_question in CURRENT_COMPONENT_RQS:
            if dataset_size == "50m" and research_question in {5, 11}:
                continue
            try:
                section = _render_current_component_question(
                    dataset_size, research_question, runs, proxy_runs
                )
            except ValueError as error:
                print(f"left {dataset_size} RQ{research_question} unchanged: {error}")
                blockers.append(f"{dataset_size} RQ{research_question}: {error}")
                continue
            _replace_question(path, research_question, section)
        if dataset_size == "50m":
            _remove_question(path, 11)
        _reorder_report_questions(path)
    tuning_path = output / "hyperparameter_tuning_50m.md"
    tuning_path.write_text(render_full_tuning_report("50m"))
    print(f"wrote {tuning_path}")
    if blockers:
        raise ValueError("component reports remain blocked:\n" + "\n".join(blockers))


def render_compact_question(dataset_size: str, research_question: int) -> str:
    if research_question == 8:
        runs = load_report_runs(dataset_size)
        proxy_runs = runs if dataset_size == "50m" else load_report_runs("50m")
        return _render_current_component_question(
            dataset_size, research_question, runs, proxy_runs
        )
    if research_question == 1:
        runs = load_report_runs(dataset_size, research_question=research_question)
        return "\n".join(
            (
                f"## RQ1 — {REPORT_QUESTION_TITLES[1]}",
                "",
                _rq1_compact_tables(dataset_size, runs),
            )
        )
    if research_question != 11:
        raise ValueError("focused compact rendering supports only RQ1 and RQ11")
    if dataset_size != "500m":
        raise ValueError("RQ11 reader evidence is native-500M only")
    runs = load_report_runs(dataset_size, research_question=research_question)
    proxy_runs = load_report_runs("50m", research_question=research_question)
    selected = select_negative_report_runs(
        dataset_size,
        runs,
        proxy_runs,
        global_batch_size=_INITIAL_BATCH_SIZE,
    )
    selected.extend(select_homework_negative_controls(dataset_size, runs, proxy_runs))
    return "\n".join(
        (
            f"## RQ{research_question} — {REPORT_QUESTION_TITLES[research_question]}",
            "",
            _combined_rq11_reader_tables(
                _rq11_table(selected),
                _require_rq11_reader_table(_load_rq11_report_bundle()),
            ),
        )
    )


def render_rq1_reader() -> str:
    compact_50m = render_compact_question("50m", 1)
    compact_500m = render_compact_question("500m", 1)
    table_pattern = re.compile(
        r"(?m)^\| [^\n]+\n^\| ---[^\n]+\n(?:^\|[^\n]+\n?)+"
    )
    tables_50m = table_pattern.findall(compact_50m)
    tables_500m = table_pattern.findall(compact_500m)
    heading = compact_500m.partition("\n")[0]
    if len(tables_50m) == 1 and len(tables_500m) == 1:
        return "\n\n".join(
            (
                heading,
                textwrap.fill(
                    "Native dataset-size LR transfer selects optimizer rates on "
                    "the native 50M cohort, then reuses them once on native 500M "
                    "with the same fixed-width conventional recipe and batch size.",
                    79,
                ),
                tables_500m[0].rstrip(),
                textwrap.fill(
                    "The calibrated confirmation validates simple LR reuse from "
                    "50M to 500M. It does not establish μP model-width transfer; "
                    "the μP model-width conclusion remains work in progress.",
                    79,
                ),
            )
        )
    if len(tables_50m) != 2 or len(tables_500m) != 2:
        raise ValueError("RQ1 reader requires native, 50M-width, and 500M-width tables")
    if tables_50m[0].rstrip() != tables_500m[0].rstrip():
        raise ValueError("RQ1 native dataset-size tables disagree")
    proxy = rq1_width_transfer.load_runs(GENERATED, rq1_width_transfer.PROXY)
    confirmation = rq1_width_transfer.load_runs(
        GENERATED, rq1_width_transfer.CONFIRMATION
    )
    return "\n\n".join(
        (
            heading,
            textwrap.fill(
                "Transfer here means that the configured rate stays optimal as "
                "the model widens. μP fixes a base width and MuAdam divides each "
                "tensor's rate by its own width multiplier, so holding 0.032/0.012 "
                "across widths already shrinks the effective step size in "
                "proportion to width — that rescaling is what is under test, not a "
                "constant step size.",
                79,
            ),
            textwrap.fill(
                "Each 50M table sweeps one rate and holds the other at the "
                "control's value, so a row's reference is that width's own best "
                "point on the same sweep and the metric cells are what the "
                "control's rate costs at that width.",
                79,
            ),
            rq1_width_transfer.sweep_table(proxy, "deep"),
            rq1_width_transfer.sweep_table(proxy, "embedding"),
            rq1_width_transfer.confirmation_table(confirmation),
            tables_50m[0].rstrip(),
            "\n".join(
                (
                    "Treatment descriptions:",
                    "",
                    "- μP width transfer fixes the item table at 64 dimensions, "
                    "varies transformer width from 16 to 256, and applies the "
                    "control's 0.032/0.012 to every width.",
                    "- The shared rate is the control's own selected rate at "
                    "width 64; the 50M-local rate is the alternative each width "
                    "selected on the superseded proxy surface.",
                    "- Native-size LR reuse is the separate conventional check: "
                    "keep the width-64 homework recipe and batch 1280 fixed and "
                    "apply the 50M-selected 0.001/0.002 once on native 500M. Its "
                    "table reports two data sizes rather than a treatment and a "
                    "control, so its rows carry no percentage change.",
                )
            ),
            "Implementation: [μP and native-size protocol]"
            "(evidence/implementation.md#rq1--μtransfer-and-dataset-size-protocol), "
            "[model-width implementation](../../dcn/config/generation.py#L545-L631), "
            "[table generator](analysis/rq1_width_transfer.py), "
            "and [artifact-level evidence](evidence/rq1_transfer.md). The "
            "parameterization follows [Tensor Programs V]"
            "(https://arxiv.org/abs/2203.03466).",
            textwrap.fill(
                "Analysis: Only the deep-LR table tests μP. The item table keeps a "
                "fixed 64 dimensions at every model width, so μP never rescales "
                "its rate; the embedding table shows that the interaction does not "
                "move the optimum, which is a weaker claim.",
                79,
            ),
            textwrap.fill(
                "The deep optimum moves by at most one 2× grid step over a 16× "
                "width range, without direction. Standard parameterization would "
                "put that drift near four steps, so the sweep could have detected "
                "a failure. It has no negative control, though: every "
                "width-varying run in this repository is μP, so nothing here rules "
                "out the model simply being width-insensitive in any "
                "parameterization.",
                79,
            ),
            textwrap.fill(
                "The 500M table is not yet readable. Four of its five shared-rate "
                "runs stopped short of the 20-epoch annealing horizon while both "
                "local-rate comparators trained it in full, which biases the "
                "comparison against the shared rate — the width-256 win is "
                "conservative rather than inflated. Its local rates also come from "
                "the truncated proxy surface; the corrected surface picks "
                "0.064/0.024 at width 16 and 0.128/0.012 at width 256, and neither "
                "has been run on 500M.",
                79,
            ),
            textwrap.fill(
                "Conclusion: On a horizon-complete 50M surface μP transfers the "
                "deep rate across a 16× width range — the optimum stays at "
                "0.012–0.024 with no drift — and the fixed-width item table's rate "
                "does not drift either. Conventional 0.001/0.002 reuse separately "
                "succeeds at native size. The 500M half is unsettled: its "
                "confirmations are horizon-truncated and test alternatives the "
                "proxy no longer selects. What clearly does not transfer is "
                "dataset size, where the embedding optimum moves 4× between 50M "
                "and 500M; see [the transfer study](evidence/transfer_study.md).",
                79,
            ),
        )
    )


def render_full_tuning_question(dataset_size: str, research_question: int) -> str:
    runs = [
        run
        for run in load_report_runs(
            dataset_size, research_question=research_question
        )
        if run.status == "completed"
    ]
    grouped: dict[str, list[ReportRun]] = defaultdict(list)
    for run in runs:
        grouped[run.method].append(run)
    sections = [
        f"## RQ{research_question} — {REPORT_QUESTION_TITLES[research_question]}"
    ]
    for method, method_runs in sorted(grouped.items()):
        control = _ledger_control_run(research_question, method, runs)
        sections += [
            "",
            f"### {method}",
            "",
            _report_table(
                method_runs,
                compact=False,
                research_question=research_question,
                control=control,
            ),
        ]
    return "\n".join(sections)


_REPORT_QUESTION_PATTERN = re.compile(
    r"(?ms)^## RQ(?P<number>\d+) — .*?(?=^## RQ\d+ — |\Z)"
)


def _report_question_sections(path: Path, text: str) -> tuple[str, dict[int, str]]:
    matches = list(_REPORT_QUESTION_PATTERN.finditer(text))
    numbers = [int(match.group("number")) for match in matches]
    duplicates = sorted(
        number for number in set(numbers) if numbers.count(number) > 1
    )
    if duplicates:
        labels = ", ".join(f"RQ{number}" for number in duplicates)
        raise ValueError(f"{path}: duplicate {labels} sections")
    unknown = set(numbers) - set(REPORT_RQ_ORDER)
    if unknown:
        labels = ", ".join(f"RQ{number}" for number in sorted(unknown))
        raise ValueError(f"{path}: unknown report questions: {labels}")
    prefix = text[: matches[0].start()].rstrip() if matches else text.rstrip()
    sections = {
        number: match.group(0).rstrip()
        for number, match in zip(numbers, matches)
    }
    return prefix, sections


def _write_report_question_sections(
    path: Path, prefix: str, sections: dict[int, str]
) -> None:
    ordered = [sections[number] for number in REPORT_RQ_ORDER if number in sections]
    path.write_text("\n\n".join((prefix, *ordered)).lstrip("\n") + "\n")


def _replace_question(path: Path, research_question: int, section: str) -> None:
    replacement_numbers = [
        int(number)
        for number in re.findall(r"(?m)^## RQ(\d+) —", section)
    ]
    if replacement_numbers != [research_question]:
        raise ValueError(
            f"{path}: replacement must contain exactly one RQ{research_question} section"
        )
    prefix, sections = _report_question_sections(path, path.read_text())
    sections[research_question] = section.rstrip()
    _write_report_question_sections(path, prefix, sections)
    print(f"wrote {path}")


def _remove_question(path: Path, research_question: int) -> None:
    prefix, sections = _report_question_sections(path, path.read_text())
    sections.pop(research_question, None)
    _write_report_question_sections(path, prefix, sections)
    print(f"wrote {path}")


def _reorder_report_questions(path: Path) -> None:
    prefix, sections = _report_question_sections(path, path.read_text())
    _write_report_question_sections(path, prefix, sections)


def write_rq8_report(output: Path) -> None:
    dedicated = output / "rq8_reinvestigation_reader_500m.md"
    if not dedicated.is_file():
        raise ValueError(f"missing dedicated RQ8 reader draft: {dedicated}")
    generated_tables = _require_rq8_reader_markdown(dedicated.read_text())
    path = output / "research_questions_500m.md"
    prefix, sections = _report_question_sections(path, path.read_text())
    section = sections.get(8)
    if section is None:
        raise ValueError(f"{path}: missing RQ8 section")
    matches = list(_READER_TABLE.finditer(section))
    generated_axes = (
        "| query objective |",
        "retained history length |",
    )
    targets = [
        match
        for match in matches
        if any(axis in match.group(0).splitlines()[0] for axis in generated_axes)
    ]
    if not targets or targets != matches[-len(targets) :]:
        raise ValueError(f"{path}: malformed generated RQ8 table block")
    start = targets[0].start()
    end = targets[-1].end()
    sections[8] = (
        section[:start] + generated_tables.rstrip() + "\n" + section[end:]
    ).rstrip()
    _write_report_question_sections(path, prefix, sections)
    print(f"wrote {path}")


def write_rq11_reports(output: Path) -> None:
    _remove_question(output / "research_questions_50m.md", 11)
    _remove_question(output / "hyperparameter_tuning_50m.md", 11)
    _replace_question(
        output / "research_questions_500m.md",
        11,
        render_compact_question("500m", 11),
    )


def write_rq1_reports(output: Path) -> None:
    for dataset_size in ("50m", "500m"):
        _replace_question(
            output / f"research_questions_{dataset_size}.md",
            1,
            render_compact_question(dataset_size, 1),
        )
    _replace_question(
        READER_REPORT,
        1,
        render_rq1_reader(),
    )
    _replace_question(
        output / "hyperparameter_tuning_50m.md",
        1,
        render_full_tuning_question("50m", 1),
    )


def write_rq5_reports() -> None:
    bundle = _load_rq5_report_bundle()
    paths = rq5_scheduler_report.write_report_bundle(
        bundle, EXPERIMENT / "scratchpad", EXPERIMENT / "evidence"
    )
    for path in paths.values():
        print(f"wrote {path}")


def write_rq10_reports(output: Path) -> None:
    bundle = _load_rq10_report_bundle()
    paths = rq10_reinvestigation_report.write_report_bundle(
        bundle, output, EXPERIMENT / "evidence"
    )
    _replace_question(
        output / "research_questions_500m.md",
        10,
        "\n".join(
            (
                f"## RQ10 — {REPORT_QUESTION_TITLES[10]}",
                "",
                bundle.reader_markdown.strip(),
            )
        ),
    )
    _replace_question(
        READER_REPORT,
        10,
        rq10_reinvestigation_report.render_readme_section(bundle),
    )
    for path in paths.values():
        print(f"wrote {path}")


_READER_TABLE = re.compile(r"(?m)^\| [^\n]+\n^\| ---[^\n]+\n(?:^\|[^\n]+\n?)+")
_READER_SECTION = re.compile(r"(?ms)^## RQ(\d+)[^\n]*\n.*?(?=^## |\Z)")


def _question_table_sources(path: Path) -> dict[int, list[str]]:
    return {
        int(match.group(1)): _READER_TABLE.findall(match.group(0))
        for match in _READER_SECTION.finditer(path.read_text())
    }


def sync_reader_tables(output: Path) -> None:
    """Replace each question's whole table block with the generated tables.

    Positional replacement would drop a table the generator has grown since the
    report was last written, so the block between the first and last table is
    substituted wholesale. No question interleaves prose with its tables.

    RQ1, RQ7, and RQ11 are skipped because dedicated writers own their
    reader-section structure.
    """
    generated = _question_table_sources(output / "research_questions_500m.md")
    replaced = []

    def replace(match: re.Match) -> str:
        question = int(match.group(1))
        tables = generated.get(question, [])
        section = match.group(0)
        spans = [found.span() for found in _READER_TABLE.finditer(section)]
        if question in {1, 7, 11} or not tables or not spans:
            return section
        block = "\n".join(table.rstrip() + "\n" for table in tables)
        replaced.append(f"RQ{question}: {len(spans)} -> {len(tables)} tables")
        return section[: spans[0][0]] + block + section[spans[-1][1] :]

    READER_REPORT.write_text(
        _READER_SECTION.sub(replace, READER_REPORT.read_text())
    )
    print(f"wrote {READER_REPORT}")
    for message in replaced:
        print(f"  {message}")


def main() -> None:
    parser = argparse.ArgumentParser()
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument(
        "--write",
        action="store_true",
        help="write the compact reports and complete 50M tuning ledger",
    )
    output_mode.add_argument(
        "--tuning-only",
        action="store_true",
        help="write only the complete 50M tuning ledger",
    )
    output_mode.add_argument(
        "--rq1-only",
        action="store_true",
        help="replace only RQ1 in the compact reports and 50M tuning ledger",
    )
    output_mode.add_argument(
        "--rq8-only",
        action="store_true",
        help="replace only native-500M RQ8 in the compact report",
    )
    output_mode.add_argument(
        "--rq11-only",
        action="store_true",
        help="replace only RQ11 in the compact reports and 50M tuning ledger",
    )
    output_mode.add_argument(
        "--rq10-only",
        action="store_true",
        help="write dedicated native-500M RQ10 tables, tuning, and evidence",
    )
    output_mode.add_argument(
        "--rq5-only",
        action="store_true",
        help="write the dedicated native-500M RQ5 ledger, reader draft, and evidence",
    )
    output_mode.add_argument(
        "--reader-tables",
        action="store_true",
        help="replace report tables with the generated 500M tables",
    )
    output_mode.add_argument(
        "--components-only",
        action="store_true",
        help="replace current RQ4-RQ11 tables without requiring RQ2/RQ3",
    )
    arguments = parser.parse_args()
    if arguments.rq1_only:
        write_rq1_reports(EXPERIMENT / "scratchpad")
        return
    if arguments.rq8_only:
        write_rq8_report(EXPERIMENT / "scratchpad")
        return
    if arguments.rq11_only:
        write_rq11_reports(EXPERIMENT / "scratchpad")
        return
    if arguments.rq10_only:
        write_rq10_reports(EXPERIMENT / "scratchpad")
        return
    if arguments.rq5_only:
        write_rq5_reports()
        return
    if arguments.reader_tables:
        sync_reader_tables(EXPERIMENT / "scratchpad")
        return
    if arguments.components_only:
        write_current_component_reports(EXPERIMENT / "scratchpad")
        return
    if arguments.tuning_only:
        path = EXPERIMENT / "scratchpad" / "hyperparameter_tuning_50m.md"
        path.write_text(render_full_tuning_report("50m"))
        print(f"wrote {path}")
        return
    if not arguments.write:
        print(render_full_tuning_report("50m"))
        return
    write_automated_reports(EXPERIMENT / "scratchpad")


if __name__ == "__main__":
    main()
