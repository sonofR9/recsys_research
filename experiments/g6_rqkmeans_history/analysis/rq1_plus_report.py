from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import statistics
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
G6_ROOT = PROJECT_ROOT / "experiments/g6_rqkmeans_history"
RQ0_SELECTION = G6_ROOT / "evidence/rq0_selection_native50m.json"
RQ0_SLICES = G6_ROOT / "evidence/rq0_slices_native50m.json"
SURFACE = G6_ROOT / "evidence/rq1_rq3_surface_native50m.json"
CONFIRMATION = G6_ROOT / "evidence/rq1_rq3_confirmation_native50m.json"
TERMINAL = G6_ROOT / "evidence/rq2_rq3_selection_native50m.json"
RQ1_UNEXPECTED = G6_ROOT / "evidence/rq1_unexpected_native50m.json"
COMPACT_OUTPUT = G6_ROOT / "evidence/rq1_plus_reader_native.md"
TUNING_OUTPUT = G6_ROOT / "evidence/rq1_plus_tuning_native.md"
EXPECTED_SHA256 = {
    "RQ0 selection": "8391def6cfddbeb4cb1b048f3d4fed62e4bf0e304270e8d21a7d2de4ded0646b",
    "RQ0 slices": "c97cb9d0d3d0dc10e3c27ab2630436d5504d78547508b50c3f985d44655ef0f0",
    "RQ1-RQ3 surface": "841101538ea333d0a9fed4c6f3e5fdd5ea03ef73a982b16af97715ae9ca8824a",
    "RQ1-RQ3 confirmation": "a97d58bc1abef95f8942eac58c555207556e1a634742e0cb7e1fe6ee91359ac6",
    "RQ2-RQ3 terminal": "4ae8783438c597b1693dff706869ac08c54076513f4acf06458ad4b0b8349c25",
    "RQ1 unexpected": "3ab3dbf315f8f6a2584a1ca7bb0fa6a71915fa49e484d56966037cc1ce2c1dc9",
}
METRICS = ("recall@100", "ndcg@100", "mrr@100", "coverage@100")
NATIVE50_RELATIVE_DISPERSION = {
    "recall@100": 0.19413750216294554,
    "ndcg@100": 0.21426688617583264,
    "mrr@100": 0.2428021146075763,
    "coverage@100": 0.8517845313987747,
}
METRIC_LABELS = {
    "recall@100": "Recall@100",
    "ndcg@100": "NDCG@100",
    "mrr@100": "MRR@100",
    "coverage@100": "Coverage@100",
}


@dataclass(frozen=True)
class ReportEvidence:
    rq0_selection_path: Path
    rq0_slices_path: Path
    surface_path: Path
    confirmation_path: Path
    terminal_path: Path
    rq1_unexpected_path: Path
    rq0: dict[str, Any]
    rq0_slices: dict[str, Any]
    surface: dict[str, Any]
    confirmation: dict[str, Any]
    terminal: dict[str, Any]
    rq1_unexpected: dict[str, Any]


def load_report_evidence(
    *,
    rq0_selection_path: Path = RQ0_SELECTION,
    rq0_slices_path: Path = RQ0_SLICES,
    surface_path: Path = SURFACE,
    confirmation_path: Path = CONFIRMATION,
    terminal_path: Path = TERMINAL,
    rq1_unexpected_path: Path = RQ1_UNEXPECTED,
) -> ReportEvidence:
    rq0 = _load_pinned(rq0_selection_path, "RQ0 selection")
    rq0_slices = _load_pinned(rq0_slices_path, "RQ0 slices")
    surface = _load_pinned(surface_path, "RQ1-RQ3 surface")
    confirmation = _load_pinned(confirmation_path, "RQ1-RQ3 confirmation")
    terminal = _load_pinned(terminal_path, "RQ2-RQ3 terminal")
    rq1_unexpected = _load_pinned(rq1_unexpected_path, "RQ1 unexpected")
    if (
        rq0_slices.get("schema_version") != 1
        or surface.get("schema") != "g6-rq1-rq3-surface/v1"
        or confirmation.get("schema") != "g6-rq1-rq3-confirmation/v1"
        or terminal.get("schema") != "g6-rq2-rq3-selection/v1"
        or rq1_unexpected.get("schema") != "g6-rq1-unexpected/v1"
    ):
        raise ValueError("G6 report evidence schema changed")
    surface_sha256 = _sha256(surface_path)
    if confirmation.get("surface_sha256") != surface_sha256:
        raise ValueError("RQ1-RQ3 confirmation no longer binds its surface")
    if rq1_unexpected["source_artifacts"]["surface"][
        "sha256"
    ] != surface_sha256 or rq1_unexpected["source_artifacts"]["confirmation"][
        "sha256"
    ] != _sha256(
        confirmation_path
    ):
        raise ValueError("RQ1 unexpected diagnostics no longer bind their sources")
    if (
        confirmation["rq2_rq3"].get("terminal_job_id")
        != terminal.get("terminal_job_id")
        or terminal.get("metrics") != rq0.get("semantic_winner", {}).get("metrics")
    ):
        raise ValueError("G6 terminal selection changed")
    return ReportEvidence(
        rq0_selection_path=rq0_selection_path,
        rq0_slices_path=rq0_slices_path,
        surface_path=surface_path,
        confirmation_path=confirmation_path,
        terminal_path=terminal_path,
        rq1_unexpected_path=rq1_unexpected_path,
        rq0=rq0,
        rq0_slices=rq0_slices,
        surface=surface,
        confirmation=confirmation,
        terminal=terminal,
        rq1_unexpected=rq1_unexpected,
    )


def render_compact_report(evidence: ReportEvidence) -> str:
    rq1 = evidence.confirmation["rq1"]
    systems = evidence.confirmation["rq2_rq3"]["systems"]
    approved_bands = evidence.rq0["bands"]
    blocks = [
        "# G6: RQ-KMeans semantic IDs in history — RQ1–RQ3",
        "## RQ1 — Does content-informed SID initialization outperform random initialization?",
        _quality_table(
            "Random initialization",
            rq1["random"]["mean_metrics"],
            [
                (
                    "Content-informed PCA initialization",
                    rq1["content_pca"]["mean_metrics"],
                )
            ],
            approved_bands,
            selected_label=None,
        ),
        _rq1_convergence_table(rq1),
        _collision_proxy_table(
            [
                (
                    "Random initialization",
                    _mean_metrics(rq1["random"]["rows"]),
                    evidence.surface["rq1"]["selected"]["random"]["diagnostics"],
                ),
                (
                    "Content-informed PCA initialization",
                    _mean_metrics(rq1["content_pca"]["rows"]),
                    evidence.surface["rq1"]["selected"]["content_pca"]["diagnostics"],
                ),
            ],
            max_depth=4,
        ),
        _geometry_table(
            [
                (
                    "Shared 4 × 512 tokenizer",
                    evidence.surface["rq1"]["selected"]["content_pca"]["diagnostics"],
                )
            ]
        ),
        _collision_structure_table(
            [("Shared 4 × 512 tokenizer", _rq1_diagnostics(evidence))]
        ),
        _rq1_unexpected_table(evidence),
        _serving_cost_table(
            [
                (
                    "Random initialization",
                    evidence.rq0["treatment_winners"]["item_learned_frozen_sid_event"][
                        "inference_cost"
                    ],
                ),
                (
                    "Content-informed PCA initialization",
                    evidence.rq0["treatment_winners"]["item_learned_frozen_sid_event"][
                        "inference_cost"
                    ],
                ),
            ]
        ),
        _availability_table(
            [
                (
                    "Both initialization arms",
                    "model parameters, artifact bytes, tokenizer fit time, epoch time, peak memory, dedicated full-catalog latency, target/history collision slices",
                )
            ]
        ),
        "## RQ2 — What RQ-KMeans setup works best with collision resolution?",
        _quality_table(
            "RQ0 suffix setup (3 levels × 512 shared codes; 20 iterations)",
            systems["rq0"]["mean_metrics"],
            [
                (
                    "Selected suffix setup (3 levels × 512 shared codes; 20 iterations)",
                    systems["suffix"]["mean_metrics"],
                )
            ],
            approved_bands,
            selected_label=(
                "Selected suffix setup (3 levels × 512 shared codes; 20 iterations)"
            ),
        ),
        _collision_proxy_table(
            [
                (
                    "Selected suffix setup",
                    _mean_metrics(systems["suffix"]["rows"]),
                    evidence.surface["rq2_rq3"]["selected"]["suffix"]["diagnostics"],
                )
            ],
            max_depth=3,
        ),
        _geometry_table(
            [
                (
                    "Selected suffix tokenizer",
                    evidence.surface["rq2_rq3"]["selected"]["suffix"]["diagnostics"],
                )
            ]
        ),
        _collision_structure_table(
            [("Selected suffix tokenizer", _suffix_diagnostics(evidence))]
        ),
        _serving_cost_table(
            [
                (
                    "Selected suffix setup",
                    evidence.rq0["semantic_winner"]["inference_cost"],
                )
            ]
        ),
        _slice_recall_table(
            evidence.rq0_slices,
            control_label="Best-G1 item-ID baseline",
            treatment_label="Selected suffix setup",
        ),
        _availability_table(
            [
                (
                    "Selected suffix setup",
                    "target-bucket slices, artifact bytes, tokenizer fit time, epoch time, peak memory, dedicated full-catalog latency",
                )
            ]
        ),
        "## RQ3 — What RQ-KMeans setup works best without collision resolution?",
        _quality_table(
            "RQ0 suffix setup (3 levels × 512 shared codes; 20 iterations)",
            systems["rq0"]["mean_metrics"],
            [
                (
                    "Selected tokenizer without suffix (2 levels × 4096 shared codes; 20 iterations)",
                    systems["none"]["mean_metrics"],
                )
            ],
            approved_bands,
            selected_label=(
                "RQ0 suffix setup (3 levels × 512 shared codes; 20 iterations)"
            ),
        ),
        _collision_proxy_table(
            [
                (
                    "RQ0 suffix setup",
                    _mean_metrics(systems["rq0"]["rows"]),
                    evidence.surface["rq2_rq3"]["selected"]["suffix"]["diagnostics"],
                ),
                (
                    "Selected tokenizer without suffix",
                    _mean_metrics(systems["none"]["rows"]),
                    evidence.surface["rq2_rq3"]["selected"]["none"]["diagnostics"],
                ),
            ],
            max_depth=3,
        ),
        _geometry_table(
            [
                (
                    "RQ0 suffix tokenizer",
                    evidence.surface["rq2_rq3"]["selected"]["suffix"]["diagnostics"],
                ),
                (
                    "Selected tokenizer without suffix",
                    evidence.surface["rq2_rq3"]["selected"]["none"]["diagnostics"],
                ),
            ]
        ),
        _collision_structure_table(
            [
                ("RQ0 suffix tokenizer", _suffix_diagnostics(evidence)),
                (
                    "Selected tokenizer without suffix",
                    evidence.surface["rq2_rq3"]["selected"]["none"]["diagnostics"],
                ),
            ]
        ),
        _serving_cost_table(
            [
                (
                    "RQ0 suffix setup",
                    evidence.rq0["semantic_winner"]["inference_cost"],
                ),
                ("Selected tokenizer without suffix", None),
            ]
        ),
        _slice_recall_table(
            evidence.rq0_slices,
            control_label="Best-G1 item-ID baseline",
            treatment_label="RQ0 suffix setup",
        ),
        _availability_table(
            [
                (
                    "Selected tokenizer without suffix",
                    "tail, target/history collision slices, model parameters, artifact bytes, tokenizer fit time, epoch time, peak memory, MACs, embedding reads, dedicated full-catalog latency",
                )
            ]
        ),
        "## Aggregated improvement",
        "### Native Yambda-50M",
        _aggregate_quality_table(evidence),
        _aggregate_components_table(evidence),
    ]
    return "\n\n".join(blocks) + "\n"


def render_tuning_report(evidence: ReportEvidence) -> str:
    blocks = [
        "# G6 RQ1–RQ3 tuning",
        "## RQ1 — SID lookup initialization",
        "### Random initialization",
        _surface_tuning_table(evidence, family="rq1", variant="random"),
        "### Content-informed PCA initialization",
        _surface_tuning_table(evidence, family="rq1", variant="content_pca"),
        "## RQ2 — Collision suffix",
        _surface_tuning_table(evidence, family="collision", variant="suffix"),
        "## RQ3 — No collision suffix",
        _surface_tuning_table(evidence, family="collision", variant="none"),
    ]
    return "\n\n".join(blocks) + "\n"


def _quality_table(
    baseline_label: str,
    baseline: dict[str, float],
    treatments: Sequence[tuple[str, dict[str, float]]],
    bands: dict[str, float],
    *,
    selected_label: str | None,
) -> str:
    rows = [
        [
            (
                f"**{baseline_label}**"
                if baseline_label == selected_label
                else baseline_label
            ),
            _score(baseline["recall@100"]),
            "baseline",
            _score(baseline["ndcg@100"]),
            _score(baseline["mrr@100"]),
            _score(baseline["coverage@100"]),
        ]
    ]
    for label, metrics in treatments:
        rows.append(
            [
                f"**{label}**" if label == selected_label else label,
                _score(metrics["recall@100"]),
                _delta(
                    metrics["recall@100"], baseline["recall@100"], bands["recall@100"]
                ),
                _delta(metrics["ndcg@100"], baseline["ndcg@100"], bands["ndcg@100"]),
                _delta(metrics["mrr@100"], baseline["mrr@100"], bands["mrr@100"]),
                _delta(
                    metrics["coverage@100"],
                    baseline["coverage@100"],
                    bands["coverage@100"],
                ),
            ]
        )
    return _markdown_table(
        [
            "Method",
            "Recall@100",
            "Delta Recall@100",
            "NDCG@100",
            "MRR@100",
            "Coverage@100",
        ],
        rows,
    )


def _rq1_convergence_table(rq1: dict[str, Any]) -> str:
    return _markdown_table(
        ["Method (convergence)", "Mean epoch to 95%", "Mean normalized Recall AUC"],
        [
            [
                "Random initialization",
                f'{rq1["random"]["convergence"]["mean_first_epoch_at_95_percent"]:.2f}',
                f'{rq1["random"]["convergence"]["mean_normalized_auc"]:.3f}',
            ],
            [
                "Content-informed PCA initialization",
                f'{rq1["content_pca"]["convergence"]["mean_first_epoch_at_95_percent"]:.2f}',
                f'{rq1["content_pca"]["convergence"]["mean_normalized_auc"]:.3f}',
            ],
        ],
    )


def _collision_proxy_table(
    rows: Sequence[tuple[str, dict[str, float], dict[str, Any]]], *, max_depth: int
) -> str:
    headers = (
        ["Method (SID diagnostics)", "Exact SID Recall@100"]
        + [f"Prefix L{depth}" for depth in range(1, max_depth + 1)]
        + ["ICR", "Collided items"]
    )
    body = []
    for label, metrics, diagnostics in rows:
        body.append(
            [label, _score(metrics["sid_exact_recall@100"])]
            + [
                (
                    _score(metrics[f"sid_prefix_recall@100_l{depth}"])
                    if f"sid_prefix_recall@100_l{depth}" in metrics
                    else "—"
                )
                for depth in range(1, max_depth + 1)
            ]
            + [
                _percent(diagnostics["identifier_collision_rate"]),
                _percent(diagnostics["collided_item_fraction"]),
            ]
        )
    return _markdown_table(headers, body)


def _geometry_table(rows: Sequence[tuple[str, dict[str, Any]]]) -> str:
    def values(
        diagnostics: dict[str, Any], key: str, *, percentage: bool = False
    ) -> str:
        items = diagnostics.get(key)
        if items is None:
            return "—"
        if percentage:
            return " / ".join(_percent(item) for item in items)
        if all(float(item).is_integer() for item in items):
            return " / ".join(str(int(item)) for item in items)
        return " / ".join(f"{item:.3f}" for item in items)

    return _markdown_table(
        [
            "Tokenizer (intrinsic diagnostics)",
            "p95 load by level",
            "p95 / mean by level",
            "Intra-code cosine by level",
            "Reconstruction MSE by depth",
            "Dead-code fraction by level",
        ],
        [
            [
                label,
                values(diagnostics, "p95_occupied_load"),
                values(diagnostics, "p95_to_mean_occupied_load"),
                values(diagnostics, "intra_code_cosine_similarity"),
                values(diagnostics, "reconstruction_mse_by_depth"),
                values(diagnostics, "dead_code_fraction", percentage=True),
            ]
            for label, diagnostics in rows
        ],
    )


def _rq1_diagnostics(evidence: ReportEvidence) -> dict[str, Any]:
    return evidence.surface["rq1"]["selected"]["content_pca"]["diagnostics"]


def _suffix_diagnostics(evidence: ReportEvidence) -> dict[str, Any]:
    full = next(
        row["diagnostics"]
        for row in evidence.surface["rq2_rq3"]["rows"]
        if row["parameters"]["collision_policy"] == "none"
        and row["parameters"]["trial"] == 0
    )
    return {
        **full,
        "collision_policy": "suffix",
        "collision_suffix_symbols": evidence.surface["rq2_rq3"]["selected"]["suffix"][
            "diagnostics"
        ]["collision_suffix_symbols"],
    }


def _collision_structure_table(
    rows: Sequence[tuple[str, dict[str, Any]]],
) -> str:
    return _markdown_table(
        [
            "Tokenizer (collision diagnostics)",
            "Unique base tuples",
            "ICR",
            "Collided items",
            "Suffix symbols",
            "Bucket p50 / p95 / p99 / max",
        ],
        [
            [
                label,
                str(diagnostics.get("unique_base_tuples", "—")),
                _percent(diagnostics["identifier_collision_rate"]),
                _percent(diagnostics["collided_item_fraction"]),
                str(diagnostics.get("collision_suffix_symbols", "—")),
                " / ".join(
                    _count(diagnostics.get(f"collision_bucket_size_{name}"))
                    for name in ("p50", "p95", "p99", "max")
                ),
            ]
            for label, diagnostics in rows
        ],
    )


def _rq1_unexpected_table(evidence: ReportEvidence) -> str:
    diagnostics = evidence.rq1_unexpected
    if diagnostics is None:
        rows = [["Saved unexpected-result diagnostics", "pending"]]
    else:
        initialization = diagnostics["initialization_identity"]
        gradient = diagnostics["gradient_probe"]
        projection = diagnostics["projection_reconstruction"]["levels"]
        retained = [level["retained_variance_fraction"] for level in projection]
        reconstruction = [level["centered_reconstruction_mse"] for level in projection]
        redundancy = diagnostics["duplicated_frozen_centroid_information"]
        warm_start = diagnostics["lr_warm_start"]
        rows = [
            [
                "Initialization hashes and RMS",
                f'all checks pass at seeds {"/".join(map(str, initialization["paired_seeds"]))}; max RMS mismatch {max(row["maximum_absolute_rms_difference"] for row in initialization["comparisons"]):.2e}',
            ],
            [
                "Learned-base-row gradient",
                f'nonzero on {gradient["learned_base_rows_with_gradient"]} touched base rows; frozen centroids unchanged',
            ],
            [
                "128→32 PCA reconstruction",
                f"retained variance {min(retained) * 100:.1f}–{max(retained) * 100:.1f}%; centered MSE {min(reconstruction):.6f}–{max(reconstruction):.6f}",
            ],
            [
                "Frozen-view redundancy",
                (
                    "initialized 32D view is a deterministic linear projection of "
                    "the same frozen 128D centroids"
                    if redundancy["deterministic_linear_projection_of_frozen_view"]
                    else "not verified"
                ),
            ],
            [
                "Paired LR warm-start erasure",
                (
                    "not supported: five fixed-deep-LR AUC deltas are non-monotone"
                    if not warm_start["monotonic_erasure_supported"]
                    else "supported"
                ),
            ],
        ]
    return _markdown_table(["RQ1 unexpected-result check", "Evidence"], rows)


def _serving_cost_table(
    rows: Sequence[tuple[str, dict[str, Any] | None]],
) -> str:
    body = []
    for label, cost in rows:
        if cost is None:
            body.append([label, "—", "—", "—", "—", "—"])
            continue
        transformer = cost["transformer_multiply_accumulates"]
        tokenizer = cost["tokenizer_multiply_accumulates"]
        body.append(
            [
                label,
                str(cost["sequence_tokens"]),
                _millions(transformer),
                _millions(tokenizer),
                _millions(transformer + tokenizer),
                f'{cost["embedding_scalar_reads"]:,}',
            ]
        )
    return _markdown_table(
        [
            "Method (serving-cost estimate)",
            "Sequence tokens",
            "Transformer MACs",
            "Tokenizer MACs",
            "Total MACs",
            "Embedding reads",
        ],
        body,
    )


def _availability_table(rows: Sequence[tuple[str, str]]) -> str:
    return _markdown_table(
        ["Method (unavailable diagnostics)", "Not committed"],
        [[label, fields] for label, fields in rows],
    )


def _slice_recall_table(
    document: dict[str, Any],
    *,
    control_label: str,
    treatment_label: str,
    treatment_available: bool = True,
) -> str:
    names = (
        ("Tail target: low", "frequency_low"),
        ("Tail target: middle", "frequency_middle"),
        ("Tail target: high", "frequency_high"),
        ("History has collided SID", "history_has_collided_base_sid"),
        ("History has no collided SID", "history_has_no_collided_base_sid"),
    )
    rows = []
    for label, key in names:
        source = document["slices"][key]
        control = source["control"]["recall@100"]
        treatment = source["semantic"]["recall@100"]
        rows.append(
            [
                label,
                control_label,
                _score(control),
                treatment_label,
                _score(treatment) if treatment_available else "—",
                (f"{treatment - control:+.3f}" if treatment_available else "—"),
                str(source["num_users"]),
                str(source["num_targets"]),
            ]
        )
    return _markdown_table(
        [
            "Slice diagnostic",
            "Control",
            "Control Recall@100",
            "Treatment",
            "Treatment Recall@100",
            "Point delta",
            "Users",
            "Targets",
        ],
        rows,
    )


def _aggregate_quality_table(evidence: ReportEvidence) -> str:
    baseline = evidence.rq0["original_control"]["metrics"]
    aggregate = evidence.rq0["semantic_winner"]["metrics"]
    rows = []
    for label, metrics, is_baseline in (
        ("Original G1 item-ID baseline", baseline, True),
        ("Best-G1 plus terminal SID history", aggregate, False),
    ):
        rows.append(
            [
                label,
                _score(metrics["recall@100"]),
                _aggregate_delta(metrics, baseline, "recall@100", is_baseline),
                _optional(metrics.get("ndcg@100")),
                _aggregate_delta(metrics, baseline, "ndcg@100", is_baseline),
                _optional(metrics.get("mrr@100")),
                _aggregate_delta(metrics, baseline, "mrr@100", is_baseline),
                _optional(metrics.get("coverage@100")),
                _aggregate_delta(metrics, baseline, "coverage@100", is_baseline),
            ]
        )
    return _markdown_table(
        [
            "Method",
            "Recall@100",
            "Delta Recall@100",
            "NDCG@100",
            "Delta NDCG@100",
            "MRR@100",
            "Delta MRR@100",
            "Coverage@100",
            "Delta Coverage@100",
        ],
        rows,
    )


def _aggregate_delta(
    metrics: dict[str, float],
    baseline: dict[str, float],
    metric: str,
    is_baseline: bool,
) -> str:
    if metric not in metrics or metric not in baseline:
        return "—"
    if is_baseline:
        return "baseline"
    gain = metrics[metric] - baseline[metric]
    return f"{gain:+.3f} ({gain / baseline[metric] * 100:+.2f}%)"


def _aggregate_components_table(evidence: ReportEvidence) -> str:
    rows = []
    original = evidence.rq0["original_control"]["metrics"]
    best_g1 = evidence.rq0["primary_control"]["metrics"]
    aggregate = evidence.rq0["semantic_winner"]["metrics"]
    for metric in METRICS:
        best_g1_gain = best_g1[metric] - original[metric]
        sid_marginal = aggregate[metric] - best_g1[metric]
        total = aggregate[metric] - original[metric]
        component_sum = best_g1_gain + sid_marginal
        interaction_gap = total - component_sum
        interaction_band = _aggregate_metric_band(original, metric)
        rows.append(
            [
                "Native 50M",
                METRIC_LABELS[metric],
                _score(original[metric]),
                _score(aggregate[metric]),
                f"{total:+.3f}",
                f"{total / original[metric] * 100:+.2f}%",
                f"{best_g1_gain:+.3f}",
                f"{sid_marginal:+.3f}",
                f"{component_sum:+.3f}",
                f"{interaction_gap:+.3f}",
                f"{interaction_band:.3f}",
                "resolved" if abs(interaction_gap) > interaction_band else "unresolved",
            ]
        )
    return _markdown_table(
        [
            "Dataset (component arithmetic)",
            "Metric",
            "Original baseline",
            "Aggregate",
            "Point gain",
            "Percent gain",
            "Best-G1 gain",
            "Terminal SID marginal",
            "Standalone sum",
            "Interaction gap",
            "Interaction band",
            "Interaction resolution",
        ],
        rows,
    )


def _aggregate_table(evidence: ReportEvidence) -> str:
    original = evidence.rq0["original_control"]["metrics"]
    primary = evidence.rq0["primary_control"]["metrics"]
    aggregate = evidence.rq0["semantic_winner"]["metrics"]
    bands = {
        metric: _aggregate_metric_band(original, metric) for metric in METRICS
    }
    rows = []
    for metric in METRICS:
        gain = aggregate[metric] - original[metric]
        component_sum = (
            primary[metric] - original[metric] + aggregate[metric] - primary[metric]
        )
        gap = gain - component_sum
        resolved = abs(gain) > bands[metric]
        rows.append(
            [
                METRIC_LABELS[metric],
                _score(original[metric]),
                _score(aggregate[metric]),
                f"{gain:+.3f}",
                f"{gain / original[metric] * 100:+.2f}%",
                f"{component_sum:+.3f}",
                f"{gap:+.3f}",
                f"{bands[metric]:.3f}",
                "resolved" if resolved else "unresolved",
            ]
        )
    return _markdown_table(
        [
            "Metric",
            "Original baseline",
            "Aggregate",
            "Point gain",
            "Percent gain",
            "Standalone sum",
            "Interaction gap",
            "Size-matched band",
            "Resolution",
        ],
        rows,
    )


def _aggregate_metric_band(original: dict[str, float], metric: str) -> float:
    return original[metric] * NATIVE50_RELATIVE_DISPERSION[metric]


def _surface_tuning_table(
    evidence: ReportEvidence, *, family: str, variant: str
) -> str:
    if family == "rq1":
        rows = [
            row
            for row in evidence.surface["rq1"]["rows"]
            if row["parameters"]["sid_lookup_initialization"] == variant
        ]
    else:
        rows = [
            row
            for row in evidence.surface["rq2_rq3"]["rows"]
            if row["parameters"]["collision_policy"] == variant
        ]
    best_job_id = max(rows, key=lambda row: row["metrics"]["recall@100"])["job_id"]
    body = []
    for row in rows:
        parameters = row["parameters"]
        if family == "rq1":
            method = (
                "Content-informed PCA"
                if parameters["sid_lookup_initialization"] == "content_pca"
                else "Random"
            )
        else:
            method = (
                "Collision suffix"
                if parameters["collision_policy"] == "suffix"
                else "No collision suffix"
            )
        recall = _score(row["metrics"]["recall@100"])
        if row["job_id"] == best_job_id:
            recall = f"**{recall}**"
        body.append(
            [
                method,
                _trial(row["job_id"]),
                _tokenizer(parameters),
                str(parameters["representation_width"]),
                _rate(parameters["embedding_learning_rate"]),
                _rate(parameters["deep_learning_rate"]),
                "15",
                str(row["training"]["best_epoch"]),
                recall,
                _score(row["metrics"]["ndcg@100"]),
                _score(row["metrics"]["mrr@100"]),
                _score(row["metrics"]["coverage@100"]),
            ]
        )
    return _tuning_markdown(body)


def _tuning_markdown(rows: list[list[str]]) -> str:
    return _markdown_table(
        [
            "Method",
            "Trial",
            "Tokenizer",
            "Width",
            "Embedding LR",
            "Deep LR",
            "Horizon",
            "Restored epoch",
            "Recall@100",
            "NDCG@100",
            "MRR@100",
            "Coverage@100",
        ],
        rows,
    )


def _mean_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    names = set.intersection(*(set(row["metrics"]) for row in rows))
    return {
        name: statistics.fmean(float(row["metrics"][name]) for row in rows)
        for name in names
        if name.startswith("sid_exact_recall@") or name.startswith("sid_prefix_recall@")
    }


def _tokenizer(parameters: dict[str, Any]) -> str:
    return (
        f'{parameters["num_levels"]} × {parameters["num_codes"]}; '
        f'{parameters["kmeans_iterations"]} iter'
    )


def _trial(job_id: str) -> str:
    return job_id.rsplit("_", 1)[-1]


def _delta(value: float, baseline: float, band: float) -> str:
    difference = value - baseline
    rendered = f"{difference / baseline * 100:+.2f}% ({value:.3f})"
    color = "green" if difference > band else "red" if difference < -band else None
    return f'<span style="color: {color}">{rendered}</span>' if color else rendered


def _score(value: float) -> str:
    return f"{value:.3f}"


def _rate(value: float) -> str:
    return f"{value:.6g}"


def _percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def _millions(value: int) -> str:
    return f"{value / 1_000_000:.3f}M"


def _count(value: int | float | None) -> str:
    if value is None:
        return "—"
    return str(int(value)) if float(value).is_integer() else str(value)


def _optional(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| :--- | " + " | ".join(":---:" for _ in headers[1:]) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _load_pinned(path: Path, label: str) -> dict[str, Any]:
    if _sha256(path) != EXPECTED_SHA256[label]:
        raise ValueError(f"{label} changed")
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compact-output", type=Path, default=COMPACT_OUTPUT)
    parser.add_argument("--tuning-output", type=Path, default=TUNING_OUTPUT)
    args = parser.parse_args()
    evidence = load_report_evidence()
    args.compact_output.parent.mkdir(parents=True, exist_ok=True)
    args.tuning_output.parent.mkdir(parents=True, exist_ok=True)
    args.compact_output.write_text(render_compact_report(evidence))
    args.tuning_output.write_text(render_tuning_report(evidence))


if __name__ == "__main__":
    main()
