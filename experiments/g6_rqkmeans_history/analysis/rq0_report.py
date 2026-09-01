from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from dcn.eval.ranking_evidence import load_ranking_evidence
from dcn.semantic import SemanticCodes
from experiments.g6_rqkmeans_history.analysis.rq0_slices import slice_comparison
from experiments.g6_rqkmeans_history.launchers.remediation_manifest import (
    load_remediation_jobs,
)
from experiments.g6_rqkmeans_history.protocol.remediation_bounded import (
    bounded_gate_jobs,
)
from experiments.g6_rqkmeans_history.protocol.remediation_bounded_evidence import (
    bounded_gate_row,
    load_bounded_gate_artifact,
    select_positive_bounded_gate,
)
from experiments.g6_rqkmeans_history.protocol.remediation_evidence import (
    load_remediation_artifact,
    select_remediation_best,
    selection_row,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
LOGS_ROOT = PROJECT_ROOT / "generated/logs"
REMEDIATION_SELECTION_PATH = (
    PROJECT_ROOT / "experiments/g6_rqkmeans_history/evidence/"
    "rq0_remediation_v3_selection_native50m.json"
)
REMEDIATION_LEDGER_PATH = (
    PROJECT_ROOT / "experiments/g6_rqkmeans_history/scratchpad/"
    "rq0_remediation_v3_compiled_native50m.json"
)
BOUNDED_GATE_SELECTION_PATH = (
    PROJECT_ROOT / "experiments/g6_rqkmeans_history/evidence/"
    "rq0_bounded_gate_v1_selection_native50m.json"
)
METHODS = {
    "best_g1_item_ids": "Best G1 combination: item IDs",
    "original_g1_item_ids": "Original G1: item IDs",
    "learned_sid_event": "Trainable SID event",
    "item_frozen_sid_event": "Item ID + frozen SID event",
    "item_learned_frozen_sid_event": "Item ID + trainable/frozen SID event",
    "learned_sid_tokens": "Trainable SID tokens",
    "learned_frozen_sid_tokens": "Trainable/frozen SID tokens",
    "frozen_sid_tokens": "Frozen SID tokens",
    "interleaved_item_sid_tokens": "Interleaved item ID/SID tokens",
}
TREATMENT_ORDER = (
    "learned_sid_event",
    "item_frozen_sid_event",
    "item_learned_frozen_sid_event",
    "learned_sid_tokens",
    "learned_frozen_sid_tokens",
    "frozen_sid_tokens",
    "interleaved_item_sid_tokens",
)


@dataclass(frozen=True)
class ReportEvidence:
    selection: dict[str, Any]
    audit: dict[str, Any]
    slices: dict[str, Any]
    candidates: dict[str, dict[str, Any]]
    semantic_diagnostics: dict[str, dict[str, Any]]
    remediation: dict[str, Any]
    remediation_rows: tuple[dict[str, Any], ...]
    bounded_gate: dict[str, Any]


def load_report_evidence(
    *,
    selection_path: Path,
    audit_path: Path,
    slices_path: Path,
    remediation_selection_path: Path = REMEDIATION_SELECTION_PATH,
    remediation_ledger_path: Path = REMEDIATION_LEDGER_PATH,
    bounded_gate_selection_path: Path = BOUNDED_GATE_SELECTION_PATH,
    logs_root: Path = LOGS_ROOT,
) -> ReportEvidence:
    selection = _read_json(selection_path)
    audit = _read_json(audit_path)
    slices = _read_json(slices_path)
    if {selection["dataset_size"], audit["dataset_size"], slices["dataset_size"]} != {
        "native-50m"
    }:
        raise ValueError("report evidence uses inconsistent dataset sizes")
    if (
        len(
            {
                selection["manifest_sha256"],
                audit["manifest_sha256"],
                slices["manifest_sha256"],
            }
        )
        != 1
    ):
        raise ValueError("report evidence uses inconsistent manifests")
    if audit["selection_sha256"] != _sha256(selection_path):
        raise ValueError("audit does not authenticate the selection evidence")
    selected = audit["selected_job_ids"]
    expected = {
        "primary_control": selection["primary_control"]["job_id"],
        "original_control": selection["original_control"]["job_id"],
        "semantic_winner": selection["semantic_winner"]["job_id"],
        "selected_primary_method": selection["selected_primary_method"]["job_id"],
        "bridge": selection["bridge"]["job_id"],
        "treatment_winners": {
            method: row["job_id"]
            for method, row in selection["treatment_winners"].items()
        },
    }
    if selected != expected:
        raise ValueError("audit and selection evidence have different selected job IDs")
    if slices["selected_job_ids"] != {
        "primary_control": expected["primary_control"],
        "semantic_winner": expected["semantic_winner"],
    }:
        raise ValueError("slice and selection evidence have different selected job IDs")
    candidates = {row["job_id"]: row for row in audit["candidates"]}
    missing = _flatten_ids(expected) - candidates.keys()
    if missing:
        raise ValueError(
            f"selected candidates are absent from the audit: {sorted(missing)}"
        )
    semantic_diagnostics = {}
    for method, job_id in expected["treatment_winners"].items():
        reference = candidates[job_id]["artifacts"]["semantic_id_diagnostics.json"]
        path = PROJECT_ROOT / reference["path"]
        if _sha256(path) != reference["sha256"]:
            raise ValueError(f"semantic diagnostics hash mismatch for {method}")
        semantic_diagnostics[method] = _read_json(path)
    _verify_slice_evidence(
        slices,
        audit,
        selection,
        candidates,
        semantic_diagnostics[
            selection["semantic_winner"]["parameters"]["representation"]
        ],
    )
    remediation, remediation_rows = _load_remediation_report_evidence(
        selection_path=remediation_selection_path,
        ledger_path=remediation_ledger_path,
        logs_root=logs_root,
        base_selection=selection,
    )
    bounded_gate = _load_bounded_gate_report_evidence(
        path=bounded_gate_selection_path,
        remediation=remediation,
        remediation_selection_path=remediation_selection_path,
        logs_root=logs_root,
    )
    return ReportEvidence(
        selection=selection,
        audit=audit,
        slices=slices,
        candidates=candidates,
        semantic_diagnostics=semantic_diagnostics,
        remediation=remediation,
        remediation_rows=remediation_rows,
        bounded_gate=bounded_gate,
    )


def render_compact_report(evidence: ReportEvidence) -> str:
    selection = evidence.selection
    primary = selection["primary_control"]
    original = selection["original_control"]
    bridge = selection["bridge"]
    blocks = [
        "# G6: RQ-KMeans semantic IDs in history",
        "## RQ0 — How should SIDs describe history?",
        _quality_table(
            "Method (best-G1 baseline)",
            primary,
            [selection["treatment_winners"][method] for method in TREATMENT_ORDER],
            selection["bands"],
        ),
        _quality_table(
            "Method (original G1 baseline)",
            original,
            [bridge],
            selection["bands"],
        ),
        _learned_residual_table(evidence),
        _sid_retrieval_table(evidence),
        _sid_geometry_table(evidence),
        _cost_table(
            "Method (best-G1 serving cost)",
            [primary]
            + [selection["treatment_winners"][method] for method in TREATMENT_ORDER],
        ),
        _cost_table("Method (original-G1 serving cost)", [original, bridge]),
        _slice_table(evidence.slices, frequency=True),
        _slice_table(evidence.slices, frequency=False),
    ]
    return "\n\n".join(blocks) + "\n"


def render_tuning_report(evidence: ReportEvidence) -> str:
    selected = evidence.audit["selected_job_ids"]
    groups = [
        (
            "Best-G1 item-ID control",
            lambda row: row["stage"] == "primary_control_tuning",
            selected["primary_control"],
        ),
        (
            "Original-G1 item-ID control",
            lambda row: row["stage"] == "original_control_tuning",
            selected["original_control"],
        ),
    ]
    groups.extend(
        (
            METHODS[method],
            lambda row, method=method: row["method"] == method
            and row["stage"] in {"treatment_tuning", "lr_boundary"},
            selected["treatment_winners"][method],
        )
        for method in TREATMENT_ORDER
    )
    groups.append(
        (
            "Selected representation on original G1",
            lambda row: row["stage"] == "bridge_tuning",
            selected["bridge"],
        )
    )
    blocks = ["# G6 RQ0 tuning: native Yambda-50M"]
    for title, predicate, selected_id in groups:
        rows = [row for row in evidence.audit["candidates"] if predicate(row)]
        blocks.extend((f"## {title}", _tuning_table(rows, selected_id)))
    blocks.extend(
        (
            "## Controlled learned-SID residual: width and learning rates",
            _residual_tuning_table(
                evidence.remediation_rows,
                evidence.remediation["treatment_winner"]["job_id"],
            ),
            "## Controlled learned-SID residual: gate bound",
            _bounded_gate_tuning_table(evidence.bounded_gate),
        )
    )
    return "\n\n".join(blocks) + "\n"


def _learned_residual_table(evidence: ReportEvidence) -> str:
    control = evidence.bounded_gate["control"]
    recall = control["metrics"]["recall@100"]
    ndcg = control["metrics"]["ndcg@100"]
    bands = evidence.remediation["metric_bands"]
    rows = [
        [
            "Item ID + frozen SID event (external tuned control)",
            _score(recall),
            _score(ndcg),
            "—",
            "—",
        ]
    ]
    unbounded = evidence.remediation["treatment_winner"]
    rows.append(
        [
            "Learned residual, unbounded",
            _delta(unbounded["metrics"]["recall@100"], recall, bands["recall@100"]),
            _delta(unbounded["metrics"]["ndcg@100"], ndcg, bands["ndcg@100"]),
            "unbounded",
            "—",
        ]
    )
    winner_id = evidence.bounded_gate["positive_winner"]["job_id"]
    for row in evidence.bounded_gate["rows"]:
        bound = row["parameters"]["learned_residual_max_scale"]
        label = (
            "Learned residual disabled (shared treatment LR)"
            if bound == 0
            else f"Learned residual, bound {_rate(bound)}"
        )
        rendered_recall = _delta(
            row["metrics"]["recall@100"], recall, bands["recall@100"]
        )
        if row["job_id"] == winner_id:
            rendered_recall = f"**{rendered_recall}**"
        rows.append(
            [
                label,
                rendered_recall,
                _delta(row["metrics"]["ndcg@100"], ndcg, bands["ndcg@100"]),
                _rate(bound),
                _rate(row["learned_residual_effective_scale"]),
            ]
        )
    return _markdown_table(
        [
            "Controlled learned-SID addition",
            "Recall@100",
            "NDCG@100",
            "Gate bound",
            "Effective gate",
        ],
        rows,
    )


def _residual_tuning_table(rows: tuple[dict[str, Any], ...], selected_id: str) -> str:
    headers = [
        "Trial",
        "Recall@100",
        "NDCG@100",
        "Batch",
        "Width",
        "Embedding LR",
        "Deep LR",
        "Best epoch",
    ]
    body = []
    for index, row in enumerate(rows):
        recall = _score(row["metrics"]["recall@100"])
        if row["job_id"] == selected_id:
            recall = f"**{recall}**"
        body.append(
            [
                _trial_label(row, index),
                recall,
                _score(row["metrics"]["ndcg@100"]),
                str(row["parameters"]["batch_size"]),
                str(row["parameters"]["representation_width"]),
                _rate(row["parameters"]["embedding_learning_rate"]),
                _rate(row["parameters"]["deep_learning_rate"]),
                str(row["training"]["best_epoch"]),
            ]
        )
    return _markdown_table(headers, body)


def _bounded_gate_tuning_table(document: dict[str, Any]) -> str:
    winner_id = document["positive_winner"]["job_id"]
    body = []
    for row in document["rows"]:
        recall = _score(row["metrics"]["recall@100"])
        if row["job_id"] == winner_id:
            recall = f"**{recall}**"
        body.append(
            [
                _rate(row["parameters"]["learned_residual_max_scale"]),
                recall,
                _score(row["metrics"]["ndcg@100"]),
                str(row["parameters"]["batch_size"]),
                str(row["parameters"]["representation_width"]),
                _rate(row["parameters"]["embedding_learning_rate"]),
                _rate(row["parameters"]["deep_learning_rate"]),
                _rate(row["learned_residual_effective_scale"]),
                str(row["best_epoch"]),
            ]
        )
    return _markdown_table(
        [
            "Gate bound",
            "Recall@100",
            "NDCG@100",
            "Batch",
            "Width",
            "Embedding LR",
            "Deep LR",
            "Effective gate",
            "Best epoch",
        ],
        body,
    )


def _quality_table(
    first_header: str,
    baseline: dict[str, Any],
    treatments: list[dict[str, Any]],
    bands: dict[str, float],
) -> str:
    headers = [
        first_header,
        "Recall@100",
        "Delta Recall@100",
        "NDCG@100",
        "MRR@100",
        "Coverage@100",
        "SID configuration",
    ]
    metrics = baseline["metrics"]
    rows = [
        [
            _method_name(baseline),
            _score(metrics["recall@100"]),
            "baseline",
            _score(metrics["ndcg@100"]),
            _score(metrics["mrr@100"]),
            _score(metrics["coverage@100"]),
            "none",
        ]
    ]
    for treatment in treatments:
        treatment_metrics = treatment["metrics"]
        params = treatment["parameters"]
        rows.append(
            [
                _method_name(treatment),
                _score(treatment_metrics["recall@100"]),
                _delta(
                    treatment_metrics["recall@100"],
                    metrics["recall@100"],
                    bands["recall@100"],
                    points_first=True,
                ),
                _delta(
                    treatment_metrics["ndcg@100"],
                    metrics["ndcg@100"],
                    bands["ndcg@100"],
                ),
                _delta(
                    treatment_metrics["mrr@100"],
                    metrics["mrr@100"],
                    bands["mrr@100"],
                ),
                _delta(
                    treatment_metrics["coverage@100"],
                    metrics["coverage@100"],
                    bands["coverage@100"],
                ),
                _sid_config(params),
            ]
        )
    return _markdown_table(headers, rows)


def _sid_retrieval_table(evidence: ReportEvidence) -> str:
    headers = [
        "Method (SID retrieval diagnostics)",
        "Exact SID Recall@100",
        "Prefix L1",
        "Prefix L2",
        "Prefix L3",
        "Prefix L4",
        "ICR",
        "Collided items",
    ]
    rows = []
    for method in TREATMENT_ORDER:
        selected = evidence.selection["treatment_winners"][method]
        metrics = selected["metrics"]
        diagnostics = evidence.semantic_diagnostics[method]
        rows.append(
            [
                METHODS[method],
                _score(metrics["sid_exact_recall@100"]),
                *[
                    (
                        _score(metrics[f"sid_prefix_recall@100_l{level}"])
                        if f"sid_prefix_recall@100_l{level}" in metrics
                        else "—"
                    )
                    for level in range(1, 5)
                ],
                _percent(diagnostics["identifier_collision_rate"]),
                _percent(diagnostics["collided_item_fraction"]),
            ]
        )
    return _markdown_table(headers, rows)


def _sid_geometry_table(evidence: ReportEvidence) -> str:
    headers = [
        "Method (SID geometry diagnostics)",
        "p95 load by level",
        "p95 / mean by level",
        "Intra-code cosine by level",
    ]
    rows = []
    for method in TREATMENT_ORDER:
        diagnostics = evidence.semantic_diagnostics[method]
        rows.append(
            [
                METHODS[method],
                _levels(diagnostics["p95_occupied_load"], integer=True),
                _levels(diagnostics["p95_to_mean_occupied_load"]),
                _levels(diagnostics["intra_code_cosine_similarity"]),
            ]
        )
    return _markdown_table(headers, rows)


def _cost_table(first_header: str, selected_rows: list[dict[str, Any]]) -> str:
    headers = [
        first_header,
        "Sequence tokens",
        "Transformer MACs",
        "Tokenizer MACs",
        "Total MACs",
        "Embedding reads",
    ]
    rows = []
    for selected in selected_rows:
        cost = selected["inference_cost"]
        rows.append(
            [
                _method_name(selected),
                str(cost["sequence_tokens"]),
                _millions(cost["transformer_multiply_accumulates"]),
                _millions(cost["tokenizer_multiply_accumulates"]),
                _millions(
                    cost["transformer_multiply_accumulates"]
                    + cost["tokenizer_multiply_accumulates"]
                ),
                f'{cost["embedding_scalar_reads"]:,}',
            ]
        )
    return _markdown_table(headers, rows)


def _slice_table(document: dict[str, Any], *, frequency: bool) -> str:
    if frequency:
        header = "Target-frequency slice"
        keys = ("frequency_low", "frequency_middle", "frequency_high")
        names = ("Low", "Middle", "High")
    else:
        header = "Collision-history slice"
        keys = (
            "history_has_collided_base_sid",
            "history_has_no_collided_base_sid",
        )
        names = ("History has collided base SID", "No collided base SID in history")
    headers = [
        header,
        "Control Recall@100",
        "Semantic Recall@100",
        "Delta Recall@100",
        "Users",
        "Targets",
    ]
    rows = []
    for key, name in zip(keys, names, strict=True):
        values = document["slices"][key]
        control = values["control"]["recall@100"]
        semantic = values["semantic"]["recall@100"]
        rows.append(
            [
                name,
                _score(control),
                _score(semantic),
                f"{semantic - control:+.3f}",
                f'{values["num_users"]:,}',
                f'{values["num_targets"]:,}',
            ]
        )
    return _markdown_table(headers, rows)


def _tuning_table(rows: list[dict[str, Any]], selected_id: str) -> str:
    headers = [
        "Trial",
        "Recall@100",
        "NDCG@100",
        "MRR@100",
        "Coverage@100",
        "Batch",
        "Levels",
        "Codes / level",
        "Width",
        "Embedding LR",
        "Deep LR",
        "Best epoch",
    ]
    body = []
    for index, row in enumerate(rows):
        metrics = row["metrics"]
        params = row["parameters"]
        recall = _score(metrics["recall@100"])
        if row["job_id"] == selected_id:
            recall = f"**{recall}**"
        body.append(
            [
                _trial_label(row, index),
                recall,
                _score(metrics["ndcg@100"]),
                _score(metrics["mrr@100"]),
                _score(metrics["coverage@100"]),
                str(params["batch_size"]),
                str(params.get("num_levels", "—")),
                str(params.get("num_codes", "—")),
                str(params.get("representation_width", "—")),
                _rate(params["embedding_learning_rate"]),
                _rate(params["deep_learning_rate"]),
                str(row["training"]["best_epoch"]),
            ]
        )
    return _markdown_table(headers, body)


def _method_name(row: dict[str, Any]) -> str:
    representation = row["parameters"].get("representation")
    if representation:
        return METHODS[representation]
    run_name = row["run_name"]
    return METHODS[
        "original_g1_item_ids" if "original_g1" in run_name else "best_g1_item_ids"
    ]


def _sid_config(params: dict[str, Any]) -> str:
    return (
        f'{params["num_levels"]} levels × {params["num_codes"]} codes; '
        f'width {params["representation_width"]}'
    )


def _delta(value: float, baseline: float, band: float, *, points_first=False) -> str:
    difference = value - baseline
    percent = difference / baseline * 100
    if points_first:
        rendered = f"{percent:+.2f}% ({difference:+.3f})"
    else:
        rendered = f"{percent:+.2f}% ({value:.3f})"
    color = "green" if difference > band else "red" if difference < -band else None
    return f'<span style="color: {color}">{rendered}</span>' if color else rendered


def _score(value: float) -> str:
    return f"{value:.3f}"


def _percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def _levels(values: list[float], *, integer=False) -> str:
    if integer:
        return " / ".join(str(int(value)) for value in values)
    return " / ".join(f"{value:.3f}" for value in values)


def _millions(value: int) -> str:
    return f"{value / 1_000_000:.3f}M"


def _rate(value: float) -> str:
    return f"{value:.6g}"


def _trial_label(row: dict[str, Any], index: int) -> str:
    if row["stage"] in {"lr_boundary", "remediation_lr_boundary"}:
        params = row["parameters"]
        return f'boundary {params["boundary_side"]} {row["job_id"].rsplit("_", 1)[-1]}'
    return f"{index:02d}"


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| :--- | " + " | ".join(":---:" for _ in headers[1:]) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _flatten_ids(selected: dict[str, Any]) -> set[str]:
    return {
        value for key, value in selected.items() if key != "treatment_winners"
    } | set(selected["treatment_winners"].values())


def _verify_slice_evidence(
    slices: dict[str, Any],
    audit: dict[str, Any],
    selection: dict[str, Any],
    candidates: dict[str, dict[str, Any]],
    semantic_diagnostics: dict[str, Any],
) -> None:
    context_path = _verified_reference(audit["ranking_context"])
    control_row = candidates[selection["primary_control"]["job_id"]]
    semantic_row = candidates[selection["semantic_winner"]["job_id"]]
    control_path = _verified_reference(control_row["artifacts"]["ranking_evidence.pt"])
    semantic_path = _verified_reference(
        semantic_row["artifacts"]["ranking_evidence.pt"]
    )
    cache = audit["semantic_codebooks"]["caches"][
        semantic_diagnostics["semantic_cache_key"]
    ]
    codes_path = _verified_reference(cache["codes"])
    recomputed = slice_comparison(
        load_ranking_evidence(context_path, control_path),
        load_ranking_evidence(context_path, semantic_path),
        semantic_codes=SemanticCodes.load(codes_path),
        semantic_base_levels=semantic_row["parameters"]["num_levels"],
        control_run_name=control_row["run_name"],
        semantic_run_name=semantic_row["run_name"],
    )
    recomputed["manifest_sha256"] = audit["manifest_sha256"]
    recomputed["selected_job_ids"] = {
        "primary_control": control_row["job_id"],
        "semantic_winner": semantic_row["job_id"],
    }
    if recomputed != slices:
        raise ValueError(
            "slice evidence does not match authenticated ranking artifacts"
        )


def _load_remediation_report_evidence(
    *,
    selection_path: Path,
    ledger_path: Path,
    logs_root: Path,
    base_selection: dict[str, Any],
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    selection = _read_json(selection_path)
    if selection.get("dataset_size") != "native-50m":
        raise ValueError("remediation report evidence uses a different dataset size")
    if selection.get("control") != base_selection.get("semantic_winner"):
        raise ValueError("remediation report control changed from base selection")
    expected_bands = {
        name: base_selection["bands"][name] for name in ("recall@100", "ndcg@100")
    }
    if selection.get("metric_bands") != expected_bands:
        raise ValueError("remediation report bands changed from base selection")
    artifacts = tuple(
        load_remediation_artifact(compiled, logs_root)
        for compiled in load_remediation_jobs(ledger_path)
    )
    if len(artifacts) != 20:
        raise ValueError("remediation report evidence requires exactly 20 runs")
    winner = selection_row(select_remediation_best(artifacts))
    if winner != selection.get("treatment_winner"):
        raise ValueError("remediation report winner does not match raw artifacts")
    rows = tuple(
        {
            **selection_row(artifact),
            "stage": artifact.compiled.approved.stage,
            "training": {"best_epoch": artifact.metadata["best_epoch"]},
        }
        for artifact in artifacts
    )
    return selection, rows


def _load_bounded_gate_report_evidence(
    *,
    path: Path,
    remediation: dict[str, Any],
    remediation_selection_path: Path,
    logs_root: Path,
) -> dict[str, Any]:
    document = _read_json(path)
    if document.get("dataset_size") != "native-50m":
        raise ValueError("bounded-gate report evidence uses a different dataset size")
    if document.get("source_selection_sha256") != _sha256(remediation_selection_path):
        raise ValueError("bounded-gate report source selection changed")
    if document.get("control") != remediation.get("control"):
        raise ValueError("bounded-gate report control changed")
    artifacts = tuple(
        load_bounded_gate_artifact(compiled, logs_root)
        for compiled in bounded_gate_jobs()
    )
    rows = [bounded_gate_row(artifact) for artifact in artifacts]
    if document.get("rows") != rows or document.get("run_count") != len(rows):
        raise ValueError("bounded-gate report rows do not match raw artifacts")
    winner = bounded_gate_row(select_positive_bounded_gate(artifacts))
    if document.get("positive_winner") != winner:
        raise ValueError("bounded-gate report winner does not match raw artifacts")
    control_recall = remediation["control"]["metrics"]["recall@100"]
    control_ndcg = remediation["control"]["metrics"]["ndcg@100"]
    bands = remediation["metric_bands"]
    noninferior = (
        winner["metrics"]["recall@100"] >= control_recall - bands["recall@100"]
        and winner["metrics"]["ndcg@100"] >= control_ndcg - bands["ndcg@100"]
    )
    promoted = (
        winner["metrics"]["recall@100"] > control_recall + bands["recall@100"]
        and winner["metrics"]["ndcg@100"] >= control_ndcg - bands["ndcg@100"]
    )
    if document.get("positive_noninferior") is not noninferior:
        raise ValueError("bounded-gate report non-inferiority changed")
    if document.get("positive_promoted") is not promoted:
        raise ValueError("bounded-gate report promotion changed")
    return document


def _verified_reference(reference: dict[str, str]) -> Path:
    path = PROJECT_ROOT / reference["path"]
    if _sha256(path) != reference["sha256"]:
        raise ValueError(f"artifact hash mismatch: {reference['path']}")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--slices", type=Path, required=True)
    parser.add_argument(
        "--remediation-selection", type=Path, default=REMEDIATION_SELECTION_PATH
    )
    parser.add_argument(
        "--remediation-ledger", type=Path, default=REMEDIATION_LEDGER_PATH
    )
    parser.add_argument(
        "--bounded-gate-selection", type=Path, default=BOUNDED_GATE_SELECTION_PATH
    )
    parser.add_argument("--logs-root", type=Path, default=LOGS_ROOT)
    parser.add_argument("--compact-output", type=Path, required=True)
    parser.add_argument("--tuning-output", type=Path, required=True)
    args = parser.parse_args()
    evidence = load_report_evidence(
        selection_path=args.selection,
        audit_path=args.audit,
        slices_path=args.slices,
        remediation_selection_path=args.remediation_selection,
        remediation_ledger_path=args.remediation_ledger,
        bounded_gate_selection_path=args.bounded_gate_selection,
        logs_root=args.logs_root,
    )
    args.compact_output.parent.mkdir(parents=True, exist_ok=True)
    args.tuning_output.parent.mkdir(parents=True, exist_ok=True)
    args.compact_output.write_text(render_compact_report(evidence))
    args.tuning_output.write_text(render_tuning_report(evidence))


if __name__ == "__main__":
    main()
