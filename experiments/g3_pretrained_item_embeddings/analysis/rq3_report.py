from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
from typing import Mapping

from experiments.g3_pretrained_item_embeddings.analysis.control_calibration import (
    _load_json,
)
from experiments.g3_pretrained_item_embeddings.analysis.rq3_boundary_results import (
    RQ3_FINAL_EVIDENCE_PATH,
    _validate_final_document,
)
from experiments.g3_pretrained_item_embeddings.launchers.control import PROJECT_ROOT
from experiments.g3_pretrained_item_embeddings.protocol.constants import (
    APPROVED_PROTOCOL,
)
from experiments.g3_pretrained_item_embeddings.protocol.rq3 import (
    RQ3_OUTPUT_FAMILY_IDS,
)


RQ3_FINAL_EVIDENCE_SHA256 = (
    "1bc6ddf12ec94a6327f97cf8dad7e7a32e376ecc6ff1b4d67f03d4213a4cff06"
)
RQ3_FINAL_EVIDENCE_FILE_SHA256 = (
    "187cdaeb8551e37ab298fb322c87fe40802a09beb36a58cf40577f8e0c16f87c"
)
RQ3_FINAL_EVIDENCE_SIZE_BYTES = 890_003
RQ3_READER_REPORT_PATH = (
    "experiments/g3_pretrained_item_embeddings/evidence/rq3_reader_native50m.md"
)
RQ3_TUNING_REPORT_PATH = (
    "experiments/g3_pretrained_item_embeddings/evidence/rq3_tuning_native50m.md"
)
_FAMILY_LABELS = {
    "rq3_output_learned": "Learned item-ID",
    "rq3_output_frozen_content": "Frozen pretrained content",
    "rq3_output_trainable_content": "Trainable pretrained content",
    "rq3_output_learned_frozen_content": "Learned ID + frozen content",
    "rq3_output_learned_trainable_content": "Learned ID + trainable content",
}
_METRICS = ("recall@100", "ndcg@100", "mrr@100", "coverage@100")
_METRIC_LABELS = {
    "recall@100": "Recall@100",
    "ndcg@100": "NDCG@100",
    "mrr@100": "MRR@100",
    "coverage@100": "Coverage@100",
}


def load_rq3_report_evidence(root: Path) -> dict[str, object]:
    path = root.resolve(strict=True) / RQ3_FINAL_EVIDENCE_PATH
    data = path.read_bytes()
    if (
        len(data) != RQ3_FINAL_EVIDENCE_SIZE_BYTES
        or hashlib.sha256(data).hexdigest() != RQ3_FINAL_EVIDENCE_FILE_SHA256
    ):
        raise ValueError("RQ3 final evidence file identity changed")
    document = _validate_final_document(_load_json(path))
    if document["sha256"] != RQ3_FINAL_EVIDENCE_SHA256:
        raise ValueError("RQ3 final evidence logical identity changed")
    return document


def render_rq3_reader_report(evidence: Mapping[str, object]) -> str:
    selections = evidence["family_selections"]
    winners = {
        family_id: selections[family_id]["selected"]
        for family_id in RQ3_OUTPUT_FAMILY_IDS
    }
    blocks = [
        "# G3 RQ3: Which prediction embedding is best?",
        _protocol_table(winners["rq3_output_learned"]),
        _selection_table(evidence),
        _quality_table(winners),
        _slice_table(winners),
        _matched_contrast_table(evidence["matched_coordinate_contrasts"]),
        _efficiency_table(winners),
        _boundary_table(evidence["family_selections"]),
        _availability_table(),
    ]
    return "\n\n".join(blocks) + "\n"


def _selection_table(evidence: Mapping[str, object]) -> str:
    decision = evidence["downstream_selection"]
    scientific = decision["best_absolute"]
    aggregate = decision["aggregate_selected"]
    reference = decision["learned_reference"]
    gain = float(decision["best_minus_learned_recall@100"])
    band = float(decision["recall@100_operational_band"])
    percent = _percent(
        float(scientific["metrics"]["recall@100"]),
        float(reference["metrics"]["recall@100"]),
    )
    promoted = decision["treatment_promoted"] is True
    return "\n".join(
        (
            "| RQ3 decision | Result |",
            "| :--- | :--- |",
            "| Scientific / RQ4 winner | "
            f"{_FAMILY_LABELS[str(scientific['family_id'])]} "
            f"(`{scientific['row_id']}`), Recall@100 "
            f"{float(scientific['metrics']['recall@100']):.6f} |",
            "| Gain over learned output | "
            f"{gain:+.6f} ({percent:+.2f}%) |",
            f"| Recall@100 operational band | ±{band:.6f} |",
            "| Aggregate promotion | "
            + ("yes" if promoted else "no; gain is inside the operational band")
            + " |",
            "| Aggregate selection | "
            f"{_FAMILY_LABELS[str(aggregate['family_id'])]} "
            f"(`{aggregate['row_id']}`) |",
        )
    )


def _protocol_table(reference: Mapping[str, object]) -> str:
    metrics = reference["metrics"]
    thresholds = ", ".join(
        f"{_METRIC_LABELS[metric]} ±{abs(float(metrics[metric])) * APPROVED_PROTOCOL.relative_dispersion('native-50m', metric):.3f}"
        for metric in _METRICS
    )
    return "\n".join(
        (
            "| Report protocol | Value |",
            "| :--- | :--- |",
            "| Dataset | native Yambda-50M likes |",
            "| Run policy | one selected run per family; batch 512; seed 42 |",
            "| Band provenance | canonical unchanged native-50M control relative dispersions, scaled to the learned-output reference |",
            f"| Operational thresholds | {thresholds} |",
        )
    )


def render_rq3_tuning_report(evidence: Mapping[str, object]) -> str:
    runs = evidence["all_tuning_opportunities"]
    selected = {
        family_id: evidence["family_selections"][family_id]["selected"]["row_id"]
        for family_id in RQ3_OUTPUT_FAMILY_IDS
    }
    blocks = ["# G3 RQ3 prediction-embedding tuning", "## RQ3 — Catalog target"]
    for family_id in RQ3_OUTPUT_FAMILY_IDS:
        family_runs = [run for run in runs if run["family_id"] == family_id]
        blocks.extend(
            (
                f"### {_FAMILY_LABELS[family_id]}",
                _tuning_table(family_runs, selected_row_id=selected[family_id]),
            )
        )
    return "\n\n".join(blocks) + "\n"


def write_rq3_reports(root: Path, evidence: Mapping[str, object]) -> tuple[Path, Path]:
    root = root.resolve(strict=True)
    reader_path = root / RQ3_READER_REPORT_PATH
    tuning_path = root / RQ3_TUNING_REPORT_PATH
    reader_path.write_text(render_rq3_reader_report(evidence))
    tuning_path.write_text(render_rq3_tuning_report(evidence))
    return reader_path, tuning_path


def _quality_table(winners: Mapping[str, Mapping[str, object]]) -> str:
    reference = winners["rq3_output_learned"]["metrics"]
    lines = [
        "| Prediction embedding | "
        + " | ".join(_METRIC_LABELS[metric] for metric in _METRICS)
        + " |",
        "| :--- | " + " | ".join(":---:" for _ in _METRICS) + " |",
    ]
    for family_id in RQ3_OUTPUT_FAMILY_IDS:
        run = winners[family_id]
        label = _FAMILY_LABELS[family_id]
        if family_id == "rq3_output_learned_frozen_content":
            label = f"**{label}**"
        values = [label]
        for metric in _METRICS:
            value = float(run["metrics"][metric])
            values.append(
                f"{value:.3f}"
                if family_id == "rq3_output_learned"
                else _metric_delta(value, float(reference[metric]), metric)
            )
        lines.append(f"| {' | '.join(values)} |")
    return "\n".join(lines)


def _slice_table(winners: Mapping[str, Mapping[str, object]]) -> str:
    reference = winners["rq3_output_learned"]["slices"]
    slices = ("head", "mid", "tail")
    lines = [
        "| Prediction embedding (descriptive frequency slices) | Head Recall@100 | Mid Recall@100 | Tail Recall@100 |",
        "| :--- | :---: | :---: | :---: |",
    ]
    for family_id in RQ3_OUTPUT_FAMILY_IDS:
        cells = [_FAMILY_LABELS[family_id]]
        for slice_name in slices:
            value = float(winners[family_id]["slices"][slice_name]["recall@100"])
            baseline = float(reference[slice_name]["recall@100"])
            cells.append(
                f"{value:.3f}"
                if family_id == "rq3_output_learned"
                else f"{_percent(value, baseline):+.1f}% ({value:.3f})"
            )
        lines.append(f"| {' | '.join(cells)} |")
    return "\n".join(lines)


def _matched_contrast_table(
    contrasts: Mapping[str, Mapping[str, object]],
) -> str:
    lines = [
        "| Matched-coordinate target contrast | Pairs | Treatment wins | Mean Recall@100 point delta | Median point delta |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]
    for contrast in contrasts.values():
        reference = contrast["reference_family_id"]
        treatment = contrast["treatment_family_id"]
        summary = contrast["summary"]
        lines.append(
            "| "
            + " | ".join(
                (
                    f"{_FAMILY_LABELS[reference]} → {_FAMILY_LABELS[treatment]}",
                    str(summary["pair_count"]),
                    str(summary["treatment_recall@100_win_count"]),
                    f"{float(summary['mean_recall@100_delta']):+.3f}",
                    f"{float(summary['median_recall@100_delta']):+.3f}",
                )
            )
            + " |"
        )
    return "\n".join(lines)


def _efficiency_table(winners: Mapping[str, Mapping[str, object]]) -> str:
    lines = [
        "| Prediction embedding (selected run) | Restored / horizon | Logged training seconds | Examples/s | Parameters | Peak GPU GB | Full-catalog observed upper bound, s |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for family_id in RQ3_OUTPUT_FAMILY_IDS:
        run = winners[family_id]
        efficiency = run["efficiency"]
        lines.append(
            "| "
            + " | ".join(
                (
                    _FAMILY_LABELS[family_id],
                    f"{run['best_epoch']} / {run['horizon_epochs']}",
                    f"{float(efficiency['logged_training_seconds']):.1f}",
                    f"{float(efficiency['examples_per_second']):.0f}",
                    str(efficiency["parameter_count"]),
                    f"{float(efficiency['peak_gpu_memory_gb']):.3f}",
                    f"{float(efficiency['full_catalog_observed_upper_bound_seconds']):.3f}",
                )
            )
            + " |"
        )
    return "\n".join(lines)


def _boundary_table(selections: Mapping[str, Mapping[str, object]]) -> str:
    lines = [
        "| Boundary family | Initial selected deep LR | Added lower probes | Final selected deep LR | Decision |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]
    for family_id in (
        "rq3_output_learned_frozen_content",
        "rq3_output_learned_trainable_content",
    ):
        decision = selections[family_id]["boundary_decision"]
        probes = ", ".join(
            f"{float(value):.7g}"
            for value in decision["tested_boundary_deep_learning_rates"]
        )
        lines.append(
            "| "
            + " | ".join(
                (
                    _FAMILY_LABELS[family_id],
                    f"{float(decision['initial_selected_deep_learning_rate']):.7g}",
                    probes,
                    f"{float(decision['selected_deep_learning_rate']):.7g}",
                    "resolved; no lower probe won",
                )
            )
            + " |"
        )
    return "\n".join(lines)


def _availability_table() -> str:
    return "\n".join(
        (
            "| Efficiency limitation | Status |",
            "| :--- | :--- |",
            "| Exact full-catalog encoding/scoring time | unavailable; the saved upper bound also includes callback, checkpoint restore, and evidence persistence |",
            "| Slice significance | descriptive only; no slice-specific repeat calibration exists |",
        )
    )


def _tuning_table(
    runs: list[Mapping[str, object]],
    *,
    selected_row_id: str,
) -> str:
    lines = [
        "| Trial | Source | Embedding LR | Deep LR | Horizon | Restored epoch | Recall@100 | NDCG@100 | MRR@100 | Coverage@100 | Logged train s | Queue wall s |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for run in runs:
        row_id = str(run["row_id"])
        trial = f"**{row_id}**" if row_id == selected_row_id else row_id
        source = (
            "reused RQ2"
            if run["reused"]
            else "lower-boundary"
            if "lower_boundary" in row_id
            else "search"
        )
        metrics = run["metrics"]
        efficiency = run["efficiency"]
        lines.append(
            "| "
            + " | ".join(
                (
                    trial,
                    source,
                    f"{float(run['embedding_learning_rate']):.10g}",
                    f"{float(run['deep_learning_rate']):.10g}",
                    str(run["horizon_epochs"]),
                    str(run["best_epoch"]),
                    f"{float(metrics['recall@100']):.6f}",
                    f"{float(metrics['ndcg@100']):.6f}",
                    f"{float(metrics['mrr@100']):.6f}",
                    f"{float(metrics['coverage@100']):.6f}",
                    f"{float(efficiency['logged_training_seconds']):.1f}",
                    f"{float(run['queue_wall_seconds']):.1f}",
                )
            )
            + " |"
        )
    return "\n".join(lines)


def _metric_delta(value: float, reference: float, metric: str) -> str:
    rendered = f"{_percent(value, reference):+.1f}% ({value:.3f})"
    band = abs(reference) * APPROVED_PROTOCOL.relative_dispersion(
        "native-50m", metric
    )
    if value - reference > band:
        return f'<span style="color: green">{rendered}</span>'
    if value - reference < -band:
        return f'<span style="color: red">{rendered}</span>'
    return rendered


def _percent(value: float, reference: float) -> float:
    if not all(math.isfinite(number) for number in (value, reference)) or reference == 0:
        raise ValueError("RQ3 percentage received invalid values")
    return 100.0 * (value / reference - 1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    arguments = parser.parse_args()
    evidence = load_rq3_report_evidence(arguments.root)
    paths = write_rq3_reports(arguments.root, evidence)
    print("\n".join(str(path) for path in paths))


if __name__ == "__main__":
    main()
