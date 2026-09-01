from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Mapping, Sequence

from experiments.g3_pretrained_item_embeddings.analysis.control_calibration import (
    _load_json,
)
from experiments.g3_pretrained_item_embeddings.analysis.rq1_results import (
    APPROVED_RQ1_EVIDENCE_SHA256,
    RQ1_EVIDENCE_PATH,
    load_rq1_evidence,
)
from experiments.g3_pretrained_item_embeddings.analysis.rq2_final_results import (
    RQ2_FINAL_EVIDENCE_PATH,
)
from experiments.g3_pretrained_item_embeddings.analysis.rq3_report import (
    RQ3_FINAL_EVIDENCE_SHA256,
    load_rq3_report_evidence,
)
from experiments.g3_pretrained_item_embeddings.launchers.control import PROJECT_ROOT
from experiments.g3_pretrained_item_embeddings.protocol.constants import (
    APPROVED_PROTOCOL,
)
from experiments.g3_pretrained_item_embeddings.protocol.rq3 import (
    RQ3_OUTPUT_FAMILY_IDS,
)


READER_REPORT_PATH = "experiments/g3_pretrained_item_embeddings/README.md"
COMPACT_REPORT_PATH = (
    "experiments/g3_pretrained_item_embeddings/scratchpad/native50m.md"
)
TUNING_REPORT_PATH = (
    "experiments/g3_pretrained_item_embeddings/scratchpad/tuning_native50m.md"
)
RQ1_EVIDENCE_FILE_SHA256 = (
    "632b521be737badf996f3dadf1a38ba6218fee7402d9eb3419fe02669845c90d"
)
RQ1_EVIDENCE_SIZE_BYTES = 57_213
RQ2_FINAL_EVIDENCE_SHA256 = (
    "a8f25319858f58f3f6e5cec2a51c513d697c478044ee9f9c5c355f7a471b7856"
)
RQ2_FINAL_EVIDENCE_FILE_SHA256 = (
    "cc113d79ea895cc0eb07c42c57b99b07cd6cd8f72963e1dc89f78e4f8d0a8555"
)
RQ2_FINAL_EVIDENCE_SIZE_BYTES = 163_758
RQ5_FINAL_EVIDENCE_PATH = (
    "experiments/g3_pretrained_item_embeddings/evidence/"
    "rq5_frequency_gate_fp32_p09_v2_final_native50m.json"
)
RQ5_FINAL_EVIDENCE_SHA256 = (
    "14ed4cee2a6103aec0e38be92248a729a0619f07c1d12747d7570219c794c2b7"
)
RQ5_FINAL_EVIDENCE_FILE_SHA256 = (
    "f7981915ae85b14004b76117e1249fc65a65f18abc04e7037f34cc7c8b3d982b"
)
RQ5_FINAL_EVIDENCE_SIZE_BYTES = 95_200

_METRICS = ("recall@100", "ndcg@100", "mrr@100", "coverage@100")
_METRIC_LABELS = {
    "recall@100": "Recall@100",
    "ndcg@100": "NDCG@100",
    "mrr@100": "MRR@100",
    "coverage@100": "Coverage@100",
}
_RQ3_LABELS = {
    "rq3_output_learned": "Learned item-ID",
    "rq3_output_frozen_content": "Frozen pretrained content",
    "rq3_output_trainable_content": "Trainable pretrained content",
    "rq3_output_learned_frozen_content": "Learned ID + frozen content",
    "rq3_output_learned_trainable_content": "Learned ID + trainable content",
}
_BASELINE_LABEL = "Tied original learned item ID"


@dataclass(frozen=True)
class ReportEvidence:
    rq1: Mapping[str, object]
    rq2: Mapping[str, object]
    rq2_diagnostic: Mapping[str, object]
    rq3: Mapping[str, object]
    rq5: Mapping[str, object]
    rq5_fixed_global: Mapping[str, object]
    rq5_initial: Mapping[str, object]


@dataclass(frozen=True)
class RenderedReports:
    reader: str
    compact: str
    tuning: str


def load_authenticated_report_evidence(root: Path) -> ReportEvidence:
    root = root.resolve(strict=True)
    rq1_path = root / RQ1_EVIDENCE_PATH
    rq2_path = root / RQ2_FINAL_EVIDENCE_PATH
    _require_file_identity(
        rq1_path,
        expected_size=RQ1_EVIDENCE_SIZE_BYTES,
        expected_sha256=RQ1_EVIDENCE_FILE_SHA256,
        label="RQ1",
    )
    _require_file_identity(
        rq2_path,
        expected_size=RQ2_FINAL_EVIDENCE_SIZE_BYTES,
        expected_sha256=RQ2_FINAL_EVIDENCE_FILE_SHA256,
        label="RQ2 final",
    )
    rq1 = load_rq1_evidence(rq1_path)
    rq2 = _load_json(rq2_path)
    rq2_diagnostic_fact = rq2.get("diagnostic_evidence")
    if not isinstance(rq2_diagnostic_fact, dict):
        raise ValueError("RQ2 final evidence has no bound diagnostic evidence")
    rq2_diagnostic_path = root / str(rq2_diagnostic_fact.get("path"))
    _require_file_identity(
        rq2_diagnostic_path,
        expected_size=int(rq2_diagnostic_fact.get("size_bytes", -1)),
        expected_sha256=str(rq2_diagnostic_fact.get("sha256")),
        label="RQ2 diagnostic",
    )
    rq2_diagnostic = _load_json(rq2_diagnostic_path)
    rq3 = load_rq3_report_evidence(root)
    rq5_path = root / RQ5_FINAL_EVIDENCE_PATH
    _require_file_identity(
        rq5_path,
        expected_size=RQ5_FINAL_EVIDENCE_SIZE_BYTES,
        expected_sha256=RQ5_FINAL_EVIDENCE_FILE_SHA256,
        label="RQ5 final",
    )
    rq5 = _load_json(rq5_path)
    rq5_fixed_global = _load_bound_evidence(
        root,
        rq5,
        input_name="fixed_global_outcome",
        label="RQ5 fixed/global",
    )
    rq5_initial = _load_bound_evidence(
        root,
        rq5_fixed_global,
        input_name="initial_evidence",
        label="RQ5 initial",
    )
    evidence = ReportEvidence(
        rq1=rq1,
        rq2=rq2,
        rq2_diagnostic=rq2_diagnostic,
        rq3=rq3,
        rq5=rq5,
        rq5_fixed_global=rq5_fixed_global,
        rq5_initial=rq5_initial,
    )
    _validate_report_evidence(evidence)
    return evidence


def render_reports(evidence: ReportEvidence) -> RenderedReports:
    _validate_report_evidence(evidence)
    rq1_tables = _rq1_tables(evidence.rq1)
    rq2_tables = _rq2_tables(evidence.rq1, evidence.rq2, evidence.rq5)
    rq3_tables = _rq3_tables(evidence.rq1, evidence.rq3)
    rq5_tables = _rq5_tables(evidence.rq1, evidence.rq5)
    pending = _pending_sections()
    reader = _reader_report(
        evidence,
        rq1_tables=rq1_tables,
        rq2_tables=rq2_tables,
        rq3_tables=rq3_tables,
        rq5_tables=rq5_tables,
        pending=pending,
    )
    compact = _compact_report(
        rq1_tables=rq1_tables,
        rq2_tables=rq2_tables,
        rq3_tables=rq3_tables,
        rq5_tables=rq5_tables,
        pending=pending,
    )
    tuning = _tuning_report(evidence)
    return RenderedReports(reader=reader, compact=compact, tuning=tuning)


def write_reports(root: Path) -> tuple[Path, Path, Path]:
    root = root.resolve(strict=True)
    reports = render_reports(load_authenticated_report_evidence(root))
    outputs = (
        (root / READER_REPORT_PATH, reports.reader),
        (root / COMPACT_REPORT_PATH, reports.compact),
        (root / TUNING_REPORT_PATH, reports.tuning),
    )
    for path, _ in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    for path, content in outputs:
        path.write_text(content)
    return outputs[0][0], outputs[1][0], outputs[2][0]


def _validate_report_evidence(evidence: ReportEvidence) -> None:
    if evidence.rq1.get("sha256") != APPROVED_RQ1_EVIDENCE_SHA256:
        raise ValueError("RQ1 report evidence identity changed")
    rq1_boundary = evidence.rq1.get("boundary_decision")
    rq1_promotion = evidence.rq1.get("promotion_decision")
    if (
        not isinstance(rq1_boundary, dict)
        or rq1_boundary.get("extension_required") is not False
        or not isinstance(rq1_promotion, dict)
        or rq1_promotion.get("promoted") is not True
    ):
        raise ValueError("RQ1 report evidence is not resolved")

    if evidence.rq2.get("sha256") != RQ2_FINAL_EVIDENCE_SHA256:
        raise ValueError("RQ2 final report evidence identity changed")
    rq2_diagnostic_fact = evidence.rq2.get("diagnostic_evidence")
    if not isinstance(rq2_diagnostic_fact, dict) or evidence.rq2_diagnostic.get(
        "sha256"
    ) != rq2_diagnostic_fact.get("logical_sha256"):
        raise ValueError("RQ2 diagnostic report evidence identity changed")
    rq2_selection = evidence.rq2.get("final_content_selection")
    rq2_rows = evidence.rq2.get("all_tuning_diagnostic_boundary_ledger")
    if not isinstance(rq2_selection, dict) or not isinstance(rq2_rows, list):
        raise ValueError("RQ2 final report evidence schema changed")
    rq2_boundary = rq2_selection.get("boundary_decision")
    rq2_selected = rq2_selection.get("selected")
    if (
        rq2_selection.get("status") != "resolved"
        or not isinstance(rq2_boundary, dict)
        or rq2_boundary.get("status") != "resolved"
        or rq2_boundary.get("next_action") != "none"
        or not isinstance(rq2_selected, dict)
        or len(rq2_rows) != 36
        or sum(row.get("family_id") == "rq2_content_concat" for row in rq2_rows) != 23
        or sum(row.get("family_id") == "rq2_id_only_densenet" for row in rq2_rows) != 12
        or sum(row.get("family_id") == "rq2_content_zero_id" for row in rq2_rows) != 1
        or sum(row.get("row_id") == rq2_selected.get("row_id") for row in rq2_rows) != 1
    ):
        raise ValueError("RQ2 final report evidence is not resolved")
    comparisons = evidence.rq2_diagnostic.get("comparisons")
    parameter_match = (
        comparisons.get("c_vs_id255_parameter_match")
        if isinstance(comparisons, dict)
        else None
    )
    if (
        not isinstance(parameter_match, dict)
        or parameter_match.get("exact_parameter_match_within_one_parameter") is not True
        or parameter_match.get("id_width_255_total_parameters") != 4_377_451
        or parameter_match.get("width_128_total_parameters") != 4_377_452
        or parameter_match.get("parameter_difference") != 1
    ):
        raise ValueError("RQ2 parameter-match evidence changed")

    if evidence.rq3.get("sha256") != RQ3_FINAL_EVIDENCE_SHA256:
        raise ValueError("RQ3 final report evidence identity changed")
    rq3_rows = evidence.rq3.get("all_tuning_opportunities")
    rq3_selections = evidence.rq3.get("family_selections")
    rq3_decision = evidence.rq3.get("downstream_selection")
    if (
        not isinstance(rq3_rows, list)
        or len(rq3_rows) != 51
        or not isinstance(rq3_selections, dict)
        or set(rq3_selections) != set(RQ3_OUTPUT_FAMILY_IDS)
        or not isinstance(rq3_decision, dict)
        or rq3_decision.get("status") != "resolved"
        or rq3_decision.get("treatment_promoted") is not False
        or rq3_decision.get("unresolved_boundary_families") != []
    ):
        raise ValueError("RQ3 final report evidence is not resolved")

    if evidence.rq5.get("sha256") != RQ5_FINAL_EVIDENCE_SHA256:
        raise ValueError("RQ5 final report evidence identity changed")
    final_selection = evidence.rq5.get("final_selection")
    corrected_rows = evidence.rq5.get("valid_corrected_frequency_tuning_rows")
    legacy_policy = evidence.rq5.get("legacy_frequency_artifact_policy")
    fixed = evidence.rq5.get("fixed_comparator")
    global_gate = evidence.rq5.get("global_comparator")
    if (
        not isinstance(final_selection, dict)
        or final_selection.get("selection_resolved") is not True
        or final_selection.get("next_action") != "none"
        or final_selection.get("selected_row_id") != "rq5_frequency_gate_v2:12"
        or not isinstance(corrected_rows, list)
        or len(corrected_rows) != 15
        or {row.get("row_id") for row in corrected_rows}
        != {f"rq5_frequency_gate_v2:{index:02d}" for index in range(1, 16)}
        or any(
            row.get("family_id") != "rq5_frequency_gate_v2"
            or row.get("content_gate") != "frequency"
            for row in corrected_rows
        )
        or not isinstance(legacy_policy, dict)
        or legacy_policy.get("reader_and_tuning_eligible") is not False
        or legacy_policy.get("raw_artifacts_preserved_by_bound_audit_inputs")
        is not True
        or legacy_policy.get("semantics") != "bfloat16_p09999_saturated_zero_gradient"
        or not isinstance(fixed, dict)
        or fixed.get("row_id") != "rq2_unexpected_diagnostic:03"
        or not isinstance(global_gate, dict)
        or global_gate.get("row_id") != "rq5_global_gate:10"
    ):
        raise ValueError("RQ5 final report evidence is not resolved")
    acceptance = final_selection.get("acceptance_analysis")
    criteria = (
        acceptance.get("acceptance_criteria") if isinstance(acceptance, dict) else None
    )
    if (
        not isinstance(criteria, dict)
        or criteria.get("accepted") is not False
        or criteria.get("frequency_aggregate_not_worse_than_fixed") is not True
        or criteria.get("frequency_aggregate_not_worse_than_global") is not True
        or criteria.get("frequency_tail_higher_than_fixed") is not False
        or criteria.get("frequency_tail_higher_than_global") is not False
        or acceptance.get("selected_treatment") != "fixed_gate"
        or acceptance.get("qualifies_frequency_gate") is not False
    ):
        raise ValueError("RQ5 operational selection changed")

    if (
        evidence.rq5_fixed_global.get("sha256")
        != evidence.rq5["inputs"]["fixed_global_outcome"]["logical_sha256"]
    ):
        raise ValueError("RQ5 fixed/global logical evidence identity changed")
    if (
        evidence.rq5_initial.get("sha256")
        != evidence.rq5_fixed_global["inputs"]["initial_evidence"]["logical_sha256"]
    ):
        raise ValueError("RQ5 initial logical evidence identity changed")
    initial_rows = evidence.rq5_initial.get("runs")
    boundary_rows = evidence.rq5_fixed_global.get("boundary_runs")
    if not isinstance(initial_rows, list) or not isinstance(boundary_rows, list):
        raise ValueError("RQ5 fixed/global tuning evidence schema changed")
    global_rows = [
        row
        for row in (*initial_rows, *boundary_rows)
        if row.get("family_id") == "rq5_global_gate"
    ]
    if (
        len(initial_rows) != 21
        or sum(row.get("family_id") == "rq5_frequency_gate" for row in initial_rows)
        != 9
        or len(global_rows) != 15
        or {row.get("row_id") for row in global_rows}
        != {f"rq5_global_gate:{index:02d}" for index in range(1, 16)}
        or evidence.rq5_fixed_global.get("global_selection", {})
        .get("selected", {})
        .get("row_id")
        != global_gate.get("row_id")
    ):
        raise ValueError("RQ5 usable fixed/global tuning surface changed")

    fixed_rq2 = evidence.rq2["final_rq2_comparison"]["content_concat"]
    _require_same_metrics(fixed, fixed_rq2, label="RQ5 fixed comparator")
    _require_same_metrics(
        global_gate,
        evidence.rq5_fixed_global["global_selection"]["selected"],
        label="RQ5 global comparator",
    )
    selected_frequency = final_selection.get("selected")
    if not isinstance(selected_frequency, dict):
        raise ValueError("RQ5 corrected-frequency selection is missing")
    _require_same_metrics(
        selected_frequency,
        next(
            row
            for row in corrected_rows
            if row.get("row_id") == final_selection.get("selected_row_id")
        ),
        label="RQ5 corrected-frequency selection",
    )


def _reader_report(
    evidence: ReportEvidence,
    *,
    rq1_tables: Sequence[str],
    rq2_tables: Sequence[str],
    rq3_tables: Sequence[str],
    rq5_tables: Sequence[str],
    pending: Sequence[tuple[str, str]],
) -> str:
    thresholds = _threshold_summary(evidence)
    sections = [
        "# G3 pretrained item embeddings",
        (
            "This experiment tests frozen pretrained item content in the history "
            "input and catalog target. Completed RQ1–RQ3 and RQ5 evidence uses native "
            "Yambda-50M likes, batch 512, and one validation-selected run per family."
        ),
        (
            "Operational bands use the canonical unchanged native-50M control "
            "relative dispersions, scaled once to the tied original learned-ID baseline; they are "
            f"practical resolution bands, not significance tests. {thresholds}"
        ),
        _reader_rq1(evidence.rq1, rq1_tables),
        _reader_rq2(evidence.rq1, evidence.rq2, rq2_tables),
        _reader_rq3(evidence.rq1, evidence.rq3, rq3_tables),
        _pending_reader_section(*pending[0]),
        _reader_rq5(evidence.rq1, evidence.rq5, rq5_tables),
    ]
    sections.extend(
        _pending_reader_section(title, reason) for title, reason in pending[1:]
    )
    return "\n\n".join(sections) + "\n"


def _reader_rq1(evidence: Mapping[str, object], tables: Sequence[str]) -> str:
    comparison = evidence["comparison"]
    baseline = comparison["tied_original"]
    treatment = comparison["treatment"]
    return "\n\n".join(
        (
            "## RQ1: How does replacing the history item ID with pretrained content affect retrieval?",
            "Tied original learned item ID — shares one learned table between history and catalog.\n\n"
            "Frozen content history — replaces only the history lookup with frozen 128-dimensional content followed by a learned projection.",
            *tables,
            (
                "Conclusion: frozen-content history changes Recall@100 by "
                f"{_percentage_change(_metric(treatment, 'recall@100'), _metric(baseline, 'recall@100')):+.1f}% "
                "and remains inside the original-baseline operational band. It does not improve the original baseline, so the original baseline remains selected. Frequency-slice deltas are descriptive only because no slice-specific repeat calibration exists."
            ),
        )
    )


def _reader_rq2(
    rq1: Mapping[str, object],
    rq2: Mapping[str, object],
    tables: Sequence[str],
) -> str:
    baseline = rq1["comparison"]["tied_original"]
    treatment = rq2["final_rq2_comparison"]["content_concat"]
    return "\n\n".join(
        (
            "## RQ2: Does concatenating pretrained content with the item-ID embedding help?",
            "Tied original learned item ID — shares one learned table between history and catalog.\n\n"
            "ID + frozen content DenseNet — concatenates learned item ID with frozen content and returns to model width through DenseNet.",
            *tables,
            (
                "Conclusion: ID/content concatenation changes Recall@100 by "
                f"{_percentage_change(_metric(treatment, 'recall@100'), _metric(baseline, 'recall@100')):+.1f}% "
                "and remains inside the original-baseline operational band. It does not improve the original baseline, so the original baseline remains selected. Frequency-slice deltas are descriptive only."
            ),
        )
    )


def _reader_rq3(
    rq1: Mapping[str, object],
    evidence: Mapping[str, object],
    tables: Sequence[str],
) -> str:
    decision = evidence["downstream_selection"]
    best = decision["best_absolute"]
    baseline = rq1["comparison"]["tied_original"]
    return "\n\n".join(
        (
            "## RQ3: With concatenated item-ID and pretrained inputs, which catalog target is best?",
            "Tied original learned item ID — uses the original shared learned-ID input and target.\n\n"
            "Learned item-ID — learns the catalog table from random initialization.\n\n"
            "Frozen pretrained content — projects the fixed content table to catalog width.\n\n"
            "Trainable pretrained content — fine-tunes a content-initialized catalog table before projection.\n\n"
            "Learned ID + frozen content — projects their concatenation while keeping content fixed.\n\n"
            "Learned ID + trainable content — projects their concatenation while fine-tuning the content-initialized copy. All five targets use the selected concatenated RQ2 history input.",
            *tables,
            (
                "Conclusion: learned ID plus frozen content is the raw scientific winner at "
                f"{_metric(best, 'recall@100'):.3f} Recall@100, "
                f"{_percentage_change(_metric(best, 'recall@100'), _metric(baseline, 'recall@100')):+.1f}% versus the original baseline. "
                "It wins seven of nine matched coordinates against learned output, while trainable content beats frozen content at all nine matched coordinates, satisfying the authenticated internal ordering required by acceptance. No requested target beats the original baseline, so the original baseline remains selected. Frequency slices are descriptive only."
            ),
        )
    )


def _reader_rq5(
    rq1: Mapping[str, object],
    evidence: Mapping[str, object],
    tables: Sequence[str],
) -> str:
    selection = evidence["final_selection"]
    frequency = selection["selected"]
    fixed = evidence["fixed_comparator"]
    global_gate = evidence["global_comparator"]
    baseline = rq1["comparison"]["tied_original"]
    return "\n\n".join(
        (
            "## RQ5: Does conditioning the item-ID/content mixture on frequency improve tail retrieval?",
            "Tied original learned item ID — shares one learned table between history and catalog.\n\n"
            "Fixed concatenation — uses the selected RQ2 input with content gate fixed at one.\n\n"
            "Learned global scalar gate — learns one shared multiplier for the frozen content branch.\n\n"
            "Frequency-conditioned gate — maps standardized training-only `log1p(item count)` through a width-8 sigmoid MLP. Gate computation and its p=0.9 initialization stay in FP32 under the outer BF16 training context.",
            *tables,
            (
                "Conclusion: the corrected frequency gate fails the explicit fixed/global tail-improvement criterion: its observed tail Recall@100 is "
                f"{_percentage_change(float(frequency['slices']['tail']['recall@100']), float(fixed['slices']['tail']['recall@100'])):+.1f}% versus fixed and "
                f"{_percentage_change(float(frequency['slices']['tail']['recall@100']), float(global_gate['slices']['tail']['recall@100'])):+.1f}% versus global. "
                f"The fixed/global/frequency variants stay below it; none beats the original baseline on Recall@100. The strongest gate is {_percentage_change(_metric(global_gate, 'recall@100'), _metric(baseline, 'recall@100')):+.1f}% below it, so the original baseline remains selected. Slice deltas are descriptive because no slice-specific repeat calibration exists. The nine earlier BF16-saturated p=0.9999 frequency rows are preserved in bound raw audit evidence but excluded from reader and tuning tables because their gate gradients were zero."
            ),
        )
    )


def _compact_report(
    *,
    rq1_tables: Sequence[str],
    rq2_tables: Sequence[str],
    rq3_tables: Sequence[str],
    rq5_tables: Sequence[str],
    pending: Sequence[tuple[str, str]],
) -> str:
    sections = [
        "# G3 pretrained item embeddings — native Yambda-50M tables",
        "\n\n".join(
            (
                "## RQ1: How does replacing the history item ID with pretrained content affect retrieval?",
                *rq1_tables,
            )
        ),
        "\n\n".join(
            (
                "## RQ2: Does concatenating pretrained content with the item-ID embedding help?",
                *rq2_tables,
            )
        ),
        "\n\n".join(
            (
                "## RQ3: With concatenated item-ID and pretrained inputs, which catalog target is best?",
                *rq3_tables,
            )
        ),
        _pending_compact_section(*pending[0]),
        "\n\n".join(
            (
                "## RQ5: Does conditioning the item-ID/content mixture on frequency improve tail retrieval?",
                *rq5_tables,
            )
        ),
    ]
    sections.extend(
        _pending_compact_section(title, reason) for title, reason in pending[1:]
    )
    return "\n\n".join(sections) + "\n"


def _rq1_tables(evidence: Mapping[str, object]) -> tuple[str, ...]:
    comparison = evidence["comparison"]
    baseline = comparison["tied_original"]
    treatment = comparison["treatment"]
    rows = ((_BASELINE_LABEL, baseline), ("Frozen content history", treatment))
    quality = _quality_table(
        rows=rows,
        reference_label=_BASELINE_LABEL,
        selected_label=_BASELINE_LABEL,
        metrics=("recall@100", "ndcg@100"),
    )
    slices = _slice_table(
        rows=tuple((label, row["slices"]) for label, row in rows),
        reference_label=_BASELINE_LABEL,
        selected_label=_BASELINE_LABEL,
    )
    return quality, slices


def _rq2_tables(
    rq1: Mapping[str, object],
    rq2: Mapping[str, object],
    rq5: Mapping[str, object],
) -> tuple[str, ...]:
    baseline = rq1["comparison"]["tied_original"]
    treatment = rq2["final_rq2_comparison"]["content_concat"]
    rows = ((_BASELINE_LABEL, baseline), ("ID + frozen content DenseNet", treatment))
    quality = _quality_table(
        rows=rows,
        reference_label=_BASELINE_LABEL,
        selected_label=_BASELINE_LABEL,
        metrics=("recall@100", "ndcg@100"),
    )
    slices = _slice_table(
        rows=(
            (_BASELINE_LABEL, baseline["slices"]),
            ("ID + frozen content DenseNet", rq5["fixed_comparator"]["slices"]),
        ),
        reference_label=_BASELINE_LABEL,
        selected_label=_BASELINE_LABEL,
    )
    return quality, slices


def _rq3_tables(
    rq1: Mapping[str, object], evidence: Mapping[str, object]
) -> tuple[str, ...]:
    selections = evidence["family_selections"]
    baseline = rq1["comparison"]["tied_original"]
    rows = (
        (_BASELINE_LABEL, baseline),
        *(
            (_RQ3_LABELS[family_id], selections[family_id]["selected"])
            for family_id in RQ3_OUTPUT_FAMILY_IDS
        ),
    )
    quality = _quality_table(
        rows=rows,
        reference_label=_BASELINE_LABEL,
        selected_label=_BASELINE_LABEL,
        metrics=("recall@100", "ndcg@100"),
    )
    slices = _slice_table(
        rows=tuple((label, row["slices"]) for label, row in rows),
        reference_label=_BASELINE_LABEL,
        selected_label=_BASELINE_LABEL,
    )
    return quality, slices


def _rq5_tables(
    rq1: Mapping[str, object], evidence: Mapping[str, object]
) -> tuple[str, ...]:
    baseline = rq1["comparison"]["tied_original"]
    fixed = evidence["fixed_comparator"]
    global_gate = evidence["global_comparator"]
    frequency = evidence["final_selection"]["selected"]
    rows = (
        (_BASELINE_LABEL, baseline),
        ("Fixed concatenation", fixed),
        ("Learned global scalar gate", global_gate),
        ("Frequency-conditioned gate", frequency),
    )
    quality = _quality_table(
        rows=rows,
        reference_label=_BASELINE_LABEL,
        selected_label=_BASELINE_LABEL,
        metrics=("recall@100", "ndcg@100"),
    )
    slices = _slice_table(
        rows=tuple((label, row["slices"]) for label, row in rows),
        reference_label=_BASELINE_LABEL,
        selected_label=_BASELINE_LABEL,
    )
    return quality, slices


def _quality_table(
    *,
    rows: Sequence[tuple[str, Mapping[str, object]]],
    reference_label: str,
    selected_label: str,
    metrics: Sequence[str],
) -> str:
    by_label = dict(rows)
    reference = by_label[reference_label]
    headers = " | ".join(_METRIC_LABELS[metric] for metric in metrics)
    lines = [
        f"| Variant (percentage reference: {reference_label}) | {headers} |",
        f"| :--- | {' | '.join(':---:' for _ in metrics)} |",
    ]
    for label, row in rows:
        rendered_label = f"**{label}**" if label == selected_label else label
        cells = [rendered_label]
        for metric in metrics:
            value = _metric(row, metric)
            cells.append(
                f"{value:.3f}"
                if label == reference_label
                else _metric_delta(value, _metric(reference, metric), metric)
            )
        lines.append(f"| {' | '.join(cells)} |")
    return "\n".join(lines)


def _slice_table(
    *,
    rows: Sequence[tuple[str, Mapping[str, object]]],
    reference_label: str,
    selected_label: str,
) -> str:
    reference = dict(rows)[reference_label]
    lines = [
        f"| Variant (descriptive slices; percentage reference: {reference_label}) | Head Recall@100 | Mid Recall@100 | Tail Recall@100 |",
        "| :--- | :---: | :---: | :---: |",
    ]
    for label, slices in rows:
        rendered_label = f"**{label}**" if label == selected_label else label
        cells = [rendered_label]
        for slice_name in ("head", "mid", "tail"):
            value = float(slices[slice_name]["recall@100"])
            reference_value = float(reference[slice_name]["recall@100"])
            cells.append(
                f"{value:.3f}"
                if label == reference_label
                else f"{100.0 * (value / reference_value - 1.0):+.1f}% ({value:.3f})"
            )
        lines.append(f"| {' | '.join(cells)} |")
    lines.append(
        "| Evidence status | descriptive only; no slice-specific repeat calibration exists | — | — |"
    )
    return "\n".join(lines)


def _tuning_report(evidence: ReportEvidence) -> str:
    sections = ["# G3 pretrained item embeddings — tuning ledger"]
    sections.append(
        _tuning_section(
            "RQ1 — Content-only history input",
            (
                (
                    "Content-only history input",
                    evidence.rq1["tuning_ledger"],
                    evidence.rq1["selected_treatment"]["row_id"],
                ),
            ),
        )
    )
    rq2_rows = evidence.rq2["all_tuning_diagnostic_boundary_ledger"]
    rq2_content_rows = [
        row for row in rq2_rows if row["family_id"] == "rq2_content_concat"
    ]
    sections.append(
        _tuning_section(
            "RQ2 — Concatenated history input",
            (
                (
                    "ID + frozen content DenseNet",
                    rq2_content_rows,
                    evidence.rq2["final_content_selection"]["selected"]["row_id"],
                ),
            ),
        )
    )
    rq3_rows = evidence.rq3["all_tuning_opportunities"]
    sections.append(
        _tuning_section(
            "RQ3 — Catalog target",
            tuple(
                (
                    _RQ3_LABELS[family_id],
                    [row for row in rq3_rows if row["family_id"] == family_id],
                    evidence.rq3["family_selections"][family_id]["selected"]["row_id"],
                )
                for family_id in RQ3_OUTPUT_FAMILY_IDS
            ),
        )
    )
    fixed = evidence.rq2["final_rq2_comparison"]["content_concat"]
    initial_global = [
        row
        for row in evidence.rq5_initial["runs"]
        if row["family_id"] == "rq5_global_gate"
    ]
    global_rows = sorted(
        (*initial_global, *evidence.rq5_fixed_global["boundary_runs"]),
        key=lambda row: int(str(row["row_id"]).rsplit(":", 1)[1]),
    )
    corrected_rows = sorted(
        evidence.rq5["valid_corrected_frequency_tuning_rows"],
        key=lambda row: int(str(row["row_id"]).rsplit(":", 1)[1]),
    )
    sections.append(
        _rq5_tuning_section(
            fixed=fixed,
            global_rows=global_rows,
            selected_global_row_id=evidence.rq5["global_comparator"]["row_id"],
            corrected_rows=corrected_rows,
            selected_corrected_row_id=evidence.rq5["final_selection"][
                "selected_row_id"
            ],
        )
    )
    return "\n\n".join(sections) + "\n"


def _rq5_tuning_section(
    *,
    fixed: Mapping[str, object],
    global_rows: Sequence[Mapping[str, object]],
    selected_global_row_id: str,
    corrected_rows: Sequence[Mapping[str, object]],
    selected_corrected_row_id: str,
) -> str:
    return "\n\n".join(
        (
            "## RQ5 — Frequency-adaptive content gate",
            "### Fixed concatenation",
            _tuning_table(
                (fixed,),
                str(fixed["row_id"]),
                capacity_label="History hidden width",
            ),
            "### Learned global scalar gate",
            _tuning_table(global_rows, selected_global_row_id),
            "### Corrected frequency gate (FP32, p=0.9)",
            _tuning_table(
                corrected_rows,
                selected_corrected_row_id,
                capacity_key="gate_hidden_dim",
                capacity_label="Gate hidden width",
            ),
            "The nine legacy p=0.9999 frequency-gate rows are preserved in bound raw audit evidence but omitted here because BF16-saturated gates had zero gradients.",
        )
    )


def _tuning_section(
    heading: str,
    families: Sequence[tuple[str, Sequence[Mapping[str, object]], str]],
) -> str:
    blocks = [f"## {heading}"]
    for label, rows, selected_row_id in families:
        blocks.extend((f"### {label}", _tuning_table(rows, selected_row_id)))
    return "\n\n".join(blocks)


def _tuning_table(
    rows: Sequence[Mapping[str, object]],
    selected_row_id: str,
    *,
    capacity_key: str = "capacity",
    capacity_label: str = "Capacity",
) -> str:
    include_capacity = any(row.get(capacity_key) is not None for row in rows)
    headers = [
        "Trial",
        "Embedding LR",
        "Deep LR",
        "Declared horizon",
        "Restored epoch",
    ]
    if include_capacity:
        headers.append(capacity_label)
    headers.extend(("Recall@100", "NDCG@100"))
    lines = [
        f"| {' | '.join(headers)} |",
        f"| {' | '.join(':---:' for _ in headers)} |",
    ]
    selected_count = 0
    for index, row in enumerate(rows, start=1):
        selected = row.get("row_id") == selected_row_id
        selected_count += selected
        trial = f"**coordinate {index}**" if selected else f"coordinate {index}"
        values = [
            trial,
            f"{float(row['embedding_learning_rate']):.10g}",
            f"{float(row['deep_learning_rate']):.10g}",
            str(row["horizon_epochs"]),
            str(row["best_epoch"]),
        ]
        if include_capacity:
            capacity = row.get(capacity_key)
            values.append(str(capacity) if capacity is not None else "—")
        values.extend(
            (
                f"{_metric(row, 'recall@100'):.3f}",
                f"{_metric(row, 'ndcg@100'):.3f}",
            )
        )
        lines.append(f"| {' | '.join(values)} |")
    if selected_count != 1:
        raise ValueError(
            f"tuning family requires exactly one selected row, got {selected_count}"
        )
    return "\n".join(lines)


def _pending_sections() -> tuple[tuple[str, str], ...]:
    return (
        (
            "RQ4: Does adding artist and album features improve the metrics?",
            "RQ4 is unresolved; partial capacity evidence is excluded until an authenticated final selection passes review.",
        ),
        (
            "RQ6: Does dataset size change the selected treatment's improvement?",
            "the native-50M/native-500M size comparison is unresolved and has no authenticated final evidence.",
        ),
        (
            "Aggregated improvement",
            "the aggregate is unresolved because RQ4 and the size companion are not final; no aggregate metrics are reported.",
        ),
    )


def _pending_reader_section(title: str, reason: str) -> str:
    return "\n\n".join(
        (
            f"## {title}",
            "Pending — authenticated final evidence is not available.",
            _pending_table(reason),
        )
    )


def _pending_compact_section(title: str, reason: str) -> str:
    return "\n\n".join((f"## {title}", _pending_table(reason)))


def _pending_table(reason: str) -> str:
    return "\n".join(
        (
            "| Status | Reason |",
            "| :--- | :--- |",
            f"| Pending | Authenticated final evidence is not available; {reason} |",
        )
    )


def _threshold_summary(evidence: ReportEvidence) -> str:
    baseline = evidence.rq1["comparison"]["tied_original"]
    values = ", ".join(
        f"{_METRIC_LABELS[metric]} ±{_metric(baseline, metric) * APPROVED_PROTOCOL.relative_dispersion('native-50m', metric):.3f}"
        for metric in _METRICS
    )
    return f"Operational thresholds — original baseline: {values}."


def _metric(row: Mapping[str, object], name: str) -> float:
    metrics = row.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("report row has no metrics")
    value = metrics.get(name)
    if type(value) not in {int, float} or not math.isfinite(float(value)):
        raise ValueError(f"report row has invalid {name}")
    return float(value)


def _metric_delta(value: float, reference: float, metric: str) -> str:
    if reference == 0.0:
        raise ValueError("report percentage comparison requires a nonzero reference")
    rendered = f"{100.0 * (value / reference - 1.0):+.1f}% ({value:.3f})"
    band = reference * APPROVED_PROTOCOL.relative_dispersion("native-50m", metric)
    if value - reference > band:
        return f'<span style="color: green">{rendered}</span>'
    if value - reference < -band:
        return f'<span style="color: red">{rendered}</span>'
    return rendered


def _percentage_change(value: float, reference: float) -> float:
    if reference == 0.0:
        raise ValueError("report percentage comparison requires a nonzero reference")
    return 100.0 * (value / reference - 1.0)


def _load_bound_evidence(
    root: Path,
    parent: Mapping[str, object],
    *,
    input_name: str,
    label: str,
) -> Mapping[str, object]:
    inputs = parent.get("inputs")
    fact = inputs.get(input_name) if isinstance(inputs, dict) else None
    if not isinstance(fact, dict):
        raise ValueError(f"{label} evidence binding is missing")
    path = root / str(fact.get("path"))
    _require_file_identity(
        path,
        expected_size=int(fact.get("size_bytes", -1)),
        expected_sha256=str(fact.get("sha256")),
        label=label,
    )
    document = _load_json(path)
    if document.get("sha256") != fact.get("logical_sha256"):
        raise ValueError(f"{label} logical evidence identity changed")
    return document


def _require_same_metrics(
    left: Mapping[str, object],
    right: Mapping[str, object],
    *,
    label: str,
) -> None:
    for metric in _METRICS:
        if _metric(left, metric) != _metric(right, metric):
            raise ValueError(f"{label} metrics changed")


def _require_file_identity(
    path: Path, *, expected_size: int, expected_sha256: str, label: str
) -> None:
    data = path.read_bytes()
    if (
        len(data) != expected_size
        or hashlib.sha256(data).hexdigest() != expected_sha256
    ):
        raise ValueError(f"{label} evidence file identity changed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--write", action="store_true")
    arguments = parser.parse_args()
    root = arguments.root.resolve(strict=True)
    if arguments.write:
        paths = write_reports(root)
        print("\n".join(str(path) for path in paths))
    else:
        reports = render_reports(load_authenticated_report_evidence(root))
        print(reports.reader, end="")


if __name__ == "__main__":
    main()
