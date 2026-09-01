from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from experiments.g6_rqkmeans_history.analysis.rq0_report import (
    load_report_evidence,
    render_compact_report,
    render_tuning_report,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def evidence():
    return load_report_evidence(
        selection_path=ROOT / "evidence/rq0_selection_native50m.json",
        audit_path=ROOT / "evidence/rq0_tuning_native50m.json",
        slices_path=ROOT / "evidence/rq0_slices_native50m.json",
    )


def _table_with_header(report: str, first_cell: str) -> str:
    start = report.index(f"| {first_cell} |")
    end = report.find("\n\n", start)
    return report[start:] if end == -1 else report[start:end]


def test_compact_report_keeps_quality_tables_separate_and_metrics_adjacent(evidence):
    report = render_compact_report(evidence)

    assert report.startswith("# G6: RQ-KMeans semantic IDs in history\n\n## RQ0")
    assert report.count("\n## RQ0") == 1
    primary = _table_with_header(report, "Method (best-G1 baseline)")
    bridge = _table_with_header(report, "Method (original G1 baseline)")
    residual = _table_with_header(report, "Controlled learned-SID addition")
    expected_header = "| Recall@100 | Delta Recall@100 |"
    assert expected_header in primary.splitlines()[0]
    assert expected_header in bridge.splitlines()[0]
    assert residual.splitlines()[0].startswith(
        "| Controlled learned-SID addition | Recall@100 | NDCG@100 |"
    )
    assert primary.count("\n|") == 9
    assert bridge.count("\n|") == 3
    assert residual.count("\n|") == 8
    assert "Item ID + frozen SID event" in primary
    assert "Item ID + frozen SID event" in bridge
    assert "Learning rate" not in report
    assert "Run" not in report
    assert "0.125" in primary
    assert "+3.77% (+0.005)" in primary
    assert "Item ID + frozen SID event (external tuned control)" in residual
    assert "Learned residual, bound 0.05" in residual
    assert "-3.39% (0.126)" in residual
    assert '**<span style="color: red">-3.39% (0.126)</span>**' in residual


def test_compact_report_contains_required_sid_cost_and_slice_tables(evidence):
    report = render_compact_report(evidence)

    sid = _table_with_header(report, "Method (SID retrieval diagnostics)")
    geometry = _table_with_header(report, "Method (SID geometry diagnostics)")
    primary_cost = _table_with_header(report, "Method (best-G1 serving cost)")
    original_cost = _table_with_header(report, "Method (original-G1 serving cost)")
    frequency = _table_with_header(report, "Target-frequency slice")
    collision = _table_with_header(report, "Collision-history slice")
    assert sid.count("\n|") == 8
    assert "Exact SID Recall@100" in sid
    assert "Prefix L1" in sid and "Prefix L4" in sid
    assert "ICR" in sid and "Collided items" in sid
    assert geometry.count("\n|") == 8
    assert "p95 load by level" in geometry
    assert "p95 / mean by level" in geometry
    assert "Intra-code cosine by level" in geometry
    assert "Total MACs" in primary_cost and "Embedding reads" in primary_cost
    assert primary_cost.count("\n|") == 9
    assert original_cost.count("\n|") == 3
    assert frequency.count("\n|") == 4
    assert collision.count("\n|") == 3


def test_compact_report_has_no_hand_written_interpretation(evidence):
    report = render_compact_report(evidence)

    assert all(not line or line.startswith(("#", "|")) for line in report.splitlines())


def test_tuning_report_has_one_complete_table_per_method(evidence):
    report = render_tuning_report(evidence)

    assert report.startswith("# G6 RQ0 tuning: native Yambda-50M")
    assert report.count("\n## ") == 12
    assert report.count("| Trial | Recall@100 | NDCG@100 |") == 11
    assert report.count("\n| ") - 24 == 205
    assert "Embedding LR" in report and "Deep LR" in report
    assert "Batch" in report and "Best epoch" in report
    assert "Run name" not in report and "Job ID" not in report
    assert report.count("**") == 24
    interleaved = report.split("## Interleaved item ID/SID tokens", 1)[1]
    interleaved = interleaved.split("\n## ", 1)[0]
    assert "| 12 | **0.122** |" in interleaved
    assert "**0.123**" not in interleaved
    residual = report.split(
        "## Controlled learned-SID residual: width and learning rates", 1
    )[1].split("\n## ", 1)[0]
    bounded = report.split("## Controlled learned-SID residual: gate bound", 1)[1]
    assert residual.count("\n|") == 22
    assert bounded.count("\n|") == 7
    assert "Gate bound" not in residual
    assert "Batch" in residual and "Batch" in bounded
    assert "Embedding LR" in residual and "Embedding LR" in bounded
    assert "Deep LR" in residual and "Deep LR" in bounded
    assert "Width" in residual and "Width" in bounded
    assert "| 0.05 | **0.126** |" in bounded


def test_reader_report_uses_the_generated_tables(evidence):
    generated = render_compact_report(evidence)
    reader = (ROOT / "README.md").read_text()

    assert (ROOT / "evidence/rq0_reader_native50m.md").read_text() == generated
    for table in generated.split("\n\n")[2:]:
        assert table.strip() in reader


def test_generated_tuning_report_is_current(evidence):
    assert (ROOT / "evidence/rq0_tuning_native50m.md").read_text() == (
        render_tuning_report(evidence)
    )


def test_loader_rejects_cross_evidence_selection_mismatch(tmp_path):
    selection = json.loads((ROOT / "evidence/rq0_selection_native50m.json").read_text())
    changed = deepcopy(selection)
    changed["primary_control"]["job_id"] = "wrong"
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(changed))

    with pytest.raises(ValueError, match="authenticate|selected job IDs"):
        load_report_evidence(
            selection_path=selection_path,
            audit_path=ROOT / "evidence/rq0_tuning_native50m.json",
            slices_path=ROOT / "evidence/rq0_slices_native50m.json",
        )


def test_loader_replays_slice_metrics_from_authenticated_artifacts(tmp_path):
    slices = json.loads((ROOT / "evidence/rq0_slices_native50m.json").read_text())
    slices["slices"]["frequency_low"]["semantic"]["recall@100"] += 0.01
    slices_path = tmp_path / "slices.json"
    slices_path.write_text(json.dumps(slices))

    with pytest.raises(ValueError, match="slice evidence does not match"):
        load_report_evidence(
            selection_path=ROOT / "evidence/rq0_selection_native50m.json",
            audit_path=ROOT / "evidence/rq0_tuning_native50m.json",
            slices_path=slices_path,
        )


@pytest.mark.parametrize("field", ["control", "metric_bands"])
def test_loader_binds_remediation_control_and_bands_to_base_selection(tmp_path, field):
    remediation_path = ROOT / "evidence/rq0_remediation_v3_selection_native50m.json"
    bounded_path = ROOT / "evidence/rq0_bounded_gate_v1_selection_native50m.json"
    remediation = json.loads(remediation_path.read_text())
    bounded = json.loads(bounded_path.read_text())
    if field == "control":
        remediation["control"]["metrics"]["recall@100"] += 0.01
        bounded["control"] = remediation["control"]
    else:
        remediation["metric_bands"]["recall@100"] = 1.0
        bounded["positive_noninferior"] = True
    changed_remediation_path = tmp_path / "remediation.json"
    changed_remediation_path.write_text(
        json.dumps(remediation, indent=2, sort_keys=True) + "\n"
    )
    bounded["source_selection_sha256"] = hashlib.sha256(
        changed_remediation_path.read_bytes()
    ).hexdigest()
    changed_bounded_path = tmp_path / "bounded.json"
    changed_bounded_path.write_text(json.dumps(bounded))

    with pytest.raises(ValueError, match="control|bands"):
        load_report_evidence(
            selection_path=ROOT / "evidence/rq0_selection_native50m.json",
            audit_path=ROOT / "evidence/rq0_tuning_native50m.json",
            slices_path=ROOT / "evidence/rq0_slices_native50m.json",
            remediation_selection_path=changed_remediation_path,
            bounded_gate_selection_path=changed_bounded_path,
        )
