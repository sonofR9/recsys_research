from pathlib import Path

import pytest

from experiments.g6_rqkmeans_history.analysis.rq1_plus_report import (
    load_report_evidence,
    render_compact_report,
    render_tuning_report,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def evidence():
    return load_report_evidence(
        rq0_selection_path=ROOT / "evidence/rq0_selection_native50m.json",
        rq0_slices_path=ROOT / "evidence/rq0_slices_native50m.json",
        surface_path=ROOT / "evidence/rq1_rq3_surface_native50m.json",
        confirmation_path=ROOT / "evidence/rq1_rq3_confirmation_native50m.json",
        terminal_path=ROOT / "evidence/rq2_rq3_selection_native50m.json",
    )


def _section(report: str, heading: str) -> str:
    start = report.index(heading)
    end = report.find("\n## ", start + len(heading))
    return report[start:] if end == -1 else report[start:end]


def test_compact_report_contains_only_active_research_questions(evidence) -> None:
    report = render_compact_report(evidence)

    assert report.count("\n## RQ1") == 1
    assert report.count("\n## RQ2") == 1
    assert report.count("\n## RQ3") == 1
    assert "RQ4" not in report
    assert "scale effect" not in report.lower()
    assert report.count("\n## Aggregated improvement") == 1
    quality_header = (
        "| Method | Recall@100 | Delta Recall@100 | NDCG@100 | MRR@100 | "
        "Coverage@100 |"
    )
    assert report.count(quality_header) == 3
    aggregate_header = (
        "| Method | Recall@100 | Delta Recall@100 | NDCG@100 | "
        "Delta NDCG@100 | MRR@100 | Delta MRR@100 | Coverage@100 | "
        "Delta Coverage@100 |"
    )
    assert report.count(aggregate_header) == 1
    assert "Learning rate" not in report
    assert "run_name" not in report


def test_compact_report_uses_confirmation_means_and_terminal_decision(evidence) -> None:
    report = render_compact_report(evidence)
    rq1 = _section(report, "## RQ1")
    rq2 = _section(report, "## RQ2")
    rq3 = _section(report, "## RQ3")

    assert "Random initialization | 0.119 | baseline | 0.045 | 0.039 | 0.234" in rq1
    assert "**Random initialization**" not in rq1
    assert "Content-informed PCA initialization" in rq1
    assert "+5.04% (0.125)" in rq1
    assert "7.00" in rq1 and "9.25" in rq1
    assert (
        "RQ0 suffix setup (3 levels × 512 shared codes; 20 iterations) | "
        "0.127 | baseline | 0.049 | 0.047 | 0.278" in rq2
    )
    assert "Selected suffix setup (3 levels × 512 shared codes; 20 iterations)" in rq2
    assert (
        "Selected tokenizer without suffix (2 levels × 4096 shared codes; "
        "20 iterations)" in rq3
    )
    assert "+0.40% (0.127)" in rq3


def test_compact_report_keeps_rq0_to_rq3_aggregate(evidence) -> None:
    aggregate = _section(render_compact_report(evidence), "## Aggregated improvement")

    assert "### Native Yambda-50M" in aggregate
    assert "Native Yambda-500M" not in aggregate
    assert (
        "| Original G1 item-ID baseline | 0.101 | baseline | 0.037 | baseline | "
        "0.030 | baseline | 0.593 | baseline |"
        in aggregate
    )
    assert (
        "| Best-G1 plus terminal SID history | 0.130 | +0.029 (+28.70%) | "
        "0.052 | +0.015 (+41.33%) |" in aggregate
    )
    assert "| Native 50M | Recall@100 | 0.101 | 0.130 | +0.029 | +28.70% |" in aggregate
    assert "| Native 50M | Recall@100 | 0.101 | 0.130 | +0.029 | +28.70% | +0.024 | +0.005 | +0.029 | +0.000 | 0.020 | unresolved |" in aggregate


def test_tuning_report_contains_only_rq1_to_rq3_surface_rows(evidence) -> None:
    report = render_tuning_report(evidence)

    assert report.count("| Random |") == 16
    assert report.count("| Content-informed PCA |") == 16
    assert report.count("| Collision suffix |") == 40
    assert report.count("| No collision suffix |") == 40
    assert "RQ4" not in report
    assert "Native 500M SID history" not in report
    assert "**0.128**" in report
    assert "**0.130**" in report
    assert any(
        line.startswith("| No collision suffix | 03 |") and "**0.127**" in line
        for line in report.splitlines()
    )


def test_every_active_rq_has_sid_and_tokenizer_diagnostics(evidence) -> None:
    report = render_compact_report(evidence)

    assert report.count("| Method (SID diagnostics) |") == 3
    assert report.count("| Tokenizer (intrinsic diagnostics) |") == 3
    assert report.count("| Tokenizer (collision diagnostics) |") == 3
    for heading in ("## RQ1", "## RQ2", "## RQ3"):
        section = _section(report, heading)
        assert "| Method (SID diagnostics) |" in section
        assert "| Tokenizer (intrinsic diagnostics) |" in section
        assert "| Tokenizer (collision diagnostics) |" in section


def test_report_contains_rq0_slice_and_cost_diagnostics(evidence) -> None:
    report = render_compact_report(evidence)
    rq2 = _section(report, "## RQ2")
    rq3 = _section(report, "## RQ3")

    assert (
        "| Selected suffix tokenizer | 30378 | 8.36% | 14.62% | 29 | "
        "1 / 2 / 3 / 28 |" in rq2
    )
    assert "| Tail target: low | Best-G1 item-ID baseline | 0.031 |" in rq2
    assert "Selected tokenizer without suffix | tail, target/history collision slices" in rq3
    assert "Method (serving-cost estimate)" in rq2


def test_report_loader_rejects_changed_surface(tmp_path: Path, evidence) -> None:
    changed = tmp_path / "surface.json"
    changed.write_bytes(evidence.surface_path.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="RQ1-RQ3 surface changed"):
        load_report_evidence(
            rq0_selection_path=evidence.rq0_selection_path,
            rq0_slices_path=evidence.rq0_slices_path,
            surface_path=changed,
            confirmation_path=evidence.confirmation_path,
            terminal_path=evidence.terminal_path,
        )


def test_reader_report_contains_every_generated_compact_table(evidence) -> None:
    compact = render_compact_report(evidence)
    reader = (ROOT / "README.md").read_text()
    tables = []
    current = []
    for line in compact.splitlines():
        if line.startswith("|"):
            current.append(line)
        elif current:
            tables.append("\n".join(current))
            current = []
    if current:
        tables.append("\n".join(current))

    assert len(tables) >= 20
    assert all(table in reader for table in tables)
    assert all(f"## RQ{number}" in reader for number in range(1, 4))
    assert "## RQ4" not in reader
    assert "## Aggregated improvement" in reader
    assert "## Conclusions" not in reader
    assert "the RQ1 selection remains unresolved pending user validation" in reader
