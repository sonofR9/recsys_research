from __future__ import annotations

import re
from pathlib import Path
import shutil

import pytest

from experiments.g3_pretrained_item_embeddings.analysis.final_report import (
    COMPACT_REPORT_PATH,
    READER_REPORT_PATH,
    TUNING_REPORT_PATH,
    load_authenticated_report_evidence,
    render_reports,
    write_reports,
)
from experiments.g3_pretrained_item_embeddings.launchers.control import PROJECT_ROOT


def test_reader_and_compact_use_numbered_baseline_centered_questions() -> None:
    reports = render_reports(load_authenticated_report_evidence(PROJECT_ROOT))

    expected_headings = [
        "## RQ1: How does replacing the history item ID with pretrained content affect retrieval?",
        "## RQ2: Does concatenating pretrained content with the item-ID embedding help?",
        "## RQ3: With concatenated item-ID and pretrained inputs, which catalog target is best?",
        "## RQ4: Does adding artist and album features improve the metrics?",
        "## RQ5: Does conditioning the item-ID/content mixture on frequency improve tail retrieval?",
        "## RQ6: Does dataset size change the selected treatment's improvement?",
        "## Aggregated improvement",
    ]
    for report in (reports.reader, reports.compact):
        assert re.findall(r"^## .+$", report, re.MULTILINE) == expected_headings
        result_tables = re.findall(
            r"^\| Variant .*?(?=\n\n)", report, re.MULTILINE | re.DOTALL
        )
        assert len(result_tables) == 8
        assert all(
            table.splitlines()[2].startswith("| **Tied original learned item ID** |")
            for table in result_tables
        )
        assert report.count("percentage reference: Tied original learned item ID") == 8
        assert report.count("| **Tied original learned item ID** |") == 8
        assert (
            report.count("| **Tied original learned item ID** | 0.104 | 0.038 |") == 4
        )
        assert (
            report.count(
                "| **Tied original learned item ID** | 0.144 | 0.037 | 0.020 |"
            )
            == 4
        )
        assert "percentage reference: Untied learned item ID" not in report
        assert "percentage reference: Parameter-matched" not in report
        assert "percentage reference: Learned item-ID" not in report
        assert (
            report.count("Pending | Authenticated final evidence is not available") == 3
        )

    assert "Operational thresholds — original baseline:" in reports.reader
    assert reports.reader.count("Operational thresholds —") == 1
    assert "RQ1/RQ2 untied" not in reports.reader
    assert "RQ5 fixed gate:" not in reports.reader
    assert "original baseline remains selected" in reports.reader
    assert (
        "fails the explicit fixed/global tail-improvement criterion" in reports.reader
    )
    assert "none beats the original baseline" in reports.reader
    assert "BF16-saturated" in reports.reader


def test_reader_tables_include_only_requested_baseline_centered_rows() -> None:
    reports = render_reports(load_authenticated_report_evidence(PROJECT_ROOT))

    for report in (reports.reader, reports.compact):
        assert "Untied learned item ID" not in report
        assert "Parameter-matched ID-only DenseNet" not in report
        assert "Content-only branch ablation" not in report
        assert "Matched-coordinate target contrast" not in report
        assert "Selected variant" not in report
        assert "Queue wall" not in report
        assert "Examples/s" not in report
        assert "Parameters" not in report
        assert "Peak GPU" not in report
        assert "Full-catalog" not in report

    rq1 = reports.reader.split("## RQ1:", 1)[1].split("\n## RQ2:", 1)[0]
    assert rq1.count("| **Tied original learned item ID** |") == 2
    assert rq1.count("| Frozen content history |") == 2
    assert "ID + frozen content DenseNet" not in rq1

    rq2 = reports.reader.split("## RQ2:", 1)[1].split("\n## RQ3:", 1)[0]
    assert rq2.count("| **Tied original learned item ID** |") == 2
    assert rq2.count("| ID + frozen content DenseNet |") == 2

    rq3 = reports.reader.split("## RQ3:", 1)[1].split("\n## RQ4:", 1)[0]
    for label in (
        "Learned item-ID",
        "Frozen pretrained content",
        "Trainable pretrained content",
        "Learned ID + frozen content",
        "Learned ID + trainable content",
    ):
        assert rq3.count(f"| {label} |") == 2
    assert "wins seven of nine matched coordinates" in rq3

    rq5 = reports.reader.split("## RQ5:", 1)[1].split("\n## RQ6:", 1)[0]
    for label in (
        "Fixed concatenation",
        "Learned global scalar gate",
        "Frequency-conditioned gate",
    ):
        assert rq5.count(f"| {label} |") == 2


def test_reports_omit_all_resource_and_efficiency_fields() -> None:
    reports = render_reports(load_authenticated_report_evidence(PROJECT_ROOT))

    for report in (reports.reader, reports.compact, reports.tuning):
        for forbidden in (
            "queue wall",
            "logged training",
            "examples/s",
            "parameter count",
            "parameters",
            "peak gpu",
            "latency",
            "throughput",
            "full-catalog",
            "efficiency",
        ):
            assert forbidden not in report.lower()


def test_pending_work_remains_explicit() -> None:
    reports = render_reports(load_authenticated_report_evidence(PROJECT_ROOT))

    assert (
        reports.reader.count("Pending — authenticated final evidence is not available.")
        == 3
    )


def test_tuning_report_preserves_every_authenticated_completed_row() -> None:
    reports = render_reports(load_authenticated_report_evidence(PROJECT_ROOT))

    rows = re.findall(
        r"^\| (?:\*\*)?coordinate \d+(?:\*\*)? \|", reports.tuning, re.MULTILINE
    )
    assert len(rows) == 9 + 23 + 51 + 1 + 15 + 15
    assert reports.tuning.count("## RQ1") == 1
    assert reports.tuning.count("## RQ2") == 1
    assert reports.tuning.count("## RQ3") == 1
    assert reports.tuning.count("## RQ5") == 1
    assert "rq1_content_input:" not in reports.tuning
    assert "rq2_content_concat:" not in reports.tuning
    assert "rq3_output_" not in reports.tuning
    assert reports.tuning.count("**coordinate") == 1 + 1 + 5 + 3
    rq2 = reports.tuning[
        reports.tuning.index("## RQ2") : reports.tuning.index("## RQ3")
    ]
    assert rq2.count("| coordinate ") + rq2.count("| **coordinate ") == 23
    assert rq2.count("### ") == 1
    assert "### ID + frozen content DenseNet" in rq2
    assert "Parameter-matched ID-only DenseNet" not in reports.tuning
    assert "Content-only branch ablation" not in reports.tuning
    assert "Queue wall" not in reports.tuning
    assert "| Capacity |" in reports.tuning
    assert "| History hidden width |" in reports.tuning
    assert "| Gate hidden width |" in reports.tuning
    rq5 = reports.tuning[reports.tuning.index("## RQ5") :]
    assert rq5.count("| coordinate ") + rq5.count("| **coordinate ") == 31
    assert "### Fixed concatenation" in rq5
    assert "### Learned global scalar gate" in rq5
    assert "### Corrected frequency gate (FP32, p=0.9)" in rq5
    assert "Legacy frequency" not in rq5
    assert "BF16-saturated" in rq5


def test_checked_in_reports_match_authenticated_renderer() -> None:
    reports = render_reports(load_authenticated_report_evidence(PROJECT_ROOT))

    assert (PROJECT_ROOT / READER_REPORT_PATH).read_text() == reports.reader
    assert (PROJECT_ROOT / COMPACT_REPORT_PATH).read_text() == reports.compact
    assert (PROJECT_ROOT / TUNING_REPORT_PATH).read_text() == reports.tuning


@pytest.mark.parametrize(
    ("changed_name", "message"),
    (
        ("rq1_content_input.json", "RQ1 evidence file identity changed"),
        ("rq2_final_native50m.json", "RQ2 final evidence file identity changed"),
        (
            "rq2_unexpected_result_diagnostic_results.json",
            "RQ2 diagnostic evidence file identity changed",
        ),
        ("rq3_final_native50m.json", "RQ3 final evidence file identity changed"),
        (
            "rq5_frequency_gate_fp32_p09_v2_final_native50m.json",
            "RQ5 final evidence file identity changed",
        ),
        (
            "rq5_outcome_premechanism_native50m.json",
            "RQ5 fixed/global evidence file identity changed",
        ),
        (
            "rq5_gate_initial_native50m.json",
            "RQ5 initial evidence file identity changed",
        ),
    ),
)
def test_identity_change_fails_before_any_report_is_written(
    tmp_path: Path, changed_name: str, message: str
) -> None:
    source = PROJECT_ROOT / "experiments/g3_pretrained_item_embeddings/evidence"
    destination = tmp_path / "experiments/g3_pretrained_item_embeddings/evidence"
    destination.mkdir(parents=True)
    for name in (
        "rq1_content_input.json",
        "rq2_final_native50m.json",
        "rq2_unexpected_result_diagnostic_results.json",
        "rq3_final_native50m.json",
        "rq5_frequency_gate_fp32_p09_v2_final_native50m.json",
        "rq5_outcome_premechanism_native50m.json",
        "rq5_gate_initial_native50m.json",
    ):
        shutil.copyfile(source / name, destination / name)
    with (destination / changed_name).open("ab") as handle:
        handle.write(b"\n")

    with pytest.raises(ValueError, match=message):
        write_reports(tmp_path)

    assert not (tmp_path / READER_REPORT_PATH).exists()
    assert not (tmp_path / COMPACT_REPORT_PATH).exists()
    assert not (tmp_path / TUNING_REPORT_PATH).exists()
