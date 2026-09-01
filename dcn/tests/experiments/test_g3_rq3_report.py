from __future__ import annotations

from experiments.g3_pretrained_item_embeddings.analysis.rq3_report import (
    RQ3_READER_REPORT_PATH,
    RQ3_TUNING_REPORT_PATH,
    load_rq3_report_evidence,
    render_rq3_reader_report,
    render_rq3_tuning_report,
)
from experiments.g3_pretrained_item_embeddings.launchers.control import PROJECT_ROOT


def test_rq3_reader_report_contains_selection_contrasts_and_limitations() -> None:
    report = render_rq3_reader_report(load_rq3_report_evidence(PROJECT_ROOT))

    assert "**Learned ID + frozen content**" in report
    assert "+6.7% (0.101)" in report
    assert "native Yambda-50M likes" in report
    assert "Recall@100 ±0.018" in report
    assert "no; gain is inside the operational band" in report
    assert "`rq3_output_learned_frozen_content:04`" in report
    assert "`rq3_output_learned:08`" in report
    assert "Frozen pretrained content → Trainable pretrained content" in report
    assert "resolved; no lower probe won" in report
    assert "descriptive only; no slice-specific repeat calibration exists" in report
    assert "unavailable; the saved upper bound" in report


def test_rq3_tuning_report_preserves_all_51_logical_rows() -> None:
    evidence = load_rq3_report_evidence(PROJECT_ROOT)
    report = render_rq3_tuning_report(evidence)

    assert len(evidence["all_tuning_opportunities"]) == 51
    assert all(
        report.count(run["row_id"]) == 1
        for run in evidence["all_tuning_opportunities"]
    )
    assert report.count("**rq3_output_") == 5
    assert report.count("lower-boundary") == 6


def test_checked_in_rq3_reports_match_the_renderer() -> None:
    evidence = load_rq3_report_evidence(PROJECT_ROOT)

    assert (PROJECT_ROOT / RQ3_READER_REPORT_PATH).read_text() == (
        render_rq3_reader_report(evidence)
    )
    assert (PROJECT_ROOT / RQ3_TUNING_REPORT_PATH).read_text() == (
        render_rq3_tuning_report(evidence)
    )
