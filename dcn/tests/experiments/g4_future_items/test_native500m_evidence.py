from pathlib import Path

import pytest

from experiments.g4_future_items.report.native500m_evidence import (
    final_metrics_from_log,
)


def test_native500m_final_metrics_come_from_authenticated_log(tmp_path: Path) -> None:
    log = tmp_path / "sweep.log"
    log.write_text(
        "prefix\n"
        "2026-09-01 - INFO - Final metrics ({'recall@100': 0.15, "
        "'coverage@100': 0.45, 'num_users': 37018.0}) -> /run/final_metrics.json\n"
    )

    assert final_metrics_from_log(log) == {
        "recall@100": 0.15,
        "coverage@100": 0.45,
        "num_users": 37018.0,
    }


def test_native500m_final_metrics_reject_ambiguous_log(tmp_path: Path) -> None:
    log = tmp_path / "sweep.log"
    line = "Final metrics ({'recall@100': 0.15}) -> /run/final_metrics.json\n"
    log.write_text(line + line)

    with pytest.raises(ValueError, match="exactly one"):
        final_metrics_from_log(log)
