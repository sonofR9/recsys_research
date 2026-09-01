from pathlib import Path

import pytest

from experiments.g6_rqkmeans_history.analysis.learning_curves import (
    load_validation_curve,
)


def _write_curve(path: Path, values: list[float]) -> None:
    path.write_text(
        "\n".join(
            f"prefix epoch {epoch} finished epoch/val.loss=1.0 "
            f"epoch/val_true.recall@100={value:.4f} suffix"
            for epoch, value in enumerate(values)
        )
        + "\n"
    )


def test_validation_curve_reports_normalized_auc_and_first_95_percent_epoch(
    tmp_path: Path,
) -> None:
    log = tmp_path / "sweep.log"
    _write_curve(log, [0.2, 0.8, 1.0])

    curve = load_validation_curve(log, expected_epochs=3)

    assert curve.recall_at_100 == pytest.approx((0.2, 0.8, 1.0))
    assert curve.normalized_auc == pytest.approx(2 / 3)
    assert curve.first_epoch_at_95_percent == 3
    assert len(curve.source_sha256) == 64


def test_validation_curve_rejects_an_incomplete_horizon(tmp_path: Path) -> None:
    log = tmp_path / "sweep.log"
    _write_curve(log, [0.2, 0.8])

    with pytest.raises(ValueError, match="expected epochs 0..2"):
        load_validation_curve(log, expected_epochs=3)
