import copy

import pytest

from experiments.g3_pretrained_item_embeddings.analysis.control_calibration import (
    assess_transfer,
    calibration_document,
    fit_power_relation,
    load_control_calibration,
    persist_control_calibration,
)


def test_power_relation_is_log_linear_and_predicts_fitted_coordinates() -> None:
    fit = fit_power_relation(
        ((15, 2.0 * 15**-0.5), (25, 2.0 * 25**-0.5), (40, 2.0 * 40**-0.5))
    )

    assert fit["coefficient"] == pytest.approx(2.0)
    assert fit["exponent"] == pytest.approx(-0.5)
    assert fit["r_squared_log_space"] == pytest.approx(1.0)
    assert fit["fitted_coordinates"]["25"] == pytest.approx(0.4)


def test_transfer_check_uses_the_approved_performance_region_not_lr_distance() -> None:
    assessment = assess_transfer(
        search_recall_at_100=0.07837992363273662,
        held_out_recall_at_100=0.07897170887245977,
        relative_dispersion=0.19413750216294554,
    )

    assert assessment["accepted"] is True
    assert assessment["absolute_difference"] == pytest.approx(0.000591785239723156)
    assert assessment["operational_band"] == pytest.approx(0.01521648259378191)
    assert assessment["comparison_horizon_epochs"] == 25
    assert assessment["validates_lr_distance"] is False


def test_calibration_evidence_is_closed_hashed_and_immutable(tmp_path) -> None:
    body = {
        "control_ledger": {},
        "queue_batch": {},
        "ranking_context": {},
        "tuning_ledger": [],
        "horizon_winners": [],
        "power_law_fits": {},
        "held_out_check": {},
        "transfer_decision": {"accepted": True},
        "finding": "accepted",
    }
    document = calibration_document(body)
    path = tmp_path / "calibration.json"

    persist_control_calibration(path, document)
    assert load_control_calibration(path) == document
    changed = copy.deepcopy(document)
    changed["finding"] = "changed"
    with pytest.raises(RuntimeError, match="immutable calibration evidence"):
        persist_control_calibration(path, calibration_document(changed, replace=True))
