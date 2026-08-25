import runpy
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "experiments/g1_sasrec_item_ids_likes/configs/baseline.py"
)


def test_g1_baseline_launcher_uses_the_shared_homework_setup(monkeypatch) -> None:
    monkeypatch.setenv("G1_VARIANT", "not_a_variant")
    monkeypatch.setenv("G1_DATASET_SIZE", "50m")

    experiment = runpy.run_path(str(SCRIPT))["experiment"]

    assert experiment.run_name == "g1_calibrated_baseline_ts2_50m"
    assert experiment.size == "50m"
    assert experiment.num_epochs == 20
    assert experiment.eval_every_n_epochs == 1
    assert experiment.early_stopping_patience == 3
    assert experiment.early_stopping_min_delta == 0
    assert experiment.restore_best_weights
    assert experiment.transformer.dim == 64
    assert experiment.negative_sampling == "offline_logq"
    assert experiment.initializer_std == 0.02


def test_g1_baseline_launcher_honors_seed_repeats(monkeypatch) -> None:
    monkeypatch.setenv("G1_SEED", "3")

    experiment = runpy.run_path(str(SCRIPT))["experiment"]

    assert experiment.seed == 3
    assert experiment.run_name == "g1_calibrated_baseline_ts2_500m_s3"


def test_g1_baseline_launcher_supports_extended_safety_cap(monkeypatch) -> None:
    monkeypatch.setenv("G1_MAX_EPOCHS", "30")

    experiment = runpy.run_path(str(SCRIPT))["experiment"]

    assert experiment.num_epochs == 30
    assert experiment.run_name == "g1_calibrated_baseline_cap30_ts2_500m"


def test_g1_baseline_launcher_rejects_short_safety_cap(monkeypatch) -> None:
    monkeypatch.setenv("G1_MAX_EPOCHS", "19")

    with pytest.raises(ValueError, match="at least 20"):
        runpy.run_path(str(SCRIPT))
