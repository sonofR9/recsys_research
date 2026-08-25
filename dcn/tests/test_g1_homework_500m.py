import runpy
import sys
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "experiments/g1_sasrec_item_ids_likes/configs/homework_500m.py"
)


def test_homework_calibration_uses_the_notebooks_dataset(monkeypatch) -> None:
    monkeypatch.setenv("G1_VARIANT", "not_a_variant")
    monkeypatch.setenv("G1_DATASET_SIZE", "50m")

    experiment = runpy.run_path(str(SCRIPT))["experiment"]

    assert experiment.size == "500m"
    assert experiment.run_name == "g1_calibrated_homework_baseline_ts2_500m"
    assert experiment.logq_correction == "baseline"
    assert experiment.user_sample is None
    assert experiment.validation_interval_seconds == 7 * 24 * 60 * 60


def test_homework_calibration_honors_seed_repeats(monkeypatch) -> None:
    monkeypatch.setenv("G1_SEED", "3")

    experiment = runpy.run_path(str(SCRIPT))["experiment"]

    assert experiment.seed == 3
    assert experiment.run_name == "g1_calibrated_homework_baseline_ts2_500m_s3"


def test_homework_calibration_clears_validation_tuning_override(monkeypatch) -> None:
    monkeypatch.setenv("G1_VAL_BATCH_SIZE", "2048")
    monkeypatch.delitem(
        sys.modules,
        "experiments.g1_sasrec_item_ids_likes.configs.variant",
        raising=False,
    )

    experiment = runpy.run_path(str(SCRIPT))["experiment"]

    assert experiment.dataloader.val_batch_size == 8192
    assert experiment.run_name == "g1_calibrated_homework_baseline_ts2_500m"
