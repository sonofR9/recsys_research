from copy import deepcopy
import os
from pathlib import Path
import runpy
import shutil
import subprocess

import pytest

from experiments.g1_sasrec_item_ids_likes.analysis import baseline_spread


EXPERIMENT = Path(__file__).resolve().parents[3] / "experiments/g1_sasrec_item_ids_likes"
LAUNCHER = EXPERIMENT / "launchers/core/baseline_spread_500m.sh"
CONFIG = EXPERIMENT / "configs/variant.py"


def _valid_metadata() -> dict:
    metadata: dict = {"dataset_size": "500m", "seed": 0}
    for path, value in baseline_spread.EXPECTED_HOMEWORK_FIELDS.items():
        target = metadata
        for key in path[:-1]:
            target = target.setdefault(key, {})
        target[path[-1]] = deepcopy(value)
    metadata.update(
        num_epochs=20,
        max_epochs=20,
        epochs_trained=9,
        best_epoch=6,
        stopped_epoch=9,
        early_stopped=True,
        best_epoch_at_cap=False,
        selection_resolved=True,
        targets_per_epoch=10,
        tokens_per_epoch=20,
        training_horizon=90,
        token_horizon=180,
        tokens_seen=180,
        optimizer_steps=7,
    )
    metadata["transfer_invariants"]["batch_size"] = 1280
    return metadata


def test_homework_repeat_accepts_validation_selected_best_epoch() -> None:
    assert baseline_spread.homework_metadata_errors(_valid_metadata(), seed=0) == []


def test_baseline_spread_markdown_omits_run_ids_and_repeat_count_column() -> None:
    summary = {
        "description": "Shared empirical resolution bands.",
        "run_prefix": "encoded-run-",
        "n": 10,
        "metrics": {
            "recall@100": {
                "mean": 0.12,
                "sample_stddev": 0.002,
                "stddev_percent_of_mean": 1.667,
            }
        },
    }

    markdown = baseline_spread.render_markdown(summary)

    assert "encoded-run" not in markdown
    assert "| metric | mean |" in markdown
    assert "| metric | n |" not in markdown


def test_homework_repeat_rejects_fixed_endpoint_or_cap_best() -> None:
    metadata = _valid_metadata()
    metadata["transfer_invariants"]["eval_every_n_epochs"] = 20
    metadata["transfer_invariants"]["restore_best_weights"] = False
    metadata["best_epoch"] = 20
    metadata["stopped_epoch"] = 20
    metadata["epochs_trained"] = 20
    metadata["best_epoch_at_cap"] = True
    metadata["selection_resolved"] = False

    errors = baseline_spread.homework_metadata_errors(metadata, seed=0)

    assert any("eval_every_n_epochs" in error for error in errors)
    assert any("restore_best_weights" in error for error in errors)
    assert any("best_epoch_at_cap" in error for error in errors)
    assert any("selection_resolved" in error for error in errors)


def test_homework_repeat_horizon_uses_actual_early_stopped_epochs() -> None:
    metadata = _valid_metadata()
    metadata["training_horizon"] = metadata["targets_per_epoch"] * 20

    errors = baseline_spread.homework_metadata_errors(metadata, seed=0)

    assert "training_horizon does not match epochs_trained" in errors


def test_native_control_contract_is_exact() -> None:
    metadata = _valid_metadata()

    assert metadata["batch_size"] == 1280
    assert metadata["physical_batch_size"] == 1280
    assert metadata["gradient_accumulation_steps"] == 1
    assert metadata["effective_batch_size"] == 1280
    assert metadata["embedding_learning_rate"] == 0.001
    assert metadata["deep_learning_rate"] == 0.002
    assert baseline_spread.homework_metadata_errors(metadata, seed=0) == []

    metadata["physical_batch_size"] = 640
    metadata["gradient_accumulation_steps"] = 2
    assert baseline_spread.homework_metadata_errors(metadata, seed=0)


def test_native_control_variant_matches_accepted_recipe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("G1_DATASET_SIZE", "500m")
    monkeypatch.setenv("G1_MAX_EPOCHS", "40")
    monkeypatch.setenv("G1_VARIANT", "homework_baseline_native500_r3")
    monkeypatch.setenv("G1_SEED", "7")

    experiment = runpy.run_path(str(CONFIG))["experiment"]

    assert experiment.run_name == (
        "g1_calibrated_homework_baseline_native500_r3_cap40_ts2_500m_s7"
    )
    assert experiment.size == "500m"
    assert experiment.seed == 7
    assert experiment.dataloader.batch_size == 1280
    assert experiment.dataloader.gradient_accumulation_steps == 1
    assert experiment.dataloader.effective_batch_size == 1280
    assert experiment.embedding_learning_rate == 0.001
    assert experiment.deep_learning_rate == 0.002
    assert experiment.eval_every_n_epochs == 1
    assert experiment.early_stopping_patience == 3
    assert experiment.early_stopping_min_delta == 0.0
    assert experiment.checkpointing.best_metric_name == "recall@100"
    assert experiment.checkpointing.best_metric_prefix == "epoch/val_true"
    assert experiment.restore_best_weights


def test_baseline_launcher_uses_revisioned_native_control_and_seed_identity(
    tmp_path: Path,
) -> None:
    launcher_directory = (
        tmp_path / "experiments/g1_sasrec_item_ids_likes/launchers"
    )
    shutil.copytree(EXPERIMENT / "launchers", launcher_directory)
    config_directory = tmp_path / "experiments/g1_sasrec_item_ids_likes/configs"
    config_directory.mkdir(parents=True)
    (config_directory / "variant.py").write_text("experiment = None\n")
    queue = tmp_path / "queue.sh"
    queue.write_text(
        "enqueue() { printf 'ENQUEUE %s\\n' \"$*\"; }\n"
        "drain() { return 0; }\n"
    )

    result = subprocess.run(
        ["bash", str(launcher_directory / "core/baseline_spread_500m.sh")],
        capture_output=True,
        text=True,
        env=os.environ
        | {
            "G1_BASELINE_SPREAD_SEEDS": "0 9",
            "G1_TRAINING_QUEUE_LIBRARY": str(queue),
        },
    )

    assert result.returncode == 0, result.stderr
    enqueues = [
        line for line in result.stdout.splitlines() if line.startswith("ENQUEUE")
    ]
    assert len(enqueues) == 2
    assert all(
        "homework_baseline_native500_r3_cap40_ts2_500m_s" in line
        for line in enqueues
    )
    assert "_s0 G1_VARIANT=homework_baseline_native500_r3" in enqueues[0]
    assert "G1_SEED=0" in enqueues[0]
    assert "G1_MAX_EPOCHS=40" in enqueues[0]
    assert "_s9 G1_VARIANT=homework_baseline_native500_r3" in enqueues[1]


def test_baseline_launcher_extended_cap_changes_run_identity(tmp_path: Path) -> None:
    queue = tmp_path / "queue.sh"
    queue.write_text(
        "enqueue() { printf 'ENQUEUE %s\\n' \"$*\"; }\n"
        "drain() { return 0; }\n"
    )

    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        capture_output=True,
        text=True,
        env=os.environ
        | {
            "G1_BASELINE_SPREAD_SEEDS": "3",
            "G1_MAX_EPOCHS": "60",
            "G1_TRAINING_QUEUE_LIBRARY": str(queue),
        },
    )

    assert result.returncode == 0, result.stderr
    assert "homework_baseline_native500_r3_cap60_ts2_500m_s3" in result.stdout
    assert "G1_MAX_EPOCHS=60" in result.stdout


def test_baseline_launcher_requires_cap_extension_for_finished_endpoint(
    tmp_path: Path,
) -> None:
    launcher_directory = (
        tmp_path / "experiments/g1_sasrec_item_ids_likes/launchers"
    )
    shutil.copytree(EXPERIMENT / "launchers", launcher_directory)
    config_directory = tmp_path / "experiments/g1_sasrec_item_ids_likes/configs"
    config_directory.mkdir(parents=True)
    (config_directory / "variant.py").write_text("experiment = None\n")
    run = (
        "g1_calibrated_homework_baseline_native500_r3_cap40_"
        "ts2_500m_s0"
    )
    (tmp_path / "generated/logs" / run).mkdir(parents=True)
    (launcher_directory / "verify_artifact.py").write_text(
        "import sys\n"
        "for line in sys.stdin:\n"
        "    request = line.rstrip().split('\\t')[1]\n"
        "    print('0' if request == 'config-recipe' else '1', flush=True)\n"
    )
    queue = tmp_path / "queue.sh"
    queue.write_text(
        "enqueue() { echo UNEXPECTED_ENQUEUE; }\n"
        "drain() { return 0; }\n"
    )

    result = subprocess.run(
        ["bash", str(launcher_directory / "core/baseline_spread_500m.sh")],
        capture_output=True,
        text=True,
        env=os.environ
        | {
            "G1_BASELINE_SPREAD_SEEDS": "0",
            "G1_TRAINING_QUEUE_LIBRARY": str(queue),
        },
    )

    assert result.returncode == 2
    assert "increase G1_MAX_EPOCHS" in result.stderr
    assert "UNEXPECTED_ENQUEUE" not in result.stdout
