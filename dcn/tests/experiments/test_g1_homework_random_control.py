from __future__ import annotations

from dataclasses import fields
import os
from pathlib import Path
import runpy
import subprocess

import pytest

from dcn.config import GenerationExperiment


ROOT = Path(__file__).resolve().parents[3]
CONFIG = (
    ROOT
    / "experiments/g1_sasrec_item_ids_likes/configs/homework_random_control.py"
)
LAUNCHER = (
    ROOT
    / "experiments/g1_sasrec_item_ids_likes/launchers/negatives/homework_random_tuning_50m.sh"
)
SELECTED_LAUNCHER = (
    ROOT
    / "experiments/g1_sasrec_item_ids_likes/launchers/negatives/homework_random_selected_500m.sh"
)


def _load(monkeypatch: pytest.MonkeyPatch, **environment: str) -> GenerationExperiment:
    defaults = {
        "G1_HOMEWORK_RANDOM_DATASET_SIZE": "50m",
        "G1_HOMEWORK_RANDOM_RUN": "initial_e0p001_d0p002_ts2_r1",
        "G1_HOMEWORK_RANDOM_EPOCHS": "20",
        "G1_HOMEWORK_RANDOM_RUN_REVISION": "1",
        "G1_HOMEWORK_RANDOM_EMBEDDING_LR": "0.001",
        "G1_HOMEWORK_RANDOM_DEEP_LR": "0.002",
    }
    for name, value in (defaults | environment).items():
        monkeypatch.setenv(name, value)
    return runpy.run_path(str(CONFIG))["experiment"]


def test_control_changes_only_sampler_from_calibrated_homework_recipe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = _load(monkeypatch)
    monkeypatch.setenv("G1_DATASET_SIZE", "50m")
    monkeypatch.setenv("G1_MAX_EPOCHS", "20")
    monkeypatch.setenv("G1_VARIANT", "homework_fixed_leave_one_out")
    control = runpy.run_path(
        str(CONFIG.parents[0] / "variant.py")
    )["VARIANTS"]["homework_baseline_native500_r3"]

    excluded = {"run_name", "negative_sampling"}
    for field in fields(GenerationExperiment):
        if field.init and field.name not in excluded:
            assert getattr(experiment, field.name) == getattr(control, field.name)

    assert experiment.run_name == (
        "g1_homework_random_initial_e0p001_d0p002_ts2_r1_50m"
    )
    assert experiment.negative_sampling == "random"
    assert experiment.num_in_batch_negatives == 512
    assert experiment.dataloader.batch_size == 1280
    assert experiment.lr_schedule.shape == "constant"
    assert experiment.max_seq_len == 100
    assert experiment.transformer.nhead == 2
    assert experiment.transformer.num_kv_heads == 2
    assert experiment.transformer.ffn == "gelu"
    assert experiment.transformer.ffn_intermediate_dim == 256
    assert experiment.eval_every_n_epochs == 1
    assert experiment.restore_best_weights
    assert experiment.early_stopping_patience == 3
    assert experiment.early_stopping_min_delta == 0.0


def test_control_applies_only_approved_lr_axes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = _load(
        monkeypatch,
        G1_HOMEWORK_RANDOM_RUN="boundary_e0p004_d0p008_ts2_r1",
        G1_HOMEWORK_RANDOM_EMBEDDING_LR="0.004",
        G1_HOMEWORK_RANDOM_DEEP_LR="0.008",
    )

    assert experiment.embedding_learning_rate == 0.004
    assert experiment.deep_learning_rate == 0.008
    assert experiment.dataloader.batch_size == 1280
    assert experiment.num_in_batch_negatives == 512


def test_control_restores_variant_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = {
        "G1_DATASET_SIZE": "500m",
        "G1_MAX_EPOCHS": "40",
        "G1_VARIANT": "dim_32",
        "G1_MAX_USERS": "9",
        "G1_VAL_BATCH_SIZE": "64",
        "G1_TRAIN_BATCH_SIZE": "32",
    }
    for name, value in expected.items():
        monkeypatch.setenv(name, value)

    _load(monkeypatch)

    assert {name: os.environ.get(name) for name in expected} == expected


def test_control_cap_continuation_preserves_twenty_epoch_lr_horizon(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = _load(
        monkeypatch,
        G1_HOMEWORK_RANDOM_RUN="initial_e0p001_d0p002_cap40_ts2_r2",
        G1_HOMEWORK_RANDOM_EPOCHS="40",
        G1_HOMEWORK_RANDOM_RUN_REVISION="2",
    )

    assert experiment.num_epochs == 40
    assert experiment.lr_schedule_horizon_epochs == 20
    assert experiment.run_name.endswith("_cap40_ts2_r2_50m")


def test_selected_control_uses_native_500m_without_recipe_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = _load(
        monkeypatch,
        G1_HOMEWORK_RANDOM_DATASET_SIZE="500m",
        G1_HOMEWORK_RANDOM_RUN="selected_e0p002_d0p004_ts2_r1",
        G1_HOMEWORK_RANDOM_EMBEDDING_LR="0.002",
        G1_HOMEWORK_RANDOM_DEEP_LR="0.004",
    )
    monkeypatch.setenv("G1_DATASET_SIZE", "500m")
    monkeypatch.setenv("G1_MAX_EPOCHS", "20")
    monkeypatch.setenv("G1_VARIANT", "homework_fixed_leave_one_out")
    control = runpy.run_path(
        str(CONFIG.parents[0] / "variant.py")
    )["VARIANTS"]["homework_baseline_native500_r3"]

    excluded = {
        "run_name",
        "negative_sampling",
        "embedding_learning_rate",
        "deep_learning_rate",
    }
    for field in fields(GenerationExperiment):
        if field.init and field.name not in excluded:
            assert getattr(experiment, field.name) == getattr(control, field.name)

    assert experiment.run_name == (
        "g1_homework_random_selected_e0p002_d0p004_ts2_r1_500m"
    )
    assert experiment.size == "500m"
    assert experiment.dataloader.batch_size == 1280
    assert experiment.num_in_batch_negatives == 512
    assert experiment.negative_sampling == "random"


def _queue_stub(tmp_path: Path) -> tuple[Path, Path]:
    output = tmp_path / "queued.txt"
    stub = tmp_path / "queue.sh"
    stub.write_text(
        "enqueue() { printf '%s\\n' \"$*\" >> \"$G1_TEST_QUEUE_OUTPUT\"; }\n"
        "drain() { printf 'drain\\n' >> \"$G1_TEST_QUEUE_OUTPUT\"; }\n"
        "g1_artifact_exists() { "
        "if [[ -n ${G1_TEST_EXISTING_ARTIFACTS+x} ]]; then "
        "[[ ' '$G1_TEST_EXISTING_ARTIFACTS' ' == *' '$(basename \"$1\")' '* ]]; "
        "else [[ -e $1 || -L $1 ]]; fi; }\n"
        "g1_classify_config_artifact() { "
        "_g1_artifact_state=${G1_TEST_PREDECESSOR_STATE:-resumable}; }\n"
        "g1_require_config_compatible_or_absent() { return 1; }\n"
    )
    return stub, output


def _launch(tmp_path: Path, **environment: str) -> subprocess.CompletedProcess[str]:
    queue_stub, output = _queue_stub(tmp_path)
    variables = os.environ | {
        "G1_TRAINING_QUEUE_LIBRARY": str(queue_stub),
        "G1_TEST_QUEUE_OUTPUT": str(output),
        "G1_HOMEWORK_RANDOM_RUN_TAG": "testgrid",
        "G1_HOMEWORK_RANDOM_EMBEDDING_LRS": "0.0005 0.001",
        "G1_HOMEWORK_RANDOM_DEEP_LRS": "0.002 0.004",
    }
    variables.update(environment)
    return subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=ROOT,
        env=variables,
        text=True,
        capture_output=True,
        check=False,
    )


def test_launcher_submits_exact_cartesian_grid(tmp_path: Path) -> None:
    result = _launch(tmp_path)

    assert result.returncode == 0, result.stderr
    lines = (tmp_path / "queued.txt").read_text().splitlines()
    assert lines[-1] == "drain"
    jobs = lines[:-1]
    assert len(jobs) == 4
    assert {
        tuple(
            assignment
            for assignment in job.split()
            if assignment.startswith(
                (
                    "G1_HOMEWORK_RANDOM_EMBEDDING_LR=",
                    "G1_HOMEWORK_RANDOM_DEEP_LR=",
                )
            )
        )
        for job in jobs
    } == {
        (
            "G1_HOMEWORK_RANDOM_EMBEDDING_LR=0.0005",
            "G1_HOMEWORK_RANDOM_DEEP_LR=0.002",
        ),
        (
            "G1_HOMEWORK_RANDOM_EMBEDDING_LR=0.0005",
            "G1_HOMEWORK_RANDOM_DEEP_LR=0.004",
        ),
        (
            "G1_HOMEWORK_RANDOM_EMBEDDING_LR=0.001",
            "G1_HOMEWORK_RANDOM_DEEP_LR=0.002",
        ),
        (
            "G1_HOMEWORK_RANDOM_EMBEDDING_LR=0.001",
            "G1_HOMEWORK_RANDOM_DEEP_LR=0.004",
        ),
    }
    assert all("G1_HOMEWORK_RANDOM_EPOCHS=20" in job for job in jobs)
    assert all("G1_HOMEWORK_RANDOM_RUN_REVISION=1" in job for job in jobs)


def test_launcher_encodes_cap_continuation(tmp_path: Path) -> None:
    result = _launch(
        tmp_path,
        G1_HOMEWORK_RANDOM_RUN_TAG="capcontinue",
        G1_HOMEWORK_RANDOM_EPOCHS="40",
        G1_HOMEWORK_RANDOM_RUN_REVISION="2",
    )

    assert result.returncode == 0, result.stderr
    jobs = (tmp_path / "queued.txt").read_text().splitlines()[:-1]
    assert all("_capcontinue_" in job for job in jobs)
    assert all("_cap40_ts2_r2_50m" in job for job in jobs)
    assert all("G1_HOMEWORK_RANDOM_EPOCHS=40" in job for job in jobs)


def test_selected_launcher_run_name_survives_sourced_queue_globals(
    tmp_path: Path,
) -> None:
    queue_stub, output = _queue_stub(tmp_path)
    selector = tmp_path / "selector.py"
    selector.write_text("print('0.0021:0.0041')\n")
    with queue_stub.open("a") as stream:
        stream.write("name=WANDB_MODE\n")
    result = subprocess.run(
        ["bash", str(SELECTED_LAUNCHER)],
        cwd=ROOT,
        env=os.environ
        | {
            "G1_TRAINING_QUEUE_LIBRARY": str(queue_stub),
            "G1_TEST_QUEUE_OUTPUT": str(output),
            "G1_HOMEWORK_RANDOM_SELECTION": "0.0021:0.0041",
            "G1_TEST_HOMEWORK_CONTROL_SELECTOR": str(selector),
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    job = output.read_text().splitlines()[0]
    assert job.split()[0] == (
        "g1_homework_random_selected_e0p0021_d0p0041_ts2_r1_500m"
    )


def test_selected_launcher_uses_selector_canonical_rates(tmp_path: Path) -> None:
    queue_stub, output = _queue_stub(tmp_path)
    selector = tmp_path / "selector.py"
    selector.write_text("print('0.0021:0.0041')\n")
    result = subprocess.run(
        ["bash", str(SELECTED_LAUNCHER)],
        cwd=ROOT,
        env=os.environ
        | {
            "G1_TRAINING_QUEUE_LIBRARY": str(queue_stub),
            "G1_TEST_QUEUE_OUTPUT": str(output),
            "G1_HOMEWORK_RANDOM_SELECTION": (
                "0.0021000000000000001:0.0041000000000000002"
            ),
            "G1_TEST_HOMEWORK_CONTROL_SELECTOR": str(selector),
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    job = output.read_text().splitlines()[0]
    assert job.split()[0] == (
        "g1_homework_random_selected_e0p0021_d0p0041_ts2_r1_500m"
    )
    assert "G1_HOMEWORK_RANDOM_EMBEDDING_LR=0.0021" in job
    assert "G1_HOMEWORK_RANDOM_DEEP_LR=0.0041" in job


@pytest.mark.parametrize(
    ("epochs", "revision"),
    [("20", "2"), ("40", "1"), ("60", "2")],
)
def test_selected_launcher_rejects_invalid_cap_revision(
    tmp_path: Path, epochs: str, revision: str
) -> None:
    queue_stub, output = _queue_stub(tmp_path)
    selector = tmp_path / "selector.py"
    selector.write_text("print('0.002:0.004')\n")
    result = subprocess.run(
        ["bash", str(SELECTED_LAUNCHER)],
        cwd=ROOT,
        env=os.environ
        | {
            "G1_TRAINING_QUEUE_LIBRARY": str(queue_stub),
            "G1_TEST_QUEUE_OUTPUT": str(output),
            "G1_HOMEWORK_RANDOM_SELECTION": "0.002:0.004",
            "G1_TEST_HOMEWORK_CONTROL_SELECTOR": str(selector),
            "G1_HOMEWORK_RANDOM_EPOCHS": epochs,
            "G1_HOMEWORK_RANDOM_RUN_REVISION": revision,
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "20/r1, 40/r2" in result.stderr
    assert not output.exists()


def test_selected_launcher_requires_exact_cap_predecessor(tmp_path: Path) -> None:
    queue_stub, output = _queue_stub(tmp_path)
    selector = tmp_path / "selector.py"
    selector.write_text("print('0.0021:0.0041')\n")
    environment = os.environ | {
        "G1_TRAINING_QUEUE_LIBRARY": str(queue_stub),
        "G1_TEST_QUEUE_OUTPUT": str(output),
        "G1_HOMEWORK_RANDOM_SELECTION": "0.0021:0.0041",
        "G1_TEST_HOMEWORK_CONTROL_SELECTOR": str(selector),
        "G1_HOMEWORK_RANDOM_EPOCHS": "40",
        "G1_HOMEWORK_RANDOM_RUN_REVISION": "2",
    }
    missing = subprocess.run(
        ["bash", str(SELECTED_LAUNCHER)],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert missing.returncode == 2
    assert "Missing selected 500M cap predecessor" in missing.stderr
    assert not output.exists()

    selector.write_text("print('0.002:0.004')\n")
    environment["G1_HOMEWORK_RANDOM_SELECTION"] = "0.002:0.004"
    predecessor = (
        ROOT
        / "generated/logs/g1_homework_random_selected_e0p002_d0p004_ts2_r1_500m"
    )
    assert predecessor.is_dir()
    compatible = subprocess.run(
        ["bash", str(SELECTED_LAUNCHER)],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert compatible.returncode == 0, compatible.stderr
    assert "_cap40_ts2_r2_500m" in output.read_text()


def test_selected_launcher_rejects_cap_chain_without_base(tmp_path: Path) -> None:
    queue_stub, output = _queue_stub(tmp_path)
    selector = tmp_path / "selector.py"
    selector.write_text("print('0.0021:0.0041')\n")
    cap40_name = (
        "g1_homework_random_selected_e0p0021_d0p0041_cap40_ts2_r2_500m"
    )
    result = subprocess.run(
        ["bash", str(SELECTED_LAUNCHER)],
        cwd=ROOT,
        env=os.environ
        | {
            "G1_TRAINING_QUEUE_LIBRARY": str(queue_stub),
            "G1_TEST_QUEUE_OUTPUT": str(output),
            "G1_HOMEWORK_RANDOM_SELECTION": "0.0021:0.0041",
            "G1_TEST_HOMEWORK_CONTROL_SELECTOR": str(selector),
            "G1_HOMEWORK_RANDOM_EPOCHS": "80",
            "G1_HOMEWORK_RANDOM_RUN_REVISION": "3",
            "G1_TEST_EXISTING_ARTIFACTS": cap40_name,
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "Missing selected 500M cap predecessor" in result.stderr
    assert "_ts2_r1_500m" in result.stderr
    assert not output.exists()


def test_selected_launcher_requires_cap_hit_predecessor(tmp_path: Path) -> None:
    queue_stub, output = _queue_stub(tmp_path)
    selector = tmp_path / "selector.py"
    selector.write_text("print('0.002:0.004')\n")
    result = subprocess.run(
        ["bash", str(SELECTED_LAUNCHER)],
        cwd=ROOT,
        env=os.environ
        | {
            "G1_TRAINING_QUEUE_LIBRARY": str(queue_stub),
            "G1_TEST_QUEUE_OUTPUT": str(output),
            "G1_HOMEWORK_RANDOM_SELECTION": "0.002:0.004",
            "G1_TEST_HOMEWORK_CONTROL_SELECTOR": str(selector),
            "G1_HOMEWORK_RANDOM_EPOCHS": "40",
            "G1_HOMEWORK_RANDOM_RUN_REVISION": "2",
            "G1_TEST_PREDECESSOR_STATE": "complete",
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "did not hit its cap" in result.stderr
    assert not output.exists()


def test_launcher_requires_tag_for_boundary_extension(tmp_path: Path) -> None:
    result = _launch(tmp_path, G1_HOMEWORK_RANDOM_RUN_TAG="")

    assert result.returncode == 2
    assert "unique G1_HOMEWORK_RANDOM_RUN_TAG" in result.stderr
    assert not (tmp_path / "queued.txt").exists()


@pytest.mark.parametrize(
    "environment,message",
    [
        (
            {
                "G1_HOMEWORK_RANDOM_EPOCHS": "19",
                "G1_HOMEWORK_RANDOM_RUN_REVISION": "1",
            },
            "at least 20",
        ),
        (
            {"G1_HOMEWORK_RANDOM_RUN_REVISION": "2"},
            "must be set together",
        ),
        ({"G1_HOMEWORK_RANDOM_EMBEDDING_LRS": "0 0.001"}, "positive number"),
        (
            {
                "G1_HOMEWORK_RANDOM_EMBEDDING_LRS": (
                    "0.1 0.100000000000000005"
                )
            },
            "duplicate",
        ),
        ({"G1_HOMEWORK_RANDOM_DEEP_LRS": "0.002 0.002"}, "duplicate"),
        ({"G1_HOMEWORK_RANDOM_DEEP_LRS": ""}, "cannot be empty"),
        (
            {
                "G1_HOMEWORK_RANDOM_RUN_TAG": "initial",
                "G1_HOMEWORK_RANDOM_EPOCHS": "40",
                "G1_HOMEWORK_RANDOM_RUN_REVISION": "2",
            },
            "reserved",
        ),
        ({"G1_HOMEWORK_RANDOM_RUN_TAG": "capcontinue"}, "reserved"),
    ],
)
def test_launcher_rejects_invalid_grid_before_queue_submission(
    tmp_path: Path,
    environment: dict[str, str],
    message: str,
) -> None:
    result = _launch(tmp_path, **environment)

    assert result.returncode == 2
    assert message in result.stderr
    assert not (tmp_path / "queued.txt").exists()
