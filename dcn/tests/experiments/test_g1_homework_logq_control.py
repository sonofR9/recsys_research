from __future__ import annotations

from dataclasses import fields
import os
from pathlib import Path
import runpy
import subprocess

import pytest

from dcn.config import GenerationExperiment


ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "experiments/g1_sasrec_item_ids_likes/configs/homework_logq_control.py"
LAUNCHER = ROOT / "experiments/g1_sasrec_item_ids_likes/launchers/negatives/homework_logq_tuning_50m.sh"
SELECTOR = ROOT / "experiments/g1_sasrec_item_ids_likes/analysis/select_homework_negative_control.py"
VERIFY_ARTIFACT = ROOT / "experiments/g1_sasrec_item_ids_likes/launchers/verify_artifact.py"


def test_logq_tuning_changes_only_learning_rates_and_identity(monkeypatch) -> None:
    environment = {
        "G1_HOMEWORK_LOGQ_RUN": "initial_e0p001_d0p002_ts2_r1",
        "G1_HOMEWORK_LOGQ_EPOCHS": "20",
        "G1_HOMEWORK_LOGQ_RUN_REVISION": "1",
        "G1_HOMEWORK_LOGQ_EMBEDDING_LR": "0.001",
        "G1_HOMEWORK_LOGQ_DEEP_LR": "0.002",
    }
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    experiment = runpy.run_path(str(CONFIG))["experiment"]
    monkeypatch.setenv("G1_DATASET_SIZE", "50m")
    monkeypatch.setenv("G1_MAX_EPOCHS", "20")
    monkeypatch.setenv("G1_VARIANT", "homework_fixed_leave_one_out")
    control = runpy.run_path(str(CONFIG.with_name("variant.py")))["VARIANTS"][
        "homework_baseline_native500_r3"
    ]

    for field in fields(GenerationExperiment):
        if field.init and field.name != "run_name":
            assert getattr(experiment, field.name) == getattr(control, field.name)
    assert experiment.negative_sampling == "offline_logq"
    assert experiment.logq_correction == "baseline"
    assert experiment.num_in_batch_negatives == 512
    assert experiment.dataloader.batch_size == 1280
    assert experiment.lr_schedule.shape == "constant"


def test_logq_tuning_applies_nondefault_rates_without_swapping(monkeypatch) -> None:
    environment = {
        "G1_HOMEWORK_LOGQ_RUN": "rates_e0p0007_d0p003_ts2_r1",
        "G1_HOMEWORK_LOGQ_EPOCHS": "20",
        "G1_HOMEWORK_LOGQ_RUN_REVISION": "1",
        "G1_HOMEWORK_LOGQ_EMBEDDING_LR": "0.0007",
        "G1_HOMEWORK_LOGQ_DEEP_LR": "0.003",
    }
    for name, value in environment.items():
        monkeypatch.setenv(name, value)

    experiment = runpy.run_path(str(CONFIG))["experiment"]

    assert experiment.embedding_learning_rate == 0.0007
    assert experiment.deep_learning_rate == 0.003
    assert experiment.negative_sampling == "offline_logq"
    assert experiment.logq_correction == "baseline"


def test_logq_control_restores_variant_environment(monkeypatch) -> None:
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
    for name, value in {
        "G1_HOMEWORK_LOGQ_RUN": "initial_e0p001_d0p002_ts2_r1",
        "G1_HOMEWORK_LOGQ_EPOCHS": "20",
        "G1_HOMEWORK_LOGQ_RUN_REVISION": "1",
        "G1_HOMEWORK_LOGQ_EMBEDDING_LR": "0.001",
        "G1_HOMEWORK_LOGQ_DEEP_LR": "0.002",
    }.items():
        monkeypatch.setenv(name, value)

    runpy.run_path(str(CONFIG))

    assert {name: os.environ.get(name) for name in expected} == expected


def test_logq_launcher_submits_exact_cartesian_grid(tmp_path: Path) -> None:
    output = tmp_path / "queued.txt"
    queue = tmp_path / "queue.sh"
    queue.write_text(
        "name=WANDB_MODE\n"
        "enqueue() { printf '%s\\n' \"$*\" >> \"$G1_TEST_QUEUE_OUTPUT\"; }\n"
        "drain() { :; }\n"
    )
    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=ROOT,
        env=os.environ
        | {
            "G1_TRAINING_QUEUE_LIBRARY": str(queue),
            "G1_TEST_QUEUE_OUTPUT": str(output),
            "G1_HOMEWORK_LOGQ_RUN_TAG": "testgrid",
            "G1_HOMEWORK_LOGQ_EMBEDDING_LRS": "0.0005 0.001",
            "G1_HOMEWORK_LOGQ_DEEP_LRS": "0.002 0.004",
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    jobs = output.read_text().splitlines()
    assert len(jobs) == 4
    pairs = {
        tuple(
            item
            for item in job.split()
            if item.startswith(
                ("G1_HOMEWORK_LOGQ_EMBEDDING_LR=", "G1_HOMEWORK_LOGQ_DEEP_LR=")
            )
        )
        for job in jobs
    }
    assert pairs == {
        ("G1_HOMEWORK_LOGQ_EMBEDDING_LR=0.0005", "G1_HOMEWORK_LOGQ_DEEP_LR=0.002"),
        ("G1_HOMEWORK_LOGQ_EMBEDDING_LR=0.0005", "G1_HOMEWORK_LOGQ_DEEP_LR=0.004"),
        ("G1_HOMEWORK_LOGQ_EMBEDDING_LR=0.001", "G1_HOMEWORK_LOGQ_DEEP_LR=0.002"),
        ("G1_HOMEWORK_LOGQ_EMBEDDING_LR=0.001", "G1_HOMEWORK_LOGQ_DEEP_LR=0.004"),
    }
    assert all(job.split()[0].startswith("g1_homework_logq_testgrid_") for job in jobs)


def test_logq_launcher_rejects_explicit_empty_axis(tmp_path: Path) -> None:
    output = tmp_path / "queued.txt"
    queue = tmp_path / "queue.sh"
    queue.write_text(
        "enqueue() { printf '%s\\n' \"$*\" >> \"$G1_TEST_QUEUE_OUTPUT\"; }\n"
        "drain() { :; }\n"
    )
    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=ROOT,
        env=os.environ
        | {
            "G1_TRAINING_QUEUE_LIBRARY": str(queue),
            "G1_TEST_QUEUE_OUTPUT": str(output),
            "G1_HOMEWORK_LOGQ_RUN_TAG": "emptyaxis",
            "G1_HOMEWORK_LOGQ_EMBEDDING_LRS": "",
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "cannot be empty" in result.stderr
    assert not output.exists()


def test_logq_launcher_rejects_numerically_duplicate_rates(tmp_path: Path) -> None:
    output = tmp_path / "queued.txt"
    queue = tmp_path / "queue.sh"
    queue.write_text(
        "enqueue() { printf '%s\\n' \"$*\" >> \"$G1_TEST_QUEUE_OUTPUT\"; }\n"
        "drain() { :; }\n"
    )
    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=ROOT,
        env=os.environ
        | {
            "G1_TRAINING_QUEUE_LIBRARY": str(queue),
            "G1_TEST_QUEUE_OUTPUT": str(output),
            "G1_HOMEWORK_LOGQ_RUN_TAG": "duplicate",
            "G1_HOMEWORK_LOGQ_DEEP_LRS": "0.1 0.100000000000000005",
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "duplicate" in result.stderr
    assert not output.exists()


def test_config_verifier_accepts_homework_control_keys_and_rejects_unknown() -> None:
    assignments = runpy.run_path(str(VERIFY_ARTIFACT))["_config_assignments"]

    assert assignments(
        [
            "G1_HOMEWORK_RANDOM_RUN=selected_e0p002_d0p004_ts2_r1",
            "G1_HOMEWORK_RANDOM_DATASET_SIZE=500m",
            "G1_HOMEWORK_LOGQ_EMBEDDING_LR=0.001",
            "G1_HOMEWORK_LOGQ_DEEP_LR=0.002",
        ]
    ) == {
        "G1_HOMEWORK_RANDOM_RUN": "selected_e0p002_d0p004_ts2_r1",
        "G1_HOMEWORK_RANDOM_DATASET_SIZE": "500m",
        "G1_HOMEWORK_LOGQ_EMBEDDING_LR": "0.001",
        "G1_HOMEWORK_LOGQ_DEEP_LR": "0.002",
    }
    with pytest.raises(ValueError, match="unsupported config-verifier"):
        assignments(["G1_HOMEWORK_UNKNOWN=1"])


def _control_run(
    selector,
    embedding: float,
    deep: float,
    recall: float,
    resolved: bool = True,
    path: str | None = None,
):
    return selector["ControlRun"](
        embedding,
        deep,
        recall,
        resolved,
        Path(path or f"e{embedding}_d{deep}"),
    )


def _closed_runs(selector):
    runs = [
        _control_run(selector, embedding, deep, 0.1)
        for embedding in (0.0005, 0.001, 0.002)
        for deep in (0.001, 0.002, 0.004)
    ]
    runs.extend(
        [
            _control_run(selector, 0.004, 0.004, 0.11),
            _control_run(selector, 0.002, 0.008, 0.105),
            _control_run(selector, 0.004, 0.008, 0.12),
            _control_run(selector, 0.008, 0.008, 0.11),
            _control_run(selector, 0.004, 0.016, 0.10),
        ]
    )
    return runs


def test_selection_preflight_requires_exact_unique_closed_winner() -> None:
    selector = runpy.run_path(str(SELECTOR))
    validate = selector["validate_selection"]
    runs = _closed_runs(selector)

    winner = validate(runs, 0.004, 0.008)
    assert (winner.embedding_lr, winner.deep_lr) == (0.004, 0.008)

    with pytest.raises(ValueError, match="not the unique 50M recall winner"):
        validate(runs, 0.002, 0.004)


def test_selection_preflight_rejects_unresolved_and_boundary_winners() -> None:
    selector = runpy.run_path(str(SELECTOR))
    validate = selector["validate_selection"]
    runs = _closed_runs(selector)
    runs.append(_control_run(selector, 0.016, 0.016, 0.2, resolved=False))
    with pytest.raises(ValueError, match="cap-unresolved"):
        validate(runs, 0.004, 0.008)

    boundary_runs = [run for run in _closed_runs(selector) if run.embedding_lr != 0.008]
    with pytest.raises(ValueError, match="embedding-LR winner.*boundaries"):
        validate(boundary_runs, 0.004, 0.008)


def test_selection_preflight_rejects_ambiguous_resolved_artifacts() -> None:
    selector = runpy.run_path(str(SELECTOR))
    runs = [
        run
        for run in _closed_runs(selector)
        if (run.embedding_lr, run.deep_lr) != (0.004, 0.008)
    ]
    runs.extend(
        [
            _control_run(
                selector,
                0.004,
                0.008,
                0.12,
                path="g1_homework_logq_initial_e0p004_d0p008_ts2_r1_50m",
            ),
            _control_run(
                selector,
                0.004,
                0.008,
                0.12,
                path=(
                    "g1_homework_logq_retry_e0p004_d0p008_cap40_ts2_r2_50m"
                ),
            ),
        ]
    )

    with pytest.raises(ValueError, match="ambiguous resolved artifacts"):
        selector["validate_selection"](runs, 0.004, 0.008)


def test_selection_preflight_uses_latest_validated_cap_continuation() -> None:
    selector = runpy.run_path(str(SELECTOR))
    runs = [
        run
        for run in _closed_runs(selector)
        if (run.embedding_lr, run.deep_lr) != (0.004, 0.008)
    ]
    runs.extend(
        [
            _control_run(
                selector,
                0.004,
                0.008,
                0.11,
                path=(
                    "g1_homework_logq_initial_e0p004_d0p008_ts2_r1_50m"
                ),
            ),
            _control_run(
                selector,
                0.004,
                0.008,
                0.12,
                path=(
                    "g1_homework_logq_capcontinue_e0p004_d0p008_"
                    "cap40_ts2_r2_50m"
                ),
            ),
        ]
    )

    winner = selector["validate_selection"](runs, 0.004, 0.008)

    assert winner.path.name.endswith("_cap40_ts2_r2_50m")


@pytest.mark.parametrize(
    ("cap_epochs", "revision"),
    [(30, 2), (80, 3), (40, 3)],
)
def test_selection_preflight_rejects_gapped_or_misaligned_cap_chain(
    cap_epochs: int, revision: int
) -> None:
    selector = runpy.run_path(str(SELECTOR))
    runs = [
        run
        for run in _closed_runs(selector)
        if (run.embedding_lr, run.deep_lr) != (0.004, 0.008)
    ]
    runs.extend(
        [
            _control_run(
                selector,
                0.004,
                0.008,
                0.11,
                resolved=False,
                path="g1_homework_logq_initial_e0p004_d0p008_ts2_r1_50m",
            ),
            _control_run(
                selector,
                0.004,
                0.008,
                0.12,
                path=(
                    "g1_homework_logq_capcontinue_e0p004_d0p008_"
                    f"cap{cap_epochs}_ts2_r{revision}_50m"
                ),
            ),
        ]
    )

    with pytest.raises(ValueError, match="ambiguous resolved artifacts"):
        selector["validate_selection"](runs, 0.004, 0.008)


@pytest.mark.parametrize("include_base", [False, True])
def test_selection_preflight_rejects_cap_chain_without_base(
    include_base: bool,
) -> None:
    selector = runpy.run_path(str(SELECTOR))
    runs = [
        run
        for run in _closed_runs(selector)
        if (run.embedding_lr, run.deep_lr) != (0.004, 0.008)
    ]
    if include_base:
        runs.append(
            _control_run(
                selector,
                0.004,
                0.008,
                0.10,
                resolved=False,
                path=(
                    "g1_homework_logq_capcontinue_e0p004_d0p008_"
                    "cap40_ts2_r2_50m"
                ),
            )
        )
    runs.append(
        _control_run(
            selector,
            0.004,
            0.008,
            0.12,
            path=(
                "g1_homework_logq_capcontinue_e0p004_d0p008_"
                "cap80_ts2_r3_50m"
            ),
        )
    )

    with pytest.raises(ValueError, match="no unique base"):
        selector["validate_selection"](runs, 0.004, 0.008)


def test_selection_preflight_does_not_normalize_tag_substrings() -> None:
    selector = runpy.run_path(str(SELECTOR))
    runs = [
        run
        for run in _closed_runs(selector)
        if (run.embedding_lr, run.deep_lr) != (0.004, 0.008)
    ]
    runs.extend(
        [
            _control_run(
                selector,
                0.004,
                0.008,
                0.11,
                path=(
                    "g1_homework_logq_initial_extra_e0p004_d0p008_ts2_r1_50m"
                ),
            ),
            _control_run(
                selector,
                0.004,
                0.008,
                0.12,
                path=(
                    "g1_homework_logq_capcontinue_extra_e0p004_d0p008_"
                    "cap40_ts2_r2_50m"
                ),
            ),
        ]
    )

    with pytest.raises(ValueError, match="ambiguous resolved artifacts"):
        selector["validate_selection"](runs, 0.004, 0.008)


@pytest.mark.parametrize(
    "environment",
    [
        {
            "G1_HOMEWORK_LOGQ_RUN_TAG": "initial",
            "G1_HOMEWORK_LOGQ_EPOCHS": "40",
            "G1_HOMEWORK_LOGQ_RUN_REVISION": "2",
        },
        {"G1_HOMEWORK_LOGQ_RUN_TAG": "capcontinue"},
    ],
)
def test_logq_launcher_enforces_reserved_lineage_tags(
    tmp_path: Path, environment: dict[str, str]
) -> None:
    output = tmp_path / "queued.txt"
    queue = tmp_path / "queue.sh"
    queue.write_text(
        "enqueue() { printf '%s\\n' \"$*\" >> \"$G1_TEST_QUEUE_OUTPUT\"; }\n"
        "drain() { :; }\n"
    )
    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=ROOT,
        env=os.environ
        | {
            "G1_TRAINING_QUEUE_LIBRARY": str(queue),
            "G1_TEST_QUEUE_OUTPUT": str(output),
        }
        | environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "reserved" in result.stderr
    assert not output.exists()


def test_selection_loader_rejects_matching_incomplete_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selector = runpy.run_path(str(SELECTOR))
    incomplete = (
        tmp_path
        / "generated/logs/g1_homework_random_upperboundary_e0p004_d0p008_ts2_r1_50m"
    )
    incomplete.mkdir(parents=True)
    monkeypatch.setattr(
        selector["runpy"],
        "run_path",
        lambda _: {"classify_config": lambda *_: "complete"},
    )

    with pytest.raises(ValueError, match="incomplete 50M artifact"):
        selector["_load_runs"](tmp_path, "random")
