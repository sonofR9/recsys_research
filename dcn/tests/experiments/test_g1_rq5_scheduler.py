from __future__ import annotations

import copy
from dataclasses import replace
import json
import os
import runpy
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.g1_sasrec_item_ids_likes.analysis.rq5_scheduler_candidates import (
    initial_candidates,
)
from experiments.g1_sasrec_item_ids_likes.analysis.rq5_scheduler_corrections import (
    filesystem_inspector,
)
from experiments.g1_sasrec_item_ids_likes.launchers import verify_artifact


CONFIG = (
    Path(__file__).parents[3]
    / "experiments/g1_sasrec_item_ids_likes/configs/rq5_scheduler_variant.py"
)
ROOT = Path(__file__).parents[3]
EXPERIMENT = ROOT / "experiments/g1_sasrec_item_ids_likes"


def _rq5_final_metrics() -> dict[str, float]:
    metrics = {
        f"{name}@{k}": 0.1
        for k in (10, 50, 100)
        for name in ("ndcg", "recall", "capped_recall", "mrr", "coverage")
    }
    metrics["num_users"] = 100.0
    return metrics


def test_initial_rq5_stage_has_67_unique_native_500m_runs() -> None:
    candidates = initial_candidates()

    assert len(candidates) == 67
    assert len({candidate.run_name for candidate in candidates}) == 67
    assert {candidate.dataset_size for candidate in candidates} == {"500m"}
    assert {candidate.seed for candidate in candidates} == {42}
    assert {candidate.embedding_lr for candidate in candidates} == {0.064}


def test_fixed_and_tuned_cosine_share_the_identical_central_run() -> None:
    shared = [
        candidate
        for candidate in initial_candidates()
        if set(candidate.treatments)
        == {"cosine_warmup5_cycles1", "cosine_warmup_tuned"}
    ]

    assert len(shared) == 2
    assert {candidate.scope for candidate in shared} == {"both", "deep_only"}
    assert {candidate.deep_lr for candidate in shared} == {0.006}
    assert {candidate.warmup_fraction for candidate in shared} == {0.05}


def test_joint_candidate_draws_are_reproducible_and_shared_between_scopes() -> None:
    candidates = initial_candidates()
    for treatment in ("inverse_sqrt", "cosine_warmup_tuned"):
        by_scope = {
            scope: [
                (candidate.deep_lr, candidate.joint_fraction)
                for candidate in candidates
                if treatment in candidate.treatments and candidate.scope == scope
            ]
            for scope in ("both", "deep_only")
        }
        assert by_scope["both"] == by_scope["deep_only"]
        assert len(by_scope["both"]) == 3


@pytest.mark.parametrize("scope", ["both", "deep_only"])
def test_rq5_config_has_fixed_embedding_and_adaptive_horizon(
    monkeypatch: pytest.MonkeyPatch, scope: str
) -> None:
    candidate = next(
        candidate
        for candidate in initial_candidates()
        if candidate.treatments == ("linear",)
        and candidate.scope == scope
        and candidate.deep_lr == 0.006
    )
    for name, value in candidate.environment().items():
        monkeypatch.setenv(name, value)
    experiment = runpy.run_path(str(CONFIG))["experiment"]

    assert experiment.size == "500m"
    assert experiment.seed == 42
    assert experiment.dataloader.effective_batch_size == 1280
    assert experiment.embedding_learning_rate == 0.064
    assert experiment.deep_learning_rate == 0.006
    assert experiment.lr_schedule.shape == "linear"
    assert experiment.lr_schedule.optimizer_group_scope == scope
    assert experiment.lr_schedule_horizon_epochs == 17
    assert experiment.num_epochs == 17
    assert experiment.adaptive_schedule_early_stopping
    assert experiment.early_stopping_patience == 3
    assert experiment.run_name == candidate.run_name


def test_linear_candidate_has_dedicated_schedule_provenance() -> None:
    linear = [
        candidate
        for candidate in initial_candidates()
        if candidate.treatments == ("linear",)
    ]

    assert len(linear) == 6
    assert all("linear" in candidate.run_name for candidate in linear)
    assert all(candidate.shape == "linear" for candidate in linear)


def test_expected_metadata_contains_adaptive_schedule_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = next(
        candidate
        for candidate in initial_candidates()
        if candidate.treatments == ("linear",)
    )
    monkeypatch.setenv("G1_RQ5_RUN", candidate.run_name)
    experiment = runpy.run_path(str(CONFIG))["experiment"]

    _, invariants = verify_artifact._expected_metadata(experiment)

    assert invariants["adaptive_schedule_early_stopping"] is True


def test_legacy_static_schedule_defaults_are_normalized() -> None:
    metadata = {
        "transfer_invariants": {
            "lr_schedule": {"shape": "linear"},
        }
    }

    normalized = verify_artifact._with_legacy_accumulation_defaults(metadata)

    assert (
        normalized["transfer_invariants"]["adaptive_schedule_early_stopping"] is False
    )
    assert (
        normalized["transfer_invariants"]["lr_schedule"]["optimizer_group_scope"]
        == "both"
    )


def test_legacy_config_metrics_keep_the_generic_contract() -> None:
    experiment = SimpleNamespace(
        adaptive_schedule_early_stopping=False,
        eval_ks=(10, 50, 100),
    )

    assert verify_artifact._valid_config_metrics({"loss": 0.1}, experiment)


def _adaptive_metadata(
    *,
    shape: str = "linear",
    scope: str = "both",
    horizon: int = 17,
    stopped: int = 14,
    early_stopped: bool = True,
    stored_status: str = "calibrated",
    timescale_fraction: float | None = None,
) -> dict:
    steps_per_epoch = 5
    total_steps = horizon * steps_per_epoch
    schedule = {
        "shape": shape,
        "warmup_fraction": 0.0,
        "min_lr_fraction": 0.0,
        "cycles": 1,
        "timescale_steps": None,
        "timescale_fraction": timescale_fraction,
        "power_exponent": -0.51,
        "power_transition_tokens": None,
        "optimizer_group_scope": scope,
    }
    if shape == "inverse_sqrt":
        timescale_steps = max(1, int(total_steps * float(timescale_fraction)))
        factors = [
            (timescale_steps / (timescale_steps + epoch * steps_per_epoch - 1)) ** 0.5
            for epoch in range(1, stopped + 1)
        ]
    else:
        timescale_steps = None
        decay_steps = total_steps - 1
        factors = [
            1 - (epoch * steps_per_epoch - 1) / decay_steps
            for epoch in range(1, stopped + 1)
        ]
    return {
        "num_epochs": horizon if shape != "inverse_sqrt" else 80,
        "max_epochs": horizon if shape != "inverse_sqrt" else 80,
        "epochs_trained": stopped,
        "stopped_epoch": stopped,
        "best_epoch": max(1, stopped - 3),
        "early_stopped": early_stopped,
        "best_epoch_at_cap": False,
        "selection_resolved": stored_status == "calibrated",
        "targets_per_epoch": 11,
        "tokens_per_epoch": 13,
        "training_horizon": 11 * stopped,
        "token_horizon": 13 * stopped,
        "tokens_seen": 13 * stopped,
        "optimizer_steps": steps_per_epoch * stopped,
        "optimizer_steps_per_epoch": steps_per_epoch,
        "lr_schedule_horizon_epochs": horizon,
        "lr_schedule_horizon_steps": total_steps,
        "lr_schedule_timescale_steps": timescale_steps,
        "embedding_learning_rate": 0.064,
        "deep_learning_rate": 0.006,
        "lr_group_traces": {
            "embedding": (
                [0.064] * stopped
                if scope == "deep_only"
                else [factor * 0.064 for factor in factors]
            ),
            "deep": [factor * 0.006 for factor in factors],
        },
        "horizon_calibration_status": stored_status,
        "next_lr_schedule_horizon_epochs": None,
        "validation_loss": 0.5,
        "transfer_invariants": {
            "adaptive_schedule_early_stopping": True,
            "lr_schedule": schedule,
        },
    }


@pytest.mark.parametrize("scope", ["both", "deep_only"])
def test_adaptive_artifact_requires_exact_steps_and_group_traces(scope: str) -> None:
    metadata = _adaptive_metadata(scope=scope)

    assert verify_artifact._valid_dynamic_metadata(metadata)
    metadata["optimizer_steps"] = 14
    assert not verify_artifact._valid_dynamic_metadata(metadata)
    metadata["optimizer_steps"] = (
        metadata["optimizer_steps_per_epoch"] * metadata["epochs_trained"]
    )
    metadata["lr_group_traces"]["deep"][0] *= 0.9
    assert not verify_artifact._valid_dynamic_metadata(metadata)


def test_inverse_sqrt_fractional_timescale_keeps_reference_horizon() -> None:
    metadata = _adaptive_metadata(
        shape="inverse_sqrt",
        horizon=23,
        stopped=21,
        timescale_fraction=0.05,
    )

    assert verify_artifact._valid_dynamic_metadata(metadata)
    metadata["lr_schedule_horizon_steps"] = None
    assert not verify_artifact._valid_dynamic_metadata(metadata)


@pytest.mark.parametrize(
    ("stopped", "early_stopped", "stored_status"),
    [
        (10, True, "calibrated"),
        (17, True, "shorten_horizon"),
        (20, False, "calibrated"),
    ],
)
def test_adaptive_verifier_recomputes_calibration_status(
    stopped: int, early_stopped: bool, stored_status: str
) -> None:
    metadata = _adaptive_metadata(
        horizon=20,
        stopped=stopped,
        early_stopped=early_stopped,
        stored_status=stored_status,
    )

    assert not verify_artifact._valid_dynamic_metadata(metadata)


def test_proportional_but_wrong_schedule_trace_is_rejected() -> None:
    metadata = _adaptive_metadata()
    metadata["lr_group_traces"]["embedding"] = [
        value * 0.9 for value in metadata["lr_group_traces"]["embedding"]
    ]
    metadata["lr_group_traces"]["deep"] = [
        value * 0.9 for value in metadata["lr_group_traces"]["deep"]
    ]

    assert not verify_artifact._valid_dynamic_metadata(metadata)


def test_unresolved_adaptive_recipe_requires_exact_schedule_execution() -> None:
    metadata = _adaptive_metadata(
        horizon=17,
        stopped=8,
        stored_status="shorten_horizon",
    )
    metadata["next_lr_schedule_horizon_epochs"] = 8

    assert verify_artifact._valid_adaptive_schedule_execution(metadata)
    for field, value in (
        ("optimizer_steps", 39),
        ("lr_schedule_horizon_steps", 84),
        ("horizon_calibration_status", "calibrated"),
        ("lr_group_traces", {}),
    ):
        malformed = copy.deepcopy(metadata)
        malformed[field] = value
        assert not verify_artifact._valid_adaptive_schedule_execution(malformed)


def test_inverse_recipe_requires_exact_fractional_timescale_execution() -> None:
    metadata = _adaptive_metadata(
        shape="inverse_sqrt",
        horizon=23,
        stopped=21,
        timescale_fraction=0.05,
    )

    assert verify_artifact._valid_adaptive_schedule_execution(metadata)
    metadata["lr_schedule_timescale_steps"] += 1
    assert not verify_artifact._valid_adaptive_schedule_execution(metadata)


def test_completed_unresolved_artifact_is_recipe_compatible_but_not_selectable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = next(
        candidate
        for candidate in initial_candidates()
        if candidate.treatments == ("linear",)
        and candidate.scope == "both"
        and candidate.deep_lr == 0.006
    )
    monkeypatch.setenv("G1_RQ5_RUN", candidate.run_name)
    experiment = runpy.run_path(str(CONFIG))["experiment"]
    top_level, invariants = verify_artifact._expected_metadata(experiment)
    stopped = 8
    metadata = top_level | {
        "training_semantics_revision": 2,
        "max_epochs": 17,
        "epochs_trained": stopped,
        "stopped_epoch": stopped,
        "best_epoch": 5,
        "early_stopped": True,
        "best_epoch_at_cap": False,
        "selection_resolved": False,
        "targets_per_epoch": 11,
        "tokens_per_epoch": 13,
        "training_horizon": 11 * stopped,
        "token_horizon": 13 * stopped,
        "tokens_seen": 13 * stopped,
        "optimizer_steps": 5 * stopped,
        "optimizer_steps_per_epoch": 5,
        "lr_schedule_horizon_steps": 85,
        "lr_schedule_timescale_steps": None,
        "lr_group_traces": {
            "embedding": [
                0.064 * (1 - (epoch * 5 - 1) / 84) for epoch in range(1, stopped + 1)
            ],
            "deep": [
                0.006 * (1 - (epoch * 5 - 1) / 84) for epoch in range(1, stopped + 1)
            ],
        },
        "horizon_calibration_status": "shorten_horizon",
        "next_lr_schedule_horizon_epochs": stopped,
        "validation_loss": 0.5,
        "transfer_invariants": invariants,
    }
    directory = tmp_path / experiment.run_name
    directory.mkdir()
    (directory / "training_metadata.json").write_text(json.dumps(metadata))
    metrics_path = directory / "final_metrics.json"
    metrics_path.write_text(json.dumps(_rq5_final_metrics()))
    assignments = [f"G1_RQ5_RUN={candidate.run_name}"]

    assert verify_artifact.verify_config_recipe(directory, CONFIG, assignments)
    assert not verify_artifact.verify_config(directory, CONFIG, assignments)
    metrics_path.write_text(json.dumps({"loss": 0.1}))
    assert not verify_artifact.verify_config_recipe(directory, CONFIG, assignments)
    assert not verify_artifact.verify_config(directory, CONFIG, assignments)
    metrics_path.write_text(json.dumps(_rq5_final_metrics()))
    valid_metadata = copy.deepcopy(metadata)
    assert (
        verify_artifact.classify_config_recipe(directory, CONFIG, assignments)
        == verify_artifact.COMPLETE
    )
    metadata["lr_group_traces"] = {}
    (directory / "training_metadata.json").write_text(json.dumps(metadata))
    metrics_path.write_text(json.dumps(_rq5_final_metrics()))
    assert not verify_artifact.verify_config_recipe(directory, CONFIG, assignments)
    assert (
        verify_artifact.classify_config_recipe(directory, CONFIG, assignments)
        == verify_artifact.RESUMABLE
    )
    assert filesystem_inspector(tmp_path)(candidate).kind == "recoverable"
    metadata = valid_metadata
    (directory / "training_metadata.json").write_text(json.dumps(metadata))
    metrics_path.write_text(json.dumps(_rq5_final_metrics()))
    metrics_path.unlink()
    assert (
        verify_artifact.classify_config_recipe(directory, CONFIG, assignments)
        == verify_artifact.RESUMABLE
    )
    (directory / "training_metadata.json").write_text("{")
    assert (
        verify_artifact.classify_config_recipe(directory, CONFIG, assignments)
        == verify_artifact.RESUMABLE
    )


def _queue_stub(tmp_path: Path) -> Path:
    queue = tmp_path / "queue.sh"
    queue.write_text(
        "enqueue() { printf 'ENQUEUE %s\\n' \"$*\" >&2; return 0; }\n"
        "drain() { return 0; }\n"
    )
    return queue


def test_initial_launcher_submits_exactly_67_unique_runs(tmp_path: Path) -> None:
    launchers = tmp_path / "experiments/g1_sasrec_item_ids_likes/launchers"
    shutil.copytree(EXPERIMENT / "launchers", launchers)
    (tmp_path / "generated/logs").mkdir(parents=True)
    (launchers / "verify_artifact.py").write_text(
        "import sys\nfor line in sys.stdin:\n    print(0, flush=True)\n"
    )
    result = subprocess.run(
        ["bash", str(launchers / "schedule/rq5_500m.sh")],
        capture_output=True,
        text=True,
        env=os.environ
        | {
            "G1_TRAINING_QUEUE_LIBRARY": str(_queue_stub(tmp_path)),
            "PYTHONPATH": os.fspath(ROOT),
        },
    )

    lines = [line for line in result.stderr.splitlines() if line.startswith("ENQUEUE ")]
    names = [line.split()[1] for line in lines]
    assert result.returncode == 0
    assert len(names) == 67
    assert len(set(names)) == 67
    assert all("G1_RQ5_RUN=" in line for line in lines)
    assert "enqueued=67, skipped=0" in result.stdout


def test_launcher_rejects_unapproved_stage(tmp_path: Path) -> None:
    result = subprocess.run(
        ["bash", os.fspath(EXPERIMENT / "launchers/schedule/rq5_500m.sh")],
        capture_output=True,
        text=True,
        env=os.environ | {"G1_RQ5_STAGE": "boundary"},
    )

    assert result.returncode == 2
    assert "must be initial, probes, or corrections" in result.stderr


def _planner_python(
    tmp_path: Path,
    *,
    output: str = "",
    status: int = 0,
    selection_output: str = "",
    selection_status: int = 0,
) -> Path:
    executable = tmp_path / "bin/python"
    executable.parent.mkdir()
    real_python = shutil.which("python")
    assert real_python is not None
    executable.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "$1" == -m && "$2" == '
        "experiments.g1_sasrec_item_ids_likes.analysis.rq5_scheduler_corrections ]]; then\n"
        '    [[ -z "${PLANNER_ARGS_FILE:-}" ]] || '
        'printf \'%s\\n\' "$*" > "$PLANNER_ARGS_FILE"\n'
        f"    printf '%b' {output!r}\n"
        f"    exit {status}\n"
        "fi\n"
        'if [[ "$1" == -m && "$2" == '
        "experiments.g1_sasrec_item_ids_likes.analysis.rq5_scheduler_selection ]]; then\n"
        '    [[ -z "${PLANNER_ARGS_FILE:-}" ]] || '
        'printf \'%s\\n\' "$*" > "$PLANNER_ARGS_FILE"\n'
        f"    printf '%b' {selection_output!r}\n"
        f"    exit {selection_status}\n"
        "fi\n"
        f'exec {real_python!r} "$@"\n'
    )
    executable.chmod(0o755)
    return executable.parent


def test_corrections_launcher_submits_only_planner_output(tmp_path: Path) -> None:
    launchers = tmp_path / "experiments/g1_sasrec_item_ids_likes/launchers"
    shutil.copytree(EXPERIMENT / "launchers", launchers)
    (tmp_path / "generated/logs").mkdir(parents=True)
    source = next(
        candidate
        for candidate in initial_candidates()
        if candidate.treatments == ("cosine",)
        and candidate.scope == "both"
        and candidate.deep_lr == 0.003
    )
    correction = source.__class__(
        **{
            **source.__dict__,
            "horizon_epochs": 22,
            "cap_epochs": 22,
            "attempt": 5,
        }
    )
    fake_bin = _planner_python(tmp_path, output=f"{correction.run_name}\\n")
    manifest = tmp_path / "rq5_scheduler_candidate_manifest.json"
    manifest.touch()
    planner_args = tmp_path / "planner_args.txt"
    result = subprocess.run(
        ["bash", str(launchers / "schedule/rq5_500m.sh")],
        capture_output=True,
        text=True,
        env=os.environ
        | {
            "G1_RQ5_STAGE": "corrections",
            "G1_TRAINING_QUEUE_LIBRARY": str(_queue_stub(tmp_path)),
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "G1_RQ5_MANIFEST": str(manifest),
            "PLANNER_ARGS_FILE": str(planner_args),
            "PYTHONPATH": os.fspath(ROOT),
        },
    )

    lines = [line for line in result.stderr.splitlines() if line.startswith("ENQUEUE ")]
    assert result.returncode == 0
    assert len(lines) == 1
    assert correction.run_name in lines[0]
    assert f"G1_RQ5_RUN={correction.run_name}" in lines[0]
    assert "corrections: enqueued=1, skipped=0" in result.stdout
    assert f"--manifest {manifest}" in planner_args.read_text()


def test_probe_launcher_submits_only_atomically_manifested_selection_output(
    tmp_path: Path,
) -> None:
    launchers = tmp_path / "experiments/g1_sasrec_item_ids_likes/launchers"
    shutil.copytree(EXPERIMENT / "launchers", launchers)
    (tmp_path / "generated/logs").mkdir(parents=True)
    probe = replace(
        next(
            candidate
            for candidate in initial_candidates()
            if candidate.treatments == ("linear",)
        ),
        deep_lr=0.0015,
        probe="b1lrlo1",
    )
    fake_bin = _planner_python(tmp_path, selection_output=f"{probe.run_name}\\n")
    manifest = tmp_path / "rq5_scheduler_candidate_manifest.json"
    planner_args = tmp_path / "planner_args.txt"
    result = subprocess.run(
        ["bash", str(launchers / "schedule/rq5_500m.sh")],
        capture_output=True,
        text=True,
        env=os.environ
        | {
            "G1_RQ5_STAGE": "probes",
            "G1_RQ5_MANIFEST": str(manifest),
            "G1_TRAINING_QUEUE_LIBRARY": str(_queue_stub(tmp_path)),
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "PLANNER_ARGS_FILE": str(planner_args),
            "PYTHONPATH": os.fspath(ROOT),
        },
    )

    lines = [line for line in result.stderr.splitlines() if line.startswith("ENQUEUE ")]
    assert result.returncode == 0
    assert len(lines) == 1
    assert probe.run_name in lines[0]
    assert "probes: enqueued=1, skipped=0" in result.stdout
    assert f"--manifest {manifest}" in planner_args.read_text()


def test_corrections_launcher_refuses_planner_failure(tmp_path: Path) -> None:
    launchers = tmp_path / "experiments/g1_sasrec_item_ids_likes/launchers"
    shutil.copytree(EXPERIMENT / "launchers", launchers)
    (tmp_path / "generated/logs").mkdir(parents=True)
    fake_bin = _planner_python(tmp_path, status=2)
    result = subprocess.run(
        ["bash", str(launchers / "schedule/rq5_500m.sh")],
        capture_output=True,
        text=True,
        env=os.environ
        | {
            "G1_RQ5_STAGE": "corrections",
            "G1_TRAINING_QUEUE_LIBRARY": str(_queue_stub(tmp_path)),
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "PYTHONPATH": os.fspath(ROOT),
        },
    )

    assert result.returncode == 2
    assert "ENQUEUE " not in result.stderr


def test_launcher_retains_unresolved_recipe_and_archives_recoverable_states(
    tmp_path: Path,
) -> None:
    launchers = tmp_path / "experiments/g1_sasrec_item_ids_likes/launchers"
    shutil.copytree(EXPERIMENT / "launchers", launchers)
    logs = tmp_path / "generated/logs"
    logs.mkdir(parents=True)
    retained, incomplete, malformed = initial_candidates()[:3]
    for candidate, marker in (
        (retained, "recipe_complete"),
        (incomplete, "resumable"),
        (malformed, "incompatible"),
    ):
        directory = logs / candidate.run_name
        directory.mkdir()
        (directory / marker).touch()
    (launchers / "verify_artifact.py").write_text(
        "import pathlib, sys\n"
        "for line in sys.stdin:\n"
        "    path, mode, *_ = line.rstrip('\\n').split('\\t')\n"
        "    directory = pathlib.Path(path)\n"
        "    if mode != 'classify-config-recipe':\n"
        "        print('2\\twrong verifier mode', flush=True)\n"
        "    elif (directory / 'recipe_complete').exists():\n"
        "        print('complete', flush=True)\n"
        "    elif (directory / 'resumable').exists():\n"
        "        print('resumable', flush=True)\n"
        "    else:\n"
        "        print('incompatible', flush=True)\n"
    )
    result = subprocess.run(
        ["bash", str(launchers / "schedule/rq5_500m.sh")],
        capture_output=True,
        text=True,
        env=os.environ
        | {
            "G1_TRAINING_QUEUE_LIBRARY": str(_queue_stub(tmp_path)),
            "PYTHONPATH": os.fspath(ROOT),
        },
    )

    lines = [line for line in result.stderr.splitlines() if line.startswith("ENQUEUE ")]
    assert result.returncode == 0
    assert len(lines) == 66
    assert (logs / retained.run_name / "recipe_complete").exists()
    assert not any(retained.run_name in line for line in lines)
    assert (logs / "old" / f"{incomplete.run_name}.incomplete-001").is_dir()
    assert (logs / "old" / f"{malformed.run_name}.incompatible-001").is_dir()
    assert any(incomplete.run_name in line for line in lines)
    assert any(malformed.run_name in line for line in lines)
    assert "enqueued=66, skipped=1" in result.stdout
