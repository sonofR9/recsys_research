import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
import torch

from neuralrec.utils import EXTRA_METRICS

from experiments.g2_esasrec.launchers.compiled import (
    build_local_experiment,
    build_official_experiment,
)
from experiments.g2_esasrec.launchers.cost import CostEvidenceCallback
from experiments.g2_esasrec.protocol.manifest import (
    approved_manifest,
    load_compiled_jobs,
)


ROOT = Path(__file__).resolve().parents[3]


def _compiled_manifest(tmp_path, job, parameters):
    path = tmp_path / "compiled.json"
    path.write_text(
        json.dumps(
            {
                "manifest_sha256": approved_manifest().sha256,
                "jobs": [
                    {
                        "id": job.id,
                        "run_name": job.run_name,
                        "parameters": parameters,
                    }
                ],
            }
        )
    )
    return path


def test_compiled_local_job_builds_the_exact_named_seeded_experiment(tmp_path):
    job = approved_manifest().jobs_for_stage("component_tuning")[1]
    path = _compiled_manifest(
        tmp_path,
        job,
        {
            "batch_size": 256,
            "embedding_learning_rate": 0.002,
            "deep_learning_rate": 0.003,
            "selected_control_job_id": "control_tuning:control_trial_01",
        },
    )

    experiment = build_local_experiment(load_compiled_jobs(path)[0])

    assert experiment.run_name == job.run_name
    assert experiment.seed == 42
    assert experiment.size == "50m"
    assert experiment.dataloader.batch_size == 256
    assert experiment.embedding_learning_rate == 0.002
    assert experiment.deep_learning_rate == 0.003


def test_queue_launcher_dry_run_validates_without_submitting(tmp_path):
    job = approved_manifest().jobs_for_stage("control_tuning")[0]
    path = _compiled_manifest(
        tmp_path,
        job,
        {
            "batch_size": 512,
            "embedding_learning_rate": 0.01,
            "deep_learning_rate": 0.02,
        },
    )
    launcher = ROOT / "experiments/g2_esasrec/launchers/queue_compiled.sh"

    result = subprocess.run(
        ["bash", str(launcher), str(path)],
        cwd=ROOT,
        env=os.environ | {"G2_QUEUE_DRY_RUN": "1"},
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.strip() == f"local\t{job.run_name}"
    assert "queued" not in result.stderr.lower()
    assert '"WANDB_MODE=offline"' in launcher.read_text()


def test_selected_benchmark_launcher_uses_offline_wandb():
    launcher = ROOT / "experiments/g2_esasrec/launchers/queue_selected_benchmark.sh"

    result = subprocess.run(
        ["bash", str(launcher), "selected-run", "encoded-job", "/tmp/result.json"],
        cwd=ROOT,
        env=os.environ | {"G2_QUEUE_DRY_RUN": "1"},
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.splitlines() == [
        "g2_selected_benchmark_selected-run_deterministic_reproduction_offline",
        "G2_COMPILED_JOB_B64=encoded-job",
        "G2_BENCHMARK_OUTPUT=/tmp/result.json",
        "WANDB_MODE=offline",
    ]
    subprocess.run(["bash", "-n", str(launcher)], check=True, env=os.environ)


@pytest.mark.parametrize("manifest", ["missing.json", "invalid.json"])
def test_queue_launcher_propagates_compiler_failure(tmp_path, manifest):
    path = tmp_path / manifest
    if manifest == "invalid.json":
        path.write_text("not json")
    launcher = ROOT / "experiments/g2_esasrec/launchers/queue_compiled.sh"

    result = subprocess.run(
        ["bash", str(launcher), str(path)],
        cwd=ROOT,
        env=os.environ | {"G2_QUEUE_DRY_RUN": "1"},
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert not result.stdout.strip()


def test_official_job_uses_the_external_rectools_interpreter(tmp_path):
    job = approved_manifest().jobs_for_stage("official")[0]
    path = _compiled_manifest(tmp_path, job, {"rectools_version": "0.19.0"})

    experiment = build_official_experiment(
        load_compiled_jobs(path)[0], Path("/bin/python3")
    )

    assert experiment.run_name == job.run_name
    assert experiment.seed == job.seed
    assert experiment.rectools_python == Path("/bin/python3")


def test_cost_callback_persists_exact_timing_resources_and_best_epoch(tmp_path):
    owner = SimpleNamespace(
        base_path=tmp_path,
        run_name="cost-evidence",
        device=torch.device("cpu"),
        training_targets_per_epoch=200,
        callbacks=SimpleNamespace(best_weights=SimpleNamespace(best_epoch=1)),
    )
    callback = CostEvidenceCallback(owner)
    callback.on_train_begin({})
    resources = {
        "params_total": 1000.0,
        "params_trainable": 900.0,
        "params_embedding": 400.0,
        "params_deep": 600.0,
    }
    for seconds in (2.0, 4.0):
        callback.on_epoch_end(
            {
                EXTRA_METRICS: {
                    "timing": {"train_epoch_time": seconds},
                    "resources": resources,
                }
            }
        )
    callback.on_train_end({})

    evidence = json.loads(
        (tmp_path / "logs/cost-evidence/cost_metrics.json").read_text()
    )
    assert evidence["training_seconds"] == 6.0
    assert evidence["wall_seconds"] >= 0.0
    assert evidence["median_train_epoch_seconds"] == 3.0
    assert evidence["targets_per_second"] == pytest.approx(200 / 3)
    assert evidence["best_epoch"] == 2
