from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import statistics

import pytest

from experiments.g1_aggregate_dataset_size.analysis.fixed26_calibration import (
    ArtifactError,
    AuthenticatedCalibrationRun,
    build_dispersion_evidence,
    collect_fixed26_calibration,
)
from experiments.g1_aggregate_dataset_size.configs.fixed26_calibration import (
    build_fixed26_experiment,
)
from experiments.g1_aggregate_dataset_size.launchers.fixed26_calibration import (
    MANIFEST_SHA_ENV,
    RECIPE_SHA_ENV,
    atomic_queue_specification,
    calibration_queue_rows,
    experiment_fingerprint,
    verify_experiment_contract,
    verify_worker_contract,
    write_job_contract,
)
from experiments.g1_aggregate_dataset_size.protocol.fixed26_calibration import (
    CALIBRATION_METRICS,
    load_fixed26_manifest,
)


def _metrics(offset: float = 0.0) -> dict[str, float]:
    return {
        metric: 0.1 + offset + index * 0.001
        for index, metric in enumerate(CALIBRATION_METRICS)
    }


def test_manifest_fixes_ten_distinct_native50m_jobs() -> None:
    manifest = load_fixed26_manifest()

    assert [job.seed for job in manifest.jobs] == list(range(42, 52))
    assert len({job.id for job in manifest.jobs}) == 10
    assert len({job.run_name for job in manifest.jobs}) == 10
    assert all("shared_calibration_fixed26" in job.run_name for job in manifest.jobs)
    assert all(job.run_name.endswith("_native50m") for job in manifest.jobs)
    assert manifest.batch_size == 512
    assert manifest.num_epochs == 26
    assert manifest.embedding_learning_rate == 0.003261002414691765
    assert manifest.deep_learning_rate == 0.025343654763668278
    assert len(manifest.sha256) == 64


def test_fixed26_config_changes_only_seed_identity_and_stopping_contract() -> None:
    manifest = load_fixed26_manifest()
    job = manifest.jobs[-1]

    experiment = build_fixed26_experiment(job, manifest)

    assert experiment.run_name == job.run_name
    assert experiment.seed == 51
    assert experiment.size == "50m"
    assert experiment.dataloader.batch_size == 512
    assert experiment.dataloader.gradient_accumulation_steps == 1
    assert experiment.embedding_learning_rate == 0.003261002414691765
    assert experiment.deep_learning_rate == 0.025343654763668278
    assert experiment.num_epochs == 26
    assert experiment.eval_every_n_epochs == 1
    assert experiment.early_stopping_patience is None
    assert experiment.restore_best_weights is True
    assert experiment.lr_schedule.shape == "constant"
    assert type(experiment).__name__ == "MuTransferGenerationExperiment"
    assert experiment.mup_base_dim == 16
    assert experiment.mup_delta_dim == 32


def test_queue_manifest_has_exactly_ten_manifest_bound_jobs() -> None:
    manifest = load_fixed26_manifest()

    rows = calibration_queue_rows(manifest)

    assert [row.run_name for row in rows] == [job.run_name for job in manifest.jobs]
    assert len(rows) == 10
    assert all(row.manifest_sha256 == manifest.sha256 for row in rows)
    assert all(
        row.environment
        == (
            f"G1_FIXED26_CALIBRATION_RUN={row.run_name}",
            f"{MANIFEST_SHA_ENV}={manifest.sha256}",
            f"{RECIPE_SHA_ENV}={manifest.recipe_sha256}",
        )
        for row in rows
    )
    specification = atomic_queue_specification(manifest, "offline")
    assert specification["version"] == 1
    assert len(specification["jobs"]) == 10
    assert [row["run"] for row in specification["jobs"]] == [
        job.run_name for job in manifest.jobs
    ]
    assert all(row["environment"][-1] == "WANDB_MODE=offline" for row in specification["jobs"])


def test_job_contract_is_immutable_and_manifest_authenticated(tmp_path: Path) -> None:
    manifest = load_fixed26_manifest()
    job = manifest.jobs[0]

    path = write_job_contract(tmp_path, job, manifest)

    document = json.loads(path.read_text())
    assert document["manifest_sha256"] == manifest.sha256
    assert document["recipe_sha256"] == manifest.recipe_sha256
    assert (
        document["experiment_fingerprint_sha256"]
        == manifest.experiment_fingerprint_sha256
    )
    assert document["job"] == job.to_dict()
    assert write_job_contract(tmp_path, job, manifest) == path
    path.write_text("{}\n")
    with pytest.raises(RuntimeError, match="immutable calibration job contract"):
        write_job_contract(tmp_path, job, manifest)


def test_dispersion_uses_unrounded_sample_statistics() -> None:
    manifest = load_fixed26_manifest()
    runs = tuple(
        AuthenticatedCalibrationRun(
            job=job,
            best_epoch=10 + index,
            epochs_trained=26,
            metrics=_metrics(index * 0.01),
            artifact_sha256={
                "fixed26_calibration_job.json": f"{index:064x}",
                "training_metadata.json": f"{index + 10:064x}",
                "final_metrics.json": f"{index + 20:064x}",
                "sweep.log": f"{index + 30:064x}",
            },
        )
        for index, job in enumerate(manifest.jobs)
    )

    evidence = build_dispersion_evidence(runs, manifest)

    values = [run.metrics["recall@100"] for run in runs]
    assert evidence["seeds"] == list(range(42, 52))
    assert evidence["metrics"]["recall@100"]["mean"] == statistics.fmean(values)
    assert evidence["metrics"]["recall@100"][
        "sample_standard_deviation"
    ] == statistics.stdev(values)
    assert evidence["metrics"]["recall@100"][
        "sample_standard_deviation_over_mean"
    ] == statistics.stdev(values) / statistics.fmean(values)
    assert evidence["trained_epochs"] == [26] * 10


def test_collector_rejects_missing_or_tampered_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = load_fixed26_manifest()
    monkeypatch.setattr(
        "experiments.g1_aggregate_dataset_size.analysis.fixed26_calibration."
        "verify_config_recipe",
        lambda *args, **kwargs: True,
    )

    with pytest.raises(ArtifactError, match="missing calibration artifact"):
        collect_fixed26_calibration(tmp_path, manifest)

    job = manifest.jobs[0]
    directory = tmp_path / job.run_name
    directory.mkdir()
    write_job_contract(tmp_path, job, manifest)
    contract = directory / "fixed26_calibration_job.json"
    contract.write_text(
        json.dumps(
            {
                "manifest_sha256": "0" * 64,
                "job": job.to_dict(),
            }
        )
        + "\n"
    )
    (directory / "training_metadata.json").write_text("{}\n")
    (directory / "final_metrics.json").write_text("{}\n")
    (directory / "sweep.log").write_text("")

    with pytest.raises(ArtifactError, match="job contract"):
        collect_fixed26_calibration(tmp_path, manifest)


def test_worker_rejects_submit_time_recipe_and_experiment_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = load_fixed26_manifest()
    recipe = tmp_path / "recipe.py"
    recipe.write_bytes(
        Path("experiments/g1_aggregate_dataset_size/launchers/run_fixed26_calibration.py")
        .resolve()
        .read_bytes()
    )
    monkeypatch.setenv(MANIFEST_SHA_ENV, manifest.sha256)
    monkeypatch.setenv(RECIPE_SHA_ENV, manifest.recipe_sha256)
    monkeypatch.setattr(
        "experiments.g1_aggregate_dataset_size.launchers.fixed26_calibration."
        "verify_source_control_artifacts",
        lambda *args: True,
    )

    verify_worker_contract(tmp_path, manifest, recipe)
    monkeypatch.setenv(MANIFEST_SHA_ENV, "0" * 64)
    with pytest.raises(RuntimeError, match="submit-time manifest"):
        verify_worker_contract(tmp_path, manifest, recipe)
    monkeypatch.setenv(MANIFEST_SHA_ENV, manifest.sha256)
    recipe.write_text("experiment = None\n")
    with pytest.raises(RuntimeError, match="queued recipe"):
        verify_worker_contract(tmp_path, manifest, recipe)

    experiment = build_fixed26_experiment(manifest.jobs[0], manifest)
    assert experiment_fingerprint(experiment) == manifest.experiment_fingerprint_sha256
    verify_experiment_contract(experiment, manifest)
    for changed in (
        replace(experiment, negative_sampling="random"),
        replace(experiment, min_seq_len=3),
        replace(experiment, stride=2.0),
        replace(experiment, validation_days=2),
        replace(experiment, prefix_cap=1),
    ):
        with pytest.raises(RuntimeError, match="experiment recipe"):
            verify_experiment_contract(changed, manifest)
