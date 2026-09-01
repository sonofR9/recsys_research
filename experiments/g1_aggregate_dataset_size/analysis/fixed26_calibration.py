from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import statistics
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from experiments.g1_aggregate_dataset_size.launchers.fixed26_calibration import (
    CONTRACT_NAME,
    artifact_sha256,
    verify_job_contract,
)
from experiments.g1_aggregate_dataset_size.protocol.fixed26_calibration import (
    CALIBRATION_METRICS,
    Fixed26CalibrationJob,
    Fixed26CalibrationManifest,
)
from experiments.g1_sasrec_item_ids_likes.launchers.verify_artifact import (
    verify_config_recipe,
)


CONFIG_PATH = Path(__file__).parents[1] / "launchers" / "run_fixed26_calibration.py"
EXPECTED_USER_COUNT = 3414
_ARTIFACT_NAMES = (
    CONTRACT_NAME,
    "training_metadata.json",
    "final_metrics.json",
    "sweep.log",
)


class ArtifactError(ValueError):
    pass


@dataclass(frozen=True)
class AuthenticatedCalibrationRun:
    job: Fixed26CalibrationJob
    best_epoch: int
    epochs_trained: int
    metrics: Mapping[str, float]
    artifact_sha256: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))
        object.__setattr__(
            self, "artifact_sha256", MappingProxyType(dict(self.artifact_sha256))
        )


def collect_fixed26_calibration(
    logs: Path,
    manifest: Fixed26CalibrationManifest,
) -> dict[str, Any]:
    runs = tuple(
        load_authenticated_run(logs / job.run_name, job, manifest)
        for job in manifest.jobs
    )
    return build_dispersion_evidence(runs, manifest)


def load_authenticated_run(
    directory: Path,
    job: Fixed26CalibrationJob,
    manifest: Fixed26CalibrationManifest,
) -> AuthenticatedCalibrationRun:
    missing = [name for name in _ARTIFACT_NAMES if not (directory / name).is_file()]
    if missing:
        raise ArtifactError(
            f"{job.run_name}: missing calibration artifact {', '.join(missing)}"
        )
    if directory.name != job.run_name or job not in manifest.jobs:
        raise ArtifactError("calibration artifact has an unapproved job identity")
    if not verify_job_contract(directory, job, manifest):
        raise ArtifactError(f"{job.run_name}: calibration job contract changed")
    try:
        config_matches = verify_config_recipe(
            directory,
            CONFIG_PATH,
            [
                f"G1_FIXED26_CALIBRATION_RUN={job.run_name}",
                f"G1_FIXED26_CALIBRATION_MANIFEST_SHA256={manifest.sha256}",
                f"G1_FIXED26_CALIBRATION_RECIPE_SHA256={manifest.recipe_sha256}",
            ],
        )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError):
        config_matches = False
    if not config_matches:
        raise ArtifactError(f"{job.run_name}: configuration authentication failed")
    metadata = _load_mapping(directory / "training_metadata.json")
    metrics = _load_mapping(directory / "final_metrics.json")
    _validate_execution(metadata, job, manifest)
    numeric_metrics = _validate_metrics(metrics, job)
    return AuthenticatedCalibrationRun(
        job=job,
        best_epoch=int(metadata["best_epoch"]),
        epochs_trained=int(metadata["epochs_trained"]),
        metrics=numeric_metrics,
        artifact_sha256={
            name: artifact_sha256(directory / name) for name in _ARTIFACT_NAMES
        },
    )


def build_dispersion_evidence(
    runs: Sequence[AuthenticatedCalibrationRun],
    manifest: Fixed26CalibrationManifest,
) -> dict[str, Any]:
    if tuple(run.job for run in runs) != manifest.jobs:
        raise ArtifactError("dispersion requires the ordered ten manifest jobs")
    if any(run.epochs_trained != manifest.num_epochs for run in runs):
        raise ArtifactError("dispersion includes a run that did not complete 26 epochs")
    if any(
        not 1 <= run.best_epoch <= manifest.num_epochs
        or set(run.metrics) != set(CALIBRATION_METRICS)
        or set(run.artifact_sha256) != set(_ARTIFACT_NAMES)
        for run in runs
    ):
        raise ArtifactError("dispersion includes malformed authenticated evidence")
    metric_evidence = {}
    for metric in CALIBRATION_METRICS:
        values = [run.metrics[metric] for run in runs]
        mean = statistics.fmean(values)
        sample_standard_deviation = statistics.stdev(values)
        if mean <= 0:
            raise ArtifactError(f"{metric} has a non-positive calibration mean")
        metric_evidence[metric] = {
            "mean": mean,
            "sample_standard_deviation": sample_standard_deviation,
            "sample_standard_deviation_over_mean": (
                sample_standard_deviation / mean
            ),
        }
    return {
        "version": 1,
        "status": "candidate_pending_user_validation",
        "dataset_size": "native-50m",
        "manifest_sha256": manifest.sha256,
        "source_control": {
            "run_name": manifest.source_run_name,
            "artifact_sha256": dict(manifest.source_artifact_sha256),
        },
        "seeds": [run.job.seed for run in runs],
        "best_epochs": [run.best_epoch for run in runs],
        "trained_epochs": [run.epochs_trained for run in runs],
        "metrics": metric_evidence,
        "runs": [
            {
                "job": run.job.to_dict(),
                "best_epoch": run.best_epoch,
                "epochs_trained": run.epochs_trained,
                "metrics": dict(run.metrics),
                "artifact_sha256": dict(run.artifact_sha256),
            }
            for run in runs
        ],
    }


def _validate_execution(
    metadata: Mapping[str, Any],
    job: Fixed26CalibrationJob,
    manifest: Fixed26CalibrationManifest,
) -> None:
    invariants = metadata.get("transfer_invariants")
    expected = {
        "dataset_size": manifest.dataset_size,
        "seed": job.seed,
        "batch_size": manifest.batch_size,
        "physical_batch_size": manifest.batch_size,
        "effective_batch_size": manifest.batch_size,
        "gradient_accumulation_steps": 1,
        "embedding_learning_rate": manifest.embedding_learning_rate,
        "deep_learning_rate": manifest.deep_learning_rate,
        "num_epochs": manifest.num_epochs,
        "max_epochs": manifest.num_epochs,
        "epochs_trained": manifest.num_epochs,
        "stopped_epoch": manifest.num_epochs,
        "early_stopped": False,
    }
    errors = [
        f"{name}={metadata.get(name)!r}"
        for name, value in expected.items()
        if type(metadata.get(name)) is not type(value) or metadata.get(name) != value
    ]
    best_epoch = metadata.get("best_epoch")
    if (
        not isinstance(best_epoch, int)
        or isinstance(best_epoch, bool)
        or not 1 <= best_epoch <= manifest.num_epochs
    ):
        errors.append(f"best_epoch={best_epoch!r}")
    if not isinstance(invariants, dict):
        errors.append("transfer_invariants are absent")
    else:
        invariant_expected = {
            "experiment_class": "MuTransferGenerationExperiment",
            "mup_base_dim": 16,
            "mup_delta_dim": 32,
            "batch_size": 512,
            "physical_batch_size": 512,
            "effective_batch_size": 512,
            "early_stopping_patience": None,
            "restore_best_weights": True,
            "eval_every_n_epochs": 1,
            "user_sample": None,
        }
        errors.extend(
            f"transfer_invariants.{name}={invariants.get(name)!r}"
            for name, value in invariant_expected.items()
            if type(invariants.get(name)) is not type(value)
            or invariants.get(name) != value
        )
        schedule = invariants.get("lr_schedule")
        if not isinstance(schedule, dict) or schedule.get("shape") != "constant":
            errors.append("transfer_invariants.lr_schedule is not constant")
    if errors:
        raise ArtifactError(
            f"{job.run_name}: invalid fixed-26 execution: " + "; ".join(errors)
        )


def _validate_metrics(
    document: Mapping[str, Any], job: Fixed26CalibrationJob
) -> dict[str, float]:
    if set(document) != {*CALIBRATION_METRICS, "num_users"}:
        raise ArtifactError(f"{job.run_name}: final metric fields changed")
    if document.get("num_users") != EXPECTED_USER_COUNT:
        raise ArtifactError(f"{job.run_name}: expected {EXPECTED_USER_COUNT} users")
    metrics = {}
    for metric in CALIBRATION_METRICS:
        value = document[metric]
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ArtifactError(f"{job.run_name}: invalid {metric}")
        metrics[metric] = float(value)
    return metrics


def _load_mapping(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactError(f"{path}: invalid JSON artifact") from error
    if not isinstance(value, dict):
        raise ArtifactError(f"{path}: expected a JSON object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", type=Path, default=Path("generated/logs"))
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    from experiments.g1_aggregate_dataset_size.protocol.fixed26_calibration import (
        load_fixed26_manifest,
    )

    evidence = collect_fixed26_calibration(
        arguments.logs.resolve(), load_fixed26_manifest()
    )
    rendered = json.dumps(evidence, indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        print(rendered, end="")
        return
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("x") as stream:
        stream.write(rendered)


if __name__ == "__main__":
    main()
