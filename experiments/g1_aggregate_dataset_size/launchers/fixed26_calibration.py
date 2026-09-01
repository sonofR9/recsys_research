from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path

from experiments.g1_sasrec_item_ids_likes.launchers.verify_artifact import (
    verify_config_recipe,
)

from experiments.g1_aggregate_dataset_size.protocol.fixed26_calibration import (
    Fixed26CalibrationJob,
    Fixed26CalibrationManifest,
    load_fixed26_manifest,
)


CONTRACT_NAME = "fixed26_calibration_job.json"
RECIPE_PATH = Path(__file__).with_name("run_fixed26_calibration.py")
SOURCE_CONFIG_PATH = Path(__file__).parents[1] / "configs" / "aggregate_variant.py"
MANIFEST_SHA_ENV = "G1_FIXED26_CALIBRATION_MANIFEST_SHA256"
RECIPE_SHA_ENV = "G1_FIXED26_CALIBRATION_RECIPE_SHA256"


@dataclass(frozen=True)
class CalibrationQueueRow:
    run_name: str
    manifest_sha256: str
    environment: tuple[str, ...]


def calibration_queue_rows(
    manifest: Fixed26CalibrationManifest,
) -> tuple[CalibrationQueueRow, ...]:
    return tuple(
        CalibrationQueueRow(
            run_name=job.run_name,
            manifest_sha256=manifest.sha256,
            environment=(
                f"G1_FIXED26_CALIBRATION_RUN={job.run_name}",
                f"{MANIFEST_SHA_ENV}={manifest.sha256}",
                f"{RECIPE_SHA_ENV}={manifest.recipe_sha256}",
            ),
        )
        for job in manifest.jobs
    )


def atomic_queue_specification(
    manifest: Fixed26CalibrationManifest, wandb_mode: str
) -> dict[str, object]:
    if wandb_mode not in {"disabled", "offline", "online"}:
        raise ValueError("unsupported WANDB_MODE")
    return {
        "version": 1,
        "jobs": [
            {
                "script": (
                    "experiments/g1_aggregate_dataset_size/launchers/"
                    "run_fixed26_calibration.py"
                ),
                "run": row.run_name,
                "data_group": "g1-aggregate-50m-seq100",
                "environment": [*row.environment, f"WANDB_MODE={wandb_mode}"],
            }
            for row in calibration_queue_rows(manifest)
        ],
    }


def job_contract_bytes(
    job: Fixed26CalibrationJob,
    manifest: Fixed26CalibrationManifest,
) -> bytes:
    if job not in manifest.jobs:
        raise ValueError("job is outside the fixed-26 calibration manifest")
    document = {
        "manifest_sha256": manifest.sha256,
        "recipe_sha256": manifest.recipe_sha256,
        "experiment_fingerprint_sha256": manifest.experiment_fingerprint_sha256,
        "job": job.to_dict(),
    }
    return (
        json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()


def write_job_contract(
    logs: Path,
    job: Fixed26CalibrationJob,
    manifest: Fixed26CalibrationManifest,
) -> Path:
    destination = logs / job.run_name / CONTRACT_NAME
    content = job_contract_bytes(job, manifest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("xb") as stream:
            stream.write(content)
    except FileExistsError:
        if destination.read_bytes() != content:
            raise RuntimeError(
                f"immutable calibration job contract differs: {destination}"
            )
    return destination


def verify_job_contract(
    directory: Path,
    job: Fixed26CalibrationJob,
    manifest: Fixed26CalibrationManifest,
) -> bool:
    path = directory / CONTRACT_NAME
    try:
        content = path.read_bytes()
    except OSError:
        return False
    return content == job_contract_bytes(job, manifest)


def artifact_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def experiment_fingerprint(experiment: object) -> str:
    fields = asdict(experiment)
    fields.pop("run_name")
    fields.pop("seed")
    payload = json.dumps(
        {
            "experiment_class": (
                f"{type(experiment).__module__}.{type(experiment).__qualname__}"
            ),
            "fields": fields,
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=_canonical_json_value,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_value(value: object) -> object:
    if isinstance(value, Path):
        return {"type": "path", "value": str(value)}
    value_type = f"{type(value).__module__}.{type(value).__qualname__}"
    if value_type == "torch.dtype":
        return {"type": value_type, "value": str(value)}
    raise TypeError(f"unsupported experiment fingerprint value: {value_type}")


def verify_experiment_contract(
    experiment: object, manifest: Fixed26CalibrationManifest
) -> None:
    if experiment_fingerprint(experiment) != manifest.experiment_fingerprint_sha256:
        raise RuntimeError("fixed-26 experiment recipe no longer reproduces")


def verify_source_control_artifacts(
    logs: Path, manifest: Fixed26CalibrationManifest
) -> bool:
    directory = logs / manifest.source_run_name
    artifacts_match = all(
        path.is_file() and artifact_sha256(path) == expected
        for name, expected in manifest.source_artifact_sha256.items()
        for path in (directory / name,)
    )
    if not artifacts_match:
        return False
    try:
        return verify_config_recipe(
            directory,
            SOURCE_CONFIG_PATH,
            [
                f"G1_AGGREGATE_RUN={manifest.source_run_name}",
                "G1_DATASET_SIZE=50m",
            ],
        )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError):
        return False


def verify_worker_contract(
    logs: Path,
    manifest: Fixed26CalibrationManifest,
    recipe_path: Path = RECIPE_PATH,
) -> None:
    if os.environ.get(MANIFEST_SHA_ENV) != manifest.sha256:
        raise RuntimeError("fixed-26 submit-time manifest hash changed")
    if os.environ.get(RECIPE_SHA_ENV) != manifest.recipe_sha256:
        raise RuntimeError("fixed-26 submit-time recipe hash changed")
    if artifact_sha256(recipe_path) != manifest.recipe_sha256:
        raise RuntimeError("fixed-26 queued recipe changed")
    if not verify_source_control_artifacts(logs, manifest):
        raise RuntimeError("fixed-26 source control no longer reproduces")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("manifest")
    subparsers.add_parser("manifest-sha256")
    subparsers.add_parser("recipe-sha256")
    subparsers.add_parser("verify-jobs")
    queue_spec = subparsers.add_parser("queue-spec")
    queue_spec.add_argument("--wandb-mode", required=True)
    source_parser = subparsers.add_parser("verify-source")
    source_parser.add_argument("--logs", type=Path, required=True)
    arguments = parser.parse_args()
    manifest = load_fixed26_manifest()
    if arguments.command == "verify-source":
        raise SystemExit(
            0 if verify_source_control_artifacts(arguments.logs, manifest) else 1
        )
    if arguments.command == "manifest-sha256":
        print(manifest.sha256)
        return
    if arguments.command == "recipe-sha256":
        print(manifest.recipe_sha256)
        return
    if arguments.command == "verify-jobs":
        from experiments.g1_aggregate_dataset_size.configs.fixed26_calibration import (
            build_fixed26_experiment,
        )

        for job in manifest.jobs:
            verify_experiment_contract(
                build_fixed26_experiment(job, manifest), manifest
            )
        return
    if arguments.command == "queue-spec":
        print(
            json.dumps(
                atomic_queue_specification(manifest, arguments.wandb_mode),
                sort_keys=True,
            )
        )
        return
    for row in calibration_queue_rows(manifest):
        print(row.run_name)


if __name__ == "__main__":
    main()
