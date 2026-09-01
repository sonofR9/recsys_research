from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any, Callable

from experiments.g6_rqkmeans_history.native500m.protocol.contracts import (
    JOB_SCHEMA,
    load_stage_manifest,
)

PROJECT_ROOT = Path(__file__).resolve().parents[4]
APPROVAL_PATH = (
    PROJECT_ROOT / "experiments/g6_rqkmeans_history/protocol/native500m_approval.json"
)
PLAN_SHA256 = "3561064c58087cff75b0029c62eb477104b8bce51ff77b4868f3733d0b218910"
JOB_ENVIRONMENT = "G6_NATIVE500M_JOB_B64"
MANIFEST_PATH_ENVIRONMENT = "G6_NATIVE500M_MANIFEST_PATH"
MANIFEST_LOGICAL_SHA256_ENVIRONMENT = "G6_NATIVE500M_MANIFEST_LOGICAL_SHA256"
MANIFEST_PHYSICAL_SHA256_ENVIRONMENT = "G6_NATIVE500M_MANIFEST_PHYSICAL_SHA256"
JOB_LOGICAL_SHA256_ENVIRONMENT = "G6_NATIVE500M_JOB_LOGICAL_SHA256"
CONFIG_LOGICAL_SHA256_ENVIRONMENT = "G6_NATIVE500M_CONFIG_LOGICAL_SHA256"
CPU_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "POLARS_MAX_THREADS": "1",
}
MAX_BATCH_SHARD_JOBS = 4
_SHA256 = re.compile(r"[0-9a-f]{64}")
_ENVIRONMENT_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@dataclass(frozen=True)
class QueueJob:
    job_id: str
    run_name: str
    runner: str
    config_logical_sha256: str
    data_group: str
    logical_sha256: str
    payload: dict[str, Any]
    environment: dict[str, str]

    @property
    def seed(self) -> int:
        parameters = self.payload.get("parameters")
        seed = parameters.get("seed") if isinstance(parameters, dict) else None
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("queue job seed is invalid")
        return seed


@dataclass(frozen=True)
class QueueManifest:
    path: Path
    stage: str
    logical_sha256: str
    physical_sha256: str
    plan_sha256: str
    approval_sha256: str
    jobs: tuple[QueueJob, ...]
    compiler_recipe_sha256: str | None = None


@dataclass(frozen=True)
class BatchSpecification:
    document: dict[str, Any]
    sha256: str


def canonical_bytes(document: object) -> bytes:
    return json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def source_identity_sha256() -> str:
    native_root = Path(__file__).resolve().parents[1]
    files = [
        path
        for root in (
            PROJECT_ROOT / "dcn",
            PROJECT_ROOT / "neuralrec",
            native_root / "configs",
        )
        for path in root.rglob("*.py")
        if not any(part in {"__pycache__", "tests", "old"} for part in path.parts)
        and not path.name.startswith("test_")
    ]
    files.extend(
        (
            native_root / "launchers/runtime.py",
            native_root / "launchers/run_native500m.py",
        )
    )
    files = sorted(set(files))
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(PROJECT_ROOT).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def load_queue_manifest(
    path: Path,
    *,
    protocol_loader: Callable[[Path], object] | None = None,
    verify_current_source: bool = False,
) -> QueueManifest:
    resolved = path.resolve(strict=True)
    if resolved.is_symlink() or not resolved.is_file():
        raise ValueError("stage manifest must be a regular file")
    raw = resolved.read_bytes()
    physical_sha256 = hashlib.sha256(raw).hexdigest()
    loader = load_stage_manifest if protocol_loader is None else protocol_loader
    loaded = loader(resolved)
    to_document = getattr(loaded, "to_document", None)
    if not callable(to_document):
        raise TypeError("protocol manifest loader returned no document")
    document = to_document()
    expected_keys = {
        "schema",
        "stage",
        "dataset_size",
        "batch_size",
        "training_horizon",
        "plan_sha256",
        "approval_sha256",
        "predecessor",
        "jobs",
        "sha256",
    }
    if not isinstance(document, dict) or set(document) != expected_keys:
        raise ValueError("stage manifest schema differs")
    payload = {key: value for key, value in document.items() if key != "sha256"}
    logical_sha256 = _require_sha256(document["sha256"], "manifest logical SHA-256")
    if hashlib.sha256(canonical_bytes(payload)).hexdigest() != logical_sha256:
        raise ValueError("stage manifest logical SHA-256 differs")
    approval_sha256 = hashlib.sha256(APPROVAL_PATH.read_bytes()).hexdigest()
    if (
        document["schema"] != "g6-native500m-stage-manifest/v1"
        or document["dataset_size"] != "native-500m"
        or document["batch_size"] != 512
        or document["training_horizon"] != 26
        or document["plan_sha256"] != PLAN_SHA256
        or document["approval_sha256"] != approval_sha256
    ):
        raise ValueError("stage manifest protocol identity differs")
    stage = document["stage"]
    if not isinstance(stage, str) or not stage:
        raise ValueError("stage manifest has no stage")
    predecessor = document["predecessor"]
    if predecessor is None and "control" not in stage:
        raise ValueError("only control stages may omit a predecessor")
    if predecessor is not None and (
        not isinstance(predecessor, dict)
        or set(predecessor) != {"stage", "selection_sha256", "resolved"}
        or predecessor.get("resolved") is not True
        or not isinstance(predecessor.get("stage"), str)
        or not _SHA256.fullmatch(str(predecessor.get("selection_sha256")))
    ):
        raise ValueError("stage manifest predecessor is unresolved")
    rows = document["jobs"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("stage manifest has no jobs")
    source_sha256 = source_identity_sha256() if verify_current_source else None
    jobs = tuple(_queue_job(row, stage, approval_sha256, source_sha256) for row in rows)
    if len({job.job_id for job in jobs}) != len(jobs) or len(
        {job.run_name for job in jobs}
    ) != len(jobs):
        raise ValueError("stage manifest job identities are not unique")
    return QueueManifest(
        path=resolved,
        stage=stage,
        logical_sha256=logical_sha256,
        physical_sha256=physical_sha256,
        plan_sha256=PLAN_SHA256,
        approval_sha256=approval_sha256,
        jobs=jobs,
    )


def load_admitted_queue_manifest(path: Path) -> QueueManifest:
    from experiments.g6_rqkmeans_history.native500m.launchers.materialize import (
        rederive_manifest_for_admission,
    )

    recipe_sha256 = rederive_manifest_for_admission(path)
    manifest = load_queue_manifest(path, verify_current_source=True)
    if any(job.payload.get("schema") != JOB_SCHEMA for job in manifest.jobs):
        raise ValueError("queue admission requires current job contracts")
    return QueueManifest(
        **{
            name: getattr(manifest, name)
            for name in (
                "path",
                "stage",
                "logical_sha256",
                "physical_sha256",
                "plan_sha256",
                "approval_sha256",
                "jobs",
            )
        },
        compiler_recipe_sha256=recipe_sha256,
    )


def build_batch_specification(
    manifest: QueueManifest,
    *,
    included_job_ids: frozenset[str] | None = None,
) -> BatchSpecification:
    physical_job_ids = {
        job.job_id for job in manifest.jobs if not job.payload["exact_reuse"]
    }
    if included_job_ids is not None:
        if not included_job_ids:
            raise ValueError("batch shard must be a nonempty subset")
        absent = included_job_ids - physical_job_ids
        if absent:
            raise ValueError(f"batch shard jobs are absent: {sorted(absent)}")
    jobs = []
    for job in manifest.jobs:
        if job.payload["exact_reuse"] or (
            included_job_ids is not None and job.job_id not in included_job_ids
        ):
            continue
        required = {
            JOB_ENVIRONMENT: base64.urlsafe_b64encode(
                canonical_bytes(job.payload)
            ).decode(),
            MANIFEST_PATH_ENVIRONMENT: str(manifest.path),
            MANIFEST_LOGICAL_SHA256_ENVIRONMENT: manifest.logical_sha256,
            MANIFEST_PHYSICAL_SHA256_ENVIRONMENT: manifest.physical_sha256,
            JOB_LOGICAL_SHA256_ENVIRONMENT: job.logical_sha256,
            CONFIG_LOGICAL_SHA256_ENVIRONMENT: job.config_logical_sha256,
            "G6_NATIVE500M_PLAN_SHA256": manifest.plan_sha256,
            "G6_NATIVE500M_APPROVAL_SHA256": manifest.approval_sha256,
            "WANDB_MODE": "offline",
            **CPU_ENVIRONMENT,
        }
        overlap = set(required) & set(job.environment)
        if overlap:
            raise ValueError(
                f"job environment overrides authenticated fields: {sorted(overlap)}"
            )
        environment = {**job.environment, **required}
        jobs.append(
            {
                "script": job.runner,
                "run": job.run_name,
                "data_group": job.data_group,
                "environment": [
                    f"{name}={value}" for name, value in sorted(environment.items())
                ],
            }
        )
    document = {"version": 1, "jobs": jobs}
    return BatchSpecification(
        document=document,
        sha256=hashlib.sha256(canonical_bytes(document)).hexdigest(),
    )


def batch_shard_job_ids(
    manifest: QueueManifest, *, shard_index: int, shard_count: int
) -> frozenset[str]:
    if (
        isinstance(shard_index, bool)
        or isinstance(shard_count, bool)
        or not isinstance(shard_index, int)
        or not isinstance(shard_count, int)
        or shard_count < 1
        or shard_index < 0
        or shard_index >= shard_count
    ):
        raise ValueError("batch shard coordinates are invalid")
    physical_jobs = tuple(
        job for job in manifest.jobs if not job.payload["exact_reuse"]
    )
    if shard_count > len(physical_jobs):
        raise ValueError("batch shard count exceeds physical job count")
    included = frozenset(
        job.job_id
        for index, job in enumerate(physical_jobs)
        if index % shard_count == shard_index
    )
    if len(included) > MAX_BATCH_SHARD_JOBS:
        raise ValueError(f"batch shard exceeds the {MAX_BATCH_SHARD_JOBS}-job maximum")
    return included


def persist_batch_specification(
    directory: Path, specification: BatchSpecification
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{specification.sha256}.json"
    content = canonical_bytes(specification.document)
    if hashlib.sha256(content).hexdigest() != specification.sha256:
        raise ValueError("batch specification SHA-256 differs")
    persist_immutable_bytes(path, content, label="batch specification")
    return path


def persist_immutable_bytes(path: Path, content: bytes, *, label: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != content:
            raise RuntimeError(f"immutable {label} differs: {path}")
        return path
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
            temporary = Path(stream.name)
        temporary.chmod(0o444)
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink() or not path.is_file() or path.read_bytes() != content:
                raise RuntimeError(f"immutable {label} differs: {path}")
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return path
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def resolve_or_submit_batch(
    manifest: QueueManifest,
    *,
    state_directory: Path,
    specification_directory: Path,
    included_job_ids: frozenset[str] | None = None,
    dry_run: bool = False,
) -> str:
    if manifest.compiler_recipe_sha256 is None:
        raise ValueError("queue admission requires compiler recipe replay")
    current_source = source_identity_sha256()
    if any(
        job.environment.get("G6_NATIVE500M_SOURCE_SHA256") != current_source
        for job in manifest.jobs
    ):
        raise ValueError("stage manifest was not rederived from the current source")
    specification = build_batch_specification(
        manifest, included_job_ids=included_job_ids
    )
    if len(specification.document["jobs"]) > MAX_BATCH_SHARD_JOBS:
        raise ValueError(
            f"queue submission exceeds the {MAX_BATCH_SHARD_JOBS}-job maximum; "
            "use authenticated shards"
        )
    specification_path = persist_batch_specification(
        specification_directory, specification
    )
    if dry_run:
        return specification_path.read_text()
    found = _queue_call("find-batch", state_directory, specification_path)
    if found.returncode == 0:
        return _batch_id(found)
    if found.returncode != 3:
        raise RuntimeError(found.stderr.strip() or "training queue lookup failed")
    submitted = _queue_call("submit-batch", state_directory, specification_path)
    if submitted.returncode != 0:
        raise RuntimeError(
            submitted.stderr.strip() or "training queue submission failed"
        )
    return _batch_id(submitted)


def _queue_job(
    value: object, stage: str, approval_sha256: str, source_sha256: str | None
) -> QueueJob:
    expected = {
        "schema",
        "job_id",
        "stage",
        "dataset_size",
        "batch_size",
        "training_horizon",
        "schedule",
        "plan_sha256",
        "approval_sha256",
        "parameters",
        "source_selection",
        "exact_reuse",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("stage manifest job schema differs")
    if (
        value["schema"] not in {"g6-native500m-job/v1", "g6-native500m-job/v2"}
        or value["stage"] != stage
        or value["dataset_size"] != "native-500m"
        or value["batch_size"] != 512
        or value["training_horizon"] != 26
        or value["schedule"] not in {"annealed", "constant"}
        or value["plan_sha256"] != PLAN_SHA256
        or value["approval_sha256"] != approval_sha256
    ):
        raise ValueError("stage jobs must use native 500M, batch 512, and horizon 26")
    job_id = value["job_id"]
    parameters = value["parameters"]
    if not isinstance(job_id, str) or not job_id or not isinstance(parameters, dict):
        raise ValueError("stage job identity is invalid")
    run_name = parameters.get("run_name")
    runner = parameters.get("runner")
    config_sha256 = parameters.get("config_logical_sha256")
    data_group = parameters.get("data_group")
    environment = parameters.get("environment", {})
    if not isinstance(run_name, str) or not run_name:
        raise ValueError("stage job has no explicit run name")
    if not isinstance(runner, str) or not runner:
        raise ValueError("stage job has no explicit runner")
    _require_sha256(config_sha256, "config logical SHA-256")
    if not isinstance(data_group, str) or not re.fullmatch(
        r"[A-Za-z0-9_.-]+", data_group
    ):
        raise ValueError("stage job data group is invalid")
    if not isinstance(environment, dict) or any(
        not isinstance(name, str)
        or not _ENVIRONMENT_NAME.fullmatch(name)
        or not isinstance(setting, str)
        for name, setting in environment.items()
    ):
        raise ValueError("stage job environment is invalid")
    if (
        source_sha256 is not None
        and environment.get("G6_NATIVE500M_SOURCE_SHA256") != source_sha256
    ):
        raise ValueError("stage job source identity differs")
    return QueueJob(
        job_id=job_id,
        run_name=run_name,
        runner=runner,
        config_logical_sha256=config_sha256,
        data_group=data_group,
        logical_sha256=hashlib.sha256(canonical_bytes(value)).hexdigest(),
        payload=dict(value),
        environment=dict(environment),
    )


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{name} is invalid")
    return value


def _queue_call(
    action: str, state_directory: Path, specification_path: Path
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "python",
            str(PROJECT_ROOT / "utils/training_queue/service.py"),
            "--state-dir",
            str(state_directory.resolve()),
            action,
            str(specification_path.resolve()),
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def _batch_id(completed: subprocess.CompletedProcess[str]) -> str:
    batch_id = completed.stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{32}", batch_id):
        raise RuntimeError("training queue returned an invalid batch ID")
    return batch_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=PROJECT_ROOT / "generated/training-queue-service",
    )
    parser.add_argument(
        "--specification-dir",
        type=Path,
        default=PROJECT_ROOT / "generated/g6-native500m/batch-specifications",
    )
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int)
    arguments = parser.parse_args()
    manifest = load_admitted_queue_manifest(arguments.manifest)
    if (arguments.shard_index is None) != (arguments.shard_count is None):
        parser.error("--shard-index and --shard-count must be supplied together")
    included_job_ids = (
        None
        if arguments.shard_index is None
        else batch_shard_job_ids(
            manifest,
            shard_index=arguments.shard_index,
            shard_count=arguments.shard_count,
        )
    )
    print(
        resolve_or_submit_batch(
            manifest,
            state_directory=arguments.state_dir,
            specification_directory=arguments.specification_dir,
            included_job_ids=included_job_ids,
            dry_run=not arguments.submit,
        )
    )


if __name__ == "__main__":
    main()
