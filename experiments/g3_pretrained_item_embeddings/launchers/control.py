from __future__ import annotations

import argparse
import base64
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
from typing import Any, Iterator

from experiments.g3_pretrained_item_embeddings.protocol.constants import APPROVED_PROTOCOL
from experiments.g3_pretrained_item_embeddings.protocol.control_ledger import (
    ControlLedger,
    load_control_ledger,
    validate_control_ledger_document,
)
from experiments.g3_pretrained_item_embeddings.protocol.manifests import (
    load_artifact_manifest,
    validate_artifact_bindings,
    validate_content_manifest,
    validate_feature_manifest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
JOB_ENVIRONMENT = "G3_CONTROL_JOB_B64"
LEDGER_ENVIRONMENT = "G3_CONTROL_LEDGER_PATH"
G3_CPU_THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS=1",
    "MKL_NUM_THREADS=1",
    "OPENBLAS_NUM_THREADS=1",
    "NUMEXPR_NUM_THREADS=1",
    "POLARS_MAX_THREADS=1",
)


@dataclass(frozen=True)
class CompiledControlJob:
    ledger_sha256: str
    row_id: str
    job: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "ledger_sha256": self.ledger_sha256,
            "row_id": self.row_id,
            "job": self.job,
        }


def encode_control_job(ledger: ControlLedger, row_id: str) -> str:
    compiled = _compile_job(ledger, row_id)
    return base64.urlsafe_b64encode(_canonical_bytes(compiled.to_dict())).decode()


def decode_control_job(encoded: str, ledger: ControlLedger) -> CompiledControlJob:
    try:
        decoded = base64.b64decode(encoded, altchars=b"-_", validate=True).decode()
        document = json.loads(
            decoded,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("compiled G3 control job is invalid") from error
    if not isinstance(document, dict) or set(document) != {
        "ledger_sha256",
        "row_id",
        "job",
    }:
        raise ValueError("compiled G3 control job has missing or unknown fields")
    row_id = document["row_id"]
    if not isinstance(row_id, str):
        raise ValueError("compiled G3 control job has an invalid row id")
    expected = _compile_job(ledger, row_id)
    if not _same_json_type_and_value(document, expected.to_dict()):
        raise ValueError("compiled G3 control job differs from its approved ledger row")
    return expected


def _same_json_type_and_value(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _same_json_type_and_value(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _same_json_type_and_value(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    return left == right


def compile_queue_commands(
    *,
    ledger_path: Path,
    ledger: Any,
    state_dir: Path,
    batch_id: str = "DRY_RUN_BATCH",
    runner_script: Path | None = None,
    job_environment: str = JOB_ENVIRONMENT,
    ledger_environment: str = LEDGER_ENVIRONMENT,
    data_group: str = "g3-native50m-likes",
    ledger_validator: Callable[[object], object] = validate_control_ledger_document,
) -> list[list[str]]:
    ledger_validator(ledger.to_dict())
    ledger_path = ledger_path.resolve()
    runner_script = (
        Path(__file__).with_name("run_control.py")
        if runner_script is None
        else runner_script
    ).resolve()
    commands = [
        _queue_command(state_dir, "status", "--json"),
        _queue_command(state_dir, "new-batch"),
    ]
    for row in ledger.rows:
        commands.append(
            _queue_command(
                state_dir,
                "enqueue-run",
                "--batch",
                batch_id,
                "--script",
                str(runner_script),
                "--run",
                row.run_name,
                "--data-group",
                data_group,
                "--",
                f"{job_environment}={encode_control_job(ledger, row.id)}",
                f"{ledger_environment}={ledger_path}",
                "WANDB_MODE=offline",
                *G3_CPU_THREAD_ENVIRONMENT,
            )
        )
    commands.append(_queue_command(state_dir, "seal-batch", batch_id))
    return commands


def find_existing_ledger_batch(
    *,
    state_dir: Path,
    ledger_path: Path,
    ledger: Any,
    runner_script: Path,
    job_environment: str,
    ledger_environment: str,
    data_group: str = "g3-native50m-likes",
    ledger_path_sensitive: bool = True,
) -> str | None:
    state_dir = state_dir.resolve()
    ledger_path = ledger_path.resolve()
    runner_script = runner_script.resolve()
    expected = []
    for row in ledger.rows:
        legacy_environment = [
            f"{job_environment}={encode_control_job(ledger, row.id)}",
            f"{ledger_environment}={ledger_path}",
            "WANDB_MODE=offline",
        ]
        environments = (
            legacy_environment,
            [*legacy_environment, *G3_CPU_THREAD_ENVIRONMENT],
        )
        expected.append(
            {
                "run": row.run_name,
                "script": str(runner_script),
                "data_group": data_group,
                "environments": tuple(
                    _normalized_ledger_environment(
                        environment,
                        ledger_environment=ledger_environment,
                        path_sensitive=ledger_path_sensitive,
                    )
                    for environment in environments
                ),
            }
        )
    jobs = {}
    for state in ("pending", "dispatched", "completed", "failed"):
        for path in (state_dir / state).glob("*.json"):
            value = _load_json(path)
            if isinstance(value, dict) and isinstance(value.get("id"), str):
                if value["id"] in jobs:
                    raise RuntimeError(f"queue job {value['id']} exists in two states")
                jobs[value["id"]] = value
    matches = []
    for path in (state_dir / "batches").glob("*.json"):
        batch = _load_json(path)
        if not isinstance(batch, dict) or batch.get("sealed") is not True:
            continue
        job_ids = batch.get("jobs")
        if not isinstance(job_ids, list) or len(job_ids) != len(expected):
            continue
        records = [jobs.get(job_id) for job_id in job_ids]
        if all(
            isinstance(record, dict)
            and all(
                record.get(key) == value
                for key, value in coordinate.items()
                if key != "environments"
            )
            and _normalized_ledger_environment(
                record.get("environment"),
                ledger_environment=ledger_environment,
                path_sensitive=ledger_path_sensitive,
            )
            in coordinate["environments"]
            and record.get("batch_id") == batch.get("id")
            for record, coordinate in zip(records, expected, strict=True)
        ):
            matches.append(batch.get("id"))
    if len(matches) > 1:
        raise RuntimeError("multiple queue batches are bound to the immutable ledger")
    if matches:
        return str(matches[0])
    ledger_binding = f"{ledger_environment}={ledger_path}"
    job_bindings = {
        f"{job_environment}={encode_control_job(ledger, row.id)}"
        for row in ledger.rows
    }
    if any(
        (
            ledger_binding in record.get("environment", [])
            if ledger_path_sensitive
            else bool(job_bindings.intersection(record.get("environment", [])))
        )
        for record in jobs.values()
        if isinstance(record.get("environment"), list)
    ):
        raise RuntimeError("partial queue state already exists for the immutable ledger")
    return None


def _normalized_ledger_environment(
    environment: object, *, ledger_environment: str, path_sensitive: bool
) -> object:
    if path_sensitive or not isinstance(environment, list):
        return environment
    prefix = f"{ledger_environment}="
    matches = [value for value in environment if str(value).startswith(prefix)]
    if len(matches) != 1:
        return environment
    return [
        f"{ledger_environment}=<immutable-ledger>"
        if str(value).startswith(prefix)
        else value
        for value in environment
    ]


@contextmanager
def ledger_submission_lock(
    *, state_dir: Path, ledger_sha256: str
) -> Iterator[None]:
    if (
        len(ledger_sha256) != 64
        or any(character not in "0123456789abcdef" for character in ledger_sha256)
    ):
        raise ValueError("ledger submission lock requires a lowercase SHA-256")
    lock_directory = state_dir.resolve() / "ledger-submission-locks"
    lock_directory.mkdir(parents=True, exist_ok=True)
    lock_path = lock_directory / f"{ledger_sha256}.lock"
    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def submit_control_jobs(
    *,
    ledger_path: Path,
    state_dir: Path,
    dry_run: bool,
) -> str:
    ledger_path = ledger_path.resolve()
    ledger = load_control_ledger(ledger_path)
    verify_ledger_inputs(PROJECT_ROOT, ledger, full_validation=True)
    if dry_run:
        return "\n".join(
            shlex.join(command)
            for command in compile_queue_commands(
                ledger_path=ledger_path,
                ledger=ledger,
                state_dir=state_dir,
            )
        )
    subprocess.run(
        _queue_command(state_dir, "status", "--json"),
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
    )
    created = subprocess.run(
        _queue_command(state_dir, "new-batch"),
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    batch_id = created.stdout.strip()
    if not batch_id:
        raise RuntimeError("training queue returned an empty batch id")
    commands = compile_queue_commands(
        ledger_path=ledger_path,
        ledger=ledger,
        state_dir=state_dir,
        batch_id=batch_id,
    )
    for command in commands[2:]:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    return batch_id


def verify_ledger_inputs(
    root: Path,
    ledger: ControlLedger,
    *,
    full_validation: bool,
) -> Path:
    root = root.resolve()
    g4_path = _resolve_reference(root, ledger.g4_control.path)
    if _file_sha256(g4_path) != ledger.g4_control.sha256:
        raise ValueError("selected G4 control manifest hash changed")
    g4 = _load_json(g4_path)
    selected = g4.get("seed_42_configuration") if isinstance(g4, dict) else None
    if not isinstance(selected, dict):
        raise ValueError("selected G4 control manifest has no seed-42 configuration")
    parameters = selected.get("selected")
    expected = {
        "batch_size": APPROVED_PROTOCOL.batch_size,
        "embedding_learning_rate": APPROVED_PROTOCOL.control.embedding_learning_rate,
        "deep_learning_rate": APPROVED_PROTOCOL.control.deep_learning_rate,
        "lr_schedule_horizon_epochs": APPROVED_PROTOCOL.control.horizon_epochs,
    }
    if selected.get("run_name") != APPROVED_PROTOCOL.control.run_name or parameters != expected:
        raise ValueError("selected G4 control manifest differs from the approved control")

    content_path = _resolve_reference(root, ledger.content.path)
    content = load_artifact_manifest(content_path)
    if content.sha256 != ledger.content.sha256:
        raise ValueError("G3 content manifest hash changed")
    validate_content_manifest(root=root, manifest=content, validate_files=False)
    validate_artifact_bindings(
        root=root,
        manifest=content,
        roles=(
            tuple(binding.role for binding in content.artifacts)
            if full_validation
            else ("compact_output", "compact_remap")
        ),
    )

    feature_path = _resolve_reference(root, ledger.features.path)
    features = load_artifact_manifest(feature_path)
    if features.sha256 != ledger.features.sha256:
        raise ValueError("G3 feature manifest hash changed")
    validate_feature_manifest(root=root, manifest=features, validate_files=False)
    runtime_feature_roles = (
        "item_features",
        "training_user_histories",
        "artist_vocab",
        "album_vocab",
    )
    validate_artifact_bindings(
        root=root,
        manifest=features,
        roles=(
            tuple(binding.role for binding in features.artifacts)
            if full_validation
            else runtime_feature_roles
        ),
    )
    item_features = next(
        binding for binding in features.artifacts if binding.role == "item_features"
    )
    return _resolve_reference(root, item_features.path)


def compiled_control_job_from_environment() -> tuple[CompiledControlJob, Path, Path]:
    encoded = os.environ.get(JOB_ENVIRONMENT)
    raw_ledger_path = os.environ.get(LEDGER_ENVIRONMENT)
    if not encoded or not raw_ledger_path:
        raise RuntimeError(f"{JOB_ENVIRONMENT} and {LEDGER_ENVIRONMENT} are required")
    ledger_path = Path(raw_ledger_path)
    ledger = load_control_ledger(ledger_path)
    feature_data_path = verify_ledger_inputs(
        PROJECT_ROOT,
        ledger,
        full_validation=False,
    )
    return decode_control_job(encoded, ledger), ledger_path, feature_data_path


def build_training_experiment(
    compiled: CompiledControlJob,
    *,
    feature_data_path: Path,
):
    from experiments.g3_pretrained_item_embeddings.configs.model import (
        G3Representation,
        build_g3_experiment,
    )

    job = compiled.job
    representation = job["representation"]
    training = job["training"]
    dataset = job["dataset"]
    if not isinstance(representation, dict) or not isinstance(training, dict) or not isinstance(dataset, dict):
        raise ValueError("compiled G3 control job has invalid nested configuration")
    if representation["id"] != "untied_learned_item_id":
        raise ValueError("compiled G3 control job is not the untied representation")
    return build_g3_experiment(
        run_name=str(job["run_name"]),
        dataset_size=dataset["size"],
        embedding_learning_rate=training["embedding_learning_rate"],
        deep_learning_rate=training["deep_learning_rate"],
        lr_schedule_horizon_epochs=training["horizon_epochs"],
        seed=training["seed"],
        representation=G3Representation(),
        feature_data_path=feature_data_path,
    )


def write_job_contract(
    compiled: CompiledControlJob,
    ledger_path: Path,
    logs_root: Path,
) -> Path:
    ledger = load_control_ledger(ledger_path)
    verified = decode_control_job(encode_control_job(ledger, compiled.row_id), ledger)
    if verified != compiled:
        raise ValueError("compiled G3 control job differs from its approved ledger row")
    return persist_job_contract(
        compiled=compiled,
        ledger_path=ledger_path,
        ledger_sha256=ledger.sha256,
        logs_root=logs_root,
        filename="g3_control_job.json",
    )


def persist_job_contract(
    *,
    compiled: CompiledControlJob,
    ledger_path: Path,
    ledger_sha256: str,
    logs_root: Path,
    filename: str,
) -> Path:
    destination = logs_root / str(compiled.job["run_name"]) / filename
    content = _canonical_bytes(
        compiled.to_dict()
        | {
            "ledger_path": str(ledger_path.resolve()),
            "ledger_sha256": ledger_sha256,
        }
    ) + b"\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("xb") as handle:
            handle.write(content)
    except FileExistsError:
        if destination.read_bytes() != content:
            raise RuntimeError(f"immutable G3 job contract differs: {destination}")
    return destination


def _compile_job(ledger: ControlLedger, row_id: str) -> CompiledControlJob:
    rows = [row for row in ledger.rows if row.id == row_id]
    if len(rows) != 1:
        raise ValueError(f"{row_id!r} is not an approved ledger row")
    return CompiledControlJob(ledger.sha256, row_id, rows[0].to_dict())


def _queue_command(state_dir: Path, *arguments: str) -> list[str]:
    return [
        "python",
        str(PROJECT_ROOT / "utils/training_queue/service.py"),
        "--state-dir",
        str(state_dir.resolve()),
        *arguments,
    ]


def _resolve_reference(root: Path, value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("manifest path must be project-relative")
    path = root / relative
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
        raise ValueError(f"manifest must be a regular project file: {value}")
    return path


def _load_json(path: Path) -> object:
    try:
        return json.loads(
            path.read_text(),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load JSON document {path}") from error


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON number {value!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ledger", type=Path)
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=PROJECT_ROOT / "generated/training-queue-service",
    )
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    print(
        submit_control_jobs(
            ledger_path=arguments.ledger,
            state_dir=arguments.state_dir,
            dry_run=arguments.dry_run,
        )
    )


if __name__ == "__main__":
    main()
