from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from experiments.g3_pretrained_item_embeddings.launchers.native500m import (
    JOB_ENVIRONMENT,
    MANIFEST_ENVIRONMENT,
    MANIFEST_LOGICAL_SHA256_ENVIRONMENT,
    MANIFEST_PHYSICAL_SHA256_ENVIRONMENT,
    ExecutionManifest,
    InputManifestReference,
    load_execution_manifest,
    resolve_input_manifest_path,
    execution_project_root,
    validate_current_source_ledger,
)


@dataclass(frozen=True)
class CompiledNative500MJob:
    job: dict[str, object]
    row_id: str
    manifest_path: Path
    manifest_logical_sha256: str
    manifest_physical_sha256: str
    protocol_sha256: str
    input_manifests: tuple[InputManifestReference, ...]
    implementation_identity: dict[str, object]
    evaluation_population: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "job": self.job,
            "row_id": self.row_id,
            "execution_manifest": {
                "path": str(self.manifest_path),
                "logical_sha256": self.manifest_logical_sha256,
                "physical_sha256": self.manifest_physical_sha256,
            },
            "protocol_sha256": self.protocol_sha256,
            "input_manifests": [value.to_dict() for value in self.input_manifests],
            "implementation_identity": self.implementation_identity,
            "evaluation_population": self.evaluation_population,
        }


def load_compiled_job(
    *, expected_protocol_sha256: str | None = None
) -> CompiledNative500MJob:
    required = {
        JOB_ENVIRONMENT: os.environ.get(JOB_ENVIRONMENT),
        MANIFEST_ENVIRONMENT: os.environ.get(MANIFEST_ENVIRONMENT),
        MANIFEST_LOGICAL_SHA256_ENVIRONMENT: os.environ.get(
            MANIFEST_LOGICAL_SHA256_ENVIRONMENT
        ),
        MANIFEST_PHYSICAL_SHA256_ENVIRONMENT: os.environ.get(
            MANIFEST_PHYSICAL_SHA256_ENVIRONMENT
        ),
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise RuntimeError(
            f"native-500M G3 runner environment is incomplete: {missing}"
        )
    manifest = load_execution_manifest(
        Path(str(required[MANIFEST_ENVIRONMENT])),
        expected_protocol_sha256=expected_protocol_sha256,
        expected_logical_sha256=str(required[MANIFEST_LOGICAL_SHA256_ENVIRONMENT]),
        expected_physical_sha256=str(required[MANIFEST_PHYSICAL_SHA256_ENVIRONMENT]),
        validate_inputs=True,
    )
    return decode_compiled_job(str(required[JOB_ENVIRONMENT]), manifest)


def decode_compiled_job(
    encoded: str, manifest: ExecutionManifest
) -> CompiledNative500MJob:
    payload = _decode_job(encoded)
    if payload.get("manifest_logical_sha256") != manifest.logical_sha256:
        raise ValueError("compiled job manifest logical SHA-256 differs")
    if payload.get("manifest_physical_sha256") != manifest.physical_sha256:
        raise ValueError("compiled job manifest physical SHA-256 differs")
    row_id = payload.get("row_id")
    matches = [row for row in manifest.rows if row["id"] == row_id]
    if len(matches) != 1 or _canonical_bytes(payload.get("job")) != _canonical_bytes(
        matches[0]["job"]
    ):
        raise ValueError("compiled job differs from its execution manifest row")
    return CompiledNative500MJob(
        job=dict(matches[0]["job"]),
        row_id=str(row_id),
        manifest_path=manifest.path,
        manifest_logical_sha256=manifest.logical_sha256,
        manifest_physical_sha256=manifest.physical_sha256,
        protocol_sha256=manifest.protocol_sha256,
        input_manifests=manifest.input_manifests,
        implementation_identity=manifest.implementation_identity,
        evaluation_population=manifest.evaluation_population,
    )


def build_training_experiment(compiled: CompiledNative500MJob):
    from experiments.g3_pretrained_item_embeddings.configs.model import (
        build_native500m_job,
    )
    from experiments.g3_pretrained_item_embeddings.protocol.native500m.artifacts import (
        load_artifact_manifest,
        validate_content_manifest,
        validate_dataset_manifest,
        validate_feature_manifest,
    )

    root = _project_root(compiled.manifest_path)
    validators = {
        "content": validate_content_manifest,
        "dataset": validate_dataset_manifest,
        "features": validate_feature_manifest,
    }
    manifests = {}
    for reference in compiled.input_manifests:
        manifest_path = resolve_input_manifest_path(compiled.manifest_path, reference)
        manifests[reference.role] = validators[reference.role](
            load_artifact_manifest(manifest_path),
            root=root,
            validate_files=True,
            validate_semantics=True,
        )
    feature_manifest = manifests["features"]
    feature_binding = next(
        binding
        for binding in feature_manifest.artifacts
        if binding.role == "item_features"
    )
    feature_data_path = root / feature_binding.path
    experiment = build_native500m_job(compiled.job, feature_data_path=feature_data_path)
    experiment.base_path = root / "generated"
    validate_current_source_ledger(compiled.implementation_identity)
    experiment.g3_execution_identity = compiled.implementation_identity
    experiment.g3_evaluation_population = compiled.evaluation_population
    validate_current_source_ledger(compiled.implementation_identity)
    return experiment


def write_job_contract(compiled: CompiledNative500MJob, logs_root: Path) -> Path:
    run_name = str(compiled.job["run_name"])
    path = logs_root / run_name / "g3_native500m_job.json"
    content = _pretty_bytes(compiled.to_dict())
    _write_immutable(path, content)
    return path


def _project_root(manifest_path: Path) -> Path:
    return execution_project_root(manifest_path)


def _decode_job(encoded: str) -> dict[str, Any]:
    try:
        document = json.loads(
            base64.b64decode(encoded, altchars=b"-_", validate=True).decode(),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("compiled native-500M G3 job is invalid") from error
    if not isinstance(document, dict) or set(document) != {
        "manifest_logical_sha256",
        "manifest_physical_sha256",
        "row_id",
        "job",
    }:
        raise ValueError("compiled native-500M G3 job schema differs")
    return document


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def _pretty_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode()


def _write_immutable(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"immutable native-500M G3 job contract differs: {path}")
        return
    with NamedTemporaryFile(
        mode="wb", prefix=f".{path.name}.", dir=path.parent, delete=False
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    try:
        try:
            os.link(temporary, path)
        except FileExistsError:
            pass
    finally:
        temporary.unlink(missing_ok=True)
    if path.read_bytes() != content:
        raise RuntimeError(f"immutable native-500M G3 job contract differs: {path}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r}")


experiment = None
if os.environ.get(JOB_ENVIRONMENT) is not None:
    compiled_job = load_compiled_job()
    experiment = build_training_experiment(compiled_job)
    write_job_contract(
        compiled_job,
        Path(experiment.base_path) / "logs",
    )
