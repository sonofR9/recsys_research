from __future__ import annotations

import argparse
import base64
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import NamedTemporaryFile, mkdtemp
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_GROUP = "g3-native500m-likes"
JOB_ENVIRONMENT = "G3_NATIVE500M_JOB_B64"
MANIFEST_ENVIRONMENT = "G3_NATIVE500M_EXECUTION_MANIFEST_PATH"
MANIFEST_LOGICAL_SHA256_ENVIRONMENT = "G3_NATIVE500M_EXECUTION_MANIFEST_LOGICAL_SHA256"
MANIFEST_PHYSICAL_SHA256_ENVIRONMENT = (
    "G3_NATIVE500M_EXECUTION_MANIFEST_PHYSICAL_SHA256"
)
SOURCE_SNAPSHOT_ROOT_ENVIRONMENT = "IMMUTABLE_SOURCE_SNAPSHOT_ROOT"
SOURCE_SNAPSHOT_IDENTITY_ENVIRONMENT = "IMMUTABLE_SOURCE_SNAPSHOT_IDENTITY_SHA256"
SOURCE_SNAPSHOT_MANIFEST_ENVIRONMENT = "IMMUTABLE_SOURCE_SNAPSHOT_MANIFEST_SHA256"
CPU_THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS=1",
    "MKL_NUM_THREADS=1",
    "OPENBLAS_NUM_THREADS=1",
    "NUMEXPR_NUM_THREADS=1",
    "POLARS_MAX_THREADS=1",
)
_INPUT_ROLES = ("content", "dataset", "features")
_IMPLEMENTATION_SOURCE_ROOTS = (
    "dcn",
    "neuralrec",
    "data",
    "utils",
    "experiments/g3_pretrained_item_embeddings",
)
_IMPLEMENTATION_EXCLUDED_PARTS = {
    "__pycache__",
    "old",
    "tests",
    "third_party",
    "third-party",
    "vendor",
    "vendored",
}
_IMPLEMENTATION_EXPLICIT_PATHS = {
    "experiments/g4_future_items/__init__.py",
    "experiments/g4_future_items/configs/__init__.py",
    "experiments/g4_future_items/configs/control.py",
    "experiments/g4_future_items/protocol/control_manifest.json",
    "experiments/generation_protocol.py",
    "utils/training_queue/queue.sh",
    "utils/training_queue/service_scheduler.sh",
}

_QUEUE_RUNTIME_PATHS = (
    "utils/training_queue/service.py",
    "utils/training_queue/service_scheduler.sh",
    "utils/training_queue/queue.sh",
    "utils/training_queue/queue_depth.py",
    "utils/training_queue/gpu_check.py",
)
_SOURCE_SNAPSHOT_DIRECTORY = "generated/g3-native500m/source-snapshots"
_SOURCE_SNAPSHOT_MANIFEST = ".g3-native500m-source-snapshot.json"
_BOUND_SOURCE_REPLAY_CACHE: set[tuple[str, ...]] = set()


def _regular_project_path(relative: str, *, kind: str = "file") -> Path:
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(
            f"execution implementation path escapes the project: {relative}"
        )
    path = PROJECT_ROOT
    for index, part in enumerate(relative_path.parts):
        path = path / part
        if path.is_symlink():
            raise ValueError(
                f"execution implementation path traverses a symlink: {relative}"
            )
        if index < len(relative_path.parts) - 1 and not path.is_dir():
            raise ValueError(f"execution implementation parent is absent: {relative}")
    if kind == "directory":
        if not path.is_dir():
            raise ValueError(f"execution implementation root is absent: {relative}")
    elif not path.is_file():
        raise ValueError(f"execution implementation file is absent: {relative}")
    return path


def _implementation_paths() -> tuple[str, ...]:
    paths = set()
    for relative in _IMPLEMENTATION_EXPLICIT_PATHS:
        _regular_project_path(relative)
        paths.add(relative)
    for root_name in _IMPLEMENTATION_SOURCE_ROOTS:
        root = _regular_project_path(root_name, kind="directory")
        for path in root.rglob("*"):
            relative = path.relative_to(PROJECT_ROOT)
            if _IMPLEMENTATION_EXCLUDED_PARTS.intersection(relative.parts):
                continue
            if path.is_symlink():
                raise ValueError(
                    f"execution implementation source is a symlink: {relative}"
                )
            if (
                path.suffix == ".py"
                and not path.name.startswith("test_")
                and path.name != "conftest.py"
            ):
                if not path.is_file():
                    raise ValueError(
                        f"execution implementation source is not regular: {relative}"
                    )
                paths.add(relative.as_posix())
    return tuple(sorted(paths))


APPROVED_EVALUATION_POPULATION = {
    "num_users": 37_018,
    "user_ids_sha256": "108108195dace6e5efdf9ebd7c7e8101ccc4c8d27b9fc949a4a43a6a89bf3d63",
}


@dataclass(frozen=True)
class InputManifestReference:
    role: str
    path: str
    size_bytes: int
    sha256: str
    logical_sha256: str

    @classmethod
    def from_path(
        cls,
        *,
        role: str,
        root: Path,
        path: Path,
        logical_sha256: str,
    ) -> InputManifestReference:
        root = root.resolve(strict=True)
        resolved = path.resolve(strict=True)
        if (
            role not in _INPUT_ROLES
            or resolved.is_symlink()
            or not resolved.is_file()
            or not resolved.is_relative_to(root)
        ):
            raise ValueError("input manifest is not a regular project artifact")
        _validate_sha256(logical_sha256, "input logical SHA-256")
        return cls(
            role=role,
            path=resolved.relative_to(root).as_posix(),
            size_bytes=resolved.stat().st_size,
            sha256=_file_sha256(resolved),
            logical_sha256=logical_sha256,
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ExecutionManifest:
    path: Path
    stage: str
    protocol_sha256: str
    rows: tuple[dict[str, object], ...]
    input_manifests: tuple[InputManifestReference, ...]
    implementation_identity: dict[str, object]
    source_snapshot: Path
    evaluation_population: dict[str, object]
    logical_sha256: str
    physical_sha256: str


@dataclass(frozen=True)
class BatchSpecification:
    document: dict[str, object]
    sha256: str
    manifest_logical_sha256: str
    manifest_physical_sha256: str


def freeze_execution_manifest(
    path: Path,
    *,
    stage: str,
    rows: Sequence[object],
    input_manifests: Sequence[InputManifestReference],
    job_payloads: Mapping[str, Mapping[str, object]] | None = None,
    protocol_sha256: str | None = None,
) -> Path:
    _reject_symlink_ancestors(path)
    protocol_sha256 = _protocol_sha256(protocol_sha256)
    if not stage or not rows:
        raise ValueError("execution manifest requires a stage and rows")
    implementation_identity = build_implementation_identity()
    implementation_prefix = str(implementation_identity["sha256"])[:12]
    if f"-i{implementation_prefix}" not in path.stem:
        path = path.with_name(f"{path.stem}-i{implementation_prefix}{path.suffix}")
    source_snapshot = _materialize_source_snapshot(
        execution_manifest_path=path,
        implementation_identity=implementation_identity,
    )
    normalized_rows = tuple(
        _execution_row(
            row,
            None if job_payloads is None else job_payloads.get(_row_id(row)),
            implementation_prefix=implementation_prefix,
        )
        for row in rows
    )
    row_ids = [str(row["id"]) for row in normalized_rows]
    run_names = [str(row["job"]["run_name"]) for row in normalized_rows]
    if len(set(row_ids)) != len(row_ids) or len(set(run_names)) != len(run_names):
        raise ValueError("execution manifest row IDs and run names must be unique")
    references = tuple(sorted(input_manifests, key=lambda value: value.role))
    if tuple(reference.role for reference in references) != _INPUT_ROLES:
        raise ValueError("execution manifest must bind dataset, content, and features")
    body = {
        "schema_version": 2,
        "kind": "g3_native500m_execution_manifest",
        "protocol_sha256": protocol_sha256,
        "stage": stage,
        "input_manifests": [reference.to_dict() for reference in references],
        "implementation_identity": implementation_identity,
        "source_snapshot": source_snapshot,
        "evaluation_population": dict(APPROVED_EVALUATION_POPULATION),
        "rows": list(normalized_rows),
    }
    document = {**body, "sha256": _canonical_sha256(body)}
    _validate_canonical_execution_manifest(
        path=path,
        stage=stage,
        rows=normalized_rows,
    )
    if build_implementation_identity() != implementation_identity:
        raise RuntimeError("execution implementation changed while freezing manifest")
    _write_immutable(path, _canonical_bytes(document), mode=0o444)
    _BOUND_SOURCE_REPLAY_CACHE.add(
        _bound_source_replay_cache_key(
            path=path.resolve(strict=True),
            protocol_sha256=protocol_sha256,
            logical_sha256=document["sha256"],
            physical_sha256=_file_sha256(path),
            source_snapshot=(
                execution_project_root(path) / str(source_snapshot["path"])
            ).resolve(strict=True),
        )
    )
    load_execution_manifest(
        path,
        expected_protocol_sha256=protocol_sha256,
        validate_inputs=True,
    )
    return path


def native500m_input_manifest_references(
    *, root: Path = PROJECT_ROOT, validate_files: bool = True
) -> tuple[InputManifestReference, ...]:
    from experiments.g3_pretrained_item_embeddings.protocol.native500m.artifacts import (
        load_artifact_manifest,
        validate_content_manifest,
        validate_dataset_manifest,
        validate_feature_manifest,
    )

    root = root.resolve(strict=True)
    manifest_directory = (
        root / "experiments/g3_pretrained_item_embeddings/protocol/native500m"
    )
    validators = {
        "content": validate_content_manifest,
        "dataset": validate_dataset_manifest,
        "features": validate_feature_manifest,
    }
    references = []
    for role, filename in (
        ("content", "content_manifest.json"),
        ("dataset", "dataset_manifest.json"),
        ("features", "feature_manifest.json"),
    ):
        path = manifest_directory / filename
        artifact_manifest = validators[role](
            load_artifact_manifest(path),
            root=root,
            validate_files=validate_files,
            validate_semantics=False,
        )
        references.append(
            InputManifestReference.from_path(
                role=role,
                root=root,
                path=path,
                logical_sha256=artifact_manifest.sha256,
            )
        )
    return tuple(references)


def materialize_baseline_execution_manifest(
    *,
    output_directory: Path | None = None,
) -> Path:
    from experiments.g3_pretrained_item_embeddings.configs.model import (
        G3Representation,
    )
    from experiments.g3_pretrained_item_embeddings.protocol.native500m.compiler import (
        compile_baseline_rows,
    )
    from experiments.g3_pretrained_item_embeddings.protocol.native500m.constants import (
        PROTOCOL_SHA256,
    )

    rows = compile_baseline_rows()
    representation = G3Representation(item_id_tying="tied").to_dict()
    directory = (
        PROJECT_ROOT / "generated/g3-native500m/execution-manifests"
        if output_directory is None
        else output_directory
    )
    implementation_prefix = str(build_implementation_identity()["sha256"])[:12]
    path = directory / f"baseline-{PROTOCOL_SHA256[:16]}-i{implementation_prefix}.json"
    return freeze_execution_manifest(
        path,
        stage="baseline",
        rows=rows,
        input_manifests=native500m_input_manifest_references(),
        job_payloads={
            row.id: {
                "resolved_representation": representation,
                "predecessor_artifacts": [],
            }
            for row in rows
        },
        protocol_sha256=PROTOCOL_SHA256,
    )


def build_implementation_identity() -> dict[str, object]:
    files = []
    for relative in _implementation_paths():
        path = _regular_project_path(relative)
        files.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
        )
    body = {"schema_version": 1, "files": files}
    return {**body, "sha256": _canonical_sha256(body)}


def validate_current_source_ledger(value: object) -> dict[str, object]:
    identity = _validated_implementation_identity(value)
    current = build_implementation_identity()
    if identity != current:
        raise ValueError("execution implementation drifted from current source")
    return current


def _validated_implementation_identity(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "files",
        "sha256",
    }:
        raise ValueError("execution implementation identity schema differs")
    files = value.get("files")
    if value.get("schema_version") != 1 or not isinstance(files, list) or not files:
        raise ValueError("execution implementation identity schema differs")
    paths = []
    for item in files:
        if not isinstance(item, dict) or set(item) != {
            "path",
            "size_bytes",
            "sha256",
        }:
            raise ValueError("execution implementation identity file schema differs")
        relative = item["path"]
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or type(item["size_bytes"]) is not int
            or item["size_bytes"] < 0
        ):
            raise ValueError("execution implementation identity file differs")
        _validate_sha256(item["sha256"], "implementation file SHA-256")
        paths.append(relative)
    if paths != sorted(set(paths)):
        raise ValueError("execution implementation identity paths differ")
    _validate_sha256(value["sha256"], "implementation identity SHA-256")
    body = {key: item for key, item in value.items() if key != "sha256"}
    if value["sha256"] != _canonical_sha256(body):
        raise ValueError("execution implementation identity SHA-256 differs")
    return _json_round_trip(value)


def _materialize_source_snapshot(
    *, execution_manifest_path: Path, implementation_identity: Mapping[str, object]
) -> dict[str, object]:
    identity = _validated_implementation_identity(dict(implementation_identity))
    repository_root = execution_project_root(execution_manifest_path)
    relative = f"{_SOURCE_SNAPSHOT_DIRECTORY}/{identity['sha256']}"
    snapshot = repository_root / relative
    _reject_symlink_ancestors(snapshot)
    expected_bytes = _source_bytes_for_identity(identity)
    marker_body = {
        "schema_version": 1,
        "kind": "g3_native500m_source_snapshot",
        "implementation_identity": identity,
    }
    marker_document = {**marker_body, "sha256": _canonical_sha256(marker_body)}
    marker_bytes = _canonical_bytes(marker_document)
    reference = {
        "path": relative,
        "implementation_identity_sha256": identity["sha256"],
        "manifest_size_bytes": len(marker_bytes),
        "manifest_sha256": hashlib.sha256(marker_bytes).hexdigest(),
        "manifest_logical_sha256": marker_document["sha256"],
    }
    if snapshot.exists():
        _validate_source_snapshot(execution_manifest_path, reference, identity)
        if build_implementation_identity() != identity:
            raise RuntimeError("execution implementation changed while snapshotting")
        return reference
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_ancestors(snapshot.parent)
    temporary = Path(mkdtemp(prefix=f".{identity['sha256']}.", dir=snapshot.parent))
    try:
        for relative_path, content in expected_bytes.items():
            destination = temporary / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
            destination.chmod(0o444)
        marker = temporary / _SOURCE_SNAPSHOT_MANIFEST
        marker.write_bytes(marker_bytes)
        marker.chmod(0o444)
        for directory in sorted(
            (path for path in temporary.rglob("*") if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            directory.chmod(0o555)
        temporary.chmod(0o555)
        try:
            temporary.rename(snapshot)
        except OSError:
            if not snapshot.exists():
                raise
            _remove_snapshot_temporary(temporary)
        _validate_source_snapshot(execution_manifest_path, reference, identity)
    except BaseException:
        if temporary.exists():
            _remove_snapshot_temporary(temporary)
        raise
    if build_implementation_identity() != identity:
        raise RuntimeError("execution implementation changed while snapshotting")
    return reference


def _source_bytes_for_identity(
    identity: Mapping[str, object],
) -> dict[str, bytes]:
    contents = {}
    for item in identity["files"]:
        relative = str(item["path"])
        source = _regular_project_path(relative)
        before = source.stat()
        content = source.read_bytes()
        after = source.stat()
        if (
            source.is_symlink()
            or before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or before.st_ctime_ns != after.st_ctime_ns
            or len(content) != item["size_bytes"]
            or hashlib.sha256(content).hexdigest() != item["sha256"]
        ):
            raise RuntimeError(
                f"execution implementation changed while snapshotting: {relative}"
            )
        contents[relative] = content
    return contents


def _remove_snapshot_temporary(path: Path) -> None:
    for directory in [path, *(item for item in path.rglob("*") if item.is_dir())]:
        directory.chmod(0o755)
    shutil.rmtree(path)


def _validate_source_snapshot(
    execution_manifest_path: Path,
    value: object,
    implementation_identity: Mapping[str, object],
) -> Path:
    identity = _validated_implementation_identity(dict(implementation_identity))
    expected_path = f"{_SOURCE_SNAPSHOT_DIRECTORY}/{identity['sha256']}"
    expected_keys = {
        "path",
        "implementation_identity_sha256",
        "manifest_size_bytes",
        "manifest_sha256",
        "manifest_logical_sha256",
    }
    if (
        not isinstance(value, dict)
        or set(value) != expected_keys
        or value.get("path") != expected_path
        or value.get("implementation_identity_sha256") != identity["sha256"]
        or type(value.get("manifest_size_bytes")) is not int
        or value["manifest_size_bytes"] < 1
    ):
        raise ValueError("execution source snapshot reference differs")
    _validate_sha256(value["manifest_sha256"], "source snapshot manifest SHA-256")
    _validate_sha256(
        value["manifest_logical_sha256"],
        "source snapshot manifest logical SHA-256",
    )
    repository_root = execution_project_root(execution_manifest_path)
    snapshot = repository_root / expected_path
    _reject_symlink_ancestors(snapshot)
    if snapshot.is_symlink() or not snapshot.is_dir():
        raise ValueError("execution source snapshot is absent or not regular")
    marker = snapshot / _SOURCE_SNAPSHOT_MANIFEST
    if (
        marker.is_symlink()
        or not marker.is_file()
        or marker.stat().st_size != value["manifest_size_bytes"]
        or _file_sha256(marker) != value["manifest_sha256"]
    ):
        raise ValueError("execution source snapshot manifest differs")
    marker_document = _load_json(marker)
    marker_body = {
        key: item for key, item in marker_document.items() if key != "sha256"
    }
    if (
        set(marker_document)
        != {"schema_version", "kind", "implementation_identity", "sha256"}
        or marker_document.get("schema_version") != 1
        or marker_document.get("kind") != "g3_native500m_source_snapshot"
        or marker_document.get("implementation_identity") != identity
        or marker_document.get("sha256") != _canonical_sha256(marker_body)
        or marker_document.get("sha256") != value["manifest_logical_sha256"]
    ):
        raise ValueError("execution source snapshot manifest identity differs")
    expected_files = {_SOURCE_SNAPSHOT_MANIFEST}
    expected_directories = set()
    for item in identity["files"]:
        relative = str(item["path"])
        source = snapshot / relative
        if (
            source.is_symlink()
            or not source.is_file()
            or source.stat().st_size != item["size_bytes"]
            or _file_sha256(source) != item["sha256"]
        ):
            raise ValueError(f"execution source snapshot file differs: {relative}")
        expected_files.add(relative)
        parent = Path(relative).parent
        while parent != Path("."):
            expected_directories.add(parent.as_posix())
            parent = parent.parent
    actual_files = set()
    actual_directories = set()
    for source in snapshot.rglob("*"):
        if source.is_symlink():
            raise ValueError("execution source snapshot contains a symlink")
        if source.is_file():
            actual_files.add(source.relative_to(snapshot).as_posix())
        elif source.is_dir():
            actual_directories.add(source.relative_to(snapshot).as_posix())
        else:
            raise ValueError("execution source snapshot contains an irregular entry")
    if actual_files != expected_files or actual_directories != expected_directories:
        raise ValueError("execution source snapshot file set differs")
    return snapshot.resolve(strict=True)


def materialize_selected_execution_manifest(
    path: Path,
    *,
    family_id: str,
    predecessor_selection_path: Path | None,
    root: Path = PROJECT_ROOT,
) -> Path:
    from experiments.g3_pretrained_item_embeddings.analysis.native500m_evidence import (
        authenticate_family_selection,
    )
    from experiments.g3_pretrained_item_embeddings.protocol.native500m import (
        compile_capacity_first_stage,
        compile_nine_cell_family,
        compile_rq5_frequency_first_stage,
        compile_rq5_global_rows,
        family_spec,
    )

    selection, authenticated_predecessor = authenticate_family_selection(
        predecessor_selection_path, root=root
    )
    spec = family_spec(family_id)
    if spec.conditional:
        raise ValueError("conditional family requires a resolved aggregate manifest")
    winner = selection["winner"]
    winner_job = winner["job"]
    if winner_job.get("family_id") != spec.search_predecessor_id:
        raise ValueError("selected evidence has the wrong predecessor role")
    predecessor = authenticated_predecessor.coordinate
    if spec.design == "nine_cell":
        rows = compile_nine_cell_family(spec, predecessor)
    elif spec.design == "capacity":
        rows = compile_capacity_first_stage(spec, predecessor)
    elif spec.design == "rq5_global":
        rows = compile_rq5_global_rows(predecessor)
    elif spec.design == "rq5_frequency":
        rows = compile_rq5_frequency_first_stage(predecessor)
    else:
        raise ValueError("selected family design is unsupported")
    predecessor_reference = _family_selection_reference(
        root=root,
        path=predecessor_selection_path,
        document=selection,
        row_id=str(winner["row_id"]),
    )
    payloads = {
        row.id: {
            "resolved_representation": _representation_for_family(
                family_id,
                capacity=row.capacity,
                predecessor=winner_job["resolved_representation"],
            ),
            "predecessor_artifacts": [predecessor_reference],
        }
        for row in rows
    }
    return freeze_execution_manifest(
        path,
        stage=f"{family_id}_initial",
        rows=rows,
        input_manifests=native500m_input_manifest_references(root=root),
        job_payloads=payloads,
    )


def materialize_continuation_execution_manifest(
    path: Path,
    *,
    family_id: str,
    continuation: str,
    evidence_paths: tuple[Path, ...],
    predecessor_selection_path: Path | None,
    root: Path = PROJECT_ROOT,
) -> Path:
    from experiments.g3_pretrained_item_embeddings.analysis.native500m_evidence import (
        derive_continuation_authorization,
        persist_continuation_authorization,
    )

    authorization, rows, predecessor_representation = derive_continuation_authorization(
        family_id=family_id,
        continuation=continuation,
        evidence_paths=evidence_paths,
        predecessor_selection_path=predecessor_selection_path,
        root=root,
    )
    authorization_path = path.with_name(f"{path.stem}.authorization.json")
    persist_continuation_authorization(authorization_path, authorization)
    authorization_reference = _bound_document_reference(
        root=root,
        path=authorization_path,
        document=authorization,
        role="continuation_authorization",
        row_id=str(authorization["selected_row_id"]),
    )
    predecessor_reference = None
    if predecessor_selection_path is not None:
        predecessor_document = authenticate_family_selection_document(
            predecessor_selection_path, root=root
        )
        predecessor_reference = _family_selection_reference(
            root=root,
            path=predecessor_selection_path,
            document=predecessor_document,
            row_id=str(predecessor_document["winner"]["row_id"]),
        )
    payloads = {
        row.id: {
            "resolved_representation": _representation_for_family(
                family_id,
                capacity=row.capacity,
                predecessor=predecessor_representation,
            ),
            "predecessor_artifacts": [
                *([] if predecessor_reference is None else [predecessor_reference]),
                authorization_reference,
            ],
        }
        for row in rows
    }
    return freeze_execution_manifest(
        path,
        stage=f"{family_id}_{continuation}",
        rows=rows,
        input_manifests=native500m_input_manifest_references(root=root),
        job_payloads=payloads,
    )


def materialize_conditional_execution_manifest(
    path: Path,
    *,
    family_id: str,
    compatibility_state_path: Path,
    root: Path = PROJECT_ROOT,
) -> Path:
    from experiments.g3_pretrained_item_embeddings.analysis.native500m_evidence import (
        authenticate_compatibility_resolution,
        authenticate_family_selection,
    )
    from experiments.g3_pretrained_item_embeddings.protocol.native500m import (
        authenticate_resolved_conditional_predecessor,
        compile_nine_cell_family,
        family_spec,
    )

    state, authenticated_state = authenticate_compatibility_resolution(
        compatibility_state_path, root=root
    )
    if state["next_conditional_family"] != family_id:
        raise ValueError("compatibility state does not authorize this family")
    resolved = authenticate_resolved_conditional_predecessor(
        target_family_id=family_id, compatibility_state=authenticated_state
    )
    spec = family_spec(family_id)
    rows = compile_nine_cell_family(spec, resolved.coordinate)
    most_reference = state["most_specific_selection"]
    most_path = root / str(most_reference["path"])
    most_document = authenticate_family_selection(most_path, root=root)[0]
    predecessor_reference = _family_selection_reference(
        root=root,
        path=most_path,
        document=most_document,
        row_id=str(most_reference["row_id"]),
    )
    state_reference = _authenticated_state_reference(
        root=root,
        document=state,
        authenticated=authenticated_state,
        row_id=str(most_reference["row_id"]),
    )
    representation = _conditional_representation(family_id, state=state, root=root)
    payloads = {
        row.id: {
            "resolved_representation": representation,
            "predecessor_artifacts": [predecessor_reference, state_reference],
        }
        for row in rows
    }
    return freeze_execution_manifest(
        path,
        stage=f"{family_id}_initial",
        rows=rows,
        input_manifests=native500m_input_manifest_references(root=root),
        job_payloads=payloads,
    )


def materialize_conditional_boundary_execution_manifest(
    path: Path,
    *,
    family_id: str,
    evidence_paths: tuple[Path, ...],
    compatibility_state_path: Path,
    root: Path = PROJECT_ROOT,
) -> Path:
    from experiments.g3_pretrained_item_embeddings.analysis.native500m_evidence import (
        authenticate_compatibility_resolution,
        authenticate_family_selection,
        derive_conditional_boundary_authorization,
        persist_conditional_boundary_authorization,
    )

    authorization, rows, _ = derive_conditional_boundary_authorization(
        family_id=family_id,
        evidence_paths=evidence_paths,
        compatibility_state_path=compatibility_state_path,
        root=root,
    )
    authorization_path = path.with_name(f"{path.stem}.authorization.json")
    persist_conditional_boundary_authorization(authorization_path, authorization)
    state, authenticated_state = authenticate_compatibility_resolution(
        compatibility_state_path, root=root
    )
    if authorization["compatibility_state"] != _selection_file_reference_for_state(
        root=root,
        document=state,
        authenticated=authenticated_state,
        role="compatibility_state",
    ):
        raise ValueError("conditional boundary compatibility state changed")
    most_reference = state["most_specific_selection"]
    most_path = root / str(most_reference["path"])
    most_document = authenticate_family_selection(most_path, root=root)[0]
    predecessor_reference = _family_selection_reference(
        root=root,
        path=most_path,
        document=most_document,
        row_id=str(most_reference["row_id"]),
    )
    state_reference = _authenticated_state_reference(
        root=root,
        document=state,
        authenticated=authenticated_state,
        row_id=str(most_reference["row_id"]),
    )
    authorization_reference = _bound_document_reference(
        root=root,
        path=authorization_path,
        document=authorization,
        role="conditional_boundary_authorization",
        row_id=str(authorization["selected_row_id"]),
    )
    representation = _conditional_representation(family_id, state=state, root=root)
    payloads = {
        row.id: {
            "resolved_representation": representation,
            "predecessor_artifacts": [
                predecessor_reference,
                state_reference,
                authorization_reference,
            ],
        }
        for row in rows
    }
    return freeze_execution_manifest(
        path,
        stage=f"{family_id}_boundary",
        rows=rows,
        input_manifests=native500m_input_manifest_references(root=root),
        job_payloads=payloads,
    )


def authenticate_family_selection_document(
    path: Path, *, root: Path
) -> dict[str, object]:
    from experiments.g3_pretrained_item_embeddings.analysis.native500m_evidence import (
        authenticate_family_selection,
    )

    return authenticate_family_selection(path, root=root)[0]


def load_execution_manifest(
    path: Path,
    *,
    expected_protocol_sha256: str | None = None,
    expected_logical_sha256: str | None = None,
    expected_physical_sha256: str | None = None,
    validate_inputs: bool = True,
) -> ExecutionManifest:
    _reject_symlink_ancestors(path)
    path = path.resolve(strict=True)
    if path.is_symlink() or not path.is_file():
        raise ValueError("execution manifest must be a regular file")
    physical_sha256 = _file_sha256(path)
    if (
        expected_physical_sha256 is not None
        and physical_sha256 != expected_physical_sha256
    ):
        raise ValueError("execution manifest physical SHA-256 differs")
    document = _load_json(path)
    expected_keys = {
        "schema_version",
        "kind",
        "protocol_sha256",
        "stage",
        "input_manifests",
        "implementation_identity",
        "source_snapshot",
        "evaluation_population",
        "rows",
        "sha256",
    }
    if set(document) != expected_keys:
        raise ValueError("execution manifest schema differs")
    protocol_sha256 = _protocol_sha256(expected_protocol_sha256)
    if (
        document["schema_version"] != 2
        or document["kind"] != "g3_native500m_execution_manifest"
        or document["protocol_sha256"] != protocol_sha256
    ):
        raise ValueError("execution manifest protocol identity differs")
    logical_sha256 = str(document["sha256"])
    body = {key: value for key, value in document.items() if key != "sha256"}
    if logical_sha256 != _canonical_sha256(body):
        raise ValueError("execution manifest logical SHA-256 differs")
    if (
        expected_logical_sha256 is not None
        and logical_sha256 != expected_logical_sha256
    ):
        raise ValueError("execution manifest logical SHA-256 differs")
    stage = document["stage"]
    raw_rows = document["rows"]
    raw_references = document["input_manifests"]
    if not isinstance(stage, str) or not stage or not isinstance(raw_rows, list):
        raise ValueError("execution manifest stage or rows are invalid")
    implementation_prefix = str(document["implementation_identity"].get("sha256", ""))[
        :12
    ]
    rows = tuple(
        _loaded_execution_row(row, implementation_prefix=implementation_prefix)
        for row in raw_rows
    )
    if not rows:
        raise ValueError("execution manifest has no rows")
    row_ids = [str(row["id"]) for row in rows]
    run_names = [str(row["job"]["run_name"]) for row in rows]
    if len(set(row_ids)) != len(row_ids) or len(set(run_names)) != len(run_names):
        raise ValueError("execution manifest row IDs and run names are not unique")
    if not isinstance(raw_references, list):
        raise ValueError("execution manifest input bindings are invalid")
    references = tuple(_input_reference(value) for value in raw_references)
    if tuple(reference.role for reference in references) != _INPUT_ROLES:
        raise ValueError("execution manifest input binding roles differ")
    source_snapshot = _validate_source_snapshot(
        path,
        document["source_snapshot"],
        document["implementation_identity"],
    )
    if validate_inputs:
        _validate_evaluation_population(document["evaluation_population"])
        for reference in references:
            _validate_input_reference(path, reference)
    from experiments.g3_pretrained_item_embeddings.analysis.native500m_evidence import (
        _authentication_scope,
    )

    with _authentication_scope():
        _validate_bound_source_semantics(
            path=path,
            protocol_sha256=protocol_sha256,
            logical_sha256=logical_sha256,
            physical_sha256=physical_sha256,
            source_snapshot=source_snapshot,
            stage=stage,
            rows=rows,
        )
    return ExecutionManifest(
        path=path,
        stage=stage,
        protocol_sha256=protocol_sha256,
        rows=rows,
        input_manifests=references,
        implementation_identity=dict(document["implementation_identity"]),
        source_snapshot=source_snapshot,
        evaluation_population=dict(document["evaluation_population"]),
        logical_sha256=logical_sha256,
        physical_sha256=physical_sha256,
    )


def replay_bound_execution_manifest(
    path: Path,
    *,
    expected_protocol_sha256: str,
    expected_logical_sha256: str,
    expected_physical_sha256: str,
    validate_inputs: bool,
) -> str:
    manifest = load_execution_manifest(
        path,
        expected_protocol_sha256=expected_protocol_sha256,
        expected_logical_sha256=expected_logical_sha256,
        expected_physical_sha256=expected_physical_sha256,
        validate_inputs=validate_inputs,
    )
    if PROJECT_ROOT.resolve(strict=True) != manifest.source_snapshot:
        raise ValueError("bound source semantic replay did not import its snapshot")
    return _bound_source_replay_token(
        protocol_sha256=manifest.protocol_sha256,
        logical_sha256=manifest.logical_sha256,
        physical_sha256=manifest.physical_sha256,
    )


def _validate_bound_source_semantics(
    *,
    path: Path,
    protocol_sha256: str,
    logical_sha256: str,
    physical_sha256: str,
    source_snapshot: Path,
    stage: str,
    rows: tuple[dict[str, object], ...],
) -> None:
    if PROJECT_ROOT.resolve(strict=True) == source_snapshot:
        _validate_canonical_execution_manifest(path=path, stage=stage, rows=rows)
        return
    cache_key = _bound_source_replay_cache_key(
        path=path,
        protocol_sha256=protocol_sha256,
        logical_sha256=logical_sha256,
        physical_sha256=physical_sha256,
        source_snapshot=source_snapshot,
    )
    if cache_key in _BOUND_SOURCE_REPLAY_CACHE:
        return
    program = (
        "from pathlib import Path; import sys; "
        "from experiments.g3_pretrained_item_embeddings.launchers.native500m "
        "import replay_bound_execution_manifest; "
        "print(replay_bound_execution_manifest(Path(sys.argv[1]), "
        "expected_protocol_sha256=sys.argv[2], "
        "expected_logical_sha256=sys.argv[3], "
        "expected_physical_sha256=sys.argv[4], "
        "validate_inputs=sys.argv[5] == '1'))"
    )
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONSAFEPATH": "1",
            "PYTHONPATH": str(source_snapshot),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                program,
                str(path),
                protocol_sha256,
                logical_sha256,
                physical_sha256,
                "0",
            ],
            cwd=execution_project_root(path),
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ValueError("bound source semantic replay failed") from error
    expected = _bound_source_replay_token(
        protocol_sha256=protocol_sha256,
        logical_sha256=logical_sha256,
        physical_sha256=physical_sha256,
    )
    if completed.returncode != 0 or completed.stdout != f"{expected}\n":
        raise ValueError("bound source semantic replay failed")
    _BOUND_SOURCE_REPLAY_CACHE.add(cache_key)


def _bound_source_replay_cache_key(
    *,
    path: Path,
    protocol_sha256: str,
    logical_sha256: str,
    physical_sha256: str,
    source_snapshot: Path,
) -> tuple[str, ...]:
    return (
        str(path),
        protocol_sha256,
        logical_sha256,
        physical_sha256,
        str(source_snapshot),
    )


def _bound_source_replay_token(
    *, protocol_sha256: str, logical_sha256: str, physical_sha256: str
) -> str:
    return "g3-native500m-source-replay-v1:" + _canonical_sha256(
        {
            "protocol_sha256": protocol_sha256,
            "logical_sha256": logical_sha256,
            "physical_sha256": physical_sha256,
        }
    )


def _validate_canonical_execution_manifest(
    *, path: Path, stage: str, rows: tuple[dict[str, object], ...]
) -> None:
    from experiments.g3_pretrained_item_embeddings.configs.model import G3Representation
    from experiments.g3_pretrained_item_embeddings.protocol.native500m import (
        authenticate_resolved_conditional_predecessor,
        compile_baseline_rows,
        compile_capacity_first_stage,
        compile_nine_cell_family,
        compile_rq5_frequency_first_stage,
        compile_rq5_global_rows,
        family_spec,
    )

    families = {str(row["job"]["family_id"]) for row in rows}
    if len(families) != 1:
        raise ValueError("execution manifest mixes compiler families")
    family_id = families.pop()
    spec = family_spec(family_id)
    predecessor_lists = [row["job"]["predecessor_artifacts"] for row in rows]
    if any(value != predecessor_lists[0] for value in predecessor_lists[1:]):
        raise ValueError("execution manifest rows bind different predecessors")
    references = predecessor_lists[0]
    for reference in references:
        _validate_predecessor_identity(path, reference)
    roles = [str(reference["role"]) for reference in references]
    root = _execution_root(path)
    predecessor_representation: dict[str, object]
    expected_rows: Sequence[object]
    if family_id == "baseline":
        predecessor_representation = G3Representation(item_id_tying="tied").to_dict()
        if stage == "baseline":
            if roles:
                raise ValueError("baseline execution predecessor chain differs")
            expected_rows = compile_baseline_rows()
        elif stage == "baseline_boundary":
            if roles != ["continuation_authorization"]:
                raise ValueError("baseline boundary predecessor chain differs")
            from experiments.g3_pretrained_item_embeddings.analysis.native500m_evidence import (
                load_continuation_authorization,
            )

            authorization = load_continuation_authorization(
                _resolve_bound_path(path, str(references[0]["path"])), root=root
            )
            if (
                authorization["family_id"] != "baseline"
                or authorization["continuation"] != "boundary"
                or authorization["predecessor"] is not None
                or references[0]["row_id"] != authorization["selected_row_id"]
            ):
                raise ValueError("baseline boundary authorization differs")
            expected_rows = tuple(
                _search_row_from_mapping(value)
                for value in authorization["continuation_rows"]
            )
        else:
            raise ValueError("baseline execution manifest stage differs")
    elif spec.conditional:
        if stage not in {f"{family_id}_initial", f"{family_id}_boundary"}:
            raise ValueError("conditional execution manifest stage differs")
        expected_roles = ["search_predecessor", "compatibility_state"]
        if stage.endswith("_boundary"):
            expected_roles.append("conditional_boundary_authorization")
        if roles != expected_roles:
            raise ValueError("conditional execution predecessor chain differs")
        state_path = _resolve_bound_path(path, str(references[1]["path"]))
        from experiments.g3_pretrained_item_embeddings.analysis.native500m_evidence import (
            authenticate_compatibility_resolution,
            load_conditional_boundary_authorization,
        )

        state, authenticated_state = authenticate_compatibility_resolution(
            state_path, root=root
        )
        if (
            not _same_bound_target(
                references[0], state["most_specific_selection"], require_row=True
            )
            or references[1]["row_id"] != state["most_specific_selection"]["row_id"]
        ):
            raise ValueError("conditional search predecessor differs from state")
        resolved = authenticate_resolved_conditional_predecessor(
            target_family_id=family_id, compatibility_state=authenticated_state
        )
        if stage.endswith("_initial"):
            expected_rows = compile_nine_cell_family(spec, resolved.coordinate)
        else:
            authorization_path = _resolve_bound_path(path, str(references[2]["path"]))
            authorization = load_conditional_boundary_authorization(
                authorization_path, root=root
            )
            if (
                authorization["family_id"] != family_id
                or references[2]["row_id"] != authorization["selected_row_id"]
                or authorization["compatibility_state"]
                != {
                    key: value
                    for key, value in references[1].items()
                    if key != "row_id"
                }
            ):
                raise ValueError("conditional boundary authorization differs")
            expected_rows = tuple(
                _search_row_from_mapping(value)
                for value in authorization["boundary_rows"]
            )
        predecessor_representation = _conditional_representation(
            family_id, state=state, root=root
        )
    else:
        initial_stage = f"{family_id}_initial"
        continuation = (
            None
            if stage == initial_stage
            else (
                "followup"
                if stage == f"{family_id}_followup"
                else "boundary" if stage == f"{family_id}_boundary" else None
            )
        )
        if stage != initial_stage and continuation is None:
            raise ValueError("standalone execution manifest stage differs")
        expected_roles = ["search_predecessor"]
        if continuation is not None:
            expected_roles.append("continuation_authorization")
        if roles != expected_roles:
            raise ValueError("standalone execution predecessor chain differs")
        from experiments.g3_pretrained_item_embeddings.analysis.native500m_evidence import (
            authenticate_family_selection,
            load_continuation_authorization,
        )

        predecessor_path = _resolve_bound_path(path, str(references[0]["path"]))
        predecessor_document, authenticated_predecessor = authenticate_family_selection(
            predecessor_path, root=root
        )
        if references[0]["row_id"] != predecessor_document["winner"]["row_id"]:
            raise ValueError("standalone search predecessor row differs")
        predecessor_representation = predecessor_document["winner"]["job"][
            "resolved_representation"
        ]
        predecessor = authenticated_predecessor.coordinate
        if continuation is None:
            if spec.design == "nine_cell":
                expected_rows = compile_nine_cell_family(spec, predecessor)
            elif spec.design == "capacity":
                expected_rows = compile_capacity_first_stage(spec, predecessor)
            elif spec.design == "rq5_global":
                expected_rows = compile_rq5_global_rows(predecessor)
            elif spec.design == "rq5_frequency":
                expected_rows = compile_rq5_frequency_first_stage(predecessor)
            else:
                raise ValueError("standalone execution family design differs")
        else:
            authorization_path = _resolve_bound_path(path, str(references[1]["path"]))
            authorization = load_continuation_authorization(
                authorization_path, root=root
            )
            if (
                authorization["family_id"] != family_id
                or authorization["continuation"] != continuation
                or references[-1]["row_id"] != authorization["selected_row_id"]
            ):
                raise ValueError("continuation execution authorization differs")
            authorized_predecessor = authorization["predecessor"]
            if (authorized_predecessor is None) != (family_id == "baseline"):
                raise ValueError("continuation predecessor authorization differs")
            if authorized_predecessor is not None and not _same_bound_target(
                references[0], authorized_predecessor, require_row=False
            ):
                raise ValueError("continuation search predecessor differs")
            expected_rows = tuple(
                _search_row_from_mapping(value)
                for value in authorization["continuation_rows"]
            )
    if [row.id for row in expected_rows] != [str(row["id"]) for row in rows]:
        raise ValueError("execution manifest rows are not the exact canonical ledger")
    for expected, actual in zip(expected_rows, rows, strict=True):
        expected_job = expected.to_dict()
        actual_job = actual["job"]
        if any(actual_job.get(key) != value for key, value in expected_job.items()):
            raise ValueError("execution manifest row differs from compiler ledger")
        expected_representation = (
            predecessor_representation
            if family_id == "baseline" or spec.conditional
            else _representation_for_family(
                family_id,
                capacity=expected.capacity,
                predecessor=predecessor_representation,
            )
        )
        if actual_job["resolved_representation"] != expected_representation:
            raise ValueError(
                "execution manifest representation differs from canonical state"
            )


def _search_row_from_mapping(value: object):
    from experiments.g3_pretrained_item_embeddings.protocol.native500m import SearchRow

    if not isinstance(value, Mapping):
        raise ValueError("authorized execution row is invalid")
    return SearchRow(
        id=str(value["id"]),
        family_id=str(value["family_id"]),
        family_code=int(value["family_code"]),
        research_question=str(value["research_question"]),
        predecessor_id=str(value["predecessor_id"]),
        promotion_predecessor_id=str(value["promotion_predecessor_id"]),
        manifest_order=int(value["manifest_order"]),
        stage=value["stage"],
        batch_size=int(value["batch_size"]),
        seed=int(value["seed"]),
        horizon_epochs=int(value["horizon_epochs"]),
        embedding_learning_rate_text=str(value["embedding_learning_rate"]),
        deep_learning_rate_text=str(value["deep_learning_rate"]),
        anchor_embedding_learning_rate_text=str(
            value["anchor_embedding_learning_rate"]
        ),
        anchor_deep_learning_rate_text=str(value["anchor_deep_learning_rate"]),
        capacity=value["capacity"],
    )


def execution_project_root(path: Path) -> Path:
    absolute = path.absolute()
    parts = absolute.parts
    matches = [
        index
        for index in range(len(parts) - 1)
        if parts[index : index + 2] == ("generated", "g3-native500m")
    ]
    if len(matches) > 1:
        raise ValueError("execution manifest repository root is ambiguous")
    if matches:
        return Path(*parts[: matches[0]])
    return absolute.parent


def _reject_symlink_ancestors(path: Path) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"path traverses a symlink: {path}")


def _execution_root(path: Path) -> Path:
    return execution_project_root(path)


def _same_bound_target(
    actual: Mapping[str, object],
    expected: object,
    *,
    require_row: bool,
) -> bool:
    if not isinstance(expected, Mapping):
        return False
    keys = {"path", "size_bytes", "sha256", "logical_sha256"}
    if require_row:
        keys.add("row_id")
    return all(actual.get(key) == expected.get(key) for key in keys)


def build_batch_specification(
    manifest_path: Path,
    *,
    expected_protocol_sha256: str | None = None,
    runner_script: Path | None = None,
) -> BatchSpecification:
    if runner_script is not None:
        raise ValueError("native-500M execution runner is fixed by its source snapshot")
    manifest = load_execution_manifest(
        manifest_path,
        expected_protocol_sha256=expected_protocol_sha256,
        validate_inputs=True,
    )
    runner_script = (
        manifest.source_snapshot
        / "experiments/g3_pretrained_item_embeddings/launchers/run_native500m.py"
    )
    if runner_script.is_symlink() or not runner_script.is_file():
        raise ValueError("native-500M source snapshot runner differs")
    snapshot_manifest_sha256 = _file_sha256(
        manifest.source_snapshot / _SOURCE_SNAPSHOT_MANIFEST
    )
    jobs = []
    for row in manifest.rows:
        payload = {
            "manifest_logical_sha256": manifest.logical_sha256,
            "manifest_physical_sha256": manifest.physical_sha256,
            "row_id": row["id"],
            "job": row["job"],
        }
        encoded = base64.urlsafe_b64encode(_canonical_bytes(payload)).decode()
        jobs.append(
            {
                "script": str(runner_script),
                "run": row["job"]["run_name"],
                "data_group": DATA_GROUP,
                "environment": [
                    f"{JOB_ENVIRONMENT}={encoded}",
                    f"{MANIFEST_ENVIRONMENT}={manifest.path}",
                    (
                        f"{MANIFEST_LOGICAL_SHA256_ENVIRONMENT}="
                        f"{manifest.logical_sha256}"
                    ),
                    (
                        f"{MANIFEST_PHYSICAL_SHA256_ENVIRONMENT}="
                        f"{manifest.physical_sha256}"
                    ),
                    "WANDB_MODE=offline",
                    "PYTHONSAFEPATH=1",
                    f"PYTHONPATH={manifest.source_snapshot}",
                    "PYTHONDONTWRITEBYTECODE=1",
                    f"{SOURCE_SNAPSHOT_ROOT_ENVIRONMENT}={manifest.source_snapshot}",
                    (
                        f"{SOURCE_SNAPSHOT_IDENTITY_ENVIRONMENT}="
                        f"{manifest.implementation_identity['sha256']}"
                    ),
                    (
                        f"{SOURCE_SNAPSHOT_MANIFEST_ENVIRONMENT}="
                        f"{snapshot_manifest_sha256}"
                    ),
                    *CPU_THREAD_ENVIRONMENT,
                ],
            }
        )
    document: dict[str, object] = {"version": 1, "jobs": jobs}
    return BatchSpecification(
        document=document,
        sha256=_canonical_sha256(document),
        manifest_logical_sha256=manifest.logical_sha256,
        manifest_physical_sha256=manifest.physical_sha256,
    )


def persist_batch_specification(
    directory: Path, specification: BatchSpecification
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{specification.sha256}.json"
    content = _canonical_bytes(specification.document)
    if hashlib.sha256(content).hexdigest() != specification.sha256:
        raise ValueError("batch specification SHA-256 differs")
    _write_immutable(path, content, mode=0o444)
    return path


def submit_execution_manifest(
    *,
    manifest_path: Path,
    state_directory: Path,
    specification_directory: Path | None = None,
    expected_protocol_sha256: str | None = None,
    existing_only: bool = False,
    dry_run: bool = False,
) -> str:
    if dry_run:
        specification = build_batch_specification(
            manifest_path,
            expected_protocol_sha256=expected_protocol_sha256,
        )
    else:
        manifest = load_execution_manifest(
            manifest_path,
            expected_protocol_sha256=expected_protocol_sha256,
            validate_inputs=True,
        )
        repository_root = execution_project_root(manifest.path)
        queue_service_identity = authenticate_training_queue_service(
            state_directory=state_directory,
            implementation_identity=manifest.implementation_identity,
            expected_project_root=repository_root,
        )
        specification = build_batch_specification(
            manifest_path,
            expected_protocol_sha256=manifest.protocol_sha256,
        )
    if dry_run:
        return _canonical_bytes(specification.document).decode()
    specification_path = persist_batch_specification(
        (
            execution_project_root(manifest.path)
            / "generated/g3-native500m/batch-specifications"
            if specification_directory is None
            else specification_directory
        ),
        specification,
    )
    if (
        authenticate_training_queue_service(
            state_directory=state_directory,
            implementation_identity=manifest.implementation_identity,
            expected_project_root=repository_root,
        )
        != queue_service_identity
    ):
        raise RuntimeError("training queue service identity changed before submission")
    command = [
        sys.executable,
        str(repository_root / "utils/training_queue/service.py"),
        "--state-dir",
        str(state_directory.resolve()),
        "find-batch" if existing_only else "submit-batch",
    ]
    command.append(str(specification_path.resolve()))
    completed = subprocess.run(
        command,
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    if (
        authenticate_training_queue_service(
            state_directory=state_directory,
            implementation_identity=manifest.implementation_identity,
            expected_project_root=repository_root,
        )
        != queue_service_identity
    ):
        raise RuntimeError("training queue service identity changed during submission")
    batch_id = completed.stdout.strip()
    if not batch_id:
        raise RuntimeError("training queue returned an empty batch ID")
    if existing_only:
        load_queue_submission_binding(
            specification_path=specification_path,
            manifest=manifest,
            batch_id=batch_id,
        )
    else:
        persist_queue_submission_binding(
            specification_path=specification_path,
            manifest=manifest,
            batch_id=batch_id,
            queue_service_identity=queue_service_identity,
        )
    return batch_id


def persist_queue_submission_binding(
    *,
    specification_path: Path,
    manifest: ExecutionManifest,
    batch_id: str,
    queue_service_identity: Mapping[str, object],
) -> Path:
    identity = _validate_queue_service_identity(
        queue_service_identity,
        implementation_identity=manifest.implementation_identity,
        require_working_directory=True,
    )
    specification_path = specification_path.resolve(strict=True)
    specification_sha256 = _file_sha256(specification_path)
    body: dict[str, object] = {
        "schema_version": 1,
        "kind": "g3_native500m_queue_submission_binding",
        "batch_id": batch_id,
        "batch_specification": {
            "path": str(specification_path),
            "size_bytes": specification_path.stat().st_size,
            "sha256": specification_sha256,
        },
        "manifest_logical_sha256": manifest.logical_sha256,
        "manifest_physical_sha256": manifest.physical_sha256,
        "implementation_identity_sha256": manifest.implementation_identity["sha256"],
        "queue_service_identity": identity,
    }
    document = {**body, "sha256": _canonical_sha256(body)}
    path = specification_path.with_name(
        f"{specification_sha256}-q{str(identity['sha256'])[:16]}-{batch_id}.submission.json"
    )
    _write_immutable(path, _canonical_bytes(document), mode=0o444)
    return path


def load_queue_submission_binding(
    *,
    specification_path: Path,
    manifest: ExecutionManifest,
    batch_id: str,
) -> dict[str, object]:
    specification_path = specification_path.resolve(strict=True)
    specification_sha256 = _file_sha256(specification_path)
    matches = sorted(
        specification_path.parent.glob(
            f"{specification_sha256}-q*-{batch_id}.submission.json"
        )
    )
    if not matches:
        raise ValueError("queue submission binding is absent")
    if len(matches) != 1 or matches[0].is_symlink() or not matches[0].is_file():
        raise ValueError("queue submission binding is not unique and regular")
    document = _load_json(matches[0])
    body = {key: value for key, value in document.items() if key != "sha256"}
    expected_keys = {
        "schema_version",
        "kind",
        "batch_id",
        "batch_specification",
        "manifest_logical_sha256",
        "manifest_physical_sha256",
        "implementation_identity_sha256",
        "queue_service_identity",
        "sha256",
    }
    expected_specification = {
        "path": str(specification_path),
        "size_bytes": specification_path.stat().st_size,
        "sha256": specification_sha256,
    }
    if (
        set(document) != expected_keys
        or document.get("schema_version") != 1
        or document.get("kind") != "g3_native500m_queue_submission_binding"
        or document.get("batch_id") != batch_id
        or document.get("batch_specification") != expected_specification
        or document.get("manifest_logical_sha256") != manifest.logical_sha256
        or document.get("manifest_physical_sha256") != manifest.physical_sha256
        or document.get("implementation_identity_sha256")
        != manifest.implementation_identity["sha256"]
        or document.get("sha256") != _canonical_sha256(body)
        or _file_sha256(matches[0])
        != hashlib.sha256(_canonical_bytes(document)).hexdigest()
    ):
        raise ValueError("queue submission binding identity differs")
    _validate_queue_service_identity(
        document["queue_service_identity"],
        implementation_identity=manifest.implementation_identity,
    )
    return document


def authenticate_training_queue_service(
    *,
    state_directory: Path,
    implementation_identity: Mapping[str, object],
    expected_project_root: Path | None = None,
    proc_root: Path = Path("/proc"),
    status_loader: Any | None = None,
) -> dict[str, object]:
    identity = _validated_implementation_identity(dict(implementation_identity))
    state_directory = state_directory.resolve()
    queue_project_root = _queue_project_root(state_directory)
    if (
        expected_project_root is not None
        and queue_project_root.resolve(strict=True)
        != expected_project_root.resolve(strict=True)
    ):
        raise RuntimeError("training queue project root differs")
    source_facts = _authenticate_queue_runtime_sources(queue_project_root, identity)
    status = (
        _load_training_queue_status(state_directory)
        if status_loader is None
        else status_loader(state_directory)
    )
    if not isinstance(status, dict) or status.get("running") is not True:
        raise RuntimeError("training queue service is not running")
    pid = status.get("pid")
    start_ticks = status.get("pid_start_time")
    instance_token = status.get("instance_token")
    if (
        type(pid) is not int
        or pid <= 0
        or type(start_ticks) is not int
        or start_ticks <= 0
        or not isinstance(instance_token, str)
        or not instance_token
    ):
        raise RuntimeError("training queue service status identity is invalid")
    process_directory = proc_root / str(pid)
    try:
        stat_fields = (process_directory / "stat").read_text().rsplit(")", 1)[1].split()
        observed_start_ticks = int(stat_fields[19])
        command = [
            value.decode()
            for value in (process_directory / "cmdline").read_bytes().split(b"\0")
            if value
        ]
        working_directory = (process_directory / "cwd").resolve(strict=True)
        boot_time = next(
            int(line.split()[1])
            for line in (proc_root / "stat").read_text().splitlines()
            if line.startswith("btime ")
        )
        child_pids = [
            int(value)
            for value in (process_directory / "task" / str(pid) / "children")
            .read_text()
            .split()
        ]
        if len(child_pids) != 1:
            raise ValueError("queue daemon does not have exactly one scheduler child")
        scheduler_pid = child_pids[0]
        scheduler_directory = proc_root / str(scheduler_pid)
        scheduler_stat_fields = (
            (scheduler_directory / "stat").read_text().rsplit(")", 1)[1].split()
        )
        scheduler_parent_pid = int(scheduler_stat_fields[1])
        scheduler_start_ticks = int(scheduler_stat_fields[19])
        scheduler_command = [
            value.decode()
            for value in (scheduler_directory / "cmdline").read_bytes().split(b"\0")
            if value
        ]
        scheduler_working_directory = (scheduler_directory / "cwd").resolve(
            strict=True
        )
    except (
        OSError,
        UnicodeDecodeError,
        ValueError,
        IndexError,
        StopIteration,
    ) as error:
        raise RuntimeError(
            "training queue service process identity is unavailable"
        ) from error
    service_script = str(
        _regular_path_below(queue_project_root, "utils/training_queue/service.py")
    )
    scheduler_script = str(
        _regular_path_below(
            queue_project_root, "utils/training_queue/service_scheduler.sh"
        )
    )
    expected_daemon_command = [
        command[0],
        service_script,
        "--state-dir",
        str(state_directory),
        "_serve",
        "--instance-token",
        instance_token,
    ]
    if (
        observed_start_ticks != start_ticks
        or Path(command[0]).resolve() != Path(sys.executable).resolve()
        or command != expected_daemon_command
        or not scheduler_command
        or Path(scheduler_command[0]).name != "bash"
        or scheduler_command[1:] != [scheduler_script, str(state_directory)]
        or scheduler_parent_pid != pid
        or scheduler_start_ticks < start_ticks
        or working_directory != queue_project_root
        or scheduler_working_directory != queue_project_root
    ):
        raise RuntimeError("training queue service process identity differs")
    clock_ticks = os.sysconf("SC_CLK_TCK")
    started_at = boot_time + start_ticks / clock_ticks
    scheduler_started_at = boot_time + scheduler_start_ticks / clock_ticks
    sources = []
    for relative, path, fact, item in source_facts:
        current = path.stat()
        if (
            current.st_dev != fact["device"]
            or current.st_ino != fact["inode"]
            or current.st_size != fact["size_bytes"]
            or current.st_mtime_ns != fact["mtime_ns"]
            or current.st_ctime_ns != fact["ctime_ns"]
            or _file_sha256(path) != fact["sha256"]
        ):
            raise RuntimeError(f"training queue source changed: {relative}")
        if current.st_mtime > started_at:
            raise RuntimeError(
                f"training queue service predates its source: {relative}"
            )
        if current.st_mtime > scheduler_started_at:
            raise RuntimeError(
                f"training queue scheduler predates its source: {relative}"
            )
        sources.append(dict(item))
    body: dict[str, object] = {
        "schema_version": 2,
        "pid": pid,
        "pid_start_time_ticks": start_ticks,
        "boot_time_unix_seconds": boot_time,
        "clock_ticks_per_second": clock_ticks,
        "instance_token": instance_token,
        "state_directory": str(state_directory),
        "working_directory": str(working_directory),
        "cmdline": command,
        "scheduler_pid": scheduler_pid,
        "scheduler_start_time_ticks": scheduler_start_ticks,
        "scheduler_cmdline": scheduler_command,
        "scheduler_working_directory": str(scheduler_working_directory),
        "sources": sources,
    }
    return {**body, "sha256": _canonical_sha256(body)}


def _authenticate_queue_runtime_sources(
    project_root: Path, implementation_identity: Mapping[str, object]
) -> list[tuple[str, Path, dict[str, object], Mapping[str, object]]]:
    ledger_files = {
        str(item["path"]): item
        for item in implementation_identity["files"]
        if isinstance(item, dict)
    }
    authenticated = []
    for relative in _QUEUE_RUNTIME_PATHS:
        path = _regular_path_below(project_root, relative)
        item = ledger_files.get(relative)
        before = path.stat()
        sha256 = _file_sha256(path)
        after = path.stat()
        if (
            item is None
            or before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or before.st_ctime_ns != after.st_ctime_ns
            or after.st_size != item["size_bytes"]
            or sha256 != item["sha256"]
        ):
            raise RuntimeError(f"training queue source differs from ledger: {relative}")
        authenticated.append(
            (
                relative,
                path,
                {
                    "device": after.st_dev,
                    "inode": after.st_ino,
                    "size_bytes": after.st_size,
                    "mtime_ns": after.st_mtime_ns,
                    "ctime_ns": after.st_ctime_ns,
                    "sha256": sha256,
                },
                item,
            )
        )
    return authenticated


def _load_training_queue_status(state_directory: Path) -> dict[str, object]:
    project_root = _queue_project_root(state_directory)
    completed = subprocess.run(
        [
            sys.executable,
            str(_regular_path_below(project_root, "utils/training_queue/service.py")),
            "--state-dir",
            str(state_directory),
            "status",
            "--json",
        ],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("training queue status is invalid") from error
    if not isinstance(value, dict):
        raise RuntimeError("training queue status is invalid")
    return value


def _queue_project_root(state_directory: Path) -> Path:
    resolved = state_directory.resolve()
    if (
        resolved.name == "training-queue-service"
        and resolved.parent.name == "generated"
    ):
        return resolved.parent.parent
    return PROJECT_ROOT


def _regular_path_below(root: Path, relative: str) -> Path:
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"project path escapes its root: {relative}")
    path = root.resolve(strict=True)
    for index, part in enumerate(relative_path.parts):
        path = path / part
        if path.is_symlink():
            raise ValueError(f"project path traverses a symlink: {relative}")
        if index < len(relative_path.parts) - 1 and not path.is_dir():
            raise ValueError(f"project path parent is absent: {relative}")
    if not path.is_file():
        raise ValueError(f"project file is absent: {relative}")
    return path


def _validate_queue_service_identity(
    value: Mapping[str, object],
    *,
    implementation_identity: Mapping[str, object],
    require_working_directory: bool = False,
) -> dict[str, object]:
    identity = _json_round_trip(value)
    if not isinstance(identity, dict):
        raise ValueError("queue service identity is invalid")
    body = {key: item for key, item in identity.items() if key != "sha256"}
    required = {
        "schema_version",
        "pid",
        "pid_start_time_ticks",
        "boot_time_unix_seconds",
        "clock_ticks_per_second",
        "instance_token",
        "state_directory",
        "cmdline",
        "scheduler_pid",
        "scheduler_start_time_ticks",
        "scheduler_cmdline",
        "sources",
        "sha256",
    }
    schema_version = identity.get("schema_version")
    if schema_version == 2:
        required |= {"working_directory", "scheduler_working_directory"}
    ledger_files = {
        str(item["path"]): item
        for item in implementation_identity.get("files", [])
        if isinstance(item, dict) and "path" in item
    }
    expected_sources = [ledger_files.get(path) for path in _QUEUE_RUNTIME_PATHS]
    if (
        set(identity) != required
        or schema_version not in ({2} if require_working_directory else {1, 2})
        or identity.get("sha256") != _canonical_sha256(body)
        or identity.get("sources") != expected_sources
        or type(identity.get("pid")) is not int
        or type(identity.get("pid_start_time_ticks")) is not int
        or type(identity.get("boot_time_unix_seconds")) is not int
        or type(identity.get("clock_ticks_per_second")) is not int
        or not isinstance(identity.get("instance_token"), str)
        or not isinstance(identity.get("state_directory"), str)
        or not isinstance(identity.get("cmdline"), list)
        or type(identity.get("scheduler_pid")) is not int
        or type(identity.get("scheduler_start_time_ticks")) is not int
        or not isinstance(identity.get("scheduler_cmdline"), list)
        or (
            schema_version == 2
            and (
                not isinstance(identity.get("working_directory"), str)
                or not isinstance(identity.get("scheduler_working_directory"), str)
                or identity.get("working_directory")
                != str(_queue_project_root(Path(str(identity["state_directory"]))))
                or identity.get("scheduler_working_directory")
                != identity.get("working_directory")
            )
        )
    ):
        raise ValueError("queue service identity differs from implementation ledger")
    return identity


def _execution_row(
    value: object,
    job_payload: Mapping[str, object] | None = None,
    *,
    implementation_prefix: str | None = None,
) -> dict[str, object]:
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    elif is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    if not isinstance(value, Mapping):
        raise ValueError("execution row is not serializable")
    mapping = dict(value)
    if set(mapping) == {"id", "job"}:
        row_id = mapping["id"]
        job = mapping["job"]
    else:
        row_id = mapping.get("id")
        job = mapping
    if not isinstance(row_id, str) or not row_id:
        raise ValueError("execution row has no ID")
    normalized_job = _json_round_trip(job)
    if not isinstance(normalized_job, dict):
        raise ValueError("execution row job must be an object")
    if job_payload is not None:
        overlap = set(normalized_job) & set(job_payload)
        if overlap:
            raise ValueError(
                f"execution payload overwrites search fields: {sorted(overlap)}"
            )
        normalized_job.update(_json_round_trip(dict(job_payload)))
    normalized_job.setdefault("run_name", _run_name(normalized_job))
    if implementation_prefix is not None:
        suffix = f"_i{implementation_prefix}"
        if not str(normalized_job["run_name"]).endswith(suffix):
            normalized_job["run_name"] = f"{normalized_job['run_name']}{suffix}"
    _validate_job(row_id, normalized_job)
    return {"id": row_id, "job": normalized_job}


def _loaded_execution_row(
    value: object, *, implementation_prefix: str
) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != {"id", "job"}:
        raise ValueError("execution manifest row schema differs")
    row = _execution_row(value)
    expected_run_name = f"{_run_name(row['job'])}_i{implementation_prefix}"
    if row["job"]["run_name"] != expected_run_name:
        raise ValueError("execution manifest run name is not canonical")
    return row


def _validate_job(row_id: str, job: dict[str, object]) -> None:
    if job.get("id") != row_id:
        raise ValueError("execution row and job IDs differ")
    if not isinstance(job.get("run_name"), str) or not job["run_name"]:
        raise ValueError("execution job has no run name")
    required = {
        "id",
        "family_id",
        "family_code",
        "research_question",
        "predecessor_id",
        "promotion_predecessor_id",
        "manifest_order",
        "stage",
        "batch_size",
        "seed",
        "horizon_epochs",
        "embedding_learning_rate",
        "deep_learning_rate",
        "anchor_embedding_learning_rate",
        "anchor_deep_learning_rate",
        "capacity",
        "run_name",
        "resolved_representation",
        "predecessor_artifacts",
    }
    if set(job) != required:
        raise ValueError("native-500M execution job schema differs")
    if job.get("batch_size") != 512 or job.get("seed") != 42:
        raise ValueError("native-500M execution requires batch 512 and seed 42")
    horizon = job.get("horizon_epochs")
    rates = (
        job.get("embedding_learning_rate"),
        job.get("deep_learning_rate"),
        job.get("anchor_embedding_learning_rate"),
        job.get("anchor_deep_learning_rate"),
    )
    if type(horizon) is not int or horizon < 1:
        raise ValueError("execution job has an invalid horizon")
    if any(
        not isinstance(value, str)
        or not 0 < float(value) < float("inf")
        or format(float(value), ".17g") != value
        for value in rates
    ):
        raise ValueError("execution job has an invalid learning rate")
    if (
        not isinstance(job.get("family_id"), str)
        or type(job.get("family_code")) is not int
        or not isinstance(job.get("research_question"), str)
        or not isinstance(job.get("predecessor_id"), str)
        or not isinstance(job.get("promotion_predecessor_id"), str)
        or type(job.get("manifest_order")) is not int
        or job.get("stage")
        not in {"initial", "capacity_followup", "frequency_followup", "boundary"}
        or (
            job.get("capacity") is not None
            and (type(job["capacity"]) is not int or int(job["capacity"]) < 1)
        )
    ):
        raise ValueError("execution job identity or capacity differs")
    representation = job.get("resolved_representation")
    predecessors = job.get("predecessor_artifacts")
    if not isinstance(representation, dict) or not representation:
        raise ValueError("execution job has no resolved representation")
    if not isinstance(predecessors, list):
        raise ValueError("execution job predecessor artifacts differ")
    for reference in predecessors:
        _validate_predecessor_reference(reference)
    if (
        job["family_id"]
        in {
            "bridge_rq3_output",
            "bridge_rq4_metadata",
            "aggregate",
        }
        and not predecessors
    ):
        raise ValueError("conditional execution job has no predecessor artifacts")


def _row_id(value: object) -> str:
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    elif is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    if not isinstance(value, Mapping) or not isinstance(value.get("id"), str):
        raise ValueError("execution row has no ID")
    return str(value["id"])


def _validate_predecessor_reference(value: object) -> None:
    if not isinstance(value, dict) or set(value) != {
        "role",
        "path",
        "size_bytes",
        "sha256",
        "logical_sha256",
        "row_id",
    }:
        raise ValueError("predecessor artifact reference schema differs")
    if (
        value["role"]
        not in {
            "search_predecessor",
            "aggregate_input",
            "output_bridge",
            "metadata_bridge",
            "continuation_authorization",
            "compatibility_state",
            "conditional_boundary_authorization",
        }
        or not isinstance(value["path"], str)
        or not value["path"]
        or type(value["size_bytes"]) is not int
        or value["size_bytes"] < 1
        or not isinstance(value["row_id"], str)
        or not value["row_id"]
    ):
        raise ValueError("predecessor artifact reference identity differs")
    _validate_sha256(value["sha256"], "predecessor physical SHA-256")
    _validate_sha256(value["logical_sha256"], "predecessor logical SHA-256")


def _validate_evaluation_population(value: object) -> None:
    if value != APPROVED_EVALUATION_POPULATION:
        raise ValueError("approved evaluation population identity differs")


def _family_selection_reference(
    *,
    root: Path,
    path: Path,
    document: Mapping[str, object],
    row_id: str,
) -> dict[str, object]:
    root = root.resolve(strict=True)
    path = path.resolve(strict=True)
    if path.is_symlink() or not path.is_file() or not path.is_relative_to(root):
        raise ValueError("selected predecessor evidence identity differs")
    return {
        "role": "search_predecessor",
        "path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _file_sha256(path),
        "logical_sha256": document["sha256"],
        "row_id": row_id,
    }


def _bound_document_reference(
    *,
    root: Path,
    path: Path,
    document: Mapping[str, object],
    role: str,
    row_id: str,
) -> dict[str, object]:
    root = root.resolve(strict=True)
    path = path.resolve(strict=True)
    if path.is_symlink() or not path.is_file() or not path.is_relative_to(root):
        raise ValueError("bound predecessor evidence identity differs")
    return {
        "role": role,
        "path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _file_sha256(path),
        "logical_sha256": document["sha256"],
        "row_id": row_id,
    }


def _selection_file_reference_for_state(
    *,
    root: Path,
    document: Mapping[str, object],
    authenticated: object,
    role: str,
) -> dict[str, object]:
    path = root.resolve(strict=True) / str(authenticated.relative_path)
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != authenticated.size_bytes
        or _file_sha256(path) != authenticated.physical_sha256
        or document.get("sha256") != authenticated.logical_sha256
    ):
        raise ValueError("authenticated compatibility state changed before binding")
    return {
        "role": role,
        "path": authenticated.relative_path,
        "size_bytes": authenticated.size_bytes,
        "sha256": authenticated.physical_sha256,
        "logical_sha256": authenticated.logical_sha256,
    }


def _authenticated_state_reference(
    *,
    root: Path,
    document: Mapping[str, object],
    authenticated: object,
    row_id: str,
) -> dict[str, object]:
    return _selection_file_reference_for_state(
        root=root,
        document=document,
        authenticated=authenticated,
        role="compatibility_state",
    ) | {"row_id": row_id}


def _representation_for_family(
    family_id: str,
    *,
    capacity: int | None,
    predecessor: object,
) -> dict[str, object]:
    from experiments.g3_pretrained_item_embeddings.configs.model import (
        G3Representation,
        RQ3_CATALOG_REPRESENTATIONS,
    )

    previous = G3Representation.from_dict(predecessor)
    if family_id == "baseline":
        representation = previous
    elif family_id == "untied_control":
        representation = G3Representation(item_id_tying="untied")
    elif family_id == "rq1_content_input":
        representation = G3Representation(history_representation="content")
    elif family_id == "rq2_content_concat":
        representation = G3Representation(
            history_representation="id_content",
            history_hidden_dim=capacity,
            item_id_tying="tied",
        )
    elif family_id in RQ3_CATALOG_REPRESENTATIONS:
        representation = G3Representation(
            history_representation="id_content",
            history_hidden_dim=previous.history_hidden_dim,
            catalog_representation=RQ3_CATALOG_REPRESENTATIONS[family_id],
            item_id_tying="untied",
        )
    elif family_id in {"rq4_artist", "rq4_album", "rq4_artist_album"}:
        metadata = {
            "rq4_artist": ("artist",),
            "rq4_album": ("album",),
            "rq4_artist_album": ("artist", "album"),
        }[family_id]
        representation = G3Representation(
            metadata=metadata,
            metadata_dim=capacity,
            item_id_tying="tied",
        )
    elif family_id == "rq5_global_gate":
        representation = G3Representation(
            history_representation="id_content",
            history_hidden_dim=previous.history_hidden_dim,
            content_gate="global",
            item_id_tying="tied",
        )
    elif family_id == "rq5_frequency_gate":
        representation = G3Representation(
            history_representation="id_content",
            history_hidden_dim=previous.history_hidden_dim,
            content_gate="frequency",
            gate_hidden_dim=capacity,
            frequency_gate_semantics="fp32_p09_v2",
            item_id_tying="tied",
        )
    else:
        raise ValueError("family representation cannot be derived")
    return representation.to_dict()


def _conditional_representation(
    family_id: str, *, state: Mapping[str, object], root: Path
) -> dict[str, object]:
    from experiments.g3_pretrained_item_embeddings.analysis.native500m_evidence import (
        authenticate_family_selection,
    )
    from experiments.g3_pretrained_item_embeddings.configs.model import G3Representation

    def representation(reference: object) -> G3Representation:
        if not isinstance(reference, dict):
            raise ValueError("conditional component target is absent")
        document = authenticate_family_selection(
            root / str(reference["path"]), root=root
        )[0]
        if document["winner"]["row_id"] != reference["row_id"]:
            raise ValueError("conditional component target row differs")
        return G3Representation.from_dict(
            document["winner"]["job"]["resolved_representation"]
        )

    current = representation(state["most_specific_selection"])
    targets = state["component_targets"]
    if family_id in {"bridge_rq3_output", "aggregate"}:
        output = representation(targets["output"])
        current = replace(
            current,
            catalog_representation=output.catalog_representation,
            item_id_tying=output.item_id_tying,
        )
    if family_id in {"bridge_rq4_metadata", "aggregate"}:
        metadata = representation(targets["metadata"])
        current = replace(
            current,
            metadata=metadata.metadata,
            metadata_dim=metadata.metadata_dim,
        )
    return current.to_dict()


def _run_name(job: Mapping[str, object]) -> str:
    row_id = str(job.get("id", ""))
    horizon = job.get("horizon_epochs")
    order = job.get("manifest_order")
    safe_id = "".join(
        character if character.isalnum() else "_" for character in row_id
    ).strip("_")
    if not safe_id or type(horizon) is not int or type(order) is not int:
        raise ValueError("cannot derive native-500M run name")
    return f"g3_native500m_{safe_id}_m{order:03d}_h{horizon}"


def _input_reference(value: object) -> InputManifestReference:
    if not isinstance(value, dict) or set(value) != {
        "role",
        "path",
        "size_bytes",
        "sha256",
        "logical_sha256",
    }:
        raise ValueError("input manifest reference schema differs")
    reference = InputManifestReference(**value)
    if reference.role not in _INPUT_ROLES or not reference.path:
        raise ValueError("input manifest reference identity differs")
    if type(reference.size_bytes) is not int or reference.size_bytes < 1:
        raise ValueError("input manifest reference size differs")
    _validate_sha256(reference.sha256, "input physical SHA-256")
    _validate_sha256(reference.logical_sha256, "input logical SHA-256")
    return reference


def _validate_input_reference(
    execution_manifest_path: Path, reference: InputManifestReference
) -> None:
    path = resolve_input_manifest_path(execution_manifest_path, reference)
    if path.is_symlink() or path.stat().st_size != reference.size_bytes:
        raise ValueError(f"input manifest artifact differs: {reference.role}")
    if _file_sha256(path) != reference.sha256:
        raise ValueError(f"input manifest physical SHA-256 differs: {reference.role}")
    document = _load_json(path)
    supplied_logical = document.get("sha256")
    if supplied_logical is None:
        logical_sha256 = _canonical_sha256(document)
    else:
        body = {key: value for key, value in document.items() if key != "sha256"}
        logical_sha256 = str(supplied_logical)
        if logical_sha256 != _canonical_sha256(body):
            raise ValueError(
                f"input manifest embedded logical SHA-256 differs: {reference.role}"
            )
    if logical_sha256 != reference.logical_sha256:
        raise ValueError(f"input manifest logical SHA-256 differs: {reference.role}")


def _validate_predecessor_identity(
    execution_manifest_path: Path, reference: dict[str, object]
) -> tuple[Path, dict[str, object]]:
    path = _resolve_bound_path(execution_manifest_path, str(reference["path"]))
    if path.is_symlink() or path.stat().st_size != reference["size_bytes"]:
        raise ValueError("predecessor artifact differs")
    if _file_sha256(path) != reference["sha256"]:
        raise ValueError("predecessor artifact physical SHA-256 differs")
    document = _load_json(path)
    supplied = document.get("sha256")
    body = {key: value for key, value in document.items() if key != "sha256"}
    if supplied != _canonical_sha256(body) or supplied != reference["logical_sha256"]:
        raise ValueError("predecessor artifact logical SHA-256 differs")
    return path, document


def resolve_input_manifest_path(
    execution_manifest_path: Path, reference: InputManifestReference
) -> Path:
    return _resolve_bound_path(execution_manifest_path, reference.path)


def _resolve_bound_path(execution_manifest_path: Path, relative_path: str) -> Path:
    path = Path(relative_path)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"bound artifact path is invalid: {relative_path}")
    resolved = execution_project_root(execution_manifest_path) / path
    if resolved.is_symlink() or not resolved.is_file():
        raise ValueError(f"bound artifact path is absent: {relative_path}")
    return resolved.resolve(strict=True)


def _protocol_sha256(value: str | None) -> str:
    if value is None:
        from experiments.g3_pretrained_item_embeddings.protocol.native500m.constants import (
            PROTOCOL_SHA256,
        )

        value = PROTOCOL_SHA256
    _validate_sha256(value, "protocol SHA-256")
    return value


def _validate_sha256(value: str, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} is invalid")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load JSON artifact {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact is not an object: {path}")
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r}")


def _json_round_trip(value: object) -> object:
    return json.loads(_canonical_bytes(value))


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_immutable(path: Path, content: bytes, *, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            dir=path.parent,
            delete=False,
        ) as handle:
            handle.write(content)
            temporary = Path(handle.name)
        try:
            temporary.chmod(mode)
            try:
                os.link(temporary, path)
            except FileExistsError:
                pass
        finally:
            temporary.unlink(missing_ok=True)
    except OSError as error:
        raise RuntimeError(f"cannot persist immutable artifact {path}") from error
    if path.read_bytes() != content:
        raise RuntimeError(f"immutable artifact differs: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path, nargs="?")
    parser.add_argument("--materialize-baseline", action="store_true")
    parser.add_argument(
        "--manifest-directory",
        type=Path,
        default=PROJECT_ROOT / "generated/g3-native500m/execution-manifests",
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=PROJECT_ROOT / "generated/training-queue-service",
    )
    parser.add_argument("--existing-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--materialize-only", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    arguments = parser.parse_args()
    if arguments.materialize_baseline:
        if arguments.manifest is not None:
            parser.error("manifest and --materialize-baseline are mutually exclusive")
        manifest_path = materialize_baseline_execution_manifest(
            output_directory=arguments.manifest_directory
        )
    elif arguments.manifest is None:
        parser.error("manifest or --materialize-baseline is required")
    else:
        manifest_path = arguments.manifest
    if arguments.materialize_only:
        if not arguments.materialize_baseline:
            parser.error("--materialize-only requires --materialize-baseline")
        print(manifest_path)
        return
    if arguments.verify_only:
        if arguments.dry_run or arguments.existing_only:
            parser.error("--verify-only cannot submit or inspect a queue batch")
        manifest = load_execution_manifest(manifest_path, validate_inputs=True)
        specification = build_batch_specification(
            manifest_path,
            expected_protocol_sha256=manifest.protocol_sha256,
        )
        print(specification.sha256)
        return
    print(
        submit_execution_manifest(
            manifest_path=manifest_path,
            state_directory=arguments.state_dir,
            existing_only=arguments.existing_only,
            dry_run=arguments.dry_run,
        )
    )


if __name__ == "__main__":
    main()
