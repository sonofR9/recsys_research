from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .constants import APPROVED_PROTOCOL, APPROVED_PROTOCOL_SHA256


CONTENT_ARTIFACT_ROLES = (
    "content_source",
    "compaction_implementation",
    "compact_remap",
    "compact_output",
)

FEATURE_ARTIFACT_ROLES = (
    "events_source",
    "compact_remap",
    "materialization_implementation",
    "item_features",
    "training_user_histories",
    "artist_vocab",
    "album_vocab",
)

FEATURE_METADATA_KEYS = {
    "dataset_size",
    "validation_interval_seconds",
    "num_items",
    "training_rows",
    "training_users",
    "artist_vocab_size",
    "album_vocab_size",
    "artist_unknown_rate",
    "album_unknown_rate",
    "artist_max_cardinality",
    "album_max_cardinality",
}


@dataclass(frozen=True)
class ArtifactBinding:
    role: str
    path: str
    size_bytes: int
    sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "path": self.path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class ArtifactManifest:
    schema_version: int
    protocol_sha256: str
    artifacts: tuple[ArtifactBinding, ...]
    metadata_json: str

    @classmethod
    def create(
        cls,
        *,
        artifacts: Sequence[ArtifactBinding],
        metadata: Mapping[str, object],
        protocol_sha256: str = APPROVED_PROTOCOL_SHA256,
    ) -> ArtifactManifest:
        return cls(
            schema_version=1,
            protocol_sha256=protocol_sha256,
            artifacts=tuple(sorted(artifacts, key=lambda binding: binding.role)),
            metadata_json=_canonical_json(dict(metadata)),
        )

    @property
    def metadata(self) -> dict[str, object]:
        value = json.loads(self.metadata_json)
        if not isinstance(value, dict):
            raise ValueError("manifest metadata must be an object")
        return value

    @property
    def sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.to_dict()).encode()).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "protocol_sha256": self.protocol_sha256,
            "artifacts": [binding.to_dict() for binding in self.artifacts],
            "metadata": self.metadata,
        }


def build_artifact_manifest(
    *,
    root: Path,
    artifacts: Mapping[str, Path],
    metadata: Mapping[str, object],
) -> ArtifactManifest:
    root = root.resolve()
    bindings = []
    for role, candidate in sorted(artifacts.items()):
        if not role:
            raise ValueError("artifact role must be nonempty")
        path = candidate if candidate.is_absolute() else root / candidate
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"artifact must be a regular file: {path}")
        resolved = path.resolve()
        if not resolved.is_relative_to(root):
            raise ValueError(f"artifact path is outside root: {path}")
        bindings.append(
            ArtifactBinding(
                role=role,
                path=resolved.relative_to(root).as_posix(),
                size_bytes=resolved.stat().st_size,
                sha256=_file_sha256(resolved),
            )
        )
    return ArtifactManifest.create(artifacts=bindings, metadata=metadata)


def validate_artifact_manifest(
    *,
    root: Path,
    manifest: ArtifactManifest,
    required_roles: Sequence[str] = (),
    required_metadata: Mapping[str, object] | None = None,
    validate_files: bool = True,
) -> ArtifactManifest:
    if manifest.schema_version != 1:
        raise ValueError("unsupported artifact manifest schema")
    if manifest.protocol_sha256 != APPROVED_PROTOCOL_SHA256:
        raise ValueError("artifact manifest does not bind the approved G3 protocol")
    metadata = manifest.metadata
    roles = [binding.role for binding in manifest.artifacts]
    paths = [binding.path for binding in manifest.artifacts]
    if any(not role for role in roles) or len(set(roles)) != len(roles):
        raise ValueError("artifact manifest has duplicate or empty roles")
    if len(set(paths)) != len(paths):
        raise ValueError("artifact manifest has duplicate artifact paths")
    missing = sorted(set(required_roles) - set(roles))
    if missing:
        raise ValueError(f"artifact manifest has missing roles {missing}")
    for key, value in (required_metadata or {}).items():
        if metadata.get(key) != value:
            raise ValueError(f"artifact manifest metadata mismatch for {key!r}")
    if validate_files:
        root = root.resolve()
        for binding in manifest.artifacts:
            _validate_binding(root, binding)
    return manifest


def validate_artifact_bindings(
    *,
    root: Path,
    manifest: ArtifactManifest,
    roles: Sequence[str],
) -> ArtifactManifest:
    validate_artifact_manifest(root=root, manifest=manifest, validate_files=False)
    by_role = {binding.role: binding for binding in manifest.artifacts}
    missing = sorted(set(roles) - set(by_role))
    if missing:
        raise ValueError(f"artifact manifest has missing roles {missing}")
    root = root.resolve()
    for role in roles:
        _validate_binding(root, by_role[role])
    return manifest


def validate_content_manifest(
    *,
    root: Path,
    manifest: ArtifactManifest,
    validate_files: bool = True,
) -> ArtifactManifest:
    validate_artifact_manifest(
        root=root,
        manifest=manifest,
        required_roles=CONTENT_ARTIFACT_ROLES,
        validate_files=validate_files,
    )
    if {binding.role for binding in manifest.artifacts} != set(CONTENT_ARTIFACT_ROLES):
        raise ValueError("content manifest roles do not match the approved closed schema")
    dataset_size = manifest.metadata.get("dataset_size")
    if not isinstance(dataset_size, str):
        raise ValueError("content manifest has no dataset size")
    expected = APPROVED_PROTOCOL.content_hash(dataset_size)
    output = next(
        binding for binding in manifest.artifacts if binding.role == "compact_output"
    )
    if output.sha256 != expected:
        raise ValueError("content manifest does not bind the approved compact-output hash")
    return manifest


def validate_feature_manifest(
    *,
    root: Path,
    manifest: ArtifactManifest,
    validate_files: bool = True,
) -> ArtifactManifest:
    validate_artifact_manifest(
        root=root,
        manifest=manifest,
        required_roles=FEATURE_ARTIFACT_ROLES,
        validate_files=validate_files,
    )
    if {binding.role for binding in manifest.artifacts} != set(
        FEATURE_ARTIFACT_ROLES
    ):
        raise ValueError("feature manifest roles do not match the approved closed schema")
    metadata = manifest.metadata
    if set(metadata) != FEATURE_METADATA_KEYS:
        raise ValueError("feature manifest metadata schema does not match the approved schema")
    if metadata["dataset_size"] not in APPROVED_PROTOCOL.dataset_sizes:
        raise ValueError("feature manifest has an unapproved dataset size")
    if metadata["validation_interval_seconds"] != 604800:
        raise ValueError("feature manifest has an unapproved validation interval")
    count_keys = FEATURE_METADATA_KEYS - {
        "dataset_size",
        "artist_unknown_rate",
        "album_unknown_rate",
    }
    if any(type(metadata[key]) is not int or metadata[key] < 0 for key in count_keys):
        raise ValueError("feature manifest has invalid count metadata")
    for key in ("artist_unknown_rate", "album_unknown_rate"):
        value = metadata[key]
        if type(value) is not float or not 0.0 <= value <= 1.0:
            raise ValueError("feature manifest has invalid unknown-rate metadata")
    return manifest


def persist_artifact_manifest(path: Path, manifest: ArtifactManifest) -> Path:
    content = (_canonical_json(manifest.to_dict()) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(content)
    except FileExistsError:
        if path.read_bytes() != content:
            raise RuntimeError(f"immutable artifact manifest differs: {path}")
    return path


def load_artifact_manifest(path: Path) -> ArtifactManifest:
    try:
        document = json.loads(
            path.read_text(),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load artifact manifest {path}") from error
    if not isinstance(document, dict):
        raise ValueError("artifact manifest must be an object")
    expected_keys = {"schema_version", "protocol_sha256", "artifacts", "metadata"}
    if set(document) != expected_keys:
        raise ValueError("artifact manifest keys do not match the closed schema")
    artifacts = document["artifacts"]
    metadata = document["metadata"]
    if not isinstance(artifacts, list) or not isinstance(metadata, dict):
        raise ValueError("artifact manifest has invalid artifacts or metadata")
    bindings = tuple(_binding_from_dict(value) for value in artifacts)
    schema_version = document["schema_version"]
    protocol_sha256 = document["protocol_sha256"]
    if type(schema_version) is not int or not isinstance(protocol_sha256, str):
        raise ValueError("artifact manifest has invalid identity fields")
    return ArtifactManifest(
        schema_version=schema_version,
        protocol_sha256=protocol_sha256,
        artifacts=tuple(sorted(bindings, key=lambda binding: binding.role)),
        metadata_json=_canonical_json(metadata),
    )


def _binding_from_dict(value: object) -> ArtifactBinding:
    if not isinstance(value, dict) or set(value) != {
        "role",
        "path",
        "size_bytes",
        "sha256",
    }:
        raise ValueError("artifact binding keys do not match the closed schema")
    role = value["role"]
    path = value["path"]
    size_bytes = value["size_bytes"]
    sha256 = value["sha256"]
    if (
        not isinstance(role, str)
        or not isinstance(path, str)
        or type(size_bytes) is not int
        or not isinstance(sha256, str)
    ):
        raise ValueError("artifact binding has invalid field types")
    return ArtifactBinding(role, path, size_bytes, sha256)


def _validate_binding(root: Path, binding: ArtifactBinding) -> None:
    if binding.size_bytes < 0 or len(binding.sha256) != 64:
        raise ValueError(f"invalid artifact identity for {binding.role!r}")
    try:
        int(binding.sha256, 16)
    except ValueError as error:
        raise ValueError(f"invalid artifact hash for {binding.role!r}") from error
    relative = Path(binding.path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"artifact path is outside root: {binding.path}")
    path = root / relative
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"artifact must be a regular file: {path}")
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"artifact path is outside root: {binding.path}")
    if resolved.stat().st_size != binding.size_bytes:
        raise ValueError(f"artifact size changed for {binding.role!r}")
    if _file_sha256(resolved) != binding.sha256:
        raise ValueError(f"artifact hash changed for {binding.role!r}")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON number {value!r}")
