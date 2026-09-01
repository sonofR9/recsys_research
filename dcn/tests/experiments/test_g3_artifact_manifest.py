import json

import pytest

from experiments.g3_pretrained_item_embeddings.protocol.manifests import (
    ArtifactBinding,
    ArtifactManifest,
    build_artifact_manifest,
    load_artifact_manifest,
    persist_artifact_manifest,
    validate_artifact_manifest,
    validate_content_manifest,
    validate_feature_manifest,
)


def test_artifact_manifest_binds_regular_files_protocol_and_metadata(tmp_path) -> None:
    source = tmp_path / "source.bin"
    compact = tmp_path / "compact.bin"
    source.write_bytes(b"source")
    compact.write_bytes(b"compact")

    manifest = build_artifact_manifest(
        root=tmp_path,
        artifacts={"content_source": source, "compact_output": compact},
        metadata={"dataset_size": "native-50m", "width": 128},
    )

    validated = validate_artifact_manifest(
        root=tmp_path,
        manifest=manifest,
        required_roles=("content_source", "compact_output"),
        required_metadata={"dataset_size": "native-50m", "width": 128},
    )
    assert validated == manifest
    assert [binding.role for binding in manifest.artifacts] == [
        "compact_output",
        "content_source",
    ]

    compact.write_bytes(b"forgery")
    with pytest.raises(ValueError, match="hash"):
        validate_artifact_manifest(root=tmp_path, manifest=manifest)


def test_manifest_rejects_paths_outside_root_symlinks_and_unknown_roles(tmp_path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    link = root / "link.bin"
    link.symlink_to(outside)

    with pytest.raises(ValueError, match="outside"):
        build_artifact_manifest(
            root=root,
            artifacts={"content_source": outside},
            metadata={},
        )
    with pytest.raises(ValueError, match="regular file"):
        build_artifact_manifest(
            root=root,
            artifacts={"content_source": link},
            metadata={},
        )

    local = root / "local.bin"
    local.write_bytes(b"local")
    manifest = build_artifact_manifest(
        root=root,
        artifacts={"known": local},
        metadata={},
    )
    with pytest.raises(ValueError, match="missing roles"):
        validate_artifact_manifest(
            root=root,
            manifest=manifest,
            required_roles=("known", "required"),
        )


def test_manifest_loader_rejects_duplicate_json_keys_and_schema_drift(tmp_path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(
        '{"schema_version":1,"schema_version":1,"protocol_sha256":"x",'
        '"artifacts":[],"metadata":{}}'
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_artifact_manifest(path)

    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "protocol_sha256": "x",
                "artifacts": [],
                "metadata": {},
                "unexpected": True,
            }
        )
    )
    with pytest.raises(ValueError, match="manifest keys"):
        load_artifact_manifest(path)

    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "protocol_sha256": "x",
                "artifacts": [],
                "metadata": {},
            }
        )
    )
    loaded = load_artifact_manifest(path)
    with pytest.raises(ValueError, match="schema"):
        validate_artifact_manifest(root=tmp_path, manifest=loaded)


def test_content_manifest_requires_all_hash_roles_and_approved_compact_hash(
    tmp_path,
) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"not the approved compact table")
    generic = build_artifact_manifest(
        root=tmp_path,
        artifacts={
            "content_source": artifact,
            "compaction_implementation": artifact,
            "compact_remap": artifact,
            "compact_output": artifact,
        },
        metadata={"dataset_size": "native-50m"},
    )

    with pytest.raises(ValueError, match="duplicate artifact paths"):
        validate_content_manifest(root=tmp_path, manifest=generic)

    bindings = tuple(
        ArtifactBinding(
            role=role,
            path=f"{role}.bin",
            size_bytes=1,
            sha256=f"{index:064x}",
        )
        for index, role in enumerate(
            (
                "content_source",
                "compaction_implementation",
                "compact_remap",
                "compact_output",
            ),
            start=1,
        )
    )
    forged = ArtifactManifest.create(
        artifacts=bindings,
        metadata={"dataset_size": "native-50m"},
    )
    with pytest.raises(ValueError, match="approved compact-output hash"):
        validate_content_manifest(
            root=tmp_path,
            manifest=forged,
            validate_files=False,
        )


def test_artifact_manifest_persistence_is_canonical_and_immutable(tmp_path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"artifact")
    manifest = build_artifact_manifest(
        root=tmp_path,
        artifacts={"artifact": artifact},
        metadata={"dataset_size": "native-50m"},
    )
    destination = tmp_path / "protocol" / "artifact.json"

    assert persist_artifact_manifest(destination, manifest) == destination
    first = destination.read_bytes()
    persist_artifact_manifest(destination, manifest)
    assert destination.read_bytes() == first
    assert load_artifact_manifest(destination) == manifest

    changed = ArtifactManifest.create(
        artifacts=manifest.artifacts,
        metadata={"dataset_size": "native-500m"},
    )
    with pytest.raises(RuntimeError, match="immutable artifact manifest"):
        persist_artifact_manifest(destination, changed)


def test_feature_manifest_has_a_closed_native_train_only_identity(tmp_path) -> None:
    roles = (
        "events_source",
        "compact_remap",
        "materialization_implementation",
        "item_features",
        "training_user_histories",
        "artist_vocab",
        "album_vocab",
    )
    paths = {}
    for role in roles:
        path = tmp_path / f"{role}.bin"
        path.write_bytes(role.encode())
        paths[role] = path
    metadata = {
        "dataset_size": "native-50m",
        "validation_interval_seconds": 604800,
        "num_items": 3,
        "training_rows": 10,
        "training_users": 2,
        "artist_vocab_size": 4,
        "album_vocab_size": 5,
        "artist_unknown_rate": 0.1,
        "album_unknown_rate": 0.2,
        "artist_max_cardinality": 2,
        "album_max_cardinality": 3,
    }
    manifest = build_artifact_manifest(
        root=tmp_path,
        artifacts=paths,
        metadata=metadata,
    )

    assert validate_feature_manifest(root=tmp_path, manifest=manifest) == manifest

    drifted = ArtifactManifest.create(
        artifacts=manifest.artifacts,
        metadata=metadata | {"unexpected": True},
    )
    with pytest.raises(ValueError, match="metadata schema"):
        validate_feature_manifest(root=tmp_path, manifest=drifted)
