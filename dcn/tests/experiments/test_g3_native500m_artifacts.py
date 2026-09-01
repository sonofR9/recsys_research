from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import polars as pl
import pytest

from experiments.g3_pretrained_item_embeddings.protocol.native500m.artifacts import (
    Native500MSummary,
    create_content_manifest,
    create_dataset_manifest,
    create_feature_manifest,
    load_artifact_manifest,
    persist_artifact_manifest,
    validate_content_manifest,
    validate_compact_content_alignment,
    validate_dataset_manifest,
    validate_feature_manifest,
    validate_native500m_summary,
    validate_training_aggregates,
)
from experiments.g3_pretrained_item_embeddings.protocol.native500m.constants import (
    PROTOCOL,
    PROTOCOL_SHA256,
)


PROTOCOL_ROOT = (
    Path(__file__).resolve().parents[3]
    / "experiments/g3_pretrained_item_embeddings/protocol/native500m"
)


def _approved_summary() -> Native500MSummary:
    return Native500MSummary(
        num_items=157357,
        filtered_event_count=8304589,
        filtered_user_count=81926,
        remapped_event_count=8013866,
        remapped_user_count=81635,
        training_interaction_count=7755722,
        training_user_count=81020,
        evaluation_user_count=37018,
        content_width=128,
        validation_cutoff_timestamp=25395195,
        artist_vocab_size=25007,
        album_vocab_size=138834,
        artist_unknown_rate=0.0012709952528327307,
        album_unknown_rate=0.0014425796119651494,
        artist_max_cardinality=2,
        album_max_cardinality=241,
    )


def test_expected_manifests_bind_only_native500m_artifacts() -> None:
    content = create_content_manifest()
    dataset = create_dataset_manifest()
    features = create_feature_manifest()

    assert content.protocol_sha256 == PROTOCOL_SHA256
    assert dataset.protocol_sha256 == PROTOCOL_SHA256
    assert features.protocol_sha256 == PROTOCOL_SHA256
    assert content.kind == "content"
    assert dataset.kind == "dataset"
    assert features.kind == "features"
    assert content.metadata == {
        "dataset_size": "native-500m",
        "normalized": True,
        "num_items": 157357,
        "source_column": "normalized_embed",
        "width": 128,
    }
    assert features.metadata == {
        "album_max_cardinality": 241,
        "album_unknown_rate": 0.0014425796119651494,
        "album_vocab_size": 138834,
        "artist_max_cardinality": 2,
        "artist_unknown_rate": 0.0012709952528327307,
        "artist_vocab_size": 25007,
        "dataset_size": "native-500m",
        "num_items": 157357,
        "remapped_event_count": 8013866,
        "training_interaction_count": 7755722,
        "training_user_count": 81020,
        "validation_cutoff_timestamp": 25395195,
        "validation_interval_seconds": 604800,
    }
    assert dataset.metadata == {
        "dataset_size": "native-500m",
        "drop_unmapped_items": True,
        "evaluation_catalog": "all",
        "exclude_seen_from_evaluation": False,
        "evaluation_user_count": 37018,
        "filtered_event_count": 8304589,
        "filtered_user_count": 81926,
        "min_item_interactions_per_item": 5,
        "minimum_query_events": 2,
        "num_items": 157357,
        "remapped_event_count": 8013866,
        "remapped_user_count": 81635,
        "training_interaction_count": 7755722,
        "training_user_count": 81020,
        "validation_cutoff_timestamp": 25395195,
        "validation_interval_seconds": 604800,
    }
    assert {binding.role for binding in content.artifacts} == {
        "content_source",
        "compaction_implementation",
        "compact_remap",
        "compact_output",
    }
    assert {binding.role for binding in features.artifacts} == {
        "events_source",
        "compact_remap",
        "materialization_implementation",
        "item_features",
        "training_user_histories",
        "artist_vocab",
        "album_vocab",
    }
    assert {binding.role for binding in dataset.artifacts} == {
        "compact_remap",
        "filtered_events",
        "protocol_implementation",
        "remapped_events",
    }
    assert all(
        not binding.path.endswith("source.lock") for binding in dataset.artifacts
    )
    assert (
        next(
            binding for binding in content.artifacts if binding.role == "compact_output"
        ).sha256
        == PROTOCOL.content_sha256
    )
    assert validate_content_manifest(content, validate_files=False) == content
    assert validate_dataset_manifest(dataset, validate_files=False) == dataset
    assert validate_feature_manifest(features, validate_files=False) == features


def test_frozen_manifests_equal_the_public_closed_manifests() -> None:
    expected = {
        "content_manifest.json": create_content_manifest(),
        "dataset_manifest.json": create_dataset_manifest(),
        "feature_manifest.json": create_feature_manifest(),
    }
    assert {
        name: load_artifact_manifest(PROTOCOL_ROOT / name) for name in expected
    } == expected


def test_volatile_source_lock_is_not_part_of_dataset_identity(tmp_path) -> None:
    lock = tmp_path / "source.lock"
    before = create_dataset_manifest()
    lock.write_text("123\n")
    lock.write_text("456789\n")

    assert create_dataset_manifest() == before
    assert validate_dataset_manifest(before, validate_files=False) == before


def test_manifest_validation_rejects_dataset_hash_and_closed_schema_drift() -> None:
    content = create_content_manifest()
    wrong_metadata = content.create(
        kind="content",
        artifacts=content.artifacts,
        metadata=content.metadata | {"dataset_size": "native-50m"},
    )
    with pytest.raises(ValueError, match="closed native-500M content manifest"):
        validate_content_manifest(wrong_metadata, validate_files=False)

    bindings = list(content.artifacts)
    bindings[0] = replace(bindings[0], sha256="0" * 64)
    wrong_hash = content.create(
        kind="content",
        artifacts=bindings,
        metadata=content.metadata,
    )
    with pytest.raises(ValueError, match="closed native-500M content manifest"):
        validate_content_manifest(wrong_hash, validate_files=False)


def test_manifest_persistence_is_canonical_and_immutable(tmp_path) -> None:
    path = tmp_path / "content.json"
    manifest = create_content_manifest()

    persist_artifact_manifest(path, manifest)
    first = path.read_bytes()
    persist_artifact_manifest(path, manifest)
    assert path.read_bytes() == first
    assert load_artifact_manifest(path) == manifest
    assert json.loads(first)["protocol_sha256"] == PROTOCOL_SHA256

    with pytest.raises(RuntimeError, match="immutable"):
        persist_artifact_manifest(path, create_feature_manifest())


def test_manifest_loader_rejects_duplicate_keys_and_schema_drift(tmp_path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(
        '{"schema_version":1,"schema_version":1,"kind":"content",'
        '"protocol_sha256":"x","artifacts":[],"metadata":{}}'
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_artifact_manifest(path)

    document = create_content_manifest().to_dict() | {"unexpected": True}
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="closed schema"):
        load_artifact_manifest(path)


def test_summary_verifier_rejects_count_split_and_content_drift() -> None:
    summary = _approved_summary()
    assert validate_native500m_summary(summary) == summary

    for field, value in (
        ("num_items", 157356),
        ("filtered_event_count", 8304588),
        ("filtered_user_count", 81925),
        ("remapped_event_count", 8013865),
        ("remapped_user_count", 81634),
        ("training_interaction_count", 7755721),
        ("training_user_count", 81019),
        ("evaluation_user_count", 37017),
        ("content_width", 127),
        ("validation_cutoff_timestamp", 25395194),
    ):
        with pytest.raises(ValueError, match="native-500M semantic summary"):
            validate_native500m_summary(replace(summary, **{field: value}))


def test_content_alignment_is_proven_through_raw_item_remap() -> None:
    source = pl.DataFrame(
        {
            "item_id": [10, 20, 30],
            "normalized_embed": [[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]],
        }
    )
    remap = pl.DataFrame({"item_id": [10, 20, 30], "compact_id": [1, 2, 3]})
    compact = pl.DataFrame(
        {
            "compact_id": [1, 2, 3],
            "normalized_embed": [[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]],
        }
    )

    validate_compact_content_alignment(remap, source, compact)

    misaligned = compact.with_columns(
        pl.Series(
            "normalized_embed",
            [[0.0, 1.0], [1.0, 0.0], [0.6, 0.8]],
        )
    )
    with pytest.raises(ValueError, match="content/remap/source alignment"):
        validate_compact_content_alignment(remap, source, misaligned)


def test_training_aggregate_verifier_rejects_equal_total_row_drift() -> None:
    events = pl.DataFrame(
        {
            "uid": [1, 2, 2, 2, 2],
            "compact_item_id": [1, 1, 1, 2, 3],
            "timestamp": [1, 2, 3, 4, 11],
        }
    )
    features = pl.DataFrame({"compact_item_id": [1, 2, 3], "training_count": [3, 1, 0]})
    histories = pl.DataFrame({"uid": [1, 2], "training_history_length": [1, 3]})

    validate_training_aggregates(
        events,
        features,
        histories,
        cutoff_timestamp=10,
        num_items=3,
    )

    with pytest.raises(ValueError, match="per-item training counts"):
        validate_training_aggregates(
            events,
            features.with_columns(pl.Series("training_count", [2, 2, 0])),
            histories,
            cutoff_timestamp=10,
            num_items=3,
        )
    with pytest.raises(ValueError, match="per-user training histories"):
        validate_training_aggregates(
            events,
            features,
            histories.with_columns(pl.Series("training_history_length", [2, 2])),
            cutoff_timestamp=10,
            num_items=3,
        )
