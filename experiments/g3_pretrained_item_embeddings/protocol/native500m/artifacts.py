from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import polars as pl

from .constants import (
    CONTENT_ARTIFACT_IDENTITIES,
    DATASET_ARTIFACT_IDENTITIES,
    FEATURE_ARTIFACT_IDENTITIES,
    PROTOCOL,
    PROTOCOL_SHA256,
    ArtifactIdentity,
)


ManifestKind = Literal["content", "dataset", "features"]


@dataclass(frozen=True)
class ArtifactBinding:
    role: str
    path: str
    size_bytes: int
    sha256: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ArtifactManifest:
    schema_version: int
    kind: ManifestKind
    protocol_sha256: str
    artifacts: tuple[ArtifactBinding, ...]
    metadata_json: str

    @classmethod
    def create(
        cls,
        *,
        kind: ManifestKind,
        artifacts: Sequence[ArtifactBinding],
        metadata: Mapping[str, object],
    ) -> ArtifactManifest:
        return cls(
            schema_version=1,
            kind=kind,
            protocol_sha256=PROTOCOL_SHA256,
            artifacts=tuple(sorted(artifacts, key=lambda binding: binding.role)),
            metadata_json=_canonical_json(dict(metadata)),
        )

    @property
    def metadata(self) -> dict[str, object]:
        value = json.loads(self.metadata_json)
        if not isinstance(value, dict):
            raise ValueError("artifact manifest metadata must be an object")
        return value

    @property
    def sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.to_dict()).encode()).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "protocol_sha256": self.protocol_sha256,
            "artifacts": [binding.to_dict() for binding in self.artifacts],
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class Native500MSummary:
    num_items: int
    filtered_event_count: int
    filtered_user_count: int
    remapped_event_count: int
    remapped_user_count: int
    training_interaction_count: int
    training_user_count: int
    evaluation_user_count: int
    content_width: int
    validation_cutoff_timestamp: int
    artist_vocab_size: int
    album_vocab_size: int
    artist_unknown_rate: float
    album_unknown_rate: float
    artist_max_cardinality: int
    album_max_cardinality: int


def create_content_manifest() -> ArtifactManifest:
    return ArtifactManifest.create(
        kind="content",
        artifacts=_bindings(CONTENT_ARTIFACT_IDENTITIES),
        metadata={
            "dataset_size": PROTOCOL.dataset_size,
            "normalized": True,
            "num_items": PROTOCOL.num_items,
            "source_column": "normalized_embed",
            "width": PROTOCOL.content_width,
        },
    )


def create_feature_manifest() -> ArtifactManifest:
    return ArtifactManifest.create(
        kind="features",
        artifacts=_bindings(FEATURE_ARTIFACT_IDENTITIES),
        metadata={
            "album_max_cardinality": 241,
            "album_unknown_rate": 0.0014425796119651494,
            "album_vocab_size": 138834,
            "artist_max_cardinality": 2,
            "artist_unknown_rate": 0.0012709952528327307,
            "artist_vocab_size": 25007,
            "dataset_size": PROTOCOL.dataset_size,
            "num_items": PROTOCOL.num_items,
            "remapped_event_count": PROTOCOL.remapped_event_count,
            "training_interaction_count": PROTOCOL.training_interaction_count,
            "training_user_count": PROTOCOL.training_user_count,
            "validation_cutoff_timestamp": PROTOCOL.validation_cutoff_timestamp,
            "validation_interval_seconds": PROTOCOL.validation_interval_seconds,
        },
    )


def create_dataset_manifest() -> ArtifactManifest:
    return ArtifactManifest.create(
        kind="dataset",
        artifacts=_bindings(DATASET_ARTIFACT_IDENTITIES),
        metadata={
            "dataset_size": PROTOCOL.dataset_size,
            "drop_unmapped_items": True,
            "evaluation_catalog": "all",
            "exclude_seen_from_evaluation": False,
            "filtered_event_count": PROTOCOL.filtered_event_count,
            "filtered_user_count": PROTOCOL.filtered_user_count,
            "min_item_interactions_per_item": 5,
            "minimum_query_events": PROTOCOL.minimum_query_events,
            "num_items": PROTOCOL.num_items,
            "remapped_event_count": PROTOCOL.remapped_event_count,
            "remapped_user_count": PROTOCOL.remapped_user_count,
            "training_interaction_count": PROTOCOL.training_interaction_count,
            "training_user_count": PROTOCOL.training_user_count,
            "evaluation_user_count": PROTOCOL.evaluation_user_count,
            "validation_cutoff_timestamp": PROTOCOL.validation_cutoff_timestamp,
            "validation_interval_seconds": PROTOCOL.validation_interval_seconds,
        },
    )


def validate_content_manifest(
    manifest: ArtifactManifest,
    *,
    root: Path | None = None,
    validate_files: bool = True,
    validate_semantics: bool = False,
) -> ArtifactManifest:
    if manifest != create_content_manifest():
        raise ValueError(
            "manifest does not match the closed native-500M content manifest"
        )
    _validate_requested_files(manifest, root, validate_files)
    if validate_semantics:
        if root is None:
            raise ValueError("semantic validation requires the repository root")
        _inspect_content(root, manifest)
    return manifest


def validate_feature_manifest(
    manifest: ArtifactManifest,
    *,
    root: Path | None = None,
    validate_files: bool = True,
    validate_semantics: bool = False,
) -> ArtifactManifest:
    if manifest != create_feature_manifest():
        raise ValueError(
            "manifest does not match the closed native-500M feature manifest"
        )
    _validate_requested_files(manifest, root, validate_files)
    if validate_semantics:
        if root is None:
            raise ValueError("semantic validation requires the repository root")
        _inspect_features(root, manifest)
    return manifest


def validate_dataset_manifest(
    manifest: ArtifactManifest,
    *,
    root: Path | None = None,
    validate_files: bool = True,
    validate_semantics: bool = False,
) -> ArtifactManifest:
    if manifest != create_dataset_manifest():
        raise ValueError(
            "manifest does not match the closed native-500M dataset manifest"
        )
    _validate_requested_files(manifest, root, validate_files)
    if validate_semantics:
        if root is None:
            raise ValueError("semantic validation requires the repository root")
        _inspect_dataset(root, manifest)
    return manifest


def verify_native500m_artifacts(root: Path) -> Native500MSummary:
    content = validate_content_manifest(
        create_content_manifest(),
        root=root,
        validate_files=True,
        validate_semantics=False,
    )
    features = validate_feature_manifest(
        create_feature_manifest(),
        root=root,
        validate_files=True,
        validate_semantics=False,
    )
    dataset = validate_dataset_manifest(
        create_dataset_manifest(),
        root=root,
        validate_files=True,
        validate_semantics=False,
    )
    content_summary = _inspect_content(root, content)
    feature_summary = _inspect_features(root, features)
    dataset_summary = _inspect_dataset(root, dataset)
    if (
        feature_summary.remapped_event_count != dataset_summary.remapped_event_count
        or feature_summary.training_interaction_count
        != dataset_summary.training_interaction_count
        or feature_summary.training_user_count != dataset_summary.training_user_count
    ):
        raise ValueError("dataset and feature split summaries disagree")
    return validate_native500m_summary(
        Native500MSummary(
            num_items=content_summary[0],
            filtered_event_count=dataset_summary.filtered_event_count,
            filtered_user_count=dataset_summary.filtered_user_count,
            content_width=content_summary[1],
            remapped_event_count=feature_summary.remapped_event_count,
            remapped_user_count=dataset_summary.remapped_user_count,
            training_interaction_count=(feature_summary.training_interaction_count),
            training_user_count=feature_summary.training_user_count,
            evaluation_user_count=dataset_summary.evaluation_user_count,
            validation_cutoff_timestamp=(feature_summary.validation_cutoff_timestamp),
            artist_vocab_size=feature_summary.artist_vocab_size,
            album_vocab_size=feature_summary.album_vocab_size,
            artist_unknown_rate=feature_summary.artist_unknown_rate,
            album_unknown_rate=feature_summary.album_unknown_rate,
            artist_max_cardinality=feature_summary.artist_max_cardinality,
            album_max_cardinality=feature_summary.album_max_cardinality,
        )
    )


def validate_native500m_summary(
    summary: Native500MSummary,
) -> Native500MSummary:
    if summary != _expected_summary():
        raise ValueError(
            "native-500M semantic summary does not match the approved data"
        )
    return summary


def validate_compact_content_alignment(
    remap: pl.DataFrame,
    source: pl.DataFrame,
    compact: pl.DataFrame,
) -> None:
    if remap.columns != ["item_id", "compact_id"]:
        raise ValueError("content remap columns do not match the closed schema")
    if source.columns != ["item_id", "normalized_embed"]:
        raise ValueError("content source columns do not match the closed schema")
    if compact.columns != ["compact_id", "normalized_embed"]:
        raise ValueError("compact content columns do not match the closed schema")
    _validate_contiguous_ids(remap["compact_id"], remap.height, "item remap")
    _validate_contiguous_ids(compact["compact_id"], compact.height, "compact content")
    if (
        remap.height != compact.height
        or remap["item_id"].n_unique() != remap.height
        or source["item_id"].n_unique() != source.height
        or not remap["item_id"].is_sorted()
    ):
        raise ValueError("content/remap/source alignment has duplicate or missing IDs")
    expected = (
        remap.join(source, on="item_id", how="inner", validate="1:1")
        .select("compact_id", "normalized_embed")
        .sort("compact_id")
    )
    if expected.height != remap.height or not expected.equals(compact):
        raise ValueError("content/remap/source alignment differs")


def validate_training_aggregates(
    events: pl.DataFrame | pl.LazyFrame,
    item_features: pl.DataFrame,
    training_user_histories: pl.DataFrame,
    *,
    cutoff_timestamp: int,
    num_items: int,
) -> None:
    if item_features.columns != ["compact_item_id", "training_count"]:
        raise ValueError("training-count columns do not match the closed schema")
    if training_user_histories.columns != [
        "uid",
        "training_history_length",
    ]:
        raise ValueError("training-history columns do not match the closed schema")
    _validate_contiguous_ids(
        item_features["compact_item_id"], num_items, "item training counts"
    )
    if training_user_histories["uid"].n_unique() != training_user_histories.height:
        raise ValueError("per-user training histories contain duplicate users")
    lazy_events = events.lazy() if isinstance(events, pl.DataFrame) else events
    training = lazy_events.filter(pl.col("timestamp") < cutoff_timestamp)
    expected_item_counts = (
        pl.DataFrame({"compact_item_id": range(1, num_items + 1)})
        .lazy()
        .join(
            training.group_by("compact_item_id").agg(
                pl.len().cast(pl.Int64).alias("training_count")
            ),
            on="compact_item_id",
            how="left",
        )
        .with_columns(pl.col("training_count").fill_null(0))
        .sort("compact_item_id")
        .collect()
    )
    actual_item_counts = item_features.select(
        pl.col("compact_item_id").cast(pl.Int64),
        pl.col("training_count").cast(pl.Int64),
    )
    if not actual_item_counts.equals(expected_item_counts):
        raise ValueError("per-item training counts differ from pre-cutoff events")
    expected_histories = (
        training.group_by("uid")
        .agg(pl.len().cast(pl.Int64).alias("training_history_length"))
        .sort("uid")
        .collect()
    )
    actual_histories = training_user_histories.select(
        pl.col("uid"),
        pl.col("training_history_length").cast(pl.Int64),
    ).sort("uid")
    if not actual_histories.equals(expected_histories):
        raise ValueError("per-user training histories differ from pre-cutoff events")


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
    if not isinstance(document, dict) or set(document) != {
        "schema_version",
        "kind",
        "protocol_sha256",
        "artifacts",
        "metadata",
    }:
        raise ValueError("artifact manifest does not match the closed schema")
    schema_version = document["schema_version"]
    kind = document["kind"]
    protocol_sha256 = document["protocol_sha256"]
    artifacts = document["artifacts"]
    metadata = document["metadata"]
    if (
        type(schema_version) is not int
        or kind not in {"content", "dataset", "features"}
        or not isinstance(protocol_sha256, str)
        or not isinstance(artifacts, list)
        or not isinstance(metadata, dict)
    ):
        raise ValueError("artifact manifest has invalid field types")
    bindings = tuple(_binding_from_dict(value) for value in artifacts)
    return ArtifactManifest(
        schema_version=schema_version,
        kind=kind,
        protocol_sha256=protocol_sha256,
        artifacts=tuple(sorted(bindings, key=lambda binding: binding.role)),
        metadata_json=_canonical_json(metadata),
    )


def _bindings(
    identities: Sequence[ArtifactIdentity],
) -> tuple[ArtifactBinding, ...]:
    return tuple(
        ArtifactBinding(
            identity.role,
            identity.path,
            identity.size_bytes,
            identity.sha256,
        )
        for identity in identities
    )


def _validate_requested_files(
    manifest: ArtifactManifest,
    root: Path | None,
    validate_files: bool,
) -> None:
    if manifest.schema_version != 1 or manifest.protocol_sha256 != PROTOCOL_SHA256:
        raise ValueError("artifact manifest is not bound to this protocol version")
    if validate_files:
        if root is None:
            raise ValueError("file validation requires the repository root")
        for binding in manifest.artifacts:
            _validate_binding(root.resolve(), binding)


def _validate_binding(root: Path, binding: ArtifactBinding) -> None:
    relative = Path(binding.path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"artifact path is outside the repository: {binding.path}")
    path = root / relative
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"artifact must be a regular file: {path}")
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"artifact path is outside the repository: {binding.path}")
    if resolved.stat().st_size != binding.size_bytes:
        raise ValueError(f"artifact size changed for {binding.role!r}")
    if _file_sha256(resolved) != binding.sha256:
        raise ValueError(f"artifact hash changed for {binding.role!r}")


def _inspect_content(
    root: Path,
    manifest: ArtifactManifest,
) -> tuple[int, int]:
    paths = _paths_by_role(root, manifest)
    remap = pl.read_parquet(paths["compact_remap"], columns=["item_id", "compact_id"])
    compact = pl.read_parquet(paths["compact_output"])
    source = (
        pl.scan_parquet(paths["content_source"])
        .select("item_id", "normalized_embed")
        .join(remap.lazy().select("item_id"), on="item_id", how="inner")
        .collect(engine="streaming")
    )
    validate_compact_content_alignment(remap, source, compact)
    content = compact.lazy()
    summary = (
        content.select(
            pl.len().alias("rows"),
            pl.col("compact_id").n_unique().alias("unique_ids"),
            pl.col("compact_id").min().alias("minimum_id"),
            pl.col("compact_id").max().alias("maximum_id"),
            pl.col("normalized_embed").list.len().min().alias("minimum_width"),
            pl.col("normalized_embed").list.len().max().alias("maximum_width"),
            pl.col("normalized_embed")
            .list.eval(pl.element().is_finite())
            .list.all()
            .all()
            .alias("finite"),
            (
                pl.col("normalized_embed")
                .list.eval(pl.element() * pl.element())
                .list.sum()
                .sqrt()
                - 1.0
            )
            .abs()
            .max()
            .alias("maximum_norm_error"),
        )
        .collect()
        .row(0, named=True)
    )
    if (
        summary["rows"] != PROTOCOL.num_items
        or summary["unique_ids"] != PROTOCOL.num_items
        or summary["minimum_id"] != 1
        or summary["maximum_id"] != PROTOCOL.num_items
        or summary["minimum_width"] != PROTOCOL.content_width
        or summary["maximum_width"] != PROTOCOL.content_width
        or not summary["finite"]
        or summary["maximum_norm_error"] > 1e-6
    ):
        raise ValueError(
            "compact content identity, order, width, or normalization drifted"
        )
    return PROTOCOL.num_items, PROTOCOL.content_width


def _inspect_features(root: Path, manifest: ArtifactManifest) -> Native500MSummary:
    paths = _paths_by_role(root, manifest)
    events = pl.scan_parquet(paths["events_source"])
    event_summary = (
        events.select(
            pl.len().alias("rows"),
            pl.col("timestamp").max().alias("maximum_timestamp"),
        )
        .collect()
        .row(0, named=True)
    )
    cutoff = (
        int(event_summary["maximum_timestamp"]) - PROTOCOL.validation_interval_seconds
    )
    training_summary = (
        events.filter(pl.col("timestamp") < cutoff)
        .select(
            pl.len().alias("rows"),
            pl.col("uid").n_unique().alias("users"),
            pl.col("compact_item_id").min().alias("minimum_item"),
            pl.col("compact_item_id").max().alias("maximum_item"),
        )
        .collect()
        .row(0, named=True)
    )

    features = pl.read_parquet(paths["item_features"])
    if set(features.columns) != {
        "compact_item_id",
        "training_count",
        "artist_compact_ids",
        "album_compact_ids",
    }:
        raise ValueError("item-feature columns drifted")
    _validate_contiguous_ids(
        features["compact_item_id"], PROTOCOL.num_items, "item features"
    )
    histories = pl.read_parquet(paths["training_user_histories"])
    if histories["uid"].n_unique() != histories.height:
        raise ValueError("training user histories contain duplicate users")
    artist_vocab = pl.read_parquet(paths["artist_vocab"])
    album_vocab = pl.read_parquet(paths["album_vocab"])
    _validate_contiguous_ids(
        artist_vocab["artist_compact_id"], artist_vocab.height, "artist vocabulary"
    )
    _validate_contiguous_ids(
        album_vocab["album_compact_id"], album_vocab.height, "album vocabulary"
    )
    if (
        artist_vocab["raw_artist_id"].n_unique() != artist_vocab.height
        or album_vocab["raw_album_id"].n_unique() != album_vocab.height
    ):
        raise ValueError("metadata vocabularies contain duplicate raw IDs")
    if features["training_count"].min() < 0:
        raise ValueError("item features contain a negative training count")
    artist_lengths = features["artist_compact_ids"].list.len()
    album_lengths = features["album_compact_ids"].list.len()
    validate_training_aggregates(
        events,
        features.select("compact_item_id", "training_count"),
        histories.select("uid", "training_history_length"),
        cutoff_timestamp=cutoff,
        num_items=PROTOCOL.num_items,
    )
    summary = Native500MSummary(
        num_items=features.height,
        filtered_event_count=PROTOCOL.filtered_event_count,
        filtered_user_count=PROTOCOL.filtered_user_count,
        remapped_event_count=int(event_summary["rows"]),
        remapped_user_count=PROTOCOL.remapped_user_count,
        training_interaction_count=int(training_summary["rows"]),
        training_user_count=int(training_summary["users"]),
        evaluation_user_count=PROTOCOL.evaluation_user_count,
        content_width=PROTOCOL.content_width,
        validation_cutoff_timestamp=cutoff,
        artist_vocab_size=artist_vocab.height,
        album_vocab_size=album_vocab.height,
        artist_unknown_rate=float((artist_lengths == 0).mean()),
        album_unknown_rate=float((album_lengths == 0).mean()),
        artist_max_cardinality=int(artist_lengths.max()),
        album_max_cardinality=int(album_lengths.max()),
    )
    if (
        training_summary["minimum_item"] < 1
        or training_summary["maximum_item"] > PROTOCOL.num_items
        or features["training_count"].sum() != summary.training_interaction_count
        or histories.height != summary.training_user_count
        or histories["training_history_length"].sum()
        != summary.training_interaction_count
    ):
        raise ValueError("native-500M training split or feature counts drifted")
    _validate_feature_ids(
        features["artist_compact_ids"], summary.artist_vocab_size, "artist"
    )
    _validate_feature_ids(
        features["album_compact_ids"], summary.album_vocab_size, "album"
    )
    return validate_native500m_summary(summary)


def _inspect_dataset(root: Path, manifest: ArtifactManifest) -> Native500MSummary:
    paths = _paths_by_role(root, manifest)
    remap = pl.read_parquet(paths["compact_remap"], columns=["compact_id"])
    _validate_contiguous_ids(remap["compact_id"], PROTOCOL.num_items, "item remap")
    filtered = pl.scan_parquet(paths["filtered_events"])
    remapped = pl.scan_parquet(paths["remapped_events"])
    filtered_summary = (
        filtered.select(
            pl.len().alias("rows"),
            pl.col("uid").n_unique().alias("users"),
            pl.col("timestamp").max().alias("maximum_timestamp"),
        )
        .collect()
        .row(0, named=True)
    )
    remapped_summary = (
        remapped.select(
            pl.len().alias("rows"),
            pl.col("uid").n_unique().alias("users"),
            pl.col("timestamp").max().alias("maximum_timestamp"),
            pl.col("compact_item_id").min().alias("minimum_item"),
            pl.col("compact_item_id").max().alias("maximum_item"),
            pl.col("compact_item_id").n_unique().alias("items"),
        )
        .collect()
        .row(0, named=True)
    )
    cutoff = (
        int(remapped_summary["maximum_timestamp"])
        - PROTOCOL.validation_interval_seconds
    )
    training = (
        remapped.filter(pl.col("timestamp") < cutoff)
        .select(
            pl.len().alias("rows"),
            pl.col("uid").n_unique().alias("users"),
        )
        .collect()
        .row(0, named=True)
    )
    user_split = remapped.group_by("uid").agg(
        (pl.col("timestamp") < cutoff).sum().alias("training_events"),
        (pl.col("timestamp") >= cutoff).sum().alias("validation_events"),
    )
    evaluation_user_count = (
        user_split.filter(
            (pl.col("training_events") >= PROTOCOL.minimum_query_events)
            & (pl.col("validation_events") >= 1)
        )
        .select(pl.len())
        .collect()
        .item()
    )
    summary = Native500MSummary(
        num_items=int(remapped_summary["items"]),
        filtered_event_count=int(filtered_summary["rows"]),
        filtered_user_count=int(filtered_summary["users"]),
        remapped_event_count=int(remapped_summary["rows"]),
        remapped_user_count=int(remapped_summary["users"]),
        training_interaction_count=int(training["rows"]),
        training_user_count=int(training["users"]),
        evaluation_user_count=int(evaluation_user_count),
        content_width=PROTOCOL.content_width,
        validation_cutoff_timestamp=cutoff,
        artist_vocab_size=25007,
        album_vocab_size=138834,
        artist_unknown_rate=0.0012709952528327307,
        album_unknown_rate=0.0014425796119651494,
        artist_max_cardinality=2,
        album_max_cardinality=241,
    )
    if (
        filtered_summary["maximum_timestamp"] != remapped_summary["maximum_timestamp"]
        or remapped_summary["minimum_item"] != 1
        or remapped_summary["maximum_item"] != PROTOCOL.num_items
    ):
        raise ValueError("native-500M dataset remap or split identity drifted")
    return validate_native500m_summary(summary)


def _validate_contiguous_ids(
    values: pl.Series,
    expected_count: int,
    label: str,
) -> None:
    cast = values.cast(pl.Int64)
    expected = pl.Series(values.name, range(1, expected_count + 1), dtype=pl.Int64)
    if (
        len(cast) != expected_count
        or cast.n_unique() != expected_count
        or not cast.equals(expected)
    ):
        raise ValueError(f"{label} IDs must be contiguous and ordered 1..N")


def _validate_feature_ids(values: pl.Series, maximum: int, label: str) -> None:
    valid = values.list.eval((pl.element() >= 1) & (pl.element() <= maximum)).list.all()
    if not valid.all():
        raise ValueError(f"{label} feature IDs are outside the compact vocabulary")


def _paths_by_role(
    root: Path,
    manifest: ArtifactManifest,
) -> dict[str, Path]:
    resolved = root.resolve()
    return {binding.role: resolved / binding.path for binding in manifest.artifacts}


def _expected_summary() -> Native500MSummary:
    return Native500MSummary(
        num_items=PROTOCOL.num_items,
        filtered_event_count=PROTOCOL.filtered_event_count,
        filtered_user_count=PROTOCOL.filtered_user_count,
        remapped_event_count=PROTOCOL.remapped_event_count,
        remapped_user_count=PROTOCOL.remapped_user_count,
        training_interaction_count=PROTOCOL.training_interaction_count,
        training_user_count=PROTOCOL.training_user_count,
        evaluation_user_count=PROTOCOL.evaluation_user_count,
        content_width=PROTOCOL.content_width,
        validation_cutoff_timestamp=PROTOCOL.validation_cutoff_timestamp,
        artist_vocab_size=25007,
        album_vocab_size=138834,
        artist_unknown_rate=0.0012709952528327307,
        album_unknown_rate=0.0014425796119651494,
        artist_max_cardinality=2,
        album_max_cardinality=241,
    )


def _binding_from_dict(value: object) -> ArtifactBinding:
    if not isinstance(value, dict) or set(value) != {
        "role",
        "path",
        "size_bytes",
        "sha256",
    }:
        raise ValueError("artifact binding does not match the closed schema")
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
