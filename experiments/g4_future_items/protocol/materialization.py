from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

import numpy as np
import polars as pl

from experiments.g4_future_items.selectors import LikeEvent, ListenEvent


TargetObjective = Literal[
    "control_next_item",
    "rq1_24h",
    "rq2_next10",
    "rq3_deterministic_hard",
    "rq3_learned_hard",
    "rq3_learned_proportional",
]
RQ3Objective = Literal[
    "rq3_deterministic_hard",
    "rq3_learned_hard",
    "rq3_learned_proportional",
]
OBJECTIVE_IDS = frozenset(
    {
        "control_next_item",
        "rq1_24h",
        "rq2_next10",
        "rq3_deterministic_hard",
        "rq3_learned_hard",
        "rq3_learned_proportional",
    }
)


@dataclass(frozen=True)
class MaterializationQuery:
    uid: int
    prefix_timestamp: int
    prefix_item_id: int
    next_item: int


@dataclass(frozen=True, order=True)
class CandidateOccurrence:
    timestamp: int
    item_id: int


@dataclass(frozen=True)
class CandidatePeriod:
    start: int
    end: int
    score: float
    occurrences: tuple[CandidateOccurrence, ...]

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise ValueError("candidate period end must be greater than start")
        if not math.isfinite(self.score) or not 0.0 <= self.score <= 1.0:
            raise ValueError("candidate period score must be finite and in [0, 1]")


@dataclass(frozen=True)
class MaterializedTarget:
    target_item_id: int
    acceptable_item_ids: frozenset[int]
    selected_period_starts: tuple[int, ...]
    used_fallback: bool
    rng_seed: int


def target_seed(
    query: MaterializationQuery,
    training_seed: int,
    epoch: int,
    objective_id: TargetObjective,
) -> int:
    if objective_id not in OBJECTIVE_IDS:
        raise ValueError(f"unknown objective id {objective_id!r}")
    payload = json.dumps(
        [
            "g4-target-v1",
            int(training_seed),
            int(epoch),
            objective_id,
            int(query.uid),
            int(query.prefix_timestamp),
            int(query.prefix_item_id),
        ],
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def materialize_target(
    query: MaterializationQuery,
    periods: Sequence[CandidatePeriod],
    *,
    objective_id: RQ3Objective,
    period_count: int,
    training_seed: int,
    epoch: int,
) -> MaterializedTarget:
    if objective_id not in {
        "rq3_deterministic_hard",
        "rq3_learned_hard",
        "rq3_learned_proportional",
    }:
        raise ValueError(f"unsupported materialization objective {objective_id!r}")
    if period_count not in {1, 2, 4}:
        raise ValueError("period count must be 1, 2, or 4")
    seed = target_seed(query, training_seed, epoch, objective_id)
    eligible = [
        period for period in periods if period.score > 0.0 and period.occurrences
    ]
    if not eligible:
        return MaterializedTarget(
            target_item_id=query.next_item,
            acceptable_item_ids=frozenset({query.next_item}),
            selected_period_starts=(),
            used_fallback=True,
            rng_seed=seed,
        )

    generator = np.random.Generator(np.random.PCG64(seed))
    if objective_id == "rq3_learned_proportional":
        ordered = sorted(eligible, key=lambda period: (period.start, period.end))
        probabilities = np.asarray(
            [min(1.0, max(1e-6, period.score)) for period in ordered],
            dtype=np.float64,
        )
        probabilities /= probabilities.sum()
        selected_indices = generator.choice(
            len(ordered),
            size=min(period_count, len(ordered)),
            replace=False,
            p=probabilities,
        )
        selected = sorted(
            (ordered[int(index)] for index in np.atleast_1d(selected_indices)),
            key=lambda period: (period.start, period.end),
        )
        acceptable_periods = ordered
    else:
        selected = sorted(
            eligible,
            key=lambda period: (-period.score, period.start, period.end),
        )[:period_count]
        acceptable_periods = selected

    occurrences = sorted(
        (
            (period.start, occurrence.timestamp, occurrence.item_id)
            for period in selected
            for occurrence in period.occurrences
        ),
        key=lambda value: (value[0], value[1], value[2]),
    )
    selected_occurrence = occurrences[int(generator.integers(len(occurrences)))]
    acceptable_items = frozenset(
        occurrence.item_id
        for period in acceptable_periods
        for occurrence in period.occurrences
    )
    return MaterializedTarget(
        target_item_id=selected_occurrence[2],
        acceptable_item_ids=acceptable_items,
        selected_period_starts=tuple(period.start for period in selected),
        used_fallback=False,
        rng_seed=seed,
    )


@dataclass(frozen=True)
class SelectorInputPaths:
    control_likes: Path
    raw_events: Path
    item_id_remap: Path
    compact_embeddings: Path

    def validate(self) -> None:
        missing = [str(path) for path in self.paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing selector inputs: {missing}")

    @property
    def paths(self) -> tuple[Path, Path, Path, Path]:
        return (
            self.control_likes,
            self.raw_events,
            self.item_id_remap,
            self.compact_embeddings,
        )


@dataclass(frozen=True)
class SelectorInputFrames:
    likes: pl.LazyFrame
    listens: pl.LazyFrame
    compact_embeddings: pl.LazyFrame


@dataclass(frozen=True)
class SelectorUserEvents:
    uid: int
    likes: tuple[LikeEvent, ...]
    listens: tuple[ListenEvent, ...]


def scan_selector_inputs(
    paths: SelectorInputPaths,
    *,
    start_timestamp: int,
    cutoff_timestamp: int,
) -> SelectorInputFrames:
    if cutoff_timestamp <= start_timestamp:
        raise ValueError("cutoff timestamp must be greater than start timestamp")
    paths.validate()
    likes = pl.scan_parquet(paths.control_likes).filter(
        pl.col("timestamp").is_between(start_timestamp, cutoff_timestamp, closed="left")
    )
    schema = likes.collect_schema()
    required_like_columns = {
        "uid",
        "item_id",
        "compact_item_id",
        "event_type",
        "timestamp",
        "artist_id",
        "album_id",
    }
    missing = required_like_columns - set(schema.names())
    if missing:
        raise ValueError(f"control likes cache lacks columns: {sorted(missing)}")
    event_types = (
        likes.select("event_type")
        .unique()
        .collect(engine="streaming")["event_type"]
        .to_list()
    )
    if event_types != ["like"]:
        raise ValueError(
            "control selector source must be the likes-only core5 known-items cache"
        )

    retained_users = likes.select("uid").unique()
    retained_items = likes.select("compact_item_id", "artist_id", "album_id").unique(
        subset=["compact_item_id"]
    )
    remap = pl.scan_parquet(paths.item_id_remap)
    remap_compact_column = _compact_id_column(
        remap.collect_schema(), paths.item_id_remap
    )
    remap = remap.select(
        "item_id", pl.col(remap_compact_column).alias("compact_item_id")
    )
    listens = (
        pl.scan_parquet(paths.raw_events)
        .filter(
            (pl.col("event_type") == "listen")
            & pl.col("timestamp").is_between(
                start_timestamp, cutoff_timestamp, closed="left"
            )
        )
        .select("uid", "item_id", "timestamp")
        .join(retained_users, on="uid", how="inner")
        .join(remap, on="item_id", how="inner")
        .join(retained_items, on="compact_item_id", how="inner")
        .select("uid", "timestamp", "compact_item_id", "artist_id")
    )
    embeddings = pl.scan_parquet(paths.compact_embeddings)
    embedding_compact_column = _compact_id_column(
        embeddings.collect_schema(), paths.compact_embeddings
    )
    if embedding_compact_column != "compact_item_id":
        embeddings = embeddings.rename({embedding_compact_column: "compact_item_id"})
    return SelectorInputFrames(
        likes=likes,
        listens=listens,
        compact_embeddings=embeddings,
    )


def iter_selector_users(
    paths: SelectorInputPaths,
    *,
    start_timestamp: int,
    cutoff_timestamp: int,
    batch_rows: int = 65_536,
    selected_user_ids: Sequence[int] | None = None,
) -> Iterable[SelectorUserEvents]:
    if batch_rows <= 0:
        raise ValueError("batch rows must be positive")
    frames = scan_selector_inputs(
        paths,
        start_timestamp=start_timestamp,
        cutoff_timestamp=cutoff_timestamp,
    )
    embedding_frame = frames.compact_embeddings.collect(engine="streaming")
    embeddings = {
        int(compact_id): np.asarray(embedding, dtype=np.float64)
        for compact_id, embedding in embedding_frame.select(
            "compact_item_id", "normalized_embed"
        ).iter_rows()
    }

    import duckdb

    control = _sql_path(paths.control_likes)
    raw = _sql_path(paths.raw_events)
    remap = _sql_path(paths.item_id_remap)
    remap_column = _compact_id_column(
        pl.scan_parquet(paths.item_id_remap).collect_schema(), paths.item_id_remap
    )
    selected_users_join = ""
    connection = duckdb.connect()
    if selected_user_ids is not None:
        selected_users = pl.DataFrame(
            {
                "uid": pl.Series(
                    "uid", sorted({int(uid) for uid in selected_user_ids}), pl.Int64
                )
            }
        )
        connection.register("selected_users", selected_users)
        selected_users_join = "JOIN selected_users selected USING (uid)"
    query = f"""
        WITH likes_source AS (
            SELECT *, row_number() OVER () - 1 AS source_order
            FROM read_parquet('{control}')
            {selected_users_join}
            WHERE timestamp >= {int(start_timestamp)}
              AND timestamp < {int(cutoff_timestamp)}
        ),
        retained_users AS (
            SELECT DISTINCT uid FROM likes_source
        ),
        retained_items AS (
            SELECT compact_item_id,
                   first(artist_id) AS artist_id,
                   first(album_id) AS album_id
            FROM likes_source
            GROUP BY compact_item_id
        ),
        mapped_listens AS (
            SELECT raw.uid,
                   raw.timestamp,
                   remap.{remap_column}::BIGINT AS compact_item_id,
                   items.artist_id,
                   items.album_id,
                   0::BIGINT AS source_order,
                   0::UTINYINT AS event_kind
            FROM read_parquet('{raw}') raw
            JOIN retained_users users USING (uid)
            JOIN read_parquet('{remap}') remap USING (item_id)
            JOIN retained_items items
              ON items.compact_item_id = remap.{remap_column}
            WHERE raw.event_type = 'listen'
              AND raw.timestamp >= {int(start_timestamp)}
              AND raw.timestamp < {int(cutoff_timestamp)}
        ),
        retained_likes AS (
            SELECT uid,
                   timestamp,
                   compact_item_id::BIGINT AS compact_item_id,
                   artist_id,
                   album_id,
                   source_order,
                   1::UTINYINT AS event_kind
            FROM likes_source
        )
        SELECT * FROM (
            SELECT * FROM mapped_listens
            UNION ALL
            SELECT * FROM retained_likes
        )
        ORDER BY uid, timestamp, event_kind, source_order
    """
    try:
        reader = connection.execute(query).fetch_record_batch(rows_per_batch=batch_rows)
        active_uid: int | None = None
        likes: list[LikeEvent] = []
        listens: list[ListenEvent] = []
        for batch in reader:
            columns = batch.to_pydict()
            for index, uid_value in enumerate(columns["uid"]):
                uid = int(uid_value)
                if active_uid is not None and uid != active_uid:
                    yield SelectorUserEvents(active_uid, tuple(likes), tuple(listens))
                    likes.clear()
                    listens.clear()
                active_uid = uid
                artist_ids = tuple(int(value) for value in columns["artist_id"][index])
                if int(columns["event_kind"][index]) == 1:
                    compact_item_id = int(columns["compact_item_id"][index])
                    likes.append(
                        LikeEvent(
                            uid=uid,
                            timestamp=int(columns["timestamp"][index]),
                            item_id=compact_item_id,
                            artist_ids=artist_ids,
                            album_ids=tuple(
                                int(value) for value in columns["album_id"][index]
                            ),
                            content_embedding=embeddings.get(compact_item_id),
                        )
                    )
                else:
                    listens.append(
                        ListenEvent(
                            uid=uid,
                            timestamp=int(columns["timestamp"][index]),
                            artist_ids=artist_ids,
                        )
                    )
        if active_uid is not None:
            yield SelectorUserEvents(active_uid, tuple(likes), tuple(listens))
    finally:
        connection.close()


def _compact_id_column(schema: pl.Schema, path: Path) -> str:
    for column in ("compact_item_id", "compact_id"):
        if column in schema:
            return column
    raise ValueError(f"{path} lacks compact_id")


def _sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


PERIOD_ARTIFACT_VERSION = "g4-period-artifact-v1"
DEFAULT_PERIOD_ARTIFACT_ROOT = (
    Path(__file__).resolve().parents[3] / "generated/g4_selector_artifacts"
)
_ARRAY_DTYPES = {
    "query_uid": "<i8",
    "query_timestamp": "<i8",
    "query_item": "<i8",
    "query_position": "<i8",
    "query_next_item": "<i8",
    "query_fold": "u1",
    "query_period_offsets": "<i8",
    "period_start": "<i8",
    "period_end": "<i8",
    "period_score": "<f8",
    "period_occurrence_offsets": "<i8",
    "occurrence_timestamp": "<i8",
    "occurrence_item": "<i8",
    "occurrence_position": "<i8",
}


@dataclass(frozen=True, order=True)
class ScoredOccurrence:
    timestamp: int
    item_id: int
    occurrence_position: int


@dataclass(frozen=True)
class ScoredPeriod:
    start: int
    end: int
    score: float
    occurrences: tuple[ScoredOccurrence, ...]


@dataclass(frozen=True)
class ScoredQuery:
    uid: int
    prefix_timestamp: int
    prefix_item_id: int
    occurrence_position: int
    next_item: int
    fold: int
    periods: tuple[ScoredPeriod, ...]

    @property
    def key(self) -> tuple[int, int, int, int]:
        return (
            self.uid,
            self.prefix_timestamp,
            self.prefix_item_id,
            self.occurrence_position,
        )


@dataclass(frozen=True)
class PeriodArtifactIdentity:
    sha256: str
    path: Path
    query_count: int
    period_count: int
    occurrence_count: int
    logical_bytes: int


def write_period_artifact(
    queries: Iterable[ScoredQuery],
    *,
    selector_kind: Literal["deterministic", "learned"],
    selected_configuration: Mapping[str, Any],
    provenance: Mapping[str, Any],
    cost: Mapping[str, Any],
    output_root: Path = DEFAULT_PERIOD_ARTIFACT_ROOT,
) -> PeriodArtifactIdentity:
    output_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".g4-period-", dir=output_root))
    sinks = {
        name: _ArraySink(temporary / f"{name}.bin", np.dtype(dtype))
        for name, dtype in _ARRAY_DTYPES.items()
    }
    query_count = 0
    period_count = 0
    occurrence_count = 0
    previous_key: tuple[int, int, int, int] | None = None
    try:
        sinks["query_period_offsets"].append(0)
        sinks["period_occurrence_offsets"].append(0)
        for query in queries:
            if previous_key is not None and query.key <= previous_key:
                raise ValueError("artifact queries must be strictly key-sorted")
            previous_key = query.key
            if query.fold not in range(5):
                raise ValueError("query fold must be in [0, 5)")
            for name, value in (
                ("query_uid", query.uid),
                ("query_timestamp", query.prefix_timestamp),
                ("query_item", query.prefix_item_id),
                ("query_position", query.occurrence_position),
                ("query_next_item", query.next_item),
                ("query_fold", query.fold),
            ):
                sinks[name].append(value)
            previous_period: tuple[int, int] | None = None
            for period in query.periods:
                period_key = (period.start, period.end)
                if previous_period is not None and period_key <= previous_period:
                    raise ValueError("query periods must be strictly time-sorted")
                previous_period = period_key
                if period.end <= period.start:
                    raise ValueError("period end must be greater than start")
                if not math.isfinite(period.score) or not 0.0 < period.score <= 1.0:
                    raise ValueError("serialized periods require a score in (0, 1]")
                sinks["period_start"].append(period.start)
                sinks["period_end"].append(period.end)
                sinks["period_score"].append(period.score)
                for occurrence in sorted(period.occurrences):
                    sinks["occurrence_timestamp"].append(occurrence.timestamp)
                    sinks["occurrence_item"].append(occurrence.item_id)
                    sinks["occurrence_position"].append(occurrence.occurrence_position)
                    occurrence_count += 1
                period_count += 1
                sinks["period_occurrence_offsets"].append(occurrence_count)
            query_count += 1
            sinks["query_period_offsets"].append(period_count)
        for sink in sinks.values():
            sink.close()
        arrays = {
            name: _array_identity(temporary / f"{name}.bin", dtype, sinks[name].count)
            for name, dtype in _ARRAY_DTYPES.items()
        }
        manifest = {
            "version": PERIOD_ARTIFACT_VERSION,
            "selector_kind": selector_kind,
            "selected_configuration": dict(selected_configuration),
            "fold_assignment": {
                "revision": "g4-fold-v1",
                "seed": 42,
                "folds": 5,
            },
            "counts": {
                "queries": query_count,
                "periods": period_count,
                "occurrences": occurrence_count,
            },
            "arrays": arrays,
            "provenance": dict(provenance),
            "cost": dict(cost),
            "logical_bytes": 0,
        }
        array_bytes = sum(identity["size"] for identity in arrays.values())
        while True:
            content = _canonical_bytes(manifest)
            logical_bytes = array_bytes + len(content)
            if manifest["logical_bytes"] == logical_bytes:
                break
            manifest["logical_bytes"] = logical_bytes
        digest = hashlib.sha256(content).hexdigest()
        (temporary / "manifest.json").write_bytes(content)
        _fsync_tree(temporary)
        destination = output_root / digest
        if destination.exists():
            existing = PeriodArtifact.open(destination, expected_sha256=digest)
            if existing.manifest != manifest:
                raise RuntimeError(f"existing selector artifact differs: {destination}")
            shutil.rmtree(temporary)
        else:
            os.replace(temporary, destination)
            _fsync_directory(output_root)
        return PeriodArtifactIdentity(
            sha256=digest,
            path=destination,
            query_count=query_count,
            period_count=period_count,
            occurrence_count=occurrence_count,
            logical_bytes=manifest["logical_bytes"],
        )
    except BaseException:
        for sink in sinks.values():
            sink.abort()
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


class PeriodArtifact:
    def __init__(
        self, path: Path, manifest: dict[str, Any], arrays: dict[str, np.ndarray]
    ):
        self.path = path
        self.manifest = manifest
        self.arrays = arrays

    @classmethod
    def open(
        cls,
        path: Path = DEFAULT_PERIOD_ARTIFACT_ROOT,
        *,
        expected_sha256: str,
    ) -> PeriodArtifact:
        path = Path(path)
        if path.name != expected_sha256:
            path = path / expected_sha256
        manifest_path = path / "manifest.json"
        content = manifest_path.read_bytes()
        if hashlib.sha256(content).hexdigest() != expected_sha256:
            raise ValueError("period artifact manifest SHA-256 differs")
        manifest = json.loads(content)
        if manifest.get("version") != PERIOD_ARTIFACT_VERSION:
            raise ValueError("period artifact version differs")
        if set(manifest.get("arrays", {})) != set(_ARRAY_DTYPES):
            raise ValueError("period artifact array schema differs")
        arrays: dict[str, np.ndarray] = {}
        for name, dtype in _ARRAY_DTYPES.items():
            identity = manifest["arrays"][name]
            array_path = path / identity["file"]
            if identity != _array_identity(
                array_path, dtype, int(identity["shape"][0])
            ):
                raise ValueError(f"period artifact array {name} differs")
            count = int(identity["shape"][0])
            arrays[name] = (
                np.memmap(array_path, dtype=dtype, mode="r", shape=(count,))
                if count
                else np.empty(0, dtype=dtype)
            )
        _validate_period_artifact_shapes(manifest, arrays)
        return cls(path, manifest, arrays)

    def lookup(
        self,
        uid: int,
        prefix_timestamp: int,
        prefix_item_id: int,
        occurrence_position: int,
    ) -> tuple[MaterializationQuery, tuple[CandidatePeriod, ...]]:
        index = self._query_index(
            uid, prefix_timestamp, prefix_item_id, occurrence_position
        )
        query = MaterializationQuery(
            uid=uid,
            prefix_timestamp=prefix_timestamp,
            prefix_item_id=prefix_item_id,
            next_item=int(self.arrays["query_next_item"][index]),
        )
        period_begin = int(self.arrays["query_period_offsets"][index])
        period_end = int(self.arrays["query_period_offsets"][index + 1])
        periods = []
        for period_index in range(period_begin, period_end):
            occurrence_begin = int(
                self.arrays["period_occurrence_offsets"][period_index]
            )
            occurrence_end = int(
                self.arrays["period_occurrence_offsets"][period_index + 1]
            )
            periods.append(
                CandidatePeriod(
                    start=int(self.arrays["period_start"][period_index]),
                    end=int(self.arrays["period_end"][period_index]),
                    score=float(self.arrays["period_score"][period_index]),
                    occurrences=tuple(
                        CandidateOccurrence(
                            timestamp=int(self.arrays["occurrence_timestamp"][offset]),
                            item_id=int(self.arrays["occurrence_item"][offset]),
                        )
                        for offset in range(occurrence_begin, occurrence_end)
                    ),
                )
            )
        return query, tuple(periods)

    def _query_index(
        self,
        uid: int,
        timestamp: int,
        item_id: int,
        position: int,
    ) -> int:
        begin, end = _equal_range(
            self.arrays["query_uid"], uid, 0, len(self.arrays["query_uid"])
        )
        begin, end = _equal_range(self.arrays["query_timestamp"], timestamp, begin, end)
        begin, end = _equal_range(self.arrays["query_item"], item_id, begin, end)
        begin, end = _equal_range(self.arrays["query_position"], position, begin, end)
        if end - begin != 1:
            raise KeyError((uid, timestamp, item_id, position))
        return begin


class _ArraySink:
    def __init__(self, path: Path, dtype: np.dtype[Any]):
        self.path = path
        self.dtype = dtype
        self.handle = path.open("wb")
        self.buffer: list[int | float] = []
        self.count = 0

    def append(self, value: int | float) -> None:
        self.buffer.append(value)
        if len(self.buffer) >= 65_536:
            self.flush()

    def flush(self) -> None:
        if self.buffer:
            values = np.asarray(self.buffer, dtype=self.dtype)
            self.handle.write(values.tobytes(order="C"))
            self.count += len(self.buffer)
            self.buffer.clear()

    def close(self) -> None:
        if self.handle.closed:
            return
        self.flush()
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()

    def abort(self) -> None:
        if not self.handle.closed:
            self.handle.close()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _array_identity(path: Path, dtype: str, count: int) -> dict[str, Any]:
    return {
        "file": path.name,
        "dtype": np.dtype(dtype).str,
        "shape": [count],
        "size": path.stat().st_size,
        "sha256": _file_sha256(path),
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_tree(path: Path) -> None:
    for child in path.iterdir():
        if child.is_file():
            with child.open("rb") as handle:
                os.fsync(handle.fileno())
    _fsync_directory(path)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_period_artifact_shapes(
    manifest: Mapping[str, Any], arrays: Mapping[str, np.ndarray]
) -> None:
    counts = manifest["counts"]
    query_count = int(counts["queries"])
    period_count = int(counts["periods"])
    occurrence_count = int(counts["occurrences"])
    for name in (
        "query_uid",
        "query_timestamp",
        "query_item",
        "query_position",
        "query_next_item",
        "query_fold",
    ):
        if len(arrays[name]) != query_count:
            raise ValueError(f"period artifact {name} shape differs")
    if len(arrays["query_period_offsets"]) != query_count + 1:
        raise ValueError("period artifact query offsets shape differs")
    for name in ("period_start", "period_end", "period_score"):
        if len(arrays[name]) != period_count:
            raise ValueError(f"period artifact {name} shape differs")
    if len(arrays["period_occurrence_offsets"]) != period_count + 1:
        raise ValueError("period artifact occurrence offsets shape differs")
    for name in ("occurrence_timestamp", "occurrence_item", "occurrence_position"):
        if len(arrays[name]) != occurrence_count:
            raise ValueError(f"period artifact {name} shape differs")
    if int(arrays["query_period_offsets"][-1]) != period_count:
        raise ValueError("period artifact query offsets do not terminate")
    if int(arrays["period_occurrence_offsets"][-1]) != occurrence_count:
        raise ValueError("period artifact occurrence offsets do not terminate")


def _equal_range(
    values: np.ndarray, target: int, begin: int, end: int
) -> tuple[int, int]:
    selected = values[begin:end]
    lower = begin + int(np.searchsorted(selected, target, side="left"))
    upper = begin + int(np.searchsorted(selected, target, side="right"))
    return lower, upper
