from __future__ import annotations

from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import polars as pl

from experiments.g4_future_items.report.native500m_evidence import canonical_bytes


_OBJECTIVES = ("control_next_item", "rq1_24h", "rq2_next10")
_WINDOW_SECONDS = 24 * 60 * 60


@dataclass(frozen=True)
class TargetEvent:
    timestamp: int
    item_id: int


@dataclass
class _Summary:
    prefixes: int = 0
    eligible: int = 0
    fallback: int = 0
    candidate_memberships: int = 0
    unique_memberships: int = 0
    candidate_count_histogram: Counter[int] = field(default_factory=Counter)
    unique_count_histogram: Counter[int] = field(default_factory=Counter)
    distance_sum: float = 0.0
    distance_minimum: int | None = None
    distance_maximum: int | None = None
    distance_bins: Counter[str] = field(default_factory=Counter)
    rank_sum: float = 0.0
    rank_minimum: int | None = None
    rank_maximum: int | None = None
    rank_bins: Counter[str] = field(default_factory=Counter)

    def add(
        self,
        events: Sequence[TargetEvent],
        prefix_position: int,
        positions: Iterable[int],
        *,
        eligible: bool,
        fallback: bool,
    ) -> None:
        positions = tuple(positions)
        candidates = [events[position] for position in positions]
        unique_count = len({event.item_id for event in candidates})
        candidate_count = len(candidates)
        if candidate_count < 1:
            raise ValueError("native-500M target set is empty")
        self.prefixes += 1
        self.eligible += int(eligible)
        self.fallback += int(fallback)
        self.candidate_memberships += candidate_count
        self.unique_memberships += unique_count
        self.candidate_count_histogram[candidate_count] += 1
        self.unique_count_histogram[unique_count] += 1
        weight = 1 / candidate_count
        prefix_timestamp = events[prefix_position].timestamp
        for position, event in zip(positions, candidates, strict=True):
            distance = event.timestamp - prefix_timestamp
            rank = position - prefix_position
            if distance < 0 or rank < 1:
                raise ValueError("native-500M target does not follow its prefix")
            self.distance_sum += weight * distance
            self.distance_minimum = _minimum(self.distance_minimum, distance)
            self.distance_maximum = _maximum(self.distance_maximum, distance)
            self.distance_bins[_distance_bin(distance)] += weight
            self.rank_sum += weight * rank
            self.rank_minimum = _minimum(self.rank_minimum, rank)
            self.rank_maximum = _maximum(self.rank_maximum, rank)
            self.rank_bins[_rank_bin(rank)] += weight

    def document(self) -> dict[str, Any]:
        if not self.prefixes:
            raise ValueError("native-500M target statistics have no prefixes")
        return {
            "prefix_positive_pairs": self.prefixes,
            "eligible_prefixes": self.eligible,
            "eligibility_rate": self.eligible / self.prefixes,
            "fallback_prefixes": self.fallback,
            "fallback_rate": self.fallback / self.prefixes,
            "post_fallback_empty_prefixes": 0,
            "candidate_occurrences": _cardinality(self.candidate_count_histogram),
            "acceptable_unique_items": _cardinality(self.unique_count_histogram),
            "duplicate_candidate_membership_rate": 1
            - self.unique_memberships / self.candidate_memberships,
            "sampled_target_distance_seconds": {
                "sampling": "uniform_over_candidate_occurrences_per_prefix",
                "minimum": self.distance_minimum,
                "maximum": self.distance_maximum,
                "expected_mean": self.distance_sum / self.prefixes,
                "bins": _weighted_bins(self.distance_bins, self.prefixes),
            },
            "sampled_target_event_rank": {
                "sampling": "uniform_over_candidate_occurrences_per_prefix",
                "minimum": self.rank_minimum,
                "maximum": self.rank_maximum,
                "expected_mean": self.rank_sum / self.prefixes,
                "bins": _weighted_bins(self.rank_bins, self.prefixes),
            },
        }


def objective_summaries(
    events_by_user: dict[int, Sequence[TargetEvent]],
) -> dict[str, dict[str, Any]]:
    summaries = {objective: _Summary() for objective in _OBJECTIVES}
    for events in events_by_user.values():
        timestamps = [event.timestamp for event in events]
        for prefix in range(len(events) - 1):
            summaries["control_next_item"].add(
                events, prefix, (prefix + 1,), eligible=True, fallback=False
            )
            summaries["rq2_next10"].add(
                events,
                prefix,
                range(prefix + 1, min(prefix + 11, len(events))),
                eligible=True,
                fallback=False,
            )
            strict_start = bisect_right(timestamps, timestamps[prefix], prefix + 1)
            window_end = bisect_right(
                timestamps, timestamps[prefix] + _WINDOW_SECONDS, strict_start
            )
            eligible = strict_start < window_end
            summaries["rq1_24h"].add(
                events,
                prefix,
                range(strict_start, window_end) if eligible else (prefix + 1,),
                eligible=eligible,
                fallback=not eligible,
            )
    return {objective: summary.document() for objective, summary in summaries.items()}


def next_item_window_population(
    user_lengths: Iterable[int], *, max_seq_len: int, min_seq_len: int
) -> dict[str, int]:
    if max_seq_len < 1 or min_seq_len != 2:
        raise ValueError(
            "native-500M window accounting requires max_seq_len >= 1 "
            "and min_seq_len == 2"
        )
    training_sequences = 0
    objective_prefixes = 0
    for length in user_lengths:
        if length < 1:
            raise ValueError("native-500M user sequences must be nonempty")
        prefixes = length - 1
        objective_prefixes += prefixes
        training_sequences += math.ceil(prefixes / max_seq_len)
    training_targets = objective_prefixes + training_sequences
    return {
        "training_sequences": training_sequences,
        "bos_next_item_targets": training_sequences,
        "objective_prefix_positive_pairs": objective_prefixes,
        "training_targets": training_targets,
        "training_tokens": training_targets + 2 * training_sequences,
    }


def build_native500m_target_statistics(
    repo_root: Path, *, control_contract_path: Path
) -> dict[str, Any]:
    root = repo_root.resolve(strict=True)
    contract_path = control_contract_path.resolve(strict=True)
    contract = _document(contract_path)
    data_identity = contract["data_identity"]
    main_identity = data_identity["main"]
    main_path = Path(main_identity["path"])
    if _file_fact(main_path) != main_identity:
        raise ValueError("native-500M target-statistics data identity differs")
    cutoff = int(data_identity["split_cutoff_timestamp"])
    frame = (
        pl.scan_parquet(main_path)
        .filter((pl.col("event_type") == "like") & (pl.col("timestamp") < cutoff))
        .select("uid", "timestamp", "compact_item_id")
        .collect(engine="streaming")
        .sort(["uid", "timestamp"], maintain_order=True)
    )
    events_by_user: dict[int, list[TargetEvent]] = {}
    for user_id, rows in frame.group_by("uid", maintain_order=True):
        uid = int(user_id[0]) if isinstance(user_id, tuple) else int(user_id)
        events_by_user[uid] = [
            TargetEvent(int(timestamp), int(item_id))
            for timestamp, item_id in rows.select(
                "timestamp", "compact_item_id"
            ).iter_rows()
        ]
    objectives = objective_summaries(events_by_user)
    metadata = _document(contract_path.parent / "training_metadata.json")
    invariants = metadata["transfer_invariants"]
    expected_training_semantics = {
        "bos": True,
        "cls_token": True,
        "cls_token_mode": "end_only",
        "max_seq_len": 100,
    }
    if (
        any(
            invariants.get(name) != value
            for name, value in expected_training_semantics.items()
        )
        or metadata.get("training_semantics_revision") != 2
    ):
        raise ValueError("native-500M target-statistics training semantics differ")
    max_seq_len = int(invariants["max_seq_len"])
    min_seq_len = 2
    window_population = next_item_window_population(
        (len(events) for events in events_by_user.values()),
        max_seq_len=max_seq_len,
        min_seq_len=min_seq_len,
    )
    population = {
        "training_like_events": frame.height,
        "training_users": len(events_by_user),
        "max_seq_len": max_seq_len,
        "min_seq_len": min_seq_len,
        **window_population,
    }
    if any(
        objective["prefix_positive_pairs"]
        != population["objective_prefix_positive_pairs"]
        for objective in objectives.values()
    ):
        raise ValueError("native-500M objectives have different training populations")
    if metadata["targets_per_epoch"] != population["training_targets"]:
        raise ValueError("native-500M target population differs from training metadata")
    if metadata["tokens_per_epoch"] != population["training_tokens"]:
        raise ValueError("native-500M token population differs from training metadata")
    training_source_paths = (
        "dcn/config/generation.py",
        "dcn/data/sequence_dataset.py",
        "experiments/g4_future_items/configs/native500m.py",
        "experiments/g4_future_items/native500m_targets.py",
        "experiments/g4_future_items/targets.py",
    )
    training_sources = contract["source_closure"]["sources"]
    snapshot_root = (
        root
        / "generated/g4_native500m_source_snapshots"
        / contract["source_closure"]["sha256"]
    )
    source_facts = {_relative(root, Path(__file__)): _file_fact(Path(__file__))}
    for relative_path in training_source_paths:
        snapshot_path = snapshot_root / relative_path
        fact = _file_fact(snapshot_path)
        if training_sources.get(relative_path) != fact["sha256"]:
            raise ValueError(
                f"native-500M target-statistics source differs: {relative_path}"
            )
        source_facts[f"training_snapshot/{relative_path}"] = fact
    return {
        "schema_version": 1,
        "kind": "g4_native500m_target_statistics",
        "data_identity": data_identity,
        "control_contract": _file_fact(contract_path),
        "training_metadata": _file_fact(
            contract_path.parent / "training_metadata.json"
        ),
        "sources": source_facts,
        "definitions": {
            "training_split": "event_type == 'like' and timestamp < split_cutoff_timestamp",
            "training_window": "next_item windows of at most max_seq_len transitions, with the boundary event shared by adjacent windows and length-one tails removed",
            "bos_next_item_target": "one common next-item target per retained training window",
            "objective_prefix": "every training like event with a later training like event",
            "rq1_24h": "strictly later timestamps within 86400 seconds, falling back to next event",
            "rq2_next10": "next ten event occurrences or all remaining occurrences",
            "acceptable_items": "unique compact item ids among candidate occurrences",
        },
        "population": population,
        "objectives": objectives,
    }


def write_native500m_target_statistics(
    artifact_path: Path, *, repo_root: Path, control_contract_path: Path
) -> str:
    payload = canonical_bytes(
        build_native500m_target_statistics(
            repo_root, control_contract_path=control_contract_path
        )
    )
    digest = hashlib.sha256(payload).hexdigest()
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    _write_immutable(artifact_path, payload)
    _write_immutable(artifact_path.with_suffix(".sha256"), digest.encode())
    return digest


def _cardinality(histogram: Counter[int]) -> dict[str, Any]:
    count = sum(histogram.values())
    total = sum(value * frequency for value, frequency in histogram.items())
    return {
        "minimum": min(histogram),
        "maximum": max(histogram),
        "mean": total / count,
        "p50": _quantile(histogram, 0.50),
        "p90": _quantile(histogram, 0.90),
        "p99": _quantile(histogram, 0.99),
        "histogram": {str(value): histogram[value] for value in sorted(histogram)},
    }


def _quantile(histogram: Counter[int], quantile: float) -> int:
    rank = max(1, math.ceil(quantile * sum(histogram.values())))
    cumulative = 0
    for value in sorted(histogram):
        cumulative += histogram[value]
        if cumulative >= rank:
            return value
    raise RuntimeError("cardinality histogram is incomplete")


def _distance_bin(distance: int) -> str:
    if distance == 0:
        return "zero"
    if distance <= 6 * 60 * 60:
        return "0_6h"
    if distance <= 24 * 60 * 60:
        return "6_24h"
    if distance <= 3 * 24 * 60 * 60:
        return "1_3d"
    if distance <= 7 * 24 * 60 * 60:
        return "3_7d"
    return "7d_plus"


def _rank_bin(rank: int) -> str:
    if rank == 1:
        return "rank_1"
    if rank <= 5:
        return "rank_2_5"
    if rank <= 10:
        return "rank_6_10"
    return "rank_11_plus"


def _weighted_bins(values: Counter[str], denominator: int) -> dict[str, float]:
    return {name: values[name] / denominator for name in sorted(values)}


def _minimum(current: int | None, value: int) -> int:
    return value if current is None else min(current, value)


def _maximum(current: int | None, value: int) -> int:
    return value if current is None else max(current, value)


def _document(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected an object: {path}")
    return value


def _file_fact(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    stat = resolved.stat()
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(resolved),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest.hexdigest(),
    }


def _relative(root: Path, path: Path) -> str:
    return path.resolve(strict=True).relative_to(root).as_posix()


def _write_immutable(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"native-500M target statistics changed: {path}")
        return
    path.write_bytes(content)
