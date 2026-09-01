from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
import hashlib
import math
from pathlib import Path
from typing import Any, Literal, Sequence

import polars as pl

from experiments.g4_future_items.protocol.manifest import (
    canonical_bytes,
    canonical_sha256,
    load_strict_json,
)


ObjectiveId = Literal["control_next_item", "rq1_24h", "rq2_next10"]
_OBJECTIVES: tuple[ObjectiveId, ...] = (
    "control_next_item",
    "rq1_24h",
    "rq2_next10",
)
_WINDOW_SECONDS = 24 * 60 * 60
_DISTANCE_BINS = (
    ("zero", 0),
    ("0_6h", 6 * 60 * 60),
    ("6_24h", 24 * 60 * 60),
    ("1_3d", 3 * 24 * 60 * 60),
    ("3_7d", 7 * 24 * 60 * 60),
    ("7d_plus", None),
)
_SOURCE_PATHS = (
    "experiments/g4_future_items/report/target_statistics.py",
    "experiments/g4_future_items/report/slices.py",
    "experiments/g4_future_items/targets.py",
    "experiments/g4_future_items/configs/treatments.py",
    "experiments/g4_future_items/protocol/plan.md",
)


@dataclass(frozen=True)
class TargetEvent:
    timestamp: int
    item_id: int


def candidate_indices(
    events: Sequence[TargetEvent], prefix_position: int, objective_id: ObjectiveId
) -> tuple[tuple[int, ...], bool]:
    if not 0 <= prefix_position < len(events) - 1:
        raise ValueError("a causal prefix must have a following event")
    if objective_id == "control_next_item":
        return (prefix_position + 1,), False
    if objective_id == "rq2_next10":
        end = min(prefix_position + 11, len(events))
        return tuple(range(prefix_position + 1, end)), False
    if objective_id != "rq1_24h":
        raise ValueError(f"unsupported objective {objective_id!r}")

    prefix_timestamp = events[prefix_position].timestamp
    end_timestamp = prefix_timestamp + _WINDOW_SECONDS
    selected = []
    for position in range(prefix_position + 1, len(events)):
        timestamp = events[position].timestamp
        if timestamp > end_timestamp:
            break
        if timestamp > prefix_timestamp:
            selected.append(position)
    if selected:
        return tuple(selected), False
    return (prefix_position + 1,), True


@dataclass
class _Accumulator:
    prefixes: int = 0
    eligible: int = 0
    fallback: int = 0
    candidate_counts: list[int] = field(default_factory=list)
    unique_counts: list[int] = field(default_factory=list)
    candidate_memberships: int = 0
    unique_memberships: int = 0
    distance_expected_sum: float = 0.0
    distance_minimum: int | None = None
    distance_maximum: int | None = None
    distance_bins: Counter[str] = field(default_factory=Counter)
    rank_expected_sum: float = 0.0
    rank_minimum: int | None = None
    rank_maximum: int | None = None
    rank_bins: Counter[str] = field(default_factory=Counter)

    def add(
        self,
        events: Sequence[TargetEvent],
        prefix_position: int,
        indices: Sequence[int],
        *,
        eligible: bool,
        fallback: bool,
    ) -> None:
        if not indices:
            raise ValueError("a training prefix has no target after fallback")
        prefix = events[prefix_position]
        candidates = [events[position] for position in indices]
        unique_count = len({event.item_id for event in candidates})
        self.prefixes += 1
        self.eligible += int(eligible)
        self.fallback += int(fallback)
        self.candidate_counts.append(len(candidates))
        self.unique_counts.append(unique_count)
        self.candidate_memberships += len(candidates)
        self.unique_memberships += unique_count

        weight = 1.0 / len(candidates)
        for position, event in zip(indices, candidates, strict=True):
            distance = event.timestamp - prefix.timestamp
            rank = position - prefix_position
            if distance < 0 or rank < 1:
                raise ValueError("target events must follow their causal prefix")
            self.distance_expected_sum += weight * distance
            self.distance_bins[_distance_bin(distance)] += weight
            self.distance_minimum = _minimum(self.distance_minimum, distance)
            self.distance_maximum = _maximum(self.distance_maximum, distance)
            self.rank_expected_sum += weight * rank
            self.rank_bins[_rank_bin(rank)] += weight
            self.rank_minimum = _minimum(self.rank_minimum, rank)
            self.rank_maximum = _maximum(self.rank_maximum, rank)

    def document(self, *, include_histograms: bool) -> dict[str, Any]:
        if self.prefixes == 0:
            raise ValueError("target statistics require causal prefixes")
        result = {
            "prefix_positive_pairs": self.prefixes,
            "eligible_prefixes": self.eligible,
            "eligibility_rate": self.eligible / self.prefixes,
            "fallback_prefixes": self.fallback,
            "fallback_rate": self.fallback / self.prefixes,
            "post_fallback_empty_prefixes": 0,
            "candidate_occurrences": _cardinality_document(
                self.candidate_counts, include_histogram=include_histograms
            ),
            "acceptable_unique_items": _cardinality_document(
                self.unique_counts, include_histogram=include_histograms
            ),
            "duplicate_candidate_membership_rate": 1.0
            - self.unique_memberships / self.candidate_memberships,
            "sampled_target_distance_seconds": {
                "sampling": "uniform_over_candidate_occurrences_per_prefix",
                "minimum": self.distance_minimum,
                "maximum": self.distance_maximum,
                "expected_mean": self.distance_expected_sum / self.prefixes,
                "bins": _weighted_bins(self.distance_bins, self.prefixes),
            },
            "sampled_target_event_rank": {
                "sampling": "uniform_over_candidate_occurrences_per_prefix",
                "minimum": self.rank_minimum,
                "maximum": self.rank_maximum,
                "expected_mean": self.rank_expected_sum / self.prefixes,
                "bins": _weighted_bins(self.rank_bins, self.prefixes),
            },
        }
        return result


def build_target_statistics_evidence(repo_root: Path) -> dict[str, Any]:
    root = repo_root.resolve(strict=True)
    protocol = root / "experiments/g4_future_items/protocol"
    control_semantics_path = protocol / "control_semantics_manifest.json"
    selected_control_path = protocol / "selected_control_manifest.json"
    treatment_semantics_path = protocol / "treatment_semantics_manifest.json"
    control_semantics = load_strict_json(control_semantics_path)
    selected_control = load_strict_json(selected_control_path)
    treatment_semantics = load_strict_json(treatment_semantics_path)
    budget = _training_budget(root, selected_control)
    main_identity = control_semantics["data_identity"]["main"]
    main_path = Path(main_identity["path"])
    _require_file_identity(main_path, main_identity)

    cutoff = int(control_semantics["data_identity"]["split_cutoff_timestamp"])
    frame = (
        pl.scan_parquet(main_path)
        .filter((pl.col("event_type") == "like") & (pl.col("timestamp") < cutoff))
        .select("uid", "timestamp", "compact_item_id")
        .collect(engine="streaming")
        .sort(["uid", "timestamp"], maintain_order=True)
    )
    events_by_user = _events_by_user(frame)
    activity_quartiles = _activity_quartiles(events_by_user)
    global_accumulators = {objective: _Accumulator() for objective in _OBJECTIVES}
    quartile_accumulators = {
        quartile: {objective: _Accumulator() for objective in _OBJECTIVES}
        for quartile in range(1, 5)
    }

    for user_id, events in events_by_user.items():
        quartile = activity_quartiles[user_id]
        for prefix_position in range(len(events) - 1):
            for objective in _OBJECTIVES:
                indices, fallback = candidate_indices(events, prefix_position, objective)
                eligible = not fallback
                global_accumulators[objective].add(
                    events,
                    prefix_position,
                    indices,
                    eligible=eligible,
                    fallback=fallback,
                )
                quartile_accumulators[quartile][objective].add(
                    events,
                    prefix_position,
                    indices,
                    eligible=eligible,
                    fallback=fallback,
                )

    population = {
        "training_like_events": len(frame),
        "training_users": len(events_by_user),
        "causal_prefixes": len(frame) - len(events_by_user),
    }
    activity = {}
    for quartile in range(1, 5):
        users = [
            user_id
            for user_id, assigned_quartile in activity_quartiles.items()
            if assigned_quartile == quartile
        ]
        counts = [len(events_by_user[user_id]) for user_id in users]
        activity[f"q{quartile}"] = {
            "users": len(users),
            "training_like_events_per_user": _cardinality_document(
                counts, include_histogram=False
            ),
            "objectives": {
                objective: quartile_accumulators[quartile][objective].document(
                    include_histograms=False
                )
                for objective in _OBJECTIVES
            },
        }

    return {
        "schema_version": 1,
        "kind": "g4_native50m_target_statistics",
        "identity": {
            "control_manifest_sha256": control_semantics[
                "control_manifest_sha256"
            ],
            "control_semantics_manifest": _manifest_identity(
                root, control_semantics_path, control_semantics
            ),
            "selected_control_manifest": _manifest_identity(
                root, selected_control_path, selected_control
            ),
            "treatment_semantics_manifest": _manifest_identity(
                root, treatment_semantics_path, treatment_semantics
            ),
            "data": control_semantics["data_identity"],
            "generation_sources": {
                path: _file_sha256(root / path) for path in _SOURCE_PATHS
            },
            "target_schema_revisions": treatment_semantics["schema_revisions"],
        },
        "definitions": {
            "split": "event_type == 'like' and timestamp < split_cutoff_timestamp",
            "causal_prefix": "every training like event with a later like event",
            "eligibility": {
                "control_next_item": "a next event exists",
                "rq1_24h": "a strictly later event exists within 86400 seconds before fallback",
                "rq2_next10": "at least one later event exists",
            },
            "fallback": {
                "control_next_item": "none",
                "rq1_24h": "next event when the strict 24-hour set is empty",
                "rq2_next10": "none",
            },
            "acceptable_items": "unique compact item ids in the complete occurrence candidate set after fallback",
            "sampled_target": "one occurrence drawn uniformly per prefix and epoch",
            "activity_quartiles": "users sorted by (training like count, uid), assigned by 4 * zero_based_rank // user_count + 1",
            "cardinality_quantiles": "nearest-rank empirical quantiles",
            "distance_bins_seconds": {
                "zero": "0",
                "0_6h": "(0,21600]",
                "6_24h": "(21600,86400]",
                "1_3d": "(86400,259200]",
                "3_7d": "(259200,604800]",
                "7d_plus": "(604800,infinity)",
            },
            "event_rank_bins": {
                "rank_1": "1",
                "rank_2_5": "[2,5]",
                "rank_6_10": "[6,10]",
                "rank_11_plus": "[11,infinity)",
            },
        },
        "population": population,
        "training_budget": budget,
        "objectives": {
            objective: global_accumulators[objective].document(
                include_histograms=True
            )
            for objective in _OBJECTIVES
        },
        "user_activity_quartiles": activity,
    }


def write_target_statistics_evidence(
    artifact_path: Path, *, repo_root: Path
) -> str:
    document = build_target_statistics_evidence(repo_root)
    payload = canonical_bytes(document)
    digest = hashlib.sha256(payload).hexdigest()
    _write_immutable(artifact_path, payload)
    _write_immutable(artifact_path.with_suffix(".sha256"), digest.encode("ascii"))
    return digest


def verify_target_statistics_evidence(
    artifact_path: Path, *, repo_root: Path
) -> dict[str, Any]:
    root = repo_root.resolve(strict=True)
    payload = artifact_path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if artifact_path.with_suffix(".sha256").read_text() != digest:
        raise ValueError("target-statistics artifact digest differs from its sidecar")
    document = load_strict_json(artifact_path)
    if canonical_bytes(document) != payload:
        raise ValueError("target-statistics artifact is not canonical JSON")
    if document.get("schema_version") != 1 or document.get("kind") != (
        "g4_native50m_target_statistics"
    ):
        raise ValueError("unsupported target-statistics artifact schema")

    identity = document["identity"]
    for name, relative_path in (
        ("control_semantics_manifest", "control_semantics_manifest.json"),
        ("selected_control_manifest", "selected_control_manifest.json"),
        ("treatment_semantics_manifest", "treatment_semantics_manifest.json"),
    ):
        path = root / "experiments/g4_future_items/protocol" / relative_path
        manifest = load_strict_json(path)
        if identity[name] != _manifest_identity(root, path, manifest):
            raise ValueError(f"{name} identity changed")
    control_semantics = load_strict_json(
        root
        / "experiments/g4_future_items/protocol/control_semantics_manifest.json"
    )
    if identity["control_manifest_sha256"] != control_semantics[
        "control_manifest_sha256"
    ]:
        raise ValueError("control-manifest identity changed")
    selected_control = load_strict_json(
        root / "experiments/g4_future_items/protocol/selected_control_manifest.json"
    )
    treatment_semantics = load_strict_json(
        root
        / "experiments/g4_future_items/protocol/treatment_semantics_manifest.json"
    )
    if selected_control["control_semantics_manifest_sha256"] != canonical_sha256(
        control_semantics
    ):
        raise ValueError("selected control is not bound to the control semantics")
    if treatment_semantics["selected_control_manifest_sha256"] != canonical_sha256(
        selected_control
    ):
        raise ValueError("treatment semantics are not bound to the selected control")
    if identity["data"] != control_semantics["data_identity"]:
        raise ValueError("target-statistics data identity differs from control semantics")
    if identity["target_schema_revisions"] != treatment_semantics[
        "schema_revisions"
    ]:
        raise ValueError("target schema revisions changed")
    if identity["generation_sources"] != {
        path: _file_sha256(root / path) for path in _SOURCE_PATHS
    }:
        raise ValueError("target-statistics generation source identity changed")
    for path in (
        "experiments/g4_future_items/targets.py",
        "experiments/g4_future_items/configs/treatments.py",
    ):
        if identity["generation_sources"][path] != treatment_semantics["sources"][
            path
        ]:
            raise ValueError(f"frozen treatment source identity changed: {path}")
    _require_file_identity(
        Path(identity["data"]["main"]["path"]), identity["data"]["main"]
    )

    population = document["population"]
    prefix_count = population["causal_prefixes"]
    if population["training_like_events"] - population["training_users"] != prefix_count:
        raise ValueError("causal-prefix population identity is inconsistent")
    budget = document["training_budget"]
    if budget["target_pairs_per_epoch"] != prefix_count:
        raise ValueError("saved training target budget differs from the population")
    for evidence in budget["evidence"].values():
        path = root / evidence["path"]
        if _file_sha256(path) != evidence["file_sha256"]:
            raise ValueError("training-budget evidence identity changed")
        metadata = load_strict_json(path)
        if any(metadata[name] != value for name, value in evidence["fields"].items()):
            raise ValueError("training-budget evidence fields changed")
    for objective in document["objectives"].values():
        if objective["prefix_positive_pairs"] != prefix_count:
            raise ValueError("objective pair budget differs from the control")
        if objective["post_fallback_empty_prefixes"] != 0:
            raise ValueError("objective leaves an empty training target")
    for name, objective in document["objectives"].items():
        activity_pairs = sum(
            quartile["objectives"][name]["prefix_positive_pairs"]
            for quartile in document["user_activity_quartiles"].values()
        )
        if activity_pairs != objective["prefix_positive_pairs"]:
            raise ValueError("activity slices do not partition the target population")
    return document


def _events_by_user(frame: pl.DataFrame) -> dict[int, tuple[TargetEvent, ...]]:
    result: dict[int, list[TargetEvent]] = {}
    for user_id, timestamp, item_id in frame.iter_rows():
        result.setdefault(int(user_id), []).append(
            TargetEvent(timestamp=int(timestamp), item_id=int(item_id))
        )
    return {user_id: tuple(events) for user_id, events in result.items()}


def _training_budget(
    root: Path, selected_control: dict[str, Any]
) -> dict[str, Any]:
    run_names = {
        "control_next_item": selected_control["selection"]["run_name"],
        "rq1_24h": "g4_rq1_24h_trial_01_native50m",
        "rq2_next10": "g4_rq2_next10_trial_01_native50m",
    }
    evidence = {}
    expected: dict[str, int] | None = None
    for objective, run_name in run_names.items():
        path = root / "generated/logs" / run_name / "training_metadata.json"
        metadata = load_strict_json(path)
        fields = {
            "batch_size": metadata["batch_size"],
            "effective_batch_size": metadata["effective_batch_size"],
            "targets_per_epoch": metadata["targets_per_epoch"],
            "optimizer_steps_per_epoch": metadata["optimizer_steps_per_epoch"],
        }
        if expected is None:
            expected = fields
        elif fields != expected:
            raise ValueError("control and treatment training budgets differ")
        evidence[objective] = {
            "path": path.relative_to(root).as_posix(),
            "file_sha256": _file_sha256(path),
            "fields": fields,
        }
    assert expected is not None
    if expected["batch_size"] != 512 or expected["effective_batch_size"] != 512:
        raise ValueError("G4 target-statistics evidence requires batch 512")
    return {
        "batch_size": expected["batch_size"],
        "effective_batch_size": expected["effective_batch_size"],
        "target_pairs_per_epoch": expected["targets_per_epoch"],
        "optimizer_steps_per_epoch": expected["optimizer_steps_per_epoch"],
        "evidence": evidence,
    }


def _activity_quartiles(
    events_by_user: dict[int, tuple[TargetEvent, ...]],
) -> dict[int, int]:
    users = sorted(
        events_by_user,
        key=lambda user_id: (len(events_by_user[user_id]), user_id),
    )
    return {
        user_id: 4 * rank // len(users) + 1
        for rank, user_id in enumerate(users)
    }


def _cardinality_document(
    values: Sequence[int], *, include_histogram: bool
) -> dict[str, Any]:
    if not values:
        raise ValueError("cardinality statistics require values")
    ordered = sorted(values)
    result: dict[str, Any] = {
        "count": len(ordered),
        "sum": sum(ordered),
        "minimum": ordered[0],
        "maximum": ordered[-1],
        "mean": sum(ordered) / len(ordered),
        "p50": _nearest_rank(ordered, 0.50),
        "p90": _nearest_rank(ordered, 0.90),
        "p95": _nearest_rank(ordered, 0.95),
        "p99": _nearest_rank(ordered, 0.99),
    }
    if include_histogram:
        result["histogram"] = {
            str(value): count for value, count in sorted(Counter(ordered).items())
        }
    return result


def _nearest_rank(ordered: Sequence[int], quantile: float) -> int:
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


def _weighted_bins(values: Counter[str], denominator: int) -> dict[str, Any]:
    return {
        name: {
            "expected_prefix_draws": values.get(name, 0.0),
            "share": values.get(name, 0.0) / denominator,
        }
        for name in values.keys() | _required_bin_names(values)
    }


def _required_bin_names(values: Counter[str]) -> set[str]:
    if any(name.startswith("rank_") for name in values):
        return {"rank_1", "rank_2_5", "rank_6_10", "rank_11_plus"}
    return {name for name, _ in _DISTANCE_BINS}


def _distance_bin(distance: int) -> str:
    if distance == 0:
        return "zero"
    for name, upper in _DISTANCE_BINS[1:]:
        if upper is None or distance <= upper:
            return name
    raise AssertionError


def _rank_bin(rank: int) -> str:
    if rank == 1:
        return "rank_1"
    if rank <= 5:
        return "rank_2_5"
    if rank <= 10:
        return "rank_6_10"
    return "rank_11_plus"


def _minimum(current: int | None, value: int) -> int:
    return value if current is None else min(current, value)


def _maximum(current: int | None, value: int) -> int:
    return value if current is None else max(current, value)


def _manifest_identity(
    root: Path, path: Path, document: dict[str, Any]
) -> dict[str, str]:
    return {
        "path": path.relative_to(root).as_posix(),
        "document_sha256": canonical_sha256(document),
        "file_sha256": _file_sha256(path),
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _require_file_identity(path: Path, expected: dict[str, Any]) -> None:
    status = path.stat()
    if status.st_size != expected["size"] or _file_sha256(path) != expected["sha256"]:
        raise ValueError(f"frozen input identity changed: {path}")


def _write_immutable(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"immutable evidence already differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "experiments/g4_future_items/evidence/target_statistics_native50m_v1.json"
        ),
    )
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    if not arguments.write:
        raise SystemExit("pass --write to create immutable target-statistics evidence")
    print(write_target_statistics_evidence(arguments.output, repo_root=root))


if __name__ == "__main__":
    main()
