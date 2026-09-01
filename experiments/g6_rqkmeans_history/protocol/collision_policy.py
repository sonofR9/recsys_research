from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from typing import Any, Literal, get_args

import torch
from torch.quasirandom import SobolEngine

from dcn.config import CollisionPolicy
from dcn.semantic import SemanticCodes


COLLISION_POLICIES: tuple[CollisionPolicy, ...] = get_args(CollisionPolicy)
NUM_LEVELS = (2, 3, 4, 5)
NUM_CODES = (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
KMEANS_ITERATIONS = (10, 20, 40)
EMBEDDING_LR_BOUNDS = (0.008, 0.512)
DEEP_LR_BOUNDS = (0.002, 0.128)
RQ0_EMBEDDING_LEARNING_RATE = 0.3620386719675124
RQ0_DEEP_LEARNING_RATE = 0.03463626154088337
SEARCH_SEED = 42
SEARCH_COORDINATES = 40
MAX_SYMBOLS_PER_LEVEL = 8192
SOURCE_SELECTION_SHA256 = (
    "8391def6cfddbeb4cb1b048f3d4fed62e4bf0e304270e8d21a7d2de4ded0646b"
)
SOURCE_JOB_ID = "lr_boundary:boundary_item_frozen_sid_event_embedding_learning_rate_0"
SOURCE_RUN_NAME = (
    "g6_rq0_boundary_item_frozen_sid_event_embedding_learning_rate_0_native50m"
)


@dataclass(frozen=True)
class CollisionSearchCoordinate:
    trial: int
    num_levels: int
    num_codes: int
    kmeans_iterations: int
    embedding_learning_rate: float
    deep_learning_rate: float

    @property
    def identity(self) -> tuple[int, int, int, float, float]:
        return (
            self.num_levels,
            self.num_codes,
            self.kmeans_iterations,
            self.embedding_learning_rate,
            self.deep_learning_rate,
        )

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class CollisionSearchJob:
    policy: CollisionPolicy
    coordinate: CollisionSearchCoordinate

    @property
    def id(self) -> str:
        return f"collision_search:{self.policy}_trial_{self.coordinate.trial:02d}"

    @property
    def run_name(self) -> str:
        rq = 2 if self.policy == "suffix" else 3
        return f"g6_rq{rq}_{self.policy}_trial_{self.coordinate.trial:02d}_native50m"

    @property
    def reused(self) -> bool:
        return self.policy == "suffix" and self.coordinate.trial == 0

    @property
    def physical_job_id(self) -> str:
        return SOURCE_JOB_ID if self.reused else self.id

    @property
    def physical_run_name(self) -> str:
        return SOURCE_RUN_NAME if self.reused else self.run_name

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            **self.coordinate.to_dict(),
            "collision_policy": self.policy,
            "batch_size": 256,
            "validation_batch_size": 8192,
            "representation": "item_frozen_sid_event",
            "representation_width": 128,
            "kmeans_seed": 42,
            "training_seed": 42,
        }

    def with_parameters(self, **changes: Any) -> CollisionSearchJob:
        coordinate_fields = CollisionSearchCoordinate.__dataclass_fields__
        unknown = changes.keys() - coordinate_fields.keys()
        if unknown:
            raise ValueError(f"unknown coordinate parameters: {sorted(unknown)}")
        return replace(self, coordinate=replace(self.coordinate, **changes))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "run_name": self.run_name,
            "policy": self.policy,
            "coordinate": self.coordinate.to_dict(),
            "parameters": self.parameters,
            "physical_identity": {
                "job_id": self.physical_job_id,
                "run_name": self.physical_run_name,
                "reused": self.reused,
            },
        }


@dataclass(frozen=True)
class CollisionSearchManifest:
    jobs: tuple[CollisionSearchJob, ...]

    @property
    def new_physical_jobs(self) -> tuple[CollisionSearchJob, ...]:
        return tuple(job for job in self.jobs if not job.reused)

    @property
    def sha256(self) -> str:
        content = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return hashlib.sha256(content.encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "dataset_size": "native-50m",
            "coordinate_count": SEARCH_COORDINATES,
            "logical_cell_count": len(self.jobs),
            "new_physical_run_count": len(self.new_physical_jobs),
            "search_seed": SEARCH_SEED,
            "source_selection_sha256": SOURCE_SELECTION_SHA256,
            "source_job_id": SOURCE_JOB_ID,
            "search_space": {
                "num_levels": list(NUM_LEVELS),
                "shared_num_codes": list(NUM_CODES),
                "kmeans_iterations": list(KMEANS_ITERATIONS),
                "embedding_learning_rate": [*EMBEDDING_LR_BOUNDS, "log_uniform"],
                "deep_learning_rate": [*DEEP_LR_BOUNDS, "log_uniform"],
            },
            "jobs": [job.to_dict() for job in self.jobs],
        }


def _choice(values: tuple[int, ...], unit: float) -> int:
    return values[min(int(unit * len(values)), len(values) - 1)]


def _log_uniform(bounds: tuple[float, float], unit: float) -> float:
    lower, upper = bounds
    return math.exp(math.log(lower) + unit * math.log(upper / lower))


def _coordinates() -> tuple[CollisionSearchCoordinate, ...]:
    anchors = (
        (3, 512, 20),
        (2, 64, 20),
        (4, 1024, 20),
        (5, 4096, 40),
    )
    coordinates = [
        CollisionSearchCoordinate(
            trial=trial,
            num_levels=num_levels,
            num_codes=num_codes,
            kmeans_iterations=iterations,
            embedding_learning_rate=RQ0_EMBEDDING_LEARNING_RATE,
            deep_learning_rate=RQ0_DEEP_LEARNING_RATE,
        )
        for trial, (num_levels, num_codes, iterations) in enumerate(anchors)
    ]
    points = SobolEngine(5, scramble=True, seed=SEARCH_SEED).draw(
        SEARCH_COORDINATES - len(anchors)
    )
    for trial, point in enumerate(points.tolist(), start=len(anchors)):
        coordinates.append(
            CollisionSearchCoordinate(
                trial=trial,
                num_levels=_choice(NUM_LEVELS, point[0]),
                num_codes=_choice(NUM_CODES, point[1]),
                kmeans_iterations=_choice(KMEANS_ITERATIONS, point[2]),
                embedding_learning_rate=_log_uniform(
                    EMBEDDING_LR_BOUNDS, point[3]
                ),
                deep_learning_rate=_log_uniform(DEEP_LR_BOUNDS, point[4]),
            )
        )
    return tuple(coordinates)


_COORDINATES = _coordinates()
_MANIFEST = CollisionSearchManifest(
    tuple(
        CollisionSearchJob(policy, coordinate)
        for coordinate in _COORDINATES
        for policy in COLLISION_POLICIES
    )
)


def collision_search_coordinates() -> tuple[CollisionSearchCoordinate, ...]:
    return _COORDINATES


def collision_search_manifest() -> CollisionSearchManifest:
    if len(_COORDINATES) != SEARCH_COORDINATES:
        raise RuntimeError("collision search coordinate budget changed")
    if len({coordinate.identity for coordinate in _COORDINATES}) != len(_COORDINATES):
        raise RuntimeError("collision search contains duplicate coordinates")
    if len(_MANIFEST.jobs) != SEARCH_COORDINATES * len(COLLISION_POLICIES):
        raise RuntimeError("collision search paired-cell budget changed")
    if len({job.id for job in _MANIFEST.jobs}) != len(_MANIFEST.jobs):
        raise RuntimeError("collision search job IDs are not unique")
    if len(_MANIFEST.new_physical_jobs) != 79:
        raise RuntimeError("collision search reused-run budget changed")
    if len({job.physical_job_id for job in _MANIFEST.jobs}) != len(_MANIFEST.jobs):
        raise RuntimeError("collision search physical job IDs are not unique")
    return _MANIFEST


def validate_collision_search_job(job: CollisionSearchJob) -> None:
    if job not in collision_search_manifest().jobs:
        raise ValueError("job is outside the approved paired search")


def validate_collision_symbol_cap(
    codes: SemanticCodes,
    *,
    policy: CollisionPolicy,
    base_levels: int,
    require_suffix_feasibility: bool = False,
) -> None:
    expected_levels = base_levels + int(policy == "suffix")
    if codes.num_levels != expected_levels:
        raise ValueError(f"{policy} policy produced the wrong number of levels")
    for level, symbols in enumerate(codes.codes_per_level):
        if symbols <= MAX_SYMBOLS_PER_LEVEL:
            continue
        name = "suffix level" if level == base_levels else f"base level {level}"
        raise ValueError(f"{name} exceeds the 8192-symbol cap")
    if require_suffix_feasibility and policy == "none":
        _, counts = torch.unique(
            codes.codes[:, :base_levels], dim=0, return_counts=True
        )
        if int(counts.max()) + 1 > MAX_SYMBOLS_PER_LEVEL:
            raise ValueError("counterfactual suffix exceeds the 8192-symbol cap")


def validate_collision_diagnostics(
    *,
    policy: CollisionPolicy,
    diagnostics: dict[str, Any],
) -> None:
    if diagnostics.get("collision_policy") != policy:
        raise ValueError("diagnostic collision policy changed")
    suffix_symbols = diagnostics.get("collision_suffix_symbols")
    if not isinstance(suffix_symbols, int) or isinstance(suffix_symbols, bool):
        raise ValueError("collision suffix symbols must be an integer")
    if policy == "none":
        if suffix_symbols != 0:
            raise ValueError("none policy must report zero suffix symbols")
    elif not 1 <= suffix_symbols <= MAX_SYMBOLS_PER_LEVEL:
        raise ValueError("collision suffix exceeds the 8192-symbol cap")
