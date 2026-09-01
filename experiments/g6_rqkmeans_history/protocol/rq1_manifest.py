from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Literal

from torch.quasirandom import SobolEngine


SidLookupInitialization = Literal["random", "content_pca"]
CoordinateProvenance = Literal["rq0_carryover", "sobol"]
EMBEDDING_LR_BOUNDS = (1e-4, 0.256)
DEEP_LR_BOUNDS = (1e-4, 0.128)
SEARCH_SEED = 42
SEARCH_COORDINATES = 16
CARRYOVER_COORDINATES = 10
SOURCE_MANIFEST_SHA256 = (
    "f8694eb0503e47a25fe6f278f66598ba6c3fdebc8800433339b8cc93ef8650b1"
)


@dataclass(frozen=True)
class Rq1SearchCoordinate:
    trial: int
    embedding_learning_rate: float
    deep_learning_rate: float
    provenance: CoordinateProvenance
    source_job_id: str | None = None
    source_run_name: str | None = None

    @property
    def identity(self) -> tuple[float, float]:
        return self.embedding_learning_rate, self.deep_learning_rate

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Rq1SearchJob:
    initialization: SidLookupInitialization
    coordinate: Rq1SearchCoordinate

    @property
    def id(self) -> str:
        return (
            f"rq1_surface:{self.initialization}_trial_"
            f"{self.coordinate.trial:02d}"
        )

    @property
    def run_name(self) -> str:
        return (
            f"g6_rq1_{self.initialization}_trial_"
            f"{self.coordinate.trial:02d}_native50m"
        )

    @property
    def reused(self) -> bool:
        return (
            self.initialization == "random"
            and self.coordinate.provenance == "rq0_carryover"
        )

    @property
    def physical_job_id(self) -> str:
        if self.reused:
            if self.coordinate.source_job_id is None:
                raise RuntimeError("RQ1 carryover has no source job identity")
            return self.coordinate.source_job_id
        return self.id

    @property
    def physical_run_name(self) -> str:
        if self.reused:
            if self.coordinate.source_run_name is None:
                raise RuntimeError("RQ1 carryover has no source run identity")
            return self.coordinate.source_run_name
        return self.run_name

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "batch_size": 256,
            "validation_batch_size": 8192,
            "representation": "item_learned_frozen_sid_event",
            "representation_width": 32,
            "num_levels": 4,
            "num_codes": 512,
            "kmeans_iterations": 20,
            "kmeans_seed": 42,
            "training_seed": 42,
            "sid_lookup_initialization": self.initialization,
            "embedding_learning_rate": self.coordinate.embedding_learning_rate,
            "deep_learning_rate": self.coordinate.deep_learning_rate,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "run_name": self.run_name,
            "initialization": self.initialization,
            "coordinate": self.coordinate.to_dict(),
            "parameters": self.parameters,
            "physical_identity": {
                "job_id": self.physical_job_id,
                "run_name": self.physical_run_name,
                "reused": self.reused,
            },
        }


@dataclass(frozen=True)
class Rq1SearchManifest:
    coordinates: tuple[Rq1SearchCoordinate, ...]
    jobs: tuple[Rq1SearchJob, ...]

    @property
    def new_physical_jobs(self) -> tuple[Rq1SearchJob, ...]:
        return tuple(job for job in self.jobs if not job.reused)

    @property
    def sha256(self) -> str:
        content = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return hashlib.sha256(content.encode()).hexdigest()

    def jobs_for_initialization(
        self, initialization: SidLookupInitialization
    ) -> tuple[Rq1SearchJob, ...]:
        return tuple(
            job for job in self.jobs if job.initialization == initialization
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "dataset_size": "native-50m",
            "coordinate_count": len(self.coordinates),
            "logical_cell_count": len(self.jobs),
            "new_physical_run_count": len(self.new_physical_jobs),
            "search_seed": SEARCH_SEED,
            "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
            "search_space": {
                "embedding_learning_rate": [*EMBEDDING_LR_BOUNDS, "log_uniform"],
                "deep_learning_rate": [*DEEP_LR_BOUNDS, "log_uniform"],
            },
            "coordinates": [coordinate.to_dict() for coordinate in self.coordinates],
            "jobs": [job.to_dict() for job in self.jobs],
        }


_CARRYOVERS = (
    (
        0.19105100385081308,
        0.011234096354949456,
        "treatment_tuning:item_learned_frozen_sid_event_trial_10",
        "g6_rq0_item_learned_frozen_sid_event_trial_10_native50m",
    ),
    (
        0.2282507260721797,
        0.008686092009279228,
        "treatment_tuning:item_learned_frozen_sid_event_trial_11",
        "g6_rq0_item_learned_frozen_sid_event_trial_11_native50m",
    ),
    (
        0.21826754014805252,
        0.009880402777876574,
        "treatment_tuning:item_learned_frozen_sid_event_trial_12",
        "g6_rq0_item_learned_frozen_sid_event_trial_12_native50m",
    ),
    (
        0.04137151856281843,
        0.007758859700325065,
        "treatment_tuning:item_learned_frozen_sid_event_trial_13",
        "g6_rq0_item_learned_frozen_sid_event_trial_13_native50m",
    ),
    (
        0.0031342959180331227,
        0.018631639825509576,
        "treatment_tuning:item_learned_frozen_sid_event_trial_14",
        "g6_rq0_item_learned_frozen_sid_event_trial_14_native50m",
    ),
    (
        0.06908817999613777,
        0.004193366561454768,
        "treatment_tuning:item_learned_frozen_sid_event_trial_15",
        "g6_rq0_item_learned_frozen_sid_event_trial_15_native50m",
    ),
    (
        0.3620386719675124,
        0.011234096354949456,
        "lr_boundary:boundary_item_learned_frozen_sid_event_embedding_learning_rate_0",
        "g6_rq0_boundary_item_learned_frozen_sid_event_"
        "embedding_learning_rate_0_native50m",
    ),
    (
        0.512,
        0.011234096354949456,
        "lr_boundary:boundary_item_learned_frozen_sid_event_embedding_learning_rate_1",
        "g6_rq0_boundary_item_learned_frozen_sid_event_"
        "embedding_learning_rate_1_native50m",
    ),
    (
        0.7240773439350248,
        0.011234096354949456,
        "lr_boundary:boundary_item_learned_frozen_sid_event_embedding_learning_rate_2",
        "g6_rq0_boundary_item_learned_frozen_sid_event_"
        "embedding_learning_rate_2_native50m",
    ),
    (
        1.024,
        0.011234096354949456,
        "lr_boundary:boundary_item_learned_frozen_sid_event_embedding_learning_rate_3",
        "g6_rq0_boundary_item_learned_frozen_sid_event_"
        "embedding_learning_rate_3_native50m",
    ),
)


def _log_uniform(bounds: tuple[float, float], unit: float) -> float:
    lower, upper = bounds
    return math.exp(math.log(lower) + unit * math.log(upper / lower))


def _coordinates() -> tuple[Rq1SearchCoordinate, ...]:
    coordinates = [
        Rq1SearchCoordinate(
            trial=trial,
            embedding_learning_rate=embedding_learning_rate,
            deep_learning_rate=deep_learning_rate,
            provenance="rq0_carryover",
            source_job_id=source_job_id,
            source_run_name=source_run_name,
        )
        for trial, (
            embedding_learning_rate,
            deep_learning_rate,
            source_job_id,
            source_run_name,
        ) in enumerate(_CARRYOVERS)
    ]
    points = SobolEngine(2, scramble=True, seed=SEARCH_SEED).draw(
        SEARCH_COORDINATES - CARRYOVER_COORDINATES
    )
    coordinates.extend(
        Rq1SearchCoordinate(
            trial=trial,
            embedding_learning_rate=_log_uniform(EMBEDDING_LR_BOUNDS, point[0]),
            deep_learning_rate=_log_uniform(DEEP_LR_BOUNDS, point[1]),
            provenance="sobol",
        )
        for trial, point in enumerate(points.tolist(), start=CARRYOVER_COORDINATES)
    )
    return tuple(coordinates)


_COORDINATES = _coordinates()
_MANIFEST = Rq1SearchManifest(
    coordinates=_COORDINATES,
    jobs=tuple(
        Rq1SearchJob(initialization, coordinate)
        for coordinate in _COORDINATES
        for initialization in ("random", "content_pca")
    ),
)


def rq1_search_manifest() -> Rq1SearchManifest:
    if len(_MANIFEST.coordinates) != SEARCH_COORDINATES:
        raise RuntimeError("RQ1 search coordinate budget changed")
    if len({coordinate.identity for coordinate in _MANIFEST.coordinates}) != len(
        _MANIFEST.coordinates
    ):
        raise RuntimeError("RQ1 search contains duplicate coordinates")
    if len(_MANIFEST.jobs) != 2 * SEARCH_COORDINATES:
        raise RuntimeError("RQ1 paired-cell budget changed")
    if len(_MANIFEST.new_physical_jobs) != 22:
        raise RuntimeError("RQ1 new physical-run budget changed")
    if len({job.physical_job_id for job in _MANIFEST.jobs}) != len(_MANIFEST.jobs):
        raise RuntimeError("RQ1 physical job identities are not unique")
    return _MANIFEST


def validate_rq1_search_job(job: Rq1SearchJob) -> None:
    if job not in rq1_search_manifest().jobs:
        raise ValueError("job is outside the approved RQ1 paired search")
