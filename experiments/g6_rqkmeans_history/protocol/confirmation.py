from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from experiments.g6_rqkmeans_history.protocol.collision_policy import (
    CollisionSearchJob,
    collision_search_manifest,
)
from experiments.g6_rqkmeans_history.protocol.rq1_manifest import (
    Rq1SearchJob,
    rq1_search_manifest,
)


SURFACE_PATH = Path(
    "experiments/g6_rqkmeans_history/evidence/rq1_rq3_surface_native50m.json"
)
Family = Literal["rq1", "collision"]


@dataclass(frozen=True)
class ConfirmationJob:
    family: Family
    variant: str
    source_job_id: str
    seed: int
    num_levels: int
    num_codes: int
    kmeans_iterations: int
    embedding_learning_rate: float
    deep_learning_rate: float

    @property
    def id(self) -> str:
        return f"confirmation:{self.family}:{self.variant}:s{self.seed}"

    @property
    def run_name(self) -> str:
        return f"g6_{self.family}_{self.variant}_confirm_s{self.seed}_native50m"

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "run_name": self.run_name, **asdict(self)}


@dataclass(frozen=True)
class ConfirmationManifest:
    surface_sha256: str
    jobs: tuple[ConfirmationJob, ...]

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        return hashlib.sha256(payload).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "dataset_size": "native-50m",
            "surface_sha256": self.surface_sha256,
            "jobs": [job.to_dict() for job in self.jobs],
        }


def load_confirmation_manifest(
    path: Path = SURFACE_PATH,
) -> ConfirmationManifest:
    payload = path.read_bytes()
    document = json.loads(payload)
    if (
        not isinstance(document, dict)
        or document.get("schema") != "g6-rq1-rq3-surface/v1"
        or document.get("rq1_manifest_sha256") != rq1_search_manifest().sha256
        or document.get("collision_manifest_sha256")
        != collision_search_manifest().sha256
    ):
        raise ValueError("G6 surface evidence does not match approved manifests")
    boundary_groups = [
        *document["rq1"]["boundary_triggered"].values(),
        *document["rq2_rq3"]["lr_boundary_triggered"].values(),
    ]
    if any(boundary_groups):
        raise ValueError("LR boundary extension must resolve before confirmation")
    rq1_jobs = []
    for initialization in ("random", "content_pca"):
        source = _resolve_rq1(document["rq1"]["selected"][initialization])
        for seed in (43, 44, 45):
            rq1_jobs.append(
                ConfirmationJob(
                    family="rq1",
                    variant=f"{initialization}_t{source.coordinate.trial:02d}",
                    source_job_id=source.id,
                    seed=seed,
                    num_levels=4,
                    num_codes=512,
                    kmeans_iterations=20,
                    embedding_learning_rate=source.coordinate.embedding_learning_rate,
                    deep_learning_rate=source.coordinate.deep_learning_rate,
                )
            )
    collision_sources = [
        _resolve_collision(document["rq2_rq3"]["selected"][policy])
        for policy in ("suffix", "none")
    ]
    collision_sources.append(
        next(
            job
            for job in collision_search_manifest().jobs
            if job.policy == "suffix" and job.coordinate.trial == 0
        )
    )
    collision_jobs = []
    identities = set()
    for source in collision_sources:
        coordinate = source.coordinate
        for seed in (43, 44):
            identity = (source.policy, coordinate.identity, seed)
            if identity in identities:
                continue
            identities.add(identity)
            variant = (
                f"{source.policy}_l{coordinate.num_levels}_c{coordinate.num_codes}_"
                f"i{coordinate.kmeans_iterations}_t{coordinate.trial:02d}"
            )
            collision_jobs.append(
                ConfirmationJob(
                    family="collision",
                    variant=variant,
                    source_job_id=source.id,
                    seed=seed,
                    num_levels=coordinate.num_levels,
                    num_codes=coordinate.num_codes,
                    kmeans_iterations=coordinate.kmeans_iterations,
                    embedding_learning_rate=coordinate.embedding_learning_rate,
                    deep_learning_rate=coordinate.deep_learning_rate,
                )
            )
    jobs = (*rq1_jobs, *collision_jobs)
    if len(jobs) > 12 or len({job.id for job in jobs}) != len(jobs):
        raise RuntimeError("confirmation run budget or identity changed")
    return ConfirmationManifest(hashlib.sha256(payload).hexdigest(), jobs)


def _resolve_rq1(row: dict[str, Any]) -> Rq1SearchJob:
    matches = [job for job in rq1_search_manifest().jobs if job.id == row["job_id"]]
    if len(matches) != 1 or row["parameters"] != matches[0].parameters:
        raise ValueError("RQ1 selected row does not match its manifest job")
    return matches[0]


def _resolve_collision(row: dict[str, Any]) -> CollisionSearchJob:
    matches = [
        job for job in collision_search_manifest().jobs if job.id == row["job_id"]
    ]
    if len(matches) != 1 or row["parameters"] != matches[0].parameters:
        raise ValueError("collision selected row does not match its manifest job")
    return matches[0]
