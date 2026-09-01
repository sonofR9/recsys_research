from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

from experiments.g6_rqkmeans_history.protocol.collision_policy import (
    CollisionSearchJob,
    collision_search_manifest,
)


SOURCE_JOB_ID = "collision_search:none_trial_30"
RECOVERY_RUN_NAME = "g6_rq3_none_trial_30_recovery_01_native50m"


@dataclass(frozen=True)
class CollisionRecoveryJob:
    source_job_id: str
    source_manifest_sha256: str
    run_name: str
    recovery_attempt: int

    @property
    def source_job(self) -> CollisionSearchJob:
        matches = [
            job
            for job in collision_search_manifest().new_physical_jobs
            if job.id == self.source_job_id
        ]
        if len(matches) != 1:
            raise RuntimeError("recovery source job has drifted")
        return matches[0]

    @property
    def manifest_sha256(self) -> str:
        payload = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        return hashlib.sha256(payload).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return {
            "source_job_id": self.source_job_id,
            "source_manifest_sha256": self.source_manifest_sha256,
            "run_name": self.run_name,
            "recovery_attempt": self.recovery_attempt,
        }


def recovery_job() -> CollisionRecoveryJob:
    manifest = collision_search_manifest()
    job = CollisionRecoveryJob(
        source_job_id=SOURCE_JOB_ID,
        source_manifest_sha256=manifest.sha256,
        run_name=RECOVERY_RUN_NAME,
        recovery_attempt=1,
    )
    if job.source_job.policy != "none" or job.source_job.coordinate.trial != 30:
        raise RuntimeError("recovery coordinate has drifted")
    return job
