from __future__ import annotations

import base64
import json
import os
from pathlib import Path

from experiments.g6_rqkmeans_history.protocol.collision_recovery import (
    CollisionRecoveryJob,
    recovery_job,
)


JOB_ENVIRONMENT = "G6_RQ23_RECOVERY_JOB_B64"
CONTRACT_FILENAME = "g6_rq2_rq3_recovery_job.json"


def encode_recovery_job(job: CollisionRecoveryJob) -> str:
    if job != recovery_job():
        raise ValueError("job is outside the approved collision recovery")
    payload = json.dumps(
        {
            "recovery_manifest_sha256": job.manifest_sha256,
            "job": job.to_dict(),
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return base64.urlsafe_b64encode(payload).decode()


def decode_recovery_job(encoded: str) -> CollisionRecoveryJob:
    try:
        contract = json.loads(base64.urlsafe_b64decode(encoded).decode())
    except (ValueError, json.JSONDecodeError) as error:
        raise RuntimeError("encoded recovery job is invalid") from error
    job = recovery_job()
    expected = {
        "recovery_manifest_sha256": job.manifest_sha256,
        "job": job.to_dict(),
    }
    if contract != expected:
        raise RuntimeError("recovery job is absent from the approved manifest")
    return job


def recovery_job_from_environment() -> CollisionRecoveryJob:
    encoded = os.environ.get(JOB_ENVIRONMENT)
    if not encoded:
        raise RuntimeError(f"{JOB_ENVIRONMENT} is required")
    return decode_recovery_job(encoded)


def write_recovery_contract(job: CollisionRecoveryJob, logs_root: Path) -> Path:
    destination = logs_root / job.run_name / CONTRACT_FILENAME
    content = (
        json.dumps(
            {
                "recovery_manifest_sha256": job.manifest_sha256,
                "job": job.to_dict(),
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.read_text() != content:
        raise RuntimeError(f"existing recovery contract differs: {destination}")
    destination.write_text(content)
    return destination


def main() -> None:
    job = recovery_job()
    print(f"{job.run_name}\t{encode_recovery_job(job)}")


if __name__ == "__main__":
    main()
