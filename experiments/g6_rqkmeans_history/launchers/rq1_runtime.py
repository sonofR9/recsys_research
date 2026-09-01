from __future__ import annotations

import base64
import argparse
import json
import os
from pathlib import Path
from typing import Iterable

from experiments.g6_rqkmeans_history.protocol.rq1_manifest import (
    Rq1SearchJob,
    rq1_search_manifest,
)


JOB_ENVIRONMENT = "G6_RQ1_JOB_B64"
CONTRACT_FILENAME = "g6_rq1_job.json"


def encode_rq1_job(job: Rq1SearchJob) -> str:
    if job not in rq1_search_manifest().jobs:
        raise ValueError("job is outside the approved RQ1 paired search")
    if job.reused:
        raise ValueError("RQ1 carryover must not be rebuilt")
    payload = json.dumps(
        {
            "manifest_sha256": rq1_search_manifest().sha256,
            "job": job.to_dict(),
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return base64.urlsafe_b64encode(payload).decode()


def decode_rq1_job(encoded: str) -> Rq1SearchJob:
    try:
        contract = json.loads(base64.urlsafe_b64decode(encoded).decode())
    except (ValueError, json.JSONDecodeError) as error:
        raise RuntimeError("encoded RQ1 job is invalid") from error
    manifest = rq1_search_manifest()
    if set(contract) != {"manifest_sha256", "job"}:
        raise RuntimeError("RQ1 job contract has unexpected keys")
    if contract["manifest_sha256"] != manifest.sha256:
        raise RuntimeError("RQ1 job references a different approved manifest")
    matches = [
        job
        for job in manifest.new_physical_jobs
        if job.to_dict() == contract["job"]
    ]
    if len(matches) != 1:
        raise RuntimeError("RQ1 job is absent from the approved manifest")
    return matches[0]


def rq1_job_from_environment() -> Rq1SearchJob:
    encoded = os.environ.get(JOB_ENVIRONMENT)
    if not encoded:
        raise RuntimeError(f"{JOB_ENVIRONMENT} is required")
    return decode_rq1_job(encoded)


def write_rq1_contract(job: Rq1SearchJob, logs_root: Path) -> Path:
    destination = logs_root / job.run_name / CONTRACT_FILENAME
    content = (
        json.dumps(
            {
                "manifest_sha256": rq1_search_manifest().sha256,
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
        raise RuntimeError(f"existing RQ1 job contract differs: {destination}")
    destination.write_text(content)
    return destination


def initial_rq1_jobs() -> tuple[Rq1SearchJob, ...]:
    return rq1_search_manifest().new_physical_jobs


def rq1_cache_safe_phase(index: int) -> tuple[Rq1SearchJob, ...]:
    jobs = initial_rq1_jobs()
    phases = (jobs[:1], jobs[1:])
    if index < 0 or index >= len(phases):
        raise ValueError(f"phase index must be in [0, {len(phases)})")
    return phases[index]


def emit_rows(jobs: Iterable[Rq1SearchJob]) -> tuple[str, ...]:
    return tuple("\t".join((job.run_name, encode_rq1_job(job))) for job in jobs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int)
    arguments = parser.parse_args()
    jobs = (
        rq1_cache_safe_phase(arguments.phase)
        if arguments.phase is not None
        else initial_rq1_jobs()
    )
    print("\n".join(emit_rows(jobs)))


if __name__ == "__main__":
    main()
