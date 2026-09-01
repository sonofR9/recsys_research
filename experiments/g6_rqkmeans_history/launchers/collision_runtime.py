from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
from typing import Iterable

from experiments.g6_rqkmeans_history.protocol.collision_policy import (
    CollisionSearchJob,
    collision_search_manifest,
)


JOB_ENVIRONMENT = "G6_RQ23_JOB_B64"
CONTRACT_FILENAME = "g6_rq2_rq3_job.json"


def encode_collision_job(job: CollisionSearchJob) -> str:
    if job not in collision_search_manifest().jobs:
        raise ValueError("job is outside the approved paired search")
    if job.reused:
        raise ValueError("collision carryover must not be rebuilt")
    contract = {
        "manifest_sha256": collision_search_manifest().sha256,
        "job": job.to_dict(),
    }
    payload = json.dumps(
        contract,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return base64.urlsafe_b64encode(payload).decode()


def decode_collision_job(encoded: str) -> CollisionSearchJob:
    try:
        contract = json.loads(base64.urlsafe_b64decode(encoded).decode())
    except (ValueError, json.JSONDecodeError) as error:
        raise RuntimeError("encoded collision job is invalid") from error
    manifest = collision_search_manifest()
    if set(contract) != {"manifest_sha256", "job"}:
        raise RuntimeError("collision job contract has unexpected keys")
    if contract["manifest_sha256"] != manifest.sha256:
        raise RuntimeError("collision job references a different approved manifest")
    matches = [
        job
        for job in manifest.new_physical_jobs
        if job.to_dict() == contract["job"]
    ]
    if len(matches) != 1:
        raise RuntimeError("collision job is absent from the approved manifest")
    return matches[0]


def collision_job_from_environment() -> CollisionSearchJob:
    encoded = os.environ.get(JOB_ENVIRONMENT)
    if not encoded:
        raise RuntimeError(f"{JOB_ENVIRONMENT} is required")
    return decode_collision_job(encoded)


def write_collision_contract(job: CollisionSearchJob, logs_root: Path) -> Path:
    destination = logs_root / job.run_name / CONTRACT_FILENAME
    content = (
        json.dumps(
            {
                "manifest_sha256": collision_search_manifest().sha256,
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
        raise RuntimeError(f"existing collision job contract differs: {destination}")
    destination.write_text(content)
    return destination


def initial_physical_jobs() -> tuple[CollisionSearchJob, ...]:
    return tuple(
        job
        for job in collision_search_manifest().jobs
        if not (job.policy == "suffix" and job.coordinate.trial == 0)
    )


def cache_safe_waves(
    jobs: Iterable[CollisionSearchJob],
) -> tuple[tuple[CollisionSearchJob, ...], ...]:
    grouped: dict[tuple[int, int, int], list[CollisionSearchJob]] = {}
    for job in jobs:
        coordinate = job.coordinate
        key = (
            coordinate.num_levels,
            coordinate.num_codes,
            coordinate.kmeans_iterations,
        )
        grouped.setdefault(key, []).append(job)
    if not grouped:
        return ()
    return tuple(
        tuple(group[round_index] for group in grouped.values() if len(group) > round_index)
        for round_index in range(max(map(len, grouped.values())))
    )


def emit_rows(jobs: Iterable[CollisionSearchJob]) -> tuple[str, ...]:
    return tuple(
        "\t".join((job.run_name, encode_collision_job(job))) for job in jobs
    )


def select_jobs(job_ids: Iterable[str]) -> tuple[CollisionSearchJob, ...]:
    requested = tuple(job_ids)
    if not requested:
        return initial_physical_jobs()
    if len(set(requested)) != len(requested):
        raise ValueError("collision job IDs must be unique")
    by_id = {job.id: job for job in initial_physical_jobs()}
    unknown = [job_id for job_id in requested if job_id not in by_id]
    if unknown:
        raise ValueError(f"unknown physical collision jobs: {unknown}")
    return tuple(by_id[job_id] for job_id in requested)


def select_wave(
    index: int, job_ids: Iterable[str] = ()
) -> tuple[CollisionSearchJob, ...]:
    waves = cache_safe_waves(select_jobs(job_ids))
    if index < 0 or index >= len(waves):
        raise ValueError(f"wave index must be in [0, {len(waves)})")
    return waves[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wave", type=int)
    parser.add_argument("job_ids", nargs="*")
    arguments = parser.parse_args()
    jobs = (
        select_wave(arguments.wave, arguments.job_ids)
        if arguments.wave is not None
        else select_jobs(arguments.job_ids)
    )
    print("\n".join(emit_rows(jobs)))


if __name__ == "__main__":
    main()
