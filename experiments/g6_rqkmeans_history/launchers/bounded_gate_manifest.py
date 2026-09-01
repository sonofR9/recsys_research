from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.g6_rqkmeans_history.launchers.remediation_bounded import (
    encode_bounded_gate_job,
)
from experiments.g6_rqkmeans_history.protocol.manifest import CompiledJob
from experiments.g6_rqkmeans_history.protocol.remediation_bounded import (
    bounded_gate_jobs,
    bounded_gate_manifest,
    validate_bounded_gate_job,
)


def write_bounded_gate_jobs(path: Path, jobs: tuple[CompiledJob, ...]) -> None:
    if jobs != bounded_gate_jobs():
        raise ValueError("bounded-gate ledger must contain the exact approved grid")
    for compiled in jobs:
        validate_bounded_gate_job(compiled)
    document = {
        "manifest_sha256": bounded_gate_manifest().sha256,
        "jobs": [
            {"id": compiled.approved.id, "parameters": compiled.parameters}
            for compiled in jobs
        ],
    }
    content = json.dumps(document, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text() != content:
        raise RuntimeError(f"existing bounded-gate ledger differs: {path}")
    path.write_text(content)


def load_bounded_gate_jobs(path: Path) -> tuple[CompiledJob, ...]:
    try:
        document = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read bounded-gate ledger {path}") from error
    if document.get("manifest_sha256") != bounded_gate_manifest().sha256:
        raise ValueError("bounded-gate ledger references a different manifest")
    approved = {job.approved.id: job for job in bounded_gate_jobs()}
    rows = document.get("jobs")
    if not isinstance(rows, list):
        raise ValueError("bounded-gate ledger jobs are absent")
    compiled = []
    for row in rows:
        if not isinstance(row, dict) or row.get("id") not in approved:
            raise ValueError("bounded-gate ledger contains an unknown job")
        candidate = CompiledJob(approved[row["id"]].approved, row.get("parameters"))
        validate_bounded_gate_job(candidate)
        compiled.append(candidate)
    if len({job.identity for job in compiled}) != len(compiled):
        raise ValueError("bounded-gate ledger contains duplicate jobs")
    result = tuple(compiled)
    if result != bounded_gate_jobs():
        raise ValueError("bounded-gate ledger must contain the exact approved grid")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ledger", type=Path)
    arguments = parser.parse_args()
    for compiled in load_bounded_gate_jobs(arguments.ledger):
        print(f"{compiled.run_name}\t{encode_bounded_gate_job(compiled)}")


if __name__ == "__main__":
    main()
