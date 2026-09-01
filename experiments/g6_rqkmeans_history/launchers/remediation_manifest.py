from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.g6_rqkmeans_history.launchers.remediation_compiled import (
    encode_remediation_job,
)
from experiments.g6_rqkmeans_history.protocol.manifest import CompiledJob
from experiments.g6_rqkmeans_history.protocol.remediation import (
    remediation_manifest,
    validate_remediation_job,
)


def load_remediation_jobs(path: Path) -> list[CompiledJob]:
    try:
        document = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read remediation manifest {path}") from error
    manifest = remediation_manifest()
    if document.get("manifest_sha256") != manifest.sha256:
        raise ValueError("ledger references a different remediation manifest")
    rows = document.get("jobs")
    if not isinstance(rows, list):
        raise ValueError("remediation ledger jobs must be a list")
    approved = {job.id: job for job in manifest.jobs}
    compiled: list[CompiledJob] = []
    for row in rows:
        if not isinstance(row, dict) or row.get("id") not in approved:
            raise ValueError("remediation ledger contains an unknown job")
        parameters = row.get("parameters")
        if not isinstance(parameters, dict):
            raise ValueError("remediation ledger parameters are absent")
        candidate = CompiledJob(
            approved[str(row["id"])],
            parameters,
            row.get("attempt", 0),
            row.get("cap_epochs"),
        )
        validate_remediation_job(candidate)
        compiled.append(candidate)
    if len({job.identity for job in compiled}) != len(compiled):
        raise ValueError("remediation ledger contains duplicate jobs")
    return compiled


def write_remediation_jobs(path: Path, jobs: list[CompiledJob]) -> None:
    for compiled in jobs:
        validate_remediation_job(compiled)
    document = {
        "manifest_sha256": remediation_manifest().sha256,
        "jobs": [
            {
                "id": compiled.approved.id,
                "parameters": compiled.parameters,
                **(
                    {
                        "attempt": compiled.attempt,
                        "cap_epochs": compiled.cap_epochs,
                    }
                    if compiled.attempt
                    else {}
                ),
            }
            for compiled in jobs
        ],
    }
    content = json.dumps(document, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ledger", type=Path)
    arguments = parser.parse_args()
    for compiled in load_remediation_jobs(arguments.ledger):
        print(f"{compiled.run_name}\t{encode_remediation_job(compiled)}")


if __name__ == "__main__":
    main()
