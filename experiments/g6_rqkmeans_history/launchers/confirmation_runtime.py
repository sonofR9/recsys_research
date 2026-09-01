from __future__ import annotations

import base64
import json
import os
from pathlib import Path

from experiments.g6_rqkmeans_history.protocol.confirmation import (
    ConfirmationJob,
    ConfirmationManifest,
    load_confirmation_manifest,
)


JOB_ENVIRONMENT = "G6_CONFIRMATION_JOB_B64"
CONTRACT_FILENAME = "g6_confirmation_job.json"


def encode_job(job: ConfirmationJob, manifest: ConfirmationManifest) -> str:
    if job not in manifest.jobs:
        raise ValueError("job is outside the frozen confirmation manifest")
    payload = json.dumps(
        {"manifest_sha256": manifest.sha256, "job": job.to_dict()},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return base64.urlsafe_b64encode(payload).decode()


def decode_job(encoded: str, manifest: ConfirmationManifest) -> ConfirmationJob:
    try:
        contract = json.loads(base64.urlsafe_b64decode(encoded).decode())
    except (ValueError, json.JSONDecodeError) as error:
        raise RuntimeError("encoded confirmation job is invalid") from error
    if set(contract) != {"manifest_sha256", "job"}:
        raise RuntimeError("confirmation job contract has unexpected keys")
    if contract["manifest_sha256"] != manifest.sha256:
        raise RuntimeError("confirmation job references another manifest")
    matches = [job for job in manifest.jobs if job.to_dict() == contract["job"]]
    if len(matches) != 1:
        raise RuntimeError("confirmation job is absent from frozen manifest")
    return matches[0]


def job_from_environment() -> tuple[ConfirmationJob, ConfirmationManifest]:
    encoded = os.environ.get(JOB_ENVIRONMENT)
    if not encoded:
        raise RuntimeError(f"{JOB_ENVIRONMENT} is required")
    manifest = load_confirmation_manifest()
    return decode_job(encoded, manifest), manifest


def write_contract(
    job: ConfirmationJob, manifest: ConfirmationManifest, logs_root: Path
) -> Path:
    destination = logs_root / job.run_name / CONTRACT_FILENAME
    content = json.dumps(
        {"manifest_sha256": manifest.sha256, "job": job.to_dict()},
        indent=2,
        sort_keys=True,
    ) + "\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.read_text() != content:
        raise RuntimeError(f"existing confirmation contract differs: {destination}")
    destination.write_text(content)
    return destination


def main() -> None:
    manifest = load_confirmation_manifest()
    print(
        "\n".join(
            "\t".join((job.run_name, encode_job(job, manifest)))
            for job in manifest.jobs
        )
    )


if __name__ == "__main__":
    main()
