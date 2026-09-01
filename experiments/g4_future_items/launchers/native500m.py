from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import subprocess
import tempfile
from typing import Any

from experiments.g4_future_items.protocol.native500m.manifest import (
    build_native_source_closure,
    build_runtime_experiment,
    canonical_bytes,
    load_frozen_ledger,
    resolve_native500m_data_identity,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
JOB_ENVIRONMENT = "G4_NATIVE500M_JOB_B64"
LEDGER_ENVIRONMENT = "G4_NATIVE500M_LEDGER_PATH"


def build_queue_specification(
    ledger_path: Path, *, row_id: str | None = None
) -> dict[str, Any]:
    ledger_path = ledger_path.resolve()
    ledger = load_frozen_ledger(ledger_path)
    if ledger.get("retry_revision") == 2 and row_id is None:
        raise ValueError("native-500M retry-2 requires one explicit row")
    rows = ledger["rows"]
    if row_id is not None:
        rows = [row for row in rows if row["id"] == row_id]
        if len(rows) != 1:
            raise ValueError(f"native-500M ledger has no unique row {row_id!r}")
    source_closure = build_native_source_closure()
    identity_experiment = build_runtime_experiment(rows[0]["job"])
    data_identity = resolve_native500m_data_identity(identity_experiment)
    script = Path(__file__).with_name("run_native500m.py").resolve()
    jobs = []
    for row in rows:
        payload = {
            "ledger_sha256": ledger["sha256"],
            "row_id": row["id"],
            "job": row["job"],
            "source_closure": source_closure,
            "data_identity": data_identity,
        }
        encoded = base64.urlsafe_b64encode(canonical_bytes(payload)).decode()
        jobs.append(
            {
                "script": str(script),
                "run": row["job"]["run_name"],
                "data_group": "g4-native500m-likes",
                "environment": [
                    f"{JOB_ENVIRONMENT}={encoded}",
                    f"{LEDGER_ENVIRONMENT}={ledger_path}",
                    "WANDB_MODE=offline",
                ],
            }
        )
    return {"version": 1, "jobs": jobs}


def submit_ledger(
    *,
    ledger_path: Path,
    state_directory: Path,
    dry_run: bool,
    row_id: str | None = None,
) -> str:
    specification = build_queue_specification(ledger_path, row_id=row_id)
    canonical = canonical_bytes(specification)
    if dry_run:
        return canonical.decode()
    state_directory = state_directory.resolve()
    with tempfile.NamedTemporaryFile(
        mode="wb",
        prefix="g4-native500m-",
        suffix=".json",
        dir=PROJECT_ROOT / "generated",
        delete=False,
    ) as handle:
        handle.write(canonical)
        specification_path = Path(handle.name)
    try:
        completed = subprocess.run(
            [
                "python",
                str(PROJECT_ROOT / "utils/training_queue/service.py"),
                "--state-dir",
                str(state_directory),
                "submit-batch",
                str(specification_path),
            ],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    finally:
        specification_path.unlink(missing_ok=True)
    batch_id = completed.stdout.strip()
    if not batch_id:
        raise RuntimeError("training queue returned an empty batch id")
    return batch_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ledger", type=Path)
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=PROJECT_ROOT / "generated/training-queue-service",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--row-id")
    arguments = parser.parse_args()
    print(
        submit_ledger(
            ledger_path=arguments.ledger,
            state_directory=arguments.state_dir,
            dry_run=arguments.dry_run,
            row_id=arguments.row_id,
        )
    )


if __name__ == "__main__":
    main()
