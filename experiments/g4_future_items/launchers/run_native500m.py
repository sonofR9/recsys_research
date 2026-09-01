from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any

from experiments.g4_future_items.launchers.native500m import (
    JOB_ENVIRONMENT,
    LEDGER_ENVIRONMENT,
)
from experiments.g4_future_items.protocol.native500m.manifest import (
    build_runtime_experiment,
    canonical_bytes,
    load_frozen_ledger,
    validate_native500m_data_identity,
    validate_native_source_closure,
)


def load_compiled_job() -> tuple[dict[str, Any], Path]:
    encoded = os.environ.get(JOB_ENVIRONMENT)
    raw_ledger_path = os.environ.get(LEDGER_ENVIRONMENT)
    if not encoded or not raw_ledger_path:
        raise RuntimeError(f"{JOB_ENVIRONMENT} and {LEDGER_ENVIRONMENT} are required")
    try:
        payload = json.loads(
            base64.b64decode(encoded, altchars=b"-_", validate=True).decode()
        )
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("compiled native-500M G4 job is invalid") from error
    if not isinstance(payload, dict) or set(payload) != {
        "ledger_sha256",
        "row_id",
        "job",
        "source_closure",
        "data_identity",
    }:
        raise RuntimeError("compiled native-500M G4 job schema differs")
    try:
        validate_native_source_closure(payload["source_closure"])
    except ValueError as error:
        raise RuntimeError("compiled native-500M source closure differs") from error
    ledger_path = Path(raw_ledger_path).resolve()
    ledger = load_frozen_ledger(ledger_path)
    if payload["ledger_sha256"] != ledger["sha256"]:
        raise RuntimeError("compiled job references another native-500M ledger")
    matching = [row for row in ledger["rows"] if row["id"] == payload["row_id"]]
    if len(matching) != 1 or canonical_bytes(matching[0]["job"]) != canonical_bytes(
        payload["job"]
    ):
        raise RuntimeError("compiled job differs from its native-500M ledger row")
    return payload, ledger_path


def build_experiment(job: dict[str, Any]):
    return build_runtime_experiment(job)


def validate_compiled_data_identity(
    payload: dict[str, Any], runtime_experiment: Any
) -> None:
    try:
        validate_native500m_data_identity(payload["data_identity"], runtime_experiment)
    except (KeyError, ValueError) as error:
        raise RuntimeError("compiled native-500M data identity differs") from error


def write_job_contract(
    payload: dict[str, Any], ledger_path: Path, logs_root: Path
) -> Path:
    document = payload | {"ledger_path": str(ledger_path)}
    content = (
        json.dumps(
            document,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )
    destination = logs_root / payload["job"]["run_name"] / "g4_job.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.read_text() != content:
        raise RuntimeError(f"existing native-500M G4 contract differs: {destination}")
    destination.write_text(content)
    return destination


experiment = None
if os.environ.get(JOB_ENVIRONMENT) is not None:
    compiled_job, compiled_ledger_path = load_compiled_job()
    experiment = build_experiment(compiled_job["job"])
    validate_compiled_data_identity(compiled_job, experiment)
    write_job_contract(
        compiled_job,
        compiled_ledger_path,
        Path(experiment.base_path) / "logs",
    )
