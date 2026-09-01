from __future__ import annotations

import base64
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import ModuleType
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
JOB_ENVIRONMENT = "G4_COMPILED_JOB_B64"
LEDGER_ENVIRONMENT = "G4_LEDGER_PATH"
SEMANTICS_ENVIRONMENT = "G4_SEMANTICS_MANIFESTS_JSON"


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load G4 runtime source {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


manifest = _load_module(
    "g4_control_manifest_runtime",
    ROOT / "experiments/g4_future_items/protocol/manifest.py",
)
control = _load_module(
    "g4_control_config_runtime",
    ROOT / "experiments/g4_future_items/configs/control.py",
)


def _decode_document(encoded: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate compiled-job key {key!r}")
            result[key] = value
        return result

    try:
        decoded = base64.b64decode(encoded, altchars=b"-_", validate=True).decode()
        value = json.loads(decoded, object_pairs_hook=reject_duplicates)
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("compiled G4 control job is invalid") from error
    if not isinstance(value, dict) or set(value) != {
        "ledger_sha256",
        "row_id",
        "job",
    }:
        raise RuntimeError("compiled G4 control job has missing or unknown fields")
    return value


def _load_job() -> tuple[Any, Path]:
    encoded = os.environ.get(JOB_ENVIRONMENT)
    ledger_value = os.environ.get(LEDGER_ENVIRONMENT)
    semantics_value = os.environ.get(SEMANTICS_ENVIRONMENT)
    if not encoded or not ledger_value or not semantics_value:
        raise RuntimeError("G4 control job environment is incomplete")
    document = _decode_document(encoded)
    compiled = manifest.CompiledJob(
        ledger_sha256=document["ledger_sha256"],
        row_id=document["row_id"],
        job=document["job"],
    )
    ledger_path = Path(ledger_value)
    manifest.verify_compiled_job(compiled, ledger_path)
    try:
        raw_semantics = json.loads(semantics_value)
    except json.JSONDecodeError as error:
        raise RuntimeError("G4 control semantics environment is invalid") from error
    if not isinstance(raw_semantics, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in raw_semantics.items()
    ):
        raise RuntimeError("G4 control semantics environment is invalid")
    manifest.verify_ledger_semantics(
        manifest.load_ledger(ledger_path),
        {key: Path(value) for key, value in raw_semantics.items()},
    )
    return compiled, ledger_path


def _write_contract(compiled: Any, ledger_path: Path, logs_root: Path) -> None:
    destination = logs_root / compiled.job["run_name"] / "g4_job.json"
    document = compiled.to_dict() | {
        "ledger_path": str(ledger_path.resolve()),
        "ledger_stage": compiled.job["protocol"]["stage"],
    }
    content = json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.read_text() != content:
        raise RuntimeError(f"existing G4 contract differs: {destination}")
    destination.write_text(content)


compiled_job, ledger_path = _load_job()
if compiled_job.job["protocol"]["stage"] not in {
    "control_tuning",
    "control_tuning_boundary",
}:
    raise RuntimeError("run_control accepts only G4 control rows")
experiment = control.build_control(
    run_name=compiled_job.job["run_name"],
    batch_size=compiled_job.job["dataloader"]["batch_size"],
    embedding_learning_rate=compiled_job.job["embedding_learning_rate"],
    deep_learning_rate=compiled_job.job["deep_learning_rate"],
    lr_schedule_horizon_epochs=compiled_job.job["lr_schedule_horizon_epochs"],
    seed=compiled_job.job["seed"],
)
_write_contract(compiled_job, ledger_path, Path(experiment.base_path) / "logs")
