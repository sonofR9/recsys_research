from __future__ import annotations

import base64
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

from dcn.config import SemanticHistoryExperiment
from experiments.g6_rqkmeans_history.configs.rq0 import (
    build_learned_sid_residual_remediation,
)
from experiments.g6_rqkmeans_history.protocol.manifest import CompiledJob
from experiments.g6_rqkmeans_history.protocol.remediation import (
    remediation_manifest,
    validate_remediation_job,
)


JOB_ENVIRONMENT = "G6_RQ0_REMEDIATION_JOB_B64"
CONTRACT_FILENAME = "g6_rq0_remediation_job.json"


def _training_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    excluded = {
        "builder",
        "source_job_id",
        "source_run_name",
        "source_attempt",
        "source_cap_epochs",
        "source_parameters",
        "boundary_side",
        "source_control_job_id",
        "source_control_run_name",
        "selected_treatment_job_id",
        "selected_treatment_run_name",
        "selected_treatment_attempt",
        "selected_treatment_cap_epochs",
        "selected_treatment_parameters",
        "selected_original_control_job_id",
        "selected_original_control_run_name",
        "representation",
    }
    return {name: value for name, value in parameters.items() if name not in excluded}


def build_remediation_experiment(compiled: CompiledJob) -> SemanticHistoryExperiment:
    validate_remediation_job(compiled)
    bridge = compiled.approved.stage in {
        "remediation_bridge_tuning",
        "remediation_bridge_lr_boundary",
    }
    experiment = build_learned_sid_residual_remediation(
        backbone="original_g1" if bridge else "best_g1",
        run_name=compiled.run_name,
        **_training_parameters(compiled.parameters),
    )
    overrides: dict[str, Any] = {"seed": compiled.approved.seed}
    if compiled.cap_epochs is not None:
        overrides["num_epochs"] = compiled.cap_epochs
    return replace(experiment, **overrides)


def encode_remediation_job(compiled: CompiledJob) -> str:
    validate_remediation_job(compiled)
    payload = json.dumps(
        compiled.to_contract(remediation_manifest()),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return base64.urlsafe_b64encode(payload).decode()


def decode_remediation_job(encoded: str) -> CompiledJob:
    try:
        contract = json.loads(base64.urlsafe_b64decode(encoded).decode())
    except (ValueError, json.JSONDecodeError) as error:
        raise RuntimeError("compiled G6 RQ0 remediation job is invalid") from error
    manifest = remediation_manifest()
    if contract.get("manifest_sha256") != manifest.sha256:
        raise RuntimeError(
            "compiled job references a different approved remediation manifest"
        )
    raw_job = contract.get("job")
    parameters = contract.get("parameters")
    if not isinstance(raw_job, dict) or not isinstance(parameters, dict):
        raise RuntimeError("compiled remediation contract is incomplete")
    matches = [job for job in manifest.jobs if job.to_dict() == raw_job]
    if len(matches) != 1:
        raise RuntimeError("compiled remediation job identity is not approved")
    continuation = contract.get("continuation")
    if continuation is None:
        attempt = 0
        cap_epochs = None
    elif isinstance(continuation, dict):
        attempt = continuation.get("attempt")
        cap_epochs = continuation.get("cap_epochs")
        if continuation.get("source_run_name") != matches[0].run_name:
            raise RuntimeError("compiled remediation continuation source changed")
    else:
        raise RuntimeError("compiled remediation continuation is invalid")
    compiled = CompiledJob(matches[0], parameters, attempt, cap_epochs)
    validate_remediation_job(compiled)
    return compiled


def write_remediation_contract(compiled: CompiledJob, logs_root: Path) -> Path:
    validate_remediation_job(compiled)
    destination = logs_root / compiled.run_name / CONTRACT_FILENAME
    content = (
        json.dumps(
            compiled.to_contract(remediation_manifest()),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.read_text() != content:
        raise RuntimeError(f"existing remediation contract differs: {destination}")
    destination.write_text(content)
    return destination
