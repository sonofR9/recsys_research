from __future__ import annotations

import base64
import json
from pathlib import Path

from dcn.config import SemanticHistoryExperiment
from experiments.g6_rqkmeans_history.configs.rq0 import (
    build_learned_sid_residual_remediation,
)
from experiments.g6_rqkmeans_history.protocol.manifest import CompiledJob
from experiments.g6_rqkmeans_history.protocol.remediation_bounded import (
    bounded_gate_manifest,
    validate_bounded_gate_job,
)


JOB_ENVIRONMENT = "G6_RQ0_BOUNDED_GATE_JOB_B64"
CONTRACT_FILENAME = "g6_rq0_bounded_gate_job.json"


def build_bounded_gate_experiment(
    compiled: CompiledJob,
) -> SemanticHistoryExperiment:
    validate_bounded_gate_job(compiled)
    parameters = compiled.parameters
    return build_learned_sid_residual_remediation(
        backbone="best_g1",
        batch_size=parameters["batch_size"],
        validation_batch_size=parameters["validation_batch_size"],
        embedding_learning_rate=parameters["embedding_learning_rate"],
        deep_learning_rate=parameters["deep_learning_rate"],
        num_levels=parameters["num_levels"],
        num_codes=parameters["num_codes"],
        representation_width=parameters["representation_width"],
        frozen_event_width=parameters["frozen_event_width"],
        learned_residual_max_scale=parameters["learned_residual_max_scale"],
        run_name=compiled.run_name,
    )


def encode_bounded_gate_job(compiled: CompiledJob) -> str:
    validate_bounded_gate_job(compiled)
    payload = json.dumps(
        compiled.to_contract(bounded_gate_manifest()),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return base64.urlsafe_b64encode(payload).decode()


def decode_bounded_gate_job(encoded: str) -> CompiledJob:
    try:
        contract = json.loads(base64.urlsafe_b64decode(encoded).decode())
    except (ValueError, json.JSONDecodeError) as error:
        raise RuntimeError("compiled bounded-gate job is invalid") from error
    manifest = bounded_gate_manifest()
    if contract.get("manifest_sha256") != manifest.sha256:
        raise RuntimeError("bounded-gate job references a different manifest")
    matches = [job for job in manifest.jobs if job.to_dict() == contract.get("job")]
    parameters = contract.get("parameters")
    if len(matches) != 1 or not isinstance(parameters, dict):
        raise RuntimeError("bounded-gate job identity is not approved")
    compiled = CompiledJob(matches[0], parameters)
    validate_bounded_gate_job(compiled)
    return compiled


def write_bounded_gate_contract(compiled: CompiledJob, logs_root: Path) -> Path:
    validate_bounded_gate_job(compiled)
    destination = logs_root / compiled.run_name / CONTRACT_FILENAME
    content = (
        json.dumps(
            compiled.to_contract(bounded_gate_manifest()),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.read_text() != content:
        raise RuntimeError(f"existing bounded-gate contract differs: {destination}")
    destination.write_text(content)
    return destination
