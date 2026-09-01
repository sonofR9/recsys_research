from __future__ import annotations

import argparse
import base64
from dataclasses import replace
import json
import os
from pathlib import Path
from typing import Any

from dcn.config import MuTransferGenerationExperiment, SemanticHistoryExperiment
from experiments.g6_rqkmeans_history.configs.rq0 import (
    build_control,
    build_semantic_treatment,
)
from experiments.g6_rqkmeans_history.protocol.manifest import (
    CompiledJob,
    approved_manifest,
    load_compiled_jobs,
    validate_compiled_job,
)


JOB_ENVIRONMENT = "G6_RQ0_COMPILED_JOB_B64"


def _training_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    excluded = {
        "builder",
        "source_job_id",
        "source_parameters",
        "boundary_side",
        "selected_primary_control_job_id",
        "selected_original_control_job_id",
        "selected_treatment_job_id",
    }
    return {name: value for name, value in parameters.items() if name not in excluded}


def build_experiment(
    compiled: CompiledJob,
) -> MuTransferGenerationExperiment | SemanticHistoryExperiment:
    validate_compiled_job(compiled)
    job = compiled.approved
    parameters = _training_parameters(compiled.parameters)
    if job.stage in {"primary_control_tuning", "primary_control_repeats"}:
        builder = "primary_control"
    elif job.stage == "original_control_tuning":
        builder = "original_control"
    elif job.stage == "treatment_tuning":
        builder = "treatment"
    elif job.stage == "bridge_tuning":
        builder = "bridge"
    elif job.stage == "lr_boundary":
        builder = compiled.parameters["builder"]
    else:
        raise ValueError(f"{job.id} is not a G6 RQ0 training job")

    if builder in {"primary_control", "original_control"}:
        backbone = "best_g1" if builder == "primary_control" else "original_g1"
        experiment = build_control(backbone, run_name=compiled.run_name, **parameters)
    elif builder in {"treatment", "bridge"}:
        representation = parameters.pop("representation")
        backbone = "best_g1" if builder == "treatment" else "original_g1"
        experiment = build_semantic_treatment(
            representation,
            backbone=backbone,
            run_name=compiled.run_name,
            **parameters,
        )
    else:
        raise ValueError(f"{job.id}: unknown boundary builder {builder!r}")
    overrides: dict[str, Any] = {"seed": job.seed}
    if compiled.cap_epochs is not None:
        overrides["num_epochs"] = compiled.cap_epochs
    return replace(experiment, **overrides)


def encode_compiled_job(compiled: CompiledJob) -> str:
    validate_compiled_job(compiled)
    payload = json.dumps(
        compiled.to_contract(approved_manifest()),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return base64.urlsafe_b64encode(payload).decode()


def decode_compiled_job(encoded: str) -> CompiledJob:
    try:
        contract = json.loads(base64.urlsafe_b64decode(encoded).decode())
    except (ValueError, json.JSONDecodeError) as error:
        raise RuntimeError("compiled G6 RQ0 job is invalid") from error
    manifest = approved_manifest()
    if contract.get("manifest_sha256") != manifest.sha256:
        raise RuntimeError("compiled job references a different approved manifest")
    raw_job = contract.get("job")
    parameters = contract.get("parameters")
    if not isinstance(raw_job, dict) or not isinstance(parameters, dict):
        raise RuntimeError("compiled job contract is incomplete")
    matches = [job for job in manifest.jobs if job.to_dict() == raw_job]
    if len(matches) != 1:
        raise RuntimeError("compiled job identity is not approved")
    continuation = contract.get("continuation")
    if continuation is None:
        attempt = 0
        cap_epochs = None
    elif isinstance(continuation, dict):
        attempt = continuation.get("attempt")
        cap_epochs = continuation.get("cap_epochs")
        if continuation.get("source_run_name") != matches[0].run_name:
            raise RuntimeError("compiled continuation source changed")
    else:
        raise RuntimeError("compiled continuation is invalid")
    compiled = CompiledJob(matches[0], parameters, attempt, cap_epochs)
    validate_compiled_job(compiled)
    return compiled


def compiled_job_from_environment() -> CompiledJob:
    encoded = os.environ.get(JOB_ENVIRONMENT)
    if not encoded:
        raise RuntimeError(f"{JOB_ENVIRONMENT} is required")
    return decode_compiled_job(encoded)


def write_job_contract(compiled: CompiledJob, logs_root: Path) -> Path:
    validate_compiled_job(compiled)
    destination = logs_root / compiled.run_name / "g6_rq0_job.json"
    content = (
        json.dumps(
            compiled.to_contract(approved_manifest()),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.read_text() != content:
        raise RuntimeError(f"existing G6 RQ0 contract differs: {destination}")
    destination.write_text(content)
    return destination


def emit_rows(path: Path) -> list[str]:
    return [
        "\t".join(
            (
                compiled.run_name,
                encode_compiled_job(compiled),
            )
        )
        for compiled in load_compiled_jobs(path)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("compiled_manifest", type=Path)
    arguments = parser.parse_args()
    print("\n".join(emit_rows(arguments.compiled_manifest)))


if __name__ == "__main__":
    main()
