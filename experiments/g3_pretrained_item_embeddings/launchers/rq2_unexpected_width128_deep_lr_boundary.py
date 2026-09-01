from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shlex
import subprocess

from experiments.g3_pretrained_item_embeddings.analysis.rq2_unexpected_diagnostic_results import (
    load_rq2_unexpected_diagnostic_evidence,
)
from experiments.g3_pretrained_item_embeddings.launchers.control import (
    PROJECT_ROOT,
    CompiledControlJob,
    compile_queue_commands,
    decode_control_job,
    encode_control_job,
    find_existing_ledger_batch,
    ledger_submission_lock,
    persist_job_contract,
)
from experiments.g3_pretrained_item_embeddings.launchers.rq2_unexpected_diagnostic import (
    verify_rq2_unexpected_diagnostic_inputs,
)
from experiments.g3_pretrained_item_embeddings.protocol.rq2_unexpected_diagnostic_ledger import (
    load_rq2_unexpected_diagnostic_ledger,
)
from experiments.g3_pretrained_item_embeddings.protocol.rq2_unexpected_width128_deep_lr_boundary_ledger import (
    APPROVED_RQ2_UNEXPECTED_DIAGNOSTIC_EVIDENCE_SHA256,
    APPROVED_RQ2_UNEXPECTED_WIDTH128_BOUNDARY_LEDGER_SHA256,
    Rq2UnexpectedWidth128BoundaryLedger,
    load_rq2_unexpected_width128_boundary_ledger,
    validate_rq2_unexpected_width128_boundary_ledger_document,
)


JOB_ENVIRONMENT = "G3_RQ2_UNEXPECTED_WIDTH128_BOUNDARY_JOB_B64"
LEDGER_ENVIRONMENT = "G3_RQ2_UNEXPECTED_WIDTH128_BOUNDARY_LEDGER_PATH"


def compile_rq2_unexpected_width128_boundary_queue_commands(
    *,
    ledger_path: Path,
    ledger: Rq2UnexpectedWidth128BoundaryLedger,
    state_dir: Path,
    batch_id: str = "DRY_RUN_BATCH",
) -> list[list[str]]:
    if len(ledger.rows) != 3 or any(row.to_dict()["reused_from"] is not None for row in ledger.rows):
        raise ValueError("RQ2 width128 boundary queue requires exactly three physical jobs")
    return compile_queue_commands(
        ledger_path=ledger_path,
        ledger=ledger,
        state_dir=state_dir,
        batch_id=batch_id,
        runner_script=Path(__file__).with_name(
            "run_rq2_unexpected_width128_deep_lr_boundary.py"
        ),
        job_environment=JOB_ENVIRONMENT,
        ledger_environment=LEDGER_ENVIRONMENT,
        ledger_validator=lambda document: validate_rq2_unexpected_width128_boundary_ledger_document(
            document, root=PROJECT_ROOT
        ),
    )


def submit_rq2_unexpected_width128_boundary_jobs(
    *, ledger_path: Path, state_dir: Path, dry_run: bool
) -> str:
    ledger_path = ledger_path.resolve()
    ledger = load_rq2_unexpected_width128_boundary_ledger(
        ledger_path, root=PROJECT_ROOT
    )
    verify_rq2_unexpected_width128_boundary_inputs(
        PROJECT_ROOT, ledger, full_validation=True
    )
    if dry_run:
        return "\n".join(
            shlex.join(command)
            for command in compile_rq2_unexpected_width128_boundary_queue_commands(
                ledger_path=ledger_path, ledger=ledger, state_dir=state_dir
            )
        )
    with ledger_submission_lock(state_dir=state_dir, ledger_sha256=ledger.sha256):
        return _submit(
            ledger_path=ledger_path, ledger=ledger, state_dir=state_dir
        )


def _submit(
    *,
    ledger_path: Path,
    ledger: Rq2UnexpectedWidth128BoundaryLedger,
    state_dir: Path,
) -> str:
    runner_script = Path(__file__).with_name(
        "run_rq2_unexpected_width128_deep_lr_boundary.py"
    )
    existing = find_existing_ledger_batch(
        state_dir=state_dir,
        ledger_path=ledger_path,
        ledger=ledger,
        runner_script=runner_script,
        job_environment=JOB_ENVIRONMENT,
        ledger_environment=LEDGER_ENVIRONMENT,
    )
    if existing is not None:
        return existing
    initial = compile_rq2_unexpected_width128_boundary_queue_commands(
        ledger_path=ledger_path, ledger=ledger, state_dir=state_dir
    )
    subprocess.run(initial[0], cwd=PROJECT_ROOT, check=True, capture_output=True)
    created = subprocess.run(
        initial[1], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True
    )
    batch_id = created.stdout.strip()
    if not batch_id:
        raise RuntimeError("training queue returned an empty batch id")
    commands = compile_rq2_unexpected_width128_boundary_queue_commands(
        ledger_path=ledger_path,
        ledger=ledger,
        state_dir=state_dir,
        batch_id=batch_id,
    )
    for command in commands[2:]:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    return batch_id


def verify_rq2_unexpected_width128_boundary_inputs(
    root: Path,
    ledger: Rq2UnexpectedWidth128BoundaryLedger,
    *,
    full_validation: bool,
) -> Path:
    root = root.resolve(strict=True)
    if ledger.sha256 != APPROVED_RQ2_UNEXPECTED_WIDTH128_BOUNDARY_LEDGER_SHA256:
        raise ValueError("RQ2 width128 boundary ledger is not approved")
    evidence_path = _resolve_reference(root, ledger.diagnostic_evidence.path)
    evidence_bytes = evidence_path.read_bytes()
    if (
        len(evidence_bytes) != ledger.diagnostic_evidence.size_bytes
        or hashlib.sha256(evidence_bytes).hexdigest() != ledger.diagnostic_evidence.sha256
    ):
        raise ValueError("RQ2 width128 boundary diagnostic evidence file changed")
    if full_validation:
        evidence = load_rq2_unexpected_diagnostic_evidence(evidence_path, root=root)
        if evidence.get("sha256") != APPROVED_RQ2_UNEXPECTED_DIAGNOSTIC_EVIDENCE_SHA256:
            raise ValueError("RQ2 width128 boundary diagnostic evidence changed")
    diagnostic_ledger_path = _resolve_reference(root, ledger.diagnostic_ledger.path)
    diagnostic_ledger_bytes = diagnostic_ledger_path.read_bytes()
    if (
        len(diagnostic_ledger_bytes) != ledger.diagnostic_ledger.size_bytes
        or hashlib.sha256(diagnostic_ledger_bytes).hexdigest()
        != ledger.diagnostic_ledger.sha256
    ):
        raise ValueError("RQ2 width128 boundary diagnostic ledger file changed")
    diagnostic_ledger = load_rq2_unexpected_diagnostic_ledger(
        diagnostic_ledger_path, root=root
    )
    if diagnostic_ledger.sha256 != ledger.diagnostic_ledger.logical_sha256:
        raise ValueError("RQ2 width128 boundary diagnostic ledger changed")
    return verify_rq2_unexpected_diagnostic_inputs(
        root, diagnostic_ledger, full_validation=False
    )


def compiled_rq2_unexpected_width128_boundary_job_from_environment() -> tuple[
    CompiledControlJob,
    Rq2UnexpectedWidth128BoundaryLedger,
    Path,
    Path,
]:
    encoded = os.environ.get(JOB_ENVIRONMENT)
    raw_ledger_path = os.environ.get(LEDGER_ENVIRONMENT)
    if not encoded or not raw_ledger_path:
        raise RuntimeError(f"{JOB_ENVIRONMENT} and {LEDGER_ENVIRONMENT} are required")
    ledger_path = Path(raw_ledger_path)
    ledger = load_rq2_unexpected_width128_boundary_ledger(
        ledger_path, root=PROJECT_ROOT
    )
    feature_data_path = verify_rq2_unexpected_width128_boundary_inputs(
        PROJECT_ROOT, ledger, full_validation=False
    )
    return decode_control_job(encoded, ledger), ledger, ledger_path, feature_data_path


def build_training_experiment(
    compiled: CompiledControlJob,
    *,
    ledger: Rq2UnexpectedWidth128BoundaryLedger,
    feature_data_path: Path,
):
    if ledger.sha256 != APPROVED_RQ2_UNEXPECTED_WIDTH128_BOUNDARY_LEDGER_SHA256:
        raise ValueError("RQ2 width128 boundary ledger is not approved")
    expected = decode_control_job(encode_control_job(ledger, compiled.row_id), ledger)
    if expected != compiled:
        raise ValueError("compiled RQ2 width128 boundary job differs from ledger")
    from experiments.g3_pretrained_item_embeddings.configs.model import (
        G3Representation,
        build_g3_experiment,
    )

    job = compiled.job
    representation = job.get("representation")
    training = job.get("training")
    dataset = job.get("dataset")
    allowed_rates = {0.005733564587228046, 0.0040542424, 0.002866782293614023}
    if (
        not all(isinstance(value, dict) for value in (representation, training, dataset))
        or job.get("family_id") != "rq2_content_concat"
        or job.get("phase") != "unexpected_width128_deep_learning_rate_lower_boundary"
        or representation.get("history_hidden_dim") != 128
        or representation.get("content_trainable") is not False
        or dataset.get("size") != "native-50m"
        or training.get("batch_size") != 512
        or training.get("seed") != 42
        or training.get("embedding_learning_rate") != 0.3041556165944196
        or training.get("deep_learning_rate") not in allowed_rates
        or training.get("horizon_epochs") != 40
    ):
        raise ValueError("compiled G3 RQ2 width128 boundary coordinate is invalid")
    return build_g3_experiment(
        run_name=str(job["run_name"]),
        dataset_size=str(dataset["size"]),
        embedding_learning_rate=float(training["embedding_learning_rate"]),
        deep_learning_rate=float(training["deep_learning_rate"]),
        lr_schedule_horizon_epochs=int(training["horizon_epochs"]),
        seed=int(training["seed"]),
        representation=G3Representation(
            history_representation="id_content",
            catalog_representation="learned_id",
            history_hidden_dim=128,
        ),
        feature_data_path=feature_data_path,
    )


def write_job_contract(
    compiled: CompiledControlJob, ledger_path: Path, logs_root: Path
) -> Path:
    ledger = load_rq2_unexpected_width128_boundary_ledger(
        ledger_path, root=PROJECT_ROOT
    )
    verified = decode_control_job(encode_control_job(ledger, compiled.row_id), ledger)
    if verified != compiled:
        raise ValueError("compiled RQ2 width128 boundary job differs from ledger")
    return persist_job_contract(
        compiled=compiled,
        ledger_path=ledger_path,
        ledger_sha256=ledger.sha256,
        logs_root=logs_root,
        filename="g3_rq2_unexpected_width128_boundary_job.json",
    )


def _resolve_reference(root: Path, value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("RQ2 width128 boundary reference must be project-relative")
    path = root / relative
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
        raise ValueError(f"RQ2 width128 boundary reference is not a project file: {value}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ledger", type=Path)
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=PROJECT_ROOT / "generated/training-queue-service",
    )
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    print(submit_rq2_unexpected_width128_boundary_jobs(
        ledger_path=arguments.ledger,
        state_dir=arguments.state_dir,
        dry_run=arguments.dry_run,
    ))


if __name__ == "__main__":
    main()
