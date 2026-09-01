from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import subprocess

from experiments.g3_pretrained_item_embeddings.configs.model import (
    G3Representation,
    build_g3_experiment,
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
from experiments.g3_pretrained_item_embeddings.protocol.rq5_frequency_v2_embedding_boundary_ledger import (
    RQ5_FREQUENCY_V2_EMBEDDING_BOUNDARY_LEDGER_PATH,
    Rq5FrequencyV2EmbeddingBoundaryLedger,
    compile_rq5_frequency_v2_embedding_boundary_ledger,
    load_rq5_frequency_v2_embedding_boundary_ledger,
    verify_rq5_frequency_v2_embedding_boundary_inputs,
)


JOB_ENVIRONMENT = "G3_RQ5_FREQUENCY_V2_EMBEDDING_BOUNDARY_JOB_B64"
LEDGER_ENVIRONMENT = "G3_RQ5_FREQUENCY_V2_EMBEDDING_BOUNDARY_LEDGER_PATH"


def compile_queue_surface(
    *,
    ledger_path: Path,
    ledger: Rq5FrequencyV2EmbeddingBoundaryLedger,
    state_dir: Path,
    batch_id: str = "DRY_RUN_BATCH",
) -> list[list[str]]:
    expected = compile_rq5_frequency_v2_embedding_boundary_ledger(PROJECT_ROOT)
    if ledger != expected or len(ledger.rows) != 3:
        raise ValueError("RQ5 embedding boundary queue received another ledger")
    commands = compile_queue_commands(
        ledger_path=ledger_path,
        ledger=ledger,
        state_dir=state_dir,
        batch_id=batch_id,
        runner_script=Path(__file__).with_name(
            "run_rq5_frequency_v2_embedding_boundary.py"
        ),
        job_environment=JOB_ENVIRONMENT,
        ledger_environment=LEDGER_ENVIRONMENT,
        ledger_validator=lambda document: _validate_document(document, expected),
    )
    if len(commands) != 6:
        raise ValueError("RQ5 embedding boundary queue requires exactly three jobs")
    return commands


def submit_jobs(
    *,
    ledger_path: Path,
    state_dir: Path,
    expected_ledger_sha256: str,
    dry_run: bool,
) -> str:
    ledger_path = ledger_path.resolve(strict=True)
    ledger = load_rq5_frequency_v2_embedding_boundary_ledger(
        ledger_path,
        root=PROJECT_ROOT,
        expected_ledger_sha256=expected_ledger_sha256,
    )
    verify_rq5_frequency_v2_embedding_boundary_inputs(PROJECT_ROOT, ledger)
    if dry_run:
        return "\n".join(
            shlex.join(command)
            for command in compile_queue_surface(
                ledger_path=ledger_path,
                ledger=ledger,
                state_dir=state_dir,
            )
        )
    runner = Path(__file__).with_name("run_rq5_frequency_v2_embedding_boundary.py")
    with ledger_submission_lock(state_dir=state_dir, ledger_sha256=ledger.sha256):
        existing = find_existing_ledger_batch(
            state_dir=state_dir,
            ledger_path=ledger_path,
            ledger=ledger,
            runner_script=runner,
            job_environment=JOB_ENVIRONMENT,
            ledger_environment=LEDGER_ENVIRONMENT,
        )
        if existing is not None:
            return existing
        commands = compile_queue_surface(
            ledger_path=ledger_path,
            ledger=ledger,
            state_dir=state_dir,
        )
        subprocess.run(commands[0], cwd=PROJECT_ROOT, check=True, capture_output=True)
        created = subprocess.run(
            commands[1],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        batch_id = created.stdout.strip()
        if not batch_id:
            raise RuntimeError("training queue returned an empty batch id")
        commands = compile_queue_surface(
            ledger_path=ledger_path,
            ledger=ledger,
            state_dir=state_dir,
            batch_id=batch_id,
        )
        for command in commands[2:]:
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        return batch_id


def compiled_job_from_environment() -> tuple[
    CompiledControlJob, Rq5FrequencyV2EmbeddingBoundaryLedger, Path, Path
]:
    encoded = os.environ.get(JOB_ENVIRONMENT)
    raw_path = os.environ.get(LEDGER_ENVIRONMENT)
    if not encoded or not raw_path:
        raise RuntimeError(f"{JOB_ENVIRONMENT} and {LEDGER_ENVIRONMENT} are required")
    expected = compile_rq5_frequency_v2_embedding_boundary_ledger(PROJECT_ROOT)
    ledger_path = Path(raw_path).resolve(strict=True)
    ledger = load_rq5_frequency_v2_embedding_boundary_ledger(
        ledger_path,
        root=PROJECT_ROOT,
        expected_ledger_sha256=expected.sha256,
    )
    compiled = decode_control_job(encoded, ledger)
    if decode_control_job(encode_control_job(ledger, compiled.row_id), ledger) != compiled:
        raise ValueError("compiled RQ5 embedding boundary job differs from ledger")
    feature = verify_rq5_frequency_v2_embedding_boundary_inputs(PROJECT_ROOT, ledger)
    return compiled, ledger, ledger_path, feature


def build_training_experiment(
    compiled: CompiledControlJob,
    *,
    ledger: Rq5FrequencyV2EmbeddingBoundaryLedger,
    feature_data_path: Path,
):
    expected = decode_control_job(encode_control_job(ledger, compiled.row_id), ledger)
    if compiled != expected:
        raise ValueError("compiled RQ5 embedding boundary job differs from ledger")
    job = compiled.job
    representation = job.get("representation")
    training = job.get("training")
    if not isinstance(representation, dict) or not isinstance(training, dict):
        raise ValueError("compiled RQ5 embedding boundary job is invalid")
    if (
        job.get("family_id") != "rq5_frequency_gate_v2"
        or job.get("stage")
        != "rq5_frequency_gate_fp32_p09_v2_embedding_lr_upper_boundary"
        or representation.get("frequency_gate_semantics") != "fp32_p09_v2"
        or representation.get("gate_hidden_dim") != 8
        or training.get("deep_learning_rate") != 0.014506684820055783
        or training.get("horizon_epochs") != 40
        or training.get("batch_size") != 512
        or training.get("seed") != 42
    ):
        raise ValueError("RQ5 embedding boundary coordinate changed")
    return build_g3_experiment(
        run_name=str(job["run_name"]),
        dataset_size="native-50m",
        embedding_learning_rate=float(training["embedding_learning_rate"]),
        deep_learning_rate=0.014506684820055783,
        lr_schedule_horizon_epochs=40,
        seed=42,
        representation=G3Representation(
            history_representation="id_content",
            catalog_representation="learned_id",
            history_hidden_dim=128,
            content_gate="frequency",
            gate_hidden_dim=8,
            frequency_gate_semantics="fp32_p09_v2",
        ),
        feature_data_path=feature_data_path,
        gate_mechanism_diagnostics=True,
    )


def write_job_contract(
    compiled: CompiledControlJob,
    ledger: Rq5FrequencyV2EmbeddingBoundaryLedger,
    ledger_path: Path,
    logs_root: Path,
) -> Path:
    if decode_control_job(encode_control_job(ledger, compiled.row_id), ledger) != compiled:
        raise ValueError("compiled RQ5 embedding boundary job differs from ledger")
    return persist_job_contract(
        compiled=compiled,
        ledger_path=ledger_path,
        ledger_sha256=ledger.sha256,
        logs_root=logs_root,
        filename="g3_rq5_frequency_v2_embedding_boundary_job.json",
    )


def _validate_document(
    document: dict[str, object], ledger: Rq5FrequencyV2EmbeddingBoundaryLedger
) -> None:
    if document != ledger.to_dict():
        raise ValueError("RQ5 embedding boundary ledger document changed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ledger", type=Path)
    parser.add_argument("--expected-ledger-sha256", required=True)
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=PROJECT_ROOT / "generated/training-queue-service",
    )
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    print(
        submit_jobs(
            ledger_path=arguments.ledger,
            state_dir=arguments.state_dir,
            expected_ledger_sha256=arguments.expected_ledger_sha256,
            dry_run=arguments.dry_run,
        )
    )


if __name__ == "__main__":
    main()
