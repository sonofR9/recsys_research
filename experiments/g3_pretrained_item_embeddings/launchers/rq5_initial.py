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
from experiments.g3_pretrained_item_embeddings.protocol.rq3_post_boundary import (
    POST_BOUNDARY_ADAPTER_KIND,
    RQ2_FINAL_EVIDENCE_LOGICAL_SHA256,
    RQ2_FINAL_EVIDENCE_PATH,
    RQ2_FINAL_SELECTED_ROW_ID,
)
from experiments.g3_pretrained_item_embeddings.protocol.rq5_initial import (
    RQ5_INITIAL_LEDGER_LOGICAL_SHA256,
    RQ5_INITIAL_LEDGER_PATH,
    Rq5InitialLedger,
    compile_rq5_initial_ledger,
    load_rq5_initial_ledger,
    persist_rq5_initial_ledger,
    validate_rq5_initial_ledger_document,
    verify_rq5_initial_input_files,
)


JOB_ENVIRONMENT = "G3_RQ5_INITIAL_JOB_B64"
LEDGER_ENVIRONMENT = "G3_RQ5_INITIAL_LEDGER_PATH"


def preview_rq5_initial_ledger(*, root: Path = PROJECT_ROOT) -> Rq5InitialLedger:
    ledger = compile_rq5_initial_ledger(
        root=root,
        final_rq2_evidence_path=Path(RQ2_FINAL_EVIDENCE_PATH),
        expected_final_rq2_sha256=RQ2_FINAL_EVIDENCE_LOGICAL_SHA256,
        expected_selected_rq2_row_id=RQ2_FINAL_SELECTED_ROW_ID,
        adapter_kind=POST_BOUNDARY_ADAPTER_KIND,
    )
    if ledger.sha256 != RQ5_INITIAL_LEDGER_LOGICAL_SHA256:
        raise ValueError("RQ5 canonical preview differs from the approved logical SHA")
    return ledger


def materialize_rq5_initial_ledger(
    path: Path | None = None, *, root: Path = PROJECT_ROOT
) -> Path:
    ledger = preview_rq5_initial_ledger(root=root)
    destination = root / RQ5_INITIAL_LEDGER_PATH if path is None else path
    return persist_rq5_initial_ledger(destination, ledger)


def compile_rq5_initial_queue_commands(
    *,
    ledger_path: Path,
    ledger: Rq5InitialLedger,
    state_dir: Path,
    batch_id: str = "DRY_RUN_BATCH",
) -> list[list[str]]:
    if len(ledger.rows) != ledger.stage_physical_jobs or len(ledger.rows) != 21:
        raise ValueError("RQ5 initial queue requires exactly 21 physical jobs")
    commands = compile_queue_commands(
        ledger_path=ledger_path,
        ledger=ledger,
        state_dir=state_dir,
        batch_id=batch_id,
        runner_script=Path(__file__).with_name("run_rq5_initial.py"),
        job_environment=JOB_ENVIRONMENT,
        ledger_environment=LEDGER_ENVIRONMENT,
        ledger_validator=lambda document: validate_rq5_initial_ledger_document(
            document, expected=ledger
        ),
    )
    if len(commands) != 24:
        raise ValueError("RQ5 initial queue command count changed")
    return commands


def submit_rq5_initial_jobs(
    *,
    ledger_path: Path,
    state_dir: Path,
    expected_ledger_sha256: str,
    dry_run: bool,
    root: Path = PROJECT_ROOT,
) -> str:
    root = root.resolve(strict=True)
    expected = preview_rq5_initial_ledger(root=root)
    if expected.sha256 != expected_ledger_sha256:
        raise ValueError("RQ5 external ledger SHA differs from verified preview")
    ledger_path = ledger_path.resolve(strict=True)
    ledger = load_rq5_initial_ledger(ledger_path, expected=expected)
    verify_rq5_initial_input_files(root, ledger)
    if dry_run:
        return "\n".join(
            shlex.join(command)
            for command in compile_rq5_initial_queue_commands(
                ledger_path=ledger_path,
                ledger=ledger,
                state_dir=state_dir,
            )
        )
    with ledger_submission_lock(state_dir=state_dir, ledger_sha256=ledger.sha256):
        return _submit(
            ledger_path=ledger_path,
            ledger=ledger,
            state_dir=state_dir,
            root=root,
        )


def _submit(
    *,
    ledger_path: Path,
    ledger: Rq5InitialLedger,
    state_dir: Path,
    root: Path,
) -> str:
    runner_script = Path(__file__).with_name("run_rq5_initial.py")
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
    initial = compile_rq5_initial_queue_commands(
        ledger_path=ledger_path,
        ledger=ledger,
        state_dir=state_dir,
    )
    subprocess.run(initial[0], cwd=root, check=True, capture_output=True)
    created = subprocess.run(
        initial[1], cwd=root, check=True, capture_output=True, text=True
    )
    batch_id = created.stdout.strip()
    if not batch_id:
        raise RuntimeError("training queue returned an empty batch id")
    commands = compile_rq5_initial_queue_commands(
        ledger_path=ledger_path,
        ledger=ledger,
        state_dir=state_dir,
        batch_id=batch_id,
    )
    for command in commands[2:]:
        subprocess.run(command, cwd=root, check=True)
    return batch_id


def compiled_rq5_initial_job_from_environment(
    *, root: Path = PROJECT_ROOT
) -> tuple[CompiledControlJob, Rq5InitialLedger, Path, Path]:
    encoded = os.environ.get(JOB_ENVIRONMENT)
    raw_ledger_path = os.environ.get(LEDGER_ENVIRONMENT)
    if not encoded or not raw_ledger_path:
        raise RuntimeError(f"{JOB_ENVIRONMENT} and {LEDGER_ENVIRONMENT} are required")
    root = root.resolve(strict=True)
    ledger_path = Path(raw_ledger_path).resolve(strict=True)
    ledger = load_rq5_initial_ledger(ledger_path)
    if ledger.sha256 != RQ5_INITIAL_LEDGER_LOGICAL_SHA256:
        raise ValueError("RQ5 worker ledger differs from the approved logical SHA")
    compiled = decode_control_job(encoded, ledger)
    feature_path = verify_rq5_initial_input_files(root, ledger)
    return compiled, ledger, ledger_path, feature_path


def build_rq5_initial_training_experiment(
    compiled: CompiledControlJob,
    *,
    ledger: Rq5InitialLedger,
    feature_data_path: Path,
):
    expected = decode_control_job(
        encode_control_job(ledger, compiled.row_id), ledger
    )
    if expected != compiled:
        raise ValueError("compiled RQ5 job differs from its immutable ledger row")
    row = next((value for value in ledger.rows if value.id == compiled.row_id), None)
    if row is None or row.reused_from is not None:
        raise ValueError("RQ5 fixed-gate evidence is not a launchable job")
    if (
        row.content_gate not in {"global", "frequency"}
        or (row.content_gate == "global" and row.gate_hidden_dim is not None)
        or (row.content_gate == "frequency" and row.gate_hidden_dim not in {4, 8, 16})
    ):
        raise ValueError("RQ5 gate row violates the approved model mapping")
    return build_g3_experiment(
        run_name=row.run_name,
        dataset_size="native-50m",
        embedding_learning_rate=row.embedding_learning_rate,
        deep_learning_rate=row.deep_learning_rate,
        lr_schedule_horizon_epochs=row.horizon_epochs,
        seed=row.seed,
        representation=G3Representation(
            history_representation="id_content",
            catalog_representation="learned_id",
            history_hidden_dim=row.history_hidden_dim,
            content_gate=row.content_gate,
            gate_hidden_dim=row.gate_hidden_dim,
        ),
        feature_data_path=feature_data_path,
    )


def write_rq5_initial_job_contract(
    compiled: CompiledControlJob,
    ledger: Rq5InitialLedger,
    ledger_path: Path,
    logs_root: Path,
) -> Path:
    loaded = load_rq5_initial_ledger(ledger_path, expected=ledger)
    verified = decode_control_job(
        encode_control_job(loaded, compiled.row_id), loaded
    )
    if verified != compiled:
        raise ValueError("compiled RQ5 job differs from its immutable ledger row")
    return persist_job_contract(
        compiled=compiled,
        ledger_path=ledger_path,
        ledger_sha256=loaded.sha256,
        logs_root=logs_root,
        filename="g3_rq5_gate_job.json",
    )


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
        submit_rq5_initial_jobs(
            ledger_path=arguments.ledger,
            state_dir=arguments.state_dir,
            expected_ledger_sha256=arguments.expected_ledger_sha256,
            dry_run=arguments.dry_run,
        )
    )


if __name__ == "__main__":
    main()
