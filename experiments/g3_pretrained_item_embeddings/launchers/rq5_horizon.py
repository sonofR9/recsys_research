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
from experiments.g3_pretrained_item_embeddings.protocol.rq5_horizon_ledger import (
    RQ5_HORIZON_LEDGER_PATH,
    RQ5_INITIAL_EVIDENCE_LOGICAL_SHA256,
    RQ5_INITIAL_EVIDENCE_PATH,
    RQ5_INITIAL_LEDGER_PATH,
    Rq5HorizonLedger,
    compile_rq5_horizon_ledger,
    load_rq5_horizon_ledger,
    persist_rq5_horizon_ledger,
    verify_rq5_horizon_input_files,
)


RQ5_HORIZON_LEDGER_LOGICAL_SHA256 = (
    "32dd7b3f3459adbfb93d4ed190c7f657a4ad21ee399da76d81e75c4a3453c68d"
)
JOB_ENVIRONMENT = "G3_RQ5_HORIZON_JOB_B64"
LEDGER_ENVIRONMENT = "G3_RQ5_HORIZON_LEDGER_PATH"


def preview_rq5_horizon_ledger(*, root: Path = PROJECT_ROOT) -> Rq5HorizonLedger:
    ledger = compile_rq5_horizon_ledger(
        root=root,
        initial_ledger_path=root / RQ5_INITIAL_LEDGER_PATH,
        initial_collection_path=root / RQ5_INITIAL_EVIDENCE_PATH,
        expected_initial_collection_sha256=RQ5_INITIAL_EVIDENCE_LOGICAL_SHA256,
    )
    if ledger.sha256 != RQ5_HORIZON_LEDGER_LOGICAL_SHA256:
        raise ValueError("RQ5 horizon preview differs from the approved logical SHA")
    return ledger


def materialize_rq5_horizon_ledger(
    path: Path | None = None, *, root: Path = PROJECT_ROOT
) -> Path:
    ledger = preview_rq5_horizon_ledger(root=root)
    destination = root / RQ5_HORIZON_LEDGER_PATH if path is None else path
    return persist_rq5_horizon_ledger(destination, ledger, root=root)


def compile_rq5_horizon_queue_commands(
    *,
    ledger_path: Path,
    ledger: Rq5HorizonLedger,
    state_dir: Path,
    batch_id: str = "DRY_RUN_BATCH",
) -> list[list[str]]:
    if len(ledger.rows) != 3 or len(ledger.physical_rows) != 3:
        raise ValueError("RQ5 horizon queue requires exactly three physical jobs")
    commands = compile_queue_commands(
        ledger_path=ledger_path,
        ledger=ledger,
        state_dir=state_dir,
        batch_id=batch_id,
        runner_script=Path(__file__).with_name("run_rq5_horizon.py"),
        job_environment=JOB_ENVIRONMENT,
        ledger_environment=LEDGER_ENVIRONMENT,
        ledger_validator=lambda document: _validate_document(document, ledger),
    )
    if len(commands) != 6:
        raise ValueError("RQ5 horizon queue command count changed")
    return commands


def submit_rq5_horizon_jobs(
    *,
    ledger_path: Path,
    state_dir: Path,
    expected_ledger_sha256: str,
    dry_run: bool,
    root: Path = PROJECT_ROOT,
) -> str:
    root = root.resolve(strict=True)
    expected = preview_rq5_horizon_ledger(root=root)
    if expected.sha256 != expected_ledger_sha256:
        raise ValueError("RQ5 external horizon SHA differs from verified preview")
    ledger_path = ledger_path.resolve(strict=True)
    ledger = load_rq5_horizon_ledger(
        ledger_path, root=root, expected_ledger_sha256=expected.sha256
    )
    verify_rq5_horizon_input_files(root, ledger)
    if dry_run:
        return "\n".join(
            shlex.join(command)
            for command in compile_rq5_horizon_queue_commands(
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
    ledger: Rq5HorizonLedger,
    state_dir: Path,
    root: Path,
) -> str:
    runner_script = Path(__file__).with_name("run_rq5_horizon.py")
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
    initial = compile_rq5_horizon_queue_commands(
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
    commands = compile_rq5_horizon_queue_commands(
        ledger_path=ledger_path,
        ledger=ledger,
        state_dir=state_dir,
        batch_id=batch_id,
    )
    for command in commands[2:]:
        subprocess.run(command, cwd=root, check=True)
    return batch_id


def compiled_rq5_horizon_job_from_environment(
    *, root: Path = PROJECT_ROOT
) -> tuple[CompiledControlJob, Rq5HorizonLedger, Path, Path]:
    encoded = os.environ.get(JOB_ENVIRONMENT)
    raw_ledger_path = os.environ.get(LEDGER_ENVIRONMENT)
    if not encoded or not raw_ledger_path:
        raise RuntimeError(f"{JOB_ENVIRONMENT} and {LEDGER_ENVIRONMENT} are required")
    root = root.resolve(strict=True)
    ledger_path = Path(raw_ledger_path).resolve(strict=True)
    ledger = load_rq5_horizon_ledger(
        ledger_path,
        root=root,
        expected_ledger_sha256=RQ5_HORIZON_LEDGER_LOGICAL_SHA256,
    )
    compiled = decode_control_job(encoded, ledger)
    feature_path = verify_rq5_horizon_input_files(root, ledger)
    return compiled, ledger, ledger_path, feature_path


def build_rq5_horizon_training_experiment(
    compiled: CompiledControlJob,
    *,
    ledger: Rq5HorizonLedger,
    feature_data_path: Path,
):
    expected = decode_control_job(encode_control_job(ledger, compiled.row_id), ledger)
    if expected != compiled:
        raise ValueError("compiled RQ5 horizon job differs from its ledger row")
    row = next((value for value in ledger.rows if value.id == compiled.row_id), None)
    if row is None or row.reused_from is not None:
        raise ValueError("RQ5 horizon row is not a launchable physical job")
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
            content_gate="frequency",
            gate_hidden_dim=row.gate_hidden_dim,
        ),
        feature_data_path=feature_data_path,
    )


def write_rq5_horizon_job_contract(
    compiled: CompiledControlJob,
    ledger: Rq5HorizonLedger,
    ledger_path: Path,
    logs_root: Path,
) -> Path:
    loaded = load_rq5_horizon_ledger(
        ledger_path,
        root=PROJECT_ROOT,
        expected_ledger_sha256=ledger.sha256,
    )
    verified = decode_control_job(encode_control_job(loaded, compiled.row_id), loaded)
    if verified != compiled:
        raise ValueError("compiled RQ5 horizon job differs from its ledger row")
    return persist_job_contract(
        compiled=compiled,
        ledger_path=ledger_path,
        ledger_sha256=loaded.sha256,
        logs_root=logs_root,
        filename="g3_rq5_horizon_job.json",
    )


def _validate_document(document: dict[str, object], ledger: Rq5HorizonLedger) -> None:
    if document != ledger.to_dict():
        raise ValueError("RQ5 horizon ledger document differs from the approved ledger")


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
        submit_rq5_horizon_jobs(
            ledger_path=arguments.ledger,
            state_dir=arguments.state_dir,
            expected_ledger_sha256=arguments.expected_ledger_sha256,
            dry_run=arguments.dry_run,
        )
    )


if __name__ == "__main__":
    main()
