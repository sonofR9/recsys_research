from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import subprocess

from experiments.g3_pretrained_item_embeddings.configs.model import (
    build_g3_experiment,
    build_rq3_representation,
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
from experiments.g3_pretrained_item_embeddings.launchers.rq3_post_boundary import (
    resolve_rq3_post_boundary_feature_data,
)
from experiments.g3_pretrained_item_embeddings.protocol.rq3_boundary_ledger import (
    RQ3_BOUNDARY_LEDGER_PATH,
    RQ3_BOUNDARY_DEEP_LRS,
    Rq3BoundaryLedger,
    compile_rq3_boundary_ledger,
    load_rq3_boundary_ledger,
    persist_rq3_boundary_ledger,
    validate_rq3_boundary_ledger_document,
)
from experiments.g3_pretrained_item_embeddings.protocol.rq3_post_boundary import (
    load_rq3_post_boundary_ledger,
)


JOB_ENVIRONMENT = "G3_RQ3_BOUNDARY_JOB_B64"
LEDGER_ENVIRONMENT = "G3_RQ3_BOUNDARY_LEDGER_PATH"


def materialize_rq3_boundary_ledger(
    *,
    root: Path = PROJECT_ROOT,
    path: Path | None = None,
) -> Path:
    root = root.resolve(strict=True)
    ledger = compile_rq3_boundary_ledger(root, full_validation=True)
    destination = root / RQ3_BOUNDARY_LEDGER_PATH if path is None else path
    return persist_rq3_boundary_ledger(destination, ledger)


def compile_rq3_boundary_queue_commands(
    *,
    ledger_path: Path,
    ledger: Rq3BoundaryLedger,
    state_dir: Path,
    batch_id: str = "DRY_RUN_BATCH",
) -> list[list[str]]:
    if len(ledger.rows) != 6:
        raise ValueError("RQ3 boundary queue requires exactly six physical jobs")
    commands = compile_queue_commands(
        ledger_path=ledger_path,
        ledger=ledger,
        state_dir=state_dir,
        batch_id=batch_id,
        runner_script=Path(__file__).with_name("run_rq3_boundary.py"),
        job_environment=JOB_ENVIRONMENT,
        ledger_environment=LEDGER_ENVIRONMENT,
        ledger_validator=lambda document: validate_rq3_boundary_ledger_document(
            document,
            root=PROJECT_ROOT,
            full_validation=False,
        ),
    )
    if len(commands) != 9:
        raise ValueError("RQ3 boundary queue command count changed")
    return commands


def submit_rq3_boundary_jobs(
    *,
    ledger_path: Path,
    state_dir: Path,
    dry_run: bool,
    root: Path = PROJECT_ROOT,
) -> str:
    root = root.resolve(strict=True)
    ledger_path = ledger_path.resolve(strict=True)
    expected = compile_rq3_boundary_ledger(root, full_validation=True)
    ledger = load_rq3_boundary_ledger(
        ledger_path,
        root=root,
        full_validation=False,
    )
    if ledger != expected:
        raise ValueError("RQ3 boundary ledger differs from authenticated evidence")
    resolve_rq3_boundary_feature_data(root, ledger)
    if dry_run:
        return "\n".join(
            shlex.join(command)
            for command in compile_rq3_boundary_queue_commands(
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
    ledger: Rq3BoundaryLedger,
    state_dir: Path,
    root: Path,
) -> str:
    runner_script = Path(__file__).with_name("run_rq3_boundary.py")
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
    initial = compile_rq3_boundary_queue_commands(
        ledger_path=ledger_path,
        ledger=ledger,
        state_dir=state_dir,
    )
    subprocess.run(initial[0], cwd=root, check=True, capture_output=True)
    created = subprocess.run(
        initial[1],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    batch_id = created.stdout.strip()
    if not batch_id:
        raise RuntimeError("training queue returned an empty batch id")
    commands = compile_rq3_boundary_queue_commands(
        ledger_path=ledger_path,
        ledger=ledger,
        state_dir=state_dir,
        batch_id=batch_id,
    )
    for command in commands[2:]:
        subprocess.run(command, cwd=root, check=True)
    return batch_id


def compiled_rq3_boundary_job_from_environment(
    *,
    root: Path = PROJECT_ROOT,
) -> tuple[CompiledControlJob, Rq3BoundaryLedger, Path, Path]:
    encoded = os.environ.get(JOB_ENVIRONMENT)
    raw_ledger_path = os.environ.get(LEDGER_ENVIRONMENT)
    if not encoded or not raw_ledger_path:
        raise RuntimeError(f"{JOB_ENVIRONMENT} and {LEDGER_ENVIRONMENT} are required")
    root = root.resolve(strict=True)
    ledger_path = Path(raw_ledger_path).resolve(strict=True)
    ledger = load_rq3_boundary_ledger(
        ledger_path,
        root=root,
        full_validation=False,
    )
    compiled = decode_control_job(encoded, ledger)
    feature_data_path = resolve_rq3_boundary_feature_data(root, ledger)
    return compiled, ledger, ledger_path, feature_data_path


def resolve_rq3_boundary_feature_data(
    root: Path,
    ledger: Rq3BoundaryLedger,
) -> Path:
    source_path = (root / ledger.source_ledger.path).resolve(strict=True)
    source = load_rq3_post_boundary_ledger(source_path)
    if source.sha256 != ledger.source_ledger.logical_sha256:
        raise ValueError("RQ3 boundary source ledger changed")
    return resolve_rq3_post_boundary_feature_data(root, source)


def build_rq3_boundary_training_experiment(
    compiled: CompiledControlJob,
    *,
    ledger: Rq3BoundaryLedger,
    feature_data_path: Path,
):
    expected = decode_control_job(encode_control_job(ledger, compiled.row_id), ledger)
    if compiled != expected:
        raise ValueError("compiled RQ3 boundary job differs from its ledger row")
    row = next(value for value in ledger.rows if value.id == compiled.row_id)
    if row.deep_learning_rate not in RQ3_BOUNDARY_DEEP_LRS:
        raise ValueError("RQ3 boundary deep learning rate changed")
    representation = build_rq3_representation(
        row.family_id,
        history_hidden_dim=128,
    )
    if representation.catalog_representation != row.catalog_representation:
        raise ValueError("RQ3 boundary output family mapping changed")
    return build_g3_experiment(
        run_name=row.run_name,
        dataset_size="native-50m",
        embedding_learning_rate=0.3041556165944196,
        deep_learning_rate=row.deep_learning_rate,
        lr_schedule_horizon_epochs=40,
        seed=42,
        representation=representation,
        feature_data_path=feature_data_path,
    )


def write_rq3_boundary_job_contract(
    compiled: CompiledControlJob,
    ledger: Rq3BoundaryLedger,
    ledger_path: Path,
    logs_root: Path,
) -> Path:
    verified = decode_control_job(encode_control_job(ledger, compiled.row_id), ledger)
    if verified != compiled:
        raise ValueError("compiled RQ3 boundary job differs from ledger")
    return persist_job_contract(
        compiled=compiled,
        ledger_path=ledger_path,
        ledger_sha256=ledger.sha256,
        logs_root=logs_root,
        filename="g3_rq3_boundary_job.json",
    )


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
    print(
        submit_rq3_boundary_jobs(
            ledger_path=arguments.ledger,
            state_dir=arguments.state_dir,
            dry_run=arguments.dry_run,
        )
    )


if __name__ == "__main__":
    main()
