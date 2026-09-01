from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shlex
import subprocess
from typing import Callable

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
from experiments.g3_pretrained_item_embeddings.protocol.rq3_post_boundary import (
    POST_BOUNDARY_ADAPTER_KIND,
    RQ2_FINAL_EVIDENCE_LOGICAL_SHA256,
    RQ2_FINAL_EVIDENCE_PATH,
    RQ2_FINAL_SELECTED_ROW_ID,
    Rq3PostBoundaryLedger,
    compile_rq3_post_boundary_ledger,
    compile_verified_rq3_post_boundary_surface,
    load_rq3_post_boundary_ledger,
    persist_rq3_post_boundary_ledger,
    validate_rq3_post_boundary_ledger_document,
    verify_final_rq2_evidence_for_rq3,
)


JOB_ENVIRONMENT = "G3_RQ3_POST_BOUNDARY_JOB_B64"
LEDGER_ENVIRONMENT = "G3_RQ3_POST_BOUNDARY_LEDGER_PATH"
RQ3_POST_BOUNDARY_LEDGER_PATH = (
    "experiments/g3_pretrained_item_embeddings/protocol/ledgers/"
    "rq3_post_boundary_output_search.json"
)


def preview_rq3_post_boundary_ledger(
    *,
    root: Path = PROJECT_ROOT,
    final_evidence_path: Path = Path(RQ2_FINAL_EVIDENCE_PATH),
) -> Rq3PostBoundaryLedger:
    return compile_rq3_post_boundary_ledger(
        compile_verified_rq3_post_boundary_surface(
            root=root,
            final_evidence_path=final_evidence_path,
            expected_final_rq2_evidence_sha256=RQ2_FINAL_EVIDENCE_LOGICAL_SHA256,
            expected_selected_rq2_row_id=RQ2_FINAL_SELECTED_ROW_ID,
            adapter_kind=POST_BOUNDARY_ADAPTER_KIND,
            verifier=verify_final_rq2_evidence_for_rq3,
        )
    )


def materialize_rq3_post_boundary_ledger(
    path: Path | None = None,
    *,
    root: Path = PROJECT_ROOT,
    final_evidence_path: Path = Path(RQ2_FINAL_EVIDENCE_PATH),
) -> Path:
    ledger = preview_rq3_post_boundary_ledger(
        root=root,
        final_evidence_path=final_evidence_path,
    )
    destination = root / RQ3_POST_BOUNDARY_LEDGER_PATH if path is None else path
    return persist_rq3_post_boundary_ledger(destination, ledger)


def encode_rq3_post_boundary_job(
    ledger: Rq3PostBoundaryLedger,
    row_id: str,
) -> str:
    if any(row.id == row_id and row.reused_from is not None for row in ledger.logical_rows):
        raise ValueError("reused RQ2 rows are not launchable RQ3 jobs")
    return encode_control_job(ledger, row_id)


def decode_rq3_post_boundary_job(
    encoded: str,
    ledger: Rq3PostBoundaryLedger,
) -> CompiledControlJob:
    compiled = decode_control_job(encoded, ledger)
    if any(
        row.id == compiled.row_id and row.reused_from is not None
        for row in ledger.logical_rows
    ):
        raise ValueError("reused RQ2 rows are not launchable RQ3 jobs")
    return compiled


def compile_rq3_post_boundary_queue_commands(
    *,
    ledger_path: Path,
    ledger: Rq3PostBoundaryLedger,
    state_dir: Path,
    batch_id: str = "DRY_RUN_BATCH",
) -> list[list[str]]:
    if len(ledger.logical_rows) != 45 or len(ledger.rows) != 38:
        raise ValueError("RQ3 queue requires exactly 45 opportunities and 38 jobs")
    commands = compile_queue_commands(
        ledger_path=ledger_path,
        ledger=ledger,
        state_dir=state_dir,
        batch_id=batch_id,
        runner_script=Path(__file__).with_name("run_rq3_post_boundary.py"),
        job_environment=JOB_ENVIRONMENT,
        ledger_environment=LEDGER_ENVIRONMENT,
        ledger_validator=lambda document: validate_rq3_post_boundary_ledger_document(
            document,
            expected=ledger,
        ),
    )
    if len(commands) != 41:
        raise ValueError("RQ3 queue command count changed")
    return commands


def submit_rq3_post_boundary_jobs(
    *,
    ledger_path: Path,
    state_dir: Path,
    dry_run: bool,
    root: Path = PROJECT_ROOT,
    preview: Callable[[], Rq3PostBoundaryLedger] | None = None,
) -> str:
    root = root.resolve(strict=True)
    expected = (
        preview()
        if preview is not None
        else preview_rq3_post_boundary_ledger(root=root)
    )
    ledger_path = ledger_path.resolve(strict=True)
    ledger = load_rq3_post_boundary_ledger(ledger_path, expected=expected)
    resolve_rq3_post_boundary_feature_data(root, ledger)
    if dry_run:
        return "\n".join(
            shlex.join(command)
            for command in compile_rq3_post_boundary_queue_commands(
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
    ledger: Rq3PostBoundaryLedger,
    state_dir: Path,
    root: Path,
) -> str:
    runner_script = Path(__file__).with_name("run_rq3_post_boundary.py")
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
    initial = compile_rq3_post_boundary_queue_commands(
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
    commands = compile_rq3_post_boundary_queue_commands(
        ledger_path=ledger_path,
        ledger=ledger,
        state_dir=state_dir,
        batch_id=batch_id,
    )
    for command in commands[2:]:
        subprocess.run(command, cwd=root, check=True)
    return batch_id


def compiled_rq3_post_boundary_job_from_environment(
    *,
    root: Path = PROJECT_ROOT,
) -> tuple[
    CompiledControlJob,
    Rq3PostBoundaryLedger,
    Path,
    Path,
]:
    encoded = os.environ.get(JOB_ENVIRONMENT)
    raw_ledger_path = os.environ.get(LEDGER_ENVIRONMENT)
    if not encoded or not raw_ledger_path:
        raise RuntimeError(f"{JOB_ENVIRONMENT} and {LEDGER_ENVIRONMENT} are required")
    root = root.resolve(strict=True)
    ledger_path = Path(raw_ledger_path).resolve(strict=True)
    ledger = load_rq3_post_boundary_ledger(ledger_path)
    compiled = decode_rq3_post_boundary_job(encoded, ledger)
    feature_data_path = resolve_rq3_post_boundary_feature_data(root, ledger)
    return compiled, ledger, ledger_path, feature_data_path


def build_rq3_post_boundary_training_experiment(
    compiled: CompiledControlJob,
    *,
    ledger: Rq3PostBoundaryLedger,
    feature_data_path: Path,
):
    expected = decode_rq3_post_boundary_job(
        encode_rq3_post_boundary_job(ledger, compiled.row_id),
        ledger,
    )
    if compiled != expected:
        raise ValueError("compiled RQ3 job differs from its immutable ledger row")
    row = next(value for value in ledger.rows if value.id == compiled.row_id)
    representation = build_rq3_representation(
        row.family_id,
        history_hidden_dim=row.history_hidden_dim,
    )
    if representation.catalog_representation != row.catalog_representation:
        raise ValueError("RQ3 output family model mapping changed")
    return build_g3_experiment(
        run_name=row.run_name,
        dataset_size="native-50m",
        embedding_learning_rate=row.embedding_learning_rate,
        deep_learning_rate=row.deep_learning_rate,
        lr_schedule_horizon_epochs=row.horizon_epochs,
        seed=row.seed,
        representation=representation,
        feature_data_path=feature_data_path,
    )


def write_rq3_post_boundary_job_contract(
    compiled: CompiledControlJob,
    ledger: Rq3PostBoundaryLedger,
    ledger_path: Path,
    logs_root: Path,
) -> Path:
    ledger = load_rq3_post_boundary_ledger(ledger_path, expected=ledger)
    verified = decode_rq3_post_boundary_job(
        encode_rq3_post_boundary_job(ledger, compiled.row_id),
        ledger,
    )
    if verified != compiled:
        raise ValueError("compiled RQ3 job differs from its immutable ledger row")
    return persist_job_contract(
        compiled=compiled,
        ledger_path=ledger_path,
        ledger_sha256=ledger.sha256,
        logs_root=logs_root,
        filename="g3_rq3_output_job.json",
    )


def resolve_rq3_post_boundary_feature_data(
    root: Path,
    ledger: Rq3PostBoundaryLedger,
) -> Path:
    root = root.resolve(strict=True)
    relative = Path(ledger.feature.data_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("RQ3 feature data path is invalid")
    path = root / relative
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
        raise ValueError("RQ3 feature data is not a project file")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != ledger.feature.data_sha256:
        raise ValueError("RQ3 feature data changed before launch")
    return path.resolve()


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
        submit_rq3_post_boundary_jobs(
            ledger_path=arguments.ledger,
            state_dir=arguments.state_dir,
            dry_run=arguments.dry_run,
        )
    )


if __name__ == "__main__":
    main()
