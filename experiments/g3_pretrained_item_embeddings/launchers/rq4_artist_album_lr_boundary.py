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
from experiments.g3_pretrained_item_embeddings.launchers.rq4_initial import (
    verify_rq4_initial_inputs,
)
from experiments.g3_pretrained_item_embeddings.protocol.rq4_artist_album_lr_boundary_ledger import (
    Rq4ArtistAlbumLrBoundaryLedger,
    compile_rq4_artist_album_lr_boundary_ledger,
    load_rq4_artist_album_lr_boundary_ledger,
    validate_rq4_artist_album_lr_boundary_ledger_document,
)
from experiments.g3_pretrained_item_embeddings.protocol.rq4_capacity_extension_ledger import (
    RQ4_CAPACITY_EXTENSION_LEDGER_PATH,
    compile_rq4_capacity_extension_ledger,
    load_rq4_capacity_extension_ledger,
)
from experiments.g3_pretrained_item_embeddings.protocol.rq4_initial_ledger import (
    RQ3_FINAL_EVIDENCE_LOGICAL_SHA256,
    RQ3_FINAL_SELECTED_ROW_ID,
    load_rq4_initial_ledger,
)


JOB_ENVIRONMENT = "G3_RQ4_ARTIST_ALBUM_LR_BOUNDARY_JOB_B64"
LEDGER_ENVIRONMENT = "G3_RQ4_ARTIST_ALBUM_LR_BOUNDARY_LEDGER_PATH"


def compile_queue_surface(
    *,
    ledger_path: Path,
    ledger: Rq4ArtistAlbumLrBoundaryLedger,
    state_dir: Path,
    batch_id: str = "DRY_RUN_BATCH",
) -> list[list[str]]:
    expected = compile_rq4_artist_album_lr_boundary_ledger(PROJECT_ROOT)
    if ledger != expected or len(ledger.rows) != 3:
        raise ValueError("RQ4 artist+album LR queue received another ledger")
    return compile_queue_commands(
        ledger_path=ledger_path,
        ledger=ledger,
        state_dir=state_dir,
        batch_id=batch_id,
        runner_script=Path(__file__).with_name(
            "run_rq4_artist_album_lr_boundary.py"
        ),
        job_environment=JOB_ENVIRONMENT,
        ledger_environment=LEDGER_ENVIRONMENT,
        ledger_validator=lambda document: (
            validate_rq4_artist_album_lr_boundary_ledger_document(
                document,
                root=PROJECT_ROOT,
                expected_ledger_sha256=expected.sha256,
            )
        ),
    )


def submit_jobs(
    *,
    ledger_path: Path,
    state_dir: Path,
    expected_ledger_sha256: str,
    dry_run: bool,
) -> str:
    ledger_path = ledger_path.resolve(strict=True)
    ledger = load_rq4_artist_album_lr_boundary_ledger(
        ledger_path,
        root=PROJECT_ROOT,
        expected_ledger_sha256=expected_ledger_sha256,
    )
    if dry_run:
        return "\n".join(
            shlex.join(command)
            for command in compile_queue_surface(
                ledger_path=ledger_path,
                ledger=ledger,
                state_dir=state_dir,
            )
        )
    runner = Path(__file__).with_name("run_rq4_artist_album_lr_boundary.py")
    with ledger_submission_lock(state_dir=state_dir, ledger_sha256=ledger.sha256):
        existing = find_existing_ledger_batch(
            state_dir=state_dir,
            ledger_path=ledger_path,
            ledger=ledger,
            runner_script=runner,
            job_environment=JOB_ENVIRONMENT,
            ledger_environment=LEDGER_ENVIRONMENT,
            ledger_path_sensitive=False,
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
    CompiledControlJob, Rq4ArtistAlbumLrBoundaryLedger, Path, Path
]:
    encoded = os.environ.get(JOB_ENVIRONMENT)
    raw_path = os.environ.get(LEDGER_ENVIRONMENT)
    if not encoded or not raw_path:
        raise RuntimeError(f"{JOB_ENVIRONMENT} and {LEDGER_ENVIRONMENT} are required")
    ledger_path = Path(raw_path)
    expected = compile_rq4_artist_album_lr_boundary_ledger(PROJECT_ROOT)
    ledger = load_rq4_artist_album_lr_boundary_ledger(
        ledger_path,
        root=PROJECT_ROOT,
        expected_ledger_sha256=expected.sha256,
    )
    compiled = decode_control_job(encoded, ledger)
    if decode_control_job(encode_control_job(ledger, compiled.row_id), ledger) != compiled:
        raise ValueError("compiled RQ4 LR-boundary job differs from immutable ledger")
    capacity_expected = compile_rq4_capacity_extension_ledger(PROJECT_ROOT)
    capacity = load_rq4_capacity_extension_ledger(
        PROJECT_ROOT / RQ4_CAPACITY_EXTENSION_LEDGER_PATH,
        root=PROJECT_ROOT,
        expected_ledger_sha256=capacity_expected.sha256,
    )
    initial = load_rq4_initial_ledger(
        PROJECT_ROOT / capacity.initial_ledger.path,
        root=PROJECT_ROOT,
        expected_ledger_sha256=capacity.initial_ledger.logical_sha256,
        expected_rq3_sha256=RQ3_FINAL_EVIDENCE_LOGICAL_SHA256,
        expected_rq3_row_id=RQ3_FINAL_SELECTED_ROW_ID,
    )
    return compiled, ledger, ledger_path, verify_rq4_initial_inputs(PROJECT_ROOT, initial)


def build_training_experiment(
    compiled: CompiledControlJob,
    *,
    ledger: Rq4ArtistAlbumLrBoundaryLedger,
    feature_data_path: Path,
):
    expected = decode_control_job(encode_control_job(ledger, compiled.row_id), ledger)
    if compiled != expected:
        raise ValueError("compiled RQ4 LR-boundary job differs from immutable ledger")
    job = compiled.job
    representation = job.get("representation")
    training = job.get("training")
    if not isinstance(representation, dict) or not isinstance(training, dict):
        raise ValueError("compiled RQ4 LR-boundary job is invalid")
    if (
        job.get("family_id") != "rq4_artist_album"
        or job.get("phase") != "rq4_metadata_joint_lr_boundary_extension"
        or representation.get("metadata") != ["artist", "album"]
        or representation.get("metadata_dim") != 64
        or training.get("horizon_epochs") != 25
        or training.get("batch_size") != 512
        or training.get("seed") != 42
    ):
        raise ValueError("RQ4 artist+album LR-boundary coordinate changed")
    return build_g3_experiment(
        run_name=str(job["run_name"]),
        dataset_size="native-50m",
        embedding_learning_rate=float(training["embedding_learning_rate"]),
        deep_learning_rate=float(training["deep_learning_rate"]),
        lr_schedule_horizon_epochs=25,
        seed=42,
        representation=G3Representation(
            history_representation="id_content",
            history_hidden_dim=128,
            catalog_representation="id_frozen_content",
            metadata=("artist", "album"),
            metadata_dim=64,
        ),
        feature_data_path=feature_data_path,
    )


def write_job_contract(
    compiled: CompiledControlJob,
    ledger: Rq4ArtistAlbumLrBoundaryLedger,
    ledger_path: Path,
    logs_root: Path,
) -> Path:
    if decode_control_job(encode_control_job(ledger, compiled.row_id), ledger) != compiled:
        raise ValueError("compiled RQ4 LR-boundary job differs from immutable ledger")
    return persist_job_contract(
        compiled=compiled,
        ledger_path=ledger_path,
        ledger_sha256=ledger.sha256,
        logs_root=logs_root,
        filename="g3_rq4_artist_album_lr_boundary_job.json",
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
        submit_jobs(
            ledger_path=arguments.ledger,
            state_dir=arguments.state_dir,
            expected_ledger_sha256=arguments.expected_ledger_sha256,
            dry_run=arguments.dry_run,
        )
    )


if __name__ == "__main__":
    main()
