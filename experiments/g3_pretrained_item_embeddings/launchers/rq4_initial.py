from __future__ import annotations

import argparse
import base64
import json
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
from experiments.g3_pretrained_item_embeddings.protocol.rq4 import (
    compile_rq4_capacity_surface,
    resolve_rq4_feature_data,
)
from experiments.g3_pretrained_item_embeddings.protocol.rq4_initial_ledger import (
    Rq4InitialLedger,
    load_rq4_initial_ledger,
    validate_rq4_initial_ledger_document,
)


JOB_ENVIRONMENT = "G3_RQ4_INITIAL_JOB_B64"
LEDGER_ENVIRONMENT = "G3_RQ4_INITIAL_LEDGER_PATH"


def compile_rq4_initial_queue_commands(
    *,
    ledger_path: Path,
    ledger: Rq4InitialLedger,
    state_dir: Path,
    batch_id: str = "DRY_RUN_BATCH",
) -> list[list[str]]:
    if len(ledger.rows) != ledger.stage_physical_jobs == 27:
        raise ValueError("RQ4 initial queue requires exactly 27 physical jobs")
    return compile_queue_commands(
        ledger_path=ledger_path,
        ledger=ledger,
        state_dir=state_dir,
        batch_id=batch_id,
        runner_script=Path(__file__).with_name("run_rq4_initial.py"),
        job_environment=JOB_ENVIRONMENT,
        ledger_environment=LEDGER_ENVIRONMENT,
        ledger_validator=lambda document: validate_rq4_initial_ledger_document(
            document,
            root=PROJECT_ROOT,
            expected_ledger_sha256=ledger.sha256,
            expected_rq3_sha256=ledger.rq3_final_evidence.logical_sha256,
            expected_rq3_row_id=ledger.expected_rq3_row_id,
        ),
    )


def submit_rq4_initial_jobs(
    *,
    ledger_path: Path,
    state_dir: Path,
    expected_ledger_sha256: str,
    expected_rq3_sha256: str,
    expected_rq3_row_id: str,
    dry_run: bool,
) -> str:
    ledger_path = ledger_path.resolve(strict=True)
    ledger = load_rq4_initial_ledger(
        ledger_path,
        root=PROJECT_ROOT,
        expected_ledger_sha256=expected_ledger_sha256,
        expected_rq3_sha256=expected_rq3_sha256,
        expected_rq3_row_id=expected_rq3_row_id,
    )
    verify_rq4_initial_inputs(PROJECT_ROOT, ledger)
    if dry_run:
        return "\n".join(
            shlex.join(command)
            for command in compile_rq4_initial_queue_commands(
                ledger_path=ledger_path, ledger=ledger, state_dir=state_dir
            )
        )
    with ledger_submission_lock(state_dir=state_dir, ledger_sha256=ledger.sha256):
        return _submit(ledger_path=ledger_path, ledger=ledger, state_dir=state_dir)


def _submit(
    *, ledger_path: Path, ledger: Rq4InitialLedger, state_dir: Path
) -> str:
    runner = Path(__file__).with_name("run_rq4_initial.py")
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
    initial = compile_rq4_initial_queue_commands(
        ledger_path=ledger_path, ledger=ledger, state_dir=state_dir
    )
    subprocess.run(initial[0], cwd=PROJECT_ROOT, check=True, capture_output=True)
    created = subprocess.run(
        initial[1], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True
    )
    batch_id = created.stdout.strip()
    if not batch_id:
        raise RuntimeError("training queue returned an empty batch id")
    commands = compile_rq4_initial_queue_commands(
        ledger_path=ledger_path,
        ledger=ledger,
        state_dir=state_dir,
        batch_id=batch_id,
    )
    for command in commands[2:]:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    return batch_id


def verify_rq4_initial_inputs(root: Path, ledger: Rq4InitialLedger) -> Path:
    surface = compile_rq4_capacity_surface(
        root=root,
        rq2_selection_path=root / ledger.rq2_final_evidence.path,
        expected_rq2_selection_sha256=ledger.rq2_final_evidence.logical_sha256,
        rq3_selection_path=root / ledger.rq3_final_evidence.path,
        expected_rq3_selection_sha256=ledger.rq3_final_evidence.logical_sha256,
        expected_rq3_row_id=ledger.expected_rq3_row_id,
    )
    return resolve_rq4_feature_data(root=root, surface=surface)


def compiled_rq4_initial_job_from_environment() -> tuple[
    CompiledControlJob, Rq4InitialLedger, Path, Path
]:
    encoded = os.environ.get(JOB_ENVIRONMENT)
    raw_path = os.environ.get(LEDGER_ENVIRONMENT)
    if not encoded or not raw_path:
        raise RuntimeError(f"{JOB_ENVIRONMENT} and {LEDGER_ENVIRONMENT} are required")
    ledger_path = Path(raw_path)
    raw = _load_json(ledger_path)
    encoded_sha = _encoded_ledger_sha(encoded)
    inputs = raw.get("inputs")
    rq3 = inputs.get("rq3_final_evidence") if isinstance(inputs, dict) else None
    row_id = raw.get("expected_rq3_row_id")
    if not isinstance(rq3, dict) or not isinstance(row_id, str):
        raise ValueError("RQ4 worker ledger lacks frozen RQ3 identity")
    ledger = load_rq4_initial_ledger(
        ledger_path,
        root=PROJECT_ROOT,
        expected_ledger_sha256=encoded_sha,
        expected_rq3_sha256=str(rq3.get("logical_sha256")),
        expected_rq3_row_id=row_id,
    )
    feature_path = verify_rq4_initial_inputs(PROJECT_ROOT, ledger)
    return decode_control_job(encoded, ledger), ledger, ledger_path, feature_path


def build_training_experiment(
    compiled: CompiledControlJob,
    *,
    ledger: Rq4InitialLedger,
    feature_data_path: Path,
):
    expected = decode_control_job(encode_control_job(ledger, compiled.row_id), ledger)
    if expected != compiled:
        raise ValueError("compiled RQ4 job differs from immutable ledger")
    job = compiled.job
    representation = job.get("representation")
    training = job.get("training")
    if not isinstance(representation, dict) or not isinstance(training, dict):
        raise ValueError("compiled RQ4 job is invalid")
    if (
        job.get("family_id") not in ledger.family_opportunity_budgets
        or job.get("phase") != "rq4_metadata_capacity"
        or representation.get("metadata_dim") not in {16, 32, 64}
        or representation.get("metadata_pooling") != "mean"
        or representation.get("metadata_attachment")
        != "history_and_catalog_concat_then_separate_densenet"
        or training.get("batch_size") != 512
        or training.get("seed") != 42
    ):
        raise ValueError("compiled RQ4 coordinate violates the approved design")
    return build_g3_experiment(
        run_name=str(job["run_name"]),
        dataset_size="native-50m",
        embedding_learning_rate=float(training["embedding_learning_rate"]),
        deep_learning_rate=float(training["deep_learning_rate"]),
        lr_schedule_horizon_epochs=int(training["horizon_epochs"]),
        seed=42,
        representation=G3Representation(
            history_representation="id_content",
            history_hidden_dim=int(representation["history_hidden_dim"]),
            catalog_representation=str(representation["catalog"]),
            metadata=tuple(representation["metadata"]),
            metadata_dim=int(representation["metadata_dim"]),
        ),
        feature_data_path=feature_data_path,
    )


def write_job_contract(
    compiled: CompiledControlJob, ledger_path: Path, logs_root: Path
) -> Path:
    raw = _load_json(ledger_path)
    inputs = raw["inputs"]
    ledger = load_rq4_initial_ledger(
        ledger_path,
        root=PROJECT_ROOT,
        expected_ledger_sha256=compiled.ledger_sha256,
        expected_rq3_sha256=inputs["rq3_final_evidence"]["logical_sha256"],
        expected_rq3_row_id=raw["expected_rq3_row_id"],
    )
    if decode_control_job(encode_control_job(ledger, compiled.row_id), ledger) != compiled:
        raise ValueError("compiled RQ4 job differs from immutable ledger")
    return persist_job_contract(
        compiled=compiled,
        ledger_path=ledger_path,
        ledger_sha256=ledger.sha256,
        logs_root=logs_root,
        filename="g3_rq4_initial_job.json",
    )


def _encoded_ledger_sha(encoded: str) -> str:
    try:
        value = json.loads(base64.urlsafe_b64decode(encoded).decode())
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("compiled RQ4 job encoding is invalid") from error
    sha = value.get("ledger_sha256") if isinstance(value, dict) else None
    if not isinstance(sha, str) or len(sha) != 64:
        raise ValueError("compiled RQ4 job lacks ledger identity")
    return sha


def _load_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(
            path.read_text(),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load RQ4 ledger {path}") from error
    if not isinstance(value, dict):
        raise ValueError("RQ4 ledger must be an object")
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant: {value}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ledger", type=Path)
    parser.add_argument("--expected-ledger-sha256", required=True)
    parser.add_argument("--expected-rq3-sha256", required=True)
    parser.add_argument("--expected-rq3-row-id", required=True)
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=PROJECT_ROOT / "generated/training-queue-service",
    )
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    print(submit_rq4_initial_jobs(
        ledger_path=arguments.ledger,
        state_dir=arguments.state_dir,
        expected_ledger_sha256=arguments.expected_ledger_sha256,
        expected_rq3_sha256=arguments.expected_rq3_sha256,
        expected_rq3_row_id=arguments.expected_rq3_row_id,
        dry_run=arguments.dry_run,
    ))


if __name__ == "__main__":
    main()
