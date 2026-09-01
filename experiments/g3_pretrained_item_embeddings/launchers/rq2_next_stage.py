from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import subprocess

from experiments.g3_pretrained_item_embeddings.analysis.control_calibration import (
    load_control_calibration,
)
from experiments.g3_pretrained_item_embeddings.analysis.rq2_capacity_results import (
    load_rq2_capacity_evidence,
    verify_rq2_capacity_evidence,
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
from experiments.g3_pretrained_item_embeddings.launchers.rq2_capacity import (
    verify_rq2_capacity_inputs,
)
from experiments.g3_pretrained_item_embeddings.protocol.rq2_capacity_ledger import (
    load_rq2_capacity_ledger,
)
from experiments.g3_pretrained_item_embeddings.protocol.rq2_next_stage_ledger import (
    Rq2NextStageLedger,
    load_rq2_next_stage_ledger,
    validate_rq2_next_stage_ledger_document,
)


JOB_ENVIRONMENT = "G3_RQ2_NEXT_STAGE_JOB_B64"
LEDGER_ENVIRONMENT = "G3_RQ2_NEXT_STAGE_LEDGER_PATH"


def compile_rq2_next_stage_queue_commands(
    *,
    ledger_path: Path,
    ledger: Rq2NextStageLedger,
    state_dir: Path,
    batch_id: str = "DRY_RUN_BATCH",
) -> list[list[str]]:
    if any(row.reused_from is not None for row in ledger.rows):
        raise ValueError("RQ2 next-stage queue cannot enqueue reused physical cells")
    return compile_queue_commands(
        ledger_path=ledger_path,
        ledger=ledger,
        state_dir=state_dir,
        batch_id=batch_id,
        runner_script=Path(__file__).with_name("run_rq2_next_stage.py"),
        job_environment=JOB_ENVIRONMENT,
        ledger_environment=LEDGER_ENVIRONMENT,
        ledger_validator=validate_rq2_next_stage_ledger_document,
    )


def submit_rq2_next_stage_jobs(
    *, ledger_path: Path, state_dir: Path, dry_run: bool
) -> str:
    ledger_path = ledger_path.resolve()
    ledger = load_rq2_next_stage_ledger(ledger_path)
    verify_rq2_next_stage_inputs(PROJECT_ROOT, ledger, full_validation=True)
    if dry_run:
        return "\n".join(
            shlex.join(command)
            for command in compile_rq2_next_stage_queue_commands(
                ledger_path=ledger_path,
                ledger=ledger,
                state_dir=state_dir,
            )
        )
    with ledger_submission_lock(
        state_dir=state_dir,
        ledger_sha256=ledger.sha256,
    ):
        return _submit_rq2_next_stage_jobs(
            ledger_path=ledger_path,
            ledger=ledger,
            state_dir=state_dir,
        )


def _submit_rq2_next_stage_jobs(
    *, ledger_path: Path, ledger: Rq2NextStageLedger, state_dir: Path
) -> str:
    existing = find_existing_ledger_batch(
        state_dir=state_dir,
        ledger_path=ledger_path,
        ledger=ledger,
        runner_script=Path(__file__).with_name("run_rq2_next_stage.py"),
        job_environment=JOB_ENVIRONMENT,
        ledger_environment=LEDGER_ENVIRONMENT,
    )
    if existing is not None:
        return existing
    initial = compile_rq2_next_stage_queue_commands(
        ledger_path=ledger_path,
        ledger=ledger,
        state_dir=state_dir,
    )
    subprocess.run(initial[0], cwd=PROJECT_ROOT, check=True, capture_output=True)
    created = subprocess.run(
        initial[1],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    batch_id = created.stdout.strip()
    if not batch_id:
        raise RuntimeError("training queue returned an empty batch id")
    commands = compile_rq2_next_stage_queue_commands(
        ledger_path=ledger_path,
        ledger=ledger,
        state_dir=state_dir,
        batch_id=batch_id,
    )
    for command in commands[2:]:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    return batch_id


def verify_rq2_next_stage_inputs(
    root: Path,
    ledger: Rq2NextStageLedger,
    *,
    full_validation: bool,
) -> Path:
    root = root.resolve()
    evidence_path = _resolve_reference(root, ledger.preselection_evidence.path)
    evidence = (
        verify_rq2_capacity_evidence(evidence_path, root=root)
        if full_validation
        else load_rq2_capacity_evidence(evidence_path)
    )
    if evidence["sha256"] != ledger.preselection_evidence.sha256:
        raise ValueError("RQ2 next-stage preselection evidence hash changed")
    preselection = load_rq2_capacity_ledger(
        _resolve_reference(root, ledger.preselection_ledger.path)
    )
    if preselection.sha256 != ledger.preselection_ledger.sha256:
        raise ValueError("RQ2 next-stage preselection ledger hash changed")
    evidence_ledger = evidence.get("rq2_capacity_ledger")
    if (
        not isinstance(evidence_ledger, dict)
        or evidence_ledger.get("path") != ledger.preselection_ledger.path
        or evidence_ledger.get("logical_sha256") != ledger.preselection_ledger.sha256
    ):
        raise ValueError("RQ2 next-stage evidence binds a different ledger")
    feature_data_path = verify_rq2_capacity_inputs(
        root,
        preselection,
        full_validation=full_validation,
    )
    calibration = load_control_calibration(
        _resolve_reference(root, ledger.predecessor_calibration.path)
    )
    if calibration["sha256"] != ledger.predecessor_calibration.sha256:
        raise ValueError("RQ2 next-stage calibration hash changed")
    evidence_calibration = evidence.get("predecessor_calibration")
    if (
        not isinstance(evidence_calibration, dict)
        or evidence_calibration.get("path") != ledger.predecessor_calibration.path
        or evidence_calibration.get("logical_sha256")
        != ledger.predecessor_calibration.sha256
    ):
        raise ValueError("RQ2 next-stage evidence binds a different calibration")
    return feature_data_path


def compiled_rq2_next_stage_job_from_environment() -> tuple[
    CompiledControlJob, Path, Path
]:
    encoded = os.environ.get(JOB_ENVIRONMENT)
    raw_ledger_path = os.environ.get(LEDGER_ENVIRONMENT)
    if not encoded or not raw_ledger_path:
        raise RuntimeError(f"{JOB_ENVIRONMENT} and {LEDGER_ENVIRONMENT} are required")
    ledger_path = Path(raw_ledger_path)
    ledger = load_rq2_next_stage_ledger(ledger_path)
    feature_data_path = verify_rq2_next_stage_inputs(
        PROJECT_ROOT, ledger, full_validation=False
    )
    return decode_control_job(encoded, ledger), ledger_path, feature_data_path


def build_training_experiment(
    compiled: CompiledControlJob,
    *,
    feature_data_path: Path,
):
    from experiments.g3_pretrained_item_embeddings.configs.model import (
        G3Representation,
        build_g3_experiment,
    )

    job = compiled.job
    representation = job.get("representation")
    training = job.get("training")
    dataset = job.get("dataset")
    if not all(isinstance(value, dict) for value in (representation, training, dataset)):
        raise ValueError("compiled G3 RQ2 next-stage job is invalid")
    family_id = job.get("family_id")
    history = {
        "rq2_content_concat": "id_content",
        "rq2_id_only_densenet": "id_only_densenet",
    }.get(family_id)
    if history is None or representation.get("id") != family_id:
        raise ValueError("compiled G3 RQ2 next-stage family is invalid")
    allowed_phases = {
        "rq2_content_concat": "capacity_boundary_extension",
        "rq2_id_only_densenet": "selected_capacity_horizon_followup",
    }
    if job.get("phase") != allowed_phases[family_id]:
        raise ValueError("compiled G3 RQ2 next-stage phase is invalid")
    capacity = representation.get("history_hidden_dim")
    if not isinstance(capacity, int) or isinstance(capacity, bool):
        raise ValueError("compiled G3 RQ2 next-stage capacity is invalid")
    return build_g3_experiment(
        run_name=str(job["run_name"]),
        dataset_size=dataset["size"],
        embedding_learning_rate=training["embedding_learning_rate"],
        deep_learning_rate=training["deep_learning_rate"],
        lr_schedule_horizon_epochs=training["horizon_epochs"],
        seed=training["seed"],
        representation=G3Representation(
            history_representation=history,
            catalog_representation="learned_id",
            history_hidden_dim=capacity,
        ),
        feature_data_path=feature_data_path,
    )


def write_job_contract(
    compiled: CompiledControlJob, ledger_path: Path, logs_root: Path
) -> Path:
    ledger = load_rq2_next_stage_ledger(ledger_path)
    verified = decode_control_job(encode_control_job(ledger, compiled.row_id), ledger)
    if verified != compiled:
        raise ValueError("compiled RQ2 next-stage job differs from its ledger row")
    return persist_job_contract(
        compiled=compiled,
        ledger_path=ledger_path,
        ledger_sha256=ledger.sha256,
        logs_root=logs_root,
        filename="g3_rq2_next_stage_job.json",
    )


def _resolve_reference(root: Path, value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("RQ2 next-stage reference must be project-relative")
    path = root / relative
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
        raise ValueError(f"RQ2 next-stage reference is not a project file: {value}")
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
    print(
        submit_rq2_next_stage_jobs(
            ledger_path=arguments.ledger,
            state_dir=arguments.state_dir,
            dry_run=arguments.dry_run,
        )
    )


if __name__ == "__main__":
    main()
