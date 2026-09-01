from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import subprocess

from experiments.g3_pretrained_item_embeddings.launchers.control import (
    PROJECT_ROOT,
    CompiledControlJob,
    compile_queue_commands,
    decode_control_job,
    encode_control_job,
    persist_job_contract,
    verify_ledger_inputs as verify_control_inputs,
)
from experiments.g3_pretrained_item_embeddings.protocol.control_ledger import (
    load_control_ledger,
)
from experiments.g3_pretrained_item_embeddings.protocol.rq2_capacity_ledger import (
    Rq2CapacityLedger,
    load_rq2_capacity_ledger,
    validate_rq2_capacity_ledger_document,
)


JOB_ENVIRONMENT = "G3_RQ2_CAPACITY_JOB_B64"
LEDGER_ENVIRONMENT = "G3_RQ2_CAPACITY_LEDGER_PATH"


def compile_rq2_capacity_queue_commands(
    *,
    ledger_path: Path,
    ledger: Rq2CapacityLedger,
    state_dir: Path,
    batch_id: str = "DRY_RUN_BATCH",
) -> list[list[str]]:
    return compile_queue_commands(
        ledger_path=ledger_path,
        ledger=ledger,
        state_dir=state_dir,
        batch_id=batch_id,
        runner_script=Path(__file__).with_name("run_rq2_capacity.py"),
        job_environment=JOB_ENVIRONMENT,
        ledger_environment=LEDGER_ENVIRONMENT,
        ledger_validator=validate_rq2_capacity_ledger_document,
    )


def submit_rq2_capacity_jobs(
    *,
    ledger_path: Path,
    state_dir: Path,
    dry_run: bool,
) -> str:
    ledger_path = ledger_path.resolve()
    ledger = load_rq2_capacity_ledger(ledger_path)
    verify_rq2_capacity_inputs(PROJECT_ROOT, ledger, full_validation=True)
    if dry_run:
        return "\n".join(
            shlex.join(command)
            for command in compile_rq2_capacity_queue_commands(
                ledger_path=ledger_path,
                ledger=ledger,
                state_dir=state_dir,
            )
        )
    initial = compile_rq2_capacity_queue_commands(
        ledger_path=ledger_path,
        ledger=ledger,
        state_dir=state_dir,
    )
    subprocess.run(
        initial[0], cwd=PROJECT_ROOT, check=True, capture_output=True
    )
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
    commands = compile_rq2_capacity_queue_commands(
        ledger_path=ledger_path,
        ledger=ledger,
        state_dir=state_dir,
        batch_id=batch_id,
    )
    for command in commands[2:]:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    return batch_id


def verify_rq2_capacity_inputs(
    root: Path,
    ledger: Rq2CapacityLedger,
    *,
    full_validation: bool,
) -> Path:
    root = root.resolve()
    control = load_control_ledger(
        _resolve_reference(root, ledger.untied_control_ledger.path)
    )
    if control.sha256 != ledger.untied_control_ledger.sha256:
        raise ValueError("RQ2 capacity untied-control ledger hash changed")
    direct_bindings = (
        (ledger.g4_control_manifest, control.g4_control, "G4 control"),
        (ledger.content_manifest, control.content, "content"),
        (ledger.feature_manifest, control.features, "feature"),
    )
    for direct, inherited, label in direct_bindings:
        if direct != inherited:
            raise ValueError(
                f"RQ2 capacity direct {label} binding differs from its control ledger"
            )
    feature_data_path = verify_control_inputs(
        root,
        control,
        full_validation=full_validation,
    )

    from experiments.g3_pretrained_item_embeddings.analysis.control_calibration import (
        load_control_calibration,
    )

    calibration = load_control_calibration(
        _resolve_reference(root, ledger.predecessor_calibration.path)
    )
    if calibration.get("sha256") != ledger.predecessor_calibration.sha256:
        raise ValueError("RQ2 capacity predecessor calibration hash changed")
    binding = calibration.get("control_ledger")
    if not isinstance(binding, dict):
        raise ValueError("RQ2 capacity predecessor does not bind its control ledger")
    if (
        binding.get("path") != ledger.untied_control_ledger.path
        or binding.get("logical_sha256") != ledger.untied_control_ledger.sha256
    ):
        raise ValueError(
            "RQ2 capacity predecessor binds a different untied control ledger"
        )
    return feature_data_path


def compiled_rq2_capacity_job_from_environment() -> tuple[
    CompiledControlJob, Path, Path
]:
    encoded = os.environ.get(JOB_ENVIRONMENT)
    raw_ledger_path = os.environ.get(LEDGER_ENVIRONMENT)
    if not encoded or not raw_ledger_path:
        raise RuntimeError(f"{JOB_ENVIRONMENT} and {LEDGER_ENVIRONMENT} are required")
    ledger_path = Path(raw_ledger_path)
    ledger = load_rq2_capacity_ledger(ledger_path)
    feature_data_path = verify_rq2_capacity_inputs(
        PROJECT_ROOT,
        ledger,
        full_validation=False,
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
    representation = job["representation"]
    training = job["training"]
    dataset = job["dataset"]
    if not all(
        isinstance(nested, dict)
        for nested in (representation, training, dataset)
    ):
        raise ValueError("compiled G3 RQ2 capacity job has invalid configuration")
    family_id = job.get("family_id")
    history_representation = {
        "rq2_content_concat": "id_content",
        "rq2_id_only_densenet": "id_only_densenet",
    }.get(family_id)
    if history_representation is None or representation.get("id") != family_id:
        raise ValueError("compiled G3 RQ2 capacity job has an invalid family")
    if job.get("phase") != "capacity_preselection":
        raise ValueError("compiled G3 RQ2 capacity job is not a preselection row")
    capacity = representation.get("history_hidden_dim")
    if not isinstance(capacity, int) or isinstance(capacity, bool):
        raise ValueError("compiled G3 RQ2 capacity job has an invalid width")
    return build_g3_experiment(
        run_name=str(job["run_name"]),
        dataset_size=dataset["size"],
        embedding_learning_rate=training["embedding_learning_rate"],
        deep_learning_rate=training["deep_learning_rate"],
        lr_schedule_horizon_epochs=training["horizon_epochs"],
        seed=training["seed"],
        representation=G3Representation(
            history_representation=history_representation,
            catalog_representation="learned_id",
            history_hidden_dim=capacity,
        ),
        feature_data_path=feature_data_path,
    )


def write_job_contract(
    compiled: CompiledControlJob,
    ledger_path: Path,
    logs_root: Path,
) -> Path:
    ledger = load_rq2_capacity_ledger(ledger_path)
    verified = decode_control_job(encode_control_job(ledger, compiled.row_id), ledger)
    if verified != compiled:
        raise ValueError(
            "compiled G3 RQ2 capacity job differs from its approved ledger row"
        )
    return persist_job_contract(
        compiled=compiled,
        ledger_path=ledger_path,
        ledger_sha256=ledger.sha256,
        logs_root=logs_root,
        filename="g3_rq2_capacity_job.json",
    )


def _resolve_reference(root: Path, value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("RQ2 capacity reference path must be project-relative")
    path = root / relative
    if (
        path.is_symlink()
        or not path.is_file()
        or not path.resolve().is_relative_to(root)
    ):
        raise ValueError(
            f"RQ2 capacity reference must be a regular project file: {value}"
        )
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
        submit_rq2_capacity_jobs(
            ledger_path=arguments.ledger,
            state_dir=arguments.state_dir,
            dry_run=arguments.dry_run,
        )
    )


if __name__ == "__main__":
    main()
