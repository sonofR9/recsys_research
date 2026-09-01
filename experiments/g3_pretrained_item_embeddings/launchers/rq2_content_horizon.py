from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import subprocess

from experiments.g3_pretrained_item_embeddings.analysis.control_calibration import (
    load_control_calibration,
)
from experiments.g3_pretrained_item_embeddings.analysis.rq2_id_boundary_results import (
    load_rq2_id_boundary_evidence,
    verify_rq2_id_boundary_evidence,
)
from experiments.g3_pretrained_item_embeddings.analysis.rq2_next_stage_results import (
    load_rq2_next_stage_evidence,
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
from experiments.g3_pretrained_item_embeddings.launchers.rq2_id_boundary import (
    verify_rq2_id_boundary_inputs,
)
from experiments.g3_pretrained_item_embeddings.protocol.rq2_content_horizon_ledger import (
    APPROVED_RQ2_CONTENT_HORIZON_LEDGER_SHA256,
    Rq2ContentHorizonLedger,
    load_rq2_content_horizon_ledger,
    validate_rq2_content_horizon_ledger_document,
)
from experiments.g3_pretrained_item_embeddings.protocol.rq2_id_boundary_ledger import (
    load_rq2_id_boundary_ledger,
)
from experiments.g3_pretrained_item_embeddings.protocol.rq2_next_stage_ledger import (
    load_rq2_next_stage_ledger,
)


JOB_ENVIRONMENT = "G3_RQ2_CONTENT_HORIZON_JOB_B64"
LEDGER_ENVIRONMENT = "G3_RQ2_CONTENT_HORIZON_LEDGER_PATH"


def compile_rq2_content_horizon_queue_commands(
    *,
    ledger_path: Path,
    ledger: Rq2ContentHorizonLedger,
    state_dir: Path,
    batch_id: str = "DRY_RUN_BATCH",
) -> list[list[str]]:
    if any(row.reused_from is not None for row in ledger.rows):
        raise ValueError("RQ2 content-horizon queue cannot enqueue reused cells")
    return compile_queue_commands(
        ledger_path=ledger_path,
        ledger=ledger,
        state_dir=state_dir,
        batch_id=batch_id,
        runner_script=Path(__file__).with_name("run_rq2_content_horizon.py"),
        job_environment=JOB_ENVIRONMENT,
        ledger_environment=LEDGER_ENVIRONMENT,
        ledger_validator=lambda document: validate_rq2_content_horizon_ledger_document(
            document, root=PROJECT_ROOT
        ),
    )


def submit_rq2_content_horizon_jobs(
    *, ledger_path: Path, state_dir: Path, dry_run: bool
) -> str:
    ledger_path = ledger_path.resolve()
    ledger = load_rq2_content_horizon_ledger(ledger_path, root=PROJECT_ROOT)
    verify_rq2_content_horizon_inputs(PROJECT_ROOT, ledger, full_validation=True)
    if dry_run:
        return "\n".join(
            shlex.join(command)
            for command in compile_rq2_content_horizon_queue_commands(
                ledger_path=ledger_path,
                ledger=ledger,
                state_dir=state_dir,
            )
        )
    with ledger_submission_lock(
        state_dir=state_dir,
        ledger_sha256=ledger.sha256,
    ):
        return _submit_rq2_content_horizon_jobs(
            ledger_path=ledger_path,
            ledger=ledger,
            state_dir=state_dir,
        )


def _submit_rq2_content_horizon_jobs(
    *, ledger_path: Path, ledger: Rq2ContentHorizonLedger, state_dir: Path
) -> str:
    existing = find_existing_ledger_batch(
        state_dir=state_dir,
        ledger_path=ledger_path,
        ledger=ledger,
        runner_script=Path(__file__).with_name("run_rq2_content_horizon.py"),
        job_environment=JOB_ENVIRONMENT,
        ledger_environment=LEDGER_ENVIRONMENT,
    )
    if existing is not None:
        return existing
    initial = compile_rq2_content_horizon_queue_commands(
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
    commands = compile_rq2_content_horizon_queue_commands(
        ledger_path=ledger_path,
        ledger=ledger,
        state_dir=state_dir,
        batch_id=batch_id,
    )
    for command in commands[2:]:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    return batch_id


def verify_rq2_content_horizon_inputs(
    root: Path,
    ledger: Rq2ContentHorizonLedger,
    *,
    full_validation: bool,
) -> Path:
    root = root.resolve(strict=True)
    boundary_evidence_path = _resolve_reference(root, ledger.id_boundary_evidence.path)
    boundary_evidence = (
        verify_rq2_id_boundary_evidence(boundary_evidence_path, root=root)
        if full_validation
        else load_rq2_id_boundary_evidence(boundary_evidence_path)
    )
    if boundary_evidence["sha256"] != ledger.id_boundary_evidence.sha256:
        raise ValueError("RQ2 content-horizon ID boundary evidence hash changed")
    boundary_ledger = load_rq2_id_boundary_ledger(
        _resolve_reference(root, ledger.id_boundary_ledger.path)
    )
    if boundary_ledger.sha256 != ledger.id_boundary_ledger.sha256:
        raise ValueError("RQ2 content-horizon ID boundary ledger hash changed")
    _verify_evidence_ledger_binding(
        boundary_evidence,
        field="id_boundary_ledger",
        path=ledger.id_boundary_ledger.path,
        sha256=ledger.id_boundary_ledger.sha256,
    )

    next_evidence_path = _resolve_reference(
        root, ledger.resolved_next_stage_evidence.path
    )
    next_evidence = load_rq2_next_stage_evidence(next_evidence_path)
    if next_evidence["sha256"] != ledger.resolved_next_stage_evidence.sha256:
        raise ValueError("RQ2 content-horizon next-stage evidence hash changed")
    next_ledger = load_rq2_next_stage_ledger(
        _resolve_reference(root, ledger.resolved_next_stage_ledger.path)
    )
    if next_ledger.sha256 != ledger.resolved_next_stage_ledger.sha256:
        raise ValueError("RQ2 content-horizon next-stage ledger hash changed")
    _verify_evidence_ledger_binding(
        next_evidence,
        field="next_stage_ledger",
        path=ledger.resolved_next_stage_ledger.path,
        sha256=ledger.resolved_next_stage_ledger.sha256,
    )
    predecessor = boundary_evidence.get("predecessor_evidence")
    if (
        not isinstance(predecessor, dict)
        or predecessor.get("path") != ledger.resolved_next_stage_evidence.path
        or predecessor.get("logical_sha256")
        != ledger.resolved_next_stage_evidence.sha256
        or boundary_ledger.next_stage_evidence.sha256
        != ledger.resolved_next_stage_evidence.sha256
        or boundary_ledger.next_stage_ledger.sha256
        != ledger.resolved_next_stage_ledger.sha256
    ):
        raise ValueError("RQ2 content-horizon predecessor chain changed")

    calibration = load_control_calibration(
        _resolve_reference(root, ledger.predecessor_calibration.path)
    )
    if (
        calibration["sha256"] != ledger.predecessor_calibration.sha256
        or next_ledger.predecessor_calibration.sha256
        != ledger.predecessor_calibration.sha256
    ):
        raise ValueError("RQ2 content-horizon calibration hash changed")
    feature_data_path = verify_rq2_id_boundary_inputs(
        root,
        boundary_ledger,
        full_validation=False,
    )
    preselection = next_ledger.preselection_ledger
    if (
        ledger.content.path
        != "experiments/g3_pretrained_item_embeddings/protocol/artifacts/"
        "native50m_content.json"
        or ledger.content.sha256
        != "5e24e5db5d3a5635433abd962b1de0753599618c2c0ab67edab6801b967ab070"
        or ledger.features.path
        != "experiments/g3_pretrained_item_embeddings/protocol/artifacts/"
        "native50m_features.json"
        or ledger.features.sha256
        != "02e919339094e5091e77d09bd77ea669b665c7f6f49a29b6f27d6708ee9cf021"
        or preselection.path
        != "experiments/g3_pretrained_item_embeddings/protocol/ledgers/"
        "rq2_capacity_preselection.json"
    ):
        raise ValueError("RQ2 content-horizon content ancestry changed")
    return feature_data_path


def _verify_evidence_ledger_binding(
    evidence: dict[str, object], *, field: str, path: str, sha256: str
) -> None:
    value = evidence.get(field)
    if (
        not isinstance(value, dict)
        or value.get("path") != path
        or value.get("logical_sha256") != sha256
    ):
        raise ValueError(f"RQ2 content-horizon evidence binds a different {field}")


def compiled_rq2_content_horizon_job_from_environment() -> (
    tuple[CompiledControlJob, Rq2ContentHorizonLedger, Path, Path]
):
    encoded = os.environ.get(JOB_ENVIRONMENT)
    raw_ledger_path = os.environ.get(LEDGER_ENVIRONMENT)
    if not encoded or not raw_ledger_path:
        raise RuntimeError(f"{JOB_ENVIRONMENT} and {LEDGER_ENVIRONMENT} are required")
    ledger_path = Path(raw_ledger_path)
    ledger = load_rq2_content_horizon_ledger(ledger_path, root=PROJECT_ROOT)
    feature_data_path = verify_rq2_content_horizon_inputs(
        PROJECT_ROOT,
        ledger,
        full_validation=False,
    )
    return decode_control_job(encoded, ledger), ledger, ledger_path, feature_data_path


def build_training_experiment(
    compiled: CompiledControlJob,
    *,
    ledger: Rq2ContentHorizonLedger,
    feature_data_path: Path,
):
    if ledger.sha256 != APPROVED_RQ2_CONTENT_HORIZON_LEDGER_SHA256:
        raise ValueError("RQ2 content-horizon ledger is not approved")
    expected = decode_control_job(
        encode_control_job(ledger, compiled.row_id),
        ledger,
    )
    if expected != compiled:
        raise ValueError(
            "compiled RQ2 content-horizon job differs from its approved immutable ledger row"
        )
    from experiments.g3_pretrained_item_embeddings.configs.model import (
        G3Representation,
        build_g3_experiment,
    )

    job = compiled.job
    representation = job.get("representation")
    training = job.get("training")
    dataset = job.get("dataset")
    if not all(
        isinstance(value, dict) for value in (representation, training, dataset)
    ):
        raise ValueError("compiled G3 RQ2 content-horizon job is invalid")
    if (
        job.get("family_id") != "rq2_content_concat"
        or job.get("phase") != "selected_width_horizon_followup"
        or representation.get("id") != "rq2_content_concat"
        or representation.get("history_hidden_dim") != 32
        or representation.get("content_width") != 128
        or training.get("batch_size") != 512
        or training.get("seed") != 42
    ):
        raise ValueError("compiled G3 RQ2 content-horizon coordinate is invalid")
    return build_g3_experiment(
        run_name=str(job["run_name"]),
        dataset_size=dataset["size"],
        embedding_learning_rate=training["embedding_learning_rate"],
        deep_learning_rate=training["deep_learning_rate"],
        lr_schedule_horizon_epochs=training["horizon_epochs"],
        seed=training["seed"],
        representation=G3Representation(
            history_representation="id_content",
            catalog_representation="learned_id",
            history_hidden_dim=32,
        ),
        feature_data_path=feature_data_path,
    )


def write_job_contract(
    compiled: CompiledControlJob, ledger_path: Path, logs_root: Path
) -> Path:
    ledger = load_rq2_content_horizon_ledger(ledger_path, root=PROJECT_ROOT)
    verified = decode_control_job(encode_control_job(ledger, compiled.row_id), ledger)
    if verified != compiled:
        raise ValueError("compiled RQ2 content-horizon job differs from its ledger row")
    return persist_job_contract(
        compiled=compiled,
        ledger_path=ledger_path,
        ledger_sha256=ledger.sha256,
        logs_root=logs_root,
        filename="g3_rq2_content_horizon_job.json",
    )


def _resolve_reference(root: Path, value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("RQ2 content-horizon reference must be project-relative")
    path = root / relative
    if (
        path.is_symlink()
        or not path.is_file()
        or not path.resolve().is_relative_to(root)
    ):
        raise ValueError(
            f"RQ2 content-horizon reference is not a project file: {value}"
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
        submit_rq2_content_horizon_jobs(
            ledger_path=arguments.ledger,
            state_dir=arguments.state_dir,
            dry_run=arguments.dry_run,
        )
    )


if __name__ == "__main__":
    main()
