from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import shlex
import subprocess
from typing import Any

from dcn.config import GenerationExperiment
from experiments.g4_future_items.configs.control import build_control
from experiments.g4_future_items.protocol.manifest import (
    CompiledJob,
    canonical_bytes,
    load_ledger,
    verify_ledger_semantics,
    verify_compiled_job,
)


JOB_ENVIRONMENT = "G4_COMPILED_JOB_B64"
LEDGER_ENVIRONMENT = "G4_LEDGER_PATH"
SEMANTICS_ENVIRONMENT = "G4_SEMANTICS_MANIFESTS_JSON"
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def encode_job(compiled: CompiledJob) -> str:
    return base64.urlsafe_b64encode(canonical_bytes(compiled.to_dict())).decode()


def _decode_document(encoded: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate compiled-job key {key!r}")
            result[key] = value
        return result

    try:
        decoded = base64.b64decode(encoded, altchars=b"-_", validate=True).decode(
            "utf-8"
        )
        value = json.loads(decoded, object_pairs_hook=reject_duplicates)
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("compiled G4 job payload is invalid") from error
    if not isinstance(value, dict) or set(value) != {
        "ledger_sha256",
        "row_id",
        "job",
    }:
        raise ValueError("compiled G4 job payload has missing or unknown fields")
    if not isinstance(value["job"], dict):
        raise ValueError("compiled G4 job payload has no resolved job")
    return value


def decode_job(encoded: str, ledger_path: Path) -> CompiledJob:
    document = _decode_document(encoded)
    compiled = CompiledJob(
        ledger_sha256=document["ledger_sha256"],
        row_id=document["row_id"],
        job=document["job"],
    )
    verify_compiled_job(compiled, ledger_path)
    return compiled


def compiled_job_from_environment() -> tuple[CompiledJob, Path]:
    encoded = os.environ.get(JOB_ENVIRONMENT)
    ledger = os.environ.get(LEDGER_ENVIRONMENT)
    semantics = os.environ.get(SEMANTICS_ENVIRONMENT)
    if not encoded or not ledger or not semantics:
        raise RuntimeError(
            f"{JOB_ENVIRONMENT}, {LEDGER_ENVIRONMENT}, and "
            f"{SEMANTICS_ENVIRONMENT} are required"
        )
    ledger_path = Path(ledger)
    try:
        raw_paths = json.loads(semantics)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"{SEMANTICS_ENVIRONMENT} is invalid") from error
    if not isinstance(raw_paths, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in raw_paths.items()
    ):
        raise RuntimeError(f"{SEMANTICS_ENVIRONMENT} is invalid")
    verify_ledger_semantics(
        load_ledger(ledger_path),
        {key: Path(value) for key, value in raw_paths.items()},
    )
    return decode_job(encoded, ledger_path), ledger_path


def build_training_experiment(compiled: CompiledJob) -> GenerationExperiment:
    stage = compiled.job["protocol"]["stage"]
    base_stage = stage.removesuffix("_boundary")
    common = {
        "run_name": compiled.job["run_name"],
        "batch_size": compiled.job["dataloader"]["batch_size"],
        "embedding_learning_rate": compiled.job["embedding_learning_rate"],
        "deep_learning_rate": compiled.job["deep_learning_rate"],
        "lr_schedule_horizon_epochs": compiled.job[
            "lr_schedule_horizon_epochs"
        ],
        "seed": compiled.job["seed"],
    }
    if base_stage == "control_tuning":
        return build_control(**common)
    if base_stage in {
        "rq1_tuning",
        "rq2_tuning",
        "rq3_deterministic_tuning",
        "rq3_learned_hard_tuning",
        "rq3_learned_proportional_tuning",
    }:
        from experiments.g4_future_items.configs.treatments import build_treatment

        return build_treatment(
            objective=compiled.job["objective"],
            valid_positive_mask_mode=compiled.job["loss"]["valid_positive_mask_mode"],
            **common,
        )
    raise ValueError(f"stage {stage!r} is not a G4 training stage")


def write_job_contract(
    compiled: CompiledJob, ledger_path: Path, logs_root: Path
) -> Path:
    verify_compiled_job(compiled, ledger_path)
    ledger = load_ledger(ledger_path)
    document = compiled.to_dict() | {
        "ledger_path": str(ledger_path.resolve()),
        "ledger_stage": ledger["stage"],
    }
    destination = logs_root / compiled.job["run_name"] / "g4_job.json"
    content = (
        json.dumps(
            document, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
        )
        + "\n"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.read_text() != content:
        raise RuntimeError(f"existing G4 contract differs: {destination}")
    destination.write_text(content)
    return destination


def emit_rows(path: Path, row_ids: list[str]) -> list[str]:
    rows = []
    ledger = load_ledger(path)
    approved_rows = {row["id"]: row for row in ledger["rows"]}
    for row_id in row_ids:
        row = approved_rows.get(row_id)
        if row is None:
            raise ValueError(f"ledger row {row_id!r} is not approved")
        compiled = CompiledJob.from_row(ledger_sha256=ledger["sha256"], row=row)
        rows.append(
            "\t".join(
                (
                    _run_name(compiled),
                    str(path.resolve()),
                    encode_job(compiled),
                )
            )
        )
    return rows


def _run_name(compiled: CompiledJob) -> str:
    run_name = compiled.job.get("run_name")
    if isinstance(run_name, str):
        return run_name
    suffix = compiled.row_id.replace(":", "_")
    return f"g4_{suffix}_native50m"


def _parse_semantics(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        key, separator, raw_path = value.partition("=")
        if not separator or not key or not raw_path or key in result:
            raise ValueError("--semantics values must be unique KEY=PATH pairs")
        result[key] = Path(raw_path).resolve()
    return result


def _selector_stage(stage: str) -> bool:
    return stage.startswith("selector_")


def _launcher_for_stage(stage: str) -> Path:
    if _selector_stage(stage):
        name = "run_selectors.py"
    elif stage.startswith("control_"):
        name = "run_control.py"
    else:
        name = "run_treatments.py"
    return Path(__file__).with_name(name).resolve()


def _queue_command(state_dir: Path, *arguments: str) -> list[str]:
    return [
        "python",
        str(PROJECT_ROOT / "utils/training_queue/service.py"),
        "--state-dir",
        str(state_dir),
        *arguments,
    ]


def submit_rows(
    *,
    ledger_path: Path,
    row_ids: list[str],
    semantics_paths: dict[str, Path],
    state_dir: Path,
    wait: bool,
    dry_run: bool,
) -> str:
    ledger_path = ledger_path.resolve()
    ledger = load_ledger(ledger_path)
    if ledger["stage"] == "selector_materialization":
        raise ValueError(
            "selector materialization must use run_selectors.py native-materialization"
        )
    verify_ledger_semantics(ledger, semantics_paths)
    approved_rows = {row["id"]: row for row in ledger["rows"]}
    compiled_jobs = []
    for row_id in row_ids:
        row = approved_rows.get(row_id)
        if row is None:
            raise ValueError(f"ledger row {row_id!r} is not approved")
        compiled_jobs.append(
            CompiledJob.from_row(ledger_sha256=ledger["sha256"], row=row)
        )
    if len({compiled.row_id for compiled in compiled_jobs}) != len(compiled_jobs):
        raise ValueError("submission contains duplicate ledger row ids")
    commands: list[list[str]] = []
    status_command = _queue_command(state_dir, "status", "--json")
    if dry_run:
        batch_id = "DRY_RUN_BATCH"
        commands.extend([status_command, _queue_command(state_dir, "new-batch")])
    else:
        subprocess.run(
            status_command, cwd=PROJECT_ROOT, check=True, capture_output=True
        )
        created = subprocess.run(
            _queue_command(state_dir, "new-batch"),
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        batch_id = created.stdout.strip()
        if not batch_id:
            raise RuntimeError("training queue returned an empty batch id")
    encoded_semantics = json.dumps(
        {key: str(path) for key, path in sorted(semantics_paths.items())},
        sort_keys=True,
        separators=(",", ":"),
    )
    for compiled in compiled_jobs:
        stage = (
            compiled.job["stage"]
            if _selector_stage(ledger["stage"])
            else compiled.job["protocol"]["stage"]
        )
        environment = [
            f"{JOB_ENVIRONMENT}={encode_job(compiled)}",
            f"{LEDGER_ENVIRONMENT}={ledger_path}",
            f"{SEMANTICS_ENVIRONMENT}={encoded_semantics}",
            "WANDB_MODE=offline",
        ]
        enqueue = _queue_command(
            state_dir,
            "enqueue-run",
            "--batch",
            batch_id,
            "--script",
            str(_launcher_for_stage(stage)),
            "--run",
            _run_name(compiled),
            "--data-group",
            (
                "g4-native50m-selector"
                if _selector_stage(stage)
                else "g4-native50m-likes"
            ),
            "--",
            *environment,
        )
        if dry_run:
            commands.append(enqueue)
        else:
            subprocess.run(enqueue, cwd=PROJECT_ROOT, check=True)
    seal = _queue_command(state_dir, "seal-batch", batch_id)
    if dry_run:
        commands.append(seal)
    else:
        subprocess.run(seal, cwd=PROJECT_ROOT, check=True)
    if wait:
        wait_command = _queue_command(state_dir, "wait-batch", batch_id)
        if dry_run:
            commands.append(wait_command)
        else:
            subprocess.run(wait_command, cwd=PROJECT_ROOT, check=True)
    return (
        "\n".join(shlex.join(command) for command in commands) if dry_run else batch_id
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    emit = subparsers.add_parser("emit")
    emit.add_argument("ledger", type=Path)
    emit.add_argument("row_ids", nargs="+")
    submit = subparsers.add_parser("submit")
    submit.add_argument("ledger", type=Path)
    submit.add_argument("row_ids", nargs="+")
    submit.add_argument("--semantics", action="append", default=[])
    submit.add_argument(
        "--state-dir",
        type=Path,
        default=PROJECT_ROOT / "generated/training-queue-service",
    )
    submit.add_argument("--no-wait", action="store_true")
    submit.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    if arguments.command == "emit":
        print("\n".join(emit_rows(arguments.ledger, arguments.row_ids)))
        return
    output = submit_rows(
        ledger_path=arguments.ledger,
        row_ids=arguments.row_ids,
        semantics_paths=_parse_semantics(arguments.semantics),
        state_dir=arguments.state_dir.resolve(),
        wait=not arguments.no_wait,
        dry_run=arguments.dry_run,
    )
    print(output)


if __name__ == "__main__":
    main()
