from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

from experiments.g3_pretrained_item_embeddings.analysis.control_calibration import (
    _file_fact,
    _load_json,
)
from experiments.g3_pretrained_item_embeddings.analysis.rq1_results import (
    load_training_item_counts,
)
from experiments.g3_pretrained_item_embeddings.analysis.rq3_post_boundary_results import (
    RQ3_INITIAL_EVIDENCE_PATH,
    _canonical_json,
    _collect_artifacts,
    _feature_identity,
    _mechanism_assessment,
    build_rq3_paired_contrasts,
    load_rq3_post_boundary_evidence,
    resolve_rq3_downstream_selection,
    select_rq3_family_winners,
)
from experiments.g3_pretrained_item_embeddings.launchers.control import PROJECT_ROOT
from experiments.g3_pretrained_item_embeddings.launchers.rq3_boundary import (
    JOB_ENVIRONMENT,
    LEDGER_ENVIRONMENT,
    resolve_rq3_boundary_feature_data,
)
from experiments.g3_pretrained_item_embeddings.protocol.constants import (
    APPROVED_PROTOCOL_SHA256,
)
from experiments.g3_pretrained_item_embeddings.protocol.rq3 import (
    RQ3_OUTPUT_FAMILY_IDS,
)
from experiments.g3_pretrained_item_embeddings.protocol.rq3_boundary_ledger import (
    RQ3_BOUNDARY_DEEP_LRS,
    RQ3_BOUNDARY_FAMILY_IDS,
    RQ3_BOUNDARY_LEDGER_PATH,
    RQ3_INITIAL_EVIDENCE_LOGICAL_SHA256,
    load_rq3_boundary_ledger,
)
from experiments.g3_pretrained_item_embeddings.protocol.rq3_post_boundary import (
    load_rq3_post_boundary_ledger,
)
from experiments.g3_pretrained_item_embeddings.launchers.control import (
    decode_control_job,
)
from experiments.g3_pretrained_item_embeddings.analysis.queue_attribution import (
    verify_artifacts_in_job_window,
    verify_unique_completed_run,
)


RQ3_BOUNDARY_BATCH_ID = "fc6977eb1afc498580492afce72cce86"
RQ3_FINAL_EVIDENCE_PATH = (
    "experiments/g3_pretrained_item_embeddings/evidence/rq3_final_native50m.json"
)
_ARTIFACT_FILENAMES = {
    "job_contract": "g3_rq3_boundary_job.json",
    "training_metadata": "training_metadata.json",
    "final_metrics": "final_metrics.json",
    "ranking_evidence": "ranking_evidence.pt",
    "top_item_rankings": "top_item_rankings.json",
    "training_diagnostics": "g3_training_diagnostics.json",
    "sweep_log": "sweep.log",
}
_QUEUE_RECORD_KEYS = {
    "id",
    "batch_id",
    "data_group",
    "dispatched_at",
    "environment",
    "exit_code",
    "finished_at",
    "run",
    "script",
    "submitted_at",
}
_ENVIRONMENT_KEYS = {
    JOB_ENVIRONMENT,
    LEDGER_ENVIRONMENT,
    "WANDB_MODE",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "POLARS_MAX_THREADS",
}
_EXPECTED_FINAL_ROW_IDS = {
    "rq3_output_learned": "rq3_output_learned:08",
    "rq3_output_frozen_content": "rq3_output_frozen_content:08",
    "rq3_output_trainable_content": "rq3_output_trainable_content:08",
    "rq3_output_learned_frozen_content": "rq3_output_learned_frozen_content:04",
    "rq3_output_learned_trainable_content": (
        "rq3_output_learned_trainable_content:04"
    ),
}


def build_rq3_final_evidence(
    root: Path,
    *,
    batch_id: str = RQ3_BOUNDARY_BATCH_ID,
) -> dict[str, object]:
    root = root.resolve(strict=True)
    if batch_id != RQ3_BOUNDARY_BATCH_ID:
        raise ValueError("final RQ3 evidence requires the exact boundary batch")
    initial_path = root / RQ3_INITIAL_EVIDENCE_PATH
    initial = load_rq3_post_boundary_evidence(initial_path, root=root)
    if initial.get("sha256") != RQ3_INITIAL_EVIDENCE_LOGICAL_SHA256:
        raise ValueError("final RQ3 evidence received another initial search")
    ledger_path = root / RQ3_BOUNDARY_LEDGER_PATH
    ledger = load_rq3_boundary_ledger(
        ledger_path,
        root=root,
        full_validation=False,
    )
    batch_path = root / "generated/training-queue-service/batches" / f"{batch_id}.json"
    job_ids = _validate_batch(_load_json(batch_path), batch_id=batch_id)
    source_ledger = load_rq3_post_boundary_ledger(root / ledger.source_ledger.path)
    feature_path = resolve_rq3_boundary_feature_data(root, ledger)
    context_path = root / "generated/logs/.ranking-evidence/g3-native50m/context.pt"
    item_counts = load_training_item_counts(feature_path)
    feature_identity = _feature_identity(source_ledger)
    boundary_runs = [
        _collect_boundary_run(
            root=root,
            ledger=ledger,
            ledger_path=ledger_path,
            row=row,
            batch_id=batch_id,
            job_id=job_id,
            context_path=context_path,
            item_counts=item_counts,
            feature_identity=feature_identity,
        )
        for row, job_id in zip(ledger.rows, job_ids, strict=True)
    ]
    initial_runs = initial.get("all_tuning_opportunities")
    if not isinstance(initial_runs, list) or len(initial_runs) != 45:
        raise ValueError("final RQ3 evidence lost its initial opportunities")
    all_runs = [*initial_runs, *boundary_runs]
    winners = select_rq3_family_winners_with_extensions(all_runs)
    boundary_resolution = resolve_rq3_boundary_extensions(
        initial=initial,
        boundary_runs=boundary_runs,
        winners=winners,
    )
    downstream = resolve_rq3_downstream_selection(winners, boundary_resolution)
    mechanism = _mechanism_assessment(winners)
    payload = {
        "schema_version": 1,
        "kind": "g3_rq3_final_native50m_evidence",
        "protocol_sha256": APPROVED_PROTOCOL_SHA256,
        "initial_evidence": _file_fact(root, initial_path)
        | {"logical_sha256": initial["sha256"]},
        "boundary_ledger": _file_fact(root, ledger_path)
        | {"logical_sha256": ledger.sha256},
        "queue_batch": _file_fact(root, batch_path) | {"batch_id": batch_id},
        "ranking_context": _file_fact(root, context_path),
        "feature_data": _file_fact(root, feature_path),
        "opportunity_accounting": {
            "initial_logical_rows": 45,
            "initial_reused_rows": 7,
            "initial_physical_rows": 38,
            "boundary_physical_rows": 6,
            "all_logical_rows": 51,
        },
        "boundary_runs": boundary_runs,
        "all_tuning_opportunities": all_runs,
        "family_selections": {
            family_id: {
                "status": (
                    "resolved"
                    if not boundary_resolution[family_id]["extension_required"]
                    else "second_boundary_requires_approval"
                ),
                "selected": winners[family_id],
                "boundary_decision": boundary_resolution[family_id],
            }
            for family_id in RQ3_OUTPUT_FAMILY_IDS
        },
        "downstream_selection": downstream,
        "reader_metrics": {
            family_id: {
                "row_id": run["row_id"],
                "overall": run["metrics"],
                "item_frequency_slices": run["slices"],
                "efficiency": run["efficiency"],
            }
            for family_id, run in winners.items()
        },
        "selected_winner_contrasts": build_rq3_paired_contrasts(winners),
        "matched_coordinate_contrasts": initial["matched_coordinate_contrasts"],
        "mechanism_assessment": mechanism,
    }
    return _final_document(payload)


def select_rq3_family_winners_with_extensions(
    runs: list[Mapping[str, object]],
) -> dict[str, Mapping[str, object]]:
    selected = {}
    for family_id in RQ3_OUTPUT_FAMILY_IDS:
        family = [run for run in runs if run.get("family_id") == family_id]
        expected = 12 if family_id in RQ3_BOUNDARY_FAMILY_IDS else 9
        if len(family) != expected:
            raise ValueError(f"final RQ3 family {family_id} has {len(family)} rows")
        selected[family_id] = min(
            family,
            key=lambda run: (
                -float(run["metrics"]["recall@100"]),
                -float(run["metrics"]["ndcg@100"]),
                float(run["queue_wall_seconds"]),
                str(run["row_id"]),
            ),
        )
    return selected


def resolve_rq3_boundary_extensions(
    *,
    initial: Mapping[str, object],
    boundary_runs: list[Mapping[str, object]],
    winners: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    initial_selections = initial.get("family_selections")
    if not isinstance(initial_selections, dict):
        raise ValueError("final RQ3 evidence lacks initial family selections")
    result = {}
    for family_id in RQ3_OUTPUT_FAMILY_IDS:
        winner = winners[family_id]
        family_boundary = [
            run for run in boundary_runs if run.get("family_id") == family_id
        ]
        if family_id not in RQ3_BOUNDARY_FAMILY_IDS:
            if family_boundary:
                raise ValueError(f"unexpected RQ3 boundary rows for {family_id}")
            result[family_id] = {
                "extension_required": False,
                "resolution": "initial_surface_resolved",
            }
            continue
        if len(family_boundary) != 3:
            raise ValueError(f"RQ3 boundary family {family_id} lacks three probes")
        initial_selected = initial_selections[family_id]["selected"]
        boundary_by_rate = {
            float(run["deep_learning_rate"]): run for run in family_boundary
        }
        if set(boundary_by_rate) != set(RQ3_BOUNDARY_DEEP_LRS):
            raise ValueError(f"RQ3 boundary rates changed for {family_id}")
        winner_rate = float(winner["deep_learning_rate"])
        unresolved = (
            winner.get("row_id") in {run["row_id"] for run in family_boundary}
            and winner_rate == min(RQ3_BOUNDARY_DEEP_LRS)
        )
        result[family_id] = {
            "extension_required": unresolved,
            "resolution": (
                "second_lower_boundary_requires_approval"
                if unresolved
                else "three_lower_probes_completed"
            ),
            "initial_selected_row_id": initial_selected["row_id"],
            "initial_selected_deep_learning_rate": initial_selected[
                "deep_learning_rate"
            ],
            "tested_boundary_deep_learning_rates": list(RQ3_BOUNDARY_DEEP_LRS),
            "selected_row_id": winner["row_id"],
            "selected_deep_learning_rate": winner_rate,
        }
    return result


def persist_rq3_final_evidence(
    path: Path,
    document: Mapping[str, object],
    *,
    root: Path,
) -> Path:
    root = root.resolve(strict=True)
    expected_path = (root / RQ3_FINAL_EVIDENCE_PATH).resolve()
    if path.resolve() != expected_path:
        raise ValueError("final RQ3 evidence path is not canonical")
    validated = _validate_final_document(dict(document))
    authenticated = build_rq3_final_evidence(root)
    if _canonical_json(validated) != _canonical_json(authenticated):
        raise ValueError("final RQ3 evidence differs from authenticated artifacts")
    content = (_canonical_json(authenticated) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(content)
    except FileExistsError:
        if path.read_bytes() != content:
            raise RuntimeError(f"immutable final RQ3 evidence differs: {path}")
    return path


def load_rq3_final_evidence(path: Path, *, root: Path) -> dict[str, object]:
    document = _validate_final_document(_load_json(path))
    if _canonical_json(document) != _canonical_json(build_rq3_final_evidence(root)):
        raise ValueError("final RQ3 evidence differs from authenticated artifacts")
    return document


def _collect_boundary_run(
    *,
    root: Path,
    ledger,
    ledger_path: Path,
    row,
    batch_id: str,
    job_id: str,
    context_path: Path,
    item_counts,
    feature_identity,
) -> dict[str, object]:
    completed_path = root / "generated/training-queue-service/completed" / f"{job_id}.json"
    queue_job = _load_json(completed_path)
    expected_script = (
        root
        / "experiments/g3_pretrained_item_embeddings/launchers/run_rq3_boundary.py"
    ).resolve(strict=True)
    values = _validate_boundary_queue_job(
        queue_job,
        job_id=job_id,
        batch_id=batch_id,
        run_name=row.run_name,
        ledger_path=ledger_path,
        expected_script=expected_script,
        row_id=row.id,
    )
    compiled = decode_control_job(values[JOB_ENVIRONMENT], ledger)
    if compiled.row_id != row.id or compiled.job != row.to_dict():
        raise ValueError(f"RQ3 boundary queue payload differs for {row.id}")
    directory = root / "generated/logs" / row.run_name
    contract = _load_json(directory / _ARTIFACT_FILENAMES["job_contract"])
    if contract != compiled.to_dict() | {
        "ledger_path": str(ledger_path),
        "ledger_sha256": ledger.sha256,
    }:
        raise ValueError(f"RQ3 boundary job contract differs for {row.id}")
    artifact_paths = {
        name: directory / filename for name, filename in _ARTIFACT_FILENAMES.items()
    }
    verify_unique_completed_run(
        root / "generated/training-queue-service",
        run_name=row.run_name,
        expected_job_id=job_id,
    )
    verify_artifacts_in_job_window(
        tuple(artifact_paths.values()),
        dispatched_at=float(queue_job["dispatched_at"]),
        finished_at=float(queue_job["finished_at"]),
        run_label=row.id,
    )
    return _collect_artifacts(
        root=root,
        row=row,
        artifact_paths=artifact_paths,
        context_path=context_path,
        item_counts=item_counts,
        feature_identity=feature_identity,
        queue_wall_seconds=float(queue_job["finished_at"])
        - float(queue_job["dispatched_at"]),
        queue_job=_file_fact(root, completed_path) | {"job_id": job_id},
        source_row_id=None,
        verify_window=None,
    )


def _validate_batch(batch: Mapping[str, object], *, batch_id: str) -> list[str]:
    jobs = batch.get("jobs")
    if (
        set(batch) != {"id", "jobs", "sealed", "sealed_at", "submitted_at"}
        or batch.get("id") != batch_id
        or batch.get("sealed") is not True
        or not isinstance(jobs, list)
        or len(jobs) != 6
        or len(set(jobs)) != 6
        or any(not isinstance(job_id, str) or not job_id for job_id in jobs)
        or not _finite_number(batch.get("submitted_at"))
        or not _finite_number(batch.get("sealed_at"))
        or float(batch["sealed_at"]) < float(batch["submitted_at"])
    ):
        raise ValueError("RQ3 boundary batch is not the exact sealed six-job batch")
    return [str(job_id) for job_id in jobs]


def _validate_boundary_queue_job(
    queue_job: Mapping[str, object],
    *,
    job_id: str,
    batch_id: str,
    run_name: str,
    ledger_path: Path,
    expected_script: Path,
    row_id: str,
) -> dict[str, str]:
    if (
        set(queue_job) != _QUEUE_RECORD_KEYS
        or queue_job.get("id") != job_id
        or queue_job.get("batch_id") != batch_id
        or queue_job.get("run") != run_name
        or queue_job.get("exit_code") != 0
        or queue_job.get("data_group") != "g3-native50m-likes"
        or Path(str(queue_job.get("script"))).resolve() != expected_script
        or not _ordered_job_times(queue_job)
    ):
        raise ValueError(f"RQ3 boundary queue completion differs for {row_id}")
    environment = queue_job.get("environment")
    pairs = (
        [value.split("=", 1) for value in environment]
        if isinstance(environment, list)
        else []
    )
    values = dict(pairs)
    thread_keys = _ENVIRONMENT_KEYS - {
        JOB_ENVIRONMENT,
        LEDGER_ENVIRONMENT,
        "WANDB_MODE",
    }
    if (
        len(pairs) != len(values) == len(_ENVIRONMENT_KEYS)
        or set(values) != _ENVIRONMENT_KEYS
        or values.get("WANDB_MODE") != "offline"
        or Path(str(values.get(LEDGER_ENVIRONMENT))).resolve() != ledger_path
        or any(values[name] != "1" for name in thread_keys)
    ):
        raise ValueError(f"RQ3 boundary queue environment differs for {row_id}")
    return values


def _validate_final_document(document: dict[str, object]) -> dict[str, object]:
    expected_keys = {
        "schema_version",
        "kind",
        "protocol_sha256",
        "initial_evidence",
        "boundary_ledger",
        "queue_batch",
        "ranking_context",
        "feature_data",
        "opportunity_accounting",
        "boundary_runs",
        "all_tuning_opportunities",
        "family_selections",
        "downstream_selection",
        "reader_metrics",
        "selected_winner_contrasts",
        "matched_coordinate_contrasts",
        "mechanism_assessment",
        "sha256",
    }
    payload = {key: value for key, value in document.items() if key != "sha256"}
    accounting = document.get("opportunity_accounting")
    boundary_runs = document.get("boundary_runs")
    all_runs = document.get("all_tuning_opportunities")
    selections = document.get("family_selections")
    readers = document.get("reader_metrics")
    downstream = document.get("downstream_selection")
    mechanism = document.get("mechanism_assessment")
    if (
        set(document) != expected_keys
        or document.get("schema_version") != 1
        or document.get("kind") != "g3_rq3_final_native50m_evidence"
        or document.get("protocol_sha256") != APPROVED_PROTOCOL_SHA256
        or document.get("sha256") != _sha256(payload)
        or accounting
        != {
            "initial_logical_rows": 45,
            "initial_reused_rows": 7,
            "initial_physical_rows": 38,
            "boundary_physical_rows": 6,
            "all_logical_rows": 51,
        }
        or not isinstance(boundary_runs, list)
        or len(boundary_runs) != 6
        or not isinstance(all_runs, list)
        or len(all_runs) != 51
        or len({run.get("row_id") for run in all_runs if isinstance(run, dict)}) != 51
        or not isinstance(selections, dict)
        or set(selections) != set(RQ3_OUTPUT_FAMILY_IDS)
        or not isinstance(readers, dict)
        or set(readers) != set(RQ3_OUTPUT_FAMILY_IDS)
        or not isinstance(downstream, dict)
        or downstream.get("status") != "resolved"
        or not isinstance(mechanism, dict)
        or mechanism.get("unexpected_ordering") is not False
    ):
        raise ValueError("final RQ3 evidence schema or identity is invalid")
    for family_id, row_id in _EXPECTED_FINAL_ROW_IDS.items():
        selection = selections[family_id]
        selected = selection.get("selected") if isinstance(selection, dict) else None
        reader = readers[family_id]
        if (
            not isinstance(selection, dict)
            or selection.get("status") != "resolved"
            or not isinstance(selected, dict)
            or selected.get("row_id") != row_id
            or not isinstance(reader, dict)
            or reader.get("row_id") != row_id
        ):
            raise ValueError(f"final RQ3 selection changed for {family_id}")
    if (
        downstream.get("rq4_scientific_selected", {}).get("row_id")
        != _EXPECTED_FINAL_ROW_IDS["rq3_output_learned_frozen_content"]
        or downstream.get("aggregate_selected", {}).get("row_id")
        != _EXPECTED_FINAL_ROW_IDS["rq3_output_learned"]
        or downstream.get("treatment_promoted") is not False
        or mechanism.get("best_family_by_selection_rule")
        != "rq3_output_learned_frozen_content"
    ):
        raise ValueError("final RQ3 downstream selection changed")
    return document


def _final_document(payload: Mapping[str, object]) -> dict[str, object]:
    document = dict(payload)
    document["sha256"] = _sha256(document)
    return document


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _ordered_job_times(job: Mapping[str, object]) -> bool:
    values = tuple(job.get(name) for name in ("submitted_at", "dispatched_at", "finished_at"))
    return all(_finite_number(value) for value in values) and (
        float(values[0]) <= float(values[1]) <= float(values[2])
    )


def _finite_number(value: object) -> bool:
    return type(value) in {int, float} and math.isfinite(float(value))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--write", action="store_true")
    arguments = parser.parse_args()
    evidence = build_rq3_final_evidence(arguments.root)
    path = arguments.root / RQ3_FINAL_EVIDENCE_PATH
    if arguments.write:
        persist_rq3_final_evidence(path, evidence, root=arguments.root)
    print(json.dumps({"path": str(path), "sha256": evidence["sha256"]}))


if __name__ == "__main__":
    main()
