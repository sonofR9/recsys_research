from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.g3_pretrained_item_embeddings.analysis.control_calibration import (
    _file_fact,
    _load_json,
    _recompute_metrics,
)
from experiments.g3_pretrained_item_embeddings.analysis.queue_attribution import (
    verify_artifacts_in_job_window,
    verify_unique_completed_run,
)
from experiments.g3_pretrained_item_embeddings.analysis.rq2_capacity_results import (
    APPROVED_RQ2_CAPACITY_EVIDENCE_SHA256,
    RQ2_CAPACITY_EVIDENCE_PATH,
    load_rq2_capacity_evidence,
)
from experiments.g3_pretrained_item_embeddings.analysis.rq2_content_horizon_results import (
    _EXPECTED_DATASET,
    _EXPECTED_REPRESENTATION,
)
from experiments.g3_pretrained_item_embeddings.analysis.rq2_id_boundary_results import (
    load_rq2_id_boundary_evidence,
)
from experiments.g3_pretrained_item_embeddings.analysis.rq2_next_stage_results import (
    load_rq2_next_stage_evidence,
)
from experiments.g3_pretrained_item_embeddings.launchers.control import (
    decode_control_job,
)
from experiments.g3_pretrained_item_embeddings.launchers.rq2_content_deep_lr_boundary import (
    JOB_ENVIRONMENT,
    LEDGER_ENVIRONMENT,
    verify_rq2_content_deep_lr_boundary_inputs,
)
from experiments.g3_pretrained_item_embeddings.protocol.constants import (
    APPROVED_PROTOCOL_SHA256,
)
from experiments.g3_pretrained_item_embeddings.protocol.rq2_content_deep_lr_boundary_ledger import (
    APPROVED_RQ2_CONTENT_DEEP_LR_BOUNDARY_LEDGER_SHA256,
    RQ2_CONTENT_DEEP_LR_BOUNDARY_LEDGER_PATH,
    load_bound_rq2_content_horizon_ancestry,
    load_rq2_content_deep_lr_boundary_ledger,
)


RQ2_CONTENT_DEEP_LR_BOUNDARY_EVIDENCE_PATH = (
    "experiments/g3_pretrained_item_embeddings/evidence/"
    "rq2_content_width32_horizon40_deep_lr_boundary_results.json"
)
EXPECTED_REPRESENTATION = _EXPECTED_REPRESENTATION

_METRIC_NAMES = tuple(
    f"{name}@{cutoff}"
    for name in ("recall", "ndcg", "mrr", "capped_recall", "coverage")
    for cutoff in (10, 50, 100)
)
_SELECTION_RULE = (
    "validation Recall@100, validation NDCG@100, lower queue wall time, then "
    "combined manifest order"
)
_EXPECTED_ROW_REPRESENTATION = {
    "id": "rq2_content_concat",
    "history": "learned_item_id_plus_frozen_content",
    "catalog": "learned_item_id",
    "history_hidden_dim": 32,
    "separate_history_catalog_tables": True,
    "content_trainable": False,
    "content_width": 128,
}


def select_rq2_content_boundary_candidate(
    candidates: Sequence[Mapping[str, object]],
) -> Mapping[str, object]:
    if not candidates:
        raise ValueError("RQ2 content deep-LR selection requires candidates")
    return min(
        candidates,
        key=lambda run: (
            -float(_metrics(run)["recall@100"]),
            -float(_metrics(run)["ndcg@100"]),
            float(run["queue_wall_seconds"]),
            int(
                run["combined_manifest_order"]
                if "combined_manifest_order" in run
                else run["manifest_order"]
            ),
        ),
    )


def build_rq2_content_deep_lr_boundary_evidence(
    root: Path,
    *,
    batch_id: str,
) -> dict[str, object]:
    root = root.resolve(strict=True)
    if not batch_id:
        raise ValueError("RQ2 content deep-LR batch id is required")
    ledger_path = root / RQ2_CONTENT_DEEP_LR_BOUNDARY_LEDGER_PATH
    ledger = load_rq2_content_deep_lr_boundary_ledger(ledger_path, root=root)
    if ledger.sha256 != APPROVED_RQ2_CONTENT_DEEP_LR_BOUNDARY_LEDGER_SHA256:
        raise ValueError("RQ2 content deep-LR evidence received another ledger")
    verify_rq2_content_deep_lr_boundary_inputs(
        root,
        ledger,
        full_validation=True,
    )
    horizon_evidence, _ = load_bound_rq2_content_horizon_ancestry(root)
    next_path = root / ledger.resolved_next_stage_evidence.path
    next_evidence = load_rq2_next_stage_evidence(next_path)
    capacity_path = root / RQ2_CAPACITY_EVIDENCE_PATH
    capacity_evidence = load_rq2_capacity_evidence(capacity_path)
    id_path = root / ledger.id_boundary_evidence.path
    id_evidence = load_rq2_id_boundary_evidence(id_path)
    _validate_predecessors(
        ledger,
        horizon_evidence=horizon_evidence,
        next_evidence=next_evidence,
        capacity_evidence=capacity_evidence,
        id_evidence=id_evidence,
    )
    source = _source_candidate(ledger, horizon_evidence)
    expected_invariants, expected_normalized_schedule = _source_runtime_identity(
        root, source
    )
    batch_path = (
        root / "generated/training-queue-service/batches" / f"{batch_id}.json"
    )
    batch = _load_json(batch_path)
    job_ids = _validate_batch(batch, batch_id=batch_id, expected_jobs=len(ledger.rows))
    context_path = root / "generated/logs/.ranking-evidence/g3-native50m/context.pt"
    runs = [
        _collect_run(
            root=root,
            ledger=ledger,
            ledger_path=ledger_path,
            context_path=context_path,
            batch_id=batch_id,
            job_id=job_id,
            row=row.to_dict(),
            manifest_order=index,
            expected_invariants=expected_invariants,
            expected_normalized_schedule=expected_normalized_schedule,
        )
        for index, (row, job_id) in enumerate(
            zip(ledger.rows, job_ids, strict=True)
        )
    ]
    expected_ids = {f"rq2_content_concat:{index:02d}" for index in range(16, 19)}
    if len(runs) != 3 or {run["row_id"] for run in runs} != expected_ids:
        raise ValueError("RQ2 content deep-LR evidence lacks the exact three probes")
    candidates = [source, *runs]
    selected = select_rq2_content_boundary_candidate(candidates)
    boundary = _boundary_decision(selected, runs)
    resolved = boundary["status"] == "resolved"
    id_selected = _selected_id_candidate(id_evidence)
    all_tuning = _all_tuning_rows(
        capacity_evidence=capacity_evidence,
        next_evidence=next_evidence,
        horizon_evidence=horizon_evidence,
        current=runs,
    )
    reusable = _reusable_width_32_rows(all_tuning)
    comparison = _final_comparison(selected, id_selected) if resolved else None
    rq3_inputs = (
        {
            "selected_content_input": selected,
            "id_only_control": id_selected,
            "reusable_width_32_content_rows": reusable,
        }
        if resolved
        else None
    )
    payload = {
        "schema_version": 1,
        "kind": "g3_rq2_content_width32_horizon40_deep_lr_boundary_evidence",
        "protocol_sha256": APPROVED_PROTOCOL_SHA256,
        "content_deep_lr_boundary_ledger": _file_fact(root, ledger_path)
        | {"logical_sha256": ledger.sha256},
        "content_horizon_evidence": _file_fact(
            root, root / ledger.content_horizon_evidence.path
        )
        | {"logical_sha256": horizon_evidence["sha256"]},
        "resolved_next_stage_evidence": _file_fact(root, next_path)
        | {"logical_sha256": next_evidence["sha256"]},
        "capacity_preselection_evidence": _file_fact(root, capacity_path)
        | {"logical_sha256": capacity_evidence["sha256"]},
        "id_boundary_evidence": _file_fact(root, id_path)
        | {"logical_sha256": id_evidence["sha256"]},
        "queue_batch": _file_fact(root, batch_path) | {"batch_id": batch_id},
        "ranking_context": _file_fact(root, context_path),
        "boundary_tuning_ledger": runs,
        "all_tuning_ledger": all_tuning,
        "final_content_selection": {
            "status": boundary["status"],
            "selection_rule": _SELECTION_RULE,
            "candidate_row_ids": [run["row_id"] for run in candidates],
            "selected": selected if resolved else None,
            "provisional_selected": None if resolved else selected,
            "boundary_decision": boundary,
        },
        "final_rq2_comparison": comparison,
        "rq3_inputs": rq3_inputs,
        "opportunity_accounting": ledger.opportunity_accounting,
    }
    return _document(payload)


def persist_rq2_content_deep_lr_boundary_evidence(
    path: Path,
    document: Mapping[str, object],
    *,
    root: Path,
) -> Path:
    validated = _validate_document(dict(document))
    _authenticate_document(validated, root=root)
    content = (_canonical_json(validated) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(content)
    except FileExistsError:
        if path.read_bytes() != content:
            raise RuntimeError(
                f"immutable RQ2 content deep-LR evidence differs: {path}"
            )
    return path


def load_rq2_content_deep_lr_boundary_evidence(
    path: Path,
    *,
    root: Path,
) -> dict[str, object]:
    return verify_rq2_content_deep_lr_boundary_evidence(path, root=root)


def verify_rq2_content_deep_lr_boundary_evidence(
    path: Path,
    *,
    root: Path,
) -> dict[str, object]:
    evidence = _validate_document(_load_json(path))
    _authenticate_document(evidence, root=root)
    return evidence


def canonical_sha256(value: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _authenticate_document(
    evidence: Mapping[str, object],
    *,
    root: Path,
) -> None:
    batch = evidence.get("queue_batch")
    if not isinstance(batch, dict) or not isinstance(batch.get("batch_id"), str):
        raise ValueError("RQ2 content deep-LR evidence has no bound batch")
    rebuilt = build_rq2_content_deep_lr_boundary_evidence(
        root,
        batch_id=batch["batch_id"],
    )
    if _canonical_json(evidence) != _canonical_json(rebuilt):
        raise ValueError("RQ2 content deep-LR evidence differs from bound artifacts")


def _validate_batch(
    batch: Mapping[str, object],
    *,
    batch_id: str,
    expected_jobs: int,
) -> list[str]:
    jobs = batch.get("jobs")
    if (
        set(batch) != {"id", "jobs", "sealed", "sealed_at", "submitted_at"}
        or batch.get("id") != batch_id
        or batch.get("sealed") is not True
        or not isinstance(jobs, list)
        or len(jobs) != expected_jobs == 3
        or any(not isinstance(job, str) or not job for job in jobs)
        or len(set(jobs)) != len(jobs)
        or not _finite_number(batch.get("submitted_at"))
        or not _finite_number(batch.get("sealed_at"))
        or float(batch["sealed_at"]) < float(batch["submitted_at"])
    ):
        raise ValueError("RQ2 content deep-LR batch is not the exact sealed batch")
    return list(jobs)


def _collect_run(
    *,
    root: Path,
    ledger: Any,
    ledger_path: Path,
    context_path: Path,
    batch_id: str,
    job_id: str,
    row: dict[str, object],
    manifest_order: int,
    expected_invariants: Mapping[str, object],
    expected_normalized_schedule: Sequence[float],
) -> dict[str, object]:
    completed_path = (
        root / "generated/training-queue-service/completed" / f"{job_id}.json"
    )
    queue_job = _load_json(completed_path)
    expected_script = (
        root
        / "experiments/g3_pretrained_item_embeddings/launchers/"
        "run_rq2_content_deep_lr_boundary.py"
    ).resolve(strict=True)
    expected_queue_keys = {
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
    if (
        set(queue_job) != expected_queue_keys
        or queue_job.get("id") != job_id
        or queue_job.get("batch_id") != batch_id
        or queue_job.get("run") != row["run_name"]
        or queue_job.get("exit_code") != 0
        or queue_job.get("data_group") != "g3-native50m-likes"
        or Path(str(queue_job.get("script"))).resolve() != expected_script
        or not _ordered_job_times(queue_job)
    ):
        raise ValueError(f"RQ2 content deep-LR completion differs for {row['id']}")
    verify_unique_completed_run(
        root / "generated/training-queue-service",
        run_name=str(row["run_name"]),
        expected_job_id=job_id,
    )
    environment = queue_job.get("environment")
    if not isinstance(environment, list) or len(environment) != 3:
        raise ValueError(f"RQ2 content deep-LR environment differs for {row['id']}")
    pairs = [value.split("=", 1) for value in environment if "=" in value]
    if len(pairs) != 3 or len({name for name, _ in pairs}) != 3:
        raise ValueError(f"RQ2 content deep-LR environment differs for {row['id']}")
    values = dict(pairs)
    if (
        set(values) != {"WANDB_MODE", JOB_ENVIRONMENT, LEDGER_ENVIRONMENT}
        or values["WANDB_MODE"] != "offline"
        or Path(values[LEDGER_ENVIRONMENT]).resolve() != ledger_path
    ):
        raise ValueError(f"RQ2 content deep-LR environment differs for {row['id']}")
    compiled = decode_control_job(values[JOB_ENVIRONMENT], ledger)
    if compiled.row_id != row["id"] or compiled.job != row:
        raise ValueError(f"RQ2 content deep-LR payload differs for {row['id']}")
    run_directory = root / "generated/logs" / str(row["run_name"])
    contract = _load_json(
        run_directory / "g3_rq2_content_deep_lr_boundary_job.json"
    )
    if contract != compiled.to_dict() | {
        "ledger_path": str(ledger_path),
        "ledger_sha256": ledger.sha256,
    }:
        raise ValueError(f"RQ2 content deep-LR contract differs for {row['id']}")
    training = row.get("training")
    representation = row.get("representation")
    if (
        row.get("family_id") != "rq2_content_concat"
        or row.get("phase") != "deep_learning_rate_lower_boundary_extension"
        or row.get("stage")
        != "rq2_content_width32_horizon40_deep_lr_boundary"
        or row.get("role") != "deep_learning_rate_boundary_probe"
        or row.get("reused_from") is not None
        or row.get("dataset") != _EXPECTED_DATASET
        or representation != _EXPECTED_ROW_REPRESENTATION
        or not isinstance(training, dict)
        or set(training)
        != {
            "batch_size",
            "seed",
            "embedding_learning_rate",
            "deep_learning_rate",
            "horizon_epochs",
            "validate_every_epoch",
            "restore_best_validation_epoch",
        }
        or training["batch_size"] != 512
        or training["seed"] != 42
        or training["embedding_learning_rate"] != 0.3041556165944196
        or training["horizon_epochs"] != 40
        or training["validate_every_epoch"] is not True
        or training["restore_best_validation_epoch"] is not True
    ):
        raise ValueError(f"RQ2 content deep-LR row identity differs for {row['id']}")
    metadata = _load_json(run_directory / "training_metadata.json")
    expected_metadata = {
        "batch_size": 512,
        "seed": 42,
        "embedding_learning_rate": training["embedding_learning_rate"],
        "deep_learning_rate": training["deep_learning_rate"],
        "lr_schedule_horizon_epochs": 40,
        "epochs_trained": 40,
        "stopped_epoch": 40,
        "lr_horizon_complete": True,
        "selection_resolved": True,
        "early_stopped": False,
        "g3_dataset_size": "native-50m",
        "g3_protocol_sha256": APPROVED_PROTOCOL_SHA256,
        "training_semantics_revision": 2,
    }
    if any(metadata.get(name) != value for name, value in expected_metadata.items()):
        raise ValueError(f"RQ2 content deep-LR metadata differs for {row['id']}")
    if metadata.get("g3_representation") != EXPECTED_REPRESENTATION:
        raise ValueError(f"RQ2 content deep-LR representation differs for {row['id']}")
    invariants = metadata.get("transfer_invariants")
    if not isinstance(invariants, dict) or invariants != expected_invariants:
        raise ValueError(f"RQ2 content deep-LR invariants differ for {row['id']}")
    best_epoch = metadata.get("best_epoch")
    if type(best_epoch) is not int or not 1 <= best_epoch <= 40:
        raise ValueError(f"RQ2 content deep-LR best epoch differs for {row['id']}")
    traces = metadata.get("lr_group_traces")
    if not isinstance(traces, dict) or set(traces) != {"embedding", "deep"}:
        raise ValueError(f"RQ2 content deep-LR schedule differs for {row['id']}")
    for trace in traces.values():
        if (
            not isinstance(trace, list)
            or len(trace) != 40
            or any(not _finite_number(value) for value in trace)
            or float(trace[-1]) != 0.0
        ):
            raise ValueError(f"RQ2 content deep-LR schedule differs for {row['id']}")
    normalized_traces = {
        "embedding": [
            float(value) / float(training["embedding_learning_rate"])
            for value in traces["embedding"]
        ],
        "deep": [
            float(value) / float(training["deep_learning_rate"])
            for value in traces["deep"]
        ],
    }
    if any(
        value < 0.0 or value > 1.0
        for trace in normalized_traces.values()
        for value in trace
    ) or any(
        later > earlier
        for trace in normalized_traces.values()
        for earlier, later in zip(trace, trace[1:])
    ):
        raise ValueError(f"RQ2 content deep-LR schedule differs for {row['id']}")
    if any(
        abs(embedding - deep) > 1e-15
        for embedding, deep in zip(
            normalized_traces["embedding"],
            normalized_traces["deep"],
            strict=True,
        )
    ):
        raise ValueError(f"RQ2 content deep-LR schedule differs for {row['id']}")
    if any(
        abs(actual - expected) > 1e-15
        for actual, expected in zip(
            normalized_traces["embedding"],
            expected_normalized_schedule,
            strict=True,
        )
    ):
        raise ValueError(f"RQ2 content deep-LR schedule differs for {row['id']}")
    metrics_path = run_directory / "final_metrics.json"
    metrics = _load_json(metrics_path)
    if set(metrics) != {*_METRIC_NAMES, "num_users"} or any(
        not _finite_number(value) for value in metrics.values()
    ):
        raise ValueError(f"RQ2 content deep-LR metrics differ for {row['id']}")
    ranking_path = run_directory / "ranking_evidence.pt"
    rankings_path = run_directory / "top_item_rankings.json"
    recomputed = _recompute_metrics(context_path, ranking_path, rankings_path)
    if set(recomputed) != set(metrics) or any(
        abs(float(metrics[name]) - float(recomputed[name])) > 1e-15
        for name in metrics
    ):
        raise ValueError(f"RQ2 content deep-LR metrics differ for {row['id']}")
    diagnostics = _load_json(run_directory / "g3_training_diagnostics.json")
    nonfinite = _values_named(diagnostics, "nonfinite_count")
    if not nonfinite or any(value != 0 for value in nonfinite):
        raise ValueError(f"RQ2 content deep-LR diagnostics differ for {row['id']}")
    artifact_filenames = (
        ("job_contract", "g3_rq2_content_deep_lr_boundary_job.json"),
        ("training_metadata", "training_metadata.json"),
        ("final_metrics", "final_metrics.json"),
        ("ranking_evidence", "ranking_evidence.pt"),
        ("top_item_rankings", "top_item_rankings.json"),
        ("training_diagnostics", "g3_training_diagnostics.json"),
        ("sweep_log", "sweep.log"),
    )
    verify_artifacts_in_job_window(
        tuple(run_directory / filename for _, filename in artifact_filenames),
        dispatched_at=float(queue_job["dispatched_at"]),
        finished_at=float(queue_job["finished_at"]),
        run_label=str(row["id"]),
    )
    return {
        "manifest_order": manifest_order,
        "combined_manifest_order": manifest_order + 1,
        "row_id": row["id"],
        "run_name": row["run_name"],
        "family_id": row["family_id"],
        "capacity": 32,
        "embedding_learning_rate": training["embedding_learning_rate"],
        "deep_learning_rate": training["deep_learning_rate"],
        "horizon_epochs": 40,
        "best_epoch": best_epoch,
        "epochs_trained": metadata["epochs_trained"],
        "queue_wall_seconds": float(queue_job["finished_at"])
        - float(queue_job["dispatched_at"]),
        "metrics": metrics,
        "metric_provenance": {
            "recomputed_from_ranking_evidence": True,
            "absolute_tolerance": 1e-15,
            "num_users": recomputed["num_users"],
        },
        "diagnostic_nonfinite_count": sum(int(value) for value in nonfinite),
        "queue_job": _file_fact(root, completed_path) | {"job_id": job_id},
        "artifacts": {
            name: _file_fact(root, run_directory / filename)
            for name, filename in artifact_filenames
        },
    }


def _validate_predecessors(
    ledger: Any,
    *,
    horizon_evidence: Mapping[str, object],
    next_evidence: Mapping[str, object],
    capacity_evidence: Mapping[str, object],
    id_evidence: Mapping[str, object],
) -> None:
    preselection = next_evidence.get("preselection_evidence")
    if (
        horizon_evidence.get("sha256") != ledger.content_horizon_evidence.sha256
        or next_evidence.get("sha256")
        != ledger.resolved_next_stage_evidence.sha256
        or id_evidence.get("sha256") != ledger.id_boundary_evidence.sha256
        or capacity_evidence.get("sha256")
        != APPROVED_RQ2_CAPACITY_EVIDENCE_SHA256
        or not isinstance(preselection, dict)
        or preselection.get("path") != RQ2_CAPACITY_EVIDENCE_PATH
        or preselection.get("sha256") != capacity_evidence.get("sha256")
    ):
        raise ValueError("RQ2 content deep-LR predecessor evidence changed")


def _source_candidate(
    ledger: Any,
    horizon_evidence: Mapping[str, object],
) -> dict[str, object]:
    selection = horizon_evidence.get("final_content_selection")
    if not isinstance(selection, dict):
        raise ValueError("RQ2 content deep-LR source selection is absent")
    selected = selection.get("provisional_selected")
    expected = ledger.source_selection
    if (
        selection.get("status") != "pending_boundary_followup"
        or selection.get("selected") is not None
        or not isinstance(selected, dict)
        or selected.get("row_id") != expected["row_id"]
        or selected.get("capacity") != expected["capacity"]
        or selected.get("horizon_epochs") != expected["horizon_epochs"]
        or selected.get("embedding_learning_rate")
        != expected["embedding_learning_rate"]
        or selected.get("deep_learning_rate") != expected["deep_learning_rate"]
        or selected.get("best_epoch") != expected["best_epoch"]
        or _metrics(selected).get("recall@100") != expected["recall_at_100"]
        or _metrics(selected).get("ndcg@100") != expected["ndcg_at_100"]
    ):
        raise ValueError("RQ2 content deep-LR source selection changed")
    return dict(selected) | {"combined_manifest_order": 0}


def _source_runtime_identity(
    root: Path,
    selected: Mapping[str, object],
) -> tuple[Mapping[str, object], Sequence[float]]:
    artifacts = selected.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("RQ2 content deep-LR source artifacts are absent")
    fact = artifacts.get("training_metadata")
    if not isinstance(fact, dict) or not isinstance(fact.get("path"), str):
        raise ValueError("RQ2 content deep-LR source metadata is absent")
    metadata_path = root / fact["path"]
    if _file_fact(root, metadata_path) != fact:
        raise ValueError("RQ2 content deep-LR source metadata changed")
    metadata = _load_json(metadata_path)
    invariants = metadata.get("transfer_invariants")
    traces = metadata.get("lr_group_traces")
    if (
        metadata.get("g3_representation") != EXPECTED_REPRESENTATION
        or not isinstance(invariants, dict)
        or invariants.get("lr_schedule_horizon_epochs") != 40
        or invariants.get("g3_dataset_size") != "native-50m"
        or invariants.get("g3_protocol_sha256") != APPROVED_PROTOCOL_SHA256
        or invariants.get("g3_representation") != EXPECTED_REPRESENTATION
        or not isinstance(traces, dict)
        or set(traces) != {"embedding", "deep"}
        or any(
            not isinstance(trace, list) or len(trace) != 40
            for trace in traces.values()
        )
    ):
        raise ValueError("RQ2 content deep-LR source runtime identity changed")
    embedding_rate = metadata.get("embedding_learning_rate")
    deep_rate = metadata.get("deep_learning_rate")
    if not _finite_number(embedding_rate) or not _finite_number(deep_rate):
        raise ValueError("RQ2 content deep-LR source schedule changed")
    normalized_embedding = [
        float(value) / float(embedding_rate) for value in traces["embedding"]
    ]
    normalized_deep = [float(value) / float(deep_rate) for value in traces["deep"]]
    if any(
        abs(embedding - deep) > 1e-15
        for embedding, deep in zip(
            normalized_embedding,
            normalized_deep,
            strict=True,
        )
    ):
        raise ValueError("RQ2 content deep-LR source schedule changed")
    return invariants, normalized_embedding


def _boundary_decision(
    selected: Mapping[str, object],
    runs: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    outward_ids = {run["row_id"] for run in runs}
    smallest = min(float(run["deep_learning_rate"]) for run in runs)
    outward_won = selected.get("row_id") in outward_ids
    selected_smallest = (
        outward_won and float(selected["deep_learning_rate"]) == smallest
    )
    pending = bool(selected_smallest)
    return {
        "status": "pending_renewed_user_approval" if pending else "resolved",
        "outward_probe_won": outward_won,
        "selected_is_smallest_tested_deep_lr": selected_smallest,
        "smallest_tested_deep_learning_rate": smallest,
        "additional_runs_authorized": False,
        "required_actions": (
            [
                {
                    "action": "renewed_user_approval",
                    "reason": "outward_winner_on_new_lower_deep_lr_boundary",
                }
            ]
            if pending
            else []
        ),
    }


def _selected_id_candidate(id_evidence: Mapping[str, object]) -> Mapping[str, object]:
    final = id_evidence.get("final_selection")
    if not isinstance(final, dict) or final.get("boundary_decision") != {
        "status": "resolved",
        "outward_winner_on_new_boundary": False,
        "additional_runs_authorized": False,
        "next_action": "none",
    }:
        raise ValueError("RQ2 final ID-only selection is unresolved")
    selected = final.get("selected")
    if (
        not isinstance(selected, dict)
        or selected.get("row_id") != "rq2_id_only_densenet:12"
        or selected.get("family_id") != "rq2_id_only_densenet"
        or selected.get("capacity") != 255
    ):
        raise ValueError("RQ2 final ID-only winner changed")
    _metrics(selected)
    return selected


def _all_tuning_rows(
    *,
    capacity_evidence: Mapping[str, object],
    next_evidence: Mapping[str, object],
    horizon_evidence: Mapping[str, object],
    current: Sequence[Mapping[str, object]],
) -> list[Mapping[str, object]]:
    groups = (
        (capacity_evidence.get("tuning_ledger"), 18),
        (next_evidence.get("tuning_ledger"), 6),
        (horizon_evidence.get("tuning_ledger"), 3),
        (list(current), 3),
    )
    if any(
        not isinstance(rows, list)
        or len(rows) != expected
        or any(not isinstance(row, dict) for row in rows)
        for rows, expected in groups
    ):
        raise ValueError("RQ2 content deep-LR prior tuning rows changed")
    combined = [row for rows, _ in groups for row in rows]
    identities = [
        (row.get("family_id"), row.get("row_id"), row.get("run_name"))
        for row in combined
    ]
    if len(combined) != 30 or len(set(identities)) != 30:
        raise ValueError("RQ2 content deep-LR tuning rows are not unique")
    return combined


def _reusable_width_32_rows(
    rows: Sequence[Mapping[str, object]],
) -> list[Mapping[str, object]]:
    reusable = [
        row
        for row in rows
        if row.get("family_id") == "rq2_content_concat"
        and row.get("capacity") == 32
    ]
    if len(reusable) != 9:
        raise ValueError("RQ3 lacks the exact reusable width-32 content rows")
    return reusable


def _final_comparison(
    content: Mapping[str, object],
    identifier: Mapping[str, object],
) -> dict[str, object]:
    content_metrics = _metrics(content)
    identifier_metrics = _metrics(identifier)
    return {
        "content_concat": content,
        "id_only_densenet": identifier,
        "recall_at_100_delta": float(content_metrics["recall@100"])
        - float(identifier_metrics["recall@100"]),
        "ndcg_at_100_delta": float(content_metrics["ndcg@100"])
        - float(identifier_metrics["ndcg@100"]),
        "content_beats_id_only": (
            float(content_metrics["recall@100"])
            > float(identifier_metrics["recall@100"])
        ),
    }


def _metrics(run: Mapping[str, object]) -> Mapping[str, object]:
    metrics = run.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("RQ2 content deep-LR run has no metrics")
    return metrics


def _values_named(value: object, name: str) -> list[float]:
    if isinstance(value, dict):
        result = [float(value[name])] if name in value else []
        for nested in value.values():
            result.extend(_values_named(nested, name))
        return result
    if isinstance(value, list):
        result = []
        for nested in value:
            result.extend(_values_named(nested, name))
        return result
    return []


def _finite_number(value: object) -> bool:
    return type(value) in {int, float} and math.isfinite(float(value))


def _ordered_job_times(job: Mapping[str, object]) -> bool:
    names = ("submitted_at", "dispatched_at", "finished_at")
    values = tuple(job.get(name) for name in names)
    return all(_finite_number(value) for value in values) and (
        float(values[0]) <= float(values[1]) <= float(values[2])
    )


def _document(payload: Mapping[str, object]) -> dict[str, object]:
    document = dict(payload)
    document["sha256"] = canonical_sha256(document)
    return document


def _validate_document(document: dict[str, object]) -> dict[str, object]:
    expected_keys = {
        "schema_version",
        "kind",
        "protocol_sha256",
        "content_deep_lr_boundary_ledger",
        "content_horizon_evidence",
        "resolved_next_stage_evidence",
        "capacity_preselection_evidence",
        "id_boundary_evidence",
        "queue_batch",
        "ranking_context",
        "boundary_tuning_ledger",
        "all_tuning_ledger",
        "final_content_selection",
        "final_rq2_comparison",
        "rq3_inputs",
        "opportunity_accounting",
        "sha256",
    }
    payload = {name: value for name, value in document.items() if name != "sha256"}
    if (
        set(document) != expected_keys
        or type(document.get("schema_version")) is not int
        or document.get("schema_version") != 1
        or document.get("kind")
        != "g3_rq2_content_width32_horizon40_deep_lr_boundary_evidence"
        or document.get("protocol_sha256") != APPROVED_PROTOCOL_SHA256
        or document.get("sha256") != canonical_sha256(payload)
    ):
        raise ValueError("RQ2 content deep-LR evidence identity or hash is invalid")
    return document


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--write", action="store_true")
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[3]
    )
    arguments = parser.parse_args()
    root = arguments.root.resolve(strict=True)
    document = build_rq2_content_deep_lr_boundary_evidence(
        root,
        batch_id=arguments.batch_id,
    )
    path = root / RQ2_CONTENT_DEEP_LR_BOUNDARY_EVIDENCE_PATH
    if arguments.write:
        persist_rq2_content_deep_lr_boundary_evidence(
            path,
            document,
            root=root,
        )
    selection = document["final_content_selection"]
    selected = selection["selected"] or selection["provisional_selected"]
    print(
        json.dumps(
            {
                "path": str(path),
                "sha256": document["sha256"],
                "runs": len(document["boundary_tuning_ledger"]),
                "selection_status": selection["status"],
                "selected_row_id": selected["row_id"],
                "status": "materialized" if arguments.write else "preview",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
