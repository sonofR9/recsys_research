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
from experiments.g3_pretrained_item_embeddings.analysis.rq2_id_boundary_results import (
    verify_rq2_id_boundary_evidence,
)
from experiments.g3_pretrained_item_embeddings.analysis.rq2_next_stage_results import (
    verify_rq2_next_stage_evidence,
)
from experiments.g3_pretrained_item_embeddings.launchers.control import (
    decode_control_job,
)
from experiments.g3_pretrained_item_embeddings.launchers.rq2_content_horizon import (
    JOB_ENVIRONMENT,
    LEDGER_ENVIRONMENT,
    verify_rq2_content_horizon_inputs,
)
from experiments.g3_pretrained_item_embeddings.protocol.constants import (
    APPROVED_PROTOCOL,
    APPROVED_PROTOCOL_SHA256,
)
from experiments.g3_pretrained_item_embeddings.protocol.rq2_content_horizon_ledger import (
    APPROVED_RQ2_CONTENT_HORIZON_LEDGER_SHA256,
    RQ2_CONTENT_HORIZON_LEDGER_PATH,
    load_rq2_content_horizon_ledger,
)


RQ2_CONTENT_HORIZON_EVIDENCE_PATH = (
    "experiments/g3_pretrained_item_embeddings/evidence/"
    "rq2_content_width32_horizon_results.json"
)

_METRIC_NAMES = tuple(
    f"{name}@{cutoff}"
    for name in ("recall", "ndcg", "mrr", "capped_recall", "coverage")
    for cutoff in (10, 50, 100)
)
_SELECTION_RULE = (
    "validation Recall@100, validation NDCG@100, lower queue wall time, then "
    "combined manifest order"
)
_EXPECTED_REPRESENTATION = {
    "catalog_representation": "learned_id",
    "content_gate": "fixed",
    "extra_item_id_dim": None,
    "gate_hidden_dim": None,
    "history_hidden_dim": 32,
    "history_representation": "id_content",
    "metadata": [],
    "metadata_dim": None,
}
_EXPECTED_DATASET = {
    "candidate_catalog": "full",
    "event_limit": 50_000_000,
    "exclude_seen": False,
    "minimum_user_interactions": 5,
    "sampling": "none",
    "size": "native-50m",
    "source": "likes",
    "validation_interval_seconds": 604800,
}
_TRANSFER_INVARIANTS = {
    "adaptive_schedule_early_stopping": False,
    "batch_size": 512,
    "correct_positive_logq": False,
    "dataset_size": "50m",
    "day_range": {"start_day": 0, "end_day": 300},
    "dense_random_negative_scores": True,
    "drop_unmapped_items": True,
    "eval_every_n_epochs": 1,
    "eval_ks": [10, 50, 100],
    "eval_max_users": 20000,
    "evaluation_catalog": "all",
    "event_type_filter": "like",
    "exclude_own_group_negatives": False,
    "exclude_seen_from_evaluation": False,
    "g3_dataset_size": "native-50m",
    "g3_protocol_sha256": APPROVED_PROTOCOL_SHA256,
    "g3_representation": _EXPECTED_REPRESENTATION,
    "logq_alpha": 0.01,
    "logq_correction": "yi2019",
    "lr_schedule": {
        "cycles": 1,
        "min_lr_fraction": 0.0,
        "optimizer_group_scope": "both",
        "power_exponent": -0.51,
        "power_transition_tokens": None,
        "shape": "linear",
        "timescale_fraction": None,
        "timescale_steps": None,
        "warmup_fraction": 0.0,
    },
    "mask_false_negatives": False,
    "min_item_interactions_per_item": 5,
    "negative_sampling": "random",
    "num_in_batch_negatives": 512,
    "random_negative_fraction": 0.5,
    "restore_best_weights": True,
    "selection_k": 100,
    "user_sample": None,
    "validation_interval_seconds": 604800,
    "window": "next_item",
}


def select_rq2_content_candidate(
    candidates: Sequence[Mapping[str, object]],
) -> Mapping[str, object]:
    if not candidates:
        raise ValueError("RQ2 content-horizon selection requires candidates")
    return min(
        candidates,
        key=lambda run: (
            -float(_metrics(run)["recall@100"]),
            -float(_metrics(run)["ndcg@100"]),
            float(run["queue_wall_seconds"]),
            int(run.get("combined_manifest_order", run["manifest_order"])),
        ),
    )


def assess_content_horizon_boundaries(
    selected: Mapping[str, object],
) -> dict[str, object]:
    if selected.get("capacity") != 32:
        raise ValueError("RQ2 content-horizon winner is not width 32")
    embedding = _rate_boundary(
        float(selected["embedding_learning_rate"]),
        APPROVED_PROTOCOL.embedding_lr_bounds,
    )
    deep = _rate_boundary(
        float(selected["deep_learning_rate"]),
        APPROVED_PROTOCOL.deep_lr_bounds,
    )
    horizon = int(selected["horizon_epochs"])
    best_epoch = int(selected["best_epoch"])
    extend_horizon = horizon == 40 and best_epoch == 40
    actions = []
    for group, decision in (
        ("embedding_learning_rate", embedding),
        ("deep_learning_rate", deep),
    ):
        if decision["direction"] is not None:
            actions.append(
                {
                    "action": "three_joint_outward_lr_probes",
                    "optimizer_group": group,
                    "direction": decision["direction"],
                }
            )
    if extend_horizon:
        actions.append({"action": "horizon_extension", "horizon_epochs": 60})
    return {
        "embedding_learning_rate": embedding,
        "deep_learning_rate": deep,
        "horizon": {
            "selected_epochs": horizon,
            "restored_best_epoch": best_epoch,
            "extension_required": extend_horizon,
            "extension_epochs": 60 if extend_horizon else None,
        },
        "capacity": {
            "selected": 32,
            "status": "resolved_user_approved",
            "additional_lower_capacity_authorized": False,
        },
        "extension_required": bool(actions),
        "required_actions": actions,
    }


def build_rq2_content_horizon_evidence(
    root: Path,
    *,
    batch_id: str,
) -> dict[str, object]:
    root = root.resolve(strict=True)
    if not batch_id:
        raise ValueError("RQ2 content-horizon batch id is required")
    ledger_path = root / RQ2_CONTENT_HORIZON_LEDGER_PATH
    ledger = load_rq2_content_horizon_ledger(ledger_path, root=root)
    if ledger.sha256 != APPROVED_RQ2_CONTENT_HORIZON_LEDGER_SHA256:
        raise ValueError("RQ2 content-horizon evidence received a different ledger")
    verify_rq2_content_horizon_inputs(root, ledger, full_validation=True)
    next_path = root / ledger.resolved_next_stage_evidence.path
    next_evidence = verify_rq2_next_stage_evidence(next_path, root=root)
    id_path = root / ledger.id_boundary_evidence.path
    id_evidence = verify_rq2_id_boundary_evidence(id_path, root=root)
    _validate_predecessors(ledger, next_evidence=next_evidence, id_evidence=id_evidence)
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
        )
        for index, (row, job_id) in enumerate(
            zip(ledger.rows, job_ids, strict=True)
        )
    ]
    if len(runs) != 3 or {run["horizon_epochs"] for run in runs} != {15, 25, 40}:
        raise ValueError("RQ2 content-horizon evidence lacks the exact three probes")
    probe_selected = select_rq2_content_candidate(runs)
    source = _source_content_candidate(ledger, next_evidence)
    final_candidates = [source, *runs]
    final_selected = select_rq2_content_candidate(final_candidates)
    boundary = assess_content_horizon_boundaries(final_selected)
    selection_resolved = not boundary["extension_required"]
    id_selected = _selected_id_candidate(id_evidence)
    reusable = _reusable_width_32_rows(next_evidence, runs)
    content_metrics = _metrics(final_selected)
    id_metrics = _metrics(id_selected)
    comparison = (
        {
            "content_concat": final_selected,
            "id_only_densenet": id_selected,
            "recall_at_100_delta": float(content_metrics["recall@100"])
            - float(id_metrics["recall@100"]),
            "ndcg_at_100_delta": float(content_metrics["ndcg@100"])
            - float(id_metrics["ndcg@100"]),
            "content_beats_id_only": (
                float(content_metrics["recall@100"])
                > float(id_metrics["recall@100"])
            ),
        }
        if selection_resolved
        else None
    )
    rq3_inputs = (
        {
            "selected_content_input": final_selected,
            "id_only_control": id_selected,
            "reusable_width_32_content_rows": reusable,
        }
        if selection_resolved
        else None
    )
    payload = {
        "schema_version": 1,
        "kind": "g3_rq2_content_width32_horizon_evidence",
        "protocol_sha256": APPROVED_PROTOCOL_SHA256,
        "content_horizon_ledger": _file_fact(root, ledger_path)
        | {"logical_sha256": ledger.sha256},
        "resolved_next_stage_evidence": _file_fact(root, next_path)
        | {"logical_sha256": next_evidence["sha256"]},
        "id_boundary_evidence": _file_fact(root, id_path)
        | {"logical_sha256": id_evidence["sha256"]},
        "queue_batch": _file_fact(root, batch_path) | {"batch_id": batch_id},
        "ranking_context": _file_fact(root, context_path),
        "tuning_ledger": runs,
        "horizon_probe_selection": {
            "selection_rule": _SELECTION_RULE,
            "selected": probe_selected,
        },
        "final_content_selection": {
            "status": "resolved" if selection_resolved else "pending_boundary_followup",
            "selection_rule": _SELECTION_RULE,
            "candidate_row_ids": [run["row_id"] for run in final_candidates],
            "selected": final_selected if selection_resolved else None,
            "provisional_selected": None if selection_resolved else final_selected,
            "boundary_decision": boundary,
        },
        "final_rq2_comparison": comparison,
        "rq3_inputs": rq3_inputs,
        "opportunity_accounting": ledger.opportunity_accounting,
    }
    return _document(payload)


def persist_rq2_content_horizon_evidence(
    path: Path, document: Mapping[str, object], *, root: Path
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
                f"immutable RQ2 content-horizon evidence differs: {path}"
            )
    return path


def load_rq2_content_horizon_evidence(
    path: Path, *, root: Path
) -> dict[str, object]:
    return verify_rq2_content_horizon_evidence(path, root=root)


def verify_rq2_content_horizon_evidence(
    path: Path, *, root: Path
) -> dict[str, object]:
    evidence = _validate_document(_load_json(path))
    _authenticate_document(evidence, root=root)
    return evidence


def _authenticate_document(
    evidence: Mapping[str, object], *, root: Path
) -> None:
    batch = evidence.get("queue_batch")
    if not isinstance(batch, dict) or not isinstance(batch.get("batch_id"), str):
        raise ValueError("RQ2 content-horizon evidence has no bound batch")
    rebuilt = build_rq2_content_horizon_evidence(
        root,
        batch_id=batch["batch_id"],
    )
    if _canonical_json(evidence) != _canonical_json(rebuilt):
        raise ValueError("RQ2 content-horizon evidence differs from bound artifacts")


def _validate_batch(
    batch: Mapping[str, object], *, batch_id: str, expected_jobs: int
) -> list[str]:
    expected_keys = {"id", "jobs", "sealed", "sealed_at", "submitted_at"}
    jobs = batch.get("jobs")
    if (
        set(batch) != expected_keys
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
        raise ValueError(
            "RQ2 content-horizon queue batch is not the exact sealed batch"
        )
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
) -> dict[str, object]:
    completed_path = (
        root / "generated/training-queue-service/completed" / f"{job_id}.json"
    )
    queue_job = _load_json(completed_path)
    expected_script = (
        root
        / "experiments/g3_pretrained_item_embeddings/launchers/"
        "run_rq2_content_horizon.py"
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
        raise ValueError(f"RQ2 content-horizon completion differs for {row['id']}")
    verify_unique_completed_run(
        root / "generated/training-queue-service",
        run_name=str(row["run_name"]),
        expected_job_id=job_id,
    )
    environment = queue_job.get("environment")
    if not isinstance(environment, list) or len(environment) != 3:
        raise ValueError(f"RQ2 content-horizon environment differs for {row['id']}")
    pairs = [value.split("=", 1) for value in environment if "=" in value]
    if len(pairs) != 3 or len({name for name, _ in pairs}) != 3:
        raise ValueError(f"RQ2 content-horizon environment differs for {row['id']}")
    values = dict(pairs)
    if (
        set(values) != {"WANDB_MODE", JOB_ENVIRONMENT, LEDGER_ENVIRONMENT}
        or values["WANDB_MODE"] != "offline"
        or Path(values[LEDGER_ENVIRONMENT]).resolve() != ledger_path
    ):
        raise ValueError(f"RQ2 content-horizon environment differs for {row['id']}")
    compiled = decode_control_job(values[JOB_ENVIRONMENT], ledger)
    if compiled.row_id != row["id"] or compiled.job != row:
        raise ValueError(f"RQ2 content-horizon queue payload differs for {row['id']}")
    run_directory = root / "generated/logs" / str(row["run_name"])
    contract = _load_json(run_directory / "g3_rq2_content_horizon_job.json")
    if contract != compiled.to_dict() | {
        "ledger_path": str(ledger_path),
        "ledger_sha256": ledger.sha256,
    }:
        raise ValueError(f"RQ2 content-horizon contract differs for {row['id']}")
    training = row.get("training")
    representation = row.get("representation")
    if (
        row.get("family_id") != "rq2_content_concat"
        or row.get("phase") != "selected_width_horizon_followup"
        or row.get("stage") != "rq2_content_width32_horizon"
        or row.get("role") != "horizon_probe"
        or row.get("reused_from") is not None
        or row.get("dataset") != _EXPECTED_DATASET
        or not isinstance(training, dict)
        or not isinstance(representation, dict)
        or representation
        != {
            "id": "rq2_content_concat",
            "history": "learned_item_id_plus_frozen_content",
            "catalog": "learned_item_id",
            "history_hidden_dim": 32,
            "separate_history_catalog_tables": True,
            "content_trainable": False,
            "content_width": 128,
        }
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
        or training["validate_every_epoch"] is not True
        or training["restore_best_validation_epoch"] is not True
    ):
        raise ValueError(f"RQ2 content-horizon row identity differs for {row['id']}")
    horizon = int(training["horizon_epochs"])
    metadata = _load_json(run_directory / "training_metadata.json")
    expected_metadata = {
        "batch_size": 512,
        "seed": 42,
        "embedding_learning_rate": training["embedding_learning_rate"],
        "deep_learning_rate": training["deep_learning_rate"],
        "lr_schedule_horizon_epochs": horizon,
        "epochs_trained": horizon,
        "stopped_epoch": horizon,
        "lr_horizon_complete": True,
        "selection_resolved": True,
        "early_stopped": False,
        "g3_dataset_size": "native-50m",
        "g3_protocol_sha256": APPROVED_PROTOCOL_SHA256,
        "training_semantics_revision": 2,
    }
    if any(metadata.get(name) != value for name, value in expected_metadata.items()):
        raise ValueError(f"RQ2 content-horizon metadata differs for {row['id']}")
    if metadata.get("g3_representation") != _EXPECTED_REPRESENTATION:
        raise ValueError(f"RQ2 content-horizon representation differs for {row['id']}")
    invariants = metadata.get("transfer_invariants")
    expected_invariants = _TRANSFER_INVARIANTS | {
        "lr_schedule_horizon_epochs": horizon
    }
    if not isinstance(invariants, dict) or any(
        invariants.get(name) != value for name, value in expected_invariants.items()
    ):
        raise ValueError(f"RQ2 content-horizon invariants differ for {row['id']}")
    best_epoch = metadata.get("best_epoch")
    if type(best_epoch) is not int or not 1 <= best_epoch <= horizon:
        raise ValueError(f"RQ2 content-horizon best epoch differs for {row['id']}")
    traces = metadata.get("lr_group_traces")
    if not isinstance(traces, dict) or set(traces) != {"embedding", "deep"}:
        raise ValueError(f"RQ2 content-horizon schedule differs for {row['id']}")
    for trace in traces.values():
        if (
            not isinstance(trace, list)
            or len(trace) != horizon
            or any(not _finite_number(value) for value in trace)
            or float(trace[-1]) != 0.0
        ):
            raise ValueError(f"RQ2 content-horizon schedule differs for {row['id']}")
    metrics_path = run_directory / "final_metrics.json"
    metrics = _load_json(metrics_path)
    if set(metrics) != {*_METRIC_NAMES, "num_users"} or any(
        not _finite_number(value) for value in metrics.values()
    ):
        raise ValueError(f"RQ2 content-horizon metrics differ for {row['id']}")
    ranking_path = run_directory / "ranking_evidence.pt"
    rankings_path = run_directory / "top_item_rankings.json"
    recomputed = _recompute_metrics(context_path, ranking_path, rankings_path)
    if set(recomputed) != set(metrics) or any(
        abs(float(metrics[name]) - float(recomputed[name])) > 1e-15
        for name in metrics
    ):
        raise ValueError(f"RQ2 content-horizon metrics differ for {row['id']}")
    diagnostics = _load_json(run_directory / "g3_training_diagnostics.json")
    nonfinite = _values_named(diagnostics, "nonfinite_count")
    if not nonfinite or any(value != 0 for value in nonfinite):
        raise ValueError(f"RQ2 content-horizon diagnostics differ for {row['id']}")
    artifact_filenames = (
        ("job_contract", "g3_rq2_content_horizon_job.json"),
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
        "capacity": representation["history_hidden_dim"],
        "embedding_learning_rate": training["embedding_learning_rate"],
        "deep_learning_rate": training["deep_learning_rate"],
        "horizon_epochs": horizon,
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
    next_evidence: Mapping[str, object],
    id_evidence: Mapping[str, object],
) -> None:
    if (
        next_evidence.get("sha256") != ledger.resolved_next_stage_evidence.sha256
        or id_evidence.get("sha256") != ledger.id_boundary_evidence.sha256
        or id_evidence.get("content_capacity_status")
        != {
            "status": "deferred_pending_user_approval",
            "changed_by_this_evidence": False,
        }
    ):
        raise ValueError("RQ2 content-horizon predecessor evidence changed")


def _source_content_candidate(
    ledger: Any, next_evidence: Mapping[str, object]
) -> dict[str, object]:
    decision = next_evidence.get("content_capacity_decision")
    if not isinstance(decision, dict):
        raise ValueError("RQ2 content-horizon source selection is absent")
    selected = decision.get("extension_selected")
    if not isinstance(selected, dict):
        raise ValueError("RQ2 content-horizon source selection is absent")
    expected = ledger.source_selection
    if any(
        selected.get(name) != expected_value
        for name, expected_value in expected.items()
        if name != "metrics"
    ) or any(
        float(_metrics(selected)[name]) != float(expected_value)
        for name, expected_value in expected["metrics"].items()
    ):
        raise ValueError("RQ2 content-horizon source selection changed")
    return dict(selected) | {"combined_manifest_order": 0}


def _selected_id_candidate(id_evidence: Mapping[str, object]) -> Mapping[str, object]:
    final = id_evidence.get("final_selection")
    if not isinstance(final, dict) or final.get("boundary_decision") != {
        "status": "resolved",
        "outward_winner_on_new_boundary": False,
        "additional_runs_authorized": False,
        "next_action": "none",
    }:
        raise ValueError("RQ2 ID-only boundary selection is unresolved")
    selected = final.get("selected")
    if (
        not isinstance(selected, dict)
        or selected.get("row_id") != "rq2_id_only_densenet:12"
        or selected.get("family_id") != "rq2_id_only_densenet"
        or selected.get("capacity") != 255
    ):
        raise ValueError("RQ2 ID-only comparison winner changed")
    _metrics(selected)
    return selected


def _reusable_width_32_rows(
    next_evidence: Mapping[str, object], current: Sequence[Mapping[str, object]]
) -> list[Mapping[str, object]]:
    previous = next_evidence.get("tuning_ledger")
    if not isinstance(previous, list):
        raise ValueError("RQ2 predecessor tuning ledger is absent")
    rows = [
        row
        for row in previous
        if isinstance(row, dict)
        and row.get("family_id") == "rq2_content_concat"
        and row.get("capacity") == 32
    ]
    if len(rows) != 3:
        raise ValueError("RQ2 predecessor lacks three reusable width-32 rows")
    combined = [*rows, *current]
    identities = [(row.get("row_id"), row.get("run_name")) for row in combined]
    if len(combined) != 6 or len(set(identities)) != 6:
        raise ValueError("RQ3 reusable width-32 content rows are not unique")
    return combined


def _rate_boundary(value: float, bounds: tuple[float, float]) -> dict[str, object]:
    lower, upper = bounds
    if not lower <= value <= upper:
        raise ValueError(
            "selected RQ2 content learning rate is outside approved bounds"
        )
    position = (value - lower) / (upper - lower)
    return {
        "selected": value,
        "bounds": list(bounds),
        "normalized_position": position,
        "direction": "lower" if position < 0.1 else "upper" if position > 0.9 else None,
    }


def _metrics(run: Mapping[str, object]) -> Mapping[str, object]:
    metrics = run.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("RQ2 content-horizon run has no metrics")
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
    document["sha256"] = _canonical_sha256(document)
    return document


def _validate_document(document: dict[str, object]) -> dict[str, object]:
    expected_keys = {
        "schema_version",
        "kind",
        "protocol_sha256",
        "content_horizon_ledger",
        "resolved_next_stage_evidence",
        "id_boundary_evidence",
        "queue_batch",
        "ranking_context",
        "tuning_ledger",
        "horizon_probe_selection",
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
        or document.get("kind") != "g3_rq2_content_width32_horizon_evidence"
        or document.get("protocol_sha256") != APPROVED_PROTOCOL_SHA256
        or document.get("sha256") != _canonical_sha256(payload)
    ):
        raise ValueError("RQ2 content-horizon evidence identity or hash is invalid")
    return document


def _canonical_sha256(value: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


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
    document = build_rq2_content_horizon_evidence(
        root,
        batch_id=arguments.batch_id,
    )
    path = root / RQ2_CONTENT_HORIZON_EVIDENCE_PATH
    if arguments.write:
        persist_rq2_content_horizon_evidence(path, document, root=root)
    selection = document["final_content_selection"]
    selected = selection["selected"] or selection["provisional_selected"]
    print(
        json.dumps(
            {
                "path": str(path),
                "sha256": document["sha256"],
                "runs": len(document["tuning_ledger"]),
                "selection_status": selection["status"],
                "selected_row_id": selected["row_id"],
                "status": "materialized" if arguments.write else "preview",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
