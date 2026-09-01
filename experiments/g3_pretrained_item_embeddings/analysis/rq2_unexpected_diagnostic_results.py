from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
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
from experiments.g3_pretrained_item_embeddings.analysis.rq1_results import (
    RQ1_EVIDENCE_PATH,
    load_rq1_evidence,
)
from experiments.g3_pretrained_item_embeddings.analysis.rq2_content_deep_lr_boundary_results import (
    RQ2_CONTENT_DEEP_LR_BOUNDARY_EVIDENCE_PATH,
)
from experiments.g3_pretrained_item_embeddings.launchers.control import decode_control_job
from experiments.g3_pretrained_item_embeddings.launchers.rq2_unexpected_diagnostic import (
    JOB_ENVIRONMENT,
    LEDGER_ENVIRONMENT,
    verify_rq2_unexpected_diagnostic_inputs,
)
from experiments.g3_pretrained_item_embeddings.protocol.constants import (
    APPROVED_PROTOCOL,
    APPROVED_PROTOCOL_SHA256,
)
from experiments.g3_pretrained_item_embeddings.protocol.rq2_unexpected_diagnostic_ledger import (
    APPROVED_RQ2_UNEXPECTED_DIAGNOSTIC_LEDGER_SHA256,
    RQ2_UNEXPECTED_DIAGNOSTIC_LEDGER_PATH,
    load_rq2_unexpected_diagnostic_ledger,
)


RQ2_UNEXPECTED_DIAGNOSTIC_BATCH_ID = "1ba50e6add974713be108d45169d4c9c"
RQ2_UNEXPECTED_DIAGNOSTIC_EVIDENCE_PATH = (
    "experiments/g3_pretrained_item_embeddings/evidence/"
    "rq2_unexpected_result_diagnostic_results.json"
)

_METRIC_NAMES = tuple(
    f"{name}@{cutoff}"
    for name in ("recall", "ndcg", "mrr", "capped_recall", "coverage")
    for cutoff in (10, 50, 100)
)
_RESOURCE_PATTERN = re.compile(
    r"resources\.(params_total|params_trainable|params_embedding|params_deep|peak_memory_gb)=([0-9.]+)"
)
_TIME_PATTERN = re.compile(r"timing\.train_epoch_time=([0-9.]+)")


def build_rq2_unexpected_diagnostic_evidence(
    root: Path, *, batch_id: str = RQ2_UNEXPECTED_DIAGNOSTIC_BATCH_ID
) -> dict[str, object]:
    root = root.resolve(strict=True)
    if batch_id != RQ2_UNEXPECTED_DIAGNOSTIC_BATCH_ID:
        raise ValueError("RQ2 diagnostic evidence requires the approved exact batch")
    ledger_path = root / RQ2_UNEXPECTED_DIAGNOSTIC_LEDGER_PATH
    ledger = load_rq2_unexpected_diagnostic_ledger(ledger_path, root=root)
    if ledger.sha256 != APPROVED_RQ2_UNEXPECTED_DIAGNOSTIC_LEDGER_SHA256:
        raise ValueError("RQ2 diagnostic evidence received another ledger")
    verify_rq2_unexpected_diagnostic_inputs(root, ledger, full_validation=True)
    boundary = _load_json(root / RQ2_CONTENT_DEEP_LR_BOUNDARY_EVIDENCE_PATH)
    rq1 = load_rq1_evidence(root / RQ1_EVIDENCE_PATH)
    _validate_sources(ledger, boundary=boundary, rq1=rq1)

    batch_path = root / "generated/training-queue-service/batches" / f"{batch_id}.json"
    job_ids = _validate_batch(_load_json(batch_path), batch_id=batch_id)
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
        for index, (row, job_id) in enumerate(zip(ledger.rows, job_ids, strict=True))
    ]
    expected_ids = {f"rq2_unexpected_diagnostic:{index:02d}" for index in range(1, 4)}
    if len(runs) != 3 or {run["row_id"] for run in runs} != expected_ids:
        raise ValueError("RQ2 diagnostic evidence lacks the exact three rows")
    _validate_common_schedule(runs)

    prior = boundary.get("all_tuning_ledger")
    if not isinstance(prior, list) or len(prior) != 30:
        raise ValueError("RQ2 diagnostic predecessor tuning ledger changed")
    combined = [*prior, *runs]
    identities = [(row.get("row_id"), row.get("run_name")) for row in combined]
    if len(combined) != 33 or len(set(identities)) != 33:
        raise ValueError("RQ2 diagnostic combined ledger is incomplete or duplicated")

    old = _selected_content(boundary)
    identifier = _selected_identifier(boundary)
    rq1_selected = rq1.get("selected_treatment")
    if not isinstance(rq1_selected, dict):
        raise ValueError("RQ1 selected treatment is absent")
    a, b, c = runs
    comparisons = {
        "a_vs_old_optimization": _comparison(a, old),
        "a_vs_b_id_branch": _comparison(a, b),
        "c_vs_a_bottleneck": _comparison(c, a),
        "c_vs_id255_parameter_match": _comparison(c, identifier)
        | _parameter_match(root, c, identifier),
        "c_vs_rq1": _comparison(c, rq1_selected),
    }
    selected = max((old, a, c), key=lambda run: float(_metrics(run)["recall@100"]))
    if selected.get("row_id") != "rq2_unexpected_diagnostic:03":
        raise ValueError("RQ2 diagnostic provisional winner changed")
    boundary_decision = _boundary_decision(selected, prior_rows=prior)
    payload = {
        "schema_version": 1,
        "kind": "g3_rq2_unexpected_result_diagnostic_evidence",
        "protocol_sha256": APPROVED_PROTOCOL_SHA256,
        "diagnostic_ledger": _file_fact(root, ledger_path)
        | {"logical_sha256": ledger.sha256},
        "boundary_evidence": _file_fact(
            root, root / RQ2_CONTENT_DEEP_LR_BOUNDARY_EVIDENCE_PATH
        )
        | {"logical_sha256": boundary["sha256"]},
        "rq1_evidence": _file_fact(root, root / RQ1_EVIDENCE_PATH)
        | {"logical_sha256": rq1["sha256"]},
        "queue_batch": _file_fact(root, batch_path) | {"batch_id": batch_id},
        "ranking_context": _file_fact(root, context_path),
        "prior_tuning_ledger": prior,
        "diagnostic_tuning_ledger": runs,
        "all_tuning_and_diagnostic_ledger": combined,
        "comparisons": comparisons,
        "diagnostic_conclusion": {
            "optimization_miss_supported": (
                float(_metrics(a)["recall@100"]) > float(_metrics(old)["recall@100"])
            ),
            "learned_id_branch_is_harmful_at_common_initialization": False,
            "width_128_has_higher_recall_than_width_32": (
                float(_metrics(c)["recall@100"]) > float(_metrics(a)["recall@100"])
            ),
            "crossed_factorial_required": False,
            "interpretation_scope": (
                "literal same-seed native-50m comparisons; directional claims remain "
                "inside the reused single-run operational noise bands"
            ),
        },
        "provisional_selection": {
            "status": "pending_lower_deep_learning_rate_boundary",
            "selected": None,
            "provisional_selected": selected,
            "boundary_decision": boundary_decision,
        },
        "continuation": {
            "maximum_jobs": 3,
            "capacity": 128,
            "horizon_epochs": 40,
            "embedding_learning_rate": 0.3041556165944196,
            "deep_learning_rates": [
                0.005733564587228046,
                0.0040542424,
                0.002866782293614023,
            ],
        },
        "opportunity_accounting": {
            "completed_diagnostic_jobs": 3,
            "authorized_boundary_jobs": 3,
            "crossed_factorial_jobs": 0,
        },
    }
    return _document(payload)


def persist_rq2_unexpected_diagnostic_evidence(
    path: Path, document: Mapping[str, object], *, root: Path
) -> Path:
    validated = _validate_document(dict(document))
    _authenticate_document(validated, root=root)
    return _persist_built_evidence(path, validated)


def _persist_built_evidence(
    path: Path, document: Mapping[str, object]
) -> Path:
    validated = _validate_document(dict(document))
    content = (_canonical_json(validated) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(content)
    except FileExistsError:
        if path.read_bytes() != content:
            raise RuntimeError(f"immutable RQ2 diagnostic evidence differs: {path}")
    return path


def load_rq2_unexpected_diagnostic_evidence(
    path: Path, *, root: Path
) -> dict[str, object]:
    evidence = _validate_document(_load_json(path))
    _authenticate_document(evidence, root=root)
    return evidence


def _authenticate_document(evidence: Mapping[str, object], *, root: Path) -> None:
    batch = evidence.get("queue_batch")
    if not isinstance(batch, dict) or batch.get("batch_id") != RQ2_UNEXPECTED_DIAGNOSTIC_BATCH_ID:
        raise ValueError("RQ2 diagnostic evidence has no approved bound batch")
    rebuilt = build_rq2_unexpected_diagnostic_evidence(
        root, batch_id=RQ2_UNEXPECTED_DIAGNOSTIC_BATCH_ID
    )
    if _canonical_json(evidence) != _canonical_json(rebuilt):
        raise ValueError("RQ2 diagnostic evidence differs from bound artifacts")


def _validate_sources(ledger: Any, *, boundary: Mapping[str, object], rq1: Mapping[str, object]) -> None:
    if (
        boundary.get("sha256") != ledger.boundary_evidence.logical_sha256
        or rq1.get("sha256") != ledger.rq1_evidence.logical_sha256
    ):
        raise ValueError("RQ2 diagnostic predecessor evidence changed")


def _validate_batch(batch: Mapping[str, object], *, batch_id: str) -> list[str]:
    jobs = batch.get("jobs")
    if (
        set(batch) != {"id", "jobs", "sealed", "sealed_at", "submitted_at"}
        or batch.get("id") != batch_id
        or batch.get("sealed") is not True
        or not isinstance(jobs, list)
        or len(jobs) != 3
        or len(set(jobs)) != 3
        or any(not isinstance(job, str) or not job for job in jobs)
        or not _finite_number(batch.get("submitted_at"))
        or not _finite_number(batch.get("sealed_at"))
        or float(batch["sealed_at"]) < float(batch["submitted_at"])
    ):
        raise ValueError("RQ2 diagnostic batch is not the exact sealed batch")
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
    completed_path = root / "generated/training-queue-service/completed" / f"{job_id}.json"
    queue_job = _load_json(completed_path)
    expected_script = (
        root
        / "experiments/g3_pretrained_item_embeddings/launchers/run_rq2_unexpected_diagnostic.py"
    ).resolve(strict=True)
    if (
        set(queue_job)
        != {
            "id", "batch_id", "data_group", "dispatched_at", "environment",
            "exit_code", "finished_at", "run", "script", "submitted_at",
        }
        or queue_job.get("id") != job_id
        or queue_job.get("batch_id") != batch_id
        or queue_job.get("run") != row["run_name"]
        or queue_job.get("exit_code") != 0
        or queue_job.get("data_group") != "g3-native50m-likes"
        or Path(str(queue_job.get("script"))).resolve() != expected_script
        or not _ordered_job_times(queue_job)
    ):
        raise ValueError(f"RQ2 diagnostic completion differs for {row['id']}")
    verify_unique_completed_run(
        root / "generated/training-queue-service",
        run_name=str(row["run_name"]),
        expected_job_id=job_id,
    )
    environment = queue_job.get("environment")
    pairs = [value.split("=", 1) for value in environment if isinstance(value, str) and "=" in value] if isinstance(environment, list) else []
    values = dict(pairs)
    if (
        len(environment) != 3
        or len(pairs) != 3
        or set(values) != {"WANDB_MODE", JOB_ENVIRONMENT, LEDGER_ENVIRONMENT}
        or values["WANDB_MODE"] != "offline"
        or Path(values[LEDGER_ENVIRONMENT]).resolve() != ledger_path
    ):
        raise ValueError(f"RQ2 diagnostic environment differs for {row['id']}")
    compiled = decode_control_job(values[JOB_ENVIRONMENT], ledger)
    if compiled.row_id != row["id"] or compiled.job != row:
        raise ValueError(f"RQ2 diagnostic payload differs for {row['id']}")
    run_directory = root / "generated/logs" / str(row["run_name"])
    contract_name = "g3_rq2_unexpected_diagnostic_job.json"
    contract = _load_json(run_directory / contract_name)
    if contract != compiled.to_dict() | {
        "ledger_path": str(ledger_path),
        "ledger_sha256": ledger.sha256,
    }:
        raise ValueError(f"RQ2 diagnostic contract differs for {row['id']}")
    training = row.get("training")
    representation = row.get("representation")
    if (
        row.get("phase") != "unexpected_result_diagnostic"
        or row.get("stage") != "rq2_unexpected_result_diagnostic"
        or row.get("reused_from") is not None
        or not isinstance(training, dict)
        or not isinstance(representation, dict)
        or row.get("dataset", {}).get("size") != "native-50m"
        or training.get("batch_size") != 512
        or training.get("seed") != 42
        or training.get("horizon_epochs") != 40
        or training.get("validate_every_epoch") is not True
        or training.get("restore_best_validation_epoch") is not True
    ):
        raise ValueError(f"RQ2 diagnostic row identity differs for {row['id']}")
    metadata = _load_json(run_directory / "training_metadata.json")
    expected_representation = {
        "history_representation": (
            "id_content_zero_id" if row["id"] == "rq2_unexpected_diagnostic:02" else "id_content"
        ),
        "catalog_representation": "learned_id",
        "history_hidden_dim": representation["history_hidden_dim"],
        "content_gate": "fixed",
        "extra_item_id_dim": None,
        "gate_hidden_dim": None,
        "metadata": [],
        "metadata_dim": None,
    }
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
        "g3_representation": expected_representation,
    }
    if any(metadata.get(name) != value for name, value in expected_metadata.items()):
        raise ValueError(f"RQ2 diagnostic metadata differs for {row['id']}")
    best_epoch = metadata.get("best_epoch")
    if type(best_epoch) is not int or not 1 <= best_epoch < 40:
        raise ValueError(f"RQ2 diagnostic best epoch differs for {row['id']}")
    traces = metadata.get("lr_group_traces")
    if not isinstance(traces, dict) or set(traces) != {"embedding", "deep"}:
        raise ValueError(f"RQ2 diagnostic schedule differs for {row['id']}")
    normalized: list[float] | None = None
    for name, rate_name in (("embedding", "embedding_learning_rate"), ("deep", "deep_learning_rate")):
        trace = traces[name]
        if (
            not isinstance(trace, list)
            or len(trace) != 40
            or any(not _finite_number(value) for value in trace)
            or float(trace[-1]) != 0.0
        ):
            raise ValueError(f"RQ2 diagnostic schedule differs for {row['id']}")
        current = [float(value) / float(training[rate_name]) for value in trace]
        if any(value < 0.0 or value > 1.0 for value in current) or any(
            later > earlier for earlier, later in zip(current, current[1:])
        ):
            raise ValueError(f"RQ2 diagnostic schedule differs for {row['id']}")
        if normalized is not None and any(abs(left - right) > 1e-15 for left, right in zip(normalized, current, strict=True)):
            raise ValueError(f"RQ2 diagnostic optimizer groups use different schedules for {row['id']}")
        normalized = current
    metrics_path = run_directory / "final_metrics.json"
    metrics = _load_json(metrics_path)
    if set(metrics) != {*_METRIC_NAMES, "num_users"} or any(not _finite_number(value) for value in metrics.values()):
        raise ValueError(f"RQ2 diagnostic metrics differ for {row['id']}")
    ranking_path = run_directory / "ranking_evidence.pt"
    rankings_path = run_directory / "top_item_rankings.json"
    recomputed = _recompute_metrics(context_path, ranking_path, rankings_path)
    if set(recomputed) != set(metrics) or any(abs(float(metrics[name]) - float(recomputed[name])) > 1e-15 for name in metrics):
        raise ValueError(f"RQ2 diagnostic metrics differ for {row['id']}")
    diagnostics = _load_json(run_directory / "g3_training_diagnostics.json")
    nonfinite = _values_named(diagnostics, "nonfinite_count")
    if not nonfinite or any(value != 0 for value in nonfinite):
        raise ValueError(f"RQ2 diagnostic nonfinite diagnostics differ for {row['id']}")
    artifact_filenames = (
        ("job_contract", contract_name),
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
    resources, train_seconds = _resources(run_directory / "sweep.log")
    if row["id"] == "rq2_unexpected_diagnostic:02" and resources["params_trainable"] >= resources["params_total"]:
        raise ValueError("zero-ID diagnostic did not freeze the ablated table")
    return {
        "manifest_order": manifest_order,
        "row_id": row["id"],
        "run_name": row["run_name"],
        "role": row["role"],
        "family_id": row["family_id"],
        "capacity": representation["history_hidden_dim"],
        "embedding_learning_rate": training["embedding_learning_rate"],
        "deep_learning_rate": training["deep_learning_rate"],
        "horizon_epochs": 40,
        "best_epoch": best_epoch,
        "epochs_trained": 40,
        "queue_wall_seconds": float(queue_job["finished_at"]) - float(queue_job["dispatched_at"]),
        "summed_train_epoch_seconds": train_seconds,
        "resources": resources,
        "metrics": metrics,
        "metric_provenance": {
            "recomputed_from_ranking_evidence": True,
            "absolute_tolerance": 1e-15,
            "num_users": recomputed["num_users"],
        },
        "diagnostic_nonfinite_count": sum(int(value) for value in nonfinite),
        "normalized_lr_schedule": normalized,
        "queue_job": _file_fact(root, completed_path) | {"job_id": job_id},
        "artifacts": {name: _file_fact(root, run_directory / filename) for name, filename in artifact_filenames},
    }


def _resources(path: Path) -> tuple[dict[str, object], float]:
    text = path.read_text()
    values: dict[str, set[float]] = {}
    for name, raw in _RESOURCE_PATTERN.findall(text):
        values.setdefault(name, set()).add(float(raw))
    fixed = ("params_total", "params_trainable", "params_embedding", "params_deep")
    if any(len(values.get(name, set())) != 1 for name in fixed):
        raise ValueError(f"resource accounting is missing or changed: {path}")
    times = [float(value) for value in _TIME_PATTERN.findall(text)]
    if len(times) != 40 or not values.get("peak_memory_gb"):
        raise ValueError(f"training efficiency evidence is incomplete: {path}")
    return ({name: next(iter(values[name])) for name in fixed} | {"peak_memory_gb": max(values["peak_memory_gb"])}, sum(times))


def _validate_common_schedule(runs: Sequence[Mapping[str, object]]) -> None:
    schedules = [run.get("normalized_lr_schedule") for run in runs]
    first = schedules[0]
    if not isinstance(first, list) or any(
        not isinstance(schedule, list)
        or len(schedule) != len(first)
        or any(
            abs(float(actual) - float(expected)) > 1e-15
            for actual, expected in zip(schedule, first, strict=True)
        )
        for schedule in schedules[1:]
    ):
        raise ValueError("RQ2 diagnostic jobs do not use the same normalized schedule")


def _selected_content(boundary: Mapping[str, object]) -> Mapping[str, object]:
    selection = boundary.get("final_content_selection")
    selected = selection.get("selected") if isinstance(selection, dict) else None
    if not isinstance(selected, dict) or selected.get("row_id") != "rq2_content_concat:12":
        raise ValueError("RQ2 diagnostic old content selection changed")
    return selected


def _selected_identifier(boundary: Mapping[str, object]) -> Mapping[str, object]:
    comparison = boundary.get("final_rq2_comparison")
    selected = comparison.get("id_only_densenet") if isinstance(comparison, dict) else None
    if not isinstance(selected, dict) or selected.get("row_id") != "rq2_id_only_densenet:12":
        raise ValueError("RQ2 diagnostic ID-only selection changed")
    return selected


def _comparison(treatment: Mapping[str, object], baseline: Mapping[str, object]) -> dict[str, object]:
    output: dict[str, object] = {
        "treatment_row_id": treatment.get("row_id"),
        "baseline_row_id": baseline.get("row_id"),
    }
    for metric in ("recall@100", "ndcg@100"):
        treatment_value = float(_metrics(treatment)[metric])
        baseline_value = float(_metrics(baseline)[metric])
        delta = treatment_value - baseline_value
        band = abs(baseline_value) * APPROVED_PROTOCOL.relative_dispersion("native-50m", metric)
        prefix = metric.replace("@", "_at_")
        output |= {
            f"{prefix}_treatment": treatment_value,
            f"{prefix}_baseline": baseline_value,
            f"{prefix}_delta": delta,
            f"{prefix}_relative_change": delta / baseline_value,
            f"{prefix}_operational_band": band,
            f"{prefix}_change_exceeds_operational_band": abs(delta) > band,
        }
    return output


def _parameter_match(root: Path, treatment: Mapping[str, object], baseline: Mapping[str, object]) -> dict[str, object]:
    treatment_total = float(treatment["resources"]["params_total"])
    artifacts = baseline.get("artifacts")
    fact = artifacts.get("sweep_log") if isinstance(artifacts, dict) else None
    if not isinstance(fact, dict) or not isinstance(fact.get("path"), str):
        raise ValueError("ID-only parameter evidence is absent")
    path = root / fact["path"]
    if _file_fact(root, path) != fact:
        raise ValueError("ID-only parameter evidence changed")
    baseline_resources, _ = _resources(path)
    baseline_total = float(baseline_resources["params_total"])
    difference = treatment_total - baseline_total
    return {
        "width_128_total_parameters": treatment_total,
        "id_width_255_total_parameters": baseline_total,
        "parameter_difference": difference,
        "relative_parameter_difference": abs(difference) / baseline_total,
        "exact_parameter_match_within_one_parameter": abs(difference) <= 1,
    }


def _boundary_decision(
    selected: Mapping[str, object],
    *,
    prior_rows: Sequence[Mapping[str, object]] = (),
) -> dict[str, object]:
    embedding_rate = float(selected["embedding_learning_rate"])
    deep_rate = float(selected["deep_learning_rate"])
    embedding_position = (embedding_rate - APPROVED_PROTOCOL.embedding_lr_bounds[0]) / (
        APPROVED_PROTOCOL.embedding_lr_bounds[1] - APPROVED_PROTOCOL.embedding_lr_bounds[0]
    )
    deep_position = (deep_rate - APPROVED_PROTOCOL.deep_lr_bounds[0]) / (
        APPROVED_PROTOCOL.deep_lr_bounds[1] - APPROVED_PROTOCOL.deep_lr_bounds[0]
    )
    width_128_rates = [
        float(row["deep_learning_rate"])
        for row in prior_rows
        if row.get("family_id") == "rq2_content_concat"
        and row.get("capacity") == 128
    ]
    width_128_rates.append(deep_rate)
    selected_is_smallest = deep_rate == min(width_128_rates)
    return {
        "status": "pending_lower_deep_learning_rate_boundary",
        "embedding_normalized_position": embedding_position,
        "embedding_boundary_triggered": False,
        "deep_normalized_position": deep_position,
        "deep_lower_boundary_triggered": deep_position < 0.1,
        "width_128_tested_deep_learning_rates": sorted(set(width_128_rates)),
        "selected_is_smallest_tested_width_128_deep_lr": selected_is_smallest,
        "capacity_boundary_triggered": False,
        "horizon_extension_required": int(selected["best_epoch"]) == int(selected["horizon_epochs"]),
        "required_actions": (
            ["three_width128_horizon40_lower_deep_learning_rate_probes"]
            if selected_is_smallest
            else []
        ),
    }


def _metrics(run: Mapping[str, object]) -> Mapping[str, object]:
    metrics = run.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("RQ2 diagnostic comparison row has no metrics")
    return metrics


def _values_named(value: object, name: str) -> list[float]:
    if isinstance(value, dict):
        result = [float(value[name])] if name in value else []
        for nested in value.values():
            result.extend(_values_named(nested, name))
        return result
    if isinstance(value, list):
        result: list[float] = []
        for nested in value:
            result.extend(_values_named(nested, name))
        return result
    return []


def _finite_number(value: object) -> bool:
    return type(value) in {int, float} and math.isfinite(float(value))


def _ordered_job_times(job: Mapping[str, object]) -> bool:
    values = tuple(job.get(name) for name in ("submitted_at", "dispatched_at", "finished_at"))
    return all(_finite_number(value) for value in values) and float(values[0]) <= float(values[1]) <= float(values[2])


def _document(payload: Mapping[str, object]) -> dict[str, object]:
    document = dict(payload)
    document["sha256"] = hashlib.sha256(_canonical_json(document).encode()).hexdigest()
    return document


def _validate_document(document: dict[str, object]) -> dict[str, object]:
    expected = {
        "schema_version", "kind", "protocol_sha256", "diagnostic_ledger",
        "boundary_evidence", "rq1_evidence", "queue_batch", "ranking_context",
        "prior_tuning_ledger", "diagnostic_tuning_ledger",
        "all_tuning_and_diagnostic_ledger", "comparisons", "diagnostic_conclusion",
        "provisional_selection", "continuation", "opportunity_accounting", "sha256",
    }
    payload = {name: value for name, value in document.items() if name != "sha256"}
    if (
        set(document) != expected
        or document.get("schema_version") != 1
        or document.get("kind") != "g3_rq2_unexpected_result_diagnostic_evidence"
        or document.get("protocol_sha256") != APPROVED_PROTOCOL_SHA256
        or document.get("sha256") != hashlib.sha256(_canonical_json(payload).encode()).hexdigest()
    ):
        raise ValueError("RQ2 diagnostic evidence identity or hash is invalid")
    return document


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-id", default=RQ2_UNEXPECTED_DIAGNOSTIC_BATCH_ID)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    arguments = parser.parse_args()
    root = arguments.root.resolve(strict=True)
    document = build_rq2_unexpected_diagnostic_evidence(root, batch_id=arguments.batch_id)
    path = root / RQ2_UNEXPECTED_DIAGNOSTIC_EVIDENCE_PATH
    if arguments.write:
        _persist_built_evidence(path, document)
    print(json.dumps({
        "path": str(path),
        "sha256": document["sha256"],
        "diagnostic_runs": len(document["diagnostic_tuning_ledger"]),
        "all_rows": len(document["all_tuning_and_diagnostic_ledger"]),
        "status": "materialized" if arguments.write else "preview",
    }, sort_keys=True))


if __name__ == "__main__":
    main()
