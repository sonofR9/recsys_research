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
from experiments.g3_pretrained_item_embeddings.analysis.rq2_unexpected_diagnostic_results import (
    RQ2_UNEXPECTED_DIAGNOSTIC_EVIDENCE_PATH,
    _values_named,
    load_rq2_unexpected_diagnostic_evidence,
)
from experiments.g3_pretrained_item_embeddings.launchers.control import decode_control_job
from experiments.g3_pretrained_item_embeddings.launchers.rq2_unexpected_width128_deep_lr_boundary import (
    JOB_ENVIRONMENT,
    LEDGER_ENVIRONMENT,
    verify_rq2_unexpected_width128_boundary_inputs,
)
from experiments.g3_pretrained_item_embeddings.protocol.constants import (
    APPROVED_PROTOCOL,
    APPROVED_PROTOCOL_SHA256,
)
from experiments.g3_pretrained_item_embeddings.protocol.rq2_capacity_ledger import (
    APPROVED_RQ2_CAPACITY_LEDGER_SHA256,
    RQ2_CAPACITY_LEDGER_PATH,
)
from experiments.g3_pretrained_item_embeddings.protocol.rq2_unexpected_diagnostic_ledger import (
    APPROVED_RQ2_UNEXPECTED_DIAGNOSTIC_LEDGER_SHA256,
    RQ2_UNEXPECTED_DIAGNOSTIC_LEDGER_PATH,
)
from experiments.g3_pretrained_item_embeddings.protocol.rq2_unexpected_width128_deep_lr_boundary_ledger import (
    APPROVED_RQ2_UNEXPECTED_WIDTH128_BOUNDARY_LEDGER_SHA256,
    RQ2_UNEXPECTED_WIDTH128_BOUNDARY_LEDGER_PATH,
    load_rq2_unexpected_width128_boundary_ledger,
)


RQ2_FINAL_BATCH_ID = "4923297369644bc7b5a247306a06aad4"
RQ2_FINAL_EVIDENCE_PATH = (
    "experiments/g3_pretrained_item_embeddings/evidence/rq2_final_native50m.json"
)

_METRIC_NAMES = tuple(
    f"{name}@{cutoff}"
    for name in ("recall", "ndcg", "mrr", "capped_recall", "coverage")
    for cutoff in (10, 50, 100)
)
_REUSABLE_ROW_IDS = (
    "rq2_content_concat:04",
    "rq2_content_concat:05",
    "rq2_content_concat:06",
    "rq2_unexpected_diagnostic:03",
    "rq2_content_concat:19",
    "rq2_content_concat:20",
    "rq2_content_concat:21",
)


def select_final_content_candidate(
    rows: Sequence[Mapping[str, object]],
) -> Mapping[str, object]:
    candidates = [row for row in rows if row.get("family_id") == "rq2_content_concat"]
    if not candidates:
        raise ValueError("final RQ2 evidence has no content-concat candidates")
    return min(
        candidates,
        key=lambda row: (
            -float(_metrics(row)["recall@100"]),
            -float(_metrics(row)["ndcg@100"]),
            float(row["queue_wall_seconds"]),
            str(row["row_id"]),
        ),
    )


def eligible_rq3_reuse_rows(
    rows: Sequence[Mapping[str, object]],
) -> list[Mapping[str, object]]:
    by_id = {row.get("row_id"): row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("final RQ2 ledger contains duplicate row ids")
    reusable = [by_id[row_id] for row_id in _REUSABLE_ROW_IDS if row_id in by_id]
    if (
        len(reusable) != 7
        or tuple(row.get("row_id") for row in reusable) != _REUSABLE_ROW_IDS
        or any(
            row.get("family_id") != "rq2_content_concat"
            or row.get("capacity") != 128
            for row in reusable
        )
    ):
        raise ValueError("final RQ2 evidence lacks the exact seven reusable rows")
    return reusable


def build_rq2_final_evidence(
    root: Path, *, batch_id: str = RQ2_FINAL_BATCH_ID
) -> dict[str, object]:
    root = root.resolve(strict=True)
    if batch_id != RQ2_FINAL_BATCH_ID:
        raise ValueError("final RQ2 evidence requires the approved exact batch")
    diagnostic_path = root / RQ2_UNEXPECTED_DIAGNOSTIC_EVIDENCE_PATH
    diagnostic = load_rq2_unexpected_diagnostic_evidence(diagnostic_path, root=root)
    boundary_ledger_path = root / RQ2_UNEXPECTED_WIDTH128_BOUNDARY_LEDGER_PATH
    ledger = load_rq2_unexpected_width128_boundary_ledger(
        boundary_ledger_path, root=root
    )
    verify_rq2_unexpected_width128_boundary_inputs(
        root, ledger, full_validation=False
    )
    if ledger.sha256 != APPROVED_RQ2_UNEXPECTED_WIDTH128_BOUNDARY_LEDGER_SHA256:
        raise ValueError("final RQ2 evidence received another boundary ledger")
    batch_path = root / "generated/training-queue-service/batches" / f"{batch_id}.json"
    job_ids = _validate_batch(_load_json(batch_path), batch_id=batch_id)
    context_path = root / "generated/logs/.ranking-evidence/g3-native50m/context.pt"
    boundary_runs = [
        _collect_boundary_run(
            root=root,
            ledger=ledger,
            ledger_path=boundary_ledger_path,
            context_path=context_path,
            batch_id=batch_id,
            job_id=job_id,
            row=row.to_dict(),
            manifest_order=index,
        )
        for index, (row, job_id) in enumerate(zip(ledger.rows, job_ids, strict=True))
    ]
    _validate_common_boundary_schedule(boundary_runs)
    prior = diagnostic.get("all_tuning_and_diagnostic_ledger")
    if not isinstance(prior, list) or len(prior) != 33:
        raise ValueError("final RQ2 diagnostic ledger changed")
    all_rows = [*prior, *boundary_runs]
    if (
        len(all_rows) != 36
        or len({row.get("row_id") for row in all_rows}) != 36
        or len({row.get("run_name") for row in all_rows}) != 36
    ):
        raise ValueError("final RQ2 evidence does not preserve exactly 36 unique rows")
    selected = select_final_content_candidate(all_rows)
    if selected.get("row_id") != "rq2_unexpected_diagnostic:03":
        raise ValueError("final RQ2 selected content candidate changed")
    reusable = eligible_rq3_reuse_rows(all_rows)
    if selected not in reusable:
        raise ValueError("final RQ2 selected candidate is absent from RQ3 reuse")
    identifier = _row_by_id(all_rows, "rq2_id_only_densenet:12")
    boundary = _resolved_boundary(selected, reusable)
    source_ledgers = _reuse_source_ledgers(root)
    if set(source_ledgers) != set(_REUSABLE_ROW_IDS):
        raise ValueError("final RQ2 reuse source-ledger mapping changed")
    payload = {
        "schema_version": 1,
        "kind": "g3_rq2_final_native50m_evidence",
        "protocol_sha256": APPROVED_PROTOCOL_SHA256,
        "diagnostic_evidence": _file_fact(root, diagnostic_path)
        | {"logical_sha256": diagnostic["sha256"]},
        "boundary_ledger": _file_fact(root, boundary_ledger_path)
        | {"logical_sha256": ledger.sha256},
        "queue_batch": _file_fact(root, batch_path) | {"batch_id": batch_id},
        "ranking_context": _file_fact(root, context_path),
        "boundary_tuning_ledger": boundary_runs,
        "all_tuning_diagnostic_boundary_ledger": all_rows,
        "final_content_selection": {
            "status": "resolved",
            "selection_rule": (
                "validation Recall@100, validation NDCG@100, lower queue wall "
                "time, then row id"
            ),
            "selected": selected,
            "provisional_selected": None,
            "boundary_decision": boundary,
        },
        "final_rq2_comparison": _comparison(selected, identifier),
        "rq3_inputs": {
            "status": "ready",
            "selected_content_input": selected,
            "id_only_control": identifier,
            "eligible_learned_output_reuse_rows": reusable,
            "reuse_source_ledgers": source_ledgers,
        },
        "opportunity_accounting": {
            "prior_tuning_and_diagnostic_rows": 33,
            "boundary_rows": 3,
            "all_preserved_rows": 36,
            "rq3_eligible_reuse_rows": 7,
        },
    }
    return _document(payload)


def persist_rq2_final_evidence(
    path: Path, document: Mapping[str, object], *, root: Path
) -> Path:
    validated = _validate_document(dict(document))
    _authenticate_document(validated, root=root)
    return _persist_built(path, validated)


def _persist_built(path: Path, document: Mapping[str, object]) -> Path:
    validated = _validate_document(dict(document))
    content = (_canonical_json(validated) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(content)
    except FileExistsError:
        if path.read_bytes() != content:
            raise RuntimeError(f"immutable final RQ2 evidence differs: {path}")
    return path


def load_rq2_final_evidence(path: Path, *, root: Path) -> dict[str, object]:
    evidence = _validate_document(_load_json(path))
    _authenticate_document(evidence, root=root)
    return evidence


def _authenticate_document(evidence: Mapping[str, object], *, root: Path) -> None:
    batch = evidence.get("queue_batch")
    if not isinstance(batch, dict) or batch.get("batch_id") != RQ2_FINAL_BATCH_ID:
        raise ValueError("final RQ2 evidence has no approved bound batch")
    rebuilt = build_rq2_final_evidence(root, batch_id=RQ2_FINAL_BATCH_ID)
    if _canonical_json(evidence) != _canonical_json(rebuilt):
        raise ValueError("final RQ2 evidence differs from bound artifacts")


def _validate_batch(batch: Mapping[str, object], *, batch_id: str) -> list[str]:
    jobs = batch.get("jobs")
    expected = [
        "54d1a07c33b54974a1356a98432812c5",
        "50c49413f8f3437081ff71432a8892fa",
        "eb0b236e4d944a83af708c4be4f90575",
    ]
    if (
        set(batch) != {"id", "jobs", "sealed", "sealed_at", "submitted_at"}
        or batch.get("id") != batch_id
        or batch.get("sealed") is not True
        or jobs != expected
        or not _finite_number(batch.get("submitted_at"))
        or not _finite_number(batch.get("sealed_at"))
        or float(batch["sealed_at"]) < float(batch["submitted_at"])
    ):
        raise ValueError("final RQ2 batch is not the approved exact sealed batch")
    return expected


def _collect_boundary_run(
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
    job = _load_json(completed_path)
    expected_script = (
        root
        / "experiments/g3_pretrained_item_embeddings/launchers/"
        "run_rq2_unexpected_width128_deep_lr_boundary.py"
    ).resolve(strict=True)
    if (
        set(job)
        != {
            "id", "batch_id", "data_group", "dispatched_at", "environment",
            "exit_code", "finished_at", "run", "script", "submitted_at",
        }
        or job.get("id") != job_id
        or job.get("batch_id") != batch_id
        or job.get("run") != row["run_name"]
        or job.get("exit_code") != 0
        or job.get("data_group") != "g3-native50m-likes"
        or Path(str(job.get("script"))).resolve() != expected_script
        or not _ordered_job_times(job)
    ):
        raise ValueError(f"final RQ2 completion differs for {row['id']}")
    verify_unique_completed_run(
        root / "generated/training-queue-service",
        run_name=str(row["run_name"]),
        expected_job_id=job_id,
    )
    environment = job.get("environment")
    pairs = [value.split("=", 1) for value in environment] if isinstance(environment, list) else []
    values = dict(pairs)
    if (
        len(pairs) != len(values) == 3
        or set(values) != {"WANDB_MODE", JOB_ENVIRONMENT, LEDGER_ENVIRONMENT}
        or values["WANDB_MODE"] != "offline"
        or Path(values[LEDGER_ENVIRONMENT]).resolve() != ledger_path
    ):
        raise ValueError(f"final RQ2 environment differs for {row['id']}")
    compiled = decode_control_job(values[JOB_ENVIRONMENT], ledger)
    if compiled.row_id != row["id"] or compiled.job != row:
        raise ValueError(f"final RQ2 payload differs for {row['id']}")
    run_directory = root / "generated/logs" / str(row["run_name"])
    contract_name = "g3_rq2_unexpected_width128_boundary_job.json"
    if _load_json(run_directory / contract_name) != compiled.to_dict() | {
        "ledger_path": str(ledger_path),
        "ledger_sha256": ledger.sha256,
    }:
        raise ValueError(f"final RQ2 contract differs for {row['id']}")
    training = row["training"]
    metadata = _load_json(run_directory / "training_metadata.json")
    expected_metadata = {
        "batch_size": 512,
        "seed": 42,
        "embedding_learning_rate": 0.3041556165944196,
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
        raise ValueError(f"final RQ2 metadata differs for {row['id']}")
    representation = metadata.get("g3_representation")
    if (
        not isinstance(representation, dict)
        or representation.get("history_representation") != "id_content"
        or representation.get("catalog_representation") != "learned_id"
        or representation.get("history_hidden_dim") != 128
    ):
        raise ValueError(f"final RQ2 representation differs for {row['id']}")
    best_epoch = metadata.get("best_epoch")
    if type(best_epoch) is not int or not 1 <= best_epoch < 40:
        raise ValueError(f"final RQ2 best epoch differs for {row['id']}")
    normalized = _validate_schedule(metadata, training=training, row_id=str(row["id"]))
    metrics = _load_json(run_directory / "final_metrics.json")
    if set(metrics) != {*_METRIC_NAMES, "num_users"} or any(
        not _finite_number(value) for value in metrics.values()
    ):
        raise ValueError(f"final RQ2 metrics differ for {row['id']}")
    recomputed = _recompute_metrics(
        context_path,
        run_directory / "ranking_evidence.pt",
        run_directory / "top_item_rankings.json",
    )
    max_delta = max(abs(float(metrics[name]) - float(recomputed[name])) for name in metrics)
    if set(recomputed) != set(metrics) or max_delta > 1e-15:
        raise ValueError(f"final RQ2 recomputed metrics differ for {row['id']}")
    diagnostics = _load_json(run_directory / "g3_training_diagnostics.json")
    nonfinite = _values_named(diagnostics, "nonfinite_count")
    if not nonfinite or any(value != 0 for value in nonfinite):
        raise ValueError(f"final RQ2 diagnostics differ for {row['id']}")
    artifacts = (
        ("job_contract", contract_name),
        ("training_metadata", "training_metadata.json"),
        ("final_metrics", "final_metrics.json"),
        ("ranking_evidence", "ranking_evidence.pt"),
        ("top_item_rankings", "top_item_rankings.json"),
        ("training_diagnostics", "g3_training_diagnostics.json"),
        ("sweep_log", "sweep.log"),
    )
    verify_artifacts_in_job_window(
        tuple(run_directory / filename for _, filename in artifacts),
        dispatched_at=float(job["dispatched_at"]),
        finished_at=float(job["finished_at"]),
        run_label=str(row["id"]),
    )
    return {
        "manifest_order": manifest_order,
        "row_id": row["id"],
        "run_name": row["run_name"],
        "role": row["role"],
        "family_id": row["family_id"],
        "capacity": 128,
        "embedding_learning_rate": training["embedding_learning_rate"],
        "deep_learning_rate": training["deep_learning_rate"],
        "horizon_epochs": 40,
        "best_epoch": best_epoch,
        "epochs_trained": 40,
        "queue_wall_seconds": float(job["finished_at"]) - float(job["dispatched_at"]),
        "metrics": metrics,
        "metric_provenance": {
            "recomputed_from_ranking_evidence": True,
            "absolute_tolerance": 1e-15,
            "maximum_absolute_difference": max_delta,
            "num_users": recomputed["num_users"],
        },
        "diagnostic_nonfinite_count": sum(int(value) for value in nonfinite),
        "normalized_lr_schedule": normalized,
        "queue_job": _file_fact(root, completed_path) | {"job_id": job_id},
        "artifacts": {
            name: _file_fact(root, run_directory / filename)
            for name, filename in artifacts
        },
    }


def _validate_schedule(
    metadata: Mapping[str, object], *, training: Mapping[str, object], row_id: str
) -> list[float]:
    traces = metadata.get("lr_group_traces")
    if not isinstance(traces, dict) or set(traces) != {"embedding", "deep"}:
        raise ValueError(f"final RQ2 schedule differs for {row_id}")
    normalized = []
    for group, rate_name in (
        ("embedding", "embedding_learning_rate"),
        ("deep", "deep_learning_rate"),
    ):
        trace = traces[group]
        if (
            not isinstance(trace, list)
            or len(trace) != 40
            or trace[-1] != 0.0
            or any(not _finite_number(value) for value in trace)
        ):
            raise ValueError(f"final RQ2 schedule differs for {row_id}")
        values = [float(value) / float(training[rate_name]) for value in trace]
        if any(value < 0.0 or value > 1.0 for value in values) or any(
            later > earlier for earlier, later in zip(values, values[1:])
        ):
            raise ValueError(f"final RQ2 schedule differs for {row_id}")
        normalized.append(values)
    if any(
        abs(left - right) > 1e-15
        for left, right in zip(normalized[0], normalized[1], strict=True)
    ):
        raise ValueError(f"final RQ2 optimizer-group schedules differ for {row_id}")
    return normalized[0]


def _validate_common_boundary_schedule(
    runs: Sequence[Mapping[str, object]],
) -> None:
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
        raise ValueError("final RQ2 boundary schedules differ")


def _resolved_boundary(
    selected: Mapping[str, object], reusable: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    rates = sorted({float(row["deep_learning_rate"]) for row in reusable})
    rate = float(selected["deep_learning_rate"])
    lower = [value for value in rates if value < rate]
    higher = [value for value in rates if value > rate]
    if not lower or not higher or int(selected["best_epoch"]) == int(selected["horizon_epochs"]):
        raise ValueError("final RQ2 selection is not interior and horizon-resolved")
    return {
        "status": "resolved",
        "selected_deep_learning_rate": rate,
        "tested_width128_deep_learning_rates": rates,
        "has_tested_lower_rate": True,
        "has_tested_higher_rate": True,
        "selected_is_interior": True,
        "horizon_extension_required": False,
        "additional_runs_authorized": False,
        "next_action": "none",
    }


def _comparison(
    treatment: Mapping[str, object], baseline: Mapping[str, object]
) -> dict[str, object]:
    output: dict[str, object] = {
        "content_concat": treatment,
        "id_only_densenet": baseline,
    }
    for metric in ("recall@100", "ndcg@100"):
        treatment_value = float(_metrics(treatment)[metric])
        baseline_value = float(_metrics(baseline)[metric])
        delta = treatment_value - baseline_value
        band = baseline_value * APPROVED_PROTOCOL.relative_dispersion(
            "native-50m", metric
        )
        prefix = metric.replace("@", "_at_")
        output |= {
            f"{prefix}_delta": delta,
            f"{prefix}_relative_change": delta / baseline_value,
            f"{prefix}_operational_band": band,
            f"{prefix}_change_exceeds_operational_band": abs(delta) > band,
        }
    output["content_beats_id_only"] = (
        float(_metrics(treatment)["recall@100"])
        > float(_metrics(baseline)["recall@100"])
    )
    return output


def _reuse_source_ledgers(root: Path) -> dict[str, object]:
    facts = {
        "capacity": _file_fact(root, root / RQ2_CAPACITY_LEDGER_PATH)
        | {"logical_sha256": APPROVED_RQ2_CAPACITY_LEDGER_SHA256},
        "diagnostic": _file_fact(root, root / RQ2_UNEXPECTED_DIAGNOSTIC_LEDGER_PATH)
        | {"logical_sha256": APPROVED_RQ2_UNEXPECTED_DIAGNOSTIC_LEDGER_SHA256},
        "boundary": _file_fact(root, root / RQ2_UNEXPECTED_WIDTH128_BOUNDARY_LEDGER_PATH)
        | {"logical_sha256": APPROVED_RQ2_UNEXPECTED_WIDTH128_BOUNDARY_LEDGER_SHA256},
    }
    return {
        row_id: facts[
            "capacity"
            if row_id in {"rq2_content_concat:04", "rq2_content_concat:05", "rq2_content_concat:06"}
            else "diagnostic"
            if row_id == "rq2_unexpected_diagnostic:03"
            else "boundary"
        ]
        for row_id in _REUSABLE_ROW_IDS
    }


def _row_by_id(
    rows: Sequence[Mapping[str, object]], row_id: str
) -> Mapping[str, object]:
    matches = [row for row in rows if row.get("row_id") == row_id]
    if len(matches) != 1:
        raise ValueError(f"final RQ2 evidence requires exactly one {row_id}")
    return matches[0]


def _metrics(run: Mapping[str, object]) -> Mapping[str, object]:
    metrics = run.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("final RQ2 row has no metrics")
    return metrics


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
        "schema_version", "kind", "protocol_sha256", "diagnostic_evidence",
        "boundary_ledger", "queue_batch", "ranking_context",
        "boundary_tuning_ledger", "all_tuning_diagnostic_boundary_ledger",
        "final_content_selection", "final_rq2_comparison", "rq3_inputs",
        "opportunity_accounting", "sha256",
    }
    payload = {name: value for name, value in document.items() if name != "sha256"}
    if (
        set(document) != expected
        or document.get("schema_version") != 1
        or document.get("kind") != "g3_rq2_final_native50m_evidence"
        or document.get("protocol_sha256") != APPROVED_PROTOCOL_SHA256
        or document.get("sha256") != hashlib.sha256(_canonical_json(payload).encode()).hexdigest()
    ):
        raise ValueError("final RQ2 evidence identity or hash is invalid")
    return document


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-id", default=RQ2_FINAL_BATCH_ID)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    arguments = parser.parse_args()
    root = arguments.root.resolve(strict=True)
    document = build_rq2_final_evidence(root, batch_id=arguments.batch_id)
    path = root / RQ2_FINAL_EVIDENCE_PATH
    if arguments.write:
        _persist_built(path, document)
    print(json.dumps({
        "path": str(path),
        "sha256": document["sha256"],
        "all_rows": len(document["all_tuning_diagnostic_boundary_ledger"]),
        "reuse_rows": len(document["rq3_inputs"]["eligible_learned_output_reuse_rows"]),
        "selected_row_id": document["final_content_selection"]["selected"]["row_id"],
        "status": "materialized" if arguments.write else "preview",
    }, sort_keys=True))


if __name__ == "__main__":
    main()
