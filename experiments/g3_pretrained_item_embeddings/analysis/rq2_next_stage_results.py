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
    load_rq2_capacity_evidence,
)
from experiments.g3_pretrained_item_embeddings.launchers.control import (
    decode_control_job,
)
from experiments.g3_pretrained_item_embeddings.launchers.rq2_next_stage import (
    JOB_ENVIRONMENT,
    LEDGER_ENVIRONMENT,
    verify_rq2_next_stage_inputs,
)
from experiments.g3_pretrained_item_embeddings.protocol.constants import (
    APPROVED_PROTOCOL,
    APPROVED_PROTOCOL_SHA256,
)
from experiments.g3_pretrained_item_embeddings.protocol.rq2_next_stage_ledger import (
    APPROVED_RQ2_NEXT_STAGE_LEDGER_SHA256,
    RQ2_NEXT_STAGE_LEDGER_PATH,
    load_rq2_next_stage_ledger,
)


RQ2_NEXT_STAGE_BATCH_ID = "39120d657485411788c58fcffed03fc8"
RQ2_NEXT_STAGE_EVIDENCE_PATH = (
    "experiments/g3_pretrained_item_embeddings/evidence/"
    "rq2_resolved_next_stage.json"
)
APPROVED_RQ2_NEXT_STAGE_EVIDENCE_SHA256 = (
    "5a74dbc285e849b2ca74f52e0584f5f6a4f8e288b9bf6b0b71a1bdc5b2282b72"
)

_METRIC_NAMES = tuple(
    f"{name}@{cutoff}"
    for name in ("recall", "ndcg", "mrr", "capped_recall", "coverage")
    for cutoff in (10, 50, 100)
)


def select_id_only_winner(
    runs: Sequence[Mapping[str, object]],
) -> Mapping[str, object]:
    if not runs:
        raise ValueError("ID-only next-stage selection requires runs")
    return min(
        runs,
        key=lambda run: (
            -float(_metrics(run)["recall@100"]),
            -float(_metrics(run)["ndcg@100"]),
            float(run["queue_wall_seconds"]),
            int(run["manifest_order"]),
        ),
    )


def assess_id_only_boundaries(selected: Mapping[str, object]) -> dict[str, object]:
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
    horizon_extension = horizon == 40 and best_epoch == 40
    return {
        "embedding_learning_rate": embedding,
        "deep_learning_rate": deep,
        "horizon": {
            "selected_epochs": horizon,
            "restored_best_epoch": best_epoch,
            "extension_required": horizon_extension,
            "extension_epochs": 60 if horizon_extension else None,
        },
        "extension_required": bool(
            embedding["direction"] or deep["direction"] or horizon_extension
        ),
    }


def build_rq2_next_stage_evidence(
    root: Path,
    *,
    batch_id: str = RQ2_NEXT_STAGE_BATCH_ID,
) -> dict[str, object]:
    root = root.resolve(strict=True)
    ledger_path = root / RQ2_NEXT_STAGE_LEDGER_PATH
    ledger = load_rq2_next_stage_ledger(ledger_path)
    if ledger.sha256 != APPROVED_RQ2_NEXT_STAGE_LEDGER_SHA256:
        raise ValueError("RQ2 next-stage evidence received a different ledger")
    verify_rq2_next_stage_inputs(root, ledger, full_validation=True)
    batch_path = (
        root / "generated/training-queue-service/batches" / f"{batch_id}.json"
    )
    batch = _load_json(batch_path)
    job_ids = batch.get("jobs")
    if (
        batch.get("id") != batch_id
        or batch.get("sealed") is not True
        or not isinstance(job_ids, list)
        or len(job_ids) != len(ledger.rows) == 6
        or len(set(job_ids)) != len(job_ids)
    ):
        raise ValueError("RQ2 next-stage queue batch is not the exact sealed ledger")
    context_path = root / "generated/logs/.ranking-evidence/g3-native50m/context.pt"
    runs = [
        _collect_run(
            root=root,
            ledger=ledger,
            ledger_path=ledger_path,
            context_path=context_path,
            batch_id=batch_id,
            job_id=str(job_id),
            row=row.to_dict(),
            manifest_order=index,
        )
        for index, (row, job_id) in enumerate(
            zip(ledger.rows, job_ids, strict=True)
        )
    ]
    identifier = [
        run for run in runs if run["family_id"] == "rq2_id_only_densenet"
    ]
    if len(identifier) != 3:
        raise ValueError("RQ2 next stage lacks three ID-only horizon probes")
    selected = select_id_only_winner(identifier)
    boundary = assess_id_only_boundaries(selected)
    expected_winner = {
        "capacity": 255,
        "horizon_epochs": 40,
        "embedding_learning_rate": 0.3041556165944196,
        "deep_learning_rate": 0.014506684820055783,
        "best_epoch": 29,
    }
    if any(selected.get(key) != value for key, value in expected_winner.items()) or (
        _metrics(selected)["recall@100"] != 0.09074562121371973
    ):
        raise ValueError("RQ2 next-stage ID-only winner differs from the resolved result")
    if (
        boundary["embedding_learning_rate"]["direction"] is not None
        or boundary["deep_learning_rate"]["direction"] != "lower"
        or boundary["horizon"]["extension_required"] is not False
    ):
        raise ValueError("RQ2 next-stage boundary decision differs from approval")
    content_runs = [run for run in runs if run["family_id"] == "rq2_content_concat"]
    if len(content_runs) != 3:
        raise ValueError("RQ2 next stage lacks three content boundary probes")
    content_selected = select_id_only_winner(content_runs)
    preselection = load_rq2_capacity_evidence(
        root / ledger.preselection_evidence.path
    )
    family_selections = preselection.get("family_selections")
    if not isinstance(family_selections, list):
        raise ValueError("RQ2 preselection evidence lacks family selections")
    previous_content = next(
        value["selected"]
        for value in family_selections
        if isinstance(value, dict) and value.get("family_id") == "rq2_content_concat"
    )
    previous_recall = float(_metrics(previous_content)["recall@100"])
    extension_recall = float(_metrics(content_selected)["recall@100"])
    if (
        previous_content.get("row_id") != "rq2_content_concat:02"
        or content_selected.get("row_id") != "rq2_content_concat:14"
        or extension_recall <= previous_recall
    ):
        raise ValueError("RQ2 content capacity boundary result differs from evidence")
    payload = {
        "schema_version": 1,
        "kind": "g3_rq2_resolved_next_stage_evidence",
        "protocol_sha256": APPROVED_PROTOCOL_SHA256,
        "next_stage_ledger": _file_fact(root, ledger_path)
        | {"logical_sha256": ledger.sha256},
        "preselection_evidence": ledger.preselection_evidence.to_dict(),
        "predecessor_calibration": ledger.predecessor_calibration.to_dict(),
        "queue_batch": _file_fact(root, batch_path) | {"batch_id": batch_id},
        "ranking_context": _file_fact(root, context_path),
        "tuning_ledger": runs,
        "id_only_selection": {
            "selection_rule": (
                "validation Recall@100, validation NDCG@100, lower queue wall "
                "time, then manifest order"
            ),
            "selected": selected,
            "boundary_decision": boundary,
        },
        "content_capacity_decision": {
            "status": "deferred_pending_user_approval",
            "family_id": "rq2_content_concat",
            "evaluated_extension_capacity": 32,
            "selection_changed": True,
            "preselection_evidence_sha256": preselection["sha256"],
            "previous_selected": previous_content,
            "extension_selected": content_selected,
            "recall_at_100_delta": extension_recall - previous_recall,
        },
        "opportunity_accounting": {
            "rq2_content_concat": {
                "base_preselection": 9,
                "base_horizon_followup": 3,
                "approved_base_total": 12,
                "conditional_capacity_extension": 3,
                "cumulative_maximum_after_extension": 15,
            },
            "rq2_id_only_densenet": {
                "base_preselection": 9,
                "base_horizon_followup": 3,
                "approved_base_total": 12,
                "conditional_extensions_completed": 0,
                "cumulative_total_before_lr_extension": 12,
            },
        },
    }
    return _document(payload)


def persist_rq2_next_stage_evidence(
    path: Path, document: Mapping[str, object]
) -> Path:
    validated = _validate_document(dict(document), enforce_approved=False)
    content = (_canonical_json(validated) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(content)
    except FileExistsError:
        if path.read_bytes() != content:
            raise RuntimeError(f"immutable RQ2 next-stage evidence differs: {path}")
    return path


def load_rq2_next_stage_evidence(path: Path) -> dict[str, object]:
    return _validate_document(_load_json(path), enforce_approved=True)


def verify_rq2_next_stage_evidence(
    path: Path, *, root: Path
) -> dict[str, object]:
    evidence = load_rq2_next_stage_evidence(path)
    if _canonical_json(evidence) != _canonical_json(build_rq2_next_stage_evidence(root)):
        raise ValueError("RQ2 next-stage evidence differs from its bound artifacts")
    return evidence


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
    if (
        queue_job.get("id") != job_id
        or queue_job.get("batch_id") != batch_id
        or queue_job.get("run") != row["run_name"]
        or queue_job.get("exit_code") != 0
        or queue_job.get("data_group") != "g3-native50m-likes"
        or Path(str(queue_job.get("script"))).name != "run_rq2_next_stage.py"
    ):
        raise ValueError(f"RQ2 next-stage completion differs for {row['id']}")
    verify_unique_completed_run(
        root / "generated/training-queue-service",
        run_name=str(row["run_name"]),
        expected_job_id=job_id,
    )
    environment = queue_job.get("environment")
    if not isinstance(environment, list):
        raise ValueError(f"RQ2 next-stage environment is absent for {row['id']}")
    values = dict(value.split("=", 1) for value in environment if "=" in value)
    if (
        values.get("WANDB_MODE") != "offline"
        or Path(values.get(LEDGER_ENVIRONMENT, "")).resolve() != ledger_path
    ):
        raise ValueError(f"RQ2 next-stage environment differs for {row['id']}")
    compiled = decode_control_job(values.get(JOB_ENVIRONMENT, ""), ledger)
    if compiled.row_id != row["id"] or compiled.job != row:
        raise ValueError(f"RQ2 next-stage payload differs for {row['id']}")
    run_directory = root / "generated/logs" / str(row["run_name"])
    contract = _load_json(run_directory / "g3_rq2_next_stage_job.json")
    if contract != compiled.to_dict() | {
        "ledger_path": str(ledger_path),
        "ledger_sha256": ledger.sha256,
    }:
        raise ValueError(f"RQ2 next-stage contract differs for {row['id']}")
    training = row.get("training")
    representation = row.get("representation")
    if not isinstance(training, dict) or not isinstance(representation, dict):
        raise ValueError(f"RQ2 next-stage row is invalid for {row['id']}")
    metadata = _load_json(run_directory / "training_metadata.json")
    horizon = int(training["horizon_epochs"])
    expected = {
        "batch_size": training["batch_size"],
        "seed": training["seed"],
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
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError(f"RQ2 next-stage metadata differs for {row['id']}")
    history = {
        "rq2_content_concat": "id_content",
        "rq2_id_only_densenet": "id_only_densenet",
    }[str(row["family_id"])]
    expected_representation = {
        "catalog_representation": "learned_id",
        "content_gate": "fixed",
        "extra_item_id_dim": None,
        "gate_hidden_dim": None,
        "history_hidden_dim": representation["history_hidden_dim"],
        "history_representation": history,
        "metadata": [],
        "metadata_dim": None,
    }
    if metadata.get("g3_representation") != expected_representation:
        raise ValueError(f"RQ2 next-stage representation differs for {row['id']}")
    best_epoch = metadata.get("best_epoch")
    if type(best_epoch) is not int or not 1 <= best_epoch <= horizon:
        raise ValueError(f"RQ2 next-stage best epoch is invalid for {row['id']}")
    traces = metadata.get("lr_group_traces")
    if not isinstance(traces, dict) or any(
        not isinstance(traces.get(group), list)
        or len(traces[group]) != horizon
        or traces[group][-1] != 0.0
        for group in ("embedding", "deep")
    ):
        raise ValueError(f"RQ2 next-stage schedule is incomplete for {row['id']}")
    metrics_path = run_directory / "final_metrics.json"
    metrics = _load_json(metrics_path)
    if set(metrics) != {*_METRIC_NAMES, "num_users"} or any(
        not isinstance(value, (int, float)) or not math.isfinite(float(value))
        for value in metrics.values()
    ):
        raise ValueError(f"RQ2 next-stage metric schema differs for {row['id']}")
    ranking_path = run_directory / "ranking_evidence.pt"
    rankings_path = run_directory / "top_item_rankings.json"
    recomputed = _recompute_metrics(context_path, ranking_path, rankings_path)
    if any(abs(float(metrics[key]) - recomputed[key]) > 1e-15 for key in metrics):
        raise ValueError(f"RQ2 next-stage metrics differ for {row['id']}")
    diagnostics = _load_json(run_directory / "g3_training_diagnostics.json")
    nonfinite = _values_named(diagnostics, "nonfinite_count")
    if not nonfinite or any(value != 0 for value in nonfinite):
        raise ValueError(f"RQ2 next-stage diagnostics differ for {row['id']}")
    artifact_filenames = (
        ("job_contract", "g3_rq2_next_stage_job.json"),
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
    artifacts = {
        name: _file_fact(root, run_directory / filename)
        for name, filename in artifact_filenames
    }
    return {
        "manifest_order": manifest_order,
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
        "artifacts": artifacts,
    }


def _rate_boundary(value: float, bounds: tuple[float, float]) -> dict[str, object]:
    lower, upper = bounds
    if not lower <= value <= upper:
        raise ValueError("selected RQ2 learning rate is outside approved bounds")
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
        raise ValueError("RQ2 next-stage run has no metrics")
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


def _document(payload: Mapping[str, object]) -> dict[str, object]:
    document = dict(payload)
    document["sha256"] = _canonical_sha256(document)
    return document


def _validate_document(
    document: dict[str, object], *, enforce_approved: bool
) -> dict[str, object]:
    expected_keys = {
        "schema_version",
        "kind",
        "protocol_sha256",
        "next_stage_ledger",
        "preselection_evidence",
        "predecessor_calibration",
        "queue_batch",
        "ranking_context",
        "tuning_ledger",
        "id_only_selection",
        "content_capacity_decision",
        "opportunity_accounting",
        "sha256",
    }
    if set(document) != expected_keys:
        raise ValueError("RQ2 next-stage evidence keys are invalid")
    payload = {key: value for key, value in document.items() if key != "sha256"}
    if (
        type(document["schema_version"]) is not int
        or document["schema_version"] != 1
        or document["kind"] != "g3_rq2_resolved_next_stage_evidence"
        or document["protocol_sha256"] != APPROVED_PROTOCOL_SHA256
        or document["sha256"] != _canonical_sha256(payload)
    ):
        raise ValueError("RQ2 next-stage evidence identity or hash is invalid")
    if (
        enforce_approved
        and document["sha256"] != APPROVED_RQ2_NEXT_STAGE_EVIDENCE_SHA256
    ):
        raise ValueError("RQ2 next-stage evidence is not the approved result")
    return document


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    arguments = parser.parse_args()
    evidence = build_rq2_next_stage_evidence(arguments.root)
    path = arguments.root.resolve() / RQ2_NEXT_STAGE_EVIDENCE_PATH
    if arguments.write:
        persist_rq2_next_stage_evidence(path, evidence)
    print(
        json.dumps(
            {
                "path": str(path),
                "sha256": evidence["sha256"],
                "runs": len(evidence["tuning_ledger"]),
                "status": "materialized" if arguments.write else "preview",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
