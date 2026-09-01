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
from experiments.g3_pretrained_item_embeddings.analysis.rq2_next_stage_results import (
    load_rq2_next_stage_evidence,
)
from experiments.g3_pretrained_item_embeddings.launchers.control import (
    decode_control_job,
)
from experiments.g3_pretrained_item_embeddings.launchers.rq2_id_boundary import (
    JOB_ENVIRONMENT,
    LEDGER_ENVIRONMENT,
    verify_rq2_id_boundary_inputs,
)
from experiments.g3_pretrained_item_embeddings.protocol.constants import (
    APPROVED_PROTOCOL_SHA256,
)
from experiments.g3_pretrained_item_embeddings.protocol.rq2_id_boundary_ledger import (
    APPROVED_RQ2_ID_BOUNDARY_LEDGER_SHA256,
    RQ2_ID_BOUNDARY_LEDGER_PATH,
    load_rq2_id_boundary_ledger,
)


RQ2_ID_BOUNDARY_BATCH_ID = "55c225b9687a4f098e858bef235e4366"
RQ2_ID_BOUNDARY_EVIDENCE_PATH = (
    "experiments/g3_pretrained_item_embeddings/evidence/"
    "rq2_id_only_deep_lr_boundary_results.json"
)
APPROVED_RQ2_ID_BOUNDARY_EVIDENCE_SHA256 = (
    "1d7d1615b13a531dcc7d2fd6e8b420b2ccd482e2a3aa7d72c5c0ee433c6cf41b"
)

_METRIC_NAMES = tuple(
    f"{name}@{cutoff}"
    for name in ("recall", "ndcg", "mrr", "capped_recall", "coverage")
    for cutoff in (10, 50, 100)
)


def select_rq2_candidate(
    candidates: Sequence[Mapping[str, object]],
) -> Mapping[str, object]:
    if not candidates:
        raise ValueError("RQ2 ID boundary selection requires candidates")
    return min(
        candidates,
        key=lambda run: (
            -float(_metrics(run)["recall@100"]),
            -float(_metrics(run)["ndcg@100"]),
            float(run["queue_wall_seconds"]),
            int(run.get("combined_manifest_order", run["manifest_order"])),
        ),
    )


def build_rq2_id_boundary_evidence(
    root: Path,
    *,
    batch_id: str = RQ2_ID_BOUNDARY_BATCH_ID,
) -> dict[str, object]:
    root = root.resolve(strict=True)
    ledger_path = root / RQ2_ID_BOUNDARY_LEDGER_PATH
    ledger = load_rq2_id_boundary_ledger(ledger_path)
    if ledger.sha256 != APPROVED_RQ2_ID_BOUNDARY_LEDGER_SHA256:
        raise ValueError("RQ2 ID boundary evidence received a different ledger")
    verify_rq2_id_boundary_inputs(root, ledger, full_validation=True)
    predecessor_path = root / ledger.next_stage_evidence.path
    predecessor = load_rq2_next_stage_evidence(predecessor_path)
    batch_path = (
        root / "generated/training-queue-service/batches" / f"{batch_id}.json"
    )
    batch = _load_json(batch_path)
    job_ids = batch.get("jobs")
    if (
        batch.get("id") != batch_id
        or batch.get("sealed") is not True
        or not isinstance(job_ids, list)
        or len(job_ids) != len(ledger.rows) == 3
        or len(set(job_ids)) != len(job_ids)
    ):
        raise ValueError("RQ2 ID boundary queue batch is not the exact sealed ledger")
    context_path = root / "generated/logs/.ranking-evidence/g3-native50m/context.pt"
    boundary_runs = [
        _collect_run(
            root=root,
            ledger=ledger,
            ledger_path=ledger_path,
            context_path=context_path,
            batch_id=batch_id,
            job_id=str(job_id),
            row=row.to_dict(),
            manifest_order=index,
            combined_manifest_order=index + 1,
        )
        for index, (row, job_id) in enumerate(
            zip(ledger.rows, job_ids, strict=True)
        )
    ]
    expected_probe_metrics = (
        ("rq2_id_only_densenet:13", 0.0848355468753444, 0.030115138384337308, 24),
        ("rq2_id_only_densenet:14", 0.0857631866845541, 0.03186042945490013, 23),
        ("rq2_id_only_densenet:15", 0.08495562995955214, 0.032102164645662984, 15),
    )
    if any(
        run["row_id"] != row_id
        or _metrics(run)["recall@100"] != recall
        or _metrics(run)["ndcg@100"] != ndcg
        or run["best_epoch"] != best_epoch
        for run, (row_id, recall, ndcg, best_epoch) in zip(
            boundary_runs, expected_probe_metrics, strict=True
        )
    ):
        raise ValueError("RQ2 ID boundary metrics differ from the completed batch")
    outward_selected = select_rq2_candidate(boundary_runs)
    if outward_selected["row_id"] != "rq2_id_only_densenet:14":
        raise ValueError("RQ2 ID outward-probe winner differs from the approved rule")
    source = predecessor.get("id_only_selection", {}).get("selected")
    if not isinstance(source, dict):
        raise ValueError("RQ2 ID boundary predecessor lacks its selected run")
    source_candidate = dict(source) | {"combined_manifest_order": 0}
    if (
        source_candidate.get("row_id") != "rq2_id_only_densenet:12"
        or _metrics(source_candidate)["recall@100"] != 0.09074562121371973
        or _metrics(source_candidate)["ndcg@100"] != 0.031100697732330106
        or source_candidate.get("best_epoch") != 29
    ):
        raise ValueError("RQ2 ID boundary predecessor winner drifted")
    final_selected = select_rq2_candidate([source_candidate, *boundary_runs])
    if final_selected["row_id"] != "rq2_id_only_densenet:12":
        raise ValueError("RQ2 ID boundary final winner differs from the approved rule")
    content = predecessor.get("content_capacity_decision")
    if (
        not isinstance(content, dict)
        or content.get("status") != "deferred_pending_user_approval"
    ):
        raise ValueError("RQ2 content capacity decision is no longer deferred")
    payload = {
        "schema_version": 1,
        "kind": "g3_rq2_id_only_deep_lr_boundary_evidence",
        "protocol_sha256": APPROVED_PROTOCOL_SHA256,
        "id_boundary_ledger": _file_fact(root, ledger_path)
        | {"logical_sha256": ledger.sha256},
        "predecessor_evidence": _file_fact(root, predecessor_path)
        | {"logical_sha256": predecessor["sha256"]},
        "queue_batch": _file_fact(root, batch_path) | {"batch_id": batch_id},
        "ranking_context": _file_fact(root, context_path),
        "boundary_runs": boundary_runs,
        "outward_probe_selection": {
            "selection_rule": _SELECTION_RULE,
            "selected": outward_selected,
            "new_outer_boundary_row_id": "rq2_id_only_densenet:15",
        },
        "final_selection": {
            "selection_rule": _SELECTION_RULE,
            "candidate_row_ids": [
                "rq2_id_only_densenet:12",
                "rq2_id_only_densenet:13",
                "rq2_id_only_densenet:14",
                "rq2_id_only_densenet:15",
            ],
            "selected": final_selected,
            "boundary_decision": {
                "status": "resolved",
                "outward_winner_on_new_boundary": False,
                "additional_runs_authorized": False,
                "next_action": "none",
            },
        },
        "content_capacity_status": {
            "status": "deferred_pending_user_approval",
            "changed_by_this_evidence": False,
        },
        "opportunity_accounting": ledger.opportunity_accounting,
    }
    return _document(payload)


_SELECTION_RULE = (
    "validation Recall@100, validation NDCG@100, lower queue wall time, then "
    "combined manifest order"
)


def persist_rq2_id_boundary_evidence(
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
            raise RuntimeError(f"immutable RQ2 ID boundary evidence differs: {path}")
    return path


def load_rq2_id_boundary_evidence(path: Path) -> dict[str, object]:
    return _validate_document(_load_json(path), enforce_approved=True)


def verify_rq2_id_boundary_evidence(
    path: Path, *, root: Path
) -> dict[str, object]:
    evidence = load_rq2_id_boundary_evidence(path)
    rebuilt = build_rq2_id_boundary_evidence(root)
    if _canonical_json(evidence) != _canonical_json(rebuilt):
        raise ValueError("RQ2 ID boundary evidence differs from its bound artifacts")
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
    combined_manifest_order: int,
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
        or Path(str(queue_job.get("script"))).name != "run_rq2_id_boundary.py"
    ):
        raise ValueError(f"RQ2 ID boundary completion differs for {row['id']}")
    verify_unique_completed_run(
        root / "generated/training-queue-service",
        run_name=str(row["run_name"]),
        expected_job_id=job_id,
    )
    environment = queue_job.get("environment")
    if not isinstance(environment, list):
        raise ValueError(f"RQ2 ID boundary environment is absent for {row['id']}")
    values = dict(value.split("=", 1) for value in environment if "=" in value)
    if (
        values.get("WANDB_MODE") != "offline"
        or Path(values.get(LEDGER_ENVIRONMENT, "")).resolve() != ledger_path
    ):
        raise ValueError(f"RQ2 ID boundary environment differs for {row['id']}")
    compiled = decode_control_job(values.get(JOB_ENVIRONMENT, ""), ledger)
    if compiled.row_id != row["id"] or compiled.job != row:
        raise ValueError(f"RQ2 ID boundary payload differs for {row['id']}")
    run_directory = root / "generated/logs" / str(row["run_name"])
    contract = _load_json(run_directory / "g3_rq2_id_boundary_job.json")
    if contract != compiled.to_dict() | {
        "ledger_path": str(ledger_path),
        "ledger_sha256": ledger.sha256,
    }:
        raise ValueError(f"RQ2 ID boundary contract differs for {row['id']}")
    training = row.get("training")
    representation = row.get("representation")
    if not isinstance(training, dict) or not isinstance(representation, dict):
        raise ValueError(f"RQ2 ID boundary row is invalid for {row['id']}")
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
        raise ValueError(f"RQ2 ID boundary metadata differs for {row['id']}")
    expected_representation = {
        "catalog_representation": "learned_id",
        "content_gate": "fixed",
        "extra_item_id_dim": None,
        "gate_hidden_dim": None,
        "history_hidden_dim": 255,
        "history_representation": "id_only_densenet",
        "metadata": [],
        "metadata_dim": None,
    }
    if metadata.get("g3_representation") != expected_representation:
        raise ValueError(f"RQ2 ID boundary representation differs for {row['id']}")
    best_epoch = metadata.get("best_epoch")
    if type(best_epoch) is not int or not 1 <= best_epoch <= horizon:
        raise ValueError(f"RQ2 ID boundary best epoch is invalid for {row['id']}")
    traces = metadata.get("lr_group_traces")
    if not isinstance(traces, dict) or any(
        not isinstance(traces.get(group), list)
        or len(traces[group]) != horizon
        or traces[group][-1] != 0.0
        for group in ("embedding", "deep")
    ):
        raise ValueError(f"RQ2 ID boundary schedule is incomplete for {row['id']}")
    metrics_path = run_directory / "final_metrics.json"
    metrics = _load_json(metrics_path)
    if set(metrics) != {*_METRIC_NAMES, "num_users"} or any(
        not isinstance(value, (int, float)) or not math.isfinite(float(value))
        for value in metrics.values()
    ):
        raise ValueError(f"RQ2 ID boundary metric schema differs for {row['id']}")
    ranking_path = run_directory / "ranking_evidence.pt"
    rankings_path = run_directory / "top_item_rankings.json"
    recomputed = _recompute_metrics(context_path, ranking_path, rankings_path)
    if any(abs(float(metrics[key]) - recomputed[key]) > 1e-15 for key in metrics):
        raise ValueError(f"RQ2 ID boundary metrics differ for {row['id']}")
    diagnostics = _load_json(run_directory / "g3_training_diagnostics.json")
    nonfinite = _values_named(diagnostics, "nonfinite_count")
    if not nonfinite or any(value != 0 for value in nonfinite):
        raise ValueError(f"RQ2 ID boundary diagnostics differ for {row['id']}")
    artifact_filenames = (
        ("job_contract", "g3_rq2_id_boundary_job.json"),
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
        "combined_manifest_order": combined_manifest_order,
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


def _metrics(run: Mapping[str, object]) -> Mapping[str, object]:
    metrics = run.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("RQ2 ID boundary run has no metrics")
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
        "id_boundary_ledger",
        "predecessor_evidence",
        "queue_batch",
        "ranking_context",
        "boundary_runs",
        "outward_probe_selection",
        "final_selection",
        "content_capacity_status",
        "opportunity_accounting",
        "sha256",
    }
    if set(document) != expected_keys:
        raise ValueError("RQ2 ID boundary evidence keys differ from the closed schema")
    if (
        document.get("schema_version") != 1
        or document.get("kind") != "g3_rq2_id_only_deep_lr_boundary_evidence"
        or document.get("protocol_sha256") != APPROVED_PROTOCOL_SHA256
    ):
        raise ValueError("RQ2 ID boundary evidence identity differs")
    actual = document.get("sha256")
    expected = _canonical_sha256(
        {key: value for key, value in document.items() if key != "sha256"}
    )
    if actual != expected:
        raise ValueError("RQ2 ID boundary evidence hash differs from its payload")
    if (
        enforce_approved
        and APPROVED_RQ2_ID_BOUNDARY_EVIDENCE_SHA256 != "0" * 64
        and actual != APPROVED_RQ2_ID_BOUNDARY_EVIDENCE_SHA256
    ):
        raise ValueError("RQ2 ID boundary evidence is not the approved document")
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
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    arguments = parser.parse_args()
    root = arguments.root.resolve()
    document = build_rq2_id_boundary_evidence(root)
    path = root / RQ2_ID_BOUNDARY_EVIDENCE_PATH
    if arguments.write:
        persist_rq2_id_boundary_evidence(path, document)
    print(
        json.dumps(
            {
                "path": str(path),
                "sha256": document["sha256"],
                "status": "materialized" if arguments.write else "preview",
                "selected_row_id": document["final_selection"]["selected"]["row_id"],
                "next_action": document["final_selection"]["boundary_decision"]["next_action"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
