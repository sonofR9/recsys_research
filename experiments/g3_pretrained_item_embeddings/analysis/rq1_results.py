from __future__ import annotations

from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Literal, Mapping, Sequence

import polars as pl

from dcn.eval.ranking_evidence import load_ranking_evidence
from experiments.g3_pretrained_item_embeddings.analysis.control_calibration import (
    _file_fact,
    _load_json,
    _recompute_metrics,
    load_control_calibration,
)
from experiments.g3_pretrained_item_embeddings.analysis.reports import (
    TuningRow,
    build_tuning_report,
)
from experiments.g3_pretrained_item_embeddings.analysis.queue_attribution import (
    verify_artifacts_in_job_window,
    verify_unique_completed_run,
)
from experiments.g3_pretrained_item_embeddings.analysis.slices import (
    _terciles,
    compute_ranking_slices,
)
from experiments.g3_pretrained_item_embeddings.launchers.rq1 import (
    JOB_ENVIRONMENT,
    LEDGER_ENVIRONMENT,
    decode_control_job,
    verify_rq1_inputs,
)
from experiments.g3_pretrained_item_embeddings.protocol.constants import (
    APPROVED_PROTOCOL,
    APPROVED_PROTOCOL_SHA256,
)
from experiments.g3_pretrained_item_embeddings.protocol.rq1_ledger import (
    APPROVED_PREDECESSOR_CALIBRATION_SHA256,
    APPROVED_RQ1_LEDGER_SHA256,
    PREDECESSOR_CALIBRATION_PATH,
    RQ1_LEDGER_PATH,
    load_rq1_ledger,
)
from experiments.g3_pretrained_item_embeddings.protocol.selection import (
    MetricEvidence,
    PromotionRule,
    decide_promotion,
)


RQ1_BATCH_ID = "05d24119ed134265897eaeabfd8b19a6"
RQ1_EVIDENCE_PATH = (
    "experiments/g3_pretrained_item_embeddings/evidence/rq1_content_input.json"
)
APPROVED_RQ1_EVIDENCE_SHA256 = (
    "99fab2310c9147f4f1797ed0073ef82bc810a801692ee59278f0508c7a30fcb4"
)
_TIED_CONTROL_EVIDENCE_PATH = (
    "experiments/g4_future_items/evidence/rq1_rq2_evaluation_native50m.json"
)
_TIED_CONTROL_EVIDENCE_SHA256 = (
    "c2d61837572f4790d6ff20995ec72f5fe335abc4b2bf97c108fe349ca0b342ba"
)
_TIED_CONTROL_MANIFEST_PATH = (
    "experiments/g4_future_items/protocol/selected_control_manifest.json"
)
_TIED_CONTROL_QUEUE_JOB_ID = "a76fded1be08464cb38ba2252b5fca97"

_METRIC_NAMES = tuple(
    f"{name}@{cutoff}"
    for name in ("recall", "ndcg", "mrr", "capped_recall", "coverage")
    for cutoff in (10, 50, 100)
)
_TIMING_PATTERN = re.compile(r"timing\.train_epoch_time=([0-9.]+)")
_MEMORY_PATTERN = re.compile(r"resources\.peak_memory_gb=([0-9.]+)")
_PARAMETER_PATTERN = re.compile(r"resources\.params_total=([0-9.]+)")
_TIMESTAMP_PATTERN = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})[,.](\d{3})"
)


def select_rq1_winner(
    runs: Sequence[dict[str, object]],
) -> dict[str, object]:
    if not runs:
        raise ValueError("RQ1 selection requires at least one run")
    return min(
        runs,
        key=lambda run: (
            -float(_metrics(run)["recall@100"]),
            -float(_metrics(run)["ndcg@100"]),
            float(run["queue_wall_seconds"]),
            str(run["row_id"]),
        ),
    )


def assess_rq1_boundaries(selected: Mapping[str, object]) -> dict[str, object]:
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
    extend_horizon = 60 if horizon == 40 and best_epoch == 40 else None
    result = {
        "embedding_learning_rate": embedding,
        "deep_learning_rate": deep,
        "horizon": {
            "selected_epochs": horizon,
            "restored_best_epoch": best_epoch,
            "extend_to_epochs": extend_horizon,
        },
    }
    result["extension_required"] = bool(
        embedding["direction"] or deep["direction"] or extend_horizon
    )
    return result


def build_rq1_evidence(root: Path, *, batch_id: str = RQ1_BATCH_ID) -> dict[str, object]:
    root = root.resolve(strict=True)
    ledger_path = root / RQ1_LEDGER_PATH
    ledger = load_rq1_ledger(ledger_path)
    if ledger.sha256 != APPROVED_RQ1_LEDGER_SHA256:
        raise ValueError("RQ1 evidence received a different ledger")
    feature_path = verify_rq1_inputs(root, ledger, full_validation=True)
    calibration_path = root / PREDECESSOR_CALIBRATION_PATH
    calibration = load_control_calibration(calibration_path)
    if calibration["sha256"] != APPROVED_PREDECESSOR_CALIBRATION_SHA256:
        raise ValueError("RQ1 evidence received a different calibration")

    batch_path = (
        root / "generated/training-queue-service/batches" / f"{batch_id}.json"
    )
    batch = _load_json(batch_path)
    job_ids = batch.get("jobs")
    if (
        batch.get("id") != batch_id
        or batch.get("sealed") is not True
        or not isinstance(job_ids, list)
        or len(job_ids) != len(ledger.rows) == 9
        or len(set(job_ids)) != len(job_ids)
    ):
        raise ValueError("RQ1 queue batch is not the exact sealed nine-row batch")

    context_path = root / "generated/logs/.ranking-evidence/g3-native50m/context.pt"
    item_counts = load_training_item_counts(feature_path)
    runs = [
        _collect_run(
            root=root,
            ledger=ledger,
            ledger_path=ledger_path,
            context_path=context_path,
            batch_id=batch_id,
            job_id=str(job_id),
            row=row.to_dict(),
            item_counts=item_counts,
        )
        for row, job_id in zip(ledger.rows, job_ids, strict=True)
    ]
    selected = select_rq1_winner(runs)
    control_runs = [
        run for run in calibration["tuning_ledger"] if run.get("role") == "search"
    ]
    if len(control_runs) != 9:
        raise ValueError("RQ1 calibration lacks the nine-row local control")
    control = select_rq1_winner(control_runs)
    control_comparison = _control_comparison(
        root=root,
        context_path=context_path,
        feature_path=feature_path,
        control=control,
    )
    tied_comparison = _tied_control_comparison(
        root=root,
        feature_path=feature_path,
    )
    treatment_comparison = {
        "row_id": selected["row_id"],
        "metrics": selected["metrics"],
        "slices": selected["slices"],
        "efficiency": selected["efficiency"],
    }
    _verify_slice_identity(control_comparison["slices"], selected["slices"])
    _verify_slice_identity(tied_comparison["slices"], selected["slices"])
    _verify_context_identity(
        root=root,
        treatment_run_name=str(selected["run_name"]),
        tied_run_name=str(tied_comparison["run_name"]),
    )
    _verify_matched_protocol(root, control, selected)
    promotion_rule = PromotionRule(
        primary_comparators=("tied_original",),
        tail_tradeoff_comparators=("tied_original",),
        tail_comparators=("untied_control",),
    )
    promotion = decide_promotion(
        candidate=MetricEvidence(
            id="rq1_content_input",
            recall_at_100=float(_metrics(selected)["recall@100"]),
            tail_recall_at_100=float(selected["slices"]["tail"]["recall@100"]),
        ),
        references={
            "untied_control": MetricEvidence(
                id="untied_control",
                recall_at_100=float(_metrics(control)["recall@100"]),
                tail_recall_at_100=float(
                    control_comparison["slices"]["tail"]["recall@100"]
                ),
            ),
            "tied_original": MetricEvidence(
                id="tied_original",
                recall_at_100=float(tied_comparison["metrics"]["recall@100"]),
                tail_recall_at_100=float(
                    tied_comparison["slices"]["tail"]["recall@100"]
                ),
            ),
        },
        rule=promotion_rule,
        relative_dispersion=APPROVED_PROTOCOL.relative_dispersion(
            "native-50m", "recall@100"
        ),
    )
    boundary = assess_rq1_boundaries(selected)
    comparison = {
        "tied_original": tied_comparison,
        "control": control_comparison,
        "treatment": treatment_comparison,
        "metric_deltas": {
            metric: float(_metrics(selected)[metric]) - float(_metrics(control)[metric])
            for metric in _METRIC_NAMES
        },
        "absolute_bands": {
            metric: abs(float(_metrics(control)[metric]))
            * APPROVED_PROTOCOL.relative_dispersion("native-50m", metric)
            for metric in _METRIC_NAMES
        },
        "direct_comparisons": {
            comparator: {
                "metric_deltas": {
                    metric: float(_metrics(selected)[metric])
                    - float(reference["metrics"][metric])
                    for metric in _METRIC_NAMES
                },
                "absolute_bands": {
                    metric: abs(float(reference["metrics"][metric]))
                    * APPROVED_PROTOCOL.relative_dispersion("native-50m", metric)
                    for metric in _METRIC_NAMES
                },
                "recall_at_100_delta_percent": 100.0
                * (
                    float(_metrics(selected)["recall@100"])
                    / float(reference["metrics"]["recall@100"])
                    - 1.0
                ),
                "recall_at_100_within_band": abs(
                    float(_metrics(selected)["recall@100"])
                    - float(reference["metrics"]["recall@100"])
                )
                <= abs(float(reference["metrics"]["recall@100"]))
                * APPROVED_PROTOCOL.relative_dispersion(
                    "native-50m", "recall@100"
                ),
            }
            for comparator, reference in (
                ("untied_control", control_comparison),
                ("tied_original", tied_comparison),
            )
        },
    }
    payload = {
        "schema_version": 1,
        "kind": "g3_rq1_content_input_evidence",
        "protocol_sha256": APPROVED_PROTOCOL_SHA256,
        "rq1_ledger": _file_fact(root, ledger_path)
        | {"logical_sha256": ledger.sha256},
        "predecessor_calibration": _file_fact(root, calibration_path)
        | {"logical_sha256": calibration["sha256"]},
        "queue_batch": _file_fact(root, batch_path) | {"batch_id": batch_id},
        "ranking_context": _file_fact(root, context_path),
        "tied_control_manifest": _file_fact(
            root, root / _TIED_CONTROL_MANIFEST_PATH
        ),
        "tied_control_evidence": _file_fact(
            root, root / _TIED_CONTROL_EVIDENCE_PATH
        ),
        "tied_ranking_context": _file_fact(
            root,
            root / "generated/logs/.ranking-evidence/g4-native50m/context.pt",
        ),
        "feature_data": _file_fact(root, feature_path),
        "tuning_ledger": runs,
        "selected_treatment": selected,
        "selected_control": control,
        "selection_rule": (
            "validation Recall@100, validation NDCG@100, lower queue wall time, "
            "then ledger order"
        ),
        "boundary_decision": boundary,
        "comparison": comparison,
        "promotion_decision": {
            "treatment_id": promotion.treatment_id,
            "promoted": promotion.promoted,
            "route": promotion.route,
            "absolute_bands": dict(promotion.absolute_bands),
            "comparators": {
                "aggregate_improvement": list(promotion_rule.primary_comparators),
                "aggregate_tail_tradeoff_band": list(
                    promotion_rule.tail_tradeoff_comparators
                ),
                "tail_improvement": list(promotion_rule.tail_comparators),
            },
            "reason": promotion.reason,
        },
        "diagnostic_assessment": {
            "matched_non_representation_protocol": True,
            "all_saved_diagnostics_finite": True,
            "unexpected_outcome": False,
            "finding": (
                "Content-only history improves overall Recall@100 beyond the "
                "approved native-50M band relative to the untied local control "
                "and improves descriptive tail, mid, and head Recall@100. Its "
                "direct effect relative to the tied original baseline is lower "
                "but unresolved inside that comparator's band. This is consistent "
                "with the declared content-generalization hypothesis and requires "
                "no corrective run."
            ),
        },
    }
    return _document(payload)


def persist_rq1_evidence(path: Path, document: Mapping[str, object]) -> Path:
    validated = _validate_document(dict(document), enforce_approved=False)
    content = (_canonical_json(validated) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(content)
    except FileExistsError:
        if path.read_bytes() != content:
            raise RuntimeError(f"immutable RQ1 evidence differs: {path}")
    return path


def load_rq1_evidence(path: Path) -> dict[str, object]:
    document = _load_json(path)
    return _validate_document(document, enforce_approved=True)


def verify_rq1_evidence(path: Path, *, root: Path) -> dict[str, object]:
    document = load_rq1_evidence(path)
    if document != build_rq1_evidence(root):
        raise ValueError("RQ1 evidence differs from its bound source artifacts")
    return document


def build_rq1_report_documents(
    evidence: Mapping[str, object],
) -> tuple[str, str]:
    runs = evidence["tuning_ledger"]
    selected = evidence["selected_treatment"]
    comparison = evidence["comparison"]
    boundary = evidence.get("boundary_decision")
    promotion = evidence.get("promotion_decision")
    if not isinstance(runs, list) or not isinstance(selected, dict):
        raise ValueError("RQ1 report evidence has invalid tuning rows")
    if not isinstance(comparison, dict):
        raise ValueError("RQ1 report evidence has no comparison")
    if not isinstance(boundary, dict) or not isinstance(promotion, dict):
        raise ValueError("RQ1 report evidence has no frozen decision")
    tuning_rows = tuple(
        TuningRow(
            research_question="RQ1",
            family="content-only history input",
            trial_id=str(run["row_id"]),
            status="usable",
            embedding_learning_rate=float(run["embedding_learning_rate"]),
            deep_learning_rate=float(run["deep_learning_rate"]),
            declared_horizon_epochs=int(run["horizon_epochs"]),
            completed_horizon_epochs=int(run["epochs_trained"]),
            restored_best_epoch=int(run["best_epoch"]),
            capacity=None,
            validation_recall_at_100=float(_metrics(run)["recall@100"]),
            validation_ndcg_at_100=float(_metrics(run)["ndcg@100"]),
            training_seconds=float(run["efficiency"]["logged_training_seconds"]),
        )
        for run in runs
    )
    tuning = build_tuning_report(tuning_rows)
    if f"**{selected['row_id']}**" not in tuning:
        raise ValueError("RQ1 tuning report bolded a different winner")

    control = comparison["control"]
    tied = comparison["tied_original"]
    treatment = comparison["treatment"]
    heading = (
        "## RQ1 — What happens when pretrained embeddings replace history item IDs?"
    )
    reader = "\n\n".join(
        (
            heading,
            "### Overall",
            _overall_table(tied, control, treatment),
            "### Frozen boundary and promotion decision",
            _decision_table(boundary, promotion),
            "### Item-frequency slices",
            _slice_table(
                tied["slices"], control["slices"], treatment["slices"]
            ),
            "### Efficiency",
            _efficiency_table(
                tied["efficiency"],
                control["efficiency"],
                treatment["efficiency"],
            ),
            "### Full-catalog timing status",
            _full_catalog_timing_table(
                tied["efficiency"],
                control["efficiency"],
                treatment["efficiency"],
            ),
        )
    )
    return tuning, reader + "\n"


def persist_rq1_reports(
    *, evidence: Mapping[str, object], tuning_path: Path, reader_path: Path
) -> tuple[Path, Path]:
    tuning, reader = build_rq1_report_documents(evidence)
    for path, content in ((tuning_path, tuning), (reader_path, reader)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    return tuning_path, reader_path


def _collect_run(
    *,
    root: Path,
    ledger: Any,
    ledger_path: Path,
    context_path: Path,
    batch_id: str,
    job_id: str,
    row: dict[str, object],
    item_counts: Mapping[int, int],
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
        or Path(str(queue_job.get("script"))).name != "run_rq1.py"
    ):
        raise ValueError(f"RQ1 queue completion differs for {row['id']}")
    environment = queue_job.get("environment")
    if not isinstance(environment, list):
        raise ValueError(f"RQ1 queue environment is absent for {row['id']}")
    values = dict(value.split("=", 1) for value in environment if "=" in value)
    if (
        values.get("WANDB_MODE") != "offline"
        or Path(values.get(LEDGER_ENVIRONMENT, "")).resolve() != ledger_path
    ):
        raise ValueError(f"RQ1 queue environment differs for {row['id']}")
    compiled = decode_control_job(values.get(JOB_ENVIRONMENT, ""), ledger)
    if compiled.row_id != row["id"] or compiled.job != row:
        raise ValueError(f"RQ1 queue payload differs for {row['id']}")

    run_directory = root / "generated/logs" / str(row["run_name"])
    contract = _load_json(run_directory / "g3_rq1_job.json")
    if contract != compiled.to_dict() | {
        "ledger_path": str(ledger_path),
        "ledger_sha256": ledger.sha256,
    }:
        raise ValueError(f"RQ1 job contract differs for {row['id']}")
    metadata = _load_json(run_directory / "training_metadata.json")
    training = row["training"]
    if not isinstance(training, dict):
        raise ValueError(f"RQ1 training coordinate is invalid for {row['id']}")
    horizon = int(training["horizon_epochs"])
    expected = {
        "batch_size": training["batch_size"],
        "seed": training["seed"],
        "embedding_learning_rate": training["embedding_learning_rate"],
        "deep_learning_rate": training["deep_learning_rate"],
        "lr_schedule_horizon_epochs": horizon,
        "num_epochs": horizon,
        "max_epochs": horizon,
        "epochs_trained": horizon,
        "stopped_epoch": horizon,
        "lr_horizon_complete": True,
        "selection_resolved": True,
        "early_stopped": False,
        "g3_dataset_size": "native-50m",
        "g3_protocol_sha256": APPROVED_PROTOCOL_SHA256,
        "g3_representation": {
            "catalog_representation": "learned_id",
            "content_gate": "fixed",
            "extra_item_id_dim": None,
            "gate_hidden_dim": None,
            "history_hidden_dim": None,
            "history_representation": "content",
            "metadata": [],
            "metadata_dim": None,
        },
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError(f"RQ1 runtime metadata differs for {row['id']}")
    best_epoch = metadata.get("best_epoch")
    if type(best_epoch) is not int or not 1 <= best_epoch <= horizon:
        raise ValueError(f"RQ1 best epoch is invalid for {row['id']}")
    if metadata.get("best_epoch_at_cap") is not (best_epoch == horizon):
        raise ValueError(f"RQ1 best-epoch cap flag is invalid for {row['id']}")
    traces = metadata.get("lr_group_traces")
    if not isinstance(traces, dict) or any(
        not isinstance(traces.get(group), list)
        or len(traces[group]) != horizon
        or traces[group][-1] != 0.0
        for group in ("embedding", "deep")
    ):
        raise ValueError(f"RQ1 schedule trace is incomplete for {row['id']}")

    metrics_path = run_directory / "final_metrics.json"
    metrics = _load_json(metrics_path)
    if set(metrics) != {*_METRIC_NAMES, "num_users"}:
        raise ValueError(f"RQ1 metric schema differs for {row['id']}")
    ranking_path = run_directory / "ranking_evidence.pt"
    rankings_path = run_directory / "top_item_rankings.json"
    recomputed = _recompute_metrics(context_path, ranking_path, rankings_path)
    if any(abs(float(metrics[key]) - recomputed[key]) > 1e-15 for key in metrics):
        raise ValueError(f"RQ1 metrics differ from ranking evidence for {row['id']}")
    slices = _ranking_slices(
        context_path=context_path,
        ranking_path=ranking_path,
        rankings_path=rankings_path,
        item_counts=item_counts,
    )
    efficiency = _efficiency(
        metadata=metadata,
        log_path=run_directory / "sweep.log",
        queue_wall_seconds=float(queue_job["finished_at"])
        - float(queue_job["dispatched_at"]),
    )
    diagnostics = _diagnostics(
        run_directory / "g3_training_diagnostics.json",
        horizon=horizon,
        best_epoch=best_epoch,
    )
    artifact_filenames = (
        ("job_contract", "g3_rq1_job.json"),
        ("training_metadata", "training_metadata.json"),
        ("final_metrics", "final_metrics.json"),
        ("ranking_evidence", "ranking_evidence.pt"),
        ("top_item_rankings", "top_item_rankings.json"),
        ("training_diagnostics", "g3_training_diagnostics.json"),
        ("sweep_log", "sweep.log"),
    )
    verify_unique_completed_run(
        root / "generated/training-queue-service",
        run_name=str(row["run_name"]),
        expected_job_id=job_id,
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
        "row_id": row["id"],
        "run_name": row["run_name"],
        "embedding_learning_rate": training["embedding_learning_rate"],
        "deep_learning_rate": training["deep_learning_rate"],
        "horizon_epochs": horizon,
        "best_epoch": best_epoch,
        "epochs_trained": metadata["epochs_trained"],
        "selection_resolved": True,
        "queue_wall_seconds": efficiency["queue_wall_seconds"],
        "metrics": metrics,
        "metric_provenance": {
            "recomputed_from_ranking_evidence": True,
            "absolute_tolerance": 1e-15,
            "num_users": int(recomputed["num_users"]),
        },
        "slices": slices,
        "efficiency": efficiency,
        "diagnostics": diagnostics,
        "queue_job": _file_fact(root, completed_path) | {"job_id": job_id},
        "artifacts": artifacts,
    }


def _control_comparison(
    *,
    root: Path,
    context_path: Path,
    feature_path: Path,
    control: dict[str, object],
) -> dict[str, object]:
    directory = root / "generated/logs" / str(control["run_name"])
    artifacts = control.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("RQ1 selected control has no bound artifacts")
    for name, filename in (
        ("training_metadata", "training_metadata.json"),
        ("final_metrics", "final_metrics.json"),
        ("ranking_evidence", "ranking_evidence.pt"),
        ("top_item_rankings", "top_item_rankings.json"),
        ("training_diagnostics", "g3_training_diagnostics.json"),
        ("sweep_log", "sweep.log"),
    ):
        if _file_fact(root, directory / filename) != artifacts.get(name):
            raise ValueError(f"RQ1 selected control {name} artifact changed")
    recomputed = _recompute_metrics(
        context_path,
        directory / "ranking_evidence.pt",
        directory / "top_item_rankings.json",
    )
    if any(
        abs(float(control["metrics"][key]) - recomputed[key]) > 1e-15
        for key in control["metrics"]
    ):
        raise ValueError("RQ1 selected control metrics changed")
    item_counts = load_training_item_counts(feature_path)
    slices = _ranking_slices(
        context_path=context_path,
        ranking_path=directory / "ranking_evidence.pt",
        rankings_path=directory / "top_item_rankings.json",
        item_counts=item_counts,
    )
    metadata = _load_json(directory / "training_metadata.json")
    efficiency = _efficiency(
        metadata=metadata,
        log_path=directory / "sweep.log",
        queue_wall_seconds=float(control["queue_wall_seconds"]),
    )
    diagnostics = _diagnostics(
        directory / "g3_training_diagnostics.json",
        horizon=int(control["horizon_epochs"]),
        best_epoch=int(control["best_epoch"]),
    )
    return {
        "row_id": control["row_id"],
        "metrics": control["metrics"],
        "slices": slices,
        "efficiency": efficiency,
        "diagnostics": diagnostics,
    }


def _ranking_slices(
    *,
    context_path: Path,
    ranking_path: Path,
    rankings_path: Path,
    item_counts: Mapping[int, int],
    rank_source: Literal["matched_snapshot", "ranking_evidence"] = "matched_snapshot",
) -> dict[str, object]:
    if rank_source not in {"matched_snapshot", "ranking_evidence"}:
        raise ValueError("unknown RQ1 slice rank source")
    evidence = load_ranking_evidence(context_path, ranking_path)
    snapshot = _load_json(rankings_path)
    rows = snapshot.get("rankings")
    expected_catalog_sha256 = hashlib.sha256(
        json.dumps(list(item_counts), separators=(",", ":")).encode()
    ).hexdigest()
    if (
        set(snapshot)
        != {
            "schema_version",
            "catalog_sha256",
            "catalog_size",
            "exclude_seen",
            "max_k",
            "rankings",
        }
        or snapshot.get("schema_version") != 1
        or snapshot.get("catalog_sha256") != expected_catalog_sha256
        or snapshot.get("catalog_size") != len(item_counts)
        or snapshot.get("exclude_seen") is not False
        or snapshot.get("max_k") != 100
        or evidence.max_k != 100
        or not isinstance(rows, list)
    ):
        raise ValueError("RQ1 ranking snapshot has a different catalog")
    rankings: dict[int, list[int]] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"user_id", "item_ids"}:
            raise ValueError("RQ1 ranking snapshot row is invalid")
        user_id = int(row["user_id"])
        if user_id in rankings or not isinstance(row["item_ids"], list):
            raise ValueError("RQ1 ranking snapshot contains duplicate users")
        rankings[user_id] = [int(item_id) for item_id in row["item_ids"]]
    users = [int(user_id) for user_id in evidence.user_ids.tolist()]
    if users != list(rankings) or len(users) != len(set(users)):
        raise ValueError("RQ1 ranking snapshot has a different user identity")
    relevant_items = [int(item_id) for item_id in evidence.relevant_item_ids.tolist()]
    relevance_offsets = [int(value) for value in evidence.relevance_offsets.tolist()]
    history_offsets = [int(value) for value in evidence.history_offsets.tolist()]
    relevant_ranks = [int(value) for value in evidence.relevant_ranks.tolist()]
    relevant: dict[int, set[int]] = {}
    histories = {}
    for position, user_id in enumerate(users):
        start, end = relevance_offsets[position : position + 2]
        targets = relevant_items[start:end]
        if len(targets) != len(set(targets)):
            raise ValueError("RQ1 ranking evidence has duplicate targets")
        relevant[user_id] = set(targets)
        histories[user_id] = history_offsets[position + 1] - history_offsets[position]
        positions = {
            item_id: rank
            for rank, item_id in enumerate(rankings[user_id], start=1)
        }
        if rank_source == "matched_snapshot" and relevant_ranks[start:end] != [
            positions.get(item_id, 0) for item_id in targets
        ]:
            raise ValueError("RQ1 ranking evidence ranks differ from the snapshot")
    expected_frequencies = [item_counts[item_id] for item_id in relevant_items]
    if [int(value) for value in evidence.relevant_train_frequencies.tolist()] != (
        expected_frequencies
    ):
        raise ValueError("RQ1 ranking evidence has different train frequencies")
    report = compute_ranking_slices(
        rankings=rankings,
        relevant_items=relevant,
        training_item_counts=item_counts,
        training_history_lengths=histories,
    )
    if rank_source == "ranking_evidence":
        return _evidence_item_frequency_slices(
            users=users,
            relevant_items=relevant_items,
            relevance_offsets=relevance_offsets,
            relevant_ranks=relevant_ranks,
            item_counts=item_counts,
        )
    result = {}
    for name in ("head", "mid", "tail"):
        item_slice = report.slice("item_frequency", name)
        result[name] = {
            "num_users": item_slice.num_users,
            "num_targets": item_slice.num_targets,
            **dict(item_slice.metrics),
            "item_membership_sha256": _canonical_sha256(list(item_slice.item_ids)),
        }
    return result


def _evidence_item_frequency_slices(
    *,
    users: Sequence[int],
    relevant_items: Sequence[int],
    relevance_offsets: Sequence[int],
    relevant_ranks: Sequence[int],
    item_counts: Mapping[int, int],
) -> dict[str, object]:
    groups = _terciles(item_counts, names=("tail", "mid", "head"))
    result = {}
    for name in ("head", "mid", "tail"):
        members = set(groups[name])
        recalls = {cutoff: 0.0 for cutoff in (10, 50, 100)}
        num_users = 0
        num_targets = 0
        for position, _ in enumerate(users):
            start, end = relevance_offsets[position : position + 2]
            ranks = [
                rank
                for item_id, rank in zip(
                    relevant_items[start:end], relevant_ranks[start:end], strict=True
                )
                if item_id in members
            ]
            if not ranks:
                continue
            num_users += 1
            num_targets += len(ranks)
            for cutoff in recalls:
                recalls[cutoff] += sum(0 < rank <= cutoff for rank in ranks) / len(ranks)
        result[name] = {
            "num_users": num_users,
            "num_targets": num_targets,
            **{
                f"recall@{cutoff}": total / num_users if num_users else 0.0
                for cutoff, total in recalls.items()
            },
            "item_membership_sha256": _canonical_sha256(list(groups[name])),
        }
    return result


def _tied_control_comparison(
    *, root: Path, feature_path: Path
) -> dict[str, object]:
    manifest_path = root / _TIED_CONTROL_MANIFEST_PATH
    if _file_fact(root, manifest_path)["sha256"] != (
        APPROVED_PROTOCOL.control.manifest_sha256
    ):
        raise ValueError("RQ1 tied original control manifest changed")
    manifest = _load_json(manifest_path)
    selected = manifest.get("selection")
    configuration = manifest.get("seed_42_configuration")
    if (
        not isinstance(selected, dict)
        or not isinstance(configuration, dict)
        or selected.get("run_name") != APPROVED_PROTOCOL.control.run_name
        or selected.get("best_epoch") != APPROVED_PROTOCOL.control.best_epoch
        or configuration.get("run_name") != APPROVED_PROTOCOL.control.run_name
    ):
        raise ValueError("RQ1 tied original control selection changed")

    evidence_path = root / _TIED_CONTROL_EVIDENCE_PATH
    if _file_fact(root, evidence_path)["sha256"] != _TIED_CONTROL_EVIDENCE_SHA256:
        raise ValueError("RQ1 tied original control evidence changed")
    sidecar = evidence_path.with_suffix(".sha256")
    if sidecar.read_text() != _TIED_CONTROL_EVIDENCE_SHA256:
        raise ValueError("RQ1 tied original control evidence sidecar changed")
    source = _load_json(evidence_path)
    source_run = source["selected_runs"]["control_next_item"]
    source_metrics = source["overall"]["rows"]["control_next_item"]
    if (
        source_run["run_name"] != APPROVED_PROTOCOL.control.run_name
        or source_run["restored_best_epoch"] != APPROVED_PROTOCOL.control.best_epoch
        or abs(
            float(source_metrics["recall@100"])
            - APPROVED_PROTOCOL.control.recall_at_100
        )
        > 1e-15
    ):
        raise ValueError("RQ1 tied original control evidence selection changed")
    directory = root / "generated/logs" / APPROVED_PROTOCOL.control.run_name
    artifacts = source_run["ledger_row_artifacts"] | source_run[
        "evaluation_artifacts"
    ]
    for name, filename in (
        ("job_contract", "g4_job.json"),
        ("training_metadata", "training_metadata.json"),
        ("sweep_log", "sweep.log"),
        ("final_metrics", "final_metrics.json"),
        ("ranking_evidence", "ranking_evidence.pt"),
        ("top_item_rankings", "top_item_rankings.json"),
    ):
        current = _file_fact(root, directory / filename)
        bound = artifacts[name]
        if any(current[key] != bound[key] for key in ("path", "sha256")):
            raise ValueError(f"RQ1 tied original control {name} changed")
    context_path = root / "generated/logs/.ranking-evidence/g4-native50m/context.pt"
    recomputed = _recompute_metrics(
        context_path,
        directory / "ranking_evidence.pt",
        directory / "top_item_rankings.json",
    )
    metrics = _load_json(directory / "final_metrics.json")
    if any(abs(float(metrics[key]) - recomputed[key]) > 1e-15 for key in metrics):
        raise ValueError("RQ1 tied original control metrics changed")
    if any(
        abs(float(source_metrics[key]) - recomputed[key]) > 1e-15
        for key in source_metrics
    ):
        raise ValueError("RQ1 tied original evidence metrics changed")
    queue_path = (
        root
        / "generated/training-queue-service/completed"
        / f"{_TIED_CONTROL_QUEUE_JOB_ID}.json"
    )
    queue = _load_json(queue_path)
    if (
        queue.get("id") != _TIED_CONTROL_QUEUE_JOB_ID
        or queue.get("run") != APPROVED_PROTOCOL.control.run_name
        or queue.get("exit_code") != 0
    ):
        raise ValueError("RQ1 tied original queue completion changed")
    metadata = _load_json(directory / "training_metadata.json")
    return {
        "row_id": source_run["row_id"],
        "run_name": source_run["run_name"],
        "metrics": metrics,
        "slices": _ranking_slices(
            context_path=context_path,
            ranking_path=directory / "ranking_evidence.pt",
            rankings_path=directory / "top_item_rankings.json",
            item_counts=load_training_item_counts(feature_path),
        ),
        "efficiency": _efficiency(
            metadata=metadata,
            log_path=directory / "sweep.log",
            queue_wall_seconds=float(queue["finished_at"])
            - float(queue["dispatched_at"]),
        ),
        "queue_job": _file_fact(root, queue_path)
        | {"job_id": _TIED_CONTROL_QUEUE_JOB_ID},
    }


def _verify_context_identity(
    *, root: Path, treatment_run_name: str, tied_run_name: str
) -> None:
    g3 = load_ranking_evidence(
        root / "generated/logs/.ranking-evidence/g3-native50m/context.pt",
        root / "generated/logs" / treatment_run_name / "ranking_evidence.pt",
    )
    tied = load_ranking_evidence(
        root / "generated/logs/.ranking-evidence/g4-native50m/context.pt",
        root / "generated/logs" / tied_run_name / "ranking_evidence.pt",
    )
    for name in (
        "user_ids",
        "history_item_ids",
        "history_offsets",
        "relevant_item_ids",
        "relevance_offsets",
        "relevant_train_frequencies",
    ):
        if not getattr(g3, name).equal(getattr(tied, name)):
            raise ValueError(f"RQ1 tied and local ranking contexts differ: {name}")


def load_training_item_counts(path: Path) -> dict[int, int]:
    table = pl.read_parquet(path, columns=("compact_item_id", "training_count"))
    identifiers = [int(value) for value in table["compact_item_id"].to_list()]
    counts = {
        int(item_id): int(count)
        for item_id, count in table.iter_rows()
        if int(item_id) != 0
    }
    if (
        identifiers.count(0) not in (0, 1)
        or len(counts) != table.height - identifiers.count(0)
        or sorted(counts) != list(range(1, len(counts) + 1))
        or any(count < 0 for count in counts.values())
        or len(counts) < 3
    ):
        raise ValueError("RQ1 feature data has duplicate or absent catalog counts")
    return counts


def _efficiency(
    *, metadata: Mapping[str, object], log_path: Path, queue_wall_seconds: float
) -> dict[str, object]:
    text = log_path.read_text()
    training_times = [float(value) for value in _TIMING_PATTERN.findall(text)]
    horizon = int(metadata["epochs_trained"])
    if len(training_times) != horizon or any(value <= 0 for value in training_times):
        raise ValueError("RQ1 log does not contain every epoch training time")
    memory = [float(value) for value in _MEMORY_PATTERN.findall(text)]
    parameters = [int(float(value)) for value in _PARAMETER_PATTERN.findall(text)]
    if not memory or not parameters or len(set(parameters)) != 1:
        raise ValueError("RQ1 log lacks stable resource measurements")
    training_seconds = sum(training_times)
    targets = int(metadata["targets_per_epoch"]) * horizon
    observed_final = _final_evaluation_upper_bound(text)
    return {
        "best_epoch": int(metadata["best_epoch"]),
        "declared_horizon_epochs": int(metadata["lr_schedule_horizon_epochs"]),
        "completed_horizon_epochs": horizon,
        "queue_wall_seconds": queue_wall_seconds,
        "logged_training_seconds": training_seconds,
        "examples_per_second": targets / training_seconds,
        "targets_per_second": targets / training_seconds,
        "peak_gpu_memory_gb": max(memory),
        "parameter_count": parameters[0],
        "full_catalog_encoding_scoring_seconds": None,
        "full_catalog_observed_upper_bound_seconds": observed_final,
        "full_catalog_timing_limitation": (
            "The run did not instrument catalog encoding and scoring separately; "
            "the observed upper bound spans the final epoch callback through "
            "checkpoint restore, full-catalog scoring, and evidence persistence."
        ),
    }


def _final_evaluation_upper_bound(text: str) -> float:
    epoch_times = []
    final_time = None
    for line in text.splitlines():
        timestamp = _timestamp(line)
        if timestamp is None:
            continue
        if " finished timing.train_epoch_time=" in line:
            epoch_times.append(timestamp)
        if "Final metrics (" in line:
            final_time = timestamp
    if not epoch_times or final_time is None or final_time < epoch_times[-1]:
        raise ValueError("RQ1 log lacks a valid full-catalog evaluation span")
    return (final_time - epoch_times[-1]).total_seconds()


def _timestamp(line: str) -> datetime | None:
    match = _TIMESTAMP_PATTERN.match(line)
    if match is None:
        return None
    return datetime.strptime(
        f"{match.group(1)}.{match.group(2)}", "%Y-%m-%d %H:%M:%S.%f"
    )


def _diagnostics(path: Path, *, horizon: int, best_epoch: int) -> dict[str, object]:
    document = _load_json(path)
    epochs = document.get("epochs")
    if (
        document.get("schema_version") != 1
        or not isinstance(epochs, list)
        or [epoch.get("epoch") for epoch in epochs] != list(range(horizon))
    ):
        raise ValueError("RQ1 diagnostic epochs are incomplete")
    nonfinite = _values_named(document, "nonfinite_count")
    if not nonfinite or any(value != 0 for value in nonfinite):
        raise ValueError("RQ1 diagnostics contain nonfinite measurements")
    restored = epochs[best_epoch - 1]
    gradients = restored.get("component_gradient_norms")
    if not isinstance(gradients, dict):
        raise ValueError("RQ1 diagnostics lack component gradients")
    means = {
        name: float(value["mean"])
        for name, value in gradients.items()
        if isinstance(value, dict) and "mean" in value
    }
    if set(means) != {"history_encoder", "catalog_encoder", "sequence_model"} or any(
        not math.isfinite(value) or value <= 0 for value in means.values()
    ):
        raise ValueError("RQ1 diagnostic gradients do not show finite gradient flow")
    return {
        "epochs": horizon,
        "restored_epoch": best_epoch,
        "nonfinite_measurement_count": sum(int(value) for value in nonfinite),
        "restored_epoch_component_gradient_norm_mean": means,
    }


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


def _verify_matched_protocol(
    root: Path,
    control: Mapping[str, object],
    treatment: Mapping[str, object],
) -> None:
    directories = {
        "control": root / "generated/logs" / str(control["run_name"]),
        "treatment": root / "generated/logs" / str(treatment["run_name"]),
    }
    metadata = {
        name: _load_json(directory / "training_metadata.json")
        for name, directory in directories.items()
    }
    invariants = {}
    for name, document in metadata.items():
        value = document.get("transfer_invariants")
        if not isinstance(value, dict):
            raise ValueError(f"RQ1 {name} metadata lacks transfer invariants")
        invariants[name] = {
            key: nested
            for key, nested in value.items()
            if key not in {"g3_representation", "lr_schedule_horizon_epochs"}
        }
    if invariants["control"] != invariants["treatment"]:
        raise ValueError("RQ1 control and treatment differ beyond representation")
    representations = {
        name: document["g3_representation"] for name, document in metadata.items()
    }
    if representations["control"]["history_representation"] != "learned_id":
        raise ValueError("RQ1 comparator is not the learned-ID history control")
    if representations["treatment"]["history_representation"] != "content":
        raise ValueError("RQ1 treatment is not content-only history")
    if any(
        value["catalog_representation"] != "learned_id"
        for value in representations.values()
    ):
        raise ValueError("RQ1 changed the catalog representation")


def _rate_boundary(
    value: float, bounds: tuple[float, float]
) -> dict[str, object]:
    lower, upper = bounds
    if not lower <= value <= upper:
        raise ValueError("selected RQ1 learning rate is outside approved bounds")
    position = (value - lower) / (upper - lower)
    direction = "lower" if position <= 0.1 else "upper" if position >= 0.9 else None
    return {
        "selected": value,
        "bounds": list(bounds),
        "normalized_position": position,
        "direction": direction,
    }


def _slice_table(
    tied: Mapping[str, object],
    control: Mapping[str, object],
    treatment: Mapping[str, object],
) -> str:
    lines = [
        "| item-frequency slice | users | targets | tied original recall@100 | untied learned item ID recall@100 | frozen content history recall@100 | change vs untied | evidence status |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |",
    ]
    for name in ("head", "mid", "tail"):
        reference = float(control[name]["recall@100"])
        candidate = float(treatment[name]["recall@100"])
        if (
            tied[name]["num_users"] != control[name]["num_users"]
            or tied[name]["num_targets"] != control[name]["num_targets"]
            or tied[name]["item_membership_sha256"]
            != control[name]["item_membership_sha256"]
            or control[name]["num_users"] != treatment[name]["num_users"]
            or control[name]["num_targets"] != treatment[name]["num_targets"]
        ):
            raise ValueError("RQ1 slice denominators differ")
        change = 100.0 * (candidate / reference - 1.0)
        lines.append(
            f"| {name} | {control[name]['num_users']} | "
            f"{control[name]['num_targets']} | "
            f"{float(tied[name]['recall@100']):.3f} | {reference:.3f} | "
            f"{candidate:.3f} | {change:+.1f}% | "
            "descriptive only; no slice-specific repeat calibration |"
        )
    return "\n".join(lines)


def _decision_table(
    boundary: Mapping[str, object], promotion: Mapping[str, object]
) -> str:
    embedding = boundary.get("embedding_learning_rate")
    deep = boundary.get("deep_learning_rate")
    horizon = boundary.get("horizon")
    comparators = promotion.get("comparators")
    if not all(isinstance(value, dict) for value in (embedding, deep, horizon)):
        raise ValueError("RQ1 report boundary decision is invalid")
    if not isinstance(comparators, dict):
        raise ValueError("RQ1 report promotion comparators are invalid")
    lines = [
        "| frozen decision | value |",
        "| :--- | :--- |",
        (
            "| embedding learning-rate boundary | "
            f"selected {embedding['selected']}; bounds {embedding['bounds']}; "
            f"normalized position {embedding['normalized_position']}; "
            f"direction {embedding['direction']} |"
        ),
        (
            "| deep learning-rate boundary | "
            f"selected {deep['selected']}; bounds {deep['bounds']}; "
            f"normalized position {deep['normalized_position']}; "
            f"direction {deep['direction']} |"
        ),
        (
            "| horizon boundary | "
            f"selected {horizon['selected_epochs']}; restored "
            f"{horizon['restored_best_epoch']}; extend to "
            f"{horizon['extend_to_epochs']} |"
        ),
        f"| extension required | {boundary['extension_required']} |",
        f"| promotion selected | {promotion['promoted']} |",
        f"| promotion route | {promotion['route']} |",
        (
            "| aggregate-improvement comparator | "
            f"{_single_comparator(comparators, 'aggregate_improvement')} |"
        ),
        (
            "| aggregate tail-tradeoff band comparator | "
            f"{_single_comparator(comparators, 'aggregate_tail_tradeoff_band')} |"
        ),
        (
            "| tail-improvement comparator | "
            f"{_single_comparator(comparators, 'tail_improvement')} |"
        ),
        f"| promotion rule outcome | {promotion['reason']} |",
        (
            "| tail-route evidence status | descriptive only; the predeclared "
            "route uses observed tail ordering without slice-specific repeat "
            "calibration |"
        ),
    ]
    return "\n".join(lines)


def _single_comparator(
    comparators: Mapping[str, object], role: str
) -> str:
    value = comparators.get(role)
    if (
        not isinstance(value, list)
        or len(value) != 1
        or not isinstance(value[0], str)
    ):
        raise ValueError(f"RQ1 report promotion role {role} is invalid")
    return value[0]


def _verify_slice_identity(
    control: Mapping[str, object], treatment: Mapping[str, object]
) -> None:
    if set(control) != {"head", "mid", "tail"} or set(treatment) != set(control):
        raise ValueError("RQ1 item-frequency slice schema differs")
    for name in control:
        identity = ("num_users", "num_targets", "item_membership_sha256")
        if any(control[name][field] != treatment[name][field] for field in identity):
            raise ValueError(f"RQ1 {name} slice identity or denominator differs")


def _efficiency_table(
    tied: Mapping[str, object],
    control: Mapping[str, object],
    treatment: Mapping[str, object],
) -> str:
    lines = [
        "| variant | best / horizon epoch | queue wall, s | targets/s | peak GPU memory, GiB | parameters | full-catalog evaluation upper bound, s |",
        "| :--- | :---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, value in (
        ("tied original learned item ID", tied),
        ("untied learned item ID", control),
        ("frozen content history", treatment),
    ):
        lines.append(
            f"| {name} | {value['best_epoch']} / "
            f"{value['declared_horizon_epochs']} | "
            f"{float(value['queue_wall_seconds']):.1f} | "
            f"{float(value['targets_per_second']):.0f} | "
            f"{float(value['peak_gpu_memory_gb']):.3f} | "
            f"{value['parameter_count']} | "
            f"{float(value['full_catalog_observed_upper_bound_seconds']):.3f} |"
        )
    return "\n".join(lines)


def _full_catalog_timing_table(
    tied: Mapping[str, object],
    control: Mapping[str, object],
    treatment: Mapping[str, object],
) -> str:
    values = (tied, control, treatment)
    if any(
        value.get("full_catalog_encoding_scoring_seconds") is not None
        for value in values
    ):
        raise ValueError("RQ1 report received an unverified exact catalog timing")
    limitations = {value.get("full_catalog_timing_limitation") for value in values}
    if len(limitations) != 1 or not all(
        isinstance(value, str) and value for value in limitations
    ):
        raise ValueError("RQ1 report catalog timing limitation differs by variant")
    return "\n".join(
        (
            "| timing field | status |",
            "| :--- | :--- |",
            "| exact full-catalog encoding/scoring time | unavailable in the saved runs |",
            (
                "| timestamped ranking-context/ranking events | advisory-lock wait "
                "only; they do not time catalog encoding or scoring |"
            ),
            (
                "| reported efficiency value | callback-to-final-metrics upper "
                "bound only; it also includes checkpoint restore and evidence "
                "persistence |"
            ),
            (
                "| smallest recovery | checkpoint-only timing replay with cached "
                "evaluation data: keep query encoding outside the interval and "
                "device-synchronize around catalog encoding and full-catalog score "
                "computation; no training or optimizer step |"
            ),
        )
    )


def _overall_table(
    tied: Mapping[str, object],
    control: Mapping[str, object],
    treatment: Mapping[str, object],
) -> str:
    tied_recall = float(tied["metrics"]["recall@100"])
    control_recall = float(control["metrics"]["recall@100"])
    treatment_recall = float(treatment["metrics"]["recall@100"])
    tied_ndcg = float(tied["metrics"]["ndcg@100"])
    control_ndcg = float(control["metrics"]["ndcg@100"])
    treatment_ndcg = float(treatment["metrics"]["ndcg@100"])
    return "\n".join(
        (
            "| variant | recall@100 | vs untied | vs tied original | ndcg@100 | interpretation |",
            "| :--- | ---: | ---: | ---: | ---: | :--- |",
            f"| tied original learned item ID | {tied_recall:.3f} | — | reference | {tied_ndcg:.3f} | aggregate baseline |",
            f"| untied learned item ID | {control_recall:.3f} | reference | {100.0 * (control_recall / tied_recall - 1.0):+.1f}% | {control_ndcg:.3f} | local scientific reference |",
            f"| frozen content history | {treatment_recall:.3f} | {100.0 * (treatment_recall / control_recall - 1.0):+.1f}% | {100.0 * (treatment_recall / tied_recall - 1.0):+.1f}% | {treatment_ndcg:.3f} | beyond untied band; tied difference unresolved within tied band |",
        )
    )


def _metrics(run: Mapping[str, object]) -> Mapping[str, object]:
    metrics = run.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("RQ1 run has no metrics")
    return metrics


def _document(payload: Mapping[str, object]) -> dict[str, object]:
    document = dict(payload)
    document["sha256"] = _canonical_sha256(document)
    return document


def _validate_document(
    document: dict[str, object], *, enforce_approved: bool
) -> dict[str, object]:
    if (
        document.get("schema_version") != 1
        or document.get("kind") != "g3_rq1_content_input_evidence"
        or document.get("protocol_sha256") != APPROVED_PROTOCOL_SHA256
        or document.get("sha256")
        != _canonical_sha256(
            {key: value for key, value in document.items() if key != "sha256"}
        )
    ):
        raise ValueError("RQ1 evidence identity or hash is invalid")
    if enforce_approved and document["sha256"] != APPROVED_RQ1_EVIDENCE_SHA256:
        raise ValueError("RQ1 evidence does not match the approved immutable hash")
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
