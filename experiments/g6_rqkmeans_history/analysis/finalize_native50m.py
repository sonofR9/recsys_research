from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Sequence

from dcn.eval.ranking_evidence import load_ranking_evidence
from experiments.g6_rqkmeans_history.protocol.manifest import RANKING_EVIDENCE_GROUP
from experiments.g6_rqkmeans_history.analysis.learning_curves import (
    load_validation_curve,
)
from experiments.g6_rqkmeans_history.analysis.select_surfaces import (
    artifact_identity,
    _validate_diagnostics,
    _validate_rq1_initialization,
)
from experiments.g6_rqkmeans_history.protocol.collision_policy import (
    CollisionSearchJob,
    collision_search_manifest,
)
from experiments.g6_rqkmeans_history.protocol.confirmation import (
    SURFACE_PATH,
    ConfirmationJob,
    load_confirmation_manifest,
)
from experiments.g6_rqkmeans_history.protocol.rq1_manifest import (
    rq1_search_manifest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
LOGS_ROOT = PROJECT_ROOT / "generated/logs"
EVIDENCE_PATH = (
    PROJECT_ROOT
    / "experiments/g6_rqkmeans_history/evidence/rq1_rq3_confirmation_native50m.json"
)
TERMINAL_PATH = (
    PROJECT_ROOT
    / "experiments/g6_rqkmeans_history/evidence/rq2_rq3_selection_native50m.json"
)
METRICS = ("recall@100", "ndcg@100", "mrr@100", "coverage@100")


def mean_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        raise ValueError("metric mean requires rows")
    return {
        name: statistics.fmean(row["metrics"][name] for row in rows)
        for name in METRICS
    }


def choose_terminal(
    *,
    rq0: dict[str, float],
    suffix: dict[str, float],
    none: dict[str, float],
    suffix_is_rq0: bool,
) -> str:
    eligible_suffix = suffix
    effective_suffix_name = "suffix"
    if suffix["recall@100"] < rq0["recall@100"] - 0.002:
        eligible_suffix = rq0
        effective_suffix_name = "rq0"
    candidates = [(effective_suffix_name, eligible_suffix), ("none", none)]
    best_recall = max(metrics["recall@100"] for _, metrics in candidates)
    recall_tied = [
        candidate
        for candidate in candidates
        if candidate[1]["recall@100"] >= best_recall - 0.002
    ]
    selected_name, selected = max(
        recall_tied,
        key=lambda candidate: (
            candidate[1]["ndcg@100"],
            candidate[0] in {"suffix", "rq0"},
        ),
    )
    if selected_name == "suffix" and suffix_is_rq0:
        return "rq0"
    promotes = (
        selected["recall@100"] > rq0["recall@100"] + 0.002
        and selected["ndcg@100"] >= rq0["ndcg@100"] - 0.002
    )
    return selected_name if promotes else "rq0"


def finalize(logs_root: Path = LOGS_ROOT) -> tuple[dict[str, Any], dict[str, Any]]:
    surface_payload = (PROJECT_ROOT / SURFACE_PATH).read_bytes()
    surface = json.loads(surface_payload)
    confirmation = load_confirmation_manifest(PROJECT_ROOT / SURFACE_PATH)
    for section, names in (
        ("rq1", ("random", "content_pca")),
        ("rq2_rq3", ("suffix", "none")),
    ):
        for name in names:
            selected = surface[section]["selected"][name]
            matches = [
                row
                for row in surface[section]["rows"]
                if row["job_id"] == selected["job_id"]
            ]
            if len(matches) != 1 or matches[0] != selected:
                raise ValueError(f"{section} selected row is not frozen in the surface")
            _authenticate_surface_row(selected, logs_root)
    confirmation_rows = {
        job.id: _load_confirmation(job, confirmation.sha256, logs_root)
        for job in confirmation.jobs
    }

    rq1 = {}
    for initialization in ("random", "content_pca"):
        seed42 = surface["rq1"]["selected"][initialization]
        repeats = sorted(
            (
                confirmation_rows[job.id]
                for job in confirmation.jobs
                if job.family == "rq1"
                and job.variant.startswith(f"{initialization}_t")
            ),
            key=lambda row: row["seed"],
        )
        rows = [seed42, *repeats]
        rq1[initialization] = {
            "source_job_id": seed42["job_id"],
            "mean_metrics": mean_metrics(rows),
            "seeds": [42, 43, 44, 45],
            "rows": rows,
            "convergence": _convergence(rows, logs_root),
        }
    noninferior = all(
        rq1["content_pca"]["mean_metrics"][name]
        >= rq1["random"]["mean_metrics"][name] - 0.002
        for name in ("recall@100", "ndcg@100")
    )
    faster = noninferior and all(
        content["first_epoch_at_95_percent"]
        < random["first_epoch_at_95_percent"]
        and content["normalized_auc"] > random["normalized_auc"]
        for random, content in zip(
            rq1["random"]["convergence"]["rows"],
            rq1["content_pca"]["convergence"]["rows"],
            strict=True,
        )
    )

    collision_manifest = collision_search_manifest()
    anchor = next(
        job
        for job in collision_manifest.jobs
        if job.policy == "suffix" and job.coordinate.trial == 0
    )
    anchor_row = _surface_row(surface, anchor.id)
    _authenticate_surface_row(anchor_row, logs_root)
    selected_jobs = {
        policy: _job_by_id(surface["rq2_rq3"]["selected"][policy]["job_id"])
        for policy in ("suffix", "none")
    }
    systems = {
        "rq0": _collision_system(
            anchor,
            anchor_row,
            confirmation,
            confirmation_rows,
        ),
        **{
            policy: _collision_system(
                job,
                surface["rq2_rq3"]["selected"][policy],
                confirmation,
                confirmation_rows,
            )
            for policy, job in selected_jobs.items()
        },
    }
    terminal_name = choose_terminal(
        rq0=systems["rq0"]["mean_metrics"],
        suffix=systems["suffix"]["mean_metrics"],
        none=systems["none"]["mean_metrics"],
        suffix_is_rq0=selected_jobs["suffix"].id == anchor.id,
    )
    terminal_job = anchor if terminal_name == "rq0" else selected_jobs[terminal_name]
    terminal_row = _surface_row(surface, terminal_job.id)
    evidence = {
        "schema": "g6-rq1-rq3-confirmation/v1",
        "dataset_size": "native-50m",
        "surface_sha256": hashlib.sha256(surface_payload).hexdigest(),
        "confirmation_manifest_sha256": confirmation.sha256,
        "rq1": {
            **rq1,
            "content_noninferior": noninferior,
            "content_faster": faster,
        },
        "rq2_rq3": {
            "systems": systems,
            "terminal": terminal_name,
            "terminal_job_id": terminal_job.id,
        },
    }
    terminal = _terminal_document(terminal_job, terminal_row)
    _write_immutable(EVIDENCE_PATH, evidence)
    _write_immutable(TERMINAL_PATH, terminal)
    return evidence, terminal


def _load_confirmation(
    job: ConfirmationJob, manifest_sha256: str, logs_root: Path
) -> dict[str, Any]:
    directory = logs_root / job.run_name
    paths = {
        "job_contract": directory / "g6_confirmation_job.json",
        "final_metrics": directory / "final_metrics.json",
        "training_metadata": directory / "training_metadata.json",
        "ranking_evidence": directory / "ranking_evidence.pt",
        "sid_diagnostics": directory / "semantic_id_diagnostics.json",
        "sweep_log": directory / "sweep.log",
    }
    if any(not path.is_file() for path in paths.values()):
        raise ValueError(f"{job.run_name}: confirmation artifacts are incomplete")
    contract = json.loads(paths["job_contract"].read_text())
    if contract != {"manifest_sha256": manifest_sha256, "job": job.to_dict()}:
        raise ValueError(f"{job.run_name}: confirmation contract changed")
    metrics = _numeric(json.loads(paths["final_metrics"].read_text()))
    metadata = json.loads(paths["training_metadata"].read_text())
    expected = {
        "dataset_size": "50m",
        "batch_size": 256,
        "seed": job.seed,
        "embedding_learning_rate": job.embedding_learning_rate,
        "deep_learning_rate": job.deep_learning_rate,
        "num_epochs": 15,
        "lr_schedule_horizon_epochs": 15,
        "lr_horizon_complete": True,
    }
    if any(metadata.get(name) != value for name, value in expected.items()):
        raise ValueError(f"{job.run_name}: confirmation metadata changed")
    diagnostics = json.loads(paths["sid_diagnostics"].read_text())
    if job.family == "rq1":
        source = next(
            source
            for source in rq1_search_manifest().jobs
            if source.id == job.source_job_id
        )
        _validate_rq1_initialization(source, metadata)
        _validate_diagnostics(source, diagnostics, require_policy=True)
    else:
        source = _job_by_id(job.source_job_id)
        _validate_diagnostics(source, diagnostics, require_policy=True)
    load_ranking_evidence(
        logs_root / ".ranking-evidence" / RANKING_EVIDENCE_GROUP / "context.pt",
        paths["ranking_evidence"],
    )
    return {
        "job_id": job.id,
        "run_name": job.run_name,
        "source_job_id": job.source_job_id,
        "seed": job.seed,
        "metrics": metrics,
        "artifact_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in paths.items()
        },
    }


def _collision_system(
    job: CollisionSearchJob,
    seed42: dict[str, Any],
    confirmation,
    confirmation_rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    repeats = sorted(
        (
            confirmation_rows[candidate.id]
            for candidate in confirmation.jobs
            if candidate.family == "collision" and candidate.source_job_id == job.id
        ),
        key=lambda row: row["seed"],
    )
    rows = [seed42, *repeats]
    if len(rows) != 3:
        raise ValueError(f"{job.id}: collision confirmation seeds are incomplete")
    return {
        "source_job_id": job.id,
        "mean_metrics": mean_metrics(rows),
        "seeds": [42, 43, 44],
        "rows": rows,
    }


def _convergence(rows: list[dict[str, Any]], logs_root: Path) -> dict[str, Any]:
    values = []
    for seed, row in zip((42, 43, 44, 45), rows, strict=True):
        curve = load_validation_curve(logs_root / row["run_name"] / "sweep.log")
        values.append(
            {
                "seed": seed,
                "normalized_auc": curve.normalized_auc,
                "first_epoch_at_95_percent": curve.first_epoch_at_95_percent,
                "sweep_log_sha256": curve.source_sha256,
            }
        )
    return {
        "mean_normalized_auc": statistics.fmean(
            row["normalized_auc"] for row in values
        ),
        "mean_first_epoch_at_95_percent": statistics.fmean(
            row["first_epoch_at_95_percent"] for row in values
        ),
        "rows": values,
    }


def _terminal_document(
    job: CollisionSearchJob, row: dict[str, Any]
) -> dict[str, Any]:
    artifacts = row["artifact_sha256"]
    return {
        "schema": "g6-rq2-rq3-selection/v1",
        "dataset_size": "native-50m",
        "selection_resolved": True,
        "terminal_policy": "suffix_on" if job.policy == "suffix" else "suffix_off",
        "terminal_job_id": job.id,
        "terminal_run_name": row["run_name"],
        "terminal_parameters": {
            "representation": "item_frozen_sid_event",
            "levels": job.coordinate.num_levels,
            "shared_codes": job.coordinate.num_codes,
            "iterations": job.coordinate.kmeans_iterations,
            "width": 128,
            "embedding_learning_rate": job.coordinate.embedding_learning_rate,
            "deep_learning_rate": job.coordinate.deep_learning_rate,
            "batch": 256,
        },
        "manifest_sha256": collision_search_manifest().sha256,
        "metrics": row["metrics"],
        "artifact_sha256": {
            name: artifacts[name]
            for name in (
                "job_contract",
                "final_metrics",
                "training_metadata",
                "ranking_evidence",
                "sid_diagnostics",
            )
        },
    }


def _surface_row(surface: dict[str, Any], job_id: str) -> dict[str, Any]:
    matches = [
        row for row in surface["rq2_rq3"]["rows"] if row["job_id"] == job_id
    ]
    if len(matches) != 1:
        raise ValueError(f"surface row {job_id!r} is not unique")
    return matches[0]


def _authenticate_surface_row(row: dict[str, Any], logs_root: Path) -> None:
    directory = logs_root / row["run_name"]
    job_id = row["job_id"]
    if job_id.startswith("rq1_surface:"):
        job = next(job for job in rq1_search_manifest().jobs if job.id == job_id)
        manifest_sha256 = rq1_search_manifest().sha256
    else:
        job = _job_by_id(job_id)
        manifest_sha256 = collision_search_manifest().sha256
    expected_run_name, contract_name, _ = artifact_identity(job, manifest_sha256)
    if row["run_name"] != expected_run_name:
        raise ValueError(f"{job_id}: frozen run identity changed")
    paths = {
        "job_contract": directory / contract_name,
        "final_metrics": directory / "final_metrics.json",
        "training_metadata": directory / "training_metadata.json",
        "ranking_evidence": directory / "ranking_evidence.pt",
        "sid_diagnostics": directory / "semantic_id_diagnostics.json",
        "sweep_log": directory / "sweep.log",
    }
    expected = row["artifact_sha256"]
    for name, path in paths.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected.get(name):
            raise ValueError(f"{row['run_name']}: frozen {name} changed")


def _job_by_id(job_id: str) -> CollisionSearchJob:
    matches = [job for job in collision_search_manifest().jobs if job.id == job_id]
    if len(matches) != 1:
        raise ValueError(f"collision job {job_id!r} is not approved")
    return matches[0]


def _numeric(document: dict[str, Any]) -> dict[str, float]:
    metrics = {
        name: float(value)
        for name, value in document.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    if any(name not in metrics or not math.isfinite(metrics[name]) for name in METRICS):
        raise ValueError("required confirmation metrics are missing")
    return metrics


def _write_immutable(path: Path, document: dict[str, Any]) -> None:
    content = json.dumps(document, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text() != content:
        raise RuntimeError(f"evidence already differs: {path}")
    path.write_text(content)


if __name__ == "__main__":
    evidence, terminal = finalize()
    print(evidence["rq2_rq3"]["terminal"], terminal["terminal_run_name"])
