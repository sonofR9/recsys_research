from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median
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
    _efficiency,
    _ranking_slices,
    load_training_item_counts,
)
from experiments.g3_pretrained_item_embeddings.launchers.control import PROJECT_ROOT
from experiments.g3_pretrained_item_embeddings.launchers.rq3_post_boundary import (
    JOB_ENVIRONMENT,
    LEDGER_ENVIRONMENT,
    RQ3_POST_BOUNDARY_LEDGER_PATH,
    decode_rq3_post_boundary_job,
    preview_rq3_post_boundary_ledger,
    resolve_rq3_post_boundary_feature_data,
)
from experiments.g3_pretrained_item_embeddings.protocol.constants import (
    APPROVED_PROTOCOL,
    APPROVED_PROTOCOL_SHA256,
)
from experiments.g3_pretrained_item_embeddings.protocol.rq3 import (
    RQ3_OUTPUT_FAMILY_IDS,
    _FeatureIdentity,
    _validate_training_diagnostics,
)
from experiments.g3_pretrained_item_embeddings.protocol.rq3_post_boundary import (
    RQ2_FINAL_EVIDENCE_LOGICAL_SHA256,
    RQ3_OUTPUT_ARTIFACT_CONTRACTS,
    Rq3PostBoundaryLedger,
    Rq3PostBoundaryLedgerRow,
    load_rq3_post_boundary_ledger,
)


RQ3_POST_BOUNDARY_BATCH_ID = "1e360bb9eb4c43a9bb276d9b6f204f22"
RQ3_POST_BOUNDARY_LEDGER_LOGICAL_SHA256 = (
    "9fc9e47e2f061379f53e21ce73ec1c46ce848fadcecf6793eb0c0f67775d0885"
)
RQ3_POST_BOUNDARY_LEDGER_FILE_SHA256 = (
    "ff54b4e80b8393ae0e292f437b5397d51440eba9bce7306fd32892530ca5b465"
)
RQ3_POST_BOUNDARY_BATCH_FILE_SHA256 = (
    "c2b9981293df688106b4094647280ded07f8cc56ca3137d9f89a730621543709"
)
RQ3_INITIAL_EVIDENCE_PATH = (
    "experiments/g3_pretrained_item_embeddings/evidence/"
    "rq3_initial_output_search_native50m.json"
)

_METRIC_NAMES = tuple(
    f"{name}@{cutoff}"
    for name in ("recall", "ndcg", "mrr", "capped_recall", "coverage")
    for cutoff in (10, 50, 100)
)
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
_ARTIFACT_FILENAMES = {
    contract.name: contract.filename for contract in RQ3_OUTPUT_ARTIFACT_CONTRACTS
}


def select_rq3_family_winners(
    runs: Sequence[Mapping[str, object]],
    *,
    family_ids: Sequence[str] = RQ3_OUTPUT_FAMILY_IDS,
) -> dict[str, Mapping[str, object]]:
    selected: dict[str, Mapping[str, object]] = {}
    run_ids = [run.get("row_id") for run in runs]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("RQ3 tuning opportunities contain duplicate row IDs")
    for family_id in family_ids:
        family = [run for run in runs if run.get("family_id") == family_id]
        if len(family) != 9:
            raise ValueError(f"RQ3 family {family_id} must contain exactly nine opportunities")
        selected[family_id] = min(
            family,
            key=lambda run: (
                -float(_metrics(run)["recall@100"]),
                -float(_metrics(run)["ndcg@100"]),
                float(run["queue_wall_seconds"]),
                str(run["row_id"]),
            ),
        )
    if set(selected) != set(family_ids):
        raise ValueError("RQ3 family selection is incomplete")
    return selected


def assess_rq3_family_boundaries(
    selected: Mapping[str, Mapping[str, object]],
    runs: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    result = {}
    for family_id, run in selected.items():
        family_runs = [candidate for candidate in runs if candidate.get("family_id") == family_id]
        if len(family_runs) != 9:
            raise ValueError(f"RQ3 family {family_id} must contain nine boundary rows")
        embedding = _tested_rate_boundary(
            float(run["embedding_learning_rate"]),
            [float(candidate["embedding_learning_rate"]) for candidate in family_runs],
        )
        deep = _tested_rate_boundary(
            float(run["deep_learning_rate"]),
            [float(candidate["deep_learning_rate"]) for candidate in family_runs],
        )
        horizon = int(run["horizon_epochs"])
        best_epoch = int(run["best_epoch"])
        if not 1 <= best_epoch <= horizon:
            raise ValueError(f"RQ3 family {family_id} has an invalid restored epoch")
        extension = 60 if horizon == 40 and best_epoch == 40 else None
        result[family_id] = {
            "embedding_learning_rate": embedding,
            "deep_learning_rate": deep,
            "horizon": {
                "selected_epochs": horizon,
                "restored_best_epoch": best_epoch,
                "extend_to_epochs": extension,
            },
            "extension_required": bool(
                embedding["direction"] or deep["direction"] or extension
            ),
        }
    return result


def resolve_rq3_downstream_selection(
    winners: Mapping[str, Mapping[str, object]],
    boundaries: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    if set(winners) != set(RQ3_OUTPUT_FAMILY_IDS) or set(boundaries) != set(
        RQ3_OUTPUT_FAMILY_IDS
    ):
        raise ValueError("RQ3 downstream selection requires all five families")
    learned = winners["rq3_output_learned"]
    best = min(
        winners.values(),
        key=lambda run: (
            -float(_metrics(run)["recall@100"]),
            -float(_metrics(run)["ndcg@100"]),
            float(run["queue_wall_seconds"]),
            str(run["row_id"]),
        ),
    )
    learned_recall = float(_metrics(learned)["recall@100"])
    best_recall = float(_metrics(best)["recall@100"])
    band = abs(learned_recall) * APPROVED_PROTOCOL.relative_dispersion(
        "native-50m", "recall@100"
    )
    treatment_promoted = (
        best["family_id"] != "rq3_output_learned"
        and best_recall > learned_recall + band
    )
    aggregate_selected = best if treatment_promoted else learned
    unresolved = [
        family_id
        for family_id, decision in boundaries.items()
        if decision.get("extension_required") is True
    ]
    return {
        "status": "boundary_extensions_required" if unresolved else "resolved",
        "scientific_selection_rule": (
            "best family by validation Recall@100, NDCG@100, wall time, and row ID"
        ),
        "aggregate_promotion_rule": (
            "a non-learned target enters the aggregate only when its Recall@100 "
            "gain over learned output exceeds the native-50M operational band"
        ),
        "learned_reference": learned,
        "best_absolute": best,
        "rq4_scientific_selected": best,
        "aggregate_selected": aggregate_selected,
        "treatment_promoted": treatment_promoted,
        "recall@100_operational_band": band,
        "best_minus_learned_recall@100": best_recall - learned_recall,
        "unresolved_boundary_families": unresolved,
    }


def build_rq3_paired_contrasts(
    winners: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    if set(winners) != set(RQ3_OUTPUT_FAMILY_IDS):
        raise ValueError("RQ3 paired contrasts require all five family winners")
    learned, frozen, trainable, learned_frozen, learned_trainable = (
        winners[family_id] for family_id in RQ3_OUTPUT_FAMILY_IDS
    )
    specifications = {
        "target_type_frozen_content_vs_learned_id": (
            "target_type",
            learned,
            frozen,
            False,
        ),
        "pretrained_initialization_trainable_content_vs_learned_id": (
            "pretrained_initialization",
            learned,
            trainable,
            False,
        ),
        "freezing_content_target": ("freezing", frozen, trainable, True),
        "learned_id_augmentation_of_frozen_content": (
            "target_composition",
            frozen,
            learned_frozen,
            True,
        ),
        "freezing_concatenated_target": (
            "freezing",
            learned_frozen,
            learned_trainable,
            True,
        ),
        "expected_variant_4_vs_learned_id": (
            "target_composition",
            learned,
            learned_frozen,
            False,
        ),
    }
    return {
        name: _paired_contrast(
            axis=axis,
            reference=reference,
            treatment=treatment,
            isolated_axis=isolated,
        )
        for name, (axis, reference, treatment, isolated) in specifications.items()
    }


def build_rq3_matched_coordinate_contrasts(
    runs: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    by_family: dict[str, dict[tuple[float, float, int], Mapping[str, object]]] = {}
    for family_id in RQ3_OUTPUT_FAMILY_IDS:
        family_rows = [row for row in runs if row.get("family_id") == family_id]
        if len(family_rows) != 9:
            raise ValueError(f"RQ3 family {family_id} must contain nine paired rows")
        by_coordinate = {_coordinate(row): row for row in family_rows}
        if len(by_coordinate) != 9:
            raise ValueError(f"RQ3 family {family_id} has duplicate coordinates")
        by_family[family_id] = by_coordinate
    coordinates = set(next(iter(by_family.values())))
    if any(set(rows) != coordinates for rows in by_family.values()):
        raise ValueError("RQ3 families do not share the same tuning coordinates")
    specifications = {
        "target_type_frozen_content_vs_learned_id": (
            "rq3_output_learned",
            "rq3_output_frozen_content",
        ),
        "pretrained_initialization_trainable_content_vs_learned_id": (
            "rq3_output_learned",
            "rq3_output_trainable_content",
        ),
        "freezing_content_target": (
            "rq3_output_frozen_content",
            "rq3_output_trainable_content",
        ),
        "learned_id_augmentation_of_frozen_content": (
            "rq3_output_frozen_content",
            "rq3_output_learned_frozen_content",
        ),
        "freezing_concatenated_target": (
            "rq3_output_learned_frozen_content",
            "rq3_output_learned_trainable_content",
        ),
        "expected_variant_4_vs_learned_id": (
            "rq3_output_learned",
            "rq3_output_learned_frozen_content",
        ),
    }
    result = {}
    for name, (reference_id, treatment_id) in specifications.items():
        pairs = []
        for coordinate in sorted(coordinates):
            reference = by_family[reference_id][coordinate]
            treatment = by_family[treatment_id][coordinate]
            reference_recall = float(_metrics(reference)["recall@100"])
            treatment_recall = float(_metrics(treatment)["recall@100"])
            pairs.append(
                {
                    "embedding_learning_rate": coordinate[0],
                    "deep_learning_rate": coordinate[1],
                    "horizon_epochs": coordinate[2],
                    "reference_row_id": reference["row_id"],
                    "treatment_row_id": treatment["row_id"],
                    "reference_recall@100": reference_recall,
                    "treatment_recall@100": treatment_recall,
                    "recall@100_delta": treatment_recall - reference_recall,
                    "recall@100_percent_delta": (
                        100.0 * (treatment_recall - reference_recall) / reference_recall
                        if reference_recall
                        else None
                    ),
                    "ndcg@100_delta": float(_metrics(treatment)["ndcg@100"])
                    - float(_metrics(reference)["ndcg@100"]),
                }
            )
        recall_deltas = [float(pair["recall@100_delta"]) for pair in pairs]
        result[name] = {
            "reference_family_id": reference_id,
            "treatment_family_id": treatment_id,
            "pairs": pairs,
            "summary": {
                "pair_count": 9,
                "treatment_recall@100_win_count": sum(
                    delta > 0 for delta in recall_deltas
                ),
                "mean_recall@100_delta": mean(recall_deltas),
                "median_recall@100_delta": median(recall_deltas),
            },
        }
    return result


def build_rq3_post_boundary_evidence(
    root: Path,
    *,
    batch_id: str = RQ3_POST_BOUNDARY_BATCH_ID,
) -> dict[str, object]:
    root = root.resolve(strict=True)
    if batch_id != RQ3_POST_BOUNDARY_BATCH_ID:
        raise ValueError("RQ3 evidence requires the approved exact batch")
    ledger_path = root / RQ3_POST_BOUNDARY_LEDGER_PATH
    if _file_sha256(ledger_path) != RQ3_POST_BOUNDARY_LEDGER_FILE_SHA256:
        raise ValueError("RQ3 evidence received another ledger file")
    ledger = load_rq3_post_boundary_ledger(ledger_path)
    if ledger.sha256 != RQ3_POST_BOUNDARY_LEDGER_LOGICAL_SHA256:
        raise ValueError("RQ3 evidence received another logical ledger")
    batch_path = (
        root / "generated/training-queue-service/batches" / f"{batch_id}.json"
    )
    if _file_sha256(batch_path) != RQ3_POST_BOUNDARY_BATCH_FILE_SHA256:
        raise ValueError("RQ3 evidence received another queue batch file")
    job_ids = _validate_batch(_load_json(batch_path), batch_id=batch_id)
    _require_completed_batch(root, job_ids)

    verified = preview_rq3_post_boundary_ledger(root=root)
    ledger = load_rq3_post_boundary_ledger(ledger_path, expected=verified)
    feature_path = resolve_rq3_post_boundary_feature_data(root, ledger)
    feature_identity = _feature_identity(ledger)
    item_counts = load_training_item_counts(feature_path)
    context_path = root / "generated/logs/.ranking-evidence/g3-native50m/context.pt"
    physical = [
        _collect_physical_run(
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
        for row, job_id in zip(ledger.physical_rows, job_ids, strict=True)
    ]
    reused = _collect_reused_runs(
        root=root,
        ledger=ledger,
        context_path=context_path,
        item_counts=item_counts,
        feature_identity=feature_identity,
    )
    by_logical_id = {run["row_id"]: run for run in (*reused, *physical)}
    if len(by_logical_id) != 45:
        raise ValueError("RQ3 result collection does not preserve 45 opportunities")
    all_runs = [by_logical_id[row.id] for row in ledger.logical_rows]
    winners = select_rq3_family_winners(all_runs)
    boundaries = assess_rq3_family_boundaries(winners, all_runs)
    downstream = resolve_rq3_downstream_selection(winners, boundaries)
    winner_contrasts = build_rq3_paired_contrasts(winners)
    matched_contrasts = build_rq3_matched_coordinate_contrasts(all_runs)
    mechanism = _mechanism_assessment(winners)
    payload = {
        "schema_version": 1,
        "kind": "g3_rq3_initial_output_search_native50m",
        "protocol_sha256": APPROVED_PROTOCOL_SHA256,
        "rq3_ledger": _file_fact(root, ledger_path)
        | {"logical_sha256": ledger.sha256},
        "final_rq2_evidence_logical_sha256": RQ2_FINAL_EVIDENCE_LOGICAL_SHA256,
        "queue_batch": _file_fact(root, batch_path) | {"batch_id": batch_id},
        "ranking_context": _file_fact(root, context_path),
        "feature_data": _file_fact(root, feature_path),
        "opportunity_accounting": {
            "logical_rows": 45,
            "reused_rows": 7,
            "physical_rows": 38,
        },
        "physical_runs": physical,
        "reused_runs": reused,
        "all_tuning_opportunities": all_runs,
        "selection_rule": (
            "validation Recall@100, validation NDCG@100, lower queue wall time, "
            "then row ID"
        ),
        "family_selections": {
            family_id: {
                "status": (
                    "boundary_extension_required"
                    if boundaries[family_id]["extension_required"]
                    else "resolved"
                ),
                "selected": winners[family_id],
                "boundary_decision": boundaries[family_id],
            }
            for family_id in RQ3_OUTPUT_FAMILY_IDS
        },
        "downstream_selection": downstream,
        "reader_metrics": {
            family_id: {
                "row_id": winner["row_id"],
                "overall": winner["metrics"],
                "item_frequency_slices": winner["slices"],
                "efficiency": winner["efficiency"],
            }
            for family_id, winner in winners.items()
        },
        "selected_winner_contrasts": winner_contrasts,
        "matched_coordinate_contrasts": matched_contrasts,
        "implementation_checks": {
            "exact_ledger_and_batch_mapping": True,
            "all_artifacts_authenticated": True,
            "all_metrics_recomputed": True,
            "all_diagnostics_validated": True,
            "matched_non_catalog_protocol": True,
        },
        "mechanism_assessment": mechanism,
    }
    return _document(payload)


def persist_rq3_post_boundary_evidence(
    path: Path,
    document: Mapping[str, object],
) -> Path:
    validated = _validate_document(dict(document))
    content = (_canonical_json(validated) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(content)
    except FileExistsError:
        if path.read_bytes() != content:
            raise RuntimeError(f"immutable RQ3 evidence differs: {path}")
    return path


def load_rq3_post_boundary_evidence(
    path: Path,
    *,
    root: Path,
) -> dict[str, object]:
    evidence = _validate_document(_load_json(path))
    rebuilt = build_rq3_post_boundary_evidence(root)
    if _canonical_json(evidence) != _canonical_json(rebuilt):
        raise ValueError("RQ3 evidence differs from authenticated artifacts")
    return evidence


def _validate_batch(batch: Mapping[str, object], *, batch_id: str) -> list[str]:
    jobs = batch.get("jobs")
    if (
        set(batch) != {"id", "jobs", "sealed", "sealed_at", "submitted_at"}
        or batch.get("id") != batch_id
        or batch.get("sealed") is not True
        or not isinstance(jobs, list)
        or len(jobs) != 38
        or len(set(jobs)) != 38
        or any(not isinstance(job_id, str) or not job_id for job_id in jobs)
        or not _finite_number(batch.get("submitted_at"))
        or not _finite_number(batch.get("sealed_at"))
        or float(batch["sealed_at"]) < float(batch["submitted_at"])
    ):
        raise ValueError("RQ3 queue batch is not the exact sealed 38-job batch")
    return list(jobs)


def _require_completed_batch(root: Path, job_ids: Sequence[str]) -> None:
    queue = root / "generated/training-queue-service"
    incomplete = []
    for job_id in job_ids:
        states = [
            state
            for state in ("pending", "dispatched", "completed", "failed")
            if (queue / state / f"{job_id}.json").is_file()
        ]
        if states != ["completed"]:
            incomplete.append((job_id, states))
    if incomplete:
        raise RuntimeError(
            f"RQ3 batch is not complete: {len(incomplete)} of 38 jobs incomplete"
        )


def _collect_physical_run(
    *,
    root: Path,
    ledger: Rq3PostBoundaryLedger,
    ledger_path: Path,
    row: Rq3PostBoundaryLedgerRow,
    batch_id: str,
    job_id: str,
    context_path: Path,
    item_counts: Mapping[int, int],
    feature_identity: _FeatureIdentity,
) -> dict[str, object]:
    completed_path = (
        root / "generated/training-queue-service/completed" / f"{job_id}.json"
    )
    queue_job = _load_json(completed_path)
    expected_script = (
        root
        / "experiments/g3_pretrained_item_embeddings/launchers/"
        "run_rq3_post_boundary.py"
    ).resolve(strict=True)
    row_document = row.to_dict()
    if (
        set(queue_job) != _QUEUE_RECORD_KEYS
        or queue_job.get("id") != job_id
        or queue_job.get("batch_id") != batch_id
        or queue_job.get("run") != row.run_name
        or queue_job.get("exit_code") != 0
        or queue_job.get("data_group") != "g3-native50m-likes"
        or Path(str(queue_job.get("script"))).resolve() != expected_script
        or not _ordered_job_times(queue_job)
    ):
        raise ValueError(f"RQ3 queue completion differs for {row.id}")
    environment = queue_job.get("environment")
    pairs = [value.split("=", 1) for value in environment] if isinstance(environment, list) else []
    values = dict(pairs)
    if (
        len(pairs) != len(values) == 3
        or set(values) != {"WANDB_MODE", JOB_ENVIRONMENT, LEDGER_ENVIRONMENT}
        or values["WANDB_MODE"] != "offline"
        or Path(values[LEDGER_ENVIRONMENT]).resolve() != ledger_path
    ):
        raise ValueError(f"RQ3 queue environment differs for {row.id}")
    compiled = decode_rq3_post_boundary_job(values[JOB_ENVIRONMENT], ledger)
    if compiled.row_id != row.id or compiled.job != row_document:
        raise ValueError(f"RQ3 queue payload differs for {row.id}")
    directory = root / "generated/logs" / row.run_name
    contract = _load_json(directory / _ARTIFACT_FILENAMES["job_contract"])
    if contract != compiled.to_dict() | {
        "ledger_path": str(ledger_path),
        "ledger_sha256": ledger.sha256,
    }:
        raise ValueError(f"RQ3 job contract differs for {row.id}")
    return _collect_artifacts(
        root=root,
        row=row,
        artifact_paths={
            name: directory / filename
            for name, filename in _ARTIFACT_FILENAMES.items()
        },
        context_path=context_path,
        item_counts=item_counts,
        feature_identity=feature_identity,
        queue_wall_seconds=float(queue_job["finished_at"])
        - float(queue_job["dispatched_at"]),
        queue_job=_file_fact(root, completed_path) | {"job_id": job_id},
        source_row_id=None,
        verify_window=(
            float(queue_job["dispatched_at"]),
            float(queue_job["finished_at"]),
        ),
    )


def _collect_reused_runs(
    *,
    root: Path,
    ledger: Rq3PostBoundaryLedger,
    context_path: Path,
    item_counts: Mapping[int, int],
    feature_identity: _FeatureIdentity,
) -> list[dict[str, object]]:
    evidence = _load_json(root / ledger.final_evidence_path)
    if evidence.get("sha256") != ledger.final_rq2_evidence_sha256:
        raise ValueError("RQ3 reused source evidence changed after verification")
    inputs = evidence.get("rq3_inputs")
    rows = inputs.get("eligible_learned_output_reuse_rows") if isinstance(inputs, dict) else None
    if not isinstance(rows, list):
        raise ValueError("RQ3 reused source rows are absent")
    by_source = {row.get("row_id"): row for row in rows if isinstance(row, dict)}
    result = []
    for logical in (row for row in ledger.logical_rows if row.reused_from is not None):
        source = by_source.get(logical.reused_from)
        if not isinstance(source, dict):
            raise ValueError(f"RQ3 reused source row is absent for {logical.id}")
        artifacts = source.get("artifacts")
        queue_reference = source.get("queue_job")
        if not isinstance(artifacts, dict) or not isinstance(queue_reference, dict):
            raise ValueError(f"RQ3 reused source bindings are absent for {logical.id}")
        for name in _ARTIFACT_FILENAMES:
            reference = artifacts.get(name)
            if not isinstance(reference, dict):
                raise ValueError(f"RQ3 reused artifact {name} is absent for {logical.id}")
            current = _file_fact(root, root / str(reference["path"]))
            if current != reference:
                raise ValueError(f"RQ3 reused artifact {name} changed for {logical.id}")
        queue_path = root / str(queue_reference["path"])
        if _file_fact(root, queue_path) != {
            key: queue_reference[key] for key in ("path", "sha256", "size_bytes")
        }:
            raise ValueError(f"RQ3 reused queue record changed for {logical.id}")
        queue_job = _load_json(queue_path)
        artifact_paths = {
            name: root / str(reference["path"])
            for name, reference in artifacts.items()
        }
        run = _collect_artifacts(
            root=root,
            row=logical,
            artifact_paths=artifact_paths,
            context_path=context_path,
            item_counts=item_counts,
            feature_identity=feature_identity,
            queue_wall_seconds=float(queue_job["finished_at"])
            - float(queue_job["dispatched_at"]),
            queue_job=dict(queue_reference),
            source_row_id=str(logical.reused_from),
            verify_window=None,
        )
        result.append(run)
    if len(result) != 7:
        raise ValueError("RQ3 reused result collection must contain seven rows")
    return result


def _collect_artifacts(
    *,
    root: Path,
    row: Rq3PostBoundaryLedgerRow,
    artifact_paths: Mapping[str, Path],
    context_path: Path,
    item_counts: Mapping[int, int],
    feature_identity: _FeatureIdentity,
    queue_wall_seconds: float,
    queue_job: Mapping[str, object],
    source_row_id: str | None,
    verify_window: tuple[float, float] | None,
) -> dict[str, object]:
    if set(artifact_paths) != set(_ARTIFACT_FILENAMES):
        raise ValueError(f"RQ3 artifact paths are incomplete for {row.id}")
    metadata = _load_json(artifact_paths["training_metadata"])
    _validate_metadata(metadata, row=row)
    metrics_path = artifact_paths["final_metrics"]
    metrics = _load_json(metrics_path)
    if set(metrics) != {*_METRIC_NAMES, "num_users"} or any(
        not _finite_number(value) for value in metrics.values()
    ):
        raise ValueError(f"RQ3 metric schema differs for {row.id}")
    ranking_path = artifact_paths["ranking_evidence"]
    rankings_path = artifact_paths["top_item_rankings"]
    recomputed = _recompute_metrics(context_path, ranking_path, rankings_path)
    if set(recomputed) != set(metrics) or any(
        abs(float(metrics[name]) - float(recomputed[name])) > 1e-15
        for name in metrics
    ):
        raise ValueError(f"RQ3 metrics differ from ranking evidence for {row.id}")
    diagnostics_path = artifact_paths["training_diagnostics"]
    diagnostics_document = _load_json(diagnostics_path)
    identity = _validate_training_diagnostics(
        diagnostics_document,
        feature_identity=feature_identity,
        horizon_epochs=row.horizon_epochs,
        catalog_representation=row.catalog_representation,
    )
    best_epoch = metadata["best_epoch"]
    restored = diagnostics_document["epochs"][best_epoch - 1]
    diagnostic_summary = {
        "schema_version": identity.schema_version,
        "epochs": identity.epoch_count,
        "restored_epoch": best_epoch,
        "nonfinite_measurement_count": sum(
            int(value) for value in _values_named(diagnostics_document, "nonfinite_count")
        ),
        "content_drift_reference": diagnostics_document["content_drift_reference"],
        "restored_epoch_mechanism": {
            name: restored[name]
            for name in (
                "training",
                "component_gradient_norms",
                "catalog_representation_norm",
                "pretrained_content",
            )
        }
        | (
            {"catalog_table_gradient_norms": restored["catalog_table_gradient_norms"]}
            if "catalog_table_gradient_norms" in restored
            else {}
        ),
    }
    if diagnostic_summary["nonfinite_measurement_count"] != 0:
        raise ValueError(f"RQ3 diagnostics contain nonfinite values for {row.id}")
    artifacts = {
        name: _file_fact(root, path) for name, path in artifact_paths.items()
    }
    if verify_window is not None:
        verify_unique_completed_run(
            root / "generated/training-queue-service",
            run_name=row.run_name,
            expected_job_id=str(queue_job["job_id"]),
        )
        verify_artifacts_in_job_window(
            tuple(artifact_paths.values()),
            dispatched_at=verify_window[0],
            finished_at=verify_window[1],
            run_label=row.id,
        )
    return {
        "row_id": row.id,
        "source_row_id": source_row_id,
        "reused": source_row_id is not None,
        "family_id": row.family_id,
        "catalog_representation": row.catalog_representation,
        "run_name": row.run_name,
        "embedding_learning_rate": row.embedding_learning_rate,
        "deep_learning_rate": row.deep_learning_rate,
        "horizon_epochs": row.horizon_epochs,
        "best_epoch": best_epoch,
        "epochs_trained": metadata["epochs_trained"],
        "queue_wall_seconds": queue_wall_seconds,
        "metrics": metrics,
        "metric_provenance": {
            "recomputed_from_ranking_evidence": True,
            "absolute_tolerance": 1e-15,
            "num_users": int(recomputed["num_users"]),
        },
        "slices": _ranking_slices(
            context_path=context_path,
            ranking_path=ranking_path,
            rankings_path=rankings_path,
            item_counts=item_counts,
        ),
        "efficiency": _efficiency(
            metadata=metadata,
            log_path=artifact_paths["sweep_log"],
            queue_wall_seconds=queue_wall_seconds,
        ),
        "diagnostics": diagnostic_summary,
        "queue_job": dict(queue_job),
        "artifacts": artifacts,
    }


def _validate_metadata(
    metadata: Mapping[str, object],
    *,
    row: Rq3PostBoundaryLedgerRow,
) -> None:
    representation = {
        "catalog_representation": row.catalog_representation,
        "content_gate": "fixed",
        "extra_item_id_dim": None,
        "gate_hidden_dim": None,
        "history_hidden_dim": 128,
        "history_representation": "id_content",
        "metadata": [],
        "metadata_dim": None,
    }
    expected = {
        "batch_size": 512,
        "seed": 42,
        "embedding_learning_rate": row.embedding_learning_rate,
        "deep_learning_rate": row.deep_learning_rate,
        "lr_schedule_horizon_epochs": row.horizon_epochs,
        "num_epochs": row.horizon_epochs,
        "max_epochs": row.horizon_epochs,
        "epochs_trained": row.horizon_epochs,
        "stopped_epoch": row.horizon_epochs,
        "lr_horizon_complete": True,
        "selection_resolved": True,
        "early_stopped": False,
        "g3_dataset_size": "native-50m",
        "g3_protocol_sha256": APPROVED_PROTOCOL_SHA256,
        "g3_representation": representation,
    }
    if any(metadata.get(name) != value for name, value in expected.items()):
        raise ValueError(f"RQ3 runtime metadata differs for {row.id}")
    best_epoch = metadata.get("best_epoch")
    if type(best_epoch) is not int or not 1 <= best_epoch <= row.horizon_epochs:
        raise ValueError(f"RQ3 restored epoch is invalid for {row.id}")
    if metadata.get("best_epoch_at_cap") is not (best_epoch == row.horizon_epochs):
        raise ValueError(f"RQ3 restored-epoch cap flag differs for {row.id}")
    traces = metadata.get("lr_group_traces")
    if not isinstance(traces, dict) or any(
        not isinstance(traces.get(group), list)
        or len(traces[group]) != row.horizon_epochs
        or traces[group][-1] != 0.0
        for group in ("embedding", "deep")
    ):
        raise ValueError(f"RQ3 schedule trace is incomplete for {row.id}")
    invariants = metadata.get("transfer_invariants")
    if (
        not isinstance(invariants, dict)
        or invariants.get("g3_protocol_sha256") != APPROVED_PROTOCOL_SHA256
        or invariants.get("g3_representation") != representation
        or invariants.get("dataset_size") != "50m"
        or invariants.get("user_sample") is not None
        or invariants.get("evaluation_catalog") != "all"
        or invariants.get("exclude_seen_from_evaluation") is not False
        or invariants.get("batch_size") != 512
        or invariants.get("lr_schedule_horizon_epochs") != row.horizon_epochs
    ):
        raise ValueError(f"RQ3 runtime invariants differ for {row.id}")


def _feature_identity(ledger: Rq3PostBoundaryLedger) -> _FeatureIdentity:
    feature = ledger.feature
    return _FeatureIdentity(
        manifest_path=feature.manifest_path,
        manifest_sha256=feature.manifest_sha256,
        manifest_file_sha256=feature.manifest_file_sha256,
        data_path=feature.data_path,
        data_sha256=feature.data_sha256,
        frequency_terciles=feature.frequency_terciles,
        training_count_reference=feature.training_count_reference,
        slice_membership_reference=feature.slice_membership_reference,
    )


def _paired_contrast(
    *,
    axis: str,
    reference: Mapping[str, object],
    treatment: Mapping[str, object],
    isolated_axis: bool,
) -> dict[str, object]:
    reference_metrics = _metrics(reference)
    treatment_metrics = _metrics(treatment)
    metric_names = sorted(set(reference_metrics) & set(treatment_metrics))
    slices = {}
    for name in ("head", "mid", "tail"):
        reference_slice = _slice(reference, name)
        treatment_slice = _slice(treatment, name)
        for key in ("num_users", "num_targets", "item_membership_sha256"):
            if reference_slice.get(key) != treatment_slice.get(key):
                raise ValueError(f"RQ3 paired contrast {axis} slice identity differs")
        slices[name] = {
            "reference_recall@100": float(reference_slice["recall@100"]),
            "treatment_recall@100": float(treatment_slice["recall@100"]),
            "delta": float(treatment_slice["recall@100"])
            - float(reference_slice["recall@100"]),
            "num_users": reference_slice["num_users"],
            "num_targets": reference_slice["num_targets"],
            "evidence_status": "descriptive_only_no_slice_repeat_calibration",
        }
    reference_efficiency = _efficiency_payload(reference)
    treatment_efficiency = _efficiency_payload(treatment)
    efficiency_names = sorted(
        name
        for name in set(reference_efficiency) & set(treatment_efficiency)
        if _finite_number(reference_efficiency[name])
        and _finite_number(treatment_efficiency[name])
    )
    return {
        "axis": axis,
        "isolated_axis": isolated_axis,
        "reference_family_id": reference["family_id"],
        "reference_row_id": reference["row_id"],
        "treatment_family_id": treatment["family_id"],
        "treatment_row_id": treatment["row_id"],
        "overall_metric_deltas": {
            name: float(treatment_metrics[name]) - float(reference_metrics[name])
            for name in metric_names
        },
        "item_frequency_slice_recall@100": slices,
        "efficiency_deltas": {
            name: float(treatment_efficiency[name])
            - float(reference_efficiency[name])
            for name in efficiency_names
        },
    }


def _mechanism_assessment(
    winners: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    learned, frozen, trainable, learned_frozen, _ = (
        winners[family_id] for family_id in RQ3_OUTPUT_FAMILY_IDS
    )
    recalls = {
        family_id: float(_metrics(run)["recall@100"])
        for family_id, run in winners.items()
    }
    expected_family = "rq3_output_learned_frozen_content"
    best_family = min(
        RQ3_OUTPUT_FAMILY_IDS,
        key=lambda family_id: (
            -recalls[family_id],
            -float(_metrics(winners[family_id])["ndcg@100"]),
            float(winners[family_id]["queue_wall_seconds"]),
            str(winners[family_id]["row_id"]),
        ),
    )
    learned_recall = float(_metrics(learned)["recall@100"])
    trainable_recall = float(_metrics(trainable)["recall@100"])
    learned_band = abs(learned_recall) * APPROVED_PROTOCOL.relative_dispersion(
        "native-50m", "recall@100"
    )
    explicit_expectations = {
        "variant_4_is_best": best_family == expected_family,
        "variant_4_beats_variant_1": recalls[expected_family]
        > learned_recall,
        "variant_4_beats_variant_2": recalls[expected_family]
        > float(_metrics(frozen)["recall@100"]),
        "variant_3_not_much_worse_than_variant_1": trainable_recall
        >= learned_recall - learned_band,
        "variant_3_beats_variant_2": trainable_recall
        > float(_metrics(frozen)["recall@100"]),
    }
    unexpected_failures = [
        name for name, satisfied in explicit_expectations.items() if not satisfied
    ]
    unexpected = bool(unexpected_failures)
    return {
        "best_family_by_selection_rule": best_family,
        "explicit_expectations": explicit_expectations,
        "variant_3_vs_variant_1": {
            "recall@100_delta": trainable_recall - learned_recall,
            "not_much_worse_threshold": learned_recall - learned_band,
            "operational_band": learned_band,
            "beats_variant_1": trainable_recall > learned_recall,
        },
        "unexpected_ordering": unexpected,
        "unexpected_expectation_failures": unexpected_failures,
        "unexpected_ordering_closure": {
            "status": "unresolved" if unexpected else "not_triggered",
            "reason": (
                "Implementation checks and saved diagnostics are present, but an "
                "unexpected ordering requires a targeted controlled mechanism test "
                "before scientific closure."
                if unexpected
                else "The explicit variant-4 ordering checks did not trigger closure."
            ),
        },
        "restored_epoch_mechanism_evidence": {
            family_id: run["diagnostics"]["restored_epoch_mechanism"]
            for family_id, run in winners.items()
        },
    }


def _validate_document(document: dict[str, object]) -> dict[str, object]:
    expected = {
        "schema_version",
        "kind",
        "protocol_sha256",
        "rq3_ledger",
        "final_rq2_evidence_logical_sha256",
        "queue_batch",
        "ranking_context",
        "feature_data",
        "opportunity_accounting",
        "physical_runs",
        "reused_runs",
        "all_tuning_opportunities",
        "selection_rule",
        "family_selections",
        "downstream_selection",
        "reader_metrics",
        "selected_winner_contrasts",
        "matched_coordinate_contrasts",
        "implementation_checks",
        "mechanism_assessment",
        "sha256",
    }
    payload = {name: value for name, value in document.items() if name != "sha256"}
    accounting = document.get("opportunity_accounting")
    if (
        set(document) != expected
        or document.get("schema_version") != 1
        or document.get("kind") != "g3_rq3_initial_output_search_native50m"
        or document.get("protocol_sha256") != APPROVED_PROTOCOL_SHA256
        or document.get("final_rq2_evidence_logical_sha256")
        != RQ2_FINAL_EVIDENCE_LOGICAL_SHA256
        or document.get("sha256") != _canonical_sha256(payload)
        or accounting != {"logical_rows": 45, "reused_rows": 7, "physical_rows": 38}
        or not isinstance(document.get("physical_runs"), list)
        or len(document["physical_runs"]) != 38
        or not isinstance(document.get("reused_runs"), list)
        or len(document["reused_runs"]) != 7
        or not isinstance(document.get("all_tuning_opportunities"), list)
        or len(document["all_tuning_opportunities"]) != 45
        or not isinstance(document.get("family_selections"), dict)
        or set(document["family_selections"]) != set(RQ3_OUTPUT_FAMILY_IDS)
        or not isinstance(document.get("downstream_selection"), dict)
        or not isinstance(document.get("selected_winner_contrasts"), dict)
        or len(document["selected_winner_contrasts"]) != 6
        or not isinstance(document.get("matched_coordinate_contrasts"), dict)
        or len(document["matched_coordinate_contrasts"]) != 6
    ):
        raise ValueError("RQ3 evidence schema or identity is invalid")
    return document


def _document(payload: Mapping[str, object]) -> dict[str, object]:
    document = dict(payload)
    document["sha256"] = _canonical_sha256(document)
    return document


def _metrics(run: Mapping[str, object]) -> Mapping[str, object]:
    metrics = run.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("RQ3 run has no metrics")
    return metrics


def _slice(run: Mapping[str, object], name: str) -> Mapping[str, object]:
    slices = run.get("slices")
    value = slices.get(name) if isinstance(slices, dict) else None
    if not isinstance(value, dict):
        raise ValueError(f"RQ3 run has no {name} slice")
    return value


def _efficiency_payload(run: Mapping[str, object]) -> Mapping[str, object]:
    value = run.get("efficiency")
    if not isinstance(value, dict):
        raise ValueError("RQ3 run has no efficiency evidence")
    return value


def _coordinate(run: Mapping[str, object]) -> tuple[float, float, int]:
    embedding = run.get("embedding_learning_rate")
    deep = run.get("deep_learning_rate")
    horizon = run.get("horizon_epochs")
    if (
        not _finite_number(embedding)
        or not _finite_number(deep)
        or type(horizon) is not int
        or int(horizon) < 1
    ):
        raise ValueError("RQ3 run has an invalid tuning coordinate")
    return float(embedding), float(deep), int(horizon)


def _tested_rate_boundary(
    value: float,
    tested_values: Sequence[float],
) -> dict[str, object]:
    tested = sorted(set(tested_values))
    if len(tested) < 3 or value not in tested:
        raise ValueError("selected RQ3 learning rate is absent from its tested surface")
    has_lower = any(candidate < value for candidate in tested)
    has_higher = any(candidate > value for candidate in tested)
    position = (value - tested[0]) / (tested[-1] - tested[0])
    return {
        "selected": value,
        "tested_values": tested,
        "has_tested_lower": has_lower,
        "has_tested_higher": has_higher,
        "normalized_position": position,
        "direction": "lower" if position <= 0.1 else "upper" if position >= 0.9 else None,
    }


def _ordered_job_times(job: Mapping[str, object]) -> bool:
    values = tuple(job.get(name) for name in ("submitted_at", "dispatched_at", "finished_at"))
    return all(_finite_number(value) for value in values) and (
        float(values[0]) <= float(values[1]) <= float(values[2])
    )


def _finite_number(value: object) -> bool:
    return type(value) in {int, float} and math.isfinite(float(value))


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


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--write", action="store_true")
    arguments = parser.parse_args()
    evidence = build_rq3_post_boundary_evidence(arguments.root)
    if arguments.write:
        path = arguments.root / RQ3_INITIAL_EVIDENCE_PATH
        persist_rq3_post_boundary_evidence(path, evidence)
        status = "materialized"
    else:
        path = arguments.root / RQ3_INITIAL_EVIDENCE_PATH
        status = "preview"
    print(
        json.dumps(
            {
                "path": str(path),
                "sha256": evidence["sha256"],
                "status": status,
                "logical_rows": 45,
                "physical_rows": 38,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
