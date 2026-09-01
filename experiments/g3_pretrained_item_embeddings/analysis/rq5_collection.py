from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

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
from experiments.g3_pretrained_item_embeddings.launchers.control import (
    G3_CPU_THREAD_ENVIRONMENT,
    decode_control_job,
)
from experiments.g3_pretrained_item_embeddings.launchers.rq5_initial import (
    JOB_ENVIRONMENT,
    LEDGER_ENVIRONMENT,
    preview_rq5_initial_ledger,
)
from experiments.g3_pretrained_item_embeddings.protocol.constants import (
    APPROVED_PROTOCOL,
    APPROVED_PROTOCOL_SHA256,
)
from experiments.g3_pretrained_item_embeddings.protocol.rq3 import (
    _FeatureIdentity,
    _validate_training_diagnostics,
)
from experiments.g3_pretrained_item_embeddings.protocol.rq5_initial import (
    RQ5_ARTIFACT_CONTRACTS,
    Rq5InitialLedger,
    Rq5InitialLedgerRow,
    load_rq5_initial_ledger,
    verify_rq5_initial_input_files,
)


_METRIC_NAMES = tuple(
    f"{name}@{cutoff}"
    for name in ("recall", "ndcg", "mrr", "capped_recall", "coverage")
    for cutoff in (10, 50, 100)
)
_QUEUE_KEYS = {
    "id", "batch_id", "data_group", "dispatched_at", "environment",
    "exit_code", "finished_at", "run", "script", "submitted_at",
}
_ARTIFACT_FILENAMES = {
    contract.name: contract.filename for contract in RQ5_ARTIFACT_CONTRACTS
}
RQ5_INITIAL_EVIDENCE_PATH = (
    "experiments/g3_pretrained_item_embeddings/evidence/"
    "rq5_gate_initial_native50m.json"
)
RQ5_HORIZON_EVIDENCE_PATH = (
    "experiments/g3_pretrained_item_embeddings/evidence/"
    "rq5_frequency_horizon_native50m.json"
)
RQ5_FINAL_EVIDENCE_PATH = (
    "experiments/g3_pretrained_item_embeddings/evidence/rq5_final_native50m.json"
)
RQ5_OUTCOME_EVIDENCE_PATH = (
    "experiments/g3_pretrained_item_embeddings/evidence/"
    "rq5_outcome_premechanism_native50m.json"
)


def build_rq5_initial_collection(
    *, root: Path, ledger_path: Path, expected_ledger_sha256: str, batch_id: str
) -> dict[str, object]:
    root = root.resolve(strict=True)
    expected = preview_rq5_initial_ledger(root=root)
    if expected.sha256 != expected_ledger_sha256:
        raise ValueError("RQ5 collection ledger SHA differs from verified preview")
    ledger_path = ledger_path.resolve(strict=True)
    ledger = load_rq5_initial_ledger(ledger_path, expected=expected)
    feature_path = verify_rq5_initial_input_files(root, ledger)
    batch_path = root / "generated/training-queue-service/batches" / f"{batch_id}.json"
    job_ids = _validate_batch(_load_json(batch_path), batch_id, len(ledger.rows))
    _require_completed_batch(root, job_ids)
    context_path = root / "generated/logs/.ranking-evidence/g3-native50m/context.pt"
    item_counts = load_training_item_counts(feature_path)
    feature_identity = _feature_identity(ledger)
    runs = [
        _collect_run(
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
    return _document({
        "schema_version": 1,
        "kind": "g3_rq5_initial_collection",
        "protocol_sha256": APPROVED_PROTOCOL_SHA256,
        "ledger": _file_fact(root, ledger_path) | {"logical_sha256": ledger.sha256},
        "queue_batch": _file_fact(root, batch_path) | {"batch_id": batch_id},
        "ranking_context": _file_fact(root, context_path),
        "feature_data": _file_fact(root, feature_path),
        "fixed_gate": ledger.fixed_gate.to_dict(),
        "fixed_gate_evidence": ledger.fixed_gate_evidence.to_dict(),
        "runs": runs,
    })


def build_rq5_horizon_collection(
    *, root: Path, ledger_path: Path, expected_ledger_sha256: str, batch_id: str
) -> dict[str, object]:
    from experiments.g3_pretrained_item_embeddings.launchers import rq5_horizon
    from experiments.g3_pretrained_item_embeddings.protocol.rq5_horizon_ledger import (
        load_rq5_horizon_ledger,
        verify_rq5_horizon_input_files,
    )

    root = root.resolve(strict=True)
    ledger_path = ledger_path.resolve(strict=True)
    ledger = load_rq5_horizon_ledger(
        ledger_path, root=root, expected_ledger_sha256=expected_ledger_sha256
    )
    feature_path = verify_rq5_horizon_input_files(root, ledger)
    initial = load_rq5_initial_ledger(root / ledger.initial_ledger.path)
    initial_collection = _load_json(root / ledger.initial_collection.path)
    _validate_runs_against_ledger(initial_collection["runs"], ledger=initial)
    batch_path = root / "generated/training-queue-service/batches" / f"{batch_id}.json"
    job_ids = _validate_batch(_load_json(batch_path), batch_id, len(ledger.rows))
    _require_completed_batch(root, job_ids)
    context_path = root / "generated/logs/.ranking-evidence/g3-native50m/context.pt"
    item_counts = load_training_item_counts(feature_path)
    feature_identity = _feature_identity(initial)
    artifact_filenames = dict(_ARTIFACT_FILENAMES) | {
        "job_contract": "g3_rq5_horizon_job.json"
    }
    runs = [
        _collect_run(
            root=root,
            ledger=ledger,
            ledger_path=ledger_path,
            row=row,
            batch_id=batch_id,
            job_id=job_id,
            context_path=context_path,
            item_counts=item_counts,
            feature_identity=feature_identity,
            runner_filename="run_rq5_horizon.py",
            job_environment=rq5_horizon.JOB_ENVIRONMENT,
            ledger_environment=rq5_horizon.LEDGER_ENVIRONMENT,
            artifact_filenames=artifact_filenames,
        )
        for row, job_id in zip(ledger.rows, job_ids, strict=True)
    ]
    for run, row in zip(runs, ledger.rows, strict=True):
        if run["row_id"] != row.id or run["gate_hidden_dim"] != 8:
            raise ValueError("RQ5 horizon collection differs from the exact ledger")
    frequency = [
        *(
            run
            for run in initial_collection["runs"]
            if run["family_id"] == "rq5_frequency_gate"
        ),
        *runs,
    ]
    if len(frequency) != 12:
        raise ValueError("RQ5 frequency collection changed its opportunity accounting")
    order = {f"rq5_frequency_gate:{index:02d}": index for index in range(1, 13)}
    winner = _select(frequency, order=order)
    boundaries = _boundaries(winner, frequency)
    return _document({
        "schema_version": 1,
        "kind": "g3_rq5_frequency_horizon_collection",
        "protocol_sha256": APPROVED_PROTOCOL_SHA256,
        "ledger": _file_fact(root, ledger_path) | {"logical_sha256": ledger.sha256},
        "initial_collection": ledger.initial_collection.to_dict(),
        "queue_batch": _file_fact(root, batch_path) | {"batch_id": batch_id},
        "ranking_context": _file_fact(root, context_path),
        "runs": runs,
        "frequency_selection": {
            "selected_row_id": winner["row_id"],
            "selected": dict(winner),
            "boundaries": boundaries,
            "extension_required": any(
                axis["direction"] is not None
                for axis in (
                    boundaries["embedding_learning_rate"],
                    boundaries["deep_learning_rate"],
                )
            ) or boundaries["horizon"]["extend_to_epochs"] is not None,
        },
    })


def build_rq5_final_collection(
    *, root: Path, ledger_path: Path, expected_ledger_sha256: str, batch_id: str
) -> dict[str, object]:
    from experiments.g3_pretrained_item_embeddings.launchers import (
        rq5_global_boundary,
    )
    from experiments.g3_pretrained_item_embeddings.protocol.rq5_global_boundary_ledger import (
        load_rq5_global_boundary_ledger,
        verify_rq5_global_boundary_inputs,
    )

    root = root.resolve(strict=True)
    ledger_path = ledger_path.resolve(strict=True)
    ledger = load_rq5_global_boundary_ledger(
        ledger_path, root=root, expected_ledger_sha256=expected_ledger_sha256
    )
    feature_path = verify_rq5_global_boundary_inputs(root, ledger)
    initial = load_rq5_initial_ledger(root / ledger.initial_ledger.path)
    initial_evidence = _load_json(root / ledger.initial_evidence.path)
    horizon_evidence = _load_json(root / ledger.horizon_evidence.path)
    _validate_runs_against_ledger(initial_evidence["runs"], ledger=initial)
    batch_path = root / "generated/training-queue-service/batches" / f"{batch_id}.json"
    job_ids = _validate_batch(_load_json(batch_path), batch_id, len(ledger.rows))
    _require_completed_batch(root, job_ids)
    context_path = root / "generated/logs/.ranking-evidence/g3-native50m/context.pt"
    item_counts = load_training_item_counts(feature_path)
    feature_identity = _feature_identity(initial)
    artifact_filenames = dict(_ARTIFACT_FILENAMES) | {
        "job_contract": "g3_rq5_global_boundary_job.json"
    }
    boundary_runs = [
        _collect_run(
            root=root,
            ledger=ledger,
            ledger_path=ledger_path,
            row=row,
            batch_id=batch_id,
            job_id=job_id,
            context_path=context_path,
            item_counts=item_counts,
            feature_identity=feature_identity,
            runner_filename="run_rq5_global_boundary.py",
            job_environment=rq5_global_boundary.JOB_ENVIRONMENT,
            ledger_environment=rq5_global_boundary.LEDGER_ENVIRONMENT,
            artifact_filenames=artifact_filenames,
        )
        for row, job_id in zip(ledger.rows, job_ids, strict=True)
    ]
    global_initial = [
        run
        for run in initial_evidence["runs"]
        if run["family_id"] == "rq5_global_gate"
    ]
    global_candidates = [*global_initial, *boundary_runs]
    if len(global_initial) != 12 or len(global_candidates) != 15:
        raise ValueError("RQ5 final global opportunity accounting changed")
    order = {f"rq5_global_gate:{index:02d}": index for index in range(1, 16)}
    outward_order = {f"rq5_global_gate:{index:02d}": index for index in range(13, 16)}
    global_winner = _select(global_candidates, order=order)
    outward_winner = _select(boundary_runs, order=outward_order)
    renewed_boundary = global_winner["row_id"] == "rq5_global_gate:15"
    frequency = horizon_evidence.get("frequency_selection")
    if (
        not isinstance(frequency, dict)
        or frequency.get("extension_required") is not False
        or not isinstance(frequency.get("selected"), dict)
    ):
        raise ValueError("RQ5 final frequency selection is unresolved")
    frequency_winner = frequency["selected"]
    fixed = _fixed_comparator(
        root=root,
        initial=initial,
        context_path=context_path,
        item_counts=item_counts,
    )
    acceptance = _rq5_acceptance(
        fixed=fixed,
        global_gate=global_winner,
        frequency_gate=frequency_winner,
    )
    payload = {
        "schema_version": 1,
        "kind": "g3_rq5_outcome_premechanism_native50m",
        "protocol_sha256": APPROVED_PROTOCOL_SHA256,
        "inputs": {
            "initial_evidence": ledger.initial_evidence.to_dict(),
            "frequency_horizon_evidence": ledger.horizon_evidence.to_dict(),
            "global_boundary_ledger": _file_fact(root, ledger_path)
            | {"logical_sha256": ledger.sha256},
        },
        "queue_batch": _file_fact(root, batch_path) | {"batch_id": batch_id},
        "ranking_context": _file_fact(root, context_path),
        "boundary_runs": boundary_runs,
        "global_selection": {
            "selected": dict(global_winner),
            "outward_probe_selected": dict(outward_winner),
            "new_outer_boundary_row_id": "rq5_global_gate:15",
            "outward_winner_on_new_boundary": renewed_boundary,
            "selection_resolved": not renewed_boundary,
            "next_action": "renewed_approval" if renewed_boundary else "none",
        },
        "frequency_selection": frequency,
        "fixed_comparator": fixed,
        "acceptance_analysis": acceptance,
        "selection_resolved": not renewed_boundary,
        "mechanism_evidence": {
            "status": "pending_targeted_selected_config_reproductions",
            "final_closure_allowed": False,
        },
    }
    return _document(payload)


def _fixed_comparator(
    *,
    root: Path,
    initial: Rq5InitialLedger,
    context_path: Path,
    item_counts: Mapping[int, int],
) -> dict[str, object]:
    evidence = _load_json(root / initial.fixed_gate_evidence.source_evidence.path)
    rows = evidence.get("diagnostic_tuning_ledger")
    matches = [
        row
        for row in rows
        if isinstance(row, dict)
        and row.get("row_id") == initial.fixed_gate_evidence.source_id
    ] if isinstance(rows, list) else []
    if len(matches) != 1:
        raise ValueError("RQ5 fixed comparator source evidence is not unique")
    source = matches[0]
    artifacts = source.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("RQ5 fixed comparator artifacts are absent")
    ranking_path = root / str(artifacts["ranking_evidence"]["path"])
    rankings_path = root / str(artifacts["top_item_rankings"]["path"])
    recomputed = _recompute_metrics(context_path, ranking_path, rankings_path)
    metrics = source.get("metrics")
    if not isinstance(metrics, dict) or any(
        abs(float(metrics[name]) - float(recomputed[name])) > 1e-15
        for name in recomputed
    ):
        raise ValueError("RQ5 fixed comparator metrics changed")
    return {
        "row_id": source["row_id"],
        "run_name": source["run_name"],
        "metrics": metrics,
        "slices": _ranking_slices(
            context_path=context_path,
            ranking_path=ranking_path,
            rankings_path=rankings_path,
            item_counts=item_counts,
        ),
        "source_evidence": initial.fixed_gate_evidence.source_evidence.to_dict(),
    }


def _rq5_acceptance(
    *,
    fixed: Mapping[str, object],
    global_gate: Mapping[str, object],
    frequency_gate: Mapping[str, object],
) -> dict[str, object]:
    dispersion = APPROVED_PROTOCOL.relative_dispersion("native-50m", "recall@100")
    comparators = {"fixed_gate": fixed, "global_gate": global_gate}
    aggregate = float(frequency_gate["metrics"]["recall@100"])
    tail = float(frequency_gate["slices"]["tail"]["recall@100"])
    comparisons = {}
    for name, comparator in comparators.items():
        comparator_aggregate = float(comparator["metrics"]["recall@100"])
        comparator_tail = float(comparator["slices"]["tail"]["recall@100"])
        band = abs(comparator_aggregate) * dispersion
        comparisons[name] = {
            "aggregate_recall_at_100": comparator_aggregate,
            "aggregate_delta": aggregate - comparator_aggregate,
            "aggregate_band": band,
            "aggregate_within_band": abs(aggregate - comparator_aggregate) <= band,
            "tail_recall_at_100": comparator_tail,
            "tail_delta": tail - comparator_tail,
            "tail_higher": tail > comparator_tail,
        }
    qualifies = all(
        comparison["aggregate_within_band"] and comparison["tail_higher"]
        for comparison in comparisons.values()
    )
    fixed_recall = float(fixed["metrics"]["recall@100"])
    global_recall = float(global_gate["metrics"]["recall@100"])
    fixed_band = abs(fixed_recall) * dispersion
    global_qualifies = global_recall - fixed_recall > fixed_band
    selected_treatment = (
        "frequency_gate"
        if qualifies
        else "global_gate" if global_qualifies else "fixed_gate"
    )
    return {
        "rule": (
            "frequency aggregate Recall@100 within both comparator bands and "
            "tail Recall@100 higher than both comparators"
        ),
        "relative_dispersion": dispersion,
        "frequency_aggregate_recall_at_100": aggregate,
        "frequency_tail_recall_at_100": tail,
        "comparisons": comparisons,
        "qualifies_frequency_gate": qualifies,
        "global_gate_vs_fixed": {
            "aggregate_delta": global_recall - fixed_recall,
            "required_improvement_beyond_band": fixed_band,
            "qualifies_global_gate": global_qualifies,
        },
        "acceptance_criteria": {
            "frequency_aggregate_not_worse_than_fixed": comparisons["fixed_gate"][
                "aggregate_within_band"
            ],
            "frequency_aggregate_not_worse_than_global": comparisons["global_gate"][
                "aggregate_within_band"
            ],
            "frequency_tail_higher_than_fixed": comparisons["fixed_gate"][
                "tail_higher"
            ],
            "frequency_tail_higher_than_global": comparisons["global_gate"][
                "tail_higher"
            ],
            "accepted": qualifies,
        },
        "selected_treatment": selected_treatment,
        "tail_significance_limitation": (
            "slice deltas are descriptive because no slice-specific repeat "
            "calibration exists"
        ),
    }


def select_rq5_initial_winners(
    runs: Sequence[Mapping[str, object]], *, ledger: Rq5InitialLedger
) -> dict[str, object]:
    _validate_runs_against_ledger(runs, ledger=ledger)
    global_rows = [run for run in runs if run["family_id"] == "rq5_global_gate"]
    frequency_rows = [
        run for run in runs if run["family_id"] == "rq5_frequency_gate"
    ]
    if len(global_rows) != 12 or len(frequency_rows) != 9:
        raise ValueError("RQ5 initial collection changed its opportunity accounting")
    capacities = tuple(int(run["gate_hidden_dim"]) for run in frequency_rows)
    if any(capacities.count(capacity) != 3 for capacity in (4, 8, 16)):
        raise ValueError("RQ5 frequency capacity surface is incomplete")
    order = {row.id: index for index, row in enumerate(ledger.rows)}
    global_winner = _select(global_rows, order=order)
    frequency_winner = _select(frequency_rows, order=order)
    capacity = int(frequency_winner["gate_hidden_dim"])
    if capacity not in {4, 8, 16}:
        raise ValueError("RQ5 selected frequency capacity is outside the tested surface")
    return {
        "global_gate": {
            "selected_row_id": global_winner["row_id"],
            "selected": dict(global_winner),
            "boundaries": _boundaries(global_winner, global_rows),
        },
        "frequency_capacity": {
            "selected_row_id": frequency_winner["row_id"],
            "selected_gate_hidden_dim": capacity,
            "selected": dict(frequency_winner),
            "capacity_boundary": {
                "tested_values": [4, 8, 16],
                "direction": (
                    "lower" if capacity == 4 else "upper" if capacity == 16 else None
                ),
            },
            "coordinate_boundaries": _boundaries(frequency_winner, frequency_rows),
        },
    }


def persist_rq5_initial_collection(
    path: Path,
    document: Mapping[str, object],
    *,
    root: Path,
    ledger_path: Path,
    expected_ledger_sha256: str,
    batch_id: str,
) -> Path:
    root = root.resolve(strict=True)
    destination = (root / RQ5_INITIAL_EVIDENCE_PATH).resolve()
    if path.resolve() != destination or destination.is_symlink():
        raise ValueError("RQ5 initial evidence destination is not canonical")
    rebuilt = build_rq5_initial_collection(
        root=root,
        ledger_path=ledger_path,
        expected_ledger_sha256=expected_ledger_sha256,
        batch_id=batch_id,
    )
    if dict(document) != rebuilt:
        raise ValueError("RQ5 initial evidence differs from freshly rebuilt evidence")
    content = (json.dumps(rebuilt, sort_keys=True, separators=(",", ":")) + "\n").encode()
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("xb") as handle:
            handle.write(content)
    except FileExistsError:
        if destination.read_bytes() != content:
            raise RuntimeError(f"immutable RQ5 initial evidence differs: {destination}")
    return destination


def persist_rq5_horizon_collection(
    path: Path,
    document: Mapping[str, object],
    *,
    root: Path,
    ledger_path: Path,
    expected_ledger_sha256: str,
    batch_id: str,
) -> Path:
    root = root.resolve(strict=True)
    destination = (root / RQ5_HORIZON_EVIDENCE_PATH).resolve()
    if path.resolve() != destination or destination.is_symlink():
        raise ValueError("RQ5 horizon evidence destination is not canonical")
    rebuilt = build_rq5_horizon_collection(
        root=root,
        ledger_path=ledger_path,
        expected_ledger_sha256=expected_ledger_sha256,
        batch_id=batch_id,
    )
    if not _exact_json_equal(dict(document), rebuilt):
        raise ValueError("RQ5 horizon evidence differs from freshly rebuilt evidence")
    content = (json.dumps(rebuilt, sort_keys=True, separators=(",", ":")) + "\n").encode()
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("xb") as handle:
            handle.write(content)
    except FileExistsError:
        if destination.read_bytes() != content:
            raise RuntimeError(f"immutable RQ5 horizon evidence differs: {destination}")
    return destination


def persist_rq5_outcome_collection(
    path: Path,
    document: Mapping[str, object],
    *,
    root: Path,
    ledger_path: Path,
    expected_ledger_sha256: str,
    batch_id: str,
) -> Path:
    root = root.resolve(strict=True)
    destination = (root / RQ5_OUTCOME_EVIDENCE_PATH).resolve()
    if path.resolve() != destination or destination.is_symlink():
        raise ValueError("RQ5 outcome evidence destination is not canonical")
    rebuilt = build_rq5_final_collection(
        root=root,
        ledger_path=ledger_path,
        expected_ledger_sha256=expected_ledger_sha256,
        batch_id=batch_id,
    )
    if not _exact_json_equal(dict(document), rebuilt):
        raise ValueError("RQ5 outcome evidence differs from freshly rebuilt evidence")
    content = (json.dumps(rebuilt, sort_keys=True, separators=(",", ":")) + "\n").encode()
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("xb") as handle:
            handle.write(content)
    except FileExistsError:
        if destination.read_bytes() != content:
            raise RuntimeError(f"immutable RQ5 outcome evidence differs: {destination}")
    return destination


def _validate_runs_against_ledger(
    runs: Sequence[Mapping[str, object]], *, ledger: Rq5InitialLedger
) -> None:
    if len(runs) != len(ledger.rows):
        raise ValueError("RQ5 collected run count differs from the ledger")
    by_id = {str(run.get("row_id")): run for run in runs}
    if len(by_id) != len(runs) or set(by_id) != {row.id for row in ledger.rows}:
        raise ValueError("RQ5 collected row identities differ from the ledger")
    for row in ledger.rows:
        run = by_id[row.id]
        expected = {
            "family_id": row.family_id,
            "run_name": row.run_name,
            "content_gate": row.content_gate,
            "gate_hidden_dim": row.gate_hidden_dim,
            "embedding_learning_rate": row.embedding_learning_rate,
            "deep_learning_rate": row.deep_learning_rate,
            "horizon_epochs": row.horizon_epochs,
        }
        if any(run.get(name) != value for name, value in expected.items()):
            raise ValueError(f"RQ5 collected coordinate differs for {row.id}")


def _select(
    runs: Sequence[Mapping[str, object]], *, order: Mapping[str, int]
) -> Mapping[str, object]:
    def key(run: Mapping[str, object]) -> tuple[float, float, float, int]:
        metrics = run.get("metrics")
        if not isinstance(metrics, dict):
            raise ValueError("RQ5 selection metrics are absent")
        recall = metrics.get("recall@100")
        ndcg = metrics.get("ndcg@100")
        wall = run.get("queue_wall_seconds")
        row_id = run.get("row_id")
        if (
            not _finite(recall)
            or not _finite(ndcg)
            or not _finite(wall)
            or not isinstance(row_id, str)
            or row_id not in order
        ):
            raise ValueError("RQ5 selection coordinate is invalid")
        return (-float(recall), -float(ndcg), float(wall), order[row_id])

    return min(runs, key=key)


def _boundaries(
    winner: Mapping[str, object], runs: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    def boundary(name: str) -> dict[str, object]:
        values = sorted({float(run[name]) for run in runs})
        selected = float(winner[name])
        span = values[-1] - values[0]
        lower_limit = values[0] + 0.1 * span
        upper_limit = values[-1] - 0.1 * span
        return {
            "selected": selected,
            "tested_min": values[0],
            "tested_max": values[-1],
            "outer_fraction": 0.1,
            "direction": (
                "lower" if selected <= lower_limit else "upper" if selected >= upper_limit else None
            ),
        }

    horizon = int(winner["horizon_epochs"])
    best_epoch = int(winner["best_epoch"])
    tested_horizons = sorted({int(run["horizon_epochs"]) for run in runs})
    return {
        "embedding_learning_rate": boundary("embedding_learning_rate"),
        "deep_learning_rate": boundary("deep_learning_rate"),
        "horizon": {
            "selected_epochs": horizon,
            "restored_best_epoch": best_epoch,
            "tested_values": tested_horizons,
            "extend_to_epochs": (
                60 if horizon == tested_horizons[-1] and best_epoch == horizon else None
            ),
        },
    }


def _validate_batch(
    batch: Mapping[str, object], batch_id: str, expected_jobs: int
) -> list[str]:
    jobs = batch.get("jobs")
    if (
        set(batch) != {"id", "jobs", "sealed", "sealed_at", "submitted_at"}
        or batch.get("id") != batch_id
        or batch.get("sealed") is not True
        or not isinstance(jobs, list)
        or len(jobs) != expected_jobs
        or len(set(jobs)) != expected_jobs
        or any(not isinstance(job_id, str) or not job_id for job_id in jobs)
        or not _finite(batch.get("submitted_at"))
        or not _finite(batch.get("sealed_at"))
        or float(batch["sealed_at"]) < float(batch["submitted_at"])
    ):
        raise ValueError("RQ5 queue batch is not the exact sealed ledger batch")
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
        raise RuntimeError(f"RQ5 batch has {len(incomplete)} incomplete jobs")


def _collect_run(
    *,
    root: Path,
    ledger: Rq5InitialLedger,
    ledger_path: Path,
    row: Rq5InitialLedgerRow,
    batch_id: str,
    job_id: str,
    context_path: Path,
    item_counts: Mapping[int, int],
    feature_identity: _FeatureIdentity,
    runner_filename: str = "run_rq5_initial.py",
    job_environment: str = JOB_ENVIRONMENT,
    ledger_environment: str = LEDGER_ENVIRONMENT,
    artifact_filenames: Mapping[str, str] = _ARTIFACT_FILENAMES,
) -> dict[str, object]:
    completed_path = root / "generated/training-queue-service/completed" / f"{job_id}.json"
    queue = _load_json(completed_path)
    expected_script = (
        root / "experiments/g3_pretrained_item_embeddings/launchers" / runner_filename
    ).resolve(strict=True)
    if (
        set(queue) != _QUEUE_KEYS
        or queue.get("id") != job_id
        or queue.get("batch_id") != batch_id
        or queue.get("run") != row.run_name
        or queue.get("exit_code") != 0
        or queue.get("data_group") != "g3-native50m-likes"
        or Path(str(queue.get("script"))).resolve() != expected_script
        or not _ordered_times(queue)
    ):
        raise ValueError(f"RQ5 queue completion differs for {row.id}")
    environment = queue.get("environment")
    pairs = [
        entry.split("=", 1)
        for entry in environment
        if isinstance(entry, str) and "=" in entry
    ] if isinstance(environment, list) else []
    values = dict(pairs)
    expected_environment = {
        "WANDB_MODE": "offline",
        job_environment: values.get(job_environment),
        ledger_environment: str(ledger_path),
        **dict(entry.split("=", 1) for entry in G3_CPU_THREAD_ENVIRONMENT),
    }
    if len(pairs) != len(values) or values != expected_environment:
        raise ValueError(f"RQ5 queue environment differs for {row.id}")
    compiled = decode_control_job(str(values[job_environment]), ledger)
    if compiled.row_id != row.id or compiled.job != row.to_dict():
        raise ValueError(f"RQ5 queue payload differs for {row.id}")
    directory = root / "generated/logs" / row.run_name
    artifact_paths = {
        name: directory / filename for name, filename in artifact_filenames.items()
    }
    contract = _load_json(artifact_paths["job_contract"])
    if contract != compiled.to_dict() | {
        "ledger_path": str(ledger_path),
        "ledger_sha256": ledger.sha256,
    }:
        raise ValueError(f"RQ5 job contract differs for {row.id}")
    _validate_metadata(_load_json(artifact_paths["training_metadata"]), row)
    metrics = _load_json(artifact_paths["final_metrics"])
    if set(metrics) != {*_METRIC_NAMES, "num_users"} or any(
        not _finite(value) for value in metrics.values()
    ):
        raise ValueError(f"RQ5 metric schema differs for {row.id}")
    recomputed = _recompute_metrics(
        context_path,
        artifact_paths["ranking_evidence"],
        artifact_paths["top_item_rankings"],
    )
    if set(recomputed) != set(metrics) or any(
        abs(float(metrics[name]) - float(recomputed[name])) > 1e-15
        for name in metrics
    ):
        raise ValueError(f"RQ5 metrics differ from ranking evidence for {row.id}")
    diagnostics = _load_json(artifact_paths["training_diagnostics"])
    identity = _validate_training_diagnostics(
        diagnostics,
        feature_identity=feature_identity,
        horizon_epochs=row.horizon_epochs,
        catalog_representation="learned_id",
    )
    nonfinite_count = sum(int(value) for value in _values_named(diagnostics, "nonfinite_count"))
    if nonfinite_count:
        raise ValueError(f"RQ5 diagnostics contain nonfinite values for {row.id}")
    verify_unique_completed_run(
        root / "generated/training-queue-service",
        run_name=row.run_name,
        expected_job_id=job_id,
    )
    verify_artifacts_in_job_window(
        tuple(artifact_paths.values()),
        dispatched_at=float(queue["dispatched_at"]),
        finished_at=float(queue["finished_at"]),
        run_label=row.id,
    )
    metadata = _load_json(artifact_paths["training_metadata"])
    wall = float(queue["finished_at"]) - float(queue["dispatched_at"])
    return {
        "row_id": row.id,
        "family_id": row.family_id,
        "run_name": row.run_name,
        "content_gate": row.content_gate,
        "gate_hidden_dim": row.gate_hidden_dim,
        "embedding_learning_rate": row.embedding_learning_rate,
        "deep_learning_rate": row.deep_learning_rate,
        "horizon_epochs": row.horizon_epochs,
        "best_epoch": metadata["best_epoch"],
        "epochs_trained": metadata["epochs_trained"],
        "queue_wall_seconds": wall,
        "metrics": metrics,
        "metric_provenance": {
            "recomputed_from_ranking_evidence": True,
            "absolute_tolerance": 1e-15,
            "num_users": int(recomputed["num_users"]),
        },
        "slices": _ranking_slices(
            context_path=context_path,
            ranking_path=artifact_paths["ranking_evidence"],
            rankings_path=artifact_paths["top_item_rankings"],
            item_counts=item_counts,
        ),
        "efficiency": _efficiency(
            metadata=metadata,
            log_path=artifact_paths["sweep_log"],
            queue_wall_seconds=wall,
        ),
        "diagnostics": {
            "schema_version": identity.schema_version,
            "epochs": identity.epoch_count,
            "nonfinite_measurement_count": nonfinite_count,
        },
        "queue_job": _file_fact(root, completed_path) | {"job_id": job_id},
        "artifacts": {
            name: _file_fact(root, path) for name, path in artifact_paths.items()
        },
    }


def _validate_metadata(metadata: Mapping[str, object], row: Rq5InitialLedgerRow) -> None:
    representation = {
        "catalog_representation": "learned_id",
        "content_gate": row.content_gate,
        "extra_item_id_dim": None,
        "gate_hidden_dim": row.gate_hidden_dim,
        "history_hidden_dim": row.history_hidden_dim,
        "history_representation": "id_content",
        "metadata": [],
        "metadata_dim": None,
    }
    row_representation = row.to_dict().get("representation")
    if isinstance(row_representation, dict) and isinstance(
        row_representation.get("frequency_gate_semantics"), str
    ):
        representation["frequency_gate_semantics"] = row_representation[
            "frequency_gate_semantics"
        ]
    expected = {
        "batch_size": row.batch_size,
        "seed": row.seed,
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
        raise ValueError(f"RQ5 runtime metadata differs for {row.id}")
    best_epoch = metadata.get("best_epoch")
    if type(best_epoch) is not int or not 1 <= best_epoch <= row.horizon_epochs:
        raise ValueError(f"RQ5 restored epoch is invalid for {row.id}")
    if metadata.get("best_epoch_at_cap") is not (best_epoch == row.horizon_epochs):
        raise ValueError(f"RQ5 restored-epoch cap flag differs for {row.id}")
    traces = metadata.get("lr_group_traces")
    if not isinstance(traces, dict) or any(
        not isinstance(traces.get(group), list)
        or len(traces[group]) != row.horizon_epochs
        or traces[group][-1] != 0.0
        for group in ("embedding", "deep")
    ):
        raise ValueError(f"RQ5 schedule trace is incomplete for {row.id}")
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
        raise ValueError(f"RQ5 runtime invariants differ for {row.id}")


def _feature_identity(ledger: Rq5InitialLedger) -> _FeatureIdentity:
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


def _ordered_times(job: Mapping[str, object]) -> bool:
    values = tuple(job.get(name) for name in ("submitted_at", "dispatched_at", "finished_at"))
    return all(_finite(value) for value in values) and float(values[0]) <= float(values[1]) <= float(values[2])


def _finite(value: object) -> bool:
    return type(value) in {int, float} and math.isfinite(float(value))


def _values_named(value: object, name: str) -> list[float]:
    if isinstance(value, dict):
        return [
            found
            for key, item in value.items()
            for found in ([item] if key == name and _finite(item) else _values_named(item, name))
        ]
    if isinstance(value, list):
        return [found for item in value for found in _values_named(item, name)]
    return []


def _document(payload: Mapping[str, object]) -> dict[str, object]:
    document = dict(payload)
    document["sha256"] = hashlib.sha256(
        json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    return document


def _exact_json_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(
            _exact_json_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _exact_json_equal(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    return left == right
