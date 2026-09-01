from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence

import polars as pl

from dcn.eval.ranking_evidence import _load_payload, load_ranking_evidence
from experiments.g4_future_items.protocol.manifest import (
    BASE_HORIZON_VALUES,
    DEEP_LR_BOUNDS,
    EMBEDDING_LR_BOUNDS,
    canonical_bytes,
    canonical_sha256,
    load_ledger,
    load_strict_json,
    verify_ledger_semantics,
)
from experiments.g4_future_items.report.artifacts import read_recommender_trial
from experiments.g4_future_items.report.evaluation import (
    _ranking_snapshot,
    evaluate_slices,
)
from experiments.g4_future_items.report.selection import (
    RecommenderTrial,
    boundary_direction,
    select_recommender_trial,
)
from experiments.g4_future_items.report.slices import _score_slice


_EVIDENCE_PATH = Path(
    "experiments/g4_future_items/evidence/rq1_rq2_evaluation_native50m.json"
)
_PROTOCOL_PATH = Path("experiments/g4_future_items/protocol")
_COMPATIBILITY_PATH = _PROTOCOL_PATH / "treatment_semantics_compatibility_v8.json"
_LOGS_PATH = Path("generated/logs")
_QUEUE_PATH = Path("generated/training-queue-service")
_FINAL_INTERVAL_SECONDS = 7 * 24 * 60 * 60
_METRIC_NAMES = tuple(
    f"{metric}@{cutoff}"
    for metric in ("recall", "capped_recall", "ndcg", "mrr", "coverage")
    for cutoff in (10, 50, 100)
)
_COMPARISONS = {
    "rq1_24h_vs_control": ("control_next_item", "rq1_24h"),
    "rq2_next10_vs_control": ("control_next_item", "rq2_next10"),
    "rq2_next10_vs_rq1_24h": ("rq1_24h", "rq2_next10"),
}
_SELECTION_STAGES = {
    "control_base": (
        "control_next_item",
        "ledgers/control_tuning.json",
        "50241e5e4b5a401bb9953f2c4d386e7e",
    ),
    "control_boundary_round_1": (
        "control_next_item",
        "ledgers/control_tuning_boundary_r1.json",
        "990885face7940c5a2c65277d620b3c4",
    ),
    "rq1_base": (
        "rq1_24h",
        "ledgers/rq1_tuning.json",
        "f26c7791b074436d8bd9ba0005dbec2a",
    ),
    "rq2_base": (
        "rq2_next10",
        "ledgers/rq2_tuning.json",
        "09b39c2f95104ff9b1edc691d7b9a679",
    ),
}
_CALIBRATION_BATCH_IDS = (
    "278bc349968e43c89e70f1cf696c1417",
    "5c6250d40aa84ad5925f8389069c2b47",
)
_CALIBRATION_CONFIGURATION_FIELDS = (
    "batch_size",
    "dataset_size",
    "deep_learning_rate",
    "effective_batch_size",
    "embedding_learning_rate",
    "gradient_accumulation_steps",
    "initializer_std",
    "item_embedding_dim",
    "max_epochs",
    "model_dim",
    "negative_sampling",
    "num_epochs",
    "optimizer_steps_per_epoch",
    "physical_batch_size",
    "targets_per_epoch",
    "tokens_per_epoch",
    "training_semantics_revision",
    "transfer_invariants",
    "weight_decay",
)
_GENERATOR_SOURCES = (
    "dcn/eval/ranking_evidence.py",
    "dcn/eval/ranking_metrics.py",
    "experiments/g4_future_items/protocol/manifest.py",
    "experiments/g4_future_items/protocol/plan.md",
    "experiments/g4_future_items/report/artifacts.py",
    "experiments/g4_future_items/report/evaluation.py",
    "experiments/g4_future_items/report/rq1_rq2_evidence.py",
    "experiments/g4_future_items/report/selection.py",
    "experiments/g4_future_items/report/slices.py",
)


def load_verified_tuning_ledger(
    path: Path,
    *,
    reference_paths: dict[str, Path],
    compatibility_path: Path | None = None,
) -> dict[str, Any]:
    ledger = load_ledger(path)
    verify_ledger_semantics(
        ledger,
        reference_paths,
        compatibility_path=compatibility_path,
    )
    return ledger


def verify_ranking_artifacts(
    *,
    context_path: Path,
    ranking_path: Path,
    top_item_rankings_path: Path,
    relevance_by_user: Mapping[int, set[int]],
) -> dict[str, Any]:
    rankings, snapshot = _ranking_snapshot(top_item_rankings_path)
    evidence = load_ranking_evidence(context_path, ranking_path)
    users = [int(user_id) for user_id in evidence.user_ids.tolist()]
    if users != list(rankings) or set(users) != set(relevance_by_user):
        raise ValueError("ranking evidence users differ from ranking and relevance")
    if evidence.max_k != snapshot["max_k"]:
        raise ValueError("ranking evidence max_k differs from top-item rankings")

    relevant_items = [int(item_id) for item_id in evidence.relevant_item_ids.tolist()]
    relevant_ranks = [int(rank) for rank in evidence.relevant_ranks.tolist()]
    offsets = [int(offset) for offset in evidence.relevance_offsets.tolist()]
    for position, user_id in enumerate(users):
        start, end = offsets[position : position + 2]
        saved_items = relevant_items[start:end]
        if len(saved_items) != len(set(saved_items)) or set(saved_items) != (
            relevance_by_user[user_id]
        ):
            raise ValueError("ranking evidence relevance differs from mapped targets")
        ranking_positions = {
            item_id: rank
            for rank, item_id in enumerate(rankings[user_id], start=1)
        }
        expected_ranks = [ranking_positions.get(item_id, 0) for item_id in saved_items]
        if relevant_ranks[start:end] != expected_ranks:
            raise ValueError("ranking evidence ranks differ from top-item rankings")

    ranking_payload = _load_payload(ranking_path)
    context_sha256 = ranking_payload.get("context_sha256")
    if not isinstance(context_sha256, str):
        raise ValueError("ranking evidence has no context identity")
    return {
        "context_payload_sha256": context_sha256,
        "num_users": len(users),
        "num_relevant_items": len(relevant_items),
        "max_k": evidence.max_k,
    }


def build_rq1_rq2_evidence(repo_root: Path) -> dict[str, Any]:
    root = repo_root.resolve(strict=True)
    protocol = _protocol_evidence(root)
    selection, selected = _selection_evidence(root, protocol)
    overall, slices, slice_protocol, selected_runs = _evaluation_evidence(
        root, protocol, selected
    )
    calibration = _calibration_evidence(root, overall["rows"]["control_next_item"])
    band = calibration["operational_bands_from_current_control"]["recall@100"]
    qualification = {}
    for role in ("rq1_24h", "rq2_next10"):
        reference = overall["rows"]["control_next_item"]["recall@100"]
        candidate = overall["rows"][role]["recall@100"]
        delta = candidate - reference
        qualification[role] = {
            "operational_band_points": band,
            "qualifies_for_aggregate": delta > band,
            "recall_at_100_delta_percent": 100.0 * delta / reference,
            "recall_at_100_delta_points": delta,
        }
    overall["aggregate_qualification"] = qualification
    return {
        "schema_version": 3,
        "kind": "g4_rq1_rq2_evaluation_native50m",
        "dataset_size": "native-50m",
        "protocol": protocol,
        "selection_provenance": selection,
        "selected_runs": selected_runs,
        "calibration": calibration,
        "overall": overall,
        "slice_protocol": slice_protocol,
        "slices": slices,
    }


def write_rq1_rq2_evidence(artifact_path: Path, *, repo_root: Path) -> str:
    payload = canonical_bytes(build_rq1_rq2_evidence(repo_root))
    digest = hashlib.sha256(payload).hexdigest()
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_bytes(payload)
    artifact_path.with_suffix(".sha256").write_text(digest)
    return digest


def verify_rq1_rq2_evidence(
    artifact_path: Path, *, repo_root: Path
) -> dict[str, Any]:
    payload = artifact_path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if artifact_path.with_suffix(".sha256").read_text() != digest:
        raise ValueError("RQ1/RQ2 evidence digest differs from its sidecar")
    document = load_strict_json(artifact_path)
    if canonical_bytes(document) != payload:
        raise ValueError("RQ1/RQ2 evidence is not canonical JSON")
    if document.get("schema_version") != 3 or document.get("kind") != (
        "g4_rq1_rq2_evaluation_native50m"
    ):
        raise ValueError("unsupported RQ1/RQ2 evidence schema")
    expected = build_rq1_rq2_evidence(repo_root)
    if document != expected:
        raise ValueError("RQ1/RQ2 evidence differs from its bound source artifacts")
    return document


def _protocol_evidence(root: Path) -> dict[str, Any]:
    paths = {
        "control_semantics": root / _PROTOCOL_PATH / "control_semantics_manifest.json",
        "selected_control": root / _PROTOCOL_PATH / "selected_control_manifest.json",
        "treatment_semantics": root
        / _PROTOCOL_PATH
        / "treatment_semantics_manifest.json",
        "treatment_compatibility": root / _COMPATIBILITY_PATH,
    }
    documents = {name: load_strict_json(path) for name, path in paths.items()}
    hashes = {name: canonical_sha256(value) for name, value in documents.items()}
    if documents["selected_control"]["control_semantics_manifest_sha256"] != hashes[
        "control_semantics"
    ]:
        raise ValueError("selected control is not bound to control semantics")
    if documents["treatment_semantics"]["selected_control_manifest_sha256"] != (
        hashes["selected_control"]
    ):
        raise ValueError("treatment semantics are not bound to selected control")
    data_identity = documents["control_semantics"]["data_identity"]
    _require_external_identity(Path(data_identity["main"]["path"]), data_identity["main"])
    return {
        "final_interval_seconds": _FINAL_INTERVAL_SECONDS,
        "split_cutoff_timestamp": data_identity["split_cutoff_timestamp"],
        "mapped_events": data_identity["main"],
        "mapped_catalog_sha256": data_identity["mapped_catalog_sha256"],
        "training_semantics_revision": documents["control_semantics"][
            "training_semantics_revisions"
        ]["generation"],
        "target_seed_revision": documents["treatment_semantics"][
            "schema_revisions"
        ]["target_rng"],
        "semantics": {
            name: {
                "canonical_sha256": hashes[name],
                "file": _file_fact(root, paths[name]),
            }
            for name in paths
        },
        "generator_sources": {
            path: _file_fact(root, root / path) for path in _GENERATOR_SOURCES
        },
    }


def _selection_evidence(
    root: Path, protocol: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    stages = {
        name: _selection_stage(root, name, role, ledger_path, batch_id, protocol)
        for name, (role, ledger_path, batch_id) in _SELECTION_STAGES.items()
    }
    budgets = {canonical_sha256(stage["training_budget"]) for stage in stages.values()}
    if len(budgets) != 1:
        raise ValueError("control and RQ1/RQ2 tuning budgets differ")
    control_base = stages["control_base"]
    control_boundary = stages["control_boundary_round_1"]
    entering = control_boundary["ledger_document"]["entering_row"]
    if entering != control_base["winner_row"]:
        raise ValueError("control boundary did not enter from the base winner")
    if canonical_sha256(entering) != control_boundary["ledger_document"][
        "entering_row_sha256"
    ]:
        raise ValueError("control boundary entering-row hash differs")
    control_winner_trial = select_recommender_trial(
        [control_base["winner_trial"], *control_boundary["trials"]],
        objective="control_tuning_boundary",
    )
    control_winner = _trial_document(root, control_winner_trial)
    selected_control = load_strict_json(
        root / _PROTOCOL_PATH / "selected_control_manifest.json"
    )
    _require_selected_control(selected_control, control_winner)
    if selected_control["ledger_sha256"] != control_base["ledger_document"]["sha256"]:
        raise ValueError("selected control is not bound to its winning control ledger")
    selected_parameters = dict(
        selected_control["selection"]["canonical_parameters"]
    )
    selected_parameters.pop("batch_size")
    for stage_name in ("rq1_base", "rq2_base"):
        if stages[stage_name]["ledger_document"]["anchor_parameters"] != (
            selected_parameters
        ):
            raise ValueError(f"{stage_name} anchor differs from the selected control")

    control_base_decision = _boundary_decision(
        control_base["winner_trial"],
        rate_bounds={
            "embedding_learning_rate": list(EMBEDDING_LR_BOUNDS),
            "deep_learning_rate": list(DEEP_LR_BOUNDS),
        },
        horizon_values=BASE_HORIZON_VALUES,
    )
    control_final_decision = _boundary_decision(
        control_winner_trial,
        rate_bounds=control_boundary["ledger_document"]["rate_bounds"],
        horizon_values=BASE_HORIZON_VALUES,
    )

    rq1_winner_trial = stages["rq1_base"]["winner_trial"]
    rq2_winner_trial = stages["rq2_base"]["winner_trial"]
    results = {
        "control_next_item": {
            "candidate_count": len(control_base["candidates"])
            + len(control_boundary["candidates"]),
            "stages": {
                "base": {
                    **_public_stage(control_base),
                    "boundary_decision": control_base_decision,
                },
                "boundary_round_1": {
                    **_public_stage(control_boundary),
                    "entering_winner": _trial_document(
                        root, control_base["winner_trial"]
                    ),
                    "cumulative_winner": control_winner,
                    "boundary_decision": control_final_decision,
                },
            },
            "candidates": [
                *control_base["candidates"],
                *control_boundary["candidates"],
            ],
            "winner": control_winner,
            "final_boundary_decision": control_final_decision,
        },
        "rq1_24h": _base_selection_result(stages["rq1_base"], root),
        "rq2_next10": _base_selection_result(stages["rq2_base"], root),
    }
    for role, result in results.items():
        if result["final_boundary_decision"]["requires_extension"]:
            raise ValueError(f"{role} winner still requires a boundary extension")
    selected = {
        "control_next_item": {
            "trial": control_winner_trial,
            "candidate": _candidate_by_row(results["control_next_item"], control_winner_trial.row_id),
        },
        "rq1_24h": {
            "trial": rq1_winner_trial,
            "candidate": _candidate_by_row(results["rq1_24h"], rq1_winner_trial.row_id),
        },
        "rq2_next10": {
            "trial": rq2_winner_trial,
            "candidate": _candidate_by_row(results["rq2_next10"], rq2_winner_trial.row_id),
        },
    }
    return results, selected


def _selection_stage(
    root: Path,
    name: str,
    role: str,
    ledger_relative_path: str,
    batch_id: str,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    ledger_path = root / _PROTOCOL_PATH / ledger_relative_path
    ledger = load_verified_tuning_ledger(
        ledger_path,
        reference_paths=_ledger_reference_paths(root, role),
        compatibility_path=root / _COMPATIBILITY_PATH,
    )
    _require_ledger_semantics(ledger, protocol, role)
    batch, queue_jobs = _queue_batch(root, batch_id)
    candidates = []
    trials = []
    training_budget = None
    for row in ledger["rows"]:
        run_name = row["job"]["run_name"]
        run_directory = root / _LOGS_PATH / run_name
        contract = load_strict_json(run_directory / "g4_job.json")
        if (
            contract["row_id"] != row["id"]
            or contract["job"] != row["job"]
            or contract["ledger_sha256"] != ledger["sha256"]
            or contract["ledger_stage"] != ledger["stage"]
            or Path(contract["ledger_path"]).resolve() != ledger_path.resolve()
        ):
            raise ValueError(f"candidate contract differs from ledger row {row['id']}")
        metadata = load_strict_json(run_directory / "training_metadata.json")
        _require_candidate_metadata(metadata, role, row["job"], protocol)
        candidate_budget = {
            name: metadata[name]
            for name in (
                "batch_size",
                "effective_batch_size",
                "targets_per_epoch",
                "optimizer_steps_per_epoch",
            )
        }
        if training_budget is None:
            training_budget = candidate_budget
        elif candidate_budget != training_budget:
            raise ValueError(f"candidate budget differs inside stage {name}")
        trial = read_recommender_trial(run_directory)
        if not trial.usable:
            raise ValueError(f"candidate is not usable: {run_name}")
        queue_job = queue_jobs.get(run_name)
        if queue_job is None:
            raise ValueError(f"candidate is absent from queue batch {batch_id}: {run_name}")
        trials.append(trial)
        candidates.append(
            {
                **_trial_document(root, trial),
                "artifacts": {
                    "job_contract": _file_fact(root, run_directory / "g4_job.json"),
                    "training_metadata": _file_fact(
                        root, run_directory / "training_metadata.json"
                    ),
                    "sweep_log": _file_fact(root, run_directory / "sweep.log"),
                },
                "queue_job": queue_job,
            }
        )
    if set(queue_jobs) != {candidate["run_name"] for candidate in candidates}:
        raise ValueError(f"queue batch contains runs outside frozen stage {name}")
    winner = select_recommender_trial(trials, objective=ledger["stage"])
    winner_row = next(row for row in ledger["rows"] if row["id"] == winner.row_id)
    assert training_budget is not None
    return {
        "name": name,
        "role": role,
        "ledger": _ledger_identity(root, ledger_path, ledger),
        "ledger_document": ledger,
        "queue_batch": batch,
        "training_budget": training_budget,
        "candidates": candidates,
        "trials": trials,
        "winner_trial": winner,
        "winner_row": winner_row,
    }


def _public_stage(stage: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_count": len(stage["candidates"]),
        "ledger": stage["ledger"],
        "queue_batch": stage["queue_batch"],
        "training_budget": stage["training_budget"],
        "winner": _candidate_by_row(
            {"candidates": stage["candidates"]}, stage["winner_trial"].row_id
        ),
    }


def _base_selection_result(stage: dict[str, Any], root: Path) -> dict[str, Any]:
    winner = stage["winner_trial"]
    decision = _boundary_decision(
        winner,
        rate_bounds={
            "embedding_learning_rate": list(EMBEDDING_LR_BOUNDS),
            "deep_learning_rate": list(DEEP_LR_BOUNDS),
        },
        horizon_values=BASE_HORIZON_VALUES,
    )
    return {
        "candidate_count": len(stage["candidates"]),
        "stages": {
            "base": {**_public_stage(stage), "boundary_decision": decision}
        },
        "candidates": stage["candidates"],
        "winner": _trial_document(root, winner),
        "final_boundary_decision": decision,
    }


def _boundary_decision(
    trial: RecommenderTrial,
    *,
    rate_bounds: Mapping[str, Sequence[float]],
    horizon_values: Sequence[int],
) -> dict[str, Any]:
    rate_directions = {
        name: boundary_direction(float(trial.parameters[name]), float(bounds[0]), float(bounds[1]))
        for name, bounds in rate_bounds.items()
    }
    ordered_horizons = tuple(int(value) for value in horizon_values)
    horizon_direction = None
    if trial.horizon_epochs == ordered_horizons[0]:
        horizon_direction = "lower"
    elif trial.horizon_epochs == ordered_horizons[-1]:
        horizon_direction = "upper"
    requires_extension = horizon_direction is not None or any(rate_directions.values())
    return {
        "rate_bounds": {name: list(bounds) for name, bounds in rate_bounds.items()},
        "rate_directions": rate_directions,
        "horizon_domain": list(ordered_horizons),
        "horizon_direction": horizon_direction,
        "requires_extension": requires_extension,
    }


def _trial_document(root: Path, trial: RecommenderTrial) -> dict[str, Any]:
    metadata = load_strict_json(
        root / _LOGS_PATH / trial.run_name / "training_metadata.json"
    )
    return {
        "row_id": trial.row_id,
        "run_name": trial.run_name,
        "parameters": {
            **trial.parameters,
            "lr_schedule_horizon_epochs": trial.horizon_epochs,
        },
        "validation_recall_at_100": trial.validation_recall_at_100,
        "validation_loss": trial.validation_loss,
        "declared_horizon_epochs": trial.horizon_epochs,
        "epochs_trained": trial.epochs_trained,
        "restored_best_epoch": metadata["best_epoch"],
    }


def _candidate_by_row(selection: dict[str, Any], row_id: str) -> dict[str, Any]:
    return next(
        candidate for candidate in selection["candidates"] if candidate["row_id"] == row_id
    )


def _require_selected_control(
    selected_control: dict[str, Any], winner: dict[str, Any]
) -> None:
    expected = selected_control["selection"]
    fields = {
        "row_id": winner["row_id"],
        "run_name": winner["run_name"],
        "best_epoch": winner["restored_best_epoch"],
        "epochs_trained": winner["epochs_trained"],
        "validation_recall_at_100": winner["validation_recall_at_100"],
        "validation_loss": winner["validation_loss"],
        "canonical_parameters": winner["parameters"],
    }
    if expected != fields:
        raise ValueError("selected-control manifest differs from deterministic winner")


def _require_candidate_metadata(
    metadata: dict[str, Any],
    role: str,
    job: dict[str, Any],
    protocol: dict[str, Any],
) -> None:
    if (
        metadata["seed"] != job["seed"]
        or metadata["training_semantics_revision"]
        != protocol["training_semantics_revision"]
    ):
        raise ValueError(f"{role} runtime metadata has different seed or semantics")
    if role == "control_next_item":
        if metadata.get("g4_objective_id") is not None:
            raise ValueError("control runtime metadata unexpectedly declares a treatment")
        return
    objective = job["objective"]
    loss = job["loss"]
    expected = {
        "g4_objective_id": objective["id"],
        "g4_objective_window_seconds": objective.get("window_seconds"),
        "g4_objective_event_lookahead": objective.get("event_lookahead"),
        "g4_objective_period_count": objective.get("period_count"),
        "g4_valid_positive_mask_mode": loss["valid_positive_mask_mode"],
        "g4_target_seed_revision": protocol["target_seed_revision"],
    }
    if any(metadata.get(name) != value for name, value in expected.items()):
        raise ValueError(f"{role} runtime target semantics differ from its contract")


def _evaluation_evidence(
    root: Path,
    protocol: dict[str, Any],
    selected: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    rankings = {}
    snapshots = {}
    selected_runs = {}
    for role, value in selected.items():
        run_name = value["trial"].run_name
        run_directory = root / _LOGS_PATH / run_name
        ranking_path = run_directory / "top_item_rankings.json"
        rankings[role], snapshots[role] = _ranking_snapshot(ranking_path)
        selected_runs[role] = {
            **_trial_document(root, value["trial"]),
            "ledger_row_artifacts": value["candidate"]["artifacts"],
            "evaluation_artifacts": {
                name: _file_fact(root, run_directory / filename)
                for name, filename in (
                    ("final_metrics", "final_metrics.json"),
                    ("ranking_evidence", "ranking_evidence.pt"),
                    ("top_item_rankings", "top_item_rankings.json"),
                )
            },
        }
    identity = _ranking_identity(snapshots)
    if identity["catalog_sha256"] != protocol["mapped_catalog_sha256"]:
        raise ValueError("ranking catalog differs from frozen mapped catalog")
    protocol["ranking_identity"] = identity

    users = sorted(rankings["control_next_item"])
    if any(sorted(value) != users for value in rankings.values()):
        raise ValueError("selected rankings have different evaluation users")
    final = (
        pl.scan_parquet(Path(protocol["mapped_events"]["path"]))
        .filter(
            (pl.col("event_type") == "like")
            & pl.col("uid").is_in(users)
            & (pl.col("timestamp") > protocol["split_cutoff_timestamp"])
            & (
                pl.col("timestamp")
                <= protocol["split_cutoff_timestamp"] + _FINAL_INTERVAL_SECONDS
            )
        )
        .select("uid", "compact_item_id")
        .collect(engine="streaming")
    )
    relevance = {user_id: set() for user_id in users}
    for user_id, item_id in final.iter_rows():
        relevance[int(user_id)].add(int(item_id))
    if any(not items for items in relevance.values()):
        raise ValueError("an evaluation user has no final-window target")

    context_path = root / _LOGS_PATH / ".ranking-evidence/g4-native50m/context.pt"
    ranking_semantics = {}
    for role, value in selected.items():
        run_directory = root / _LOGS_PATH / value["trial"].run_name
        ranking_semantics[role] = verify_ranking_artifacts(
            context_path=context_path,
            ranking_path=run_directory / "ranking_evidence.pt",
            top_item_rankings_path=run_directory / "top_item_rankings.json",
            relevance_by_user=relevance,
        )
        selected_runs[role]["ranking_semantics"] = ranking_semantics[role]
    context_hashes = {
        value["context_payload_sha256"] for value in ranking_semantics.values()
    }
    context_counts = {
        (value["num_users"], value["num_relevant_items"])
        for value in ranking_semantics.values()
    }
    if len(context_hashes) != 1 or len(context_counts) != 1:
        raise ValueError("selected runs do not share one ranking context")
    context_users, context_targets = context_counts.pop()
    protocol["ranking_context"] = {
        "artifact": _file_fact(root, context_path),
        "payload_sha256": context_hashes.pop(),
        "num_users": context_users,
        "num_relevant_items": context_targets,
    }

    overall_rows = {}
    slice_results = {}
    quartiles = None
    for role, ranking in rankings.items():
        scored = _score_slice(ranking, relevance, catalog_size=identity["catalog_size"])
        overall_rows[role] = scored["metrics"]
        final_metrics = load_strict_json(
            root / _LOGS_PATH / selected[role]["trial"].run_name / "final_metrics.json"
        )
        _require_saved_metrics(scored, final_metrics, role)
        result = evaluate_slices(
            ranking_snapshot_path=root
            / _LOGS_PATH
            / selected[role]["trial"].run_name
            / "top_item_rankings.json",
            mapped_events_path=Path(protocol["mapped_events"]["path"]),
            cutoff_timestamp=protocol["split_cutoff_timestamp"],
            final_interval_seconds=_FINAL_INTERVAL_SECONDS,
        )
        if result["identity"] != identity:
            raise ValueError(f"{role} slice ranking identity changed")
        if quartiles is None:
            quartiles = result["activity_quartiles"]
        elif result["activity_quartiles"] != quartiles:
            raise ValueError("activity quartile membership differs by selected run")
        slice_results[role] = result["slices"]
    assert quartiles is not None

    slices = {}
    for name in slice_results["control_next_item"]:
        rows = {role: slice_results[role][name]["metrics"] for role in selected}
        counts = {
            (slice_results[role][name]["num_users"], slice_results[role][name]["num_targets"])
            for role in selected
        }
        if len(counts) != 1:
            raise ValueError(f"slice denominator differs by selected run: {name}")
        num_users, num_targets = counts.pop()
        slices[name] = {
            "num_users": num_users,
            "num_targets": num_targets,
            "rows": rows,
            "comparisons": _comparison_table(rows),
        }
    overall = {
        "num_users": len(users),
        "num_targets": sum(map(len, relevance.values())),
        "rows": overall_rows,
        "comparisons": _comparison_table(overall_rows),
    }
    slice_protocol = {
        "activity_assignment": "sort evaluation users by (training_like_count, uid), then assign floor(4 * zero_based_rank / num_users)",
        "activity_measure": "number of mapped like events with timestamp strictly before the split cutoff for each evaluation user",
        "activity_quartile_membership_sha256": canonical_sha256(quartiles),
        "activity_quartile_user_counts": {
            name: len(members) for name, members in quartiles.items()
        },
        "distance_assignment": "mapped final-window like event timestamp minus the split cutoff timestamp",
        "distance_interval_semantics": "(lower, upper]",
        "relevance_deduplication": "deduplicate compact item ids within each user and slice",
        "slice_user_inclusion": "distance and target-rank slices include users with at least one target in that slice; activity slices include every evaluation user assigned to that quartile",
        "target_distance_seconds": {
            "target_distance_0_6h": [0, 21600],
            "target_distance_6_24h": [21600, 86400],
            "target_distance_1_3d": [86400, 259200],
            "target_distance_3_7d": [259200, 604800],
        },
        "target_event_rank_assignment": "sort each user's mapped final-window like events by (timestamp, compact_item_id), enumerate all events from one, then deduplicate compact item ids within each rank slice",
        "target_event_rank_interval_semantics": "[lower, upper]",
        "target_event_rank": {
            "target_event_rank_1": [1, 1],
            "target_event_rank_2_5": [2, 5],
            "target_event_rank_6_10": [6, 10],
            "target_event_rank_11_plus": [11, None],
        },
    }
    return overall, slices, slice_protocol, selected_runs


def _calibration_evidence(
    root: Path, control_metrics: Mapping[str, float]
) -> dict[str, Any]:
    batches = [_queue_batch(root, batch_id) for batch_id in _CALIBRATION_BATCH_IDS]
    queue_jobs = {
        run_name: fact
        for _, jobs in batches
        for run_name, fact in jobs.items()
    }
    sources = []
    series = {metric: [] for metric in _METRIC_NAMES}
    configuration = None
    for seed in range(42, 52):
        run_name = _calibration_run_name(seed)
        run_directory = root / _LOGS_PATH / run_name
        metadata = load_strict_json(run_directory / "training_metadata.json")
        if metadata["seed"] != seed or metadata["selection_resolved"] is not True:
            raise ValueError(f"calibration seed metadata is invalid: {run_name}")
        current_configuration = {
            name: metadata[name] for name in _CALIBRATION_CONFIGURATION_FIELDS
        }
        if configuration is None:
            configuration = current_configuration
        elif current_configuration != configuration:
            raise ValueError("calibration runs do not share one frozen configuration")
        final_metrics = load_strict_json(run_directory / "final_metrics.json")
        for metric in _METRIC_NAMES:
            series[metric].append(float(final_metrics[metric]))
        queue_job = queue_jobs.get(run_name)
        if queue_job is None:
            raise ValueError(f"calibration run is absent from its queue batch: {run_name}")
        sources.append(
            {
                "seed": seed,
                "run_name": run_name,
                "outcome": {
                    "best_epoch": metadata["best_epoch"],
                    "epochs_trained": metadata["epochs_trained"],
                    "early_stopped": metadata["early_stopped"],
                    "selection_resolved": metadata["selection_resolved"],
                },
                "artifacts": {
                    "final_metrics": _file_fact(
                        root, run_directory / "final_metrics.json"
                    ),
                    "training_metadata": _file_fact(
                        root, run_directory / "training_metadata.json"
                    ),
                    "sweep_log": _file_fact(root, run_directory / "sweep.log"),
                },
                "queue_job": queue_job,
            }
        )
    assert configuration is not None
    relative = {
        metric: statistics.stdev(values) / statistics.mean(values)
        for metric, values in series.items()
    }
    return {
        "dataset_size": "native-50m",
        "seeds": list(range(42, 52)),
        "sample_standard_deviation_ddof": 1,
        "configuration": configuration,
        "configuration_sha256": canonical_sha256(configuration),
        "queue_batches": [batch for batch, _ in batches],
        "sources": sources,
        "relative_dispersion": relative,
        "operational_bands_from_current_control": {
            metric: relative[metric] * control_metrics[metric]
            for metric in _METRIC_NAMES
        },
    }


def _queue_batch(
    root: Path, batch_id: str
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    batch_path = root / _QUEUE_PATH / "batches" / f"{batch_id}.json"
    batch = load_strict_json(batch_path)
    if batch["id"] != batch_id or batch["sealed"] is not True:
        raise ValueError(f"queue batch is not sealed: {batch_id}")
    jobs = {}
    for job_id in batch["jobs"]:
        path = root / _QUEUE_PATH / "completed" / f"{job_id}.json"
        document = load_strict_json(path)
        if (
            document["id"] != job_id
            or document["batch_id"] != batch_id
            or document["exit_code"] != 0
        ):
            raise ValueError(f"queue job is not a successful member of {batch_id}")
        run_name = document["run"]
        if run_name in jobs:
            raise ValueError(f"queue batch duplicates run {run_name}")
        jobs[run_name] = {
            "batch_id": batch_id,
            "job_id": job_id,
            "artifact": _file_fact(root, path),
        }
    return {
        "batch_id": batch_id,
        "job_count": len(batch["jobs"]),
        "artifact": _file_fact(root, batch_path),
    }, jobs


def _calibration_run_name(seed: int) -> str:
    suffix = "batch_lr_calibration" if seed == 42 else "repeat"
    return (
        "g1_aggregate_dataset_size_baseline_none_l2_b512_"
        f"s{seed}_e0p003261002414691765_d0p025343654763668278_hnone_"
        f"{suffix}_ts2_r1_50m"
    )


def _comparison_table(rows: Mapping[str, Mapping[str, float]]) -> dict[str, Any]:
    result = {}
    for name, (reference_role, candidate_role) in _COMPARISONS.items():
        metrics = {}
        for metric in _METRIC_NAMES:
            reference = rows[reference_role][metric]
            candidate = rows[candidate_role][metric]
            delta = candidate - reference
            metrics[metric] = {
                "reference": reference,
                "candidate": candidate,
                "delta_points": delta,
                "delta_percent": None if reference == 0 else 100.0 * delta / reference,
            }
        result[name] = metrics
    return result


def _ranking_identity(snapshots: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    identities = []
    for snapshot in snapshots.values():
        identities.append(
            {
                "catalog_sha256": snapshot["catalog_sha256"],
                "catalog_size": snapshot["catalog_size"],
                "evaluation_users": len(snapshot["rankings"]),
                "exclude_seen": snapshot["exclude_seen"],
                "max_k": snapshot["max_k"],
            }
        )
    if any(identity != identities[0] for identity in identities[1:]):
        raise ValueError("selected ranking snapshots have different identities")
    return identities[0]


def _require_saved_metrics(
    scored: Mapping[str, Any], saved: Mapping[str, Any], role: str
) -> None:
    if int(saved["num_users"]) != scored["num_users"]:
        raise ValueError(f"{role} saved evaluation user count changed")
    for metric in _METRIC_NAMES:
        if not math.isclose(
            float(saved[metric]), float(scored["metrics"][metric]), rel_tol=0, abs_tol=1e-12
        ):
            raise ValueError(f"{role} saved {metric} differs from ranking replay")


def _ledger_identity(
    root: Path, path: Path, ledger: dict[str, Any]
) -> dict[str, Any]:
    return {
        "canonical_sha256": ledger["sha256"],
        "file": _file_fact(root, path),
    }


def _ledger_reference_paths(root: Path, role: str) -> dict[str, Path]:
    references = {
        "control_semantics_manifest_sha256": root
        / _PROTOCOL_PATH
        / "control_semantics_manifest.json"
    }
    if role != "control_next_item":
        references.update(
            {
                "selected_control_manifest_sha256": root
                / _PROTOCOL_PATH
                / "selected_control_manifest.json",
                "treatment_semantics_manifest_sha256": root
                / _PROTOCOL_PATH
                / "treatment_semantics_manifest.json",
            }
        )
    return references


def _require_ledger_semantics(
    ledger: dict[str, Any], protocol: dict[str, Any], role: str
) -> None:
    semantics = protocol["semantics"]
    if role == "control_next_item":
        if ledger["control_semantics_manifest_sha256"] != semantics[
            "control_semantics"
        ]["canonical_sha256"]:
            raise ValueError("control ledger semantics changed")
        return
    if ledger["selected_control_manifest_sha256"] != semantics["selected_control"][
        "canonical_sha256"
    ]:
        raise ValueError(f"{role} ledger selected-control identity changed")
    if ledger["treatment_semantics_manifest_sha256"] != semantics[
        "treatment_semantics"
    ]["canonical_sha256"]:
        raise ValueError(f"{role} ledger treatment semantics changed")


def _file_fact(root: Path, path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    try:
        display = resolved.relative_to(root).as_posix()
    except ValueError:
        display = resolved.as_posix()
    return {
        "path": display,
        "size": resolved.stat().st_size,
        "sha256": _file_sha256(resolved),
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _require_external_identity(path: Path, identity: Mapping[str, Any]) -> None:
    status = path.stat()
    if status.st_size != identity["size"] or _file_sha256(path) != identity["sha256"]:
        raise ValueError(f"frozen external artifact identity changed: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--output", type=Path, default=_EVIDENCE_PATH)
    arguments = parser.parse_args()
    if arguments.write == arguments.verify:
        raise SystemExit("pass exactly one of --write or --verify")
    root = Path(__file__).resolve().parents[3]
    if arguments.write:
        print(write_rq1_rq2_evidence(arguments.output, repo_root=root))
    else:
        verify_rq1_rq2_evidence(arguments.output, repo_root=root)
        print(hashlib.sha256(arguments.output.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
