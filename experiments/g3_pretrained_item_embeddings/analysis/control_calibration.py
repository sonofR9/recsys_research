from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from dcn.eval.ranking_evidence import load_ranking_evidence
from experiments.g3_pretrained_item_embeddings.analysis.queue_attribution import (
    verify_artifacts_in_job_window,
    verify_unique_completed_run,
)
from experiments.g3_pretrained_item_embeddings.launchers.control import (
    JOB_ENVIRONMENT,
    LEDGER_ENVIRONMENT,
    decode_control_job,
    verify_ledger_inputs,
)
from experiments.g3_pretrained_item_embeddings.protocol.constants import (
    APPROVED_PROTOCOL,
    APPROVED_PROTOCOL_SHA256,
)
from experiments.g3_pretrained_item_embeddings.protocol.control_ledger import (
    ControlLedger,
    load_control_ledger,
)


_KIND = "g3_untied_control_calibration"
_BODY_KEYS = {
    "control_ledger",
    "queue_batch",
    "ranking_context",
    "tuning_ledger",
    "horizon_winners",
    "power_law_fits",
    "held_out_check",
    "transfer_decision",
    "finding",
}
_METRIC_NAMES = tuple(
    f"{name}@{cutoff}"
    for name in ("recall", "ndcg", "mrr", "capped_recall", "coverage")
    for cutoff in (10, 50, 100)
)


def fit_power_relation(
    points: Sequence[tuple[int, float]],
) -> dict[str, object]:
    if len(points) != 3 or {horizon for horizon, _ in points} != {15, 25, 40}:
        raise ValueError("power relation requires exactly horizons 15, 25, and 40")
    if any(horizon <= 0 or not math.isfinite(rate) or rate <= 0 for horizon, rate in points):
        raise ValueError("power relation inputs must be finite and positive")
    ordered = sorted(points)
    xs = [math.log(horizon) for horizon, _ in ordered]
    ys = [math.log(rate) for _, rate in ordered]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    denominator = sum((value - mean_x) ** 2 for value in xs)
    exponent = sum(
        (x_value - mean_x) * (y_value - mean_y)
        for x_value, y_value in zip(xs, ys, strict=True)
    ) / denominator
    log_coefficient = mean_y - exponent * mean_x
    coefficient = math.exp(log_coefficient)
    fitted_logs = [log_coefficient + exponent * value for value in xs]
    residual = sum(
        (actual - fitted) ** 2
        for actual, fitted in zip(ys, fitted_logs, strict=True)
    )
    total = sum((value - mean_y) ** 2 for value in ys)
    r_squared = 1.0 if total == 0.0 and residual == 0.0 else 1.0 - residual / total
    return {
        "method": "ordinary_least_squares_log_rate_on_log_horizon",
        "formula": "learning_rate = coefficient * horizon_epochs ** exponent",
        "coefficient": coefficient,
        "exponent": exponent,
        "r_squared_log_space": r_squared,
        "source_coordinates": [
            {"horizon_epochs": horizon, "learning_rate": rate}
            for horizon, rate in ordered
        ],
        "fitted_coordinates": {
            str(horizon): coefficient * horizon**exponent for horizon, _ in ordered
        },
    }


def assess_transfer(
    *,
    search_recall_at_100: float,
    held_out_recall_at_100: float,
    relative_dispersion: float,
) -> dict[str, object]:
    values = (search_recall_at_100, held_out_recall_at_100, relative_dispersion)
    if not all(math.isfinite(value) for value in values) or relative_dispersion < 0:
        raise ValueError("transfer evidence must be finite with nonnegative dispersion")
    difference = abs(held_out_recall_at_100 - search_recall_at_100)
    band = abs(search_recall_at_100) * relative_dispersion
    return {
        "accepted": difference <= band,
        "comparison_horizon_epochs": 25,
        "search_recall_at_100": search_recall_at_100,
        "held_out_recall_at_100": held_out_recall_at_100,
        "absolute_difference": difference,
        "relative_dispersion": relative_dispersion,
        "operational_band": band,
        "comparison": "absolute_difference <= search_recall_at_100 * relative_dispersion",
        "interpretation": "conservative performance-region interpretation approved by the lead after protocol ambiguity was identified",
        "validates_lr_distance": False,
    }


def calibration_document(
    body: Mapping[str, object],
    *,
    replace: bool = False,
) -> dict[str, object]:
    payload = dict(body)
    if replace:
        payload = {key: payload[key] for key in _BODY_KEYS}
    if set(payload) != _BODY_KEYS:
        raise ValueError("calibration body keys do not match the closed schema")
    document = {
        "schema_version": 1,
        "kind": _KIND,
        "protocol_sha256": APPROVED_PROTOCOL_SHA256,
        **payload,
    }
    document["sha256"] = _canonical_sha256(document)
    return document


def load_control_calibration(path: Path) -> dict[str, object]:
    document = _load_json(path)
    expected = {
        "schema_version",
        "kind",
        "protocol_sha256",
        *_BODY_KEYS,
        "sha256",
    }
    if set(document) != expected:
        raise ValueError("calibration evidence keys do not match the closed schema")
    if (
        document["schema_version"] != 1
        or document["kind"] != _KIND
        or document["protocol_sha256"] != APPROVED_PROTOCOL_SHA256
    ):
        raise ValueError("calibration evidence identity is not approved")
    expected_hash = _canonical_sha256(
        {key: value for key, value in document.items() if key != "sha256"}
    )
    if document["sha256"] != expected_hash:
        raise ValueError("calibration evidence hash changed")
    decision = document["transfer_decision"]
    if not isinstance(decision, dict) or decision.get("accepted") is not True:
        raise ValueError("calibration evidence does not approve horizon transfer")
    return document


def persist_control_calibration(path: Path, document: Mapping[str, object]) -> Path:
    content = (_canonical_json(document) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(content)
    except FileExistsError:
        if path.read_bytes() != content:
            raise RuntimeError(f"immutable calibration evidence differs: {path}")
    load_control_calibration(path)
    return path


def build_control_calibration(
    *,
    root: Path,
    ledger_path: Path,
    batch_id: str,
) -> dict[str, object]:
    root = root.resolve()
    ledger_path = ledger_path.resolve()
    ledger = load_control_ledger(ledger_path)
    verify_ledger_inputs(root, ledger, full_validation=False)
    batch_path = root / "generated/training-queue-service/batches" / f"{batch_id}.json"
    batch = _load_json(batch_path)
    job_ids = batch.get("jobs")
    if (
        batch.get("id") != batch_id
        or batch.get("sealed") is not True
        or not isinstance(job_ids, list)
        or len(job_ids) != len(ledger.rows)
        or len(set(job_ids)) != len(job_ids)
    ):
        raise ValueError("queue batch does not contain exactly the sealed control ledger")
    context_path = root / "generated/logs/.ranking-evidence/g3-native50m/context.pt"
    runs = []
    for row, job_id in zip(ledger.rows, job_ids, strict=True):
        runs.append(
            _collect_run(
                root=root,
                ledger=ledger,
                ledger_path=ledger_path,
                context_path=context_path,
                batch_id=batch_id,
                job_id=job_id,
                row=row.to_dict(),
            )
        )
    search = [run for run in runs if run["role"] == "search"]
    held_out = [run for run in runs if run["role"] == "transfer_check"]
    if len(search) != 9 or len(held_out) != 1:
        raise ValueError("control batch does not contain nine search cells and one check")
    winners = [_winner_at_horizon(search, horizon) for horizon in (15, 25, 40)]
    embedding_fit = fit_power_relation(
        tuple(
            (winner["horizon_epochs"], winner["embedding_learning_rate"])
            for winner in winners
        )
    )
    deep_fit = fit_power_relation(
        tuple(
            (winner["horizon_epochs"], winner["deep_learning_rate"])
            for winner in winners
        )
    )
    horizon_25 = next(winner for winner in winners if winner["horizon_epochs"] == 25)
    relative_dispersion = APPROVED_PROTOCOL.relative_dispersion(
        "native-50m", "recall@100"
    )
    held_out_check = assess_transfer(
        search_recall_at_100=horizon_25["metrics"]["recall@100"],
        held_out_recall_at_100=held_out[0]["metrics"]["recall@100"],
        relative_dispersion=relative_dispersion,
    )
    held_out_check |= {
        "search_row_id": horizon_25["row_id"],
        "held_out_row_id": held_out[0]["row_id"],
        "held_out_coordinate": {
            name: held_out[0][name]
            for name in (
                "embedding_learning_rate",
                "deep_learning_rate",
                "horizon_epochs",
            )
        },
        "fitted_horizon_25_coordinate": {
            "embedding_learning_rate": embedding_fit["fitted_coordinates"]["25"],
            "deep_learning_rate": deep_fit["fitted_coordinates"]["25"],
        },
    }
    accepted = held_out_check["accepted"]
    finding = (
        "Horizon transfer accepted under the lead-approved conservative performance-region check; the held-out check is resolution-equivalent to the best horizon-25 search control. This does not validate LR-space distance."
        if accepted
        else "Horizon transfer rejected because the held-out horizon-25 control is outside the approved performance-resolution band."
    )
    return calibration_document(
        {
            "control_ledger": _file_fact(root, ledger_path) | {"logical_sha256": ledger.sha256},
            "queue_batch": _file_fact(root, batch_path) | {"batch_id": batch_id},
            "ranking_context": _file_fact(root, context_path),
            "tuning_ledger": runs,
            "horizon_winners": winners,
            "power_law_fits": {
                "embedding_learning_rate": embedding_fit,
                "deep_learning_rate": deep_fit,
            },
            "held_out_check": held_out_check,
            "transfer_decision": {
                "accepted": accepted,
                "rule": "lead_approved_conservative_performance_region_v1",
                "dependent_search_design": "transferred_capacity_first_then_selected_capacity_horizon_followup" if accepted else "balanced_direct_capacity_search",
            },
            "finding": finding,
        }
    )


def _collect_run(
    *,
    root: Path,
    ledger: ControlLedger,
    ledger_path: Path,
    context_path: Path,
    batch_id: str,
    job_id: str,
    row: dict[str, object],
) -> dict[str, object]:
    completed_path = root / "generated/training-queue-service/completed" / f"{job_id}.json"
    queue_job = _load_json(completed_path)
    if (
        queue_job.get("id") != job_id
        or queue_job.get("batch_id") != batch_id
        or queue_job.get("run") != row["run_name"]
        or queue_job.get("exit_code") != 0
        or queue_job.get("data_group") != "g3-native50m-likes"
        or Path(str(queue_job.get("script"))).name != "run_control.py"
    ):
        raise ValueError(f"queue completion differs for {row['id']}")
    environment = queue_job.get("environment")
    if not isinstance(environment, list):
        raise ValueError(f"queue environment is absent for {row['id']}")
    values = dict(value.split("=", 1) for value in environment if "=" in value)
    if values.get("WANDB_MODE") != "offline" or Path(values.get(LEDGER_ENVIRONMENT, "")).resolve() != ledger_path:
        raise ValueError(f"queue environment differs for {row['id']}")
    compiled = decode_control_job(values.get(JOB_ENVIRONMENT, ""), ledger)
    if compiled.row_id != row["id"] or compiled.job != row:
        raise ValueError(f"compiled queue payload differs for {row['id']}")

    run_directory = root / "generated/logs" / str(row["run_name"])
    contract_path = run_directory / "g3_control_job.json"
    contract = _load_json(contract_path)
    expected_contract = compiled.to_dict() | {
        "ledger_path": str(ledger_path),
        "ledger_sha256": ledger.sha256,
    }
    if contract != expected_contract:
        raise ValueError(f"job contract differs for {row['id']}")
    metadata_path = run_directory / "training_metadata.json"
    metadata = _load_json(metadata_path)
    training = row["training"]
    if not isinstance(training, dict):
        raise ValueError(f"training coordinate is invalid for {row['id']}")
    horizon = training["horizon_epochs"]
    expected_metadata = {
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
    }
    if any(metadata.get(key) != value for key, value in expected_metadata.items()):
        raise ValueError(f"runtime metadata differs from contract for {row['id']}")
    best_epoch = metadata.get("best_epoch")
    if type(best_epoch) is not int or not 1 <= best_epoch <= horizon:
        raise ValueError(f"best epoch is invalid for {row['id']}")
    if metadata.get("best_epoch_at_cap") is not (best_epoch == horizon):
        raise ValueError(f"best-epoch cap status is invalid for {row['id']}")
    traces = metadata.get("lr_group_traces")
    if not isinstance(traces, dict) or any(
        not isinstance(traces.get(group), list)
        or len(traces[group]) != horizon
        or traces[group][-1] != 0.0
        for group in ("embedding", "deep")
    ):
        raise ValueError(f"schedule trace is incomplete for {row['id']}")

    metrics_path = run_directory / "final_metrics.json"
    saved_metrics = _load_json(metrics_path)
    if set(saved_metrics) != {*_METRIC_NAMES, "num_users"} or any(
        not isinstance(value, (int, float)) or not math.isfinite(float(value))
        for value in saved_metrics.values()
    ):
        raise ValueError(f"final metrics schema is invalid for {row['id']}")
    ranking_path = run_directory / "ranking_evidence.pt"
    top_rankings_path = run_directory / "top_item_rankings.json"
    recomputed = _recompute_metrics(context_path, ranking_path, top_rankings_path)
    if any(abs(float(saved_metrics[key]) - recomputed[key]) > 1e-15 for key in saved_metrics):
        raise ValueError(f"final metrics differ from ranking evidence for {row['id']}")
    artifact_filenames = (
        ("job_contract", "g3_control_job.json"),
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
        "role": row["role"],
        "embedding_learning_rate": training["embedding_learning_rate"],
        "deep_learning_rate": training["deep_learning_rate"],
        "horizon_epochs": horizon,
        "best_epoch": best_epoch,
        "epochs_trained": metadata["epochs_trained"],
        "selection_resolved": True,
        "queue_wall_seconds": queue_job["finished_at"] - queue_job["dispatched_at"],
        "metrics": saved_metrics,
        "metric_provenance": {
            "recomputed_from_ranking_evidence": True,
            "absolute_tolerance": 1e-15,
            "num_users": recomputed["num_users"],
        },
        "queue_job": _file_fact(root, completed_path) | {"job_id": job_id},
        "artifacts": artifacts,
    }


def _winner_at_horizon(
    runs: Sequence[dict[str, object]], horizon: int
) -> dict[str, object]:
    candidates = [run for run in runs if run["horizon_epochs"] == horizon]
    if len(candidates) != 3:
        raise ValueError(f"horizon {horizon} does not have three search cells")
    ordered = sorted(
        candidates,
        key=lambda run: (
            -run["metrics"]["recall@100"],
            -run["metrics"]["ndcg@100"],
            run["queue_wall_seconds"],
            run["row_id"],
        ),
    )
    return ordered[0]


def _recompute_metrics(
    context_path: Path,
    ranking_path: Path,
    top_rankings_path: Path,
) -> dict[str, float]:
    evidence = load_ranking_evidence(context_path, ranking_path)
    snapshot = _load_json(top_rankings_path)
    rows = snapshot.get("rankings")
    catalog_size = snapshot.get("catalog_size")
    if not isinstance(rows, list) or type(catalog_size) is not int or catalog_size < 1:
        raise ValueError("top-item ranking snapshot is invalid")
    by_user = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"user_id", "item_ids"}:
            raise ValueError("top-item ranking row is invalid")
        by_user[row["user_id"]] = row["item_ids"]
    users = [int(value) for value in evidence.user_ids.tolist()]
    if set(users) != set(by_user) or len(by_user) != len(users):
        raise ValueError("ranking users differ from ranking evidence")
    offsets = [int(value) for value in evidence.relevance_offsets.tolist()]
    ranks = [int(value) for value in evidence.relevant_ranks.tolist()]
    result = {}
    for cutoff in (10, 50, 100):
        totals = {name: 0.0 for name in ("recall", "capped_recall", "ndcg", "mrr")}
        covered = set()
        for position, user_id in enumerate(users):
            relevant = ranks[offsets[position] : offsets[position + 1]]
            hits = [rank for rank in relevant if 0 < rank <= cutoff]
            ideal_length = min(cutoff, len(relevant))
            totals["recall"] += len(hits) / len(relevant)
            totals["capped_recall"] += len(hits) / ideal_length
            totals["ndcg"] += sum(1 / math.log2(rank + 1) for rank in hits) / sum(
                1 / math.log2(rank + 1) for rank in range(1, ideal_length + 1)
            )
            totals["mrr"] += 1 / min(hits) if hits else 0.0
            ranking = by_user[user_id]
            if not isinstance(ranking, list) or len(ranking) < cutoff:
                raise ValueError("top-item ranking is shorter than its cutoff")
            covered.update(ranking[:cutoff])
        for name, total in totals.items():
            result[f"{name}@{cutoff}"] = total / len(users)
        result[f"coverage@{cutoff}"] = len(covered) / catalog_size
    result["num_users"] = float(len(users))
    return result


def _file_fact(root: Path, path: Path) -> dict[str, object]:
    resolved = path.resolve()
    if resolved.is_symlink() or not resolved.is_file() or not resolved.is_relative_to(root):
        raise ValueError(f"evidence artifact is not a regular project file: {path}")
    return {
        "path": resolved.relative_to(root).as_posix(),
        "size_bytes": resolved.stat().st_size,
        "sha256": _file_sha256(resolved),
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load JSON object {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON number {value!r}")
