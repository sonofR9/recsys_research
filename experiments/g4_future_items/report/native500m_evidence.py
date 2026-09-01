from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from experiments.g4_future_items.report.artifacts import read_recommender_trial
from experiments.g4_future_items.report.native500m_slices import (
    evaluate_native500m_rank_slices,
)


_ANCHOR = 0.032703745675187676
_EMBEDDING_LR = 0.0468526465053628
_HORIZON = 15
_ROLE_STAGES = {
    "control_next_item": "control_tuning",
    "rq1_24h": "rq1_tuning",
    "rq2_next10": "rq2_tuning",
}
_ROLE_SUFFIXES = {
    "control_next_item": "control",
    "rq1_24h": "rq1_24h",
    "rq2_next10": "rq2_next10",
}
_ROLE_OBJECTIVES = {
    "control_next_item": (
        {"id": "control_next_item"},
        {"valid_positive_mask_mode": "next_item_unique"},
    ),
    "rq1_24h": (
        {"id": "rq1_24h", "window_seconds": 86_400},
        {"valid_positive_mask_mode": "next_24h_unique"},
    ),
    "rq2_next10": (
        {"id": "rq2_next10", "event_lookahead": 10},
        {"valid_positive_mask_mode": "next_10_unique"},
    ),
}
_NON_COVERAGE_METRICS = tuple(
    f"{metric}@{cutoff}"
    for metric in ("recall", "capped_recall", "ndcg", "mrr")
    for cutoff in (10, 50, 100)
)
_ALL_METRICS = (
    *_NON_COVERAGE_METRICS,
    *(f"coverage@{cutoff}" for cutoff in (10, 50, 100)),
)


def canonical_bytes(document: dict[str, Any]) -> bytes:
    return json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def final_metrics_from_log(path: Path) -> dict[str, float]:
    marker = "Final metrics ("
    matches = [line for line in path.read_text().splitlines() if marker in line]
    if len(matches) != 1:
        raise ValueError(
            "native-500M sweep log must contain exactly one final metric row"
        )
    payload = matches[0].split(marker, 1)[1].rsplit(") -> ", 1)[0]
    value = ast.literal_eval(payload)
    if not isinstance(value, dict) or any(
        not isinstance(name, str)
        or isinstance(metric, bool)
        or not isinstance(metric, (int, float))
        for name, metric in value.items()
    ):
        raise ValueError("native-500M logged final metric schema differs")
    return {name: float(metric) for name, metric in value.items()}


def build_native500m_evidence(
    repo_root: Path,
    *,
    role_ledgers: dict[str, Sequence[Path]],
) -> dict[str, Any]:
    root = repo_root.resolve(strict=True)
    selected: dict[str, dict[str, Any]] = {}
    provenance: dict[str, Any] = {}
    common_source: dict[str, Any] | None = None
    common_data: dict[str, Any] | None = None
    for role in _ROLE_STAGES:
        ledgers = role_ledgers.get(role)
        if not ledgers:
            raise ValueError(f"native-500M role {role} has no ledgers")
        role_evidence, winner, source, data = _collect_role(root, role, ledgers)
        if common_source is None:
            _verify_snapshot(root, source)
            _verify_data_identity(data)
            common_source = source
            common_data = data
        elif source != common_source or data != common_data:
            raise ValueError("native-500M roles have different execution identities")
        provenance[role] = role_evidence
        selected[role] = winner

    assert common_source is not None and common_data is not None
    rows: dict[str, dict[str, Any]] = {}
    slices_by_role: dict[str, dict[str, Any]] = {}
    selected_runs: dict[str, dict[str, Any]] = {}
    context_path = root / "generated/logs/.ranking-evidence/g4-native500m/context.pt"
    mapped_events = Path(common_data["main"]["path"])
    cutoff = int(common_data["split_cutoff_timestamp"])
    for role, winner in selected.items():
        run_directory = Path(winner["run_directory"])
        metrics = _load_document(run_directory / "final_metrics.json")
        if set(metrics) != {*_ALL_METRICS, "num_users"}:
            raise ValueError(f"native-500M final metric schema differs for {role}")
        logged_metrics = final_metrics_from_log(run_directory / "sweep.log")
        if set(logged_metrics) != {*_ALL_METRICS, "num_users"} or any(
            float(metrics[name]) != logged_metrics[name] for name in logged_metrics
        ):
            raise ValueError(
                f"native-500M final metrics differ from authenticated log for {role}"
            )
        artifact_facts = {
            name: _file_fact(run_directory / name)
            for name in (
                "g4_job.json",
                "training_metadata.json",
                "sweep.log",
                "final_metrics.json",
                "ranking_evidence.pt",
            )
        }
        if any(
            artifact_facts[name]["mtime_ns"] > artifact_facts["sweep.log"]["mtime_ns"]
            for name in ("final_metrics.json", "ranking_evidence.pt")
        ):
            raise ValueError(
                f"native-500M final artifact postdates authenticated log for {role}"
            )
        slice_evidence = evaluate_native500m_rank_slices(
            context_path=context_path,
            ranking_path=run_directory / "ranking_evidence.pt",
            mapped_events_path=mapped_events,
            cutoff_timestamp=cutoff,
        )
        for metric in _NON_COVERAGE_METRICS:
            if (
                abs(
                    float(metrics[metric])
                    - slice_evidence["overall"]["metrics"][metric]
                )
                > 1e-12
            ):
                raise ValueError(
                    f"native-500M ranking metric differs for {role}: {metric}"
                )
        if metrics["num_users"] != slice_evidence["overall"]["num_users"]:
            raise ValueError(f"native-500M final user count differs for {role}")
        rows[role] = logged_metrics
        slices_by_role[role] = slice_evidence
        selected_runs[role] = {
            key: winner[key]
            for key in (
                "row_id",
                "run_name",
                "deep_learning_rate",
                "validation_recall_at_100",
                "validation_loss",
                "declared_horizon_epochs",
                "restored_best_epoch",
            )
        } | {
            "final_metrics_source": "boundary-ledger-authenticated sweep log",
            "artifacts": artifact_facts,
        }

    calibration = _calibration(root, rows["control_next_item"])
    band = calibration["operational_bands_from_current_control"]["recall@100"]
    control_recall = float(rows["control_next_item"]["recall@100"])
    rq1_recall = float(rows["rq1_24h"]["recall@100"])
    rq2_recall = float(rows["rq2_next10"]["recall@100"])
    qualification = {
        "rq1_24h": {
            "recall_at_100_delta_points": rq1_recall - control_recall,
            "recall_at_100_delta_percent": 100 * (rq1_recall / control_recall - 1),
            "supported": rq1_recall - control_recall > band,
            "qualifies_for_aggregate": rq1_recall - control_recall > band,
        },
        "rq2_next10": {
            "recall_at_100_delta_points": rq2_recall - control_recall,
            "recall_at_100_delta_percent": 100 * (rq2_recall / control_recall - 1),
            "non_inferior": rq2_recall >= control_recall - band,
            "supported": rq2_recall - control_recall > band,
            "qualifies_for_aggregate": rq2_recall - control_recall > band,
        },
    }
    aggregate_role = _aggregate_role(rows, qualification, band)
    return {
        "schema_version": 1,
        "kind": "g4_rq1_rq2_evaluation_native500m",
        "dataset_size": "native-500m",
        "protocol": {
            "source_closure_sha256": common_source["sha256"],
            "data_identity": common_data,
            "ranking_context": _file_fact(context_path),
        },
        "selection_provenance": provenance,
        "selected_runs": selected_runs,
        "calibration": calibration,
        "overall": {
            "rows": rows,
            "qualification": qualification,
            "aggregate_role": aggregate_role,
        },
        "slice_protocol": {
            "final_interval_seconds": 7 * 24 * 60 * 60,
            "validation_boundary": "timestamp >= split_cutoff_timestamp",
            "activity_measure": "mapped training likes per evaluation user",
        },
        "slices": {
            name: {
                "num_users": {
                    role: slices_by_role[role]["slices"][name]["num_users"]
                    for role in _ROLE_STAGES
                },
                "num_targets": {
                    role: slices_by_role[role]["slices"][name]["num_targets"]
                    for role in _ROLE_STAGES
                },
                "rows": {
                    role: slices_by_role[role]["slices"][name]["metrics"]
                    for role in _ROLE_STAGES
                },
            }
            for name in slices_by_role["control_next_item"]["slices"]
        },
    }


def write_native500m_evidence(
    artifact_path: Path,
    *,
    repo_root: Path,
    role_ledgers: dict[str, Sequence[Path]],
) -> str:
    payload = canonical_bytes(
        build_native500m_evidence(repo_root, role_ledgers=role_ledgers)
    )
    digest = hashlib.sha256(payload).hexdigest()
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    if artifact_path.exists():
        if artifact_path.read_bytes() != payload:
            raise RuntimeError(f"native-500M evidence changed: {artifact_path}")
    else:
        artifact_path.write_bytes(payload)
    sidecar = artifact_path.with_suffix(".sha256")
    if sidecar.exists():
        if sidecar.read_text() != digest:
            raise RuntimeError(f"native-500M evidence digest changed: {sidecar}")
    else:
        sidecar.write_text(digest)
    return digest


def _collect_role(
    root: Path, role: str, ledger_paths: Sequence[Path]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    stage = _ROLE_STAGES[role]
    candidates: list[dict[str, Any]] = []
    authenticated_rows: set[str] = set()
    ledger_facts = []
    source: dict[str, Any] | None = None
    data: dict[str, Any] | None = None
    for ledger_index, supplied_path in enumerate(ledger_paths):
        path = supplied_path if supplied_path.is_absolute() else root / supplied_path
        ledger = _load_ledger(path)
        _validate_ledger_shape(root, ledger, role, ledger_index)
        if ledger_index:
            winner = _winner(candidates)
            if ledger["entering_row"] != winner["row"]:
                raise ValueError(f"native-500M {role} boundary enters from another row")
            if ledger["entering_row_sha256"] != _canonical_sha256(winner["row"]):
                raise ValueError(f"native-500M {role} boundary entering hash differs")
            expected_edge = {
                ("lower", 1): _ANCHOR / 2,
                ("upper", 1): _ANCHOR * 2,
                ("lower", 2): _ANCHOR / 8,
                ("upper", 2): _ANCHOR * 8,
            }[(ledger["direction"], ledger["round_number"])]
            if winner["deep_learning_rate"] != expected_edge:
                raise ValueError(f"native-500M {role} boundary trigger differs")
            _verify_predecessor_evidence(
                ledger["predecessor_evidence"],
                expected_ledgers=ledger_facts,
                candidates=candidates,
            )
            authenticated_rows.update(candidate["row_id"] for candidate in candidates)
        ledger_facts.append(_file_fact(path))
        for row in ledger["rows"]:
            run_directory = root / "generated/logs" / row["job"]["run_name"]
            contract = _load_document(run_directory / "g4_job.json")
            if (
                contract.get("ledger_sha256") != ledger["sha256"]
                or contract.get("row_id") != row["id"]
                or contract.get("job") != row["job"]
                or not _contract_ledger_matches(root, contract, path)
            ):
                raise ValueError(f"native-500M {role} run contract differs from ledger")
            if source is None:
                source = contract["source_closure"]
                data = contract["data_identity"]
            elif (
                contract["source_closure"] != source
                or contract["data_identity"] != data
            ):
                raise ValueError(f"native-500M {role} candidate identities differ")
            trial = read_recommender_trial(run_directory)
            if not trial.usable or trial.row_id != row["id"]:
                raise ValueError(f"native-500M {role} candidate is not usable")
            metadata = _load_document(run_directory / "training_metadata.json")
            candidates.append(
                {
                    "row": row,
                    "row_id": row["id"],
                    "run_name": trial.run_name,
                    "run_directory": str(run_directory.resolve()),
                    "deep_learning_rate": trial.parameters["deep_learning_rate"],
                    "validation_recall_at_100": trial.validation_recall_at_100,
                    "validation_loss": trial.validation_loss,
                    "declared_horizon_epochs": trial.horizon_epochs,
                    "restored_best_epoch": metadata["best_epoch"],
                }
            )
    assert source is not None and data is not None
    winner = _winner(candidates)
    if winner["row_id"] not in authenticated_rows:
        raise ValueError(
            f"native-500M {role} winner lacks authenticated successor evidence"
        )
    return (
        {
            "ledgers": ledger_facts,
            "candidates": [
                {
                    key: value
                    for key, value in candidate.items()
                    if key not in {"row", "run_directory"}
                }
                for candidate in candidates
            ],
            "winner_row_id": winner["row_id"],
        },
        winner,
        source,
        data,
    )


def _load_ledger(path: Path) -> dict[str, Any]:
    document = _load_document(path)
    supplied = document.get("sha256")
    unsigned = {key: value for key, value in document.items() if key != "sha256"}
    if not isinstance(supplied, str) or supplied != _canonical_sha256(unsigned):
        raise ValueError(f"native-500M ledger seal differs: {path}")
    return document


def _contract_ledger_matches(
    root: Path, contract: dict[str, Any], ledger_path: Path
) -> bool:
    bound_path = Path(contract.get("ledger_path", "")).resolve()
    current_path = ledger_path.resolve()
    if bound_path == current_path:
        return True
    closure = contract.get("source_closure")
    if not isinstance(closure, dict) or not isinstance(closure.get("sha256"), str):
        return False
    try:
        relative_path = current_path.relative_to(root)
    except ValueError:
        return False
    expected = (
        root
        / "generated/g4_native500m_source_snapshots"
        / closure["sha256"]
        / relative_path
    ).resolve()
    return (
        bound_path == expected
        and bound_path.is_file()
        and bound_path.read_bytes() == current_path.read_bytes()
    )


def _validate_ledger_shape(
    root: Path, ledger: dict[str, Any], role: str, index: int
) -> None:
    stage = _ROLE_STAGES[role]
    if (
        ledger.get("version") != 1
        or ledger.get("lineage") != "native500m-v1"
        or ledger.get("dataset_size") != "500m"
        or ledger.get("control_manifest_sha256")
        != _canonical_sha256(
            _load_document(
                root
                / "experiments/g4_future_items/protocol/native500m/control_manifest.json"
            )
        )
    ):
        raise ValueError(f"native-500M {role} ledger protocol differs")
    expected_rates: tuple[float, ...]
    if index == 0:
        if ledger.get("stage") != stage:
            raise ValueError(f"native-500M {role} base stage differs")
        expected_rates = (_ANCHOR / 2, _ANCHOR, _ANCHOR * 2)
        if role == "control_next_item":
            incident = ledger.get("retry_incident")
            incident_path = (
                root / incident.get("path", "") if isinstance(incident, dict) else root
            )
            if (
                ledger.get("retry_revision") != 2
                or not isinstance(incident, dict)
                or incident.get("batch_id") != "a02f5ffba08d4fb9bcb9260ac149bb40"
                or incident.get("sha256")
                != _canonical_sha256(_load_document(incident_path))
            ):
                raise ValueError("native-500M control retry authorization differs")
        elif "retry_revision" in ledger or "retry_incident" in ledger:
            raise ValueError(f"native-500M {role} has an unexpected retry")
    else:
        direction = ledger.get("direction")
        round_number = ledger.get("round_number")
        factors = {
            ("lower", 1): (1 / 8, 1 / 4),
            ("upper", 1): (4, 8),
            ("lower", 2): (1 / 32, 1 / 16),
            ("upper", 2): (16, 32),
        }.get((direction, round_number))
        if (
            ledger.get("stage") != f"{stage}_boundary"
            or ledger.get("base_stage") != stage
            or round_number != index
            or factors is None
        ):
            raise ValueError(f"native-500M {role} boundary sequence differs")
        expected_rates = tuple(_ANCHOR * factor for factor in factors)
    rows = ledger.get("rows")
    if not isinstance(rows, list) or len(rows) != len(expected_rates):
        raise ValueError(f"native-500M {role} ledger rows differ")
    objective, loss = _ROLE_OBJECTIVES[role]
    for trial_id, (row, deep_lr) in enumerate(zip(rows, expected_rates), 1):
        job = row.get("job")
        if index == 0:
            expected_protocol: dict[str, Any] = {
                "stage": stage,
                "trial_id": trial_id,
            }
            retry = ""
            expected_row_id = f"{stage}:{trial_id:02d}"
            if role == "control_next_item":
                expected_protocol["retry_revision"] = 2
                retry = "_retry2"
                expected_row_id = f"{stage}:retry2:{trial_id:02d}"
            boundary = ""
        else:
            direction = ledger["direction"]
            round_number = ledger["round_number"]
            expected_protocol = {
                "stage": stage,
                "trial_id": trial_id,
                "boundary_direction": direction,
                "boundary_round": round_number,
            }
            retry = ""
            boundary = f"_boundary_{direction}_r{round_number}"
            expected_row_id = (
                f"{stage}_boundary:{direction}:r{round_number}:{trial_id:02d}"
            )
        expected_run_name = (
            f"g4_{_ROLE_SUFFIXES[role]}{boundary}_trial_{trial_id:02d}"
            f"{retry}_native500m"
        )
        if not isinstance(job, dict) or (
            row.get("id") != expected_row_id
            or job.get("protocol") != expected_protocol
            or job.get("run_name") != expected_run_name
            or job.get("dataloader") != {"batch_size": 512}
            or job.get("embedding_learning_rate") != _EMBEDDING_LR
            or job.get("deep_learning_rate") != deep_lr
            or job.get("lr_schedule_horizon_epochs") != _HORIZON
            or job.get("seed") != 42
            or job.get("objective") != objective
            or job.get("loss") != loss
        ):
            raise ValueError(f"native-500M {role} job differs")


def _winner(candidates: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        raise ValueError("native-500M selection has no candidates")
    return min(
        enumerate(candidates),
        key=lambda indexed: (
            -indexed[1]["validation_recall_at_100"],
            indexed[1]["validation_loss"],
            indexed[0],
        ),
    )[1]


def _verify_predecessor_evidence(
    evidence: Any,
    *,
    expected_ledgers: Sequence[dict[str, Any]],
    candidates: Sequence[dict[str, Any]],
) -> None:
    if not isinstance(evidence, dict) or set(evidence) != {"ledgers", "runs"}:
        raise ValueError("native-500M boundary predecessor schema differs")
    expected_runs = [
        {
            "job_contract": _file_fact(
                Path(candidate["run_directory"]) / "g4_job.json"
            ),
            "training_metadata": _file_fact(
                Path(candidate["run_directory"]) / "training_metadata.json"
            ),
            "sweep_log": _file_fact(Path(candidate["run_directory"]) / "sweep.log"),
        }
        for candidate in candidates
    ]
    if (
        evidence["ledgers"] != list(expected_ledgers)
        or evidence["runs"] != expected_runs
    ):
        raise ValueError("native-500M boundary predecessor evidence differs")


def _verify_snapshot(root: Path, closure: dict[str, Any]) -> None:
    unsigned = {key: value for key, value in closure.items() if key != "sha256"}
    if closure.get("sha256") != _canonical_sha256(unsigned):
        raise ValueError("native-500M source closure seal differs")
    snapshot = root / "generated/g4_native500m_source_snapshots" / closure["sha256"]
    if closure.get("paths") != sorted(closure.get("sources", {})):
        raise ValueError("native-500M source closure path set differs")
    for relative_path, expected_sha256 in closure["sources"].items():
        path = snapshot / relative_path
        if _sha256(path) != expected_sha256:
            raise ValueError(f"native-500M snapshot source differs: {relative_path}")


def _verify_data_identity(identity: dict[str, Any]) -> None:
    if identity.get("dataset_size") != "500m":
        raise ValueError("native-500M data size differs")
    for name in ("main", "remap"):
        _verify_file_fact(identity[name])


def _calibration(root: Path, control: dict[str, Any]) -> dict[str, Any]:
    path = (
        root
        / "experiments/g1_sasrec_item_ids_likes/scratchpad/baseline_spread_500m.json"
    )
    source = _load_document(path)
    if source.get("n") != 10 or set(source.get("metrics", {})) != set(_ALL_METRICS):
        raise ValueError("shared native-500M dispersion schema differs")
    relative = {
        metric: float(source["metrics"][metric]["stddev_percent_of_mean"]) / 100
        for metric in _ALL_METRICS
    }
    return {
        "source": _file_fact(path),
        "relative_dispersion": relative,
        "operational_bands_from_current_control": {
            metric: float(control[metric]) * relative[metric] for metric in _ALL_METRICS
        },
    }


def _aggregate_role(
    rows: dict[str, dict[str, Any]],
    qualification: dict[str, dict[str, Any]],
    band: float,
) -> str:
    eligible = [
        role
        for role in ("rq2_next10", "rq1_24h")
        if qualification[role]["qualifies_for_aggregate"]
    ]
    if not eligible:
        return "control_next_item"
    best = max(float(rows[role]["recall@100"]) for role in eligible)
    tied = [role for role in eligible if best - float(rows[role]["recall@100"]) <= band]
    return tied[0]


def _load_document(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected an object: {path}")
    return value


def _canonical_sha256(document: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_bytes(document)).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_fact(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": _sha256(resolved),
    }


def _verify_file_fact(identity: Any) -> None:
    if not isinstance(identity, dict) or set(identity) != {
        "path",
        "size",
        "mtime_ns",
        "sha256",
    }:
        raise ValueError("native-500M file identity schema differs")
    if _file_fact(Path(identity["path"])) != identity:
        raise ValueError(f"native-500M file identity differs: {identity['path']}")
