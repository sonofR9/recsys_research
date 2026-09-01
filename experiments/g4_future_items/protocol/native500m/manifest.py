from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal

import torch

from dcn.config import MuTransferGenerationExperiment

from experiments.g4_future_items.configs.native500m import (
    BATCH_SIZE,
    DEEP_LEARNING_RATE_ANCHOR,
    EMBEDDING_LEARNING_RATE,
    TRAINING_HORIZON_EPOCHS,
    build_native500m_control,
    build_native500m_treatment,
)
from experiments.g4_future_items.protocol.manifest import (
    _validate_materialization_cost_evidence,
    _verify_materialization_artifacts,
    derive_current_entrypoint_source_paths,
    source_manifest,
)


Stage = Literal[
    "control_tuning",
    "rq1_tuning",
    "rq2_tuning",
    "rq3_deterministic_tuning",
    "rq3_learned_hard_tuning",
    "rq3_learned_proportional_tuning",
]
Direction = Literal["lower", "upper"]

BASE_DEEP_LEARNING_RATES = (
    DEEP_LEARNING_RATE_ANCHOR / 2,
    DEEP_LEARNING_RATE_ANCHOR,
    DEEP_LEARNING_RATE_ANCHOR * 2,
)
PROTOCOL_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PROTOCOL_ROOT.parents[3]
CONTROL_MANIFEST_PATH = PROTOCOL_ROOT / "control_manifest.json"
CONTROL_RETRY1_INCIDENT_PATH = PROTOCOL_ROOT / "evidence/control_retry1_incident.json"
CONTROL_RETRY1_INCIDENT_SHA256 = (
    "cd5c945bccac6fdc8e31f154f53148d60497f358a0374e286732f0e5f760ae03"
)
CONTROL_RETRY1_BATCH_ID = "fa9ae431997d4b968ea3596bb4c0aa6f"
CONTROL_RETRY2_INCIDENT_PATH = PROTOCOL_ROOT / "evidence/control_retry2_incident.json"
CONTROL_RETRY2_INCIDENT_SHA256 = (
    "bb3c9391e0a9a8ea7064d6e424df8f0826a1c258f5c030f20af996c4b65498c2"
)
CONTROL_RETRY2_BATCH_ID = "a02f5ffba08d4fb9bcb9260ac149bb40"
NATIVE_PERIOD_ARTIFACT_ROOT = (
    PROJECT_ROOT / "generated/g4_native500m/selector_artifacts"
)
NATIVE_ENTRYPOINT = "experiments/g4_future_items/launchers/run_native500m.py"
_STAGE_OBJECTIVES: dict[Stage, tuple[dict[str, Any], dict[str, Any]]] = {
    "control_tuning": (
        {"id": "control_next_item"},
        {"valid_positive_mask_mode": "next_item_unique"},
    ),
    "rq1_tuning": (
        {"id": "rq1_24h", "window_seconds": 86_400},
        {"valid_positive_mask_mode": "next_24h_unique"},
    ),
    "rq2_tuning": (
        {"id": "rq2_next10", "event_lookahead": 10},
        {"valid_positive_mask_mode": "next_10_unique"},
    ),
}
_RQ3_OBJECTIVES = {
    "rq3_deterministic_tuning": (
        "rq3_deterministic_hard",
        "selected_period_union_unique",
    ),
    "rq3_learned_hard_tuning": (
        "rq3_learned_hard",
        "selected_period_union_unique",
    ),
    "rq3_learned_proportional_tuning": (
        "rq3_learned_proportional",
        "all_positive_probability_periods_unique",
    ),
}
_APPROVED_STAGES = frozenset((*_STAGE_OBJECTIVES, *_RQ3_OBJECTIVES))
_FILE_IDENTITY_KEYS = {"path", "size", "mtime_ns", "sha256"}


def canonical_bytes(document: dict[str, Any]) -> bytes:
    return json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def canonical_sha256(document: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_bytes(document)).hexdigest()


def load_strict_json(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}: {path}")
            result[key] = value
        return result

    value = json.loads(path.read_text(), object_pairs_hook=reject_duplicates)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def build_native_source_closure() -> dict[str, Any]:
    entrypoints = derive_current_entrypoint_source_paths(
        {NATIVE_ENTRYPOINT: [NATIVE_ENTRYPOINT]}, project_root=PROJECT_ROOT
    )
    paths = sorted(
        {
            *entrypoints[NATIVE_ENTRYPOINT],
            CONTROL_MANIFEST_PATH.resolve()
            .relative_to(PROJECT_ROOT.resolve())
            .as_posix(),
            CONTROL_RETRY1_INCIDENT_PATH.resolve()
            .relative_to(PROJECT_ROOT.resolve())
            .as_posix(),
            CONTROL_RETRY2_INCIDENT_PATH.resolve()
            .relative_to(PROJECT_ROOT.resolve())
            .as_posix(),
        }
    )
    unsigned = {
        "version": 1,
        "entrypoint": NATIVE_ENTRYPOINT,
        "paths": paths,
        "sources": source_manifest(PROJECT_ROOT, paths),
    }
    return unsigned | {"sha256": canonical_sha256(unsigned)}


def validate_native_source_closure(document: dict[str, Any]) -> None:
    if not isinstance(document, dict) or set(document) != {
        "version",
        "entrypoint",
        "paths",
        "sources",
        "sha256",
    }:
        raise ValueError("native-500M source closure schema differs")
    unsigned = {key: value for key, value in document.items() if key != "sha256"}
    if document["sha256"] != canonical_sha256(unsigned):
        raise ValueError("native-500M source closure hash differs")
    if document != build_native_source_closure():
        raise ValueError("native-500M source closure differs from current sources")


def resolve_native500m_data_identity(experiment: Any) -> dict[str, Any]:
    if hasattr(experiment, "base_path"):
        from utils.global_config import config as global_config

        global_config.initialize(Path(experiment.base_path))
    artifacts = experiment.artifacts
    main_path = Path(artifacts.main_parquet).resolve()
    remap_path = main_path.with_name("item_id_remap.parquet")
    identity = {
        "version": 1,
        "dataset_size": "500m",
        "dataset_key": experiment.dataset_key,
        "main": _file_identity(main_path),
        "remap": _file_identity(remap_path),
        "split_cutoff_timestamp": experiment.validation_cutoff_timestamp,
        "mapped_catalog_sha256": _mapped_catalog_sha256(remap_path),
    }
    _validate_native500m_data_identity_schema(identity)
    return identity


def validate_native500m_data_identity(
    document: dict[str, Any], experiment: Any
) -> None:
    _validate_native500m_data_identity_schema(document)
    if document != resolve_native500m_data_identity(experiment):
        raise ValueError("native-500M data identity differs from current artifacts")


def _validate_native500m_data_identity_schema(document: dict[str, Any]) -> None:
    if not isinstance(document, dict) or set(document) != {
        "version",
        "dataset_size",
        "dataset_key",
        "main",
        "remap",
        "split_cutoff_timestamp",
        "mapped_catalog_sha256",
    }:
        raise ValueError("native-500M data identity schema differs")
    if (
        document["version"] != 1
        or document["dataset_size"] != "500m"
        or not isinstance(document["dataset_key"], str)
        or not document["dataset_key"]
        or isinstance(document["split_cutoff_timestamp"], bool)
        or not isinstance(document["split_cutoff_timestamp"], int)
    ):
        raise ValueError("native-500M data identity fields differ")
    _validate_sha256("mapped_catalog_sha256", document["mapped_catalog_sha256"])
    for name in ("main", "remap"):
        identity = document[name]
        if not isinstance(identity, dict) or set(identity) != _FILE_IDENTITY_KEYS:
            raise ValueError(f"native-500M {name} identity schema differs")
        if not isinstance(identity["path"], str) or not identity["path"]:
            raise ValueError(f"native-500M {name} identity path differs")
        for field in ("size", "mtime_ns"):
            if (
                isinstance(identity[field], bool)
                or not isinstance(identity[field], int)
                or identity[field] < 0
            ):
                raise ValueError(f"native-500M {name} identity {field} differs")
        _validate_sha256(f"native-500M {name} sha256", identity["sha256"])


def _file_identity(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise ValueError(f"native-500M identity path is not a regular file: {path}")
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"native-500M identity path is not a regular file: {path}")
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest.hexdigest(),
    }


def _mapped_catalog_sha256(remap_path: Path) -> str:
    import polars as pl

    values = sorted(
        {
            int(value)
            for value in pl.read_parquet(remap_path, columns=["compact_id"])[
                "compact_id"
            ].to_list()
            if value is not None and int(value) > 0
        }
    )
    return canonical_sha256(values)


def materialization_evidence_identity(path: Path) -> dict[str, str]:
    if path.is_symlink():
        raise ValueError("native-500M materialization evidence path is invalid")
    resolved = path.resolve()
    root = PROJECT_ROOT.resolve()
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError("native-500M materialization evidence path is invalid")
    document = load_strict_json(resolved)
    if resolved.read_bytes() != canonical_bytes(document):
        raise ValueError("native-500M materialization evidence is not canonical")
    _validate_materialization_cost_evidence(document)
    _verify_materialization_artifacts(document, NATIVE_PERIOD_ARTIFACT_ROOT)
    return {
        "path": resolved.relative_to(root).as_posix(),
        "sha256": canonical_sha256(document),
    }


def validate_materialization_evidence_identity(
    identity: dict[str, str], *, objective_id: str, selector_artifact_sha256: str
) -> None:
    if not isinstance(identity, dict) or set(identity) != {"path", "sha256"}:
        raise ValueError("native-500M materialization evidence identity differs")
    _validate_sha256("materialization evidence sha256", identity["sha256"])
    path = (PROJECT_ROOT / identity["path"]).resolve()
    if materialization_evidence_identity(path) != identity:
        raise ValueError("native-500M materialization evidence file differs")
    evidence = load_strict_json(path)
    if not objective_id.startswith("rq3_"):
        raise ValueError("materialization evidence applies only to RQ3")
    expected = (
        evidence["deterministic_artifact_sha256"]
        if objective_id == "rq3_deterministic_hard"
        else evidence["learned_artifact_sha256"]
    )
    if selector_artifact_sha256 != expected:
        raise ValueError("native-500M selector differs from materialization evidence")


def load_control_manifest() -> dict[str, Any]:
    document = load_strict_json(CONTROL_MANIFEST_PATH)
    expected = {
        "version",
        "dataset_size",
        "lineage",
        "baseline",
        "selection",
        "historical_lineage_policy",
    }
    if set(document) != expected or document.get("version") != 1:
        raise ValueError("native-500M control manifest schema differs")
    if document["dataset_size"] != "500m" or document["lineage"] != "native500m-v1":
        raise ValueError("native-500M control manifest identity differs")
    selection = document["selection"]
    if selection != {
        "batch_size": BATCH_SIZE,
        "deep_learning_rate_anchor": DEEP_LEARNING_RATE_ANCHOR,
        "embedding_learning_rate": EMBEDDING_LEARNING_RATE,
        "lr_schedule_horizon_epochs": TRAINING_HORIZON_EPOCHS,
        "only_tuned_recommender_field": "deep_learning_rate",
        "seed": 42,
    }:
        raise ValueError("native-500M selection contract differs")
    baseline = document["baseline"]
    if baseline != {
        "attention_heads": 2,
        "attention_window": None,
        "bos": True,
        "cls_token_mode": "end_only",
        "ffn": "swiglu",
        "ffn_intermediate_width": 192,
        "g1_aggregate_members": [
            "swiglu",
            "scheduler",
            "position",
            "post_norm",
            "input_final_rms",
            "cls",
            "time",
            "popularity",
            "gqa",
            "bos",
        ],
        "item_embedding_width": 64,
        "kv_heads": 1,
        "model_width": 64,
        "mup_base_width": 16,
        "mup_delta_width": 32,
        "negative_count": 2048,
        "transformer_layers": 2,
    }:
        raise ValueError("native-500M control manifest baseline differs")
    if document["historical_lineage_policy"] != {
        "native50m_artifacts": "immutable_audit_only",
        "native50m_metrics_reused": False,
    }:
        raise ValueError("native-500M control manifest historical policy differs")
    return document


def compile_base_ledger(
    stage: Stage,
    *,
    retry_revision: int = 0,
    selector_artifact_sha256: str | None = None,
    materialization_evidence_path: Path | None = None,
    _materialization_evidence_identity: dict[str, str] | None = None,
) -> dict[str, Any]:
    retry = _control_retry_reference(stage, retry_revision)
    objective, loss = _objective_and_loss(stage, selector_artifact_sha256)
    conditional = _conditional_references(
        stage,
        selector_artifact_sha256=selector_artifact_sha256,
        materialization_evidence_path=materialization_evidence_path,
        materialization_identity=_materialization_evidence_identity,
    )
    protocol_sha256 = canonical_sha256(load_control_manifest())
    rows = [
        {
            "id": (
                f"{stage}:{trial_id:02d}"
                if retry_revision == 0
                else f"{stage}:retry{retry_revision}:{trial_id:02d}"
            ),
            "job": _job(
                stage=stage,
                trial_id=trial_id,
                retry_revision=retry_revision,
                deep_learning_rate=deep_learning_rate,
                objective=objective,
                loss=loss,
                materialization_evidence=conditional.get("materialization_evidence"),
            ),
        }
        for trial_id, deep_learning_rate in enumerate(BASE_DEEP_LEARNING_RATES, 1)
    ]
    return _seal(
        {
            "version": 1,
            "lineage": "native500m-v1",
            "dataset_size": "500m",
            "stage": stage,
            "control_manifest_sha256": protocol_sha256,
            **retry,
            **conditional,
            "rows": rows,
        }
    )


def compile_boundary_ledger(
    *,
    stage: Stage,
    direction: Direction,
    round_number: int,
    predecessor_ledger_paths: list[Path] | None = None,
    candidate_run_directories: list[Path] | None = None,
    selector_artifact_sha256: str | None = None,
    materialization_evidence_path: Path | None = None,
    _materialization_evidence_identity: dict[str, str] | None = None,
) -> dict[str, Any]:
    if direction not in {"lower", "upper"}:
        raise ValueError("direction must be lower or upper")
    factors = {
        ("lower", 1): (1 / 8, 1 / 4),
        ("upper", 1): (4, 8),
        ("lower", 2): (1 / 32, 1 / 16),
        ("upper", 2): (16, 32),
    }.get((direction, round_number))
    if factors is None or isinstance(round_number, bool):
        raise ValueError("round_number must be 1 or 2")
    objective, loss = _objective_and_loss(stage, selector_artifact_sha256)
    conditional = _conditional_references(
        stage,
        selector_artifact_sha256=selector_artifact_sha256,
        materialization_evidence_path=materialization_evidence_path,
        materialization_identity=_materialization_evidence_identity,
    )
    predecessor = _boundary_predecessor_winner(
        stage=stage,
        direction=direction,
        round_number=round_number,
        ledger_paths=predecessor_ledger_paths or [],
        run_directories=candidate_run_directories or [],
    )
    boundary_stage = f"{stage}_boundary"
    rows = [
        {
            "id": f"{boundary_stage}:{direction}:r{round_number}:{trial_id:02d}",
            "job": _job(
                stage=stage,
                trial_id=trial_id,
                deep_learning_rate=DEEP_LEARNING_RATE_ANCHOR * factor,
                objective=objective,
                loss=loss,
                boundary_direction=direction,
                boundary_round=round_number,
                materialization_evidence=conditional.get("materialization_evidence"),
            ),
        }
        for trial_id, factor in enumerate(factors, 1)
    ]
    return _seal(
        {
            "version": 1,
            "lineage": "native500m-v1",
            "dataset_size": "500m",
            "stage": boundary_stage,
            "base_stage": stage,
            "direction": direction,
            "round_number": round_number,
            "control_manifest_sha256": canonical_sha256(load_control_manifest()),
            **conditional,
            "entering_row_sha256": canonical_sha256(predecessor["winner_row"]),
            "entering_row": predecessor["winner_row"],
            "predecessor_evidence": {
                "ledgers": [
                    _file_identity(path) for path in predecessor["ledger_paths"]
                ],
                "runs": [
                    {
                        "job_contract": _file_identity(path / "g4_job.json"),
                        "training_metadata": _file_identity(
                            path / "training_metadata.json"
                        ),
                        "sweep_log": _file_identity(path / "sweep.log"),
                    }
                    for path in predecessor["run_directories"]
                ],
            },
            "rows": rows,
        }
    )


def _boundary_predecessor_winner(
    *,
    stage: Stage,
    direction: Direction,
    round_number: int,
    ledger_paths: list[Path],
    run_directories: list[Path],
) -> dict[str, Any]:
    from experiments.g4_future_items.report.artifacts import read_recommender_trial

    resolved_ledgers = [path.resolve() for path in ledger_paths]
    resolved_runs = [path.resolve() for path in run_directories]
    if (
        len(resolved_ledgers) != round_number
        or len(set(resolved_ledgers)) != len(resolved_ledgers)
        or len(set(resolved_runs)) != len(resolved_runs)
    ):
        raise ValueError("native-500M boundary predecessor set is incomplete")
    ledgers = [load_frozen_ledger(path) for path in resolved_ledgers]
    if not ledgers or ledgers[0]["stage"] != stage:
        raise ValueError("native-500M boundary predecessor base stage differs")
    for index, ledger in enumerate(ledgers[1:], 1):
        if (
            ledger["stage"] != f"{stage}_boundary"
            or ledger["base_stage"] != stage
            or ledger["direction"] != direction
            or ledger["round_number"] != index
        ):
            raise ValueError("native-500M boundary predecessor round sequence differs")

    contracts: dict[tuple[str, str], tuple[Path, dict[str, Any]]] = {}
    for run_directory in resolved_runs:
        contract = load_strict_json(run_directory / "g4_job.json")
        if set(contract) != {
            "ledger_sha256",
            "row_id",
            "job",
            "source_closure",
            "data_identity",
            "ledger_path",
        }:
            raise ValueError("native-500M boundary job contract schema differs")
        key = (contract["ledger_sha256"], contract["row_id"])
        if key in contracts:
            raise ValueError("native-500M boundary evidence is duplicated")
        contracts[key] = (run_directory, contract)

    candidates = []
    source_closure = None
    data_identity = None
    for ledger_path, ledger in zip(resolved_ledgers, ledgers):
        for row in ledger["rows"]:
            candidate = contracts.pop((ledger["sha256"], row["id"]), None)
            if candidate is None:
                raise ValueError(
                    "native-500M boundary candidate evidence is incomplete"
                )
            run_directory, contract = candidate
            if (
                Path(contract["ledger_path"]).resolve() != ledger_path
                or contract["job"] != row["job"]
            ):
                raise ValueError("native-500M boundary candidate differs from ledger")
            experiment = build_runtime_experiment(row["job"])
            if source_closure is None:
                try:
                    validate_native_source_closure(contract["source_closure"])
                    validate_native500m_data_identity(
                        contract["data_identity"], experiment
                    )
                except ValueError as error:
                    raise ValueError(
                        "native-500M boundary execution identity is unauthenticated"
                    ) from error
                source_closure = contract["source_closure"]
                data_identity = contract["data_identity"]
            elif (
                contract["source_closure"] != source_closure
                or contract["data_identity"] != data_identity
            ):
                raise ValueError("native-500M boundary execution identity differs")
            trial = read_recommender_trial(run_directory)
            if not trial.usable:
                raise ValueError("native-500M boundary candidate is not usable")
            candidates.append((trial, row))
    if contracts:
        raise ValueError("native-500M boundary evidence contains extra runs")

    _, (winner_trial, winner_row) = min(
        enumerate(candidates),
        key=lambda indexed: (
            -indexed[1][0].validation_recall_at_100,
            indexed[1][0].validation_loss,
            indexed[0],
        ),
    )
    expected_edge = {
        ("lower", 1): DEEP_LEARNING_RATE_ANCHOR / 2,
        ("upper", 1): DEEP_LEARNING_RATE_ANCHOR * 2,
        ("lower", 2): DEEP_LEARNING_RATE_ANCHOR / 8,
        ("upper", 2): DEEP_LEARNING_RATE_ANCHOR * 8,
    }[(direction, round_number)]
    if winner_row["job"]["deep_learning_rate"] != expected_edge:
        raise ValueError(
            "native-500M cumulative winner is not on the requested outer edge"
        )
    if winner_trial.row_id != winner_row["id"]:
        raise ValueError("native-500M boundary winner identity differs")
    return {
        "ledger_paths": resolved_ledgers,
        "run_directories": resolved_runs,
        "winner_row": winner_row,
    }


def load_frozen_ledger(path: Path) -> dict[str, Any]:
    document = load_strict_json(path)
    supplied_sha256 = document.get("sha256")
    if not isinstance(supplied_sha256, str) or len(supplied_sha256) != 64:
        raise ValueError("native-500M ledger has no canonical SHA-256")
    unsealed = {key: value for key, value in document.items() if key != "sha256"}
    if canonical_sha256(unsealed) != supplied_sha256:
        raise ValueError("native-500M ledger SHA-256 differs")
    try:
        expected = _compile_document_contract(document)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("native-500M ledger is not compiler-equivalent") from error
    if canonical_bytes(document) != canonical_bytes(expected):
        raise ValueError("native-500M ledger is not compiler-equivalent")
    return document


def validate_runtime_experiment(experiment: Any, job: dict[str, Any]) -> None:
    load_control_manifest()
    actual = _experiment_projection(experiment) | {
        "objective": (
            experiment.objective_id,
            experiment.objective_window_seconds,
            experiment.objective_event_lookahead,
            experiment.selector_artifact_sha256,
            experiment.objective_period_count,
            experiment.valid_positive_mask_mode,
        ),
    }
    objective = job["objective"]
    expected = _runtime_contract(job) | {
        "objective": (
            objective["id"],
            objective.get("window_seconds"),
            objective.get("event_lookahead"),
            objective.get("selector_artifact_sha256"),
            objective.get("period_count"),
            job["loss"]["valid_positive_mask_mode"],
        ),
    }
    if actual != expected:
        raise ValueError("native-500M runtime experiment differs from its manifest row")


def build_runtime_experiment(job: dict[str, Any]) -> Any:
    common = {
        "run_name": job["run_name"],
        "deep_learning_rate": job["deep_learning_rate"],
        "seed": job["seed"],
    }
    objective = job["objective"]
    if objective["id"] == "control_next_item":
        experiment = build_native500m_control(**common)
    else:
        experiment = build_native500m_treatment(
            **common,
            objective=objective,
            valid_positive_mask_mode=job["loss"]["valid_positive_mask_mode"],
        )
    validate_runtime_experiment(experiment, job)
    if objective["id"].startswith("rq3_"):
        validate_materialization_evidence_identity(
            job["materialization_evidence"],
            objective_id=objective["id"],
            selector_artifact_sha256=objective["selector_artifact_sha256"],
        )
    elif "materialization_evidence" in job:
        raise ValueError("materialization evidence applies only to RQ3")
    return experiment


def _compile_document_contract(document: dict[str, Any]) -> dict[str, Any]:
    stage = document["stage"]
    selector_sha256 = document.get("selector_artifact_sha256")
    materialization_identity = document.get("materialization_evidence")
    if stage in _APPROVED_STAGES:
        return compile_base_ledger(
            stage,
            retry_revision=document.get("retry_revision", 0),
            selector_artifact_sha256=selector_sha256,
            _materialization_evidence_identity=materialization_identity,
        )
    base_stage = document["base_stage"]
    if base_stage not in _APPROVED_STAGES or stage != f"{base_stage}_boundary":
        raise ValueError("unapproved native-500M boundary stage")
    ledger_paths, run_directories = _predecessor_paths(document["predecessor_evidence"])
    return compile_boundary_ledger(
        stage=base_stage,
        direction=document["direction"],
        round_number=document["round_number"],
        predecessor_ledger_paths=ledger_paths,
        candidate_run_directories=run_directories,
        selector_artifact_sha256=selector_sha256,
        _materialization_evidence_identity=materialization_identity,
    )


def _predecessor_paths(evidence: Any) -> tuple[list[Path], list[Path]]:
    if not isinstance(evidence, dict) or set(evidence) != {"ledgers", "runs"}:
        raise ValueError("native-500M boundary predecessor evidence schema differs")
    ledgers = evidence["ledgers"]
    runs = evidence["runs"]
    if not isinstance(ledgers, list) or not ledgers or not isinstance(runs, list):
        raise ValueError("native-500M boundary predecessor evidence is incomplete")
    identities = list(ledgers)
    for run in runs:
        if not isinstance(run, dict) or set(run) != {
            "job_contract",
            "training_metadata",
            "sweep_log",
        }:
            raise ValueError("native-500M boundary run evidence differs")
        identities.extend(run.values())
    for identity in identities:
        if (
            not isinstance(identity, dict)
            or set(identity) != _FILE_IDENTITY_KEYS
            or _file_identity(Path(identity.get("path", ""))) != identity
        ):
            raise ValueError("native-500M boundary file evidence differs")
    return (
        [Path(identity["path"]) for identity in ledgers],
        [Path(run["job_contract"]["path"]).parent for run in runs],
    )


def _experiment_projection(experiment: Any) -> dict[str, Any]:
    excluded = {"base_path", "_prepared_train_iterator"}
    return {
        field.name: _project_value(getattr(experiment, field.name))
        for field in fields(MuTransferGenerationExperiment)
        if field.name not in excluded
    }


def _project_value(value: Any) -> Any:
    return asdict(value) if is_dataclass(value) else value


def _runtime_contract(job: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_name": job["run_name"],
        "seed": job["seed"],
        "invalidate_cache": False,
        "runtime": {
            "dtype": torch.bfloat16,
            "compile": False,
            "gradient_clip_norm": None,
        },
        "day_range": {"start_day": 0, "end_day": 300},
        "dataloader": {
            "batch_size": BATCH_SIZE,
            "val_batch_size": 8192,
            "num_workers": 4,
            "prefetch_factor": 4,
            "gradient_accumulation_steps": 1,
        },
        "pretrain": {"days": 0, "num_epochs": 1, "shuffle_days": True},
        "checkpointing": {
            "enabled": False,
            "best_strategy": "best_n",
            "best_n_checkpoints": 3,
            "best_metric_name": "recall@100",
            "best_metric_mode": "max",
            "best_metric_prefix": "epoch/val_true",
            "load_checkpoint": False,
            "last_n_checkpoints": 1,
        },
        "logging": {
            "log_interval": 100,
            "enable_predictions": True,
            "wandb_project": "ysda_recsys",
            "prediction_int_columns": {},
            "prediction_float_columns": {},
        },
        "lr_schedule": {
            "shape": "cosine",
            "warmup_fraction": 0.05,
            "min_lr_fraction": 0.0,
            "cycles": 1,
            "timescale_steps": None,
            "timescale_fraction": None,
            "power_exponent": -0.51,
            "power_transition_tokens": None,
            "optimizer_group_scope": "deep_only",
        },
        "num_epochs": TRAINING_HORIZON_EPOCHS,
        "lr_schedule_horizon_epochs": TRAINING_HORIZON_EPOCHS,
        "max_seq_len": 100,
        "min_seq_len": 2,
        "window": "next_item",
        "stride": 1.0,
        "prefix_length_rule": "truncated",
        "prefix_cap": None,
        "validation_days": 1,
        "validation_interval_seconds": 604_800,
        "embedding_learning_rate": EMBEDDING_LEARNING_RATE,
        "deep_learning_rate": job["deep_learning_rate"],
        "weight_decay": 0.0,
        "size": "500m",
        "user_sample": None,
        "listen_sample_fraction": 1.0,
        "event_type_filter": "like",
        "min_item_interactions_per_item": 5,
        "drop_unmapped_items": True,
        "transformer": {
            "dim": 64,
            "num_layers": 2,
            "nhead": 2,
            "num_kv_heads": 1,
            "ffn_intermediate_dim": 192,
            "dropout": 0.1,
            "input_dropout": 0.1,
            "ffn_dropout": 0.1,
            "gated_ffn_dropout": True,
            "ffn": "swiglu",
            "norm": "layer",
            "norm_place": "post",
            "input_norm": "rms",
            "final_norm": "rms",
            "alibi": True,
            "rope": "timestamp_reverse",
            "rope_base": 10000.0,
            "learned_positions": ("forward", "reverse"),
            "learned_position_fusion": "concat",
            "learned_position_fusion_normalization": None,
            "learned_position_fusion_residual": "rezero",
            "learned_position_initialization": "default",
            "learned_position_reverse_correction": "bounded_tanh",
            "learned_position_reverse_max_scale": 0.025,
            "learned_position_reverse_initializer_rng_nonadvancing": True,
            "attention_window": None,
        },
        "bos": True,
        "cls_token": False,
        "cls_token_mode": "end_only",
        "num_in_batch_negatives": 2048,
        "logq_alpha": 0.01,
        "eval_ks": (10, 50, 100),
        "eval_max_users": 20_000,
        "selection_k": 100,
        "evaluation_catalog": "all",
        "exclude_seen_from_evaluation": False,
        "eval_every_n_epochs": 1,
        "restore_best_weights": True,
        "early_stopping_patience": 3,
        "early_stopping_min_delta": 0.0,
        "adaptive_schedule_early_stopping": False,
        "per_layer_item_embeddings": False,
        "per_layer_item_features": "none",
        "per_layer_item_feature_dim": None,
        "item_embedding_dim": 64,
        "timestamp_delta": "bins",
        "timestamp_combination": "add",
        "timestamp_num_bins": 32,
        "negative_sampling": "random_offline_logq",
        "logq_correction": "yi2019",
        "correct_positive_logq": True,
        "mask_false_negatives": False,
        "exclude_own_group_negatives": False,
        "dense_random_negative_scores": True,
        "random_negative_fraction": 0.5,
        "initializer_std": 0.02,
        "final_ranking_evidence_group": "g4-native500m",
        "mup_base_dim": 16,
        "mup_delta_dim": 32,
        "mup_base_ffn_dim": None,
        "mup_delta_ffn_dim": None,
    }


def write_frozen_document(path: Path, document: dict[str, Any]) -> None:
    content = canonical_bytes(document)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"frozen native-500M document differs: {path}")
        return
    path.write_bytes(content)
    path.chmod(0o444)


def _job(
    *,
    stage: Stage,
    trial_id: int,
    retry_revision: int = 0,
    deep_learning_rate: float,
    objective: dict[str, Any],
    loss: dict[str, Any],
    boundary_direction: Direction | None = None,
    boundary_round: int | None = None,
    materialization_evidence: dict[str, str] | None = None,
) -> dict[str, Any]:
    suffix = {
        "control_tuning": "control",
        "rq1_tuning": "rq1_24h",
        "rq2_tuning": "rq2_next10",
        "rq3_deterministic_tuning": "rq3_deterministic_hard",
        "rq3_learned_hard_tuning": "rq3_learned_hard",
        "rq3_learned_proportional_tuning": "rq3_learned_proportional",
    }[stage]
    boundary = (
        ""
        if boundary_round is None
        else f"_boundary_{boundary_direction}_r{boundary_round}"
    )
    protocol: dict[str, Any] = {"stage": stage, "trial_id": trial_id}
    retry = "" if retry_revision == 0 else f"_retry{retry_revision}"
    if retry_revision != 0:
        protocol["retry_revision"] = retry_revision
    if boundary_round is not None:
        protocol.update(
            boundary_direction=boundary_direction,
            boundary_round=boundary_round,
        )
    job = {
        "run_name": (f"g4_{suffix}{boundary}_trial_{trial_id:02d}{retry}_native500m"),
        "protocol": protocol,
        "dataloader": {"batch_size": BATCH_SIZE},
        "embedding_learning_rate": EMBEDDING_LEARNING_RATE,
        "deep_learning_rate": deep_learning_rate,
        "lr_schedule_horizon_epochs": TRAINING_HORIZON_EPOCHS,
        "seed": 42,
        "objective": dict(objective),
        "loss": dict(loss),
    }
    if materialization_evidence is not None:
        job["materialization_evidence"] = dict(materialization_evidence)
    return job


def _control_retry_reference(stage: Stage, retry_revision: int) -> dict[str, Any]:
    if retry_revision == 0 and not isinstance(retry_revision, bool):
        return {}
    if isinstance(retry_revision, bool) or stage != "control_tuning":
        raise ValueError("native-500M retry revision is not authorized")
    retry_evidence = {
        1: (
            CONTROL_RETRY1_INCIDENT_PATH,
            CONTROL_RETRY1_INCIDENT_SHA256,
            CONTROL_RETRY1_BATCH_ID,
        ),
        2: (
            CONTROL_RETRY2_INCIDENT_PATH,
            CONTROL_RETRY2_INCIDENT_SHA256,
            CONTROL_RETRY2_BATCH_ID,
        ),
    }
    try:
        incident_path, incident_sha256, batch_id = retry_evidence[retry_revision]
    except KeyError as error:
        raise ValueError("native-500M retry revision is not authorized") from error
    document = load_strict_json(incident_path)
    if (
        canonical_sha256(document) != incident_sha256
        or document.get("batch_id") != batch_id
        or document.get("retry_revision") != retry_revision
        or document.get("stage") != stage
    ):
        raise ValueError("native-500M retry incident differs")
    return {
        "retry_revision": retry_revision,
        "retry_incident": {
            "path": incident_path.resolve()
            .relative_to(PROJECT_ROOT.resolve())
            .as_posix(),
            "sha256": incident_sha256,
            "batch_id": batch_id,
        },
    }


def _objective_and_loss(
    stage: Stage, selector_artifact_sha256: str | None
) -> tuple[dict[str, Any], dict[str, Any]]:
    fixed = _STAGE_OBJECTIVES.get(stage)
    if fixed is not None:
        if selector_artifact_sha256 is not None:
            raise ValueError("selector artifact applies only to RQ3")
        return fixed
    rq3 = _RQ3_OBJECTIVES.get(stage)
    if rq3 is None:
        raise ValueError(f"stage {stage!r} is not approved")
    _validate_sha256("selector_artifact_sha256", selector_artifact_sha256)
    objective_id, mask = rq3
    return (
        {
            "id": objective_id,
            "selector_artifact_sha256": selector_artifact_sha256,
            "period_count": 1,
        },
        {"valid_positive_mask_mode": mask},
    )


def _conditional_references(
    stage: Stage,
    *,
    selector_artifact_sha256: str | None,
    materialization_evidence_path: Path | None,
    materialization_identity: dict[str, str] | None,
) -> dict[str, Any]:
    if stage not in _RQ3_OBJECTIVES:
        if (
            materialization_evidence_path is not None
            or materialization_identity is not None
        ):
            raise ValueError("materialization evidence applies only to RQ3")
        return {}
    assert selector_artifact_sha256 is not None
    if (materialization_evidence_path is None) == (materialization_identity is None):
        raise ValueError("RQ3 requires exactly one materialization evidence source")
    identity = (
        materialization_evidence_identity(materialization_evidence_path)
        if materialization_evidence_path is not None
        else materialization_identity
    )
    assert identity is not None
    objective_id = _RQ3_OBJECTIVES[stage][0]
    validate_materialization_evidence_identity(
        identity,
        objective_id=objective_id,
        selector_artifact_sha256=selector_artifact_sha256,
    )
    return {
        "selector_artifact_sha256": selector_artifact_sha256,
        "materialization_evidence": identity,
    }


def _validate_sha256(name: str, value: str | None) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")


def _seal(document: dict[str, Any]) -> dict[str, Any]:
    return document | {"sha256": canonical_sha256(document)}
