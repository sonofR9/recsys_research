from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.g3_pretrained_item_embeddings.data import load_feature_data
from experiments.g3_pretrained_item_embeddings.diagnostics import (
    build_frequency_identity,
)

from .constants import APPROVED_PROTOCOL_SHA256
from .manifests import load_artifact_manifest, validate_feature_manifest
from .search import (
    APPROVED_FAMILY_SPECS,
    ReusableCoordinate,
    SearchCoordinate,
    compile_family,
)

RQ3_OUTPUT_FAMILY_IDS = (
    "rq3_output_learned",
    "rq3_output_frozen_content",
    "rq3_output_trainable_content",
    "rq3_output_learned_frozen_content",
    "rq3_output_learned_trainable_content",
)

NATIVE50_FEATURE_MANIFEST_PATH = (
    "experiments/g3_pretrained_item_embeddings/protocol/artifacts/"
    "native50m_features.json"
)
NATIVE50_FEATURE_MANIFEST_SHA256 = (
    "02e919339094e5091e77d09bd77ea669b665c7f6f49a29b6f27d6708ee9cf021"
)
NATIVE50_FEATURE_MANIFEST_FILE_SHA256 = (
    "3899b01471f31c0ec331d975adf563ddd45d203ebc1786797b8a63fa9d495098"
)

_RQ2_DATASET = {
    "candidate_catalog": "full",
    "event_limit": 50_000_000,
    "exclude_seen": False,
    "minimum_user_interactions": 5,
    "sampling": "none",
    "size": "native-50m",
    "source": "likes",
    "validation_interval_seconds": 604800,
}

_RQ2_DATA_INVARIANTS = {
    "dataset_size": "50m",
    "user_sample": None,
    "event_type_filter": "like",
    "min_item_interactions_per_item": 5,
    "drop_unmapped_items": True,
    "validation_interval_seconds": 604800,
    "day_range": {"start_day": 0, "end_day": 300},
    "window": "next_item",
    "evaluation_catalog": "all",
    "exclude_seen_from_evaluation": False,
    "eval_ks": [10, 50, 100],
    "selection_k": 100,
    "eval_max_users": 20000,
    "eval_every_n_epochs": 1,
}

_RQ2_LOSS_INVARIANTS = {
    "negative_sampling": "random",
    "num_in_batch_negatives": 512,
    "logq_correction": "yi2019",
    "random_negative_fraction": 0.5,
    "logq_alpha": 0.01,
    "correct_positive_logq": False,
    "mask_false_negatives": False,
    "exclude_own_group_negatives": False,
    "dense_random_negative_scores": True,
}

_RQ2_SCHEDULE_INVARIANTS = {
    "restore_best_weights": True,
    "adaptive_schedule_early_stopping": False,
    "lr_schedule": {
        "cycles": 1,
        "min_lr_fraction": 0.0,
        "optimizer_group_scope": "both",
        "power_exponent": -0.51,
        "power_transition_tokens": None,
        "shape": "linear",
        "timescale_fraction": None,
        "timescale_steps": None,
        "warmup_fraction": 0.0,
    },
}


@dataclass(frozen=True)
class AuthenticatedRq2Coordinate:
    source_id: str
    run_name: str
    history_hidden_dim: int
    embedding_learning_rate: float
    deep_learning_rate: float
    horizon_epochs: int
    source_ledger_path: str
    source_ledger_sha256: str
    source_evidence_sha256: str
    artifact_sha256: tuple[tuple[str, str], ...]
    training_count_sha256: str
    slice_membership_sha256: str
    diagnostics_schema_version: int
    diagnostics_epoch_count: int


@dataclass(frozen=True)
class AuthenticatedRq2ReuseSet:
    coordinates: tuple[AuthenticatedRq2Coordinate, ...]
    feature_manifest_path: str
    feature_manifest_sha256: str
    feature_manifest_file_sha256: str
    feature_data_path: str
    feature_data_sha256: str
    frequency_terciles: dict[str, object]
    training_count_reference: dict[str, object]
    slice_membership_reference: dict[str, object]


@dataclass(frozen=True)
class Rq3OutputRow:
    id: str
    family_id: str
    run_name: str
    batch_size: int
    seed: int
    embedding_learning_rate: float
    deep_learning_rate: float
    horizon_epochs: int
    history_hidden_dim: int
    reused_from: str | None
    authenticated_source: AuthenticatedRq2Coordinate | None


@dataclass(frozen=True)
class Rq3OutputSurface:
    selection_path: str
    selection_sha256: str
    selected_history_hidden_dim: int
    feature_manifest_path: str
    feature_manifest_sha256: str
    feature_manifest_file_sha256: str
    feature_data_path: str
    feature_data_sha256: str
    frequency_terciles: dict[str, object]
    training_count_reference: dict[str, object]
    slice_membership_reference: dict[str, object]
    rows_by_family: dict[str, tuple[Rq3OutputRow, ...]]
    final_rq2_evidence_sha256: str | None = None
    selected_rq2_row_id: str | None = None
    source_ledgers: tuple[tuple[str, str, str], ...] = ()


@dataclass(frozen=True)
class _FeatureIdentity:
    manifest_path: str
    manifest_sha256: str
    manifest_file_sha256: str
    data_path: str
    data_sha256: str
    frequency_terciles: dict[str, object]
    training_count_reference: dict[str, object]
    slice_membership_reference: dict[str, object]


@dataclass(frozen=True)
class _DiagnosticsIdentity:
    schema_version: int
    epoch_count: int


@dataclass(frozen=True)
class _CatalogDiagnosticsContract:
    components: frozenset[str]
    table_parameters: frozenset[str]
    content_trainable: bool | None


_BASE_DIAGNOSTIC_COMPONENTS = frozenset(
    {"catalog_encoder", "history_encoder", "sequence_model"}
)
_CONTENT_DIAGNOSTIC_COMPONENTS = frozenset(
    {"catalog_content_table", "catalog_projection"}
)
_ITEM_CONTENT_DIAGNOSTIC_COMPONENTS = frozenset({"catalog_item_table"})
_CATALOG_DIAGNOSTICS = {
    "learned_id": _CatalogDiagnosticsContract(
        _BASE_DIAGNOSTIC_COMPONENTS,
        frozenset({"weight"}),
        None,
    ),
    "frozen_content": _CatalogDiagnosticsContract(
        _BASE_DIAGNOSTIC_COMPONENTS | _CONTENT_DIAGNOSTIC_COMPONENTS,
        frozenset(),
        False,
    ),
    "trainable_content": _CatalogDiagnosticsContract(
        _BASE_DIAGNOSTIC_COMPONENTS | _CONTENT_DIAGNOSTIC_COMPONENTS,
        frozenset({"content.embedding.weight"}),
        True,
    ),
    "id_frozen_content": _CatalogDiagnosticsContract(
        _BASE_DIAGNOSTIC_COMPONENTS
        | _CONTENT_DIAGNOSTIC_COMPONENTS
        | _ITEM_CONTENT_DIAGNOSTIC_COMPONENTS,
        frozenset({"item_embedding.weight"}),
        False,
    ),
    "id_trainable_content": _CatalogDiagnosticsContract(
        _BASE_DIAGNOSTIC_COMPONENTS
        | _CONTENT_DIAGNOSTIC_COMPONENTS
        | _ITEM_CONTENT_DIAGNOSTIC_COMPONENTS,
        frozenset({"item_embedding.weight", "content.embedding.weight"}),
        True,
    ),
}


def authenticate_rq2_reuse_rows(
    *,
    root: Path,
    rows: Sequence[Mapping[str, object]],
    source_ledgers: Mapping[str, object],
    source_evidence_sha256: str,
) -> AuthenticatedRq2ReuseSet:
    root = root.resolve(strict=True)
    if not isinstance(source_evidence_sha256, str) or not _valid_sha256(
        source_evidence_sha256
    ):
        raise ValueError("RQ3 source evidence hash is invalid")
    manifest_path = _resolve_input_path(root, root / NATIVE50_FEATURE_MANIFEST_PATH)
    feature_identity = _authenticate_feature_manifest(
        root,
        {
            "path": NATIVE50_FEATURE_MANIFEST_PATH,
            "sha256": NATIVE50_FEATURE_MANIFEST_FILE_SHA256,
            "size_bytes": manifest_path.stat().st_size,
            "logical_sha256": NATIVE50_FEATURE_MANIFEST_SHA256,
        },
    )
    caches: dict[tuple[str, str], dict[str, object]] = {}
    coordinates = tuple(
        _authenticate_final_evidence_coordinate(
            root,
            row,
            source_ledger_reference=source_ledgers.get(row.get("row_id")),
            source_evidence_sha256=source_evidence_sha256,
            feature_identity=feature_identity,
            caches=caches,
        )
        for row in rows
    )
    return AuthenticatedRq2ReuseSet(
        coordinates=coordinates,
        feature_manifest_path=feature_identity.manifest_path,
        feature_manifest_sha256=feature_identity.manifest_sha256,
        feature_manifest_file_sha256=feature_identity.manifest_file_sha256,
        feature_data_path=feature_identity.data_path,
        feature_data_sha256=feature_identity.data_sha256,
        frequency_terciles=feature_identity.frequency_terciles,
        training_count_reference=feature_identity.training_count_reference,
        slice_membership_reference=feature_identity.slice_membership_reference,
    )


def _authenticate_final_evidence_coordinate(
    root: Path,
    row: Mapping[str, object],
    *,
    source_ledger_reference: object,
    source_evidence_sha256: str,
    feature_identity: _FeatureIdentity,
    caches: dict[tuple[str, str], dict[str, object]],
) -> AuthenticatedRq2Coordinate:
    source_id = row.get("row_id")
    run_name = row.get("run_name")
    if (
        not isinstance(source_id, str)
        or not source_id
        or not isinstance(run_name, str)
        or not run_name
        or row.get("family_id") != "rq2_content_concat"
        or row.get("capacity") != 128
    ):
        raise ValueError("RQ3 final-evidence reuse row identity is invalid")
    selected = {
        "run_name": run_name,
        "embedding_learning_rate": row.get("embedding_learning_rate"),
        "deep_learning_rate": row.get("deep_learning_rate"),
        "horizon_epochs": row.get("horizon_epochs"),
    }
    _validate_coordinate_types(selected)
    ledger, ledger_reference = _load_reference(
        root,
        source_ledger_reference,
        caches=caches,
    )
    if ledger.get("protocol_sha256") != APPROVED_PROTOCOL_SHA256:
        raise ValueError("RQ3 source ledger uses another protocol")
    ledger_row = _unique_row(ledger.get("rows"), "id", source_id, "source ledger")
    _validate_ledger_row(selected, ledger_row, width=128)
    raw_artifacts = row.get("artifacts")
    required_artifacts = {
        "job_contract",
        "training_metadata",
        "final_metrics",
        "training_diagnostics",
    }
    if not isinstance(raw_artifacts, dict) or not required_artifacts <= set(raw_artifacts):
        raise ValueError("RQ3 final-evidence reuse artifacts are incomplete")
    artifacts = _validate_artifacts(
        root,
        {name: raw_artifacts[name] for name in required_artifacts},
    )
    diagnostics = _validate_runtime_artifacts(
        root,
        artifacts,
        ledger_row=ledger_row,
        ledger_reference=ledger_reference,
        width=128,
        feature_identity=feature_identity,
    )
    if diagnostics.schema_version == 1:
        _validate_ledger_feature_binding(
            root,
            ledger,
            expected=feature_identity,
            visited={str(ledger_reference["path"])},
        )
    return AuthenticatedRq2Coordinate(
        source_id=source_id,
        run_name=run_name,
        history_hidden_dim=128,
        embedding_learning_rate=float(selected["embedding_learning_rate"]),
        deep_learning_rate=float(selected["deep_learning_rate"]),
        horizon_epochs=int(selected["horizon_epochs"]),
        source_ledger_path=str(ledger_reference["path"]),
        source_ledger_sha256=str(ledger_reference["logical_sha256"]),
        source_evidence_sha256=source_evidence_sha256,
        artifact_sha256=tuple(
            (name, str(reference["sha256"]))
            for name, reference in sorted(artifacts.items())
        ),
        training_count_sha256=str(feature_identity.training_count_reference["sha256"]),
        slice_membership_sha256=str(
            feature_identity.slice_membership_reference["sha256"]
        ),
        diagnostics_schema_version=diagnostics.schema_version,
        diagnostics_epoch_count=diagnostics.epoch_count,
    )


def compile_rq3_output_surface(
    *,
    root: Path,
    selection_path: Path,
    expected_selection_sha256: str,
    expected_final_rq2_evidence_sha256: str | None = None,
) -> Rq3OutputSurface:
    root = root.resolve(strict=True)
    selection_path = _resolve_input_path(root, selection_path)
    selection = _load_json(selection_path)
    coordinates, width, feature_identity = _authenticate_selection(
        root,
        selection,
        expected_selection_sha256=expected_selection_sha256,
        expected_final_rq2_evidence_sha256=expected_final_rq2_evidence_sha256,
    )
    specifications = {
        specification.id: specification
        for specification in APPROVED_FAMILY_SPECS
        if specification.id in RQ3_OUTPUT_FAMILY_IDS
    }
    if tuple(specifications) != RQ3_OUTPUT_FAMILY_IDS:
        raise ValueError("approved RQ3 family definitions changed")
    reusable = tuple(
        ReusableCoordinate(
            source_id=coordinate.source_id,
            embedding_learning_rate=coordinate.embedding_learning_rate,
            deep_learning_rate=coordinate.deep_learning_rate,
            horizon_epochs=coordinate.horizon_epochs,
        )
        for coordinate in coordinates
    )
    learned = compile_family(
        specifications["rq3_output_learned"],
        reusable=reusable,
        allow_reusable_outside_search_space=True,
    )
    authenticated = {coordinate.source_id: coordinate for coordinate in coordinates}
    rows_by_family = {
        family_id: tuple(
            _row_from_coordinate(
                family_id=family_id,
                coordinate=replace(
                    coordinate,
                    id=f"{family_id}:{index + 1:02d}",
                    family_id=family_id,
                    reused_from=(
                        coordinate.reused_from
                        if family_id == "rq3_output_learned"
                        else None
                    ),
                ),
                width=width,
                authenticated=authenticated,
            )
            for index, coordinate in enumerate(learned)
        )
        for family_id in RQ3_OUTPUT_FAMILY_IDS
    }
    return Rq3OutputSurface(
        selection_path=str(selection_path.relative_to(root)),
        selection_sha256=expected_selection_sha256,
        selected_history_hidden_dim=width,
        feature_manifest_path=feature_identity.manifest_path,
        feature_manifest_sha256=feature_identity.manifest_sha256,
        feature_manifest_file_sha256=feature_identity.manifest_file_sha256,
        feature_data_path=feature_identity.data_path,
        feature_data_sha256=feature_identity.data_sha256,
        frequency_terciles=feature_identity.frequency_terciles,
        training_count_reference=feature_identity.training_count_reference,
        slice_membership_reference=feature_identity.slice_membership_reference,
        rows_by_family=rows_by_family,
        final_rq2_evidence_sha256=expected_final_rq2_evidence_sha256,
        source_ledgers=tuple(
            (
                coordinate.source_id,
                coordinate.source_ledger_path,
                coordinate.source_ledger_sha256,
            )
            for coordinate in coordinates
        ),
    )


def _row_from_coordinate(
    *,
    family_id: str,
    coordinate: SearchCoordinate,
    width: int,
    authenticated: Mapping[str, AuthenticatedRq2Coordinate],
) -> Rq3OutputRow:
    source = (
        authenticated[coordinate.reused_from]
        if coordinate.reused_from is not None
        else None
    )
    run_name = (
        source.run_name
        if source is not None
        else f"g3_{family_id}_trial_{coordinate.opportunity_index + 1:02d}_native50m"
    )
    return Rq3OutputRow(
        id=coordinate.id,
        family_id=family_id,
        run_name=run_name,
        batch_size=coordinate.batch_size,
        seed=coordinate.seed,
        embedding_learning_rate=coordinate.embedding_learning_rate,
        deep_learning_rate=coordinate.deep_learning_rate,
        horizon_epochs=coordinate.horizon_epochs,
        history_hidden_dim=width,
        reused_from=coordinate.reused_from,
        authenticated_source=source,
    )


def _authenticate_selection(
    root: Path,
    document: dict[str, object],
    *,
    expected_selection_sha256: str,
    expected_final_rq2_evidence_sha256: str | None,
) -> tuple[tuple[AuthenticatedRq2Coordinate, ...], int, _FeatureIdentity]:
    expected_keys = {
        "schema_version",
        "kind",
        "protocol_sha256",
        "selected_family_id",
        "selected_history_hidden_dim",
        "selection_resolved",
        "feature_manifest",
        "source_evidence",
        "rows",
        "sha256",
    }
    if set(document) != expected_keys:
        raise ValueError("RQ3 source-selection keys are invalid")
    payload = {key: value for key, value in document.items() if key != "sha256"}
    logical_sha256 = _canonical_sha256(payload)
    if (
        type(document["schema_version"]) is not int
        or document["schema_version"] != 1
        or document["kind"] != "g3_rq2_content_selection_for_rq3"
        or document["protocol_sha256"] != APPROVED_PROTOCOL_SHA256
        or document["selected_family_id"] != "rq2_content_concat"
        or document["selection_resolved"] is not True
        or document["sha256"] != logical_sha256
        or logical_sha256 != expected_selection_sha256
    ):
        raise ValueError(
            "RQ3 source selection is not the bound resolved content result"
        )
    width = document["selected_history_hidden_dim"]
    if type(width) is not int or width < 1:
        raise ValueError("RQ3 selected RQ2 content width is invalid")
    feature_identity = _authenticate_feature_manifest(
        root, document["feature_manifest"]
    )
    rows = document["rows"]
    if not isinstance(rows, list) or not rows or len(rows) > 9:
        raise ValueError("RQ3 source selection must contain one to nine reusable rows")
    caches: dict[tuple[str, str], dict[str, object]] = {}
    approved_source_ids, approved_evidence_references = _approved_reusable_sources(
        root,
        document["source_evidence"],
        width=width,
        caches=caches,
        expected_final_rq2_evidence_sha256=expected_final_rq2_evidence_sha256,
    )
    coordinates = tuple(
        _authenticate_coordinate(
            root,
            row,
            width=width,
            caches=caches,
            feature_identity=feature_identity,
        )
        for row in rows
    )
    source_ids = [coordinate.source_id for coordinate in coordinates]
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("RQ3 source selection contains duplicate RQ2 rows")
    if set(source_ids) != approved_source_ids or any(
        _reference_identity(row["source_evidence"]) not in approved_evidence_references
        for row in rows
    ):
        raise ValueError(
            "RQ3 source selection is not the complete approved reusable RQ2 set"
        )
    return coordinates, width, feature_identity


def _authenticate_coordinate(
    root: Path,
    value: object,
    *,
    width: int,
    caches: dict[tuple[str, str], dict[str, object]],
    feature_identity: _FeatureIdentity,
) -> AuthenticatedRq2Coordinate:
    expected_keys = {
        "source_id",
        "source_ledger_row_id",
        "source_ledger",
        "source_evidence",
        "run_name",
        "family_id",
        "history_hidden_dim",
        "embedding_learning_rate",
        "deep_learning_rate",
        "horizon_epochs",
        "artifacts",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise ValueError("RQ3 source row schema is invalid")
    if (
        value["family_id"] != "rq2_content_concat"
        or value["history_hidden_dim"] != width
        or value["source_id"] != value["source_ledger_row_id"]
    ):
        raise ValueError("RQ3 source row is not from the selected RQ2 content family")
    source_id = value["source_id"]
    run_name = value["run_name"]
    if (
        not isinstance(source_id, str)
        or not source_id
        or not isinstance(run_name, str)
        or not run_name
    ):
        raise ValueError("RQ3 source row identity is invalid")
    _validate_coordinate_types(value)
    ledger, ledger_reference = _load_reference(
        root, value["source_ledger"], caches=caches
    )
    evidence, evidence_reference = _load_reference(
        root, value["source_evidence"], caches=caches
    )
    if ledger.get("protocol_sha256") != APPROVED_PROTOCOL_SHA256:
        raise ValueError("RQ3 source ledger uses another protocol")
    _validate_ledger_feature_binding(
        root,
        ledger,
        expected=feature_identity,
        visited={str(ledger_reference["path"])},
    )
    ledger_rows = ledger.get("rows")
    ledger_row = _unique_row(ledger_rows, "id", source_id, "source ledger")
    _validate_ledger_row(value, ledger_row, width=width)
    if evidence.get("protocol_sha256") != APPROVED_PROTOCOL_SHA256:
        raise ValueError("RQ3 source evidence uses another protocol")
    evidence_row = _unique_row(
        evidence.get("tuning_ledger"), "row_id", source_id, "source evidence"
    )
    _validate_evidence_row(value, evidence_row, width=width)
    artifacts = _validate_artifacts(root, value["artifacts"])
    evidence_artifacts = evidence_row.get("artifacts")
    if not isinstance(evidence_artifacts, dict):
        raise ValueError("RQ3 source evidence has no artifact bindings")
    for name, reference in artifacts.items():
        source_reference = evidence_artifacts.get(name)
        if not isinstance(source_reference, dict) or any(
            source_reference.get(key) != reference[key]
            for key in ("path", "sha256", "size_bytes")
        ):
            raise ValueError("RQ3 selected row artifacts differ from source evidence")
    diagnostics_identity = _validate_runtime_artifacts(
        root,
        artifacts,
        ledger_row=ledger_row,
        ledger_reference=ledger_reference,
        width=width,
        feature_identity=feature_identity,
    )
    return AuthenticatedRq2Coordinate(
        source_id=source_id,
        run_name=run_name,
        history_hidden_dim=width,
        embedding_learning_rate=float(value["embedding_learning_rate"]),
        deep_learning_rate=float(value["deep_learning_rate"]),
        horizon_epochs=int(value["horizon_epochs"]),
        source_ledger_path=str(ledger_reference["path"]),
        source_ledger_sha256=str(ledger_reference["logical_sha256"]),
        source_evidence_sha256=str(evidence_reference["logical_sha256"]),
        artifact_sha256=tuple(
            (name, str(reference["sha256"]))
            for name, reference in sorted(artifacts.items())
        ),
        training_count_sha256=str(feature_identity.training_count_reference["sha256"]),
        slice_membership_sha256=str(
            feature_identity.slice_membership_reference["sha256"]
        ),
        diagnostics_schema_version=diagnostics_identity.schema_version,
        diagnostics_epoch_count=diagnostics_identity.epoch_count,
    )


def _approved_reusable_sources(
    root: Path,
    value: object,
    *,
    width: int,
    caches: dict[tuple[str, str], dict[str, object]],
    expected_final_rq2_evidence_sha256: str | None,
) -> tuple[set[str], set[tuple[object, ...]]]:
    if not isinstance(value, list) or not value:
        raise ValueError("RQ3 approved reusable source evidence is missing")
    references: set[tuple[object, ...]] = set()
    source_ids: set[str] = set()
    for raw_reference in value:
        evidence, reference = _load_reference(root, raw_reference, caches=caches)
        identity = _reference_identity(reference)
        if identity in references:
            raise ValueError("RQ3 approved reusable source evidence is duplicated")
        references.add(identity)
        if evidence.get("protocol_sha256") != APPROVED_PROTOCOL_SHA256:
            raise ValueError("RQ3 approved reusable evidence uses another protocol")
        validator = _RQ2_REUSE_SOURCE_VALIDATORS.get(evidence.get("kind"))
        if validator is None:
            raise ValueError("RQ3 approved reusable evidence kind is unsupported")
        validator(
            root,
            evidence,
            width=width,
            caches=caches,
            expected_final_rq2_evidence_sha256=expected_final_rq2_evidence_sha256,
        )
        rows = evidence.get("tuning_ledger")
        if not isinstance(rows, list):
            raise ValueError("RQ3 approved reusable evidence has no tuning ledger")
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("RQ3 approved reusable evidence row is invalid")
            if (
                row.get("family_id") != "rq2_content_concat"
                or row.get("capacity") != width
                or row.get("selection_resolved") is not True
            ):
                continue
            source_id = row.get("row_id")
            if not isinstance(source_id, str) or not source_id:
                raise ValueError("RQ3 approved reusable source ID is invalid")
            if source_id in source_ids:
                raise ValueError("RQ3 approved reusable source ID is duplicated")
            source_ids.add(source_id)
    if not source_ids or len(source_ids) > 9:
        raise ValueError("RQ3 approved reusable set must contain one to nine rows")
    return source_ids, references


def _validate_rq2_reuse_bridge(
    root: Path,
    bridge: Mapping[str, object],
    *,
    width: int,
    caches: dict[tuple[str, str], dict[str, object]],
    expected_final_rq2_evidence_sha256: str | None,
) -> None:
    expected_keys = {
        "schema_version",
        "kind",
        "protocol_sha256",
        "selection_resolved",
        "selected_row_id",
        "final_rq2_evidence",
        "all_tuning_ledger",
        "tuning_ledger",
        "sha256",
    }
    if (
        set(bridge) != expected_keys
        or bridge.get("schema_version") != 1
        or bridge.get("selection_resolved") is not True
    ):
        raise ValueError("RQ3 RQ2 reuse bridge schema is invalid")
    final_evidence, final_reference = _load_reference(
        root,
        bridge["final_rq2_evidence"],
        caches=caches,
    )
    if (
        not isinstance(expected_final_rq2_evidence_sha256, str)
        or not _valid_sha256(expected_final_rq2_evidence_sha256)
        or final_reference["logical_sha256"]
        != expected_final_rq2_evidence_sha256
    ):
        raise ValueError("RQ3 RQ2 reuse bridge lacks its frozen final evidence binding")
    if final_evidence.get("kind") != (
        "g3_rq2_content_width32_horizon40_deep_lr_boundary_evidence"
    ) or (
        final_evidence.get("schema_version") != 1
        or final_evidence.get("protocol_sha256") != APPROVED_PROTOCOL_SHA256
    ):
        raise ValueError("RQ3 RQ2 reuse bridge has no supported source adapter")
    final_selection = final_evidence.get("final_content_selection")
    rq3_inputs = final_evidence.get("rq3_inputs")
    all_tuning = final_evidence.get("all_tuning_ledger")
    if (
        not isinstance(final_selection, dict)
        or final_selection.get("status") != "resolved"
        or final_selection.get("provisional_selected") is not None
        or not isinstance(final_selection.get("selected"), dict)
        or final_selection["selected"].get("capacity") != width
        or not isinstance(rq3_inputs, dict)
        or rq3_inputs.get("selected_content_input") != final_selection["selected"]
        or not isinstance(all_tuning, list)
        or any(not isinstance(row, dict) for row in all_tuning)
    ):
        raise ValueError("RQ3 RQ2 reuse bridge source selection is unresolved")
    reusable = rq3_inputs.get("reusable_width_32_content_rows")
    expected_reusable = [
        row
        for row in all_tuning
        if isinstance(row, dict)
        and row.get("family_id") == "rq2_content_concat"
        and row.get("capacity") == width
    ]
    expected_bridge_rows = [
        dict(row) | {"selection_resolved": True} for row in expected_reusable
    ]
    if (
        reusable != expected_reusable
        or bridge.get("selected_row_id") != final_selection["selected"].get("row_id")
        or bridge.get("all_tuning_ledger") != all_tuning
        or bridge.get("tuning_ledger") != expected_bridge_rows
    ):
        raise ValueError("RQ3 RQ2 reuse bridge differs from final RQ2 evidence")


_RQ2_REUSE_SOURCE_VALIDATORS = {
    "g3_rq2_content_rq3_reuse_bridge": _validate_rq2_reuse_bridge,
}


def _reference_identity(value: object) -> tuple[object, ...]:
    if not isinstance(value, dict):
        raise ValueError("RQ3 source evidence reference is invalid")
    return tuple(
        value.get(key) for key in ("path", "sha256", "size_bytes", "logical_sha256")
    )


def _validate_ledger_row(
    selected: Mapping[str, object],
    row: Mapping[str, object],
    *,
    width: int,
) -> None:
    if row.get("dataset") != _RQ2_DATASET:
        raise ValueError("RQ3 source dataset differs from the approved RQ2 dataset")
    representation = row.get("representation")
    training = row.get("training")
    if not isinstance(representation, dict) or not isinstance(training, dict):
        raise ValueError("RQ3 source ledger row lacks representation or training")
    expected = {
        "id": "rq2_content_concat",
        "history": "learned_item_id_plus_frozen_content",
        "catalog": "learned_item_id",
        "history_hidden_dim": width,
        "content_trainable": False,
        "content_width": 128,
        "separate_history_catalog_tables": True,
    }
    if (
        row.get("family_id") != "rq2_content_concat"
        or row.get("run_name") != selected["run_name"]
        or any(representation.get(key) != value for key, value in expected.items())
        or training.get("batch_size") != 512
        or training.get("seed") != 42
        or training.get("embedding_learning_rate")
        != selected["embedding_learning_rate"]
        or training.get("deep_learning_rate") != selected["deep_learning_rate"]
        or training.get("horizon_epochs") != selected["horizon_epochs"]
    ):
        raise ValueError("RQ3 selected coordinate differs from its RQ2 ledger row")


def _validate_evidence_row(
    selected: Mapping[str, object],
    row: Mapping[str, object],
    *,
    width: int,
) -> None:
    expected = {
        "run_name": selected["run_name"],
        "family_id": "rq2_content_concat",
        "capacity": width,
        "embedding_learning_rate": selected["embedding_learning_rate"],
        "deep_learning_rate": selected["deep_learning_rate"],
        "horizon_epochs": selected["horizon_epochs"],
        "selection_resolved": True,
    }
    if any(row.get(key) != value for key, value in expected.items()):
        raise ValueError("RQ3 selected coordinate differs from its RQ2 evidence row")


def _validate_coordinate_types(value: Mapping[str, object]) -> None:
    for name in ("embedding_learning_rate", "deep_learning_rate"):
        number = value[name]
        if (
            type(number) not in {int, float}
            or not math.isfinite(float(number))
            or float(number) <= 0
        ):
            raise ValueError(f"RQ3 selected {name} is invalid")
    horizon = value["horizon_epochs"]
    if type(horizon) is not int or horizon < 1:
        raise ValueError("RQ3 selected horizon is invalid")


def _validate_runtime_artifacts(
    root: Path,
    artifacts: Mapping[str, Mapping[str, object]],
    *,
    ledger_row: Mapping[str, object],
    ledger_reference: Mapping[str, object],
    width: int,
    feature_identity: _FeatureIdentity,
) -> _DiagnosticsIdentity:
    contract = _load_json(root / str(artifacts["job_contract"]["path"]))
    ledger_path = (root / str(ledger_reference["path"])).resolve()
    if (
        contract.get("row_id") != ledger_row["id"]
        or contract.get("job") != ledger_row
        or contract.get("ledger_sha256") != ledger_reference["logical_sha256"]
        or Path(str(contract.get("ledger_path"))).resolve() != ledger_path
    ):
        raise ValueError("RQ3 source job contract is not bound to its ledger row")
    metadata = _load_json(root / str(artifacts["training_metadata"]["path"]))
    training = ledger_row["training"]
    if not isinstance(training, dict):
        raise ValueError("RQ3 source ledger training is invalid")
    expected = {
        "batch_size": 512,
        "seed": 42,
        "embedding_learning_rate": training["embedding_learning_rate"],
        "deep_learning_rate": training["deep_learning_rate"],
        "lr_schedule_horizon_epochs": training["horizon_epochs"],
        "epochs_trained": training["horizon_epochs"],
        "stopped_epoch": training["horizon_epochs"],
        "lr_horizon_complete": True,
        "selection_resolved": True,
        "g3_dataset_size": "native-50m",
        "g3_protocol_sha256": APPROVED_PROTOCOL_SHA256,
        "training_semantics_revision": 2,
    }
    representation = {
        "history_representation": "id_content",
        "catalog_representation": "learned_id",
        "history_hidden_dim": width,
        "content_gate": "fixed",
        "gate_hidden_dim": None,
        "metadata": [],
        "metadata_dim": None,
        "extra_item_id_dim": None,
    }
    if any(metadata.get(key) != value for key, value in expected.items()) or (
        metadata.get("g3_representation") != representation
    ):
        raise ValueError("RQ3 source runtime metadata is incompatible")
    invariants = metadata.get("transfer_invariants")
    if not isinstance(invariants, dict):
        raise ValueError("RQ3 source runtime dataset invariants are missing")
    if any(invariants.get(key) != value for key, value in _RQ2_DATA_INVARIANTS.items()):
        raise ValueError("RQ3 source runtime dataset/evaluation invariants changed")
    if any(invariants.get(key) != value for key, value in _RQ2_LOSS_INVARIANTS.items()):
        raise ValueError("RQ3 source runtime loss/negative invariants changed")
    expected_schedule = {
        **_RQ2_SCHEDULE_INVARIANTS,
        "lr_schedule_horizon_epochs": training["horizon_epochs"],
    }
    if any(invariants.get(key) != value for key, value in expected_schedule.items()):
        raise ValueError("RQ3 source runtime schedule invariants changed")
    metrics = _load_json(root / str(artifacts["final_metrics"]["path"]))
    if not metrics or any(
        type(value) not in {int, float} or not math.isfinite(float(value))
        for value in metrics.values()
    ):
        raise ValueError("RQ3 source final metrics are invalid")
    diagnostics = _load_json(root / str(artifacts["training_diagnostics"]["path"]))
    return _validate_training_diagnostics(
        diagnostics,
        feature_identity=feature_identity,
        horizon_epochs=int(training["horizon_epochs"]),
        catalog_representation="learned_id",
    )


def _authenticate_feature_manifest(root: Path, value: object) -> _FeatureIdentity:
    reference = _validate_file_reference(root, value, logical=True)
    expected_reference = {
        "path": NATIVE50_FEATURE_MANIFEST_PATH,
        "sha256": NATIVE50_FEATURE_MANIFEST_FILE_SHA256,
        "logical_sha256": NATIVE50_FEATURE_MANIFEST_SHA256,
    }
    if any(
        reference.get(key) != expected for key, expected in expected_reference.items()
    ):
        raise ValueError("RQ3 feature manifest is not the approved native-50M manifest")
    manifest_path = root / str(reference["path"])
    manifest = load_artifact_manifest(manifest_path)
    if manifest.sha256 != NATIVE50_FEATURE_MANIFEST_SHA256:
        raise ValueError("RQ3 feature manifest logical hash changed")
    validate_feature_manifest(root=root, manifest=manifest, validate_files=True)
    bindings = {binding.role: binding for binding in manifest.artifacts}
    feature_binding = bindings["item_features"]
    feature_data = load_feature_data(root / feature_binding.path)
    metadata = manifest.metadata
    if (
        feature_data.training_counts.numel() - 1 != metadata["num_items"]
        or int(feature_data.training_counts.sum()) != metadata["training_rows"]
        or len(feature_data.training_history_lengths) != metadata["training_users"]
        or feature_data.artist_vocab_size != metadata["artist_vocab_size"]
        or feature_data.album_vocab_size != metadata["album_vocab_size"]
    ):
        raise ValueError("RQ3 feature counts differ from the native-50M manifest")
    frequency = build_frequency_identity(feature_data.training_counts)
    return _FeatureIdentity(
        manifest_path=str(reference["path"]),
        manifest_sha256=str(reference["logical_sha256"]),
        manifest_file_sha256=str(reference["sha256"]),
        data_path=feature_binding.path,
        data_sha256=feature_binding.sha256,
        frequency_terciles=dict(frequency["frequency_terciles"]),
        training_count_reference=dict(frequency["training_count_reference"]),
        slice_membership_reference=dict(frequency["slice_membership_reference"]),
    )


def _validate_ledger_feature_binding(
    root: Path,
    ledger: Mapping[str, object],
    *,
    expected: _FeatureIdentity,
    visited: set[str],
) -> None:
    inputs = ledger.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("RQ3 source ledger has no authenticated feature inputs")
    found_feature = False
    child_ledgers: list[tuple[str, dict[str, object]]] = []
    for value in inputs.values():
        if not isinstance(value, dict):
            continue
        if value.get("kind") == "native50m_features":
            found_feature = True
            if value != {
                "kind": "native50m_features",
                "path": expected.manifest_path,
                "sha256": expected.manifest_sha256,
            }:
                raise ValueError("RQ3 source ledger binds another feature manifest")
            continue
        kind = value.get("kind")
        path = value.get("path")
        if not isinstance(kind, str) or not isinstance(path, str):
            continue
        if "ledger" not in kind and "protocol/ledgers/" not in path:
            continue
        child_ledgers.append((path, _load_ledger_input(root, value)))
    if found_feature:
        return
    for path, child in child_ledgers:
        if path in visited:
            continue
        if child.get("protocol_sha256") != APPROVED_PROTOCOL_SHA256:
            raise ValueError("RQ3 source ledger ancestry uses another protocol")
        try:
            _validate_ledger_feature_binding(
                root,
                child,
                expected=expected,
                visited={*visited, path},
            )
        except ValueError as error:
            if "no authenticated feature inputs" not in str(error):
                raise
        else:
            return
    raise ValueError("RQ3 source ledger has no authenticated feature inputs")


def _load_ledger_input(root: Path, value: Mapping[str, object]) -> dict[str, object]:
    if set(value) != {"kind", "path", "sha256"}:
        raise ValueError("RQ3 source ledger input schema is invalid")
    expected_sha256 = value["sha256"]
    if not isinstance(expected_sha256, str):
        raise ValueError("RQ3 source ledger input hash is invalid")
    path = _resolve_input_path(root, root / str(value["path"]))
    document = _load_json(path)
    payload = {key: item for key, item in document.items() if key != "sha256"}
    logical_sha256 = _canonical_sha256(payload)
    if document.get("sha256") != logical_sha256 or logical_sha256 != expected_sha256:
        raise ValueError("RQ3 source ledger ancestry logical hash changed")
    if not isinstance(document.get("rows"), list):
        raise ValueError("RQ3 source ledger ancestry is not a ledger")
    return document


def _validate_training_diagnostics(
    document: Mapping[str, object],
    *,
    feature_identity: _FeatureIdentity,
    horizon_epochs: int,
    catalog_representation: str,
    diagnostics_contract: _CatalogDiagnosticsContract | None = None,
) -> _DiagnosticsIdentity:
    if diagnostics_contract is None:
        try:
            contract = _CATALOG_DIAGNOSTICS[catalog_representation]
        except KeyError as error:
            raise ValueError("RQ3 diagnostics catalog representation is invalid") from error
    else:
        contract = diagnostics_contract
    schema_version = document.get("schema_version")
    common = {
        "schema_version",
        "frequency_terciles",
        "content_drift_reference",
        "epochs",
    }
    if schema_version == 1:
        if set(document) != common:
            raise ValueError("RQ3 reused diagnostics schema is invalid")
    elif schema_version == 2:
        if set(document) != common | {
            "training_count_reference",
            "slice_membership_reference",
        }:
            raise ValueError("RQ3 fresh diagnostics schema is invalid")
        if (
            document.get("training_count_reference")
            != feature_identity.training_count_reference
        ):
            raise ValueError("RQ3 diagnostics training-count reference changed")
        if (
            document.get("slice_membership_reference")
            != feature_identity.slice_membership_reference
        ):
            raise ValueError("RQ3 diagnostics slice-membership reference changed")
    else:
        raise ValueError("RQ3 diagnostics schema is unsupported")
    if document.get("frequency_terciles") != feature_identity.frequency_terciles:
        raise ValueError("RQ3 diagnostics use different training-frequency slices")
    epochs = document.get("epochs")
    if not isinstance(epochs, list):
        raise ValueError("RQ3 diagnostics payload is invalid")
    _validate_content_drift_reference(
        document.get("content_drift_reference"),
        content_available=contract.content_trainable is not None,
        catalog_count=int(feature_identity.frequency_terciles["num_catalog_items"]),
    )
    if len(epochs) != horizon_epochs:
        raise ValueError("RQ3 diagnostics are not horizon-complete")
    common_epoch_keys = {
        "epoch",
        "training",
        "component_gradient_norms",
        "catalog_representation_norm",
        "pretrained_content",
    }
    scopes = {"global", "tail", "mid", "head"}
    for expected_epoch, entry in enumerate(epochs):
        if not isinstance(entry, dict):
            raise ValueError("RQ3 diagnostics lack the common epoch fields")
        epoch_keys = set(entry)
        catalog_table_available = "catalog_table_gradient_norms" in epoch_keys
        if (
            not common_epoch_keys <= epoch_keys
            or epoch_keys - common_epoch_keys
            != ({"catalog_table_gradient_norms"} if catalog_table_available else set())
            or (schema_version == 2 and not catalog_table_available)
        ):
            raise ValueError("RQ3 diagnostics lack the common epoch fields")
        if entry.get("epoch") != expected_epoch:
            raise ValueError("RQ3 diagnostics epochs are not complete and ordered")
        training = entry.get("training")
        catalog = entry.get("catalog_representation_norm")
        if (
            not isinstance(training, dict)
            or set(training) != scopes
            or not isinstance(catalog, dict)
            or set(catalog) != scopes
        ):
            raise ValueError("RQ3 diagnostics common epoch payload is invalid")
        _validate_training_scopes(training)
        _validate_catalog_scopes(
            catalog,
            frequency_terciles=feature_identity.frequency_terciles,
        )
        _validate_component_gradients(
            entry.get("component_gradient_norms"),
            expected=contract.components,
        )
        _validate_pretrained_content(
            entry.get("pretrained_content"),
            frequency_terciles=feature_identity.frequency_terciles,
            expected_trainable=contract.content_trainable,
        )
        if catalog_table_available:
            _validate_catalog_table_gradients(
                entry.get("catalog_table_gradient_norms"),
                detailed_schema_required=schema_version == 2,
                expected_parameters=contract.table_parameters,
            )
    return _DiagnosticsIdentity(schema_version=schema_version, epoch_count=len(epochs))


_DISTRIBUTION_KEYS = {
    "count",
    "nonfinite_count",
    "mean",
    "standard_deviation",
    "minimum",
    "maximum",
}


def _validate_distribution(
    value: object,
    *,
    expected_count: int | None = None,
    allow_empty: bool = False,
) -> None:
    if not isinstance(value, dict) or set(value) != _DISTRIBUTION_KEYS:
        raise ValueError("RQ3 diagnostics distribution schema is invalid")
    count = value["count"]
    nonfinite_count = value["nonfinite_count"]
    if (
        type(count) is not int
        or count < 0
        or type(nonfinite_count) is not int
        or nonfinite_count != 0
        or (expected_count is not None and count != expected_count)
        or (not allow_empty and count == 0)
    ):
        raise ValueError("RQ3 diagnostics distribution counts are invalid")
    statistics = tuple(
        value[key] for key in ("mean", "standard_deviation", "minimum", "maximum")
    )
    if count == 0:
        if any(item is not None for item in statistics):
            raise ValueError("RQ3 diagnostics empty distribution is invalid")
        return
    if any(
        type(item) not in {int, float} or not math.isfinite(float(item))
        for item in statistics
    ):
        raise ValueError("RQ3 diagnostics distribution values are invalid")
    mean, standard_deviation, minimum, maximum = map(float, statistics)
    if standard_deviation < 0 or minimum > mean or mean > maximum:
        raise ValueError("RQ3 diagnostics distribution bounds are invalid")


def _validate_training_scopes(value: Mapping[str, object]) -> None:
    expected_keys = {
        "num_examples",
        "query_norm",
        "positive_logit",
        "negative_logit",
    }
    for scope in ("global", "tail", "mid", "head"):
        payload = value[scope]
        if not isinstance(payload, dict) or set(payload) != expected_keys:
            raise ValueError("RQ3 diagnostics training-scope schema is invalid")
        num_examples = payload["num_examples"]
        if type(num_examples) is not int or num_examples < 1:
            raise ValueError("RQ3 diagnostics training scope is empty")
        _validate_distribution(payload["query_norm"], expected_count=num_examples)
        _validate_distribution(payload["positive_logit"], expected_count=num_examples)
        _validate_distribution(payload["negative_logit"])


def _frequency_scope_counts(
    frequency_terciles: Mapping[str, object],
) -> dict[str, int]:
    catalog_count = frequency_terciles.get("num_catalog_items")
    slices = frequency_terciles.get("slices")
    if type(catalog_count) is not int or not isinstance(slices, dict):
        raise ValueError("RQ3 diagnostics frequency manifest is invalid")
    result = {"global": catalog_count}
    for scope in ("tail", "mid", "head"):
        payload = slices.get(scope)
        if not isinstance(payload, dict) or type(payload.get("num_items")) is not int:
            raise ValueError("RQ3 diagnostics frequency-slice manifest is invalid")
        result[scope] = payload["num_items"]
    return result


def _validate_catalog_scopes(
    value: Mapping[str, object], *, frequency_terciles: Mapping[str, object]
) -> None:
    for scope, count in _frequency_scope_counts(frequency_terciles).items():
        _validate_distribution(value[scope], expected_count=count)


def _validate_content_drift_reference(
    value: object, *, content_available: bool, catalog_count: int
) -> None:
    if not content_available:
        if value != {"available": False}:
            raise ValueError("RQ3 diagnostics content-drift reference is invalid")
        return
    if (
        not isinstance(value, dict)
        or set(value) != {"available", "shape", "dtype", "sha256"}
        or value["available"] is not True
        or value["shape"] != [catalog_count, 128]
        or value["dtype"] != "torch.float32"
        or not isinstance(value["sha256"], str)
        or not _valid_sha256(value["sha256"])
    ):
        raise ValueError("RQ3 diagnostics content-drift reference is invalid")


def _validate_component_gradients(
    value: object, *, expected: frozenset[str]
) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("RQ3 diagnostics component gradient set is invalid")
    for distribution in value.values():
        _validate_distribution(distribution)


def _validate_pretrained_content(
    value: object,
    *,
    frequency_terciles: Mapping[str, object],
    expected_trainable: bool | None,
) -> None:
    if expected_trainable is None:
        if value != {"available": False}:
            raise ValueError("RQ3 diagnostics learned-ID catalog content is invalid")
        return
    if (
        not isinstance(value, dict)
        or set(value)
        != {"available", "trainable", "drift_l2", "cosine_to_initial"}
        or value["available"] is not True
    ):
        raise ValueError("RQ3 diagnostics pretrained-content schema is invalid")
    if value["trainable"] is not expected_trainable:
        raise ValueError("RQ3 diagnostics pretrained-content trainability is invalid")
    expected_scopes = {"global", "tail", "mid", "head"}
    for name in ("drift_l2", "cosine_to_initial"):
        scoped = value[name]
        if not isinstance(scoped, dict) or set(scoped) != expected_scopes:
            raise ValueError("RQ3 diagnostics pretrained-content scopes are invalid")
        _validate_catalog_scopes(
            scoped,
            frequency_terciles=frequency_terciles,
        )


def _validate_catalog_table_gradients(
    value: object,
    *,
    detailed_schema_required: bool,
    expected_parameters: frozenset[str],
) -> None:
    if not isinstance(value, dict) or set(value) != expected_parameters:
        raise ValueError("RQ3 diagnostics catalog-table parameter set is invalid")
    scopes = {"global", "tail", "mid", "head"}
    statistics = {
        "all_row_exposure_weighted_norm",
        "conditional_on_active_row_norm",
        "active_row_count",
        "active_row_fraction",
    }
    for scoped in value.values():
        if not isinstance(scoped, dict) or set(scoped) != scopes:
            raise ValueError("RQ3 diagnostics catalog-table scopes are invalid")
        for scope in scopes:
            payload = scoped[scope]
            if not detailed_schema_required:
                _validate_distribution(payload)
                continue
            if not isinstance(payload, dict) or set(payload) != statistics:
                raise ValueError("RQ3 diagnostics catalog-table schema is invalid")
            for name in statistics:
                _validate_distribution(
                    payload[name],
                    allow_empty=name == "conditional_on_active_row_norm",
                )


def resolve_rq3_feature_data(*, root: Path, surface: Rq3OutputSurface) -> Path:
    root = root.resolve(strict=True)
    manifest_path = _resolve_input_path(root, root / surface.feature_manifest_path)
    reference = {
        "path": surface.feature_manifest_path,
        "sha256": surface.feature_manifest_file_sha256,
        "size_bytes": manifest_path.stat().st_size,
        "logical_sha256": surface.feature_manifest_sha256,
    }
    identity = _authenticate_feature_manifest(root, reference)
    expected = (
        surface.feature_data_path,
        surface.feature_data_sha256,
        surface.frequency_terciles,
        surface.training_count_reference,
        surface.slice_membership_reference,
    )
    actual = (
        identity.data_path,
        identity.data_sha256,
        identity.frequency_terciles,
        identity.training_count_reference,
        identity.slice_membership_reference,
    )
    if actual != expected:
        raise ValueError("RQ3 compiled feature/count identity changed before launch")
    return _resolve_input_path(root, root / identity.data_path)


def _validate_artifacts(root: Path, value: object) -> dict[str, dict[str, object]]:
    expected = {
        "job_contract",
        "training_metadata",
        "final_metrics",
        "training_diagnostics",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("RQ3 source artifacts are incomplete")
    result = {}
    for name, reference in value.items():
        result[name] = _validate_file_reference(root, reference, logical=False)
    return result


def _load_reference(
    root: Path,
    value: object,
    *,
    caches: dict[tuple[str, str], dict[str, object]],
) -> tuple[dict[str, object], dict[str, object]]:
    reference = _validate_file_reference(root, value, logical=True)
    key = (str(reference["path"]), str(reference["logical_sha256"]))
    document = caches.get(key)
    if document is None:
        document = _load_json(root / str(reference["path"]))
        payload = {key: value for key, value in document.items() if key != "sha256"}
        logical_sha256 = _canonical_sha256(payload)
        if (
            document.get("sha256") != logical_sha256
            or logical_sha256 != reference["logical_sha256"]
        ):
            raise ValueError("RQ3 source document logical hash changed")
        caches[key] = document
    return document, reference


def _validate_file_reference(
    root: Path, value: object, *, logical: bool
) -> dict[str, object]:
    keys = {"path", "sha256", "size_bytes"}
    if logical:
        keys.add("logical_sha256")
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError("RQ3 source file reference schema is invalid")
    if (
        not isinstance(value["path"], str)
        or not isinstance(value["sha256"], str)
        or type(value["size_bytes"]) is not int
        or value["size_bytes"] < 0
        or (logical and not isinstance(value["logical_sha256"], str))
    ):
        raise ValueError("RQ3 source file reference types are invalid")
    relative = Path(value["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("RQ3 source file reference must be project-relative")
    path = _resolve_input_path(root, root / relative)
    if (
        path.stat().st_size != value["size_bytes"]
        or _file_sha256(path) != value["sha256"]
    ):
        raise ValueError("RQ3 source file reference differs from disk")
    return dict(value)


def _unique_row(
    rows: object, key: str, expected: object, label: str
) -> dict[str, object]:
    if not isinstance(rows, list):
        raise ValueError(f"RQ3 {label} rows are invalid")
    matches = [
        row for row in rows if isinstance(row, dict) and row.get(key) == expected
    ]
    if len(matches) != 1:
        raise ValueError(f"RQ3 selected row is not unique in {label}")
    return matches[0]


def _resolve_input_path(root: Path, path: Path) -> Path:
    path = path if path.is_absolute() else root / path
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"RQ3 bound input is not a regular file: {path}")
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("RQ3 bound input escapes the project root")
    return resolved


def _load_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(
            path.read_text(),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load RQ3 bound JSON {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"RQ3 bound JSON must be an object: {path}")
    return value


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _valid_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON number {value!r}")
