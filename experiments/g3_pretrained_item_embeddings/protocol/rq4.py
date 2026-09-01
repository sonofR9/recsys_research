from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.g3_pretrained_item_embeddings.configs.model import (
    RQ3_CATALOG_REPRESENTATIONS,
)
from experiments.g3_pretrained_item_embeddings.data import (
    LoadedFeatureData,
    load_feature_data,
)
from experiments.g3_pretrained_item_embeddings.diagnostics import (
    build_frequency_identity,
)

from .constants import APPROVED_PROTOCOL_SHA256
from .manifests import load_artifact_manifest, validate_feature_manifest
from .rq3 import _FeatureIdentity, _validate_training_diagnostics
from .search import (
    APPROVED_FAMILY_SPECS,
    FamilySpec,
    SearchCoordinate,
    TransferredHorizonRate,
    compile_capacity_first_stage,
    compile_capacity_horizon_followup,
    compile_rq4_extra_id_control,
)


RQ4_METADATA_FAMILIES = (
    "rq4_artist",
    "rq4_album",
    "rq4_artist_album",
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

_METADATA_BY_FAMILY = {
    "rq4_artist": ("artist",),
    "rq4_album": ("album",),
    "rq4_artist_album": ("artist", "album"),
}


@dataclass(frozen=True)
class Rq4Predecessor:
    rq2_selection_path: str
    rq2_selection_sha256: str
    rq2_selection_file_sha256: str
    rq2_selection_size_bytes: int
    rq3_selection_path: str
    rq3_selection_sha256: str
    rq3_selection_file_sha256: str
    rq3_selection_size_bytes: int
    rq3_artifacts: tuple[tuple[str, str, str, int], ...]
    rq3_row_id: str
    rq3_family_id: str
    history_hidden_dim: int
    catalog_representation: str


@dataclass(frozen=True)
class Rq4MetadataIdentity:
    manifest_path: str
    manifest_sha256: str
    manifest_file_sha256: str
    feature_data_path: str
    feature_data_sha256: str
    num_items: int
    artist_vocab_size: int
    album_vocab_size: int
    frequency_terciles: dict[str, object]
    training_count_reference: dict[str, object]
    slice_membership_reference: dict[str, object]
    training_count_sha256: str
    artist_membership_sha256: str
    album_membership_sha256: str


@dataclass(frozen=True)
class Rq4MetadataRow:
    id: str
    family_id: str
    run_name: str
    batch_size: int
    seed: int
    embedding_learning_rate: float
    deep_learning_rate: float
    horizon_epochs: int
    metadata: tuple[str, ...]
    metadata_dim: int
    reused_from: str | None


@dataclass(frozen=True)
class Rq4CapacitySurface:
    predecessor: Rq4Predecessor
    metadata_identity: Rq4MetadataIdentity
    rows_by_family: dict[str, tuple[Rq4MetadataRow, ...]]
    extra_id_rows: tuple[object, ...] = ()


@dataclass(frozen=True)
class Rq4HorizonFollowup:
    predecessor: Rq4Predecessor
    metadata_identity: Rq4MetadataIdentity
    selected_capacities: dict[str, int]
    selected_capacity_inputs: tuple[tuple[str, int], ...]
    transferred_horizon_rate_inputs: tuple[
        tuple[str, tuple[TransferredHorizonRate, ...]], ...
    ]
    rows_by_family: dict[str, tuple[Rq4MetadataRow, ...]]


@dataclass(frozen=True)
class Rq4ExtraIdRow:
    id: str
    family_id: str
    run_name: str
    batch_size: int
    seed: int
    embedding_learning_rate: float
    deep_learning_rate: float
    horizon_epochs: int
    extra_item_id_dim: int
    matched_metadata_family: str
    matched_metadata_dim: int
    parameter_mismatch_fraction: float


@dataclass(frozen=True)
class Rq4ExtraIdSurface:
    predecessor: Rq4Predecessor
    metadata_identity: Rq4MetadataIdentity
    winner_selection_path: str
    winner_selection_sha256: str
    winner_selection_file_sha256: str
    winner_selection_size_bytes: int
    capacity_surface: Rq4CapacitySurface
    horizon_followup: Rq4HorizonFollowup
    rows: tuple[Rq4ExtraIdRow, ...]


def compile_rq4_capacity_surface(
    *,
    root: Path,
    rq2_selection_path: Path,
    expected_rq2_selection_sha256: str,
    rq3_selection_path: Path,
    expected_rq3_selection_sha256: str,
    expected_rq3_row_id: str | None = None,
) -> Rq4CapacitySurface:
    root = root.resolve(strict=True)
    metadata_identity, manifest_reference = _authenticate_metadata(root)
    rq2_path, rq2 = _load_bound_document(
        root,
        rq2_selection_path,
        expected_sha256=expected_rq2_selection_sha256,
        label="RQ2 selection",
    )
    width = _authenticate_rq2_selection(
        rq2,
        expected_feature_manifest=manifest_reference,
    )
    rq3_path, rq3 = _load_bound_document(
        root,
        rq3_selection_path,
        expected_sha256=expected_rq3_selection_sha256,
        label="RQ3 selection",
    )
    predecessor = _authenticate_rq3_selection(
        root,
        rq3,
        rq2_path=rq2_path,
        rq2_sha256=expected_rq2_selection_sha256,
        rq3_path=rq3_path,
        rq3_sha256=expected_rq3_selection_sha256,
        history_hidden_dim=width,
        expected_feature_manifest=manifest_reference,
        feature_identity=metadata_identity,
        expected_row_id=expected_rq3_row_id,
    )
    specifications = _metadata_specifications()
    rows_by_family = {
        family_id: tuple(
            _metadata_row(
                coordinate,
                metadata=_METADATA_BY_FAMILY[family_id],
                phase="capacity",
            )
            for coordinate in compile_capacity_first_stage(specifications[family_id])
        )
        for family_id in RQ4_METADATA_FAMILIES
    }
    _validate_equal_budget(rows_by_family, expected_rows=9)
    return Rq4CapacitySurface(
        predecessor=predecessor,
        metadata_identity=metadata_identity,
        rows_by_family=rows_by_family,
    )


def compile_rq4_horizon_followup(
    capacity_surface: Rq4CapacitySurface,
    *,
    selected_capacities: Mapping[str, int],
    transferred_horizon_rates: Mapping[str, Sequence[TransferredHorizonRate]],
) -> Rq4HorizonFollowup:
    if set(selected_capacities) != set(RQ4_METADATA_FAMILIES):
        raise ValueError("RQ4 selected capacities must cover every metadata family")
    if set(transferred_horizon_rates) != set(RQ4_METADATA_FAMILIES):
        raise ValueError("RQ4 transferred rates must cover every metadata family")
    specifications = _metadata_specifications()
    expected_capacity_rows = {
        family_id: tuple(
            _metadata_row(
                coordinate,
                metadata=_METADATA_BY_FAMILY[family_id],
                phase="capacity",
            )
            for coordinate in compile_capacity_first_stage(specifications[family_id])
        )
        for family_id in RQ4_METADATA_FAMILIES
    }
    if capacity_surface.rows_by_family != expected_capacity_rows:
        raise ValueError("RQ4 capacity surface differs from the approved design")
    rows_by_family = {}
    normalized_capacities = {}
    normalized_rates = {}
    for family_id in RQ4_METADATA_FAMILIES:
        capacity = selected_capacities[family_id]
        if type(capacity) is not int:
            raise ValueError("RQ4 selected metadata width must be an integer")
        first_stage = compile_capacity_first_stage(specifications[family_id])
        rates = tuple(transferred_horizon_rates[family_id])
        coordinates = compile_capacity_horizon_followup(
            specifications[family_id],
            selected_capacity=capacity,
            transferred_horizon_rates=rates,
            first_stage=first_stage,
        )
        first_by_id = {
            coordinate.id: row
            for coordinate, row in zip(
                first_stage,
                capacity_surface.rows_by_family[family_id],
                strict=True,
            )
        }
        rows_by_family[family_id] = tuple(
            _metadata_row(
                coordinate,
                metadata=_METADATA_BY_FAMILY[family_id],
                phase="horizon",
                reused_run_name=(
                    first_by_id[coordinate.reused_from].run_name
                    if coordinate.reused_from is not None
                    else None
                ),
            )
            for coordinate in coordinates
        )
        normalized_capacities[family_id] = capacity
        normalized_rates[family_id] = rates
    _validate_equal_budget(
        rows_by_family,
        expected_rows=3,
        require_equal_capacities=False,
    )
    return Rq4HorizonFollowup(
        predecessor=capacity_surface.predecessor,
        metadata_identity=capacity_surface.metadata_identity,
        selected_capacities=normalized_capacities,
        selected_capacity_inputs=tuple(normalized_capacities.items()),
        transferred_horizon_rate_inputs=tuple(normalized_rates.items()),
        rows_by_family=rows_by_family,
    )


def compile_rq4_extra_id_surface(
    *,
    root: Path,
    capacity_surface: Rq4CapacitySurface,
    horizon_followup: Rq4HorizonFollowup,
    winner_selection_path: Path,
    expected_winner_selection_sha256: str,
) -> Rq4ExtraIdSurface:
    root = root.resolve(strict=True)
    _validate_recompiled_surface(root, capacity_surface)
    _validate_recompiled_surface(root, horizon_followup)
    if (
        horizon_followup.predecessor != capacity_surface.predecessor
        or horizon_followup.metadata_identity != capacity_surface.metadata_identity
    ):
        raise ValueError(
            "RQ4 staged surfaces do not share one authenticated predecessor"
        )
    winner_path, winner = _load_bound_document(
        root,
        winner_selection_path,
        expected_sha256=expected_winner_selection_sha256,
        label="RQ4 metadata winner",
    )
    family_id, metadata_dim = _authenticate_metadata_winner(
        winner,
        capacity_surface=capacity_surface,
        horizon_followup=horizon_followup,
    )
    specification = _metadata_specifications()[family_id]
    first_stage = compile_capacity_first_stage(specification)
    followup = compile_capacity_horizon_followup(
        specification,
        selected_capacity=horizon_followup.selected_capacities[family_id],
        transferred_horizon_rates=tuple(
            TransferredHorizonRate(
                row.horizon_epochs,
                row.embedding_learning_rate,
                row.deep_learning_rate,
            )
            for row in horizon_followup.rows_by_family[family_id]
        ),
        first_stage=first_stage,
    )
    predecessor_coordinates = (*first_stage, *followup)
    extra_specification = next(
        value for value in APPROVED_FAMILY_SPECS if value.id == "rq4_extra_item_id"
    )
    coordinates = compile_rq4_extra_id_control(
        extra_specification,
        predecessor=predecessor_coordinates,
    )
    rows = tuple(
        _extra_id_row(
            coordinate,
            predecessor_coordinate=predecessor_coordinate,
            predecessor=capacity_surface.predecessor,
            metadata_identity=capacity_surface.metadata_identity,
            metadata_family=family_id,
        )
        for coordinate, predecessor_coordinate in zip(
            coordinates, predecessor_coordinates, strict=True
        )
    )
    if any(row.matched_metadata_dim == metadata_dim for row in rows) is False:
        raise ValueError("RQ4 extra-ID surface omits the selected metadata width")
    return Rq4ExtraIdSurface(
        predecessor=capacity_surface.predecessor,
        metadata_identity=capacity_surface.metadata_identity,
        winner_selection_path=str(winner_path.relative_to(root)),
        winner_selection_sha256=expected_winner_selection_sha256,
        winner_selection_file_sha256=_file_sha256(winner_path),
        winner_selection_size_bytes=winner_path.stat().st_size,
        capacity_surface=capacity_surface,
        horizon_followup=horizon_followup,
        rows=rows,
    )


def resolve_rq4_feature_data(
    *,
    root: Path,
    surface: Rq4CapacitySurface | Rq4HorizonFollowup | Rq4ExtraIdSurface,
) -> Path:
    root = root.resolve(strict=True)
    _validate_predecessor_files(root, surface.predecessor)
    if isinstance(surface, Rq4ExtraIdSurface):
        winner_path = _resolve_input_path(root, root / surface.winner_selection_path)
        if (
            winner_path.stat().st_size != surface.winner_selection_size_bytes
            or _file_sha256(winner_path) != surface.winner_selection_file_sha256
        ):
            raise ValueError("RQ4 metadata-winner predecessor changed before launch")
    _validate_recompiled_surface(root, surface)
    identity, _ = _authenticate_metadata(root)
    if identity != surface.metadata_identity:
        raise ValueError("RQ4 training-only metadata identity changed before launch")
    return _resolve_input_path(root, root / identity.feature_data_path)


def _validate_recompiled_surface(
    root: Path,
    surface: Rq4CapacitySurface | Rq4HorizonFollowup | Rq4ExtraIdSurface,
) -> None:
    if isinstance(surface, Rq4CapacitySurface):
        rebuilt = _recompile_capacity_surface(root, surface.predecessor)
    elif isinstance(surface, Rq4HorizonFollowup):
        capacity = _recompile_capacity_surface(root, surface.predecessor)
        rebuilt = compile_rq4_horizon_followup(
            capacity,
            selected_capacities=dict(surface.selected_capacity_inputs),
            transferred_horizon_rates=dict(surface.transferred_horizon_rate_inputs),
        )
    else:
        _validate_recompiled_surface(root, surface.capacity_surface)
        _validate_recompiled_surface(root, surface.horizon_followup)
        rebuilt = compile_rq4_extra_id_surface(
            root=root,
            capacity_surface=surface.capacity_surface,
            horizon_followup=surface.horizon_followup,
            winner_selection_path=root / surface.winner_selection_path,
            expected_winner_selection_sha256=surface.winner_selection_sha256,
        )
    if rebuilt != surface:
        raise ValueError("RQ4 launch input differs from its approved staged surface")


def _recompile_capacity_surface(
    root: Path, predecessor: Rq4Predecessor
) -> Rq4CapacitySurface:
    return compile_rq4_capacity_surface(
        root=root,
        rq2_selection_path=root / predecessor.rq2_selection_path,
        expected_rq2_selection_sha256=predecessor.rq2_selection_sha256,
        rq3_selection_path=root / predecessor.rq3_selection_path,
        expected_rq3_selection_sha256=predecessor.rq3_selection_sha256,
        expected_rq3_row_id=predecessor.rq3_row_id,
    )


def _metadata_specifications() -> dict[str, FamilySpec]:
    specifications = {
        specification.id: specification
        for specification in APPROVED_FAMILY_SPECS
        if specification.id in RQ4_METADATA_FAMILIES
    }
    if tuple(specifications) != RQ4_METADATA_FAMILIES:
        raise ValueError("approved RQ4 metadata family definitions changed")
    return specifications


def _metadata_row(
    coordinate: SearchCoordinate,
    *,
    metadata: tuple[str, ...],
    phase: str,
    reused_run_name: str | None = None,
) -> Rq4MetadataRow:
    capacity = coordinate.capacity
    if type(capacity) is not int:
        raise ValueError("RQ4 metadata coordinate has no width")
    run_name = reused_run_name or (
        f"g3_{coordinate.family_id}_{phase}_{coordinate.opportunity_index + 1:02d}_"
        f"width_{capacity}_horizon_{coordinate.horizon_epochs}_native50m"
    )
    return Rq4MetadataRow(
        id=coordinate.id,
        family_id=coordinate.family_id,
        run_name=run_name,
        batch_size=coordinate.batch_size,
        seed=coordinate.seed,
        embedding_learning_rate=coordinate.embedding_learning_rate,
        deep_learning_rate=coordinate.deep_learning_rate,
        horizon_epochs=coordinate.horizon_epochs,
        metadata=metadata,
        metadata_dim=capacity,
        reused_from=coordinate.reused_from,
    )


def _extra_id_row(
    coordinate: SearchCoordinate,
    *,
    predecessor_coordinate: SearchCoordinate,
    predecessor: Rq4Predecessor,
    metadata_identity: Rq4MetadataIdentity,
    metadata_family: str,
) -> Rq4ExtraIdRow:
    metadata_dim = predecessor_coordinate.capacity
    if type(metadata_dim) is not int:
        raise ValueError("RQ4 extra-ID predecessor has no metadata width")
    extra_dim, mismatch = _matched_extra_id_dimension(
        predecessor=predecessor,
        identity=metadata_identity,
        metadata_family=metadata_family,
        metadata_dim=metadata_dim,
    )
    return Rq4ExtraIdRow(
        id=coordinate.id,
        family_id=coordinate.family_id,
        run_name=(
            f"g3_rq4_extra_item_id_trial_{coordinate.opportunity_index + 1:02d}_"
            f"width_{extra_dim}_horizon_{coordinate.horizon_epochs}_native50m"
        ),
        batch_size=coordinate.batch_size,
        seed=coordinate.seed,
        embedding_learning_rate=coordinate.embedding_learning_rate,
        deep_learning_rate=coordinate.deep_learning_rate,
        horizon_epochs=coordinate.horizon_epochs,
        extra_item_id_dim=extra_dim,
        matched_metadata_family=metadata_family,
        matched_metadata_dim=metadata_dim,
        parameter_mismatch_fraction=mismatch,
    )


def _matched_extra_id_dimension(
    *,
    predecessor: Rq4Predecessor,
    identity: Rq4MetadataIdentity,
    metadata_family: str,
    metadata_dim: int,
) -> tuple[int, float]:
    vocabularies = {
        "artist": identity.artist_vocab_size,
        "album": identity.album_vocab_size,
    }
    metadata = _METADATA_BY_FAMILY[metadata_family]
    history_parameters = _history_encoder_parameters(
        identity.num_items, predecessor.history_hidden_dim
    )
    catalog_parameters = _catalog_encoder_parameters(
        identity.num_items, predecessor.catalog_representation
    )
    wrapper_constant = 2 * (64 * 64 + 64 + 64 * 64 + 64)
    common = history_parameters + catalog_parameters + wrapper_constant
    metadata_variable = (
        2
        * metadata_dim
        * (sum(vocabularies[name] + 1 for name in metadata) + 64 * len(metadata))
    )
    extra_coefficient = 2 * (identity.num_items + 1 + 64)
    target = metadata_variable / extra_coefficient
    candidates = {max(1, math.floor(target)), max(1, math.ceil(target))}
    extra_dim = min(
        candidates,
        key=lambda value: (abs(extra_coefficient * value - metadata_variable), value),
    )
    metadata_total = common + metadata_variable
    extra_total = common + extra_coefficient * extra_dim
    mismatch = abs(extra_total - metadata_total) / metadata_total
    if mismatch >= 0.01:
        raise ValueError("RQ4 extra item-ID control cannot match parameters within 1%")
    return extra_dim, mismatch


def _history_encoder_parameters(num_items: int, hidden_dim: int) -> int:
    return (num_items + 1) * 64 + hidden_dim * (192 + 1 + 64) + 64


def _catalog_encoder_parameters(num_items: int, representation: str) -> int:
    item_table = (num_items + 1) * 64
    content_table = (num_items + 1) * 128
    return {
        "learned_id": item_table,
        "frozen_content": 128 * 64,
        "trainable_content": content_table + 128 * 64,
        "id_frozen_content": item_table + 192 * 64,
        "id_trainable_content": item_table + content_table + 192 * 64,
    }[representation]


def _authenticate_metadata(
    root: Path,
) -> tuple[Rq4MetadataIdentity, dict[str, object]]:
    manifest_path = _resolve_input_path(root, root / NATIVE50_FEATURE_MANIFEST_PATH)
    if _file_sha256(manifest_path) != NATIVE50_FEATURE_MANIFEST_FILE_SHA256:
        raise ValueError("RQ4 native-50M feature manifest bytes changed")
    manifest = load_artifact_manifest(manifest_path)
    if manifest.sha256 != NATIVE50_FEATURE_MANIFEST_SHA256:
        raise ValueError("RQ4 native-50M feature manifest logical identity changed")
    validate_feature_manifest(root=root, manifest=manifest, validate_files=True)
    metadata = manifest.metadata
    if metadata["dataset_size"] != "native-50m":
        raise ValueError("RQ4 metadata must use native-50M")
    bindings = {binding.role: binding for binding in manifest.artifacts}
    feature_binding = bindings["item_features"]
    feature_data = load_feature_data(root / feature_binding.path)
    _validate_loaded_metadata(feature_data, metadata)
    frequency = build_frequency_identity(feature_data.training_counts)
    identity = Rq4MetadataIdentity(
        manifest_path=str(manifest_path.relative_to(root)),
        manifest_sha256=manifest.sha256,
        manifest_file_sha256=NATIVE50_FEATURE_MANIFEST_FILE_SHA256,
        feature_data_path=feature_binding.path,
        feature_data_sha256=feature_binding.sha256,
        num_items=int(metadata["num_items"]),
        artist_vocab_size=feature_data.artist_vocab_size,
        album_vocab_size=feature_data.album_vocab_size,
        frequency_terciles=dict(frequency["frequency_terciles"]),
        training_count_reference=dict(frequency["training_count_reference"]),
        slice_membership_reference=dict(frequency["slice_membership_reference"]),
        training_count_sha256=_canonical_sha256(
            tuple(int(value) for value in feature_data.training_counts.tolist())
        ),
        artist_membership_sha256=_canonical_sha256(feature_data.artist_rows),
        album_membership_sha256=_canonical_sha256(feature_data.album_rows),
    )
    reference = {
        "path": identity.manifest_path,
        "sha256": identity.manifest_file_sha256,
        "size_bytes": manifest_path.stat().st_size,
        "logical_sha256": identity.manifest_sha256,
    }
    return identity, reference


def _validate_loaded_metadata(
    feature_data: LoadedFeatureData, metadata: Mapping[str, object]
) -> None:
    num_items = int(metadata["num_items"])
    if (
        len(feature_data.training_counts) != num_items + 1
        or int(feature_data.training_counts.sum()) != metadata["training_rows"]
        or len(feature_data.training_history_lengths) != metadata["training_users"]
        or feature_data.artist_vocab_size != metadata["artist_vocab_size"]
        or feature_data.album_vocab_size != metadata["album_vocab_size"]
        or len(feature_data.artist_rows) != num_items + 1
        or len(feature_data.album_rows) != num_items + 1
    ):
        raise ValueError("RQ4 loaded training-only metadata counts changed")
    for name in ("artist", "album"):
        rows = getattr(feature_data, f"{name}_rows")
        vocab_size = getattr(feature_data, f"{name}_vocab_size")
        if rows[0] != () or any(
            tuple(sorted(set(row))) != row
            or any(
                type(value) is not int or not 1 <= value <= vocab_size for value in row
            )
            for row in rows[1:]
        ):
            raise ValueError(f"RQ4 {name} memberships are invalid")
        maximum = max(map(len, rows[1:]), default=0)
        unknown_rate = sum(not row for row in rows[1:]) / num_items
        if maximum != metadata[f"{name}_max_cardinality"] or not math.isclose(
            unknown_rate,
            float(metadata[f"{name}_unknown_rate"]),
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError(f"RQ4 {name} metadata summary changed")


def _authenticate_rq2_selection(
    document: Mapping[str, object], *, expected_feature_manifest: Mapping[str, object]
) -> int:
    if document.get("kind") == "g3_rq2_final_native50m_evidence":
        expected_keys = {
            "schema_version",
            "kind",
            "protocol_sha256",
            "diagnostic_evidence",
            "boundary_ledger",
            "queue_batch",
            "ranking_context",
            "boundary_tuning_ledger",
            "all_tuning_diagnostic_boundary_ledger",
            "final_content_selection",
            "final_rq2_comparison",
            "rq3_inputs",
            "opportunity_accounting",
            "sha256",
        }
        selection = document.get("final_content_selection")
        selected = selection.get("selected") if isinstance(selection, dict) else None
        rq3_inputs = document.get("rq3_inputs")
        reusable = (
            rq3_inputs.get("eligible_learned_output_reuse_rows")
            if isinstance(rq3_inputs, dict)
            else None
        )
        width = selected.get("capacity") if isinstance(selected, dict) else None
        if (
            set(document) != expected_keys
            or document.get("schema_version") != 1
            or document.get("protocol_sha256") != APPROVED_PROTOCOL_SHA256
            or not isinstance(selection, dict)
            or selection.get("status") != "resolved"
            or selection.get("provisional_selected") is not None
            or not isinstance(selected, dict)
            or selected.get("row_id") != "rq2_unexpected_diagnostic:03"
            or selected.get("family_id") != "rq2_content_concat"
            or type(width) is not int
            or width < 1
            or not isinstance(rq3_inputs, dict)
            or rq3_inputs.get("status") != "ready"
            or rq3_inputs.get("selected_content_input") != selected
            or not isinstance(reusable, list)
            or len(reusable) != 7
            or selected not in reusable
            or document.get("opportunity_accounting", {}).get("all_preserved_rows")
            != 36
        ):
            raise ValueError("RQ4 final RQ2 predecessor is not resolved and reusable")
        return width
    expected_keys = {
        "schema_version",
        "kind",
        "protocol_sha256",
        "selected_family_id",
        "selected_history_hidden_dim",
        "selection_resolved",
        "feature_manifest",
        "rows",
        "sha256",
    }
    width = document.get("selected_history_hidden_dim")
    if (
        set(document) != expected_keys
        or document.get("schema_version") != 1
        or document.get("kind") != "g3_rq2_content_selection_for_rq3"
        or document.get("protocol_sha256") != APPROVED_PROTOCOL_SHA256
        or document.get("selected_family_id") != "rq2_content_concat"
        or document.get("selection_resolved") is not True
        or type(width) is not int
        or width < 1
        or document.get("feature_manifest") != expected_feature_manifest
        or not isinstance(document.get("rows"), list)
        or not document["rows"]
    ):
        raise ValueError("RQ4 RQ2 predecessor is not the resolved content selection")
    return width


def _authenticate_rq3_selection(
    root: Path,
    document: Mapping[str, object],
    *,
    rq2_path: Path,
    rq2_sha256: str,
    rq3_path: Path,
    rq3_sha256: str,
    history_hidden_dim: int,
    expected_feature_manifest: Mapping[str, object],
    feature_identity: Rq4MetadataIdentity,
    expected_row_id: str | None,
) -> Rq4Predecessor:
    if document.get("kind") == "g3_rq3_final_native50m_evidence":
        document = _adapt_final_rq3_evidence(
            root,
            document,
            rq2_path=rq2_path,
            rq2_sha256=rq2_sha256,
            expected_feature_manifest=expected_feature_manifest,
        )
    expected_keys = {
        "schema_version",
        "kind",
        "protocol_sha256",
        "selection_resolved",
        "source_rq2_selection",
        "feature_manifest",
        "ranking_context",
        "selected",
        "sha256",
    }
    if (
        set(document) != expected_keys
        or document.get("schema_version") != 1
        or document.get("kind") != "g3_rq3_catalog_selection_for_rq4"
        or document.get("protocol_sha256") != APPROVED_PROTOCOL_SHA256
        or document.get("selection_resolved") is not True
        or document.get("feature_manifest") != expected_feature_manifest
    ):
        raise ValueError("RQ4 RQ3 predecessor is not a resolved catalog selection")
    rq2_reference = _validate_file_reference(
        root, document["source_rq2_selection"], logical=True
    )
    if (
        _resolve_input_path(root, root / str(rq2_reference["path"])) != rq2_path
        or rq2_reference["logical_sha256"] != rq2_sha256
    ):
        raise ValueError("RQ4 RQ3 predecessor binds another RQ2 selection")
    selected = document["selected"]
    selected_keys = {
        "row_id",
        "family_id",
        "run_name",
        "history_hidden_dim",
        "catalog_representation",
        "embedding_learning_rate",
        "deep_learning_rate",
        "horizon_epochs",
        "queue_job",
        "artifacts",
    }
    if not isinstance(selected, dict) or set(selected) != selected_keys:
        raise ValueError("RQ4 selected RQ3 row schema changed")
    family_id = selected["family_id"]
    expected_catalog = RQ3_CATALOG_REPRESENTATIONS.get(family_id)
    if (
        expected_catalog is None
        or selected["catalog_representation"] != expected_catalog
        or selected["history_hidden_dim"] != history_hidden_dim
        or not isinstance(selected["row_id"], str)
        or not selected["row_id"].startswith(f"{family_id}:")
        or not isinstance(selected["run_name"], str)
        or not selected["run_name"]
        or (expected_row_id is not None and selected["row_id"] != expected_row_id)
    ):
        raise ValueError("RQ4 selected RQ3 representation differs from its predecessor")
    _validate_positive_coordinate(selected)
    _validate_file_reference(root, document["ranking_context"], logical=False)
    _validate_file_reference(root, selected["queue_job"], logical=False)
    artifacts = _validate_artifacts(root, selected["artifacts"])
    _validate_rq3_runtime_artifacts(
        root,
        selected,
        artifacts=artifacts,
        history_hidden_dim=history_hidden_dim,
        catalog_representation=expected_catalog,
        feature_identity=feature_identity,
        expected_rq2_sha256=rq2_sha256,
    )
    return Rq4Predecessor(
        rq2_selection_path=str(rq2_path.relative_to(root)),
        rq2_selection_sha256=rq2_sha256,
        rq2_selection_file_sha256=_file_sha256(rq2_path),
        rq2_selection_size_bytes=rq2_path.stat().st_size,
        rq3_selection_path=str(rq3_path.relative_to(root)),
        rq3_selection_sha256=rq3_sha256,
        rq3_selection_file_sha256=_file_sha256(rq3_path),
        rq3_selection_size_bytes=rq3_path.stat().st_size,
        rq3_artifacts=tuple(
            (
                name,
                str(reference["path"]),
                str(reference["sha256"]),
                int(reference["size_bytes"]),
            )
            for name, reference in sorted(artifacts.items())
        ),
        rq3_row_id=str(selected["row_id"]),
        rq3_family_id=str(family_id),
        history_hidden_dim=history_hidden_dim,
        catalog_representation=expected_catalog,
    )


def _adapt_final_rq3_evidence(
    root: Path,
    document: Mapping[str, object],
    *,
    rq2_path: Path,
    rq2_sha256: str,
    expected_feature_manifest: Mapping[str, object],
) -> dict[str, object]:
    expected_keys = {
        "schema_version",
        "kind",
        "protocol_sha256",
        "initial_evidence",
        "boundary_ledger",
        "queue_batch",
        "ranking_context",
        "feature_data",
        "opportunity_accounting",
        "all_tuning_opportunities",
        "boundary_runs",
        "family_selections",
        "downstream_selection",
        "reader_metrics",
        "selected_winner_contrasts",
        "matched_coordinate_contrasts",
        "mechanism_assessment",
        "sha256",
    }
    downstream = document.get("downstream_selection")
    selected = (
        downstream.get("rq4_scientific_selected")
        if isinstance(downstream, dict)
        else None
    )
    best = downstream.get("best_absolute") if isinstance(downstream, dict) else None
    family_selections = document.get("family_selections")
    family = (
        family_selections.get(selected.get("family_id"))
        if isinstance(family_selections, dict) and isinstance(selected, dict)
        else None
    )
    family_selected = family.get("selected") if isinstance(family, dict) else None
    artifacts = selected.get("artifacts") if isinstance(selected, dict) else None
    queue = selected.get("queue_job") if isinstance(selected, dict) else None
    if (
        set(document) != expected_keys
        or document.get("schema_version") != 1
        or document.get("protocol_sha256") != APPROVED_PROTOCOL_SHA256
        or not isinstance(downstream, dict)
        or downstream.get("status") != "resolved"
        or downstream.get("unresolved_boundary_families") != []
        or not isinstance(selected, dict)
        or selected != best
        or selected != family_selected
        or not isinstance(artifacts, dict)
        or not isinstance(queue, dict)
        or set(queue) != {"path", "size_bytes", "sha256", "job_id"}
        or selected.get("metric_provenance", {}).get(
            "recomputed_from_ranking_evidence"
        )
        is not True
    ):
        raise ValueError("RQ4 final RQ3 scientific selection is unresolved")
    validated_artifacts = _validate_artifacts(root, artifacts)
    context = _validate_file_reference(
        root, document.get("ranking_context"), logical=False
    )
    queue_reference = _validate_file_reference(
        root,
        {name: queue[name] for name in ("path", "size_bytes", "sha256")},
        logical=False,
    )
    contract = _load_json(root / validated_artifacts["job_contract"]["path"])
    job = contract.get("job") if isinstance(contract, dict) else None
    representation = job.get("representation") if isinstance(job, dict) else None
    if (
        not isinstance(representation, dict)
        or job.get("id") != selected.get("row_id")
        or job.get("family_id") != selected.get("family_id")
        or job.get("run_name") != selected.get("run_name")
        or contract.get("row_id") != selected.get("row_id")
        or queue.get("job_id") != _load_json(root / queue_reference["path"]).get("id")
    ):
        raise ValueError("RQ4 final RQ3 selected runtime binding changed")
    return {
        "schema_version": 1,
        "kind": "g3_rq3_catalog_selection_for_rq4",
        "protocol_sha256": APPROVED_PROTOCOL_SHA256,
        "selection_resolved": True,
        "source_rq2_selection": {
            "path": str(rq2_path.relative_to(root)),
            "sha256": _file_sha256(rq2_path),
            "size_bytes": rq2_path.stat().st_size,
            "logical_sha256": rq2_sha256,
        },
        "feature_manifest": dict(expected_feature_manifest),
        "ranking_context": context,
        "selected": {
            "row_id": selected["row_id"],
            "family_id": selected["family_id"],
            "run_name": selected["run_name"],
            "history_hidden_dim": representation["history_hidden_dim"],
            "catalog_representation": selected["catalog_representation"],
            "embedding_learning_rate": selected["embedding_learning_rate"],
            "deep_learning_rate": selected["deep_learning_rate"],
            "horizon_epochs": selected["horizon_epochs"],
            "queue_job": queue_reference,
            "artifacts": validated_artifacts,
        },
        "sha256": document["sha256"],
    }


def _validate_predecessor_files(root: Path, predecessor: Rq4Predecessor) -> None:
    selections = (
        (
            predecessor.rq2_selection_path,
            predecessor.rq2_selection_file_sha256,
            predecessor.rq2_selection_size_bytes,
        ),
        (
            predecessor.rq3_selection_path,
            predecessor.rq3_selection_file_sha256,
            predecessor.rq3_selection_size_bytes,
        ),
    )
    for relative, sha256, size_bytes in selections:
        path = _resolve_input_path(root, root / relative)
        if path.stat().st_size != size_bytes or _file_sha256(path) != sha256:
            raise ValueError("RQ4 frozen predecessor changed before launch")
    for _, relative, sha256, size_bytes in predecessor.rq3_artifacts:
        path = _resolve_input_path(root, root / relative)
        if path.stat().st_size != size_bytes or _file_sha256(path) != sha256:
            raise ValueError("RQ4 frozen predecessor artifact changed before launch")


def _validate_positive_coordinate(value: Mapping[str, object]) -> None:
    for key in ("embedding_learning_rate", "deep_learning_rate"):
        number = value[key]
        if type(number) not in {int, float} or not math.isfinite(number) or number <= 0:
            raise ValueError(f"RQ4 predecessor {key} is invalid")
    horizon = value["horizon_epochs"]
    if type(horizon) is not int or horizon < 1:
        raise ValueError("RQ4 predecessor horizon is invalid")


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_rq3_runtime_artifacts(
    root: Path,
    selected: Mapping[str, object],
    *,
    artifacts: Mapping[str, Mapping[str, object]],
    history_hidden_dim: int,
    catalog_representation: str,
    feature_identity: Rq4MetadataIdentity,
    expected_rq2_sha256: str,
) -> None:
    contract = _load_json(root / str(artifacts["job_contract"]["path"]))
    if set(contract) != {"row_id", "job", "ledger_path", "ledger_sha256"}:
        raise ValueError("RQ4 RQ3 job contract schema changed")
    job = contract.get("job")
    ledger_path_value = contract.get("ledger_path")
    if not isinstance(job, dict) or not isinstance(ledger_path_value, str):
        raise ValueError("RQ4 RQ3 job contract is incomplete")
    ledger_path = _resolve_input_path(root, Path(ledger_path_value))
    source_ledger = _load_rq3_source_ledger(ledger_path)
    source_rows = [row for row in source_ledger.logical_rows if row.id == selected["row_id"]]
    if len(source_rows) != 1:
        raise ValueError("RQ4 selected RQ3 row is absent from its source ledger")
    source_row = source_rows[0]
    representation = job.get("representation")
    training = job.get("training")
    if (
        contract.get("row_id") != selected["row_id"]
        or contract.get("ledger_sha256") != source_ledger.sha256
        or source_ledger.final_rq2_evidence_sha256 != expected_rq2_sha256
        or job != source_row.to_dict()
        or job.get("id") != selected["row_id"]
        or job.get("family_id") != selected["family_id"]
        or job.get("run_name") != selected["run_name"]
        or not isinstance(representation, dict)
        or representation.get("history_hidden_dim") != selected["history_hidden_dim"]
        or representation.get("catalog_representation")
        != selected["catalog_representation"]
        or not isinstance(training, dict)
        or training.get("embedding_learning_rate")
        != selected["embedding_learning_rate"]
        or training.get("deep_learning_rate") != selected["deep_learning_rate"]
        or training.get("horizon_epochs") != selected["horizon_epochs"]
    ):
        raise ValueError("RQ4 RQ3 job contract differs from its ledger-backed selected row")
    metadata = _load_json(root / str(artifacts["training_metadata"]["path"]))
    expected_representation = {
        "history_representation": "id_content",
        "catalog_representation": catalog_representation,
        "history_hidden_dim": history_hidden_dim,
        "content_gate": "fixed",
        "gate_hidden_dim": None,
        "metadata": [],
        "metadata_dim": None,
        "extra_item_id_dim": None,
    }
    if (
        metadata.get("g3_dataset_size") != "native-50m"
        or metadata.get("g3_protocol_sha256") != APPROVED_PROTOCOL_SHA256
        or metadata.get("training_semantics_revision") != 2
        or metadata.get("g3_representation") != expected_representation
    ):
        raise ValueError("RQ4 selected RQ3 runtime metadata is incompatible")
    metrics = _load_json(root / str(artifacts["final_metrics"]["path"]))
    if not metrics or any(
        type(value) not in {int, float} or not math.isfinite(value)
        for value in metrics.values()
    ):
        raise ValueError("RQ4 selected RQ3 metrics are invalid")
    diagnostics = _load_json(root / str(artifacts["training_diagnostics"]["path"]))
    _validate_training_diagnostics(
        diagnostics,
        feature_identity=_FeatureIdentity(
            manifest_path=feature_identity.manifest_path,
            manifest_sha256=feature_identity.manifest_sha256,
            manifest_file_sha256=feature_identity.manifest_file_sha256,
            data_path=feature_identity.feature_data_path,
            data_sha256=feature_identity.feature_data_sha256,
            frequency_terciles=feature_identity.frequency_terciles,
            training_count_reference=feature_identity.training_count_reference,
            slice_membership_reference=feature_identity.slice_membership_reference,
        ),
        horizon_epochs=int(selected["horizon_epochs"]),
        catalog_representation=catalog_representation,
    )


def _load_rq3_source_ledger(path: Path) -> object:
    from .rq3_post_boundary import load_rq3_post_boundary_ledger

    return load_rq3_post_boundary_ledger(path)


def _authenticate_metadata_winner(
    document: Mapping[str, object],
    *,
    capacity_surface: Rq4CapacitySurface,
    horizon_followup: Rq4HorizonFollowup,
) -> tuple[str, int]:
    expected_keys = {
        "schema_version",
        "kind",
        "protocol_sha256",
        "selection_resolved",
        "rq2_selection_sha256",
        "rq3_selection_sha256",
        "selected_family_id",
        "selected_metadata_dim",
        "initial_ledger_sha256",
        "horizon_ledger_sha256",
        "selected",
        "family_boundaries",
        "sha256",
    }
    family_id = document.get("selected_family_id")
    metadata_dim = document.get("selected_metadata_dim")
    selected = document.get("selected")
    boundaries = document.get("family_boundaries")
    available_rows = (
        *capacity_surface.rows_by_family.get(str(family_id), ()),
        *horizon_followup.rows_by_family.get(str(family_id), ()),
    )
    selected_row = next(
        (
            row
            for row in available_rows
            if isinstance(selected, dict) and row.id == selected.get("row_id")
        ),
        None,
    )
    selected_job = selected.get("job") if isinstance(selected, dict) else None
    selected_representation = (
        selected_job.get("representation") if isinstance(selected_job, dict) else None
    )
    selected_training = (
        selected_job.get("training") if isinstance(selected_job, dict) else None
    )
    metric_provenance = (
        selected.get("metric_provenance") if isinstance(selected, dict) else None
    )
    horizon_ids = {
        row.id for row in horizon_followup.rows_by_family.get(str(family_id), ())
    }
    expected_ledger_sha256 = (
        document.get("horizon_ledger_sha256")
        if selected_row is not None and selected_row.id in horizon_ids
        else document.get("initial_ledger_sha256")
    )
    if (
        set(document) != expected_keys
        or document.get("schema_version") != 1
        or document.get("kind") != "g3_rq4_metadata_winner_for_extra_id"
        or document.get("protocol_sha256") != APPROVED_PROTOCOL_SHA256
        or document.get("selection_resolved") is not True
        or document.get("rq2_selection_sha256")
        != capacity_surface.predecessor.rq2_selection_sha256
        or document.get("rq3_selection_sha256")
        != capacity_surface.predecessor.rq3_selection_sha256
        or family_id not in RQ4_METADATA_FAMILIES
        or type(metadata_dim) is not int
        or metadata_dim != horizon_followup.selected_capacities.get(family_id)
        or not _valid_sha256(document.get("initial_ledger_sha256"))
        or not _valid_sha256(document.get("horizon_ledger_sha256"))
        or not isinstance(boundaries, dict)
        or set(boundaries) != set(RQ4_METADATA_FAMILIES)
        or any(
            not isinstance(value, dict)
            or value.get("extension_required") is not False
            for value in boundaries.values()
        )
        or selected_row is None
        or selected.get("family_id") != family_id
        or selected.get("metadata_dim") != metadata_dim
        or selected.get("embedding_learning_rate")
        != selected_row.embedding_learning_rate
        or selected.get("deep_learning_rate") != selected_row.deep_learning_rate
        or selected.get("horizon_epochs") != selected_row.horizon_epochs
        or selected.get("ledger_sha256") != expected_ledger_sha256
        or not isinstance(selected_job, dict)
        or selected_job.get("id") != selected_row.id
        or selected_job.get("family_id") != family_id
        or not isinstance(selected_representation, dict)
        or selected_representation.get("metadata_dim") != selected_row.metadata_dim
        or selected_representation.get("metadata") != list(selected_row.metadata)
        or not isinstance(selected_training, dict)
        or selected_training.get("embedding_learning_rate")
        != selected_row.embedding_learning_rate
        or selected_training.get("deep_learning_rate")
        != selected_row.deep_learning_rate
        or selected_training.get("horizon_epochs") != selected_row.horizon_epochs
        or not isinstance(selected.get("artifacts"), dict)
        or not isinstance(metric_provenance, dict)
        or metric_provenance.get("recomputed_from_ranking_evidence") is not True
    ):
        raise ValueError(
            "RQ4 extra-ID control lacks the exact selected metadata winner"
        )
    return family_id, metadata_dim


def _validate_equal_budget(
    rows_by_family: Mapping[str, tuple[Rq4MetadataRow, ...]],
    *,
    expected_rows: int,
    require_equal_capacities: bool = True,
) -> None:
    if tuple(rows_by_family) != RQ4_METADATA_FAMILIES or any(
        len(rows) != expected_rows for rows in rows_by_family.values()
    ):
        raise ValueError("RQ4 metadata families no longer have equal budgets")
    signatures = [
        tuple(
            (
                row.embedding_learning_rate,
                row.deep_learning_rate,
                row.horizon_epochs,
                row.metadata_dim if require_equal_capacities else None,
            )
            for row in rows
        )
        for rows in rows_by_family.values()
    ]
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise ValueError("RQ4 metadata families no longer share one search surface")


def _validate_artifacts(root: Path, value: object) -> dict[str, dict[str, object]]:
    expected = {
        "job_contract",
        "training_metadata",
        "final_metrics",
        "ranking_evidence",
        "top_item_rankings",
        "training_diagnostics",
        "sweep_log",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("RQ4 selected RQ3 artifacts are incomplete")
    return {
        name: _validate_file_reference(root, reference, logical=False)
        for name, reference in value.items()
    }


def _load_bound_document(
    root: Path,
    path: Path,
    *,
    expected_sha256: str,
    label: str,
) -> tuple[Path, dict[str, object]]:
    resolved = _resolve_input_path(root, path)
    document = _load_json(resolved)
    payload = {key: value for key, value in document.items() if key != "sha256"}
    logical_sha256 = _canonical_sha256(payload)
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or document.get("sha256") != logical_sha256
        or logical_sha256 != expected_sha256
    ):
        raise ValueError(f"{label} is not the exact frozen document")
    return resolved, document


def _validate_file_reference(
    root: Path, value: object, *, logical: bool
) -> dict[str, object]:
    keys = {"path", "sha256", "size_bytes"}
    if logical:
        keys.add("logical_sha256")
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError("RQ4 predecessor file reference schema changed")
    if (
        not isinstance(value["path"], str)
        or not isinstance(value["sha256"], str)
        or type(value["size_bytes"]) is not int
        or value["size_bytes"] < 0
        or (logical and not isinstance(value["logical_sha256"], str))
    ):
        raise ValueError("RQ4 predecessor file reference types changed")
    relative = Path(value["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("RQ4 predecessor reference must be project-relative")
    path = _resolve_input_path(root, root / relative)
    if (
        path.stat().st_size != value["size_bytes"]
        or _file_sha256(path) != value["sha256"]
    ):
        raise ValueError("RQ4 predecessor file differs from its frozen reference")
    return dict(value)


def _resolve_input_path(root: Path, path: Path) -> Path:
    candidate = path if path.is_absolute() else root / path
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError(f"RQ4 bound input is not a regular file: {candidate}")
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("RQ4 bound input escapes the project root")
    return resolved


def _load_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(
            path.read_text(),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load RQ4 bound JSON {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"RQ4 bound JSON must be an object: {path}")
    return value


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


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
