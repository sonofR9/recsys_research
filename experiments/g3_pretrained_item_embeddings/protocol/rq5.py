from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
import re
from typing import Literal, Mapping, Protocol, Sequence

from .search import (
    APPROVED_FAMILY_SPECS,
    FamilySpec,
    SearchCoordinate,
    TransferredHorizonRate,
    compile_capacity_first_stage,
    compile_capacity_horizon_followup,
    compile_family,
)


RQ5_COMPARATOR_FAMILY_IDS = (
    "rq2_content_concat",
    "rq5_global_gate",
    "rq5_frequency_gate",
)
RQ5_OPPORTUNITY_BUDGET_PER_NEW_FAMILY = 12

_SHA256 = re.compile(r"[0-9a-f]{64}")


class _AuthenticatedSource(Protocol):
    source_id: str
    run_name: str
    history_hidden_dim: int
    embedding_learning_rate: float
    deep_learning_rate: float
    horizon_epochs: int


class _AuthenticatedRow(Protocol):
    run_name: str
    batch_size: int
    seed: int
    embedding_learning_rate: float
    deep_learning_rate: float
    horizon_epochs: int
    history_hidden_dim: int
    reused_from: str | None
    authenticated_source: _AuthenticatedSource | None


class AuthenticatedRq2ContentSelection(Protocol):
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
    rows_by_family: Mapping[str, Sequence[_AuthenticatedRow]]


@dataclass(frozen=True)
class FrozenRq2ContentBinding:
    selection_path: str
    selection_sha256: str
    selected_source_id: str


@dataclass(frozen=True)
class Rq5GateRow:
    id: str
    family_id: str
    run_name: str
    batch_size: int
    seed: int
    embedding_learning_rate: float
    deep_learning_rate: float
    horizon_epochs: int
    history_hidden_dim: int
    content_gate: Literal["fixed", "global", "frequency"]
    gate_hidden_dim: int | None
    reused_from: str | None


@dataclass(frozen=True)
class Rq5GateSurface:
    predecessor: AuthenticatedRq2ContentSelection
    binding: FrozenRq2ContentBinding
    selected_history_hidden_dim: int
    feature_manifest_path: str
    feature_manifest_sha256: str
    feature_manifest_file_sha256: str
    feature_data_path: str
    feature_data_sha256: str
    frequency_terciles: dict[str, object]
    training_count_reference: dict[str, object]
    slice_membership_reference: dict[str, object]
    fixed_gate: Rq5GateRow
    global_gate_rows: tuple[Rq5GateRow, ...]
    frequency_gate_rows: tuple[Rq5GateRow, ...]
    selected_frequency_gate_hidden_dim: int | None = None
    transferred_horizon_rates: tuple[TransferredHorizonRate, ...] = ()

    @property
    def comparator_family_ids(self) -> tuple[str, str, str]:
        return RQ5_COMPARATOR_FAMILY_IDS

    @property
    def new_training_rows(self) -> tuple[Rq5GateRow, ...]:
        return tuple(
            row
            for row in (*self.global_gate_rows, *self.frequency_gate_rows)
            if row.reused_from is None
        )


def authenticate_and_compile_rq5_gate_surface(
    *,
    root: Path,
    binding: FrozenRq2ContentBinding,
) -> Rq5GateSurface:
    from .rq3 import compile_rq3_output_surface

    predecessor = compile_rq3_output_surface(
        root=root,
        selection_path=Path(binding.selection_path),
        expected_selection_sha256=binding.selection_sha256,
    )
    return compile_rq5_gate_surface(predecessor=predecessor, binding=binding)


def compile_rq5_gate_surface(
    *,
    predecessor: AuthenticatedRq2ContentSelection,
    binding: FrozenRq2ContentBinding,
) -> Rq5GateSurface:
    _validate_binding(predecessor, binding)
    width = predecessor.selected_history_hidden_dim
    if type(width) is not int or width < 1:
        raise ValueError("RQ5 selected RQ2 history width is invalid")
    source = _selected_fixed_gate_source(predecessor, binding, width=width)
    global_spec, frequency_spec = _approved_gate_specs()
    global_coordinates = compile_family(global_spec)
    frequency_coordinates = compile_capacity_first_stage(frequency_spec)
    if (
        len(global_coordinates) != RQ5_OPPORTUNITY_BUDGET_PER_NEW_FAMILY
        or len(frequency_coordinates) != 9
    ):
        raise ValueError("RQ5 approved gate opportunity budget changed")
    identity = _feature_identity(predecessor)
    fixed_gate = Rq5GateRow(
        id="rq5_fixed_gate:reuse",
        family_id="rq2_content_concat",
        run_name=source.run_name,
        batch_size=512,
        seed=42,
        embedding_learning_rate=float(source.embedding_learning_rate),
        deep_learning_rate=float(source.deep_learning_rate),
        horizon_epochs=int(source.horizon_epochs),
        history_hidden_dim=width,
        content_gate="fixed",
        gate_hidden_dim=None,
        reused_from=source.source_id,
    )
    return Rq5GateSurface(
        predecessor=predecessor,
        binding=binding,
        selected_history_hidden_dim=width,
        feature_manifest_path=identity[0],
        feature_manifest_sha256=identity[1],
        feature_manifest_file_sha256=identity[2],
        feature_data_path=identity[3],
        feature_data_sha256=identity[4],
        frequency_terciles=identity[5],
        training_count_reference=identity[6],
        slice_membership_reference=identity[7],
        fixed_gate=fixed_gate,
        global_gate_rows=tuple(
            _row_from_coordinate(coordinate, width=width)
            for coordinate in global_coordinates
        ),
        frequency_gate_rows=tuple(
            _row_from_coordinate(coordinate, width=width)
            for coordinate in frequency_coordinates
        ),
    )


def complete_rq5_frequency_surface(
    surface: Rq5GateSurface,
    *,
    selected_gate_hidden_dim: int,
    transferred_horizon_rates: Sequence[TransferredHorizonRate],
) -> Rq5GateSurface:
    if surface.selected_frequency_gate_hidden_dim is not None:
        raise ValueError("RQ5 frequency-gate horizon surface is already complete")
    expected_initial = compile_rq5_gate_surface(
        predecessor=surface.predecessor,
        binding=surface.binding,
    )
    if surface != expected_initial:
        raise ValueError("RQ5 initial gate surface changed before completion")
    _, frequency_spec = _approved_gate_specs()
    first_stage = compile_capacity_first_stage(frequency_spec)
    if not _rows_match_coordinates(surface.frequency_gate_rows, first_stage):
        raise ValueError("RQ5 frequency-gate capacity surface changed")
    followup = compile_capacity_horizon_followup(
        frequency_spec,
        selected_capacity=selected_gate_hidden_dim,
        transferred_horizon_rates=transferred_horizon_rates,
        first_stage=first_stage,
    )
    existing = {row.id: row for row in surface.frequency_gate_rows}
    followup_rows = tuple(
        _row_from_coordinate(
            coordinate, width=surface.selected_history_hidden_dim, existing=existing
        )
        for coordinate in followup
    )
    completed = (*surface.frequency_gate_rows, *followup_rows)
    if (
        len(surface.global_gate_rows) != RQ5_OPPORTUNITY_BUDGET_PER_NEW_FAMILY
        or len(completed) != RQ5_OPPORTUNITY_BUDGET_PER_NEW_FAMILY
    ):
        raise ValueError("RQ5 global and frequency gates do not have equal budgets")
    return replace(
        surface,
        frequency_gate_rows=completed,
        selected_frequency_gate_hidden_dim=selected_gate_hidden_dim,
        transferred_horizon_rates=tuple(
            TransferredHorizonRate(
                row.horizon_epochs,
                row.embedding_learning_rate,
                row.deep_learning_rate,
            )
            for row in followup_rows
        ),
    )


def resolve_rq5_feature_data(*, root: Path, surface: Rq5GateSurface) -> Path:
    from .rq3 import compile_rq3_output_surface, resolve_rq3_feature_data

    predecessor = compile_rq3_output_surface(
        root=root,
        selection_path=Path(surface.binding.selection_path),
        expected_selection_sha256=surface.binding.selection_sha256,
    )
    try:
        rebound = compile_rq5_gate_surface(
            predecessor=predecessor,
            binding=surface.binding,
        )
        expected = (
            rebound
            if surface.selected_frequency_gate_hidden_dim is None
            else complete_rq5_frequency_surface(
                rebound,
                selected_gate_hidden_dim=surface.selected_frequency_gate_hidden_dim,
                transferred_horizon_rates=surface.transferred_horizon_rates,
            )
        )
    except (TypeError, ValueError) as error:
        raise ValueError("RQ5 gate surface changed before launch") from error
    if surface != expected:
        raise ValueError("RQ5 gate surface changed before launch")
    return resolve_rq3_feature_data(root=root, surface=predecessor)


def _validate_binding(
    predecessor: AuthenticatedRq2ContentSelection,
    binding: FrozenRq2ContentBinding,
) -> None:
    path = Path(binding.selection_path)
    if (
        not binding.selection_path
        or path.is_absolute()
        or ".." in path.parts
        or not _valid_sha256(binding.selection_sha256)
        or predecessor.selection_path != binding.selection_path
        or predecessor.selection_sha256 != binding.selection_sha256
    ):
        raise ValueError("RQ5 predecessor selection binding is invalid")
    if (
        not isinstance(binding.selected_source_id, str)
        or not binding.selected_source_id
    ):
        raise ValueError("RQ5 selected fixed-gate source is invalid")


def _selected_fixed_gate_source(
    predecessor: AuthenticatedRq2ContentSelection,
    binding: FrozenRq2ContentBinding,
    *,
    width: int,
) -> _AuthenticatedSource:
    rows = predecessor.rows_by_family.get("rq3_output_learned", ())
    matches = [
        row
        for row in rows
        if row.reused_from == binding.selected_source_id
        and row.authenticated_source is not None
        and row.authenticated_source.source_id == binding.selected_source_id
    ]
    if len(matches) != 1:
        raise ValueError("RQ5 selected fixed-gate source is not uniquely authenticated")
    row = matches[0]
    source = row.authenticated_source
    assert source is not None
    if (
        row.run_name != source.run_name
        or row.batch_size != 512
        or row.seed != 42
        or row.history_hidden_dim != width
        or source.history_hidden_dim != width
        or row.embedding_learning_rate != source.embedding_learning_rate
        or row.deep_learning_rate != source.deep_learning_rate
        or row.horizon_epochs != source.horizon_epochs
        or not _valid_rate(source.embedding_learning_rate)
        or not _valid_rate(source.deep_learning_rate)
        or type(source.horizon_epochs) is not int
        or source.horizon_epochs < 1
    ):
        raise ValueError("RQ5 selected fixed-gate authenticated coordinate changed")
    return source


def _approved_gate_specs() -> tuple[FamilySpec, FamilySpec]:
    specifications = {
        specification.id: specification
        for specification in APPROVED_FAMILY_SPECS
        if specification.id in RQ5_COMPARATOR_FAMILY_IDS[1:]
    }
    if (
        tuple(specifications) != RQ5_COMPARATOR_FAMILY_IDS[1:]
        or specifications["rq5_global_gate"].budget
        != RQ5_OPPORTUNITY_BUDGET_PER_NEW_FAMILY
        or specifications["rq5_global_gate"].capacities
        or specifications["rq5_frequency_gate"].budget
        != RQ5_OPPORTUNITY_BUDGET_PER_NEW_FAMILY
        or specifications["rq5_frequency_gate"].capacities != (4, 8, 16)
    ):
        raise ValueError("approved RQ5 comparator structure or budget changed")
    return specifications["rq5_global_gate"], specifications["rq5_frequency_gate"]


def _feature_identity(
    predecessor: AuthenticatedRq2ContentSelection,
) -> tuple[
    str,
    str,
    str,
    str,
    str,
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    strings = (
        predecessor.feature_manifest_path,
        predecessor.feature_manifest_sha256,
        predecessor.feature_manifest_file_sha256,
        predecessor.feature_data_path,
        predecessor.feature_data_sha256,
    )
    if not all(isinstance(value, str) and value for value in strings) or not all(
        _valid_sha256(value) for value in (strings[1], strings[2], strings[4])
    ):
        raise ValueError("RQ5 predecessor feature identity is invalid")
    values = (
        predecessor.frequency_terciles,
        predecessor.training_count_reference,
        predecessor.slice_membership_reference,
    )
    if (
        not all(isinstance(value, dict) and value for value in values)
        or not _valid_sha256(str(predecessor.training_count_reference.get("sha256")))
        or not _valid_sha256(str(predecessor.slice_membership_reference.get("sha256")))
    ):
        raise ValueError("RQ5 predecessor training-count or slice identity is invalid")
    json.dumps(values, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return (*strings, *(deepcopy(value) for value in values))


def _row_from_coordinate(
    coordinate: SearchCoordinate,
    *,
    width: int,
    existing: Mapping[str, Rq5GateRow] | None = None,
) -> Rq5GateRow:
    if coordinate.family_id == "rq5_global_gate":
        gate: Literal["global", "frequency"] = "global"
        gate_hidden_dim = None
        run_name = (
            f"g3_rq5_global_gate_trial_{coordinate.opportunity_index + 1:02d}_native50m"
        )
    elif coordinate.family_id == "rq5_frequency_gate":
        gate = "frequency"
        gate_hidden_dim = coordinate.capacity
        if type(gate_hidden_dim) is not int or gate_hidden_dim not in {4, 8, 16}:
            raise ValueError("RQ5 frequency-gate capacity is invalid")
        run_name = (
            f"g3_rq5_frequency_gate_width_{gate_hidden_dim}_trial_"
            f"{coordinate.opportunity_index + 1:02d}_native50m"
        )
    else:
        raise ValueError("RQ5 coordinate has an unknown gate family")
    if coordinate.reused_from is not None:
        source = (existing or {}).get(coordinate.reused_from)
        if source is None:
            raise ValueError("RQ5 reused horizon coordinate has no source row")
        run_name = source.run_name
    return Rq5GateRow(
        id=coordinate.id,
        family_id=coordinate.family_id,
        run_name=run_name,
        batch_size=coordinate.batch_size,
        seed=coordinate.seed,
        embedding_learning_rate=coordinate.embedding_learning_rate,
        deep_learning_rate=coordinate.deep_learning_rate,
        horizon_epochs=coordinate.horizon_epochs,
        history_hidden_dim=width,
        content_gate=gate,
        gate_hidden_dim=gate_hidden_dim,
        reused_from=coordinate.reused_from,
    )


def _rows_match_coordinates(
    rows: Sequence[Rq5GateRow], coordinates: Sequence[SearchCoordinate]
) -> bool:
    return len(rows) == len(coordinates) and all(
        (
            row.id,
            row.family_id,
            row.batch_size,
            row.seed,
            row.embedding_learning_rate,
            row.deep_learning_rate,
            row.horizon_epochs,
            row.gate_hidden_dim,
            row.reused_from,
        )
        == (
            coordinate.id,
            coordinate.family_id,
            coordinate.batch_size,
            coordinate.seed,
            coordinate.embedding_learning_rate,
            coordinate.deep_learning_rate,
            coordinate.horizon_epochs,
            coordinate.capacity,
            coordinate.reused_from,
        )
        for row, coordinate in zip(rows, coordinates, strict=True)
    )


def _valid_sha256(value: str) -> bool:
    return bool(_SHA256.fullmatch(value))


def _valid_rate(value: object) -> bool:
    return (
        type(value) in {int, float} and math.isfinite(float(value)) and float(value) > 0
    )
