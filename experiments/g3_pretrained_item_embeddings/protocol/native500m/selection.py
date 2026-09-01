from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Literal, Sequence

from .compiler import SearchRow, SelectedCoordinate
from .constants import PROTOCOL_SHA256, FamilySpec, family_spec

if TYPE_CHECKING:
    from .conditional import AuthenticatedResolvedConditionalPredecessor


BoundaryAxis = Literal[
    "embedding_learning_rate",
    "deep_learning_rate",
    "capacity",
    "horizon_epochs",
]
BoundaryDirection = Literal["low", "high"]


@dataclass(frozen=True)
class CandidateResult:
    row: SearchRow
    recall_at_100: float
    ndcg_at_100: float
    best_epoch: int
    epochs_trained: int

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(value) for value in (self.recall_at_100, self.ndcg_at_100)
        ):
            raise ValueError("selection metrics must be finite")
        if not all(
            0.0 <= value <= 1.0 for value in (self.recall_at_100, self.ndcg_at_100)
        ):
            raise ValueError("selection metrics must be between zero and one")
        if (
            type(self.best_epoch) is not int
            or not 1 <= self.best_epoch <= self.row.horizon_epochs
        ):
            raise ValueError("best epoch must be inside the declared horizon")
        if self.epochs_trained != self.row.horizon_epochs:
            raise ValueError("annealed candidates must complete their declared horizon")


@dataclass(frozen=True)
class BoundaryRequest:
    axis: BoundaryAxis
    direction: BoundaryDirection
    value: float | int


_AUTHENTICATION_SEAL = object()


class AuthenticatedSelectedCoordinate:
    __slots__ = (
        "_conditional_family_id",
        "_conditional_generation",
        "_conditional_state_identity",
        "_protocol_sha256",
        "_selected_result",
    )

    def __init__(
        self,
        *,
        selected_result: CandidateResult,
        protocol_sha256: str,
        conditional_state_identity: str | None = None,
        conditional_generation: int | None = None,
        conditional_family_id: str | None = None,
        _seal: object,
    ) -> None:
        if _seal is not _AUTHENTICATION_SEAL:
            raise ValueError("selected coordinate was not produced by authentication")
        object.__setattr__(self, "_selected_result", selected_result)
        object.__setattr__(self, "_protocol_sha256", protocol_sha256)
        object.__setattr__(
            self,
            "_conditional_state_identity",
            conditional_state_identity,
        )
        object.__setattr__(self, "_conditional_generation", conditional_generation)
        object.__setattr__(self, "_conditional_family_id", conditional_family_id)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("authenticated selected coordinates are immutable")

    @property
    def selected_result(self) -> CandidateResult:
        return self._selected_result

    @property
    def protocol_sha256(self) -> str:
        return self._protocol_sha256

    @property
    def conditional_state_identity(self) -> str | None:
        return self._conditional_state_identity

    @property
    def conditional_generation(self) -> int | None:
        return self._conditional_generation

    @property
    def conditional_family_id(self) -> str | None:
        return self._conditional_family_id

    @property
    def coordinate(self) -> SelectedCoordinate:
        row = self.selected_result.row
        return SelectedCoordinate.create(
            source_id=row.id,
            family_id=row.family_id,
            embedding_learning_rate=row.embedding_learning_rate_text,
            deep_learning_rate=row.deep_learning_rate_text,
            horizon_epochs=row.horizon_epochs,
            capacity=row.capacity,
        )


def authenticate_selected_coordinate(
    results: Sequence[CandidateResult],
    *,
    expected_rows: Sequence[SearchRow],
    predecessor: (
        AuthenticatedSelectedCoordinate
        | AuthenticatedResolvedConditionalPredecessor
        | None
    ) = None,
) -> AuthenticatedSelectedCoordinate:
    winner = select_winner(
        results,
        expected_rows=expected_rows,
        predecessor=predecessor,
    )
    from .conditional import AuthenticatedResolvedConditionalPredecessor

    resolved = (
        predecessor
        if isinstance(predecessor, AuthenticatedResolvedConditionalPredecessor)
        else None
    )
    return AuthenticatedSelectedCoordinate(
        selected_result=winner,
        protocol_sha256=PROTOCOL_SHA256,
        conditional_state_identity=(
            None if resolved is None else resolved.compatibility_state.identity_sha256
        ),
        conditional_generation=(
            None if resolved is None else resolved.compatibility_state.generation
        ),
        conditional_family_id=(None if resolved is None else resolved.target_family_id),
        _seal=_AUTHENTICATION_SEAL,
    )


def select_preliminary_winner(
    results: Sequence[CandidateResult],
    *,
    expected_rows: Sequence[SearchRow],
    predecessor: (
        AuthenticatedSelectedCoordinate
        | AuthenticatedResolvedConditionalPredecessor
        | None
    ) = None,
) -> CandidateResult:
    result_by_id = _validate_result_coverage(results, expected_rows)
    if any(row.stage == "boundary" for row in expected_rows):
        raise ValueError("preliminary selection accepts only non-boundary rows")
    _authenticate_family_rows(expected_rows, result_by_id, predecessor)
    return _choose_winner(results)


def select_winner(
    results: Sequence[CandidateResult],
    *,
    expected_rows: Sequence[SearchRow],
    predecessor: (
        AuthenticatedSelectedCoordinate
        | AuthenticatedResolvedConditionalPredecessor
        | None
    ) = None,
) -> CandidateResult:
    result_by_id = _validate_result_coverage(results, expected_rows)
    family_rows, boundary_rows = _split_family_and_boundary_rows(expected_rows)
    spec = family_spec(family_rows[0].family_id)
    if len(family_rows) != spec.budget:
        raise ValueError("final selection requires the exact metric-selected followup")
    _authenticate_family_rows(family_rows, result_by_id, predecessor)
    family_results = tuple(result_by_id[row.id] for row in family_rows)
    family_winner = _choose_winner(family_results)
    required_requests = required_boundary_extensions(family_winner, family_rows)
    if required_requests and not boundary_rows:
        raise ValueError("final selection is missing the required boundary block")
    if boundary_rows:
        from .compiler import compile_boundary_rows

        canonical_boundary = compile_boundary_rows(
            family_winner,
            family_rows,
            existing_results=family_results,
            predecessor=predecessor,
            requests=required_requests,
        )
        if boundary_rows != canonical_boundary:
            raise ValueError(
                "expected rows are not the compiler-generated boundary ledger"
            )
    winner = _choose_winner(results)
    if boundary_rows and required_boundary_extensions(winner, expected_rows):
        raise ValueError("winner remains unresolved after one boundary round")
    return winner


def authenticate_complete_family_ledger(
    results: Sequence[CandidateResult],
    rows: Sequence[SearchRow],
    *,
    predecessor: (
        AuthenticatedSelectedCoordinate
        | AuthenticatedResolvedConditionalPredecessor
        | None
    ) = None,
) -> CandidateResult:
    if not rows:
        raise ValueError(
            "boundary compilation requires a complete canonical family ledger"
        )
    if any(row.stage == "boundary" for row in rows):
        raise ValueError(
            "complete canonical family ledger cannot contain boundary rows"
        )
    spec = family_spec(rows[0].family_id)
    if len(rows) != spec.budget:
        raise ValueError(
            "boundary compilation requires a complete canonical family ledger"
        )
    return select_preliminary_winner(
        results,
        expected_rows=rows,
        predecessor=predecessor,
    )


def _choose_winner(results: Sequence[CandidateResult]) -> CandidateResult:
    return min(
        results,
        key=lambda result: (
            -result.recall_at_100,
            -result.ndcg_at_100,
            result.row.manifest_order,
        ),
    )


def _validate_result_coverage(
    results: Sequence[CandidateResult],
    rows: Sequence[SearchRow],
) -> dict[str, CandidateResult]:
    if not results or not rows:
        raise ValueError("selection requires results and an expected-row ledger")
    families = {
        *(result.row.family_id for result in results),
        *(row.family_id for row in rows),
    }
    if len(families) != 1:
        raise ValueError("selection candidates must belong to the same family")
    expected_by_id = {row.id: row for row in rows}
    if len(expected_by_id) != len(rows):
        raise ValueError("expected-row ledger contains duplicate rows")
    result_by_id = {result.row.id: result for result in results}
    if len(result_by_id) != len(results):
        raise ValueError("selection candidates contain duplicate rows")
    if set(result_by_id) != set(expected_by_id):
        raise ValueError("results do not cover the complete expected-row ledger")
    if any(
        result_by_id[row_id].row != expected_row
        for row_id, expected_row in expected_by_id.items()
    ):
        raise ValueError("selection result rows differ from the expected-row ledger")
    return result_by_id


def _split_family_and_boundary_rows(
    rows: Sequence[SearchRow],
) -> tuple[tuple[SearchRow, ...], tuple[SearchRow, ...]]:
    if [row.manifest_order for row in rows] != list(range(len(rows))):
        raise ValueError("expected-row ledger manifest order is incomplete")
    boundary_start = next(
        (index for index, row in enumerate(rows) if row.stage == "boundary"),
        len(rows),
    )
    if any(row.stage != "boundary" for row in rows[boundary_start:]):
        raise ValueError("boundary rows must be a trailing ledger block")
    if boundary_start == 0:
        raise ValueError("expected-row ledger does not contain family rows")
    return tuple(rows[:boundary_start]), tuple(rows[boundary_start:])


def _authenticate_family_rows(
    rows: Sequence[SearchRow],
    result_by_id: dict[str, CandidateResult],
    predecessor: (
        AuthenticatedSelectedCoordinate
        | AuthenticatedResolvedConditionalPredecessor
        | None
    ),
) -> None:
    if [row.manifest_order for row in rows] != list(range(len(rows))):
        raise ValueError("expected-row ledger manifest order is incomplete")
    canonical_family_rows = _compile_canonical_family_rows(
        rows,
        result_by_id,
        predecessor,
    )
    if tuple(rows) != canonical_family_rows:
        raise ValueError("expected rows are not the compiler-generated family ledger")


def _compile_canonical_family_rows(
    rows: Sequence[SearchRow],
    result_by_id: dict[str, CandidateResult],
    authenticated_predecessor: (
        AuthenticatedSelectedCoordinate
        | AuthenticatedResolvedConditionalPredecessor
        | None
    ),
) -> tuple[SearchRow, ...]:
    if len(rows) < 9:
        raise ValueError("expected rows do not contain a complete family stage")
    spec = family_spec(rows[0].family_id)
    from .compiler import (
        compile_baseline_rows,
        compile_capacity_first_stage,
        compile_capacity_followup,
        compile_nine_cell_family,
        compile_rq5_frequency_first_stage,
        compile_rq5_frequency_followup,
        compile_rq5_global_rows,
    )

    if spec.id == "baseline":
        if authenticated_predecessor is not None:
            raise ValueError("baseline does not accept an authenticated predecessor")
        initial_rows = compile_baseline_rows()
    else:
        predecessor = _validate_authenticated_predecessor(
            spec,
            authenticated_predecessor,
        )
        first = rows[0]
        if (
            first.predecessor_id != predecessor.source_id
            or first.anchor_learning_rate_pair != predecessor.learning_rate_pair
        ):
            raise ValueError("ledger differs from its authenticated predecessor")
        if spec.design == "nine_cell":
            initial_rows = compile_nine_cell_family(spec, predecessor)
        elif spec.design == "capacity":
            initial_rows = compile_capacity_first_stage(spec, predecessor)
        elif spec.design == "rq5_global":
            initial_rows = compile_rq5_global_rows(predecessor)
        elif spec.design == "rq5_frequency":
            initial_rows = compile_rq5_frequency_first_stage(predecessor)
        else:
            raise ValueError(f"unsupported family design {spec.design!r}")

    if tuple(rows[:9]) != initial_rows:
        raise ValueError("expected rows are not the compiler-generated initial ledger")
    if spec.design == "capacity":
        if len(rows) == 9:
            return initial_rows
        if len(rows) != spec.budget:
            raise ValueError("expected rows do not contain a complete family stage")
        initial_results = tuple(result_by_id[row.id] for row in initial_rows)
        winner = _choose_winner(initial_results)
        return (
            *initial_rows,
            *compile_capacity_followup(spec, predecessor, winner.row),
        )
    if spec.design == "rq5_frequency":
        if len(rows) == 9:
            return initial_rows
        if len(rows) != spec.budget:
            raise ValueError("expected rows do not contain a complete family stage")
        initial_results = tuple(result_by_id[row.id] for row in initial_rows)
        winner = _choose_winner(initial_results)
        return (
            *initial_rows,
            *compile_rq5_frequency_followup(predecessor, winner.row),
        )
    if len(rows) != spec.budget:
        raise ValueError("expected rows do not contain a complete family stage")
    return initial_rows


def _validate_authenticated_predecessor(
    spec: FamilySpec,
    authenticated: (
        AuthenticatedSelectedCoordinate
        | AuthenticatedResolvedConditionalPredecessor
        | None
    ),
) -> SelectedCoordinate:
    from .conditional import AuthenticatedResolvedConditionalPredecessor

    if spec.conditional:
        if (
            not isinstance(authenticated, AuthenticatedResolvedConditionalPredecessor)
            or authenticated.protocol_sha256 != PROTOCOL_SHA256
            or authenticated.target_family_id != spec.id
            or authenticated.synthetic_role != spec.search_predecessor_id
        ):
            raise ValueError(
                "conditional family requires its authenticated resolved predecessor"
            )
        return authenticated.coordinate
    if (
        not isinstance(authenticated, AuthenticatedSelectedCoordinate)
        or authenticated.protocol_sha256 != PROTOCOL_SHA256
    ):
        raise ValueError("non-baseline family requires an authenticated predecessor")
    coordinate = authenticated.coordinate
    if coordinate.family_id != spec.search_predecessor_id:
        raise ValueError(
            "authenticated predecessor has the wrong search-predecessor family"
        )
    return coordinate


def required_boundary_extensions(
    winner: CandidateResult,
    family_rows: Sequence[SearchRow],
) -> tuple[BoundaryRequest, ...]:
    if not family_rows or winner.row not in family_rows:
        raise ValueError("boundary winner must belong to the family rows")
    if {row.family_id for row in family_rows} != {winner.row.family_id}:
        raise ValueError("boundary inspection requires one family")
    spec = family_spec(winner.row.family_id)
    requests: list[BoundaryRequest] = []
    if spec.id not in {"rq5_global_gate", "rq5_frequency_gate"}:
        embedding_request = _rate_boundary(
            "embedding_learning_rate",
            winner.row.embedding_learning_rate,
            [row.embedding_learning_rate for row in family_rows],
        )
        if embedding_request is not None:
            requests.append(embedding_request)
    deep_request = _rate_boundary(
        "deep_learning_rate",
        winner.row.deep_learning_rate,
        [row.deep_learning_rate for row in family_rows],
    )
    if deep_request is not None:
        requests.append(deep_request)
    capacity_request = _capacity_boundary(spec.id, winner.row.capacity, family_rows)
    if capacity_request is not None:
        requests.append(capacity_request)
    tested_horizons = {row.horizon_epochs for row in family_rows}
    if (
        winner.row.horizon_epochs >= 40
        and winner.row.horizon_epochs == max(tested_horizons)
        and winner.best_epoch == winner.row.horizon_epochs
    ):
        requests.append(
            BoundaryRequest(
                "horizon_epochs",
                "high",
                winner.row.horizon_epochs + 20,
            )
        )
    return tuple(requests)


def _rate_boundary(
    axis: Literal["embedding_learning_rate", "deep_learning_rate"],
    selected: float,
    values: Sequence[float],
) -> BoundaryRequest | None:
    unique = sorted(set(values))
    if len(unique) < 2:
        return None
    if selected == unique[0]:
        return BoundaryRequest(axis, "low", selected)
    if selected == unique[-1]:
        return BoundaryRequest(axis, "high", selected)
    return None


def _capacity_boundary(
    family_id: str,
    selected: int | None,
    rows: Sequence[SearchRow],
) -> BoundaryRequest | None:
    if selected is None:
        return None
    tested = sorted({row.capacity for row in rows if row.capacity is not None})
    if len(tested) < 2:
        return None
    if family_id == "rq2_content_concat":
        low_extension, high_extension = 32, 512
    elif family_id in {"rq4_artist", "rq4_album", "rq4_artist_album"}:
        low_extension, high_extension = None, 128
    elif family_id == "rq5_frequency_gate":
        low_extension, high_extension = 16, 128
    else:
        return None
    if selected == tested[0] and low_extension is not None:
        return BoundaryRequest("capacity", "low", low_extension)
    if selected == tested[-1] and high_extension is not None:
        return BoundaryRequest("capacity", "high", high_extension)
    return None
