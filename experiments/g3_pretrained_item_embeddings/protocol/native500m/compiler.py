from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Literal, Sequence

from scipy.stats import qmc

from .constants import (
    BASELINE_ANCHOR,
    PROTOCOL,
    FamilySpec,
    family_spec,
)

if TYPE_CHECKING:
    from .conditional import AuthenticatedResolvedConditionalPredecessor
    from .selection import (
        AuthenticatedSelectedCoordinate,
        BoundaryRequest,
        CandidateResult,
    )


RowStage = Literal[
    "initial",
    "capacity_followup",
    "frequency_followup",
    "boundary",
]


@dataclass(frozen=True)
class SelectedCoordinate:
    source_id: str
    family_id: str
    embedding_learning_rate_text: str
    deep_learning_rate_text: str
    horizon_epochs: int
    capacity: int | None = None

    @classmethod
    def create(
        cls,
        *,
        source_id: str,
        family_id: str,
        embedding_learning_rate: float,
        deep_learning_rate: float,
        horizon_epochs: int,
        capacity: int | None = None,
    ) -> SelectedCoordinate:
        if not source_id or not family_id:
            raise ValueError("selected coordinate identities must be nonempty")
        if type(horizon_epochs) is not int or horizon_epochs < 1:
            raise ValueError("selected horizon must be a positive integer")
        if capacity is not None and (type(capacity) is not int or capacity < 1):
            raise ValueError("selected capacity must be a positive integer")
        return cls(
            source_id=source_id,
            family_id=family_id,
            embedding_learning_rate_text=_canonical_rate(embedding_learning_rate),
            deep_learning_rate_text=_canonical_rate(deep_learning_rate),
            horizon_epochs=horizon_epochs,
            capacity=capacity,
        )

    @property
    def embedding_learning_rate(self) -> float:
        return float(self.embedding_learning_rate_text)

    @property
    def deep_learning_rate(self) -> float:
        return float(self.deep_learning_rate_text)

    @property
    def learning_rate_pair(self) -> tuple[str, str]:
        return self.embedding_learning_rate_text, self.deep_learning_rate_text


@dataclass(frozen=True)
class SearchRow:
    id: str
    family_id: str
    family_code: int
    research_question: str
    predecessor_id: str
    promotion_predecessor_id: str
    manifest_order: int
    stage: RowStage
    batch_size: int
    seed: int
    horizon_epochs: int
    embedding_learning_rate_text: str
    deep_learning_rate_text: str
    anchor_embedding_learning_rate_text: str
    anchor_deep_learning_rate_text: str
    capacity: int | None = None

    @property
    def embedding_learning_rate(self) -> float:
        return float(self.embedding_learning_rate_text)

    @property
    def deep_learning_rate(self) -> float:
        return float(self.deep_learning_rate_text)

    @property
    def learning_rate_pair(self) -> tuple[str, str]:
        return self.embedding_learning_rate_text, self.deep_learning_rate_text

    @property
    def anchor_learning_rate_pair(self) -> tuple[str, str]:
        return (
            self.anchor_embedding_learning_rate_text,
            self.anchor_deep_learning_rate_text,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "family_id": self.family_id,
            "family_code": self.family_code,
            "research_question": self.research_question,
            "predecessor_id": self.predecessor_id,
            "promotion_predecessor_id": self.promotion_predecessor_id,
            "manifest_order": self.manifest_order,
            "stage": self.stage,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "horizon_epochs": self.horizon_epochs,
            "embedding_learning_rate": self.embedding_learning_rate_text,
            "deep_learning_rate": self.deep_learning_rate_text,
            "anchor_embedding_learning_rate": (
                self.anchor_embedding_learning_rate_text
            ),
            "anchor_deep_learning_rate": self.anchor_deep_learning_rate_text,
            "capacity": self.capacity,
        }


def compile_baseline_rows() -> tuple[SearchRow, ...]:
    predecessor = SelectedCoordinate.create(
        source_id=BASELINE_ANCHOR.source_id,
        family_id="g1_aggregate",
        embedding_learning_rate=BASELINE_ANCHOR.embedding_learning_rate,
        deep_learning_rate=BASELINE_ANCHOR.deep_learning_rate,
        horizon_epochs=20,
    )
    return compile_nine_cell_family(family_spec("baseline"), predecessor)


def compile_nine_cell_family(
    spec: FamilySpec,
    predecessor: SelectedCoordinate,
) -> tuple[SearchRow, ...]:
    if spec.design != "nine_cell" or spec.budget != 9:
        raise ValueError(f"{spec.id} is not a nine-cell family")
    _validate_predecessor(spec, predecessor)
    rows: list[SearchRow] = []
    for horizon_epochs in PROTOCOL.horizon_epochs:
        pairs = (
            predecessor.learning_rate_pair,
            *_sobol_pairs(spec, horizon_epochs, predecessor, count=2),
        )
        for embedding_rate, deep_rate in pairs:
            rows.append(
                _row(
                    spec,
                    predecessor,
                    len(rows),
                    horizon_epochs=horizon_epochs,
                    embedding_rate=embedding_rate,
                    deep_rate=deep_rate,
                    stage="initial",
                )
            )
    return tuple(rows)


def compile_capacity_first_stage(
    spec: FamilySpec,
    predecessor: SelectedCoordinate,
) -> tuple[SearchRow, ...]:
    if spec.design != "capacity" or len(spec.capacities) != 3 or spec.budget != 12:
        raise ValueError(f"{spec.id} is not a three-capacity family")
    _validate_predecessor(spec, predecessor)
    pairs = (
        predecessor.learning_rate_pair,
        *_sobol_pairs(spec, 20, predecessor, count=2),
    )
    rows: list[SearchRow] = []
    for capacity in spec.capacities:
        for embedding_rate, deep_rate in pairs:
            rows.append(
                _row(
                    spec,
                    predecessor,
                    len(rows),
                    horizon_epochs=20,
                    embedding_rate=embedding_rate,
                    deep_rate=deep_rate,
                    capacity=capacity,
                    stage="initial",
                )
            )
    return tuple(rows)


def compile_capacity_followup(
    spec: FamilySpec,
    predecessor: SelectedCoordinate,
    selected: SearchRow,
) -> tuple[SearchRow, ...]:
    first_stage = compile_capacity_first_stage(spec, predecessor)
    if selected not in first_stage:
        raise ValueError("selected capacity row is not in the compiled first stage")
    third_pair = _sobol_pairs(spec, 20, predecessor, count=3)[2]
    parameters = (
        (10, selected.learning_rate_pair),
        (40, selected.learning_rate_pair),
        (20, third_pair),
    )
    return tuple(
        _row(
            spec,
            predecessor,
            len(first_stage) + offset,
            horizon_epochs=horizon,
            embedding_rate=pair[0],
            deep_rate=pair[1],
            capacity=selected.capacity,
            stage="capacity_followup",
        )
        for offset, (horizon, pair) in enumerate(parameters)
    )


def compile_rq5_global_rows(
    selected_rq2: SelectedCoordinate,
) -> tuple[SearchRow, ...]:
    spec = family_spec("rq5_global_gate")
    _validate_predecessor(spec, selected_rq2)
    rows: list[SearchRow] = []
    for horizon_epochs in PROTOCOL.horizon_epochs:
        for factor in (0.5, 1.0, 2.0):
            rows.append(
                _row(
                    spec,
                    selected_rq2,
                    len(rows),
                    horizon_epochs=horizon_epochs,
                    embedding_rate=selected_rq2.embedding_learning_rate_text,
                    deep_rate=_canonical_rate(selected_rq2.deep_learning_rate * factor),
                    stage="initial",
                )
            )
    return tuple(rows)


def compile_rq5_frequency_first_stage(
    selected_rq2: SelectedCoordinate,
) -> tuple[SearchRow, ...]:
    spec = family_spec("rq5_frequency_gate")
    _validate_predecessor(spec, selected_rq2)
    rows: list[SearchRow] = []
    for capacity in spec.capacities:
        for factor in (0.5, 1.0, 2.0):
            rows.append(
                _row(
                    spec,
                    selected_rq2,
                    len(rows),
                    horizon_epochs=20,
                    embedding_rate=selected_rq2.embedding_learning_rate_text,
                    deep_rate=_canonical_rate(selected_rq2.deep_learning_rate * factor),
                    capacity=capacity,
                    stage="initial",
                )
            )
    return tuple(rows)


def compile_rq5_frequency_followup(
    selected_rq2: SelectedCoordinate,
    selected: SearchRow,
) -> tuple[SearchRow, ...]:
    spec = family_spec("rq5_frequency_gate")
    first_stage = compile_rq5_frequency_first_stage(selected_rq2)
    if selected not in first_stage:
        raise ValueError("selected frequency row is not in the compiled first stage")
    return tuple(
        _row(
            spec,
            selected_rq2,
            len(first_stage) + offset,
            horizon_epochs=horizon,
            embedding_rate=selected.embedding_learning_rate_text,
            deep_rate=selected.deep_learning_rate_text,
            capacity=selected.capacity,
            stage="frequency_followup",
        )
        for offset, horizon in enumerate((10, 40))
    )


def compile_boundary_rows(
    winner: CandidateResult,
    existing_rows: Sequence[SearchRow],
    *,
    existing_results: Sequence[CandidateResult],
    predecessor: (
        AuthenticatedSelectedCoordinate
        | AuthenticatedResolvedConditionalPredecessor
        | None
    ) = None,
    requests: Sequence[BoundaryRequest],
    round_number: int = 1,
) -> tuple[SearchRow, ...]:
    if round_number != 1:
        raise ValueError("only one boundary round is approved")
    if not existing_rows or winner.row not in existing_rows:
        raise ValueError("boundary winner must belong to the existing family rows")
    if len({row.family_id for row in existing_rows}) != 1:
        raise ValueError("boundary rows must belong to one family")
    if any(row.stage == "boundary" for row in existing_rows):
        raise ValueError("a second boundary round requires new approval")
    from .selection import (
        authenticate_complete_family_ledger,
        required_boundary_extensions,
    )

    authenticated_winner = authenticate_complete_family_ledger(
        existing_results,
        existing_rows,
        predecessor=predecessor,
    )
    if authenticated_winner != winner:
        raise ValueError("boundary winner is not the selected family-ledger winner")

    required_sequence = required_boundary_extensions(winner, existing_rows)
    if tuple(requests) != required_sequence:
        raise ValueError(
            "compile_boundary_rows requires the exact canonical boundary request order"
        )
    spec = family_spec(winner.row.family_id)
    if spec.id == "baseline":
        selected_predecessor = SelectedCoordinate.create(
            source_id=BASELINE_ANCHOR.source_id,
            family_id="g1_aggregate",
            embedding_learning_rate=BASELINE_ANCHOR.embedding_learning_rate,
            deep_learning_rate=BASELINE_ANCHOR.deep_learning_rate,
            horizon_epochs=20,
        )
    else:
        if predecessor is None:
            raise ValueError(
                "non-baseline boundary requires an authenticated predecessor"
            )
        selected_predecessor = predecessor.coordinate
    rows: list[SearchRow] = []
    next_order = max(row.manifest_order for row in existing_rows) + 1
    seen_axes: set[str] = set()
    for request in required_sequence:
        if request.axis in seen_axes:
            raise ValueError(f"duplicate boundary request for {request.axis}")
        seen_axes.add(request.axis)
        if request.axis in {"embedding_learning_rate", "deep_learning_rate"}:
            factors = (
                (0.5, 0.25, 0.125) if request.direction == "low" else (2.0, 4.0, 8.0)
            )
            for factor in factors:
                embedding_rate = winner.row.embedding_learning_rate_text
                deep_rate = winner.row.deep_learning_rate_text
                if request.axis == "embedding_learning_rate":
                    embedding_rate = _canonical_rate(
                        winner.row.embedding_learning_rate * factor
                    )
                else:
                    deep_rate = _canonical_rate(winner.row.deep_learning_rate * factor)
                rows.append(
                    _row(
                        spec,
                        selected_predecessor,
                        next_order + len(rows),
                        horizon_epochs=winner.row.horizon_epochs,
                        embedding_rate=embedding_rate,
                        deep_rate=deep_rate,
                        capacity=winner.row.capacity,
                        stage="boundary",
                    )
                )
        elif request.axis == "capacity":
            capacity = int(request.value)
            if spec.id == "rq5_frequency_gate":
                for factor in (0.5, 1.0, 2.0):
                    rows.append(
                        _row(
                            spec,
                            selected_predecessor,
                            next_order + len(rows),
                            horizon_epochs=winner.row.horizon_epochs,
                            embedding_rate=winner.row.embedding_learning_rate_text,
                            deep_rate=_canonical_rate(
                                selected_predecessor.deep_learning_rate * factor
                            ),
                            capacity=capacity,
                            stage="boundary",
                        )
                    )
            else:
                pairs = [winner.row.learning_rate_pair]
                for pair in _sobol_pairs(
                    spec,
                    winner.row.horizon_epochs,
                    selected_predecessor,
                    count=4,
                ):
                    if pair not in pairs:
                        pairs.append(pair)
                    if len(pairs) == 3:
                        break
                if len(pairs) != 3:
                    raise RuntimeError("could not compile three unique capacity probes")
                for embedding_rate, deep_rate in pairs:
                    rows.append(
                        _row(
                            spec,
                            selected_predecessor,
                            next_order + len(rows),
                            horizon_epochs=winner.row.horizon_epochs,
                            embedding_rate=embedding_rate,
                            deep_rate=deep_rate,
                            capacity=capacity,
                            stage="boundary",
                        )
                    )
        elif request.axis == "horizon_epochs":
            rows.append(
                _row(
                    spec,
                    selected_predecessor,
                    next_order + len(rows),
                    horizon_epochs=int(request.value),
                    embedding_rate=winner.row.embedding_learning_rate_text,
                    deep_rate=winner.row.deep_learning_rate_text,
                    capacity=winner.row.capacity,
                    stage="boundary",
                )
            )
        else:
            raise ValueError(f"unsupported boundary axis {request.axis!r}")
    _validate_hard_domain(rows, selected_predecessor, spec)
    return tuple(rows)


def _validate_predecessor(spec: FamilySpec, predecessor: SelectedCoordinate) -> None:
    if spec.id == "baseline":
        expected_pair = (
            _canonical_rate(BASELINE_ANCHOR.embedding_learning_rate),
            _canonical_rate(BASELINE_ANCHOR.deep_learning_rate),
        )
        if (
            predecessor.source_id != BASELINE_ANCHOR.source_id
            or predecessor.family_id != "g1_aggregate"
            or predecessor.learning_rate_pair != expected_pair
            or predecessor.capacity is not None
        ):
            raise ValueError("baseline must use the exact selected G1 anchor")
        return
    if predecessor.family_id != spec.search_predecessor_id:
        raise ValueError(
            f"{spec.id} requires selected predecessor {spec.search_predecessor_id!r}"
        )


def _row(
    spec: FamilySpec,
    predecessor: SelectedCoordinate,
    manifest_order: int,
    *,
    horizon_epochs: int,
    embedding_rate: float | str,
    deep_rate: float | str,
    stage: RowStage,
    capacity: int | None = None,
) -> SearchRow:
    embedding_text = _canonical_rate(embedding_rate)
    deep_text = _canonical_rate(deep_rate)
    return SearchRow(
        id=f"{spec.id}:{manifest_order + 1:02d}",
        family_id=spec.id,
        family_code=spec.code,
        research_question=spec.research_question,
        predecessor_id=predecessor.source_id,
        promotion_predecessor_id=spec.promotion_predecessor_id,
        manifest_order=manifest_order,
        stage=stage,
        batch_size=PROTOCOL.batch_size,
        seed=PROTOCOL.seed,
        horizon_epochs=horizon_epochs,
        embedding_learning_rate_text=embedding_text,
        deep_learning_rate_text=deep_text,
        anchor_embedding_learning_rate_text=(predecessor.embedding_learning_rate_text),
        anchor_deep_learning_rate_text=predecessor.deep_learning_rate_text,
        capacity=capacity,
    )


def _sobol_pairs(
    spec: FamilySpec,
    horizon_epochs: int,
    predecessor: SelectedCoordinate,
    *,
    count: int,
) -> tuple[tuple[str, str], ...]:
    if count < 1:
        return ()
    exponent = math.ceil(math.log2(count))
    points = qmc.Sobol(
        d=2,
        scramble=True,
        seed=300000 + 100 * spec.code + horizon_epochs,
    ).random_base2(m=exponent)
    anchors = (
        predecessor.embedding_learning_rate,
        predecessor.deep_learning_rate,
    )
    result = []
    for point in points[:count]:
        rates = []
        for anchor, coordinate in zip(anchors, point, strict=True):
            low = anchor / PROTOCOL.local_learning_rate_factor
            high = anchor * PROTOCOL.local_learning_rate_factor
            value = math.exp(
                math.log(low) + float(coordinate) * (math.log(high) - math.log(low))
            )
            rates.append(_canonical_rate(value))
        result.append((rates[0], rates[1]))
    return tuple(result)


def _validate_hard_domain(
    rows: Sequence[SearchRow],
    predecessor: SelectedCoordinate,
    spec: FamilySpec,
) -> None:
    low_embedding = (
        predecessor.embedding_learning_rate / PROTOCOL.hard_learning_rate_factor
    )
    high_embedding = (
        predecessor.embedding_learning_rate * PROTOCOL.hard_learning_rate_factor
    )
    low_deep = predecessor.deep_learning_rate / PROTOCOL.hard_learning_rate_factor
    high_deep = predecessor.deep_learning_rate * PROTOCOL.hard_learning_rate_factor
    for row in rows:
        if not low_deep <= row.deep_learning_rate <= high_deep:
            raise ValueError(f"{spec.id} deep LR boundary exceeds the hard domain")
        if (
            spec.id not in {"rq5_global_gate", "rq5_frequency_gate"}
            and not low_embedding <= row.embedding_learning_rate <= high_embedding
        ):
            raise ValueError(f"{spec.id} embedding LR boundary exceeds the hard domain")
        if spec.id in {"rq5_global_gate", "rq5_frequency_gate"} and (
            row.embedding_learning_rate_text != predecessor.embedding_learning_rate_text
        ):
            raise ValueError(f"{spec.id} must keep the selected RQ2 embedding LR")


def _canonical_rate(value: float | str) -> str:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError("learning rates must be finite and positive")
    return format(parsed, ".17g")
