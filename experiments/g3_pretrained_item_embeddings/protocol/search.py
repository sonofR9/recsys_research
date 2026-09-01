from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Literal, Mapping, Sequence

from scipy.stats import qmc

from .constants import APPROVED_PROTOCOL


Design = Literal["direct", "capacity", "control"]


@dataclass(frozen=True)
class FamilySpec:
    id: str
    research_question: str
    budget: int
    design: Design = "direct"
    capacities: tuple[int, ...] = ()
    conditional: bool = False


@dataclass(frozen=True)
class ReusableCoordinate:
    source_id: str
    embedding_learning_rate: float
    deep_learning_rate: float
    horizon_epochs: int
    capacity: int | None = None


@dataclass(frozen=True)
class TransferredHorizonRate:
    horizon_epochs: int
    embedding_learning_rate: float
    deep_learning_rate: float


@dataclass(frozen=True)
class SearchCoordinate:
    id: str
    family_id: str
    research_question: str
    opportunity_index: int
    batch_size: int
    seed: int
    embedding_learning_rate: float
    deep_learning_rate: float
    horizon_epochs: int
    capacity: int | None
    conditional: bool
    role: Literal["search", "transfer_check", "horizon_probe"]
    reused_from: str | None = None

    @property
    def physical_id(self) -> str:
        return self.reused_from or self.id


@dataclass(frozen=True)
class CompiledSearch:
    initial: tuple[SearchCoordinate, ...]
    conditional: tuple[SearchCoordinate, ...]

    @property
    def initial_opportunity_count(self) -> int:
        return len(self.initial)

    @property
    def conditional_opportunity_count(self) -> int:
        return len(self.conditional)

    @property
    def maximum_opportunity_count(self) -> int:
        return self.initial_opportunity_count + self.conditional_opportunity_count

    @property
    def physical_coordinates(self) -> tuple[SearchCoordinate, ...]:
        unique: dict[str, SearchCoordinate] = {}
        for coordinate in (*self.initial, *self.conditional):
            unique.setdefault(coordinate.physical_id, coordinate)
        return tuple(unique.values())

    @property
    def new_coordinates(self) -> tuple[SearchCoordinate, ...]:
        return tuple(
            coordinate
            for coordinate in (*self.initial, *self.conditional)
            if coordinate.reused_from is None
        )


APPROVED_FAMILY_SPECS = (
    FamilySpec("untied_control", "control", 10, "control"),
    FamilySpec("rq1_content_input", "rq1", 9),
    FamilySpec("rq2_content_concat", "rq2", 12, "capacity", (64, 128, 256)),
    FamilySpec(
        "rq2_id_only_densenet", "rq2", 12, "capacity", (128, 255, 510)
    ),
    FamilySpec("rq3_output_learned", "rq3", 9),
    FamilySpec("rq3_output_frozen_content", "rq3", 9),
    FamilySpec("rq3_output_trainable_content", "rq3", 9),
    FamilySpec("rq3_output_learned_frozen_content", "rq3", 9),
    FamilySpec("rq3_output_learned_trainable_content", "rq3", 9),
    FamilySpec("rq4_artist", "rq4", 12, "capacity", (16, 32, 64)),
    FamilySpec("rq4_album", "rq4", 12, "capacity", (16, 32, 64)),
    FamilySpec("rq4_artist_album", "rq4", 12, "capacity", (16, 32, 64)),
    FamilySpec("rq4_extra_item_id", "rq4", 12),
    FamilySpec("rq5_global_gate", "rq5", 12),
    FamilySpec("rq5_frequency_gate", "rq5", 12, "capacity", (4, 8, 16)),
    FamilySpec("size_500m_tied_baseline", "dataset_size", 9),
    FamilySpec("size_500m_frozen_treatment", "dataset_size", 9),
)

CONDITIONAL_FAMILY_SPECS = (
    FamilySpec("bridge_rq3_output", "aggregate", 9, conditional=True),
    FamilySpec("bridge_rq4_metadata", "aggregate", 9, conditional=True),
    FamilySpec("aggregate", "aggregate", 9, conditional=True),
)


def compile_family(
    spec: FamilySpec,
    *,
    selected_capacity: int | None = None,
    transfer_accepted: bool = True,
    reusable: Sequence[ReusableCoordinate] = (),
    transferred_horizon_rates: Sequence[TransferredHorizonRate] = (),
    allow_reusable_outside_search_space: bool = False,
) -> tuple[SearchCoordinate, ...]:
    if spec.id == "rq4_extra_item_id":
        raise ValueError("rq4_extra_item_id requires selected metadata predecessor opportunities")
    if spec.capacities:
        if reusable:
            raise ValueError(
                f"{spec.id} capacity cells require explicit transferred horizon rates"
            )
        if transfer_accepted:
            if selected_capacity is not None or transferred_horizon_rates:
                raise ValueError(
                    f"{spec.id} capacity follow-up must use the explicit staged compiler"
                )
            return compile_capacity_first_stage(spec)
        if selected_capacity is not None:
            raise ValueError(
                f"{spec.id} rejected transfer must not preselect a capacity"
            )
        if transferred_horizon_rates:
            raise ValueError(
                f"{spec.id} rejected transfer must not provide transferred horizon rates"
            )
        return _compile_balanced_capacity_family(spec)
    elif selected_capacity is not None:
        raise ValueError(f"{spec.id} has no capacity axis")
    if transferred_horizon_rates:
        raise ValueError(f"{spec.id} has no capacity-stage horizon probes")

    ordered_reusable = sorted(reusable, key=lambda coordinate: coordinate.source_id)
    if len(ordered_reusable) > spec.budget:
        raise ValueError(f"{spec.id} has more reusable cells than opportunities")
    if len({coordinate.source_id for coordinate in ordered_reusable}) != len(
        ordered_reusable
    ):
        raise ValueError(f"{spec.id} has duplicate reusable source IDs")

    result: list[SearchCoordinate] = []
    signatures: set[tuple[float, float, int, int | None]] = set()
    for reusable_coordinate in ordered_reusable:
        _validate_reusable(
            spec,
            reusable_coordinate,
            selected_capacity,
            allow_outside_search_space=allow_reusable_outside_search_space,
        )
        signature = _signature(reusable_coordinate)
        if signature in signatures:
            raise ValueError(f"{spec.id} has duplicate reusable coordinates")
        signatures.add(signature)
        result.append(
            _coordinate(
                spec,
                opportunity_index=len(result),
                embedding_learning_rate=reusable_coordinate.embedding_learning_rate,
                deep_learning_rate=reusable_coordinate.deep_learning_rate,
                horizon_epochs=reusable_coordinate.horizon_epochs,
                capacity=reusable_coordinate.capacity,
                reused_from=reusable_coordinate.source_id,
            )
        )

    candidates = _candidate_parameters(spec)
    for embedding_rate, deep_rate, horizon, capacity in candidates:
        if len(result) == spec.budget:
            break
        signature = (embedding_rate, deep_rate, horizon, capacity)
        if signature in signatures:
            continue
        signatures.add(signature)
        result.append(
            _coordinate(
                spec,
                opportunity_index=len(result),
                embedding_learning_rate=embedding_rate,
                deep_learning_rate=deep_rate,
                horizon_epochs=horizon,
                capacity=capacity,
            )
        )
    if len(result) != spec.budget:
        raise ValueError(f"could not compile {spec.budget} unique cells for {spec.id}")
    return tuple(result)


def compile_approved_search(
    *,
    selected_capacities: Mapping[str, int],
    transfer_accepted: bool = True,
    reusable_by_family: Mapping[str, Sequence[ReusableCoordinate]] | None = None,
    transferred_horizon_rates: Mapping[
        str, Sequence[TransferredHorizonRate]
    ] | None = None,
    rq4_extra_id_predecessor: Sequence[SearchCoordinate] = (),
) -> CompiledSearch:
    reusable_by_family = reusable_by_family or {}
    transferred_horizon_rates = transferred_horizon_rates or {}
    known = {spec.id for spec in (*APPROVED_FAMILY_SPECS, *CONDITIONAL_FAMILY_SPECS)}
    capacity_families = {
        spec.id
        for spec in (*APPROVED_FAMILY_SPECS, *CONDITIONAL_FAMILY_SPECS)
        if spec.capacities
    }
    unknown_reuse = set(reusable_by_family) - known
    unknown_capacities = set(selected_capacities) - known
    unknown_horizon_rates = set(transferred_horizon_rates) - capacity_families
    if unknown_reuse or unknown_capacities or unknown_horizon_rates:
        raise ValueError(
            "unknown family IDs: "
            f"{sorted(unknown_reuse | unknown_capacities | unknown_horizon_rates)}"
        )

    def compile_specs(specs: Sequence[FamilySpec]) -> tuple[SearchCoordinate, ...]:
        result = []
        for spec in specs:
            if spec.id == "rq4_extra_item_id":
                result.extend(
                    compile_rq4_extra_id_control(
                        spec,
                        predecessor=rq4_extra_id_predecessor,
                    )
                )
            elif spec.capacities and transfer_accepted:
                first_stage = compile_capacity_first_stage(spec)
                selected_capacity = selected_capacities.get(spec.id)
                if selected_capacity is None:
                    raise ValueError(f"{spec.id} requires a selected capacity")
                followup = compile_capacity_horizon_followup(
                    spec,
                    selected_capacity=selected_capacity,
                    transferred_horizon_rates=transferred_horizon_rates.get(spec.id, ()),
                    first_stage=first_stage,
                )
                result.extend((*first_stage, *followup))
            else:
                result.extend(
                    compile_family(
                        spec,
                        selected_capacity=selected_capacities.get(spec.id),
                        transfer_accepted=transfer_accepted,
                        reusable=reusable_by_family.get(spec.id, ()),
                        transferred_horizon_rates=transferred_horizon_rates.get(spec.id, ()),
                    )
                )
        return tuple(result)

    compiled = CompiledSearch(
        initial=compile_specs(APPROVED_FAMILY_SPECS),
        conditional=compile_specs(CONDITIONAL_FAMILY_SPECS),
    )
    if compiled.initial_opportunity_count != APPROVED_PROTOCOL.initial_opportunity_budget:
        raise ValueError("initial G3 search no longer matches its approved budget")
    if (
        compiled.conditional_opportunity_count
        != APPROVED_PROTOCOL.conditional_opportunity_budget
    ):
        raise ValueError("conditional G3 search no longer matches its approved budget")
    return compiled


def _coordinate(
    spec: FamilySpec,
    *,
    opportunity_index: int,
    embedding_learning_rate: float,
    deep_learning_rate: float,
    horizon_epochs: int,
    capacity: int | None,
    reused_from: str | None = None,
    role: Literal["search", "transfer_check", "horizon_probe"] | None = None,
) -> SearchCoordinate:
    if role is None:
        role = (
            "transfer_check"
            if spec.design == "control" and opportunity_index == spec.budget - 1
            else "search"
        )
    return SearchCoordinate(
        id=f"{spec.id}:{opportunity_index + 1:02d}",
        family_id=spec.id,
        research_question=spec.research_question,
        opportunity_index=opportunity_index,
        batch_size=APPROVED_PROTOCOL.batch_size,
        seed=APPROVED_PROTOCOL.seed,
        embedding_learning_rate=embedding_learning_rate,
        deep_learning_rate=deep_learning_rate,
        horizon_epochs=horizon_epochs,
        capacity=capacity,
        conditional=spec.conditional,
        role=role,
        reused_from=reused_from,
    )


def _candidate_parameters(
    spec: FamilySpec,
) -> tuple[tuple[float, float, int, int | None], ...]:
    surface_id = (
        "rq3_catalog_output"
        if spec.id.startswith("rq3_output_")
        else spec.id
    )
    points = qmc.Sobol(
        d=4,
        scramble=True,
        seed=int.from_bytes(hashlib.sha256(surface_id.encode()).digest()[:4]),
    ).random_base2(8)
    candidates: list[tuple[float, float, int, int | None]] = []
    if spec.id == "rq5_global_gate":
        frequency_spec = next(
            value for value in APPROVED_FAMILY_SPECS if value.id == "rq5_frequency_gate"
        )
        candidates.extend(
            (
                coordinate.embedding_learning_rate,
                coordinate.deep_learning_rate,
                coordinate.horizon_epochs,
                None,
            )
            for coordinate in compile_capacity_first_stage(frequency_spec)[:3]
        )
    for index, point in enumerate(points):
        embedding_rate = _log_scale(
            float(point[0]), APPROVED_PROTOCOL.embedding_lr_bounds
        )
        deep_rate = _log_scale(float(point[1]), APPROVED_PROTOCOL.deep_lr_bounds)
        capacity = None
        horizon = APPROVED_PROTOCOL.horizon_epochs[index % 3]

        if index == 0 and not spec.capacities:
            embedding_rate = APPROVED_PROTOCOL.control.embedding_learning_rate
            deep_rate = APPROVED_PROTOCOL.control.deep_learning_rate
            horizon = APPROVED_PROTOCOL.control.horizon_epochs
        anchor_index = 1
        if index == anchor_index:
            embedding_rate = APPROVED_PROTOCOL.control.embedding_learning_rate
            deep_rate = APPROVED_PROTOCOL.control.deep_learning_rate
            horizon = APPROVED_PROTOCOL.control.horizon_epochs
        candidates.append((embedding_rate, deep_rate, horizon, capacity))
    return tuple(candidates)


def compile_rq4_extra_id_control(
    spec: FamilySpec,
    *,
    predecessor: Sequence[SearchCoordinate],
) -> tuple[SearchCoordinate, ...]:
    if spec.id != "rq4_extra_item_id" or spec.budget != 12:
        raise ValueError("extra-ID control requires the approved RQ4 extra-ID spec")
    if len(predecessor) != 12:
        raise ValueError("extra-ID control requires 12 predecessor opportunities")
    family_ids = {coordinate.family_id for coordinate in predecessor}
    allowed = {"rq4_artist", "rq4_album", "rq4_artist_album"}
    if len(family_ids) != 1 or not family_ids <= allowed:
        raise ValueError("extra-ID control predecessor must be one RQ4 metadata family")
    return tuple(
        _coordinate(
            spec,
            opportunity_index=index,
            embedding_learning_rate=coordinate.embedding_learning_rate,
            deep_learning_rate=coordinate.deep_learning_rate,
            horizon_epochs=coordinate.horizon_epochs,
            capacity=coordinate.capacity,
            role=coordinate.role,
        )
        for index, coordinate in enumerate(predecessor)
    )


def compile_capacity_first_stage(
    spec: FamilySpec,
) -> tuple[SearchCoordinate, ...]:
    if not spec.capacities or spec.design != "capacity" or spec.budget != 12:
        raise ValueError(f"{spec.id} is not an approved 12-cell capacity family")
    surface_id = {
        "rq2_content_concat": "rq2_capacity",
        "rq2_id_only_densenet": "rq2_capacity",
        "rq4_artist": "rq4_metadata_capacity",
        "rq4_album": "rq4_metadata_capacity",
        "rq4_artist_album": "rq4_metadata_capacity",
    }.get(spec.id, spec.id)
    points = _sobol_points_for_id(surface_id)
    rate_pairs = (
        (
            _log_scale(float(points[0][0]), APPROVED_PROTOCOL.embedding_lr_bounds),
            _log_scale(float(points[0][1]), APPROVED_PROTOCOL.deep_lr_bounds),
        ),
        (
            APPROVED_PROTOCOL.control.embedding_learning_rate,
            APPROVED_PROTOCOL.control.deep_learning_rate,
        ),
        (
            _log_scale(float(points[2][0]), APPROVED_PROTOCOL.embedding_lr_bounds),
            _log_scale(float(points[2][1]), APPROVED_PROTOCOL.deep_lr_bounds),
        ),
    )
    result = []
    for index in range(9):
        capacity = spec.capacities[index // 3]
        embedding_rate, deep_rate = rate_pairs[index % 3]
        result.append(
            _coordinate(
                spec,
                opportunity_index=index,
                embedding_learning_rate=embedding_rate,
                deep_learning_rate=deep_rate,
                horizon_epochs=APPROVED_PROTOCOL.control.horizon_epochs,
                capacity=capacity,
            )
        )
    return tuple(result)


def compile_capacity_horizon_followup(
    spec: FamilySpec,
    *,
    selected_capacity: int,
    transferred_horizon_rates: Sequence[TransferredHorizonRate],
    first_stage: Sequence[SearchCoordinate],
) -> tuple[SearchCoordinate, ...]:
    if not spec.capacities or spec.design != "capacity" or spec.budget != 12:
        raise ValueError(f"{spec.id} is not an approved 12-cell capacity family")
    if selected_capacity not in spec.capacities:
        raise ValueError(f"{spec.id} selected capacity is outside its search space")
    expected_first_stage = compile_capacity_first_stage(spec)
    if tuple(first_stage) != expected_first_stage:
        raise ValueError(f"{spec.id} capacity first stage differs from its approved design")
    rates = _validated_horizon_rates(transferred_horizon_rates)
    by_signature = {
        _search_signature(coordinate): coordinate for coordinate in first_stage
    }
    result = []
    for offset, horizon in enumerate(APPROVED_PROTOCOL.horizon_epochs):
        rate = rates[horizon]
        signature = (
            rate.embedding_learning_rate,
            rate.deep_learning_rate,
            horizon,
            selected_capacity,
        )
        matched = by_signature.get(signature)
        result.append(
            _coordinate(
                spec,
                opportunity_index=9 + offset,
                embedding_learning_rate=rate.embedding_learning_rate,
                deep_learning_rate=rate.deep_learning_rate,
                horizon_epochs=horizon,
                capacity=selected_capacity,
                reused_from=matched.id if matched is not None else None,
                role="horizon_probe",
            )
        )
    return tuple(result)


def _compile_balanced_capacity_family(
    spec: FamilySpec,
) -> tuple[SearchCoordinate, ...]:
    points = _sobol_points(spec)
    result = []
    for index in range(spec.budget):
        capacity = spec.capacities[index % len(spec.capacities)]
        horizon_index = (index + index // len(spec.capacities)) % len(
            APPROVED_PROTOCOL.horizon_epochs
        )
        horizon = APPROVED_PROTOCOL.horizon_epochs[horizon_index]
        embedding_rate = _log_scale(
            float(points[index][0]), APPROVED_PROTOCOL.embedding_lr_bounds
        )
        deep_rate = _log_scale(
            float(points[index][1]), APPROVED_PROTOCOL.deep_lr_bounds
        )
        if index == 1:
            embedding_rate = APPROVED_PROTOCOL.control.embedding_learning_rate
            deep_rate = APPROVED_PROTOCOL.control.deep_learning_rate
            horizon = APPROVED_PROTOCOL.control.horizon_epochs
        result.append(
            _coordinate(
                spec,
                opportunity_index=index,
                embedding_learning_rate=embedding_rate,
                deep_learning_rate=deep_rate,
                horizon_epochs=horizon,
                capacity=capacity,
            )
        )
    return tuple(result)


def _validated_horizon_rates(
    rates: Sequence[TransferredHorizonRate],
) -> dict[int, TransferredHorizonRate]:
    by_horizon = {rate.horizon_epochs: rate for rate in rates}
    if len(by_horizon) != len(rates) or set(by_horizon) != set(
        APPROVED_PROTOCOL.horizon_epochs
    ):
        raise ValueError(
            "transferred horizon rates must cover exactly horizons 15, 25, and 40"
        )
    for rate in rates:
        _validate_learning_rates(
            rate.embedding_learning_rate,
            rate.deep_learning_rate,
        )
    return by_horizon


def _sobol_points(spec: FamilySpec) -> Sequence[Sequence[float]]:
    return _sobol_points_for_id(spec.id)


def _sobol_points_for_id(surface_id: str) -> Sequence[Sequence[float]]:
    return qmc.Sobol(
        d=4,
        scramble=True,
        seed=int.from_bytes(hashlib.sha256(surface_id.encode()).digest()[:4]),
    ).random_base2(8)


def _search_signature(
    coordinate: SearchCoordinate,
) -> tuple[float, float, int, int | None]:
    return (
        coordinate.embedding_learning_rate,
        coordinate.deep_learning_rate,
        coordinate.horizon_epochs,
        coordinate.capacity,
    )


def _log_scale(value: float, bounds: tuple[float, float]) -> float:
    return math.exp(math.log(bounds[0]) + value * math.log(bounds[1] / bounds[0]))


def _signature(
    coordinate: ReusableCoordinate,
) -> tuple[float, float, int, int | None]:
    return (
        coordinate.embedding_learning_rate,
        coordinate.deep_learning_rate,
        coordinate.horizon_epochs,
        coordinate.capacity,
    )


def _validate_reusable(
    spec: FamilySpec,
    coordinate: ReusableCoordinate,
    selected_capacity: int | None,
    *,
    allow_outside_search_space: bool,
) -> None:
    if not coordinate.source_id:
        raise ValueError("reusable source ID must be nonempty")
    if allow_outside_search_space:
        _validate_positive_rate(
            coordinate.embedding_learning_rate,
            "embedding learning rate",
        )
        _validate_positive_rate(
            coordinate.deep_learning_rate,
            "deep learning rate",
        )
        if type(coordinate.horizon_epochs) is not int or coordinate.horizon_epochs < 1:
            raise ValueError("reusable horizon is invalid")
    else:
        _validate_learning_rates(
            coordinate.embedding_learning_rate,
            coordinate.deep_learning_rate,
        )
        if coordinate.horizon_epochs not in APPROVED_PROTOCOL.horizon_epochs:
            raise ValueError("reusable horizon is outside the approved search")
    if spec.capacities:
        if coordinate.capacity != selected_capacity:
            raise ValueError("reusable capacity does not match the selected capacity")
    elif coordinate.capacity is not None:
        raise ValueError("reusable coordinate has an incompatible capacity")


def _validate_learning_rates(embedding_rate: float, deep_rate: float) -> None:
    lower, upper = APPROVED_PROTOCOL.embedding_lr_bounds
    if not lower <= embedding_rate <= upper:
        raise ValueError("embedding learning rate is outside approved bounds")
    lower, upper = APPROVED_PROTOCOL.deep_lr_bounds
    if not lower <= deep_rate <= upper:
        raise ValueError("deep learning rate is outside approved bounds")


def _validate_positive_rate(value: float, label: str) -> None:
    if type(value) not in {int, float} or not math.isfinite(value) or value <= 0:
        raise ValueError(f"reusable {label} is invalid")
