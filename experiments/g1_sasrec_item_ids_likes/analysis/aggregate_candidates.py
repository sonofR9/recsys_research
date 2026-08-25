from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import cache
import math
import re
from typing import Literal

from dcn.training_metadata import GENERATION_TRAINING_SEMANTICS_REVISION


FixedMember = Literal[
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
]
Member = FixedMember | Literal["depth"]
Family = Literal["baseline", "bridge", "aggregate"]
Stage = Literal[
    "initial",
    "baseline_boundary",
    "bridge",
    "local",
    "optimizer_boundary",
    "horizon_correction",
    "full_horizon_rerun",
]

FIXED_MEMBERS: tuple[FixedMember, ...] = (
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
)
BASELINE_DEEP_LRS = (0.006, 0.012, 0.024)
AGGREGATE_INITIAL_PAIRS = (
    (0.064, 0.048),
    (0.07764674795069047, 0.02484672863178322),
    (0.0468526465053628, 0.032703745675187676),
)
_LOCAL_PAIRS = {
    AGGREGATE_INITIAL_PAIRS[1]: (
        (0.06957712293357378, 0.04045869601192933),
        (0.058511920889791694, 0.03301809071022853),
        (0.08595109115349953, 0.03835282700655922),
    ),
    AGGREGATE_INITIAL_PAIRS[2]: (
        (0.06547725593418215, 0.027941927666112344),
        (0.06295219371532267, 0.04651981979406204),
        (0.046467420627589774, 0.0404566728379668),
    ),
}
_OPTIMIZER_BOUNDARIES = {
    ("embedding", "low"): (0.008, 0.012699208415745596, 0.02015873679831797),
    ("embedding", "high"): (0.20318733465192954, 0.32253978877308753, 0.512),
    ("deep", "low"): (0.006, 0.009524406311809197, 0.01511905259873848),
    ("deep", "high"): (0.15239050098894713, 0.24190484157981565, 0.384),
}
_THIRD_CORRECTION_CHAINS = {
    (6, AGGREGATE_INITIAL_PAIRS[2][0], AGGREGATE_INITIAL_PAIRS[2][1]): (
        15,
        23,
        12,
        18,
    ),
    (8, AGGREGATE_INITIAL_PAIRS[1][0], AGGREGATE_INITIAL_PAIRS[1][1]): (
        15,
        23,
        17,
        13,
    ),
}
_FOURTH_CORRECTION_CHAIN = (
    15,
    23,
    12,
    18,
    27,
)
_FOURTH_CORRECTION_RECIPE = (
    6,
    AGGREGATE_INITIAL_PAIRS[2][0],
    AGGREGATE_INITIAL_PAIRS[2][1],
)
_FULL_HORIZON_RERUN_RECIPES = (
    (6, AGGREGATE_INITIAL_PAIRS[0][0], AGGREGATE_INITIAL_PAIRS[0][1]),
    (8, AGGREGATE_INITIAL_PAIRS[2][0], AGGREGATE_INITIAL_PAIRS[2][1]),
)
_RUN_PATTERN = re.compile(
    r"^g1_aggregate_(?P<family>baseline|bridge|aggregate)_"
    r"(?P<member>[a-z_]+)_l(?P<layers>\d+)_"
    r"e(?P<embedding>[\dp]+)_d(?P<deep>[\dp]+)_"
    r"h(?P<horizon>none|\d+)_c(?P<correction>\d+)_"
    r"(?:from(?P<predecessors>\d+(?:p\d+)*)_)?"
    r"(?P<stage>[a-z_]+)_ts(?P<semantics>\d+)_r1_500m$"
)


class ApprovalRequired(RuntimeError):
    pass


@dataclass(frozen=True)
class AggregateCandidate:
    family: Family
    embedding_lr: float
    deep_lr: float
    stage: Stage
    num_layers: int = 2
    member: Member | None = None
    horizon_epochs: int | None = None
    correction: int = 0
    horizon_chain: tuple[int, ...] = field(default=(), compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.stage not in {
            "initial",
            "baseline_boundary",
            "bridge",
            "local",
            "optimizer_boundary",
            "horizon_correction",
            "full_horizon_rerun",
        }:
            raise ValueError(f"unknown aggregate stage {self.stage!r}")
        if not 0 <= self.correction <= 4:
            raise ValueError("at most four horizon corrections are representable")
        if self.correction and self.stage != "horizon_correction":
            raise ValueError("corrected horizons need the correction stage")
        if self.stage == "horizon_correction" and not self.correction:
            raise ValueError("the horizon-correction stage needs a correction")
        if self.num_layers not in (2, 4, 6, 8):
            raise ValueError("aggregate candidates use 2, 4, 6, or 8 layers")
        for name, value in (
            ("embedding LR", self.embedding_lr),
            ("deep LR", self.deep_lr),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.family == "baseline" and (
            self.member is not None or self.num_layers != 2 or self.horizon_epochs is not None
        ):
            raise ValueError("the baseline has no treatment, depth change, or horizon")
        if self.family == "bridge" and self.member is None:
            raise ValueError("a bridge must name exactly one member")
        if self.family == "bridge" and self.member != "depth" and self.num_layers != 2:
            raise ValueError("only the depth bridge changes baseline depth")
        if self.family == "aggregate" and (
            self.member is not None or self.num_layers not in (4, 6, 8)
        ):
            raise ValueError("aggregate candidates combine every fixed member")
        legal_stage = {
            "baseline": self.stage in {"initial", "baseline_boundary"},
            "bridge": self.stage == "bridge"
            or (
                self.stage == "horizon_correction"
                and self.member == "scheduler"
                and self.correction > 0
            ),
            "aggregate": self.stage
            in {"initial", "local", "optimizer_boundary", "full_horizon_rerun"}
            or (self.stage == "horizon_correction" and self.correction > 0),
        }.get(self.family, False)
        if not legal_stage:
            raise ValueError(
                f"stage {self.stage!r} is not valid for {self.family!r} candidates"
            )
        if self.stage == "full_horizon_rerun" and (
            self.correction != 0
            or self.horizon_epochs != 15
            or (self.num_layers, self.embedding_lr, self.deep_lr)
            not in _FULL_HORIZON_RERUN_RECIPES
        ):
            raise ValueError("unapproved full-H15 rerun identity")
        scheduled = self.family == "aggregate" or self.member == "scheduler"
        if scheduled != (self.horizon_epochs is not None):
            raise ValueError("only scheduled candidates declare an adaptive horizon")
        if self.horizon_epochs is not None and self.horizon_epochs < 1:
            raise ValueError("adaptive horizons must be positive")
        if self.horizon_chain and (
            self.horizon_epochs is None
            or len(self.horizon_chain) != self.correction + 1
            or self.horizon_chain[-1] != self.horizon_epochs
        ):
            raise ValueError("horizon chain does not match the correction")
        if (
            self.correction in {1, 2}
            and self.horizon_chain
            and self.horizon_chain[0] != 15
        ):
            raise ValueError("corrected horizon chains must start at canonical H15")
        if self.correction == 3 and self.horizon_chain != (
            _THIRD_CORRECTION_CHAINS.get(
                (self.num_layers, self.embedding_lr, self.deep_lr)
            )
        ):
            raise ValueError("unapproved third horizon correction predecessor chain")
        if self.correction == 4 and (
            (self.num_layers, self.embedding_lr, self.deep_lr)
            != _FOURTH_CORRECTION_RECIPE
            or self.horizon_chain != _FOURTH_CORRECTION_CHAIN
        ):
            raise ValueError("unapproved fourth horizon correction predecessor chain")

    @property
    def run_name(self) -> str:
        member = "none" if self.member is None else self.member
        horizon = "none" if self.horizon_epochs is None else str(self.horizon_epochs)
        lineage = (
            ()
            if self.correction not in {3, 4}
            else ("from" + "p".join(map(str, self.horizon_chain[:-1])),)
        )
        return "_".join(
            (
                "g1",
                "aggregate",
                self.family,
                member,
                f"l{self.num_layers}",
                f"e{_slug(self.embedding_lr)}",
                f"d{_slug(self.deep_lr)}",
                f"h{horizon}",
                f"c{self.correction}",
                *lineage,
                self.stage,
                f"ts{GENERATION_TRAINING_SEMANTICS_REVISION}",
                "r1",
                "500m",
            )
        )

    @property
    def num_epochs(self) -> int:
        return 80 if self.horizon_epochs is None else self.horizon_epochs


@cache
def initial_candidates() -> tuple[AggregateCandidate, ...]:
    candidates = tuple(
        AggregateCandidate("baseline", 0.064, deep_lr, "initial")
        for deep_lr in BASELINE_DEEP_LRS
    ) + tuple(
        AggregateCandidate(
            "aggregate",
            embedding_lr,
            deep_lr,
            "initial",
            num_layers=layers,
            horizon_epochs=15,
        )
        for layers in (4, 6, 8)
        for embedding_lr, deep_lr in AGGREGATE_INITIAL_PAIRS
    )
    _require_unique(candidates, 12)
    return candidates


@cache
def full_horizon_rerun_candidates() -> tuple[AggregateCandidate, ...]:
    candidates = tuple(
        replace(candidate, stage="full_horizon_rerun")
        for recipe in _FULL_HORIZON_RERUN_RECIPES
        for candidate in initial_candidates()
        if (
            candidate.num_layers,
            candidate.embedding_lr,
            candidate.deep_lr,
        )
        == recipe
    )
    _require_unique(candidates, 2)
    return candidates


@cache
def selection_initial_candidates() -> tuple[AggregateCandidate, ...]:
    reruns = {
        (candidate.num_layers, candidate.embedding_lr, candidate.deep_lr): candidate
        for candidate in full_horizon_rerun_candidates()
    }
    candidates = tuple(
        reruns.get(
            (candidate.num_layers, candidate.embedding_lr, candidate.deep_lr),
            candidate,
        )
        for candidate in initial_candidates()
    )
    _require_unique(candidates, 12)
    return candidates


def bridge_candidates(
    baseline_deep_lr: float, *, selected_depth: int | None = None
) -> tuple[AggregateCandidate, ...]:
    candidates = tuple(
        AggregateCandidate(
            "bridge",
            0.064,
            baseline_deep_lr,
            "bridge",
            member=member,
            horizon_epochs=15 if member == "scheduler" else None,
        )
        for member in FIXED_MEMBERS
    )
    if selected_depth is not None:
        candidates += (
            AggregateCandidate(
                "bridge",
                0.064,
                baseline_deep_lr,
                "bridge",
                member="depth",
                num_layers=selected_depth,
            ),
        )
    _require_unique(candidates, 10 + int(selected_depth is not None))
    return candidates


def baseline_boundary_candidates(
    winner: AggregateCandidate,
) -> tuple[AggregateCandidate, ...]:
    if winner.family != "baseline":
        raise ValueError("baseline boundaries require a baseline winner")
    if winner.stage == "baseline_boundary":
        if winner.deep_lr in {0.00075, 0.192}:
            raise ApprovalRequired("outer baseline boundary still wins")
        return ()
    if winner.stage != "initial":
        raise ValueError("baseline boundaries extend the initial surface")
    if winner.deep_lr == BASELINE_DEEP_LRS[0]:
        rates = (0.003, 0.0015, 0.00075)
    elif winner.deep_lr == BASELINE_DEEP_LRS[-1]:
        rates = (0.048, 0.096, 0.192)
    else:
        return ()
    candidates = tuple(
        replace(winner, deep_lr=rate, stage="baseline_boundary") for rate in rates
    )
    _require_unique(candidates, 3)
    return candidates


def aggregate_local_candidates(
    winner: AggregateCandidate,
) -> tuple[AggregateCandidate, ...]:
    if winner.family != "aggregate" or winner.stage != "initial":
        raise ValueError("a local round extends an initial aggregate candidate")
    pairs = _LOCAL_PAIRS.get((winner.embedding_lr, winner.deep_lr))
    if pairs is None:
        return ()
    candidates = tuple(
        replace(winner, embedding_lr=embedding, deep_lr=deep, stage="local")
        for embedding, deep in pairs
    )
    _require_unique(candidates, 3)
    return candidates


def aggregate_boundary_candidates(
    winner: AggregateCandidate,
) -> tuple[AggregateCandidate, ...]:
    if winner.family != "aggregate":
        raise ValueError("optimizer boundaries require an aggregate winner")
    if winner.stage == "optimizer_boundary":
        if winner.embedding_lr in {0.008, 0.512} or winner.deep_lr in {0.006, 0.384}:
            raise ApprovalRequired("outer aggregate optimizer boundary still wins")
        return ()
    if winner.stage not in {"initial", "local"}:
        raise ValueError("optimizer boundaries follow initial or local selection")
    sides = []
    for coordinate, value, low, high in (
        ("embedding", winner.embedding_lr, 0.032, 0.128),
        ("deep", winner.deep_lr, 0.024, 0.096),
    ):
        log_value = math.log(value)
        log_low = math.log(low)
        log_high = math.log(high)
        if log_value <= log_low + 0.25 * (log_high - log_low):
            sides.append((coordinate, "low"))
        elif log_value >= log_low + 0.75 * (log_high - log_low):
            sides.append((coordinate, "high"))
    candidates = []
    for coordinate, side in sides:
        for rate in _OPTIMIZER_BOUNDARIES[(coordinate, side)]:
            candidates.append(
                replace(
                    winner,
                    embedding_lr=(rate if coordinate == "embedding" else winner.embedding_lr),
                    deep_lr=(rate if coordinate == "deep" else winner.deep_lr),
                    stage="optimizer_boundary",
                )
            )
    result = tuple(candidates)
    _require_unique(result, 3 * len(sides))
    return result


@cache
def _approved_scheduled_start_names() -> frozenset[str]:
    aggregate_starts: list[AggregateCandidate] = []
    for initial in initial_candidates():
        if initial.family != "aggregate":
            continue
        local = aggregate_local_candidates(initial)
        aggregate_starts.extend((initial, *local))
        for candidate in (initial, *local):
            aggregate_starts.extend(aggregate_boundary_candidates(candidate))

    baseline_rates = set(BASELINE_DEEP_LRS)
    for baseline in initial_candidates()[:3]:
        baseline_rates.update(
            candidate.deep_lr for candidate in baseline_boundary_candidates(baseline)
        )
    scheduler_starts = (
        next(
            candidate
            for candidate in bridge_candidates(deep_lr)
            if candidate.member == "scheduler"
        )
        for deep_lr in baseline_rates
    )
    return frozenset(
        candidate.run_name for candidate in (*aggregate_starts, *scheduler_starts)
    )


def make_horizon_correction(
    source: AggregateCandidate, horizon_epochs: int
) -> AggregateCandidate:
    if source.horizon_epochs is None:
        raise ValueError("constant-LR candidates have no horizon correction")
    if (
        source.correction == 0
        and source.run_name not in _approved_scheduled_start_names()
    ):
        raise ValueError(
            "first correction requires a canonical approved H15 scheduled start"
        )
    source_chain = source.horizon_chain
    if not source_chain:
        if source.correction == 0:
            source_chain = (source.horizon_epochs,)
        elif source.correction == 1:
            source_chain = (15, source.horizon_epochs)
    target_chain = (*source_chain, horizon_epochs)
    if source.correction == 2:
        recipe = (
            source.num_layers,
            source.embedding_lr,
            source.deep_lr,
        )
        approved_chain = _THIRD_CORRECTION_CHAINS.get(recipe)
        if approved_chain is None:
            raise ApprovalRequired("two horizon corrections did not calibrate the run")
        if horizon_epochs != approved_chain[-1]:
            raise ApprovalRequired(
                f"third horizon correction is fixed at {approved_chain[-1]} epochs"
            )
        if target_chain != approved_chain:
            raise ApprovalRequired(
                "third horizon correction requires the full approved predecessor chain"
            )
    elif source.correction == 3:
        recipe = (source.num_layers, source.embedding_lr, source.deep_lr)
        if recipe != _FOURTH_CORRECTION_RECIPE:
            raise ApprovalRequired("third horizon correction did not calibrate the run")
        if horizon_epochs != _FOURTH_CORRECTION_CHAIN[-1]:
            raise ApprovalRequired(
                f"fourth horizon correction is fixed at "
                f"{_FOURTH_CORRECTION_CHAIN[-1]} epochs"
            )
        if target_chain != _FOURTH_CORRECTION_CHAIN:
            raise ApprovalRequired(
                "fourth horizon correction requires the full approved predecessor chain"
            )
    elif source.correction == 4:
        raise ApprovalRequired("four horizon corrections did not calibrate the run")
    return replace(
        source,
        horizon_epochs=horizon_epochs,
        correction=source.correction + 1,
        stage="horizon_correction",
        horizon_chain=target_chain,
    )


@cache
def recovery_candidates() -> tuple[AggregateCandidate, ...]:
    six_layer = next(
        candidate
        for candidate in initial_candidates()
        if candidate.family == "aggregate"
        and candidate.num_layers == 6
        and (candidate.embedding_lr, candidate.deep_lr) == AGGREGATE_INITIAL_PAIRS[2]
    )
    eight_layer = next(
        candidate
        for candidate in initial_candidates()
        if candidate.family == "aggregate"
        and candidate.num_layers == 8
        and (candidate.embedding_lr, candidate.deep_lr) == AGGREGATE_INITIAL_PAIRS[1]
    )
    six_third = make_horizon_correction(
        make_horizon_correction(make_horizon_correction(six_layer, 23), 12), 18
    )
    eight_third = make_horizon_correction(
        make_horizon_correction(make_horizon_correction(eight_layer, 23), 17), 13
    )
    baseline_lower = baseline_boundary_candidates(initial_candidates()[0])
    four_layer_second = next(
        candidate
        for candidate in initial_candidates()
        if candidate.family == "aggregate"
        and candidate.num_layers == 4
        and (candidate.embedding_lr, candidate.deep_lr) == AGGREGATE_INITIAL_PAIRS[2]
    )
    candidates = (
        six_third,
        eight_third,
        *baseline_lower,
        *aggregate_local_candidates(four_layer_second),
    )
    _require_unique(candidates, 8)
    return candidates


def candidate_by_run(run_name: str) -> AggregateCandidate:
    match = _RUN_PATTERN.fullmatch(run_name)
    if match is None:
        raise ValueError(f"unknown aggregate run {run_name!r}")
    values = match.groupdict()
    if int(values["semantics"]) != GENERATION_TRAINING_SEMANTICS_REVISION:
        raise ValueError("aggregate run uses historical training semantics")
    member = None if values["member"] == "none" else values["member"]
    if member is not None and member not in (*FIXED_MEMBERS, "depth"):
        raise ValueError(f"unknown aggregate member {member!r}")
    correction = int(values["correction"])
    horizon = None if values["horizon"] == "none" else int(values["horizon"])
    lineage: tuple[int, ...] = ()
    if values["predecessors"] is not None:
        assert horizon is not None
        lineage = (
            *tuple(int(value) for value in values["predecessors"].split("p")),
            horizon,
        )
    elif correction == 1 and horizon is not None:
        lineage = (15, horizon)
    elif correction == 2 and horizon is not None:
        recipe = (
            int(values["layers"]),
            _rate(values["embedding"]),
            _rate(values["deep"]),
        )
        approved_chain = _THIRD_CORRECTION_CHAINS.get(recipe)
        if approved_chain is not None and horizon == approved_chain[-2]:
            lineage = approved_chain[:-1]
    candidate = AggregateCandidate(
        family=values["family"],  # type: ignore[arg-type]
        member=member,  # type: ignore[arg-type]
        num_layers=int(values["layers"]),
        embedding_lr=_rate(values["embedding"]),
        deep_lr=_rate(values["deep"]),
        horizon_epochs=horizon,
        correction=correction,
        stage=values["stage"],  # type: ignore[arg-type]
        horizon_chain=lineage,
    )
    if candidate.run_name != run_name:
        raise ValueError(f"non-canonical aggregate run {run_name!r}")
    return candidate


def _require_unique(candidates: tuple[AggregateCandidate, ...], size: int) -> None:
    if len(candidates) != size or len({candidate.run_name for candidate in candidates}) != size:
        raise RuntimeError(f"expected {size} unique aggregate candidates")


def _slug(value: float) -> str:
    return repr(value).replace(".", "p")


def _rate(value: str) -> float:
    return float(value.replace("p", "."))
