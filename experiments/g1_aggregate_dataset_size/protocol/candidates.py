from __future__ import annotations

from dataclasses import dataclass, replace
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
    "batch_lr_calibration",
    "batch_initial",
    "batch_boundary",
    "baseline_initial",
    "baseline_local",
    "baseline_optimizer_boundary",
    "repeat",
    "bridge",
    "aggregate_initial",
    "aggregate_local",
    "aggregate_optimizer_boundary",
    "horizon_correction",
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
INITIAL_BATCH_SIZES = (640, 1280, 2560)
BATCH_LR_CALIBRATION_BATCH_SIZES = (512, 1280)
BATCH_LR_CALIBRATION_PAIRS = (
    (0.003261002414691765, 0.025343654763668278),
    (0.0011832644052772452, 0.06640811442971185),
    (0.006775906584815153, 0.012851178723155708),
)
INITIAL_JOINT_LR_PAIRS = (
    (0.032, 0.012),
    (0.019275929014542306, 0.01942482874826853),
    (0.046127309413540894, 0.00854511881682427),
)
LOCAL_LR_PAIRS = {
    INITIAL_JOINT_LR_PAIRS[1]: (
        (0.02667329716331745, 0.011583588892357813),
        (0.03739158126074027, 0.022404888315289213),
        (0.024128597404038644, 0.017700854385094125),
    ),
    INITIAL_JOINT_LR_PAIRS[2]: (
        (0.0268290495956486, 0.011433182352002583),
        (0.027424687121848843, 0.012648955509984123),
        (0.042889501869815085, 0.008971599022863638),
    ),
}
OPTIMIZER_BOUNDARY_RATES = {
    ("embedding", "low"): (0.004, 0.006349604207872798, 0.010079368399158985),
    ("embedding", "high"): (0.10159366732596477, 0.16126989438654377, 0.256),
    ("deep", "low"): (0.0015, 0.0023811015779522994, 0.00377976314968462),
    ("deep", "high"): (0.03809762524723678, 0.06047621039495391, 0.096),
}
_BOUNDARY_EMBEDDING_RATES = frozenset(
    (*OPTIMIZER_BOUNDARY_RATES[("embedding", "low")],
     *OPTIMIZER_BOUNDARY_RATES[("embedding", "high")])
)
_BOUNDARY_DEEP_RATES = frozenset(
    (*OPTIMIZER_BOUNDARY_RATES[("deep", "low")],
     *OPTIMIZER_BOUNDARY_RATES[("deep", "high")])
)
MAX_INITIAL_RUNS = 37
MAX_PRE_HORIZON_RUNS = 64
MAX_HORIZON_CORRECTION_RUNS = 74
MAX_APPROVED_RUNS = 138

_RUN_PATTERN = re.compile(
    r"^g1_aggregate_dataset_size_(?P<family>baseline|bridge|aggregate)_"
    r"(?P<member>[a-z_]+)_l(?P<layers>\d+)_b(?P<batch>\d+)_s(?P<seed>\d+)_"
    r"e(?P<embedding>[\dp]+)_d(?P<deep>[\dp]+)_h(?P<horizon>none|\d+)_"
    r"(?P<stage>[a-z_]+)_ts(?P<semantics>\d+)_r1_50m$"
)


class ApprovalRequired(RuntimeError):
    pass


@dataclass(frozen=True)
class AggregateCandidate:
    family: Family
    embedding_lr: float
    deep_lr: float
    batch_size: int
    stage: Stage
    seed: int = 42
    num_layers: int = 2
    member: Member | None = None
    horizon_epochs: int | None = None
    dataset_size: Literal["50m"] = "50m"

    def __post_init__(self) -> None:
        if self.dataset_size != "50m":
            raise ValueError("the aggregate dataset-size runtime owns native 50M only")
        if self.batch_size < 1 or self.seed < 0:
            raise ValueError("batch size must be positive and seed must be nonnegative")
        if self.num_layers not in (2, 4, 6, 8):
            raise ValueError("aggregate candidates use 2, 4, 6, or 8 layers")
        for name, value in (
            ("embedding LR", self.embedding_lr),
            ("deep LR", self.deep_lr),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.family == "baseline":
            if self.member is not None or self.num_layers != 2 or self.horizon_epochs:
                raise ValueError("baseline candidates have no member, depth, or horizon")
            if self.stage == "repeat" and self.seed not in range(43, 52):
                raise ValueError("baseline repeats use seeds 43 through 51")
            if self.stage != "repeat" and self.seed != 42:
                raise ValueError("selection candidates use seed 42")
        elif self.family == "bridge":
            if self.member is None:
                raise ValueError("bridges name exactly one aggregate member")
            if self.member == "depth":
                if self.num_layers not in (4, 6, 8) or self.horizon_epochs is not None:
                    raise ValueError("depth bridges use one approved depth without a horizon")
            elif self.num_layers != 2:
                raise ValueError("fixed-member bridges retain the two-layer baseline")
        elif self.family == "aggregate":
            if self.member is not None or self.num_layers not in (4, 6, 8):
                raise ValueError("aggregates combine all fixed members at one approved depth")
        else:
            raise ValueError(f"unknown aggregate family {self.family!r}")
        scheduled = self.family == "aggregate" or self.member == "scheduler"
        if scheduled != (self.horizon_epochs is not None):
            raise ValueError("only aggregate and scheduler-bridge runs declare a horizon")
        if self.horizon_epochs is not None and self.horizon_epochs not in (15, 24, 36):
            raise ValueError("approved schedule horizons are H15, H24, and H36")
        legal_stages = {
            "baseline": {
                "batch_lr_calibration",
                "batch_initial",
                "batch_boundary",
                "baseline_initial",
                "baseline_local",
                "baseline_optimizer_boundary",
                "repeat",
            },
            "bridge": {"bridge", "horizon_correction"},
            "aggregate": {
                "aggregate_initial",
                "aggregate_local",
                "aggregate_optimizer_boundary",
                "horizon_correction",
            },
        }
        if self.stage not in legal_stages[self.family]:
            raise ValueError(f"stage {self.stage!r} is invalid for {self.family}")
        if self.stage == "horizon_correction" and self.horizon_epochs not in (24, 36):
            raise ValueError("a horizon correction must be H24 or H36")
        if self.stage != "horizon_correction" and scheduled and self.horizon_epochs != 15:
            raise ValueError("scheduled initial candidates start at H15")

    @property
    def run_name(self) -> str:
        member = "none" if self.member is None else self.member
        horizon = "none" if self.horizon_epochs is None else str(self.horizon_epochs)
        return "_".join(
            (
                "g1",
                "aggregate",
                "dataset",
                "size",
                self.family,
                member,
                f"l{self.num_layers}",
                f"b{self.batch_size}",
                f"s{self.seed}",
                f"e{_slug(self.embedding_lr)}",
                f"d{_slug(self.deep_lr)}",
                f"h{horizon}",
                self.stage,
                f"ts{GENERATION_TRAINING_SEMANTICS_REVISION}",
                "r1",
                "50m",
            )
        )

    @property
    def num_epochs(self) -> int:
        return self.horizon_epochs or 80


def batch_initial_candidates() -> tuple[AggregateCandidate, ...]:
    candidates = tuple(
        AggregateCandidate(
            "baseline", 0.032, 0.012, batch_size, "batch_initial"
        )
        for batch_size in INITIAL_BATCH_SIZES
    )
    return _unique(candidates, 3)


def batch_lr_calibration_candidates() -> tuple[AggregateCandidate, ...]:
    return _unique(
        tuple(
            AggregateCandidate(
                "baseline",
                embedding_lr,
                deep_lr,
                batch_size,
                "batch_lr_calibration",
            )
            for batch_size in BATCH_LR_CALIBRATION_BATCH_SIZES
            for embedding_lr, deep_lr in BATCH_LR_CALIBRATION_PAIRS
        ),
        6,
    )


def batch_followup_candidates(
    winner: AggregateCandidate,
) -> tuple[AggregateCandidate, ...]:
    if winner.family != "baseline" or (
        winner.embedding_lr,
        winner.deep_lr,
    ) != INITIAL_JOINT_LR_PAIRS[0]:
        raise ValueError("batch calibration needs an unchanged central-rate baseline")
    if winner.stage == "batch_boundary":
        if winner.batch_size in {160, 7680}:
            raise ApprovalRequired("outer batch boundary still wins")
        return ()
    if winner.stage != "batch_initial":
        raise ValueError("batch follow-ups extend the initial batch surface")
    if winner.batch_size == 640:
        sizes = (160, 320, 480)
    elif winner.batch_size == 2560:
        sizes = (3840, 5120, 7680)
    elif winner.batch_size == 1280:
        return ()
    else:
        raise ValueError("winner is outside the initial batch surface")
    return _unique(
        tuple(replace(winner, batch_size=size, stage="batch_boundary") for size in sizes),
        3,
    )


def baseline_initial_candidates(batch_size: int) -> tuple[AggregateCandidate, ...]:
    candidates = []
    for embedding_lr, deep_lr in INITIAL_JOINT_LR_PAIRS:
        stage: Stage = (
            "batch_initial"
            if (embedding_lr, deep_lr) == INITIAL_JOINT_LR_PAIRS[0]
            and batch_size in INITIAL_BATCH_SIZES
            else "baseline_initial"
        )
        candidates.append(
            AggregateCandidate(
                "baseline", embedding_lr, deep_lr, batch_size, stage
            )
        )
    return _unique(tuple(candidates), 3)


def repeat_candidates(selected_baseline: AggregateCandidate) -> tuple[AggregateCandidate, ...]:
    _require_selectable(selected_baseline, "baseline")
    return _unique(
        tuple(
            replace(selected_baseline, stage="repeat", seed=seed)
            for seed in range(43, 52)
        ),
        9,
    )


def bridge_candidates(selected_baseline: AggregateCandidate) -> tuple[AggregateCandidate, ...]:
    _require_selectable(selected_baseline, "baseline")
    candidates = tuple(
        AggregateCandidate(
            "bridge",
            selected_baseline.embedding_lr,
            selected_baseline.deep_lr,
            selected_baseline.batch_size,
            "bridge",
            member=member,
            horizon_epochs=15 if member == "scheduler" else None,
        )
        for member in FIXED_MEMBERS
    ) + tuple(
        AggregateCandidate(
            "bridge",
            selected_baseline.embedding_lr,
            selected_baseline.deep_lr,
            selected_baseline.batch_size,
            "bridge",
            member="depth",
            num_layers=layers,
        )
        for layers in (4, 6, 8)
    )
    return _unique(candidates, 13)


def aggregate_initial_candidates(batch_size: int) -> tuple[AggregateCandidate, ...]:
    candidates = tuple(
        AggregateCandidate(
            "aggregate",
            embedding_lr,
            deep_lr,
            batch_size,
            "aggregate_initial",
            num_layers=layers,
            horizon_epochs=15,
        )
        for layers in (4, 6, 8)
        for embedding_lr, deep_lr in INITIAL_JOINT_LR_PAIRS
    )
    return _unique(candidates, 9)


def local_lr_candidates(winner: AggregateCandidate) -> tuple[AggregateCandidate, ...]:
    if winner.family not in {"baseline", "aggregate"}:
        raise ValueError("joint LR follow-ups apply to baseline and aggregate families")
    _require_selectable(winner, winner.family)
    if winner.stage not in {"baseline_initial", "aggregate_initial", "horizon_correction"}:
        if winner.stage == "batch_initial" and winner.family == "baseline":
            return ()
        raise ValueError("local LR candidates extend an initial three-pair surface")
    pairs = LOCAL_LR_PAIRS.get((winner.embedding_lr, winner.deep_lr))
    if pairs is None:
        return ()
    stage: Stage = "baseline_local" if winner.family == "baseline" else "aggregate_local"
    if winner.horizon_epochs in (24, 36):
        stage = "horizon_correction"
    return _unique(
        tuple(
            replace(winner, embedding_lr=embedding, deep_lr=deep, stage=stage)
            for embedding, deep in pairs
        ),
        3,
    )


def optimizer_boundary_candidates(
    winner: AggregateCandidate,
) -> tuple[AggregateCandidate, ...]:
    if winner.family not in {"baseline", "aggregate"}:
        raise ValueError("optimizer boundaries apply to baseline and aggregate families")
    _require_selectable(winner, winner.family)
    if winner.stage in {
        "baseline_optimizer_boundary",
        "aggregate_optimizer_boundary",
    } or (
        winner.stage == "horizon_correction"
        and (
            winner.embedding_lr in _BOUNDARY_EMBEDDING_RATES
            or winner.deep_lr in _BOUNDARY_DEEP_RATES
        )
    ):
        outer_embedding = {0.004, 0.256}
        outer_deep = {0.0015, 0.096}
        if winner.embedding_lr in outer_embedding or winner.deep_lr in outer_deep:
            raise ApprovalRequired("outer optimizer boundary still wins")
        return ()
    sides: list[tuple[str, str]] = []
    for coordinate, value, low, high in (
        ("embedding", winner.embedding_lr, 0.016, 0.064),
        ("deep", winner.deep_lr, 0.006, 0.024),
    ):
        position = (math.log(value) - math.log(low)) / (math.log(high) - math.log(low))
        if position <= 0.25:
            sides.append((coordinate, "low"))
        elif position >= 0.75:
            sides.append((coordinate, "high"))
    stage: Stage = (
        "baseline_optimizer_boundary"
        if winner.family == "baseline"
        else "aggregate_optimizer_boundary"
    )
    if winner.horizon_epochs in (24, 36):
        stage = "horizon_correction"
    candidates = []
    for coordinate, side in sides:
        for rate in OPTIMIZER_BOUNDARY_RATES[(coordinate, side)]:
            candidates.append(
                replace(
                    winner,
                    embedding_lr=rate if coordinate == "embedding" else winner.embedding_lr,
                    deep_lr=rate if coordinate == "deep" else winner.deep_lr,
                    stage=stage,
                )
            )
    return _unique(tuple(candidates), 3 * len(sides))


def horizon_followup_candidates(
    surface: tuple[AggregateCandidate, ...], horizon_epochs: Literal[24, 36]
) -> tuple[AggregateCandidate, ...]:
    if not surface:
        raise ValueError("cannot correct an empty horizon surface")
    if any(candidate.horizon_epochs not in (15, 24) for candidate in surface):
        raise ValueError("horizon follow-ups extend H15 or H24 surfaces")
    if (surface[0].horizon_epochs, horizon_epochs) not in {(15, 24), (24, 36)}:
        raise ValueError("approved horizon progression is H15 to H24 to H36")
    surface_keys = {
        (
            candidate.family,
            candidate.member,
            candidate.num_layers,
            candidate.batch_size,
            candidate.horizon_epochs,
        )
        for candidate in surface
    }
    if len(surface_keys) != 1:
        raise ValueError("a corrected horizon surface must have one family and depth")
    return _unique(
        tuple(
            replace(candidate, horizon_epochs=horizon_epochs, stage="horizon_correction")
            for candidate in surface
        ),
        len(surface),
    )


def candidate_by_run(run_name: str) -> AggregateCandidate:
    match = _RUN_PATTERN.fullmatch(run_name)
    if match is None:
        raise ValueError(f"unknown native-50M aggregate run {run_name!r}")
    values = match.groupdict()
    if int(values["semantics"]) != GENERATION_TRAINING_SEMANTICS_REVISION:
        raise ValueError("aggregate run uses historical training semantics")
    member = None if values["member"] == "none" else values["member"]
    if member is not None and member not in (*FIXED_MEMBERS, "depth"):
        raise ValueError(f"unknown aggregate member {member!r}")
    candidate = AggregateCandidate(
        family=values["family"],  # type: ignore[arg-type]
        member=member,  # type: ignore[arg-type]
        num_layers=int(values["layers"]),
        batch_size=int(values["batch"]),
        seed=int(values["seed"]),
        embedding_lr=_rate(values["embedding"]),
        deep_lr=_rate(values["deep"]),
        horizon_epochs=(
            None if values["horizon"] == "none" else int(values["horizon"])
        ),
        stage=values["stage"],  # type: ignore[arg-type]
    )
    if candidate.run_name != run_name:
        raise ValueError(f"non-canonical native-50M aggregate run {run_name!r}")
    return candidate


def _require_selectable(candidate: AggregateCandidate, family: Family) -> None:
    if candidate.family != family or candidate.seed != 42:
        raise ValueError(f"selection requires one seed-42 {family} candidate")


def _unique(
    candidates: tuple[AggregateCandidate, ...], size: int
) -> tuple[AggregateCandidate, ...]:
    if len(candidates) != size or len({candidate.run_name for candidate in candidates}) != size:
        raise RuntimeError(f"expected {size} unique native-50M aggregate candidates")
    return candidates


def _slug(value: float) -> str:
    return repr(value).replace(".", "p")


def _rate(value: str) -> float:
    return float(value.replace("p", "."))
