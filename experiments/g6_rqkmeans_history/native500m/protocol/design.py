from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
import math
from typing import Literal

import torch


DATASET_SIZE = "native-500m"
BATCH_SIZE = 512
FIXED_HORIZON = 26
TRAINING_HORIZON = FIXED_HORIZON
REPRESENTATION_WIDTH = 128
TOKENIZER_LEVELS = (3, 4)
SHARED_CODEBOOK_SIZES = (512, 2048, 8192)
CODEBOOK_SYMBOL_CAP = 8192
KMEANS_MAX_ITERATIONS = 300
KMEANS_TOLERANCE = 1e-4
EMBEDDING_LR_BOUNDS = (0.008, 0.512)
DEEP_LR_BOUNDS = (0.002, 0.128)
BOUNDARY_FACTORS = (math.sqrt(2), 2.0, 2 * math.sqrt(2), 4.0)
EXPECTED_RUN_TOTALS = (130, 132, 134, 138, 140, 142)
MAXIMUM_RUNS = 262
RQ0_EXPECTED_STAGE_COUNTS = (
    ("best_g1_control", 12),
    ("original_g1_control", 12),
    ("first_representation", 12),
    ("inherited_representations", 48),
    ("original_g1_bridge", 8),
)

REPRESENTATIONS = (
    "learned_sid_event",
    "item_frozen_sid_event",
    "item_learned_frozen_sid_event",
    "learned_sid_tokens",
    "learned_frozen_sid_tokens",
    "frozen_sid_tokens",
    "interleaved_item_sid_tokens",
)
FIRST_RQ0_REPRESENTATION = "item_frozen_sid_event"


@dataclass(frozen=True)
class LearningRateCoordinate:
    embedding: float
    deep: float

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(value) and value > 0 for value in (self.embedding, self.deep)
        ):
            raise ValueError("learning rates must be positive finite values")


@dataclass(frozen=True, order=True)
class TokenizerCoordinate:
    levels: int
    shared_codes: int

    def __post_init__(self) -> None:
        if self.levels not in TOKENIZER_LEVELS:
            raise ValueError("tokenizer levels are outside the approved domain")
        if self.shared_codes not in SHARED_CODEBOOK_SIZES:
            raise ValueError("shared code count is outside the approved domain")
        if self.shared_codes > CODEBOOK_SYMBOL_CAP:
            raise ValueError("shared code count exceeds the per-level symbol cap")


@dataclass(frozen=True)
class SurfaceCoordinate:
    tokenizer: TokenizerCoordinate
    learning_rates: LearningRateCoordinate


Initialization = Literal["random", "content_pca"]
CollisionPolicy = Literal["suffix", "none"]


@dataclass(frozen=True)
class InitializationCoordinate:
    initialization: Initialization
    coordinate: SurfaceCoordinate


@dataclass(frozen=True)
class PolicyCoordinate:
    policy: CollisionPolicy
    coordinate: SurfaceCoordinate


class BoundaryStatus(Enum):
    RESOLVED = "resolved"
    EXTEND = "extend"
    REQUIRES_APPROVAL = "requires_approval"


@dataclass(frozen=True)
class BoundaryDesign:
    status: BoundaryStatus
    coordinates: tuple[LearningRateCoordinate, ...]


@dataclass(frozen=True)
class RunBudget:
    rq0_expected: int
    rq1_expected: int
    rq23_expected: tuple[int, ...]
    terminal_bridge_expected: tuple[int, ...]
    rq0_maximum: int
    rq1_maximum: int
    rq23_maximum: int
    terminal_bridge_maximum: int

    @property
    def expected_totals(self) -> tuple[int, ...]:
        return tuple(
            sorted(
                {
                    self.rq0_expected + self.rq1_expected + rq23 + bridge
                    for rq23 in self.rq23_expected
                    for bridge in self.terminal_bridge_expected
                }
            )
        )

    @property
    def maximum(self) -> int:
        return (
            self.rq0_maximum
            + self.rq1_maximum
            + self.rq23_maximum
            + self.terminal_bridge_maximum
        )


BEST_G1_ANCHOR = LearningRateCoordinate(0.0468526465053628, 0.032703745675187676)
ORIGINAL_G1_ANCHOR = LearningRateCoordinate(0.001, 0.002)


def tokenizer_coordinates() -> tuple[TokenizerCoordinate, ...]:
    return tuple(
        TokenizerCoordinate(levels, codes)
        for levels in TOKENIZER_LEVELS
        for codes in SHARED_CODEBOOK_SIZES
    )


def sobol_learning_rates(count: int) -> tuple[LearningRateCoordinate, ...]:
    return _sobol_learning_rates_excluding(count, ())


@lru_cache(maxsize=None)
def _sobol_learning_rates_excluding(
    count: int,
    excluded: tuple[LearningRateCoordinate, ...],
) -> tuple[LearningRateCoordinate, ...]:
    if count < 0:
        raise ValueError("Sobol coordinate count must be nonnegative")
    if count == 0:
        return ()
    engine = torch.quasirandom.SobolEngine(dimension=2, scramble=True, seed=42)
    blocked = set(excluded)
    coordinates: list[LearningRateCoordinate] = []
    while len(coordinates) < count:
        row = engine.draw(1)[0]
        coordinate = LearningRateCoordinate(
            _log_uniform(EMBEDDING_LR_BOUNDS, float(row[0])),
            _log_uniform(DEEP_LR_BOUNDS, float(row[1])),
        )
        if coordinate in blocked:
            continue
        blocked.add(coordinate)
        coordinates.append(coordinate)
    return tuple(coordinates)


def control_surface(
    anchor: LearningRateCoordinate,
) -> tuple[LearningRateCoordinate, ...]:
    return (anchor, *_sobol_learning_rates_excluding(11, (anchor,)))


def rq0_first_surface() -> tuple[SurfaceCoordinate, ...]:
    tokenizers = tokenizer_coordinates()
    rates = sobol_learning_rates(12)
    return tuple(
        SurfaceCoordinate(tokenizers[index % len(tokenizers)], rate)
        for index, rate in enumerate(rates)
    )


def inherited_rq0_surface(
    inherited_tokenizer: TokenizerCoordinate,
    inherited_learning_rates: LearningRateCoordinate,
) -> tuple[SurfaceCoordinate, ...]:
    alternate_tokenizers = tuple(
        candidate
        for candidate in tokenizer_coordinates()
        if candidate != inherited_tokenizer
    )
    return (
        SurfaceCoordinate(inherited_tokenizer, inherited_learning_rates),
        *(
            SurfaceCoordinate(tokenizer, inherited_learning_rates)
            for tokenizer in alternate_tokenizers
        ),
        *(
            SurfaceCoordinate(inherited_tokenizer, rates)
            for rates in _sobol_learning_rates_excluding(2, (inherited_learning_rates,))
        ),
    )


def bridge_surface(
    inherited_learning_rates: LearningRateCoordinate,
) -> tuple[LearningRateCoordinate, ...]:
    return (
        inherited_learning_rates,
        *_sobol_learning_rates_excluding(7, (inherited_learning_rates,)),
    )


def rq1_paired_surface(
    inherited_tokenizer: TokenizerCoordinate,
    inherited_learning_rates: LearningRateCoordinate,
) -> tuple[InitializationCoordinate, ...]:
    coordinates = (
        SurfaceCoordinate(inherited_tokenizer, inherited_learning_rates),
        *(
            SurfaceCoordinate(inherited_tokenizer, rates)
            for rates in _sobol_learning_rates_excluding(5, (inherited_learning_rates,))
        ),
    )
    return tuple(
        InitializationCoordinate(initialization, coordinate)
        for coordinate in coordinates
        for initialization in ("random", "content_pca")
    )


def rq23_paired_surface(
    inherited_tokenizer: TokenizerCoordinate,
    inherited_learning_rates: LearningRateCoordinate,
    *,
    suffix_winner: TokenizerCoordinate,
    no_suffix_winner: TokenizerCoordinate,
) -> tuple[PolicyCoordinate, ...]:
    initial = tuple(
        PolicyCoordinate(policy, SurfaceCoordinate(tokenizer, inherited_learning_rates))
        for tokenizer in tokenizer_coordinates()
        for policy in ("suffix", "none")
    )
    refinements = tuple(
        PolicyCoordinate(
            policy,
            SurfaceCoordinate(
                suffix_winner if policy == "suffix" else no_suffix_winner,
                rates,
            ),
        )
        for rates in _sobol_learning_rates_excluding(4, (inherited_learning_rates,))
        for policy in ("suffix", "none")
    )
    return initial + refinements


def expected_rq23_runs(
    suffix_winner_equals_rq0_anchor: bool,
    rq1_already_confirmed_anchor: bool,
) -> int:
    return (
        25
        - 2 * int(suffix_winner_equals_rq0_anchor)
        - 2 * int(rq1_already_confirmed_anchor)
    )


def run_budget() -> RunBudget:
    budget = RunBudget(
        rq0_expected=92,
        rq1_expected=17,
        rq23_expected=(21, 23, 25),
        terminal_bridge_expected=(0, 8),
        rq0_maximum=172,
        rq1_maximum=33,
        rq23_maximum=41,
        terminal_bridge_maximum=16,
    )
    if budget.expected_totals != EXPECTED_RUN_TOTALS or budget.maximum != MAXIMUM_RUNS:
        raise RuntimeError("native-500M run arithmetic no longer matches the approval")
    return budget


def boundary_coordinates(
    selected: LearningRateCoordinate,
    *,
    round_number: int,
    boundary_won: bool = False,
) -> BoundaryDesign:
    if round_number not in (0, 1):
        raise ValueError("only one boundary extension round is approved")
    if round_number == 1:
        return BoundaryDesign(
            (
                BoundaryStatus.REQUIRES_APPROVAL
                if boundary_won
                else BoundaryStatus.RESOLVED
            ),
            (),
        )
    axes = (
        ("embedding", selected.embedding, EMBEDDING_LR_BOUNDS),
        ("deep", selected.deep, DEEP_LR_BOUNDS),
    )
    coordinates: list[LearningRateCoordinate] = []
    for axis, value, bounds in axes:
        side = _boundary_side(value, bounds)
        if side is None:
            continue
        for factor in BOUNDARY_FACTORS:
            outward = value / factor if side == "lower" else value * factor
            coordinates.append(
                LearningRateCoordinate(
                    outward if axis == "embedding" else selected.embedding,
                    outward if axis == "deep" else selected.deep,
                )
            )
    status = BoundaryStatus.EXTEND if coordinates else BoundaryStatus.RESOLVED
    return BoundaryDesign(status, tuple(coordinates))


def _boundary_side(
    value: float, bounds: tuple[float, float]
) -> Literal["lower", "upper"] | None:
    lower, upper = bounds
    if value <= lower:
        return "lower"
    if value >= upper:
        return "upper"
    position = math.log(value / lower) / math.log(upper / lower)
    if position <= 0.1:
        return "lower"
    if position >= 0.9:
        return "upper"
    return None


def _log_uniform(bounds: tuple[float, float], value: float) -> float:
    low, high = bounds
    return low * (high / low) ** value
