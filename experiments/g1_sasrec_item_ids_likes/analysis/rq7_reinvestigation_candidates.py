from __future__ import annotations

from dataclasses import InitVar, dataclass
from functools import cache
import math
import re
from typing import Literal

from dcn.training_metadata import GENERATION_TRAINING_SEMANTICS_REVISION


DatasetSize = Literal["50m", "500m"]
Fusion = Literal["add", "concat"]
Stage = Literal[
    "diagnostic",
    "initial",
    "rope_base",
    "rope_base_extension",
    "boundary",
    "confirmation",
]
BoundarySide = Literal["low", "high"]
ImplementationRevision = Literal[1, 2, 3, 4, 5, 6, 7]
LegacyConcatRevision = Literal[1, 2]

_INITIAL_DEEP_LRS = (0.006, 0.012, 0.024)


@dataclass(frozen=True)
class PositionTreatment:
    learned_positions: str | tuple[str, str] | None = None
    learned_position_fusion: Fusion = "add"
    rope: str | None = None
    rope_base: float = 10000.0
    alibi: bool = False


_PRIMARY_TREATMENTS: dict[str, PositionTreatment] = {
    "learned_forward_add": PositionTreatment(learned_positions="forward"),
    "learned_forward_concat": PositionTreatment(
        learned_positions="forward", learned_position_fusion="concat"
    ),
    "learned_forward_reverse_add": PositionTreatment(
        learned_positions=("forward", "reverse")
    ),
    "learned_forward_reverse_concat": PositionTreatment(
        learned_positions=("forward", "reverse"),
        learned_position_fusion="concat",
    ),
    "learned_forward_add_alibi": PositionTreatment(
        learned_positions="forward", alibi=True
    ),
    "learned_forward_concat_alibi": PositionTreatment(
        learned_positions="forward",
        learned_position_fusion="concat",
        alibi=True,
    ),
    "learned_forward_reverse_add_alibi": PositionTreatment(
        learned_positions=("forward", "reverse"), alibi=True
    ),
    "learned_forward_reverse_concat_alibi": PositionTreatment(
        learned_positions=("forward", "reverse"),
        learned_position_fusion="concat",
        alibi=True,
    ),
    "none": PositionTreatment(),
    "alibi": PositionTreatment(alibi=True),
    "rope_forward_base10000": PositionTreatment(rope="forward"),
    "rope_forward_base10000_alibi": PositionTreatment(rope="forward", alibi=True),
}

_ROPE_BASE_TREATMENTS: dict[str, PositionTreatment] = {
    "rope_forward_base100": PositionTreatment(rope="forward", rope_base=100.0),
    "rope_forward_base1000": PositionTreatment(rope="forward", rope_base=1000.0),
}

_OUTER_ROPE_BASE_TREATMENTS: dict[str, PositionTreatment] = {
    "rope_forward_base10": PositionTreatment(rope="forward", rope_base=10.0),
    "rope_forward_base100000": PositionTreatment(rope="forward", rope_base=100000.0),
}

_TREATMENTS = _PRIMARY_TREATMENTS | _ROPE_BASE_TREATMENTS | _OUTER_ROPE_BASE_TREATMENTS


@dataclass(frozen=True)
class Rq7Candidate:
    treatment: str
    deep_lr: float
    dataset_size: DatasetSize
    stage: Stage
    embedding_lr: float = 0.064
    batch_size: int = 1280
    seed: int = 42
    horizon_epochs: int = 20
    implementation_revision: ImplementationRevision = 1
    boundary_side: BoundarySide | None = None
    boundary_step: int | None = None
    _allow_historical_native: InitVar[bool] = False

    def __post_init__(self, _allow_historical_native: bool) -> None:
        if self.treatment not in _TREATMENTS:
            raise ValueError(f"unknown RQ7 treatment {self.treatment!r}")
        if self.embedding_lr != 0.064 or self.batch_size != 1280:
            raise ValueError("RQ7 embedding LR and batch are fixed")
        if self.horizon_epochs != 20:
            raise ValueError("RQ7 uses the completed 20-epoch linear horizon")
        if self.implementation_revision not in (1, 2, 3, 4, 5, 6, 7):
            raise ValueError("unknown RQ7 implementation revision")
        if (
            self.implementation_revision
            != current_implementation_revision(self.treatment)
            and not self._is_historical_diagnostic()
            and not (_allow_historical_native and self._is_historical_native_combined())
        ):
            raise ValueError(
                "historical RQ7 implementation revisions are restricted to the "
                "completed 50M diagnostic surfaces"
            )
        if (
            not isinstance(self.deep_lr, (int, float))
            or isinstance(self.deep_lr, bool)
            or not math.isfinite(self.deep_lr)
            or self.deep_lr <= 0
        ):
            raise ValueError("RQ7 deep LR must be finite and positive")
        if self.stage == "diagnostic":
            if self.dataset_size != "50m" or self.seed != 42 or self.deep_lr != 0.012:
                raise ValueError("invalid RQ7 diagnostic candidate")
        elif self.stage == "initial":
            if (
                self.dataset_size != "500m"
                or self.treatment not in _PRIMARY_TREATMENTS
                or self.seed != 42
                or self.deep_lr not in _INITIAL_DEEP_LRS
            ):
                raise ValueError("invalid RQ7 initial candidate")
        elif self.stage == "rope_base":
            if (
                self.dataset_size != "500m"
                or self.treatment not in _ROPE_BASE_TREATMENTS
                or self.seed != 42
                or self.deep_lr not in _INITIAL_DEEP_LRS
            ):
                raise ValueError("invalid RQ7 RoPE-base candidate")
        elif self.stage == "rope_base_extension":
            if (
                self.dataset_size != "500m"
                or self.treatment not in _OUTER_ROPE_BASE_TREATMENTS
                or self.seed != 42
                or self.deep_lr not in _INITIAL_DEEP_LRS
            ):
                raise ValueError("invalid RQ7 outer RoPE-base candidate")
        elif self.stage == "boundary":
            self._validate_boundary()
        elif self.stage == "confirmation":
            if (
                self.dataset_size != "500m"
                or self.seed not in (43, 44)
                or not _is_grid_rate(self.deep_lr)
            ):
                raise ValueError("invalid RQ7 confirmation candidate")
        else:
            raise ValueError("invalid RQ7 candidate stage")
        if self.stage != "boundary" and (
            self.boundary_side is not None or self.boundary_step is not None
        ):
            raise ValueError("only boundary candidates carry a boundary coordinate")

    def _validate_boundary(self) -> None:
        if (
            self.dataset_size != "500m"
            or self.seed != 42
            or self.boundary_side not in ("low", "high")
            or not isinstance(self.boundary_step, int)
            or isinstance(self.boundary_step, bool)
            or self.boundary_step < 1
            or self.deep_lr != _boundary_deep_lr(self.boundary_side, self.boundary_step)
        ):
            raise ValueError("invalid RQ7 boundary candidate")

    def _is_historical_diagnostic(self) -> bool:
        if not (
            self.stage == "diagnostic"
            and self.dataset_size == "50m"
            and self.seed == 42
            and self.deep_lr == 0.012
        ):
            return False
        position = self.position
        combined = position.learned_positions == ("forward", "reverse")
        return (
            position.learned_position_fusion == "concat"
            and self.implementation_revision in (1, 2)
        ) or (
            combined
            and self.implementation_revision
            in (
                (1, 2, 3, 4, 5, 6)
                if position.learned_position_fusion == "concat"
                else (1, 4, 5, 6)
            )
        )

    def _is_historical_native_combined(self) -> bool:
        if self.dataset_size != "500m":
            return False
        position = self.position
        if position.learned_positions != ("forward", "reverse"):
            return False
        return self.implementation_revision in (
            (1, 2, 3, 4, 5, 6)
            if position.learned_position_fusion == "concat"
            else (1, 4, 5, 6)
        )

    @property
    def position(self) -> PositionTreatment:
        return _TREATMENTS[self.treatment]

    @property
    def run_name(self) -> str:
        if self.stage == "boundary":
            stage = f"boundary{self.boundary_side}{self.boundary_step}"
        elif self.stage == "confirmation":
            stage = "confirm"
        else:
            stage = self.stage
        return "_".join(
            (
                "g1",
                "rq7",
                self.treatment,
                f"e{_slug(self.embedding_lr)}",
                f"d{_slug(self.deep_lr)}",
                f"b{self.batch_size}",
                f"seed{self.seed}",
                f"h{self.horizon_epochs}",
                stage,
                f"ts{GENERATION_TRAINING_SEMANTICS_REVISION}",
                f"r{self.implementation_revision}",
                self.dataset_size,
            )
        )

    def environment(self) -> dict[str, str]:
        return {"G1_RQ7_RUN": self.run_name}


@cache
def diagnostic_candidates() -> tuple[Rq7Candidate, ...]:
    candidates = tuple(
        _current_candidate(treatment, 0.012, "50m", "diagnostic")
        for treatment in _PRIMARY_TREATMENTS | _ROPE_BASE_TREATMENTS
    )
    _require_unique_size(candidates, 14, "diagnostic")
    return candidates


@cache
def zero_reverse_diagnostic_candidates() -> tuple[Rq7Candidate, ...]:
    candidates = tuple(
        Rq7Candidate(
            treatment,
            0.012,
            "50m",
            "diagnostic",
            implementation_revision=4,
        )
        for treatment, position in _PRIMARY_TREATMENTS.items()
        if position.learned_positions == ("forward", "reverse")
    )
    _require_unique_size(candidates, 4, "zero-reverse diagnostic")
    return candidates


@cache
def bounded_reverse_diagnostic_candidates() -> tuple[Rq7Candidate, ...]:
    candidates = tuple(
        candidate
        for candidate in diagnostic_candidates()
        if candidate.implementation_revision == 7
    )
    _require_unique_size(candidates, 4, "bounded-reverse diagnostic")
    return candidates


@cache
def bounded_reverse_r5_diagnostic_candidates() -> tuple[Rq7Candidate, ...]:
    candidates = tuple(
        Rq7Candidate(
            treatment,
            0.012,
            "50m",
            "diagnostic",
            implementation_revision=5,
        )
        for treatment, position in _PRIMARY_TREATMENTS.items()
        if position.learned_positions == ("forward", "reverse")
    )
    _require_unique_size(candidates, 4, "bounded-reverse r5 diagnostic")
    return candidates


@cache
def bounded_reverse_r6_diagnostic_candidates() -> tuple[Rq7Candidate, ...]:
    candidates = tuple(
        Rq7Candidate(
            treatment,
            0.012,
            "50m",
            "diagnostic",
            implementation_revision=6,
        )
        for treatment, position in _PRIMARY_TREATMENTS.items()
        if position.learned_positions == ("forward", "reverse")
    )
    _require_unique_size(candidates, 4, "bounded-reverse r6 diagnostic")
    return candidates


@cache
def legacy_concat_diagnostic_candidates(
    revision: LegacyConcatRevision = 1,
) -> tuple[Rq7Candidate, ...]:
    if revision not in (1, 2):
        raise ValueError("legacy RQ7 concat revision must be 1 or 2")
    candidates = tuple(
        Rq7Candidate(
            treatment,
            0.012,
            "50m",
            "diagnostic",
            implementation_revision=revision,
        )
        for treatment, position in _PRIMARY_TREATMENTS.items()
        if position.learned_position_fusion == "concat"
    )
    _require_unique_size(candidates, 4, "legacy concat diagnostic")
    return candidates


@cache
def historical_combined_diagnostic_candidates() -> tuple[Rq7Candidate, ...]:
    candidates = tuple(
        Rq7Candidate(
            treatment,
            0.012,
            "50m",
            "diagnostic",
            implementation_revision=revision,
        )
        for treatment, position in _PRIMARY_TREATMENTS.items()
        if position.learned_positions == ("forward", "reverse")
        for revision in (
            (1, 2, 3, 4, 5, 6)
            if position.learned_position_fusion == "concat"
            else (1, 4, 5, 6)
        )
    )
    _require_unique_size(candidates, 20, "historical combined diagnostic")
    return candidates


@cache
def initial_candidates() -> tuple[Rq7Candidate, ...]:
    candidates = tuple(
        _current_candidate(treatment, deep_lr, "500m", "initial")
        for treatment in _PRIMARY_TREATMENTS
        for deep_lr in _INITIAL_DEEP_LRS
    )
    _require_unique_size(candidates, 36, "initial")
    return candidates


@cache
def rope_base_candidates() -> tuple[Rq7Candidate, ...]:
    candidates = tuple(
        Rq7Candidate(treatment, deep_lr, "500m", "rope_base")
        for treatment in _ROPE_BASE_TREATMENTS
        for deep_lr in _INITIAL_DEEP_LRS
    )
    _require_unique_size(candidates, 6, "RoPE-base")
    return candidates


@cache
def make_rope_base_extension_candidates(
    side: BoundarySide,
) -> tuple[Rq7Candidate, ...]:
    treatment = {
        "low": "rope_forward_base10",
        "high": "rope_forward_base100000",
    }[side]
    candidates = tuple(
        Rq7Candidate(treatment, deep_lr, "500m", "rope_base_extension")
        for deep_lr in _INITIAL_DEEP_LRS
    )
    _require_unique_size(candidates, 3, "outer RoPE-base")
    return candidates


def make_boundary_candidate(
    surface: Rq7Candidate, side: BoundarySide, step: int
) -> Rq7Candidate:
    if surface.dataset_size != "500m" or surface.stage not in (
        "initial",
        "rope_base",
        "rope_base_extension",
        "boundary",
    ):
        raise ValueError("RQ7 boundary extensions require a native-500M surface")
    if not isinstance(step, int) or isinstance(step, bool) or step < 1:
        raise ValueError("boundary step must be a positive integer")
    return Rq7Candidate(
        treatment=surface.treatment,
        deep_lr=_boundary_deep_lr(side, step),
        dataset_size="500m",
        stage="boundary",
        boundary_side=side,
        boundary_step=step,
        implementation_revision=surface.implementation_revision,
    )


def make_confirmation_candidate(
    winner: Rq7Candidate, seed: Literal[43, 44]
) -> Rq7Candidate:
    if seed not in (43, 44):
        raise ValueError("RQ7 confirmation seed must be seed 43 or 44")
    if winner.dataset_size != "500m" or winner.stage not in (
        "initial",
        "rope_base",
        "rope_base_extension",
        "boundary",
    ):
        raise ValueError("RQ7 confirmations require a native-500M winner")
    return Rq7Candidate(
        treatment=winner.treatment,
        deep_lr=winner.deep_lr,
        dataset_size="500m",
        stage="confirmation",
        seed=seed,
        implementation_revision=winner.implementation_revision,
    )


def candidate_by_run(run_name: str) -> Rq7Candidate:
    declared = {
        candidate.run_name: candidate
        for candidate in diagnostic_candidates()
        + legacy_concat_diagnostic_candidates(1)
        + legacy_concat_diagnostic_candidates(2)
        + historical_combined_diagnostic_candidates()
        + initial_candidates()
        + rope_base_candidates()
        + make_rope_base_extension_candidates("low")
        + make_rope_base_extension_candidates("high")
    }
    if run_name in declared:
        return declared[run_name]
    match = re.fullmatch(
        r"g1_rq7_(.+)_e([^_]+)_d([^_]+)_b(\d+)_seed(\d+)_h(\d+)_"
        r"(initial|boundary(low|high)(\d+)|confirm)_ts(\d+)_r(\d+)_(500m)",
        run_name,
    )
    if match is None:
        raise ValueError(f"unknown RQ7 candidate run {run_name!r}")
    (
        treatment,
        embedding_lr,
        deep_lr,
        batch_size,
        seed,
        horizon_epochs,
        stage_token,
        boundary_side,
        boundary_step,
        semantics_revision,
        implementation_revision,
        dataset_size,
    ) = match.groups()
    if int(semantics_revision) != GENERATION_TRAINING_SEMANTICS_REVISION:
        raise ValueError(f"unknown RQ7 candidate run {run_name!r}")
    try:
        candidate = Rq7Candidate(
            treatment=treatment,
            deep_lr=_unslug(deep_lr),
            dataset_size=dataset_size,
            stage=(
                "confirmation"
                if stage_token == "confirm"
                else "initial" if stage_token == "initial" else "boundary"
            ),
            embedding_lr=_unslug(embedding_lr),
            batch_size=int(batch_size),
            seed=int(seed),
            horizon_epochs=int(horizon_epochs),
            implementation_revision=int(implementation_revision),
            boundary_side=boundary_side,
            boundary_step=None if boundary_step is None else int(boundary_step),
            _allow_historical_native=True,
        )
    except ValueError as error:
        raise ValueError(f"unknown RQ7 candidate run {run_name!r}") from error
    if candidate.run_name != run_name:
        raise ValueError(f"unknown RQ7 candidate run {run_name!r}")
    return candidate


def _current_candidate(
    treatment: str,
    deep_lr: float,
    dataset_size: DatasetSize,
    stage: Stage,
) -> Rq7Candidate:
    return Rq7Candidate(
        treatment,
        deep_lr,
        dataset_size,
        stage,
        implementation_revision=current_implementation_revision(treatment),
    )


def current_implementation_revision(treatment: str) -> ImplementationRevision:
    position = _TREATMENTS[treatment]
    if position.learned_positions == ("forward", "reverse"):
        return 7
    return 3 if position.learned_position_fusion == "concat" else 1


def _require_unique_size(
    candidates: tuple[Rq7Candidate, ...], expected: int, label: str
) -> None:
    if (
        len(candidates) != expected
        or len({item.run_name for item in candidates}) != expected
    ):
        raise AssertionError(f"RQ7 {label} surface is not {expected} unique runs")


def _slug(value: float) -> str:
    return f"{value:.12g}".replace("-", "m").replace(".", "p")


def _unslug(value: str) -> float:
    return float(value.replace("m", "-").replace("p", "."))


def _boundary_deep_lr(side: BoundarySide, step: int) -> float:
    value = 0.006 / (2**step) if side == "low" else 0.024 * (2**step)
    return float(f"{value:.12g}")


def _is_grid_rate(value: float) -> bool:
    if value in _INITIAL_DEEP_LRS:
        return True
    anchor = 0.006 if value < 0.006 else 0.024
    ratio = anchor / value if value < 0.006 else value / anchor
    exponent = math.log2(ratio)
    return exponent >= 1 and exponent.is_integer()
