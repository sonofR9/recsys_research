from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cache
import math
import re
from typing import Literal


Family = Literal[
    "uniform_catalog",
    "streaming_global_q",
    "popularity_global_q",
    "aggregate_uniform_streaming_global_q",
    "aggregate_uniform_streaming_global_q_negative_only",
]
PrimaryFamily = Literal[
    "uniform_catalog",
    "streaming_global_q",
    "popularity_global_q",
    "aggregate_uniform_streaming_global_q",
]
Stage = Literal["joint", "local_lr", "diagnostic", "boundary"]
BoundaryAxis = Literal["negative_count", "alpha", "uniform_fraction", "deep_lr"]
BoundarySide = Literal["low", "high"]

PRIMARY_FAMILIES: tuple[PrimaryFamily, ...] = (
    "uniform_catalog",
    "streaming_global_q",
    "popularity_global_q",
    "aggregate_uniform_streaming_global_q",
)
DEEP_LRS = (0.006, 0.012, 0.024)
IMPLEMENTATION_REVISION = 1

_PLAIN = (
    (0.006, 512),
    (0.012, 1024),
    (0.024, 2048),
    (0.006, 2048),
    (0.012, 512),
    (0.024, 1024),
)
_STREAMING = (
    (0.006, 512, 0.005),
    (0.012, 1024, 0.01),
    (0.024, 2048, 0.02),
    (0.006, 2048, 0.01),
    (0.012, 512, 0.02),
    (0.024, 1024, 0.005),
)
_MIXTURE = (
    (0.006, 512, 0.005, 0.25),
    (0.012, 1024, 0.02, 0.5),
    (0.024, 2048, 0.01, 0.75),
    (0.006, 2048, 0.02, 0.5),
    (0.012, 512, 0.005, 0.75),
    (0.024, 1024, 0.01, 0.25),
)


@dataclass(frozen=True)
class Rq11Candidate:
    family: Family
    deep_lr: float
    negative_count: int
    alpha: float | None = None
    uniform_fraction: float | None = None
    stage: Stage = "joint"
    dataset_size: Literal["500m"] = "500m"
    embedding_lr: float = 0.064
    batch_size: int = 1280
    seed: int = 42
    horizon_epochs: int = 20
    boundary_axis: BoundaryAxis | None = None
    boundary_side: BoundarySide | None = None
    boundary_step: int | None = None
    implementation_revision: int = IMPLEMENTATION_REVISION

    def __post_init__(self) -> None:
        if self.family not in (
            *PRIMARY_FAMILIES,
            "aggregate_uniform_streaming_global_q_negative_only",
        ):
            raise ValueError(f"unknown RQ11 family {self.family!r}")
        if (
            self.dataset_size != "500m"
            or self.embedding_lr != 0.064
            or self.batch_size != 1280
            or self.seed != 42
            or self.horizon_epochs != 20
            or self.implementation_revision != IMPLEMENTATION_REVISION
        ):
            raise ValueError("RQ11 uses fixed native-500M training invariants")
        if not _positive_finite(self.deep_lr):
            raise ValueError("RQ11 deep LR must be finite and positive")
        if (
            not isinstance(self.negative_count, int)
            or isinstance(self.negative_count, bool)
            or self.negative_count < 2
        ):
            raise ValueError("RQ11 negative count must be an integer of at least two")
        needs_alpha = self.family in {
            "streaming_global_q",
            "aggregate_uniform_streaming_global_q",
            "aggregate_uniform_streaming_global_q_negative_only",
        }
        needs_fraction = self.family in {
            "aggregate_uniform_streaming_global_q",
            "aggregate_uniform_streaming_global_q_negative_only",
        }
        if needs_alpha != (self.alpha is not None) or (
            self.alpha is not None and not _positive_finite(self.alpha)
        ):
            raise ValueError("RQ11 alpha does not match the family")
        if needs_fraction != (self.uniform_fraction is not None) or (
            self.uniform_fraction is not None and not 0 < self.uniform_fraction < 1
        ):
            raise ValueError("RQ11 uniform fraction does not match the family")
        if self.family.endswith("negative_only") and self.stage != "diagnostic":
            raise ValueError("negative-only aggregate mixture is diagnostic only")
        boundary_fields = (self.boundary_axis, self.boundary_side, self.boundary_step)
        if self.stage == "boundary":
            if (
                self.family not in PRIMARY_FAMILIES
                or self.boundary_axis is None
                or self.boundary_side not in ("low", "high")
                or not isinstance(self.boundary_step, int)
                or isinstance(self.boundary_step, bool)
                or self.boundary_step < 1
            ):
                raise ValueError("invalid RQ11 boundary identity")
        elif any(value is not None for value in boundary_fields):
            raise ValueError("only boundary candidates carry a boundary identity")

    @property
    def secondary_coordinates(self) -> tuple[int, float | None, float | None]:
        return self.negative_count, self.alpha, self.uniform_fraction

    @property
    def run_name(self) -> str:
        alpha = "none" if self.alpha is None else _slug(self.alpha)
        fraction = (
            "none" if self.uniform_fraction is None else _slug(self.uniform_fraction)
        )
        stage = self.stage
        if self.stage == "boundary":
            stage = f"boundary-{self.boundary_axis}-{self.boundary_side}-{self.boundary_step}"
        return (
            f"g1_rq11_{self.family}_d{_slug(self.deep_lr)}_n{self.negative_count}_"
            f"a{alpha}_f{fraction}_seed{self.seed}_h{self.horizon_epochs}_{stage}_"
            f"r{self.implementation_revision}_{self.dataset_size}"
        )

    def environment(self) -> dict[str, str]:
        return {"G1_RQ11_RUN": self.run_name}


@cache
def initial_candidates() -> tuple[Rq11Candidate, ...]:
    candidates = (
        *(Rq11Candidate("uniform_catalog", lr, count) for lr, count in _PLAIN),
        *(Rq11Candidate("popularity_global_q", lr, count) for lr, count in _PLAIN),
        *(
            Rq11Candidate("streaming_global_q", lr, count, alpha)
            for lr, count, alpha in _STREAMING
        ),
        *(
            Rq11Candidate(
                "aggregate_uniform_streaming_global_q", lr, count, alpha, fraction
            )
            for lr, count, alpha, fraction in _MIXTURE
        ),
    )
    _require_unique(candidates, 24, "initial")
    return candidates


def manifest_payload() -> dict[str, object]:
    joint_search = {}
    for family in PRIMARY_FAMILIES:
        rows = []
        for candidate in initial_candidates():
            if candidate.family != family:
                continue
            row: list[int | float] = [
                candidate.deep_lr,
                candidate.negative_count,
            ]
            if candidate.alpha is not None:
                row.append(candidate.alpha)
            if candidate.uniform_fraction is not None:
                row.append(candidate.uniform_fraction)
            rows.append(row)
        joint_search[family] = rows
    return {
        "dataset_size": "500m",
        "fixed": {
            "batch_size": 1280,
            "embedding_lr": 0.064,
            "horizon_epochs": 20,
            "seed": 42,
        },
        "joint_search": joint_search,
    }


def local_lr_candidates(winner: Rq11Candidate) -> tuple[Rq11Candidate, ...]:
    if winner.family not in PRIMARY_FAMILIES:
        raise ValueError("local LR completion requires a primary family")
    candidates = []
    for deep_lr in DEEP_LRS:
        existing = (
            winner
            if deep_lr == winner.deep_lr
            else next(
                (
                    candidate
                    for candidate in initial_candidates()
                    if candidate.family == winner.family
                    and candidate.deep_lr == deep_lr
                    and candidate.secondary_coordinates == winner.secondary_coordinates
                ),
                None,
            )
        )
        candidates.append(
            existing
            or replace(
                winner,
                deep_lr=deep_lr,
                stage="local_lr",
                boundary_axis=None,
                boundary_side=None,
                boundary_step=None,
            )
        )
    return tuple(candidates)


def diagnostic_candidates(primary_winner: Rq11Candidate) -> tuple[Rq11Candidate, ...]:
    if primary_winner.family != "aggregate_uniform_streaming_global_q":
        raise ValueError("RQ11 diagnostics require the primary mixture winner")
    return tuple(
        Rq11Candidate(
            "aggregate_uniform_streaming_global_q_negative_only",
            deep_lr,
            primary_winner.negative_count,
            primary_winner.alpha,
            primary_winner.uniform_fraction,
            stage="diagnostic",
        )
        for deep_lr in DEEP_LRS
    )


def make_boundary_candidate(
    winner: Rq11Candidate,
    axis: BoundaryAxis,
    side: BoundarySide,
    step: int,
) -> Rq11Candidate:
    if winner.family not in PRIMARY_FAMILIES or side not in ("low", "high"):
        raise ValueError("RQ11 boundaries require a primary family")
    if not isinstance(step, int) or isinstance(step, bool) or step < 1:
        raise ValueError("boundary step must be a positive integer")
    if axis == "alpha" and winner.alpha is None:
        raise ValueError("only streaming families have an alpha boundary")
    if axis == "uniform_fraction" and winner.uniform_fraction is None:
        raise ValueError("only the aggregate family has a fraction boundary")
    changes: dict[str, object] = {}
    if axis == "negative_count":
        changes[axis] = int(512 / (2**step)) if side == "low" else 2048 * (2**step)
    elif axis == "alpha":
        changes[axis] = 0.005 / (2**step) if side == "low" else 0.02 * (2**step)
    elif axis == "uniform_fraction":
        changes[axis] = 0.25 / (2**step) if side == "low" else 1 - 0.25 / (2**step)
    elif axis == "deep_lr":
        changes[axis] = 0.006 / (2**step) if side == "low" else 0.024 * (2**step)
    else:
        raise ValueError(f"unknown RQ11 boundary axis {axis!r}")
    return replace(
        winner,
        **changes,
        stage="boundary",
        boundary_axis=axis,
        boundary_side=side,
        boundary_step=step,
    )


def candidate_by_run(run_name: str) -> Rq11Candidate:
    declared = {candidate.run_name: candidate for candidate in initial_candidates()}
    if run_name in declared:
        return declared[run_name]
    match = re.fullmatch(
        r"g1_rq11_(.+)_d([^_]+)_n(\d+)_a([^_]+)_f([^_]+)_seed(\d+)_h(\d+)_"
        r"(local_lr|diagnostic|boundary-(negative_count|alpha|uniform_fraction|deep_lr)-(low|high)-(\d+))_r(\d+)_(500m)",
        run_name,
    )
    if match is None:
        raise ValueError(f"unknown RQ11 candidate run {run_name!r}")
    (
        family,
        deep_lr,
        count,
        alpha,
        fraction,
        seed,
        horizon,
        stage_token,
        axis,
        side,
        step,
        revision,
        dataset,
    ) = match.groups()
    try:
        candidate = Rq11Candidate(
            family=family,  # type: ignore[arg-type]
            deep_lr=_unslug(deep_lr),
            negative_count=int(count),
            alpha=None if alpha == "none" else _unslug(alpha),
            uniform_fraction=None if fraction == "none" else _unslug(fraction),
            stage="boundary" if stage_token.startswith("boundary-") else stage_token,  # type: ignore[arg-type]
            dataset_size=dataset,  # type: ignore[arg-type]
            seed=int(seed),
            horizon_epochs=int(horizon),
            boundary_axis=axis,  # type: ignore[arg-type]
            boundary_side=side,  # type: ignore[arg-type]
            boundary_step=None if step is None else int(step),
            implementation_revision=int(revision),
        )
    except ValueError as error:
        raise ValueError(f"unknown RQ11 candidate run {run_name!r}") from error
    if candidate.run_name != run_name:
        raise ValueError(f"unknown RQ11 candidate run {run_name!r}")
    return candidate


def _positive_finite(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def _require_unique(
    candidates: tuple[Rq11Candidate, ...], expected: int, label: str
) -> None:
    if (
        len(candidates) != expected
        or len({item.run_name for item in candidates}) != expected
    ):
        raise AssertionError(f"RQ11 {label} surface is not {expected} unique runs")


def _slug(value: float) -> str:
    return f"{value:.12g}".replace("-", "m").replace(".", "p")


def _unslug(value: str) -> float:
    return float(value.replace("m", "-").replace("p", "."))
