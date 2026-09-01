from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cache
import math
import re
from typing import Literal


Treatment = Literal[
    "shared_cls_only",
    "distinct_cls_only",
    "shared_history",
    "distinct_history",
]
Stage = Literal["initial", "lr_boundary"]
Direction = Literal["low", "high"]

TREATMENTS: tuple[Treatment, ...] = (
    "shared_cls_only",
    "distinct_cls_only",
    "shared_history",
    "distinct_history",
)
DEEP_LRS = (0.000375, 0.00075, 0.0015)
EMBEDDING_LR = 0.00025
RQ15_REUSED_RUNS = {
    0.000375: (
        "g1_rq15_pretrained_finetune_e0p00025_d0p000375_b1280_seed42_h20_"
        "lrdeeplow1_r1_500m"
    ),
    0.00075: (
        "g1_rq15_pretrained_finetune_e0p00025_d0p00075_b1280_seed42_h20_"
        "lrembeddinglow7_r1_500m"
    ),
    0.0015: (
        "g1_rq15_pretrained_finetune_e0p00025_d0p0015_b1280_seed42_h20_"
        "lrembeddinglow7_r1_500m"
    ),
}


@dataclass(frozen=True)
class Rq14PretrainedCandidate:
    treatment: Treatment
    deep_lr: float
    dataset_size: Literal["500m"] = "500m"
    embedding_lr: float = EMBEDDING_LR
    batch_size: int = 1280
    seed: int = 42
    horizon_epochs: int = 20
    stage: Stage = "initial"
    boundary_direction: Direction | None = None
    boundary_step: int = 0

    def __post_init__(self) -> None:
        if self.treatment not in TREATMENTS:
            raise ValueError("unknown pretrained RQ14 treatment")
        if not math.isfinite(self.deep_lr) or self.deep_lr <= 0:
            raise ValueError("deep LR must be finite and positive")
        if (
            self.dataset_size != "500m"
            or self.embedding_lr != EMBEDDING_LR
            or self.batch_size != 1280
            or self.seed != 42
            or self.horizon_epochs != 20
        ):
            raise ValueError("pretrained RQ14 uses fixed native-500M invariants")
        if self.stage == "initial":
            if (
                self.deep_lr not in DEEP_LRS
                or self.boundary_direction is not None
                or self.boundary_step != 0
            ):
                raise ValueError("initial pretrained RQ14 candidate is off-grid")
        elif self.stage == "lr_boundary":
            if self.boundary_direction not in {"low", "high"} or self.boundary_step < 1:
                raise ValueError("boundary direction and positive step are required")
            anchor = min(DEEP_LRS) if self.boundary_direction == "low" else max(DEEP_LRS)
            factor = 2.0 ** (
                -self.boundary_step
                if self.boundary_direction == "low"
                else self.boundary_step
            )
            if self.deep_lr != anchor * factor:
                raise ValueError("boundary deep LR does not match its geometric step")
        else:
            raise ValueError("unknown pretrained RQ14 stage")

    @property
    def query_slots_shared(self) -> bool:
        return self.treatment.startswith("shared_")

    @property
    def include_history_memory(self) -> bool:
        return self.treatment.endswith("_history")

    @property
    def reused_rq15_run_name(self) -> str | None:
        if self.stage == "initial" and self.treatment == "distinct_cls_only":
            return RQ15_REUSED_RUNS[self.deep_lr]
        return None

    @property
    def run_name(self) -> str:
        boundary = (
            ""
            if self.stage == "initial"
            else f"_lrdeep{self.boundary_direction}{self.boundary_step}"
        )
        return (
            f"g1_rq14_pretrained_{self.treatment}_e{_slug(self.embedding_lr)}_"
            f"d{_slug(self.deep_lr)}_b{self.batch_size}_seed{self.seed}_"
            f"h{self.horizon_epochs}{boundary}_r1_{self.dataset_size}"
        )

    @property
    def artifact_run_name(self) -> str:
        return self.reused_rq15_run_name or self.run_name


@cache
def initial_candidates() -> tuple[Rq14PretrainedCandidate, ...]:
    result = tuple(
        Rq14PretrainedCandidate(treatment, deep_lr)
        for treatment in TREATMENTS
        for deep_lr in DEEP_LRS
    )
    if len(result) != 12 or len({item.run_name for item in result}) != 12:
        raise RuntimeError("pretrained RQ14 manifest must contain 12 unique cells")
    return result


@cache
def reused_candidates() -> tuple[Rq14PretrainedCandidate, ...]:
    result = tuple(item for item in initial_candidates() if item.reused_rq15_run_name)
    if len(result) != 3:
        raise RuntimeError("pretrained RQ14 must reuse exactly three RQ15 cells")
    return result


@cache
def launch_candidates() -> tuple[Rq14PretrainedCandidate, ...]:
    result = tuple(item for item in initial_candidates() if not item.reused_rq15_run_name)
    if len(result) != 9:
        raise RuntimeError("pretrained RQ14 launcher must contain nine new cells")
    return result


def make_boundary_candidate(
    anchor: Rq14PretrainedCandidate,
    direction: Direction,
    step: int,
) -> Rq14PretrainedCandidate:
    if anchor.stage != "initial":
        raise ValueError("boundary anchor must come from the initial surface")
    edge = min(DEEP_LRS) if direction == "low" else max(DEEP_LRS)
    factor = 2.0 ** (-step if direction == "low" else step)
    return replace(
        anchor,
        deep_lr=edge * factor,
        stage="lr_boundary",
        boundary_direction=direction,
        boundary_step=step,
    )


def candidate_by_run(run_name: str) -> Rq14PretrainedCandidate:
    matches = [item for item in initial_candidates() if item.run_name == run_name]
    if len(matches) == 1:
        return matches[0]
    match = re.fullmatch(
        r"g1_rq14_pretrained_(?P<treatment>[a-z0-9_]+)_e0p00025_"
        r"d(?P<deep>[0-9a-z]+)_b1280_seed42_h20_"
        r"lrdeep(?P<direction>low|high)(?P<step>[1-9][0-9]*)_r1_500m",
        run_name,
    )
    if match is None:
        raise ValueError(f"unknown pretrained RQ14 run {run_name!r}")
    direction = match.group("direction")
    step = int(match.group("step"))
    anchor = next(
        item
        for item in initial_candidates()
        if item.treatment == match.group("treatment")
        and item.deep_lr == (min(DEEP_LRS) if direction == "low" else max(DEEP_LRS))
    )
    candidate = make_boundary_candidate(anchor, direction, step)
    if candidate.run_name != run_name or candidate.deep_lr != _unslug(match.group("deep")):
        raise ValueError(f"noncanonical pretrained RQ14 run {run_name!r}")
    return candidate


def _slug(value: float) -> str:
    return format(value, ".12g").replace(".", "p")


def _unslug(value: str) -> float:
    return float(value.replace("p", "."))
