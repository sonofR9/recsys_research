from __future__ import annotations

from dataclasses import dataclass
from functools import cache
import math
import re
from typing import Literal

from dcn.training_metadata import GENERATION_TRAINING_SEMANTICS_REVISION


Family = Literal[
    "input_output_only",
    "direct_add",
    "concat_residual",
    "gemma_ple",
]
Stage = Literal["initial", "selected_width_lr", "width_boundary", "lr_boundary"]
BoundarySide = Literal["low", "high"]

_INITIAL_DEEP_LRS = (0.006, 0.012, 0.024)
_CENTRAL_DEEP_LR = 0.012
_INITIAL_WIDTHS = {
    "concat_residual": (16, 32, 64),
    "gemma_ple": (8, 16, 32),
}
_RUN_PATTERN = re.compile(
    r"^g1_rq10_(?P<family>[a-z_]+)_w(?P<width>none|\d+)_"
    r"e(?P<embedding>[\dp]+)_d(?P<deep>[\dp]+)_b(?P<batch>\d+)_"
    r"seed(?P<seed>\d+)_l(?P<layers>\d+)_h(?P<horizon>\d+)_"
    r"(?P<stage>[a-z_]+)_ts(?P<semantics>\d+)_r1_(?P<dataset>50m|500m)$"
)


@dataclass(frozen=True)
class Rq10Candidate:
    family: Family
    feature_width: int | None
    deep_lr: float
    stage: Stage
    dataset_size: Literal["500m"] = "500m"
    embedding_lr: float = 0.064
    batch_size: int = 1280
    seed: int = 42
    num_layers: int = 4
    horizon_epochs: int = 20

    def __post_init__(self) -> None:
        if self.dataset_size != "500m":
            raise ValueError("RQ10 selection uses native Yambda-500M only")
        if self.embedding_lr != 0.064 or self.batch_size != 1280:
            raise ValueError("RQ10 embedding LR and batch are fixed")
        if self.seed != 42 or self.num_layers != 4 or self.horizon_epochs != 20:
            raise ValueError("RQ10 uses seed 42, four layers, and a 20-epoch horizon")
        if not math.isfinite(self.deep_lr) or self.deep_lr <= 0:
            raise ValueError("RQ10 deep LR must be finite and positive")
        if self.family == "input_output_only" and self.feature_width is not None:
            raise ValueError("the control has no per-layer feature width")
        if self.family == "direct_add" and self.feature_width != 64:
            raise ValueError("direct addition uses the full model width")
        if self.family in _INITIAL_WIDTHS and (
            not isinstance(self.feature_width, int) or self.feature_width <= 0
        ):
            raise ValueError("learned per-layer features need a positive width")
        self._validate_stage()

    def _validate_stage(self) -> None:
        if self.stage == "initial":
            if self.family in ("input_output_only", "direct_add"):
                valid = self.deep_lr in _INITIAL_DEEP_LRS
            else:
                valid = (
                    self.deep_lr == _CENTRAL_DEEP_LR
                    and self.feature_width in _INITIAL_WIDTHS[self.family]
                )
            if not valid:
                raise ValueError("candidate is outside the initial RQ10 surface")
            return
        if self.stage == "selected_width_lr":
            if self.family not in _INITIAL_WIDTHS or self.deep_lr not in (0.006, 0.024):
                raise ValueError("invalid selected-width LR candidate")
            return
        if self.stage == "width_boundary":
            if self.family not in _INITIAL_WIDTHS or self.deep_lr != _CENTRAL_DEEP_LR:
                raise ValueError("invalid RQ10 width-boundary candidate")
            return
        if self.stage == "lr_boundary":
            if self.deep_lr in _INITIAL_DEEP_LRS:
                raise ValueError("an LR-boundary candidate must extend the initial grid")
            return
        raise ValueError(f"unknown RQ10 stage {self.stage!r}")

    @property
    def run_name(self) -> str:
        width = "none" if self.feature_width is None else str(self.feature_width)
        return "_".join(
            (
                "g1",
                "rq10",
                self.family,
                f"w{width}",
                f"e{_slug(self.embedding_lr)}",
                f"d{_slug(self.deep_lr)}",
                f"b{self.batch_size}",
                f"seed{self.seed}",
                f"l{self.num_layers}",
                f"h{self.horizon_epochs}",
                self.stage,
                f"ts{GENERATION_TRAINING_SEMANTICS_REVISION}",
                "r1",
                self.dataset_size,
            )
        )


@cache
def initial_candidates() -> tuple[Rq10Candidate, ...]:
    candidates = tuple(
        Rq10Candidate(family, width, rate, "initial")
        for family, width in (
            ("input_output_only", None),
            ("direct_add", 64),
        )
        for rate in _INITIAL_DEEP_LRS
    ) + tuple(
        Rq10Candidate(family, width, _CENTRAL_DEEP_LR, "initial")
        for family, widths in _INITIAL_WIDTHS.items()
        for width in widths
    )
    _require_unique_size(candidates, 12)
    return candidates


def selected_width_lr_candidates(
    *, concat_feature_width: int, gemma_feature_width: int
) -> tuple[Rq10Candidate, ...]:
    candidates = tuple(
        Rq10Candidate(family, width, rate, "selected_width_lr")
        for family, width in (
            ("concat_residual", concat_feature_width),
            ("gemma_ple", gemma_feature_width),
        )
        for rate in (0.006, 0.024)
    )
    _require_unique_size(candidates, 4)
    return candidates


def make_width_boundary_candidate(
    source: Rq10Candidate, side: BoundarySide
) -> Rq10Candidate:
    if source.family not in _INITIAL_WIDTHS or source.stage != "initial":
        raise ValueError("width boundaries extend an initial compact-family candidate")
    widths = _INITIAL_WIDTHS[source.family]
    if source.feature_width != (widths[0] if side == "low" else widths[-1]):
        raise ValueError("width boundary must extend the winning outer width")
    width = source.feature_width // 2 if side == "low" else source.feature_width * 2
    return Rq10Candidate(source.family, width, _CENTRAL_DEEP_LR, "width_boundary")


def make_lr_boundary_candidate(
    source: Rq10Candidate, side: BoundarySide, step: int = 1
) -> Rq10Candidate:
    if not isinstance(step, int) or isinstance(step, bool) or step < 1:
        raise ValueError("LR boundary step must be a positive integer")
    edge = _INITIAL_DEEP_LRS[0] if side == "low" else _INITIAL_DEEP_LRS[-1]
    if source.deep_lr != edge:
        raise ValueError("LR boundary must extend the winning outer learning rate")
    deep_lr = edge / 2**step if side == "low" else edge * 2**step
    return Rq10Candidate(
        source.family,
        source.feature_width,
        deep_lr,
        "lr_boundary",
    )


def candidate_by_run(run_name: str) -> Rq10Candidate:
    match = _RUN_PATTERN.fullmatch(run_name)
    if match is None:
        raise ValueError(f"unknown RQ10 run {run_name!r}")
    values = match.groupdict()
    if int(values["semantics"]) != GENERATION_TRAINING_SEMANTICS_REVISION:
        raise ValueError("RQ10 run uses a historical training semantics revision")
    candidate = Rq10Candidate(
        family=values["family"],  # type: ignore[arg-type]
        feature_width=None if values["width"] == "none" else int(values["width"]),
        embedding_lr=_rate(values["embedding"]),
        deep_lr=_rate(values["deep"]),
        batch_size=int(values["batch"]),
        seed=int(values["seed"]),
        num_layers=int(values["layers"]),
        horizon_epochs=int(values["horizon"]),
        stage=values["stage"],  # type: ignore[arg-type]
        dataset_size=values["dataset"],  # type: ignore[arg-type]
    )
    if candidate.run_name != run_name:
        raise ValueError(f"non-canonical RQ10 run {run_name!r}")
    return candidate


def _require_unique_size(candidates: tuple[Rq10Candidate, ...], size: int) -> None:
    if len(candidates) != size or len({candidate.run_name for candidate in candidates}) != size:
        raise RuntimeError(f"expected {size} unique RQ10 candidates")


def _slug(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _rate(value: str) -> float:
    return float(value.replace("p", "."))
