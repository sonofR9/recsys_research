from __future__ import annotations

from dataclasses import dataclass
from functools import cache
import math
import re
from typing import Literal

from dcn.training_metadata import GENERATION_TRAINING_SEMANTICS_REVISION


Study = Literal["query", "sequence"]
QueryMethod = Literal["standard", "end_only", "interleaved"]
PositionMethod = Literal["learned_forward", "alibi", "rope_reverse_alibi"]
Stage = Literal["initial", "boundary", "confirmation"]
BoundarySide = Literal["low", "high"]

_DEEP_LRS = (0.006, 0.012, 0.024)
_SEQUENCE_LENGTHS = (12, 25, 50, 100, 128, 200, 256, 512)


@dataclass(frozen=True)
class Rq8Candidate:
    study: Study
    query_method: QueryMethod
    position_method: PositionMethod
    max_seq_len: int
    deep_lr: float
    dataset_size: str = "500m"
    embedding_lr: float = 0.064
    batch_size: int = 1280
    seed: int = 42
    cap_epochs: int = 20
    stage: Stage = "initial"
    boundary_side: BoundarySide | None = None
    boundary_step: int | None = None

    def __post_init__(self) -> None:
        if self.dataset_size != "500m":
            raise ValueError("RQ8 reinvestigation uses native Yambda-500M only")
        if self.embedding_lr != 0.064 or self.batch_size != 1280:
            raise ValueError("RQ8 embedding LR and physical batch are fixed")
        if self.cap_epochs != 20:
            raise ValueError("RQ8 uses the declared 20-epoch linear horizon")
        if (
            not isinstance(self.deep_lr, (int, float))
            or isinstance(self.deep_lr, bool)
            or not math.isfinite(self.deep_lr)
            or self.deep_lr <= 0
        ):
            raise ValueError("RQ8 deep LR must be finite and positive")
        if self.stage not in ("initial", "boundary", "confirmation"):
            raise ValueError("invalid RQ8 candidate stage")
        if self.study == "query":
            if (
                self.query_method not in ("standard", "end_only", "interleaved")
                or self.position_method != "learned_forward"
                or self.max_seq_len != 128
            ):
                raise ValueError("invalid RQ8 query surface")
        elif self.study == "sequence":
            if (
                self.query_method != "standard"
                or self.position_method not in ("alibi", "rope_reverse_alibi")
                or self.max_seq_len not in _SEQUENCE_LENGTHS
            ):
                raise ValueError("invalid RQ8 sequence surface")
        else:
            raise ValueError("invalid RQ8 study")
        if self.stage == "initial" and (
            self.seed != 42 or self.deep_lr not in _DEEP_LRS
        ):
            raise ValueError("invalid RQ8 initial candidate")
        if self.stage == "boundary":
            if (
                self.seed != 42
                or self.boundary_side not in ("low", "high")
                or not isinstance(self.boundary_step, int)
                or isinstance(self.boundary_step, bool)
                or self.boundary_step < 1
            ):
                raise ValueError("invalid RQ8 boundary candidate")
            expected_lr = _boundary_deep_lr(
                self.boundary_side, self.boundary_step
            )
            if self.deep_lr != expected_lr:
                raise ValueError("RQ8 boundary LR does not match its log2 step")
        elif self.boundary_side is not None or self.boundary_step is not None:
            raise ValueError("only boundary candidates carry a boundary coordinate")
        if self.stage == "confirmation" and (
            self.study != "query" or self.seed not in (43, 44)
        ):
            raise ValueError("RQ8 confirmations are query-only seeds 43 and 44")
        if self.stage == "confirmation" and not _is_approved_grid_rate(self.deep_lr):
            raise ValueError("RQ8 confirmation LR is outside the tested log2 grid")

    @property
    def surface_key(self) -> tuple[Study, str, int]:
        treatment = (
            self.query_method if self.study == "query" else self.position_method
        )
        return self.study, treatment, self.max_seq_len

    @property
    def run_name(self) -> str:
        treatment = (
            self.query_method if self.study == "query" else self.position_method
        )
        components = ["g1", "rq8", self.study]
        if self.study == "sequence":
            components.append("fullcausal")
        components.extend(
            [
                treatment,
                f"s{self.max_seq_len}",
                f"e{_slug(self.embedding_lr)}",
                f"d{_slug(self.deep_lr)}",
                f"b{self.batch_size}",
                f"seed{self.seed}",
                f"cap{self.cap_epochs}",
                f"ts{GENERATION_TRAINING_SEMANTICS_REVISION}",
            ]
        )
        if self.stage == "boundary":
            components.append(f"boundary{self.boundary_side}{self.boundary_step}")
        elif self.stage == "confirmation":
            components.append("confirm")
        components.extend(
            ("r1" if self.study == "query" else "r2", self.dataset_size)
        )
        return "_".join(components)

    def environment(self) -> dict[str, str]:
        return {"G1_RQ8_RUN": self.run_name}


@cache
def query_initial_candidates() -> tuple[Rq8Candidate, ...]:
    return tuple(
        Rq8Candidate(
            study="query",
            query_method=method,
            position_method="learned_forward",
            max_seq_len=128,
            deep_lr=deep_lr,
        )
        for method in ("standard", "end_only", "interleaved")
        for deep_lr in _DEEP_LRS
    )


@cache
def sequence_initial_candidates() -> tuple[Rq8Candidate, ...]:
    candidates = tuple(
        Rq8Candidate(
            study="sequence",
            query_method="standard",
            position_method=position,
            max_seq_len=max_seq_len,
            deep_lr=deep_lr,
        )
        for position in ("alibi", "rope_reverse_alibi")
        for max_seq_len in _SEQUENCE_LENGTHS
        for deep_lr in _DEEP_LRS
    )
    if len(candidates) != 48 or len({item.run_name for item in candidates}) != 48:
        raise AssertionError("RQ8 corrected sequence surface is not 48 unique runs")
    return candidates


@cache
def initial_candidates() -> tuple[Rq8Candidate, ...]:
    candidates = query_initial_candidates() + sequence_initial_candidates()
    if len(candidates) != 57 or len({item.run_name for item in candidates}) != 57:
        raise AssertionError("RQ8 initial surface is not 57 unique runs")
    return candidates


def candidate_by_run(run_name: str) -> Rq8Candidate:
    initial = {candidate.run_name: candidate for candidate in initial_candidates()}
    if run_name in initial:
        return initial[run_name]
    match = re.fullmatch(
        r"g1_rq8_(query|sequence_fullcausal)_(.+)_s(\d+)_e([^_]+)_d([^_]+)_"
        r"b(\d+)_seed(\d+)_cap(\d+)_ts(\d+)_"
        r"(boundary(low|high)(\d+)|confirm)_r([12])_(500m)",
        run_name,
    )
    if match is None:
        raise ValueError(f"unknown RQ8 candidate run {run_name!r}")
    (
        study,
        treatment,
        max_seq_len,
        embedding_lr,
        deep_lr,
        batch_size,
        seed,
        cap_epochs,
        semantics_revision,
        stage_token,
        boundary_side,
        boundary_step,
        protocol_revision,
        dataset_size,
    ) = match.groups()
    if int(semantics_revision) != GENERATION_TRAINING_SEMANTICS_REVISION:
        raise ValueError(f"unknown RQ8 candidate run {run_name!r}")
    if study == "query" and protocol_revision == "1" and treatment in (
        "standard",
        "end_only",
        "interleaved",
    ):
        query_method: QueryMethod = treatment
        position_method: PositionMethod = "learned_forward"
    elif (
        study == "sequence_fullcausal"
        and protocol_revision == "2"
        and treatment in ("alibi", "rope_reverse_alibi")
    ):
        study = "sequence"
        query_method = "standard"
        position_method = treatment
    else:
        raise ValueError(f"unknown RQ8 candidate run {run_name!r}")
    candidate = Rq8Candidate(
        study=study,
        query_method=query_method,
        position_method=position_method,
        max_seq_len=int(max_seq_len),
        deep_lr=_unslug(deep_lr),
        dataset_size=dataset_size,
        embedding_lr=_unslug(embedding_lr),
        batch_size=int(batch_size),
        seed=int(seed),
        cap_epochs=int(cap_epochs),
        stage="confirmation" if stage_token == "confirm" else "boundary",
        boundary_side=boundary_side,
        boundary_step=None if boundary_step is None else int(boundary_step),
    )
    if candidate.run_name != run_name:
        raise ValueError(f"unknown RQ8 candidate run {run_name!r}")
    return candidate


def make_boundary_candidate(
    surface: Rq8Candidate, side: BoundarySide, step: int
) -> Rq8Candidate:
    if not isinstance(step, int) or isinstance(step, bool) or step < 1:
        raise ValueError("boundary step must be a positive integer")
    deep_lr = _boundary_deep_lr(side, step)
    return Rq8Candidate(
        study=surface.study,
        query_method=surface.query_method,
        position_method=surface.position_method,
        max_seq_len=surface.max_seq_len,
        deep_lr=_canonical_float(deep_lr),
        stage="boundary",
        boundary_side=side,
        boundary_step=step,
    )


def make_confirmation_candidate(
    winner: Rq8Candidate, seed: Literal[43, 44]
) -> Rq8Candidate:
    if winner.study != "query":
        raise ValueError("sequence treatments do not receive seed repeats")
    return Rq8Candidate(
        study="query",
        query_method=winner.query_method,
        position_method=winner.position_method,
        max_seq_len=winner.max_seq_len,
        deep_lr=winner.deep_lr,
        seed=seed,
        stage="confirmation",
    )


def _slug(value: float) -> str:
    return f"{value:.12g}".replace("-", "m").replace(".", "p")


def _unslug(value: str) -> float:
    return float(value.replace("m", "-").replace("p", "."))


def _canonical_float(value: float) -> float:
    return float(f"{value:.12g}")


def _boundary_deep_lr(side: BoundarySide, step: int) -> float:
    value = 0.006 / (2**step) if side == "low" else 0.024 * (2**step)
    return _canonical_float(value)


def _is_approved_grid_rate(value: float) -> bool:
    if value in _DEEP_LRS:
        return True
    anchor = 0.006 if value < 0.006 else 0.024
    ratio = anchor / value if value < 0.006 else value / anchor
    exponent = math.log2(ratio)
    return exponent >= 1 and exponent.is_integer()
