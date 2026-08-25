from __future__ import annotations

from dataclasses import dataclass, fields, replace
import json
import math
from pathlib import Path
import random
import re

from neuralrec.run.callbacks.lr_schedule import OptimizerGroupScope, ScheduleShape


_DEEP_LRS = (0.003, 0.006, 0.012)
_SCOPES: tuple[OptimizerGroupScope, ...] = ("both", "deep_only")
_APPROVAL_RECORD = Path(__file__).parents[1] / "protocol/rq5_scheduler_approval.json"
_FIXED = (
    ("linear", "linear", 17, 0.0, 1),
    ("cosine", "cosine", 21, 0.0, 1),
    ("polynomial", "polynomial", 20, 0.0, 1),
    ("exponential", "exponential", 18, 0.0, 1),
    ("step", "step", 20, 0.0, 1),
    ("wsd", "warmup_stable_decay", 20, 0.05, 1),
    ("cosine_warmup5_cycles1", "cosine", 19, 0.05, 1),
    ("cosine_warmup5_cycles2", "cosine", 22, 0.05, 2),
    ("cosine_warmup5_cycles4", "cosine", 22, 0.05, 4),
)


@dataclass(frozen=True)
class Rq5Candidate:
    treatments: tuple[str, ...]
    shape: ScheduleShape
    scope: OptimizerGroupScope
    deep_lr: float
    horizon_epochs: int | None
    cap_epochs: int
    warmup_fraction: float = 0.0
    timescale_fraction: float | None = None
    cycles: int = 1
    attempt: int = 0
    probe: str | None = None
    dataset_size: str = "500m"
    seed: int = 42
    embedding_lr: float = 0.064

    @property
    def joint_fraction(self) -> float | None:
        return (
            self.timescale_fraction
            if self.shape == "inverse_sqrt"
            else self.warmup_fraction
        )

    @property
    def run_name(self) -> str:
        treatment = (
            "cosine_warmup5_shared" if len(self.treatments) > 1 else self.treatments[0]
        )
        parts = [
            "g1_rq5",
            treatment,
            self.scope,
            f"d{_slug(self.deep_lr)}",
        ]
        if self.shape == "inverse_sqrt":
            parts.append(f"t{_slug(self.timescale_fraction)}")
        elif "cosine_warmup_tuned" in self.treatments:
            parts.append(f"w{_slug(self.warmup_fraction)}")
        if self.probe is not None:
            if re.fullmatch(r"[a-z0-9]+", self.probe) is None:
                raise ValueError(f"invalid RQ5 probe identity {self.probe!r}")
            parts.append(f"p{self.probe}")
        if self.horizon_epochs is not None:
            parts.append(f"h{self.horizon_epochs}")
        parts.extend((f"cap{self.cap_epochs}", f"a{self.attempt}", "ts2", "r1", "500m"))
        return "_".join(parts)

    def environment(self) -> dict[str, str]:
        return {"G1_RQ5_RUN": self.run_name}


def initial_candidates() -> tuple[Rq5Candidate, ...]:
    candidates = [
        Rq5Candidate(
            treatments=("constant",),
            shape="constant",
            scope="both",
            deep_lr=deep_lr,
            horizon_epochs=None,
            cap_epochs=80,
        )
        for deep_lr in _DEEP_LRS
    ]
    for treatment, shape, horizon, warmup, cycles in _FIXED:
        for scope in _SCOPES:
            for deep_lr in _DEEP_LRS:
                candidates.append(
                    Rq5Candidate(
                        treatments=(treatment,),
                        shape=shape,
                        scope=scope,
                        deep_lr=deep_lr,
                        horizon_epochs=horizon,
                        cap_epochs=horizon,
                        warmup_fraction=warmup,
                        cycles=cycles,
                    )
                )
    pairs = _joint_pairs()
    for scope in _SCOPES:
        for deep_lr, fraction in pairs:
            candidates.append(
                Rq5Candidate(
                    treatments=("inverse_sqrt",),
                    shape="inverse_sqrt",
                    scope=scope,
                    deep_lr=deep_lr,
                    horizon_epochs=23,
                    cap_epochs=80,
                    timescale_fraction=fraction,
                )
            )
        for deep_lr, fraction in pairs:
            if deep_lr == 0.006 and fraction == 0.05:
                shared_index = next(
                    index
                    for index, candidate in enumerate(candidates)
                    if candidate.treatments == ("cosine_warmup5_cycles1",)
                    and candidate.scope == scope
                    and candidate.deep_lr == deep_lr
                )
                candidates[shared_index] = replace(
                    candidates[shared_index],
                    treatments=(
                        "cosine_warmup5_cycles1",
                        "cosine_warmup_tuned",
                    ),
                )
                continue
            candidates.append(
                Rq5Candidate(
                    treatments=("cosine_warmup_tuned",),
                    shape="cosine",
                    scope=scope,
                    deep_lr=deep_lr,
                    horizon_epochs=19,
                    cap_epochs=19,
                    warmup_fraction=fraction,
                )
            )
    result = tuple(candidates)
    if len(result) != 67 or len({candidate.run_name for candidate in result}) != 67:
        raise AssertionError("RQ5 initial candidate surface is not 67 unique runs")
    return result


def candidate_by_run(run_name: str) -> Rq5Candidate:
    initial_match = next(
        (
            candidate
            for candidate in initial_candidates()
            if candidate.run_name == run_name
        ),
        None,
    )
    if initial_match is not None:
        return initial_match
    treatments = ["cosine_warmup5_shared", "cosine_warmup_tuned", "inverse_sqrt"]
    treatments.extend(treatment for treatment, *_ in _FIXED)
    treatments.append("constant")
    alternatives = "|".join(
        re.escape(treatment)
        for treatment in sorted(set(treatments), key=len, reverse=True)
    )
    match = re.fullmatch(
        rf"g1_rq5_(?P<treatment>{alternatives})_"
        r"(?P<scope>both|deep_only)_d(?P<deep>[0-9emp]+)"
        r"(?:_t(?P<timescale>[0-9emp]+))?"
        r"(?:_w(?P<warmup>[0-9emp]+))?"
        r"(?:_p(?P<probe>[a-z0-9]+))?"
        r"(?:_h(?P<horizon>[1-9][0-9]*))?"
        r"_cap(?P<cap>[1-9][0-9]*)_a(?P<attempt>[0-5])_ts2_r1_500m",
        run_name,
    )
    if match is None:
        raise ValueError(f"unknown RQ5 candidate run {run_name!r}")
    treatment = match.group("treatment")
    scope = match.group("scope")
    deep_lr = _unslug(match.group("deep"))
    horizon = None if match.group("horizon") is None else int(match.group("horizon"))
    timescale = (
        None if match.group("timescale") is None else _unslug(match.group("timescale"))
    )
    encoded_warmup = (
        None if match.group("warmup") is None else _unslug(match.group("warmup"))
    )
    if treatment == "constant":
        shape, warmup, cycles, treatment_ids = "constant", 0.0, 1, (treatment,)
        if horizon is not None or scope != "both":
            raise ValueError(f"invalid constant RQ5 run {run_name!r}")
    elif treatment == "inverse_sqrt":
        shape, warmup, cycles, treatment_ids = "inverse_sqrt", 0.0, 1, (treatment,)
        if timescale is None or horizon is None:
            raise ValueError(f"invalid inverse-sqrt RQ5 run {run_name!r}")
    elif treatment == "cosine_warmup_tuned":
        shape, warmup, cycles, treatment_ids = "cosine", encoded_warmup, 1, (treatment,)
        if warmup is None or horizon is None:
            raise ValueError(f"invalid tuned-cosine RQ5 run {run_name!r}")
    elif treatment == "cosine_warmup5_shared":
        shape, warmup, cycles = "cosine", 0.05, 1
        treatment_ids = ("cosine_warmup5_cycles1", "cosine_warmup_tuned")
        if horizon is None or encoded_warmup != 0.05:
            raise ValueError(f"invalid shared-cosine RQ5 run {run_name!r}")
    else:
        fixed = next(spec for spec in _FIXED if spec[0] == treatment)
        _, shape, _, warmup, cycles = fixed
        treatment_ids = (treatment,)
        if horizon is None:
            raise ValueError(f"invalid finite-horizon RQ5 run {run_name!r}")
    if treatment != "inverse_sqrt" and timescale is not None:
        raise ValueError(f"unexpected inverse timescale in RQ5 run {run_name!r}")
    if (
        treatment not in {"cosine_warmup_tuned", "cosine_warmup5_shared"}
        and encoded_warmup is not None
    ):
        raise ValueError(f"unexpected tuned warmup in RQ5 run {run_name!r}")
    candidate = Rq5Candidate(
        treatments=treatment_ids,
        shape=shape,
        scope=scope,
        deep_lr=deep_lr,
        horizon_epochs=horizon,
        cap_epochs=int(match.group("cap")),
        warmup_fraction=warmup,
        timescale_fraction=timescale,
        cycles=cycles,
        attempt=int(match.group("attempt")),
        probe=match.group("probe"),
    )
    template = None
    if candidate.probe is None:
        template = next(
            (
                initial
                for initial in initial_candidates()
                if initial.treatments == candidate.treatments
                and initial.scope == candidate.scope
                and _slug(initial.deep_lr) == _slug(candidate.deep_lr)
                and (
                    initial.timescale_fraction is None
                    and candidate.timescale_fraction is None
                    or _slug(initial.timescale_fraction)
                    == _slug(candidate.timescale_fraction)
                )
                and (
                    "cosine_warmup_tuned" not in initial.treatments
                    or _slug(initial.warmup_fraction)
                    == _slug(candidate.warmup_fraction)
                )
            ),
            None,
        )
        if template is not None:
            candidate = replace(
                candidate,
                deep_lr=template.deep_lr,
                warmup_fraction=template.warmup_fraction,
                timescale_fraction=template.timescale_fraction,
            )
    if candidate.run_name != run_name:
        raise ValueError(f"non-canonical RQ5 run identity {run_name!r}")
    if candidate.attempt == 5:
        approved_surfaces, final_attempt_horizons = _load_a5_approval()
        matches = tuple(
            surface
            for surface in approved_surfaces
            if _surface_identity(surface) == _surface_identity(candidate)
        )
        if (
            len(matches) != 1
            or final_attempt_horizons.get(matches[0].run_name)
            != candidate.horizon_epochs
            or candidate.cap_epochs != candidate.horizon_epochs
        ):
            raise ValueError(f"unauthorized RQ5 a5 run {run_name!r}")
    return candidate


def _surface_identity(candidate: Rq5Candidate) -> tuple[object, ...]:
    variable_fields = {"horizon_epochs", "cap_epochs", "attempt"}
    return tuple(
        getattr(candidate, field.name)
        for field in fields(candidate)
        if field.name not in variable_fields
    )


def _load_a5_approval() -> tuple[tuple[Rq5Candidate, ...], dict[str, int]]:
    try:
        horizon_followup = json.loads(_APPROVAL_RECORD.read_text())["horizon_followup"]
        approved_names = horizon_followup["approved_initial_surfaces"]
        final_attempt_horizons = horizon_followup["final_attempt_horizons"]
        if (
            not isinstance(approved_names, list)
            or any(not isinstance(name, str) for name in approved_names)
            or approved_names != sorted(set(approved_names))
            or not isinstance(final_attempt_horizons, dict)
            or any(
                not isinstance(name, str)
                or not isinstance(horizon, int)
                or isinstance(horizon, bool)
                or horizon < 1
                for name, horizon in final_attempt_horizons.items()
            )
            or not set(final_attempt_horizons) <= set(approved_names)
            or any(
                not name.endswith("_a0_ts2_r1_500m") for name in approved_names
            )
        ):
            raise ValueError("invalid a5 approval fields")
        approved_surfaces = tuple(candidate_by_run(name) for name in approved_names)
        identities = tuple(_surface_identity(surface) for surface in approved_surfaces)
        if (
            any(surface.attempt != 0 for surface in approved_surfaces)
            or len(set(identities)) != len(identities)
        ):
            raise ValueError("invalid a5 approval surfaces")
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("RQ5 a5 approval record is invalid") from error
    return approved_surfaces, final_attempt_horizons


def _joint_pairs() -> tuple[tuple[float, float], ...]:
    generator = random.Random(42)
    random_pairs = tuple(
        (
            _log_uniform(generator, 0.0015, 0.024),
            _log_uniform(generator, 0.0125, 0.20),
        )
        for _ in range(2)
    )
    return ((0.006, 0.05), *random_pairs)


def _log_uniform(generator: random.Random, low: float, high: float) -> float:
    return math.exp(generator.uniform(math.log(low), math.log(high)))


def _slug(value: float | None) -> str:
    if value is None:
        raise ValueError("cannot encode an absent candidate value")
    return f"{value:.12g}".replace("-", "m").replace(".", "p")


def _unslug(value: str) -> float:
    return float(value.replace("m", "-").replace("p", "."))
