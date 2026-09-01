from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from functools import cache
import math
import re
from typing import Literal


Study = Literal["rq13", "rq14"]
Stage = Literal["initial", "cap_anchor", "selected_cap", "lr_boundary"]
BoundaryDirection = Literal["low", "high"]
Treatment = str
_FixedTreatment = Literal[
    "one_example",
    "truncated_8",
    "truncated_16",
    "required_8",
    "required_16",
    "shared_cls_only",
    "distinct_cls_only",
    "shared_history",
    "distinct_history",
]

DEEP_LRS = (0.006, 0.012, 0.024)
IMPLEMENTATION_REVISION = 1
_RQ13_TREATMENTS: tuple[Treatment, ...] = (
    "one_example",
    "truncated_8",
    "truncated_16",
    "required_8",
    "required_16",
)
_RQ13_CAP_ANCHOR_TREATMENT = "truncated_4"
_RQ14_TREATMENTS: tuple[Treatment, ...] = (
    "shared_cls_only",
    "distinct_cls_only",
    "shared_history",
    "distinct_history",
)


@dataclass(frozen=True)
class QueryCandidate:
    study: Study
    treatment: Treatment
    deep_lr: float
    dataset_size: Literal["500m"] = "500m"
    embedding_lr: float = 0.064
    batch_size: int = 1280
    seed: int = 42
    horizon_epochs: int = 20
    implementation_revision: int = IMPLEMENTATION_REVISION
    stage: Stage = "initial"
    boundary_direction: BoundaryDirection | None = None
    boundary_step: int = 0

    def __post_init__(self) -> None:
        if self.study not in {"rq13", "rq14"}:
            raise ValueError(f"unknown study {self.study!r}")
        if self.study == "rq13":
            valid_treatment = self.treatment in {
                *_RQ13_TREATMENTS,
                _RQ13_CAP_ANCHOR_TREATMENT,
            } or re.fullmatch(r"selected_cap_[1-9][0-9]*", self.treatment)
        else:
            valid_treatment = self.treatment in _RQ14_TREATMENTS
        if not valid_treatment:
            raise ValueError(
                f"treatment {self.treatment!r} does not belong to {self.study}"
            )
        if not math.isfinite(self.deep_lr) or self.deep_lr <= 0:
            raise ValueError("deep LR must be finite and positive")
        if self.stage == "initial":
            if (
                self.treatment
                not in (_RQ13_TREATMENTS if self.study == "rq13" else _RQ14_TREATMENTS)
                or self.deep_lr not in DEEP_LRS
                or self.boundary_direction is not None
                or self.boundary_step != 0
            ):
                raise ValueError("initial candidates must use the approved LR grid")
        elif self.stage == "cap_anchor":
            if (
                self.study != "rq13"
                or self.treatment != _RQ13_CAP_ANCHOR_TREATMENT
                or self.deep_lr not in DEEP_LRS
                or self.boundary_direction is not None
                or self.boundary_step != 0
            ):
                raise ValueError("cap-anchor candidates must use the cap-4 LR grid")
        elif self.stage == "selected_cap":
            if (
                self.study != "rq13"
                or re.fullmatch(r"selected_cap_[1-9][0-9]*", self.treatment) is None
                or self.deep_lr not in DEEP_LRS
                or self.boundary_direction is not None
                or self.boundary_step != 0
            ):
                raise ValueError(
                    "selected-cap candidates must use the approved LR grid"
                )
        elif self.stage == "lr_boundary":
            if self.boundary_direction not in {"low", "high"} or self.boundary_step < 1:
                raise ValueError(
                    "LR-boundary candidates need a direction and positive step"
                )
            anchor = (
                min(DEEP_LRS) if self.boundary_direction == "low" else max(DEEP_LRS)
            )
            factor = 0.5 if self.boundary_direction == "low" else 2.0
            if self.deep_lr != anchor * factor**self.boundary_step:
                raise ValueError("LR-boundary rate does not match its geometric step")
        else:
            raise ValueError(f"unknown candidate stage {self.stage!r}")
        if (
            self.dataset_size != "500m"
            or self.embedding_lr != 0.064
            or self.batch_size != 1280
            or self.seed != 42
            or self.horizon_epochs != 20
            or self.implementation_revision != IMPLEMENTATION_REVISION
        ):
            raise ValueError("RQ13/RQ14 use fixed native-500M training invariants")

    @property
    def run_name(self) -> str:
        stage = (
            ""
            if self.stage in {"initial", "cap_anchor", "selected_cap"}
            else f"_lr_{self.boundary_direction}{self.boundary_step}"
        )
        return (
            f"g1_{self.study}_{self.treatment}{stage}_d{_slug(self.deep_lr)}_"
            f"seed{self.seed}_h{self.horizon_epochs}_"
            f"r{self.implementation_revision}_{self.dataset_size}"
        )

    def environment(self) -> dict[str, str]:
        return {"G1_QUERY_RUN": self.run_name}


@cache
def initial_candidates() -> tuple[QueryCandidate, ...]:
    result = tuple(
        QueryCandidate(study, treatment, deep_lr)
        for study, treatments in (
            ("rq13", _RQ13_TREATMENTS),
            ("rq14", _RQ14_TREATMENTS),
        )
        for treatment in treatments
        for deep_lr in DEEP_LRS
    )
    if len(result) != 27 or len({candidate.run_name for candidate in result}) != 27:
        raise RuntimeError("RQ13/RQ14 initial manifest must contain 27 unique runs")
    return result


@cache
def rq13_initial_candidates() -> tuple[QueryCandidate, ...]:
    result = tuple(
        candidate for candidate in initial_candidates() if candidate.study == "rq13"
    )
    if len(result) != 15 or len({candidate.run_name for candidate in result}) != 15:
        raise RuntimeError("RQ13 initial manifest must contain 15 unique runs")
    return result


@cache
def rq14_initial_candidates() -> tuple[QueryCandidate, ...]:
    result = tuple(
        candidate for candidate in initial_candidates() if candidate.study == "rq14"
    )
    if len(result) != 12 or len({candidate.run_name for candidate in result}) != 12:
        raise RuntimeError("RQ14 initial manifest must contain 12 unique runs")
    return result


@cache
def rq13_cap4_candidates() -> tuple[QueryCandidate, ...]:
    result = tuple(
        QueryCandidate("rq13", _RQ13_CAP_ANCHOR_TREATMENT, deep_lr, stage="cap_anchor")
        for deep_lr in DEEP_LRS
    )
    if len(result) != 3 or len({candidate.run_name for candidate in result}) != 3:
        raise RuntimeError("RQ13 cap-4 manifest must contain three unique runs")
    return result


def make_selected_cap_candidates(cap: int) -> tuple[QueryCandidate, ...]:
    if isinstance(cap, bool) or not isinstance(cap, int) or not 17 <= cap <= 32:
        raise ValueError("the fitted practical cap must be an integer from 17 to 32")
    result = tuple(
        QueryCandidate("rq13", f"selected_cap_{cap}", deep_lr, stage="selected_cap")
        for deep_lr in DEEP_LRS
    )
    if len({candidate.run_name for candidate in result}) != 3:
        raise RuntimeError("RQ13 selected-cap manifest is not unique")
    return result


def candidate_by_run(run_name: str) -> QueryCandidate:
    matches = [
        candidate
        for candidate in initial_candidates()
        if candidate.run_name == run_name
    ]
    if len(matches) == 1:
        return matches[0]
    matches = [
        candidate
        for candidate in rq13_cap4_candidates()
        if candidate.run_name == run_name
    ]
    if len(matches) == 1:
        return matches[0]
    selected_match = re.fullmatch(
        r"g1_rq13_selected_cap_(?P<cap>[1-9][0-9]*)"
        r"_d(?P<lr>[0-9a-z]+)_seed(?P<seed>[0-9]+)_h(?P<horizon>[0-9]+)"
        r"_r(?P<revision>[0-9]+)_(?P<dataset>[a-z0-9]+)",
        run_name,
    )
    if selected_match is not None:
        cap = int(selected_match.group("cap"))
        candidates = make_selected_cap_candidates(cap)
        matches = [
            candidate for candidate in candidates if candidate.run_name == run_name
        ]
        if len(matches) == 1:
            return matches[0]
    match = re.fullmatch(
        r"g1_(?P<study>rq13|rq14)_(?P<treatment>[a-z0-9_]+)"
        r"_lr_(?P<direction>low|high)(?P<step>[1-9][0-9]*)"
        r"_d[0-9a-z]+_seed(?P<seed>[0-9]+)_h(?P<horizon>[0-9]+)"
        r"_r(?P<revision>[0-9]+)_(?P<dataset>[a-z0-9]+)",
        run_name,
    )
    if match is None:
        raise ValueError(f"unknown RQ13/RQ14 run {run_name!r}")
    direction = match.group("direction")
    step = int(match.group("step"))
    anchor = min(DEEP_LRS) if direction == "low" else max(DEEP_LRS)
    factor = 0.5 if direction == "low" else 2.0
    candidate = QueryCandidate(
        study=match.group("study"),
        treatment=match.group("treatment"),
        deep_lr=anchor * factor**step,
        dataset_size=match.group("dataset"),
        seed=int(match.group("seed")),
        horizon_epochs=int(match.group("horizon")),
        implementation_revision=int(match.group("revision")),
        stage="lr_boundary",
        boundary_direction=direction,
        boundary_step=step,
    )
    if candidate.run_name != run_name:
        raise ValueError(f"non-canonical RQ13/RQ14 run {run_name!r}")
    return candidate


def make_boundary_candidate(
    candidate: QueryCandidate,
    direction: BoundaryDirection,
    step: int,
) -> QueryCandidate:
    anchor = min(DEEP_LRS) if direction == "low" else max(DEEP_LRS)
    factor = 0.5 if direction == "low" else 2.0
    return replace(
        candidate,
        deep_lr=anchor * factor**step,
        stage="lr_boundary",
        boundary_direction=direction,
        boundary_step=step,
    )


def validated_required_boundary_candidates(
    evidence: Mapping[str, object], requested_names: Sequence[str]
) -> tuple[QueryCandidate, ...]:
    if (
        evidence.get("research_question") != "RQ13 encoder-decoder prefix expansion"
        or evidence.get("dataset_size") != "500m"
    ):
        raise ValueError("RQ13 boundary evidence identifies the wrong study")
    missing = evidence.get("missing_initial_artifacts")
    required = evidence.get("required_followups")
    boundary = evidence.get("required_boundary_followups")
    if not isinstance(missing, list) or not all(
        isinstance(name, str) for name in missing
    ):
        raise ValueError("RQ13 evidence has invalid missing-initial metadata")
    if missing:
        raise ValueError("RQ13 initial grid is incomplete")
    if (
        not isinstance(boundary, list)
        or not boundary
        or not all(isinstance(name, str) for name in boundary)
    ):
        raise ValueError("RQ13 has no required boundary followups")
    if required != boundary:
        raise ValueError("RQ13 evidence has unresolved non-boundary followups")
    if len(set(boundary)) != len(boundary):
        raise ValueError("RQ13 evidence repeats a boundary followup")
    if len(set(requested_names)) != len(requested_names):
        raise ValueError("requested RQ13 boundary followups contain duplicates")
    if set(requested_names) != set(boundary):
        raise ValueError("requested runs differ from current RQ13 boundary followups")
    candidates = tuple(candidate_by_run(name) for name in boundary)
    if any(
        candidate.study != "rq13" or candidate.stage != "lr_boundary"
        for candidate in candidates
    ):
        raise ValueError("RQ13 evidence contains a non-boundary candidate")
    return candidates


def validated_rq14_boundary_candidates(
    evidence: Mapping[str, object], requested_names: Sequence[str]
) -> tuple[QueryCandidate, ...]:
    if (
        evidence.get("research_question") != "RQ14 decoder-decoder query memory"
        or evidence.get("dataset_size") != "500m"
    ):
        raise ValueError("RQ14 boundary evidence identifies the wrong study")
    missing = evidence.get("missing_initial_artifacts")
    required = evidence.get("required_followups")
    boundary = evidence.get("required_boundary_followups")
    if not isinstance(missing, list) or not all(
        isinstance(name, str) for name in missing
    ):
        raise ValueError("RQ14 evidence has invalid missing-initial metadata")
    if missing:
        raise ValueError("RQ14 initial grid is incomplete")
    if (
        not isinstance(boundary, list)
        or not boundary
        or not all(isinstance(name, str) for name in boundary)
        or required != boundary
    ):
        raise ValueError("RQ14 evidence has no exact boundary-only followup stage")
    if len(set(boundary)) != len(boundary):
        raise ValueError("RQ14 evidence repeats a boundary followup")
    if len(set(requested_names)) != len(requested_names) or set(requested_names) != set(
        boundary
    ):
        raise ValueError("requested runs differ from the exact RQ14 boundary stage")
    candidates = tuple(candidate_by_run(name) for name in boundary)
    if any(
        candidate.study != "rq14" or candidate.stage != "lr_boundary"
        for candidate in candidates
    ):
        raise ValueError("RQ14 evidence contains a non-boundary candidate")
    return candidates


def validated_cap4_candidates(
    evidence: Mapping[str, object], requested_names: Sequence[str]
) -> tuple[QueryCandidate, ...]:
    candidates = rq13_cap4_candidates()
    expected = [candidate.run_name for candidate in candidates]
    if (
        evidence.get("research_question") != "RQ13 encoder-decoder prefix expansion"
        or evidence.get("dataset_size") != "500m"
        or evidence.get("missing_initial_artifacts") != expected
        or evidence.get("required_followups") != expected
        or evidence.get("required_boundary_followups") != []
        or evidence.get("cap_fit") != {"status": "pending_cap4"}
    ):
        raise ValueError("RQ13 original surface is not resolved for cap-4 staging")
    audit = evidence.get("correctness_audit")
    if not isinstance(audit, Mapping) or audit.get("status") != "passed":
        raise ValueError("RQ13 original source-exact correctness audit is not current")
    winners = evidence.get("surface_winners")
    if not isinstance(winners, Mapping) or set(winners) != set(_RQ13_TREATMENTS):
        raise ValueError("RQ13 original LR winners are incomplete")
    if (
        len(set(requested_names)) != len(requested_names)
        or list(requested_names) != expected
    ):
        raise ValueError("requested runs differ from the exact cap-4 stage")
    return candidates


def validated_selected_cap_candidates(
    evidence: Mapping[str, object], requested_names: Sequence[str]
) -> tuple[QueryCandidate, ...]:
    if (
        evidence.get("research_question") != "RQ13 encoder-decoder prefix expansion"
        or evidence.get("dataset_size") != "500m"
    ):
        raise ValueError("RQ13 selected-cap evidence identifies the wrong study")
    cap_fit = evidence.get("cap_fit")
    if not isinstance(cap_fit, Mapping):
        raise ValueError("RQ13 selected-cap fit evidence is absent")
    if cap_fit.get("status") != "selected_cap_pending":
        raise ValueError("RQ13 cap fit is not ready for selected-cap training")
    metric = cap_fit.get("metric")
    if not isinstance(metric, str) or "validation Recall@100" not in metric:
        raise ValueError("RQ13 cap selection must use validation Recall@100 only")
    target = cap_fit.get("selection_target")
    reader_target = cap_fit.get("reader_success_target")
    if (
        not isinstance(target, Mapping)
        or target.get("metric") != "mean validation Recall@100"
        or target.get("multiplier") != 1.10
        or target.get("control_values") != [0.1367, 0.1343, 0.1363]
        or target.get("control_mean") != sum((0.1367, 0.1343, 0.1363)) / 3
        or target.get("value") != 1.10 * (sum((0.1367, 0.1343, 0.1363)) / 3)
        or not isinstance(reader_target, Mapping)
        or reader_target.get("metric") != "mean full-user Recall@100"
        or reader_target.get("multiplier") != 1.10
        or reader_target.get("control_mean") != 0.13468336146286186
        or reader_target.get("value") != 1.10 * 0.13468336146286186
    ):
        raise ValueError(
            "RQ13 validation-selection and reader targets are not distinct"
        )
    bindings = cap_fit.get("input_bindings")
    contributors = (
        bindings.get("contributing_artifacts")
        if isinstance(bindings, Mapping)
        else None
    )
    stage_one_audit = (
        bindings.get("stage_one_correctness_audit")
        if isinstance(bindings, Mapping)
        else None
    )
    if (
        not isinstance(contributors, Mapping)
        or set(contributors)
        != {"one_example", "truncated_4", "truncated_8", "truncated_16"}
        or not isinstance(bindings.get("eligible_target_counts_sha256"), str)
        or re.fullmatch(
            r"[0-9a-f]{64}", str(bindings.get("eligible_target_counts_sha256"))
        )
        is None
        or not isinstance(stage_one_audit, Mapping)
        or stage_one_audit.get("status") != "passed"
        or stage_one_audit.get("schema_version") != 1
        or re.fullmatch(r"[0-9a-f]{64}", str(stage_one_audit.get("artifact_sha256")))
        is None
        or any(
            not isinstance(record, Mapping)
            or not isinstance(record.get("artifact_sha256"), Mapping)
            or set(record["artifact_sha256"])
            != {"training_metadata.json", "final_metrics.json", "sweep.log"}
            or not isinstance(record.get("source_manifest_sha256"), str)
            for record in contributors.values()
        )
    ):
        raise ValueError("RQ13 selected-cap fit is not bound to exact source evidence")
    cap = cap_fit.get("selected_cap")
    ceiling = cap_fit.get("practical_ceiling")
    practical = ceiling.get("selected") if isinstance(ceiling, Mapping) else None
    target_cap = cap_fit.get("target_cap")
    if (
        isinstance(cap, bool)
        or not isinstance(cap, int)
        or isinstance(practical, bool)
        or not isinstance(practical, int)
        or not 17 <= cap <= practical <= 32
        or (
            target_cap is not None
            and (isinstance(target_cap, bool) or not isinstance(target_cap, int))
        )
        or cap != min(practical, max(17, target_cap or practical))
    ):
        raise ValueError("RQ13 selected cap contradicts its practical fit rule")
    candidates = make_selected_cap_candidates(cap)  # type: ignore[arg-type]
    expected = [candidate.run_name for candidate in candidates]
    required = evidence.get("required_followups")
    if required != expected:
        raise ValueError("RQ13 selected-cap runs are not the exact current followups")
    if len(set(requested_names)) != len(requested_names) or set(requested_names) != set(
        expected
    ):
        raise ValueError(
            "requested runs differ from current RQ13 selected-cap followups"
        )
    return candidates


def _slug(value: float) -> str:
    return re.sub(r"[^0-9a-z]+", "", f"{value:g}".lower())
