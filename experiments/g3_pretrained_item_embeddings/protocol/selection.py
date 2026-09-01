from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Mapping, Sequence


@dataclass(frozen=True)
class MetricEvidence:
    id: str
    recall_at_100: float
    tail_recall_at_100: float | None = None


@dataclass(frozen=True)
class PromotionRule:
    primary_comparators: tuple[str, ...]
    tail_tradeoff_comparators: tuple[str, ...] = ()
    tail_comparators: tuple[str, ...] = ()


@dataclass(frozen=True)
class PromotionDecision:
    treatment_id: str
    promoted: bool
    route: Literal["aggregate", "tail_tradeoff", "rejected"]
    absolute_bands: tuple[tuple[str, float], ...]
    reason: str


@dataclass(frozen=True)
class TreatmentCandidate:
    id: str
    promoted: bool
    gain: float
    latency_seconds: float


@dataclass(frozen=True)
class ArithmeticTerm:
    predecessor_components: tuple[str, ...]
    result_components: tuple[str, ...]
    added_components: tuple[str, ...]
    includes_untying: bool


@dataclass(frozen=True)
class AggregateResolution:
    selected_input: str
    included_treatments: tuple[str, ...]
    components: tuple[str, ...]
    arithmetic_terms: tuple[ArithmeticTerm, ...]
    required_search_families: tuple[str, ...]
    omissions: tuple[tuple[str, str], ...]
    reuse_original_baseline: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "selected_input": self.selected_input,
            "included_treatments": list(self.included_treatments),
            "components": list(self.components),
            "arithmetic_terms": [
                {
                    "predecessor_components": list(term.predecessor_components),
                    "result_components": list(term.result_components),
                    "added_components": list(term.added_components),
                    "includes_untying": term.includes_untying,
                }
                for term in self.arithmetic_terms
            ],
            "required_search_families": list(self.required_search_families),
            "omissions": dict(self.omissions),
            "reuse_original_baseline": self.reuse_original_baseline,
        }


@dataclass(frozen=True)
class DatasetScaleResolution:
    gain_50m: float
    gain_500m: float
    gain_difference: float
    combined_band: float
    effect_detected: bool
    direction: Literal["smaller_at_500m", "larger_at_500m", "unresolved"]


_TREATMENT_ORDER = (
    "rq1_content_input",
    "rq2_content_concat",
    "rq5_frequency_gate",
    "rq3_catalog_output",
    "rq4_metadata",
)

_KNOWN_TRAINED_COMBINATIONS = {
    frozenset(("untied_tables", "rq1_content_input")),
    frozenset(("untied_tables", "rq2_content_concat")),
    frozenset(
        ("untied_tables", "rq2_content_concat", "rq5_frequency_gate")
    ),
    frozenset(
        ("untied_tables", "rq2_content_concat", "rq3_catalog_output")
    ),
    frozenset(
        (
            "untied_tables",
            "rq2_content_concat",
            "rq3_catalog_output",
            "rq4_metadata",
        )
    ),
}


def decide_promotion(
    *,
    candidate: MetricEvidence,
    references: Mapping[str, MetricEvidence],
    rule: PromotionRule,
    relative_dispersion: float,
) -> PromotionDecision:
    _validate_evidence(candidate)
    if not math.isfinite(relative_dispersion) or relative_dispersion < 0:
        raise ValueError("relative dispersion must be finite and nonnegative")
    comparator_ids = tuple(
        dict.fromkeys(
            (
                *rule.primary_comparators,
                *rule.tail_tradeoff_comparators,
                *rule.tail_comparators,
            )
        )
    )
    if not rule.primary_comparators:
        raise ValueError("promotion rule requires a primary comparator")
    missing = set(comparator_ids) - set(references)
    if missing:
        raise ValueError(f"promotion evidence is missing comparators {sorted(missing)}")
    for comparator_id in comparator_ids:
        _validate_evidence(references[comparator_id])
    bands = tuple(
        (
            comparator_id,
            abs(references[comparator_id].recall_at_100) * relative_dispersion,
        )
        for comparator_id in comparator_ids
    )
    band_by_id = dict(bands)

    primary_passes = all(
        candidate.recall_at_100
        > references[comparator_id].recall_at_100 + band_by_id[comparator_id]
        for comparator_id in rule.primary_comparators
    )
    if primary_passes:
        return PromotionDecision(
            candidate.id,
            True,
            "aggregate",
            bands,
            "Recall@100 improves beyond every applicable band",
        )

    aggregate_is_within_tail_bands = bool(rule.tail_tradeoff_comparators) and all(
        abs(candidate.recall_at_100 - references[comparator_id].recall_at_100)
        <= band_by_id[comparator_id]
        for comparator_id in rule.tail_tradeoff_comparators
    )
    tail_improves = (
        candidate.tail_recall_at_100 is not None
        and bool(rule.tail_comparators)
        and all(
            references[comparator_id].tail_recall_at_100 is not None
            and candidate.tail_recall_at_100
            > references[comparator_id].tail_recall_at_100
            for comparator_id in rule.tail_comparators
        )
    )
    if aggregate_is_within_tail_bands and tail_improves:
        return PromotionDecision(
            candidate.id,
            True,
            "tail_tradeoff",
            bands,
            "aggregate Recall@100 is within band and tail Recall@100 is higher",
        )
    return PromotionDecision(
        candidate.id,
        False,
        "rejected",
        bands,
        "neither the aggregate nor predeclared tail route qualifies",
    )


def resolve_aggregate(
    candidates: Sequence[TreatmentCandidate],
    *,
    input_resolution_band: float,
) -> AggregateResolution:
    if not math.isfinite(input_resolution_band) or input_resolution_band < 0:
        raise ValueError("input resolution band must be finite and nonnegative")
    by_id = {candidate.id: candidate for candidate in candidates}
    if len(by_id) != len(candidates):
        raise ValueError("aggregate candidates contain duplicate treatment IDs")
    if set(by_id) != set(_TREATMENT_ORDER):
        raise ValueError("aggregate candidates do not match the approved G3 treatments")
    for candidate in candidates:
        if not math.isfinite(candidate.gain) or not math.isfinite(
            candidate.latency_seconds
        ):
            raise ValueError("aggregate candidate values must be finite")
        if candidate.latency_seconds < 0:
            raise ValueError("aggregate candidate latency must be nonnegative")

    omissions: dict[str, str] = {
        candidate.id: "not promoted"
        for candidate in candidates
        if not candidate.promoted
    }
    promoted_inputs: list[TreatmentCandidate] = []
    if by_id["rq1_content_input"].promoted:
        promoted_inputs.append(by_id["rq1_content_input"])
    if by_id["rq5_frequency_gate"].promoted:
        promoted_inputs.append(by_id["rq5_frequency_gate"])
        if by_id["rq2_content_concat"].promoted:
            omissions["rq2_content_concat"] = (
                "superseded by promoted rq5_frequency_gate"
            )
    elif by_id["rq2_content_concat"].promoted:
        promoted_inputs.append(by_id["rq2_content_concat"])

    selected_input_candidate = _select_input(
        promoted_inputs,
        resolution_band=input_resolution_band,
    )
    selected_input = (
        selected_input_candidate.id
        if selected_input_candidate is not None
        else "tied_learned_item_id"
    )
    if (
        selected_input == "rq5_frequency_gate"
        and not by_id["rq2_content_concat"].promoted
    ):
        omissions["rq2_content_concat"] = (
            "included only as an atomic prerequisite of rq5_frequency_gate"
        )
    for candidate in promoted_inputs:
        if candidate is not selected_input_candidate:
            omissions[candidate.id] = (
                f"input conflict resolved in favor of {selected_input}"
            )

    included: list[str] = []
    if selected_input_candidate is not None:
        included.append(selected_input_candidate.id)
    for treatment_id in ("rq3_catalog_output", "rq4_metadata"):
        if by_id[treatment_id].promoted:
            included.append(treatment_id)

    if not included:
        return AggregateResolution(
            selected_input="tied_learned_item_id",
            included_treatments=(),
            components=(),
            arithmetic_terms=(),
            required_search_families=(),
            omissions=tuple((key, omissions[key]) for key in _TREATMENT_ORDER),
            reuse_original_baseline=True,
        )

    components, terms = _arithmetic(selected_input, tuple(included))
    required = _required_search_families(selected_input, tuple(included), components)
    for treatment_id in _TREATMENT_ORDER:
        if treatment_id not in included and treatment_id not in omissions:
            omissions[treatment_id] = "not selected for the compatible aggregate"
    return AggregateResolution(
        selected_input=(
            selected_input
            if selected_input_candidate is not None
            else "untied_learned_item_id"
        ),
        included_treatments=tuple(included),
        components=components,
        arithmetic_terms=terms,
        required_search_families=required,
        omissions=tuple(
            (treatment_id, omissions[treatment_id])
            for treatment_id in _TREATMENT_ORDER
            if treatment_id in omissions
        ),
        reuse_original_baseline=False,
    )


def resolve_dataset_scale(
    *,
    gain_50m: float,
    gain_500m: float,
    absolute_band_50m: float,
    absolute_band_500m: float,
) -> DatasetScaleResolution:
    values = (gain_50m, gain_500m, absolute_band_50m, absolute_band_500m)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("dataset-scale inputs must be finite")
    if absolute_band_50m < 0 or absolute_band_500m < 0:
        raise ValueError("dataset-scale bands must be nonnegative")
    difference = gain_500m - gain_50m
    combined_band = absolute_band_50m + absolute_band_500m
    detected = abs(difference) > combined_band
    if not detected:
        direction: Literal[
            "smaller_at_500m", "larger_at_500m", "unresolved"
        ] = "unresolved"
    elif difference > 0:
        direction = "larger_at_500m"
    else:
        direction = "smaller_at_500m"
    return DatasetScaleResolution(
        gain_50m,
        gain_500m,
        difference,
        combined_band,
        detected,
        direction,
    )


def _validate_evidence(evidence: MetricEvidence) -> None:
    values = [evidence.recall_at_100]
    if evidence.tail_recall_at_100 is not None:
        values.append(evidence.tail_recall_at_100)
    if not evidence.id or not all(math.isfinite(value) for value in values):
        raise ValueError("metric evidence must have an ID and finite values")
    if any(not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("ranking metrics must be between zero and one")


def _select_input(
    candidates: Sequence[TreatmentCandidate],
    *,
    resolution_band: float,
) -> TreatmentCandidate | None:
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    largest_gain = max(candidate.gain for candidate in candidates)
    smallest_gain = min(candidate.gain for candidate in candidates)
    if largest_gain - smallest_gain > resolution_band:
        return max(candidates, key=lambda candidate: candidate.gain)
    order = {treatment_id: index for index, treatment_id in enumerate(_TREATMENT_ORDER)}
    return min(candidates, key=lambda candidate: (candidate.latency_seconds, order[candidate.id]))


def _arithmetic(
    selected_input: str,
    included: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[ArithmeticTerm, ...]]:
    input_components: tuple[str, ...]
    if selected_input == "rq1_content_input":
        input_components = ("rq1_content_input",)
    elif selected_input == "rq2_content_concat":
        input_components = ("rq2_content_concat",)
    elif selected_input == "rq5_frequency_gate":
        input_components = ("rq2_content_concat", "rq5_frequency_gate")
    else:
        input_components = ()

    additions: list[tuple[str, ...]] = []
    if input_components:
        additions.append(("untied_tables", *input_components))
    if "rq3_catalog_output" in included:
        if additions:
            additions.append(("rq3_catalog_output",))
        else:
            additions.append(("untied_tables", "rq3_catalog_output"))
    if "rq4_metadata" in included:
        if additions:
            additions.append(("rq4_metadata",))
        else:
            additions.append(("untied_tables", "rq4_metadata"))

    current: tuple[str, ...] = ()
    terms = []
    for added in additions:
        result = (*current, *added)
        terms.append(
            ArithmeticTerm(
                predecessor_components=current,
                result_components=result,
                added_components=added,
                includes_untying="untied_tables" in added,
            )
        )
        current = result
    return current, tuple(terms)


def _required_search_families(
    selected_input: str,
    included: tuple[str, ...],
    components: tuple[str, ...],
) -> tuple[str, ...]:
    desired = frozenset(components)
    if desired in _KNOWN_TRAINED_COMBINATIONS:
        return ()
    required: list[str] = []
    if "rq3_catalog_output" in included and selected_input != "rq2_content_concat":
        required.append("bridge_rq3_output")
    if "rq4_metadata" in included:
        required.append("bridge_rq4_metadata")
    if not required:
        required.append("aggregate")
    return tuple(required)
