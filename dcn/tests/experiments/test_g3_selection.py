import pytest

from experiments.g3_pretrained_item_embeddings.protocol.selection import (
    MetricEvidence,
    PromotionRule,
    TreatmentCandidate,
    decide_promotion,
    resolve_aggregate,
    resolve_dataset_scale,
)


def test_promotion_supports_aggregate_and_predeclared_tail_routes() -> None:
    control = MetricEvidence("control", recall_at_100=0.10, tail_recall_at_100=0.20)
    aggregate_winner = MetricEvidence(
        "aggregate_winner", recall_at_100=0.112, tail_recall_at_100=0.19
    )
    tail_winner = MetricEvidence(
        "tail_winner", recall_at_100=0.105, tail_recall_at_100=0.21
    )
    aggregate_only = PromotionRule(primary_comparators=("control",))
    tail_rule = PromotionRule(
        primary_comparators=("control",),
        tail_tradeoff_comparators=("control",),
        tail_comparators=("control",),
    )

    primary = decide_promotion(
        candidate=aggregate_winner,
        references={"control": control},
        rule=aggregate_only,
        relative_dispersion=0.10,
    )
    tail = decide_promotion(
        candidate=tail_winner,
        references={"control": control},
        rule=tail_rule,
        relative_dispersion=0.10,
    )

    assert primary.promoted is True
    assert primary.route == "aggregate"
    assert tail.promoted is True
    assert tail.route == "tail_tradeoff"
    assert dict(tail.absolute_bands) == {"control": pytest.approx(0.01)}


def test_promotion_is_strict_at_bands_and_rejects_tail_with_aggregate_harm() -> None:
    control = MetricEvidence("control", recall_at_100=0.10, tail_recall_at_100=0.20)
    rule = PromotionRule(
        primary_comparators=("control",),
        tail_tradeoff_comparators=("control",),
        tail_comparators=("control",),
    )

    boundary = decide_promotion(
        candidate=MetricEvidence("boundary", 0.11, 0.20),
        references={"control": control},
        rule=rule,
        relative_dispersion=0.10,
    )
    harmed = decide_promotion(
        candidate=MetricEvidence("harmed", 0.089, 0.30),
        references={"control": control},
        rule=rule,
        relative_dispersion=0.10,
    )

    assert boundary.promoted is False
    assert harmed.promoted is False
    assert harmed.route == "rejected"


def test_dependency_resolution_counts_atomic_untying_once_and_builds_bridges() -> None:
    candidates = (
        TreatmentCandidate("rq1_content_input", True, gain=0.025, latency_seconds=2.0),
        TreatmentCandidate("rq2_content_concat", True, gain=0.026, latency_seconds=3.0),
        TreatmentCandidate("rq5_frequency_gate", False, gain=0.0, latency_seconds=0.0),
        TreatmentCandidate("rq3_catalog_output", True, gain=0.02, latency_seconds=1.0),
        TreatmentCandidate("rq4_metadata", True, gain=0.02, latency_seconds=1.0),
    )

    resolution = resolve_aggregate(candidates, input_resolution_band=0.005)

    assert resolution.selected_input == "rq1_content_input"
    assert resolution.included_treatments == (
        "rq1_content_input",
        "rq3_catalog_output",
        "rq4_metadata",
    )
    assert resolution.components == (
        "untied_tables",
        "rq1_content_input",
        "rq3_catalog_output",
        "rq4_metadata",
    )
    assert [term.includes_untying for term in resolution.arithmetic_terms] == [
        True,
        False,
        False,
    ]
    assert resolution.required_search_families == (
        "bridge_rq3_output",
        "bridge_rq4_metadata",
    )
    assert dict(resolution.omissions)["rq2_content_concat"].startswith(
        "input conflict"
    )


def test_frequency_gate_supersedes_concat_as_one_atomic_input_bundle() -> None:
    candidates = (
        TreatmentCandidate("rq1_content_input", False, gain=0.0, latency_seconds=0.0),
        TreatmentCandidate("rq2_content_concat", True, gain=0.02, latency_seconds=2.0),
        TreatmentCandidate("rq5_frequency_gate", True, gain=0.03, latency_seconds=2.5),
        TreatmentCandidate("rq3_catalog_output", False, gain=0.0, latency_seconds=0.0),
        TreatmentCandidate("rq4_metadata", False, gain=0.0, latency_seconds=0.0),
    )

    resolution = resolve_aggregate(candidates, input_resolution_band=0.005)

    assert resolution.selected_input == "rq5_frequency_gate"
    assert resolution.components == (
        "untied_tables",
        "rq2_content_concat",
        "rq5_frequency_gate",
    )
    assert len(resolution.arithmetic_terms) == 1
    assert resolution.arithmetic_terms[0].added_components == (
        "untied_tables",
        "rq2_content_concat",
        "rq5_frequency_gate",
    )
    assert dict(resolution.omissions)["rq2_content_concat"] == (
        "superseded by promoted rq5_frequency_gate"
    )


def test_no_promotions_reuses_tied_baseline_without_duplicate_aggregate() -> None:
    candidates = tuple(
        TreatmentCandidate(treatment_id, False, gain=0.0, latency_seconds=0.0)
        for treatment_id in (
            "rq1_content_input",
            "rq2_content_concat",
            "rq5_frequency_gate",
            "rq3_catalog_output",
            "rq4_metadata",
        )
    )

    resolution = resolve_aggregate(candidates, input_resolution_band=0.005)

    assert resolution.selected_input == "tied_learned_item_id"
    assert resolution.components == ()
    assert resolution.arithmetic_terms == ()
    assert resolution.required_search_families == ()
    assert resolution.reuse_original_baseline is True


def test_dataset_scale_rule_is_two_sided_and_strict() -> None:
    smaller = resolve_dataset_scale(
        gain_50m=0.04,
        gain_500m=0.01,
        absolute_band_50m=0.01,
        absolute_band_500m=0.01,
    )
    larger = resolve_dataset_scale(
        gain_50m=0.01,
        gain_500m=0.04,
        absolute_band_50m=0.01,
        absolute_band_500m=0.01,
    )
    boundary = resolve_dataset_scale(
        gain_50m=0.01,
        gain_500m=0.03,
        absolute_band_50m=0.01,
        absolute_band_500m=0.01,
    )

    assert smaller.effect_detected is True
    assert smaller.direction == "smaller_at_500m"
    assert larger.direction == "larger_at_500m"
    assert boundary.effect_detected is False
    assert boundary.direction == "unresolved"
