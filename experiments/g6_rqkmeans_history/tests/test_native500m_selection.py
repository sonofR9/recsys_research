import pytest

from experiments.g6_rqkmeans_history.native500m.protocol.selection import (
    Candidate,
    MetricValues,
    SeedEvidence,
    boundary_action,
    comparison_band,
    decide_rq1_initialization,
    decide_rq23,
    mean_seed_evidence,
    promote_against_two_baselines,
    resolve_terminal_bundle,
    select_by_quality,
)


RECALL_DISPERSION = 0.01685
NDCG_DISPERSION = 0.01966


def _candidate(
    identifier: str, recall: float, ndcg: float, order: int = 0
) -> Candidate:
    return Candidate(identifier, MetricValues(recall, ndcg), order)


def _seeds(
    recalls: tuple[float, ...],
    ndcgs: tuple[float, ...],
    *,
    first_epoch: float = 10,
    auc: float = 0.7,
    first_seed: int = 42,
) -> tuple[SeedEvidence, ...]:
    return tuple(
        SeedEvidence(
            seed=first_seed + offset,
            metrics=MetricValues(recall, ndcg),
            first_epoch_at_95_percent=first_epoch,
            normalized_recall_auc=auc,
        )
        for offset, (recall, ndcg) in enumerate(zip(recalls, ndcgs, strict=True))
    )


def test_bands_are_scaled_to_each_comparison_reference() -> None:
    assert comparison_band(0.1, RECALL_DISPERSION) == pytest.approx(0.001685)
    assert comparison_band(0.2, RECALL_DISPERSION) == pytest.approx(0.00337)


def test_quality_selection_uses_recall_then_ndcg_then_manifest_order() -> None:
    candidates = (
        _candidate("first", 0.1000, 0.040, 0),
        _candidate("ndcg", 0.1010, 0.050, 1),
        _candidate("outside", 0.1030, 0.030, 2),
    )
    assert (
        select_by_quality(
            candidates, recall_relative_dispersion=RECALL_DISPERSION
        ).identifier
        == "outside"
    )

    tied = (
        _candidate("first", 0.1000, 0.050, 0),
        _candidate("later", 0.1010, 0.050, 1),
    )
    assert (
        select_by_quality(tied, recall_relative_dispersion=RECALL_DISPERSION).identifier
        == "first"
    )


def test_rq0_promotion_requires_both_references_and_no_ndcg_regression() -> None:
    original = _candidate("original", 0.09, 0.04)
    best_g1 = _candidate("best_g1", 0.10, 0.05)
    promoted = promote_against_two_baselines(
        _candidate("sid", 0.103, 0.050),
        original=original,
        best_g1=best_g1,
        recall_relative_dispersion=RECALL_DISPERSION,
        ndcg_relative_dispersion=NDCG_DISPERSION,
    )
    assert promoted.promoted

    regressed = promote_against_two_baselines(
        _candidate("sid", 0.103, 0.048),
        original=original,
        best_g1=best_g1,
        recall_relative_dispersion=RECALL_DISPERSION,
        ndcg_relative_dispersion=NDCG_DISPERSION,
    )
    assert not regressed.promoted


def test_seed_means_require_the_exact_confirmation_seed_set() -> None:
    evidence = _seeds((0.1, 0.2, 0.3, 0.4), (0.04, 0.05, 0.06, 0.07))
    mean = mean_seed_evidence(evidence, expected_seeds=(42, 43, 44, 45))
    assert mean.metrics.recall_at_100 == pytest.approx(0.25)
    assert mean.metrics.ndcg_at_100 == pytest.approx(0.055)

    with pytest.raises(ValueError, match="seeds"):
        mean_seed_evidence(evidence[:-1], expected_seeds=(42, 43, 44, 45))


def test_rq1_quality_improvement_precedes_slower_convergence() -> None:
    random = _seeds((0.10,) * 4, (0.05,) * 4, first_epoch=8, auc=0.8)
    content = _seeds((0.103,) * 4, (0.05,) * 4, first_epoch=12, auc=0.7)

    decision = decide_rq1_initialization(
        random,
        content,
        recall_relative_dispersion=RECALL_DISPERSION,
        ndcg_relative_dispersion=NDCG_DISPERSION,
    )
    assert decision.selected == "content_pca"
    assert decision.reason == "quality"


@pytest.mark.parametrize(
    ("content_epoch", "content_auc", "selected"),
    ((7, 0.81, "content_pca"), (8, 0.81, "random"), (7, 0.79, "random")),
)
def test_rq1_within_band_requires_both_faster_indicators(
    content_epoch: float, content_auc: float, selected: str
) -> None:
    random = _seeds((0.10,) * 4, (0.05,) * 4, first_epoch=8, auc=0.8)
    content = _seeds(
        (0.101,) * 4,
        (0.05,) * 4,
        first_epoch=content_epoch,
        auc=content_auc,
    )
    decision = decide_rq1_initialization(
        random,
        content,
        recall_relative_dispersion=RECALL_DISPERSION,
        ndcg_relative_dispersion=NDCG_DISPERSION,
    )
    assert decision.selected == selected


def test_rq23_uses_three_seed_means_and_never_accepts_suffix_recall_regression() -> (
    None
):
    rq0 = _seeds((0.10,) * 3, (0.05,) * 3)
    suffix = _seeds((0.099999,) * 3, (0.06,) * 3)
    none = _seeds((0.103,) * 3, (0.05,) * 3)
    decision = decide_rq23(
        rq0,
        suffix,
        none,
        recall_relative_dispersion=RECALL_DISPERSION,
        ndcg_relative_dispersion=NDCG_DISPERSION,
    )

    assert decision.rq2_selected == "rq0"
    assert decision.terminal_selected == "none"
    assert decision.promoted


def test_failed_sid_promotion_keeps_no_sid_aggregate_without_duplicate_run() -> None:
    original = _candidate("original", 0.09, 0.04)
    best_g1 = _candidate("best_g1", 0.10, 0.05)
    diagnostic = _candidate("suffix", 0.101, 0.05)

    resolution = resolve_terminal_bundle(
        diagnostic,
        rq0_bridge_bundle_id="rq0_suffix",
        original=original,
        best_g1=best_g1,
        recall_relative_dispersion=RECALL_DISPERSION,
        ndcg_relative_dispersion=NDCG_DISPERSION,
    )

    assert resolution.diagnostic_bundle_id == "suffix"
    assert resolution.aggregate_bundle_id == "best_g1"
    assert not resolution.launch_aggregate_run
    assert resolution.requires_terminal_bridge


def test_boundary_actions_extend_once_then_require_approval_for_another_win() -> None:
    assert (
        boundary_action(at_outer_boundary=False, extension_round=0, boundary_won=False)
        == "resolved"
    )
    assert (
        boundary_action(at_outer_boundary=True, extension_round=0, boundary_won=False)
        == "extend"
    )
    assert (
        boundary_action(at_outer_boundary=True, extension_round=1, boundary_won=True)
        == "requires_approval"
    )
