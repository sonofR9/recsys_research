import math

import pytest

from experiments.g6_rqkmeans_history.native500m.protocol.design import (
    BATCH_SIZE,
    BEST_G1_ANCHOR,
    CODEBOOK_SYMBOL_CAP,
    EXPECTED_RUN_TOTALS,
    FIXED_HORIZON,
    ORIGINAL_G1_ANCHOR,
    REPRESENTATION_WIDTH,
    SHARED_CODEBOOK_SIZES,
    TOKENIZER_LEVELS,
    BoundaryStatus,
    LearningRateCoordinate,
    TokenizerCoordinate,
    boundary_coordinates,
    bridge_surface,
    control_surface,
    expected_rq23_runs,
    inherited_rq0_surface,
    rq0_first_surface,
    rq1_paired_surface,
    rq23_paired_surface,
    run_budget,
    sobol_learning_rates,
    tokenizer_coordinates,
)


def test_fixed_domains_and_control_anchors_match_the_approval() -> None:
    assert TOKENIZER_LEVELS == (3, 4)
    assert SHARED_CODEBOOK_SIZES == (512, 2048, 8192)
    assert REPRESENTATION_WIDTH == 128
    assert CODEBOOK_SYMBOL_CAP == 8192
    assert FIXED_HORIZON == 26
    assert BATCH_SIZE == 512
    assert BEST_G1_ANCHOR == LearningRateCoordinate(
        0.0468526465053628, 0.032703745675187676
    )
    assert ORIGINAL_G1_ANCHOR == LearningRateCoordinate(0.001, 0.002)


def test_sobol_learning_rates_use_the_approved_seed_and_index_zero() -> None:
    coordinates = sobol_learning_rates(2)

    assert coordinates[0].embedding == pytest.approx(0.5067323497)
    assert coordinates[0].deep == pytest.approx(0.0030869847)
    assert coordinates[1].embedding == pytest.approx(0.0127004185)
    assert coordinates[1].deep == pytest.approx(0.0529977694)


def test_first_rq0_surface_covers_every_tokenizer_pair_twice() -> None:
    surface = rq0_first_surface()

    assert len(surface) == 12
    expected = {
        TokenizerCoordinate(levels, codes)
        for levels in TOKENIZER_LEVELS
        for codes in SHARED_CODEBOOK_SIZES
    }
    assert {row.tokenizer for row in surface} == expected
    assert all(
        sum(row.tokenizer == tokenizer for row in surface) == 2
        for tokenizer in expected
    )


def test_remaining_rq0_surfaces_inherit_one_frozen_anchor() -> None:
    tokenizer = TokenizerCoordinate(4, 2048)
    learning_rates = LearningRateCoordinate(0.04, 0.02)
    surface = inherited_rq0_surface(tokenizer, learning_rates)

    assert len(surface) == 8
    assert surface[0].tokenizer == tokenizer
    assert surface[0].learning_rates == learning_rates
    assert [row.tokenizer for row in surface[1:6]] == [
        candidate
        for candidate in (
            TokenizerCoordinate(levels, codes)
            for levels in TOKENIZER_LEVELS
            for codes in SHARED_CODEBOOK_SIZES
        )
        if candidate != tokenizer
    ]
    assert all(row.learning_rates == learning_rates for row in surface[:6])
    assert all(row.tokenizer == tokenizer for row in surface[6:])


def test_rq1_and_rq23_designs_are_paired_and_sequential() -> None:
    tokenizer = TokenizerCoordinate(3, 8192)
    rates = LearningRateCoordinate(0.03, 0.01)

    rq1 = rq1_paired_surface(tokenizer, rates)
    assert len(rq1) == 12
    assert [row.initialization for row in rq1[::2]] == ["random"] * 6
    assert [row.initialization for row in rq1[1::2]] == ["content_pca"] * 6
    assert all(
        left.coordinate == right.coordinate
        for left, right in zip(rq1[::2], rq1[1::2], strict=True)
    )
    assert rq1[0].coordinate.learning_rates == rates

    rq23 = rq23_paired_surface(
        tokenizer,
        rates,
        suffix_winner=TokenizerCoordinate(4, 512),
        no_suffix_winner=TokenizerCoordinate(3, 2048),
    )
    assert len(rq23) == 20
    assert all(
        left.policy == "suffix" and right.policy == "none"
        for left, right in zip(rq23[::2], rq23[1::2], strict=True)
    )


def test_exact_run_arithmetic_and_maximum() -> None:
    assert expected_rq23_runs(False, False) == 25
    assert expected_rq23_runs(True, False) == 23
    assert expected_rq23_runs(False, True) == 23
    assert expected_rq23_runs(True, True) == 21
    assert EXPECTED_RUN_TOTALS == (130, 132, 134, 138, 140, 142)

    budget = run_budget()
    assert budget.rq0_expected == 92
    assert budget.rq1_expected == 17
    assert budget.rq23_expected == (21, 23, 25)
    assert budget.terminal_bridge_expected == (0, 8)
    assert budget.maximum == 262


def test_boundary_extension_uses_four_outward_points_and_stops_after_one_round() -> (
    None
):
    source = LearningRateCoordinate(0.008, 0.02)
    extension = boundary_coordinates(source, round_number=0)

    assert extension.status is BoundaryStatus.EXTEND
    assert len(extension.coordinates) == 4
    assert [row.embedding for row in extension.coordinates] == pytest.approx(
        [0.008 / math.sqrt(2), 0.004, 0.008 / (2 * math.sqrt(2)), 0.002]
    )
    assert all(row.deep == source.deep for row in extension.coordinates)

    stopped = boundary_coordinates(source, round_number=1, boundary_won=True)
    assert stopped.status is BoundaryStatus.REQUIRES_APPROVAL
    assert stopped.coordinates == ()


def test_explicit_original_control_anchor_triggers_both_lower_boundaries() -> None:
    extension = boundary_coordinates(ORIGINAL_G1_ANCHOR, round_number=0)

    assert extension.status is BoundaryStatus.EXTEND
    assert len(extension.coordinates) == 8
    assert all(
        row.embedding < ORIGINAL_G1_ANCHOR.embedding
        for row in extension.coordinates[:4]
    )
    assert all(row.deep < ORIGINAL_G1_ANCHOR.deep for row in extension.coordinates[4:])


def test_every_possible_predecessor_produces_full_duplicate_free_surfaces() -> None:
    possible_rates = tuple(
        dict.fromkeys(
            (
                BEST_G1_ANCHOR,
                ORIGINAL_G1_ANCHOR,
                *sobol_learning_rates(24),
            )
        )
    )
    tokenizers = tokenizer_coordinates()

    for rates in possible_rates:
        controls = control_surface(rates)
        bridge = bridge_surface(rates)
        assert len(controls) == len(set(controls)) == 12
        assert len(bridge) == len(set(bridge)) == 8
        assert controls == control_surface(rates)
        assert bridge == bridge_surface(rates)
        for tokenizer in tokenizers:
            inherited = inherited_rq0_surface(tokenizer, rates)
            rq1 = rq1_paired_surface(tokenizer, rates)
            assert len(inherited) == len(set(inherited)) == 8
            assert len(rq1) == len(set(rq1)) == 12
            for suffix_winner in tokenizers:
                for no_suffix_winner in tokenizers:
                    rq23 = rq23_paired_surface(
                        tokenizer,
                        rates,
                        suffix_winner=suffix_winner,
                        no_suffix_winner=no_suffix_winner,
                    )
                    assert len(rq23) == len(set(rq23)) == 20
