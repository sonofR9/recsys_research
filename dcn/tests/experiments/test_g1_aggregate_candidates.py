from __future__ import annotations

import pytest

from experiments.g1_sasrec_item_ids_likes.analysis.aggregate_candidates import (
    FIXED_MEMBERS,
    ApprovalRequired,
    AggregateCandidate,
    aggregate_boundary_candidates,
    aggregate_local_candidates,
    baseline_boundary_candidates,
    bridge_candidates,
    candidate_by_run,
    initial_candidates,
    make_horizon_correction,
    recovery_candidates,
    full_horizon_rerun_candidates,
    selection_initial_candidates,
)


def test_initial_manifest_is_the_approved_three_plus_nine_surface() -> None:
    candidates = initial_candidates()

    assert len(candidates) == 12
    assert len({candidate.run_name for candidate in candidates}) == 12
    assert [candidate.deep_lr for candidate in candidates[:3]] == [
        0.006,
        0.012,
        0.024,
    ]
    assert {
        (candidate.num_layers, candidate.embedding_lr, candidate.deep_lr)
        for candidate in candidates[3:]
    } == {
        (layers, embedding_lr, deep_lr)
        for layers in (4, 6, 8)
        for embedding_lr, deep_lr in (
            (0.064, 0.048),
            (0.07764674795069047, 0.02484672863178322),
            (0.0468526465053628, 0.032703745675187676),
        )
    }
    assert all(candidate_by_run(candidate.run_name) == candidate for candidate in candidates)


def test_selection_surface_replaces_only_the_two_incomplete_h15_runs() -> None:
    reruns = full_horizon_rerun_candidates()
    selection = selection_initial_candidates()

    assert [
        (candidate.num_layers, candidate.embedding_lr, candidate.deep_lr)
        for candidate in reruns
    ] == [
        (6, 0.064, 0.048),
        (8, 0.0468526465053628, 0.032703745675187676),
    ]
    assert all(candidate.stage == "full_horizon_rerun" for candidate in reruns)
    assert all(candidate.horizon_epochs == 15 for candidate in reruns)
    assert all("full_horizon_rerun" in candidate.run_name for candidate in reruns)
    assert len(selection) == 12
    assert set(selection) - set(initial_candidates()) == set(reruns)
    assert all(candidate_by_run(candidate.run_name) == candidate for candidate in reruns)


def test_bridge_manifest_changes_each_fixed_member_once_then_selected_depth() -> None:
    candidates = bridge_candidates(0.012, selected_depth=6)

    assert len(candidates) == 11
    assert tuple(candidate.member for candidate in candidates[:-1]) == FIXED_MEMBERS
    assert candidates[-1].member == "depth"
    assert candidates[-1].num_layers == 6
    assert all(candidate.embedding_lr == 0.064 for candidate in candidates)
    assert all(candidate.deep_lr == 0.012 for candidate in candidates)
    assert all(candidate_by_run(candidate.run_name) == candidate for candidate in candidates)


def test_baseline_boundary_round_is_exact_and_outer_winner_requires_approval() -> None:
    low = initial_candidates()[0]
    probes = baseline_boundary_candidates(low)

    assert [candidate.deep_lr for candidate in probes] == [0.003, 0.0015, 0.00075]
    assert baseline_boundary_candidates(probes[0]) == ()
    with pytest.raises(ApprovalRequired, match="outer baseline boundary"):
        baseline_boundary_candidates(probes[-1])


@pytest.mark.parametrize(
    ("initial_index", "expected"),
    [
        (
            1,
            (
                (0.06957712293357378, 0.04045869601192933),
                (0.058511920889791694, 0.03301809071022853),
                (0.08595109115349953, 0.03835282700655922),
            ),
        ),
        (
            2,
            (
                (0.06547725593418215, 0.027941927666112344),
                (0.06295219371532267, 0.04651981979406204),
                (0.046467420627589774, 0.0404566728379668),
            ),
        ),
    ],
)
def test_local_aggregate_round_is_frozen_by_initial_random_winner(
    initial_index: int, expected: tuple[tuple[float, float], ...]
) -> None:
    source = [
        candidate
        for candidate in initial_candidates()
        if candidate.family == "aggregate" and candidate.num_layers == 4
    ][initial_index]

    assert tuple(
        (candidate.embedding_lr, candidate.deep_lr)
        for candidate in aggregate_local_candidates(source)
    ) == expected


def test_optimizer_boundary_adds_three_probes_for_each_adjacent_coordinate() -> None:
    source = [
        candidate
        for candidate in initial_candidates()
        if candidate.family == "aggregate" and candidate.num_layers == 4
    ][1]
    local_low_deep = aggregate_local_candidates(source)[1]
    probes = aggregate_boundary_candidates(local_low_deep)

    assert [(candidate.embedding_lr, candidate.deep_lr) for candidate in probes] == [
        (local_low_deep.embedding_lr, 0.006),
        (local_low_deep.embedding_lr, 0.009524406311809197),
        (local_low_deep.embedding_lr, 0.01511905259873848),
    ]
    assert aggregate_boundary_candidates(probes[-1]) == ()
    with pytest.raises(ApprovalRequired, match="outer aggregate optimizer"):
        aggregate_boundary_candidates(probes[0])


def test_horizon_correction_preserves_recipe_and_stops_after_two() -> None:
    source = [
        candidate
        for candidate in initial_candidates()
        if candidate.family == "aggregate"
    ][0]
    first = make_horizon_correction(source, 18)
    second = make_horizon_correction(first, 21)

    assert first.horizon_epochs == 18
    assert second.horizon_epochs == 21
    assert second.correction == 2
    assert candidate_by_run(second.run_name) == second
    with pytest.raises(ApprovalRequired, match="two horizon corrections"):
        make_horizon_correction(second, 24)


@pytest.mark.parametrize(
    "run_name",
    [
        initial_candidates()[0].run_name.replace("_initial_", "_garbage_"),
        initial_candidates()[0].run_name.replace("_initial_", "_local_"),
        bridge_candidates(0.012)[0].run_name.replace("_bridge_ts2_", "_local_ts2_"),
    ],
)
def test_parser_rejects_unknown_or_illegal_family_stage_combinations(
    run_name: str,
) -> None:
    with pytest.raises(ValueError):
        candidate_by_run(run_name)


def test_constructor_rejects_uncorrected_horizon_correction_stage() -> None:
    with pytest.raises(ValueError, match="horizon-correction stage"):
        AggregateCandidate(
            "aggregate",
            0.064,
            0.048,
            "horizon_correction",
            num_layers=4,
            horizon_epochs=15,
        )


def test_recovery_manifest_is_the_exact_approved_eight_runs() -> None:
    candidates = recovery_candidates()

    assert len(candidates) == 8
    assert len({candidate.run_name for candidate in candidates}) == 8
    assert [
        (
            candidate.num_layers,
            candidate.embedding_lr,
            candidate.deep_lr,
            candidate.horizon_epochs,
            candidate.correction,
        )
        for candidate in candidates[:2]
    ] == [
        (6, 0.0468526465053628, 0.032703745675187676, 18, 3),
        (8, 0.07764674795069047, 0.02484672863178322, 13, 3),
    ]
    assert [candidate.deep_lr for candidate in candidates[2:5]] == [
        0.003,
        0.0015,
        0.00075,
    ]
    assert [
        (candidate.embedding_lr, candidate.deep_lr)
        for candidate in candidates[5:]
    ] == [
        (0.06547725593418215, 0.027941927666112344),
        (0.06295219371532267, 0.04651981979406204),
        (0.046467420627589774, 0.0404566728379668),
    ]
    assert all(candidate_by_run(candidate.run_name) == candidate for candidate in candidates)


def test_third_horizon_correction_is_limited_to_the_two_exact_recovery_chains() -> None:
    six_source = next(
        candidate
        for candidate in initial_candidates()
        if candidate.family == "aggregate"
        and candidate.num_layers == 6
        and candidate.embedding_lr == 0.0468526465053628
    )
    six_second = make_horizon_correction(
        make_horizon_correction(six_source, 23), 12
    )
    assert make_horizon_correction(six_second, 18) == recovery_candidates()[0]
    assert "_from15p23p12_horizon_correction_" in recovery_candidates()[0].run_name

    with pytest.raises(ApprovalRequired, match="third horizon correction"):
        make_horizon_correction(six_second, 19)

    wrong_middle = make_horizon_correction(
        make_horizon_correction(six_source, 22), 12
    )
    with pytest.raises(ApprovalRequired, match="full approved predecessor chain"):
        make_horizon_correction(wrong_middle, 18)

    eight_source = next(
        candidate
        for candidate in initial_candidates()
        if candidate.family == "aggregate"
        and candidate.num_layers == 8
        and candidate.embedding_lr == 0.07764674795069047
    )
    wrong_eight_middle = make_horizon_correction(
        make_horizon_correction(eight_source, 23), 16
    )
    with pytest.raises(ApprovalRequired, match="full approved predecessor chain"):
        make_horizon_correction(wrong_eight_middle, 13)

    ordinary = next(
        candidate
        for candidate in initial_candidates()
        if candidate.family == "aggregate"
        and candidate.num_layers == 4
        and candidate.embedding_lr == 0.064
    )
    ordinary_second = make_horizon_correction(
        make_horizon_correction(ordinary, 23), 12
    )
    with pytest.raises(ApprovalRequired, match="two horizon corrections"):
        make_horizon_correction(ordinary_second, 18)


@pytest.mark.parametrize(
    "run_name",
    [
        recovery_candidates()[0].run_name.replace("_h18_", "_h19_"),
        recovery_candidates()[1].run_name.replace("_l8_", "_l4_"),
        recovery_candidates()[0].run_name.replace("from15p23p12", "from15p22p12"),
    ],
)
def test_parser_rejects_unapproved_third_correction_runs(run_name: str) -> None:
    with pytest.raises(ValueError, match="third horizon correction"):
        candidate_by_run(run_name)


def test_noncanonical_h20_origin_cannot_launder_through_correction_round_trip() -> None:
    source = next(
        candidate
        for candidate in initial_candidates()
        if candidate.family == "aggregate"
        and candidate.num_layers == 6
        and candidate.embedding_lr == 0.0468526465053628
    )
    fabricated_origin = AggregateCandidate(
        source.family,
        source.embedding_lr,
        source.deep_lr,
        source.stage,
        num_layers=source.num_layers,
        horizon_epochs=20,
    )

    with pytest.raises(ValueError, match="canonical approved H15"):
        first = make_horizon_correction(fabricated_origin, 23)
        parsed_first = candidate_by_run(first.run_name)
        second = make_horizon_correction(parsed_first, 12)
        make_horizon_correction(candidate_by_run(second.run_name), 18)


def test_legitimate_existing_correction_names_parse_and_continue() -> None:
    source = next(
        candidate
        for candidate in initial_candidates()
        if candidate.family == "aggregate"
        and candidate.num_layers == 6
        and candidate.embedding_lr == 0.0468526465053628
    )
    first = make_horizon_correction(source, 23)
    parsed_first = candidate_by_run(first.run_name)
    second = make_horizon_correction(parsed_first, 12)
    parsed_second = candidate_by_run(second.run_name)

    assert parsed_first.run_name == first.run_name
    assert parsed_second.run_name == second.run_name
    assert make_horizon_correction(parsed_second, 18) == recovery_candidates()[0]


@pytest.mark.parametrize(
    ("correction", "horizon", "chain"),
    [
        (1, 23, (20, 23)),
        (2, 12, (20, 23, 12)),
    ],
)
def test_direct_forged_corrections_cannot_serialize_noncanonical_origins(
    correction: int,
    horizon: int,
    chain: tuple[int, ...],
) -> None:
    source = next(
        candidate
        for candidate in initial_candidates()
        if candidate.family == "aggregate"
        and candidate.num_layers == 6
        and candidate.embedding_lr == 0.0468526465053628
    )

    with pytest.raises(ValueError, match="canonical H15"):
        forged = AggregateCandidate(
            source.family,
            source.embedding_lr,
            source.deep_lr,
            "horizon_correction",
            num_layers=source.num_layers,
            horizon_epochs=horizon,
            correction=correction,
            horizon_chain=chain,
        )
        forged.run_name


def test_fourth_correction_is_the_exact_h27_post_recovery_chain() -> None:
    third = recovery_candidates()[0]
    fourth = make_horizon_correction(third, 27)

    assert fourth.correction == 4
    assert fourth.horizon_chain == (15, 23, 12, 18, 27)
    assert "_from15p23p12p18_horizon_correction_" in fourth.run_name
    assert candidate_by_run(fourth.run_name) == fourth

    with pytest.raises(ApprovalRequired, match="fourth horizon correction"):
        make_horizon_correction(third, 26)
    with pytest.raises(ApprovalRequired, match="third horizon correction"):
        make_horizon_correction(recovery_candidates()[1], 27)


@pytest.mark.parametrize(
    "run_name",
    [
        make_horizon_correction(recovery_candidates()[0], 27).run_name.replace(
            "_h27_", "_h28_"
        ),
        make_horizon_correction(recovery_candidates()[0], 27).run_name.replace(
            "from15p23p12p18", "from15p23p12p17"
        ),
    ],
)
def test_parser_rejects_unapproved_fourth_correction_runs(run_name: str) -> None:
    with pytest.raises(ValueError, match="fourth horizon correction"):
        candidate_by_run(run_name)
