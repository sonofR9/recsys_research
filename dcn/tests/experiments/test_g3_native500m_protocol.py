from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from experiments.g3_pretrained_item_embeddings.protocol.native500m.compiler import (
    SelectedCoordinate,
    compile_baseline_rows,
    compile_boundary_rows,
    compile_capacity_first_stage,
    compile_capacity_followup,
    compile_nine_cell_family,
    compile_rq5_frequency_first_stage,
    compile_rq5_frequency_followup,
    compile_rq5_global_rows,
)
from experiments.g3_pretrained_item_embeddings.protocol.native500m.constants import (
    BASELINE_ANCHOR,
    FAMILY_SPECS,
    PROTOCOL,
    boundary_extension_budget,
)
from experiments.g3_pretrained_item_embeddings.protocol.native500m.selection import (
    AuthenticatedSelectedCoordinate,
    CandidateResult,
    authenticate_selected_coordinate,
    required_boundary_extensions,
    select_preliminary_winner,
    select_winner,
)


def _family(family_id: str):
    return next(spec for spec in FAMILY_SPECS if spec.id == family_id)


def _selected(
    *,
    family_id: str = "baseline",
    embedding_learning_rate: float = BASELINE_ANCHOR.embedding_learning_rate,
    deep_learning_rate: float = BASELINE_ANCHOR.deep_learning_rate,
    horizon_epochs: int = 20,
    capacity: int | None = None,
) -> SelectedCoordinate:
    return SelectedCoordinate.create(
        source_id=f"{family_id}:selected",
        family_id=family_id,
        embedding_learning_rate=embedding_learning_rate,
        deep_learning_rate=deep_learning_rate,
        horizon_epochs=horizon_epochs,
        capacity=capacity,
    )


def _results(
    rows,
    *,
    winner_index: int = 0,
) -> tuple[CandidateResult, ...]:
    return tuple(
        CandidateResult(
            row,
            recall_at_100=0.13 if index == winner_index else 0.12,
            ndcg_at_100=0.05 if index == winner_index else 0.04,
            best_epoch=row.horizon_epochs,
            epochs_trained=row.horizon_epochs,
        )
        for index, row in enumerate(rows)
    )


def _authenticated_baseline():
    rows = compile_baseline_rows()
    results = _results(rows, winner_index=1)
    return authenticate_selected_coordinate(results, expected_rows=rows)


def _authenticated_rq2():
    predecessor = _authenticated_baseline()
    spec = _family("rq2_content_concat")
    initial = compile_capacity_first_stage(spec, predecessor.coordinate)
    initial_results = _results(initial, winner_index=3)
    followup = compile_capacity_followup(spec, predecessor.coordinate, initial[3])
    rows = (*initial, *followup)
    results = (*initial_results, *_results(followup))
    return authenticate_selected_coordinate(
        results,
        expected_rows=rows,
        predecessor=predecessor,
    )


def test_protocol_is_closed_native500m_and_budgeted() -> None:
    assert PROTOCOL.schema_version == 1
    assert PROTOCOL.dataset_size == "native-500m"
    assert PROTOCOL.data_group == "g3-native500m-likes"
    assert PROTOCOL.batch_size == 512
    assert PROTOCOL.seed == 42
    assert PROTOCOL.horizon_epochs == (10, 20, 40)
    assert PROTOCOL.minimum_query_events == 2
    assert PROTOCOL.num_items == 157357
    assert PROTOCOL.filtered_event_count == 8304589
    assert PROTOCOL.filtered_user_count == 81926
    assert PROTOCOL.remapped_event_count == 8013866
    assert PROTOCOL.remapped_user_count == 81635
    assert PROTOCOL.training_interaction_count == 7755722
    assert PROTOCOL.training_user_count == 81020
    assert PROTOCOL.evaluation_user_count == 37018
    assert PROTOCOL.evaluable_user_count == 37018
    assert PROTOCOL.validation_cutoff_timestamp == 25395195
    assert PROTOCOL.content_width == 128
    assert PROTOCOL.content_sha256 == (
        "647b62ccc6cb214181e6aa44768fe94abd69e840b7758824f8e521dfe040043c"
    )
    assert PROTOCOL.initial_opportunity_budget == 140
    assert PROTOCOL.with_conditional_opportunity_budget == 167
    assert PROTOCOL.maximum_opportunity_budget == 295
    assert sum(spec.budget for spec in FAMILY_SPECS if not spec.conditional) == 140
    assert sum(spec.budget for spec in FAMILY_SPECS if spec.conditional) == 27
    assert 167 + sum(boundary_extension_budget(spec) for spec in FAMILY_SPECS) == 295
    assert all(width % 16 == 0 for spec in FAMILY_SPECS for width in spec.capacities)

    with pytest.raises(FrozenInstanceError):
        PROTOCOL.batch_size = 128


def test_family_predecessors_are_explicit_and_rq4_is_independent() -> None:
    assert _family("untied_control").search_predecessor_id == "baseline"
    assert _family("rq1_content_input").search_predecessor_id == "baseline"
    assert _family("rq2_content_concat").search_predecessor_id == "baseline"
    assert {
        _family(f"rq3_output_{variant}").search_predecessor_id
        for variant in (
            "learned",
            "frozen_content",
            "trainable_content",
            "learned_frozen_content",
            "learned_trainable_content",
        )
    } == {"rq2_content_concat"}
    assert {
        _family(family_id).search_predecessor_id
        for family_id in ("rq4_artist", "rq4_album", "rq4_artist_album")
    } == {"baseline"}
    assert _family("rq5_global_gate").search_predecessor_id == "rq2_content_concat"
    assert _family("rq5_frequency_gate").search_predecessor_id == "rq2_content_concat"
    assert _family("rq5_frequency_gate").promotion_predecessor_id == "rq5_global_gate"


def test_baseline_rows_are_exact_anchor_plus_scrambled_sobol_pairs() -> None:
    rows = compile_baseline_rows()

    assert len(rows) == 9
    assert [row.horizon_epochs for row in rows] == [10] * 3 + [20] * 3 + [40] * 3
    assert [row.manifest_order for row in rows] == list(range(9))
    assert all(row.batch_size == 512 and row.seed == 42 for row in rows)
    assert [
        (row.embedding_learning_rate_text, row.deep_learning_rate_text) for row in rows
    ] == [
        ("0.046852646505362798", "0.032703745675187676"),
        ("0.06375957559078467", "0.033592533248942007"),
        ("0.03996662827497631", "0.025850902250806791"),
        ("0.046852646505362798", "0.032703745675187676"),
        ("0.093265638113829", "0.052869980163324198"),
        ("0.037244925692432665", "0.031983368359407911"),
        ("0.046852646505362798", "0.032703745675187676"),
        ("0.09152246809261437", "0.03128309208399048"),
        ("0.027995819103053991", "0.034516357983428329"),
    ]
    assert all(
        row.embedding_learning_rate_text == format(row.embedding_learning_rate, ".17g")
        and row.deep_learning_rate_text == format(row.deep_learning_rate, ".17g")
        for row in rows
    )
    assert compile_baseline_rows() == rows

    baseline = _family("baseline")
    for forged in (
        SelectedCoordinate.create(
            source_id=BASELINE_ANCHOR.source_id,
            family_id="g1_aggregate",
            embedding_learning_rate=0.05,
            deep_learning_rate=BASELINE_ANCHOR.deep_learning_rate,
            horizon_epochs=20,
        ),
        SelectedCoordinate.create(
            source_id=BASELINE_ANCHOR.source_id,
            family_id="wrong_g1_family",
            embedding_learning_rate=BASELINE_ANCHOR.embedding_learning_rate,
            deep_learning_rate=BASELINE_ANCHOR.deep_learning_rate,
            horizon_epochs=20,
        ),
    ):
        with pytest.raises(ValueError, match="exact selected G1 anchor"):
            compile_nine_cell_family(baseline, forged)


def test_regular_and_capacity_compilers_use_selected_predecessor() -> None:
    predecessor = _selected(
        embedding_learning_rate=0.04,
        deep_learning_rate=0.03,
    )
    regular = compile_nine_cell_family(_family("rq1_content_input"), predecessor)
    assert len(regular) == 9
    assert all(row.predecessor_id == predecessor.source_id for row in regular)
    assert [row.horizon_epochs for row in regular] == [10] * 3 + [20] * 3 + [40] * 3
    assert [row.embedding_learning_rate for row in regular[::3]] == [0.04] * 3
    assert [row.deep_learning_rate for row in regular[::3]] == [0.03] * 3

    spec = _family("rq2_content_concat")
    first = compile_capacity_first_stage(spec, predecessor)
    assert len(first) == 9
    assert [row.capacity for row in first] == [64] * 3 + [128] * 3 + [256] * 3
    assert all(row.horizon_epochs == 20 for row in first)
    selected = first[4]
    followup = compile_capacity_followup(spec, predecessor, selected)
    assert len(followup) == 3
    assert [row.horizon_epochs for row in followup] == [10, 40, 20]
    assert [row.capacity for row in followup] == [128, 128, 128]
    assert followup[0].learning_rate_pair == selected.learning_rate_pair
    assert followup[1].learning_rate_pair == selected.learning_rate_pair
    assert followup[2].learning_rate_pair not in {
        row.learning_rate_pair for row in first
    }
    assert [row.manifest_order for row in (*first, *followup)] == list(range(12))


def test_rq5_fixes_selected_rq2_embedding_rate() -> None:
    rq2 = _selected(
        family_id="rq2_content_concat",
        embedding_learning_rate=0.061,
        deep_learning_rate=0.025,
        capacity=128,
    )
    global_rows = compile_rq5_global_rows(rq2)
    frequency_first = compile_rq5_frequency_first_stage(rq2)
    selected = frequency_first[4]
    frequency_followup = compile_rq5_frequency_followup(rq2, selected)

    assert len(global_rows) == 9
    assert [row.horizon_epochs for row in global_rows] == [10] * 3 + [20] * 3 + [40] * 3
    assert [row.deep_learning_rate for row in global_rows[:3]] == [0.0125, 0.025, 0.05]
    assert len(frequency_first) == 9
    assert [row.capacity for row in frequency_first] == [32] * 3 + [64] * 3 + [96] * 3
    assert len(frequency_followup) == 2
    assert [row.horizon_epochs for row in frequency_followup] == [10, 40]
    assert all(
        row.embedding_learning_rate_text == rq2.embedding_learning_rate_text
        for row in (*global_rows, *frequency_first, *frequency_followup)
    )
    assert [
        row.manifest_order for row in (*frequency_first, *frequency_followup)
    ] == list(range(11))


def test_selection_uses_recall_ndcg_then_manifest_order() -> None:
    rows = compile_baseline_rows()
    results = [
        CandidateResult(
            row,
            0.12 if row.manifest_order in {1, 2} else 0.11,
            0.05 if row.manifest_order in {1, 2} else 0.04,
            best_epoch=min(8, row.horizon_epochs),
            epochs_trained=row.horizon_epochs,
        )
        for row in rows
    ]

    assert select_winner(results, expected_rows=rows) == results[1]

    with pytest.raises(ValueError, match="complete expected-row ledger"):
        select_winner(results[:3], expected_rows=rows)
    with pytest.raises(ValueError, match="metric-selected followup"):
        select_winner(results[:3], expected_rows=rows[:3])

    with pytest.raises(ValueError, match="same family"):
        select_winner(
            [
                results[0],
                CandidateResult(
                    replace(rows[1], family_id="wrong"),
                    recall_at_100=0.13,
                    ndcg_at_100=0.06,
                    best_epoch=10,
                    epochs_trained=10,
                ),
            ],
            expected_rows=rows[:2],
        )
    tampered = replace(
        results[0],
        row=replace(rows[0], deep_learning_rate_text="0.01"),
    )
    with pytest.raises(ValueError, match="differ from the expected-row ledger"):
        select_winner([tampered, *results[1:]], expected_rows=rows)

    forged_rows = tuple(
        replace(
            row,
            embedding_learning_rate_text=format(
                row.embedding_learning_rate * 1.1,
                ".17g",
            ),
        )
        for row in rows
    )
    forged_results = tuple(
        replace(result, row=forged_row)
        for result, forged_row in zip(results, forged_rows, strict=True)
    )
    with pytest.raises(ValueError, match="compiler-generated"):
        select_winner(forged_results, expected_rows=forged_rows)


def test_selection_authenticates_capacity_followup_against_initial_winner() -> None:
    authenticated_predecessor = _authenticated_baseline()
    predecessor = authenticated_predecessor.coordinate
    spec = _family("rq2_content_concat")
    initial = compile_capacity_first_stage(spec, predecessor)
    initial_results = _results(initial, winner_index=3)
    winner = select_preliminary_winner(
        initial_results,
        expected_rows=initial,
        predecessor=authenticated_predecessor,
    )
    with pytest.raises(ValueError, match="followup"):
        select_winner(
            initial_results,
            expected_rows=initial,
            predecessor=authenticated_predecessor,
        )
    followup = compile_capacity_followup(spec, predecessor, winner.row)
    rows = (*initial, *followup)
    results = (*initial_results, *_results(followup))

    assert (
        select_winner(
            results,
            expected_rows=rows,
            predecessor=authenticated_predecessor,
        ).row
        == initial[3]
    )

    wrong_followup = compile_capacity_followup(spec, predecessor, initial[0])
    wrong_rows = (*initial, *wrong_followup)
    wrong_results = (*initial_results, *_results(wrong_followup))
    with pytest.raises(ValueError, match="compiler-generated"):
        select_winner(
            wrong_results,
            expected_rows=wrong_rows,
            predecessor=authenticated_predecessor,
        )


def test_nonbaseline_selection_rejects_forged_predecessor() -> None:
    authenticated_predecessor = _authenticated_baseline()
    attacker = SelectedCoordinate.create(
        source_id="attacker:selected",
        family_id="baseline",
        embedding_learning_rate=0.2,
        deep_learning_rate=0.1,
        horizon_epochs=20,
    )
    rows = compile_nine_cell_family(_family("rq1_content_input"), attacker)
    results = _results(rows, winner_index=4)

    for selector in (select_preliminary_winner, select_winner):
        with pytest.raises(ValueError, match="authenticated predecessor"):
            selector(
                results,
                expected_rows=rows,
                predecessor=authenticated_predecessor,
            )

    winner = results[4]
    requests = required_boundary_extensions(winner, rows)
    with pytest.raises(ValueError, match="authenticated predecessor"):
        compile_boundary_rows(
            winner,
            rows,
            existing_results=results,
            predecessor=authenticated_predecessor,
            requests=requests,
        )


def test_authenticated_coordinate_binds_the_exact_final_winner() -> None:
    authenticated = _authenticated_rq2()
    row = authenticated.selected_result.row
    coordinate = authenticated.coordinate

    assert coordinate.source_id == row.id
    assert coordinate.family_id == row.family_id
    assert coordinate.learning_rate_pair == row.learning_rate_pair
    assert coordinate.horizon_epochs == row.horizon_epochs
    assert coordinate.capacity == row.capacity
    with pytest.raises(AttributeError, match="immutable"):
        authenticated.protocol_sha256 = "forged"
    with pytest.raises(ValueError, match="not produced by authentication"):
        AuthenticatedSelectedCoordinate(
            selected_result=authenticated.selected_result,
            protocol_sha256=authenticated.protocol_sha256,
            _seal=object(),
        )


def test_one_round_boundary_compiler_is_directional_and_bounded() -> None:
    rows = compile_baseline_rows()
    selected_row = max(rows, key=lambda row: row.embedding_learning_rate)
    winner_index = rows.index(selected_row)
    results = _results(rows, winner_index=winner_index)
    winner = results[winner_index]
    requests = required_boundary_extensions(winner, rows)

    assert [(request.axis, request.direction) for request in requests] == [
        ("embedding_learning_rate", "high"),
        ("deep_learning_rate", "high"),
    ]
    with pytest.raises(ValueError, match="canonical boundary request order"):
        compile_boundary_rows(
            winner,
            rows,
            existing_results=results,
            requests=(requests[0],),
        )
    extension = compile_boundary_rows(
        winner,
        rows,
        existing_results=results,
        requests=requests,
    )
    with pytest.raises(ValueError, match="canonical boundary request order"):
        compile_boundary_rows(
            winner,
            rows,
            existing_results=results,
            requests=tuple(reversed(requests)),
        )
    assert len(extension) == 6
    assert [row.embedding_learning_rate for row in extension[:3]] == pytest.approx(
        [
            selected_row.embedding_learning_rate * 2,
            selected_row.embedding_learning_rate * 4,
            selected_row.embedding_learning_rate * 8,
        ]
    )
    assert all(
        row.deep_learning_rate_text == selected_row.deep_learning_rate_text
        and row.horizon_epochs == selected_row.horizon_epochs
        for row in extension[:3]
    )

    with pytest.raises(ValueError, match="one boundary round"):
        compile_boundary_rows(
            winner,
            rows,
            existing_results=results,
            requests=requests,
            round_number=2,
        )
    with pytest.raises(ValueError, match="canonical boundary request order"):
        compile_boundary_rows(
            winner,
            rows,
            existing_results=results,
            requests=(replace(requests[0], value=123.0),),
        )
    with pytest.raises(ValueError, match="second boundary round"):
        compile_boundary_rows(
            CandidateResult(extension[-1], 0.14, 0.06, 20, 20),
            (*rows, *extension),
            existing_results=(*results, *_results(extension)),
            requests=(),
        )

    boundary_results = _results(extension)
    assert select_winner(
        (*results, *boundary_results),
        expected_rows=(*rows, *extension),
    )

    with pytest.raises(ValueError, match="required boundary"):
        select_winner(results, expected_rows=rows)
    unresolved_results = (
        *results,
        *_results(extension, winner_index=len(extension) - 1),
    )
    unresolved_results = tuple(
        (
            replace(result, recall_at_100=0.14, ndcg_at_100=0.06)
            if result.row == extension[-1]
            else result
        )
        for result in unresolved_results
    )
    with pytest.raises(ValueError, match="unresolved after one boundary round"):
        select_winner(
            unresolved_results,
            expected_rows=(*rows, *extension),
        )


def test_capacity_and_rq5_boundaries_follow_family_rules() -> None:
    authenticated_predecessor = _authenticated_baseline()
    predecessor = authenticated_predecessor.coordinate
    spec = _family("rq2_content_concat")
    first = compile_capacity_first_stage(spec, predecessor)
    first_results = _results(first)
    first_winner = first_results[0]
    followup = compile_capacity_followup(spec, predecessor, first_winner.row)
    rows = (*first, *followup)
    results = (*first_results, *_results(followup, winner_index=2))
    winner = select_preliminary_winner(
        results,
        expected_rows=rows,
        predecessor=authenticated_predecessor,
    )
    requests = required_boundary_extensions(winner, rows)
    capacity = next(request for request in requests if request.axis == "capacity")
    assert (capacity.direction, capacity.value) == ("low", 32)
    capacity_rows = compile_boundary_rows(
        winner,
        rows,
        existing_results=results,
        predecessor=authenticated_predecessor,
        requests=requests,
    )
    assert len([row for row in capacity_rows if row.capacity == 32]) == 3
    capacity_results = _results(capacity_rows, winner_index=0)
    capacity_results = tuple(
        (
            replace(result, recall_at_100=0.14, ndcg_at_100=0.06)
            if result.row.capacity == 32
            else result
        )
        for result in capacity_results
    )
    with pytest.raises(ValueError, match="unresolved after one boundary round"):
        select_winner(
            (*results, *capacity_results),
            expected_rows=(*rows, *capacity_rows),
            predecessor=authenticated_predecessor,
        )

    with pytest.raises(ValueError, match="complete canonical family ledger"):
        compile_boundary_rows(
            first_results[0],
            first[:3],
            existing_results=first_results[:3],
            predecessor=authenticated_predecessor,
            requests=(),
        )

    authenticated_rq2 = _authenticated_rq2()
    rq2 = authenticated_rq2.coordinate
    frequency = compile_rq5_frequency_first_stage(rq2)
    frequency_initial_results = _results(frequency)
    frequency_followup = compile_rq5_frequency_followup(rq2, frequency[0])
    frequency_rows = (*frequency, *frequency_followup)
    frequency_results = (
        *frequency_initial_results,
        *_results(frequency_followup, winner_index=1),
    )
    with pytest.raises(ValueError, match="followup"):
        select_winner(
            frequency_initial_results,
            expected_rows=frequency,
            predecessor=authenticated_rq2,
        )
    frequency_winner = select_preliminary_winner(
        frequency_results,
        expected_rows=frequency_rows,
        predecessor=authenticated_rq2,
    )
    frequency_requests = required_boundary_extensions(
        frequency_winner,
        frequency_rows,
    )
    assert all(
        request.axis != "embedding_learning_rate" for request in frequency_requests
    )
    frequency_capacity = next(
        request for request in frequency_requests if request.axis == "capacity"
    )
    assert (frequency_capacity.direction, frequency_capacity.value) == ("low", 16)
    with pytest.raises(ValueError, match="canonical boundary request order"):
        compile_boundary_rows(
            frequency_winner,
            frequency_rows,
            existing_results=frequency_results,
            predecessor=authenticated_rq2,
            requests=(frequency_capacity,),
        )
    new_width = compile_boundary_rows(
        frequency_winner,
        frequency_rows,
        existing_results=frequency_results,
        predecessor=authenticated_rq2,
        requests=frequency_requests,
    )
    capacity_rows = [row for row in new_width if row.capacity == 16]
    assert [row.deep_learning_rate for row in capacity_rows] == pytest.approx(
        [rq2.deep_learning_rate * factor for factor in (0.5, 1.0, 2.0)]
    )
    assert all(
        row.embedding_learning_rate == rq2.embedding_learning_rate for row in new_width
    )
    assert (
        select_winner(
            (*frequency_results, *_results(new_width)),
            expected_rows=(*frequency_rows, *new_width),
            predecessor=authenticated_rq2,
        )
        == frequency_winner
    )


def test_horizon_60_requires_h40_winner_restored_at_endpoint() -> None:
    rows = compile_baseline_rows()
    row = rows[-3]
    incomplete = CandidateResult(row, 0.13, 0.05, 39, 40)
    endpoint = CandidateResult(row, 0.13, 0.05, 40, 40)

    assert all(
        request.axis != "horizon_epochs"
        for request in required_boundary_extensions(incomplete, rows)
    )
    request = next(
        request
        for request in required_boundary_extensions(endpoint, rows)
        if request.axis == "horizon_epochs"
    )
    results = list(_results(rows, winner_index=rows.index(row)))
    results[rows.index(row)] = endpoint
    extension = compile_boundary_rows(
        endpoint,
        rows,
        existing_results=results,
        requests=(request,),
    )
    assert len(extension) == 1
    assert extension[0].horizon_epochs == 60

    unresolved = CandidateResult(extension[0], 0.14, 0.06, 60, 60)
    with pytest.raises(ValueError, match="unresolved after one boundary round"):
        select_winner(
            (*results, unresolved),
            expected_rows=(*rows, *extension),
        )
