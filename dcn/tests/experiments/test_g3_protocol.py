from dataclasses import FrozenInstanceError

import pytest

from experiments.g3_pretrained_item_embeddings.protocol.constants import (
    APPROVED_PROTOCOL,
)
from experiments.g3_pretrained_item_embeddings.protocol.search import (
    APPROVED_FAMILY_SPECS,
    CONDITIONAL_FAMILY_SPECS,
    ReusableCoordinate,
    TransferredHorizonRate,
    compile_capacity_first_stage,
    compile_capacity_horizon_followup,
    compile_approved_search,
    compile_family,
    compile_rq4_extra_id_control,
)


EXPECTED_INITIAL_BUDGETS = {
    "untied_control": 10,
    "rq1_content_input": 9,
    "rq2_content_concat": 12,
    "rq2_id_only_densenet": 12,
    "rq3_output_learned": 9,
    "rq3_output_frozen_content": 9,
    "rq3_output_trainable_content": 9,
    "rq3_output_learned_frozen_content": 9,
    "rq3_output_learned_trainable_content": 9,
    "rq4_artist": 12,
    "rq4_album": 12,
    "rq4_artist_album": 12,
    "rq4_extra_item_id": 12,
    "rq5_global_gate": 12,
    "rq5_frequency_gate": 12,
    "size_500m_tied_baseline": 9,
    "size_500m_frozen_treatment": 9,
}


def _transferred_horizon_rates() -> tuple[TransferredHorizonRate, ...]:
    return (
        TransferredHorizonRate(15, 0.10, 0.02),
        TransferredHorizonRate(
            25,
            APPROVED_PROTOCOL.control.embedding_learning_rate,
            APPROVED_PROTOCOL.control.deep_learning_rate,
        ),
        TransferredHorizonRate(40, 0.20, 0.04),
    )


def _rq4_predecessor(*, transfer_accepted: bool):
    spec = next(spec for spec in APPROVED_FAMILY_SPECS if spec.id == "rq4_artist_album")
    if not transfer_accepted:
        return compile_family(spec, transfer_accepted=False)
    first_stage = compile_capacity_first_stage(spec)
    return (*first_stage, *compile_capacity_horizon_followup(
        spec,
        selected_capacity=32,
        transferred_horizon_rates=_transferred_horizon_rates(),
        first_stage=first_stage,
    ))


def test_approved_protocol_is_immutable_and_binds_the_selected_control() -> None:
    assert APPROVED_PROTOCOL.batch_size == 512
    assert APPROVED_PROTOCOL.dataset_sizes == ("native-50m", "native-500m")
    assert APPROVED_PROTOCOL.control.manifest_sha256 == (
        "c30fb4eafcea2cefa1099631a40ca1531245e412c1cedcdbd02d9f7fea7aafd6"
    )
    assert APPROVED_PROTOCOL.control.run_name == "g4_control_trial_16_native50m"
    assert APPROVED_PROTOCOL.control.embedding_learning_rate == 0.1474458978470563
    assert APPROVED_PROTOCOL.control.deep_learning_rate == 0.032433939334700325
    assert APPROVED_PROTOCOL.control.horizon_epochs == 25
    assert APPROVED_PROTOCOL.control.best_epoch == 20
    assert APPROVED_PROTOCOL.control.recall_at_100 == 0.10435560161495364
    assert APPROVED_PROTOCOL.content_sha256 == (
        ("native-50m", "aa14c76ea36d5a9b8730bd856ba0f0e90bc7230a7179e04650b22d5a9572dd64"),
        ("native-500m", "647b62ccc6cb214181e6aa44768fe94abd69e840b7758824f8e521dfe040043c"),
    )

    with pytest.raises(FrozenInstanceError):
        APPROVED_PROTOCOL.batch_size = 128


def test_compiler_accounts_for_every_equal_budget_opportunity() -> None:
    assert {spec.id: spec.budget for spec in APPROVED_FAMILY_SPECS} == (
        EXPECTED_INITIAL_BUDGETS
    )
    assert {spec.id: spec.budget for spec in CONDITIONAL_FAMILY_SPECS} == {
        "bridge_rq3_output": 9,
        "bridge_rq4_metadata": 9,
        "aggregate": 9,
    }
    selected_capacities = {
        spec.id: spec.capacities[len(spec.capacities) // 2]
        for spec in (*APPROVED_FAMILY_SPECS, *CONDITIONAL_FAMILY_SPECS)
        if spec.capacities
    }
    transferred_horizon_rates = {
        spec.id: _transferred_horizon_rates()
        for spec in APPROVED_FAMILY_SPECS
        if spec.capacities
    }

    first = compile_approved_search(
        selected_capacities=selected_capacities,
        transferred_horizon_rates=transferred_horizon_rates,
        rq4_extra_id_predecessor=_rq4_predecessor(transfer_accepted=True),
    )
    second = compile_approved_search(
        selected_capacities=selected_capacities,
        transferred_horizon_rates=transferred_horizon_rates,
        rq4_extra_id_predecessor=_rq4_predecessor(transfer_accepted=True),
    )

    assert first == second
    assert first.initial_opportunity_count == 178
    assert first.conditional_opportunity_count == 27
    assert first.maximum_opportunity_count == 205
    assert len(first.physical_coordinates) == 199
    assert len({coordinate.id for coordinate in first.physical_coordinates}) == 199
    assert all(coordinate.batch_size == 512 for coordinate in first.physical_coordinates)
    assert all(
        APPROVED_PROTOCOL.embedding_lr_bounds[0]
        <= coordinate.embedding_learning_rate
        <= APPROVED_PROTOCOL.embedding_lr_bounds[1]
        for coordinate in first.physical_coordinates
    )
    assert all(
        APPROVED_PROTOCOL.deep_lr_bounds[0]
        <= coordinate.deep_learning_rate
        <= APPROVED_PROTOCOL.deep_lr_bounds[1]
        for coordinate in first.physical_coordinates
    )


def test_reused_compatible_coordinates_count_toward_equal_family_budget() -> None:
    spec = next(
        spec for spec in APPROVED_FAMILY_SPECS if spec.id == "rq3_output_learned"
    )
    reusable = (
        ReusableCoordinate(
            source_id="rq2:selected:one",
            embedding_learning_rate=0.12,
            deep_learning_rate=0.03,
            horizon_epochs=15,
        ),
        ReusableCoordinate(
            source_id="rq2:selected:two",
            embedding_learning_rate=0.14,
            deep_learning_rate=0.04,
            horizon_epochs=25,
        ),
    )

    compiled = compile_family(spec, reusable=reusable)

    assert len(compiled) == 9
    assert [coordinate.reused_from for coordinate in compiled[:2]] == [
        "rq2:selected:one",
        "rq2:selected:two",
    ]
    assert sum(coordinate.reused_from is None for coordinate in compiled) == 7


def test_rq3_output_families_share_exact_search_coordinates() -> None:
    specs = [
        spec for spec in APPROVED_FAMILY_SPECS if spec.id.startswith("rq3_output_")
    ]
    signatures = [
        [
            (
                coordinate.embedding_learning_rate,
                coordinate.deep_learning_rate,
                coordinate.horizon_epochs,
            )
            for coordinate in compile_family(spec)
        ]
        for spec in specs
    ]

    assert len(signatures) == 5
    assert all(signature == signatures[0] for signature in signatures[1:])


def test_rq5_global_shares_capacity_stage_rate_horizon_opportunities() -> None:
    global_spec = next(spec for spec in APPROVED_FAMILY_SPECS if spec.id == "rq5_global_gate")
    frequency_spec = next(spec for spec in APPROVED_FAMILY_SPECS if spec.id == "rq5_frequency_gate")
    global_signatures = {
        (coordinate.embedding_learning_rate, coordinate.deep_learning_rate, coordinate.horizon_epochs)
        for coordinate in compile_family(global_spec)
    }
    capacity_signatures = {
        (coordinate.embedding_learning_rate, coordinate.deep_learning_rate, coordinate.horizon_epochs)
        for coordinate in compile_capacity_first_stage(frequency_spec)
    }

    assert capacity_signatures <= global_signatures


def test_rq4_extra_id_requires_and_matches_completed_metadata_opportunities() -> None:
    extra_spec = next(spec for spec in APPROVED_FAMILY_SPECS if spec.id == "rq4_extra_item_id")
    metadata_spec = next(spec for spec in APPROVED_FAMILY_SPECS if spec.id == "rq4_artist_album")
    first_stage = compile_capacity_first_stage(metadata_spec)
    predecessor = (*first_stage, *compile_capacity_horizon_followup(
        metadata_spec,
        selected_capacity=32,
        transferred_horizon_rates=_transferred_horizon_rates(),
        first_stage=first_stage,
    ))

    with pytest.raises(ValueError, match="predecessor"):
        compile_family(extra_spec)
    compiled = compile_rq4_extra_id_control(extra_spec, predecessor=predecessor)
    assert len(compiled) == 12
    assert [
        (row.embedding_learning_rate, row.deep_learning_rate, row.horizon_epochs)
        for row in compiled
    ] == [
        (row.embedding_learning_rate, row.deep_learning_rate, row.horizon_epochs)
        for row in predecessor
    ]


def test_compiler_rejects_incompatible_reuse_and_preserves_rejected_transfer() -> None:
    capacity_spec = next(
        spec for spec in APPROVED_FAMILY_SPECS if spec.id == "rq2_content_concat"
    )
    rejected_transfer = compile_family(capacity_spec, transfer_accepted=False)
    assert len(rejected_transfer) == 12
    assert {
        capacity: sum(coordinate.capacity == capacity for coordinate in rejected_transfer)
        for capacity in capacity_spec.capacities
    } == {64: 4, 128: 4, 256: 4}
    assert {
        horizon: sum(
            coordinate.horizon_epochs == horizon for coordinate in rejected_transfer
        )
        for horizon in APPROVED_PROTOCOL.horizon_epochs
    } == {15: 4, 25: 4, 40: 4}

    fixed_spec = next(
        spec for spec in APPROVED_FAMILY_SPECS if spec.id == "rq1_content_input"
    )
    bad = ReusableCoordinate(
        source_id="outside-bounds",
        embedding_learning_rate=10.0,
        deep_learning_rate=0.03,
        horizon_epochs=25,
    )
    with pytest.raises(ValueError, match="embedding learning rate"):
        compile_family(fixed_spec, reusable=(bad,))


def test_compiler_requires_explicit_opt_in_for_followup_reuse() -> None:
    spec = next(
        spec for spec in APPROVED_FAMILY_SPECS if spec.id == "rq3_output_learned"
    )
    followup = ReusableCoordinate(
        source_id="authenticated-followup",
        embedding_learning_rate=0.3041556165944196,
        deep_learning_rate=0.005733564587228046,
        horizon_epochs=60,
    )

    with pytest.raises(ValueError, match="deep learning rate"):
        compile_family(spec, reusable=(followup,))

    compiled = compile_family(
        spec,
        reusable=(followup,),
        allow_reusable_outside_search_space=True,
    )

    assert compiled[0].reused_from == "authenticated-followup"
    assert compiled[0].horizon_epochs == 60


def test_capacity_first_stage_needs_no_future_selection() -> None:
    spec = next(
        spec for spec in APPROVED_FAMILY_SPECS if spec.id == "rq2_content_concat"
    )

    first_stage = compile_capacity_first_stage(spec)
    assert {
        capacity: sum(coordinate.capacity == capacity for coordinate in first_stage)
        for capacity in spec.capacities
    } == {64: 3, 128: 3, 256: 3}
    anchor = [
        coordinate
        for coordinate in first_stage
        if coordinate.embedding_learning_rate
        == APPROVED_PROTOCOL.control.embedding_learning_rate
        and coordinate.deep_learning_rate == APPROVED_PROTOCOL.control.deep_learning_rate
    ]
    assert len(anchor) == 3
    assert {coordinate.capacity for coordinate in anchor} == {64, 128, 256}
    signatures = {
        capacity: {
            (
                coordinate.embedding_learning_rate,
                coordinate.deep_learning_rate,
            )
            for coordinate in first_stage
            if coordinate.capacity == capacity
        }
        for capacity in spec.capacities
    }
    assert signatures[64] == signatures[128] == signatures[256]
    assert all(coordinate.horizon_epochs == 25 for coordinate in first_stage)

    matched_control = next(
        value
        for value in APPROVED_FAMILY_SPECS
        if value.id == "rq2_id_only_densenet"
    )
    matched_signatures = {
        (
            coordinate.embedding_learning_rate,
            coordinate.deep_learning_rate,
        )
        for coordinate in compile_capacity_first_stage(matched_control)
    }
    assert matched_signatures == signatures[64]


def test_capacity_followup_uses_only_the_resolved_capacity() -> None:
    spec = next(
        spec for spec in APPROVED_FAMILY_SPECS if spec.id == "rq2_content_concat"
    )

    probes = compile_capacity_horizon_followup(
        spec,
        selected_capacity=256,
        transferred_horizon_rates=_transferred_horizon_rates(),
        first_stage=compile_capacity_first_stage(spec),
    )

    assert len(probes) == 3
    assert {coordinate.capacity for coordinate in probes} == {256}
    assert {coordinate.horizon_epochs for coordinate in probes} == {15, 25, 40}


def test_capacity_horizon_probes_use_explicit_rates_and_deduplicate_reuse() -> None:
    spec = next(
        spec for spec in APPROVED_FAMILY_SPECS if spec.id == "rq2_content_concat"
    )

    first_stage = compile_capacity_first_stage(spec)
    probes = compile_capacity_horizon_followup(
        spec,
        selected_capacity=128,
        transferred_horizon_rates=_transferred_horizon_rates(),
        first_stage=first_stage,
    )

    assert [
        (
            coordinate.horizon_epochs,
            coordinate.embedding_learning_rate,
            coordinate.deep_learning_rate,
        )
        for coordinate in probes
    ] == [
        (15, 0.10, 0.02),
        (
            25,
            APPROVED_PROTOCOL.control.embedding_learning_rate,
            APPROVED_PROTOCOL.control.deep_learning_rate,
        ),
        (40, 0.20, 0.04),
    ]
    assert all(coordinate.role == "horizon_probe" for coordinate in probes)
    assert probes[1].reused_from is not None
    compiled = (*first_stage, *probes)
    assert len(compiled) == 12
    assert len({coordinate.physical_id for coordinate in compiled}) == 11


def test_capacity_transfer_fails_closed_without_exact_horizon_rates() -> None:
    spec = next(
        spec for spec in APPROVED_FAMILY_SPECS if spec.id == "rq2_content_concat"
    )
    first_stage = compile_capacity_first_stage(spec)
    with pytest.raises(ValueError, match="exactly horizons"):
        compile_capacity_horizon_followup(
            spec,
            selected_capacity=128,
            transferred_horizon_rates=(),
            first_stage=first_stage,
        )
    with pytest.raises(ValueError, match="exactly horizons"):
        compile_capacity_horizon_followup(
            spec,
            selected_capacity=128,
            transferred_horizon_rates=_transferred_horizon_rates()[:2],
            first_stage=first_stage,
        )


def test_rejected_transfer_compiles_all_capacity_families_without_selections() -> None:
    compiled = compile_approved_search(
        selected_capacities={},
        transfer_accepted=False,
        rq4_extra_id_predecessor=_rq4_predecessor(transfer_accepted=False),
    )

    assert compiled.maximum_opportunity_count == 205
    for spec in APPROVED_FAMILY_SPECS:
        if not spec.capacities:
            continue
        coordinates = [
            coordinate
            for coordinate in compiled.initial
            if coordinate.family_id == spec.id
        ]
        assert {
            capacity: sum(coordinate.capacity == capacity for coordinate in coordinates)
            for capacity in spec.capacities
        } == {capacity: 4 for capacity in spec.capacities}
