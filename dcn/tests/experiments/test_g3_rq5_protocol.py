from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.g3_pretrained_item_embeddings.launchers.rq5 import (
    build_rq5_training_experiment,
)
from experiments.g3_pretrained_item_embeddings.protocol.rq5 import (
    RQ5_COMPARATOR_FAMILY_IDS,
    FrozenRq2ContentBinding,
    compile_rq5_gate_surface,
    complete_rq5_frequency_surface,
)
from experiments.g3_pretrained_item_embeddings.protocol.search import (
    TransferredHorizonRate,
)
import experiments.g3_pretrained_item_embeddings.protocol.rq3 as rq3_protocol


def _authenticated_predecessor() -> SimpleNamespace:
    source = SimpleNamespace(
        source_id="rq2_content_concat:07",
        run_name="g3_rq2_content_concat_width_64_trial_07_native50m",
        history_hidden_dim=64,
        embedding_learning_rate=0.12,
        deep_learning_rate=0.03,
        horizon_epochs=25,
    )
    reused = SimpleNamespace(
        id="rq3_output_learned:01",
        family_id="rq3_output_learned",
        run_name=source.run_name,
        batch_size=512,
        seed=42,
        embedding_learning_rate=source.embedding_learning_rate,
        deep_learning_rate=source.deep_learning_rate,
        horizon_epochs=source.horizon_epochs,
        history_hidden_dim=source.history_hidden_dim,
        reused_from=source.source_id,
        authenticated_source=source,
    )
    return SimpleNamespace(
        selection_path="evidence/rq2_content_selection.json",
        selection_sha256="a" * 64,
        selected_history_hidden_dim=64,
        feature_manifest_path="protocol/artifacts/native50m_features.json",
        feature_manifest_sha256="b" * 64,
        feature_manifest_file_sha256="c" * 64,
        feature_data_path="generated/native50m_features.parquet",
        feature_data_sha256="d" * 64,
        frequency_terciles={"num_catalog_items": 3},
        training_count_reference={"sha256": "e" * 64},
        slice_membership_reference={"sha256": "f" * 64},
        rows_by_family={"rq3_output_learned": (reused,)},
    )


def _binding() -> FrozenRq2ContentBinding:
    return FrozenRq2ContentBinding(
        selection_path="evidence/rq2_content_selection.json",
        selection_sha256="a" * 64,
        selected_source_id="rq2_content_concat:07",
    )


def _horizon_rates() -> tuple[TransferredHorizonRate, ...]:
    return (
        TransferredHorizonRate(15, 0.10, 0.02),
        TransferredHorizonRate(25, 0.12, 0.03),
        TransferredHorizonRate(40, 0.14, 0.04),
    )


def _rates_with_reuse(surface) -> tuple[TransferredHorizonRate, ...]:
    source = next(
        row
        for row in surface.frequency_gate_rows
        if row.gate_hidden_dim == 8 and row.horizon_epochs == 25
    )
    return (
        TransferredHorizonRate(15, 0.10, 0.02),
        TransferredHorizonRate(
            25,
            source.embedding_learning_rate,
            source.deep_learning_rate,
        ),
        TransferredHorizonRate(40, 0.14, 0.04),
    )


def _patch_launch_reauthentication(monkeypatch, predecessor) -> Path:
    feature_path = Path("/tmp/test-rq5-features.parquet")
    monkeypatch.setattr(
        rq3_protocol,
        "compile_rq3_output_surface",
        lambda **_: predecessor,
    )
    monkeypatch.setattr(
        rq3_protocol,
        "resolve_rq3_feature_data",
        lambda **_: feature_path,
    )
    return feature_path


def test_rq5_surface_binds_fixed_gate_and_prepares_equal_gate_budgets() -> None:
    surface = compile_rq5_gate_surface(
        predecessor=_authenticated_predecessor(),
        binding=_binding(),
    )

    assert surface.comparator_family_ids == RQ5_COMPARATOR_FAMILY_IDS
    assert surface.fixed_gate.reused_from == "rq2_content_concat:07"
    assert surface.fixed_gate.content_gate == "fixed"
    assert surface.fixed_gate.history_hidden_dim == 64
    assert len(surface.global_gate_rows) == 12
    assert {row.content_gate for row in surface.global_gate_rows} == {"global"}
    assert {row.gate_hidden_dim for row in surface.global_gate_rows} == {None}
    assert len(surface.frequency_gate_rows) == 9
    assert {row.gate_hidden_dim for row in surface.frequency_gate_rows} == {4, 8, 16}
    assert all(row.horizon_epochs == 25 for row in surface.frequency_gate_rows)
    assert surface.training_count_reference == {"sha256": "e" * 64}
    assert surface.slice_membership_reference == {"sha256": "f" * 64}

    completed = complete_rq5_frequency_surface(
        surface,
        selected_gate_hidden_dim=8,
        transferred_horizon_rates=_horizon_rates(),
    )

    assert completed.selected_frequency_gate_hidden_dim == 8
    assert len(completed.global_gate_rows) == len(completed.frequency_gate_rows) == 12
    assert [row.horizon_epochs for row in completed.frequency_gate_rows[-3:]] == [
        15,
        25,
        40,
    ]
    assert {row.gate_hidden_dim for row in completed.frequency_gate_rows[-3:]} == {8}
    assert len(completed.new_training_rows) == 24


@pytest.mark.parametrize(
    ("binding", "message"),
    [
        (
            FrozenRq2ContentBinding(
                "evidence/rq2_content_selection.json",
                "0" * 64,
                "rq2_content_concat:07",
            ),
            "selection binding",
        ),
        (
            FrozenRq2ContentBinding(
                "evidence/rq2_content_selection.json",
                "a" * 64,
                "rq2_content_concat:missing",
            ),
            "selected fixed-gate source",
        ),
    ],
)
def test_rq5_surface_rejects_an_unbound_or_missing_fixed_gate(
    binding: FrozenRq2ContentBinding,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        compile_rq5_gate_surface(
            predecessor=_authenticated_predecessor(),
            binding=binding,
        )


def test_rq5_surface_rejects_a_tampered_authenticated_coordinate() -> None:
    predecessor = _authenticated_predecessor()
    predecessor.rows_by_family["rq3_output_learned"][0].deep_learning_rate = 0.04

    with pytest.raises(ValueError, match="authenticated coordinate"):
        compile_rq5_gate_surface(predecessor=predecessor, binding=_binding())


@pytest.mark.parametrize(
    "mutation",
    [
        "global_run_name",
        "global_lr",
        "frequency_capacity",
        "fixed_reuse",
        "feature_identity",
    ],
)
def test_rq5_completion_reconstructs_the_exact_initial_surface(mutation: str) -> None:
    surface = compile_rq5_gate_surface(
        predecessor=_authenticated_predecessor(),
        binding=_binding(),
    )
    if mutation == "global_run_name":
        rows = list(surface.global_gate_rows)
        rows[0] = replace(rows[0], run_name="tampered")
        surface = replace(surface, global_gate_rows=tuple(rows))
    elif mutation == "global_lr":
        rows = list(surface.global_gate_rows)
        rows[0] = replace(rows[0], embedding_learning_rate=0.2)
        surface = replace(surface, global_gate_rows=tuple(rows))
    elif mutation == "frequency_capacity":
        rows = list(surface.frequency_gate_rows)
        rows[0] = replace(rows[0], gate_hidden_dim=8)
        surface = replace(surface, frequency_gate_rows=tuple(rows))
    elif mutation == "fixed_reuse":
        surface = replace(
            surface,
            fixed_gate=replace(surface.fixed_gate, reused_from="rq2_content_concat:08"),
        )
    else:
        surface = replace(surface, feature_data_sha256="0" * 64)

    with pytest.raises(ValueError, match="initial gate surface changed"):
        complete_rq5_frequency_surface(
            surface,
            selected_gate_hidden_dim=8,
            transferred_horizon_rates=_horizon_rates(),
        )


def test_rq5_launcher_builds_only_new_bound_gate_rows(monkeypatch) -> None:
    predecessor = _authenticated_predecessor()
    initial = compile_rq5_gate_surface(
        predecessor=predecessor,
        binding=_binding(),
    )
    surface = complete_rq5_frequency_surface(
        initial,
        selected_gate_hidden_dim=8,
        transferred_horizon_rates=_horizon_rates(),
    )
    feature_path = _patch_launch_reauthentication(monkeypatch, predecessor)

    global_row = surface.global_gate_rows[0]
    global_experiment = build_rq5_training_experiment(
        surface,
        global_row,
        root=Path("/tmp"),
    )
    frequency_row = surface.frequency_gate_rows[-1]
    frequency_experiment = build_rq5_training_experiment(
        surface,
        frequency_row,
        root=Path("/tmp"),
    )

    assert global_experiment.representation.to_dict() == {
        "history_representation": "id_content",
        "catalog_representation": "learned_id",
        "history_hidden_dim": 64,
        "content_gate": "global",
        "gate_hidden_dim": None,
        "metadata": [],
        "metadata_dim": None,
        "extra_item_id_dim": None,
    }
    assert frequency_experiment.representation.content_gate == "frequency"
    assert frequency_experiment.representation.gate_hidden_dim == 8
    assert frequency_experiment.feature_data_path == feature_path

    with pytest.raises(ValueError, match="must not launch"):
        build_rq5_training_experiment(
            surface,
            surface.fixed_gate,
            root=Path("/tmp"),
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "global_run_name",
        "global_lr",
        "selected_capacity",
        "selected_source_id",
        "feature_identity",
        "followup_lr",
        "followup_reuse",
    ],
)
def test_rq5_launcher_reauthenticates_the_exact_completed_surface(
    monkeypatch, mutation: str
) -> None:
    predecessor = _authenticated_predecessor()
    initial = compile_rq5_gate_surface(predecessor=predecessor, binding=_binding())
    surface = complete_rq5_frequency_surface(
        initial,
        selected_gate_hidden_dim=8,
        transferred_horizon_rates=_rates_with_reuse(initial),
    )
    _patch_launch_reauthentication(monkeypatch, predecessor)
    if mutation.startswith("global"):
        rows = list(surface.global_gate_rows)
        rows[0] = replace(
            rows[0],
            **(
                {"run_name": "tampered"}
                if mutation == "global_run_name"
                else {"deep_learning_rate": rows[0].deep_learning_rate * 1.1}
            ),
        )
        surface = replace(surface, global_gate_rows=tuple(rows))
        row = rows[0]
    elif mutation == "selected_capacity":
        surface = replace(surface, selected_frequency_gate_hidden_dim=16)
        row = surface.frequency_gate_rows[0]
    elif mutation == "selected_source_id":
        surface = replace(
            surface,
            binding=replace(
                surface.binding,
                selected_source_id="rq2_content_concat:missing",
            ),
        )
        row = surface.global_gate_rows[0]
    elif mutation == "feature_identity":
        surface = replace(surface, training_count_reference={"sha256": "0" * 64})
        row = surface.global_gate_rows[0]
    else:
        rows = list(surface.frequency_gate_rows)
        index = next(
            index
            for index, candidate in enumerate(rows[9:], start=9)
            if candidate.reused_from is not None
        )
        rows[index] = replace(
            rows[index],
            **(
                {"embedding_learning_rate": rows[index].embedding_learning_rate * 1.1}
                if mutation == "followup_lr"
                else {"reused_from": None}
            ),
        )
        surface = replace(surface, frequency_gate_rows=tuple(rows))
        row = rows[index]

    with pytest.raises(ValueError, match="gate surface changed"):
        build_rq5_training_experiment(surface, row, root=Path("/tmp"))
