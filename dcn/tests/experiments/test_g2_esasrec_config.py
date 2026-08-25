import json
from dataclasses import replace

import pytest
import torch
from torch import nn

from dcn.config import GenerationExperiment
from dcn.nn.esasrec import LiGRBlock, SASRecBlock
from dcn.nn.sampled_softmax import GeneralizedBCELoss
from experiments.g2_esasrec.configs.local import (
    COMPONENT_METHODS,
    CONTROL_BATCHES,
    DEEP_LR_BOUNDS,
    EMBEDDING_LR_BOUNDS,
    GBCE_T_BOUNDS,
    LIGR_WIDTHS,
    MATCHED_STANDARD_WIDTHS,
    MIXED_UNIFORM_FRACTION_BOUNDS,
    LocalG2Experiment,
    build_component,
    build_control,
    build_mixed_sampler,
)


@pytest.fixture(autouse=True)
def _restore_sdp_backends():
    flash = torch.backends.cuda.flash_sdp_enabled()
    memory_efficient = torch.backends.cuda.mem_efficient_sdp_enabled()
    math_sdp = torch.backends.cuda.math_sdp_enabled()
    yield
    torch.backends.cuda.enable_flash_sdp(flash)
    torch.backends.cuda.enable_mem_efficient_sdp(memory_efficient)
    torch.backends.cuda.enable_math_sdp(math_sdp)


def test_approved_tuning_surfaces_are_public_configuration_invariants() -> None:
    assert CONTROL_BATCHES == (128, 256, 512, 1024, 1280)
    assert EMBEDDING_LR_BOUNDS == (1e-4, 0.256)
    assert DEEP_LR_BOUNDS == (1e-4, 0.128)
    assert GBCE_T_BOUNDS == (0.25, 1.0)
    assert MIXED_UNIFORM_FRACTION_BOUNDS == (0.2, 0.8)


def test_control_is_the_unchanged_g1_native_50m_structure() -> None:
    control = build_control(
        batch_size=1024,
        embedding_learning_rate=0.004,
        deep_learning_rate=0.006,
    )

    assert type(control) is GenerationExperiment
    assert control.size == "50m"
    assert control.user_sample is None
    assert control.event_type_filter == "like"
    assert control.min_item_interactions_per_item == 5
    assert control.drop_unmapped_items is True
    assert control.window == "next_item"
    assert control.validation_interval_seconds == 7 * 24 * 60 * 60
    assert control.evaluation_catalog == "all"
    assert control.exclude_seen_from_evaluation is False
    assert control.max_seq_len == 128
    assert control.timestamp_delta == "bins"
    assert control.timestamp_combination == "add"
    assert control.timestamp_num_bins == 16
    assert control.negative_sampling == "random"
    assert control.num_in_batch_negatives == 512
    assert control.dense_random_negative_scores is True
    assert control.num_epochs == 20
    assert control.lr_schedule.shape == "linear"
    assert control.lr_schedule_horizon_epochs == 20
    assert control.initializer_std == 0.02
    assert control.dataloader.batch_size == 1024
    assert control.embedding_learning_rate == 0.004
    assert control.deep_learning_rate == 0.006

    transformer = control.transformer
    assert transformer.dim == 64
    assert transformer.num_layers == 2
    assert transformer.nhead == 2
    assert transformer.num_kv_heads == 1
    assert transformer.ffn == "swiglu"
    assert transformer.ffn_intermediate_dim == 171
    assert transformer.norm == "layer"
    assert transformer.norm_place == "pre"
    assert transformer.input_norm == "rms"
    assert transformer.final_norm == "layer"
    assert transformer.learned_positions == "forward"
    assert transformer.attention_window == 50


def test_component_matrix_has_exactly_the_six_approved_methods() -> None:
    assert COMPONENT_METHODS == (
        "standard_sampled_softmax",
        "standard_gbce",
        "matched_standard_sampled_softmax",
        "matched_standard_gbce",
        "ligr_sampled_softmax",
        "ligr_gbce",
    )

    methods = {
        name: build_component(name, ligr_multiplier=4) for name in COMPONENT_METHODS
    }
    assert {
        name: (experiment.layer_family, experiment.loss_kind)
        for name, experiment in methods.items()
    } == {
        "standard_sampled_softmax": ("sasrec", "sampled_softmax"),
        "standard_gbce": ("sasrec", "gbce"),
        "matched_standard_sampled_softmax": ("sasrec", "sampled_softmax"),
        "matched_standard_gbce": ("sasrec", "gbce"),
        "ligr_sampled_softmax": ("ligr", "sampled_softmax"),
        "ligr_gbce": ("ligr", "gbce"),
    }
    assert methods["standard_sampled_softmax"].transformer.ffn_intermediate_dim == 256
    assert methods["standard_gbce"].transformer.ffn_intermediate_dim == 256
    assert (
        methods["matched_standard_sampled_softmax"].transformer.ffn_intermediate_dim
        == 1792
    )
    assert methods["matched_standard_gbce"].transformer.ffn_intermediate_dim == 1792
    assert methods["ligr_sampled_softmax"].transformer.ffn_intermediate_dim == 1024
    assert methods["ligr_gbce"].transformer.ffn_intermediate_dim == 1024


def test_component_builders_select_the_local_block_implementations() -> None:
    standard = build_component("standard_sampled_softmax").create_sequence_model(1)
    ligr = build_component("ligr_sampled_softmax").create_sequence_model(1)

    assert all(isinstance(block, SASRecBlock) for block in standard.layers)
    assert all(isinstance(block, LiGRBlock) for block in ligr.layers)


def test_components_shift_reverse_positions_past_the_terminal_training_target() -> None:
    experiment = build_component("ligr_sampled_softmax")

    assert experiment.training_reverse_position_offset == 1


def test_component_model_receives_the_rectools_training_position_offset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = nn.Linear(1, 1)
    monkeypatch.setattr(GenerationExperiment, "_create_model", lambda self: model)

    created = build_component("ligr_sampled_softmax")._create_model()

    assert created.training_reverse_position_offset == 1


def test_component_final_normalization_matches_each_rectools_family() -> None:
    standard_experiment = build_component("standard_sampled_softmax")
    matched_experiment = build_component("matched_standard_sampled_softmax")
    ligr_experiment = build_component("ligr_sampled_softmax")

    standard = standard_experiment.create_sequence_model(1)
    matched = matched_experiment.create_sequence_model(1)
    ligr = ligr_experiment.create_sequence_model(1)

    assert standard_experiment.transformer.final_norm == "layer"
    assert matched_experiment.transformer.final_norm == "layer"
    assert ligr_experiment.transformer.final_norm is None
    assert isinstance(standard.final_norm, nn.LayerNorm)
    assert isinstance(matched.final_norm, nn.LayerNorm)
    assert standard.final_norm.eps == 1e-8
    assert matched.final_norm.eps == 1e-8
    assert isinstance(ligr.final_norm, nn.Identity)
    assert isinstance(standard.input_norm, nn.Identity)
    assert isinstance(matched.input_norm, nn.Identity)
    assert isinstance(ligr.input_norm, nn.Identity)


@pytest.mark.parametrize("multiplier", [2, 4, 6])
def test_new_ligr_and_matched_widths_follow_the_approved_capacity_pairs(
    multiplier: int,
) -> None:
    ligr = build_component("ligr_sampled_softmax", ligr_multiplier=multiplier)
    matched = build_component(
        "matched_standard_sampled_softmax", ligr_multiplier=multiplier
    )

    assert ligr.transformer.ffn_intermediate_dim == LIGR_WIDTHS[multiplier]
    assert ligr.transformer.ffn_intermediate_dim % 32 == 0
    assert (
        matched.transformer.ffn_intermediate_dim == MATCHED_STANDARD_WIDTHS[multiplier]
    )
    assert matched.transformer.ffn_intermediate_dim % 32 == 0


@pytest.mark.parametrize("multiplier", [2, 4, 6])
def test_matched_standard_stack_is_within_two_percent_of_ligr_capacity(
    multiplier: int,
) -> None:
    ligr = build_component(
        "ligr_sampled_softmax", ligr_multiplier=multiplier
    ).create_sequence_model(tokens_per_event=1)
    matched = build_component(
        "matched_standard_sampled_softmax", ligr_multiplier=multiplier
    ).create_sequence_model(tokens_per_event=1)

    ligr_parameters = sum(parameter.numel() for parameter in ligr.parameters())
    matched_parameters = sum(parameter.numel() for parameter in matched.parameters())
    assert abs(matched_parameters - ligr_parameters) / ligr_parameters <= 0.02


def test_components_share_protocol_and_expose_family_tuning_parameters() -> None:
    sampled = build_component(
        "ligr_sampled_softmax",
        batch_size=512,
        embedding_learning_rate=0.003,
        deep_learning_rate=0.007,
        ligr_multiplier=6,
    )
    gbce = build_component(
        "ligr_gbce",
        batch_size=512,
        embedding_learning_rate=0.003,
        deep_learning_rate=0.007,
        ligr_multiplier=6,
        gbce_t=0.25,
    )

    for experiment in (sampled, gbce):
        assert isinstance(experiment, LocalG2Experiment)
        assert experiment.size == "50m"
        assert experiment.user_sample is None
        assert experiment.window == "next_item"
        assert experiment.max_seq_len == 100
        assert experiment.transformer.dim == 256
        assert experiment.transformer.num_layers == 2
        assert experiment.transformer.nhead == 4
        assert experiment.transformer.num_kv_heads == 4
        assert experiment.transformer.dropout == 0.2
        assert experiment.num_in_batch_negatives == 256
        assert experiment.negative_sampling == "random"
        assert experiment.logq_correction == "none"
        assert experiment.dataloader.batch_size == 512
        assert experiment.embedding_learning_rate == 0.003
        assert experiment.deep_learning_rate == 0.007
        assert experiment.initializer_std == 0.02
        assert experiment.num_epochs == 100
        assert experiment.early_stopping_patience == 10
        assert experiment.lr_schedule.shape == "constant"
        assert experiment.restore_best_weights is True

    assert sampled.gbce_t is None
    assert gbce.gbce_t == 0.25


@pytest.mark.parametrize("method", ["standard_gbce", "ligr_gbce"])
@pytest.mark.parametrize("t", [0.25, 0.75, 1.0])
def test_gbce_treatment_exposes_the_approved_range(method: str, t: float) -> None:
    assert build_component(method, gbce_t=t).gbce_t == t


@pytest.mark.parametrize("t", [0.0, 1.01])
def test_gbce_treatment_rejects_values_outside_the_approved_range(t: float) -> None:
    with pytest.raises(ValueError, match="gbce_t must be in"):
        build_component("ligr_gbce", gbce_t=t)


def test_sampled_softmax_rejects_a_gbce_treatment_value() -> None:
    with pytest.raises(ValueError, match="only gBCE methods accept gbce_t"):
        build_component("ligr_sampled_softmax", gbce_t=0.75)


@pytest.mark.parametrize("logq_correction", ["none", "yi2019"])
def test_mixed_sampler_exposes_fraction_and_logq_without_changing_the_model(
    logq_correction: str,
) -> None:
    experiment = build_mixed_sampler(
        uniform_fraction=0.6,
        logq_correction=logq_correction,
        batch_size=256,
        embedding_learning_rate=0.005,
        deep_learning_rate=0.009,
        ligr_multiplier=2,
    )

    assert experiment.layer_family == "ligr"
    assert experiment.loss_kind == "sampled_softmax"
    assert experiment.transformer.ffn_intermediate_dim == 512
    assert experiment.negative_sampling == "mixed_online_global_q"
    assert experiment.num_in_batch_negatives == 256
    assert experiment.random_negative_fraction == 0.6
    assert experiment.logq_correction == logq_correction
    assert experiment.dataloader.batch_size == 256
    assert experiment.embedding_learning_rate == 0.005
    assert experiment.deep_learning_rate == 0.009


@pytest.mark.parametrize("fraction", [0.19, 0.81])
def test_mixed_sampler_rejects_fractions_outside_the_approved_range(
    fraction: float,
) -> None:
    with pytest.raises(ValueError, match="uniform_fraction must be in"):
        build_mixed_sampler(uniform_fraction=fraction, logq_correction="none")


def test_g2_component_initialization_keeps_norm_gains_at_one(base_path) -> None:
    base = build_component("ligr_sampled_softmax")
    experiment = replace(
        base,
        base_path=base_path,
        min_item_interactions_per_item=1,
        max_seq_len=4,
        min_seq_len=2,
        dataloader=replace(
            base.dataloader,
            batch_size=2,
            val_batch_size=2,
            num_workers=0,
            prefetch_factor=None,
        ),
    )
    experiment.setup()
    model = experiment.base_model

    norm_weights = [
        module.weight
        for module in model.modules()
        if isinstance(module, (nn.LayerNorm, nn.RMSNorm))
    ]
    assert norm_weights
    assert all(weight.eq(1.0).all() for weight in norm_weights)
    assert model.item_embedding.weight.abs().max().item() <= 0.04


def test_gbce_component_wires_the_local_loss_and_uniform_negatives(base_path) -> None:
    base = build_component("ligr_gbce", gbce_t=0.75)
    experiment = replace(
        base,
        base_path=base_path,
        min_item_interactions_per_item=1,
        max_seq_len=4,
        min_seq_len=2,
        dataloader=replace(
            base.dataloader,
            batch_size=2,
            val_batch_size=2,
            num_workers=0,
            prefetch_factor=None,
        ),
    )
    experiment.setup()
    criterion = experiment.create_criterion()

    assert isinstance(criterion.loss, GeneralizedBCELoss)
    assert criterion.loss.t == 0.75
    assert criterion.loss.catalog_size == experiment.num_items
    assert criterion.loss.random_negatives.catalog_size == experiment.catalog_size
    assert criterion.loss.num_in_batch_negatives == 0
    assert criterion.loss.random_negatives.num_negatives == 256
    assert criterion.loss.random_negatives.probabilities is None


def test_local_component_records_the_executed_recipe(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiment = replace(
        build_component("ligr_gbce", gbce_t=0.75),
        base_path=tmp_path,
    )
    destination = tmp_path / "logs" / experiment.run_name / "training_metadata.json"

    def write_base_metadata(self, runner) -> None:
        destination.parent.mkdir(parents=True)
        destination.write_text(json.dumps({"dataset_size": "50m"}))

    monkeypatch.setattr(
        GenerationExperiment,
        "_report_training_metadata",
        write_base_metadata,
    )

    experiment._report_training_metadata(object())

    assert json.loads(destination.read_text())["g2_recipe"] == {
        "layer_family": "ligr",
        "loss_kind": "gbce",
        "gbce_t": 0.75,
    }
