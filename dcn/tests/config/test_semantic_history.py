from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from dcn.config import SemanticHistoryExperiment, SemanticIdConfig
from dcn.semantic import ResidualCodebooks, SemanticCodes


@pytest.fixture
def semantic_artifacts() -> tuple[SemanticCodes, ResidualCodebooks]:
    codes = SemanticCodes(
        item_ids=torch.tensor([1, 2, 3, 4]),
        codes=torch.tensor([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]]),
        codes_per_level=(2, 2, 2),
    )
    codebooks = ResidualCodebooks(torch.randn(2, 2, 8))
    return codes, codebooks


def _experiment(
    representation: str,
    artifacts: tuple[SemanticCodes, ResidualCodebooks],
) -> SemanticHistoryExperiment:
    codes, codebooks = artifacts
    experiment = SemanticHistoryExperiment(
        history_representation=representation,
        representation_width=32,
        semantic=SemanticIdConfig(num_levels=2, num_codes=2),
        transformer=replace(
            SemanticHistoryExperiment.transformer,
            dim=64,
            nhead=2,
            num_kv_heads=2,
        ),
    )
    experiment.__dict__["item_embeddings"] = SimpleNamespace(num_known_ids=4)
    experiment.__dict__["artifacts"] = SimpleNamespace(item_id_column="compact_item_id")
    experiment.__dict__["semantic_codes"] = codes
    experiment.__dict__["semantic_codebooks"] = codebooks
    return experiment


@pytest.mark.parametrize(
    ("representation", "tokens_per_event"),
    [
        ("learned_sid_event", 1),
        ("item_frozen_sid_event", 1),
        ("item_learned_frozen_sid_event", 1),
        ("item_frozen_sid_learned_residual_event", 1),
        ("learned_sid_tokens", 3),
        ("learned_frozen_sid_tokens", 3),
        ("frozen_sid_tokens", 3),
        ("interleaved_item_sid_tokens", 4),
    ],
)
def test_every_approved_representation_builds_the_expected_event_layout(
    semantic_artifacts: tuple[SemanticCodes, ResidualCodebooks],
    representation: str,
    tokens_per_event: int,
) -> None:
    experiment = _experiment(representation, semantic_artifacts)
    tokenizer = experiment.create_tokenizer()

    assert tokenizer.tokens_per_event == tokens_per_event
    assert tokenizer.out_dim == 64
    assert experiment.history_tokens_per_event == tokens_per_event


def test_true_metric_uses_only_rq_kmeans_levels(
    semantic_artifacts: tuple[SemanticCodes, ResidualCodebooks],
) -> None:
    experiment = _experiment("learned_sid_tokens", semantic_artifacts)

    assert experiment.true_metric_options() == {
        "semantic_codes": semantic_artifacts[0],
        "semantic_base_levels": 2,
    }


def test_semantic_lookups_and_projections_use_their_approved_optimizer_groups(
    semantic_artifacts: tuple[SemanticCodes, ResidualCodebooks],
) -> None:
    experiment = _experiment(
        "item_frozen_sid_learned_residual_event", semantic_artifacts
    )
    tokenizer = experiment.create_tokenizer()
    embedding_parameters, deep_parameters = experiment.split_parameters(
        tokenizer, experiment.embedding_types
    )
    group_by_parameter = {
        **{id(parameter): "embedding" for parameter in embedding_parameters},
        **{id(parameter): "deep" for parameter in deep_parameters},
    }

    lookup_parameters = [
        parameter
        for module in tokenizer.modules()
        if isinstance(module, nn.Embedding)
        for parameter in module.parameters(recurse=False)
    ]
    projection_parameters = [
        parameter
        for module in tokenizer.modules()
        if isinstance(module, nn.Linear)
        for parameter in module.parameters(recurse=False)
    ]

    assert lookup_parameters
    assert projection_parameters
    assert all(
        group_by_parameter[id(parameter)] == "embedding"
        for parameter in lookup_parameters
    )
    assert all(
        group_by_parameter[id(parameter)] == "deep"
        for parameter in projection_parameters
    )


def test_learned_sid_residual_uses_fixed_frozen_event_width(
    semantic_artifacts: tuple[SemanticCodes, ResidualCodebooks],
) -> None:
    experiment = _experiment(
        "item_frozen_sid_learned_residual_event", semantic_artifacts
    )
    experiment.frozen_event_width = 128
    tokenizer = experiment.create_tokenizer()

    assert tokenizer.encoder.input_projection.out_features == 128
    assert tokenizer.residual_semantic_embedding.level_dim == 32


def test_learned_sid_residual_passes_the_bounded_gate(
    semantic_artifacts: tuple[SemanticCodes, ResidualCodebooks],
) -> None:
    experiment = _experiment(
        "item_frozen_sid_learned_residual_event", semantic_artifacts
    )
    experiment.learned_residual_max_scale = 0.025

    tokenizer = experiment.create_tokenizer()

    assert tokenizer.event_residual.max_scale == 0.025
    assert experiment.generation_architecture_metadata() == {
        "history_representation": "item_frozen_sid_learned_residual_event",
        "representation_width": experiment.representation_width,
        "frozen_event_width": experiment.frozen_event_width,
        "learned_residual_max_scale": 0.025,
    }


def test_bounded_gate_is_rejected_for_other_representations() -> None:
    with pytest.raises(ValueError, match="bounded learned residual"):
        SemanticHistoryExperiment(
            history_representation="item_frozen_sid_event",
            learned_residual_max_scale=0.025,
        )


def test_unknown_representation_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown semantic history representation"):
        SemanticHistoryExperiment(history_representation="unknown")
