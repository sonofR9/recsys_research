import torch
import pytest
from torch import nn

from dcn.config.generation import _initialize_standard_parameters
from dcn.data.features import FeatureValues
from dcn.nn import (
    ContentProjection,
    FrequencyContentGate,
    GlobalContentGate,
    ItemContentCatalogEncoder,
    ItemContentDenseNetEncoder,
    ItemMetadataDenseNetEncoder,
    ItemMetadataEmbedding,
    MeanPooledIdEmbedding,
    PrecomputedEmbeddingLookup,
    PretrainedCatalogEncoder,
)


def _content_lookup() -> PrecomputedEmbeddingLookup:
    return PrecomputedEmbeddingLookup(
        torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        learnable_default=False,
        strict=False,
    )


def _unnormalized_content_lookup() -> PrecomputedEmbeddingLookup:
    return PrecomputedEmbeddingLookup(
        torch.tensor(
            [
                [3.0, 4.0, 0.0],
                [0.0, 0.0, 7.0],
                [8.0, 6.0, 0.0],
            ]
        ),
        learnable_default=False,
        strict=False,
    )


def test_content_projection_reports_and_returns_requested_width() -> None:
    encoder = ContentProjection(_content_lookup(), output_dim=5)

    output = encoder(torch.tensor([1, 3]))

    assert encoder.out_dim == 5
    assert output.shape == (2, 5)


def test_content_projection_normalizes_before_projection_without_normalizing_output() -> (
    None
):
    encoder = ContentProjection(
        _unnormalized_content_lookup(), output_dim=3, normalize_content=True
    )
    with torch.no_grad():
        encoder.projection.weight.copy_(2 * torch.eye(3))

    output = encoder(torch.tensor([0, 1, 2, 99]))

    assert torch.equal(output[[0, 3]], torch.zeros(2, 3))
    torch.testing.assert_close(output[1:].norm(dim=-1)[:2], torch.full((2,), 2.0))


def test_content_projection_preserves_legacy_raw_lookup_by_default() -> None:
    encoder = ContentProjection(_unnormalized_content_lookup(), output_dim=3)
    with torch.no_grad():
        encoder.projection.weight.copy_(torch.eye(3))

    output = encoder(torch.tensor([1]))

    torch.testing.assert_close(output, torch.tensor([[3.0, 4.0, 0.0]]))


def test_item_content_densenet_reports_width_and_trains_learned_branch() -> None:
    encoder = ItemContentDenseNetEncoder(
        num_items=3,
        item_dim=2,
        content=_content_lookup(),
        output_dim=4,
        hidden_dim=6,
    )

    output = encoder(torch.tensor([1, 2, 3]))
    output.square().sum().backward()

    assert encoder.out_dim == 4
    assert output.shape == (3, 4)
    assert encoder.item_embedding.weight.grad is not None
    assert all(
        not parameter.requires_grad for parameter in encoder.content.parameters()
    )


def test_item_content_densenet_uses_an_injected_item_table_and_normalized_content() -> (
    None
):
    item_embedding = nn.Embedding(4, 2)
    encoder = ItemContentDenseNetEncoder(
        num_items=3,
        item_dim=2,
        content=_unnormalized_content_lookup(),
        output_dim=4,
        hidden_dim=6,
        item_embedding=item_embedding,
        normalize_content=True,
    )

    features = encoder.composed_features(torch.tensor([0, 1, 2, 3, 99]))
    encoder(torch.tensor([1, 2, 3])).sum().backward()

    assert encoder.item_embedding is item_embedding
    assert item_embedding.weight.grad is not None
    assert torch.equal(features[[0, 4], 2:], torch.zeros(2, 3))
    torch.testing.assert_close(features[1:4, 2:].norm(dim=-1), torch.ones(3))


def test_item_content_densenet_masks_zero_and_invalid_injected_item_ids() -> None:
    item_embedding = nn.Embedding(4, 2)
    with torch.no_grad():
        item_embedding.weight[0].fill_(17)
    encoder = ItemContentDenseNetEncoder(
        num_items=3,
        item_dim=2,
        content=_content_lookup(),
        output_dim=4,
        hidden_dim=6,
        item_embedding=item_embedding,
    )

    features = encoder.composed_features(torch.tensor([0, 99, 1]))
    features.sum().backward()

    assert torch.equal(features[:2, :2], torch.zeros(2, 2))
    assert item_embedding.weight.grad is not None
    assert torch.count_nonzero(item_embedding.weight.grad[0]) == 0
    assert torch.count_nonzero(item_embedding.weight.grad[1]) > 0


def test_item_content_densenet_rejects_an_incompatible_injected_table() -> None:
    with pytest.raises(ValueError, match="item embedding"):
        ItemContentDenseNetEncoder(
            num_items=3,
            item_dim=2,
            content=_content_lookup(),
            output_dim=4,
            hidden_dim=6,
            item_embedding=nn.Embedding(5, 2),
        )


def test_item_content_encoders_reject_misaligned_catalogs() -> None:
    with pytest.raises(ValueError, match="catalog"):
        ItemContentDenseNetEncoder(
            num_items=2,
            item_dim=2,
            content=_content_lookup(),
            output_dim=4,
            hidden_dim=6,
        )

    with pytest.raises(ValueError, match="catalog"):
        ItemContentCatalogEncoder(
            num_items=2,
            item_dim=2,
            content=_content_lookup(),
            output_dim=4,
            trainable_content=False,
        )


def test_pretrained_catalog_frozen_and_trainable_variants_start_equal() -> None:
    content = _content_lookup()
    frozen = PretrainedCatalogEncoder(content, output_dim=2, trainable=False)
    trainable = PretrainedCatalogEncoder(content, output_dim=2, trainable=True)
    item_ids = torch.tensor([0, 1, 2, 3])

    assert torch.equal(
        frozen.content_embeddings(item_ids),
        trainable.content_embeddings(item_ids),
    )
    assert frozen.out_dim == trainable.out_dim == 2


def test_trainable_pretrained_content_is_discoverable_as_an_embedding() -> None:
    frozen = PretrainedCatalogEncoder(_content_lookup(), output_dim=2, trainable=False)
    trainable = PretrainedCatalogEncoder(
        _content_lookup(), output_dim=2, trainable=True
    )

    assert not any(
        isinstance(module, nn.Embedding) for module in frozen.content.modules()
    )
    assert any(
        isinstance(module, nn.Embedding) for module in trainable.content.modules()
    )
    embedding_parameter_ids = {
        id(parameter)
        for module in trainable.content.modules()
        if isinstance(module, nn.Embedding)
        for parameter in module.parameters()
    }
    assert {
        id(parameter) for parameter in trainable.content_parameters()
    } == embedding_parameter_ids


def test_pretrained_catalog_content_gradient_follows_trainable_flag() -> None:
    frozen = PretrainedCatalogEncoder(_content_lookup(), output_dim=2, trainable=False)
    trainable = PretrainedCatalogEncoder(
        _content_lookup(), output_dim=2, trainable=True
    )
    item_ids = torch.tensor([1, 2, 3])

    frozen(item_ids).sum().backward()
    trainable(item_ids).sum().backward()

    assert list(frozen.content_parameters()) == []
    trainable_parameters = list(trainable.content_parameters())
    assert trainable_parameters
    assert all(parameter.grad is not None for parameter in trainable_parameters)


def test_frozen_and_trainable_catalog_content_is_normalized_on_every_lookup() -> None:
    content = _unnormalized_content_lookup()
    frozen = PretrainedCatalogEncoder(
        content, output_dim=3, trainable=False, normalize_content=True
    )
    trainable = PretrainedCatalogEncoder(
        content, output_dim=3, trainable=True, normalize_content=True
    )
    with torch.no_grad():
        assert trainable.content.embedding is not None
        trainable.content.embedding.weight[1:].mul_(5)
    item_ids = torch.tensor([0, 1, 2, 3, 99])

    frozen_content = frozen.content_embeddings(item_ids)
    trainable_content = trainable.content_embeddings(item_ids)
    trainable_content.sum().backward()

    assert torch.equal(frozen_content[[0, 4]], torch.zeros(2, 3))
    assert torch.equal(trainable_content[[0, 4]], torch.zeros(2, 3))
    torch.testing.assert_close(frozen_content[1:4].norm(dim=-1), torch.ones(3))
    torch.testing.assert_close(trainable_content[1:4].norm(dim=-1), torch.ones(3))
    assert trainable.content.embedding.weight.grad is not None


def test_trainable_pretrained_initialization_survives_declared_initialization() -> None:
    encoder = PretrainedCatalogEncoder(_content_lookup(), output_dim=2, trainable=True)
    item_ids = torch.tensor([0, 1, 2, 3])
    expected = encoder.content_embeddings(item_ids).clone()

    _initialize_standard_parameters(encoder, initializer_std=0.02)

    assert torch.equal(encoder.content_embeddings(item_ids), expected)


def test_item_content_catalog_concatenates_frozen_or_trainable_content() -> None:
    frozen = ItemContentCatalogEncoder(
        num_items=3,
        item_dim=2,
        content=_content_lookup(),
        output_dim=4,
        trainable_content=False,
    )
    trainable = ItemContentCatalogEncoder(
        num_items=3,
        item_dim=2,
        content=_content_lookup(),
        output_dim=4,
        trainable_content=True,
    )

    assert frozen(torch.tensor([1, 2])).shape == (2, 4)
    assert trainable(torch.tensor([1, 2])).shape == (2, 4)
    assert frozen.out_dim == trainable.out_dim == 4
    assert list(frozen.content_parameters()) == []
    assert list(trainable.content_parameters())


def test_item_content_catalog_uses_an_injected_item_table() -> None:
    item_embedding = nn.Embedding(4, 2)
    encoder = ItemContentCatalogEncoder(
        num_items=3,
        item_dim=2,
        content=_content_lookup(),
        output_dim=4,
        trainable_content=False,
        item_embedding=item_embedding,
    )

    encoder(torch.tensor([1, 2, 3])).sum().backward()

    assert encoder.item_embedding is item_embedding
    assert item_embedding.weight.grad is not None


def test_item_content_catalog_masks_zero_and_invalid_injected_item_ids() -> None:
    item_embedding = nn.Embedding(4, 2)
    with torch.no_grad():
        item_embedding.weight[0].fill_(19)
    encoder = ItemContentCatalogEncoder(
        num_items=3,
        item_dim=2,
        content=_content_lookup(),
        output_dim=2,
        trainable_content=False,
        item_embedding=item_embedding,
    )
    with torch.no_grad():
        encoder.projection.weight.zero_()
        encoder.projection.weight[:, :2].copy_(torch.eye(2))

    output = encoder(torch.tensor([0, 99, 1]))
    output.sum().backward()

    assert torch.equal(output[:2], torch.zeros(2, 2))
    assert item_embedding.weight.grad is not None
    assert torch.count_nonzero(item_embedding.weight.grad[0]) == 0
    assert torch.count_nonzero(item_embedding.weight.grad[1]) > 0


def test_metadata_embedding_mean_pools_known_values_and_ignores_unknown_zero() -> None:
    embedding = MeanPooledIdEmbedding(num_known_ids=3, embedding_dim=2)
    with torch.no_grad():
        embedding.embedding.weight.copy_(
            torch.tensor(
                [
                    [0.0, 0.0],
                    [2.0, 0.0],
                    [0.0, 4.0],
                    [2.0, 4.0],
                ]
            )
        )
    values = FeatureValues(
        values=torch.tensor([1, 2, 0, 3, 0, 99]),
        offsets=torch.tensor([0, 2, 4, 6]),
    )

    output = embedding(values)

    assert embedding.out_dim == 2
    assert torch.equal(
        output,
        torch.tensor([[1.0, 2.0], [2.0, 4.0], [0.0, 0.0]]),
    )


def _item_metadata_embedding() -> ItemMetadataEmbedding:
    embedding = ItemMetadataEmbedding(
        item_offsets=torch.tensor([0, 0, 2, 2, 4]),
        feature_ids=torch.tensor([1, 2, 0, 3]),
        num_known_feature_ids=3,
        embedding_dim=2,
    )
    with torch.no_grad():
        embedding.embedding.weight.copy_(
            torch.tensor(
                [
                    [0.0, 0.0],
                    [2.0, 0.0],
                    [0.0, 4.0],
                    [2.0, 4.0],
                ]
            )
        )
    return embedding


def test_item_metadata_embedding_maps_items_and_mean_pools_known_features() -> None:
    embedding = _item_metadata_embedding()

    output = embedding(torch.tensor([1, 2, 3, 0, 99]))

    assert embedding.out_dim == 2
    assert torch.equal(
        output,
        torch.tensor(
            [
                [1.0, 2.0],
                [0.0, 0.0],
                [2.0, 4.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        ),
    )


def test_item_metadata_embedding_preserves_arbitrary_item_id_shape() -> None:
    embedding = _item_metadata_embedding()
    shaped_ids = torch.tensor([[1, 2], [3, 0]])

    shaped = embedding(shaped_ids)
    flat = embedding(shaped_ids.flatten()).reshape(2, 2, -1)

    assert shaped.shape == (2, 2, 2)
    assert torch.equal(shaped, flat)


def test_item_metadata_embedding_gradients_and_embedding_discovery() -> None:
    embedding = _item_metadata_embedding()

    embedding(torch.tensor([1, 3])).sum().backward()

    assert isinstance(embedding.embedding, nn.Embedding)
    assert embedding.embedding.weight.grad is not None
    assert torch.count_nonzero(embedding.embedding.weight.grad[0]) == 0
    assert torch.count_nonzero(embedding.embedding.weight.grad[1:]) > 0
    assert {name for name, _ in embedding.named_buffers()} == {
        "item_offsets",
        "feature_ids",
    }


@pytest.mark.parametrize(
    ("offsets", "feature_ids", "error"),
    [
        ([1, 1, 1], [], "start"),
        ([0, 2, 1], [1], "nondecreasing"),
        ([0, 0, 2], [1], "final"),
        ([0, 1, 1], [1], "unknown item"),
        ([0, 0, 1], [4], "feature ID"),
    ],
)
def test_item_metadata_embedding_validates_the_compact_map(
    offsets: list[int], feature_ids: list[int], error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        ItemMetadataEmbedding(
            item_offsets=torch.tensor(offsets, dtype=torch.long),
            feature_ids=torch.tensor(feature_ids, dtype=torch.long),
            num_known_feature_ids=3,
            embedding_dim=2,
        )


def test_item_metadata_densenet_composes_existing_encoder_and_branches() -> None:
    item_encoder = nn.Embedding(4, 3, padding_idx=0)
    artist = _item_metadata_embedding()
    album = _item_metadata_embedding()
    encoder = ItemMetadataDenseNetEncoder(
        item_encoder=item_encoder,
        metadata_branches=[artist, album],
        output_dim=5,
        hidden_dim=7,
    )
    shaped_ids = torch.tensor([[1, 2], [3, 0]])

    shaped = encoder(shaped_ids)
    flat = encoder(shaped_ids.flatten()).reshape(2, 2, -1)
    shaped.sum().backward()

    assert encoder.out_dim == 5
    assert shaped.shape == (2, 2, 5)
    assert torch.allclose(shaped, flat)
    nested_embeddings = [
        module for module in encoder.modules() if isinstance(module, nn.Embedding)
    ]
    assert len(nested_embeddings) == 3
    assert all(module.weight.grad is not None for module in nested_embeddings)


def test_item_metadata_densenet_requires_a_metadata_branch() -> None:
    with pytest.raises(ValueError, match="metadata branch"):
        ItemMetadataDenseNetEncoder(
            item_encoder=nn.Embedding(4, 3),
            metadata_branches=[],
            output_dim=5,
            hidden_dim=7,
        )


def test_item_metadata_densenet_rejects_misaligned_item_encoder() -> None:
    with pytest.raises(ValueError, match="item catalog"):
        ItemMetadataDenseNetEncoder(
            item_encoder=nn.Embedding(5, 3),
            metadata_branches=[_item_metadata_embedding()],
            output_dim=5,
            hidden_dim=7,
        )


def test_global_content_gate_is_shared_and_trainable() -> None:
    gate = GlobalContentGate(initial_probability=0.75)

    output = gate(torch.tensor([[1, 2], [3, 0]]))
    output.sum().backward()

    assert gate.out_dim == 1
    assert torch.allclose(output, torch.full((2, 2, 1), 0.75))
    assert gate.logit.grad is not None


@pytest.mark.parametrize("probability", [0.0, 1.0, -0.1, 1.1, float("nan")])
def test_global_content_gate_requires_an_interior_probability(
    probability: float,
) -> None:
    with pytest.raises(ValueError, match="probability"):
        GlobalContentGate(initial_probability=probability)


def test_global_content_gate_remains_sigmoid_bounded() -> None:
    gate = GlobalContentGate()

    default = gate(torch.tensor([1]))

    gate.logit.data.fill_(-100)
    low = gate(torch.tensor([1]))
    gate.logit.data.fill_(100)
    high = gate(torch.tensor([1]))

    assert 0.99 < default.item() < 1
    assert 0 <= low.item() <= 1
    assert 0 <= high.item() <= 1


def test_frequency_gate_uses_only_the_registered_training_frequencies() -> None:
    gate = FrequencyContentGate(
        training_counts=torch.tensor([0.0, 1.0, 3.0, 15.0]),
        hidden_dim=4,
    )
    with torch.no_grad():
        gate.network[0].weight.fill_(1.0)
        gate.network[0].bias.zero_()
        gate.network[2].weight.fill_(1.0)
        gate.network[2].bias.zero_()

    output = gate(torch.tensor([1, 3, 0, 99]))

    assert gate.out_dim == 1
    assert output.shape == (4, 1)
    assert output[1] > output[0]
    assert torch.equal(output[2], output[3])
    assert not any(parameter.requires_grad for parameter in gate.buffers())


def test_frequency_gate_starts_at_fixed_control_and_survives_initialization() -> None:
    gate = FrequencyContentGate(
        training_counts=torch.tensor([0, 1, 4, 9]),
        hidden_dim=4,
    )
    item_ids = torch.tensor([0, 1, 2, 3])
    expected = torch.full((4, 1), 0.9999)

    torch.testing.assert_close(gate(item_ids), expected, atol=1e-6, rtol=0)
    _initialize_standard_parameters(gate, initializer_std=0.02)
    torch.testing.assert_close(gate(item_ids), expected, atol=1e-6, rtol=0)


def test_frequency_gate_fp32_math_keeps_nonzero_gradients_under_bfloat16_autocast() -> None:
    gate = FrequencyContentGate(
        training_counts=torch.tensor([0, 1, 4, 9]),
        hidden_dim=4,
        initial_probability=0.9,
        fp32_math=True,
    )

    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = gate(torch.tensor([1, 2, 3]))
        output.sum().backward()

    assert output.dtype == torch.float32
    torch.testing.assert_close(output, torch.full((3, 1), 0.9))
    assert gate.network[2].bias.grad is not None
    assert gate.network[2].bias.grad.abs().item() > 0


def test_item_content_encoder_applies_content_gate_before_composition() -> None:
    gate = GlobalContentGate(initial_probability=0.5)
    encoder = ItemContentDenseNetEncoder(
        num_items=3,
        item_dim=2,
        content=_content_lookup(),
        output_dim=4,
        hidden_dim=6,
        content_gate=gate,
    )

    gated = encoder.composed_features(torch.tensor([1, 2]))

    expected = 0.5 * _content_lookup().lookup(torch.tensor([1, 2]))
    assert torch.equal(gated[:, -3:], expected)
