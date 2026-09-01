import pytest
import torch
from torch import nn

from dcn.nn.layer_item_features import (
    ConcatenatedItemFeatureResidual,
    DirectAddItemFeature,
    GemmaItemFeatureResidual,
)
from dcn.nn.transformer import TransformerEncoder
from dcn.nn.types import ModuleWithDim
from dcn.models.history_tokens import ItemTokenizer
from dcn.models.sequence_retrieval import SequenceRetrievalModel
from dcn.tests.helpers import ITEM_COLUMN, packed_batch, packed_lens


class _DoubleBlock(ModuleWithDim):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self._out_dim = dim

    @property
    def out_dim(self) -> int:
        return self._out_dim

    def forward(
        self,
        hidden: torch.Tensor,
        cumulative_lens: torch.Tensor,
        position_values: torch.Tensor | None = None,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        return 2 * hidden


@pytest.mark.parametrize(
    "fusion",
    [
        ConcatenatedItemFeatureResidual(8, 4),
        GemmaItemFeatureResidual(8, 4),
    ],
    ids=["concat_residual", "gemma_ple"],
)
def test_zero_start_fusions_preserve_the_control(fusion: nn.Module) -> None:
    hidden = torch.randn(5, 8)
    original = torch.randn(5, 8)
    features = torch.randn(5, 4)

    assert torch.equal(fusion(hidden, original, features), hidden)


def test_bounded_concat_residual_limits_its_effective_gate() -> None:
    fusion = ConcatenatedItemFeatureResidual(8, 4, max_scale=0.025)

    assert fusion.effective_residual_scale().item() == 0

    fusion.residual_scale.data.fill_(100)

    assert fusion.effective_residual_scale().item() == pytest.approx(0.025)


@pytest.mark.parametrize(
    "fusion",
    [
        DirectAddItemFeature(8),
        ConcatenatedItemFeatureResidual(8, 4),
        GemmaItemFeatureResidual(8, 4),
    ],
    ids=["direct_add", "concat_residual", "gemma_ple"],
)
def test_item_feature_fusions_preserve_shape_and_train(fusion: nn.Module) -> None:
    if hasattr(fusion, "residual_scale"):
        fusion.residual_scale.data.fill_(1)
    hidden = torch.randn(5, 8, requires_grad=True)
    original = torch.randn(5, 8, requires_grad=True)
    features = torch.randn(5, fusion.feature_dim, requires_grad=True)

    output = fusion(hidden, original, features)
    output.square().sum().backward()

    assert output.shape == hidden.shape
    assert hidden.grad is not None
    assert features.grad is not None
    assert all(parameter.grad is not None for parameter in fusion.parameters())


def test_masked_rows_never_receive_item_features() -> None:
    fusion = ConcatenatedItemFeatureResidual(8, 4)
    fusion.residual_scale.data.fill_(1)
    hidden = torch.randn(3, 8)
    original = torch.randn(3, 8)
    features = torch.randn(3, 4)
    active = torch.tensor([True, False, True])

    output = fusion(hidden, original, features, active)

    assert torch.equal(output[~active], hidden[~active])
    assert not torch.equal(output[active], hidden[active])


def test_matching_layer_feature_is_applied_at_each_depth() -> None:
    encoder = TransformerEncoder(
        blocks=[_DoubleBlock(8), _DoubleBlock(8)],
        input_norm=nn.Identity(),
    )
    hidden = torch.zeros(3, 8)
    first = torch.ones(3, 8)
    second = torch.full((3, 8), 3.0)

    output = encoder(
        hidden,
        packed_lens([3]),
        layer_item_features=[first, second],
        layer_item_feature_fusions=[
            DirectAddItemFeature(8),
            DirectAddItemFeature(8),
        ],
    )

    assert torch.equal(output, torch.full_like(hidden, 10.0))


def test_gemma_feature_is_applied_after_its_block() -> None:
    fusion = GemmaItemFeatureResidual(8, 4)
    fusion.residual_scale.data.fill_(1)
    encoder = TransformerEncoder(
        blocks=[_DoubleBlock(8)],
        input_norm=nn.Identity(),
    )
    hidden = torch.randn(3, 8)
    features = torch.randn(3, 4)

    output = encoder(
        hidden,
        packed_lens([3]),
        layer_item_features=[features],
        layer_item_feature_fusions=[fusion],
    )
    expected = fusion(2 * hidden, hidden, features)

    assert torch.allclose(output, expected)


def test_sequence_model_masks_queries_and_maps_each_layer_lookup() -> None:
    item_embedding = nn.Embedding(16, 8)
    layer_embeddings = [nn.Embedding(16, 8) for _ in range(2)]
    item_embedding.weight.data.zero_()
    for item_id in range(16):
        layer_embeddings[0].weight.data[item_id].fill_(item_id)
        layer_embeddings[1].weight.data[item_id].fill_(10 * item_id)
    layer_embeddings[0].weight.data[0].fill_(1_000)
    layer_embeddings[1].weight.data[0].fill_(2_000)
    model = SequenceRetrievalModel(
        tokenizer=ItemTokenizer(item_embedding, item_id_column=ITEM_COLUMN),
        sequence_model=TransformerEncoder(
            blocks=[_DoubleBlock(8), _DoubleBlock(8)],
            final_norm=nn.Identity(),
            input_norm=nn.Identity(),
        ),
        item_embedding=item_embedding,
        item_id_column=ITEM_COLUMN,
        layer_item_embeddings=layer_embeddings,
        layer_item_feature_fusions=[
            DirectAddItemFeature(8),
            DirectAddItemFeature(8),
        ],
        cls_token_mode="end_only",
    )
    model.end_of_history_query.embedding.data.fill_(1)

    output = model.encode_queries(packed_batch([1, 2, 3], [2, 1]))

    expected_values = torch.tensor([24, 48, 4, 72, 4], dtype=output.dtype)
    assert torch.equal(output, expected_values.unsqueeze(1).expand(-1, 8))
