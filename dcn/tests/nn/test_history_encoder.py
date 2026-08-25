import pytest
import torch
from torch import nn

from dcn.nn.history_encoder import HistoryEncoder
from dcn.nn.resnet import ResNet1D
from dcn.nn.transformer import TransformerBlock, TransformerEncoder

DIM = 8
FEATURE_DIM = 5


pytestmark = pytest.mark.usefixtures("cpu_attention")


def _encoder() -> HistoryEncoder:
    torch.manual_seed(0)
    return HistoryEncoder(
        projection=ResNet1D(
            input_dim=FEATURE_DIM, hidden_dims=[DIM], norm_factory=nn.LayerNorm
        ),
        sequence_model=TransformerEncoder(
            blocks=[
                TransformerBlock(
                    dim=DIM,
                    nhead=2,
                    num_kv_heads=1,
                    ffn_factory=lambda dim: nn.Linear(dim, dim),
                    dropout=0.0,
                    use_alibi=False,
                )
            ]
        ),
    )


def test_emits_one_vector_per_token() -> None:
    encoder = _encoder().eval()
    features = torch.randn(5, FEATURE_DIM)
    cumulative_lens = torch.tensor([0, 3, 5])

    output = encoder(features, cumulative_lens)

    assert output.shape == (5, encoder.out_dim)


def test_a_token_never_sees_a_later_one() -> None:
    encoder = _encoder().eval()
    features = torch.randn(4, FEATURE_DIM)
    cumulative_lens = torch.tensor([0, 4])

    baseline = encoder(features, cumulative_lens)
    perturbed_input = features.clone()
    perturbed_input[3] += 10.0
    perturbed = encoder(perturbed_input, cumulative_lens)

    torch.testing.assert_close(baseline[:3], perturbed[:3])
    assert not torch.allclose(baseline[3], perturbed[3])


def test_sequences_in_one_batch_stay_independent() -> None:
    encoder = _encoder().eval()
    first = torch.randn(3, FEATURE_DIM)
    second = torch.randn(2, FEATURE_DIM)

    alone = encoder(first, torch.tensor([0, 3]))
    together = encoder(torch.cat([first, second]), torch.tensor([0, 3, 5]))

    torch.testing.assert_close(alone, together[:3])


def test_projection_and_sequence_model_widths_must_agree() -> None:
    with pytest.raises(AssertionError, match="dim"):
        HistoryEncoder(
            projection=ResNet1D(
                input_dim=FEATURE_DIM, hidden_dims=[DIM * 2], norm_factory=nn.LayerNorm
            ),
            sequence_model=TransformerEncoder(
                blocks=[
                    TransformerBlock(
                        dim=DIM,
                        nhead=2,
                        num_kv_heads=1,
                        ffn_factory=lambda dim: nn.Linear(dim, dim),
                        dropout=0.0,
                        use_alibi=False,
                    )
                ]
            ),
        )
