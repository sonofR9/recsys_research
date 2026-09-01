import torch
from torch import nn

from dcn.nn import DenseNet


def test_hidden_width_is_independent_of_input_and_output_widths() -> None:
    network = DenseNet(input_dim=12, output_dim=8, hidden_dim=32)

    output = network(torch.randn(4, 12))

    assert output.shape == (4, 8)
    assert network.input_projection.out_features == 32
    assert network.output_projection.in_features == 32
    assert isinstance(network.activation, nn.GELU)
