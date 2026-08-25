import pytest
import torch

from dcn.nn import GEGLU, ReGLU, RegularMLP, SwiGLU


@pytest.mark.parametrize(
    "ffn",
    [
        RegularMLP(2, 3, dropout=1.0, activation="relu"),
        RegularMLP(2, 3, dropout=1.0, activation="gelu"),
        RegularMLP(2, 3, dropout=1.0, activation="silu"),
        ReGLU(2, 3, dropout=1.0),
        GEGLU(2, 3, dropout=1.0),
        SwiGLU(2, 3, dropout=1.0),
    ],
)
def test_every_ffn_applies_dropout_before_the_output_projection(ffn) -> None:
    ffn.train()
    torch.nn.init.ones_(ffn.output_projection.bias)

    assert torch.equal(ffn(torch.ones(4, 2)), torch.ones(4, 2))


@pytest.mark.parametrize(
    ("activation", "expected"),
    [
        ("relu", torch.tensor([[0.0, 2.0]])),
        ("gelu", torch.nn.functional.gelu(torch.tensor([[-1.0, 2.0]]))),
        ("silu", torch.nn.functional.silu(torch.tensor([[-1.0, 2.0]]))),
    ],
)
def test_regular_mlp_uses_the_selected_activation(activation, expected) -> None:
    ffn = RegularMLP(2, 2, activation=activation)
    with torch.no_grad():
        ffn.input_projection.weight.copy_(torch.eye(2))
        ffn.input_projection.bias.zero_()
        ffn.output_projection.weight.copy_(torch.eye(2))
        ffn.output_projection.bias.zero_()

    assert torch.allclose(ffn(torch.tensor([[-1.0, 2.0]])), expected)


@pytest.mark.parametrize(
    ("ffn_type", "activation"),
    [
        (ReGLU, torch.nn.functional.relu),
        (GEGLU, torch.nn.functional.gelu),
        (SwiGLU, torch.nn.functional.silu),
    ],
)
def test_gated_ffn_multiplies_activated_gate_and_value(ffn_type, activation) -> None:
    ffn = ffn_type(2, 2)
    with torch.no_grad():
        ffn.gate_projection.weight.copy_(torch.eye(2))
        ffn.gate_projection.bias.zero_()
        ffn.value_projection.weight.copy_(2 * torch.eye(2))
        ffn.value_projection.bias.zero_()
        ffn.output_projection.weight.copy_(torch.eye(2))
        ffn.output_projection.bias.zero_()
    inputs = torch.tensor([[-1.0, 2.0]])

    assert torch.allclose(ffn(inputs), activation(inputs) * (2 * inputs))


def test_swiglu_config_loader_requires_explicit_gated_dropout_opt_in() -> None:
    legacy = SwiGLU.from_config({"dim": 2, "intermediate_dim": 3, "dropout": 0.4})
    matched = SwiGLU.from_config(
        {
            "dim": 2,
            "intermediate_dim": 3,
            "dropout": 0.4,
            "gated_ffn_dropout": True,
        }
    )

    assert legacy.dropout.p == 0.0
    assert matched.dropout.p == 0.4
