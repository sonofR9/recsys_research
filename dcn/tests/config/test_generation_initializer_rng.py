from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import mup
import pytest
import torch
from torch import nn

from dcn.config.generation import (
    _initialize_mup_parameters,
    _initialize_standard_parameters,
)
from dcn.config.networks import build_causal_transformer
from dcn.config.settings import TRANSFORMER
from dcn.tests.helpers import packed_lens


pytestmark = pytest.mark.usefixtures("cpu_attention")


class _PositionModel(nn.Module):
    def __init__(self, dim: int, fusion: str, bounded: bool) -> None:
        super().__init__()
        transformer = replace(
            TRANSFORMER,
            dim=dim,
            num_layers=1,
            nhead=2,
            num_kv_heads=1,
            ffn_intermediate_dim=2 * dim,
            dropout=0.0,
            input_dropout=0.0,
            ffn_dropout=0.0,
            alibi=False,
            learned_positions=("forward", "reverse") if bounded else "forward",
            learned_position_fusion=fusion,
            learned_position_fusion_residual=("rezero" if fusion == "concat" else None),
            learned_position_reverse_correction=("bounded_tanh" if bounded else None),
            learned_position_reverse_max_scale=0.025 if bounded else 0.1,
            learned_position_reverse_initializer_rng_nonadvancing=bounded,
        )
        self.sequence_model = build_causal_transformer(transformer, max_seq_len=16)
        self.downstream_norm = nn.LayerNorm(dim)
        self.downstream = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, cumulative_lens: torch.Tensor) -> torch.Tensor:
        return self.downstream(
            self.downstream_norm(self.sequence_model(x, cumulative_lens))
        )


def _initialized_model(kind: str, fusion: str, bounded: bool) -> _PositionModel:
    torch.manual_seed(31)
    model = _PositionModel(8, fusion, bounded)
    if kind == "standard":
        _initialize_standard_parameters(model, 0.02)
    else:
        base = _PositionModel(4, fusion, bounded)
        delta = _PositionModel(6, fusion, bounded)
        mup.set_base_shapes(model, base, delta=delta)
        _initialize_mup_parameters(model, 0.02)
    return model.eval()


@pytest.mark.parametrize("kind", ["standard", "mup"])
@pytest.mark.parametrize("fusion", ["add", "concat"])
def test_r7_initializer_matches_the_complete_forward_control(
    kind: str, fusion: str
) -> None:
    control = _initialized_model(kind, fusion, False)
    bounded = _initialized_model(kind, fusion, True)
    control_position = control.sequence_model.position_inputs[0]
    bounded_position = bounded.sequence_model.position_inputs[0]
    control_state = control_position.state_dict()
    bounded_control_state = bounded_position.forward_control.state_dict()

    assert control_state.keys() == bounded_control_state.keys()
    assert all(
        torch.equal(control_state[name], bounded_control_state[name])
        for name in control_state
    )
    control_shared = {
        name: parameter
        for name, parameter in control.named_parameters()
        if not name.startswith("sequence_model.position_inputs.")
    }
    bounded_shared = {
        name: parameter
        for name, parameter in bounded.named_parameters()
        if not name.startswith("sequence_model.position_inputs.")
    }
    assert control_shared.keys() == bounded_shared.keys()
    assert any("sequence_model.layers" in name for name in control_shared)
    assert any("norm" in name for name in control_shared)
    assert any(name.startswith("downstream") for name in control_shared)
    assert all(
        torch.equal(control_shared[name], bounded_shared[name])
        for name in control_shared
    )

    correction = bounded_position.reverse_correction
    correction_parameters = dict(correction.named_parameters())
    assert correction_parameters["gate"].item() == 0.0
    assert all(
        torch.isfinite(parameter).all() for parameter in correction_parameters.values()
    )
    assert all(
        torch.count_nonzero(parameter) > 0
        for name, parameter in correction_parameters.items()
        if name != "gate" and parameter.ndim >= 2
    )
    tokens = torch.randn(5, 8)
    lengths = packed_lens([2, 3])

    assert torch.equal(control(tokens, lengths), bounded(tokens, lengths))


class _MarkedLinear(nn.Module):
    initializer_rng_nonadvancing = True

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)


class _PretrainedModule(nn.Module):
    preserve_declared_initialization = True

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(4, 3)
        with torch.no_grad():
            self.encoder.weight.copy_(torch.arange(12).reshape(3, 4))
            self.encoder.bias.copy_(torch.tensor([5.0, 6.0, 7.0]))


def test_marked_pretrained_parameters_survive_standard_initialization() -> None:
    pretrained = _PretrainedModule()
    expected = deepcopy(pretrained.state_dict())
    model = nn.Sequential(pretrained, nn.Linear(3, 2))

    _initialize_standard_parameters(model, 0.02)

    assert all(
        torch.equal(value, expected[name])
        for name, value in pretrained.state_dict().items()
    )


class _RunModel(nn.Module):
    def __init__(self, marked: bool) -> None:
        super().__init__()
        self.before = nn.Linear(4, 4)
        if marked:
            self.correction = _MarkedLinear()
        self.after = nn.Linear(4, 4)


@pytest.mark.parametrize("kind", ["standard", "mup"])
def test_marked_parameter_run_does_not_advance_initializer_rng(kind: str) -> None:
    torch.manual_seed(7)
    control = _RunModel(False)
    torch.manual_seed(7)
    marked = _RunModel(True)
    if kind == "mup":
        mup.set_base_shapes(control, deepcopy(control))
        mup.set_base_shapes(marked, deepcopy(marked))
    initializer = (
        _initialize_standard_parameters
        if kind == "standard"
        else _initialize_mup_parameters
    )
    torch.manual_seed(101)
    initializer(control, 0.02)
    control_rng = torch.get_rng_state()
    torch.manual_seed(101)
    initializer(marked, 0.02)
    marked_rng = torch.get_rng_state()

    assert torch.equal(control.before.weight, marked.before.weight)
    assert torch.equal(control.after.weight, marked.after.weight)
    assert torch.equal(control_rng, marked_rng)
    assert torch.count_nonzero(marked.correction.linear.weight) > 0


def _legacy_standard_initializer(model: nn.Module, initializer_std: float) -> None:
    for name, parameter in model.named_parameters():
        if "weight" in name:
            nn.init.trunc_normal_(
                parameter,
                std=initializer_std,
                a=-2 * initializer_std,
                b=2 * initializer_std,
            )
        elif "bias" in name:
            nn.init.zeros_(parameter)


def _legacy_mup_initializer(model: nn.Module, initializer_std: float) -> None:
    for name, parameter in model.named_parameters():
        if name == "query_projection.weight" or name.endswith(".q_proj.weight"):
            nn.init.zeros_(parameter)
        elif "weight" in name and parameter.ndim >= 2:
            mup.init.trunc_normal_(
                parameter,
                std=initializer_std,
                a=-2 * initializer_std,
                b=2 * initializer_std,
            )
        elif name.endswith("weight"):
            nn.init.ones_(parameter)
        elif "bias" in name:
            nn.init.zeros_(parameter)


@pytest.mark.parametrize("kind", ["standard", "mup"])
def test_unmarked_initializer_behavior_is_unchanged(kind: str) -> None:
    model = _RunModel(False)
    expected = deepcopy(model)
    if kind == "mup":
        mup.set_base_shapes(model, deepcopy(model))
        mup.set_base_shapes(expected, deepcopy(expected))
    initializer = (
        _initialize_standard_parameters
        if kind == "standard"
        else _initialize_mup_parameters
    )
    legacy = (
        _legacy_standard_initializer if kind == "standard" else _legacy_mup_initializer
    )
    torch.manual_seed(211)
    initializer(model, 0.02)
    actual_rng = torch.get_rng_state()
    torch.manual_seed(211)
    legacy(expected, 0.02)
    expected_rng = torch.get_rng_state()

    assert all(
        torch.equal(actual, reference)
        for actual, reference in zip(
            model.parameters(), expected.parameters(), strict=True
        )
    )
    assert torch.equal(actual_rng, expected_rng)
