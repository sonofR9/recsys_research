"""The architecture knobs a sweep turns are only useful if they reach the stack."""

from dataclasses import replace

import pytest
import torch
from torch import nn

from dcn.config.networks import build_causal_transformer
from dcn.config.settings import TRANSFORMER, TransformerConfig
from dcn.nn import DenseNet, GEGLU, ReGLU, RegularMLP, SwiGLU
from dcn.nn.transformer import (
    BoundedReverseAdditivePositionInput,
    BoundedReverseConcatenatedPositionInput,
    ConcatenatedLearnedPositionInput,
    ForwardPositionInput,
    LogValuePositions,
    ReverseRelativePositionInput,
    ValuePositions,
)
from dcn.tests.helpers import packed_lens

pytestmark = pytest.mark.usefixtures("cpu_attention")

_SMALL = replace(TRANSFORMER, dim=8, nhead=2, num_kv_heads=1, ffn_intermediate_dim=16)


def _encoder(**knobs):
    return build_causal_transformer(replace(_SMALL, **knobs), max_seq_len=16).eval()


def _run(encoder) -> torch.Tensor:
    torch.manual_seed(0)
    return encoder(torch.randn(5, 8), packed_lens([2, 3]))


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        ("relu", RegularMLP),
        ("gelu", RegularMLP),
        ("silu", RegularMLP),
        ("reglu", ReGLU),
        ("geglu", GEGLU),
        ("swiglu", SwiGLU),
    ],
)
def test_ffn_kind_selects_the_feedforward(kind, expected) -> None:
    assert isinstance(_encoder(ffn=kind).layers[0].ffn, expected)


@pytest.mark.parametrize("kind", ["relu", "gelu", "silu"])
def test_ffn_dropout_reaches_every_ordinary_family(kind) -> None:
    ffn = _encoder(ffn=kind, ffn_dropout=0.25).layers[0].ffn

    assert ffn.dropout.p == 0.25


@pytest.mark.parametrize("kind", ["reglu", "geglu", "swiglu"])
def test_gated_ffn_dropout_is_explicitly_opted_in(kind) -> None:
    legacy = _encoder(ffn=kind, ffn_dropout=0.25).layers[0].ffn
    matched = _encoder(ffn=kind, ffn_dropout=0.25, gated_ffn_dropout=True).layers[0].ffn

    assert legacy.dropout.p == 0.0
    assert matched.dropout.p == 0.25


@pytest.mark.parametrize(
    "kind, expected",
    [("rms", nn.RMSNorm), ("layer", nn.LayerNorm), ("batch", nn.BatchNorm1d)],
)
def test_norm_kind_selects_the_normalization(kind, expected) -> None:
    assert isinstance(_encoder(norm=kind).layers[0].attention_norm.norm, expected)


@pytest.mark.parametrize(
    ("input_kind", "final_kind", "input_type", "final_type"),
    [
        ("rms", "rms", nn.RMSNorm, nn.RMSNorm),
        (None, None, nn.Identity, nn.Identity),
    ],
)
def test_input_and_final_normalization_are_independent(
    input_kind, final_kind, input_type, final_type
) -> None:
    encoder = _encoder(input_norm=input_kind, final_norm=final_kind)

    assert isinstance(encoder.input_norm, input_type)
    assert isinstance(encoder.final_norm, final_type)


def test_alibi_can_be_turned_off() -> None:
    assert _encoder(alibi=True).layers[0].alibi_slopes is not None
    assert _encoder(alibi=False).layers[0].alibi_slopes is None


def test_attention_window_reaches_every_block() -> None:
    encoder = _encoder(attention_window=25)

    assert [layer.attention_window for layer in encoder.layers] == [25, 25]


@pytest.mark.parametrize("order", ["forward", "reverse"])
def test_rope_and_learned_positions_are_independent_choices(order) -> None:
    assert _encoder(rope=order).layers[0].rope is not None
    assert _encoder(rope=None).layers[0].rope is None
    assert len(_encoder(learned_positions=order).position_inputs) == 1
    assert len(_encoder(learned_positions=None).position_inputs) == 0


def test_forward_and_reverse_learned_positions_can_be_combined() -> None:
    encoder = _encoder(learned_positions=("forward", "reverse"))

    assert [type(position) for position in encoder.position_inputs] == [
        ForwardPositionInput,
        ReverseRelativePositionInput,
    ]
    assert _run(encoder).shape == (5, 8)


def test_zero_reverse_add_starts_as_forward_only_and_learns_on_first_step() -> None:
    combined = _encoder(
        learned_positions=("forward", "reverse"),
        learned_position_initialization="zero_reverse",
        input_dropout=0.0,
    )
    forward_only = _encoder(learned_positions="forward", input_dropout=0.0)
    forward_only.position_inputs[0].position_embeddings.weight.data.copy_(
        combined.position_inputs[0].position_embeddings.weight
    )
    item = torch.randn(5, 8)
    lengths = packed_lens([2, 3])
    reverse = combined.position_inputs[1].position_embeddings.weight

    combined_output = item
    for position in combined.position_inputs:
        combined_output = position(combined_output, lengths)
    forward_output = forward_only.position_inputs[0](item, lengths)
    combined_output.square().mean().backward()

    assert combined.position_inputs[0].position_embeddings.weight.abs().sum() > 0
    assert torch.count_nonzero(reverse) == 0
    assert torch.equal(combined_output, forward_output)
    assert reverse.grad.abs().sum() > 0


def test_zero_reverse_concat_starts_at_item_and_only_gate_learns_on_first_step() -> (
    None
):
    encoder = _encoder(
        learned_positions=("forward", "reverse"),
        learned_position_fusion="concat",
        learned_position_fusion_residual="rezero",
        learned_position_initialization="zero_reverse",
        input_dropout=0.0,
    )
    position_input = encoder.position_inputs[0]
    reverse = position_input.position_inputs[1].position_embeddings.weight
    item = torch.randn(5, 8)
    lengths = packed_lens([2, 3])

    output = position_input(item, lengths)
    (output * torch.randn_like(output)).sum().backward()

    assert torch.equal(output, item)
    assert torch.count_nonzero(reverse) == 0
    assert position_input.gate.grad.abs() > 0
    assert reverse.grad is not None
    assert torch.count_nonzero(reverse.grad) == 0

    encoder.zero_grad(set_to_none=True)
    position_input.gate.data.fill_(0.25)
    position_input(item, lengths).square().mean().backward()

    assert reverse.grad.abs().sum() > 0


def _initialize_like_project(
    model: nn.Module, standard_deviation: float = 0.02
) -> None:
    for name, parameter in model.named_parameters():
        if "weight" in name:
            nn.init.trunc_normal_(
                parameter,
                std=standard_deviation,
                a=-2 * standard_deviation,
                b=2 * standard_deviation,
            )
        elif "bias" in name:
            nn.init.zeros_(parameter)


@pytest.mark.parametrize("fusion", ["add", "concat"])
def test_bounded_reverse_starts_as_exact_forward_control_after_project_initialization(
    fusion: str,
) -> None:
    control_knobs = {
        "learned_positions": "forward",
        "learned_position_fusion": fusion,
        "learned_position_fusion_residual": "rezero" if fusion == "concat" else None,
        "input_dropout": 0.0,
    }
    bounded_knobs = control_knobs | {
        "learned_positions": ("forward", "reverse"),
        "learned_position_reverse_correction": "bounded_tanh",
    }
    torch.manual_seed(19)
    control = _encoder(**control_knobs)
    _initialize_like_project(control)
    torch.manual_seed(19)
    bounded = _encoder(**bounded_knobs)
    _initialize_like_project(bounded)
    position_input = bounded.position_inputs[0]
    forward_control = position_input.forward_control
    item = torch.randn(5, 8)
    lengths = packed_lens([2, 3])

    assert all(
        torch.equal(control_value, bounded_value)
        for control_value, bounded_value in zip(
            control.position_inputs[0].state_dict().values(),
            forward_control.state_dict().values(),
            strict=True,
        )
    )
    assert torch.equal(
        position_input(item, lengths), control.position_inputs[0](item, lengths)
    )
    parameter_names = [name for name, _ in position_input.named_parameters()]
    first_correction = next(
        index
        for index, name in enumerate(parameter_names)
        if name.startswith("reverse_correction.")
    )
    assert all(
        name.startswith("forward_control.")
        for name in parameter_names[:first_correction]
    )


@pytest.mark.parametrize(
    ("fusion", "expected_type"),
    [
        ("add", BoundedReverseAdditivePositionInput),
        ("concat", BoundedReverseConcatenatedPositionInput),
    ],
)
def test_bounded_reverse_gate_learns_on_first_step(
    fusion: str, expected_type: type[nn.Module]
) -> None:
    encoder = _encoder(
        learned_positions=("forward", "reverse"),
        learned_position_fusion=fusion,
        learned_position_fusion_residual="rezero" if fusion == "concat" else None,
        learned_position_reverse_correction="bounded_tanh",
        input_dropout=0.0,
    )
    position_input = encoder.position_inputs[0]
    assert isinstance(position_input, expected_type)
    item = torch.randn(5, 8)
    lengths = packed_lens([2, 3])

    (position_input(item, lengths) * torch.randn_like(item)).sum().backward()

    correction = position_input.reverse_correction
    assert correction.gate.item() == 0.0
    assert correction.gate.grad.abs() > 0
    assert correction.reverse_input.position_embeddings.weight.grad is not None
    assert (
        torch.count_nonzero(correction.reverse_input.position_embeddings.weight.grad)
        == 0
    )


@pytest.mark.parametrize("fusion", ["add", "concat"])
def test_bounded_reverse_coefficient_uses_tanh_scale_capped_at_point_one(
    fusion: str,
) -> None:
    encoder = _encoder(
        learned_positions=("forward", "reverse"),
        learned_position_fusion=fusion,
        learned_position_fusion_residual="rezero" if fusion == "concat" else None,
        learned_position_reverse_correction="bounded_tanh",
        learned_position_reverse_max_scale=0.1,
        input_dropout=0.0,
    )
    position_input = encoder.position_inputs[0]
    correction = position_input.reverse_correction
    item = torch.randn(5, 8)
    lengths = packed_lens([2, 3])
    correction.gate.data.fill_(1e6)
    control = position_input.forward_control(item, lengths)
    if fusion == "add":
        raw_correction = correction.reverse_input.embeddings(lengths, item.shape[0])
    else:
        forward_embeddings = position_input.forward_control.position_inputs[
            0
        ].embeddings(lengths, item.shape[0])
        reverse_embeddings = correction.reverse_input.embeddings(lengths, item.shape[0])
        raw_correction = correction.encoder(
            torch.cat([item, forward_embeddings, reverse_embeddings], dim=-1)
        )

    contribution = position_input(item, lengths) - control

    assert correction.max_scale == 0.1
    assert correction.scale.item() == pytest.approx(0.1, abs=2e-8)
    assert torch.allclose(contribution, 0.1 * raw_correction, atol=1e-7)
    if fusion == "concat":
        assert correction.encoder.input_dim == 24
        assert correction.encoder.preserve_input_rms


@pytest.mark.parametrize("fusion", ["add", "concat"])
def test_bounded_reverse_composes_with_alibi_and_preserves_causality(
    fusion: str,
) -> None:
    encoder = _encoder(
        learned_positions=("forward", "reverse"),
        learned_position_fusion=fusion,
        learned_position_fusion_residual="rezero" if fusion == "concat" else None,
        learned_position_reverse_correction="bounded_tanh",
        alibi=True,
    )
    encoder.position_inputs[0].reverse_correction.gate.data.fill_(0.5)
    original = torch.randn(4, 8)
    changed = original.clone()
    changed[-1] = torch.randn(8)

    original_output = encoder(original, packed_lens([4]))
    changed_output = encoder(changed, packed_lens([4]))

    assert all(layer.alibi_slopes is not None for layer in encoder.layers)
    assert torch.allclose(original_output[:-1], changed_output[:-1], atol=1e-5)


@pytest.mark.parametrize(
    ("orders", "position_types", "input_dim"),
    [
        ("forward", [ForwardPositionInput], 16),
        (
            ("forward", "reverse"),
            [ForwardPositionInput, ReverseRelativePositionInput],
            24,
        ),
    ],
)
def test_concatenated_positions_use_one_densenet_fusion(
    orders, position_types, input_dim
) -> None:
    encoder = _encoder(
        learned_positions=orders,
        learned_position_fusion="concat",
    )

    assert len(encoder.position_inputs) == 1
    position_input = encoder.position_inputs[0]
    assert isinstance(position_input, ConcatenatedLearnedPositionInput)
    assert [
        type(position) for position in position_input.position_inputs
    ] == position_types
    assert isinstance(position_input.encoder, DenseNet)
    assert position_input.encoder.input_dim == input_dim
    assert position_input.encoder.out_dim == 8
    assert _run(encoder).shape == (5, 8)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_normalized_concat_preserves_additive_input_scale_under_project_initializer(
    dtype: torch.dtype,
) -> None:
    torch.manual_seed(0)
    additive = _encoder(
        dim=64,
        nhead=2,
        num_kv_heads=1,
        ffn_intermediate_dim=171,
        learned_positions="forward",
        learned_position_fusion="add",
        input_dropout=0.0,
    )
    concatenated = _encoder(
        dim=64,
        nhead=2,
        num_kv_heads=1,
        ffn_intermediate_dim=171,
        learned_positions="forward",
        learned_position_fusion="concat",
        learned_position_fusion_normalization="input_rms",
        input_dropout=0.0,
    )
    for model in (additive, concatenated):
        for name, parameter in model.named_parameters():
            if "weight" in name:
                nn.init.trunc_normal_(parameter, std=0.02, a=-0.04, b=0.04)
            elif "bias" in name:
                nn.init.zeros_(parameter)
        model.to(dtype=dtype)

    tokens = torch.empty(16, 64)
    nn.init.trunc_normal_(tokens, std=0.02, a=-0.04, b=0.04)
    additive_tokens = tokens.to(dtype=dtype).requires_grad_()
    concat_tokens = tokens.to(dtype=dtype).requires_grad_()
    lengths = packed_lens([8, 8])
    additive_output = additive.position_inputs[0](additive_tokens, lengths)
    concat_output = concatenated.position_inputs[0](concat_tokens, lengths)
    scale_ratio = (
        concat_output.square().mean().sqrt() / additive_output.square().mean().sqrt()
    )
    upstream = torch.randn_like(additive_output)
    (additive_output * upstream).sum().backward()
    (concat_output * upstream).sum().backward()
    gradient_ratio = (
        concat_tokens.grad.square().mean().sqrt()
        / additive_tokens.grad.square().mean().sqrt()
    )

    assert 0.5 < scale_ratio < 2.0
    assert 0.25 < gradient_ratio < 4.0


def test_rezero_concat_starts_as_item_identity_not_additive_position() -> None:
    encoder = _encoder(
        learned_positions="forward",
        learned_position_fusion="concat",
        learned_position_fusion_residual="rezero",
        input_dropout=0.0,
    )
    item = torch.randn(5, 8)
    lengths = packed_lens([2, 3])

    output = encoder.position_inputs[0](item, lengths)
    additive = _encoder(
        learned_positions="forward",
        learned_position_fusion="add",
        input_dropout=0.0,
    ).position_inputs[0](item, lengths)

    assert torch.equal(output, item)
    assert not torch.equal(output, additive)


def test_rezero_concat_branch_receives_gradients_after_gate_moves() -> None:
    encoder = _encoder(
        learned_positions=("forward", "reverse"),
        learned_position_fusion="concat",
        learned_position_fusion_residual="rezero",
        input_dropout=0.0,
    )
    position_input = encoder.position_inputs[0]
    position_input.gate.data.fill_(0.25)

    position_input(torch.randn(5, 8), packed_lens([2, 3])).square().mean().backward()

    assert position_input.encoder.input_projection.weight.grad.abs().sum() > 0
    assert all(
        position.position_embeddings.weight.grad.abs().sum() > 0
        for position in position_input.position_inputs
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rezero_concat_scale_and_gradients_are_finite_under_project_initializer(
    dtype: torch.dtype,
) -> None:
    encoder = _encoder(
        dim=64,
        nhead=2,
        num_kv_heads=1,
        ffn_intermediate_dim=171,
        learned_positions="forward",
        learned_position_fusion="concat",
        learned_position_fusion_residual="rezero",
        input_dropout=0.0,
    )
    for name, parameter in encoder.named_parameters():
        if "weight" in name:
            nn.init.trunc_normal_(parameter, std=0.02, a=-0.04, b=0.04)
        elif "bias" in name:
            nn.init.zeros_(parameter)
    encoder.to(dtype=dtype)
    position_input = encoder.position_inputs[0]
    assert position_input.gate.item() == 0.0

    tokens = torch.empty(16, 64)
    nn.init.trunc_normal_(tokens, std=0.02, a=-0.04, b=0.04)
    tokens = tokens.to(dtype=dtype).requires_grad_()
    lengths = packed_lens([8, 8])
    position_input.gate.data.fill_(1.0)
    branch = (position_input(tokens, lengths) - tokens).detach()
    position_input.gate.data.zero_()
    output = position_input(tokens, packed_lens([8, 8]))
    (output * branch).mean().backward()
    input_rms = tokens.float().square().mean().sqrt()
    branch_rms = branch.float().square().mean().sqrt()
    gate_gradient_scale = position_input.gate.grad.float().abs().sqrt()

    assert torch.equal(output, tokens)
    assert 0.5 < branch_rms / input_rms < 2.0
    assert 0.5 < gate_gradient_scale / input_rms < 2.0
    assert torch.isfinite(output).all()
    assert torch.isfinite(tokens.grad).all()
    assert torch.isfinite(position_input.gate.grad)

    encoder.zero_grad(set_to_none=True)
    position_input.gate.data.fill_(0.25)
    moved_tokens = tokens.detach().clone().requires_grad_()
    moved_output = position_input(moved_tokens, packed_lens([8, 8]))
    moved_output.square().mean().backward()

    assert torch.isfinite(moved_output).all()
    assert torch.isfinite(moved_tokens.grad).all()
    assert torch.isfinite(position_input.encoder.input_projection.weight.grad).all()


def test_additive_position_variants_keep_the_existing_composition() -> None:
    encoder = _encoder(
        learned_positions=("forward", "reverse"),
        learned_position_fusion="add",
    )

    assert [type(position) for position in encoder.position_inputs] == [
        ForwardPositionInput,
        ReverseRelativePositionInput,
    ]


def test_concatenated_position_fusion_composes_with_alibi_and_causal_masking() -> None:
    encoder = _encoder(
        learned_positions=("forward", "reverse"),
        learned_position_fusion="concat",
        learned_position_fusion_residual="rezero",
        alibi=True,
    )
    encoder.position_inputs[0].gate.data.fill_(0.25)
    original = torch.randn(4, 8)
    changed = original.clone()
    changed[-1] = torch.randn(8)

    original_output = encoder(original, packed_lens([4]))
    changed_output = encoder(changed, packed_lens([4]))

    assert all(layer.alibi_slopes is not None for layer in encoder.layers)
    assert torch.allclose(original_output[:-1], changed_output[:-1], atol=1e-5)


@pytest.mark.parametrize("learned_positions", [None, ()])
def test_concatenated_position_fusion_requires_at_least_one_table(
    learned_positions,
) -> None:
    with pytest.raises(
        ValueError, match="concatenated position fusion requires learned positions"
    ):
        TransformerConfig(
            learned_positions=learned_positions,
            learned_position_fusion="concat",
        )


def test_transformer_config_rejects_unknown_position_fusion_normalization() -> None:
    with pytest.raises(ValueError, match="unknown position fusion normalization"):
        TransformerConfig(
            learned_positions="forward",
            learned_position_fusion="concat",
            learned_position_fusion_normalization="typo",  # type: ignore[arg-type]
        )


def test_transformer_config_rejects_unknown_position_fusion_residual() -> None:
    with pytest.raises(ValueError, match="unknown position fusion residual"):
        TransformerConfig(
            learned_positions="forward",
            learned_position_fusion="concat",
            learned_position_fusion_residual="typo",  # type: ignore[arg-type]
        )


def test_transformer_config_rejects_two_concat_corrections() -> None:
    with pytest.raises(ValueError, match="normalization and residual are exclusive"):
        TransformerConfig(
            learned_positions="forward",
            learned_position_fusion="concat",
            learned_position_fusion_normalization="input_rms",
            learned_position_fusion_residual="rezero",
        )


def test_zero_reverse_initialization_requires_exact_combined_positions() -> None:
    with pytest.raises(ValueError, match="requires forward and reverse positions"):
        TransformerConfig(
            learned_positions="reverse",
            learned_position_initialization="zero_reverse",
        )


def test_transformer_config_rejects_unknown_position_initialization() -> None:
    with pytest.raises(ValueError, match="unknown learned position initialization"):
        TransformerConfig(
            learned_positions=("forward", "reverse"),
            learned_position_initialization="typo",  # type: ignore[arg-type]
        )


def test_bounded_reverse_requires_exact_combined_forward_control() -> None:
    with pytest.raises(ValueError, match="bounded reverse correction requires"):
        TransformerConfig(
            learned_positions="reverse",
            learned_position_reverse_correction="bounded_tanh",
        )
    with pytest.raises(ValueError, match="bounded reverse correction requires"):
        TransformerConfig(
            learned_positions=("forward", "reverse"),
            learned_position_fusion="concat",
            learned_position_reverse_correction="bounded_tanh",
        )


@pytest.mark.parametrize("max_scale", [0.0, -0.1, float("inf"), float("nan")])
def test_bounded_reverse_rejects_invalid_max_scale(max_scale: float) -> None:
    with pytest.raises(ValueError, match="reverse max scale must be positive finite"):
        TransformerConfig(
            learned_positions=("forward", "reverse"),
            learned_position_reverse_correction="bounded_tanh",
            learned_position_reverse_max_scale=max_scale,
        )


def test_reverse_max_scale_is_not_a_latent_inactive_knob() -> None:
    with pytest.raises(ValueError, match="reverse max scale requires"):
        TransformerConfig(learned_position_reverse_max_scale=0.2)


def test_reverse_initializer_rng_isolation_requires_a_correction() -> None:
    with pytest.raises(ValueError, match="requires bounded reverse correction"):
        TransformerConfig(
            learned_position_reverse_initializer_rng_nonadvancing=True,
        )


def test_rope_base_reaches_every_block() -> None:
    encoder = _encoder(rope="forward", rope_base=100.0)

    assert all(layer.rope.base == 100.0 for layer in encoder.layers)


@pytest.mark.parametrize("rope_base", [0.0, -1.0, float("inf"), float("nan")])
def test_transformer_config_rejects_invalid_rope_base(rope_base: float) -> None:
    with pytest.raises(ValueError, match="rope_base must be positive finite"):
        TransformerConfig(rope_base=rope_base)


@pytest.mark.parametrize(
    ("kind", "positions", "reverse"),
    [
        ("timestamp", ValuePositions, False),
        ("timestamp_reverse", ValuePositions, True),
        ("timestamp_log", LogValuePositions, False),
        ("timestamp_log_reverse", LogValuePositions, True),
    ],
)
def test_timestamp_rope_selects_value_positions(kind, positions, reverse) -> None:
    rope = _encoder(rope=kind).layers[0].rope

    assert isinstance(rope.positions, positions)
    assert rope.positions.reverse is reverse


def test_every_position_choice_builds_and_keeps_the_output_shape() -> None:
    for knobs in [
        {"alibi": False, "rope": "forward", "learned_positions": None},
        {"alibi": False, "rope": "reverse", "learned_positions": "reverse"},
        {"alibi": True, "rope": "forward", "learned_positions": "forward"},
        {"alibi": False, "rope": None, "learned_positions": None},
        {"norm_place": "post", "norm": "layer", "ffn": "gelu"},
    ]:
        assert _run(_encoder(**knobs)).shape == (5, 8)
