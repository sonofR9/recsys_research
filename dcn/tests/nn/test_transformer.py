import pytest
import torch

import dcn.nn.transformer as transformer_module
from dcn.nn.transformer import (
    ConcatenatedLearnedPositionInput,
    ForwardPositionInput,
    IndexPositions,
    LogValuePositions,
    ReverseRelativePositionInput,
    Rope,
    TransformerBlock,
    TransformerEncoder,
    ValuePositions,
)
from dcn.nn.types import ModuleWithDim
from dcn.tests.helpers import TinyFFN, packed_lens
from utils.global_config import config


pytestmark = pytest.mark.usefixtures("cpu_attention")


def _make_block(
    dim: int = 8,
    nhead: int = 2,
    num_kv_heads: int = 2,
    dropout: float = 0.0,
    use_alibi: bool = False,
    rope: Rope | None = None,
    **norm,
) -> TransformerBlock:
    return TransformerBlock(
        dim=dim,
        nhead=nhead,
        num_kv_heads=num_kv_heads,
        ffn_factory=TinyFFN,
        dropout=dropout,
        use_alibi=use_alibi,
        rope=rope,
        **norm,
    )


def _capture_flash_call(monkeypatch) -> dict:
    captured: dict = {}

    def fake_flash(**kwargs):
        captured.update(kwargs)
        return kwargs["q"]

    monkeypatch.setattr(
        transformer_module, "flash_attn_varlen_func", fake_flash, raising=False
    )
    config.set_cpu_attention(False)
    return captured


class TestReverseRelativePositionInput:
    def test_output_shape_matches_input(self) -> None:
        dim = 8
        module = ReverseRelativePositionInput(dim=dim, max_seq_len=16)
        x = torch.randn(7, dim)
        cumulative_lens = packed_lens([3, 4])

        output = module(x, cumulative_lens)

        assert output.shape == x.shape

    def test_position_zero_added_at_last_token_of_each_sequence(self) -> None:
        dim = 8
        module = ReverseRelativePositionInput(dim=dim, max_seq_len=16)
        with torch.no_grad():
            module.position_embeddings.weight.zero_()
            module.position_embeddings.weight[0] = 1.0

        x = torch.zeros(5, dim)
        cumulative_lens = packed_lens([2, 3])

        output = module(x, cumulative_lens)

        last_positions = (cumulative_lens[1:] - 1).tolist()
        for position in range(output.shape[0]):
            expected = (
                torch.ones(dim) if position in last_positions else torch.zeros(dim)
            )
            assert torch.equal(output[position], expected)

    def test_terminal_target_offset_matches_rectools_shifted_sessions(self) -> None:
        module = ReverseRelativePositionInput(dim=1, max_seq_len=8)
        with torch.no_grad():
            module.position_embeddings.weight.copy_(
                torch.arange(8, dtype=torch.float32).unsqueeze(1)
            )

        training = module(
            torch.zeros(4, 1),
            packed_lens([4]),
            reverse_position_offset=1,
        )
        inference = module(torch.zeros(3, 1), packed_lens([3]))

        assert training[:, 0].tolist() == [2.0, 1.0, 0.0, 0.0]
        assert training[:-1].tolist() == inference.tolist()


class TestForwardPositionInput:
    def test_position_zero_added_at_first_token_of_each_sequence(self) -> None:
        dim = 8
        module = ForwardPositionInput(dim=dim, max_seq_len=16)
        with torch.no_grad():
            module.position_embeddings.weight.zero_()
            module.position_embeddings.weight[0] = 1.0

        output = module(torch.zeros(5, dim), packed_lens([2, 3]))

        assert torch.equal(output[0], torch.ones(dim))
        assert torch.equal(output[2], torch.ones(dim))
        assert torch.equal(output[1], torch.zeros(dim))

    def test_a_token_keeps_its_position_when_the_sequence_grows(self) -> None:
        """What makes these usable for generation: appending renumbers nothing."""
        dim = 8
        module = ForwardPositionInput(dim=dim, max_seq_len=16)
        x = torch.zeros(4, dim)

        short = module(x[:3], packed_lens([3]))
        grown = module(x, packed_lens([4]))

        assert torch.equal(short, grown[:3])


class _CaptureItemSlice(ModuleWithDim):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self._out_dim = dim
        self.input: torch.Tensor | None = None

    @property
    def out_dim(self) -> int:
        return self._out_dim

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        self.input = values
        return values[:, : self.out_dim]


def test_concatenated_positions_encode_item_and_each_position_table_directly() -> None:
    dim = 2
    forward = ForwardPositionInput(dim, max_seq_len=2)
    reverse = ReverseRelativePositionInput(dim, max_seq_len=2)
    with torch.no_grad():
        forward.position_embeddings.weight.copy_(
            torch.tensor([[10.0, 11.0], [20.0, 21.0]])
        )
        reverse.position_embeddings.weight.copy_(
            torch.tensor([[100.0, 101.0], [200.0, 201.0]])
        )
    encoder = _CaptureItemSlice(dim)
    position_input = ConcatenatedLearnedPositionInput(
        [forward, reverse], encoder
    )
    items = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    output = position_input(items, packed_lens([2]))

    assert torch.equal(output, items)
    assert torch.equal(
        encoder.input,
        torch.tensor(
            [
                [1.0, 2.0, 10.0, 11.0, 200.0, 201.0],
                [3.0, 4.0, 20.0, 21.0, 100.0, 101.0],
            ]
        ),
    )


class TestTransformerBlockCpu:
    def test_output_shape_matches_flattened_input(self) -> None:
        torch.manual_seed(0)
        dim = 8
        block = _make_block(dim=dim)
        block.eval()

        x = torch.randn(6, dim)
        cumulative_lens = packed_lens([2, 4])

        output = block(x, cumulative_lens)

        assert output.shape == x.shape

    def test_causal_masking_future_tokens_do_not_leak(self) -> None:
        torch.manual_seed(0)
        dim = 8
        block = _make_block(dim=dim)
        block.eval()

        cumulative_lens = packed_lens([4])
        x_a = torch.randn(4, dim)
        x_b = x_a.clone()
        x_b[3] = torch.randn(dim)

        out_a = block(x_a, cumulative_lens)
        out_b = block(x_b, cumulative_lens)

        for position in range(3):
            assert torch.allclose(out_a[position], out_b[position], atol=1e-5)

    def test_bidirectional_attention_reads_later_history_tokens(self) -> None:
        torch.manual_seed(0)
        block = _make_block(dim=8, is_causal=False).eval()
        cumulative_lens = packed_lens([4])
        original = torch.randn(4, 8)
        changed = original.clone()
        changed[3] += 10

        original_output = block(original, cumulative_lens)
        changed_output = block(changed, cumulative_lens)

        assert not torch.allclose(original_output[0], changed_output[0], atol=1e-5)

    @pytest.mark.parametrize("is_causal", [True, False])
    def test_cross_sequence_isolation(self, is_causal: bool) -> None:
        torch.manual_seed(0)
        dim = 8
        block = _make_block(dim=dim, is_causal=is_causal)
        block.eval()

        cumulative_lens = packed_lens([3, 3])
        x_a = torch.randn(6, dim)
        x_b = x_a.clone()
        x_b[3:] = torch.randn(3, dim)

        out_a = block(x_a, cumulative_lens)
        out_b = block(x_b, cumulative_lens)

        assert torch.allclose(out_a[:3], out_b[:3], atol=1e-5)

    def test_local_attention_cannot_see_beyond_its_window(self) -> None:
        torch.manual_seed(0)
        block = _make_block(dim=8, attention_window=2).eval()
        cumulative_lens = packed_lens([4])
        original = torch.randn(4, 8)
        changed = original.clone()
        changed[0] = torch.randn(8)

        original_output = block(original, cumulative_lens)
        changed_output = block(changed, cumulative_lens)

        assert torch.allclose(original_output[3], changed_output[3], atol=1e-5)

    def test_grouped_query_attention_runs(self) -> None:
        torch.manual_seed(0)
        dim = 8
        block = _make_block(dim=dim, nhead=4, num_kv_heads=2)
        block.eval()

        x = torch.randn(5, dim)
        cumulative_lens = packed_lens([2, 3])

        output = block(x, cumulative_lens)

        assert output.shape == x.shape


def test_alibi_requires_power_of_two_nhead() -> None:
    with pytest.raises(AssertionError):
        _make_block(dim=6, nhead=3, num_kv_heads=3, use_alibi=True)


class TestNormalization:
    def test_post_norm_normalizes_the_residual_stream(self) -> None:
        """Pre-norm leaves the residual untouched, so it can grow with depth;
        post-norm hands the next layer something of unit scale."""
        torch.manual_seed(0)
        dim = 8
        x = torch.randn(4, dim) * 100
        cumulative_lens = packed_lens([4])

        pre = _make_block(dim=dim, norm_place="pre").eval()(x, cumulative_lens)
        post = _make_block(dim=dim, norm_place="post").eval()(x, cumulative_lens)

        assert pre.norm(dim=-1).min() > 10 * post.norm(dim=-1).max()

    def test_layer_norm_centres_what_rms_norm_only_rescales(self) -> None:
        torch.manual_seed(0)
        dim = 8
        x = torch.randn(4, dim) + 50
        cumulative_lens = packed_lens([4])

        def block(norm: str) -> torch.Tensor:
            return _make_block(dim=dim, norm=norm, norm_place="post").eval()(
                x, cumulative_lens
            )

        assert block("layer").mean(dim=-1).abs().max() < 1e-5
        assert block("rms").mean(dim=-1).abs().min() > 1e-2


class TestRopePositions:
    def test_index_positions_restart_at_each_sequence(self) -> None:
        positions = IndexPositions()(5, packed_lens([2, 3]))

        assert positions.tolist() == [0, 1, 0, 1, 2]

    def test_reversed_index_positions_count_down_to_the_last_token(self) -> None:
        positions = IndexPositions(reverse=True)(5, packed_lens([2, 3]))

        assert positions.tolist() == [1, 0, 2, 1, 0]

    def test_value_positions_are_relative_to_the_sequence_start(self) -> None:
        values = torch.tensor([100.0, 100.5, 7.0, 9.0, 9.25])

        positions = ValuePositions()(5, packed_lens([2, 3]), values)

        assert positions.tolist() == [0.0, 0.5, 0.0, 2.0, 2.25]

    def test_reversed_value_positions_are_relative_to_the_sequence_end(self) -> None:
        values = torch.tensor([100.0, 100.5, 7.0, 9.0, 9.25])

        positions = ValuePositions(reverse=True)(5, packed_lens([2, 3]), values)

        assert positions.tolist() == [0.5, 0.0, 2.25, 0.25, 0.0]

    def test_log_value_positions_take_the_log_of_the_delta(self) -> None:
        values = torch.tensor([0.0, 1.0, 100.0])

        positions = LogValuePositions()(3, packed_lens([3]), values)

        assert torch.allclose(positions, torch.log1p(values))

    def test_log_value_positions_keep_the_shared_reverse_and_scale_behaviour(
        self,
    ) -> None:
        values = torch.tensor([0.0, 1.0, 3.0])

        positions = LogValuePositions(scale=2.0, reverse=True)(
            3, packed_lens([3]), values
        )

        assert torch.allclose(
            positions, 2.0 * torch.log1p(torch.tensor([3.0, 2.0, 0.0]))
        )

    def test_value_positions_are_scaled(self) -> None:
        values = torch.tensor([0.0, 4.0])

        positions = ValuePositions(scale=0.25)(2, packed_lens([2]), values)

        assert positions.tolist() == [0.0, 1.0]

    def test_value_positions_require_values(self) -> None:
        with pytest.raises(AssertionError):
            ValuePositions()(2, packed_lens([2]))


class TestRope:
    def test_matches_independent_expected_values(self) -> None:
        rope = Rope(head_dim=4, base=10.0)
        x = torch.tensor(
            [
                [[1.0, 2.0, 3.0, 4.0]],
                [[0.5, -1.0, 2.0, -3.0]],
            ]
        )

        output = rope(x, packed_lens([2]))

        expected = torch.tensor(
            [
                [[1.0, 2.0, 3.0, 4.0]],
                [[-1.4127908, -0.0174645, 1.5013401, -3.1622294]],
            ]
        )
        assert torch.allclose(output, expected, atol=1e-6)

    def test_changes_block_output_and_preserves_shape(self) -> None:
        dim = 8
        torch.manual_seed(0)
        block_without = _make_block(dim=dim)
        torch.manual_seed(0)
        block_with = _make_block(dim=dim, rope=Rope(head_dim=dim // 2))
        block_without.eval()
        block_with.eval()

        x = torch.randn(5, dim)
        cumulative_lens = packed_lens([2, 3])

        out_without = block_without(x, cumulative_lens)
        out_with = block_with(x, cumulative_lens)

        assert out_with.shape == x.shape
        assert not torch.allclose(out_with, out_without)

    def test_positions_restart_at_each_sequence(self) -> None:
        torch.manual_seed(0)
        dim = 8
        block = _make_block(dim=dim, rope=Rope(head_dim=dim // 2))
        block.eval()

        first = torch.randn(3, dim)
        second = torch.randn(4, dim)

        streamed = block(torch.cat([first, second]), packed_lens([3, 4]))
        first_alone = block(first, packed_lens([3]))
        second_alone = block(second, packed_lens([4]))

        assert torch.allclose(streamed[:3], first_alone, atol=1e-5)
        assert torch.allclose(streamed[3:], second_alone, atol=1e-5)

    def test_works_with_grouped_query_attention(self) -> None:
        torch.manual_seed(0)
        dim = 8
        block = _make_block(dim=dim, nhead=4, num_kv_heads=2, rope=Rope(head_dim=2))
        block.eval()

        x = torch.randn(5, dim)

        output = block(x, packed_lens([2, 3]))

        assert output.shape == x.shape

    def test_reverse_positions_mirror_forward_positions_on_a_flipped_sequence(
        self,
    ) -> None:
        torch.manual_seed(0)
        head_dim = 4
        x = torch.randn(4, 1, head_dim)
        cumulative_lens = packed_lens([4])

        forward = Rope(head_dim=head_dim)(x, cumulative_lens)
        reverse = Rope(head_dim=head_dim, positions=IndexPositions(reverse=True))(
            x.flip(0), cumulative_lens
        )

        assert torch.allclose(forward, reverse.flip(0), atol=1e-6)

    def test_evenly_spaced_values_reproduce_integer_positions(self) -> None:
        torch.manual_seed(0)
        head_dim = 4
        x = torch.randn(3, 1, head_dim)
        cumulative_lens = packed_lens([3])

        integer = Rope(head_dim=head_dim)(x, cumulative_lens)
        by_value = Rope(head_dim=head_dim, positions=ValuePositions())(
            x, cumulative_lens, torch.tensor([10.0, 11.0, 12.0])
        )

        assert torch.allclose(integer, by_value, atol=1e-6)

    def test_fractional_positions_differ_from_integer_ones(self) -> None:
        torch.manual_seed(0)
        head_dim = 4
        x = torch.randn(3, 1, head_dim)
        cumulative_lens = packed_lens([3])
        rope = Rope(head_dim=head_dim, positions=ValuePositions())

        evenly_spaced = rope(x, cumulative_lens, torch.tensor([10.0, 11.0, 12.0]))
        irregular = rope(x, cumulative_lens, torch.tensor([10.0, 10.5, 40.0]))

        assert not torch.allclose(evenly_spaced, irregular)

    def test_position_values_reach_the_rope_through_the_encoder(self) -> None:
        torch.manual_seed(0)
        dim = 8
        encoder = TransformerEncoder(
            blocks=[
                _make_block(
                    dim=dim, rope=Rope(head_dim=dim // 2, positions=ValuePositions())
                )
            ]
        )
        encoder.eval()

        x = torch.randn(3, dim)
        cumulative_lens = packed_lens([3])

        evenly_spaced = encoder(x, cumulative_lens, torch.tensor([0.0, 1.0, 2.0]))
        irregular = encoder(x, cumulative_lens, torch.tensor([0.0, 0.5, 20.0]))

        assert evenly_spaced.shape == x.shape
        assert not torch.allclose(evenly_spaced, irregular)


class TestTransformerEncoder:
    def test_output_preserves_flattened_shape(self) -> None:
        torch.manual_seed(0)
        dim = 8
        encoder = TransformerEncoder(blocks=[_make_block(dim=dim) for _ in range(2)])
        encoder.eval()

        x = torch.randn(5, dim)
        cumulative_lens = packed_lens([2, 3])

        output = encoder(x, cumulative_lens)

        assert output.shape == x.shape

    def test_multiple_layers_compose(self) -> None:
        torch.manual_seed(0)
        dim = 8
        encoder_one_layer = TransformerEncoder(blocks=[_make_block(dim=dim)])
        encoder_two_layers = TransformerEncoder(
            blocks=[_make_block(dim=dim) for _ in range(2)]
        )
        encoder_one_layer.eval()
        encoder_two_layers.eval()

        x = torch.randn(4, dim)
        cumulative_lens = packed_lens([4])

        out_one = encoder_one_layer(x, cumulative_lens)
        out_two = encoder_two_layers(x, cumulative_lens)

        assert not torch.allclose(out_one, out_two)

    def test_per_layer_inputs_are_added_at_the_matching_layers(self) -> None:
        torch.manual_seed(0)
        encoder = TransformerEncoder(blocks=[_make_block() for _ in range(2)]).eval()
        x = torch.randn(4, 8)
        cumulative_lens = packed_lens([4])

        without = encoder(x, cumulative_lens)
        with_inputs = encoder(
            x,
            cumulative_lens,
            layer_inputs=[torch.ones_like(x), torch.full_like(x, 2)],
        )

        assert not torch.allclose(without, with_inputs)

    def test_per_layer_inputs_must_match_the_stack_depth(self) -> None:
        encoder = TransformerEncoder(blocks=[_make_block() for _ in range(2)]).eval()

        with pytest.raises(ValueError, match="2 layer inputs"):
            encoder(
                torch.randn(4, 8),
                packed_lens([4]),
                layer_inputs=[torch.randn(4, 8)],
            )

    def test_position_input_changes_output(self) -> None:
        torch.manual_seed(0)
        dim = 8
        blocks = [_make_block(dim=dim)]
        position_input = ReverseRelativePositionInput(dim=dim, max_seq_len=16)
        encoder_with_position = TransformerEncoder(
            blocks=blocks, position_inputs=[position_input]
        )
        encoder_without_position = TransformerEncoder(blocks=blocks)
        encoder_with_position.eval()
        encoder_without_position.eval()

        x = torch.randn(5, dim)
        cumulative_lens = packed_lens([2, 3])

        out_with = encoder_with_position(x, cumulative_lens)
        out_without = encoder_without_position(x, cumulative_lens)

        assert out_with.shape == out_without.shape
        assert not torch.allclose(out_with, out_without)

    def test_position_inputs_are_order_independent(self) -> None:
        torch.manual_seed(0)
        dim = 8
        blocks = [_make_block(dim=dim)]
        first = ReverseRelativePositionInput(dim=dim, max_seq_len=16)
        second = ReverseRelativePositionInput(dim=dim, max_seq_len=16)
        encoder_ab = TransformerEncoder(blocks=blocks, position_inputs=[first, second])
        encoder_ba = TransformerEncoder(blocks=blocks, position_inputs=[second, first])
        encoder_ab.eval()
        encoder_ba.eval()

        x = torch.randn(5, dim)
        cumulative_lens = packed_lens([2, 3])

        assert torch.allclose(
            encoder_ab(x, cumulative_lens), encoder_ba(x, cumulative_lens), atol=1e-6
        )


def test_flash_path_gets_int32_cu_seqlens_and_derived_max_seqlen(monkeypatch) -> None:
    captured = _capture_flash_call(monkeypatch)
    encoder = TransformerEncoder(blocks=[_make_block(dim=8)]).to(torch.bfloat16)
    encoder.eval()
    x = torch.randn(7, 8, dtype=torch.bfloat16)
    cumulative_lens = torch.tensor([0, 2, 6, 7], dtype=torch.int64)

    encoder(x, cumulative_lens)

    assert captured["cu_seqlens_q"].dtype == torch.int32
    assert captured["cu_seqlens_k"].dtype == torch.int32
    assert captured["max_seqlen_q"] == 4
    assert captured["max_seqlen_k"] == 4


def test_flash_path_uses_configured_sequence_length_without_readback(monkeypatch) -> None:
    captured = _capture_flash_call(monkeypatch)
    encoder = TransformerEncoder(
        blocks=[_make_block(dim=8)], max_seqlen=16
    ).to(torch.bfloat16)
    encoder.eval()

    encoder(
        torch.randn(7, 8, dtype=torch.bfloat16),
        torch.tensor([0, 2, 6, 7], dtype=torch.int64),
    )

    assert captured["max_seqlen_q"] == 16
    assert captured["max_seqlen_k"] == 16


def test_flash_path_gets_local_attention_window(monkeypatch) -> None:
    captured = _capture_flash_call(monkeypatch)
    encoder = TransformerEncoder(blocks=[_make_block(dim=8, attention_window=25)]).to(
        torch.bfloat16
    )
    encoder.eval()

    encoder(
        torch.randn(4, 8, dtype=torch.bfloat16),
        torch.tensor([0, 4], dtype=torch.int64),
    )

    assert captured["window_size"] == (24, 0)


def test_flash_path_gets_bidirectional_attention_and_symmetric_window(
    monkeypatch,
) -> None:
    captured = _capture_flash_call(monkeypatch)
    encoder = TransformerEncoder(
        blocks=[_make_block(dim=8, attention_window=25, is_causal=False)]
    ).to(torch.bfloat16)
    encoder.eval()

    encoder(
        torch.randn(4, 8, dtype=torch.bfloat16),
        torch.tensor([0, 4], dtype=torch.int64),
    )

    assert captured["causal"] is False
    assert captured["window_size"] == (24, 24)


def test_flash_path_combines_alibi_rope_and_position_inputs(monkeypatch) -> None:
    captured = _capture_flash_call(monkeypatch)
    dim = 8
    nhead = 2
    encoder = TransformerEncoder(
        blocks=[
            _make_block(
                dim=dim, nhead=nhead, use_alibi=True, rope=Rope(head_dim=dim // nhead)
            )
        ],
        position_inputs=[ReverseRelativePositionInput(dim=dim, max_seq_len=16)],
    ).to(torch.bfloat16)
    encoder.eval()
    x = torch.randn(5, dim, dtype=torch.bfloat16)

    output = encoder(x, torch.tensor([0, 2, 5], dtype=torch.int64))

    assert captured["alibi_slopes"].shape == (nhead,)
    assert output.shape == x.shape
