import pytest
import torch
from torch import nn

from dcn.nn.transformer import CrossAttentionBlock, TransformerBlock, TransformerDecoder
from dcn.tests.helpers import TinyFFN, packed_lens

pytestmark = pytest.mark.usefixtures("cpu_attention")

DIM = 8


def _make_cross_block(nhead: int = 2, num_kv_heads: int = 2) -> CrossAttentionBlock:
    return CrossAttentionBlock(
        dim=DIM,
        nhead=nhead,
        num_kv_heads=num_kv_heads,
        ffn_factory=TinyFFN,
        dropout=0.0,
    )


def _make_self_block() -> TransformerBlock:
    return TransformerBlock(
        dim=DIM,
        nhead=2,
        num_kv_heads=2,
        ffn_factory=TinyFFN,
        dropout=0.0,
        use_alibi=False,
    )


def _make_decoder(num_layers: int = 2) -> TransformerDecoder:
    return TransformerDecoder(
        self_attention_blocks=[_make_self_block() for _ in range(num_layers)],
        cross_attention_blocks=[_make_cross_block() for _ in range(num_layers)],
    )


@pytest.fixture(params=[_make_cross_block, _make_decoder], ids=["block", "decoder"])
def reads_memory(request: pytest.FixtureRequest) -> nn.Module:
    torch.manual_seed(0)
    module = request.param()
    module.eval()
    return module


class TestAttendingToAMemory:
    def test_output_keeps_the_query_shape(self, reads_memory: nn.Module) -> None:
        x = torch.randn(5, DIM)

        output = reads_memory(
            x, packed_lens([2, 3]), torch.randn(9, DIM), packed_lens([4, 5])
        )

        assert output.shape == x.shape

    def test_the_memory_changes_the_output(self, reads_memory: nn.Module) -> None:
        x = torch.randn(3, DIM)
        lens, memory_lens = packed_lens([3]), packed_lens([4])

        original = reads_memory(x, lens, torch.randn(4, DIM), memory_lens)
        changed = reads_memory(x, lens, torch.randn(4, DIM), memory_lens)

        assert not torch.allclose(original, changed)

    def test_another_sequence_memory_does_not_leak(
        self, reads_memory: nn.Module
    ) -> None:
        x = torch.randn(5, DIM)
        lens, memory_lens = packed_lens([2, 3]), packed_lens([4, 3])
        memory = torch.randn(7, DIM)
        other_memory = memory.clone()
        other_memory[4:] = torch.randn(3, DIM)

        original = reads_memory(x, lens, memory, memory_lens)
        changed = reads_memory(x, lens, other_memory, memory_lens)

        assert torch.allclose(original[:2], changed[:2], atol=1e-5)
        assert not torch.allclose(original[2:], changed[2:])

    def test_batched_sequences_match_running_them_one_at_a_time(
        self, reads_memory: nn.Module
    ) -> None:
        first_x, second_x = torch.randn(2, DIM), torch.randn(3, DIM)
        first_memory, second_memory = torch.randn(5, DIM), torch.randn(1, DIM)

        batched = reads_memory(
            torch.cat([first_x, second_x]),
            packed_lens([2, 3]),
            torch.cat([first_memory, second_memory]),
            packed_lens([5, 1]),
        )
        first_alone = reads_memory(
            first_x, packed_lens([2]), first_memory, packed_lens([5])
        )
        second_alone = reads_memory(
            second_x, packed_lens([3]), second_memory, packed_lens([1])
        )

        assert torch.allclose(batched[:2], first_alone, atol=1e-5)
        assert torch.allclose(batched[2:], second_alone, atol=1e-5)

    def test_a_query_token_attends_to_the_whole_memory(
        self, reads_memory: nn.Module
    ) -> None:
        x = torch.randn(1, DIM)
        lens, memory_lens = packed_lens([1]), packed_lens([3])
        memory = torch.randn(3, DIM)
        last_token_changed = memory.clone()
        last_token_changed[2] = torch.randn(DIM)

        original = reads_memory(x, lens, memory, memory_lens)
        changed = reads_memory(x, lens, last_token_changed, memory_lens)

        assert not torch.allclose(original, changed)


class TestCrossAttentionBlock:
    def test_grouped_query_attention_runs(self) -> None:
        torch.manual_seed(0)
        block = _make_cross_block(nhead=4, num_kv_heads=2)
        block.eval()

        x = torch.randn(5, DIM)

        output = block(x, packed_lens([2, 3]), torch.randn(6, DIM), packed_lens([2, 4]))

        assert output.shape == x.shape


class TestTransformerDecoder:
    def test_causal_self_attention_hides_later_tokens(self) -> None:
        torch.manual_seed(0)
        decoder = _make_decoder()
        decoder.eval()

        lens, memory_lens = packed_lens([4]), packed_lens([3])
        memory = torch.randn(3, DIM)
        x = torch.randn(4, DIM)
        new_last_token = x.clone()
        new_last_token[3] = torch.randn(DIM)

        original = decoder(x, lens, memory, memory_lens)
        changed = decoder(new_last_token, lens, memory, memory_lens)

        assert torch.allclose(original[:3], changed[:3], atol=1e-5)
        assert not torch.allclose(original[3], changed[3])

    def test_requires_matching_block_counts(self) -> None:
        with pytest.raises(AssertionError):
            TransformerDecoder(
                self_attention_blocks=[_make_self_block() for _ in range(2)],
                cross_attention_blocks=[_make_cross_block()],
            )
