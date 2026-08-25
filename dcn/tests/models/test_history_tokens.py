import math

import pytest
import torch

from dcn.models.history_tokens import (
    ActionTokenizer,
    BosTokenizer,
    ItemTokenizer,
    SemanticIdTokenizer,
    TimestampDeltaTokenizer,
)
from dcn.models.sequence_targets import NextItemTargets
from dcn.nn.semantic_embedding import SemanticIdEmbedding
from dcn.tests.helpers import CODES, ACTION_COLUMN, ITEM_COLUMN, packed_batch


def _embedding(num_items: int = 8, dim: int = 4) -> torch.nn.Embedding:
    return torch.nn.Embedding(num_items, dim)


class TestItemTokenizer:
    def test_one_token_per_event(self) -> None:
        tokenizer = ItemTokenizer(_embedding(), item_id_column=ITEM_COLUMN)

        tokens = tokenizer(packed_batch([1, 2, 3, 4], [3, 1]))

        assert tokens.embeddings.shape == (4, 4)
        assert tokens.cumulative_lens.tolist() == [0, 3, 4]
        assert tokens.is_target.tolist() == [True] * 4
        assert tokens.item_ids.tolist() == [1, 2, 3, 4]
        assert tokens.timestamps.tolist() == [0, 1, 2, 3]

    def test_projection_decouples_item_and_transformer_widths(self) -> None:
        tokenizer = ItemTokenizer(
            _embedding(dim=4),
            item_id_column=ITEM_COLUMN,
            projection=torch.nn.Linear(4, 8),
        )

        tokens = tokenizer(packed_batch([1, 2, 3], [3]))

        assert tokenizer.out_dim == 8
        assert tokens.embeddings.shape == (3, 8)


class TestTimestampDeltaTokenizer:
    @staticmethod
    def _tokenizer(
        kind: str, combination: str = "add", num_bins: int = 32
    ) -> TimestampDeltaTokenizer:
        item_embedding = _embedding()
        item_embedding.weight.data.zero_()
        return TimestampDeltaTokenizer(
            ItemTokenizer(item_embedding, item_id_column=ITEM_COLUMN),
            kind=kind,
            combination=combination,
            num_bins=num_bins,
        )

    @pytest.mark.parametrize("kind", ["plain", "log", "bins"])
    def test_time_gap_changes_the_token_embedding(self, kind: str) -> None:
        tokenizer = self._tokenizer(kind)
        short_gap = packed_batch([1, 1], [2])
        long_gap = packed_batch([1, 1], [2])
        short_gap["timestamp"] = torch.tensor([100, 101])
        long_gap["timestamp"] = torch.tensor([100, 100_000])

        short = tokenizer(short_gap)
        long = tokenizer(long_gap)

        assert short.embeddings.shape == (2, 4)
        assert not torch.allclose(short.embeddings[1], long.embeddings[1])

    def test_time_gaps_restart_at_each_sequence(self) -> None:
        tokenizer = self._tokenizer("log")
        batch = packed_batch([1, 1, 1, 1], [2, 2])
        batch["timestamp"] = torch.tensor([10, 20, 1000, 1010])

        tokens = tokenizer(batch)

        assert torch.allclose(tokens.embeddings[0], tokens.embeddings[2])
        assert torch.allclose(tokens.embeddings[1], tokens.embeddings[3])

    def test_concatenation_is_projected_back_to_the_model_width(self) -> None:
        tokenizer = self._tokenizer("bins", combination="concat")

        tokens = tokenizer(packed_batch([1, 2, 3], [3]))

        assert tokens.embeddings.shape == (3, 4)
        assert tokenizer.out_dim == 4

    @pytest.mark.parametrize("num_bins", [8, 16, 32, 64])
    def test_every_configured_time_bin_is_reachable(self, num_bins: int) -> None:
        tokenizer = self._tokenizer("bins", num_bins=num_bins)
        time_encoder = tokenizer.time_encoder
        assert isinstance(time_encoder, torch.nn.Embedding)
        time_encoder.weight.data.copy_(
            torch.arange(num_bins, dtype=torch.float32).unsqueeze(1).expand(-1, 4)
        )
        interior = (torch.arange(1, num_bins - 1) - 0.5) / (num_bins - 2)
        normalized_gaps = torch.cat([torch.tensor([0.0]), interior, torch.tensor([1.0])])
        gaps = torch.expm1(normalized_gaps * math.log1p(tokenizer.max_delta_seconds))
        batch = packed_batch([1] * (2 * num_bins), [2] * num_bins)
        batch["timestamp"] = torch.stack([torch.zeros_like(gaps), gaps], dim=1).flatten()

        tokens = tokenizer(batch)

        assert tokens.embeddings[1::2, 0].tolist() == pytest.approx(range(num_bins))


def _bos_tokenizer() -> BosTokenizer:
    return BosTokenizer(ItemTokenizer(_embedding(), item_id_column=ITEM_COLUMN))


class TestBosTokenizer:
    def test_a_sequence_gains_a_leading_token_that_is_never_a_target(self) -> None:
        tokenizer = _bos_tokenizer()

        tokens = tokenizer(packed_batch([1, 2, 3, 4], [3, 1]))

        assert tokens.embeddings.shape == (6, 4)
        assert tokens.cumulative_lens.tolist() == [0, 4, 6]
        assert tokens.is_target.tolist() == [False, True, True, True, False, True]
        assert tokens.timestamps.tolist() == [0, 0, 1, 2, 3, 3]

    def test_the_first_item_of_a_sequence_becomes_predictable(self) -> None:
        """The point of the token. Without it the first item of every history is
        read as context and never has to be produced."""
        tokenizer = _bos_tokenizer()
        tokens = tokenizer(packed_batch([1, 2, 3, 4], [3, 1]))

        pairs = NextItemTargets()(
            {
                "query_repr": torch.zeros(6, 4),
                "item_repr": torch.zeros(6, 4),
                "item_ids": tokens.item_ids,
                "lengths": tokens.cumulative_lens.diff(),
                "is_target": tokens.is_target,
                "timestamps": tokens.timestamps,
            }
        )

        assert pairs.positive_ids.tolist() == [1, 2, 3, 4]

    def test_the_leading_token_is_the_same_vector_everywhere(self) -> None:
        tokenizer = _bos_tokenizer()

        tokens = tokenizer(packed_batch([1, 2, 3, 4], [3, 1]))

        assert torch.equal(tokens.embeddings[0], tokens.embeddings[4])


class TestActionTokenizer:
    def test_each_event_becomes_an_item_token_then_an_action_token(self) -> None:
        tokenizer = ActionTokenizer(
            _embedding(),
            action_embedding=torch.nn.Embedding(3, 4),
            item_id_column=ITEM_COLUMN,
            action_column=ACTION_COLUMN,
        )

        tokens = tokenizer(packed_batch([1, 2, 3], [2, 1], actions=[0, 2, 1]))

        assert tokens.embeddings.shape == (6, 4)
        assert tokens.cumulative_lens.tolist() == [0, 4, 6]
        assert tokens.is_target.tolist() == [True, False] * 3
        assert tokens.item_ids.tolist() == [1, 1, 2, 2, 3, 3]
        assert tokens.timestamps.tolist() == [0, 0, 1, 1, 2, 2]

    def test_the_action_token_carries_the_action_not_the_item(self) -> None:
        item_embedding = _embedding()
        action_embedding = torch.nn.Embedding(3, 4)
        tokenizer = ActionTokenizer(
            item_embedding,
            action_embedding=action_embedding,
            item_id_column=ITEM_COLUMN,
            action_column=ACTION_COLUMN,
        )

        tokens = tokenizer(packed_batch([5], [1], actions=[2]))

        assert torch.equal(tokens.embeddings[0], item_embedding.weight[5])
        assert torch.equal(tokens.embeddings[1], action_embedding.weight[2])


class TestSemanticIdTokenizer:
    def test_each_event_becomes_one_token_per_level(self) -> None:
        tokenizer = SemanticIdTokenizer(
            SemanticIdEmbedding.learned(CODES, num_items=4, embedding_dim=4),
            item_id_column=ITEM_COLUMN,
        )

        tokens = tokenizer(packed_batch([1, 2, 3], [2, 1]))

        assert tokens.embeddings.shape == (6, 4)
        assert tokens.cumulative_lens.tolist() == [0, 4, 6]
        assert tokens.is_target.tolist() == [True, False] * 3
        assert tokens.item_ids.tolist() == [1, 1, 2, 2, 3, 3]

    def test_items_with_a_shared_prefix_share_their_first_token(self) -> None:
        tokenizer = SemanticIdTokenizer(
            SemanticIdEmbedding.learned(CODES, num_items=4, embedding_dim=4),
            item_id_column=ITEM_COLUMN,
        )

        tokens = tokenizer(packed_batch([1, 2, 3], [3]))

        # items 1 and 2 are (0, *); item 3 is (1, *)
        assert torch.equal(tokens.embeddings[0], tokens.embeddings[2])
        assert not torch.equal(tokens.embeddings[0], tokens.embeddings[4])
