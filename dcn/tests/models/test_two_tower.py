import pytest
import torch

from dcn.data.dataset import collate_sequence_batch
from dcn.models.sequence_targets import NextItemTargets
from dcn.tests.helpers import (
    MODEL_DIM,
    two_tower_loss,
    two_tower_model,
)
from neuralrec.utils import LOSS_DENOMINATOR


pytestmark = pytest.mark.usefixtures("cpu_attention")


def _sequence(item_ids: list[int]) -> dict:
    length = len(item_ids)
    return {
        "int_columns": {
            "compact_item_id": list(item_ids),
            "artist_id": [i % 7 + 1 for i in item_ids],
        },
        "float_columns": {
            "history_count": [float(i) for i in range(length)],
            "item_count": [float(i) * 0.5 for i in range(length)],
        },
        "timestamp": list(range(length)),
    }


def _batch(sequences: list[list[int]] | None = None) -> tuple[dict, int]:
    sequences = (
        sequences if sequences is not None else [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
    )
    total = sum(len(s) for s in sequences)
    return collate_sequence_batch([_sequence(s) for s in sequences]), total


def test_forward_output_shapes_and_ids():
    torch.manual_seed(0)
    model = two_tower_model()
    batch, total = _batch()

    out = model(batch)

    assert out["query_repr"].shape == (total, MODEL_DIM)
    assert out["item_repr"].shape == (total, MODEL_DIM)
    assert out["item_ids"].dtype == torch.long
    assert out["item_ids"].tolist() == [1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert out["lengths"].tolist() == [3, 2, 4]


def test_forward_without_item_counters():
    torch.manual_seed(0)
    model = two_tower_model([])
    batch, total = _batch()

    out = model(batch)

    assert out["item_repr"].shape == (total, MODEL_DIM)


def test_loss_forward_and_backward():
    torch.manual_seed(0)
    model = two_tower_model()
    loss = two_tower_loss(model)
    batch, _ = _batch()

    result = loss(batch)

    assert set(result) == {"loss", "hit_rate", LOSS_DENOMINATOR}
    assert result[LOSS_DENOMINATOR] == 6
    assert result["loss"].ndim == 0 and torch.isfinite(result["loss"])
    assert not result["hit_rate"].requires_grad
    assert 0.0 <= float(result["hit_rate"]) <= 1.0

    result["loss"].backward()

    item_table = model.item_tower.encoder.embedding.unified_embedding.weight
    query_table = model.query_tower.encoder.embedding.unified_embedding.weight
    query_attention = model.query_tower.sequence_model.layers[0].q_proj.weight
    assert item_table is not query_table
    for parameter in (item_table, query_table, query_attention):
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0


def test_all_singleton_sequences_yield_zero_backpropagable_loss():
    torch.manual_seed(0)
    model = two_tower_model()
    loss = two_tower_loss(model)
    batch, _ = _batch([[1], [2], [3]])

    result = loss(batch)

    assert float(result["loss"]) == 0.0
    assert float(result["hit_rate"]) == 0.0
    assert result[LOSS_DENOMINATOR] == 0
    assert not result["hit_rate"].requires_grad

    result["loss"].backward()


class TestNextItemTargets:
    def test_every_token_pairs_with_its_successor(self):
        lengths = torch.tensor([3, 1, 2])
        total = int(lengths.sum())
        query_repr = torch.arange(total, dtype=torch.float32).unsqueeze(1)
        item_repr = query_repr + 100
        item_ids = torch.arange(10, 10 + total)

        pairs = NextItemTargets()(
            {
                "query_repr": query_repr,
                "item_repr": item_repr,
                "item_ids": item_ids,
                "lengths": lengths,
            }
        )

        assert pairs.query_repr.squeeze(1).tolist() == [0, 1, 4]
        assert pairs.positive_repr.squeeze(1).tolist() == [101, 102, 105]
        assert pairs.positive_ids.tolist() == [11, 12, 15]
        assert pairs.group_sizes.tolist() == [2, 1]

    def test_non_target_tokens_are_skipped_over(self):
        lengths = torch.tensor([4])
        query_repr = torch.arange(4, dtype=torch.float32).unsqueeze(1)
        item_repr = query_repr + 100
        item_ids = torch.arange(10, 14)
        # token 2 is an input-only token, so token 1 predicts token 3
        is_target = torch.tensor([True, True, False, True])

        pairs = NextItemTargets()(
            {
                "query_repr": query_repr,
                "item_repr": item_repr,
                "item_ids": item_ids,
                "lengths": lengths,
                "is_target": is_target,
            }
        )

        assert pairs.query_repr.squeeze(1).tolist() == [0, 1, 2]
        assert pairs.positive_ids.tolist() == [11, 13, 13]
        assert pairs.group_sizes.tolist() == [3]

    def test_no_targets_yields_no_pairs(self):
        lengths = torch.tensor([2])
        query_repr = torch.zeros(2, 1)

        pairs = NextItemTargets()(
            {
                "query_repr": query_repr,
                "item_repr": query_repr,
                "item_ids": torch.arange(2),
                "lengths": lengths,
                "is_target": torch.tensor([False, False]),
            }
        )

        assert pairs.query_repr.shape[0] == 0
        assert pairs.group_sizes.numel() == 0
