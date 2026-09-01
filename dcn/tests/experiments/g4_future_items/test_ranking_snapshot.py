from __future__ import annotations

import torch

from dcn.eval.true_metric import prepare_ranking
from experiments.g4_future_items.protocol.metrics import top_item_rankings


def test_top_item_rankings_reuses_the_exact_evaluable_population_and_seen_mask() -> (
    None
):
    query_user_ids = torch.tensor([10, 20])
    item_ids = torch.tensor([11, 12, 13])
    prepared = prepare_ranking(
        query_user_ids,
        item_ids,
        relevance={10: {12}, 20: {13}},
        train_seen={10: {11}},
        device=torch.device("cpu"),
        user_chunk=1,
        exclude_seen=True,
    )

    result = top_item_rankings(
        query_repr=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        item_repr=torch.tensor([[1.0, 0.1], [0.5, 0.2], [0.0, 1.0]]),
        item_ids=item_ids,
        prepared=prepared,
        max_k=2,
    )

    assert result == {10: (12, 13), 20: (13, 12)}


def test_top_item_rankings_rejects_a_different_catalog_order() -> None:
    item_ids = torch.tensor([11, 12])
    prepared = prepare_ranking(
        torch.tensor([10]),
        item_ids,
        relevance={10: {11}},
        train_seen={},
        device=torch.device("cpu"),
        exclude_seen=False,
    )

    try:
        top_item_rankings(
            query_repr=torch.tensor([[1.0]]),
            item_repr=torch.tensor([[1.0], [0.0]]),
            item_ids=torch.tensor([12, 11]),
            prepared=prepared,
            max_k=2,
        )
    except ValueError as error:
        assert "catalog differs" in str(error)
    else:
        raise AssertionError("catalog mismatch was accepted")
