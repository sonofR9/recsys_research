from __future__ import annotations

import torch

from dcn.eval.true_metric import PreparedRanking


@torch.inference_mode()
def top_item_rankings(
    *,
    query_repr: torch.Tensor,
    item_repr: torch.Tensor,
    item_ids: torch.Tensor,
    prepared: PreparedRanking,
    max_k: int,
) -> dict[int, tuple[int, ...]]:
    if query_repr.ndim != 2 or item_repr.ndim != 2:
        raise ValueError("query and item representations must be matrices")
    if query_repr.shape[1] != item_repr.shape[1]:
        raise ValueError("query and item representation widths differ")
    if item_ids.ndim != 1 or item_ids.shape[0] != item_repr.shape[0]:
        raise ValueError("item ids must align with item representations")
    if item_ids.tolist() != prepared.item_id_list:
        raise ValueError("ranked catalog differs from the prepared evaluation catalog")
    if max_k < 1 or max_k > item_ids.shape[0]:
        raise ValueError("max_k must fit inside the ranked catalog")

    device = item_repr.device
    query_repr = query_repr.to(device)
    item_ids = item_ids.to(device)
    result: dict[int, tuple[int, ...]] = {}
    user_offset = 0
    for query_rows, (seen_rows, seen_columns), _ in prepared.chunks():
        query_rows = query_rows.to(device)
        scores = query_repr[query_rows] @ item_repr.t()
        scores[seen_rows.to(device), seen_columns.to(device)] = float("-inf")
        top_positions = torch.topk(scores, max_k, dim=1).indices
        ranked_ids = item_ids[top_positions].cpu().tolist()
        chunk_users = prepared.evaluable[user_offset : user_offset + len(ranked_ids)]
        for user, ranking in zip(chunk_users, ranked_ids, strict=True):
            result[user.user_id] = tuple(map(int, ranking))
        user_offset += len(ranked_ids)

    if user_offset != len(prepared.evaluable):
        raise ValueError("prepared evaluation chunks do not cover every user")
    return result
