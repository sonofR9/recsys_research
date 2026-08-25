"""Binary-relevance top-k ranking metrics."""

from math import log2
from typing import Iterable

import torch

IdCollection = Iterable[int] | torch.Tensor


def _as_id_list(ids: IdCollection) -> list[int]:
    if isinstance(ids, torch.Tensor):
        return ids.tolist()
    return list(ids)


def _as_id_set(ids: IdCollection) -> set[int]:
    if isinstance(ids, torch.Tensor):
        return set(ids.tolist())
    return set(ids)


def _discount(zero_based_rank: int) -> float:
    return 1.0 / log2(zero_based_rank + 2)


def ndcg_at_k(ranked_ids: IdCollection, relevant_ids: IdCollection, k: int) -> float:
    relevant = _as_id_set(relevant_ids)
    if not relevant:
        return 0.0

    dcg = sum(
        _discount(rank)
        for rank, candidate_id in enumerate(_as_id_list(ranked_ids)[:k])
        if candidate_id in relevant
    )
    idcg = sum(_discount(rank) for rank in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranked_ids: IdCollection, relevant_ids: IdCollection, k: int) -> float:
    relevant = _as_id_set(relevant_ids)
    if not relevant:
        return 0.0

    hits = sum(
        1 for candidate_id in _as_id_list(ranked_ids)[:k] if candidate_id in relevant
    )
    return hits / len(relevant)


def capped_recall_at_k(
    ranked_ids: IdCollection, relevant_ids: IdCollection, k: int
) -> float:
    """Recall over the positives a top-k list can actually hold.

    Dividing by ``min(|relevant|, k)`` rather than ``|relevant|`` stops a heavy
    user being scored against positives no ranking of length k could reach.
    This is how the Yambda benchmark defines recall
    (``yambda_original/yambda/evaluation/metrics.py``).
    """
    relevant = _as_id_set(relevant_ids)
    if not relevant:
        return 0.0

    hits = sum(
        1 for candidate_id in _as_id_list(ranked_ids)[:k] if candidate_id in relevant
    )
    return hits / min(len(relevant), k)


def mrr_at_k(ranked_ids: IdCollection, relevant_ids: IdCollection, k: int) -> float:
    relevant = _as_id_set(relevant_ids)
    if not relevant:
        return 0.0

    for rank, candidate_id in enumerate(_as_id_list(ranked_ids)[:k]):
        if candidate_id in relevant:
            return 1.0 / (rank + 1)
    return 0.0
