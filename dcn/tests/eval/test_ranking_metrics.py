from math import log2

import pytest
import torch

from dcn.eval.ranking_metrics import (
    capped_recall_at_k,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)

RANKED = [0, 1, 2, 3, 4]


class TestNdcg:
    def test_relevant_at_ranks_1_and_3(self):
        relevant = {RANKED[0], RANKED[2]}
        dcg = 1 / log2(2) + 1 / log2(4)
        idcg = 1 / log2(2) + 1 / log2(3)
        assert ndcg_at_k(RANKED, relevant, 5) == pytest.approx(dcg / idcg)
        assert ndcg_at_k(RANKED, relevant, 5) == pytest.approx(1.5 / 1.6309297535)

    def test_perfect_ranking_is_one(self):
        relevant = {RANKED[0], RANKED[1]}
        assert ndcg_at_k(RANKED, relevant, 5) == pytest.approx(1.0)
        assert ndcg_at_k(RANKED, relevant, 2) == pytest.approx(1.0)

    def test_no_relevant_is_zero(self):
        assert ndcg_at_k(RANKED, set(), 5) == 0.0
        assert ndcg_at_k(RANKED, {99}, 5) == 0.0

    def test_k_larger_than_list(self):
        relevant = {RANKED[0], RANKED[2]}
        dcg = 1 / log2(2) + 1 / log2(4)
        idcg = 1 / log2(2) + 1 / log2(3)
        assert ndcg_at_k(RANKED, relevant, 100) == pytest.approx(dcg / idcg)

    def test_idcg_counts_only_reachable_slots(self):
        # 3 relevant but k=1: the single top slot is a hit, so NDCG@1 = 1.
        relevant = {RANKED[0], RANKED[1], RANKED[2]}
        assert ndcg_at_k(RANKED, relevant, 1) == pytest.approx(1.0)

    def test_accepts_tensor_inputs(self):
        relevant = torch.tensor([0, 2])
        ranked = torch.tensor(RANKED)
        dcg = 1 / log2(2) + 1 / log2(4)
        idcg = 1 / log2(2) + 1 / log2(3)
        assert ndcg_at_k(ranked, relevant, 5) == pytest.approx(dcg / idcg)


class TestRecall:
    def test_full_recall(self):
        assert recall_at_k(RANKED, {0, 2}, 5) == pytest.approx(1.0)

    def test_partial_recall_at_small_k(self):
        assert recall_at_k(RANKED, {0, 2}, 2) == pytest.approx(0.5)
        assert recall_at_k(RANKED, {0, 2}, 1) == pytest.approx(0.5)

    def test_no_relevant_is_zero(self):
        assert recall_at_k(RANKED, set(), 5) == 0.0


class TestCappedRecall:
    def test_matches_plain_recall_when_k_exceeds_relevant_count(self):
        assert capped_recall_at_k(RANKED, {0, 2}, 5) == pytest.approx(1.0)
        assert capped_recall_at_k(RANKED, {0, 2}, 2) == pytest.approx(0.5)

    def test_unreachable_positives_do_not_count_against_the_user(self):
        # 3 relevant but only 1 slot: plain recall caps at 1/3, capped at 1.
        assert recall_at_k(RANKED, {0, 1, 2}, 1) == pytest.approx(1 / 3)
        assert capped_recall_at_k(RANKED, {0, 1, 2}, 1) == pytest.approx(1.0)

    def test_partial_hits_within_a_truncated_denominator(self):
        assert capped_recall_at_k(RANKED, {0, 3, 4}, 2) == pytest.approx(0.5)

    def test_no_relevant_is_zero(self):
        assert capped_recall_at_k(RANKED, set(), 5) == 0.0


class TestMrr:
    def test_first_relevant_rank_one(self):
        assert mrr_at_k(RANKED, {0, 2}, 5) == pytest.approx(1.0)

    def test_first_relevant_rank_three(self):
        assert mrr_at_k(RANKED, {2, 4}, 5) == pytest.approx(1 / 3)

    def test_none_within_k_is_zero(self):
        assert mrr_at_k(RANKED, {2, 4}, 2) == 0.0

    def test_no_relevant_is_zero(self):
        assert mrr_at_k(RANKED, set(), 5) == 0.0
