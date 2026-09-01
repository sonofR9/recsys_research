from math import log2
from pathlib import Path

import polars as pl
import pytest
import torch

from dcn.eval.true_metric import build_interaction_sets, evaluate_true_ndcg


def _rigged_inputs():
    item_ids = torch.tensor([10, 20, 30])
    item_repr = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    # user 0: top item is its only relevant one -> perfect.
    # user 1: relevant item (30) lands at rank 2.
    # user 2: a higher-scoring item (10) is train-seen -> masked -> relevant top.
    # user 3: no relevance -> skipped. user 4: relevant item is train-seen -> skipped.
    query_repr = torch.tensor(
        [[2.0, 0.0], [0.0, 2.0], [3.0, 0.0], [1.0, 1.0], [5.0, 5.0]]
    )
    query_user_ids = torch.tensor([0, 1, 2, 3, 4])
    relevance = {0: {10}, 1: {30}, 2: {30}, 3: set(), 4: {10}}
    train_seen = {2: {10}, 4: {10}}
    return query_repr, query_user_ids, item_repr, item_ids, relevance, train_seen


class TestEvaluateTrueNdcg:
    def test_hand_computed_metrics(self):
        query, user_ids, items, item_ids, relevance, seen = _rigged_inputs()
        out = evaluate_true_ndcg(
            query, user_ids, items, item_ids, relevance, seen, ks=[1, 2, 3]
        )

        assert out["num_users"] == 3.0

        inv_log3 = 1.0 / log2(3)
        # ndcg@1: user0=1, user1=0 (relevant at rank 2), user2=1.
        assert out["ndcg@1"] == pytest.approx(2 / 3)
        # ndcg@2: user0=1, user1=1/log2(3) (single relevant -> idcg=1), user2=1.
        assert out["ndcg@2"] == pytest.approx((2 + inv_log3) / 3)
        assert out["ndcg@3"] == pytest.approx((2 + inv_log3) / 3)

        assert out["recall@1"] == pytest.approx(2 / 3)
        assert out["recall@2"] == pytest.approx(1.0)

        assert out["mrr@1"] == pytest.approx(2 / 3)
        assert out["mrr@2"] == pytest.approx((1 + 0.5 + 1) / 3)

    def test_masking_changes_winner(self):
        query, user_ids, items, item_ids, _, _ = _rigged_inputs()
        masked = evaluate_true_ndcg(
            query[2:3], user_ids[2:3], items, item_ids, {2: {30}}, {2: {10}}, ks=[1]
        )
        assert masked["num_users"] == 1.0
        assert masked["ndcg@1"] == pytest.approx(1.0)

        unmasked = evaluate_true_ndcg(
            query[2:3], user_ids[2:3], items, item_ids, {2: {30}}, {}, ks=[1]
        )
        assert unmasked["ndcg@1"] == pytest.approx(0.0)

    def test_homework_mode_keeps_seen_targets_and_does_not_mask_them(self):
        query, user_ids, items, item_ids, _, _ = _rigged_inputs()
        out = evaluate_true_ndcg(
            query[4:5],
            user_ids[4:5],
            items,
            item_ids,
            relevance={4: {10}},
            train_seen={4: {10}},
            ks=[1],
            exclude_seen=False,
        )

        assert out["num_users"] == 1.0
        assert out["recall@1"] == pytest.approx(1.0)

    def test_out_of_catalog_relevance_skips_user(self):
        query, user_ids, items, item_ids, _, _ = _rigged_inputs()
        out = evaluate_true_ndcg(
            query[0:1], user_ids[0:1], items, item_ids, {0: {99}}, {}, ks=[1]
        )
        assert out["num_users"] == 0.0
        assert out["ndcg@1"] == 0.0

    def test_chunking_does_not_change_results(self):
        query, user_ids, items, item_ids, relevance, seen = _rigged_inputs()
        whole = evaluate_true_ndcg(
            query, user_ids, items, item_ids, relevance, seen, ks=[1, 2]
        )
        per_user = evaluate_true_ndcg(
            query, user_ids, items, item_ids, relevance, seen, ks=[1, 2], user_chunk=1
        )
        assert per_user == pytest.approx(whole)

    def test_can_return_each_relevant_items_rank_without_changing_metrics(self):
        query, user_ids, items, item_ids, relevance, seen = _rigged_inputs()

        metrics, ranks = evaluate_true_ndcg(
            query,
            user_ids,
            items,
            item_ids,
            relevance,
            seen,
            ks=[1, 2],
            return_relevant_ranks=True,
        )

        assert metrics == pytest.approx(
            evaluate_true_ndcg(
                query, user_ids, items, item_ids, relevance, seen, ks=[1, 2]
            )
        )
        assert ranks.tolist() == [1, 2, 1]

    def test_can_return_ranks_and_top_items_from_the_same_pass(self):
        query, user_ids, items, item_ids, relevance, seen = _rigged_inputs()

        metrics, details = evaluate_true_ndcg(
            query,
            user_ids,
            items,
            item_ids,
            relevance,
            seen,
            ks=[1, 2],
            return_ranking_details=True,
        )

        assert metrics == pytest.approx(
            evaluate_true_ndcg(
                query, user_ids, items, item_ids, relevance, seen, ks=[1, 2]
            )
        )
        assert details.relevant_ranks.tolist() == [1, 2, 1]
        assert details.top_item_ids.tolist() == [
            [10, 30],
            [20, 30],
            [30, 20],
        ]

    def test_ranking_detail_modes_are_mutually_exclusive(self):
        query, user_ids, items, item_ids, relevance, seen = _rigged_inputs()

        with pytest.raises(ValueError, match="mutually exclusive"):
            evaluate_true_ndcg(
                query,
                user_ids,
                items,
                item_ids,
                relevance,
                seen,
                ks=[1],
                return_relevant_ranks=True,
                return_ranking_details=True,
            )


class TestCoverage:
    def test_full_catalog_covered(self):
        query, user_ids, items, item_ids, relevance, seen = _rigged_inputs()
        out = evaluate_true_ndcg(
            query, user_ids, items, item_ids, relevance, seen, ks=[1]
        )
        # The three evaluable users each rank a different item first.
        assert out["coverage@1"] == pytest.approx(1.0)

    def test_one_item_for_everyone_covers_a_third(self):
        _, _, items, item_ids, _, _ = _rigged_inputs()
        same_query = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
        out = evaluate_true_ndcg(
            same_query,
            torch.tensor([0, 1, 2]),
            items,
            item_ids,
            relevance={0: {30}, 1: {30}, 2: {30}},
            train_seen={},
            ks=[1, 2],
        )
        assert out["coverage@1"] == pytest.approx(1 / 3)
        assert out["coverage@2"] == pytest.approx(2 / 3)


class TestCappedRecallMetric:
    def test_denominator_stops_at_k(self):
        _, _, items, item_ids, _, _ = _rigged_inputs()
        out = evaluate_true_ndcg(
            torch.tensor([[1.0, 0.0]]),
            torch.tensor([0]),
            items,
            item_ids,
            relevance={0: {10, 20}},
            train_seen={},
            ks=[1],
        )
        assert out["recall@1"] == pytest.approx(0.5)
        assert out["capped_recall@1"] == pytest.approx(1.0)


class TestSemanticIdRecall:
    def test_wrong_item_with_the_target_prefix_is_a_prefix_hit_only(self):
        out = evaluate_true_ndcg(
            query_repr=torch.tensor([[1.0]]),
            query_user_ids=torch.tensor([0]),
            item_repr=torch.tensor([[3.0], [2.0], [1.0]]),
            item_ids=torch.tensor([10, 20, 30]),
            relevance={0: {30}},
            train_seen={},
            ks=[1, 2, 3],
            item_semantic_codes=torch.tensor([[0, 0], [1, 1], [1, 0]]),
        )

        assert out["sid_prefix_recall@1_l1"] == 0.0
        assert out["sid_prefix_recall@2_l1"] == 1.0
        assert out["sid_exact_recall@2"] == 0.0
        assert out["sid_exact_recall@3"] == 1.0
        assert out["sid_prefix_recall@3_l2"] == 1.0

    def test_semantic_metrics_are_absent_without_codes(self):
        query, user_ids, items, item_ids, relevance, seen = _rigged_inputs()
        out = evaluate_true_ndcg(
            query, user_ids, items, item_ids, relevance, seen, ks=[1]
        )

        assert not any(name.startswith("sid_") for name in out)


class TestUserSubsampling:
    @staticmethod
    def _many_users(count: int):
        """Every user ranks item 10 first; only the even ones wanted it.

        So a metric average pins down *which* users were scored, not just how
        many.
        """
        item_ids = torch.tensor([10, 20, 30])
        item_repr = torch.eye(3)
        query_repr = torch.tensor([[1.0, 0.0, 0.0]]).repeat(count, 1)
        query_user_ids = torch.arange(count)
        relevance = {
            user_id: {10 if user_id % 2 == 0 else 20} for user_id in range(count)
        }
        return query_repr, query_user_ids, item_repr, item_ids, relevance

    def test_caps_the_number_of_scored_users(self):
        query, user_ids, items, item_ids, relevance = self._many_users(50)
        out = evaluate_true_ndcg(
            query, user_ids, items, item_ids, relevance, {}, ks=[1], max_users=10
        )
        assert out["num_users"] == 10.0

    def test_below_the_cap_every_user_is_scored(self):
        query, user_ids, items, item_ids, relevance = self._many_users(7)
        out = evaluate_true_ndcg(
            query, user_ids, items, item_ids, relevance, {}, ks=[1], max_users=10
        )
        assert out["num_users"] == 7.0

    def test_the_same_population_is_sampled_the_same_way_twice(self):
        query, user_ids, items, item_ids, relevance = self._many_users(50)
        arguments = (query, user_ids, items, item_ids, relevance, {})

        first = evaluate_true_ndcg(*arguments, ks=[1], max_users=10)
        second = evaluate_true_ndcg(*arguments, ks=[1], max_users=10)

        assert first == pytest.approx(second)

    def test_same_users_regardless_of_the_order_they_arrive_in(self):
        query, user_ids, items, item_ids, relevance = self._many_users(50)
        shuffle = torch.randperm(50)

        in_order = evaluate_true_ndcg(
            query, user_ids, items, item_ids, relevance, {}, ks=[1], max_users=10
        )
        shuffled = evaluate_true_ndcg(
            query[shuffle],
            user_ids[shuffle],
            items,
            item_ids,
            relevance,
            {},
            ks=[1],
            max_users=10,
        )
        assert shuffled == pytest.approx(in_order)


class TestBuildInteractionSets:
    def test_deduplicates_across_files(self, tmp_path: Path):
        pl.DataFrame({"uid": [1, 1, 1], "item": [10, 10, 11]}).write_parquet(
            tmp_path / "day1.parquet"
        )
        pl.DataFrame({"uid": [1, 2], "item": [10, 12]}).write_parquet(
            tmp_path / "day2.parquet"
        )

        sets = build_interaction_sets(
            [tmp_path / "day1.parquet", tmp_path / "day2.parquet"],
            user_column="uid",
            item_id_column="item",
        )
        assert sets == {1: {10, 11}, 2: {12}}

    def test_row_filter_keeps_only_matching_events(self, tmp_path: Path):
        pl.DataFrame(
            {"uid": [1, 1, 2], "item": [10, 11, 12], "event_type_id": [1, 2, 2]}
        ).write_parquet(tmp_path / "day.parquet")

        sets = build_interaction_sets(
            [tmp_path / "day.parquet"],
            user_column="uid",
            item_id_column="item",
            row_filter=pl.col("event_type_id") == 1,
        )
        assert sets == {1: {10}}
