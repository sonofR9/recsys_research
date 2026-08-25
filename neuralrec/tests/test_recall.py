import pytest
import torch

from neuralrec.nn.metrics.recall import RecallAtK


class TestRecallAtK:
    def test_perfect_recall(self) -> None:
        metric = RecallAtK(k=3, aggregation="mean")

        user_ids = torch.tensor([0, 0, 0, 0, 0])
        scores = torch.tensor([0.9, 0.8, 0.7, 0.2, 0.1])
        labels = torch.tensor([1, 1, 1, 0, 0])

        recall = metric(user_ids, scores, labels)
        assert recall.item() == pytest.approx(1.0)

    def test_zero_recall(self) -> None:
        metric = RecallAtK(k=2, aggregation="mean")

        user_ids = torch.tensor([0, 0, 0, 0, 0])
        scores = torch.tensor([0.9, 0.8, 0.3, 0.2, 0.1])
        labels = torch.tensor([0, 0, 1, 1, 1])

        recall = metric(user_ids, scores, labels)
        assert recall.item() == pytest.approx(0.0)

    def test_partial_recall(self) -> None:
        metric = RecallAtK(k=3, aggregation="mean")

        user_ids = torch.tensor([0, 0, 0, 0, 0])
        scores = torch.tensor([0.9, 0.8, 0.7, 0.2, 0.1])
        labels = torch.tensor([1, 0, 1, 1, 0])

        recall = metric(user_ids, scores, labels)
        assert recall.item() == pytest.approx(2.0 / 3.0)

    def test_k_larger_than_items(self) -> None:
        metric = RecallAtK(k=100, aggregation="mean")

        user_ids = torch.tensor([0, 0, 0])
        scores = torch.tensor([0.9, 0.5, 0.1])
        labels = torch.tensor([1, 1, 0])

        recall = metric(user_ids, scores, labels)
        assert recall.item() == pytest.approx(1.0)

    def test_no_positives_for_user_skipped(self) -> None:
        metric = RecallAtK(k=2, aggregation="mean")

        user_ids = torch.tensor([0, 0, 1, 1])
        scores = torch.tensor([0.9, 0.8, 0.7, 0.6])
        labels = torch.tensor([1, 1, 0, 0])

        recall = metric(user_ids, scores, labels)
        assert recall.item() == pytest.approx(1.0)

    def test_multiple_users_mean_aggregation(self) -> None:
        metric = RecallAtK(k=2, aggregation="mean")

        user_ids = torch.tensor([0, 0, 0, 1, 1, 1])
        scores = torch.tensor([0.9, 0.8, 0.1, 0.9, 0.1, 0.8])
        labels = torch.tensor([1, 0, 1, 0, 1, 1])

        recall = metric(user_ids, scores, labels)
        expected = (0.5 + 0.5) / 2
        assert recall.item() == pytest.approx(expected)

    def test_sum_aggregation(self) -> None:
        metric = RecallAtK(k=2, aggregation="sum")

        user_ids = torch.tensor([0, 0, 0, 1, 1, 1])
        scores = torch.tensor([0.9, 0.8, 0.1, 0.9, 0.1, 0.8])
        labels = torch.tensor([1, 0, 1, 0, 1, 1])

        recall = metric(user_ids, scores, labels)
        expected = 0.5 + 0.5
        assert recall.item() == pytest.approx(expected)

    def test_none_aggregation_returns_per_user(self) -> None:
        metric = RecallAtK(k=2, aggregation="none")

        user_ids = torch.tensor([0, 0, 0, 1, 1, 1])
        scores = torch.tensor([0.9, 0.8, 0.1, 0.9, 0.1, 0.8])
        labels = torch.tensor([1, 0, 1, 0, 1, 1])

        recall = metric(user_ids, scores, labels)
        assert recall.shape == (2,)
        assert recall[0].item() == pytest.approx(0.5)
        assert recall[1].item() == pytest.approx(0.5)

    def test_empty_result_returns_zero(self) -> None:
        metric = RecallAtK(k=2, aggregation="mean")

        user_ids = torch.tensor([0, 0])
        scores = torch.tensor([0.9, 0.8])
        labels = torch.tensor([0, 0])

        recall = metric(user_ids, scores, labels)
        assert recall.item() == pytest.approx(0.0)

    def test_different_k_values(self) -> None:
        user_ids = torch.tensor([0, 0, 0, 0, 0])
        scores = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5])
        labels = torch.tensor([0, 1, 0, 1, 1])

        metric_k1 = RecallAtK(k=1, aggregation="mean")
        metric_k2 = RecallAtK(k=2, aggregation="mean")
        metric_k3 = RecallAtK(k=3, aggregation="mean")

        assert metric_k1(user_ids, scores, labels).item() == pytest.approx(0.0)
        assert metric_k2(user_ids, scores, labels).item() == pytest.approx(1.0 / 3.0)
        assert metric_k3(user_ids, scores, labels).item() == pytest.approx(1.0 / 3.0)
