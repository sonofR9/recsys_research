import pytest

from experiments.g3_pretrained_item_embeddings.analysis.slices import (
    compute_ranking_slices,
)


def _evidence() -> tuple[
    dict[int, list[int]],
    dict[int, set[int]],
    dict[int, int],
    dict[int, int],
]:
    rankings = {
        1: [10, 13, 16],
        2: [18, 10, 11],
        3: [13, 14, 15],
        4: [17, 16, 18],
        5: [18, 12, 10],
        6: [16, 15, 14],
    }
    relevant = {
        1: {10, 13, 16},
        2: {10, 11},
        3: {14},
        4: {17},
        5: {12, 18},
        6: {15, 16},
    }
    item_counts = {
        10: 1,
        11: 1,
        12: 2,
        13: 3,
        14: 3,
        15: 4,
        16: 5,
        17: 5,
        18: 6,
    }
    history_lengths = {1: 1, 2: 1, 3: 2, 4: 3, 5: 3, 6: 4}
    return rankings, relevant, item_counts, history_lengths


def test_item_frequency_slices_keep_global_rank_and_exact_denominators() -> None:
    rankings, relevant, item_counts, history_lengths = _evidence()

    report = compute_ranking_slices(
        rankings=rankings,
        relevant_items=relevant,
        training_item_counts=item_counts,
        training_history_lengths=history_lengths,
        cutoffs=(1, 3),
    )

    tail = report.slice("item_frequency", "tail")
    middle = report.slice("item_frequency", "mid")
    head = report.slice("item_frequency", "head")
    assert tail.item_ids == (10, 11, 12)
    assert middle.item_ids == (13, 14, 15)
    assert head.item_ids == (16, 17, 18)
    assert (tail.num_users, tail.num_targets) == (3, 4)
    assert tail.metric("recall@1") == pytest.approx(1 / 3)
    assert tail.metric("recall@3") == 1.0
    assert (middle.num_users, middle.num_targets) == (3, 3)
    assert middle.metric("recall@1") == 0.0
    assert middle.metric("recall@3") == 1.0
    assert (head.num_users, head.num_targets) == (4, 4)
    assert head.metric("recall@1") == 0.75
    assert head.metric("recall@3") == 1.0


def test_history_slices_use_deterministic_user_terciles_and_full_relevance() -> None:
    rankings, relevant, item_counts, history_lengths = _evidence()

    report = compute_ranking_slices(
        rankings=rankings,
        relevant_items=relevant,
        training_item_counts=item_counts,
        training_history_lengths=history_lengths,
        cutoffs=(1, 3),
    )

    low = report.slice("user_history", "low")
    middle = report.slice("user_history", "mid")
    high = report.slice("user_history", "high")
    assert low.user_ids == (1, 2)
    assert middle.user_ids == (3, 4)
    assert high.user_ids == (5, 6)
    assert (low.num_users, low.num_targets) == (2, 5)
    assert low.metric("recall@1") == pytest.approx(1 / 6)
    assert low.metric("recall@3") == 1.0
    assert (middle.num_users, middle.num_targets) == (2, 2)
    assert middle.metric("recall@1") == 0.5
    assert (high.num_users, high.num_targets) == (2, 4)
    assert high.metric("recall@1") == 0.5
    assert low.metric("coverage@1") == pytest.approx(2 / 9)


def test_slice_inputs_fail_closed_on_user_drift_and_duplicate_rankings() -> None:
    rankings, relevant, item_counts, history_lengths = _evidence()
    history_lengths.pop(6)
    with pytest.raises(ValueError, match="user sets"):
        compute_ranking_slices(
            rankings=rankings,
            relevant_items=relevant,
            training_item_counts=item_counts,
            training_history_lengths=history_lengths,
            cutoffs=(1, 3),
        )

    rankings, relevant, item_counts, history_lengths = _evidence()
    rankings[1] = [10, 10, 16]
    with pytest.raises(ValueError, match="duplicate"):
        compute_ranking_slices(
            rankings=rankings,
            relevant_items=relevant,
            training_item_counts=item_counts,
            training_history_lengths=history_lengths,
            cutoffs=(1, 3),
        )
