from __future__ import annotations

import pytest

from experiments.g4_future_items.report.slices import RelevanceEvent, slice_metrics


_CUTOFF = 1_000


def _ranking(first: int, second: int) -> tuple[int, ...]:
    tail = [item_id for item_id in range(1, 121) if item_id not in {first, second}]
    return (first, second, *tail)


def test_slices_follow_occurrence_order_deduplicate_within_bin_and_split_ties() -> None:
    rankings = {
        10: _ranking(1, 2),
        20: _ranking(5, 1),
        30: _ranking(6, 1),
        40: _ranking(7, 1),
    }
    events = [
        RelevanceEvent(10, 1, _CUTOFF + 3_600),
        RelevanceEvent(10, 1, _CUTOFF + 7_200),
        RelevanceEvent(10, 2, _CUTOFF + 7 * 3_600),
        RelevanceEvent(10, 3, _CUTOFF + 2 * 86_400),
        RelevanceEvent(10, 4, _CUTOFF + 4 * 86_400),
        RelevanceEvent(20, 5, _CUTOFF + 1),
        RelevanceEvent(30, 6, _CUTOFF + 2),
        RelevanceEvent(40, 7, _CUTOFF + 3),
    ]

    result = slice_metrics(
        rankings=rankings,
        relevance_events=events,
        training_like_counts={10: 5, 20: 5, 30: 9, 40: 9},
        cutoff_timestamp=_CUTOFF,
        catalog_size=120,
    )

    near = result["slices"]["target_distance_0_6h"]
    assert near["num_users"] == 4
    assert near["num_targets"] == 4
    assert near["metrics"]["recall@10"] == 1.0
    assert near["metrics"]["coverage@10"] == 10 / 120

    first = result["slices"]["target_event_rank_1"]
    ranks_two_to_five = result["slices"]["target_event_rank_2_5"]
    assert first["num_targets"] == 4
    assert ranks_two_to_five["num_targets"] == 4
    assert first["metrics"]["recall@10"] == 1.0
    assert ranks_two_to_five["metrics"]["recall@10"] == 1.0

    assert result["activity_quartiles"] == {
        "user_activity_q1": [10],
        "user_activity_q2": [20],
        "user_activity_q3": [30],
        "user_activity_q4": [40],
    }
    assert result["slices"]["user_activity_q1"]["num_targets"] == 4


def test_slices_reject_context_that_cannot_be_compared() -> None:
    rankings = {10: (1, 2, 3)}
    with pytest.raises(ValueError, match="outside the final seven-day interval"):
        slice_metrics(
            rankings=rankings,
            relevance_events=[RelevanceEvent(10, 1, _CUTOFF)],
            training_like_counts={10: 1},
            cutoff_timestamp=_CUTOFF,
            catalog_size=3,
        )

    with pytest.raises(ValueError, match="duplicate ranked items"):
        slice_metrics(
            rankings={10: (1, 1, 2)},
            relevance_events=[RelevanceEvent(10, 1, _CUTOFF + 1)],
            training_like_counts={10: 1},
            cutoff_timestamp=_CUTOFF,
            catalog_size=3,
        )

    with pytest.raises(ValueError, match="evaluation user sets differ"):
        slice_metrics(
            rankings=rankings,
            relevance_events=[RelevanceEvent(10, 1, _CUTOFF + 1)],
            training_like_counts={20: 1},
            cutoff_timestamp=_CUTOFF,
            catalog_size=3,
        )
