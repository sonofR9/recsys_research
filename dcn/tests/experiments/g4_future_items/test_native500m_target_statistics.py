import pytest

from experiments.g4_future_items.report.native500m_target_statistics import (
    TargetEvent,
    next_item_window_population,
    objective_summaries,
)


def test_native500m_target_statistics_match_frozen_objectives() -> None:
    summaries = objective_summaries(
        {
            7: (
                TargetEvent(100, 1),
                TargetEvent(100, 2),
                TargetEvent(110, 3),
                TargetEvent(100_000, 4),
            )
        }
    )

    assert summaries["control_next_item"]["prefix_positive_pairs"] == 3
    assert summaries["control_next_item"]["candidate_occurrences"]["mean"] == 1
    assert summaries["rq1_24h"]["eligible_prefixes"] == 2
    assert summaries["rq1_24h"]["fallback_prefixes"] == 1
    assert summaries["rq1_24h"]["candidate_occurrences"]["mean"] == 1
    assert summaries["rq2_next10"]["candidate_occurrences"]["mean"] == 2
    assert summaries["rq2_next10"]["sampled_target_event_rank"][
        "expected_mean"
    ] == pytest.approx(1.5)


def test_native500m_window_population_matches_bos_end_cls_accounting() -> None:
    population = next_item_window_population(
        [1, 2, 100, 101, 102, 201], max_seq_len=100, min_seq_len=2
    )

    assert population == {
        "training_sequences": 7,
        "bos_next_item_targets": 7,
        "objective_prefix_positive_pairs": 501,
        "training_targets": 508,
        "training_tokens": 522,
    }


def test_native500m_window_population_rejects_different_window_semantics() -> None:
    with pytest.raises(ValueError, match="min_seq_len"):
        next_item_window_population([2], max_seq_len=100, min_seq_len=1)
