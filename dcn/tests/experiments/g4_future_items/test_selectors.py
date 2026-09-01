from __future__ import annotations

import hashlib
import json
import math

import numpy as np
import pytest

from experiments.g4_future_items.selectors import (
    CandidateIdentity,
    ChronologicalBounds,
    LikeEvent,
    ListenEvent,
    QueryIdentity,
    SelectorConfiguration,
    SelectorExample,
    build_selector_examples,
    content_similarity,
    cross_fit_learned_selector,
    evaluate_selector,
    fit_learned_selector,
    fit_relevance_threshold,
    fold_for_user,
    paired_user_bootstrap_gate,
    time_similarity,
    weighted_jaccard,
)


DAY = 86_400


def _configuration(family: str = "learned") -> SelectorConfiguration:
    return SelectorConfiguration(
        family=family,
        period_width_seconds=3_600,
        lookahead_seconds=7 * DAY,
        minimum_liked_events=1,
        time_tolerance_seconds=3_600 if family == "time" else None,
        frequency_entity="artist" if family == "frequency" else None,
        max_leaf_nodes=7 if family == "learned" else None,
        learning_rate=0.1 if family == "learned" else None,
        l2_regularization=1e-3 if family == "learned" else None,
    )


def _example(
    *,
    uid: int,
    query_ordinal: int,
    candidate_ordinal: int,
    outcome: float,
    signal: float,
    eligible: bool = True,
) -> SelectorExample:
    query = QueryIdentity(uid, DAY + query_ordinal, 10, query_ordinal)
    candidate = CandidateIdentity(
        uid,
        2 * DAY + candidate_ordinal,
        100 + candidate_ordinal,
        candidate_ordinal,
    )
    return SelectorExample(
        query=query,
        candidate=candidate,
        period_start=2 * DAY,
        period_end=2 * DAY + 3_600,
        eligible=eligible,
        relevance_outcome=outcome,
        circular_time_similarity=signal,
        content_similarity=signal,
        item_jaccard=signal,
        artist_jaccard=signal,
        album_jaccard=signal,
        time_gap_seconds=DAY,
        past_like_count=2,
        candidate_like_count=2,
        trailing_7d_like_count=3,
        trailing_28d_like_count=4,
        trailing_28d_active_days=2,
        prefix_hour_sine=0.0,
        prefix_hour_cosine=1.0,
        prefix_weekday_sine=0.0,
        prefix_weekday_cosine=1.0,
        candidate_hour_sine=0.0,
        candidate_hour_cosine=1.0,
        candidate_weekday_sine=0.0,
        candidate_weekday_cosine=1.0,
    )


def test_chronological_bounds_use_exact_half_open_cut_points() -> None:
    bounds = ChronologicalBounds.from_interval(101, 1_104)

    assert (bounds.train.start, bounds.train.end) == (101, 803)
    assert (bounds.validation.start, bounds.validation.end) == (803, 953)
    assert (bounds.test.start, bounds.test.end) == (953, 1_104)
    assert bounds.partition_at(802).name == "train"
    assert bounds.partition_at(803).name == "validation"
    assert bounds.partition_at(953).name == "test"
    assert bounds.partition_at(1_104) is None


def test_fold_assignment_matches_canonical_sha256_contract() -> None:
    uid = 98_765
    payload = json.dumps(
        ["g4-fold-v1", uid, 42], separators=(",", ":"), ensure_ascii=False
    ).encode()
    expected = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % 5

    assert fold_for_user(uid) == expected
    assert fold_for_user(uid) == fold_for_user(uid)


def test_occurrence_positions_follow_stable_timestamp_source_order() -> None:
    bounds = ChronologicalBounds.from_interval(0, 40 * DAY)
    prefix = 2 * DAY
    likes = [
        LikeEvent(1, prefix, 20),
        LikeEvent(1, prefix, 10),
        LikeEvent(1, prefix + 3_600, 30),
    ]

    examples = build_selector_examples(likes, [], bounds, _configuration("content"))

    query = next(row.query for row in examples if row.query.item_id == 20)
    assert query.occurrence_ordinal == 0
    assert (
        next(
            row.query for row in examples if row.query.item_id == 10
        ).occurrence_ordinal
        == 1
    )


def test_signal_primitives_cover_unknowns_zero_norms_and_utc_ring() -> None:
    assert weighted_jaccard({1: 2.0, 2: 1.0}, {1: 1.0, 3: 3.0}) == pytest.approx(1 / 6)
    assert weighted_jaccard({}, {}) == 0.0
    assert content_similarity([np.array([1.0, 0.0])], [np.array([-1.0, 0.0])]) == 0.0
    assert content_similarity([np.zeros(2)], [np.array([1.0, 0.0])]) == 0.0
    assert time_similarity(0, 7 * DAY - 1) == pytest.approx(1 - 1 / 3_600 / 84)


def test_common_universe_keeps_rows_but_zero_scores_structurally_ineligible_ones() -> (
    None
):
    bounds = ChronologicalBounds.from_interval(0, 40 * DAY)
    prefix = 2 * DAY + 1_800
    likes = [
        LikeEvent(1, prefix - 1_000, 1, (1,), (11,), np.array([1.0, 0.0])),
        LikeEvent(1, prefix, 2, (2,), (12,), np.array([1.0, 0.0])),
        LikeEvent(1, prefix + 600, 3, (2,), (12,), np.array([1.0, 0.0])),
        LikeEvent(1, 2 * DAY + 3_700, 4, (2,), (12,), np.array([1.0, 0.0])),
        LikeEvent(1, prefix + 7 * DAY, 5, (3,), (13,), np.array([0.0, 1.0])),
        LikeEvent(1, prefix + 7 * DAY + 1, 6, (3,), (13,), np.array([0.0, 1.0])),
    ]
    listens = [
        ListenEvent(1, prefix - 10, (7, 0)),
        ListenEvent(1, 2 * DAY + 4_000, (7,)),
    ]

    examples = build_selector_examples(likes, listens, bounds, _configuration("time"))
    query_rows = [row for row in examples if row.query.timestamp == prefix]

    assert [row.candidate.item_id for row in query_rows] == [3, 4, 5, 6]
    assert [row.eligible for row in query_rows] == [False, True, True, False]
    assert query_rows[0].deterministic_score("time") == 0.0
    assert query_rows[1].relevance_outcome == 1.0
    assert query_rows[2].relevance_outcome == 0.0


def test_relevance_threshold_is_nearest_rank_and_uses_strict_comparator() -> None:
    outcomes = [0.0] * 8 + [0.3, 0.9]
    threshold = fit_relevance_threshold(outcomes)

    assert threshold == 0.0
    assert [value > threshold for value in (0.0, 0.3)] == [False, True]
    with pytest.raises(ValueError, match="at least one"):
        fit_relevance_threshold([])


def test_learned_selector_fits_exact_feature_schema_and_zeroes_ineligible_rows() -> (
    None
):
    examples = []
    for uid in range(30):
        for candidate in range(5):
            positive = candidate == 4
            examples.append(
                _example(
                    uid=uid,
                    query_ordinal=uid,
                    candidate_ordinal=candidate,
                    outcome=0.8 if positive else 0.0,
                    signal=1.0 if positive else 0.0,
                    eligible=candidate != 3,
                )
            )

    fitted = fit_learned_selector(examples, _configuration())
    scores = fitted.score(examples)

    assert fitted.relevance_threshold == 0.0
    assert fitted.class_weights == {0: 0.625, 1: 2.5}
    assert fitted.normalizer.means.shape == (6,)
    assert fitted.normalizer.standard_deviations.shape == (6,)
    assert np.all(scores >= 0.0) and np.all(scores <= 1.0)
    assert all(scores[index] == 0.0 for index in range(3, len(scores), 5))
    assert np.mean(scores[4::5]) > np.mean(scores[0::5])


def test_selector_metrics_are_user_balanced_and_keep_zero_positive_queries() -> None:
    examples = [
        _example(uid=1, query_ordinal=1, candidate_ordinal=0, outcome=1.0, signal=1.0),
        _example(uid=1, query_ordinal=1, candidate_ordinal=1, outcome=0.0, signal=0.0),
        _example(uid=1, query_ordinal=2, candidate_ordinal=0, outcome=0.0, signal=1.0),
        _example(uid=2, query_ordinal=1, candidate_ordinal=0, outcome=1.0, signal=0.0),
        _example(uid=2, query_ordinal=1, candidate_ordinal=1, outcome=0.0, signal=1.0),
    ]
    metrics = evaluate_selector(examples, [1, 0, 1, 0, 1], relevance_threshold=0.5)

    expected_uid_1 = (1.0 + 0.0) / 2
    expected_uid_2 = 1 / math.log2(3)
    assert metrics.user_balanced_ndcg_at_10 == pytest.approx(
        (expected_uid_1 + expected_uid_2) / 2
    )
    assert metrics.query_count == 3
    assert metrics.user_count == 2
    assert metrics.positive_count == 2
    assert metrics.negative_count == 3
    assert metrics.auroc == pytest.approx(5 / 12)


def test_paired_bootstrap_gate_uses_query_means_then_user_resampling() -> None:
    examples = []
    learned_scores = []
    deterministic_scores = []
    for uid in range(20):
        examples.extend(
            [
                _example(
                    uid=uid,
                    query_ordinal=uid,
                    candidate_ordinal=0,
                    outcome=1.0,
                    signal=1.0,
                ),
                _example(
                    uid=uid,
                    query_ordinal=uid,
                    candidate_ordinal=1,
                    outcome=0.0,
                    signal=0.0,
                ),
            ]
        )
        learned_scores.extend([1.0, 0.0])
        deterministic_scores.extend([0.0, 1.0])

    gate = paired_user_bootstrap_gate(
        examples,
        learned_scores,
        deterministic_scores,
        relevance_threshold=0.5,
    )

    expected = 1 - 1 / math.log2(3)
    assert gate.user_count == 20
    assert gate.mean_difference == pytest.approx(expected)
    assert gate.lower_95 == pytest.approx(expected)
    assert gate.upper_95 == pytest.approx(expected)
    assert gate.passes


def test_cross_fit_excludes_each_scored_user_fold() -> None:
    examples = []
    for uid in range(50):
        for candidate in range(5):
            positive = candidate == 4
            examples.append(
                _example(
                    uid=uid,
                    query_ordinal=uid,
                    candidate_ordinal=candidate,
                    outcome=1.0 if positive else 0.0,
                    signal=1.0 if positive else 0.0,
                )
            )

    result = cross_fit_learned_selector(examples, _configuration())

    assert len(result.scores) == len(examples)
    assert {artifact.scored_fold for artifact in result.artifacts} == set(range(5))
    for artifact in result.artifacts:
        expected_users = {
            example.query.uid
            for example in examples
            if fold_for_user(example.query.uid) != artifact.scored_fold
        }
        assert artifact.fit_user_ids == frozenset(expected_users)
        assert all(
            fold_for_user(examples[index].query.uid) == artifact.scored_fold
            for index in artifact.scored_indices
        )
