from __future__ import annotations

import hashlib
import json
from pathlib import Path

import polars as pl
import pytest

from experiments.g4_future_items.protocol.materialization import (
    CandidateOccurrence,
    CandidatePeriod,
    MaterializationQuery,
    PeriodArtifact,
    ScoredOccurrence,
    ScoredPeriod,
    ScoredQuery,
    SelectorInputPaths,
    materialize_target,
    scan_selector_inputs,
    target_seed,
    write_period_artifact,
)


def _query() -> MaterializationQuery:
    return MaterializationQuery(
        uid=17,
        prefix_timestamp=-5,
        prefix_item_id=9,
        next_item=99,
    )


def _period(
    start: int,
    score: float,
    *occurrences: tuple[int, int],
) -> CandidatePeriod:
    return CandidatePeriod(
        start=start,
        end=start + 100,
        score=score,
        occurrences=tuple(
            CandidateOccurrence(timestamp, item_id)
            for timestamp, item_id in occurrences
        ),
    )


def test_target_seed_matches_canonical_compact_json_contract() -> None:
    query = _query()
    payload = json.dumps(
        [
            "g4-target-v1",
            42,
            3,
            "rq3_learned_hard",
            query.uid,
            query.prefix_timestamp,
            query.prefix_item_id,
        ],
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()

    assert target_seed(query, 42, 3, "rq3_learned_hard") == int.from_bytes(
        hashlib.sha256(payload).digest()[:8], "big"
    )


def test_hard_materialization_ranks_periods_samples_union_and_masks_only_union() -> (
    None
):
    periods = [
        _period(300, 0.4, (310, 30)),
        _period(200, 0.9, (220, 20), (210, 21)),
        _period(100, 0.9, (110, 10)),
        _period(400, 0.0, (410, 40)),
    ]

    result = materialize_target(
        _query(),
        periods,
        objective_id="rq3_learned_hard",
        period_count=2,
        training_seed=42,
        epoch=0,
    )

    assert result.selected_period_starts == (100, 200)
    assert result.target_item_id in {10, 20, 21}
    assert result.acceptable_item_ids == frozenset({10, 20, 21})
    assert not result.used_fallback


def test_proportional_materialization_masks_every_positive_probability_period() -> None:
    periods = [
        _period(100, 0.9, (110, 10)),
        _period(200, 0.2, (210, 20)),
        _period(300, 0.0, (310, 30)),
    ]

    result = materialize_target(
        _query(),
        periods,
        objective_id="rq3_learned_proportional",
        period_count=1,
        training_seed=42,
        epoch=0,
    )

    assert len(result.selected_period_starts) == 1
    assert result.target_item_id in {10, 20}
    assert result.acceptable_item_ids == frozenset({10, 20})


def test_materialization_is_independent_of_period_and_occurrence_traversal() -> None:
    periods = [
        _period(100, 0.8, (130, 13), (110, 11), (120, 12)),
        _period(200, 0.7, (230, 23), (210, 21), (220, 22)),
    ]
    reversed_periods = [
        CandidatePeriod(
            start=period.start,
            end=period.end,
            score=period.score,
            occurrences=tuple(reversed(period.occurrences)),
        )
        for period in reversed(periods)
    ]

    first = materialize_target(
        _query(),
        periods,
        objective_id="rq3_learned_proportional",
        period_count=1,
        training_seed=42,
        epoch=7,
    )
    second = materialize_target(
        _query(),
        reversed_periods,
        objective_id="rq3_learned_proportional",
        period_count=1,
        training_seed=42,
        epoch=7,
    )

    assert first == second


def test_duplicate_occurrences_retain_sampling_mass() -> None:
    periods = [_period(100, 1.0, (110, 10), (110, 10), (120, 20))]
    selected = [
        materialize_target(
            _query(),
            periods,
            objective_id="rq3_deterministic_hard",
            period_count=1,
            training_seed=42,
            epoch=epoch,
        ).target_item_id
        for epoch in range(200)
    ]

    assert selected.count(10) > selected.count(20)


def test_no_positive_period_falls_back_and_masks_only_next_item() -> None:
    result = materialize_target(
        _query(),
        [_period(100, 0.0, (110, 10))],
        objective_id="rq3_deterministic_hard",
        period_count=4,
        training_seed=42,
        epoch=0,
    )

    assert result.target_item_id == 99
    assert result.acceptable_item_ids == frozenset({99})
    assert result.selected_period_starts == ()
    assert result.used_fallback


def test_selector_input_scan_uses_control_likes_and_raw_inner_joined_listens(
    tmp_path: Path,
) -> None:
    control_path = tmp_path / "events_remapped.parquet"
    raw_path = tmp_path / "multi_event.parquet"
    remap_path = tmp_path / "item_id_remap.parquet"
    embeddings_path = tmp_path / "embeddings_compact.parquet"
    pl.DataFrame(
        {
            "uid": [1],
            "item_id": [10],
            "compact_item_id": [1],
            "event_type": ["like"],
            "timestamp": [100],
            "artist_id": [[7]],
            "album_id": [[8]],
        }
    ).write_parquet(control_path)
    pl.DataFrame(
        {
            "uid": [1, 1, 2, 1, 1],
            "item_id": [10, 20, 10, 10, 10],
            "event_type": ["listen", "listen", "listen", "like", "listen"],
            "timestamp": [110, 120, 130, 140, 500],
        }
    ).write_parquet(raw_path)
    pl.DataFrame({"item_id": [10, 20], "compact_id": [1, 2]}).write_parquet(remap_path)
    pl.DataFrame(
        {"compact_id": [1, 2], "normalized_embed": [[1.0], [2.0]]}
    ).write_parquet(embeddings_path)

    frames = scan_selector_inputs(
        SelectorInputPaths(control_path, raw_path, remap_path, embeddings_path),
        start_timestamp=0,
        cutoff_timestamp=200,
    )

    assert frames.likes.collect()["compact_item_id"].to_list() == [1]
    assert frames.listens.collect().to_dicts() == [
        {"uid": 1, "timestamp": 110, "compact_item_id": 1, "artist_id": [7]}
    ]


def test_selector_input_scan_rejects_a_mixed_event_core_cache(tmp_path: Path) -> None:
    control_path = tmp_path / "events_remapped.parquet"
    raw_path = tmp_path / "multi_event.parquet"
    remap_path = tmp_path / "item_id_remap.parquet"
    embeddings_path = tmp_path / "embeddings_compact.parquet"
    pl.DataFrame(
        {
            "uid": [1, 1],
            "item_id": [10, 10],
            "compact_item_id": [1, 1],
            "event_type": ["like", "listen"],
            "timestamp": [100, 110],
            "artist_id": [[7], [7]],
            "album_id": [[8], [8]],
        }
    ).write_parquet(control_path)
    pl.DataFrame(
        {"uid": [1], "item_id": [10], "event_type": ["listen"], "timestamp": [110]}
    ).write_parquet(raw_path)
    pl.DataFrame({"item_id": [10], "compact_id": [1]}).write_parquet(remap_path)
    pl.DataFrame({"compact_id": [1], "normalized_embed": [[1.0]]}).write_parquet(
        embeddings_path
    )

    with pytest.raises(ValueError, match="likes-only core5"):
        scan_selector_inputs(
            SelectorInputPaths(control_path, raw_path, remap_path, embeddings_path),
            start_timestamp=0,
            cutoff_timestamp=200,
        )


def test_period_artifact_is_digest_addressed_verified_and_exactly_lookupable(
    tmp_path: Path,
) -> None:
    exact_score = 0.7500000000000001
    identity = write_period_artifact(
        [
            ScoredQuery(
                uid=1,
                prefix_timestamp=100,
                prefix_item_id=10,
                occurrence_position=0,
                next_item=11,
                fold=2,
                periods=(
                    ScoredPeriod(
                        start=200,
                        end=300,
                        score=exact_score,
                        occurrences=(
                            ScoredOccurrence(220, 20, 3),
                            ScoredOccurrence(210, 21, 2),
                        ),
                    ),
                ),
            ),
            ScoredQuery(
                uid=2,
                prefix_timestamp=110,
                prefix_item_id=12,
                occurrence_position=0,
                next_item=13,
                fold=4,
                periods=(),
            ),
        ],
        selector_kind="learned",
        selected_configuration={"family": "learned"},
        provenance={"input_sha256": "abc"},
        cost={"wall_seconds": 1.0, "peak_aggregate_rss_bytes": 100},
        output_root=tmp_path,
    )

    assert identity.path == tmp_path / identity.sha256
    artifact = PeriodArtifact.open(tmp_path, expected_sha256=identity.sha256)
    query, periods = artifact.lookup(1, 100, 10, 0)
    assert query.next_item == 11
    assert periods == (
        CandidatePeriod(
            200,
            300,
            exact_score,
            (CandidateOccurrence(210, 21), CandidateOccurrence(220, 20)),
        ),
    )
    with pytest.raises(KeyError):
        artifact.lookup(1, 100, 10, 1)


def test_period_artifact_rejects_a_tampered_mmap_array(tmp_path: Path) -> None:
    identity = write_period_artifact(
        [ScoredQuery(1, 100, 10, 0, 11, 2, ())],
        selector_kind="deterministic",
        selected_configuration={"family": "time"},
        provenance={},
        cost={},
        output_root=tmp_path,
    )
    item_path = identity.path / "query_item.bin"
    item_path.write_bytes((99).to_bytes(8, "little", signed=True))

    with pytest.raises(ValueError, match="query_item differs"):
        PeriodArtifact.open(identity.path, expected_sha256=identity.sha256)
