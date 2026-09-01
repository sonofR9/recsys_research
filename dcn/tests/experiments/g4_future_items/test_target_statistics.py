from __future__ import annotations

import hashlib
from pathlib import Path

from experiments.g4_future_items.report.target_statistics import (
    TargetEvent,
    candidate_indices,
    verify_target_statistics_evidence,
)
from experiments.g4_future_items.targets import FutureEventIndex


ROOT = Path(__file__).resolve().parents[4]
EXPECTED_NATIVE50M_SHA256 = (
    "00215e07fafeeb0f7a24a644cf44865d539c9b11864dd00bdeb1907a540ba220"
)


def test_candidate_indices_match_the_three_frozen_objectives() -> None:
    events = (
        TargetEvent(timestamp=100, item_id=1),
        TargetEvent(timestamp=100, item_id=2),
        TargetEvent(timestamp=110, item_id=3),
        TargetEvent(timestamp=100_000, item_id=4),
    )

    assert candidate_indices(events, 0, "control_next_item") == ((1,), False)
    assert candidate_indices(events, 0, "rq1_24h") == ((2,), False)
    assert candidate_indices(events, 1, "rq1_24h") == ((2,), False)
    assert candidate_indices(events, 2, "rq1_24h") == ((3,), True)
    assert candidate_indices(events, 0, "rq2_next10") == ((1, 2, 3), False)

    runtime = FutureEventIndex.from_columns(
        user_ids=[7] * len(events),
        timestamps=[event.timestamp for event in events],
        item_ids=[event.item_id for event in events],
    )
    for prefix_position in range(len(events) - 1):
        for objective_id in ("rq1_24h", "rq2_next10"):
            positions, _ = candidate_indices(events, prefix_position, objective_id)
            expected = sorted(
                (events[position].timestamp, events[position].item_id)
                for position in positions
            )
            actual = runtime.candidates(
                uid=7,
                occurrence_position=prefix_position,
                prefix_timestamp=events[prefix_position].timestamp,
                prefix_item_id=events[prefix_position].item_id,
                objective_id=objective_id,
                window_seconds=86400 if objective_id == "rq1_24h" else None,
                event_lookahead=10 if objective_id == "rq2_next10" else None,
                training_cutoff_timestamp=200_000,
            )
            assert [(event.timestamp, event.item_id) for event in actual] == expected


def test_frozen_native_target_statistics_evidence_verifies() -> None:
    artifact = (
        ROOT
        / "experiments/g4_future_items/evidence/target_statistics_native50m_v1.json"
    )

    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == (
        EXPECTED_NATIVE50M_SHA256
    )

    document = verify_target_statistics_evidence(artifact, repo_root=ROOT)

    assert document["population"] == {
        "causal_prefixes": 606267,
        "training_like_events": 614244,
        "training_users": 7977,
    }
    assert document["training_budget"] | {"evidence": {}} == {
        "batch_size": 512,
        "effective_batch_size": 512,
        "target_pairs_per_epoch": 606267,
        "optimizer_steps_per_epoch": 20,
        "evidence": {},
    }
    assert set(document["objectives"]) == {
        "control_next_item",
        "rq1_24h",
        "rq2_next10",
    }
    for objective in document["objectives"].values():
        assert objective["prefix_positive_pairs"] == 606267
        assert objective["post_fallback_empty_prefixes"] == 0
        assert objective["acceptable_unique_items"]["minimum"] >= 1
