from __future__ import annotations

import dataclasses
import hashlib
import json

import numpy as np

from experiments.g4_future_items.selectors import (
    DAY_SECONDS,
    ChronologicalBounds,
    LikeEvent,
    ListenEvent,
    SelectorConfiguration,
    build_selector_examples,
)


def test_duplicate_timestamp_examples_match_uncached_reference() -> None:
    day = DAY_SECONDS
    likes = (
        LikeEvent(7, 2 * day + 10, 1, (11,), (21,), np.array([1.0, 0.0, 0.0])),
        LikeEvent(7, 3 * day + 20, 2, (12,), (22,), np.array([0.0, 1.0, 0.0])),
        LikeEvent(7, 4 * day + 30, 3, (11,), (23,), np.array([0.0, 0.0, 1.0])),
        LikeEvent(7, 4 * day + 30, 4, (13,), (24,), np.array([1.0, 1.0, 0.0])),
        LikeEvent(7, 4 * day + 30, 5, (13,), (24,), np.array([1.0, 0.0, 1.0])),
        LikeEvent(7, 4 * day + 30, 6, (14,), (25,), np.array([0.0, 1.0, 1.0])),
        LikeEvent(7, 5 * day + 40, 7, (11,), (21,), np.array([-1.0, 0.0, 0.0])),
    )
    listens = (
        ListenEvent(7, 2 * day + 30, (11,)),
        ListenEvent(7, 3 * day + 30, (12,)),
        ListenEvent(7, 4 * day + 40, (13,)),
        ListenEvent(7, 5 * day + 30, (11,)),
    )

    examples = build_selector_examples(
        likes,
        listens,
        ChronologicalBounds.from_interval(0, 60 * day),
        SelectorConfiguration("content", day, 28 * day, 1),
    )
    payload = json.dumps(
        [dataclasses.asdict(example) for example in examples],
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()

    assert len(examples) == 15
    assert hashlib.sha256(payload).hexdigest() == (
        "d6cbb93493af12909a9ea8399288fcf30a9d9c059b95cdb7c5461c67e67dcfcc"
    )


def test_representative_duplicate_burst_matches_uncached_reference() -> None:
    day = DAY_SECONDS
    likes = []
    for index in range(30):
        embedding = np.zeros(32, dtype=np.float64)
        embedding[index % 32] = 1.0
        likes.append(
            LikeEvent(
                9,
                (2 + index) * day + index,
                index + 1,
                (index % 11 + 1,),
                (index % 7 + 1,),
                embedding,
            )
        )
    burst_timestamp = 33 * day + 123
    for index in range(130):
        embedding = np.zeros(32, dtype=np.float64)
        embedding[index % 32] = 1.0
        likes.append(
            LikeEvent(
                9,
                burst_timestamp,
                1000 + index,
                (index % 11 + 1,),
                (index % 7 + 1,),
                embedding,
            )
        )
    listens = tuple(
        ListenEvent(9, (2 + index) * day + 50, (index % 11 + 1,)) for index in range(32)
    )

    examples = build_selector_examples(
        tuple(likes),
        listens,
        ChronologicalBounds.from_interval(0, 60 * day),
        SelectorConfiguration("content", day, 28 * day, 1),
    )
    payload = json.dumps(
        [dataclasses.asdict(example) for example in examples],
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()

    assert len(examples) == 3812
    assert hashlib.sha256(payload).hexdigest() == (
        "f31aad7809d99d40a10d2cf696047473186cd101064ff5b171ab146723aa7e0a"
    )
