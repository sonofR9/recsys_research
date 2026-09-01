from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path
import time

import polars as pl
import pytest

from experiments.g4_future_items.launchers.run_selectors import (
    classify_native500m_materialization_cost,
    evaluate_native500m_feasibility,
    native500m_feasibility_trials,
    native500m_feasibility_user_selected,
    prepare_selector_data,
    finalize_materialization_folds,
    run_native500m_feasibility_sample,
    run_materialization_fold,
    SupervisedProcessTreeRssMonitor,
)
from experiments.g4_future_items.protocol.manifest import (
    MATERIALIZATION_COST_LIMITS,
    _validate_materialization_cost_evidence,
)
from experiments.g4_future_items.protocol.materialization import SelectorInputPaths
from experiments.g4_future_items.selectors import ChronologicalBounds, DAY_SECONDS


def _selected(uid: int, percent: int) -> bool:
    payload = json.dumps(["g4-feasibility-v1", uid, 42], separators=(",", ":")).encode()
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return value % (100 // percent) == 0


def _input_paths(tmp_path: Path, user_ids: list[int]) -> SelectorInputPaths:
    control = tmp_path / "events_remapped.parquet"
    raw = tmp_path / "multi_event.parquet"
    remap = tmp_path / "item_id_remap.parquet"
    embeddings = tmp_path / "embeddings_compact.parquet"
    likes = []
    listens = []
    for uid in user_ids:
        for day in range(40):
            item_id = 10 + day % 3
            likes.append(
                {
                    "uid": uid,
                    "item_id": item_id,
                    "compact_item_id": 1 + day % 3,
                    "event_type": "like",
                    "timestamp": day * DAY_SECONDS + 12 * 3_600,
                    "artist_id": [7 + day % 2],
                    "album_id": [20 + day % 2],
                }
            )
            listens.append(
                {
                    "uid": uid,
                    "item_id": item_id,
                    "event_type": "listen",
                    "timestamp": day * DAY_SECONDS + 13 * 3_600,
                }
            )
    pl.DataFrame(likes).write_parquet(control)
    pl.DataFrame(listens).write_parquet(raw)
    pl.DataFrame({"item_id": [10, 11, 12], "compact_id": [1, 2, 3]}).write_parquet(
        remap
    )
    pl.DataFrame(
        {
            "compact_id": [1, 2, 3],
            "normalized_embed": [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
        }
    ).write_parquet(embeddings)
    return SelectorInputPaths(control, raw, remap, embeddings)


def test_feasibility_user_hash_is_exact_and_nested() -> None:
    for uid in range(1, 2_000):
        assert native500m_feasibility_user_selected(uid, 5) == _selected(uid, 5)
        assert native500m_feasibility_user_selected(uid, 10) == _selected(uid, 10)
        if native500m_feasibility_user_selected(uid, 5):
            assert native500m_feasibility_user_selected(uid, 10)
    with pytest.raises(ValueError, match="5 or 10"):
        native500m_feasibility_user_selected(1, 20)


def test_feasibility_selectors_are_fixed_independently_of_quality() -> None:
    deterministic, learned = native500m_feasibility_trials()

    assert deterministic.family == "content"
    assert deterministic.configuration.period_width_seconds == 3_600
    assert deterministic.configuration.lookahead_seconds == 7 * DAY_SECONDS
    assert deterministic.configuration.minimum_liked_events == 1
    assert learned.family == "learned"
    assert learned.configuration.period_width_seconds == 3_600
    assert learned.configuration.lookahead_seconds == 7 * DAY_SECONDS
    assert learned.configuration.minimum_liked_events == 1
    assert learned.configuration.max_leaf_nodes == 31
    assert learned.configuration.learning_rate == 0.05
    assert learned.configuration.l2_regularization == 1e-5


def test_preparation_materializes_only_the_nested_hashed_user_sample(
    tmp_path: Path,
) -> None:
    user_ids = list(range(1, 101))
    paths = _input_paths(tmp_path, user_ids)
    bounds = ChronologicalBounds.from_interval(0, 40 * DAY_SECONDS)

    five = prepare_selector_data(
        paths,
        bounds,
        tmp_path / "five",
        feasibility_percent=5,
    )
    ten = prepare_selector_data(
        paths,
        bounds,
        tmp_path / "ten",
        feasibility_percent=10,
    )

    five_users = set(pl.read_parquet(five.query_path)["uid"].to_list())
    ten_users = set(pl.read_parquet(ten.query_path)["uid"].to_list())
    assert five_users == {uid for uid in user_ids if _selected(uid, 5)}
    assert ten_users == {uid for uid in user_ids if _selected(uid, 10)}
    assert five_users <= ten_users
    assert five.manifest["user_sample"] == {
        "revision": "g4-feasibility-v1",
        "seed": 42,
        "percent": 5,
        "modulus": 20,
        "remainder": 0,
    }
    assert ten.manifest["user_sample"]["percent"] == 10


def test_feasibility_projection_applies_every_preflight_limit() -> None:
    gib = 1024**3
    passing = evaluate_native500m_feasibility(
        {
            "sample_percent": 5,
            "wall_seconds": 1_600.0,
            "peak_rss_bytes": 160 * gib,
            "logical_bytes": 9 * gib,
        },
        {
            "sample_percent": 10,
            "wall_seconds": 3_200.0,
            "peak_rss_bytes": 190 * gib,
            "logical_bytes": 18 * gib,
        },
    )
    assert passing["wall_seconds_projection"] == 32_000.0
    assert passing["logical_bytes_projection"] == 180 * gib
    assert passing["peak_rss_growth"] == pytest.approx(0.1875)
    assert passing["passes"] is True
    assert passing["selection_eligible"] is False

    wall_failure = evaluate_native500m_feasibility(
        passing["measurements"]["5"],
        passing["measurements"]["10"] | {"wall_seconds": 3_600.0},
    )
    assert wall_failure["passes"] is False
    assert "wall_projection" in wall_failure["failed_conditions"]

    rss_failure = evaluate_native500m_feasibility(
        passing["measurements"]["5"],
        passing["measurements"]["10"] | {"peak_rss_bytes": 201 * gib},
    )
    assert rss_failure["passes"] is False
    assert "ten_percent_rss" in rss_failure["failed_conditions"]

    growth_failure = evaluate_native500m_feasibility(
        passing["measurements"]["5"] | {"peak_rss_bytes": 100 * gib},
        passing["measurements"]["10"] | {"peak_rss_bytes": 126 * gib},
    )
    assert growth_failure["passes"] is False
    assert "rss_growth" in growth_failure["failed_conditions"]


def test_five_fold_materializer_has_no_population_frame_accumulator() -> None:
    fold_source = inspect.getsource(run_materialization_fold)
    finalizer_source = inspect.getsource(finalize_materialization_folds)

    assert "_load_partition(" not in fold_source
    assert "_load_partition(" not in finalizer_source
    assert "pl.concat(" not in finalizer_source
    assert "pl.read_parquet(" not in finalizer_source


def test_feasibility_runner_requires_fresh_directory_and_freezes_measurement_path(
    tmp_path: Path,
) -> None:
    occupied = tmp_path / "native500m-5pct"
    occupied.mkdir()
    (occupied / "partial").write_text("immutable")

    with pytest.raises(ValueError, match="empty"):
        run_native500m_feasibility_sample(
            None,
            None,
            5,
            measurement_directory=occupied,
            enforce_reference_fixture=False,
        )

    runner_source = inspect.getsource(run_native500m_feasibility_sample)
    child_source = inspect.getsource(
        __import__(
            "experiments.g4_future_items.launchers.run_selectors",
            fromlist=["_native500m_feasibility_child"],
        )._native500m_feasibility_child
    )
    assert "interval_seconds=0.1" in runner_source
    assert "workers=16" in child_source
    assert "range(5)" in child_source
    assert "native500m_feasibility_trials()" in child_source


def test_full_gate_uses_contention_only_for_one_wall_time_rerun() -> None:
    limits = {
        "wall_seconds": 12 * 60 * 60,
        "peak_aggregate_rss_bytes": 250 * 1024**3,
        "logical_output_scratch_bytes": 250 * 1024**3,
    }
    assert (
        classify_native500m_materialization_cost(
            wall_seconds=100,
            peak_rss_bytes=1,
            logical_bytes=1,
            post_launch_contention=True,
            attempt=1,
            limits=limits,
        )
        == "pass"
    )
    assert (
        classify_native500m_materialization_cost(
            wall_seconds=limits["wall_seconds"] + 1,
            peak_rss_bytes=1,
            logical_bytes=1,
            post_launch_contention=True,
            attempt=1,
            limits=limits,
        )
        == "inconclusive"
    )
    assert (
        classify_native500m_materialization_cost(
            wall_seconds=limits["wall_seconds"] + 1,
            peak_rss_bytes=1,
            logical_bytes=1,
            post_launch_contention=True,
            attempt=2,
            limits=limits,
        )
        == "stop"
    )
    assert (
        classify_native500m_materialization_cost(
            wall_seconds=1,
            peak_rss_bytes=limits["peak_aggregate_rss_bytes"] + 1,
            logical_bytes=1,
            post_launch_contention=True,
            attempt=1,
            limits=limits,
        )
        == "stop"
    )
    with pytest.raises(ValueError, match="1 or 2"):
        classify_native500m_materialization_cost(
            wall_seconds=1,
            peak_rss_bytes=1,
            logical_bytes=1,
            post_launch_contention=False,
            attempt=3,
            limits=limits,
        )


def test_monitor_starts_load_sampling_after_launch_and_tracks_peak_scratch(
    tmp_path: Path,
) -> None:
    monitor = SupervisedProcessTreeRssMonitor(
        interval_seconds=0.01,
        load_interval_seconds=0.01,
        logical_root=tmp_path,
    )
    monitor.start()
    time.sleep(0.03)
    assert monitor.load_samples == []

    scratch = tmp_path / "scratch.bin"
    scratch.write_bytes(b"x" * 4_096)
    monitor.set_root_pid(os.getpid())
    time.sleep(0.03)
    scratch.unlink()
    monitor.stop()

    assert monitor.load_samples
    assert monitor.peak_logical_bytes >= 4_096


def test_passing_native_cost_evidence_ignores_post_launch_contention() -> None:
    evidence = {
        "version": "g4-materialization-cost-v1",
        "measurement_id": "1" * 64,
        "passes": True,
        "decision": "pass",
        "attempt": 1,
        "post_launch_contention": True,
        "deterministic_artifact_sha256": "2" * 64,
        "learned_artifact_sha256": "3" * 64,
        "runtime": {
            "wall_seconds": 1.0,
            "peak_aggregate_rss_bytes": 1,
        },
        "logical_output_scratch_bytes": 1,
        "timed_load_valid": False,
        "limits": MATERIALIZATION_COST_LIMITS,
    }

    _validate_materialization_cost_evidence(evidence)

    legacy = {
        key: value
        for key, value in evidence.items()
        if key
        not in {
            "decision",
            "attempt",
            "post_launch_contention",
        }
    }
    with pytest.raises(ValueError, match="did not pass"):
        _validate_materialization_cost_evidence(legacy)
