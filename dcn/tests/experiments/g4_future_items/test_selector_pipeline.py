from __future__ import annotations

import hashlib
import json
import multiprocessing
import time
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import experiments.g4_future_items.launchers.run_selectors as selector_launcher

from experiments.g4_future_items.configs.selectors import (
    SelectorTrial,
    SelectorTrialResult,
    compile_learned_boundary,
    compile_selector_search,
    select_family_winner,
    select_strongest_deterministic,
)
from experiments.g4_future_items.launchers.run_selectors import (
    finalize_materialization_folds,
    all_input_pages_resident,
    load_gate_result,
    load_search_result,
    open_prepared_selector_data,
    prepare_selector_data,
    prewarm_and_verify_inputs,
    run_materialization_fold,
    run_search_trial,
    SelectorQueueExperiment,
    SupervisedProcessTreeRssMonitor,
)
from experiments.g4_future_items.protocol.materialization import (
    PeriodArtifact,
    SelectorInputPaths,
)
from experiments.g4_future_items.selectors import (
    DAY_SECONDS,
    ChronologicalBounds,
    SelectorMetrics,
    SelectorConfiguration,
    fold_for_user,
)


def _input_paths(tmp_path: Path) -> SelectorInputPaths:
    control = tmp_path / "events_remapped.parquet"
    raw = tmp_path / "multi_event.parquet"
    remap = tmp_path / "item_id_remap.parquet"
    embeddings = tmp_path / "embeddings_compact.parquet"
    timestamps = [day * DAY_SECONDS + 12 * 3_600 for day in range(40)]
    items = [10 + day % 3 for day in range(40)]
    pl.DataFrame(
        {
            "uid": [1] * 40,
            "item_id": items,
            "compact_item_id": [1 + day % 3 for day in range(40)],
            "event_type": ["like"] * 40,
            "timestamp": timestamps,
            "artist_id": [[7 + day % 2] for day in range(40)],
            "album_id": [[20 + day % 2] for day in range(40)],
        }
    ).write_parquet(control)
    pl.DataFrame(
        {
            "uid": [1] * 80,
            "item_id": [10 + day % 3 for day in range(40) for _ in range(2)],
            "event_type": ["listen"] * 80,
            "timestamp": [
                day * DAY_SECONDS + hour * 3_600
                for day in range(40)
                for hour in (11, 13)
            ],
        }
    ).write_parquet(raw)
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


def test_selector_search_compiles_exact_equal_family_budgets_without_duplicates() -> (
    None
):
    trials = compile_selector_search()

    assert len(trials) == 48
    assert {
        family: sum(trial.family == family for trial in trials)
        for family in ("time", "content", "frequency", "learned")
    } == {"time": 12, "content": 12, "frequency": 12, "learned": 12}
    for family in ("time", "content", "frequency", "learned"):
        configurations = [
            json.dumps(trial.to_dict(), sort_keys=True)
            for trial in trials
            if trial.family == family
        ]
        assert len(configurations) == len(set(configurations))


def test_learned_boundary_uses_full_expanded_interval_and_keeps_job_seed() -> None:
    entering = next(
        trial for trial in compile_selector_search() if trial.family == "learned"
    )
    configuration = entering.configuration
    low_entering = type(entering)(
        family="learned",
        trial_id=entering.trial_id,
        configuration=type(configuration)(
            family="learned",
            period_width_seconds=configuration.period_width_seconds,
            lookahead_seconds=configuration.lookahead_seconds,
            minimum_liked_events=configuration.minimum_liked_events,
            max_leaf_nodes=configuration.max_leaf_nodes,
            learning_rate=0.01,
            l2_regularization=configuration.l2_regularization,
        ),
    )

    boundary = compile_learned_boundary(low_entering, boundary_round=1)

    assert len(boundary) == 4
    assert {trial.seed for trial in boundary} == {42}
    assert {trial.sampler_seed for trial in boundary} == {43}
    assert all(
        0.01 / 4 <= trial.configuration.learning_rate <= 0.2 for trial in boundary
    )
    assert all(
        trial.configuration.max_leaf_nodes == configuration.max_leaf_nodes
        and trial.configuration.l2_regularization == configuration.l2_regularization
        for trial in boundary
    )


def test_selector_ndcg_tolerance_is_relative_to_the_actual_maximum() -> None:
    trials = [
        trial for trial in compile_selector_search() if trial.family == "content"
    ][:2]
    metrics = [
        SelectorMetrics(0.1000000000004, 0.8, 1, 1, 2, 1, 1, 0.5),
        SelectorMetrics(0.1000000000013, 0.7, 1, 1, 2, 1, 1, 0.5),
    ]
    results = [
        SelectorTrialResult(trial, metric, f"sha-{index}")
        for index, (trial, metric) in enumerate(zip(trials, metrics))
    ]

    assert select_family_winner(results) == results[0]


def test_cross_family_tie_uses_canonical_parameters_not_family_simplicity() -> None:
    time = SelectorTrial(
        "time",
        1,
        SelectorConfiguration("time", DAY_SECONDS, 14 * DAY_SECONDS, 1, 0),
    )
    frequency = SelectorTrial(
        "frequency",
        1,
        SelectorConfiguration(
            "frequency",
            DAY_SECONDS,
            14 * DAY_SECONDS,
            1,
            frequency_entity="album",
        ),
    )
    metrics = SelectorMetrics(0.5, 0.7, 1, 1, 2, 1, 1, 0.5)
    results = [
        SelectorTrialResult(time, metrics, "time"),
        SelectorTrialResult(frequency, metrics, "frequency"),
    ]

    assert select_strongest_deterministic(results) == results[1]


def test_preparation_streams_width_rows_and_preserves_every_control_prefix(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("G4_COMPILED_JOB_B64", "invalid-in-spawn-worker")
    bounds = ChronologicalBounds.from_interval(0, 40 * DAY_SECONDS)
    prepared = prepare_selector_data(
        _input_paths(tmp_path), bounds, tmp_path / "prepared", workers=2
    )
    reopened = open_prepared_selector_data(tmp_path / "prepared", prepared.sha256)
    repeated = prepare_selector_data(
        _input_paths(tmp_path), bounds, tmp_path / "prepared_repeated"
    )

    assert reopened.manifest["pair_budget"] == {
        "control_prefixes": 39,
        "selector_common_queries": reopened.manifest["pair_budget"][
            "selector_common_queries"
        ],
        "next_item_fallback_queries": 39
        - reopened.manifest["pair_budget"]["selector_common_queries"],
    }
    assert reopened.manifest["pair_budget"]["selector_common_queries"] < 39
    assert all(
        reopened.manifest["widths"][str(width)]["queries"]
        == reopened.manifest["pair_budget"]["selector_common_queries"]
        for width in (3_600, 21_600, DAY_SECONDS)
    )
    query_frame = pl.read_parquet(reopened.query_path)
    assert query_frame.height == 39
    assert query_frame["occurrence_position"].to_list() == list(range(39))
    assert repeated.semantics_sha256 == prepared.semantics_sha256


def test_one_compiled_deterministic_trial_runs_from_columnar_artifact(
    tmp_path: Path,
) -> None:
    bounds = ChronologicalBounds.from_interval(0, 40 * DAY_SECONDS)
    prepared = prepare_selector_data(
        _input_paths(tmp_path), bounds, tmp_path / "prepared"
    )
    trial = next(
        trial for trial in compile_selector_search() if trial.family == "content"
    )

    result = run_search_trial(prepared, trial, tmp_path / "search")

    assert result.metrics.query_count > 0
    assert result.metrics.pair_count > 0
    assert result.artifact_path == tmp_path / "search" / result.artifact_sha256
    assert (result.artifact_path / "artifact.json").is_file()
    assert not (result.artifact_path / "model.pkl").exists()

    result_path = result.artifact_path / "result.json"
    original_result = result_path.read_bytes()
    tampered_result = json.loads(original_result)
    tampered_result["validation_metrics"]["ndcg_at_10"] = 1.0
    result_path.write_text(json.dumps(tampered_result))
    with pytest.raises(ValueError, match="hashed artifact"):
        load_search_result(tmp_path / "search", result.artifact_sha256)

    result_path.write_bytes(original_result)
    artifact_path = result.artifact_path / "artifact.json"
    tampered_artifact = json.loads(artifact_path.read_bytes())
    tampered_artifact["validation_metrics"]["ndcg_at_10"] = 1.0
    tampered_content = json.dumps(
        tampered_artifact,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    artifact_path.write_bytes(tampered_content)
    (result.artifact_path / "artifact.sha256").write_text(
        hashlib.sha256(tampered_content).hexdigest()
    )
    with pytest.raises(ValueError, match="frozen payload"):
        load_search_result(
            tmp_path / "search",
            result.artifact_sha256,
            expected_payload_sha256=result.artifact_payload_sha256,
        )


def test_gate_payload_is_bound_before_materialization(tmp_path: Path) -> None:
    artifact_sha256 = "b" * 64
    gate_path = tmp_path / "gate" / artifact_sha256 / "gate.json"
    document = {
        "version": selector_launcher.GATE_VERSION,
        "output_artifact_sha256": artifact_sha256,
        "deterministic": {
            "artifact_sha256": "c" * 64,
            "artifact_payload_sha256": "d" * 64,
        },
        "learned": {
            "artifact_sha256": "e" * 64,
            "artifact_payload_sha256": "f" * 64,
        },
        "bootstrap": {"lower_95": 0.1, "passes": True},
        "passes": True,
    }
    content = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    gate_path.parent.mkdir(parents=True)
    gate_path.write_bytes(content)
    payload_sha256 = hashlib.sha256(content).hexdigest()
    gate_path.with_name("gate.sha256").write_text(payload_sha256)

    load_gate_result(
        tmp_path / "gate",
        artifact_sha256,
        expected_payload_sha256=payload_sha256,
    )
    tampered = json.loads(gate_path.read_bytes())
    tampered["passes"] = not tampered["passes"]
    tampered_content = json.dumps(
        tampered,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    gate_path.write_bytes(tampered_content)
    gate_path.with_name("gate.sha256").write_text(
        hashlib.sha256(tampered_content).hexdigest()
    )
    with pytest.raises(ValueError, match="frozen payload"):
        load_gate_result(
            tmp_path / "gate",
            artifact_sha256,
            expected_payload_sha256=payload_sha256,
        )
    with pytest.raises(ValueError, match="decision differs"):
        load_gate_result(
            tmp_path / "gate",
            artifact_sha256,
            expected_payload_sha256=hashlib.sha256(tampered_content).hexdigest(),
        )


def test_queue_experiment_consumes_frozen_prepared_artifact_and_releases_marker(
    tmp_path: Path, monkeypatch
) -> None:
    bounds = ChronologicalBounds.from_interval(0, 40 * DAY_SECONDS)
    prepared = prepare_selector_data(
        _input_paths(tmp_path), bounds, tmp_path / "prepared"
    )
    trial = next(
        trial for trial in compile_selector_search() if trial.family == "content"
    )
    output_sha256 = "a" * 64
    job = trial.to_dict() | {
        "input_artifact_sha256": prepared.sha256,
        "input_payload_sha256": None,
        "deterministic_artifact_sha256": None,
        "deterministic_payload_sha256": None,
        "learned_artifact_sha256": None,
        "learned_payload_sha256": None,
        "output_artifact_sha256": output_sha256,
        "fold_id": None,
    }
    marker = tmp_path / "prepared.marker"
    release = tmp_path / "released.marker"
    release.touch()
    monkeypatch.setattr(selector_launcher, "PREPARED_ROOT", tmp_path / "prepared")
    monkeypatch.setattr(selector_launcher, "SEARCH_ROOT", tmp_path / "search")
    monkeypatch.setenv("DCN_PREPARED_MARKER", str(marker))
    monkeypatch.setenv("DCN_TRAINING_RELEASE", str(release))

    SelectorQueueExperiment(
        run_name="selector_queue_test",
        compiled_job={"job": job},
    ).run()

    assert marker.is_file()
    assert (tmp_path / "search" / output_sha256 / "result.json").is_file()


def test_queue_adapter_cannot_bypass_native_materialization_gate() -> None:
    experiment = SelectorQueueExperiment(
        run_name="selector_materialization_rejected",
        compiled_job={"job": {"stage": "selector_materialization"}},
    )

    with pytest.raises(RuntimeError, match="native cost gate"):
        experiment.run()


def test_prewarm_sha_verification_requires_regular_resident_inputs(
    tmp_path: Path,
) -> None:
    names = (
        "control_likes",
        "raw_events",
        "item_id_remap",
        "compact_embeddings",
    )
    paths = []
    expected = {}
    for index, name in enumerate(names):
        path = tmp_path / f"{name}.bin"
        content = bytes([index + 1]) * 8_193
        path.write_bytes(content)
        paths.append(path)
        expected[name] = hashlib.sha256(content).hexdigest()

    identities = prewarm_and_verify_inputs(paths, expected_sha256=expected)

    assert {name: value["sha256"] for name, value in identities.items()} == expected
    assert all_input_pages_resident(paths)
    expected["raw_events"] = "0" * 64
    with pytest.raises(ValueError, match="raw_events"):
        prewarm_and_verify_inputs(paths, expected_sha256=expected)
    paths[-1].unlink()
    paths[-1].symlink_to(paths[0])
    with pytest.raises(ValueError, match="regular file"):
        prewarm_and_verify_inputs(paths)


def test_materialized_period_excludes_zero_score_occurrences() -> None:
    queries = pl.DataFrame(
        {
            "uid": [1],
            "prefix_timestamp": [100],
            "prefix_item_id": [10],
            "occurrence_position": [0],
            "next_item": [11],
            "fold": [2],
        }
    )
    data = {
        "query_uid": np.array([1, 1]),
        "query_timestamp": np.array([100, 100]),
        "query_item": np.array([10, 10]),
        "query_position": np.array([0, 0]),
        "period_start": np.array([200, 200]),
        "period_end": np.array([300, 300]),
        "candidate_timestamp": np.array([210, 290]),
        "candidate_item": np.array([11, 12]),
        "candidate_position": np.array([1, 2]),
    }

    [query] = list(
        selector_launcher._scored_query_stream(queries, data, np.array([0.8, 0.0]))
    )

    assert query.periods[0].occurrences == (
        selector_launcher.ScoredOccurrence(210, 11, 1),
    )


def test_spawned_child_is_supervised_through_reap() -> None:
    context = multiprocessing.get_context("spawn")
    release = context.Event()
    monitor = SupervisedProcessTreeRssMonitor(interval_seconds=0.01)
    monitor.start()
    process = context.Process(target=release.wait)
    try:
        process.start()
        monitor.set_root_pid(process.pid)
        deadline = time.monotonic() + 10
        while monitor.peak_bytes == 0 and time.monotonic() < deadline:
            monitor.sample()
            time.sleep(0.01)
        assert monitor.peak_bytes > 0
        release.set()
        process.join()
        monitor.sample()
        assert process.exitcode == 0
    finally:
        monitor.stop()
        process.close()


def test_timed_monitor_invalidates_two_consecutive_overload_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(selector_launcher.os, "getloadavg", lambda: (17.0, 0.0, 0.0))
    monitor = SupervisedProcessTreeRssMonitor()

    monitor.sample()
    monitor._next_load_sample = 0.0
    monitor.sample()

    assert monitor.load_samples == [17.0, 17.0]
    assert monitor.load_valid is False


def test_five_fold_outputs_assemble_every_control_prefix_without_refit(
    tmp_path: Path,
) -> None:
    paths = _input_paths(tmp_path)
    users_by_fold: dict[int, int] = {}
    uid = 1
    while len(users_by_fold) < 5:
        users_by_fold.setdefault(fold_for_user(uid), uid)
        uid += 1
    users = [users_by_fold[fold] for fold in range(5)]
    raw_rows = []
    for day in range(40):
        first_artist_count = day % 10 + 1
        for occurrence in range(10):
            raw_rows.append(
                {
                    "uid": 1,
                    "item_id": 10 if occurrence < first_artist_count else 11,
                    "event_type": "listen",
                    "timestamp": day * DAY_SECONDS + 13 * 3_600 + occurrence,
                }
            )
    pl.DataFrame(raw_rows).write_parquet(paths.raw_events)
    for path in (paths.control_likes, paths.raw_events):
        source = pl.read_parquet(path)
        pl.concat(
            [source.with_columns(pl.lit(user).alias("uid")) for user in users]
        ).write_parquet(path)
    bounds = ChronologicalBounds.from_interval(0, 40 * DAY_SECONDS)
    prepared = prepare_selector_data(paths, bounds, tmp_path / "prepared")
    for width in (3_600, 21_600, DAY_SECONDS):
        frame = pl.read_parquet(prepared.width_path(width)).with_row_index("row")
        frame.with_columns(
            ((pl.col("row") % 101) / 100).alias("relevance_outcome")
        ).drop("row").write_parquet(prepared.width_path(width))
    trials = compile_selector_search()
    deterministic = run_search_trial(
        prepared,
        next(trial for trial in trials if trial.family == "content"),
        tmp_path / "search",
    )
    learned = run_search_trial(
        prepared,
        next(trial for trial in trials if trial.family == "learned"),
        tmp_path / "search",
    )
    gate = {
        "passes": True,
        "prepared_semantics_sha256": prepared.semantics_sha256,
        "deterministic": {
            "artifact_sha256": deterministic.artifact_sha256,
            "artifact_payload_sha256": deterministic.artifact_payload_sha256,
        },
        "learned": {
            "artifact_sha256": learned.artifact_sha256,
            "artifact_payload_sha256": learned.artifact_payload_sha256,
        },
    }
    fold_sha256s = []
    for fold in range(5):
        sha256 = f"{fold + 1:064x}"
        run_materialization_fold(
            prepared,
            learned,
            fold,
            tmp_path / "folds",
            output_artifact_sha256=sha256,
        )
        fold_sha256s.append(sha256)

    with pytest.raises(ValueError, match="quality gate"):
        finalize_materialization_folds(
            prepared,
            gate | {"passes": False},
            deterministic,
            learned,
            fold_sha256s,
            fold_root=tmp_path / "folds",
            output_root=tmp_path / "rejected_periods",
        )
    with pytest.raises(ValueError, match="winners"):
        finalize_materialization_folds(
            prepared,
            gate
            | {
                "learned": {"artifact_sha256": "f" * 64},
            },
            deterministic,
            learned,
            fold_sha256s,
            fold_root=tmp_path / "folds",
            output_root=tmp_path / "wrong_winner_periods",
        )

    deterministic_identity, learned_identity = finalize_materialization_folds(
        prepared,
        gate,
        deterministic,
        learned,
        fold_sha256s,
        fold_root=tmp_path / "folds",
        output_root=tmp_path / "periods",
    )

    expected_queries = prepared.manifest["pair_budget"]["control_prefixes"]
    assert deterministic_identity.query_count == expected_queries
    assert learned_identity.query_count == expected_queries
    artifact = PeriodArtifact.open(
        tmp_path / "periods", expected_sha256=learned_identity.sha256
    )
    assert artifact.manifest["provenance"]["fold_artifact_sha256s"] == fold_sha256s

    fold_path = tmp_path / "folds" / fold_sha256s[0]
    score_path = fold_path / "scores.parquet"
    score_frame = pl.read_parquet(score_path)
    pl.concat([score_frame.slice(1, 1), score_frame.slice(1)]).write_parquet(score_path)
    manifest_path = fold_path / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    stat = score_path.stat()
    manifest["scores"] = {
        "file": score_path.name,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": hashlib.sha256(score_path.read_bytes()).hexdigest(),
    }
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    )
    with pytest.raises(ValueError, match="exactly cover"):
        finalize_materialization_folds(
            prepared,
            gate,
            deterministic,
            learned,
            fold_sha256s,
            fold_root=tmp_path / "folds",
            output_root=tmp_path / "incomplete_periods",
        )
