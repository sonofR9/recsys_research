from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import pytest

from experiments.g1_sasrec_item_ids_likes.analysis import aggregate_report

from experiments.g1_sasrec_item_ids_likes.analysis.aggregate_candidates import (
    aggregate_boundary_candidates,
    aggregate_local_candidates,
    bridge_candidates,
    initial_candidates,
    make_horizon_correction,
    recovery_candidates,
)
from experiments.g1_sasrec_item_ids_likes.analysis.aggregate_report import (
    AggregateRun,
    build_report_bundle,
    classify_aggregate_outcome,
    collect_report_bundle,
)


METRICS = ("recall@100", "ndcg@100", "recall@10", "ndcg@10", "coverage@100")


def _run(candidate, validation: float, metrics: dict[str, float]) -> AggregateRun:
    return AggregateRun(
        candidate=candidate,
        best_epoch=10,
        stopped_epoch=candidate.num_epochs,
        validation_recall=validation,
        validation_ndcg=metrics["ndcg@100"],
        metrics=metrics,
    )


def test_empty_bundle_requests_the_parallel_twelve_run_initial_stage() -> None:
    bundle = build_report_bundle([])

    assert bundle.evidence["claims_status"] == "pending"
    assert bundle.evidence["required_followups"] == [
        candidate.run_name for candidate in initial_candidates()
    ]


def test_complete_bundle_selects_depth_and_uses_unrounded_bridge_arithmetic() -> None:
    baseline_metrics = {
        "recall@100": 0.12,
        "ndcg@100": 0.045,
        "recall@10": 0.02,
        "ndcg@10": 0.015,
        "coverage@100": 0.6,
    }
    runs = []
    for candidate in initial_candidates()[:3]:
        validation = 0.13 if candidate.deep_lr == 0.012 else 0.12
        runs.append(_run(candidate, validation, baseline_metrics))

    for candidate in initial_candidates()[3:]:
        pair_bonus = 0.002 if candidate.embedding_lr == 0.064 else 0.0
        depth_bonus = {4: 0.001, 6: 0.003, 8: 0.002}[candidate.num_layers]
        metrics = {
            metric: value + pair_bonus + depth_bonus
            for metric, value in baseline_metrics.items()
        }
        runs.append(_run(candidate, 0.13 + pair_bonus + depth_bonus, metrics))

    bridge_metrics = {
        metric: value + 0.001 for metric, value in baseline_metrics.items()
    }
    for candidate in bridge_candidates(0.012, selected_depth=6):
        runs.append(_run(candidate, 0.131, bridge_metrics))

    selected = next(
        candidate
        for candidate in initial_candidates()
        if candidate.family == "aggregate"
        and candidate.num_layers == 6
        and candidate.embedding_lr == 0.064
    )
    selected_metrics = {
        metric: value + 0.02 for metric, value in baseline_metrics.items()
    }
    runs = [
        _run(run.candidate, 0.15, selected_metrics)
        if run.candidate == selected
        else run
        for run in runs
    ]

    bundle = build_report_bundle(runs)

    assert bundle.evidence["claims_status"] == "ready"
    assert bundle.evidence["aggregate_outcome"]["classification"] == "positive"
    assert bundle.evidence["selected_depth"] == 6
    recall = bundle.evidence["aggregated_improvement"]["recall@100"]
    assert recall["baseline"] == 0.12
    assert recall["aggregate"] == 0.13999999999999999
    assert recall["aggregate_gain_points"] == 0.01999999999999999
    assert recall["summed_standalone_gain_points"] == 0.01100000000000001
    assert recall["interaction_gap"] == 0.00899999999999998
    assert recall["interaction"] == "positive"
    assert "## Aggregated improvement" in bundle.reader_markdown
    assert "interaction gap" in bundle.reader_markdown
    assert "eleven one-factor bridges" in bundle.reader_markdown
    assert "Outcome: **positive**" in bundle.reader_markdown
    assert "| recall@100 | 0.120 | 0.140 | +0.020 |" in bundle.reader_markdown
    assert "| aggregate |" in bundle.tuning_markdown


def test_outcome_reports_a_primary_secondary_tradeoff() -> None:
    baseline = {"recall@100": 0.12, "ndcg@100": 0.045}
    aggregate = {"recall@100": 0.124, "ndcg@100": 0.0439}

    outcome = classify_aggregate_outcome(baseline, aggregate)

    assert outcome["classification"] == "trade-off"
    assert outcome["recall@100_gain"] == 0.0040000000000000036
    assert outcome["ndcg@100_gain"] == -0.0010999999999999968


def test_outcome_reports_an_ndcg_only_regression() -> None:
    baseline = {"recall@100": 0.12, "ndcg@100": 0.045}
    aggregate = {"recall@100": 0.119, "ndcg@100": 0.0439}

    outcome = classify_aggregate_outcome(baseline, aggregate)

    assert outcome["classification"] == "regression"


def test_partial_bridge_surface_requests_only_the_missing_selected_bridge() -> None:
    metrics = {
        "recall@100": 0.12,
        "ndcg@100": 0.045,
        "recall@10": 0.02,
        "ndcg@10": 0.015,
        "coverage@100": 0.6,
    }
    runs = [
        _run(
            candidate,
            0.13
            + (0.01 if candidate.family == "baseline" and candidate.deep_lr == 0.012 else 0)
            + (
                0.01
                if candidate.family == "aggregate"
                and candidate.embedding_lr == 0.064
                else 0
            )
            + (0.001 * candidate.num_layers if candidate.family == "aggregate" else 0),
            metrics,
        )
        for candidate in initial_candidates()
    ]
    bridges = bridge_candidates(0.012, selected_depth=8)
    runs.extend(_run(candidate, 0.13, metrics) for candidate in bridges[:-1])

    bundle = build_report_bundle(runs)

    assert bundle.evidence["required_followups"] == [bridges[-1].run_name]


def test_corrected_initial_winner_still_requests_its_local_surface() -> None:
    metrics = {
        "recall@100": 0.12,
        "ndcg@100": 0.045,
        "recall@10": 0.02,
        "ndcg@10": 0.015,
        "coverage@100": 0.6,
    }
    candidates = initial_candidates()
    runs = [
        _run(
            candidate,
            0.14
            if candidate.family == "baseline" and candidate.deep_lr == 0.012
            else 0.13
            if candidate.family == "aggregate" and candidate.embedding_lr == 0.064
            else 0.12,
            metrics,
        )
        for candidate in candidates
    ]
    source = next(
        candidate
        for candidate in candidates
        if candidate.family == "aggregate"
        and candidate.num_layers == 4
        and candidate.embedding_lr == 0.07764674795069047
    )
    corrected = make_horizon_correction(source, 18)
    runs.append(_run(corrected, 0.2, metrics))

    bundle = build_report_bundle(runs)

    assert bundle.evidence["required_followups"] == [
        candidate.run_name for candidate in aggregate_local_candidates(source)
    ]


def test_initial_launcher_enqueues_the_exact_parallel_twelve(tmp_path: Path) -> None:
    queue = tmp_path / "queue.sh"
    queue.write_text(
        'enqueue() { printf "ENQUEUE %s %s\\n" "$1" "$2"; }\n'
        'drain() { printf "DRAIN\\n"; }\n'
    )
    launcher = Path(
        "experiments/g1_sasrec_item_ids_likes/launchers/aggregate_500m.sh"
    )

    result = subprocess.run(
        ["bash", str(launcher)],
        env=os.environ
        | {
            "G1_AGGREGATE_STAGE": "initial",
            "G1_AGGREGATE_LOGS": str(tmp_path / "logs"),
            "G1_TRAINING_QUEUE_LIBRARY": str(queue),
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    enqueued = [line for line in result.stdout.splitlines() if line.startswith("ENQUEUE")]
    assert [line.split()[1] for line in enqueued] == [
        candidate.run_name for candidate in initial_candidates()
    ]
    assert all(
        line.split()[2] == f"G1_AGGREGATE_RUN={candidate.run_name}"
        for line, candidate in zip(enqueued, initial_candidates(), strict=True)
    )
    assert result.stdout.splitlines().count("DRAIN") == 1


def test_recovery_launcher_enqueues_the_exact_approved_eight(tmp_path: Path) -> None:
    queue = tmp_path / "queue.sh"
    queue.write_text(
        'enqueue() { printf "ENQUEUE %s %s\\n" "$1" "$2"; }\n'
        'drain() { printf "DRAIN\\n"; }\n'
    )
    launcher = Path(
        "experiments/g1_sasrec_item_ids_likes/launchers/aggregate_500m.sh"
    )

    result = subprocess.run(
        ["bash", str(launcher)],
        env=os.environ
        | {
            "G1_AGGREGATE_STAGE": "recovery",
            "G1_AGGREGATE_LOGS": str(tmp_path / "logs"),
            "G1_TRAINING_QUEUE_LIBRARY": str(queue),
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    enqueued = [line for line in result.stdout.splitlines() if line.startswith("ENQUEUE")]
    assert [line.split()[1] for line in enqueued] == [
        candidate.run_name for candidate in recovery_candidates()
    ]
    assert result.stdout.splitlines().count("DRAIN") == 1


def _collect_with_unresolved_correction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source,
    next_horizon: int,
):
    metrics = {
        "recall@100": 0.12,
        "ndcg@100": 0.045,
        "recall@10": 0.02,
        "ndcg@10": 0.015,
        "coverage@100": 0.6,
    }
    for candidate in initial_candidates():
        if (
            candidate.family,
            candidate.num_layers,
            candidate.embedding_lr,
            candidate.deep_lr,
        ) == (
            source.family,
            source.num_layers,
            source.embedding_lr,
            source.deep_lr,
        ):
            continue
        (tmp_path / candidate.run_name).mkdir()
    source_directory = tmp_path / source.run_name
    source_directory.mkdir()
    (source_directory / "training_metadata.json").write_text(
        json.dumps({"next_lr_schedule_horizon_epochs": next_horizon})
    )
    monkeypatch.setattr(
        aggregate_report,
        "verify_config",
        lambda directory, *_: directory.name != source.run_name,
    )
    monkeypatch.setattr(
        aggregate_report,
        "verify_config_recipe",
        lambda directory, *_: directory.name == source.run_name,
    )
    monkeypatch.setattr(
        aggregate_report,
        "_load_run",
        lambda _directory, candidate: _run(candidate, 0.1, metrics),
    )
    return collect_report_bundle(tmp_path)


def test_post_recovery_followup_is_the_exact_h27_c4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = recovery_candidates()[0]
    expected = make_horizon_correction(source, 27)

    bundle = _collect_with_unresolved_correction(
        tmp_path, monkeypatch, source, next_horizon=27
    )

    assert bundle.evidence["claims_status"] == "pending"
    assert bundle.evidence["required_followups"] == [expected.run_name]


def test_exhausted_c4_returns_explicit_approval_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = make_horizon_correction(recovery_candidates()[0], 27)

    bundle = _collect_with_unresolved_correction(
        tmp_path, monkeypatch, source, next_horizon=40
    )

    assert bundle.evidence["claims_status"] == "approval_required"
    assert bundle.evidence["required_followups"] == []
    assert bundle.evidence["approval_required"] == [
        {
            "run": source.run_name,
            "reason": "four horizon corrections did not calibrate the run",
        }
    ]


def test_exhausted_losing_6l_c4_does_not_block_clear_4l_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    c3 = recovery_candidates()[0]
    c4 = make_horizon_correction(c3, 27)
    target_recipe = (
        c3.family,
        c3.num_layers,
        c3.embedding_lr,
        c3.deep_lr,
    )
    candidates = [
        candidate
        for candidate in initial_candidates()
        if (
            candidate.family,
            candidate.num_layers,
            candidate.embedding_lr,
            candidate.deep_lr,
        )
        != target_recipe
    ]
    candidates.extend((c3, *bridge_candidates(0.012, selected_depth=4)))
    for candidate in (*candidates, c4):
        (tmp_path / candidate.run_name).mkdir()
    (tmp_path / c4.run_name / "training_metadata.json").write_text(
        json.dumps({"next_lr_schedule_horizon_epochs": 40})
    )
    metrics = {
        "recall@100": 0.12,
        "ndcg@100": 0.045,
        "recall@10": 0.02,
        "ndcg@10": 0.015,
        "coverage@100": 0.6,
    }

    def validation(candidate) -> float:
        if candidate.family == "baseline":
            return 0.2 if candidate.deep_lr == 0.012 else 0.1
        if candidate.family == "aggregate":
            return (
                0.3 + {4: 0.03, 6: 0.01, 8: 0.02}[candidate.num_layers]
                if candidate.embedding_lr == 0.064
                else 0.1
            )
        return 0.1

    monkeypatch.setattr(
        aggregate_report,
        "verify_config",
        lambda directory, *_: directory.name != c4.run_name,
    )
    monkeypatch.setattr(
        aggregate_report,
        "verify_config_recipe",
        lambda directory, *_: directory.name == c4.run_name,
    )
    monkeypatch.setattr(
        aggregate_report,
        "_load_run",
        lambda _directory, candidate: _run(candidate, validation(candidate), metrics),
    )

    bundle = collect_report_bundle(tmp_path)

    assert bundle.evidence["claims_status"] == "ready"
    assert bundle.evidence["selected_depth"] == 4
    assert bundle.evidence["required_followups"] == []


def test_exhausted_selected_c4_overrides_its_valid_c3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    c3 = recovery_candidates()[0]
    c4 = make_horizon_correction(c3, 27)
    target_initial = next(
        candidate
        for candidate in initial_candidates()
        if candidate.family == "aggregate"
        and candidate.num_layers == c3.num_layers
        and candidate.embedding_lr == c3.embedding_lr
        and candidate.deep_lr == c3.deep_lr
    )
    target_recipe = (
        c3.family,
        c3.num_layers,
        c3.embedding_lr,
        c3.deep_lr,
    )
    candidates = [
        candidate
        for candidate in initial_candidates()
        if (
            candidate.family,
            candidate.num_layers,
            candidate.embedding_lr,
            candidate.deep_lr,
        )
        != target_recipe
    ]
    candidates.extend(
        (
            c3,
            *aggregate_local_candidates(target_initial),
            *aggregate_boundary_candidates(target_initial),
            *bridge_candidates(0.012, selected_depth=6),
        )
    )
    for candidate in (*candidates, c4):
        (tmp_path / candidate.run_name).mkdir()
    (tmp_path / c4.run_name / "training_metadata.json").write_text(
        json.dumps({"next_lr_schedule_horizon_epochs": 40})
    )
    metrics = {
        "recall@100": 0.12,
        "ndcg@100": 0.045,
        "recall@10": 0.02,
        "ndcg@10": 0.015,
        "coverage@100": 0.6,
    }

    def validation(candidate) -> float:
        if candidate.family == "baseline":
            return 0.2 if candidate.deep_lr == 0.012 else 0.1
        if candidate == c3:
            return 0.5
        if candidate.family == "aggregate" and candidate.embedding_lr == 0.064:
            return 0.3
        return 0.1

    monkeypatch.setattr(
        aggregate_report,
        "verify_config",
        lambda directory, *_: directory.name != c4.run_name,
    )
    monkeypatch.setattr(
        aggregate_report,
        "verify_config_recipe",
        lambda directory, *_: directory.name == c4.run_name,
    )
    monkeypatch.setattr(
        aggregate_report,
        "_load_run",
        lambda _directory, candidate: _run(candidate, validation(candidate), metrics),
    )

    bundle = collect_report_bundle(tmp_path)

    assert bundle.evidence["claims_status"] == "approval_required"
    assert bundle.evidence["required_followups"] == []
    assert bundle.evidence["approval_required"] == [
        {
            "run": c4.run_name,
            "reason": "four horizon corrections did not calibrate the run",
        }
    ]
