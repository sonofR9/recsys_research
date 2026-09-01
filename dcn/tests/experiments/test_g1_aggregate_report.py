from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import pytest

from experiments.g1_sasrec_item_ids_likes.analysis import aggregate_report

from experiments.g1_sasrec_item_ids_likes.analysis.aggregate_candidates import (
    bridge_candidates,
    full_horizon_rerun_candidates,
    initial_candidates,
    make_horizon_correction,
    selection_initial_candidates,
)
from experiments.g1_sasrec_item_ids_likes.analysis.aggregate_report import (
    AggregateRun,
    build_report_bundle,
    classify_aggregate_outcome,
    collect_report_bundle,
)
from experiments.g1_sasrec_item_ids_likes.launchers import verify_artifact


METRICS = ("recall@100", "ndcg@100", "recall@10", "ndcg@10", "coverage@100")
AGGREGATE_CONFIG = Path(
    "experiments/g1_sasrec_item_ids_likes/configs/aggregate_variant.py"
)


def _run(candidate, validation: float, metrics: dict[str, float]) -> AggregateRun:
    return AggregateRun(
        candidate=candidate,
        best_epoch=10,
        stopped_epoch=candidate.num_epochs,
        validation_recall=validation,
        validation_ndcg=metrics["ndcg@100"],
        metrics=metrics,
    )


def _write_current_horizon_artifact(
    logs: Path,
    candidate,
    *,
    stopped_epoch: int,
    horizon_complete: bool,
) -> Path:
    directory = logs / candidate.run_name
    directory.mkdir(parents=True)
    assignments = [f"G1_AGGREGATE_RUN={candidate.run_name}"]
    experiment = verify_artifact._config_experiment(
        AGGREGATE_CONFIG, verify_artifact._config_assignments(assignments)
    )
    top_level, invariants = verify_artifact._expected_metadata(experiment)
    best_epoch = min(10, stopped_epoch)
    metadata = top_level | {
        "training_semantics_revision": 2,
        "max_epochs": 15,
        "epochs_trained": stopped_epoch,
        "stopped_epoch": stopped_epoch,
        "best_epoch": best_epoch,
        "early_stopped": stopped_epoch < 15,
        "best_epoch_at_cap": False,
        "selection_resolved": stopped_epoch == 15 and horizon_complete,
        "lr_horizon_complete": horizon_complete,
        "targets_per_epoch": 2,
        "tokens_per_epoch": 3,
        "optimizer_steps_per_epoch": 4,
        "optimizer_steps": 4 * stopped_epoch,
        "training_horizon": 2 * stopped_epoch,
        "token_horizon": 3 * stopped_epoch,
        "tokens_seen": 3 * stopped_epoch,
        "validation_loss": 0.5,
        "transfer_invariants": invariants,
    }
    (directory / "training_metadata.json").write_text(json.dumps(metadata))
    metrics = {
        "recall@100": 0.12,
        "ndcg@100": 0.045,
        "recall@10": 0.02,
        "ndcg@10": 0.015,
        "coverage@100": 0.6,
        "num_users": 37018,
    }
    (directory / "final_metrics.json").write_text(json.dumps(metrics))
    (directory / "sweep.log").write_text(
        f"epoch {best_epoch - 1} finished "
        "epoch/val_true.recall@100=0.12 epoch/val_true.ndcg@100=0.045\n"
    )
    return directory


def test_empty_bundle_requests_the_selection_surface_with_exact_two_reruns() -> None:
    bundle = build_report_bundle([])

    assert bundle.evidence["claims_status"] == "pending"
    assert bundle.evidence["required_followups"] == [
        candidate.run_name for candidate in selection_initial_candidates()
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
    assert "gain (%)" not in bundle.reader_markdown
    assert (
        '| recall@100 | 0.120 | <span style="color: green">'
        "+16.7% (0.140)</span> | +0.020 |" in bundle.reader_markdown
    )
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


def test_horizon_correction_cannot_replace_a_selection_surface_result() -> None:
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

    assert bundle.evidence["selected_aggregate"]["run"] != corrected.run_name
    assert aggregate_report.candidate_by_run(
        bundle.evidence["selected_aggregate"]["run"]
    ).correction == 0


def test_full_horizon_launcher_enqueues_the_exact_two(tmp_path: Path) -> None:
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
            "G1_AGGREGATE_STAGE": "full_horizon",
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
        candidate.run_name for candidate in full_horizon_rerun_candidates()
    ]
    assert all(
        line.split()[2] == f"G1_AGGREGATE_RUN={candidate.run_name}"
        for line, candidate in zip(
            enqueued, full_horizon_rerun_candidates(), strict=True
        )
    )
    assert result.stdout.splitlines().count("DRAIN") == 1


def test_launcher_refuses_to_replace_an_existing_artifact(tmp_path: Path) -> None:
    queue = tmp_path / "queue.sh"
    queue.write_text(
        'enqueue() { printf "ENQUEUE %s %s\\n" "$1" "$2"; }\n'
        'drain() { printf "DRAIN\\n"; }\n'
    )
    logs = tmp_path / "logs"
    directory = logs / full_horizon_rerun_candidates()[0].run_name
    directory.mkdir(parents=True)
    marker = directory / "keep-me"
    marker.write_text("immutable")
    launcher = Path(
        "experiments/g1_sasrec_item_ids_likes/launchers/aggregate_500m.sh"
    )

    result = subprocess.run(
        ["bash", str(launcher)],
        env=os.environ
        | {
            "G1_AGGREGATE_STAGE": "full_horizon",
            "G1_AGGREGATE_LOGS": str(logs),
            "G1_TRAINING_QUEUE_LIBRARY": str(queue),
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "Refusing to replace or archive" in result.stderr
    assert marker.read_text() == "immutable"
    assert not (logs / "old").exists()
    assert "ENQUEUE" not in result.stdout


@pytest.mark.parametrize(
    ("stopped_epoch", "horizon_complete", "accepted"),
    [(14, True, False), (15, False, False), (15, True, True)],
)
def test_launcher_path_uses_strict_full_horizon_verification(
    tmp_path: Path,
    stopped_epoch: int,
    horizon_complete: bool,
    accepted: bool,
) -> None:
    queue = tmp_path / "queue.sh"
    queue.write_text(
        'enqueue() { printf "ENQUEUE %s %s\\n" "$1" "$2"; }\n'
        'drain() { printf "DRAIN\\n"; }\n'
    )
    logs = tmp_path / "logs"
    candidate = full_horizon_rerun_candidates()[0]
    _write_current_horizon_artifact(
        logs,
        candidate,
        stopped_epoch=stopped_epoch,
        horizon_complete=horizon_complete,
    )
    launcher = Path(
        "experiments/g1_sasrec_item_ids_likes/launchers/aggregate_500m.sh"
    )

    result = subprocess.run(
        ["bash", str(launcher)],
        env=os.environ
        | {
            "G1_AGGREGATE_STAGE": "full_horizon",
            "G1_AGGREGATE_LOGS": str(logs),
            "G1_TRAINING_QUEUE_LIBRARY": str(queue),
        },
        check=False,
        capture_output=True,
        text=True,
    )

    if accepted:
        assert result.returncode == 0, result.stderr
        assert f"skipped compatible {candidate.run_name}" in result.stdout
        enqueued = [
            line for line in result.stdout.splitlines() if line.startswith("ENQUEUE")
        ]
        assert [line.split()[1] for line in enqueued] == [
            full_horizon_rerun_candidates()[1].run_name
        ]
    else:
        assert result.returncode == 2
        assert "Refusing to replace or archive" in result.stderr
        assert "ENQUEUE" not in result.stdout


@pytest.mark.parametrize("stage", ["initial", "recovery"])
def test_historical_launcher_stages_are_read_only(
    tmp_path: Path, stage: str
) -> None:
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
            "G1_AGGREGATE_STAGE": stage,
            "G1_AGGREGATE_LOGS": str(tmp_path / "logs"),
            "G1_TRAINING_QUEUE_LIBRARY": str(queue),
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "historical audit-only" in result.stderr
    assert "ENQUEUE" not in result.stdout
    assert "DRAIN" not in result.stdout


@pytest.mark.parametrize(
    ("stopped_epoch", "horizon_complete", "accepted"),
    [(14, True, False), (15, False, False), (15, True, True)],
)
def test_report_collector_uses_strict_full_horizon_verification(
    tmp_path: Path,
    stopped_epoch: int,
    horizon_complete: bool,
    accepted: bool,
) -> None:
    candidate = full_horizon_rerun_candidates()[0]
    _write_current_horizon_artifact(
        tmp_path,
        candidate,
        stopped_epoch=stopped_epoch,
        horizon_complete=horizon_complete,
    )

    bundle = collect_report_bundle(tmp_path)

    required = bundle.evidence["required_followups"]
    assert (candidate.run_name not in required) is accepted


def test_collector_ignores_corrections_and_requests_only_two_full_h15_reruns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    incomplete_recipes = {
        (
            candidate.num_layers,
            candidate.embedding_lr,
            candidate.deep_lr,
        )
        for candidate in full_horizon_rerun_candidates()
    }
    historical = initial_candidates()
    corrections = [
        make_horizon_correction(candidate, 18)
        for candidate in historical
        if candidate.family == "aggregate"
    ]
    for candidate in (*historical, *corrections):
        (tmp_path / candidate.run_name).mkdir()
    metrics = {
        "recall@100": 0.12,
        "ndcg@100": 0.045,
        "recall@10": 0.02,
        "ndcg@10": 0.015,
        "coverage@100": 0.6,
    }
    verified_corrections: list[str] = []

    def verify_current(directory: Path, *_args) -> bool:
        candidate = aggregate_report.candidate_by_run(directory.name)
        if candidate.correction:
            verified_corrections.append(candidate.run_name)
        return candidate.family == "baseline"

    def verify_historical(directory: Path, *_args) -> bool:
        candidate = aggregate_report.candidate_by_run(directory.name)
        return (
            candidate.num_layers,
            candidate.embedding_lr,
            candidate.deep_lr,
        ) not in incomplete_recipes

    monkeypatch.setattr(aggregate_report, "verify_config", verify_current)
    monkeypatch.setattr(
        aggregate_report,
        "verify_config_completed_historical_horizon",
        verify_historical,
    )
    monkeypatch.setattr(
        aggregate_report,
        "_load_run",
        lambda _directory, candidate: _run(candidate, 0.1, metrics),
    )

    bundle = collect_report_bundle(tmp_path)

    assert bundle.evidence["required_followups"] == [
        candidate.run_name for candidate in full_horizon_rerun_candidates()
    ]
    assert verified_corrections == []
