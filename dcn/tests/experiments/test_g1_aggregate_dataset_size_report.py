from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import statistics

import pytest

from experiments.g1_aggregate_dataset_size.analysis import report
from experiments.g1_aggregate_dataset_size.analysis.report import (
    EvidenceError,
    RunEvidence,
    SelectionPending,
    build_combined_reader,
    build_native50m_bands,
    build_size_evidence,
    build_study_bundle,
    collect_native50m_runs,
    load_native500m_reuse,
    write_study_bundle,
)
from experiments.g1_aggregate_dataset_size.launchers.runtime import (
    CandidateResult,
    archive_infeasible_batch_artifact,
)
from experiments.g1_aggregate_dataset_size.protocol.candidates import (
    aggregate_initial_candidates,
    baseline_initial_candidates,
    batch_initial_candidates,
    batch_lr_calibration_candidates,
    bridge_candidates as native50m_bridge_candidates,
    repeat_candidates,
)
from experiments.g1_sasrec_item_ids_likes.analysis.aggregate_candidates import (
    FIXED_MEMBERS,
)


METRICS = ("recall@100", "ndcg@100", "recall@10", "ndcg@10", "coverage@100")
SOURCE_EXPERIMENT = Path("experiments/g1_sasrec_item_ids_likes")


def _metrics(offset: float = 0.0) -> dict[str, float]:
    return {
        "recall@100": 0.1 + offset,
        "ndcg@100": 0.04 + offset,
        "recall@10": 0.02 + offset,
        "ndcg@10": 0.015 + offset,
        "coverage@100": 0.5 + offset,
    }


def _run(
    name: str,
    *,
    family: str,
    seed: int = 42,
    member: str | None = None,
    layers: int = 2,
    metrics: dict[str, float] | None = None,
    recipe: str = "baseline-recipe",
) -> RunEvidence:
    return RunEvidence(
        run_name=name,
        dataset_size="50m",
        family=family,
        stage=(
            "aggregate_initial"
            if family == "aggregate"
            else "bridge" if family == "bridge" else "baseline_initial"
        ),
        seed=seed,
        batch_size=1280,
        embedding_lr=0.032,
        deep_lr=0.012,
        num_layers=layers,
        member=member,
        best_epoch=12,
        stopped_epoch=15 if family == "aggregate" or member == "scheduler" else 20,
        horizon_epochs=15 if family == "aggregate" or member == "scheduler" else None,
        horizon_complete=(
            True if family == "aggregate" or member == "scheduler" else False
        ),
        selection_resolved=True,
        validation_recall=0.1,
        validation_ndcg=0.04,
        num_users=3414,
        metrics=_metrics() if metrics is None else metrics,
        recipe_sha256=recipe,
        artifact_sha256={
            "training_metadata.json": "a" * 64,
            "final_metrics.json": "b" * 64,
            "sweep.log": "c" * 64,
        },
    )


def _repeats(baseline: RunEvidence) -> list[RunEvidence]:
    rows = [baseline]
    for seed in range(43, 52):
        offset = (seed - 47) * 0.0001
        rows.append(
            _run(
                f"baseline-seed-{seed}",
                family="baseline",
                seed=seed,
                metrics=_metrics(offset),
            )
        )
    return rows


def _bridges(baseline: RunEvidence, selected_depth: int = 6) -> list[RunEvidence]:
    rows = [
        _run(
            f"bridge-{member}",
            family="bridge",
            member=member,
            metrics={
                metric: value + 0.001 for metric, value in baseline.metrics.items()
            },
            recipe=f"bridge-{member}-recipe",
        )
        for member in FIXED_MEMBERS
    ]
    for depth in (4, 6, 8):
        gain = 0.002 if depth == selected_depth else 0.05
        rows.append(
            _run(
                f"bridge-depth-{depth}",
                family="bridge",
                member="depth",
                layers=depth,
                metrics={
                    metric: value + gain for metric, value in baseline.metrics.items()
                },
                recipe=f"bridge-depth-{depth}-recipe",
            )
        )
    return rows


def _candidate_run(
    candidate,
    *,
    validation_recall: float,
    offset: float = 0.0,
    recipe: str | None = None,
) -> RunEvidence:
    scheduled = candidate.horizon_epochs is not None
    return RunEvidence(
        run_name=candidate.run_name,
        dataset_size=candidate.dataset_size,
        family=candidate.family,
        stage=candidate.stage,
        seed=candidate.seed,
        batch_size=candidate.batch_size,
        embedding_lr=candidate.embedding_lr,
        deep_lr=candidate.deep_lr,
        num_layers=candidate.num_layers,
        member=candidate.member,
        best_epoch=candidate.horizon_epochs - 1 if scheduled else 12,
        stopped_epoch=candidate.horizon_epochs if scheduled else 15,
        horizon_epochs=candidate.horizon_epochs,
        horizon_complete=scheduled,
        selection_resolved=True,
        validation_recall=validation_recall,
        validation_ndcg=0.04 + offset,
        num_users=3414,
        metrics=_metrics(offset),
        recipe_sha256=recipe or candidate.run_name,
        artifact_sha256={
            "training_metadata.json": "a" * 64,
            "final_metrics.json": "b" * 64,
            "sweep.log": "c" * 64,
        },
    )


def test_native50m_bands_require_exact_selected_baseline_seeds_and_recipe() -> None:
    baseline = _run("selected-baseline", family="baseline")
    repeats = _repeats(baseline)

    document = build_native50m_bands(baseline, repeats)

    recall_values = [row.metrics["recall@100"] for row in repeats]
    sample_stddev = statistics.stdev(recall_values)
    assert document["seeds"] == list(range(42, 52))
    assert (
        document["metrics"]["recall@100"]["sample_standard_deviation"] == sample_stddev
    )
    assert document["metrics"]["recall@100"]["reader_threshold"] == 0.001
    assert document["selected_baseline_run"] == baseline.run_name
    assert document["selected_baseline_recipe_sha256"] == baseline.recipe_sha256
    assert set(document["artifact_sha256"]) == {run.run_name for run in repeats}

    with pytest.raises(EvidenceError, match="seeds 42 through 51"):
        build_native50m_bands(baseline, repeats[:-1])

    incompatible = list(repeats)
    incompatible[-1] = _run(
        incompatible[-1].run_name,
        family="baseline",
        seed=51,
        recipe="different-recipe",
    )
    with pytest.raises(EvidenceError, match="exact selected baseline"):
        build_native50m_bands(baseline, incompatible)


def test_size_evidence_uses_only_selected_depth_in_eleven_member_sum() -> None:
    baseline = _run("selected-baseline", family="baseline")
    bands = build_native50m_bands(baseline, _repeats(baseline))
    aggregate = _run(
        "selected-aggregate",
        family="aggregate",
        layers=6,
        metrics={metric: value + 0.02 for metric, value in baseline.metrics.items()},
        recipe="aggregate-recipe",
    )

    evidence = build_size_evidence(
        "50m", baseline, aggregate, _bridges(baseline), bands
    )

    recall = evidence["aggregated_improvement"]["recall@100"]
    assert evidence["bridge_input_count"] == 13
    assert evidence["summed_bridge_count"] == 11
    assert evidence["selected_depth"] == 6
    assert recall["aggregate_gain_points"] == pytest.approx(0.02)
    assert recall["summed_standalone_gain_points"] == pytest.approx(0.012)
    assert recall["interaction_gap"] == pytest.approx(0.008)
    assert recall["interaction"] == "unresolved"
    assert evidence["excluded_diagnostic_depth_bridges"] == [4, 8]


def test_size_evidence_uses_unrounded_values_for_uplift_and_interaction() -> None:
    baseline = _run(
        "selected-baseline",
        family="baseline",
        metrics={metric: 0.123456789 for metric in METRICS},
    )
    repeats = [
        _run(
            f"baseline-seed-{seed}",
            family="baseline",
            seed=seed,
            metrics={metric: 0.123456789 + seed * 1e-8 for metric in METRICS},
        )
        for seed in range(42, 52)
    ]
    repeats[0] = baseline
    bands = build_native50m_bands(baseline, repeats)
    aggregate = _run(
        "selected-aggregate",
        family="aggregate",
        layers=4,
        metrics={metric: 0.126543219 for metric in METRICS},
        recipe="aggregate-recipe",
    )
    bridges = _bridges(baseline, selected_depth=4)

    evidence = build_size_evidence("50m", baseline, aggregate, bridges, bands)

    recall = evidence["aggregated_improvement"]["recall@100"]
    expected_gain = 0.126543219 - 0.123456789
    assert recall["aggregate_gain_points"] == expected_gain
    assert recall["aggregate_gain_percent"] == 100 * expected_gain / 0.123456789
    assert recall["interaction_gap"] == pytest.approx(expected_gain - 0.012)


def test_reporting_fails_closed_before_bands_and_all_bridges_are_ready() -> None:
    baseline = _run("selected-baseline", family="baseline")
    aggregate = _run(
        "selected-aggregate",
        family="aggregate",
        layers=8,
        recipe="aggregate-recipe",
    )
    bands = build_native50m_bands(baseline, _repeats(baseline))

    with pytest.raises(EvidenceError, match="validation_recall"):
        build_native50m_bands(
            replace(baseline, validation_recall=float("nan")), _repeats(baseline)
        )
    with pytest.raises(EvidenceError, match="artifact provenance"):
        build_native50m_bands(replace(baseline, artifact_sha256={}), _repeats(baseline))
    with pytest.raises(EvidenceError, match="band provenance"):
        build_size_evidence(
            "50m",
            baseline,
            aggregate,
            _bridges(baseline),
            bands | {"selected_baseline_run": "another-run"},
        )
    with pytest.raises(EvidenceError, match="thirteen bridge inputs"):
        build_size_evidence("50m", baseline, aggregate, _bridges(baseline)[:-1], bands)
    with pytest.raises(EvidenceError, match="dataset size"):
        build_size_evidence("500m", baseline, aggregate, _bridges(baseline), bands)


def test_combined_reader_requires_both_sizes_and_renders_separate_tables() -> None:
    baseline = _run("selected-baseline", family="baseline")
    bands = build_native50m_bands(baseline, _repeats(baseline))
    aggregate = _run(
        "selected-aggregate",
        family="aggregate",
        layers=4,
        metrics={metric: value + 0.02 for metric, value in baseline.metrics.items()},
        recipe="aggregate-recipe",
    )
    native50m = build_size_evidence(
        "50m", baseline, aggregate, _bridges(baseline, selected_depth=4), bands
    )
    native500m = load_native500m_reuse(SOURCE_EXPERIMENT).evidence

    with pytest.raises(EvidenceError, match="both native sizes"):
        build_combined_reader({"50m": native50m})

    markdown = build_combined_reader({"50m": native50m, "500m": native500m})

    assert markdown.startswith("## Aggregated improvement\n")
    assert markdown.count("| metric | baseline | aggregate |") == 1
    assert markdown.count("| metric | MuTransfer control | aggregate |") == 1
    assert "### Native Yambda-50M" in markdown
    assert "### Native Yambda-500M" in markdown
    assert "0.118" in markdown
    assert "+31.5% (0.155)" in markdown
    assert "descriptive and unresolved" in markdown


def test_native500m_reuse_is_byte_stable_with_frozen_outputs() -> None:
    reuse = load_native500m_reuse(SOURCE_EXPERIMENT)

    serialized = (
        json.dumps(reuse.evidence, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    assert (
        serialized
        == (
            SOURCE_EXPERIMENT / "evidence/aggregate_improvement_results.json"
        ).read_text()
    )
    assert (
        reuse.reader_markdown
        == (
            SOURCE_EXPERIMENT / "scratchpad/aggregate_improvement_reader_500m.md"
        ).read_text()
    )
    assert (
        reuse.tuning_markdown
        == (
            SOURCE_EXPERIMENT / "scratchpad/aggregate_improvement_tuning_500m.md"
        ).read_text()
    )


def _complete_native50m_runs() -> list[RunEvidence]:
    runs_by_name: dict[str, RunEvidence] = {}
    calibration = batch_lr_calibration_candidates()
    for index, candidate in enumerate(calibration):
        runs_by_name[candidate.run_name] = _candidate_run(
            candidate,
            validation_recall=(
                0.2 if candidate.batch_size == 1280 and index == 3 else 0.1
            ),
            recipe=(
                "baseline-recipe"
                if candidate.batch_size == 1280 and index == 3
                else None
            ),
        )
    selected_baseline = calibration[3]
    for candidate in repeat_candidates(selected_baseline):
        runs_by_name[candidate.run_name] = _candidate_run(
            candidate,
            validation_recall=0.19,
            offset=(candidate.seed - 47) * 0.0001,
            recipe="baseline-recipe",
        )
    for candidate in native50m_bridge_candidates(selected_baseline):
        runs_by_name[candidate.run_name] = _candidate_run(
            candidate,
            validation_recall=0.15,
            offset=0.001,
        )
    for candidate in aggregate_initial_candidates(1280):
        central = (candidate.embedding_lr, candidate.deep_lr) == (0.032, 0.012)
        depth_bonus = {4: 0.001, 6: 0.003, 8: 0.002}[candidate.num_layers]
        runs_by_name[candidate.run_name] = _candidate_run(
            candidate,
            validation_recall=0.15 + depth_bonus + (0.01 if central else 0),
            offset=0.02 + depth_bonus + (0.01 if central else 0),
        )
    return list(runs_by_name.values())


def test_complete_native50m_manifest_selects_only_after_repeats_and_bridges() -> None:
    runs = _complete_native50m_runs()

    incomplete = [run for run in runs if run.stage != "repeat"]
    with pytest.raises(SelectionPending) as pending:
        build_study_bundle(incomplete, SOURCE_EXPERIMENT, native500m_logs=None)
    assert len(pending.value.required_runs) == 9

    bundle = build_study_bundle(runs, SOURCE_EXPERIMENT, native500m_logs=None)

    assert bundle.native50m_evidence["claims_status"] == "ready"
    assert bundle.native50m_evidence["selected_depth"] == 6
    assert bundle.native50m_evidence["bridge_input_count"] == 13
    assert bundle.native50m_evidence["summed_bridge_count"] == 11
    assert bundle.native50m_bands["seeds"] == list(range(42, 52))
    assert (
        "native 50m MuTransfer control" in bundle.native50m_tuning_markdown
    )


def test_corrected_report_requires_all_six_and_ignores_old_batch_surface() -> None:
    calibration = batch_lr_calibration_candidates()
    runs = _complete_native50m_runs()
    missing = calibration[-1]
    incomplete = [run for run in runs if run.run_name != missing.run_name]
    with pytest.raises(SelectionPending) as pending:
        build_study_bundle(incomplete, SOURCE_EXPERIMENT, native500m_logs=None)
    assert pending.value.required_runs == (missing.run_name,)

    old = batch_initial_candidates()[0]
    old_run = _candidate_run(old, validation_recall=0.99)
    bundle = build_study_bundle(
        [*runs, old_run], SOURCE_EXPERIMENT, native500m_logs=None
    )

    calibration = bundle.native50m_evidence["batch_calibration"]
    assert calibration["selected_batch_size"] == 1280
    assert calibration["selected_embedding_lr"] == batch_lr_calibration_candidates()[3].embedding_lr
    assert calibration["selected_deep_lr"] == batch_lr_calibration_candidates()[3].deep_lr
    assert old.run_name not in bundle.native50m_tuning_markdown


def test_tuning_renderer_does_not_restore_fixed_lr_batch_table() -> None:
    old = batch_initial_candidates()[0]
    markdown = report.render_tuning_ledger(
        "50m", [_candidate_run(old, validation_recall=0.99)], set()
    )

    assert "| family |" not in markdown
    assert "audit-only" in markdown


def test_tuning_renderer_preserves_validation_log_precision() -> None:
    candidate = batch_lr_calibration_candidates()[0]
    markdown = report.render_tuning_ledger(
        "50m",
        [_candidate_run(candidate, validation_recall=0.0935)],
        {candidate.run_name},
    )

    assert "| 0.0935 | 0.0400 | yes |" in markdown
    assert "0.093500000" not in markdown


def test_raw_collection_requires_verified_candidate_and_full_user_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = batch_initial_candidates()[0]
    directory = tmp_path / candidate.run_name
    directory.mkdir()
    metadata = {
        "best_epoch": 12,
        "stopped_epoch": 15,
        "selection_resolved": True,
        "dataset_size": "50m",
        "num_epochs": 80,
        "batch_size": candidate.batch_size,
        "embedding_learning_rate": candidate.embedding_lr,
        "deep_learning_rate": candidate.deep_lr,
        "transfer_invariants": {"dataset_size": "50m"},
    }
    metrics = _metrics() | {"num_users": 3414.0}
    (directory / "training_metadata.json").write_text(json.dumps(metadata))
    (directory / "final_metrics.json").write_text(json.dumps(metrics))
    (directory / "sweep.log").write_text("verified raw log\n")
    monkeypatch.setattr(report, "verify_candidate_artifact", lambda *_args: True)
    monkeypatch.setattr(
        report,
        "load_candidate_result",
        lambda *_args: CandidateResult(candidate, 0.1, 0.04, 12),
    )

    runs = collect_native50m_runs(tmp_path)

    assert len(runs) == 1
    assert runs[0].run_name == candidate.run_name
    assert runs[0].num_users == 3414
    assert set(runs[0].artifact_sha256) == {
        "training_metadata.json",
        "final_metrics.json",
        "sweep.log",
    }

    metrics["num_users"] = 3413
    (directory / "final_metrics.json").write_text(json.dumps(metrics))
    with pytest.raises(EvidenceError, match="wrong user count"):
        collect_native50m_runs(tmp_path)

    monkeypatch.setattr(report, "verify_candidate_artifact", lambda *_args: False)
    with pytest.raises(EvidenceError, match="verification failed"):
        collect_native50m_runs(tmp_path)


def test_writer_keeps_size_artifacts_separate_and_requires_raw_500m_replay(
    tmp_path: Path,
) -> None:
    bundle = build_study_bundle(
        _complete_native50m_runs(), SOURCE_EXPERIMENT, native500m_logs=None
    )

    with pytest.raises(EvidenceError, match="raw evidence must be replayed"):
        write_study_bundle(bundle, tmp_path)

    forged = replace(
        bundle,
        native500m_provenance=bundle.native500m_provenance | {"raw_replayed": True},
    )
    with pytest.raises(EvidenceError, match="raw evidence must be replayed"):
        write_study_bundle(forged, tmp_path)

    with pytest.raises(EvidenceError, match="structured run evidence"):
        write_study_bundle(
            replace(
                bundle,
                native50m_bands=bundle.native50m_bands | {"seeds": [42]},
            ),
            tmp_path,
        )
    with pytest.raises(EvidenceError, match="structured run evidence"):
        write_study_bundle(
            replace(
                bundle,
                native50m_evidence=bundle.native50m_evidence
                | {"claims_status": "pending"},
            ),
            tmp_path,
        )
    with pytest.raises(EvidenceError, match="combined reader"):
        write_study_bundle(
            replace(bundle, reader_markdown=bundle.reader_markdown + "tampered\n"),
            tmp_path,
        )

    in_place_mutation = replace(
        bundle,
        native50m_evidence=json.loads(json.dumps(bundle.native50m_evidence)),
    )
    in_place_mutation.native50m_evidence["selected_baseline"]["metrics"][
        "recall@100"
    ] += 0.01
    with pytest.raises(EvidenceError, match="structured run evidence"):
        write_study_bundle(
            in_place_mutation,
            tmp_path,
        )
    with pytest.raises(TypeError):
        bundle.native50m_runs[0].artifact_sha256["sweep.log"] = "d" * 64

    tampered_bands = json.loads(json.dumps(bundle.native50m_bands))
    repeated_run = tampered_bands["run_names"][1]
    tampered_bands["artifact_sha256"][repeated_run]["sweep.log"] = "d" * 64
    tampered_band_evidence = json.loads(json.dumps(bundle.native50m_evidence))
    tampered_band_evidence["bands"] = tampered_bands
    with pytest.raises(EvidenceError, match="structured run evidence"):
        write_study_bundle(
            replace(
                bundle,
                native50m_bands=tampered_bands,
                native50m_evidence=tampered_band_evidence,
            ),
            tmp_path,
        )
    with pytest.raises(EvidenceError, match="structured run evidence"):
        write_study_bundle(
            replace(
                bundle,
                native50m_tuning_markdown=bundle.native50m_tuning_markdown
                + "tampered\n",
            ),
            tmp_path,
        )

    replayed = load_native500m_reuse(SOURCE_EXPERIMENT)
    paths = write_study_bundle(
        replace(
            bundle,
            native500m_evidence=replayed.evidence,
            native500m_tuning_markdown=replayed.tuning_markdown,
            native500m_provenance=replayed.provenance,
        ),
        tmp_path,
    )

    assert set(paths) == {
        "native50m_evidence",
        "native500m_evidence",
        "native50m_bands",
        "native500m_provenance",
        "native50m_tuning",
        "native500m_tuning",
        "reader",
    }
    assert (
        paths["native500m_evidence"].read_text()
        == (
            SOURCE_EXPERIMENT / "evidence/aggregate_improvement_results.json"
        ).read_text()
    )
    assert (
        paths["native500m_tuning"].read_text()
        == (
            SOURCE_EXPERIMENT / "scratchpad/aggregate_improvement_tuning_500m.md"
        ).read_text()
    )
