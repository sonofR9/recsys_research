from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import re

import polars as pl
import pytest

from experiments.g1_sasrec_item_ids_likes.analysis import (
    rq13_prefix_expansion_report as rq13_report,
)
from experiments.g1_sasrec_item_ids_likes.analysis.rq13_prefix_expansion_audit import (
    current_implementation_sha256,
)
from experiments.g1_sasrec_item_ids_likes.analysis.rq13_prefix_expansion_report import (
    _CACHE_PATTERN,
    _RESULTS_EVIDENCE,
    _RETAINED_LEGACY_CACHE_FINGERPRINTS,
    Control,
    Rq13ReportError,
    Run,
    _legacy_cache_fingerprints,
    _sequence_cache_content,
    build_report_bundle,
    collect_report_bundle,
    write_report_bundle,
)
from experiments.g1_sasrec_item_ids_likes.analysis.rq13_rq14_query_candidates import (
    QueryCandidate,
    make_boundary_candidate,
    make_selected_cap_candidates,
    rq13_cap4_candidates,
    rq13_initial_candidates,
)


def test_default_migration_evidence_matches_pinned_legacy_identities() -> None:
    assert (
        _legacy_cache_fingerprints(_RESULTS_EVIDENCE)
        == _RETAINED_LEGACY_CACHE_FINGERPRINTS
    )


def _event_count_migration_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    event_count: int = 3,
    extra_field: bool = False,
    migration_before_run: bool = False,
) -> tuple[Path, Path, datetime]:
    cache = tmp_path / "train_provenance"
    buckets = cache / "buckets"
    buckets.mkdir(parents=True)
    bucket = buckets / "bucket_00000.parquet"
    pl.DataFrame(
        {
            "uid": [1, 2],
            "timestamp": [[1, 2], [3]],
            "compact_item_id": [[4, 5], [6]],
        }
    ).write_parquet(bucket)
    legacy = {
        "params": {"timestamp_column": "timestamp"},
        "bucket_files": [bucket.name],
        "bucket_lengths": [2],
        "int_columns": ["compact_item_id"],
        "float_columns": [],
    }
    metadata = {**legacy, "event_count": event_count}
    if extra_field:
        metadata["unproved"] = True
    (cache / "metadata.json").write_text(json.dumps(metadata))
    legacy_manifest = {
        "metadata.json": hashlib.sha256(json.dumps(legacy).encode()).hexdigest(),
        "buckets/bucket_00000.parquet": hashlib.sha256(bucket.read_bytes()).hexdigest(),
    }
    fingerprint = hashlib.sha256(
        json.dumps(legacy_manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    monkeypatch.setattr(
        rq13_report,
        "_RETAINED_LEGACY_CACHE_FINGERPRINTS",
        {
            "train": fingerprint,
            "val": fingerprint,
            "true_metric_query": fingerprint,
        },
    )
    evidence = tmp_path / "prior-rq13-results.json"
    evidence.write_text(
        json.dumps(
            {
                "research_question": "RQ13 encoder-decoder prefix expansion",
                "dataset_size": "500m",
                "treatments": {
                    "one_example": {
                        "artifacts": [
                            {
                                "compatibility_fingerprints": {
                                    "train_cache_manifest_sha256": fingerprint,
                                    "validation_cache_manifest_sha256": fingerprint,
                                    "query_cache_manifest_sha256": fingerprint,
                                }
                            }
                        ]
                    }
                },
            }
        )
    )
    run_start = datetime.now() - timedelta(seconds=1)
    old = run_start.timestamp() - 10
    os.utime(bucket, (old, old))
    os.utime(evidence, (old, old))
    if migration_before_run:
        migrated = run_start.timestamp() - 5
        os.utime(cache / "metadata.json", (migrated, migrated))
    return cache, evidence, run_start


def test_event_count_migration_reuses_the_retained_legacy_cache_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache, evidence, run_start = _event_count_migration_fixture(tmp_path, monkeypatch)

    manifest = _sequence_cache_content(
        cache, "train", run_start, "run", evidence
    )

    assert manifest["metadata.json"] != hashlib.sha256(
        (cache / "metadata.json").read_bytes()
    ).hexdigest()


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"event_count": 4}, "event_count"),
        ({"extra_field": True}, "sole metadata change"),
    ],
)
def test_event_count_migration_fails_closed_without_the_exact_derived_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kwargs: dict[str, object],
    match: str,
) -> None:
    cache, evidence, run_start = _event_count_migration_fixture(
        tmp_path, monkeypatch, **kwargs
    )

    with pytest.raises(Rq13ReportError, match=match):
        _sequence_cache_content(cache, "train", run_start, "run", evidence)


def test_event_count_migration_fails_closed_without_prior_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache, evidence, run_start = _event_count_migration_fixture(tmp_path, monkeypatch)
    evidence.unlink()

    with pytest.raises(Rq13ReportError, match="missing event_count migration evidence"):
        _sequence_cache_content(cache, "train", run_start, "run", evidence)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"event_count": 4}, "event_count"),
        ({"extra_field": True}, "sole metadata change"),
    ],
)
def test_pre_run_event_count_migration_near_miss_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kwargs: dict[str, object],
    match: str,
) -> None:
    cache, evidence, run_start = _event_count_migration_fixture(
        tmp_path, monkeypatch, migration_before_run=True, **kwargs
    )

    with pytest.raises(Rq13ReportError, match=match):
        _sequence_cache_content(cache, "train", run_start, "run", evidence)


def test_event_count_migration_proof_ignores_evidence_file_rewrite_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache, evidence, run_start = _event_count_migration_fixture(tmp_path, monkeypatch)
    first = _sequence_cache_content(cache, "train", run_start, "run", evidence)
    content = evidence.read_text()
    evidence.write_text(content)
    newer = datetime.now().timestamp() + 10
    os.utime(evidence, (newer, newer))

    second = _sequence_cache_content(cache, "train", run_start, "run", evidence)

    assert second == first


def test_event_count_migration_evidence_cannot_redefine_pinned_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache, evidence, run_start = _event_count_migration_fixture(tmp_path, monkeypatch)
    document = json.loads(evidence.read_text())
    fingerprints = document["treatments"]["one_example"]["artifacts"][0][
        "compatibility_fingerprints"
    ]
    fingerprints["train_cache_manifest_sha256"] = "0" * 64
    evidence.write_text(json.dumps(document))

    with pytest.raises(Rq13ReportError, match="pinned legacy identities"):
        _sequence_cache_content(cache, "train", run_start, "run", evidence)


def test_event_count_migration_rehashes_buckets_on_repeated_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache, evidence, run_start = _event_count_migration_fixture(tmp_path, monkeypatch)
    _sequence_cache_content(cache, "train", run_start, "run", evidence)
    bucket = cache / "buckets/bucket_00000.parquet"
    old = bucket.stat().st_mtime
    pl.DataFrame(
        {
            "uid": [10, 20],
            "timestamp": [[10, 20], [30]],
            "compact_item_id": [[40, 50], [60]],
        }
    ).write_parquet(bucket)
    os.utime(bucket, (old, old))

    with pytest.raises(Rq13ReportError, match="retained legacy evidence"):
        _sequence_cache_content(cache, "train", run_start, "run", evidence)


@pytest.mark.parametrize(
    ("line", "split"),
    [
        (
            "Loaded cached user sequences from /data/sequences/train_abc123",
            "train",
        ),
        (
            "Built 42 user sequences at /data/sequences/val_abc123 in 3 bucket(s) from 2 parquet file(s)",
            "val",
        ),
    ],
)
def test_cache_provenance_accepts_loaded_and_freshly_built_logs(
    line: str, split: str
) -> None:
    match = _CACHE_PATTERN.search(line)

    assert match is not None
    assert match.group("path") == f"/data/sequences/{split}_abc123"
    assert match.group("split") == split


def _control(*, dataset_manifest: str = "dataset") -> Control:
    return Control(
        label="regular decoder-only SASRec",
        source="RQ12 standard item-state",
        quality={
            "recall@100": 0.135,
            "ndcg@100": 0.051,
            "recall@10": 0.028,
            "ndcg@10": 0.022,
            "coverage@100": 0.728,
        },
        original_users_per_epoch=100,
        expanded_examples_per_epoch=100,
        candidate_targets_per_epoch=0,
        ntp_targets_per_epoch=700,
        input_tokens_per_epoch=800,
        best_epochs="16 / 15 / 20",
        steady_state_targets_per_second=400.0,
        time_through_selected_checkpoint_seconds=300.0,
        total_required_training_wall_seconds=2_500.0,
        dataset_content_manifest_sha256=dataset_manifest,
        validation_cache_manifest_sha256="validation",
        query_cache_manifest_sha256="query",
        evaluator_fingerprint="evaluator",
        scoring_fingerprint="scoring",
    )


def _rq12_control_document() -> dict[str, object]:
    return {
        "research_question": "RQ12 decoder-only query layout",
        "dataset_size": "500m",
        "methods": [
            {
                "method": "standard",
                "artifacts": [
                    {"validation_metrics": {"recall@100": value}}
                    for value in (0.1367, 0.1343, 0.1363)
                ],
                "mean_full_user_metrics": {"recall@100": 0.13468336146286186},
            }
        ],
    }


def _run(
    candidate: QueryCandidate,
    recall: float,
    *,
    ndcg: float = 0.05,
    training_seconds: float = 200.0,
    best_epoch: int = 10,
    final_recall: float | None = None,
    examples: int | None = None,
) -> Run:
    caps = {
        "one_example": 1,
        "truncated_4": 4,
        "truncated_8": 8,
        "truncated_16": 16,
        "required_8": 8,
        "required_16": 16,
    }
    cap = caps.get(candidate.treatment)
    if cap is None:
        cap = int(candidate.treatment.rsplit("_", 1)[1])
    expanded = examples if examples is not None else 100 * cap
    return Run(
        candidate=candidate,
        best_epoch=best_epoch,
        stopped_epoch=20,
        validation_recall=recall,
        validation_ndcg=ndcg,
        validation_curve=tuple(
            (
                epoch,
                recall if epoch == best_epoch else max(0.0, recall - 0.01),
                ndcg if epoch == best_epoch else max(0.0, ndcg - 0.01),
            )
            for epoch in range(1, 21)
        ),
        metrics={
            "recall@100": recall if final_recall is None else final_recall,
            "ndcg@100": ndcg,
            "recall@10": recall / 4,
            "ndcg@10": ndcg / 2,
            "coverage@100": 0.5,
        },
        original_users_per_epoch=100,
        expanded_examples_per_epoch=expanded,
        candidate_targets_per_epoch=expanded,
        ntp_targets_per_epoch=0,
        input_tokens_per_epoch=expanded * 64,
        optimizer_steps_per_epoch=(expanded + 1279) // 1280,
        steady_state_targets_per_second=1_000.0,
        time_through_selected_checkpoint_seconds=training_seconds / 2,
        required_horizon_train_validation_seconds=training_seconds,
        observed_end_to_end_wall_seconds=training_seconds + 10,
        dataset_content_manifest_sha256="dataset",
        train_cache_manifest_sha256=f"train-{candidate.treatment}",
        validation_cache_manifest_sha256="validation",
        query_cache_manifest_sha256="query",
        evaluator_fingerprint="evaluator",
        scoring_fingerprint="scoring",
        artifact_sha256={
            "training_metadata.json": "a",
            "final_metrics.json": "b",
            "sweep.log": "c",
        },
    )


def _resolved_runs() -> list[Run]:
    result = []
    for candidate in rq13_initial_candidates():
        score = 0.13 - abs(candidate.deep_lr - 0.012)
        result.append(
            _run(
                candidate,
                score,
                best_epoch={1: 12, 8: 8, 16: 6}[
                    {
                        "one_example": 1,
                        "truncated_8": 8,
                        "truncated_16": 16,
                        "required_8": 8,
                        "required_16": 16,
                    }[candidate.treatment]
                ],
            )
        )
    return result


def _correctness_audit(runs: list[Run]) -> dict[str, object]:
    checks = (
        "prefix_counts_and_latest_slices",
        "target_exclusion_no_leakage",
        "encoder_attention_mask",
        "gradient_flow",
        "candidate_only_loss",
        "learning_curves",
        "lr_boundary",
    )
    details = {name: {"passed": True} for name in checks}
    treatments = {run.candidate.treatment for run in runs}
    details["prefix_counts_and_latest_slices"].update(
        {
            "source_history_matches": {name: True for name in treatments},
            "expanded_examples": {name: 1 for name in treatments},
            "cache_names": {name: name for name in treatments},
            "cache_files_sha256": {name: {"file": "hash"} for name in treatments},
            "cap1_is_latest_cap8_slice": True,
            "cap8_is_latest_cap16_slice": True,
            "source_file_count": 1,
            "source_files_manifest_sha256": "source",
        }
    )
    details["target_exclusion_no_leakage"].update(
        {
            "target_only_query_max_delta": 0.0,
            "cached_target_position": "final event of each cached sequence",
        }
    )
    details["encoder_attention_mask"].update(
        {
            "encoder_is_bidirectional": True,
            "future_history_changes_earlier_state": True,
            "other_user_history_max_delta": 0.0,
            "production_run_name": "run",
            "production_experiment_class": "MuTransferCrossAttentionGenerationExperiment",
            "production_query_architecture": "encoder_decoder",
            "production_window": "bounded_prefix",
            "production_prefix_length_rule": "truncated",
            "production_prefix_cap": 16,
            "production_targets_class": "NextItemTargets",
            "production_criterion_class": "TwoTowerLoss",
            "production_optimizer_class": "Adam",
        }
    )
    details["gradient_flow"].update(
        {
            "initial_readout_l1": 0,
            "bootstrap_readout_gradient_l1": 1.0,
            "bootstrap_memory_encoder_gradient_l1": 0,
            "bootstrap_cross_attention_gradient_l1": 0,
            "bootstrapped_readout_l1": 1.0,
            "memory_encoder_gradient_l1": 1.0,
            "cross_attention_gradient_l1": 1.0,
        }
    )
    details["candidate_only_loss"].update(
        {
            "all_candidate_targets_equal_examples": True,
            "all_ntp_targets_zero": True,
            "micro_candidate_targets_per_example": 1,
            "duplicated_batch_loss_delta": 0.0,
            "reduction": "proxy",
        }
    )
    curves = {
        run.candidate.run_name: [
            {"epoch": epoch, "recall@100": recall, "ndcg@100": ndcg}
            for epoch, recall, ndcg in run.validation_curve
        ]
        for run in runs
    }
    selected = {
        treatment: max(
            (run for run in runs if run.candidate.treatment == treatment),
            key=lambda run: (run.validation_recall, run.validation_ndcg),
        )
        for treatment in treatments
    }
    crossings = {}
    for treatment, reference in (
        ("truncated_8", "one_example"),
        ("truncated_16", "truncated_8"),
    ):
        run = selected[treatment]
        threshold = selected[reference].validation_recall
        first = next(
            (
                point
                for point in curves[run.candidate.run_name]
                if point["recall@100"] >= threshold
            ),
            curves[run.candidate.run_name][run.best_epoch - 1],
        )
        crossings[f"{treatment}_vs_{reference}"] = {
            "threshold_recall@100": min(threshold, first["recall@100"]),
            "first_matching_epoch": first["epoch"],
            "recall@100": first["recall@100"],
            "selected_epoch": run.best_epoch,
            "selected_recall@100": max(run.validation_recall, first["recall@100"]),
        }
    details["learning_curves"].update(
        {
            "runs": curves,
            "selected_threshold_crossings": crossings,
        }
    )
    details["lr_boundary"].update(
        {
            "required_followups": [],
            "surfaces": {
                treatment: [
                    {
                        "run_name": run.candidate.run_name,
                        "deep_learning_rate": run.candidate.deep_lr,
                        "best_epoch": run.best_epoch,
                        "validation_recall@100": run.validation_recall,
                    }
                    for run in runs
                    if run.candidate.treatment == treatment
                ]
                for treatment in treatments
            },
        }
    )
    return {
        "schema_version": 1,
        "research_question": "RQ13 encoder-decoder prefix expansion",
        "dataset_size": "500m",
        "status": "passed",
        "checks": details,
        "run_artifacts": {run.candidate.run_name: run.artifact_sha256 for run in runs},
        "implementation_sha256": current_implementation_sha256(),
    }


def test_empty_surface_is_partial_and_requests_exact_initial_grid() -> None:
    bundle = build_report_bundle([], _control())

    assert bundle.evidence["claims_status"] == "pending_artifacts"
    assert bundle.evidence["required_followups"] == [
        candidate.run_name for candidate in rq13_initial_candidates()
    ]
    assert (
        bundle.evidence["missing_initial_artifacts"]
        == bundle.evidence["required_followups"]
    )
    assert bundle.evidence["required_boundary_followups"] == []
    assert bundle.reader_markdown is None
    assert "## RQ13" in bundle.tuning_markdown


def test_cap4_extension_fits_validation_only_and_requests_selected_cap_grid() -> None:
    runs = [
        (
            replace(run, input_tokens_per_epoch=14_960)
            if run.candidate.treatment == "truncated_16"
            else run
        )
        for run in _resolved_runs()
    ]
    for candidate in rq13_cap4_candidates():
        runs.append(
            _run(candidate, 0.107 - abs(candidate.deep_lr - 0.012), final_recall=0.99)
        )

    bundle = build_report_bundle(
        runs,
        _control(),
        cap_extension=True,
        rq12_evidence_document=_rq12_control_document(),
        eligible_target_counts=[100] * 10,
    )

    assert bundle.evidence["cap_fit"]["selected_cap"] == 32
    assert bundle.evidence["cap_fit"]["fit_points"]["4"] == pytest.approx(0.107)
    assert (
        bundle.evidence["cap_fit"]["reader_success_target"]["metric"]
        == "mean full-user Recall@100"
    )
    assert bundle.evidence["cap_fit"]["status"] == "stage_one_audit_required"
    assert bundle.evidence["required_followups"] == []
    assert bundle.evidence["cap_fit"]["proposed_followups"] == [
        candidate.run_name for candidate in make_selected_cap_candidates(32)
    ]
    assert bundle.diagnostics_markdown is not None
    assert bundle.diagnostics_markdown.count("| -0.003/") == 8
    assert "sensitivity, not a confidence interval" in bundle.diagnostics_markdown

    audited = build_report_bundle(
        runs,
        _control(),
        cap_extension=True,
        rq12_evidence_document=_rq12_control_document(),
        eligible_target_counts=[100] * 10,
        correctness_audit=_correctness_audit(runs),
    )
    assert audited.evidence["cap_fit"]["status"] == "selected_cap_pending"
    assert audited.evidence["required_followups"] == [
        candidate.run_name for candidate in make_selected_cap_candidates(32)
    ]
    assert (
        audited.evidence["cap_fit"]["input_bindings"]["stage_one_correctness_audit"][
            "status"
        ]
        == "passed"
    )


def test_completed_selected_cap_keeps_stage_one_audit_and_requests_lr_boundary() -> (
    None
):
    stage_one_runs = [
        (
            replace(run, input_tokens_per_epoch=14_960)
            if run.candidate.treatment == "truncated_16"
            else run
        )
        for run in _resolved_runs()
    ]
    stage_one_runs.extend(
        _run(candidate, 0.107 - abs(candidate.deep_lr - 0.012))
        for candidate in rq13_cap4_candidates()
    )
    fitted_candidates = make_selected_cap_candidates(32)
    completed = [
        _run(candidate, {0.006: 0.151, 0.012: 0.149, 0.024: 0.147}[candidate.deep_lr])
        for candidate in fitted_candidates
    ]

    bundle = build_report_bundle(
        [*stage_one_runs, *completed],
        _control(),
        cap_extension=True,
        rq12_evidence_document=_rq12_control_document(),
        eligible_target_counts=[100] * 10,
        correctness_audit=_correctness_audit(stage_one_runs),
    )

    expected_boundary = make_boundary_candidate(fitted_candidates[0], "low", 1)
    assert bundle.evidence["cap_fit"]["status"] == "selected_cap_pending"
    assert bundle.evidence["missing_initial_artifacts"] == []
    assert bundle.evidence["required_boundary_followups"] == [
        expected_boundary.run_name
    ]
    assert bundle.evidence["required_followups"] == [expected_boundary.run_name]
    assert bundle.evidence["surface_winners"]["selected_cap_32"]["run_name"] == (
        fitted_candidates[0].run_name
    )


def test_final_all_artifact_audit_closes_resolved_selected_cap_report() -> None:
    stage_one_runs = [
        (
            replace(run, input_tokens_per_epoch=14_960)
            if run.candidate.treatment == "truncated_16"
            else run
        )
        for run in _resolved_runs()
    ]
    stage_one_runs.extend(
        _run(candidate, 0.107 - abs(candidate.deep_lr - 0.012))
        for candidate in rq13_cap4_candidates()
    )
    fitted_candidates = make_selected_cap_candidates(32)
    selected_cap_runs = [
        _run(candidate, {0.006: 0.151, 0.012: 0.149, 0.024: 0.147}[candidate.deep_lr])
        for candidate in fitted_candidates
    ]
    boundary = _run(make_boundary_candidate(fitted_candidates[0], "low", 1), 0.150)
    all_runs = [*stage_one_runs, *selected_cap_runs, boundary]

    bundle = build_report_bundle(
        all_runs,
        _control(),
        cap_extension=True,
        rq12_evidence_document=_rq12_control_document(),
        eligible_target_counts=[100] * 10,
        correctness_audit=_correctness_audit(all_runs),
    )

    bindings = bundle.evidence["cap_fit"]["input_bindings"]
    assert bundle.evidence["cap_fit"]["status"] == "resolved"
    assert bundle.evidence["required_followups"] == []
    assert bundle.evidence["selected"]["selected_cap_32"]["run_name"] == (
        fitted_candidates[0].run_name
    )
    assert bundle.evidence["correctness_audit"]["status"] == "passed"
    assert bindings["stage_one_correctness_audit"]["status"] == (
        "covered_by_final_audit"
    )
    assert bindings["final_correctness_audit"]["artifact_sha256"] == (
        bundle.evidence["correctness_audit"]["artifact_sha256"]
    )
    assert bundle.reader_markdown is not None


def test_cap4_extension_requires_current_audit_before_cap4_runs() -> None:
    runs = _resolved_runs()

    unaudited = build_report_bundle(
        runs,
        _control(),
        cap_extension=True,
    )
    assert unaudited.evidence["cap_fit"] == {"status": "pending_cap4"}
    assert unaudited.evidence["correctness_audit"]["status"] == "failed"

    audited = build_report_bundle(
        runs,
        _control(),
        cap_extension=True,
        correctness_audit=_correctness_audit(runs),
    )
    assert audited.evidence["correctness_audit"]["status"] == "passed"
    assert audited.evidence["correctness_audit"]["required"] is True


def test_selection_uses_recall_then_ndcg_then_lower_training_time() -> None:
    runs = _resolved_runs()
    one = [run for run in runs if run.candidate.treatment == "one_example"]
    runs = [run for run in runs if run.candidate.treatment != "one_example"]
    runs.extend(
        [
            replace(one[0], validation_recall=0.14, validation_ndcg=0.05),
            replace(
                one[1],
                validation_recall=0.14,
                validation_ndcg=0.06,
                required_horizon_train_validation_seconds=300.0,
            ),
            replace(
                one[2],
                validation_recall=0.14,
                validation_ndcg=0.06,
                required_horizon_train_validation_seconds=100.0,
            ),
        ]
    )

    bundle = build_report_bundle(runs, _control())

    assert (
        bundle.evidence["surface_winners"]["one_example"]["deep_learning_rate"] == 0.024
    )
    assert bundle.evidence["selected"] == {}


def test_geometric_boundary_followups_continue_until_winner_is_interior() -> None:
    runs = _resolved_runs()
    runs = [
        (
            replace(run, validation_recall=run.candidate.deep_lr)
            if run.candidate.treatment == "one_example"
            else run
        )
        for run in runs
    ]

    anchor = next(
        candidate
        for candidate in rq13_initial_candidates()
        if candidate.treatment == "one_example" and candidate.deep_lr == 0.024
    )
    first = build_report_bundle(runs, _control())
    assert first.evidence["required_followups"] == [
        make_boundary_candidate(anchor, "high", 1).run_name
    ]
    assert first.evidence["missing_initial_artifacts"] == []
    assert (
        first.evidence["required_boundary_followups"]
        == first.evidence["required_followups"]
    )

    boundary_one = _run(make_boundary_candidate(anchor, "high", 1), 0.048)
    second = build_report_bundle([*runs, boundary_one], _control())
    assert second.evidence["required_followups"] == [
        make_boundary_candidate(anchor, "high", 2).run_name
    ]

    boundary_two = _run(make_boundary_candidate(anchor, "high", 2), 0.04)
    final = build_report_bundle([*runs, boundary_one, boundary_two], _control())
    assert final.evidence["required_followups"] == []
    assert final.evidence["selected"]["one_example"]["deep_learning_rate"] == 0.048


def test_geometric_boundary_followups_extend_the_low_side() -> None:
    runs = [
        (
            replace(run, validation_recall=0.15 - run.candidate.deep_lr)
            if run.candidate.treatment == "one_example"
            else run
        )
        for run in _resolved_runs()
    ]
    anchor = next(
        candidate
        for candidate in rq13_initial_candidates()
        if candidate.treatment == "one_example" and candidate.deep_lr == 0.006
    )

    first = build_report_bundle(runs, _control())
    assert first.evidence["required_boundary_followups"] == [
        make_boundary_candidate(anchor, "low", 1).run_name
    ]

    boundary_one = _run(make_boundary_candidate(anchor, "low", 1), 0.147)
    second = build_report_bundle([*runs, boundary_one], _control())
    assert second.evidence["required_boundary_followups"] == [
        make_boundary_candidate(anchor, "low", 2).run_name
    ]

    boundary_two = _run(make_boundary_candidate(anchor, "low", 2), 0.14)
    final = build_report_bundle([*runs, boundary_one, boundary_two], _control())
    assert final.evidence["required_followups"] == []
    assert final.evidence["selected"]["one_example"]["deep_learning_rate"] == 0.003


def test_unneeded_or_noncontiguous_boundary_artifact_is_rejected() -> None:
    runs = _resolved_runs()
    anchor = next(
        candidate
        for candidate in rq13_initial_candidates()
        if candidate.treatment == "one_example" and candidate.deep_lr == 0.024
    )

    with pytest.raises(Rq13ReportError, match="unneeded, noncontiguous"):
        build_report_bundle(
            [*runs, _run(make_boundary_candidate(anchor, "high", 1), 0.20)],
            _control(),
        )

    high_winner = [
        (
            replace(run, validation_recall=run.candidate.deep_lr)
            if run.candidate.treatment == "one_example"
            else run
        )
        for run in runs
    ]
    with pytest.raises(Rq13ReportError, match="unneeded, noncontiguous"):
        build_report_bundle(
            [*high_winner, _run(make_boundary_candidate(anchor, "high", 2), 0.20)],
            _control(),
        )


def test_ready_reader_has_control_and_separate_quality_and_efficiency_tables() -> None:
    bundle = build_report_bundle(_resolved_runs(), _control())

    assert bundle.evidence["claims_status"] == "ready_for_user_validation"
    assert bundle.evidence["selected_method"]["run_name"].startswith(
        "g1_rq13_one_example_"
    )
    assert bundle.reader_markdown is not None
    assert bundle.reader_markdown.count("| architecture |") == 2
    assert "regular decoder-only SASRec" in bundle.reader_markdown
    assert "original users/epoch" in bundle.reader_markdown
    assert "candidate targets/epoch" in bundle.reader_markdown
    assert "NTP targets/epoch" in bundle.reader_markdown
    assert "total required training wall" in bundle.reader_markdown
    assert "deep LR" not in bundle.reader_markdown
    assert "| runs |" not in bundle.reader_markdown.lower()
    assert (
        bundle.reader_markdown.count("| **encoder-decoder** | **no expansion** |") == 2
    )
    assert re.search(r"\b0\.\d{3}\b", bundle.reader_markdown)
    assert all(
        not line or line.startswith(("#", "|"))
        for line in bundle.reader_markdown.splitlines()
    )
    assert bundle.tuning_markdown.count("### ") == 5


def test_control_and_treatment_cache_compatibility_fail_closed() -> None:
    runs = _resolved_runs()

    with pytest.raises(Rq13ReportError, match="incompatible RQ12 control"):
        build_report_bundle(
            [replace(runs[0], query_cache_manifest_sha256="other-query")],
            _control(),
        )

    same_treatment = [run for run in runs if run.candidate.treatment == "truncated_8"]
    with pytest.raises(Rq13ReportError, match="cache identity varies"):
        build_report_bundle(
            [
                replace(same_treatment[0], train_cache_manifest_sha256="other-train"),
                *same_treatment[1:],
            ],
            _control(),
        )


def test_unexpected_epoch_order_or_large_quality_regression_requires_diagnostics() -> (
    None
):
    runs = _resolved_runs()
    adjusted = []
    for run in runs:
        if run.candidate.treatment == "truncated_8":
            adjusted.append(
                replace(run, best_epoch=14, metrics={**run.metrics, "recall@100": 0.10})
            )
        elif run.candidate.treatment == "truncated_16":
            adjusted.append(replace(run, best_epoch=16))
        else:
            adjusted.append(run)

    bundle = build_report_bundle(adjusted, _control())

    assert bundle.evidence["claims_status"] == "diagnostics_required"
    checks = bundle.evidence["acceptance_checks"]
    assert checks["truncated_selected_epoch_order"] is False
    assert "truncated_8" in checks["quality_regressions_past_recall_band"]
    assert bundle.evidence["required_diagnostics"]


def test_later_cap16_peak_is_explained_when_it_reaches_cap8_quality_earlier(
    tmp_path: Path,
) -> None:
    adjusted = []
    for run in _resolved_runs():
        if run.candidate.treatment == "one_example" and run.candidate.deep_lr == 0.012:
            curve = tuple(
                (epoch, 0.0766 if epoch == 3 else 0.07, 0.0331 if epoch == 3 else 0.03)
                for epoch in range(1, 21)
            )
            adjusted.append(
                replace(
                    run,
                    best_epoch=3,
                    validation_recall=0.0766,
                    validation_ndcg=0.0331,
                    validation_curve=curve,
                )
            )
        elif (
            run.candidate.treatment == "truncated_8" and run.candidate.deep_lr == 0.012
        ):
            curve = tuple(
                (
                    epoch,
                    {1: 0.0814, 5: 0.1164}.get(epoch, 0.10),
                    {1: 0.0303, 5: 0.0475}.get(epoch, 0.04),
                )
                for epoch in range(1, 21)
            )
            adjusted.append(
                replace(
                    run,
                    best_epoch=5,
                    validation_recall=0.1164,
                    validation_ndcg=0.0475,
                    validation_curve=curve,
                )
            )
        elif (
            run.candidate.treatment == "truncated_16" and run.candidate.deep_lr == 0.012
        ):
            curve = tuple(
                (
                    epoch,
                    {4: 0.1167, 7: 0.1233}.get(epoch, 0.11),
                    {4: 0.0456, 7: 0.0493}.get(epoch, 0.04),
                )
                for epoch in range(1, 21)
            )
            adjusted.append(
                replace(
                    run,
                    best_epoch=7,
                    validation_recall=0.1233,
                    validation_ndcg=0.0493,
                    validation_curve=curve,
                )
            )
        elif run.candidate.treatment in {
            "one_example",
            "truncated_8",
            "truncated_16",
        }:
            fallback_recall = {
                "one_example": 0.06,
                "truncated_8": 0.07,
                "truncated_16": 0.08,
            }[run.candidate.treatment]
            adjusted.append(
                replace(
                    run,
                    validation_recall=fallback_recall,
                    validation_ndcg=0.02,
                    validation_curve=tuple(
                        (
                            epoch,
                            (
                                fallback_recall
                                if epoch == run.best_epoch
                                else fallback_recall - 0.01
                            ),
                            0.02 if epoch == run.best_epoch else 0.01,
                        )
                        for epoch in range(1, 21)
                    ),
                )
            )
        else:
            adjusted.append(run)

    unverified = build_report_bundle(adjusted, _control())

    assert unverified.evidence["claims_status"] == "diagnostics_required"
    assert unverified.reader_markdown is None
    assert any(
        "saved correctness audit" in diagnostic
        for diagnostic in unverified.evidence["required_diagnostics"]
    )

    scratchpad = tmp_path / "scratchpad"
    scratchpad.mkdir()
    stale_reader = scratchpad / "rq13_prefix_expansion_reader_500m.md"
    stale_reader.write_text("stale claims\n")
    write_report_bundle(unverified, scratchpad, scratchpad / "evidence")
    assert not stale_reader.exists()

    bundle = build_report_bundle(
        adjusted, _control(), correctness_audit=_correctness_audit(adjusted)
    )

    cap8_resolution = bundle.evidence["resolved_diagnostics"][
        "truncated_8_selected_epoch_vs_no_expansion"
    ]
    cap16_resolution = bundle.evidence["resolved_diagnostics"][
        "truncated_16_selected_epoch_vs_truncated_8"
    ]
    assert (
        bundle.evidence["acceptance_checks"]["truncated_selected_epoch_order"] is False
    )
    assert (
        bundle.evidence["acceptance_checks"][
            "truncated_8_selected_epoch_before_no_expansion"
        ]
        is False
    )
    assert (
        bundle.evidence["acceptance_checks"][
            "truncated_16_selected_epoch_before_truncated_8"
        ]
        is False
    )
    assert (
        cap8_resolution[
            "truncated_8_first_epoch_at_or_above_no_expansion_selected_recall"
        ]
        == 1
    )
    assert (
        cap16_resolution[
            "truncated_16_first_epoch_at_or_above_truncated_8_selected_recall"
        ]
        == 4
    )
    assert cap16_resolution["truncated_16_selected_epoch"] == 7
    assert bundle.evidence["required_diagnostics"] == []
    assert bundle.evidence["claims_status"] == "ready_for_user_validation"
    assert bundle.diagnostics_markdown is not None
    assert (
        "first epoch at preceding row's selected quality" in bundle.diagnostics_markdown
    )
    assert "| 0.012 | 5 / 0.1164 | 7 / 0.1233 |" in bundle.diagnostics_markdown


def test_material_truncated_16_regression_against_truncated_8_requires_diagnostics() -> (
    None
):
    runs = _resolved_runs()
    adjusted = []
    for run in runs:
        if run.candidate.treatment == "one_example":
            metrics = {**run.metrics, "recall@100": 0.130}
        elif run.candidate.treatment == "truncated_8":
            metrics = {**run.metrics, "recall@100": 0.150}
        elif run.candidate.treatment == "truncated_16":
            metrics = {**run.metrics, "recall@100": 0.140}
        else:
            metrics = run.metrics
        adjusted.append(replace(run, metrics=metrics))

    bundle = build_report_bundle(adjusted, _control())

    checks = bundle.evidence["acceptance_checks"]
    assert checks["truncated_16_recall_difference_from_truncated_8"] == pytest.approx(
        -0.01
    )
    assert checks["truncated_16_quality_regression_past_recall_band"] is True
    assert bundle.evidence["claims_status"] == "diagnostics_required"
    assert any(
        "truncated 16 regressed materially versus truncated 8" in diagnostic
        for diagnostic in bundle.evidence["required_diagnostics"]
    )


def _write_artifact(logs: Path, candidate: QueryCandidate) -> None:
    directory = logs / candidate.run_name
    directory.mkdir(parents=True)
    cap = {
        "one_example": 1,
        "truncated_8": 8,
        "truncated_16": 16,
        "required_8": 8,
        "required_16": 16,
    }[candidate.treatment]
    rule = "required" if candidate.treatment.startswith("required") else "truncated"
    examples = 100 * cap
    architecture = {
        "query_architecture": "encoder_decoder",
        "prefix_length_rule": rule,
        "prefix_cap": cap,
        "query_slots_shared": False,
        "include_history_memory": False,
        "num_query_slots": 4,
        "original_users_per_epoch": 100,
        "expanded_examples_per_epoch": examples,
        "candidate_targets_per_epoch": examples,
        "ntp_targets_per_epoch": 0,
        "input_tokens_per_epoch": examples * 64,
    }
    invariants = {
        "experiment_class": "MuTransferCrossAttentionGenerationExperiment",
        "dataset_size": "500m",
        "user_sample": None,
        "event_type_filter": "like",
        "min_item_interactions_per_item": 5,
        "drop_unmapped_items": True,
        "validation_interval_seconds": 604800,
        "day_range": {"start_day": 0, "end_day": 300},
        "batch_size": 1280,
        "physical_batch_size": 1280,
        "gradient_accumulation_steps": 1,
        "effective_batch_size": 1280,
        "model_dim": 64,
        "item_embedding_dim": 64,
        "max_seq_len": 128,
        "window": "bounded_prefix",
        "bos": False,
        "cls_token": False,
        "cls_token_mode": "none",
        "timestamp_delta": "bins",
        "timestamp_combination": "add",
        "timestamp_num_bins": 16,
        "timestamp_bin_semantics_revision": 2,
        "per_layer_item_embeddings": False,
        "negative_sampling": "random",
        "num_in_batch_negatives": 512,
        "logq_correction": "yi2019",
        "random_negative_fraction": 0.5,
        "logq_alpha": 0.01,
        "correct_positive_logq": False,
        "mask_false_negatives": False,
        "exclude_own_group_negatives": False,
        "dense_random_negative_scores": False,
        "eval_ks": [10, 50, 100],
        "eval_max_users": 20000,
        "eval_every_n_epochs": 1,
        "early_stopping_metric": "recall@100",
        "early_stopping_metric_prefix": "epoch/val_true",
        "selection_k": 100,
        "evaluation_catalog": "all",
        "exclude_seen_from_evaluation": False,
        "restore_best_weights": True,
        "lr_schedule_horizon_epochs": 20,
        "lr_schedule": {"shape": "linear"},
        **architecture,
    }
    metadata = {
        "training_semantics_revision": 2,
        "dataset_size": "500m",
        "seed": 42,
        "num_epochs": 20,
        "max_epochs": 20,
        "epochs_trained": 20,
        "best_epoch": 2,
        "stopped_epoch": 20,
        "early_stopped": False,
        "lr_horizon_complete": True,
        "selection_resolved": True,
        "batch_size": 1280,
        "physical_batch_size": 1280,
        "gradient_accumulation_steps": 1,
        "effective_batch_size": 1280,
        "model_dim": 64,
        "item_embedding_dim": 64,
        "embedding_learning_rate": 0.064,
        "deep_learning_rate": candidate.deep_lr,
        "targets_per_epoch": examples,
        "tokens_per_epoch": examples * 64,
        **architecture,
        "transfer_invariants": invariants,
    }
    metrics = {
        "recall@100": 0.13,
        "ndcg@100": 0.05,
        "recall@10": 0.03,
        "ndcg@10": 0.02,
        "coverage@100": 0.5,
        "num_users": 37018,
    }
    start = datetime(2026, 1, 1)
    lines = [f"{start:%Y-%m-%d %H:%M:%S},000 - INFO - Prepared stage 'training'"]
    for epoch in range(20):
        recall = 0.14 if epoch == 1 else 0.13
        lines.append(
            f"{start + timedelta(seconds=epoch + 1):%Y-%m-%d %H:%M:%S}.000 | INFO - "
            f"epoch {epoch} finished timing.train_epoch_time=1.0 "
            f"timing.val_inference_time=0.1 timing.val_save_time=0.01 "
            f"epoch/val_true.recall@100={recall} epoch/val_true.ndcg@100=0.05"
        )
    lines.append(
        f"{start + timedelta(seconds=30):%Y-%m-%d %H:%M:%S},000 - INFO - Final metrics ({metrics!r})"
    )
    (directory / "training_metadata.json").write_text(json.dumps(metadata))
    (directory / "final_metrics.json").write_text(json.dumps(metrics))
    (directory / "sweep.log").write_text("\n".join(lines) + "\n")


def test_collection_rejects_recipe_or_exact_count_metadata_mismatch(
    tmp_path: Path,
) -> None:
    logs = tmp_path / "logs"
    candidate = rq13_initial_candidates()[0]
    _write_artifact(logs, candidate)

    with pytest.raises(Rq13ReportError, match="recipe-incompatible"):
        collect_report_bundle(
            logs, _control(), verify_recipe=lambda _path, _candidate: False
        )

    with pytest.raises(Rq13ReportError, match="count metadata"):
        collect_report_bundle(
            logs,
            _control(),
            verify_recipe=lambda _path, _candidate: True,
            expected_counts={
                candidate.treatment: {
                    "original_users_per_epoch": 100,
                    "expanded_examples_per_epoch": 999,
                    "candidate_targets_per_epoch": 999,
                    "ntp_targets_per_epoch": 0,
                    "input_tokens_per_epoch": 999,
                }
            },
        )
