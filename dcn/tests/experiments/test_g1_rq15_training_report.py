from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from experiments.g1_sasrec_item_ids_likes.analysis.rq15_training_candidates import (
    Rq15Candidate,
    _best_source_epoch,
    candidate_by_run,
    initial_candidates,
    make_auxiliary_weight_candidate,
    make_boundary_candidate,
    source_candidates,
)
from experiments.g1_sasrec_item_ids_likes.analysis.rq15_training_report import (
    ACCEPTANCE_CRITERION,
    Rq15ReportError,
    Run,
    _best,
    _select_checkpoint,
    _selected_validation_point,
    _validation_best_epoch,
    build_report_bundle,
    validate_treatment_metadata,
    write_report_bundle,
)


def test_source_and_report_match_trainer_recall_then_earliest_epoch() -> None:
    report_curve = (
        (1, 0.2, 0.05),
        (2, 0.2, 0.06),
        (3, 0.2, 0.06),
    )
    source_curve = tuple((*point, float(point[0])) for point in report_curve)

    assert _validation_best_epoch(report_curve) == 1
    assert _best_source_epoch(source_curve) == 1


def test_selected_epoch_accepts_a_later_maximum_tied_by_log_rounding() -> None:
    curve = (
        (1, 0.1257, 0.0500),
        (2, 0.1257, 0.0510),
        (3, 0.1256, 0.0520),
    )

    assert _selected_validation_point(curve, 2, "rounded-run") == curve[1]


def test_selected_epoch_rejects_strictly_submaximum_logged_recall() -> None:
    curve = (
        (1, 0.1257, 0.0500),
        (2, 0.1256, 0.0510),
        (3, 0.1255, 0.0520),
    )

    with pytest.raises(Rq15ReportError, match="invalid selected validation epoch"):
        _selected_validation_point(curve, 2, "submaximum-run")


def _run(
    role: str,
    recall: float,
    *,
    candidate: Rq15Candidate | None = None,
    validation_ndcg: float = 0.05,
    best_epoch: int = 6,
    checkpoint_seconds: float = 60.0,
    horizon_seconds: float = 180.0,
    ntp_targets: int = 0,
    checkpoint_sha256: str | None = None,
    run_name: str | None = None,
    embedding_lr: float | None = None,
    deep_lr: float | None = None,
) -> Run:
    run_name = run_name or (
        candidate.run_name
        if candidate is not None
        else "g1_rq14_distinct_cls_only_control"
        if role == "scratch_candidate_only"
        else "g1_rq15_first_stage_checkpoint"
    )
    candidate_targets = 0 if role == "checkpoint_pretraining" else 100
    return Run(
        role=role,
        run_name=run_name,
        candidate=candidate,
        embedding_lr=(
            embedding_lr if embedding_lr is not None else 0.064 if candidate is None else candidate.embedding_lr
        ),
        deep_lr=(
            deep_lr if deep_lr is not None else 0.0015 if role == "scratch_candidate_only" and candidate is None else 0.048 if candidate is None else candidate.deep_lr
        ),
        auxiliary_ntp_weight=(
            candidate.auxiliary_ntp_weight
            if role == "auxiliary_ntp" and candidate is not None
            else 0.0
        ),
        best_epoch=best_epoch,
        stopped_epoch=20,
        validation_recall=recall,
        validation_ndcg=validation_ndcg,
        validation_curve=tuple(
            (
                epoch,
                recall if epoch == best_epoch else recall - abs(epoch - best_epoch) / 100,
                validation_ndcg
                if epoch == best_epoch
                else validation_ndcg - 0.001,
            )
            for epoch in range(1, 21)
        ),
        metrics={
            "recall@100": recall,
            "ndcg@100": validation_ndcg,
            "recall@10": recall / 4,
            "ndcg@10": validation_ndcg / 2,
            "coverage@100": 0.4,
        },
        original_users_per_epoch=(1_000 if role == "checkpoint_pretraining" else 100),
        expanded_examples_per_epoch=(1_000 if role == "checkpoint_pretraining" else 100),
        candidate_targets_per_epoch=candidate_targets,
        ntp_targets_per_epoch=ntp_targets,
        input_tokens_per_epoch=6_000,
        optimizer_steps_per_epoch=5,
        steady_state_candidate_targets_per_second=(
            None if candidate_targets == 0 else 1_000.0
        ),
        steady_state_total_targets_per_second=10_000.0,
        time_through_selected_checkpoint_seconds=checkpoint_seconds,
        required_horizon_train_validation_seconds=horizon_seconds,
        observed_end_to_end_wall_seconds=horizon_seconds + 10,
        dataset_fingerprint="dataset",
        train_cache_fingerprint=(
            "checkpoint-train" if role == "checkpoint_pretraining" else "train"
        ),
        validation_cache_fingerprint="validation",
        query_cache_fingerprint="query",
        evaluator_fingerprint="evaluator",
        scoring_fingerprint="scoring",
        checkpoint_sha256=checkpoint_sha256,
        artifact_sha256={
            "training_metadata.json": run_name + "-metadata",
            "final_metrics.json": run_name + "-metrics",
            "sweep.log": run_name + "-log",
        },
    )


def _resolved_runs(
    *,
    pretrained_recall: float = 0.142,
    pretrained_seconds: float = 40.0,
    auxiliary_recall: float = 0.138,
) -> tuple[Run, list[Run], list[Run]]:
    scratch = _run(
        "scratch_candidate_only",
        0.135,
        checkpoint_seconds=300.0,
        horizon_seconds=320.0,
    )
    checkpoints = [
        _run(
            "checkpoint_pretraining",
            0.137 - abs(source.deep_lr - 0.048),
            ntp_targets=5_000,
            checkpoint_sha256=("checkpoint" if source.deep_lr == 0.048 else f"checkpoint-{source.deep_lr}"),
            horizon_seconds=180.0,
            run_name=source.run_name,
            embedding_lr=source.embedding_lr,
            deep_lr=source.deep_lr,
        )
        for source in source_candidates()
    ]
    runs = []
    for candidate in initial_candidates():
        if (
            candidate.training_method == "scratch_candidate_only"
            and candidate.embedding_lr == 0.064
            and candidate.deep_lr == 0.0015
        ):
            continue
        expected_deep = 0.012 if candidate.training_method == "auxiliary_ntp" else 0.0015
        center_penalty = abs(candidate.deep_lr - expected_deep) + abs(candidate.embedding_lr - 0.064)
        if candidate.training_method == "pretrained_finetune":
            runs.append(
                _run(
                    "pretrained_finetune",
                    pretrained_recall - center_penalty,
                    candidate=candidate,
                    checkpoint_seconds=pretrained_seconds,
                    checkpoint_sha256="checkpoint",
                )
            )
        elif candidate.training_method == "auxiliary_ntp":
            runs.append(
                _run(
                    "auxiliary_ntp",
                    auxiliary_recall - center_penalty,
                    candidate=candidate,
                    ntp_targets=5_000,
                )
            )
        else:
            runs.append(
                _run(
                    "scratch_candidate_only",
                    0.135 - center_penalty,
                    candidate=candidate,
                )
            )
    return scratch, checkpoints, runs


def _correctness(scratch: Run, checkpoints: list[Run], runs: list[Run]) -> dict[str, object]:
    checks = {}
    for name in (
        "target_leakage",
        "attention_masks",
        "gradient_flow",
        "separate_loss_normalization_and_counts",
        "checkpoint_copy_identity",
        "config_code_and_artifact_hashes",
    ):
        payload = {"passed": True}
        checks[name] = {
            **payload,
            "artifact_sha256": hashlib.sha256(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        }
    checkpoint = _select_checkpoint(checkpoints)
    grouped = {
        method: [
            run
            for run in (scratch, *runs)
            if run.role == method
        ]
        for method in (
            "scratch_candidate_only",
            "pretrained_finetune",
            "auxiliary_ntp",
        )
    }
    return {
        "schema_version": 1,
        "research_question": "RQ15 decoder-decoder training method",
        "dataset_size": "500m",
        "status": "passed",
        "checks": checks,
        "run_artifacts": {
            run.run_name: run.artifact_sha256
            for run in (scratch, *checkpoints, *runs)
        },
        "implementation_sha256": "current",
        "result_binding": {
            "missing_artifacts": [],
            "required_followups": [],
            "checkpoint_pretraining_run_name": (
                None if checkpoint is None else checkpoint.run_name
            ),
            "surface_winner_run_names": {
                method: _best(method_runs).run_name
                for method, method_runs in grouped.items()
            },
        },
    }


def _bundle(
    scratch: Run,
    checkpoints: list[Run],
    runs: list[Run],
    **kwargs: object,
):
    return build_report_bundle(
        scratch,
        checkpoints,
        runs,
        correctness_evidence=_correctness(scratch, checkpoints, runs),
        current_implementation_hash="current",
        **kwargs,
    )


def test_resolved_report_renders_quality_efficiency_and_preserves_acceptance() -> None:
    scratch, checkpoint, runs = _resolved_runs()

    bundle = _bundle(scratch, checkpoint, runs)

    assert bundle.evidence["claims_status"] == "ready_for_user_validation"
    assert bundle.evidence["acceptance_criterion"] == ACCEPTANCE_CRITERION
    assert bundle.evidence["required_followups"] == []
    assert bundle.evidence["selected_method"]["training_method"] == (
        "pretrained_finetune"
    )
    assert bundle.reader_markdown is not None
    assert "## Candidate-generation quality" in bundle.reader_markdown
    assert "## Training efficiency" in bundle.reader_markdown
    assert "fine-tune time" in bundle.reader_markdown
    assert "cold-start time" in bundle.reader_markdown
    assert "examples/epoch" in bundle.reader_markdown
    assert "input tokens/epoch" in bundle.reader_markdown
    assert "processed examples" in bundle.reader_markdown
    assert "processed candidate targets" in bundle.reader_markdown
    assert "processed NTP targets" in bundle.reader_markdown
    assert "| 20,600 | 600 | 100,000 |" in bundle.reader_markdown
    assert bundle.readme_section_markdown is not None
    assert ACCEPTANCE_CRITERION in bundle.readme_section_markdown
    assert "Include pretraining in total training cost." in (
        bundle.readme_section_markdown
    )
    assert "Selected embedding/deep LRs" in bundle.readme_section_markdown
    assert "same candidate targets and input tokens per epoch" in (
        bundle.readme_section_markdown
    )
    assert "quality/compute tradeoff" in bundle.readme_section_markdown
    assert "## Pretrain then fine-tune" in bundle.tuning_markdown
    assert "## Acceptance diagnostics" in bundle.diagnostics_markdown


def test_user_validation_promotes_resolved_claims_to_complete() -> None:
    scratch, checkpoint, runs = _resolved_runs()

    bundle = _bundle(
        scratch,
        checkpoint,
        runs,
        result_claims_user_validated=True,
    )

    assert bundle.evidence["claims_status"] == "complete"
    assert bundle.evidence["result_claims_user_validated"] is True
    assert bundle.reader_markdown is not None
    assert bundle.readme_section_markdown is not None


def test_efficiency_uses_downstream_examples_when_checkpoint_metadata_omits_them() -> None:
    scratch, checkpoints, runs = _resolved_runs()
    selected_index = next(
        index for index, run in enumerate(checkpoints) if run.deep_lr == 0.048
    )
    checkpoints[selected_index] = replace(
        checkpoints[selected_index],
        original_users_per_epoch=None,
        expanded_examples_per_epoch=None,
    )

    bundle = _bundle(scratch, checkpoints, runs)

    assert bundle.evidence["claims_status"] == "ready_for_user_validation"
    assert "100 pre + 100 fine" in bundle.reader_markdown


def test_missing_artifacts_never_render_reader_claims() -> None:
    scratch, checkpoint, runs = _resolved_runs()

    bundle = build_report_bundle(scratch, checkpoint, runs[:-1])

    assert bundle.evidence["claims_status"] == "pending_artifacts"
    assert bundle.evidence["artifact_audit"]["status"] == "incomplete"
    assert bundle.evidence["artifact_audit"]["checks"][
        "complete_treatment_surface"
    ] is False
    assert bundle.evidence["missing_artifacts"]
    assert bundle.reader_markdown is None
    assert bundle.readme_section_markdown is None


def test_two_axis_outer_winner_requires_both_orthogonal_boundaries() -> None:
    scratch, checkpoint, runs = _resolved_runs()
    runs = [
        replace(
            run,
            validation_recall=(
                0.150
                if run.role == "pretrained_finetune"
                and run.embedding_lr == 0.128
                and run.deep_lr == 0.003
                else run.validation_recall
            ),
        )
        for run in runs
    ]

    bundle = build_report_bundle(scratch, checkpoint, runs)

    assert bundle.evidence["claims_status"] == "pending_boundary"
    followup = bundle.evidence["required_followups"]
    assert len(followup) == 4
    parsed = [candidate_by_run(record["run_name"]) for record in followup]
    assert {candidate.boundary_axis for candidate in parsed} == {"embedding", "deep"}
    assert {(candidate.embedding_lr, candidate.deep_lr) for candidate in parsed} == {
        (0.256, 0.00075),
        (0.256, 0.0015),
        (0.256, 0.003),
        (0.128, 0.006),
    }
    assert bundle.reader_markdown is None


def test_boundary_artifact_is_collectable_and_resolves_or_extends_iteratively() -> None:
    scratch, checkpoint, runs = _resolved_runs()
    high_index = next(
        index
        for index, run in enumerate(runs)
        if run.role == "pretrained_finetune"
        and run.embedding_lr == 0.128
        and run.deep_lr == 0.003
    )
    high = replace(runs[high_index], validation_recall=0.150)
    runs[high_index] = high
    assert high.candidate is not None
    boundary_candidate = make_boundary_candidate(high.candidate, "deep", "high", 1)
    embedding_candidates = [
        make_boundary_candidate(candidate, "embedding", "high", 1)
        for candidate in initial_candidates()
        if candidate.training_method == "pretrained_finetune"
        and candidate.embedding_lr == 0.128
    ]

    resolved_runs = [
        *runs,
        _run(
            "pretrained_finetune",
            0.149,
            candidate=boundary_candidate,
            checkpoint_sha256="checkpoint",
        ),
        *[
            _run(
                "pretrained_finetune",
                0.149,
                candidate=candidate,
                checkpoint_sha256="checkpoint",
            )
            for candidate in embedding_candidates
        ],
    ]
    resolved = _bundle(scratch, checkpoint, resolved_runs)
    assert resolved.evidence["claims_status"] != "pending_boundary"
    assert resolved.evidence["required_followups"] == []

    extending_runs = [
        *runs,
        _run(
            "pretrained_finetune",
            0.151,
            candidate=boundary_candidate,
            checkpoint_sha256="checkpoint",
        ),
        *[
            _run(
                "pretrained_finetune",
                0.149,
                candidate=candidate,
                checkpoint_sha256="checkpoint",
            )
            for candidate in embedding_candidates
        ],
    ]
    extending = build_report_bundle(scratch, checkpoint, extending_runs)
    followup = extending.evidence["required_followups"]
    assert len(followup) == 1
    parsed = [candidate_by_run(record["run_name"]) for record in followup]
    assert {(item.embedding_lr, item.deep_lr) for item in parsed} == {
        (0.128, 0.012),
    }


def test_boundary_frontier_rechecks_deep_axis_after_embedding_winner_moves() -> None:
    scratch, checkpoints, runs = _resolved_runs()
    corner_index = next(
        index
        for index, run in enumerate(runs)
        if run.role == "pretrained_finetune"
        and run.embedding_lr == 0.128
        and run.deep_lr == 0.003
    )
    corner = replace(runs[corner_index], validation_recall=0.150)
    runs[corner_index] = corner
    assert corner.candidate is not None
    embedding = [
        make_boundary_candidate(candidate, "embedding", "high", 1)
        for candidate in initial_candidates()
        if candidate.training_method == "pretrained_finetune"
        and candidate.embedding_lr == 0.128
    ]
    deep = make_boundary_candidate(corner.candidate, "deep", "high", 1)

    first_frontier = [
        *runs,
        *[
            _run(
                "pretrained_finetune",
                0.152 if candidate.deep_lr == 0.003 else 0.149,
                candidate=candidate,
                checkpoint_sha256="checkpoint",
            )
            for candidate in embedding
        ],
        _run(
            "pretrained_finetune",
            0.149,
            candidate=deep,
            checkpoint_sha256="checkpoint",
        ),
    ]
    bundle = build_report_bundle(scratch, checkpoints, first_frontier)

    followups = [
        candidate_by_run(record["run_name"])
        for record in bundle.evidence["required_followups"]
        if record["training_method"] == "pretrained_finetune"
    ]
    assert {(item.embedding_lr, item.deep_lr) for item in followups} == {
        (0.512, 0.00075),
        (0.512, 0.0015),
        (0.512, 0.003),
        (0.256, 0.006),
    }


def test_two_completed_frontiers_replay_in_order() -> None:
    scratch, checkpoints, runs = _resolved_runs()
    corner_index = next(
        index
        for index, run in enumerate(runs)
        if run.role == "pretrained_finetune"
        and run.embedding_lr == 0.128
        and run.deep_lr == 0.003
    )
    corner = replace(runs[corner_index], validation_recall=0.150)
    runs[corner_index] = corner
    assert corner.candidate is not None
    first_embedding_row = [
        make_boundary_candidate(candidate, "embedding", "high", 1)
        for candidate in initial_candidates()
        if candidate.training_method == "pretrained_finetune"
        and candidate.embedding_lr == 0.128
    ]
    first_deep = make_boundary_candidate(corner.candidate, "deep", "high", 1)
    second_embedding_row = [
        replace(
            candidate,
            embedding_lr=0.512,
            boundary_step=2,
        )
        for candidate in first_embedding_row
    ]
    second_deep = replace(
        first_deep,
        embedding_lr=0.256,
    )
    completed = [
        *runs,
        *[
            _run(
                "pretrained_finetune",
                0.152 if candidate.deep_lr == 0.003 else 0.149,
                candidate=candidate,
                checkpoint_sha256="checkpoint",
            )
            for candidate in first_embedding_row
        ],
        _run(
            "pretrained_finetune",
            0.149,
            candidate=first_deep,
            checkpoint_sha256="checkpoint",
        ),
        *[
            _run(
                "pretrained_finetune",
                0.151,
                candidate=candidate,
                checkpoint_sha256="checkpoint",
            )
            for candidate in second_embedding_row
        ],
        _run(
            "pretrained_finetune",
            0.151,
            candidate=second_deep,
            checkpoint_sha256="checkpoint",
        ),
    ]

    bundle = _bundle(scratch, checkpoints, completed)

    assert bundle.evidence["claims_status"] != "pending_boundary"
    assert bundle.evidence["required_followups"] == []


def _pretrained_low_embedding_frontier(runs: list[Run]) -> list[Run]:
    anchors = [
        candidate
        for candidate in initial_candidates()
        if candidate.training_method == "pretrained_finetune"
        and candidate.embedding_lr == 0.032
    ]
    frontier = [
        replace(
            run,
            validation_recall=0.146,
        )
        if run.role == "pretrained_finetune"
        and run.embedding_lr == 0.032
        and run.deep_lr == 0.0015
        else run
        for run in runs
    ]
    for step in range(1, 8):
        candidates = [
            make_boundary_candidate(anchor, "embedding", "low", step)
            for anchor in anchors
        ]
        for candidate in candidates:
            recall = 0.147 + step / 1_000
            if candidate.deep_lr != (0.00075 if step == 7 else 0.0015):
                recall -= 0.001
            frontier.append(
                _run(
                    "pretrained_finetune",
                    recall,
                    candidate=candidate,
                    checkpoint_sha256="checkpoint",
                )
            )
    return frontier


def test_step_seven_winner_requests_frozen_row_and_local_deep_probe() -> None:
    scratch, checkpoints, runs = _resolved_runs(pretrained_recall=0.145)
    runs = _pretrained_low_embedding_frontier(runs)

    bundle = build_report_bundle(scratch, checkpoints, runs)

    pretrained = [
        candidate_by_run(record["run_name"])
        for record in bundle.evidence["required_followups"]
        if record["training_method"] == "pretrained_finetune"
    ]
    assert {(candidate.embedding_lr, candidate.deep_lr) for candidate in pretrained} == {
        (0.0, 0.00075),
        (0.0, 0.0015),
        (0.0, 0.003),
        (0.00025, 0.000375),
    }
    frozen = [candidate for candidate in pretrained if candidate.embedding_lr == 0.0]
    assert all(candidate.boundary_axis == "embedding" for candidate in frozen)
    assert all(candidate.boundary_step == 8 for candidate in frozen)


def test_frozen_embedding_keeps_deep_boundary_resolvable() -> None:
    scratch, checkpoints, runs = _resolved_runs(pretrained_recall=0.145)
    runs = _pretrained_low_embedding_frontier(runs)
    frozen = [
        make_boundary_candidate(candidate, "embedding", "low", 8)
        for candidate in initial_candidates()
        if candidate.training_method == "pretrained_finetune"
        and candidate.embedding_lr == 0.032
    ]
    local_probe = Rq15Candidate(
        "pretrained_finetune",
        embedding_lr=0.00025,
        deep_lr=0.000375,
        stage="lr_boundary",
        boundary_axis="deep",
        boundary_direction="low",
        boundary_step=1,
    )
    runs.extend(
        [
            _run(
                "pretrained_finetune",
                0.156 if candidate.deep_lr == 0.00075 else 0.149,
                candidate=candidate,
                checkpoint_sha256="checkpoint",
            )
            for candidate in frozen
        ]
    )
    runs.append(
        _run(
            "pretrained_finetune",
            0.149,
            candidate=local_probe,
            checkpoint_sha256="checkpoint",
        )
    )

    bundle = build_report_bundle(scratch, checkpoints, runs)

    followups = [
        candidate_by_run(record["run_name"])
        for record in bundle.evidence["required_followups"]
        if record["training_method"] == "pretrained_finetune"
    ]
    assert [(candidate.embedding_lr, candidate.deep_lr) for candidate in followups] == [
        (0.0, 0.000375)
    ]
    assert followups[0].boundary_axis == "deep"


def test_auxiliary_deep_frontier_uses_factor_four() -> None:
    scratch, checkpoints, runs = _resolved_runs(auxiliary_recall=0.138)
    high_index = next(
        index
        for index, run in enumerate(runs)
        if run.role == "auxiliary_ntp"
        and run.embedding_lr == 0.064
        and run.deep_lr == 0.048
    )
    runs[high_index] = replace(runs[high_index], validation_recall=0.150)

    bundle = build_report_bundle(scratch, checkpoints, runs)

    auxiliary = [
        candidate_by_run(record["run_name"])
        for record in bundle.evidence["required_followups"]
        if record["training_method"] == "auxiliary_ntp"
    ]
    assert [(candidate.embedding_lr, candidate.deep_lr) for candidate in auxiliary] == [
        (0.064, 0.192)
    ]


def test_auxiliary_regression_requires_preapproved_weight_followups() -> None:
    scratch, checkpoint, runs = _resolved_runs(auxiliary_recall=0.125)

    bundle = build_report_bundle(scratch, checkpoint, runs)

    assert bundle.evidence["claims_status"] == "pending_auxiliary_weights"
    followups = bundle.evidence["required_followups"]
    assert [item["auxiliary_ntp_weight"] for item in followups] == [0.1, 0.3]
    assert all(item["stage"] == "auxiliary_weight" for item in followups)
    assert all(candidate_by_run(item["run_name"]).stage == "auxiliary_weight" for item in followups)
    assert bundle.reader_markdown is None


def test_auxiliary_weight_artifacts_are_collectable_and_selectable() -> None:
    scratch, checkpoint, runs = _resolved_runs(auxiliary_recall=0.125)
    weight_one = max(
        (run for run in runs if run.role == "auxiliary_ntp"),
        key=lambda run: run.validation_recall,
    )
    assert weight_one.candidate is not None
    weighted = [
        _run(
            "auxiliary_ntp",
            recall,
            candidate=make_auxiliary_weight_candidate(weight_one.candidate, weight),
            ntp_targets=5_000,
        )
        for weight, recall in ((0.1, 0.140), (0.3, 0.139))
    ]

    bundle = _bundle(scratch, checkpoint, [*runs, *weighted])

    assert bundle.evidence["claims_status"] == "ready_for_user_validation"
    assert bundle.evidence["required_followups"] == []
    assert bundle.evidence["surface_winners"]["auxiliary_ntp"][
        "auxiliary_ntp_weight"
    ] == 0.1


def test_auxiliary_weight_preserves_a_selected_boundary_lr() -> None:
    anchor = next(
        candidate
        for candidate in initial_candidates()
        if candidate.training_method == "auxiliary_ntp"
        and candidate.embedding_lr == 0.128
        and candidate.deep_lr == 0.048
    )
    boundary = make_boundary_candidate(anchor, "deep", "high", 2)

    weighted = make_auxiliary_weight_candidate(boundary, 0.3)

    assert weighted.stage == "auxiliary_weight"
    assert weighted.deep_lr == boundary.deep_lr == 0.768
    assert candidate_by_run(weighted.run_name) == weighted


@pytest.mark.parametrize(
    ("pretrained_recall", "pretrained_seconds", "failed_check"),
    [
        (0.130, 40.0, "quality_non_inferior"),
        (0.142, 140.0, "cold_start_faster"),
    ],
)
def test_acceptance_failure_is_diagnostic_only(
    pretrained_recall: float,
    pretrained_seconds: float,
    failed_check: str,
) -> None:
    scratch, checkpoint, runs = _resolved_runs(
        pretrained_recall=pretrained_recall,
        pretrained_seconds=pretrained_seconds,
    )

    bundle = _bundle(scratch, checkpoint, runs)

    assert bundle.evidence["claims_status"] == "acceptance_requires_explanation"
    assert bundle.evidence["acceptance"][failed_check] is False
    assert bundle.reader_markdown is None
    assert bundle.readme_section_markdown is None


def test_probable_metric_improvement_is_an_expectation_not_a_hard_gate() -> None:
    scratch, checkpoint, runs = _resolved_runs(pretrained_recall=0.136)

    bundle = _bundle(scratch, checkpoint, runs)

    assert bundle.evidence["claims_status"] == "ready_for_user_validation"
    assert bundle.evidence["acceptance"]["minimum_acceptance_met"] is True
    assert bundle.evidence["acceptance"]["main_metrics_improved"] is False


def test_complete_results_require_current_artifact_bound_correctness() -> None:
    scratch, checkpoint, runs = _resolved_runs()

    missing = build_report_bundle(scratch, checkpoint, runs)
    assert missing.evidence["claims_status"] == "correctness_audit_required"
    assert missing.reader_markdown is None

    stale = _correctness(scratch, checkpoint, runs)
    stale["run_artifacts"] = {}
    rejected = build_report_bundle(
        scratch,
        checkpoint,
        runs,
        correctness_evidence=stale,
        current_implementation_hash="current",
    )
    assert rejected.evidence["correctness_audit"]["status"] == "stale_or_failed"
    assert rejected.reader_markdown is None

    stale_selection = _correctness(scratch, checkpoint, runs)
    stale_selection["result_binding"]["surface_winner_run_names"][
        "pretrained_finetune"
    ] = "different-run"
    rejected = build_report_bundle(
        scratch,
        checkpoint,
        runs,
        correctness_evidence=stale_selection,
        current_implementation_hash="current",
    )
    assert rejected.evidence["correctness_audit"]["status"] == "stale_or_failed"
    assert rejected.reader_markdown is None

    tampered = _correctness(scratch, checkpoint, runs)
    tampered["checks"]["target_leakage"]["target_only_query_max_delta"] = 1.0
    rejected = build_report_bundle(
        scratch,
        checkpoint,
        runs,
        correctness_evidence=tampered,
        current_implementation_hash="current",
    )
    assert rejected.evidence["correctness_audit"]["status"] == "stale_or_failed"
    assert rejected.reader_markdown is None


def test_bound_targeted_explanation_can_resolve_unexpected_acceptance() -> None:
    scratch, checkpoints, runs = _resolved_runs(pretrained_recall=0.130)
    checkpoint = next(run for run in checkpoints if run.deep_lr == 0.048)
    correctness = _correctness(scratch, checkpoints, runs)
    pending = build_report_bundle(
        scratch,
        checkpoints,
        runs,
        correctness_evidence=correctness,
        current_implementation_hash="current",
    )
    selected = {
        method: max(
            (run for run in runs if run.role == method),
            key=lambda run: run.validation_recall,
        )
        for method in ("pretrained_finetune", "auxiliary_ntp")
    }
    selected_artifacts = {
        name: {
            "run_name": run.run_name,
            "artifact_sha256": run.artifact_sha256,
        }
        for name, run in {
            "scratch_candidate_only": max(
                (run for run in (scratch, *runs) if run.role == "scratch_candidate_only"),
                key=lambda run: run.validation_recall,
            ),
            "checkpoint_pretraining": checkpoint,
            **selected,
        }.items()
    }

    def digest(value: object) -> str:
        return hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    explanation = {
        "schema_version": 1,
        "research_question": "RQ15 decoder-decoder training method",
        "dataset_size": "500m",
        "status": "passed",
        "selected_artifacts": selected_artifacts,
        "acceptance_sha256": digest(pending.evidence["acceptance"]),
        "correctness_audit_sha256": digest(correctness),
        "targeted_evidence": {
            "forgetting_curve_ablation": {
                "passed": True,
                "artifact_sha256": "a" * 64,
            }
        },
        "conclusion": "The targeted ablation reproduces the unexpected gap.",
    }

    resolved = build_report_bundle(
        scratch,
        checkpoints,
        runs,
        correctness_evidence=correctness,
        explanation_evidence=explanation,
        current_implementation_hash="current",
    )

    assert resolved.evidence["claims_status"] == "ready_for_user_validation"
    assert resolved.evidence["experimental_explanation"]["status"] == "passed"
    assert resolved.reader_markdown is not None
    assert "minimum acceptance criterion is not met" in resolved.readme_section_markdown


def test_cross_run_identity_mismatch_is_rejected() -> None:
    scratch, checkpoint, runs = _resolved_runs()
    runs[0] = replace(runs[0], query_cache_fingerprint="different")

    with pytest.raises(Rq15ReportError, match="query cache"):
        build_report_bundle(scratch, checkpoint, runs)


def test_source_objective_caches_and_scoring_are_not_compared_with_downstream() -> None:
    scratch, checkpoints, runs = _resolved_runs()
    checkpoints[0] = replace(
        checkpoints[0],
        train_cache_fingerprint="source-ntp-train",
        validation_cache_fingerprint="source-ntp-validation",
        scoring_fingerprint="source-ntp-scoring",
    )

    bundle = _bundle(scratch, checkpoints, runs)

    assert bundle.evidence["claims_status"] == "ready_for_user_validation"

    runs[0] = replace(runs[0], scoring_fingerprint="foreign-downstream-scoring")
    with pytest.raises(Rq15ReportError, match="scoring identity"):
        build_report_bundle(scratch, checkpoints, runs)

    runs[0] = replace(
        runs[0],
        scoring_fingerprint=scratch.scoring_fingerprint,
        validation_cache_fingerprint="foreign-downstream-validation",
    )
    with pytest.raises(Rq15ReportError, match="validation cache identity"):
        build_report_bundle(scratch, checkpoints, runs)


def test_treatment_metadata_requires_exact_objective_and_checkpoint_identity() -> None:
    candidate = next(
        item
        for item in initial_candidates()
        if item.training_method == "pretrained_finetune"
        and item.embedding_lr == 0.064
        and item.deep_lr == 0.0015
    )
    initialization = {
        "schema_version": 1,
        "checkpoint_sha256": "checkpoint",
        "source_metadata": {
            "dataset_size": "500m",
            "source_recipe_run_name": (
                "g1_rq8_query_standard_s128_e0p064_d0p048_b1280_seed42_cap20_"
                "ts2_boundaryhigh1_r1_500m"
            ),
            "training_objective": "standard_next_item_prediction",
            "max_seq_len": 128,
            "embedding_learning_rate": 0.064,
            "deep_learning_rate": 0.048,
            "batch_size": 1280,
            "seed": 42,
            "horizon_epochs": 20,
        },
        "history_position_count": 128,
        "copied_modules": ["item_embedding", "memory_encoder", "tokenizer"],
        "newly_initialized_modules": [
            "decoder",
            "decoder_query",
            "query_projection",
            "query_slots",
        ],
    }
    architecture = {
        "query_architecture": "decoder_decoder",
        "prefix_length_rule": "truncated",
        "prefix_cap": 1,
        "query_slots_shared": False,
        "include_history_memory": False,
        "num_query_slots": 4,
        "training_method": "pretrained_finetune",
        "candidate_targets_per_epoch": 100,
        "ntp_targets_per_epoch": 0,
        "auxiliary_ntp_weight": 0.0,
        "loss_normalization": "candidate_and_ntp_separately_mean_normalized",
        "first_stage_initialization": initialization,
        "original_users_per_epoch": 100,
        "expanded_examples_per_epoch": 100,
        "input_tokens_per_epoch": 6_000,
    }
    metadata = {
        "training_semantics_revision": 2,
        "dataset_size": "500m",
        "seed": 42,
        "num_epochs": 20,
        "max_epochs": 20,
        "epochs_trained": 20,
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
        "deep_learning_rate": 0.0015,
        "best_epoch": 6,
        "targets_per_epoch": 100,
        "tokens_per_epoch": 6_000,
        "optimizer_steps_per_epoch": 5,
        **architecture,
        "transfer_invariants": architecture,
    }

    counts, loaded_sha = validate_treatment_metadata(
        metadata,
        candidate,
        expected_checkpoint_sha256="checkpoint",
        expected_checkpoint_source=next(
            source for source in source_candidates() if source.deep_lr == 0.048
        ),
    )

    assert counts["candidate_targets_per_epoch"] == 100
    assert loaded_sha == "checkpoint"
    frozen_candidate = make_boundary_candidate(
        replace(candidate, embedding_lr=0.032),
        "embedding",
        "low",
        8,
    )
    frozen_metadata = dict(metadata)
    frozen_metadata["embedding_learning_rate"] = 0.0
    frozen_counts, frozen_loaded_sha = validate_treatment_metadata(
        frozen_metadata,
        frozen_candidate,
        expected_checkpoint_sha256="checkpoint",
        expected_checkpoint_source=next(
            source for source in source_candidates() if source.deep_lr == 0.048
        ),
    )
    assert frozen_counts["candidate_targets_per_epoch"] == 100
    assert frozen_loaded_sha == "checkpoint"
    wrong_frozen_metadata = dict(frozen_metadata)
    wrong_frozen_metadata["embedding_learning_rate"] = 0.000125
    with pytest.raises(Rq15ReportError, match="embedding_learning_rate"):
        validate_treatment_metadata(
            wrong_frozen_metadata,
            frozen_candidate,
            expected_checkpoint_sha256="checkpoint",
            expected_checkpoint_source=next(
                source for source in source_candidates() if source.deep_lr == 0.048
            ),
        )
    invalid = dict(metadata)
    invalid["include_history_memory"] = True
    with pytest.raises(Rq15ReportError, match="include_history_memory"):
        validate_treatment_metadata(
            invalid,
            candidate,
            expected_checkpoint_sha256="checkpoint",
            expected_checkpoint_source=next(
                source for source in source_candidates() if source.deep_lr == 0.048
            ),
        )


def test_writer_never_updates_readme_from_incomplete_evidence(tmp_path: Path) -> None:
    scratch, checkpoint, runs = _resolved_runs()
    incomplete = build_report_bundle(scratch, checkpoint, runs[:-1])
    readme = tmp_path / "README.md"
    readme.write_text("# Existing\n")

    paths = write_report_bundle(
        incomplete,
        tmp_path / "scratchpad",
        tmp_path / "evidence",
        readme=readme,
    )

    assert set(paths) == {"tuning", "diagnostics", "evidence"}
    assert readme.read_text() == "# Existing\n"
    assert not (tmp_path / "scratchpad/rq15_training_reader_500m.md").exists()
