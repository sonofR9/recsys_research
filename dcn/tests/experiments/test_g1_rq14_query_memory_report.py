from __future__ import annotations

from dataclasses import replace

import pytest

from dcn.training_metadata import GENERATION_TRAINING_SEMANTICS_REVISION
from experiments.g1_sasrec_item_ids_likes.analysis import (
    rq14_query_memory_explanation as explanation_module,
)
from experiments.g1_sasrec_item_ids_likes.analysis.rq13_rq14_query_candidates import (
    QueryCandidate,
    make_boundary_candidate,
    rq14_initial_candidates,
)
from experiments.g1_sasrec_item_ids_likes.analysis.rq14_query_memory_report import (
    Rq14ReportError,
    Run,
    _validation_best_epoch,
    build_report_bundle,
    validate_training_metadata,
    write_report_bundle,
)
from experiments.g1_sasrec_item_ids_likes.analysis.rq14_query_memory_explanation import (
    build_unexpected_result_explanation,
)


def _run(
    candidate: QueryCandidate,
    recall: float,
    *,
    validation_ndcg: float = 0.05,
    full_recall: float | None = None,
    best_epoch: int = 7,
    wall_seconds: float = 200.0,
) -> Run:
    return Run(
        candidate=candidate,
        best_epoch=best_epoch,
        stopped_epoch=20,
        validation_recall=recall,
        validation_ndcg=validation_ndcg,
        validation_curve=tuple(
            (
                epoch,
                recall if epoch == best_epoch else recall - abs(epoch - best_epoch) / 100,
                validation_ndcg if epoch == best_epoch else validation_ndcg - 0.001,
            )
            for epoch in range(1, 21)
        ),
        metrics={
            "recall@100": recall if full_recall is None else full_recall,
            "ndcg@100": validation_ndcg,
            "recall@10": recall / 4,
            "ndcg@10": validation_ndcg / 2,
            "coverage@100": 0.5,
        },
        original_users_per_epoch=100,
        expanded_examples_per_epoch=100,
        candidate_targets_per_epoch=100,
        ntp_targets_per_epoch=0,
        input_tokens_per_epoch=6_000,
        optimizer_steps_per_epoch=1,
        steady_state_targets_per_second=1_000.0,
        time_through_selected_checkpoint_seconds=wall_seconds / 2,
        required_horizon_train_validation_seconds=wall_seconds,
        observed_end_to_end_wall_seconds=wall_seconds + 10,
        train_cache="train",
        validation_cache="validation",
        query_cache="query",
        evaluator_fingerprint="evaluator",
        scoring_fingerprint="scoring",
        artifact_sha256={
            "training_metadata.json": candidate.run_name + "-metadata",
            "final_metrics.json": candidate.run_name + "-metrics",
            "sweep.log": candidate.run_name + "-log",
        },
    )


def _resolved_runs(
    scores: dict[str, float] | None = None,
) -> list[Run]:
    scores = scores or {
        "shared_cls_only": 0.130,
        "distinct_cls_only": 0.135,
        "shared_history": 0.136,
        "distinct_history": 0.141,
    }
    runs = []
    for candidate in rq14_initial_candidates():
        recall = scores[candidate.treatment] - abs(candidate.deep_lr - 0.012)
        runs.append(_run(candidate, recall))
    return runs


def test_validation_epoch_ties_keep_the_earliest_recall_checkpoint() -> None:
    assert _validation_best_epoch(
        (
            (1, 0.10, 0.05),
            (2, 0.12, 0.04),
            (3, 0.12, 0.09),
        )
    ) == 2


def _audit(runs: list[Run]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "research_question": "RQ14 decoder-decoder query memory",
        "dataset_size": "500m",
        "status": "passed",
        "checks": {
            "artifact_and_recipe_integrity": {"passed": True},
            "query_slot_identity_and_order": {"passed": True},
            "memory_content_and_lengths": {"passed": True},
            "target_exclusion_and_candidate_only_loss": {"passed": True},
            "gradient_flow_to_every_slot_and_history": {"passed": True},
            "learning_curves_and_lr_boundaries": {"passed": True},
        },
        "run_artifacts": {
            run.candidate.run_name: run.artifact_sha256 for run in runs
        },
        "implementation_sha256": "current",
    }


def _rq13_explanation_inputs() -> tuple[dict[str, object], dict[str, object]]:
    import hashlib
    import json

    audit = {
        "schema_version": 1,
        "research_question": "RQ13 encoder-decoder prefix expansion",
        "dataset_size": "500m",
        "status": "passed",
        "checks": {"source": {"passed": True}},
    }
    audit_hash = hashlib.sha256(
        json.dumps(audit, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    records = {
        "one_example": ("one", 0.0774, 100),
        "selected_cap_32": ("cap32", 0.1255, 2_350),
    }
    results = {
        "research_question": "RQ13 encoder-decoder prefix expansion",
        "dataset_size": "500m",
        "claims_status": "ready_for_user_validation",
        "required_followups": [],
        "correctness_audit": {
            "status": "passed",
            "schema_version": 1,
            "artifact_sha256": audit_hash,
        },
        "selected": {
            treatment: {
                "run_name": run,
                "full_user_metrics": {"recall@100": recall},
            }
            for treatment, (run, recall, _) in records.items()
        },
        "treatments": {
            treatment: {
                "artifacts": [
                    {
                        "run_name": run,
                        "original_users_per_epoch": 100,
                        "expanded_examples_per_epoch": targets,
                        "candidate_targets_per_epoch": targets,
                        "ntp_targets_per_epoch": 0,
                        "artifact_sha256": {
                            "training_metadata.json": run + "-metadata",
                            "final_metrics.json": run + "-metrics",
                            "sweep.log": run + "-log",
                        },
                    }
                ]
            }
            for treatment, (run, _, targets) in records.items()
        },
    }
    return results, audit


def test_report_selects_each_lr_and_renders_separate_quality_and_efficiency() -> None:
    runs = _resolved_runs()
    bundle = build_report_bundle(
        runs,
        correctness_audit=_audit(runs),
        current_implementation_hash="current",
    )

    assert bundle.evidence["claims_status"] == "ready_for_user_validation"
    assert bundle.evidence["required_followups"] == []
    assert bundle.evidence["required_diagnostics"] == []
    assert bundle.evidence["selected_method"]["treatment"] == "distinct_history"
    assert bundle.evidence["rq15_distinct_memory"]["treatment"] == "distinct_history"
    assert all(
        record["deep_lr"] == 0.012
        for record in bundle.evidence["selected"].values()
    )
    assert "## Candidate-generation quality" in bundle.reader_markdown
    assert "## Training efficiency" in bundle.reader_markdown
    assert "processed examples" in bundle.reader_markdown
    assert "total tuning wall" in bundle.reader_markdown
    assert "## shared CLS, four CLS states" in bundle.tuning_markdown


def test_report_requires_geometric_boundary_before_selection() -> None:
    runs = _resolved_runs()
    treatment = "shared_cls_only"
    runs = [
        replace(
            run,
            validation_recall=(
                0.150 if run.candidate.treatment == treatment and run.candidate.deep_lr == 0.006
                else run.validation_recall
            ),
        )
        for run in runs
    ]

    pending = build_report_bundle(runs)
    expected = make_boundary_candidate(
        next(
            run.candidate
            for run in runs
            if run.candidate.treatment == treatment and run.candidate.deep_lr == 0.006
        ),
        "low",
        1,
    )

    assert pending.evidence["claims_status"] == "pending_boundary"
    assert pending.evidence["required_followups"] == [expected.run_name]

    boundary = _run(expected, 0.149)
    completed_runs = [*runs, boundary]
    completed = build_report_bundle(
        completed_runs,
        correctness_audit=_audit(completed_runs),
        current_implementation_hash="current",
    )
    assert completed.evidence["required_followups"] == []
    assert completed.evidence["selected"][treatment]["deep_lr"] == 0.006


@pytest.mark.parametrize(
    "boundaries",
    [
        (("high", 1),),
        (("low", 2),),
        (("low", 1), ("high", 1)),
    ],
)
def test_report_rejects_rogue_or_noncontiguous_boundaries(boundaries) -> None:
    runs = _resolved_runs()
    treatment = "shared_cls_only"
    runs = [
        replace(
            run,
            validation_recall=(
                0.150
                if run.candidate.treatment == treatment
                and run.candidate.deep_lr == 0.006
                else run.validation_recall
            ),
        )
        for run in runs
    ]
    anchor = next(
        run.candidate
        for run in runs
        if run.candidate.treatment == treatment and run.candidate.deep_lr == 0.006
    )
    for direction, step in boundaries:
        candidate = make_boundary_candidate(anchor, direction, step)
        runs.append(_run(candidate, 0.160))

    with pytest.raises(Rq14ReportError, match="boundary"):
        build_report_bundle(runs)


def test_report_rejects_boundary_after_surface_was_already_resolved() -> None:
    runs = _resolved_runs()
    treatment = "shared_cls_only"
    runs = [
        replace(
            run,
            validation_recall=(
                0.150
                if run.candidate.treatment == treatment
                and run.candidate.deep_lr == 0.006
                else run.validation_recall
            ),
        )
        for run in runs
    ]
    anchor = next(
        run.candidate
        for run in runs
        if run.candidate.treatment == treatment and run.candidate.deep_lr == 0.006
    )
    low1 = make_boundary_candidate(anchor, "low", 1)
    low2 = make_boundary_candidate(anchor, "low", 2)
    runs.extend([_run(low1, 0.140), _run(low2, 0.170)])

    with pytest.raises(Rq14ReportError, match="resolved"):
        build_report_bundle(runs)


def test_unexpected_effects_require_explanation_after_correctness_passes() -> None:
    runs = _resolved_runs(
        {
            "shared_cls_only": 0.140,
            "distinct_cls_only": 0.136,
            "shared_history": 0.137,
            "distinct_history": 0.135,
        }
    )
    bundle = build_report_bundle(
        runs,
        correctness_audit=_audit(runs),
        current_implementation_hash="current",
    )

    assert bundle.evidence["claims_status"] == "unexpected_result_requires_explanation"
    assert bundle.evidence["correctness_audit"]["status"] == "passed"
    assert bundle.evidence["unexpected_effects"]
    assert bundle.evidence["required_diagnostics"] == [
        "explain each unexpected token-identity or memory effect with experimental evidence"
    ]


def test_current_bound_explanation_opens_reader_gate_for_unexpected_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        explanation_module,
        "validate_rq13_correctness_audit",
        lambda audit, artifacts: {"status": "passed"},
    )
    runs = _resolved_runs(
        {
            "shared_cls_only": 0.0784,
            "distinct_cls_only": 0.0786,
            "shared_history": 0.0787,
            "distinct_history": 0.0791,
        }
    )
    audit = _audit(runs)
    preliminary = build_report_bundle(
        runs,
        correctness_audit=audit,
        current_implementation_hash="current",
    )
    rq13_results, rq13_audit = _rq13_explanation_inputs()
    explanation = build_unexpected_result_explanation(
        preliminary.evidence,
        audit,
        rq13_results,
        rq13_audit,
    )

    final = build_report_bundle(
        runs,
        correctness_audit=audit,
        unexpected_explanation=explanation,
        rq13_results=rq13_results,
        rq13_correctness_audit=rq13_audit,
        current_implementation_hash="current",
    )

    assert final.evidence["claims_status"] == "ready_for_user_validation"
    assert final.evidence["required_diagnostics"] == []
    assert final.evidence["unexpected_result_explanation"]["status"] == "passed"
    assert final.reader_markdown is not None
    assert "## Bound explanation" in final.diagnostics_markdown
    assert "explanation gates passed" in final.diagnostics_markdown
    assert "require a current correctness audit" not in final.diagnostics_markdown


def test_stale_explanation_keeps_reader_gate_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        explanation_module,
        "validate_rq13_correctness_audit",
        lambda audit, artifacts: {"status": "passed"},
    )
    runs = _resolved_runs(
        {
            "shared_cls_only": 0.0784,
            "distinct_cls_only": 0.0786,
            "shared_history": 0.0787,
            "distinct_history": 0.0791,
        }
    )
    audit = _audit(runs)
    preliminary = build_report_bundle(
        runs,
        correctness_audit=audit,
        current_implementation_hash="current",
    )
    rq13_results, rq13_audit = _rq13_explanation_inputs()
    explanation = build_unexpected_result_explanation(
        preliminary.evidence,
        audit,
        rq13_results,
        rq13_audit,
    )
    explanation["conclusion"] = "stale"

    final = build_report_bundle(
        runs,
        correctness_audit=audit,
        unexpected_explanation=explanation,
        rq13_results=rq13_results,
        rq13_correctness_audit=rq13_audit,
        current_implementation_hash="current",
    )

    assert final.evidence["claims_status"] == "unexpected_result_requires_explanation"
    assert final.evidence["unexpected_result_explanation"]["status"] == "stale_or_invalid"
    assert final.reader_markdown is None


def test_resolved_results_require_a_current_correctness_audit() -> None:
    runs = _resolved_runs()

    missing = build_report_bundle(runs)
    stale = build_report_bundle(
        runs,
        correctness_audit={**_audit(runs), "implementation_sha256": "stale"},
        current_implementation_hash="current",
    )

    assert missing.evidence["claims_status"] == "correctness_audit_required"
    assert stale.evidence["claims_status"] == "correctness_audit_required"
    assert missing.reader_markdown is None
    assert stale.reader_markdown is None


def test_pending_bundle_removes_stale_reader_table(tmp_path) -> None:
    scratchpad = tmp_path / "scratchpad"
    evidence = tmp_path / "evidence"
    scratchpad.mkdir()
    stale = scratchpad / "rq14_query_memory_reader_500m.md"
    stale.write_text("stale reader")

    write_report_bundle(build_report_bundle([]), scratchpad, evidence)

    assert not stale.exists()
    assert (scratchpad / "rq14_query_memory_tuning_500m.md").is_file()


def test_training_metadata_rejects_wrong_architecture_or_counts() -> None:
    candidate = rq14_initial_candidates()[0]
    metadata = {
        "training_semantics_revision": GENERATION_TRAINING_SEMANTICS_REVISION,
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
        "deep_learning_rate": candidate.deep_lr,
        "best_epoch": 7,
        "query_architecture": "decoder_decoder",
        "prefix_length_rule": "truncated",
        "prefix_cap": 1,
        "query_slots_shared": True,
        "include_history_memory": False,
        "num_query_slots": 4,
        "original_users_per_epoch": 100,
        "expanded_examples_per_epoch": 100,
        "candidate_targets_per_epoch": 100,
        "ntp_targets_per_epoch": 0,
        "input_tokens_per_epoch": 6_000,
        "targets_per_epoch": 100,
        "tokens_per_epoch": 6_000,
    }
    metadata["transfer_invariants"] = {
        key: value
        for key, value in metadata.items()
        if key
        in {
            "query_architecture",
            "prefix_length_rule",
            "prefix_cap",
            "query_slots_shared",
            "include_history_memory",
            "num_query_slots",
            "original_users_per_epoch",
            "expanded_examples_per_epoch",
            "candidate_targets_per_epoch",
            "ntp_targets_per_epoch",
            "input_tokens_per_epoch",
        }
    }

    counts, _ = validate_training_metadata(metadata, candidate)
    assert counts["candidate_targets_per_epoch"] == 100

    with pytest.raises(Rq14ReportError, match="query_slots_shared"):
        validate_training_metadata(
            {**metadata, "query_slots_shared": False}, candidate
        )
    with pytest.raises(Rq14ReportError, match="candidate target"):
        invalid = {**metadata, "candidate_targets_per_epoch": 99}
        invalid["transfer_invariants"] = {
            **metadata["transfer_invariants"],
            "candidate_targets_per_epoch": 99,
        }
        validate_training_metadata(invalid, candidate)
