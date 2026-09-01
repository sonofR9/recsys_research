from __future__ import annotations

import copy
import os
import hashlib
import json
from pathlib import Path
import runpy

import pytest

from dcn.config.query_retrieval_training import (
    MuTransferRq14PretrainedCrossAttentionGenerationExperiment,
)
from dcn.config.query_retrieval import MuTransferCrossAttentionGenerationExperiment
from experiments.g1_sasrec_item_ids_likes.analysis.rq14_pretrained_candidates import (
    DEEP_LRS,
    candidate_by_run,
    initial_candidates,
    launch_candidates,
    make_boundary_candidate,
    reused_candidates,
)
from experiments.g1_sasrec_item_ids_likes.analysis.rq14_pretrained_report import (
    LesionDiagnostics,
    PretrainedRun,
    Rq14PretrainedReportError,
    SourceCheckpointBinding,
    build_report_bundle,
    collect_report_bundle,
    current_implementation_sha256,
    current_lesion_implementation_sha256,
    rq15_reuse_is_recipe_compatible,
    update_readme,
    validate_source_initialization,
    validate_selected_checkpoint,
)
from experiments.g1_sasrec_item_ids_likes.analysis import rq14_pretrained_report
from experiments.g1_sasrec_item_ids_likes.analysis.rq14_pretrained_audit import (
    _production_recipe_probe,
)
from experiments.g1_sasrec_item_ids_likes.analysis.rq14_query_memory_audit import (
    run_query_memory_model_probe,
)
from experiments.g1_sasrec_item_ids_likes.analysis.rq15_training_audit import (
    run_model_correctness_probe,
)
from experiments.g1_sasrec_item_ids_likes.analysis.rq15_training_candidates import (
    source_candidates,
)
from experiments.g1_sasrec_item_ids_likes.launchers import verify_artifact


EXPERIMENT = Path("experiments/g1_sasrec_item_ids_likes")
CONFIG = EXPERIMENT / "configs/rq14_pretrained_query_variant.py"
LAUNCHER = EXPERIMENT / "launchers/architecture/rq14_pretrained_query_500m.sh"


def _experiment(candidate):
    keys = (
        "G1_RQ14_PRETRAINED_RUN",
        "G1_RQ15_SOURCE_RUN",
        "G1_RQ15_FIRST_STAGE_CHECKPOINT",
    )
    previous = {key: os.environ.get(key) for key in keys}
    try:
        os.environ["G1_RQ14_PRETRAINED_RUN"] = candidate.run_name
        os.environ["G1_RQ15_SOURCE_RUN"] = source_candidates()[1].run_name
        os.environ["G1_RQ15_FIRST_STAGE_CHECKPOINT"] = "/tmp/exact-rq15-source.pt"
        return runpy.run_path(str(CONFIG))["experiment"]
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_manifest_has_twelve_cells_three_reuses_and_nine_new_runs() -> None:
    manifest = initial_candidates()

    assert len(manifest) == 12
    assert len(reused_candidates()) == 3
    assert len(launch_candidates()) == 9
    assert {item.deep_lr for item in manifest} == set(DEEP_LRS)
    assert {item.embedding_lr for item in manifest} == {0.00025}
    assert {item.treatment for item in reused_candidates()} == {"distinct_cls_only"}
    assert all(item.artifact_run_name.startswith("g1_rq15_") for item in reused_candidates())
    assert set(launch_candidates()).isdisjoint(reused_candidates())


@pytest.mark.parametrize("candidate", launch_candidates())
def test_new_recipe_uses_exact_pretraining_and_treatment_semantics(candidate) -> None:
    experiment = _experiment(candidate)

    assert isinstance(
        experiment, MuTransferRq14PretrainedCrossAttentionGenerationExperiment
    )
    assert experiment.size == "500m"
    assert experiment.dataloader.effective_batch_size == 1280
    assert experiment.query_architecture == "decoder_decoder"
    assert experiment.num_query_slots == 4
    assert experiment.query_slots_shared is candidate.query_slots_shared
    assert experiment.include_history_memory is candidate.include_history_memory
    assert experiment.training_method == "pretrained_finetune"
    assert experiment.first_stage_checkpoint == Path("/tmp/exact-rq15-source.pt")
    assert experiment.auxiliary_ntp_weight == 0
    assert experiment.embedding_learning_rate == 0.00025
    assert experiment.deep_learning_rate == candidate.deep_lr
    assert experiment.num_epochs == 20
    assert experiment.lr_schedule_horizon_epochs == 20
    assert experiment.lr_schedule.shape == "linear"
    assert experiment.restore_best_weights is True
    assert experiment.early_stopping_patience is None


def test_finetuning_uses_only_the_candidate_criterion(monkeypatch) -> None:
    candidate = launch_candidates()[0]
    experiment = _experiment(candidate)
    candidate_criterion = object()
    monkeypatch.setattr(
        MuTransferCrossAttentionGenerationExperiment,
        "create_criterion",
        lambda self: candidate_criterion,
    )

    assert experiment.create_criterion() is candidate_criterion
    assert experiment.auxiliary_ntp_targets_per_epoch == 0


def test_reused_cell_reconstructs_the_exact_rq14_compatibility_recipe() -> None:
    candidate = reused_candidates()[0]
    experiment = _experiment(candidate)

    assert experiment.run_name == candidate.run_name
    assert experiment.query_slots_shared is False
    assert experiment.include_history_memory is False
    assert experiment.embedding_learning_rate == 0.00025
    assert experiment.deep_learning_rate == candidate.deep_lr


def test_boundary_followup_is_geometric_and_canonical() -> None:
    low_anchor = next(item for item in launch_candidates() if item.deep_lr == min(DEEP_LRS))
    high_anchor = next(item for item in launch_candidates() if item.deep_lr == max(DEEP_LRS))
    low = make_boundary_candidate(low_anchor, "low", 1)
    high = make_boundary_candidate(high_anchor, "high", 2)

    assert low.deep_lr == min(DEEP_LRS) / 2
    assert high.deep_lr == max(DEEP_LRS) * 4
    assert candidate_by_run(low.run_name) == low
    assert candidate_by_run(high.run_name) == high


def _run(candidate, recall: float, *, ndcg: float = 0.05) -> PretrainedRun:
    return PretrainedRun(
        candidate=candidate,
        artifact_run_name=candidate.artifact_run_name,
        reused_from_rq15=candidate.reused_rq15_run_name is not None,
        validation_recall=recall,
        validation_ndcg=ndcg,
        best_epoch=10,
        stopped_epoch=20,
        metrics={
            "recall@100": recall,
            "ndcg@100": ndcg,
            "recall@10": recall / 4,
            "ndcg@10": ndcg / 2,
            "coverage@100": 0.4,
        },
        expanded_examples_per_epoch=100,
        candidate_targets_per_epoch=100,
        ntp_targets_per_epoch=0,
        input_tokens_per_epoch=6_400,
        steady_state_targets_per_second=1_000.0,
        time_through_best_seconds=10.0,
        horizon_seconds=20.0,
        artifact_sha256={"training_metadata.json": candidate.run_name},
        checkpoint_sha256="a" * 64,
    )


def _source_binding() -> SourceCheckpointBinding:
    return SourceCheckpointBinding(
        run_name="selected_source",
        checkpoint_path="/tmp/selected_source/checkpoint.pt",
        checkpoint_sha256="a" * 64,
        verified=True,
    )


def _audit(runs: list[PretrainedRun]) -> dict[str, object]:
    checks = {
        name: {"passed": True}
        for name in (
            "treatment_recipes",
            "exact_checkpoint_load_scope",
            "candidate_only_loss_and_target_exclusion",
            "memory_ordering_and_gradients",
            "horizon_and_artifact_binding",
        )
    }
    return {
        "schema_version": 1,
        "research_question": "RQ14 pretrained decoder-decoder query memory",
        "dataset_size": "500m",
        "status": "passed",
        "checks": checks,
        "run_artifacts": {
            run.artifact_run_name: run.artifact_sha256 for run in runs
        },
        "checkpoint_sha256": "a" * 64,
        "implementation_sha256": current_implementation_sha256(),
    }


def _diagnostics(runs: list[PretrainedRun]) -> LesionDiagnostics:
    selected = {
        treatment: max(
            (run for run in runs if run.candidate.treatment == treatment),
            key=lambda run: run.validation_recall,
        )
        for treatment in (
            "shared_cls_only",
            "distinct_cls_only",
            "shared_history",
            "distinct_history",
        )
    }
    lesions = {}
    findings = {}
    for treatment, run in selected.items():
        names = [f"drop_cls_{index}" for index in range(4)]
        if treatment.endswith("history"):
            names.append("remove_history")
        effect = {
            "state_use": "states_used",
            "recommendation_effect": "within_noise_or_redundant",
            "lesion_minus_normal_recall@100": -0.001,
            "lesion_minus_normal_ndcg@100": -0.0005,
        }
        lesion_rows = {
            name: {
                "metrics": dict(run.metrics),
                "query_change": {
                    "num_users": 100,
                    "changed_user_fraction": 1.0,
                    "mean_l2_change": 0.1,
                    "max_l2_change": 0.2,
                    "mean_relative_l2_change": 0.03,
                    "mean_cosine_distance": 0.001,
                },
                "effect": effect,
            }
            for name in names
        }
        lesions[treatment] = {
            "run_name": f"{treatment}_lesions",
            "treatment": treatment,
            "source_selected_run_name": run.artifact_run_name,
            "selected_rerun_compatibility": {
                "diagnostic_minus_source_recall@100": 0.0,
                "diagnostic_minus_source_ndcg@100": 0.0,
            },
            "normal_metrics": dict(run.metrics),
            "lesions": lesion_rows,
            "artifact_sha256": {"rq14_lesion_diagnostics.json": "b" * 64},
        }
        findings[treatment] = {name: effect for name in names}
    evidence = {
        "schema_version": 1,
        "research_question": "RQ14 pretrained decoder-decoder query-memory lesions",
        "dataset_size": "500m",
        "status": "passed",
        "claims_status": "diagnostics_complete_claims_not_published",
        "source_rq14_results_sha256": "e" * 64,
        "source_checkpoint_sha256": "a" * 64,
        "source_unexpected_effects": {
            "distinct_vs_shared_cls_only": (
                selected["distinct_cls_only"].metrics["recall@100"]
                - selected["shared_cls_only"].metrics["recall@100"]
            ),
            "distinct_vs_shared_history": (
                selected["distinct_history"].metrics["recall@100"]
                - selected["shared_history"].metrics["recall@100"]
            ),
            "history_vs_cls_only_shared": (
                selected["shared_history"].metrics["recall@100"]
                - selected["shared_cls_only"].metrics["recall@100"]
            ),
            "history_vs_cls_only_distinct": (
                selected["distinct_history"].metrics["recall@100"]
                - selected["distinct_cls_only"].metrics["recall@100"]
            ),
        },
        "implementation_sha256": current_lesion_implementation_sha256(),
        "run_artifacts": {
            record["run_name"]: record["artifact_sha256"]
            for record in lesions.values()
        },
        "runs": lesions,
    }
    canonical = hashlib.sha256(
        json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    explanation = {
        "schema_version": 1,
        "research_question": "RQ14 pretrained decoder-decoder query-memory lesions",
        "status": "passed",
        "claims_status": "diagnostics_complete_claims_not_published",
        "evidence_sha256": canonical,
        "findings": findings,
        "summary": {
            "states_used": 18,
            "states_ignored": 0,
            "within_noise_or_redundant": 18,
            "resolved_degradation_after_removal": 0,
            "resolved_change_after_removal": 0,
        },
    }
    return LesionDiagnostics(
        evidence=evidence,
        explanation=explanation,
        evidence_file_sha256="c" * 64,
        explanation_file_sha256="d" * 64,
        raw_artifacts_verified=True,
    )


def test_report_preserves_candidate_only_and_waits_for_all_pretrained_cells() -> None:
    bundle = build_report_bundle(
        [_run(initial_candidates()[0], 0.15)],
        candidate_only_markdown="OLD CANDIDATE-ONLY TABLE",
    )

    assert bundle.evidence["claims_status"] == "pending_artifacts"
    assert len(bundle.evidence["required_followups"]) == 11
    assert "OLD CANDIDATE-ONLY TABLE" in bundle.reader_markdown
    assert "NTP-pretrained quality" not in bundle.reader_markdown


def test_report_resolves_interior_surface_and_renders_quality_and_efficiency() -> None:
    runs = []
    treatment_gain = {
        "shared_cls_only": 0.0,
        "distinct_cls_only": 0.004,
        "shared_history": 0.004,
        "distinct_history": 0.008,
    }
    for candidate in initial_candidates():
        score = (
            0.15
            + treatment_gain[candidate.treatment]
            - abs(candidate.deep_lr - 0.00075)
        )
        runs.append(_run(candidate, score))
    bundle = build_report_bundle(
        runs,
        candidate_only_markdown="OLD CANDIDATE-ONLY TABLE",
        correctness_audit={
            **_audit(runs),
        },
        source_checkpoint=_source_binding(),
    )

    assert bundle.evidence["claims_status"] == "ready_for_user_validation"
    assert bundle.evidence["required_followups"] == []
    assert all(
        record["deep_lr"] == 0.00075
        for record in bundle.evidence["selected"].values()
    )
    assert "## NTP-pretrained quality" in bundle.reader_markdown
    assert "## NTP-pretrained training efficiency" in bundle.reader_markdown


def test_report_requests_exact_next_boundary_and_rejects_wrong_checkpoint() -> None:
    runs = []
    for candidate in initial_candidates():
        score = 0.16 if candidate.deep_lr == min(DEEP_LRS) else 0.14
        runs.append(_run(candidate, score))
    bundle = build_report_bundle(runs, candidate_only_markdown="old")

    assert len(bundle.evidence["required_boundary_followups"]) == 4
    assert all("lrdeeplow1" in name for name in bundle.evidence["required_boundary_followups"])

    runs[-1] = PretrainedRun(**{**runs[-1].__dict__, "checkpoint_sha256": "b" * 64})
    with pytest.raises(Rq14PretrainedReportError, match="checkpoint"):
        build_report_bundle(runs, candidate_only_markdown="old")


def test_improvements_within_the_shared_band_still_require_investigation() -> None:
    gains = {
        "shared_cls_only": 0.0,
        "distinct_cls_only": 0.002,
        "shared_history": 0.002,
        "distinct_history": 0.004,
    }
    runs = [
        _run(
            candidate,
            0.15 + gains[candidate.treatment] - abs(candidate.deep_lr - 0.00075),
        )
        for candidate in initial_candidates()
    ]
    audit = _audit(runs)

    bundle = build_report_bundle(
        runs,
        candidate_only_markdown="old",
        correctness_audit=audit,
        source_checkpoint=_source_binding(),
    )

    assert bundle.evidence["claims_status"] == "unexpected_result_requires_investigation"


def test_report_rejects_skeletal_or_mutated_correctness_audits() -> None:
    runs = [
        _run(candidate, 0.15 - abs(candidate.deep_lr - 0.00075))
        for candidate in initial_candidates()
    ]
    audit = _audit(runs)
    mutations = []
    for key, value in (
        ("schema_version", 2),
        ("research_question", "wrong"),
        ("dataset_size", "50m"),
    ):
        mutated = copy.deepcopy(audit)
        mutated[key] = value
        mutations.append(mutated)
    missing_check = copy.deepcopy(audit)
    missing_check["checks"].pop("memory_ordering_and_gradients")
    mutations.append(missing_check)
    extra_check = copy.deepcopy(audit)
    extra_check["checks"]["invented"] = {"passed": True}
    mutations.append(extra_check)
    failed_check = copy.deepcopy(audit)
    failed_check["checks"]["treatment_recipes"]["passed"] = False
    mutations.append(failed_check)
    mutations.append(
        {
            "status": "passed",
            "run_artifacts": audit["run_artifacts"],
            "checkpoint_sha256": "a" * 64,
            "implementation_sha256": current_implementation_sha256(),
        }
    )

    for mutated in mutations:
        bundle = build_report_bundle(
            runs,
            candidate_only_markdown="old",
            correctness_audit=mutated,
            source_checkpoint=_source_binding(),
        )
        assert bundle.evidence["claims_status"] == "correctness_audit_required"
        assert bundle.evidence["correctness_audit"]["status"] == "stale_or_invalid"


def test_selected_source_initialization_rejects_run_path_or_digest_mutation() -> None:
    source = _source_binding()
    initialization = {
        "checkpoint_path": source.checkpoint_path,
        "checkpoint_sha256": source.checkpoint_sha256,
    }

    validate_source_initialization(initialization, source, context="run")

    for key, value in (
        ("checkpoint_path", "/tmp/other_source/checkpoint.pt"),
        ("checkpoint_sha256", "b" * 64),
    ):
        mutated = dict(initialization)
        mutated[key] = value
        with pytest.raises(Rq14PretrainedReportError, match="selected source"):
            validate_source_initialization(mutated, source, context="run")
    mutated_source = SourceCheckpointBinding(
        run_name="other_source",
        checkpoint_path=source.checkpoint_path,
        checkpoint_sha256=source.checkpoint_sha256,
        verified=True,
    )
    with pytest.raises(Rq14PretrainedReportError, match="selected source"):
        validate_source_initialization(initialization, mutated_source, context="run")


def test_report_publication_requires_a_selected_source_binding() -> None:
    gains = {
        "shared_cls_only": 0.0,
        "distinct_cls_only": 0.004,
        "shared_history": 0.004,
        "distinct_history": 0.008,
    }
    runs = [
        _run(
            candidate,
            0.15 + gains[candidate.treatment] - abs(candidate.deep_lr - 0.00075),
        )
        for candidate in initial_candidates()
    ]

    bundle = build_report_bundle(
        runs,
        candidate_only_markdown="old",
        correctness_audit=_audit(runs),
    )

    assert bundle.evidence["claims_status"] == "source_checkpoint_required"
    assert bundle.evidence["source_checkpoint"]["status"] == "missing"
    assert "NTP-pretrained quality (current decision)" not in bundle.reader_markdown


def test_bound_lesions_publish_full_metrics_efficiency_and_simplicity_selection() -> None:
    gains = {
        "shared_cls_only": 0.0,
        "distinct_cls_only": 0.002,
        "shared_history": 0.002,
        "distinct_history": 0.004,
    }
    runs = [
        _run(
            candidate,
            0.15 + gains[candidate.treatment] - abs(candidate.deep_lr - 0.00075),
        )
        for candidate in initial_candidates()
    ]
    audit = _audit(runs)

    bundle = build_report_bundle(
        runs,
        candidate_only_markdown="OLD CANDIDATE-ONLY TABLE",
        correctness_audit=audit,
        lesion_diagnostics=_diagnostics(runs),
        source_checkpoint=_source_binding(),
    )

    assert bundle.evidence["claims_status"] == "ready_for_user_validation"
    assert bundle.evidence["selected_method"]["treatment"] == "shared_cls_only"
    assert bundle.evidence["lesion_diagnostics"]["status"] == "passed"
    assert "recall@10" in bundle.reader_markdown
    assert "coverage@100" in bundle.reader_markdown
    assert "examples/epoch" in bundle.reader_markdown
    assert "input tokens/epoch" in bundle.reader_markdown
    assert "3-cell tuning GPU time" in bundle.reader_markdown
    assert "### Acceptance criteria" in bundle.readme_markdown
    assert "cross-attends both CLS tokens and history" in bundle.readme_markdown
    assert "\n## Historical candidate-only comparison" not in bundle.readme_markdown
    assert "\n### Historical candidate-only comparison" in bundle.readme_markdown
    assert "16 individual CLS-state removals" in bundle.readme_markdown
    assert "both history-memory removals" in bundle.readme_markdown
    assert "unresolved or redundant" in bundle.readme_markdown
    assert "| **shared CLS** | **four CLS states** | **0.150**" in bundle.reader_markdown
    assert "+1% (0.152)" in bundle.reader_markdown
    assert "<span style=\"color: green\">" in bundle.reader_markdown
    assert "**0.00075**" in bundle.tuning_markdown


def test_diagnostics_fail_closed_when_a_selected_normal_metric_changes() -> None:
    runs = [
        _run(candidate, 0.15 - abs(candidate.deep_lr - 0.00075))
        for candidate in initial_candidates()
    ]
    diagnostics = _diagnostics(runs)
    diagnostics.evidence["runs"]["shared_cls_only"]["normal_metrics"][
        "recall@10"
    ] += 0.01
    audit = _audit(runs)

    bundle = build_report_bundle(
        runs,
        candidate_only_markdown="old",
        correctness_audit=audit,
        lesion_diagnostics=diagnostics,
        source_checkpoint=_source_binding(),
    )

    assert bundle.evidence["claims_status"] != "ready_for_user_validation"
    assert bundle.evidence["lesion_diagnostics"]["status"] == "stale_or_invalid"
    assert "NTP-pretrained quality (current decision)" not in bundle.reader_markdown


def test_diagnostics_fail_closed_when_the_investigation_basis_changes() -> None:
    gains = {
        "shared_cls_only": 0.0,
        "distinct_cls_only": 0.002,
        "shared_history": 0.002,
        "distinct_history": 0.004,
    }
    runs = [
        _run(
            candidate,
            0.15 + gains[candidate.treatment] - abs(candidate.deep_lr - 0.00075),
        )
        for candidate in initial_candidates()
    ]
    diagnostics = _diagnostics(runs)
    diagnostics.evidence["source_unexpected_effects"][
        "history_vs_cls_only_shared"
    ] += 0.0001
    audit = _audit(runs)

    bundle = build_report_bundle(
        runs,
        candidate_only_markdown="old",
        correctness_audit=audit,
        lesion_diagnostics=diagnostics,
        source_checkpoint=_source_binding(),
    )

    assert bundle.evidence["claims_status"] == "unexpected_result_requires_investigation"
    assert bundle.evidence["lesion_diagnostics"]["status"] == "stale_or_invalid"


def test_readme_update_is_idempotent_and_stops_before_rq15(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(
        "before\n\n## RQ14 — old\n\nold body\n\n"
        "<!-- rq15-training-generated:start -->\n## RQ15 — keep\n"
    )
    block = "## RQ14 — new\n\nnew body\n"

    update_readme(readme, block)
    update_readme(readme, block)

    text = readme.read_text()
    assert text.count("<!-- rq14-pretrained-generated:start -->") == 1
    assert text.count("## RQ14 — new") == 1
    assert "## RQ15 — keep" in text


def test_reuse_is_rejected_when_rq15_artifact_binding_changes(tmp_path: Path) -> None:
    original = json.loads((EXPERIMENT / "evidence/rq15_training_results.json").read_text())
    records = original["treatments"]["pretrained_finetune"]["artifacts"]
    reused_name = reused_candidates()[0].artifact_run_name
    record = next(item for item in records if item["run_name"] == reused_name)
    record["artifact_sha256"]["sweep.log"] = "0" * 64
    results = tmp_path / "rq15.json"
    candidate_only = tmp_path / "old.md"
    results.write_text(json.dumps(original))
    candidate_only.write_text("old candidate-only table")

    with pytest.raises(Rq14PretrainedReportError, match="reuse binding"):
        collect_report_bundle(
            Path("generated/logs"),
            rq15_results_path=results,
            candidate_only_path=candidate_only,
        )


def test_collection_rejects_a_nonselected_source_candidate(
    monkeypatch, tmp_path: Path
) -> None:
    candidate_only = tmp_path / "old.md"
    candidate_only.write_text("old candidate-only table")
    monkeypatch.setattr(
        rq14_pretrained_report,
        "selected_source_candidate",
        lambda logs: source_candidates()[0],
    )

    with pytest.raises(Rq14PretrainedReportError, match="selected source"):
        collect_report_bundle(
            Path("generated/logs"),
            rq15_results_path=EXPERIMENT / "evidence/rq15_training_results.json",
            candidate_only_path=candidate_only,
        )


def test_rq15_reuse_requires_the_full_normalized_rq14_recipe(tmp_path: Path) -> None:
    candidate = reused_candidates()[0]
    source = source_candidates()[1]
    assignments = [
        f"G1_RQ14_PRETRAINED_RUN={candidate.run_name}",
        f"G1_RQ15_SOURCE_RUN={source.run_name}",
        f"G1_RQ15_FIRST_STAGE_CHECKPOINT={(Path('generated/logs') / source.run_name / source.checkpoint_name).resolve()}",
    ]
    original = Path("generated/logs") / candidate.artifact_run_name
    metadata = json.loads((original / "training_metadata.json").read_text())
    directory = tmp_path / candidate.artifact_run_name
    directory.mkdir()
    (directory / "training_metadata.json").write_text(json.dumps(metadata))

    assert rq15_reuse_is_recipe_compatible(directory, assignments) is True

    metadata["transfer_invariants"]["evaluation_catalog"] = "changed"
    (directory / "training_metadata.json").write_text(json.dumps(metadata))

    assert rq15_reuse_is_recipe_compatible(directory, assignments) is False


def test_selected_checkpoint_must_match_all_three_reused_cells(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"selected")
    digest = hashlib.sha256(b"selected").hexdigest()

    assert validate_selected_checkpoint({"checkpoint_sha256": digest}, checkpoint) == digest
    with pytest.raises(Rq14PretrainedReportError, match="differs"):
        validate_selected_checkpoint({"checkpoint_sha256": "0" * 64}, checkpoint)


def test_preflight_proves_memory_order_gradients_and_new_module_initialization() -> None:
    memory = run_query_memory_model_probe()
    checkpoint = run_model_correctness_probe()["checkpoint_copy_identity"]
    recipes = _production_recipe_probe()

    assert memory["slot_order_preserved"] is True
    assert memory["history_precedes_slots"] is True
    assert all(value > 0 for value in memory["distinct_slot_gradient_l1"])
    assert all(value > 0 for value in memory["shared_slot_gradient_l1"])
    assert memory["history_embedding_gradient_l1"] > 0
    assert memory["decoder_gradient_l1"] > 0
    assert checkpoint["copied_item_embedding"] is True
    assert checkpoint["copied_memory_encoder"] is True
    assert checkpoint["copied_tokenizer"] is True
    assert checkpoint["preserved_query_slots"] is True
    assert checkpoint["preserved_decoder"] is True
    assert recipes["passed"] is True
    assert recipes["all_candidate_only"] is True


def test_launcher_lists_only_nine_new_cells_and_supports_followups() -> None:
    text = LAUNCHER.read_text()

    assert "launch_candidates" in text
    assert "exactly nine new cells" in text
    assert "--followups" in text
    assert "required_boundary_followups" in text
    assert "utils/training_queue/queue.sh" in text
    assert "g1_require_config_compatible_or_absent" in text
    assert 'enqueue "$run"' in text
    assert "validate_selected_checkpoint" in text


def test_artifact_verifier_accepts_the_new_recipe_identity() -> None:
    candidate = launch_candidates()[0]
    assignments = verify_artifact._config_assignments(
        [
            f"G1_RQ14_PRETRAINED_RUN={candidate.run_name}",
            f"G1_RQ15_SOURCE_RUN={source_candidates()[1].run_name}",
            "G1_RQ15_FIRST_STAGE_CHECKPOINT=/tmp/selected.pt",
        ]
    )

    assert assignments["G1_RQ14_PRETRAINED_RUN"] == candidate.run_name
