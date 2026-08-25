import json
from pathlib import Path

import pytest

from experiments.g2_esasrec.analysis.evidence import (
    METRICS,
    VerifiedArtifact,
    build_composition_evidence,
    require_selected_control_lineage,
    require_selected_ligr_lineage,
    select_aggregate_bundle,
    write_composition_evidence,
)
from experiments.g2_esasrec.analysis.generate import (
    require_explicit_reversal_validation,
)
from experiments.g2_esasrec.analysis.report import (
    aggregate_section_heading,
    render_compact_report,
)
from experiments.g2_esasrec.protocol.manifest import (
    ApprovedJob,
    CompiledJob,
    approved_manifest,
)
from experiments.g2_esasrec.protocol.optuna_driver import (
    G2OptunaDriver,
    require_triggered_lr_boundaries,
)


def _artifact(
    compiled: CompiledJob,
    path: Path,
    *,
    recall: float = 0.2,
    ndcg: float = 0.1,
    coverage: float = 0.3,
    wall_seconds: float = 2.0,
) -> VerifiedArtifact:
    return VerifiedArtifact(
        job=compiled.approved,
        path=path / compiled.approved.run_name,
        parameters=compiled.parameters,
        metrics={
            "recall@100": recall,
            "ndcg@100": ndcg,
            "coverage@100": coverage,
        },
        metadata={"selection_resolved": True},
        costs={"training_seconds": 1.0, "wall_seconds": wall_seconds},
    )


def _control_artifact(path: Path, *, batch_size: int = 512) -> VerifiedArtifact:
    job = approved_manifest().jobs_for_stage("control_tuning")[1]
    return _artifact(
        CompiledJob(
            job,
            {
                "batch_size": batch_size,
                "embedding_learning_rate": 0.01,
                "deep_learning_rate": 0.02,
            },
        ),
        path,
    )


def _component_artifact(
    path: Path,
    method: str = "ligr_sampled_softmax",
    *,
    embedding_learning_rate: float = 0.01,
    deep_learning_rate: float = 0.02,
    ligr_multiplier: int = 4,
) -> VerifiedArtifact:
    job = next(
        job
        for job in approved_manifest().jobs_for_stage("component_tuning")
        if job.method == method and job.trial == 1
    )
    parameters = {
        "batch_size": 512,
        "embedding_learning_rate": embedding_learning_rate,
        "deep_learning_rate": deep_learning_rate,
        "selected_control_job_id": "control_tuning:control_trial_01",
    }
    if method.startswith(("ligr_", "matched_standard_")):
        parameters["ligr_multiplier"] = ligr_multiplier
    if method.endswith("_gbce"):
        parameters["gbce_t"] = 0.6
    return _artifact(CompiledJob(job, parameters), path)


def test_study_emits_one_slot_at_a_time_and_resumes_idempotently(
    tmp_path: Path,
) -> None:
    database = tmp_path / "study.sqlite3"
    driver = G2OptunaDriver(database, seed=17)

    first = driver.next_control()
    assert first is not None
    assert first.approved.id == "control_tuning:control_trial_00"
    assert driver.next_control() == first

    artifact = _artifact(first, tmp_path, recall=0.31)
    record = driver.record_observation(first, artifact)
    assert record.trial_number == 0
    assert record.parameters == first.parameters
    assert record.objective == 0.31
    assert record.artifact == artifact.path
    assert driver.record_observation(first, artifact) == record

    second = driver.next_control()
    assert second is not None
    resumed = G2OptunaDriver(database, seed=17)
    assert resumed.next_control() == second
    copy_driver = G2OptunaDriver(tmp_path / "copy.sqlite3", seed=17)
    copy_first = copy_driver.next_control()
    assert copy_first == first
    assert copy_first is not None
    copy_driver.record_observation(
        copy_first, _artifact(copy_first, tmp_path, recall=0.31)
    )
    assert copy_driver.next_control() == second


def test_component_and_mixed_studies_force_the_approved_anchors(
    tmp_path: Path,
) -> None:
    driver = G2OptunaDriver(tmp_path / "study.sqlite3")
    control = _control_artifact(tmp_path)
    ligr = _component_artifact(tmp_path, ligr_multiplier=6)

    component = driver.next_component("ligr_gbce", control, ligr_selection=ligr)
    assert component is not None
    assert component.parameters == {
        "batch_size": 512,
        "selected_control_job_id": control.job.id,
        "source_job_id": ligr.job.id,
        "embedding_learning_rate": 0.001,
        "deep_learning_rate": 0.001,
        "ligr_multiplier": 6,
        "gbce_t": 0.75,
    }
    ligr_anchor = driver.next_component("ligr_sampled_softmax", control)
    assert ligr_anchor is not None
    assert ligr_anchor.parameters["ligr_multiplier"] == 4

    mixed_none = driver.next_mixed(ligr)
    assert mixed_none is not None
    assert mixed_none.parameters["uniform_fraction"] == 0.6
    assert mixed_none.parameters["logq_correction"] == "none"
    assert mixed_none.parameters["source_job_id"] == ligr.job.id
    assert mixed_none.parameters["selected_control_job_id"] == (
        "control_tuning:control_trial_01"
    )
    driver.record_observation(mixed_none, _artifact(mixed_none, tmp_path))
    mixed_yi = driver.next_mixed(ligr)
    assert mixed_yi is not None
    assert mixed_yi.parameters["uniform_fraction"] == 0.6
    assert mixed_yi.parameters["logq_correction"] == "yi2019"


def test_study_rejects_changed_prerequisites_and_unapproved_methods(
    tmp_path: Path,
) -> None:
    driver = G2OptunaDriver(tmp_path / "study.sqlite3")
    driver.next_component("standard_sampled_softmax", _control_artifact(tmp_path))

    with pytest.raises(ValueError, match="fixed parameters changed"):
        driver.next_component(
            "standard_sampled_softmax",
            _control_artifact(tmp_path, batch_size=256),
        )
    with pytest.raises(ValueError, match="approved component method"):
        driver.next_component(  # type: ignore[arg-type]
            "unknown", _control_artifact(tmp_path)
        )


def test_completed_study_never_emits_beyond_approved_slots(tmp_path: Path) -> None:
    driver = G2OptunaDriver(tmp_path / "study.sqlite3")
    ligr = _component_artifact(tmp_path)
    approved_ids = {
        job.id for job in approved_manifest().jobs_for_stage("mixed_tuning")
    }
    emitted = []

    for trial in range(12):
        compiled = driver.next_mixed(ligr)
        assert compiled is not None
        emitted.append(compiled.approved.id)
        driver.record_observation(
            compiled,
            _artifact(compiled, tmp_path, recall=trial / 100),
        )

    assert set(emitted) == approved_ids
    assert driver.next_mixed(ligr) is None


def test_boundary_compiler_uses_only_approved_slots_and_verified_winner(
    tmp_path: Path,
) -> None:
    driver = G2OptunaDriver(tmp_path / "study.sqlite3")
    winner = _component_artifact(
        tmp_path,
        deep_learning_rate=0.128,
    )

    jobs = driver.compile_lr_boundary(winner)

    assert [job.approved.id for job in jobs] == [
        "lr_boundary:boundary_ligr_sampled_softmax_0",
        "lr_boundary:boundary_ligr_sampled_softmax_1",
    ]
    assert jobs[0].parameters["deep_learning_rate"] == pytest.approx(0.384)
    assert jobs[1].parameters["deep_learning_rate"] == pytest.approx(0.128 * 3**0.5)
    assert all(job.parameters["builder"] == "component" for job in jobs)
    assert all(job.parameters["method"] == winner.job.method for job in jobs)
    assert all(job.parameters["source_job_id"] == winner.job.id for job in jobs)

    middle = _component_artifact(tmp_path)
    assert driver.compile_lr_boundary(middle) == ()

    both_edges = _component_artifact(
        tmp_path,
        embedding_learning_rate=0.0001,
        deep_learning_rate=0.128,
    )
    with pytest.raises(ValueError, match="both learning rates"):
        driver.compile_lr_boundary(both_edges)


def test_confirmation_compiler_sources_existing_verified_artifacts_directly(
    tmp_path: Path,
) -> None:
    driver = G2OptunaDriver(tmp_path / "study.sqlite3")
    first = _component_artifact(tmp_path, "ligr_sampled_softmax")
    second = _component_artifact(tmp_path, "standard_gbce")

    confirmations = driver.compile_reversal_confirmation((first, second))
    assert [(job.approved.trial, job.approved.seed) for job in confirmations] == [
        (0, 43),
        (0, 44),
        (1, 43),
        (1, 44),
    ]
    assert confirmations[0].parameters["method"] == first.job.method
    assert confirmations[2].parameters["method"] == second.job.method
    assert confirmations[0].parameters["source_job_id"] == first.job.id
    assert confirmations[2].parameters["source_job_id"] == second.job.id

    with pytest.raises(ValueError, match="exactly two"):
        driver.compile_reversal_confirmation((first,))
    closing_source = VerifiedArtifact(
        ApprovedJob(
            "removed:closing",
            "removed_closing",
            "closing",  # type: ignore[arg-type]
            "closing_combination",
            42,
        ),
        tmp_path,
        first.parameters,
        first.metrics,
        first.metadata,
        first.costs,
    )
    with pytest.raises(ValueError, match="stage 'closing' is not allowed"):
        driver.compile_reversal_confirmation((closing_source, second))

    confirmation_artifacts = []
    for compiled in confirmations:
        artifact = _artifact(compiled, tmp_path)
        confirmation_artifacts.append(
            VerifiedArtifact(
                artifact.job,
                artifact.path,
                artifact.parameters,
                {metric: artifact.metrics.get(metric, 0.1) for metric in METRICS},
                artifact.metadata,
                artifact.costs,
            )
        )
    evidence_path = tmp_path / "reversal.json"
    report_path = tmp_path / "reversal.md"
    compact_path = tmp_path / "compact.md"
    selection_path = tmp_path / "selection.json"
    compact_path.write_text("existing compact\n")
    selection_path.write_text("existing selection\n")

    with pytest.raises(ValueError, match="explicit user-validated interpretation"):
        require_explicit_reversal_validation(
            [first, second, *confirmation_artifacts],
            evidence_path=evidence_path,
            report_path=report_path,
        )

    evidence = json.loads(evidence_path.read_text())
    assert evidence["interpretation_state"] == "explicit_validation_required"
    assert [row["seed"] for row in evidence["confirmations"]] == [43, 44, 43, 44]
    assert {row["source_job_id"] for row in evidence["confirmations"]} == {
        first.job.id,
        second.job.id,
    }
    assert set(evidence["confirmations"][0]["metrics"]) == set(METRICS)
    report = report_path.read_text()
    assert "Explicit user validation is required" in report
    assert "| LiGR with sampled softmax | 43 |" in report
    assert "g2_" not in report
    assert compact_path.read_text() == "existing compact\n"
    assert selection_path.read_text() == "existing selection\n"
    with pytest.raises(ValueError, match="explicit user-validated interpretation"):
        require_explicit_reversal_validation(
            [first, second, *confirmation_artifacts],
            evidence_path=evidence_path,
            report_path=report_path,
        )


def test_control_repeat_compiler_reuses_the_verified_selection(tmp_path: Path) -> None:
    driver = G2OptunaDriver(tmp_path / "study.sqlite3")
    selection = _control_artifact(tmp_path, batch_size=256)

    repeats = driver.compile_control_repeats(selection)

    assert [job.approved.seed for job in repeats] == list(range(43, 52))
    assert {job.approved.stage for job in repeats} == {"control_repeats"}
    assert all(job.parameters["batch_size"] == 256 for job in repeats)
    assert all(job.parameters["selected_control"] is True for job in repeats)
    assert all(
        job.parameters["selected_control_job_id"] == selection.job.id for job in repeats
    )


def test_standalone_boundary_check_rejects_an_omitted_triggered_slot(
    tmp_path: Path,
) -> None:
    driver = G2OptunaDriver(tmp_path / "study.sqlite3")
    winner = _component_artifact(tmp_path, deep_learning_rate=0.128)
    required = driver.compile_lr_boundary(winner)

    with pytest.raises(ValueError, match="omits triggered LR boundaries"):
        require_triggered_lr_boundaries((winner,), required[:1])

    assert require_triggered_lr_boundaries((winner,), required) == required


def test_boundary_winners_are_valid_downstream_prerequisites(tmp_path: Path) -> None:
    driver = G2OptunaDriver(tmp_path / "study.sqlite3")
    control = _control_artifact(tmp_path)
    control = VerifiedArtifact(
        job=control.job,
        path=control.path,
        parameters={**control.parameters, "deep_learning_rate": 0.128},
        metrics=control.metrics,
        metadata=control.metadata,
        costs=control.costs,
    )
    boundary_job = driver.compile_lr_boundary(control)[0]
    boundary = _artifact(boundary_job, tmp_path)

    repeats = driver.compile_control_repeats(boundary)
    component = driver.next_component("standard_sampled_softmax", boundary)

    assert len(repeats) == 9
    assert all("builder" not in job.parameters for job in repeats)
    assert component is not None
    assert component.parameters["batch_size"] == boundary.parameters["batch_size"]


def test_capacity_dependent_study_pins_the_exact_ligr_selection(
    tmp_path: Path,
) -> None:
    driver = G2OptunaDriver(tmp_path / "study.sqlite3")
    control = _control_artifact(tmp_path)
    selected_ligr = _component_artifact(tmp_path, ligr_multiplier=6)

    compiled = driver.next_component(
        "matched_standard_gbce",
        control,
        ligr_selection=selected_ligr,
    )

    assert compiled is not None
    assert compiled.parameters["source_job_id"] == selected_ligr.job.id
    assert compiled.parameters["ligr_multiplier"] == 6


def test_program_verifier_rejects_an_internally_consistent_stale_ligr_source(
    tmp_path: Path,
) -> None:
    ligr_jobs = [
        job
        for job in approved_manifest().jobs_for_stage("component_tuning")
        if job.method == "ligr_sampled_softmax" and job.trial in {1, 2}
    ]
    selected = _artifact(
        CompiledJob(
            ligr_jobs[0],
            {
                "batch_size": 512,
                "embedding_learning_rate": 0.01,
                "deep_learning_rate": 0.02,
                "selected_control_job_id": "control_tuning:control_trial_01",
                "ligr_multiplier": 6,
            },
        ),
        tmp_path,
    )
    stale = _artifact(
        CompiledJob(
            ligr_jobs[1],
            {
                **selected.parameters,
                "embedding_learning_rate": 0.03,
            },
        ),
        tmp_path,
    )
    dependent_job = next(
        job
        for job in approved_manifest().jobs_for_stage("component_tuning")
        if job.method == "ligr_gbce" and job.trial == 1
    )
    dependent = _artifact(
        CompiledJob(
            dependent_job,
            {
                "batch_size": stale.parameters["batch_size"],
                "embedding_learning_rate": 0.04,
                "deep_learning_rate": 0.05,
                "selected_control_job_id": stale.parameters["selected_control_job_id"],
                "source_job_id": stale.job.id,
                "ligr_multiplier": stale.parameters["ligr_multiplier"],
                "gbce_t": 0.6,
            },
        ),
        tmp_path,
    )

    with pytest.raises(ValueError, match="not the selected LiGR winner"):
        require_selected_ligr_lineage((selected, stale, dependent), selected)


@pytest.mark.parametrize("dependent_stage", ("component_tuning", "control_repeats"))
def test_program_verifier_rejects_an_internally_consistent_stale_control_source(
    tmp_path: Path,
    dependent_stage: str,
) -> None:
    controls = approved_manifest().jobs_for_stage("control_tuning")[1:3]
    selected = _artifact(
        CompiledJob(
            controls[0],
            {
                "batch_size": 512,
                "embedding_learning_rate": 0.01,
                "deep_learning_rate": 0.02,
            },
        ),
        tmp_path,
    )
    stale = _artifact(
        CompiledJob(
            controls[1],
            {
                "batch_size": 512,
                "embedding_learning_rate": 0.01,
                "deep_learning_rate": 0.02,
            },
        ),
        tmp_path,
    )
    if dependent_stage == "component_tuning":
        job = next(
            job
            for job in approved_manifest().jobs_for_stage("component_tuning")
            if job.method == "standard_sampled_softmax" and job.trial == 1
        )
        parameters = {
            **stale.parameters,
            "selected_control_job_id": stale.job.id,
        }
    else:
        job = next(
            job
            for job in approved_manifest().jobs_for_stage("control_repeats")
            if job.seed == 43
        )
        parameters = {
            **stale.parameters,
            "selected_control": True,
            "selected_control_job_id": stale.job.id,
        }
    dependent = _artifact(CompiledJob(job, parameters), tmp_path)

    with pytest.raises(ValueError, match="not the selected control winner"):
        require_selected_control_lineage((selected, stale, dependent), selected)


def test_aggregate_selection_records_qualification_selection_and_exact_arithmetic(
    tmp_path: Path,
) -> None:
    baseline = _control_artifact(tmp_path)
    baseline = VerifiedArtifact(
        baseline.job,
        baseline.path,
        baseline.parameters,
        {"recall@100": 0.2, "ndcg@100": 0.1, "coverage@100": 0.3},
        baseline.metadata,
        baseline.costs,
    )
    components = [
        _component_artifact(tmp_path, method)
        for method in (
            "standard_sampled_softmax",
            "standard_gbce",
            "matched_standard_sampled_softmax",
            "matched_standard_gbce",
            "ligr_sampled_softmax",
            "ligr_gbce",
        )
    ]
    components = [
        VerifiedArtifact(
            artifact.job,
            artifact.path,
            artifact.parameters,
            {
                "recall@100": 0.22 if index == 0 else 0.21,
                "ndcg@100": 0.105,
                "coverage@100": 0.33,
            },
            artifact.metadata,
            artifact.costs,
        )
        for index, artifact in enumerate(components)
    ]
    bands = {"recall@100": 0.005, "ndcg@100": 0.002, "coverage@100": 0.01}

    selected = select_aggregate_bundle(baseline, components, None, bands)
    evidence = build_composition_evidence(baseline, selected, components, None, bands)

    assert selected == components[0]
    assert all(candidate["qualified"] for candidate in evidence["candidates"])
    assert sum(candidate["selected"] for candidate in evidence["candidates"]) == 1
    assert evidence["baseline_fallback"]["selected"] is False
    assert len(evidence["candidates"]) == 6
    assert len(evidence["omissions"]) == 6
    assert evidence["mixed_candidate_status"] == {
        "status": "omitted",
        "method": "mixed_sampler",
        "reason": "no_eligible_mixed_winner",
    }
    recall = evidence["metrics"]["recall@100"]
    assert recall["aggregate_gain_points"] == pytest.approx(0.02)
    assert recall["aggregate_gain_percent"] == pytest.approx(10.0)
    assert recall["standalone_sum_points"] == recall["aggregate_gain_points"]
    assert recall["interaction_gap_points"] == 0.0
    assert recall["interaction_label"] == "unresolved"
    assert recall["interaction_band"] == bands["recall@100"]
    path = tmp_path / "composition.json"
    write_composition_evidence(evidence, path)
    assert json.loads(path.read_text()) == evidence
    report = render_compact_report(
        {"Aggregated improvement": [baseline, selected]},
        reference=baseline,
        metric_bands=bands,
        composition_evidence=evidence,
        benchmark_evidence={
            "latency_p50_seconds": 0.01,
            "latency_p95_seconds": 0.02,
            "queries_per_second": 25_600.0,
        },
    )
    assert (
        "| recall@100 | 0.200 | 0.220 | +0.020 | +10.000% | +0.020 | "
        "+0.000 | 0.005 | unresolved |"
    ) in report
    assert "| official SASRec block with sampled softmax | qualified | selected |" in report
    assert "| LiGR with mixed sampling | not qualified | omitted |" in report
    assert "| official SASRec block with sampled softmax | 10.000 | 20.000 | " in report
    assert "g2_" not in report
    assert "qualified_but_not_selected_by_band_aware_rule" not in report
    nonempty = [line for line in report.splitlines() if line]
    assert all(line.startswith(("#", "|")) for line in nonempty)


def test_aggregate_selection_falls_back_to_baseline_when_nothing_qualifies(
    tmp_path: Path,
) -> None:
    baseline = _control_artifact(tmp_path)
    components = [
        _component_artifact(tmp_path, method)
        for method in (
            "standard_sampled_softmax",
            "standard_gbce",
            "matched_standard_sampled_softmax",
            "matched_standard_gbce",
            "ligr_sampled_softmax",
            "ligr_gbce",
        )
    ]
    bands = {"recall@100": 0.01, "ndcg@100": 0.01, "coverage@100": 0.01}

    selected = select_aggregate_bundle(baseline, components, None, bands)
    evidence = build_composition_evidence(baseline, selected, components, None, bands)

    assert selected == baseline
    assert not any(candidate["qualified"] for candidate in evidence["candidates"])
    assert evidence["baseline_fallback"]["selected"] is True
    heading = aggregate_section_heading(baseline, selected)
    assert heading == "Aggregated improvement"
    report = render_compact_report(
        {heading: [baseline]},
        reference=baseline,
        metric_bands=bands,
        composition_evidence=evidence,
        benchmark_evidence={
            "latency_p50_seconds": 0.01,
            "latency_p95_seconds": 0.02,
            "queries_per_second": 25_600.0,
        },
    )
    assert report.startswith(
        "# G2 eSASRec on native Yambda-50M\n\n## Aggregated improvement\n"
    )
    assert "| recalibrated G1 control | retained | selected |" in report
    assert report.count("| control | G1 control |") == 1
    assert "not_qualified_for_promotion" not in report


def test_aggregate_ndcg_promotion_requires_the_quality_cost_pareto_gate(
    tmp_path: Path,
) -> None:
    baseline = _control_artifact(tmp_path)
    components = [
        _component_artifact(tmp_path, method)
        for method in (
            "standard_sampled_softmax",
            "standard_gbce",
            "matched_standard_sampled_softmax",
            "matched_standard_gbce",
            "ligr_sampled_softmax",
            "ligr_gbce",
        )
    ]
    candidate = components[0]
    candidate = VerifiedArtifact(
        candidate.job,
        candidate.path,
        candidate.parameters,
        {**candidate.metrics, "ndcg@100": 0.12},
        candidate.metadata,
        {**candidate.costs, "wall_seconds": 3.0},
    )
    components[0] = candidate
    bands = {"recall@100": 0.01, "ndcg@100": 0.01, "coverage@100": 0.01}

    assert select_aggregate_bundle(baseline, components, None, bands) == baseline

    components[0] = VerifiedArtifact(
        candidate.job,
        candidate.path,
        candidate.parameters,
        candidate.metrics,
        candidate.metadata,
        {**candidate.costs, "wall_seconds": 1.0},
    )
    assert select_aggregate_bundle(baseline, components, None, bands) == components[0]


def test_aggregate_selection_includes_an_eligible_mixed_candidate(
    tmp_path: Path,
) -> None:
    baseline = _control_artifact(tmp_path)
    components = [
        _component_artifact(tmp_path, method)
        for method in (
            "standard_sampled_softmax",
            "standard_gbce",
            "matched_standard_sampled_softmax",
            "matched_standard_gbce",
            "ligr_sampled_softmax",
            "ligr_gbce",
        )
    ]
    mixed_job = approved_manifest().jobs_for_stage("mixed_tuning")[2]
    mixed = _artifact(
        CompiledJob(mixed_job, {}),
        tmp_path,
        recall=0.23,
        ndcg=0.11,
        coverage=0.4,
    )
    bands = {"recall@100": 0.01, "ndcg@100": 0.01, "coverage@100": 0.01}

    selected = select_aggregate_bundle(baseline, components, mixed, bands)
    evidence = build_composition_evidence(baseline, selected, components, mixed, bands)

    assert selected == mixed
    assert len(evidence["candidates"]) == 7
    assert evidence["mixed_candidate_status"]["status"] == "eligible"
    assert evidence["mixed_candidate_status"]["selected"] is True
    assert len(evidence["omissions"]) == 6
