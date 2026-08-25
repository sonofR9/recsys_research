from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.g2_esasrec.analysis.benchmark import load_selected_benchmark
from experiments.g2_esasrec.analysis.evidence import (
    VerifiedArtifact,
    aggregate_artifacts,
    build_composition_evidence,
    build_reversal_confirmation_evidence,
    control_band_artifacts,
    empirical_bands,
    load_exact_artifacts,
    mixed_sampler_winner,
    persist_reversal_confirmation_evidence,
    require_selected_control_lineage,
    require_selected_ligr_lineage,
    select_best,
    select_aggregate_bundle,
    select_control_with_fit_gate,
    write_empirical_bands,
    write_composition_evidence,
)
from experiments.g2_esasrec.analysis.fit_evidence import load_fit_evidence
from experiments.g2_esasrec.analysis.report import (
    aggregate_section_heading,
    persist_reversal_confirmation_report,
    write_reports,
)
from experiments.g2_esasrec.configs.local import COMPONENT_METHODS
from experiments.g2_esasrec.protocol.manifest import (
    CompiledJob,
    approved_manifest,
    load_compiled_jobs,
)
from experiments.g2_esasrec.protocol.optuna_driver import (
    require_triggered_lr_boundaries,
)


def require_explicit_reversal_validation(
    artifacts: list[VerifiedArtifact],
    *,
    evidence_path: Path,
    report_path: Path,
) -> None:
    confirmations = [
        artifact
        for artifact in artifacts
        if artifact.job.stage == "reversal_confirmation"
    ]
    if not confirmations:
        return
    artifacts_by_id = {artifact.job.id: artifact for artifact in artifacts}
    source_ids = {
        artifact.parameters.get("source_job_id") for artifact in confirmations
    }
    sources = [
        artifacts_by_id[source_id]
        for source_id in source_ids
        if isinstance(source_id, str) and source_id in artifacts_by_id
    ]
    document = build_reversal_confirmation_evidence(confirmations, sources)
    persist_reversal_confirmation_evidence(document, evidence_path)
    persist_reversal_confirmation_report(document, report_path)
    raise ValueError(
        "reversal confirmation evidence requires explicit user-validated "
        "interpretation before final selection or report generation"
    )


def generate(
    compiled_path: Path,
    logs_root: Path,
    *,
    ledger_path: Path,
    compact_path: Path,
    bands_path: Path,
    selection_path: Path,
    composition_path: Path,
    reversal_evidence_path: Path,
    reversal_report_path: Path,
    fit_evidence_path: Path,
    benchmark_path: Path,
) -> None:
    compiled = load_compiled_jobs(compiled_path)
    approved = approved_manifest()
    required = {job.id for job in approved.jobs if not job.conditional}
    compiled_ids = {job.approved.id for job in compiled}
    missing = required - compiled_ids
    if missing:
        raise ValueError(
            "complete report manifest omits required jobs: "
            + ", ".join(sorted(missing))
        )
    artifacts = load_exact_artifacts(
        [job.approved for job in compiled],
        logs_root,
    )
    require_explicit_reversal_validation(
        artifacts,
        evidence_path=reversal_evidence_path,
        report_path=reversal_report_path,
    )
    by_stage: dict[str, list] = {}
    for artifact in artifacts:
        by_stage.setdefault(artifact.job.stage, []).append(artifact)
    artifacts_by_id = {artifact.job.id: artifact for artifact in artifacts}
    control_trials = by_stage["control_tuning"]
    fit_evidence = load_fit_evidence(fit_evidence_path)
    initial_control = select_control_with_fit_gate(control_trials, fit_evidence)
    required_control_boundaries = require_triggered_lr_boundaries(
        (initial_control,), compiled
    )
    control_pool = [
        *control_trials,
        *(artifacts_by_id[job.approved.id] for job in required_control_boundaries),
    ]
    selected_control = select_control_with_fit_gate(control_pool, fit_evidence)
    require_selected_control_lineage(artifacts, selected_control)
    control_repeats = control_band_artifacts(
        selected_control,
        by_stage["control_repeats"],
    )
    write_empirical_bands(control_repeats, bands_path)
    bands = {
        metric: band.reader_threshold
        for metric, band in empirical_bands(control_repeats).items()
    }
    initial_component_winners = {
        method: select_best(
            [
                artifact
                for artifact in by_stage["component_tuning"]
                if artifact.job.method == method
            ],
            metric_bands=bands,
        )
        for method in COMPONENT_METHODS
    }
    required_component_boundaries = require_triggered_lr_boundaries(
        tuple(initial_component_winners.values()), compiled
    )
    component_boundaries = [
        artifacts_by_id[job.approved.id] for job in required_component_boundaries
    ]
    component_winners = [
        select_best(
            [
                artifact
                for artifact in by_stage["component_tuning"] + component_boundaries
                if artifact.job.method == method
            ],
            metric_bands=bands,
        )
        for method in COMPONENT_METHODS
    ]
    uniform_esasrec = next(
        artifact
        for artifact in component_winners
        if artifact.job.method == "ligr_sampled_softmax"
    )
    require_selected_ligr_lineage(artifacts, uniform_esasrec)
    mixed_trials = by_stage["mixed_tuning"]
    mixed_report_candidates = [
        artifact
        for artifact in mixed_trials
        if artifact.parameters["uniform_fraction"] == 0.6
    ]
    best_mixed_trial = select_best(mixed_trials, metric_bands=bands)
    if best_mixed_trial not in mixed_report_candidates:
        mixed_report_candidates.append(best_mixed_trial)
    mixed_winner = mixed_sampler_winner(
        uniform_esasrec,
        mixed_trials,
        bands,
    )
    aggregate = select_aggregate_bundle(
        selected_control, component_winners, mixed_winner, bands
    )
    benchmark = load_selected_benchmark(
        benchmark_path,
        run_name=aggregate.job.run_name,
        expected_compiled=CompiledJob(aggregate.job, aggregate.parameters),
        logs_root=logs_root,
    )
    official = sorted(by_stage["official"], key=lambda artifact: artifact.job.seed)
    official_mean = aggregate_artifacts(official, run_name="official 3-seed mean")
    rq1_heading = "RQ1: What are official and local eSASRec's metrics?"
    rq3_heading = "RQ3: Does mixed sampling improve coverage without a recall loss?"
    composition = build_composition_evidence(
        selected_control,
        aggregate,
        component_winners,
        mixed_winner,
        bands,
    )
    write_composition_evidence(composition, composition_path)
    aggregate_heading = aggregate_section_heading(selected_control, aggregate)
    aggregate_rows = (
        [selected_control]
        if aggregate == selected_control
        else [selected_control, aggregate]
    )
    questions = {
        rq1_heading: [
            selected_control,
            uniform_esasrec,
            official_mean,
        ],
        "RQ2: What does each pluggable eSASRec component buy?": component_winners,
        rq3_heading: [
            uniform_esasrec,
            *mixed_report_candidates,
        ],
        aggregate_heading: aggregate_rows,
    }
    write_reports(
        artifacts,
        questions,
        selected_control,
        ledger_path=ledger_path,
        compact_path=compact_path,
        references={rq1_heading: selected_control, rq3_heading: uniform_esasrec},
        metric_bands=bands,
        composition_evidence=composition,
        benchmark_evidence=benchmark,
    )
    selection = {
        "manifest_sha256": approved.sha256,
        "dataset_size": "native-50m",
        "control": selected_control.job.run_name,
        "component_winners": {
            artifact.job.method: artifact.job.run_name for artifact in component_winners
        },
        "mixed_winner": None if mixed_winner is None else mixed_winner.job.run_name,
        "best_mixed_trial": best_mixed_trial.job.run_name,
        "mixed_report_trials": [
            artifact.job.run_name for artifact in mixed_report_candidates
        ],
        "aggregate": aggregate.job.run_name,
        "aggregate_section": aggregate_heading,
        "aggregate_selection": composition,
        "selected_model_benchmark": benchmark,
        "bands": bands,
    }
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    selection_path.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("compiled_manifest", type=Path)
    parser.add_argument("--logs-root", type=Path, default=Path("generated/logs"))
    parser.add_argument(
        "--ledger",
        type=Path,
        default=Path("experiments/g2_esasrec/scratchpad/tuning_native50m.md"),
    )
    parser.add_argument(
        "--compact",
        type=Path,
        default=Path("experiments/g2_esasrec/scratchpad/compact_native50m.md"),
    )
    parser.add_argument(
        "--bands",
        type=Path,
        default=Path("experiments/g2_esasrec/evidence/bands_native50m.json"),
    )
    parser.add_argument(
        "--selection",
        type=Path,
        default=Path("experiments/g2_esasrec/evidence/selection_native50m.json"),
    )
    parser.add_argument(
        "--composition",
        type=Path,
        default=Path("experiments/g2_esasrec/evidence/composition_native50m.json"),
    )
    parser.add_argument(
        "--reversal-evidence",
        type=Path,
        default=Path(
            "experiments/g2_esasrec/evidence/reversal_confirmations_native50m.json"
        ),
    )
    parser.add_argument(
        "--reversal-report",
        type=Path,
        default=Path(
            "experiments/g2_esasrec/scratchpad/reversal_confirmations_native50m.md"
        ),
    )
    parser.add_argument("--fit-evidence", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    arguments = parser.parse_args()
    generate(
        arguments.compiled_manifest,
        arguments.logs_root,
        ledger_path=arguments.ledger,
        compact_path=arguments.compact,
        bands_path=arguments.bands,
        selection_path=arguments.selection,
        composition_path=arguments.composition,
        reversal_evidence_path=arguments.reversal_evidence,
        reversal_report_path=arguments.reversal_report,
        fit_evidence_path=arguments.fit_evidence,
        benchmark_path=arguments.benchmark,
    )


if __name__ == "__main__":
    main()
    persist_reversal_confirmation_evidence,
