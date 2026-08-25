from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Callable

from experiments.g2_esasrec.analysis.benchmark import load_selected_benchmark
from experiments.g2_esasrec.analysis.evidence import (
    VerifiedArtifact,
    control_band_artifacts,
    empirical_bands,
    load_exact_artifacts,
    load_verified_artifact,
    mixed_sampler_winner,
    select_best,
    select_aggregate_bundle,
    select_control_with_fit_gate,
    write_empirical_bands,
)
from experiments.g2_esasrec.analysis.fit_evidence import load_fit_evidence
from experiments.g2_esasrec.analysis.generate import (
    generate,
    require_explicit_reversal_validation,
)
from experiments.g2_esasrec.configs.local import COMPONENT_METHODS
from experiments.g2_esasrec.launchers.compiled import encode_compiled_job
from experiments.g2_esasrec.protocol.manifest import (
    CompiledJob,
    approved_manifest,
    load_compiled_jobs,
)
from experiments.g2_esasrec.protocol.optuna_driver import G2OptunaDriver


class CompiledManifestWriter:
    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, compiled: CompiledJob) -> None:
        existing = [] if not self.path.exists() else load_compiled_jobs(self.path)
        by_id = {job.approved.id: job for job in existing}
        prior = by_id.get(compiled.approved.id)
        if prior is not None and prior != compiled:
            raise ValueError(f"compiled job changed: {compiled.approved.id}")
        by_id[compiled.approved.id] = compiled
        order = {job.id: index for index, job in enumerate(approved_manifest().jobs)}
        jobs = sorted(by_id.values(), key=lambda job: order[job.approved.id])
        document = {
            "manifest_sha256": approved_manifest().sha256,
            "jobs": [
                {
                    "id": job.approved.id,
                    "run_name": job.approved.run_name,
                    "parameters": job.parameters,
                }
                for job in jobs
            ],
        }
        self._atomic_write(self.path, document)

    @staticmethod
    def write_single(path: Path, compiled: CompiledJob) -> None:
        CompiledManifestWriter.write_many(path, (compiled,))

    @staticmethod
    def write_many(path: Path, compiled_jobs: tuple[CompiledJob, ...]) -> None:
        document = {
            "manifest_sha256": approved_manifest().sha256,
            "jobs": [
                {
                    "id": compiled.approved.id,
                    "run_name": compiled.approved.run_name,
                    "parameters": compiled.parameters,
                }
                for compiled in compiled_jobs
            ],
        }
        CompiledManifestWriter._atomic_write(path, document)

    @staticmethod
    def _atomic_write(path: Path, document: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
        temporary.replace(path)


class OptunaStudyWorkflow:
    def __init__(
        self,
        driver: G2OptunaDriver,
        *,
        logs_root: Path,
        compiled_path: Path,
        submit: Callable[[CompiledJob], None],
        submit_batch: Callable[[tuple[CompiledJob, ...]], None] | None = None,
    ) -> None:
        self.driver = driver
        self.logs_root = logs_root
        self.writer = CompiledManifestWriter(compiled_path)
        self.submit = submit
        self.submit_batch = submit_batch or self._submit_sequentially

    def _submit_sequentially(self, jobs: tuple[CompiledJob, ...]) -> None:
        for compiled in jobs:
            self.submit(compiled)

    def advance(self, ask: Callable[[], CompiledJob | None]) -> int:
        completed = 0
        while (compiled := ask()) is not None:
            self.writer.append(compiled)
            artifact = self._load_or_submit(compiled)
            self.driver.record_observation(compiled, artifact)
            completed += 1
        return completed

    def run_compiled(self, jobs: tuple[CompiledJob, ...]) -> list[VerifiedArtifact]:
        pending = []
        for compiled in jobs:
            self.writer.append(compiled)
            if self._terminal_state(compiled) == "missing":
                pending.append(compiled)
        if pending:
            self.submit_batch(tuple(pending))
        return [
            load_verified_artifact(compiled.approved, self.logs_root)
            for compiled in jobs
        ]

    def run_selected_benchmark(
        self,
        compiled: CompiledJob,
        destination: Path,
        submit: Callable[[CompiledJob, Path], None],
    ) -> dict[str, object]:
        if destination.exists():
            return load_selected_benchmark(
                destination,
                run_name=compiled.approved.run_name,
                expected_compiled=compiled,
                logs_root=self.logs_root,
            )
        submit(compiled, destination)
        return load_selected_benchmark(
            destination,
            run_name=compiled.approved.run_name,
            expected_compiled=compiled,
            logs_root=self.logs_root,
        )

    def _load_or_submit(self, compiled: CompiledJob) -> VerifiedArtifact:
        if self._terminal_state(compiled) == "complete":
            return load_verified_artifact(compiled.approved, self.logs_root)
        self.submit(compiled)
        return load_verified_artifact(compiled.approved, self.logs_root)

    def _terminal_state(self, compiled: CompiledJob) -> str:
        directory = self.logs_root / compiled.approved.run_name
        terminal = [
            directory / "final_metrics.json",
            directory / "training_metadata.json",
        ]
        if compiled.approved.stage != "official":
            terminal.append(directory / "cost_metrics.json")
        if all(path.exists() for path in terminal):
            load_verified_artifact(compiled.approved, self.logs_root)
            return "complete"
        if directory.exists() and any(directory.iterdir()):
            run_name = compiled.approved.run_name
            raise ValueError(f"partial terminal artifact blocks resume: {run_name}")
        return "missing"

    def advance_parallel(
        self, asks: dict[str, Callable[[], CompiledJob | None]]
    ) -> int:
        completed = 0
        active = dict(asks)
        while active:
            emitted = []
            for name, ask in tuple(active.items()):
                compiled = ask()
                if compiled is None:
                    del active[name]
                else:
                    emitted.append(compiled)
            if not emitted:
                continue
            artifacts = self.run_compiled(tuple(emitted))
            for compiled, artifact in zip(emitted, artifacts, strict=True):
                self.driver.record_observation(compiled, artifact)
                completed += 1
        return completed


def _job_by_run(run_name: str, logs_root: Path):
    matches = [job for job in approved_manifest().jobs if job.run_name == run_name]
    if len(matches) != 1:
        raise ValueError(f"run name is not one approved job: {run_name}")
    return load_verified_artifact(matches[0], logs_root)


def compile_official_jobs() -> tuple[CompiledJob, ...]:
    return tuple(
        CompiledJob(job, {"rectools_version": "0.19.0"})
        for job in approved_manifest().jobs_for_stage("official")
    )


def _compiled_for_artifact(path: Path, artifact: VerifiedArtifact) -> CompiledJob:
    matches = [
        compiled
        for compiled in load_compiled_jobs(path)
        if compiled.approved == artifact.job
        and compiled.parameters == artifact.parameters
    ]
    if len(matches) != 1:
        raise ValueError("selected artifact has no unique verified compiled job")
    return matches[0]


def _component_study(
    workflow: OptunaStudyWorkflow,
    driver: G2OptunaDriver,
    method: str,
    control: VerifiedArtifact,
    ligr: VerifiedArtifact | None,
    bands: dict[str, float],
) -> VerifiedArtifact:
    def ask() -> CompiledJob | None:
        return driver.next_component(method, control, ligr_selection=ligr)

    workflow.advance(ask)
    initial_jobs = [
        job
        for job in approved_manifest().jobs_for_stage("component_tuning")
        if job.method == method
    ]
    initial = load_exact_artifacts(initial_jobs, workflow.logs_root)
    initial_winner = select_best(initial, metric_bands=bands)
    boundaries = workflow.run_compiled(driver.compile_lr_boundary(initial_winner))
    return select_best([*initial, *boundaries], metric_bands=bands)


def run_program(
    workflow: OptunaStudyWorkflow,
    driver: G2OptunaDriver,
    *,
    fit_evidence_path: Path,
    bands_path: Path,
    ledger_path: Path,
    compact_path: Path,
    selection_path: Path,
    composition_path: Path,
    reversal_evidence_path: Path,
    reversal_report_path: Path,
    benchmark_path: Path,
    submit_benchmark: Callable[[CompiledJob, Path], None],
) -> None:
    if workflow.writer.path.exists():
        existing = load_compiled_jobs(workflow.writer.path)
        reversal_jobs = [
            compiled.approved
            for compiled in existing
            if compiled.approved.stage == "reversal_confirmation"
        ]
        if reversal_jobs:
            confirmations = load_exact_artifacts(reversal_jobs, workflow.logs_root)
            source_ids = {
                artifact.parameters["source_job_id"] for artifact in confirmations
            }
            source_jobs = [
                job for job in approved_manifest().jobs if job.id in source_ids
            ]
            sources = load_exact_artifacts(source_jobs, workflow.logs_root)
            require_explicit_reversal_validation(
                [*sources, *confirmations],
                evidence_path=reversal_evidence_path,
                report_path=reversal_report_path,
            )
    fit = load_fit_evidence(fit_evidence_path)
    workflow.advance(driver.next_control)
    control_trials = load_exact_artifacts(
        approved_manifest().jobs_for_stage("control_tuning"),
        workflow.logs_root,
    )
    initial_control = select_control_with_fit_gate(control_trials, fit)
    control_boundaries = workflow.run_compiled(
        driver.compile_lr_boundary(initial_control)
    )
    control = select_control_with_fit_gate([*control_trials, *control_boundaries], fit)
    repeats = workflow.run_compiled(driver.compile_control_repeats(control))
    band_artifacts = control_band_artifacts(control, repeats)
    write_empirical_bands(band_artifacts, bands_path)
    bands = {
        metric: band.reader_threshold
        for metric, band in empirical_bands(band_artifacts).items()
    }

    ordered_methods = (
        "ligr_sampled_softmax",
        "standard_sampled_softmax",
        "standard_gbce",
        "matched_standard_sampled_softmax",
        "matched_standard_gbce",
        "ligr_gbce",
    )
    if set(ordered_methods) != set(COMPONENT_METHODS):
        raise RuntimeError("component dependency order does not cover the manifest")

    def finish_methods(methods: tuple[str, ...]) -> dict[str, VerifiedArtifact]:
        initial_by_method = {
            method: load_exact_artifacts(
                [
                    job
                    for job in approved_manifest().jobs_for_stage("component_tuning")
                    if job.method == method
                ],
                workflow.logs_root,
            )
            for method in methods
        }
        initial_winners = {
            method: select_best(rows, metric_bands=bands)
            for method, rows in initial_by_method.items()
        }
        boundary_jobs = tuple(
            compiled
            for method in methods
            for compiled in driver.compile_lr_boundary(initial_winners[method])
        )
        boundary_artifacts = workflow.run_compiled(boundary_jobs)
        return {
            method: select_best(
                [
                    *initial_by_method[method],
                    *[
                        artifact
                        for artifact in boundary_artifacts
                        if artifact.job.method == method
                    ],
                ],
                metric_bands=bands,
            )
            for method in methods
        }

    independent = ordered_methods[:3]
    independent_asks = {
        method: (lambda method=method: driver.next_component(method, control))
        for method in independent
    }
    workflow.advance_parallel(independent_asks)
    winners = finish_methods(independent)

    ligr = winners["ligr_sampled_softmax"]
    dependent = ordered_methods[3:]
    dependent_asks = {
        method: (
            lambda method=method: driver.next_component(
                method,
                control,
                ligr_selection=ligr,
            )
        )
        for method in dependent
    }
    workflow.advance_parallel(dependent_asks)
    winners.update(finish_methods(dependent))

    workflow.advance(lambda: driver.next_mixed(ligr))
    mixed = load_exact_artifacts(
        approved_manifest().jobs_for_stage("mixed_tuning"), workflow.logs_root
    )
    selected_mixed = mixed_sampler_winner(ligr, mixed, bands)
    selected_aggregate = select_aggregate_bundle(
        control, winners.values(), selected_mixed, bands
    )

    workflow.run_selected_benchmark(
        _compiled_for_artifact(workflow.writer.path, selected_aggregate),
        benchmark_path,
        submit_benchmark,
    )
    workflow.run_compiled(compile_official_jobs())
    generate(
        workflow.writer.path,
        workflow.logs_root,
        ledger_path=ledger_path,
        compact_path=compact_path,
        bands_path=bands_path,
        selection_path=selection_path,
        composition_path=composition_path,
        reversal_evidence_path=reversal_evidence_path,
        reversal_report_path=reversal_report_path,
        fit_evidence_path=fit_evidence_path,
        benchmark_path=benchmark_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "study", choices=("control", "component", "mixed", "program", "confirmation")
    )
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--compiled", type=Path, required=True)
    parser.add_argument("--logs-root", type=Path, default=Path("generated/logs"))
    parser.add_argument("--method")
    parser.add_argument("--selected-control-run")
    parser.add_argument("--ligr-selection-run")
    parser.add_argument("--fit-evidence", type=Path)
    parser.add_argument("--bands", type=Path)
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path(
            "experiments/g2_esasrec/evidence/selected_benchmark_native50m.json"
        ),
    )
    parser.add_argument("--implicated-run", action="append", default=[])
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
    arguments = parser.parse_args()
    driver = G2OptunaDriver(arguments.database)
    next_path = arguments.compiled.with_suffix(".next.json")
    launcher = Path(__file__).with_name("queue_compiled.sh")
    benchmark_launcher = Path(__file__).with_name("queue_selected_benchmark.sh")

    def submit(compiled: CompiledJob) -> None:
        CompiledManifestWriter.write_single(next_path, compiled)
        subprocess.run(["bash", str(launcher), str(next_path)], check=True)

    def submit_batch(compiled: tuple[CompiledJob, ...]) -> None:
        CompiledManifestWriter.write_many(next_path, compiled)
        subprocess.run(["bash", str(launcher), str(next_path)], check=True)

    def submit_benchmark(compiled: CompiledJob, destination: Path) -> None:
        subprocess.run(
            [
                "bash",
                str(benchmark_launcher),
                compiled.approved.run_name,
                encode_compiled_job(compiled),
                str(destination.resolve()),
            ],
            check=True,
        )

    workflow = OptunaStudyWorkflow(
        driver,
        logs_root=arguments.logs_root,
        compiled_path=arguments.compiled,
        submit=submit,
        submit_batch=submit_batch,
    )
    if arguments.study == "program":
        if arguments.fit_evidence is None or arguments.bands is None:
            parser.error("program requires --fit-evidence and --bands output path")
        run_program(
            workflow,
            driver,
            fit_evidence_path=arguments.fit_evidence,
            bands_path=arguments.bands,
            ledger_path=arguments.ledger,
            compact_path=arguments.compact,
            selection_path=arguments.selection,
            composition_path=arguments.composition,
            reversal_evidence_path=arguments.reversal_evidence,
            reversal_report_path=arguments.reversal_report,
            benchmark_path=arguments.benchmark,
            submit_benchmark=submit_benchmark,
        )
        return
    if arguments.study == "confirmation":
        if len(arguments.implicated_run) != 2:
            parser.error("confirmation requires exactly two --implicated-run values")
        implicated = tuple(
            _job_by_run(run_name, arguments.logs_root)
            for run_name in arguments.implicated_run
        )
        confirmations = workflow.run_compiled(
            driver.compile_reversal_confirmation(implicated)
        )
        require_explicit_reversal_validation(
            [*implicated, *confirmations],
            evidence_path=arguments.reversal_evidence,
            report_path=arguments.reversal_report,
        )
        return
    if arguments.study == "control":
        ask = driver.next_control
    elif arguments.study == "component":
        if (
            arguments.method is None
            or arguments.selected_control_run is None
            or arguments.fit_evidence is None
        ):
            parser.error(
                "component requires --method, --selected-control-run, "
                "and --fit-evidence"
            )
        control = _job_by_run(arguments.selected_control_run, arguments.logs_root)
        select_control_with_fit_gate(
            [control], load_fit_evidence(arguments.fit_evidence)
        )
        ligr = (
            None
            if arguments.ligr_selection_run is None
            else _job_by_run(arguments.ligr_selection_run, arguments.logs_root)
        )

        def ask() -> CompiledJob | None:
            return driver.next_component(
                arguments.method,
                control,
                ligr_selection=ligr,
            )

    else:
        if arguments.ligr_selection_run is None:
            parser.error("mixed requires --ligr-selection-run")
        ligr = _job_by_run(arguments.ligr_selection_run, arguments.logs_root)

        def ask() -> CompiledJob | None:
            return driver.next_mixed(ligr)

    workflow.advance(ask)
    if arguments.study == "control":
        if arguments.fit_evidence is None:
            parser.error("control requires --fit-evidence")
        initial = load_exact_artifacts(
            approved_manifest().jobs_for_stage("control_tuning"),
            arguments.logs_root,
        )
        fit = load_fit_evidence(arguments.fit_evidence)
        initial_winner = select_control_with_fit_gate(initial, fit)
        boundaries = workflow.run_compiled(driver.compile_lr_boundary(initial_winner))
        winner = select_control_with_fit_gate([*initial, *boundaries], fit)
        print(winner.job.run_name)
    elif arguments.study == "component":
        if arguments.bands is None or arguments.method is None:
            parser.error("component requires --bands and --method")
        bands_document = json.loads(arguments.bands.read_text())
        bands = {
            name: float(values["reader_threshold"])
            for name, values in bands_document["metrics"].items()
        }
        initial_jobs = [
            job
            for job in approved_manifest().jobs_for_stage("component_tuning")
            if job.method == arguments.method
        ]
        initial = load_exact_artifacts(initial_jobs, arguments.logs_root)
        initial_winner = select_best(initial, metric_bands=bands)
        boundaries = workflow.run_compiled(driver.compile_lr_boundary(initial_winner))
        winner = select_best([*initial, *boundaries], metric_bands=bands)
        print(winner.job.run_name)


if __name__ == "__main__":
    main()
