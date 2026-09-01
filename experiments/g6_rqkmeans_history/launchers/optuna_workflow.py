from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Callable

from dcn.config import SemanticHistoryExperiment
from dcn.eval.ranking_evidence import load_ranking_evidence
from experiments.g6_rqkmeans_history.analysis.preflight import (
    PREFLIGHT_EVIDENCE_PATH,
    load_preflight_evidence,
)
from experiments.g6_rqkmeans_history.analysis.rq0_slices import (
    slice_comparison,
    write_slice_comparison,
)
from experiments.g6_rqkmeans_history.launchers.compiled import build_experiment
from experiments.g6_rqkmeans_history.protocol.evidence import (
    VerifiedArtifact,
    archive_run_artifact,
    artifact_state,
    empirical_bands,
    inference_cost_contract,
    load_verified_artifact,
    require_resolved_boundary,
    select_best,
    write_empirical_bands,
)
from experiments.g6_rqkmeans_history.protocol.manifest import (
    REPRESENTATIONS,
    ApprovedJob,
    CompiledJob,
    RANKING_EVIDENCE_GROUP,
    approved_manifest,
    compile_cap_continuation,
    load_compiled_jobs,
    validate_boundary_source,
)
from experiments.g6_rqkmeans_history.protocol.optuna_driver import (
    G6Rq0OptunaDriver,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
QUEUE_LAUNCHER = Path(__file__).with_name("queue_compiled.sh")


class CompiledManifestWriter:
    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, compiled: CompiledJob) -> None:
        existing = [] if not self.path.exists() else load_compiled_jobs(self.path)
        by_identity = {job.identity: job for job in existing}
        prior = by_identity.get(compiled.identity)
        if prior is not None and prior != compiled:
            raise ValueError(f"compiled job changed: {compiled.identity}")
        by_identity[compiled.identity] = compiled
        combined = tuple(by_identity.values())
        for candidate in combined:
            validate_boundary_source(candidate, combined)
        order = {job.id: index for index, job in enumerate(approved_manifest().jobs)}
        self.write_many(
            self.path,
            tuple(
                sorted(
                    by_identity.values(),
                    key=lambda job: (order[job.approved.id], job.attempt),
                )
            ),
        )

    @staticmethod
    def write_many(path: Path, compiled_jobs: tuple[CompiledJob, ...]) -> None:
        document = {
            "manifest_sha256": approved_manifest().sha256,
            "jobs": [
                {
                    "id": compiled.approved.id,
                    "run_name": compiled.run_name,
                    "parameters": compiled.parameters,
                    **(
                        {
                            "attempt": compiled.attempt,
                            "cap_epochs": compiled.cap_epochs,
                        }
                        if compiled.attempt
                        else {}
                    ),
                }
                for compiled in compiled_jobs
            ],
        }
        content = json.dumps(document, indent=2, sort_keys=True) + "\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(content)
        temporary.replace(path)


class QueueSubmitter:
    def __init__(self, batches_root: Path) -> None:
        self.batches_root = batches_root

    def __call__(self, compiled_jobs: tuple[CompiledJob, ...]) -> None:
        digest = hashlib.sha256(
            json.dumps(
                [job.to_contract(approved_manifest()) for job in compiled_jobs],
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()[:16]
        batch = self.batches_root / f"batch_{digest}.json"
        CompiledManifestWriter.write_many(batch, compiled_jobs)
        subprocess.run(
            ["bash", str(QUEUE_LAUNCHER), str(batch)],
            cwd=PROJECT_ROOT,
            check=True,
        )


class OptunaStudyWorkflow:
    def __init__(
        self,
        driver: G6Rq0OptunaDriver,
        *,
        logs_root: Path,
        compiled_path: Path,
        submit: Callable[[tuple[CompiledJob, ...]], None],
    ) -> None:
        self.driver = driver
        self.logs_root = logs_root
        self.compiled_path = compiled_path
        self.writer = CompiledManifestWriter(compiled_path)
        self.submit = submit

    def advance(self, ask: Callable[[], CompiledJob | None]) -> int:
        completed = 0
        while (compiled := ask()) is not None:
            artifact = self.run_compiled((compiled,))[0]
            self.driver.tell(
                compiled,
                artifact.metrics["recall@100"],
                artifact.path,
            )
            completed += 1
        return completed

    def advance_parallel(
        self, asks: dict[str, Callable[[], CompiledJob | None]]
    ) -> int:
        completed = 0
        active = dict(asks)
        while active:
            emitted: list[CompiledJob] = []
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
                self.driver.tell(
                    compiled,
                    artifact.metrics["recall@100"],
                    artifact.path,
                )
                completed += 1
        return completed

    def run_compiled(
        self, compiled_jobs: tuple[CompiledJob, ...]
    ) -> list[VerifiedArtifact]:
        for compiled in compiled_jobs:
            self.writer.append(compiled)
        unresolved = {
            compiled.approved.id: self._latest_attempt(compiled)
            for compiled in compiled_jobs
        }
        verified: dict[str, VerifiedArtifact] = {}
        while unresolved:
            pending: list[CompiledJob] = []
            for job_id, compiled in tuple(unresolved.items()):
                state = artifact_state(compiled, self.logs_root)
                if state == "complete":
                    verified[job_id] = load_verified_artifact(compiled, self.logs_root)
                    del unresolved[job_id]
                elif state == "partial":
                    archive_run_artifact(compiled, self.logs_root, reason="incomplete")
                    pending.append(compiled)
                elif state == "extend_cap":
                    continuation = compile_cap_continuation(compiled)
                    self.writer.append(continuation)
                    archive_run_artifact(
                        compiled, self.logs_root, reason="cap-exhausted"
                    )
                    unresolved[job_id] = continuation
                    pending.append(continuation)
                else:
                    pending.append(compiled)
            remaining = list(pending)
            while remaining:
                wave, remaining = _cache_safe_wave(remaining)
                try:
                    self.submit(tuple(wave))
                except subprocess.CalledProcessError:
                    recovered = False
                    for compiled in wave:
                        if artifact_state(compiled, self.logs_root) == "partial":
                            archive_run_artifact(
                                compiled, self.logs_root, reason="incomplete"
                            )
                            recovered = True
                    if not recovered:
                        raise
        return [verified[compiled.approved.id] for compiled in compiled_jobs]

    def _latest_attempt(self, compiled: CompiledJob) -> CompiledJob:
        candidates = [
            candidate
            for candidate in load_compiled_jobs(self.compiled_path)
            if candidate.approved.id == compiled.approved.id
        ]
        if any(candidate.parameters != compiled.parameters for candidate in candidates):
            raise ValueError(f"compiled job changed: {compiled.approved.id}")
        return max(candidates, key=lambda candidate: candidate.attempt)

    def artifacts(self, jobs: list[ApprovedJob]) -> list[VerifiedArtifact]:
        compiled_by_id: dict[str, CompiledJob] = {}
        for compiled in load_compiled_jobs(self.compiled_path):
            previous = compiled_by_id.get(compiled.approved.id)
            if previous is None or compiled.attempt > previous.attempt:
                compiled_by_id[compiled.approved.id] = compiled
        missing = [job.id for job in jobs if job.id not in compiled_by_id]
        if missing:
            raise ValueError(
                "compiled manifest omits completed study jobs: "
                + ", ".join(sorted(missing))
            )
        return [
            load_verified_artifact(compiled_by_id[job.id], self.logs_root)
            for job in jobs
        ]


@dataclass(frozen=True)
class ProgramResult:
    primary_control: VerifiedArtifact
    original_control: VerifiedArtifact
    treatment_winners: dict[str, VerifiedArtifact]
    semantic_winner: VerifiedArtifact
    semantic_promoted: bool
    selected_primary_method: VerifiedArtifact
    bridge: VerifiedArtifact
    bands: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return {
            "manifest_sha256": approved_manifest().sha256,
            "dataset_size": "native-50m",
            "primary_control": _selection_row(self.primary_control),
            "original_control": _selection_row(self.original_control),
            "treatment_winners": {
                method: _selection_row(artifact)
                for method, artifact in self.treatment_winners.items()
            },
            "semantic_winner": _selection_row(self.semantic_winner),
            "semantic_promoted": self.semantic_promoted,
            "selected_primary_method": _selection_row(self.selected_primary_method),
            "bridge": _selection_row(self.bridge),
            "bands": self.bands,
        }


def run_program(
    workflow: OptunaStudyWorkflow,
    driver: G6Rq0OptunaDriver,
    *,
    bands_path: Path,
) -> ProgramResult:
    workflow.advance(driver.next_primary_control)
    primary_initial = workflow.artifacts(
        approved_manifest().jobs_for_stage("primary_control_tuning")
    )
    primary_seed_winner = select_best(primary_initial, recall_band=0, ndcg_band=0)
    primary_boundaries = workflow.run_compiled(
        driver.compile_lr_boundaries(primary_seed_winner.selection())
    )
    primary = select_best(
        [*primary_initial, *primary_boundaries], recall_band=0, ndcg_band=0
    )
    require_resolved_boundary(primary)

    repeats = workflow.run_compiled(driver.compile_primary_repeats(primary.selection()))
    bands = empirical_bands([primary, *repeats])
    write_empirical_bands(bands_path, bands)
    recall_band = bands["recall@100"]
    ndcg_band = bands["ndcg@100"]

    primary_selection = primary.selection()
    asks: dict[str, Callable[[], CompiledJob | None]] = {
        "original_control": lambda: driver.next_original_control(primary_selection),
        **{
            representation: (
                lambda representation=representation: driver.next_treatment(
                    representation, primary_selection
                )
            )
            for representation in REPRESENTATIONS
        },
    }
    workflow.advance_parallel(asks)

    original, _ = _initial_winner(
        workflow,
        driver,
        approved_manifest().jobs_for_stage("original_control_tuning"),
        recall_band,
        ndcg_band,
    )
    treatment_winners: dict[str, VerifiedArtifact] = {}
    treatment_initial: dict[str, list[VerifiedArtifact]] = {}
    boundary_jobs: list[CompiledJob] = []
    for representation in REPRESENTATIONS:
        artifacts = workflow.artifacts(
            [
                job
                for job in approved_manifest().jobs_for_stage("treatment_tuning")
                if job.method == representation
            ]
        )
        treatment_initial[representation] = artifacts
        winner = select_best(
            artifacts,
            recall_band=recall_band,
            ndcg_band=ndcg_band,
        )
        boundary_jobs.extend(driver.compile_lr_boundaries(winner.selection()))
    treatment_boundaries = workflow.run_compiled(tuple(boundary_jobs))

    for representation in REPRESENTATIONS:
        candidates = [
            *treatment_initial[representation],
            *[
                artifact
                for artifact in treatment_boundaries
                if artifact.compiled.approved.method == representation
            ],
        ]
        winner = select_best(
            candidates,
            recall_band=recall_band,
            ndcg_band=ndcg_band,
        )
        require_resolved_boundary(winner)
        treatment_winners[representation] = winner

    semantic = select_best(
        list(treatment_winners.values()),
        recall_band=recall_band,
        ndcg_band=ndcg_band,
    )
    semantic_promoted = semantic_promotion_eligible(
        primary,
        semantic,
        recall_band=recall_band,
        ndcg_band=ndcg_band,
    )
    workflow.advance(
        lambda: driver.next_bridge(
            primary.selection(),
            original.selection(),
            semantic.selection(),
        )
    )
    bridge, _ = _initial_winner(
        workflow,
        driver,
        approved_manifest().jobs_for_stage("bridge_tuning"),
        recall_band,
        ndcg_band,
    )
    return ProgramResult(
        primary_control=primary,
        original_control=original,
        treatment_winners=treatment_winners,
        semantic_winner=semantic,
        semantic_promoted=semantic_promoted,
        selected_primary_method=semantic if semantic_promoted else primary,
        bridge=bridge,
        bands=bands,
    )


def _initial_winner(
    workflow: OptunaStudyWorkflow,
    driver: G6Rq0OptunaDriver,
    jobs: list[ApprovedJob],
    recall_band: float,
    ndcg_band: float,
) -> tuple[VerifiedArtifact, list[VerifiedArtifact]]:
    initial = workflow.artifacts(jobs)
    first = select_best(initial, recall_band=recall_band, ndcg_band=ndcg_band)
    boundaries = workflow.run_compiled(driver.compile_lr_boundaries(first.selection()))
    winner = select_best(
        [*initial, *boundaries],
        recall_band=recall_band,
        ndcg_band=ndcg_band,
    )
    require_resolved_boundary(winner)
    return winner, initial


def _selection_row(artifact: VerifiedArtifact) -> dict[str, object]:
    return {
        "job_id": artifact.compiled.approved.id,
        "run_name": artifact.compiled.run_name,
        "attempt": artifact.compiled.attempt,
        "cap_epochs": artifact.compiled.cap_epochs,
        "parameters": artifact.compiled.parameters,
        "metrics": artifact.metrics,
        "inference_cost": asdict(inference_cost_contract(artifact)),
    }


def semantic_promotion_eligible(
    control: VerifiedArtifact,
    treatment: VerifiedArtifact,
    *,
    recall_band: float,
    ndcg_band: float,
) -> bool:
    return (
        treatment.metrics["recall@100"] > control.metrics["recall@100"] + recall_band
        and treatment.metrics["ndcg@100"] >= control.metrics["ndcg@100"] - ndcg_band
    )


def _cache_safe_wave(
    pending: list[CompiledJob],
) -> tuple[list[CompiledJob], list[CompiledJob]]:
    wave: list[CompiledJob] = []
    deferred: list[CompiledJob] = []
    semantic_keys: set[tuple[int, int]] = set()
    for compiled in pending:
        if "num_levels" not in compiled.parameters:
            wave.append(compiled)
            continue
        key = (
            int(compiled.parameters["num_levels"]),
            int(compiled.parameters["num_codes"]),
        )
        if key in semantic_keys:
            deferred.append(compiled)
            continue
        semantic_keys.add(key)
        wave.append(compiled)
    return wave, deferred


def _write_program_result(path: Path, result: ProgramResult) -> None:
    content = json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text() != content:
        raise RuntimeError(f"existing G6 RQ0 selection differs: {path}")
    path.write_text(content)


def _write_program_slices(
    path: Path,
    result: ProgramResult,
    *,
    logs_root: Path,
) -> None:
    context_path = (
        logs_root / ".ranking-evidence" / RANKING_EVIDENCE_GROUP / "context.pt"
    )
    control = load_ranking_evidence(
        context_path, result.primary_control.path / "ranking_evidence.pt"
    )
    semantic = load_ranking_evidence(
        context_path, result.semantic_winner.path / "ranking_evidence.pt"
    )
    experiment = build_experiment(result.semantic_winner.compiled)
    if not isinstance(experiment, SemanticHistoryExperiment):
        raise TypeError("G6 RQ0 semantic winner is not a semantic-history experiment")
    experiment.setup()
    document = slice_comparison(
        control,
        semantic,
        semantic_codes=experiment.semantic_codes,
        semantic_base_levels=experiment.semantic.num_levels,
        control_run_name=result.primary_control.compiled.run_name,
        semantic_run_name=result.semantic_winner.compiled.run_name,
    )
    document["manifest_sha256"] = approved_manifest().sha256
    document["selected_job_ids"] = {
        "primary_control": result.primary_control.compiled.approved.id,
        "semantic_winner": result.semantic_winner.compiled.approved.id,
    }
    write_slice_comparison(path, document)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--compiled", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, default=PREFLIGHT_EVIDENCE_PATH)
    parser.add_argument("--logs-root", type=Path, default=Path("generated/logs"))
    parser.add_argument("--bands", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--slices", type=Path)
    arguments = parser.parse_args()
    preflight = load_preflight_evidence(arguments.preflight)
    driver = G6Rq0OptunaDriver(
        arguments.database,
        feasible_training_batches=preflight.feasible_training_batches,
        validation_batch_size=preflight.validation_batch_size,
    )
    workflow = OptunaStudyWorkflow(
        driver,
        logs_root=arguments.logs_root,
        compiled_path=arguments.compiled,
        submit=QueueSubmitter(arguments.compiled.parent / "queue_batches"),
    )
    result = run_program(workflow, driver, bands_path=arguments.bands)
    _write_program_result(arguments.selection, result)
    slices_path = arguments.slices or arguments.selection.with_name(
        "rq0_slices_native50m.json"
    )
    _write_program_slices(slices_path, result, logs_root=arguments.logs_root)


if __name__ == "__main__":
    main()
