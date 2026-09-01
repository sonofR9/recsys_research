from __future__ import annotations

import copy
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import statistics
from types import MappingProxyType
from typing import Any, Literal, Mapping, Sequence

from experiments.g1_aggregate_dataset_size.protocol.candidates import (
    AggregateCandidate,
    ApprovalRequired,
    FIXED_MEMBERS,
    aggregate_initial_candidates,
    baseline_initial_candidates,
    batch_followup_candidates,
    batch_initial_candidates,
    batch_lr_calibration_candidates,
    bridge_candidates,
    candidate_by_run,
    horizon_followup_candidates,
    local_lr_candidates,
    optimizer_boundary_candidates,
    repeat_candidates,
)
from experiments.g1_aggregate_dataset_size.launchers.runtime import (
    CandidateOutcome,
    CandidateResult,
    InfeasibleBatchCell,
    load_candidate_result,
    load_infeasible_batch_cells,
    stage_candidates,
    verify_candidate_artifact,
)

from experiments.g1_sasrec_item_ids_likes.analysis.aggregate_report import (
    collect_report_bundle,
)


DatasetSize = Literal["50m", "500m"]
METRICS = ("recall@100", "ndcg@100", "recall@10", "ndcg@10", "coverage@100")
EXPECTED_USER_COUNTS = {"50m": 3414, "500m": 37018}
NATIVE500M_BANDS = {
    "recall@100": 0.003,
    "ndcg@100": 0.001,
    "recall@10": 0.003,
    "ndcg@10": 0.001,
    "coverage@100": 0.1,
}


class EvidenceError(ValueError):
    pass


class SelectionPending(EvidenceError):
    def __init__(self, required_runs: Sequence[str]) -> None:
        self.required_runs = tuple(dict.fromkeys(required_runs))
        super().__init__(
            "selection evidence is incomplete: " + ", ".join(self.required_runs)
        )


@dataclass(frozen=True)
class RunEvidence:
    run_name: str
    dataset_size: DatasetSize
    family: str
    stage: str
    seed: int
    batch_size: int
    embedding_lr: float
    deep_lr: float
    num_layers: int
    member: str | None
    best_epoch: int
    stopped_epoch: int
    horizon_epochs: int | None
    horizon_complete: bool
    selection_resolved: bool
    validation_recall: float
    validation_ndcg: float
    num_users: int
    metrics: Mapping[str, float]
    recipe_sha256: str
    artifact_sha256: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))
        object.__setattr__(
            self, "artifact_sha256", MappingProxyType(dict(self.artifact_sha256))
        )


@dataclass(frozen=True)
class Native500mReuse:
    evidence: dict[str, Any]
    reader_markdown: str
    tuning_markdown: str
    provenance: dict[str, Any]


@dataclass(frozen=True)
class StudyBundle:
    native50m_runs: tuple[RunEvidence, ...]
    native50m_infeasible_batch_cells: tuple[InfeasibleBatchCell, ...]
    native50m_infeasible_ledger: str | None
    native50m_infeasible_ledger_sha256: str | None
    native50m_evidence: dict[str, Any]
    native500m_evidence: dict[str, Any]
    native50m_bands: dict[str, Any]
    native50m_tuning_markdown: str
    native500m_tuning_markdown: str
    reader_markdown: str
    native500m_provenance: dict[str, Any]


def build_study_bundle(
    native50m_runs: Sequence[RunEvidence],
    source_experiment: Path,
    native500m_logs: Path | None = Path("generated/logs"),
    native50m_infeasible_ledger: Path | None = None,
) -> StudyBundle:
    structured_runs = tuple(_detach_run(run) for run in native50m_runs)
    infeasible_cells = (
        tuple(
            sorted(
                load_infeasible_batch_cells(native50m_infeasible_ledger).values(),
                key=lambda cell: cell.candidate.run_name,
            )
        )
        if native50m_infeasible_ledger is not None
        else ()
    )
    infeasible_ledger_sha256 = (
        _sha256(native50m_infeasible_ledger)
        if native50m_infeasible_ledger is not None
        else None
    )
    native50m, bands, native50m_tuning = _build_native50m_documents(
        structured_runs, infeasible_cells, infeasible_ledger_sha256
    )
    native500m = load_native500m_reuse(source_experiment, native500m_logs)
    reader = build_combined_reader({"50m": native50m, "500m": native500m.evidence})
    return StudyBundle(
        native50m_runs=structured_runs,
        native50m_infeasible_batch_cells=infeasible_cells,
        native50m_infeasible_ledger=(
            str(native50m_infeasible_ledger)
            if native50m_infeasible_ledger is not None
            else None
        ),
        native50m_infeasible_ledger_sha256=infeasible_ledger_sha256,
        native50m_evidence=native50m,
        native500m_evidence=native500m.evidence,
        native50m_bands=bands,
        native50m_tuning_markdown=native50m_tuning,
        native500m_tuning_markdown=native500m.tuning_markdown,
        reader_markdown=reader,
        native500m_provenance=native500m.provenance,
    )


def _build_native50m_documents(
    native50m_runs: Sequence[RunEvidence],
    infeasible_batch_cells: Sequence[InfeasibleBatchCell],
    infeasible_ledger_sha256: str | None,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    by_name = _indexed_runs(native50m_runs)
    selected_baseline = _select_calibrated_baseline(by_name)
    selected_batch_size = selected_baseline.batch_size
    selected_baseline_candidate = candidate_by_run(selected_baseline.run_name)
    repeats = [
        selected_baseline,
        *_require_runs(repeat_candidates(selected_baseline_candidate), by_name),
    ]
    bands = build_native50m_bands(selected_baseline, repeats)
    bridges = _resolve_bridges(bridge_candidates(selected_baseline_candidate), by_name)
    aggregate_initial = aggregate_initial_candidates(selected_batch_size)
    selected_by_depth = [
        _select_lr_family(
            tuple(
                candidate
                for candidate in aggregate_initial
                if candidate.num_layers == depth
            ),
            by_name,
        )
        for depth in (4, 6, 8)
    ]
    selected_aggregate = _select(selected_by_depth)
    native50m = build_size_evidence(
        "50m", selected_baseline, selected_aggregate, bridges, bands
    )
    native50m["batch_calibration"] = {
        "selected_batch_size": selected_batch_size,
        "selected_embedding_lr": selected_baseline.embedding_lr,
        "selected_deep_lr": selected_baseline.deep_lr,
        "infeasible_ledger_sha256": infeasible_ledger_sha256,
        "infeasible_cells": [
            {
                "run": cell.candidate.run_name,
                "batch_size": cell.candidate.batch_size,
                "reason": cell.reason,
                "archive_path": cell.archive_path,
            }
            for cell in infeasible_batch_cells
        ],
    }
    tuning = render_tuning_ledger(
        "50m",
        native50m_runs,
        {selected_baseline.run_name, selected_aggregate.run_name},
    )
    return native50m, bands, tuning


def render_tuning_ledger(
    dataset_size: DatasetSize,
    runs: Sequence[RunEvidence],
    selected_run_names: set[str],
) -> str:
    rows = sorted(
        (
            run
            for run in runs
            if run.dataset_size == dataset_size
            and run.stage not in {"batch_initial", "batch_boundary"}
        ),
        key=lambda run: run.run_name,
    )
    if not rows:
        return (
            f"# Aggregate dataset-size tuning ledger — native {dataset_size} "
            "MuTransfer control\n\n"
            "The completed fixed-LR batch diagnostic is audit-only and is "
            "excluded from this tuning ledger.\n"
        )
    lines = [
        f"# Aggregate dataset-size tuning ledger — native {dataset_size} "
        "MuTransfer control",
        "",
        "| family | member | depth | batch | embedding LR | deep LR | horizon | best/stopped | validation Recall@100 | validation NDCG@100 | selected |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for run in rows:
        lines.append(
            f"| {run.family} | {run.member or '—'} | {run.num_layers} | "
            f"{run.batch_size} | {run.embedding_lr!r} | {run.deep_lr!r} | "
            f"{run.horizon_epochs or '—'} | {run.best_epoch}/{run.stopped_epoch} | "
            f"{run.validation_recall:.4f} | {run.validation_ndcg:.4f} | "
            f"{'yes' if run.run_name in selected_run_names else ''} |"
        )
    return "\n".join(lines) + "\n"


def collect_native50m_runs(logs: Path) -> list[RunEvidence]:
    if not logs.exists():
        return []
    runs = []
    prefix = "g1_aggregate_dataset_size_"
    for directory in sorted(logs.iterdir()):
        if not directory.is_dir() or not directory.name.startswith(prefix):
            continue
        try:
            candidate = candidate_by_run(directory.name)
        except ValueError as error:
            raise EvidenceError(
                f"invalid native-50M candidate directory {directory.name}"
            ) from error
        if not verify_candidate_artifact(directory, candidate):
            raise EvidenceError(f"{candidate.run_name}: artifact verification failed")
        try:
            result = load_candidate_result(directory, candidate)
            metadata = _load_json(directory / "training_metadata.json")
            metrics = _load_json(directory / "final_metrics.json")
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise EvidenceError(f"cannot collect {candidate.run_name}") from error
        if not isinstance(metadata, dict) or not isinstance(metrics, dict):
            raise EvidenceError(f"{candidate.run_name}: malformed artifact evidence")
        run = RunEvidence(
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
            best_epoch=_positive_integer(metadata.get("best_epoch"), "best_epoch"),
            stopped_epoch=_positive_integer(
                metadata.get("stopped_epoch"), "stopped_epoch"
            ),
            horizon_epochs=candidate.horizon_epochs,
            horizon_complete=metadata.get("lr_horizon_complete") is True,
            selection_resolved=metadata.get("selection_resolved") is True,
            validation_recall=result.validation_recall,
            validation_ndcg=result.validation_ndcg,
            num_users=_positive_count(metrics.get("num_users"), "num_users"),
            metrics={
                metric: _probability(metrics.get(metric), metric) for metric in METRICS
            },
            recipe_sha256=_recipe_sha256(metadata),
            artifact_sha256=_artifact_sha256(directory),
        )
        _validate_run(run, "50m")
        _validate_tuning_horizon(run)
        runs.append(run)
    return runs


def write_study_bundle(bundle: StudyBundle, experiment: Path) -> dict[str, Path]:
    _verify_native50m_bundle(bundle)
    _verify_native500m_bundle(bundle)
    paths = {
        "native50m_evidence": experiment / "evidence/aggregate_50m.json",
        "native500m_evidence": experiment / "evidence/aggregate_500m.json",
        "native50m_bands": experiment / "evidence/bands_50m.json",
        "native500m_provenance": experiment / "evidence/native500m_provenance.json",
        "native50m_tuning": experiment / "scratchpad/tuning_50m.md",
        "native500m_tuning": experiment / "scratchpad/tuning_500m.md",
        "reader": experiment / "scratchpad/aggregate_reader.md",
    }
    json_documents = {
        "native50m_evidence": bundle.native50m_evidence,
        "native500m_evidence": bundle.native500m_evidence,
        "native50m_bands": bundle.native50m_bands,
        "native500m_provenance": bundle.native500m_provenance,
    }
    for name, document in json_documents.items():
        _write(
            paths[name],
            json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        )
    _write(paths["native50m_tuning"], bundle.native50m_tuning_markdown)
    _write(paths["native500m_tuning"], bundle.native500m_tuning_markdown)
    _write(paths["reader"], bundle.reader_markdown)
    return paths


def build_native50m_bands(
    selected_baseline: RunEvidence,
    repeats: Sequence[RunEvidence],
) -> dict[str, Any]:
    _validate_run(selected_baseline, "50m")
    if selected_baseline.family != "baseline" or selected_baseline.seed != 42:
        raise EvidenceError("native-50M bands require the seed-42 selected baseline")
    rows = sorted(repeats, key=lambda run: run.seed)
    if [run.seed for run in rows] != list(range(42, 52)):
        raise EvidenceError("native-50M bands require exact seeds 42 through 51")
    if len({run.run_name for run in rows}) != len(rows):
        raise EvidenceError("native-50M repeat run names must be unique")
    for run in rows:
        _validate_run(run, "50m")
        _validate_selected_horizon(run)
        if run.family not in {"baseline", "repeat"}:
            raise EvidenceError("native-50M bands accept only baseline repeats")
        if run.recipe_sha256 != selected_baseline.recipe_sha256:
            raise EvidenceError("all repeats must use the exact selected baseline")
    seed_42 = rows[0]
    if seed_42.run_name != selected_baseline.run_name:
        raise EvidenceError("seed 42 must be the exact selected baseline artifact")

    metrics = {}
    for metric in METRICS:
        sample_standard_deviation = statistics.stdev(
            run.metrics[metric] for run in rows
        )
        metrics[metric] = {
            "sample_standard_deviation": sample_standard_deviation,
            "reader_threshold": _round_up_three_decimals(sample_standard_deviation),
        }
    return {
        "dataset_size": "native-50m",
        "description": (
            "Sample standard deviations from ten exact selected-baseline seeds; "
            "not confidence intervals."
        ),
        "selected_baseline_run": selected_baseline.run_name,
        "selected_baseline_recipe_sha256": selected_baseline.recipe_sha256,
        "run_names": [run.run_name for run in rows],
        "seeds": [run.seed for run in rows],
        "artifact_sha256": {run.run_name: dict(run.artifact_sha256) for run in rows},
        "metrics": metrics,
    }


def build_size_evidence(
    dataset_size: DatasetSize,
    baseline: RunEvidence,
    aggregate: RunEvidence,
    bridge_inputs: Sequence[RunEvidence],
    bands: Mapping[str, Any],
) -> dict[str, Any]:
    _validate_run(baseline, dataset_size)
    _validate_run(aggregate, dataset_size)
    if baseline.family != "baseline" or aggregate.family != "aggregate":
        raise EvidenceError(
            "size evidence requires selected baseline and aggregate runs"
        )
    if baseline.seed != 42 or aggregate.seed != 42:
        raise EvidenceError("selection evidence requires seed 42")
    if baseline.batch_size != aggregate.batch_size:
        raise EvidenceError("aggregate batch must match the frozen baseline")
    _validate_selected_horizon(baseline)
    _validate_selected_horizon(aggregate)
    _validate_band_provenance(dataset_size, baseline, bands)

    bridges = list(bridge_inputs)
    expected_count = 13 if dataset_size == "50m" else 11
    if len(bridges) != expected_count:
        label = "thirteen" if dataset_size == "50m" else "eleven"
        raise EvidenceError(
            f"{dataset_size} evidence requires exactly {label} bridge inputs"
        )
    if len({run.run_name for run in bridges}) != len(bridges):
        raise EvidenceError("bridge run names must be unique")
    for bridge in bridges:
        _validate_run(bridge, dataset_size)
        if bridge.family != "bridge" or bridge.seed != 42:
            raise EvidenceError("bridge inputs must be seed-42 bridge runs")
        if bridge.batch_size != baseline.batch_size:
            raise EvidenceError("bridge batch must match the frozen baseline")
        if (bridge.embedding_lr, bridge.deep_lr) != (
            baseline.embedding_lr,
            baseline.deep_lr,
        ):
            raise EvidenceError("bridge rates must match the frozen baseline")
        _validate_selected_horizon(bridge)

    fixed = {bridge.member: bridge for bridge in bridges if bridge.member != "depth"}
    if set(fixed) != set(FIXED_MEMBERS):
        raise EvidenceError("bridge inputs must contain each fixed member exactly once")
    depths = {
        bridge.num_layers: bridge for bridge in bridges if bridge.member == "depth"
    }
    expected_depths = {4, 6, 8} if dataset_size == "50m" else {aggregate.num_layers}
    if set(depths) != expected_depths:
        raise EvidenceError("bridge inputs have the wrong depth diagnostics")
    if aggregate.num_layers not in depths:
        raise EvidenceError("the selected aggregate depth lacks its matched bridge")
    summed_bridges = [*fixed.values(), depths[aggregate.num_layers]]

    improvements: dict[str, dict[str, float | str]] = {}
    for metric in METRICS:
        baseline_value = baseline.metrics[metric]
        aggregate_value = aggregate.metrics[metric]
        aggregate_gain = aggregate_value - baseline_value
        standalone = sum(
            bridge.metrics[metric] - baseline_value for bridge in summed_bridges
        )
        interaction_gap = aggregate_gain - standalone
        threshold = _band_threshold(dataset_size, bands, metric)
        interaction = (
            "unresolved"
            if dataset_size == "50m"
            else (
                "positive"
                if interaction_gap > threshold
                else "negative" if interaction_gap < -threshold else "unresolved"
            )
        )
        improvements[metric] = {
            "baseline": baseline_value,
            "aggregate": aggregate_value,
            "aggregate_gain_points": aggregate_gain,
            "aggregate_gain_percent": 100 * aggregate_gain / baseline_value,
            "summed_standalone_gain_points": standalone,
            "interaction_gap": interaction_gap,
            "interaction": interaction,
        }

    return {
        "claims_status": "ready",
        "dataset_size": dataset_size,
        "selected_baseline": _run_document(baseline),
        "selected_aggregate": _run_document(aggregate),
        "selected_depth": aggregate.num_layers,
        "bridge_input_count": len(bridges),
        "summed_bridge_count": len(summed_bridges),
        "bridges": [_run_document(bridge) for bridge in bridges],
        "summed_bridge_runs": [bridge.run_name for bridge in summed_bridges],
        "excluded_diagnostic_depth_bridges": sorted(
            expected_depths - {aggregate.num_layers}
        ),
        "bands": copy.deepcopy(bands),
        "aggregated_improvement": improvements,
    }


def build_combined_reader(evidence_by_size: Mapping[str, Mapping[str, Any]]) -> str:
    if set(evidence_by_size) != {"50m", "500m"}:
        raise EvidenceError("combined reporting requires both native sizes")
    for dataset_size, evidence in evidence_by_size.items():
        if evidence.get("claims_status") != "ready":
            raise EvidenceError(f"{dataset_size} evidence is not ready")
    lines = [
        "## Aggregated improvement",
        "",
        (
            "Native-50M interaction gaps are descriptive and unresolved; the "
            "single-control band does not calibrate an eleven-bridge estimator."
        ),
        "",
    ]
    for dataset_size, title in (
        ("50m", "Native Yambda-50M"),
        ("500m", "Native Yambda-500M"),
    ):
        lines.extend((f"### {title}", ""))
        lines.extend(_render_size_table(dataset_size, evidence_by_size[dataset_size]))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def load_native500m_reuse(
    source_experiment: Path,
    logs: Path | None = Path("generated/logs"),
) -> Native500mReuse:
    evidence_path = source_experiment / "evidence/aggregate_improvement_results.json"
    reader_path = source_experiment / "scratchpad/aggregate_improvement_reader_500m.md"
    tuning_path = source_experiment / "scratchpad/aggregate_improvement_tuning_500m.md"
    evidence = _load_json(evidence_path)
    if not isinstance(evidence, dict) or evidence.get("claims_status") != "ready":
        raise EvidenceError("frozen native-500M evidence is not ready")
    reader_markdown = reader_path.read_text()
    tuning_markdown = tuning_path.read_text()
    if logs is not None:
        replay = collect_report_bundle(logs)
        if replay.evidence != evidence:
            raise EvidenceError("native-500M raw replay does not match frozen evidence")
        if replay.reader_markdown != reader_markdown:
            raise EvidenceError("native-500M reader replay is not byte-stable")
        if replay.tuning_markdown != tuning_markdown:
            raise EvidenceError("native-500M tuning replay is not byte-stable")
    provenance = {
        "source_experiment": str(source_experiment),
        "evidence_sha256": _sha256(evidence_path),
        "reader_sha256": _sha256(reader_path),
        "tuning_sha256": _sha256(tuning_path),
        "raw_replayed": logs is not None,
        "raw_logs": None if logs is None else str(logs),
    }
    return Native500mReuse(evidence, reader_markdown, tuning_markdown, provenance)


def _indexed_runs(runs: Sequence[RunEvidence]) -> dict[str, RunEvidence]:
    result: dict[str, RunEvidence] = {}
    for run in runs:
        _validate_run(run, "50m")
        _validate_tuning_horizon(run)
        try:
            candidate = candidate_by_run(run.run_name)
        except ValueError as error:
            raise EvidenceError(
                f"unrecognized native-50M candidate {run.run_name}"
            ) from error
        expected = (
            candidate.dataset_size,
            candidate.family,
            candidate.stage,
            candidate.seed,
            candidate.batch_size,
            candidate.embedding_lr,
            candidate.deep_lr,
            candidate.num_layers,
            candidate.member,
            candidate.horizon_epochs,
        )
        actual = (
            run.dataset_size,
            run.family,
            run.stage,
            run.seed,
            run.batch_size,
            run.embedding_lr,
            run.deep_lr,
            run.num_layers,
            run.member,
            run.horizon_epochs,
        )
        if actual != expected:
            raise EvidenceError(
                f"{run.run_name}: candidate identity does not match evidence"
            )
        if run.run_name in result:
            raise EvidenceError(f"duplicate native-50M run evidence: {run.run_name}")
        result[run.run_name] = run
    return result


def _detach_run(run: RunEvidence) -> RunEvidence:
    return replace(
        run,
        metrics=dict(run.metrics),
        artifact_sha256=dict(run.artifact_sha256),
    )


def _select_calibrated_baseline(
    by_name: Mapping[str, RunEvidence],
) -> RunEvidence:
    candidates = batch_lr_calibration_candidates()
    runs = _require_runs(candidates, by_name)
    return _select(runs)


def _require_batch_outcomes(
    candidates: Sequence[AggregateCandidate],
    by_name: Mapping[str, RunEvidence],
    infeasible: Mapping[str, InfeasibleBatchCell],
) -> None:
    missing = [
        candidate.run_name
        for candidate in candidates
        if candidate.run_name not in by_name and candidate.run_name not in infeasible
    ]
    if missing:
        raise SelectionPending(missing)


def _select_lr_family(
    initial: tuple[AggregateCandidate, ...],
    by_name: Mapping[str, RunEvidence],
) -> RunEvidence:
    candidates = list(initial)
    runs = _require_runs(initial, by_name)
    winner = _select(runs)
    local = local_lr_candidates(candidate_by_run(winner.run_name))
    candidates, runs = _extend_surface(candidates, runs, local, by_name)
    winner = _select(runs)
    try:
        boundary = optimizer_boundary_candidates(candidate_by_run(winner.run_name))
    except ApprovalRequired as error:
        raise EvidenceError(str(error)) from error
    candidates, runs = _extend_surface(candidates, runs, boundary, by_name)
    winner = _select(runs)
    try:
        optimizer_boundary_candidates(candidate_by_run(winner.run_name))
    except ApprovalRequired as error:
        raise EvidenceError(str(error)) from error

    horizon = winner.horizon_epochs
    if horizon is not None and winner.best_epoch == horizon:
        if horizon == 36:
            raise EvidenceError("H36 still ends at its best epoch")
        next_horizon: Literal[24, 36] = 24 if horizon == 15 else 36
        corrected = horizon_followup_candidates(tuple(candidates), next_horizon)
        return _select_lr_family(corrected, by_name)
    _validate_selected_horizon(winner)
    return winner


def _resolve_bridges(
    initial: tuple[AggregateCandidate, ...],
    by_name: Mapping[str, RunEvidence],
) -> list[RunEvidence]:
    runs = _require_runs(initial, by_name)
    scheduler_index = next(
        index
        for index, candidate in enumerate(initial)
        if candidate.member == "scheduler"
    )
    scheduler = runs[scheduler_index]
    while scheduler.best_epoch == scheduler.horizon_epochs:
        horizon = scheduler.horizon_epochs
        if horizon == 36:
            raise EvidenceError("scheduler bridge H36 still ends at its best epoch")
        next_horizon: Literal[24, 36] = 24 if horizon == 15 else 36
        corrected = horizon_followup_candidates(
            (candidate_by_run(scheduler.run_name),), next_horizon
        )
        scheduler = _require_runs(corrected, by_name)[0]
    _validate_selected_horizon(scheduler)
    runs[scheduler_index] = scheduler
    return runs


def _extend_surface(
    candidates: list[AggregateCandidate],
    runs: list[RunEvidence],
    additions: tuple[AggregateCandidate, ...],
    by_name: Mapping[str, RunEvidence],
) -> tuple[list[AggregateCandidate], list[RunEvidence]]:
    known = {candidate.run_name for candidate in candidates}
    missing_from_surface = tuple(
        candidate for candidate in additions if candidate.run_name not in known
    )
    candidates.extend(missing_from_surface)
    runs.extend(_require_runs(missing_from_surface, by_name))
    return candidates, runs


def _require_runs(
    candidates: Sequence[AggregateCandidate],
    by_name: Mapping[str, RunEvidence],
) -> list[RunEvidence]:
    missing = [
        candidate.run_name
        for candidate in candidates
        if candidate.run_name not in by_name
    ]
    if missing:
        raise SelectionPending(missing)
    return [by_name[candidate.run_name] for candidate in candidates]


def _select(runs: Sequence[RunEvidence]) -> RunEvidence:
    if not runs:
        raise EvidenceError("selection surface is empty")
    return min(
        runs,
        key=lambda run: (
            -run.validation_recall,
            -run.validation_ndcg,
            run.run_name,
        ),
    )


def _validate_tuning_horizon(run: RunEvidence) -> None:
    scheduled = run.family == "aggregate" or run.member == "scheduler"
    if scheduled:
        if (
            run.horizon_epochs not in {15, 24, 36}
            or run.stopped_epoch != run.horizon_epochs
            or run.horizon_complete is not True
            or not 1 <= run.best_epoch <= run.horizon_epochs
        ):
            raise EvidenceError("scheduled tuning evidence must complete its horizon")
    elif (
        run.horizon_epochs is not None
        or run.horizon_complete is not False
        or run.selection_resolved is not True
        or not 1 <= run.best_epoch <= run.stopped_epoch < 80
    ):
        raise EvidenceError("horizon-free tuning evidence must resolve before cap 80")


def _validate_run(run: RunEvidence, dataset_size: DatasetSize) -> None:
    if run.dataset_size != dataset_size:
        raise EvidenceError("candidate dataset size does not match the report size")
    if run.num_users != EXPECTED_USER_COUNTS[dataset_size]:
        raise EvidenceError(f"{dataset_size} artifact has the wrong user count")
    if not run.run_name or not run.recipe_sha256:
        raise EvidenceError("run identity and recipe provenance are required")
    expected_artifacts = {"training_metadata.json", "final_metrics.json", "sweep.log"}
    if set(run.artifact_sha256) != expected_artifacts or any(
        len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        for digest in run.artifact_sha256.values()
    ):
        raise EvidenceError("raw artifact provenance is incomplete")
    if not isinstance(run.seed, int) or isinstance(run.seed, bool):
        raise EvidenceError("run seed must be an integer")
    _probability(run.validation_recall, "validation_recall")
    _probability(run.validation_ndcg, "validation_ndcg")
    for metric in METRICS:
        value = run.metrics.get(metric)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise EvidenceError(f"{run.run_name}: invalid {metric}")


def _validate_selected_horizon(run: RunEvidence) -> None:
    if run.selection_resolved is not True:
        raise EvidenceError("selected evidence must be selection-resolved")
    scheduled = run.family == "aggregate" or run.member == "scheduler"
    if scheduled:
        if (
            run.horizon_epochs not in {15, 24, 36}
            or run.stopped_epoch != run.horizon_epochs
            or run.horizon_complete is not True
            or not 1 <= run.best_epoch < run.horizon_epochs
        ):
            raise EvidenceError(
                "scheduled evidence must complete a non-endpoint horizon"
            )
    elif (
        run.horizon_epochs is not None
        or run.horizon_complete is not False
        or not 1 <= run.best_epoch <= run.stopped_epoch < 80
    ):
        raise EvidenceError("horizon-free evidence must resolve strictly before cap 80")


def _validate_band_provenance(
    dataset_size: DatasetSize,
    baseline: RunEvidence,
    bands: Mapping[str, Any],
) -> None:
    if dataset_size == "50m":
        if (
            bands.get("dataset_size") != "native-50m"
            or bands.get("selected_baseline_run") != baseline.run_name
            or bands.get("selected_baseline_recipe_sha256") != baseline.recipe_sha256
            or bands.get("seeds") != list(range(42, 52))
        ):
            raise EvidenceError(
                "native-50M band provenance does not match the baseline"
            )
    elif bands.get("dataset_size") not in {None, "native-500m"}:
        raise EvidenceError("native-500M band provenance has the wrong dataset size")


def _band_threshold(
    dataset_size: DatasetSize,
    bands: Mapping[str, Any],
    metric: str,
) -> float:
    if dataset_size == "500m":
        return NATIVE500M_BANDS[metric]
    metrics = bands.get("metrics")
    if not isinstance(metrics, Mapping) or metric not in metrics:
        raise EvidenceError(f"native-50M band is missing {metric}")
    threshold = metrics[metric].get("reader_threshold")
    if not isinstance(threshold, (int, float)) or not math.isfinite(threshold):
        raise EvidenceError(f"native-50M band has invalid {metric}")
    return float(threshold)


def _render_size_table(
    dataset_size: str,
    evidence: Mapping[str, Any],
) -> list[str]:
    control_label = "MuTransfer control" if dataset_size == "50m" else "baseline"
    lines = [
        f"| metric | {control_label} | aggregate | aggregate gain | "
        "summed standalone gain | interaction gap | interaction |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    rows = evidence.get("aggregated_improvement")
    if not isinstance(rows, Mapping):
        raise EvidenceError(
            f"{dataset_size} aggregate evidence lacks metric arithmetic"
        )
    for metric in METRICS:
        row = rows[metric]
        threshold = (
            NATIVE500M_BANDS[metric]
            if dataset_size == "500m"
            else evidence["bands"]["metrics"][metric]["reader_threshold"]
        )
        aggregate_cell = _metric_cell(row, float(threshold))
        lines.append(
            f"| {metric} | {row['baseline']:.3f} | {aggregate_cell} | "
            f"{row['aggregate_gain_points']:+.3f} | "
            f"{row['summed_standalone_gain_points']:+.3f} | "
            f"{row['interaction_gap']:+.3f} | {row['interaction']} |"
        )
    return lines


def _metric_cell(row: Mapping[str, Any], threshold: float) -> str:
    value = f"{row['aggregate_gain_percent']:+.1f}% ({row['aggregate']:.3f})"
    gain = row["aggregate_gain_points"]
    if gain > threshold:
        return f'<span style="color: green">{value}</span>'
    if gain < -threshold:
        return f'<span style="color: red">{value}</span>'
    return value


def _run_document(run: RunEvidence) -> dict[str, Any]:
    return {
        "run": run.run_name,
        "best_epoch": run.best_epoch,
        "stopped_epoch": run.stopped_epoch,
        "validation_recall@100": run.validation_recall,
        "validation_ndcg@100": run.validation_ndcg,
        "metrics": dict(run.metrics),
        "recipe_sha256": run.recipe_sha256,
        "artifact_sha256": dict(run.artifact_sha256),
    }


def _round_up_three_decimals(value: float) -> float:
    return math.ceil(value * 1000 - 1e-12) / 1000


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise EvidenceError(f"cannot read {path}") from error


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact_sha256(directory: Path) -> dict[str, str]:
    try:
        return {
            name: _sha256(directory / name)
            for name in ("training_metadata.json", "final_metrics.json", "sweep.log")
        }
    except OSError as error:
        raise EvidenceError(f"cannot hash raw artifacts in {directory}") from error


def _verify_native500m_bundle(bundle: StudyBundle) -> None:
    provenance = bundle.native500m_provenance
    source = provenance.get("source_experiment")
    logs = provenance.get("raw_logs")
    if (
        provenance.get("raw_replayed") is not True
        or not isinstance(source, str)
        or not source
        or not isinstance(logs, str)
        or not logs
    ):
        raise EvidenceError("native-500M raw evidence must be replayed before writing")
    replay = load_native500m_reuse(Path(source), Path(logs))
    if (
        replay.evidence != bundle.native500m_evidence
        or replay.tuning_markdown != bundle.native500m_tuning_markdown
        or replay.provenance != provenance
    ):
        raise EvidenceError("native-500M bundle does not match independent raw replay")


def _verify_native50m_bundle(bundle: StudyBundle) -> None:
    evidence = bundle.native50m_evidence
    bands = bundle.native50m_bands
    ledger = bundle.native50m_infeasible_ledger
    try:
        replayed_infeasible = (
            tuple(
                sorted(
                    load_infeasible_batch_cells(Path(ledger)).values(),
                    key=lambda cell: cell.candidate.run_name,
                )
            )
            if ledger is not None
            else ()
        )
        replayed_ledger_sha256 = _sha256(Path(ledger)) if ledger is not None else None
    except (OSError, ValueError) as error:
        raise EvidenceError("native-50M infeasible ledger replay failed") from error
    if (
        replayed_infeasible != bundle.native50m_infeasible_batch_cells
        or replayed_ledger_sha256 != bundle.native50m_infeasible_ledger_sha256
    ):
        raise EvidenceError("native-50M infeasible ledger provenance changed")
    rebuilt_evidence, rebuilt_bands, rebuilt_tuning = _build_native50m_documents(
        bundle.native50m_runs,
        replayed_infeasible,
        replayed_ledger_sha256,
    )
    if (
        evidence != rebuilt_evidence
        or bands != rebuilt_bands
        or bundle.native50m_tuning_markdown != rebuilt_tuning
    ):
        raise EvidenceError("native-50M outputs do not match structured run evidence")
    expected_reader = build_combined_reader(
        {"50m": evidence, "500m": bundle.native500m_evidence}
    )
    if bundle.reader_markdown != expected_reader:
        raise EvidenceError("combined reader does not match size-specific evidence")


def _recipe_sha256(metadata: Mapping[str, Any]) -> str:
    keys = (
        "dataset_size",
        "num_epochs",
        "batch_size",
        "physical_batch_size",
        "gradient_accumulation_steps",
        "effective_batch_size",
        "model_dim",
        "item_embedding_dim",
        "embedding_learning_rate",
        "deep_learning_rate",
        "weight_decay",
        "initializer_std",
        "negative_sampling",
        "transfer_invariants",
    )
    recipe = {key: metadata.get(key) for key in keys}
    serialized = json.dumps(recipe, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()


def _positive_integer(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise EvidenceError(f"invalid {name}")
    return value


def _positive_count(value: Any, name: str) -> int:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value < 1
        or not float(value).is_integer()
    ):
        raise EvidenceError(f"invalid {name}")
    return int(value)


def _probability(value: Any, name: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise EvidenceError(f"invalid {name}")
    return float(value)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content)
    temporary.replace(path)
