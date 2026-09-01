from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import json
import math
import os
from pathlib import Path
import re
from typing import Iterator, Literal, Mapping

from experiments.g1_aggregate_dataset_size.protocol.candidates import (
    AggregateCandidate,
    ApprovalRequired,
    aggregate_initial_candidates,
    baseline_initial_candidates,
    batch_followup_candidates,
    batch_initial_candidates,
    batch_lr_calibration_candidates,
    bridge_candidates,
    horizon_followup_candidates,
    local_lr_candidates,
    optimizer_boundary_candidates,
    repeat_candidates,
)
from experiments.g1_sasrec_item_ids_likes.launchers.verify_artifact import (
    verify_config,
)


StageRequest = Literal[
    "batch_lr_calibration",
    "batch_initial",
    "batch_followup",
    "baseline_initial",
    "baseline_followup",
    "repeats",
    "bridges",
    "aggregate_initial",
    "aggregate_followups",
    "scheduler_followups",
]
CONFIG_PATH = (
    Path(__file__).parents[1] / "configs" / "aggregate_variant.py"
)
EXPECTED_FULL_USER_COUNT = 3414
INFEASIBLE_LEDGER_NAME = ".g1-aggregate-50m-infeasible.json"
_METRIC_NUMBER = r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"


@dataclass(frozen=True)
class CandidateResult:
    candidate: AggregateCandidate
    validation_recall: float
    validation_ndcg: float
    best_epoch: int

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(value) and 0 <= value <= 1
            for value in (self.validation_recall, self.validation_ndcg)
        ):
            raise ValueError("validation metrics must be finite probabilities")
        if not 1 <= self.best_epoch <= self.candidate.num_epochs:
            raise ValueError("best epoch is outside the candidate training horizon")


@dataclass(frozen=True)
class InfeasibleBatchCell:
    candidate: AggregateCandidate
    archive_path: str
    reason: Literal["cuda_out_of_memory"] = "cuda_out_of_memory"

    def __post_init__(self) -> None:
        if not _is_batch_cell(self.candidate):
            raise ValueError("only approved batch-calibration cells may be infeasible")
        archive = Path(self.archive_path)
        if (
            archive.parent.name != "old"
            or not archive.name.startswith(f"{self.candidate.run_name}.infeasible-")
        ):
            raise ValueError("infeasible evidence must name its preserved old/ artifact")


CandidateOutcome = CandidateResult | InfeasibleBatchCell


def completion_is_valid(
    candidate: AggregateCandidate,
    metadata: Mapping[str, object],
    metrics: Mapping[str, object],
) -> bool:
    if metrics.get("num_users") != EXPECTED_FULL_USER_COUNT:
        return False
    if metadata.get("max_epochs") != candidate.num_epochs:
        return False
    best_epoch = metadata.get("best_epoch")
    stopped_epoch = metadata.get("stopped_epoch")
    if not _positive_int(best_epoch) or not _positive_int(stopped_epoch):
        return False
    if best_epoch > stopped_epoch:
        return False
    if candidate.horizon_epochs is None:
        return (
            stopped_epoch < 80
            and metadata.get("early_stopped") is True
            and metadata.get("selection_resolved") is True
            and metadata.get("best_epoch_at_cap") is False
        )
    horizon = candidate.horizon_epochs
    return (
        metadata.get("epochs_trained") == horizon
        and stopped_epoch == horizon
        and metadata.get("lr_schedule_horizon_epochs") == horizon
        and metadata.get("lr_horizon_complete") is True
        and _valid_scheduled_lr_traces(candidate, metadata)
    )


def verify_candidate_artifact(
    directory: Path, candidate: AggregateCandidate
) -> bool:
    try:
        if not verify_config(
            directory,
            CONFIG_PATH,
            [f"G1_AGGREGATE_RUN={candidate.run_name}"],
        ):
            return False
        metadata = _load_mapping(directory / "training_metadata.json")
        metrics = _load_mapping(directory / "final_metrics.json")
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return completion_is_valid(candidate, metadata, metrics)


def load_candidate_result(
    directory: Path, candidate: AggregateCandidate
) -> CandidateResult:
    if not verify_candidate_artifact(directory, candidate):
        raise ValueError(f"{candidate.run_name}: artifact is incomplete or incompatible")
    metadata = _load_mapping(directory / "training_metadata.json")
    best_epoch = metadata["best_epoch"]
    assert isinstance(best_epoch, int)
    recall, ndcg = _best_epoch_metrics(directory / "sweep.log", best_epoch)
    return CandidateResult(candidate, recall, ndcg, best_epoch)


def load_verified_results(
    logs: Path, infeasible_ledger: Path | None = None
) -> dict[str, CandidateOutcome]:
    from experiments.g1_aggregate_dataset_size.protocol.candidates import candidate_by_run

    results: dict[str, CandidateOutcome] = {}
    ledger = infeasible_ledger or logs / INFEASIBLE_LEDGER_NAME
    results.update(load_infeasible_batch_cells(ledger))
    if not logs.exists():
        return results
    for directory in logs.iterdir():
        if not directory.is_dir() or directory.parent.name == "old":
            continue
        try:
            candidate = candidate_by_run(directory.name)
            result = load_candidate_result(directory, candidate)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        results[candidate.run_name] = result
    return results


def load_infeasible_batch_cells(path: Path) -> dict[str, InfeasibleBatchCell]:
    if not path.exists():
        return {}
    document = _load_mapping(path)
    if document.get("version") != 1 or document.get("dataset_size") != "50m":
        raise ValueError("infeasible batch ledger has an incompatible schema")
    rows = document.get("cells")
    if not isinstance(rows, list):
        raise ValueError("infeasible batch ledger cells are absent")
    from experiments.g1_aggregate_dataset_size.protocol.candidates import candidate_by_run

    cells: dict[str, InfeasibleBatchCell] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"run_name", "reason", "archive_path"}:
            raise ValueError("malformed infeasible batch ledger row")
        run_name = row["run_name"]
        if not isinstance(run_name, str) or row["reason"] != "cuda_out_of_memory":
            raise ValueError("unsupported infeasible batch ledger state")
        archive_path = row["archive_path"]
        if not isinstance(archive_path, str):
            raise ValueError("infeasible batch archive path is absent")
        cell = InfeasibleBatchCell(candidate_by_run(run_name), archive_path)
        archive = path.parent / archive_path
        if not archive.is_dir():
            raise ValueError(f"missing preserved infeasible artifact: {archive_path}")
        if not _has_recognized_cuda_oom(archive):
            raise ValueError(f"infeasible artifact lacks CUDA OOM evidence: {archive_path}")
        if run_name in cells:
            raise ValueError("duplicate infeasible batch ledger cell")
        cells[run_name] = cell
    return cells


def archive_infeasible_batch_artifact(
    directory: Path, candidate: AggregateCandidate, ledger_path: Path
) -> InfeasibleBatchCell:
    if directory.name != candidate.run_name or not _is_batch_cell(candidate):
        raise ValueError("infeasible archival requires the exact batch candidate directory")
    if not _has_recognized_cuda_oom(directory):
        raise ValueError("batch artifact has no recognized CUDA OOM evidence")
    logs = directory.parent
    lock_path = logs / ".run-locks" / f"{directory.name}.lock"
    ledger_lock_path = ledger_path.with_name(f"{ledger_path.name}.lock")
    with _exclusive_lock(lock_path, f"artifact is owned by an active run: {directory}"):
        with _exclusive_lock(
            ledger_lock_path, f"infeasible ledger is locked: {ledger_path}"
        ):
            if not directory.is_dir() or not _has_recognized_cuda_oom(directory):
                raise RuntimeError("infeasible artifact changed before archival")
            existing = load_infeasible_batch_cells(ledger_path)
            archive = _move_to_archive(directory, "infeasible")
            relative_archive = os.path.relpath(archive, ledger_path.parent)
            cell = InfeasibleBatchCell(candidate, relative_archive)
            if candidate.run_name in existing and existing[candidate.run_name] != cell:
                os.replace(archive, directory)
                raise ValueError("batch cell already has different infeasibility evidence")
            existing[candidate.run_name] = cell
            document = _infeasible_document(existing)
            try:
                _write_json_atomic(ledger_path, document)
            except Exception:
                os.replace(archive, directory)
                raise
            return cell


def archive_retry_artifact(directory: Path) -> Path:
    return _archive_artifact(directory, "incomplete")


def stage_candidates(
    stage: StageRequest,
    results: Mapping[str, CandidateOutcome],
    *,
    aggregate_depth: int | None = None,
) -> tuple[AggregateCandidate, ...]:
    _validate_outcomes(results)
    if stage == "batch_lr_calibration":
        return tuple(
            candidate
            for candidate in batch_lr_calibration_candidates()
            if candidate.run_name not in results
        )
    if stage == "batch_initial":
        raise ValueError("the fixed-LR batch surface is immutable audit-only evidence")
    if stage == "batch_followup":
        raise ValueError("the fixed-LR batch surface is immutable audit-only evidence")

    selected_baseline = _selected_calibrated_baseline(results)
    selected_batch = selected_baseline.batch_size
    if stage == "baseline_initial":
        return ()
    if stage == "baseline_followup":
        return ()

    repeats = repeat_candidates(selected_baseline)
    if stage == "repeats":
        return repeats

    _require_results(
        (selected_baseline, *repeats), results, "ten baseline repeats"
    )
    bridges = bridge_candidates(selected_baseline)
    if stage == "bridges":
        return bridges
    if stage == "scheduler_followups":
        scheduler = next(candidate for candidate in bridges if candidate.member == "scheduler")
        _require_results((scheduler,), results, "scheduler bridge")
        return _scheduled_followups((scheduler,), results)

    _require_results(bridges, results, "thirteen matched bridges")
    scheduler = next(candidate for candidate in bridges if candidate.member == "scheduler")
    if _scheduled_followups((scheduler,), results):
        raise RuntimeError("scheduler bridge horizon correction is unresolved")
    aggregates = aggregate_initial_candidates(selected_batch)
    if stage == "aggregate_initial":
        return aggregates
    if stage == "aggregate_followups":
        if aggregate_depth not in (4, 6, 8):
            raise ValueError("aggregate_followups requires depth 4, 6, or 8")
        surface = tuple(
            candidate
            for candidate in aggregates
            if candidate.num_layers == aggregate_depth
        )
        _require_results(surface, results, f"depth-{aggregate_depth} aggregate surface")
        return _scheduled_followups(surface, results)
    raise ValueError(f"unknown aggregate stage {stage!r}")


def _selected_batch(results: Mapping[str, CandidateOutcome]) -> int:
    return _selected_calibrated_baseline(results).batch_size


def _selected_calibrated_baseline(
    results: Mapping[str, CandidateOutcome],
) -> AggregateCandidate:
    candidates = batch_lr_calibration_candidates()
    _require_results(candidates, results, "six-cell batch/LR calibration")
    if any(not isinstance(results[candidate.run_name], CandidateResult) for candidate in candidates):
        raise RuntimeError("six-cell batch/LR calibration requires six verified results")
    return _select(candidates, results)


def _select_resolved_batch_boundary(
    initial_winner: AggregateCandidate,
    initial: tuple[AggregateCandidate, ...],
    followups: tuple[AggregateCandidate, ...],
    results: Mapping[str, CandidateOutcome],
) -> AggregateCandidate:
    winner = _select((*initial, *followups), results)
    feasible_batches = tuple(
        candidate.batch_size
        for candidate in (*initial, *followups)
        if isinstance(results[candidate.run_name], CandidateResult)
    )
    feasible_edge = (
        min(feasible_batches)
        if initial_winner.batch_size == 640
        else max(feasible_batches)
    )
    if winner.batch_size == feasible_edge:
        raise ApprovalRequired("outer feasible batch boundary still wins")
    return winner


def _selected_baseline(
    batch_size: int, results: Mapping[str, CandidateOutcome]
) -> AggregateCandidate:
    initial = baseline_initial_candidates(batch_size)
    _require_results(initial, results, "baseline selection")
    followups = _lr_followups(initial, results)
    if followups:
        _require_results(followups, results, "baseline bounded LR selection")
        followups = _lr_followups(initial, results)
        if followups:
            raise RuntimeError("baseline bounded LR selection is unresolved")
    candidates = _resolved_lr_surface(initial, results)
    return _select(candidates, results)


def _lr_followups(
    initial: tuple[AggregateCandidate, ...],
    results: Mapping[str, CandidateOutcome],
) -> tuple[AggregateCandidate, ...]:
    winner = _select(initial, results)
    local = local_lr_candidates(winner)
    if local and not _has_results(local, results):
        return local
    selection = (*initial, *local)
    winner = _select(selection, results)
    boundary = optimizer_boundary_candidates(winner)
    if boundary and not _has_results(boundary, results):
        return boundary
    if boundary:
        winner = _select((*selection, *boundary), results)
        optimizer_boundary_candidates(winner)
    return ()


def _scheduled_followups(
    initial: tuple[AggregateCandidate, ...],
    results: Mapping[str, CandidateOutcome],
) -> tuple[AggregateCandidate, ...]:
    horizon = initial[0].horizon_epochs
    assert horizon is not None
    surface = initial
    while True:
        _require_results(surface, results, f"H{horizon} schedule surface")
        bounded = _scheduled_bounded_surface(surface, results)
        if bounded:
            return bounded
        completed = _resolved_lr_surface(surface, results)
        winner = _select(completed, results)
        if _result(results[winner.run_name]).best_epoch != horizon:
            return ()
        if horizon == 36:
            raise ApprovalRequired("H36 still ends at its best epoch")
        next_horizon = 24 if horizon == 15 else 36
        surface = horizon_followup_candidates(surface, next_horizon)
        if not _has_results(surface, results):
            return surface
        horizon = next_horizon


def _scheduled_bounded_surface(
    initial: tuple[AggregateCandidate, ...],
    results: Mapping[str, CandidateOutcome],
) -> tuple[AggregateCandidate, ...]:
    if initial[0].family == "bridge":
        return ()
    winner = _select(initial, results)
    local = local_lr_candidates(winner)
    if local and not _has_results(local, results):
        return local
    selection = (*initial, *local)
    winner = _select(selection, results)
    boundary = optimizer_boundary_candidates(winner)
    if boundary and not _has_results(boundary, results):
        return boundary
    if boundary:
        winner = _select((*selection, *boundary), results)
        optimizer_boundary_candidates(winner)
    return ()


def _resolved_lr_surface(
    initial: tuple[AggregateCandidate, ...],
    results: Mapping[str, CandidateOutcome],
) -> tuple[AggregateCandidate, ...]:
    if initial[0].family == "bridge":
        return initial
    winner = _select(initial, results)
    local = local_lr_candidates(winner)
    if local:
        _require_results(local, results, "approved local LR surface")
    selection = (*initial, *local)
    winner = _select(selection, results)
    boundary = optimizer_boundary_candidates(winner)
    if boundary:
        _require_results(boundary, results, "approved optimizer boundary surface")
        winner = _select((*selection, *boundary), results)
        optimizer_boundary_candidates(winner)
    return (*selection, *boundary)


def _select(
    candidates: tuple[AggregateCandidate, ...],
    results: Mapping[str, CandidateOutcome],
) -> AggregateCandidate:
    _require_results(candidates, results, "selection surface")
    feasible = tuple(
        candidate
        for candidate in candidates
        if isinstance(results[candidate.run_name], CandidateResult)
    )
    if not feasible:
        raise ApprovalRequired("approved selection surface has no feasible candidates")
    return min(
        feasible,
        key=lambda candidate: (
            -_result(results[candidate.run_name]).validation_recall,
            -_result(results[candidate.run_name]).validation_ndcg,
            candidate.run_name,
        ),
    )


def _has_results(
    candidates: tuple[AggregateCandidate, ...], results: Mapping[str, CandidateOutcome]
) -> bool:
    return all(candidate.run_name in results for candidate in candidates)


def _require_results(
    candidates: tuple[AggregateCandidate, ...],
    results: Mapping[str, CandidateOutcome],
    prerequisite: str,
) -> None:
    missing = [candidate.run_name for candidate in candidates if candidate.run_name not in results]
    if missing:
        raise RuntimeError(
            f"{prerequisite} is incomplete: missing {len(missing)} verified run(s)"
        )


def _best_epoch_metrics(path: Path, best_epoch: int) -> tuple[float, float]:
    values: set[tuple[float, float]] = set()
    for line in path.read_text().splitlines():
        if re.search(rf"\bepoch {best_epoch - 1} finished\b", line) is None:
            continue
        recall = re.search(rf"\bepoch/val_true\.recall@100=({_METRIC_NUMBER})\b", line)
        ndcg = re.search(rf"\bepoch/val_true\.ndcg@100=({_METRIC_NUMBER})\b", line)
        if recall is not None and ndcg is not None:
            values.add((float(recall.group(1)), float(ndcg.group(1))))
    if len(values) != 1:
        raise ValueError(f"{path.parent.name}: missing or conflicting best-epoch metrics")
    return next(iter(values))


def _load_mapping(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _valid_scheduled_lr_traces(
    candidate: AggregateCandidate, metadata: Mapping[str, object]
) -> bool:
    horizon = candidate.horizon_epochs
    steps_per_epoch = metadata.get("optimizer_steps_per_epoch")
    if horizon is None or not _positive_int(steps_per_epoch):
        return False
    total_steps = steps_per_epoch * horizon
    if metadata.get("lr_schedule_horizon_steps") != total_steps:
        return False
    embedding_lr = metadata.get("embedding_learning_rate")
    deep_lr = metadata.get("deep_learning_rate")
    if not _same_rate(embedding_lr, candidate.embedding_lr) or not _same_rate(
        deep_lr, candidate.deep_lr
    ):
        return False
    traces = metadata.get("lr_group_traces")
    if not isinstance(traces, dict) or set(traces) != {"embedding", "deep"}:
        return False
    embedding_trace = traces["embedding"]
    deep_trace = traces["deep"]
    if not _valid_trace(embedding_trace, horizon) or not _valid_trace(
        deep_trace, horizon
    ):
        return False
    warmup_steps = int(total_steps * 0.05)
    decay_steps = max(1, total_steps - warmup_steps - 1)
    factors = []
    for epoch in range(1, horizon + 1):
        step = epoch * steps_per_epoch - 1
        if step < warmup_steps:
            factors.append((step + 1) / warmup_steps)
            continue
        progress = min(1.0, (step - warmup_steps) / decay_steps)
        factors.append(
            0.0
            if progress == 1
            else 0.5 * (1 + math.cos(math.pi * progress))
        )
    return all(
        _same_rate(actual, candidate.embedding_lr) for actual in embedding_trace
    ) and all(
        _same_rate(actual, candidate.deep_lr * factor)
        for actual, factor in zip(deep_trace, factors)
    )


def _valid_trace(value: object, horizon: int) -> bool:
    return (
        isinstance(value, list)
        and len(value) == horizon
        and all(
            isinstance(item, (int, float))
            and not isinstance(item, bool)
            and math.isfinite(item)
            and item >= 0
            for item in value
        )
    )


def _same_rate(actual: object, expected: float) -> bool:
    return (
        isinstance(actual, (int, float))
        and not isinstance(actual, bool)
        and math.isfinite(actual)
        and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12)
    )


def _result(outcome: CandidateOutcome) -> CandidateResult:
    if not isinstance(outcome, CandidateResult):
        raise ValueError("infeasible batch cells have no selection metrics")
    return outcome


def _validate_outcomes(results: Mapping[str, CandidateOutcome]) -> None:
    for run_name, outcome in results.items():
        if run_name != outcome.candidate.run_name:
            raise ValueError("candidate outcome key does not match its run identity")
        if isinstance(outcome, InfeasibleBatchCell) and not _is_batch_cell(
            outcome.candidate
        ):
            raise ValueError("only batch calibration accepts infeasible outcomes")


def _is_batch_cell(candidate: AggregateCandidate) -> bool:
    initial = candidate.stage == "batch_initial" and candidate.batch_size in {
        640,
        1280,
        2560,
    }
    boundary = candidate.stage == "batch_boundary" and candidate.batch_size in {
        160,
        320,
        480,
        3840,
        5120,
        7680,
    }
    return (
        candidate.family == "baseline"
        and (initial or boundary)
        and (candidate.embedding_lr, candidate.deep_lr) == (0.032, 0.012)
        and candidate.seed == 42
    )


def _has_recognized_cuda_oom(directory: Path) -> bool:
    patterns = (
        "torch.OutOfMemoryError: CUDA out of memory",
        "RuntimeError: CUDA out of memory",
    )
    paths = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix in {".log", ".txt", ".err"}
    )
    for path in paths:
        try:
            content = path.read_text(errors="replace")
        except OSError:
            continue
        if any(pattern in content for pattern in patterns):
            return True
    return False


def _archive_artifact(directory: Path, reason: Literal["incomplete", "infeasible"]) -> Path:
    logs = directory.parent
    if not directory.is_dir() or not directory.name.startswith("g1_"):
        raise ValueError("artifact archival requires one concrete G1 run directory")
    lock_path = logs / ".run-locks" / f"{directory.name}.lock"
    with _exclusive_lock(lock_path, f"artifact is owned by an active run: {directory}"):
        if not directory.is_dir():
            raise RuntimeError(f"artifact disappeared before archival: {directory}")
        return _move_to_archive(directory, reason)


@contextmanager
def _exclusive_lock(path: Path, message: str) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(message) from error
        yield


def _move_to_archive(
    directory: Path, reason: Literal["incomplete", "infeasible"]
) -> Path:
    archive_directory = directory.parent / "old"
    archive_directory.mkdir(parents=True, exist_ok=True)
    attempt = 1
    while True:
        archive = archive_directory / f"{directory.name}.{reason}-{attempt:03d}"
        if not archive.exists():
            break
        attempt += 1
    os.replace(directory, archive)
    return archive


def _infeasible_document(
    cells: Mapping[str, InfeasibleBatchCell],
) -> dict[str, object]:
    return {
        "version": 1,
        "dataset_size": "50m",
        "cells": [
            {
                "run_name": item.candidate.run_name,
                "reason": item.reason,
                "archive_path": item.archive_path,
            }
            for item in sorted(cells.values(), key=lambda item: item.candidate.run_name)
        ],
    }


def _write_json_atomic(path: Path, document: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    manifest_parser = subparsers.add_parser("manifest")
    manifest_parser.add_argument("--stage", required=True)
    manifest_parser.add_argument("--logs", type=Path, required=True)
    manifest_parser.add_argument("--depth", type=int)
    manifest_parser.add_argument("--infeasible-ledger", type=Path)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("directory", type=Path)
    verify_parser.add_argument("run_name")
    infeasible_parser = subparsers.add_parser("archive-infeasible")
    infeasible_parser.add_argument("directory", type=Path)
    infeasible_parser.add_argument("run_name")
    infeasible_parser.add_argument("ledger", type=Path)
    retry_parser = subparsers.add_parser("archive-retry")
    retry_parser.add_argument("directory", type=Path)
    arguments = parser.parse_args()
    if arguments.command == "verify":
        from experiments.g1_aggregate_dataset_size.protocol.candidates import (
            candidate_by_run,
        )

        candidate = candidate_by_run(arguments.run_name)
        raise SystemExit(0 if verify_candidate_artifact(arguments.directory, candidate) else 1)
    if arguments.command == "archive-infeasible":
        from experiments.g1_aggregate_dataset_size.protocol.candidates import (
            candidate_by_run,
        )

        candidate = candidate_by_run(arguments.run_name)
        if not _is_batch_cell(candidate) or not _has_recognized_cuda_oom(
            arguments.directory
        ):
            raise SystemExit(3)
        cell = archive_infeasible_batch_artifact(
            arguments.directory, candidate, arguments.ledger
        )
        print(cell.archive_path)
        return
    if arguments.command == "archive-retry":
        print(archive_retry_artifact(arguments.directory))
        return
    results = load_verified_results(arguments.logs, arguments.infeasible_ledger)
    candidates = stage_candidates(
        arguments.stage,
        results,
        aggregate_depth=arguments.depth,
    )
    for candidate in candidates:
        print(candidate.run_name)


if __name__ == "__main__":
    main()
