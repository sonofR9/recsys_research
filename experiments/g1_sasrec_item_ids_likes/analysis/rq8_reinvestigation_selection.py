from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re

from experiments.g1_sasrec_item_ids_likes.analysis.rq8_reinvestigation_candidates import (
    BoundarySide,
    Rq8Candidate,
    initial_candidates,
    make_boundary_candidate,
    make_confirmation_candidate,
)
from experiments.g1_sasrec_item_ids_likes.launchers import verify_artifact


_CONFIG = Path(__file__).parents[1] / "configs/rq8_reinvestigation_variant.py"
_METRIC_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_FINAL_METRICS = frozenset(
    {"num_users"}
    | {
        f"{name}@{k}"
        for k in (10, 50, 100)
        for name in ("ndcg", "recall", "capped_recall", "mrr", "coverage")
    }
)


class SelectionEvidenceError(ValueError):
    pass


@dataclass(frozen=True)
class ArtifactEvidence:
    candidate: Rq8Candidate
    validation_recall: float
    validation_ndcg: float

    def __post_init__(self) -> None:
        metrics = (self.validation_recall, self.validation_ndcg)
        if any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or not 0 <= value <= 1
            for value in metrics
        ):
            raise SelectionEvidenceError(
                f"{self.candidate.run_name}: invalid validation metrics"
            )


@dataclass(frozen=True)
class FollowupPlan:
    boundary: tuple[Rq8Candidate, ...]
    confirmations: tuple[Rq8Candidate, ...]

    @property
    def candidates(self) -> tuple[Rq8Candidate, ...]:
        return (*self.boundary, *self.confirmations)


Inspector = Callable[[Rq8Candidate], ArtifactEvidence | None]


def build_followup_plan(inspect: Inspector) -> FollowupPlan:
    initial = initial_candidates()
    initial_evidence = {
        candidate.run_name: _inspect_exact(inspect, candidate) for candidate in initial
    }
    missing = [name for name, evidence in initial_evidence.items() if evidence is None]
    if missing:
        raise SelectionEvidenceError(
            f"RQ8 initial surface is incomplete: {len(initial) - len(missing)}/"
            f"{len(initial)} compatible artifacts"
        )

    grouped: dict[tuple[str, str, int], list[ArtifactEvidence]] = {}
    for candidate in initial:
        evidence = initial_evidence[candidate.run_name]
        if evidence is None:
            raise AssertionError("checked above")
        grouped.setdefault(candidate.surface_key, []).append(evidence)

    boundary: list[Rq8Candidate] = []
    winners: dict[tuple[str, str, int], Rq8Candidate] = {}
    for key, evidence in grouped.items():
        winner, next_probe = _resolve_surface(evidence, inspect)
        if next_probe is not None:
            boundary.append(next_probe)
        elif winner is not None:
            winners[key] = winner

    query_keys = [key for key in grouped if key[0] == "query"]
    confirmations: list[Rq8Candidate] = []
    if all(key in winners for key in query_keys):
        for key in query_keys:
            for seed in (43, 44):
                candidate = make_confirmation_candidate(winners[key], seed)
                if _inspect_exact(inspect, candidate) is None:
                    confirmations.append(candidate)
    return FollowupPlan(tuple(boundary), tuple(confirmations))


def filesystem_inspector(logs: Path) -> Inspector:
    def inspect(candidate: Rq8Candidate) -> ArtifactEvidence | None:
        directory = logs / candidate.run_name
        metadata_path = directory / "training_metadata.json"
        metrics_path = directory / "final_metrics.json"
        if not metadata_path.exists() or not metrics_path.exists():
            return None
        assignments = [f"G1_RQ8_RUN={candidate.run_name}"]
        if not verify_artifact.verify_config(directory, _CONFIG, assignments):
            raise SelectionEvidenceError(
                f"{candidate.run_name}: protocol-incompatible artifact"
            )
        try:
            metadata = json.loads(metadata_path.read_text())
            final_metrics = json.loads(metrics_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise SelectionEvidenceError(
                f"{candidate.run_name}: unreadable final artifact"
            ) from error
        _require_final_metrics(candidate, final_metrics)
        recall, ndcg = _best_epoch_metrics(
            directory, metadata, candidate.cap_epochs
        )
        return ArtifactEvidence(candidate, recall, ndcg)

    return inspect


def _resolve_surface(
    initial: list[ArtifactEvidence], inspect: Inspector
) -> tuple[Rq8Candidate | None, Rq8Candidate | None]:
    evidence = list(initial)
    while True:
        winner = _select_winner(evidence)
        rates = sorted(item.candidate.deep_lr for item in evidence)
        side: BoundarySide | None = None
        if winner.candidate.deep_lr == rates[0]:
            side = "low"
        elif winner.candidate.deep_lr == rates[-1]:
            side = "high"
        if side is None:
            return winner.candidate, None
        next_probe = _next_boundary_candidate(winner.candidate, side, evidence)
        next_evidence = _inspect_exact(inspect, next_probe)
        if next_evidence is None:
            return None, next_probe
        evidence.append(next_evidence)


def _select_winner(evidence: list[ArtifactEvidence]) -> ArtifactEvidence:
    ordered = sorted(
        evidence,
        key=lambda item: (item.validation_recall, item.validation_ndcg),
        reverse=True,
    )
    if len(ordered) > 1 and (
        ordered[0].validation_recall,
        ordered[0].validation_ndcg,
    ) == (
        ordered[1].validation_recall,
        ordered[1].validation_ndcg,
    ):
        raise SelectionEvidenceError(
            f"{ordered[0].candidate.surface_key}: exact validation tie after "
            "recall@100 and NDCG@100"
        )
    return ordered[0]


def _next_boundary_candidate(
    surface: Rq8Candidate,
    side: BoundarySide,
    evidence: list[ArtifactEvidence],
) -> Rq8Candidate:
    steps = [
        item.candidate.boundary_step
        for item in evidence
        if item.candidate.stage == "boundary"
        and item.candidate.boundary_side == side
    ]
    step = max((value for value in steps if value is not None), default=0) + 1
    return make_boundary_candidate(surface, side, step)


def _inspect_exact(
    inspect: Inspector, candidate: Rq8Candidate
) -> ArtifactEvidence | None:
    evidence = inspect(candidate)
    if evidence is not None and evidence.candidate != candidate:
        raise SelectionEvidenceError(
            f"{candidate.run_name}: inspector returned evidence for "
            f"{evidence.candidate.run_name}"
        )
    return evidence


def _best_epoch_metrics(
    directory: Path, metadata: object, expected_horizon: int
) -> tuple[float, float]:
    if not isinstance(metadata, dict):
        raise SelectionEvidenceError(f"{directory.name}: invalid training metadata")
    best_epoch = metadata.get("best_epoch")
    stopped_epoch = metadata.get("stopped_epoch")
    if (
        not isinstance(best_epoch, int)
        or isinstance(best_epoch, bool)
        or not isinstance(stopped_epoch, int)
        or isinstance(stopped_epoch, bool)
        or not 1 <= best_epoch <= stopped_epoch
    ):
        raise SelectionEvidenceError(
            f"{directory.name}: invalid best-epoch selection evidence"
        )
    if stopped_epoch != expected_horizon:
        raise SelectionEvidenceError(
            f"{directory.name}: did not finish the declared "
            f"{expected_horizon}-epoch annealing horizon"
        )
    try:
        lines = (directory / "sweep.log").read_text().splitlines()
    except OSError as error:
        raise SelectionEvidenceError(
            f"{directory.name}: missing sweep.log for validation selection"
        ) from error
    epoch_index = best_epoch - 1
    values: list[tuple[float, float]] = []
    for line in lines:
        if re.search(rf"\bepoch {epoch_index} finished\b", line) is None:
            continue
        recall = re.search(
            rf"\bepoch/val_true\.recall@100=({_METRIC_NUMBER})\b", line
        )
        ndcg = re.search(
            rf"\bepoch/val_true\.ndcg@100=({_METRIC_NUMBER})\b", line
        )
        if recall is not None and ndcg is not None:
            values.append((float(recall.group(1)), float(ndcg.group(1))))
    unique = set(values)
    if len(unique) != 1:
        raise SelectionEvidenceError(
            f"{directory.name}: best epoch has missing or conflicting validation metrics"
        )
    return unique.pop()


def _require_final_metrics(
    candidate: Rq8Candidate, metrics: object
) -> None:
    if not isinstance(metrics, dict) or set(metrics) != _FINAL_METRICS:
        raise SelectionEvidenceError(
            f"{candidate.run_name}: incomplete final metrics"
        )
    for name, value in metrics.items():
        maximum = math.inf if name == "num_users" else 1
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or not 0 <= value <= maximum
            or (name == "num_users" and value < 1)
        ):
            raise SelectionEvidenceError(
                f"{candidate.run_name}: invalid final metric {name}={value!r}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", type=Path, required=True)
    args = parser.parse_args()
    plan = build_followup_plan(filesystem_inspector(args.logs))
    for candidate in plan.candidates:
        print(f"{candidate.run_name}\t{candidate.max_seq_len}")


if __name__ == "__main__":
    main()
