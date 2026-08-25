from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re

from experiments.g1_sasrec_item_ids_likes.analysis.rq7_reinvestigation_candidates import (
    BoundarySide,
    Rq7Candidate,
    diagnostic_candidates,
    initial_candidates,
    make_boundary_candidate,
    make_confirmation_candidate,
    make_rope_base_extension_candidates,
    rope_base_candidates,
)
from experiments.g1_sasrec_item_ids_likes.launchers import verify_artifact


_CONFIG = Path(__file__).parents[1] / "configs/rq7_reinvestigation_variant.py"
_METRIC_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_FINAL_METRICS = frozenset(
    {"num_users"}
    | {
        f"{name}@{k}"
        for k in (10, 50, 100)
        for name in ("ndcg", "recall", "capped_recall", "mrr", "coverage")
    }
)
_RECALL_BAND = 0.003
_NDCG_BAND = 0.001


class SelectionEvidenceError(ValueError):
    pass


@dataclass(frozen=True)
class ArtifactEvidence:
    candidate: Rq7Candidate
    validation_recall: float
    validation_ndcg: float
    final_recall: float
    final_ndcg: float

    def __post_init__(self) -> None:
        values = (
            self.validation_recall,
            self.validation_ndcg,
            self.final_recall,
            self.final_ndcg,
        )
        if any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or not 0 <= value <= 1
            for value in values
        ):
            raise SelectionEvidenceError(
                f"{self.candidate.run_name}: invalid selection metrics"
            )


@dataclass(frozen=True)
class FollowupPlan:
    boundary: tuple[Rq7Candidate, ...]
    rope_base: tuple[Rq7Candidate, ...]
    confirmations: tuple[Rq7Candidate, ...]

    @property
    def candidates(self) -> tuple[Rq7Candidate, ...]:
        return (*self.boundary, *self.rope_base, *self.confirmations)


Inspector = Callable[[Rq7Candidate], ArtifactEvidence | None]


def build_followup_plan(inspect: Inspector) -> FollowupPlan:
    initial = initial_candidates()
    evidence = _require_complete(inspect, initial, "RQ7 initial surface")
    winners, boundary = _resolve_treatments(evidence, inspect)
    if boundary:
        return FollowupPlan(tuple(boundary), (), ())

    alibi = winners["alibi"]
    plain = winners["rope_forward_base10000"]
    selected_plain = plain
    if _materially_worse(plain, alibi):
        lower = rope_base_candidates()
        missing = _missing(inspect, lower)
        if missing:
            return FollowupPlan((), tuple(missing), ())
        lower_winners, boundary = _resolve_treatments(
            _require_complete(inspect, lower, "RQ7 RoPE-base surface"), inspect
        )
        if boundary:
            return FollowupPlan(tuple(boundary), (), ())
        base_winners = {
            plain.candidate.treatment: plain,
            **lower_winners,
        }
        axis_winner = _select_winner(list(base_winners.values()))
        side = {
            "rope_forward_base100": "low",
            "rope_forward_base10000": "high",
        }.get(axis_winner.candidate.treatment)
        if side is not None:
            outer = make_rope_base_extension_candidates(side)
            missing = _missing(inspect, outer)
            if missing:
                return FollowupPlan((), tuple(missing), ())
            outer_winners, boundary = _resolve_treatments(
                _require_complete(inspect, outer, "RQ7 outer RoPE-base surface"),
                inspect,
            )
            if boundary:
                return FollowupPlan(tuple(boundary), (), ())
            base_winners.update(outer_winners)
        selected_plain = _select_winner(list(base_winners.values()))

    confirmations: list[Rq7Candidate] = []
    if not _materially_better(selected_plain, alibi):
        selected = (
            alibi.candidate,
            selected_plain.candidate,
            winners["rope_forward_base10000_alibi"].candidate,
        )
        for winner in selected:
            for seed in (43, 44):
                candidate = make_confirmation_candidate(winner, seed)
                if _inspect_exact(inspect, candidate) is None:
                    confirmations.append(candidate)
    return FollowupPlan((), (), tuple(confirmations))


def require_diagnostic_gate(inspect: Inspector) -> None:
    evidence = _require_complete(
        inspect, diagnostic_candidates(), "RQ7 native-50M diagnostic surface"
    )
    by_treatment = {item.candidate.treatment: item for item in evidence}
    references = {
        "learned_forward_reverse_add": "learned_forward_add",
        "learned_forward_reverse_concat": "learned_forward_concat",
        "learned_forward_reverse_add_alibi": "learned_forward_add_alibi",
        "learned_forward_reverse_concat_alibi": "learned_forward_concat_alibi",
    }
    failed = [
        treatment
        for treatment, reference in references.items()
        if (
            _strictly_below(
                by_treatment[treatment].final_recall
                - by_treatment[reference].final_recall,
                -_RECALL_BAND,
            )
            or _strictly_below(
                by_treatment[treatment].final_ndcg - by_treatment[reference].final_ndcg,
                -_NDCG_BAND,
            )
        )
    ]
    if failed:
        raise SelectionEvidenceError(
            "RQ7 bounded-reverse diagnostic gate failed: " + ", ".join(failed)
        )


def filesystem_inspector(logs: Path) -> Inspector:
    def inspect(candidate: Rq7Candidate) -> ArtifactEvidence | None:
        directory = logs / candidate.run_name
        metadata_path = directory / "training_metadata.json"
        metrics_path = directory / "final_metrics.json"
        if not metadata_path.exists() or not metrics_path.exists():
            return None
        if not verify_artifact.verify_config(
            directory, _CONFIG, [f"G1_RQ7_RUN={candidate.run_name}"]
        ):
            raise SelectionEvidenceError(
                f"{candidate.run_name}: protocol-incompatible artifact"
            )
        try:
            metadata = json.loads(metadata_path.read_text())
            metrics = json.loads(metrics_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise SelectionEvidenceError(
                f"{candidate.run_name}: unreadable final artifact"
            ) from error
        _require_final_metrics(candidate, metrics)
        recall, ndcg = _best_epoch_metrics(
            directory, metadata, candidate.horizon_epochs
        )
        return ArtifactEvidence(
            candidate, recall, ndcg, metrics["recall@100"], metrics["ndcg@100"]
        )

    return inspect


def _require_complete(
    inspect: Inspector, candidates: Iterable[Rq7Candidate], label: str
) -> list[ArtifactEvidence]:
    candidates = tuple(candidates)
    evidence = [_inspect_exact(inspect, candidate) for candidate in candidates]
    missing = sum(item is None for item in evidence)
    if missing:
        raise SelectionEvidenceError(
            f"{label} is incomplete: {len(candidates) - missing}/{len(candidates)} "
            "compatible artifacts"
        )
    return [item for item in evidence if item is not None]


def _missing(
    inspect: Inspector, candidates: Iterable[Rq7Candidate]
) -> list[Rq7Candidate]:
    return [
        candidate
        for candidate in candidates
        if _inspect_exact(inspect, candidate) is None
    ]


def _resolve_treatments(
    evidence: list[ArtifactEvidence], inspect: Inspector
) -> tuple[dict[str, ArtifactEvidence], list[Rq7Candidate]]:
    grouped: dict[str, list[ArtifactEvidence]] = {}
    for item in evidence:
        grouped.setdefault(item.candidate.treatment, []).append(item)
    winners: dict[str, ArtifactEvidence] = {}
    boundary: list[Rq7Candidate] = []
    for treatment, surface in grouped.items():
        winner, probe = _resolve_surface(surface, inspect)
        if probe is not None:
            boundary.append(probe)
        elif winner is not None:
            winners[treatment] = winner
    return winners, boundary


def _resolve_surface(
    evidence: list[ArtifactEvidence], inspect: Inspector
) -> tuple[ArtifactEvidence | None, Rq7Candidate | None]:
    evidence = list(evidence)
    while True:
        winner = _select_winner(evidence)
        rates = sorted(item.candidate.deep_lr for item in evidence)
        side: BoundarySide | None = None
        if winner.candidate.deep_lr == rates[0]:
            side = "low"
        elif winner.candidate.deep_lr == rates[-1]:
            side = "high"
        if side is None:
            return winner, None
        probe = _next_boundary_candidate(winner.candidate, side, evidence)
        next_evidence = _inspect_exact(inspect, probe)
        if next_evidence is None:
            return None, probe
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
    ) == (ordered[1].validation_recall, ordered[1].validation_ndcg):
        raise SelectionEvidenceError(
            f"{ordered[0].candidate.treatment}: exact validation tie after "
            "recall@100 and NDCG@100"
        )
    return ordered[0]


def _next_boundary_candidate(
    surface: Rq7Candidate,
    side: BoundarySide,
    evidence: list[ArtifactEvidence],
) -> Rq7Candidate:
    steps = [
        item.candidate.boundary_step
        for item in evidence
        if item.candidate.stage == "boundary" and item.candidate.boundary_side == side
    ]
    step = max((value for value in steps if value is not None), default=0) + 1
    return make_boundary_candidate(surface, side, step)


def _materially_worse(candidate: ArtifactEvidence, reference: ArtifactEvidence) -> bool:
    recall_delta = candidate.final_recall - reference.final_recall
    if _strictly_below(recall_delta, -_RECALL_BAND):
        return True
    return recall_delta <= _RECALL_BAND and (
        _strictly_below(candidate.final_ndcg - reference.final_ndcg, -_NDCG_BAND)
    )


def _materially_better(
    candidate: ArtifactEvidence, reference: ArtifactEvidence
) -> bool:
    recall_delta = candidate.final_recall - reference.final_recall
    if _strictly_above(recall_delta, _RECALL_BAND):
        return True
    return recall_delta >= -_RECALL_BAND and (
        _strictly_above(candidate.final_ndcg - reference.final_ndcg, _NDCG_BAND)
    )


def _strictly_below(value: float, threshold: float) -> bool:
    return value < threshold and not math.isclose(value, threshold, abs_tol=1e-12)


def _strictly_above(value: float, threshold: float) -> bool:
    return value > threshold and not math.isclose(value, threshold, abs_tol=1e-12)


def _inspect_exact(
    inspect: Inspector, candidate: Rq7Candidate
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
        recall = re.search(rf"\bepoch/val_true\.recall@100=({_METRIC_NUMBER})\b", line)
        ndcg = re.search(rf"\bepoch/val_true\.ndcg@100=({_METRIC_NUMBER})\b", line)
        if recall is not None and ndcg is not None:
            values.append((float(recall.group(1)), float(ndcg.group(1))))
    unique = set(values)
    if len(unique) != 1:
        raise SelectionEvidenceError(
            f"{directory.name}: best epoch has missing or conflicting validation metrics"
        )
    return unique.pop()


def _require_final_metrics(candidate: Rq7Candidate, metrics: object) -> None:
    if not isinstance(metrics, dict) or set(metrics) != _FINAL_METRICS:
        raise SelectionEvidenceError(f"{candidate.run_name}: incomplete final metrics")
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
    parser.add_argument("--require-diagnostic-gate", action="store_true")
    args = parser.parse_args()
    if args.require_diagnostic_gate:
        require_diagnostic_gate(filesystem_inspector(args.logs))
        return
    for candidate in build_followup_plan(filesystem_inspector(args.logs)).candidates:
        print(candidate.run_name)


if __name__ == "__main__":
    main()
