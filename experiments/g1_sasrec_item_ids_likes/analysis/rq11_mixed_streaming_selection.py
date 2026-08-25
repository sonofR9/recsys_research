from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re

from experiments.g1_sasrec_item_ids_likes.analysis.rq11_mixed_streaming_candidates import (
    BoundaryAxis,
    PRIMARY_FAMILIES,
    PrimaryFamily,
    Rq11Candidate,
    diagnostic_candidates,
    initial_candidates,
    local_lr_candidates,
    make_boundary_candidate,
)
from experiments.g1_sasrec_item_ids_likes.launchers import verify_artifact


_CONFIG = Path(__file__).parents[1] / "configs/rq11_mixed_streaming_variant.py"
_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"


class SelectionEvidenceError(ValueError):
    pass


@dataclass(frozen=True)
class ArtifactEvidence:
    candidate: Rq11Candidate
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
    stage: str
    candidates: tuple[Rq11Candidate, ...]


Inspector = Callable[[Rq11Candidate], ArtifactEvidence | None]


def select_family_winner(evidence: Iterable[ArtifactEvidence]) -> ArtifactEvidence:
    evidence = tuple(evidence)
    if not evidence:
        raise SelectionEvidenceError("cannot select an empty RQ11 family")
    families = {item.candidate.family for item in evidence}
    if len(families) != 1:
        raise SelectionEvidenceError("RQ11 family selection cannot mix families")
    return max(
        evidence,
        key=lambda item: (
            item.validation_recall,
            item.validation_ndcg,
            -item.candidate.negative_count,
            item.candidate.run_name,
        ),
    )


def build_followup_plan(inspect: Inspector) -> FollowupPlan:
    initial = initial_candidates()
    initial_evidence = _require_complete(inspect, initial, "RQ11 joint search")
    by_family = {
        family: [item for item in initial_evidence if item.candidate.family == family]
        for family in PRIMARY_FAMILIES
    }
    followups: list[Rq11Candidate] = []
    resolved_winners: dict[PrimaryFamily, ArtifactEvidence] = {}
    for family in PRIMARY_FAMILIES:
        family_evidence = list(by_family[family])
        winner = select_family_winner(family_evidence)
        local = local_lr_candidates(winner.candidate)
        missing = _missing(inspect, local)
        if missing:
            followups.extend(missing)
            continue
        family_evidence.extend(
            item
            for candidate in local
            if (item := _inspect_exact(inspect, candidate)) is not None
        )
        resolved, followup = _resolve_family_boundaries(
            family,
            select_family_winner(family_evidence),
            family_evidence,
            inspect,
        )
        if followup is not None:
            followups.extend(followup.candidates)
        else:
            resolved_winners[family] = resolved

    if followups:
        return _followup_plan(followups)

    mixture = resolved_winners["aggregate_uniform_streaming_global_q"]
    diagnostics = diagnostic_candidates(mixture.candidate)
    diagnostic_missing = _missing(inspect, diagnostics)
    if diagnostic_missing:
        return FollowupPlan("diagnostic", tuple(diagnostic_missing))
    return FollowupPlan("complete", ())


def _resolve_family_boundaries(
    family: PrimaryFamily,
    selected: ArtifactEvidence,
    evidence: list[ArtifactEvidence],
    inspect: Inspector,
) -> tuple[ArtifactEvidence, FollowupPlan | None]:
    axes: tuple[BoundaryAxis, ...] = (
        "negative_count",
        *(
            ("alpha",)
            if family in {"streaming_global_q", "aggregate_uniform_streaming_global_q"}
            else ()
        ),
        *(
            ("uniform_fraction",)
            if family == "aggregate_uniform_streaming_global_q"
            else ()
        ),
        "deep_lr",
    )
    anchors = {
        "negative_count": (512, 2048),
        "alpha": (0.005, 0.02),
        "uniform_fraction": (0.25, 0.75),
        "deep_lr": (0.006, 0.024),
    }
    for axis in axes:
        low, high = anchors[axis]
        coordinate = getattr(selected.candidate, axis)
        side = "low" if coordinate == low else "high" if coordinate == high else None
        if side is None:
            continue
        step = 1
        current = selected
        while True:
            probe = make_boundary_candidate(current.candidate, axis, side, step)
            probe_evidence = _inspect_exact(inspect, probe)
            if probe_evidence is None:
                return current, FollowupPlan("boundary", (probe,))
            if select_family_winner((current, probe_evidence)) != probe_evidence:
                break
            evidence.append(probe_evidence)
            current = probe_evidence
            if axis != "deep_lr":
                local = local_lr_candidates(current.candidate)
                missing = _missing(inspect, local)
                if missing:
                    return current, FollowupPlan("local_lr", tuple(missing))
                local_evidence = [
                    item
                    for candidate in local
                    if (item := _inspect_exact(inspect, candidate)) is not None
                ]
                evidence.extend(item for item in local_evidence if item not in evidence)
                current = select_family_winner(local_evidence)
            step += 1
        selected = select_family_winner(evidence)
    return selected, None


def classify_against_control(
    mixture: ArtifactEvidence, control: ArtifactEvidence
) -> str:
    recall_delta = mixture.final_recall - control.final_recall
    ndcg_delta = mixture.final_ndcg - control.final_ndcg
    if recall_delta > 0.003 and ndcg_delta < -0.001:
        return "trade-off"
    if recall_delta > 0.003 and ndcg_delta >= -0.001:
        return "better"
    if recall_delta < -0.003 or (abs(recall_delta) <= 0.003 and ndcg_delta < -0.001):
        return "worse"
    return "unresolved"


def filesystem_inspector(logs: Path) -> Inspector:
    def inspect(candidate: Rq11Candidate) -> ArtifactEvidence | None:
        directory = logs / candidate.run_name
        metadata_path = directory / "training_metadata.json"
        metrics_path = directory / "final_metrics.json"
        if not metadata_path.exists() or not metrics_path.exists():
            return None
        if not verify_artifact.verify_config(
            directory, _CONFIG, [f"G1_RQ11_RUN={candidate.run_name}"]
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
        recall, ndcg = _best_epoch_metrics(
            directory, metadata, candidate.horizon_epochs
        )
        try:
            final_recall = float(metrics["recall@100"])
            final_ndcg = float(metrics["ndcg@100"])
        except (KeyError, TypeError, ValueError) as error:
            raise SelectionEvidenceError(
                f"{candidate.run_name}: incomplete final metrics"
            ) from error
        return ArtifactEvidence(candidate, recall, ndcg, final_recall, final_ndcg)

    return inspect


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
        or stopped_epoch != expected_horizon
        or not 1 <= best_epoch <= stopped_epoch
    ):
        raise SelectionEvidenceError(
            f"{directory.name}: incomplete 20-epoch validation-selected horizon"
        )
    values = []
    for line in (directory / "sweep.log").read_text().splitlines():
        if re.search(rf"\bepoch {best_epoch - 1} finished\b", line) is None:
            continue
        recall = re.search(rf"\bepoch/val_true\.recall@100=({_NUMBER})\b", line)
        ndcg = re.search(rf"\bepoch/val_true\.ndcg@100=({_NUMBER})\b", line)
        if recall and ndcg:
            values.append((float(recall.group(1)), float(ndcg.group(1))))
    if len(set(values)) != 1:
        raise SelectionEvidenceError(
            f"{directory.name}: missing or conflicting best-epoch metrics"
        )
    return values[0]


def _require_complete(
    inspect: Inspector, candidates: Iterable[Rq11Candidate], label: str
) -> list[ArtifactEvidence]:
    candidates = tuple(candidates)
    evidence = [_inspect_exact(inspect, candidate) for candidate in candidates]
    if any(item is None for item in evidence):
        present = sum(item is not None for item in evidence)
        raise SelectionEvidenceError(
            f"{label} is incomplete: {present}/{len(candidates)} compatible artifacts"
        )
    return [item for item in evidence if item is not None]


def _missing(
    inspect: Inspector, candidates: Iterable[Rq11Candidate]
) -> list[Rq11Candidate]:
    return [
        candidate
        for candidate in candidates
        if _inspect_exact(inspect, candidate) is None
    ]


def _inspect_exact(
    inspect: Inspector, candidate: Rq11Candidate
) -> ArtifactEvidence | None:
    evidence = inspect(candidate)
    if evidence is not None and evidence.candidate != candidate:
        raise SelectionEvidenceError(
            f"{candidate.run_name}: inspector returned mismatched evidence"
        )
    return evidence


def _unique(candidates: Iterable[Rq11Candidate]) -> tuple[Rq11Candidate, ...]:
    return tuple({candidate.run_name: candidate for candidate in candidates}.values())


def _followup_plan(candidates: Iterable[Rq11Candidate]) -> FollowupPlan:
    candidates = _unique(candidates)
    stages = {candidate.stage for candidate in candidates}
    stage = stages.pop() if len(stages) == 1 else "mixed"
    return FollowupPlan(stage, candidates)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", type=Path, required=True)
    args = parser.parse_args()
    plan = build_followup_plan(filesystem_inspector(args.logs))
    for candidate in plan.candidates:
        print(candidate.run_name)


if __name__ == "__main__":
    main()
