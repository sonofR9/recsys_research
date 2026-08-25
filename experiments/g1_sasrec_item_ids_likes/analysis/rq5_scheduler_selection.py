from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import tempfile
from typing import Callable

from experiments.g1_sasrec_item_ids_likes.analysis.rq5_scheduler_candidates import (
    Rq5Candidate,
    candidate_by_run,
    initial_candidates,
)
from experiments.g1_sasrec_item_ids_likes.analysis.rq5_scheduler_corrections import (
    ArtifactEvidence,
    CorrectionEvidenceError,
    HorizonFollowupApproval,
    HorizonObservation,
    filesystem_inspector,
    load_horizon_followup_approval,
    recompute_correction,
    require_canonical_horizon_followup_approval,
)


_INITIAL_DEEP_LRS = (0.003, 0.006, 0.012)
_JOINT_TREATMENTS = frozenset({"inverse_sqrt", "cosine_warmup_tuned"})
_JOINT_DEEP_BOUNDS = (0.0015, 0.024)
_JOINT_FRACTION_BOUNDS = (0.0125, 0.20)
_METRIC_NUMBER = r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?"
_REPO_ROOT = Path(__file__).parents[3]
DEFAULT_MANIFEST = (
    Path(__file__).parents[1] / "scratchpad" / "rq5_scheduler_candidate_manifest.json"
)
APPROVAL_RECORD = Path(__file__).parents[1] / "protocol" / "rq5_scheduler_approval.json"


class SelectionEvidenceError(RuntimeError):
    pass


class ProbeApprovalRequired(SelectionEvidenceError):
    pass


class SelectionPolicyApprovalRequired(SelectionEvidenceError):
    pass


class FinalSelectionIncomplete(SelectionEvidenceError):
    pass


class BoundaryOutcomeApprovalRequired(SelectionEvidenceError):
    def __init__(self, candidates: tuple[Rq5Candidate, ...]) -> None:
        self.candidates = candidates
        names = ", ".join(candidate.run_name for candidate in candidates)
        super().__init__(
            f"RQ5 outer-boundary winner requires explicit user approval: {names}"
        )


@dataclass(frozen=True)
class ProbePolicyApproval:
    reference: str
    local_factor: float = 2.0
    boundary_extension: float = 4.0
    boundary_points: int = 3

    def __post_init__(self) -> None:
        if not self.reference.strip():
            raise ValueError("probe approval needs a non-empty reference")
        if (
            self.local_factor != 2.0
            or self.boundary_extension != 4.0
            or self.boundary_points != 3
        ):
            raise ValueError("probe approval does not match the pending RQ5 policy")


@dataclass(frozen=True)
class SelectionPolicyApproval:
    reference: str
    metric_decimals: int = 4
    primary: str = "best_epoch_validation_recall@100"
    secondary: str = "same_epoch_validation_ndcg@100"
    tie_breaker: str = "surface_run_name"

    def __post_init__(self) -> None:
        if not self.reference.strip():
            raise ValueError("selection approval needs a non-empty reference")
        if (
            self.metric_decimals != 4
            or self.primary != "best_epoch_validation_recall@100"
            or self.secondary != "same_epoch_validation_ndcg@100"
            or self.tie_breaker != "surface_run_name"
        ):
            raise ValueError("selection approval does not match the pending RQ5 policy")


@dataclass(frozen=True)
class CandidateManifest:
    candidates: tuple[Rq5Candidate, ...]
    approval: ProbePolicyApproval | None = None
    selection_approval: SelectionPolicyApproval | None = None
    horizon_followup_approval: HorizonFollowupApproval | None = None
    version: int = 1

    def __post_init__(self) -> None:
        if (
            not isinstance(self.version, int)
            or isinstance(self.version, bool)
            or self.version != 1
        ):
            raise ValueError(
                f"unsupported RQ5 candidate manifest version {self.version!r}"
            )
        names = [candidate.run_name for candidate in self.candidates]
        if names != sorted(names) or len(names) != len(set(names)):
            raise ValueError("manifest candidates must be sorted and unique")
        if any(candidate.attempt != 0 for candidate in self.candidates):
            raise ValueError("manifest contains a correction attempt, not a surface")
        if any(
            candidate.probe is not None and len(candidate.treatments) != 1
            for candidate in self.candidates
        ):
            raise ValueError("manifest probe belongs to more than one treatment")
        initial_names = {candidate.run_name for candidate in initial_candidates()}
        if not initial_names <= set(names):
            raise ValueError("manifest omits an initial RQ5 candidate")

    @property
    def digest(self) -> str:
        return hashlib.sha256(_canonical_json(self.payload()).encode()).hexdigest()

    def payload(self) -> dict:
        return {
            "version": self.version,
            "approval": None if self.approval is None else asdict(self.approval),
            "selection_approval": (
                None
                if self.selection_approval is None
                else asdict(self.selection_approval)
            ),
            "horizon_followup_approval": (
                None
                if self.horizon_followup_approval is None
                else asdict(self.horizon_followup_approval)
            ),
            "candidates": [
                {"run_name": candidate.run_name, **asdict(candidate)}
                for candidate in self.candidates
            ],
        }

    def freeze(self) -> str:
        return _canonical_json({**self.payload(), "digest": self.digest}) + "\n"

    @classmethod
    def thaw(cls, raw: str) -> CandidateManifest:
        document = json.loads(raw)
        if not isinstance(document, dict) or set(document) != {
            "version",
            "approval",
            "selection_approval",
            "horizon_followup_approval",
            "candidates",
            "digest",
        }:
            raise ValueError("invalid RQ5 candidate manifest document")
        digest = document.pop("digest", None)
        if digest != hashlib.sha256(_canonical_json(document).encode()).hexdigest():
            raise ValueError("RQ5 candidate manifest digest mismatch")
        approval = document.get("approval")
        selection_approval = document.get("selection_approval")
        horizon_followup_approval = document.get("horizon_followup_approval")
        candidates = tuple(
            candidate_by_run(record["run_name"])
            for record in document.get("candidates", ())
        )
        manifest = cls(
            candidates=tuple(
                sorted(candidates, key=lambda candidate: candidate.run_name)
            ),
            approval=(None if approval is None else ProbePolicyApproval(**approval)),
            selection_approval=(
                None
                if selection_approval is None
                else SelectionPolicyApproval(**selection_approval)
            ),
            horizon_followup_approval=(
                None
                if horizon_followup_approval is None
                else HorizonFollowupApproval(**horizon_followup_approval)
            ),
            version=document.get("version"),
        )
        if json.loads(_canonical_json(manifest.payload())) != document:
            raise ValueError("RQ5 candidate manifest is non-canonical")
        return manifest

    def extend(
        self,
        candidates: tuple[Rq5Candidate, ...],
        approval: ProbePolicyApproval,
    ) -> CandidateManifest:
        if self.approval is not None and self.approval != approval:
            raise ValueError("RQ5 probe approval changed after manifest freeze")
        by_name = {candidate.run_name: candidate for candidate in self.candidates}
        for candidate in candidates:
            existing = by_name.get(candidate.run_name)
            if existing is not None and existing != candidate:
                raise ValueError("RQ5 manifest candidate identity collision")
            by_name[candidate.run_name] = candidate
        return CandidateManifest(
            candidates=tuple(
                sorted(by_name.values(), key=lambda candidate: candidate.run_name)
            ),
            approval=approval,
            selection_approval=self.selection_approval,
            horizon_followup_approval=self.horizon_followup_approval,
        )


@dataclass(frozen=True, order=True)
class TreatmentSlot:
    treatment: str
    scope: str


@dataclass(frozen=True)
class CorrectionAttemptEvidence:
    candidate: Rq5Candidate
    metadata: dict
    metrics: dict | None
    selection_metrics: dict | None
    strictly_eligible: bool
    optimizer_group_traces_verified: bool
    calibration_status: str
    terminal_state: str | None = None


@dataclass(frozen=True)
class LedgerEntry:
    slot: TreatmentSlot
    initial: Rq5Candidate
    current: Rq5Candidate
    metrics: dict[str, float] | None
    selection_metrics: dict[str, float] | None
    exhausted: bool = False
    ineligible_exclusion: bool = False
    correction_chain: tuple[CorrectionAttemptEvidence, ...] = ()


@dataclass(frozen=True)
class CalibratedLedger:
    entries: tuple[LedgerEntry, ...]

    @property
    def slots(self) -> tuple[TreatmentSlot, ...]:
        return tuple(sorted({entry.slot for entry in self.entries}))

    def for_slot(self, slot: TreatmentSlot) -> tuple[LedgerEntry, ...]:
        return tuple(entry for entry in self.entries if entry.slot == slot)


@dataclass(frozen=True)
class ProbePlan:
    boundary: tuple[Rq5Candidate, ...]
    local: tuple[Rq5Candidate, ...]
    unresolved: tuple[Rq5Candidate, ...] = ()

    @property
    def candidates(self) -> tuple[Rq5Candidate, ...]:
        return (*self.boundary, *self.local)


def _load_approval_record() -> dict:
    approval = json.loads(APPROVAL_RECORD.read_text())
    if not isinstance(approval, dict) or set(approval) != {
        "probe_policy",
        "selection_policy",
        "horizon_followup",
    }:
        raise ValueError("invalid RQ5 approval record")
    return approval


def initial_manifest() -> CandidateManifest:
    approval = _load_approval_record()
    return CandidateManifest(
        candidates=tuple(
            sorted(initial_candidates(), key=lambda candidate: candidate.run_name)
        ),
        approval=ProbePolicyApproval(**approval["probe_policy"]),
        selection_approval=SelectionPolicyApproval(**approval["selection_policy"]),
        horizon_followup_approval=HorizonFollowupApproval(
            **approval["horizon_followup"]
        ),
    )


def _require_canonical_horizon_followup_approval(
    approval: HorizonFollowupApproval | None,
) -> HorizonFollowupApproval:
    if approval is None:
        raise SelectionEvidenceError(
            "RQ5 manifest lacks approved bracketed horizon follow-up"
        )
    try:
        return require_canonical_horizon_followup_approval(approval)
    except CorrectionEvidenceError as error:
        raise SelectionEvidenceError(str(error)) from error


def _require_canonical_probe_approval(
    approval: ProbePolicyApproval | None,
) -> ProbePolicyApproval:
    if approval is None:
        raise ProbeApprovalRequired(
            "RQ5 local and boundary probe policy is pending explicit approval"
        )
    canonical = ProbePolicyApproval(**_load_approval_record()["probe_policy"])
    if approval != canonical:
        raise SelectionEvidenceError(
            "RQ5 supplied approval does not equal the canonical probe approval"
        )
    return approval


def _require_canonical_selection_approval(
    approval: SelectionPolicyApproval | None,
) -> SelectionPolicyApproval:
    if approval is None:
        raise SelectionPolicyApprovalRequired(
            "RQ5 four-decimal validation selection policy is pending explicit approval"
        )
    canonical = SelectionPolicyApproval(**_load_approval_record()["selection_policy"])
    if approval != canonical:
        raise SelectionEvidenceError(
            "RQ5 supplied approval does not equal the canonical selection approval"
        )
    return approval


def build_calibrated_ledger(
    inspect: Callable[[Rq5Candidate], ArtifactEvidence],
    surfaces: tuple[Rq5Candidate, ...] | None = None,
    horizon_followup_approval: HorizonFollowupApproval | None = None,
) -> CalibratedLedger:
    surfaces = initial_candidates() if surfaces is None else surfaces
    if horizon_followup_approval is None:
        horizon_followup_approval = load_horizon_followup_approval()
    else:
        horizon_followup_approval = _require_canonical_horizon_followup_approval(
            horizon_followup_approval
        )
    resolved = {
        surface.run_name: _resolve_chain(surface, inspect, horizon_followup_approval)
        for surface in surfaces
    }
    entries = []
    for surface in surfaces:
        current, evidence, exhausted, correction_chain = resolved[surface.run_name]
        ineligible_exclusion = (
            surface.run_name in horizon_followup_approval.ineligible_initial_surfaces
        )
        if ineligible_exclusion and not exhausted:
            raise SelectionEvidenceError(
                f"{surface.run_name}: approved ineligible exclusion resolved as calibrated"
            )
        for treatment in surface.treatments:
            entries.append(
                LedgerEntry(
                    TreatmentSlot(treatment, surface.scope),
                    surface,
                    current,
                    evidence.metrics,
                    evidence.selection_metrics,
                    exhausted,
                    ineligible_exclusion,
                    correction_chain,
                )
            )
    ledger = CalibratedLedger(tuple(entries))
    if len(ledger.entries) < 69 or len(ledger.slots) != 23:
        raise SelectionEvidenceError(
            "RQ5 ledger has fewer than 69 entries across 23 treatments"
        )
    if any(len(ledger.for_slot(slot)) < 3 for slot in ledger.slots):
        raise SelectionEvidenceError("an RQ5 treatment has fewer than three candidates")
    current_names = {entry.current.run_name for entry in ledger.entries}
    if len(current_names) != len(surfaces):
        raise SelectionEvidenceError(
            "RQ5 ledger does not resolve to one artifact per candidate surface"
        )
    _validate_shared_central_provenance(ledger)
    return ledger


def _resolve_chain(
    initial: Rq5Candidate,
    inspect: Callable[[Rq5Candidate], ArtifactEvidence],
    horizon_followup_approval: HorizonFollowupApproval,
) -> tuple[
    Rq5Candidate,
    ArtifactEvidence,
    bool,
    tuple[CorrectionAttemptEvidence, ...],
]:
    current = initial
    observations: list[HorizonObservation] = []
    correction_chain: list[CorrectionAttemptEvidence] = []
    while True:
        evidence = inspect(current)
        if evidence.kind != "complete" or evidence.metadata is None:
            raise SelectionEvidenceError(
                f"{current.run_name}: required correction evidence is {evidence.kind}"
            )
        try:
            decision = recompute_correction(
                current,
                evidence.metadata,
                prior_observations=tuple(observations),
                horizon_followup_approval=horizon_followup_approval,
                initial_surface=initial,
            )
        except CorrectionEvidenceError as error:
            raise SelectionEvidenceError(str(error)) from error
        if decision.status == "calibrated":
            if not evidence.strictly_eligible:
                raise SelectionEvidenceError(
                    f"{current.run_name}: current candidate is not strictly eligible"
                )
            _ranking_metrics(current, evidence.metrics, "full-user final")
            _ranking_metrics(
                current, evidence.selection_metrics, "best-epoch validation"
            )
            correction_chain.append(
                _correction_attempt(current, evidence, decision.status, "calibrated")
            )
            return current, evidence, False, tuple(correction_chain)
        if evidence.strictly_eligible:
            raise SelectionEvidenceError(
                f"{current.run_name}: unresolved evidence passed strict verification"
            )
        if current.horizon_epochs is not None:
            observations.append(
                HorizonObservation(current.horizon_epochs, decision.status)
            )
        if decision.exhausted:
            correction_chain.append(
                _correction_attempt(current, evidence, decision.status, "exhausted")
            )
            return current, evidence, True, tuple(correction_chain)
        correction_chain.append(
            _correction_attempt(current, evidence, decision.status, None)
        )
        assert decision.target is not None
        next_evidence = inspect(decision.target)
        if next_evidence.kind != "complete":
            raise SelectionEvidenceError(
                f"{decision.target.run_name}: required correction is "
                f"{next_evidence.kind}"
            )
        current = decision.target


def _correction_attempt(
    candidate: Rq5Candidate,
    evidence: ArtifactEvidence,
    calibration_status: str,
    terminal_state: str | None,
) -> CorrectionAttemptEvidence:
    if evidence.metadata is None:
        raise SelectionEvidenceError(
            f"{candidate.run_name}: correction attempt lacks metadata"
        )
    return CorrectionAttemptEvidence(
        candidate=candidate,
        metadata=evidence.metadata,
        metrics=evidence.metrics,
        selection_metrics=evidence.selection_metrics,
        strictly_eligible=evidence.strictly_eligible,
        optimizer_group_traces_verified=evidence.optimizer_group_traces_verified,
        calibration_status=calibration_status,
        terminal_state=terminal_state,
    )


def _validate_shared_central_provenance(ledger: CalibratedLedger) -> None:
    shared = [entry for entry in ledger.entries if len(entry.initial.treatments) == 2]
    if len(shared) != 4:
        raise SelectionEvidenceError(
            "shared fixed/tuned central provenance is incomplete"
        )
    by_initial: dict[str, list[LedgerEntry]] = {}
    for entry in shared:
        by_initial.setdefault(entry.initial.run_name, []).append(entry)
    if len(by_initial) != 2 or any(
        len(entries) != 2 for entries in by_initial.values()
    ):
        raise SelectionEvidenceError(
            "shared central artifacts are not reused exactly twice"
        )
    if any(
        len({entry.current.run_name for entry in entries}) != 1
        for entries in by_initial.values()
    ):
        raise SelectionEvidenceError("shared central correction provenance diverged")


def select_winners(
    ledger: CalibratedLedger,
    approval: SelectionPolicyApproval | None = None,
) -> dict[TreatmentSlot, LedgerEntry]:
    _require_selection_approval(approval)
    _raise_on_exhausted(ledger.entries)
    return {slot: _select_winner(ledger.for_slot(slot)) for slot in ledger.slots}


def plan_next_probes(
    ledger: CalibratedLedger,
    manifest: CandidateManifest | None = None,
    approval: ProbePolicyApproval | None = None,
    selection_approval: SelectionPolicyApproval | None = None,
) -> ProbePlan:
    manifest = initial_manifest() if manifest is None else manifest
    _require_canonical_probe_approval(manifest.approval)
    _require_canonical_selection_approval(manifest.selection_approval)
    _require_canonical_horizon_followup_approval(manifest.horizon_followup_approval)
    if approval is not None:
        _require_canonical_probe_approval(approval)
    if selection_approval is not None:
        _require_canonical_selection_approval(selection_approval)
    _validate_manifest_ledger(manifest, ledger)
    _raise_on_exhausted(ledger.entries)
    if (
        manifest.approval is not None
        and approval is not None
        and manifest.approval != approval
    ):
        raise SelectionEvidenceError("RQ5 manifest and supplied approvals disagree")
    if (
        manifest.selection_approval is not None
        and selection_approval is not None
        and manifest.selection_approval != selection_approval
    ):
        raise SelectionEvidenceError(
            "RQ5 manifest and supplied selection approvals disagree"
        )
    effective_approval = manifest.approval or approval
    _require_selection_approval(manifest.selection_approval or selection_approval)
    boundary: list[Rq5Candidate] = []
    local: list[Rq5Candidate] = []
    unresolved: list[Rq5Candidate] = []
    for slot in ledger.slots:
        entries = ledger.for_slot(slot)
        if slot.treatment in _JOINT_TREATMENTS:
            slot_boundary, slot_local, slot_unresolved = _plan_joint_slot(slot, entries)
        else:
            slot_boundary, slot_local, slot_unresolved = _plan_one_dimensional_slot(
                slot, entries
            )
        boundary.extend(slot_boundary)
        local.extend(slot_local)
        unresolved.extend(slot_unresolved)
    plan = ProbePlan(tuple(boundary), tuple(local), tuple(unresolved))
    _validate_probe_budget(manifest, plan)
    if plan.unresolved:
        raise BoundaryOutcomeApprovalRequired(plan.unresolved)
    if plan.candidates and effective_approval is None:
        raise ProbeApprovalRequired(
            "RQ5 local and boundary probe policy is pending explicit approval"
        )
    return plan


def select_final_winners(
    ledger: CalibratedLedger,
    manifest: CandidateManifest,
) -> dict[TreatmentSlot, LedgerEntry]:
    plan = plan_next_probes(ledger, manifest)
    if plan.candidates:
        raise FinalSelectionIncomplete(
            "RQ5 final selection refused because approved probe stages remain"
        )
    return select_winners(ledger, manifest.selection_approval)


def _plan_one_dimensional_slot(
    slot: TreatmentSlot, entries: tuple[LedgerEntry, ...]
) -> tuple[
    tuple[Rq5Candidate, ...],
    tuple[Rq5Candidate, ...],
    tuple[Rq5Candidate, ...],
]:
    initial, local, boundary = _split_probe_stages(slot, entries)
    if local:
        raise SelectionEvidenceError(f"{slot}: one-dimensional slot has local probes")
    initial_winner = _select_winner(initial)
    expected = _one_dimensional_boundary(slot, initial_winner.current)
    if not boundary:
        return expected, (), ()
    _require_exact_surface(slot, "boundary", boundary, expected)
    winner = _select_winner((*initial, *boundary))
    unresolved = (winner.current,) if _is_outer_boundary(winner.initial) else ()
    return (), (), unresolved


def _plan_joint_slot(slot: TreatmentSlot, entries: tuple[LedgerEntry, ...]) -> tuple[
    tuple[Rq5Candidate, ...],
    tuple[Rq5Candidate, ...],
    tuple[Rq5Candidate, ...],
]:
    initial, local, boundary = _split_probe_stages(slot, entries)
    initial_winner = _select_winner(initial)
    expected_local = _joint_local_probes(slot, initial_winner.current)
    if not local and not boundary:
        return (), expected_local, ()
    if not local:
        raise SelectionEvidenceError(f"{slot}: boundary stage precedes local stage")
    _require_exact_surface(slot, "local", local, expected_local)
    expanded_winner = _select_winner((*initial, *local))
    expected_boundary = _joint_boundary_probes(slot, expanded_winner.current)
    if not boundary:
        return expected_boundary, (), ()
    _require_exact_surface(slot, "boundary", boundary, expected_boundary)
    winner = _select_winner((*initial, *local, *boundary))
    unresolved = (winner.current,) if _is_outer_boundary(winner.initial) else ()
    return (), (), unresolved


def _split_probe_stages(slot: TreatmentSlot, entries: tuple[LedgerEntry, ...]) -> tuple[
    tuple[LedgerEntry, ...],
    tuple[LedgerEntry, ...],
    tuple[LedgerEntry, ...],
]:
    initial = tuple(entry for entry in entries if entry.initial.probe is None)
    local = tuple(
        entry
        for entry in entries
        if entry.initial.probe is not None and entry.initial.probe.startswith("local")
    )
    boundary = tuple(
        entry
        for entry in entries
        if entry.initial.probe is not None and entry.initial.probe.startswith("b1")
    )
    recognized = len(initial) + len(local) + len(boundary)
    if recognized != len(entries):
        raise SelectionEvidenceError(f"{slot}: unknown RQ5 probe stage")
    if len(initial) != 3:
        raise SelectionEvidenceError(f"{slot}: initial surface is not three candidates")
    return initial, local, boundary


def _require_exact_surface(
    slot: TreatmentSlot,
    stage: str,
    entries: tuple[LedgerEntry, ...],
    expected: tuple[Rq5Candidate, ...],
) -> None:
    actual_names = {entry.initial.run_name for entry in entries}
    expected_names = {candidate.run_name for candidate in expected}
    if actual_names != expected_names or len(entries) != len(expected):
        raise SelectionEvidenceError(
            f"{slot}: {stage} manifest surface is partial or does not match its winner"
        )


def _select_winner(entries: tuple[LedgerEntry, ...]) -> LedgerEntry:
    eligible = tuple(entry for entry in entries if not entry.ineligible_exclusion)
    if not eligible:
        raise SelectionEvidenceError(
            "cannot select an RQ5 winner from an empty surface"
        )
    return min(
        eligible,
        key=lambda entry: (
            -_selection_metrics(entry.current, entry.selection_metrics)[0],
            -_selection_metrics(entry.current, entry.selection_metrics)[1],
            entry.initial.run_name,
        ),
    )


def _require_selection_approval(
    approval: SelectionPolicyApproval | None,
) -> None:
    _require_canonical_selection_approval(approval)


def _raise_on_exhausted(entries: tuple[LedgerEntry, ...]) -> None:
    exhausted = [
        entry.current.run_name
        for entry in entries
        if entry.exhausted and not entry.ineligible_exclusion
    ]
    if exhausted:
        raise SelectionEvidenceError(
            f"RQ5 contains exhausted candidates: {sorted(set(exhausted))}"
        )


def _validate_manifest_ledger(
    manifest: CandidateManifest, ledger: CalibratedLedger
) -> None:
    manifest_names = {candidate.run_name for candidate in manifest.candidates}
    ledger_names = {entry.initial.run_name for entry in ledger.entries}
    if manifest_names != ledger_names:
        raise SelectionEvidenceError("RQ5 ledger does not match its candidate manifest")


def _validate_probe_budget(manifest: CandidateManifest, plan: ProbePlan) -> None:
    new_names = [candidate.run_name for candidate in plan.candidates]
    existing_names = {candidate.run_name for candidate in manifest.candidates}
    if len(new_names) != len(set(new_names)) or existing_names.intersection(new_names):
        raise SelectionEvidenceError("RQ5 probe identities collide or were re-emitted")
    combined = (*manifest.candidates, *plan.candidates)
    boundary_count = sum(
        candidate.probe is not None and candidate.probe.startswith("b1")
        for candidate in combined
    )
    local_count = sum(
        candidate.probe is not None and candidate.probe.startswith("local")
        for candidate in combined
    )
    if boundary_count > 81 or local_count > 12:
        raise SelectionEvidenceError("RQ5 conditional probe budget exceeded")
    configurations = len(combined)
    if configurations > 160 or configurations * 3 > 480:
        raise SelectionEvidenceError("RQ5 candidate/run ceiling exceeded")


def _ranking_metrics(
    candidate: Rq5Candidate,
    metrics: dict | None,
    evidence_name: str = "best-epoch validation",
) -> tuple[float, float]:
    if not isinstance(metrics, dict):
        raise SelectionEvidenceError(
            f"{candidate.run_name}: missing {evidence_name} metrics"
        )
    values = (metrics.get("recall@100"), metrics.get("ndcg@100"))
    if any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        for value in values
    ):
        raise SelectionEvidenceError(
            f"{candidate.run_name}: invalid {evidence_name} ranking metrics"
        )
    return float(values[0]), float(values[1])


def _selection_metrics(
    candidate: Rq5Candidate, metrics: dict | None
) -> tuple[float, float]:
    recall, ndcg = _ranking_metrics(candidate, metrics)
    return float(f"{recall:.4f}"), float(f"{ndcg:.4f}")


def _one_dimensional_boundary(
    slot: TreatmentSlot, winner: Rq5Candidate
) -> tuple[Rq5Candidate, ...]:
    if math.isclose(winner.deep_lr, _INITIAL_DEEP_LRS[0]):
        values = (0.0015, 0.00075, 0.000375)
        side = "lo"
    elif math.isclose(winner.deep_lr, _INITIAL_DEEP_LRS[-1]):
        values = (0.024, 0.048, 0.096)
        side = "hi"
    else:
        return ()
    return tuple(
        _probe_candidate(
            winner,
            slot.treatment,
            deep_lr=value,
            probe=f"b1lr{side}{index}",
        )
        for index, value in enumerate(values, start=1)
    )


def _joint_boundary_probes(
    slot: TreatmentSlot, winner: Rq5Candidate
) -> tuple[Rq5Candidate, ...]:
    fraction = winner.joint_fraction
    if fraction is None:
        raise SelectionEvidenceError(f"{winner.run_name}: missing joint fraction")
    boundary = []
    for parameter, value, bounds in (
        ("deep", winner.deep_lr, _JOINT_DEEP_BOUNDS),
        ("fraction", fraction, _JOINT_FRACTION_BOUNDS),
    ):
        side = _boundary_side(value, bounds)
        if side is None:
            continue
        for index, probe_value in enumerate(
            _adjacent_log_probes(bounds, side), start=1
        ):
            kwargs = (
                {"deep_lr": probe_value}
                if parameter == "deep"
                else _fraction_kwargs(winner, probe_value)
            )
            boundary.append(
                _probe_candidate(
                    winner,
                    slot.treatment,
                    probe=f"b1{parameter[0]}{side}{index}",
                    **kwargs,
                )
            )
    return tuple(boundary)


def _joint_local_probes(
    slot: TreatmentSlot, winner: Rq5Candidate
) -> tuple[Rq5Candidate, ...]:
    fraction = winner.joint_fraction
    if fraction is None:
        raise SelectionEvidenceError(f"{winner.run_name}: missing joint fraction")
    local = []
    if not (math.isclose(winner.deep_lr, 0.006) and math.isclose(fraction, 0.05)):
        generator = random.Random(42)
        deep_bounds = _local_bounds(winner.deep_lr, _JOINT_DEEP_BOUNDS)
        fraction_bounds = _local_bounds(fraction, _JOINT_FRACTION_BOUNDS)
        for index in range(1, 4):
            deep_lr = _log_uniform(generator, *deep_bounds)
            local_fraction = _log_uniform(generator, *fraction_bounds)
            local.append(
                _probe_candidate(
                    winner,
                    slot.treatment,
                    deep_lr=deep_lr,
                    probe=f"local{index}",
                    **_fraction_kwargs(winner, local_fraction),
                )
            )
    return tuple(local)


def _is_outer_boundary(candidate: Rq5Candidate) -> bool:
    return (
        candidate.probe is not None
        and re.fullmatch(r"b1(?:lr|d|f)(?:lo|hi)3", candidate.probe) is not None
    )


def _probe_candidate(
    winner: Rq5Candidate,
    treatment: str,
    *,
    probe: str,
    deep_lr: float | None = None,
    warmup_fraction: float | None = None,
    timescale_fraction: float | None = None,
) -> Rq5Candidate:
    return replace(
        winner,
        treatments=(treatment,),
        deep_lr=_canonical_float(winner.deep_lr if deep_lr is None else deep_lr),
        warmup_fraction=_canonical_float(
            winner.warmup_fraction if warmup_fraction is None else warmup_fraction
        ),
        timescale_fraction=(
            None
            if timescale_fraction is None and winner.timescale_fraction is None
            else _canonical_float(
                winner.timescale_fraction
                if timescale_fraction is None
                else timescale_fraction
            )
        ),
        attempt=0,
        probe=probe,
    )


def _fraction_kwargs(winner: Rq5Candidate, value: float) -> dict[str, float]:
    return (
        {"timescale_fraction": value}
        if winner.shape == "inverse_sqrt"
        else {"warmup_fraction": value}
    )


def _boundary_side(value: float, bounds: tuple[float, float]) -> str | None:
    position = math.log(value / bounds[0]) / math.log(bounds[1] / bounds[0])
    if position <= 0.25:
        return "lo"
    if position >= 0.75:
        return "hi"
    return None


def _adjacent_log_probes(
    bounds: tuple[float, float], side: str
) -> tuple[float, float, float]:
    low, high = bounds
    if side == "lo":
        return tuple(_canonical_float(low / 4 ** (index / 3)) for index in range(1, 4))
    if side == "hi":
        return tuple(_canonical_float(high * 4 ** (index / 3)) for index in range(1, 4))
    raise SelectionEvidenceError(f"invalid RQ5 boundary side {side!r}")


def _local_bounds(
    value: float, global_bounds: tuple[float, float]
) -> tuple[float, float]:
    return max(global_bounds[0], value / 2), min(global_bounds[1], value * 2)


def _log_uniform(generator: random.Random, low: float, high: float) -> float:
    return _canonical_float(math.exp(generator.uniform(math.log(low), math.log(high))))


def _canonical_float(value: float) -> float:
    return float(f"{value:.12g}")


def load_best_epoch_selection_metrics(
    directory: Path, metadata: dict
) -> dict[str, float]:
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
    try:
        lines = (directory / "sweep.log").read_text().splitlines()
    except OSError as error:
        raise SelectionEvidenceError(
            f"{directory.name}: missing sweep.log for validation selection"
        ) from error
    epoch_index = best_epoch - 1
    values = []
    for line in lines:
        if re.search(rf"\bepoch {epoch_index} finished\b", line) is None:
            continue
        recall = re.search(rf"\bepoch/val_true\.recall@100=({_METRIC_NUMBER})\b", line)
        ndcg = re.search(rf"\bepoch/val_true\.ndcg@100=({_METRIC_NUMBER})\b", line)
        if recall is None or ndcg is None:
            continue
        pair = (float(recall.group(1)), float(ndcg.group(1)))
        if not all(math.isfinite(value) for value in pair):
            continue
        values.append(pair)
    unique_values = set(values)
    if len(unique_values) != 1:
        raise SelectionEvidenceError(
            f"{directory.name}: best epoch has missing or conflicting validation metrics"
        )
    recall, ndcg = unique_values.pop()
    return {"recall@100": recall, "ndcg@100": ndcg}


def selection_filesystem_inspector(
    logs: Path,
) -> Callable[[Rq5Candidate], ArtifactEvidence]:
    inspect_artifact = filesystem_inspector(logs)

    def inspect(candidate: Rq5Candidate) -> ArtifactEvidence:
        evidence = inspect_artifact(candidate)
        if evidence.kind != "complete":
            return evidence
        if evidence.metadata is None:
            raise SelectionEvidenceError(
                f"{candidate.run_name}: complete artifact has no training metadata"
            )
        return replace(
            evidence,
            selection_metrics=load_best_epoch_selection_metrics(
                logs / candidate.run_name, evidence.metadata
            ),
        )

    return inspect


def advance_candidate_manifest(
    path: Path,
    build_ledger: Callable[[tuple[Rq5Candidate, ...]], CalibratedLedger],
    inspect_surface: Callable[[Rq5Candidate], ArtifactEvidence] | None = None,
) -> ProbePlan:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(f"{path.suffix}.lock")
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            manifest = CandidateManifest.thaw(path.read_text())
        else:
            manifest = initial_manifest()
            _write_manifest_atomically(path, manifest)
        _require_canonical_probe_approval(manifest.approval)
        _require_canonical_selection_approval(manifest.selection_approval)
        _require_canonical_horizon_followup_approval(manifest.horizon_followup_approval)
        pending = _pending_manifested_probes(manifest, inspect_surface)
        if pending.candidates:
            return pending
        ledger = build_ledger(manifest.candidates)
        plan = plan_next_probes(ledger, manifest)
        if plan.candidates:
            if manifest.approval is None:
                raise ProbeApprovalRequired(
                    "RQ5 probe policy disappeared before manifest update"
                )
            manifest = manifest.extend(plan.candidates, manifest.approval)
            _write_manifest_atomically(path, manifest)
        return plan


def _pending_manifested_probes(
    manifest: CandidateManifest,
    inspect_surface: Callable[[Rq5Candidate], ArtifactEvidence] | None,
) -> ProbePlan:
    if inspect_surface is None:
        return ProbePlan((), ())
    pending = []
    for candidate in manifest.candidates:
        if candidate.probe is None:
            continue
        evidence = inspect_surface(candidate)
        if evidence.kind in {"missing", "recoverable"}:
            pending.append(candidate)
        elif evidence.kind == "in_flight":
            raise SelectionEvidenceError(
                f"{candidate.run_name}: manifested probe is in flight"
            )
    boundary = tuple(
        candidate
        for candidate in pending
        if candidate.probe is not None and candidate.probe.startswith("b1")
    )
    local = tuple(
        candidate
        for candidate in pending
        if candidate.probe is not None and candidate.probe.startswith("local")
    )
    if len(boundary) + len(local) != len(pending):
        raise SelectionEvidenceError("manifest contains an unknown pending probe stage")
    return ProbePlan(boundary=boundary, local=local)


def _write_manifest_atomically(path: Path, manifest: CandidateManifest) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(manifest.freeze())
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    arguments = parser.parse_args(argv)
    try:
        manifest_path = arguments.manifest.resolve()
        if not manifest_path.is_relative_to(_REPO_ROOT):
            raise SelectionEvidenceError(
                "RQ5 candidate manifest must be stored inside the repository"
            )
        inspect = selection_filesystem_inspector(arguments.logs)
        plan = advance_candidate_manifest(
            manifest_path,
            lambda surfaces: build_calibrated_ledger(inspect, surfaces),
            inspect_surface=inspect,
        )
    except (
        OSError,
        ValueError,
        SelectionEvidenceError,
        CorrectionEvidenceError,
    ) as error:
        print(error, file=sys.stderr)
        return 2
    for candidate in plan.candidates:
        print(candidate.run_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
