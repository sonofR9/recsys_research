from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import fcntl
import json
import math
from pathlib import Path
import sys
from typing import Callable, Literal

from experiments.g1_sasrec_item_ids_likes.analysis.rq5_scheduler_candidates import (
    Rq5Candidate,
    candidate_by_run,
    initial_candidates,
)
from experiments.g1_sasrec_item_ids_likes.launchers import verify_artifact


CONFIG = Path(__file__).parents[1] / "configs/rq5_scheduler_variant.py"
APPROVAL_RECORD = Path(__file__).parents[1] / "protocol/rq5_scheduler_approval.json"
ArtifactKind = Literal["complete", "missing", "recoverable", "in_flight"]
_ORIGINAL_APPROVED_INITIAL_SURFACES = (
    "g1_rq5_cosine_both_d0p003_h21_cap21_a0_ts2_r1_500m",
    "g1_rq5_cosine_warmup5_cycles2_both_d0p003_h22_cap22_a0_ts2_r1_500m",
    "g1_rq5_cosine_warmup5_cycles2_both_d0p006_h22_cap22_a0_ts2_r1_500m",
    "g1_rq5_cosine_warmup5_cycles2_both_d0p012_h22_cap22_a0_ts2_r1_500m",
    "g1_rq5_cosine_warmup5_cycles4_both_d0p003_h22_cap22_a0_ts2_r1_500m",
    "g1_rq5_exponential_both_d0p003_h18_cap18_a0_ts2_r1_500m",
    "g1_rq5_exponential_both_d0p006_h18_cap18_a0_ts2_r1_500m",
    "g1_rq5_linear_both_d0p003_h17_cap17_a0_ts2_r1_500m",
    "g1_rq5_polynomial_both_d0p006_h20_cap20_a0_ts2_r1_500m",
    "g1_rq5_step_both_d0p012_h20_cap20_a0_ts2_r1_500m",
    "g1_rq5_wsd_both_d0p003_h20_cap20_a0_ts2_r1_500m",
    "g1_rq5_wsd_deep_only_d0p006_h20_cap20_a0_ts2_r1_500m",
)
_ORIGINAL_FINAL_ATTEMPT_HORIZONS = (
    ("g1_rq5_cosine_both_d0p003_h21_cap21_a0_ts2_r1_500m", 22),
    (
        "g1_rq5_cosine_warmup5_cycles2_both_d0p003_h22_cap22_a0_ts2_r1_500m",
        23,
    ),
    (
        "g1_rq5_cosine_warmup5_cycles2_both_d0p012_h22_cap22_a0_ts2_r1_500m",
        17,
    ),
    ("g1_rq5_exponential_both_d0p006_h18_cap18_a0_ts2_r1_500m", 21),
)
_ORIGINAL_INELIGIBLE_INITIAL_SURFACES = (
    "g1_rq5_cosine_both_d0p003_h21_cap21_a0_ts2_r1_500m",
    "g1_rq5_cosine_warmup5_cycles2_both_d0p012_h22_cap22_a0_ts2_r1_500m",
    "g1_rq5_cosine_warmup5_cycles4_both_d0p003_h22_cap22_a0_ts2_r1_500m",
    "g1_rq5_exponential_both_d0p006_h18_cap18_a0_ts2_r1_500m",
)


class CorrectionEvidenceError(RuntimeError):
    pass


@dataclass(frozen=True)
class ArtifactEvidence:
    kind: ArtifactKind
    metadata: dict | None = None
    strictly_eligible: bool = False
    metrics: dict | None = None
    selection_metrics: dict | None = None
    optimizer_group_traces_verified: bool = False


@dataclass(frozen=True)
class HorizonObservation:
    horizon: int
    status: str


@dataclass(frozen=True)
class HorizonFollowupApproval:
    reference: str
    approved_initial_surfaces: tuple[str, ...]
    final_resolution_reference: str
    final_attempt_horizons: dict[str, int]
    ineligible_initial_surfaces: tuple[str, ...]
    max_additional_attempts: int = 2
    max_final_attempts: int = 1
    midpoint_rounding: str = "lower"
    stagewise: bool = True
    preserve_acceptance_tolerance: bool = True

    def __post_init__(self) -> None:
        approved_initial_surfaces = tuple(self.approved_initial_surfaces)
        object.__setattr__(self, "approved_initial_surfaces", approved_initial_surfaces)
        final_attempt_horizons = dict(self.final_attempt_horizons)
        object.__setattr__(self, "final_attempt_horizons", final_attempt_horizons)
        ineligible_initial_surfaces = tuple(self.ineligible_initial_surfaces)
        object.__setattr__(
            self, "ineligible_initial_surfaces", ineligible_initial_surfaces
        )
        if not self.reference.strip() or not self.final_resolution_reference.strip():
            raise ValueError("horizon follow-up approval needs a non-empty reference")
        if (
            self.max_additional_attempts != 2
            or self.max_final_attempts != 1
            or self.midpoint_rounding != "lower"
            or self.stagewise is not True
            or self.preserve_acceptance_tolerance is not True
        ):
            raise ValueError(
                "horizon follow-up approval does not match the approved RQ5 policy"
            )
        if tuple(sorted(set(approved_initial_surfaces))) != approved_initial_surfaces:
            raise ValueError(
                "horizon follow-up approval must name sorted unique surfaces"
            )
        approved_surface_names = set(approved_initial_surfaces)
        if not set(_ORIGINAL_APPROVED_INITIAL_SURFACES) <= approved_surface_names:
            raise ValueError(
                "horizon follow-up approval must retain the original 12 surfaces"
            )
        for run_name in approved_initial_surfaces:
            try:
                candidate = candidate_by_run(run_name)
            except ValueError as error:
                raise ValueError(
                    "horizon follow-up approval must name canonical attempt-0 "
                    "RQ5 surfaces"
                ) from error
            if candidate.attempt != 0:
                raise ValueError(
                    "horizon follow-up approval must name attempt-0 RQ5 surfaces"
                )
        if any(
            final_attempt_horizons.get(run_name) != horizon
            for run_name, horizon in _ORIGINAL_FINAL_ATTEMPT_HORIZONS
        ):
            raise ValueError(
                "horizon follow-up approval must retain the original four exact "
                "final-attempt horizons"
            )
        if not set(final_attempt_horizons) <= approved_surface_names:
            raise ValueError(
                "horizon follow-up approval final-attempt horizons must target "
                "approved canonical attempt-0 surfaces"
            )
        if any(
            not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1
            for horizon in final_attempt_horizons.values()
        ):
            raise ValueError(
                "horizon follow-up approval final-attempt horizons must be positive "
                "integers"
            )
        if ineligible_initial_surfaces != tuple(
            sorted(set(ineligible_initial_surfaces))
        ):
            raise ValueError(
                "horizon follow-up approval must name sorted unique ineligible "
                "surfaces"
            )
        if not set(_ORIGINAL_INELIGIBLE_INITIAL_SURFACES) <= set(
            ineligible_initial_surfaces
        ):
            raise ValueError(
                "horizon follow-up approval must retain the original four ineligible "
                "surfaces"
            )
        if not set(ineligible_initial_surfaces) <= approved_surface_names:
            raise ValueError(
                "horizon follow-up approval ineligible surfaces must be approved "
                "canonical attempt-0 surfaces"
            )


@dataclass(frozen=True)
class CorrectionDecision:
    status: str
    next_value: int | None
    target: Rq5Candidate | None
    exhausted: bool = False


@dataclass(frozen=True)
class CorrectionPlan:
    corrections: tuple[Rq5Candidate, ...]
    exhausted: tuple[Rq5Candidate, ...]
    excluded: tuple[Rq5Candidate, ...] = ()


def recompute_correction(
    candidate: Rq5Candidate,
    metadata: dict,
    *,
    prior_observations: tuple[HorizonObservation, ...] = (),
    horizon_followup_approval: HorizonFollowupApproval | None = None,
    initial_surface: Rq5Candidate | None = None,
) -> CorrectionDecision:
    stopped = metadata.get("stopped_epoch")
    early_stopped = metadata.get("early_stopped")
    if (
        not isinstance(stopped, int)
        or isinstance(stopped, bool)
        or stopped < 1
        or stopped > candidate.cap_epochs
        or not isinstance(early_stopped, bool)
    ):
        raise CorrectionEvidenceError(
            f"{candidate.run_name}: invalid stopping evidence"
        )
    if candidate.shape == "constant":
        if early_stopped and stopped < candidate.cap_epochs:
            status, next_value = "calibrated", None
        else:
            status = "extend_cap"
            next_value = math.ceil(1.5 * candidate.cap_epochs)
    else:
        horizon = candidate.horizon_epochs
        if horizon is None:
            raise CorrectionEvidenceError(
                f"{candidate.run_name}: missing reference horizon"
            )
        tolerance = max(3, round(0.1 * horizon))
        if candidate.shape == "inverse_sqrt":
            if not early_stopped:
                status = "extend_cap"
                next_value = math.ceil(1.5 * candidate.cap_epochs)
            elif abs(horizon - stopped) <= tolerance:
                status, next_value = "calibrated", None
            else:
                status, next_value = "recalibrate_horizon", stopped
        elif not early_stopped:
            status = "extend_horizon"
            next_value = math.ceil(1.5 * horizon)
        elif 0 <= horizon - stopped <= tolerance:
            status, next_value = "calibrated", None
        else:
            status, next_value = "shorten_horizon", stopped
    if (
        metadata.get("horizon_calibration_status") != status
        or metadata.get("next_lr_schedule_horizon_epochs") != next_value
    ):
        raise CorrectionEvidenceError(
            f"{candidate.run_name}: stored horizon status disagrees with evidence"
        )
    if status == "calibrated":
        return CorrectionDecision(status, next_value, None)
    if candidate.attempt >= 2:
        target = _bracketed_target(
            candidate,
            status,
            prior_observations,
            horizon_followup_approval,
            initial_surface,
        )
        return CorrectionDecision(
            status,
            None if target is None else target.horizon_epochs,
            target,
            exhausted=target is None,
        )
    if status == "extend_cap":
        target = replace(
            candidate,
            cap_epochs=next_value,
            attempt=candidate.attempt + 1,
        )
    elif candidate.shape == "inverse_sqrt":
        target = replace(
            candidate,
            horizon_epochs=next_value,
            cap_epochs=max(80, 2 * next_value),
            attempt=candidate.attempt + 1,
        )
    else:
        target = replace(
            candidate,
            horizon_epochs=next_value,
            cap_epochs=next_value,
            attempt=candidate.attempt + 1,
        )
    return CorrectionDecision(status, next_value, target)


def _bracketed_target(
    candidate: Rq5Candidate,
    status: str,
    prior_observations: tuple[HorizonObservation, ...],
    approval: HorizonFollowupApproval | None,
    initial_surface: Rq5Candidate | None,
) -> Rq5Candidate | None:
    if candidate.attempt >= 5 or status not in {
        "extend_horizon",
        "shorten_horizon",
    }:
        return None
    if approval is None:
        raise CorrectionEvidenceError(
            f"{candidate.run_name}: bracketed horizon follow-up lacks approval"
        )
    if initial_surface is None:
        raise CorrectionEvidenceError(
            f"{candidate.run_name}: bracketed follow-up lacks its initial surface"
        )
    if (
        initial_surface.attempt != 0
        or replace(
            candidate,
            horizon_epochs=initial_surface.horizon_epochs,
            cap_epochs=initial_surface.cap_epochs,
            attempt=0,
        )
        != initial_surface
    ):
        raise CorrectionEvidenceError(
            f"{candidate.run_name}: correction chain does not match its initial surface"
        )
    if initial_surface.run_name not in approval.approved_initial_surfaces:
        return None
    if candidate.horizon_epochs is None:
        raise CorrectionEvidenceError(
            f"{candidate.run_name}: bracketed follow-up lacks a horizon"
        )
    observations = (
        *prior_observations,
        HorizonObservation(candidate.horizon_epochs, status),
    )
    too_short = [
        observation.horizon
        for observation in observations
        if observation.status == "extend_horizon"
    ]
    too_long = [
        observation.horizon
        for observation in observations
        if observation.status == "shorten_horizon"
    ]
    if not too_short or not too_long:
        return None
    low, high = max(too_short), min(too_long)
    if low >= high:
        raise CorrectionEvidenceError(
            f"{candidate.run_name}: contradictory horizon bracket [{low}, {high}]"
        )
    tried = {observation.horizon for observation in observations}
    midpoint = (low + high) // 2
    if candidate.attempt == 4:
        final_horizon = approval.final_attempt_horizons.get(initial_surface.run_name)
        if final_horizon is None:
            return None
        if not low < final_horizon < high:
            raise CorrectionEvidenceError(
                f"{candidate.run_name}: approved final horizon {final_horizon} "
                f"is not strictly inside bracket [{low}, {high}]"
            )
        if final_horizon != midpoint:
            raise CorrectionEvidenceError(
                f"{candidate.run_name}: approved final horizon {final_horizon} "
                f"is not lower midpoint {midpoint} of bracket [{low}, {high}]"
            )
        if final_horizon in tried:
            raise CorrectionEvidenceError(
                f"{candidate.run_name}: approved final horizon was already tried"
            )
        return replace(
            candidate,
            horizon_epochs=final_horizon,
            cap_epochs=final_horizon,
            attempt=5,
        )
    if not low < midpoint < high or midpoint in tried:
        return None
    return replace(
        candidate,
        horizon_epochs=midpoint,
        cap_epochs=midpoint,
        attempt=candidate.attempt + 1,
    )


def load_horizon_followup_approval() -> HorizonFollowupApproval:
    document = json.loads(APPROVAL_RECORD.read_text())
    if not isinstance(document, dict) or not isinstance(
        document.get("horizon_followup"), dict
    ):
        raise CorrectionEvidenceError("RQ5 horizon follow-up approval is missing")
    try:
        return HorizonFollowupApproval(**document["horizon_followup"])
    except (TypeError, ValueError) as error:
        raise CorrectionEvidenceError(
            "RQ5 horizon follow-up approval is invalid"
        ) from error


def require_canonical_horizon_followup_approval(
    approval: HorizonFollowupApproval,
) -> HorizonFollowupApproval:
    canonical = load_horizon_followup_approval()
    if asdict(approval) != asdict(canonical):
        raise CorrectionEvidenceError(
            "RQ5 supplied approval does not equal the canonical horizon "
            "follow-up approval"
        )
    return canonical


def plan_corrections(
    inspect: Callable[[Rq5Candidate], ArtifactEvidence],
    surfaces: tuple[Rq5Candidate, ...] | None = None,
    horizon_followup_approval: HorizonFollowupApproval | None = None,
) -> CorrectionPlan:
    surfaces = initial_candidates() if surfaces is None else surfaces
    if horizon_followup_approval is None:
        horizon_followup_approval = load_horizon_followup_approval()
    else:
        horizon_followup_approval = require_canonical_horizon_followup_approval(
            horizon_followup_approval
        )
    corrections: list[Rq5Candidate] = []
    exhausted: list[Rq5Candidate] = []
    excluded: list[Rq5Candidate] = []
    for initial in surfaces:
        current = initial
        observations: list[HorizonObservation] = []
        while True:
            evidence = inspect(current)
            if evidence.kind != "complete" or evidence.metadata is None:
                raise CorrectionEvidenceError(
                    f"{current.run_name}: required evidence is {evidence.kind}"
                )
            decision = recompute_correction(
                current,
                evidence.metadata,
                prior_observations=tuple(observations),
                horizon_followup_approval=horizon_followup_approval,
                initial_surface=initial,
            )
            if decision.status == "calibrated":
                if not evidence.strictly_eligible:
                    raise CorrectionEvidenceError(
                        f"{current.run_name}: calibrated evidence is not strictly eligible"
                    )
                break
            if evidence.strictly_eligible:
                raise CorrectionEvidenceError(
                    f"{current.run_name}: unresolved evidence passed strict verification"
                )
            if current.horizon_epochs is not None:
                observations.append(
                    HorizonObservation(current.horizon_epochs, decision.status)
                )
            if decision.exhausted:
                destination = (
                    excluded
                    if initial.run_name
                    in horizon_followup_approval.ineligible_initial_surfaces
                    else exhausted
                )
                destination.append(current)
                break
            assert decision.target is not None
            target_evidence = inspect(decision.target)
            if target_evidence.kind == "complete":
                current = decision.target
                continue
            if target_evidence.kind == "in_flight":
                raise CorrectionEvidenceError(
                    f"{decision.target.run_name}: correction is in flight"
                )
            corrections.append(decision.target)
            break
    names = [candidate.run_name for candidate in corrections]
    if len(names) != len(set(names)):
        raise CorrectionEvidenceError("correction plan contains colliding run names")
    return CorrectionPlan(tuple(corrections), tuple(exhausted), tuple(excluded))


def filesystem_inspector(logs: Path) -> Callable[[Rq5Candidate], ArtifactEvidence]:
    def inspect(candidate: Rq5Candidate) -> ArtifactEvidence:
        directory = logs / candidate.run_name
        if not directory.exists() and not directory.is_symlink():
            return ArtifactEvidence("missing")
        if _in_flight(logs, candidate.run_name):
            return ArtifactEvidence("in_flight")
        assignments = [f"G1_RQ5_RUN={candidate.run_name}"]
        if not verify_artifact.verify_config_recipe(directory, CONFIG, assignments):
            return ArtifactEvidence("recoverable")
        try:
            metadata = json.loads((directory / "training_metadata.json").read_text())
            metrics = json.loads((directory / "final_metrics.json").read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise CorrectionEvidenceError(
                f"{candidate.run_name}: recipe verifier accepted malformed metadata"
            ) from error
        return ArtifactEvidence(
            "complete",
            metadata=verify_artifact._with_legacy_accumulation_defaults(metadata),
            strictly_eligible=verify_artifact.verify_config(
                directory, CONFIG, assignments
            ),
            metrics=metrics,
            optimizer_group_traces_verified=verify_artifact._valid_group_lr_traces(
                metadata,
                (metadata.get("transfer_invariants") or {}).get("lr_schedule", {}),
            ),
        )

    return inspect


def _in_flight(logs: Path, run_name: str) -> bool:
    lock_path = logs / ".run-locks" / f"{run_name}.lock"
    if not lock_path.exists():
        return False
    with lock_path.open("r+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(stream, fcntl.LOCK_UN)
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    arguments = parser.parse_args(argv)
    try:
        surfaces = None
        horizon_followup_approval = None
        if arguments.manifest is not None:
            from experiments.g1_sasrec_item_ids_likes.analysis.rq5_scheduler_selection import (
                CandidateManifest,
            )

            manifest = CandidateManifest.thaw(arguments.manifest.read_text())
            surfaces = manifest.candidates
            horizon_followup_approval = manifest.horizon_followup_approval
            if horizon_followup_approval is None:
                raise CorrectionEvidenceError(
                    "RQ5 manifest lacks horizon follow-up approval"
                )
        plan = plan_corrections(
            filesystem_inspector(arguments.logs),
            surfaces=surfaces,
            horizon_followup_approval=horizon_followup_approval,
        )
    except (OSError, ValueError, CorrectionEvidenceError) as error:
        print(error, file=sys.stderr)
        return 2
    for candidate in plan.corrections:
        print(candidate.run_name)
    for candidate in plan.exhausted:
        print(
            f"unresolved after approved correction attempts: {candidate.run_name}",
            file=sys.stderr,
        )
    for candidate in plan.excluded:
        print(
            f"approved ineligible exclusion: {candidate.run_name}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
