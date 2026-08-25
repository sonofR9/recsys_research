from collections.abc import Callable
from dataclasses import asdict, fields, make_dataclass, replace
import fcntl
import json
from pathlib import Path
import runpy

import pytest

from experiments.g1_sasrec_item_ids_likes.analysis.rq5_scheduler_candidates import (
    Rq5Candidate,
    candidate_by_run,
    initial_candidates,
)
from experiments.g1_sasrec_item_ids_likes.analysis.rq5_scheduler_corrections import (
    ArtifactEvidence,
    CorrectionDecision,
    CorrectionEvidenceError,
    HorizonFollowupApproval,
    HorizonObservation,
    _in_flight,
    main,
    plan_corrections,
    recompute_correction,
    require_canonical_horizon_followup_approval,
)
from experiments.g1_sasrec_item_ids_likes.analysis.rq5_scheduler_selection import (
    CandidateManifest,
    initial_manifest,
)


CONFIG = (
    Path(__file__).parents[3]
    / "experiments/g1_sasrec_item_ids_likes/configs/rq5_scheduler_variant.py"
)


def test_canonical_approval_accepts_equivalent_module_identity() -> None:
    approval = initial_manifest().horizon_followup_approval
    assert approval is not None
    foreign_type = make_dataclass(
        "ForeignHorizonFollowupApproval",
        [(field.name, field.type) for field in fields(approval)],
        frozen=True,
    )
    foreign_approval = foreign_type(**asdict(approval))

    assert require_canonical_horizon_followup_approval(foreign_approval) == approval


def test_horizon_followup_approval_accepts_sorted_attempt_zero_probe_superset() -> None:
    approval = initial_manifest().horizon_followup_approval
    assert approval is not None
    probe = replace(_initial("linear"), probe="future1")

    expanded = replace(
        approval,
        approved_initial_surfaces=tuple(
            sorted((*approval.approved_initial_surfaces, probe.run_name))
        ),
    )

    assert expanded.approved_initial_surfaces == tuple(
        sorted((*approval.approved_initial_surfaces, probe.run_name))
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda surfaces, probe: (*surfaces, probe.run_name),
            "sorted unique",
        ),
        (
            lambda surfaces, probe: tuple(sorted((*surfaces, probe.run_name)))
            + (probe.run_name,),
            "sorted unique",
        ),
        (
            lambda surfaces, probe: surfaces[1:],
            "original 12",
        ),
        (
            lambda surfaces, probe: tuple(
                sorted((*surfaces, replace(probe, attempt=1).run_name))
            ),
            "attempt-0",
        ),
        (
            lambda surfaces, probe: tuple(sorted((*surfaces, "not-an-rq5-run"))),
            "canonical attempt-0",
        ),
        (
            lambda surfaces, probe: tuple(
                sorted(
                    (
                        *surfaces,
                        "g1_rq5_linear_both_d0p0030_h17_cap17_a0_ts2_r1_500m",
                    )
                )
            ),
            "canonical attempt-0",
        ),
    ],
)
def test_horizon_followup_approval_rejects_invalid_supersets(
    mutate,
    message: str,
) -> None:
    approval = initial_manifest().horizon_followup_approval
    assert approval is not None
    probe = replace(_initial("linear"), probe="future1")

    with pytest.raises(ValueError, match=message):
        replace(
            approval,
            approved_initial_surfaces=mutate(
                approval.approved_initial_surfaces,
                probe,
            ),
        )


def _metadata(
    candidate: Rq5Candidate,
    *,
    stopped: int,
    early_stopped: bool,
    status: str,
    next_value: int | None,
) -> dict:
    return {
        "stopped_epoch": stopped,
        "early_stopped": early_stopped,
        "horizon_calibration_status": status,
        "next_lr_schedule_horizon_epochs": next_value,
    }


def _initial(treatment: str) -> Rq5Candidate:
    return next(
        candidate
        for candidate in initial_candidates()
        if candidate.treatments == (treatment,)
    )


def _original_resolution_approval() -> HorizonFollowupApproval:
    document = json.loads(
        (CONFIG.parents[1] / "protocol/rq5_scheduler_approval.json").read_text()
    )["horizon_followup"]
    document["ineligible_initial_surfaces"] = tuple(
        run_name
        for run_name in document["ineligible_initial_surfaces"]
        if candidate_by_run(run_name).probe is None
    )
    return HorizonFollowupApproval(**document)


def test_horizon_followup_approval_accepts_added_probe_exclusion() -> None:
    approval = _original_resolution_approval()
    probe = next(
        run_name
        for run_name in approval.approved_initial_surfaces
        if candidate_by_run(run_name).probe is not None
    )

    expanded = replace(
        approval,
        ineligible_initial_surfaces=tuple(
            sorted((*approval.ineligible_initial_surfaces, probe))
        ),
    )

    assert probe in expanded.ineligible_initial_surfaces


def test_horizon_followup_approval_accepts_added_probe_final_horizon() -> None:
    approval = _original_resolution_approval()
    probe = next(
        run_name
        for run_name in approval.approved_initial_surfaces
        if candidate_by_run(run_name).probe is not None
    )

    expanded = replace(
        approval,
        final_attempt_horizons={**approval.final_attempt_horizons, probe: 19},
    )

    assert expanded.final_attempt_horizons[probe] == 19


def test_horizon_followup_approval_rejects_original_exclusion_omission() -> None:
    approval = _original_resolution_approval()

    with pytest.raises(ValueError, match="original four ineligible"):
        replace(
            approval,
            ineligible_initial_surfaces=approval.ineligible_initial_surfaces[1:],
        )


@pytest.mark.parametrize(
    "ineligible_initial_surfaces",
    [
        lambda approval: tuple(reversed(approval.ineligible_initial_surfaces)),
        lambda approval: (
            *approval.ineligible_initial_surfaces,
            approval.ineligible_initial_surfaces[-1],
        ),
    ],
)
def test_horizon_followup_approval_rejects_nondeterministic_exclusions(
    ineligible_initial_surfaces: Callable[
        [HorizonFollowupApproval], tuple[str, ...]
    ],
) -> None:
    approval = _original_resolution_approval()

    with pytest.raises(ValueError, match="sorted unique"):
        replace(
            approval,
            ineligible_initial_surfaces=ineligible_initial_surfaces(approval),
        )


def test_horizon_followup_approval_rejects_unapproved_probe_exclusion() -> None:
    approval = _original_resolution_approval()
    unapproved = replace(_initial("linear"), probe="unapproved").run_name

    with pytest.raises(ValueError, match="approved"):
        replace(
            approval,
            ineligible_initial_surfaces=tuple(
                sorted((*approval.ineligible_initial_surfaces, unapproved))
            ),
        )


def test_horizon_followup_approval_rejects_original_final_mapping_omission() -> None:
    approval = _original_resolution_approval()
    final_attempt_horizons = dict(approval.final_attempt_horizons)
    final_attempt_horizons.pop(next(iter(final_attempt_horizons)))

    with pytest.raises(ValueError, match="original four exact"):
        replace(approval, final_attempt_horizons=final_attempt_horizons)


def test_horizon_followup_approval_rejects_changed_original_final_horizon() -> None:
    approval = _original_resolution_approval()
    run_name = next(iter(approval.final_attempt_horizons))

    with pytest.raises(ValueError, match="original four exact"):
        replace(
            approval,
            final_attempt_horizons={
                **approval.final_attempt_horizons,
                run_name: approval.final_attempt_horizons[run_name] + 1,
            },
        )


@pytest.mark.parametrize("horizon", [0, -1, True, 1.5, "19"])
def test_horizon_followup_approval_rejects_invalid_added_probe_horizon(
    horizon: object,
) -> None:
    approval = _original_resolution_approval()
    probe = next(
        run_name
        for run_name in approval.approved_initial_surfaces
        if candidate_by_run(run_name).probe is not None
    )

    with pytest.raises(ValueError, match="positive integer"):
        replace(
            approval,
            final_attempt_horizons={**approval.final_attempt_horizons, probe: horizon},
        )


def test_horizon_followup_approval_rejects_unapproved_probe_final_mapping() -> None:
    approval = _original_resolution_approval()
    unapproved = replace(_initial("linear"), probe="unapproved").run_name

    with pytest.raises(ValueError, match="approved"):
        replace(
            approval,
            final_attempt_horizons={
                **approval.final_attempt_horizons,
                unapproved: 19,
            },
        )


@pytest.mark.parametrize(
    ("treatment", "stopped", "early", "status", "next_value", "horizon", "cap"),
    [
        ("linear", 8, True, "shorten_horizon", 8, 8, 8),
        ("linear", 17, False, "extend_horizon", 26, 26, 26),
        ("inverse_sqrt", 10, True, "recalibrate_horizon", 10, 10, 80),
        ("inverse_sqrt", 80, False, "extend_cap", 120, 23, 120),
        ("constant", 80, False, "extend_cap", 120, None, 120),
    ],
)
def test_recomputes_each_approved_correction_from_stopping_evidence(
    treatment: str,
    stopped: int,
    early: bool,
    status: str,
    next_value: int,
    horizon: int | None,
    cap: int,
) -> None:
    candidate = _initial(treatment)
    metadata = _metadata(
        candidate,
        stopped=stopped,
        early_stopped=early,
        status=status,
        next_value=next_value,
    )

    decision = recompute_correction(candidate, metadata)

    assert decision.status == status
    assert decision.target is not None
    assert decision.target.attempt == 1
    assert decision.target.horizon_epochs == horizon
    assert decision.target.cap_epochs == cap
    assert candidate_by_run(decision.target.run_name) == decision.target


def test_stored_status_cannot_override_independently_recomputed_action() -> None:
    candidate = _initial("linear")
    metadata = _metadata(
        candidate,
        stopped=8,
        early_stopped=True,
        status="calibrated",
        next_value=None,
    )

    with pytest.raises(CorrectionEvidenceError, match="disagrees"):
        recompute_correction(candidate, metadata)


def test_correction_identity_loads_as_a_from_scratch_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _initial("linear")
    decision = recompute_correction(
        source,
        _metadata(
            source,
            stopped=8,
            early_stopped=True,
            status="shorten_horizon",
            next_value=8,
        ),
    )
    assert decision.target is not None
    monkeypatch.setenv("G1_RQ5_RUN", decision.target.run_name)

    experiment = runpy.run_path(str(CONFIG))["experiment"]

    assert experiment.run_name == decision.target.run_name
    assert experiment.lr_schedule_horizon_epochs == 8
    assert experiment.num_epochs == 8
    assert not experiment.checkpointing.load_checkpoint


def test_approved_attempt_five_is_a_hard_limit() -> None:
    candidate = replace(_initial("cosine"), horizon_epochs=22, cap_epochs=22, attempt=5)
    metadata = _metadata(
        candidate,
        stopped=4,
        early_stopped=True,
        status="shorten_horizon",
        next_value=4,
    )

    decision = recompute_correction(candidate, metadata)

    assert decision.exhausted
    assert decision.target is None


def test_run_identity_accepts_approved_attempt_five_and_rejects_attempt_six() -> None:
    candidate = replace(_initial("cosine"), horizon_epochs=22, cap_epochs=22, attempt=5)

    assert candidate_by_run(candidate.run_name) == candidate
    with pytest.raises(ValueError, match="unknown RQ5 candidate run"):
        candidate_by_run(candidate.run_name.replace("_a5_", "_a6_"))


def test_attempt_two_uses_lower_integer_midpoint_of_evidence_bracket() -> None:
    source = _initial("linear")
    attempt_one = replace(source, horizon_epochs=26, cap_epochs=26, attempt=1)
    attempt_two = replace(source, horizon_epochs=15, cap_epochs=15, attempt=2)
    expected = replace(source, horizon_epochs=21, cap_epochs=21, attempt=3)
    evidence = {
        source: ArtifactEvidence(
            "complete",
            _metadata(
                source,
                stopped=17,
                early_stopped=False,
                status="extend_horizon",
                next_value=26,
            ),
        ),
        attempt_one: ArtifactEvidence(
            "complete",
            _metadata(
                attempt_one,
                stopped=15,
                early_stopped=True,
                status="shorten_horizon",
                next_value=15,
            ),
        ),
        attempt_two: ArtifactEvidence(
            "complete",
            _metadata(
                attempt_two,
                stopped=15,
                early_stopped=False,
                status="extend_horizon",
                next_value=23,
            ),
        ),
        expected: ArtifactEvidence("missing"),
    }

    plan = plan_corrections(evidence.__getitem__, surfaces=(source,))

    assert plan.corrections == (expected,)
    assert candidate_by_run(expected.run_name) == expected


def test_unapproved_probe_surface_cannot_receive_bracketed_followups() -> None:
    source = replace(_initial("linear"), probe="future")
    attempt_one = replace(source, horizon_epochs=26, cap_epochs=26, attempt=1)
    attempt_two = replace(source, horizon_epochs=15, cap_epochs=15, attempt=2)
    evidence = {
        source: ArtifactEvidence(
            "complete",
            _metadata(
                source,
                stopped=17,
                early_stopped=False,
                status="extend_horizon",
                next_value=26,
            ),
        ),
        attempt_one: ArtifactEvidence(
            "complete",
            _metadata(
                attempt_one,
                stopped=15,
                early_stopped=True,
                status="shorten_horizon",
                next_value=15,
            ),
        ),
        attempt_two: ArtifactEvidence(
            "complete",
            _metadata(
                attempt_two,
                stopped=15,
                early_stopped=False,
                status="extend_horizon",
                next_value=23,
            ),
        ),
    }

    plan = plan_corrections(evidence.__getitem__, surfaces=(source,))

    assert plan.corrections == ()
    assert plan.exhausted == (attempt_two,)


def test_bracketed_attempt_loads_as_a_fresh_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = replace(_initial("linear"), horizon_epochs=21, cap_epochs=21, attempt=3)
    monkeypatch.setenv("G1_RQ5_RUN", candidate.run_name)

    experiment = runpy.run_path(str(CONFIG))["experiment"]

    assert experiment.run_name == candidate.run_name
    assert experiment.lr_schedule_horizon_epochs == 21
    assert experiment.num_epochs == 21
    assert not experiment.checkpointing.load_checkpoint


def test_final_approved_attempt_loads_as_a_fresh_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = replace(_initial("cosine"), horizon_epochs=22, cap_epochs=22, attempt=5)
    monkeypatch.setenv("G1_RQ5_RUN", candidate.run_name)

    experiment = runpy.run_path(str(CONFIG))["experiment"]

    assert experiment.run_name == candidate.run_name
    assert experiment.lr_schedule_horizon_epochs == 22
    assert experiment.num_epochs == 22
    assert not experiment.checkpointing.load_checkpoint


def test_attempt_three_updates_the_bracket_before_emitting_attempt_four() -> None:
    source = _initial("linear")
    attempt_one = replace(source, horizon_epochs=26, cap_epochs=26, attempt=1)
    attempt_two = replace(source, horizon_epochs=15, cap_epochs=15, attempt=2)
    attempt_three = replace(source, horizon_epochs=21, cap_epochs=21, attempt=3)
    expected = replace(source, horizon_epochs=19, cap_epochs=19, attempt=4)
    evidence = {
        source: ArtifactEvidence(
            "complete",
            _metadata(
                source,
                stopped=17,
                early_stopped=False,
                status="extend_horizon",
                next_value=26,
            ),
        ),
        attempt_one: ArtifactEvidence(
            "complete",
            _metadata(
                attempt_one,
                stopped=15,
                early_stopped=True,
                status="shorten_horizon",
                next_value=15,
            ),
        ),
        attempt_two: ArtifactEvidence(
            "complete",
            _metadata(
                attempt_two,
                stopped=15,
                early_stopped=False,
                status="extend_horizon",
                next_value=23,
            ),
        ),
        attempt_three: ArtifactEvidence(
            "complete",
            _metadata(
                attempt_three,
                stopped=17,
                early_stopped=True,
                status="shorten_horizon",
                next_value=17,
            ),
        ),
        expected: ArtifactEvidence("missing"),
    }

    plan = plan_corrections(evidence.__getitem__, surfaces=(source,))

    assert plan.corrections == (expected,)


def test_exact_approved_attempt_five_uses_the_sole_remaining_integer() -> None:
    source = _initial("cosine")
    attempt_one = replace(source, horizon_epochs=32, cap_epochs=32, attempt=1)
    attempt_two = replace(source, horizon_epochs=15, cap_epochs=15, attempt=2)
    attempt_three = replace(source, horizon_epochs=26, cap_epochs=26, attempt=3)
    attempt_four = replace(source, horizon_epochs=23, cap_epochs=23, attempt=4)
    expected = replace(source, horizon_epochs=22, cap_epochs=22, attempt=5)
    evidence = {
        source: ArtifactEvidence(
            "complete",
            _metadata(
                source,
                stopped=21,
                early_stopped=False,
                status="extend_horizon",
                next_value=32,
            ),
        ),
        attempt_one: ArtifactEvidence(
            "complete",
            _metadata(
                attempt_one,
                stopped=15,
                early_stopped=True,
                status="shorten_horizon",
                next_value=15,
            ),
        ),
        attempt_two: ArtifactEvidence(
            "complete",
            _metadata(
                attempt_two,
                stopped=15,
                early_stopped=False,
                status="extend_horizon",
                next_value=23,
            ),
        ),
        attempt_three: ArtifactEvidence(
            "complete",
            _metadata(
                attempt_three,
                stopped=22,
                early_stopped=True,
                status="shorten_horizon",
                next_value=22,
            ),
        ),
        attempt_four: ArtifactEvidence(
            "complete",
            _metadata(
                attempt_four,
                stopped=15,
                early_stopped=True,
                status="shorten_horizon",
                next_value=15,
            ),
        ),
        expected: ArtifactEvidence("missing"),
    }

    plan = plan_corrections(evidence.__getitem__, surfaces=(source,))

    assert plan.corrections == (expected,)
    assert candidate_by_run(expected.run_name) == expected


def _attempt_four_decision(
    low: int,
    high: int,
    approved_horizon: int,
    *,
    additional_observations: tuple[HorizonObservation, ...] = (),
) -> CorrectionDecision:
    approval = initial_manifest().horizon_followup_approval
    assert approval is not None
    source = next(
        candidate
        for candidate in map(candidate_by_run, approval.approved_initial_surfaces)
        if candidate.probe is not None
    )
    approval = replace(
        approval,
        final_attempt_horizons={
            **approval.final_attempt_horizons,
            source.run_name: approved_horizon,
        },
    )
    candidate = replace(
        source,
        horizon_epochs=high,
        cap_epochs=high,
        attempt=4,
    )
    return recompute_correction(
        candidate,
        _metadata(
            candidate,
            stopped=low - 1,
            early_stopped=True,
            status="shorten_horizon",
            next_value=low - 1,
        ),
        prior_observations=(
            HorizonObservation(low, "extend_horizon"),
            *additional_observations,
        ),
        horizon_followup_approval=approval,
        initial_surface=source,
    )


@pytest.mark.parametrize(
    ("low", "high", "approved_horizon"),
    [(27, 30, 28), (29, 32, 30)],
)
def test_attempt_four_accepts_approved_lower_midpoint_with_multiple_interiors(
    low: int,
    high: int,
    approved_horizon: int,
) -> None:
    decision = _attempt_four_decision(low, high, approved_horizon)

    assert decision.target is not None
    assert decision.target.attempt == 5
    assert decision.target.horizon_epochs == approved_horizon
    assert decision.target.cap_epochs == approved_horizon


def test_attempt_four_rejects_approved_non_midpoint_interior_horizon() -> None:
    with pytest.raises(CorrectionEvidenceError, match="lower midpoint"):
        _attempt_four_decision(27, 30, 29)


@pytest.mark.parametrize("endpoint", [27, 30])
def test_attempt_four_rejects_approved_bracket_endpoint(endpoint: int) -> None:
    with pytest.raises(CorrectionEvidenceError, match="strictly inside"):
        _attempt_four_decision(27, 30, endpoint)


def test_attempt_four_rejects_approved_midpoint_that_was_already_tried() -> None:
    with pytest.raises(CorrectionEvidenceError, match="already tried"):
        _attempt_four_decision(
            27,
            30,
            28,
            additional_observations=(HorizonObservation(28, "calibrated"),),
        )


def test_terminal_attempt_five_is_an_audited_approved_exclusion() -> None:
    source = _initial("cosine")
    attempts = (
        source,
        replace(source, horizon_epochs=32, cap_epochs=32, attempt=1),
        replace(source, horizon_epochs=15, cap_epochs=15, attempt=2),
        replace(source, horizon_epochs=26, cap_epochs=26, attempt=3),
        replace(source, horizon_epochs=23, cap_epochs=23, attempt=4),
        replace(source, horizon_epochs=22, cap_epochs=22, attempt=5),
    )
    stopping = (
        (21, False, "extend_horizon", 32),
        (15, True, "shorten_horizon", 15),
        (15, False, "extend_horizon", 23),
        (22, True, "shorten_horizon", 22),
        (15, True, "shorten_horizon", 15),
        (17, True, "shorten_horizon", 17),
    )
    evidence = {
        candidate: ArtifactEvidence(
            "complete",
            _metadata(
                candidate,
                stopped=stopped,
                early_stopped=early_stopped,
                status=status,
                next_value=next_value,
            ),
        )
        for candidate, (stopped, early_stopped, status, next_value) in zip(
            attempts, stopping, strict=True
        )
    }

    plan = plan_corrections(evidence.__getitem__, surfaces=(source,))

    assert plan.corrections == ()
    assert plan.exhausted == ()
    assert plan.excluded == (attempts[-1],)


def test_parser_rejects_unapproved_attempt_five_identity() -> None:
    source = next(
        candidate
        for candidate in initial_candidates()
        if candidate.treatments == ("linear",)
        and candidate.scope == "both"
        and candidate.deep_lr == 0.006
    )
    unauthorized = replace(source, horizon_epochs=22, cap_epochs=22, attempt=5)

    with pytest.raises(ValueError, match="unauthorized RQ5 a5"):
        candidate_by_run(unauthorized.run_name)


def test_parser_rejects_probe_shaped_attempt_five_with_value_error() -> None:
    unauthorized = replace(
        _initial("cosine"),
        horizon_epochs=22,
        cap_epochs=22,
        attempt=5,
        probe="future",
    )

    with pytest.raises(ValueError, match="unauthorized RQ5 a5"):
        candidate_by_run(unauthorized.run_name)


def test_parser_accepts_exact_approved_probe_attempt_five() -> None:
    run_name = (
        "g1_rq5_exponential_both_d0p048_pb1lrhi2_"
        "h28_cap28_a5_ts2_r1_500m"
    )

    candidate = candidate_by_run(run_name)

    assert candidate.run_name == run_name
    assert candidate.probe == "b1lrhi2"
    assert candidate.attempt == 5
    assert candidate.horizon_epochs == 28


@pytest.mark.parametrize(
    "run_name",
    [
        "g1_rq5_exponential_both_d0p048_pb1lrhi2_"
        "h29_cap29_a5_ts2_r1_500m",
        "g1_rq5_exponential_both_d0p048_pb1lrhi2_"
        "h28_cap29_a5_ts2_r1_500m",
        "g1_rq5_exponential_both_d0p048_pwrong_"
        "h28_cap28_a5_ts2_r1_500m",
    ],
)
def test_parser_rejects_wrong_approved_probe_attempt_five(run_name: str) -> None:
    with pytest.raises(ValueError, match="unauthorized RQ5 a5"):
        candidate_by_run(run_name)


@pytest.mark.parametrize(
    "bad_surface",
    [
        "not-an-rq5-run",
        "g1_rq5_linear_both_d0p0030_h17_cap17_a0_ts2_r1_500m",
        "g1_rq5_exponential_both_d0p048_pb1lrhi2_"
        "h19_cap19_a0_ts2_r1_500m",
    ],
)
def test_attempt_five_parser_fails_closed_on_invalid_approval_surface(
    bad_surface: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    approval_record = json.loads(
        (CONFIG.parents[1] / "protocol/rq5_scheduler_approval.json").read_text()
    )
    approved = approval_record["horizon_followup"]["approved_initial_surfaces"]
    approval_record["horizon_followup"]["approved_initial_surfaces"] = sorted(
        (*approved, bad_surface)
    )
    path = tmp_path / "approval.json"
    path.write_text(json.dumps(approval_record))
    monkeypatch.setattr(
        "experiments.g1_sasrec_item_ids_likes.analysis."
        "rq5_scheduler_candidates._APPROVAL_RECORD",
        path,
    )
    approved_nonprobe = replace(
        _initial("cosine"),
        horizon_epochs=22,
        cap_epochs=22,
        attempt=5,
    )

    with pytest.raises(ValueError, match="a5 approval record is invalid"):
        candidate_by_run(approved_nonprobe.run_name)


def test_approved_ineligible_surface_is_audited_without_blocking_corrections() -> None:
    source = _initial("cosine_warmup5_cycles4")
    attempt_one = replace(source, horizon_epochs=13, cap_epochs=13, attempt=1)
    attempt_two = replace(source, horizon_epochs=20, cap_epochs=20, attempt=2)
    attempt_three = replace(source, horizon_epochs=21, cap_epochs=21, attempt=3)
    evidence = {
        source: ArtifactEvidence(
            "complete",
            _metadata(
                source,
                stopped=13,
                early_stopped=True,
                status="shorten_horizon",
                next_value=13,
            ),
        ),
        attempt_one: ArtifactEvidence(
            "complete",
            _metadata(
                attempt_one,
                stopped=13,
                early_stopped=False,
                status="extend_horizon",
                next_value=20,
            ),
        ),
        attempt_two: ArtifactEvidence(
            "complete",
            _metadata(
                attempt_two,
                stopped=20,
                early_stopped=False,
                status="extend_horizon",
                next_value=30,
            ),
        ),
        attempt_three: ArtifactEvidence(
            "complete",
            _metadata(
                attempt_three,
                stopped=17,
                early_stopped=True,
                status="shorten_horizon",
                next_value=17,
            ),
        ),
    }

    plan = plan_corrections(evidence.__getitem__, surfaces=(source,))

    assert plan.corrections == ()
    assert plan.exhausted == ()
    assert plan.excluded == (attempt_three,)


def test_narrow_integer_bracket_stops_without_colliding_with_a_tried_horizon() -> None:
    source = _initial("cosine_warmup5_cycles2")
    attempt_one = replace(source, horizon_epochs=13, cap_epochs=13, attempt=1)
    attempt_two = replace(source, horizon_epochs=20, cap_epochs=20, attempt=2)
    attempt_three = replace(source, horizon_epochs=21, cap_epochs=21, attempt=3)
    evidence = {
        source: ArtifactEvidence(
            "complete",
            _metadata(
                source,
                stopped=13,
                early_stopped=True,
                status="shorten_horizon",
                next_value=13,
            ),
        ),
        attempt_one: ArtifactEvidence(
            "complete",
            _metadata(
                attempt_one,
                stopped=13,
                early_stopped=False,
                status="extend_horizon",
                next_value=20,
            ),
        ),
        attempt_two: ArtifactEvidence(
            "complete",
            _metadata(
                attempt_two,
                stopped=20,
                early_stopped=False,
                status="extend_horizon",
                next_value=30,
            ),
        ),
        attempt_three: ArtifactEvidence(
            "complete",
            _metadata(
                attempt_three,
                stopped=21,
                early_stopped=False,
                status="extend_horizon",
                next_value=32,
            ),
        ),
    }

    plan = plan_corrections(evidence.__getitem__, surfaces=(source,))

    assert plan.corrections == ()
    assert plan.exhausted == (attempt_three,)


def test_bracketed_followup_keeps_the_existing_acceptance_tolerance() -> None:
    candidate = replace(_initial("linear"), horizon_epochs=20, cap_epochs=20, attempt=3)
    approval = initial_manifest().horizon_followup_approval
    assert approval is not None
    decision = recompute_correction(
        candidate,
        _metadata(
            candidate,
            stopped=17,
            early_stopped=True,
            status="calibrated",
            next_value=None,
        ),
        prior_observations=(
            HorizonObservation(13, "extend_horizon"),
            HorizonObservation(22, "shorten_horizon"),
        ),
        horizon_followup_approval=approval,
    )

    assert decision.status == "calibrated"
    assert decision.target is None


def _calibrated(candidate: Rq5Candidate) -> ArtifactEvidence:
    if candidate.shape == "constant":
        stopped = min(20, candidate.cap_epochs - 1)
    else:
        assert candidate.horizon_epochs is not None
        stopped = candidate.horizon_epochs
    return ArtifactEvidence(
        "complete",
        _metadata(
            candidate,
            stopped=stopped,
            early_stopped=True,
            status="calibrated",
            next_value=None,
        ),
        strictly_eligible=True,
    )


def test_plan_waits_for_all_67_initial_artifacts_before_emitting() -> None:
    missing = initial_candidates()[-1]

    def inspect(candidate: Rq5Candidate) -> ArtifactEvidence:
        return (
            ArtifactEvidence("missing")
            if candidate == missing
            else _calibrated(candidate)
        )

    with pytest.raises(CorrectionEvidenceError, match="required evidence is missing"):
        plan_corrections(inspect)


@pytest.mark.parametrize("kind", ["recoverable", "in_flight"])
def test_plan_refuses_malformed_or_in_flight_initial_evidence(kind: str) -> None:
    blocked = initial_candidates()[0]

    def inspect(candidate: Rq5Candidate) -> ArtifactEvidence:
        return (
            ArtifactEvidence(kind) if candidate == blocked else _calibrated(candidate)
        )

    with pytest.raises(CorrectionEvidenceError, match=f"required evidence is {kind}"):
        plan_corrections(inspect)


def test_plan_emits_only_evidence_determined_attempt_and_follows_completed_chain() -> (
    None
):
    source = _initial("linear")
    attempt_one = replace(source, horizon_epochs=8, cap_epochs=8, attempt=1)
    expected = replace(attempt_one, horizon_epochs=4, cap_epochs=4, attempt=2)

    def inspect(candidate: Rq5Candidate) -> ArtifactEvidence:
        if candidate == source:
            return ArtifactEvidence(
                "complete",
                _metadata(
                    candidate,
                    stopped=8,
                    early_stopped=True,
                    status="shorten_horizon",
                    next_value=8,
                ),
            )
        if candidate == attempt_one:
            return ArtifactEvidence(
                "complete",
                _metadata(
                    candidate,
                    stopped=4,
                    early_stopped=True,
                    status="shorten_horizon",
                    next_value=4,
                ),
            )
        if candidate == expected:
            return ArtifactEvidence("missing")
        return _calibrated(candidate)

    plan = plan_corrections(inspect)

    assert plan.corrections == (expected,)
    assert plan.exhausted == ()


def test_plan_accepts_probe_surfaces_and_emits_their_attempt_chain() -> None:
    source = replace(
        _initial("linear"),
        treatments=("linear",),
        deep_lr=0.0015,
        probe="b1lrlo1",
    )
    expected = replace(source, horizon_epochs=8, cap_epochs=8, attempt=1)

    def inspect(candidate: Rq5Candidate) -> ArtifactEvidence:
        if candidate == source:
            return ArtifactEvidence(
                "complete",
                _metadata(
                    candidate,
                    stopped=8,
                    early_stopped=True,
                    status="shorten_horizon",
                    next_value=8,
                ),
            )
        if candidate == expected:
            return ArtifactEvidence("missing")
        raise AssertionError(f"unexpected surface {candidate.run_name}")

    plan = plan_corrections(inspect, surfaces=(source,))

    assert plan.corrections == (expected,)
    assert candidate_by_run(expected.run_name) == expected


def test_correction_cli_reads_and_verifies_frozen_manifest_surfaces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = replace(
        _initial("linear"),
        treatments=("linear",),
        deep_lr=0.0015,
        probe="b1lrlo1",
    )
    expected = replace(source, horizon_epochs=8, cap_epochs=8, attempt=1)
    base_manifest = initial_manifest()
    assert base_manifest.approval is not None
    manifest = base_manifest.extend((source,), base_manifest.approval)
    path = tmp_path / "manifest.json"
    path.write_text(manifest.freeze())

    def inspect(candidate: Rq5Candidate) -> ArtifactEvidence:
        if candidate == source:
            return ArtifactEvidence(
                "complete",
                _metadata(
                    candidate,
                    stopped=8,
                    early_stopped=True,
                    status="shorten_horizon",
                    next_value=8,
                ),
            )
        if candidate == expected:
            return ArtifactEvidence("missing")
        return _calibrated(candidate)

    monkeypatch.setattr(
        "experiments.g1_sasrec_item_ids_likes.analysis."
        "rq5_scheduler_corrections.filesystem_inspector",
        lambda logs: inspect,
    )

    assert main(["--logs", str(tmp_path), "--manifest", str(path)]) == 0
    assert capsys.readouterr().out.splitlines() == [expected.run_name]


def test_correction_cli_rejects_valid_manifest_with_swapped_horizon_approval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = next(
        candidate
        for candidate in initial_candidates()
        if candidate.treatments == ("linear",)
        and candidate.scope == "both"
        and candidate.deep_lr == 0.006
    )
    attempt_one = replace(source, horizon_epochs=26, cap_epochs=26, attempt=1)
    attempt_two = replace(source, horizon_epochs=15, cap_epochs=15, attempt=2)
    expected = replace(source, horizon_epochs=21, cap_epochs=21, attempt=3)
    manifest = initial_manifest()
    approval = manifest.horizon_followup_approval
    assert approval is not None
    probe = replace(_initial("linear"), probe="future1")
    swapped = replace(
        approval,
        approved_initial_surfaces=tuple(
            sorted((*approval.approved_initial_surfaces, probe.run_name))
        ),
    )
    manifest = replace(manifest, horizon_followup_approval=swapped)
    path = tmp_path / "manifest.json"
    path.write_text(manifest.freeze())
    assert CandidateManifest.thaw(path.read_text()) == manifest

    def inspect(candidate: Rq5Candidate) -> ArtifactEvidence:
        if candidate == source:
            return ArtifactEvidence(
                "complete",
                _metadata(
                    source,
                    stopped=17,
                    early_stopped=False,
                    status="extend_horizon",
                    next_value=26,
                ),
            )
        if candidate == attempt_one:
            return ArtifactEvidence(
                "complete",
                _metadata(
                    attempt_one,
                    stopped=15,
                    early_stopped=True,
                    status="shorten_horizon",
                    next_value=15,
                ),
            )
        if candidate == attempt_two:
            return ArtifactEvidence(
                "complete",
                _metadata(
                    attempt_two,
                    stopped=15,
                    early_stopped=False,
                    status="extend_horizon",
                    next_value=23,
                ),
            )
        if candidate == expected:
            return ArtifactEvidence("missing")
        return _calibrated(candidate)

    monkeypatch.setattr(
        "experiments.g1_sasrec_item_ids_likes.analysis."
        "rq5_scheduler_corrections.filesystem_inspector",
        lambda logs: inspect,
    )

    assert main(["--logs", str(tmp_path), "--manifest", str(path)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "canonical horizon follow-up approval" in captured.err


def test_correction_cli_refuses_a_tampered_manifest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    document = json.loads(
        CandidateManifest(
            tuple(
                sorted(initial_candidates(), key=lambda candidate: candidate.run_name)
            )
        ).freeze()
    )
    document["candidates"][0]["deep_lr"] *= 2
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(document))

    assert main(["--logs", str(tmp_path), "--manifest", str(path)]) == 2
    assert "digest mismatch" in capsys.readouterr().err


def test_correction_cli_refuses_manifest_without_followup_approval(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = replace(initial_manifest(), horizon_followup_approval=None)
    path = tmp_path / "manifest.json"
    path.write_text(manifest.freeze())

    assert main(["--logs", str(tmp_path), "--manifest", str(path)]) == 2
    assert "lacks horizon follow-up approval" in capsys.readouterr().err


def test_completed_calibrated_evidence_must_pass_strict_eligibility() -> None:
    blocked = initial_candidates()[0]

    def inspect(candidate: Rq5Candidate) -> ArtifactEvidence:
        evidence = _calibrated(candidate)
        return (
            replace(evidence, strictly_eligible=False)
            if candidate == blocked
            else evidence
        )

    with pytest.raises(CorrectionEvidenceError, match="not strictly eligible"):
        plan_corrections(inspect)


def test_active_training_lock_is_detected(tmp_path: Path) -> None:
    run_name = initial_candidates()[0].run_name
    lock = tmp_path / ".run-locks" / f"{run_name}.lock"
    lock.parent.mkdir()
    lock.touch()
    with lock.open("r+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert _in_flight(tmp_path, run_name)
        fcntl.flock(stream, fcntl.LOCK_UN)

    assert not _in_flight(tmp_path, run_name)
