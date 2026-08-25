import json
from dataclasses import replace

import pytest

from experiments.g1_sasrec_item_ids_likes.analysis.rq5_scheduler_candidates import (
    Rq5Candidate,
    candidate_by_run,
    initial_candidates,
)
from experiments.g1_sasrec_item_ids_likes.analysis.rq5_scheduler_corrections import (
    ArtifactEvidence,
)
from experiments.g1_sasrec_item_ids_likes.analysis.rq5_scheduler_selection import (
    BoundaryOutcomeApprovalRequired,
    CandidateManifest,
    FinalSelectionIncomplete,
    ProbeApprovalRequired,
    SelectionEvidenceError,
    SelectionPolicyApprovalRequired,
    TreatmentSlot,
    advance_candidate_manifest,
    build_calibrated_ledger,
    initial_manifest,
    load_best_epoch_selection_metrics,
    main as selection_main,
    plan_next_probes,
    select_final_winners,
    select_winners,
)


APPROVAL = initial_manifest().approval
SELECTION_APPROVAL = initial_manifest().selection_approval
assert APPROVAL is not None
assert SELECTION_APPROVAL is not None


def _approved_manifest() -> CandidateManifest:
    return initial_manifest()


def _calibrated(
    candidate: Rq5Candidate,
    recall: float,
    ndcg: float,
    *,
    final_recall: float | None = None,
    final_ndcg: float | None = None,
) -> ArtifactEvidence:
    stopped = (
        min(20, candidate.cap_epochs - 1)
        if candidate.shape == "constant"
        else candidate.horizon_epochs
    )
    assert stopped is not None
    return ArtifactEvidence(
        "complete",
        metadata={
            "best_epoch": max(1, stopped - 1),
            "stopped_epoch": stopped,
            "early_stopped": True,
            "horizon_calibration_status": "calibrated",
            "next_lr_schedule_horizon_epochs": None,
        },
        strictly_eligible=True,
        metrics={
            "recall@100": recall if final_recall is None else final_recall,
            "ndcg@100": ndcg if final_ndcg is None else final_ndcg,
        },
        selection_metrics={"recall@100": recall, "ndcg@100": ndcg},
    )


def _central_score(candidate: Rq5Candidate) -> float:
    if any(
        treatment in candidate.treatments
        for treatment in ("inverse_sqrt", "cosine_warmup_tuned")
    ):
        return (
            1.0
            if candidate.deep_lr == 0.006 and candidate.joint_fraction == 0.05
            else 0.0
        )
    return 1.0 if candidate.deep_lr == 0.006 else 0.0


def _inspector(
    scores: dict[str, tuple[float, float]] | None = None,
    evidence: dict[str, ArtifactEvidence] | None = None,
    final_scores: dict[str, tuple[float, float]] | None = None,
):
    scores = {} if scores is None else scores
    supplied_evidence = {} if evidence is None else evidence
    evidence = _approved_exclusion_evidence()
    evidence.update(supplied_evidence)
    final_scores = {} if final_scores is None else final_scores

    def inspect(candidate: Rq5Candidate) -> ArtifactEvidence:
        if candidate.run_name in evidence:
            return evidence[candidate.run_name]
        recall, ndcg = scores.get(
            candidate.run_name,
            (_central_score(candidate), _central_score(candidate)),
        )
        final_recall, final_ndcg = final_scores.get(candidate.run_name, (recall, ndcg))
        return _calibrated(
            candidate,
            recall,
            ndcg,
            final_recall=final_recall,
            final_ndcg=final_ndcg,
        )

    return inspect


def _candidate(treatment: str, scope: str, deep_lr: float) -> Rq5Candidate:
    return next(
        candidate
        for candidate in initial_candidates()
        if treatment in candidate.treatments
        and candidate.scope == scope
        and candidate.deep_lr == deep_lr
    )


def _approved_exclusion_evidence() -> dict[str, ArtifactEvidence]:
    def unresolved(
        candidate: Rq5Candidate,
        stopped: int,
        early_stopped: bool,
        status: str,
        next_value: int,
    ) -> ArtifactEvidence:
        return ArtifactEvidence(
            "complete",
            metadata={
                "stopped_epoch": stopped,
                "early_stopped": early_stopped,
                "horizon_calibration_status": status,
                "next_lr_schedule_horizon_epochs": next_value,
            },
            metrics={"recall@100": 100.0, "ndcg@100": 100.0},
            selection_metrics={"recall@100": 100.0, "ndcg@100": 100.0},
        )

    def chain(
        initial: Rq5Candidate,
        attempts: tuple[tuple[int, int, bool, str, int], ...],
    ) -> dict[str, ArtifactEvidence]:
        result = {}
        for attempt, horizon, stopped, early_stopped, status, next_value in (
            (index, *values) for index, values in enumerate(attempts)
        ):
            candidate = replace(
                initial,
                horizon_epochs=horizon,
                cap_epochs=horizon,
                attempt=attempt,
            )
            result[candidate.run_name] = unresolved(
                candidate, stopped, early_stopped, status, next_value
            )
        return result

    evidence = {}
    evidence.update(
        chain(
            _candidate("cosine", "both", 0.003),
            (
                (21, 21, False, "extend_horizon", 32),
                (32, 15, True, "shorten_horizon", 15),
                (15, 15, False, "extend_horizon", 23),
                (26, 22, True, "shorten_horizon", 22),
                (23, 15, True, "shorten_horizon", 15),
                (22, 17, True, "shorten_horizon", 17),
            ),
        )
    )
    evidence.update(
        chain(
            _candidate("exponential", "both", 0.006),
            (
                (18, 18, False, "extend_horizon", 27),
                (27, 13, True, "shorten_horizon", 13),
                (13, 13, False, "extend_horizon", 20),
                (22, 16, True, "shorten_horizon", 16),
                (20, 20, False, "extend_horizon", 30),
                (21, 16, True, "shorten_horizon", 16),
            ),
        )
    )
    evidence.update(
        chain(
            _candidate("cosine_warmup5_cycles2", "both", 0.012),
            (
                (22, 13, True, "shorten_horizon", 13),
                (13, 13, False, "extend_horizon", 20),
                (20, 12, True, "shorten_horizon", 12),
                (16, 16, False, "extend_horizon", 24),
                (18, 10, True, "shorten_horizon", 10),
                (17, 17, False, "extend_horizon", 26),
            ),
        )
    )
    evidence.update(
        chain(
            _candidate("cosine_warmup5_cycles4", "both", 0.003),
            (
                (22, 13, True, "shorten_horizon", 13),
                (13, 13, False, "extend_horizon", 20),
                (20, 20, False, "extend_horizon", 30),
                (21, 17, True, "shorten_horizon", 17),
            ),
        )
    )
    return evidence


def _ledger(
    manifest: CandidateManifest | None = None,
    scores: dict[str, tuple[float, float]] | None = None,
    evidence: dict[str, ArtifactEvidence] | None = None,
    final_scores: dict[str, tuple[float, float]] | None = None,
):
    manifest = initial_manifest() if manifest is None else manifest
    return build_calibrated_ledger(
        _inspector(scores, evidence, final_scores), manifest.candidates
    )


def test_ledger_maps_67_artifacts_into_69_three_candidate_treatment_slots() -> None:
    ledger = _ledger()

    assert len(ledger.entries) == 69
    assert len(ledger.slots) == 23
    assert {len(ledger.for_slot(slot)) for slot in ledger.slots} == {3}
    assert len({entry.current.run_name for entry in ledger.entries}) == 67
    shared = [entry for entry in ledger.entries if len(entry.initial.treatments) == 2]
    assert len(shared) == 4
    assert len({entry.current.run_name for entry in shared}) == 2


def test_selection_ranks_by_best_epoch_validation_recall_then_ndcg_then_name() -> None:
    slot = TreatmentSlot("linear", "both")
    low = _candidate("linear", "both", 0.003)
    central = _candidate("linear", "both", 0.006)
    high = _candidate("linear", "both", 0.012)
    scores = {
        low.run_name: (0.2, 0.1),
        central.run_name: (0.2, 0.2),
        high.run_name: (0.19, 0.9),
    }
    ledger = _ledger(scores=scores)

    assert select_winners(ledger, SELECTION_APPROVAL)[slot].current == central

    tied = {candidate.run_name: (0.2, 0.2) for candidate in (low, central, high)}
    ledger = _ledger(scores=tied)
    assert select_winners(ledger, SELECTION_APPROVAL)[slot].initial.run_name == min(
        tied
    )


def test_full_user_final_metrics_do_not_select_the_candidate() -> None:
    slot = TreatmentSlot("linear", "both")
    validation_winner = _candidate("linear", "both", 0.003)
    central = _candidate("linear", "both", 0.006)
    final_winner = _candidate("linear", "both", 0.012)
    scores = {
        validation_winner.run_name: (0.3, 0.2),
        central.run_name: (0.1, 0.1),
        final_winner.run_name: (0.2, 0.2),
    }
    final_scores = {
        validation_winner.run_name: (0.1, 0.1),
        final_winner.run_name: (0.9, 0.9),
    }

    winner = select_winners(
        _ledger(scores=scores, final_scores=final_scores), SELECTION_APPROVAL
    )[slot]

    assert winner.current == validation_winner
    assert winner.metrics == {"recall@100": 0.1, "ndcg@100": 0.1}
    assert winner.selection_metrics == {"recall@100": 0.3, "ndcg@100": 0.2}


def test_selection_normalizes_recall_and_ndcg_to_four_decimals_before_ranking() -> None:
    slot = TreatmentSlot("linear", "both")
    low = _candidate("linear", "both", 0.003)
    central = _candidate("linear", "both", 0.006)
    high = _candidate("linear", "both", 0.012)
    scores = {
        low.run_name: (0.20004, 0.100041),
        central.run_name: (0.1, 0.1),
        high.run_name: (0.20003, 0.100042),
    }

    winner = select_winners(_ledger(scores=scores), SELECTION_APPROVAL)[slot]

    assert winner.current == min((low, high), key=lambda candidate: candidate.run_name)


def test_four_decimal_selection_policy_is_pending_and_required() -> None:
    with pytest.raises(SelectionPolicyApprovalRequired, match="four-decimal"):
        select_winners(_ledger())


def test_best_epoch_selection_metrics_come_from_the_same_validation_epoch(
    tmp_path,
) -> None:
    log = tmp_path / "sweep.log"
    log.write_text(
        "epoch 0 finished epoch/val_true.recall@100=0.1000 "
        "epoch/val_true.ndcg@100=0.0500\n"
        "epoch 1 finished epoch/val_true.recall@100=0.2000 "
        "epoch/val_true.ndcg@100=0.0600\n"
        "epoch 2 finished epoch/val_true.recall@100=0.1500 "
        "epoch/val_true.ndcg@100=0.0900\n"
    )

    assert load_best_epoch_selection_metrics(
        tmp_path, {"best_epoch": 2, "stopped_epoch": 3}
    ) == {"recall@100": 0.2, "ndcg@100": 0.06}


@pytest.mark.parametrize(
    "log_text",
    [
        "epoch 1 finished epoch/val_true.recall@100=0.2\n",
        (
            "epoch 1 finished epoch/val_true.recall@100=0.2 "
            "epoch/val_true.ndcg@100=0.1\n"
            "epoch 1 finished epoch/val_true.recall@100=0.3 "
            "epoch/val_true.ndcg@100=0.1\n"
        ),
    ],
)
def test_best_epoch_selection_metrics_fail_closed_on_incomplete_or_conflicting_log(
    tmp_path, log_text: str
) -> None:
    (tmp_path / "sweep.log").write_text(log_text)

    with pytest.raises(SelectionEvidenceError):
        load_best_epoch_selection_metrics(
            tmp_path, {"best_epoch": 2, "stopped_epoch": 3}
        )


def test_ledger_follows_a_completed_correction_chain_to_strict_eligibility() -> None:
    initial = _candidate("linear", "both", 0.003)
    correction = replace(initial, horizon_epochs=8, cap_epochs=8, attempt=1)
    evidence = {
        initial.run_name: ArtifactEvidence(
            "complete",
            metadata={
                "stopped_epoch": 8,
                "early_stopped": True,
                "horizon_calibration_status": "shorten_horizon",
                "next_lr_schedule_horizon_epochs": 8,
            },
        ),
        correction.run_name: _calibrated(correction, 0.4, 0.2),
    }

    ledger = _ledger(evidence=evidence)
    entry = next(entry for entry in ledger.entries if entry.initial == initial)

    assert entry.current == correction
    assert entry.metrics == {"recall@100": 0.4, "ndcg@100": 0.2}
    assert entry.selection_metrics == {"recall@100": 0.4, "ndcg@100": 0.2}


def test_selection_fails_closed_when_unresolved_evidence_passes_strict_verification() -> (
    None
):
    initial = _candidate("linear", "both", 0.003)
    evidence = {
        initial.run_name: ArtifactEvidence(
            "complete",
            metadata={
                "stopped_epoch": 8,
                "early_stopped": True,
                "horizon_calibration_status": "shorten_horizon",
                "next_lr_schedule_horizon_epochs": 8,
            },
            strictly_eligible=True,
        )
    }

    with pytest.raises(
        SelectionEvidenceError, match="unresolved evidence passed strict"
    ):
        _ledger(evidence=evidence)


@pytest.mark.parametrize(
    ("winner_lr", "expected"),
    [
        (0.003, (0.0015, 0.00075, 0.000375)),
        (0.012, (0.024, 0.048, 0.096)),
    ],
)
def test_one_dimensional_boundary_winner_emits_exactly_three_approved_probes(
    winner_lr: float, expected: tuple[float, ...]
) -> None:
    winner = _candidate("linear", "both", winner_lr)
    plan = plan_next_probes(
        _ledger(scores={winner.run_name: (2.0, 2.0)}),
        _approved_manifest(),
        APPROVAL,
    )
    probes = [
        candidate
        for candidate in plan.boundary
        if candidate.treatments == ("linear",) and candidate.scope == "both"
    ]

    assert tuple(candidate.deep_lr for candidate in probes) == expected
    assert {candidate.attempt for candidate in probes} == {0}
    assert all(candidate.probe.startswith("b1lr") for candidate in probes)
    assert all(
        candidate_by_run(candidate.run_name) == candidate for candidate in probes
    )


def test_probe_policy_is_pending_and_blocks_all_candidate_emission() -> None:
    winner = _candidate("linear", "both", 0.003)
    unapproved = replace(_approved_manifest(), approval=None)

    with pytest.raises(ProbeApprovalRequired, match="approval"):
        plan_next_probes(_ledger(scores={winner.run_name: (2.0, 2.0)}), unapproved)


def test_joint_search_waits_for_local_results_before_proposing_boundary() -> None:
    initial = next(
        candidate
        for candidate in initial_candidates()
        if candidate.treatments == ("inverse_sqrt",)
        and candidate.scope == "both"
        and candidate.joint_fraction != 0.05
    )
    scores = {initial.run_name: (2.0, 2.0)}

    first = plan_next_probes(_ledger(scores=scores), _approved_manifest(), APPROVAL)
    local = tuple(
        candidate
        for candidate in first.local
        if candidate.treatments == ("inverse_sqrt",) and candidate.scope == "both"
    )

    assert len(local) == 3
    assert not any(
        candidate.treatments == ("inverse_sqrt",) and candidate.scope == "both"
        for candidate in first.boundary
    )

    local_manifest = _approved_manifest().extend(local, APPROVAL)
    second = plan_next_probes(_ledger(local_manifest, scores=scores), local_manifest)
    boundary = tuple(
        candidate
        for candidate in second.boundary
        if candidate.treatments == ("inverse_sqrt",) and candidate.scope == "both"
    )

    assert second.local == ()
    assert len(boundary) == 3
    assert all(candidate.timescale_fraction != 0.0125 for candidate in boundary)
    assert min(candidate.timescale_fraction for candidate in boundary) == 0.003125


def test_probe_surface_enters_the_same_correction_chain() -> None:
    initial = next(
        candidate
        for candidate in initial_candidates()
        if candidate.treatments == ("inverse_sqrt",)
        and candidate.scope == "both"
        and candidate.joint_fraction != 0.05
    )
    scores = {initial.run_name: (2.0, 2.0)}
    local = tuple(
        candidate
        for candidate in plan_next_probes(
            _ledger(scores=scores), _approved_manifest(), APPROVAL
        ).local
        if candidate.treatments == ("inverse_sqrt",) and candidate.scope == "both"
    )
    manifest = _approved_manifest().extend(local, APPROVAL)
    probe = local[0]
    correction = replace(probe, horizon_epochs=11, cap_epochs=80, attempt=1)
    evidence = {
        probe.run_name: ArtifactEvidence(
            "complete",
            metadata={
                "stopped_epoch": 11,
                "early_stopped": True,
                "horizon_calibration_status": "recalibrate_horizon",
                "next_lr_schedule_horizon_epochs": 11,
            },
        ),
        correction.run_name: ArtifactEvidence("missing"),
    }

    with pytest.raises(SelectionEvidenceError, match="required correction is missing"):
        _ledger(manifest, scores=scores, evidence=evidence)

    evidence[correction.run_name] = _calibrated(correction, 3.0, 3.0)
    ledger = _ledger(manifest, scores=scores, evidence=evidence)
    entry = next(entry for entry in ledger.entries if entry.initial == probe)
    assert entry.current == correction


def test_completed_local_probes_are_not_reemitted() -> None:
    initial = next(
        candidate
        for candidate in initial_candidates()
        if candidate.treatments == ("inverse_sqrt",)
        and candidate.scope == "both"
        and candidate.joint_fraction != 0.05
    )
    scores = {initial.run_name: (2.0, 2.0)}
    first = plan_next_probes(_ledger(scores=scores), _approved_manifest(), APPROVAL)
    local = tuple(
        candidate
        for candidate in first.local
        if candidate.treatments == ("inverse_sqrt",) and candidate.scope == "both"
    )
    manifest = _approved_manifest().extend(local, APPROVAL)

    second = plan_next_probes(_ledger(manifest, scores=scores), manifest)

    assert not any(
        candidate.treatments == ("inverse_sqrt",) and candidate.scope == "both"
        for candidate in second.local
    )


def test_outer_boundary_winner_is_unresolved_and_never_triggers_another_round() -> None:
    linear = _candidate("linear", "both", 0.003)
    scores = {linear.run_name: (2.0, 2.0)}
    first = plan_next_probes(_ledger(scores=scores), _approved_manifest(), APPROVAL)
    boundary = tuple(
        candidate
        for candidate in first.boundary
        if candidate.treatments == ("linear",) and candidate.scope == "both"
    )
    manifest = _approved_manifest().extend(boundary, APPROVAL)
    outer = min(boundary, key=lambda candidate: candidate.deep_lr)
    scores[outer.run_name] = (3.0, 3.0)

    with pytest.raises(BoundaryOutcomeApprovalRequired) as raised:
        plan_next_probes(_ledger(manifest, scores=scores), manifest)

    assert outer in raised.value.candidates
    with pytest.raises(BoundaryOutcomeApprovalRequired):
        select_final_winners(_ledger(manifest, scores=scores), manifest)


def test_selection_cli_returns_outer_boundary_winner_for_explicit_approval(
    tmp_path, monkeypatch, capsys
) -> None:
    linear = _candidate("linear", "both", 0.003)
    scores = {linear.run_name: (2.0, 2.0)}
    first = plan_next_probes(_ledger(scores=scores), _approved_manifest())
    boundary = tuple(
        candidate
        for candidate in first.boundary
        if candidate.treatments == ("linear",) and candidate.scope == "both"
    )
    manifest = _approved_manifest().extend(boundary, APPROVAL)
    outer = min(boundary, key=lambda candidate: candidate.deep_lr)
    scores[outer.run_name] = (3.0, 3.0)
    path = tmp_path / "rq5_scheduler_candidate_manifest.json"
    path.write_text(manifest.freeze())
    monkeypatch.setattr(
        "experiments.g1_sasrec_item_ids_likes.analysis."
        "rq5_scheduler_selection._REPO_ROOT",
        tmp_path,
    )
    monkeypatch.setattr(
        "experiments.g1_sasrec_item_ids_likes.analysis."
        "rq5_scheduler_selection.selection_filesystem_inspector",
        lambda logs: _inspector(scores),
    )

    result = selection_main(["--logs", str(tmp_path / "logs"), "--manifest", str(path)])
    output = capsys.readouterr()

    assert result == 2
    assert output.out == ""
    assert outer.run_name in output.err
    assert "explicit user approval" in output.err


def test_joint_boundary_grid_is_strictly_new_and_stops_after_one_round() -> None:
    initial = next(
        candidate
        for candidate in initial_candidates()
        if candidate.treatments == ("inverse_sqrt",)
        and candidate.scope == "both"
        and candidate.joint_fraction != 0.05
    )
    scores = {initial.run_name: (2.0, 2.0)}
    first = plan_next_probes(_ledger(scores=scores), _approved_manifest(), APPROVAL)
    local = tuple(
        candidate
        for candidate in first.local
        if candidate.treatments == ("inverse_sqrt",) and candidate.scope == "both"
    )
    local_manifest = _approved_manifest().extend(local, APPROVAL)
    second = plan_next_probes(_ledger(local_manifest, scores=scores), local_manifest)
    boundary = tuple(
        candidate
        for candidate in second.boundary
        if candidate.treatments == ("inverse_sqrt",) and candidate.scope == "both"
    )

    assert all(candidate.timescale_fraction != 0.0125 for candidate in boundary)
    assert min(candidate.timescale_fraction for candidate in boundary) == 0.003125

    manifest = local_manifest.extend(boundary, APPROVAL)
    outer = min(boundary, key=lambda candidate: candidate.timescale_fraction)
    scores[outer.run_name] = (3.0, 3.0)
    with pytest.raises(BoundaryOutcomeApprovalRequired) as raised:
        plan_next_probes(_ledger(manifest, scores=scores), manifest)

    assert outer in raised.value.candidates


def test_central_joint_winner_does_not_trigger_local_trials() -> None:
    plan = plan_next_probes(_ledger(), _approved_manifest())

    assert plan.local == ()


def test_final_selection_refuses_an_incomplete_probe_surface() -> None:
    winner = _candidate("linear", "both", 0.003)
    manifest = replace(_approved_manifest(), approval=APPROVAL)

    with pytest.raises(FinalSelectionIncomplete, match="probe stages remain"):
        select_final_winners(_ledger(scores={winner.run_name: (2.0, 2.0)}), manifest)


def test_final_selection_returns_only_after_every_slot_is_resolved() -> None:
    exponential = _candidate("exponential", "both", 0.003)
    scores = {exponential.run_name: (1.0, 1.0)}
    manifest = _approved_manifest()
    probes = plan_next_probes(_ledger(scores=scores), manifest).candidates
    assert probes
    assert manifest.approval is not None
    manifest = manifest.extend(probes, manifest.approval)

    winners = select_final_winners(_ledger(manifest=manifest, scores=scores), manifest)

    assert len(winners) == 23


@pytest.mark.parametrize("kind", ["missing", "in_flight", "recoverable"])
def test_selection_refuses_missing_inflight_or_malformed_correction(kind: str) -> None:
    initial = _candidate("linear", "both", 0.003)
    correction = replace(initial, horizon_epochs=8, cap_epochs=8, attempt=1)
    evidence = {
        initial.run_name: ArtifactEvidence(
            "complete",
            metadata={
                "stopped_epoch": 8,
                "early_stopped": True,
                "horizon_calibration_status": "shorten_horizon",
                "next_lr_schedule_horizon_epochs": 8,
            },
        ),
        correction.run_name: ArtifactEvidence(kind),
    }

    with pytest.raises(SelectionEvidenceError, match=f"required correction is {kind}"):
        _ledger(evidence=evidence)


def test_exhausted_candidate_is_explicit_in_ledger_but_blocks_probe_planning() -> None:
    initial = _candidate("linear", "both", 0.003)
    attempt_one = replace(initial, horizon_epochs=8, cap_epochs=8, attempt=1)
    attempt_two = replace(initial, horizon_epochs=4, cap_epochs=4, attempt=2)
    attempt_three = replace(initial, horizon_epochs=6, cap_epochs=6, attempt=3)
    attempt_four = replace(initial, horizon_epochs=7, cap_epochs=7, attempt=4)
    evidence = {
        initial.run_name: ArtifactEvidence(
            "complete",
            metadata={
                "stopped_epoch": 8,
                "early_stopped": True,
                "horizon_calibration_status": "shorten_horizon",
                "next_lr_schedule_horizon_epochs": 8,
            },
        ),
        attempt_one.run_name: ArtifactEvidence(
            "complete",
            metadata={
                "stopped_epoch": 4,
                "early_stopped": True,
                "horizon_calibration_status": "shorten_horizon",
                "next_lr_schedule_horizon_epochs": 4,
            },
        ),
        attempt_two.run_name: ArtifactEvidence(
            "complete",
            metadata={
                "stopped_epoch": 4,
                "early_stopped": False,
                "horizon_calibration_status": "extend_horizon",
                "next_lr_schedule_horizon_epochs": 6,
            },
        ),
        attempt_three.run_name: ArtifactEvidence(
            "complete",
            metadata={
                "stopped_epoch": 6,
                "early_stopped": False,
                "horizon_calibration_status": "extend_horizon",
                "next_lr_schedule_horizon_epochs": 9,
            },
        ),
        attempt_four.run_name: ArtifactEvidence(
            "complete",
            metadata={
                "stopped_epoch": 7,
                "early_stopped": False,
                "horizon_calibration_status": "extend_horizon",
                "next_lr_schedule_horizon_epochs": 11,
            },
        ),
    }
    ledger = _ledger(evidence=evidence)

    assert any(entry.exhausted for entry in ledger.entries)
    with pytest.raises(SelectionEvidenceError, match="exhausted"):
        plan_next_probes(ledger, _approved_manifest())


def test_approved_ineligible_candidate_is_auditable_but_never_selected() -> None:
    initial = _candidate("cosine_warmup5_cycles4", "both", 0.003)
    attempt_one = replace(initial, horizon_epochs=13, cap_epochs=13, attempt=1)
    attempt_two = replace(initial, horizon_epochs=20, cap_epochs=20, attempt=2)
    attempt_three = replace(initial, horizon_epochs=21, cap_epochs=21, attempt=3)

    def unresolved(
        candidate: Rq5Candidate,
        stopped: int,
        early_stopped: bool,
        status: str,
        next_value: int,
    ) -> ArtifactEvidence:
        return ArtifactEvidence(
            "complete",
            metadata={
                "stopped_epoch": stopped,
                "early_stopped": early_stopped,
                "horizon_calibration_status": status,
                "next_lr_schedule_horizon_epochs": next_value,
            },
            metrics={"recall@100": 100.0, "ndcg@100": 100.0},
            selection_metrics={"recall@100": 100.0, "ndcg@100": 100.0},
        )

    evidence = {
        initial.run_name: unresolved(initial, 13, True, "shorten_horizon", 13),
        attempt_one.run_name: unresolved(attempt_one, 13, False, "extend_horizon", 20),
        attempt_two.run_name: unresolved(attempt_two, 20, False, "extend_horizon", 30),
        attempt_three.run_name: unresolved(
            attempt_three, 17, True, "shorten_horizon", 17
        ),
    }
    ledger = _ledger(evidence=evidence)
    audit_entry = next(entry for entry in ledger.entries if entry.initial == initial)

    assert audit_entry.exhausted
    assert audit_entry.ineligible_exclusion
    winners = select_winners(ledger, SELECTION_APPROVAL)
    winner = winners[TreatmentSlot("cosine_warmup5_cycles4", "both")]
    assert winner.initial != initial


def test_all_terminal_exclusions_remain_auditable_and_out_of_ranking() -> None:
    ledger = _ledger()
    exclusions = {
        entry.initial.run_name for entry in ledger.entries if entry.ineligible_exclusion
    }
    ledger_surfaces = {entry.initial.run_name for entry in ledger.entries}

    assert exclusions == (
        set(initial_manifest().horizon_followup_approval.ineligible_initial_surfaces)
        & ledger_surfaces
    )
    assert all(
        entry.exhausted for entry in ledger.entries if entry.ineligible_exclusion
    )
    winners = select_winners(ledger, SELECTION_APPROVAL)
    assert not exclusions.intersection(
        winner.initial.run_name for winner in winners.values()
    )


def test_manifest_is_complete_stable_round_trippable_and_tamper_evident() -> None:
    manifest = initial_manifest()

    assert len(manifest.candidates) == 67
    assert sum(len(candidate.treatments) == 2 for candidate in manifest.candidates) == 2
    assert manifest.digest == initial_manifest().digest
    assert CandidateManifest.thaw(manifest.freeze()) == manifest
    assert manifest.approval == APPROVAL
    assert manifest.selection_approval == SELECTION_APPROVAL
    assert {
        "run_name",
        "treatments",
        "shape",
        "scope",
        "deep_lr",
        "horizon_epochs",
        "cap_epochs",
        "warmup_fraction",
        "timescale_fraction",
        "cycles",
        "attempt",
        "probe",
        "dataset_size",
        "seed",
        "embedding_lr",
    } <= set(manifest.payload()["candidates"][0])

    document = json.loads(manifest.freeze())
    document["candidates"][0]["deep_lr"] *= 2
    with pytest.raises(ValueError, match="digest mismatch"):
        CandidateManifest.thaw(json.dumps(document))


def test_selection_rejects_valid_manifest_with_swapped_horizon_approval(
    tmp_path,
) -> None:
    manifest = initial_manifest()
    approval = manifest.horizon_followup_approval
    assert approval is not None
    probe = replace(_candidate("linear", "both", 0.003), probe="future1")
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

    with pytest.raises(
        SelectionEvidenceError, match="canonical horizon follow-up approval"
    ):
        advance_candidate_manifest(
            path,
            lambda surfaces: pytest.fail("ledger resolution must not start"),
        )


@pytest.mark.parametrize(
    ("field", "error"),
    [
        ("approval", "canonical probe approval"),
        ("selection_approval", "canonical selection approval"),
    ],
)
def test_manifest_advance_rejects_forged_policy_reference_before_callbacks(
    tmp_path,
    field: str,
    error: str,
) -> None:
    manifest = initial_manifest()
    assert manifest.approval is not None
    probe = replace(
        _candidate("linear", "both", 0.003),
        deep_lr=0.0015,
        probe="b1lrlo1",
    )
    manifest = manifest.extend((probe,), manifest.approval)
    policy = getattr(manifest, field)
    assert policy is not None
    manifest = replace(
        manifest,
        **{field: replace(policy, reference="forged valid-digest approval")},
    )
    path = tmp_path / "manifest.json"
    path.write_text(manifest.freeze())
    assert CandidateManifest.thaw(path.read_text()) == manifest

    with pytest.raises(SelectionEvidenceError, match=error):
        advance_candidate_manifest(
            path,
            lambda surfaces: pytest.fail("ledger callback must not start"),
            inspect_surface=lambda candidate: pytest.fail(
                "artifact callback must not start"
            ),
        )


def test_repo_approval_record_pins_approved_rq5_policies() -> None:
    manifest = initial_manifest()

    assert manifest.approval is not None
    assert manifest.approval.reference == "User approved in chat on 2026-08-22"
    assert (
        manifest.approval.local_factor,
        manifest.approval.boundary_extension,
        manifest.approval.boundary_points,
    ) == (2.0, 4.0, 3)
    assert manifest.selection_approval is not None
    assert manifest.selection_approval.reference == (
        "User approved in chat on 2026-08-22"
    )
    assert manifest.selection_approval.metric_decimals == 4
    assert manifest.selection_approval.tie_breaker == "surface_run_name"
    assert manifest.horizon_followup_approval is not None
    assert manifest.horizon_followup_approval.reference == (
        "User approved initial corrections on 2026-08-22 and standing small "
        "follow-ups on 2026-08-23"
    )
    assert manifest.horizon_followup_approval.max_additional_attempts == 2
    assert manifest.horizon_followup_approval.midpoint_rounding == "lower"
    assert manifest.horizon_followup_approval.stagewise
    assert manifest.horizon_followup_approval.preserve_acceptance_tolerance
    assert manifest.horizon_followup_approval.final_resolution_reference == (
        "User approved exact initial resolutions and standing small follow-ups "
        "in chat on 2026-08-23"
    )
    assert manifest.horizon_followup_approval.max_final_attempts == 1
    assert len(manifest.horizon_followup_approval.approved_initial_surfaces) == 28
    assert (
        _candidate("linear", "both", 0.003).run_name
        in manifest.horizon_followup_approval.approved_initial_surfaces
    )
    assert manifest.horizon_followup_approval.final_attempt_horizons == {
        _candidate("cosine", "both", 0.003).run_name: 22,
        _candidate("exponential", "both", 0.006).run_name: 21,
        _candidate("cosine_warmup5_cycles2", "both", 0.003).run_name: 23,
        _candidate("cosine_warmup5_cycles2", "both", 0.012).run_name: 17,
        candidate_by_run(
            "g1_rq5_cosine_both_d0p024_pb1lrhi1_h15_cap15_a0_ts2_r1_500m"
        ).run_name: 20,
        candidate_by_run(
            "g1_rq5_cosine_deep_only_d0p048_pb1lrhi2_h15_cap15_a0_ts2_r1_500m"
        ).run_name: 22,
        candidate_by_run(
            "g1_rq5_cosine_warmup5_cycles1_both_d0p096_pb1lrhi3_"
            "h15_cap15_a0_ts2_r1_500m"
        ).run_name: 22,
        candidate_by_run(
            "g1_rq5_cosine_warmup5_cycles1_deep_only_d0p096_pb1lrhi3_"
            "h15_cap15_a0_ts2_r1_500m"
        ).run_name: 16,
        candidate_by_run(
            "g1_rq5_exponential_both_d0p048_pb1lrhi2_h18_cap18_a0_ts2_r1_500m"
        ).run_name: 28,
        candidate_by_run(
            "g1_rq5_exponential_deep_only_d0p096_pb1lrhi3_"
            "h18_cap18_a0_ts2_r1_500m"
        ).run_name: 14,
        candidate_by_run(
            "g1_rq5_polynomial_both_d0p048_pb1lrhi2_h15_cap15_a0_ts2_r1_500m"
        ).run_name: 30,
    }
    assert manifest.horizon_followup_approval.ineligible_initial_surfaces == tuple(
        sorted(
            (
                _candidate("cosine", "both", 0.003).run_name,
                candidate_by_run(
                    "g1_rq5_cosine_both_d0p024_pb1lrhi1_"
                    "h15_cap15_a0_ts2_r1_500m"
                ).run_name,
                candidate_by_run(
                    "g1_rq5_cosine_both_d0p096_pb1lrhi3_"
                    "h15_cap15_a0_ts2_r1_500m"
                ).run_name,
                candidate_by_run(
                    "g1_rq5_cosine_deep_only_d0p048_pb1lrhi2_"
                    "h15_cap15_a0_ts2_r1_500m"
                ).run_name,
                candidate_by_run(
                    "g1_rq5_cosine_warmup5_cycles1_both_d0p096_pb1lrhi3_"
                    "h15_cap15_a0_ts2_r1_500m"
                ).run_name,
                candidate_by_run(
                    "g1_rq5_cosine_warmup5_cycles1_deep_only_d0p096_pb1lrhi3_"
                    "h15_cap15_a0_ts2_r1_500m"
                ).run_name,
                _candidate("exponential", "both", 0.006).run_name,
                candidate_by_run(
                    "g1_rq5_exponential_deep_only_d0p096_pb1lrhi3_"
                    "h18_cap18_a0_ts2_r1_500m"
                ).run_name,
                _candidate("cosine_warmup5_cycles2", "both", 0.012).run_name,
                _candidate("cosine_warmup5_cycles4", "both", 0.003).run_name,
                candidate_by_run(
                    "g1_rq5_polynomial_both_d0p024_pb1lrhi1_"
                    "h15_cap15_a0_ts2_r1_500m"
                ).run_name,
                candidate_by_run(
                    "g1_rq5_polynomial_both_d0p048_pb1lrhi2_"
                    "h15_cap15_a0_ts2_r1_500m"
                ).run_name,
                candidate_by_run(
                    "g1_rq5_wsd_both_d0p048_pb1lrhi2_"
                    "h20_cap20_a0_ts2_r1_500m"
                ).run_name,
            )
        )
    )


def test_manifest_records_approval_and_every_added_surface_in_its_digest() -> None:
    winner = _candidate("linear", "both", 0.003)
    plan = plan_next_probes(
        _ledger(scores={winner.run_name: (2.0, 2.0)}),
        _approved_manifest(),
        APPROVAL,
    )
    extended = _approved_manifest().extend(plan.boundary, APPROVAL)

    assert extended.approval == APPROVAL
    assert extended.selection_approval == SELECTION_APPROVAL
    assert len(extended.candidates) == 67 + len(plan.boundary)
    assert extended.digest != _approved_manifest().digest
    assert CandidateManifest.thaw(extended.freeze()) == extended


def test_operational_advance_freezes_manifest_before_returning_candidates(
    tmp_path,
) -> None:
    winner = _candidate("linear", "both", 0.003)
    scores = {winner.run_name: (2.0, 2.0)}
    path = tmp_path / "rq5_scheduler_candidate_manifest.json"
    approved = replace(_approved_manifest(), approval=APPROVAL)
    path.write_text(approved.freeze())

    plan = advance_candidate_manifest(
        path,
        lambda surfaces: build_calibrated_ledger(_inspector(scores), surfaces),
    )
    frozen = CandidateManifest.thaw(path.read_text())

    assert plan.candidates
    assert {candidate.run_name for candidate in plan.candidates} <= {
        candidate.run_name for candidate in frozen.candidates
    }
    assert frozen.digest != approved.digest


def test_operational_retry_emits_only_manifested_probe_surfaces_without_evidence(
    tmp_path,
) -> None:
    winner = _candidate("linear", "both", 0.003)
    scores = {winner.run_name: (2.0, 2.0)}
    approved = replace(_approved_manifest(), approval=APPROVAL)
    probes = plan_next_probes(_ledger(scores=scores), approved).boundary
    manifest = approved.extend(probes, APPROVAL)
    path = tmp_path / "rq5_scheduler_candidate_manifest.json"
    path.write_text(manifest.freeze())
    missing = probes[1]

    plan = advance_candidate_manifest(
        path,
        lambda surfaces: pytest.fail("selection ran before manifested probes existed"),
        inspect_surface=lambda candidate: (
            ArtifactEvidence("missing")
            if candidate == missing
            else _calibrated(candidate, 0.1, 0.1)
        ),
    )

    assert plan.candidates == (missing,)
    assert CandidateManifest.thaw(path.read_text()) == manifest


def test_operational_retry_never_reemits_completed_manifested_probes(tmp_path) -> None:
    winner = _candidate("linear", "both", 0.003)
    exponential = _candidate("exponential", "both", 0.003)
    scores = {
        winner.run_name: (2.0, 2.0),
        exponential.run_name: (1.0, 1.0),
    }
    approved = replace(_approved_manifest(), approval=APPROVAL)
    probes = plan_next_probes(_ledger(scores=scores), approved).boundary
    manifest = approved.extend(probes, APPROVAL)
    path = tmp_path / "rq5_scheduler_candidate_manifest.json"
    path.write_text(manifest.freeze())

    plan = advance_candidate_manifest(
        path,
        lambda surfaces: build_calibrated_ledger(_inspector(scores), surfaces),
        inspect_surface=_inspector(scores),
    )

    assert plan.candidates == ()


def test_operational_advance_refuses_a_tampered_reused_manifest(tmp_path) -> None:
    path = tmp_path / "rq5_scheduler_candidate_manifest.json"
    document = json.loads(_approved_manifest().freeze())
    document["candidates"][0]["deep_lr"] *= 2
    path.write_text(json.dumps(document))

    with pytest.raises(ValueError, match="digest mismatch"):
        advance_candidate_manifest(
            path,
            lambda surfaces: build_calibrated_ledger(_inspector(), surfaces),
        )


def test_operational_advance_freezes_unapproved_initial_manifest_but_emits_nothing(
    tmp_path,
) -> None:
    path = tmp_path / "rq5_scheduler_candidate_manifest.json"
    unapproved = replace(initial_manifest(), selection_approval=None)
    path.write_text(unapproved.freeze())

    with pytest.raises(SelectionPolicyApprovalRequired):
        advance_candidate_manifest(
            path,
            lambda surfaces: build_calibrated_ledger(_inspector(), surfaces),
        )

    assert CandidateManifest.thaw(path.read_text()) == unapproved


def test_probe_budget_and_candidate_run_ceiling_are_enforced() -> None:
    plan = plan_next_probes(_ledger(), _approved_manifest())

    assert len(plan.boundary) <= 81
    assert len(plan.local) <= 12
    assert 67 + len(plan.candidates) <= 160
    assert (67 + len(plan.candidates)) * 3 <= 480
