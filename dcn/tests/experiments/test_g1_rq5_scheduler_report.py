from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.g1_sasrec_item_ids_likes.analysis.rq5_scheduler_candidates import (
    Rq5Candidate,
)
from experiments.g1_sasrec_item_ids_likes.analysis.rq5_scheduler_report import (
    Rq5ReportError,
    build_report_bundle,
    collect_report_bundle,
    write_report_bundle,
)
from experiments.g1_sasrec_item_ids_likes.analysis.rq5_scheduler_selection import (
    BoundaryOutcomeApprovalRequired,
    CalibratedLedger,
    CandidateManifest,
    CorrectionAttemptEvidence,
    LedgerEntry,
    SelectionEvidenceError,
    TreatmentSlot,
    initial_manifest,
    plan_next_probes,
)
from experiments.g1_sasrec_item_ids_likes.launchers import verify_artifact


COLLECT = (
    Path(__file__).resolve().parents[3]
    / "experiments/g1_sasrec_item_ids_likes/analysis/collect.py"
)


def _score(candidate: Rq5Candidate) -> float:
    if candidate.probe is not None:
        return 0.7
    if any(
        treatment in candidate.treatments
        for treatment in ("inverse_sqrt", "cosine_warmup_tuned")
    ):
        return (
            0.9
            if candidate.deep_lr == 0.006 and candidate.joint_fraction == 0.05
            else 0.8
        )
    return 0.9 if candidate.deep_lr == 0.006 else 0.8


def _ledger(
    manifest: CandidateManifest | None = None,
    *,
    validation: dict[str, tuple[float, float]] | None = None,
    final: dict[str, tuple[float, float]] | None = None,
) -> CalibratedLedger:
    manifest = initial_manifest() if manifest is None else manifest
    validation = {} if validation is None else validation
    final = {} if final is None else final
    exclusions = set(
        manifest.horizon_followup_approval.ineligible_initial_surfaces
        if manifest.horizon_followup_approval is not None
        else ()
    )
    entries = []
    for candidate in manifest.candidates:
        validation_recall, validation_ndcg = validation.get(
            candidate.run_name, (_score(candidate), _score(candidate) / 2)
        )
        final_recall, final_ndcg = final.get(
            candidate.run_name,
            (validation_recall / 10, validation_ndcg / 10),
        )
        excluded = candidate.run_name in exclusions
        stopped = candidate.horizon_epochs or 20
        metrics = {
            "recall@100": final_recall,
            "ndcg@100": final_ndcg,
            "recall@10": final_recall / 2,
            "ndcg@10": final_ndcg / 2,
            "coverage@100": 0.5,
        }
        selection_metrics = {
            "recall@100": validation_recall,
            "ndcg@100": validation_ndcg,
        }
        schedule = {
            "shape": candidate.shape,
            "optimizer_group_scope": candidate.scope,
            "warmup_fraction": candidate.warmup_fraction,
            "timescale_fraction": candidate.timescale_fraction,
            "cycles": candidate.cycles,
            "min_lr_fraction": 0.0,
        }
        metadata = {
            "batch_size": 1280,
            "best_epoch": max(1, stopped - 1),
            "stopped_epoch": stopped,
            "horizon_calibration_status": (
                "shorten_horizon" if excluded else "calibrated"
            ),
            "early_stopped": True,
            "lr_horizon_complete": not excluded,
            "epochs_trained": stopped,
            "optimizer_steps_per_epoch": 10,
            "optimizer_steps": stopped * 10,
            "lr_schedule_horizon_epochs": candidate.horizon_epochs,
            "lr_schedule_horizon_steps": (
                None if candidate.horizon_epochs is None else candidate.horizon_epochs * 10
            ),
            "lr_schedule_timescale_steps": (
                None
                if candidate.timescale_fraction is None
                else round(stopped * 10 * candidate.timescale_fraction)
            ),
            "next_lr_schedule_horizon_epochs": None,
            "embedding_learning_rate": 0.064,
            "deep_learning_rate": candidate.deep_lr,
            "transfer_invariants": {"lr_schedule": schedule},
        }
        factors = verify_artifact._expected_schedule_factors(metadata, schedule)
        assert factors is not None
        metadata["lr_group_traces"] = {
            "embedding": (
                [0.064] * stopped
                if candidate.scope == "deep_only"
                else [0.064 * factor for factor in factors]
            ),
            "deep": [candidate.deep_lr * factor for factor in factors],
        }
        for treatment in candidate.treatments:
            entries.append(
                LedgerEntry(
                    slot=TreatmentSlot(treatment, candidate.scope),
                    initial=candidate,
                    current=candidate,
                    metrics=metrics,
                    selection_metrics=selection_metrics,
                    exhausted=excluded,
                    ineligible_exclusion=excluded,
                    correction_chain=(
                        CorrectionAttemptEvidence(
                            candidate=candidate,
                            metadata=metadata,
                            metrics=metrics,
                            selection_metrics=selection_metrics,
                            strictly_eligible=not excluded,
                            optimizer_group_traces_verified=True,
                            calibration_status=metadata[
                                "horizon_calibration_status"
                            ],
                            terminal_state=(
                                "exhausted" if excluded else "calibrated"
                            ),
                        ),
                    ),
                )
            )
    return CalibratedLedger(tuple(entries))


def _candidate(
    manifest: CandidateManifest,
    treatment: str,
    scope: str,
    deep_lr: float,
) -> Rq5Candidate:
    return next(
        candidate
        for candidate in manifest.candidates
        if treatment in candidate.treatments
        and candidate.scope == scope
        and candidate.deep_lr == deep_lr
        and candidate.probe is None
    )


def _resolved(
    *,
    validation: dict[str, tuple[float, float]] | None = None,
    final: dict[str, tuple[float, float]] | None = None,
) -> tuple[CandidateManifest, CalibratedLedger]:
    manifest = initial_manifest()
    for _ in range(3):
        ledger = _ledger(manifest, validation=validation, final=final)
        plan = plan_next_probes(ledger, manifest)
        if not plan.candidates:
            return manifest, ledger
        manifest = manifest.extend(plan.candidates, manifest.approval)
    raise AssertionError("test surface did not resolve")


def test_bundle_uses_validation_to_select_and_full_user_metrics_to_display() -> None:
    initial = initial_manifest()
    low = _candidate(initial, "linear", "both", 0.003)
    central = _candidate(initial, "linear", "both", 0.006)
    high = _candidate(initial, "linear", "both", 0.012)
    validation = {
        low.run_name: (0.1000, 0.0500),
        central.run_name: (0.1400, 0.0700),
        high.run_name: (0.1300, 0.0900),
    }
    final = {central.run_name: (0.01234, 0.00567)}
    manifest, ledger = _resolved(validation=validation, final=final)

    bundle = build_report_bundle(ledger, manifest)

    assert len(bundle.winners) == 23
    assert bundle.manifest_digest == manifest.digest
    assert "| linear | both |" in bundle.reader_markdown
    assert "0.012" in bundle.reader_markdown
    assert "0.140" not in bundle.reader_markdown
    assert "validation recall@100" in bundle.tuning_markdown
    assert "full-user recall@100" in bundle.tuning_markdown


def test_reader_uses_report_thresholds_and_contains_only_heading_and_table() -> None:
    initial = initial_manifest()
    linear = _candidate(initial, "linear", "both", 0.006)
    constant = _candidate(initial, "constant", "both", 0.006)
    final = {
        constant.run_name: (0.1, 0.05),
        linear.run_name: (0.1025, 0.051),
    }
    manifest, ledger = _resolved(final=final)

    bundle = build_report_bundle(ledger, manifest)

    assert "+2% (0.102)" in bundle.reader_markdown
    assert '<span style="color: green">+2% (0.102)</span>' not in bundle.reader_markdown
    assert "manifest" not in bundle.reader_markdown.lower()
    assert "SHA-256" not in bundle.reader_markdown
    assert bundle.reader_markdown.count("## ") == 1
    assert "selected deep LR" not in bundle.reader_markdown
    assert "| schedule parameter |" in bundle.reader_markdown
    assert "embedding LR" not in bundle.reader_markdown
    assert "batch size" not in bundle.reader_markdown


def test_tuning_ledger_exposes_training_and_stopping_fields() -> None:
    manifest, ledger = _resolved()

    bundle = build_report_bundle(ledger, manifest)

    assert "| batch size |" in bundle.tuning_markdown
    assert "| best epoch |" in bundle.tuning_markdown
    assert "| stopped epoch |" in bundle.tuning_markdown
    assert bundle.tuning_markdown.count("## ") == 23


def test_bundle_keeps_four_initial_exclusions_auditable_but_not_selectable() -> None:
    manifest, ledger = _resolved()
    bundle = build_report_bundle(ledger, manifest)

    assert bundle.initial_surface_counts == {"eligible": 63, "excluded": 4}
    assert bundle.evidence["manifest_digest"] == manifest.digest
    assert len(bundle.evidence["ineligible_exclusions"]) == 4
    assert all(
        name not in bundle.reader_markdown
        for name in bundle.evidence["ineligible_exclusions"]
    )
    assert bundle.tuning_markdown.count("ineligible exclusion") == 0


def test_raw_evidence_contains_every_surface_chain_and_trace_summary() -> None:
    manifest, ledger = _resolved()

    bundle = build_report_bundle(ledger, manifest)

    surfaces = bundle.evidence["surfaces"]
    assert len(surfaces) == len(manifest.candidates)
    assert sum(surface["ineligible_exclusion"] for surface in surfaces) == 4
    assert sum(len(surface["selected_for"]) for surface in surfaces) == 23
    for surface in surfaces:
        assert surface["treatments"]
        assert surface["scope"] in {"both", "deep_only"}
        assert surface["embedding_learning_rate"] == 0.064
        assert surface["batch_size"] == 1280
        assert surface["attempts"]
        terminal = surface["attempts"][-1]
        assert terminal["terminal_state"] in {"calibrated", "exhausted"}
        assert terminal["best_epoch"] is not None
        assert set(terminal["optimizer_group_trace_verification"]) == {
            "verified",
            "cached_verification",
            "embedding",
            "deep",
        }
        assert terminal["strictly_eligible"] is not None
        assert terminal["lr_group_traces"]
        training = terminal["training_evidence"]
        assert training["optimizer_steps"] > 0
        assert training["optimizer_steps_per_epoch"] > 0
        assert "early_stopped" in training
        assert "lr_schedule_horizon_steps" in training
        assert "lr_schedule_timescale_steps" in training
    shared = bundle.evidence["shared_central_mappings"]
    assert len(shared) == 2
    assert {mapping["treatments"][0]["scope"] for mapping in shared} == {
        "both",
        "deep_only",
    }
    assert all(
        {item["treatment"] for item in mapping["treatments"]}
        == {"cosine_warmup5_cycles1", "cosine_warmup_tuned"}
        for mapping in shared
    )


def test_bundle_recomputes_and_rejects_a_false_optimizer_trace() -> None:
    manifest, ledger = _resolved()
    first = ledger.entries[0]
    bad_metadata = {
        **first.correction_chain[0].metadata,
        "lr_group_traces": {
            **first.correction_chain[0].metadata["lr_group_traces"],
            "deep": [
                first.correction_chain[0].metadata["deep_learning_rate"] * 0.5
            ]
            + first.correction_chain[0].metadata["lr_group_traces"]["deep"][1:],
        },
    }
    bad_attempt = replace(
        first.correction_chain[0],
        metadata=bad_metadata,
        optimizer_group_traces_verified=True,
    )
    bad_entry = replace(first, correction_chain=(bad_attempt,))

    with pytest.raises(Rq5ReportError, match="optimizer-group LR trace"):
        build_report_bundle(
            CalibratedLedger((bad_entry, *ledger.entries[1:])), manifest
        )


def test_bundle_does_not_trust_a_stale_false_cached_trace_flag() -> None:
    manifest, ledger = _resolved()
    first = ledger.entries[0]
    stale_attempt = replace(
        first.correction_chain[0], optimizer_group_traces_verified=False
    )
    stale_entry = replace(first, correction_chain=(stale_attempt,))

    bundle = build_report_bundle(
        CalibratedLedger((stale_entry, *ledger.entries[1:])), manifest
    )

    surface = next(
        surface
        for surface in bundle.evidence["surfaces"]
        if surface["surface_run_name"] == first.initial.run_name
    )
    verification = surface["attempts"][0]["optimizer_group_trace_verification"]
    assert verification["verified"] is True
    assert verification["cached_verification"] is False


def test_raw_evidence_preserves_every_attempt_in_a_correction_chain() -> None:
    manifest, ledger = _resolved()
    entry_index = next(
        index
        for index, entry in enumerate(ledger.entries)
        if entry.slot == TreatmentSlot("cosine", "deep_only")
        and entry.initial.deep_lr == 0.006
    )
    entry = ledger.entries[entry_index]
    terminal = entry.correction_chain[0]
    first_attempt = replace(
        terminal,
        calibration_status="shorten_horizon",
        terminal_state=None,
    )
    corrected_candidate = replace(
        terminal.candidate,
        horizon_epochs=15,
        cap_epochs=15,
        attempt=1,
    )
    corrected_metadata = {
        **terminal.metadata,
        "best_epoch": 12,
        "stopped_epoch": 15,
        "horizon_calibration_status": "calibrated",
    }
    corrected_attempt = replace(
        terminal,
        candidate=corrected_candidate,
        metadata=corrected_metadata,
    )
    corrected_entry = replace(
        entry,
        current=corrected_candidate,
        correction_chain=(first_attempt, corrected_attempt),
    )
    entries = list(ledger.entries)
    entries[entry_index] = corrected_entry

    bundle = build_report_bundle(CalibratedLedger(tuple(entries)), manifest)

    surface = next(
        surface
        for surface in bundle.evidence["surfaces"]
        if surface["surface_run_name"] == entry.initial.run_name
    )
    assert [attempt["terminal_state"] for attempt in surface["attempts"]] == [
        None,
        "calibrated",
    ]
    assert [attempt["attempt"] for attempt in surface["attempts"]] == [0, 1]


def test_reader_bolds_constant_when_it_coleads_on_full_user_metrics() -> None:
    initial = initial_manifest()
    constant = _candidate(initial, "constant", "both", 0.006)
    linear = _candidate(initial, "linear", "both", 0.006)
    validation = {linear.run_name: (0.99, 0.99)}
    final = {
        constant.run_name: (0.1, 0.05),
        linear.run_name: (0.101, 0.0505),
    }
    manifest, ledger = _resolved(validation=validation, final=final)

    reader = build_report_bundle(ledger, manifest).reader_markdown

    assert "| **constant** | **both** |" in reader
    assert "| **linear** |" not in reader


def test_reader_bolds_full_user_leader_when_constant_is_not_coleader() -> None:
    initial = initial_manifest()
    constant = _candidate(initial, "constant", "both", 0.006)
    linear = _candidate(initial, "linear", "both", 0.006)
    final = {
        constant.run_name: (0.1, 0.05),
        linear.run_name: (0.104, 0.052),
    }
    manifest, ledger = _resolved(final=final)

    reader = build_report_bundle(ledger, manifest).reader_markdown

    assert "| **linear** | **both** |" in reader
    assert "| **constant** |" not in reader


@pytest.mark.parametrize(
    ("field", "value"),
    (("dataset_size", "50m"), ("seed", 43), ("embedding_lr", 0.032)),
)
def test_bundle_rejects_non_native_rq5_candidates(field: str, value: object) -> None:
    manifest, ledger = _resolved()
    first = ledger.entries[0]
    bad_candidate = replace(first.initial, **{field: value})
    bad_entry = replace(first, initial=bad_candidate, current=bad_candidate)
    bad_ledger = CalibratedLedger((bad_entry, *ledger.entries[1:]))

    with pytest.raises(Rq5ReportError, match="native Yambda-500M"):
        build_report_bundle(bad_ledger, manifest)


def test_bundle_fails_closed_when_manifested_probe_is_missing() -> None:
    manifest = initial_manifest()
    initial_ledger = _ledger(manifest)
    linear_high = _candidate(manifest, "linear", "both", 0.012)
    scores = {linear_high.run_name: (1.1, 1.1)}
    probe_plan = plan_next_probes(
        _ledger(manifest, validation=scores), manifest
    )
    assert probe_plan.boundary
    expanded = manifest.extend(probe_plan.boundary, manifest.approval)

    with pytest.raises(SelectionEvidenceError, match="does not match"):
        build_report_bundle(initial_ledger, expanded)


def test_bundle_fails_closed_on_unresolved_outer_boundary() -> None:
    manifest, _ = _resolved()
    linear_high = _candidate(manifest, "linear", "both", 0.012)
    plan = plan_next_probes(
        _ledger(manifest, validation={linear_high.run_name: (1.1, 1.1)}), manifest
    )
    expanded = manifest.extend(plan.boundary, manifest.approval)
    outer = next(candidate for candidate in plan.boundary if candidate.probe.endswith("3"))

    with pytest.raises(BoundaryOutcomeApprovalRequired):
        build_report_bundle(
            _ledger(
                expanded,
                validation={
                    linear_high.run_name: (1.1, 1.1),
                    outer.run_name: (1.2, 1.2),
                },
            ),
            expanded,
        )


def test_collect_report_bundle_fails_closed_on_missing_corrections(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(initial_manifest().freeze())

    with pytest.raises(SelectionEvidenceError, match="required correction evidence"):
        collect_report_bundle(tmp_path / "logs", manifest_path)


def test_writer_emits_dedicated_ledger_reader_draft_and_evidence(tmp_path: Path) -> None:
    manifest, ledger = _resolved()
    bundle = build_report_bundle(ledger, manifest)

    paths = write_report_bundle(bundle, tmp_path / "scratchpad", tmp_path / "evidence")

    assert set(paths) == {"tuning", "reader", "evidence"}
    assert paths["tuning"].name == "rq5_scheduler_tuning_500m.md"
    assert paths["reader"].name == "rq5_scheduler_reader_500m.md"
    assert paths["evidence"].name == "rq5_scheduler_results.json"
    evidence = json.loads(paths["evidence"].read_text())
    assert evidence["manifest_digest"] == manifest.digest
    assert "conclusion" not in paths["reader"].read_text().lower()


def test_legacy_schedule_artifacts_no_longer_route_to_rq5() -> None:
    import runpy

    namespace = runpy.run_path(str(COLLECT))

    assert namespace["_manifest_identity"]("schedule_cosine") == (
        6,
        "cosine warmup",
    )
    assert namespace["_manifest_identity"]("schedule_linear") is None
    assert namespace["_report_identity"](
        "g1_rq5_linear_both_d0p006_h17_cap17_a0_ts2_r1_500m"
    ) is None


def test_collector_rq5_hook_writes_only_dedicated_native_500m_outputs(
    monkeypatch, tmp_path: Path
) -> None:
    import runpy

    namespace = runpy.run_path(str(COLLECT))
    sentinel = object()
    calls = []
    globals_ = namespace["write_rq5_reports"].__globals__
    monkeypatch.setitem(globals_, "GENERATED", tmp_path / "generated")
    monkeypatch.setitem(globals_, "EXPERIMENT", tmp_path / "experiment")
    monkeypatch.setattr(
        globals_["rq5_scheduler_report"],
        "collect_report_bundle",
        lambda logs, manifest: calls.append(("collect", logs, manifest)) or sentinel,
    )
    monkeypatch.setattr(
        globals_["rq5_scheduler_report"],
        "write_report_bundle",
        lambda bundle, scratchpad, evidence: calls.append(
            ("write", bundle, scratchpad, evidence)
        ) or {},
    )

    namespace["write_rq5_reports"]()

    assert calls[0][1] == tmp_path / "generated/logs"
    assert calls[1] == (
        "write",
        sentinel,
        tmp_path / "experiment/scratchpad",
        tmp_path / "experiment/evidence",
    )


def test_ordinary_compact_path_includes_dedicated_rq5_only_on_500m(
    monkeypatch,
) -> None:
    import runpy

    namespace = runpy.run_path(str(COLLECT))
    globals_ = namespace["render_compact_report"].__globals__
    calls = []
    monkeypatch.setitem(globals_, "load_report_runs", lambda *_: [])
    monkeypatch.setitem(globals_, "_metric_bands", lambda **_: {})
    monkeypatch.setitem(
        globals_,
        "_load_rq5_report_bundle",
        lambda: calls.append("500m")
        or SimpleNamespace(
            reader_markdown=(
                "## RQ5 — Which learning-rate scheduler works best?\n\n"
                "| scheduler | recall@100 |\n| --- | ---: |\n| constant | 0.1 |\n"
            )
        ),
    )

    report_500m = namespace["render_compact_report"]("500m")
    report_50m = namespace["render_compact_report"]("50m")

    assert "## RQ5 —" in report_500m
    assert "## RQ5 —" not in report_50m
    assert calls == ["500m"]


@pytest.mark.parametrize("dataset_size", ("50m", "500m"))
def test_compact_coverage_validator_uses_the_complete_generic_question_set(
    monkeypatch, dataset_size: str
) -> None:
    import runpy

    namespace = runpy.run_path(str(COLLECT))
    report_run = namespace["ReportRun"]
    runs = [
        report_run(
            name=f"rq{research_question}",
            configuration=f"rq{research_question}",
            dataset_size=dataset_size,
            research_question=research_question,
            method=f"method {research_question}",
            status="completed",
            metrics={},
            metadata={},
        )
        for research_question in namespace["REPORT_RQ_ORDER"]
        if research_question != 5
    ]
    globals_ = namespace["render_compact_report"].__globals__
    monkeypatch.setitem(globals_, "load_report_runs", lambda *_: runs)
    monkeypatch.setitem(globals_, "_metric_bands", lambda **_: {})
    if dataset_size == "500m":
        monkeypatch.setitem(
            globals_,
            "_load_rq5_report_bundle",
            lambda: SimpleNamespace(
                reader_markdown="## RQ5 — current\n\n| x |\n| --- |"
            ),
        )
    else:
        monkeypatch.setitem(
            globals_,
            "_load_rq5_report_bundle",
            lambda: pytest.fail("50M must not require native RQ5"),
        )

    def stop_after_validation(actual_dataset_size, grouped) -> None:
        assert actual_dataset_size == dataset_size
        assert set(grouped) == set(namespace["REPORT_RQ_ORDER"]) - {5}
        raise RuntimeError("coverage validator invoked")

    monkeypatch.setitem(globals_, "_validate_compact_coverage", stop_after_validation)

    with pytest.raises(RuntimeError, match="coverage validator invoked"):
        namespace["render_compact_report"](dataset_size)


def test_native_compact_report_fails_before_coverage_without_dedicated_rq5(
    monkeypatch,
) -> None:
    import runpy

    namespace = runpy.run_path(str(COLLECT))
    globals_ = namespace["render_compact_report"].__globals__
    monkeypatch.setitem(globals_, "load_report_runs", lambda *_: [])
    monkeypatch.setitem(globals_, "_metric_bands", lambda **_: {})
    monkeypatch.setitem(
        globals_,
        "_load_rq5_report_bundle",
        lambda: (_ for _ in ()).throw(Rq5ReportError("dedicated RQ5 unavailable")),
    )
    monkeypatch.setitem(
        globals_,
        "_validate_compact_coverage",
        lambda *_: pytest.fail("coverage cannot run without dedicated RQ5"),
    )

    with pytest.raises(Rq5ReportError, match="dedicated RQ5 unavailable"):
        namespace["render_compact_report"]("500m")


def test_native_compact_report_rejects_a_malformed_dedicated_rq5_table(
    monkeypatch,
) -> None:
    import runpy

    namespace = runpy.run_path(str(COLLECT))
    globals_ = namespace["render_compact_report"].__globals__
    monkeypatch.setitem(globals_, "load_report_runs", lambda *_: [])
    monkeypatch.setitem(globals_, "_metric_bands", lambda **_: {})
    monkeypatch.setitem(
        globals_,
        "_load_rq5_report_bundle",
        lambda: SimpleNamespace(reader_markdown="## RQ5 — missing table\n"),
    )

    with pytest.raises(ValueError, match="reader draft is malformed"):
        namespace["render_compact_report"]("500m")
