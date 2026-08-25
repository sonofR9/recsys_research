from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import tempfile

from experiments.g1_sasrec_item_ids_likes.analysis import reporting
from experiments.g1_sasrec_item_ids_likes.analysis.rq5_scheduler_candidates import (
    Rq5Candidate,
)
from experiments.g1_sasrec_item_ids_likes.analysis.rq5_scheduler_selection import (
    CalibratedLedger,
    CandidateManifest,
    CorrectionAttemptEvidence,
    LedgerEntry,
    TreatmentSlot,
    build_calibrated_ledger,
    selection_filesystem_inspector,
    select_final_winners,
)
from experiments.g1_sasrec_item_ids_likes.launchers import verify_artifact
_METRICS = (
    "recall@100",
    "ndcg@100",
    "recall@10",
    "ndcg@10",
    "coverage@100",
)
_TREATMENT_ORDER = (
    "constant",
    "linear",
    "cosine",
    "polynomial",
    "exponential",
    "step",
    "wsd",
    "inverse_sqrt",
    "cosine_warmup5_cycles1",
    "cosine_warmup5_cycles2",
    "cosine_warmup5_cycles4",
    "cosine_warmup_tuned",
)
_SCOPE_ORDER = {"both": 0, "deep_only": 1}
_REPORT_THRESHOLDS = {
    "recall@100": 0.003,
    "ndcg@100": 0.001,
    "recall@10": 0.003,
    "ndcg@10": 0.001,
    "coverage@100": 0.1,
}


class Rq5ReportError(RuntimeError):
    pass


@dataclass(frozen=True)
class Rq5ReportBundle:
    manifest_digest: str
    winners: dict[TreatmentSlot, LedgerEntry]
    initial_surface_counts: dict[str, int]
    tuning_markdown: str
    reader_markdown: str
    evidence: dict[str, object]


def build_report_bundle(
    ledger: CalibratedLedger,
    manifest: CandidateManifest,
) -> Rq5ReportBundle:
    _validate_native_surface(ledger, manifest)
    _validate_correction_chains(ledger)
    winners = select_final_winners(ledger, manifest)
    _validate_winners(winners)
    initial_counts = _initial_surface_counts(ledger, manifest)
    metric_bands = _REPORT_THRESHOLDS
    return Rq5ReportBundle(
        manifest_digest=manifest.digest,
        winners=winners,
        initial_surface_counts=initial_counts,
        tuning_markdown=_render_tuning_ledger(ledger, winners, manifest.digest),
        reader_markdown=_render_reader_table(winners, metric_bands),
        evidence=_build_evidence(ledger, winners, manifest, initial_counts),
    )


def collect_report_bundle(logs: Path, manifest_path: Path) -> Rq5ReportBundle:
    manifest = CandidateManifest.thaw(manifest_path.read_text())
    ledger = build_calibrated_ledger(
        selection_filesystem_inspector(logs),
        manifest.candidates,
        manifest.horizon_followup_approval,
    )
    return build_report_bundle(ledger, manifest)


def write_report_bundle(
    bundle: Rq5ReportBundle,
    scratchpad: Path,
    evidence: Path,
) -> dict[str, Path]:
    paths = {
        "tuning": scratchpad / "rq5_scheduler_tuning_500m.md",
        "reader": scratchpad / "rq5_scheduler_reader_500m.md",
        "evidence": evidence / "rq5_scheduler_results.json",
    }
    _write_atomically(paths["tuning"], bundle.tuning_markdown)
    _write_atomically(paths["reader"], bundle.reader_markdown)
    _write_atomically(
        paths["evidence"],
        json.dumps(bundle.evidence, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    return paths


def _validate_native_surface(
    ledger: CalibratedLedger,
    manifest: CandidateManifest,
) -> None:
    candidates = [
        *(entry.initial for entry in ledger.entries),
        *(entry.current for entry in ledger.entries),
        *manifest.candidates,
    ]
    invalid = [
        candidate.run_name
        for candidate in candidates
        if candidate.dataset_size != "500m"
        or candidate.seed != 42
        or not math.isclose(candidate.embedding_lr, 0.064)
    ]
    if invalid:
        raise Rq5ReportError(
            "RQ5 report accepts only native Yambda-500M seed-42 candidates with "
            f"embedding LR 0.064: {sorted(invalid)}"
        )
    initial_names = {
        candidate.run_name
        for candidate in manifest.candidates
        if candidate.probe is None
    }
    if len(initial_names) != 67:
        raise Rq5ReportError(
            "RQ5 manifest must retain all 67 initial candidate surfaces"
        )
    if len(ledger.entries) < 69 or len(ledger.slots) != 23:
        raise Rq5ReportError("RQ5 report requires all 23 treatment/scope slots")


def _validate_winners(winners: dict[TreatmentSlot, LedgerEntry]) -> None:
    expected = {
        TreatmentSlot(treatment, scope)
        for treatment in _TREATMENT_ORDER
        for scope in (("both",) if treatment == "constant" else ("both", "deep_only"))
    }
    if set(winners) != expected:
        raise Rq5ReportError("RQ5 winner set is not the exact 23-slot comparison")
    for entry in winners.values():
        _required_metrics(entry.metrics, entry.current.run_name)
        _selection_metrics(entry)


def _validate_correction_chains(ledger: CalibratedLedger) -> None:
    checked = set()
    for entry in ledger.entries:
        if entry.initial.run_name in checked:
            continue
        checked.add(entry.initial.run_name)
        if not entry.correction_chain:
            raise Rq5ReportError(
                f"{entry.initial.run_name}: correction-chain evidence is absent"
            )
        for attempt in entry.correction_chain:
            if not _recompute_trace_verification(attempt.metadata):
                raise Rq5ReportError(
                    f"{attempt.candidate.run_name}: optimizer-group LR trace "
                    "verification failed"
                )


def _initial_surface_counts(
    ledger: CalibratedLedger,
    manifest: CandidateManifest,
) -> dict[str, int]:
    initial = {
        candidate.run_name: candidate
        for candidate in manifest.candidates
        if candidate.probe is None
    }
    excluded = {
        entry.initial.run_name
        for entry in ledger.entries
        if entry.initial.probe is None and entry.ineligible_exclusion
    }
    if len(initial) != 67 or len(excluded) != 4 or not excluded <= set(initial):
        raise Rq5ReportError(
            "RQ5 initial audit must contain 63 eligible and four approved exclusions"
        )
    return {"eligible": len(initial) - len(excluded), "excluded": len(excluded)}


def _render_tuning_ledger(
    ledger: CalibratedLedger,
    winners: dict[TreatmentSlot, LedgerEntry],
    manifest_digest: str,
) -> str:
    sections = [
        "# RQ5 — scheduler tuning on native Yambda-500M",
        "",
        "Selection uses best-epoch validation recall@100, then same-epoch validation "
        "NDCG@100. Displayed result metrics are full-user metrics.",
        f"Candidate manifest SHA-256: `{manifest_digest}`.",
    ]
    for slot in sorted(ledger.slots, key=_slot_order):
        sections += [
            "",
            f"## {_treatment_label(slot.treatment)} — {_scope_label(slot.scope)}",
            "",
            "| embedding LR | deep LR | batch size | schedule parameter | horizon "
            "| best epoch | stopped epoch | validation recall@100 | "
            "validation ndcg@100 | full-user recall@100 | full-user ndcg@100 | "
            "recall@10 | ndcg@10 | coverage@100 |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: "
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
        eligible = [
            entry
            for entry in ledger.for_slot(slot)
            if not entry.ineligible_exclusion
        ]
        for entry in sorted(eligible, key=_entry_order):
            candidate = entry.current
            validation_recall, validation_ndcg = _selection_metrics(entry)
            metrics = _required_metrics(entry.metrics, candidate.run_name)
            metadata = _terminal_metadata(entry)
            values = [
                _number(candidate.embedding_lr),
                _number(candidate.deep_lr),
                str(_required_int(metadata, "batch_size", candidate.run_name)),
                _schedule_parameter(candidate),
                (
                    "—"
                    if candidate.horizon_epochs is None
                    else str(candidate.horizon_epochs)
                ),
                str(_required_int(metadata, "best_epoch", candidate.run_name)),
                str(_required_int(metadata, "stopped_epoch", candidate.run_name)),
                f"{validation_recall:.4f}",
                f"{validation_ndcg:.4f}",
                *[reporting.absolute(metrics[metric]) for metric in _METRICS],
            ]
            if winners[entry.slot] == entry:
                values = [f"**{value}**" for value in values]
            sections.append("| " + " | ".join(values) + " |")
    return "\n".join(sections) + "\n"


def _render_reader_table(
    winners: dict[TreatmentSlot, LedgerEntry], metric_bands: dict[str, float]
) -> str:
    constant = winners[TreatmentSlot("constant", "both")]
    constant_metrics = _required_metrics(constant.metrics, constant.current.run_name)
    recommended = _recommended_reader_entry(winners, metric_bands)
    lines = [
        "## RQ5 — Which learning-rate scheduler works best?",
        "",
        "| scheduler | optimizer groups scheduled | schedule parameter | "
        "recall@100 | ndcg@100 | recall@10 | ndcg@10 | "
        "coverage@100 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for slot in sorted(winners, key=_slot_order):
        entry = winners[slot]
        metrics = _required_metrics(entry.metrics, entry.current.run_name)
        cells = [
            _treatment_label(slot.treatment),
            _scope_label(slot.scope),
            _schedule_parameter(entry.current),
        ]
        for metric in _METRICS:
            if slot.treatment == "constant":
                cells.append(reporting.absolute(metrics[metric]))
            else:
                cells.append(
                    _empirical_change_cell(
                        metrics[metric],
                        constant_metrics[metric],
                        metric,
                        metric_bands,
                    )
                )
        if entry == recommended:
            cells = [f"**{cell}**" for cell in cells]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _build_evidence(
    ledger: CalibratedLedger,
    winners: dict[TreatmentSlot, LedgerEntry],
    manifest: CandidateManifest,
    initial_counts: dict[str, int],
) -> dict[str, object]:
    exclusions = sorted(
        {
            entry.initial.run_name
            for entry in ledger.entries
            if entry.ineligible_exclusion
        }
    )
    return {
        "dataset_size": "500m",
        "seed": 42,
        "embedding_learning_rate": 0.064,
        "manifest_digest": manifest.digest,
        "initial_surfaces": initial_counts,
        "treatment_slots": len(winners),
        "shared_central_mappings": _shared_central_mappings(ledger),
        "ineligible_exclusions": exclusions,
        "surfaces": _surface_evidence(ledger, winners, manifest),
        "winners": _winner_evidence(winners),
    }


def _winner_evidence(
    winners: dict[TreatmentSlot, LedgerEntry],
) -> list[dict[str, object]]:
    records = []
    for slot, entry in sorted(
        winners.items(), key=lambda item: _slot_order(item[0])
    ):
        metrics = _required_metrics(entry.metrics, entry.current.run_name)
        records.append(
            {
                "treatment": slot.treatment,
                "scope": slot.scope,
                "surface_run_name": entry.initial.run_name,
                "artifact_run_name": entry.current.run_name,
                "validation_metrics": dict(entry.selection_metrics or {}),
                "full_user_metrics": {
                    metric: metrics[metric] for metric in _METRICS
                },
            }
        )
    return records


def _surface_evidence(
    ledger: CalibratedLedger,
    winners: dict[TreatmentSlot, LedgerEntry],
    manifest: CandidateManifest,
) -> list[dict[str, object]]:
    entries_by_surface: dict[str, list[LedgerEntry]] = {}
    for entry in ledger.entries:
        entries_by_surface.setdefault(entry.initial.run_name, []).append(entry)
    records = []
    for candidate in manifest.candidates:
        entries = entries_by_surface.get(candidate.run_name, [])
        if not entries:
            raise Rq5ReportError(
                f"{candidate.run_name}: manifest surface is absent from evidence"
            )
        first = entries[0]
        if any(entry.correction_chain != first.correction_chain for entry in entries[1:]):
            raise Rq5ReportError(
                f"{candidate.run_name}: shared treatment correction chains disagree"
            )
        terminal_metadata = _terminal_metadata(first)
        selected_for = [
            {"treatment": slot.treatment, "scope": slot.scope}
            for slot, winner in sorted(
                winners.items(), key=lambda item: _slot_order(item[0])
            )
            if winner.initial.run_name == candidate.run_name
        ]
        record = {
            "surface_run_name": candidate.run_name,
            **_candidate_evidence(candidate),
            "batch_size": _required_int(
                terminal_metadata, "batch_size", candidate.run_name
            ),
            "ineligible_exclusion": first.ineligible_exclusion,
            "selected_winner": bool(selected_for),
            "selected_for": selected_for,
            "attempts": [
                _attempt_evidence(attempt) for attempt in first.correction_chain
            ],
        }
        records.append(record)
    return records


def _candidate_evidence(candidate: Rq5Candidate) -> dict[str, object]:
    return {
        "treatments": list(candidate.treatments),
        "scope": candidate.scope,
        "deep_learning_rate": candidate.deep_lr,
        "embedding_learning_rate": candidate.embedding_lr,
        "schedule": {
            "shape": candidate.shape,
            "warmup_fraction": candidate.warmup_fraction,
            "timescale_fraction": candidate.timescale_fraction,
            "cycles": candidate.cycles,
        },
        "horizon_epochs": candidate.horizon_epochs,
        "cap_epochs": candidate.cap_epochs,
        "attempt": candidate.attempt,
        "probe": candidate.probe,
    }


def _attempt_evidence(attempt: CorrectionAttemptEvidence) -> dict[str, object]:
    metadata = attempt.metadata
    training_keys = (
        "optimizer_steps",
        "optimizer_steps_per_epoch",
        "epochs_trained",
        "lr_schedule_horizon_epochs",
        "lr_schedule_horizon_steps",
        "lr_schedule_timescale_steps",
        "early_stopped",
        "lr_horizon_complete",
        "horizon_calibration_status",
        "next_lr_schedule_horizon_epochs",
    )
    return {
        "artifact_run_name": attempt.candidate.run_name,
        **_candidate_evidence(attempt.candidate),
        "batch_size": _required_int(
            metadata, "batch_size", attempt.candidate.run_name
        ),
        "best_epoch": metadata.get("best_epoch"),
        "stopped_epoch": _required_int(
            metadata, "stopped_epoch", attempt.candidate.run_name
        ),
        "calibration_status": attempt.calibration_status,
        "terminal_state": attempt.terminal_state,
        "validation_metrics": dict(attempt.selection_metrics or {}),
        "full_user_metrics": dict(attempt.metrics or {}),
        "strictly_eligible": attempt.strictly_eligible,
        "lr_group_traces": metadata.get("lr_group_traces"),
        "training_evidence": {key: metadata.get(key) for key in training_keys},
        "optimizer_group_trace_verification": _trace_verification_summary(
            metadata, attempt.optimizer_group_traces_verified
        ),
    }


def _trace_verification_summary(
    metadata: dict, optimizer_group_traces_verified: bool
) -> dict[str, object]:
    traces = metadata.get("lr_group_traces")
    summaries = {}
    valid = isinstance(traces, dict) and set(traces) == {"embedding", "deep"}
    for group in ("embedding", "deep"):
        trace = traces.get(group) if isinstance(traces, dict) else None
        if not isinstance(trace, list) or not trace or any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            for value in trace
        ):
            valid = False
            summaries[group] = None
            continue
        values = [float(value) for value in trace]
        summaries[group] = {
            "epochs": len(values),
            "first": values[0],
            "last": values[-1],
            "minimum": min(values),
            "maximum": max(values),
            "constant": all(math.isclose(value, values[0]) for value in values),
        }
    return {
        "verified": bool(valid and _recompute_trace_verification(metadata)),
        "cached_verification": optimizer_group_traces_verified,
        **summaries,
    }


def _recompute_trace_verification(metadata: dict) -> bool:
    invariants = metadata.get("transfer_invariants")
    schedule = invariants.get("lr_schedule") if isinstance(invariants, dict) else None
    return bool(
        isinstance(schedule, dict)
        and verify_artifact._valid_group_lr_traces(metadata, schedule)
    )


def _shared_central_mappings(ledger: CalibratedLedger) -> list[dict[str, object]]:
    grouped: dict[str, list[LedgerEntry]] = {}
    for entry in ledger.entries:
        if len(entry.initial.treatments) == 2:
            grouped.setdefault(entry.current.run_name, []).append(entry)
    mappings = []
    for artifact, entries in sorted(grouped.items()):
        mappings.append(
            {
                "artifact_run_name": artifact,
                "treatments": [
                    {"treatment": entry.slot.treatment, "scope": entry.slot.scope}
                    for entry in sorted(entries, key=lambda item: _slot_order(item.slot))
                ],
            }
        )
    if len(mappings) != 2 or any(len(mapping["treatments"]) != 2 for mapping in mappings):
        raise Rq5ReportError("RQ5 shared-central mapping is incomplete")
    return mappings


def _terminal_metadata(entry: LedgerEntry) -> dict:
    if not entry.correction_chain:
        raise Rq5ReportError(
            f"{entry.initial.run_name}: correction-chain evidence is absent"
        )
    terminal = entry.correction_chain[-1]
    if terminal.terminal_state not in {"calibrated", "exhausted"}:
        raise Rq5ReportError(
            f"{entry.initial.run_name}: correction chain lacks a terminal state"
        )
    return terminal.metadata


def _required_int(metadata: dict, key: str, run_name: str) -> int:
    value = metadata.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise Rq5ReportError(f"{run_name}: invalid {key}")
    return value


def _recommended_reader_entry(
    winners: dict[TreatmentSlot, LedgerEntry], metric_bands: dict[str, float]
) -> LedgerEntry:
    ranked = sorted(
        winners.values(),
        key=lambda entry: (
            -_required_metrics(entry.metrics, entry.current.run_name)["recall@100"],
            -_required_metrics(entry.metrics, entry.current.run_name)["ndcg@100"],
            entry.initial.run_name,
        ),
    )
    leader = ranked[0]
    constant = winners[TreatmentSlot("constant", "both")]
    leader_metrics = _required_metrics(leader.metrics, leader.current.run_name)
    constant_metrics = _required_metrics(constant.metrics, constant.current.run_name)
    if (
        leader_metrics["recall@100"] - constant_metrics["recall@100"]
        <= metric_bands["recall@100"]
        and leader_metrics["ndcg@100"] - constant_metrics["ndcg@100"]
        <= metric_bands["ndcg@100"]
    ):
        return constant
    return leader


def _empirical_change_cell(
    value: float,
    reference: float,
    metric: str,
    metric_bands: dict[str, float],
) -> str:
    difference = value - reference
    percent = 100 * difference / reference
    rendered = "0%" if round(percent) == 0 else f"{percent:+.0f}%"
    cell = f"{rendered} ({reporting.absolute(value)})"
    if abs(difference) <= metric_bands[metric]:
        return cell
    color = "green" if difference > 0 else "red"
    return f'<span style="color: {color}">{cell}</span>'


def _required_metrics(
    metrics: dict[str, float] | None, run_name: str
) -> dict[str, float]:
    if not isinstance(metrics, dict):
        raise Rq5ReportError(f"{run_name}: missing full-user metrics")
    values = {metric: metrics.get(metric) for metric in _METRICS}
    if any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        for value in values.values()
    ):
        raise Rq5ReportError(f"{run_name}: incomplete full-user metric set")
    return {metric: float(value) for metric, value in values.items()}


def _selection_metrics(entry: LedgerEntry) -> tuple[float, float]:
    metrics = entry.selection_metrics
    if not isinstance(metrics, dict):
        raise Rq5ReportError(f"{entry.current.run_name}: missing validation metrics")
    values = metrics.get("recall@100"), metrics.get("ndcg@100")
    if any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        for value in values
    ):
        raise Rq5ReportError(f"{entry.current.run_name}: invalid validation metrics")
    return float(f"{values[0]:.4f}"), float(f"{values[1]:.4f}")


def _entry_order(entry: LedgerEntry) -> tuple[object, ...]:
    return (
        _slot_order(entry.slot),
        entry.initial.probe is not None,
        entry.initial.probe or "",
        entry.initial.deep_lr,
        entry.initial.run_name,
    )


def _slot_order(slot: TreatmentSlot) -> tuple[int, int]:
    return _TREATMENT_ORDER.index(slot.treatment), _SCOPE_ORDER[slot.scope]


def _schedule_parameter(candidate: Rq5Candidate) -> str:
    timescale = candidate.timescale_fraction
    if timescale is not None:
        return f"timescale={_number(timescale)}"
    warmup = candidate.warmup_fraction
    cycles = candidate.cycles
    if warmup or cycles != 1:
        return f"warmup={_number(warmup)}, cycles={cycles}"
    return "—"


def _treatment_label(treatment: str) -> str:
    labels = {
        "inverse_sqrt": "inverse sqrt",
        "wsd": "WSD",
        "cosine_warmup5_cycles1": "cosine, warmup 5%, 1 cycle",
        "cosine_warmup5_cycles2": "cosine, warmup 5%, 2 cycles",
        "cosine_warmup5_cycles4": "cosine, warmup 5%, 4 cycles",
        "cosine_warmup_tuned": "cosine, tuned warmup",
    }
    return labels.get(treatment, treatment)


def _scope_label(scope: str) -> str:
    return "both" if scope == "both" else "deep only"


def _number(value: float) -> str:
    return f"{value:.12g}"


def _write_atomically(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
