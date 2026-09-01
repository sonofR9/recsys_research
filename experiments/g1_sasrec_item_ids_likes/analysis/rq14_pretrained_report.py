from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile

from experiments.g1_sasrec_item_ids_likes.analysis.rq14_pretrained_candidates import (
    DEEP_LRS,
    TREATMENTS,
    Rq14PretrainedCandidate,
    candidate_by_run,
    initial_candidates,
    make_boundary_candidate,
)
from experiments.g1_sasrec_item_ids_likes.analysis import reporting
from experiments.g1_sasrec_item_ids_likes.analysis.rq15_training_candidates import (
    selected_source_candidate,
)
from experiments.g1_sasrec_item_ids_likes.launchers import verify_artifact


class Rq14PretrainedReportError(RuntimeError):
    pass


_IMPLEMENTATION_FILES = (
    Path("dcn/config/query_retrieval_training.py"),
    Path("dcn/models/cross_attention_training.py"),
    Path("dcn/models/cross_attention_retrieval.py"),
    Path("dcn/models/history_tokens.py"),
    Path("experiments/g1_sasrec_item_ids_likes/analysis/rq14_pretrained_candidates.py"),
    Path("experiments/g1_sasrec_item_ids_likes/analysis/rq14_pretrained_audit.py"),
    Path("experiments/g1_sasrec_item_ids_likes/analysis/rq14_pretrained_report.py"),
    Path("experiments/g1_sasrec_item_ids_likes/configs/rq14_pretrained_query_variant.py"),
)
_EXPERIMENT = Path(__file__).parents[1]
_RQ14_CONFIG = _EXPERIMENT / "configs/rq14_pretrained_query_variant.py"
_RQ15_CONFIG = _EXPERIMENT / "configs/rq15_decoder_training_variant.py"
_METRICS = ("recall@100", "ndcg@100", "recall@10", "ndcg@10", "coverage@100")
_LESION_IMPLEMENTATION_FILES = (
    Path("dcn/config/query_retrieval_training.py"),
    Path("dcn/eval/callback.py"),
    Path("dcn/models/cross_attention_retrieval.py"),
    Path("dcn/models/history_tokens.py"),
    Path(
        "experiments/g1_sasrec_item_ids_likes/analysis/rq14_pretrained_lesion_candidates.py"
    ),
    Path(
        "experiments/g1_sasrec_item_ids_likes/analysis/rq14_pretrained_lesion_evidence.py"
    ),
    Path(
        "experiments/g1_sasrec_item_ids_likes/configs/rq14_pretrained_lesion_variant.py"
    ),
    Path(
        "experiments/g1_sasrec_item_ids_likes/launchers/architecture/rq14_pretrained_lesions_500m.sh"
    ),
)
_EXPECTED_EFFECTS = {
    "distinct_vs_shared_cls_only",
    "distinct_vs_shared_history",
    "history_vs_cls_only_shared",
    "history_vs_cls_only_distinct",
}
_AUDIT_CHECKS = {
    "treatment_recipes",
    "exact_checkpoint_load_scope",
    "candidate_only_loss_and_target_exclusion",
    "memory_ordering_and_gradients",
    "horizon_and_artifact_binding",
}
_RQ14_START = "<!-- rq14-pretrained-generated:start -->"
_RQ14_END = "<!-- rq14-pretrained-generated:end -->"
_RQ15_START = "<!-- rq15-training-generated:start -->"


@dataclass(frozen=True)
class PretrainedRun:
    candidate: Rq14PretrainedCandidate
    artifact_run_name: str
    reused_from_rq15: bool
    validation_recall: float
    validation_ndcg: float
    best_epoch: int
    stopped_epoch: int
    metrics: dict[str, float]
    expanded_examples_per_epoch: int
    candidate_targets_per_epoch: int
    ntp_targets_per_epoch: int
    input_tokens_per_epoch: int
    steady_state_targets_per_second: float
    time_through_best_seconds: float
    horizon_seconds: float
    artifact_sha256: dict[str, str]
    checkpoint_sha256: str


@dataclass(frozen=True)
class Rq14PretrainedReportBundle:
    reader_markdown: str
    readme_markdown: str
    tuning_markdown: str
    evidence: dict[str, object]


@dataclass(frozen=True)
class LesionDiagnostics:
    evidence: Mapping[str, object]
    explanation: Mapping[str, object]
    evidence_file_sha256: str
    explanation_file_sha256: str
    raw_artifacts_verified: bool


@dataclass(frozen=True)
class SourceCheckpointBinding:
    run_name: str
    checkpoint_path: str
    checkpoint_sha256: str
    verified: bool


def build_report_bundle(
    runs: Iterable[PretrainedRun],
    *,
    candidate_only_markdown: str,
    correctness_audit: Mapping[str, object] | None = None,
    lesion_diagnostics: LesionDiagnostics | None = None,
    source_checkpoint: SourceCheckpointBinding | None = None,
) -> Rq14PretrainedReportBundle:
    run_list = list(runs)
    by_candidate = {run.candidate.run_name: run for run in run_list}
    if len(by_candidate) != len(run_list):
        raise Rq14PretrainedReportError("duplicate pretrained RQ14 cell")
    if not candidate_only_markdown.strip():
        raise Rq14PretrainedReportError("candidate-only report cannot be empty")
    _validate_run_invariants(run_list)

    missing = [
        candidate.run_name
        for candidate in initial_candidates()
        if candidate.run_name not in by_candidate
    ]
    selected: dict[str, PretrainedRun] = {}
    boundaries: list[str] = []
    if not missing:
        for treatment in TREATMENTS:
            winner, followup = _resolve_surface(
                [run for run in run_list if run.candidate.treatment == treatment]
            )
            selected[treatment] = winner
            if followup is not None and followup.run_name not in by_candidate:
                boundaries.append(followup.run_name)
    elif any(run.candidate.stage == "lr_boundary" for run in run_list):
        raise Rq14PretrainedReportError("boundary artifact precedes the initial grid")

    required = [*missing, *boundaries]
    expected_artifacts = {
        run.artifact_run_name: run.artifact_sha256 for run in run_list
    }
    audit = _validate_audit(
        correctness_audit,
        expected_artifacts,
        run_list[0].checkpoint_sha256 if run_list else None,
    )
    unexpected = _unexpected_effects(selected) if not required else {}
    diagnostics = _validate_lesion_diagnostics(
        lesion_diagnostics,
        selected=selected,
        unexpected=unexpected,
        checkpoint_sha256=run_list[0].checkpoint_sha256 if run_list else None,
    )
    source = _validate_source_binding(source_checkpoint, run_list)
    if missing:
        claims_status = "pending_artifacts"
    elif boundaries:
        claims_status = "pending_boundary"
    elif audit["status"] != "passed":
        claims_status = "correctness_audit_required"
    elif source["status"] != "passed":
        claims_status = "source_checkpoint_required"
    elif unexpected and diagnostics["status"] != "passed":
        claims_status = "unexpected_result_requires_investigation"
    else:
        claims_status = "ready_for_user_validation"

    evidence: dict[str, object] = {
        "schema_version": 1,
        "research_question": "RQ14 pretrained decoder-decoder query memory",
        "dataset_size": "500m",
        "training_method": "NTP-pretrained then joint candidate-only fine-tuning",
        "embedding_lr": 0.00025,
        "deep_lr_surface": list(DEEP_LRS),
        "horizon_epochs": 20,
        "claims_status": claims_status,
        "result_claims_user_validated": False,
        "missing_initial_artifacts": missing,
        "required_boundary_followups": boundaries,
        "required_followups": required,
        "boundary_rule": "extend a winning outer deep LR geometrically by two",
        "reused_rq15_cells": [
            run.artifact_run_name for run in run_list if run.reused_from_rq15
        ],
        "new_cells": [
            run.artifact_run_name for run in run_list if not run.reused_from_rq15
        ],
        "checkpoint_sha256": run_list[0].checkpoint_sha256 if run_list else None,
        "source_checkpoint": source,
        "correctness_audit": audit,
        "lesion_diagnostics": diagnostics,
        "unexpected_effects": unexpected,
        "acceptance_criteria": [
            "The decoder which cross-attends both CLS tokens and history probably should be better.",
            "Four separate class tokens should probably have better metrics than the same CLS token.",
            "If the statements above do not hold true, first debug, and if everything is correct, explain why experimentally.",
        ],
        "selected": {
            treatment: _run_record(run) for treatment, run in selected.items()
        },
        "selected_method": (
            _run_record(_overall_selection(selected)) if selected else None
        ),
        "overall_rule": (
            "within 0.003 validation Recall@100 choose CLS-only memory, then shared tokens"
        ),
        "timing_definition": {
            "steady_state_targets_per_second": (
                "candidate targets divided by logged train time over epochs 2-20"
            ),
            "time_through_selected_checkpoint_seconds": (
                "logged train, validation inference, and validation save time through the selected checkpoint"
            ),
            "three_cell_tuning_gpu_seconds": (
                "sum of logged train, validation inference, and validation save GPU time across the treatment's three LR cells; not parallel batch wall time"
            ),
        },
        "treatments": {
            treatment: {
                "artifacts": [
                    _run_record(run)
                    for run in sorted(
                        (item for item in run_list if item.candidate.treatment == treatment),
                        key=lambda item: item.candidate.deep_lr,
                    )
                ]
            }
            for treatment in TREATMENTS
        },
    }
    pretrained = (
        _pretrained_reader(selected, run_list)
        if claims_status == "ready_for_user_validation"
        else ""
    )
    reader = (
        "# RQ14 decoder-decoder query memory\n\n"
        "## Historical candidate-only comparison\n\n"
        f"{candidate_only_markdown.strip()}\n"
        f"{pretrained}"
    )
    return Rq14PretrainedReportBundle(
        reader_markdown=reader,
        readme_markdown=_readme_markdown(reader, claims_status),
        tuning_markdown=_tuning_markdown(run_list, selected),
        evidence=evidence,
    )


def collect_report_bundle(
    logs: Path,
    *,
    rq15_results_path: Path,
    candidate_only_path: Path,
    correctness_path: Path | None = None,
    lesion_evidence_path: Path | None = None,
    lesion_explanation_path: Path | None = None,
) -> Rq14PretrainedReportBundle:
    rq15_results = _load_json(rq15_results_path)
    rq15_records = _rq15_reuse_records(rq15_results)
    source_candidate = selected_source_candidate(logs)
    source_checkpoint_path = source_candidate.checkpoint_path(logs).resolve()
    source_checkpoint_sha256 = _file_sha256(source_checkpoint_path)
    source_checkpoint = SourceCheckpointBinding(
        run_name=source_candidate.run_name,
        checkpoint_path=str(source_checkpoint_path),
        checkpoint_sha256=source_checkpoint_sha256,
        verified=True,
    )
    runs: list[PretrainedRun] = []
    for candidate in initial_candidates():
        if candidate.reused_rq15_run_name is not None:
            record = rq15_records.get(candidate.reused_rq15_run_name)
            if record is None:
                continue
            runs.append(
                _load_run(
                    logs / candidate.artifact_run_name,
                    candidate,
                    record,
                    source_checkpoint=source_checkpoint,
                )
            )
            continue
        directory = logs / candidate.run_name
        if _complete(directory):
            runs.append(
                _load_run(
                    directory,
                    candidate,
                    None,
                    source_checkpoint=source_checkpoint,
                )
            )
    for directory in sorted(logs.glob("g1_rq14_pretrained_*_500m")):
        if not _complete(directory):
            continue
        try:
            candidate = candidate_by_run(directory.name)
        except ValueError:
            continue
        if candidate.stage == "lr_boundary":
            runs.append(
                _load_run(
                    directory,
                    candidate,
                    None,
                    source_checkpoint=source_checkpoint,
                )
            )
    audit = None if correctness_path is None else _load_optional_json(correctness_path)
    diagnostics = _load_lesion_diagnostics(
        logs,
        evidence_path=lesion_evidence_path,
        explanation_path=lesion_explanation_path,
    )
    return build_report_bundle(
        runs,
        candidate_only_markdown=candidate_only_path.read_text(),
        correctness_audit=audit,
        lesion_diagnostics=diagnostics,
        source_checkpoint=source_checkpoint,
    )


def write_report_bundle(
    bundle: Rq14PretrainedReportBundle, *, evidence: Path, scratchpad: Path
) -> None:
    _write_json(evidence / "rq14_pretrained_results.json", bundle.evidence)
    _write_text(
        scratchpad / "rq14_pretrained_reader_500m.md", bundle.reader_markdown
    )
    _write_text(
        scratchpad / "rq14_pretrained_tuning_500m.md", bundle.tuning_markdown
    )


def current_implementation_sha256() -> dict[str, str]:
    return {str(path): _file_sha256(path) for path in _IMPLEMENTATION_FILES}


def current_lesion_implementation_sha256() -> dict[str, str]:
    return {
        str(path): _file_sha256(path) for path in _LESION_IMPLEMENTATION_FILES
    }


def _validate_run_invariants(runs: list[PretrainedRun]) -> None:
    checkpoint_hashes = {run.checkpoint_sha256 for run in runs}
    if len(checkpoint_hashes) > 1 or any(not _sha256(value) for value in checkpoint_hashes):
        raise Rq14PretrainedReportError("all cells must load the same exact checkpoint")
    artifact_names = [run.artifact_run_name for run in runs]
    if len(set(artifact_names)) != len(artifact_names):
        raise Rq14PretrainedReportError("one artifact cannot supply multiple cells")
    for run in runs:
        if (
            run.candidate.embedding_lr != 0.00025
            or run.stopped_epoch != 20
            or run.ntp_targets_per_epoch != 0
            or run.best_epoch not in range(1, 21)
        ):
            raise Rq14PretrainedReportError(
                f"{run.artifact_run_name}: incompatible training semantics"
            )


def _resolve_surface(
    runs: list[PretrainedRun],
) -> tuple[PretrainedRun, Rq14PretrainedCandidate | None]:
    initial = [run for run in runs if run.candidate.stage == "initial"]
    if {run.candidate.deep_lr for run in initial} != set(DEEP_LRS):
        raise Rq14PretrainedReportError("treatment lacks its exact initial LR surface")
    boundary = [run for run in runs if run.candidate.stage == "lr_boundary"]
    initial_winner = _best(initial)
    if initial_winner.candidate.deep_lr == min(DEEP_LRS):
        direction = "low"
    elif initial_winner.candidate.deep_lr == max(DEEP_LRS):
        direction = "high"
    else:
        if boundary:
            raise Rq14PretrainedReportError("boundary exists after an interior winner")
        return initial_winner, None
    if any(run.candidate.boundary_direction != direction for run in boundary):
        raise Rq14PretrainedReportError("boundary direction contradicts initial winner")
    by_step = {run.candidate.boundary_step: run for run in boundary}
    if len(by_step) != len(boundary) or set(by_step) != set(range(1, len(boundary) + 1)):
        raise Rq14PretrainedReportError("boundary chain must be unique and contiguous")
    surface = list(initial)
    for step in range(1, len(boundary) + 1):
        expected = make_boundary_candidate(initial_winner.candidate, direction, step)
        if by_step[step].candidate != expected:
            raise Rq14PretrainedReportError("boundary cell is not the exact next point")
        if step > 1 and _best(surface) != by_step[step - 1]:
            raise Rq14PretrainedReportError("boundary continued after the surface resolved")
        surface.append(by_step[step])
    winner = _best(surface)
    if boundary and winner != by_step[len(boundary)]:
        return winner, None
    return winner, make_boundary_candidate(
        initial_winner.candidate, direction, len(boundary) + 1
    )


def _best(runs: Iterable[PretrainedRun]) -> PretrainedRun:
    return max(
        runs,
        key=lambda run: (
            run.validation_recall,
            run.validation_ndcg,
            -run.horizon_seconds,
        ),
    )


def _unexpected_effects(selected: Mapping[str, PretrainedRun]) -> dict[str, object]:
    if set(selected) != set(TREATMENTS):
        return {}
    effects = {
        "distinct_vs_shared_cls_only": (
            selected["distinct_cls_only"].metrics["recall@100"]
            - selected["shared_cls_only"].metrics["recall@100"]
        ),
        "distinct_vs_shared_history": (
            selected["distinct_history"].metrics["recall@100"]
            - selected["shared_history"].metrics["recall@100"]
        ),
        "history_vs_cls_only_shared": (
            selected["shared_history"].metrics["recall@100"]
            - selected["shared_cls_only"].metrics["recall@100"]
        ),
        "history_vs_cls_only_distinct": (
            selected["distinct_history"].metrics["recall@100"]
            - selected["distinct_cls_only"].metrics["recall@100"]
        ),
    }
    return {name: delta for name, delta in effects.items() if delta <= 0.003}


def _overall_selection(selected: Mapping[str, PretrainedRun]) -> PretrainedRun:
    if set(selected) != set(TREATMENTS):
        raise Rq14PretrainedReportError("cannot select from an incomplete treatment set")
    cls_only = selected["shared_cls_only"]
    if (
        selected["distinct_cls_only"].validation_recall
        - cls_only.validation_recall
        > 0.003
    ):
        cls_only = selected["distinct_cls_only"]
    history = selected["shared_history"]
    if (
        selected["distinct_history"].validation_recall
        - history.validation_recall
        > 0.003
    ):
        history = selected["distinct_history"]
    if history.validation_recall - cls_only.validation_recall > 0.003:
        return history
    return cls_only


def _validate_lesion_diagnostics(
    diagnostics: LesionDiagnostics | None,
    *,
    selected: Mapping[str, PretrainedRun],
    unexpected: Mapping[str, object],
    checkpoint_sha256: str | None,
) -> dict[str, object]:
    if diagnostics is None:
        return {"status": "missing"}
    evidence = diagnostics.evidence
    explanation = diagnostics.explanation
    valid = (
        diagnostics.raw_artifacts_verified
        and _sha256(diagnostics.evidence_file_sha256)
        and _sha256(diagnostics.explanation_file_sha256)
        and evidence.get("schema_version") == 1
        and evidence.get("research_question")
        == "RQ14 pretrained decoder-decoder query-memory lesions"
        and evidence.get("dataset_size") == "500m"
        and evidence.get("status") == "passed"
        and evidence.get("claims_status")
        == "diagnostics_complete_claims_not_published"
        and _sha256(evidence.get("source_rq14_results_sha256"))
        and evidence.get("source_checkpoint_sha256") == checkpoint_sha256
        and evidence.get("source_unexpected_effects") == dict(unexpected)
        and evidence.get("implementation_sha256")
        == current_lesion_implementation_sha256()
        and explanation.get("schema_version") == 1
        and explanation.get("research_question")
        == "RQ14 pretrained decoder-decoder query-memory lesions"
        and explanation.get("status") == "passed"
        and explanation.get("claims_status")
        == "diagnostics_complete_claims_not_published"
        and explanation.get("evidence_sha256") == _canonical_sha256(evidence)
        and explanation.get("summary")
        == {
            "states_used": 18,
            "states_ignored": 0,
            "within_noise_or_redundant": 18,
            "resolved_degradation_after_removal": 0,
            "resolved_change_after_removal": 0,
        }
        and set(selected) == set(TREATMENTS)
        and set(unexpected) == _EXPECTED_EFFECTS
    )
    runs = evidence.get("runs")
    run_artifacts = evidence.get("run_artifacts")
    findings = explanation.get("findings")
    if not (
        valid
        and isinstance(runs, Mapping)
        and set(runs) == set(TREATMENTS)
        and isinstance(run_artifacts, Mapping)
        and isinstance(findings, Mapping)
        and set(findings) == set(TREATMENTS)
    ):
        return {"status": "stale_or_invalid"}
    expected_artifacts = {}
    for treatment in TREATMENTS:
        run = selected[treatment]
        diagnostic = runs[treatment]
        expected_lesions = {f"drop_cls_{index}" for index in range(4)}
        if treatment.endswith("history"):
            expected_lesions.add("remove_history")
        if not isinstance(diagnostic, Mapping):
            return {"status": "stale_or_invalid"}
        lesions = diagnostic.get("lesions")
        artifact_sha256 = diagnostic.get("artifact_sha256")
        run_name = diagnostic.get("run_name")
        treatment_findings = findings.get(treatment)
        compatibility = diagnostic.get("selected_rerun_compatibility")
        if not (
            isinstance(run_name, str)
            and Path(run_name).name == run_name
            and diagnostic.get("treatment") == treatment
            and diagnostic.get("source_selected_run_name") == run.artifact_run_name
            and diagnostic.get("normal_metrics") == run.metrics
            and compatibility
            == {
                "diagnostic_minus_source_recall@100": 0.0,
                "diagnostic_minus_source_ndcg@100": 0.0,
            }
            and isinstance(artifact_sha256, Mapping)
            and artifact_sha256
            and all(_sha256(value) for value in artifact_sha256.values())
            and isinstance(lesions, Mapping)
            and set(lesions) == expected_lesions
            and isinstance(treatment_findings, Mapping)
            and set(treatment_findings) == expected_lesions
        ):
            return {"status": "stale_or_invalid"}
        expected_artifacts[run_name] = dict(artifact_sha256)
        for lesion_name in expected_lesions:
            lesion = lesions[lesion_name]
            if not isinstance(lesion, Mapping):
                return {"status": "stale_or_invalid"}
            effect = lesion.get("effect")
            query_change = lesion.get("query_change")
            if not (
                isinstance(effect, Mapping)
                and effect.get("state_use") == "states_used"
                and effect.get("recommendation_effect")
                == "within_noise_or_redundant"
                and treatment_findings[lesion_name] == effect
                and isinstance(query_change, Mapping)
                and query_change.get("changed_user_fraction") == 1
                and _positive_number(query_change.get("mean_l2_change"))
                and _positive_number(query_change.get("max_l2_change"))
            ):
                return {"status": "stale_or_invalid"}
    if dict(run_artifacts) != expected_artifacts:
        return {"status": "stale_or_invalid"}
    investigation_basis = {
        "checkpoint_sha256": checkpoint_sha256,
        "selected": {
            treatment: {
                "run_name": selected[treatment].artifact_run_name,
                "full_user_metrics": selected[treatment].metrics,
            }
            for treatment in TREATMENTS
        },
        "unexpected_effects": dict(unexpected),
    }
    return {
        "status": "passed",
        "source_rq14_results_sha256": evidence["source_rq14_results_sha256"],
        "investigation_basis_sha256": _canonical_sha256(investigation_basis),
        "evidence_file_sha256": diagnostics.evidence_file_sha256,
        "evidence_canonical_sha256": _canonical_sha256(evidence),
        "explanation_file_sha256": diagnostics.explanation_file_sha256,
        "implementation_sha256": evidence["implementation_sha256"],
        "run_artifacts": expected_artifacts,
        "summary": explanation["summary"],
    }


def _load_lesion_diagnostics(
    logs: Path,
    *,
    evidence_path: Path | None,
    explanation_path: Path | None,
) -> LesionDiagnostics | None:
    if evidence_path is None and explanation_path is None:
        return None
    if evidence_path is None or explanation_path is None:
        raise Rq14PretrainedReportError(
            "lesion evidence and explanation must be supplied together"
        )
    evidence = _load_json(evidence_path)
    explanation = _load_json(explanation_path)
    artifacts = evidence.get("run_artifacts")
    if not isinstance(artifacts, Mapping):
        raise Rq14PretrainedReportError("lesion run-artifact binding is absent")
    for run_name, hashes in artifacts.items():
        if (
            not isinstance(run_name, str)
            or Path(run_name).name != run_name
            or not isinstance(hashes, Mapping)
            or not hashes
        ):
            raise Rq14PretrainedReportError("lesion run-artifact binding is invalid")
        for name, expected_sha256 in hashes.items():
            if (
                not isinstance(name, str)
                or Path(name).name != name
                or not _sha256(expected_sha256)
                or _file_sha256(logs / run_name / name) != expected_sha256
            ):
                raise Rq14PretrainedReportError(
                    f"{run_name}: lesion artifact hash changed"
                )
    return LesionDiagnostics(
        evidence=evidence,
        explanation=explanation,
        evidence_file_sha256=_file_sha256(evidence_path),
        explanation_file_sha256=_file_sha256(explanation_path),
        raw_artifacts_verified=True,
    )


def _validate_audit(
    audit: Mapping[str, object] | None,
    artifacts: Mapping[str, Mapping[str, str]],
    checkpoint_sha256: str | None,
) -> dict[str, object]:
    if audit is None:
        return {"status": "missing"}
    checks = audit.get("checks")
    if (
        audit.get("schema_version") == 1
        and audit.get("research_question")
        == "RQ14 pretrained decoder-decoder query memory"
        and audit.get("dataset_size") == "500m"
        and audit.get("status") == "passed"
        and isinstance(checks, Mapping)
        and set(checks) == _AUDIT_CHECKS
        and all(
            isinstance(checks[name], Mapping) and checks[name].get("passed") is True
            for name in _AUDIT_CHECKS
        )
        and audit.get("run_artifacts") == dict(artifacts)
        and audit.get("checkpoint_sha256") == checkpoint_sha256
        and audit.get("implementation_sha256") == current_implementation_sha256()
    ):
        return {"status": "passed", "artifact_sha256": _canonical_sha256(audit)}
    return {"status": "stale_or_invalid"}


def _validate_source_binding(
    source: SourceCheckpointBinding | None, runs: list[PretrainedRun]
) -> dict[str, object]:
    if source is None:
        return {"status": "missing"}
    valid = (
        source.verified
        and isinstance(source.run_name, str)
        and Path(source.run_name).name == source.run_name
        and Path(source.checkpoint_path).is_absolute()
        and Path(source.checkpoint_path).parent.name == source.run_name
        and _sha256(source.checkpoint_sha256)
        and all(run.checkpoint_sha256 == source.checkpoint_sha256 for run in runs)
    )
    if not valid:
        return {"status": "stale_or_invalid"}
    return {
        "status": "passed",
        "run_name": source.run_name,
        "checkpoint_path": source.checkpoint_path,
        "checkpoint_sha256": source.checkpoint_sha256,
    }


def _run_record(run: PretrainedRun) -> dict[str, object]:
    return {
        "treatment": run.candidate.treatment,
        "run_name": run.artifact_run_name,
        "source": "reused RQ15" if run.reused_from_rq15 else "new RQ14",
        "embedding_lr": run.candidate.embedding_lr,
        "deep_lr": run.candidate.deep_lr,
        "best_epoch": run.best_epoch,
        "stopped_epoch": run.stopped_epoch,
        "validation_recall@100": run.validation_recall,
        "validation_ndcg@100": run.validation_ndcg,
        "full_user_metrics": run.metrics,
        "expanded_examples_per_epoch": run.expanded_examples_per_epoch,
        "candidate_targets_per_epoch": run.candidate_targets_per_epoch,
        "ntp_targets_per_epoch": run.ntp_targets_per_epoch,
        "input_tokens_per_epoch": run.input_tokens_per_epoch,
        "steady_state_targets_per_second": run.steady_state_targets_per_second,
        "time_through_best_seconds": run.time_through_best_seconds,
        "required_horizon_seconds": run.horizon_seconds,
        "artifact_sha256": run.artifact_sha256,
        "checkpoint_sha256": run.checkpoint_sha256,
    }


def _pretrained_reader(
    selected: Mapping[str, PretrainedRun], runs: list[PretrainedRun]
) -> str:
    selected_method = _overall_selection(selected)
    reference = selected["shared_cls_only"]
    quality = [
        "\n## NTP-pretrained quality (current decision)\n\n",
        "| query slots | second-decoder memory | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 |\n",
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: |\n",
    ]
    efficiency = [
        "\n## NTP-pretrained training efficiency\n\n",
        "| query slots | second-decoder memory | examples/epoch | candidate targets/epoch | NTP targets/epoch | input tokens/epoch | targets/s | best epoch | processed examples | processed candidate targets | time to checkpoint | 3-cell tuning GPU time |\n",
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n",
    ]
    for treatment in TREATMENTS:
        run = selected[treatment]
        slots = "shared CLS" if run.candidate.query_slots_shared else "distinct CLS_0..3"
        memory = "history + four CLS states" if run.candidate.include_history_memory else "four CLS states"
        quality_cells = [slots, memory]
        quality_cells.extend(
            reporting.absolute(run.metrics[metric])
            if treatment == "shared_cls_only"
            else reporting.change_cell(
                run.metrics[metric], reference.metrics[metric], metric
            )
            for metric in _METRICS
        )
        tuning_horizon = sum(
            item.horizon_seconds
            for item in runs
            if item.candidate.treatment == treatment
        )
        efficiency_cells = [
            slots,
            memory,
            f"{run.expanded_examples_per_epoch:,}",
            f"{run.candidate_targets_per_epoch:,}",
            f"{run.ntp_targets_per_epoch:,}",
            f"{run.input_tokens_per_epoch:,}",
            f"{run.steady_state_targets_per_second:,.0f}",
            str(run.best_epoch),
            f"{run.expanded_examples_per_epoch * run.best_epoch:,}",
            f"{run.candidate_targets_per_epoch * run.best_epoch:,}",
            _duration(run.time_through_best_seconds),
            _duration(tuning_horizon),
        ]
        quality.append(_reader_row(quality_cells, selected=run == selected_method))
        efficiency.append(
            _reader_row(efficiency_cells, selected=run == selected_method)
        )
    return "".join([*quality, *efficiency])


def _tuning_markdown(
    runs: list[PretrainedRun], selected: Mapping[str, PretrainedRun]
) -> str:
    lines = ["# RQ14 NTP-pretrained tuning\n"]
    for treatment in TREATMENTS:
        lines.extend(
            [
                f"\n## {treatment}\n",
                "| source | embedding LR | deep LR | validation recall@100 | validation ndcg@100 | best epoch |\n",
                "| :--- | ---: | ---: | ---: | ---: | ---: |\n",
            ]
        )
        for run in sorted(
            (item for item in runs if item.candidate.treatment == treatment),
            key=lambda item: item.candidate.deep_lr,
        ):
            source = "reused RQ15" if run.reused_from_rq15 else "new RQ14"
            cells = [
                source,
                f"{run.candidate.embedding_lr:g}",
                f"{run.candidate.deep_lr:g}",
                f"{run.validation_recall:.6f}",
                f"{run.validation_ndcg:.6f}",
                str(run.best_epoch),
            ]
            if selected.get(treatment) == run:
                cells = [f"**{cell}**" for cell in cells]
            lines.append("| " + " | ".join(cells) + " |\n")
    return "".join(lines)


def _readme_markdown(reader: str, claims_status: str) -> str:
    title = "# RQ14 decoder-decoder query memory"
    if not reader.startswith(title):
        raise Rq14PretrainedReportError("RQ14 reader title is invalid")
    body = "\n".join(
        f"#{line}" if line.startswith("## ") else line
        for line in reader[len(title) :].lstrip().splitlines()
    )
    introduction = (
        "The current comparison initializes the first causal decoder from the "
        "RQ15-selected NTP checkpoint, then jointly fine-tunes both decoders with "
        "candidate loss only. The first decoder appends four shared or distinct "
        "query slots; the second decoder cross-attends either those four states or "
        "the complete history followed by them. The earlier candidate-only-from-scratch "
        "comparison is preserved separately because it measures a different training regime."
    )
    lines = [
        "## RQ14 — Should the second decoder attend distinct CLS tokens or history too?",
        "",
        introduction,
        "",
        body.rstrip(),
    ]
    if claims_status == "ready_for_user_validation":
        lines.extend(
            [
                "",
                "### Acceptance criteria",
                "",
                "- The decoder which cross-attends both CLS tokens and history probably should be better.",
                "",
                "- Four separate class tokens should probably have better metrics than the same CLS token.",
                "",
                "- If the statements above do not hold true, first debug, and if everything is correct, explain why experimentally.",
                "",
                "### Analysis and conclusion",
                "",
                "All four expected effects point in the requested direction, but every Recall@100 difference is inside the native-500M 0.003 single-run band, so none is reported as a gain. All 16 individual CLS-state removals and both history-memory removals changed every evaluated user's query representation. Every lesion metric change remains inside the Recall@100 and NDCG@100 bands, so the extra states' marginal recommendation contribution is unresolved or redundant.",
                "",
                "The approved simplicity rule therefore selects shared CLS with CLS-only memory. Distinct CLS with history is numerically highest, but its advantage is unresolved. The pretrained comparison is the current RQ14 architecture decision pending user validation; the candidate-only table remains historical evidence, not the current selection regime.",
                "",
                "Implementation and evidence: [pretrained tuning ledger](scratchpad/rq14_pretrained_tuning_500m.md), [machine-readable result](evidence/rq14_pretrained_results.json), [correctness audit](evidence/rq14_pretrained_correctness.json), [lesion evidence](evidence/rq14_pretrained_lesion_results.json), and [bound lesion explanation](evidence/rq14_pretrained_lesion_explanation.json).",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def update_readme(path: Path, block: str) -> None:
    text = path.read_text()
    replacement = f"{_RQ14_START}\n{block.rstrip()}\n{_RQ14_END}\n\n"
    if _RQ14_START in text or _RQ14_END in text:
        if text.count(_RQ14_START) != 1 or text.count(_RQ14_END) != 1:
            raise Rq14PretrainedReportError("RQ14 README markers are malformed")
        start = text.index(_RQ14_START)
        end = text.index(_RQ14_END, start) + len(_RQ14_END)
        updated = text[:start] + replacement.rstrip() + text[end:]
    else:
        heading = "## RQ14 —"
        start = text.find(heading)
        end = text.find(_RQ15_START, start)
        if start < 0 or end < 0:
            raise Rq14PretrainedReportError(
                "cannot locate the existing RQ14 block before RQ15"
            )
        updated = text[:start] + replacement + text[end:]
    _write_text(path, updated)


def _load_run(
    directory: Path,
    candidate: Rq14PretrainedCandidate,
    rq15_record: Mapping[str, object] | None,
    *,
    source_checkpoint: SourceCheckpointBinding,
) -> PretrainedRun:
    metadata = _load_json(directory / "training_metadata.json")
    metrics = _load_json(directory / "final_metrics.json")
    expected = {
        "dataset_size": "500m",
        "seed": 42,
        "stopped_epoch": 20,
        "lr_horizon_complete": True,
        "selection_resolved": True,
        "effective_batch_size": 1280,
        "embedding_learning_rate": 0.00025,
        "deep_learning_rate": candidate.deep_lr,
        "query_architecture": "decoder_decoder",
        "query_slots_shared": candidate.query_slots_shared,
        "include_history_memory": candidate.include_history_memory,
        "num_query_slots": 4,
        "training_method": "pretrained_finetune",
        "ntp_targets_per_epoch": 0,
        "auxiliary_ntp_weight": 0.0,
        "lr_schedule_horizon_epochs": 20,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise Rq14PretrainedReportError(
                f"{directory.name}: {key}={metadata.get(key)!r}, expected {value!r}"
            )
    initialization = metadata.get("first_stage_initialization")
    if not isinstance(initialization, Mapping) or not _sha256(
        initialization.get("checkpoint_sha256")
    ):
        raise Rq14PretrainedReportError(f"{directory.name}: invalid checkpoint identity")
    if initialization.get("copied_modules") != [
        "item_embedding",
        "memory_encoder",
        "tokenizer",
    ] or initialization.get("newly_initialized_modules") != [
        "decoder",
        "decoder_query",
        "query_projection",
        "query_slots",
    ]:
        raise Rq14PretrainedReportError(f"{directory.name}: wrong checkpoint load scope")
    validate_source_initialization(
        initialization, source_checkpoint, context=directory.name
    )
    checkpoint_path = str(initialization.get("checkpoint_path"))
    source_run = Path(checkpoint_path).parent.name
    assignments = [
        f"G1_RQ15_SOURCE_RUN={source_run}",
        f"G1_RQ15_FIRST_STAGE_CHECKPOINT={checkpoint_path}",
    ]
    config = _RQ15_CONFIG if rq15_record is not None else _RQ14_CONFIG
    run_assignment = (
        f"G1_RQ15_RUN={directory.name}"
        if rq15_record is not None
        else f"G1_RQ14_PRETRAINED_RUN={directory.name}"
    )
    if not verify_artifact.verify_config(
        directory, config, [run_assignment, *assignments]
    ):
        raise Rq14PretrainedReportError(
            f"{directory.name}: recipe-incompatible artifact"
        )
    if rq15_record is not None and not rq15_reuse_is_recipe_compatible(
        directory,
        [f"G1_RQ14_PRETRAINED_RUN={candidate.run_name}", *assignments],
    ):
        raise Rq14PretrainedReportError(
            f"{directory.name}: RQ15 artifact is not recipe-compatible with RQ14"
        )
    artifact_hashes = {
        name: _file_sha256(directory / name)
        for name in ("training_metadata.json", "final_metrics.json", "sweep.log")
    }
    if rq15_record is not None:
        if (
            rq15_record.get("run_name") != directory.name
            or rq15_record.get("artifact_sha256") != artifact_hashes
            or rq15_record.get("checkpoint_sha256")
            != initialization["checkpoint_sha256"]
        ):
            raise Rq14PretrainedReportError(
                f"{directory.name}: RQ15 reuse binding does not match the artifact"
            )
    curve, timings = _validation_curve(directory / "sweep.log")
    best_epoch = int(metadata["best_epoch"])
    if best_epoch not in range(1, 21):
        raise Rq14PretrainedReportError(f"{directory.name}: invalid best epoch")
    selected = curve[best_epoch - 1]
    train_times = [point[3] for point in timings]
    total_times = [point[4] for point in timings]
    expanded_examples = _positive_int(
        metadata, "expanded_examples_per_epoch", directory.name
    )
    candidate_targets = _positive_int(
        metadata, "candidate_targets_per_epoch", directory.name
    )
    input_tokens = _positive_int(metadata, "input_tokens_per_epoch", directory.name)
    if expanded_examples != candidate_targets:
        raise Rq14PretrainedReportError(
            f"{directory.name}: expected one candidate target per example"
        )
    return PretrainedRun(
        candidate=candidate,
        artifact_run_name=directory.name,
        reused_from_rq15=rq15_record is not None,
        validation_recall=selected[1],
        validation_ndcg=selected[2],
        best_epoch=best_epoch,
        stopped_epoch=20,
        metrics={
            metric: _finite_metric(metrics, metric, directory.name)
            for metric in _METRICS
        },
        expanded_examples_per_epoch=expanded_examples,
        candidate_targets_per_epoch=candidate_targets,
        ntp_targets_per_epoch=0,
        input_tokens_per_epoch=input_tokens,
        steady_state_targets_per_second=(
            candidate_targets * 19 / sum(train_times[1:])
        ),
        time_through_best_seconds=sum(total_times[:best_epoch]),
        horizon_seconds=sum(total_times),
        artifact_sha256=artifact_hashes,
        checkpoint_sha256=str(initialization["checkpoint_sha256"]),
    )


def validate_source_initialization(
    initialization: Mapping[str, object],
    source: SourceCheckpointBinding,
    *,
    context: str,
) -> None:
    path = initialization.get("checkpoint_path")
    if (
        source.verified is not True
        or not isinstance(path, str)
        or Path(path).resolve() != Path(source.checkpoint_path).resolve()
        or Path(path).parent.name != source.run_name
        or initialization.get("checkpoint_sha256") != source.checkpoint_sha256
    ):
        raise Rq14PretrainedReportError(
            f"{context}: first-stage initialization does not match the selected source"
        )


def _validation_curve(
    path: Path,
) -> tuple[
    list[tuple[int, float, float]],
    list[tuple[int, float, float, float, float]],
]:
    pattern = re.compile(
        r"epoch (?P<epoch>\d+) finished .*?timing\.train_epoch_time=(?P<time>[0-9.]+)"
        r".*?timing\.val_inference_time=(?P<validation>[0-9.]+)"
        r".*?timing\.val_save_time=(?P<save>[0-9.]+)"
        r".*?epoch/val_true\.ndcg@100=(?P<ndcg>[0-9.]+)"
        r".*?epoch/val_true\.recall@100=(?P<recall>[0-9.]+)"
    )
    points = []
    timings = []
    for line in path.read_text().splitlines():
        match = pattern.search(line)
        if match is None:
            continue
        epoch = int(match.group("epoch")) + 1
        recall = float(match.group("recall"))
        ndcg = float(match.group("ndcg"))
        time = float(match.group("time"))
        total_time = time + float(match.group("validation")) + float(match.group("save"))
        points.append((epoch, recall, ndcg))
        timings.append((epoch, recall, ndcg, time, total_time))
    if [point[0] for point in points] != list(range(1, 21)):
        raise Rq14PretrainedReportError(f"{path.parent.name}: incomplete horizon curve")
    return points, timings


def rq15_reuse_is_recipe_compatible(
    directory: Path, assignments: list[str]
) -> bool:
    normalized = verify_artifact._config_assignments(assignments)
    experiment = verify_artifact._config_experiment(_RQ14_CONFIG, normalized)
    expected_top, expected_invariants = verify_artifact._expected_metadata(experiment)
    actual = _load_json(directory / "training_metadata.json")
    if any(actual.get(key) != value for key, value in expected_top.items()):
        return False
    actual_invariants = actual.get("transfer_invariants")
    if not isinstance(actual_invariants, Mapping):
        return False
    actual_recipe = dict(actual_invariants)
    expected_recipe = dict(expected_invariants)
    actual_recipe.pop("experiment_class", None)
    expected_recipe.pop("experiment_class", None)
    return actual_recipe == expected_recipe


def validate_selected_checkpoint(
    evidence: Mapping[str, object], checkpoint: Path
) -> str:
    if not checkpoint.is_file():
        raise Rq14PretrainedReportError("selected source checkpoint is absent")
    selected_sha256 = _file_sha256(checkpoint)
    if evidence.get("checkpoint_sha256") != selected_sha256:
        raise Rq14PretrainedReportError(
            "selected source checkpoint differs from the three reused RQ15 cells"
        )
    return selected_sha256


def _rq15_reuse_records(results: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    treatments = results.get("treatments")
    pretrained = treatments.get("pretrained_finetune") if isinstance(treatments, Mapping) else None
    artifacts = pretrained.get("artifacts") if isinstance(pretrained, Mapping) else None
    if not isinstance(artifacts, list):
        raise Rq14PretrainedReportError("RQ15 pretrained evidence is absent")
    return {
        str(record["run_name"]): record
        for record in artifacts
        if isinstance(record, Mapping) and isinstance(record.get("run_name"), str)
    }


def _complete(directory: Path) -> bool:
    return all(
        (directory / name).is_file()
        for name in ("training_metadata.json", "final_metrics.json", "sweep.log")
    )


def _finite_metric(document: Mapping[str, object], key: str, run: str) -> float:
    value = document.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise Rq14PretrainedReportError(f"{run}: invalid {key}")
    return float(value)


def _positive_int(document: Mapping[str, object], key: str, run: str) -> int:
    value = document.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise Rq14PretrainedReportError(f"{run}: invalid {key}")
    return value


def _positive_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def _reader_row(cells: list[str], *, selected: bool) -> str:
    if selected:
        cells = [f"**{cell}**" for cell in cells]
    return "| " + " | ".join(cells) + " |\n"


def _duration(seconds: float) -> str:
    hours, remainder = divmod(round(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}"


def _load_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise Rq14PretrainedReportError(f"cannot read {path}") from error
    if not isinstance(value, dict):
        raise Rq14PretrainedReportError(f"{path}: expected JSON object")
    return value


def _load_optional_json(path: Path) -> dict[str, object] | None:
    return _load_json(path) if path.is_file() else None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _write_json(path: Path, value: object) -> None:
    _write_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        handle.write(value)
        temporary = Path(handle.name)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--rq15-results", type=Path, required=True)
    parser.add_argument("--candidate-only", type=Path, required=True)
    parser.add_argument("--correctness", type=Path)
    parser.add_argument("--lesion-evidence", type=Path)
    parser.add_argument("--lesion-explanation", type=Path)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--scratchpad", type=Path, required=True)
    parser.add_argument("--readme", type=Path)
    args = parser.parse_args()
    bundle = collect_report_bundle(
        args.logs,
        rq15_results_path=args.rq15_results,
        candidate_only_path=args.candidate_only,
        correctness_path=args.correctness,
        lesion_evidence_path=args.lesion_evidence,
        lesion_explanation_path=args.lesion_explanation,
    )
    write_report_bundle(bundle, evidence=args.evidence, scratchpad=args.scratchpad)
    if args.readme is not None:
        if bundle.evidence["claims_status"] != "ready_for_user_validation":
            raise Rq14PretrainedReportError(
                "README publication requires validation-ready RQ14 evidence"
            )
        update_readme(args.readme, bundle.readme_markdown)
    print(json.dumps(bundle.evidence, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
