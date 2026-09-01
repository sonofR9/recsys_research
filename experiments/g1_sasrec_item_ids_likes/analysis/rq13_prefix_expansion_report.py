from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile
from typing import Any

import polars as pl

from dcn.training_metadata import GENERATION_TRAINING_SEMANTICS_REVISION
from experiments.g1_sasrec_item_ids_likes.analysis import reporting
from experiments.g1_sasrec_item_ids_likes.analysis.rq13_prefix_expansion_audit import (
    Rq13AuditError,
    eligible_target_counts_from_cache,
    load_correctness_audit,
    validate_correctness_audit,
)
from experiments.g1_sasrec_item_ids_likes.analysis.rq13_prefix_cap_fit import (
    Rq13CapFitError,
    build_cap_fit,
)
from experiments.g1_sasrec_item_ids_likes.analysis.rq13_rq14_query_candidates import (
    DEEP_LRS,
    QueryCandidate,
    candidate_by_run,
    make_boundary_candidate,
    make_selected_cap_candidates,
    rq13_cap4_candidates,
    rq13_initial_candidates,
)
from experiments.g1_sasrec_item_ids_likes.launchers import verify_artifact
from utils.report_file_facts import ReportFileFacts, report_file_facts


_CONFIG = Path(__file__).parents[1] / "configs/rq13_rq14_query_variant.py"
_RQ12_EVIDENCE = Path(__file__).parents[1] / "evidence/rq12_decoder_query_results.json"
_CORRECTNESS_EVIDENCE = (
    Path(__file__).parents[1] / "evidence/rq13_prefix_expansion_correctness.json"
)
_RESULTS_EVIDENCE = (
    Path(__file__).parents[1] / "evidence/rq13_prefix_expansion_results.json"
)
_RETAINED_LEGACY_CACHE_FINGERPRINTS = {
    "train": "54b84af1d3dcea6019f60adbf06043eac6e495ea5eb865df086712b7189bc4cf",
    "val": "d491e48cfdc655d88a4c994807ab0d73254f33240dd614464a9910fec28ae9c4",
    "true_metric_query": "c28bde9bbefd14b76db15dae5537960d8e76d6c103fbc6fb7255fbd40656aff0",
}
_TREATMENTS = (
    "one_example",
    "truncated_8",
    "truncated_16",
    "required_8",
    "required_16",
)
_CAP_ANCHOR_TREATMENT = "truncated_4"
_STAGE_ONE_TREATMENTS = {*_TREATMENTS, _CAP_ANCHOR_TREATMENT}
_LABELS = {
    "one_example": "no expansion",
    "truncated_4": "latest 4 truncated prefixes",
    "truncated_8": "latest 8 truncated prefixes",
    "truncated_16": "latest 16 truncated prefixes",
    "required_8": "latest 8 required-length prefixes",
    "required_16": "latest 16 required-length prefixes",
}
_METRICS = ("recall@100", "ndcg@100", "recall@10", "ndcg@10", "coverage@100")
_METRIC_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_EPOCH_PATTERN = re.compile(
    rf"\bepoch (\d+) finished\b.*?"
    rf"timing\.train_epoch_time=({_METRIC_NUMBER}).*?"
    rf"timing\.val_inference_time=({_METRIC_NUMBER}).*?"
    rf"timing\.val_save_time=({_METRIC_NUMBER})"
)
_TIMESTAMP_PATTERN = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:[.,]\d+)?)")
_DATASET_PATH_PATTERN = re.compile(r"\bPreparing yambda .* in (?P<path>\S+)$")
_CACHE_PATTERN = re.compile(
    r"(?:\bLoaded cached user sequences from |\bBuilt \d+ user sequences at )"
    r"(?P<path>\S+/sequences/(?P<split>train|val|true_metric_query)_[^\s/]+)"
    r"(?:$|\s)"
)
_DATASET_FILES = ("events.parquet", "events_remapped.parquet", "item_id_remap.parquet")
_EVALUATOR_FIELDS = (
    "eval_ks",
    "eval_max_users",
    "eval_every_n_epochs",
    "early_stopping_metric",
    "early_stopping_metric_prefix",
    "selection_k",
    "evaluation_catalog",
    "exclude_seen_from_evaluation",
    "restore_best_weights",
)
_SCORING_FIELDS = (
    "negative_sampling",
    "num_in_batch_negatives",
    "logq_correction",
    "random_negative_fraction",
    "logq_alpha",
    "correct_positive_logq",
    "mask_false_negatives",
    "exclude_own_group_negatives",
    "dense_random_negative_scores",
)
_COUNT_FIELDS = (
    "original_users_per_epoch",
    "expanded_examples_per_epoch",
    "candidate_targets_per_epoch",
    "ntp_targets_per_epoch",
    "input_tokens_per_epoch",
)


class Rq13ReportError(RuntimeError):
    pass


@dataclass(frozen=True)
class Control:
    label: str
    source: str
    quality: dict[str, float]
    original_users_per_epoch: int | None
    expanded_examples_per_epoch: int
    candidate_targets_per_epoch: int
    ntp_targets_per_epoch: int
    input_tokens_per_epoch: int
    best_epochs: str
    steady_state_targets_per_second: float
    time_through_selected_checkpoint_seconds: float
    total_required_training_wall_seconds: float
    dataset_content_manifest_sha256: str
    validation_cache_manifest_sha256: str
    query_cache_manifest_sha256: str
    evaluator_fingerprint: str
    scoring_fingerprint: str
    validation_recall_mean: float = sum((0.1367, 0.1343, 0.1363)) / 3


@dataclass(frozen=True)
class Run:
    candidate: QueryCandidate
    best_epoch: int
    stopped_epoch: int
    validation_recall: float
    validation_ndcg: float
    validation_curve: tuple[tuple[int, float, float], ...]
    metrics: dict[str, float]
    original_users_per_epoch: int
    expanded_examples_per_epoch: int
    candidate_targets_per_epoch: int
    ntp_targets_per_epoch: int
    input_tokens_per_epoch: int
    optimizer_steps_per_epoch: int
    steady_state_targets_per_second: float
    time_through_selected_checkpoint_seconds: float
    required_horizon_train_validation_seconds: float
    observed_end_to_end_wall_seconds: float
    dataset_content_manifest_sha256: str
    train_cache_manifest_sha256: str
    validation_cache_manifest_sha256: str
    query_cache_manifest_sha256: str
    evaluator_fingerprint: str
    scoring_fingerprint: str
    artifact_sha256: dict[str, str]


@dataclass(frozen=True)
class Rq13ReportBundle:
    reader_markdown: str | None
    tuning_markdown: str
    diagnostics_markdown: str | None
    evidence: dict[str, object]


RecipeVerifier = Callable[[Path, QueryCandidate], bool]


def collect_report_bundle(
    logs: Path,
    control: Control | None = None,
    *,
    verify_recipe: RecipeVerifier | None = None,
    expected_counts: Mapping[str, Mapping[str, int]] | None = None,
    rq12_evidence: Path = _RQ12_EVIDENCE,
    correctness_evidence: Path = _CORRECTNESS_EVIDENCE,
    cap_extension: bool = True,
) -> Rq13ReportBundle:
    checked_control = load_rq12_control(rq12_evidence) if control is None else control
    verifier = _verify_recipe if verify_recipe is None else verify_recipe
    facts = report_file_facts(logs.parent)
    runs = []
    for directory in sorted(logs.glob("g1_rq13_*_500m")):
        if not directory.is_dir():
            continue
        try:
            candidate = _candidate_from_run_name(directory.name)
        except ValueError:
            continue
        required = tuple(
            directory / name
            for name in ("training_metadata.json", "final_metrics.json", "sweep.log")
        )
        if not all(path.is_file() for path in required):
            continue
        if not verifier(directory, candidate):
            raise Rq13ReportError(f"{directory.name}: recipe-incompatible artifact")
        runs.append(
            _load_run(
                directory,
                candidate,
                (
                    None
                    if expected_counts is None
                    else expected_counts.get(candidate.treatment)
                ),
                facts,
            )
        )
    cap_counts = None
    rq12_document = None
    if cap_extension and any(
        run.candidate.treatment == _CAP_ANCHOR_TREATMENT for run in runs
    ):
        source_run = next(
            run for run in runs if run.candidate.treatment == "truncated_16"
        )
        log = logs / source_run.candidate.run_name / "sweep.log"
        train_caches = {
            Path(match.group("path"))
            for line in log.read_text().splitlines()
            if (match := _CACHE_PATTERN.search(line)) is not None
            and match.group("split") == "train"
        }
        if len(train_caches) != 1:
            raise Rq13ReportError("RQ13 cap fit has ambiguous train-cache provenance")
        cap_counts = eligible_target_counts_from_cache(train_caches.pop())
        rq12_document = _load_json(rq12_evidence)
    return build_report_bundle(
        runs,
        checked_control,
        correctness_audit=load_correctness_audit(correctness_evidence),
        cap_extension=cap_extension,
        rq12_evidence_document=rq12_document,
        eligible_target_counts=cap_counts,
    )


def build_report_bundle(
    runs: Iterable[Run],
    control: Control,
    *,
    correctness_audit: Mapping[str, object] | None = None,
    cap_extension: bool = False,
    rq12_evidence_document: Mapping[str, object] | None = None,
    eligible_target_counts: list[int] | None = None,
) -> Rq13ReportBundle:
    run_list = list(runs)
    by_name = {run.candidate.run_name: run for run in run_list}
    if len(by_name) != len(run_list):
        raise Rq13ReportError("duplicate RQ13 artifact identity")
    if any(run.candidate.study != "rq13" for run in run_list):
        raise Rq13ReportError("non-RQ13 run entered the RQ13 report")
    _require_workload_consistency(run_list)
    _require_control_compatibility(run_list, control)

    active_treatments = list(_TREATMENTS)
    if cap_extension:
        active_treatments.append(_CAP_ANCHOR_TREATMENT)
    selected: dict[str, Run] = {}
    missing_initial_artifacts: list[str] = []
    required_boundary_followups: list[str] = []
    treatment_runs: dict[str, list[Run]] = {}
    for treatment in active_treatments:
        available = [run for run in run_list if run.candidate.treatment == treatment]
        treatment_runs[treatment] = available
        missing = [
            candidate.run_name
            for candidate in _initial_candidates_for_treatment(treatment)
            if candidate.treatment == treatment and candidate.run_name not in by_name
        ]
        if missing:
            if any(run.candidate.stage == "lr_boundary" for run in available):
                raise Rq13ReportError(
                    f"{treatment}: boundary artifact precedes the complete initial surface"
                )
            missing_initial_artifacts.extend(missing)
            continue
        winner, followup = _resolve_treatment_surface(available)
        selected[treatment] = winner
        if followup is not None and followup.run_name not in by_name:
            required_boundary_followups.append(followup.run_name)

    required_followups = [*missing_initial_artifacts, *required_boundary_followups]
    cap_fit: dict[str, object] = {
        "status": "not_requested" if not cap_extension else "pending_cap4"
    }
    if (
        cap_extension
        and not required_followups
        and len(selected) == len(active_treatments)
    ):
        if rq12_evidence_document is None or eligible_target_counts is None:
            raise Rq13ReportError("RQ13 cap-fit source evidence is absent")
        fit_inputs = {
            "research_question": "RQ13 encoder-decoder prefix expansion",
            "dataset_size": "500m",
            "missing_initial_artifacts": [],
            "required_followups": [],
            "required_boundary_followups": [],
            "surface_winners": {
                treatment: {
                    **_selected_record(selected[treatment]),
                    "artifact_sha256": selected[treatment].artifact_sha256,
                    "source_manifest_sha256": selected[
                        treatment
                    ].dataset_content_manifest_sha256,
                }
                for treatment in (
                    "one_example",
                    "truncated_4",
                    "truncated_8",
                    "truncated_16",
                )
            },
            "treatments": {
                treatment: {
                    "artifacts": [_run_record(run) for run in treatment_runs[treatment]]
                }
                for treatment in (
                    "one_example",
                    "truncated_4",
                    "truncated_8",
                    "truncated_16",
                )
            },
        }
        try:
            cap_fit = build_cap_fit(
                fit_inputs,
                rq12_evidence_document,
                eligible_target_counts,
                selected["truncated_16"].input_tokens_per_epoch,
            )
        except Rq13CapFitError as error:
            cap_fit = {"status": "failed", "error": str(error)}
        if cap_fit.get("status") == "selected_cap_pending":
            cap = cap_fit.get("selected_cap")
            try:
                fitted_candidates = make_selected_cap_candidates(cap)  # type: ignore[arg-type]
            except ValueError as error:
                raise Rq13ReportError("RQ13 fit selected an invalid cap") from error
            expected_stage_one_artifacts = {
                run.candidate.run_name: run.artifact_sha256
                for run in run_list
                if run.candidate.treatment in _STAGE_ONE_TREATMENTS
            }
            try:
                if correctness_audit is None:
                    raise Rq13AuditError("saved correctness audit is missing")
                stage_one_audit = validate_correctness_audit(
                    correctness_audit, expected_stage_one_artifacts
                )
            except Rq13AuditError as error:
                selected_cap_initial_complete = all(
                    candidate.run_name in by_name for candidate in fitted_candidates
                )
                final_audit = None
                if selected_cap_initial_complete and correctness_audit is not None:
                    expected_final_artifacts = {
                        run.candidate.run_name: run.artifact_sha256 for run in run_list
                    }
                    try:
                        final_audit = validate_correctness_audit(
                            correctness_audit, expected_final_artifacts
                        )
                    except Rq13AuditError:
                        pass
                if final_audit is None:
                    cap_fit["status"] = "stage_one_audit_required"
                    cap_fit["proposed_followups"] = [
                        candidate.run_name for candidate in fitted_candidates
                    ]
                    cap_fit["audit_error"] = str(error)
                else:
                    cap_fit["input_bindings"]["stage_one_correctness_audit"] = {
                        "status": "covered_by_final_audit",
                        "schema_version": final_audit["schema_version"],
                        "artifact_sha256": final_audit["artifact_sha256"],
                        "expected_run_artifacts_sha256": _canonical_sha256(
                            expected_stage_one_artifacts
                        ),
                    }
                    cap_fit["input_bindings"]["final_correctness_audit"] = final_audit
            else:
                cap_fit["input_bindings"][
                    "stage_one_correctness_audit"
                ] = stage_one_audit
            if cap_fit.get("status") != "selected_cap_pending":
                fitted_candidates = ()
        if cap_fit.get("status") == "selected_cap_pending":
            treatment = fitted_candidates[0].treatment
            active_treatments.append(treatment)
            available = [
                run for run in run_list if run.candidate.treatment == treatment
            ]
            treatment_runs[treatment] = available
            missing = [
                candidate.run_name
                for candidate in fitted_candidates
                if candidate.run_name not in by_name
            ]
            if missing:
                if any(run.candidate.stage == "lr_boundary" for run in available):
                    raise Rq13ReportError(
                        "selected cap boundary artifact precedes its initial surface"
                    )
                missing_initial_artifacts.extend(missing)
                required_followups.extend(missing)
            else:
                winner, followup = _resolve_treatment_surface(available)
                selected[treatment] = winner
                if followup is not None and followup.run_name not in by_name:
                    required_boundary_followups.append(followup.run_name)
                    required_followups.append(followup.run_name)
                elif followup is None:
                    cap_fit["status"] = "resolved"
    resolved_selected: dict[str, Run] = {}
    if not required_followups and len(selected) == len(active_treatments):
        resolved_selected = {
            treatment: _best(treatment_runs[treatment])
            for treatment in active_treatments
        }
    selected_method = _select_method(resolved_selected)
    comparisons = _method_comparisons(resolved_selected)
    acceptance = _acceptance_checks(resolved_selected)
    resolved_diagnostics = _resolved_diagnostics(resolved_selected, treatment_runs)
    diagnostics = _required_diagnostics(acceptance, resolved_diagnostics)
    if cap_fit.get("status") == "failed":
        diagnostics.append(f"cap-fit selection failed closed: {cap_fit['error']}")
    elif cap_fit.get("status") == "stage_one_audit_required":
        diagnostics.append(
            "generate the source-exact stage-one correctness audit before launching the fitted cap"
        )
    cap_fit_bindings = cap_fit.get("input_bindings")
    final_audit_bound = isinstance(cap_fit_bindings, Mapping) and isinstance(
        cap_fit_bindings.get("final_correctness_audit"), Mapping
    )
    correctness_required = (
        bool(resolved_diagnostics)
        or final_audit_bound
        or (
            cap_extension
            and cap_fit.get("status") in {"pending_cap4", "stage_one_audit_required"}
            and set(_TREATMENTS).issubset(selected)
        )
    )
    correctness_record: dict[str, object] = {"required": correctness_required}
    if correctness_required:
        expected_artifacts = {
            run.candidate.run_name: run.artifact_sha256 for run in run_list
        }
        try:
            if correctness_audit is None:
                raise Rq13AuditError("saved correctness audit is missing")
            correctness_record.update(
                validate_correctness_audit(correctness_audit, expected_artifacts)
            )
        except Rq13AuditError as error:
            correctness_record.update({"status": "failed", "error": str(error)})
            diagnostics.append("generate and validate the saved correctness audit")
    else:
        correctness_record["status"] = "not_required"
    if required_followups:
        claims_status = "pending_artifacts"
    elif diagnostics:
        claims_status = "diagnostics_required"
    else:
        claims_status = "ready_for_user_validation"
    total_cost = {
        treatment: sum(
            run.observed_end_to_end_wall_seconds for run in treatment_runs[treatment]
        )
        for treatment in active_treatments
    }
    evidence: dict[str, object] = {
        "research_question": "RQ13 encoder-decoder prefix expansion",
        "dataset_size": "500m",
        "claims_status": claims_status,
        "result_claims_user_validated": False,
        "selection_rule": "validation Recall@100, then same-epoch NDCG@100, then lower logged horizon training time",
        "boundary_rule": "extend a winning outer deep LR geometrically by a factor of two until the winner is interior",
        "cap_fit": cap_fit,
        "missing_initial_artifacts": missing_initial_artifacts,
        "required_boundary_followups": required_boundary_followups,
        "required_followups": required_followups,
        "required_diagnostics": diagnostics,
        "resolved_diagnostics": resolved_diagnostics,
        "correctness_audit": correctness_record,
        "acceptance_checks": acceptance,
        "control": _control_record(control),
        "surface_winners": {
            treatment: _selected_record(run) for treatment, run in selected.items()
        },
        "selected": {
            treatment: _selected_record(run)
            for treatment, run in resolved_selected.items()
        },
        "selected_method": (
            None if selected_method is None else _selected_record(selected_method)
        ),
        "expansion_comparisons_to_no_expansion": comparisons,
        "timing_definition": {
            "steady_state_targets_per_second": "candidate targets divided by logged train time over epochs 2-20; epoch 1 is excluded",
            "time_through_selected_checkpoint_seconds": "logged train, validation inference, and validation save time through the validation-selected checkpoint",
            "total_required_training_wall_seconds": "Prepared-stage to Final-metrics wall time summed over every LR-tuning and boundary artifact required for the treatment",
        },
        "treatments": {
            treatment: {
                "total_required_training_wall_seconds": total_cost[treatment],
                "artifacts": [
                    _run_record(run)
                    for run in sorted(
                        treatment_runs[treatment],
                        key=lambda item: item.candidate.deep_lr,
                    )
                ],
            }
            for treatment in active_treatments
        },
    }
    ready = (
        len(resolved_selected) == len(active_treatments)
        and claims_status == "ready_for_user_validation"
    )
    pending_boundary_treatments = {
        candidate_by_run(run_name).treatment for run_name in required_boundary_followups
    }
    staged_reader_treatments = tuple(
        treatment
        for treatment in active_treatments
        if treatment in selected and treatment not in pending_boundary_treatments
    )
    staged_reader_ready = (
        cap_extension
        and claims_status == "pending_artifacts"
        and set(_TREATMENTS).issubset(staged_reader_treatments)
    )
    reader_selected = resolved_selected if ready else selected
    reader_treatments = tuple(active_treatments) if ready else staged_reader_treatments
    return Rq13ReportBundle(
        reader_markdown=(
            _reader_markdown(
                reader_selected, total_cost, control, reader_treatments, cap_fit
            )
            if ready or staged_reader_ready
            else None
        ),
        tuning_markdown=_tuning_markdown(
            treatment_runs, selected, tuple(active_treatments)
        ),
        diagnostics_markdown=_combined_diagnostics_markdown(
            resolved_diagnostics, cap_fit
        ),
        evidence=evidence,
    )


def load_rq12_control(path: Path = _RQ12_EVIDENCE) -> Control:
    try:
        evidence = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise Rq13ReportError(
            f"cannot read compatible RQ12 control evidence: {path}"
        ) from error
    if (
        evidence.get("research_question") != "RQ12 decoder-only query layout"
        or evidence.get("dataset_size") != "500m"
    ):
        raise Rq13ReportError("RQ12 control evidence identifies the wrong study")
    methods = [
        method
        for method in evidence.get("methods", [])
        if method.get("method") == "standard"
    ]
    if len(methods) != 1:
        raise Rq13ReportError("RQ12 evidence must contain exactly one standard control")
    method = methods[0]
    artifacts = method.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 3:
        raise Rq13ReportError(
            "RQ12 standard control must retain three selected artifacts"
        )
    first = artifacts[0]
    identities = [artifact.get("dataset_identity") for artifact in artifacts]
    if any(identity != identities[0] for identity in identities[1:]):
        raise Rq13ReportError("RQ12 control artifacts disagree on dataset identity")
    identity = first["dataset_identity"]
    cache_content = identity["sequence_cache_content_sha256"]
    cache_names = identity["sequence_caches"]
    evaluator = {artifact["fingerprints"]["evaluator"] for artifact in artifacts}
    scoring = {artifact["fingerprints"]["common_objective"] for artifact in artifacts}
    if len(evaluator) != 1 or len(scoring) != 1:
        raise Rq13ReportError("RQ12 standard control fingerprints disagree")
    quality = _metric_mapping(method.get("mean_full_user_metrics"), "RQ12 control")
    validation_recalls = []
    for artifact in artifacts:
        validation = artifact.get("validation_metrics")
        recall = validation.get("recall@100") if isinstance(validation, dict) else None
        if not isinstance(recall, (int, float)) or not math.isfinite(recall):
            raise Rq13ReportError("RQ12 control validation Recall@100 is absent")
        validation_recalls.append(float(recall))
    efficiency = method.get("mean_efficiency")
    if not isinstance(efficiency, dict):
        raise Rq13ReportError("RQ12 control efficiency is absent")
    best = efficiency.get("best_epochs_by_seed")
    if not isinstance(best, dict) or sorted(best) != ["42", "43", "44"]:
        raise Rq13ReportError("RQ12 control best-epoch evidence is incomplete")
    total = (
        evidence.get("total_required_cost", {}).get("by_method", {}).get("standard", {})
    )
    return Control(
        label="regular decoder-only SASRec",
        source="RQ12 standard item-state",
        quality=quality,
        original_users_per_epoch=None,
        expanded_examples_per_epoch=_positive_int(
            efficiency, "examples_per_epoch", "RQ12 control"
        ),
        candidate_targets_per_epoch=0,
        ntp_targets_per_epoch=_positive_int(
            efficiency, "next_item_targets_per_epoch", "RQ12 control"
        ),
        input_tokens_per_epoch=_positive_int(
            efficiency, "input_tokens_per_epoch", "RQ12 control"
        ),
        best_epochs=" / ".join(str(best[str(seed)]) for seed in (42, 43, 44)),
        steady_state_targets_per_second=_finite_positive(
            efficiency.get("steady_state_targets_per_second"), "RQ12 control throughput"
        ),
        time_through_selected_checkpoint_seconds=_finite_positive(
            efficiency.get("time_through_selected_checkpoint_seconds"),
            "RQ12 control checkpoint time",
        ),
        total_required_training_wall_seconds=_finite_positive(
            total.get("observed_end_to_end_wall_seconds"), "RQ12 control total wall"
        ),
        dataset_content_manifest_sha256=_canonical_sha256(
            identity["dataset_content_sha256"]
        ),
        validation_cache_manifest_sha256=_canonical_sha256(
            cache_content[cache_names["val"]]
        ),
        query_cache_manifest_sha256=_canonical_sha256(
            cache_content[cache_names["true_metric_query"]]
        ),
        evaluator_fingerprint=evaluator.pop(),
        scoring_fingerprint=scoring.pop(),
        validation_recall_mean=sum(validation_recalls) / len(validation_recalls),
    )


def write_report_bundle(
    bundle: Rq13ReportBundle, scratchpad: Path, evidence: Path
) -> dict[str, Path]:
    paths = {
        "tuning": scratchpad / "rq13_prefix_expansion_tuning_500m.md",
        "evidence": evidence / "rq13_prefix_expansion_results.json",
    }
    _write(paths["tuning"], bundle.tuning_markdown)
    _write(
        paths["evidence"],
        json.dumps(bundle.evidence, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    if bundle.reader_markdown is not None:
        paths["reader"] = scratchpad / "rq13_prefix_expansion_reader_500m.md"
        _write(paths["reader"], bundle.reader_markdown)
    else:
        (scratchpad / "rq13_prefix_expansion_reader_500m.md").unlink(missing_ok=True)
    if bundle.diagnostics_markdown is not None:
        paths["diagnostics"] = scratchpad / "rq13_prefix_expansion_diagnostics_500m.md"
        _write(paths["diagnostics"], bundle.diagnostics_markdown)
    return paths


def _verify_recipe(directory: Path, candidate: QueryCandidate) -> bool:
    return verify_artifact.verify_config(
        directory, _CONFIG, [f"G1_QUERY_RUN={candidate.run_name}"]
    )


def _candidate_from_run_name(run_name: str) -> QueryCandidate:
    candidate = candidate_by_run(run_name)
    if candidate.study != "rq13":
        raise ValueError(f"non-RQ13 run {run_name!r}")
    return candidate


def _load_run(
    directory: Path,
    candidate: QueryCandidate,
    expected_counts: Mapping[str, int] | None,
    facts: ReportFileFacts | None = None,
) -> Run:
    metadata = _load_json(directory / "training_metadata.json")
    metrics = _load_json(directory / "final_metrics.json")
    counts, invariants = _validate_metadata(metadata, candidate, expected_counts)
    timings, prepared, final = _timings(directory)
    validation_curve = _validation_curve(directory)
    selected_epoch, recall, ndcg = validation_curve[counts["best_epoch"] - 1]
    if selected_epoch != counts["best_epoch"]:
        raise Rq13ReportError(
            f"{directory.name}: selected epoch is absent from the validation curve"
        )
    identity = _dataset_identity(directory, prepared, facts=facts)
    evaluator = _canonical_sha256(
        {
            **{key: invariants[key] for key in _EVALUATOR_FIELDS},
            "full_user_count": metrics.get("num_users"),
        }
    )
    scoring = _canonical_sha256(
        {
            **{key: invariants[key] for key in _SCORING_FIELDS},
            "training_semantics_revision": metadata["training_semantics_revision"],
        }
    )
    steady = timings[1:]
    steady_train = sum(item[1] for item in steady)
    if not steady or steady_train <= 0:
        raise Rq13ReportError(f"{directory.name}: no steady-state timing evidence")
    best_timings = timings[: counts["best_epoch"]]
    return Run(
        candidate=candidate,
        best_epoch=counts["best_epoch"],
        stopped_epoch=20,
        validation_recall=recall,
        validation_ndcg=ndcg,
        validation_curve=validation_curve,
        metrics=_metric_mapping(metrics, directory.name),
        original_users_per_epoch=counts["original_users_per_epoch"],
        expanded_examples_per_epoch=counts["expanded_examples_per_epoch"],
        candidate_targets_per_epoch=counts["candidate_targets_per_epoch"],
        ntp_targets_per_epoch=counts["ntp_targets_per_epoch"],
        input_tokens_per_epoch=counts["input_tokens_per_epoch"],
        optimizer_steps_per_epoch=_positive_int(
            metadata, "optimizer_steps_per_epoch", candidate.run_name
        ),
        steady_state_targets_per_second=counts["candidate_targets_per_epoch"]
        * len(steady)
        / steady_train,
        time_through_selected_checkpoint_seconds=sum(
            sum(item[1:]) for item in best_timings
        ),
        required_horizon_train_validation_seconds=sum(
            sum(item[1:]) for item in timings
        ),
        observed_end_to_end_wall_seconds=(final - prepared).total_seconds(),
        dataset_content_manifest_sha256=_canonical_sha256(identity["dataset_content"]),
        train_cache_manifest_sha256=_canonical_sha256(
            identity["cache_content"][identity["caches"]["train"]]
        ),
        validation_cache_manifest_sha256=_canonical_sha256(
            identity["cache_content"][identity["caches"]["val"]]
        ),
        query_cache_manifest_sha256=_canonical_sha256(
            identity["cache_content"][identity["caches"]["true_metric_query"]]
        ),
        evaluator_fingerprint=evaluator,
        scoring_fingerprint=scoring,
        artifact_sha256={
            name: _file_sha256(directory / name, facts)
            for name in ("training_metadata.json", "final_metrics.json", "sweep.log")
        },
    )


def _validate_metadata(
    metadata: dict[str, Any],
    candidate: QueryCandidate,
    expected_counts: Mapping[str, int] | None,
) -> tuple[dict[str, int], dict[str, Any]]:
    context = candidate.run_name
    expected = {
        "training_semantics_revision": GENERATION_TRAINING_SEMANTICS_REVISION,
        "dataset_size": "500m",
        "seed": 42,
        "num_epochs": 20,
        "max_epochs": 20,
        "epochs_trained": 20,
        "stopped_epoch": 20,
        "early_stopped": False,
        "lr_horizon_complete": True,
        "selection_resolved": True,
        "batch_size": 1280,
        "physical_batch_size": 1280,
        "gradient_accumulation_steps": 1,
        "effective_batch_size": 1280,
        "model_dim": 64,
        "item_embedding_dim": 64,
        "embedding_learning_rate": 0.064,
        "deep_learning_rate": candidate.deep_lr,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise Rq13ReportError(
                f"{context}: {key}={metadata.get(key)!r}, expected {value!r}"
            )
    best_epoch = _positive_int(metadata, "best_epoch", context)
    if best_epoch > 20:
        raise Rq13ReportError(f"{context}: best_epoch exceeds the completed horizon")
    invariants = metadata.get("transfer_invariants")
    if not isinstance(invariants, dict):
        raise Rq13ReportError(f"{context}: transfer_invariants is absent")
    cap = _cap(candidate.treatment)
    rule = "required" if candidate.treatment.startswith("required") else "truncated"
    architecture_expected = {
        "query_architecture": "encoder_decoder",
        "prefix_length_rule": rule,
        "prefix_cap": cap,
        "query_slots_shared": False,
        "include_history_memory": False,
        "num_query_slots": 4,
    }
    for key, value in architecture_expected.items():
        if metadata.get(key) != value or invariants.get(key) != value:
            raise Rq13ReportError(f"{context}: incompatible {key} metadata")
    counts: dict[str, int] = {"best_epoch": best_epoch}
    for key in _COUNT_FIELDS:
        top = metadata.get(key)
        invariant = invariants.get(key)
        if key == "ntp_targets_per_epoch":
            valid = isinstance(top, int) and not isinstance(top, bool) and top == 0
        else:
            valid = isinstance(top, int) and not isinstance(top, bool) and top > 0
        if not valid or invariant != top:
            raise Rq13ReportError(
                f"{context}: invalid or inconsistent count metadata {key}"
            )
        counts[key] = top
    if expected_counts is not None and any(
        counts[key] != expected_counts.get(key) for key in _COUNT_FIELDS
    ):
        raise Rq13ReportError(
            f"{context}: count metadata does not match the expected treatment manifest"
        )
    if (
        metadata.get("targets_per_epoch") != counts["candidate_targets_per_epoch"]
        or metadata.get("tokens_per_epoch") != counts["input_tokens_per_epoch"]
    ):
        raise Rq13ReportError(
            f"{context}: generic and architecture count metadata disagree"
        )
    if counts["candidate_targets_per_epoch"] != counts["expanded_examples_per_epoch"]:
        raise Rq13ReportError(
            f"{context}: encoder-decoder needs one candidate target per example"
        )
    if (
        not counts["original_users_per_epoch"]
        <= counts["expanded_examples_per_epoch"]
        <= counts["original_users_per_epoch"] * cap
    ):
        raise Rq13ReportError(f"{context}: expanded example count violates prefix cap")
    if (
        candidate.treatment == "one_example"
        and counts["expanded_examples_per_epoch"] != counts["original_users_per_epoch"]
    ):
        raise Rq13ReportError(
            f"{context}: no-expansion must contain one example per user"
        )
    return counts, invariants


def _timings(
    directory: Path,
) -> tuple[list[tuple[int, float, float, float]], datetime, datetime]:
    lines = (directory / "sweep.log").read_text().splitlines()
    timings: dict[int, tuple[int, float, float, float]] = {}
    prepared = []
    final = []
    for line in lines:
        if "Prepared stage '" in line:
            prepared.append(_timestamp(line, directory.name))
        if "Final metrics (" in line:
            final.append(_timestamp(line, directory.name))
        match = _EPOCH_PATTERN.search(line)
        if match is not None:
            epoch = int(match.group(1))
            values = tuple(float(match.group(index)) for index in range(2, 5))
            if epoch in timings or any(
                not math.isfinite(value) or value < 0 for value in values
            ):
                raise Rq13ReportError(f"{directory.name}: malformed timing evidence")
            timings[epoch] = (epoch, *values)
    if (
        sorted(timings) != list(range(20))
        or len(prepared) != 1
        or len(final) != 1
        or final[0] < prepared[0]
    ):
        raise Rq13ReportError(f"{directory.name}: incomplete timing evidence")
    return [timings[index] for index in range(20)], prepared[0], final[0]


def _validation_curve(directory: Path) -> tuple[tuple[int, float, float], ...]:
    values: dict[int, tuple[float, float]] = {}
    for line in (directory / "sweep.log").read_text().splitlines():
        epoch_match = re.search(r"\bepoch (\d+) finished\b", line)
        if epoch_match is None:
            continue
        recall = re.search(rf"\bepoch/val_true\.recall@100=({_METRIC_NUMBER})\b", line)
        ndcg = re.search(rf"\bepoch/val_true\.ndcg@100=({_METRIC_NUMBER})\b", line)
        if recall is not None and ndcg is not None:
            pair = (float(recall.group(1)), float(ndcg.group(1)))
            epoch = int(epoch_match.group(1)) + 1
            if epoch in values or not all(
                math.isfinite(value) and 0 <= value <= 1 for value in pair
            ):
                raise Rq13ReportError(
                    f"{directory.name}: conflicting validation curve evidence"
                )
            values[epoch] = pair
    if sorted(values) != list(range(1, 21)):
        raise Rq13ReportError(f"{directory.name}: incomplete validation curve evidence")
    return tuple((epoch, *values[epoch]) for epoch in range(1, 21))


def _dataset_identity(
    directory: Path,
    run_start: datetime,
    migration_evidence: Path = _RESULTS_EVIDENCE,
    *,
    facts: ReportFileFacts | None = None,
) -> dict[str, object]:
    lines = (directory / "sweep.log").read_text().splitlines()
    datasets = {
        Path(match.group("path"))
        for line in lines
        if (match := _DATASET_PATH_PATTERN.search(line)) is not None
    }
    caches: dict[str, set[Path]] = {
        "train": set(),
        "val": set(),
        "true_metric_query": set(),
    }
    for line in lines:
        match = _CACHE_PATTERN.search(line)
        if match is not None:
            caches[match.group("split")].add(Path(match.group("path")))
    if len(datasets) != 1 or any(len(paths) != 1 for paths in caches.values()):
        raise Rq13ReportError(f"{directory.name}: incomplete dataset/cache provenance")
    dataset = datasets.pop()
    resolved = {split: paths.pop() for split, paths in caches.items()}
    dataset_content = _content_manifest(
        dataset,
        tuple(dataset / name for name in _DATASET_FILES),
        run_start,
        directory.name,
        facts,
    )
    cache_content = {}
    for split, path in resolved.items():
        cache_content[path.name] = _sequence_cache_content(
            path,
            split,
            run_start,
            directory.name,
            migration_evidence,
            facts,
        )
    return {
        "dataset_content": dataset_content,
        "caches": {split: path.name for split, path in resolved.items()},
        "cache_content": cache_content,
    }


def _sequence_cache_content(
    path: Path,
    split: str,
    run_start: datetime,
    context: str,
    migration_evidence: Path = _RESULTS_EVIDENCE,
    facts: ReportFileFacts | None = None,
) -> dict[str, str]:
    buckets = tuple(sorted((path / "buckets").glob("*.parquet")))
    if not buckets:
        raise Rq13ReportError(f"{context}: {path.name} has no cache buckets")
    bucket_manifest = _content_manifest(path, buckets, run_start, context, facts)
    metadata_path = path / "metadata.json"
    try:
        modified = metadata_path.stat().st_mtime
    except OSError as error:
        raise Rq13ReportError(
            f"{context}: missing provenance file {metadata_path}"
        ) from error
    metadata = _load_json(metadata_path)
    if "event_count" in metadata:
        return _verified_event_count_migration(
            path,
            split,
            metadata,
            bucket_manifest,
            context,
            migration_evidence,
            facts,
        )
    if modified <= run_start.timestamp():
        return {
            "metadata.json": _file_sha256(metadata_path, facts),
            **bucket_manifest,
        }
    raise Rq13ReportError(
        f"{context}: provenance file postdates run start: {metadata_path}"
    )


def _verified_event_count_migration(
    path: Path,
    split: str,
    metadata: dict[str, Any],
    bucket_manifest: dict[str, str],
    context: str,
    migration_evidence: Path,
    facts: ReportFileFacts | None,
) -> dict[str, str]:
    if not migration_evidence.is_file():
        raise Rq13ReportError(
            f"{context}: missing event_count migration evidence {migration_evidence}"
        )
    legacy_keys = (
        "params",
        "bucket_files",
        "bucket_lengths",
        "int_columns",
        "float_columns",
    )
    if set(metadata) != {*legacy_keys, "event_count"}:
        raise Rq13ReportError(
            f"{context}: event_count is not the sole metadata change"
        )
    event_count = metadata["event_count"]
    if not isinstance(event_count, int) or isinstance(event_count, bool):
        raise Rq13ReportError(f"{context}: invalid migrated event_count")
    declared_buckets = metadata["bucket_files"]
    if not isinstance(declared_buckets, list) or not all(
        isinstance(name, str) for name in declared_buckets
    ):
        raise Rq13ReportError(f"{context}: migrated cache bucket list is invalid")
    bucket_names = [Path(name).name for name in declared_buckets]
    actual_names = [Path(name).name for name in sorted(bucket_manifest)]
    if bucket_names != actual_names:
        raise Rq13ReportError(f"{context}: migrated cache bucket list disagrees")
    params = metadata["params"]
    timestamp_column = params.get("timestamp_column") if isinstance(params, dict) else None
    if not isinstance(timestamp_column, str):
        raise Rq13ReportError(f"{context}: migrated cache timestamp column is absent")
    derived = _derived_cache_event_count(
        path,
        timestamp_column,
        tuple(sorted(bucket_manifest.items())),
        facts,
    )
    if event_count != derived:
        raise Rq13ReportError(
            f"{context}: migrated event_count is not exactly derived from buckets"
        )
    legacy = {key: metadata[key] for key in legacy_keys}
    legacy_metadata_sha256 = hashlib.sha256(json.dumps(legacy).encode()).hexdigest()
    manifest = {"metadata.json": legacy_metadata_sha256, **bucket_manifest}
    expected = _legacy_cache_fingerprints(migration_evidence).get(split)
    if expected is None or _canonical_sha256(manifest) != expected:
        raise Rq13ReportError(
            f"{context}: migrated cache does not match retained legacy evidence"
        )
    return manifest


def _derived_cache_event_count(
    path: Path,
    timestamp_column: str,
    bucket_manifest: tuple[tuple[str, str], ...],
    facts: ReportFileFacts | None = None,
) -> int:
    buckets = tuple(path / relative for relative, _ in bucket_manifest)

    def compute() -> int:
        return sum(
            int(
                pl.scan_parquet(bucket)
                .select(pl.col(timestamp_column).list.len().sum())
                .collect()
                .item()
                or 0
            )
            for bucket in buckets
        )
    return (
        compute()
        if facts is None
        else facts.load_or_compute(
            f"parquet_list_length_sum:{timestamp_column}", buckets, compute
        )
    )


def _legacy_cache_fingerprints(path: Path) -> dict[str, str]:
    evidence = _load_json(path)
    if (
        evidence.get("research_question") != "RQ13 encoder-decoder prefix expansion"
        or evidence.get("dataset_size") != "500m"
    ):
        raise Rq13ReportError("event_count migration evidence identifies another study")
    treatments = evidence.get("treatments")
    one_example = treatments.get("one_example") if isinstance(treatments, dict) else None
    artifacts = one_example.get("artifacts") if isinstance(one_example, dict) else None
    if not isinstance(artifacts, list) or not artifacts:
        raise Rq13ReportError("event_count migration evidence has no one-example artifacts")
    fingerprints_by_artifact = [
        artifact.get("compatibility_fingerprints")
        for artifact in artifacts
        if isinstance(artifact, dict)
    ]
    if len(fingerprints_by_artifact) != len(artifacts) or not all(
        isinstance(fingerprints, dict) for fingerprints in fingerprints_by_artifact
    ):
        raise Rq13ReportError("event_count migration evidence has malformed artifacts")
    fields = {
        "train": "train_cache_manifest_sha256",
        "val": "validation_cache_manifest_sha256",
        "true_metric_query": "query_cache_manifest_sha256",
    }
    result = {}
    for split, field in fields.items():
        values = {
            fingerprints.get(field)
            for fingerprints in fingerprints_by_artifact
        }
        if len(values) != 1 or not all(
            isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
            for value in values
        ):
            raise Rq13ReportError(
                f"event_count migration evidence has inconsistent {split} identity"
            )
        result[split] = values.pop()
    if result != _RETAINED_LEGACY_CACHE_FINGERPRINTS:
        raise Rq13ReportError(
            "event_count migration evidence disagrees with pinned legacy identities"
        )
    return dict(_RETAINED_LEGACY_CACHE_FINGERPRINTS)


def _content_manifest(
    root: Path,
    paths: tuple[Path, ...],
    run_start: datetime,
    context: str,
    facts: ReportFileFacts | None = None,
) -> dict[str, str]:
    result = {}
    for path in paths:
        try:
            modified = path.stat().st_mtime
        except OSError as error:
            raise Rq13ReportError(
                f"{context}: missing provenance file {path}"
            ) from error
        if modified > run_start.timestamp():
            raise Rq13ReportError(
                f"{context}: provenance file postdates run start: {path}"
            )
        result[path.relative_to(root).as_posix()] = _file_sha256(path, facts)
    return result


def _require_workload_consistency(runs: list[Run]) -> None:
    original_users = {run.original_users_per_epoch for run in runs}
    if len(original_users) > 1:
        raise Rq13ReportError("RQ13 artifacts disagree on original users per epoch")
    for treatment in sorted({run.candidate.treatment for run in runs}):
        treatment_runs = [run for run in runs if run.candidate.treatment == treatment]
        values = {
            tuple(getattr(run, key) for key in _COUNT_FIELDS) for run in treatment_runs
        }
        if len(values) > 1:
            raise Rq13ReportError(
                f"{treatment}: count metadata varies across learning rates"
            )
        cache_identities = {
            (
                run.train_cache_manifest_sha256,
                run.validation_cache_manifest_sha256,
                run.query_cache_manifest_sha256,
            )
            for run in treatment_runs
        }
        if len(cache_identities) > 1:
            raise Rq13ReportError(
                f"{treatment}: sequence-cache identity varies across learning rates"
            )
    counts = {
        treatment: runs_for_treatment[0].expanded_examples_per_epoch
        for treatment in _TREATMENTS
        if (
            runs_for_treatment := [
                run for run in runs if run.candidate.treatment == treatment
            ]
        )
    }
    if all(treatment in counts for treatment in _TREATMENTS):
        if not (
            counts["truncated_16"] >= counts["truncated_8"] >= counts["one_example"]
            and counts["required_16"] >= counts["required_8"] >= counts["one_example"]
            and counts["truncated_8"] >= counts["required_8"]
            and counts["truncated_16"] >= counts["required_16"]
        ):
            raise Rq13ReportError(
                "RQ13 expanded-example counts contradict the prefix policies"
            )


def _require_control_compatibility(runs: list[Run], control: Control) -> None:
    for run in runs:
        comparisons = {
            "dataset content": (
                run.dataset_content_manifest_sha256,
                control.dataset_content_manifest_sha256,
            ),
            "query cache": (
                run.query_cache_manifest_sha256,
                control.query_cache_manifest_sha256,
            ),
            "evaluator": (run.evaluator_fingerprint, control.evaluator_fingerprint),
            "candidate scoring": (run.scoring_fingerprint, control.scoring_fingerprint),
        }
        mismatches = [name for name, pair in comparisons.items() if pair[0] != pair[1]]
        if mismatches:
            raise Rq13ReportError(
                f"{run.candidate.run_name}: incompatible RQ12 control ({', '.join(mismatches)})"
            )


def _best(runs: list[Run]) -> Run:
    if not runs:
        raise Rq13ReportError("cannot select from an empty RQ13 surface")
    return min(
        runs,
        key=lambda run: (
            -run.validation_recall,
            -run.validation_ndcg,
            run.required_horizon_train_validation_seconds,
            run.candidate.run_name,
        ),
    )


def _select_method(selected: dict[str, Run]) -> Run | None:
    if not set(_TREATMENTS).issubset(selected):
        return None
    best_recall = max(run.validation_recall for run in selected.values())
    eligible = [
        run for run in selected.values() if run.validation_recall >= best_recall - 0.003
    ]
    simplicity = {
        treatment: (
            _cap(treatment),
            1 if treatment.startswith("required_") else 0,
        )
        for treatment in selected
    }
    return min(
        eligible,
        key=lambda run: (
            run.time_through_selected_checkpoint_seconds,
            simplicity[run.candidate.treatment],
            run.candidate.run_name,
        ),
    )


def _method_comparisons(selected: dict[str, Run]) -> dict[str, object]:
    baseline = selected.get("one_example")
    if baseline is None:
        return {}
    return {
        treatment: {
            "validation_recall@100_difference": (
                run.validation_recall - baseline.validation_recall
            ),
            "selected_checkpoint_epoch_difference": (
                run.best_epoch - baseline.best_epoch
            ),
            "selected_checkpoint_time_difference_seconds": (
                run.time_through_selected_checkpoint_seconds
                - baseline.time_through_selected_checkpoint_seconds
            ),
            "full_user_metric_differences": {
                metric: run.metrics[metric] - baseline.metrics[metric]
                for metric in _METRICS
            },
        }
        for treatment, run in selected.items()
        if treatment != "one_example"
    }


def _boundary_followup(winner: Run, runs: list[Run]) -> QueryCandidate | None:
    rates = sorted({run.candidate.deep_lr for run in runs})
    rate = winner.candidate.deep_lr
    if rate == rates[0]:
        direction = "low"
    elif rate == rates[-1]:
        direction = "high"
    else:
        return None
    step = (
        winner.candidate.boundary_step + 1
        if winner.candidate.stage == "lr_boundary"
        and winner.candidate.boundary_direction == direction
        else 1
    )
    anchor = next(
        candidate
        for candidate in _initial_candidates_for_treatment(winner.candidate.treatment)
        if candidate.treatment == winner.candidate.treatment
        and candidate.deep_lr
        == (min(DEEP_LRS) if direction == "low" else max(DEEP_LRS))
    )
    return make_boundary_candidate(anchor, direction, step)


def _resolve_treatment_surface(
    runs: list[Run],
) -> tuple[Run, QueryCandidate | None]:
    initial = [run for run in runs if run.candidate.stage != "lr_boundary"]
    boundary = [run for run in runs if run.candidate.stage == "lr_boundary"]
    if len(initial) != len(DEEP_LRS):
        raise Rq13ReportError("RQ13 treatment does not have its exact initial LR grid")
    current = list(initial)
    for index, run in enumerate(
        sorted(boundary, key=lambda item: item.candidate.boundary_step), start=1
    ):
        expected = _boundary_followup(_best(current), current)
        if (
            run.candidate.boundary_step != index
            or expected is None
            or run.candidate != expected
        ):
            raise Rq13ReportError(
                f"{run.candidate.treatment}: unneeded, noncontiguous, or wrong-direction LR boundary artifact {run.candidate.run_name}"
            )
        current.append(run)
    winner = _best(current)
    return winner, _boundary_followup(winner, current)


def _acceptance_checks(selected: dict[str, Run]) -> dict[str, object]:
    if not set(_TREATMENTS).issubset(selected):
        return {
            "truncated_selected_epoch_order": None,
            "truncated_8_selected_epoch_before_no_expansion": None,
            "truncated_16_selected_epoch_before_truncated_8": None,
            "truncated_16_quality_not_lower_than_truncated_8": None,
            "truncated_16_recall_difference_from_truncated_8": None,
            "truncated_16_quality_regression_past_recall_band": None,
            "quality_regressions_past_recall_band": [],
        }
    no = selected["one_example"]
    eight = selected["truncated_8"]
    sixteen = selected["truncated_16"]
    regressions = [
        treatment
        for treatment in _TREATMENTS[1:]
        if selected[treatment].metrics["recall@100"] < no.metrics["recall@100"] - 0.003
    ]
    return {
        "truncated_selected_epoch_order": sixteen.best_epoch
        < eight.best_epoch
        < no.best_epoch,
        "truncated_8_selected_epoch_before_no_expansion": eight.best_epoch
        < no.best_epoch,
        "truncated_16_selected_epoch_before_truncated_8": sixteen.best_epoch
        < eight.best_epoch,
        "truncated_16_quality_not_lower_than_truncated_8": sixteen.metrics["recall@100"]
        >= eight.metrics["recall@100"],
        "truncated_16_recall_difference_from_truncated_8": sixteen.metrics["recall@100"]
        - eight.metrics["recall@100"],
        "truncated_16_quality_regression_past_recall_band": sixteen.metrics[
            "recall@100"
        ]
        < eight.metrics["recall@100"] - 0.003,
        "quality_regressions_past_recall_band": regressions,
    }


def _resolved_diagnostics(
    selected: dict[str, Run], treatment_runs: dict[str, list[Run]]
) -> dict[str, object]:
    if not set(_TREATMENTS).issubset(selected):
        return {}
    no_expansion = selected["one_example"]
    eight = selected["truncated_8"]
    sixteen = selected["truncated_16"]
    result: dict[str, object] = {}

    cap8_first_matching_epoch = next(
        (
            epoch
            for epoch, recall, _ in eight.validation_curve
            if recall >= no_expansion.validation_recall
        ),
        None,
    )
    if (
        eight.best_epoch >= no_expansion.best_epoch
        and cap8_first_matching_epoch is not None
        and cap8_first_matching_epoch < no_expansion.best_epoch
        and eight.validation_recall > no_expansion.validation_recall
    ):
        result["truncated_8_selected_epoch_vs_no_expansion"] = {
            "status": "explained_by_earlier_threshold_then_higher_late_peak",
            "no_expansion_selected_epoch": no_expansion.best_epoch,
            "no_expansion_selected_validation_recall@100": no_expansion.validation_recall,
            "truncated_8_first_epoch_at_or_above_no_expansion_selected_recall": cap8_first_matching_epoch,
            "truncated_8_recall_at_that_epoch": eight.validation_curve[
                cap8_first_matching_epoch - 1
            ][1],
            "truncated_8_selected_epoch": eight.best_epoch,
            "truncated_8_selected_validation_recall@100": eight.validation_recall,
            "no_expansion_expanded_examples_per_epoch": no_expansion.expanded_examples_per_epoch,
            "truncated_8_expanded_examples_per_epoch": eight.expanded_examples_per_epoch,
            "no_expansion_selected_optimizer_steps": no_expansion.best_epoch
            * no_expansion.optimizer_steps_per_epoch,
            "truncated_8_threshold_optimizer_steps": cap8_first_matching_epoch
            * eight.optimizer_steps_per_epoch,
            "no_expansion_targets_seen_through_selected_epoch": no_expansion.best_epoch
            * no_expansion.candidate_targets_per_epoch,
            "truncated_8_targets_seen_through_threshold_epoch": cap8_first_matching_epoch
            * eight.candidate_targets_per_epoch,
        }

    cap16_first_matching_epoch = next(
        (
            epoch
            for epoch, recall, _ in sixteen.validation_curve
            if recall >= eight.validation_recall
        ),
        None,
    )
    if (
        sixteen.best_epoch < eight.best_epoch
        or cap16_first_matching_epoch is None
        or cap16_first_matching_epoch >= eight.best_epoch
        or sixteen.validation_recall <= eight.validation_recall
    ):
        return result
    lr_comparisons = []
    for deep_lr in DEEP_LRS:
        cap8 = next(
            run
            for run in treatment_runs["truncated_8"]
            if run.candidate.deep_lr == deep_lr
        )
        cap16 = next(
            run
            for run in treatment_runs["truncated_16"]
            if run.candidate.deep_lr == deep_lr
        )
        lr_comparisons.append(
            {
                "deep_learning_rate": deep_lr,
                "truncated_8_best_epoch": cap8.best_epoch,
                "truncated_8_validation_recall@100": cap8.validation_recall,
                "truncated_16_best_epoch": cap16.best_epoch,
                "truncated_16_validation_recall@100": cap16.validation_recall,
            }
        )
    result["truncated_16_selected_epoch_vs_truncated_8"] = {
        "status": "explained_by_earlier_threshold_then_higher_late_peak",
        "truncated_8_selected_epoch": eight.best_epoch,
        "truncated_8_selected_validation_recall@100": eight.validation_recall,
        "truncated_16_first_epoch_at_or_above_truncated_8_selected_recall": cap16_first_matching_epoch,
        "truncated_16_recall_at_that_epoch": sixteen.validation_curve[
            cap16_first_matching_epoch - 1
        ][1],
        "truncated_16_selected_epoch": sixteen.best_epoch,
        "truncated_16_selected_validation_recall@100": sixteen.validation_recall,
        "truncated_8_expanded_examples_per_epoch": eight.expanded_examples_per_epoch,
        "truncated_16_expanded_examples_per_epoch": sixteen.expanded_examples_per_epoch,
        "truncated_8_selected_optimizer_steps": eight.best_epoch
        * eight.optimizer_steps_per_epoch,
        "truncated_16_threshold_optimizer_steps": cap16_first_matching_epoch
        * sixteen.optimizer_steps_per_epoch,
        "truncated_16_selected_optimizer_steps": sixteen.best_epoch
        * sixteen.optimizer_steps_per_epoch,
        "truncated_8_targets_seen_through_selected_epoch": eight.best_epoch
        * eight.candidate_targets_per_epoch,
        "truncated_16_targets_seen_through_threshold_epoch": cap16_first_matching_epoch
        * sixteen.candidate_targets_per_epoch,
        "lr_surface": lr_comparisons,
    }
    result["explanation"] = (
        "the selected checkpoints are later at caps 8 and 16, but each expanded treatment reached the preceding treatment's selected quality in fewer epochs and then continued to a higher peak; the matched-quality epoch crossings do not imply lower work because larger caps used more optimizer steps and targets"
    )
    return result


def _required_diagnostics(
    checks: dict[str, object], resolved: dict[str, object]
) -> list[str]:
    diagnostics = []
    if checks.get("truncated_8_selected_epoch_before_no_expansion") is False and (
        "truncated_8_selected_epoch_vs_no_expansion" not in resolved
    ):
        diagnostics.append(
            "verify prefix counts, history slices, learning curves, and selected epochs for no expansion versus truncated 8"
        )
    if checks.get("truncated_16_selected_epoch_before_truncated_8") is False and (
        "truncated_16_selected_epoch_vs_truncated_8" not in resolved
    ):
        diagnostics.append(
            "verify prefix counts, history slices, learning curves, and selected epochs for truncated 8 versus truncated 16"
        )
    if checks.get("truncated_16_quality_regression_past_recall_band") is True:
        diagnostics.append(
            "verify prefix targets, history slices, learning curves, and LR boundaries because truncated 16 regressed materially versus truncated 8"
        )
    regressions = checks.get("quality_regressions_past_recall_band")
    if isinstance(regressions, list) and regressions:
        diagnostics.append(
            "verify prefix targets, leakage, attention masks, gradient flow, and LR boundaries for regressed methods: "
            + ", ".join(regressions)
        )
    return diagnostics


def _reader_markdown(
    selected: dict[str, Run],
    total_cost: dict[str, float],
    control: Control,
    treatments: tuple[str, ...],
    cap_fit: Mapping[str, object],
) -> str:
    selected_method = _select_method(selected)
    no_expansion = selected["one_example"]
    lines = [
        "## RQ13 — Does bounded prefix expansion improve an encoder-decoder?",
        "",
        "### Candidate-generation quality",
        "",
        "| architecture | prefix expansion | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        _reader_row(
            [
                "encoder-decoder",
                "no expansion",
                *(
                    reporting.absolute(no_expansion.metrics[metric])
                    for metric in _METRICS
                ),
            ],
            selected=no_expansion is selected_method,
        ),
    ]
    for treatment in treatments[1:]:
        run = selected[treatment]
        label = _label(treatment)
        cells = [
            reporting.change_cell(
                run.metrics[metric], no_expansion.metrics[metric], metric
            )
            for metric in _METRICS
        ]
        lines.append(
            _reader_row(
                ["encoder-decoder", label, *cells],
                selected=run is selected_method,
            )
        )
    control_cells = [
        reporting.change_cell(
            control.quality[metric], no_expansion.metrics[metric], metric
        )
        for metric in _METRICS
    ]
    lines.append(f"| {control.label} | none | " + " | ".join(control_cells) + " |")
    lines += [
        "",
        "### Training efficiency",
        "",
        "| architecture | prefix expansion | original users/epoch | expanded examples/epoch | candidate targets/epoch | NTP targets/epoch | input tokens/epoch | selected checkpoint epoch | steady-state targets/s | time through selected checkpoint (train+validation), s | total required training wall (all tuning and boundary artifacts), s |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| {control.label} | none | — | {control.expanded_examples_per_epoch} | {control.candidate_targets_per_epoch} | {control.ntp_targets_per_epoch} | {control.input_tokens_per_epoch} | {control.best_epochs} | {control.steady_state_targets_per_second:.3f} | {control.time_through_selected_checkpoint_seconds:.3f} | {control.total_required_training_wall_seconds:.3f} |",
    ]
    for treatment in treatments:
        run = selected[treatment]
        label = _label(treatment)
        lines.append(
            _reader_row(
                [
                    "encoder-decoder",
                    label,
                    str(run.original_users_per_epoch),
                    str(run.expanded_examples_per_epoch),
                    str(run.candidate_targets_per_epoch),
                    str(run.ntp_targets_per_epoch),
                    str(run.input_tokens_per_epoch),
                    str(run.best_epoch),
                    f"{run.steady_state_targets_per_second:.3f}",
                    f"{run.time_through_selected_checkpoint_seconds:.3f}",
                    f"{total_cost[treatment]:.3f}",
                ],
                selected=run is selected_method,
            )
        )
    if cap_fit.get("status") == "resolved":
        target = cap_fit["reader_success_target"]
        assert isinstance(target, Mapping)
        fitted = selected[f"selected_cap_{cap_fit['selected_cap']}"]
        lines += [
            "",
            "### Aggregated cap-target evaluation",
            "",
            f"The selected practical cap is {cap_fit['selected_cap']}. Its full-user Recall@100 is {fitted.metrics['recall@100']:.6f}; the reader-only 1.10× decoder target is {target['value']:.6f}. Cap selection used validation Recall@100 only.",
        ]
    elif cap_fit.get("status") != "not_requested":
        lines += [
            "",
            "The approved cap-response extension is pending; the completed rows above remain preserved while its additional artifacts are collected.",
        ]
    return "\n".join(lines) + "\n"


def _reader_row(cells: list[str], *, selected: bool) -> str:
    if selected:
        cells = [f"**{cell}**" for cell in cells]
    return "| " + " | ".join(cells) + " |"


def _diagnostics_markdown(resolved: dict[str, object]) -> str:
    cap8 = resolved["truncated_8_selected_epoch_vs_no_expansion"]
    cap16 = resolved["truncated_16_selected_epoch_vs_truncated_8"]
    assert isinstance(cap8, dict) and isinstance(cap16, dict)
    lines = [
        "## RQ13 diagnostic — truncated-prefix selected epochs",
        "",
        "| observation | no expansion | truncated 8 | truncated 16 |",
        "| --- | ---: | ---: | ---: |",
        f"| expanded examples/epoch | {cap8['no_expansion_expanded_examples_per_epoch']} | {cap8['truncated_8_expanded_examples_per_epoch']} | {cap16['truncated_16_expanded_examples_per_epoch']} |",
        f"| selected epoch | {cap8['no_expansion_selected_epoch']} | {cap8['truncated_8_selected_epoch']} | {cap16['truncated_16_selected_epoch']} |",
        f"| selected validation recall@100 | {cap8['no_expansion_selected_validation_recall@100']:.4f} | {cap8['truncated_8_selected_validation_recall@100']:.4f} | {cap16['truncated_16_selected_validation_recall@100']:.4f} |",
        f"| first epoch at preceding row's selected quality | {cap8['no_expansion_selected_epoch']} | {cap8['truncated_8_first_epoch_at_or_above_no_expansion_selected_recall']} | {cap16['truncated_16_first_epoch_at_or_above_truncated_8_selected_recall']} |",
        f"| optimizer steps to that point | {cap8['no_expansion_selected_optimizer_steps']} | {cap8['truncated_8_threshold_optimizer_steps']} | {cap16['truncated_16_threshold_optimizer_steps']} |",
        f"| targets seen to that point | {cap8['no_expansion_targets_seen_through_selected_epoch']} | {cap8['truncated_8_targets_seen_through_threshold_epoch']} | {cap16['truncated_16_targets_seen_through_threshold_epoch']} |",
        "",
        "| deep LR | truncated-8 best epoch / recall@100 | truncated-16 best epoch / recall@100 |",
        "| ---: | ---: | ---: |",
    ]
    for comparison in cap16["lr_surface"]:
        lines.append(
            f"| {comparison['deep_learning_rate']:g} | {comparison['truncated_8_best_epoch']} / {comparison['truncated_8_validation_recall@100']:.4f} | {comparison['truncated_16_best_epoch']} / {comparison['truncated_16_validation_recall@100']:.4f} |"
        )
    lines += ["", str(resolved["explanation"]), ""]
    return "\n".join(lines)


def _combined_diagnostics_markdown(
    resolved: dict[str, object], cap_fit: Mapping[str, object]
) -> str | None:
    parts = []
    if resolved:
        parts.append(_diagnostics_markdown(resolved).rstrip())
    sensitivity = cap_fit.get("sensitivity")
    if isinstance(sensitivity, Mapping):
        fits = sensitivity.get("fits")
        if not isinstance(fits, list) or len(fits) != 16:
            raise Rq13ReportError("RQ13 cap sensitivity does not contain 16 fits")
        lines = [
            "## RQ13 diagnostic — cap-response sensitivity",
            "",
            str(sensitivity["kind"]),
            "",
            "| perturbations at caps 1/4/8/16 | A | B | p | RMSE | fitted recall@100 at cap 32 | fitted target cap |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for fit in fits:
            if not isinstance(fit, Mapping):
                raise Rq13ReportError("RQ13 cap sensitivity fit is malformed")
            perturbations = fit.get("perturbations")
            if not isinstance(perturbations, Mapping):
                raise Rq13ReportError("RQ13 cap perturbations are absent")
            label = "/".join(
                f"{float(perturbations[str(cap)]):+.3f}" for cap in (1, 4, 8, 16)
            )
            target_cap = fit.get("target_cap")
            lines.append(
                f"| {label} | {fit['asymptote']:.6f} | {fit['B']:.6f} | {fit['shape']:.6f} | {fit['rmse']:.6f} | {fit['prediction_at_32']:.6f} | {'infinity' if target_cap is None else target_cap} |"
            )
        parts.append("\n".join(lines))
    return "\n\n".join(parts) + "\n" if parts else None


def _tuning_markdown(
    treatment_runs: dict[str, list[Run]],
    selected: dict[str, Run],
    treatments: tuple[str, ...] = _TREATMENTS,
) -> str:
    lines = ["## RQ13 — Encoder-decoder prefix-expansion tuning"]
    for treatment in treatments:
        lines += [
            "",
            f"### {_label(treatment)}",
            "",
            "| deep LR | batch | best/stopped epoch | validation recall@100 | validation ndcg@100 | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 | original users/epoch | expanded examples/epoch | candidate targets/epoch | NTP targets/epoch | input tokens/epoch | steady-state targets/s | selected-checkpoint time, s | horizon train+validation, s | observed wall, s |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        winner = selected.get(treatment)
        for run in sorted(
            treatment_runs[treatment], key=lambda item: item.candidate.deep_lr
        ):
            row = [
                f"{run.candidate.deep_lr:g}",
                "1280",
                f"{run.best_epoch}/{run.stopped_epoch}",
                f"{run.validation_recall:.6f}",
                f"{run.validation_ndcg:.6f}",
                *(f"{run.metrics[metric]:.6f}" for metric in _METRICS),
                str(run.original_users_per_epoch),
                str(run.expanded_examples_per_epoch),
                str(run.candidate_targets_per_epoch),
                str(run.ntp_targets_per_epoch),
                str(run.input_tokens_per_epoch),
                f"{run.steady_state_targets_per_second:.3f}",
                f"{run.time_through_selected_checkpoint_seconds:.3f}",
                f"{run.required_horizon_train_validation_seconds:.3f}",
                f"{run.observed_end_to_end_wall_seconds:.3f}",
            ]
            if winner is run:
                row = [f"**{cell}**" for cell in row]
            lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def _control_record(control: Control) -> dict[str, object]:
    return {
        "label": control.label,
        "source": control.source,
        "quality": control.quality,
        "compatibility": {
            "dataset_content_manifest_sha256": control.dataset_content_manifest_sha256,
            "validation_cache_manifest_sha256": control.validation_cache_manifest_sha256,
            "query_cache_manifest_sha256": control.query_cache_manifest_sha256,
            "evaluator_fingerprint": control.evaluator_fingerprint,
            "scoring_fingerprint": control.scoring_fingerprint,
        },
    }


def _selected_record(run: Run) -> dict[str, object]:
    return {
        "run_name": run.candidate.run_name,
        "deep_learning_rate": run.candidate.deep_lr,
        "best_epoch": run.best_epoch,
        "validation": {
            "recall@100": run.validation_recall,
            "ndcg@100": run.validation_ndcg,
        },
        "full_user_metrics": run.metrics,
    }


def _run_record(run: Run) -> dict[str, object]:
    return {
        **_selected_record(run),
        "stopped_epoch": run.stopped_epoch,
        "batch_size": 1280,
        "config_recipe_verified": True,
        "query_architecture": "encoder_decoder",
        "prefix_length_rule": (
            "required"
            if run.candidate.treatment.startswith("required")
            else "truncated"
        ),
        "prefix_cap": _cap(run.candidate.treatment),
        **{key: getattr(run, key) for key in _COUNT_FIELDS},
        "steady_state_targets_per_second": run.steady_state_targets_per_second,
        "time_through_selected_checkpoint_seconds": run.time_through_selected_checkpoint_seconds,
        "required_horizon_train_validation_seconds": run.required_horizon_train_validation_seconds,
        "observed_end_to_end_wall_seconds": run.observed_end_to_end_wall_seconds,
        "optimizer_steps_per_epoch": run.optimizer_steps_per_epoch,
        "validation_curve": [
            {"epoch": epoch, "recall@100": recall, "ndcg@100": ndcg}
            for epoch, recall, ndcg in run.validation_curve
        ],
        "compatibility_fingerprints": {
            "dataset_content_manifest_sha256": run.dataset_content_manifest_sha256,
            "train_cache_manifest_sha256": run.train_cache_manifest_sha256,
            "validation_cache_manifest_sha256": run.validation_cache_manifest_sha256,
            "query_cache_manifest_sha256": run.query_cache_manifest_sha256,
            "evaluator_fingerprint": run.evaluator_fingerprint,
            "scoring_fingerprint": run.scoring_fingerprint,
        },
        "artifact_sha256": run.artifact_sha256,
    }


def _cap(treatment: str) -> int:
    return 1 if treatment == "one_example" else int(treatment.rsplit("_", 1)[1])


def _label(treatment: str) -> str:
    return _LABELS.get(
        treatment,
        f"latest {_cap(treatment)} truncated prefixes (fitted practical cap)",
    )


def _initial_candidates_for_treatment(
    treatment: str,
) -> tuple[QueryCandidate, ...]:
    if treatment == _CAP_ANCHOR_TREATMENT:
        return rq13_cap4_candidates()
    if treatment.startswith("selected_cap_"):
        return make_selected_cap_candidates(_cap(treatment))
    return tuple(
        candidate
        for candidate in rq13_initial_candidates()
        if candidate.treatment == treatment
    )


def _metric_mapping(value: object, context: str) -> dict[str, float]:
    if not isinstance(value, dict):
        raise Rq13ReportError(f"{context}: metrics object is absent")
    result = {}
    for metric in _METRICS:
        number = value.get(metric)
        if (
            not isinstance(number, (int, float))
            or isinstance(number, bool)
            or not math.isfinite(number)
            or not 0 <= number <= 1
        ):
            raise Rq13ReportError(f"{context}: invalid {metric}")
        result[metric] = float(number)
    return result


def _positive_int(mapping: Mapping[str, Any], key: str, context: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise Rq13ReportError(f"{context}: {key} must be a positive integer")
    return value


def _finite_positive(value: object, context: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise Rq13ReportError(f"{context} must be finite and positive")
    return float(value)


def _timestamp(line: str, context: str) -> datetime:
    match = _TIMESTAMP_PATTERN.search(line)
    if match is None:
        raise Rq13ReportError(f"{context}: missing timestamp")
    return datetime.fromisoformat(match.group(1).replace(",", "."))


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise Rq13ReportError(f"cannot read {path}") from error
    if not isinstance(value, dict):
        raise Rq13ReportError(f"{path} must contain an object")
    return value


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _file_sha256(path: Path, facts: ReportFileFacts | None = None) -> str:
    if facts is not None:
        return facts.sha256(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        stream.write(content)
        temporary = Path(stream.name)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", type=Path, default=Path("generated/logs"))
    parser.add_argument("--rq12-evidence", type=Path, default=_RQ12_EVIDENCE)
    parser.add_argument(
        "--correctness-evidence", type=Path, default=_CORRECTNESS_EVIDENCE
    )
    parser.add_argument(
        "--scratchpad",
        type=Path,
        default=Path("experiments/g1_sasrec_item_ids_likes/scratchpad"),
    )
    parser.add_argument(
        "--evidence",
        type=Path,
        default=Path("experiments/g1_sasrec_item_ids_likes/evidence"),
    )
    arguments = parser.parse_args()
    bundle = collect_report_bundle(
        arguments.logs,
        rq12_evidence=arguments.rq12_evidence,
        correctness_evidence=arguments.correctness_evidence,
    )
    for path in write_report_bundle(
        bundle, arguments.scratchpad, arguments.evidence
    ).values():
        print(path)


if __name__ == "__main__":
    main()
