from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import tempfile
from typing import Any, Literal

from dcn.training_metadata import GENERATION_TRAINING_SEMANTICS_REVISION
from experiments.g1_sasrec_item_ids_likes.analysis import reporting
from experiments.g1_sasrec_item_ids_likes.analysis.rq13_prefix_expansion_report import (
    _canonical_sha256,
    _dataset_identity,
    _file_sha256,
    _metric_mapping,
    _timings,
    _validation_curve,
)
from experiments.g1_sasrec_item_ids_likes.analysis.rq13_rq14_query_candidates import (
    candidate_by_run as rq14_candidate_by_run,
)
from experiments.g1_sasrec_item_ids_likes.analysis.rq14_query_memory_report import (
    _load_run as load_rq14_run,
)
from experiments.g1_sasrec_item_ids_likes.analysis.rq15_training_candidates import (
    AUXILIARY_DEEP_LRS,
    CANDIDATE_ONLY_DEEP_LRS,
    DEEP_LR_BOUNDARY_RATIOS,
    EMBEDDING_LRS,
    PRETRAINED_FROZEN_EMBEDDING_STEP,
    Rq15Candidate,
    Rq15SourceCandidate,
    candidate_followup_record,
    candidate_by_run,
    initial_candidates,
    make_auxiliary_weight_candidate,
    source_candidate_by_run,
    source_candidates,
    source_checkpoint_metadata,
)
from experiments.g1_sasrec_item_ids_likes.launchers import verify_artifact
from utils.report_file_facts import ReportFileFacts, report_file_facts


ACCEPTANCE_CRITERION = (
    "Adding a pretraining stage should at minimum decrease training time without "
    "losing quality, and will most probably improve the main metrics."
)

_EXPERIMENT = Path(__file__).parents[1]
_RQ14_CONFIG = _EXPERIMENT / "configs/rq13_rq14_query_variant.py"
_TREATMENT_CONFIG = _EXPERIMENT / "configs/rq15_decoder_training_variant.py"
_CHECKPOINT_CONFIG = _EXPERIMENT / "configs/rq15_rq8_checkpoint_variant.py"
_RQ14_RESULTS = _EXPERIMENT / "evidence/rq14_query_memory_results.json"
_CORRECTNESS_EVIDENCE = _EXPERIMENT / "evidence/rq15_training_correctness.json"
_EXPLANATION_EVIDENCE = _EXPERIMENT / "evidence/rq15_training_explanation.json"
_METRICS = ("recall@100", "ndcg@100", "recall@10", "ndcg@10", "coverage@100")
_METHODS = ("scratch_candidate_only", "pretrained_finetune", "auxiliary_ntp")
_METHOD_LABELS = {
    "scratch_candidate_only": "joint scratch, candidate-only",
    "pretrained_finetune": "NTP pretraining, then candidate-only fine-tuning",
    "auxiliary_ntp": "joint scratch, candidate + auxiliary NTP",
}
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
_REQUIRED_ARTIFACTS = ("training_metadata.json", "final_metrics.json", "sweep.log")
_README_START = "<!-- rq15-training-generated:start -->"
_README_END = "<!-- rq15-training-generated:end -->"
_CORRECTNESS_CHECKS = {
    "target_leakage",
    "attention_masks",
    "gradient_flow",
    "separate_loss_normalization_and_counts",
    "checkpoint_copy_identity",
    "config_code_and_artifact_hashes",
}
_IMPLEMENTATION_FILES = (
    Path("dcn/config/generation.py"),
    Path("dcn/config/networks.py"),
    Path("dcn/config/query_retrieval.py"),
    Path("dcn/config/query_retrieval_training.py"),
    Path("dcn/config/retrieval.py"),
    Path("dcn/config/sequence.py"),
    Path("dcn/config/settings.py"),
    Path("dcn/data/packed.py"),
    Path("dcn/data/sequence_dataset.py"),
    Path("dcn/models/cross_attention_retrieval.py"),
    Path("dcn/models/cross_attention_training.py"),
    Path("dcn/models/history_tokens.py"),
    Path("dcn/models/loss_wrapper.py"),
    Path("dcn/models/sequence_retrieval.py"),
    Path("dcn/models/sequence_targets.py"),
    Path("dcn/models/two_tower.py"),
    Path("dcn/nn/sampled_softmax.py"),
    Path("dcn/nn/transformer.py"),
    Path("dcn/training_metadata.py"),
    Path("experiments/g1_sasrec_item_ids_likes/analysis/rq15_training_candidates.py"),
    Path("experiments/g1_sasrec_item_ids_likes/analysis/rq15_training_audit.py"),
    Path("experiments/g1_sasrec_item_ids_likes/analysis/rq15_training_report.py"),
    Path("experiments/g1_sasrec_item_ids_likes/analysis/rq8_reinvestigation_candidates.py"),
    Path("experiments/g1_sasrec_item_ids_likes/configs/rq8_reinvestigation_variant.py"),
    Path("experiments/g1_sasrec_item_ids_likes/configs/rq15_decoder_training_variant.py"),
    Path("experiments/g1_sasrec_item_ids_likes/configs/rq15_rq8_checkpoint_variant.py"),
    Path("experiments/g1_sasrec_item_ids_likes/configs/variant.py"),
    Path("neuralrec/run/train.py"),
    Path("neuralrec/utils/__init__.py"),
    Path("neuralrec/utils/utils.py"),
)

RunRole = Literal[
    "scratch_candidate_only",
    "checkpoint_pretraining",
    "pretrained_finetune",
    "auxiliary_ntp",
]


class Rq15ReportError(RuntimeError):
    pass


@dataclass(frozen=True)
class Run:
    role: RunRole | str
    run_name: str
    candidate: Rq15Candidate | None
    embedding_lr: float
    deep_lr: float
    auxiliary_ntp_weight: float
    best_epoch: int
    stopped_epoch: int
    validation_recall: float
    validation_ndcg: float
    validation_curve: tuple[tuple[int, float, float], ...]
    metrics: dict[str, float]
    original_users_per_epoch: int | None
    expanded_examples_per_epoch: int | None
    candidate_targets_per_epoch: int
    ntp_targets_per_epoch: int
    input_tokens_per_epoch: int
    optimizer_steps_per_epoch: int
    steady_state_candidate_targets_per_second: float | None
    steady_state_total_targets_per_second: float
    time_through_selected_checkpoint_seconds: float
    required_horizon_train_validation_seconds: float
    observed_end_to_end_wall_seconds: float
    dataset_fingerprint: str
    train_cache_fingerprint: str
    validation_cache_fingerprint: str
    query_cache_fingerprint: str
    evaluator_fingerprint: str
    scoring_fingerprint: str
    checkpoint_sha256: str | None
    artifact_sha256: dict[str, str]


@dataclass(frozen=True)
class Rq15ReportBundle:
    reader_markdown: str | None
    readme_section_markdown: str | None
    tuning_markdown: str
    diagnostics_markdown: str
    evidence: dict[str, object]


def collect_report_bundle(
    logs: Path,
    *,
    rq14_results: Path = _RQ14_RESULTS,
    correctness_evidence: Path = _CORRECTNESS_EVIDENCE,
    explanation_evidence: Path = _EXPLANATION_EVIDENCE,
    result_claims_user_validated: bool = False,
) -> Rq15ReportBundle:
    facts = report_file_facts(logs.parent)
    checkpoints = []
    for source in source_candidates():
        checkpoint = _collect_checkpoint(logs / source.run_name, source, facts)
        if checkpoint is not None:
            checkpoints.append(checkpoint)
    selected_checkpoint = _select_checkpoint(checkpoints)
    selected_source = (
        None
        if selected_checkpoint is None
        else source_candidate_by_run(selected_checkpoint.run_name)
    )
    selected_checkpoint_path = (
        None if selected_source is None else selected_source.checkpoint_path(logs)
    )
    scratch = _collect_scratch(logs, rq14_results, facts)
    treatment_runs = []
    for directory in sorted(logs.glob("g1_rq15_*_500m")):
        if not directory.is_dir() or any(
            directory.name == source.run_name for source in source_candidates()
        ):
            continue
        try:
            candidate = candidate_by_run(directory.name)
        except ValueError:
            continue
        if not _complete_directory(directory):
            continue
        assignments = [f"G1_RQ15_RUN={candidate.run_name}"]
        if candidate.training_method == "pretrained_finetune":
            if selected_source is None or selected_checkpoint_path is None:
                continue
            assignments.append(f"G1_RQ15_SOURCE_RUN={selected_source.run_name}")
        if not verify_artifact.verify_config(directory, _TREATMENT_CONFIG, assignments):
            raise Rq15ReportError(f"{candidate.run_name}: recipe-incompatible artifact")
        treatment_runs.append(
            _load_treatment_run(
                directory,
                candidate,
                selected_checkpoint,
                facts,
            )
        )
    return build_report_bundle(
        scratch,
        checkpoints,
        treatment_runs,
        correctness_evidence=_load_optional_json(correctness_evidence),
        explanation_evidence=_load_optional_json(explanation_evidence),
        current_implementation_hash=current_implementation_sha256(facts),
        result_claims_user_validated=result_claims_user_validated,
    )


def build_report_bundle(
    scratch: Run | None,
    checkpoint_runs: Iterable[Run],
    treatment_runs: Iterable[Run],
    *,
    correctness_evidence: Mapping[str, object] | None = None,
    explanation_evidence: Mapping[str, object] | None = None,
    current_implementation_hash: object | None = None,
    result_claims_user_validated: bool = False,
) -> Rq15ReportBundle:
    checkpoints = list(checkpoint_runs)
    runs = list(treatment_runs)
    checkpoint = _select_checkpoint(checkpoints)
    _validate_roles(scratch, checkpoints, runs)
    _validate_cross_run_identity(scratch, checkpoints, runs)
    by_name = {run.run_name: run for run in runs}
    if len(by_name) != len(runs):
        raise Rq15ReportError("duplicate RQ15 artifact identity")
    expected = {
        candidate.run_name
        for candidate in initial_candidates()
        if not (
            candidate.training_method == "scratch_candidate_only"
            and candidate.embedding_lr == 0.064
            and candidate.deep_lr == 0.0015
            and scratch is not None
        )
    }
    missing = sorted(expected - set(by_name))
    if scratch is None:
        missing.insert(0, "RQ14 distinct-CLS control")
    missing[:0] = _missing_checkpoint_runs(checkpoints)

    grouped = {
        method: [run for run in runs if run.role == method]
        for method in _METHODS
    }
    if scratch is not None:
        grouped["scratch_candidate_only"].append(scratch)
    selected_surfaces: dict[str, Run] = {}
    boundary_followups: list[dict[str, object]] = []
    if not missing:
        for method, method_runs in grouped.items():
            selected, followups = _resolve_lr_surface(method_runs)
            selected_surfaces[method] = selected
            boundary_followups.extend(
                _candidate_followup(candidate) for candidate in followups
            )

    auxiliary_followups: list[dict[str, object]] = []
    if (
        not missing
        and not boundary_followups
        and "scratch_candidate_only" in selected_surfaces
        and selected_surfaces["auxiliary_ntp"].metrics["recall@100"]
        < selected_surfaces["scratch_candidate_only"].metrics["recall@100"] - 0.003
    ):
        selected_surfaces["auxiliary_ntp"], auxiliary_candidates = (
            _resolve_auxiliary_weights(
                selected_surfaces["auxiliary_ntp"],
                grouped["auxiliary_ntp"],
            )
        )
        auxiliary_followups = [
            _candidate_followup(candidate) for candidate in auxiliary_candidates
        ]
    elif not missing and not boundary_followups:
        weighted = [
            run
            for run in grouped["auxiliary_ntp"]
            if run.candidate is not None
            and run.candidate.stage == "auxiliary_weight"
        ]
        if weighted:
            raise Rq15ReportError(
                "auxiliary-weight artifacts exist without a weight-1 regression"
            )

    required_followups = [*boundary_followups, *auxiliary_followups]
    acceptance: dict[str, object] = {"status": "pending"}
    selected_method: Run | None = None
    method_costs: dict[str, float] = {}
    if (
        not missing
        and not required_followups
        and "scratch_candidate_only" in selected_surfaces
        and checkpoint is not None
    ):
        selected = {
            **selected_surfaces,
        }
        method_costs = _method_cold_start_costs(selected, checkpoint)
        selected_method = _select_method(selected, method_costs)
        acceptance = _acceptance(
            selected_surfaces["scratch_candidate_only"],
            selected_surfaces["pretrained_finetune"],
            checkpoint,
        )

    implementation_hash = (
        current_implementation_sha256()
        if current_implementation_hash is None
        else current_implementation_hash
    )
    run_artifacts = {
        run.run_name: run.artifact_sha256
        for run in (scratch, *checkpoints, *runs)
        if run is not None
    }
    correctness_result_binding = {
        "missing_artifacts": missing,
        "required_followups": required_followups,
        "checkpoint_pretraining_run_name": (
            None if checkpoint is None else checkpoint.run_name
        ),
        "surface_winner_run_names": {
            method: run.run_name for method, run in selected_surfaces.items()
        },
    }
    correctness = _validate_correctness_evidence(
        correctness_evidence,
        run_artifacts,
        implementation_hash,
        correctness_result_binding,
    )
    explanation: dict[str, object] = {"status": "not_required"}
    if (
        acceptance.get("minimum_acceptance_met") is False
        and correctness.get("status") == "passed"
        and checkpoint is not None
        and len(selected_surfaces) == 3
    ):
        explanation = _validate_explanation_evidence(
            explanation_evidence,
            _selected_artifact_binding(checkpoint, selected_surfaces),
            acceptance,
            correctness_evidence,
        )

    if missing:
        claims_status = "pending_artifacts"
    elif boundary_followups:
        claims_status = "pending_boundary"
    elif auxiliary_followups:
        claims_status = "pending_auxiliary_weights"
    elif correctness.get("status") != "passed":
        claims_status = "correctness_audit_required"
    elif (
        acceptance.get("minimum_acceptance_met") is not True
        and explanation.get("status") != "passed"
    ):
        claims_status = "acceptance_requires_explanation"
    else:
        claims_status = "ready_for_user_validation"

    if result_claims_user_validated:
        if claims_status != "ready_for_user_validation":
            raise Rq15ReportError("cannot validate unresolved RQ15 claims")
        claims_status = "complete"

    evidence = {
        "schema_version": 1,
        "research_question": "RQ15 decoder-decoder training method",
        "dataset_size": "500m",
        "claims_status": claims_status,
        "result_claims_user_validated": result_claims_user_validated,
        "acceptance_criterion": ACCEPTANCE_CRITERION,
        "selection_rule": (
            "within each method: validation Recall@100, then same-epoch "
            "NDCG@100, then lower required-horizon training time; methods "
            "within 0.003 Recall@100 choose lower cold-start cost, then scratch"
        ),
        "boundary_rule": (
            "follow the global validation winner iteratively; an embedding "
            "boundary adds a full three-deep-LR row, while a deep boundary adds "
            "one point at that embedding using the method grid ratio (two for "
            "scratch/pretrained, four for auxiliary)"
        ),
        "missing_artifacts": missing,
        "required_followups": required_followups,
        "acceptance": acceptance,
        "correctness_audit": correctness,
        "experimental_explanation": explanation,
        "artifact_audit": _artifact_audit(scratch, checkpoints, runs),
        "checkpoint_pretraining": (
            None if checkpoint is None else _run_record(checkpoint)
        ),
        "checkpoint_pretraining_surface": [
            _run_record(run)
            for run in sorted(checkpoints, key=lambda item: item.deep_lr)
        ],
        "scratch_control": (
            None
            if "scratch_candidate_only" not in selected_surfaces
            else _run_record(selected_surfaces["scratch_candidate_only"])
        ),
        "surface_winners": {
            method: _run_record(run) for method, run in selected_surfaces.items()
        },
        "selected_method": (
            None if selected_method is None else _run_record(selected_method)
        ),
        "method_cold_start_seconds": method_costs,
        "treatments": {
            method: {
                "total_tuning_wall_seconds": sum(
                    run.observed_end_to_end_wall_seconds for run in grouped[method]
                )
                + (
                    0
                    if method != "pretrained_finetune" or checkpoint is None
                    else sum(
                        run.observed_end_to_end_wall_seconds for run in checkpoints
                    )
                ),
                "artifacts": [_run_record(run) for run in sorted(grouped[method], key=lambda item: item.deep_lr)],
            }
            for method in grouped
        },
    }
    ready = claims_status in {"ready_for_user_validation", "complete"}
    reader = None
    readme = None
    if ready and scratch is not None and checkpoint is not None:
        selected = dict(selected_surfaces)
        reader = _reader_markdown(selected, checkpoint, checkpoints, grouped)
        readme = _readme_section(
            reader,
            acceptance,
            selected_method,
            selected,
            checkpoint,
            explanation_status=str(explanation.get("status")),
        )
    return Rq15ReportBundle(
        reader_markdown=reader,
        readme_section_markdown=readme,
        tuning_markdown=_tuning_markdown(
            checkpoints, checkpoint, grouped, selected_surfaces
        ),
        diagnostics_markdown=_diagnostics_markdown(
            scratch, checkpoint, selected_surfaces, acceptance, required_followups
        ),
        evidence=evidence,
    )


def validate_treatment_metadata(
    metadata: Mapping[str, Any],
    candidate: Rq15Candidate,
    *,
    expected_checkpoint_sha256: str | None,
    expected_checkpoint_source: Rq15SourceCandidate | None = None,
) -> tuple[dict[str, int], str | None]:
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
        "embedding_learning_rate": candidate.embedding_lr,
        "deep_learning_rate": candidate.deep_lr,
        "query_architecture": "decoder_decoder",
        "prefix_length_rule": "truncated",
        "prefix_cap": 1,
        "query_slots_shared": False,
        "include_history_memory": False,
        "num_query_slots": 4,
        "training_method": candidate.training_method,
        "auxiliary_ntp_weight": (
            candidate.auxiliary_ntp_weight
            if candidate.training_method == "auxiliary_ntp"
            else 0.0
        ),
        "loss_normalization": "candidate_and_ntp_separately_mean_normalized",
    }
    invariants = metadata.get("transfer_invariants")
    if not isinstance(invariants, Mapping):
        raise Rq15ReportError(f"{candidate.run_name}: transfer invariants are absent")
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise Rq15ReportError(
                f"{candidate.run_name}: {key}={metadata.get(key)!r}, expected {value!r}"
            )
        if key in {
            "query_architecture",
            "prefix_length_rule",
            "prefix_cap",
            "query_slots_shared",
            "include_history_memory",
            "num_query_slots",
            "training_method",
            "auxiliary_ntp_weight",
            "loss_normalization",
        } and invariants.get(key) != value:
            raise Rq15ReportError(f"{candidate.run_name}: inconsistent {key} invariant")
    counts = {
        name: _nonnegative_int(metadata, name, candidate.run_name)
        for name in (
            "original_users_per_epoch",
            "expanded_examples_per_epoch",
            "candidate_targets_per_epoch",
            "ntp_targets_per_epoch",
            "input_tokens_per_epoch",
        )
    }
    if any(invariants.get(name) != value for name, value in counts.items()):
        raise Rq15ReportError(f"{candidate.run_name}: count invariants disagree")
    if not (
        counts["original_users_per_epoch"]
        == counts["expanded_examples_per_epoch"]
        == counts["candidate_targets_per_epoch"]
        > 0
    ):
        raise Rq15ReportError(f"{candidate.run_name}: candidate target identity is invalid")
    expected_ntp_positive = candidate.training_method == "auxiliary_ntp"
    if (counts["ntp_targets_per_epoch"] > 0) is not expected_ntp_positive:
        raise Rq15ReportError(f"{candidate.run_name}: NTP target identity is invalid")
    if metadata.get("targets_per_epoch") != (
        counts["candidate_targets_per_epoch"] + counts["ntp_targets_per_epoch"]
    ) or metadata.get("tokens_per_epoch") != counts["input_tokens_per_epoch"]:
        raise Rq15ReportError(f"{candidate.run_name}: generic target counts disagree")
    initialization = metadata.get("first_stage_initialization")
    loaded_sha = _validate_initialization(
        initialization,
        candidate,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_checkpoint_source=expected_checkpoint_source,
    )
    if invariants.get("first_stage_initialization") != initialization:
        raise Rq15ReportError(f"{candidate.run_name}: initialization invariant disagrees")
    best_epoch = _positive_int(metadata, "best_epoch", candidate.run_name)
    if best_epoch > 20:
        raise Rq15ReportError(f"{candidate.run_name}: best epoch exceeds horizon")
    counts["best_epoch"] = best_epoch
    return counts, loaded_sha


def write_report_bundle(
    bundle: Rq15ReportBundle,
    scratchpad: Path,
    evidence: Path,
    *,
    readme: Path | None = None,
) -> dict[str, Path]:
    paths = {
        "tuning": scratchpad / "rq15_training_tuning_500m.md",
        "diagnostics": scratchpad / "rq15_training_diagnostics_500m.md",
        "evidence": evidence / "rq15_training_results.json",
    }
    _write(paths["tuning"], bundle.tuning_markdown)
    _write(paths["diagnostics"], bundle.diagnostics_markdown)
    _write(
        paths["evidence"],
        json.dumps(bundle.evidence, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    reader_path = scratchpad / "rq15_training_reader_500m.md"
    if bundle.reader_markdown is None:
        reader_path.unlink(missing_ok=True)
        return paths
    paths["reader"] = reader_path
    _write(reader_path, bundle.reader_markdown)
    if readme is not None and bundle.readme_section_markdown is not None:
        _update_readme(readme, bundle.readme_section_markdown)
        paths["readme"] = readme
    return paths


def _collect_scratch(
    logs: Path,
    results_path: Path,
    facts: ReportFileFacts | None = None,
) -> Run | None:
    if not results_path.is_file():
        return None
    results = _load_json(results_path)
    record = results.get("rq15_distinct_memory")
    if not isinstance(record, Mapping):
        return None
    run_name = record.get("run_name")
    if not isinstance(run_name, str):
        raise Rq15ReportError("RQ14 RQ15 control has no run identity")
    candidate = rq14_candidate_by_run(run_name)
    if candidate.treatment != "distinct_cls_only":
        raise Rq15ReportError("RQ15 control is not distinct-CLS, CLS-only")
    directory = logs / run_name
    if not _complete_directory(directory):
        return None
    if not verify_artifact.verify_config(
        directory, _RQ14_CONFIG, [f"G1_QUERY_RUN={run_name}"]
    ):
        raise Rq15ReportError(f"{run_name}: recipe-incompatible scratch control")
    source = load_rq14_run(directory, candidate, facts)
    if record.get("artifact_sha256") not in (None, source.artifact_sha256):
        raise Rq15ReportError("RQ14 RQ15 control artifact digest disagrees")
    _, prepared, _ = _timings(directory)
    identity = _dataset_identity(directory, prepared, facts=facts)
    return Run(
        role="scratch_candidate_only",
        run_name=run_name,
        candidate=None,
        embedding_lr=0.064,
        deep_lr=candidate.deep_lr,
        auxiliary_ntp_weight=0.0,
        best_epoch=source.best_epoch,
        stopped_epoch=source.stopped_epoch,
        validation_recall=source.validation_recall,
        validation_ndcg=source.validation_ndcg,
        validation_curve=source.validation_curve,
        metrics=source.metrics,
        original_users_per_epoch=source.original_users_per_epoch,
        expanded_examples_per_epoch=source.expanded_examples_per_epoch,
        candidate_targets_per_epoch=source.candidate_targets_per_epoch,
        ntp_targets_per_epoch=source.ntp_targets_per_epoch,
        input_tokens_per_epoch=source.input_tokens_per_epoch,
        optimizer_steps_per_epoch=source.optimizer_steps_per_epoch,
        steady_state_candidate_targets_per_second=source.steady_state_targets_per_second,
        steady_state_total_targets_per_second=source.steady_state_targets_per_second,
        time_through_selected_checkpoint_seconds=source.time_through_selected_checkpoint_seconds,
        required_horizon_train_validation_seconds=source.required_horizon_train_validation_seconds,
        observed_end_to_end_wall_seconds=source.observed_end_to_end_wall_seconds,
        dataset_fingerprint=_canonical_sha256(identity["dataset_content"]),
        train_cache_fingerprint=source.train_cache,
        validation_cache_fingerprint=source.validation_cache,
        query_cache_fingerprint=source.query_cache,
        evaluator_fingerprint=source.evaluator_fingerprint,
        scoring_fingerprint=source.scoring_fingerprint,
        checkpoint_sha256=None,
        artifact_sha256=source.artifact_sha256,
    )


def _collect_checkpoint(
    directory: Path,
    source: Rq15SourceCandidate,
    facts: ReportFileFacts | None = None,
) -> Run | None:
    checkpoint_path = source.checkpoint_path(directory.parent)
    if not _complete_directory(directory) or not checkpoint_path.is_file():
        return None
    if not verify_artifact.verify_config(
        directory,
        _CHECKPOINT_CONFIG,
        [f"G1_RQ15_SOURCE_RUN={source.run_name}"],
    ):
        raise Rq15ReportError(f"{directory.name}: recipe-incompatible checkpoint run")
    checkpoint_sha256 = _validate_checkpoint_document(checkpoint_path, source, facts)
    metadata = _load_json(directory / "training_metadata.json")
    _validate_checkpoint_metadata(metadata, source)
    ntp_targets = _positive_int(metadata, "targets_per_epoch", directory.name)
    input_tokens = _positive_int(metadata, "tokens_per_epoch", directory.name)
    examples = input_tokens - ntp_targets
    if examples <= 0:
        raise Rq15ReportError(
            f"{directory.name}: cannot derive NTP examples from tokens and targets"
        )
    return _load_generic_run(
        directory,
        role="checkpoint_pretraining",
        candidate=None,
        metadata=metadata,
        candidate_targets=0,
        ntp_targets=ntp_targets,
        checkpoint_sha256=checkpoint_sha256,
        original_users=examples,
        expanded_examples=examples,
        facts=facts,
    )


def _load_treatment_run(
    directory: Path,
    candidate: Rq15Candidate,
    expected_checkpoint: Run | None,
    facts: ReportFileFacts | None = None,
) -> Run:
    expected_source = (
        None
        if expected_checkpoint is None
        else source_candidate_by_run(expected_checkpoint.run_name)
    )
    metadata = _load_json(directory / "training_metadata.json")
    counts, loaded_sha = validate_treatment_metadata(
        metadata,
        candidate,
        expected_checkpoint_sha256=(
            None if expected_checkpoint is None else expected_checkpoint.checkpoint_sha256
        ),
        expected_checkpoint_source=expected_source,
    )
    return _load_generic_run(
        directory,
        role=candidate.training_method,
        candidate=candidate,
        metadata=metadata,
        candidate_targets=counts["candidate_targets_per_epoch"],
        ntp_targets=counts["ntp_targets_per_epoch"],
        checkpoint_sha256=loaded_sha,
        original_users=counts["original_users_per_epoch"],
        expanded_examples=counts["expanded_examples_per_epoch"],
        facts=facts,
    )


def _load_generic_run(
    directory: Path,
    *,
    role: RunRole,
    candidate: Rq15Candidate | None,
    metadata: Mapping[str, Any],
    candidate_targets: int,
    ntp_targets: int,
    checkpoint_sha256: str | None,
    original_users: int | None = None,
    expanded_examples: int | None = None,
    facts: ReportFileFacts | None = None,
) -> Run:
    metrics_document = _load_json(directory / "final_metrics.json")
    metrics = _metric_mapping(metrics_document, directory.name)
    timings, prepared, final = _timings(directory)
    curve = _validation_curve(directory)
    best_epoch = _positive_int(metadata, "best_epoch", directory.name)
    selected = _selected_validation_point(curve, best_epoch, directory.name)
    identity = _dataset_identity(directory, prepared, facts=facts)
    invariants = metadata.get("transfer_invariants")
    if not isinstance(invariants, Mapping):
        raise Rq15ReportError(f"{directory.name}: transfer invariants are absent")
    caches = identity["caches"]
    cache_content = identity["cache_content"]
    evaluator = _canonical_sha256(
        {
            **{key: invariants[key] for key in _EVALUATOR_FIELDS},
            "full_user_count": metrics_document.get("num_users"),
        }
    )
    scoring = _canonical_sha256(
        {
            **{key: invariants[key] for key in _SCORING_FIELDS},
            "training_semantics_revision": metadata["training_semantics_revision"],
        }
    )
    steady = timings[1:]
    steady_seconds = sum(point[1] for point in steady)
    total_targets = candidate_targets + ntp_targets
    if steady_seconds <= 0 or total_targets <= 0:
        raise Rq15ReportError(f"{directory.name}: invalid throughput evidence")
    best_timings = timings[:best_epoch]
    return Run(
        role=role,
        run_name=directory.name,
        candidate=candidate,
        embedding_lr=float(metadata["embedding_learning_rate"]),
        deep_lr=float(metadata["deep_learning_rate"]),
        auxiliary_ntp_weight=float(metadata.get("auxiliary_ntp_weight", 0.0)),
        best_epoch=best_epoch,
        stopped_epoch=_positive_int(metadata, "stopped_epoch", directory.name),
        validation_recall=selected[1],
        validation_ndcg=selected[2],
        validation_curve=curve,
        metrics=metrics,
        original_users_per_epoch=original_users,
        expanded_examples_per_epoch=expanded_examples,
        candidate_targets_per_epoch=candidate_targets,
        ntp_targets_per_epoch=ntp_targets,
        input_tokens_per_epoch=_positive_int(metadata, "tokens_per_epoch", directory.name),
        optimizer_steps_per_epoch=_positive_int(metadata, "optimizer_steps_per_epoch", directory.name),
        steady_state_candidate_targets_per_second=(
            None
            if candidate_targets == 0
            else candidate_targets * len(steady) / steady_seconds
        ),
        steady_state_total_targets_per_second=total_targets * len(steady) / steady_seconds,
        time_through_selected_checkpoint_seconds=sum(sum(point[1:]) for point in best_timings),
        required_horizon_train_validation_seconds=sum(sum(point[1:]) for point in timings),
        observed_end_to_end_wall_seconds=(final - prepared).total_seconds(),
        dataset_fingerprint=_canonical_sha256(identity["dataset_content"]),
        train_cache_fingerprint=_canonical_sha256(cache_content[caches["train"]]),
        validation_cache_fingerprint=_canonical_sha256(cache_content[caches["val"]]),
        query_cache_fingerprint=_canonical_sha256(cache_content[caches["true_metric_query"]]),
        evaluator_fingerprint=evaluator,
        scoring_fingerprint=scoring,
        checkpoint_sha256=checkpoint_sha256,
        artifact_sha256={
            name: _file_sha256(directory / name, facts)
            for name in _REQUIRED_ARTIFACTS
        }
        | (
            {
                source_candidate_by_run(directory.name).checkpoint_name:
                checkpoint_sha256
            }
            if checkpoint_sha256 and role == "checkpoint_pretraining"
            else {}
        ),
    )


def _validate_checkpoint_metadata(
    metadata: Mapping[str, Any], source: Rq15SourceCandidate
) -> None:
    context = source.run_name
    expected = source_checkpoint_metadata(source)
    top = {
        "training_semantics_revision": GENERATION_TRAINING_SEMANTICS_REVISION,
        "dataset_size": expected["dataset_size"],
        "seed": expected["seed"],
        "num_epochs": expected["horizon_epochs"],
        "max_epochs": expected["horizon_epochs"],
        "epochs_trained": expected["horizon_epochs"],
        "stopped_epoch": expected["horizon_epochs"],
        "early_stopped": False,
        "lr_horizon_complete": True,
        "selection_resolved": True,
        "batch_size": expected["batch_size"],
        "physical_batch_size": expected["batch_size"],
        "effective_batch_size": expected["batch_size"],
        "embedding_learning_rate": expected["embedding_learning_rate"],
        "deep_learning_rate": expected["deep_learning_rate"],
        "model_dim": 64,
        "item_embedding_dim": 64,
    }
    for key, value in top.items():
        if metadata.get(key) != value:
            raise Rq15ReportError(f"{context}: incompatible checkpoint {key}")
    invariants = metadata.get("transfer_invariants")
    if not isinstance(invariants, Mapping):
        raise Rq15ReportError(f"{context}: checkpoint invariants are absent")
    if (
        invariants.get("window") != "next_item"
        or invariants.get("max_seq_len") != 128
        or invariants.get("cls_token_mode") != "none"
        or invariants.get("evaluation_catalog") != "all"
        or invariants.get("exclude_seen_from_evaluation") is not False
    ):
        raise Rq15ReportError(f"{context}: checkpoint objective/data identity is invalid")
    _positive_int(metadata, "best_epoch", context)
    _positive_int(metadata, "targets_per_epoch", context)
    _positive_int(metadata, "tokens_per_epoch", context)


def _validate_checkpoint_document(
    path: Path,
    source: Rq15SourceCandidate,
    facts: ReportFileFacts | None = None,
) -> str:
    expected = source_checkpoint_metadata(source)

    def validate() -> bool:
        import torch

        try:
            document = torch.load(path, map_location="cpu", weights_only=True)
        except (OSError, RuntimeError, ValueError) as error:
            raise Rq15ReportError(f"cannot read checkpoint {path}") from error
        return (
            isinstance(document, Mapping)
            and document.get("schema_version") == 1
            and document.get("metadata") == expected
            and document.get("history_position_count") == 128
            and document.get("model_dim") == 64
            and document.get("item_embedding_dim") == 64
            and isinstance(document.get("catalog_size"), int)
            and document.get("catalog_size") > 0
            and isinstance(document.get("tokenizer"), Mapping)
            and isinstance(document.get("memory_encoder"), Mapping)
            and isinstance(document.get("item_embedding"), Mapping)
        )

    valid = (
        validate()
        if facts is None
        else facts.load_or_compute(
            "rq15_checkpoint_document:" + _canonical_sha256(expected),
            (path,),
            validate,
        )
    )
    if not valid:
        raise Rq15ReportError("first-stage checkpoint document is incompatible")
    return _file_sha256(path, facts)


def _validate_initialization(
    value: object,
    candidate: Rq15Candidate,
    *,
    expected_checkpoint_sha256: str | None,
    expected_checkpoint_source: Rq15SourceCandidate | None,
) -> str | None:
    if candidate.training_method in {"scratch_candidate_only", "auxiliary_ntp"}:
        if value != "scratch":
            raise Rq15ReportError(f"{candidate.run_name}: scratch run loaded a checkpoint")
        return None
    if expected_checkpoint_sha256 is None or expected_checkpoint_source is None:
        raise Rq15ReportError(f"{candidate.run_name}: checkpoint evidence is absent")
    if not isinstance(value, Mapping):
        raise Rq15ReportError(f"{candidate.run_name}: load report is absent")
    expected = {
        "schema_version": 1,
        "checkpoint_sha256": expected_checkpoint_sha256,
        "source_metadata": source_checkpoint_metadata(expected_checkpoint_source),
        "history_position_count": 128,
        "copied_modules": ["item_embedding", "memory_encoder", "tokenizer"],
        "newly_initialized_modules": [
            "decoder",
            "decoder_query",
            "query_projection",
            "query_slots",
        ],
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise Rq15ReportError(f"{candidate.run_name}: incompatible checkpoint {key}")
    return expected_checkpoint_sha256


def _missing_checkpoint_runs(checkpoints: list[Run]) -> list[str]:
    expected = {candidate.run_name for candidate in source_candidates()}
    return sorted(expected - {run.run_name for run in checkpoints})


def _select_checkpoint(checkpoints: list[Run]) -> Run | None:
    if _missing_checkpoint_runs(checkpoints):
        return None
    if len({run.run_name for run in checkpoints}) != len(checkpoints):
        raise Rq15ReportError("duplicate RQ15 checkpoint artifact identity")
    return _best(checkpoints)


def _validate_roles(
    scratch: Run | None, checkpoints: list[Run], runs: list[Run]
) -> None:
    if scratch is not None and (
        scratch.role != "scratch_candidate_only" or scratch.candidate is not None
    ):
        raise Rq15ReportError("invalid scratch control role")
    expected_sources = {candidate.run_name for candidate in source_candidates()}
    for checkpoint in checkpoints:
        if (
            checkpoint.role != "checkpoint_pretraining"
            or checkpoint.candidate is not None
            or checkpoint.run_name not in expected_sources
        ):
            raise Rq15ReportError("invalid checkpoint role")
    checkpoint = _select_checkpoint(checkpoints)
    for run in runs:
        if run.candidate is None or run.role != run.candidate.training_method:
            raise Rq15ReportError(f"{run.run_name}: treatment role disagrees with manifest")
        if run.run_name != run.candidate.run_name:
            raise Rq15ReportError(f"{run.run_name}: treatment identity is noncanonical")
        expected_weight = (
            run.candidate.auxiliary_ntp_weight
            if run.role == "auxiliary_ntp"
            else 0.0
        )
        if (
            run.embedding_lr != run.candidate.embedding_lr
            or run.deep_lr != run.candidate.deep_lr
            or run.auxiliary_ntp_weight != expected_weight
        ):
            raise Rq15ReportError(
                f"{run.run_name}: tuned fields disagree with candidate identity"
            )
        if run.role == "pretrained_finetune" and (
            checkpoint is None or run.checkpoint_sha256 != checkpoint.checkpoint_sha256
        ):
            raise Rq15ReportError(f"{run.run_name}: checkpoint digest disagrees")
        if run.role != "pretrained_finetune" and run.checkpoint_sha256 is not None:
            raise Rq15ReportError(f"{run.run_name}: scratch run loaded a checkpoint")


def _validate_cross_run_identity(
    scratch: Run | None, checkpoints: list[Run], runs: list[Run]
) -> None:
    available = [run for run in (scratch, *checkpoints, *runs) if run is not None]
    if not available:
        return
    fields = {
        "dataset": "dataset_fingerprint",
        "query cache": "query_cache_fingerprint",
        "evaluator": "evaluator_fingerprint",
    }
    for label, field in fields.items():
        values = {getattr(run, field) for run in available}
        if len(values) != 1:
            raise Rq15ReportError(f"RQ15 {label} identity differs across runs")
    downstream = [run for run in (scratch, *runs) if run is not None]
    if len({run.validation_cache_fingerprint for run in downstream}) > 1:
        raise Rq15ReportError(
            "RQ15 validation cache identity differs across downstream runs"
        )
    if len({run.train_cache_fingerprint for run in downstream}) > 1:
        raise Rq15ReportError("RQ15 train cache identity differs across downstream runs")
    if len({run.scoring_fingerprint for run in downstream}) > 1:
        raise Rq15ReportError("RQ15 scoring identity differs across downstream runs")
    if scratch is not None:
        for run in runs:
            if (
                run.original_users_per_epoch != scratch.original_users_per_epoch
                or run.expanded_examples_per_epoch != scratch.expanded_examples_per_epoch
                or run.candidate_targets_per_epoch != scratch.candidate_targets_per_epoch
                or run.input_tokens_per_epoch != scratch.input_tokens_per_epoch
            ):
                raise Rq15ReportError(f"{run.run_name}: downstream workload identity differs")


def _resolve_lr_surface(
    runs: list[Run],
) -> tuple[Run, list[Rq15Candidate]]:
    if not runs:
        raise Rq15ReportError("cannot resolve an empty RQ15 LR surface")
    method = str(runs[0].role)
    initial = [
        run
        for run in runs
        if run.candidate is None or run.candidate.stage == "initial"
    ]
    expected_coordinates = {
        (candidate.embedding_lr, candidate.deep_lr)
        for candidate in initial_candidates()
        if candidate.training_method == method
    }
    initial_coordinates = {(run.embedding_lr, run.deep_lr) for run in initial}
    if (
        initial_coordinates != expected_coordinates
        or len(initial_coordinates) != len(initial)
    ):
        raise Rq15ReportError(f"{method}: initial LR surface is incomplete")
    boundary = [
        run
        for run in runs
        if run.candidate is not None and run.candidate.stage == "lr_boundary"
    ]
    by_name = {run.run_name: run for run in boundary}
    if len(by_name) != len(boundary):
        raise Rq15ReportError("RQ15 boundary artifact identity is duplicate")
    encountered: list[Run] = []
    while True:
        current = _best([*initial, *encountered])
        expected = _next_frontier_candidates(
            method, current, [*initial, *encountered]
        )
        missing = [
            candidate for candidate in expected if candidate.run_name not in by_name
        ]
        encountered_names = {run.run_name for run in encountered}
        expected_names = {candidate.run_name for candidate in expected}
        if missing:
            foreign = set(by_name) - encountered_names - expected_names
            if foreign:
                raise Rq15ReportError("RQ15 boundary chain is noncontiguous")
            return current, missing
        if not expected:
            if set(by_name) - encountered_names:
                raise Rq15ReportError("RQ15 boundary chain is noncontiguous")
            return current, []
        neighbors = [by_name[candidate.run_name] for candidate in expected]
        if any(
            run.candidate != candidate
            for run, candidate in zip(neighbors, expected)
        ):
            raise Rq15ReportError(
                "RQ15 boundary artifact is not the exact frontier point"
            )
        encountered.extend(neighbors)


def _next_frontier_candidates(
    method: str,
    winner: Run,
    explored: list[Run],
) -> list[Rq15Candidate]:
    candidates: list[Rq15Candidate] = []
    embedding_rates = sorted({run.embedding_lr for run in explored})
    embedding_direction = (
        None
        if method == "pretrained_finetune" and winner.embedding_lr == 0
        else _boundary_direction(winner.embedding_lr, embedding_rates)
    )
    if embedding_direction is not None:
        step, embedding_lr = _next_geometric_rate(
            winner.embedding_lr,
            EMBEDDING_LRS,
            embedding_direction,
            2,
        )
        if (
            method == "pretrained_finetune"
            and embedding_direction == "low"
            and step == PRETRAINED_FROZEN_EMBEDDING_STEP
        ):
            embedding_lr = 0.0
        elif (
            method == "pretrained_finetune"
            and embedding_direction == "low"
            and step > PRETRAINED_FROZEN_EMBEDDING_STEP
        ):
            raise Rq15ReportError(
                "pretrained embedding boundary continued past the frozen endpoint"
            )
        candidates.extend(
            Rq15Candidate(
                training_method=method,
                embedding_lr=embedding_lr,
                deep_lr=deep_lr,
                stage="lr_boundary",
                boundary_axis="embedding",
                boundary_direction=embedding_direction,
                boundary_step=step,
            )
            for deep_lr in _method_deep_lrs(method)
        )

    local_deep_rates = sorted(
        {
            run.deep_lr
            for run in explored
            if math.isclose(run.embedding_lr, winner.embedding_lr)
        }
    )
    deep_direction = _boundary_direction(winner.deep_lr, local_deep_rates)
    if deep_direction is not None:
        step, deep_lr = _next_geometric_rate(
            winner.deep_lr,
            _method_deep_lrs(method),
            deep_direction,
            DEEP_LR_BOUNDARY_RATIOS[method],
        )
        candidates.append(
            Rq15Candidate(
                training_method=method,
                embedding_lr=winner.embedding_lr,
                deep_lr=deep_lr,
                stage="lr_boundary",
                boundary_axis="deep",
                boundary_direction=deep_direction,
                boundary_step=step,
            )
        )
    return candidates


def _method_deep_lrs(method: str) -> tuple[float, ...]:
    if method in {"scratch_candidate_only", "pretrained_finetune"}:
        return CANDIDATE_ONLY_DEEP_LRS
    if method == "auxiliary_ntp":
        return AUXILIARY_DEEP_LRS
    raise Rq15ReportError(f"unknown RQ15 method {method}")


def _boundary_direction(
    rate: float,
    rates: list[float],
) -> Literal["low", "high"] | None:
    if len(rates) < 2:
        raise Rq15ReportError("RQ15 boundary requires at least two tested rates")
    if math.isclose(rate, rates[0]):
        return "low"
    if math.isclose(rate, rates[-1]):
        return "high"
    return None


def _next_geometric_rate(
    current_rate: float,
    initial_rates: tuple[float, ...],
    direction: Literal["low", "high"],
    ratio: int,
) -> tuple[int, float]:
    initial_rate = initial_rates[-1] if direction == "high" else initial_rates[0]
    distance = (
        current_rate / initial_rate
        if direction == "high"
        else initial_rate / current_rate
    )
    completed_steps = round(math.log(distance, ratio))
    if not math.isclose(distance, ratio**completed_steps):
        raise Rq15ReportError("RQ15 boundary is not a method-grid geometric chain")
    next_step = completed_steps + 1
    factor = ratio**next_step if direction == "high" else ratio ** (-next_step)
    return next_step, initial_rate * factor


def _resolve_auxiliary_weights(
    weight_one: Run,
    runs: list[Run],
) -> tuple[Run, list[Rq15Candidate]]:
    assert weight_one.candidate is not None
    expected = [
        make_auxiliary_weight_candidate(weight_one.candidate, weight)
        for weight in (0.1, 0.3)
    ]
    by_name = {run.run_name: run for run in runs}
    missing = [candidate for candidate in expected if candidate.run_name not in by_name]
    if missing:
        return weight_one, missing
    weighted = [by_name[candidate.run_name] for candidate in expected]
    extras = [
        run
        for run in runs
        if run.candidate is not None
        and run.candidate.stage == "auxiliary_weight"
        and run.run_name not in {candidate.run_name for candidate in expected}
    ]
    if extras:
        raise Rq15ReportError("RQ15 auxiliary-weight artifact has the wrong LR anchor")
    return _best([weight_one, *weighted]), []


def _best(runs: Iterable[Run]) -> Run:
    values = list(runs)
    if not values:
        raise Rq15ReportError("cannot select from an empty RQ15 surface")
    return max(
        values,
        key=lambda run: (
            run.validation_recall,
            run.validation_ndcg,
            -run.required_horizon_train_validation_seconds,
        ),
    )


def _validation_best_epoch(curve: Iterable[tuple[int, float, float]]) -> int:
    points = list(curve)
    if not points:
        raise Rq15ReportError("empty validation curve")
    return max(points, key=lambda point: (point[1], -point[0]))[0]


def _selected_validation_point(
    curve: tuple[tuple[int, float, float], ...],
    best_epoch: int,
    context: str,
) -> tuple[int, float, float]:
    if not 1 <= best_epoch <= len(curve):
        raise Rq15ReportError(f"{context}: invalid selected validation epoch")
    selected = curve[best_epoch - 1]
    if selected[0] != best_epoch or selected[1] != max(point[1] for point in curve):
        raise Rq15ReportError(f"{context}: invalid selected validation epoch")
    return selected


def _method_cold_start_costs(
    selected: Mapping[str, Run], checkpoint: Run
) -> dict[str, float]:
    return {
        "scratch_candidate_only": selected[
            "scratch_candidate_only"
        ].time_through_selected_checkpoint_seconds,
        "pretrained_finetune": (
            checkpoint.required_horizon_train_validation_seconds
            + selected["pretrained_finetune"].time_through_selected_checkpoint_seconds
        ),
        "auxiliary_ntp": selected["auxiliary_ntp"].time_through_selected_checkpoint_seconds,
    }


def _select_method(selected: Mapping[str, Run], costs: Mapping[str, float]) -> Run:
    best_recall = max(run.validation_recall for run in selected.values())
    eligible = [run for run in selected.values() if best_recall - run.validation_recall <= 0.003]
    priority = {
        "scratch_candidate_only": 0,
        "pretrained_finetune": 1,
        "auxiliary_ntp": 2,
    }
    return min(eligible, key=lambda run: (costs[str(run.role)], priority[str(run.role)]))


def _acceptance(scratch: Run, pretrained: Run, checkpoint: Run) -> dict[str, object]:
    recall_delta = pretrained.metrics["recall@100"] - scratch.metrics["recall@100"]
    ndcg_delta = pretrained.metrics["ndcg@100"] - scratch.metrics["ndcg@100"]
    cold_start = (
        checkpoint.required_horizon_train_validation_seconds
        + pretrained.time_through_selected_checkpoint_seconds
    )
    quality_non_inferior = recall_delta >= -0.003 and ndcg_delta >= -0.001
    cold_start_faster = cold_start < scratch.time_through_selected_checkpoint_seconds
    return {
        "status": "evaluated",
        "quality_non_inferior": quality_non_inferior,
        "cold_start_faster": cold_start_faster,
        "minimum_acceptance_met": quality_non_inferior and cold_start_faster,
        "main_metrics_improved": recall_delta > 0.003 and ndcg_delta > 0.001,
        "recall@100_delta": recall_delta,
        "ndcg@100_delta": ndcg_delta,
        "scratch_time_to_checkpoint_seconds": scratch.time_through_selected_checkpoint_seconds,
        "pretraining_required_horizon_seconds": checkpoint.required_horizon_train_validation_seconds,
        "fine_tuning_time_to_checkpoint_seconds": pretrained.time_through_selected_checkpoint_seconds,
        "pretrained_cold_start_seconds": cold_start,
    }


def current_implementation_sha256(
    facts: ReportFileFacts | None = None,
) -> dict[str, str]:
    return {str(path): _file_sha256(path, facts) for path in _IMPLEMENTATION_FILES}


def _validate_correctness_evidence(
    evidence: Mapping[str, object] | None,
    run_artifacts: Mapping[str, Mapping[str, str]],
    implementation_hash: object,
    result_binding: Mapping[str, object],
) -> dict[str, object]:
    if evidence is None:
        return {"status": "missing", "required": True}
    checks = evidence.get("checks")
    valid = (
        evidence.get("schema_version") == 1
        and evidence.get("research_question")
        == "RQ15 decoder-decoder training method"
        and evidence.get("dataset_size") == "500m"
        and evidence.get("status") == "passed"
        and isinstance(checks, Mapping)
        and set(checks) == _CORRECTNESS_CHECKS
        and all(
            isinstance(checks[name], Mapping)
            and checks[name].get("passed") is True
            and _is_sha256(checks[name].get("artifact_sha256"))
            and checks[name].get("artifact_sha256")
            == _canonical_sha256(
                {
                    key: value
                    for key, value in checks[name].items()
                    if key != "artifact_sha256"
                }
            )
            for name in _CORRECTNESS_CHECKS
        )
        and evidence.get("run_artifacts") == dict(run_artifacts)
        and evidence.get("implementation_sha256") == implementation_hash
        and evidence.get("result_binding") == dict(result_binding)
    )
    if not valid:
        return {"status": "stale_or_failed", "required": True}
    return {
        "status": "passed",
        "required": True,
        "schema_version": 1,
        "artifact_sha256": _canonical_sha256(evidence),
    }


def _validate_explanation_evidence(
    evidence: Mapping[str, object] | None,
    selected_artifacts: Mapping[str, object],
    acceptance: Mapping[str, object],
    correctness_evidence: Mapping[str, object] | None,
) -> dict[str, object]:
    if evidence is None or correctness_evidence is None:
        return {"status": "missing", "required": True}
    targeted = evidence.get("targeted_evidence")
    valid_targeted = (
        isinstance(targeted, Mapping)
        and bool(targeted)
        and all(
            isinstance(record, Mapping)
            and record.get("passed") is True
            and _is_sha256(record.get("artifact_sha256"))
            for record in targeted.values()
        )
    )
    valid = (
        evidence.get("schema_version") == 1
        and evidence.get("research_question")
        == "RQ15 decoder-decoder training method"
        and evidence.get("dataset_size") == "500m"
        and evidence.get("status") == "passed"
        and evidence.get("selected_artifacts") == dict(selected_artifacts)
        and evidence.get("acceptance_sha256") == _canonical_sha256(acceptance)
        and evidence.get("correctness_audit_sha256")
        == _canonical_sha256(correctness_evidence)
        and isinstance(evidence.get("conclusion"), str)
        and bool(str(evidence["conclusion"]).strip())
        and valid_targeted
    )
    if not valid:
        return {"status": "stale_or_failed", "required": True}
    return {
        "status": "passed",
        "required": True,
        "artifact_sha256": _canonical_sha256(evidence),
        "conclusion": evidence["conclusion"],
    }


def _selected_artifact_binding(
    checkpoint: Run,
    selected: Mapping[str, Run],
) -> dict[str, object]:
    runs = {
        "checkpoint_pretraining": checkpoint,
        **selected,
    }
    return {
        name: {
            "run_name": run.run_name,
            "artifact_sha256": run.artifact_sha256,
        }
        for name, run in runs.items()
    }


def _artifact_audit(
    scratch: Run | None, checkpoints: list[Run], runs: list[Run]
) -> dict[str, object]:
    expected_treatments = {
        candidate.run_name
        for candidate in initial_candidates()
        if not (
            candidate.training_method == "scratch_candidate_only"
            and candidate.embedding_lr == 0.064
            and candidate.deep_lr == 0.0015
            and scratch is not None
        )
    }
    complete_surface = expected_treatments.issubset(
        {run.run_name for run in runs}
    )
    complete_checkpoint_surface = not _missing_checkpoint_runs(checkpoints)
    checkpoint = _select_checkpoint(checkpoints)
    complete_curves = all(
        len(run.validation_curve) == run.stopped_epoch
        for run in (scratch, *checkpoints, *runs)
        if run is not None
    )
    passed = (
        scratch is not None
        and checkpoint is not None
        and complete_surface
        and complete_checkpoint_surface
    )
    return {
        "status": "passed" if passed and complete_curves else "incomplete",
        "checks": {
            "config_and_dataset_identity": bool(scratch and checkpoint),
            "complete_treatment_surface": complete_surface,
            "complete_checkpoint_surface": complete_checkpoint_surface,
            "distinct_cls_only_identity": bool(scratch)
            and all(
                run.candidate is not None
                and run.candidate.training_method == run.role
                for run in runs
            ),
            "objective_and_target_counts": all(
                (run.ntp_targets_per_epoch > 0) is (run.role == "auxiliary_ntp")
                for run in runs
            ),
            "checkpoint_load_identity": bool(checkpoint) and all(
                run.role != "pretrained_finetune"
                or run.checkpoint_sha256 == checkpoint.checkpoint_sha256
                for run in runs
            ),
            "complete_learning_curves": complete_curves,
        },
        "run_artifacts": {
            run.run_name: run.artifact_sha256
            for run in (scratch, *checkpoints, *runs)
            if run is not None
        },
    }


def _reader_markdown(
    selected: Mapping[str, Run],
    checkpoint: Run,
    checkpoints: list[Run],
    grouped: Mapping[str, list[Run]],
) -> str:
    scratch = selected["scratch_candidate_only"]
    quality = [
        "## Candidate-generation quality",
        "",
        "| training method | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in _METHODS:
        run = selected[method]
        cells = [
            _METHOD_LABELS[method],
            *(
                reporting.absolute(run.metrics[metric])
                if method == "scratch_candidate_only"
                else reporting.change_cell(run.metrics[metric], scratch.metrics[metric], metric)
                for metric in _METRICS
            ),
        ]
        quality.append("| " + " | ".join(cells) + " |")
    efficiency = [
        "",
        "## Training efficiency",
        "",
        "| training method | examples/epoch | input tokens/epoch | candidate targets/epoch | NTP targets/epoch | candidate targets/s | total targets/s | best epoch | processed examples | processed candidate targets | processed NTP targets | fine-tune time | pretraining horizon | cold-start time | total tuning wall |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    costs = _method_cold_start_costs(selected, checkpoint)
    tuning_cost = {
        "scratch_candidate_only": sum(
            run.observed_end_to_end_wall_seconds
            for run in grouped["scratch_candidate_only"]
        ),
        "pretrained_finetune": sum(
            run.observed_end_to_end_wall_seconds for run in checkpoints
        )
        + sum(run.observed_end_to_end_wall_seconds for run in grouped["pretrained_finetune"]),
        "auxiliary_ntp": sum(run.observed_end_to_end_wall_seconds for run in grouped["auxiliary_ntp"]),
    }
    for method in _METHODS:
        run = selected[method]
        counts = _processed_counts(method, run, checkpoint)
        efficiency.append(
            "| "
            + " | ".join(
                [
                    _METHOD_LABELS[method],
                    counts["examples_per_epoch"],
                    counts["input_tokens_per_epoch"],
                    f"{run.candidate_targets_per_epoch:,}",
                    f"{run.ntp_targets_per_epoch:,}",
                    _number_or_dash(run.steady_state_candidate_targets_per_second),
                    f"{run.steady_state_total_targets_per_second:,.0f}",
                    str(run.best_epoch),
                    counts["processed_examples"],
                    counts["processed_candidate_targets"],
                    counts["processed_ntp_targets"],
                    _duration(run.time_through_selected_checkpoint_seconds),
                    (_duration(checkpoint.required_horizon_train_validation_seconds) if method == "pretrained_finetune" else "—"),
                    _duration(costs[method]),
                    _duration(tuning_cost[method]),
                ]
            )
            + " |"
        )
    return "\n".join([*quality, *efficiency]).rstrip() + "\n"


def _readme_section(
    reader: str,
    acceptance: Mapping[str, object],
    selected_method: Run | None,
    selected_runs: Mapping[str, Run],
    checkpoint: Run,
    *,
    explanation_status: str,
) -> str:
    tables = reader.replace("## Candidate", "### Candidate").replace(
        "## Training", "### Training"
    )
    selected_label = (
        "none" if selected_method is None else _METHOD_LABELS[str(selected_method.role)]
    )
    expectation = (
        "The probable main-metric improvement was observed."
        if acceptance.get("main_metrics_improved") is True
        else "The probable main-metric improvement was not resolved beyond the empirical bands."
    )
    minimum = (
        "the minimum acceptance criterion is met"
        if acceptance.get("minimum_acceptance_met") is True
        else "the minimum acceptance criterion is not met, and the unexpected result has artifact-bound experimental evidence"
        if explanation_status == "passed"
        else "the minimum acceptance criterion is unresolved"
    )
    scratch = selected_runs["scratch_candidate_only"]
    pretrained = selected_runs["pretrained_finetune"]
    auxiliary = selected_runs["auxiliary_ntp"]
    throughput_delta = abs(
        pretrained.steady_state_candidate_targets_per_second
        - scratch.steady_state_candidate_targets_per_second
    ) / scratch.steady_state_candidate_targets_per_second
    first_pretrained_recall = pretrained.validation_curve[0][1]
    lr_summary = (
        f"scratch {scratch.embedding_lr:g}/{scratch.deep_lr:g}, "
        f"pretrain/fine-tune {pretrained.embedding_lr:g}/{pretrained.deep_lr:g}, "
        f"and auxiliary NTP {auxiliary.embedding_lr:g}/{auxiliary.deep_lr:g}"
    )
    analysis = (
        "Analysis: every embedding LR was paired with the method's three deep LRs, "
        "with deterministic boundary probes where selection reached an edge. "
        f"Selected embedding/deep LRs are {lr_summary}. Pretraining raises full-user "
        f"Recall@100 from {scratch.metrics['recall@100']:.6f} to "
        f"{pretrained.metrics['recall@100']:.6f} and NDCG@100 from "
        f"{scratch.metrics['ndcg@100']:.6f} to {pretrained.metrics['ndcg@100']:.6f}. "
        f"Its fine-tuning epoch 1 validation Recall@100 is {first_pretrained_recall:.4f}, "
        f"already above scratch's selected {scratch.validation_recall:.4f}. Scratch and "
        "pretrained fine-tuning process the same candidate targets and input tokens per epoch; "
        f"their candidate throughput differs by only {throughput_delta:.2%}. The training-cost "
        f"gap instead comes from selecting fine-tuning epoch {pretrained.best_epoch} rather "
        f"than scratch epoch {scratch.best_epoch}, plus the complete "
        f"{checkpoint.stopped_epoch}-epoch NTP source horizon. That horizon costs "
        f"{_duration(checkpoint.required_horizon_train_validation_seconds)}, producing "
        f"a {_duration(float(acceptance['pretrained_cold_start_seconds']))} cold start versus "
        f"{_duration(float(acceptance['scratch_time_to_checkpoint_seconds']))} for scratch."
    )
    decision = (
        "Use NTP pretraining followed by candidate-only fine-tuning when candidate-generation "
        "quality is the objective. Under the approved cold-start accounting it is a "
        "quality/compute tradeoff, not a training-speed optimization."
    )
    return (
        "## RQ15 — For the decoder-decoder model with four distinct CLS tokens and the CLS-only or CLS-plus-history memory selected in RQ14, which training method works best: joint downstream-only training from scratch, first-decoder NTP pretraining followed by joint downstream-only fine-tuning, or joint training from scratch with an auxiliary first-decoder NTP loss? Include pretraining in total training cost.\n\n"
        "Joint scratch uses candidate loss only. Pretrain then fine-tune initializes the first decoder from dense NTP training and jointly fine-tunes both decoders without NTP loss. Auxiliary NTP jointly trains candidate and separately normalized NTP losses from scratch.\n\n"
        f"Acceptance criterion: {ACCEPTANCE_CRITERION}\n\n"
        f"{tables}\n"
        f"{analysis}\n\n"
        f"Conclusion: {minimum}. {expectation} {decision} "
        f"The validation-selected method is {selected_label}. Cold-start accounting includes the complete pretraining horizon.\n"
    )


def _tuning_markdown(
    checkpoints: list[Run],
    checkpoint: Run | None,
    grouped: Mapping[str, list[Run]],
    selected: Mapping[str, Run],
) -> str:
    lines = ["# RQ15 native-500M tuning", ""]
    if checkpoints:
        lines.extend(["## First-decoder NTP checkpoint", "", _tuning_header()])
        for run in sorted(checkpoints, key=lambda item: item.deep_lr):
            lines.append(_tuning_row(run, checkpoint == run))
        lines.append("")
    titles = {
        "scratch_candidate_only": "Joint scratch candidate-only",
        "pretrained_finetune": "Pretrain then fine-tune",
        "auxiliary_ntp": "Simultaneous auxiliary NTP",
    }
    for method in _METHODS:
        lines.extend([f"## {titles[method]}", "", _tuning_header()])
        for run in sorted(grouped[method], key=lambda item: (item.embedding_lr, item.deep_lr)):
            lines.append(_tuning_row(run, selected.get(method) == run))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _tuning_header() -> str:
    return (
        "| embedding LR | deep LR | auxiliary NTP weight | validation recall@100 | validation ndcg@100 | recall@100 | ndcg@100 | best/stopped epoch | horizon wall |\n"
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )


def _tuning_row(run: Run, selected: bool) -> str:
    cells = [
        f"{run.embedding_lr:g}",
        f"{run.deep_lr:g}",
        f"{run.auxiliary_ntp_weight:g}",
        f"{run.validation_recall:.8f}",
        f"{run.validation_ndcg:.8f}",
        f"{run.metrics['recall@100']:.8f}",
        f"{run.metrics['ndcg@100']:.8f}",
        f"{run.best_epoch}/{run.stopped_epoch}",
        _duration(run.required_horizon_train_validation_seconds),
    ]
    if selected:
        cells = [f"**{cell}**" for cell in cells]
    return "| " + " | ".join(cells) + " |"


def _diagnostics_markdown(
    scratch: Run | None,
    checkpoint: Run | None,
    selected: Mapping[str, Run],
    acceptance: Mapping[str, object],
    followups: list[dict[str, object]],
) -> str:
    lines = [
        "# RQ15 diagnostics",
        "",
        "## Acceptance diagnostics",
        "",
        f"Acceptance criterion: {ACCEPTANCE_CRITERION}",
        "",
        "| check | result |",
        "| --- | --- |",
    ]
    for key in ("quality_non_inferior", "cold_start_faster", "main_metrics_improved"):
        value = acceptance.get(key)
        lines.append(f"| {key.replace('_', ' ')} | {_status(value)} |")
    lines.extend(["", "## Supervision and learning curves", "", "| stage | candidate targets/epoch | NTP targets/epoch | best/stopped epoch |", "| --- | ---: | ---: | ---: |"])
    for run in (scratch, checkpoint, *selected.values()):
        if run is None:
            continue
        lines.append(
            f"| {str(run.role).replace('_', ' ')} | {run.candidate_targets_per_epoch:,} | {run.ntp_targets_per_epoch:,} | {run.best_epoch}/{run.stopped_epoch} |"
        )
    if followups:
        lines.extend(["", "Required follow-ups: `" + json.dumps(followups, sort_keys=True) + "`"])
    return "\n".join(lines).rstrip() + "\n"


def _run_record(run: Run) -> dict[str, object]:
    return {
        "training_method": run.role,
        "run_name": run.run_name,
        "embedding_lr": run.embedding_lr,
        "deep_lr": run.deep_lr,
        "auxiliary_ntp_weight": run.auxiliary_ntp_weight,
        "best_epoch": run.best_epoch,
        "stopped_epoch": run.stopped_epoch,
        "validation_metrics": {"recall@100": run.validation_recall, "ndcg@100": run.validation_ndcg},
        "validation_curve": [
            {"epoch": epoch, "recall@100": recall, "ndcg@100": ndcg}
            for epoch, recall, ndcg in run.validation_curve
        ],
        "full_user_metrics": run.metrics,
        "efficiency": {
            "original_users_per_epoch": run.original_users_per_epoch,
            "expanded_examples_per_epoch": run.expanded_examples_per_epoch,
            "candidate_targets_per_epoch": run.candidate_targets_per_epoch,
            "ntp_targets_per_epoch": run.ntp_targets_per_epoch,
            "input_tokens_per_epoch": run.input_tokens_per_epoch,
            "optimizer_steps_per_epoch": run.optimizer_steps_per_epoch,
            "steady_state_candidate_targets_per_second": run.steady_state_candidate_targets_per_second,
            "steady_state_total_targets_per_second": run.steady_state_total_targets_per_second,
            "time_through_selected_checkpoint_seconds": run.time_through_selected_checkpoint_seconds,
            "required_horizon_train_validation_seconds": run.required_horizon_train_validation_seconds,
            "observed_end_to_end_wall_seconds": run.observed_end_to_end_wall_seconds,
        },
        "checkpoint_sha256": run.checkpoint_sha256,
        "artifact_sha256": run.artifact_sha256,
    }


def _candidate_followup(candidate: Rq15Candidate) -> dict[str, object]:
    return candidate_followup_record(candidate)


def _processed_counts(
    method: str, run: Run, checkpoint: Run
) -> dict[str, str]:
    examples = run.expanded_examples_per_epoch
    if examples is None:
        raise Rq15ReportError(f"{run.run_name}: examples-per-epoch evidence is absent")
    processed_examples = examples * run.best_epoch
    processed_ntp = run.ntp_targets_per_epoch * run.best_epoch
    examples_per_epoch = f"{examples:,}"
    input_tokens_per_epoch = f"{run.input_tokens_per_epoch:,}"
    if method == "pretrained_finetune":
        checkpoint_examples = checkpoint.expanded_examples_per_epoch or examples
        processed_examples += checkpoint_examples * checkpoint.stopped_epoch
        processed_ntp += checkpoint.ntp_targets_per_epoch * checkpoint.stopped_epoch
        examples_per_epoch = (
            f"{checkpoint_examples:,} pre + {examples:,} fine"
        )
        input_tokens_per_epoch = (
            f"{checkpoint.input_tokens_per_epoch:,} pre + "
            f"{run.input_tokens_per_epoch:,} fine"
        )
    return {
        "examples_per_epoch": examples_per_epoch,
        "input_tokens_per_epoch": input_tokens_per_epoch,
        "processed_examples": f"{processed_examples:,}",
        "processed_candidate_targets": (
            f"{run.candidate_targets_per_epoch * run.best_epoch:,}"
        ),
        "processed_ntp_targets": f"{processed_ntp:,}",
    }


def _complete_directory(directory: Path) -> bool:
    return directory.is_dir() and all((directory / name).is_file() for name in _REQUIRED_ARTIFACTS)


def _positive_int(mapping: Mapping[str, Any], key: str, context: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise Rq15ReportError(f"{context}: {key} must be a positive integer")
    return value


def _nonnegative_int(mapping: Mapping[str, Any], key: str, context: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise Rq15ReportError(f"{context}: {key} must be a nonnegative integer")
    return value


def _number_or_dash(value: float | None) -> str:
    return "—" if value is None else f"{value:,.0f}"


def _duration(seconds: float) -> str:
    hours, remainder = divmod(round(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}"


def _status(value: object) -> str:
    return "yes" if value is True else "no" if value is False else "pending"


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise Rq15ReportError(f"cannot read {path}") from error
    if not isinstance(value, dict):
        raise Rq15ReportError(f"{path}: expected a JSON object")
    return value


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    return _load_json(path) if path.is_file() else None


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    temporary.replace(path)


def _update_readme(path: Path, section: str) -> None:
    current = path.read_text()
    replacement = f"{_README_START}\n{section.rstrip()}\n{_README_END}"
    if _README_START in current or _README_END in current:
        if current.count(_README_START) != 1 or current.count(_README_END) != 1:
            raise Rq15ReportError("RQ15 README markers are malformed")
        before, rest = current.split(_README_START, 1)
        _, after = rest.split(_README_END, 1)
        updated = before + replacement + after
    else:
        marker = "## Aggregated improvement"
        if marker not in current:
            raise Rq15ReportError("README has no aggregate insertion point")
        updated = current.replace(marker, replacement + "\n\n" + marker, 1)
    _write(path, updated)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--scratchpad", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--rq14-results", type=Path)
    parser.add_argument("--correctness-evidence", type=Path)
    parser.add_argument("--explanation-evidence", type=Path)
    parser.add_argument("--readme", type=Path)
    parser.add_argument("--user-validated", action="store_true")
    args = parser.parse_args()
    bundle = collect_report_bundle(
        args.logs,
        rq14_results=args.rq14_results or args.evidence / "rq14_query_memory_results.json",
        correctness_evidence=(
            args.correctness_evidence
            or args.evidence / "rq15_training_correctness.json"
        ),
        explanation_evidence=(
            args.explanation_evidence
            or args.evidence / "rq15_training_explanation.json"
        ),
        result_claims_user_validated=args.user_validated,
    )
    write_report_bundle(
        bundle,
        args.scratchpad,
        args.evidence,
        readme=args.readme,
    )


if __name__ == "__main__":
    main()
