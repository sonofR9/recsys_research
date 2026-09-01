from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
from typing import Any

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
    DEEP_LRS,
    QueryCandidate,
    candidate_by_run,
    make_boundary_candidate,
    rq14_initial_candidates,
)
from experiments.g1_sasrec_item_ids_likes.analysis.rq14_query_memory_explanation import (
    Rq14ExplanationError,
    validate_unexpected_result_explanation,
)
from experiments.g1_sasrec_item_ids_likes.launchers import verify_artifact
from utils.report_file_facts import ReportFileFacts, report_file_facts


_CONFIG = Path(__file__).parents[1] / "configs/rq13_rq14_query_variant.py"
_CORRECTNESS_EVIDENCE = (
    Path(__file__).parents[1] / "evidence/rq14_query_memory_correctness.json"
)
_EXPLANATION_EVIDENCE = (
    Path(__file__).parents[1] / "evidence/rq14_query_memory_explanation.json"
)
_RQ13_RESULTS = (
    Path(__file__).parents[1] / "evidence/rq13_prefix_expansion_results.json"
)
_RQ13_CORRECTNESS = (
    Path(__file__).parents[1] / "evidence/rq13_prefix_expansion_correctness.json"
)
_TREATMENTS = (
    "shared_cls_only",
    "distinct_cls_only",
    "shared_history",
    "distinct_history",
)
_LABELS = {
    "shared_cls_only": ("shared CLS", "four CLS states"),
    "distinct_cls_only": ("distinct CLS_0..3", "four CLS states"),
    "shared_history": ("shared CLS", "history + four CLS states"),
    "distinct_history": ("distinct CLS_0..3", "history + four CLS states"),
}
_METRICS = ("recall@100", "ndcg@100", "recall@10", "ndcg@10", "coverage@100")
_COUNT_FIELDS = (
    "original_users_per_epoch",
    "expanded_examples_per_epoch",
    "candidate_targets_per_epoch",
    "ntp_targets_per_epoch",
    "input_tokens_per_epoch",
)
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
_AUDIT_CHECKS = {
    "artifact_and_recipe_integrity",
    "query_slot_identity_and_order",
    "memory_content_and_lengths",
    "target_exclusion_and_candidate_only_loss",
    "gradient_flow_to_every_slot_and_history",
    "learning_curves_and_lr_boundaries",
}
_IMPLEMENTATION_FILES = (
    Path("dcn/config/query_retrieval.py"),
    Path("dcn/models/cross_attention_retrieval.py"),
    Path("dcn/models/history_tokens.py"),
    Path("dcn/nn/transformer.py"),
    Path("experiments/g1_sasrec_item_ids_likes/analysis/rq13_rq14_query_candidates.py"),
    Path("experiments/g1_sasrec_item_ids_likes/analysis/rq14_query_memory_audit.py"),
    Path(
        "experiments/g1_sasrec_item_ids_likes/analysis/"
        "rq14_query_memory_explanation.py"
    ),
    Path("experiments/g1_sasrec_item_ids_likes/analysis/rq14_query_memory_report.py"),
    Path("experiments/g1_sasrec_item_ids_likes/configs/rq13_rq14_query_variant.py"),
    Path("experiments/g1_sasrec_item_ids_likes/configs/variant.py"),
)


class Rq14ReportError(RuntimeError):
    pass


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
    train_cache: str
    validation_cache: str
    query_cache: str
    evaluator_fingerprint: str
    scoring_fingerprint: str
    artifact_sha256: dict[str, str]


@dataclass(frozen=True)
class Rq14ReportBundle:
    reader_markdown: str | None
    tuning_markdown: str
    diagnostics_markdown: str | None
    evidence: dict[str, object]


def collect_report_bundle(
    logs: Path,
    *,
    correctness_evidence: Path = _CORRECTNESS_EVIDENCE,
    explanation_evidence: Path = _EXPLANATION_EVIDENCE,
    rq13_results: Path = _RQ13_RESULTS,
    rq13_correctness: Path = _RQ13_CORRECTNESS,
) -> Rq14ReportBundle:
    facts = report_file_facts(logs.parent)
    runs = []
    for directory in sorted(logs.glob("g1_rq14_*_500m")):
        if not directory.is_dir():
            continue
        try:
            candidate = candidate_by_run(directory.name)
        except ValueError:
            continue
        if candidate.study != "rq14":
            continue
        required = tuple(
            directory / name
            for name in ("training_metadata.json", "final_metrics.json", "sweep.log")
        )
        if not all(path.is_file() for path in required):
            continue
        if not verify_artifact.verify_config(
            directory, _CONFIG, [f"G1_QUERY_RUN={candidate.run_name}"]
        ):
            raise Rq14ReportError(f"{directory.name}: recipe-incompatible artifact")
        runs.append(_load_run(directory, candidate, facts))
    audit = _load_optional_json(correctness_evidence)
    return build_report_bundle(
        runs,
        correctness_audit=audit,
        unexpected_explanation=_load_optional_json(explanation_evidence),
        rq13_results=_load_optional_json(rq13_results),
        rq13_correctness_audit=_load_optional_json(rq13_correctness),
    )


def build_report_bundle(
    runs: Iterable[Run],
    *,
    correctness_audit: Mapping[str, object] | None = None,
    unexpected_explanation: Mapping[str, object] | None = None,
    rq13_results: Mapping[str, object] | None = None,
    rq13_correctness_audit: Mapping[str, object] | None = None,
    current_implementation_hash: object | None = None,
) -> Rq14ReportBundle:
    run_list = list(runs)
    by_name = {run.candidate.run_name: run for run in run_list}
    if len(by_name) != len(run_list):
        raise Rq14ReportError("duplicate RQ14 artifact identity")
    if any(run.candidate.study != "rq14" for run in run_list):
        raise Rq14ReportError("non-RQ14 run entered the RQ14 report")
    _require_workload_consistency(run_list)

    missing_initial = [
        candidate.run_name
        for candidate in rq14_initial_candidates()
        if candidate.run_name not in by_name
    ]
    selected_surface: dict[str, Run] = {}
    required_boundary: list[str] = []
    treatment_runs = {
        treatment: [
            run for run in run_list if run.candidate.treatment == treatment
        ]
        for treatment in _TREATMENTS
    }
    if not missing_initial:
        for treatment in _TREATMENTS:
            available = treatment_runs[treatment]
            winner, followup = _resolve_treatment_surface(available)
            selected_surface[treatment] = winner
            if followup is not None and followup.run_name not in by_name:
                required_boundary.append(followup.run_name)
    elif any(run.candidate.stage == "lr_boundary" for run in run_list):
        raise Rq14ReportError("RQ14 boundary artifact precedes its complete initial grid")

    required_followups = [*missing_initial, *required_boundary]
    selected = selected_surface if not required_followups else {}
    total_cost = {
        treatment: sum(
            run.observed_end_to_end_wall_seconds for run in treatment_runs[treatment]
        )
        for treatment in _TREATMENTS
    }
    effects = _paired_effects(selected)
    unexpected = {
        name: effect
        for name, effect in effects.items()
        if effect["full_recall_delta"] <= 0.003
    }
    implementation_hash = (
        current_implementation_sha256()
        if current_implementation_hash is None
        else current_implementation_hash
    )
    audit_record = _validate_correctness_audit(
        correctness_audit,
        {run.candidate.run_name: run.artifact_sha256 for run in run_list},
        implementation_hash,
    )
    selected_records = {
        treatment: _selected_record(run) for treatment, run in selected.items()
    }
    treatment_records = {
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
        for treatment in _TREATMENTS
    }
    overall_rule = (
        "within 0.003 validation Recall@100 choose CLS-only memory, then shared tokens"
    )
    selected_method = _select_overall(selected)
    selected_method_record = (
        None if selected_method is None else _selected_record(selected_method)
    )
    explanation_record: dict[str, object] = {
        "required": bool(unexpected),
        "status": "not_required" if not unexpected else "missing",
    }
    if unexpected and audit_record["status"] == "passed":
        explanation_context = {
            "research_question": "RQ14 decoder-decoder query memory",
            "dataset_size": "500m",
            "claims_status": "unexpected_result_requires_explanation",
            "required_followups": [],
            "paired_effects": effects,
            "unexpected_effects": unexpected,
            "correctness_audit": audit_record,
            "selected": selected_records,
            "selected_method": selected_method_record,
            "overall_rule": overall_rule,
            "treatments": treatment_records,
        }
        if (
            unexpected_explanation is not None
            and correctness_audit is not None
            and rq13_results is not None
            and rq13_correctness_audit is not None
        ):
            try:
                explanation_record = {
                    "required": True,
                    **validate_unexpected_result_explanation(
                        unexpected_explanation,
                        explanation_context,
                        correctness_audit,
                        rq13_results,
                        rq13_correctness_audit,
                    ),
                }
            except Rq14ExplanationError as error:
                explanation_record = {
                    "required": True,
                    "status": "stale_or_invalid",
                    "error": str(error),
                }
    required_diagnostics: list[str] = []
    if missing_initial:
        claims_status = "pending_artifacts"
    elif required_boundary:
        claims_status = "pending_boundary"
    elif audit_record["status"] != "passed":
        claims_status = "correctness_audit_required"
        required_diagnostics.append(
            "generate a current RQ14 correctness audit before interpreting paired effects"
        )
    elif unexpected and explanation_record["status"] != "passed":
        claims_status = "unexpected_result_requires_explanation"
        required_diagnostics.append(
            "explain each unexpected token-identity or memory effect with experimental evidence"
        )
    else:
        claims_status = "ready_for_user_validation"

    rq15_memory = _select_rq15_memory(selected)
    evidence: dict[str, object] = {
        "research_question": "RQ14 decoder-decoder query memory",
        "dataset_size": "500m",
        "claims_status": claims_status,
        "result_claims_user_validated": False,
        "selection_rule": (
            "within each treatment: validation Recall@100, then same-epoch "
            "NDCG@100, then lower logged horizon training time"
        ),
        "overall_rule": overall_rule,
        "rq15_rule": (
            "select the higher validation-Recall distinct-token memory; within 0.003 choose CLS-only"
        ),
        "boundary_rule": (
            "extend a winning outer deep LR geometrically by a factor of two until interior"
        ),
        "missing_initial_artifacts": missing_initial,
        "required_boundary_followups": required_boundary,
        "required_followups": required_followups,
        "required_diagnostics": required_diagnostics,
        "unexpected_effects": unexpected,
        "paired_effects": effects,
        "correctness_audit": audit_record,
        "unexpected_result_explanation": explanation_record,
        "surface_winners": {
            treatment: _selected_record(run)
            for treatment, run in selected_surface.items()
        },
        "selected": selected_records,
        "selected_method": selected_method_record,
        "rq15_distinct_memory": (
            None if rq15_memory is None else _selected_record(rq15_memory)
        ),
        "timing_definition": {
            "steady_state_targets_per_second": (
                "candidate targets divided by logged train time over epochs 2-20"
            ),
            "time_through_selected_checkpoint_seconds": (
                "logged train, validation inference, and validation save time through the selected checkpoint"
            ),
            "total_required_training_wall_seconds": (
                "Prepared-stage to Final-metrics wall time over all required tuning and boundary runs"
            ),
        },
        "treatments": treatment_records,
    }
    reader_selected = selected if selected else selected_surface
    reader_ready = claims_status == "ready_for_user_validation"
    reader = (
        _reader_markdown(reader_selected, total_cost, selected_method)
        if reader_ready and len(reader_selected) == len(_TREATMENTS)
        else None
    )
    diagnostics = (
        _diagnostics_markdown(
            effects,
            unexpected,
            (
                unexpected_explanation
                if explanation_record["status"] == "passed"
                else None
            ),
        )
        if effects
        else None
    )
    return Rq14ReportBundle(
        reader_markdown=reader,
        tuning_markdown=_tuning_markdown(treatment_runs, selected_surface),
        diagnostics_markdown=diagnostics,
        evidence=evidence,
    )


def validate_training_metadata(
    metadata: dict[str, Any], candidate: QueryCandidate
) -> tuple[dict[str, int], dict[str, Any]]:
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
            raise Rq14ReportError(
                f"{candidate.run_name}: {key}={metadata.get(key)!r}, expected {value!r}"
            )
    invariants = metadata.get("transfer_invariants")
    if not isinstance(invariants, dict):
        raise Rq14ReportError(f"{candidate.run_name}: transfer_invariants is absent")
    architecture = {
        "query_architecture": "decoder_decoder",
        "prefix_length_rule": "truncated",
        "prefix_cap": 1,
        "query_slots_shared": candidate.treatment.startswith("shared_"),
        "include_history_memory": candidate.treatment.endswith("_history"),
        "num_query_slots": 4,
    }
    for key, value in architecture.items():
        if metadata.get(key) != value or invariants.get(key) != value:
            raise Rq14ReportError(f"{candidate.run_name}: incompatible {key} metadata")
    counts = {"best_epoch": _positive_int(metadata, "best_epoch", candidate.run_name)}
    if counts["best_epoch"] > 20:
        raise Rq14ReportError(f"{candidate.run_name}: best epoch exceeds horizon")
    for key in _COUNT_FIELDS:
        top = metadata.get(key)
        valid = (
            isinstance(top, int)
            and not isinstance(top, bool)
            and (top == 0 if key == "ntp_targets_per_epoch" else top > 0)
        )
        if not valid or invariants.get(key) != top:
            raise Rq14ReportError(
                f"{candidate.run_name}: invalid or inconsistent count metadata {key}"
            )
        counts[key] = top
    if (
        counts["original_users_per_epoch"] != counts["expanded_examples_per_epoch"]
        or counts["candidate_targets_per_epoch"]
        != counts["expanded_examples_per_epoch"]
    ):
        raise Rq14ReportError(
            f"{candidate.run_name}: decoder-decoder needs one candidate target and example per user"
        )
    if (
        metadata.get("targets_per_epoch") != counts["candidate_targets_per_epoch"]
        or metadata.get("tokens_per_epoch") != counts["input_tokens_per_epoch"]
    ):
        raise Rq14ReportError(
            f"{candidate.run_name}: generic and architecture count metadata disagree"
        )
    return counts, invariants


def write_report_bundle(
    bundle: Rq14ReportBundle, scratchpad: Path, evidence: Path
) -> dict[str, Path]:
    paths = {
        "tuning": scratchpad / "rq14_query_memory_tuning_500m.md",
        "evidence": evidence / "rq14_query_memory_results.json",
    }
    _write(paths["tuning"], bundle.tuning_markdown)
    _write(
        paths["evidence"],
        json.dumps(bundle.evidence, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    if bundle.reader_markdown is not None:
        paths["reader"] = scratchpad / "rq14_query_memory_reader_500m.md"
        _write(paths["reader"], bundle.reader_markdown)
    else:
        (scratchpad / "rq14_query_memory_reader_500m.md").unlink(missing_ok=True)
    if bundle.diagnostics_markdown is not None:
        paths["diagnostics"] = scratchpad / "rq14_query_memory_diagnostics_500m.md"
        _write(paths["diagnostics"], bundle.diagnostics_markdown)
    else:
        (scratchpad / "rq14_query_memory_diagnostics_500m.md").unlink(
            missing_ok=True
        )
    return paths


def current_implementation_sha256() -> dict[str, str]:
    return {str(path): _file_sha256(path) for path in _IMPLEMENTATION_FILES}


def _load_run(
    directory: Path,
    candidate: QueryCandidate,
    facts: ReportFileFacts | None = None,
) -> Run:
    metadata = _load_json(directory / "training_metadata.json")
    metrics = _load_json(directory / "final_metrics.json")
    counts, invariants = validate_training_metadata(metadata, candidate)
    timings, prepared, final = _timings(directory)
    validation_curve = _validation_curve(directory)
    selected_epoch, recall, ndcg = validation_curve[counts["best_epoch"] - 1]
    if selected_epoch != counts["best_epoch"]:
        raise Rq14ReportError(
            f"{candidate.run_name}: selected epoch is absent from validation curve"
        )
    expected_epoch = _validation_best_epoch(validation_curve)
    if selected_epoch != expected_epoch:
        raise Rq14ReportError(
            f"{candidate.run_name}: best_epoch is not the validation-curve winner"
        )
    identity = _dataset_identity(directory, prepared, facts=facts)
    caches = identity["caches"]
    cache_content = identity["cache_content"]
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
        raise Rq14ReportError(f"{candidate.run_name}: no steady-state timing evidence")
    best_timings = timings[: counts["best_epoch"]]
    return Run(
        candidate=candidate,
        best_epoch=counts["best_epoch"],
        stopped_epoch=20,
        validation_recall=recall,
        validation_ndcg=ndcg,
        validation_curve=validation_curve,
        metrics=_metric_mapping(metrics, candidate.run_name),
        original_users_per_epoch=counts["original_users_per_epoch"],
        expanded_examples_per_epoch=counts["expanded_examples_per_epoch"],
        candidate_targets_per_epoch=counts["candidate_targets_per_epoch"],
        ntp_targets_per_epoch=counts["ntp_targets_per_epoch"],
        input_tokens_per_epoch=counts["input_tokens_per_epoch"],
        optimizer_steps_per_epoch=_positive_int(
            metadata, "optimizer_steps_per_epoch", candidate.run_name
        ),
        steady_state_targets_per_second=(
            counts["candidate_targets_per_epoch"] * len(steady) / steady_train
        ),
        time_through_selected_checkpoint_seconds=sum(
            sum(item[1:]) for item in best_timings
        ),
        required_horizon_train_validation_seconds=sum(
            sum(item[1:]) for item in timings
        ),
        observed_end_to_end_wall_seconds=(final - prepared).total_seconds(),
        train_cache=_canonical_sha256(cache_content[caches["train"]]),
        validation_cache=_canonical_sha256(cache_content[caches["val"]]),
        query_cache=_canonical_sha256(cache_content[caches["true_metric_query"]]),
        evaluator_fingerprint=evaluator,
        scoring_fingerprint=scoring,
        artifact_sha256={
            name: _file_sha256(directory / name, facts)
            for name in ("training_metadata.json", "final_metrics.json", "sweep.log")
        },
    )


def _require_workload_consistency(runs: list[Run]) -> None:
    if not runs:
        return
    fields = (
        "original_users_per_epoch",
        "expanded_examples_per_epoch",
        "candidate_targets_per_epoch",
        "ntp_targets_per_epoch",
        "input_tokens_per_epoch",
        "optimizer_steps_per_epoch",
        "train_cache",
        "validation_cache",
        "query_cache",
        "evaluator_fingerprint",
        "scoring_fingerprint",
    )
    reference = runs[0]
    for run in runs[1:]:
        differing = [
            name for name in fields if getattr(run, name) != getattr(reference, name)
        ]
        if differing:
            raise Rq14ReportError(
                f"{run.candidate.run_name}: RQ14 workload differs in {', '.join(differing)}"
            )


def _resolve_treatment_surface(
    runs: list[Run],
) -> tuple[Run, QueryCandidate | None]:
    initial = [run for run in runs if run.candidate.stage == "initial"]
    if {run.candidate.deep_lr for run in initial} != set(DEEP_LRS):
        raise Rq14ReportError("RQ14 treatment lacks its exact initial LR grid")
    boundary = [run for run in runs if run.candidate.stage == "lr_boundary"]
    initial_winner = _best(initial)
    if initial_winner.candidate.deep_lr == min(DEEP_LRS):
        direction = "low"
    elif initial_winner.candidate.deep_lr == max(DEEP_LRS):
        direction = "high"
    else:
        if boundary:
            raise Rq14ReportError(
                "RQ14 boundary artifact exists after an interior initial winner"
            )
        return initial_winner, None
    if any(run.candidate.boundary_direction != direction for run in boundary):
        raise Rq14ReportError("RQ14 boundary direction contradicts the initial winner")
    by_step = {run.candidate.boundary_step: run for run in boundary}
    if len(by_step) != len(boundary) or set(by_step) != set(
        range(1, len(boundary) + 1)
    ):
        raise Rq14ReportError("RQ14 boundary chain is duplicate or noncontiguous")

    surface = list(initial)
    for step in range(1, len(boundary) + 1):
        expected = make_boundary_candidate(initial_winner.candidate, direction, step)
        current = by_step[step]
        if current.candidate != expected:
            raise Rq14ReportError("RQ14 boundary artifact is not the exact next point")
        if step > 1 and _best(surface).candidate != by_step[step - 1].candidate:
            raise Rq14ReportError(
                "RQ14 boundary artifact follows an already resolved surface"
            )
        surface.append(current)
    winner = _best(surface)
    if boundary and winner.candidate != by_step[len(boundary)].candidate:
        return winner, None
    return winner, make_boundary_candidate(
        initial_winner.candidate, direction, len(boundary) + 1
    )


def _best(runs: Iterable[Run]) -> Run:
    values = list(runs)
    if not values:
        raise Rq14ReportError("cannot select from an empty RQ14 surface")
    return max(
        values,
        key=lambda run: (
            run.validation_recall,
            run.validation_ndcg,
            -run.required_horizon_train_validation_seconds,
        ),
    )


def _validation_best_epoch(
    curve: Iterable[tuple[int, float, float]],
) -> int:
    points = list(curve)
    if not points:
        raise Rq14ReportError("cannot select an epoch from an empty curve")
    return max(points, key=lambda point: (point[1], -point[0]))[0]


def _select_overall(selected: Mapping[str, Run]) -> Run | None:
    if len(selected) != len(_TREATMENTS):
        return None
    best_recall = max(run.validation_recall for run in selected.values())
    eligible = [
        run for run in selected.values() if best_recall - run.validation_recall <= 0.003
    ]
    return min(
        eligible,
        key=lambda run: (
            run.candidate.treatment.endswith("_history"),
            run.candidate.treatment.startswith("distinct_"),
            -run.validation_recall,
            -run.validation_ndcg,
        ),
    )


def _select_rq15_memory(selected: Mapping[str, Run]) -> Run | None:
    names = ("distinct_cls_only", "distinct_history")
    if any(name not in selected for name in names):
        return None
    cls_only, history = (selected[name] for name in names)
    return (
        cls_only
        if history.validation_recall - cls_only.validation_recall <= 0.003
        else history
    )


def _paired_effects(selected: Mapping[str, Run]) -> dict[str, dict[str, object]]:
    pairs = {
        "distinct_minus_shared_cls_only": ("shared_cls_only", "distinct_cls_only"),
        "distinct_minus_shared_history": ("shared_history", "distinct_history"),
        "history_minus_cls_only_shared": ("shared_cls_only", "shared_history"),
        "history_minus_cls_only_distinct": ("distinct_cls_only", "distinct_history"),
    }
    if len(selected) != len(_TREATMENTS):
        return {}
    return {
        name: {
            "control": control,
            "treatment": treatment,
            "validation_recall_delta": (
                selected[treatment].validation_recall
                - selected[control].validation_recall
            ),
            "full_recall_delta": (
                selected[treatment].metrics["recall@100"]
                - selected[control].metrics["recall@100"]
            ),
            "expected_improvement_beyond_band": (
                selected[treatment].metrics["recall@100"]
                - selected[control].metrics["recall@100"]
                > 0.003
            ),
        }
        for name, (control, treatment) in pairs.items()
    }


def _validate_correctness_audit(
    audit: Mapping[str, object] | None,
    expected_artifacts: Mapping[str, Mapping[str, str]],
    implementation_hash: object,
) -> dict[str, object]:
    if audit is None:
        return {"required": True, "status": "missing"}
    checks = audit.get("checks")
    valid = (
        audit.get("schema_version") == 1
        and audit.get("research_question") == "RQ14 decoder-decoder query memory"
        and audit.get("dataset_size") == "500m"
        and audit.get("status") == "passed"
        and isinstance(checks, Mapping)
        and set(checks) == _AUDIT_CHECKS
        and all(
            isinstance(checks[name], Mapping) and checks[name].get("passed") is True
            for name in _AUDIT_CHECKS
        )
        and audit.get("run_artifacts") == dict(expected_artifacts)
        and audit.get("implementation_sha256") == implementation_hash
    )
    if not valid:
        return {"required": True, "status": "stale_or_failed"}
    return {
        "required": True,
        "status": "passed",
        "schema_version": 1,
        "artifact_sha256": _canonical_sha256(audit),
    }


def _reader_markdown(
    selected: Mapping[str, Run],
    total_cost: Mapping[str, float],
    selected_method: Run | None,
) -> str:
    reference = selected["shared_cls_only"]
    quality = [
        "## Candidate-generation quality",
        "",
        "| query tokens | cross-attention memory | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for treatment in _TREATMENTS:
        run = selected[treatment]
        token, memory = _LABELS[treatment]
        cells = [
            token,
            memory,
            *(
                reporting.absolute(run.metrics[metric])
                if treatment == "shared_cls_only"
                else reporting.change_cell(
                    run.metrics[metric], reference.metrics[metric], metric
                )
                for metric in _METRICS
            ),
        ]
        quality.append(
            _reader_row(
                cells, selected=selected_method is not None and run == selected_method
            )
        )
    efficiency = [
        "",
        "## Training efficiency",
        "",
        "| query tokens | cross-attention memory | examples/epoch | candidate targets/epoch | NTP targets/epoch | input tokens/epoch | targets/s | best epoch | processed examples | processed candidate targets | time to checkpoint | total tuning wall |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for treatment in _TREATMENTS:
        run = selected[treatment]
        token, memory = _LABELS[treatment]
        cells = [
            token,
            memory,
            f"{run.expanded_examples_per_epoch:,}",
            f"{run.candidate_targets_per_epoch:,}",
            f"{run.ntp_targets_per_epoch:,}",
            f"{run.input_tokens_per_epoch:,}",
            f"{run.steady_state_targets_per_second:,.0f}",
            str(run.best_epoch),
            f"{run.expanded_examples_per_epoch * run.best_epoch:,}",
            f"{run.candidate_targets_per_epoch * run.best_epoch:,}",
            _duration(run.time_through_selected_checkpoint_seconds),
            _duration(total_cost[treatment]),
        ]
        efficiency.append(
            _reader_row(
                cells, selected=selected_method is not None and run == selected_method
            )
        )
    return "\n".join([*quality, *efficiency]).rstrip() + "\n"


def _tuning_markdown(
    treatment_runs: Mapping[str, list[Run]], selected: Mapping[str, Run]
) -> str:
    lines = ["# RQ14 native-500M tuning", ""]
    for treatment in _TREATMENTS:
        token, memory = _LABELS[treatment]
        lines.extend(
            [
                f"## {token}, {memory}",
                "",
                "| deep LR | validation recall@100 | validation ndcg@100 | recall@100 | ndcg@100 | best/stopped epoch | horizon wall |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for run in sorted(
            treatment_runs[treatment], key=lambda item: item.candidate.deep_lr
        ):
            cells = [
                f"{run.candidate.deep_lr:g}",
                f"{run.validation_recall:.8f}",
                f"{run.validation_ndcg:.8f}",
                f"{run.metrics['recall@100']:.8f}",
                f"{run.metrics['ndcg@100']:.8f}",
                f"{run.best_epoch}/{run.stopped_epoch}",
                _duration(run.required_horizon_train_validation_seconds),
            ]
            if selected.get(treatment) == run:
                cells = [f"**{cell}**" for cell in cells]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _diagnostics_markdown(
    effects: Mapping[str, Mapping[str, object]],
    unexpected: Mapping[str, Mapping[str, object]],
    explanation: Mapping[str, object] | None,
) -> str:
    lines = [
        "# RQ14 paired-effect diagnostics",
        "",
        "| effect | validation recall delta | full recall delta | exceeds +0.003 |",
        "| --- | ---: | ---: | --- |",
    ]
    for name, effect in effects.items():
        lines.append(
            "| "
            + " | ".join(
                [
                    name.replace("_", " "),
                    f"{float(effect['validation_recall_delta']):+.8f}",
                    f"{float(effect['full_recall_delta']):+.8f}",
                    "yes" if effect["expected_improvement_beyond_band"] else "no",
                ]
            )
            + " |"
        )
    if unexpected and explanation is None:
        lines.extend(
            [
                "",
                "Unexpected effects require a current correctness audit and an experimental explanation before claims are accepted.",
            ]
        )
    if explanation is not None and explanation.get("status") == "passed":
        findings = explanation.get("findings")
        axis = findings.get("axis_effects") if isinstance(findings, Mapping) else None
        supervision = (
            findings.get("supervision_density")
            if isinstance(findings, Mapping)
            else None
        )
        if isinstance(axis, Mapping) and isinstance(supervision, Mapping):
            lines.extend(
                [
                    "",
                    "## Bound explanation",
                    "",
                    "The correctness and unexpected-result explanation gates passed for the current artifacts.",
                    "",
                    str(axis["distinct_token_identity"]["claim"]),
                    "",
                    str(axis["history_memory"]["claim"]),
                    "",
                    str(supervision["claim"]),
                ]
            )
    return "\n".join(lines) + "\n"


def _selected_record(run: Run) -> dict[str, object]:
    return {
        "treatment": run.candidate.treatment,
        "run_name": run.candidate.run_name,
        "deep_lr": run.candidate.deep_lr,
        "best_epoch": run.best_epoch,
        "validation_metrics": {
            "recall@100": run.validation_recall,
            "ndcg@100": run.validation_ndcg,
        },
        "full_user_metrics": run.metrics,
    }


def _run_record(run: Run) -> dict[str, object]:
    return {
        **_selected_record(run),
        "stage": run.candidate.stage,
        "boundary_direction": run.candidate.boundary_direction,
        "boundary_step": run.candidate.boundary_step,
        "stopped_epoch": run.stopped_epoch,
        "validation_curve": [
            {"epoch": epoch, "recall@100": recall, "ndcg@100": ndcg}
            for epoch, recall, ndcg in run.validation_curve
        ],
        "efficiency": {
            "original_users_per_epoch": run.original_users_per_epoch,
            "expanded_examples_per_epoch": run.expanded_examples_per_epoch,
            "candidate_targets_per_epoch": run.candidate_targets_per_epoch,
            "ntp_targets_per_epoch": run.ntp_targets_per_epoch,
            "input_tokens_per_epoch": run.input_tokens_per_epoch,
            "optimizer_steps_per_epoch": run.optimizer_steps_per_epoch,
            "steady_state_targets_per_second": run.steady_state_targets_per_second,
            "time_through_selected_checkpoint_seconds": run.time_through_selected_checkpoint_seconds,
            "required_horizon_train_validation_seconds": run.required_horizon_train_validation_seconds,
            "observed_end_to_end_wall_seconds": run.observed_end_to_end_wall_seconds,
        },
        "artifact_sha256": run.artifact_sha256,
    }


def _reader_row(cells: list[str], *, selected: bool) -> str:
    if selected:
        cells = [f"**{cell}**" for cell in cells]
    return "| " + " | ".join(cells) + " |"


def _duration(seconds: float) -> str:
    hours, remainder = divmod(round(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}"


def _positive_int(mapping: Mapping[str, Any], key: str, context: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise Rq14ReportError(f"{context}: {key} must be a positive integer")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise Rq14ReportError(f"cannot read {path}") from error
    if not isinstance(value, dict):
        raise Rq14ReportError(f"{path}: expected a JSON object")
    return value


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return _load_json(path)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--scratchpad", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--correctness-evidence", type=Path)
    parser.add_argument("--explanation-evidence", type=Path)
    parser.add_argument("--rq13-results", type=Path)
    parser.add_argument("--rq13-correctness", type=Path)
    args = parser.parse_args()
    bundle = collect_report_bundle(
        args.logs,
        correctness_evidence=(
            args.correctness_evidence
            if args.correctness_evidence is not None
            else args.evidence / "rq14_query_memory_correctness.json"
        ),
        explanation_evidence=(
            args.explanation_evidence
            if args.explanation_evidence is not None
            else args.evidence / "rq14_query_memory_explanation.json"
        ),
        rq13_results=(
            args.rq13_results
            if args.rq13_results is not None
            else args.evidence / "rq13_prefix_expansion_results.json"
        ),
        rq13_correctness=(
            args.rq13_correctness
            if args.rq13_correctness is not None
            else args.evidence / "rq13_prefix_expansion_correctness.json"
        ),
    )
    write_report_bundle(bundle, args.scratchpad, args.evidence)


if __name__ == "__main__":
    main()
