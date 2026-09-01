from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path
import tempfile
from typing import Any

from experiments.g1_sasrec_item_ids_likes.analysis import reporting
from experiments.g1_sasrec_item_ids_likes.analysis.rq15_training_audit import (
    Rq15AuditError,
    _results_binding,
    validate_correctness_audit,
)
from experiments.g1_sasrec_item_ids_likes.analysis.rq13_prefix_expansion_report import (
    _timings,
    _validation_curve,
)


_RESEARCH_QUESTION = "RQ15 decoder-decoder training method"
_METHODS = {
    "scratch_candidate_only",
    "pretrained_finetune",
    "auxiliary_ntp",
}
_DEFAULT_LOGS = Path("generated/logs")
_REPORTING_IMPLEMENTATION = Path(
    "experiments/g1_sasrec_item_ids_likes/analysis/reporting.py"
)
_ARCHITECTURE_PROTOCOL = Path(
    "experiments/g1_sasrec_item_ids_likes/protocol/rq12_rq15_architecture_plan.md"
)
_BASELINE_SPREAD_EVIDENCE = Path(
    "experiments/g1_sasrec_item_ids_likes/scratchpad/baseline_spread_500m.json"
)
_ACCEPTANCE_CRITERION = (
    "Adding a pretraining stage should at minimum decrease training time without "
    "losing quality, and will most probably improve the main metrics."
)
_QUALITY_BANDS = {
    "recall@100": reporting.difference_threshold("recall@100"),
    "ndcg@100": reporting.difference_threshold("ndcg@100"),
}
_RECALL_BAND = _QUALITY_BANDS["recall@100"]
_NDCG_BAND = _QUALITY_BANDS["ndcg@100"]


class Rq15ExplanationError(RuntimeError):
    pass


def build_training_budget_explanation(
    results: Mapping[str, object],
    correctness: Mapping[str, object],
    *,
    logs: Path = _DEFAULT_LOGS,
) -> dict[str, object]:
    _validate_resolved_results(results)
    acceptance_protocol = _acceptance_protocol()
    _validate_correctness(results, correctness)
    selected = _selected_records(results)
    _validate_selected_raw_artifacts(selected, logs)
    selected_artifacts = _selected_artifact_binding(selected)
    scratch = selected["scratch_candidate_only"]
    pretrained = selected["pretrained_finetune"]
    checkpoint = selected["checkpoint_pretraining"]

    scratch_recall = _metric(scratch, "recall@100")
    pretrained_recall = _metric(pretrained, "recall@100")
    scratch_ndcg = _metric(scratch, "ndcg@100")
    pretrained_ndcg = _metric(pretrained, "ndcg@100")
    recall_delta = pretrained_recall - scratch_recall
    ndcg_delta = pretrained_ndcg - scratch_ndcg
    scratch_seconds = _efficiency(
        scratch, "time_through_selected_checkpoint_seconds"
    )
    fine_tuning_seconds = _efficiency(
        pretrained, "time_through_selected_checkpoint_seconds"
    )
    pretraining_seconds = _efficiency(
        checkpoint, "required_horizon_train_validation_seconds"
    )
    combined_seconds = pretraining_seconds + fine_tuning_seconds

    acceptance = _acceptance(results)
    _validate_acceptance(
        acceptance,
        recall_delta=recall_delta,
        ndcg_delta=ndcg_delta,
        scratch_seconds=scratch_seconds,
        fine_tuning_seconds=fine_tuning_seconds,
        pretraining_seconds=pretraining_seconds,
        combined_seconds=combined_seconds,
    )
    _validate_method_costs(
        results,
        scratch_seconds=scratch_seconds,
        fine_tuning_seconds=fine_tuning_seconds,
        combined_seconds=combined_seconds,
    )
    quality_improved = recall_delta > _RECALL_BAND and ndcg_delta > _NDCG_BAND
    cold_start_faster = combined_seconds < scratch_seconds
    fine_tuning_exceeds_scratch = fine_tuning_seconds >= scratch_seconds
    if (
        not quality_improved
        or cold_start_faster
        or not fine_tuning_exceeds_scratch
        or acceptance.get("minimum_acceptance_met") is not False
    ):
        raise Rq15ExplanationError(
            "RQ15 does not have the expected quality-improved, cold-start-slower tradeoff"
        )

    budgets = {
        "scratch_from_random_initialization_seconds": scratch_seconds,
        "fine_tuning_stage_only_seconds": fine_tuning_seconds,
        "pretraining_stage_seconds": pretraining_seconds,
        "pretraining_plus_fine_tuning_seconds": combined_seconds,
    }
    quality_compute = _evidence_record(
        {
            "claim": (
                "Pretraining followed by fine-tuning improves the final candidate "
                "metrics, but both the downstream fine-tuning stage alone and the "
                "complete cold-start pipeline require more measured training time than "
                "the selected scratch checkpoint."
            ),
            "scratch_run_name": scratch["run_name"],
            "pretrained_finetune_run_name": pretrained["run_name"],
            "checkpoint_pretraining_run_name": checkpoint["run_name"],
            "scratch_full_recall@100": scratch_recall,
            "pretrained_full_recall@100": pretrained_recall,
            "full_recall@100_delta": recall_delta,
            "scratch_full_ndcg@100": scratch_ndcg,
            "pretrained_full_ndcg@100": pretrained_ndcg,
            "full_ndcg@100_delta": ndcg_delta,
            "quality_improved": quality_improved,
            "quality_resolution_bands": _QUALITY_BANDS,
            "cold_start_faster": cold_start_faster,
            "budgets": budgets,
            "fine_tuning_stage_over_scratch_ratio": (
                fine_tuning_seconds / scratch_seconds
            ),
            "complete_cold_start_over_scratch_ratio": (
                combined_seconds / scratch_seconds
            ),
            "pretraining_fraction_of_complete_cold_start": (
                pretraining_seconds / combined_seconds
            ),
        }
    )
    cold_start = _evidence_record(
        {
            "claim": (
                "The approved speed criterion counts the required pretraining horizon "
                "plus fine-tuning through its selected checkpoint. That measured sum is "
                "larger than scratch through its selected checkpoint, so the speed part "
                "of the minimum acceptance criterion fails even though quality improves."
            ),
            "approved_budget_definition": (
                "pretraining required horizon plus fine-tuning time through the selected "
                "checkpoint"
            ),
            "scratch_seconds": scratch_seconds,
            "pretraining_seconds": pretraining_seconds,
            "fine_tuning_seconds": fine_tuning_seconds,
            "combined_cold_start_seconds": combined_seconds,
            "cold_start_excess_seconds": combined_seconds - scratch_seconds,
            "criterion_met": False,
            "quality_non_inferior": acceptance["quality_non_inferior"],
            "minimum_acceptance_met": acceptance["minimum_acceptance_met"],
        }
    )
    amortization = _evidence_record(
        {
            "claim": (
                "The source checkpoint could in principle be reused, but this experiment "
                "measured no reuse workload and does not establish an amortized serving "
                "or retraining win. Even if pretraining cost approached zero per reuse, "
                "the measured fine-tuning stage remains slower than scratch, so no finite "
                "reuse count can satisfy the approved speed comparison for this recipe."
            ),
            "per_reuse_budget_formula": (
                "fine_tuning_stage_seconds + pretraining_stage_seconds / reuse_count"
            ),
            "measured_reuse_count": None,
            "scratch_seconds": scratch_seconds,
            "fine_tuning_stage_seconds": fine_tuning_seconds,
            "pretraining_stage_seconds": pretraining_seconds,
            "fine_tuning_stage_exceeds_scratch": fine_tuning_exceeds_scratch,
            "asymptotic_seconds_per_fine_tune": fine_tuning_seconds,
            "asymptotic_excess_over_scratch_seconds": (
                fine_tuning_seconds - scratch_seconds
            ),
            "finite_reuse_count_can_beat_scratch": False,
        }
    )
    optimization_path = _optimization_path_evidence(
        scratch=scratch,
        pretrained=pretrained,
        checkpoint=checkpoint,
    )
    return {
        "schema_version": 1,
        "research_question": _RESEARCH_QUESTION,
        "dataset_size": "500m",
        "status": "passed",
        "selected_artifacts": selected_artifacts,
        "acceptance_sha256": _canonical_sha256(acceptance),
        "correctness_audit_sha256": _canonical_sha256(correctness),
        "acceptance_protocol": acceptance_protocol,
        "input_bindings": {
            "artifact_audit_sha256": _canonical_sha256(results["artifact_audit"]),
            "acceptance_protocol_sha256": _canonical_sha256(acceptance_protocol),
            "selected_records_sha256": _canonical_sha256(selected),
            "selected_artifacts_sha256": _canonical_sha256(selected_artifacts),
        },
        "targeted_evidence": {
            "quality_compute_tradeoff": quality_compute,
            "cold_start_criterion": cold_start,
            "amortization_bound": amortization,
            "optimization_path": optimization_path,
        },
        "conclusion": (
            "Pretraining plus fine-tuning improves Recall@100 and NDCG@100 beyond the "
            "shared G1 reporting bands, but it does not meet the approved cold-start speed "
            "criterion. Fine-tuning alone takes longer than scratch through the selected "
            "checkpoint, and adding the required pretraining horizon makes the complete "
            "cold start slower still; no amortized efficiency win was measured."
        ),
    }


def validate_training_budget_explanation(
    explanation: Mapping[str, object],
    results: Mapping[str, object],
    correctness: Mapping[str, object],
    *,
    logs: Path = _DEFAULT_LOGS,
) -> dict[str, object]:
    expected = build_training_budget_explanation(results, correctness, logs=logs)
    if dict(explanation) != expected:
        raise Rq15ExplanationError(
            "RQ15 training-budget explanation is stale or differs from evidence"
        )
    return {
        "status": "passed",
        "schema_version": 1,
        "artifact_sha256": _canonical_sha256(explanation),
    }


def write_explanation(document: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(document, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _validate_resolved_results(results: Mapping[str, object]) -> None:
    claims_status = results.get("claims_status")
    user_validated = results.get("result_claims_user_validated")
    if (
        results.get("schema_version") != 1
        or results.get("research_question") != _RESEARCH_QUESTION
        or results.get("dataset_size") != "500m"
        or results.get("acceptance_criterion") != _ACCEPTANCE_CRITERION
        or claims_status
        not in {
            "acceptance_requires_explanation",
            "ready_for_user_validation",
            "complete",
        }
        or (claims_status == "complete") is not (user_validated is True)
        or results.get("missing_artifacts") != []
        or results.get("required_followups") != []
    ):
        raise Rq15ExplanationError("RQ15 results are not resolved for explanation")


def _acceptance_protocol() -> dict[str, object]:
    try:
        protocol_text = _ARCHITECTURE_PROTOCOL.read_text()
    except OSError as error:
        raise Rq15ExplanationError(
            f"cannot read {_ARCHITECTURE_PROTOCOL}"
        ) from error
    rq15_marker = "### RQ15: decoder-decoder training method"
    approval_marker = "## Approval"
    if rq15_marker not in protocol_text or approval_marker not in protocol_text:
        raise Rq15ExplanationError("RQ15 acceptance section is absent from protocol")
    rq15_section = protocol_text.split(rq15_marker, 1)[1].split(approval_marker, 1)[0]
    if _ACCEPTANCE_CRITERION not in " ".join(rq15_section.split()):
        raise Rq15ExplanationError("RQ15 approved criterion is absent from protocol")
    baseline = _load_json(_BASELINE_SPREAD_EVIDENCE)
    metrics = baseline.get("metrics")
    if baseline.get("n") != 10 or not isinstance(metrics, Mapping):
        raise Rq15ExplanationError("RQ15 empirical band provenance is invalid")
    empirical_bands: dict[str, float] = {}
    for metric, operational_band in _QUALITY_BANDS.items():
        record = metrics.get(metric)
        empirical = record.get("absolute_band") if isinstance(record, Mapping) else None
        empirical_bands[metric] = _positive_number(
            empirical, f"RQ15 {metric} empirical band"
        )
        derived_band = math.ceil(empirical_bands[metric] * 1000) / 1000
        if not math.isclose(derived_band, operational_band, rel_tol=0, abs_tol=1e-12):
            raise Rq15ExplanationError(
                f"RQ15 {metric} operational band does not match empirical rounding"
            )
    return {
        "criterion": _ACCEPTANCE_CRITERION,
        "quality_resolution_bands": _QUALITY_BANDS,
        "empirical_absolute_bands": empirical_bands,
        "operational_band_derivation": "ceil empirical absolute band to 0.001",
        "sources": {
            "architecture_protocol": {
                "path": str(_ARCHITECTURE_PROTOCOL),
                "artifact_sha256": _file_sha256(_ARCHITECTURE_PROTOCOL),
            },
            "baseline_spread": {
                "path": str(_BASELINE_SPREAD_EVIDENCE),
                "artifact_sha256": _file_sha256(_BASELINE_SPREAD_EVIDENCE),
            },
            "reporting_implementation": {
                "path": str(_REPORTING_IMPLEMENTATION),
                "artifact_sha256": _file_sha256(_REPORTING_IMPLEMENTATION),
            },
        },
    }


def _validate_correctness(
    results: Mapping[str, object], correctness: Mapping[str, object]
) -> None:
    artifact_audit = results.get("artifact_audit")
    run_artifacts = (
        artifact_audit.get("run_artifacts")
        if isinstance(artifact_audit, Mapping)
        else None
    )
    if not isinstance(run_artifacts, Mapping) or not run_artifacts:
        raise Rq15ExplanationError("RQ15 artifact audit is absent")
    try:
        validate_correctness_audit(
            correctness,
            run_artifacts,
            _results_binding(results),
        )
    except (Rq15AuditError, Rq15ExplanationError) as error:
        raise Rq15ExplanationError("RQ15 correctness audit is stale or invalid") from error
    record = results.get("correctness_audit")
    if (
        not isinstance(record, Mapping)
        or record.get("status") != "passed"
        or record.get("artifact_sha256") != _canonical_sha256(correctness)
    ):
        raise Rq15ExplanationError("RQ15 result-to-correctness binding is stale")


def _selected_records(
    results: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    winners = results.get("surface_winners")
    checkpoint = results.get("checkpoint_pretraining")
    selected_method = results.get("selected_method")
    if (
        not isinstance(winners, Mapping)
        or set(winners) != _METHODS
        or any(not isinstance(winners[name], Mapping) for name in _METHODS)
        or not isinstance(checkpoint, Mapping)
        or not isinstance(selected_method, Mapping)
        or selected_method.get("training_method") != "pretrained_finetune"
        or selected_method.get("run_name")
        != winners["pretrained_finetune"].get("run_name")
        or selected_method.get("artifact_sha256")
        != winners["pretrained_finetune"].get("artifact_sha256")
    ):
        raise Rq15ExplanationError(
            "RQ15 selected training methods or artifact binding are invalid"
        )
    selected = {name: winners[name] for name in _METHODS}
    selected["checkpoint_pretraining"] = checkpoint
    _validate_selected_ledgers(results, selected)
    return selected


def _validate_selected_ledgers(
    results: Mapping[str, object],
    selected: Mapping[str, Mapping[str, object]],
) -> None:
    artifact_audit = results["artifact_audit"]
    assert isinstance(artifact_audit, Mapping)
    audited = artifact_audit.get("run_artifacts")
    treatments = results.get("treatments")
    source_surface = results.get("checkpoint_pretraining_surface")
    if (
        not isinstance(audited, Mapping)
        or not isinstance(treatments, Mapping)
        or not isinstance(source_surface, list)
    ):
        raise Rq15ExplanationError("RQ15 selected artifact ledgers are absent")
    for method in _METHODS:
        treatment = treatments.get(method)
        artifacts = treatment.get("artifacts") if isinstance(treatment, Mapping) else None
        selected_record = selected[method]
        matches = [
            record
            for record in artifacts or []
            if isinstance(record, Mapping)
            and record.get("run_name") == selected_record.get("run_name")
        ]
        if len(matches) != 1 or dict(matches[0]) != dict(selected_record):
            raise Rq15ExplanationError(f"RQ15 {method} selected artifact is stale")
    checkpoint = selected["checkpoint_pretraining"]
    checkpoint_matches = [
        record
        for record in source_surface
        if isinstance(record, Mapping)
        and record.get("run_name") == checkpoint.get("run_name")
    ]
    if len(checkpoint_matches) != 1 or dict(checkpoint_matches[0]) != dict(checkpoint):
        raise Rq15ExplanationError("RQ15 checkpoint selected artifact is stale")
    for name, record in selected.items():
        run_name = record.get("run_name")
        hashes = record.get("artifact_sha256")
        if (
            not isinstance(run_name, str)
            or not isinstance(hashes, Mapping)
            or audited.get(run_name) != hashes
        ):
            raise Rq15ExplanationError(f"RQ15 {name} artifact hashes are stale")


def _selected_artifact_binding(
    selected: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    return {
        name: {
            "run_name": record["run_name"],
            "artifact_sha256": record["artifact_sha256"],
        }
        for name, record in selected.items()
    }


def _acceptance(results: Mapping[str, object]) -> Mapping[str, object]:
    acceptance = results.get("acceptance")
    if not isinstance(acceptance, Mapping) or acceptance.get("status") != "evaluated":
        raise Rq15ExplanationError("RQ15 acceptance evidence is absent")
    return acceptance


def _validate_acceptance(
    acceptance: Mapping[str, object],
    *,
    recall_delta: float,
    ndcg_delta: float,
    scratch_seconds: float,
    fine_tuning_seconds: float,
    pretraining_seconds: float,
    combined_seconds: float,
) -> None:
    expected_numbers = {
        "recall@100_delta": recall_delta,
        "ndcg@100_delta": ndcg_delta,
        "scratch_time_to_checkpoint_seconds": scratch_seconds,
        "fine_tuning_time_to_checkpoint_seconds": fine_tuning_seconds,
        "pretraining_required_horizon_seconds": pretraining_seconds,
        "pretrained_cold_start_seconds": combined_seconds,
    }
    if any(
        not _same_number(acceptance.get(name), value)
        for name, value in expected_numbers.items()
    ):
        raise Rq15ExplanationError("RQ15 acceptance arithmetic is inconsistent")
    quality_non_inferior = recall_delta >= -_RECALL_BAND and ndcg_delta >= -_NDCG_BAND
    cold_start_faster = combined_seconds < scratch_seconds
    expected_flags = {
        "quality_non_inferior": quality_non_inferior,
        "cold_start_faster": cold_start_faster,
        "minimum_acceptance_met": quality_non_inferior and cold_start_faster,
        "main_metrics_improved": (
            recall_delta > _RECALL_BAND and ndcg_delta > _NDCG_BAND
        ),
    }
    if any(acceptance.get(name) is not value for name, value in expected_flags.items()):
        raise Rq15ExplanationError("RQ15 acceptance flags are inconsistent")


def _validate_method_costs(
    results: Mapping[str, object],
    *,
    scratch_seconds: float,
    fine_tuning_seconds: float,
    combined_seconds: float,
) -> None:
    costs = results.get("method_cold_start_seconds")
    if not isinstance(costs, Mapping) or any(
        not _same_number(costs.get(name), value)
        for name, value in {
            "scratch_candidate_only": scratch_seconds,
            "pretrained_finetune": combined_seconds,
        }.items()
    ):
        raise Rq15ExplanationError("RQ15 method cold-start costs are inconsistent")
    if fine_tuning_seconds <= 0:
        raise Rq15ExplanationError("RQ15 fine-tuning time is invalid")


def _optimization_path_evidence(
    *,
    scratch: Mapping[str, object],
    pretrained: Mapping[str, object],
    checkpoint: Mapping[str, object],
) -> dict[str, object]:
    scratch_best_epoch = _positive_integer(scratch.get("best_epoch"), "scratch epoch")
    pretrained_best_epoch = _positive_integer(
        pretrained.get("best_epoch"), "pretrained fine-tuning epoch"
    )
    checkpoint_horizon = _positive_integer(
        checkpoint.get("stopped_epoch"), "checkpoint pretraining horizon"
    )
    checkpoint_best_epoch = _positive_integer(
        checkpoint.get("best_epoch"), "checkpoint pretraining best epoch"
    )
    scratch_efficiency = _efficiency_record(scratch)
    pretrained_efficiency = _efficiency_record(pretrained)
    scratch_candidate_targets = _positive_integer(
        scratch_efficiency.get("candidate_targets_per_epoch"),
        "scratch candidate targets per epoch",
    )
    pretrained_candidate_targets = _positive_integer(
        pretrained_efficiency.get("candidate_targets_per_epoch"),
        "pretrained candidate targets per epoch",
    )
    scratch_input_tokens = _positive_integer(
        scratch_efficiency.get("input_tokens_per_epoch"),
        "scratch input tokens per epoch",
    )
    pretrained_input_tokens = _positive_integer(
        pretrained_efficiency.get("input_tokens_per_epoch"),
        "pretrained input tokens per epoch",
    )
    scratch_throughput = _positive_number(
        scratch_efficiency.get("steady_state_candidate_targets_per_second"),
        "scratch throughput",
    )
    pretrained_throughput = _positive_number(
        pretrained_efficiency.get("steady_state_candidate_targets_per_second"),
        "pretrained fine-tuning throughput",
    )
    throughput_relative_difference = abs(
        pretrained_throughput - scratch_throughput
    ) / scratch_throughput
    scratch_best_recall = _validation_recall(
        scratch, scratch_best_epoch, "scratch best epoch"
    )
    pretrained_epoch_1_recall = _validation_recall(
        pretrained, 1, "pretrained fine-tuning epoch 1"
    )
    if (
        scratch_candidate_targets != pretrained_candidate_targets
        or scratch_input_tokens != pretrained_input_tokens
        or throughput_relative_difference > 0.05
        or pretrained_best_epoch <= scratch_best_epoch
        or checkpoint_best_epoch > checkpoint_horizon
        or pretrained_epoch_1_recall <= scratch_best_recall
    ):
        raise Rq15ExplanationError("RQ15 optimization-path evidence is inconsistent")
    return _evidence_record(
        {
            "claim": (
                "The selected fine-tuning stage processes the same candidate targets "
                "and input tokens per epoch as scratch at comparable measured "
                "throughput. Its larger selected-checkpoint cost therefore comes from "
                f"selecting epoch {pretrained_best_epoch} instead of epoch "
                f"{scratch_best_epoch}, while cold start additionally includes the "
                f"mandatory {checkpoint_horizon}-epoch source horizon. The selected runs use "
                "different embedding/deep learning rates, but this observed association "
                "does not establish learning-rate causality."
            ),
            "scratch_best_epoch": scratch_best_epoch,
            "pretrained_fine_tuning_best_epoch": pretrained_best_epoch,
            "scratch_lrs": _learning_rates(scratch, "scratch"),
            "pretrained_fine_tuning_lrs": _learning_rates(
                pretrained, "pretrained fine-tuning"
            ),
            "candidate_targets_per_epoch": scratch_candidate_targets,
            "candidate_targets_per_epoch_identical": True,
            "input_tokens_per_epoch": scratch_input_tokens,
            "input_tokens_per_epoch_identical": True,
            "scratch_candidate_targets_per_second": scratch_throughput,
            "pretrained_candidate_targets_per_second": pretrained_throughput,
            "throughput_relative_difference": throughput_relative_difference,
            "throughput_comparable_within_5_percent": True,
            "checkpoint_pretraining_horizon_epochs": checkpoint_horizon,
            "checkpoint_pretraining_best_epoch": checkpoint_best_epoch,
            "pretrained_epoch_1_validation_recall@100": pretrained_epoch_1_recall,
            "scratch_best_validation_recall@100": scratch_best_recall,
            "pretrained_epoch_1_already_above_scratch_best": True,
        }
    )


def _efficiency_record(record: Mapping[str, object]) -> Mapping[str, object]:
    efficiency = record.get("efficiency")
    if not isinstance(efficiency, Mapping):
        raise Rq15ExplanationError("RQ15 efficiency evidence is absent")
    return efficiency


def _validate_selected_raw_artifacts(
    selected: Mapping[str, Mapping[str, object]], logs: Path
) -> None:
    for name, record in selected.items():
        run_name = record["run_name"]
        assert isinstance(run_name, str)
        directory = logs / run_name
        hashes = record["artifact_sha256"]
        assert isinstance(hashes, Mapping)
        for artifact_name, expected_hash in hashes.items():
            if (
                not isinstance(artifact_name, str)
                or not isinstance(expected_hash, str)
                or _file_sha256(directory / artifact_name) != expected_hash
            ):
                raise Rq15ExplanationError(
                    f"RQ15 {name} raw artifact binding is stale"
                )
        _validate_raw_claim_inputs(record, directory, name)
    checkpoint_sha = selected["checkpoint_pretraining"].get("checkpoint_sha256")
    if (
        not isinstance(checkpoint_sha, str)
        or selected["pretrained_finetune"].get("checkpoint_sha256") != checkpoint_sha
    ):
        raise Rq15ExplanationError("RQ15 pretrained checkpoint binding is stale")


def _validate_raw_claim_inputs(
    record: Mapping[str, object], directory: Path, context: str
) -> None:
    metadata = _load_json(directory / "training_metadata.json")
    metrics = _load_json(directory / "final_metrics.json")
    efficiency = _efficiency_record(record)
    scalar_pairs = {
        "best_epoch": (record.get("best_epoch"), metadata.get("best_epoch")),
        "stopped_epoch": (record.get("stopped_epoch"), metadata.get("stopped_epoch")),
        "embedding_lr": (
            record.get("embedding_lr"),
            metadata.get("embedding_learning_rate"),
        ),
        "deep_lr": (record.get("deep_lr"), metadata.get("deep_learning_rate")),
        "optimizer_steps_per_epoch": (
            efficiency.get("optimizer_steps_per_epoch"),
            metadata.get("optimizer_steps_per_epoch"),
        ),
    }
    for name, (reported, raw) in scalar_pairs.items():
        if not _same_scalar(reported, raw):
            raise Rq15ExplanationError(f"RQ15 {context} raw {name} disagrees")
    raw_counts = _raw_epoch_counts(metadata, context)
    for name, raw in raw_counts.items():
        if not _same_scalar(efficiency.get(name), raw):
            raise Rq15ExplanationError(f"RQ15 {context} raw {name} disagrees")
    reported_metrics = record.get("full_user_metrics")
    if not isinstance(reported_metrics, Mapping) or any(
        not _same_scalar(value, metrics.get(name))
        for name, value in reported_metrics.items()
    ):
        raise Rq15ExplanationError(f"RQ15 {context} raw metrics disagree")
    raw_curve = [
        {"epoch": epoch, "recall@100": recall, "ndcg@100": ndcg}
        for epoch, recall, ndcg in _validation_curve(directory)
    ]
    if record.get("validation_curve") != raw_curve:
        raise Rq15ExplanationError(f"RQ15 {context} raw validation curve disagrees")
    best_epoch = metadata.get("best_epoch")
    best_points = [point for point in raw_curve if point["epoch"] == best_epoch]
    validation = record.get("validation_metrics")
    if (
        len(best_points) != 1
        or not isinstance(validation, Mapping)
        or not _same_scalar(validation.get("recall@100"), best_points[0]["recall@100"])
        or not _same_scalar(validation.get("ndcg@100"), best_points[0]["ndcg@100"])
    ):
        raise Rq15ExplanationError(f"RQ15 {context} selected validation point disagrees")
    timings, prepared, finished = _timings(directory)
    selected_timings = timings[: int(best_epoch)]
    candidate_targets = int(raw_counts["candidate_targets_per_epoch"])
    ntp_targets = int(raw_counts["ntp_targets_per_epoch"])
    steady_seconds = sum(point[1] for point in timings[1:])
    steady_epochs = len(timings) - 1
    raw_efficiency = {
        "time_through_selected_checkpoint_seconds": sum(
            sum(point[1:]) for point in selected_timings
        ),
        "required_horizon_train_validation_seconds": sum(
            sum(point[1:]) for point in timings
        ),
        "observed_end_to_end_wall_seconds": (finished - prepared).total_seconds(),
        "steady_state_candidate_targets_per_second": (
            None
            if candidate_targets == 0
            else candidate_targets * steady_epochs / steady_seconds
        ),
        "steady_state_total_targets_per_second": (
            (candidate_targets + ntp_targets) * steady_epochs / steady_seconds
        ),
    }
    if any(
        not _same_scalar(efficiency.get(name), raw)
        for name, raw in raw_efficiency.items()
    ):
        raise Rq15ExplanationError(f"RQ15 {context} raw timing evidence disagrees")


def _raw_epoch_counts(
    metadata: Mapping[str, object], context: str
) -> dict[str, int]:
    if context == "checkpoint_pretraining":
        input_tokens = _positive_integer(
            metadata.get("tokens_per_epoch"), f"{context} input tokens"
        )
        ntp_targets = _positive_integer(
            metadata.get("targets_per_epoch"), f"{context} NTP targets"
        )
        examples = input_tokens - ntp_targets
        if examples <= 0:
            raise Rq15ExplanationError(f"RQ15 {context} examples are invalid")
        return {
            "original_users_per_epoch": examples,
            "expanded_examples_per_epoch": examples,
            "candidate_targets_per_epoch": 0,
            "ntp_targets_per_epoch": ntp_targets,
            "input_tokens_per_epoch": input_tokens,
        }
    return {
        name: _nonnegative_integer(metadata.get(name), f"{context} {name}")
        for name in (
            "original_users_per_epoch",
            "expanded_examples_per_epoch",
            "candidate_targets_per_epoch",
            "ntp_targets_per_epoch",
            "input_tokens_per_epoch",
        )
    }


def _same_scalar(value: object, expected: object) -> bool:
    if value is None or expected is None:
        return value is expected
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and isinstance(expected, (int, float))
        and not isinstance(expected, bool)
    ):
        return math.isclose(float(value), float(expected), rel_tol=0, abs_tol=1e-12)
    return value == expected


def _nonnegative_integer(value: object, context: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise Rq15ExplanationError(f"RQ15 {context} is invalid")
    return value


def _file_sha256(path: Path) -> str:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError as error:
        raise Rq15ExplanationError(f"cannot read {path}") from error


def _learning_rates(record: Mapping[str, object], context: str) -> dict[str, float]:
    return {
        "embedding": _positive_number(
            record.get("embedding_lr"), f"RQ15 {context} embedding LR"
        ),
        "deep": _positive_number(record.get("deep_lr"), f"RQ15 {context} deep LR"),
    }


def _validation_recall(
    record: Mapping[str, object], epoch: int, context: str
) -> float:
    curve = record.get("validation_curve")
    if not isinstance(curve, list):
        raise Rq15ExplanationError(f"RQ15 {context} validation curve is absent")
    matches = [
        point
        for point in curve
        if isinstance(point, Mapping) and point.get("epoch") == epoch
    ]
    if len(matches) != 1:
        raise Rq15ExplanationError(f"RQ15 {context} validation point is absent")
    return _positive_number(matches[0].get("recall@100"), f"RQ15 {context} recall")


def _positive_integer(value: object, context: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise Rq15ExplanationError(f"RQ15 {context} is invalid")
    return value


def _metric(record: Mapping[str, object], name: str) -> float:
    metrics = record.get("full_user_metrics")
    value = metrics.get(name) if isinstance(metrics, Mapping) else None
    return _positive_number(value, f"RQ15 {name}")


def _efficiency(record: Mapping[str, object], name: str) -> float:
    efficiency = record.get("efficiency")
    value = efficiency.get(name) if isinstance(efficiency, Mapping) else None
    return _positive_number(value, f"RQ15 {name}")


def _positive_number(value: object, context: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise Rq15ExplanationError(f"{context} is invalid")
    return float(value)


def _same_number(value: object, expected: float) -> bool:
    return bool(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and math.isclose(float(value), expected, rel_tol=0, abs_tol=1e-12)
    )


def _evidence_record(payload: dict[str, object]) -> dict[str, object]:
    document = {"passed": True, **payload}
    return {**document, "artifact_sha256": _canonical_sha256(document)}


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise Rq15ExplanationError(f"cannot read {path}") from error
    if not isinstance(value, dict):
        raise Rq15ExplanationError(f"{path}: expected a JSON object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the saved RQ15 training-budget explanation"
    )
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--correctness", type=Path, required=True)
    parser.add_argument("--logs", type=Path, default=_DEFAULT_LOGS)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    write_explanation(
        build_training_budget_explanation(
            _load_json(arguments.results),
            _load_json(arguments.correctness),
            logs=arguments.logs,
        ),
        arguments.output,
    )


if __name__ == "__main__":
    main()
