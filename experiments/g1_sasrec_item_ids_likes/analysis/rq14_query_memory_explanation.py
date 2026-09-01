from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path
import tempfile
from typing import Any

from experiments.g1_sasrec_item_ids_likes.analysis.rq13_prefix_expansion_audit import (
    Rq13AuditError,
    validate_correctness_audit as validate_rq13_correctness_audit,
)


_RESEARCH_QUESTION = "RQ14 decoder-decoder query memory"
_EXPECTED_EFFECTS = {
    "distinct_minus_shared_cls_only",
    "distinct_minus_shared_history",
    "history_minus_cls_only_shared",
    "history_minus_cls_only_distinct",
}
_TOKEN_EFFECTS = {
    "distinct_minus_shared_cls_only",
    "distinct_minus_shared_history",
}
_MEMORY_EFFECTS = {
    "history_minus_cls_only_shared",
    "history_minus_cls_only_distinct",
}
_RECALL_BAND = 0.003


class Rq14ExplanationError(RuntimeError):
    pass


def build_unexpected_result_explanation(
    rq14_results: Mapping[str, object],
    rq14_audit: Mapping[str, object],
    rq13_results: Mapping[str, object],
    rq13_audit: Mapping[str, object],
) -> dict[str, object]:
    effects = _effect_mapping(rq14_results, "paired_effects")
    unexpected = _effect_mapping(rq14_results, "unexpected_effects")
    if set(effects) != _EXPECTED_EFFECTS or set(unexpected) != _EXPECTED_EFFECTS:
        raise Rq14ExplanationError("RQ14 unexpected effect set is incomplete")
    if effects != unexpected:
        raise Rq14ExplanationError("RQ14 paired and unexpected effects differ")
    for name, effect in effects.items():
        if (
            effect.get("expected_improvement_beyond_band") is not False
            or not _within_band(effect.get("validation_recall_delta"))
            or not _within_band(effect.get("full_recall_delta"))
        ):
            raise Rq14ExplanationError(
                f"{name}: effect is not an unresolved unexpected result"
            )
    _validate_audit(
        rq14_audit,
        rq14_results,
        research_question=_RESEARCH_QUESTION,
        label="RQ14",
    )
    _validate_rq13_source(rq13_results, rq13_audit)
    if (
        rq14_results.get("research_question") != _RESEARCH_QUESTION
        or rq14_results.get("dataset_size") != "500m"
        or rq14_results.get("required_followups") != []
        or rq13_results.get("research_question")
        != "RQ13 encoder-decoder prefix expansion"
        or rq13_results.get("dataset_size") != "500m"
        or rq13_results.get("required_followups") != []
        or rq13_results.get("claims_status") != "ready_for_user_validation"
    ):
        raise Rq14ExplanationError("RQ13/RQ14 source evidence is unresolved")

    rq14_selected = _selected_mapping(rq14_results, _EXPECTED_EFFECTS)
    overall_rule = rq14_results.get("overall_rule")
    selected_method = rq14_results.get("selected_method")
    if (
        overall_rule
        != "within 0.003 validation Recall@100 choose CLS-only memory, then shared tokens"
        or not isinstance(selected_method, Mapping)
        or selected_method.get("treatment") != _overall_selection(rq14_selected)
        or selected_method.get("run_name")
        != rq14_selected[str(selected_method.get("treatment"))].get("run_name")
    ):
        raise Rq14ExplanationError("RQ14 overall selection or rule is invalid")
    _validate_effect_deltas(effects, rq14_selected)
    selected_treatment = str(selected_method["treatment"])
    rq14_workloads = {
        treatment: _selected_workload(rq14_results, treatment, selection)
        for treatment, selection in rq14_selected.items()
    }
    if any(
        workload["candidate_targets_per_epoch"]
        != workload["original_users_per_epoch"]
        or workload["expanded_examples_per_epoch"]
        != workload["original_users_per_epoch"]
        or workload["ntp_targets_per_epoch"] != 0
        for workload in rq14_workloads.values()
    ):
        raise Rq14ExplanationError(
            "RQ14 treatments are not one-target-per-user candidate-only runs"
        )
    rq14_recalls = {
        treatment: _recall(selection, f"RQ14 {treatment}")
        for treatment, selection in rq14_selected.items()
    }
    one = _rq13_selection(rq13_results, "one_example")
    cap32 = _rq13_selection(rq13_results, "selected_cap_32")
    one_recall = _recall(one["selection"], "RQ13 one-example")
    cap32_recall = _recall(cap32["selection"], "RQ13 cap-32")
    one_targets_per_user = _targets_per_user(one["artifact"], "RQ13 one-example")
    cap32_targets_per_user = _targets_per_user(cap32["artifact"], "RQ13 cap-32")
    if (
        one_targets_per_user != 1
        or cap32_targets_per_user <= one_targets_per_user
        or cap32_recall - one_recall <= _RECALL_BAND
        or max(abs(value - one_recall) for value in rq14_recalls.values())
        > _RECALL_BAND
    ):
        raise Rq14ExplanationError(
            "RQ13 evidence does not support the supervision-density finding"
        )

    token_evidence = _axis_evidence(effects, _TOKEN_EFFECTS)
    memory_evidence = _axis_evidence(effects, _MEMORY_EFFECTS)
    return {
        "schema_version": 1,
        "research_question": _RESEARCH_QUESTION,
        "dataset_size": "500m",
        "status": "passed",
        "input_bindings": {
            "rq14_paired_effects_sha256": _canonical_sha256(effects),
            "rq14_selected_and_workloads_sha256": _canonical_sha256(
                {
                    "selected": rq14_selected,
                    "workloads": rq14_workloads,
                    "overall_rule": overall_rule,
                    "selected_method": selected_method,
                }
            ),
            "rq14_correctness_audit_sha256": _canonical_sha256(rq14_audit),
            "rq13_results_sha256": _canonical_sha256(rq13_results),
            "rq13_correctness_audit_sha256": _canonical_sha256(rq13_audit),
        },
        "findings": {
            "audit_scope": {
                "claim": (
                    "The current audit verifies exact memory contents, target exclusion, "
                    "gradient flow to every slot and history, candidate-only training, and "
                    "complete LR/horizon resolution. Shared repeated tokens still occupy "
                    "ordered causal positions and therefore form contextual slot states."
                ),
                "supported_by": [
                    "query_slot_identity_and_order",
                    "memory_content_and_lengths",
                    "target_exclusion_and_candidate_only_loss",
                    "gradient_flow_to_every_slot_and_history",
                    "learning_curves_and_lr_boundaries",
                ],
            },
            "axis_effects": {
                "distinct_token_identity": {
                    "covers": sorted(_TOKEN_EFFECTS),
                    "claim": (
                        "Four distinct query-token identities do not produce a resolved "
                        "Recall@100 gain over one repeated token on either memory surface."
                    ),
                    "evidence": token_evidence,
                },
                "history_memory": {
                    "covers": sorted(_MEMORY_EFFECTS),
                    "claim": (
                        "Exposing history plus CLS states does not produce a resolved "
                        "Recall@100 gain for either shared or distinct query tokens."
                    ),
                    "evidence": memory_evidence,
                },
            },
            "supervision_density": {
                "claim": (
                    "The unresolved token and memory effects are consistent with a "
                    "one-target-per-user supervision bottleneck, not evidence that the "
                    "added states are inaccessible. RQ13 supplies controlled supporting "
                    "evidence; it does not prove this mechanism causally for RQ14."
                ),
                "rq14_candidate_targets_per_user": 1.0,
                "rq14_full_recall_range": [
                    min(rq14_recalls.values()),
                    max(rq14_recalls.values()),
                ],
                "rq13_one_example_candidate_targets_per_user": one_targets_per_user,
                "rq13_one_example_recall": one_recall,
                "rq13_cap32_candidate_targets_per_user": cap32_targets_per_user,
                "rq13_cap32_recall": cap32_recall,
                "rq13_cap32_recall_gain": cap32_recall - one_recall,
            },
        },
        "covered_unexpected_effects": sorted(_EXPECTED_EFFECTS),
        "conclusion": (
            "All four planned effects are unresolved inside the native-500M 0.003 "
            "Recall@100 band after correctness and LR checks. Distinct tokens and "
            "history memory have no individually resolved effect; the predeclared "
            f"overall rule selects {selected_treatment}."
        ),
    }


def validate_unexpected_result_explanation(
    explanation: Mapping[str, object],
    rq14_results: Mapping[str, object],
    rq14_audit: Mapping[str, object],
    rq13_results: Mapping[str, object],
    rq13_audit: Mapping[str, object],
) -> dict[str, object]:
    expected = build_unexpected_result_explanation(
        rq14_results, rq14_audit, rq13_results, rq13_audit
    )
    if dict(explanation) != expected:
        raise Rq14ExplanationError(
            "RQ14 unexpected-result explanation is stale or differs from evidence"
        )
    return {
        "status": "passed",
        "schema_version": 1,
        "artifact_sha256": _canonical_sha256(explanation),
        "covered_unexpected_effects": sorted(_EXPECTED_EFFECTS),
    }


def write_explanation(document: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(document, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _effect_mapping(
    results: Mapping[str, object], key: str
) -> dict[str, dict[str, object]]:
    value = results.get(key)
    if not isinstance(value, Mapping) or any(
        not isinstance(name, str) or not isinstance(effect, Mapping)
        for name, effect in value.items()
    ):
        raise Rq14ExplanationError(f"RQ14 {key} is invalid")
    return {str(name): dict(effect) for name, effect in value.items()}


def _selected_mapping(
    results: Mapping[str, object], effects: set[str]
) -> dict[str, dict[str, object]]:
    selected = results.get("selected")
    paired = _effect_mapping(results, "paired_effects")
    treatments = set()
    for name in effects:
        for axis in ("control", "treatment"):
            treatment = paired[name].get(axis)
            if not isinstance(treatment, str):
                raise Rq14ExplanationError(f"RQ14 {name} has invalid {axis}")
            treatments.add(treatment)
    if (
        not isinstance(selected, Mapping)
        or set(selected) != treatments
        or any(not isinstance(value, Mapping) for value in selected.values())
    ):
        raise Rq14ExplanationError("RQ14 selected treatments are incomplete")
    return {str(name): dict(value) for name, value in selected.items()}


def _overall_selection(selected: Mapping[str, Mapping[str, object]]) -> str:
    metrics = {}
    for treatment, record in selected.items():
        validation = record.get("validation_metrics")
        recall = validation.get("recall@100") if isinstance(validation, Mapping) else None
        ndcg = validation.get("ndcg@100") if isinstance(validation, Mapping) else None
        if any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            for value in (recall, ndcg)
        ):
            raise Rq14ExplanationError(f"RQ14 {treatment} validation metrics are invalid")
        metrics[treatment] = (float(recall), float(ndcg))
    best_recall = max(recall for recall, _ in metrics.values())
    eligible = [
        treatment
        for treatment, (recall, _) in metrics.items()
        if best_recall - recall <= _RECALL_BAND
    ]
    return min(
        eligible,
        key=lambda treatment: (
            treatment.endswith("_history"),
            treatment.startswith("distinct_"),
            -metrics[treatment][0],
            -metrics[treatment][1],
        ),
    )


def _validate_effect_deltas(
    effects: Mapping[str, Mapping[str, object]],
    selected: Mapping[str, Mapping[str, object]],
) -> None:
    for name, effect in effects.items():
        control = selected[str(effect["control"])]
        treatment = selected[str(effect["treatment"])]
        for result_key, metrics_key in (
            ("validation_recall_delta", "validation_metrics"),
            ("full_recall_delta", "full_user_metrics"),
        ):
            control_metrics = control.get(metrics_key)
            treatment_metrics = treatment.get(metrics_key)
            control_recall = (
                control_metrics.get("recall@100")
                if isinstance(control_metrics, Mapping)
                else None
            )
            treatment_recall = (
                treatment_metrics.get("recall@100")
                if isinstance(treatment_metrics, Mapping)
                else None
            )
            expected = effect.get(result_key)
            if any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                for value in (control_recall, treatment_recall, expected)
            ) or not math.isclose(
                float(treatment_recall) - float(control_recall),
                float(expected),
                rel_tol=0,
                abs_tol=1e-12,
            ):
                raise Rq14ExplanationError(f"RQ14 {name} {result_key} is inconsistent")


def _selected_workload(
    results: Mapping[str, object], treatment: str, selection: Mapping[str, object]
) -> dict[str, int]:
    treatments = results.get("treatments")
    record = treatments.get(treatment) if isinstance(treatments, Mapping) else None
    artifacts = record.get("artifacts") if isinstance(record, Mapping) else None
    run_name = selection.get("run_name")
    matches = [
        artifact
        for artifact in artifacts or []
        if isinstance(artifact, Mapping) and artifact.get("run_name") == run_name
    ]
    if len(matches) != 1 or not isinstance(matches[0].get("efficiency"), Mapping):
        raise Rq14ExplanationError(f"RQ14 {treatment} selected workload is absent")
    efficiency = matches[0]["efficiency"]
    keys = (
        "original_users_per_epoch",
        "expanded_examples_per_epoch",
        "candidate_targets_per_epoch",
        "ntp_targets_per_epoch",
    )
    if any(
        not isinstance(efficiency.get(key), int)
        or isinstance(efficiency.get(key), bool)
        or efficiency[key] < 0
        for key in keys
    ):
        raise Rq14ExplanationError(f"RQ14 {treatment} workload counts are invalid")
    return {key: int(efficiency[key]) for key in keys}


def _rq13_selection(
    results: Mapping[str, object], treatment: str
) -> dict[str, Mapping[str, object]]:
    selected = results.get("selected")
    selection = selected.get(treatment) if isinstance(selected, Mapping) else None
    treatments = results.get("treatments")
    treatment_record = (
        treatments.get(treatment) if isinstance(treatments, Mapping) else None
    )
    artifacts = (
        treatment_record.get("artifacts")
        if isinstance(treatment_record, Mapping)
        else None
    )
    if not isinstance(selection, Mapping) or not isinstance(artifacts, list):
        raise Rq14ExplanationError(f"RQ13 {treatment} evidence is absent")
    matches = [
        artifact
        for artifact in artifacts
        if isinstance(artifact, Mapping)
        and artifact.get("run_name") == selection.get("run_name")
    ]
    if len(matches) != 1:
        raise Rq14ExplanationError(f"RQ13 {treatment} selected artifact is absent")
    return {"selection": selection, "artifact": matches[0]}


def _recall(selection: Mapping[str, object], context: str) -> float:
    metrics = selection.get("full_user_metrics")
    value = metrics.get("recall@100") if isinstance(metrics, Mapping) else None
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise Rq14ExplanationError(f"{context} Recall@100 is invalid")
    return float(value)


def _targets_per_user(artifact: Mapping[str, object], context: str) -> float:
    users = artifact.get("original_users_per_epoch")
    targets = artifact.get("candidate_targets_per_epoch")
    if (
        not isinstance(users, int)
        or isinstance(users, bool)
        or users <= 0
        or not isinstance(targets, int)
        or isinstance(targets, bool)
        or targets <= 0
        or artifact.get("expanded_examples_per_epoch") != targets
        or artifact.get("ntp_targets_per_epoch") != 0
    ):
        raise Rq14ExplanationError(f"{context} supervision counts are invalid")
    return targets / users


def _axis_evidence(
    effects: Mapping[str, Mapping[str, object]], names: set[str]
) -> dict[str, dict[str, float]]:
    return {
        name: {
            "validation_recall_delta": float(effects[name]["validation_recall_delta"]),
            "full_recall_delta": float(effects[name]["full_recall_delta"]),
            "recall_resolution_band": _RECALL_BAND,
        }
        for name in sorted(names)
    }


def _validate_audit(
    audit: Mapping[str, object],
    results: Mapping[str, object],
    *,
    research_question: str,
    label: str,
) -> None:
    checks = audit.get("checks")
    record = results.get("correctness_audit")
    if (
        audit.get("schema_version") != 1
        or audit.get("research_question") != research_question
        or audit.get("dataset_size") != "500m"
        or audit.get("status") != "passed"
        or not isinstance(checks, Mapping)
        or not checks
        or any(
            not isinstance(check, Mapping) or check.get("passed") is not True
            for check in checks.values()
        )
        or not isinstance(record, Mapping)
        or record.get("status") != "passed"
        or record.get("artifact_sha256") != _canonical_sha256(audit)
    ):
        raise Rq14ExplanationError(f"{label} correctness audit is stale or invalid")


def _validate_rq13_source(
    results: Mapping[str, object], audit: Mapping[str, object]
) -> None:
    treatments = results.get("treatments")
    if not isinstance(treatments, Mapping):
        raise Rq14ExplanationError("RQ13 treatment evidence is absent")
    expected_artifacts = {}
    for record in treatments.values():
        artifacts = record.get("artifacts") if isinstance(record, Mapping) else None
        if not isinstance(artifacts, list):
            raise Rq14ExplanationError("RQ13 treatment artifacts are invalid")
        for artifact in artifacts:
            run_name = artifact.get("run_name") if isinstance(artifact, Mapping) else None
            hashes = (
                artifact.get("artifact_sha256")
                if isinstance(artifact, Mapping)
                else None
            )
            if (
                not isinstance(run_name, str)
                or not isinstance(hashes, Mapping)
                or run_name in expected_artifacts
            ):
                raise Rq14ExplanationError("RQ13 artifact binding is invalid")
            expected_artifacts[run_name] = dict(hashes)
    if not expected_artifacts:
        raise Rq14ExplanationError("RQ13 artifact binding is empty")
    try:
        validate_rq13_correctness_audit(audit, expected_artifacts)
    except Rq13AuditError as error:
        raise Rq14ExplanationError(
            "RQ13 correctness audit is stale or invalid"
        ) from error
    record = results.get("correctness_audit")
    if (
        not isinstance(record, Mapping)
        or record.get("status") != "passed"
        or record.get("artifact_sha256") != _canonical_sha256(audit)
    ):
        raise Rq14ExplanationError("RQ13 result-to-audit binding is stale or invalid")


def _within_band(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and abs(value) <= _RECALL_BAND
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise Rq14ExplanationError(f"cannot read {path}") from error
    if not isinstance(value, dict):
        raise Rq14ExplanationError(f"{path}: expected a JSON object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rq14-results", type=Path, required=True)
    parser.add_argument("--rq14-audit", type=Path, required=True)
    parser.add_argument("--rq13-results", type=Path, required=True)
    parser.add_argument("--rq13-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_explanation(
        build_unexpected_result_explanation(
            _load_json(args.rq14_results),
            _load_json(args.rq14_audit),
            _load_json(args.rq13_results),
            _load_json(args.rq13_audit),
        ),
        args.output,
    )


if __name__ == "__main__":
    main()
