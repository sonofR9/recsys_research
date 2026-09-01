from __future__ import annotations

import copy

import pytest

from experiments.g1_sasrec_item_ids_likes.analysis import (
    rq14_query_memory_explanation as explanation_module,
)
from experiments.g1_sasrec_item_ids_likes.analysis.rq14_query_memory_explanation import (
    Rq14ExplanationError,
    build_unexpected_result_explanation,
    validate_unexpected_result_explanation,
)


EFFECTS = {
    "distinct_minus_shared_cls_only": {
        "control": "shared_cls_only",
        "treatment": "distinct_cls_only",
        "validation_recall_delta": 0.0003,
        "full_recall_delta": 0.0002,
        "expected_improvement_beyond_band": False,
    },
    "distinct_minus_shared_history": {
        "control": "shared_history",
        "treatment": "distinct_history",
        "validation_recall_delta": 0.0002,
        "full_recall_delta": 0.0004,
        "expected_improvement_beyond_band": False,
    },
    "history_minus_cls_only_shared": {
        "control": "shared_cls_only",
        "treatment": "shared_history",
        "validation_recall_delta": 0.0,
        "full_recall_delta": 0.0003,
        "expected_improvement_beyond_band": False,
    },
    "history_minus_cls_only_distinct": {
        "control": "distinct_cls_only",
        "treatment": "distinct_history",
        "validation_recall_delta": -0.0001,
        "full_recall_delta": 0.0005,
        "expected_improvement_beyond_band": False,
    },
}


def _rq14_results() -> dict[str, object]:
    metrics = {
        "shared_cls_only": (0.0784, 0.0784),
        "distinct_cls_only": (0.0787, 0.0786),
        "shared_history": (0.0784, 0.0787),
        "distinct_history": (0.0786, 0.0791),
    }
    return {
        "research_question": "RQ14 decoder-decoder query memory",
        "dataset_size": "500m",
        "claims_status": "unexpected_result_requires_explanation",
        "required_followups": [],
        "overall_rule": (
            "within 0.003 validation Recall@100 choose CLS-only memory, then shared tokens"
        ),
        "paired_effects": copy.deepcopy(EFFECTS),
        "unexpected_effects": copy.deepcopy(EFFECTS),
        "correctness_audit": {
            "status": "passed",
            "schema_version": 1,
            "artifact_sha256": "rq14-audit-hash",
        },
        "selected": {
            treatment: {
                "run_name": treatment,
                "validation_metrics": {"recall@100": validation, "ndcg@100": 0.05},
                "full_user_metrics": {"recall@100": full},
            }
            for treatment, (validation, full) in metrics.items()
        },
        "selected_method": {
            "treatment": "shared_cls_only",
            "run_name": "shared_cls_only",
        },
        "treatments": {
            treatment: {
                "artifacts": [
                    {
                        "run_name": treatment,
                        "efficiency": {
                            "original_users_per_epoch": 100,
                            "expanded_examples_per_epoch": 100,
                            "candidate_targets_per_epoch": 100,
                            "ntp_targets_per_epoch": 0,
                        },
                    }
                ]
            }
            for treatment in metrics
        },
    }


def _audit(research_question: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "research_question": research_question,
        "dataset_size": "500m",
        "status": "passed",
        "checks": {"check": {"passed": True}},
    }


def _rq13_results() -> dict[str, object]:
    records = {
        "one_example": ("one", 0.0774, 100, 100),
        "selected_cap_32": ("cap32", 0.1255, 100, 2_350),
    }
    return {
        "research_question": "RQ13 encoder-decoder prefix expansion",
        "dataset_size": "500m",
        "claims_status": "ready_for_user_validation",
        "required_followups": [],
        "correctness_audit": {
            "status": "passed",
            "schema_version": 1,
            "artifact_sha256": "rq13-audit-hash",
        },
        "selected": {
            treatment: {
                "run_name": run_name,
                "full_user_metrics": {"recall@100": recall},
            }
            for treatment, (run_name, recall, _, _) in records.items()
        },
        "treatments": {
            treatment: {
                "artifacts": [
                    {
                        "run_name": run_name,
                        "original_users_per_epoch": users,
                        "expanded_examples_per_epoch": targets,
                        "candidate_targets_per_epoch": targets,
                        "ntp_targets_per_epoch": 0,
                        "artifact_sha256": {
                            "training_metadata.json": run_name + "-metadata",
                            "final_metrics.json": run_name + "-metrics",
                            "sweep.log": run_name + "-log",
                        },
                    }
                ]
            }
            for treatment, (run_name, _, users, targets) in records.items()
        },
    }


def _inputs():
    rq14 = _rq14_results()
    rq14_audit = _audit("RQ14 decoder-decoder query memory")
    rq13 = _rq13_results()
    rq13_audit = _audit("RQ13 encoder-decoder prefix expansion")
    rq14["correctness_audit"]["artifact_sha256"] = _canonical(rq14_audit)
    rq13["correctness_audit"]["artifact_sha256"] = _canonical(rq13_audit)
    return rq14, rq14_audit, rq13, rq13_audit


def _canonical(value: object) -> str:
    import hashlib
    import json

    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _accept_toy_rq13_audit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        explanation_module,
        "validate_rq13_correctness_audit",
        lambda audit, artifacts: {"status": "passed"},
    )


def test_explanation_covers_every_unexpected_effect_with_bound_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _accept_toy_rq13_audit(monkeypatch)
    inputs = _inputs()

    explanation = build_unexpected_result_explanation(*inputs)

    assert explanation["status"] == "passed"
    covered = {
        effect
        for finding in explanation["findings"]["axis_effects"].values()
        for effect in finding["covers"]
    }
    assert covered == set(EFFECTS)
    mechanism = explanation["findings"]["supervision_density"]
    assert mechanism["rq14_candidate_targets_per_user"] == 1.0
    assert mechanism["rq13_cap32_candidate_targets_per_user"] == 23.5
    assert mechanism["rq13_cap32_recall_gain"] == pytest.approx(0.0481)
    assert validate_unexpected_result_explanation(explanation, *inputs)["status"] == "passed"


@pytest.mark.parametrize("changed", ["effects", "rq14_audit", "rq13_results"])
def test_explanation_fails_closed_when_bound_evidence_changes(
    changed: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _accept_toy_rq13_audit(monkeypatch)
    inputs = list(_inputs())
    explanation = build_unexpected_result_explanation(*inputs)
    if changed == "effects":
        inputs[0]["paired_effects"]["distinct_minus_shared_cls_only"][
            "full_recall_delta"
        ] = 0.01
    elif changed == "rq14_audit":
        inputs[1]["checks"]["new"] = {"passed": True}
    else:
        inputs[2]["selected"]["selected_cap_32"]["full_user_metrics"][
            "recall@100"
        ] = 0.13

    with pytest.raises(Rq14ExplanationError, match="stale|differ"):
        validate_unexpected_result_explanation(explanation, *inputs)


def test_explanation_rejects_uncovered_effect() -> None:
    inputs = _inputs()
    inputs[0]["unexpected_effects"]["another_unexpected_effect"] = {
        "full_recall_delta": 0.0,
        "validation_recall_delta": 0.0,
        "expected_improvement_beyond_band": False,
    }

    with pytest.raises(Rq14ExplanationError, match="effect set"):
        build_unexpected_result_explanation(*inputs)


def test_explanation_rejects_self_consistent_but_incomplete_rq13_audit() -> None:
    with pytest.raises(Rq14ExplanationError, match="RQ13 correctness audit"):
        build_unexpected_result_explanation(*_inputs())


def test_explanation_binds_cumulative_overall_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _accept_toy_rq13_audit(monkeypatch)
    inputs = list(_inputs())
    recalls = {
        "shared_cls_only": 0.0774,
        "distinct_cls_only": 0.0799,
        "shared_history": 0.0799,
        "distinct_history": 0.0824,
    }
    for treatment, recall in recalls.items():
        inputs[0]["selected"][treatment]["validation_metrics"]["recall@100"] = recall
        inputs[0]["selected"][treatment]["full_user_metrics"]["recall@100"] = recall
    inputs[0]["selected_method"] = {
        "treatment": "distinct_cls_only",
        "run_name": "distinct_cls_only",
    }
    inputs[2]["selected"]["one_example"]["full_user_metrics"]["recall@100"] = 0.0799
    inputs[2]["selected"]["selected_cap_32"]["full_user_metrics"]["recall@100"] = 0.128
    for effect in inputs[0]["paired_effects"].values():
        effect["validation_recall_delta"] = 0.0025
        effect["full_recall_delta"] = 0.0025
    inputs[0]["unexpected_effects"] = copy.deepcopy(inputs[0]["paired_effects"])

    explanation = build_unexpected_result_explanation(*inputs)

    assert "selects distinct_cls_only" in explanation["conclusion"]
