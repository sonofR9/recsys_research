import json
from pathlib import Path

import pytest

from experiments.g4_future_items.report.native500m_rq3_decision import (
    load_rq3_decision,
    validate_rq3_decision_document,
    validate_rq3_feasibility_audit,
)


_AUDIT_SHA256 = "dd679b2b2c580e87bd831d67063bfddafbf2a24380b9c430058b65fb9842c28a"


def _decision() -> dict:
    return {
        "schema_version": 1,
        "kind": "g4_native500m_rq3_decision",
        "dataset_size": "native-500m",
        "status": "preselector_stop",
        "audit": {
            "path": "experiments/g4_future_items/protocol/native500m/evidence/rq3_feasibility_audit_v3.json",
            "sha256": _AUDIT_SHA256,
        },
        "user_validation": {
            "validated": True,
            "scope": "RQ3 pre-selector feasibility stop",
            "message_sha256": "da0fd6b46d4bb41f96bc9c01a4e975c3260eae4130f158d12e68f11ebe892f6a",
            "audit_sha256": _AUDIT_SHA256,
        },
    }


def _audit() -> dict:
    return {
        "schema_version": 3,
        "kind": "g4_native500m_rq3_feasibility_audit",
        "dataset_size": "native-500m",
        "method": "Static exact-source audit of the approved sklearn 1.6.1 fit path; no absolute memory or runtime projection is claimed.",
        "environment": {
            "classifier": "sklearn.ensemble.HistGradientBoostingClassifier",
            "sklearn_version": "1.6.1",
        },
        "implementation_sources": [
            {"identity": "source", "sha256": "2" * 64, "size": 1}
        ],
        "structural_findings": {
            "external_memory_fit": False,
            "fit_accepts_complete_feature_matrix": True,
            "fit_materializes_complete_binned_matrix": True,
            "row_state_scaling": "linear",
        },
        "acceptance_rule": (
            "If it is too hard to implement or it will run too long, don't try it."
        ),
        "decision_basis": {
            "reason": "The approved classifier requires a population-sized in-memory fit and has no external-memory fit path.",
            "user_validated_stop": True,
        },
        "conclusion": "stop before selector search",
    }


def test_rq3_stop_requires_exact_audit_and_user_validation() -> None:
    assert validate_rq3_decision_document(_decision())["status"] == "preselector_stop"

    missing_validation = _decision()
    missing_validation.pop("user_validation")
    with pytest.raises(ValueError, match="user validation"):
        validate_rq3_decision_document(missing_validation)

    changed_approval = _decision()
    changed_approval["user_validation"]["message_sha256"] = "1" * 64
    with pytest.raises(ValueError, match="message hash"):
        validate_rq3_decision_document(changed_approval)

    changed_audit = _audit()
    changed_audit["decision_basis"]["user_validated_stop"] = False
    with pytest.raises(ValueError, match="identity"):
        validate_rq3_feasibility_audit(changed_audit)

    changed_binding = _decision()
    changed_binding["audit"]["sha256"] = "1" * 64
    changed_binding["user_validation"]["audit_sha256"] = "1" * 64
    with pytest.raises(ValueError, match="audit fact"):
        validate_rq3_decision_document(changed_binding)


def test_rq3_pending_decision_cannot_freeze_the_aggregate() -> None:
    pending = _decision()
    pending["status"] = "pending_validation"
    pending.pop("user_validation")

    assert validate_rq3_decision_document(pending)["status"] == "pending_validation"


def test_rq3_decision_loads_only_from_the_fixed_evidence_path(tmp_path) -> None:
    root = Path.cwd()
    expected = (
        root
        / "experiments/g4_future_items/protocol/native500m/evidence/rq3_decision.json"
    )

    loaded = load_rq3_decision(root, expected)
    assert loaded["status"] == "preselector_stop"
    assert loaded["audit_document"]["decision_basis"]["user_validated_stop"] is True

    alternate = tmp_path / "alternate.json"
    alternate.write_text(json.dumps(_decision()))
    with pytest.raises(ValueError, match="fixed evidence path"):
        load_rq3_decision(root, alternate)
