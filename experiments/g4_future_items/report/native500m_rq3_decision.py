from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from pathlib import Path
from typing import Any, Mapping


_APPROVAL_MESSAGE_SHA256 = (
    "da0fd6b46d4bb41f96bc9c01a4e975c3260eae4130f158d12e68f11ebe892f6a"
)
_APPROVED_AUDIT_SHA256 = (
    "dd679b2b2c580e87bd831d67063bfddafbf2a24380b9c430058b65fb9842c28a"
)
_AUDIT_PATH = (
    "experiments/g4_future_items/protocol/native500m/evidence/"
    "rq3_feasibility_audit_v3.json"
)
_DECISION_IDENTITY = {
    "schema_version": 1,
    "kind": "g4_native500m_rq3_decision",
    "dataset_size": "native-500m",
}
_VALIDATION = {
    "validated": True,
    "scope": "RQ3 pre-selector feasibility stop",
    "message_sha256": _APPROVAL_MESSAGE_SHA256,
    "audit_sha256": _APPROVED_AUDIT_SHA256,
}


def validate_rq3_decision_document(document: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(document, Mapping):
        raise ValueError("RQ3 decision must be an object")
    status = document.get("status")
    if status not in {"pending_validation", "preselector_stop"}:
        raise ValueError("unsupported RQ3 decision status")
    expected_keys = {*_DECISION_IDENTITY, "status", "audit"}
    if status == "preselector_stop":
        expected_keys.add("user_validation")
    if set(document) != expected_keys:
        if status == "preselector_stop" and "user_validation" not in document:
            raise ValueError("RQ3 stop requires user validation")
        raise ValueError("RQ3 decision schema differs")
    for field, expected in _DECISION_IDENTITY.items():
        if document[field] != expected:
            raise ValueError(f"RQ3 decision identity differs at {field}")
    _validate_audit_fact(document["audit"])
    if status == "preselector_stop":
        _validate_user_validation(document["user_validation"])
    return dict(document)


def validate_rq3_feasibility_audit(document: Mapping[str, Any]) -> dict[str, Any]:
    expected_keys = {
        "schema_version",
        "kind",
        "dataset_size",
        "method",
        "environment",
        "implementation_sources",
        "structural_findings",
        "acceptance_rule",
        "decision_basis",
        "conclusion",
    }
    if not isinstance(document, Mapping) or set(document) != expected_keys:
        raise ValueError("RQ3 feasibility audit schema differs")
    if (
        document["schema_version"] != 3
        or document["kind"] != "g4_native500m_rq3_feasibility_audit"
        or document["dataset_size"] != "native-500m"
        or document["method"]
        != "Static exact-source audit of the approved sklearn 1.6.1 fit path; no absolute memory or runtime projection is claimed."
        or document["environment"]
        != {
            "classifier": "sklearn.ensemble.HistGradientBoostingClassifier",
            "sklearn_version": "1.6.1",
        }
        or document["structural_findings"]
        != {
            "external_memory_fit": False,
            "fit_accepts_complete_feature_matrix": True,
            "fit_materializes_complete_binned_matrix": True,
            "row_state_scaling": "linear",
        }
        or document["acceptance_rule"]
        != "If it is too hard to implement or it will run too long, don't try it."
        or document["decision_basis"]
        != {
            "reason": "The approved classifier requires a population-sized in-memory fit and has no external-memory fit path.",
            "user_validated_stop": True,
        }
        or document["conclusion"] != "stop before selector search"
    ):
        raise ValueError("RQ3 feasibility audit identity differs")
    sources = document["implementation_sources"]
    if not isinstance(sources, list) or not sources:
        raise ValueError("RQ3 feasibility audit sources differ")
    for source in sources:
        if (
            not isinstance(source, Mapping)
            or set(source) != {"identity", "sha256", "size"}
            or not isinstance(source["identity"], str)
            or not isinstance(source["size"], int)
            or source["size"] < 1
            or not _is_sha256(source["sha256"])
        ):
            raise ValueError("RQ3 feasibility audit source fact differs")
    return dict(document)


def load_rq3_decision(repo_root: Path, decision_path: Path) -> dict[str, Any]:
    root = repo_root.resolve(strict=True)
    expected = root / (
        "experiments/g4_future_items/protocol/native500m/evidence/rq3_decision.json"
    )
    resolved = decision_path.resolve(strict=True)
    if resolved != expected:
        raise ValueError("RQ3 decision must use the fixed evidence path")
    decision = validate_rq3_decision_document(_document(resolved))
    audit_path = root / _AUDIT_PATH
    audit_payload = audit_path.read_bytes()
    if hashlib.sha256(audit_payload).hexdigest() != decision["audit"]["sha256"]:
        raise ValueError("RQ3 feasibility audit hash differs")
    audit = validate_rq3_feasibility_audit(json.loads(audit_payload))
    _verify_audit_sources(root, audit["implementation_sources"])
    return decision | {"audit_document": audit}


def _validate_audit_fact(value: Any) -> None:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"path", "sha256"}
        or value["path"] != _AUDIT_PATH
        or value["sha256"] != _APPROVED_AUDIT_SHA256
    ):
        raise ValueError("RQ3 feasibility audit fact differs")


def _validate_user_validation(value: Any) -> None:
    if not isinstance(value, Mapping) or dict(value) != _VALIDATION:
        if isinstance(value, Mapping) and value.get("message_sha256") != (
            _APPROVAL_MESSAGE_SHA256
        ):
            raise ValueError("RQ3 user validation message hash differs")
        raise ValueError("RQ3 user validation differs")


def _verify_audit_sources(root: Path, sources: list[dict[str, Any]]) -> None:
    for source in sources:
        identity = source["identity"]
        if identity.startswith("repo:"):
            path = (root / identity.removeprefix("repo:")).resolve()
            if not path.is_relative_to(root):
                raise ValueError("RQ3 feasibility audit source escapes repository")
        elif identity.startswith("python:"):
            module_name = identity.removeprefix("python:")
            spec = importlib.util.find_spec(module_name)
            if spec is None or spec.origin is None:
                raise ValueError(f"RQ3 feasibility audit source is missing: {identity}")
            path = Path(spec.origin).resolve()
        else:
            raise ValueError(
                f"RQ3 feasibility audit source identity differs: {identity}"
            )
        if (
            not path.is_file()
            or path.stat().st_size != source["size"]
            or hashlib.sha256(path.read_bytes()).hexdigest() != source["sha256"]
        ):
            raise ValueError(f"RQ3 feasibility audit source fact differs: {identity}")


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _document(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"RQ3 evidence is not valid JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"RQ3 evidence must be an object: {path}")
    return value
