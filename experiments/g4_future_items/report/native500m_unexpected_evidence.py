from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from experiments.g4_future_items.report.native500m_evidence import canonical_bytes


_ROLES = ("control_next_item", "rq2_next10", "rq1_24h")
_TREATMENTS = ("rq1_24h", "rq2_next10")
_APPROVAL_MESSAGE_SHA256 = (
    "da0fd6b46d4bb41f96bc9c01a4e975c3260eae4130f158d12e68f11ebe892f6a"
)


def build_unexpected_result_diagnostics(
    evaluation: Mapping[str, Any],
    target_statistics: Mapping[str, Any],
    *,
    user_validation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if evaluation.get("kind") != "g4_rq1_rq2_evaluation_native500m":
        raise ValueError("unexpected-result evaluation identity differs")
    if target_statistics.get("kind") != "g4_native500m_target_statistics":
        raise ValueError("unexpected-result target-statistics identity differs")
    overall = evaluation["overall"]["rows"]
    target_rows = target_statistics["objectives"]
    candidate_means = {
        role: float(target_rows[role]["candidate_occurrences"]["mean"])
        for role in _ROLES
    }
    recall = {role: float(overall[role]["recall@100"]) for role in _ROLES}
    breadth_monotonic = all(
        candidate_means[left] < candidate_means[right] and recall[left] > recall[right]
        for left, right in zip(_ROLES, _ROLES[1:])
    )
    activity_changes = {
        role: [
            _relative_change(
                evaluation["slices"][f"user_activity_q{quartile}"]["rows"][
                    "control_next_item"
                ]["recall@100"],
                evaluation["slices"][f"user_activity_q{quartile}"]["rows"][role][
                    "recall@100"
                ],
            )
            for quartile in range(1, 5)
        ]
        for role in _TREATMENTS
    }
    activity_monotonic = {
        role: all(left < right for left, right in zip(values, values[1:]))
        for role, values in activity_changes.items()
    }
    rq1_targets = target_rows["rq1_24h"]
    document = {
        "schema_version": 1,
        "kind": "g4_native500m_unexpected_result_diagnostics",
        "status": "awaiting_user_validation",
        "proposed_explanation": (
            "broader uniformly sampled future-positive sets dilute next-item "
            "alignment most strongly for less-active users"
        ),
        "breadth_dose_response": {
            "candidate_means": candidate_means,
            "recall_at_100": recall,
            "monotonic": breadth_monotonic,
        },
        "activity_moderation": {
            "recall_at_100_change_percent": activity_changes,
            "monotonic": activity_monotonic,
        },
        "rq1_target_availability": {
            "eligibility_rate": float(rq1_targets["eligibility_rate"]),
            "fallback_rate": float(rq1_targets["fallback_rate"]),
        },
    }
    if user_validation is not None:
        document["status"] = "conclusion_validated"
        document["mechanism_status"] = "tentative"
        document["user_validation"] = _validate_user_validation(user_validation)
    return document


def write_unexpected_result_diagnostics(
    artifact_path: Path,
    *,
    evaluation_path: Path,
    target_statistics_path: Path,
    user_validation_path: Path | None = None,
) -> str:
    evaluation = _document(evaluation_path)
    target_statistics = _document(target_statistics_path)
    user_validation = (
        _document(user_validation_path) if user_validation_path is not None else None
    )
    document = build_unexpected_result_diagnostics(
        evaluation,
        target_statistics,
        user_validation=user_validation,
    )
    document["sources"] = {
        "evaluation": _file_fact(evaluation_path),
        "target_statistics": _file_fact(target_statistics_path),
    }
    if user_validation_path is not None:
        document["sources"]["user_validation"] = _file_fact(user_validation_path)
    payload = canonical_bytes(document)
    digest = hashlib.sha256(payload).hexdigest()
    _write_immutable(artifact_path, payload)
    _write_immutable(artifact_path.with_suffix(".sha256"), digest.encode())
    return digest


def _validate_user_validation(value: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "schema_version": 1,
        "kind": "g4_native500m_rq1_rq2_conclusion_validation",
        "dataset_size": "native-500m",
        "status": "validated",
        "scope": "RQ1/RQ2 inferior; control retained",
    }
    if not isinstance(value, Mapping) or set(value) != {
        *expected,
        "message_sha256",
    }:
        raise ValueError("RQ1/RQ2 user validation schema differs")
    for field, expected_value in expected.items():
        if value[field] != expected_value:
            raise ValueError(f"RQ1/RQ2 user validation differs at {field}")
    message_hash = value["message_sha256"]
    if message_hash != _APPROVAL_MESSAGE_SHA256:
        raise ValueError("RQ1/RQ2 user validation message hash differs")
    return dict(value)


def _relative_change(reference: float, value: float) -> float:
    reference = float(reference)
    value = float(value)
    if reference <= 0:
        raise ValueError("unexpected-result slice reference must be positive")
    return 100 * (value / reference - 1)


def _document(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected an object: {path}")
    return value


def _file_fact(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
    }


def _write_immutable(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"native-500M unexpected evidence changed: {path}")
        return
    path.write_bytes(content)
