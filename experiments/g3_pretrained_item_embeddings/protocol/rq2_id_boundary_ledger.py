from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from experiments.g3_pretrained_item_embeddings.analysis.rq2_next_stage_results import (
    APPROVED_RQ2_NEXT_STAGE_EVIDENCE_SHA256,
    RQ2_NEXT_STAGE_EVIDENCE_PATH,
    load_rq2_next_stage_evidence,
)

from .constants import APPROVED_PROTOCOL, APPROVED_PROTOCOL_SHA256
from .control_ledger import ManifestReference
from .rq2_next_stage_ledger import (
    APPROVED_RQ2_NEXT_STAGE_LEDGER_SHA256,
    RQ2_NEXT_STAGE_LEDGER_PATH,
)


RQ2_ID_BOUNDARY_LEDGER_PATH = (
    "experiments/g3_pretrained_item_embeddings/protocol/ledgers/"
    "rq2_id_only_deep_lr_boundary.json"
)
APPROVED_RQ2_ID_BOUNDARY_LEDGER_SHA256 = (
    "42912b228d1fce066f50863163e3eb5f5f91230ffa843fb076a098ac3ac3b1a3"
)


@dataclass(frozen=True)
class Rq2IdBoundaryJob:
    id: str
    run_name: str
    deep_learning_rate: float
    family_id: str = "rq2_id_only_densenet"
    capacity: int = 255
    embedding_learning_rate: float = 0.3041556165944196
    horizon_epochs: int = 40
    seed: int = 42

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "family_id": self.family_id,
            "phase": "deep_learning_rate_lower_boundary_extension",
            "run_name": self.run_name,
            "stage": "rq2_id_only_deep_lr_boundary",
            "role": "deep_learning_rate_boundary_probe",
            "reused_from": None,
            "representation": {
                "id": self.family_id,
                "history": "learned_item_id_densenet",
                "catalog": "learned_item_id",
                "history_hidden_dim": self.capacity,
                "separate_history_catalog_tables": True,
                "content_trainable": False,
                "content_width": None,
            },
            "dataset": {
                "size": APPROVED_PROTOCOL.main_dataset_size,
                "source": "likes",
                "event_limit": 50_000_000,
                "sampling": "none",
                "minimum_user_interactions": 5,
                "validation_interval_seconds": 604800,
                "candidate_catalog": "full",
                "exclude_seen": False,
            },
            "training": {
                "batch_size": APPROVED_PROTOCOL.batch_size,
                "seed": self.seed,
                "embedding_learning_rate": self.embedding_learning_rate,
                "deep_learning_rate": self.deep_learning_rate,
                "horizon_epochs": self.horizon_epochs,
                "validate_every_epoch": True,
                "restore_best_validation_epoch": True,
            },
        }


@dataclass(frozen=True)
class Rq2IdBoundaryLedger:
    schema_version: int
    kind: str
    protocol_sha256: str
    maximum_jobs: int
    next_stage_evidence: ManifestReference
    next_stage_ledger: ManifestReference
    source_selection: dict[str, object]
    content_capacity_status: dict[str, object]
    opportunity_accounting: dict[str, dict[str, int]]
    rows: tuple[Rq2IdBoundaryJob, ...]

    @property
    def inputs(self) -> dict[str, ManifestReference]:
        return {
            "next_stage_evidence": self.next_stage_evidence,
            "next_stage_ledger": self.next_stage_ledger,
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self._payload()).encode()).hexdigest()

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "protocol_sha256": self.protocol_sha256,
            "maximum_jobs": self.maximum_jobs,
            "inputs": {key: value.to_dict() for key, value in self.inputs.items()},
            "source_selection": self.source_selection,
            "content_capacity_status": self.content_capacity_status,
            "opportunity_accounting": self.opportunity_accounting,
            "rows": [row.to_dict() for row in self.rows],
        }

    def to_dict(self) -> dict[str, object]:
        return self._payload() | {"sha256": self.sha256}


def initial_rq2_id_boundary_ledger(
    *, evidence: Mapping[str, object]
) -> Rq2IdBoundaryLedger:
    if evidence.get("sha256") != APPROVED_RQ2_NEXT_STAGE_EVIDENCE_SHA256:
        raise ValueError("RQ2 ID boundary requires the approved next-stage evidence")
    selection = evidence.get("id_only_selection")
    content = evidence.get("content_capacity_decision")
    accounting = evidence.get("opportunity_accounting")
    if not isinstance(selection, dict) or not isinstance(content, dict):
        raise ValueError("RQ2 ID boundary evidence lacks resolved decisions")
    selected = selection.get("selected")
    boundary = selection.get("boundary_decision")
    if not isinstance(selected, dict) or not isinstance(boundary, dict):
        raise ValueError("RQ2 ID boundary evidence lacks a resolved selection")
    if (
        selected.get("row_id") != "rq2_id_only_densenet:12"
        or selected.get("capacity") != 255
        or selected.get("horizon_epochs") != 40
        or selected.get("embedding_learning_rate") != 0.3041556165944196
        or selected.get("deep_learning_rate") != 0.014506684820055783
        or selected.get("best_epoch") != 29
        or selected.get("metrics", {}).get("recall@100")
        != 0.09074562121371973
        or boundary.get("embedding_learning_rate", {}).get("direction") is not None
        or boundary.get("deep_learning_rate", {}).get("direction") != "lower"
        or boundary.get("horizon", {}).get("extension_required") is not False
    ):
        raise ValueError("RQ2 ID boundary source selection drifted")
    if (
        content.get("status") != "deferred_pending_user_approval"
        or content.get("selection_changed") is not True
        or content.get("extension_selected", {}).get("row_id")
        != "rq2_content_concat:14"
    ):
        raise ValueError("RQ2 content decision is not safely deferred")
    expected_base = {
        "base_preselection": 9,
        "base_horizon_followup": 3,
        "approved_base_total": 12,
    }
    if not isinstance(accounting, dict) or any(
        accounting.get(family, {}).get(key) != value
        for family in ("rq2_content_concat", "rq2_id_only_densenet")
        for key, value in expected_base.items()
    ):
        raise ValueError("RQ2 opportunity accounting drifted")
    lower = APPROVED_PROTOCOL.deep_lr_bounds[0]
    divisors = (math.sqrt(2), 2.0, 2 * math.sqrt(2))
    rows = tuple(
        Rq2IdBoundaryJob(
            id=f"rq2_id_only_densenet:{index + 13:02d}",
            run_name=(
                "g3_rq2_id_only_densenet_width_255_horizon_40_deep_lr_lower_"
                f"probe_{index + 1:02d}_native50m"
            ),
            deep_learning_rate=lower / divisor,
        )
        for index, divisor in enumerate(divisors)
    )
    ledger = Rq2IdBoundaryLedger(
        schema_version=1,
        kind="g3_rq2_id_only_deep_lr_boundary",
        protocol_sha256=APPROVED_PROTOCOL_SHA256,
        maximum_jobs=3,
        next_stage_evidence=ManifestReference(
            "g3_rq2_resolved_next_stage_evidence",
            RQ2_NEXT_STAGE_EVIDENCE_PATH,
            APPROVED_RQ2_NEXT_STAGE_EVIDENCE_SHA256,
        ),
        next_stage_ledger=ManifestReference(
            "g3_rq2_resolved_next_stage",
            RQ2_NEXT_STAGE_LEDGER_PATH,
            APPROVED_RQ2_NEXT_STAGE_LEDGER_SHA256,
        ),
        source_selection={
            "row_id": "rq2_id_only_densenet:12",
            "capacity": 255,
            "horizon_epochs": 40,
            "embedding_learning_rate": 0.3041556165944196,
            "deep_learning_rate": 0.014506684820055783,
            "recall_at_100": 0.09074562121371973,
            "best_epoch": 29,
            "approved_lower_bound": lower,
            "outward_divisors": list(divisors),
        },
        content_capacity_status={
            "status": "deferred_pending_user_approval",
            "changed_by_this_ledger": False,
        },
        opportunity_accounting={
            "rq2_content_concat": {
                "approved_base_total": 12,
                "conditional_capacity_extension": 3,
                "cumulative_maximum_after_extension": 15,
            },
            "rq2_id_only_densenet": {
                "approved_base_total": 12,
                "conditional_deep_lr_extension": 3,
                "cumulative_maximum_after_extension": 15,
            },
        },
        rows=rows,
    )
    if (
        APPROVED_RQ2_ID_BOUNDARY_LEDGER_SHA256 != "0" * 64
        and ledger.sha256 != APPROVED_RQ2_ID_BOUNDARY_LEDGER_SHA256
    ):
        raise ValueError("approved RQ2 ID boundary ledger definition drifted")
    return ledger


def validate_rq2_id_boundary_ledger_document(
    document: object,
) -> Rq2IdBoundaryLedger:
    if not isinstance(document, dict):
        raise ValueError("RQ2 ID boundary ledger must be an object")
    expected_keys = {
        "schema_version",
        "kind",
        "protocol_sha256",
        "maximum_jobs",
        "inputs",
        "source_selection",
        "content_capacity_status",
        "opportunity_accounting",
        "rows",
        "sha256",
    }
    if set(document) != expected_keys:
        raise ValueError("RQ2 ID boundary ledger keys do not match the closed schema")
    evidence_path = Path(__file__).resolve().parents[3] / RQ2_NEXT_STAGE_EVIDENCE_PATH
    expected = initial_rq2_id_boundary_ledger(
        evidence=load_rq2_next_stage_evidence(evidence_path)
    )
    _validate_exact_json_types(document, expected.to_dict(), path="ledger")
    payload = {key: value for key, value in document.items() if key != "sha256"}
    actual_sha256 = hashlib.sha256(_canonical_json(payload).encode()).hexdigest()
    if actual_sha256 != expected.sha256:
        raise ValueError("RQ2 ledger no longer matches approved ID-only boundary coordinates")
    if document["sha256"] != actual_sha256:
        raise ValueError("RQ2 ID boundary ledger hash differs from its payload")
    return expected


def load_rq2_id_boundary_ledger(path: Path) -> Rq2IdBoundaryLedger:
    try:
        document = json.loads(
            path.read_text(),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load RQ2 ID boundary ledger {path}") from error
    ledger = validate_rq2_id_boundary_ledger_document(document)
    if ledger.sha256 != APPROVED_RQ2_ID_BOUNDARY_LEDGER_SHA256:
        raise ValueError("RQ2 ID boundary ledger is not approved")
    return ledger


def persist_rq2_id_boundary_ledger(
    path: Path, ledger: Rq2IdBoundaryLedger
) -> Path:
    validate_rq2_id_boundary_ledger_document(ledger.to_dict())
    content = (_canonical_json(ledger.to_dict()) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(content)
    except FileExistsError:
        if path.read_bytes() != content:
            raise RuntimeError(f"immutable RQ2 ID boundary ledger differs: {path}")
    return path


def _validate_exact_json_types(actual: object, expected: object, *, path: str) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            raise ValueError(f"RQ2 ID boundary {path} has an invalid JSON type")
        for key, expected_value in expected.items():
            if key in actual:
                _validate_exact_json_types(
                    actual[key], expected_value, path=f"{path}.{key}"
                )
        return
    if isinstance(expected, list):
        if not isinstance(actual, list):
            raise ValueError(f"RQ2 ID boundary {path} has an invalid JSON type")
        for index, (actual_value, expected_value) in enumerate(
            zip(actual, expected, strict=False)
        ):
            _validate_exact_json_types(
                actual_value, expected_value, path=f"{path}[{index}]"
            )
        return
    if type(actual) is not type(expected):
        raise ValueError(f"RQ2 ID boundary {path} has an invalid JSON type")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON number {value!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    arguments = parser.parse_args()
    root = arguments.root.resolve()
    ledger = initial_rq2_id_boundary_ledger(
        evidence=load_rq2_next_stage_evidence(root / RQ2_NEXT_STAGE_EVIDENCE_PATH)
    )
    path = root / RQ2_ID_BOUNDARY_LEDGER_PATH
    if arguments.write:
        persist_rq2_id_boundary_ledger(path, ledger)
    print(
        json.dumps(
            {
                "path": str(path),
                "sha256": ledger.sha256,
                "jobs": len(ledger.rows),
                "status": "materialized" if arguments.write else "preview",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
