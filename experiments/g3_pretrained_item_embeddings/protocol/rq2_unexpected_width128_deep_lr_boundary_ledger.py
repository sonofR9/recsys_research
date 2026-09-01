from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from experiments.g3_pretrained_item_embeddings.analysis.rq2_unexpected_diagnostic_results import (
    RQ2_UNEXPECTED_DIAGNOSTIC_EVIDENCE_PATH,
)

from .constants import APPROVED_PROTOCOL, APPROVED_PROTOCOL_SHA256
from .rq2_unexpected_diagnostic_ledger import (
    APPROVED_RQ2_UNEXPECTED_DIAGNOSTIC_LEDGER_SHA256,
    EvidenceReference,
    RQ2_UNEXPECTED_DIAGNOSTIC_LEDGER_PATH,
    _validate_exact_json_types,
)


RQ2_UNEXPECTED_WIDTH128_BOUNDARY_LEDGER_PATH = (
    "experiments/g3_pretrained_item_embeddings/protocol/ledgers/"
    "rq2_unexpected_width128_horizon40_deep_lr_boundary.json"
)
APPROVED_RQ2_UNEXPECTED_DIAGNOSTIC_EVIDENCE_SHA256 = (
    "48f71f485df0e51ded8017fe19a31df8dba4c179cbcf43196674c6b0d268df50"
)
RQ2_UNEXPECTED_DIAGNOSTIC_EVIDENCE_FILE_SHA256 = (
    "3b2c86382860ce8fea10ef80ac5b0d888166c37dc5dfe3809daea7fcdef63b1a"
)
APPROVED_RQ2_UNEXPECTED_WIDTH128_BOUNDARY_LEDGER_SHA256 = (
    "7e966e7aaefa5c4d29fec6d110dbb24c64bf5e0a80f068adbc7c7812f4956dd8"
)


@dataclass(frozen=True)
class Rq2UnexpectedWidth128BoundaryJob:
    id: str
    run_name: str
    deep_learning_rate: float

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "family_id": "rq2_content_concat",
            "phase": "unexpected_width128_deep_learning_rate_lower_boundary",
            "run_name": self.run_name,
            "stage": "rq2_unexpected_width128_horizon40_deep_lr_boundary",
            "role": "deep_learning_rate_boundary_probe",
            "reused_from": None,
            "representation": {
                "id": "rq2_content_concat",
                "history": "learned_item_id_plus_frozen_content",
                "catalog": "learned_item_id",
                "history_hidden_dim": 128,
                "separate_history_catalog_tables": True,
                "content_trainable": False,
                "content_width": 128,
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
                "seed": APPROVED_PROTOCOL.seed,
                "embedding_learning_rate": 0.3041556165944196,
                "deep_learning_rate": self.deep_learning_rate,
                "horizon_epochs": 40,
                "validate_every_epoch": True,
                "restore_best_validation_epoch": True,
            },
        }


@dataclass(frozen=True)
class Rq2UnexpectedWidth128BoundaryLedger:
    schema_version: int
    kind: str
    protocol_sha256: str
    maximum_jobs: int
    diagnostic_evidence: EvidenceReference
    diagnostic_ledger: EvidenceReference
    source_selection: dict[str, object]
    boundary_decision: dict[str, object]
    opportunity_accounting: dict[str, int]
    rows: tuple[Rq2UnexpectedWidth128BoundaryJob, ...]

    @property
    def inputs(self) -> dict[str, EvidenceReference]:
        return {
            "diagnostic_evidence": self.diagnostic_evidence,
            "diagnostic_ledger": self.diagnostic_ledger,
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
            "inputs": {name: value.to_dict() for name, value in self.inputs.items()},
            "source_selection": self.source_selection,
            "boundary_decision": self.boundary_decision,
            "opportunity_accounting": self.opportunity_accounting,
            "rows": [row.to_dict() for row in self.rows],
        }

    def to_dict(self) -> dict[str, object]:
        return self._payload() | {"sha256": self.sha256}


def compile_rq2_unexpected_width128_boundary_ledger(
    root: Path,
) -> Rq2UnexpectedWidth128BoundaryLedger:
    root = root.resolve(strict=True)
    evidence_path = root / RQ2_UNEXPECTED_DIAGNOSTIC_EVIDENCE_PATH
    evidence_bytes = evidence_path.read_bytes()
    if (
        len(evidence_bytes) != 200_904
        or hashlib.sha256(evidence_bytes).hexdigest()
        != RQ2_UNEXPECTED_DIAGNOSTIC_EVIDENCE_FILE_SHA256
    ):
        raise ValueError("RQ2 width128 boundary diagnostic evidence file changed")
    evidence = _load_json(evidence_path)
    _validate_diagnostic_evidence(evidence)
    diagnostic_ledger_path = root / RQ2_UNEXPECTED_DIAGNOSTIC_LEDGER_PATH
    diagnostic_ledger_reference = _reference(
        root,
        diagnostic_ledger_path,
        logical_sha256=APPROVED_RQ2_UNEXPECTED_DIAGNOSTIC_LEDGER_SHA256,
    )
    rates = (0.005733564587228046, 0.0040542424, 0.002866782293614023)
    rows = tuple(
        Rq2UnexpectedWidth128BoundaryJob(
            id=f"rq2_content_concat:{index:02d}",
            run_name=(
                "g3_rq2_content_concat_width_128_horizon_40_deep_lr_lower_"
                f"probe_{index - 18:02d}_native50m"
            ),
            deep_learning_rate=rate,
        )
        for index, rate in zip(range(19, 22), rates, strict=True)
    )
    ledger = Rq2UnexpectedWidth128BoundaryLedger(
        schema_version=1,
        kind="g3_rq2_unexpected_width128_horizon40_deep_lr_boundary",
        protocol_sha256=APPROVED_PROTOCOL_SHA256,
        maximum_jobs=3,
        diagnostic_evidence=EvidenceReference(
            RQ2_UNEXPECTED_DIAGNOSTIC_EVIDENCE_PATH,
            200_904,
            RQ2_UNEXPECTED_DIAGNOSTIC_EVIDENCE_FILE_SHA256,
            APPROVED_RQ2_UNEXPECTED_DIAGNOSTIC_EVIDENCE_SHA256,
        ),
        diagnostic_ledger=diagnostic_ledger_reference,
        source_selection={
            "row_id": "rq2_unexpected_diagnostic:03",
            "capacity": 128,
            "horizon_epochs": 40,
            "embedding_learning_rate": 0.3041556165944196,
            "deep_learning_rate": 0.014506684820055783,
            "recall_at_100": 0.09338954031207222,
            "ndcg_at_100": 0.03302666460800764,
            "best_epoch": 19,
        },
        boundary_decision={
            "status": "approved_pending_lower_deep_learning_rate_boundary",
            "selected_is_smallest_tested_width_128_deep_lr": True,
            "direction": "lower",
            "embedding_learning_rate_fixed": True,
            "capacity_fixed": True,
            "horizon_fixed": True,
            "additional_width_authorized": False,
            "crossed_factorial_authorized": False,
        },
        opportunity_accounting={
            "prior_tuning_and_diagnostic_rows": 33,
            "conditional_boundary_jobs": 3,
            "cumulative_rows_after_extension": 36,
            "new_physical_jobs": 3,
        },
        rows=rows,
    )
    if (
        APPROVED_RQ2_UNEXPECTED_WIDTH128_BOUNDARY_LEDGER_SHA256 != "0" * 64
        and ledger.sha256 != APPROVED_RQ2_UNEXPECTED_WIDTH128_BOUNDARY_LEDGER_SHA256
    ):
        raise ValueError("approved RQ2 width128 boundary ledger drifted")
    return ledger


def _validate_diagnostic_evidence(evidence: dict[str, object]) -> None:
    continuation = evidence.get("continuation")
    selection = evidence.get("provisional_selection")
    selected = selection.get("provisional_selected") if isinstance(selection, dict) else None
    boundary = selection.get("boundary_decision") if isinstance(selection, dict) else None
    payload = {name: value for name, value in evidence.items() if name != "sha256"}
    if (
        evidence.get("sha256") != APPROVED_RQ2_UNEXPECTED_DIAGNOSTIC_EVIDENCE_SHA256
        or evidence.get("sha256") != hashlib.sha256(_canonical_json(payload).encode()).hexdigest()
        or not isinstance(continuation, dict)
        or continuation.get("maximum_jobs") != 3
        or continuation.get("capacity") != 128
        or continuation.get("horizon_epochs") != 40
        or continuation.get("embedding_learning_rate") != 0.3041556165944196
        or continuation.get("deep_learning_rates")
        != [0.005733564587228046, 0.0040542424, 0.002866782293614023]
        or not isinstance(selected, dict)
        or selected.get("row_id") != "rq2_unexpected_diagnostic:03"
        or not isinstance(boundary, dict)
        or boundary.get("selected_is_smallest_tested_width_128_deep_lr") is not True
        or boundary.get("required_actions")
        != ["three_width128_horizon40_lower_deep_learning_rate_probes"]
    ):
        raise ValueError("RQ2 width128 boundary diagnostic evidence changed")


def validate_rq2_unexpected_width128_boundary_ledger_document(
    document: object, *, root: Path
) -> Rq2UnexpectedWidth128BoundaryLedger:
    if not isinstance(document, dict):
        raise ValueError("RQ2 width128 boundary ledger must be an object")
    expected = compile_rq2_unexpected_width128_boundary_ledger(root)
    expected_document = expected.to_dict()
    _validate_exact_json_types(document, expected_document, path="ledger")
    if document != expected_document:
        raise ValueError("RQ2 width128 boundary ledger differs from approval")
    return expected


def load_rq2_unexpected_width128_boundary_ledger(
    path: Path, *, root: Path
) -> Rq2UnexpectedWidth128BoundaryLedger:
    ledger = validate_rq2_unexpected_width128_boundary_ledger_document(
        _load_json(path), root=root
    )
    if ledger.sha256 != APPROVED_RQ2_UNEXPECTED_WIDTH128_BOUNDARY_LEDGER_SHA256:
        raise ValueError("RQ2 width128 boundary ledger is not approved")
    return ledger


def persist_rq2_unexpected_width128_boundary_ledger(
    path: Path, ledger: Rq2UnexpectedWidth128BoundaryLedger, *, root: Path
) -> Path:
    validate_rq2_unexpected_width128_boundary_ledger_document(
        ledger.to_dict(), root=root
    )
    content = (_canonical_json(ledger.to_dict()) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(content)
    except FileExistsError:
        if path.read_bytes() != content:
            raise RuntimeError(f"immutable RQ2 width128 boundary ledger differs: {path}")
    return path


def _reference(root: Path, path: Path, *, logical_sha256: str) -> EvidenceReference:
    relative = str(path.relative_to(root))
    return EvidenceReference(
        relative,
        path.stat().st_size,
        hashlib.sha256(path.read_bytes()).hexdigest(),
        logical_sha256,
    )


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    arguments = parser.parse_args()
    root = arguments.root.resolve(strict=True)
    ledger = compile_rq2_unexpected_width128_boundary_ledger(root)
    path = root / RQ2_UNEXPECTED_WIDTH128_BOUNDARY_LEDGER_PATH
    if arguments.write:
        persist_rq2_unexpected_width128_boundary_ledger(path, ledger, root=root)
    print(json.dumps({
        "path": str(path),
        "sha256": ledger.sha256,
        "jobs": len(ledger.rows),
        "status": "materialized" if arguments.write else "preview",
    }, sort_keys=True))


if __name__ == "__main__":
    main()
