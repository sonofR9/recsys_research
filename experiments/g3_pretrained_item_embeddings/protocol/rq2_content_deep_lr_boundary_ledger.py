from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from experiments.g3_pretrained_item_embeddings.analysis.rq2_content_horizon_results import (
    RQ2_CONTENT_HORIZON_EVIDENCE_PATH,
)
from experiments.g3_pretrained_item_embeddings.analysis.rq2_id_boundary_results import (
    APPROVED_RQ2_ID_BOUNDARY_EVIDENCE_SHA256,
    RQ2_ID_BOUNDARY_EVIDENCE_PATH,
)
from experiments.g3_pretrained_item_embeddings.analysis.rq2_next_stage_results import (
    APPROVED_RQ2_NEXT_STAGE_EVIDENCE_SHA256,
    RQ2_NEXT_STAGE_EVIDENCE_PATH,
)

from .constants import APPROVED_PROTOCOL, APPROVED_PROTOCOL_SHA256
from .control_ledger import ManifestReference
from .rq2_capacity_ledger import (
    CONTENT_MANIFEST_PATH,
    CONTENT_MANIFEST_SHA256,
    FEATURE_MANIFEST_PATH,
    FEATURE_MANIFEST_SHA256,
    PREDECESSOR_CALIBRATION_PATH,
    PREDECESSOR_CALIBRATION_SHA256,
)
from .rq2_content_horizon_ledger import (
    APPROVED_RQ2_CONTENT_HORIZON_LEDGER_SHA256,
    RQ2_CONTENT_HORIZON_LEDGER_PATH,
    load_rq2_content_horizon_ledger,
)
from .rq2_id_boundary_ledger import (
    APPROVED_RQ2_ID_BOUNDARY_LEDGER_SHA256,
    RQ2_ID_BOUNDARY_LEDGER_PATH,
)
from .rq2_next_stage_ledger import (
    APPROVED_RQ2_NEXT_STAGE_LEDGER_SHA256,
    RQ2_NEXT_STAGE_LEDGER_PATH,
)


RQ2_CONTENT_DEEP_LR_BOUNDARY_LEDGER_PATH = (
    "experiments/g3_pretrained_item_embeddings/protocol/ledgers/"
    "rq2_content_width32_horizon40_deep_lr_boundary.json"
)
APPROVED_RQ2_CONTENT_HORIZON_EVIDENCE_SHA256 = (
    "673394896d8ca262e293032f148af7e005d9ce1402d81c47599b3fbe2cf8365b"
)
APPROVED_RQ2_CONTENT_DEEP_LR_BOUNDARY_LEDGER_SHA256 = (
    "955d877132d0828a47ffb38025c5e5e3acaa9180ff6b7f628bbd82b45b3d1769"
)


@dataclass(frozen=True)
class Rq2ContentDeepLrBoundaryJob:
    id: str
    run_name: str
    deep_learning_rate: float
    family_id: str = "rq2_content_concat"
    capacity: int = 32
    embedding_learning_rate: float = 0.3041556165944196
    horizon_epochs: int = 40
    batch_size: int = 512
    seed: int = 42
    reused_from: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "family_id": self.family_id,
            "phase": "deep_learning_rate_lower_boundary_extension",
            "run_name": self.run_name,
            "stage": "rq2_content_width32_horizon40_deep_lr_boundary",
            "role": "deep_learning_rate_boundary_probe",
            "reused_from": self.reused_from,
            "representation": {
                "id": self.family_id,
                "history": "learned_item_id_plus_frozen_content",
                "catalog": "learned_item_id",
                "history_hidden_dim": self.capacity,
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
                "batch_size": self.batch_size,
                "seed": self.seed,
                "embedding_learning_rate": self.embedding_learning_rate,
                "deep_learning_rate": self.deep_learning_rate,
                "horizon_epochs": self.horizon_epochs,
                "validate_every_epoch": True,
                "restore_best_validation_epoch": True,
            },
        }


@dataclass(frozen=True)
class Rq2ContentDeepLrBoundaryLedger:
    schema_version: int
    kind: str
    protocol_sha256: str
    maximum_jobs: int
    content_horizon_evidence: ManifestReference
    content_horizon_ledger: ManifestReference
    resolved_next_stage_evidence: ManifestReference
    resolved_next_stage_ledger: ManifestReference
    id_boundary_evidence: ManifestReference
    id_boundary_ledger: ManifestReference
    predecessor_calibration: ManifestReference
    content: ManifestReference
    features: ManifestReference
    source_selection: dict[str, object]
    boundary_decision: dict[str, object]
    opportunity_accounting: dict[str, int]
    rows: tuple[Rq2ContentDeepLrBoundaryJob, ...]

    @property
    def inputs(self) -> dict[str, ManifestReference]:
        return {
            "content_horizon_evidence": self.content_horizon_evidence,
            "content_horizon_ledger": self.content_horizon_ledger,
            "resolved_next_stage_evidence": self.resolved_next_stage_evidence,
            "resolved_next_stage_ledger": self.resolved_next_stage_ledger,
            "id_boundary_evidence": self.id_boundary_evidence,
            "id_boundary_ledger": self.id_boundary_ledger,
            "predecessor_calibration": self.predecessor_calibration,
            "content": self.content,
            "features": self.features,
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


def compile_rq2_content_deep_lr_boundary_ledger(
    root: Path,
) -> Rq2ContentDeepLrBoundaryLedger:
    root = root.resolve(strict=True)
    evidence, horizon_ledger = load_bound_rq2_content_horizon_ancestry(root)
    selected = _validate_boundary_decision(evidence)
    lower = APPROVED_PROTOCOL.deep_lr_bounds[0]
    if lower != 0.0081084848:
        raise ValueError("RQ2 approved deep learning-rate lower bound changed")
    divisors = (math.sqrt(2), 2.0, 2 * math.sqrt(2))
    rows = tuple(
        Rq2ContentDeepLrBoundaryJob(
            id=f"rq2_content_concat:{index + 16:02d}",
            run_name=(
                "g3_rq2_content_concat_width_32_horizon_40_deep_lr_lower_"
                f"probe_{index + 1:02d}_native50m"
            ),
            deep_learning_rate=lower / divisor,
        )
        for index, divisor in enumerate(divisors)
    )
    ledger = Rq2ContentDeepLrBoundaryLedger(
        schema_version=1,
        kind="g3_rq2_content_width32_horizon40_deep_lr_boundary",
        protocol_sha256=APPROVED_PROTOCOL_SHA256,
        maximum_jobs=3,
        content_horizon_evidence=ManifestReference(
            "g3_rq2_content_width32_horizon_evidence",
            RQ2_CONTENT_HORIZON_EVIDENCE_PATH,
            APPROVED_RQ2_CONTENT_HORIZON_EVIDENCE_SHA256,
        ),
        content_horizon_ledger=ManifestReference(
            "g3_rq2_content_width32_horizon",
            RQ2_CONTENT_HORIZON_LEDGER_PATH,
            APPROVED_RQ2_CONTENT_HORIZON_LEDGER_SHA256,
        ),
        resolved_next_stage_evidence=ManifestReference(
            "g3_rq2_resolved_next_stage_evidence",
            RQ2_NEXT_STAGE_EVIDENCE_PATH,
            APPROVED_RQ2_NEXT_STAGE_EVIDENCE_SHA256,
        ),
        resolved_next_stage_ledger=ManifestReference(
            "g3_rq2_resolved_next_stage",
            RQ2_NEXT_STAGE_LEDGER_PATH,
            APPROVED_RQ2_NEXT_STAGE_LEDGER_SHA256,
        ),
        id_boundary_evidence=ManifestReference(
            "g3_rq2_id_only_deep_lr_boundary_evidence",
            RQ2_ID_BOUNDARY_EVIDENCE_PATH,
            APPROVED_RQ2_ID_BOUNDARY_EVIDENCE_SHA256,
        ),
        id_boundary_ledger=ManifestReference(
            "g3_rq2_id_only_deep_lr_boundary",
            RQ2_ID_BOUNDARY_LEDGER_PATH,
            APPROVED_RQ2_ID_BOUNDARY_LEDGER_SHA256,
        ),
        predecessor_calibration=ManifestReference(
            "g3_untied_control_calibration",
            PREDECESSOR_CALIBRATION_PATH,
            PREDECESSOR_CALIBRATION_SHA256,
        ),
        content=ManifestReference(
            "native50m_content",
            CONTENT_MANIFEST_PATH,
            CONTENT_MANIFEST_SHA256,
        ),
        features=ManifestReference(
            "native50m_features",
            FEATURE_MANIFEST_PATH,
            FEATURE_MANIFEST_SHA256,
        ),
        source_selection={
            "row_id": "rq2_content_concat:12",
            "capacity": 32,
            "horizon_epochs": 40,
            "embedding_learning_rate": 0.3041556165944196,
            "deep_learning_rate": 0.014506684820055783,
            "recall_at_100": 0.08893693160875873,
            "ndcg_at_100": 0.03244652591410125,
            "best_epoch": 12,
            "approved_lower_bound": lower,
            "outward_divisors": list(divisors),
        },
        boundary_decision={
            "status": "approved_pending_lower_deep_lr_followup",
            "optimizer_group": "deep_learning_rate",
            "direction": "lower",
            "embedding_learning_rate_fixed": True,
            "capacity_fixed": True,
            "horizon_fixed": True,
            "additional_width_authorized": False,
        },
        opportunity_accounting={
            "prior_cumulative_maximum": 15,
            "conditional_deep_lr_extension": 3,
            "cumulative_maximum_after_extension": 18,
            "new_physical_jobs": 3,
        },
        rows=rows,
    )
    if (
        selected["embedding_learning_rate"]
        != ledger.source_selection["embedding_learning_rate"]
    ):
        raise ValueError("RQ2 content boundary embedding learning rate changed")
    if (
        APPROVED_RQ2_CONTENT_DEEP_LR_BOUNDARY_LEDGER_SHA256 != "0" * 64
        and ledger.sha256 != APPROVED_RQ2_CONTENT_DEEP_LR_BOUNDARY_LEDGER_SHA256
    ):
        raise ValueError("approved RQ2 content deep-LR ledger definition drifted")
    return ledger


def load_bound_rq2_content_horizon_ancestry(
    root: Path,
) -> tuple[dict[str, object], object]:
    root = root.resolve(strict=True)
    evidence = _load_horizon_evidence(root / RQ2_CONTENT_HORIZON_EVIDENCE_PATH)
    horizon_ledger = load_rq2_content_horizon_ledger(
        root / RQ2_CONTENT_HORIZON_LEDGER_PATH,
        root=root,
    )
    _validate_ancestry(root, evidence=evidence, horizon_ledger=horizon_ledger)
    return evidence, horizon_ledger


def _validate_boundary_decision(
    evidence: Mapping[str, object],
) -> Mapping[str, object]:
    selection = evidence.get("final_content_selection")
    if not isinstance(selection, dict):
        raise ValueError("RQ2 content deep-LR evidence lacks a final selection")
    selected = selection.get("provisional_selected")
    boundary = selection.get("boundary_decision")
    if not isinstance(selected, dict) or not isinstance(boundary, dict):
        raise ValueError("RQ2 content deep-LR evidence lacks a boundary decision")
    expected_action = [
        {
            "action": "three_joint_outward_lr_probes",
            "direction": "lower",
            "optimizer_group": "deep_learning_rate",
        }
    ]
    if (
        selection.get("status") != "pending_boundary_followup"
        or selection.get("selected") is not None
        or selected.get("row_id") != "rq2_content_concat:12"
        or selected.get("family_id") != "rq2_content_concat"
        or selected.get("capacity") != 32
        or selected.get("horizon_epochs") != 40
        or selected.get("embedding_learning_rate") != 0.3041556165944196
        or selected.get("deep_learning_rate") != 0.014506684820055783
        or selected.get("best_epoch") != 12
        or selected.get("metrics", {}).get("recall@100") != 0.08893693160875873
        or selected.get("metrics", {}).get("ndcg@100") != 0.03244652591410125
        or boundary.get("extension_required") is not True
        or boundary.get("required_actions") != expected_action
        or boundary.get("embedding_learning_rate", {}).get("direction") is not None
        or boundary.get("deep_learning_rate", {}).get("direction") != "lower"
        or boundary.get("horizon", {}).get("extension_required") is not False
        or boundary.get("capacity", {}).get("selected") != 32
        or boundary.get("capacity", {}).get("additional_lower_capacity_authorized")
        is not False
    ):
        raise ValueError("RQ2 content deep-LR boundary decision changed")
    return selected


def _validate_ancestry(
    root: Path,
    *,
    evidence: Mapping[str, object],
    horizon_ledger: object,
) -> None:
    if (
        evidence.get("schema_version") != 1
        or evidence.get("kind") != "g3_rq2_content_width32_horizon_evidence"
        or evidence.get("protocol_sha256") != APPROVED_PROTOCOL_SHA256
        or evidence.get("sha256") != APPROVED_RQ2_CONTENT_HORIZON_EVIDENCE_SHA256
        or horizon_ledger.sha256 != APPROVED_RQ2_CONTENT_HORIZON_LEDGER_SHA256
    ):
        raise ValueError("RQ2 content deep-LR approved horizon ancestry changed")
    _validate_file_fact(
        root,
        evidence.get("content_horizon_ledger"),
        path=RQ2_CONTENT_HORIZON_LEDGER_PATH,
        logical_sha256=APPROVED_RQ2_CONTENT_HORIZON_LEDGER_SHA256,
    )
    _validate_file_fact(
        root,
        evidence.get("resolved_next_stage_evidence"),
        path=RQ2_NEXT_STAGE_EVIDENCE_PATH,
        logical_sha256=APPROVED_RQ2_NEXT_STAGE_EVIDENCE_SHA256,
    )
    _validate_file_fact(
        root,
        evidence.get("id_boundary_evidence"),
        path=RQ2_ID_BOUNDARY_EVIDENCE_PATH,
        logical_sha256=APPROVED_RQ2_ID_BOUNDARY_EVIDENCE_SHA256,
    )
    expected_inputs = {
        "resolved_next_stage_evidence": APPROVED_RQ2_NEXT_STAGE_EVIDENCE_SHA256,
        "resolved_next_stage_ledger": APPROVED_RQ2_NEXT_STAGE_LEDGER_SHA256,
        "id_boundary_evidence": APPROVED_RQ2_ID_BOUNDARY_EVIDENCE_SHA256,
        "id_boundary_ledger": APPROVED_RQ2_ID_BOUNDARY_LEDGER_SHA256,
        "predecessor_calibration": PREDECESSOR_CALIBRATION_SHA256,
        "content": CONTENT_MANIFEST_SHA256,
        "features": FEATURE_MANIFEST_SHA256,
    }
    if any(
        horizon_ledger.inputs[name].sha256 != sha256
        for name, sha256 in expected_inputs.items()
    ):
        raise ValueError("RQ2 content deep-LR latest ancestry changed")


def _validate_file_fact(
    root: Path,
    value: object,
    *,
    path: str,
    logical_sha256: str,
) -> None:
    target = root / path
    if (
        not isinstance(value, dict)
        or value.get("path") != path
        or value.get("logical_sha256") != logical_sha256
        or value.get("size_bytes") != target.stat().st_size
        or value.get("sha256") != _file_sha256(target)
    ):
        raise ValueError("RQ2 content deep-LR predecessor file binding changed")


def validate_rq2_content_deep_lr_boundary_ledger_document(
    document: object, *, root: Path
) -> Rq2ContentDeepLrBoundaryLedger:
    if not isinstance(document, dict):
        raise ValueError("RQ2 content deep-LR ledger must be an object")
    expected = compile_rq2_content_deep_lr_boundary_ledger(root)
    if set(document) != set(expected.to_dict()):
        raise ValueError("RQ2 content deep-LR ledger keys do not match closed schema")
    _validate_exact_json_types(document, expected.to_dict(), path="ledger")
    payload = {name: value for name, value in document.items() if name != "sha256"}
    actual_sha256 = hashlib.sha256(_canonical_json(payload).encode()).hexdigest()
    if actual_sha256 != expected.sha256:
        raise ValueError(
            "RQ2 ledger no longer matches approved content deep-LR boundary"
        )
    if document.get("sha256") != actual_sha256:
        raise ValueError("RQ2 content deep-LR ledger hash differs from its payload")
    return expected


def load_rq2_content_deep_lr_boundary_ledger(
    path: Path, *, root: Path
) -> Rq2ContentDeepLrBoundaryLedger:
    document = _load_json(path)
    ledger = validate_rq2_content_deep_lr_boundary_ledger_document(document, root=root)
    if ledger.sha256 != APPROVED_RQ2_CONTENT_DEEP_LR_BOUNDARY_LEDGER_SHA256:
        raise ValueError("RQ2 content deep-LR boundary ledger is not approved")
    return ledger


def persist_rq2_content_deep_lr_boundary_ledger(
    path: Path,
    ledger: Rq2ContentDeepLrBoundaryLedger,
    *,
    root: Path,
) -> Path:
    validate_rq2_content_deep_lr_boundary_ledger_document(ledger.to_dict(), root=root)
    content = (_canonical_json(ledger.to_dict()) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(content)
    except FileExistsError:
        if path.read_bytes() != content:
            raise RuntimeError(f"immutable RQ2 content deep-LR ledger differs: {path}")
    return path


def _load_horizon_evidence(path: Path) -> dict[str, object]:
    document = _load_json(path)
    payload = {name: value for name, value in document.items() if name != "sha256"}
    if (
        document.get("sha256")
        != hashlib.sha256(_canonical_json(payload).encode()).hexdigest()
        or document.get("sha256") != APPROVED_RQ2_CONTENT_HORIZON_EVIDENCE_SHA256
    ):
        raise ValueError("RQ2 content-horizon evidence hash changed")
    return document


def _load_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(
            path.read_text(),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load RQ2 content deep-LR input {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"RQ2 content deep-LR input is not an object: {path}")
    return value


def _validate_exact_json_types(actual: object, expected: object, *, path: str) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            raise ValueError(f"RQ2 content deep-LR {path} has an invalid JSON type")
        for name, expected_value in expected.items():
            if name in actual:
                _validate_exact_json_types(
                    actual[name], expected_value, path=f"{path}.{name}"
                )
        return
    if isinstance(expected, list):
        if not isinstance(actual, list):
            raise ValueError(f"RQ2 content deep-LR {path} has an invalid JSON type")
        for index, (actual_value, expected_value) in enumerate(
            zip(actual, expected, strict=False)
        ):
            _validate_exact_json_types(
                actual_value, expected_value, path=f"{path}[{index}]"
            )
        return
    if type(actual) is not type(expected):
        raise ValueError(f"RQ2 content deep-LR {path} has an invalid JSON type")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    for name, value in pairs:
        if name in result:
            raise ValueError(f"duplicate JSON key {name!r}")
        result[name] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON number {value!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[3]
    )
    arguments = parser.parse_args()
    root = arguments.root.resolve(strict=True)
    ledger = compile_rq2_content_deep_lr_boundary_ledger(root)
    path = root / RQ2_CONTENT_DEEP_LR_BOUNDARY_LEDGER_PATH
    if arguments.write:
        persist_rq2_content_deep_lr_boundary_ledger(path, ledger, root=root)
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
