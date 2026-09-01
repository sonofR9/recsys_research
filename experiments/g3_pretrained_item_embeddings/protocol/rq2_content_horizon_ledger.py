from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from experiments.g3_pretrained_item_embeddings.analysis.control_calibration import (
    load_control_calibration,
)
from experiments.g3_pretrained_item_embeddings.analysis.rq2_id_boundary_results import (
    APPROVED_RQ2_ID_BOUNDARY_EVIDENCE_SHA256,
    RQ2_ID_BOUNDARY_EVIDENCE_PATH,
    load_rq2_id_boundary_evidence,
)
from experiments.g3_pretrained_item_embeddings.analysis.rq2_next_stage_results import (
    APPROVED_RQ2_NEXT_STAGE_EVIDENCE_SHA256,
    RQ2_NEXT_STAGE_EVIDENCE_PATH,
    load_rq2_next_stage_evidence,
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
from .rq2_id_boundary_ledger import (
    APPROVED_RQ2_ID_BOUNDARY_LEDGER_SHA256,
    RQ2_ID_BOUNDARY_LEDGER_PATH,
    load_rq2_id_boundary_ledger,
)
from .rq2_next_stage_ledger import (
    APPROVED_RQ2_NEXT_STAGE_LEDGER_SHA256,
    RQ2_NEXT_STAGE_LEDGER_PATH,
    load_rq2_next_stage_ledger,
)


RQ2_CONTENT_HORIZON_LEDGER_PATH = (
    "experiments/g3_pretrained_item_embeddings/protocol/ledgers/"
    "rq2_content_width32_horizon.json"
)
APPROVED_RQ2_CONTENT_HORIZON_LEDGER_SHA256 = (
    "1256ba6c279a9996e8a229fa46c59f34d656782dbfcc5ea1ecfb7752cd0c78cb"
)

_EXPECTED_SELECTED_METRICS = {
    "capped_recall@10": 0.01650637896244223,
    "capped_recall@100": 0.07918962182678012,
    "capped_recall@50": 0.04806416492998638,
    "coverage@10": 0.21738868106673104,
    "coverage@100": 0.6993182092433933,
    "coverage@50": 0.5264872692168456,
    "mrr@10": 0.020631549827507645,
    "mrr@100": 0.02637553971528536,
    "mrr@50": 0.025300673056365193,
    "ndcg@10": 0.012219642351512836,
    "ndcg@100": 0.029330978863554256,
    "ndcg@50": 0.02181192912897356,
    "num_users": 3414.0,
    "recall@10": 0.015142520857295744,
    "recall@100": 0.07914654839561677,
    "recall@50": 0.04791037690954188,
}


@dataclass(frozen=True)
class Rq2ContentHorizonJob:
    id: str
    run_name: str
    horizon_epochs: int
    embedding_learning_rate: float
    deep_learning_rate: float
    family_id: str = "rq2_content_concat"
    capacity: int = 32
    batch_size: int = 512
    seed: int = 42
    reused_from: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "family_id": self.family_id,
            "phase": "selected_width_horizon_followup",
            "run_name": self.run_name,
            "stage": "rq2_content_width32_horizon",
            "role": "horizon_probe",
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
class Rq2ContentHorizonLedger:
    schema_version: int
    kind: str
    protocol_sha256: str
    maximum_opportunities: int
    maximum_physical_jobs: int
    resolved_next_stage_evidence: ManifestReference
    resolved_next_stage_ledger: ManifestReference
    id_boundary_evidence: ManifestReference
    id_boundary_ledger: ManifestReference
    predecessor_calibration: ManifestReference
    content: ManifestReference
    features: ManifestReference
    source_selection: dict[str, object]
    content_capacity_decision: dict[str, object]
    opportunity_accounting: dict[str, int]
    rows: tuple[Rq2ContentHorizonJob, ...]

    @property
    def inputs(self) -> dict[str, ManifestReference]:
        return {
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
            "maximum_opportunities": self.maximum_opportunities,
            "maximum_physical_jobs": self.maximum_physical_jobs,
            "inputs": {name: value.to_dict() for name, value in self.inputs.items()},
            "source_selection": self.source_selection,
            "content_capacity_decision": self.content_capacity_decision,
            "opportunity_accounting": self.opportunity_accounting,
            "rows": [row.to_dict() for row in self.rows],
        }

    def to_dict(self) -> dict[str, object]:
        return self._payload() | {"sha256": self.sha256}


def compile_rq2_content_horizon_ledger(root: Path) -> Rq2ContentHorizonLedger:
    root = root.resolve(strict=True)
    next_evidence = load_rq2_next_stage_evidence(root / RQ2_NEXT_STAGE_EVIDENCE_PATH)
    boundary_evidence = load_rq2_id_boundary_evidence(
        root / RQ2_ID_BOUNDARY_EVIDENCE_PATH
    )
    next_ledger = load_rq2_next_stage_ledger(root / RQ2_NEXT_STAGE_LEDGER_PATH)
    boundary_ledger = load_rq2_id_boundary_ledger(root / RQ2_ID_BOUNDARY_LEDGER_PATH)
    calibration = load_control_calibration(root / PREDECESSOR_CALIBRATION_PATH)
    _validate_ancestry(
        root,
        next_evidence=next_evidence,
        boundary_evidence=boundary_evidence,
        next_ledger=next_ledger,
        boundary_ledger=boundary_ledger,
        calibration=calibration,
    )
    selected = next_evidence.get("content_capacity_decision", {}).get(
        "extension_selected"
    )
    _validate_selected_width_32(selected)
    fitted = calibration["power_law_fits"]
    embedding = fitted["embedding_learning_rate"]["fitted_coordinates"]
    deep = fitted["deep_learning_rate"]["fitted_coordinates"]
    rows = tuple(
        Rq2ContentHorizonJob(
            id=f"rq2_content_concat:{index + 10:02d}",
            run_name=(f"g3_rq2_content_concat_width_32_horizon_{horizon}_native50m"),
            horizon_epochs=horizon,
            embedding_learning_rate=float(embedding[str(horizon)]),
            deep_learning_rate=float(deep[str(horizon)]),
        )
        for index, horizon in enumerate((15, 25, 40))
    )
    expected_rates = (
        (15, 0.047134737607146836, 0.04127129308065626),
        (25, 0.12447135415265811, 0.023941907610393703),
        (40, 0.3041556165944196, 0.014506684820055783),
    )
    if (
        tuple(
            (
                row.horizon_epochs,
                row.embedding_learning_rate,
                row.deep_learning_rate,
            )
            for row in rows
        )
        != expected_rates
    ):
        raise ValueError("RQ2 content-horizon transferred rates changed")
    ledger = Rq2ContentHorizonLedger(
        schema_version=1,
        kind="g3_rq2_content_width32_horizon",
        protocol_sha256=APPROVED_PROTOCOL_SHA256,
        maximum_opportunities=3,
        maximum_physical_jobs=3,
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
            "row_id": "rq2_content_concat:14",
            "family_id": "rq2_content_concat",
            "capacity": 32,
            "horizon_epochs": 25,
            "embedding_learning_rate": 0.1474458978470563,
            "deep_learning_rate": 0.032433939334700325,
            "epochs_trained": 25,
            "best_epoch": 24,
            "diagnostic_nonfinite_count": 0,
            "metrics": _EXPECTED_SELECTED_METRICS,
        },
        content_capacity_decision={
            "status": "resolved_user_approved",
            "approved_on": "2026-08-29",
            "selected_capacity": 32,
            "next_lower_capacity_authorized": False,
        },
        opportunity_accounting={
            "capacity_search": 9,
            "horizon_followup": 3,
            "approved_base_total": 12,
            "conditional_capacity_extension": 3,
            "cumulative_maximum": 15,
            "new_physical_jobs": 3,
        },
        rows=rows,
    )
    if (
        APPROVED_RQ2_CONTENT_HORIZON_LEDGER_SHA256 != "0" * 64
        and ledger.sha256 != APPROVED_RQ2_CONTENT_HORIZON_LEDGER_SHA256
    ):
        raise ValueError("approved RQ2 content-horizon ledger definition drifted")
    return ledger


def _validate_ancestry(
    root: Path,
    *,
    next_evidence: dict[str, object],
    boundary_evidence: dict[str, object],
    next_ledger: object,
    boundary_ledger: object,
    calibration: dict[str, object],
) -> None:
    if (
        next_evidence.get("sha256") != APPROVED_RQ2_NEXT_STAGE_EVIDENCE_SHA256
        or boundary_evidence.get("sha256") != APPROVED_RQ2_ID_BOUNDARY_EVIDENCE_SHA256
        or next_ledger.sha256 != APPROVED_RQ2_NEXT_STAGE_LEDGER_SHA256
        or boundary_ledger.sha256 != APPROVED_RQ2_ID_BOUNDARY_LEDGER_SHA256
        or calibration.get("sha256") != PREDECESSOR_CALIBRATION_SHA256
        or next_evidence.get("protocol_sha256") != APPROVED_PROTOCOL_SHA256
        or boundary_evidence.get("protocol_sha256") != APPROVED_PROTOCOL_SHA256
        or calibration.get("protocol_sha256") != APPROVED_PROTOCOL_SHA256
    ):
        raise ValueError("RQ2 content-horizon approved ancestry changed")
    _validate_bound_file(
        root,
        next_evidence.get("next_stage_ledger"),
        path=RQ2_NEXT_STAGE_LEDGER_PATH,
        logical_sha256=APPROVED_RQ2_NEXT_STAGE_LEDGER_SHA256,
    )
    _validate_bound_file(
        root,
        boundary_evidence.get("id_boundary_ledger"),
        path=RQ2_ID_BOUNDARY_LEDGER_PATH,
        logical_sha256=APPROVED_RQ2_ID_BOUNDARY_LEDGER_SHA256,
    )
    _validate_bound_file(
        root,
        boundary_evidence.get("predecessor_evidence"),
        path=RQ2_NEXT_STAGE_EVIDENCE_PATH,
        logical_sha256=APPROVED_RQ2_NEXT_STAGE_EVIDENCE_SHA256,
    )
    if (
        boundary_ledger.next_stage_evidence.sha256
        != APPROVED_RQ2_NEXT_STAGE_EVIDENCE_SHA256
        or boundary_ledger.next_stage_ledger.sha256
        != APPROVED_RQ2_NEXT_STAGE_LEDGER_SHA256
        or next_ledger.predecessor_calibration.sha256 != PREDECESSOR_CALIBRATION_SHA256
        or next_ledger.preselection_evidence.sha256
        != next_evidence["preselection_evidence"]["sha256"]
        or boundary_evidence.get("content_capacity_status")
        != {
            "status": "deferred_pending_user_approval",
            "changed_by_this_evidence": False,
        }
        or calibration.get("transfer_decision", {}).get("accepted") is not True
        or calibration.get("held_out_check", {}).get("accepted") is not True
    ):
        raise ValueError("RQ2 content-horizon predecessor chain changed")


def _validate_bound_file(
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
        raise ValueError("RQ2 content-horizon predecessor file binding changed")


def _validate_selected_width_32(value: object) -> None:
    if not isinstance(value, dict):
        raise ValueError("RQ2 content-horizon width-32 selection is missing")
    expected = {
        "row_id": "rq2_content_concat:14",
        "family_id": "rq2_content_concat",
        "capacity": 32,
        "horizon_epochs": 25,
        "embedding_learning_rate": 0.1474458978470563,
        "deep_learning_rate": 0.032433939334700325,
        "epochs_trained": 25,
        "best_epoch": 24,
        "diagnostic_nonfinite_count": 0,
    }
    if any(
        value.get(name) != expected_value for name, expected_value in expected.items()
    ):
        raise ValueError("RQ2 content-horizon selected width-32 row changed")
    if value.get("metrics") != _EXPECTED_SELECTED_METRICS:
        raise ValueError("RQ2 content-horizon selected width-32 metrics changed")


def validate_rq2_content_horizon_ledger_document(
    document: object, *, root: Path
) -> Rq2ContentHorizonLedger:
    if not isinstance(document, dict):
        raise ValueError("RQ2 content-horizon ledger must be an object")
    expected = compile_rq2_content_horizon_ledger(root)
    if set(document) != set(expected.to_dict()):
        raise ValueError(
            "RQ2 content-horizon ledger keys do not match the closed schema"
        )
    _validate_exact_json_types(document, expected.to_dict(), path="ledger")
    payload = {name: value for name, value in document.items() if name != "sha256"}
    actual_sha256 = hashlib.sha256(_canonical_json(payload).encode()).hexdigest()
    if actual_sha256 != expected.sha256:
        raise ValueError(
            "RQ2 ledger no longer matches approved content-horizon coordinates"
        )
    if document.get("sha256") != actual_sha256:
        raise ValueError("RQ2 content-horizon ledger hash differs from its payload")
    return expected


def load_rq2_content_horizon_ledger(
    path: Path, *, root: Path
) -> Rq2ContentHorizonLedger:
    try:
        document = json.loads(
            path.read_text(),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load RQ2 content-horizon ledger {path}") from error
    ledger = validate_rq2_content_horizon_ledger_document(document, root=root)
    if ledger.sha256 != APPROVED_RQ2_CONTENT_HORIZON_LEDGER_SHA256:
        raise ValueError("RQ2 content-horizon ledger is not approved")
    return ledger


def persist_rq2_content_horizon_ledger(
    path: Path, ledger: Rq2ContentHorizonLedger, *, root: Path
) -> Path:
    validate_rq2_content_horizon_ledger_document(ledger.to_dict(), root=root)
    content = (_canonical_json(ledger.to_dict()) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(content)
    except FileExistsError:
        if path.read_bytes() != content:
            raise RuntimeError(f"immutable RQ2 content-horizon ledger differs: {path}")
    return path


def _validate_exact_json_types(actual: object, expected: object, *, path: str) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            raise ValueError(f"RQ2 content-horizon {path} has an invalid JSON type")
        for name, expected_value in expected.items():
            if name in actual:
                _validate_exact_json_types(
                    actual[name], expected_value, path=f"{path}.{name}"
                )
        return
    if isinstance(expected, list):
        if not isinstance(actual, list):
            raise ValueError(f"RQ2 content-horizon {path} has an invalid JSON type")
        for index, (actual_value, expected_value) in enumerate(
            zip(actual, expected, strict=False)
        ):
            _validate_exact_json_types(
                actual_value, expected_value, path=f"{path}[{index}]"
            )
        return
    if type(actual) is not type(expected):
        raise ValueError(f"RQ2 content-horizon {path} has an invalid JSON type")


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
    ledger = compile_rq2_content_horizon_ledger(root)
    path = root / RQ2_CONTENT_HORIZON_LEDGER_PATH
    if arguments.write:
        persist_rq2_content_horizon_ledger(path, ledger, root=root)
    print(
        json.dumps(
            {
                "path": str(path),
                "sha256": ledger.sha256,
                "opportunities": len(ledger.rows),
                "physical_jobs": sum(row.reused_from is None for row in ledger.rows),
                "status": "materialized" if arguments.write else "preview",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
