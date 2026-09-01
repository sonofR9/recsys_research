from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .constants import (
    APPROVED_PROTOCOL,
    APPROVED_PROTOCOL_SHA256,
    APPROVED_UNTIED_CONTROL_LEDGER_SHA256,
)
from .control_ledger import ManifestReference
from .search import (
    APPROVED_FAMILY_SPECS,
    SearchCoordinate,
    compile_capacity_first_stage,
)


RQ2_CAPACITY_LEDGER_PATH = (
    "experiments/g3_pretrained_item_embeddings/protocol/ledgers/"
    "rq2_capacity_preselection.json"
)
APPROVED_RQ2_CAPACITY_LEDGER_SHA256 = (
    "65b3c58e40b90aea1df59443da2e29c23dc21667c70c1d5fb3b0638c6ada36d4"
)
PREDECESSOR_CALIBRATION_PATH = (
    "experiments/g3_pretrained_item_embeddings/evidence/"
    "untied_control_calibration.json"
)
PREDECESSOR_CALIBRATION_SHA256 = (
    "015c94a182bc0df4179092e098e69b9b12c4fc62474ff4a2f15ad5d3e693e896"
)
UNTIED_CONTROL_LEDGER_PATH = (
    "experiments/g3_pretrained_item_embeddings/protocol/ledgers/untied_control.json"
)
G4_CONTROL_MANIFEST_PATH = (
    "experiments/g4_future_items/protocol/selected_control_manifest.json"
)
G4_CONTROL_MANIFEST_SHA256 = (
    "c30fb4eafcea2cefa1099631a40ca1531245e412c1cedcdbd02d9f7fea7aafd6"
)
CONTENT_MANIFEST_PATH = (
    "experiments/g3_pretrained_item_embeddings/protocol/artifacts/"
    "native50m_content.json"
)
CONTENT_MANIFEST_SHA256 = (
    "5e24e5db5d3a5635433abd962b1de0753599618c2c0ab67edab6801b967ab070"
)
FEATURE_MANIFEST_PATH = (
    "experiments/g3_pretrained_item_embeddings/protocol/artifacts/"
    "native50m_features.json"
)
FEATURE_MANIFEST_SHA256 = (
    "02e919339094e5091e77d09bd77ea669b665c7f6f49a29b6f27d6708ee9cf021"
)


@dataclass(frozen=True)
class Rq2CapacityJob:
    id: str
    family_id: str
    phase: str
    run_name: str
    dataset_size: str
    batch_size: int
    seed: int
    capacity: int
    embedding_learning_rate: float
    deep_learning_rate: float
    horizon_epochs: int

    def to_dict(self) -> dict[str, object]:
        history = {
            "rq2_content_concat": "learned_item_id_plus_frozen_content",
            "rq2_id_only_densenet": "learned_item_id_densenet",
        }[self.family_id]
        return {
            "id": self.id,
            "family_id": self.family_id,
            "phase": self.phase,
            "run_name": self.run_name,
            "stage": "rq2_capacity_preselection",
            "role": "search",
            "representation": {
                "id": self.family_id,
                "history": history,
                "catalog": "learned_item_id",
                "history_hidden_dim": self.capacity,
                "separate_history_catalog_tables": True,
                "content_trainable": False,
                "content_width": (
                    128 if self.family_id == "rq2_content_concat" else None
                ),
            },
            "dataset": {
                "size": self.dataset_size,
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
class Rq2CapacityLedger:
    schema_version: int
    kind: str
    protocol_sha256: str
    maximum_jobs: int
    predecessor_calibration: ManifestReference
    untied_control_ledger: ManifestReference
    g4_control_manifest: ManifestReference
    content_manifest: ManifestReference
    feature_manifest: ManifestReference
    rows: tuple[Rq2CapacityJob, ...]

    @property
    def inputs(self) -> dict[str, ManifestReference]:
        return {
            "predecessor_calibration": self.predecessor_calibration,
            "untied_control_ledger": self.untied_control_ledger,
            "g4_control_manifest": self.g4_control_manifest,
            "content_manifest": self.content_manifest,
            "feature_manifest": self.feature_manifest,
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
            "inputs": {
                key: reference.to_dict() for key, reference in self.inputs.items()
            },
            "rows": [row.to_dict() for row in self.rows],
        }

    def to_dict(self) -> dict[str, object]:
        return self._payload() | {"sha256": self.sha256}


def initial_rq2_capacity_ledger() -> Rq2CapacityLedger:
    family_ids = ("rq2_content_concat", "rq2_id_only_densenet")
    specs = {
        spec.id: spec for spec in APPROVED_FAMILY_SPECS if spec.id in family_ids
    }
    if set(specs) != set(family_ids):
        raise ValueError("approved RQ2 capacity family definitions are incomplete")
    rows = tuple(
        _job_from_coordinate(coordinate)
        for family_id in family_ids
        for coordinate in compile_capacity_first_stage(specs[family_id])
    )
    if len(rows) != 18:
        raise ValueError("approved RQ2 capacity-preselection budget drifted")
    ledger = Rq2CapacityLedger(
        schema_version=1,
        kind="g3_rq2_capacity_preselection",
        protocol_sha256=APPROVED_PROTOCOL_SHA256,
        maximum_jobs=18,
        predecessor_calibration=ManifestReference(
            "g3_untied_control_calibration",
            PREDECESSOR_CALIBRATION_PATH,
            PREDECESSOR_CALIBRATION_SHA256,
        ),
        untied_control_ledger=ManifestReference(
            "g3_untied_control_ledger",
            UNTIED_CONTROL_LEDGER_PATH,
            APPROVED_UNTIED_CONTROL_LEDGER_SHA256,
        ),
        g4_control_manifest=ManifestReference(
            "g4_selected_control",
            G4_CONTROL_MANIFEST_PATH,
            G4_CONTROL_MANIFEST_SHA256,
        ),
        content_manifest=ManifestReference(
            "native50m_content",
            CONTENT_MANIFEST_PATH,
            CONTENT_MANIFEST_SHA256,
        ),
        feature_manifest=ManifestReference(
            "native50m_features",
            FEATURE_MANIFEST_PATH,
            FEATURE_MANIFEST_SHA256,
        ),
        rows=rows,
    )
    if ledger.sha256 != APPROVED_RQ2_CAPACITY_LEDGER_SHA256:
        raise ValueError("approved RQ2 capacity ledger definition drifted")
    return ledger


def validate_rq2_capacity_ledger_document(document: object) -> Rq2CapacityLedger:
    if not isinstance(document, dict):
        raise ValueError("RQ2 capacity ledger must be an object")
    _require_keys(
        document,
        {
            "schema_version",
            "kind",
            "protocol_sha256",
            "maximum_jobs",
            "inputs",
            "rows",
            "sha256",
        },
        "RQ2 capacity ledger keys",
    )
    if (
        document["schema_version"] != 1
        or document["kind"] != "g3_rq2_capacity_preselection"
    ):
        raise ValueError("unsupported RQ2 capacity ledger identity")
    if document["protocol_sha256"] != APPROVED_PROTOCOL_SHA256:
        raise ValueError("RQ2 capacity ledger does not bind the approved protocol")
    inputs = document["inputs"]
    if not isinstance(inputs, dict):
        raise ValueError("RQ2 capacity ledger inputs must be an object")
    expected = initial_rq2_capacity_ledger()
    _require_keys(inputs, set(expected.inputs), "RQ2 capacity input keys")
    for key, reference in expected.inputs.items():
        if _reference_from_document(inputs[key]) != reference:
            raise ValueError(f"RQ2 capacity ledger {key} binding is not approved")
    rows = document["rows"]
    if not isinstance(rows, list):
        raise ValueError("RQ2 capacity ledger rows must be a list")
    for row in rows:
        _validate_row_schema(row)
    _validate_exact_json_types(document, expected.to_dict(), path="ledger")
    payload = {key: value for key, value in document.items() if key != "sha256"}
    actual_sha256 = hashlib.sha256(_canonical_json(payload).encode()).hexdigest()
    if actual_sha256 != expected.sha256:
        raise ValueError(
            "RQ2 capacity ledger no longer matches the approved "
            "capacity-preselection coordinates"
        )
    if document["sha256"] != actual_sha256:
        raise ValueError(
            "RQ2 capacity ledger hash does not match its canonical payload"
        )
    return expected


def load_rq2_capacity_ledger(path: Path) -> Rq2CapacityLedger:
    try:
        document = json.loads(
            path.read_text(),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load RQ2 capacity ledger {path}") from error
    ledger = validate_rq2_capacity_ledger_document(document)
    if ledger.sha256 != APPROVED_RQ2_CAPACITY_LEDGER_SHA256:
        raise ValueError(
            "RQ2 capacity ledger does not match the approved immutable hash"
        )
    return ledger


def persist_rq2_capacity_ledger(path: Path, ledger: Rq2CapacityLedger) -> Path:
    validate_rq2_capacity_ledger_document(ledger.to_dict())
    content = (_canonical_json(ledger.to_dict()) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(content)
    except FileExistsError:
        if path.read_bytes() != content:
            raise RuntimeError(f"immutable RQ2 capacity ledger differs: {path}")
    return path


def materialize_rq2_capacity_ledger(*, root: Path, write: bool) -> dict[str, object]:
    ledger = initial_rq2_capacity_ledger()
    path = root.resolve() / RQ2_CAPACITY_LEDGER_PATH
    if write:
        persist_rq2_capacity_ledger(path, ledger)
    return {
        "path": str(path),
        "rows": len(ledger.rows),
        "sha256": ledger.sha256,
        "status": "materialized" if write else "preview",
    }


def _job_from_coordinate(coordinate: SearchCoordinate) -> Rq2CapacityJob:
    if (
        coordinate.family_id
        not in {"rq2_content_concat", "rq2_id_only_densenet"}
        or coordinate.role != "search"
        or coordinate.capacity is None
        or coordinate.horizon_epochs != APPROVED_PROTOCOL.control.horizon_epochs
    ):
        raise ValueError(
            "RQ2 capacity ledger received an invalid first-stage coordinate"
        )
    return Rq2CapacityJob(
        id=coordinate.id,
        family_id=coordinate.family_id,
        phase="capacity_preselection",
        run_name=(
            f"g3_{coordinate.family_id}_width_{coordinate.capacity}_"
            f"trial_{coordinate.opportunity_index % 3 + 1:02d}_native50m"
        ),
        dataset_size=APPROVED_PROTOCOL.main_dataset_size,
        batch_size=coordinate.batch_size,
        seed=coordinate.seed,
        capacity=coordinate.capacity,
        embedding_learning_rate=coordinate.embedding_learning_rate,
        deep_learning_rate=coordinate.deep_learning_rate,
        horizon_epochs=coordinate.horizon_epochs,
    )


def _reference_from_document(value: object) -> ManifestReference:
    if not isinstance(value, dict):
        raise ValueError("RQ2 capacity manifest reference must be an object")
    _require_keys(
        value,
        {"kind", "path", "sha256"},
        "RQ2 capacity manifest reference keys",
    )
    if not all(isinstance(value[key], str) for key in value):
        raise ValueError("RQ2 capacity manifest reference fields must be strings")
    return ManifestReference(value["kind"], value["path"], value["sha256"])


def _validate_row_schema(value: object) -> None:
    if not isinstance(value, dict):
        raise ValueError("RQ2 capacity ledger row must be an object")
    _require_keys(
        value,
        {
            "id",
            "family_id",
            "phase",
            "run_name",
            "stage",
            "role",
            "representation",
            "dataset",
            "training",
        },
        "RQ2 capacity row keys",
    )
    representation = value["representation"]
    dataset = value["dataset"]
    training = value["training"]
    if not all(isinstance(item, dict) for item in (representation, dataset, training)):
        raise ValueError("RQ2 capacity row nested fields must be objects")
    _require_keys(
        representation,
        {
            "id",
            "history",
            "catalog",
            "history_hidden_dim",
            "separate_history_catalog_tables",
            "content_trainable",
            "content_width",
        },
        "RQ2 capacity representation keys",
    )
    _require_keys(
        dataset,
        {
            "size",
            "source",
            "event_limit",
            "sampling",
            "minimum_user_interactions",
            "validation_interval_seconds",
            "candidate_catalog",
            "exclude_seen",
        },
        "RQ2 capacity dataset keys",
    )
    _require_keys(
        training,
        {
            "batch_size",
            "seed",
            "embedding_learning_rate",
            "deep_learning_rate",
            "horizon_epochs",
            "validate_every_epoch",
            "restore_best_validation_epoch",
        },
        "RQ2 capacity training keys",
    )


def _require_keys(
    value: Mapping[str, object], expected: set[str], message: str
) -> None:
    if set(value) != expected:
        raise ValueError(f"{message} do not match the closed schema")


def _validate_exact_json_types(actual: object, expected: object, *, path: str) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            raise ValueError(f"RQ2 capacity {path} has an invalid JSON type")
        for key, expected_value in expected.items():
            if key in actual:
                _validate_exact_json_types(
                    actual[key], expected_value, path=f"{path}.{key}"
                )
        return
    if isinstance(expected, list):
        if not isinstance(actual, list):
            raise ValueError(f"RQ2 capacity {path} has an invalid JSON type")
        for index, (actual_value, expected_value) in enumerate(
            zip(actual, expected, strict=False)
        ):
            _validate_exact_json_types(
                actual_value,
                expected_value,
                path=f"{path}[{index}]",
            )
        return
    if type(actual) is not type(expected):
        raise ValueError(f"RQ2 capacity {path} has an invalid JSON type")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
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
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
    )
    arguments = parser.parse_args()
    print(
        json.dumps(
            materialize_rq2_capacity_ledger(
                root=arguments.root,
                write=arguments.write,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
