from __future__ import annotations

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
from .search import APPROVED_FAMILY_SPECS, SearchCoordinate, compile_family


_G4_CONTROL_KIND = "g4_selected_control"
_G4_CONTROL_PATH = "experiments/g4_future_items/protocol/selected_control_manifest.json"
_CONTENT_KIND = "native50m_content"
_FEATURE_KIND = "native50m_features"


@dataclass(frozen=True)
class ManifestReference:
    kind: str
    path: str
    sha256: str

    def to_dict(self) -> dict[str, object]:
        return {"kind": self.kind, "path": self.path, "sha256": self.sha256}


@dataclass(frozen=True)
class ControlJob:
    id: str
    run_name: str
    role: str
    dataset_size: str
    batch_size: int
    seed: int
    representation: str
    embedding_learning_rate: float
    deep_learning_rate: float
    horizon_epochs: int

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "run_name": self.run_name,
            "stage": "initial_control",
            "role": self.role,
            "representation": {
                "id": self.representation,
                "history": "learned_item_id",
                "catalog": "learned_item_id",
                "separate_history_catalog_tables": True,
                "tied": False,
                "width": 64,
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
class ControlLedger:
    schema_version: int
    kind: str
    protocol_sha256: str
    maximum_jobs: int
    g4_control: ManifestReference
    content: ManifestReference
    features: ManifestReference
    rows: tuple[ControlJob, ...]

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
                "g4_control": self.g4_control.to_dict(),
                "content": self.content.to_dict(),
                "features": self.features.to_dict(),
            },
            "rows": [row.to_dict() for row in self.rows],
        }

    def to_dict(self) -> dict[str, object]:
        return self._payload() | {"sha256": self.sha256}


def initial_control_ledger(
    *,
    g4_control: ManifestReference,
    content: ManifestReference,
    features: ManifestReference,
) -> ControlLedger:
    _validate_reference(g4_control, expected_kind=_G4_CONTROL_KIND)
    _validate_reference(content, expected_kind=_CONTENT_KIND)
    _validate_reference(features, expected_kind=_FEATURE_KIND)
    if g4_control.sha256 != APPROVED_PROTOCOL.control.manifest_sha256:
        raise ValueError("G3 control ledger does not bind the approved G4 control")
    if g4_control.path != _G4_CONTROL_PATH:
        raise ValueError(
            "G3 control ledger does not bind the exact G4 control manifest"
        )
    spec = next(spec for spec in APPROVED_FAMILY_SPECS if spec.id == "untied_control")
    coordinates = compile_family(spec)
    rows = tuple(_job_from_coordinate(coordinate) for coordinate in coordinates)
    if len(rows) != spec.budget:
        raise ValueError("approved untied-control budget drifted")
    return ControlLedger(
        schema_version=1,
        kind="g3_untied_control",
        protocol_sha256=APPROVED_PROTOCOL_SHA256,
        maximum_jobs=spec.budget,
        g4_control=g4_control,
        content=content,
        features=features,
        rows=rows,
    )


def validate_control_ledger_document(document: object) -> ControlLedger:
    if not isinstance(document, dict):
        raise ValueError("control ledger must be an object")
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
        "ledger keys",
    )
    if document["schema_version"] != 1 or document["kind"] != "g3_untied_control":
        raise ValueError("unsupported control ledger identity")
    if document["protocol_sha256"] != APPROVED_PROTOCOL_SHA256:
        raise ValueError("control ledger does not bind the approved G3 protocol")
    inputs = document["inputs"]
    if not isinstance(inputs, dict):
        raise ValueError("control ledger inputs must be an object")
    _require_keys(inputs, {"g4_control", "content", "features"}, "input keys")
    ledger = initial_control_ledger(
        g4_control=_reference_from_document(inputs["g4_control"]),
        content=_reference_from_document(inputs["content"]),
        features=_reference_from_document(inputs["features"]),
    )
    rows = document["rows"]
    if not isinstance(rows, list):
        raise ValueError("control ledger rows must be a list")
    for row in rows:
        _validate_row_schema(row)
    if document != ledger.to_dict():
        expected_without_hash = ledger._payload()
        actual_without_hash = {
            key: value for key, value in document.items() if key != "sha256"
        }
        if actual_without_hash != expected_without_hash:
            raise ValueError(
                "control ledger no longer matches the approved control coordinates"
            )
        raise ValueError("control ledger hash does not match its canonical payload")
    return ledger


def load_control_ledger(path: Path) -> ControlLedger:
    try:
        document = json.loads(
            path.read_text(),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load control ledger {path}") from error
    ledger = validate_control_ledger_document(document)
    if ledger.sha256 != APPROVED_UNTIED_CONTROL_LEDGER_SHA256:
        raise ValueError("control ledger does not match the approved immutable hash")
    return ledger


def persist_control_ledger(path: Path, ledger: ControlLedger) -> Path:
    validate_control_ledger_document(ledger.to_dict())
    content = (_canonical_json(ledger.to_dict()) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(content)
    except FileExistsError:
        if path.read_bytes() != content:
            raise RuntimeError(f"immutable control ledger differs: {path}")
    return path


def _job_from_coordinate(coordinate: SearchCoordinate) -> ControlJob:
    return ControlJob(
        id=coordinate.id,
        run_name=f"g3_untied_control_trial_{coordinate.opportunity_index + 1:02d}_native50m",
        role=coordinate.role,
        dataset_size=APPROVED_PROTOCOL.main_dataset_size,
        batch_size=coordinate.batch_size,
        seed=coordinate.seed,
        representation="untied_learned_item_id",
        embedding_learning_rate=coordinate.embedding_learning_rate,
        deep_learning_rate=coordinate.deep_learning_rate,
        horizon_epochs=coordinate.horizon_epochs,
    )


def _reference_from_document(value: object) -> ManifestReference:
    if not isinstance(value, dict):
        raise ValueError("manifest reference must be an object")
    _require_keys(value, {"kind", "path", "sha256"}, "manifest reference keys")
    if not all(isinstance(value[key], str) for key in value):
        raise ValueError("manifest reference fields must be strings")
    return ManifestReference(value["kind"], value["path"], value["sha256"])


def _validate_reference(reference: ManifestReference, *, expected_kind: str) -> None:
    if reference.kind != expected_kind:
        raise ValueError(f"expected {expected_kind!r} manifest reference")
    relative = Path(reference.path)
    if not reference.path or relative.is_absolute() or ".." in relative.parts:
        raise ValueError("manifest reference path must be project-relative")
    if len(reference.sha256) != 64:
        raise ValueError("manifest reference hash must be SHA-256")
    try:
        int(reference.sha256, 16)
    except ValueError as error:
        raise ValueError("manifest reference hash must be SHA-256") from error


def _validate_row_schema(value: object) -> None:
    if not isinstance(value, dict):
        raise ValueError("control ledger row must be an object")
    _require_keys(
        value,
        {"id", "run_name", "stage", "role", "representation", "dataset", "training"},
        "control row keys",
    )
    representation = value["representation"]
    dataset = value["dataset"]
    training = value["training"]
    if (
        not isinstance(representation, dict)
        or not isinstance(dataset, dict)
        or not isinstance(training, dict)
    ):
        raise ValueError("control row nested fields must be objects")
    _require_keys(
        representation,
        {
            "id",
            "history",
            "catalog",
            "separate_history_catalog_tables",
            "tied",
            "width",
        },
        "representation keys",
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
        "dataset keys",
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
        "training keys",
    )


def _require_keys(
    value: Mapping[str, object], expected: set[str], message: str
) -> None:
    if set(value) != expected:
        raise ValueError(f"{message} do not match the closed schema")


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
