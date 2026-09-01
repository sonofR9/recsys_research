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
from .control_ledger import ManifestReference
from .search import APPROVED_FAMILY_SPECS, SearchCoordinate, compile_family


UNTIED_CONTROL_LEDGER_PATH = (
    "experiments/g3_pretrained_item_embeddings/protocol/ledgers/untied_control.json"
)
PREDECESSOR_CALIBRATION_PATH = (
    "experiments/g3_pretrained_item_embeddings/evidence/"
    "untied_control_calibration.json"
)
APPROVED_PREDECESSOR_CALIBRATION_SHA256 = (
    "015c94a182bc0df4179092e098e69b9b12c4fc62474ff4a2f15ad5d3e693e896"
)
APPROVED_RQ1_LEDGER_SHA256 = (
    "dfeb5e9f24bdd721de0feefb05224b4e6fdb1b78279eb8b98a2c322ca4a2a69b"
)
RQ1_LEDGER_PATH = (
    "experiments/g3_pretrained_item_embeddings/protocol/ledgers/"
    "rq1_content_input.json"
)


@dataclass(frozen=True)
class Rq1Job:
    id: str
    run_name: str
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
            "stage": "rq1_content_input",
            "role": "search",
            "representation": {
                "id": self.representation,
                "history": "frozen_pretrained_content_projection",
                "catalog": "learned_item_id",
                "separate_history_catalog_tables": True,
                "content_trainable": False,
                "content_width": 128,
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
class Rq1Ledger:
    schema_version: int
    kind: str
    protocol_sha256: str
    maximum_jobs: int
    untied_control_ledger: ManifestReference
    predecessor_calibration: ManifestReference
    rows: tuple[Rq1Job, ...]

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
                "untied_control_ledger": self.untied_control_ledger.to_dict(),
                "predecessor_calibration": self.predecessor_calibration.to_dict(),
            },
            "rows": [row.to_dict() for row in self.rows],
        }

    def to_dict(self) -> dict[str, object]:
        return self._payload() | {"sha256": self.sha256}


def initial_rq1_ledger(
    *, predecessor_calibration: ManifestReference
) -> Rq1Ledger:
    _validate_reference(
        predecessor_calibration,
        expected_kind="g3_untied_control_calibration",
    )
    if predecessor_calibration.path != PREDECESSOR_CALIBRATION_PATH:
        raise ValueError("RQ1 predecessor calibration path is not canonical")
    if predecessor_calibration.sha256 != APPROVED_PREDECESSOR_CALIBRATION_SHA256:
        raise ValueError("RQ1 predecessor calibration hash is not approved")
    control = ManifestReference(
        kind="g3_untied_control_ledger",
        path=UNTIED_CONTROL_LEDGER_PATH,
        sha256=APPROVED_UNTIED_CONTROL_LEDGER_SHA256,
    )
    spec = next(
        spec for spec in APPROVED_FAMILY_SPECS if spec.id == "rq1_content_input"
    )
    rows = tuple(
        _job_from_coordinate(coordinate) for coordinate in compile_family(spec)
    )
    if len(rows) != 9 or len(rows) != spec.budget:
        raise ValueError("approved RQ1 direct-family budget drifted")
    ledger = Rq1Ledger(
        schema_version=1,
        kind="g3_rq1_content_input",
        protocol_sha256=APPROVED_PROTOCOL_SHA256,
        maximum_jobs=spec.budget,
        untied_control_ledger=control,
        predecessor_calibration=predecessor_calibration,
        rows=rows,
    )
    if ledger.sha256 != APPROVED_RQ1_LEDGER_SHA256:
        raise ValueError("approved RQ1 ledger definition drifted")
    return ledger


def validate_rq1_ledger_document(document: object) -> Rq1Ledger:
    if not isinstance(document, dict):
        raise ValueError("RQ1 ledger must be an object")
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
        "RQ1 ledger keys",
    )
    if document["schema_version"] != 1 or document["kind"] != "g3_rq1_content_input":
        raise ValueError("unsupported RQ1 ledger identity")
    if document["protocol_sha256"] != APPROVED_PROTOCOL_SHA256:
        raise ValueError("RQ1 ledger does not bind the approved G3 protocol")
    inputs = document["inputs"]
    if not isinstance(inputs, dict):
        raise ValueError("RQ1 ledger inputs must be an object")
    _require_keys(
        inputs,
        {"untied_control_ledger", "predecessor_calibration"},
        "RQ1 input keys",
    )
    control = _reference_from_document(inputs["untied_control_ledger"])
    predecessor = _reference_from_document(inputs["predecessor_calibration"])
    expected = initial_rq1_ledger(predecessor_calibration=predecessor)
    if control != expected.untied_control_ledger:
        raise ValueError("RQ1 ledger does not bind the immutable untied control")
    rows = document["rows"]
    if not isinstance(rows, list):
        raise ValueError("RQ1 ledger rows must be a list")
    for row in rows:
        _validate_row_schema(row)
    if document != expected.to_dict():
        actual_payload = {
            key: value for key, value in document.items() if key != "sha256"
        }
        if actual_payload != expected._payload():
            raise ValueError(
                "RQ1 ledger no longer matches the approved RQ1 coordinates"
            )
        raise ValueError("RQ1 ledger hash does not match its canonical payload")
    return expected


def load_rq1_ledger(path: Path) -> Rq1Ledger:
    try:
        document = json.loads(
            path.read_text(),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load RQ1 ledger {path}") from error
    ledger = validate_rq1_ledger_document(document)
    if ledger.sha256 != APPROVED_RQ1_LEDGER_SHA256:
        raise ValueError("RQ1 ledger does not match the approved immutable hash")
    return ledger


def persist_rq1_ledger(path: Path, ledger: Rq1Ledger) -> Path:
    validate_rq1_ledger_document(ledger.to_dict())
    content = (_canonical_json(ledger.to_dict()) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(content)
    except FileExistsError:
        if path.read_bytes() != content:
            raise RuntimeError(f"immutable RQ1 ledger differs: {path}")
    return path


def _job_from_coordinate(coordinate: SearchCoordinate) -> Rq1Job:
    if coordinate.family_id != "rq1_content_input" or coordinate.role != "search":
        raise ValueError("RQ1 ledger received a non-RQ1 coordinate")
    return Rq1Job(
        id=coordinate.id,
        run_name=(
            f"g3_rq1_content_input_trial_{coordinate.opportunity_index + 1:02d}_"
            "native50m"
        ),
        dataset_size=APPROVED_PROTOCOL.main_dataset_size,
        batch_size=coordinate.batch_size,
        seed=coordinate.seed,
        representation="content_only_history",
        embedding_learning_rate=coordinate.embedding_learning_rate,
        deep_learning_rate=coordinate.deep_learning_rate,
        horizon_epochs=coordinate.horizon_epochs,
    )


def _reference_from_document(value: object) -> ManifestReference:
    if not isinstance(value, dict):
        raise ValueError("RQ1 manifest reference must be an object")
    _require_keys(value, {"kind", "path", "sha256"}, "RQ1 manifest reference keys")
    if not all(isinstance(value[key], str) for key in value):
        raise ValueError("RQ1 manifest reference fields must be strings")
    return ManifestReference(value["kind"], value["path"], value["sha256"])


def _validate_reference(reference: ManifestReference, *, expected_kind: str) -> None:
    if reference.kind != expected_kind:
        raise ValueError(f"expected {expected_kind!r} manifest reference")
    relative = Path(reference.path)
    if not reference.path or relative.is_absolute() or ".." in relative.parts:
        raise ValueError("RQ1 manifest reference path must be project-relative")
    if len(reference.sha256) != 64:
        raise ValueError("RQ1 manifest reference hash must be SHA-256")
    try:
        int(reference.sha256, 16)
    except ValueError as error:
        raise ValueError("RQ1 manifest reference hash must be SHA-256") from error


def _validate_row_schema(value: object) -> None:
    if not isinstance(value, dict):
        raise ValueError("RQ1 ledger row must be an object")
    _require_keys(
        value,
        {"id", "run_name", "stage", "role", "representation", "dataset", "training"},
        "RQ1 row keys",
    )
    representation = value["representation"]
    dataset = value["dataset"]
    training = value["training"]
    if not all(
        isinstance(nested, dict)
        for nested in (representation, dataset, training)
    ):
        raise ValueError("RQ1 row nested fields must be objects")
    _require_keys(
        representation,
        {
            "id",
            "history",
            "catalog",
            "separate_history_catalog_tables",
            "content_trainable",
            "content_width",
            "width",
        },
        "RQ1 representation keys",
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
        "RQ1 dataset keys",
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
        "RQ1 training keys",
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
