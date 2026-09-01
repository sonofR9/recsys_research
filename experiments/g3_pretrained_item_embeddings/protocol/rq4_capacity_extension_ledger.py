from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from .constants import APPROVED_PROTOCOL_SHA256
from .rq4 import RQ4_METADATA_FAMILIES
from .rq4_initial_ledger import (
    RQ3_FINAL_EVIDENCE_LOGICAL_SHA256,
    RQ3_FINAL_SELECTED_ROW_ID,
    RQ4_INITIAL_ARTIFACT_CONTRACTS,
    RQ4_INITIAL_LEDGER_LOGICAL_SHA256,
    RQ4_INITIAL_LEDGER_PATH,
    Rq4InputReference,
    load_rq4_initial_ledger,
)


RQ4_CAPACITY_SELECTION_PATH = (
    "experiments/g3_pretrained_item_embeddings/evidence/"
    "rq4_metadata_capacity_selection_native50m.json"
)
RQ4_CAPACITY_SELECTION_SHA256 = (
    "c5cde11bd82d52858c852870bdb1396ee70645bf0431b3bc81457708d32ce1e0"
)
RQ4_CAPACITY_EXTENSION_LEDGER_PATH = (
    "experiments/g3_pretrained_item_embeddings/protocol/ledgers/"
    "rq4_metadata_capacity_width128.json"
)


@dataclass(frozen=True)
class Rq4CapacityExtensionJob:
    id: str
    family_id: str
    run_name: str
    embedding_learning_rate: float
    deep_learning_rate: float
    horizon_epochs: int
    metadata: tuple[str, ...]
    history_hidden_dim: int
    catalog_representation: str

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "family_id": self.family_id,
            "phase": "rq4_metadata_capacity_boundary_extension",
            "stage": "rq4_metadata_capacity_width128_post_initial",
            "role": "metadata_capacity_boundary_extension",
            "run_name": self.run_name,
            "reused_from": None,
            "representation": {
                "history": "selected_rq2_content_concat",
                "history_hidden_dim": self.history_hidden_dim,
                "catalog": self.catalog_representation,
                "metadata": list(self.metadata),
                "metadata_dim": 128,
                "metadata_pooling": "mean",
                "metadata_attachment": (
                    "history_and_catalog_concat_then_separate_densenet"
                ),
            },
            "dataset": {
                "size": "native-50m",
                "source": "likes",
                "event_limit": 50_000_000,
                "sampling": "none",
                "batch_size": 512,
                "seed": 42,
            },
            "training": {
                "batch_size": 512,
                "seed": 42,
                "embedding_learning_rate": self.embedding_learning_rate,
                "deep_learning_rate": self.deep_learning_rate,
                "horizon_epochs": self.horizon_epochs,
                "validate_every_epoch": True,
                "restore_best_validation_epoch": True,
            },
        }


@dataclass(frozen=True)
class Rq4CapacityExtensionLedger:
    initial_ledger: Rq4InputReference
    capacity_selection: Rq4InputReference
    rows: tuple[Rq4CapacityExtensionJob, ...]

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self._payload())

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "kind": "g3_rq4_metadata_capacity_width128_extension",
            "protocol_sha256": APPROVED_PROTOCOL_SHA256,
            "inputs": {
                "initial_ledger": self.initial_ledger.to_dict(),
                "capacity_selection": self.capacity_selection.to_dict(),
            },
            "boundary_rule": {
                "selected_capacity": 64,
                "direction": "upper",
                "extension_capacity": 128,
                "families": list(RQ4_METADATA_FAMILIES),
            },
            "opportunity_accounting": {
                "logical_per_family": 3,
                "logical_total": 9,
                "physical_total": 9,
                "equal_opportunities": True,
            },
            "artifact_contracts": [
                (
                    contract.to_dict()
                    if contract.name != "job_contract"
                    else contract.to_dict()
                    | {"filename": "g3_rq4_capacity_extension_job.json"}
                )
                for contract in RQ4_INITIAL_ARTIFACT_CONTRACTS
            ],
            "rows": [row.to_dict() for row in self.rows],
        }

    def to_dict(self) -> dict[str, object]:
        return self._payload() | {"sha256": self.sha256}


def compile_rq4_capacity_extension_ledger(root: Path) -> Rq4CapacityExtensionLedger:
    root = root.resolve(strict=True)
    initial_path = root / RQ4_INITIAL_LEDGER_PATH
    initial = load_rq4_initial_ledger(
        initial_path,
        root=root,
        expected_ledger_sha256=RQ4_INITIAL_LEDGER_LOGICAL_SHA256,
        expected_rq3_sha256=RQ3_FINAL_EVIDENCE_LOGICAL_SHA256,
        expected_rq3_row_id=RQ3_FINAL_SELECTED_ROW_ID,
    )
    selection_path = root / RQ4_CAPACITY_SELECTION_PATH
    selection = _load_logical(selection_path, RQ4_CAPACITY_SELECTION_SHA256)
    family_selections = selection.get("family_selections")
    if (
        selection.get("capacity_extensions_required")
        != sorted(RQ4_METADATA_FAMILIES)
        or not isinstance(family_selections, dict)
        or set(family_selections) != set(RQ4_METADATA_FAMILIES)
    ):
        raise ValueError("RQ4 width-128 extension requires all three upper boundaries")
    rows = []
    for family_id in RQ4_METADATA_FAMILIES:
        selected = family_selections[family_id]
        if (
            selected.get("selected_metadata_dim") != 64
            or selected.get("selected_horizon_epochs") != 25
            or selected.get("capacity_boundary")
            != {"direction": "upper", "extension_capacity": 128}
        ):
            raise ValueError(f"RQ4 {family_id} upper capacity boundary changed")
        source_rows = sorted(
            (
                row
                for row in initial.rows
                if row.family_id == family_id and row.metadata_dim == 64
            ),
            key=lambda row: row.id,
        )
        if len(source_rows) != 3:
            raise ValueError(f"RQ4 {family_id} lacks three width-64 LR probes")
        for index, source in enumerate(source_rows, start=1):
            rows.append(
                Rq4CapacityExtensionJob(
                    id=f"{family_id}_capacity_extension:{index:02d}",
                    family_id=family_id,
                    run_name=(
                        f"g3_{family_id}_capacity_extension_{index:02d}_width_128_"
                        "horizon_25_native50m"
                    ),
                    embedding_learning_rate=source.embedding_learning_rate,
                    deep_learning_rate=source.deep_learning_rate,
                    horizon_epochs=25,
                    metadata=source.metadata,
                    history_hidden_dim=source.history_hidden_dim,
                    catalog_representation=source.catalog_representation,
                )
            )
    if len(rows) != 9 or len({row.run_name for row in rows}) != 9:
        raise ValueError("RQ4 width-128 extension must contain nine unique jobs")
    return Rq4CapacityExtensionLedger(
        initial_ledger=_reference(root, initial_path, initial.sha256),
        capacity_selection=_reference(
            root, selection_path, RQ4_CAPACITY_SELECTION_SHA256
        ),
        rows=tuple(rows),
    )


def persist_rq4_capacity_extension_ledger(
    path: Path, ledger: Rq4CapacityExtensionLedger, *, root: Path
) -> Path:
    _require_canonical_ledger_path(path, root=root, strict=False)
    if compile_rq4_capacity_extension_ledger(root) != ledger:
        raise ValueError("RQ4 width-128 ledger differs from authenticated inputs")
    content = (_canonical_json(ledger.to_dict()) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(content)
    except FileExistsError:
        if path.read_bytes() != content:
            raise RuntimeError(f"immutable RQ4 width-128 ledger differs: {path}")
    return path


def load_rq4_capacity_extension_ledger(
    path: Path,
    *,
    root: Path,
    expected_ledger_sha256: str,
) -> Rq4CapacityExtensionLedger:
    _require_canonical_ledger_path(path, root=root, strict=True)
    document = _load_json_object(path)
    rebuilt = compile_rq4_capacity_extension_ledger(root)
    return _validate_ledger_document(
        document, rebuilt=rebuilt, expected_ledger_sha256=expected_ledger_sha256
    )


def validate_rq4_capacity_extension_ledger_document(
    document: object, *, root: Path, expected_ledger_sha256: str
) -> Rq4CapacityExtensionLedger:
    rebuilt = compile_rq4_capacity_extension_ledger(root)
    return _validate_ledger_document(
        document, rebuilt=rebuilt, expected_ledger_sha256=expected_ledger_sha256
    )


def _validate_ledger_document(
    document: object,
    *,
    rebuilt: Rq4CapacityExtensionLedger,
    expected_ledger_sha256: str,
) -> Rq4CapacityExtensionLedger:
    if not isinstance(document, dict):
        raise ValueError("RQ4 width-128 ledger differs from authenticated inputs")
    payload = {name: item for name, item in document.items() if name != "sha256"}
    if (
        rebuilt.sha256 != expected_ledger_sha256
        or document.get("sha256") != expected_ledger_sha256
        or _canonical_sha256(payload) != expected_ledger_sha256
        or not _same_json_type_and_value(document, rebuilt.to_dict())
    ):
        raise ValueError("RQ4 width-128 ledger differs from authenticated inputs")
    return rebuilt


def _require_canonical_ledger_path(
    path: Path, *, root: Path, strict: bool
) -> None:
    canonical = root.resolve(strict=True) / RQ4_CAPACITY_EXTENSION_LEDGER_PATH
    if path.is_symlink() or path.resolve(strict=strict) != canonical.resolve(strict=strict):
        raise ValueError("RQ4 width-128 ledger must use its canonical project path")


def _same_json_type_and_value(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _same_json_type_and_value(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _same_json_type_and_value(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    return left == right


def _reference(root: Path, path: Path, logical_sha256: str) -> Rq4InputReference:
    resolved = path.resolve(strict=True)
    if path.is_symlink() or not resolved.is_relative_to(root):
        raise ValueError("RQ4 width-128 input reference is invalid")
    return Rq4InputReference(
        path=str(resolved.relative_to(root)),
        size_bytes=resolved.stat().st_size,
        sha256=hashlib.sha256(resolved.read_bytes()).hexdigest(),
        logical_sha256=logical_sha256,
    )


def _load_logical(path: Path, expected_sha256: str) -> dict[str, object]:
    value = _load_json_object(path)
    payload = {name: item for name, item in value.items() if name != "sha256"}
    if value.get("sha256") != expected_sha256 or _canonical_sha256(payload) != expected_sha256:
        raise ValueError("RQ4 capacity selection logical hash changed")
    return value


def _load_json_object(path: Path) -> dict[str, object]:
    if path.is_symlink():
        raise ValueError(f"RQ4 width-128 input cannot be a symlink: {path}")
    try:
        value = json.loads(
            path.read_text(),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load RQ4 width-128 input {path}") from error
    if not isinstance(value, dict):
        raise ValueError("RQ4 width-128 input must be a JSON object")
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant: {value}")


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()
