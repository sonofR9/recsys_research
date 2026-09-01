from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

from .constants import APPROVED_PROTOCOL_SHA256
from .rq4_initial_ledger import RQ4_INITIAL_ARTIFACT_CONTRACTS, Rq4InputReference


RQ4_CAPACITY_EXTENSION_SELECTION_PATH = (
    "experiments/g3_pretrained_item_embeddings/evidence/"
    "rq4_metadata_capacity_extension_selection_native50m.json"
)
RQ4_CAPACITY_EXTENSION_SELECTION_SHA256 = (
    "6400618960f1d300f0390adc20c7c0bcbd8d32e28dcdaab6e9623f2b157d3559"
)
RQ4_SINGLE_METADATA_WIDTH256_LEDGER_PATH = (
    "experiments/g3_pretrained_item_embeddings/protocol/ledgers/"
    "rq4_single_metadata_width256_horizon25_boundary.json"
)


@dataclass(frozen=True)
class Rq4SingleMetadataWidth256Job:
    id: str
    family_id: str
    run_name: str
    metadata: str
    embedding_learning_rate: float
    deep_learning_rate: float

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "family_id": self.family_id,
            "phase": "rq4_single_metadata_width256_boundary_extension",
            "stage": "rq4_single_metadata_width256_horizon25",
            "role": "approved_joint_capacity_lr_boundary_probe",
            "run_name": self.run_name,
            "reused_from": None,
            "representation": {
                "history": "selected_rq2_content_concat",
                "history_hidden_dim": 128,
                "catalog": "id_frozen_content",
                "metadata": [self.metadata],
                "metadata_dim": 256,
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
                "horizon_epochs": 25,
                "validate_every_epoch": True,
                "restore_best_validation_epoch": True,
            },
        }


@dataclass(frozen=True)
class Rq4SingleMetadataWidth256Ledger:
    capacity_extension_selection: Rq4InputReference
    rows: tuple[Rq4SingleMetadataWidth256Job, ...]

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self._payload())

    def _payload(self) -> dict[str, object]:
        factors = [math.sqrt(2.0), 2.0, 2.0 * math.sqrt(2.0)]
        return {
            "schema_version": 1,
            "kind": "g3_rq4_single_metadata_width256_horizon25_boundary",
            "protocol_sha256": APPROVED_PROTOCOL_SHA256,
            "inputs": {
                "capacity_extension_selection": (
                    self.capacity_extension_selection.to_dict()
                )
            },
            "approval": {
                "decision": "exactly_six_width256_jobs",
                "further_capacity_width_requires_renewed_approval": True,
            },
            "boundary_rule": {
                "capacity": 256,
                "horizon_epochs": 25,
                "outward_factors": factors,
                "rq4_artist": {
                    "source_row_id": "rq4_artist_capacity_extension:03",
                    "embedding_base": 0.17783052497147875,
                    "embedding_operation": "multiply",
                    "deep_base": 0.010430488535480936,
                    "deep_operation": "divide",
                },
                "rq4_album": {
                    "source_row_id": "rq4_album_capacity_extension:01",
                    "embedding_base": 0.05753144041634071,
                    "embedding_operation": "divide",
                    "deep_fixed": 0.01852175330591617,
                },
            },
            "opportunity_accounting": {
                "logical_per_family": 3,
                "logical_total": 6,
                "physical_total": 6,
                "families": ["rq4_artist", "rq4_album"],
                "artist_album_jobs": 0,
            },
            "artifact_contracts": [
                (
                    contract.to_dict()
                    if contract.name != "job_contract"
                    else contract.to_dict()
                    | {"filename": "g3_rq4_single_metadata_width256_job.json"}
                )
                for contract in RQ4_INITIAL_ARTIFACT_CONTRACTS
            ],
            "rows": [row.to_dict() for row in self.rows],
        }

    def to_dict(self) -> dict[str, object]:
        return self._payload() | {"sha256": self.sha256}


def compile_rq4_single_metadata_width256_ledger(
    root: Path,
) -> Rq4SingleMetadataWidth256Ledger:
    root = root.resolve(strict=True)
    selection_path = root / RQ4_CAPACITY_EXTENSION_SELECTION_PATH
    selection = _load_selection(selection_path)
    _validate_source_selections(selection)
    factors = (math.sqrt(2.0), 2.0, 2.0 * math.sqrt(2.0))
    rows: list[Rq4SingleMetadataWidth256Job] = []
    for index, factor in enumerate(factors, start=1):
        rows.append(
            Rq4SingleMetadataWidth256Job(
                id=f"rq4_artist_width256_boundary:{index:02d}",
                family_id="rq4_artist",
                run_name=(
                    "g3_rq4_artist_width256_horizon25_boundary_"
                    f"probe_{index:02d}_native50m"
                ),
                metadata="artist",
                embedding_learning_rate=0.17783052497147875 * factor,
                deep_learning_rate=0.010430488535480936 / factor,
            )
        )
        rows.append(
            Rq4SingleMetadataWidth256Job(
                id=f"rq4_album_width256_boundary:{index:02d}",
                family_id="rq4_album",
                run_name=(
                    "g3_rq4_album_width256_horizon25_boundary_"
                    f"probe_{index:02d}_native50m"
                ),
                metadata="album",
                embedding_learning_rate=0.05753144041634071 / factor,
                deep_learning_rate=0.01852175330591617,
            )
        )
    if len(rows) != 6 or len({row.run_name for row in rows}) != 6:
        raise ValueError("RQ4 width-256 boundary ledger must have six unique jobs")
    return Rq4SingleMetadataWidth256Ledger(
        capacity_extension_selection=_reference(
            root, selection_path, RQ4_CAPACITY_EXTENSION_SELECTION_SHA256
        ),
        rows=tuple(rows),
    )


def persist_rq4_single_metadata_width256_ledger(
    path: Path, ledger: Rq4SingleMetadataWidth256Ledger, *, root: Path
) -> Path:
    _require_canonical_path(path, root=root, strict=False)
    if ledger != compile_rq4_single_metadata_width256_ledger(root):
        raise ValueError("RQ4 width-256 boundary ledger changed")
    content = (_canonical_json(ledger.to_dict()) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(content)
    except FileExistsError:
        if path.read_bytes() != content:
            raise RuntimeError(f"immutable RQ4 width-256 ledger differs: {path}")
    return path


def load_rq4_single_metadata_width256_ledger(
    path: Path, *, root: Path, expected_ledger_sha256: str
) -> Rq4SingleMetadataWidth256Ledger:
    _require_canonical_path(path, root=root, strict=True)
    return validate_rq4_single_metadata_width256_ledger_document(
        _load_json(path),
        root=root,
        expected_ledger_sha256=expected_ledger_sha256,
    )


def validate_rq4_single_metadata_width256_ledger_document(
    document: object, *, root: Path, expected_ledger_sha256: str
) -> Rq4SingleMetadataWidth256Ledger:
    expected = compile_rq4_single_metadata_width256_ledger(root)
    if not isinstance(document, dict):
        raise ValueError("RQ4 width-256 boundary ledger must be an object")
    payload = {name: value for name, value in document.items() if name != "sha256"}
    if (
        expected.sha256 != expected_ledger_sha256
        or document.get("sha256") != expected_ledger_sha256
        or _canonical_sha256(payload) != expected_ledger_sha256
        or not _same_json_type_and_value(document, expected.to_dict())
    ):
        raise ValueError("RQ4 width-256 boundary ledger changed")
    return expected


def _validate_source_selections(selection: dict[str, object]) -> None:
    families = selection.get("family_selections")
    if not isinstance(families, dict):
        raise ValueError("RQ4 capacity-extension selections are absent")
    expected = {
        "rq4_artist": (
            "rq4_artist_capacity_extension:03",
            0.17783052497147875,
            0.010430488535480936,
            "upper",
            "lower",
        ),
        "rq4_album": (
            "rq4_album_capacity_extension:01",
            0.05753144041634071,
            0.01852175330591617,
            "lower",
            None,
        ),
    }
    for family_id, values in expected.items():
        family = families.get(family_id)
        selected = family.get("selected") if isinstance(family, dict) else None
        boundary = family.get("boundary_decision") if isinstance(family, dict) else None
        if not isinstance(selected, dict) or not isinstance(boundary, dict):
            raise ValueError(f"RQ4 {family_id} selection is absent")
        if (
            selected.get("row_id") != values[0]
            or selected.get("metadata_dim") != 128
            or selected.get("horizon_epochs") != 25
            or selected.get("embedding_learning_rate") != values[1]
            or selected.get("deep_learning_rate") != values[2]
            or boundary.get("capacity", {}).get("direction") != "upper"
            or boundary.get("capacity", {}).get("renewed_approval_required") is not True
            or boundary.get("embedding_learning_rate", {}).get("direction") != values[3]
            or boundary.get("deep_learning_rate", {}).get("direction") != values[4]
        ):
            raise ValueError(f"RQ4 {family_id} source selection changed")


def _load_selection(path: Path) -> dict[str, object]:
    document = _load_json(path)
    payload = {name: value for name, value in document.items() if name != "sha256"}
    if (
        document.get("sha256") != RQ4_CAPACITY_EXTENSION_SELECTION_SHA256
        or _canonical_sha256(payload) != RQ4_CAPACITY_EXTENSION_SELECTION_SHA256
    ):
        raise ValueError("RQ4 capacity-extension selection changed")
    return document


def _reference(root: Path, path: Path, logical_sha256: str) -> Rq4InputReference:
    resolved = path.resolve(strict=True)
    if path.is_symlink() or not resolved.is_relative_to(root):
        raise ValueError("RQ4 width-256 input reference is invalid")
    return Rq4InputReference(
        path=str(resolved.relative_to(root)),
        size_bytes=resolved.stat().st_size,
        sha256=hashlib.sha256(resolved.read_bytes()).hexdigest(),
        logical_sha256=logical_sha256,
    )


def _require_canonical_path(path: Path, *, root: Path, strict: bool) -> None:
    canonical = root.resolve(strict=True) / RQ4_SINGLE_METADATA_WIDTH256_LEDGER_PATH
    if path.is_symlink() or path.resolve(strict=strict) != canonical.resolve(strict=strict):
        raise ValueError("RQ4 width-256 ledger must use its canonical project path")


def _load_json(path: Path) -> dict[str, object]:
    document = json.loads(path.read_text())
    if not isinstance(document, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return document


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _same_json_type_and_value(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _same_json_type_and_value(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _same_json_type_and_value(a, b)
            for a, b in zip(left, right, strict=True)
        )
    return left == right
