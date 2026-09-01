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
RQ4_ARTIST_ALBUM_LR_BOUNDARY_LEDGER_PATH = (
    "experiments/g3_pretrained_item_embeddings/protocol/ledgers/"
    "rq4_artist_album_width64_horizon25_lr_boundary.json"
)


@dataclass(frozen=True)
class Rq4ArtistAlbumLrBoundaryJob:
    id: str
    run_name: str
    embedding_learning_rate: float
    deep_learning_rate: float

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "family_id": "rq4_artist_album",
            "phase": "rq4_metadata_joint_lr_boundary_extension",
            "stage": "rq4_artist_album_width64_horizon25_lr_boundary",
            "role": "joint_learning_rate_boundary_probe",
            "run_name": self.run_name,
            "reused_from": None,
            "representation": {
                "history": "selected_rq2_content_concat",
                "history_hidden_dim": 128,
                "catalog": "id_frozen_content",
                "metadata": ["artist", "album"],
                "metadata_dim": 64,
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
class Rq4ArtistAlbumLrBoundaryLedger:
    capacity_extension_selection: Rq4InputReference
    rows: tuple[Rq4ArtistAlbumLrBoundaryJob, ...]

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self._payload())

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "kind": "g3_rq4_artist_album_width64_horizon25_lr_boundary",
            "protocol_sha256": APPROVED_PROTOCOL_SHA256,
            "inputs": {
                "capacity_extension_selection": (
                    self.capacity_extension_selection.to_dict()
                )
            },
            "source_selection": {
                "row_id": "rq4_artist_album:09",
                "family_id": "rq4_artist_album",
                "metadata_dim": 64,
                "horizon_epochs": 25,
                "embedding_learning_rate": 0.17783052497147875,
                "deep_learning_rate": 0.010430488535480936,
                "embedding_direction": "upper",
                "deep_direction": "lower",
            },
            "boundary_rule": {
                "joint_outward_probes": 3,
                "embedding_direction": "upper",
                "deep_direction": "lower",
                "outward_factors": [
                    math.sqrt(2.0),
                    2.0,
                    2.0 * math.sqrt(2.0),
                ],
                "capacity_fixed": 64,
                "horizon_fixed": 25,
                "artist_and_album_capacity_families_unlaunched": True,
            },
            "opportunity_accounting": {
                "logical_total": 3,
                "physical_total": 3,
            },
            "artifact_contracts": [
                (
                    contract.to_dict()
                    if contract.name != "job_contract"
                    else contract.to_dict()
                    | {"filename": "g3_rq4_artist_album_lr_boundary_job.json"}
                )
                for contract in RQ4_INITIAL_ARTIFACT_CONTRACTS
            ],
            "rows": [row.to_dict() for row in self.rows],
        }

    def to_dict(self) -> dict[str, object]:
        return self._payload() | {"sha256": self.sha256}


def compile_rq4_artist_album_lr_boundary_ledger(
    root: Path,
) -> Rq4ArtistAlbumLrBoundaryLedger:
    root = root.resolve(strict=True)
    selection_path = root / RQ4_CAPACITY_EXTENSION_SELECTION_PATH
    selection = _load_selection(selection_path)
    _validate_source_selection(selection)
    embedding = 0.17783052497147875
    deep = 0.010430488535480936
    factors = (math.sqrt(2.0), 2.0, 2.0 * math.sqrt(2.0))
    rows = tuple(
        Rq4ArtistAlbumLrBoundaryJob(
            id=f"rq4_artist_album_lr_boundary:{index:02d}",
            run_name=(
                "g3_rq4_artist_album_width64_horizon25_joint_lr_boundary_"
                f"probe_{index:02d}_native50m"
            ),
            embedding_learning_rate=embedding * factor,
            deep_learning_rate=deep / factor,
        )
        for index, factor in enumerate(factors, start=1)
    )
    return Rq4ArtistAlbumLrBoundaryLedger(
        capacity_extension_selection=_reference(
            root,
            selection_path,
            RQ4_CAPACITY_EXTENSION_SELECTION_SHA256,
        ),
        rows=rows,
    )


def persist_rq4_artist_album_lr_boundary_ledger(
    path: Path, ledger: Rq4ArtistAlbumLrBoundaryLedger, *, root: Path
) -> Path:
    _require_canonical_path(path, root=root, strict=False)
    if ledger != compile_rq4_artist_album_lr_boundary_ledger(root):
        raise ValueError("RQ4 artist+album LR-boundary ledger changed")
    content = (_canonical_json(ledger.to_dict()) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(content)
    except FileExistsError:
        if path.read_bytes() != content:
            raise RuntimeError(f"immutable RQ4 LR-boundary ledger differs: {path}")
    return path


def load_rq4_artist_album_lr_boundary_ledger(
    path: Path, *, root: Path, expected_ledger_sha256: str
) -> Rq4ArtistAlbumLrBoundaryLedger:
    _require_canonical_path(path, root=root, strict=True)
    document = _load_json(path)
    expected = compile_rq4_artist_album_lr_boundary_ledger(root)
    payload = {name: value for name, value in document.items() if name != "sha256"}
    if (
        expected.sha256 != expected_ledger_sha256
        or document.get("sha256") != expected_ledger_sha256
        or _canonical_sha256(payload) != expected_ledger_sha256
        or not _same_json_type_and_value(document, expected.to_dict())
    ):
        raise ValueError("RQ4 artist+album LR-boundary ledger changed")
    return expected


def validate_rq4_artist_album_lr_boundary_ledger_document(
    document: object, *, root: Path, expected_ledger_sha256: str
) -> Rq4ArtistAlbumLrBoundaryLedger:
    expected = compile_rq4_artist_album_lr_boundary_ledger(root)
    if not isinstance(document, dict):
        raise ValueError("RQ4 artist+album LR-boundary ledger must be an object")
    payload = {name: value for name, value in document.items() if name != "sha256"}
    if (
        expected.sha256 != expected_ledger_sha256
        or document.get("sha256") != expected_ledger_sha256
        or _canonical_sha256(payload) != expected_ledger_sha256
        or not _same_json_type_and_value(document, expected.to_dict())
    ):
        raise ValueError("RQ4 artist+album LR-boundary ledger changed")
    return expected


def _validate_source_selection(selection: dict[str, object]) -> None:
    families = selection.get("family_selections")
    selected = families.get("rq4_artist_album") if isinstance(families, dict) else None
    boundary = selected.get("boundary_decision") if isinstance(selected, dict) else None
    winner = selected.get("selected") if isinstance(selected, dict) else None
    if (
        selection.get("opportunity_accounting", {}).get("combined_total") != 36
        or selection.get("capacity_renewed_approval_required")
        != ["rq4_album", "rq4_artist"]
        or not isinstance(boundary, dict)
        or not isinstance(winner, dict)
        or winner.get("row_id") != "rq4_artist_album:09"
        or winner.get("metadata_dim") != 64
        or winner.get("horizon_epochs") != 25
        or winner.get("embedding_learning_rate") != 0.17783052497147875
        or winner.get("deep_learning_rate") != 0.010430488535480936
        or boundary.get("capacity", {}).get("direction") is not None
        or boundary.get("embedding_learning_rate", {}).get("direction") != "upper"
        or boundary.get("deep_learning_rate", {}).get("direction") != "lower"
        or boundary.get("learning_rate_extension_required") is not True
    ):
        raise ValueError("RQ4 artist+album LR-boundary source selection changed")


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
        raise ValueError("RQ4 LR-boundary input reference is invalid")
    return Rq4InputReference(
        path=str(resolved.relative_to(root)),
        size_bytes=resolved.stat().st_size,
        sha256=hashlib.sha256(resolved.read_bytes()).hexdigest(),
        logical_sha256=logical_sha256,
    )


def _require_canonical_path(path: Path, *, root: Path, strict: bool) -> None:
    canonical = root.resolve(strict=True) / RQ4_ARTIST_ALBUM_LR_BOUNDARY_LEDGER_PATH
    if path.is_symlink() or path.resolve(strict=strict) != canonical.resolve(strict=strict):
        raise ValueError("RQ4 artist+album LR ledger must use its canonical path")


def _load_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(
            path.read_text(),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load RQ4 LR-boundary input {path}") from error
    if not isinstance(value, dict):
        raise ValueError("RQ4 LR-boundary input must be an object")
    return value


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


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"duplicate JSON key: {name}")
        result[name] = value
    return result


def _reject_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant: {value}")


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()
