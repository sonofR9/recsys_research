from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping

from experiments.g3_pretrained_item_embeddings.analysis.rq3_post_boundary_results import (
    RQ3_INITIAL_EVIDENCE_PATH,
    load_rq3_post_boundary_evidence,
)

from .constants import APPROVED_PROTOCOL_SHA256
from .rq3_post_boundary import (
    load_rq3_post_boundary_ledger,
)


RQ3_BOUNDARY_LEDGER_PATH = (
    "experiments/g3_pretrained_item_embeddings/protocol/ledgers/"
    "rq3_output_deep_lr_lower_boundary.json"
)
RQ3_SOURCE_LEDGER_PATH = (
    "experiments/g3_pretrained_item_embeddings/protocol/ledgers/"
    "rq3_post_boundary_output_search.json"
)
RQ3_INITIAL_EVIDENCE_LOGICAL_SHA256 = (
    "972a4fbe2dc213e87b09e1acc12ca1dcfe6f93549691a6a33763cc157ef4f7bb"
)
RQ3_INITIAL_EVIDENCE_FILE_SHA256 = (
    "fb6d65d266c058d0a6915bb972ad89d2f2b1d308ba207b299425b0f1504a19de"
)
RQ3_INITIAL_EVIDENCE_SIZE_BYTES = 1_254_193
RQ3_SOURCE_LEDGER_LOGICAL_SHA256 = (
    "9fc9e47e2f061379f53e21ce73ec1c46ce848fadcecf6793eb0c0f67775d0885"
)
RQ3_BOUNDARY_FAMILY_IDS = (
    "rq3_output_learned_frozen_content",
    "rq3_output_learned_trainable_content",
)
RQ3_BOUNDARY_DEEP_LRS = (
    0.0020271211999999994,
    0.0014333911468070114,
    0.0010135605999999997,
)
RQ3_BOUNDARY_EMBEDDING_LR = 0.3041556165944196
RQ3_BOUNDARY_HORIZON = 40


@dataclass(frozen=True)
class EvidenceReference:
    path: str
    size_bytes: int
    sha256: str
    logical_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "logical_sha256": self.logical_sha256,
        }


@dataclass(frozen=True)
class Rq3BoundaryRow:
    id: str
    family_id: str
    run_name: str
    catalog_representation: str
    deep_learning_rate: float

    @property
    def embedding_learning_rate(self) -> float:
        return RQ3_BOUNDARY_EMBEDDING_LR

    @property
    def horizon_epochs(self) -> int:
        return RQ3_BOUNDARY_HORIZON

    @property
    def history_hidden_dim(self) -> int:
        return 128

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "family_id": self.family_id,
            "phase": "rq3_catalog_output_deep_lr_lower_boundary",
            "stage": "rq3_output_deep_lr_lower_boundary",
            "role": "deep_learning_rate_boundary_probe",
            "run_name": self.run_name,
            "reused_from": None,
            "source_ledger": None,
            "representation": {
                "id": self.family_id,
                "history_representation": "id_content",
                "history_hidden_dim": 128,
                "catalog_representation": self.catalog_representation,
            },
            "dataset": {
                "size": "native-50m",
                "source": "likes",
                "event_limit": 50_000_000,
                "sampling": "none",
                "minimum_user_interactions": 5,
                "validation_interval_seconds": 604800,
                "candidate_catalog": "full",
                "exclude_seen": False,
            },
            "training": {
                "batch_size": 512,
                "seed": 42,
                "embedding_learning_rate": RQ3_BOUNDARY_EMBEDDING_LR,
                "deep_learning_rate": self.deep_learning_rate,
                "horizon_epochs": RQ3_BOUNDARY_HORIZON,
                "validate_every_epoch": True,
                "restore_best_validation_epoch": True,
            },
        }


@dataclass(frozen=True)
class Rq3BoundaryLedger:
    schema_version: int
    kind: str
    protocol_sha256: str
    initial_evidence: EvidenceReference
    source_ledger: EvidenceReference
    opportunity_accounting: dict[str, object]
    rows: tuple[Rq3BoundaryRow, ...]

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self._payload())

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "protocol_sha256": self.protocol_sha256,
            "initial_evidence": self.initial_evidence.to_dict(),
            "source_ledger": self.source_ledger.to_dict(),
            "opportunity_accounting": self.opportunity_accounting,
            "rows": [row.to_dict() for row in self.rows],
        }

    def to_dict(self) -> dict[str, object]:
        return self._payload() | {"sha256": self.sha256}


def compile_rq3_boundary_ledger(
    root: Path,
    *,
    full_validation: bool = True,
) -> Rq3BoundaryLedger:
    root = root.resolve(strict=True)
    evidence_path = root / RQ3_INITIAL_EVIDENCE_PATH
    evidence = verify_rq3_initial_evidence(
        root,
        evidence_path,
        full_validation=full_validation,
    )
    _validate_boundary_decisions(evidence)
    source_path = root / RQ3_SOURCE_LEDGER_PATH
    source = load_rq3_post_boundary_ledger(source_path)
    if source.sha256 != RQ3_SOURCE_LEDGER_LOGICAL_SHA256:
        raise ValueError("RQ3 boundary source ledger changed")
    representations = {
        "rq3_output_learned_frozen_content": "id_frozen_content",
        "rq3_output_learned_trainable_content": "id_trainable_content",
    }
    rows = tuple(
        Rq3BoundaryRow(
            id=f"{family_id}_deep_lr_lower_boundary:{index:02d}",
            family_id=family_id,
            run_name=(
                f"g3_{family_id}_deep_lr_lower_boundary_probe_{index:02d}_native50m"
            ),
            catalog_representation=representations[family_id],
            deep_learning_rate=rate,
        )
        for family_id in RQ3_BOUNDARY_FAMILY_IDS
        for index, rate in enumerate(RQ3_BOUNDARY_DEEP_LRS, start=1)
    )
    return Rq3BoundaryLedger(
        schema_version=1,
        kind="g3_rq3_output_deep_lr_lower_boundary",
        protocol_sha256=APPROVED_PROTOCOL_SHA256,
        initial_evidence=_reference(
            root,
            evidence_path,
            logical_sha256=RQ3_INITIAL_EVIDENCE_LOGICAL_SHA256,
        ),
        source_ledger=_reference(
            root,
            source_path,
            logical_sha256=source.sha256,
        ),
        opportunity_accounting={
            "initial_logical_opportunities": 45,
            "initial_physical_jobs": 38,
            "initial_reused_rows": 7,
            "boundary_families": 2,
            "boundary_jobs_per_family": 3,
            "new_physical_jobs": 6,
            "cumulative_logical_opportunities": 51,
        },
        rows=rows,
    )


def verify_rq3_initial_evidence(
    root: Path,
    path: Path,
    *,
    full_validation: bool,
) -> dict[str, object]:
    path = _project_file(root, path)
    data = path.read_bytes()
    if (
        len(data) != RQ3_INITIAL_EVIDENCE_SIZE_BYTES
        or hashlib.sha256(data).hexdigest() != RQ3_INITIAL_EVIDENCE_FILE_SHA256
    ):
        raise ValueError("RQ3 initial evidence file changed")
    if full_validation:
        evidence = load_rq3_post_boundary_evidence(path, root=root)
    else:
        evidence = _strict_json_bytes(data)
    payload = {key: value for key, value in evidence.items() if key != "sha256"}
    if (
        evidence.get("sha256") != RQ3_INITIAL_EVIDENCE_LOGICAL_SHA256
        or _canonical_sha256(payload) != RQ3_INITIAL_EVIDENCE_LOGICAL_SHA256
    ):
        raise ValueError("RQ3 initial evidence logical identity changed")
    return evidence


def validate_rq3_boundary_ledger_document(
    document: Mapping[str, object],
    *,
    root: Path,
    full_validation: bool = False,
) -> Rq3BoundaryLedger:
    expected = compile_rq3_boundary_ledger(root, full_validation=full_validation)
    if document != expected.to_dict():
        raise ValueError("RQ3 boundary ledger differs from the approved continuation")
    return expected


def load_rq3_boundary_ledger(
    path: Path,
    *,
    root: Path,
    full_validation: bool = False,
) -> Rq3BoundaryLedger:
    document = _strict_json_bytes(path.read_bytes())
    if not isinstance(document, dict):
        raise ValueError("RQ3 boundary ledger must be an object")
    return validate_rq3_boundary_ledger_document(
        document,
        root=root,
        full_validation=full_validation,
    )


def persist_rq3_boundary_ledger(
    path: Path,
    ledger: Rq3BoundaryLedger,
) -> Path:
    content = (_canonical_json(ledger.to_dict()) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(content)
    except FileExistsError:
        if path.read_bytes() != content:
            raise RuntimeError(f"immutable RQ3 boundary ledger differs: {path}")
    return path


def _validate_boundary_decisions(evidence: Mapping[str, object]) -> None:
    selections = evidence.get("family_selections")
    if not isinstance(selections, dict) or set(selections) != {
        "rq3_output_learned",
        "rq3_output_frozen_content",
        "rq3_output_trainable_content",
        *RQ3_BOUNDARY_FAMILY_IDS,
    }:
        raise ValueError("RQ3 initial evidence family selections changed")
    expected_rows = {
        "rq3_output_learned_frozen_content": "rq3_output_learned_frozen_content:04",
        "rq3_output_learned_trainable_content": (
            "rq3_output_learned_trainable_content:04"
        ),
    }
    for family_id, row_id in expected_rows.items():
        selection = selections[family_id]
        selected = selection.get("selected") if isinstance(selection, dict) else None
        boundary = (
            selection.get("boundary_decision") if isinstance(selection, dict) else None
        )
        deep = boundary.get("deep_learning_rate") if isinstance(boundary, dict) else None
        horizon = boundary.get("horizon") if isinstance(boundary, dict) else None
        if (
            selection.get("status") != "boundary_extension_required"
            or not isinstance(selected, dict)
            or selected.get("row_id") != row_id
            or selected.get("embedding_learning_rate") != RQ3_BOUNDARY_EMBEDDING_LR
            or selected.get("deep_learning_rate") != 0.005733564587228046
            or selected.get("horizon_epochs") != RQ3_BOUNDARY_HORIZON
            or selected.get("best_epoch") != 18
            or not isinstance(deep, dict)
            or deep.get("direction") != "lower"
            or not isinstance(horizon, dict)
            or horizon.get("extend_to_epochs") is not None
        ):
            raise ValueError(f"RQ3 boundary decision changed for {family_id}")
    for family_id in set(selections) - set(RQ3_BOUNDARY_FAMILY_IDS):
        if selections[family_id].get("status") != "resolved":
            raise ValueError(f"RQ3 unexpected boundary family {family_id}")


def _reference(root: Path, path: Path, *, logical_sha256: str) -> EvidenceReference:
    path = _project_file(root, path)
    return EvidenceReference(
        path=str(path.relative_to(root)),
        size_bytes=path.stat().st_size,
        sha256=_file_sha256(path),
        logical_sha256=logical_sha256,
    )


def _project_file(root: Path, path: Path) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError("RQ3 boundary reference is not a project file")
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ValueError("RQ3 boundary reference escapes the project root")
    return resolved


def _strict_json_bytes(data: bytes) -> dict[str, object]:
    def object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise ValueError(f"non-finite JSON value {value}")

    try:
        value = json.loads(
            data,
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid RQ3 boundary JSON") from error
    if not isinstance(value, dict):
        raise ValueError("RQ3 boundary JSON must be an object")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
