from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

from experiments.g3_pretrained_item_embeddings.analysis.rq5_collection import (
    RQ5_HORIZON_EVIDENCE_PATH,
    RQ5_INITIAL_EVIDENCE_PATH,
    select_rq5_initial_winners,
)

from .constants import APPROVED_PROTOCOL_SHA256
from .rq5_initial import (
    RQ5_ARTIFACT_CONTRACTS,
    RQ5_INITIAL_LEDGER_LOGICAL_SHA256,
    RQ5_INITIAL_LEDGER_PATH,
    Rq5FileReference,
    load_rq5_initial_ledger,
    verify_rq5_initial_input_files,
)


RQ5_GLOBAL_BOUNDARY_LEDGER_PATH = (
    "experiments/g3_pretrained_item_embeddings/protocol/ledgers/"
    "rq5_global_gate_deep_lr_lower_boundary.json"
)
RQ5_INITIAL_EVIDENCE_LOGICAL_SHA256 = (
    "0998f7767cc09c06e8d49d77ab30417c80d92ad502608787a9e9f6f05fd2468f"
)
RQ5_HORIZON_EVIDENCE_LOGICAL_SHA256 = (
    "cf6aa0bc045fd7016d5310144d6007517cdcd6974490115d2dfef864f246cce9"
)


@dataclass(frozen=True)
class Rq5GlobalBoundaryJob:
    id: str
    run_name: str
    embedding_learning_rate: float
    deep_learning_rate: float
    horizon_epochs: int
    history_hidden_dim: int

    @property
    def family_id(self) -> str:
        return "rq5_global_gate"

    @property
    def content_gate(self) -> str:
        return "global"

    @property
    def gate_hidden_dim(self) -> None:
        return None

    @property
    def batch_size(self) -> int:
        return 512

    @property
    def seed(self) -> int:
        return 42

    @property
    def reused_from(self) -> None:
        return None

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "family_id": self.family_id,
            "phase": "rq5_content_gate",
            "stage": "rq5_global_gate_deep_lr_lower_boundary",
            "role": "deep_learning_rate_boundary_probe",
            "run_name": self.run_name,
            "reused_from": None,
            "representation": {
                "history_representation": "id_content",
                "history_hidden_dim": self.history_hidden_dim,
                "catalog_representation": "learned_id",
                "content_gate": "global",
                "gate_hidden_dim": None,
                "gate_input": None,
                "gate_activation": "sigmoid",
                "content_attachment": "before_id_content_densenet",
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
class Rq5GlobalBoundaryLedger:
    initial_ledger: Rq5FileReference
    initial_evidence: Rq5FileReference
    horizon_evidence: Rq5FileReference
    source_row_id: str
    rows: tuple[Rq5GlobalBoundaryJob, ...]

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self._payload())

    @property
    def physical_rows(self) -> tuple[Rq5GlobalBoundaryJob, ...]:
        return self.rows

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "kind": "g3_rq5_global_gate_deep_lr_lower_boundary",
            "protocol_sha256": APPROVED_PROTOCOL_SHA256,
            "inputs": {
                "initial_ledger": self.initial_ledger.to_dict(),
                "initial_evidence": self.initial_evidence.to_dict(),
                "frequency_horizon_evidence": self.horizon_evidence.to_dict(),
            },
            "source_selection": {
                "row_id": self.source_row_id,
                "direction": "lower_deep_learning_rate",
                "outward_divisors": [math.sqrt(2.0), 2.0, 2.0 * math.sqrt(2.0)],
                "embedding_learning_rate_fixed": True,
                "horizon_fixed": True,
                "treatment_fixed": True,
            },
            "opportunity_accounting": {
                "initial_global_logical": 12,
                "boundary_logical": 3,
                "boundary_physical": 3,
                "cumulative_global_logical": 15,
            },
            "artifact_contracts": [
                (
                    contract.to_dict()
                    if contract.name != "job_contract"
                    else contract.to_dict()
                    | {"filename": "g3_rq5_global_boundary_job.json"}
                )
                for contract in RQ5_ARTIFACT_CONTRACTS
            ],
            "rows": [row.to_dict() for row in self.rows],
        }

    def to_dict(self) -> dict[str, object]:
        return self._payload() | {"sha256": self.sha256}


def compile_rq5_global_boundary_ledger(*, root: Path) -> Rq5GlobalBoundaryLedger:
    root = root.resolve(strict=True)
    initial_ledger_path = root / RQ5_INITIAL_LEDGER_PATH
    initial = load_rq5_initial_ledger(initial_ledger_path)
    if initial.sha256 != RQ5_INITIAL_LEDGER_LOGICAL_SHA256:
        raise ValueError("RQ5 boundary initial ledger changed")
    initial_evidence_path = root / RQ5_INITIAL_EVIDENCE_PATH
    initial_evidence = _load_logical_document(
        root,
        initial_evidence_path,
        expected_sha256=RQ5_INITIAL_EVIDENCE_LOGICAL_SHA256,
    )
    if initial_evidence.get("ledger") != _reference(
        root, initial_ledger_path, initial.sha256
    ).to_dict():
        raise ValueError("RQ5 boundary initial evidence binding changed")
    selections = select_rq5_initial_winners(initial_evidence["runs"], ledger=initial)
    global_selection = selections["global_gate"]
    selected = global_selection["selected"]
    boundaries = global_selection["boundaries"]
    if (
        global_selection.get("selected_row_id") != "rq5_global_gate:10"
        or boundaries["deep_learning_rate"]["direction"] != "lower"
        or boundaries["embedding_learning_rate"]["direction"] is not None
        or boundaries["horizon"]["extend_to_epochs"] is not None
        or selected.get("horizon_epochs") != 40
        or selected.get("best_epoch") != 22
    ):
        raise ValueError("RQ5 global boundary decision changed")
    horizon_evidence_path = root / RQ5_HORIZON_EVIDENCE_PATH
    horizon_evidence = _load_logical_document(
        root,
        horizon_evidence_path,
        expected_sha256=RQ5_HORIZON_EVIDENCE_LOGICAL_SHA256,
    )
    frequency = horizon_evidence.get("frequency_selection")
    if (
        horizon_evidence.get("initial_collection")
        != _reference(
            root, initial_evidence_path, RQ5_INITIAL_EVIDENCE_LOGICAL_SHA256
        ).to_dict()
        or not isinstance(frequency, dict)
        or frequency.get("extension_required") is not False
    ):
        raise ValueError("RQ5 frequency horizon evidence is not resolved and bound")
    embedding = float(selected["embedding_learning_rate"])
    deep = float(selected["deep_learning_rate"])
    divisors = (math.sqrt(2.0), 2.0, 2.0 * math.sqrt(2.0))
    rows = tuple(
        Rq5GlobalBoundaryJob(
            id=f"rq5_global_gate:{12 + index:02d}",
            run_name=f"g3_rq5_global_gate_deep_lr_lower_probe_{index:02d}_native50m",
            embedding_learning_rate=embedding,
            deep_learning_rate=deep / divisor,
            horizon_epochs=40,
            history_hidden_dim=initial.fixed_gate.history_hidden_dim,
        )
        for index, divisor in enumerate(divisors, start=1)
    )
    ledger = Rq5GlobalBoundaryLedger(
        initial_ledger=_reference(root, initial_ledger_path, initial.sha256),
        initial_evidence=_reference(
            root, initial_evidence_path, RQ5_INITIAL_EVIDENCE_LOGICAL_SHA256
        ),
        horizon_evidence=_reference(
            root, horizon_evidence_path, RQ5_HORIZON_EVIDENCE_LOGICAL_SHA256
        ),
        source_row_id="rq5_global_gate:10",
        rows=rows,
    )
    _validate_program(ledger)
    return ledger


def load_rq5_global_boundary_ledger(
    path: Path, *, root: Path, expected_ledger_sha256: str | None = None
) -> Rq5GlobalBoundaryLedger:
    document = _load_json(path)
    payload = {name: value for name, value in document.items() if name != "sha256"}
    logical_sha256 = document.get("sha256")
    if (
        not isinstance(logical_sha256, str)
        or _canonical_sha256(payload) != logical_sha256
        or (
            expected_ledger_sha256 is not None
            and logical_sha256 != expected_ledger_sha256
        )
    ):
        raise ValueError("RQ5 global boundary ledger logical SHA changed")
    rebuilt = compile_rq5_global_boundary_ledger(root=root)
    if rebuilt.sha256 != logical_sha256 or not _exact_json_equal(
        document, rebuilt.to_dict()
    ):
        raise ValueError("RQ5 global boundary ledger differs from frozen inputs")
    return rebuilt


def persist_rq5_global_boundary_ledger(
    path: Path, ledger: Rq5GlobalBoundaryLedger, *, root: Path
) -> Path:
    root = root.resolve(strict=True)
    destination = (root / RQ5_GLOBAL_BOUNDARY_LEDGER_PATH).resolve()
    if path.resolve() != destination or destination.is_symlink():
        raise ValueError("RQ5 global boundary ledger destination is not canonical")
    rebuilt = compile_rq5_global_boundary_ledger(root=root)
    if rebuilt != ledger:
        raise ValueError("RQ5 global boundary ledger differs from authenticated inputs")
    content = (_canonical_json(ledger.to_dict()) + "\n").encode()
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("xb") as handle:
            handle.write(content)
    except FileExistsError:
        if destination.read_bytes() != content:
            raise RuntimeError(f"immutable RQ5 global boundary ledger differs: {destination}")
    return destination


def verify_rq5_global_boundary_inputs(
    root: Path, ledger: Rq5GlobalBoundaryLedger
) -> Path:
    root = root.resolve(strict=True)
    for reference in (
        ledger.initial_ledger,
        ledger.initial_evidence,
        ledger.horizon_evidence,
    ):
        if _reference(root, root / reference.path, reference.logical_sha256) != reference:
            raise ValueError(f"RQ5 global boundary input changed: {reference.path}")
    initial = load_rq5_initial_ledger(root / ledger.initial_ledger.path)
    return verify_rq5_initial_input_files(root, initial)


def _validate_program(ledger: Rq5GlobalBoundaryLedger) -> None:
    expected_deep = (
        0.008017812814887691,
        0.0056694498116914875,
        0.004008906407443846,
    )
    if (
        [row.id for row in ledger.rows]
        != ["rq5_global_gate:13", "rq5_global_gate:14", "rq5_global_gate:15"]
        or tuple(row.deep_learning_rate for row in ledger.rows) != expected_deep
        or any(
            row.embedding_learning_rate != 0.12305770976863895
            or row.horizon_epochs != 40
            or row.content_gate != "global"
            for row in ledger.rows
        )
    ):
        raise ValueError("RQ5 global boundary lost its exact three-probe design")


def _reference(root: Path, path: Path, logical_sha256: str) -> Rq5FileReference:
    if path.is_symlink():
        raise ValueError("RQ5 global boundary input must not be a symlink")
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ValueError("RQ5 global boundary input escapes project root")
    return Rq5FileReference(
        path=str(resolved.relative_to(root)),
        size_bytes=resolved.stat().st_size,
        sha256=hashlib.sha256(resolved.read_bytes()).hexdigest(),
        logical_sha256=logical_sha256,
    )


def _load_logical_document(
    root: Path, path: Path, *, expected_sha256: str
) -> dict[str, object]:
    if path.is_symlink() or not path.resolve(strict=True).is_relative_to(root):
        raise ValueError("RQ5 global boundary input path is invalid")
    document = _load_json(path)
    payload = {name: value for name, value in document.items() if name != "sha256"}
    if document.get("sha256") != expected_sha256 or _canonical_sha256(
        payload
    ) != expected_sha256:
        raise ValueError("RQ5 global boundary logical input changed")
    return document


def _load_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(
            path.read_text(),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load RQ5 global boundary JSON {path}") from error
    if not isinstance(value, dict):
        raise ValueError("RQ5 global boundary JSON must be an object")
    return value


def _exact_json_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(
            _exact_json_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _exact_json_equal(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    return left == right


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant: {value}")
