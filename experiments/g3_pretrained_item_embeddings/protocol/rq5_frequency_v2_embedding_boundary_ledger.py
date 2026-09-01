from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from .constants import APPROVED_PROTOCOL_SHA256
from .rq3_post_boundary import Rq3ArtifactContract
from .rq5_frequency_v2_horizon_ledger import (
    RQ5_FREQUENCY_V2_HORIZON_ARTIFACT_CONTRACTS,
    RQ5_FREQUENCY_V2_HORIZON_LEDGER_PATH,
    Rq5FrequencyV2HorizonLedger,
    load_rq5_frequency_v2_horizon_ledger,
    verify_rq5_frequency_v2_horizon_inputs,
)
from .rq5_initial import Rq5FileReference


RQ5_FREQUENCY_V2_EMBEDDING_BOUNDARY_LEDGER_PATH = (
    "experiments/g3_pretrained_item_embeddings/protocol/ledgers/"
    "rq5_frequency_gate_fp32_p09_v2_embedding_lr_upper_boundary.json"
)
RQ5_FREQUENCY_V2_HORIZON_LEDGER_LOGICAL_SHA256 = (
    "9d3c5ff0b62fc93139698080f5e6766778a2b98533d548fa72f9c5a84eab2f10"
)
RQ5_FREQUENCY_V2_HORIZON_EVIDENCE_PATH = (
    "experiments/g3_pretrained_item_embeddings/evidence/"
    "rq5_frequency_gate_fp32_p09_v2_horizons_native50m.json"
)
RQ5_FREQUENCY_V2_HORIZON_EVIDENCE_LOGICAL_SHA256 = (
    "8ea522f0a0c05418649b262ef06e7ad84b4f9beb562c9813834ac93f35e8b272"
)
RQ5_OUTCOME_EVIDENCE_PATH = (
    "experiments/g3_pretrained_item_embeddings/evidence/"
    "rq5_outcome_premechanism_native50m.json"
)
RQ5_OUTCOME_EVIDENCE_LOGICAL_SHA256 = (
    "a42314f93ffad3c51d6ea43a7e7c07d87486e6cadb7bb03f60c75be98ca4d442"
)
RQ5_FREQUENCY_V2_EMBEDDING_BOUNDARY_ARTIFACT_CONTRACTS = tuple(
    contract
    if contract.name != "job_contract"
    else Rq3ArtifactContract(
        "job_contract",
        "g3_rq5_frequency_v2_embedding_boundary_job.json",
        contract.required_keys,
        contract.schema_versions,
    )
    for contract in RQ5_FREQUENCY_V2_HORIZON_ARTIFACT_CONTRACTS
)


@dataclass(frozen=True)
class Rq5FrequencyV2EmbeddingBoundaryJob:
    id: str
    run_name: str
    embedding_learning_rate: float

    @property
    def family_id(self) -> str:
        return "rq5_frequency_gate_v2"

    @property
    def content_gate(self) -> str:
        return "frequency"

    @property
    def history_hidden_dim(self) -> int:
        return 128

    @property
    def deep_learning_rate(self) -> float:
        return 0.014506684820055783

    @property
    def gate_hidden_dim(self) -> int:
        return 8

    @property
    def horizon_epochs(self) -> int:
        return 40

    @property
    def batch_size(self) -> int:
        return 512

    @property
    def seed(self) -> int:
        return 42

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "family_id": self.family_id,
            "phase": "rq5_content_gate",
            "stage": "rq5_frequency_gate_fp32_p09_v2_embedding_lr_upper_boundary",
            "role": "renewed_approval_embedding_lr_upper_boundary_probe",
            "run_name": self.run_name,
            "representation": {
                "history_representation": "id_content",
                "history_hidden_dim": 128,
                "catalog_representation": "learned_id",
                "content_gate": "frequency",
                "gate_hidden_dim": self.gate_hidden_dim,
                "gate_input": "standardized_log1p_training_count",
                "gate_activation": "sigmoid",
                "content_attachment": "before_id_content_densenet",
                "frequency_gate_semantics": "fp32_p09_v2",
                "initial_probability": 0.9,
                "math_dtype": "float32",
            },
            "dataset": {
                "size": "native-50m",
                "source": "likes",
                "event_limit": 50_000_000,
                "sampling": "none",
                "batch_size": self.batch_size,
                "seed": self.seed,
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
class Rq5FrequencyV2EmbeddingBoundaryLedger:
    horizon_ledger: Rq5FileReference
    horizon_evidence: Rq5FileReference
    premechanism_outcome: Rq5FileReference
    rows: tuple[Rq5FrequencyV2EmbeddingBoundaryJob, ...]

    @property
    def sha256(self) -> str:
        return _sha(self._payload())

    @property
    def physical_rows(self) -> tuple[Rq5FrequencyV2EmbeddingBoundaryJob, ...]:
        return self.rows

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "kind": "g3_rq5_frequency_gate_fp32_p09_v2_embedding_lr_upper_boundary",
            "protocol_sha256": APPROVED_PROTOCOL_SHA256,
            "inputs": {
                "horizon_ledger": self.horizon_ledger.to_dict(),
                "horizon_evidence": self.horizon_evidence.to_dict(),
                "premechanism_outcome": self.premechanism_outcome.to_dict(),
            },
            "approval": {
                "decision": "exactly_three_upper_embedding_lr_jobs",
                "further_jobs_require_renewed_approval": True,
            },
            "frozen_coordinate": {
                "source_row_id": "rq5_frequency_gate_v2:12",
                "source_embedding_learning_rate": 0.3041556165944196,
                "embedding_learning_rates": [
                    0.4301409980597794,
                    0.6083112331888392,
                    0.8602819961195588,
                ],
                "deep_learning_rate": 0.014506684820055783,
                "gate_hidden_dim": 8,
                "horizon_epochs": 40,
                "batch_size": 512,
                "seed": 42,
                "frequency_gate_semantics": "fp32_p09_v2",
            },
            "opportunity_accounting": {
                "prior_valid_corrected_frequency_rows": 12,
                "new_logical": 3,
                "new_physical": 3,
                "combined_valid_corrected_frequency_rows": 15,
            },
            "artifact_contracts": [
                contract.to_dict()
                for contract in RQ5_FREQUENCY_V2_EMBEDDING_BOUNDARY_ARTIFACT_CONTRACTS
            ],
            "rows": [row.to_dict() for row in self.rows],
        }

    def to_dict(self) -> dict[str, object]:
        return self._payload() | {"sha256": self.sha256}


def compile_rq5_frequency_v2_embedding_boundary_ledger(
    root: Path,
) -> Rq5FrequencyV2EmbeddingBoundaryLedger:
    root = root.resolve(strict=True)
    horizon_ledger_path = root / RQ5_FREQUENCY_V2_HORIZON_LEDGER_PATH
    horizon_ledger = load_rq5_frequency_v2_horizon_ledger(
        horizon_ledger_path,
        root=root,
        expected_ledger_sha256=RQ5_FREQUENCY_V2_HORIZON_LEDGER_LOGICAL_SHA256,
    )
    horizon_evidence_path = root / RQ5_FREQUENCY_V2_HORIZON_EVIDENCE_PATH
    horizon_evidence = _load_logical(
        horizon_evidence_path,
        RQ5_FREQUENCY_V2_HORIZON_EVIDENCE_LOGICAL_SHA256,
    )
    _validate_horizon_selection(horizon_evidence)
    outcome_path = root / RQ5_OUTCOME_EVIDENCE_PATH
    outcome = _load_logical(outcome_path, RQ5_OUTCOME_EVIDENCE_LOGICAL_SHA256)
    _validate_comparators(outcome)
    rates = (
        0.4301409980597794,
        0.6083112331888392,
        0.8602819961195588,
    )
    rows = tuple(
        Rq5FrequencyV2EmbeddingBoundaryJob(
            id=f"rq5_frequency_gate_v2:{12 + index:02d}",
            run_name=(
                "g3_rq5_frequency_gate_v2_width_8_horizon_40_"
                f"embedding_lr_upper_probe_{index:02d}_native50m"
            ),
            embedding_learning_rate=rate,
        )
        for index, rate in enumerate(rates, start=1)
    )
    return Rq5FrequencyV2EmbeddingBoundaryLedger(
        horizon_ledger=_reference(root, horizon_ledger_path, horizon_ledger.sha256),
        horizon_evidence=_reference(
            root,
            horizon_evidence_path,
            RQ5_FREQUENCY_V2_HORIZON_EVIDENCE_LOGICAL_SHA256,
        ),
        premechanism_outcome=_reference(
            root, outcome_path, RQ5_OUTCOME_EVIDENCE_LOGICAL_SHA256
        ),
        rows=rows,
    )


def load_rq5_frequency_v2_embedding_boundary_ledger(
    path: Path, *, root: Path, expected_ledger_sha256: str
) -> Rq5FrequencyV2EmbeddingBoundaryLedger:
    _require_canonical(path, root=root, strict=True)
    document = _load_json(path)
    expected = compile_rq5_frequency_v2_embedding_boundary_ledger(root)
    payload = {key: value for key, value in document.items() if key != "sha256"}
    if (
        expected.sha256 != expected_ledger_sha256
        or document.get("sha256") != expected_ledger_sha256
        or _sha(payload) != expected_ledger_sha256
        or not _exact(document, expected.to_dict())
    ):
        raise ValueError("RQ5 frequency v2 embedding boundary ledger changed")
    return expected


def persist_rq5_frequency_v2_embedding_boundary_ledger(
    path: Path,
    ledger: Rq5FrequencyV2EmbeddingBoundaryLedger,
    *,
    root: Path,
) -> Path:
    _require_canonical(path, root=root, strict=False)
    if ledger != compile_rq5_frequency_v2_embedding_boundary_ledger(root):
        raise ValueError("RQ5 frequency v2 embedding boundary ledger changed")
    content = (_canonical(ledger.to_dict()) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(content)
    except FileExistsError:
        if path.read_bytes() != content:
            raise RuntimeError(f"immutable RQ5 embedding boundary ledger differs: {path}")
    return path


def verify_rq5_frequency_v2_embedding_boundary_inputs(
    root: Path, ledger: Rq5FrequencyV2EmbeddingBoundaryLedger
) -> Path:
    if ledger != compile_rq5_frequency_v2_embedding_boundary_ledger(root):
        raise ValueError("RQ5 frequency v2 embedding boundary inputs changed")
    horizon = load_rq5_frequency_v2_horizon_ledger(
        root / ledger.horizon_ledger.path,
        root=root,
        expected_ledger_sha256=ledger.horizon_ledger.logical_sha256,
    )
    return verify_rq5_frequency_v2_horizon_inputs(root, horizon)


def _validate_horizon_selection(document: dict[str, object]) -> None:
    selection = document.get("combined_selection")
    boundaries = selection.get("boundaries") if isinstance(selection, dict) else None
    selected = selection.get("selected") if isinstance(selection, dict) else None
    if (
        not isinstance(boundaries, dict)
        or not isinstance(selected, dict)
        or selection.get("selected_row_id") != "rq5_frequency_gate_v2:12"
        or selection.get("selected_gate_hidden_dim") != 8
        or selection.get("second_boundary_unresolved") is not True
        or selection.get("next_action") != "renewed_approval"
        or boundaries.get("embedding_learning_rate", {}).get("direction") != "upper"
        or boundaries.get("deep_learning_rate", {}).get("direction") is not None
        or boundaries.get("horizon", {}).get("extend_to_epochs") is not None
        or selected.get("embedding_learning_rate") != 0.3041556165944196
        or selected.get("deep_learning_rate") != 0.014506684820055783
        or selected.get("gate_hidden_dim") != 8
        or selected.get("horizon_epochs") != 40
    ):
        raise ValueError("RQ5 corrected-frequency horizon selection changed")


def _validate_comparators(document: dict[str, object]) -> None:
    fixed = document.get("fixed_comparator")
    global_selection = document.get("global_selection")
    global_selected = (
        global_selection.get("selected") if isinstance(global_selection, dict) else None
    )
    if (
        not isinstance(fixed, dict)
        or fixed.get("row_id") != "rq2_unexpected_diagnostic:03"
        or not isinstance(global_selected, dict)
        or global_selection.get("selection_resolved") is not True
        or global_selected.get("row_id") != "rq5_global_gate:10"
    ):
        raise ValueError("RQ5 fixed/global comparator selection changed")


def _reference(root: Path, path: Path, logical_sha256: str) -> Rq5FileReference:
    resolved = path.resolve(strict=True)
    if path.is_symlink() or not resolved.is_relative_to(root):
        raise ValueError("RQ5 embedding boundary input is invalid")
    return Rq5FileReference(
        path=str(resolved.relative_to(root)),
        size_bytes=resolved.stat().st_size,
        sha256=hashlib.sha256(resolved.read_bytes()).hexdigest(),
        logical_sha256=logical_sha256,
    )


def _load_logical(path: Path, expected: str) -> dict[str, object]:
    document = _load_json(path)
    payload = {key: value for key, value in document.items() if key != "sha256"}
    if document.get("sha256") != expected or _sha(payload) != expected:
        raise ValueError(f"RQ5 logical input changed: {path}")
    return document


def _require_canonical(path: Path, *, root: Path, strict: bool) -> None:
    expected = root.resolve(strict=True) / RQ5_FREQUENCY_V2_EMBEDDING_BOUNDARY_LEDGER_PATH
    if path.is_symlink() or path.resolve(strict=strict) != expected.resolve(strict=strict):
        raise ValueError("RQ5 embedding boundary ledger must use its canonical path")


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(
        path.read_text(),
        object_pairs_hook=_pairs,
        parse_constant=_constant,
    )
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant: {value}")


def _exact(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _exact(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _exact(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right
