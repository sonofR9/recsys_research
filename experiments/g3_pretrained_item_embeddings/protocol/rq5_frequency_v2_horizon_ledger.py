from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from experiments.g3_pretrained_item_embeddings.analysis.rq5_frequency_v2_results import (
    RQ5_FREQUENCY_V2_INITIAL_EVIDENCE_PATH,
)

from .constants import APPROVED_PROTOCOL_SHA256
from .rq3_post_boundary import Rq3ArtifactContract
from .rq5_frequency_v2_ledger import (
    RQ5_FREQUENCY_V2_ARTIFACT_CONTRACTS,
    RQ5_FREQUENCY_V2_LEDGER_PATH,
    Rq5FrequencyV2Ledger,
    load_rq5_frequency_v2_ledger,
    verify_rq5_frequency_v2_inputs,
)
from .rq5_initial import Rq5FileReference


RQ5_FREQUENCY_V2_HORIZON_LEDGER_PATH = (
    "experiments/g3_pretrained_item_embeddings/protocol/ledgers/"
    "rq5_frequency_gate_fp32_p09_v2_horizons.json"
)
RQ5_FREQUENCY_V2_INITIAL_LEDGER_LOGICAL_SHA256 = (
    "c7745c8ee5b2683c9e4001924926d1a92224a220b87269a36130224c8cfde8f4"
)
RQ5_FREQUENCY_V2_INITIAL_EVIDENCE_LOGICAL_SHA256 = (
    "a157fdf535c879438809285dc914ca12b2912d5b5cf207a093978eee3c375b84"
)
RQ5_FREQUENCY_V2_HORIZON_ARTIFACT_CONTRACTS = tuple(
    contract
    if contract.name != "job_contract"
    else Rq3ArtifactContract(
        "job_contract",
        "g3_rq5_frequency_v2_horizon_job.json",
        contract.required_keys,
        contract.schema_versions,
    )
    for contract in RQ5_FREQUENCY_V2_ARTIFACT_CONTRACTS
)


@dataclass(frozen=True)
class Rq5FrequencyV2HorizonJob:
    id: str
    run_name: str
    embedding_learning_rate: float
    deep_learning_rate: float
    horizon_epochs: int

    @property
    def family_id(self) -> str:
        return "rq5_frequency_gate_v2"

    @property
    def content_gate(self) -> str:
        return "frequency"

    @property
    def gate_hidden_dim(self) -> int:
        return 8

    @property
    def history_hidden_dim(self) -> int:
        return 128

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
            "stage": "rq5_frequency_gate_fp32_p09_v2_horizons",
            "role": "selected_width_horizon_cell",
            "run_name": self.run_name,
            "representation": {
                "history_representation": "id_content",
                "history_hidden_dim": self.history_hidden_dim,
                "catalog_representation": "learned_id",
                "content_gate": self.content_gate,
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
class Rq5FrequencyV2HorizonLedger:
    initial_ledger: Rq5FileReference
    initial_evidence: Rq5FileReference
    rows: tuple[Rq5FrequencyV2HorizonJob, ...]

    @property
    def sha256(self) -> str:
        return _sha(self._payload())

    @property
    def physical_rows(self) -> tuple[Rq5FrequencyV2HorizonJob, ...]:
        return self.rows

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "kind": "g3_rq5_frequency_gate_fp32_p09_v2_horizons",
            "protocol_sha256": APPROVED_PROTOCOL_SHA256,
            "inputs": {
                "initial_ledger": self.initial_ledger.to_dict(),
                "initial_evidence": self.initial_evidence.to_dict(),
            },
            "source_selection": {
                "selected_row_id": "rq5_frequency_gate_v2:04",
                "selected_gate_hidden_dim": 8,
                "capacity_boundary": None,
                "embedding_learning_rate_boundary": "lower",
                "deep_learning_rate_boundary": None,
                "restored_best_epoch_at_cap": True,
            },
            "opportunity_accounting": {
                "initial": 9,
                "horizon": 3,
                "cumulative": 12,
                "horizons": [15, 25, 40],
            },
            "artifact_contracts": [
                contract.to_dict()
                for contract in RQ5_FREQUENCY_V2_HORIZON_ARTIFACT_CONTRACTS
            ],
            "rows": [row.to_dict() for row in self.rows],
        }

    def to_dict(self) -> dict[str, object]:
        return self._payload() | {"sha256": self.sha256}


def compile_rq5_frequency_v2_horizon_ledger(
    *, root: Path
) -> Rq5FrequencyV2HorizonLedger:
    root = root.resolve(strict=True)
    initial_path = root / RQ5_FREQUENCY_V2_LEDGER_PATH
    initial = load_rq5_frequency_v2_ledger(
        initial_path,
        root=root,
        expected_ledger_sha256=RQ5_FREQUENCY_V2_INITIAL_LEDGER_LOGICAL_SHA256,
    )
    evidence_path = root / RQ5_FREQUENCY_V2_INITIAL_EVIDENCE_PATH
    evidence = _logical(evidence_path)
    selection = evidence.get("selection")
    if (
        evidence.get("sha256") != RQ5_FREQUENCY_V2_INITIAL_EVIDENCE_LOGICAL_SHA256
        or evidence.get("ledger") != _reference(root, initial_path, initial.sha256).to_dict()
        or not isinstance(selection, dict)
        or selection.get("selected_row_id") != "rq5_frequency_gate_v2:04"
        or selection.get("selected_gate_hidden_dim") != 8
        or selection.get("capacity_boundary", {}).get("direction") is not None
        or selection.get("coordinate_boundaries", {})
        .get("embedding_learning_rate", {})
        .get("direction")
        != "lower"
        or selection.get("coordinate_boundaries", {})
        .get("deep_learning_rate", {})
        .get("direction")
        is not None
        or selection.get("coordinate_boundaries", {})
        .get("horizon", {})
        .get("restored_best_epoch")
        != 25
    ):
        raise ValueError("RQ5 frequency v2 selected-width decision changed")
    coordinates = (
        (15, 0.047134737607146836, 0.04127129308065626),
        (25, 0.12447135415265811, 0.023941907610393703),
        (40, 0.3041556165944196, 0.014506684820055783),
    )
    rows = tuple(
        Rq5FrequencyV2HorizonJob(
            id=f"rq5_frequency_gate_v2:{9 + index:02d}",
            run_name=(
                f"g3_rq5_frequency_gate_v2_width_8_horizon_{horizon}_"
                f"cell_{index:02d}_native50m"
            ),
            embedding_learning_rate=embedding,
            deep_learning_rate=deep,
            horizon_epochs=horizon,
        )
        for index, (horizon, embedding, deep) in enumerate(coordinates, start=1)
    )
    return Rq5FrequencyV2HorizonLedger(
        initial_ledger=_reference(root, initial_path, initial.sha256),
        initial_evidence=_reference(root, evidence_path, str(evidence["sha256"])),
        rows=rows,
    )


def load_rq5_frequency_v2_horizon_ledger(
    path: Path, *, root: Path, expected_ledger_sha256: str | None = None
) -> Rq5FrequencyV2HorizonLedger:
    document = _json(path)
    payload = {key: value for key, value in document.items() if key != "sha256"}
    logical = document.get("sha256")
    if not isinstance(logical, str) or _sha(payload) != logical or (
        expected_ledger_sha256 is not None and logical != expected_ledger_sha256
    ):
        raise ValueError("RQ5 frequency v2 horizon ledger logical SHA changed")
    rebuilt = compile_rq5_frequency_v2_horizon_ledger(root=root)
    if rebuilt.sha256 != logical or not _exact(document, rebuilt.to_dict()):
        raise ValueError("RQ5 frequency v2 horizon ledger differs from frozen inputs")
    return rebuilt


def persist_rq5_frequency_v2_horizon_ledger(
    path: Path, ledger: Rq5FrequencyV2HorizonLedger, *, root: Path
) -> Path:
    root = root.resolve(strict=True)
    destination = (root / RQ5_FREQUENCY_V2_HORIZON_LEDGER_PATH).resolve()
    if path.resolve() != destination or destination.is_symlink():
        raise ValueError("RQ5 frequency v2 horizon destination is not canonical")
    if compile_rq5_frequency_v2_horizon_ledger(root=root) != ledger:
        raise ValueError("RQ5 frequency v2 horizon inputs changed")
    content = (_canonical(ledger.to_dict()) + "\n").encode()
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("xb") as handle:
            handle.write(content)
    except FileExistsError:
        if destination.read_bytes() != content:
            raise RuntimeError(f"immutable RQ5 frequency v2 horizon ledger differs: {destination}")
    return destination


def verify_rq5_frequency_v2_horizon_inputs(
    root: Path, ledger: Rq5FrequencyV2HorizonLedger
) -> Path:
    if compile_rq5_frequency_v2_horizon_ledger(root=root) != ledger:
        raise ValueError("RQ5 frequency v2 horizon bound inputs changed")
    initial = load_rq5_frequency_v2_ledger(
        root / ledger.initial_ledger.path,
        root=root,
        expected_ledger_sha256=ledger.initial_ledger.logical_sha256,
    )
    return verify_rq5_frequency_v2_inputs(root, initial)


def _reference(root: Path, path: Path, logical: str) -> Rq5FileReference:
    resolved = path.resolve(strict=True)
    return Rq5FileReference(
        path=str(resolved.relative_to(root)),
        size_bytes=resolved.stat().st_size,
        sha256=hashlib.sha256(resolved.read_bytes()).hexdigest(),
        logical_sha256=logical,
    )


def _logical(path: Path) -> dict[str, object]:
    document = _json(path)
    payload = {key: value for key, value in document.items() if key != "sha256"}
    if document.get("sha256") != _sha(payload):
        raise ValueError("RQ5 frequency v2 horizon logical input changed")
    return document


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(), object_pairs_hook=_pairs, parse_constant=_constant)
    if not isinstance(value, dict):
        raise ValueError("RQ5 frequency v2 horizon JSON must be an object")
    return value


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


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
        return set(left) == set(right) and all(_exact(left[key], right[key]) for key in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _exact(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right
