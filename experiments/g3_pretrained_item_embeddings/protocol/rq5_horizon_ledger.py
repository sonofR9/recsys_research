from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping

from experiments.g3_pretrained_item_embeddings.analysis.control_calibration import (
    load_control_calibration,
)
from experiments.g3_pretrained_item_embeddings.analysis.rq5_collection import (
    RQ5_INITIAL_EVIDENCE_PATH,
    select_rq5_initial_winners,
)

from .constants import APPROVED_PROTOCOL_SHA256
from .rq5_initial import (
    RQ5_ARTIFACT_CONTRACTS,
    RQ5_INITIAL_LEDGER_LOGICAL_SHA256,
    RQ5_INITIAL_LEDGER_PATH,
    Rq5FileReference,
    Rq5InitialLedger,
    load_rq5_initial_ledger,
    verify_rq5_initial_input_files,
)
from .search import (
    APPROVED_FAMILY_SPECS,
    TransferredHorizonRate,
    compile_capacity_first_stage,
    compile_capacity_horizon_followup,
)


RQ5_INITIAL_EVIDENCE_LOGICAL_SHA256 = (
    "0998f7767cc09c06e8d49d77ab30417c80d92ad502608787a9e9f6f05fd2468f"
)
RQ5_HORIZON_LEDGER_PATH = (
    "experiments/g3_pretrained_item_embeddings/protocol/ledgers/"
    "rq5_frequency_horizon_post_capacity.json"
)
CONTROL_CALIBRATION_PATH = (
    "experiments/g3_pretrained_item_embeddings/evidence/"
    "untied_control_calibration.json"
)
CONTROL_CALIBRATION_SHA256 = (
    "015c94a182bc0df4179092e098e69b9b12c4fc62474ff4a2f15ad5d3e693e896"
)


@dataclass(frozen=True)
class Rq5HorizonJob:
    id: str
    family_id: str
    run_name: str
    batch_size: int
    seed: int
    embedding_learning_rate: float
    deep_learning_rate: float
    horizon_epochs: int
    history_hidden_dim: int
    gate_hidden_dim: int
    reused_from: str | None = None

    @property
    def content_gate(self) -> str:
        return "frequency"

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "family_id": self.family_id,
            "phase": "rq5_content_gate",
            "stage": "rq5_frequency_horizon_post_capacity",
            "role": "horizon_probe",
            "run_name": self.run_name,
            "reused_from": self.reused_from,
            "representation": {
                "history_representation": "id_content",
                "history_hidden_dim": self.history_hidden_dim,
                "catalog_representation": "learned_id",
                "content_gate": "frequency",
                "gate_hidden_dim": self.gate_hidden_dim,
                "gate_input": "standardized_log1p_training_count",
                "gate_activation": "sigmoid",
                "content_attachment": "before_id_content_densenet",
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
class Rq5HorizonLedger:
    initial_ledger: Rq5FileReference
    initial_collection: Rq5FileReference
    control_calibration: Rq5FileReference
    initial_batch_id: str
    selected_gate_hidden_dim: int
    history_hidden_dim: int
    rows: tuple[Rq5HorizonJob, ...]

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self._payload())

    @property
    def physical_rows(self) -> tuple[Rq5HorizonJob, ...]:
        return tuple(row for row in self.rows if row.reused_from is None)

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "kind": "g3_rq5_frequency_horizon_post_capacity",
            "protocol_sha256": APPROVED_PROTOCOL_SHA256,
            "inputs": {
                "initial_ledger": self.initial_ledger.to_dict(),
                "initial_collection": self.initial_collection.to_dict(),
                "control_calibration": self.control_calibration.to_dict(),
            },
            "initial_batch_id": self.initial_batch_id,
            "selected_gate_hidden_dim": self.selected_gate_hidden_dim,
            "history_hidden_dim": self.history_hidden_dim,
            "opportunity_accounting": {
                "initial_logical": 9,
                "followup_logical": 3,
                "cumulative_logical": 12,
                "followup_physical": len(self.physical_rows),
            },
            "artifact_contracts": [
                (
                    contract.to_dict()
                    if contract.name != "job_contract"
                    else contract.to_dict() | {"filename": "g3_rq5_horizon_job.json"}
                )
                for contract in RQ5_ARTIFACT_CONTRACTS
            ],
            "rows": [row.to_dict() for row in self.rows],
        }

    def to_dict(self) -> dict[str, object]:
        return self._payload() | {"sha256": self.sha256}


def compile_rq5_horizon_ledger(
    *,
    root: Path,
    initial_ledger_path: Path,
    initial_collection_path: Path,
    expected_initial_collection_sha256: str,
) -> Rq5HorizonLedger:
    root = root.resolve(strict=True)
    initial = load_rq5_initial_ledger(initial_ledger_path)
    if initial.sha256 != RQ5_INITIAL_LEDGER_LOGICAL_SHA256:
        raise ValueError("RQ5 horizon initial ledger differs from the approved ledger")
    collection = _load_logical_document(
        root,
        initial_collection_path,
        expected_sha256=expected_initial_collection_sha256,
    )
    initial_reference = _reference(root, initial_ledger_path, initial.sha256)
    if (
        collection.get("kind") != "g3_rq5_initial_collection"
        or collection.get("protocol_sha256") != APPROVED_PROTOCOL_SHA256
        or collection.get("ledger") != initial_reference.to_dict()
        or not isinstance(collection.get("runs"), list)
    ):
        raise ValueError("RQ5 initial collection is not bound to the approved ledger")
    selections = select_rq5_initial_winners(collection["runs"], ledger=initial)
    frequency = selections["frequency_capacity"]
    boundary = frequency["capacity_boundary"]
    capacity = frequency["selected_gate_hidden_dim"]
    if capacity != 8 or boundary.get("direction") is not None:
        raise ValueError("RQ5 frequency capacity is not interior and resolved")
    calibration_path = root / CONTROL_CALIBRATION_PATH
    calibration = load_control_calibration(calibration_path)
    if calibration.get("sha256") != CONTROL_CALIBRATION_SHA256:
        raise ValueError("RQ5 control calibration changed")
    rows = _jobs(initial, capacity=capacity, rates=_transferred_rates(calibration))
    ledger = Rq5HorizonLedger(
        initial_ledger=initial_reference,
        initial_collection=_reference(
            root, initial_collection_path, expected_initial_collection_sha256
        ),
        control_calibration=_reference(
            root, calibration_path, CONTROL_CALIBRATION_SHA256
        ),
        initial_batch_id=str(collection["queue_batch"]["batch_id"]),
        selected_gate_hidden_dim=capacity,
        history_hidden_dim=initial.fixed_gate.history_hidden_dim,
        rows=rows,
    )
    _validate_program(ledger)
    return ledger


def load_rq5_horizon_ledger(
    path: Path, *, root: Path, expected_ledger_sha256: str | None = None
) -> Rq5HorizonLedger:
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
        raise ValueError("RQ5 horizon ledger logical SHA changed")
    inputs = document.get("inputs")
    initial = inputs.get("initial_ledger") if isinstance(inputs, dict) else None
    collection = inputs.get("initial_collection") if isinstance(inputs, dict) else None
    if not isinstance(initial, dict) or not isinstance(collection, dict):
        raise ValueError("RQ5 horizon ledger input bindings are absent")
    rebuilt = compile_rq5_horizon_ledger(
        root=root,
        initial_ledger_path=root / str(initial.get("path")),
        initial_collection_path=root / str(collection.get("path")),
        expected_initial_collection_sha256=str(collection.get("logical_sha256")),
    )
    if (
        rebuilt.sha256 != logical_sha256
        or not _exact_json_equal(document, rebuilt.to_dict())
    ):
        raise ValueError("RQ5 horizon ledger differs from its frozen inputs")
    return rebuilt


def persist_rq5_horizon_ledger(
    path: Path, ledger: Rq5HorizonLedger, *, root: Path
) -> Path:
    root = root.resolve(strict=True)
    destination = (root / RQ5_HORIZON_LEDGER_PATH).resolve()
    if path.resolve() != destination or destination.is_symlink():
        raise ValueError("RQ5 horizon ledger destination is not canonical")
    rebuilt = compile_rq5_horizon_ledger(
        root=root,
        initial_ledger_path=root / ledger.initial_ledger.path,
        initial_collection_path=root / ledger.initial_collection.path,
        expected_initial_collection_sha256=ledger.initial_collection.logical_sha256,
    )
    if rebuilt != ledger:
        raise ValueError("RQ5 horizon ledger differs from authenticated inputs")
    content = (_canonical_json(ledger.to_dict()) + "\n").encode()
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("xb") as handle:
            handle.write(content)
    except FileExistsError:
        if destination.read_bytes() != content:
            raise RuntimeError(f"immutable RQ5 horizon ledger differs: {destination}")
    return destination


def verify_rq5_horizon_input_files(
    root: Path, ledger: Rq5HorizonLedger
) -> Path:
    root = root.resolve(strict=True)
    for reference in (
        ledger.initial_ledger,
        ledger.initial_collection,
        ledger.control_calibration,
    ):
        if _reference(root, root / reference.path, reference.logical_sha256) != reference:
            raise ValueError(f"RQ5 horizon bound file changed: {reference.path}")
    initial = load_rq5_initial_ledger(
        root / ledger.initial_ledger.path,
    )
    if initial.sha256 != RQ5_INITIAL_LEDGER_LOGICAL_SHA256:
        raise ValueError("RQ5 horizon initial ledger logical SHA changed")
    return verify_rq5_initial_input_files(root, initial)


def _transferred_rates(
    calibration: Mapping[str, object],
) -> tuple[TransferredHorizonRate, ...]:
    fits = calibration.get("power_law_fits")
    embedding = fits.get("embedding_learning_rate") if isinstance(fits, dict) else None
    deep = fits.get("deep_learning_rate") if isinstance(fits, dict) else None
    embedding_values = embedding.get("fitted_coordinates") if isinstance(embedding, dict) else None
    deep_values = deep.get("fitted_coordinates") if isinstance(deep, dict) else None
    if not isinstance(embedding_values, dict) or not isinstance(deep_values, dict):
        raise ValueError("RQ5 horizon search requires accepted transferred rates")
    rates = tuple(
        TransferredHorizonRate(
            horizon_epochs=horizon,
            embedding_learning_rate=float(embedding_values[str(horizon)]),
            deep_learning_rate=float(deep_values[str(horizon)]),
        )
        for horizon in (15, 25, 40)
    )
    expected = (
        (15, 0.047134737607146836, 0.04127129308065626),
        (25, 0.12447135415265811, 0.023941907610393703),
        (40, 0.3041556165944196, 0.014506684820055783),
    )
    if tuple(
        (rate.horizon_epochs, rate.embedding_learning_rate, rate.deep_learning_rate)
        for rate in rates
    ) != expected:
        raise ValueError("RQ5 transferred horizon rates changed")
    return rates


def _jobs(
    initial: Rq5InitialLedger,
    *,
    capacity: int,
    rates: tuple[TransferredHorizonRate, ...],
) -> tuple[Rq5HorizonJob, ...]:
    specification = next(
        spec for spec in APPROVED_FAMILY_SPECS if spec.id == "rq5_frequency_gate"
    )
    first_stage = compile_capacity_first_stage(specification)
    initial_rows = [
        row for row in initial.rows if row.family_id == "rq5_frequency_gate"
    ]
    if len(initial_rows) != len(first_stage) or any(
        (
            row.id,
            row.embedding_learning_rate,
            row.deep_learning_rate,
            row.horizon_epochs,
            row.gate_hidden_dim,
        )
        != (
            coordinate.id,
            coordinate.embedding_learning_rate,
            coordinate.deep_learning_rate,
            coordinate.horizon_epochs,
            coordinate.capacity,
        )
        for row, coordinate in zip(initial_rows, first_stage, strict=True)
    ):
        raise ValueError("RQ5 horizon initial capacity surface changed")
    followup = compile_capacity_horizon_followup(
        specification,
        selected_capacity=capacity,
        transferred_horizon_rates=rates,
        first_stage=first_stage,
    )
    by_id = {row.id: row for row in initial_rows}
    rows = []
    for coordinate in followup:
        source = by_id.get(coordinate.reused_from)
        if coordinate.reused_from is not None and source is None:
            raise ValueError("RQ5 horizon reuse source is absent")
        run_name = (
            source.run_name
            if source is not None
            else (
                f"g3_rq5_frequency_gate_width_{capacity}_trial_"
                f"{coordinate.opportunity_index + 1:02d}_native50m"
            )
        )
        rows.append(
            Rq5HorizonJob(
                id=coordinate.id,
                family_id=coordinate.family_id,
                run_name=run_name,
                batch_size=coordinate.batch_size,
                seed=coordinate.seed,
                embedding_learning_rate=coordinate.embedding_learning_rate,
                deep_learning_rate=coordinate.deep_learning_rate,
                horizon_epochs=coordinate.horizon_epochs,
                history_hidden_dim=initial.fixed_gate.history_hidden_dim,
                gate_hidden_dim=capacity,
                reused_from=coordinate.reused_from,
            )
        )
    return tuple(rows)


def _validate_program(ledger: Rq5HorizonLedger) -> None:
    if (
        ledger.selected_gate_hidden_dim != 8
        or ledger.history_hidden_dim < 1
        or len(ledger.rows) != 3
        or len(ledger.physical_rows) != 3
        or [row.id for row in ledger.rows]
        != ["rq5_frequency_gate:10", "rq5_frequency_gate:11", "rq5_frequency_gate:12"]
        or [row.horizon_epochs for row in ledger.rows] != [15, 25, 40]
        or any(row.gate_hidden_dim != 8 for row in ledger.rows)
    ):
        raise ValueError("RQ5 horizon ledger lost its exact three-cell design")


def _reference(
    root: Path, path: Path, logical_sha256: str
) -> Rq5FileReference:
    if path.is_symlink():
        raise ValueError("RQ5 horizon input must not be a symlink")
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ValueError("RQ5 horizon input escapes the project root")
    return Rq5FileReference(
        path=str(resolved.relative_to(root)),
        size_bytes=resolved.stat().st_size,
        sha256=hashlib.sha256(resolved.read_bytes()).hexdigest(),
        logical_sha256=logical_sha256,
    )


def _load_logical_document(
    root: Path, path: Path, *, expected_sha256: str
) -> dict[str, object]:
    document = _load_json(path)
    payload = {name: value for name, value in document.items() if name != "sha256"}
    if (
        path.is_symlink()
        or not path.resolve(strict=True).is_relative_to(root)
        or document.get("sha256") != expected_sha256
        or _canonical_sha256(payload) != expected_sha256
    ):
        raise ValueError("RQ5 horizon logical input differs from its expected hash")
    return document


def _load_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(
            path.read_text(),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load RQ5 horizon JSON {path}") from error
    if not isinstance(value, dict):
        raise ValueError("RQ5 horizon JSON must be an object")
    return value


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


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


def _reject_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result
