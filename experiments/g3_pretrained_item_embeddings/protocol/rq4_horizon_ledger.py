from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping

from .constants import APPROVED_PROTOCOL_SHA256
from .rq4 import (
    RQ4_METADATA_FAMILIES,
    Rq4HorizonFollowup,
    compile_rq4_capacity_surface,
    compile_rq4_horizon_followup,
)
from .rq4_initial_ledger import (
    RQ4_INITIAL_ARTIFACT_CONTRACTS,
    Rq4InputReference,
    Rq4InitialLedger,
    load_rq4_initial_ledger,
)
from .rq4_followup_gate import require_rq4_horizon_materialization_approval
from .search import TransferredHorizonRate


RQ4_HORIZON_LEDGER_PATH = (
    "experiments/g3_pretrained_item_embeddings/protocol/ledgers/"
    "rq4_metadata_horizon_post_capacity.json"
)
CONTROL_CALIBRATION_PATH = (
    "experiments/g3_pretrained_item_embeddings/evidence/"
    "untied_control_calibration.json"
)
CONTROL_CALIBRATION_SHA256 = (
    "015c94a182bc0df4179092e098e69b9b12c4fc62474ff4a2f15ad5d3e693e896"
)


@dataclass(frozen=True)
class Rq4HorizonJob:
    id: str
    family_id: str
    run_name: str
    reused_from: str | None
    embedding_learning_rate: float
    deep_learning_rate: float
    horizon_epochs: int
    metadata: tuple[str, ...]
    metadata_dim: int
    history_hidden_dim: int
    catalog_representation: str

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "family_id": self.family_id,
            "phase": "rq4_metadata_horizon",
            "stage": "rq4_metadata_horizon_post_capacity",
            "role": "metadata_horizon_search",
            "run_name": self.run_name,
            "reused_from": self.reused_from,
            "representation": {
                "history": "selected_rq2_content_concat",
                "history_hidden_dim": self.history_hidden_dim,
                "catalog": self.catalog_representation,
                "metadata": list(self.metadata),
                "metadata_dim": self.metadata_dim,
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
class Rq4HorizonLedger:
    initial_ledger: Rq4InputReference
    capacity_selection: Rq4InputReference
    control_calibration: Rq4InputReference
    expected_rq3_sha256: str
    expected_rq3_row_id: str
    selected_capacities: dict[str, int]
    rows: tuple[Rq4HorizonJob, ...]

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self._payload())

    @property
    def physical_rows(self) -> tuple[Rq4HorizonJob, ...]:
        return tuple(row for row in self.rows if row.reused_from is None)

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "kind": "g3_rq4_metadata_horizon_post_capacity",
            "protocol_sha256": APPROVED_PROTOCOL_SHA256,
            "inputs": {
                "initial_ledger": self.initial_ledger.to_dict(),
                "capacity_selection": self.capacity_selection.to_dict(),
                "control_calibration": self.control_calibration.to_dict(),
            },
            "expected_rq3_sha256": self.expected_rq3_sha256,
            "expected_rq3_row_id": self.expected_rq3_row_id,
            "selected_capacities": self.selected_capacities,
            "opportunity_accounting": {
                "logical_per_family": 3,
                "logical_total": 9,
                "physical_total": len(self.physical_rows),
                "cumulative_logical_per_family": 12,
            },
            "artifact_contracts": [
                (
                    contract.to_dict()
                    if contract.name != "job_contract"
                    else contract.to_dict()
                    | {"filename": "g3_rq4_horizon_job.json"}
                )
                for contract in RQ4_INITIAL_ARTIFACT_CONTRACTS
            ],
            "rows": [row.to_dict() for row in self.rows],
        }

    def to_dict(self) -> dict[str, object]:
        return self._payload() | {"sha256": self.sha256}


def compile_rq4_horizon_ledger(
    *,
    root: Path,
    initial_ledger_path: Path,
    expected_initial_ledger_sha256: str,
    capacity_selection_path: Path,
    expected_capacity_selection_sha256: str,
    expected_rq3_sha256: str,
    expected_rq3_row_id: str,
) -> Rq4HorizonLedger:
    root = root.resolve(strict=True)
    require_rq4_horizon_materialization_approval(root)
    initial = load_rq4_initial_ledger(
        initial_ledger_path,
        root=root,
        expected_ledger_sha256=expected_initial_ledger_sha256,
        expected_rq3_sha256=expected_rq3_sha256,
        expected_rq3_row_id=expected_rq3_row_id,
    )
    capacity = _load_logical_document(
        root,
        capacity_selection_path,
        expected_sha256=expected_capacity_selection_sha256,
    )
    selected_capacities = _selected_capacities(
        capacity,
        initial_ledger_reference=_reference(
            root, initial_ledger_path, expected_initial_ledger_sha256
        ),
        initial_ledger=initial,
    )
    calibration_path = root / CONTROL_CALIBRATION_PATH
    calibration = _load_logical_document(
        root, calibration_path, expected_sha256=CONTROL_CALIBRATION_SHA256
    )
    rates = _transferred_rates(calibration)
    surface = _capacity_surface(root, initial)
    followup = compile_rq4_horizon_followup(
        surface,
        selected_capacities=selected_capacities,
        transferred_horizon_rates={family: rates for family in RQ4_METADATA_FAMILIES},
    )
    ledger = Rq4HorizonLedger(
        initial_ledger=_reference(root, initial_ledger_path, initial.sha256),
        capacity_selection=_reference(
            root, capacity_selection_path, expected_capacity_selection_sha256
        ),
        control_calibration=_reference(
            root, calibration_path, CONTROL_CALIBRATION_SHA256
        ),
        expected_rq3_sha256=expected_rq3_sha256,
        expected_rq3_row_id=expected_rq3_row_id,
        selected_capacities=selected_capacities,
        rows=_jobs(followup),
    )
    _validate_program(ledger)
    return ledger


def load_rq4_horizon_ledger(
    path: Path,
    *,
    root: Path,
    expected_ledger_sha256: str,
    expected_rq3_sha256: str,
    expected_rq3_row_id: str,
) -> Rq4HorizonLedger:
    document = _load_json(path)
    inputs = document.get("inputs")
    initial = inputs.get("initial_ledger") if isinstance(inputs, dict) else None
    selection = inputs.get("capacity_selection") if isinstance(inputs, dict) else None
    if not isinstance(initial, dict) or not isinstance(selection, dict):
        raise ValueError("RQ4 horizon ledger input bindings are absent")
    rebuilt = compile_rq4_horizon_ledger(
        root=root,
        initial_ledger_path=root / str(initial.get("path")),
        expected_initial_ledger_sha256=str(initial.get("logical_sha256")),
        capacity_selection_path=root / str(selection.get("path")),
        expected_capacity_selection_sha256=str(selection.get("logical_sha256")),
        expected_rq3_sha256=expected_rq3_sha256,
        expected_rq3_row_id=expected_rq3_row_id,
    )
    if rebuilt.sha256 != expected_ledger_sha256 or document != rebuilt.to_dict():
        raise ValueError("RQ4 horizon ledger differs from its frozen inputs")
    return rebuilt


def persist_rq4_horizon_ledger(
    path: Path,
    ledger: Rq4HorizonLedger,
    *,
    root: Path,
) -> Path:
    root = root.resolve(strict=True)
    rebuilt = compile_rq4_horizon_ledger(
        root=root,
        initial_ledger_path=root / ledger.initial_ledger.path,
        expected_initial_ledger_sha256=ledger.initial_ledger.logical_sha256,
        capacity_selection_path=root / ledger.capacity_selection.path,
        expected_capacity_selection_sha256=ledger.capacity_selection.logical_sha256,
        expected_rq3_sha256=ledger.expected_rq3_sha256,
        expected_rq3_row_id=ledger.expected_rq3_row_id,
    )
    if rebuilt != ledger:
        raise ValueError("RQ4 horizon ledger differs from its authenticated inputs")
    content = (_canonical_json(ledger.to_dict()) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(content)
    except FileExistsError:
        if path.read_bytes() != content:
            raise RuntimeError(f"immutable RQ4 horizon ledger differs: {path}")
    return path


def reconstruct_rq4_horizon_surface(
    *, root: Path, ledger: Rq4HorizonLedger
) -> Rq4HorizonFollowup:
    initial = load_rq4_initial_ledger(
        root / ledger.initial_ledger.path,
        root=root,
        expected_ledger_sha256=ledger.initial_ledger.logical_sha256,
        expected_rq3_sha256=ledger.expected_rq3_sha256,
        expected_rq3_row_id=ledger.expected_rq3_row_id,
    )
    capacity = _capacity_surface(root.resolve(strict=True), initial)
    rates = _transferred_rates(
        _load_logical_document(
            root,
            root / ledger.control_calibration.path,
            expected_sha256=ledger.control_calibration.logical_sha256,
        )
    )
    return compile_rq4_horizon_followup(
        capacity,
        selected_capacities=ledger.selected_capacities,
        transferred_horizon_rates={family: rates for family in RQ4_METADATA_FAMILIES},
    )


def _capacity_surface(root: Path, initial: Rq4InitialLedger):
    return compile_rq4_capacity_surface(
        root=root,
        rq2_selection_path=root / initial.rq2_final_evidence.path,
        expected_rq2_selection_sha256=initial.rq2_final_evidence.logical_sha256,
        rq3_selection_path=root / initial.rq3_final_evidence.path,
        expected_rq3_selection_sha256=initial.rq3_final_evidence.logical_sha256,
        expected_rq3_row_id=initial.expected_rq3_row_id,
    )


def _selected_capacities(
    document: Mapping[str, object],
    *,
    initial_ledger_reference: Rq4InputReference,
    initial_ledger: Rq4InitialLedger,
) -> dict[str, int]:
    expected_keys = {
        "schema_version",
        "kind",
        "protocol_sha256",
        "initial_ledger",
        "opportunity_accounting",
        "selection_rule",
        "family_selections",
        "capacity_extensions_required",
        "sha256",
    }
    selections = document.get("family_selections")
    if (
        set(document) != expected_keys
        or document.get("schema_version") != 1
        or document.get("kind")
        != "g3_rq4_metadata_capacity_selection_native50m"
        or document.get("protocol_sha256") != APPROVED_PROTOCOL_SHA256
        or document.get("initial_ledger") != initial_ledger_reference.to_dict()
        or document.get("opportunity_accounting")
        != {
            "families": 3,
            "opportunities_per_family": 9,
            "total_opportunities": 27,
        }
        or not isinstance(selections, dict)
        or set(selections) != set(RQ4_METADATA_FAMILIES)
        or document.get("capacity_extensions_required") != []
    ):
        raise ValueError("RQ4 capacity selection is incomplete or boundary-unresolved")
    result = {}
    rows = {row.id: row for row in initial_ledger.rows}
    for family_id, selection in selections.items():
        if not isinstance(selection, dict):
            raise ValueError("RQ4 capacity selection row is invalid")
        capacity = selection.get("selected_metadata_dim")
        boundary = selection.get("capacity_boundary")
        selected = selection.get("selected")
        row_id = selection.get("selected_row_id")
        row = rows.get(row_id) if isinstance(row_id, str) else None
        if (
            type(capacity) is not int
            or capacity not in {16, 32, 64}
            or row is None
            or row.family_id != family_id
            or not isinstance(selected, dict)
            or selected.get("row_id") != row.id
            or selected.get("family_id") != family_id
            or selected.get("ledger_sha256") != initial_ledger.sha256
            or selected.get("job") != row.to_dict()
            or selected.get("metadata_dim") != row.metadata_dim
            or selected.get("embedding_learning_rate")
            != row.embedding_learning_rate
            or selected.get("deep_learning_rate") != row.deep_learning_rate
            or selected.get("horizon_epochs") != row.horizon_epochs
            or selection.get("selected_metadata_dim") != row.metadata_dim
            or selection.get("selected_embedding_learning_rate")
            != row.embedding_learning_rate
            or selection.get("selected_deep_learning_rate")
            != row.deep_learning_rate
            or selection.get("selected_horizon_epochs") != row.horizon_epochs
            or not isinstance(boundary, dict)
            or boundary.get("direction") is not None
            or boundary.get("extension_capacity") is not None
        ):
            raise ValueError("RQ4 capacity boundary must resolve before horizon search")
        result[family_id] = capacity
    return result


def _transferred_rates(
    calibration: Mapping[str, object],
) -> tuple[TransferredHorizonRate, ...]:
    decision = calibration.get("transfer_decision")
    fits = calibration.get("power_law_fits")
    embedding = fits.get("embedding_learning_rate") if isinstance(fits, dict) else None
    deep = fits.get("deep_learning_rate") if isinstance(fits, dict) else None
    embedding_values = embedding.get("fitted_coordinates") if isinstance(embedding, dict) else None
    deep_values = deep.get("fitted_coordinates") if isinstance(deep, dict) else None
    if (
        not isinstance(decision, dict)
        or decision.get("accepted") is not True
        or not isinstance(embedding_values, dict)
        or not isinstance(deep_values, dict)
    ):
        raise ValueError("RQ4 horizon search requires the accepted control transfer")
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
        raise ValueError("RQ4 transferred horizon rates changed")
    return rates


def _jobs(surface: Rq4HorizonFollowup) -> tuple[Rq4HorizonJob, ...]:
    rows = tuple(
        Rq4HorizonJob(
            id=row.id,
            family_id=row.family_id,
            run_name=row.run_name,
            reused_from=row.reused_from,
            embedding_learning_rate=row.embedding_learning_rate,
            deep_learning_rate=row.deep_learning_rate,
            horizon_epochs=row.horizon_epochs,
            metadata=row.metadata,
            metadata_dim=row.metadata_dim,
            history_hidden_dim=surface.predecessor.history_hidden_dim,
            catalog_representation=surface.predecessor.catalog_representation,
        )
        for family in RQ4_METADATA_FAMILIES
        for row in surface.rows_by_family[family]
    )
    if len(rows) != 9 or len({row.id for row in rows}) != 9:
        raise ValueError("RQ4 horizon stage must preserve nine logical opportunities")
    return rows


def _validate_program(ledger: Rq4HorizonLedger) -> None:
    if set(ledger.selected_capacities) != set(RQ4_METADATA_FAMILIES):
        raise ValueError("RQ4 horizon ledger omits a metadata family")
    for family in RQ4_METADATA_FAMILIES:
        rows = [row for row in ledger.rows if row.family_id == family]
        if (
            len(rows) != 3
            or [row.horizon_epochs for row in rows] != [15, 25, 40]
            or any(row.metadata_dim != ledger.selected_capacities[family] for row in rows)
        ):
            raise ValueError("RQ4 horizon ledger lost its equal three-cell design")


def _reference(root: Path, path: Path, logical_sha256: str) -> Rq4InputReference:
    if path.is_symlink():
        raise ValueError("RQ4 horizon input reference must not be a symlink")
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ValueError("RQ4 horizon input reference escapes the project root")
    return Rq4InputReference(
        path=str(resolved.relative_to(root)),
        size_bytes=resolved.stat().st_size,
        sha256=hashlib.sha256(resolved.read_bytes()).hexdigest(),
        logical_sha256=logical_sha256,
    )


def _load_logical_document(
    root: Path, path: Path, *, expected_sha256: str
) -> dict[str, object]:
    if path.is_symlink():
        raise ValueError("RQ4 logical input must not be a symlink")
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ValueError("RQ4 logical input escapes the project root")
    document = _load_json(resolved)
    payload = {name: value for name, value in document.items() if name != "sha256"}
    if (
        len(expected_sha256) != 64
        or document.get("sha256") != expected_sha256
        or _canonical_sha256(payload) != expected_sha256
    ):
        raise ValueError("RQ4 logical input differs from its expected hash")
    return document


def _load_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(
            path.read_text(),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load RQ4 horizon JSON {path}") from error
    if not isinstance(value, dict):
        raise ValueError("RQ4 horizon JSON must be an object")
    return value


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
