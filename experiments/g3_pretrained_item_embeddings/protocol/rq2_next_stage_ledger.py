from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from experiments.g3_pretrained_item_embeddings.analysis.control_calibration import (
    load_control_calibration,
)
from experiments.g3_pretrained_item_embeddings.analysis.rq2_capacity_results import (
    APPROVED_RQ2_CAPACITY_EVIDENCE_SHA256,
    RQ2_CAPACITY_EVIDENCE_PATH,
)

from .constants import APPROVED_PROTOCOL, APPROVED_PROTOCOL_SHA256
from .control_ledger import ManifestReference
from .rq2_capacity_ledger import (
    APPROVED_RQ2_CAPACITY_LEDGER_SHA256,
    PREDECESSOR_CALIBRATION_PATH,
    PREDECESSOR_CALIBRATION_SHA256,
    RQ2_CAPACITY_LEDGER_PATH,
    initial_rq2_capacity_ledger,
)
from .search import (
    APPROVED_FAMILY_SPECS,
    TransferredHorizonRate,
    compile_capacity_first_stage,
    compile_capacity_horizon_followup,
)


RQ2_NEXT_STAGE_LEDGER_PATH = (
    "experiments/g3_pretrained_item_embeddings/protocol/ledgers/"
    "rq2_resolved_next_stage.json"
)
APPROVED_RQ2_NEXT_STAGE_LEDGER_SHA256 = (
    "7134d4dfbe80efdbac23c18f614bee8e6822a5379d34a0fc256301d3c9393559"
)


@dataclass(frozen=True)
class Rq2NextStageJob:
    id: str
    family_id: str
    phase: str
    run_name: str
    capacity: int
    embedding_learning_rate: float
    deep_learning_rate: float
    horizon_epochs: int
    reused_from: str | None

    def to_dict(self) -> dict[str, object]:
        history = {
            "rq2_content_concat": "learned_item_id_plus_frozen_content",
            "rq2_id_only_densenet": "learned_item_id_densenet",
        }[self.family_id]
        return {
            "id": self.id,
            "family_id": self.family_id,
            "phase": self.phase,
            "run_name": self.run_name,
            "stage": "rq2_resolved_next_stage",
            "role": (
                "capacity_extension"
                if self.phase == "capacity_boundary_extension"
                else "horizon_probe"
            ),
            "reused_from": self.reused_from,
            "representation": {
                "id": self.family_id,
                "history": history,
                "catalog": "learned_item_id",
                "history_hidden_dim": self.capacity,
                "separate_history_catalog_tables": True,
                "content_trainable": False,
                "content_width": (
                    128 if self.family_id == "rq2_content_concat" else None
                ),
            },
            "dataset": {
                "size": APPROVED_PROTOCOL.main_dataset_size,
                "source": "likes",
                "event_limit": 50_000_000,
                "sampling": "none",
                "minimum_user_interactions": 5,
                "validation_interval_seconds": 604800,
                "candidate_catalog": "full",
                "exclude_seen": False,
            },
            "training": {
                "batch_size": APPROVED_PROTOCOL.batch_size,
                "seed": APPROVED_PROTOCOL.seed,
                "embedding_learning_rate": self.embedding_learning_rate,
                "deep_learning_rate": self.deep_learning_rate,
                "horizon_epochs": self.horizon_epochs,
                "validate_every_epoch": True,
                "restore_best_validation_epoch": True,
            },
        }


@dataclass(frozen=True)
class Rq2NextStageLedger:
    schema_version: int
    kind: str
    protocol_sha256: str
    maximum_opportunities: int
    maximum_physical_jobs: int
    preselection_evidence: ManifestReference
    preselection_ledger: ManifestReference
    predecessor_calibration: ManifestReference
    deferred_content_horizon_followup: dict[str, object]
    rows: tuple[Rq2NextStageJob, ...]

    @property
    def inputs(self) -> dict[str, ManifestReference]:
        return {
            "preselection_evidence": self.preselection_evidence,
            "preselection_ledger": self.preselection_ledger,
            "predecessor_calibration": self.predecessor_calibration,
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self._payload()).encode()).hexdigest()

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "protocol_sha256": self.protocol_sha256,
            "maximum_opportunities": self.maximum_opportunities,
            "maximum_physical_jobs": self.maximum_physical_jobs,
            "inputs": {
                key: value.to_dict() for key, value in self.inputs.items()
            },
            "deferred_content_horizon_followup": (
                self.deferred_content_horizon_followup
            ),
            "rows": [row.to_dict() for row in self.rows],
        }

    def to_dict(self) -> dict[str, object]:
        return self._payload() | {"sha256": self.sha256}


def initial_rq2_next_stage_ledger(
    *, evidence: Mapping[str, object]
) -> Rq2NextStageLedger:
    if evidence.get("sha256") != APPROVED_RQ2_CAPACITY_EVIDENCE_SHA256:
        raise ValueError("RQ2 next stage requires the approved preselection evidence")
    selections = evidence.get("family_selections")
    if not isinstance(selections, list):
        raise ValueError("RQ2 preselection evidence has no family selections")
    by_family = {
        str(value.get("family_id")): value
        for value in selections
        if isinstance(value, dict)
    }
    if set(by_family) != {"rq2_content_concat", "rq2_id_only_densenet"}:
        raise ValueError("RQ2 preselection evidence family selections drifted")
    content = by_family["rq2_content_concat"]
    identifier = by_family["rq2_id_only_densenet"]
    if (
        content.get("selected", {}).get("capacity") != 64
        or content.get("boundary_decision", {}).get("capacity")
        != {
            "selected": 64,
            "direction": "lower",
            "extension_capacity": 32,
        }
        or identifier.get("selected", {}).get("capacity") != 255
        or identifier.get("boundary_decision", {}).get("extension_required")
        is not False
    ):
        raise ValueError("RQ2 preselection evidence does not resolve the next stage")
    preselection = initial_rq2_capacity_ledger()
    content_source = [
        row for row in preselection.rows if row.family_id == "rq2_content_concat"
    ][:3]
    content_rows = tuple(
        Rq2NextStageJob(
            id=f"rq2_content_concat:{index + 13:02d}",
            family_id="rq2_content_concat",
            phase="capacity_boundary_extension",
            run_name=(
                f"g3_rq2_content_concat_width_32_boundary_trial_{index + 1:02d}_"
                "native50m"
            ),
            capacity=32,
            embedding_learning_rate=source.embedding_learning_rate,
            deep_learning_rate=source.deep_learning_rate,
            horizon_epochs=25,
            reused_from=None,
        )
        for index, source in enumerate(content_source)
    )
    calibration = load_control_calibration(
        Path(__file__).resolve().parents[3] / PREDECESSOR_CALIBRATION_PATH
    )
    fits = calibration.get("power_law_fits")
    if not isinstance(fits, dict):
        raise ValueError("RQ2 calibration has no fitted horizon rates")
    embedding_fit = fits.get("embedding_learning_rate")
    deep_fit = fits.get("deep_learning_rate")
    if not isinstance(embedding_fit, dict) or not isinstance(deep_fit, dict):
        raise ValueError("RQ2 calibration fitted rate groups are invalid")
    embedding = embedding_fit.get("fitted_coordinates")
    deep = deep_fit.get("fitted_coordinates")
    if not isinstance(embedding, dict) or not isinstance(deep, dict):
        raise ValueError("RQ2 calibration fitted coordinates are invalid")
    transferred = tuple(
        TransferredHorizonRate(
            horizon,
            float(embedding[str(horizon)]),
            float(deep[str(horizon)]),
        )
        for horizon in (15, 25, 40)
    )
    spec = next(
        value
        for value in APPROVED_FAMILY_SPECS
        if value.id == "rq2_id_only_densenet"
    )
    coordinates = compile_capacity_horizon_followup(
        spec,
        selected_capacity=255,
        transferred_horizon_rates=transferred,
        first_stage=compile_capacity_first_stage(spec),
    )
    identifier_rows = tuple(
        Rq2NextStageJob(
            id=coordinate.id,
            family_id=coordinate.family_id,
            phase="selected_capacity_horizon_followup",
            run_name=(
                "g3_rq2_id_only_densenet_width_255_horizon_"
                f"{coordinate.horizon_epochs}_native50m"
            ),
            capacity=255,
            embedding_learning_rate=coordinate.embedding_learning_rate,
            deep_learning_rate=coordinate.deep_learning_rate,
            horizon_epochs=coordinate.horizon_epochs,
            reused_from=coordinate.reused_from,
        )
        for coordinate in coordinates
    )
    rows = (*content_rows, *identifier_rows)
    if any(row.reused_from is not None for row in rows):
        raise ValueError("RQ2 next stage unexpectedly reuses a non-identical cell")
    ledger = Rq2NextStageLedger(
        schema_version=1,
        kind="g3_rq2_resolved_next_stage",
        protocol_sha256=APPROVED_PROTOCOL_SHA256,
        maximum_opportunities=6,
        maximum_physical_jobs=6,
        preselection_evidence=ManifestReference(
            "g3_rq2_capacity_preselection_evidence",
            RQ2_CAPACITY_EVIDENCE_PATH,
            APPROVED_RQ2_CAPACITY_EVIDENCE_SHA256,
        ),
        preselection_ledger=ManifestReference(
            "g3_rq2_capacity_preselection",
            RQ2_CAPACITY_LEDGER_PATH,
            APPROVED_RQ2_CAPACITY_LEDGER_SHA256,
        ),
        predecessor_calibration=ManifestReference(
            "g3_untied_control_calibration",
            PREDECESSOR_CALIBRATION_PATH,
            PREDECESSOR_CALIBRATION_SHA256,
        ),
        deferred_content_horizon_followup={
            "family_id": "rq2_content_concat",
            "reason": "selected capacity is unresolved at the lower boundary",
            "pending_extension_capacity": 32,
        },
        rows=rows,
    )
    if (
        APPROVED_RQ2_NEXT_STAGE_LEDGER_SHA256 != "0" * 64
        and ledger.sha256 != APPROVED_RQ2_NEXT_STAGE_LEDGER_SHA256
    ):
        raise ValueError("approved RQ2 next-stage ledger definition drifted")
    return ledger


def validate_rq2_next_stage_ledger_document(
    document: object,
) -> Rq2NextStageLedger:
    if not isinstance(document, dict):
        raise ValueError("RQ2 next-stage ledger must be an object")
    expected_keys = {
        "schema_version",
        "kind",
        "protocol_sha256",
        "maximum_opportunities",
        "maximum_physical_jobs",
        "inputs",
        "deferred_content_horizon_followup",
        "rows",
        "sha256",
    }
    if set(document) != expected_keys:
        raise ValueError("RQ2 next-stage ledger keys do not match the closed schema")
    evidence_path = Path(__file__).resolve().parents[3] / RQ2_CAPACITY_EVIDENCE_PATH
    from experiments.g3_pretrained_item_embeddings.analysis.rq2_capacity_results import (
        load_rq2_capacity_evidence,
    )

    expected = initial_rq2_next_stage_ledger(
        evidence=load_rq2_capacity_evidence(evidence_path)
    )
    _validate_exact_json_types(document, expected.to_dict(), path="ledger")
    payload = {key: value for key, value in document.items() if key != "sha256"}
    actual_sha256 = hashlib.sha256(_canonical_json(payload).encode()).hexdigest()
    if actual_sha256 != expected.sha256:
        raise ValueError("RQ2 ledger no longer matches approved next-stage coordinates")
    if document["sha256"] != actual_sha256:
        raise ValueError("RQ2 next-stage ledger hash differs from its payload")
    return expected


def load_rq2_next_stage_ledger(path: Path) -> Rq2NextStageLedger:
    try:
        document = json.loads(
            path.read_text(),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load RQ2 next-stage ledger {path}") from error
    ledger = validate_rq2_next_stage_ledger_document(document)
    if ledger.sha256 != APPROVED_RQ2_NEXT_STAGE_LEDGER_SHA256:
        raise ValueError("RQ2 next-stage ledger is not the approved immutable ledger")
    return ledger


def persist_rq2_next_stage_ledger(path: Path, ledger: Rq2NextStageLedger) -> Path:
    validate_rq2_next_stage_ledger_document(ledger.to_dict())
    content = (_canonical_json(ledger.to_dict()) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(content)
    except FileExistsError:
        if path.read_bytes() != content:
            raise RuntimeError(f"immutable RQ2 next-stage ledger differs: {path}")
    return path


def _validate_exact_json_types(actual: object, expected: object, *, path: str) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            raise ValueError(f"RQ2 next-stage {path} has an invalid JSON type")
        for key, expected_value in expected.items():
            if key in actual:
                _validate_exact_json_types(
                    actual[key], expected_value, path=f"{path}.{key}"
                )
        return
    if isinstance(expected, list):
        if not isinstance(actual, list):
            raise ValueError(f"RQ2 next-stage {path} has an invalid JSON type")
        for index, (actual_value, expected_value) in enumerate(
            zip(actual, expected, strict=False)
        ):
            _validate_exact_json_types(
                actual_value, expected_value, path=f"{path}[{index}]"
            )
        return
    if type(actual) is not type(expected):
        raise ValueError(f"RQ2 next-stage {path} has an invalid JSON type")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON number {value!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    arguments = parser.parse_args()
    from experiments.g3_pretrained_item_embeddings.analysis.rq2_capacity_results import (
        load_rq2_capacity_evidence,
    )

    root = arguments.root.resolve()
    ledger = initial_rq2_next_stage_ledger(
        evidence=load_rq2_capacity_evidence(root / RQ2_CAPACITY_EVIDENCE_PATH)
    )
    path = root / RQ2_NEXT_STAGE_LEDGER_PATH
    if arguments.write:
        persist_rq2_next_stage_ledger(path, ledger)
    print(
        json.dumps(
            {
                "path": str(path),
                "sha256": ledger.sha256,
                "opportunities": len(ledger.rows),
                "physical_jobs": sum(row.reused_from is None for row in ledger.rows),
                "status": "materialized" if arguments.write else "preview",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
