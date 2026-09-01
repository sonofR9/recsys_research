from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping

from experiments.g3_pretrained_item_embeddings.analysis.rq2_final_results import (
    RQ2_FINAL_EVIDENCE_PATH,
)

from .constants import APPROVED_PROTOCOL_SHA256
from .rq4 import (
    RQ4_METADATA_FAMILIES,
    Rq4CapacitySurface,
    compile_rq4_capacity_surface,
)
from .rq3_post_boundary import Rq3ArtifactContract


RQ4_INITIAL_LEDGER_PATH = (
    "experiments/g3_pretrained_item_embeddings/protocol/ledgers/"
    "rq4_metadata_capacity_post_rq3.json"
)
RQ4_INITIAL_LEDGER_LOGICAL_SHA256 = (
    "8c25ebe5bed1b2557bf81721445fc699316eca5ed516a5638a83cc060fffba3b"
)
RQ3_FINAL_EVIDENCE_PATH = (
    "experiments/g3_pretrained_item_embeddings/evidence/rq3_final_native50m.json"
)
RQ3_FINAL_EVIDENCE_LOGICAL_SHA256 = (
    "1bc6ddf12ec94a6327f97cf8dad7e7a32e376ecc6ff1b4d67f03d4213a4cff06"
)
RQ3_FINAL_EVIDENCE_FILE_SHA256 = (
    "187cdaeb8551e37ab298fb322c87fe40802a09beb36a58cf40577f8e0c16f87c"
)
RQ3_FINAL_EVIDENCE_SIZE_BYTES = 890_003
RQ3_FINAL_SELECTED_ROW_ID = "rq3_output_learned_frozen_content:04"
RQ2_FINAL_EVIDENCE_SHA256 = (
    "a8f25319858f58f3f6e5cec2a51c513d697c478044ee9f9c5c355f7a471b7856"
)
RQ4_INITIAL_ARTIFACT_CONTRACTS = (
    Rq3ArtifactContract(
        "job_contract",
        "g3_rq4_initial_job.json",
        ("ledger_sha256", "row_id", "job", "ledger_path"),
    ),
    Rq3ArtifactContract(
        "training_metadata",
        "training_metadata.json",
        (
            "batch_size",
            "seed",
            "embedding_learning_rate",
            "deep_learning_rate",
            "lr_schedule_horizon_epochs",
            "best_epoch",
            "epochs_trained",
            "lr_horizon_complete",
            "g3_protocol_sha256",
            "g3_representation",
        ),
    ),
    Rq3ArtifactContract(
        "final_metrics",
        "final_metrics.json",
        ("recall@100", "ndcg@100", "num_users"),
    ),
    Rq3ArtifactContract("ranking_evidence", "ranking_evidence.pt"),
    Rq3ArtifactContract("top_item_rankings", "top_item_rankings.json"),
    Rq3ArtifactContract(
        "training_diagnostics",
        "g3_training_diagnostics.json",
        (
            "schema_version",
            "frequency_terciles",
            "training_count_reference",
            "slice_membership_reference",
            "content_drift_reference",
            "epochs",
        ),
        (2,),
    ),
    Rq3ArtifactContract("sweep_log", "sweep.log"),
)


@dataclass(frozen=True)
class Rq4InputReference:
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
class Rq4InitialJob:
    id: str
    family_id: str
    run_name: str
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
            "phase": "rq4_metadata_capacity",
            "stage": "rq4_metadata_capacity_post_rq3",
            "role": "metadata_capacity_search",
            "run_name": self.run_name,
            "reused_from": None,
            "representation": {
                "history": "selected_rq2_content_concat",
                "history_hidden_dim": self.history_hidden_dim,
                "catalog": self.catalog_representation,
                "metadata": list(self.metadata),
                "metadata_dim": self.metadata_dim,
                "metadata_pooling": "mean",
                "metadata_attachment": "history_and_catalog_concat_then_separate_densenet",
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
class Rq4InitialLedger:
    schema_version: int
    kind: str
    protocol_sha256: str
    rq2_final_evidence: Rq4InputReference
    rq3_final_evidence: Rq4InputReference
    feature_manifest: Rq4InputReference
    feature_identity: dict[str, object]
    expected_rq3_row_id: str
    family_opportunity_budgets: dict[str, int]
    stage_physical_jobs: int
    deferred_stages: dict[str, object]
    rows: tuple[Rq4InitialJob, ...]

    @property
    def sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self._payload()).encode()).hexdigest()

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "protocol_sha256": self.protocol_sha256,
            "inputs": {
                "rq2_final_evidence": self.rq2_final_evidence.to_dict(),
                "rq3_final_evidence": self.rq3_final_evidence.to_dict(),
                "feature_manifest": self.feature_manifest.to_dict(),
            },
            "feature_identity": self.feature_identity,
            "expected_rq3_row_id": self.expected_rq3_row_id,
            "family_opportunity_budgets": self.family_opportunity_budgets,
            "stage_physical_jobs": self.stage_physical_jobs,
            "deferred_stages": self.deferred_stages,
            "artifact_contracts": [
                contract.to_dict() for contract in RQ4_INITIAL_ARTIFACT_CONTRACTS
            ],
            "rows": [row.to_dict() for row in self.rows],
        }

    def to_dict(self) -> dict[str, object]:
        return self._payload() | {"sha256": self.sha256}


def compile_rq4_initial_ledger(
    *,
    root: Path,
    rq3_final_path: Path | None = None,
    expected_rq3_sha256: str = RQ3_FINAL_EVIDENCE_LOGICAL_SHA256,
    expected_rq3_row_id: str = RQ3_FINAL_SELECTED_ROW_ID,
    rq2_final_path: Path | None = None,
    expected_rq2_sha256: str = RQ2_FINAL_EVIDENCE_SHA256,
) -> Rq4InitialLedger:
    root = root.resolve(strict=True)
    rq2_path = rq2_final_path or root / RQ2_FINAL_EVIDENCE_PATH
    rq3_path = rq3_final_path or root / RQ3_FINAL_EVIDENCE_PATH
    if expected_rq3_sha256 == RQ3_FINAL_EVIDENCE_LOGICAL_SHA256 and (
        hashlib.sha256(rq3_path.read_bytes()).hexdigest()
        != RQ3_FINAL_EVIDENCE_FILE_SHA256
        or rq3_path.stat().st_size != RQ3_FINAL_EVIDENCE_SIZE_BYTES
    ):
        raise ValueError("RQ4 exact final RQ3 evidence file changed")
    if not expected_rq3_row_id or ":" not in expected_rq3_row_id:
        raise ValueError("RQ4 requires the exact selected RQ3 row id")
    surface = compile_rq4_capacity_surface(
        root=root,
        rq2_selection_path=rq2_path,
        expected_rq2_selection_sha256=expected_rq2_sha256,
        rq3_selection_path=rq3_path,
        expected_rq3_selection_sha256=expected_rq3_sha256,
        expected_rq3_row_id=expected_rq3_row_id,
    )
    rows = _jobs(surface)
    feature_path = root / surface.metadata_identity.manifest_path
    ledger = Rq4InitialLedger(
        schema_version=1,
        kind="g3_rq4_metadata_capacity_post_rq3",
        protocol_sha256=APPROVED_PROTOCOL_SHA256,
        rq2_final_evidence=_reference(root, rq2_path, expected_rq2_sha256),
        rq3_final_evidence=_reference(
            root, rq3_path, expected_rq3_sha256
        ),
        feature_manifest=_reference(
            root, feature_path, surface.metadata_identity.manifest_sha256
        ),
        feature_identity={
            "manifest_sha256": surface.metadata_identity.manifest_sha256,
            "feature_data_sha256": surface.metadata_identity.feature_data_sha256,
            "frequency_terciles": surface.metadata_identity.frequency_terciles,
            "training_count_reference": (
                surface.metadata_identity.training_count_reference
            ),
            "slice_membership_reference": (
                surface.metadata_identity.slice_membership_reference
            ),
        },
        expected_rq3_row_id=expected_rq3_row_id,
        family_opportunity_budgets={family: 12 for family in RQ4_METADATA_FAMILIES},
        stage_physical_jobs=27,
        deferred_stages={
            "metadata_horizon_followup": {
                "logical_opportunities_per_family": 3,
                "materialize_only_after_capacity_results": True,
            },
            "parameter_matched_extra_item_id": {
                "logical_opportunities": 12,
                "materialize_only_after_metadata_winner": True,
                "maximum_parameter_mismatch_fraction": 0.01,
            },
        },
        rows=rows,
    )
    _validate_program(ledger)
    if (
        expected_rq2_sha256 == RQ2_FINAL_EVIDENCE_SHA256
        and expected_rq3_sha256 == RQ3_FINAL_EVIDENCE_LOGICAL_SHA256
        and expected_rq3_row_id == RQ3_FINAL_SELECTED_ROW_ID
        and ledger.sha256 != RQ4_INITIAL_LEDGER_LOGICAL_SHA256
    ):
        raise ValueError("RQ4 canonical initial ledger logical SHA changed")
    return ledger


def validate_rq4_initial_ledger_document(
    document: object,
    *,
    root: Path,
    expected_ledger_sha256: str,
    expected_rq3_sha256: str,
    expected_rq3_row_id: str,
) -> Rq4InitialLedger:
    if not isinstance(document, dict):
        raise ValueError("RQ4 initial ledger must be an object")
    inputs = document.get("inputs")
    rq2 = inputs.get("rq2_final_evidence") if isinstance(inputs, dict) else None
    rq3 = inputs.get("rq3_final_evidence") if isinstance(inputs, dict) else None
    if not isinstance(rq2, dict) or not isinstance(rq3, dict):
        raise ValueError("RQ4 initial ledger input bindings are absent")
    rebuilt = compile_rq4_initial_ledger(
        root=root,
        rq2_final_path=root / str(rq2.get("path")),
        expected_rq2_sha256=str(rq2.get("logical_sha256")),
        rq3_final_path=root / str(rq3.get("path")),
        expected_rq3_sha256=expected_rq3_sha256,
        expected_rq3_row_id=expected_rq3_row_id,
    )
    expected = rebuilt.to_dict()
    _validate_exact_json_types(document, expected, path="ledger")
    if (
        expected_ledger_sha256 != rebuilt.sha256
        or rq3.get("logical_sha256") != expected_rq3_sha256
        or document.get("expected_rq3_row_id") != expected_rq3_row_id
        or document != expected
    ):
        raise ValueError("RQ4 initial ledger differs from frozen post-RQ3 inputs")
    return rebuilt


def load_rq4_initial_ledger(
    path: Path,
    *,
    root: Path,
    expected_ledger_sha256: str,
    expected_rq3_sha256: str,
    expected_rq3_row_id: str,
) -> Rq4InitialLedger:
    return validate_rq4_initial_ledger_document(
        _load_json(path),
        root=root,
        expected_ledger_sha256=expected_ledger_sha256,
        expected_rq3_sha256=expected_rq3_sha256,
        expected_rq3_row_id=expected_rq3_row_id,
    )


def persist_rq4_initial_ledger(
    path: Path,
    ledger: Rq4InitialLedger,
    *,
    root: Path,
    expected_rq3_sha256: str,
    expected_rq3_row_id: str,
) -> Path:
    validated = validate_rq4_initial_ledger_document(
        ledger.to_dict(),
        root=root,
        expected_ledger_sha256=ledger.sha256,
        expected_rq3_sha256=expected_rq3_sha256,
        expected_rq3_row_id=expected_rq3_row_id,
    )
    if validated != ledger:
        raise ValueError("RQ4 initial ledger changed during validation")
    content = (_canonical_json(ledger.to_dict()) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(content)
    except FileExistsError:
        if path.read_bytes() != content:
            raise RuntimeError(f"immutable RQ4 initial ledger differs: {path}")
    return path


def _jobs(surface: Rq4CapacitySurface) -> tuple[Rq4InitialJob, ...]:
    rows = tuple(
        Rq4InitialJob(
            id=row.id,
            family_id=row.family_id,
            run_name=row.run_name,
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
    if len(rows) != 27 or len({row.id for row in rows}) != 27:
        raise ValueError("RQ4 capacity stage must contain exactly 27 unique jobs")
    return rows


def _validate_program(ledger: Rq4InitialLedger) -> None:
    grouped = {
        family: [row for row in ledger.rows if row.family_id == family]
        for family in RQ4_METADATA_FAMILIES
    }
    if (
        ledger.family_opportunity_budgets
        != {family: 12 for family in RQ4_METADATA_FAMILIES}
        or any(len(rows) != 9 for rows in grouped.values())
        or any(
            {row.metadata_dim for row in rows} != {16, 32, 64}
            for rows in grouped.values()
        )
        or ledger.stage_physical_jobs != 27
    ):
        raise ValueError("RQ4 metadata families lost equal approved budgets")


def _reference(root: Path, path: Path, logical_sha256: str) -> Rq4InputReference:
    if path.is_symlink():
        raise ValueError("RQ4 input reference must not be a symlink")
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ValueError("RQ4 input reference escapes the project root")
    return Rq4InputReference(
        path=str(resolved.relative_to(root)),
        size_bytes=resolved.stat().st_size,
        sha256=hashlib.sha256(resolved.read_bytes()).hexdigest(),
        logical_sha256=logical_sha256,
    )


def _load_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(
            path.read_text(),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load RQ4 initial ledger {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"RQ4 ledger must be a JSON object: {path}")
    return value


def _validate_exact_json_types(actual: object, expected: object, *, path: str) -> None:
    if type(actual) is not type(expected):
        raise ValueError(f"RQ4 initial {path} has an invalid JSON type")
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise ValueError(f"RQ4 initial {path} has invalid keys")
        for name, expected_value in expected.items():
            _validate_exact_json_types(
                actual[name], expected_value, path=f"{path}.{name}"
            )
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError(f"RQ4 initial {path} has invalid length")
        for index, (actual_value, expected_value) in enumerate(
            zip(actual, expected, strict=True)
        ):
            _validate_exact_json_types(
                actual_value, expected_value, path=f"{path}[{index}]"
            )


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
