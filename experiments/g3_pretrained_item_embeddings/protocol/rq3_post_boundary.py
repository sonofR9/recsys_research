from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Callable, Mapping

from experiments.g3_pretrained_item_embeddings.configs.model import (
    RQ3_CATALOG_REPRESENTATIONS,
)

from .constants import APPROVED_PROTOCOL, APPROVED_PROTOCOL_SHA256
from .rq3 import (
    AuthenticatedRq2Coordinate,
    RQ3_OUTPUT_FAMILY_IDS,
    Rq3OutputRow,
    Rq3OutputSurface,
    authenticate_rq2_reuse_rows,
)
from .search import APPROVED_FAMILY_SPECS, ReusableCoordinate, compile_family


POST_BOUNDARY_ADAPTER_KIND = "g3_rq2_post_boundary_verifier_v1"
RQ2_FINAL_EVIDENCE_PATH = (
    "experiments/g3_pretrained_item_embeddings/evidence/rq2_final_native50m.json"
)
RQ2_FINAL_EVIDENCE_LOGICAL_SHA256 = (
    "a8f25319858f58f3f6e5cec2a51c513d697c478044ee9f9c5c355f7a471b7856"
)
RQ2_FINAL_SELECTED_ROW_ID = "rq2_unexpected_diagnostic:03"
RQ3_ELIGIBLE_REUSE_ROW_IDS = (
    "rq2_content_concat:04",
    "rq2_content_concat:05",
    "rq2_content_concat:06",
    RQ2_FINAL_SELECTED_ROW_ID,
    "rq2_content_concat:19",
    "rq2_content_concat:20",
    "rq2_content_concat:21",
)
RQ3_ELIGIBLE_REUSE_IDS = frozenset(RQ3_ELIGIBLE_REUSE_ROW_IDS)


@dataclass(frozen=True)
class Rq3FeatureBinding:
    manifest_path: str
    manifest_sha256: str
    manifest_file_sha256: str
    data_path: str
    data_sha256: str
    frequency_terciles: dict[str, object]
    training_count_reference: dict[str, object]
    slice_membership_reference: dict[str, object]


@dataclass(frozen=True)
class Rq3VerifiedReuse:
    coordinate: AuthenticatedRq2Coordinate
    source_ledger_path: str
    source_ledger_sha256: str


@dataclass(frozen=True)
class Rq3PostBoundaryVerification:
    adapter_kind: str
    final_evidence_sha256: str
    selected_row_id: str
    selected_history_hidden_dim: int
    feature: Rq3FeatureBinding
    reusable: tuple[Rq3VerifiedReuse, ...]


Rq3PostBoundaryVerifier = Callable[
    [Path, Path, str, str],
    Rq3PostBoundaryVerification,
]


def verify_final_rq2_evidence_for_rq3(
    root: Path,
    path: Path,
    expected_final_sha256: str,
    expected_selected_row_id: str,
) -> Rq3PostBoundaryVerification:
    from experiments.g3_pretrained_item_embeddings.analysis.rq2_final_results import (
        eligible_rq3_reuse_rows,
        load_rq2_final_evidence,
    )

    root = root.resolve(strict=True)
    path = _project_file(root, path)
    if (
        expected_final_sha256 != RQ2_FINAL_EVIDENCE_LOGICAL_SHA256
        or expected_selected_row_id != RQ2_FINAL_SELECTED_ROW_ID
        or str(path.relative_to(root)) != RQ2_FINAL_EVIDENCE_PATH
    ):
        raise ValueError("RQ3 requires the exact materialized final RQ2 evidence")
    evidence = load_rq2_final_evidence(path, root=root)
    selection = evidence.get("final_content_selection")
    inputs = evidence.get("rq3_inputs")
    if (
        evidence.get("sha256") != expected_final_sha256
        or not isinstance(selection, dict)
        or selection.get("status") != "resolved"
        or not isinstance(selection.get("selected"), dict)
        or selection["selected"].get("row_id") != expected_selected_row_id
        or selection.get("provisional_selected") is not None
        or not isinstance(inputs, dict)
        or inputs.get("status") != "ready"
        or inputs.get("selected_content_input") != selection["selected"]
    ):
        raise ValueError("RQ3 final RQ2 selection is not the exact resolved result")
    raw_rows = inputs.get("eligible_learned_output_reuse_rows")
    source_ledgers = inputs.get("reuse_source_ledgers")
    if not isinstance(raw_rows, list) or not isinstance(source_ledgers, dict):
        raise ValueError("RQ3 final evidence has no reusable row bindings")
    rows = eligible_rq3_reuse_rows(raw_rows)
    if (
        tuple(row.get("row_id") for row in rows) != RQ3_ELIGIBLE_REUSE_ROW_IDS
        or set(source_ledgers) != RQ3_ELIGIBLE_REUSE_IDS
        or len(source_ledgers) != len(RQ3_ELIGIBLE_REUSE_ROW_IDS)
    ):
        raise ValueError("RQ3 final evidence reusable row mapping changed")
    authenticated = authenticate_rq2_reuse_rows(
        root=root,
        rows=rows,
        source_ledgers=source_ledgers,
        source_evidence_sha256=expected_final_sha256,
    )
    reusable = tuple(
        Rq3VerifiedReuse(
            coordinate=coordinate,
            source_ledger_path=coordinate.source_ledger_path,
            source_ledger_sha256=coordinate.source_ledger_sha256,
        )
        for coordinate in authenticated.coordinates
    )
    return Rq3PostBoundaryVerification(
        adapter_kind=POST_BOUNDARY_ADAPTER_KIND,
        final_evidence_sha256=expected_final_sha256,
        selected_row_id=expected_selected_row_id,
        selected_history_hidden_dim=128,
        feature=Rq3FeatureBinding(
            manifest_path=authenticated.feature_manifest_path,
            manifest_sha256=authenticated.feature_manifest_sha256,
            manifest_file_sha256=authenticated.feature_manifest_file_sha256,
            data_path=authenticated.feature_data_path,
            data_sha256=authenticated.feature_data_sha256,
            frequency_terciles=authenticated.frequency_terciles,
            training_count_reference=authenticated.training_count_reference,
            slice_membership_reference=authenticated.slice_membership_reference,
        ),
        reusable=reusable,
    )


@dataclass(frozen=True)
class Rq3ArtifactContract:
    name: str
    filename: str
    required_keys: tuple[str, ...] = ()
    schema_versions: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "filename": self.filename,
            "required_keys": list(self.required_keys),
            "schema_versions": list(self.schema_versions),
        }


RQ3_OUTPUT_ARTIFACT_CONTRACTS = (
    Rq3ArtifactContract(
        "job_contract",
        "g3_rq3_output_job.json",
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
    Rq3ArtifactContract("final_metrics", "final_metrics.json", ("recall@100", "ndcg@100", "num_users")),
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
class Rq3PostBoundaryLedgerRow:
    id: str
    family_id: str
    run_name: str
    batch_size: int
    seed: int
    embedding_learning_rate: float
    deep_learning_rate: float
    horizon_epochs: int
    history_hidden_dim: int
    catalog_representation: str
    reused_from: str | None
    source_ledger_path: str | None
    source_ledger_sha256: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "family_id": self.family_id,
            "phase": "rq3_catalog_output",
            "stage": "rq3_post_boundary_output_search",
            "role": "reused_rq2" if self.reused_from is not None else "search",
            "run_name": self.run_name,
            "reused_from": self.reused_from,
            "source_ledger": (
                None
                if self.reused_from is None
                else {
                    "path": self.source_ledger_path,
                    "logical_sha256": self.source_ledger_sha256,
                }
            ),
            "representation": {
                "id": self.family_id,
                "history_representation": "id_content",
                "history_hidden_dim": self.history_hidden_dim,
                "catalog_representation": self.catalog_representation,
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
class Rq3PostBoundaryLedger:
    schema_version: int
    kind: str
    adapter_kind: str
    protocol_sha256: str
    final_evidence_path: str
    final_rq2_evidence_sha256: str
    selected_rq2_row_id: str
    feature: Rq3FeatureBinding
    source_ledgers: tuple[tuple[str, str, str], ...]
    logical_rows: tuple[Rq3PostBoundaryLedgerRow, ...]

    @property
    def physical_rows(self) -> tuple[Rq3PostBoundaryLedgerRow, ...]:
        return tuple(row for row in self.logical_rows if row.reused_from is None)

    @property
    def rows(self) -> tuple[Rq3PostBoundaryLedgerRow, ...]:
        return self.physical_rows

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self._payload())

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "adapter_kind": self.adapter_kind,
            "protocol_sha256": self.protocol_sha256,
            "final_evidence_path": self.final_evidence_path,
            "final_rq2_evidence_sha256": self.final_rq2_evidence_sha256,
            "selected_rq2_row_id": self.selected_rq2_row_id,
            "opportunity_accounting": {
                "logical_rows": len(self.logical_rows),
                "reused_rows": len(self.logical_rows) - len(self.physical_rows),
                "physical_rows": len(self.physical_rows),
            },
            "feature": _feature_dict(self.feature),
            "source_ledgers": [
                {"source_id": source_id, "path": path, "logical_sha256": sha256}
                for source_id, path, sha256 in self.source_ledgers
            ],
            "artifact_contracts": [
                contract.to_dict() for contract in RQ3_OUTPUT_ARTIFACT_CONTRACTS
            ],
            "logical_rows": [row.to_dict() for row in self.logical_rows],
            "physical_rows": [row.to_dict() for row in self.physical_rows],
        }

    def to_dict(self) -> dict[str, object]:
        return self._payload() | {"sha256": self.sha256}


def compile_verified_rq3_post_boundary_surface(
    *,
    root: Path,
    final_evidence_path: Path,
    expected_final_rq2_evidence_sha256: str,
    expected_selected_rq2_row_id: str,
    adapter_kind: str,
    verifier: Rq3PostBoundaryVerifier,
) -> Rq3OutputSurface:
    root = root.resolve(strict=True)
    path = _project_file(root, final_evidence_path)
    if (
        adapter_kind != POST_BOUNDARY_ADAPTER_KIND
        or not _valid_sha256(expected_final_rq2_evidence_sha256)
        or expected_selected_rq2_row_id not in RQ3_ELIGIBLE_REUSE_IDS
    ):
        raise ValueError("RQ3 post-boundary adapter inputs are invalid")
    verified = verifier(
        root,
        path,
        expected_final_rq2_evidence_sha256,
        expected_selected_rq2_row_id,
    )
    _validate_verification(
        verified,
        expected_final_sha256=expected_final_rq2_evidence_sha256,
        expected_selected_row_id=expected_selected_rq2_row_id,
    )
    sources = tuple(value.coordinate for value in verified.reusable)
    reusable = tuple(
        ReusableCoordinate(
            source_id=source.source_id,
            embedding_learning_rate=source.embedding_learning_rate,
            deep_learning_rate=source.deep_learning_rate,
            horizon_epochs=source.horizon_epochs,
        )
        for source in sources
    )
    specifications = {
        value.id: value
        for value in APPROVED_FAMILY_SPECS
        if value.id in RQ3_OUTPUT_FAMILY_IDS
    }
    if tuple(specifications) != RQ3_OUTPUT_FAMILY_IDS:
        raise ValueError("approved RQ3 output family definitions changed")
    learned = compile_family(
        specifications["rq3_output_learned"],
        reusable=reusable,
        allow_reusable_outside_search_space=True,
    )
    authenticated = {source.source_id: source for source in sources}
    rows_by_family = {
        family_id: tuple(
            _surface_row(
                family_id,
                index,
                replace(
                    coordinate,
                    id=f"{family_id}:{index + 1:02d}",
                    family_id=family_id,
                    reused_from=(
                        coordinate.reused_from
                        if family_id == "rq3_output_learned"
                        else None
                    ),
                ),
                width=verified.selected_history_hidden_dim,
                authenticated=authenticated,
            )
            for index, coordinate in enumerate(learned)
        )
        for family_id in RQ3_OUTPUT_FAMILY_IDS
    }
    feature = verified.feature
    surface = Rq3OutputSurface(
        selection_path=str(path.relative_to(root)),
        selection_sha256=verified.final_evidence_sha256,
        selected_history_hidden_dim=verified.selected_history_hidden_dim,
        feature_manifest_path=feature.manifest_path,
        feature_manifest_sha256=feature.manifest_sha256,
        feature_manifest_file_sha256=feature.manifest_file_sha256,
        feature_data_path=feature.data_path,
        feature_data_sha256=feature.data_sha256,
        frequency_terciles=feature.frequency_terciles,
        training_count_reference=feature.training_count_reference,
        slice_membership_reference=feature.slice_membership_reference,
        rows_by_family=rows_by_family,
        final_rq2_evidence_sha256=verified.final_evidence_sha256,
        selected_rq2_row_id=verified.selected_row_id,
        source_ledgers=tuple(
            sorted(
                (
                    value.coordinate.source_id,
                    value.source_ledger_path,
                    value.source_ledger_sha256,
                )
                for value in verified.reusable
            )
        ),
    )
    _validate_surface(surface)
    return surface


def compile_rq3_post_boundary_ledger(
    surface: Rq3OutputSurface,
) -> Rq3PostBoundaryLedger:
    _validate_surface(surface)
    source_ledgers = {
        source_id: (path, sha256)
        for source_id, path, sha256 in surface.source_ledgers
    }
    logical_rows = tuple(
        _ledger_row(row, source_ledgers=source_ledgers)
        for family_id in RQ3_OUTPUT_FAMILY_IDS
        for row in surface.rows_by_family[family_id]
    )
    ledger = Rq3PostBoundaryLedger(
        schema_version=1,
        kind="g3_rq3_post_boundary_output_search",
        adapter_kind=POST_BOUNDARY_ADAPTER_KIND,
        protocol_sha256=APPROVED_PROTOCOL_SHA256,
        final_evidence_path=surface.selection_path,
        final_rq2_evidence_sha256=str(surface.final_rq2_evidence_sha256),
        selected_rq2_row_id=str(surface.selected_rq2_row_id),
        feature=Rq3FeatureBinding(
            manifest_path=surface.feature_manifest_path,
            manifest_sha256=surface.feature_manifest_sha256,
            manifest_file_sha256=surface.feature_manifest_file_sha256,
            data_path=surface.feature_data_path,
            data_sha256=surface.feature_data_sha256,
            frequency_terciles=surface.frequency_terciles,
            training_count_reference=surface.training_count_reference,
            slice_membership_reference=surface.slice_membership_reference,
        ),
        source_ledgers=tuple(sorted(surface.source_ledgers)),
        logical_rows=logical_rows,
    )
    _validate_ledger(ledger)
    return ledger


def validate_rq3_post_boundary_ledger_document(
    document: object,
    *,
    expected: Rq3PostBoundaryLedger,
) -> Rq3PostBoundaryLedger:
    if not isinstance(document, dict) or document != expected.to_dict():
        raise ValueError("RQ3 post-boundary ledger differs from verified preview")
    _validate_ledger(expected)
    return expected


def load_rq3_post_boundary_ledger(
    path: Path,
    *,
    expected: Rq3PostBoundaryLedger | None = None,
) -> Rq3PostBoundaryLedger:
    document = json.loads(
        path.read_text(),
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_constant,
    )
    ledger = _ledger_from_document(document)
    if expected is not None and ledger != expected:
        raise ValueError("RQ3 post-boundary ledger differs from verified preview")
    return ledger


def persist_rq3_post_boundary_ledger(
    path: Path,
    ledger: Rq3PostBoundaryLedger,
) -> Path:
    _validate_ledger(ledger)
    content = (
        json.dumps(
            ledger.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(content)
    except FileExistsError:
        if path.read_bytes() != content:
            raise RuntimeError(f"immutable RQ3 post-boundary ledger differs: {path}")
    return path


def _validate_verification(
    verified: Rq3PostBoundaryVerification,
    *,
    expected_final_sha256: str,
    expected_selected_row_id: str,
) -> None:
    source_ids = [value.coordinate.source_id for value in verified.reusable]
    if (
        verified.adapter_kind != POST_BOUNDARY_ADAPTER_KIND
        or verified.final_evidence_sha256 != expected_final_sha256
        or verified.selected_row_id != expected_selected_row_id
        or verified.selected_history_hidden_dim != 128
        or len(source_ids) != len(RQ3_ELIGIBLE_REUSE_IDS)
        or set(source_ids) != RQ3_ELIGIBLE_REUSE_IDS
        or len(set(source_ids)) != len(source_ids)
    ):
        raise ValueError("RQ3 post-boundary verification differs from expected evidence")
    _validate_feature(verified.feature)
    training_count_sha256 = verified.feature.training_count_reference["sha256"]
    slice_membership_sha256 = verified.feature.slice_membership_reference["sha256"]
    expected_artifacts = {
        "job_contract",
        "training_metadata",
        "final_metrics",
        "training_diagnostics",
    }
    for value in verified.reusable:
        source = value.coordinate
        artifact_names = [name for name, _ in source.artifact_sha256]
        if (
            source.training_count_sha256 != training_count_sha256
            or source.slice_membership_sha256 != slice_membership_sha256
        ):
            raise ValueError(
                "RQ3 post-boundary reusable source feature count/slice identity changed"
            )
        if (
            source.history_hidden_dim != 128
            or source.source_ledger_path != value.source_ledger_path
            or source.source_ledger_sha256 != value.source_ledger_sha256
            or not _safe_relative_path(value.source_ledger_path)
            or not _valid_sha256(value.source_ledger_sha256)
            or set(artifact_names) != expected_artifacts
            or len(artifact_names) != len(expected_artifacts)
            or any(not _valid_sha256(sha256) for _, sha256 in source.artifact_sha256)
            or not _valid_sha256(source.source_evidence_sha256)
            or not _valid_sha256(source.training_count_sha256)
            or not _valid_sha256(source.slice_membership_sha256)
            or source.diagnostics_schema_version not in {1, 2}
            or source.diagnostics_epoch_count < 1
            or not math.isfinite(source.embedding_learning_rate)
            or source.embedding_learning_rate <= 0
            or not math.isfinite(source.deep_learning_rate)
            or source.deep_learning_rate <= 0
            or not isinstance(source.horizon_epochs, int)
            or source.horizon_epochs < 1
        ):
            raise ValueError("RQ3 post-boundary reusable source is not authenticated")


def _validate_surface(surface: Rq3OutputSurface) -> None:
    if (
        surface.final_rq2_evidence_sha256 != RQ2_FINAL_EVIDENCE_LOGICAL_SHA256
        or surface.selected_rq2_row_id != RQ2_FINAL_SELECTED_ROW_ID
        or surface.selected_history_hidden_dim != 128
        or not _safe_relative_path(surface.selection_path)
        or tuple(surface.rows_by_family) != RQ3_OUTPUT_FAMILY_IDS
    ):
        raise ValueError("RQ3 post-boundary surface binding is invalid")
    rows = tuple(
        row
        for family_id in RQ3_OUTPUT_FAMILY_IDS
        for row in surface.rows_by_family[family_id]
    )
    reused = [row for row in rows if row.reused_from is not None]
    source_ledgers = surface.source_ledgers
    ledger_ids = [source_id for source_id, _, _ in source_ledgers]
    source_map = {
        source_id: (path, sha256) for source_id, path, sha256 in source_ledgers
    }
    if (
        len(rows) != 45
        or any(len(surface.rows_by_family[family_id]) != 9 for family_id in RQ3_OUTPUT_FAMILY_IDS)
        or len(reused) != 7
        or {row.reused_from for row in reused} != RQ3_ELIGIBLE_REUSE_IDS
        or any(row.family_id != "rq3_output_learned" for row in reused)
        or len({row.id for row in rows}) != 45
        or len({row.run_name for row in rows}) != 45
        or any(row.batch_size != 512 or row.seed != 42 for row in rows)
        or set(ledger_ids) != RQ3_ELIGIBLE_REUSE_IDS
        or len(set(ledger_ids)) != len(ledger_ids)
        or source_ledgers != tuple(sorted(source_ledgers))
        or any(
            not _safe_relative_path(path) or not _valid_sha256(sha256)
            for _, path, sha256 in source_ledgers
        )
    ):
        raise ValueError("RQ3 post-boundary surface does not preserve 45/38 accounting")
    for row in rows:
        if (
            row.family_id not in RQ3_OUTPUT_FAMILY_IDS
            or row.history_hidden_dim != 128
            or row.batch_size != APPROVED_PROTOCOL.batch_size
            or not math.isfinite(row.embedding_learning_rate)
            or row.embedding_learning_rate <= 0
            or not math.isfinite(row.deep_learning_rate)
            or row.deep_learning_rate <= 0
            or not isinstance(row.horizon_epochs, int)
            or row.horizon_epochs < 1
        ):
            raise ValueError("RQ3 post-boundary surface row is invalid")
        source = row.authenticated_source
        if row.reused_from is None:
            if source is not None:
                raise ValueError("new RQ3 row unexpectedly has an authenticated source")
            continue
        if source is None or source.source_id != row.reused_from:
            raise ValueError("reused RQ3 row has no matching authenticated source")
        if source_map[row.reused_from] != (
            source.source_ledger_path,
            source.source_ledger_sha256,
        ):
            raise ValueError("reused RQ3 row source-ledger binding changed")
        if (
            source.history_hidden_dim != row.history_hidden_dim
            or source.embedding_learning_rate != row.embedding_learning_rate
            or source.deep_learning_rate != row.deep_learning_rate
            or source.horizon_epochs != row.horizon_epochs
            or source.run_name != row.run_name
            or {name for name, _ in source.artifact_sha256}
            != {
                "job_contract",
                "training_metadata",
                "final_metrics",
                "training_diagnostics",
            }
            or len(source.artifact_sha256) != 4
            or any(not _valid_sha256(sha256) for _, sha256 in source.artifact_sha256)
            or source.training_count_sha256
            != surface.training_count_reference["sha256"]
            or source.slice_membership_sha256
            != surface.slice_membership_reference["sha256"]
        ):
            raise ValueError("reused RQ3 row authentication changed")
    if surface.selected_rq2_row_id not in {row.reused_from for row in reused}:
        raise ValueError("selected RQ2 row is not present in the reuse surface")


def _validate_ledger(ledger: Rq3PostBoundaryLedger) -> None:
    _validate_feature(ledger.feature)
    logical = ledger.logical_rows
    physical = ledger.physical_rows
    reused = tuple(row for row in logical if row.reused_from is not None)
    family_counts = {
        family_id: sum(row.family_id == family_id for row in logical)
        for family_id in RQ3_OUTPUT_FAMILY_IDS
    }
    source_map = {
        source_id: (path, sha256)
        for source_id, path, sha256 in ledger.source_ledgers
    }
    expected_ids = tuple(
        f"{family_id}:{index:02d}"
        for family_id in RQ3_OUTPUT_FAMILY_IDS
        for index in range(1, 10)
    )
    if (
        ledger.schema_version != 1
        or ledger.kind != "g3_rq3_post_boundary_output_search"
        or ledger.adapter_kind != POST_BOUNDARY_ADAPTER_KIND
        or ledger.protocol_sha256 != APPROVED_PROTOCOL_SHA256
        or not _safe_relative_path(ledger.final_evidence_path)
        or ledger.final_rq2_evidence_sha256 != RQ2_FINAL_EVIDENCE_LOGICAL_SHA256
        or ledger.selected_rq2_row_id != RQ2_FINAL_SELECTED_ROW_ID
        or len(logical) != 45
        or len(physical) != 38
        or len({row.id for row in logical}) != 45
        or tuple(row.id for row in logical) != expected_ids
        or len({row.run_name for row in logical}) != 45
        or any(row.reused_from is not None for row in physical)
        or any(row.batch_size != 512 or row.seed != 42 for row in logical)
        or family_counts != {family_id: 9 for family_id in RQ3_OUTPUT_FAMILY_IDS}
        or {row.reused_from for row in reused} != RQ3_ELIGIBLE_REUSE_IDS
        or len(reused) != 7
        or any(row.family_id != "rq3_output_learned" for row in reused)
        or ledger.selected_rq2_row_id not in {row.reused_from for row in reused}
        or tuple(source_map) != tuple(sorted(RQ3_ELIGIBLE_REUSE_IDS))
        or ledger.source_ledgers != tuple(sorted(ledger.source_ledgers))
        or any(
            not _safe_relative_path(path) or not _valid_sha256(sha256)
            for _, path, sha256 in ledger.source_ledgers
        )
    ):
        raise ValueError("RQ3 post-boundary ledger is invalid")
    for row in logical:
        if (
            row.catalog_representation != RQ3_CATALOG_REPRESENTATIONS.get(row.family_id)
            or row.history_hidden_dim != 128
            or not math.isfinite(row.embedding_learning_rate)
            or row.embedding_learning_rate <= 0
            or not math.isfinite(row.deep_learning_rate)
            or row.deep_learning_rate <= 0
            or not isinstance(row.horizon_epochs, int)
            or row.horizon_epochs < 1
        ):
            raise ValueError("RQ3 post-boundary ledger row is invalid")
        if row.reused_from is None:
            if row.source_ledger_path is not None or row.source_ledger_sha256 is not None:
                raise ValueError("new RQ3 job has a source-ledger binding")
        elif source_map[row.reused_from] != (
            row.source_ledger_path,
            row.source_ledger_sha256,
        ):
            raise ValueError("reused RQ3 row source-ledger binding changed")


def _ledger_from_document(document: object) -> Rq3PostBoundaryLedger:
    expected_keys = {
        "schema_version",
        "kind",
        "adapter_kind",
        "protocol_sha256",
        "final_evidence_path",
        "final_rq2_evidence_sha256",
        "selected_rq2_row_id",
        "opportunity_accounting",
        "feature",
        "source_ledgers",
        "artifact_contracts",
        "logical_rows",
        "physical_rows",
        "sha256",
    }
    if not isinstance(document, dict) or set(document) != expected_keys:
        raise ValueError("RQ3 post-boundary ledger schema is invalid")
    payload = {name: value for name, value in document.items() if name != "sha256"}
    if document["sha256"] != _canonical_sha256(payload):
        raise ValueError("RQ3 post-boundary ledger hash is invalid")
    feature = document["feature"]
    source_ledgers = document["source_ledgers"]
    logical_rows = document["logical_rows"]
    if (
        not isinstance(feature, dict)
        or set(feature)
        != {
            "manifest_path",
            "manifest_sha256",
            "manifest_file_sha256",
            "data_path",
            "data_sha256",
            "frequency_terciles",
            "training_count_reference",
            "slice_membership_reference",
        }
        or not isinstance(source_ledgers, list)
        or not isinstance(logical_rows, list)
    ):
        raise ValueError("RQ3 post-boundary ledger nested schema is invalid")
    ledger = Rq3PostBoundaryLedger(
        schema_version=document["schema_version"],
        kind=document["kind"],
        adapter_kind=document["adapter_kind"],
        protocol_sha256=document["protocol_sha256"],
        final_evidence_path=document["final_evidence_path"],
        final_rq2_evidence_sha256=document["final_rq2_evidence_sha256"],
        selected_rq2_row_id=document["selected_rq2_row_id"],
        feature=Rq3FeatureBinding(**feature),
        source_ledgers=tuple(
            (value["source_id"], value["path"], value["logical_sha256"])
            for value in source_ledgers
            if isinstance(value, dict)
            and set(value) == {"source_id", "path", "logical_sha256"}
        ),
        logical_rows=tuple(_ledger_row_from_document(value) for value in logical_rows),
    )
    if (
        len(ledger.source_ledgers) != len(source_ledgers)
        or document["opportunity_accounting"]
        != ledger._payload()["opportunity_accounting"]
        or document["artifact_contracts"] != ledger._payload()["artifact_contracts"]
        or document["physical_rows"] != ledger._payload()["physical_rows"]
    ):
        raise ValueError("RQ3 post-boundary ledger derived fields changed")
    _validate_ledger(ledger)
    return ledger


def _ledger_row_from_document(value: object) -> Rq3PostBoundaryLedgerRow:
    if not isinstance(value, dict):
        raise ValueError("RQ3 post-boundary ledger row is invalid")
    expected_keys = {
        "id",
        "family_id",
        "phase",
        "stage",
        "role",
        "run_name",
        "reused_from",
        "source_ledger",
        "representation",
        "dataset",
        "training",
    }
    if set(value) != expected_keys:
        raise ValueError("RQ3 post-boundary ledger row schema is invalid")
    representation = value["representation"]
    training = value["training"]
    source = value["source_ledger"]
    if not isinstance(representation, dict) or not isinstance(training, dict):
        raise ValueError("RQ3 post-boundary row configuration is invalid")
    if source is not None and (
        not isinstance(source, dict)
        or set(source) != {"path", "logical_sha256"}
    ):
        raise ValueError("RQ3 post-boundary source-ledger binding is invalid")
    row = Rq3PostBoundaryLedgerRow(
        id=value["id"],
        family_id=value["family_id"],
        run_name=value["run_name"],
        batch_size=training["batch_size"],
        seed=training["seed"],
        embedding_learning_rate=training["embedding_learning_rate"],
        deep_learning_rate=training["deep_learning_rate"],
        horizon_epochs=training["horizon_epochs"],
        history_hidden_dim=representation["history_hidden_dim"],
        catalog_representation=representation["catalog_representation"],
        reused_from=value["reused_from"],
        source_ledger_path=source["path"] if source is not None else None,
        source_ledger_sha256=(
            source["logical_sha256"] if source is not None else None
        ),
    )
    if row.to_dict() != value:
        raise ValueError("RQ3 post-boundary ledger row differs from its contract")
    return row


def _surface_row(
    family_id: str,
    index: int,
    coordinate: object,
    *,
    width: int,
    authenticated: Mapping[str, AuthenticatedRq2Coordinate],
) -> Rq3OutputRow:
    source = (
        authenticated[coordinate.reused_from]
        if coordinate.reused_from is not None
        else None
    )
    return Rq3OutputRow(
        id=coordinate.id,
        family_id=family_id,
        run_name=(
            source.run_name
            if source is not None
            else f"g3_{family_id}_trial_{index + 1:02d}_native50m"
        ),
        batch_size=coordinate.batch_size,
        seed=coordinate.seed,
        embedding_learning_rate=coordinate.embedding_learning_rate,
        deep_learning_rate=coordinate.deep_learning_rate,
        horizon_epochs=coordinate.horizon_epochs,
        history_hidden_dim=width,
        reused_from=coordinate.reused_from,
        authenticated_source=source,
    )


def _ledger_row(
    row: Rq3OutputRow,
    *,
    source_ledgers: Mapping[str, tuple[str, str]],
) -> Rq3PostBoundaryLedgerRow:
    source = source_ledgers.get(row.reused_from) if row.reused_from is not None else None
    return Rq3PostBoundaryLedgerRow(
        id=row.id,
        family_id=row.family_id,
        run_name=row.run_name,
        batch_size=row.batch_size,
        seed=row.seed,
        embedding_learning_rate=row.embedding_learning_rate,
        deep_learning_rate=row.deep_learning_rate,
        horizon_epochs=row.horizon_epochs,
        history_hidden_dim=row.history_hidden_dim,
        catalog_representation=RQ3_CATALOG_REPRESENTATIONS[row.family_id],
        reused_from=row.reused_from,
        source_ledger_path=source[0] if source is not None else None,
        source_ledger_sha256=source[1] if source is not None else None,
    )


def _validate_feature(feature: Rq3FeatureBinding) -> None:
    if (
        not _safe_relative_path(feature.manifest_path)
        or not _safe_relative_path(feature.data_path)
        or not isinstance(feature.frequency_terciles, dict)
        or not isinstance(feature.training_count_reference, dict)
        or not isinstance(feature.slice_membership_reference, dict)
        or not all(
            _valid_sha256(value)
            for value in (
                feature.manifest_sha256,
                feature.manifest_file_sha256,
                feature.data_sha256,
            )
        )
    ):
        raise ValueError("RQ3 post-boundary feature binding is invalid")
    references: tuple[dict[str, object], ...] = (
        feature.training_count_reference,
        feature.slice_membership_reference,
    )
    catalog_count = feature.frequency_terciles.get("num_catalog_items")
    slices = feature.frequency_terciles.get("slices")
    if (
        type(catalog_count) is not int
        or catalog_count < 1
        or not isinstance(slices, dict)
        or set(slices) != {"tail", "mid", "head"}
        or any(
            not isinstance(reference, dict)
            or set(reference) != {"encoding", "length", "sha256"}
            or reference.get("encoding") != "canonical-json-integers"
            or reference.get("length") != catalog_count + 1
            or not _valid_sha256(reference.get("sha256"))
            for reference in references
        )
    ):
        raise ValueError("RQ3 post-boundary nested feature references are invalid")
    slice_counts = []
    for name in ("tail", "mid", "head"):
        value = slices[name]
        if (
            not isinstance(value, dict)
            or set(value) != {"num_items", "training_interactions"}
            or type(value["num_items"]) is not int
            or value["num_items"] < 1
            or type(value["training_interactions"]) is not int
            or value["training_interactions"] < 1
        ):
            raise ValueError("RQ3 post-boundary frequency slices are invalid")
        slice_counts.append(value["num_items"])
    if sum(slice_counts) != catalog_count:
        raise ValueError("RQ3 post-boundary frequency slice counts changed")


def _feature_dict(feature: Rq3FeatureBinding) -> dict[str, object]:
    return {
        "manifest_path": feature.manifest_path,
        "manifest_sha256": feature.manifest_sha256,
        "manifest_file_sha256": feature.manifest_file_sha256,
        "data_path": feature.data_path,
        "data_sha256": feature.data_sha256,
        "frequency_terciles": feature.frequency_terciles,
        "training_count_reference": feature.training_count_reference,
        "slice_membership_reference": feature.slice_membership_reference,
    }


def _project_file(root: Path, path: Path) -> Path:
    path = path if path.is_absolute() else root / path
    if path.is_symlink() or not path.is_file():
        raise ValueError("RQ3 post-boundary final evidence is not a regular file")
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("RQ3 post-boundary final evidence escapes the project root")
    return resolved


def _safe_relative_path(value: str) -> bool:
    path = Path(value)
    return bool(value) and not path.is_absolute() and ".." not in path.parts


def _valid_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON number {value!r}")
