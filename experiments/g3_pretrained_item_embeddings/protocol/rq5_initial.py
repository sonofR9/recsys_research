from __future__ import annotations

from dataclasses import dataclass
import base64
import hashlib
import json
import math
from pathlib import Path

from .constants import APPROVED_PROTOCOL, APPROVED_PROTOCOL_SHA256
from .rq3_post_boundary import (
    POST_BOUNDARY_ADAPTER_KIND,
    RQ2_FINAL_EVIDENCE_LOGICAL_SHA256,
    RQ2_FINAL_EVIDENCE_PATH,
    RQ2_FINAL_SELECTED_ROW_ID,
    Rq3ArtifactContract,
    Rq3FeatureBinding,
    Rq3PostBoundaryVerifier,
    compile_verified_rq3_post_boundary_surface,
    verify_final_rq2_evidence_for_rq3,
)
from .rq5 import FrozenRq2ContentBinding, Rq5GateRow, compile_rq5_gate_surface


RQ5_INITIAL_LEDGER_PATH = (
    "experiments/g3_pretrained_item_embeddings/protocol/ledgers/"
    "rq5_gate_initial_post_chain.json"
)
RQ5_INITIAL_LEDGER_LOGICAL_SHA256 = (
    "f44fe00d3a0083deb02fd2eeaccd7ae51450feb8bc7255bf6d3a26fc1f267355"
)
RQ5_ARTIFACT_CONTRACTS = (
    Rq3ArtifactContract(
        "job_contract",
        "g3_rq5_gate_job.json",
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
class Rq5FileReference:
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
class Rq5PhysicalFileReference:
    path: str
    size_bytes: int
    sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class Rq5FixedGateEvidence:
    source_id: str
    run_name: str
    history_hidden_dim: int
    embedding_learning_rate: float
    deep_learning_rate: float
    horizon_epochs: int
    source_ledger: Rq5FileReference
    source_evidence: Rq5FileReference
    queue_job_id: str
    queue_record: Rq5PhysicalFileReference
    artifacts: tuple[tuple[str, Rq5PhysicalFileReference], ...]
    training_count_sha256: str
    slice_membership_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "run_name": self.run_name,
            "history_hidden_dim": self.history_hidden_dim,
            "embedding_learning_rate": self.embedding_learning_rate,
            "deep_learning_rate": self.deep_learning_rate,
            "horizon_epochs": self.horizon_epochs,
            "source_ledger": self.source_ledger.to_dict(),
            "source_evidence": self.source_evidence.to_dict(),
            "queue_job": self.queue_record.to_dict() | {"job_id": self.queue_job_id},
            "artifacts": {
                name: reference.to_dict() for name, reference in self.artifacts
            },
            "training_count_sha256": self.training_count_sha256,
            "slice_membership_sha256": self.slice_membership_sha256,
        }


@dataclass(frozen=True)
class Rq5InitialLedgerRow:
    id: str
    family_id: str
    run_name: str
    batch_size: int
    seed: int
    embedding_learning_rate: float
    deep_learning_rate: float
    horizon_epochs: int
    history_hidden_dim: int
    content_gate: str
    gate_hidden_dim: int | None
    reused_from: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "family_id": self.family_id,
            "phase": "rq5_content_gate",
            "stage": "rq5_initial_post_chain",
            "role": "fixed_gate_reuse" if self.reused_from else "gate_search",
            "run_name": self.run_name,
            "reused_from": self.reused_from,
            "representation": {
                "history_representation": "id_content",
                "history_hidden_dim": self.history_hidden_dim,
                "catalog_representation": "learned_id",
                "content_gate": self.content_gate,
                "gate_hidden_dim": self.gate_hidden_dim,
                "gate_input": (
                    None
                    if self.content_gate != "frequency"
                    else "standardized_log1p_training_count"
                ),
                "gate_activation": (
                    None if self.content_gate == "fixed" else "sigmoid"
                ),
                "content_attachment": "before_id_content_densenet",
            },
            "dataset": {
                "size": APPROVED_PROTOCOL.main_dataset_size,
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
class Rq5InitialLedger:
    schema_version: int
    kind: str
    adapter_kind: str
    protocol_sha256: str
    final_rq2_evidence: Rq5FileReference
    selected_rq2_row_id: str
    feature: Rq3FeatureBinding
    fixed_gate: Rq5InitialLedgerRow
    fixed_gate_evidence: Rq5FixedGateEvidence
    family_opportunity_budgets: dict[str, int]
    stage_physical_jobs: int
    deferred_frequency_horizon: dict[str, object]
    physical_rows: tuple[Rq5InitialLedgerRow, ...]

    @property
    def rows(self) -> tuple[Rq5InitialLedgerRow, ...]:
        return self.physical_rows

    @property
    def logical_rows(self) -> tuple[Rq5InitialLedgerRow, ...]:
        return (self.fixed_gate, *self.physical_rows)

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self._payload())

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "adapter_kind": self.adapter_kind,
            "protocol_sha256": self.protocol_sha256,
            "final_rq2_evidence": self.final_rq2_evidence.to_dict(),
            "selected_rq2_row_id": self.selected_rq2_row_id,
            "feature": _feature_dict(self.feature),
            "fixed_gate": self.fixed_gate.to_dict(),
            "fixed_gate_evidence": self.fixed_gate_evidence.to_dict(),
            "family_opportunity_budgets": self.family_opportunity_budgets,
            "stage_physical_jobs": self.stage_physical_jobs,
            "deferred_frequency_horizon": self.deferred_frequency_horizon,
            "artifact_contracts": [
                contract.to_dict() for contract in RQ5_ARTIFACT_CONTRACTS
            ],
            "logical_rows": [row.to_dict() for row in self.logical_rows],
            "physical_rows": [row.to_dict() for row in self.physical_rows],
        }

    def to_dict(self) -> dict[str, object]:
        return self._payload() | {"sha256": self.sha256}


def compile_rq5_initial_ledger(
    *,
    root: Path,
    final_rq2_evidence_path: Path = Path(RQ2_FINAL_EVIDENCE_PATH),
    expected_final_rq2_sha256: str = RQ2_FINAL_EVIDENCE_LOGICAL_SHA256,
    expected_selected_rq2_row_id: str = RQ2_FINAL_SELECTED_ROW_ID,
    adapter_kind: str = POST_BOUNDARY_ADAPTER_KIND,
    verifier: Rq3PostBoundaryVerifier = verify_final_rq2_evidence_for_rq3,
) -> Rq5InitialLedger:
    root = root.resolve(strict=True)
    if (
        expected_final_rq2_sha256 != RQ2_FINAL_EVIDENCE_LOGICAL_SHA256
        or expected_selected_rq2_row_id != RQ2_FINAL_SELECTED_ROW_ID
        or adapter_kind != POST_BOUNDARY_ADAPTER_KIND
    ):
        raise ValueError("RQ5 requires the exact reviewed final RQ2 selection")
    predecessor = compile_verified_rq3_post_boundary_surface(
        root=root,
        final_evidence_path=final_rq2_evidence_path,
        expected_final_rq2_evidence_sha256=expected_final_rq2_sha256,
        expected_selected_rq2_row_id=expected_selected_rq2_row_id,
        adapter_kind=adapter_kind,
        verifier=verifier,
    )
    surface = compile_rq5_gate_surface(
        predecessor=predecessor,
        binding=FrozenRq2ContentBinding(
            selection_path=predecessor.selection_path,
            selection_sha256=predecessor.selection_sha256,
            selected_source_id=expected_selected_rq2_row_id,
        ),
    )
    selected = next(
        row
        for row in predecessor.rows_by_family["rq3_output_learned"]
        if row.reused_from == expected_selected_rq2_row_id
    )
    source = selected.authenticated_source
    if source is None:
        raise ValueError("RQ5 fixed gate has no authenticated final RQ2 evidence")
    final_path = _project_file(root, final_rq2_evidence_path)
    feature = Rq3FeatureBinding(
        manifest_path=surface.feature_manifest_path,
        manifest_sha256=surface.feature_manifest_sha256,
        manifest_file_sha256=surface.feature_manifest_file_sha256,
        data_path=surface.feature_data_path,
        data_sha256=surface.feature_data_sha256,
        frequency_terciles=surface.frequency_terciles,
        training_count_reference=surface.training_count_reference,
        slice_membership_reference=surface.slice_membership_reference,
    )
    fixed_evidence = _fixed_gate_evidence(
        root=root,
        final_document=_load_json(final_path),
        source=source,
        feature=feature,
    )
    physical = tuple(
        _row(row)
        for row in (*surface.global_gate_rows, *surface.frequency_gate_rows)
        if row.reused_from is None
    )
    ledger = Rq5InitialLedger(
        schema_version=1,
        kind="g3_rq5_initial_gate_search",
        adapter_kind=adapter_kind,
        protocol_sha256=APPROVED_PROTOCOL_SHA256,
        final_rq2_evidence=Rq5FileReference(
            path=str(final_path.relative_to(root)),
            size_bytes=final_path.stat().st_size,
            sha256=_file_sha256(final_path),
            logical_sha256=expected_final_rq2_sha256,
        ),
        selected_rq2_row_id=expected_selected_rq2_row_id,
        feature=feature,
        fixed_gate=_row(surface.fixed_gate),
        fixed_gate_evidence=fixed_evidence,
        family_opportunity_budgets={
            "rq5_global_gate": 12,
            "rq5_frequency_gate": 12,
        },
        stage_physical_jobs=21,
        deferred_frequency_horizon={
            "logical_opportunities": 3,
            "materialize_only_after_capacity_selection": True,
        },
        physical_rows=physical,
    )
    _validate_ledger(ledger)
    return ledger


def validate_rq5_initial_ledger_document(
    document: object, *, expected: Rq5InitialLedger
) -> Rq5InitialLedger:
    if not isinstance(document, dict):
        raise ValueError("RQ5 initial ledger must be an object")
    _validate_exact_json_types(document, expected.to_dict(), path="ledger")
    if document != expected.to_dict():
        raise ValueError("RQ5 initial ledger differs from verified preview")
    _validate_ledger(expected)
    return expected


def load_rq5_initial_ledger(
    path: Path, *, expected: Rq5InitialLedger | None = None
) -> Rq5InitialLedger:
    document = _load_json(path)
    ledger = _ledger_from_document(document)
    if expected is not None and ledger != expected:
        raise ValueError("RQ5 initial ledger differs from verified preview")
    return ledger


def persist_rq5_initial_ledger(path: Path, ledger: Rq5InitialLedger) -> Path:
    _validate_ledger(ledger)
    content = (_canonical_json(ledger.to_dict()) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(content)
    except FileExistsError:
        if path.read_bytes() != content:
            raise RuntimeError(f"immutable RQ5 initial ledger differs: {path}")
    return path


def verify_rq5_initial_input_files(root: Path, ledger: Rq5InitialLedger) -> Path:
    root = root.resolve(strict=True)
    final_path = _bound_file(root, ledger.final_rq2_evidence.path)
    if (
        final_path.stat().st_size != ledger.final_rq2_evidence.size_bytes
        or _file_sha256(final_path) != ledger.final_rq2_evidence.sha256
    ):
        raise ValueError("RQ5 final RQ2 evidence changed before launch")
    _verify_fixed_gate_evidence(root, ledger)
    manifest_path = _bound_file(root, ledger.feature.manifest_path)
    if _file_sha256(manifest_path) != ledger.feature.manifest_file_sha256:
        raise ValueError("RQ5 feature manifest changed before launch")
    feature_path = _bound_file(root, ledger.feature.data_path)
    if _file_sha256(feature_path) != ledger.feature.data_sha256:
        raise ValueError("RQ5 feature data changed before launch")
    return feature_path


def _fixed_gate_evidence(
    *, root: Path, final_document: dict[str, object], source: object, feature: Rq3FeatureBinding
) -> Rq5FixedGateEvidence:
    selection = final_document.get("final_content_selection")
    inputs = final_document.get("rq3_inputs")
    selected = selection.get("selected") if isinstance(selection, dict) else None
    source_ledgers = inputs.get("reuse_source_ledgers") if isinstance(inputs, dict) else None
    if not isinstance(selected, dict) or not isinstance(source_ledgers, dict):
        raise ValueError("RQ5 final evidence lacks the selected source bindings")
    source_id = getattr(source, "source_id")
    artifacts = selected.get("artifacts")
    queue = selected.get("queue_job")
    source_ledger = source_ledgers.get(source_id)
    source_evidence = final_document.get("diagnostic_evidence")
    if (
        selected.get("row_id") != source_id
        or selected.get("run_name") != getattr(source, "run_name")
        or selected.get("capacity") != getattr(source, "history_hidden_dim")
        or selected.get("embedding_learning_rate") != getattr(source, "embedding_learning_rate")
        or selected.get("deep_learning_rate") != getattr(source, "deep_learning_rate")
        or selected.get("horizon_epochs") != getattr(source, "horizon_epochs")
        or not isinstance(artifacts, dict)
        or set(artifacts) != {contract.name for contract in RQ5_ARTIFACT_CONTRACTS}
        or not isinstance(queue, dict)
        or not isinstance(source_ledger, dict)
        or not isinstance(source_evidence, dict)
    ):
        raise ValueError("RQ5 selected source differs from authenticated RQ2 evidence")
    artifact_refs = tuple(
        sorted(
            (name, _physical_reference(root, reference))
            for name, reference in artifacts.items()
        )
    )
    authenticated_artifacts = dict(getattr(source, "artifact_sha256"))
    if any(
        dict(artifact_refs)[name].sha256 != value
        for name, value in authenticated_artifacts.items()
    ):
        raise ValueError("RQ5 selected artifacts differ from authenticated RQ2 source")
    ledger_ref = _logical_reference(root, source_ledger)
    evidence_ref = _logical_reference(root, source_evidence)
    if (
        ledger_ref.path != getattr(source, "source_ledger_path")
        or ledger_ref.logical_sha256 != getattr(source, "source_ledger_sha256")
        or getattr(source, "training_count_sha256")
        != feature.training_count_reference.get("sha256")
        or getattr(source, "slice_membership_sha256")
        != feature.slice_membership_reference.get("sha256")
    ):
        raise ValueError("RQ5 selected source provenance differs from authentication")
    if set(queue) != {"job_id", "path", "size_bytes", "sha256"}:
        raise ValueError("RQ5 selected queue reference is invalid")
    return Rq5FixedGateEvidence(
        source_id=source_id,
        run_name=getattr(source, "run_name"),
        history_hidden_dim=getattr(source, "history_hidden_dim"),
        embedding_learning_rate=getattr(source, "embedding_learning_rate"),
        deep_learning_rate=getattr(source, "deep_learning_rate"),
        horizon_epochs=getattr(source, "horizon_epochs"),
        source_ledger=ledger_ref,
        source_evidence=evidence_ref,
        queue_job_id=str(queue["job_id"]),
        queue_record=_physical_reference(root, queue),
        artifacts=artifact_refs,
        training_count_sha256=getattr(source, "training_count_sha256"),
        slice_membership_sha256=getattr(source, "slice_membership_sha256"),
    )


def _verify_fixed_gate_evidence(root: Path, ledger: Rq5InitialLedger) -> None:
    evidence = ledger.fixed_gate_evidence
    _verify_reference(root, evidence.source_ledger)
    _verify_reference(root, evidence.source_evidence)
    _verify_reference(root, evidence.queue_record)
    for _, reference in evidence.artifacts:
        _verify_reference(root, reference)
    final = _load_json(_bound_file(root, ledger.final_rq2_evidence.path))
    selection = final.get("final_content_selection")
    inputs = final.get("rq3_inputs")
    selected = selection.get("selected") if isinstance(selection, dict) else None
    source_ledgers = inputs.get("reuse_source_ledgers") if isinstance(inputs, dict) else None
    if (
        not isinstance(selected, dict)
        or not isinstance(source_ledgers, dict)
        or selected.get("row_id") != evidence.source_id
        or selected.get("run_name") != evidence.run_name
        or selected.get("artifacts")
        != {name: reference.to_dict() for name, reference in evidence.artifacts}
        or selected.get("queue_job")
        != evidence.queue_record.to_dict() | {"job_id": evidence.queue_job_id}
        or source_ledgers.get(evidence.source_id) != evidence.source_ledger.to_dict()
        or final.get("diagnostic_evidence") != evidence.source_evidence.to_dict()
    ):
        raise ValueError("RQ5 fixed-gate binding differs from final RQ2 evidence")
    source_ledger = _load_json(_bound_file(root, evidence.source_ledger.path))
    source_document = _load_json(_bound_file(root, evidence.source_evidence.path))
    if (
        source_ledger.get("sha256") != evidence.source_ledger.logical_sha256
        or source_document.get("sha256") != evidence.source_evidence.logical_sha256
    ):
        raise ValueError("RQ5 fixed-gate logical source hash changed")
    rows = source_ledger.get("rows")
    source_row = next(
        (row for row in rows if isinstance(row, dict) and row.get("id") == evidence.source_id),
        None,
    ) if isinstance(rows, list) else None
    queue = _load_json(_bound_file(root, evidence.queue_record.path))
    artifacts = dict(evidence.artifacts)
    contract = _load_json(_bound_file(root, artifacts["job_contract"].path))
    source_batch = source_document.get("queue_batch")
    queue_times = tuple(
        queue.get(name) for name in ("submitted_at", "dispatched_at", "finished_at")
    )
    expected_source_script = (
        root
        / "experiments/g3_pretrained_item_embeddings/launchers/"
        "run_rq2_unexpected_diagnostic.py"
    ).resolve(strict=True)
    if (
        not isinstance(source_row, dict)
        or set(queue) != {
            "id", "batch_id", "data_group", "dispatched_at", "environment",
            "exit_code", "finished_at", "run", "script", "submitted_at",
        }
        or queue.get("id") != evidence.queue_job_id
        or not isinstance(source_batch, dict)
        or queue.get("batch_id") != source_batch.get("batch_id")
        or queue.get("data_group") != "g3-native50m-likes"
        or queue.get("run") != evidence.run_name
        or queue.get("exit_code") != 0
        or Path(str(queue.get("script"))).resolve() != expected_source_script
        or not all(_finite_number(value) for value in queue_times)
        or not float(queue_times[0]) <= float(queue_times[1]) <= float(queue_times[2])
        or contract != {
            "job": source_row,
            "ledger_path": str((root / evidence.source_ledger.path).resolve()),
            "ledger_sha256": evidence.source_ledger.logical_sha256,
            "row_id": evidence.source_id,
        }
    ):
        raise ValueError("RQ5 fixed-gate queue or job contract differs")
    environment = queue.get("environment")
    pairs = [entry.split("=", 1) for entry in environment if isinstance(entry, str) and "=" in entry] if isinstance(environment, list) else []
    values = dict(pairs)
    encoded = values.get("G3_RQ2_UNEXPECTED_DIAGNOSTIC_JOB_B64")
    try:
        payload = json.loads(base64.urlsafe_b64decode(str(encoded)).decode())
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("RQ5 fixed-gate queue payload is invalid") from error
    if (
        len(pairs) != len(values) == 3
        or values.get("WANDB_MODE") != "offline"
        or set(values) != {
            "WANDB_MODE",
            "G3_RQ2_UNEXPECTED_DIAGNOSTIC_JOB_B64",
            "G3_RQ2_UNEXPECTED_DIAGNOSTIC_LEDGER_PATH",
        }
        or values.get("G3_RQ2_UNEXPECTED_DIAGNOSTIC_LEDGER_PATH")
        != str((root / evidence.source_ledger.path).resolve())
        or payload != {
            "job": source_row,
            "ledger_sha256": evidence.source_ledger.logical_sha256,
            "row_id": evidence.source_id,
        }
    ):
        raise ValueError("RQ5 fixed-gate queue payload differs")
    source_rows = source_document.get("diagnostic_tuning_ledger")
    source_result = next(
        (row for row in source_rows if isinstance(row, dict) and row.get("row_id") == evidence.source_id),
        None,
    ) if isinstance(source_rows, list) else None
    if (
        not isinstance(source_result, dict)
        or source_result.get("run_name") != evidence.run_name
        or source_result.get("artifacts")
        != {name: reference.to_dict() for name, reference in evidence.artifacts}
        or source_result.get("queue_job")
        != evidence.queue_record.to_dict() | {"job_id": evidence.queue_job_id}
    ):
        raise ValueError("RQ5 fixed-gate result differs from bound source evidence")


def _row(row: Rq5GateRow) -> Rq5InitialLedgerRow:
    return Rq5InitialLedgerRow(
        id=row.id,
        family_id=row.family_id,
        run_name=row.run_name,
        batch_size=row.batch_size,
        seed=row.seed,
        embedding_learning_rate=row.embedding_learning_rate,
        deep_learning_rate=row.deep_learning_rate,
        horizon_epochs=row.horizon_epochs,
        history_hidden_dim=row.history_hidden_dim,
        content_gate=row.content_gate,
        gate_hidden_dim=row.gate_hidden_dim,
        reused_from=row.reused_from,
    )


def _validate_ledger(ledger: Rq5InitialLedger) -> None:
    physical = ledger.physical_rows
    global_rows = tuple(row for row in physical if row.family_id == "rq5_global_gate")
    frequency_rows = tuple(
        row for row in physical if row.family_id == "rq5_frequency_gate"
    )
    if (
        ledger.schema_version != 1
        or ledger.kind != "g3_rq5_initial_gate_search"
        or ledger.adapter_kind != POST_BOUNDARY_ADAPTER_KIND
        or ledger.protocol_sha256 != APPROVED_PROTOCOL_SHA256
        or ledger.final_rq2_evidence.logical_sha256
        != RQ2_FINAL_EVIDENCE_LOGICAL_SHA256
        or ledger.selected_rq2_row_id != RQ2_FINAL_SELECTED_ROW_ID
        or ledger.family_opportunity_budgets
        != {"rq5_global_gate": 12, "rq5_frequency_gate": 12}
        or ledger.stage_physical_jobs != 21
        or len(physical) != 21
        or len(global_rows) != 12
        or len(frequency_rows) != 9
        or any(
            row.content_gate != "global" or row.gate_hidden_dim is not None
            for row in global_rows
        )
        or any(row.content_gate != "frequency" for row in frequency_rows)
        or {row.gate_hidden_dim for row in frequency_rows} != {4, 8, 16}
        or ledger.deferred_frequency_horizon
        != {
            "logical_opportunities": 3,
            "materialize_only_after_capacity_selection": True,
        }
        or len({row.id for row in ledger.logical_rows}) != 22
        or len({row.run_name for row in physical}) != 21
        or any(row.reused_from is not None for row in physical)
        or ledger.fixed_gate.id != "rq5_fixed_gate:reuse"
        or ledger.fixed_gate.family_id != "rq2_content_concat"
        or ledger.fixed_gate.reused_from != RQ2_FINAL_SELECTED_ROW_ID
        or ledger.fixed_gate.content_gate != "fixed"
        or ledger.fixed_gate_evidence.source_id != RQ2_FINAL_SELECTED_ROW_ID
        or (
            ledger.fixed_gate.run_name,
            ledger.fixed_gate.history_hidden_dim,
            ledger.fixed_gate.embedding_learning_rate,
            ledger.fixed_gate.deep_learning_rate,
            ledger.fixed_gate.horizon_epochs,
        )
        != (
            ledger.fixed_gate_evidence.run_name,
            ledger.fixed_gate_evidence.history_hidden_dim,
            ledger.fixed_gate_evidence.embedding_learning_rate,
            ledger.fixed_gate_evidence.deep_learning_rate,
            ledger.fixed_gate_evidence.horizon_epochs,
        )
    ):
        raise ValueError("RQ5 initial ledger violates the approved opportunity surface")
    if (
        not _safe_path(ledger.final_rq2_evidence.path)
        or not _valid_sha(ledger.final_rq2_evidence.sha256)
        or not _valid_sha(ledger.feature.manifest_sha256)
        or not _valid_sha(ledger.feature.manifest_file_sha256)
        or not _valid_sha(ledger.feature.data_sha256)
        or not _safe_path(ledger.feature.data_path)
        or ledger.fixed_gate_evidence.training_count_sha256
        != ledger.feature.training_count_reference.get("sha256")
        or ledger.fixed_gate_evidence.slice_membership_sha256
        != ledger.feature.slice_membership_reference.get("sha256")
        or not _valid_logical_reference(ledger.fixed_gate_evidence.source_ledger)
        or not _valid_logical_reference(ledger.fixed_gate_evidence.source_evidence)
        or not ledger.fixed_gate_evidence.queue_job_id
        or not _valid_physical_reference(ledger.fixed_gate_evidence.queue_record)
        or {name for name, _ in ledger.fixed_gate_evidence.artifacts}
        != {contract.name for contract in RQ5_ARTIFACT_CONTRACTS}
        or len(ledger.fixed_gate_evidence.artifacts) != len(RQ5_ARTIFACT_CONTRACTS)
        or any(
            not _valid_physical_reference(reference)
            for _, reference in ledger.fixed_gate_evidence.artifacts
        )
    ):
        raise ValueError("RQ5 fixed-gate evidence or feature identity is invalid")
    for row in ledger.logical_rows:
        if (
            row.batch_size != 512
            or row.seed != 42
            or row.history_hidden_dim != 128
            or row.content_gate not in {"fixed", "global", "frequency"}
            or (
                row.content_gate == "frequency"
                and row.gate_hidden_dim not in {4, 8, 16}
            )
            or (row.content_gate != "frequency" and row.gate_hidden_dim is not None)
            or not math.isfinite(row.embedding_learning_rate)
            or row.embedding_learning_rate <= 0
            or not math.isfinite(row.deep_learning_rate)
            or row.deep_learning_rate <= 0
            or type(row.horizon_epochs) is not int
            or row.horizon_epochs < 1
        ):
            raise ValueError("RQ5 initial ledger row is invalid")


def _ledger_from_document(document: object) -> Rq5InitialLedger:
    if not isinstance(document, dict):
        raise ValueError("RQ5 initial ledger must be an object")
    expected_keys = {
        "schema_version",
        "kind",
        "adapter_kind",
        "protocol_sha256",
        "final_rq2_evidence",
        "selected_rq2_row_id",
        "feature",
        "fixed_gate",
        "fixed_gate_evidence",
        "family_opportunity_budgets",
        "stage_physical_jobs",
        "deferred_frequency_horizon",
        "artifact_contracts",
        "logical_rows",
        "physical_rows",
        "sha256",
    }
    if set(document) != expected_keys:
        raise ValueError("RQ5 initial ledger schema is invalid")
    payload = {name: value for name, value in document.items() if name != "sha256"}
    if document["sha256"] != _canonical_sha256(payload):
        raise ValueError("RQ5 initial ledger hash is invalid")
    reference = document["final_rq2_evidence"]
    feature = document["feature"]
    fixed_evidence = document["fixed_gate_evidence"]
    physical = document["physical_rows"]
    if not all(isinstance(value, dict) for value in (reference, feature, fixed_evidence)):
        raise ValueError("RQ5 initial ledger nested schema is invalid")
    if not isinstance(physical, list):
        raise ValueError("RQ5 initial ledger rows are invalid")
    source_ledger = fixed_evidence.get("source_ledger")
    source_evidence = fixed_evidence.get("source_evidence")
    queue_job = fixed_evidence.get("queue_job")
    artifacts = fixed_evidence.get("artifacts")
    if not all(
        isinstance(value, dict)
        for value in (source_ledger, source_evidence, queue_job, artifacts)
    ):
        raise ValueError("RQ5 fixed-gate evidence schema is invalid")
    ledger = Rq5InitialLedger(
        schema_version=document["schema_version"],
        kind=document["kind"],
        adapter_kind=document["adapter_kind"],
        protocol_sha256=document["protocol_sha256"],
        final_rq2_evidence=Rq5FileReference(**reference),
        selected_rq2_row_id=document["selected_rq2_row_id"],
        feature=Rq3FeatureBinding(**feature),
        fixed_gate=_row_from_document(document["fixed_gate"]),
        fixed_gate_evidence=Rq5FixedGateEvidence(
            source_id=fixed_evidence["source_id"],
            run_name=fixed_evidence["run_name"],
            history_hidden_dim=fixed_evidence["history_hidden_dim"],
            embedding_learning_rate=fixed_evidence["embedding_learning_rate"],
            deep_learning_rate=fixed_evidence["deep_learning_rate"],
            horizon_epochs=fixed_evidence["horizon_epochs"],
            source_ledger=Rq5FileReference(**source_ledger),
            source_evidence=Rq5FileReference(**source_evidence),
            queue_job_id=queue_job["job_id"],
            queue_record=Rq5PhysicalFileReference(
                **{name: queue_job[name] for name in ("path", "size_bytes", "sha256")}
            ),
            artifacts=tuple(
                sorted(
                    (name, Rq5PhysicalFileReference(**reference))
                    for name, reference in artifacts.items()
                )
            ),
            training_count_sha256=fixed_evidence["training_count_sha256"],
            slice_membership_sha256=fixed_evidence["slice_membership_sha256"],
        ),
        family_opportunity_budgets=document["family_opportunity_budgets"],
        stage_physical_jobs=document["stage_physical_jobs"],
        deferred_frequency_horizon=document["deferred_frequency_horizon"],
        physical_rows=tuple(_row_from_document(value) for value in physical),
    )
    if (
        document["artifact_contracts"]
        != [contract.to_dict() for contract in RQ5_ARTIFACT_CONTRACTS]
        or document["logical_rows"]
        != [row.to_dict() for row in ledger.logical_rows]
        or document["physical_rows"]
        != [row.to_dict() for row in ledger.physical_rows]
    ):
        raise ValueError("RQ5 initial ledger derived fields changed")
    _validate_ledger(ledger)
    return ledger


def _row_from_document(value: object) -> Rq5InitialLedgerRow:
    if not isinstance(value, dict):
        raise ValueError("RQ5 initial ledger row is invalid")
    representation = value.get("representation")
    training = value.get("training")
    if not isinstance(representation, dict) or not isinstance(training, dict):
        raise ValueError("RQ5 initial ledger row schema is invalid")
    row = Rq5InitialLedgerRow(
        id=value.get("id"),
        family_id=value.get("family_id"),
        run_name=value.get("run_name"),
        batch_size=training.get("batch_size"),
        seed=training.get("seed"),
        embedding_learning_rate=training.get("embedding_learning_rate"),
        deep_learning_rate=training.get("deep_learning_rate"),
        horizon_epochs=training.get("horizon_epochs"),
        history_hidden_dim=representation.get("history_hidden_dim"),
        content_gate=representation.get("content_gate"),
        gate_hidden_dim=representation.get("gate_hidden_dim"),
        reused_from=value.get("reused_from"),
    )
    if value != row.to_dict():
        raise ValueError("RQ5 initial ledger row fields changed")
    return row


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
    candidate = path if path.is_absolute() else root / path
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError("RQ5 bound input is not a regular file")
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("RQ5 bound input escapes the project root")
    return resolved


def _physical_reference(
    root: Path, value: object
) -> Rq5PhysicalFileReference:
    if not isinstance(value, dict) or not {"path", "size_bytes", "sha256"}.issubset(value):
        raise ValueError("RQ5 bound physical file reference is invalid")
    reference = Rq5PhysicalFileReference(
        path=value["path"],
        size_bytes=value["size_bytes"],
        sha256=value["sha256"],
    )
    if not _valid_physical_reference(reference):
        raise ValueError("RQ5 bound physical file reference is invalid")
    _verify_reference(root, reference)
    return reference


def _logical_reference(root: Path, value: object) -> Rq5FileReference:
    if not isinstance(value, dict) or set(value) != {
        "path", "size_bytes", "sha256", "logical_sha256"
    }:
        raise ValueError("RQ5 bound logical file reference is invalid")
    reference = Rq5FileReference(**value)
    if not _valid_logical_reference(reference):
        raise ValueError("RQ5 bound logical file reference is invalid")
    _verify_reference(root, reference)
    return reference


def _verify_reference(
    root: Path, reference: Rq5FileReference | Rq5PhysicalFileReference
) -> Path:
    path = _bound_file(root, reference.path)
    if path.stat().st_size != reference.size_bytes or _file_sha256(path) != reference.sha256:
        raise ValueError(f"RQ5 bound file changed: {reference.path}")
    return path


def _valid_physical_reference(reference: Rq5PhysicalFileReference) -> bool:
    return (
        _safe_path(reference.path)
        and type(reference.size_bytes) is int
        and reference.size_bytes >= 0
        and _valid_sha(reference.sha256)
    )


def _valid_logical_reference(reference: Rq5FileReference) -> bool:
    return _valid_physical_reference(
        Rq5PhysicalFileReference(reference.path, reference.size_bytes, reference.sha256)
    ) and _valid_sha(reference.logical_sha256)


def _bound_file(root: Path, value: str) -> Path:
    if not _safe_path(value):
        raise ValueError("RQ5 bound path is invalid")
    return _project_file(root, root / value)


def _safe_path(value: object) -> bool:
    if not isinstance(value, str) or not value:
        return False
    path = Path(value)
    return not path.is_absolute() and ".." not in path.parts


def _valid_sha(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite_number(value: object) -> bool:
    return type(value) in {int, float} and math.isfinite(float(value))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _load_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(
            path.read_text(),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load RQ5 initial ledger {path}") from error
    if not isinstance(value, dict):
        raise ValueError("RQ5 initial ledger must be an object")
    return value


def _validate_exact_json_types(actual: object, expected: object, *, path: str) -> None:
    if type(actual) is not type(expected):
        raise ValueError(f"RQ5 initial {path} has an invalid JSON type")
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise ValueError(f"RQ5 initial {path} has invalid keys")
        for name, expected_value in expected.items():
            _validate_exact_json_types(
                actual[name], expected_value, path=f"{path}.{name}"
            )
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError(f"RQ5 initial {path} has invalid length")
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
