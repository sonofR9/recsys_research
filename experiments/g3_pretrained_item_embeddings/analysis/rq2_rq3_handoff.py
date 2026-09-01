from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Callable, Mapping

from experiments.g3_pretrained_item_embeddings.analysis.control_calibration import (
    _load_json,
)
from experiments.g3_pretrained_item_embeddings.analysis.rq2_content_deep_lr_boundary_results import (
    verify_rq2_content_deep_lr_boundary_evidence,
)
from experiments.g3_pretrained_item_embeddings.protocol.constants import (
    APPROVED_PROTOCOL_SHA256,
)
from experiments.g3_pretrained_item_embeddings.protocol.rq3 import (
    NATIVE50_FEATURE_MANIFEST_FILE_SHA256,
    NATIVE50_FEATURE_MANIFEST_PATH,
    NATIVE50_FEATURE_MANIFEST_SHA256,
)


RQ2_RQ3_REUSE_BRIDGE_PATH = (
    "experiments/g3_pretrained_item_embeddings/evidence/"
    "rq2_content_rq3_reuse_bridge.json"
)
RQ2_RQ3_SELECTION_PATH = (
    "experiments/g3_pretrained_item_embeddings/evidence/"
    "rq2_content_selection_for_rq3.json"
)

_REUSABLE_ARTIFACTS = {
    "job_contract",
    "training_metadata",
    "final_metrics",
    "training_diagnostics",
}


@dataclass(frozen=True)
class AuthenticatedRq2HandoffSource:
    evidence: Mapping[str, object]
    selected: Mapping[str, object]
    all_tuning_rows: tuple[Mapping[str, object], ...]
    reusable_rows: tuple[Mapping[str, object], ...]
    source_ledgers: Mapping[str, Mapping[str, object]]


Rq2HandoffSourceAdapter = Callable[
    [Path, Path, str],
    AuthenticatedRq2HandoffSource,
]


def build_rq2_rq3_handoff(
    *,
    root: Path,
    final_evidence_path: Path,
    expected_final_evidence_sha256: str,
    bridge_path: Path,
    selection_path: Path,
    source_adapter: Rq2HandoffSourceAdapter | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    root = root.resolve(strict=True)
    final_evidence_path = _existing_project_file(root, final_evidence_path)
    bridge_path = _project_output(root, bridge_path)
    selection_path = _project_output(root, selection_path)
    if bridge_path == selection_path:
        raise ValueError("RQ2-to-RQ3 bridge and selection paths must differ")
    adapter = source_adapter or authenticate_boundary_rq2_handoff_source
    source = adapter(
        root,
        final_evidence_path,
        expected_final_evidence_sha256,
    )
    evidence = source.evidence
    if evidence.get("sha256") != expected_final_evidence_sha256:
        raise ValueError("RQ2-to-RQ3 handoff received another final evidence")
    selection = source.selected
    all_tuning = list(source.all_tuning_rows)
    reusable = list(source.reusable_rows)
    _validate_source(root, selection, all_tuning, reusable, source.source_ledgers)
    final_reference = _document_reference(
        root,
        final_evidence_path,
        logical_sha256=expected_final_evidence_sha256,
    )
    bridge_rows = [dict(row) | {"selection_resolved": True} for row in reusable]
    bridge = _document(
        {
            "schema_version": 1,
            "kind": "g3_rq2_content_rq3_reuse_bridge",
            "protocol_sha256": APPROVED_PROTOCOL_SHA256,
            "selection_resolved": True,
            "selected_row_id": selection["row_id"],
            "final_rq2_evidence": final_reference,
            "all_tuning_ledger": all_tuning,
            "tuning_ledger": bridge_rows,
        }
    )
    bridge_reference = _future_document_reference(root, bridge_path, bridge)
    selection_rows = [
        _selection_row(
            root,
            row,
            source_ledger=source.source_ledgers[str(row["row_id"])],
            bridge_reference=bridge_reference,
        )
        for row in reusable
    ]
    feature_manifest = _feature_manifest_reference(root)
    handoff = _document(
        {
            "schema_version": 1,
            "kind": "g3_rq2_content_selection_for_rq3",
            "protocol_sha256": APPROVED_PROTOCOL_SHA256,
            "selected_family_id": "rq2_content_concat",
            "selected_history_hidden_dim": selection["capacity"],
            "selection_resolved": True,
            "feature_manifest": feature_manifest,
            "source_evidence": [bridge_reference],
            "rows": selection_rows,
        }
    )
    return bridge, handoff


def persist_rq2_rq3_handoff(
    *,
    root: Path,
    final_evidence_path: Path,
    expected_final_evidence_sha256: str,
    bridge_path: Path,
    selection_path: Path,
    source_adapter: Rq2HandoffSourceAdapter | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    root = root.resolve(strict=True)
    bridge_path = _project_output(root, bridge_path)
    selection_path = _project_output(root, selection_path)
    bridge, selection = build_rq2_rq3_handoff(
        root=root,
        final_evidence_path=final_evidence_path,
        expected_final_evidence_sha256=expected_final_evidence_sha256,
        bridge_path=bridge_path,
        selection_path=selection_path,
        source_adapter=source_adapter,
    )
    _persist_immutable(bridge_path, bridge)
    _persist_immutable(selection_path, selection)
    return bridge, selection


def load_rq2_rq3_handoff(
    *,
    root: Path,
    final_evidence_path: Path,
    expected_final_evidence_sha256: str,
    bridge_path: Path,
    selection_path: Path,
    source_adapter: Rq2HandoffSourceAdapter | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    root = root.resolve(strict=True)
    bridge_path = _project_output(root, bridge_path)
    selection_path = _project_output(root, selection_path)
    expected_bridge, expected_selection = build_rq2_rq3_handoff(
        root=root,
        final_evidence_path=final_evidence_path,
        expected_final_evidence_sha256=expected_final_evidence_sha256,
        bridge_path=bridge_path,
        selection_path=selection_path,
        source_adapter=source_adapter,
    )
    bridge = _load_json(bridge_path)
    selection = _load_json(selection_path)
    if bridge != expected_bridge or selection != expected_selection:
        raise ValueError("RQ2-to-RQ3 handoff differs from authenticated RQ2 evidence")
    return bridge, selection


def authenticate_boundary_rq2_handoff_source(
    root: Path,
    evidence_path: Path,
    expected_sha256: str,
) -> AuthenticatedRq2HandoffSource:
    evidence = verify_rq2_content_deep_lr_boundary_evidence(
        evidence_path,
        root=root,
    )
    if evidence.get("sha256") != expected_sha256:
        raise ValueError("RQ2 boundary adapter received another evidence document")
    final_selection = evidence.get("final_content_selection")
    rq3_inputs = evidence.get("rq3_inputs")
    all_tuning = evidence.get("all_tuning_ledger")
    if (
        not isinstance(final_selection, dict)
        or final_selection.get("status") != "resolved"
        or not isinstance(final_selection.get("selected"), dict)
        or final_selection.get("provisional_selected") is not None
        or not isinstance(rq3_inputs, dict)
        or rq3_inputs.get("selected_content_input") != final_selection["selected"]
        or not isinstance(all_tuning, list)
        or any(not isinstance(row, dict) for row in all_tuning)
    ):
        raise ValueError("RQ2-to-RQ3 final content selection is not resolved")
    reusable = rq3_inputs.get("reusable_width_32_content_rows")
    expected_reusable = [
        row
        for row in all_tuning
        if row.get("family_id") == "rq2_content_concat"
        and row.get("capacity") == final_selection["selected"].get("capacity")
    ]
    if not isinstance(reusable, list) or reusable != expected_reusable:
        raise ValueError("RQ2-to-RQ3 reusable rows differ from final tuning evidence")
    return AuthenticatedRq2HandoffSource(
        evidence=evidence,
        selected=final_selection["selected"],
        all_tuning_rows=tuple(all_tuning),
        reusable_rows=tuple(reusable),
        source_ledgers=_boundary_source_ledger_references(root, evidence),
    )


def _validate_source(
    root: Path,
    selection: Mapping[str, object],
    all_tuning: list[Mapping[str, object]],
    reusable: list[Mapping[str, object]],
    source_ledgers: Mapping[str, Mapping[str, object]],
) -> None:
    if (
        selection.get("family_id") != "rq2_content_concat"
        or type(selection.get("capacity")) is not int
        or int(selection["capacity"]) < 1
        or not all_tuning
        or not reusable
        or len(reusable) > 9
    ):
        raise ValueError("RQ2-to-RQ3 authenticated source is invalid")
    all_identities = [row.get("row_id") for row in all_tuning]
    reusable_identities = [row.get("row_id") for row in reusable]
    if (
        any(not isinstance(identity, str) or not identity for identity in all_identities)
        or len(set(all_identities)) != len(all_identities)
        or any(row not in all_tuning for row in reusable)
        or any(
            row.get("family_id") != "rq2_content_concat"
            or row.get("capacity") != selection["capacity"]
            for row in reusable
        )
        or set(reusable_identities) != set(source_ledgers)
    ):
        raise ValueError("RQ2-to-RQ3 authenticated rows are inconsistent")
    selected_row_id = selection.get("row_id")
    selected_rows = [
        row for row in all_tuning if row.get("row_id") == selected_row_id
    ]
    if (
        not isinstance(selected_row_id, str)
        or len(selected_rows) != 1
        or selected_row_id not in reusable_identities
        or any(
            value != selected_rows[0].get(name)
            for name, value in selection.items()
            if name != "combined_manifest_order"
        )
    ):
        raise ValueError("RQ2-to-RQ3 selected row is absent from tuning evidence")
    for reference in source_ledgers.values():
        _referenced_document(root, reference)


def _boundary_source_ledger_references(
    root: Path,
    evidence: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    horizon = _referenced_document(root, evidence.get("content_horizon_evidence"))
    next_stage = _referenced_document(
        root, evidence.get("resolved_next_stage_evidence")
    )
    references = {
        **{
            f"rq2_content_concat:{index:02d}": horizon["content_horizon_ledger"]
            for index in range(10, 13)
        },
        **{
            f"rq2_content_concat:{index:02d}": next_stage["next_stage_ledger"]
            for index in range(13, 16)
        },
        **{
            f"rq2_content_concat:{index:02d}": evidence[
                "content_deep_lr_boundary_ledger"
            ]
            for index in range(16, 19)
        },
    }
    for reference in references.values():
        _referenced_document(root, reference)
    return references


def _selection_row(
    root: Path,
    row: Mapping[str, object],
    *,
    source_ledger: Mapping[str, object],
    bridge_reference: Mapping[str, object],
) -> dict[str, object]:
    artifacts = row.get("artifacts")
    if not isinstance(artifacts, dict) or not _REUSABLE_ARTIFACTS <= set(artifacts):
        raise ValueError("RQ2-to-RQ3 reusable row artifacts are incomplete")
    selected_artifacts = {
        name: artifacts[name] for name in sorted(_REUSABLE_ARTIFACTS)
    }
    for reference in selected_artifacts.values():
        _validate_physical_reference(root, reference, logical=False)
    return {
        "source_id": row["row_id"],
        "source_ledger_row_id": row["row_id"],
        "source_ledger": source_ledger,
        "source_evidence": bridge_reference,
        "run_name": row["run_name"],
        "family_id": row["family_id"],
        "history_hidden_dim": row["capacity"],
        "embedding_learning_rate": row["embedding_learning_rate"],
        "deep_learning_rate": row["deep_learning_rate"],
        "horizon_epochs": row["horizon_epochs"],
        "artifacts": selected_artifacts,
    }


def _feature_manifest_reference(root: Path) -> dict[str, object]:
    path = _existing_project_file(root, root / NATIVE50_FEATURE_MANIFEST_PATH)
    reference = _document_reference(
        root,
        path,
        logical_sha256=NATIVE50_FEATURE_MANIFEST_SHA256,
        document_has_hash=False,
    )
    if reference["sha256"] != NATIVE50_FEATURE_MANIFEST_FILE_SHA256:
        raise ValueError("RQ2-to-RQ3 feature-manifest bytes changed")
    return reference


def _referenced_document(root: Path, value: object) -> dict[str, object]:
    reference = _validate_physical_reference(root, value, logical=True)
    document = _load_json(root / str(reference["path"]))
    payload = {name: item for name, item in document.items() if name != "sha256"}
    logical_sha256 = _canonical_sha256(payload)
    if (
        document.get("sha256") != logical_sha256
        or reference["logical_sha256"] != logical_sha256
    ):
        raise ValueError("RQ2-to-RQ3 source document logical hash changed")
    return document


def _document_reference(
    root: Path,
    path: Path,
    *,
    logical_sha256: str,
    document_has_hash: bool = True,
) -> dict[str, object]:
    reference = _validate_physical_reference(
        root,
        {
            "path": str(path.relative_to(root)),
            "sha256": _file_sha256(path),
            "size_bytes": path.stat().st_size,
            "logical_sha256": logical_sha256,
        },
        logical=True,
    )
    document = _load_json(path)
    payload = (
        {name: value for name, value in document.items() if name != "sha256"}
        if document_has_hash
        else document
    )
    if _canonical_sha256(payload) != logical_sha256:
        raise ValueError("RQ2-to-RQ3 source document logical hash changed")
    return reference


def _future_document_reference(
    root: Path,
    path: Path,
    document: Mapping[str, object],
) -> dict[str, object]:
    content = _document_bytes(document)
    return {
        "path": str(path.relative_to(root)),
        "sha256": hashlib.sha256(content).hexdigest(),
        "size_bytes": len(content),
        "logical_sha256": document["sha256"],
    }


def _validate_physical_reference(
    root: Path,
    value: object,
    *,
    logical: bool,
) -> dict[str, object]:
    keys = {"path", "sha256", "size_bytes"}
    if logical:
        keys.add("logical_sha256")
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError("RQ2-to-RQ3 source reference schema is invalid")
    path = _existing_project_file(root, root / str(value["path"]))
    if (
        value["size_bytes"] != path.stat().st_size
        or value["sha256"] != _file_sha256(path)
    ):
        raise ValueError("RQ2-to-RQ3 source reference differs from disk")
    return dict(value)


def _persist_immutable(path: Path, document: Mapping[str, object]) -> None:
    content = _document_bytes(document)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(content)
    except FileExistsError:
        if path.read_bytes() != content:
            raise RuntimeError(f"immutable RQ2-to-RQ3 handoff differs: {path}")


def _project_output(root: Path, path: Path) -> Path:
    path = path if path.is_absolute() else root / path
    resolved = path.resolve()
    if not resolved.is_relative_to(root) or path.is_symlink():
        raise ValueError("RQ2-to-RQ3 output escapes the project root")
    return resolved


def _existing_project_file(root: Path, path: Path) -> Path:
    path = path if path.is_absolute() else root / path
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"RQ2-to-RQ3 input is not a project file: {path}")
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("RQ2-to-RQ3 input escapes the project root")
    return resolved


def _document(payload: Mapping[str, object]) -> dict[str, object]:
    document = dict(payload)
    document["sha256"] = _canonical_sha256(document)
    return document


def _document_bytes(document: Mapping[str, object]) -> bytes:
    return (_canonical_json(document) + "\n").encode()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
