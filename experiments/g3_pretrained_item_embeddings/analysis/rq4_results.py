from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from dcn.eval.ranking_evidence import load_ranking_evidence
from experiments.g3_pretrained_item_embeddings.analysis.control_calibration import (
    _file_fact,
    _load_json,
    _recompute_metrics,
)
from experiments.g3_pretrained_item_embeddings.analysis.queue_attribution import (
    verify_artifacts_in_job_window,
    verify_unique_completed_run,
)
from experiments.g3_pretrained_item_embeddings.analysis.rq1_results import (
    _efficiency,
    _ranking_slices,
    load_training_item_counts,
)
from experiments.g3_pretrained_item_embeddings.launchers.control import (
    G3_CPU_THREAD_ENVIRONMENT,
    decode_control_job,
)
from experiments.g3_pretrained_item_embeddings.protocol.constants import (
    APPROVED_PROTOCOL_SHA256,
)
from experiments.g3_pretrained_item_embeddings.protocol.rq4 import (
    RQ4_METADATA_FAMILIES,
    Rq4ExtraIdSurface,
    Rq4MetadataIdentity,
    compile_rq4_capacity_surface,
    resolve_rq4_feature_data,
)
from experiments.g3_pretrained_item_embeddings.protocol.rq3 import (
    _CatalogDiagnosticsContract,
    _FeatureIdentity,
    _validate_training_diagnostics,
)
from experiments.g3_pretrained_item_embeddings.protocol.rq4_horizon_ledger import (
    Rq4HorizonLedger,
    reconstruct_rq4_horizon_surface,
)
from experiments.g3_pretrained_item_embeddings.protocol.rq4_capacity_extension_ledger import (
    RQ4_CAPACITY_EXTENSION_LEDGER_PATH,
    Rq4CapacityExtensionLedger,
    compile_rq4_capacity_extension_ledger,
    load_rq4_capacity_extension_ledger,
)
from experiments.g3_pretrained_item_embeddings.protocol.rq4_artist_album_lr_boundary_ledger import (
    RQ4_CAPACITY_EXTENSION_SELECTION_SHA256,
    Rq4ArtistAlbumLrBoundaryLedger,
)
from experiments.g3_pretrained_item_embeddings.protocol.rq4_single_metadata_width256_ledger import (
    Rq4SingleMetadataWidth256Ledger,
)
from experiments.g3_pretrained_item_embeddings.protocol.rq4_initial_ledger import (
    RQ3_FINAL_EVIDENCE_LOGICAL_SHA256,
    RQ3_FINAL_SELECTED_ROW_ID,
    RQ4_INITIAL_ARTIFACT_CONTRACTS,
    Rq4InitialLedger,
    load_rq4_initial_ledger,
)


_CAPACITIES = (16, 32, 64)
_METRIC_NAMES = tuple(
    f"{name}@{cutoff}"
    for name in ("recall", "ndcg", "mrr", "capped_recall", "coverage")
    for cutoff in (10, 50, 100)
)
_QUEUE_RECORD_KEYS = {
    "id",
    "batch_id",
    "data_group",
    "dispatched_at",
    "environment",
    "exit_code",
    "finished_at",
    "run",
    "script",
    "submitted_at",
}
_RQ4_METADATA_DIAGNOSTICS = _CatalogDiagnosticsContract(
    components=frozenset({"catalog_encoder", "history_encoder", "sequence_model"}),
    table_parameters=frozenset({"item_encoder.item_embedding.weight"}),
    content_trainable=False,
)
_RANK_DIAGNOSTIC_RUN = "g3_rq4_album_capacity_07_rank_diagnostic_v3_native50m"
_RANK_DIAGNOSTIC_JOB_ID = "adcdfedfea454b06bba40d61ff592691"
_RANK_DIAGNOSTIC_SHA256 = (
    "2e2ae6a58a02af8e0e82d9e382c385f791f2ee5f011a1ba1cf44f5151829157f"
)
RQ4_CAPACITY_EXTENSION_SELECTION_PATH = (
    "experiments/g3_pretrained_item_embeddings/evidence/"
    "rq4_metadata_capacity_extension_selection_native50m.json"
)
RQ4_ARTIST_ALBUM_LR_SELECTION_PATH = (
    "experiments/g3_pretrained_item_embeddings/evidence/"
    "rq4_artist_album_lr_boundary_selection_native50m.json"
)
RQ4_SINGLE_METADATA_WIDTH256_SELECTION_PATH = (
    "experiments/g3_pretrained_item_embeddings/evidence/"
    "rq4_single_metadata_width256_boundary_selection_native50m.json"
)


def collect_authenticated_rq4_stage_runs(
    root: Path,
    *,
    ledger: Rq4InitialLedger
    | Rq4CapacityExtensionLedger
    | Rq4ArtistAlbumLrBoundaryLedger
    | Rq4SingleMetadataWidth256Ledger
    | Rq4HorizonLedger,
    ledger_path: Path,
    batch_id: str,
) -> list[dict[str, object]]:
    root = root.resolve(strict=True)
    if ledger_path.is_symlink():
        raise ValueError("RQ4 result ledger must not be a symlink")
    ledger_path = ledger_path.resolve(strict=True)
    _authenticated_file_fact(root, ledger_path)
    if _load_json(ledger_path) != ledger.to_dict():
        raise ValueError("RQ4 result ledger differs from the authenticated ledger")
    stage = _stage_contract(ledger)
    rows = tuple(ledger.rows)
    batch_path = root / "generated/training-queue-service/batches" / f"{batch_id}.json"
    batch = _load_json(batch_path)
    job_ids = _validate_batch(batch, batch_id=batch_id, expected_jobs=len(rows))
    identity = _metadata_identity(root, ledger)
    feature_path = root / identity.feature_data_path
    item_counts = load_training_item_counts(feature_path)
    context_path = root / "generated/logs/.ranking-evidence/g3-native50m/context.pt"
    context_reference = _authenticated_file_fact(root, context_path)
    runs = [
        _collect_stage_run(
            root=root,
            ledger=ledger,
            ledger_path=ledger_path,
            row=row,
            batch_id=batch_id,
            job_id=job_id,
            stage=stage,
            context_path=context_path,
            context_reference=context_reference,
            item_counts=item_counts,
            identity=identity,
        )
        for row, job_id in zip(rows, job_ids, strict=True)
    ]
    _validate_run_set_against_ledger(runs, ledger=ledger)
    _validate_slice_identity(runs, identity=identity)
    return runs


def select_rq4_capacity_winners(
    runs: Sequence[Mapping[str, object]],
    *,
    ledger: Rq4InitialLedger,
) -> dict[str, Mapping[str, object]]:
    _validate_run_set_against_ledger(runs, ledger=ledger)
    grouped = _group_runs(runs, opportunities=9)
    coordinate_surfaces = []
    selected = {}
    for family_id in RQ4_METADATA_FAMILIES:
        family = grouped[family_id]
        capacities = [int(run["metadata_dim"]) for run in family]
        if any(capacities.count(capacity) != 3 for capacity in _CAPACITIES):
            raise ValueError(f"RQ4 family {family_id} must test each metadata width three times")
        coordinate_surfaces.append(
            tuple(
                (
                    float(run["embedding_learning_rate"]),
                    float(run["deep_learning_rate"]),
                    int(run["horizon_epochs"]),
                    int(run["metadata_dim"]),
                )
                for run in family
            )
        )
        selected[family_id] = _select(family)
    if any(surface != coordinate_surfaces[0] for surface in coordinate_surfaces[1:]):
        raise ValueError("RQ4 metadata families do not share one capacity surface")
    return selected


def select_rq4_capacity_extension_winners(
    initial_runs: Sequence[Mapping[str, object]],
    extension_runs: Sequence[Mapping[str, object]],
    *,
    initial_ledger: Rq4InitialLedger,
    extension_ledger: Rq4CapacityExtensionLedger,
) -> dict[str, Mapping[str, object]]:
    _validate_run_set_against_ledger(initial_runs, ledger=initial_ledger)
    _validate_run_set_against_ledger(extension_runs, ledger=extension_ledger)
    runs = [*initial_runs, *extension_runs]
    grouped = _group_runs(runs, opportunities=12)
    selected = {}
    coordinate_surfaces = []
    for family_id in RQ4_METADATA_FAMILIES:
        family = grouped[family_id]
        capacities = [int(run["metadata_dim"]) for run in family]
        if any(capacities.count(capacity) != 3 for capacity in (16, 32, 64, 128)):
            raise ValueError(
                f"RQ4 family {family_id} must test each extended width three times"
            )
        coordinate_surfaces.append(
            tuple(
                sorted(
                    (
                        float(run["embedding_learning_rate"]),
                        float(run["deep_learning_rate"]),
                        int(run["horizon_epochs"]),
                        int(run["metadata_dim"]),
                    )
                    for run in family
                )
            )
        )
        selected[family_id] = _select(family)
    if any(surface != coordinate_surfaces[0] for surface in coordinate_surfaces[1:]):
        raise ValueError("RQ4 metadata families do not share one extended capacity surface")
    return selected


def assess_rq4_capacity_extension_boundaries(
    selected: Mapping[str, Mapping[str, object]],
    runs: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    grouped = _group_runs(runs, opportunities=12)
    if set(selected) != set(RQ4_METADATA_FAMILIES):
        raise ValueError("RQ4 extended boundary assessment requires every family")
    decisions = {}
    for family_id in RQ4_METADATA_FAMILIES:
        winner = selected[family_id]
        family = grouped[family_id]
        if winner not in family:
            raise ValueError(f"RQ4 extended winner is absent from {family_id}")
        capacity = int(winner["metadata_dim"])
        tested_capacities = sorted({int(run["metadata_dim"]) for run in family})
        if tested_capacities != [16, 32, 64, 128]:
            raise ValueError("RQ4 extended capacity range changed")
        capacity_direction = (
            "lower" if capacity == tested_capacities[0]
            else "upper" if capacity == tested_capacities[-1]
            else None
        )
        embedding = _tested_rate_boundary(
            float(winner["embedding_learning_rate"]),
            [float(run["embedding_learning_rate"]) for run in family],
        )
        deep = _tested_rate_boundary(
            float(winner["deep_learning_rate"]),
            [float(run["deep_learning_rate"]) for run in family],
        )
        decisions[family_id] = {
            "selected_row_id": winner["row_id"],
            "capacity": {
                "selected": capacity,
                "tested_values": tested_capacities,
                "direction": capacity_direction,
                "renewed_approval_required": capacity_direction is not None,
            },
            "embedding_learning_rate": embedding,
            "deep_learning_rate": deep,
            "learning_rate_extension_required": bool(
                embedding["direction"] or deep["direction"]
            ),
            "renewed_approval_required": capacity_direction is not None,
        }
    return decisions


def build_rq4_capacity_extension_selection_document(
    *,
    root: Path,
    initial_ledger: Rq4InitialLedger,
    initial_ledger_path: Path,
    initial_batch_id: str,
    extension_ledger: Rq4CapacityExtensionLedger,
    extension_ledger_path: Path,
    extension_batch_id: str,
) -> dict[str, object]:
    root = root.resolve(strict=True)
    initial_runs = collect_authenticated_rq4_stage_runs(
        root,
        ledger=initial_ledger,
        ledger_path=initial_ledger_path,
        batch_id=initial_batch_id,
    )
    extension_runs = collect_authenticated_rq4_stage_runs(
        root,
        ledger=extension_ledger,
        ledger_path=extension_ledger_path,
        batch_id=extension_batch_id,
    )
    selected = select_rq4_capacity_extension_winners(
        initial_runs,
        extension_runs,
        initial_ledger=initial_ledger,
        extension_ledger=extension_ledger,
    )
    runs = [*initial_runs, *extension_runs]
    boundaries = assess_rq4_capacity_extension_boundaries(selected, runs)
    family_selections = {
        family_id: {
            "selected": dict(selected[family_id]),
            "boundary_decision": boundaries[family_id],
        }
        for family_id in RQ4_METADATA_FAMILIES
    }
    payload = {
        "schema_version": 1,
        "kind": "g3_rq4_metadata_capacity_extension_selection_native50m",
        "protocol_sha256": APPROVED_PROTOCOL_SHA256,
        "inputs": {
            "initial_ledger": _authenticated_file_fact(root, initial_ledger_path)
            | {"logical_sha256": initial_ledger.sha256},
            "extension_ledger": _authenticated_file_fact(root, extension_ledger_path)
            | {"logical_sha256": extension_ledger.sha256},
            "initial_batch": _authenticated_file_fact(
                root,
                root
                / "generated/training-queue-service/batches"
                / f"{initial_batch_id}.json",
            )
            | {"batch_id": initial_batch_id},
            "extension_batch": _authenticated_file_fact(
                root,
                root
                / "generated/training-queue-service/batches"
                / f"{extension_batch_id}.json",
            )
            | {"batch_id": extension_batch_id},
        },
        "opportunity_accounting": {
            "families": 3,
            "initial_per_family": 9,
            "extension_per_family": 3,
            "combined_per_family": 12,
            "combined_total": 36,
        },
        "selection_rule": (
            "validation Recall@100, validation NDCG@100, lower queue wall time, "
            "then row ID"
        ),
        "family_selections": family_selections,
        "capacity_renewed_approval_required": sorted(
            family_id
            for family_id, decision in boundaries.items()
            if decision["renewed_approval_required"]
        ),
        "learning_rate_extensions_required": sorted(
            family_id
            for family_id, decision in boundaries.items()
            if decision["learning_rate_extension_required"]
        ),
        "authenticated_runs": runs,
    }
    return payload | {"sha256": _canonical_sha256(payload)}


def persist_rq4_capacity_extension_selection_document(
    path: Path, document: Mapping[str, object], *, root: Path
) -> Path:
    root = root.resolve(strict=True)
    canonical = root / RQ4_CAPACITY_EXTENSION_SELECTION_PATH
    if path.is_symlink() or path.resolve(strict=False) != canonical.resolve(strict=False):
        raise ValueError("RQ4 extension selection must use its canonical project path")
    payload = {name: value for name, value in document.items() if name != "sha256"}
    if document.get("sha256") != _canonical_sha256(payload):
        raise ValueError("RQ4 extension selection logical hash is invalid")
    content = (
        json.dumps(
            dict(document),
            sort_keys=True,
            separators=(",", ":"),
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
            raise RuntimeError(f"immutable RQ4 extension selection differs: {path}")
    return path


def build_rq4_artist_album_lr_selection_document(
    *,
    root: Path,
    ledger: Rq4ArtistAlbumLrBoundaryLedger,
    ledger_path: Path,
    batch_id: str,
) -> dict[str, object]:
    root = root.resolve(strict=True)
    capacity_path = root / RQ4_CAPACITY_EXTENSION_SELECTION_PATH
    capacity = _load_json(capacity_path)
    capacity_payload = {
        name: value for name, value in capacity.items() if name != "sha256"
    }
    if (
        capacity.get("sha256") != RQ4_CAPACITY_EXTENSION_SELECTION_SHA256
        or _canonical_sha256(capacity_payload)
        != RQ4_CAPACITY_EXTENSION_SELECTION_SHA256
    ):
        raise ValueError("RQ4 capacity-extension selection changed")
    families = capacity.get("family_selections")
    family = families.get("rq4_artist_album") if isinstance(families, dict) else None
    source = family.get("selected") if isinstance(family, dict) else None
    authenticated_runs = capacity.get("authenticated_runs")
    if not isinstance(source, dict) or not isinstance(authenticated_runs, list):
        raise ValueError("RQ4 artist+album source selection is absent")
    boundary_runs = collect_authenticated_rq4_stage_runs(
        root,
        ledger=ledger,
        ledger_path=ledger_path,
        batch_id=batch_id,
    )
    selected = _select([source, *boundary_runs])
    family_runs = [
        run
        for run in authenticated_runs
        if isinstance(run, dict) and run.get("family_id") == "rq4_artist_album"
    ]
    if len(family_runs) != 12:
        raise ValueError("RQ4 artist+album capacity evidence is incomplete")
    all_runs = [*family_runs, *boundary_runs]
    embedding = _tested_rate_boundary(
        float(selected["embedding_learning_rate"]),
        [float(run["embedding_learning_rate"]) for run in all_runs],
    )
    deep = _tested_rate_boundary(
        float(selected["deep_learning_rate"]),
        [float(run["deep_learning_rate"]) for run in all_runs],
    )
    payload = {
        "schema_version": 1,
        "kind": "g3_rq4_artist_album_lr_boundary_selection_native50m",
        "protocol_sha256": APPROVED_PROTOCOL_SHA256,
        "inputs": {
            "capacity_extension_selection": _authenticated_file_fact(
                root, capacity_path
            )
            | {"logical_sha256": RQ4_CAPACITY_EXTENSION_SELECTION_SHA256},
            "lr_boundary_ledger": _authenticated_file_fact(root, ledger_path)
            | {"logical_sha256": ledger.sha256},
            "lr_boundary_batch": _authenticated_file_fact(
                root,
                root
                / "generated/training-queue-service/batches"
                / f"{batch_id}.json",
            )
            | {"batch_id": batch_id},
        },
        "source_row_id": source["row_id"],
        "selected": dict(selected),
        "boundary_decision": {
            "capacity": {
                "selected": selected["metadata_dim"],
                "tested_values": [16, 32, 64, 128],
                "direction": None,
            },
            "embedding_learning_rate": embedding,
            "deep_learning_rate": deep,
            "second_boundary_approval_required": bool(
                embedding["direction"] or deep["direction"]
            ),
        },
        "authenticated_boundary_runs": boundary_runs,
    }
    return payload | {"sha256": _canonical_sha256(payload)}


def persist_rq4_artist_album_lr_selection_document(
    path: Path, document: Mapping[str, object], *, root: Path
) -> Path:
    root = root.resolve(strict=True)
    canonical = root / RQ4_ARTIST_ALBUM_LR_SELECTION_PATH
    if path.is_symlink() or path.resolve(strict=False) != canonical.resolve(strict=False):
        raise ValueError("RQ4 artist+album LR selection must use its canonical path")
    payload = {name: value for name, value in document.items() if name != "sha256"}
    if document.get("sha256") != _canonical_sha256(payload):
        raise ValueError("RQ4 artist+album LR selection hash is invalid")
    content = (
        json.dumps(dict(document), sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(content)
    except FileExistsError:
        if path.read_bytes() != content:
            raise RuntimeError(f"immutable RQ4 LR selection differs: {path}")
    return path


def build_rq4_single_metadata_width256_selection_document(
    *,
    root: Path,
    ledger: Rq4SingleMetadataWidth256Ledger,
    ledger_path: Path,
    batch_id: str,
) -> dict[str, object]:
    root = root.resolve(strict=True)
    capacity_path = root / RQ4_CAPACITY_EXTENSION_SELECTION_PATH
    capacity = _load_json(capacity_path)
    capacity_payload = {
        name: value for name, value in capacity.items() if name != "sha256"
    }
    if (
        capacity.get("sha256") != RQ4_CAPACITY_EXTENSION_SELECTION_SHA256
        or _canonical_sha256(capacity_payload)
        != RQ4_CAPACITY_EXTENSION_SELECTION_SHA256
    ):
        raise ValueError("RQ4 capacity-extension selection changed")
    source_runs = capacity.get("authenticated_runs")
    if not isinstance(source_runs, list):
        raise ValueError("RQ4 width-256 source runs are absent")
    single_metadata_source_runs = [
        run
        for run in source_runs
        if isinstance(run, dict)
        and run.get("family_id") in {"rq4_artist", "rq4_album"}
    ]
    if (
        len(single_metadata_source_runs) != 24
        or {run["family_id"] for run in single_metadata_source_runs}
        != {"rq4_artist", "rq4_album"}
    ):
        raise ValueError("RQ4 width-256 source surface is incomplete")
    boundary_runs = collect_authenticated_rq4_stage_runs(
        root,
        ledger=ledger,
        ledger_path=ledger_path,
        batch_id=batch_id,
    )
    family_selections, renewed_capacity_approval, renewed_lr_approval = (
        select_rq4_single_metadata_width256_boundaries(
            single_metadata_source_runs, boundary_runs
        )
    )
    payload = {
        "schema_version": 1,
        "kind": "g3_rq4_single_metadata_width256_boundary_selection_native50m",
        "protocol_sha256": APPROVED_PROTOCOL_SHA256,
        "inputs": {
            "capacity_extension_selection": _authenticated_file_fact(
                root, capacity_path
            )
            | {"logical_sha256": RQ4_CAPACITY_EXTENSION_SELECTION_SHA256},
            "width256_ledger": _authenticated_file_fact(root, ledger_path)
            | {"logical_sha256": ledger.sha256},
            "width256_batch": _authenticated_file_fact(
                root,
                root
                / "generated/training-queue-service/batches"
                / f"{batch_id}.json",
            )
            | {"batch_id": batch_id},
        },
        "opportunity_accounting": {
            "families": ["rq4_artist", "rq4_album"],
            "source_per_family": 12,
            "width256_per_family": 3,
            "combined_per_family": 15,
            "combined_total": 30,
        },
        "selection_rule": (
            "validation Recall@100, validation NDCG@100, lower queue wall time, "
            "then row ID"
        ),
        "family_selections": family_selections,
        "renewed_capacity_approval_required": renewed_capacity_approval,
        "renewed_learning_rate_approval_required": renewed_lr_approval,
        "further_capacity_width_requires_renewed_approval": True,
        "authenticated_boundary_runs": boundary_runs,
    }
    return payload | {"sha256": _canonical_sha256(payload)}


def select_rq4_single_metadata_width256_boundaries(
    source_runs: Sequence[Mapping[str, object]],
    boundary_runs: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], list[str], list[str]]:
    all_runs = [*source_runs, *boundary_runs]
    family_selections = {}
    renewed_capacity_approval = []
    renewed_lr_approval = []
    for family_id in ("rq4_artist", "rq4_album"):
        family_runs = [run for run in all_runs if run["family_id"] == family_id]
        if len(family_runs) != 15:
            raise ValueError(f"RQ4 {family_id} width-256 surface is incomplete")
        selected = _select(family_runs)
        capacities = sorted({int(run["metadata_dim"]) for run in family_runs})
        if capacities != [16, 32, 64, 128, 256]:
            raise ValueError(f"RQ4 {family_id} capacity range changed")
        capacity_direction = (
            "lower" if selected["metadata_dim"] == capacities[0]
            else "upper" if selected["metadata_dim"] == capacities[-1]
            else None
        )
        embedding = _tested_rate_boundary(
            float(selected["embedding_learning_rate"]),
            [float(run["embedding_learning_rate"]) for run in family_runs],
        )
        deep = _tested_rate_boundary(
            float(selected["deep_learning_rate"]),
            [float(run["deep_learning_rate"]) for run in family_runs],
        )
        if capacity_direction is not None:
            renewed_capacity_approval.append(family_id)
        if embedding["direction"] or deep["direction"]:
            renewed_lr_approval.append(family_id)
        family_selections[family_id] = {
            "selected": dict(selected),
            "boundary_decision": {
                "capacity": {
                    "selected": selected["metadata_dim"],
                    "tested_values": capacities,
                    "direction": capacity_direction,
                    "renewed_approval_required": capacity_direction is not None,
                },
                "embedding_learning_rate": embedding,
                "deep_learning_rate": deep,
                "learning_rate_extension_required": bool(
                    embedding["direction"] or deep["direction"]
                ),
            },
        }
    return family_selections, renewed_capacity_approval, renewed_lr_approval


def persist_rq4_single_metadata_width256_selection_document(
    path: Path, document: Mapping[str, object], *, root: Path
) -> Path:
    root = root.resolve(strict=True)
    canonical = root / RQ4_SINGLE_METADATA_WIDTH256_SELECTION_PATH
    if path.is_symlink() or path.resolve(strict=False) != canonical.resolve(strict=False):
        raise ValueError("RQ4 width-256 selection must use its canonical path")
    payload = {name: value for name, value in document.items() if name != "sha256"}
    if document.get("sha256") != _canonical_sha256(payload):
        raise ValueError("RQ4 width-256 selection hash is invalid")
    content = (
        json.dumps(dict(document), sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(content)
    except FileExistsError:
        if path.read_bytes() != content:
            raise RuntimeError(f"immutable RQ4 width-256 selection differs: {path}")
    return path


def build_rq4_capacity_selection_document(
    *,
    root: Path,
    ledger: Rq4InitialLedger,
    ledger_path: Path,
    batch_id: str,
) -> dict[str, object]:
    runs = collect_authenticated_rq4_stage_runs(
        root,
        ledger=ledger,
        ledger_path=ledger_path,
        batch_id=batch_id,
    )
    selected = select_rq4_capacity_winners(runs, ledger=ledger)
    initial_ledger = _authenticated_file_fact(root.resolve(strict=True), ledger_path) | {
        "logical_sha256": ledger.sha256
    }
    family_selections = {}
    for family_id, winner in selected.items():
        capacity = int(winner["metadata_dim"])
        direction = "lower" if capacity == 16 else "upper" if capacity == 64 else None
        family_selections[family_id] = {
            "selected": dict(winner),
            "selected_row_id": winner["row_id"],
            "selected_metadata_dim": capacity,
            "selected_embedding_learning_rate": winner["embedding_learning_rate"],
            "selected_deep_learning_rate": winner["deep_learning_rate"],
            "selected_horizon_epochs": winner["horizon_epochs"],
            "capacity_boundary": {
                "direction": direction,
                "extension_capacity": (
                    capacity // 2 if direction == "lower"
                    else capacity * 2 if direction == "upper"
                    else None
                ),
            },
        }
    payload = {
        "schema_version": 1,
        "kind": "g3_rq4_metadata_capacity_selection_native50m",
        "protocol_sha256": APPROVED_PROTOCOL_SHA256,
        "initial_ledger": dict(initial_ledger),
        "opportunity_accounting": {
            "families": 3,
            "opportunities_per_family": 9,
            "total_opportunities": 27,
        },
        "selection_rule": (
            "validation Recall@100, validation NDCG@100, lower queue wall time, "
            "then row ID"
        ),
        "ranking_provenance": _ranking_provenance_resolution(root, runs),
        "family_selections": family_selections,
        "capacity_extensions_required": sorted(
            family_id
            for family_id, selection in family_selections.items()
            if selection["capacity_boundary"]["direction"] is not None
        ),
    }
    return payload | {"sha256": _canonical_sha256(payload)}


def persist_rq4_capacity_selection_document(
    path: Path, document: Mapping[str, object]
) -> Path:
    payload = {name: value for name, value in document.items() if name != "sha256"}
    if document.get("sha256") != _canonical_sha256(payload):
        raise ValueError("RQ4 capacity selection logical hash is invalid")
    content = (
        json.dumps(
            dict(document),
            sort_keys=True,
            separators=(",", ":"),
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
            raise RuntimeError(f"immutable RQ4 capacity selection differs: {path}")
    return path


def select_rq4_family_winners(
    runs: Sequence[Mapping[str, object]],
    *,
    initial_ledger: Rq4InitialLedger,
    horizon_ledger: Rq4HorizonLedger,
) -> dict[str, Mapping[str, object]]:
    _validate_run_set_against_ledger(
        runs,
        ledger=(initial_ledger, horizon_ledger),
    )
    grouped = _group_runs(runs, opportunities=12)
    selected = {}
    for family_id, family in grouped.items():
        capacities = {int(run["metadata_dim"]) for run in family}
        if not set(_CAPACITIES).issubset(capacities):
            raise ValueError(f"RQ4 family {family_id} omits an approved metadata width")
        selected[family_id] = _select(family)
        _validate_slice_identity(family)
    return selected


def assess_rq4_family_boundaries(
    selected: Mapping[str, Mapping[str, object]],
    runs: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    grouped = _group_runs(runs, opportunities=12)
    if set(selected) != set(RQ4_METADATA_FAMILIES):
        raise ValueError("RQ4 boundary assessment requires every metadata family")
    decisions = {}
    for family_id in RQ4_METADATA_FAMILIES:
        winner = selected[family_id]
        if winner not in grouped[family_id]:
            raise ValueError(f"RQ4 selected row is absent from {family_id}")
        capacity = int(winner["metadata_dim"])
        if capacity not in _CAPACITIES:
            raise ValueError("RQ4 selected metadata width is outside the approved surface")
        capacity_direction = (
            "lower" if capacity == _CAPACITIES[0]
            else "upper" if capacity == _CAPACITIES[-1]
            else None
        )
        horizon = int(winner["horizon_epochs"])
        best_epoch = int(winner["best_epoch"])
        if not 1 <= best_epoch <= horizon:
            raise ValueError(f"RQ4 family {family_id} has an invalid restored epoch")
        embedding = _tested_rate_boundary(
            float(winner["embedding_learning_rate"]),
            [float(run["embedding_learning_rate"]) for run in grouped[family_id]],
        )
        deep = _tested_rate_boundary(
            float(winner["deep_learning_rate"]),
            [float(run["deep_learning_rate"]) for run in grouped[family_id]],
        )
        horizon_extension = 60 if horizon == 40 and best_epoch == 40 else None
        decisions[family_id] = {
            "embedding_learning_rate": embedding,
            "deep_learning_rate": deep,
            "capacity": {
                "selected": capacity,
                "tested_values": list(_CAPACITIES),
                "direction": capacity_direction,
                "extension_capacity": (
                    capacity // 2 if capacity_direction == "lower"
                    else capacity * 2 if capacity_direction == "upper"
                    else None
                ),
            },
            "horizon": {
                "selected_epochs": horizon,
                "restored_best_epoch": best_epoch,
                "extend_to_epochs": horizon_extension,
            },
            "extension_required": bool(
                capacity_direction
                or embedding["direction"]
                or deep["direction"]
                or horizon_extension
            ),
        }
    return decisions


def build_rq4_metadata_winner_document(
    *,
    root: Path,
    initial_ledger: Rq4InitialLedger,
    initial_ledger_path: Path,
    initial_batch_id: str,
    horizon_ledger: Rq4HorizonLedger,
    horizon_ledger_path: Path,
    horizon_batch_id: str,
) -> dict[str, object]:
    runs = [
        *collect_authenticated_rq4_stage_runs(
            root,
            ledger=initial_ledger,
            ledger_path=initial_ledger_path,
            batch_id=initial_batch_id,
        ),
        *collect_authenticated_rq4_stage_runs(
            root,
            ledger=horizon_ledger,
            ledger_path=horizon_ledger_path,
            batch_id=horizon_batch_id,
        ),
    ]
    winners = select_rq4_family_winners(
        runs,
        initial_ledger=initial_ledger,
        horizon_ledger=horizon_ledger,
    )
    boundaries = assess_rq4_family_boundaries(winners, runs)
    unresolved = sorted(
        family_id
        for family_id, decision in boundaries.items()
        if decision["extension_required"]
    )
    if unresolved:
        raise ValueError("RQ4 metadata winner has unresolved boundaries")
    selected = _select(tuple(winners.values()))
    payload = {
        "schema_version": 1,
        "kind": "g3_rq4_metadata_winner_for_extra_id",
        "protocol_sha256": APPROVED_PROTOCOL_SHA256,
        "selection_resolved": True,
        "rq2_selection_sha256": initial_ledger.rq2_final_evidence.logical_sha256,
        "rq3_selection_sha256": initial_ledger.rq3_final_evidence.logical_sha256,
        "initial_ledger_sha256": initial_ledger.sha256,
        "horizon_ledger_sha256": horizon_ledger.sha256,
        "selected_family_id": selected["family_id"],
        "selected_metadata_dim": selected["metadata_dim"],
        "selected": dict(selected),
        "family_boundaries": boundaries,
    }
    return payload | {"sha256": _canonical_sha256(payload)}


def resolve_rq4_metadata_selection(
    *,
    root: Path,
    predecessor: Mapping[str, object],
    metadata_runs: Sequence[Mapping[str, object]],
    extra_id_runs: Sequence[Mapping[str, object]],
    initial_ledger: Rq4InitialLedger,
    horizon_ledger: Rq4HorizonLedger,
    extra_id_ledger: Mapping[str, object],
    extra_id_surface: Rq4ExtraIdSurface,
    recall_relative_dispersion: float,
) -> dict[str, object]:
    if not math.isfinite(recall_relative_dispersion) or recall_relative_dispersion < 0:
        raise ValueError("RQ4 Recall@100 dispersion must be finite and nonnegative")
    root = root.resolve(strict=True)
    identity = _metadata_identity(root, initial_ledger)
    _validate_run_set_against_ledger(
        metadata_runs,
        ledger=(initial_ledger, horizon_ledger),
    )
    for run in metadata_runs:
        _reauthenticate_result_files(root, run, identity=identity)
    family_winners = select_rq4_family_winners(
        metadata_runs,
        initial_ledger=initial_ledger,
        horizon_ledger=horizon_ledger,
    )
    boundaries = assess_rq4_family_boundaries(family_winners, metadata_runs)
    if any(
        decision.get("extension_required") is not False
        for decision in boundaries.values()
    ):
        raise ValueError("RQ4 metadata selection has unresolved boundaries")
    candidates = tuple(family_winners.values())
    metadata = _select(candidates)
    _validate_predecessor_run(
        root,
        predecessor,
        ledger=initial_ledger,
        identity=identity,
    )
    _validate_extra_id_runs(
        root,
        extra_id_runs,
        ledger=extra_id_ledger,
        metadata_runs=metadata_runs,
        metadata_family=str(metadata["family_id"]),
        metadata_dim=int(metadata["metadata_dim"]),
        metadata_row_id=str(metadata["row_id"]),
        initial_ledger=initial_ledger,
        horizon_ledger=horizon_ledger,
        approved_surface=extra_id_surface,
        identity=identity,
    )
    extra_id_winner = _select(extra_id_runs)
    _validate_slice_identity((*candidates, predecessor, extra_id_winner))
    predecessor_recall = _metric(predecessor, "recall@100")
    metadata_recall = _metric(metadata, "recall@100")
    extra_recall = _metric(extra_id_winner, "recall@100")
    predecessor_band = abs(predecessor_recall) * recall_relative_dispersion
    extra_band = abs(extra_recall) * recall_relative_dispersion
    metadata_tail = _slice_metric(metadata, "tail", "recall@100")
    predecessor_tail = _slice_metric(predecessor, "tail", "recall@100")
    extra_tail = _slice_metric(extra_id_winner, "tail", "recall@100")
    aggregate_noninferior = metadata_recall >= predecessor_recall - predecessor_band
    tail_beats_predecessor = metadata_tail > predecessor_tail
    tail_beats_extra = metadata_tail > extra_tail
    extra_overall_relation = (
        "better"
        if metadata_recall > extra_recall + extra_band
        else "worse"
        if metadata_recall < extra_recall - extra_band
        else "within_band"
    )
    beats_extra_control = extra_overall_relation == "better" or (
        extra_overall_relation == "within_band" and tail_beats_extra
    )
    return {
        "selected_metadata_family": metadata["family_id"],
        "selected_metadata_row_id": metadata["row_id"],
        "selected_metadata_dim": metadata["metadata_dim"],
        "predecessor_row_id": predecessor["row_id"],
        "extra_item_id_row_id": extra_id_winner["row_id"],
        "recall@100_operational_bands": {
            "predecessor": predecessor_band,
            "extra_item_id": extra_band,
        },
        "recall@100_deltas": {
            "metadata_minus_predecessor": metadata_recall - predecessor_recall,
            "metadata_minus_extra_item_id": metadata_recall - extra_recall,
        },
        "tail_recall@100_deltas": {
            "metadata_minus_predecessor": metadata_tail - predecessor_tail,
            "metadata_minus_extra_item_id": metadata_tail - extra_tail,
        },
        "aggregate_noninferior": aggregate_noninferior,
        "tail_beats_predecessor": tail_beats_predecessor,
        "tail_beats_extra_item_id": tail_beats_extra,
        "extra_item_id_overall_relation": extra_overall_relation,
        "beats_parameter_matched_extra_item_id": beats_extra_control,
        "metadata_promoted": (
            aggregate_noninferior
            and tail_beats_predecessor
            and beats_extra_control
        ),
        "slice_evidence_status": "descriptive_only_no_slice_repeat_calibration",
    }


def _group_runs(
    runs: Sequence[Mapping[str, object]], *, opportunities: int
) -> dict[str, list[Mapping[str, object]]]:
    row_ids = [run.get("row_id") for run in runs]
    if len(row_ids) != len(set(row_ids)):
        raise ValueError("RQ4 tuning opportunities contain duplicate row IDs")
    if len(runs) != opportunities * len(RQ4_METADATA_FAMILIES):
        raise ValueError(
            f"RQ4 requires exactly {opportunities} opportunities for each family"
        )
    result = {}
    for family_id in RQ4_METADATA_FAMILIES:
        family = [run for run in runs if run.get("family_id") == family_id]
        if len(family) != opportunities:
            word = "nine" if opportunities == 9 else "twelve"
            raise ValueError(f"RQ4 family {family_id} must contain exactly {word} opportunities")
        result[family_id] = family
    if any(run.get("family_id") not in result for run in runs):
        raise ValueError("RQ4 tuning opportunities contain another family")
    return result


def _select(runs: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    if not runs:
        raise ValueError("RQ4 selection requires at least one run")
    return min(
        runs,
        key=lambda run: (
            -_metric(run, "recall@100"),
            -_metric(run, "ndcg@100"),
            float(run["queue_wall_seconds"]),
            str(run["row_id"]),
        ),
    )


def _tested_rate_boundary(
    selected: float, values: Sequence[float]
) -> dict[str, object]:
    tested = sorted(set(values))
    if (
        len(tested) < 3
        or any(not math.isfinite(value) or value <= 0 for value in tested)
        or selected not in tested
    ):
        raise ValueError("RQ4 learning-rate boundary lacks an actual tested range")
    lower, upper = tested[0], tested[-1]
    position = (selected - lower) / (upper - lower)
    return {
        "selected": selected,
        "tested_values": tested,
        "tested_interval": [lower, upper],
        "normalized_position": position,
        "has_tested_lower": any(value < selected for value in tested),
        "has_tested_higher": any(value > selected for value in tested),
        "direction": "lower" if selected == lower else "upper" if selected == upper else None,
    }


def _validate_slice_identity(
    runs: Sequence[Mapping[str, object]],
    *,
    identity: Rq4MetadataIdentity | None = None,
) -> None:
    expected = None
    expected_feature = (
        _feature_identity_document(identity) if identity is not None else None
    )
    for run in runs:
        feature = run.get("feature_identity")
        if expected_feature is None:
            if not isinstance(feature, dict) or set(feature) != {
                "manifest_sha256",
                "feature_data_sha256",
                "frequency_terciles",
                "training_count_reference",
                "slice_membership_reference",
            }:
                raise ValueError("RQ4 run lacks authenticated feature/count identity")
            expected_feature = feature
        elif feature != expected_feature:
            raise ValueError("RQ4 feature/count identity differs")
        slices = run.get("slices")
        if not isinstance(slices, dict) or set(slices) != {"head", "mid", "tail"}:
            raise ValueError("RQ4 run lacks complete item-frequency slices")
        slice_identity = tuple(
            (
                name,
                slices[name].get("num_users"),
                slices[name].get("num_targets"),
                slices[name].get("item_membership_sha256"),
            )
            for name in ("head", "mid", "tail")
        )
        if expected is None:
            expected = slice_identity
        elif slice_identity != expected:
            raise ValueError("RQ4 item-frequency slice identity differs")


def _stage_contract(
    ledger: Rq4InitialLedger
    | Rq4CapacityExtensionLedger
    | Rq4ArtistAlbumLrBoundaryLedger
    | Rq4SingleMetadataWidth256Ledger
    | Rq4HorizonLedger,
) -> dict[str, object]:
    if isinstance(ledger, Rq4InitialLedger):
        return {
            "job_environment": "G3_RQ4_INITIAL_JOB_B64",
            "ledger_environment": "G3_RQ4_INITIAL_LEDGER_PATH",
            "runner": "run_rq4_initial.py",
            "job_contract": "g3_rq4_initial_job.json",
        }
    if isinstance(ledger, Rq4HorizonLedger):
        return {
            "job_environment": "G3_RQ4_HORIZON_JOB_B64",
            "ledger_environment": "G3_RQ4_HORIZON_LEDGER_PATH",
            "runner": "run_rq4_horizon.py",
            "job_contract": "g3_rq4_horizon_job.json",
        }
    if isinstance(ledger, Rq4CapacityExtensionLedger):
        return {
            "job_environment": "G3_RQ4_CAPACITY_EXTENSION_JOB_B64",
            "ledger_environment": "G3_RQ4_CAPACITY_EXTENSION_LEDGER_PATH",
            "runner": "run_rq4_capacity_extension.py",
            "job_contract": "g3_rq4_capacity_extension_job.json",
        }
    if isinstance(ledger, Rq4ArtistAlbumLrBoundaryLedger):
        return {
            "job_environment": "G3_RQ4_ARTIST_ALBUM_LR_BOUNDARY_JOB_B64",
            "ledger_environment": "G3_RQ4_ARTIST_ALBUM_LR_BOUNDARY_LEDGER_PATH",
            "runner": "run_rq4_artist_album_lr_boundary.py",
            "job_contract": "g3_rq4_artist_album_lr_boundary_job.json",
        }
    if isinstance(ledger, Rq4SingleMetadataWidth256Ledger):
        return {
            "job_environment": "G3_RQ4_SINGLE_METADATA_WIDTH256_JOB_B64",
            "ledger_environment": "G3_RQ4_SINGLE_METADATA_WIDTH256_LEDGER_PATH",
            "runner": "run_rq4_single_metadata_width256.py",
            "job_contract": "g3_rq4_single_metadata_width256_job.json",
        }
    raise TypeError("RQ4 result collection received another ledger type")


def _metadata_identity(
    root: Path,
    ledger: Rq4InitialLedger
    | Rq4CapacityExtensionLedger
    | Rq4ArtistAlbumLrBoundaryLedger
    | Rq4SingleMetadataWidth256Ledger
    | Rq4HorizonLedger,
) -> Rq4MetadataIdentity:
    if isinstance(ledger, Rq4InitialLedger):
        surface = compile_rq4_capacity_surface(
            root=root,
            rq2_selection_path=root / ledger.rq2_final_evidence.path,
            expected_rq2_selection_sha256=ledger.rq2_final_evidence.logical_sha256,
            rq3_selection_path=root / ledger.rq3_final_evidence.path,
            expected_rq3_selection_sha256=ledger.rq3_final_evidence.logical_sha256,
            expected_rq3_row_id=ledger.expected_rq3_row_id,
        )
    elif isinstance(ledger, Rq4HorizonLedger):
        surface = reconstruct_rq4_horizon_surface(root=root, ledger=ledger)
    elif isinstance(ledger, Rq4CapacityExtensionLedger):
        initial = load_rq4_initial_ledger(
            root / ledger.initial_ledger.path,
            root=root,
            expected_ledger_sha256=ledger.initial_ledger.logical_sha256,
            expected_rq3_sha256=RQ3_FINAL_EVIDENCE_LOGICAL_SHA256,
            expected_rq3_row_id=RQ3_FINAL_SELECTED_ROW_ID,
        )
        return _metadata_identity(root, initial)
    else:
        expected = compile_rq4_capacity_extension_ledger(root)
        capacity = load_rq4_capacity_extension_ledger(
            root / RQ4_CAPACITY_EXTENSION_LEDGER_PATH,
            root=root,
            expected_ledger_sha256=expected.sha256,
        )
        return _metadata_identity(root, capacity)
    identity = surface.metadata_identity
    initial = ledger if isinstance(ledger, Rq4InitialLedger) else None
    if initial is not None and initial.feature_identity != _feature_identity_document(identity):
        raise ValueError("RQ4 ledger feature/count identity changed")
    return identity


def _feature_identity_document(identity: Rq4MetadataIdentity) -> dict[str, object]:
    return {
        "manifest_sha256": identity.manifest_sha256,
        "feature_data_sha256": identity.feature_data_sha256,
        "frequency_terciles": identity.frequency_terciles,
        "training_count_reference": identity.training_count_reference,
        "slice_membership_reference": identity.slice_membership_reference,
    }


def _validate_batch(
    document: Mapping[str, object], *, batch_id: str, expected_jobs: int
) -> list[str]:
    jobs = document.get("jobs")
    if (
        set(document) != {"id", "jobs", "sealed", "sealed_at", "submitted_at"}
        or document.get("id") != batch_id
        or document.get("sealed") is not True
        or not isinstance(jobs, list)
        or len(jobs) != expected_jobs
        or len(set(jobs)) != expected_jobs
        or any(not isinstance(job_id, str) or not job_id for job_id in jobs)
        or not _finite_number(document.get("submitted_at"))
        or not _finite_number(document.get("sealed_at"))
        or float(document["sealed_at"]) < float(document["submitted_at"])
    ):
        raise ValueError("RQ4 queue batch differs from the exact sealed stage batch")
    return list(jobs)


def _collect_stage_run(
    *,
    root: Path,
    ledger: Rq4InitialLedger
    | Rq4CapacityExtensionLedger
    | Rq4ArtistAlbumLrBoundaryLedger
    | Rq4SingleMetadataWidth256Ledger
    | Rq4HorizonLedger,
    ledger_path: Path,
    row: object,
    batch_id: str,
    job_id: str,
    stage: Mapping[str, object],
    context_path: Path,
    context_reference: Mapping[str, object],
    item_counts: Mapping[int, int],
    identity: Rq4MetadataIdentity,
) -> dict[str, object]:
    row_document = row.to_dict()
    row_id = str(row_document["id"])
    run_name = str(row_document["run_name"])
    completed_path = root / "generated/training-queue-service/completed" / f"{job_id}.json"
    queue = _load_json(completed_path)
    runner = (
        root
        / "experiments/g3_pretrained_item_embeddings/launchers"
        / str(stage["runner"])
    ).resolve(strict=True)
    if (
        set(queue) != _QUEUE_RECORD_KEYS
        or queue.get("id") != job_id
        or queue.get("batch_id") != batch_id
        or queue.get("data_group") != "g3-native50m-likes"
        or queue.get("run") != run_name
        or queue.get("exit_code") != 0
        or Path(str(queue.get("script"))).resolve() != runner
        or not _ordered_job_times(queue)
    ):
        raise ValueError(f"RQ4 queue completion differs for {row_id}")
    environment = queue.get("environment")
    pairs = [value.split("=", 1) for value in environment] if isinstance(environment, list) else []
    values = dict(pairs)
    job_environment = str(stage["job_environment"])
    ledger_environment = str(stage["ledger_environment"])
    expected_environment = {
        "WANDB_MODE": "offline",
        job_environment: values.get(job_environment),
        ledger_environment: str(ledger_path),
    } | {
        name: value
        for name, value in (
            assignment.split("=", 1)
            for assignment in G3_CPU_THREAD_ENVIRONMENT
        )
    }
    if (
        len(pairs) != len(values) == len(expected_environment)
        or values != expected_environment
        or Path(values[ledger_environment]).resolve() != ledger_path
    ):
        raise ValueError(f"RQ4 queue environment differs for {row_id}")
    compiled = decode_control_job(values[job_environment], ledger)
    if compiled.row_id != row_id or compiled.job != row_document:
        raise ValueError(f"RQ4 queue job differs from ledger row {row_id}")
    directory = root / "generated/logs" / run_name
    artifacts = {
        contract.name: directory
        / (
            str(stage["job_contract"])
            if contract.name == "job_contract"
            else contract.filename
        )
        for contract in RQ4_INITIAL_ARTIFACT_CONTRACTS
    }
    contract = _load_json(artifacts["job_contract"])
    if contract != compiled.to_dict() | {
        "ledger_path": str(ledger_path),
        "ledger_sha256": ledger.sha256,
    }:
        raise ValueError(f"RQ4 job contract differs for {row_id}")
    metadata = _load_json(artifacts["training_metadata"])
    _validate_training_metadata(metadata, row_document=row_document)
    reported_metrics = _load_json(artifacts["final_metrics"])
    recomputed = _recompute_metrics(
        context_path,
        artifacts["ranking_evidence"],
        artifacts["top_item_rankings"],
    )
    metrics, coverage_deltas = _authoritative_ranking_metrics(
        reported_metrics, recomputed, row_label=row_id
    )
    diagnostics = _load_json(artifacts["training_diagnostics"])
    representation = row_document["representation"]
    training = row_document["training"]
    _validate_training_diagnostics(
        diagnostics,
        feature_identity=_FeatureIdentity(
            manifest_path=identity.manifest_path,
            manifest_sha256=identity.manifest_sha256,
            manifest_file_sha256=identity.manifest_file_sha256,
            data_path=identity.feature_data_path,
            data_sha256=identity.feature_data_sha256,
            frequency_terciles=identity.frequency_terciles,
            training_count_reference=identity.training_count_reference,
            slice_membership_reference=identity.slice_membership_reference,
        ),
        horizon_epochs=int(training["horizon_epochs"]),
        catalog_representation=str(representation["catalog"]),
        diagnostics_contract=_RQ4_METADATA_DIAGNOSTICS,
    )
    verify_unique_completed_run(
        root / "generated/training-queue-service",
        run_name=run_name,
        expected_job_id=job_id,
    )
    verify_artifacts_in_job_window(
        tuple(artifacts.values()),
        dispatched_at=float(queue["dispatched_at"]),
        finished_at=float(queue["finished_at"]),
        run_label=row_id,
    )
    slices = _ranking_slices(
        context_path=context_path,
        ranking_path=artifacts["ranking_evidence"],
        rankings_path=artifacts["top_item_rankings"],
        item_counts=item_counts,
        rank_source="ranking_evidence",
    )
    efficiency = _efficiency(
        metadata=metadata,
        log_path=artifacts["sweep_log"],
        queue_wall_seconds=float(queue["finished_at"])
        - float(queue["dispatched_at"]),
    )
    return {
        "row_id": row_id,
        "family_id": row_document["family_id"],
        "run_name": run_name,
        "ledger_sha256": ledger.sha256,
        "job": row_document,
        "metadata_dim": representation["metadata_dim"],
        "embedding_learning_rate": training["embedding_learning_rate"],
        "deep_learning_rate": training["deep_learning_rate"],
        "horizon_epochs": training["horizon_epochs"],
        "best_epoch": metadata["best_epoch"],
        "epochs_trained": metadata["epochs_trained"],
        "queue_wall_seconds": efficiency["queue_wall_seconds"],
        "metrics": metrics,
        "metric_provenance": {
            "recomputed_from_ranking_evidence": True,
            "absolute_tolerance": 1e-15,
            "num_users": int(recomputed["num_users"]),
            "ranking_context": dict(context_reference),
            "authoritative_noncoverage_source": "ranking_evidence",
            "authoritative_coverage_source": "top_item_rankings",
            "reported_minus_snapshot_coverage": coverage_deltas,
        },
        "feature_identity": _feature_identity_document(identity),
        "slices": slices,
        "efficiency": efficiency,
        "queue_job": _authenticated_file_fact(root, completed_path)
        | {"job_id": job_id},
        "artifacts": {
            name: _authenticated_file_fact(root, path)
            for name, path in artifacts.items()
        },
    }


def _validate_training_metadata(
    metadata: Mapping[str, object], *, row_document: Mapping[str, object]
) -> None:
    representation = row_document.get("representation")
    training = row_document.get("training")
    if not isinstance(representation, dict) or not isinstance(training, dict):
        raise ValueError("RQ4 ledger row lacks representation or training")
    horizon = training["horizon_epochs"]
    expected_representation = {
        "catalog_representation": representation["catalog"],
        "content_gate": "fixed",
        "extra_item_id_dim": None,
        "gate_hidden_dim": None,
        "history_hidden_dim": representation["history_hidden_dim"],
        "history_representation": "id_content",
        "metadata": representation["metadata"],
        "metadata_dim": representation["metadata_dim"],
    }
    expected = {
        "batch_size": 512,
        "seed": 42,
        "embedding_learning_rate": training["embedding_learning_rate"],
        "deep_learning_rate": training["deep_learning_rate"],
        "lr_schedule_horizon_epochs": horizon,
        "num_epochs": horizon,
        "max_epochs": horizon,
        "epochs_trained": horizon,
        "stopped_epoch": horizon,
        "lr_horizon_complete": True,
        "selection_resolved": True,
        "early_stopped": False,
        "g3_dataset_size": "native-50m",
        "g3_protocol_sha256": APPROVED_PROTOCOL_SHA256,
        "g3_representation": expected_representation,
    }
    if any(metadata.get(name) != value for name, value in expected.items()):
        raise ValueError(f"RQ4 runtime metadata differs for {row_document['id']}")
    best_epoch = metadata.get("best_epoch")
    if type(best_epoch) is not int or not 1 <= best_epoch <= int(horizon):
        raise ValueError(f"RQ4 restored epoch is invalid for {row_document['id']}")
    if metadata.get("best_epoch_at_cap") is not (best_epoch == horizon):
        raise ValueError(f"RQ4 restored-epoch cap flag differs for {row_document['id']}")
    traces = metadata.get("lr_group_traces")
    if not isinstance(traces, dict) or any(
        not isinstance(traces.get(group), list)
        or len(traces[group]) != horizon
        or traces[group][-1] != 0.0
        for group in ("embedding", "deep")
    ):
        raise ValueError(f"RQ4 schedule trace is incomplete for {row_document['id']}")


def _validate_run_set_against_ledger(
    runs: Sequence[Mapping[str, object]],
    *,
    ledger: Rq4InitialLedger
    | Rq4CapacityExtensionLedger
    | Rq4ArtistAlbumLrBoundaryLedger
    | Rq4SingleMetadataWidth256Ledger
    | Rq4HorizonLedger
    | tuple[
        Rq4InitialLedger
        | Rq4CapacityExtensionLedger
        | Rq4ArtistAlbumLrBoundaryLedger
        | Rq4SingleMetadataWidth256Ledger
        | Rq4HorizonLedger,
        ...,
    ],
) -> None:
    ledgers = ledger if isinstance(ledger, tuple) else (ledger,)
    expected = {
        row.id: (value.sha256, row.to_dict())
        for value in ledgers
        for row in value.rows
    }
    if len(runs) != len(expected):
        raise ValueError("RQ4 results do not cover the exact authenticated ledger rows")
    seen = set()
    for run in runs:
        row_id = run.get("row_id")
        binding = expected.get(row_id) if isinstance(row_id, str) else None
        metric_provenance = run.get("metric_provenance")
        if (
            binding is None
            or row_id in seen
            or run.get("ledger_sha256") != binding[0]
            or run.get("job") != binding[1]
            or not _run_matches_job(run, binding[1])
            or not isinstance(metric_provenance, dict)
            or metric_provenance.get("recomputed_from_ranking_evidence") is not True
            or not isinstance(run.get("queue_job"), dict)
            or not isinstance(run.get("artifacts"), dict)
            or set(run["artifacts"])
            != {contract.name for contract in RQ4_INITIAL_ARTIFACT_CONTRACTS}
        ):
            raise ValueError("RQ4 result differs from its authenticated ledger row")
        seen.add(row_id)
    if seen != set(expected):
        raise ValueError("RQ4 results omit authenticated ledger rows")


def _run_matches_job(
    run: Mapping[str, object], job: Mapping[str, object]
) -> bool:
    representation = job.get("representation")
    training = job.get("training")
    return (
        isinstance(representation, dict)
        and isinstance(training, dict)
        and run.get("family_id") == job.get("family_id")
        and run.get("run_name") == job.get("run_name")
        and run.get("metadata_dim") == representation.get("metadata_dim")
        and run.get("embedding_learning_rate")
        == training.get("embedding_learning_rate")
        and run.get("deep_learning_rate") == training.get("deep_learning_rate")
        and run.get("horizon_epochs") == training.get("horizon_epochs")
    )


def _validate_predecessor_run(
    root: Path,
    run: Mapping[str, object],
    *,
    ledger: Rq4InitialLedger,
    identity: Rq4MetadataIdentity,
) -> None:
    reference = ledger.rq3_final_evidence
    path = root / reference.path
    if (
        path.is_symlink()
        or _authenticated_file_fact(root, path)
        != {
            "path": reference.path,
            "size_bytes": reference.size_bytes,
            "sha256": reference.sha256,
        }
    ):
        raise ValueError("RQ4 predecessor selection file changed")
    document = _load_json(path)
    payload = {name: value for name, value in document.items() if name != "sha256"}
    downstream = document.get("downstream_selection")
    selected = (
        downstream.get("rq4_scientific_selected")
        if document.get("kind") == "g3_rq3_final_native50m_evidence"
        and isinstance(downstream, dict)
        else document.get("selected")
    )
    artifacts = selected.get("artifacts") if isinstance(selected, dict) else None
    queue = selected.get("queue_job") if isinstance(selected, dict) else None
    provenance = run.get("metric_provenance")
    job = run.get("job")
    representation = job.get("representation") if isinstance(job, dict) else None
    training = job.get("training") if isinstance(job, dict) else None
    selected_history_hidden_dim = (
        selected.get("history_hidden_dim")
        if isinstance(selected, dict) and "history_hidden_dim" in selected
        else (
            representation.get("history_hidden_dim")
            if isinstance(representation, dict)
            else None
        )
    )
    if (
        document.get("sha256") != reference.logical_sha256
        or _canonical_sha256(payload) != reference.logical_sha256
        or not isinstance(selected, dict)
        or selected.get("row_id") != ledger.expected_rq3_row_id
        or run.get("row_id") != selected.get("row_id")
        or run.get("family_id") != selected.get("family_id")
        or run.get("run_name") != selected.get("run_name")
        or run.get("artifacts") != artifacts
        or not isinstance(run.get("queue_job"), dict)
        or not _predecessor_queue_reference_matches(run["queue_job"], queue)
        or not isinstance(provenance, dict)
        or provenance.get("ranking_context") != document.get("ranking_context")
        or not isinstance(representation, dict)
        or representation.get("history_hidden_dim")
        != selected_history_hidden_dim
        or representation.get("catalog_representation")
        != selected.get("catalog_representation")
        or not isinstance(training, dict)
        or training.get("embedding_learning_rate")
        != selected.get("embedding_learning_rate")
        or training.get("deep_learning_rate") != selected.get("deep_learning_rate")
        or training.get("horizon_epochs") != selected.get("horizon_epochs")
    ):
        raise ValueError("RQ4 predecessor differs from the exact RQ3 selection")
    _reauthenticate_result_files(root, run, identity=identity)


def _predecessor_queue_reference_matches(
    run_reference: Mapping[str, object], selected_reference: object
) -> bool:
    if not isinstance(selected_reference, dict):
        return False
    selected = {
        name: selected_reference.get(name)
        for name in ("path", "size_bytes", "sha256")
    }
    run = {
        name: run_reference.get(name)
        for name in ("path", "size_bytes", "sha256")
    }
    selected_job_id = selected_reference.get("job_id")
    return run == selected and (
        selected_job_id is None or run_reference.get("job_id") == selected_job_id
    )


def _validate_extra_id_runs(
    root: Path,
    runs: Sequence[Mapping[str, object]],
    *,
    ledger: Mapping[str, object],
    metadata_runs: Sequence[Mapping[str, object]],
    metadata_family: str,
    metadata_dim: int,
    metadata_row_id: str,
    initial_ledger: Rq4InitialLedger,
    horizon_ledger: Rq4HorizonLedger,
    approved_surface: Rq4ExtraIdSurface,
    identity: Rq4MetadataIdentity,
) -> None:
    payload = {name: value for name, value in ledger.items() if name != "sha256"}
    rows = ledger.get("rows")
    metadata_rows = [
        value.to_dict()
        for source in (initial_ledger, horizon_ledger)
        for value in source.rows
        if value.family_id == metadata_family
    ]
    expected_coordinates = [
        (
            value["representation"]["metadata_dim"],
            value["training"]["embedding_learning_rate"],
            value["training"]["deep_learning_rate"],
            value["training"]["horizon_epochs"],
        )
        for value in metadata_rows
    ]
    extra_coordinates = (
        [_extra_id_coordinate(value) for value in rows]
        if isinstance(rows, list)
        else []
    )
    approved_rows = _recompiled_extra_id_documents(
        root,
        approved_surface,
        initial_ledger=initial_ledger,
        horizon_ledger=horizon_ledger,
        metadata_row_id=metadata_row_id,
    )
    if (
        set(ledger) != {
            "schema_version",
            "kind",
            "protocol_sha256",
            "selected_metadata_family",
            "selected_metadata_dim",
            "maximum_parameter_mismatch_fraction",
            "artifact_contracts",
            "rows",
            "sha256",
        }
        or ledger.get("schema_version") != 1
        or ledger.get("kind") != "g3_rq4_parameter_matched_extra_item_id"
        or ledger.get("protocol_sha256") != APPROVED_PROTOCOL_SHA256
        or ledger.get("sha256") != _canonical_sha256(payload)
        or not isinstance(rows, list)
        or len(rows) != 12
        or rows != approved_rows
        or len({value.get("id") for value in rows if isinstance(value, dict)}) != 12
        or ledger.get("selected_metadata_family") != metadata_family
        or ledger.get("selected_metadata_dim") != metadata_dim
        or ledger.get("maximum_parameter_mismatch_fraction") != 0.01
        or ledger.get("artifact_contracts")
        != [contract.to_dict() for contract in RQ4_INITIAL_ARTIFACT_CONTRACTS]
        or len(metadata_rows) != 12
        or extra_coordinates != expected_coordinates
        or any(
            not _valid_extra_id_row(
                value,
                family_id=metadata_family,
                opportunity_index=index,
            )
            for index, value in enumerate(rows, start=1)
        )
    ):
        raise ValueError("RQ4 extra-ID ledger is not the exact matched surface")
    expected = {
        str(row["id"]): row
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("id"), str)
    }
    if len(runs) != 12 or len(expected) != 12:
        raise ValueError("RQ4 extra-ID results must cover all twelve ledger rows")
    by_id: dict[str, Mapping[str, object]] = {}
    for run in runs:
        row_id = run.get("row_id")
        row = expected.get(str(row_id)) if isinstance(row_id, str) else None
        provenance = run.get("metric_provenance")
        if (
            row is None
            or row_id in by_id
            or run.get("family_id") != "rq4_extra_item_id"
            or run.get("ledger_sha256") != ledger.get("sha256")
            or run.get("job") != row
            or not _extra_run_matches_job(run, row)
            or not isinstance(run.get("artifacts"), dict)
            or set(run["artifacts"])
            != {contract.name for contract in RQ4_INITIAL_ARTIFACT_CONTRACTS}
            or not isinstance(run.get("queue_job"), dict)
            or not isinstance(provenance, dict)
            or provenance.get("recomputed_from_ranking_evidence") is not True
        ):
            raise ValueError("RQ4 extra-ID result differs from its exact ledger row")
        _reauthenticate_result_files(root, run, identity=identity)
        by_id[row_id] = run
    if set(by_id) != set(expected):
        raise ValueError("RQ4 extra-ID results omit exact ledger rows")
    metadata_by_id = {
        str(run["row_id"]): run
        for run in metadata_runs
        if run.get("family_id") == metadata_family
    }
    for row, metadata_row in zip(rows, metadata_rows, strict=True):
        metadata_run = metadata_by_id.get(str(metadata_row["id"]))
        extra_run = by_id[str(row["id"])]
        representation = row["representation"]
        metadata_efficiency = (
            metadata_run.get("efficiency")
            if isinstance(metadata_run, Mapping)
            else None
        )
        extra_efficiency = extra_run.get("efficiency")
        metadata_parameters = (
            metadata_efficiency.get("parameter_count")
            if isinstance(metadata_efficiency, dict)
            else None
        )
        extra_parameters = (
            extra_efficiency.get("parameter_count")
            if isinstance(extra_efficiency, dict)
            else None
        )
        if (
            metadata_run is None
            or type(metadata_parameters) is not int
            or type(extra_parameters) is not int
            or metadata_parameters <= 0
            or abs(extra_parameters - metadata_parameters) / metadata_parameters >= 0.01
            or not 0 <= float(representation["parameter_mismatch_fraction"]) < 0.01
        ):
            raise ValueError("RQ4 extra-ID parameter mismatch is not below one percent")


def _extra_run_matches_job(
    run: Mapping[str, object], job: Mapping[str, object]
) -> bool:
    representation = job.get("representation")
    training = job.get("training")
    return (
        isinstance(representation, dict)
        and isinstance(training, dict)
        and run.get("run_name") == job.get("run_name")
        and run.get("metadata_dim") == representation.get("matched_metadata_dim")
        and run.get("embedding_learning_rate")
        == training.get("embedding_learning_rate")
        and run.get("deep_learning_rate") == training.get("deep_learning_rate")
        and run.get("horizon_epochs") == training.get("horizon_epochs")
    )


def _recompiled_extra_id_documents(
    root: Path,
    surface: Rq4ExtraIdSurface,
    *,
    initial_ledger: Rq4InitialLedger,
    horizon_ledger: Rq4HorizonLedger,
    metadata_row_id: str,
) -> list[dict[str, object]]:
    if not isinstance(surface, Rq4ExtraIdSurface):
        raise ValueError("RQ4 extra-ID surface has another type")
    capacity = compile_rq4_capacity_surface(
        root=root,
        rq2_selection_path=root / initial_ledger.rq2_final_evidence.path,
        expected_rq2_selection_sha256=(
            initial_ledger.rq2_final_evidence.logical_sha256
        ),
        rq3_selection_path=root / initial_ledger.rq3_final_evidence.path,
        expected_rq3_selection_sha256=(
            initial_ledger.rq3_final_evidence.logical_sha256
        ),
        expected_rq3_row_id=initial_ledger.expected_rq3_row_id,
    )
    horizon = reconstruct_rq4_horizon_surface(root=root, ledger=horizon_ledger)
    if surface.capacity_surface != capacity or surface.horizon_followup != horizon:
        raise ValueError("RQ4 extra-ID surface differs from authenticated staged inputs")
    winner = _load_json(root / surface.winner_selection_path)
    selected = winner.get("selected")
    if (
        winner.get("sha256") != surface.winner_selection_sha256
        or not isinstance(selected, dict)
        or selected.get("row_id") != metadata_row_id
    ):
        raise ValueError("RQ4 extra-ID surface binds another metadata winner")
    if resolve_rq4_feature_data(root=root, surface=surface) != (
        root / surface.metadata_identity.feature_data_path
    ).resolve(strict=True):
        raise ValueError("RQ4 extra-ID surface feature binding changed")
    predecessor = surface.predecessor
    return [
        {
            "id": row.id,
            "family_id": row.family_id,
            "run_name": row.run_name,
            "representation": {
                "catalog": predecessor.catalog_representation,
                "history_hidden_dim": predecessor.history_hidden_dim,
                "extra_item_id_dim": row.extra_item_id_dim,
                "matched_metadata_family": row.matched_metadata_family,
                "matched_metadata_dim": row.matched_metadata_dim,
                "parameter_mismatch_fraction": row.parameter_mismatch_fraction,
            },
            "training": {
                "batch_size": row.batch_size,
                "seed": row.seed,
                "embedding_learning_rate": row.embedding_learning_rate,
                "deep_learning_rate": row.deep_learning_rate,
                "horizon_epochs": row.horizon_epochs,
                "validate_every_epoch": True,
                "restore_best_validation_epoch": True,
            },
        }
        for row in surface.rows
    ]


def _extra_id_coordinate(value: object) -> tuple[object, ...] | None:
    if not isinstance(value, dict):
        return None
    representation = value.get("representation")
    training = value.get("training")
    if not isinstance(representation, dict) or not isinstance(training, dict):
        return None
    return (
        representation.get("matched_metadata_dim"),
        training.get("embedding_learning_rate"),
        training.get("deep_learning_rate"),
        training.get("horizon_epochs"),
    )


def _valid_extra_id_row(
    value: object, *, family_id: object, opportunity_index: int
) -> bool:
    if not isinstance(value, dict):
        return False
    representation = value.get("representation")
    mismatch = (
        representation.get("parameter_mismatch_fraction")
        if isinstance(representation, dict)
        else None
    )
    training = value.get("training")
    return (
        set(value) == {"id", "family_id", "run_name", "representation", "training"}
        and value.get("id") == f"rq4_extra_item_id:{opportunity_index:02d}"
        and value.get("family_id") == "rq4_extra_item_id"
        and isinstance(value.get("run_name"), str)
        and bool(value.get("run_name"))
        and isinstance(representation, dict)
        and set(representation)
        == {
            "catalog",
            "history_hidden_dim",
            "extra_item_id_dim",
            "matched_metadata_family",
            "matched_metadata_dim",
            "parameter_mismatch_fraction",
        }
        and isinstance(representation.get("catalog"), str)
        and type(representation.get("history_hidden_dim")) is int
        and int(representation["history_hidden_dim"]) > 0
        and type(representation.get("extra_item_id_dim")) is int
        and int(representation["extra_item_id_dim"]) > 0
        and representation.get("matched_metadata_family") == family_id
        and type(mismatch) in {int, float}
        and math.isfinite(float(mismatch))
        and 0 <= float(mismatch) < 0.01
        and isinstance(training, dict)
        and set(training)
        == {
            "batch_size",
            "seed",
            "embedding_learning_rate",
            "deep_learning_rate",
            "horizon_epochs",
            "validate_every_epoch",
            "restore_best_validation_epoch",
        }
        and training.get("batch_size") == 512
        and training.get("seed") == 42
        and training.get("validate_every_epoch") is True
        and training.get("restore_best_validation_epoch") is True
    )


def _reauthenticate_result_files(
    root: Path,
    run: Mapping[str, object],
    *,
    identity: Rq4MetadataIdentity,
) -> None:
    root = root.resolve(strict=True)
    if run.get("feature_identity") != _feature_identity_document(identity):
        raise ValueError("RQ4 selected result uses another feature/count identity")
    artifacts = run.get("artifacts")
    queue_reference = run.get("queue_job")
    provenance = run.get("metric_provenance")
    if (
        not isinstance(artifacts, dict)
        or set(artifacts)
        != {contract.name for contract in RQ4_INITIAL_ARTIFACT_CONTRACTS}
        or not isinstance(queue_reference, dict)
        or not isinstance(provenance, dict)
    ):
        raise ValueError("RQ4 selected result lacks bound runtime evidence")
    paths = {
        name: _verify_file_reference(root, reference)
        for name, reference in artifacts.items()
    }
    queue_path = _verify_file_reference(root, queue_reference, allow_job_id=True)
    context_reference = provenance.get("ranking_context")
    context_path = _verify_file_reference(root, context_reference)
    job = run.get("job")
    contract = _load_json(paths["job_contract"])
    queue = _load_json(queue_path)
    if (
        contract.get("row_id") != run.get("row_id")
        or contract.get("ledger_sha256") != run.get("ledger_sha256")
        or contract.get("job") != job
        or queue.get("id") != queue_reference.get("job_id")
        or queue.get("run") != run.get("run_name")
        or queue.get("exit_code") != 0
        or queue.get("data_group") != "g3-native50m-likes"
        or not _ordered_job_times(queue)
    ):
        raise ValueError("RQ4 selected result queue or contract changed")
    reported_metrics = _load_json(paths["final_metrics"])
    recomputed = _recompute_metrics(
        context_path,
        paths["ranking_evidence"],
        paths["top_item_rankings"],
    )
    metrics, coverage_deltas = _authoritative_ranking_metrics(
        reported_metrics, recomputed, row_label=str(run.get("row_id"))
    )
    provenance = run.get("metric_provenance")
    if not isinstance(provenance, dict):
        raise ValueError("RQ4 selected result metric provenance changed")
    if (
        metrics != run.get("metrics")
        or provenance.get("authoritative_noncoverage_source")
        != "ranking_evidence"
        or provenance.get("authoritative_coverage_source") != "top_item_rankings"
        or provenance.get("reported_minus_snapshot_coverage") != coverage_deltas
    ):
        raise ValueError("RQ4 selected result metrics changed")
    item_counts = load_training_item_counts(root / identity.feature_data_path)
    slices = _ranking_slices(
        context_path=context_path,
        ranking_path=paths["ranking_evidence"],
        rankings_path=paths["top_item_rankings"],
        item_counts=item_counts,
        rank_source="ranking_evidence",
    )
    if slices != run.get("slices"):
        raise ValueError("RQ4 selected result slices changed")
    metadata = _load_json(paths["training_metadata"])
    job = run.get("job")
    if not isinstance(job, dict):
        raise ValueError("RQ4 selected result has no exact job")
    if run.get("family_id") == "rq4_extra_item_id":
        _validate_extra_id_training_metadata(metadata, row_document=job)
    elif run.get("family_id") in RQ4_METADATA_FAMILIES:
        _validate_training_metadata(metadata, row_document=job)
    else:
        _validate_predecessor_training_metadata(metadata, row_document=job)
    efficiency = _efficiency(
        metadata=metadata,
        log_path=paths["sweep_log"],
        queue_wall_seconds=float(queue["finished_at"])
        - float(queue["dispatched_at"]),
    )
    if (
        efficiency != run.get("efficiency")
        or run.get("queue_wall_seconds") != efficiency.get("queue_wall_seconds")
    ):
        raise ValueError("RQ4 selected result efficiency changed")


def _validate_extra_id_training_metadata(
    metadata: Mapping[str, object], *, row_document: Mapping[str, object]
) -> None:
    representation = row_document.get("representation")
    training = row_document.get("training")
    if not isinstance(representation, dict) or not isinstance(training, dict):
        raise ValueError("RQ4 extra-ID ledger row is incomplete")
    horizon = training.get("horizon_epochs")
    expected_representation = {
        "catalog_representation": representation.get("catalog"),
        "content_gate": "fixed",
        "extra_item_id_dim": representation.get("extra_item_id_dim"),
        "gate_hidden_dim": None,
        "history_hidden_dim": representation.get("history_hidden_dim"),
        "history_representation": "id_content",
        "metadata": [],
        "metadata_dim": None,
    }
    expected = {
        "batch_size": 512,
        "seed": 42,
        "embedding_learning_rate": training.get("embedding_learning_rate"),
        "deep_learning_rate": training.get("deep_learning_rate"),
        "lr_schedule_horizon_epochs": horizon,
        "num_epochs": horizon,
        "max_epochs": horizon,
        "epochs_trained": horizon,
        "stopped_epoch": horizon,
        "lr_horizon_complete": True,
        "selection_resolved": True,
        "early_stopped": False,
        "g3_dataset_size": "native-50m",
        "g3_protocol_sha256": APPROVED_PROTOCOL_SHA256,
        "g3_representation": expected_representation,
    }
    if any(metadata.get(name) != value for name, value in expected.items()):
        raise ValueError(f"RQ4 extra-ID runtime metadata differs for {row_document['id']}")
    best_epoch = metadata.get("best_epoch")
    if type(horizon) is not int or type(best_epoch) is not int or not 1 <= best_epoch <= horizon:
        raise ValueError(f"RQ4 extra-ID restored epoch is invalid for {row_document['id']}")


def _validate_predecessor_training_metadata(
    metadata: Mapping[str, object], *, row_document: Mapping[str, object]
) -> None:
    representation = row_document.get("representation")
    training = row_document.get("training")
    if not isinstance(representation, dict) or not isinstance(training, dict):
        raise ValueError("RQ4 predecessor job is incomplete")
    expected_representation = {
        "history_representation": representation.get("history_representation"),
        "catalog_representation": representation.get("catalog_representation"),
        "history_hidden_dim": representation.get("history_hidden_dim"),
        "content_gate": "fixed",
        "gate_hidden_dim": None,
        "metadata": [],
        "metadata_dim": None,
        "extra_item_id_dim": None,
    }
    expected = {
        "batch_size": training.get("batch_size"),
        "seed": training.get("seed"),
        "embedding_learning_rate": training.get("embedding_learning_rate"),
        "deep_learning_rate": training.get("deep_learning_rate"),
        "lr_schedule_horizon_epochs": training.get("horizon_epochs"),
        "g3_dataset_size": "native-50m",
        "g3_protocol_sha256": APPROVED_PROTOCOL_SHA256,
        "g3_representation": expected_representation,
    }
    if any(metadata.get(name) != value for name, value in expected.items()):
        raise ValueError("RQ4 predecessor runtime metadata changed")
    best_epoch = metadata.get("best_epoch")
    epochs_trained = metadata.get("epochs_trained")
    horizon = training.get("horizon_epochs")
    if (
        type(horizon) is not int
        or type(best_epoch) is not int
        or type(epochs_trained) is not int
        or not 1 <= best_epoch <= epochs_trained <= horizon
    ):
        raise ValueError("RQ4 predecessor restored epoch is invalid")


def _verify_file_reference(
    root: Path,
    value: object,
    *,
    allow_job_id: bool = False,
) -> Path:
    required = {"path", "size_bytes", "sha256"}
    allowed = required | ({"job_id"} if allow_job_id else set())
    if not isinstance(value, dict) or set(value) != allowed:
        raise ValueError("RQ4 result file reference is invalid")
    raw_path = value.get("path")
    if not isinstance(raw_path, str):
        raise ValueError("RQ4 result file path is invalid")
    path = Path(raw_path)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("RQ4 result file path escapes the project")
    resolved = root / path
    if (
        resolved.is_symlink()
        or not resolved.is_file()
        or not resolved.resolve().is_relative_to(root)
        or type(value.get("size_bytes")) is not int
        or _authenticated_file_fact(root, resolved)
        != {name: value[name] for name in required}
    ):
        raise ValueError("RQ4 result file changed")
    return resolved.resolve(strict=True)


def _ordered_job_times(queue: Mapping[str, object]) -> bool:
    values = [queue.get(name) for name in ("submitted_at", "dispatched_at", "finished_at")]
    return all(_finite_number(value) for value in values) and (
        float(values[0]) <= float(values[1]) <= float(values[2])
    )


def _finite_number(value: object) -> bool:
    return type(value) in {int, float} and math.isfinite(float(value))


def _authoritative_ranking_metrics(
    reported: Mapping[str, object],
    recomputed: Mapping[str, object],
    *,
    row_label: str,
) -> tuple[dict[str, float], dict[str, float]]:
    expected = {*_METRIC_NAMES, "num_users"}
    if set(reported) != expected or set(recomputed) != expected:
        raise ValueError(f"RQ4 metric schema differs for {row_label}")
    result = {}
    coverage_deltas = {}
    for name in sorted(expected):
        if not _finite_number(reported[name]) or not _finite_number(recomputed[name]):
            raise ValueError(f"RQ4 metric is non-finite for {row_label}")
        if name.startswith("coverage@"):
            result[name] = float(recomputed[name])
            coverage_deltas[name] = float(reported[name]) - float(recomputed[name])
        else:
            if abs(float(reported[name]) - float(recomputed[name])) > 1e-15:
                raise ValueError(
                    f"RQ4 non-coverage metric differs from ranking evidence for {row_label}"
                )
            result[name] = float(reported[name])
    return result, coverage_deltas


def _ranking_provenance_resolution(
    root: Path, runs: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    differences = []
    for run in runs:
        artifacts = run.get("artifacts")
        provenance = run.get("metric_provenance")
        if not isinstance(artifacts, dict) or not isinstance(provenance, dict):
            raise ValueError("RQ4 rank resolution lacks authenticated run artifacts")
        ranking_path = root / str(artifacts["ranking_evidence"]["path"])
        snapshot_path = root / str(artifacts["top_item_rankings"]["path"])
        context_path = root / str(provenance["ranking_context"]["path"])
        differences.extend(
            {
                "row_id": run["row_id"],
                **difference,
            }
            for difference in _relevant_rank_differences(
                context_path, ranking_path, snapshot_path
            )
        )
    if differences != [
        {
            "row_id": "rq4_album:07",
            "user_id": 543400,
            "item_id": 14455,
            "ranking_evidence_rank": 42,
            "snapshot_rank": 43,
        }
    ]:
        raise ValueError("RQ4 historical relevant-rank discrepancy changed")
    nonzero_coverage = [
        {
            "row_id": run["row_id"],
            "reported_minus_snapshot": {
                name: value
                for name, value in run["metric_provenance"][
                    "reported_minus_snapshot_coverage"
                ].items()
                if abs(float(value)) > 1e-15
            },
        }
        for run in runs
        if any(
            abs(float(value)) > 1e-15
            for value in run["metric_provenance"][
                "reported_minus_snapshot_coverage"
            ].values()
        )
    ]
    if nonzero_coverage != [
        {
            "row_id": "rq4_artist_album:09",
            "reported_minus_snapshot": {
                "coverage@10": -6.033546518643962e-05,
                "coverage@100": 3.0167732593233687e-05,
            },
        }
    ]:
        raise ValueError("RQ4 historical coverage discrepancy changed")
    diagnostic_path = root / "generated/logs" / _RANK_DIAGNOSTIC_RUN / "rank_diagnostic.json"
    diagnostic_reference = _authenticated_file_fact(root, diagnostic_path)
    if diagnostic_reference["sha256"] != _RANK_DIAGNOSTIC_SHA256:
        raise ValueError("RQ4 rank diagnostic changed")
    diagnostic = _load_json(diagnostic_path)
    passes = diagnostic.get("passes")
    deltas = diagnostic.get("snapshot_minus_evidence")
    if (
        diagnostic.get("run_name") != _RANK_DIAGNOSTIC_RUN
        or not isinstance(passes, list)
        or len(passes) != 2
        or passes[0].get("aggregate_metrics") != passes[1].get("aggregate_metrics")
        or not isinstance(deltas, dict)
        or any(float(value) != 0.0 for value in deltas.values())
    ):
        raise ValueError("RQ4 rank diagnostic does not close both ranking passes")
    queue_path = (
        root
        / "generated/training-queue-service/completed"
        / f"{_RANK_DIAGNOSTIC_JOB_ID}.json"
    )
    queue = _load_json(queue_path)
    if (
        queue.get("id") != _RANK_DIAGNOSTIC_JOB_ID
        or queue.get("run") != _RANK_DIAGNOSTIC_RUN
        or queue.get("exit_code") != 0
        or not _ordered_job_times(queue)
    ):
        raise ValueError("RQ4 rank diagnostic queue completion changed")
    return {
        "authoritative_noncoverage_source": "ranking_evidence",
        "authoritative_item_frequency_slice_source": "ranking_evidence",
        "authoritative_coverage_source": "top_item_rankings",
        "snapshot_scope": ["catalog_identity", "user_identity", "coverage"],
        "historical_relevant_rank_difference": differences[0],
        "hypothetical_snapshot_minus_evidence_metrics": {
            "recall@10": 0.0,
            "recall@50": 0.0,
            "recall@100": 0.0,
            "ndcg@10": 0.0,
            "ndcg@50": -4.813707648958245e-08,
            "ndcg@100": -4.813707648958245e-08,
            "mrr@10": 0.0,
            "mrr@50": -1.6218800704025672e-07,
            "mrr@100": -1.6218800704025672e-07,
        },
        "historical_coverage_difference": nonzero_coverage[0],
        "negative_reproduction": {
            "result": "both passes identical",
            "limitation": (
                "the isolated historical rank difference was not reproduced; "
                "the original checkpoint and logits were not persisted"
            ),
            "diagnostic": diagnostic_reference,
            "queue_job": _authenticated_file_fact(root, queue_path)
            | {"job_id": _RANK_DIAGNOSTIC_JOB_ID},
        },
    }


def _relevant_rank_differences(
    context_path: Path, ranking_path: Path, snapshot_path: Path
) -> list[dict[str, int]]:
    evidence = load_ranking_evidence(context_path, ranking_path)
    snapshot = _load_json(snapshot_path)
    rows = snapshot.get("rankings")
    if not isinstance(rows, list):
        raise ValueError("RQ4 rank-resolution snapshot is invalid")
    rankings = {
        int(row["user_id"]): {
            int(item_id): rank
            for rank, item_id in enumerate(row["item_ids"], start=1)
        }
        for row in rows
    }
    users = [int(value) for value in evidence.user_ids.tolist()]
    items = [int(value) for value in evidence.relevant_item_ids.tolist()]
    offsets = [int(value) for value in evidence.relevance_offsets.tolist()]
    ranks = [int(value) for value in evidence.relevant_ranks.tolist()]
    differences = []
    for position, user_id in enumerate(users):
        start, end = offsets[position : position + 2]
        for item_id, evidence_rank in zip(items[start:end], ranks[start:end], strict=True):
            snapshot_rank = rankings[user_id].get(item_id, 0)
            if evidence_rank != snapshot_rank:
                differences.append(
                    {
                        "user_id": user_id,
                        "item_id": item_id,
                        "ranking_evidence_rank": evidence_rank,
                        "snapshot_rank": snapshot_rank,
                    }
                )
    return differences


def _authenticated_file_fact(root: Path, path: Path) -> dict[str, object]:
    if path.is_symlink():
        raise ValueError(f"RQ4 evidence path must not be a symlink: {path}")
    return _file_fact(root, path)


def _metric(run: Mapping[str, object], name: str) -> float:
    metrics = run.get("metrics")
    value = metrics.get(name) if isinstance(metrics, dict) else None
    if type(value) not in {int, float} or not math.isfinite(float(value)):
        raise ValueError(f"RQ4 run lacks finite {name}")
    return float(value)


def _slice_metric(run: Mapping[str, object], slice_name: str, metric: str) -> float:
    slices = run.get("slices")
    slice_value = slices.get(slice_name) if isinstance(slices, dict) else None
    value = slice_value.get(metric) if isinstance(slice_value, dict) else None
    if type(value) not in {int, float} or not math.isfinite(float(value)):
        raise ValueError(f"RQ4 run lacks finite {slice_name} {metric}")
    return float(value)


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()
