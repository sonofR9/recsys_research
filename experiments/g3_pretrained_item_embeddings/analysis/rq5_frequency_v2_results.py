from __future__ import annotations

import json
from pathlib import Path

from experiments.g3_pretrained_item_embeddings.analysis.control_calibration import (
    _file_fact,
    _load_json,
)
from experiments.g3_pretrained_item_embeddings.analysis.rq1_results import (
    load_training_item_counts,
)
from experiments.g3_pretrained_item_embeddings.analysis.rq5_collection import (
    _ARTIFACT_FILENAMES,
    _boundaries,
    _collect_run,
    _document,
    _exact_json_equal,
    _feature_identity,
    _require_completed_batch,
    _select,
    _validate_batch,
)
from experiments.g3_pretrained_item_embeddings.launchers import rq5_frequency_v2
from experiments.g3_pretrained_item_embeddings.protocol.constants import (
    APPROVED_PROTOCOL_SHA256,
)
from experiments.g3_pretrained_item_embeddings.protocol.rq5_frequency_v2_ledger import (
    RQ5_FREQUENCY_V2_LEDGER_PATH,
    load_rq5_frequency_v2_ledger,
    verify_rq5_frequency_v2_inputs,
)
from experiments.g3_pretrained_item_embeddings.protocol.rq5_initial import (
    load_rq5_initial_ledger,
)


RQ5_FREQUENCY_V2_INITIAL_EVIDENCE_PATH = (
    "experiments/g3_pretrained_item_embeddings/evidence/"
    "rq5_frequency_gate_fp32_p09_v2_initial_native50m.json"
)
RQ5_FREQUENCY_V2_HORIZON_EVIDENCE_PATH = (
    "experiments/g3_pretrained_item_embeddings/evidence/"
    "rq5_frequency_gate_fp32_p09_v2_horizons_native50m.json"
)


def build_rq5_frequency_v2_initial_collection(
    *, root: Path, ledger_path: Path, expected_ledger_sha256: str, batch_id: str
) -> dict[str, object]:
    root = root.resolve(strict=True)
    ledger_path = ledger_path.resolve(strict=True)
    ledger = load_rq5_frequency_v2_ledger(
        ledger_path,
        root=root,
        expected_ledger_sha256=expected_ledger_sha256,
    )
    feature_path = verify_rq5_frequency_v2_inputs(root, ledger)
    initial = load_rq5_initial_ledger(root / ledger.initial_ledger.path)
    batch_path = root / "generated/training-queue-service/batches" / f"{batch_id}.json"
    job_ids = _validate_batch(_load_json(batch_path), batch_id, len(ledger.rows))
    _require_completed_batch(root, job_ids)
    context_path = root / "generated/logs/.ranking-evidence/g3-native50m/context.pt"
    artifact_filenames = dict(_ARTIFACT_FILENAMES) | {
        "job_contract": "g3_rq5_frequency_v2_job.json",
        "gate_diagnostics": "g3_gate_diagnostics.json",
    }
    item_counts = load_training_item_counts(feature_path)
    runs = [
        _collect_run(
            root=root,
            ledger=ledger,
            ledger_path=ledger_path,
            row=row,
            batch_id=batch_id,
            job_id=job_id,
            context_path=context_path,
            item_counts=item_counts,
            feature_identity=_feature_identity(initial),
            runner_filename="run_rq5_frequency_v2.py",
            job_environment=rq5_frequency_v2.JOB_ENVIRONMENT,
            ledger_environment=rq5_frequency_v2.LEDGER_ENVIRONMENT,
            artifact_filenames=artifact_filenames,
        )
        for row, job_id in zip(ledger.rows, job_ids, strict=True)
    ]
    _validate_gate_mechanism(root, runs, ledger.rows)
    order = {row.id: index for index, row in enumerate(ledger.rows)}
    selected = _select(runs, order=order)
    width = int(selected["gate_hidden_dim"])
    width_rows = [run for run in runs if run["gate_hidden_dim"] == width]
    return _document(
        {
            "schema_version": 1,
            "kind": "g3_rq5_frequency_gate_fp32_p09_v2_initial_collection",
            "protocol_sha256": APPROVED_PROTOCOL_SHA256,
            "ledger": _file_fact(root, ledger_path)
            | {"logical_sha256": ledger.sha256},
            "queue_batch": _file_fact(root, batch_path) | {"batch_id": batch_id},
            "ranking_context": _file_fact(root, context_path),
            "runs": runs,
            "selection": {
                "selected": selected,
                "selected_row_id": selected["row_id"],
                "selected_gate_hidden_dim": width,
                "capacity_boundary": {
                    "tested_values": [4, 8, 16],
                    "direction": (
                        "lower" if width == 4 else "upper" if width == 16 else None
                    ),
                },
                "coordinate_boundaries": _boundaries(selected, width_rows),
                "required_horizon_cells": [15, 25, 40],
            },
        }
    )


def persist_rq5_frequency_v2_initial_collection(
    path: Path,
    document: dict[str, object],
    *,
    root: Path,
    ledger_path: Path,
    expected_ledger_sha256: str,
    batch_id: str,
) -> Path:
    root = root.resolve(strict=True)
    destination = (root / RQ5_FREQUENCY_V2_INITIAL_EVIDENCE_PATH).resolve()
    if path.resolve() != destination or destination.is_symlink():
        raise ValueError("RQ5 frequency v2 evidence destination is not canonical")
    rebuilt = build_rq5_frequency_v2_initial_collection(
        root=root,
        ledger_path=ledger_path,
        expected_ledger_sha256=expected_ledger_sha256,
        batch_id=batch_id,
    )
    if not _exact_json_equal(document, rebuilt):
        raise ValueError("RQ5 frequency v2 evidence differs from rebuilt evidence")
    content = (json.dumps(rebuilt, sort_keys=True, separators=(",", ":")) + "\n").encode()
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("xb") as handle:
            handle.write(content)
    except FileExistsError:
        if destination.read_bytes() != content:
            raise RuntimeError(f"immutable RQ5 frequency v2 evidence differs: {destination}")
    return destination


def build_rq5_frequency_v2_horizon_collection(
    *, root: Path, ledger_path: Path, expected_ledger_sha256: str, batch_id: str
) -> dict[str, object]:
    from experiments.g3_pretrained_item_embeddings.launchers import (
        rq5_frequency_v2_horizon,
    )
    from experiments.g3_pretrained_item_embeddings.protocol.rq5_frequency_v2_horizon_ledger import (
        load_rq5_frequency_v2_horizon_ledger,
        verify_rq5_frequency_v2_horizon_inputs,
    )

    root = root.resolve(strict=True)
    ledger_path = ledger_path.resolve(strict=True)
    ledger = load_rq5_frequency_v2_horizon_ledger(
        ledger_path,
        root=root,
        expected_ledger_sha256=expected_ledger_sha256,
    )
    feature_path = verify_rq5_frequency_v2_horizon_inputs(root, ledger)
    initial_ledger = load_rq5_frequency_v2_ledger(
        root / ledger.initial_ledger.path,
        root=root,
        expected_ledger_sha256=ledger.initial_ledger.logical_sha256,
    )
    original_initial = load_rq5_initial_ledger(
        root / initial_ledger.initial_ledger.path
    )
    initial_evidence = _load_json(root / ledger.initial_evidence.path)
    batch_path = root / "generated/training-queue-service/batches" / f"{batch_id}.json"
    job_ids = _validate_batch(_load_json(batch_path), batch_id, len(ledger.rows))
    _require_completed_batch(root, job_ids)
    context_path = root / "generated/logs/.ranking-evidence/g3-native50m/context.pt"
    artifact_filenames = dict(_ARTIFACT_FILENAMES) | {
        "job_contract": "g3_rq5_frequency_v2_horizon_job.json",
        "gate_diagnostics": "g3_gate_diagnostics.json",
    }
    item_counts = load_training_item_counts(feature_path)
    runs = [
        _collect_run(
            root=root,
            ledger=ledger,
            ledger_path=ledger_path,
            row=row,
            batch_id=batch_id,
            job_id=job_id,
            context_path=context_path,
            item_counts=item_counts,
            feature_identity=_feature_identity(original_initial),
            runner_filename="run_rq5_frequency_v2_horizon.py",
            job_environment=rq5_frequency_v2_horizon.JOB_ENVIRONMENT,
            ledger_environment=rq5_frequency_v2_horizon.LEDGER_ENVIRONMENT,
            artifact_filenames=artifact_filenames,
        )
        for row, job_id in zip(ledger.rows, job_ids, strict=True)
    ]
    _validate_gate_mechanism(root, runs, ledger.rows)
    combined = [*initial_evidence["runs"], *runs]
    if len(combined) != 12:
        raise ValueError("RQ5 frequency v2 combined opportunity count changed")
    order = {
        f"rq5_frequency_gate_v2:{index:02d}": index for index in range(1, 13)
    }
    selected = _select(combined, order=order)
    width = int(selected["gate_hidden_dim"])
    boundaries = _boundaries(selected, combined)
    return _document(
        {
            "schema_version": 1,
            "kind": "g3_rq5_frequency_gate_fp32_p09_v2_horizon_collection",
            "protocol_sha256": APPROVED_PROTOCOL_SHA256,
            "initial_evidence": ledger.initial_evidence.to_dict(),
            "ledger": _file_fact(root, ledger_path)
            | {"logical_sha256": ledger.sha256},
            "queue_batch": _file_fact(root, batch_path) | {"batch_id": batch_id},
            "runs": runs,
            "combined_selection": {
                "selected": selected,
                "selected_row_id": selected["row_id"],
                "selected_gate_hidden_dim": width,
                "capacity_boundary": {
                    "tested_values": [4, 8, 16],
                    "direction": (
                        "lower" if width == 4 else "upper" if width == 16 else None
                    ),
                },
                "boundaries": boundaries,
                "second_boundary_unresolved": any(
                    boundaries[name]["direction"] is not None
                    for name in ("embedding_learning_rate", "deep_learning_rate")
                )
                or boundaries["horizon"]["extend_to_epochs"] is not None,
                "next_action": "renewed_approval"
                if any(
                    boundaries[name]["direction"] is not None
                    for name in ("embedding_learning_rate", "deep_learning_rate")
                )
                or boundaries["horizon"]["extend_to_epochs"] is not None
                else "none",
            },
        }
    )


def persist_rq5_frequency_v2_horizon_collection(
    path: Path,
    document: dict[str, object],
    *,
    root: Path,
    ledger_path: Path,
    expected_ledger_sha256: str,
    batch_id: str,
) -> Path:
    root = root.resolve(strict=True)
    destination = (root / RQ5_FREQUENCY_V2_HORIZON_EVIDENCE_PATH).resolve()
    if path.resolve() != destination or destination.is_symlink():
        raise ValueError("RQ5 frequency v2 horizon evidence destination is not canonical")
    rebuilt = build_rq5_frequency_v2_horizon_collection(
        root=root,
        ledger_path=ledger_path,
        expected_ledger_sha256=expected_ledger_sha256,
        batch_id=batch_id,
    )
    if not _exact_json_equal(document, rebuilt):
        raise ValueError("RQ5 frequency v2 horizon evidence differs from rebuilt evidence")
    content = (json.dumps(rebuilt, sort_keys=True, separators=(",", ":")) + "\n").encode()
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("xb") as handle:
            handle.write(content)
    except FileExistsError:
        if destination.read_bytes() != content:
            raise RuntimeError(f"immutable RQ5 frequency v2 horizon evidence differs: {destination}")
    return destination


def _validate_gate_mechanism(root: Path, runs: list[dict], rows: tuple) -> None:
    for run, row in zip(runs, rows, strict=True):
        path = root / run["artifacts"]["gate_diagnostics"]["path"]
        document = _load_json(path)
        epochs = document.get("epochs")
        if (
            document.get("frequency_input_parity") is not True
            or not isinstance(epochs, list)
            or len(epochs) != row.horizon_epochs
            or any(
                entry["gate_parameter_gradient_norm"]["nonfinite_count"] != 0
                or entry["gate_parameter_gradient_norm"]["mean"] <= 0
                for entry in epochs
            )
        ):
            raise ValueError(f"corrected frequency gate did not train for {row.id}")
