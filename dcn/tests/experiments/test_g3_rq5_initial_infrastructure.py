from __future__ import annotations

import base64
from dataclasses import replace
import hashlib
import inspect
import json
from pathlib import Path

import pytest

from dcn.tests.experiments.test_g3_rq3_post_boundary import _verification
from experiments.g3_pretrained_item_embeddings.analysis import rq5_collection
from experiments.g3_pretrained_item_embeddings.launchers import rq5_initial
from experiments.g3_pretrained_item_embeddings.launchers.control import (
    encode_control_job,
)
from experiments.g3_pretrained_item_embeddings.protocol.rq3_post_boundary import (
    POST_BOUNDARY_ADAPTER_KIND,
    RQ2_FINAL_EVIDENCE_LOGICAL_SHA256,
    RQ2_FINAL_SELECTED_ROW_ID,
)
from experiments.g3_pretrained_item_embeddings.protocol.rq5_initial import (
    RQ5_ARTIFACT_CONTRACTS,
    compile_rq5_initial_ledger,
    persist_rq5_initial_ledger,
    validate_rq5_initial_ledger_document,
)
from experiments.g3_pretrained_item_embeddings.protocol import rq5_horizon_ledger
from experiments.g3_pretrained_item_embeddings.launchers import rq5_horizon
from experiments.g3_pretrained_item_embeddings.launchers import rq5_global_boundary
from experiments.g3_pretrained_item_embeddings.protocol import rq5_global_boundary_ledger


def _ledger(root: Path):
    verification, final_path = _source_fixture(root)

    def verifier(root: Path, path: Path, expected_sha: str, selected_row: str):
        return verification

    ledger = compile_rq5_initial_ledger(
        root=root,
        final_rq2_evidence_path=final_path,
        expected_final_rq2_sha256=RQ2_FINAL_EVIDENCE_LOGICAL_SHA256,
        expected_selected_rq2_row_id=RQ2_FINAL_SELECTED_ROW_ID,
        adapter_kind=POST_BOUNDARY_ADAPTER_KIND,
        verifier=verifier,
    )
    return ledger, verifier


def _reference(root: Path, path: Path) -> dict[str, object]:
    return {
        "path": str(path.relative_to(root)),
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _source_fixture(root: Path, *, feature_bytes: bytes = b"rq5 features"):
    verification = _verification()
    selected_index = next(
        index
        for index, value in enumerate(verification.reusable)
        if value.coordinate.source_id == RQ2_FINAL_SELECTED_ROW_ID
    )
    selected = verification.reusable[selected_index]
    source = selected.coordinate
    feature_path = root / verification.feature.data_path
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    feature_path.write_bytes(feature_bytes)
    manifest_path = root / verification.feature.manifest_path
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(b"feature manifest")
    feature = replace(
        verification.feature,
        data_sha256=hashlib.sha256(feature_bytes).hexdigest(),
        manifest_file_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    )
    source_row = {
        "id": source.source_id,
        "run_name": source.run_name,
        "training": {
            "embedding_learning_rate": source.embedding_learning_rate,
            "deep_learning_rate": source.deep_learning_rate,
            "horizon_epochs": source.horizon_epochs,
        },
    }
    source_ledger_path = root / source.source_ledger_path
    source_ledger_path.parent.mkdir(parents=True, exist_ok=True)
    source_ledger_path.write_text(json.dumps({"rows": [source_row], "sha256": source.source_ledger_sha256}))
    directory = root / "generated/logs" / source.run_name
    directory.mkdir(parents=True, exist_ok=True)
    artifact_paths = {
        contract.name: directory / (
            "g3_rq2_unexpected_diagnostic_job.json"
            if contract.name == "job_contract"
            else contract.filename
        )
        for contract in RQ5_ARTIFACT_CONTRACTS
    }
    artifact_paths["job_contract"].write_text(json.dumps({
        "job": source_row,
        "ledger_path": str(source_ledger_path.resolve()),
        "ledger_sha256": source.source_ledger_sha256,
        "row_id": source.source_id,
    }))
    for name, path in artifact_paths.items():
        if name != "job_contract":
            path.write_bytes(f"source:{name}".encode())
    artifacts = {name: _reference(root, path) for name, path in artifact_paths.items()}
    queue_id = "source-job"
    queue_path = root / "generated/training-queue-service/completed/source-job.json"
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "job": source_row,
        "ledger_sha256": source.source_ledger_sha256,
        "row_id": source.source_id,
    }
    encoded = base64.urlsafe_b64encode(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).decode()
    source_runner = (
        root / "experiments/g3_pretrained_item_embeddings/launchers/"
        "run_rq2_unexpected_diagnostic.py"
    )
    source_runner.parent.mkdir(parents=True, exist_ok=True)
    source_runner.write_text("")
    queue_path.write_text(json.dumps({
        "id": queue_id,
        "batch_id": "source-batch",
        "data_group": "g3-native50m-likes",
        "submitted_at": 1.0,
        "dispatched_at": 2.0,
        "finished_at": 3.0,
        "run": source.run_name,
        "exit_code": 0,
        "script": str(source_runner.resolve()),
        "environment": [
            f"G3_RQ2_UNEXPECTED_DIAGNOSTIC_JOB_B64={encoded}",
            f"G3_RQ2_UNEXPECTED_DIAGNOSTIC_LEDGER_PATH={source_ledger_path.resolve()}",
            "WANDB_MODE=offline",
        ],
    }))
    queue = _reference(root, queue_path) | {"job_id": queue_id}
    source_evidence_path = root / "evidence/source_results.json"
    source_evidence_path.parent.mkdir(parents=True, exist_ok=True)
    source_evidence_logical = "9" * 64
    source_evidence_path.write_text(json.dumps({
        "queue_batch": {"batch_id": "source-batch"},
        "diagnostic_tuning_ledger": [{
            "row_id": source.source_id,
            "run_name": source.run_name,
            "artifacts": artifacts,
            "queue_job": queue,
        }],
        "sha256": source_evidence_logical,
    }))
    source_evidence = _reference(root, source_evidence_path) | {
        "logical_sha256": source_evidence_logical
    }
    source_ledger = _reference(root, source_ledger_path) | {
        "logical_sha256": source.source_ledger_sha256
    }
    artifact_hashes = tuple(
        (name, str(artifacts[name]["sha256"]))
        for name in ("job_contract", "training_metadata", "final_metrics", "training_diagnostics")
    )
    coordinate = replace(source, artifact_sha256=artifact_hashes)
    reusable = list(verification.reusable)
    reusable[selected_index] = replace(selected, coordinate=coordinate)
    verification = replace(verification, feature=feature, reusable=tuple(reusable))
    final_path = root / "final_rq2.json"
    final_path.write_text(json.dumps({
        "final_content_selection": {"selected": {
            "row_id": coordinate.source_id,
            "run_name": coordinate.run_name,
            "capacity": coordinate.history_hidden_dim,
            "embedding_learning_rate": coordinate.embedding_learning_rate,
            "deep_learning_rate": coordinate.deep_learning_rate,
            "horizon_epochs": coordinate.horizon_epochs,
            "artifacts": artifacts,
            "queue_job": queue,
        }},
        "rq3_inputs": {"reuse_source_ledgers": {coordinate.source_id: source_ledger}},
        "diagnostic_evidence": source_evidence,
    }))
    return verification, final_path


def test_initial_ledger_preserves_fixed_reuse_and_equal_gate_budgets(
    tmp_path: Path,
) -> None:
    ledger, _ = _ledger(tmp_path)

    assert ledger.family_opportunity_budgets == {
        "rq5_global_gate": 12,
        "rq5_frequency_gate": 12,
    }
    assert ledger.fixed_gate.content_gate == "fixed"
    assert ledger.fixed_gate.reused_from == RQ2_FINAL_SELECTED_ROW_ID
    assert ledger.fixed_gate_evidence.source_id == RQ2_FINAL_SELECTED_ROW_ID
    assert len(ledger.logical_rows) == 22
    assert len(ledger.rows) == ledger.stage_physical_jobs == 21
    assert sum(row.family_id == "rq5_global_gate" for row in ledger.rows) == 12
    frequency = [row for row in ledger.rows if row.family_id == "rq5_frequency_gate"]
    assert len(frequency) == 9
    assert {row.gate_hidden_dim for row in frequency} == {4, 8, 16}
    assert ledger.deferred_frequency_horizon == {
        "logical_opportunities": 3,
        "materialize_only_after_capacity_selection": True,
    }


@pytest.mark.parametrize(
    ("final_sha", "selected_row"),
    (
        ("0" * 64, RQ2_FINAL_SELECTED_ROW_ID),
        (RQ2_FINAL_EVIDENCE_LOGICAL_SHA256, "rq2_content_concat:20"),
    ),
)
def test_initial_ledger_rejects_wrong_final_rq2_binding(
    tmp_path: Path, final_sha: str, selected_row: str
) -> None:
    final_path = tmp_path / "final_rq2.json"
    final_path.write_text("{}\n")

    with pytest.raises(ValueError, match="final RQ2"):
        compile_rq5_initial_ledger(
            root=tmp_path,
            final_rq2_evidence_path=final_path,
            expected_final_rq2_sha256=final_sha,
            expected_selected_rq2_row_id=selected_row,
            adapter_kind=POST_BOUNDARY_ADAPTER_KIND,
            verifier=lambda *args: _verification(),
        )


def test_ledger_validation_and_queue_cover_only_twenty_one_new_jobs(
    tmp_path: Path,
) -> None:
    ledger, _ = _ledger(tmp_path)
    validate_rq5_initial_ledger_document(ledger.to_dict(), expected=ledger)
    commands = rq5_initial.compile_rq5_initial_queue_commands(
        ledger_path=tmp_path / "rq5.json",
        ledger=ledger,
        state_dir=tmp_path / "queue",
    )

    assert len(commands) == 24
    assert sum("enqueue-run" in command for command in commands) == 21
    assert all(ledger.fixed_gate.run_name not in command for command in commands)
    tampered = ledger.to_dict()
    tampered["physical_rows"][0]["training"]["seed"] = 7
    with pytest.raises(ValueError, match="differs"):
        validate_rq5_initial_ledger_document(tampered, expected=ledger)


def test_worker_bootstrap_and_model_mapping_use_immutable_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verification, final_path = _source_fixture(tmp_path)
    feature_path = tmp_path / verification.feature.data_path
    ledger = compile_rq5_initial_ledger(
        root=tmp_path,
        final_rq2_evidence_path=final_path,
        expected_final_rq2_sha256=RQ2_FINAL_EVIDENCE_LOGICAL_SHA256,
        expected_selected_rq2_row_id=RQ2_FINAL_SELECTED_ROW_ID,
        adapter_kind=POST_BOUNDARY_ADAPTER_KIND,
        verifier=lambda *args: verification,
    )
    monkeypatch.setattr(
        rq5_initial,
        "RQ5_INITIAL_LEDGER_LOGICAL_SHA256",
        ledger.sha256,
    )
    ledger_path = persist_rq5_initial_ledger(tmp_path / "rq5.json", ledger)
    global_row = next(row for row in ledger.rows if row.content_gate == "global")
    monkeypatch.setenv(
        rq5_initial.JOB_ENVIRONMENT,
        encode_control_job(ledger, global_row.id),
    )
    monkeypatch.setenv(rq5_initial.LEDGER_ENVIRONMENT, str(ledger_path))

    compiled, loaded, loaded_path, loaded_features = (
        rq5_initial.compiled_rq5_initial_job_from_environment(root=tmp_path)
    )
    experiment = rq5_initial.build_rq5_initial_training_experiment(
        compiled,
        ledger=loaded,
        feature_data_path=loaded_features,
    )

    assert loaded == ledger
    assert loaded_path == ledger_path
    assert loaded_features == feature_path
    assert experiment.representation.content_gate == "global"
    assert experiment.representation.gate_hidden_dim is None

    encoded = encode_control_job(ledger, global_row.id)
    document = json.loads(base64.urlsafe_b64decode(encoded).decode())
    document["job"]["representation"]["content_gate"] = "frequency"
    monkeypatch.setenv(
        rq5_initial.JOB_ENVIRONMENT,
        base64.urlsafe_b64encode(
            json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
        ).decode(),
    )
    with pytest.raises(ValueError, match="approved ledger row"):
        rq5_initial.compiled_rq5_initial_job_from_environment(root=tmp_path)

    source_metric = dict(ledger.fixed_gate_evidence.artifacts)["final_metrics"]
    (tmp_path / source_metric.path).write_bytes(b"tampered")
    monkeypatch.setenv(
        rq5_initial.JOB_ENVIRONMENT,
        encode_control_job(ledger, global_row.id),
    )
    with pytest.raises(ValueError, match="bound file changed"):
        rq5_initial.compiled_rq5_initial_job_from_environment(root=tmp_path)


def test_worker_rejects_self_consistent_substituted_ledger_and_encoded_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger, _ = _ledger(tmp_path)
    approved_row = ledger.rows[0]
    substituted_row = replace(
        approved_row,
        deep_learning_rate=approved_row.deep_learning_rate * 1.5,
    )
    substituted = replace(
        ledger,
        physical_rows=(substituted_row, *ledger.physical_rows[1:]),
    )
    substituted_path = persist_rq5_initial_ledger(
        tmp_path / "substituted.json", substituted
    )
    monkeypatch.setattr(
        rq5_initial,
        "RQ5_INITIAL_LEDGER_LOGICAL_SHA256",
        ledger.sha256,
    )
    monkeypatch.setenv(
        rq5_initial.JOB_ENVIRONMENT,
        encode_control_job(substituted, substituted_row.id),
    )
    monkeypatch.setenv(rq5_initial.LEDGER_ENVIRONMENT, str(substituted_path))

    with pytest.raises(ValueError, match="approved logical SHA"):
        rq5_initial.compiled_rq5_initial_job_from_environment(root=tmp_path)


def test_collection_rejects_arbitrary_artifact_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger, _ = _ledger(tmp_path)
    ledger_path = persist_rq5_initial_ledger(tmp_path / "rq5.json", ledger)
    batch_id = "batch"
    jobs = [f"job-{index:02d}" for index in range(len(ledger.rows))]
    batch_path = tmp_path / "generated/training-queue-service/batches/batch.json"
    batch_path.parent.mkdir(parents=True, exist_ok=True)
    batch_path.write_text(json.dumps({
        "id": batch_id, "jobs": jobs, "sealed": True,
        "submitted_at": 1.0, "sealed_at": 2.0,
    }))
    context = tmp_path / "generated/logs/.ranking-evidence/g3-native50m/context.pt"
    context.parent.mkdir(parents=True, exist_ok=True)
    context.write_bytes(b"context")
    runner = tmp_path / "experiments/g3_pretrained_item_embeddings/launchers/run_rq5_initial.py"
    runner.parent.mkdir(parents=True, exist_ok=True)
    runner.write_text("")
    thread_env = [
        "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1",
        "NUMEXPR_NUM_THREADS=1", "POLARS_MAX_THREADS=1",
    ]
    for row, job_id in zip(ledger.rows, jobs, strict=True):
        completed = tmp_path / f"generated/training-queue-service/completed/{job_id}.json"
        completed.parent.mkdir(parents=True, exist_ok=True)
        completed.write_text(json.dumps({
            "id": job_id, "batch_id": batch_id, "data_group": "g3-native50m-likes",
            "submitted_at": 1.0, "dispatched_at": 2.0, "finished_at": 3.0,
            "environment": [
                f"{rq5_initial.JOB_ENVIRONMENT}={encode_control_job(ledger, row.id)}",
                f"{rq5_initial.LEDGER_ENVIRONMENT}={ledger_path}",
                "WANDB_MODE=offline", *thread_env,
            ],
            "exit_code": 0, "run": row.run_name, "script": str(runner),
        }))
        directory = tmp_path / "generated/logs" / row.run_name
        directory.mkdir(parents=True, exist_ok=True)
        for contract in RQ5_ARTIFACT_CONTRACTS:
            (directory / contract.filename).write_bytes(b"arbitrary")
    monkeypatch.setattr(rq5_collection, "preview_rq5_initial_ledger", lambda root: ledger)
    monkeypatch.setattr(rq5_collection, "load_training_item_counts", lambda path: {})

    with pytest.raises(ValueError, match="cannot load"):
        rq5_collection.build_rq5_initial_collection(
            root=tmp_path,
            ledger_path=ledger_path,
            expected_ledger_sha256=ledger.sha256,
            batch_id=batch_id,
        )


def test_submission_has_no_preview_injection_bypass() -> None:
    assert "preview" not in inspect.signature(rq5_initial.submit_rq5_initial_jobs).parameters


def _selection_runs(ledger):
    runs = []
    for row in ledger.rows:
        recall = 0.1
        if row.id in {"rq5_global_gate:10", "rq5_frequency_gate:04"}:
            recall = 0.3
        runs.append({
            "row_id": row.id,
            "family_id": row.family_id,
            "run_name": row.run_name,
            "content_gate": row.content_gate,
            "gate_hidden_dim": row.gate_hidden_dim,
            "embedding_learning_rate": row.embedding_learning_rate,
            "deep_learning_rate": row.deep_learning_rate,
            "horizon_epochs": row.horizon_epochs,
            "best_epoch": min(22, row.horizon_epochs),
            "queue_wall_seconds": 10.0,
            "metrics": {"recall@100": recall, "ndcg@100": recall},
        })
    return runs


def test_initial_selection_uses_outer_ten_percent_boundaries(tmp_path: Path) -> None:
    ledger, _ = _ledger(tmp_path)

    selected = rq5_collection.select_rq5_initial_winners(
        _selection_runs(ledger), ledger=ledger
    )

    assert selected["global_gate"]["selected_row_id"] == "rq5_global_gate:10"
    assert (
        selected["global_gate"]["boundaries"]["deep_learning_rate"]["direction"]
        == "lower"
    )
    frequency = selected["frequency_capacity"]
    assert frequency["selected_row_id"] == "rq5_frequency_gate:04"
    assert frequency["selected_gate_hidden_dim"] == 8
    assert frequency["capacity_boundary"]["direction"] is None
    assert (
        frequency["coordinate_boundaries"]["embedding_learning_rate"]["direction"]
        == "lower"
    )


def test_horizon_ledger_has_exact_three_physical_width_eight_cells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger, _ = _ledger(tmp_path)
    initial_path = persist_rq5_initial_ledger(tmp_path / "rq5.json", ledger)
    initial_reference = _reference(tmp_path, initial_path) | {
        "logical_sha256": ledger.sha256
    }
    collection_payload = {
        "schema_version": 1,
        "kind": "g3_rq5_initial_collection",
        "protocol_sha256": rq5_horizon_ledger.APPROVED_PROTOCOL_SHA256,
        "ledger": initial_reference,
        "queue_batch": {"batch_id": "initial-batch"},
        "runs": _selection_runs(ledger),
    }
    collection_sha = hashlib.sha256(
        json.dumps(
            collection_payload, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    collection_path = tmp_path / "initial.json"
    collection_path.write_text(json.dumps(collection_payload | {"sha256": collection_sha}))
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text("{}")
    monkeypatch.setattr(
        rq5_horizon_ledger, "RQ5_INITIAL_LEDGER_LOGICAL_SHA256", ledger.sha256
    )
    monkeypatch.setattr(
        rq5_horizon_ledger, "CONTROL_CALIBRATION_PATH", "calibration.json"
    )
    monkeypatch.setattr(
        rq5_horizon_ledger, "CONTROL_CALIBRATION_SHA256", "a" * 64
    )
    calibration = {
        "sha256": "a" * 64,
        "power_law_fits": {
            "embedding_learning_rate": {"fitted_coordinates": {
                "15": 0.047134737607146836,
                "25": 0.12447135415265811,
                "40": 0.3041556165944196,
            }},
            "deep_learning_rate": {"fitted_coordinates": {
                "15": 0.04127129308065626,
                "25": 0.023941907610393703,
                "40": 0.014506684820055783,
            }},
        },
    }
    monkeypatch.setattr(
        rq5_horizon_ledger, "load_control_calibration", lambda path: calibration
    )

    horizon = rq5_horizon_ledger.compile_rq5_horizon_ledger(
        root=tmp_path,
        initial_ledger_path=initial_path,
        initial_collection_path=collection_path,
        expected_initial_collection_sha256=collection_sha,
    )
    commands = rq5_horizon.compile_rq5_horizon_queue_commands(
        ledger_path=tmp_path / "horizon.json",
        ledger=horizon,
        state_dir=tmp_path / "queue",
    )

    assert [row.id for row in horizon.rows] == [
        "rq5_frequency_gate:10",
        "rq5_frequency_gate:11",
        "rq5_frequency_gate:12",
    ]
    assert [row.horizon_epochs for row in horizon.rows] == [15, 25, 40]
    assert all(row.gate_hidden_dim == 8 for row in horizon.rows)
    assert len(horizon.physical_rows) == 3
    assert len(commands) == 6
    assert sum("enqueue-run" in command for command in commands) == 3

    tampered = horizon.to_dict()
    tampered["schema_version"] = True
    tampered_path = tmp_path / "tampered_horizon.json"
    tampered_path.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="logical SHA changed"):
        rq5_horizon_ledger.load_rq5_horizon_ledger(
            tampered_path,
            root=tmp_path,
            expected_ledger_sha256=horizon.sha256,
        )


def test_global_boundary_has_exact_three_joint_lower_deep_probes() -> None:
    root = Path(__file__).resolve().parents[3]
    ledger = rq5_global_boundary_ledger.compile_rq5_global_boundary_ledger(
        root=root
    )
    commands = rq5_global_boundary.compile_rq5_global_boundary_queue_commands(
        ledger_path=root / rq5_global_boundary_ledger.RQ5_GLOBAL_BOUNDARY_LEDGER_PATH,
        ledger=ledger,
        state_dir=root / "generated/training-queue-service",
    )

    assert ledger.sha256 == rq5_global_boundary.RQ5_GLOBAL_BOUNDARY_LEDGER_LOGICAL_SHA256
    assert [row.id for row in ledger.rows] == [
        "rq5_global_gate:13",
        "rq5_global_gate:14",
        "rq5_global_gate:15",
    ]
    assert tuple(row.deep_learning_rate for row in ledger.rows) == (
        0.008017812814887691,
        0.0056694498116914875,
        0.004008906407443846,
    )
    assert {row.embedding_learning_rate for row in ledger.rows} == {
        0.12305770976863895
    }
    assert {row.horizon_epochs for row in ledger.rows} == {40}
    assert len(commands) == 6
    assert sum("enqueue-run" in command for command in commands) == 3


def test_global_boundary_loader_rejects_stale_and_self_consistent_mutations(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[3]
    ledger = rq5_global_boundary_ledger.compile_rq5_global_boundary_ledger(
        root=root
    )
    stale = ledger.to_dict()
    stale["schema_version"] = True
    stale_path = tmp_path / "stale.json"
    stale_path.write_text(json.dumps(stale))
    with pytest.raises(ValueError, match="logical SHA changed"):
        rq5_global_boundary_ledger.load_rq5_global_boundary_ledger(
            stale_path, root=root, expected_ledger_sha256=ledger.sha256
        )

    substituted = ledger.to_dict()
    substituted["rows"][0]["training"]["deep_learning_rate"] *= 0.5
    substituted["sha256"] = hashlib.sha256(
        json.dumps(
            {key: value for key, value in substituted.items() if key != "sha256"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    substituted_path = tmp_path / "substituted.json"
    substituted_path.write_text(json.dumps(substituted))
    with pytest.raises(ValueError, match="frozen inputs"):
        rq5_global_boundary_ledger.load_rq5_global_boundary_ledger(
            substituted_path, root=root
        )
