from __future__ import annotations

import base64
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.g3_pretrained_item_embeddings.configs.model import (
    RQ3_CATALOG_REPRESENTATIONS,
)
from experiments.g3_pretrained_item_embeddings.launchers import rq3_post_boundary
from experiments.g3_pretrained_item_embeddings.protocol.rq3 import (
    AuthenticatedRq2Coordinate,
)
from experiments.g3_pretrained_item_embeddings.protocol.rq3_post_boundary import (
    POST_BOUNDARY_ADAPTER_KIND,
    RQ2_FINAL_EVIDENCE_LOGICAL_SHA256,
    RQ3_ELIGIBLE_REUSE_IDS,
    RQ3_OUTPUT_ARTIFACT_CONTRACTS,
    Rq3FeatureBinding,
    Rq3PostBoundaryVerification,
    Rq3VerifiedReuse,
    compile_rq3_post_boundary_ledger,
    compile_verified_rq3_post_boundary_surface,
    persist_rq3_post_boundary_ledger,
    validate_rq3_post_boundary_ledger_document,
)


SHA = RQ2_FINAL_EVIDENCE_LOGICAL_SHA256
SELECTED_ROW_ID = "rq2_unexpected_diagnostic:03"


def _source(source_id: str, index: int) -> Rq3VerifiedReuse:
    coordinate = AuthenticatedRq2Coordinate(
        source_id=source_id,
        run_name=f"authenticated_source_{index:02d}",
        history_hidden_dim=128,
        embedding_learning_rate=0.1 + index * 0.01,
        deep_learning_rate=0.005 + index * 0.001,
        horizon_epochs=40,
        source_ledger_path=f"evidence/source_{index:02d}.json",
        source_ledger_sha256=f"{index + 1:064x}",
        source_evidence_sha256=f"{index + 20:064x}",
        artifact_sha256=(
            ("job_contract", f"{index + 40:064x}"),
            ("training_metadata", f"{index + 50:064x}"),
            ("final_metrics", f"{index + 60:064x}"),
            ("training_diagnostics", f"{index + 70:064x}"),
        ),
        training_count_sha256="e" * 64,
        slice_membership_sha256="f" * 64,
        diagnostics_schema_version=2,
        diagnostics_epoch_count=40,
    )
    return Rq3VerifiedReuse(
        coordinate=coordinate,
        source_ledger_path=coordinate.source_ledger_path,
        source_ledger_sha256=coordinate.source_ledger_sha256,
    )


def _verification(*, data_sha256: str = "d" * 64) -> Rq3PostBoundaryVerification:
    return Rq3PostBoundaryVerification(
        adapter_kind=POST_BOUNDARY_ADAPTER_KIND,
        final_evidence_sha256=SHA,
        selected_row_id=SELECTED_ROW_ID,
        selected_history_hidden_dim=128,
        feature=Rq3FeatureBinding(
            manifest_path="protocol/features.json",
            manifest_sha256="b" * 64,
            manifest_file_sha256="c" * 64,
            data_path="features/item_features.parquet",
            data_sha256=data_sha256,
            frequency_terciles={
                "num_catalog_items": 3,
                "slices": {
                    "tail": {"num_items": 1, "training_interactions": 1},
                    "mid": {"num_items": 1, "training_interactions": 2},
                    "head": {"num_items": 1, "training_interactions": 3},
                },
            },
            training_count_reference={
                "encoding": "canonical-json-integers",
                "length": 4,
                "sha256": "e" * 64,
            },
            slice_membership_reference={
                "encoding": "canonical-json-integers",
                "length": 4,
                "sha256": "f" * 64,
            },
        ),
        reusable=tuple(
            _source(source_id, index)
            for index, source_id in enumerate(sorted(RQ3_ELIGIBLE_REUSE_IDS))
        ),
    )


def _compile(tmp_path: Path, verification: Rq3PostBoundaryVerification | None = None):
    evidence_path = tmp_path / "final.json"
    evidence_path.write_text("{}\n")
    verified = verification or _verification()
    calls = []

    def verifier(root: Path, path: Path, expected_sha: str, selected_row: str):
        calls.append((root, path, expected_sha, selected_row))
        return verified

    surface = compile_verified_rq3_post_boundary_surface(
        root=tmp_path,
        final_evidence_path=evidence_path,
        expected_final_rq2_evidence_sha256=SHA,
        expected_selected_rq2_row_id=SELECTED_ROW_ID,
        adapter_kind=POST_BOUNDARY_ADAPTER_KIND,
        verifier=verifier,
    )
    assert calls == [(tmp_path.resolve(), evidence_path, SHA, SELECTED_ROW_ID)]
    return surface


def test_verified_surface_and_preview_ledger_have_45_opportunities_and_38_jobs(
    tmp_path: Path,
) -> None:
    surface = _compile(tmp_path)
    ledger = compile_rq3_post_boundary_ledger(surface)

    assert surface.final_rq2_evidence_sha256 == SHA
    assert surface.selected_rq2_row_id == SELECTED_ROW_ID
    assert len(ledger.logical_rows) == 45
    assert len(ledger.physical_rows) == len(ledger.rows) == 38
    assert {row.reused_from for row in ledger.logical_rows if row.reused_from} == (
        RQ3_ELIGIBLE_REUSE_IDS
    )
    assert len({row.id for row in ledger.logical_rows}) == 45
    assert len({row.run_name for row in ledger.logical_rows}) == 45
    assert all(row.batch_size == 512 and row.seed == 42 for row in ledger.logical_rows)
    assert all(row.reused_from is None for row in ledger.physical_rows)
    source_ledgers = {
        source_id: (path, sha256)
        for source_id, path, sha256 in ledger.source_ledgers
    }
    assert tuple(source_ledgers) == tuple(sorted(RQ3_ELIGIBLE_REUSE_IDS))
    assert all(
        source_ledgers[row.reused_from]
        == (row.source_ledger_path, row.source_ledger_sha256)
        for row in ledger.logical_rows
        if row.reused_from is not None
    )


def test_every_output_family_has_exact_model_mapping(tmp_path: Path) -> None:
    ledger = compile_rq3_post_boundary_ledger(_compile(tmp_path))

    assert {
        row.family_id: row.catalog_representation
        for row in ledger.logical_rows
    } == RQ3_CATALOG_REPRESENTATIONS
    for family_id, catalog_representation in RQ3_CATALOG_REPRESENTATIONS.items():
        row = next(value for value in ledger.rows if value.family_id == family_id)
        compiled = rq3_post_boundary.decode_rq3_post_boundary_job(
            rq3_post_boundary.encode_rq3_post_boundary_job(ledger, row.id),
            ledger,
        )
        experiment = rq3_post_boundary.build_rq3_post_boundary_training_experiment(
            compiled,
            ledger=ledger,
            feature_data_path=tmp_path / "features.parquet",
        )
        assert experiment.representation.history_representation == "id_content"
        assert experiment.representation.history_hidden_dim == 128
        assert experiment.representation.catalog_representation == catalog_representation


@pytest.mark.parametrize("mutation", ["duplicate", "ineligible", "sha", "selected"])
def test_adapter_rejects_ineligible_or_unauthenticated_handoffs(
    tmp_path: Path, mutation: str
) -> None:
    verification = _verification()
    if mutation == "duplicate":
        verification = replace(
            verification,
            reusable=(*verification.reusable[:-1], verification.reusable[0]),
        )
    elif mutation == "ineligible":
        verification = replace(
            verification,
            reusable=(*verification.reusable[:-1], _source("rq2_content_concat:18", 90)),
        )
    elif mutation == "sha":
        verification = replace(verification, final_evidence_sha256="0" * 64)
    else:
        verification = replace(verification, selected_row_id="rq2_content_concat:20")

    with pytest.raises(ValueError):
        _compile(tmp_path, verification)


@pytest.mark.parametrize("reference", ["training_count", "slice_membership"])
def test_adapter_rejects_reuse_coordinate_with_another_feature_identity(
    tmp_path: Path,
    reference: str,
) -> None:
    verification = _verification()
    source = verification.reusable[0]
    coordinate = source.coordinate
    if reference == "training_count":
        coordinate = replace(coordinate, training_count_sha256="0" * 64)
    else:
        coordinate = replace(coordinate, slice_membership_sha256="0" * 64)
    verification = replace(
        verification,
        reusable=(replace(source, coordinate=coordinate), *verification.reusable[1:]),
    )

    with pytest.raises(ValueError, match="feature count/slice"):
        _compile(tmp_path, verification)


def test_adapter_rejects_malformed_nested_feature_reference(tmp_path: Path) -> None:
    verification = _verification()
    feature = replace(
        verification.feature,
        training_count_reference={
            **verification.feature.training_count_reference,
            "length": 3,
        },
    )

    with pytest.raises(ValueError, match="nested feature"):
        _compile(tmp_path, replace(verification, feature=feature))


def test_ledger_tamper_and_reused_launch_are_rejected(tmp_path: Path) -> None:
    ledger = compile_rq3_post_boundary_ledger(_compile(tmp_path))
    document = ledger.to_dict()
    document["physical_rows"][0]["training"]["seed"] = 7

    with pytest.raises(ValueError, match="differs"):
        validate_rq3_post_boundary_ledger_document(document, expected=ledger)

    reused = next(row for row in ledger.logical_rows if row.reused_from is not None)
    with pytest.raises(ValueError, match="reused"):
        rq3_post_boundary.encode_rq3_post_boundary_job(ledger, reused.id)

    row = ledger.rows[0]
    encoded = rq3_post_boundary.encode_rq3_post_boundary_job(ledger, row.id)
    job = json.loads(base64.urlsafe_b64decode(encoded))
    job["job"]["training"]["seed"] = 7
    tampered = base64.urlsafe_b64encode(
        json.dumps(job, sort_keys=True, separators=(",", ":")).encode()
    ).decode()
    with pytest.raises(ValueError, match="differs"):
        rq3_post_boundary.decode_rq3_post_boundary_job(tampered, ledger)


def test_queue_compiles_only_38_physical_jobs(tmp_path: Path) -> None:
    ledger = compile_rq3_post_boundary_ledger(_compile(tmp_path))
    commands = rq3_post_boundary.compile_rq3_post_boundary_queue_commands(
        ledger_path=tmp_path / "preview.json",
        ledger=ledger,
        state_dir=tmp_path / "queue",
    )

    assert len(commands) == 41
    assert sum("enqueue-run" in command for command in commands) == 38


def test_result_and_diagnostic_artifact_contract_is_closed() -> None:
    assert {contract.name: contract.filename for contract in RQ3_OUTPUT_ARTIFACT_CONTRACTS} == {
        "job_contract": "g3_rq3_output_job.json",
        "training_metadata": "training_metadata.json",
        "final_metrics": "final_metrics.json",
        "ranking_evidence": "ranking_evidence.pt",
        "top_item_rankings": "top_item_rankings.json",
        "training_diagnostics": "g3_training_diagnostics.json",
        "sweep_log": "sweep.log",
    }
    diagnostics = next(
        value for value in RQ3_OUTPUT_ARTIFACT_CONTRACTS
        if value.name == "training_diagnostics"
    )
    assert diagnostics.schema_versions == (2,)
    assert {"epochs", "frequency_terciles"} <= set(diagnostics.required_keys)


def test_worker_bootstraps_from_immutable_ledger_without_process_registration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_path = tmp_path / "features/item_features.parquet"
    feature_path.parent.mkdir(parents=True)
    feature_path.write_bytes(b"authenticated features")
    verification = _verification(
        data_sha256=hashlib.sha256(feature_path.read_bytes()).hexdigest()
    )
    ledger = compile_rq3_post_boundary_ledger(_compile(tmp_path, verification))
    ledger_path = persist_rq3_post_boundary_ledger(tmp_path / "preview.json", ledger)
    row = ledger.rows[0]
    monkeypatch.setenv(
        rq3_post_boundary.JOB_ENVIRONMENT,
        rq3_post_boundary.encode_rq3_post_boundary_job(ledger, row.id),
    )
    monkeypatch.setenv(rq3_post_boundary.LEDGER_ENVIRONMENT, str(ledger_path))
    compiled, loaded, loaded_path, loaded_features = (
        rq3_post_boundary.compiled_rq3_post_boundary_job_from_environment(
            root=tmp_path,
        )
    )

    assert compiled.row_id == row.id
    assert loaded == ledger
    assert loaded_path == ledger_path
    assert loaded_features == feature_path


@pytest.mark.parametrize("dry_run", [True, False])
def test_submission_verifies_before_any_queue_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dry_run: bool,
) -> None:
    feature_path = tmp_path / "features/item_features.parquet"
    feature_path.parent.mkdir(parents=True)
    feature_path.write_bytes(b"authenticated features")
    verification = _verification(
        data_sha256=hashlib.sha256(feature_path.read_bytes()).hexdigest()
    )
    ledger = compile_rq3_post_boundary_ledger(_compile(tmp_path, verification))
    ledger_path = persist_rq3_post_boundary_ledger(tmp_path / "preview.json", ledger)
    events = []

    def preview():
        events.append("verify")
        return ledger

    def run(command, **kwargs):
        events.append("queue")
        stdout = "BATCH\n" if "new-batch" in command else ""
        return SimpleNamespace(stdout=stdout)

    monkeypatch.setattr(rq3_post_boundary.subprocess, "run", run)
    result = rq3_post_boundary.submit_rq3_post_boundary_jobs(
        ledger_path=ledger_path,
        state_dir=tmp_path / "queue",
        dry_run=dry_run,
        root=tmp_path,
        preview=preview,
    )

    assert events[0] == "verify"
    assert events.count("verify") == 1
    if dry_run:
        assert events == ["verify"]
        assert result.count("enqueue-run") == 38
    else:
        assert result == "BATCH"
        assert events.count("queue") == 41


def test_actual_final_evidence_adapter_previews_exact_rq3_ledger_and_bootstraps_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = rq3_post_boundary.preview_rq3_post_boundary_ledger()

    assert ledger.final_rq2_evidence_sha256 == (
        "a8f25319858f58f3f6e5cec2a51c513d697c478044ee9f9c5c355f7a471b7856"
    )
    assert ledger.selected_rq2_row_id == SELECTED_ROW_ID
    assert len(ledger.logical_rows) == 45
    assert len(ledger.physical_rows) == 38
    assert {
        row.reused_from for row in ledger.logical_rows if row.reused_from is not None
    } == RQ3_ELIGIBLE_REUSE_IDS
    ledger_path = persist_rq3_post_boundary_ledger(tmp_path / "rq3.json", ledger)
    row = ledger.rows[0]
    monkeypatch.setenv(
        rq3_post_boundary.JOB_ENVIRONMENT,
        rq3_post_boundary.encode_rq3_post_boundary_job(ledger, row.id),
    )
    monkeypatch.setenv(rq3_post_boundary.LEDGER_ENVIRONMENT, str(ledger_path))

    compiled, loaded, _, feature_path = (
        rq3_post_boundary.compiled_rq3_post_boundary_job_from_environment()
    )

    assert compiled.row_id == row.id
    assert loaded == ledger
    assert feature_path.is_file()
