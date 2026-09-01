import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.g3_pretrained_item_embeddings.analysis import rq4_results
from experiments.g3_pretrained_item_embeddings.launchers import (
    rq4_single_metadata_width256,
)
from experiments.g3_pretrained_item_embeddings.launchers.control import (
    PROJECT_ROOT,
    decode_control_job,
    encode_control_job,
)
from experiments.g3_pretrained_item_embeddings.protocol.rq4_single_metadata_width256_ledger import (
    RQ4_CAPACITY_EXTENSION_SELECTION_SHA256,
    RQ4_SINGLE_METADATA_WIDTH256_LEDGER_PATH,
    compile_rq4_single_metadata_width256_ledger,
    load_rq4_single_metadata_width256_ledger,
    persist_rq4_single_metadata_width256_ledger,
    validate_rq4_single_metadata_width256_ledger_document,
)


def test_width256_ledger_has_exactly_three_artist_and_three_album_rows() -> None:
    ledger = compile_rq4_single_metadata_width256_ledger(PROJECT_ROOT)

    assert ledger == compile_rq4_single_metadata_width256_ledger(PROJECT_ROOT)
    assert (
        ledger.capacity_extension_selection.logical_sha256
        == RQ4_CAPACITY_EXTENSION_SELECTION_SHA256
    )
    assert len(ledger.rows) == 6
    assert {row.family_id for row in ledger.rows} == {"rq4_artist", "rq4_album"}
    assert all("artist_album" not in row.family_id for row in ledger.rows)
    assert {row.family_id: sum(r.family_id == row.family_id for r in ledger.rows)
            for row in ledger.rows} == {"rq4_artist": 3, "rq4_album": 3}
    assert {row.to_dict()["representation"]["metadata_dim"] for row in ledger.rows} == {256}


def test_width256_ledger_uses_exact_approved_double_precision_coordinates() -> None:
    ledger = compile_rq4_single_metadata_width256_ledger(PROJECT_ROOT)
    factors = (math.sqrt(2.0), 2.0, 2.0 * math.sqrt(2.0))
    artist = [row for row in ledger.rows if row.family_id == "rq4_artist"]
    album = [row for row in ledger.rows if row.family_id == "rq4_album"]

    assert [(row.embedding_learning_rate, row.deep_learning_rate) for row in artist] == [
        (0.17783052497147875 * factor, 0.010430488535480936 / factor)
        for factor in factors
    ]
    assert [(row.embedding_learning_rate, row.deep_learning_rate) for row in album] == [
        (0.05753144041634071 / factor, 0.01852175330591617)
        for factor in factors
    ]


def test_width256_ledger_persistence_is_immutable(tmp_path: Path) -> None:
    ledger = compile_rq4_single_metadata_width256_ledger(PROJECT_ROOT)
    canonical = PROJECT_ROOT / RQ4_SINGLE_METADATA_WIDTH256_LEDGER_PATH

    persist_rq4_single_metadata_width256_ledger(canonical, ledger, root=PROJECT_ROOT)
    persist_rq4_single_metadata_width256_ledger(canonical, ledger, root=PROJECT_ROOT)
    loaded = load_rq4_single_metadata_width256_ledger(
        canonical, root=PROJECT_ROOT, expected_ledger_sha256=ledger.sha256
    )

    assert loaded == ledger
    with pytest.raises(ValueError, match="canonical project path"):
        persist_rq4_single_metadata_width256_ledger(
            tmp_path / "ledger.json", ledger, root=PROJECT_ROOT
        )


def test_width256_validator_rejects_numeric_type_drift() -> None:
    ledger = compile_rq4_single_metadata_width256_ledger(PROJECT_ROOT)
    document = ledger.to_dict()
    document["opportunity_accounting"]["logical_total"] = 6.0

    with pytest.raises(ValueError, match="ledger changed"):
        validate_rq4_single_metadata_width256_ledger_document(
            document,
            root=PROJECT_ROOT,
            expected_ledger_sha256=ledger.sha256,
        )


def test_width256_queue_has_status_new_six_enqueue_and_seal(tmp_path: Path) -> None:
    ledger = compile_rq4_single_metadata_width256_ledger(PROJECT_ROOT)
    commands = rq4_single_metadata_width256.compile_queue_surface(
        ledger_path=PROJECT_ROOT / RQ4_SINGLE_METADATA_WIDTH256_LEDGER_PATH,
        ledger=ledger,
        state_dir=tmp_path / "queue",
    )

    assert len(commands) == 9
    assert sum("enqueue-run" in command for command in commands) == 6
    assert {command[command.index("--run") + 1] for command in commands[2:8]} == {
        row.run_name for row in ledger.rows
    }


def test_width256_training_builder_uses_exact_ledger_coordinate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = compile_rq4_single_metadata_width256_ledger(PROJECT_ROOT)
    compiled = decode_control_job(encode_control_job(ledger, ledger.rows[0].id), ledger)
    captured = {}

    def fake_builder(**arguments):
        captured.update(arguments)
        return SimpleNamespace(base_path="unused")

    monkeypatch.setattr(
        rq4_single_metadata_width256, "build_g3_experiment", fake_builder
    )
    rq4_single_metadata_width256.build_training_experiment(
        compiled, ledger=ledger, feature_data_path=PROJECT_ROOT / "feature-data"
    )

    assert captured["embedding_learning_rate"] == compiled.job["training"][
        "embedding_learning_rate"
    ]
    assert captured["deep_learning_rate"] == compiled.job["training"][
        "deep_learning_rate"
    ]
    assert captured["lr_schedule_horizon_epochs"] == 25
    assert captured["representation"].metadata_dim == 256


def test_width256_results_collector_has_exact_stage_contract() -> None:
    ledger = compile_rq4_single_metadata_width256_ledger(PROJECT_ROOT)

    assert rq4_results._stage_contract(ledger) == {
        "job_environment": "G3_RQ4_SINGLE_METADATA_WIDTH256_JOB_B64",
        "ledger_environment": "G3_RQ4_SINGLE_METADATA_WIDTH256_LEDGER_PATH",
        "runner": "run_rq4_single_metadata_width256.py",
        "job_contract": "g3_rq4_single_metadata_width256_job.json",
    }
    identity = rq4_results._metadata_identity(PROJECT_ROOT, ledger)
    assert identity.feature_data_path == (
        "generated/g3_pretrained_item_embeddings/native-50m/"
        "item_features.parquet"
    )
