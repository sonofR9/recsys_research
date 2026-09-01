from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import pytest

from experiments.g3_pretrained_item_embeddings.launchers import rq5_mechanism
from experiments.g3_pretrained_item_embeddings.launchers.control import (
    encode_control_job,
)
from experiments.g3_pretrained_item_embeddings.protocol import rq5_mechanism_ledger


def _root() -> Path:
    return Path(__file__).resolve().parents[3]


def test_mechanism_ledger_has_exact_two_evidence_only_reproductions() -> None:
    root = _root()
    ledger = rq5_mechanism_ledger.compile_rq5_mechanism_ledger(root=root)
    commands = rq5_mechanism.compile_rq5_mechanism_queue_commands(
        ledger_path=root / rq5_mechanism_ledger.RQ5_MECHANISM_LEDGER_PATH,
        ledger=ledger,
        state_dir=root / "generated/training-queue-service",
    )

    assert ledger.sha256 == rq5_mechanism.RQ5_MECHANISM_LEDGER_LOGICAL_SHA256
    assert [row.id for row in ledger.rows] == [
        "rq5_gate_mechanism:01",
        "rq5_gate_mechanism:02",
    ]
    assert [row.source_row_id for row in ledger.rows] == [
        "rq5_global_gate:10",
        "rq5_frequency_gate:04",
    ]
    assert [row.content_gate for row in ledger.rows] == ["global", "frequency"]
    assert [row.gate_hidden_dim for row in ledger.rows] == [None, 8]
    assert [row.horizon_epochs for row in ledger.rows] == [40, 25]
    assert all(row.batch_size == 512 and row.seed == 42 for row in ledger.rows)
    assert all(row.to_dict()["selection_eligible"] is False for row in ledger.rows)
    assert len(commands) == 5
    assert sum("enqueue-run" in command for command in commands) == 2


def test_mechanism_artifact_contract_requires_gate_diagnostics() -> None:
    ledger = rq5_mechanism_ledger.compile_rq5_mechanism_ledger(root=_root())
    contracts = {
        contract["name"]: contract
        for contract in ledger.to_dict()["artifact_contracts"]
    }

    assert contracts["job_contract"]["filename"] == "g3_rq5_mechanism_job.json"
    assert contracts["gate_diagnostics"] == {
        "name": "gate_diagnostics",
        "filename": "g3_gate_diagnostics.json",
        "required_keys": [
            "schema_version",
            "frequency_terciles",
            "training_count_reference",
            "slice_membership_reference",
            "frequency_input_parity",
            "epochs",
        ],
        "schema_versions": [1],
    }


def test_mechanism_loader_rejects_stale_and_self_consistent_mutations(
    tmp_path: Path,
) -> None:
    root = _root()
    ledger = rq5_mechanism_ledger.compile_rq5_mechanism_ledger(root=root)
    stale = ledger.to_dict()
    stale["schema_version"] = True
    stale_path = tmp_path / "stale.json"
    stale_path.write_text(json.dumps(stale))
    with pytest.raises(ValueError, match="logical SHA changed"):
        rq5_mechanism_ledger.load_rq5_mechanism_ledger(
            stale_path,
            root=root,
            expected_ledger_sha256=ledger.sha256,
        )

    substituted = ledger.to_dict()
    substituted["rows"][0]["training"]["deep_learning_rate"] *= 0.5
    substituted["sha256"] = hashlib.sha256(
        json.dumps(
            {
                key: value
                for key, value in substituted.items()
                if key != "sha256"
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    substituted_path = tmp_path / "substituted.json"
    substituted_path.write_text(json.dumps(substituted))
    with pytest.raises(ValueError, match="frozen inputs"):
        rq5_mechanism_ledger.load_rq5_mechanism_ledger(
            substituted_path,
            root=root,
        )


@pytest.mark.parametrize(
    ("row_id", "gate", "width", "horizon", "embedding_lr", "deep_lr"),
    (
        (
            "rq5_gate_mechanism:01",
            "global",
            None,
            40,
            0.12305770976863895,
            0.011338899623382975,
        ),
        (
            "rq5_gate_mechanism:02",
            "frequency",
            8,
            25,
            0.11386115952375567,
            0.021533016497665633,
        ),
    ),
)
def test_mechanism_worker_maps_only_frozen_rows_to_opt_in_diagnostics(
    row_id: str,
    gate: str,
    width: int | None,
    horizon: int,
    embedding_lr: float,
    deep_lr: float,
) -> None:
    root = _root()
    ledger_path = root / rq5_mechanism_ledger.RQ5_MECHANISM_LEDGER_PATH
    ledger = rq5_mechanism_ledger.load_rq5_mechanism_ledger(
        ledger_path,
        root=root,
        expected_ledger_sha256=rq5_mechanism.RQ5_MECHANISM_LEDGER_LOGICAL_SHA256,
    )
    feature_path = rq5_mechanism_ledger.verify_rq5_mechanism_inputs(root, ledger)
    compiled = rq5_mechanism.decode_control_job(
        encode_control_job(ledger, row_id), ledger
    )

    experiment = rq5_mechanism.build_rq5_mechanism_training_experiment(
        compiled,
        ledger=ledger,
        feature_data_path=feature_path,
    )

    assert experiment.representation.content_gate == gate
    assert experiment.representation.gate_hidden_dim == width
    assert experiment.lr_schedule_horizon_epochs == horizon
    assert experiment.embedding_learning_rate == embedding_lr
    assert experiment.deep_learning_rate == deep_lr
    assert experiment.gate_mechanism_diagnostics is True


def test_mechanism_preview_does_not_accept_arbitrary_inputs() -> None:
    assert tuple(inspect.signature(rq5_mechanism.preview_rq5_mechanism_ledger).parameters) == (
        "root",
    )


def test_mechanism_worker_bootstrap_reloads_the_frozen_ledger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _root()
    ledger_path = root / rq5_mechanism_ledger.RQ5_MECHANISM_LEDGER_PATH
    ledger = rq5_mechanism_ledger.load_rq5_mechanism_ledger(
        ledger_path,
        root=root,
        expected_ledger_sha256=rq5_mechanism.RQ5_MECHANISM_LEDGER_LOGICAL_SHA256,
    )
    row = ledger.rows[1]
    monkeypatch.setenv(
        rq5_mechanism.JOB_ENVIRONMENT,
        encode_control_job(ledger, row.id),
    )
    monkeypatch.setenv(rq5_mechanism.LEDGER_ENVIRONMENT, str(ledger_path))

    compiled, loaded, loaded_path, feature_path = (
        rq5_mechanism.compiled_rq5_mechanism_job_from_environment(root=root)
    )

    assert compiled.row_id == row.id
    assert loaded == ledger
    assert loaded_path == ledger_path
    assert feature_path.is_file()
