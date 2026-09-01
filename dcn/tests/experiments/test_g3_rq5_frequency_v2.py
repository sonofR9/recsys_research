from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments.g3_pretrained_item_embeddings.launchers import rq5_frequency_v2
from experiments.g3_pretrained_item_embeddings.launchers import (
    rq5_frequency_v2_horizon,
)
from experiments.g3_pretrained_item_embeddings.launchers.control import (
    decode_control_job,
    encode_control_job,
)
from experiments.g3_pretrained_item_embeddings.protocol import (
    rq5_frequency_v2_ledger,
    rq5_frequency_v2_horizon_ledger,
)


def _root() -> Path:
    return Path(__file__).resolve().parents[3]


def test_frequency_v2_ledger_rebuilds_exact_nine_cell_surface() -> None:
    root = _root()
    ledger = rq5_frequency_v2_ledger.compile_rq5_frequency_v2_ledger(root=root)
    commands = rq5_frequency_v2.compile_rq5_frequency_v2_queue_commands(
        ledger_path=root / rq5_frequency_v2_ledger.RQ5_FREQUENCY_V2_LEDGER_PATH,
        ledger=ledger,
        state_dir=root / "generated/training-queue-service",
    )

    assert ledger.sha256 == rq5_frequency_v2.RQ5_FREQUENCY_V2_LEDGER_LOGICAL_SHA256
    assert [row.gate_hidden_dim for row in ledger.rows] == [
        4,
        4,
        4,
        8,
        8,
        8,
        16,
        16,
        16,
    ]
    assert all(row.horizon_epochs == 25 for row in ledger.rows)
    assert len(commands) == 12
    assert sum("enqueue-run" in command for command in commands) == 9
    payload = ledger.to_dict()
    assert payload["defect_resolution"]["reader_and_tuning_eligible"] is False
    assert payload["defect_resolution"]["corrected_semantics"] == "fp32_p09_v2"
    assert payload["opportunity_accounting"]["deferred_selected_width_horizons"] == [
        15,
        25,
        40,
    ]


def test_frequency_v2_worker_maps_frozen_row_to_corrected_semantics() -> None:
    root = _root()
    path = root / rq5_frequency_v2_ledger.RQ5_FREQUENCY_V2_LEDGER_PATH
    ledger = rq5_frequency_v2_ledger.load_rq5_frequency_v2_ledger(
        path,
        root=root,
        expected_ledger_sha256=rq5_frequency_v2.RQ5_FREQUENCY_V2_LEDGER_LOGICAL_SHA256,
    )
    row = ledger.rows[4]
    feature_path = rq5_frequency_v2_ledger.verify_rq5_frequency_v2_inputs(
        root, ledger
    )
    compiled = decode_control_job(encode_control_job(ledger, row.id), ledger)

    experiment = rq5_frequency_v2.build_rq5_frequency_v2_training_experiment(
        compiled,
        ledger=ledger,
        feature_data_path=feature_path,
    )

    assert experiment.representation.frequency_gate_semantics == "fp32_p09_v2"
    assert experiment.gate_mechanism_diagnostics is True


def test_frequency_v2_loader_rejects_stale_and_self_consistent_mutations(
    tmp_path: Path,
) -> None:
    root = _root()
    ledger = rq5_frequency_v2_ledger.compile_rq5_frequency_v2_ledger(root=root)
    stale = ledger.to_dict()
    stale["schema_version"] = True
    stale_path = tmp_path / "stale.json"
    stale_path.write_text(json.dumps(stale))
    with pytest.raises(ValueError, match="logical SHA changed"):
        rq5_frequency_v2_ledger.load_rq5_frequency_v2_ledger(
            stale_path,
            root=root,
            expected_ledger_sha256=ledger.sha256,
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
        rq5_frequency_v2_ledger.load_rq5_frequency_v2_ledger(
            substituted_path,
            root=root,
        )


def test_frequency_v2_binds_exact_parity_and_a100_precision_proof() -> None:
    ledger = rq5_frequency_v2_ledger.compile_rq5_frequency_v2_ledger(root=_root())
    artifacts = dict(ledger.defect_artifacts)

    assert len(artifacts) == 14
    assert {name for name in artifacts if name.endswith("gate_diagnostics")} == {
        "global_gate_diagnostics",
        "frequency_gate_diagnostics",
    }
    assert ledger.precision_probe.logical_sha256 == (
        rq5_frequency_v2_ledger.RQ5_PRECISION_PROBE_LOGICAL_SHA256
    )


def test_frequency_v2_horizon_ledger_has_exact_three_selected_width_cells() -> None:
    root = _root()
    ledger = (
        rq5_frequency_v2_horizon_ledger.compile_rq5_frequency_v2_horizon_ledger(
            root=root
        )
    )
    commands = rq5_frequency_v2_horizon.compile_rq5_frequency_v2_horizon_queue_commands(
        ledger_path=(
            root
            / rq5_frequency_v2_horizon_ledger.RQ5_FREQUENCY_V2_HORIZON_LEDGER_PATH
        ),
        ledger=ledger,
        state_dir=root / "generated/training-queue-service",
    )

    assert ledger.sha256 == (
        rq5_frequency_v2_horizon.RQ5_FREQUENCY_V2_HORIZON_LEDGER_LOGICAL_SHA256
    )
    assert [row.id for row in ledger.rows] == [
        "rq5_frequency_gate_v2:10",
        "rq5_frequency_gate_v2:11",
        "rq5_frequency_gate_v2:12",
    ]
    assert [row.horizon_epochs for row in ledger.rows] == [15, 25, 40]
    assert all(row.gate_hidden_dim == 8 for row in ledger.rows)
    assert len(commands) == 6
    assert sum("enqueue-run" in command for command in commands) == 3


def test_frequency_v2_horizon_worker_preserves_corrected_semantics() -> None:
    root = _root()
    ledger = (
        rq5_frequency_v2_horizon_ledger.compile_rq5_frequency_v2_horizon_ledger(
            root=root
        )
    )
    feature = rq5_frequency_v2_horizon_ledger.verify_rq5_frequency_v2_horizon_inputs(
        root, ledger
    )
    row = ledger.rows[2]
    compiled = decode_control_job(encode_control_job(ledger, row.id), ledger)

    experiment = (
        rq5_frequency_v2_horizon.build_rq5_frequency_v2_horizon_training_experiment(
            compiled,
            ledger=ledger,
            feature_data_path=feature,
        )
    )

    assert experiment.lr_schedule_horizon_epochs == 40
    assert experiment.representation.frequency_gate_semantics == "fp32_p09_v2"
    assert experiment.gate_mechanism_diagnostics is True
