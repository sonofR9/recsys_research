from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import pytest

from experiments.g3_pretrained_item_embeddings.protocol import rq4_horizon_ledger
from dcn.tests.experiments.test_g3_rq4_initial_infrastructure import (
    _persisted_ledger,
)
from experiments.g3_pretrained_item_embeddings.protocol.constants import (
    APPROVED_PROTOCOL_SHA256,
)
from experiments.g3_pretrained_item_embeddings.protocol.rq4 import (
    RQ4_METADATA_FAMILIES,
)
from experiments.g3_pretrained_item_embeddings.protocol.rq4_horizon_ledger import (
    CONTROL_CALIBRATION_PATH,
    compile_rq4_horizon_ledger,
    load_rq4_horizon_ledger,
    persist_rq4_horizon_ledger,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _canonical_sha(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _selection(
    root: Path,
    *,
    initial_path: Path,
    initial_sha: str,
    capacity: int = 32,
) -> tuple[Path, str]:
    initial_reference = {
        "path": str(initial_path.relative_to(root)),
        "sha256": hashlib.sha256(initial_path.read_bytes()).hexdigest(),
        "size_bytes": initial_path.stat().st_size,
        "logical_sha256": initial_sha,
    }
    direction = "lower" if capacity == 16 else "upper" if capacity == 64 else None
    selected_rows = {
        family: next(
            row
            for row in json.loads(initial_path.read_text())["rows"]
            if row["family_id"] == family and row["representation"]["metadata_dim"] == capacity
        )
        for family in RQ4_METADATA_FAMILIES
    }
    payload = {
        "schema_version": 1,
        "kind": "g3_rq4_metadata_capacity_selection_native50m",
        "protocol_sha256": APPROVED_PROTOCOL_SHA256,
        "initial_ledger": initial_reference,
        "opportunity_accounting": {
            "families": 3,
            "opportunities_per_family": 9,
            "total_opportunities": 27,
        },
        "selection_rule": (
            "validation Recall@100, validation NDCG@100, lower queue wall time, "
            "then row ID"
        ),
        "family_selections": {
            family: {
                "selected": {
                    "row_id": selected_rows[family]["id"],
                    "family_id": family,
                    "ledger_sha256": initial_sha,
                    "job": selected_rows[family],
                    "metadata_dim": capacity,
                    "embedding_learning_rate": selected_rows[family]["training"][
                        "embedding_learning_rate"
                    ],
                    "deep_learning_rate": selected_rows[family]["training"][
                        "deep_learning_rate"
                    ],
                    "horizon_epochs": selected_rows[family]["training"][
                        "horizon_epochs"
                    ],
                },
                "selected_row_id": selected_rows[family]["id"],
                "selected_metadata_dim": capacity,
                "selected_embedding_learning_rate": selected_rows[family]["training"][
                    "embedding_learning_rate"
                ],
                "selected_deep_learning_rate": selected_rows[family]["training"][
                    "deep_learning_rate"
                ],
                "selected_horizon_epochs": selected_rows[family]["training"][
                    "horizon_epochs"
                ],
                "capacity_boundary": {
                    "direction": direction,
                    "extension_capacity": (
                        capacity // 2 if direction == "lower"
                        else capacity * 2 if direction == "upper"
                        else None
                    ),
                },
            }
            for family in RQ4_METADATA_FAMILIES
        },
        "capacity_extensions_required": (
            [] if direction is None else list(RQ4_METADATA_FAMILIES)
        ),
    }
    document = payload | {"sha256": _canonical_sha(payload)}
    path = root / "capacity_selection.json"
    path.write_text(json.dumps(document, sort_keys=True, separators=(",", ":")))
    return path, str(document["sha256"])


def _copy_calibration(root: Path) -> None:
    source = PROJECT_ROOT / CONTROL_CALIBRATION_PATH
    destination = root / CONTROL_CALIBRATION_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def test_horizon_ledger_adds_three_opportunities_per_family(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        rq4_horizon_ledger,
        "require_rq4_horizon_materialization_approval",
        lambda root: None,
    )
    initial_path, initial, rq3_sha, row_id = _persisted_ledger(tmp_path, monkeypatch)
    selection_path, selection_sha = _selection(
        tmp_path, initial_path=initial_path, initial_sha=initial.sha256
    )
    _copy_calibration(tmp_path)

    ledger = compile_rq4_horizon_ledger(
        root=tmp_path,
        initial_ledger_path=initial_path,
        expected_initial_ledger_sha256=initial.sha256,
        capacity_selection_path=selection_path,
        expected_capacity_selection_sha256=selection_sha,
        expected_rq3_sha256=rq3_sha,
        expected_rq3_row_id=row_id,
    )

    assert len(ledger.rows) == len(ledger.physical_rows) == 9
    for family in RQ4_METADATA_FAMILIES:
        rows = [row for row in ledger.rows if row.family_id == family]
        assert [row.horizon_epochs for row in rows] == [15, 25, 40]
        assert {row.metadata_dim for row in rows} == {32}
    path = persist_rq4_horizon_ledger(
        tmp_path / "horizon.json", ledger, root=tmp_path
    )
    assert load_rq4_horizon_ledger(
        path,
        root=tmp_path,
        expected_ledger_sha256=ledger.sha256,
        expected_rq3_sha256=rq3_sha,
        expected_rq3_row_id=row_id,
    ) == ledger


def test_horizon_ledger_blocks_unresolved_capacity_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        rq4_horizon_ledger,
        "require_rq4_horizon_materialization_approval",
        lambda root: None,
    )
    initial_path, initial, rq3_sha, row_id = _persisted_ledger(tmp_path, monkeypatch)
    selection_path, selection_sha = _selection(
        tmp_path,
        initial_path=initial_path,
        initial_sha=initial.sha256,
        capacity=64,
    )
    _copy_calibration(tmp_path)

    with pytest.raises(ValueError, match="boundary-unresolved"):
        compile_rq4_horizon_ledger(
            root=tmp_path,
            initial_ledger_path=initial_path,
            expected_initial_ledger_sha256=initial.sha256,
            capacity_selection_path=selection_path,
            expected_capacity_selection_sha256=selection_sha,
            expected_rq3_sha256=rq3_sha,
            expected_rq3_row_id=row_id,
        )


def test_horizon_compile_and_persist_require_gate_when_both_inputs_are_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValueError, match="requires its canonical approval gate"):
        compile_rq4_horizon_ledger(
            root=tmp_path,
            initial_ledger_path=tmp_path / "missing-initial.json",
            expected_initial_ledger_sha256="missing",
            capacity_selection_path=tmp_path / "missing-capacity.json",
            expected_capacity_selection_sha256="missing",
            expected_rq3_sha256="missing",
            expected_rq3_row_id="missing",
        )

    with monkeypatch.context() as gate_bypass:
        gate_bypass.setattr(
            rq4_horizon_ledger,
            "require_rq4_horizon_materialization_approval",
            lambda root: None,
        )
        initial_path, initial, rq3_sha, row_id = _persisted_ledger(
            tmp_path, monkeypatch
        )
        selection_path, selection_sha = _selection(
            tmp_path, initial_path=initial_path, initial_sha=initial.sha256
        )
        _copy_calibration(tmp_path)
        ledger = compile_rq4_horizon_ledger(
            root=tmp_path,
            initial_ledger_path=initial_path,
            expected_initial_ledger_sha256=initial.sha256,
            capacity_selection_path=selection_path,
            expected_capacity_selection_sha256=selection_sha,
            expected_rq3_sha256=rq3_sha,
            expected_rq3_row_id=row_id,
        )

    with pytest.raises(ValueError, match="requires its canonical approval gate"):
        persist_rq4_horizon_ledger(
            tmp_path / "horizon.json", ledger, root=tmp_path
        )
