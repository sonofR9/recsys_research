from pathlib import Path
import json
import shutil

import pytest

from experiments.g3_pretrained_item_embeddings.protocol.rq4_followup_gate import (
    RQ4_FOLLOWUP_GATE_PATH,
    RQ4_WIDTH256_SELECTION_PATH,
    compile_rq4_followup_gate,
    persist_rq4_followup_gate,
    require_rq4_further_capacity_width_approval,
    require_rq4_horizon_materialization_approval,
)
from experiments.g3_pretrained_item_embeddings.protocol.rq4_horizon_ledger import (
    compile_rq4_horizon_ledger,
)
from experiments.g3_pretrained_item_embeddings.protocol.rq4_next_capacity_ledger import (
    compile_rq4_next_capacity_ledger,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _blocked_root(tmp_path: Path) -> Path:
    selection = tmp_path / RQ4_WIDTH256_SELECTION_PATH
    selection.parent.mkdir(parents=True)
    shutil.copyfile(PROJECT_ROOT / RQ4_WIDTH256_SELECTION_PATH, selection)
    gate = compile_rq4_followup_gate(tmp_path)
    persist_rq4_followup_gate(
        tmp_path / RQ4_FOLLOWUP_GATE_PATH, gate, root=tmp_path
    )
    return tmp_path


def test_followup_gate_binds_evidence_and_blocks_horizon(tmp_path: Path) -> None:
    root = _blocked_root(tmp_path)
    gate = compile_rq4_followup_gate(root)

    assert gate["approval_state"] == {
        "further_capacity_width_approved": False,
        "horizon_materialization_approved": False,
    }
    with pytest.raises(ValueError, match="requires renewed user approval"):
        require_rq4_horizon_materialization_approval(root)
    with pytest.raises(ValueError, match="requires renewed user approval"):
        require_rq4_further_capacity_width_approval(root)


def test_horizon_compiler_fails_before_reading_any_old_inputs(tmp_path: Path) -> None:
    root = _blocked_root(tmp_path)

    with pytest.raises(ValueError, match="requires renewed user approval"):
        compile_rq4_horizon_ledger(
            root=root,
            initial_ledger_path=root / "missing-initial.json",
            expected_initial_ledger_sha256="missing",
            capacity_selection_path=root / "missing-capacity.json",
            expected_capacity_selection_sha256="missing",
            expected_rq3_sha256="missing",
            expected_rq3_row_id="missing",
        )


def test_missing_gate_fails_closed_when_width256_evidence_exists(
    tmp_path: Path,
) -> None:
    selection = tmp_path / RQ4_WIDTH256_SELECTION_PATH
    selection.parent.mkdir(parents=True)
    shutil.copyfile(PROJECT_ROOT / RQ4_WIDTH256_SELECTION_PATH, selection)

    with pytest.raises(ValueError, match="requires its canonical"):
        require_rq4_horizon_materialization_approval(tmp_path)
    with pytest.raises(ValueError, match="requires its canonical"):
        require_rq4_further_capacity_width_approval(tmp_path)


def test_missing_gate_and_missing_evidence_still_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires its canonical"):
        require_rq4_horizon_materialization_approval(tmp_path)
    with pytest.raises(ValueError, match="requires its canonical"):
        require_rq4_further_capacity_width_approval(tmp_path)


def test_next_capacity_compiler_enforces_followup_gate(tmp_path: Path) -> None:
    root = _blocked_root(tmp_path)

    with pytest.raises(ValueError, match="requires renewed user approval"):
        compile_rq4_next_capacity_ledger(root=root)


def test_gate_rejects_noncanonical_byte_reserialization(tmp_path: Path) -> None:
    root = _blocked_root(tmp_path)
    gate_path = root / RQ4_FOLLOWUP_GATE_PATH
    gate_path.write_text(json.dumps(json.loads(gate_path.read_text()), indent=2))

    with pytest.raises(ValueError, match="exact canonical bytes"):
        require_rq4_horizon_materialization_approval(root)


def test_gate_rejects_symlink_at_canonical_path(tmp_path: Path) -> None:
    root = _blocked_root(tmp_path)
    gate_path = root / RQ4_FOLLOWUP_GATE_PATH
    target = tmp_path / "copied-gate.json"
    target.write_bytes(gate_path.read_bytes())
    gate_path.unlink()
    gate_path.symlink_to(target)

    with pytest.raises(ValueError, match="must not be a symlink"):
        require_rq4_horizon_materialization_approval(root)


def test_gate_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    root = _blocked_root(tmp_path)
    gate_path = root / RQ4_FOLLOWUP_GATE_PATH
    content = gate_path.read_text().replace(
        '"schema_version":1', '"schema_version":1,"schema_version":1'
    )
    gate_path.write_text(content)

    with pytest.raises(ValueError, match="duplicate JSON key"):
        require_rq4_horizon_materialization_approval(root)


def test_gate_persistence_rejects_noncanonical_path(tmp_path: Path) -> None:
    selection = tmp_path / RQ4_WIDTH256_SELECTION_PATH
    selection.parent.mkdir(parents=True)
    shutil.copyfile(PROJECT_ROOT / RQ4_WIDTH256_SELECTION_PATH, selection)
    gate = compile_rq4_followup_gate(tmp_path)

    with pytest.raises(ValueError, match="canonical project path"):
        persist_rq4_followup_gate(
            tmp_path / "copied-gate.json", gate, root=tmp_path
        )
