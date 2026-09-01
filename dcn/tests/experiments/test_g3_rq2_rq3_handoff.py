import copy
import hashlib
import json
from pathlib import Path
import shutil
import uuid

import pytest

from experiments.g3_pretrained_item_embeddings.analysis import rq2_rq3_handoff
from experiments.g3_pretrained_item_embeddings.analysis.rq2_content_deep_lr_boundary_results import (
    RQ2_CONTENT_DEEP_LR_BOUNDARY_EVIDENCE_PATH,
)
from experiments.g3_pretrained_item_embeddings.protocol.rq3 import (
    compile_rq3_output_surface,
)


ROOT = Path(__file__).resolve().parents[3]


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _actual_evidence() -> tuple[Path, dict[str, object]]:
    path = ROOT / RQ2_CONTENT_DEEP_LR_BOUNDARY_EVIDENCE_PATH
    return path, json.loads(path.read_text())


@pytest.fixture
def handoff_directory():
    root = ROOT / "generated" / f"test-rq2-rq3-handoff-{uuid.uuid4().hex}"
    root.mkdir(parents=True)
    try:
        yield root
    finally:
        shutil.rmtree(root)


def test_materialized_actual_rq2_handoff_compiles_and_loads_in_rq3(
    handoff_directory: Path,
    monkeypatch,
) -> None:
    evidence_path, evidence = _actual_evidence()
    verification_calls = []

    def verify(path: Path, *, root: Path):
        verification_calls.append((path, root))
        return evidence

    monkeypatch.setattr(
        rq2_rq3_handoff,
        "verify_rq2_content_deep_lr_boundary_evidence",
        verify,
    )
    bridge_path = handoff_directory / "rq2_rq3_reuse.json"
    selection_path = handoff_directory / "rq2_selection.json"

    bridge, selection = rq2_rq3_handoff.persist_rq2_rq3_handoff(
        root=ROOT,
        final_evidence_path=evidence_path,
        expected_final_evidence_sha256=evidence["sha256"],
        bridge_path=bridge_path,
        selection_path=selection_path,
    )
    loaded_bridge, loaded_selection = rq2_rq3_handoff.load_rq2_rq3_handoff(
        root=ROOT,
        final_evidence_path=evidence_path,
        expected_final_evidence_sha256=evidence["sha256"],
        bridge_path=bridge_path,
        selection_path=selection_path,
    )
    with pytest.raises(ValueError, match="frozen final evidence binding"):
        compile_rq3_output_surface(
            root=ROOT,
            selection_path=selection_path,
            expected_selection_sha256=selection["sha256"],
        )
    surface = compile_rq3_output_surface(
        root=ROOT,
        selection_path=selection_path,
        expected_selection_sha256=selection["sha256"],
        expected_final_rq2_evidence_sha256=evidence["sha256"],
    )

    assert loaded_bridge == bridge
    assert loaded_selection == selection
    assert len(bridge["all_tuning_ledger"]) == 30
    assert bridge["all_tuning_ledger"] == evidence["all_tuning_ledger"]
    assert len(bridge["tuning_ledger"]) == 9
    assert all(row["selection_resolved"] is True for row in bridge["tuning_ledger"])
    assert selection["kind"] == "g3_rq2_content_selection_for_rq3"
    assert selection["selection_resolved"] is True
    assert selection["selected_history_hidden_dim"] == 32
    assert len(selection["rows"]) == 9
    assert surface.selected_history_hidden_dim == 32
    assert {
        row.authenticated_source.source_id
        for row in surface.rows_by_family["rq3_output_learned"]
        if row.authenticated_source is not None
    } == {f"rq2_content_concat:{index:02d}" for index in range(10, 19)}
    assert verification_calls


def test_handoff_rejects_unresolved_final_rq2_selection(monkeypatch) -> None:
    evidence_path, evidence = _actual_evidence()
    unresolved = copy.deepcopy(evidence)
    unresolved["final_content_selection"]["status"] = (
        "pending_renewed_user_approval"
    )
    unresolved["final_content_selection"]["provisional_selected"] = unresolved[
        "final_content_selection"
    ]["selected"]
    unresolved["final_content_selection"]["selected"] = None
    unresolved["rq3_inputs"] = None
    unresolved["final_rq2_comparison"] = None
    monkeypatch.setattr(
        rq2_rq3_handoff,
        "verify_rq2_content_deep_lr_boundary_evidence",
        lambda *args, **kwargs: unresolved,
    )

    with pytest.raises(ValueError, match="not resolved"):
        rq2_rq3_handoff.build_rq2_rq3_handoff(
            root=ROOT,
            final_evidence_path=evidence_path,
            expected_final_evidence_sha256=evidence["sha256"],
            bridge_path=ROOT / "generated/not-written-bridge.json",
            selection_path=ROOT / "generated/not-written-selection.json",
        )


def test_handoff_accepts_an_explicit_authenticated_source_adapter(
    monkeypatch,
) -> None:
    evidence_path, evidence = _actual_evidence()
    monkeypatch.setattr(
        rq2_rq3_handoff,
        "verify_rq2_content_deep_lr_boundary_evidence",
        lambda *args, **kwargs: evidence,
    )
    calls = []

    def adapter(root: Path, path: Path, expected_sha256: str):
        calls.append((root, path, expected_sha256))
        return rq2_rq3_handoff.authenticate_boundary_rq2_handoff_source(
            root,
            path,
            expected_sha256,
        )

    bridge, selection = rq2_rq3_handoff.build_rq2_rq3_handoff(
        root=ROOT,
        final_evidence_path=evidence_path,
        expected_final_evidence_sha256=evidence["sha256"],
        bridge_path=ROOT / "generated/not-written-bridge.json",
        selection_path=ROOT / "generated/not-written-selection.json",
        source_adapter=adapter,
    )

    assert calls == [(ROOT, evidence_path, evidence["sha256"])]
    assert bridge["final_rq2_evidence"]["logical_sha256"] == evidence["sha256"]
    assert selection["selected_history_hidden_dim"] == 32


def test_handoff_load_rejects_a_rehashed_copied_claim(
    handoff_directory: Path,
    monkeypatch,
) -> None:
    evidence_path, evidence = _actual_evidence()
    monkeypatch.setattr(
        rq2_rq3_handoff,
        "verify_rq2_content_deep_lr_boundary_evidence",
        lambda *args, **kwargs: evidence,
    )
    bridge_path = handoff_directory / "rq2_rq3_reuse.json"
    selection_path = handoff_directory / "rq2_selection.json"
    rq2_rq3_handoff.persist_rq2_rq3_handoff(
        root=ROOT,
        final_evidence_path=evidence_path,
        expected_final_evidence_sha256=evidence["sha256"],
        bridge_path=bridge_path,
        selection_path=selection_path,
    )
    selection = json.loads(selection_path.read_text())
    selection["rows"][0]["deep_learning_rate"] = 1.0
    selection.pop("sha256")
    selection["sha256"] = hashlib.sha256(_canonical(selection).encode()).hexdigest()
    selection_path.write_text(_canonical(selection) + "\n")

    with pytest.raises(ValueError, match="differs from authenticated RQ2"):
        rq2_rq3_handoff.load_rq2_rq3_handoff(
            root=ROOT,
            final_evidence_path=evidence_path,
            expected_final_evidence_sha256=evidence["sha256"],
            bridge_path=bridge_path,
            selection_path=selection_path,
        )


def test_rq3_rejects_a_fully_rehashed_truncated_evidence_chain(
    handoff_directory: Path,
    monkeypatch,
) -> None:
    evidence_path, evidence = _actual_evidence()
    monkeypatch.setattr(
        rq2_rq3_handoff,
        "verify_rq2_content_deep_lr_boundary_evidence",
        lambda *args, **kwargs: evidence,
    )
    bridge_path = handoff_directory / "rq2_rq3_reuse.json"
    selection_path = handoff_directory / "rq2_selection.json"
    rq2_rq3_handoff.persist_rq2_rq3_handoff(
        root=ROOT,
        final_evidence_path=evidence_path,
        expected_final_evidence_sha256=evidence["sha256"],
        bridge_path=bridge_path,
        selection_path=selection_path,
    )
    bridge = json.loads(bridge_path.read_text())
    chosen = bridge["tuning_ledger"][0]
    chosen_without_resolution = {
        name: value for name, value in chosen.items() if name != "selection_resolved"
    }
    forged_final = copy.deepcopy(evidence)
    forged_final["all_tuning_ledger"] = [
        row
        for row in forged_final["all_tuning_ledger"]
        if row.get("family_id") != "rq2_content_concat"
        or row.get("capacity") != 32
        or row.get("row_id") == chosen["row_id"]
    ]
    forged_final["final_content_selection"]["selected"] = chosen_without_resolution
    forged_final["final_content_selection"]["provisional_selected"] = None
    forged_final["final_content_selection"]["status"] = "resolved"
    forged_final["rq3_inputs"]["selected_content_input"] = chosen_without_resolution
    forged_final["rq3_inputs"]["reusable_width_32_content_rows"] = [
        chosen_without_resolution
    ]
    forged_final.pop("sha256")
    forged_final["sha256"] = hashlib.sha256(
        _canonical(forged_final).encode()
    ).hexdigest()
    forged_final_path = handoff_directory / "forged_final_rq2.json"
    forged_final_path.write_text(_canonical(forged_final) + "\n")
    bridge["final_rq2_evidence"] = {
        "path": str(forged_final_path.relative_to(ROOT)),
        "sha256": hashlib.sha256(forged_final_path.read_bytes()).hexdigest(),
        "size_bytes": forged_final_path.stat().st_size,
        "logical_sha256": forged_final["sha256"],
    }
    bridge["all_tuning_ledger"] = forged_final["all_tuning_ledger"]
    bridge["tuning_ledger"] = [chosen]
    bridge["selected_row_id"] = chosen["row_id"]
    bridge.pop("sha256")
    bridge["sha256"] = hashlib.sha256(_canonical(bridge).encode()).hexdigest()
    bridge_path.write_text(_canonical(bridge) + "\n")
    bridge_reference = {
        "path": str(bridge_path.relative_to(ROOT)),
        "sha256": hashlib.sha256(bridge_path.read_bytes()).hexdigest(),
        "size_bytes": bridge_path.stat().st_size,
        "logical_sha256": bridge["sha256"],
    }
    selection = json.loads(selection_path.read_text())
    selection["source_evidence"] = [bridge_reference]
    selection["rows"] = selection["rows"][:1]
    selection["rows"][0]["source_evidence"] = bridge_reference
    selection.pop("sha256")
    selection["sha256"] = hashlib.sha256(_canonical(selection).encode()).hexdigest()
    selection_path.write_text(_canonical(selection) + "\n")

    with pytest.raises(ValueError, match="frozen final evidence binding"):
        compile_rq3_output_surface(
            root=ROOT,
            selection_path=selection_path,
            expected_selection_sha256=selection["sha256"],
            expected_final_rq2_evidence_sha256=evidence["sha256"],
        )


def test_rq3_rejects_an_unrecognized_reusable_evidence_kind(
    handoff_directory: Path,
    monkeypatch,
) -> None:
    evidence_path, evidence = _actual_evidence()
    monkeypatch.setattr(
        rq2_rq3_handoff,
        "verify_rq2_content_deep_lr_boundary_evidence",
        lambda *args, **kwargs: evidence,
    )
    bridge_path = handoff_directory / "rq2_rq3_reuse.json"
    selection_path = handoff_directory / "rq2_selection.json"
    rq2_rq3_handoff.persist_rq2_rq3_handoff(
        root=ROOT,
        final_evidence_path=evidence_path,
        expected_final_evidence_sha256=evidence["sha256"],
        bridge_path=bridge_path,
        selection_path=selection_path,
    )
    bridge = json.loads(bridge_path.read_text())
    bridge["kind"] = "unrecognized_reusable_evidence"
    bridge["tuning_ledger"] = bridge["tuning_ledger"][:1]
    bridge.pop("sha256")
    bridge["sha256"] = hashlib.sha256(_canonical(bridge).encode()).hexdigest()
    bridge_path.write_text(_canonical(bridge) + "\n")
    bridge_reference = {
        "path": str(bridge_path.relative_to(ROOT)),
        "sha256": hashlib.sha256(bridge_path.read_bytes()).hexdigest(),
        "size_bytes": bridge_path.stat().st_size,
        "logical_sha256": bridge["sha256"],
    }
    selection = json.loads(selection_path.read_text())
    selection["source_evidence"] = [bridge_reference]
    selection["rows"] = selection["rows"][:1]
    selection["rows"][0]["source_evidence"] = bridge_reference
    selection.pop("sha256")
    selection["sha256"] = hashlib.sha256(_canonical(selection).encode()).hexdigest()
    selection_path.write_text(_canonical(selection) + "\n")

    with pytest.raises(ValueError, match="kind is unsupported"):
        compile_rq3_output_surface(
            root=ROOT,
            selection_path=selection_path,
            expected_selection_sha256=selection["sha256"],
            expected_final_rq2_evidence_sha256=evidence["sha256"],
        )
