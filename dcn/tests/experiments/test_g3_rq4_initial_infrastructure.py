from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from dcn.tests.experiments.test_g3_rq4_protocol import _predecessors
from experiments.g3_pretrained_item_embeddings.launchers import rq4_initial
from experiments.g3_pretrained_item_embeddings.launchers.control import (
    encode_control_job,
)
from experiments.g3_pretrained_item_embeddings.protocol.rq4 import (
    RQ4_METADATA_FAMILIES,
)
from experiments.g3_pretrained_item_embeddings.protocol.rq4_initial_ledger import (
    RQ3_FINAL_EVIDENCE_FILE_SHA256,
    RQ3_FINAL_EVIDENCE_LOGICAL_SHA256,
    RQ3_FINAL_EVIDENCE_SIZE_BYTES,
    RQ3_FINAL_SELECTED_ROW_ID,
    RQ4_INITIAL_LEDGER_LOGICAL_SHA256,
    compile_rq4_initial_ledger,
    persist_rq4_initial_ledger,
    validate_rq4_initial_ledger_document,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_canonical_initial_preview_binds_final_rq3_scientific_selection() -> None:
    ledger = compile_rq4_initial_ledger(root=PROJECT_ROOT)

    assert ledger.sha256 == RQ4_INITIAL_LEDGER_LOGICAL_SHA256
    assert ledger.expected_rq3_row_id == RQ3_FINAL_SELECTED_ROW_ID
    assert ledger.rq3_final_evidence.logical_sha256 == RQ3_FINAL_EVIDENCE_LOGICAL_SHA256
    assert ledger.rq3_final_evidence.sha256 == RQ3_FINAL_EVIDENCE_FILE_SHA256
    assert ledger.rq3_final_evidence.size_bytes == RQ3_FINAL_EVIDENCE_SIZE_BYTES
    assert len(ledger.rows) == 27
    assert {row.catalog_representation for row in ledger.rows} == {
        "id_frozen_content"
    }
    assert {row.history_hidden_dim for row in ledger.rows} == {128}


def _ledger(root: Path, monkeypatch: pytest.MonkeyPatch):
    rq2_path, rq2_sha, rq3_path, rq3_sha = _predecessors(root, monkeypatch)
    selected_row = json.loads(rq3_path.read_text())["selected"]["row_id"]
    ledger = compile_rq4_initial_ledger(
        root=root,
        rq2_final_path=rq2_path,
        expected_rq2_sha256=rq2_sha,
        rq3_final_path=rq3_path,
        expected_rq3_sha256=rq3_sha,
        expected_rq3_row_id=selected_row,
    )
    return ledger, rq3_sha, selected_row


def _persisted_ledger(root: Path, monkeypatch: pytest.MonkeyPatch):
    ledger, rq3_sha, selected_row = _ledger(root, monkeypatch)
    path = root / "rq4.json"
    persist_rq4_initial_ledger(
        path,
        ledger,
        root=root,
        expected_rq3_sha256=rq3_sha,
        expected_rq3_row_id=selected_row,
    )
    return path, ledger, rq3_sha, selected_row


def test_initial_ledger_has_equal_twelve_opportunity_family_budgets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger, _, _ = _ledger(tmp_path, monkeypatch)

    assert ledger.family_opportunity_budgets == {
        family: 12 for family in RQ4_METADATA_FAMILIES
    }
    assert len(ledger.rows) == ledger.stage_physical_jobs == 27
    for family in RQ4_METADATA_FAMILIES:
        rows = [row for row in ledger.rows if row.family_id == family]
        assert len(rows) == 9
        assert {row.metadata_dim for row in rows} == {16, 32, 64}
    assert ledger.deferred_stages["metadata_horizon_followup"] == {
        "logical_opportunities_per_family": 3,
        "materialize_only_after_capacity_results": True,
    }
    assert ledger.deferred_stages["parameter_matched_extra_item_id"] == {
        "logical_opportunities": 12,
        "materialize_only_after_metadata_winner": True,
        "maximum_parameter_mismatch_fraction": 0.01,
    }


def test_initial_ledger_requires_exact_injected_rq3_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rq2_path, rq2_sha, rq3_path, rq3_sha = _predecessors(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="representation differs"):
        compile_rq4_initial_ledger(
            root=tmp_path,
            rq2_final_path=rq2_path,
            expected_rq2_sha256=rq2_sha,
            rq3_final_path=rq3_path,
            expected_rq3_sha256=rq3_sha,
            expected_rq3_row_id="wrong:01",
        )


def test_validator_rejects_rq3_or_ledger_hash_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger, rq3_sha, selected_row = _ledger(tmp_path, monkeypatch)

    validate_rq4_initial_ledger_document(
        ledger.to_dict(),
        root=tmp_path,
        expected_ledger_sha256=ledger.sha256,
        expected_rq3_sha256=rq3_sha,
        expected_rq3_row_id=selected_row,
    )
    with pytest.raises(ValueError):
        validate_rq4_initial_ledger_document(
            ledger.to_dict(),
            root=tmp_path,
            expected_ledger_sha256="0" * 64,
            expected_rq3_sha256=rq3_sha,
            expected_rq3_row_id=selected_row,
        )


def test_queue_surface_contains_exactly_27_jobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger, _, _ = _ledger(tmp_path, monkeypatch)
    ledger_path = tmp_path / "rq4.json"
    monkeypatch.setattr(rq4_initial, "PROJECT_ROOT", tmp_path)
    commands = rq4_initial.compile_rq4_initial_queue_commands(
        ledger_path=ledger_path,
        ledger=ledger,
        state_dir=tmp_path / "queue",
    )

    assert len(commands) == 30
    assert sum("enqueue-run" in command for command in commands) == 27
    assert all(
        str(command[command.index("--run") + 1]).startswith("g3_rq4_")
        for command in commands[2:29]
    )


@pytest.mark.parametrize(
    ("wrong_sha", "wrong_row"),
    ((True, False), (False, True)),
)
def test_submission_rejects_wrong_external_rq3_binding_before_queue_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    wrong_sha: bool,
    wrong_row: bool,
) -> None:
    ledger_path, ledger, selected_sha, selected_row = _persisted_ledger(
        tmp_path, monkeypatch
    )
    state_dir = tmp_path / "queue"
    monkeypatch.setattr(rq4_initial, "PROJECT_ROOT", tmp_path)

    def unexpected_subprocess(*args, **kwargs):
        raise AssertionError("invalid RQ3 binding reached queue mutation")

    monkeypatch.setattr(rq4_initial.subprocess, "run", unexpected_subprocess)

    with pytest.raises(ValueError):
        rq4_initial.submit_rq4_initial_jobs(
            ledger_path=ledger_path,
            state_dir=state_dir,
            expected_ledger_sha256=ledger.sha256,
            expected_rq3_sha256=("0" * 64 if wrong_sha else selected_sha),
            expected_rq3_row_id=(
                "rq3_output_learned:99" if wrong_row else selected_row
            ),
            dry_run=False,
        )

    assert not state_dir.exists()


def test_worker_rejects_encoded_job_ledger_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger_path, ledger, _, _ = _persisted_ledger(tmp_path, monkeypatch)
    encoded = encode_control_job(ledger, ledger.rows[0].id)
    document = json.loads(base64.urlsafe_b64decode(encoded).decode())
    document["job"]["run_name"] = "tampered_rq4_job"
    mismatched = base64.urlsafe_b64encode(
        json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    ).decode()
    monkeypatch.setattr(rq4_initial, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv(rq4_initial.JOB_ENVIRONMENT, mismatched)
    monkeypatch.setenv(rq4_initial.LEDGER_ENVIRONMENT, str(ledger_path))

    with pytest.raises(ValueError, match="differs from its approved ledger row"):
        rq4_initial.compiled_rq4_initial_job_from_environment()
