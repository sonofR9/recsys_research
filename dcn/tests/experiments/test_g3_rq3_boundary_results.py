from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from experiments.g3_pretrained_item_embeddings.analysis import rq3_boundary_results
from experiments.g3_pretrained_item_embeddings.analysis.rq3_boundary_results import (
    _ENVIRONMENT_KEYS,
    _EXPECTED_FINAL_ROW_IDS,
    _final_document,
    _validate_batch,
    _validate_boundary_queue_job,
    _validate_final_document,
    persist_rq3_final_evidence,
)
from experiments.g3_pretrained_item_embeddings.launchers.rq3_boundary import (
    JOB_ENVIRONMENT,
    LEDGER_ENVIRONMENT,
)
from experiments.g3_pretrained_item_embeddings.protocol.constants import (
    APPROVED_PROTOCOL_SHA256,
)


def _batch() -> dict[str, object]:
    return {
        "id": "batch",
        "jobs": [f"job-{index}" for index in range(6)],
        "sealed": True,
        "submitted_at": 1.0,
        "sealed_at": 2.0,
    }


def _queue_job(ledger_path: Path, script_path: Path) -> dict[str, object]:
    environment = {
        JOB_ENVIRONMENT: "encoded",
        LEDGER_ENVIRONMENT: str(ledger_path),
        "WANDB_MODE": "offline",
        **{
            name: "1"
            for name in _ENVIRONMENT_KEYS
            - {JOB_ENVIRONMENT, LEDGER_ENVIRONMENT, "WANDB_MODE"}
        },
    }
    return {
        "id": "job",
        "batch_id": "batch",
        "data_group": "g3-native50m-likes",
        "dispatched_at": 2.0,
        "environment": [f"{key}={value}" for key, value in environment.items()],
        "exit_code": 0,
        "finished_at": 3.0,
        "run": "run",
        "script": str(script_path),
        "submitted_at": 1.0,
    }


def _final_document_fixture() -> dict[str, object]:
    selections = {
        family_id: {
            "status": "resolved",
            "selected": {"row_id": row_id},
            "boundary_decision": {"extension_required": False},
        }
        for family_id, row_id in _EXPECTED_FINAL_ROW_IDS.items()
    }
    readers = {
        family_id: {"row_id": row_id}
        for family_id, row_id in _EXPECTED_FINAL_ROW_IDS.items()
    }
    payload = {
        "schema_version": 1,
        "kind": "g3_rq3_final_native50m_evidence",
        "protocol_sha256": APPROVED_PROTOCOL_SHA256,
        "initial_evidence": {},
        "boundary_ledger": {},
        "queue_batch": {},
        "ranking_context": {},
        "feature_data": {},
        "opportunity_accounting": {
            "initial_logical_rows": 45,
            "initial_reused_rows": 7,
            "initial_physical_rows": 38,
            "boundary_physical_rows": 6,
            "all_logical_rows": 51,
        },
        "boundary_runs": [{} for _ in range(6)],
        "all_tuning_opportunities": [
            {"row_id": f"row-{index}"} for index in range(51)
        ],
        "family_selections": selections,
        "downstream_selection": {
            "status": "resolved",
            "rq4_scientific_selected": {
                "row_id": _EXPECTED_FINAL_ROW_IDS[
                    "rq3_output_learned_frozen_content"
                ]
            },
            "aggregate_selected": {
                "row_id": _EXPECTED_FINAL_ROW_IDS["rq3_output_learned"]
            },
            "treatment_promoted": False,
        },
        "reader_metrics": readers,
        "selected_winner_contrasts": {},
        "matched_coordinate_contrasts": {},
        "mechanism_assessment": {
            "unexpected_ordering": False,
            "best_family_by_selection_rule": (
                "rq3_output_learned_frozen_content"
            ),
        },
    }
    return _final_document(payload)


def test_boundary_batch_requires_exact_schema_and_ordered_times() -> None:
    assert _validate_batch(_batch(), batch_id="batch") == [
        f"job-{index}" for index in range(6)
    ]
    extra = _batch() | {"extra": True}
    with pytest.raises(ValueError, match="exact sealed"):
        _validate_batch(extra, batch_id="batch")
    reversed_times = _batch() | {"sealed_at": 0.5}
    with pytest.raises(ValueError, match="exact sealed"):
        _validate_batch(reversed_times, batch_id="batch")


def test_boundary_queue_job_requires_exact_schema_script_and_environment(
    tmp_path: Path,
) -> None:
    ledger_path = tmp_path / "ledger.json"
    script_path = tmp_path / "run.py"
    job = _queue_job(ledger_path, script_path)
    values = _validate_boundary_queue_job(
        job,
        job_id="job",
        batch_id="batch",
        run_name="run",
        ledger_path=ledger_path.resolve(),
        expected_script=script_path.resolve(),
        row_id="row",
    )
    assert set(values) == _ENVIRONMENT_KEYS
    for mutation in (
        job | {"extra": True},
        job | {"data_group": "wrong"},
        job | {"script": str(tmp_path / "other.py")},
        job | {"finished_at": 0.0},
        job
        | {
            "environment": [
                value
                for value in job["environment"]
                if not value.startswith("OMP_NUM_THREADS=")
            ]
        },
    ):
        with pytest.raises(ValueError, match="queue"):
            _validate_boundary_queue_job(
                mutation,
                job_id="job",
                batch_id="batch",
                run_name="run",
                ledger_path=ledger_path.resolve(),
                expected_script=script_path.resolve(),
                row_id="row",
            )


def test_final_document_requires_exact_schema_accounting_and_selections() -> None:
    document = _final_document_fixture()
    assert _validate_final_document(document) == document

    extra = document | {"extra": True}
    with pytest.raises(ValueError, match="schema"):
        _validate_final_document(extra)

    wrong_accounting_payload = {
        key: deepcopy(value) for key, value in document.items() if key != "sha256"
    }
    wrong_accounting_payload["opportunity_accounting"]["all_logical_rows"] = 50
    with pytest.raises(ValueError, match="schema"):
        _validate_final_document(_final_document(wrong_accounting_payload))

    wrong_selection_payload = {
        key: deepcopy(value) for key, value in document.items() if key != "sha256"
    }
    wrong_selection_payload["family_selections"]["rq3_output_learned"][
        "selected"
    ]["row_id"] = "wrong"
    with pytest.raises(ValueError, match="selection changed"):
        _validate_final_document(_final_document(wrong_selection_payload))

    malformed_selection_payload = {
        key: deepcopy(value) for key, value in document.items() if key != "sha256"
    }
    malformed_selection_payload["family_selections"]["rq3_output_learned"] = None
    with pytest.raises(ValueError, match="selection changed"):
        _validate_final_document(_final_document(malformed_selection_payload))


def test_persistence_requires_complete_authenticated_document(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authenticated = _final_document_fixture()
    monkeypatch.setattr(
        rq3_boundary_results,
        "build_rq3_final_evidence",
        lambda root: authenticated,
    )
    path = tmp_path / rq3_boundary_results.RQ3_FINAL_EVIDENCE_PATH

    incomplete = dict(authenticated)
    incomplete.pop("reader_metrics")
    with pytest.raises(ValueError, match="schema"):
        persist_rq3_final_evidence(path, incomplete, root=tmp_path)

    substituted_payload = {
        key: deepcopy(value)
        for key, value in authenticated.items()
        if key != "sha256"
    }
    substituted_payload["initial_evidence"] = {"substituted": True}
    substituted = _final_document(substituted_payload)
    with pytest.raises(ValueError, match="authenticated artifacts"):
        persist_rq3_final_evidence(path, substituted, root=tmp_path)

    assert not path.exists()


def test_persistence_writes_only_authenticated_canonical_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authenticated = _final_document_fixture()
    monkeypatch.setattr(
        rq3_boundary_results,
        "build_rq3_final_evidence",
        lambda root: authenticated,
    )
    path = tmp_path / rq3_boundary_results.RQ3_FINAL_EVIDENCE_PATH

    assert persist_rq3_final_evidence(
        path,
        authenticated,
        root=tmp_path,
    ) == path
    assert path.read_text() == rq3_boundary_results._canonical_json(authenticated) + "\n"
