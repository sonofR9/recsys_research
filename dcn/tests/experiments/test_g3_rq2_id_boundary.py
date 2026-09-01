import copy
import math
from pathlib import Path

import pytest

from experiments.g3_pretrained_item_embeddings.analysis.rq2_next_stage_results import (
    APPROVED_RQ2_NEXT_STAGE_EVIDENCE_SHA256,
    RQ2_NEXT_STAGE_EVIDENCE_PATH,
    load_rq2_next_stage_evidence,
    verify_rq2_next_stage_evidence,
)
from experiments.g3_pretrained_item_embeddings.launchers.control import (
    decode_control_job,
    encode_control_job,
)
from experiments.g3_pretrained_item_embeddings.launchers.rq2_id_boundary import (
    build_training_experiment,
    compile_rq2_id_boundary_queue_commands,
    write_job_contract,
)
from experiments.g3_pretrained_item_embeddings.protocol.rq2_id_boundary_ledger import (
    APPROVED_RQ2_ID_BOUNDARY_LEDGER_SHA256,
    RQ2_ID_BOUNDARY_LEDGER_PATH,
    initial_rq2_id_boundary_ledger,
    load_rq2_id_boundary_ledger,
    persist_rq2_id_boundary_ledger,
    validate_rq2_id_boundary_ledger_document,
)


ROOT = Path(__file__).resolve().parents[3]


def test_materialized_next_stage_evidence_freezes_id_winner_and_defers_content() -> None:
    evidence = verify_rq2_next_stage_evidence(
        ROOT / RQ2_NEXT_STAGE_EVIDENCE_PATH,
        root=ROOT,
    )

    assert evidence["sha256"] == APPROVED_RQ2_NEXT_STAGE_EVIDENCE_SHA256
    assert evidence["queue_batch"]["batch_id"] == (
        "39120d657485411788c58fcffed03fc8"
    )
    assert len(evidence["tuning_ledger"]) == 6
    content = evidence["content_capacity_decision"]
    assert content["status"] == "deferred_pending_user_approval"
    assert content["family_id"] == "rq2_content_concat"
    assert content["evaluated_extension_capacity"] == 32
    assert content["selection_changed"] is True
    assert content["previous_selected"]["row_id"] == "rq2_content_concat:02"
    assert content["extension_selected"]["row_id"] == "rq2_content_concat:14"
    assert content["recall_at_100_delta"] == pytest.approx(
        0.07914654839561677 - 0.0789308100270813
    )
    assert evidence["opportunity_accounting"]["rq2_content_concat"] == {
        "base_preselection": 9,
        "base_horizon_followup": 3,
        "approved_base_total": 12,
        "conditional_capacity_extension": 3,
        "cumulative_maximum_after_extension": 15,
    }
    winner = evidence["id_only_selection"]["selected"]
    assert winner["capacity"] == 255
    assert winner["horizon_epochs"] == 40
    assert winner["embedding_learning_rate"] == 0.3041556165944196
    assert winner["deep_learning_rate"] == 0.014506684820055783
    assert winner["metrics"]["recall@100"] == 0.09074562121371973
    assert winner["best_epoch"] == 29
    boundary = evidence["id_only_selection"]["boundary_decision"]
    assert boundary["embedding_learning_rate"]["direction"] is None
    assert boundary["deep_learning_rate"]["direction"] == "lower"
    assert boundary["horizon"]["extension_required"] is False
    assert boundary["extension_required"] is True


def test_id_boundary_ledger_is_exactly_three_deep_rate_probes() -> None:
    evidence = load_rq2_next_stage_evidence(ROOT / RQ2_NEXT_STAGE_EVIDENCE_PATH)
    ledger = initial_rq2_id_boundary_ledger(evidence=evidence)
    lower = 0.0081084848

    assert ledger.maximum_jobs == 3
    assert len(ledger.rows) == 3
    assert [row.id for row in ledger.rows] == [
        "rq2_id_only_densenet:13",
        "rq2_id_only_densenet:14",
        "rq2_id_only_densenet:15",
    ]
    assert ledger.sha256 == APPROVED_RQ2_ID_BOUNDARY_LEDGER_SHA256
    assert all(row.family_id == "rq2_id_only_densenet" for row in ledger.rows)
    assert all(row.capacity == 255 for row in ledger.rows)
    assert all(row.horizon_epochs == 40 for row in ledger.rows)
    assert all(row.seed == 42 for row in ledger.rows)
    assert all(
        row.embedding_learning_rate == 0.3041556165944196
        for row in ledger.rows
    )
    assert [row.deep_learning_rate for row in ledger.rows] == [
        lower / math.sqrt(2),
        lower / 2,
        lower / (2 * math.sqrt(2)),
    ]
    assert ledger.content_capacity_status == {
        "status": "deferred_pending_user_approval",
        "changed_by_this_ledger": False,
    }
    assert ledger.opportunity_accounting["rq2_id_only_densenet"] == {
        "approved_base_total": 12,
        "conditional_deep_lr_extension": 3,
        "cumulative_maximum_after_extension": 15,
    }
    assert ledger.opportunity_accounting["rq2_content_concat"] == {
        "approved_base_total": 12,
        "conditional_capacity_extension": 3,
        "cumulative_maximum_after_extension": 15,
    }


def test_id_boundary_ledger_fails_closed_on_content_or_axis_drift() -> None:
    evidence = load_rq2_next_stage_evidence(ROOT / RQ2_NEXT_STAGE_EVIDENCE_PATH)
    ledger = initial_rq2_id_boundary_ledger(evidence=evidence)
    changed = copy.deepcopy(ledger.to_dict())
    changed["rows"][0]["representation"]["id"] = "rq2_content_concat"
    with pytest.raises(ValueError, match="approved ID-only boundary coordinates"):
        validate_rq2_id_boundary_ledger_document(changed)

    changed = copy.deepcopy(ledger.to_dict())
    changed["rows"][0]["training"]["embedding_learning_rate"] = 0.3
    with pytest.raises(ValueError, match="approved ID-only boundary coordinates"):
        validate_rq2_id_boundary_ledger_document(changed)


def test_id_boundary_ledger_is_immutable_materialized_and_runtime_mapped(
    tmp_path: Path,
) -> None:
    evidence = load_rq2_next_stage_evidence(ROOT / RQ2_NEXT_STAGE_EVIDENCE_PATH)
    ledger = initial_rq2_id_boundary_ledger(evidence=evidence)
    path = tmp_path / "boundary.json"
    persist_rq2_id_boundary_ledger(path, ledger)
    persist_rq2_id_boundary_ledger(path, ledger)
    path.write_text("{}")
    with pytest.raises(RuntimeError, match="immutable RQ2 ID boundary ledger"):
        persist_rq2_id_boundary_ledger(path, ledger)

    assert load_rq2_id_boundary_ledger(ROOT / RQ2_ID_BOUNDARY_LEDGER_PATH) == ledger
    feature_data = tmp_path / "features.parquet"
    for row in ledger.rows:
        compiled = decode_control_job(encode_control_job(ledger, row.id), ledger)
        experiment = build_training_experiment(
            compiled, feature_data_path=feature_data
        )
        assert experiment.representation.history_representation == "id_only_densenet"
        assert experiment.representation.history_hidden_dim == 255
        assert experiment.lr_schedule_horizon_epochs == 40


def test_id_boundary_queue_and_contract_are_three_persistent_jobs(
    tmp_path: Path,
) -> None:
    evidence = load_rq2_next_stage_evidence(ROOT / RQ2_NEXT_STAGE_EVIDENCE_PATH)
    ledger = initial_rq2_id_boundary_ledger(evidence=evidence)
    ledger_path = tmp_path / "boundary.json"
    persist_rq2_id_boundary_ledger(ledger_path, ledger)
    commands = compile_rq2_id_boundary_queue_commands(
        ledger_path=ledger_path,
        ledger=ledger,
        state_dir=tmp_path / "queue",
    )
    enqueue = [command for command in commands if "enqueue-run" in command]
    assert len(enqueue) == 3
    assert all(
        Path(command[command.index("--script") + 1]).name
        == "run_rq2_id_boundary.py"
        for command in enqueue
    )
    assert commands[-1][-2:] == ["seal-batch", "DRY_RUN_BATCH"]

    compiled = decode_control_job(encode_control_job(ledger, ledger.rows[0].id), ledger)
    destination = write_job_contract(
        compiled, ledger_path, tmp_path / "logs"
    )
    assert destination.name == "g3_rq2_id_boundary_job.json"
