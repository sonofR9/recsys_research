import copy
import json
from pathlib import Path

import pytest

from experiments.g3_pretrained_item_embeddings.analysis.rq2_capacity_results import (
    APPROVED_RQ2_CAPACITY_EVIDENCE_SHA256,
    RQ2_CAPACITY_EVIDENCE_PATH,
    assess_capacity_boundaries,
    load_rq2_capacity_evidence,
    select_capacity_winner,
    verify_rq2_capacity_evidence,
)
from experiments.g3_pretrained_item_embeddings.launchers.control import (
    G3_CPU_THREAD_ENVIRONMENT,
    decode_control_job,
    encode_control_job,
)
from experiments.g3_pretrained_item_embeddings.launchers.rq2_next_stage import (
    build_training_experiment,
    compile_rq2_next_stage_queue_commands,
    write_job_contract,
)
from experiments.g3_pretrained_item_embeddings.launchers import rq2_next_stage
from experiments.g3_pretrained_item_embeddings.protocol.rq2_next_stage_ledger import (
    APPROVED_RQ2_NEXT_STAGE_LEDGER_SHA256,
    RQ2_NEXT_STAGE_LEDGER_PATH,
    initial_rq2_next_stage_ledger,
    load_rq2_next_stage_ledger,
    persist_rq2_next_stage_ledger,
    validate_rq2_next_stage_ledger_document,
)


ROOT = Path(__file__).resolve().parents[3]


def _run(
    row_id: str,
    capacity: int,
    recall: float,
    ndcg: float,
    seconds: float,
    manifest_order: int,
    *,
    embedding_learning_rate: float = 0.2,
    deep_learning_rate: float = 0.03,
) -> dict[str, object]:
    return {
        "row_id": row_id,
        "capacity": capacity,
        "embedding_learning_rate": embedding_learning_rate,
        "deep_learning_rate": deep_learning_rate,
        "queue_wall_seconds": seconds,
        "manifest_order": manifest_order,
        "metrics": {"recall@100": recall, "ndcg@100": ndcg},
    }


def test_capacity_selection_uses_recall_ndcg_time_then_manifest_order() -> None:
    runs = (
        _run("rq2:01", 64, 0.1, 0.04, 10.0, 0),
        _run("rq2:02", 128, 0.1, 0.05, 12.0, 1),
        _run("rq2:03", 256, 0.1, 0.05, 11.0, 2),
        _run("rq2:04", 256, 0.1, 0.05, 11.0, 3),
    )

    assert select_capacity_winner(runs)["row_id"] == "rq2:03"


def test_capacity_boundary_decision_requires_only_the_approved_extension() -> None:
    lower = assess_capacity_boundaries(
        _run("content:01", 64, 0.1, 0.04, 10.0, 0),
        approved_capacities=(64, 128, 256),
    )
    assert lower["extension_required"] is True
    assert lower["capacity"] == {
        "selected": 64,
        "direction": "lower",
        "extension_capacity": 32,
    }

    interior = assess_capacity_boundaries(
        _run("id:05", 255, 0.1, 0.04, 10.0, 0),
        approved_capacities=(128, 255, 510),
    )
    assert interior["extension_required"] is False
    assert interior["capacity"]["direction"] is None
    assert interior["embedding_learning_rate"]["direction"] is None
    assert interior["deep_learning_rate"]["direction"] is None


def test_materialized_capacity_evidence_binds_all_runs_and_family_decisions() -> None:
    evidence = verify_rq2_capacity_evidence(
        ROOT / RQ2_CAPACITY_EVIDENCE_PATH,
        root=ROOT,
    )

    assert evidence["sha256"] == APPROVED_RQ2_CAPACITY_EVIDENCE_SHA256
    assert len(evidence["tuning_ledger"]) == 18
    assert evidence["queue_batch"]["batch_id"] == (
        "3e42f3c7926149a79ddcdb8b89e9c18e"
    )
    selections = {
        value["family_id"]: value for value in evidence["family_selections"]
    }
    assert selections["rq2_content_concat"]["selected"]["capacity"] == 64
    assert selections["rq2_content_concat"]["boundary_decision"] == {
        "embedding_learning_rate": {
            "selected": 0.1474458978470563,
            "bounds": [0.0368614745, 0.5897835914],
            "normalized_position": pytest.approx(0.20000000008),
            "direction": None,
        },
        "deep_learning_rate": {
            "selected": 0.032433939334700325,
            "bounds": [0.0081084848, 0.1297357573],
            "normalized_position": pytest.approx(0.20000000028),
            "direction": None,
        },
        "capacity": {
            "selected": 64,
            "direction": "lower",
            "extension_capacity": 32,
        },
        "extension_required": True,
    }
    assert selections["rq2_id_only_densenet"]["selected"]["capacity"] == 255
    assert selections["rq2_id_only_densenet"]["boundary_decision"][
        "extension_required"
    ] is False


def test_next_stage_is_three_content_extension_and_three_id_horizon_rows() -> None:
    evidence = load_rq2_capacity_evidence(ROOT / RQ2_CAPACITY_EVIDENCE_PATH)
    ledger = initial_rq2_next_stage_ledger(evidence=evidence)

    assert ledger.maximum_opportunities == 6
    assert len(ledger.rows) == 6
    assert ledger.sha256 == APPROVED_RQ2_NEXT_STAGE_LEDGER_SHA256
    assert [row.phase for row in ledger.rows] == [
        "capacity_boundary_extension",
        "capacity_boundary_extension",
        "capacity_boundary_extension",
        "selected_capacity_horizon_followup",
        "selected_capacity_horizon_followup",
        "selected_capacity_horizon_followup",
    ]
    content = ledger.rows[:3]
    identifier = ledger.rows[3:]
    assert {(row.family_id, row.capacity, row.horizon_epochs) for row in content} == {
        ("rq2_content_concat", 32, 25)
    }
    assert [
        (row.embedding_learning_rate, row.deep_learning_rate) for row in content
    ] == [
        (0.1064948941833123, 0.12582682908321982),
        (0.1474458978470563, 0.032433939334700325),
        (0.33864447155037025, 0.03993056713468058),
    ]
    assert {(row.family_id, row.capacity) for row in identifier} == {
        ("rq2_id_only_densenet", 255)
    }
    assert [
        (
            row.horizon_epochs,
            row.embedding_learning_rate,
            row.deep_learning_rate,
        )
        for row in identifier
    ] == [
        (15, 0.047134737607146836, 0.04127129308065626),
        (25, 0.12447135415265811, 0.023941907610393703),
        (40, 0.3041556165944196, 0.014506684820055783),
    ]
    assert all(row.reused_from is None for row in ledger.rows)
    assert ledger.deferred_content_horizon_followup == {
        "family_id": "rq2_content_concat",
        "reason": "selected capacity is unresolved at the lower boundary",
        "pending_extension_capacity": 32,
    }


def test_next_stage_fails_closed_on_selection_drift_and_future_content_horizon() -> None:
    evidence = load_rq2_capacity_evidence(ROOT / RQ2_CAPACITY_EVIDENCE_PATH)
    ledger = initial_rq2_next_stage_ledger(evidence=evidence)
    changed = copy.deepcopy(ledger.to_dict())
    changed["rows"][0]["capacity"] = 64
    with pytest.raises(ValueError, match="approved next-stage coordinates"):
        validate_rq2_next_stage_ledger_document(changed)

    future = copy.deepcopy(ledger.to_dict())
    future["rows"].append(copy.deepcopy(future["rows"][3]))
    future["rows"][-1]["family_id"] = "rq2_content_concat"
    with pytest.raises(ValueError, match="approved next-stage coordinates"):
        validate_rq2_next_stage_ledger_document(future)


def test_next_stage_ledger_is_immutable_materialized_and_runtime_mapped(
    tmp_path: Path,
) -> None:
    evidence = load_rq2_capacity_evidence(ROOT / RQ2_CAPACITY_EVIDENCE_PATH)
    ledger = initial_rq2_next_stage_ledger(evidence=evidence)
    path = tmp_path / "next.json"
    persist_rq2_next_stage_ledger(path, ledger)
    persist_rq2_next_stage_ledger(path, ledger)
    path.write_text("{}")
    with pytest.raises(RuntimeError, match="immutable RQ2 next-stage ledger"):
        persist_rq2_next_stage_ledger(path, ledger)

    materialized = load_rq2_next_stage_ledger(ROOT / RQ2_NEXT_STAGE_LEDGER_PATH)
    assert materialized == ledger
    feature_data = tmp_path / "features.parquet"
    for row in ledger.rows:
        compiled = decode_control_job(encode_control_job(ledger, row.id), ledger)
        experiment = build_training_experiment(
            compiled, feature_data_path=feature_data
        )
        assert experiment.representation.history_hidden_dim == row.capacity
        assert experiment.representation.history_representation == (
            "id_content"
            if row.family_id == "rq2_content_concat"
            else "id_only_densenet"
        )


def test_next_stage_queue_and_contract_are_persistent_and_immutable(
    tmp_path: Path,
) -> None:
    evidence = load_rq2_capacity_evidence(ROOT / RQ2_CAPACITY_EVIDENCE_PATH)
    ledger = initial_rq2_next_stage_ledger(evidence=evidence)
    ledger_path = tmp_path / "next.json"
    persist_rq2_next_stage_ledger(ledger_path, ledger)
    commands = compile_rq2_next_stage_queue_commands(
        ledger_path=ledger_path,
        ledger=ledger,
        state_dir=tmp_path / "queue",
    )
    enqueue = [command for command in commands if "enqueue-run" in command]
    assert len(enqueue) == 6
    assert commands[-1][-2:] == ["seal-batch", "DRY_RUN_BATCH"]

    compiled = decode_control_job(encode_control_job(ledger, ledger.rows[0].id), ledger)
    destination = write_job_contract(
        compiled, ledger_path, tmp_path / "logs"
    )
    assert destination.name == "g3_rq2_next_stage_job.json"


@pytest.mark.parametrize("cpu_environment", [(), G3_CPU_THREAD_ENVIRONMENT])
def test_next_stage_submission_returns_existing_exact_ledger_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cpu_environment: tuple[str, ...],
) -> None:
    ledger_path = ROOT / RQ2_NEXT_STAGE_LEDGER_PATH
    ledger = load_rq2_next_stage_ledger(ledger_path)
    state_dir = tmp_path / "queue"
    (state_dir / "batches").mkdir(parents=True)
    (state_dir / "completed").mkdir()
    batch_id = "existing-exact-batch"
    job_ids = []
    runner = Path(rq2_next_stage.__file__).with_name("run_rq2_next_stage.py")
    for index, row in enumerate(ledger.rows):
        job_id = f"job-{index}"
        job_ids.append(job_id)
        record = {
            "id": job_id,
            "batch_id": batch_id,
            "run": row.run_name,
            "script": str(runner.resolve()),
            "data_group": "g3-native50m-likes",
            "environment": [
                f"{rq2_next_stage.JOB_ENVIRONMENT}="
                f"{encode_control_job(ledger, row.id)}",
                f"{rq2_next_stage.LEDGER_ENVIRONMENT}={ledger_path.resolve()}",
                "WANDB_MODE=offline",
                *cpu_environment,
            ],
        }
        (state_dir / "completed" / f"{job_id}.json").write_text(
            json.dumps(record)
        )
    (state_dir / "batches" / f"{batch_id}.json").write_text(
        json.dumps({"id": batch_id, "jobs": job_ids, "sealed": True})
    )
    monkeypatch.setattr(
        rq2_next_stage,
        "verify_rq2_next_stage_inputs",
        lambda *args, **kwargs: tmp_path / "features.parquet",
    )
    monkeypatch.setattr(
        rq2_next_stage.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("existing batch must not resubmit"),
    )

    assert rq2_next_stage.submit_rq2_next_stage_jobs(
        ledger_path=ledger_path,
        state_dir=state_dir,
        dry_run=False,
    ) == batch_id
