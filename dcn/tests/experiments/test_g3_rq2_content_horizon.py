import copy
from pathlib import Path

import pytest

from experiments.g3_pretrained_item_embeddings.launchers.control import (
    CompiledControlJob,
    decode_control_job,
    encode_control_job,
)
from experiments.g3_pretrained_item_embeddings.launchers.rq2_content_horizon import (
    build_training_experiment,
    compile_rq2_content_horizon_queue_commands,
    verify_rq2_content_horizon_inputs,
    write_job_contract,
)
from experiments.g3_pretrained_item_embeddings.protocol.rq2_content_horizon_ledger import (
    APPROVED_RQ2_CONTENT_HORIZON_LEDGER_SHA256,
    compile_rq2_content_horizon_ledger,
    load_rq2_content_horizon_ledger,
    persist_rq2_content_horizon_ledger,
    validate_rq2_content_horizon_ledger_document,
)


ROOT = Path(__file__).resolve().parents[3]


def test_content_horizon_ledger_is_exactly_three_approved_width_32_rows() -> None:
    ledger = compile_rq2_content_horizon_ledger(ROOT)

    assert ledger.sha256 == APPROVED_RQ2_CONTENT_HORIZON_LEDGER_SHA256
    assert ledger.maximum_opportunities == ledger.maximum_physical_jobs == 3
    assert [row.id for row in ledger.rows] == [
        "rq2_content_concat:10",
        "rq2_content_concat:11",
        "rq2_content_concat:12",
    ]
    assert [
        (
            row.horizon_epochs,
            row.embedding_learning_rate,
            row.deep_learning_rate,
        )
        for row in ledger.rows
    ] == [
        (15, 0.047134737607146836, 0.04127129308065626),
        (25, 0.12447135415265811, 0.023941907610393703),
        (40, 0.3041556165944196, 0.014506684820055783),
    ]
    assert all(row.capacity == 32 for row in ledger.rows)
    assert all(row.batch_size == 512 and row.seed == 42 for row in ledger.rows)
    assert all(row.reused_from is None for row in ledger.rows)
    assert ledger.content_capacity_decision == {
        "status": "resolved_user_approved",
        "approved_on": "2026-08-29",
        "selected_capacity": 32,
        "next_lower_capacity_authorized": False,
    }


def test_content_horizon_ledger_binds_exact_latest_ancestry_and_selected_metrics() -> (
    None
):
    ledger = compile_rq2_content_horizon_ledger(ROOT)

    assert set(ledger.inputs) == {
        "resolved_next_stage_evidence",
        "resolved_next_stage_ledger",
        "id_boundary_evidence",
        "id_boundary_ledger",
        "predecessor_calibration",
        "content",
        "features",
    }
    assert ledger.source_selection == {
        "row_id": "rq2_content_concat:14",
        "family_id": "rq2_content_concat",
        "capacity": 32,
        "horizon_epochs": 25,
        "embedding_learning_rate": 0.1474458978470563,
        "deep_learning_rate": 0.032433939334700325,
        "epochs_trained": 25,
        "best_epoch": 24,
        "diagnostic_nonfinite_count": 0,
        "metrics": {
            "capped_recall@10": 0.01650637896244223,
            "capped_recall@100": 0.07918962182678012,
            "capped_recall@50": 0.04806416492998638,
            "coverage@10": 0.21738868106673104,
            "coverage@100": 0.6993182092433933,
            "coverage@50": 0.5264872692168456,
            "mrr@10": 0.020631549827507645,
            "mrr@100": 0.02637553971528536,
            "mrr@50": 0.025300673056365193,
            "ndcg@10": 0.012219642351512836,
            "ndcg@100": 0.029330978863554256,
            "ndcg@50": 0.02181192912897356,
            "num_users": 3414.0,
            "recall@10": 0.015142520857295744,
            "recall@100": 0.07914654839561677,
            "recall@50": 0.04791037690954188,
        },
    }
    assert verify_rq2_content_horizon_inputs(
        ROOT,
        ledger,
        full_validation=False,
    ).is_file()


def test_content_horizon_ledger_fails_closed_and_is_immutable(tmp_path: Path) -> None:
    ledger = compile_rq2_content_horizon_ledger(ROOT)
    changed = copy.deepcopy(ledger.to_dict())
    changed["rows"][0]["training"]["horizon_epochs"] = 25
    with pytest.raises(ValueError, match="approved content-horizon coordinates"):
        validate_rq2_content_horizon_ledger_document(changed, root=ROOT)

    changed = copy.deepcopy(ledger.to_dict())
    changed["content_capacity_decision"]["selected_capacity"] = 16
    with pytest.raises(ValueError, match="approved content-horizon coordinates"):
        validate_rq2_content_horizon_ledger_document(changed, root=ROOT)

    path = tmp_path / "content_horizon.json"
    persist_rq2_content_horizon_ledger(path, ledger, root=ROOT)
    persist_rq2_content_horizon_ledger(path, ledger, root=ROOT)
    assert load_rq2_content_horizon_ledger(path, root=ROOT) == ledger
    path.write_text("{}")
    with pytest.raises(RuntimeError, match="immutable RQ2 content-horizon ledger"):
        persist_rq2_content_horizon_ledger(path, ledger, root=ROOT)


def test_content_horizon_queue_runtime_and_contract_are_exactly_three_jobs(
    tmp_path: Path,
) -> None:
    ledger = compile_rq2_content_horizon_ledger(ROOT)
    ledger_path = tmp_path / "content_horizon.json"
    persist_rq2_content_horizon_ledger(ledger_path, ledger, root=ROOT)
    commands = compile_rq2_content_horizon_queue_commands(
        ledger_path=ledger_path,
        ledger=ledger,
        state_dir=tmp_path / "queue",
    )
    enqueue = [command for command in commands if "enqueue-run" in command]
    assert len(enqueue) == 3
    assert all(
        Path(command[command.index("--script") + 1]).name
        == "run_rq2_content_horizon.py"
        for command in enqueue
    )
    assert commands[-1][-2:] == ["seal-batch", "DRY_RUN_BATCH"]

    feature_data = tmp_path / "features.parquet"
    for row in ledger.rows:
        compiled = decode_control_job(encode_control_job(ledger, row.id), ledger)
        experiment = build_training_experiment(
            compiled,
            ledger=ledger,
            feature_data_path=feature_data,
        )
        assert experiment.representation.history_representation == "id_content"
        assert experiment.representation.catalog_representation == "learned_id"
        assert experiment.representation.history_hidden_dim == 32
        assert experiment.lr_schedule_horizon_epochs == row.horizon_epochs

    compiled = decode_control_job(encode_control_job(ledger, ledger.rows[0].id), ledger)
    destination = write_job_contract(compiled, ledger_path, tmp_path / "logs")
    assert destination.name == "g3_rq2_content_horizon_job.json"


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("run_name",), "changed-run"),
        (("dataset", "size"), "native-500m"),
        (("dataset", "source"), "views"),
        (("dataset", "event_limit"), 49_999_999),
        (("dataset", "sampling"), "user_id"),
        (("dataset", "minimum_user_interactions"), 6),
        (("dataset", "validation_interval_seconds"), 1),
        (("dataset", "candidate_catalog"), "sampled"),
        (("dataset", "exclude_seen"), True),
        (("training", "horizon_epochs"), 16),
        (("training", "embedding_learning_rate"), 0.1),
        (("training", "deep_learning_rate"), 0.1),
        (("representation", "id"), "rq2_id_only_densenet"),
        (("representation", "history"), "learned_item_id"),
        (("representation", "catalog"), "frozen_content"),
        (("representation", "history_hidden_dim"), 16),
        (("representation", "separate_history_catalog_tables"), False),
        (("representation", "content_trainable"), True),
        (("representation", "content_width"), 64),
    ],
)
def test_content_horizon_builder_rejects_every_mutated_runnable_coordinate(
    path: tuple[str, ...], replacement: object, tmp_path: Path
) -> None:
    ledger = compile_rq2_content_horizon_ledger(ROOT)
    compiled = decode_control_job(encode_control_job(ledger, ledger.rows[0].id), ledger)
    changed_job = copy.deepcopy(compiled.job)
    target = changed_job
    for name in path[:-1]:
        target = target[name]
    target[path[-1]] = replacement
    changed = CompiledControlJob(
        ledger_sha256=compiled.ledger_sha256,
        row_id=compiled.row_id,
        job=changed_job,
    )

    with pytest.raises(ValueError, match="approved immutable ledger row"):
        build_training_experiment(
            changed,
            ledger=ledger,
            feature_data_path=tmp_path / "features.parquet",
        )
