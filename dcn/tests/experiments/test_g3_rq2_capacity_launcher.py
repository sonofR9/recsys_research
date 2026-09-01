import copy
from dataclasses import replace
from pathlib import Path

import pytest

from experiments.g3_pretrained_item_embeddings.launchers.control import (
    decode_control_job,
    encode_control_job,
)
from experiments.g3_pretrained_item_embeddings.launchers.rq2_capacity import (
    build_training_experiment,
    compile_rq2_capacity_queue_commands,
    verify_rq2_capacity_inputs,
    write_job_contract,
)
from experiments.g3_pretrained_item_embeddings.protocol.constants import (
    APPROVED_PROTOCOL_SHA256,
)
from experiments.g3_pretrained_item_embeddings.protocol.rq2_capacity_ledger import (
    APPROVED_RQ2_CAPACITY_LEDGER_SHA256,
    RQ2_CAPACITY_LEDGER_PATH,
    initial_rq2_capacity_ledger,
    load_rq2_capacity_ledger,
    persist_rq2_capacity_ledger,
    validate_rq2_capacity_ledger_document,
)


def _ledger():
    return initial_rq2_capacity_ledger()


def test_rq2_capacity_ledger_is_exactly_two_paired_nine_run_first_stages() -> None:
    ledger = _ledger()

    assert ledger.schema_version == 1
    assert ledger.kind == "g3_rq2_capacity_preselection"
    assert ledger.protocol_sha256 == APPROVED_PROTOCOL_SHA256
    assert ledger.maximum_jobs == 18
    assert ledger.sha256 == APPROVED_RQ2_CAPACITY_LEDGER_SHA256
    assert len(ledger.rows) == len({row.id for row in ledger.rows}) == 18
    assert set(ledger.inputs) == {
        "predecessor_calibration",
        "untied_control_ledger",
        "g4_control_manifest",
        "content_manifest",
        "feature_manifest",
    }
    assert all(row.phase == "capacity_preselection" for row in ledger.rows)
    assert all(row.horizon_epochs == 25 for row in ledger.rows)
    assert {row.family_id for row in ledger.rows} == {
        "rq2_content_concat",
        "rq2_id_only_densenet",
    }
    assert {
        width: sum(
            row.capacity == width and row.family_id == "rq2_content_concat"
            for row in ledger.rows
        )
        for width in (64, 128, 256)
    } == {64: 3, 128: 3, 256: 3}
    assert {
        width: sum(
            row.capacity == width and row.family_id == "rq2_id_only_densenet"
            for row in ledger.rows
        )
        for width in (128, 255, 510)
    } == {128: 3, 255: 3, 510: 3}
    signatures = {}
    for family_id in ("rq2_content_concat", "rq2_id_only_densenet"):
        widths = {
            row.capacity for row in ledger.rows if row.family_id == family_id
        }
        for width in sorted(widths):
            signatures[(family_id, width)] = [
                (row.embedding_learning_rate, row.deep_learning_rate)
                for row in ledger.rows
                if row.family_id == family_id and row.capacity == width
            ]
    assert len({tuple(value) for value in signatures.values()}) == 1
    assert validate_rq2_capacity_ledger_document(ledger.to_dict()) == ledger


def test_rq2_capacity_schema_cannot_encode_future_selection_or_followup() -> None:
    document = _ledger().to_dict()
    document["selected_capacity"] = 128
    with pytest.raises(ValueError, match="ledger keys"):
        validate_rq2_capacity_ledger_document(document)

    changed = copy.deepcopy(_ledger().to_dict())
    changed["rows"][0]["training"]["horizon_epochs"] = 15
    with pytest.raises(ValueError, match="approved capacity-preselection coordinates"):
        validate_rq2_capacity_ledger_document(changed)


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("schema_version",), True),
        (("rows", 0, "training", "batch_size"), 512.0),
        (("rows", 0, "training", "restore_best_validation_epoch"), 1),
    ),
)
def test_rq2_capacity_schema_rejects_json_equivalent_wrong_types(
    path: tuple[object, ...], value: object
) -> None:
    document = copy.deepcopy(_ledger().to_dict())
    target = document
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(ValueError, match="invalid JSON type"):
        validate_rq2_capacity_ledger_document(document)


def test_rq2_capacity_ledger_is_immutable_and_materialized(tmp_path: Path) -> None:
    ledger = _ledger()
    path = tmp_path / "rq2.json"
    persist_rq2_capacity_ledger(path, ledger)
    first = path.read_bytes()
    persist_rq2_capacity_ledger(path, ledger)
    assert path.read_bytes() == first
    path.write_text("{}")
    with pytest.raises(RuntimeError, match="immutable RQ2 capacity ledger"):
        persist_rq2_capacity_ledger(path, ledger)

    materialized = Path(__file__).resolve().parents[3] / RQ2_CAPACITY_LEDGER_PATH
    assert load_rq2_capacity_ledger(materialized) == ledger


def test_rq2_runtime_maps_each_family_and_capacity_exactly(tmp_path: Path) -> None:
    ledger = _ledger()
    feature_data = tmp_path / "item_features.parquet"

    for row in ledger.rows:
        compiled = decode_control_job(encode_control_job(ledger, row.id), ledger)
        experiment = build_training_experiment(compiled, feature_data_path=feature_data)
        assert experiment.representation.history_hidden_dim == row.capacity
        assert experiment.representation.catalog_representation == "learned_id"
        expected = (
            "id_content"
            if row.family_id == "rq2_content_concat"
            else "id_only_densenet"
        )
        assert experiment.representation.history_representation == expected
        assert experiment.feature_data_path == feature_data


def test_rq2_queue_is_eighteen_granular_persistent_jobs(tmp_path: Path) -> None:
    ledger = _ledger()
    commands = compile_rq2_capacity_queue_commands(
        ledger_path=tmp_path / "rq2.json",
        ledger=ledger,
        state_dir=tmp_path / "queue",
    )

    enqueue = [command for command in commands if "enqueue-run" in command]
    assert len(enqueue) == 18
    assert len({command[command.index("--run") + 1] for command in enqueue}) == 18
    assert all(
        Path(command[command.index("--script") + 1]).name == "run_rq2_capacity.py"
        for command in enqueue
    )
    assert all(
        any(value.startswith("G3_RQ2_CAPACITY_JOB_B64=") for value in command)
        for command in enqueue
    )
    assert commands[-1][-2:] == ["seal-batch", "DRY_RUN_BATCH"]


def test_rq2_contract_is_bound_to_the_immutable_ledger(tmp_path: Path) -> None:
    ledger = _ledger()
    ledger_path = tmp_path / "rq2.json"
    persist_rq2_capacity_ledger(ledger_path, ledger)
    compiled = decode_control_job(encode_control_job(ledger, ledger.rows[0].id), ledger)

    destination = write_job_contract(compiled, ledger_path, tmp_path / "logs")
    assert destination == (
        tmp_path / "logs" / ledger.rows[0].run_name / "g3_rq2_capacity_job.json"
    )


def test_rq2_runtime_verifies_all_exact_predecessor_bindings() -> None:
    root = Path(__file__).resolve().parents[3]
    ledger = _ledger()

    feature_data = verify_rq2_capacity_inputs(
        root, ledger, full_validation=False
    )
    assert feature_data.name == "item_features.parquet"

    changed_control_input = replace(
        ledger,
        g4_control_manifest=ledger.content_manifest,
    )
    with pytest.raises(ValueError, match="direct G4 control binding"):
        verify_rq2_capacity_inputs(
            root, changed_control_input, full_validation=False
        )

    changed_calibration = replace(
        ledger,
        predecessor_calibration=replace(
            ledger.predecessor_calibration,
            sha256="0" * 64,
        ),
    )
    with pytest.raises(ValueError, match="predecessor calibration hash"):
        verify_rq2_capacity_inputs(
            root, changed_calibration, full_validation=False
        )
