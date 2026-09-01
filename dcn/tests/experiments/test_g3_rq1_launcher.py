import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.g3_pretrained_item_embeddings.analysis import control_calibration
from experiments.g3_pretrained_item_embeddings.launchers import rq1 as rq1_launcher
from experiments.g3_pretrained_item_embeddings.launchers.control import (
    decode_control_job,
    encode_control_job,
)
from experiments.g3_pretrained_item_embeddings.launchers.rq1 import (
    build_training_experiment,
    compile_rq1_queue_commands,
    verify_rq1_inputs,
    write_job_contract,
)
from experiments.g3_pretrained_item_embeddings.protocol.constants import (
    APPROVED_PROTOCOL_SHA256,
    APPROVED_UNTIED_CONTROL_LEDGER_SHA256,
)
from experiments.g3_pretrained_item_embeddings.protocol.control_ledger import (
    ManifestReference,
)
from experiments.g3_pretrained_item_embeddings.protocol.rq1_ledger import (
    APPROVED_PREDECESSOR_CALIBRATION_SHA256,
    APPROVED_RQ1_LEDGER_SHA256,
    RQ1_LEDGER_PATH,
    UNTIED_CONTROL_LEDGER_PATH,
    initial_rq1_ledger,
    load_rq1_ledger,
    persist_rq1_ledger,
    validate_rq1_ledger_document,
)


PREDECESSOR_PATH = (
    "experiments/g3_pretrained_item_embeddings/evidence/"
    "untied_control_calibration.json"
)


def _ledger():
    return initial_rq1_ledger(
        predecessor_calibration=ManifestReference(
            kind="g3_untied_control_calibration",
            path=PREDECESSOR_PATH,
            sha256=APPROVED_PREDECESSOR_CALIBRATION_SHA256,
        )
    )


def test_rq1_ledger_is_the_closed_nine_row_direct_family() -> None:
    ledger = _ledger()

    assert ledger.schema_version == 1
    assert ledger.kind == "g3_rq1_content_input"
    assert ledger.protocol_sha256 == APPROVED_PROTOCOL_SHA256
    assert ledger.maximum_jobs == 9
    assert ledger.sha256 == APPROVED_RQ1_LEDGER_SHA256
    assert ledger.untied_control_ledger == ManifestReference(
        kind="g3_untied_control_ledger",
        path=UNTIED_CONTROL_LEDGER_PATH,
        sha256=APPROVED_UNTIED_CONTROL_LEDGER_SHA256,
    )
    assert len(ledger.rows) == len({row.id for row in ledger.rows}) == 9
    assert [row.id for row in ledger.rows] == [
        f"rq1_content_input:{index:02d}" for index in range(1, 10)
    ]
    assert [row.run_name for row in ledger.rows] == [
        f"g3_rq1_content_input_trial_{index:02d}_native50m"
        for index in range(1, 10)
    ]
    assert all(row.dataset_size == "native-50m" for row in ledger.rows)
    assert all(row.batch_size == 512 and row.seed == 42 for row in ledger.rows)
    assert all(row.representation == "content_only_history" for row in ledger.rows)
    assert [
        (
            row.embedding_learning_rate,
            row.deep_learning_rate,
            row.horizon_epochs,
        )
        for row in ledger.rows
    ] == [
        (0.1474458978470563, 0.032433939334700325, 25),
        (0.2183583071089141, 0.021004505318001004, 40),
        (0.11428370130933307, 0.05300550286872779, 15),
        (0.08429586287895262, 0.028951461141444014, 25),
        (0.14809387656173142, 0.03825068924780311, 40),
        (0.3864036698067569, 0.009533778806521353, 15),
        (0.046175373484370126, 0.1142898501613739, 25),
        (0.038786694133655535, 0.0168097543197011, 40),
        (0.32348065643909973, 0.060726563467928006, 15),
    ]
    assert validate_rq1_ledger_document(ledger.to_dict()) == ledger


def test_rq1_ledger_requires_the_exact_predecessor_interface_and_fails_closed(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="predecessor calibration path"):
        initial_rq1_ledger(
            predecessor_calibration=ManifestReference(
                kind="g3_untied_control_calibration",
                path="somewhere/else.json",
                sha256="a" * 64,
            )
        )
    with pytest.raises(ValueError, match="predecessor calibration hash"):
        initial_rq1_ledger(
            predecessor_calibration=ManifestReference(
                kind="g3_untied_control_calibration",
                path=PREDECESSOR_PATH,
                sha256="a" * 64,
            )
        )

    ledger = _ledger()
    changed = copy.deepcopy(ledger.to_dict())
    changed["rows"][0]["representation"]["history"] = "learned_item_id"
    with pytest.raises(ValueError, match="approved RQ1 coordinates"):
        validate_rq1_ledger_document(changed)

    destination = tmp_path / "rq1.json"
    persist_rq1_ledger(destination, ledger)
    first = destination.read_bytes()
    persist_rq1_ledger(destination, ledger)
    assert destination.read_bytes() == first
    destination.write_text("{}")
    with pytest.raises(RuntimeError, match="immutable RQ1 ledger"):
        persist_rq1_ledger(destination, ledger)


def test_materialized_rq1_ledger_binds_the_accepted_predecessor() -> None:
    path = rq1_launcher.PROJECT_ROOT / RQ1_LEDGER_PATH
    ledger = load_rq1_ledger(path)

    assert ledger == _ledger()
    assert ledger.sha256 == APPROVED_RQ1_LEDGER_SHA256


def test_rq1_runtime_builds_content_history_and_bound_all_run_features(
    tmp_path: Path,
) -> None:
    ledger = _ledger()
    feature_data = tmp_path / "bound" / "item_features.parquet"

    for row in ledger.rows:
        compiled = decode_control_job(
            encode_control_job(ledger, row.id),
            ledger,
        )
        experiment = build_training_experiment(
            compiled,
            feature_data_path=feature_data,
        )

        assert experiment.representation.history_representation == "content"
        assert experiment.representation.catalog_representation == "learned_id"
        assert experiment.feature_data_path == feature_data
        assert experiment.g3_dataset_size == "native-50m"


def test_rq1_runtime_requires_the_bound_control_and_calibration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = _ledger()
    control_path = tmp_path / ledger.untied_control_ledger.path
    calibration_path = tmp_path / ledger.predecessor_calibration.path
    control_path.parent.mkdir(parents=True)
    control_path.write_text("{}")
    calibration_path.parent.mkdir(parents=True)
    calibration_path.write_text("{}")
    feature_data = tmp_path / "features/item_features.parquet"
    monkeypatch.setattr(
        rq1_launcher,
        "load_control_ledger",
        lambda path: SimpleNamespace(sha256=ledger.untied_control_ledger.sha256),
    )
    monkeypatch.setattr(
        rq1_launcher,
        "verify_control_inputs",
        lambda root, control, full_validation: feature_data,
    )
    calibration = {
        "sha256": ledger.predecessor_calibration.sha256,
        "control_ledger": {
            "path": ledger.untied_control_ledger.path,
            "logical_sha256": ledger.untied_control_ledger.sha256,
        },
    }
    monkeypatch.setattr(
        control_calibration,
        "load_control_calibration",
        lambda path: calibration,
    )

    assert verify_rq1_inputs(tmp_path, ledger, full_validation=False) == feature_data

    calibration["control_ledger"]["logical_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="different untied control ledger"):
        verify_rq1_inputs(tmp_path, ledger, full_validation=False)


def test_rq1_queue_commands_are_nine_granular_persistent_jobs(
    tmp_path: Path,
) -> None:
    ledger = _ledger()
    ledger_path = tmp_path / "rq1.json"

    commands = compile_rq1_queue_commands(
        ledger_path=ledger_path,
        ledger=ledger,
        state_dir=tmp_path / "queue",
    )

    enqueue = [command for command in commands if "enqueue-run" in command]
    assert len(enqueue) == 9
    assert len({command[command.index("--run") + 1] for command in enqueue}) == 9
    assert all(
        command[command.index("--data-group") + 1] == "g3-native50m-likes"
        for command in enqueue
    )
    assert all(
        Path(command[command.index("--script") + 1]).name == "run_rq1.py"
        for command in enqueue
    )
    assert all(
        any(value.startswith("G3_RQ1_JOB_B64=") for value in command)
        for command in enqueue
    )
    assert all(
        any(value.startswith("G3_RQ1_LEDGER_PATH=") for value in command)
        for command in enqueue
    )
    assert commands[0][-2:] == ["status", "--json"]
    assert commands[1][-1] == "new-batch"
    assert commands[-1][-2:] == ["seal-batch", "DRY_RUN_BATCH"]


def test_rq1_job_contract_is_bound_to_its_immutable_ledger(tmp_path: Path) -> None:
    ledger = _ledger()
    ledger_path = tmp_path / "rq1.json"
    persist_rq1_ledger(ledger_path, ledger)
    compiled = decode_control_job(
        encode_control_job(ledger, ledger.rows[0].id),
        ledger,
    )

    destination = write_job_contract(
        compiled,
        ledger_path,
        tmp_path / "logs",
    )

    assert destination == (
        tmp_path / "logs" / ledger.rows[0].run_name / "g3_rq1_job.json"
    )
