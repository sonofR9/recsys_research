from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.g3_pretrained_item_embeddings.launchers.rq3_boundary import (
    build_rq3_boundary_training_experiment,
    compile_rq3_boundary_queue_commands,
)
from experiments.g3_pretrained_item_embeddings.launchers.control import (
    decode_control_job,
    encode_control_job,
)
from experiments.g3_pretrained_item_embeddings.protocol.rq3_boundary_ledger import (
    RQ3_BOUNDARY_DEEP_LRS,
    RQ3_BOUNDARY_FAMILY_IDS,
    RQ3_INITIAL_EVIDENCE_LOGICAL_SHA256,
    _project_file,
    compile_rq3_boundary_ledger,
    load_rq3_boundary_ledger,
    persist_rq3_boundary_ledger,
    validate_rq3_boundary_ledger_document,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def ledger():
    return compile_rq3_boundary_ledger(PROJECT_ROOT, full_validation=False)


def test_boundary_ledger_is_exact_six_job_continuation(ledger) -> None:
    assert len(ledger.rows) == 6
    assert ledger.initial_evidence.logical_sha256 == RQ3_INITIAL_EVIDENCE_LOGICAL_SHA256
    assert ledger.opportunity_accounting == {
        "initial_logical_opportunities": 45,
        "initial_physical_jobs": 38,
        "initial_reused_rows": 7,
        "boundary_families": 2,
        "boundary_jobs_per_family": 3,
        "new_physical_jobs": 6,
        "cumulative_logical_opportunities": 51,
    }
    assert tuple(row.family_id for row in ledger.rows) == tuple(
        family_id
        for family_id in RQ3_BOUNDARY_FAMILY_IDS
        for _ in RQ3_BOUNDARY_DEEP_LRS
    )
    assert tuple(row.deep_learning_rate for row in ledger.rows) == (
        RQ3_BOUNDARY_DEEP_LRS * 2
    )
    for row in ledger.rows:
        training = row.to_dict()["training"]
        assert training["embedding_learning_rate"] == 0.3041556165944196
        assert training["horizon_epochs"] == 40
        assert training["batch_size"] == 512


def test_boundary_ledger_round_trip_and_tamper_rejection(
    ledger,
    tmp_path: Path,
) -> None:
    path = persist_rq3_boundary_ledger(tmp_path / "ledger.json", ledger)
    assert load_rq3_boundary_ledger(
        path,
        root=PROJECT_ROOT,
        full_validation=False,
    ) == ledger
    document = json.loads(path.read_text())
    document["rows"][0]["training"]["deep_learning_rate"] = 0.9
    with pytest.raises(ValueError, match="differs"):
        validate_rq3_boundary_ledger_document(
            document,
            root=PROJECT_ROOT,
            full_validation=False,
        )


@pytest.mark.parametrize(
    "mutate, message",
    (
        (
            lambda text: text.replace(
                "{",
                '{"schema_version":1,',
                1,
            ),
            "duplicate JSON key",
        ),
        (
            lambda text: text.replace("0.0020271211999999994", "NaN", 1),
            "non-finite JSON value",
        ),
    ),
)
def test_boundary_ledger_rejects_ambiguous_json(
    ledger,
    tmp_path: Path,
    mutate,
    message: str,
) -> None:
    path = persist_rq3_boundary_ledger(tmp_path / "ledger.json", ledger)
    path.write_text(mutate(path.read_text()))

    with pytest.raises(ValueError, match=message):
        load_rq3_boundary_ledger(
            path,
            root=PROJECT_ROOT,
            full_validation=False,
        )


def test_boundary_project_file_rejects_symlink_before_resolution(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.json"
    target.write_text("{}")
    link = tmp_path / "link.json"
    link.symlink_to(target)

    with pytest.raises(ValueError, match="not a project file"):
        _project_file(tmp_path.resolve(), link)


def test_boundary_queue_has_six_unique_enqueues_and_one_seal(
    ledger,
    tmp_path: Path,
) -> None:
    ledger_path = persist_rq3_boundary_ledger(tmp_path / "ledger.json", ledger)
    commands = compile_rq3_boundary_queue_commands(
        ledger_path=ledger_path,
        ledger=ledger,
        state_dir=tmp_path / "queue",
    )

    assert len(commands) == 9
    assert [command[command.index("--run") + 1] for command in commands[2:8]] == [
        row.run_name for row in ledger.rows
    ]
    assert commands[-1][-2:] == ["seal-batch", "DRY_RUN_BATCH"]
    encoded = [
        next(value for value in command if value.startswith("G3_RQ3_BOUNDARY_JOB_B64="))
        for command in commands[2:8]
    ]
    assert len(set(encoded)) == 6


@pytest.mark.parametrize("row_index", range(6))
def test_boundary_worker_builds_exact_catalog_family(ledger, row_index: int) -> None:
    row = ledger.rows[row_index]
    compiled = decode_control_job(encode_control_job(ledger, row.id), ledger)
    experiment = build_rq3_boundary_training_experiment(
        compiled,
        ledger=ledger,
        feature_data_path=(
            PROJECT_ROOT
            / "generated/preprocessing/g3-native50m-likes/item_features.pt"
        ),
    )

    assert experiment.run_name == row.run_name
    assert experiment.dataloader.batch_size == 512
    assert experiment.embedding_learning_rate == 0.3041556165944196
    assert experiment.deep_learning_rate == row.deep_learning_rate
    assert experiment.lr_schedule_horizon_epochs == 40
    assert (
        experiment.representation.catalog_representation
        == row.catalog_representation
    )
