import copy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from experiments.g3_pretrained_item_embeddings.launchers.control import (
    CompiledControlJob,
    decode_control_job,
    encode_control_job,
)
import experiments.g3_pretrained_item_embeddings.launchers.rq2_unexpected_diagnostic as diagnostic_launcher
from experiments.g3_pretrained_item_embeddings.launchers.rq2_unexpected_diagnostic import (
    build_training_experiment,
    compile_rq2_unexpected_diagnostic_queue_commands,
    verify_rq2_unexpected_diagnostic_inputs,
    write_job_contract,
)
from experiments.g3_pretrained_item_embeddings.protocol.rq2_unexpected_diagnostic_ledger import (
    APPROVED_RQ2_BOUNDARY_EVIDENCE_SHA256,
    APPROVED_RQ2_UNEXPECTED_DIAGNOSTIC_LEDGER_SHA256,
    RQ2_UNEXPECTED_DIAGNOSTIC_LEDGER_PATH,
    compile_rq2_unexpected_diagnostic_ledger,
    load_rq2_unexpected_diagnostic_ledger,
    persist_rq2_unexpected_diagnostic_ledger,
    validate_rq2_unexpected_diagnostic_ledger_document,
)


ROOT = Path(__file__).resolve().parents[3]


def test_diagnostic_ledger_is_the_exact_three_approved_rows() -> None:
    ledger = compile_rq2_unexpected_diagnostic_ledger(ROOT)

    assert ledger.sha256 == APPROVED_RQ2_UNEXPECTED_DIAGNOSTIC_LEDGER_SHA256
    assert ledger.maximum_jobs == 3
    assert [row.id for row in ledger.rows] == [
        "rq2_unexpected_diagnostic:01",
        "rq2_unexpected_diagnostic:02",
        "rq2_unexpected_diagnostic:03",
    ]
    assert [row.history_hidden_dim for row in ledger.rows] == [32, 32, 128]
    assert [row.embedding_learning_rate for row in ledger.rows] == [
        0.2183583071089141,
        0.2183583071089141,
        0.3041556165944196,
    ]
    assert [row.deep_learning_rate for row in ledger.rows] == [
        0.021004505318001004,
        0.021004505318001004,
        0.014506684820055783,
    ]
    assert all(
        row.to_dict()["dataset"]["size"] == "native-50m"
        and row.to_dict()["training"]["batch_size"] == 512
        and row.to_dict()["training"]["seed"] == 42
        and row.to_dict()["training"]["horizon_epochs"] == 40
        and row.to_dict()["reused_from"] is None
        for row in ledger.rows
    )
    assert ledger.boundary_evidence.logical_sha256 == (
        APPROVED_RQ2_BOUNDARY_EVIDENCE_SHA256
    )
    assert ledger.isolation_contract == {
        "rows": [
            "rq2_unexpected_diagnostic:01",
            "rq2_unexpected_diagnostic:02",
        ],
        "common_initialization": "bit_identical_common_parameters",
        "isolated_factor": "learned_history_item_id_branch",
        "ablated_table": "instantiated_then_zeroed_after_global_initialization",
        "ablated_table_trainable": False,
        "ablated_table_in_optimizer": False,
        "id_only_padding_idx": 0,
    }


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("family_id",), "rq2_id_only_densenet"),
        (("run_name",), "changed-run"),
        (("dataset", "size"), "native-500m"),
        (("training", "batch_size"), 1280),
        (("training", "seed"), 43),
        (("training", "embedding_learning_rate"), 0.1),
        (("training", "deep_learning_rate"), 0.1),
        (("training", "horizon_epochs"), 25),
        (("representation", "history_hidden_dim"), 64),
        (("representation", "history"), "learned_item_id"),
        (("representation", "content_trainable"), True),
    ],
)
def test_diagnostic_builder_rejects_every_drifted_coordinate(
    path: tuple[str, ...], replacement: object, tmp_path: Path
) -> None:
    ledger = compile_rq2_unexpected_diagnostic_ledger(ROOT)
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


def test_diagnostic_document_is_closed_and_immutable(tmp_path: Path) -> None:
    ledger = compile_rq2_unexpected_diagnostic_ledger(ROOT)
    changed = copy.deepcopy(ledger.to_dict())
    changed["rows"][0]["training"]["batch_size"] = 1280
    with pytest.raises(ValueError, match="differs from approval"):
        validate_rq2_unexpected_diagnostic_ledger_document(changed, root=ROOT)

    changed = replace(
        ledger,
        boundary_evidence=replace(
            ledger.boundary_evidence,
            sha256="0" * 64,
        ),
    )
    with pytest.raises(ValueError, match="differs from approval"):
        validate_rq2_unexpected_diagnostic_ledger_document(changed.to_dict(), root=ROOT)

    path = tmp_path / "diagnostic.json"
    persist_rq2_unexpected_diagnostic_ledger(path, ledger, root=ROOT)
    persist_rq2_unexpected_diagnostic_ledger(path, ledger, root=ROOT)
    assert load_rq2_unexpected_diagnostic_ledger(path, root=ROOT) == ledger
    path.write_text("{}")
    with pytest.raises(RuntimeError, match="immutable RQ2 unexpected diagnostic"):
        persist_rq2_unexpected_diagnostic_ledger(path, ledger, root=ROOT)


def test_submission_authenticates_each_completed_evidence_once(
    monkeypatch,
) -> None:
    ledger = compile_rq2_unexpected_diagnostic_ledger(ROOT)
    boundary = json.loads((ROOT / ledger.boundary_evidence.path).read_text())
    rq1 = json.loads((ROOT / ledger.rq1_evidence.path).read_text())
    calls: list[str] = []

    def verify_boundary(path: Path, *, root: Path):
        calls.append("boundary")
        return boundary

    def verify_rq1(path: Path, *, root: Path):
        calls.append("rq1")
        return rq1

    monkeypatch.setattr(
        diagnostic_launcher,
        "verify_rq2_content_deep_lr_boundary_evidence",
        verify_boundary,
    )
    monkeypatch.setattr(diagnostic_launcher, "verify_rq1_evidence", verify_rq1)

    feature_path = verify_rq2_unexpected_diagnostic_inputs(
        ROOT,
        ledger,
        full_validation=True,
    )

    assert feature_path.is_file()
    assert calls == ["boundary", "rq1"]


def test_worker_uses_only_lightweight_evidence_authentication(monkeypatch) -> None:
    ledger = compile_rq2_unexpected_diagnostic_ledger(ROOT)

    def reject_full_verification(*args, **kwargs):
        raise AssertionError("queued workers must not rebuild predecessor evidence")

    monkeypatch.setattr(
        diagnostic_launcher,
        "verify_rq2_content_deep_lr_boundary_evidence",
        reject_full_verification,
    )
    monkeypatch.setattr(
        diagnostic_launcher,
        "verify_rq1_evidence",
        reject_full_verification,
    )

    assert verify_rq2_unexpected_diagnostic_inputs(
        ROOT,
        ledger,
        full_validation=False,
    ).is_file()


def test_diagnostic_queue_builder_experiments_and_contract_are_exact(
    tmp_path: Path,
) -> None:
    ledger = compile_rq2_unexpected_diagnostic_ledger(ROOT)
    ledger_path = tmp_path / "diagnostic.json"
    persist_rq2_unexpected_diagnostic_ledger(ledger_path, ledger, root=ROOT)
    commands = compile_rq2_unexpected_diagnostic_queue_commands(
        ledger_path=ledger_path,
        ledger=ledger,
        state_dir=tmp_path / "queue",
    )
    enqueue = [command for command in commands if "enqueue-run" in command]
    assert len(enqueue) == 3
    assert all(
        Path(command[command.index("--script") + 1]).name
        == "run_rq2_unexpected_diagnostic.py"
        for command in enqueue
    )
    assert commands[-1][-2:] == ["seal-batch", "DRY_RUN_BATCH"]

    feature_data = tmp_path / "features.parquet"
    expected = [
        ("id_content", 32, 0.2183583071089141, 0.021004505318001004),
        ("id_content_zero_id", 32, 0.2183583071089141, 0.021004505318001004),
        ("id_content", 128, 0.3041556165944196, 0.014506684820055783),
    ]
    for row, coordinate in zip(ledger.rows, expected, strict=True):
        compiled = decode_control_job(encode_control_job(ledger, row.id), ledger)
        experiment = build_training_experiment(
            compiled,
            ledger=ledger,
            feature_data_path=feature_data,
        )
        representation, hidden, embedding_lr, deep_lr = coordinate
        assert experiment.representation.history_representation == representation
        assert experiment.representation.history_hidden_dim == hidden
        assert experiment.lr_schedule_horizon_epochs == 40
        assert experiment.embedding_learning_rate == embedding_lr
        assert experiment.deep_learning_rate == deep_lr

    compiled = decode_control_job(encode_control_job(ledger, ledger.rows[0].id), ledger)
    destination = write_job_contract(compiled, ledger_path, tmp_path / "logs")
    assert destination.name == "g3_rq2_unexpected_diagnostic_job.json"


def test_preview_does_not_require_a_materialized_approved_ledger() -> None:
    ledger = compile_rq2_unexpected_diagnostic_ledger(ROOT)

    assert ledger.sha256 == APPROVED_RQ2_UNEXPECTED_DIAGNOSTIC_LEDGER_SHA256
    assert RQ2_UNEXPECTED_DIAGNOSTIC_LEDGER_PATH.endswith(
        "rq2_unexpected_result_diagnostic.json"
    )
