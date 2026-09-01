import copy
from dataclasses import replace
import json
import math
from pathlib import Path

import pytest

from experiments.g3_pretrained_item_embeddings.launchers.control import (
    CompiledControlJob,
    decode_control_job,
    encode_control_job,
)
from experiments.g3_pretrained_item_embeddings.launchers.rq2_content_deep_lr_boundary import (
    build_training_experiment,
    compile_rq2_content_deep_lr_boundary_queue_commands,
    submit_rq2_content_deep_lr_boundary_jobs,
    verify_rq2_content_deep_lr_boundary_inputs,
    write_job_contract,
)
import experiments.g3_pretrained_item_embeddings.launchers.rq2_content_deep_lr_boundary as boundary_launcher
from experiments.g3_pretrained_item_embeddings.protocol.rq2_content_deep_lr_boundary_ledger import (
    APPROVED_RQ2_CONTENT_DEEP_LR_BOUNDARY_LEDGER_SHA256,
    RQ2_CONTENT_DEEP_LR_BOUNDARY_LEDGER_PATH,
    compile_rq2_content_deep_lr_boundary_ledger,
    load_rq2_content_deep_lr_boundary_ledger,
    persist_rq2_content_deep_lr_boundary_ledger,
    validate_rq2_content_deep_lr_boundary_ledger_document,
)


ROOT = Path(__file__).resolve().parents[3]


def test_content_deep_lr_boundary_is_exactly_three_approved_probes() -> None:
    ledger = compile_rq2_content_deep_lr_boundary_ledger(ROOT)

    assert ledger.sha256 == APPROVED_RQ2_CONTENT_DEEP_LR_BOUNDARY_LEDGER_SHA256
    assert ledger.maximum_jobs == 3
    assert [row.id for row in ledger.rows] == [
        "rq2_content_concat:16",
        "rq2_content_concat:17",
        "rq2_content_concat:18",
    ]
    assert [row.deep_learning_rate for row in ledger.rows] == [
        0.0081084848 / math.sqrt(2),
        0.0081084848 / 2,
        0.0081084848 / (2 * math.sqrt(2)),
    ]
    assert all(
        row.family_id == "rq2_content_concat"
        and row.capacity == 32
        and row.embedding_learning_rate == 0.3041556165944196
        and row.horizon_epochs == 40
        and row.batch_size == 512
        and row.seed == 42
        and row.reused_from is None
        for row in ledger.rows
    )
    assert set(ledger.inputs) == {
        "content_horizon_evidence",
        "content_horizon_ledger",
        "resolved_next_stage_evidence",
        "resolved_next_stage_ledger",
        "id_boundary_evidence",
        "id_boundary_ledger",
        "predecessor_calibration",
        "content",
        "features",
    }
    assert ledger.source_selection == {
        "row_id": "rq2_content_concat:12",
        "capacity": 32,
        "horizon_epochs": 40,
        "embedding_learning_rate": 0.3041556165944196,
        "deep_learning_rate": 0.014506684820055783,
        "recall_at_100": 0.08893693160875873,
        "ndcg_at_100": 0.03244652591410125,
        "best_epoch": 12,
        "approved_lower_bound": 0.0081084848,
        "outward_divisors": [math.sqrt(2), 2.0, 2 * math.sqrt(2)],
    }


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("family_id",), "rq2_id_only_densenet"),
        (("run_name",), "changed-run"),
        (("dataset", "size"), "native-500m"),
        (("training", "batch_size"), 256),
        (("training", "seed"), 43),
        (("training", "embedding_learning_rate"), 0.1),
        (("training", "deep_learning_rate"), 0.1),
        (("training", "horizon_epochs"), 25),
        (("representation", "history_hidden_dim"), 16),
        (("representation", "history"), "learned_item_id"),
        (("representation", "content_trainable"), True),
    ],
)
def test_content_deep_lr_boundary_rejects_every_drifted_coordinate(
    path: tuple[str, ...], replacement: object, tmp_path: Path
) -> None:
    ledger = compile_rq2_content_deep_lr_boundary_ledger(ROOT)
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


def test_content_deep_lr_boundary_document_is_closed_and_immutable(
    tmp_path: Path,
) -> None:
    ledger = compile_rq2_content_deep_lr_boundary_ledger(ROOT)
    changed = copy.deepcopy(ledger.to_dict())
    changed["rows"][0]["training"]["deep_learning_rate"] = 0.1
    with pytest.raises(ValueError, match="approved content deep-LR boundary"):
        validate_rq2_content_deep_lr_boundary_ledger_document(changed, root=ROOT)

    changed = replace(
        ledger,
        content_horizon_evidence=replace(
            ledger.content_horizon_evidence,
            sha256="0" * 64,
        ),
    )
    with pytest.raises(ValueError, match="approved content deep-LR boundary"):
        validate_rq2_content_deep_lr_boundary_ledger_document(
            changed.to_dict(), root=ROOT
        )

    path = tmp_path / "boundary.json"
    persist_rq2_content_deep_lr_boundary_ledger(path, ledger, root=ROOT)
    persist_rq2_content_deep_lr_boundary_ledger(path, ledger, root=ROOT)
    assert load_rq2_content_deep_lr_boundary_ledger(path, root=ROOT) == ledger
    path.write_text("{}")
    with pytest.raises(RuntimeError, match="immutable RQ2 content deep-LR ledger"):
        persist_rq2_content_deep_lr_boundary_ledger(path, ledger, root=ROOT)


def test_content_deep_lr_boundary_submission_fully_authenticates_horizon_evidence(
    monkeypatch, tmp_path: Path
) -> None:
    ledger = compile_rq2_content_deep_lr_boundary_ledger(ROOT)
    ledger_path = tmp_path / "boundary.json"
    persist_rq2_content_deep_lr_boundary_ledger(ledger_path, ledger, root=ROOT)
    evidence_path = ROOT / ledger.content_horizon_evidence.path
    evidence = json.loads(evidence_path.read_text())
    calls = []

    def verify(path: Path, *, root: Path):
        calls.append((path, root))
        return evidence

    monkeypatch.setattr(
        boundary_launcher,
        "verify_rq2_content_horizon_evidence",
        verify,
    )

    commands = submit_rq2_content_deep_lr_boundary_jobs(
        ledger_path=ledger_path,
        state_dir=tmp_path / "queue",
        dry_run=True,
    )

    assert calls == [(evidence_path, ROOT)]
    assert commands.count("enqueue-run") == 3


def test_content_deep_lr_boundary_worker_uses_lightweight_authentication(
    monkeypatch,
) -> None:
    ledger = compile_rq2_content_deep_lr_boundary_ledger(ROOT)

    def reject_full_verification(*args, **kwargs):
        raise AssertionError("queued workers must not rebuild predecessor rankings")

    monkeypatch.setattr(
        boundary_launcher,
        "verify_rq2_content_horizon_evidence",
        reject_full_verification,
    )

    feature_data_path = verify_rq2_content_deep_lr_boundary_inputs(
        ROOT,
        ledger,
        full_validation=False,
    )

    assert feature_data_path.is_file()


def test_content_deep_lr_boundary_queue_builder_and_contract_are_exact(
    tmp_path: Path,
) -> None:
    ledger = compile_rq2_content_deep_lr_boundary_ledger(ROOT)
    ledger_path = tmp_path / "boundary.json"
    persist_rq2_content_deep_lr_boundary_ledger(ledger_path, ledger, root=ROOT)
    commands = compile_rq2_content_deep_lr_boundary_queue_commands(
        ledger_path=ledger_path,
        ledger=ledger,
        state_dir=tmp_path / "queue",
    )
    enqueue = [command for command in commands if "enqueue-run" in command]
    assert len(enqueue) == 3
    assert all(
        Path(command[command.index("--script") + 1]).name
        == "run_rq2_content_deep_lr_boundary.py"
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
        assert experiment.lr_schedule_horizon_epochs == 40
        assert experiment.embedding_learning_rate == 0.3041556165944196
        assert experiment.deep_learning_rate == row.deep_learning_rate

    compiled = decode_control_job(encode_control_job(ledger, ledger.rows[0].id), ledger)
    destination = write_job_contract(compiled, ledger_path, tmp_path / "logs")
    assert destination.name == "g3_rq2_content_deep_lr_boundary_job.json"


def test_boundary_preview_does_not_require_a_materialized_approved_ledger() -> None:
    ledger = compile_rq2_content_deep_lr_boundary_ledger(ROOT)

    assert ledger.sha256 == APPROVED_RQ2_CONTENT_DEEP_LR_BOUNDARY_LEDGER_SHA256
    assert RQ2_CONTENT_DEEP_LR_BOUNDARY_LEDGER_PATH.endswith(
        "rq2_content_width32_horizon40_deep_lr_boundary.json"
    )
