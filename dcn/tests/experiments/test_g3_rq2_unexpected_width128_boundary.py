from __future__ import annotations

from pathlib import Path

import pytest

from experiments.g3_pretrained_item_embeddings.launchers import (
    rq2_unexpected_width128_deep_lr_boundary as launcher,
)
from experiments.g3_pretrained_item_embeddings.protocol.rq2_unexpected_width128_deep_lr_boundary_ledger import (
    APPROVED_RQ2_UNEXPECTED_DIAGNOSTIC_EVIDENCE_SHA256,
    APPROVED_RQ2_UNEXPECTED_WIDTH128_BOUNDARY_LEDGER_SHA256,
    RQ2_UNEXPECTED_WIDTH128_BOUNDARY_LEDGER_PATH,
    compile_rq2_unexpected_width128_boundary_ledger,
    validate_rq2_unexpected_width128_boundary_ledger_document,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_ledger_binds_exact_evidence_and_three_coordinates() -> None:
    ledger = compile_rq2_unexpected_width128_boundary_ledger(PROJECT_ROOT)

    assert ledger.sha256 == APPROVED_RQ2_UNEXPECTED_WIDTH128_BOUNDARY_LEDGER_SHA256
    assert ledger.maximum_jobs == len(ledger.rows) == 3
    assert ledger.diagnostic_evidence.logical_sha256 == (
        APPROVED_RQ2_UNEXPECTED_DIAGNOSTIC_EVIDENCE_SHA256
    )
    assert [row.id for row in ledger.rows] == [
        "rq2_content_concat:19",
        "rq2_content_concat:20",
        "rq2_content_concat:21",
    ]
    assert [row.deep_learning_rate for row in ledger.rows] == [
        0.005733564587228046,
        0.0040542424,
        0.002866782293614023,
    ]
    for row in ledger.rows:
        document = row.to_dict()
        assert document["training"]["batch_size"] == 512
        assert document["training"]["seed"] == 42
        assert document["training"]["embedding_learning_rate"] == 0.3041556165944196
        assert document["training"]["horizon_epochs"] == 40
        assert document["representation"]["history_hidden_dim"] == 128


def test_ledger_validator_rejects_coordinate_drift() -> None:
    ledger = compile_rq2_unexpected_width128_boundary_ledger(PROJECT_ROOT)
    document = ledger.to_dict()
    document["rows"][0]["training"]["deep_learning_rate"] = 0.01

    with pytest.raises(ValueError, match="differs from approval"):
        validate_rq2_unexpected_width128_boundary_ledger_document(
            document, root=PROJECT_ROOT
        )


def test_queue_compiles_exactly_three_physical_jobs(tmp_path: Path) -> None:
    ledger = compile_rq2_unexpected_width128_boundary_ledger(PROJECT_ROOT)
    commands = launcher.compile_rq2_unexpected_width128_boundary_queue_commands(
        ledger_path=PROJECT_ROOT / RQ2_UNEXPECTED_WIDTH128_BOUNDARY_LEDGER_PATH,
        ledger=ledger,
        state_dir=tmp_path,
    )

    assert len(commands) == 6
    rendered = [" ".join(command) for command in commands]
    assert sum(" enqueue-run " in f" {command} " for command in rendered) == 3
    assert all(
        "run_rq2_unexpected_width128_deep_lr_boundary.py" in command
        for command in rendered[2:5]
    )


def test_worker_validation_skips_recursive_evidence_rebuild(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = compile_rq2_unexpected_width128_boundary_ledger(PROJECT_ROOT)
    monkeypatch.setattr(
        launcher,
        "load_rq2_unexpected_diagnostic_evidence",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("worker called full evidence verification")
        ),
    )

    feature_path = launcher.verify_rq2_unexpected_width128_boundary_inputs(
        PROJECT_ROOT, ledger, full_validation=False
    )

    assert feature_path.is_file()
