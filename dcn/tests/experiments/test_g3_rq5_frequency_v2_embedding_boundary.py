from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.g3_pretrained_item_embeddings.launchers import (
    rq5_frequency_v2_embedding_boundary,
)
from experiments.g3_pretrained_item_embeddings.analysis.rq5_frequency_v2_embedding_boundary_results import (
    select_corrected_frequency_boundary_outcome,
)
from experiments.g3_pretrained_item_embeddings.launchers.control import (
    PROJECT_ROOT,
    decode_control_job,
    encode_control_job,
)
from experiments.g3_pretrained_item_embeddings.protocol.rq5_frequency_v2_embedding_boundary_ledger import (
    RQ5_FREQUENCY_V2_EMBEDDING_BOUNDARY_LEDGER_PATH,
    compile_rq5_frequency_v2_embedding_boundary_ledger,
    load_rq5_frequency_v2_embedding_boundary_ledger,
    persist_rq5_frequency_v2_embedding_boundary_ledger,
)


APPROVED_EMBEDDING_RATES = [
    0.4301409980597794,
    0.6083112331888392,
    0.8602819961195588,
]


def test_boundary_ledger_has_exact_approved_three_jobs() -> None:
    ledger = compile_rq5_frequency_v2_embedding_boundary_ledger(PROJECT_ROOT)

    assert len(ledger.rows) == 3
    assert [row.embedding_learning_rate for row in ledger.rows] == APPROVED_EMBEDDING_RATES
    assert {row.deep_learning_rate for row in ledger.rows} == {
        0.014506684820055783
    }
    assert {row.gate_hidden_dim for row in ledger.rows} == {8}
    assert {row.horizon_epochs for row in ledger.rows} == {40}
    assert {row.batch_size for row in ledger.rows} == {512}
    assert {row.seed for row in ledger.rows} == {42}
    assert ledger.to_dict()["approval"] == {
        "decision": "exactly_three_upper_embedding_lr_jobs",
        "further_jobs_require_renewed_approval": True,
    }


def test_boundary_ledger_is_immutable_and_canonical(tmp_path: Path) -> None:
    ledger = compile_rq5_frequency_v2_embedding_boundary_ledger(PROJECT_ROOT)
    path = PROJECT_ROOT / RQ5_FREQUENCY_V2_EMBEDDING_BOUNDARY_LEDGER_PATH

    persist_rq5_frequency_v2_embedding_boundary_ledger(
        path, ledger, root=PROJECT_ROOT
    )
    assert load_rq5_frequency_v2_embedding_boundary_ledger(
        path,
        root=PROJECT_ROOT,
        expected_ledger_sha256=ledger.sha256,
    ) == ledger
    with pytest.raises(ValueError, match="canonical"):
        persist_rq5_frequency_v2_embedding_boundary_ledger(
            tmp_path / "other.json", ledger, root=PROJECT_ROOT
        )


def test_boundary_queue_has_exactly_three_enqueues() -> None:
    ledger = compile_rq5_frequency_v2_embedding_boundary_ledger(PROJECT_ROOT)
    commands = rq5_frequency_v2_embedding_boundary.compile_queue_surface(
        ledger_path=(
            PROJECT_ROOT / RQ5_FREQUENCY_V2_EMBEDDING_BOUNDARY_LEDGER_PATH
        ),
        ledger=ledger,
        state_dir=PROJECT_ROOT / "generated/training-queue-service",
    )

    assert len(commands) == 6
    assert sum("enqueue-run" in command for command in commands) == 3
    assert {command[command.index("--run") + 1] for command in commands[2:5]} == {
        row.run_name for row in ledger.rows
    }


def test_boundary_training_builder_preserves_corrected_gate_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = compile_rq5_frequency_v2_embedding_boundary_ledger(PROJECT_ROOT)
    row = ledger.rows[2]
    compiled = decode_control_job(encode_control_job(ledger, row.id), ledger)
    captured = {}

    def fake_builder(**arguments):
        captured.update(arguments)
        return SimpleNamespace(base_path="unused")

    monkeypatch.setattr(
        rq5_frequency_v2_embedding_boundary, "build_g3_experiment", fake_builder
    )
    rq5_frequency_v2_embedding_boundary.build_training_experiment(
        compiled,
        ledger=ledger,
        feature_data_path=PROJECT_ROOT / "feature-data",
    )

    assert captured["embedding_learning_rate"] == APPROVED_EMBEDDING_RATES[2]
    assert captured["deep_learning_rate"] == 0.014506684820055783
    assert captured["lr_schedule_horizon_epochs"] == 40
    assert captured["seed"] == 42
    assert captured["representation"].content_gate == "frequency"
    assert captured["representation"].gate_hidden_dim == 8
    assert captured["representation"].frequency_gate_semantics == "fp32_p09_v2"
    assert captured["gate_mechanism_diagnostics"] is True


def _run(row_id: str, embedding_rate: float, recall: float, tail: float) -> dict:
    return {
        "row_id": row_id,
        "embedding_learning_rate": embedding_rate,
        "deep_learning_rate": 0.014506684820055783,
        "gate_hidden_dim": 8,
        "horizon_epochs": 40,
        "best_epoch": 20,
        "queue_wall_seconds": 10.0,
        "metrics": {"recall@100": recall, "ndcg@100": recall / 3},
        "slices": {"tail": {"recall@100": tail}},
    }


def _comparator(recall: float, tail: float) -> dict:
    return {
        "metrics": {"recall@100": recall},
        "slices": {"tail": {"recall@100": tail}},
    }


def test_boundary_outcome_selects_interior_and_applies_overall_tail_rule() -> None:
    runs = [
        _run("rq5_frequency_gate_v2:12", 0.3041556165944196, 0.092, 0.006),
        _run("rq5_frequency_gate_v2:13", APPROVED_EMBEDDING_RATES[0], 0.093, 0.010),
        _run("rq5_frequency_gate_v2:14", APPROVED_EMBEDDING_RATES[1], 0.095, 0.014),
        _run("rq5_frequency_gate_v2:15", APPROVED_EMBEDDING_RATES[2], 0.094, 0.012),
    ]

    outcome = select_corrected_frequency_boundary_outcome(
        runs=runs,
        fixed=_comparator(0.094, 0.012),
        global_gate=_comparator(0.094, 0.011),
    )

    assert outcome["selected_row_id"] == "rq5_frequency_gate_v2:14"
    assert outcome["embedding_learning_rate_boundary"]["direction"] is None
    assert outcome["selection_resolved"] is True
    assert outcome["acceptance_analysis"]["qualifies_frequency_gate"] is True


def test_boundary_outcome_marks_highest_rate_unresolved() -> None:
    runs = [
        _run("rq5_frequency_gate_v2:12", 0.3041556165944196, 0.092, 0.006),
        _run("rq5_frequency_gate_v2:13", APPROVED_EMBEDDING_RATES[0], 0.093, 0.010),
        _run("rq5_frequency_gate_v2:14", APPROVED_EMBEDDING_RATES[1], 0.094, 0.012),
        _run("rq5_frequency_gate_v2:15", APPROVED_EMBEDDING_RATES[2], 0.096, 0.014),
    ]

    outcome = select_corrected_frequency_boundary_outcome(
        runs=runs,
        fixed=_comparator(0.094, 0.012),
        global_gate=_comparator(0.094, 0.011),
    )

    assert outcome["selected_row_id"] == "rq5_frequency_gate_v2:15"
    assert outcome["embedding_learning_rate_boundary"]["direction"] == "upper"
    assert outcome["selection_resolved"] is False
    assert outcome["next_action"] == "renewed_approval"
