import json
import runpy
from pathlib import Path

import pytest


COLLECT = (
    Path(__file__).resolve().parents[3]
    / "experiments/g1_sasrec_item_ids_likes/analysis/collect.py"
)
INITIAL_EMBEDDING_LRS = (0.008, 0.016, 0.032)
INITIAL_DEEP_LRS = (0.003, 0.006, 0.012)


def _run(
    namespace: dict,
    width: int,
    embedding_lr: float,
    deep_lr: float,
    recall: float,
):
    embedding_slug = str(embedding_lr).replace(".", "p")
    deep_slug = str(deep_lr).replace(".", "p")
    return namespace["ReportRun"](
        name=f"width{width}_{embedding_lr}_{deep_lr}",
        configuration=f"mup_dim{width}_e{embedding_slug}_d{deep_slug}",
        dataset_size="50m",
        research_question=1,
        method="μP model-width transfer",
        status="completed",
        metrics={"recall@100": recall},
        metadata={
            "effective_batch_size": 1280,
            "embedding_learning_rate": embedding_lr,
            "deep_learning_rate": deep_lr,
            "item_embedding_dim": 64,
            "model_dim": width,
            "transfer_invariants": {
                "experiment_class": "MuTransferGenerationExperiment",
                "mup_base_dim": 16,
                "mup_delta_dim": 32,
            },
        },
    )


def _width_runs(namespace: dict) -> list:
    runs = []
    for width in (16, 32, 64, 128, 256):
        for embedding_lr in INITIAL_EMBEDDING_LRS:
            for deep_lr in INITIAL_DEEP_LRS:
                recall = 0.2 if (embedding_lr, deep_lr) == (0.016, 0.006) else 0.1
                runs.append(_run(namespace, width, embedding_lr, deep_lr, recall))

    for embedding_lr, deep_lr in ((0.064, 0.012), (0.032, 0.024)):
        runs.append(_run(namespace, 16, embedding_lr, deep_lr, 0.19))
    for run in runs:
        if run.configuration == "mup_dim16_e0p032_d0p012":
            run.metrics["recall@100"] = 0.21
            break
    return runs


def test_rq1_width_evidence_accepts_sparse_axis_aligned_extensions() -> None:
    namespace = runpy.run_path(str(COLLECT))

    selected_rates, comparisons = namespace["_rq1_width_evidence"](
        _width_runs(namespace)
    )

    assert selected_rates == (0.016, 0.006)
    assert [width for width, _, _ in comparisons] == [16, 32, 64, 128, 256]
    assert next(oracle for width, _, oracle in comparisons if width == 16).metrics[
        "recall@100"
    ] == 0.21


def test_rq1_width_evidence_rejects_missing_axis_neighbor() -> None:
    namespace = runpy.run_path(str(COLLECT))
    runs = [
        run
        for run in _width_runs(namespace)
        if run.configuration != "mup_dim16_e0p064_d0p012"
    ]

    with pytest.raises(ValueError, match="axis-aligned lower or upper neighbor"):
        namespace["_rq1_width_evidence"](runs)


def _write_alias_source(root: Path, configuration: str, model_dim: int) -> None:
    directory = root / "logs" / f"g1_rqtune_{configuration}_50m"
    directory.mkdir(parents=True)
    (directory / "final_metrics.json").write_text(json.dumps({"recall@100": 0.1}))
    (directory / "training_metadata.json").write_text(
        json.dumps(
            {
                "dataset_size": "50m",
                "model_dim": model_dim,
                "item_embedding_dim": 64,
                "embedding_learning_rate": 0.016,
                "deep_learning_rate": 0.006,
                "effective_batch_size": 1280,
                "transfer_invariants": {
                    "experiment_class": "MuTransferGenerationExperiment",
                    "mup_base_dim": 16,
                    "mup_delta_dim": 32,
                },
            }
        )
    )


def test_rq1_scoped_loader_materializes_dimension_and_control_aliases(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    namespace = runpy.run_path(str(COLLECT))
    _write_alias_source(tmp_path, "dimension_16_e16d6", 16)
    _write_alias_source(tmp_path, "architecture_control_e16d6", 64)
    globals_ = namespace["load_report_runs"].__globals__
    monkeypatch.setitem(globals_, "GENERATED", tmp_path)
    monkeypatch.setitem(globals_, "_run_status", lambda *_: "completed")

    runs = namespace["load_report_runs"]("50m", research_question=1)

    assert {(run.research_question, run.configuration) for run in runs} == {
        (1, "mup_dim16_e16d6"),
        (1, "mup_dim64_e16d6"),
    }


def _stub_bands(namespace: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        namespace["_rq1_width_table"].__globals__,
        "_metric_bands",
        lambda **_: {metric: 0.001 for metric in namespace["REPORT_METRICS"]},
    )


def _table_rows(table: str) -> dict[str, str]:
    return {
        line.split("|")[1].strip(): line
        for line in table.splitlines()
        if line.startswith("| ")
    }


def test_width_table_marks_a_transfer_row_compared_with_itself(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = runpy.run_path(str(COLLECT))
    _stub_bands(namespace, monkeypatch)

    rows = _table_rows(namespace["_rq1_width_table"](_width_runs(namespace)))

    assert "same run" in rows["32"]
    assert "same run" not in rows["16"]
    assert "%" in rows["16"]


def test_native_confirmation_table_marks_a_row_compared_with_itself(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = runpy.run_path(str(COLLECT))
    _stub_bands(namespace, monkeypatch)
    target_runs = [
        _run(namespace, width, 0.016, 0.006, 0.3) for width in (16, 32, 64, 128, 256)
    ] + [_run(namespace, 16, 0.032, 0.012, 0.31)]

    rows = _table_rows(
        namespace["_rq1_width_table"](_width_runs(namespace), target_runs)
    )

    assert "same run" in rows["32"]
    assert "same run" not in rows["16"]
    assert "%" in rows["16"]
