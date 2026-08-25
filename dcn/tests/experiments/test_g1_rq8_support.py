from __future__ import annotations

from pathlib import Path
import runpy

import polars as pl
import pytest

from dcn.training_metadata import GENERATION_TRAINING_SEMANTICS_REVISION
from experiments.g1_sasrec_item_ids_likes.analysis.sequence_length_distribution import (
    SequenceLengthAnalysis,
)


VARIANT_SCRIPT = (
    Path(__file__).parents[3]
    / "experiments/g1_sasrec_item_ids_likes/configs/variant.py"
)
TUNING_SCRIPT = VARIANT_SCRIPT.with_name("rq_tuning_variant.py")
TUNE_RUN_SUFFIX = f"_ts{GENERATION_TRAINING_SEMANTICS_REVISION}_r2"


def _variants(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("G1_VARIANT", "baseline")
    monkeypatch.setenv("G1_DATASET_SIZE", "50m")
    monkeypatch.delenv("G1_MAX_USERS", raising=False)
    return runpy.run_path(str(VARIANT_SCRIPT))["VARIANTS"]


def test_rq8_variants_cover_extended_sequence_window_and_dropout_ranges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    variants = _variants(monkeypatch)

    assert [variants[name].max_seq_len for name in ("seq_256", "seq_512")] == [
        256,
        512,
    ]
    assert [
        variants[name].transformer.attention_window
        for name in (
            "window_none",
            "window_10",
            "window_25",
            "window_50",
            "window_75",
            "window_100",
        )
    ] == [None, 10, 25, 50, 75, 100]
    assert [
        variants[name].transformer.dropout
        for name in (
            "dropout_0",
            "dropout_5",
            "baseline",
            "dropout_20",
            "dropout_30",
            "dropout_50",
        )
    ] == [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
    for name in ("dropout_0", "dropout_5", "baseline", "dropout_20", "dropout_30", "dropout_50"):
        transformer = variants[name].transformer
        assert transformer.input_dropout == transformer.dropout
        assert transformer.ffn_dropout == transformer.dropout


@pytest.mark.parametrize(
    ("source", "transformer_fields", "experiment_fields", "expected"),
    [
        ("seq_512", "", "max_seq_len", 512),
        ("window_75", "attention_window", "", 75),
        ("dropout_5", "dropout,input_dropout,ffn_dropout", "", 0.05),
    ],
)
def test_rq8_proxy_changes_only_the_selected_factor(
    monkeypatch: pytest.MonkeyPatch,
    source: str,
    transformer_fields: str,
    experiment_fields: str,
    expected: int | float,
) -> None:
    monkeypatch.setenv("G1_DATASET_SIZE", "50m")
    monkeypatch.setenv("G1_TUNE_RUN", f"rq8_test{TUNE_RUN_SUFFIX}")
    monkeypatch.setenv("G1_TUNE_SOURCE_VARIANT", source)
    monkeypatch.setenv("G1_TUNE_TRANSFORMER_FIELDS", transformer_fields)
    monkeypatch.setenv("G1_TUNE_EXPERIMENT_FIELDS", experiment_fields)
    experiment = runpy.run_path(str(TUNING_SCRIPT))["experiment"]

    actual = (
        experiment.max_seq_len
        if experiment_fields
        else getattr(experiment.transformer, transformer_fields.split(",")[0])
    )
    assert actual == expected
    assert experiment.transformer.dim == 64
    assert experiment.transformer.ffn == "swiglu"


def test_sequence_length_analysis_uses_training_events_and_whole_users(
    tmp_path: Path,
) -> None:
    events = tmp_path / "events.parquet"
    pl.DataFrame(
        {
            "uid": [1, 1, 1, 2, 2, 3, 3, 3, 3],
            "timestamp": [1, 2, 100, 1, 2, 1, 2, 3, 4],
        }
    ).write_parquet(events)

    analysis = SequenceLengthAnalysis.from_parquet(
        events,
        validation_interval_seconds=10,
        max_users=2,
        sample_seed=7,
    )

    assert analysis.user_count in (1, 2)
    assert set(analysis.lengths).issubset({2, 4})
    assert all(length >= 2 for length in analysis.lengths)


def test_sequence_length_analysis_writes_median_and_plot(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence"
    analysis = SequenceLengthAnalysis(
        lengths=(1, 2, 3, 8),
        source=Path("events.parquet"),
        validation_interval_seconds=604_800,
        sample_name=None,
    )

    summary_path, plot_path = analysis.write(evidence)

    assert "Median training-history length: **2.5 events**" in summary_path.read_text()
    assert plot_path.is_file()
    assert plot_path.stat().st_size > 0
