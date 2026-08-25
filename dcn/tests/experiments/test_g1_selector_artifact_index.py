import sys
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

from experiments.g1_sasrec_item_ids_likes.analysis import collect
from experiments.g1_sasrec_item_ids_likes.analysis import (
    select_architecture_500m,
    select_width_transfer_500m,
)


def _artifact(directory: Path) -> None:
    directory.mkdir()
    (directory / "training_metadata.json").write_text('{"dataset_size": "50m"}')
    (directory / "final_metrics.json").write_text('{"recall@100": 0.1}')


def test_report_run_base_filter_prunes_before_artifact_validation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    selected = tmp_path / (
        "g1_rqtune_dimension_16_e0p032_d0p012_b1280_ts2_r2_50m"
    )
    unrelated = tmp_path / (
        "g1_rqtune_sequence_25_e0p016_d0p006_b1280_ts2_r2_50m"
    )
    _artifact(selected)
    _artifact(unrelated)
    validated: list[str] = []
    monkeypatch.setattr(
        collect,
        "_run_status",
        lambda directory, *_args: validated.append(directory.name) or "completed",
    )

    runs = collect.load_report_runs(
        "50m",
        directories=(selected, unrelated),
        configuration_base_filter=lambda base: base == "dimension_16",
    )

    assert {run.name for run in runs} == {selected.name}
    assert validated == [selected.name]


def _winner(base: str) -> collect.ReportRun:
    return collect.ReportRun(
        name=f"{base}_winner_50m",
        configuration=f"{base}_e0p016_d0p006_b1280_ts2_r2",
        dataset_size="50m",
        research_question=4 if base.startswith("ffn_") else 8,
        method=base,
        status="completed",
        metrics={"recall@100": 0.1},
        metadata={
            "batch_size": 1280,
            "embedding_learning_rate": 0.016,
            "deep_learning_rate": 0.006,
        },
    )


def test_architecture_selector_indexes_requested_and_full_ffn_surfaces(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    requested = {
        "sequence_128": (0.016, 0.006),
        "position_none": (0.016, 0.006),
        "ffn_gelu171": (0.016, 0.006),
        "ffn_swiglu128": (0.016, 0.006),
    }
    observed: dict[str, object] = {}

    def load_report_runs(dataset_size: str, **kwargs: object) -> list[collect.ReportRun]:
        observed["dataset_size"] = dataset_size
        observed.update(kwargs)
        return []

    monkeypatch.setattr(collect, "load_report_runs", load_report_runs)
    def validate(
        _runs: list[collect.ReportRun],
        _requested: dict[str, tuple[float, float]],
        *,
        exploratory_bases: frozenset[str] = frozenset(),
    ) -> dict[str, collect.ReportRun]:
        observed["exploratory_bases"] = exploratory_bases
        return {base: _winner(base) for base in requested}

    monkeypatch.setattr(
        collect, "validate_architecture_final_selections", validate
    )
    arguments = [
        "select_architecture_500m.py",
        "--generated",
        str(tmp_path),
        *(
            argument
            for base, rates in requested.items()
            for argument in ("--selection", f"{base}:{rates[0]}:{rates[1]}")
        ),
    ]
    monkeypatch.setattr(sys, "argv", arguments)

    select_architecture_500m.main()

    assert len(capsys.readouterr().out.splitlines()) == len(requested)
    assert observed["dataset_size"] == "50m"
    base_filter = cast(Callable[[str], bool], observed["configuration_base_filter"])
    assert base_filter("architecture_control")
    assert base_filter("control_control")
    assert base_filter("sequence_128")
    assert base_filter("position_none")
    assert base_filter("ffn_gelu999")
    assert base_filter("ffn_swiglu8")
    assert not base_filter("sequence_25")
    assert not base_filter("time_none")
    assert observed["exploratory_bases"] == frozenset()


def test_width_selector_indexes_only_approved_source_bases(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    observed: dict[str, object] = {}

    def load_report_runs(dataset_size: str, **kwargs: object) -> list[collect.ReportRun]:
        observed["dataset_size"] = dataset_size
        observed.update(kwargs)
        return [_winner("dimension_16"), _winner("dimension_256")]

    monkeypatch.setattr(collect, "load_report_runs", load_report_runs)
    monkeypatch.setattr(
        select_width_transfer_500m,
        "select_width_transfer_confirmations",
        lambda _runs: [
            (16, "dimension_16_winner_50m", 0.032, 0.012, 1280),
            (256, "dimension_256_winner_50m", 0.032, 0.012, 1280),
        ],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["select_width_transfer_500m.py", "--generated", str(tmp_path)],
    )

    select_width_transfer_500m.main()

    assert len(capsys.readouterr().out.splitlines()) == 2
    assert observed["dataset_size"] == "50m"
    assert observed["research_question"] == 8
    base_filter = cast(Callable[[str], bool], observed["configuration_base_filter"])
    assert base_filter("dimension_16")
    assert base_filter("dimension_256")
    assert not base_filter("dimension_32")
    assert not base_filter("sequence_128")
