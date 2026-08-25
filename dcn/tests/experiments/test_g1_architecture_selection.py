import os
import runpy
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest


COLLECT = (
    Path(__file__).resolve().parents[3]
    / "experiments/g1_sasrec_item_ids_likes/analysis/collect.py"
)
EXPERIMENT = COLLECT.parents[1]
TUNING_LAUNCHER = EXPERIMENT / "launchers/architecture/tuning_50m.sh"
FINAL_LAUNCHER = EXPERIMENT / "launchers/architecture/selected_500m.sh"
MANIFEST = EXPERIMENT / "launchers/architecture/manifest.sh"
QUEUE_STUB = Path(__file__).resolve().parents[1] / "fixtures/training_queue_stub.sh"
GLOBAL_BATCH_STUB = (
    Path(__file__).resolve().parents[1]
    / "fixtures/global_batch_verifier_stub.sh"
)
EMBEDDING_LRS = (0.008, 0.016, 0.032)
DEEP_LRS = (0.003, 0.006, 0.012)
COMMON_EMBEDDING_LR = 0.032


def _run(
    report_run,
    base: str,
    batch_size: int,
    embedding_lr: float,
    deep_lr: float,
    recall: float,
    *,
    dataset_size: str = "50m",
) -> object:
    embedding_slug = f"{embedding_lr:g}".replace(".", "p")
    deep_slug = f"{deep_lr:g}".replace(".", "p")
    suffix = f"e{embedding_slug}_d{deep_slug}_b{batch_size}"
    return report_run(
        name=f"{base}_{suffix}_{dataset_size}",
        configuration=f"{base}_{suffix}",
        dataset_size=dataset_size,
        research_question=(
            9 if base.startswith("time_") else 4 if base.startswith("ffn_") else 8
        ),
        method=base,
        status="completed",
        metrics={"recall@100": recall},
        metadata={
            "batch_size": batch_size,
            "embedding_learning_rate": embedding_lr,
            "deep_learning_rate": deep_lr,
        },
    )


def _grid(report_run, base: str, batch_size: int, peak: float) -> list[object]:
    return [
        _run(
            report_run,
            base,
            batch_size,
            embedding_lr,
            deep_lr,
            peak if (embedding_lr, deep_lr) == (0.016, 0.006) else peak - 0.1,
        )
        for embedding_lr in EMBEDDING_LRS
        for deep_lr in DEEP_LRS
    ]


def _deep_line(
    report_run,
    base: str,
    batch_size: int,
    peak: float,
    *,
    best_deep_lr: float = 0.006,
    deep_lrs: tuple[float, ...] = DEEP_LRS,
) -> list[object]:
    return [
        _run(
            report_run,
            base,
            batch_size,
            COMMON_EMBEDDING_LR,
            deep_lr,
            peak if deep_lr == best_deep_lr else peak - 0.1,
        )
        for deep_lr in deep_lrs
    ]


def _rq4_grid(report_run, peaks: dict[str, float]) -> list[object]:
    control = _grid(report_run, "sequence_128", 1280, 0.90)
    control.extend(
        _run(report_run, "sequence_128", batch, 0.016, 0.006, 0.80)
        for batch in (1024, 1536, 2048)
    )
    return control + [
        run
        for base, peak in peaks.items()
        for run in _grid(report_run, base, 1280, peak)
    ]


def _write_selector_stub(tmp_path: Path, body: str = "raise SystemExit(0)\n") -> Path:
    selector = tmp_path / "architecture_selector.py"
    selector.write_text(body)
    return selector


def test_ffn_manifest_wires_swiglu64_width() -> None:
    result = subprocess.run(
        ["bash", "-c", f'source "{MANIFEST}"; g1_manifest_rows'],
        check=True,
        capture_output=True,
        text=True,
    )

    assert (
        "ffn|swiglu64|ffn_swiglu|ffn||G1_TUNE_FFN_DIM=64|"
        in result.stdout.splitlines()
    )
    assert (
        "ffn|swiglu32|ffn_swiglu|ffn||G1_TUNE_FFN_DIM=32|"
        in result.stdout.splitlines()
    )
    assert (
        "ffn|swiglu16|ffn_swiglu|ffn||G1_TUNE_FFN_DIM=16|"
        in result.stdout.splitlines()
    )


def test_rq4_requires_all_predeclared_swiglu_widths() -> None:
    namespace = runpy.run_path(str(COLLECT))
    report_run = namespace["ReportRun"]
    proxy_runs = _rq4_grid(
        report_run,
        {
            "ffn_gelu128": 0.70,
            "ffn_gelu171": 0.73,
            "ffn_gelu256": 0.72,
            "ffn_gelu384": 0.71,
            "ffn_swiglu32": 0.72,
            "ffn_swiglu64": 0.73,
            "ffn_swiglu96": 0.74,
            "ffn_swiglu128": 0.77,
            "ffn_swiglu171": 0.76,
            "ffn_swiglu224": 0.75,
        },
    )

    with pytest.raises(ValueError, match="ffn_swiglu16"):
        namespace["select_rq4_report_runs"]("50m", proxy_runs, proxy_runs)


def test_rq4_selects_one_closed_width_winner_per_ffn_family() -> None:
    namespace = runpy.run_path(str(COLLECT))
    report_run = namespace["ReportRun"]
    proxy_runs = _rq4_grid(
        report_run,
        {
            "ffn_gelu128": 0.70,
            "ffn_gelu171": 0.73,
            "ffn_gelu256": 0.72,
            "ffn_gelu384": 0.71,
            "ffn_swiglu16": 0.72,
            "ffn_swiglu32": 0.79,
            "ffn_swiglu64": 0.78,
            "ffn_swiglu96": 0.77,
            "ffn_swiglu128": 0.76,
            "ffn_swiglu171": 0.75,
            "ffn_swiglu224": 0.75,
        },
    )

    selected = namespace["select_rq4_report_runs"](
        "50m", proxy_runs, proxy_runs
    )

    assert {namespace["_manifest_base"](run.configuration) for run in selected} == {
        "ffn_gelu171",
        "ffn_swiglu32",
    }


def test_rq4_rejects_family_winner_on_finite_width_boundary() -> None:
    namespace = runpy.run_path(str(COLLECT))
    report_run = namespace["ReportRun"]
    proxy_runs = _rq4_grid(
        report_run,
        {
            "ffn_gelu128": 0.70,
            "ffn_gelu171": 0.71,
            "ffn_gelu256": 0.72,
            "ffn_gelu384": 0.73,
            "ffn_swiglu16": 0.71,
            "ffn_swiglu32": 0.72,
            "ffn_swiglu64": 0.73,
            "ffn_swiglu96": 0.74,
            "ffn_swiglu128": 0.77,
            "ffn_swiglu171": 0.76,
            "ffn_swiglu224": 0.75,
        },
    )

    with pytest.raises(ValueError, match="GELU width winner.*finite boundary"):
        namespace["select_rq4_report_runs"]("50m", proxy_runs, proxy_runs)


def test_rq4_500m_requires_only_exact_family_winner_confirmations() -> None:
    namespace = runpy.run_path(str(COLLECT))
    report_run = namespace["ReportRun"]
    peaks = {
        "ffn_gelu128": 0.70,
        "ffn_gelu171": 0.73,
        "ffn_gelu256": 0.72,
        "ffn_gelu384": 0.71,
        "ffn_swiglu16": 0.71,
        "ffn_swiglu32": 0.72,
        "ffn_swiglu64": 0.73,
        "ffn_swiglu96": 0.74,
        "ffn_swiglu128": 0.77,
        "ffn_swiglu171": 0.76,
        "ffn_swiglu224": 0.75,
    }
    proxy_runs = _rq4_grid(report_run, peaks)
    target_runs = [
        _run(
            report_run,
            base,
            1280,
            0.016,
            0.006,
            peaks[base],
            dataset_size="500m",
        )
        for base in ("ffn_gelu171", "ffn_swiglu128")
    ]

    selected = namespace["select_rq4_report_runs"](
        "500m", target_runs, proxy_runs
    )

    assert {namespace["_manifest_base"](run.configuration) for run in selected} == {
        "ffn_gelu171",
        "ffn_swiglu128",
    }


def test_rq4_500m_takes_the_cap_continuation_of_a_family_confirmation() -> None:
    namespace = runpy.run_path(str(COLLECT))
    report_run = namespace["ReportRun"]
    peaks = {
        "ffn_gelu128": 0.70,
        "ffn_gelu171": 0.73,
        "ffn_gelu256": 0.72,
        "ffn_gelu384": 0.71,
        "ffn_swiglu16": 0.71,
        "ffn_swiglu32": 0.72,
        "ffn_swiglu64": 0.73,
        "ffn_swiglu96": 0.74,
        "ffn_swiglu128": 0.77,
        "ffn_swiglu171": 0.76,
        "ffn_swiglu224": 0.75,
    }
    proxy_runs = _rq4_grid(report_run, peaks)
    confirmations = []
    for base in ("ffn_gelu171", "ffn_swiglu128"):
        confirmation = _run(
            report_run, base, 1280, 0.016, 0.006, peaks[base], dataset_size="500m"
        )
        confirmation = replace(
            confirmation,
            name=f"{confirmation.name[: -len('_500m')]}_ts2_r2_500m",
            configuration=f"{confirmation.configuration}_ts2_r2",
        )
        continuation = replace(
            confirmation,
            name=f"{confirmation.name[: -len('_ts2_r2_500m')]}_cap40_ts2_r2_500m",
            configuration=(
                f"{confirmation.configuration[: -len('_ts2_r2')]}_cap40_ts2_r2"
            ),
            metrics={"recall@100": peaks[base] + 0.01},
        )
        confirmations += [confirmation, continuation]

    selected = namespace["select_rq4_report_runs"]("500m", confirmations, proxy_runs)

    assert {run.name for run in selected} == {
        run.name for run in confirmations if "_cap40_" in run.name
    }


def test_compact_500m_coverage_accepts_only_two_rq4_family_confirmations(
    monkeypatch,
) -> None:
    namespace = runpy.run_path(str(COLLECT))
    report_run = namespace["ReportRun"]
    peaks = {
        "ffn_gelu128": 0.70,
        "ffn_gelu171": 0.73,
        "ffn_gelu256": 0.72,
        "ffn_gelu384": 0.71,
        "ffn_swiglu16": 0.71,
        "ffn_swiglu32": 0.72,
        "ffn_swiglu64": 0.73,
        "ffn_swiglu96": 0.74,
        "ffn_swiglu128": 0.77,
        "ffn_swiglu171": 0.76,
        "ffn_swiglu224": 0.75,
    }
    proxy_runs = _rq4_grid(report_run, peaks)
    target_runs = [
        _run(
            report_run,
            base,
            1280,
            0.016,
            0.006,
            peaks[base],
            dataset_size="500m",
        )
        for base in ("ffn_gelu171", "ffn_swiglu128")
    ]
    globals_ = namespace["_validate_compact_coverage"].__globals__
    monkeypatch.setitem(globals_, "load_report_runs", lambda *_: proxy_runs)
    monkeypatch.setitem(globals_, "_rq1_width_evidence", lambda *_: None)
    monkeypatch.setitem(globals_, "_validate_rq8_cls", lambda *_: None)
    monkeypatch.setitem(globals_, "_validate_negative_tuning", lambda *_: None)
    monkeypatch.setitem(globals_, "_rq2_rq3_evidence", lambda *_: None)
    monkeypatch.setitem(globals_, "_architecture_bases", lambda: set(peaks))
    monkeypatch.setitem(globals_, "_aggregate_homework_baseline", lambda *_: object())
    monkeypatch.setitem(globals_, "validate_homework_reproduction_runs", lambda *_: None)
    monkeypatch.setitem(
        globals_,
        "_REQUIRED_COMPLETED_BASES",
        {4: set(peaks)},
    )

    namespace["_validate_compact_coverage"]("500m", {4: target_runs})


def test_rq4_report_names_family_and_width_as_configuration() -> None:
    namespace = runpy.run_path(str(COLLECT))
    report_run = namespace["ReportRun"]
    runs = [
        _run(report_run, "ffn_gelu171", 1280, 0.016, 0.006, 0.73),
        _run(report_run, "ffn_swiglu128", 1280, 0.016, 0.006, 0.77),
    ]
    namespace["_rq4_table"].__globals__["_metric_bands"] = lambda **_: {
        metric: 1.0 for metric in namespace["REPORT_METRICS"]
    }

    table = namespace["_rq4_table"](runs)

    assert table.splitlines()[0].startswith(
        "| proxy-selected FFN family | selected width | recall@100 |"
    )
    assert "| GELU | 171 |" in table
    assert "| **SwiGLU** | 128 |" in table


def test_final_preflight_rejects_non_winner_rq4_width_or_rate() -> None:
    namespace = runpy.run_path(str(COLLECT))
    report_run = namespace["ReportRun"]
    proxy_runs = _rq4_grid(
        report_run,
        {
            "ffn_gelu128": 0.70,
            "ffn_gelu171": 0.73,
            "ffn_gelu256": 0.72,
            "ffn_gelu384": 0.71,
            "ffn_swiglu16": 0.71,
            "ffn_swiglu32": 0.72,
            "ffn_swiglu64": 0.73,
            "ffn_swiglu96": 0.74,
            "ffn_swiglu128": 0.77,
            "ffn_swiglu171": 0.76,
            "ffn_swiglu224": 0.75,
        },
    )

    with pytest.raises(ValueError, match="exactly the two closed family winners"):
        namespace["validate_architecture_final_selections"](
            proxy_runs,
            {
                "ffn_gelu256": (0.016, 0.006),
                "ffn_swiglu128": (0.016, 0.006),
            },
        )
    with pytest.raises(ValueError, match="selected LR.*does not match"):
        namespace["validate_architecture_final_selections"](
            proxy_runs,
            {
                "ffn_gelu171": (0.032, 0.006),
                "ffn_swiglu128": (0.016, 0.006),
            },
        )


def test_final_preflight_returns_exact_proxy_winner_provenance() -> None:
    namespace = runpy.run_path(str(COLLECT))
    report_run = namespace["ReportRun"]
    proxy_runs = _rq4_grid(
        report_run,
        {
            "ffn_gelu128": 0.70,
            "ffn_gelu171": 0.73,
            "ffn_gelu256": 0.72,
            "ffn_gelu384": 0.71,
            "ffn_swiglu16": 0.71,
            "ffn_swiglu32": 0.72,
            "ffn_swiglu64": 0.73,
            "ffn_swiglu96": 0.74,
            "ffn_swiglu128": 0.77,
            "ffn_swiglu171": 0.76,
            "ffn_swiglu224": 0.75,
        },
    )

    selected = namespace["validate_architecture_final_selections"](
        proxy_runs,
        {
            "ffn_gelu171": (0.016, 0.006),
            "ffn_swiglu128": (0.016, 0.006),
        },
    )

    assert selected["ffn_gelu171"].name.endswith("_50m")
    assert selected["ffn_swiglu128"].name.endswith("_50m")


def _rq4_grid_with_winners_at_gelu171_and_swiglu128(report_run) -> list[object]:
    return _rq4_grid(
        report_run,
        {
            "ffn_gelu128": 0.70,
            "ffn_gelu171": 0.73,
            "ffn_gelu256": 0.72,
            "ffn_gelu384": 0.71,
            "ffn_swiglu16": 0.71,
            "ffn_swiglu32": 0.72,
            "ffn_swiglu64": 0.73,
            "ffn_swiglu96": 0.74,
            "ffn_swiglu128": 0.77,
            "ffn_swiglu171": 0.76,
            "ffn_swiglu224": 0.75,
        },
    )


def test_exploratory_ffn_width_confirms_beside_the_family_winners() -> None:
    namespace = runpy.run_path(str(COLLECT))
    proxy_runs = _rq4_grid_with_winners_at_gelu171_and_swiglu128(
        namespace["ReportRun"]
    )

    selected = namespace["validate_architecture_final_selections"](
        proxy_runs,
        {
            "ffn_gelu171": (0.016, 0.006),
            "ffn_swiglu128": (0.016, 0.006),
            "ffn_swiglu64": (0.016, 0.006),
        },
        exploratory_bases=frozenset({"ffn_swiglu64"}),
    )

    assert selected["ffn_swiglu64"].configuration.startswith("ffn_swiglu64_")


def test_exploratory_ffn_width_still_needs_its_own_closed_winner() -> None:
    namespace = runpy.run_path(str(COLLECT))
    proxy_runs = _rq4_grid_with_winners_at_gelu171_and_swiglu128(
        namespace["ReportRun"]
    )

    with pytest.raises(ValueError, match="ffn_swiglu64: selected LR"):
        namespace["validate_architecture_final_selections"](
            proxy_runs,
            {
                "ffn_gelu171": (0.016, 0.006),
                "ffn_swiglu128": (0.016, 0.006),
                "ffn_swiglu64": (0.032, 0.012),
            },
            exploratory_bases=frozenset({"ffn_swiglu64"}),
        )


def test_exploratory_base_without_a_selection_is_rejected() -> None:
    namespace = runpy.run_path(str(COLLECT))
    proxy_runs = _rq4_grid_with_winners_at_gelu171_and_swiglu128(
        namespace["ReportRun"]
    )

    with pytest.raises(ValueError, match="exploratory bases without a selection"):
        namespace["validate_architecture_final_selections"](
            proxy_runs,
            {
                "ffn_gelu171": (0.016, 0.006),
                "ffn_swiglu128": (0.016, 0.006),
            },
            exploratory_bases=frozenset({"ffn_swiglu64"}),
        )


def test_control_uses_fixed_batch_1280_per_comparison() -> None:
    namespace = runpy.run_path(str(COLLECT))
    report_run = namespace["ReportRun"]
    bases = {
        "sequence_128",
        "position_none",
        "time_none",
        "time_plain_delta_add",
    }
    runs = [
        *_grid(report_run, "sequence_128", 1280, 0.9),
        _run(report_run, "sequence_128", 1024, 0.016, 0.006, 0.99),
        *_grid(report_run, "position_none", 1280, 0.7),
        *_grid(report_run, "position_none", 1024, 0.99),
        *_grid(report_run, "time_none", 1280, 0.8),
        *_grid(report_run, "time_none", 1024, 0.98),
        *_grid(report_run, "time_plain_delta_add", 1280, 0.6),
        *_grid(report_run, "time_plain_delta_add", 1024, 0.98),
    ]

    selected = namespace["select_architecture_report_runs"](
        "50m", runs, runs, bases=bases
    )
    selected_batches = {
        namespace["_manifest_base"](run.configuration): run.metadata["batch_size"]
        for run in selected
    }

    assert selected_batches == {
        "sequence_128": 1280,
        "position_none": 1280,
        "time_none": 1280,
        "time_plain_delta_add": 1280,
    }


def test_control_keeps_control_selection_without_treatment_grid() -> None:
    namespace = runpy.run_path(str(COLLECT))
    report_run = namespace["ReportRun"]
    control_runs = _grid(report_run, "sequence_128", 1280, 0.8)
    control_runs.append(
        _run(report_run, "sequence_128", 1024, 0.016, 0.006, 0.9)
    )

    selected = namespace["select_architecture_report_runs"](
        "50m", control_runs, control_runs, bases={"sequence_128"}
    )

    assert [run.name for run in selected] == [
        "sequence_128_e0p016_d0p006_b1280_50m"
    ]


def test_treatment_searches_only_the_deep_rate_at_the_common_embedding_rate() -> None:
    namespace = runpy.run_path(str(COLLECT))
    report_run = namespace["ReportRun"]
    runs = [
        *_grid(report_run, "sequence_128", 1280, 0.9),
        *_deep_line(report_run, "position_none", 1280, 0.7),
    ]

    selected = namespace["select_architecture_report_runs"](
        "50m", runs, runs, bases={"sequence_128", "position_none"}
    )

    assert "position_none_e0p032_d0p006_b1280_50m" in {run.name for run in selected}


def test_treatment_still_needs_every_deep_rate_of_that_line() -> None:
    namespace = runpy.run_path(str(COLLECT))
    report_run = namespace["ReportRun"]
    runs = [
        *_grid(report_run, "sequence_128", 1280, 0.9),
        *_deep_line(
            report_run, "position_none", 1280, 0.7, deep_lrs=(0.003, 0.006)
        ),
    ]

    with pytest.raises(ValueError, match=r"position_none: missing.*0.032/0.012"):
        namespace["select_architecture_report_runs"](
            "50m", runs, runs, bases={"sequence_128", "position_none"}
        )


def test_treatment_rejects_a_deep_winner_on_the_edge_of_its_line() -> None:
    namespace = runpy.run_path(str(COLLECT))
    report_run = namespace["ReportRun"]
    runs = [
        *_grid(report_run, "sequence_128", 1280, 0.9),
        *_deep_line(report_run, "position_none", 1280, 0.7, best_deep_lr=0.012),
    ]

    with pytest.raises(ValueError, match="lacks an axis-aligned"):
        namespace["select_architecture_report_runs"](
            "50m", runs, runs, bases={"sequence_128", "position_none"}
        )


def test_control_ignores_obsolete_batch_screen_evidence() -> None:
    namespace = runpy.run_path(str(COLLECT))
    report_run = namespace["ReportRun"]
    runs = [
        *_grid(report_run, "sequence_128", 1280, 0.9),
        _run(report_run, "sequence_128", 2048, 0.016, 0.006, 0.99),
    ]

    selected = namespace["select_architecture_report_runs"](
        "50m", runs, runs, bases={"sequence_128"}
    )

    assert [run.name for run in selected] == [
        "sequence_128_e0p016_d0p006_b1280_50m"
    ]


def test_final_selection_rejects_per_treatment_batch_winners() -> None:
    namespace = runpy.run_path(str(COLLECT))
    report_run = namespace["ReportRun"]
    bases = {"sequence_128", "position_none"}
    proxy_runs = [
        *_grid(report_run, "sequence_128", 1280, 0.9),
        _run(report_run, "sequence_128", 1024, 0.016, 0.006, 0.99),
        *_grid(report_run, "position_none", 1280, 0.7),
    ]
    target_runs = [
        _run(
            report_run,
            "sequence_128",
            1280,
            0.016,
            0.006,
            0.8,
            dataset_size="500m",
        ),
        _run(
            report_run,
            "position_none",
            1280,
            0.016,
            0.006,
            0.7,
            dataset_size="500m",
        ),
        _run(
            report_run,
            "position_none",
            1024,
            0.016,
            0.006,
            0.99,
            dataset_size="500m",
        ),
    ]

    selected = namespace["select_architecture_report_runs"](
        "500m", target_runs, proxy_runs, bases=bases
    )

    assert {run.metadata["batch_size"] for run in selected} == {1280}
    assert len(selected) == 2


def test_final_selection_takes_the_cap_continuation_of_a_confirmation() -> None:
    namespace = runpy.run_path(str(COLLECT))
    report_run = namespace["ReportRun"]
    proxy_runs = _grid(report_run, "sequence_128", 1280, 0.9)
    confirmation = _run(
        report_run, "sequence_128", 1280, 0.016, 0.006, 0.80, dataset_size="500m"
    )
    confirmation = replace(
        confirmation,
        name=f"{confirmation.name[: -len('_500m')]}_ts2_r2_500m",
        configuration=f"{confirmation.configuration}_ts2_r2",
    )
    continuation = replace(
        confirmation,
        name=f"{confirmation.name[: -len('_ts2_r2_500m')]}_cap40_ts2_r2_500m",
        configuration=(
            f"{confirmation.configuration[: -len('_ts2_r2')]}_cap40_ts2_r2"
        ),
        metrics={"recall@100": 0.83},
    )

    selected = namespace["select_architecture_report_runs"](
        "500m", [confirmation, continuation], proxy_runs, bases={"sequence_128"}
    )

    assert [run.name for run in selected] == [continuation.name]


def test_rq9_reader_table_does_not_report_per_treatment_batches() -> None:
    namespace = runpy.run_path(str(COLLECT))
    report_run = namespace["ReportRun"]
    control = _run(report_run, "time_none", 768, 0.016, 0.006, 0.1)
    treatment = _run(report_run, "time_plain_delta_add", 768, 0.032, 0.006, 0.11)
    namespace["_rq9_table"].__globals__["_metric_bands"] = lambda **_: {
        metric: 1.0 for metric in namespace["REPORT_METRICS"]
    }

    table = namespace["_rq9_table"]([control, treatment])

    assert "selected batch" not in table


def test_batch_launcher_accepts_only_one_control_treatment() -> None:
    environment = os.environ | {
        "G1_TRAINING_QUEUE_LIBRARY": str(QUEUE_STUB),
        "G1_TUNE_AXES": "time",
        "G1_TUNE_STAGE": "batch",
        "G1_TUNE_TREATMENTS": "time/none time/plain_delta_add",
    }

    result = subprocess.run(
        ["bash", str(TUNING_LAUNCHER)],
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 2
    assert "requires exactly one control treatment" in result.stderr


def test_final_launcher_requires_one_global_batch() -> None:
    environment = os.environ | {
        "G1_TRAINING_QUEUE_LIBRARY": str(QUEUE_STUB),
        "G1_FINAL_AXES": "time",
        "G1_GLOBAL_BATCH_SIZE": "1536",
        "G1_FINAL_SELECTIONS": "time/none:0.016:0.006:1280",
    }

    result = subprocess.run(
        ["bash", str(FINAL_LAUNCHER)],
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 2
    assert "Invalid G1_FINAL_SELECTIONS entry" in result.stderr


def test_architecture_launcher_uses_injected_queue_stub() -> None:
    environment = os.environ | {
        "G1_TRAINING_QUEUE_LIBRARY": str(QUEUE_STUB),
        "G1_TUNE_AXES": "control",
        "G1_TUNE_TREATMENTS": "control/control",
        "G1_TUNE_EMBEDDING_LRS": "0.013",
        "G1_TUNE_DEEP_LRS": "0.007",
        "G1_TUNE_RUN_TAG": "queueguard",
    }

    result = subprocess.run(
        ["bash", str(TUNING_LAUNCHER)],
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 1
    assert "G1_TEST_QUEUE_STUB_SOURCED" in result.stderr
    assert "G1_TEST_QUEUE_STUB_ENQUEUE" in result.stderr
    assert "dcn.main" not in result.stdout + result.stderr


def test_architecture_extended_cap_has_run_and_assignment_provenance(
    tmp_path: Path,
) -> None:
    queue_stub = tmp_path / "queue.sh"
    queue_stub.write_text(
        "enqueue() { printf 'ENQUEUE %s\\n' \"$*\" >&2; return 97; }\n"
        "drain() { return 97; }\n"
    )
    environment = os.environ | {
        "G1_TRAINING_QUEUE_LIBRARY": str(queue_stub),
        "G1_TUNE_AXES": "control",
        "G1_TUNE_TREATMENTS": "control/control",
        "G1_TUNE_EMBEDDING_LRS": "0.013",
        "G1_TUNE_DEEP_LRS": "0.007",
        "G1_TUNE_RUN_TAG": "capcheck",
        "G1_TUNE_EPOCHS": "30",
        "G1_TUNE_RUN_REVISION": "3",
    }

    result = subprocess.run(
        ["bash", str(TUNING_LAUNCHER)],
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 1
    assert "_cap30_ts2_r3_50m" in result.stderr
    assert "G1_TUNE_EPOCHS=30" in result.stderr
    assert "G1_TUNE_RUN_REVISION=3" in result.stderr


@pytest.mark.parametrize("run_tag", ["cap30", "extension_cap30"])
def test_architecture_rejects_reserved_cap_suffix(run_tag: str) -> None:
    result = subprocess.run(
        ["bash", str(TUNING_LAUNCHER)],
        capture_output=True,
        text=True,
        env=os.environ
        | {
            "G1_TRAINING_QUEUE_LIBRARY": str(QUEUE_STUB),
            "G1_TUNE_AXES": "control",
            "G1_TUNE_RUN_TAG": run_tag,
        },
    )

    assert result.returncode == 2
    assert "reserved tag" in result.stderr
    assert "ENQUEUE" not in result.stderr


def test_architecture_tuning_leaves_queue_depth_adaptive() -> None:
    environment = os.environ | {
        "G1_TRAINING_QUEUE_LIBRARY": str(QUEUE_STUB),
        "G1_TEST_REPORT_QUEUE_IN_FLIGHT": "1",
        "G1_TUNE_AXES": "control",
        "G1_TUNE_TREATMENTS": "control/control",
        "G1_TUNE_EMBEDDING_LRS": "0.013",
        "G1_TUNE_DEEP_LRS": "0.007",
        "G1_TUNE_RUN_TAG": "adaptivequeue",
    }
    environment.pop("TRAINING_QUEUE_IN_FLIGHT", None)

    result = subprocess.run(
        ["bash", str(TUNING_LAUNCHER)],
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 1
    assert "G1_TEST_QUEUE_IN_FLIGHT=unset" in result.stderr


@pytest.mark.parametrize(
    ("axis", "treatment", "expected_group"),
    [
        ("control", "control/control", "g1-rq-architecture-50m-seq128"),
        ("sequence", "sequence/12", "g1-rq-architecture-50m-seq12"),
        ("sequence", "sequence/25", "g1-rq-architecture-50m-seq25"),
        ("sequence", "sequence/50", "g1-rq-architecture-50m-seq50"),
        ("sequence", "sequence/512", "g1-rq-architecture-50m-seq512"),
    ],
)
def test_architecture_tuning_scopes_data_group_per_sequence_length(
    tmp_path: Path,
    axis: str,
    treatment: str,
    expected_group: str,
) -> None:
    launcher_directory = (
        tmp_path / "experiments/g1_sasrec_item_ids_likes/launchers"
    )
    shutil.copytree(EXPERIMENT / "launchers", launcher_directory)
    proxy_control = (
        tmp_path / "generated/logs/g1_rqtune_architecture_control_stub_50m"
    )
    proxy_control.mkdir(parents=True)
    (launcher_directory / "verify_artifact.py").write_text(
        "import sys\n"
        "for _ in sys.stdin:\n"
        "    print('0', flush=True)\n"
    )
    environment = os.environ | {
        "G1_TRAINING_QUEUE_LIBRARY": str(QUEUE_STUB),
        "G1_TEST_GLOBAL_BATCH_VERIFIER": str(GLOBAL_BATCH_STUB),
        "G1_TEST_REPORT_DATA_GROUP": "1",
        "G1_GLOBAL_BATCH_SELECTION": "control/control:0.016:0.006:1280",
        "G1_TUNE_BATCH_CONTROL": "control/control:0.016:0.006:1280",
        "G1_TUNE_AXES": axis,
        "G1_TUNE_TREATMENTS": treatment,
        "G1_TUNE_EMBEDDING_LRS": "0.013",
        "G1_TUNE_DEEP_LRS": "0.007",
        "G1_TUNE_RUN_TAG": "datagroupcheck",
    }

    result = subprocess.run(
        ["bash", str(launcher_directory / "architecture/tuning_50m.sh")],
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 1
    assert f"G1_TEST_QUEUE_DATA_GROUP={expected_group}" in result.stderr


def test_architecture_tuning_accepts_extended_control_provenance(
    tmp_path: Path,
) -> None:
    launcher_directory = (
        tmp_path / "experiments/g1_sasrec_item_ids_likes/launchers"
    )
    shutil.copytree(EXPERIMENT / "launchers", launcher_directory)
    proxy_control = (
        tmp_path
        / "generated/logs/"
        "g1_rqtune_architecture_control_e32d12_capcont_cap40_ts2_r2_50m"
    )
    proxy_control.mkdir(parents=True)
    (launcher_directory / "verify_artifact.py").write_text(
        "import sys\n"
        "for line in sys.stdin:\n"
        "    valid = ('G1_TUNE_EPOCHS=40' in line and "
        "'G1_TUNE_RUN_REVISION=2' in line and "
        "'G1_TUNE_GRADIENT_ACCUMULATION_STEPS=1' in line)\n"
        "    print('0' if valid else '1', flush=True)\n"
    )
    queue_stub = tmp_path / "queue.sh"
    queue_stub.write_text(
        "enqueue() { printf 'ENQUEUE %s\\n' \"$*\" >&2; }\n"
        "drain() { return 97; }\n"
    )
    environment = os.environ | {
        "G1_TRAINING_QUEUE_LIBRARY": str(queue_stub),
        "G1_TEST_GLOBAL_BATCH_VERIFIER": str(GLOBAL_BATCH_STUB),
        "G1_GLOBAL_BATCH_SIZE": "1280",
        "G1_GLOBAL_BATCH_SELECTION": "control/control:0.032:0.012:1280",
        "G1_TUNE_BATCH_CONTROL": "control/control:0.032:0.012:1280",
        "G1_TUNE_CONTROL_EPOCHS": "40",
        "G1_TUNE_CONTROL_RUN_REVISION": "2",
        "G1_TUNE_AXES": "dimension",
        "G1_TUNE_TREATMENTS": "dimension/16",
        "G1_TUNE_EMBEDDING_LRS": "0.013",
        "G1_TUNE_DEEP_LRS": "0.007",
        "G1_TUNE_RUN_TAG": "extendedcontrol",
    }

    result = subprocess.run(
        ["bash", str(launcher_directory / "architecture/tuning_50m.sh")],
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 1
    assert "ENQUEUE g1_rqtune_dimension_16" in result.stderr


@pytest.mark.parametrize(
    ("treatment", "physical_batch_size", "accumulation_steps", "run_fragment"),
    [
        ("sequence/512", 640, 2, "b1280_pb640_ga2_batchcontract"),
        ("sequence/256", 1280, 1, "sequence_256_e0p013d0p007_batchcontract"),
    ],
)
def test_architecture_tuning_uses_fixed_effective_batch_contract(
    tmp_path: Path,
    treatment: str,
    physical_batch_size: int,
    accumulation_steps: int,
    run_fragment: str,
) -> None:
    launcher_directory = (
        tmp_path / "experiments/g1_sasrec_item_ids_likes/launchers"
    )
    shutil.copytree(EXPERIMENT / "launchers", launcher_directory)
    proxy_control = (
        tmp_path / "generated/logs/g1_rqtune_architecture_control_stub_50m"
    )
    proxy_control.mkdir(parents=True)
    (launcher_directory / "verify_artifact.py").write_text(
        "import sys\n"
        "for _ in sys.stdin:\n"
        "    print('0', flush=True)\n"
    )
    queue_stub = tmp_path / "queue.sh"
    queue_stub.write_text(
        "enqueue() { printf 'ENQUEUE %s\\n' \"$*\" >&2; }\n"
        "drain() { return 97; }\n"
    )
    environment = os.environ | {
        "G1_TRAINING_QUEUE_LIBRARY": str(queue_stub),
        "G1_TEST_GLOBAL_BATCH_VERIFIER": str(GLOBAL_BATCH_STUB),
        "G1_GLOBAL_BATCH_SIZE": "1280",
        "G1_GLOBAL_BATCH_SELECTION": "control/control:0.016:0.006:1280",
        "G1_TUNE_BATCH_CONTROL": "control/control:0.016:0.006:1280",
        "G1_TUNE_AXES": "sequence",
        "G1_TUNE_TREATMENTS": treatment,
        "G1_TUNE_EMBEDDING_LRS": "0.013",
        "G1_TUNE_DEEP_LRS": "0.007",
        "G1_TUNE_RUN_TAG": "batchcontract",
    }

    result = subprocess.run(
        ["bash", str(launcher_directory / "architecture/tuning_50m.sh")],
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 1
    assert f"G1_TUNE_BATCH_SIZE={physical_batch_size}" in result.stderr
    assert (
        f"G1_TUNE_GRADIENT_ACCUMULATION_STEPS={accumulation_steps}"
        in result.stderr
    )
    assert run_fragment in result.stderr


def test_architecture_final_leaves_queue_depth_adaptive(tmp_path: Path) -> None:
    launcher_directory = (
        tmp_path / "experiments/g1_sasrec_item_ids_likes/launchers"
    )
    shutil.copytree(EXPERIMENT / "launchers", launcher_directory)
    proxy_control = (
        tmp_path / "generated/logs/g1_rqtune_architecture_control_stub_50m"
    )
    proxy_control.mkdir(parents=True)
    (launcher_directory / "verify_artifact.py").write_text(
        "import sys\n"
        "for _ in sys.stdin:\n"
        "    print('0', flush=True)\n"
    )
    environment = os.environ | {
        "G1_TRAINING_QUEUE_LIBRARY": str(QUEUE_STUB),
        "G1_TEST_GLOBAL_BATCH_VERIFIER": str(GLOBAL_BATCH_STUB),
        "G1_TEST_REPORT_QUEUE_IN_FLIGHT": "1",
        "G1_GLOBAL_BATCH_SIZE": "1280",
        "G1_GLOBAL_BATCH_SELECTION": "control/control:0.016:0.006:1280",
        "G1_FINAL_AXES": "control",
        "G1_FINAL_TREATMENTS": "control/control",
        "G1_FINAL_SELECTIONS": "control/control:0.016:0.006",
        "G1_ARCHITECTURE_SELECTOR": str(_write_selector_stub(tmp_path)),
    }
    environment.pop("TRAINING_QUEUE_IN_FLIGHT", None)

    result = subprocess.run(
        [
            "bash",
            str(launcher_directory / "architecture/selected_500m.sh"),
        ],
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 1
    assert "G1_TEST_QUEUE_IN_FLIGHT=unset" in result.stderr


def test_architecture_final_accepts_extended_proxy_and_final_provenance(
    tmp_path: Path,
) -> None:
    launcher_directory = (
        tmp_path / "experiments/g1_sasrec_item_ids_likes/launchers"
    )
    shutil.copytree(EXPERIMENT / "launchers", launcher_directory)
    proxy_control = (
        tmp_path
        / "generated/logs/"
        "g1_rqtune_architecture_control_e16d6_cap30_ts2_r3_50m"
    )
    proxy_control.mkdir(parents=True)
    (launcher_directory / "verify_artifact.py").write_text(
        "import sys\n"
        "for line in sys.stdin:\n"
        "    valid = ('G1_TUNE_EPOCHS=30' in line and "
        "'G1_TUNE_RUN_REVISION=3' in line)\n"
        "    print('0' if valid else '1', flush=True)\n"
    )
    queue_stub = tmp_path / "queue.sh"
    queue_stub.write_text(
        "enqueue() { printf 'ENQUEUE %s\\n' \"$*\" >&2; return 97; }\n"
        "drain() { return 97; }\n"
    )
    environment = os.environ | {
        "G1_TRAINING_QUEUE_LIBRARY": str(queue_stub),
        "G1_TEST_GLOBAL_BATCH_VERIFIER": str(GLOBAL_BATCH_STUB),
        "G1_GLOBAL_BATCH_SIZE": "1280",
        "G1_GLOBAL_BATCH_SELECTION": "control/control:0.016:0.006:1280",
        "G1_FINAL_AXES": "control",
        "G1_FINAL_TREATMENTS": "control/control",
        "G1_FINAL_SELECTIONS": "control/control:0.016:0.006",
        "G1_FINAL_CONTROL_EPOCHS": "30",
        "G1_FINAL_CONTROL_RUN_REVISION": "3",
        "G1_FINAL_EPOCHS": "40",
        "G1_FINAL_RUN_REVISION": "4",
        "G1_ARCHITECTURE_SELECTOR": str(_write_selector_stub(tmp_path)),
    }

    result = subprocess.run(
        ["bash", str(launcher_directory / "architecture/selected_500m.sh")],
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 1
    assert "_cap40_ts2_r4_500m" in result.stderr
    assert "G1_TUNE_EPOCHS=40" in result.stderr
    assert "G1_TUNE_RUN_REVISION=4" in result.stderr


@pytest.mark.parametrize("control_cap,accepted", [(40, True), (20, False)])
def test_aliased_final_control_requires_exact_final_provenance(
    tmp_path: Path, control_cap: int, accepted: bool
) -> None:
    launcher_directory = (
        tmp_path / "experiments/g1_sasrec_item_ids_likes/launchers"
    )
    shutil.copytree(EXPERIMENT / "launchers", launcher_directory)
    proxy_control = (
        tmp_path / "generated/logs/g1_rqtune_architecture_control_stub_50m"
    )
    proxy_control.mkdir(parents=True)
    final_control = (
        tmp_path
        / "generated/logs/"
        f"g1_rqtune_rqfinal_architecture_control_stub_cap{control_cap}_"
        "ts2_r3_500m"
    )
    final_control.mkdir(parents=True)
    (launcher_directory / "verify_artifact.py").write_text(
        "import sys\n"
        "for _ in sys.stdin:\n"
        "    print('0', flush=True)\n"
    )
    control_verifier = tmp_path / "control_verifier.sh"
    control_verifier.write_text(
        "#!/usr/bin/env bash\n"
        "[[ \"$1\" == *_cap40_ts2_r3_500m && \"$6\" == 40 && \"$7\" == 3 ]]\n"
    )
    control_verifier.chmod(0o755)
    environment = os.environ | {
        "G1_TRAINING_QUEUE_LIBRARY": str(QUEUE_STUB),
        "G1_TEST_GLOBAL_BATCH_VERIFIER": str(GLOBAL_BATCH_STUB),
        "G1_TEST_CONTROL_ARTIFACT_VERIFIER": str(control_verifier),
        "G1_GLOBAL_BATCH_SIZE": "1280",
        "G1_GLOBAL_BATCH_SELECTION": "control/control:0.016:0.006:1280",
        "G1_FINAL_AXES": "ffn",
        "G1_FINAL_TREATMENTS": "ffn/swiglu171",
        "G1_FINAL_SELECTIONS": "control/control:0.016:0.006",
        "G1_FINAL_EPOCHS": "40",
        "G1_FINAL_RUN_REVISION": "3",
        "G1_ARCHITECTURE_SELECTOR": str(_write_selector_stub(tmp_path)),
    }

    result = subprocess.run(
        ["bash", str(launcher_directory / "architecture/selected_500m.sh")],
        capture_output=True,
        text=True,
        env=environment,
    )

    if accepted:
        assert result.returncode == 1
        assert "G1_TEST_QUEUE_STUB_SOURCED" in result.stderr
    else:
        assert result.returncode == 2
        assert "Missing compatible final control artifact" in result.stderr
        assert "G1_TEST_QUEUE_STUB_SOURCED" not in result.stderr


def test_sequence_512_final_uses_accumulation_without_changing_effective_batch(
    tmp_path: Path,
) -> None:
    launcher_directory = (
        tmp_path / "experiments/g1_sasrec_item_ids_likes/launchers"
    )
    shutil.copytree(EXPERIMENT / "launchers", launcher_directory)
    proxy_control = (
        tmp_path / "generated/logs/g1_rqtune_architecture_control_stub_50m"
    )
    proxy_control.mkdir(parents=True)
    (launcher_directory / "verify_artifact.py").write_text(
        "import sys\n"
        "for _ in sys.stdin:\n"
        "    print('0', flush=True)\n"
    )
    queue_stub = tmp_path / "queue.sh"
    queue_stub.write_text(
        "enqueue() { printf 'ENQUEUE %s\\n' \"$*\" >&2; }\n"
        "drain() { return 97; }\n"
    )
    environment = os.environ | {
        "G1_TRAINING_QUEUE_LIBRARY": str(queue_stub),
        "G1_TEST_GLOBAL_BATCH_VERIFIER": str(GLOBAL_BATCH_STUB),
        "G1_GLOBAL_BATCH_SIZE": "1280",
        "G1_GLOBAL_BATCH_SELECTION": "control/control:0.016:0.006:1280",
        "G1_FINAL_AXES": "sequence",
        "G1_FINAL_TREATMENTS": "sequence/512",
        "G1_FINAL_SELECTIONS": (
            "control/control:0.016:0.006 sequence/512:0.016:0.006"
        ),
        "G1_FINAL_RUN_TAG": "accumulation",
        "G1_ARCHITECTURE_SELECTOR": str(_write_selector_stub(tmp_path)),
    }

    result = subprocess.run(
        ["bash", str(launcher_directory / "architecture/selected_500m.sh")],
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 1
    assert "G1_TUNE_BATCH_SIZE=640" in result.stderr
    assert "G1_TUNE_GRADIENT_ACCUMULATION_STEPS=2" in result.stderr
    assert "b1280_pb640_ga2_accumulation" in result.stderr


def test_architecture_final_rejects_unselected_batch_before_queue() -> None:
    environment = os.environ | {
        "G1_TRAINING_QUEUE_LIBRARY": str(QUEUE_STUB),
        "G1_TEST_GLOBAL_BATCH_VERIFIER": str(GLOBAL_BATCH_STUB),
        "G1_GLOBAL_BATCH_SIZE": "999",
        "G1_GLOBAL_BATCH_SELECTION": "control/control:0.016:0.006:999",
        "G1_FINAL_AXES": "control",
        "G1_FINAL_TREATMENTS": "control/control",
        "G1_FINAL_SELECTIONS": "control/control:0.016:0.006",
    }

    result = subprocess.run(
        ["bash", str(FINAL_LAUNCHER)],
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 2
    assert "G1_TEST_GLOBAL_BATCH_VERIFIER" in result.stderr
    assert "G1_TEST_QUEUE_STUB_SOURCED" not in result.stderr


def test_architecture_final_rejects_wrong_control_rates_before_queue() -> None:
    environment = os.environ | {
        "G1_TRAINING_QUEUE_LIBRARY": str(QUEUE_STUB),
        "G1_TEST_GLOBAL_BATCH_VERIFIER": str(GLOBAL_BATCH_STUB),
        "G1_GLOBAL_BATCH_SIZE": "1536",
        "G1_GLOBAL_BATCH_SELECTION": "control/control:0.016:0.006:1536",
        "G1_FINAL_AXES": "control",
        "G1_FINAL_TREATMENTS": "control/control",
        "G1_FINAL_SELECTIONS": "control/control:0.032:0.006",
    }

    result = subprocess.run(
        ["bash", str(FINAL_LAUNCHER)],
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 2
    assert "rates must match G1_GLOBAL_BATCH_SELECTION" in result.stderr
    assert "G1_TEST_QUEUE_STUB_SOURCED" not in result.stderr


def test_architecture_final_preflights_every_requested_proxy_winner(
    tmp_path: Path,
) -> None:
    launcher_directory = (
        tmp_path / "experiments/g1_sasrec_item_ids_likes/launchers"
    )
    shutil.copytree(EXPERIMENT / "launchers", launcher_directory)
    proxy_control = (
        tmp_path / "generated/logs/g1_rqtune_architecture_control_stub_50m"
    )
    proxy_control.mkdir(parents=True)
    (launcher_directory / "verify_artifact.py").write_text(
        "import sys\n"
        "for _ in sys.stdin:\n"
        "    print('0', flush=True)\n"
    )
    selector = _write_selector_stub(
        tmp_path,
        "import sys\n"
        "arguments = set(sys.argv[1:])\n"
        "expected = {\n"
        "    'sequence_128:0.016:0.006',\n"
        "    'ffn_gelu171:0.032:0.006',\n"
        "    'ffn_swiglu128:0.016:0.012',\n"
        "}\n"
        "selected = {sys.argv[index + 1] for index, value in "
        "enumerate(sys.argv) if value == '--selection'}\n"
        "print('SELECTOR', sorted(selected), file=sys.stderr)\n"
        "raise SystemExit(73 if selected == expected else 74)\n",
    )
    environment = os.environ | {
        "G1_TRAINING_QUEUE_LIBRARY": str(QUEUE_STUB),
        "G1_TEST_GLOBAL_BATCH_VERIFIER": str(GLOBAL_BATCH_STUB),
        "G1_GLOBAL_BATCH_SIZE": "1280",
        "G1_GLOBAL_BATCH_SELECTION": "control/control:0.016:0.006:1280",
        "G1_FINAL_AXES": "ffn",
        "G1_FINAL_TREATMENTS": "ffn/gelu171 ffn/swiglu128",
        "G1_FINAL_SELECTIONS": (
            "control/control:0.016:0.006 "
            "ffn/gelu171:0.032:0.006 ffn/swiglu128:0.016:0.012"
        ),
        "G1_ARCHITECTURE_SELECTOR": str(selector),
    }

    result = subprocess.run(
        ["bash", str(launcher_directory / "architecture/selected_500m.sh")],
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 2
    assert "SELECTOR" in result.stderr
    assert "Native-50M final-selection preflight failed" in result.stderr
    assert "G1_TEST_QUEUE_STUB_SOURCED" not in result.stderr


def test_architecture_final_preflights_an_exploratory_treatment_apart(
    tmp_path: Path,
) -> None:
    launcher_directory = (
        tmp_path / "experiments/g1_sasrec_item_ids_likes/launchers"
    )
    shutil.copytree(EXPERIMENT / "launchers", launcher_directory)
    (tmp_path / "generated/logs/g1_rqtune_architecture_control_stub_50m").mkdir(
        parents=True
    )
    (launcher_directory / "verify_artifact.py").write_text(
        "import sys\n"
        "for _ in sys.stdin:\n"
        "    print('0', flush=True)\n"
    )
    selector = _write_selector_stub(
        tmp_path,
        "import sys\n"
        "def collected(flag):\n"
        "    return {sys.argv[index + 1] for index, value in "
        "enumerate(sys.argv) if value == flag}\n"
        "print(\n"
        "    'SELECTOR',\n"
        "    sorted(collected('--selection')),\n"
        "    sorted(collected('--exploratory-selection')),\n"
        "    file=sys.stderr,\n"
        ")\n"
        "raise SystemExit(74)\n",
    )
    environment = os.environ | {
        "G1_TRAINING_QUEUE_LIBRARY": str(QUEUE_STUB),
        "G1_TEST_GLOBAL_BATCH_VERIFIER": str(GLOBAL_BATCH_STUB),
        "G1_GLOBAL_BATCH_SIZE": "1280",
        "G1_GLOBAL_BATCH_SELECTION": "control/control:0.016:0.006:1280",
        "G1_FINAL_AXES": "ffn",
        "G1_FINAL_TREATMENTS": "ffn/gelu171 ffn/swiglu128",
        "G1_FINAL_EXPLORATORY_TREATMENTS": "ffn/swiglu128",
        "G1_FINAL_RUN_TAG": "widthprobe",
        "G1_FINAL_SELECTIONS": (
            "control/control:0.016:0.006 "
            "ffn/gelu171:0.032:0.006 ffn/swiglu128:0.016:0.012"
        ),
        "G1_ARCHITECTURE_SELECTOR": str(selector),
    }

    result = subprocess.run(
        ["bash", str(launcher_directory / "architecture/selected_500m.sh")],
        capture_output=True,
        text=True,
        env=environment,
    )

    assert (
        "SELECTOR ['ffn_gelu171:0.032:0.006', 'sequence_128:0.016:0.006'] "
        "['ffn_swiglu128:0.016:0.012']" in result.stderr
    )
    assert result.returncode == 2
    assert "Native-50M final-selection preflight failed" in result.stderr
    assert "G1_TEST_QUEUE_STUB_SOURCED" not in result.stderr


def test_exploratory_treatment_requires_a_run_tag_of_its_own() -> None:
    environment = os.environ | {
        "G1_TRAINING_QUEUE_LIBRARY": str(QUEUE_STUB),
        "G1_TEST_GLOBAL_BATCH_VERIFIER": str(GLOBAL_BATCH_STUB),
        "G1_GLOBAL_BATCH_SIZE": "1280",
        "G1_GLOBAL_BATCH_SELECTION": "control/control:0.016:0.006:1280",
        "G1_FINAL_AXES": "ffn",
        "G1_FINAL_TREATMENTS": "ffn/gelu171 ffn/swiglu128",
        "G1_FINAL_EXPLORATORY_TREATMENTS": "ffn/swiglu128",
        "G1_FINAL_SELECTIONS": (
            "control/control:0.016:0.006 "
            "ffn/gelu171:0.032:0.006 ffn/swiglu128:0.016:0.012"
        ),
    }

    result = subprocess.run(
        ["bash", str(FINAL_LAUNCHER)],
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 2
    assert "G1_FINAL_RUN_TAG" in result.stderr
    assert "G1_TEST_QUEUE_STUB_SOURCED" not in result.stderr


def test_exploratory_treatment_must_be_among_the_selected_treatments() -> None:
    environment = os.environ | {
        "G1_TRAINING_QUEUE_LIBRARY": str(QUEUE_STUB),
        "G1_TEST_GLOBAL_BATCH_VERIFIER": str(GLOBAL_BATCH_STUB),
        "G1_GLOBAL_BATCH_SIZE": "1280",
        "G1_GLOBAL_BATCH_SELECTION": "control/control:0.016:0.006:1280",
        "G1_FINAL_AXES": "ffn",
        "G1_FINAL_TREATMENTS": "ffn/gelu171",
        "G1_FINAL_EXPLORATORY_TREATMENTS": "ffn/swiglu128",
        "G1_FINAL_RUN_TAG": "widthprobe",
        "G1_FINAL_SELECTIONS": (
            "control/control:0.016:0.006 ffn/gelu171:0.032:0.006"
        ),
    }

    result = subprocess.run(
        ["bash", str(FINAL_LAUNCHER)],
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 2
    assert "outside the selected treatments" in result.stderr
    assert "G1_TEST_QUEUE_STUB_SOURCED" not in result.stderr
