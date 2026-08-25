import json
import re
import runpy
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest


COLLECT = (
    Path(__file__).resolve().parents[3]
    / "experiments/g1_sasrec_item_ids_likes/analysis/collect.py"
)
SCRATCHPAD = COLLECT.parent.parent / "scratchpad"
READER_REPORT = COLLECT.parent.parent / "README.md"
REPORT_RQ_ORDER = (1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 2)


def _write_run(
    root: Path,
    name: str,
    recall: float,
    *,
    dataset_size: str = "50m",
    embedding_learning_rate: float = 0.016,
    deep_learning_rate: float = 0.006,
) -> None:
    directory = root / "logs" / name
    directory.mkdir(parents=True)
    (directory / "final_metrics.json").write_text(
        json.dumps({"recall@100": recall, "ndcg@100": recall / 2})
    )
    (directory / "training_metadata.json").write_text(
        json.dumps(
            {
                "training_semantics_revision": 1,
                "dataset_size": dataset_size,
                "batch_size": 512,
                "embedding_learning_rate": embedding_learning_rate,
                "deep_learning_rate": deep_learning_rate,
                "transfer_invariants": {
                    "max_seq_len": 128,
                    "transformer": {
                        "learned_positions": "reverse",
                        "dropout": 0.1,
                    },
                },
            }
        )
    )


def test_full_tuning_report_keeps_only_readable_completed_results_and_bolds_best(
    monkeypatch, tmp_path: Path
) -> None:
    namespace = runpy.run_path(str(COLLECT))
    _write_run(
        tmp_path,
        "g1_rqtune_position_learned_reverse_e16d6_50m",
        0.12,
    )
    _write_run(
        tmp_path,
        "g1_rqtune_position_learned_reverse_e32d12_50m",
        0.13,
        embedding_learning_rate=0.032,
        deep_learning_rate=0.012,
    )
    _write_run(
        tmp_path,
        "g1_rqtune_position_alibi_e16d6_50m",
        0.11,
    )
    _write_run(
        tmp_path,
        "g1_rqtune_position_none_e16d6_50m",
        0.99,
    )
    _write_run(
        tmp_path,
        "g1_rqtune_position_learned_reverse_e8d3_50m",
        0.777,
        embedding_learning_rate=0.008,
        deep_learning_rate=0.003,
    )
    monkeypatch.setitem(
        namespace["load_report_runs"].__globals__, "GENERATED", tmp_path
    )
    monkeypatch.setitem(
        namespace["load_report_runs"].__globals__,
        "_run_status",
        lambda directory, *_: (
            "unusable" if directory.name.endswith("e8d3_50m") else "completed"
        ),
    )

    report = namespace["render_full_tuning_report"]("50m")

    assert "## RQ7" in report
    assert "### position learned reverse" in report
    assert "highest recall@100 displayed row" in report
    assert "| embedding learning rate | deep learning rate |" in report
    assert "**0.990**" in report
    assert "0.777" not in report
    assert "artifact" not in report
    assert "configuration" not in report
    assert "role" not in report
    assert "status" not in report
    assert "training semantics revision" not in report
    assert "experiment class" not in report


def test_report_loader_can_scope_artifact_verification(
    monkeypatch, tmp_path: Path
) -> None:
    namespace = runpy.run_path(str(COLLECT))
    _write_run(tmp_path, "g1_rqtune_position_none_e16d6_50m", 0.12)
    _write_run(tmp_path, "g1_rqtune_position_alibi_e16d6_50m", 0.13)
    monkeypatch.setitem(
        namespace["load_report_runs"].__globals__, "_run_status", lambda *_: "completed"
    )
    selected_directory = tmp_path / "logs/g1_rqtune_position_none_e16d6_50m"

    runs = namespace["load_report_runs"](
        "50m", directories=[selected_directory]
    )

    assert {run.name for run in runs} == {selected_directory.name}


def test_report_loader_materializes_each_control_alias_once(
    monkeypatch, tmp_path: Path
) -> None:
    namespace = runpy.run_path(str(COLLECT))
    _write_run(tmp_path, "g1_rqtune_architecture_control_e16d6_50m", 0.12)
    globals_ = namespace["load_report_runs"].__globals__
    monkeypatch.setitem(globals_, "GENERATED", tmp_path)
    monkeypatch.setitem(globals_, "_run_status", lambda *_: "completed")

    runs = namespace["load_report_runs"]("50m")
    aliases = [
        run
        for run in runs
        if run.configuration.startswith("combination_baseline_")
    ]

    assert len(aliases) == 1


def test_scratchpad_report_contains_questions_and_reader_tables_only(
    monkeypatch,
) -> None:
    namespace = runpy.run_path(str(COLLECT))
    globals_ = namespace["render_compact_report"].__globals__
    runs = [
        namespace["ReportRun"](
            name=f"rq{research_question}",
            configuration=f"rq{research_question}",
            dataset_size="50m",
            research_question=research_question,
            method=f"variant {research_question}",
            status="completed",
            metrics={},
            metadata={},
        )
        for research_question in range(1, 12)
    ]
    monkeypatch.setitem(globals_, "load_report_runs", lambda _: runs)
    monkeypatch.setitem(globals_, "_metric_bands", lambda **_: {})
    monkeypatch.setitem(globals_, "_validate_compact_coverage", lambda *_: None)
    monkeypatch.setitem(
        globals_,
        "select_architecture_report_runs",
        lambda _, runs, __, **___: runs,
    )
    monkeypatch.setitem(
        globals_,
        "select_rq4_report_runs",
        lambda _, runs, __: [run for run in runs if run.research_question == 4],
    )
    monkeypatch.setitem(
        globals_,
        "select_negative_report_runs",
        lambda _, runs, __: [run for run in runs if run.research_question == 11],
    )
    monkeypatch.setitem(globals_, "select_homework_negative_controls", lambda *_: [])
    monkeypatch.setitem(
        globals_,
        "_rq1_compact_tables",
        lambda *_: "| width | recall@100 |\n| --- | --- |",
    )
    monkeypatch.setitem(
        globals_,
        "_rq4_table",
        lambda *_: "| configuration | recall@100 |\n| --- | --- |",
    )
    monkeypatch.setitem(
        globals_, "_rq5_tables", lambda *_: "| scheduler | recall@100 |\n| --- | --- |"
    )
    monkeypatch.setitem(
        globals_, "_rq6_table", lambda *_: "| schedule | recall@100 |\n| --- | --- |"
    )
    monkeypatch.setitem(
        globals_,
        "_rq8_compact_tables",
        lambda *_: "| dimension | recall@100 |\n| --- | --- |",
    )
    monkeypatch.setitem(
        globals_, "_rq2_table", lambda *_: "| variant | recall@100 |\n| --- | --- |"
    )
    monkeypatch.setitem(
        globals_, "_rq3_table", lambda *_: "| variant | epoch time |\n| --- | --- |"
    )
    monkeypatch.setitem(
        globals_,
        "_rq10_table",
        lambda *_: "| item embeddings | recall@100 |\n| --- | --- |",
    )
    monkeypatch.setitem(
        globals_,
        "_rq11_table",
        lambda *_: "| negative sampling | recall@100 |\n| --- | --- |",
    )
    monkeypatch.setitem(
        globals_,
        "_reader_report_table",
        lambda *_, **__: "| variant | recall@100 |\n| --- | --- |",
    )

    report = namespace["render_compact_report"]("50m")

    assert report.startswith("# G1 — Yambda-50M results\n")
    assert report.count("## RQ") == 9
    assert [
        line.split(" ", 2)[1]
        for line in report.splitlines()
        if line.startswith("## RQ")
    ] == [
        f"RQ{research_question}"
        for research_question in namespace["REPORT_RQ_ORDER"]
        if research_question not in {5, 11}
    ]
    assert "| dimension | recall@100 |" in report
    assert "| scheduler | recall@100 |" not in report
    assert "best configuration" not in report
    assert "| role |" not in report
    assert "| status |" not in report
    assert "Historical generated snapshot" not in report
    assert "Green/red changes" not in report
    assert "analysis" not in report.lower()
    assert "conclusion" not in report.lower()


def test_compact_report_omits_questions_without_completed_rows(monkeypatch) -> None:
    namespace = runpy.run_path(str(COLLECT))
    report_run = namespace["ReportRun"]
    runs = [
        report_run(
            name="rq1-completed",
            configuration="rq1-completed",
            dataset_size="50m",
            research_question=1,
            method="completed",
            status="completed",
            metrics={"recall@100": 0.1},
            metadata={},
        ),
        report_run(
            name="rq6-unusable",
            configuration="schedule_constant",
            dataset_size="50m",
            research_question=6,
            method="constant warmup",
            status="unusable",
            metrics={"recall@100": 0.2},
            metadata={},
        ),
    ]
    globals_ = namespace["render_compact_report"].__globals__
    monkeypatch.setitem(globals_, "load_report_runs", lambda *_: runs)
    monkeypatch.setitem(globals_, "_metric_bands", lambda **_: {})
    monkeypatch.setitem(
        globals_,
        "_rq1_compact_tables",
        lambda *_: "| width | recall@100 |\n| --- | --- |\n| 64 | 0.1 |",
    )

    report = namespace["render_compact_report"]("50m")

    assert "## RQ1 —" in report
    assert "## RQ6 —" not in report


def test_compact_report_does_not_route_rq4_ffn_runs_to_rq8(
    monkeypatch,
) -> None:
    namespace = runpy.run_path(str(COLLECT))
    report_run = namespace["ReportRun"]
    runs = [
        report_run(
            name=f"rq{research_question}",
            configuration=f"rq{research_question}",
            dataset_size="500m",
            research_question=research_question,
            method=f"variant {research_question}",
            status="completed",
            metrics={},
            metadata={},
        )
        for research_question in range(1, 12)
    ]
    rq4 = [
        report_run(
            name=base,
            configuration=base,
            dataset_size="500m",
            research_question=4,
            method=f"ffn {family}",
            status="completed",
            metrics={"recall@100": recall},
            metadata={
                "embedding_learning_rate": 0.016,
                "deep_learning_rate": 0.006,
            },
        )
        for base, family, recall in (
            ("ffn_gelu171", "gelu", 0.12),
            ("ffn_swiglu128", "swiglu", 0.13),
        )
    ]
    globals_ = namespace["render_compact_report"].__globals__
    monkeypatch.setitem(globals_, "load_report_runs", lambda *_: runs)
    monkeypatch.setitem(globals_, "_metric_bands", lambda **_: {})
    monkeypatch.setitem(globals_, "_validate_compact_coverage", lambda *_: None)
    monkeypatch.setitem(
        globals_,
        "select_architecture_report_runs",
        lambda *_, bases, **__: runs,
    )
    monkeypatch.setitem(globals_, "select_rq4_report_runs", lambda *_: rq4)
    monkeypatch.setitem(
        globals_,
        "select_negative_report_runs",
        lambda *_: [run for run in runs if run.research_question == 11],
    )
    monkeypatch.setitem(globals_, "select_homework_negative_controls", lambda *_: [])
    monkeypatch.setitem(
        globals_,
        "_load_rq5_report_bundle",
        lambda: SimpleNamespace(
            reader_markdown="## RQ5 — current\n\n| scheduler |\n| --- |"
        ),
    )
    monkeypatch.setitem(
        globals_, "_load_rq8_report_bundle", lambda: SimpleNamespace()
    )
    monkeypatch.setitem(
        globals_,
        "_require_rq8_reader_tables",
        lambda _: "| query objective |\n| --- |",
    )
    for name in (
        "_rq1_compact_tables",
        "_rq2_table",
        "_rq3_table",
        "_rq5_tables",
        "_rq6_table",
        "_rq9_table",
        "_rq10_table",
        "_rq11_table",
        "_reader_report_table",
    ):
        monkeypatch.setitem(globals_, name, lambda *_, **__: "| x |\n| --- |")
    seen: dict[str, set[str]] = {}

    def record(name: str, table_runs) -> str:
        seen[name] = {
            run.configuration
            for run in table_runs
            if run.configuration.startswith("ffn_")
        }
        return "| configuration |\n| --- |"

    monkeypatch.setitem(
        globals_, "_rq4_table", lambda runs, candidates=None: record("rq4", runs)
    )
    monkeypatch.setitem(
        globals_, "_rq8_compact_tables", lambda runs: record("rq8", runs)
    )

    namespace["render_compact_report"]("500m")

    assert seen == {"rq8": set()}


def test_native_500m_compact_report_includes_corrected_rq8_reader_tables(
    monkeypatch,
) -> None:
    namespace = runpy.run_path(str(COLLECT))
    report_run = namespace["ReportRun"]
    rq8_run = report_run(
        name="rq8-dimension-control",
        configuration="dimension_64",
        dataset_size="500m",
        research_question=8,
        method="dimension 64",
        status="completed",
        metrics={"recall@100": 0.135, "ndcg@100": 0.052},
        metadata={},
    )
    globals_ = namespace["render_compact_report"].__globals__
    monkeypatch.setitem(
        globals_, "load_report_runs", lambda *_args, **_kwargs: [rq8_run]
    )
    monkeypatch.setitem(globals_, "_metric_bands", lambda **_: {})
    monkeypatch.setitem(
        globals_,
        "select_architecture_report_runs",
        lambda *_args, **_kwargs: [rq8_run],
    )
    monkeypatch.setitem(
        globals_, "_RQ8_AXIS_SPECS", (("dimension", (), (), {"dimension_64"}),)
    )
    monkeypatch.setitem(
        globals_,
        "_reader_report_table",
        lambda *_args, **_kwargs: "| dimension | recall@100 |\n| --- | ---: |\n| 64 | 0.135 |",
    )
    monkeypatch.setitem(
        globals_,
        "_load_rq5_report_bundle",
        lambda: SimpleNamespace(reader_markdown="## RQ5 — current\n\n| scheduler |\n| --- |"),
    )
    monkeypatch.setitem(
        globals_,
        "_load_rq8_report_bundle",
        lambda: SimpleNamespace(
            reader_markdown=(
                "## RQ8 — current\n\n"
                "| query objective | recall@100 |\n"
                "| --- | ---: |\n"
                "| standard item-state | 0.135 |\n"
                "| **end-only CLS** | +11% (0.149) |\n"
                "| interleaved CLS | -1% (0.133) |\n\n"
                "| causal ALiBi retained history length | recall@100 |\n"
                "| ---: | ---: |\n"
                + "".join(
                    f"| {length} | 0.135 |\n"
                    for length in (12, 25, 50, 100, 128, 200, 256, 512)
                )
                + "\n| reverse-RoPE + ALiBi retained history length | recall@100 |\n"
                "| ---: | ---: |\n"
                + "".join(
                    f"| {length} | 0.135 |\n"
                    for length in (12, 25, 50, 100, 128, 200, 256, 512)
                )
            )
        ),
    )

    report = namespace["render_compact_report"]("500m")

    assert report.count("| query objective |") == 1
    assert report.count("retained history length") == 2


def test_native_500m_compact_report_rejects_missing_corrected_rq8_table() -> None:
    namespace = runpy.run_path(str(COLLECT))
    bundle = SimpleNamespace(
        reader_markdown=(
            "## RQ8 — current\n\n"
            "| query objective | recall@100 |\n"
            "| --- | ---: |\n"
            "| standard item-state | 0.135 |\n"
        )
    )

    with pytest.raises(ValueError, match="three-table draft is malformed"):
        namespace["_require_rq8_reader_tables"](bundle)


def test_rq8_writer_replaces_only_native_500m_rq8(
    tmp_path: Path,
) -> None:
    namespace = runpy.run_path(str(COLLECT))
    report = (
        "# report\n\n"
        "## RQ7 — positions\n\n| old 7 |\n| --- |\n\n"
        "## RQ8 — architecture\n\n"
        "| query objective | recall@100 |\n| --- | ---: |\n"
        "| standard item-state | 0.100 |\n"
        "| end-only CLS | 0.110 |\n"
        "| interleaved CLS | 0.100 |\n\n"
        "## RQ9 — time\n\n| old 9 |\n| --- |\n"
    )
    path = tmp_path / "research_questions_500m.md"
    path.write_text(report)
    dedicated = (
        "## RQ8 — architecture\n\n"
        "| query objective | recall@100 |\n| --- | ---: |\n"
        "| standard item-state | 0.135 |\n"
        "| end-only CLS | 0.149 |\n"
        "| interleaved CLS | 0.133 |\n\n"
    )
    for label in (
        "causal ALiBi retained history length",
        "reverse-RoPE + ALiBi retained history length",
    ):
        dedicated += f"| {label} | recall@100 |\n| ---: | ---: |\n"
        dedicated += "".join(
            f"| {length} | 0.135 |\n"
            for length in (12, 25, 50, 100, 128, 200, 256, 512)
        )
        dedicated += "\n"
    (tmp_path / "rq8_reinvestigation_reader_500m.md").write_text(dedicated)

    namespace["write_rq8_report"](tmp_path)

    rendered = path.read_text()
    assert rendered.count("retained history length") == 2
    assert "| end-only CLS | 0.149 |" in rendered
    assert "| end-only CLS | 0.110 |" not in rendered
    assert "| old 7 |" in rendered
    assert "| old 9 |" in rendered


def test_checked_in_scratchpad_reports_are_table_only_readme_drafts() -> None:
    for dataset_size in ("50m", "500m"):
        report = (SCRATCHPAD / f"research_questions_{dataset_size}.md").read_text()
        headings = re.findall(r"(?m)^## RQ\d+ — .+$", report)

        research_questions = [heading.split(" ", 2)[1] for heading in headings]
        assert research_questions
        assert research_questions == [
            f"RQ{research_question}"
            for research_question in REPORT_RQ_ORDER
            if f"RQ{research_question}" in research_questions
        ]
        assert report.startswith(f"# G1 — Yambda-{dataset_size.upper()} results\n")
        assert report.count("\n| ---") >= len(headings)
        assert "Historical generated snapshot" not in report
        assert "best configuration" not in report
        assert "reference configuration" not in report
        assert "current-protocol rerun" not in report
        assert "blocked on accepted component selections" not in report
        assert "| role |" not in report
        assert "| status |" not in report
        assert all(
            not line or line.startswith(("# G1 — ", "## RQ", "|"))
            for line in report.splitlines()
        )
        for section in re.split(r"(?m)^## RQ\d+ — .+$", report)[1:]:
            lines = section.splitlines()
            table_axes = [
                line.split("|")[1].strip()
                for line, following in zip(lines, lines[1:])
                if line.startswith("| ") and following.startswith("| ---")
            ]
            assert len(table_axes) == len(set(table_axes))

    reader_lines = READER_REPORT.read_text().splitlines()
    headers = [
        line
        for line, following in zip(reader_lines, reader_lines[1:])
        if line.startswith("| ") and following.startswith("| ---")
    ]
    assert not [
        header
        for header in headers
        if re.search(r"(?i)\bruns?(?:/configuration)?\b", header)
    ]


def test_rq1_table_schema_matches_reader_and_compact_reports() -> None:
    headers = []
    for report in (
        READER_REPORT.read_text(),
        (SCRATCHPAD / "research_questions_50m.md").read_text(),
        (SCRATCHPAD / "research_questions_500m.md").read_text(),
    ):
        section = re.search(
            r"(?ms)^## RQ1 — .*?(?=^## RQ\d+ — |\Z)", report
        ).group(0)
        lines = section.splitlines()
        headers.append(
            [
                line
                for line, following in zip(lines, lines[1:])
                if line.startswith("| ") and following.startswith("| ---")
            ]
        )
        assert "selection digest" not in section

    reader, compact_50m, compact_500m = headers
    assert compact_50m[0] == reader[-1]
    assert compact_500m[0] == reader[-1]


def test_reader_report_contains_no_placeholder_result_sections() -> None:
    report = READER_REPORT.read_text()

    assert "current-protocol rerun" not in report
    assert "blocked on" not in report.lower()
    assert not re.search(r"(?m)^\|[^|]+\|(?:\s*—\s*\|)+$", report)

    sections = re.findall(
        r"(?ms)^## RQ\d+ — .*?(?=^## RQ\d+ — |\Z)", report
    )
    assert sections
    for section in sections:
        table_rows = [line for line in section.splitlines() if line.startswith("|")]
        assert len(table_rows) >= 3


def test_reader_reports_keep_performance_columns_in_rq3_only() -> None:
    for path in (
        READER_REPORT,
        SCRATCHPAD / "research_questions_50m.md",
        SCRATCHPAD / "research_questions_500m.md",
    ):
        report = path.read_text()
        for match in re.finditer(
            r"(?ms)^## RQ(?P<number>\d+) — .*?(?=^## RQ\d+ — |\Z)", report
        ):
            table_lines = "\n".join(
                line for line in match.group(0).splitlines() if line.startswith("|")
            ).lower()
            has_performance = any(
                label in table_lines
                for label in ("epoch time", "peak memory", "embedding parameters")
            )
            assert not has_performance or match.group("number") == "3"


def test_reader_report_does_not_promote_an_unverified_combination() -> None:
    reports = [
        READER_REPORT.read_text().lower(),
        *(
            (SCRATCHPAD / f"research_questions_{dataset_size}.md")
            .read_text()
            .lower()
            for dataset_size in ("50m", "500m")
        ),
    ]

    for report in reports:
        assert "future baseline" not in report
        assert "baseline for later experiment groups" not in report
        assert "final combination" not in report
        if "## rq2" in report and "## rq11" in report:
            assert report.index("## rq2") > report.index("## rq11")

    reader = reports[0]
    for phrase in (
        "use the tuned random-plus-rmsnorm",
        "keep swiglu",
        "use linear decay",
        "keep dim 64",
    ):
        assert phrase not in reader


def test_accepted_component_tables_share_reader_schemas() -> None:
    reports = [
        READER_REPORT.read_text(),
        *(
            (SCRATCHPAD / f"research_questions_{dataset_size}.md").read_text()
            for dataset_size in ("50m", "500m")
        ),
    ]
    for research_question in (5, 6, 7, 9, 10, 11):
        headers = []
        for report in reports:
            match = re.search(
                rf"(?ms)^## RQ{research_question} — .*?(?=^## RQ\d+ — |\Z)",
                report,
            )
            if match is None:
                continue
            section = match.group(0)
            headers.append(
                next(line for line in section.splitlines() if line.startswith("|"))
            )
        assert len(set(headers)) <= 1


def test_report_writer_creates_both_dataset_tables_and_full_proxy_tuning(
    monkeypatch, tmp_path: Path
) -> None:
    namespace = runpy.run_path(str(COLLECT))
    output = tmp_path / "scratchpad"
    globals_ = namespace["write_automated_reports"].__globals__
    monkeypatch.setitem(
        globals_, "render_compact_report", lambda size: f"{size} report\n"
    )
    monkeypatch.setitem(
        globals_, "render_full_tuning_report", lambda size: f"{size} tuning\n"
    )

    namespace["write_automated_reports"](output)

    assert (output / "research_questions_50m.md").exists()
    assert (output / "research_questions_500m.md").exists()
    assert (output / "hyperparameter_tuning_50m.md").exists()


def test_focused_rq11_writer_preserves_other_questions(
    monkeypatch, tmp_path: Path
) -> None:
    namespace = runpy.run_path(str(COLLECT))
    output = tmp_path / "scratchpad"
    output.mkdir()
    original = (
        "# report\n\n"
        "## RQ10 — previous\n\n| old |\n\n"
        "## RQ11 — stale\n\n| stale |\n"
    )
    for name in (
        "research_questions_50m.md",
        "research_questions_500m.md",
        "hyperparameter_tuning_50m.md",
    ):
        (output / name).write_text(original)
    globals_ = namespace["write_rq11_reports"].__globals__
    monkeypatch.setitem(
        globals_,
        "render_compact_question",
        lambda size, _: f"## RQ11 — current {size}\n\n| current |",
    )
    monkeypatch.setitem(
        globals_,
        "render_full_tuning_question",
        lambda size, _: f"## RQ11 — tuning {size}\n\n| tuning |",
    )

    namespace["write_rq11_reports"](output)

    assert "## RQ10 — previous\n\n| old |" in (
        output / "research_questions_50m.md"
    ).read_text()
    assert "## RQ11" not in (output / "research_questions_50m.md").read_text()
    assert "## RQ11 — current 500m\n\n| current |" in (
        output / "research_questions_500m.md"
    ).read_text()
    tuning = (output / "hyperparameter_tuning_50m.md").read_text()
    assert "## RQ10 — previous\n\n| old |" in tuning
    assert "## RQ11" not in tuning


def test_shared_reader_sync_leaves_dedicated_rq7_and_rq11_sections_intact(
    monkeypatch, tmp_path: Path
) -> None:
    namespace = runpy.run_path(str(COLLECT))
    output = tmp_path / "scratchpad"
    output.mkdir()
    generated = (
        "# generated\n\n"
        "## RQ7 — generated\n\n"
        "| replacement r7 | value |\n| --- | --- |\n| row | 1 |\n\n"
        "## RQ11 — generated\n\n"
        "| replacement r11 | value |\n| --- | --- |\n| row | 1 |\n"
    )
    (output / "research_questions_500m.md").write_text(generated)
    readme = tmp_path / "README.md"
    original = (
        "# reader\n\n"
        "## RQ7 — reader\n\n"
        "### Earlier broad position-encoding comparison\n\n"
        "| historical r7 | value |\n| --- | --- |\n| row | 1 |\n\n"
        "## RQ11 — reader\n\n"
        "### Earlier broad negative-sampling comparison\n\n"
        "| historical r11 | value |\n| --- | --- |\n| row | 1 |\n"
    )
    readme.write_text(original)
    globals_ = namespace["sync_reader_tables"].__globals__
    monkeypatch.setitem(globals_, "READER_REPORT", readme)

    namespace["sync_reader_tables"](output)

    assert readme.read_text() == original


def test_focused_rq11_reader_uses_selection_complete_native_500m_table(
    monkeypatch,
) -> None:
    namespace = runpy.run_path(str(COLLECT))
    table = (
        "| negative sampling | recall@100 |\n"
        "| --- | ---: |\n"
        "| uniform | 0.136 |\n"
        "| streaming | 0.134 |\n"
        "| popularity | 0.137 |\n"
        "| mixture | 0.136 |\n"
    )
    globals_ = namespace["render_compact_question"].__globals__
    monkeypatch.setitem(globals_, "load_report_runs", lambda *_args, **_kwargs: [])
    monkeypatch.setitem(
        globals_, "select_negative_report_runs", lambda *_args, **_kwargs: []
    )
    monkeypatch.setitem(
        globals_, "select_homework_negative_controls", lambda *_args, **_kwargs: []
    )
    historical = (
        "| negative sampling | recall@100 |\n"
        "| --- | ---: |\n"
        "| uniform random + fixed logQ on in-batch negatives | 0.129 |\n"
    )
    monkeypatch.setitem(globals_, "_rq11_table", lambda _runs: historical)
    monkeypatch.setitem(
        globals_,
        "_load_rq11_report_bundle",
        lambda: SimpleNamespace(
            evidence={"claims_status": "ready"}, reader_markdown=table
        ),
    )

    section = namespace["render_compact_question"]("500m", 11)

    assert "How do online logQ, offline logQ, random, mixed" in section
    assert "### Earlier broad negative-sampling comparison" in section
    assert historical.strip() in section
    assert "### Corrected uniform/streaming mixture comparison" in section
    assert table.strip() in section
    with pytest.raises(ValueError, match="native-500M only"):
        namespace["render_compact_question"]("50m", 11)


def test_current_tagged_negative_names_share_one_method_table(
    monkeypatch, tmp_path: Path
) -> None:
    namespace = runpy.run_path(str(COLLECT))
    for name, recall, embedding, deep in [
        (
            "g1_rqtune_neg_fixed_inbatch_global_q_yi2019_" "initial_e0p008_d0p003_50m",
            0.12,
            0.008,
            0.003,
        ),
        (
            "g1_rqtune_neg_fixed_inbatch_global_q_yi2019_"
            "secondary_e0p016_d0p006_n1024_50m",
            0.13,
            0.016,
            0.006,
        ),
    ]:
        _write_run(
            tmp_path,
            name,
            recall,
            embedding_learning_rate=embedding,
            deep_learning_rate=deep,
        )
    monkeypatch.setitem(
        namespace["load_report_runs"].__globals__, "GENERATED", tmp_path
    )
    monkeypatch.setitem(
        namespace["load_report_runs"].__globals__,
        "_run_status",
        lambda *_: "completed",
    )

    report = namespace["render_full_tuning_report"]("50m")
    unbolded = report.replace("**", "")

    assert report.count("### fixed in-batch global-q Yi-2019") == 1
    assert "| 0.008 | 0.003 |" in unbolded
    assert "| 0.016 | 0.006 |" in unbolded
    assert "initial_e0p008_d0p003" not in report
    assert "secondary_e0p016_d0p006_n1024" not in report


def test_report_loader_filter_does_not_leak_between_runs(
    monkeypatch, tmp_path: Path
) -> None:
    namespace = runpy.run_path(str(COLLECT))
    _write_run(
        tmp_path,
        "g1_rqtune_position_none_initial_e0p016_d0p006_50m",
        0.12,
    )
    _write_run(
        tmp_path,
        "g1_rqtune_neg_uniform_random_initial_r2_e0p016_d0p006_50m",
        0.13,
    )
    globals_ = namespace["load_report_runs"].__globals__
    monkeypatch.setitem(globals_, "GENERATED", tmp_path)
    monkeypatch.setitem(globals_, "_run_status", lambda *_: "completed")

    all_runs = namespace["load_report_runs"]("50m")
    negative_runs = namespace["load_report_runs"]("50m", research_question=11)

    assert {run.research_question for run in all_runs} == {7, 11}
    assert {run.research_question for run in negative_runs} == {11}


def test_accumulated_tuning_artifact_reconstructs_physical_batch_recipe() -> None:
    namespace = runpy.run_path(str(COLLECT))
    name = (
        "g1_rqtune_rqfinal_sequence_512_e0p128_d0p012_b1280_"
        "pb640_ga2_accumulation_pb640_ga2_ts2_r2_500m"
    )

    assignments = namespace["_tuning_assignments"](
        Path(name),
        name.removeprefix("g1_rqtune_rqfinal_").removesuffix("_500m"),
        "500m",
    )

    assert "G1_TUNE_BATCH_SIZE=640" in assignments
    assert "G1_TUNE_GRADIENT_ACCUMULATION_STEPS=2" in assignments


def test_report_selection_uses_effective_batch_for_accumulated_run() -> None:
    namespace = runpy.run_path(str(COLLECT))
    run = namespace["ReportRun"](
        name="sequence-512",
        configuration="sequence_512",
        dataset_size="500m",
        research_question=8,
        method="sequence length — 512",
        status="completed",
        metrics={"recall@100": 0.12},
        metadata={
            "batch_size": 640,
            "physical_batch_size": 640,
            "gradient_accumulation_steps": 2,
            "effective_batch_size": 1280,
        },
    )

    assert namespace["_run_batch_size"](run) == 1280


def test_rq8_component_selection_excludes_old_sequence_artifacts() -> None:
    namespace = runpy.run_path(str(COLLECT))

    assert not any(
        base.startswith("sequence_")
        for base in namespace["_component_bases"](8)
    )


def test_rq11_reader_table_shows_only_applicable_negative_axes(
    monkeypatch,
) -> None:
    namespace = runpy.run_path(str(COLLECT))
    report_run = namespace["ReportRun"]
    metrics = {
        "recall@100": 0.1,
        "ndcg@100": 0.04,
        "recall@10": 0.02,
        "ndcg@10": 0.015,
        "coverage@100": 0.5,
    }
    common_metadata = {
        "batch_size": 1280,
        "embedding_learning_rate": 0.1,
        "deep_learning_rate": 0.01,
        "transfer_invariants": {
            "num_in_batch_negatives": 512,
            "logq_alpha": 0.01,
            "random_negative_fraction": 0.5,
        },
    }
    control = report_run(
        "control",
        "neg_fixed_inbatch_leave_one_out",
        "500m",
        11,
        "fixed in-batch leave-one-out logQ",
        "completed",
        metrics,
        common_metadata,
    )
    random = report_run(
        "random",
        "neg_uniform_random",
        "500m",
        11,
        "uniform random",
        "completed",
        metrics | {"recall@100": 0.08},
        common_metadata,
    )
    mixed = report_run(
        "mixed",
        "neg_uniform_random_plus_fixed_logq_negative_only",
        "500m",
        11,
        "uniform random + fixed logQ on in-batch negatives",
        "completed",
        metrics | {"recall@100": 0.12},
        common_metadata
        | {
            "transfer_invariants": common_metadata["transfer_invariants"]
            | {"random_negative_fraction": 0.875}
        },
    )
    monkeypatch.setitem(
        namespace["_rq11_table"].__globals__,
        "_metric_bands",
        lambda **_: {metric: 0.001 for metric in namespace["REPORT_METRICS"]},
    )

    streaming = report_run(
        "streaming",
        "neg_streaming_inbatch_global_q_yi2019",
        "500m",
        11,
        "streaming in-batch global-q Yi-2019",
        "completed",
        metrics | {"recall@100": 0.11},
        common_metadata
        | {
            "transfer_invariants": common_metadata["transfer_invariants"]
            | {"logq_alpha": 0.005}
        },
    )
    monkeypatch.setitem(
        namespace["_rq11_table"].__globals__,
        "_metric_bands",
        lambda **_: {metric: 0.001 for metric in namespace["REPORT_METRICS"]},
    )

    table = namespace["_rq11_table"]([control, random, mixed, streaming])

    assert "| negative sampling | negatives | logQ alpha | random fraction |" in table
    assert "| uniform random | 512 | — | — |" in table
    assert "| fixed in-batch leave-one-out logQ | 512 | — | — |" in table
    assert "| streaming in-batch global-q Yi-2019 | 512 | 0.005 | — |" in table
    assert "| **uniform random + fixed logQ on in-batch negatives** | 512 | — | 0.875 |" in table
    assert '<span style="color: red">' in table
    assert '<span style="color: green">' in table


def test_rq11_homework_controls_have_report_identity_and_separate_table(
    monkeypatch,
) -> None:
    namespace = runpy.run_path(str(COLLECT))
    assert namespace["_report_identity"](
        "g1_homework_random_selected_e0p002_d0p004_cap40_ts2_r2_500m"
    ) == (
        11,
        "homework-matched uniform random",
        "homework_random_selected_e0p002_d0p004_cap40_ts2_r2",
    )
    report_run = namespace["ReportRun"]
    metrics = {
        "recall@100": 0.12,
        "ndcg@100": 0.04,
        "recall@10": 0.02,
        "ndcg@10": 0.015,
        "coverage@100": 0.5,
    }
    metadata = {
        "embedding_learning_rate": 0.001,
        "deep_learning_rate": 0.002,
        "transfer_invariants": {
            "num_in_batch_negatives": 512,
            "logq_alpha": 0.01,
            "random_negative_fraction": 0.5,
        },
    }
    main = report_run(
        "main",
        "neg_fixed_inbatch_leave_one_out",
        "500m",
        11,
        "fixed in-batch leave-one-out logQ",
        "completed",
        metrics,
        metadata,
    )
    homework_logq = replace(
        main,
        name="homework-logq",
        configuration="homework_logq_selected",
        method="homework-matched fixed leave-one-out logQ",
    )
    homework_random = replace(
        homework_logq,
        name="homework-random",
        configuration="homework_random_selected",
        method="homework-matched uniform random",
        metrics=metrics | {"recall@100": 0.13},
    )
    monkeypatch.setitem(
        namespace["_rq11_table"].__globals__,
        "_metric_bands",
        lambda **_: {metric: 0.001 for metric in namespace["REPORT_METRICS"]},
    )

    table = namespace["_rq11_table"]([main, homework_logq, homework_random])

    assert table.count("| negative sampling |") == 1
    assert table.count("| homework-matched objective |") == 1
    assert "| fixed leave-one-out logQ |" in table
    assert "| **uniform random** |" in table


def _homework_report_run(
    report_run,
    epochs: int,
    revision: int,
    status: str,
):
    tag = "initial" if epochs == 20 else "capcontinue"
    cap = "" if epochs == 20 else f"_cap{epochs}"
    return report_run(
        name=f"logq-{epochs}",
        configuration=(
            f"homework_logq_{tag}_e0p001_d0p002{cap}_ts2_r{revision}"
        ),
        dataset_size="50m",
        research_question=11,
        method="homework-matched fixed leave-one-out logQ",
        status=status,
        metrics={"recall@100": 0.1},
        metadata={
            "batch_size": 1280,
            "embedding_learning_rate": 0.001,
            "deep_learning_rate": 0.002,
            "max_epochs": epochs,
            "stopped_epoch": epochs if status == "unusable" else epochs - 2,
            "selection_resolved": status == "completed",
        },
    )


@pytest.mark.parametrize(
    "chain",
    [
        ((20, 1, "unusable"), (80, 3, "completed")),
        ((20, 1, "unusable"), (30, 2, "completed")),
        ((20, 1, "unusable"), (40, 3, "completed")),
    ],
)
def test_homework_report_selector_rejects_invalid_cap_lineage(chain) -> None:
    namespace = runpy.run_path(str(COLLECT))
    runs = [
        _homework_report_run(namespace["ReportRun"], *entry)
        for entry in chain
    ]

    with pytest.raises(ValueError, match="invalid cap lineage"):
        namespace["_latest_homework_configuration"]("logQ", runs)


def test_homework_report_selector_accepts_complete_cap_lineage() -> None:
    namespace = runpy.run_path(str(COLLECT))
    runs = [
        _homework_report_run(namespace["ReportRun"], 20, 1, "unusable"),
        _homework_report_run(namespace["ReportRun"], 40, 2, "unusable"),
        _homework_report_run(namespace["ReportRun"], 80, 3, "completed"),
    ]

    selected = namespace["_latest_homework_configuration"]("logQ", runs)

    assert selected.name == "logq-80"


def test_homework_control_verifier_assignments_reconstruct_exact_config(
    monkeypatch, tmp_path
) -> None:
    namespace = runpy.run_path(str(COLLECT))
    directory = (
        tmp_path
        / "g1_homework_random_selected_e0p002_d0p004_cap40_ts2_r2_500m"
    )

    assignments = namespace["_homework_negative_assignments"](
        directory,
        "homework_random_selected_e0p002_d0p004_cap40_ts2_r2",
        "500m",
        "random",
    )

    assert assignments == [
        "G1_HOMEWORK_RANDOM_RUN=selected_e0p002_d0p004_cap40_ts2_r2",
        "G1_HOMEWORK_RANDOM_EPOCHS=40",
        "G1_HOMEWORK_RANDOM_RUN_REVISION=2",
        "G1_HOMEWORK_RANDOM_EMBEDDING_LR=0.002",
        "G1_HOMEWORK_RANDOM_DEEP_LR=0.004",
        "G1_HOMEWORK_RANDOM_DATASET_SIZE=500m",
    ]
    directory.mkdir()
    verifier = namespace["_exact_artifact_matches"].__globals__["verify_artifact"]
    calls = []
    monkeypatch.setattr(
        verifier,
        "verify_config",
        lambda path, config, values: calls.append((path, config, values)) or True,
    )

    assert namespace["_exact_artifact_matches"](
        directory,
        "homework_random_selected_e0p002_d0p004_cap40_ts2_r2",
        "500m",
    )
    assert calls[0][1].name == "homework_random_control.py"
    assert calls[0][2] == assignments


def test_fixed_terminal_epoch_runs_are_unusable(monkeypatch, tmp_path: Path) -> None:
    namespace = runpy.run_path(str(COLLECT))
    globals_ = namespace["load_report_runs"].__globals__
    monkeypatch.setitem(globals_, "GENERATED", tmp_path)
    monkeypatch.setitem(globals_, "has_current_generation_semantics", lambda _: True)
    monkeypatch.setitem(globals_, "_exact_artifact_matches", lambda *_: True)

    for dataset_size, epochs in (("50m", 127), ("500m", 10)):
        name = f"g1_rqtune_position_alibi_e16d6_{dataset_size}"
        _write_run(tmp_path, name, 0.12, dataset_size=dataset_size)
        metadata_path = tmp_path / "logs" / name / "training_metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["num_epochs"] = epochs
        metadata["transfer_invariants"].update(
            experiment_class="MuTransferGenerationExperiment",
            eval_every_n_epochs=epochs,
            restore_best_weights=False,
        )
        metadata_path.write_text(json.dumps(metadata))
        metrics_path = tmp_path / "logs" / name / "final_metrics.json"
        metrics_path.touch()

        [run] = namespace["load_report_runs"](dataset_size)
        assert run.status == "unusable"


def test_report_requires_validation_selected_early_stopping_protocol(
    monkeypatch, tmp_path: Path
) -> None:
    namespace = runpy.run_path(str(COLLECT))
    globals_ = namespace["load_report_runs"].__globals__
    monkeypatch.setitem(globals_, "GENERATED", tmp_path)
    monkeypatch.setitem(globals_, "has_current_generation_semantics", lambda _: True)
    monkeypatch.setitem(globals_, "_exact_artifact_matches", lambda *_: True)
    name = "g1_rqtune_position_alibi_e16d6_50m"
    _write_run(tmp_path, name, 0.12)
    metadata_path = tmp_path / "logs" / name / "training_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata.update(
        num_epochs=30,
        max_epochs=30,
        epochs_trained=11,
        best_epoch=8,
        stopped_epoch=11,
        early_stopped=True,
        best_epoch_at_cap=False,
        selection_resolved=True,
    )
    metadata["transfer_invariants"].update(
        experiment_class="MuTransferGenerationExperiment",
        eval_every_n_epochs=1,
        restore_best_weights=True,
        early_stopping_patience=3,
        early_stopping_min_delta=0.0,
        early_stopping_metric="recall@100",
        early_stopping_metric_prefix="epoch/val_true",
    )
    metadata_path.write_text(json.dumps(metadata))
    (tmp_path / "logs" / name / "final_metrics.json").touch()

    [run] = namespace["load_report_runs"]("50m")
    assert run.status == "completed"

    metadata["transfer_invariants"]["early_stopping_patience"] = 4
    metadata_path.write_text(json.dumps(metadata))
    (tmp_path / "logs" / name / "final_metrics.json").touch()
    [run] = namespace["load_report_runs"]("50m")
    assert run.status == "unusable"


def test_collector_reconstructs_current_transfer_cap() -> None:
    namespace = runpy.run_path(str(COLLECT))

    assignments = namespace["_transfer_assignments"](
        Path("g1_transfer_power_e8e3_d3e3_t1500k_ts2_r2_50m"),
        "power_e8e3_d3e3_t1500k_ts2_r2",
        "50m",
    )

    assert "G1_TRANSFER_EPOCHS=20" in assignments
    assert "G1_TRANSFER_RUN=power_e8e3_d3e3_t1500k_ts2_r2" in assignments

    extended = namespace["_transfer_assignments"](
        Path("g1_transfer_batchscale_b1280_e0p008_d0p002_cap40_ts2_r2_50m"),
        "batchscale_b1280_e0p008_d0p002_cap40_ts2_r2",
        "50m",
    )
    assert "G1_TRANSFER_EPOCHS=40" in extended
    assert "G1_TRANSFER_RUN=batchscale_b1280_e0p008_d0p002_cap40_ts2_r2" in extended

    selected = namespace["_transfer_assignments"](
        Path(
            "g1_transfer_selected_native50_abcdef012345_e0p001_d0p002_"
            "cap40_ts2_r3_500m"
        ),
        "selected_native50_abcdef012345_e0p001_d0p002_cap40_ts2_r3",
        "500m",
    )
    assert "G1_TRANSFER_EPOCHS=40" in selected
    assert "G1_TRANSFER_RUN_REVISION=3" in selected
    assert "G1_TRANSFER_SOURCE_VARIANT=homework_fixed_leave_one_out" in selected


def test_collector_reconstructs_extended_tuning_cap_and_revision() -> None:
    namespace = runpy.run_path(str(COLLECT))
    configuration = "position_none_e16d6_boundary_cap30_ts2_r3"

    assignments = namespace["_tuning_assignments"](
        Path(f"g1_rqtune_{configuration}_50m"),
        configuration,
        "50m",
    )

    assert "G1_TUNE_EPOCHS=30" in assignments
    assert "G1_TUNE_RUN_REVISION=3" in assignments
    assert f"G1_TUNE_RUN={configuration}" in assignments


def test_collector_reconstructs_calibrated_cap_and_semantics(
    monkeypatch, tmp_path: Path
) -> None:
    namespace = runpy.run_path(str(COLLECT))
    verifier = namespace["_exact_artifact_matches"].__globals__["verify_artifact"]
    calls = []
    monkeypatch.setattr(
        verifier,
        "verify_config",
        lambda _directory, _config, assignments: calls.append(assignments) or True,
    )
    directory = tmp_path / (
        "g1_calibrated_baseline_cap30_ts2_50m_1000users_seed42_val2048_s3"
    )
    directory.mkdir()
    (directory / "final_metrics.json").write_text("{}")
    metadata = directory / "training_metadata.json"
    metadata.write_text("{}")

    assert namespace["_exact_artifact_matches"](
        directory, "baseline_cap30_ts2", "50m"
    )
    assert namespace["_exact_artifact_matches"](
        directory, "baseline_cap30_ts2", "50m"
    )
    assert calls == [
        [
            "G1_DATASET_SIZE=50m",
            "G1_VARIANT=baseline",
            "G1_MAX_EPOCHS=30",
            "G1_MAX_USERS=1000",
            "G1_VAL_BATCH_SIZE=2048",
            "G1_SEED=3",
        ]
    ]
    metadata.write_text('{"updated": true}')
    assert namespace["_exact_artifact_matches"](
        directory, "baseline_cap30_ts2", "50m"
    )
    assert len(calls) == 2
    assert not namespace["_exact_artifact_matches"](
        Path("g1_calibrated_baseline_50m"), "baseline", "50m"
    )


def test_rq1_500m_table_requires_native_confirmations(monkeypatch) -> None:
    namespace = runpy.run_path(str(COLLECT))
    report_run = namespace["ReportRun"]
    widths = (16, 32, 64, 128, 256)

    def run(width: int, dataset_size: str, recall: float):
        return report_run(
            name=f"mup_dim{width}_{dataset_size}",
            configuration=f"mup_dim{width}_e16d6_mup_r2",
            dataset_size=dataset_size,
            research_question=1,
            method="μP model-width transfer",
            status="completed",
            metrics={"recall@100": recall},
            metadata={
                "batch_size": 1280,
                "embedding_learning_rate": 0.016,
                "deep_learning_rate": 0.006,
            },
        )

    proxies = [run(width, "50m", 0.01 + width / 10000) for width in widths]
    targets = [run(width, "500m", 0.1 + width / 10000) for width in widths]
    monkeypatch.setitem(
        namespace["_rq1_compact_tables"].__globals__,
        "load_report_runs",
        lambda dataset_size, **_: proxies if dataset_size == "50m" else targets,
    )
    monkeypatch.setitem(
        namespace["_rq1_compact_tables"].__globals__,
        "_rq1_width_evidence",
        lambda _: ((0.016, 0.006), [(width, proxy, proxy) for width, proxy in zip(widths, proxies)]),
    )
    monkeypatch.setitem(
        namespace["_rq1_compact_tables"].__globals__,
        "_metric_bands",
        lambda **_: {metric: 0.001 for metric in namespace["REPORT_METRICS"]},
    )

    table = namespace["_rq1_compact_tables"]("500m", targets)

    assert "0.102" in table
    assert "0.012" not in table
    with pytest.raises(ValueError, match="500M confirmation"):
        namespace["_rq1_compact_tables"]("500m", targets[:-1])


def test_rq1_native_dataset_size_table_reports_recipe_and_stopping(
    monkeypatch,
) -> None:
    namespace = runpy.run_path(str(COLLECT))
    report_run = namespace["ReportRun"]
    common = {
        "research_question": 1,
        "method": "batch-scaling proxy, batch 1280",
        "status": "completed",
    }
    proxy = report_run(
        name="g1_transfer_batchscale_b1280_e0p001_d0p002_cap60_ts2_r2_50m",
        configuration="batchscale_b1280_e0p001_d0p002_cap60_ts2_r2",
        dataset_size="50m",
        metrics={"recall@100": 0.1002400236, "ndcg@100": 0.03746495},
        metadata={
            "batch_size": 1280,
            "embedding_learning_rate": 0.001,
            "deep_learning_rate": 0.002,
            "best_epoch": 56,
            "stopped_epoch": 59,
            "max_epochs": 60,
        },
        **common,
    )
    target = replace(
        proxy,
        name=(
            "g1_transfer_selected_native50_af8b8a8133c7_e0p001_d0p002_"
            "cap40_ts2_r2_500m"
        ),
        configuration=(
            "selected_native50_af8b8a8133c7_e0p001_d0p002_cap40_ts2_r2"
        ),
        dataset_size="500m",
        method="token-horizon response surface",
        metrics={"recall@100": 0.127361882, "ndcg@100": 0.047710985},
        metadata={
            "batch_size": 1280,
            "embedding_learning_rate": 0.001,
            "deep_learning_rate": 0.002,
            "best_epoch": 20,
            "stopped_epoch": 23,
            "max_epochs": 40,
        },
    )
    globals_ = namespace["render_compact_question"].__globals__
    selection = {
        "source_digest": (
            "af8b8a8133c769b1103e67add6bb520b9b3e50f94707302486a1764b69dac778"
        ),
        "source_id": "af8b8a8133c7",
        "source_artifacts": 42,
        "winner_run": proxy.name,
        "embedding_lr": "0.001",
        "deep_lr": "0.002",
    }
    monkeypatch.setattr(
        globals_["select_native_500m"],
        "select_native_500m",
        lambda _: selection,
    )
    monkeypatch.setattr(
        globals_["native_500m_provenance"],
        "validate",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setitem(
        globals_,
        "load_report_runs",
        lambda dataset_size, **_: [proxy] if dataset_size == "50m" else [target],
    )
    monkeypatch.setitem(globals_, "_metric_bands", lambda **_: {})

    section = namespace["render_compact_question"]("500m", 1)

    assert "Native dataset-size LR transfer (not μP width transfer)" not in section
    assert "selection digest" not in section
    assert "| 50M | 1280 | 0.001 | 0.002 | 56/59 | 60 |" in section
    assert "| 500M | 1280 | 0.001 | 0.002 | 20/23 | 40 |" in section
    assert "0.100" in section
    assert "0.037" in section
    assert "0.127" in section
    assert "0.048" in section

    reader = namespace["render_rq1_reader"]()
    normalized_reader = " ".join(reader.split())
    assert "same fixed-width conventional recipe and batch size" in normalized_reader
    assert "validates simple LR reuse from 50M to 500M" in normalized_reader
    assert "does not establish μP model-width transfer" in normalized_reader
    assert (
        "μP model-width conclusion remains work in progress" in normalized_reader
    )
    prose = [line for line in reader.splitlines() if not line.startswith("|")]
    assert max(map(len, prose)) <= 79

    forged_target = replace(
        target,
        name=target.name.replace("af8b8a8133c7", "deadbeef1234"),
        configuration=target.configuration.replace(
            "af8b8a8133c7", "deadbeef1234"
        ),
    )
    with pytest.raises(ValueError, match="source id"):
        namespace["_rq1_native_dataset_size_evidence"](
            [proxy], [forged_target]
        )

    partial_width = replace(
        proxy,
        name="mup_dim32_50m",
        configuration="mup_dim32_e0p001_d0p002",
        method="μP model-width transfer",
    )
    monkeypatch.setitem(
        globals_,
        "load_report_runs",
        lambda dataset_size, **_: (
            [proxy, partial_width] if dataset_size == "50m" else [target]
        ),
    )
    section = namespace["render_compact_question"]("500m", 1)
    assert "| dataset | batch size |" in section
    assert "**μP model-width transfer**" not in section

    failed_calibration = replace(
        target, metrics=target.metrics | {"recall@100": 0.14}
    )
    monkeypatch.setitem(
        globals_,
        "load_report_runs",
        lambda dataset_size, **_: (
            [proxy] if dataset_size == "50m" else [failed_calibration]
        ),
    )
    with pytest.raises(ValueError, match=r"must be in \[0.1235, 0.13\]"):
        namespace["render_compact_question"]("500m", 1)


def test_rq1_writer_updates_reader_and_both_scratchpads(
    monkeypatch, tmp_path: Path
) -> None:
    namespace = runpy.run_path(str(COLLECT))
    output = tmp_path / "scratchpad"
    output.mkdir()
    reader = tmp_path / "README.md"
    original = (
        "# report\n\n## RQ1 — stale\n\n| stale |\n\n"
        "## RQ2 — kept\n\n| kept |\n"
    )
    for name in (
        "research_questions_50m.md",
        "research_questions_500m.md",
        "hyperparameter_tuning_50m.md",
    ):
        (output / name).write_text(original)
    reader.write_text(original)
    globals_ = namespace["write_rq1_reports"].__globals__
    monkeypatch.setitem(globals_, "READER_REPORT", reader)
    monkeypatch.setitem(
        globals_,
        "render_compact_question",
        lambda size, _: f"## RQ1 — current {size}\n\n| native transfer |",
    )
    monkeypatch.setitem(
        globals_,
        "render_full_tuning_question",
        lambda size, _: f"## RQ1 — tuning {size}\n\n| tuning |",
    )
    monkeypatch.setitem(
        globals_,
        "render_rq1_reader",
        lambda: "## RQ1 — current reader\n\nreader narrative\n\n| native transfer |",
    )

    namespace["write_rq1_reports"](output)

    assert "## RQ1 — current 50m" in (
        output / "research_questions_50m.md"
    ).read_text()
    assert "## RQ1 — current 500m" in (
        output / "research_questions_500m.md"
    ).read_text()
    assert "## RQ1 — current reader" in reader.read_text()
    assert "reader narrative" in reader.read_text()
    assert "| native transfer |\n\n## RQ2 — kept\n\n| kept |" in reader.read_text()


def test_homework_calibration_uses_strict_reproduction_range() -> None:
    namespace = runpy.run_path(str(COLLECT))
    accepts = namespace["homework_recall_in_calibration_range"]

    assert accepts(0.1235)
    assert accepts(0.13)
    assert not accepts(0.123499)
    assert not accepts(0.130001)


def test_strict_gate_targets_homework_baseline_reproduction() -> None:
    namespace = runpy.run_path(str(COLLECT))
    run = namespace["ReportRun"](
        name="g1_calibrated_homework_baseline_500m",
        configuration="homework_baseline",
        dataset_size="500m",
        research_question=2,
        method="homework-compatible baseline",
        status="completed",
        metrics={"recall@100": 0.12},
        metadata={},
    )
    with pytest.raises(ValueError, match=r"must be in \[0.1235, 0.13\]"):
        namespace["validate_homework_reproduction_runs"]([run])

    passed = replace(
        run,
        metrics={"recall@100": 0.125},
    )
    namespace["validate_homework_reproduction_runs"]([passed])


def test_focused_compact_renderer_supports_rq1(monkeypatch) -> None:
    namespace = runpy.run_path(str(COLLECT))
    globals_ = namespace["render_compact_question"].__globals__
    monkeypatch.setitem(globals_, "load_report_runs", lambda *_args, **_kwargs: [])
    monkeypatch.setitem(globals_, "_rq1_compact_tables", lambda *_: "| width | recall@100 |\n| --- | --- |")

    section = namespace["render_compact_question"]("50m", 1)

    assert section.startswith("## RQ1 —")
    assert "| width | recall@100 |" in section


def test_reader_table_has_rq_specific_axis_and_no_ledger_columns(
    monkeypatch,
) -> None:
    namespace = runpy.run_path(str(COLLECT))
    make_run = lambda configuration, recall: namespace["ReportRun"](
        name=configuration,
        configuration=configuration,
        dataset_size="500m",
        research_question=8,
        method=configuration,
        status="completed",
        metrics={"recall@100": recall, "ndcg@100": recall / 2},
        metadata={"embedding_learning_rate": 0.032, "deep_learning_rate": 0.012},
    )
    runs = [make_run("dimension_64", 0.14), make_run("dimension_128", 0.15)]
    monkeypatch.setitem(
        namespace["_reader_report_table"].__globals__,
        "_metric_bands",
        lambda **_: {metric: 0.001 for metric in namespace["REPORT_METRICS"]},
    )

    table = namespace["_reader_report_table"](
        8,
        runs,
        control_patterns=("dimension_64",),
        axis_label="dimension",
    )

    assert table.startswith("| dimension | recall@100 |")
    assert "| 64 |" in table
    assert "| **128** |" in table
    assert "best configuration" not in table
    assert "role" not in table
    assert "status" not in table
    assert "runs" not in table


def test_rq6_table_reports_both_independently_tuned_learning_rates(
    monkeypatch,
) -> None:
    namespace = runpy.run_path(str(COLLECT))
    report_run = namespace["ReportRun"]
    runs = []
    for base, embedding, deep in (
        ("schedule_constant", 0.128, 0.012),
        ("schedule_constant_warmup5", 0.032, 0.012),
        ("schedule_cosine", 0.016, 0.012),
        ("schedule_cosine_warmup5_cycles1", 0.032, 0.006),
        ("schedule_inverse_sqrt", 0.016, 0.012),
        ("schedule_inverse_sqrt_warmup5", 0.032, 0.006),
    ):
        runs.append(
            report_run(
                name=base,
                configuration=base,
                dataset_size="50m",
                research_question=6,
                method=base,
                status="completed",
                metrics={"recall@100": 0.1, "ndcg@100": 0.04},
                metadata={
                    "embedding_learning_rate": embedding,
                    "deep_learning_rate": deep,
                },
            )
        )
    monkeypatch.setitem(
        namespace["_rq6_table"].__globals__,
        "_metric_bands",
        lambda **_: {"recall@100": 0.001, "ndcg@100": 0.0005},
    )

    table = namespace["_rq6_table"](runs)

    assert (
        "| schedule | no-warmup LR | warmup LR | no-warmup recall@100 | "
        "warmup=5% recall@100 | no-warmup ndcg@100 | warmup=5% ndcg@100 |"
        in table
    )
    assert (
        "| constant | 0.128/0.012 | 0.032/0.012 | 0.100 | 0% (0.100) | "
        "0.040 | 0% (0.040) |"
        in table.replace("**", "")
    )


def test_rq6_table_bolds_the_best_completed_warmup_row(monkeypatch) -> None:
    namespace = runpy.run_path(str(COLLECT))
    report_run = namespace["ReportRun"]
    recalls = {
        "schedule_constant": 0.11,
        "schedule_constant_warmup5": 0.12,
        "schedule_cosine": 0.13,
        "schedule_cosine_warmup5_cycles1": 0.14,
        "schedule_inverse_sqrt": 0.15,
        "schedule_inverse_sqrt_warmup5": 0.16,
    }
    runs = [
        report_run(
            name=base,
            configuration=base,
            dataset_size="50m",
            research_question=6,
            method=base,
            status="completed",
            metrics={"recall@100": recall, "ndcg@100": recall / 2},
            metadata={
                "embedding_learning_rate": 0.016,
                "deep_learning_rate": 0.006,
            },
        )
        for base, recall in recalls.items()
    ]
    monkeypatch.setitem(
        namespace["_rq6_table"].__globals__,
        "_metric_bands",
        lambda **_: {"recall@100": 0.001, "ndcg@100": 0.0005},
    )

    table = namespace["_rq6_table"](runs)

    assert "| **inverse sqrt** |" in table
    assert "| **constant** |" not in table


def test_tuning_ledger_rejects_reader_visible_duplicate_treatment_rows(
    monkeypatch,
) -> None:
    namespace = runpy.run_path(str(COLLECT))
    report_run = namespace["ReportRun"]
    metadata = {
        "embedding_learning_rate": 0.016,
        "deep_learning_rate": 0.006,
    }
    runs = [
        report_run(
            name=name,
            configuration=name,
            dataset_size="50m",
            research_question=7,
            method="learned positions",
            status="completed",
            metrics={"recall@100": recall, "ndcg@100": recall / 2},
            metadata=metadata,
        )
        for name, recall in (("encoded-run-a", 0.12), ("encoded-run-b", 0.13))
    ]
    monkeypatch.setitem(
        namespace["_report_table"].__globals__, "_metric_bands", lambda **_: {}
    )

    with pytest.raises(
        ValueError, match="ambiguous completed artifacts for reader configuration"
    ):
        namespace["_report_table"](
            runs,
            compact=False,
            research_question=7,
            control=None,
        )


def test_tuning_ledger_stage_precedence_ignores_alias_recall(
    monkeypatch,
) -> None:
    namespace = runpy.run_path(str(COLLECT))
    report_run = namespace["ReportRun"]
    metadata = {
        "embedding_learning_rate": 0.016,
        "deep_learning_rate": 0.006,
    }
    first = report_run(
        name="first",
        configuration="position_none_initial_e0p016_d0p006_ts2_r2",
        dataset_size="50m",
        research_question=7,
        method="learned positions",
        status="completed",
        metrics={"recall@100": 0.12, "ndcg@100": 0.06},
        metadata=metadata,
    )
    alias = replace(
        first,
        name="second",
        configuration="position_none_local_e0p016_d0p006_ts2_r2",
        metrics={"recall@100": 0.11, "ndcg@100": 0.055, "epoch_time": 0.5},
    )
    monkeypatch.setitem(
        namespace["_report_table"].__globals__, "_metric_bands", lambda **_: {}
    )

    table = namespace["_report_table"](
        [first, alias], compact=False, research_question=7, control=None
    )

    assert len([line for line in table.splitlines() if line.startswith("|")]) == 3
    assert "0.110" in table
    assert "0.120" not in table


def test_tuning_ledger_collapses_an_identically_repeated_launch(monkeypatch) -> None:
    namespace = runpy.run_path(str(COLLECT))
    report_run = namespace["ReportRun"]
    first = report_run(
        name="first",
        configuration="position_none_e0p016_d0p006_lrboundary2_ts2_r2",
        dataset_size="50m",
        research_question=7,
        method="learned positions",
        status="completed",
        metrics={"recall@100": 0.12, "ndcg@100": 0.06, "epoch_time": 0.47},
        metadata={
            "embedding_learning_rate": 0.016,
            "deep_learning_rate": 0.006,
            "seed": 42,
        },
    )
    repeated_launch = replace(
        first,
        name="second",
        configuration="position_none_e0p016_d0p006_lrboundary4_ts2_r2",
        metrics=first.metrics | {"epoch_time": 0.51},
    )
    monkeypatch.setitem(
        namespace["_report_table"].__globals__, "_metric_bands", lambda **_: {}
    )

    table = namespace["_report_table"](
        [first, repeated_launch], compact=False, research_question=7, control=None
    )

    assert len([line for line in table.splitlines() if line.startswith("|")]) == 3
    assert "0.120" in table


def test_tuning_ledger_prefers_a_cap_continuation_of_its_own_revision(
    monkeypatch,
) -> None:
    namespace = runpy.run_path(str(COLLECT))
    report_run = namespace["ReportRun"]
    initial = report_run(
        name="first",
        configuration="position_none_e0p016_d0p006_mup_ts2_r2",
        dataset_size="50m",
        research_question=7,
        method="learned positions",
        status="completed",
        metrics={"recall@100": 0.12, "ndcg@100": 0.06},
        metadata={
            "embedding_learning_rate": 0.016,
            "deep_learning_rate": 0.006,
            "transfer_invariants": {"random_negative_fraction": 0.5},
        },
    )
    continuation = replace(
        initial,
        name="second",
        configuration="position_none_e0p016_d0p006_capcont_cap40_ts2_r2",
        metrics={"recall@100": 0.13, "ndcg@100": 0.07},
    )
    monkeypatch.setitem(
        namespace["_report_table"].__globals__, "_metric_bands", lambda **_: {}
    )

    table = namespace["_report_table"](
        [initial, continuation], compact=False, research_question=7, control=None
    )

    assert "0.130" in table
    assert "0.120" not in table


def test_tuning_ledger_reads_a_renamed_continuation_as_the_same_lineage(
    monkeypatch,
) -> None:
    namespace = runpy.run_path(str(COLLECT))
    report_run = namespace["ReportRun"]
    initial = report_run(
        name="first",
        configuration="neg_uniform_random_initial_e0p016_d0p006_ts2_r2",
        dataset_size="50m",
        research_question=11,
        method="uniform random",
        status="completed",
        metrics={"recall@100": 0.12, "ndcg@100": 0.06},
        metadata={
            "embedding_learning_rate": 0.016,
            "deep_learning_rate": 0.006,
            "transfer_invariants": {"random_negative_fraction": 0.5},
        },
    )
    continuation = replace(
        initial,
        name="second",
        configuration="neg_uniform_random_capresolve1_e0p016_d0p006_cap40_ts2_r2",
        metrics={"recall@100": 0.13, "ndcg@100": 0.07},
    )
    monkeypatch.setitem(
        namespace["_report_table"].__globals__, "_metric_bands", lambda **_: {}
    )

    table = namespace["_report_table"](
        [initial, continuation], compact=False, research_question=11, control=None
    )

    assert "0.130" in table
    assert "0.120" not in table


def test_tuning_ledger_rejects_a_shorter_cap_at_a_later_revision(monkeypatch) -> None:
    namespace = runpy.run_path(str(COLLECT))
    report_run = namespace["ReportRun"]
    continuation = report_run(
        name="first",
        configuration="position_none_e0p016_d0p006_capcont_cap40_ts2_r2",
        dataset_size="50m",
        research_question=7,
        method="learned positions",
        status="completed",
        metrics={"recall@100": 0.12, "ndcg@100": 0.06},
        metadata={
            "embedding_learning_rate": 0.016,
            "deep_learning_rate": 0.006,
            "transfer_invariants": {"random_negative_fraction": 0.5},
        },
    )
    later = replace(
        continuation,
        name="second",
        configuration="position_none_e0p016_d0p006_horizon_ts2_r3",
        metrics={"recall@100": 0.13, "ndcg@100": 0.07},
    )
    monkeypatch.setitem(
        namespace["_report_table"].__globals__, "_metric_bands", lambda **_: {}
    )

    with pytest.raises(
        ValueError, match="ambiguous completed artifacts for reader configuration"
    ):
        namespace["_report_table"](
            [continuation, later], compact=False, research_question=7, control=None
        )


def test_tuning_ledger_rejects_multiple_artifacts_at_same_stage(monkeypatch) -> None:
    namespace = runpy.run_path(str(COLLECT))
    report_run = namespace["ReportRun"]
    first = report_run(
        name="first",
        configuration="position_none_local_e0p016_d0p006_ts2_r2",
        dataset_size="50m",
        research_question=7,
        method="learned positions",
        status="completed",
        metrics={"recall@100": 0.12, "ndcg@100": 0.06},
        metadata={"embedding_learning_rate": 0.016, "deep_learning_rate": 0.006},
    )
    repeated = replace(first, name="second", metrics={"recall@100": 0.13})
    monkeypatch.setitem(
        namespace["_report_table"].__globals__, "_metric_bands", lambda **_: {}
    )

    with pytest.raises(
        ValueError, match="ambiguous completed artifacts for reader configuration"
    ):
        namespace["_report_table"](
            [first, repeated], compact=False, research_question=7, control=None
        )


def test_tuning_ledger_collapses_cap_continuations_but_keeps_tuned_axes(
    monkeypatch,
) -> None:
    namespace = runpy.run_path(str(COLLECT))
    report_run = namespace["ReportRun"]

    def make_run(
        name: str,
        embedding_learning_rate: float,
        num_epochs: int,
        recall: float,
    ):
        return report_run(
            name=name,
            configuration=name,
            dataset_size="50m",
            research_question=7,
            method="learned positions",
            status="completed",
            metrics={"recall@100": recall, "ndcg@100": recall / 2},
            metadata={
                "embedding_learning_rate": embedding_learning_rate,
                "deep_learning_rate": 0.006,
                "num_epochs": num_epochs,
                "training_horizon": num_epochs * 1_000,
                "token_horizon": num_epochs * 2_000,
                "transfer_invariants": {"random_negative_fraction": 0.5},
            },
        )

    runs = [
        make_run("position_none_e16d6_ts2_r1", 0.016, 20, 0.12),
        make_run("position_none_e16d6_cap40_ts2_r2", 0.016, 40, 0.13),
        make_run("different-lr", 0.032, 40, 0.14),
    ]
    monkeypatch.setitem(
        namespace["_report_table"].__globals__, "_metric_bands", lambda **_: {}
    )

    table = namespace["_report_table"](
        runs,
        compact=False,
        research_question=7,
        control=None,
    )
    unbolded = table.replace("**", "")

    assert "num epochs" not in table
    assert "training horizon" not in table
    assert "token horizon" not in table
    assert len([line for line in table.splitlines() if line.startswith("|")]) == 4
    assert "| 0.016 | 0.006 |" in unbolded
    assert "| 0.032 | 0.006 |" in unbolded
    assert "0.130" in table
    assert "0.140" in table
    assert "0.120" not in table


def test_tuning_ledger_rejects_ambiguous_reader_identical_artifacts(
    monkeypatch,
) -> None:
    namespace = runpy.run_path(str(COLLECT))
    report_run = namespace["ReportRun"]
    metadata = {
        "embedding_learning_rate": 0.016,
        "deep_learning_rate": 0.006,
    }
    control = report_run(
        name="control",
        configuration="control",
        dataset_size="50m",
        research_question=7,
        method="control method",
        status="completed",
        metrics={"recall@100": 0.1, "ndcg@100": 0.05},
        metadata=metadata,
    )
    treatments = [
        replace(
            control,
            name=name,
            configuration=name,
            method="treatment method",
            metrics={"recall@100": recall, "ndcg@100": recall / 2},
        )
        for name, recall in (("treatment-a", 0.12), ("treatment-b", 0.13))
    ]
    monkeypatch.setitem(
        namespace["_report_table"].__globals__, "_metric_bands", lambda **_: {}
    )

    with pytest.raises(
        ValueError, match="ambiguous completed artifacts for reader configuration"
    ):
        namespace["_report_table"](
            treatments,
            compact=False,
            research_question=7,
            control=control,
        )


def test_tuning_ledger_rejects_repeated_artifacts_of_control_method(
    monkeypatch,
) -> None:
    namespace = runpy.run_path(str(COLLECT))
    report_run = namespace["ReportRun"]
    control = report_run(
        name="control",
        configuration="same-treatment",
        dataset_size="50m",
        research_question=7,
        method="same method",
        status="completed",
        metrics={"recall@100": 0.1, "ndcg@100": 0.05},
        metadata={
            "embedding_learning_rate": 0.016,
            "deep_learning_rate": 0.006,
        },
    )
    repeated_artifact = replace(
        control,
        name="repeat",
        metrics={"recall@100": 0.12, "ndcg@100": 0.06},
    )
    monkeypatch.setitem(
        namespace["_report_table"].__globals__, "_metric_bands", lambda **_: {}
    )

    with pytest.raises(
        ValueError, match="ambiguous completed artifacts for reader configuration"
    ):
        namespace["_report_table"](
            [control, repeated_artifact],
            compact=False,
            research_question=7,
            control=control,
        )


def test_reinvestigated_rq8_axes_are_not_promoted_from_old_artifacts() -> None:
    namespace = runpy.run_path(str(COLLECT))
    bases = set().union(
        *(set(spec[3]) for spec in namespace["_RQ8_AXIS_SPECS"])
    )

    assert not bases & {"cls_off", "cls_on"}
    assert not any(base.startswith("sequence_") for base in bases)


def test_rq4_table_matches_reader_ffn_family_schema(monkeypatch) -> None:
    namespace = runpy.run_path(str(COLLECT))
    report_run = namespace["ReportRun"]
    runs = [
        report_run(
            name=base,
            configuration=base,
            dataset_size="500m",
            research_question=4,
            method=f"ffn {family.lower()}",
            status="completed",
            metrics={
                "recall@100": recall,
                "ndcg@100": ndcg,
                "coverage@100": coverage,
            },
            metadata={
                "embedding_learning_rate": embedding,
                "deep_learning_rate": deep,
            },
        )
        for base, family, embedding, deep, recall, ndcg, coverage in (
            ("ffn_gelu171", "GELU", 0.128, 0.012, 0.131, 0.0491, 0.72),
            ("ffn_swiglu32", "SwiGLU", 0.064, 0.024, 0.130, 0.0494, 0.51),
        )
    ]
    monkeypatch.setitem(
        namespace["_rq4_table"].__globals__,
        "_metric_bands",
        lambda **_: {
            "recall@100": 0.00215,
            "ndcg@100": 0.000951,
            "coverage@100": 0.071,
        },
    )

    table = namespace["_rq4_table"](runs)

    assert (
        "| proxy-selected FFN family | selected width | recall@100 | "
        "ndcg@100 | coverage@100 |" in table
    )
    assert "| **GELU** | 171 | 0.131 | 0.049 | 0.720 |" in table
    assert "| SwiGLU | 32 |" in table


def test_current_rq8_selection_excludes_ffn_capacity_bases(monkeypatch) -> None:
    namespace = runpy.run_path(str(COLLECT))
    report_run = namespace["ReportRun"]
    selected = report_run(
        name="selected",
        configuration="dimension_64",
        dataset_size="50m",
        research_question=8,
        method="dimension",
        status="completed",
        metrics={},
        metadata={},
    )
    observed = {}
    globals_ = namespace["_render_current_component_question"].__globals__

    def select_architecture(*_, bases, **__):
        observed["bases"] = bases
        return [selected]

    monkeypatch.setitem(
        globals_, "select_architecture_report_runs", select_architecture
    )
    monkeypatch.setitem(globals_, "_rq8_compact_tables", lambda _: "complete RQ8")

    section = namespace["_render_current_component_question"](
        "50m", 8, [], []
    )

    assert namespace["_rq4_bases"]().isdisjoint(observed["bases"])
    assert section.endswith("complete RQ8")


def test_component_writer_preserves_unresolved_noncomponent_questions(
    monkeypatch, tmp_path: Path
) -> None:
    namespace = runpy.run_path(str(COLLECT))
    output = tmp_path / "scratchpad"
    output.mkdir()
    original = "# report\n"
    for research_question in range(1, 12):
        original += f"\n## RQ{research_question} — stale\n\n| stale {research_question} |\n"
    for dataset_size in ("50m", "500m"):
        (output / f"research_questions_{dataset_size}.md").write_text(original)
    (output / "hyperparameter_tuning_50m.md").write_text(original)
    globals_ = namespace["write_current_component_reports"].__globals__
    monkeypatch.setitem(
        globals_,
        "load_report_runs",
        lambda *_: [],
    )
    monkeypatch.setitem(globals_, "_metric_bands", lambda **_: {})
    monkeypatch.setitem(
        globals_,
        "_render_current_component_question",
        lambda dataset_size, research_question, *_: (
            f"## RQ{research_question} — current {dataset_size}\n\n"
            f"| current {research_question} |"
        ),
    )
    monkeypatch.setitem(
        globals_, "render_full_tuning_report", lambda _: "# current tuning\n"
    )

    namespace["write_current_component_reports"](output)

    for dataset_size in ("50m", "500m"):
        report = (output / f"research_questions_{dataset_size}.md").read_text()
        assert "## RQ1 — stale\n\n| stale 1 |" in report
        assert "## RQ3 — stale\n\n| stale 3 |" in report
        assert "## RQ4 — current" in report
        if dataset_size == "500m":
            assert "## RQ11 — current" in report
        else:
            assert "## RQ11" not in report
    assert (output / "hyperparameter_tuning_50m.md").read_text() == (
        "# current tuning\n"
    )


def test_component_writer_inserts_missing_rq4_in_canonical_order(
    monkeypatch, tmp_path: Path
) -> None:
    namespace = runpy.run_path(str(COLLECT))
    output = tmp_path / "scratchpad"
    output.mkdir()
    original = "# report\n" + "".join(
        f"\n## RQ{research_question} — stale\n\n| stale {research_question} |\n"
        for research_question in REPORT_RQ_ORDER
        if research_question != 4
    )
    for dataset_size in ("50m", "500m"):
        (output / f"research_questions_{dataset_size}.md").write_text(original)
    (output / "hyperparameter_tuning_50m.md").write_text("# stale tuning\n")
    globals_ = namespace["write_current_component_reports"].__globals__
    monkeypatch.setitem(globals_, "load_report_runs", lambda *_: [])
    monkeypatch.setitem(globals_, "_metric_bands", lambda **_: {})
    monkeypatch.setitem(
        globals_,
        "_render_current_component_question",
        lambda dataset_size, research_question, *_: (
            f"## RQ{research_question} — current {dataset_size}\n\n"
            f"| current {research_question} |"
        ),
    )
    monkeypatch.setitem(
        globals_, "render_full_tuning_report", lambda _: "# current tuning\n"
    )

    namespace["write_current_component_reports"](output)

    for dataset_size in ("50m", "500m"):
        report = (output / f"research_questions_{dataset_size}.md").read_text()
        headings = re.findall(r"(?m)^## RQ(?P<number>\d+) —", report)
        assert [int(number) for number in headings] == list(REPORT_RQ_ORDER)
        assert report.count("## RQ4 —") == 1
        assert all(
            not line or line.startswith(("# report", "## RQ", "|"))
            for line in report.splitlines()
        )


def test_question_writer_rejects_duplicate_existing_sections(tmp_path: Path) -> None:
    namespace = runpy.run_path(str(COLLECT))
    path = tmp_path / "report.md"
    path.write_text(
        "# report\n\n"
        "## RQ4 — first\n\n| first |\n\n"
        "## RQ4 — duplicate\n\n| duplicate |\n"
    )

    with pytest.raises(ValueError, match="duplicate RQ4 sections"):
        namespace["_replace_question"](
            path, 4, "## RQ4 — replacement\n\n| replacement |"
        )


def test_component_writer_reports_evidence_blockers_after_safe_updates(
    monkeypatch, tmp_path: Path
) -> None:
    namespace = runpy.run_path(str(COLLECT))
    output = tmp_path / "scratchpad"
    output.mkdir()
    original = "# report\n" + "".join(
        f"\n## RQ{research_question} — stale\n\n| stale {research_question} |\n"
        for research_question in range(1, 12)
    )
    for dataset_size in ("50m", "500m"):
        (output / f"research_questions_{dataset_size}.md").write_text(original)
    (output / "hyperparameter_tuning_50m.md").write_text(original)
    globals_ = namespace["write_current_component_reports"].__globals__
    monkeypatch.setitem(globals_, "load_report_runs", lambda *_: [])
    monkeypatch.setitem(globals_, "_metric_bands", lambda **_: {})

    def render(dataset_size: str, research_question: int, *_) -> str:
        if research_question == 4:
            raise ValueError("boundary is open")
        return f"## RQ{research_question} — current {dataset_size}\n\n| current |"

    monkeypatch.setitem(globals_, "_render_current_component_question", render)
    monkeypatch.setitem(
        globals_, "render_full_tuning_report", lambda _: "# current tuning\n"
    )

    with pytest.raises(ValueError, match="50m RQ4: boundary is open"):
        namespace["write_current_component_reports"](output)

    report = (output / "research_questions_500m.md").read_text()
    assert "## RQ4 — stale" in report
    assert "## RQ5 — current" in report
