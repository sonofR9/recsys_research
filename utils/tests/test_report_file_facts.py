from __future__ import annotations

import os
from pathlib import Path

import pytest

from utils.report_file_facts import ReportFileFacts


def test_fact_persists_between_cache_instances(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("payload")
    database = tmp_path / "facts.sqlite3"
    calls = 0

    def compute() -> dict[str, int]:
        nonlocal calls
        calls += 1
        return {"length": len(source.read_bytes())}

    with ReportFileFacts(database) as facts:
        assert facts.load_or_compute("length", (source,), compute) == {"length": 7}
    with ReportFileFacts(database) as facts:
        assert facts.load_or_compute("length", (source,), compute) == {"length": 7}

    assert calls == 1


def test_canonical_paths_share_an_identity(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("payload")
    alias = tmp_path / "alias.txt"
    alias.symlink_to(source)

    with ReportFileFacts(tmp_path / "facts.sqlite3") as facts:
        assert facts.load_or_compute("value", (source,), lambda: 1) == 1
        assert facts.load_or_compute("value", (alias,), lambda: 2) == 1


def test_retargeted_symlink_gets_a_new_identity(tmp_path: Path) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("first")
    second.write_text("other")
    alias = tmp_path / "alias.txt"
    alias.symlink_to(first)

    with ReportFileFacts(tmp_path / "facts.sqlite3") as facts:
        assert facts.load_or_compute("value", (alias,), alias.read_text) == "first"
        alias.unlink()
        alias.symlink_to(second)
        assert facts.load_or_compute("value", (alias,), alias.read_text) == "other"


def test_fact_invalidates_after_size_or_mtime_change(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("first")
    database = tmp_path / "facts.sqlite3"

    with ReportFileFacts(database) as facts:
        assert facts.load_or_compute("content", (source,), source.read_text) == "first"
    source.write_text("second value")
    with ReportFileFacts(database) as facts:
        assert (
            facts.load_or_compute("content", (source,), source.read_text)
            == "second value"
        )


def test_fact_trusts_unchanged_path_size_and_mtime(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("first")
    identity = source.stat()
    database = tmp_path / "facts.sqlite3"

    with ReportFileFacts(database) as facts:
        assert facts.load_or_compute("content", (source,), source.read_text) == "first"
    source.write_text("other")
    os.utime(source, ns=(identity.st_atime_ns, identity.st_mtime_ns))
    with ReportFileFacts(database) as facts:
        assert facts.load_or_compute("content", (source,), source.read_text) == "first"


def test_failed_computation_is_not_cached(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("payload")
    database = tmp_path / "facts.sqlite3"

    with ReportFileFacts(database) as facts:
        with pytest.raises(RuntimeError, match="broken"):
            facts.load_or_compute(
                "value", (source,), lambda: (_ for _ in ()).throw(RuntimeError("broken"))
            )
        assert facts.load_or_compute("value", (source,), lambda: 3) == 3


def test_group_fact_invalidates_when_any_input_changes(tmp_path: Path) -> None:
    left = tmp_path / "left.txt"
    right = tmp_path / "right.txt"
    left.write_text("a")
    right.write_text("b")
    database = tmp_path / "facts.sqlite3"

    def combined() -> str:
        return left.read_text() + right.read_text()

    with ReportFileFacts(database) as facts:
        assert facts.load_or_compute("combined", (left, right), combined) == "ab"
    right.write_text("changed")
    with ReportFileFacts(database) as facts:
        assert facts.load_or_compute("combined", (left, right), combined) == "achanged"
