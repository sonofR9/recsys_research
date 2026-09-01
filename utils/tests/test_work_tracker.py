from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).parents[2]


def run_workflow(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "work.workflow", *arguments],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def make_work_root(root: Path) -> Path:
    work_root = root / "work"
    for status in ("not_started", "wip", "blocked", "human_review", "done"):
        (work_root / status).mkdir(parents=True)
    return work_root


def write_task(path: Path, *, task_id: str, ideas_tag: str = "") -> None:
    ideas = f"\n  - {ideas_tag}" if ideas_tag else " []"
    path.write_text(
        f"""version: 1
id: {task_id}
title: Example task
description: Short description.
blocked_by: []
ideas_tags:{ideas}
""",
        encoding="utf-8",
    )


def test_automatic_monitor_command_is_not_available() -> None:
    result = run_workflow("monitor")

    assert result.returncode == 2
    assert "invalid choice" in result.stderr


def test_sync_uses_folder_as_canonical_ideas_status(tmp_path: Path) -> None:
    work_root = make_work_root(tmp_path)
    write_task(
        work_root / "human_review" / "g1-rq7.yaml",
        task_id="g1-rq7",
        ideas_tag="g1-rq7",
    )
    ideas = tmp_path / "ideas.md"
    ideas.write_text(
        "- [wip] rq7 Which position encoding? <!-- work:g1-rq7 -->\n"
        "- [not_started] unrelated\n",
        encoding="utf-8",
    )

    result = run_workflow(
        "sync",
        "--work-root",
        str(work_root),
        "--ideas",
        str(ideas),
    )

    assert result.returncode == 0, result.stderr
    assert ideas.read_text(encoding="utf-8") == (
        "- [review] rq7 Which position encoding? <!-- work:g1-rq7 -->\n"
        "- [not_started] unrelated\n"
    )


def test_sync_preserves_ideas_file_mode(tmp_path: Path) -> None:
    work_root = make_work_root(tmp_path)
    write_task(
        work_root / "human_review" / "g1-rq7.yaml",
        task_id="g1-rq7",
        ideas_tag="g1-rq7",
    )
    ideas = tmp_path / "ideas.md"
    ideas.write_text(
        "- [wip] rq7 Which position encoding? <!-- work:g1-rq7 -->\n",
        encoding="utf-8",
    )
    ideas.chmod(0o640)

    result = run_workflow(
        "sync",
        "--work-root",
        str(work_root),
        "--ideas",
        str(ideas),
    )

    assert result.returncode == 0, result.stderr
    assert ideas.stat().st_mode & 0o777 == 0o640


def test_sync_rejects_duplicate_ideas_tags(tmp_path: Path) -> None:
    work_root = make_work_root(tmp_path)
    write_task(
        work_root / "wip" / "first.yaml",
        task_id="first",
        ideas_tag="shared",
    )
    write_task(
        work_root / "blocked" / "second.yaml",
        task_id="second",
        ideas_tag="shared",
    )
    ideas = tmp_path / "ideas.md"
    original = "- [not_started] question <!-- work:shared -->\n"
    ideas.write_text(original, encoding="utf-8")

    result = run_workflow(
        "sync",
        "--work-root",
        str(work_root),
        "--ideas",
        str(ideas),
    )

    assert result.returncode == 2
    assert "duplicate ideas tag: shared" in result.stderr
    assert ideas.read_text(encoding="utf-8") == original


def test_validation_accepts_task_without_owner(tmp_path: Path) -> None:
    work_root = make_work_root(tmp_path)
    task = work_root / "wip" / "lead-task.yaml"
    write_task(task, task_id="lead-task")

    result = run_workflow("validate", "--work-root", str(work_root))

    assert result.returncode == 0, result.stderr


def test_validation_rejects_unknown_and_self_dependencies(tmp_path: Path) -> None:
    work_root = make_work_root(tmp_path)
    task = work_root / "blocked" / "blocked-task.yaml"
    write_task(task, task_id="blocked-task")
    original = task.read_text(encoding="utf-8")
    task.write_text(
        original.replace("blocked_by: []", "blocked_by: [missing-task]"),
        encoding="utf-8",
    )

    unknown = run_workflow("validate", "--work-root", str(work_root))

    assert unknown.returncode == 2
    assert "unknown blocked_by task: missing-task" in unknown.stderr

    task.write_text(
        original.replace("blocked_by: []", "blocked_by: [blocked-task]"),
        encoding="utf-8",
    )
    self_dependency = run_workflow("validate", "--work-root", str(work_root))

    assert self_dependency.returncode == 2
    assert "task cannot block itself" in self_dependency.stderr


def test_validation_rejects_dependency_cycles(tmp_path: Path) -> None:
    work_root = make_work_root(tmp_path)
    first = work_root / "blocked" / "first.yaml"
    second = work_root / "blocked" / "second.yaml"
    write_task(first, task_id="first")
    write_task(second, task_id="second")
    first.write_text(
        first.read_text(encoding="utf-8").replace(
            "blocked_by: []", "blocked_by: [second]"
        ),
        encoding="utf-8",
    )
    second.write_text(
        second.read_text(encoding="utf-8").replace(
            "blocked_by: []", "blocked_by: [first]"
        ),
        encoding="utf-8",
    )

    result = run_workflow("validate", "--work-root", str(work_root))

    assert result.returncode == 2
    assert "dependency cycle: first -> second -> first" in result.stderr
