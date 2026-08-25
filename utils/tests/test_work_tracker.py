from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).parents[2]
REMINDER = "there are unfinished tasks, finish them please"


def run_workflow(
    *arguments: str, environment: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "work.workflow", *arguments],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
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


def make_fake_tmux(
    tmp_path: Path,
    composer: str,
    *,
    fail_first_display: bool = False,
    ignore_first_submit: bool = False,
    ignore_tab: bool = False,
    wrap_inserted_message: bool = False,
) -> tuple[dict[str, str], Path]:
    binaries = tmp_path / "bin"
    binaries.mkdir()
    log = tmp_path / "tmux.log"
    capture = tmp_path / "capture.txt"
    capture.write_text(composer, encoding="utf-8")
    joined_capture = tmp_path / "joined-capture.txt"
    joined_capture.write_text(composer, encoding="utf-8")
    tmux = binaries / "tmux"
    tmux.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env bash
            printf '%s\\n' "$*" >> "$FAKE_TMUX_LOG"
            case "$1" in
                display-message)
                    if [[ "${FAKE_TMUX_FAIL_FIRST_DISPLAY:-0}" == 1 && \
                          ! -e "$FAKE_TMUX_FAILURE_MARKER" ]]; then
                        touch "$FAKE_TMUX_FAILURE_MARKER"
                        exit 1
                    fi
                    printf '%s\\n' '0 node 100'
                    ;;
                capture-pane)
                    if [[ "${FAKE_TMUX_WRAP_INSERTION:-0}" == 1 && "$*" == *-J* ]]; then
                        cat "$FAKE_TMUX_JOINED_CAPTURE"
                    else
                        cat "$FAKE_TMUX_CAPTURE"
                    fi
                    ;;
                send-keys)
                    if [[ "$4" == "-l" ]]; then
                        if [[ "${FAKE_TMUX_WRAP_INSERTION:-0}" == 1 ]]; then
                            printf 'history\\n› %s\\n  %s\\n' \
                                "${5:0:20}" "${5:20}" > "$FAKE_TMUX_CAPTURE"
                        else
                            printf 'history\\n› %s\\n' "$5" > "$FAKE_TMUX_CAPTURE"
                        fi
                        printf 'history\\n› %s\\n' "$5" > "$FAKE_TMUX_JOINED_CAPTURE"
                    elif [[ "${FAKE_TMUX_IGNORE_TAB:-0}" == 1 && "$4" == "Tab" ]]; then
                        :
                    elif [[ "${FAKE_TMUX_IGNORE_FIRST_SUBMIT:-0}" == 1 && \
                            ! -e "$FAKE_TMUX_SUBMIT_MARKER" ]]; then
                        touch "$FAKE_TMUX_SUBMIT_MARKER"
                    else
                        printf 'history\\n› Implement {feature}\\n' > "$FAKE_TMUX_CAPTURE"
                        printf 'history\\n› Implement {feature}\\n' > "$FAKE_TMUX_JOINED_CAPTURE"
                    fi
                    ;;
            esac
            """
        ),
        encoding="utf-8",
    )
    tmux.chmod(0o755)
    process_status = binaries / "ps"
    process_status.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env bash
            cat <<'EOF'
                100       1 bash            -bash
                101     100 node            node /usr/bin/codex --yolo resume
                102     101 codex           /opt/codex --yolo resume
            EOF
            """
        ),
        encoding="utf-8",
    )
    process_status.chmod(0o755)
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{binaries}:{environment['PATH']}",
            "FAKE_TMUX_LOG": str(log),
            "FAKE_TMUX_CAPTURE": str(capture),
            "FAKE_TMUX_JOINED_CAPTURE": str(joined_capture),
            "FAKE_TMUX_FAILURE_MARKER": str(tmp_path / "tmux-failed-once"),
            "FAKE_TMUX_SUBMIT_MARKER": str(tmp_path / "tmux-submit-ignored-once"),
        }
    )
    if fail_first_display:
        environment["FAKE_TMUX_FAIL_FIRST_DISPLAY"] = "1"
    if ignore_first_submit:
        environment["FAKE_TMUX_IGNORE_FIRST_SUBMIT"] = "1"
    if ignore_tab:
        environment["FAKE_TMUX_IGNORE_TAB"] = "1"
    if wrap_inserted_message:
        environment["FAKE_TMUX_WRAP_INSERTION"] = "1"
    return environment, log


def test_monitor_reminds_without_task_details(tmp_path: Path) -> None:
    work_root = make_work_root(tmp_path)
    write_task(work_root / "wip" / "repair-report.yaml", task_id="repair-report")
    ideas = tmp_path / "ideas.md"
    ideas.write_text("# Ideas\n", encoding="utf-8")

    result = run_workflow(
        "monitor",
        "--once",
        "--notify-stdout",
        "--work-root",
        str(work_root),
        "--ideas",
        str(ideas),
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == "there are unfinished tasks, finish them please\n"
    assert "repair-report" not in result.stdout


def test_monitor_reminds_for_not_started_work(tmp_path: Path) -> None:
    work_root = make_work_root(tmp_path)
    write_task(work_root / "not_started" / "next.yaml", task_id="next")
    ideas = tmp_path / "ideas.md"
    ideas.write_text("# Ideas\n", encoding="utf-8")

    result = run_workflow(
        "monitor",
        "--once",
        "--notify-stdout",
        "--work-root",
        str(work_root),
        "--ideas",
        str(ideas),
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == "there are unfinished tasks, finish them please\n"


def test_monitor_is_silent_when_every_task_is_done(tmp_path: Path) -> None:
    work_root = make_work_root(tmp_path)
    write_task(work_root / "done" / "finished.yaml", task_id="finished")
    ideas = tmp_path / "ideas.md"
    ideas.write_text("# Ideas\n", encoding="utf-8")

    result = run_workflow(
        "monitor",
        "--once",
        "--notify-stdout",
        "--work-root",
        str(work_root),
        "--ideas",
        str(ideas),
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == ""


def test_monitor_is_silent_when_tasks_are_blocked_or_awaiting_review(
    tmp_path: Path,
) -> None:
    work_root = make_work_root(tmp_path)
    write_task(work_root / "blocked" / "waiting.yaml", task_id="waiting")
    write_task(
        work_root / "human_review" / "ready.yaml",
        task_id="ready",
    )
    ideas = tmp_path / "ideas.md"
    ideas.write_text("# Ideas\n", encoding="utf-8")

    result = run_workflow(
        "monitor",
        "--once",
        "--notify-stdout",
        "--work-root",
        str(work_root),
        "--ideas",
        str(ideas),
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == ""


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


def test_tmux_notifier_uses_enter_for_idle_empty_composer(
    tmp_path: Path,
) -> None:
    work_root = make_work_root(tmp_path)
    write_task(work_root / "wip" / "active.yaml", task_id="active")
    ideas = tmp_path / "ideas.md"
    ideas.write_text("# Ideas\n", encoding="utf-8")
    environment, log = make_fake_tmux(tmp_path, "history\n› Implement {feature}\n")

    result = run_workflow(
        "monitor",
        "--once",
        "--tmux-pane",
        "%1",
        "--work-root",
        str(work_root),
        "--ideas",
        str(ideas),
        environment=environment,
    )

    assert result.returncode == 0, result.stderr
    calls = log.read_text(encoding="utf-8")
    assert "display-message -p -t %1" in calls
    assert "capture-pane -p -J -t %1" in calls
    assert f"send-keys -t %1 -l {REMINDER}" in calls
    assert "send-keys -t %1 Enter" in calls
    assert len([line for line in calls.splitlines() if line.startswith("send-keys")]) == 2


def test_tmux_notifier_uses_tab_for_working_empty_composer(tmp_path: Path) -> None:
    work_root = make_work_root(tmp_path)
    write_task(work_root / "wip" / "active.yaml", task_id="active")
    ideas = tmp_path / "ideas.md"
    ideas.write_text("# Ideas\n", encoding="utf-8")
    environment, log = make_fake_tmux(
        tmp_path,
        "history\n• Working (1m 2s • esc to interrupt)\n\n› Implement {feature}\n",
    )

    result = run_workflow(
        "monitor",
        "--once",
        "--tmux-pane",
        "%1",
        "--work-root",
        str(work_root),
        "--ideas",
        str(ideas),
        environment=environment,
    )

    assert result.returncode == 0, result.stderr
    calls = log.read_text(encoding="utf-8")
    assert f"send-keys -t %1 -l {REMINDER}" in calls
    assert "send-keys -t %1 Tab" in calls
    assert len([line for line in calls.splitlines() if line.startswith("send-keys")]) == 2


def test_tmux_notifier_retries_an_ignored_submission_key(tmp_path: Path) -> None:
    work_root = make_work_root(tmp_path)
    write_task(work_root / "wip" / "active.yaml", task_id="active")
    ideas = tmp_path / "ideas.md"
    ideas.write_text("# Ideas\n", encoding="utf-8")
    environment, log = make_fake_tmux(
        tmp_path,
        "history\n• Working (1m 2s • esc to interrupt)\n\n› Implement {feature}\n",
        ignore_first_submit=True,
    )

    result = run_workflow(
        "monitor",
        "--once",
        "--tmux-pane",
        "%1",
        "--work-root",
        str(work_root),
        "--ideas",
        str(ideas),
        environment=environment,
    )

    assert result.returncode == 0, result.stderr
    calls = log.read_text(encoding="utf-8")
    assert f"send-keys -t %1 -l {REMINDER}" in calls
    assert calls.count("send-keys -t %1 Tab") == 2
    assert len([line for line in calls.splitlines() if line.startswith("send-keys")]) == 3


def test_tmux_notifier_falls_back_to_enter_when_tab_is_not_consumed(
    tmp_path: Path,
) -> None:
    work_root = make_work_root(tmp_path)
    write_task(work_root / "wip" / "active.yaml", task_id="active")
    ideas = tmp_path / "ideas.md"
    ideas.write_text("# Ideas\n", encoding="utf-8")
    environment, log = make_fake_tmux(
        tmp_path,
        "history\n• Working (1m 2s • esc to interrupt)\n\n› Implement {feature}\n",
        ignore_tab=True,
    )

    result = run_workflow(
        "monitor",
        "--once",
        "--tmux-pane",
        "%1",
        "--work-root",
        str(work_root),
        "--ideas",
        str(ideas),
        environment=environment,
    )

    assert result.returncode == 0, result.stderr
    calls = log.read_text(encoding="utf-8")
    assert calls.count("send-keys -t %1 Tab") == 2
    assert calls.count("send-keys -t %1 Enter") == 1


def test_tmux_notifier_submits_a_soft_wrapped_reminder(tmp_path: Path) -> None:
    work_root = make_work_root(tmp_path)
    write_task(work_root / "wip" / "active.yaml", task_id="active")
    ideas = tmp_path / "ideas.md"
    ideas.write_text("# Ideas\n", encoding="utf-8")
    environment, log = make_fake_tmux(
        tmp_path,
        "history\n› Implement {feature}\n",
        wrap_inserted_message=True,
    )

    result = run_workflow(
        "monitor",
        "--once",
        "--tmux-pane",
        "%1",
        "--work-root",
        str(work_root),
        "--ideas",
        str(ideas),
        environment=environment,
    )

    assert result.returncode == 0, result.stderr
    calls = log.read_text(encoding="utf-8")
    assert "capture-pane -p -J -t %1" in calls
    assert f"send-keys -t %1 -l {REMINDER}" in calls
    assert "send-keys -t %1 Enter" in calls


def test_tmux_notifier_defers_when_composer_contains_input(tmp_path: Path) -> None:
    work_root = make_work_root(tmp_path)
    write_task(work_root / "wip" / "active.yaml", task_id="active")
    ideas = tmp_path / "ideas.md"
    ideas.write_text("# Ideas\n", encoding="utf-8")
    environment, log = make_fake_tmux(
        tmp_path,
        "history\n• Working (1m 2s • esc to interrupt)\n\n› user is typing\n",
    )

    result = run_workflow(
        "monitor",
        "--once",
        "--tmux-pane",
        "%1",
        "--work-root",
        str(work_root),
        "--ideas",
        str(ideas),
        environment=environment,
    )

    assert result.returncode == 0, result.stderr
    calls = log.read_text(encoding="utf-8")
    assert "capture-pane -p -J -t %1" in calls
    assert "send-keys" not in calls


def test_monitor_recovers_after_transient_delivery_error(tmp_path: Path) -> None:
    work_root = make_work_root(tmp_path)
    write_task(work_root / "wip" / "active.yaml", task_id="active")
    ideas = tmp_path / "ideas.md"
    ideas.write_text("# Ideas\n", encoding="utf-8")
    environment, log = make_fake_tmux(
        tmp_path,
        "history\n› Implement {feature}\n",
        fail_first_display=True,
    )

    result = run_workflow(
        "monitor",
        "--max-cycles",
        "2",
        "--interval-seconds",
        "0.01",
        "--tmux-pane",
        "%1",
        "--work-root",
        str(work_root),
        "--ideas",
        str(ideas),
        environment=environment,
    )

    assert result.returncode == 0, result.stderr
    assert "returned non-zero exit status 1" in result.stderr
    calls = log.read_text(encoding="utf-8")
    assert calls.count("display-message -p -t %1") == 2
    assert f"send-keys -t %1 -l {REMINDER}" in calls
    assert "send-keys -t %1 Enter" in calls
