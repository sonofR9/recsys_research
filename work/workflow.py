from __future__ import annotations

import argparse
import os
import re
import stat
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import yaml


REPOSITORY_ROOT = Path(__file__).parents[1]
DEFAULT_WORK_ROOT = REPOSITORY_ROOT / "work"
DEFAULT_IDEAS = REPOSITORY_ROOT / "experiments" / "ideas.md"
REMINDER = "there are unfinished tasks, finish them please"
STATUSES = ("not_started", "wip", "blocked", "human_review", "done")
REMINDER_STATUSES = frozenset(("not_started", "wip"))
IDEAS_STATUS = {
    "not_started": "not_started",
    "wip": "wip",
    "blocked": "wip",
    "human_review": "review",
    "done": "complete",
}
ID_PATTERN = re.compile(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?")
IDEAS_TAG_PATTERN = re.compile(r"<!--\s*work:([a-z0-9][a-z0-9-]*)\s*-->")
IDEAS_STATE_PATTERN = re.compile(r"\[(not_started|wip|review|complete)\]")


class WorkflowError(ValueError):
    pass


@dataclass(frozen=True)
class Task:
    id: str
    status: str
    blocked_by: tuple[str, ...]
    ideas_tags: tuple[str, ...]
    path: Path


class Notifier(Protocol):
    def notify(self, message: str) -> None: ...


class StdoutNotifier:
    def notify(self, message: str) -> None:
        print(message, flush=True)


@dataclass(frozen=True)
class TmuxNotifier:
    pane: str

    @dataclass(frozen=True)
    class Composer:
        text: str
        submission_key: str

    def _has_codex_descendant(self, pane_pid: int) -> bool:
        process_rows = subprocess.run(
            ["ps", "-eo", "pid=,ppid=,comm=,args="],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        processes: dict[int, tuple[int, str, str]] = {}
        for row in process_rows:
            columns = row.split(maxsplit=3)
            if len(columns) != 4:
                continue
            try:
                process_id, parent_id = (int(columns[0]), int(columns[1]))
            except ValueError:
                continue
            processes[process_id] = (parent_id, columns[2], columns[3])

        descendants = {pane_pid}
        while True:
            discovered = {
                process_id
                for process_id, (parent_id, _, _) in processes.items()
                if parent_id in descendants
            }
            expanded = descendants | discovered
            if expanded == descendants:
                break
            descendants = expanded
        for process_id in descendants:
            process = processes.get(process_id)
            if process is None:
                continue
            _, command, arguments = process
            argument_names = {Path(argument).name for argument in arguments.split()}
            if command == "codex" or (command == "node" and "codex" in argument_names):
                return True
        return False

    def _composer(self) -> Composer | None:
        pane = subprocess.run(
            ["tmux", "capture-pane", "-p", "-J", "-t", self.pane],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        pane_lines = pane.splitlines()
        composer_indexes = [
            index
            for index, line in enumerate(pane_lines)
            if line.lstrip().startswith("›")
        ]
        if not composer_indexes:
            return None
        composer_index = composer_indexes[-1]
        composer_line = pane_lines[composer_index].lstrip()
        text = composer_line.removeprefix("›").strip()
        if text == "Implement {feature}":
            text = ""
        recent_ui = " ".join(pane_lines[max(0, composer_index - 12) : composer_index])
        queue_required = (
            "Working (" in recent_ui and "esc to interrupt" in recent_ui
        ) or "tab to queue message" in recent_ui.lower()
        return self.Composer(
            text=text,
            submission_key="Tab" if queue_required else "Enter",
        )

    def _send_keys(self, *keys: str) -> None:
        subprocess.run(
            ["tmux", "send-keys", "-t", self.pane, *keys],
            check=True,
        )

    def _wait_for_message(self, message: str, *, present: bool) -> bool:
        for _ in range(10):
            composer = self._composer()
            message_is_present = composer is not None and composer.text == message
            if message_is_present == present:
                return True
            time.sleep(0.1)
        return False

    def notify(self, message: str) -> None:
        pane_state = subprocess.run(
            [
                "tmux",
                "display-message",
                "-p",
                "-t",
                self.pane,
                "#{pane_dead} #{pane_current_command} #{pane_pid}",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        state_parts = pane_state.split()
        if len(state_parts) != 3 or state_parts[0] != "0":
            raise WorkflowError(
                f"tmux target {self.pane} is not an active Codex pane: {pane_state}"
            )
        command, pane_pid_text = state_parts[1:]
        try:
            pane_pid = int(pane_pid_text)
        except ValueError as error:
            raise WorkflowError(f"invalid tmux pane pid: {pane_pid_text}") from error
        if command not in {"codex", "node"} or not self._has_codex_descendant(
            pane_pid
        ):
            raise WorkflowError(f"tmux target {self.pane} is not a Codex process")
        composer = self._composer()
        if composer is None or composer.text:
            return
        self._send_keys("-l", message)
        if not self._wait_for_message(message, present=True):
            raise WorkflowError("tmux reminder insertion could not be verified")
        submission_keys = [composer.submission_key] * 2
        if composer.submission_key == "Tab":
            submission_keys.append("Enter")
        for submission_key in submission_keys:
            self._send_keys(submission_key)
            if self._wait_for_message(message, present=False):
                return
        raise WorkflowError("tmux reminder remained in the composer after submission")


def _string_list(value: object, *, field: str, path: Path) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise WorkflowError(f"{path}: {field} must be a list of strings")
    return tuple(value)


def _load_task(path: Path, status: str) -> Task:
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise WorkflowError(f"{path}: cannot read task: {error}") from error
    if not isinstance(document, dict):
        raise WorkflowError(f"{path}: task must be a YAML mapping")
    if document.get("version") != 1:
        raise WorkflowError(f"{path}: version must be 1")
    task_id = document.get("id")
    if not isinstance(task_id, str) or ID_PATTERN.fullmatch(task_id) is None:
        raise WorkflowError(f"{path}: id must be a lowercase kebab-case name")
    if path.stem != task_id:
        raise WorkflowError(f"{path}: filename must match task id {task_id}.yaml")
    for field in ("title", "description"):
        if not isinstance(document.get(field), str) or not document[field].strip():
            raise WorkflowError(f"{path}: {field} must be a non-empty string")
    blocked_by = _string_list(
        document.get("blocked_by"), field="blocked_by", path=path
    )
    ideas_tags = _string_list(
        document.get("ideas_tags"), field="ideas_tags", path=path
    )
    for tag in ideas_tags:
        if ID_PATTERN.fullmatch(tag) is None:
            raise WorkflowError(f"{path}: invalid ideas tag: {tag}")
    return Task(
        id=task_id,
        status=status,
        blocked_by=blocked_by,
        ideas_tags=ideas_tags,
        path=path,
    )


def load_tasks(work_root: Path) -> tuple[Task, ...]:
    tasks: list[Task] = []
    for status in STATUSES:
        status_directory = work_root / status
        if not status_directory.is_dir():
            raise WorkflowError(f"missing work status directory: {status_directory}")
        task_paths = sorted(status_directory.glob("*.yaml"))
        tasks.extend(_load_task(path, status) for path in task_paths)

    ids: set[str] = set()
    tags: set[str] = set()
    for task in tasks:
        if task.id in ids:
            raise WorkflowError(f"duplicate task id: {task.id}")
        ids.add(task.id)
        for tag in task.ideas_tags:
            if tag in tags:
                raise WorkflowError(f"duplicate ideas tag: {tag}")
            tags.add(tag)
    for task in tasks:
        for dependency in task.blocked_by:
            if dependency == task.id:
                raise WorkflowError(f"{task.path}: task cannot block itself")
            if dependency not in ids:
                raise WorkflowError(
                    f"{task.path}: unknown blocked_by task: {dependency}"
                )
    _validate_dependency_graph(tasks)
    return tuple(tasks)


def _validate_dependency_graph(tasks: Sequence[Task]) -> None:
    dependencies = {task.id: task.blocked_by for task in tasks}
    visited: set[str] = set()
    visiting: set[str] = set()
    path: list[str] = []

    def visit(task_id: str) -> None:
        if task_id in visited:
            return
        if task_id in visiting:
            cycle_start = path.index(task_id)
            cycle = path[cycle_start:] + [task_id]
            raise WorkflowError(f"dependency cycle: {' -> '.join(cycle)}")
        visiting.add(task_id)
        path.append(task_id)
        for dependency in dependencies[task_id]:
            visit(dependency)
        path.pop()
        visiting.remove(task_id)
        visited.add(task_id)

    for task in tasks:
        visit(task.id)


def _replace_ideas_state(line: str, tag: str, status: str) -> str:
    if len(IDEAS_STATE_PATTERN.findall(line)) != 1:
        raise WorkflowError(f"ideas tag {tag} must share a line with exactly one status")
    return IDEAS_STATE_PATTERN.sub(f"[{status}]", line, count=1)


def synchronize_ideas(tasks: Sequence[Task], ideas_path: Path) -> None:
    task_statuses = {
        tag: IDEAS_STATUS[task.status]
        for task in tasks
        for tag in task.ideas_tags
    }
    try:
        ideas_stat = ideas_path.stat()
        lines = ideas_path.read_text(encoding="utf-8").splitlines(keepends=True)
    except OSError as error:
        raise WorkflowError(f"cannot read ideas file {ideas_path}: {error}") from error

    seen: set[str] = set()
    synchronized: list[str] = []
    for line in lines:
        line_tags = IDEAS_TAG_PATTERN.findall(line)
        if len(line_tags) > 1:
            raise WorkflowError("an ideas status line may have only one work tag")
        for tag in line_tags:
            if tag in seen:
                raise WorkflowError(f"duplicate ideas marker: {tag}")
            if tag not in task_statuses:
                raise WorkflowError(f"ideas marker has no task: {tag}")
            seen.add(tag)
            line = _replace_ideas_state(line, tag, task_statuses[tag])
        synchronized.append(line)

    missing = sorted(set(task_statuses) - seen)
    if missing:
        raise WorkflowError(f"task ideas tag has no marker: {missing[0]}")
    new_text = "".join(synchronized)
    old_text = "".join(lines)
    if new_text == old_text:
        return
    ideas_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=ideas_path.parent, delete=False
    ) as temporary:
        os.fchmod(temporary.fileno(), stat.S_IMODE(ideas_stat.st_mode))
        temporary.write(new_text)
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, ideas_path)


def run_cycle(work_root: Path, ideas_path: Path, notifier: Notifier) -> None:
    tasks = load_tasks(work_root)
    synchronize_ideas(tasks, ideas_path)
    if any(task.status in REMINDER_STATUSES for task in tasks):
        notifier.notify(REMINDER)


def _add_paths(parser: argparse.ArgumentParser, *, ideas: bool) -> None:
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    if ideas:
        parser.add_argument("--ideas", type=Path, default=DEFAULT_IDEAS)


def _parse_arguments(arguments: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lead-owned work tracker")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate")
    _add_paths(validate, ideas=False)

    sync = subparsers.add_parser("sync")
    _add_paths(sync, ideas=True)

    monitor = subparsers.add_parser("monitor")
    _add_paths(monitor, ideas=True)
    monitor.add_argument("--interval-seconds", type=float, default=1800)
    monitor.add_argument("--once", action="store_true")
    monitor.add_argument("--max-cycles", type=int)
    notification = monitor.add_mutually_exclusive_group(required=True)
    notification.add_argument("--notify-stdout", action="store_true")
    notification.add_argument("--tmux-pane")
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    options = _parse_arguments(arguments)
    try:
        if options.command == "validate":
            load_tasks(options.work_root)
            return 0
        if options.command == "sync":
            synchronize_ideas(load_tasks(options.work_root), options.ideas)
            return 0
        if options.interval_seconds <= 0:
            raise WorkflowError("interval must be positive")
        if options.max_cycles is not None and options.max_cycles <= 0:
            raise WorkflowError("max cycles must be positive")
        notifier: Notifier = (
            StdoutNotifier()
            if options.notify_stdout
            else TmuxNotifier(options.tmux_pane)
        )
        completed_cycles = 0
        while True:
            cycle_error: OSError | subprocess.SubprocessError | WorkflowError | None = None
            try:
                run_cycle(options.work_root, options.ideas, notifier)
            except (OSError, subprocess.SubprocessError, WorkflowError) as error:
                cycle_error = error
                print(f"work monitor cycle failed: {error}", file=sys.stderr, flush=True)
            completed_cycles += 1
            if options.once:
                return 2 if cycle_error is not None else 0
            if (
                options.max_cycles is not None
                and completed_cycles >= options.max_cycles
            ):
                return 2 if cycle_error is not None else 0
            time.sleep(options.interval_seconds)
    except (OSError, subprocess.SubprocessError, WorkflowError) as error:
        print(error, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
