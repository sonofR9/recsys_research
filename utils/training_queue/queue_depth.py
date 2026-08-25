import argparse
import fcntl
import json
import math
import os
import re
import sys
from pathlib import Path


_PREPARATION = re.compile(r"Prepared stage .+ in ([0-9.]+)s")
_TRAINING = re.compile(r"Trained stage .+ in ([0-9.]+)s")


def _build_history_index(root: Path, service_state: Path) -> dict[str, object]:
    ratios: dict[tuple[str, str], float] = {}
    for record in service_state.glob("completed/*.json"):
        try:
            job = json.loads(record.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        script = job.get("script")
        data_group = job.get("data_group")
        run = job.get("run")
        if not all(isinstance(value, str) for value in (script, data_group, run)):
            continue
        ratio = timing_ratio([root / run / "sweep.log"])
        if ratio is None:
            continue
        key = (script, data_group)
        ratios[key] = max(ratios.get(key, 0.0), ratio)
    return {
        "entries": [
            {"data_group": data_group, "ratio": ratio, "script": script}
            for (script, data_group), ratio in sorted(ratios.items())
        ],
        "version": 1,
    }


def historical_timing_ratio(
    root: Path,
    *,
    script: str,
    data_group: str,
    service_state: Path | None = None,
    history_index: Path | None = None,
) -> float | None:
    index = None
    if history_index is not None:
        try:
            index = json.loads(history_index.read_text())
            if (
                not isinstance(index, dict)
                or index.get("version") != 1
                or not isinstance(index.get("entries"), list)
            ):
                raise ValueError
        except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError):
            index = None
    if index is None:
        if service_state is None:
            return None
        index = _build_history_index(root, service_state)
        if history_index is not None:
            _write_json(history_index, index)
    for entry in index["entries"]:
        if (
            isinstance(entry, dict)
            and entry.get("script") == script
            and entry.get("data_group") == data_group
            and isinstance(entry.get("ratio"), (int, float))
        ):
            return float(entry["ratio"])
    return None


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, sort_keys=True) + "\n")
    temporary.replace(path)


def update_history_index(
    path: Path,
    *,
    script: str,
    data_group: str,
    ratio: float,
) -> None:
    lock_path = path.with_suffix(f"{path.suffix}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            history = json.loads(path.read_text())
            if (
                not isinstance(history, dict)
                or history.get("version") != 1
                or not isinstance(history.get("entries"), list)
            ):
                raise ValueError
        except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError):
            history = {"entries": [], "version": 1}
        entries = history["entries"]
        matching = next(
            (
                entry
                for entry in entries
                if isinstance(entry, dict)
                and entry.get("script") == script
                and entry.get("data_group") == data_group
            ),
            None,
        )
        if matching is None:
            entries.append(
                {"data_group": data_group, "ratio": ratio, "script": script}
            )
        else:
            matching["ratio"] = max(float(matching.get("ratio", 0.0)), ratio)
        _write_json(path, history)


def timing_ratio(logs: list[Path]) -> float | None:
    ratios: list[float] = []
    for log in logs:
        if not log.is_file():
            continue
        text = log.read_text()
        preparation = sum(map(float, _PREPARATION.findall(text)))
        training = sum(map(float, _TRAINING.findall(text)))
        if preparation > 0 and training > 0:
            ratios.append(preparation / training)
    return max(ratios) if ratios else None


def queue_depth(logs: list[Path], gpu_count: int = 1, max_depth: int = 4) -> int:
    ratio = timing_ratio(logs)
    return _depth_from_ratio(ratio, max_depth=max_depth)


def _depth_from_ratio(ratio: float | None, *, max_depth: int) -> int:
    measured = 1 + math.ceil(ratio) if ratio is not None else 2
    return min(measured, max_depth)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-count", type=int, default=1)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--history-root", type=Path)
    parser.add_argument("--history-index", type=Path)
    parser.add_argument("--service-state", type=Path)
    parser.add_argument("--script")
    parser.add_argument("--data-group", default="")
    parser.add_argument("--record-timing-log", type=Path)
    parser.add_argument("logs", nargs="*", type=Path)
    arguments = parser.parse_args()
    if arguments.gpu_count < 1 or arguments.max_depth < 1:
        parser.error("counts must be positive")
    if arguments.record_timing_log is not None:
        if arguments.history_index is None or arguments.script is None:
            parser.error("--record-timing-log requires --history-index and --script")
        recorded_ratio = timing_ratio([arguments.record_timing_log])
        if recorded_ratio is not None:
            update_history_index(
                arguments.history_index,
                script=arguments.script,
                data_group=arguments.data_group,
                ratio=recorded_ratio,
            )
        raise SystemExit(0)
    if arguments.history_root is not None:
        if arguments.script is None:
            parser.error("--history-root requires --script")
        historical_ratio = historical_timing_ratio(
            arguments.history_root,
            script=arguments.script,
            data_group=arguments.data_group,
            service_state=arguments.service_state,
            history_index=arguments.history_index,
        )
    else:
        historical_ratio = None
    ratio = timing_ratio(arguments.logs)
    if historical_ratio is not None:
        ratio = historical_ratio if ratio is None else max(ratio, historical_ratio)
    if ratio is not None and ratio >= 1:
        print(
            "warning: preprocessing is slower than one GPU's training consumption; "
            "lookahead cannot prevent all GPU stalls",
            file=sys.stderr,
        )
    print(_depth_from_ratio(ratio, max_depth=arguments.max_depth))
