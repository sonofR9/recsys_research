from __future__ import annotations

import json
from pathlib import Path
from typing import Any


FILESYSTEM_TIMESTAMP_TOLERANCE_SECONDS = 0.001
_QUEUE_STATES = ("pending", "dispatched", "completed", "failed")


def verify_unique_completed_run(
    queue_root: Path,
    *,
    run_name: str,
    expected_job_id: str,
) -> None:
    matches = []
    for state in _QUEUE_STATES:
        for path in (queue_root / state).glob("*.json"):
            value = _load_json(path)
            if isinstance(value, dict) and value.get("run") == run_name:
                matches.append((state, value.get("id")))
    if matches != [("completed", expected_job_id)]:
        raise ValueError(f"queue run {run_name!r} is not uniquely attributable")


def verify_artifacts_in_job_window(
    paths: tuple[Path, ...],
    *,
    dispatched_at: float,
    finished_at: float,
    run_label: str,
    tolerance_seconds: float = FILESYSTEM_TIMESTAMP_TOLERANCE_SECONDS,
) -> None:
    if tolerance_seconds < 0 or dispatched_at > finished_at:
        raise ValueError(f"invalid queue timestamp window for {run_label}")
    lower = dispatched_at - tolerance_seconds
    upper = finished_at + tolerance_seconds
    if any(
        not path.is_file() or not lower <= path.stat().st_mtime <= upper
        for path in paths
    ):
        raise ValueError(f"artifacts fall outside the job window for {run_label}")


def _load_json(path: Path) -> object:
    try:
        return json.loads(
            path.read_text(),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load queue record {path}") from error


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r}")
