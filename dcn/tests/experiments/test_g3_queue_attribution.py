import json
import multiprocessing
import os
from pathlib import Path

import pytest

from experiments.g3_pretrained_item_embeddings.analysis.queue_attribution import (
    FILESYSTEM_TIMESTAMP_TOLERANCE_SECONDS,
    verify_artifacts_in_job_window,
    verify_unique_completed_run,
)
from experiments.g3_pretrained_item_embeddings.launchers.control import (
    ledger_submission_lock,
)


def _hold_ledger_lock(
    state_dir: str,
    ledger_sha256: str,
    acquired,
    release,
) -> None:
    with ledger_submission_lock(
        state_dir=Path(state_dir), ledger_sha256=ledger_sha256
    ):
        acquired.set()
        release.wait(5)


def test_queue_run_attribution_rejects_duplicate_run_records(tmp_path: Path) -> None:
    for state in ("pending", "dispatched", "completed", "failed"):
        (tmp_path / state).mkdir()
    record = {"id": "expected", "run": "run-name"}
    (tmp_path / "completed" / "expected.json").write_text(json.dumps(record))
    verify_unique_completed_run(
        tmp_path,
        run_name="run-name",
        expected_job_id="expected",
    )
    duplicate = {"id": "duplicate", "run": "run-name"}
    (tmp_path / "failed" / "duplicate.json").write_text(json.dumps(duplicate))

    with pytest.raises(ValueError, match="not uniquely attributable"):
        verify_unique_completed_run(
            tmp_path,
            run_name="run-name",
            expected_job_id="expected",
        )


def test_artifact_attribution_uses_only_small_explicit_tolerance(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}")
    dispatched_at = 1000.0
    finished_at = 1010.0
    os.utime(
        artifact,
        (dispatched_at, dispatched_at - FILESYSTEM_TIMESTAMP_TOLERANCE_SECONDS / 2),
    )
    verify_artifacts_in_job_window(
        (artifact,),
        dispatched_at=dispatched_at,
        finished_at=finished_at,
        run_label="row",
    )
    os.utime(
        artifact,
        (dispatched_at, dispatched_at - FILESYSTEM_TIMESTAMP_TOLERANCE_SECONDS * 2),
    )

    with pytest.raises(ValueError, match="outside the job window"):
        verify_artifacts_in_job_window(
            (artifact,),
            dispatched_at=dispatched_at,
            finished_at=finished_at,
            run_label="row",
        )


def test_ledger_submission_lock_serializes_processes(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    first_acquired = context.Event()
    first_release = context.Event()
    second_acquired = context.Event()
    second_release = context.Event()
    sha256 = "a" * 64
    first = context.Process(
        target=_hold_ledger_lock,
        args=(str(tmp_path), sha256, first_acquired, first_release),
    )
    second = context.Process(
        target=_hold_ledger_lock,
        args=(str(tmp_path), sha256, second_acquired, second_release),
    )
    first.start()
    assert first_acquired.wait(2)
    second.start()
    assert not second_acquired.wait(0.2)
    first_release.set()
    assert second_acquired.wait(2)
    second_release.set()
    first.join(2)
    second.join(2)
    assert first.exitcode == 0
    assert second.exitcode == 0
