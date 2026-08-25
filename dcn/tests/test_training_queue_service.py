import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

import pytest


SERVICE = (
    Path(__file__).resolve().parents[2]
    / "utils"
    / "training_queue"
    / "service.py"
)
QUEUE = SERVICE.with_name("queue.sh")
SCHEDULER = SERVICE.with_name("service_scheduler.sh")
REPOSITORY = SERVICE.parents[2]


def _executable(path: Path, source: str) -> None:
    path.write_text(source)
    path.chmod(0o755)


def _service(
    state: Path,
    *arguments: str,
    environment: dict[str, str] | None = None,
    working_directory: Path | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SERVICE), "--state-dir", str(state), *arguments],
        check=check,
        capture_output=True,
        text=True,
        timeout=8,
        env=environment,
        cwd=working_directory,
    )


def _wait_for_status(
    state: Path,
    predicate: Callable[[dict[str, object]], bool],
    timeout: float = 8,
) -> dict[str, object]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = _service(state, "status", "--json", check=False)
        if result.returncode == 0:
            status = json.loads(result.stdout)
            if predicate(status):
                return status
        time.sleep(0.02)
    raise AssertionError(f"service state did not satisfy predicate: {result.stdout}")


def _queue_environment(tmp_path: Path, state: Path) -> dict[str, str]:
    binaries = tmp_path / "bin"
    binaries.mkdir()
    _executable(
        binaries / "nvidia-smi",
        """#!/usr/bin/env bash
if [[ "$1" == --query-gpu=* ]]; then
    printf '0, GPU-a\n1, GPU-b\n'
elif [[ "$1" == --query-compute-apps=* ]]; then
    exit 0
elif [[ "$1" == --id=* ]]; then
    printf '0, 0, 1000\n'
fi
""",
    )
    _executable(
        binaries / "python",
        """#!/usr/bin/env bash
touch "$DCN_PREPARED_MARKER"
while [ ! -e "$DCN_TRAINING_RELEASE" ]; do sleep 0.01; done
exec 8>"generated/gpu-$CUDA_VISIBLE_DEVICES-slot-$DCN_GPU_LOCK_SLOT.lock"
flock -x 8
printf 'start:%s:%s:%s\n' "$VARIANT" "$CUDA_VISIBLE_DEVICES" "$(date +%s%N)" >> "$QUEUE_RECORD"
if [ "${HOLD_RUN:-0}" = 1 ]; then
    while [ ! -e "$QUEUE_RELEASE" ]; do sleep 0.01; done
else
    sleep "${RUN_SECONDS:-0.05}"
fi
printf 'end:%s:%s:%s\n' "$VARIANT" "$CUDA_VISIBLE_DEVICES" "$(date +%s%N)" >> "$QUEUE_RECORD"
[ "${FAIL_RUN:-0}" = 0 ]
""",
    )
    return {
        **os.environ,
        "PATH": f"{binaries}:{os.environ['PATH']}",
        "QUEUE_RECORD": str(tmp_path / "runs"),
        "QUEUE_RELEASE": str(tmp_path / "release"),
        "TRAINING_QUEUE_IN_FLIGHT": "1",
        "TRAINING_QUEUE_GPU_RECHECK_SECONDS": "600",
        "TRAINING_QUEUE_SERVICE_STATE_DIR": str(state),
    }


def _real_dcn_queue_environment(tmp_path: Path, state: Path) -> dict[str, str]:
    binaries = tmp_path / "bin"
    binaries.mkdir()
    _executable(
        binaries / "nvidia-smi",
        """#!/usr/bin/env bash
if [[ "$1" == --query-gpu=* ]]; then
    printf '0, GPU-a\n'
elif [[ "$1" == --query-compute-apps=* ]]; then
    exit 0
elif [[ "$1" == --id=* ]]; then
    printf '0, 0, 1000\n'
fi
""",
    )
    python_path = str(REPOSITORY)
    if os.environ.get("PYTHONPATH"):
        python_path += f":{os.environ['PYTHONPATH']}"
    return {
        **os.environ,
        "PATH": f"{binaries}:{os.environ['PATH']}",
        "PYTHONPATH": python_path,
        "TRAINING_QUEUE_IN_FLIGHT": "1",
        "TRAINING_QUEUE_GPU_RECHECK_SECONDS": "600",
        "TRAINING_QUEUE_SERVICE_STATE_DIR": str(state),
    }


def _launcher(
    tmp_path: Path,
    environment: dict[str, str],
    variant: str,
    *assignments: str,
) -> subprocess.Popen[str]:
    assignment_text = " ".join(assignments)
    command = (
        f'TRAINING_QUEUE_SCRIPT=experiment.py; source "{QUEUE}" && '
        f"enqueue {variant} VARIANT={variant} {assignment_text} && drain"
    )
    return subprocess.Popen(
        ["bash", "-c", command],
        cwd=tmp_path,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _stop(state: Path) -> None:
    _service(state, "stop", check=False)


def test_scheduler_executes_one_command_only_after_its_newline(
    tmp_path: Path,
) -> None:
    state = tmp_path / "service"
    state.mkdir()
    environment = _queue_environment(tmp_path, state)
    environment["TRAINING_QUEUE_MONITOR_LIGHT_GPUS"] = "0"
    record = tmp_path / "scheduler-record"
    process = subprocess.Popen(
        ["bash", str(SCHEDULER), str(state)],
        cwd=tmp_path,
        env=environment,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdin is not None
    try:
        deadline = time.monotonic() + 4
        while time.monotonic() < deadline and not (state / "engine.ready").exists():
            assert process.poll() is None
            time.sleep(0.02)
        assert (state / "engine.ready").exists()

        process.stdin.write("p")
        process.stdin.flush()
        time.sleep(0.25)
        assert process.poll() is None
        assert not record.exists()

        process.stdin.write("rintf 'executed\\n' >> scheduler-record\n")
        process.stdin.flush()
        deadline = time.monotonic() + 4
        while time.monotonic() < deadline and not record.exists():
            assert process.poll() is None
            time.sleep(0.02)
        assert record.read_text().splitlines() == ["executed"]

        process.stdin.close()
        assert process.wait(timeout=4) == 0
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=4)


def test_granular_submissions_from_two_launchers_overlap_across_gpus(
    tmp_path: Path,
) -> None:
    state = tmp_path / "service"
    environment = _queue_environment(tmp_path, state)
    release = tmp_path / "release"
    try:
        _service(
            state,
            "start",
            environment=environment,
            working_directory=tmp_path,
        )
        duplicate = _service(state, "start", check=False)
        assert duplicate.returncode != 0
        assert "already running" in duplicate.stderr

        first = _launcher(tmp_path, environment, "first", "HOLD_RUN=1")
        _wait_for_status(state, lambda value: value["active"] >= 1)
        second = _launcher(tmp_path, environment, "second", "RUN_SECONDS=0.2")

        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            if (tmp_path / "runs").exists() and "start:second:" in (
                tmp_path / "runs"
            ).read_text():
                break
            time.sleep(0.02)
        else:
            raise AssertionError("second launcher did not overlap the held first run")
        release.touch()
        assert first.wait(timeout=8) == 0
        assert second.wait(timeout=8) == 0
        events = [
            line.split(":")
            for line in (tmp_path / "runs").read_text().splitlines()
        ]
        starts = {row[1]: (row[2], int(row[3])) for row in events if row[0] == "start"}
        ends = {row[1]: int(row[3]) for row in events if row[0] == "end"}
        assert {gpu for gpu, _ in starts.values()} == {"GPU-a", "GPU-b"}
        assert starts["second"][1] < ends["first"]

        status = json.loads(_service(state, "status", "--json").stdout)
        assert status["completed"] == 2
        assert status["failed"] == 0
    finally:
        release.touch(exist_ok=True)
        _stop(state)


def test_service_deduplicates_stable_run_across_concurrent_batches(
    tmp_path: Path,
) -> None:
    state = tmp_path / "service"
    environment = _queue_environment(tmp_path, state)
    release = tmp_path / "release"
    try:
        _service(
            state,
            "start",
            environment=environment,
            working_directory=tmp_path,
        )
        launchers = [
            _launcher(tmp_path, environment, "stable", "HOLD_RUN=1")
            for _ in range(4)
        ]
        _wait_for_status(state, lambda value: value["active"] == 1)
        time.sleep(0.25)

        status = json.loads(_service(state, "status", "--json").stdout)
        assert status["active"] == 1
        assert status["queued"] == 0
        release.touch()
        assert [launcher.wait(timeout=8) for launcher in launchers] == [0] * 4
        starts = [
            line
            for line in (tmp_path / "runs").read_text().splitlines()
            if line.startswith("start:stable:")
        ]
        assert len(starts) == 1
        batches = [
            json.loads(path.read_text())
            for path in (state / "batches").glob("*.json")
        ]
        assert len(batches) == 4
        assert len({job for batch in batches for job in batch["jobs"]}) == 1
    finally:
        release.touch(exist_ok=True)
        _stop(state)


def test_service_rejects_different_payload_for_active_stable_run(
    tmp_path: Path,
) -> None:
    state = tmp_path / "service"
    environment = _queue_environment(tmp_path, state)
    release = tmp_path / "release"
    try:
        _service(
            state,
            "start",
            environment=environment,
            working_directory=tmp_path,
        )
        first = _launcher(tmp_path, environment, "stable", "HOLD_RUN=1")
        _wait_for_status(state, lambda value: value["active"] == 1)
        conflicting = _launcher(
            tmp_path, environment, "stable", "HOLD_RUN=1", "RUN_SECONDS=1"
        )

        _, stderr = conflicting.communicate(timeout=8)

        assert conflicting.returncode != 0
        assert "different payload" in stderr
        release.touch()
        assert first.wait(timeout=8) == 0
    finally:
        release.touch(exist_ok=True)
        _stop(state)


def test_service_allows_stable_run_retry_after_failure(tmp_path: Path) -> None:
    state = tmp_path / "service"
    environment = _queue_environment(tmp_path, state)
    try:
        _service(
            state,
            "start",
            environment=environment,
            working_directory=tmp_path,
        )
        failed = _launcher(tmp_path, environment, "stable", "FAIL_RUN=1")
        assert failed.wait(timeout=8) == 1
        retry = _launcher(tmp_path, environment, "stable", "FAIL_RUN=0")
        assert retry.wait(timeout=8) == 0

        batches = [
            json.loads(path.read_text())
            for path in (state / "batches").glob("*.json")
        ]
        assert len({job for batch in batches for job in batch["jobs"]}) == 2
    finally:
        _stop(state)


def test_service_allows_stable_run_resubmission_after_completion(
    tmp_path: Path,
) -> None:
    state = tmp_path / "service"
    environment = _queue_environment(tmp_path, state)
    try:
        _service(
            state,
            "start",
            environment=environment,
            working_directory=tmp_path,
        )
        first = _launcher(tmp_path, environment, "stable")
        assert first.wait(timeout=8) == 0
        second = _launcher(tmp_path, environment, "stable")
        assert second.wait(timeout=8) == 0

        starts = [
            line
            for line in (tmp_path / "runs").read_text().splitlines()
            if line.startswith("start:stable:")
        ]
        batches = [
            json.loads(path.read_text())
            for path in (state / "batches").glob("*.json")
        ]
        assert len(starts) == 2
        assert len({job for batch in batches for job in batch["jobs"]}) == 2
    finally:
        _stop(state)


def test_service_recovers_claim_published_without_a_job(tmp_path: Path) -> None:
    state = tmp_path / "service"
    environment = _queue_environment(tmp_path, state)
    try:
        _service(
            state,
            "start",
            environment=environment,
            working_directory=tmp_path,
        )
        stable_key = hashlib.sha256(b"stable").hexdigest()
        claim = state / "stable" / f"{stable_key}.json"
        claim.write_text(json.dumps({"id": "interrupted", "run": "stable"}))

        launcher = _launcher(tmp_path, environment, "stable")

        assert launcher.wait(timeout=8) == 0
        starts = [
            line
            for line in (tmp_path / "runs").read_text().splitlines()
            if line.startswith("start:stable:")
        ]
        assert len(starts) == 1
        assert json.loads(claim.read_text())["id"] != "interrupted"
    finally:
        _stop(state)


def test_service_reconciles_active_job_with_missing_claim(tmp_path: Path) -> None:
    state = tmp_path / "service"
    environment = _queue_environment(tmp_path, state)
    release = tmp_path / "release"
    try:
        _service(
            state,
            "start",
            environment=environment,
            working_directory=tmp_path,
        )
        first = _launcher(tmp_path, environment, "stable", "HOLD_RUN=1")
        _wait_for_status(state, lambda value: value["active"] == 1)
        stable_key = hashlib.sha256(b"stable").hexdigest()
        (state / "stable" / f"{stable_key}.json").unlink()

        second = _launcher(tmp_path, environment, "stable", "HOLD_RUN=1")
        time.sleep(0.25)

        status = json.loads(_service(state, "status", "--json").stdout)
        assert status["active"] == 1
        assert status["queued"] == 0
        release.touch()
        assert first.wait(timeout=8) == 0
        assert second.wait(timeout=8) == 0
        batches = [
            json.loads(path.read_text())
            for path in (state / "batches").glob("*.json")
        ]
        assert len({job for batch in batches for job in batch["jobs"]}) == 1
    finally:
        release.touch(exist_ok=True)
        _stop(state)


def test_service_waits_for_pending_gpu_check_before_stacking_jobs(
    tmp_path: Path,
) -> None:
    state = tmp_path / "service"
    environment = _queue_environment(tmp_path, state)
    _executable(
        tmp_path / "bin" / "nvidia-smi",
        """#!/usr/bin/env bash
if [[ "$1" == --query-gpu=* ]]; then
    printf '0, GPU-a\n1, GPU-b\n'
elif [[ "$1" == --query-compute-apps=* ]]; then
    printf 'GPU-b\n'
elif [[ "$1" == --id=GPU-b ]]; then
    printf '4, 100, 1000\n'
fi
""",
    )
    environment.update(
        {
            "TRAINING_QUEUE_IN_FLIGHT": "3",
            "TRAINING_QUEUE_MONITOR_LIGHT_GPUS": "1",
            "TRAINING_QUEUE_GPU_CHECK_SECONDS": "1",
            "TRAINING_QUEUE_GPU_SAMPLE_SECONDS": "0.05",
        }
    )
    try:
        _service(
            state,
            "start",
            environment=environment,
            working_directory=tmp_path,
        )
        command = (
            f'TRAINING_QUEUE_SCRIPT=experiment.py; source "{QUEUE}" && '
            "enqueue first VARIANT=first && "
            "enqueue second VARIANT=second && drain"
        )

        result = subprocess.run(
            ["bash", "-c", command],
            cwd=tmp_path,
            env=environment,
            capture_output=True,
            text=True,
            timeout=8,
        )

        assert result.returncode == 0, result.stderr
        events = [line.split(":") for line in (tmp_path / "runs").read_text().splitlines()]
        starts = {row[1]: row[2] for row in events if row[0] == "start"}
        assert starts == {"first": "GPU-a", "second": "GPU-b"}
    finally:
        _stop(state)


def test_start_recovers_when_stale_status_pid_was_reused(tmp_path: Path) -> None:
    state = tmp_path / "service"
    environment = _queue_environment(tmp_path, state)
    state.mkdir()
    process_fields = (
        Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(")", 1)[1].split()
    )
    (state / "status.json").write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "pid_start_time": int(process_fields[19]),
                "instance_token": "stale-recycled-instance",
                "running": True,
            }
        )
    )
    (state / "engine.ready").touch()
    try:
        result = _service(
            state,
            "start",
            environment=environment,
            working_directory=tmp_path,
        )

        assert int(result.stdout) != os.getpid()
        assert json.loads(_service(state, "status", "--json").stdout)["running"]
    finally:
        _stop(state)


def test_pause_keeps_granular_run_queued_until_resume(tmp_path: Path) -> None:
    state = tmp_path / "service"
    environment = _queue_environment(tmp_path, state)
    try:
        _service(
            state,
            "start",
            environment=environment,
            working_directory=tmp_path,
        )
        _service(state, "pause")
        launcher = _launcher(tmp_path, environment, "paused")

        status = _wait_for_status(
            state,
            lambda value: value["paused"] and value["queued"] == 1,
        )
        assert status["active"] == 0
        assert not (tmp_path / "runs").exists()

        _service(state, "resume")
        assert launcher.wait(timeout=8) == 0
        assert "start:paused:" in (tmp_path / "runs").read_text()
    finally:
        _stop(state)


def test_submitted_run_survives_shell_exit_without_launcher_drain(
    tmp_path: Path,
) -> None:
    state = tmp_path / "service"
    environment = _queue_environment(tmp_path, state)
    try:
        _service(
            state,
            "start",
            environment=environment,
            working_directory=tmp_path,
        )
        command = (
            f'TRAINING_QUEUE_SCRIPT=experiment.py; source "{QUEUE}" && '
            "enqueue detached VARIANT=detached"
        )
        subprocess.run(
            ["bash", "-c", command],
            cwd=tmp_path,
            env=environment,
            check=True,
        )

        _wait_for_status(state, lambda value: value["completed"] == 1)
        assert "start:detached:" in (tmp_path / "runs").read_text()
    finally:
        _stop(state)


def test_service_periodically_reaps_after_scheduler_input_becomes_idle(
    tmp_path: Path,
) -> None:
    state = tmp_path / "service"
    service = tmp_path / "service.py"
    service.write_text(SERVICE.read_text())
    _executable(
        tmp_path / "service_scheduler.sh",
        """#!/usr/bin/env bash
state=$1
touch "$state/engine.ready"
job=
results=
while IFS= read -r command; do
    eval "fields=($command)"
    case "${fields[0]}" in
        _service_enqueue)
            job=${fields[2]}
            results=${fields[3]}
            touch "$state/acks/$job"
            ;;
        _service_reap)
            if [[ -n "$job" ]]; then
                printf '0\n' > "$results/$job.result"
                job=
            fi
            ;;
    esac
done
""",
    )

    def command(
        *arguments: str, check: bool = True
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(service), "--state-dir", str(state), *arguments],
            check=check,
            capture_output=True,
            text=True,
            timeout=8,
            cwd=tmp_path,
        )

    try:
        command("start")
        batch = command("new-batch").stdout.strip()
        command(
            "enqueue-run",
            "--batch",
            batch,
            "--script",
            "experiment.py",
            "--run",
            "idle-reap",
        )
        command("seal-batch", batch)

        deadline = time.monotonic() + 4
        while time.monotonic() < deadline:
            status = json.loads(command("status", "--json").stdout)
            if status["completed"] == 1:
                break
            time.sleep(0.02)
        else:
            raise AssertionError(f"service did not reap the idle job: {status}")
        assert status["active"] == 0
    finally:
        command("stop", check=False)


def test_global_drain_waits_active_runs_without_consuming_pending_runs(
    tmp_path: Path,
) -> None:
    state = tmp_path / "service"
    environment = _queue_environment(tmp_path, state)
    try:
        _service(
            state,
            "start",
            environment=environment,
            working_directory=tmp_path,
        )
        first = _launcher(tmp_path, environment, "first", "RUN_SECONDS=0.4")
        _wait_for_status(state, lambda value: value["active"] == 1)

        drain = subprocess.Popen(
            [sys.executable, str(SERVICE), "--state-dir", str(state), "drain"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        _wait_for_status(state, lambda value: value["paused"])
        second = _launcher(tmp_path, environment, "second")
        _wait_for_status(state, lambda value: value["queued"] == 1)

        assert drain.wait(timeout=8) == 0
        assert first.wait(timeout=8) == 0
        status = json.loads(_service(state, "status", "--json").stdout)
        assert status["active"] == 0
        assert status["queued"] == 1
        assert "second" not in (tmp_path / "runs").read_text()

        _service(state, "resume")
        assert second.wait(timeout=8) == 0
    finally:
        _stop(state)


def test_failed_run_fails_only_its_launcher_batch(tmp_path: Path) -> None:
    state = tmp_path / "service"
    environment = _queue_environment(tmp_path, state)
    try:
        _service(
            state,
            "start",
            environment=environment,
            working_directory=tmp_path,
        )
        broken = _launcher(tmp_path, environment, "broken", "FAIL_RUN=1")
        healthy = _launcher(tmp_path, environment, "healthy")

        assert broken.wait(timeout=8) != 0
        assert healthy.wait(timeout=8) == 0
        status = json.loads(_service(state, "status", "--json").stdout)
        assert status["completed"] == 1
        assert status["failed"] == 1
    finally:
        _stop(state)


def test_forwarded_environment_reaches_actual_dcn_main_ownership(
    tmp_path: Path,
) -> None:
    state = tmp_path / "service"
    environment = _real_dcn_queue_environment(tmp_path, state)
    experiment = tmp_path / "ownership_experiment.py"
    experiment.write_text(
        """import os
from pathlib import Path
from time import sleep

from experiments.g1_sasrec_item_ids_likes.configs.rq_tuning_variant import experiment


class OwnershipStage:
    name = "ownership"
    def run(self):
        marker = Path(os.environ["DCN_PREPARED_MARKER"])
        release = Path(os.environ["DCN_TRAINING_RELEASE"])
        marker.touch()
        while not release.exists():
            sleep(0.01)


experiment.setup = lambda: None
experiment.__class__.stages = property(lambda self: (OwnershipStage(),))
"""
    )
    artifacts = (
        REPOSITORY
        / "experiments"
        / "g1_sasrec_item_ids_likes"
        / "launchers"
        / "artifacts.sh"
    )
    try:
        _service(
            state,
            "start",
            environment=environment,
            working_directory=tmp_path,
        )
        command = (
            f'source "{artifacts}"; '
            "export G1_DATASET_SIZE=50m; "
            "export WANDB_MODE=offline; "
            f'TRAINING_QUEUE_SCRIPT="{experiment}"; source "{QUEUE}" && '
            "enqueue g1_rqtune_ownership_ts2_r2_50m "
            "G1_TUNE_RUN=ownership_ts2_r2 "
            "G1_TUNE_SOURCE_VARIANT=baseline && drain"
        )

        result = subprocess.run(
            ["bash", "-c", command],
            cwd=tmp_path,
            env=environment,
            capture_output=True,
            text=True,
            timeout=20,
        )

        assert result.returncode == 0, result.stderr
        status = json.loads(_service(state, "status", "--json").stdout)
        assert status["completed"] == 1
        assert status["failed"] == 0
    finally:
        _stop(state)


def test_required_forwarded_environment_fails_before_batch_creation(
    tmp_path: Path,
) -> None:
    state = tmp_path / "service"
    environment = _queue_environment(tmp_path, state)
    try:
        _service(
            state,
            "start",
            environment=environment,
            working_directory=tmp_path,
        )
        command = (
            "unset REQUIRED_DATASET; "
            "export TRAINING_QUEUE_FORWARD_ENV=REQUIRED_DATASET; "
            "export TRAINING_QUEUE_REQUIRED_FORWARD_ENV=REQUIRED_DATASET; "
            f'TRAINING_QUEUE_SCRIPT=experiment.py; source "{QUEUE}"'
        )

        result = subprocess.run(
            ["bash", "-c", command],
            cwd=tmp_path,
            env=environment,
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0
        assert "Required forwarded variable is unset: REQUIRED_DATASET" in result.stderr
        assert not list((state / "batches").glob("*.json"))
    finally:
        _stop(state)


def test_secret_forwarded_environment_fails_before_batch_creation(
    tmp_path: Path,
) -> None:
    state = tmp_path / "service"
    environment = _queue_environment(tmp_path, state)
    try:
        _service(
            state,
            "start",
            environment=environment,
            working_directory=tmp_path,
        )
        for name in ("wandb_api_key", "GITHUB_TOKEN", "HF_TOKEN"):
            command = (
                f"export {name}=secret-value; "
                f"export TRAINING_QUEUE_FORWARD_ENV={name}; "
                f'TRAINING_QUEUE_SCRIPT=experiment.py; source "{QUEUE}"'
            )
            result = subprocess.run(
                ["bash", "-c", command],
                cwd=tmp_path,
                env=environment,
                capture_output=True,
                text=True,
            )
            assert result.returncode != 0
            assert f"Cannot persist secret-like variable: {name}" in result.stderr
            assert not list((state / "batches").glob("*.json"))

        safe_names = (
            "BEGINNING_TOKEN CLS_TOKEN PADDING_TOKEN NUM_SPECIAL_TOKENS "
            "G1_TRANSFER_POWER_TOKENS"
        )
        command = (
            "export BEGINNING_TOKEN=101 CLS_TOKEN=102 PADDING_TOKEN=0; "
            "export NUM_SPECIAL_TOKENS=3 G1_TRANSFER_POWER_TOKENS=1500000; "
            f"export TRAINING_QUEUE_FORWARD_ENV='{safe_names}'; "
            f'TRAINING_QUEUE_SCRIPT=experiment.py; source "{QUEUE}"'
        )
        result = subprocess.run(
            ["bash", "-c", command],
            cwd=tmp_path,
            env=environment,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert len(list((state / "batches").glob("*.json"))) == 1
    finally:
        _stop(state)


def test_service_rejects_secret_assignment_before_persisting_job(
    tmp_path: Path,
) -> None:
    state = tmp_path / "service"
    environment = _queue_environment(tmp_path, state)
    try:
        _service(
            state,
            "start",
            environment=environment,
            working_directory=tmp_path,
        )
        batch_id = _service(state, "new-batch").stdout.strip()

        for assignment in (
            "wandb_api_key=secret-value",
            "GITHUB_TOKEN=secret-value",
            "HF_TOKEN=secret-value",
        ):
            result = _service(
                state,
                "enqueue-run",
                "--batch",
                batch_id,
                "--script",
                "experiment.py",
                "--run",
                "secret",
                "--",
                assignment,
                check=False,
            )
            assert result.returncode != 0
            assert "refusing to persist secret-like variable" in result.stderr
        assert not list((state / "pending").glob("*.json"))
        assert json.loads((state / "batches" / f"{batch_id}.json").read_text())[
            "jobs"
        ] == []

        safe = _service(
            state,
            "enqueue-run",
            "--batch",
            batch_id,
            "--script",
            "experiment.py",
            "--run",
            "token-count",
            "--",
            "G1_TRANSFER_POWER_TOKENS=1500000",
            "CLS_TOKEN=101",
            "BEGINNING_TOKEN=102",
            "PADDING_TOKEN=0",
            "NUM_SPECIAL_TOKENS=3",
            check=False,
        )
        assert safe.returncode == 0, safe.stderr
    finally:
        _stop(state)


def test_g1_launchers_register_dataset_and_wandb_provenance() -> None:
    artifacts = (
        REPOSITORY
        / "experiments"
        / "g1_sasrec_item_ids_likes"
        / "launchers"
        / "artifacts.sh"
    )

    result = subprocess.run(
        [
            "bash",
            "-c",
            f'source "{artifacts}"; '
            "printf '%s\\n%s\\n' \"$TRAINING_QUEUE_FORWARD_ENV\" "
            "\"$TRAINING_QUEUE_REQUIRED_FORWARD_ENV\"",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    forwarded, required = result.stdout.splitlines()
    assert set(forwarded.split()) >= {"G1_DATASET_SIZE", "WANDB_MODE"}
    assert set(required.split()) >= {"G1_DATASET_SIZE", "WANDB_MODE"}


@pytest.mark.parametrize("require_kind", ["tuning", "config"])
@pytest.mark.parametrize("artifact_state", ["incompatible", "resumable"])
def test_g1_launcher_archives_invalidated_artifacts(
    tmp_path: Path,
    artifact_state: str,
    require_kind: str,
) -> None:
    artifacts = (
        REPOSITORY
        / "experiments"
        / "g1_sasrec_item_ids_likes"
        / "launchers"
        / "artifacts.sh"
    )
    logs = tmp_path / "logs"
    run = logs / "g1_example_50m"
    run.mkdir(parents=True)
    (run / "marker").write_text("preserved")
    if require_kind == "tuning":
        classifier = "g1_classify_tuning_artifact"
        requirement = f'g1_require_compatible_or_absent "{run}" 50m example'
    else:
        classifier = "g1_classify_config_artifact"
        requirement = f'g1_require_config_compatible_or_absent "{run}" config.py example'
    command = f"""
source "{artifacts}"
{classifier}() {{ _g1_artifact_state={artifact_state}; }}
{requirement}
printf '%s\n' "$?"
mkdir -p "{run}"
printf newer > "{run}/marker"
{requirement}
printf '%s\n' "$?"
"""

    result = subprocess.run(
        ["bash", "-c", command],
        capture_output=True,
        text=True,
        check=True,
    )

    reason = "incomplete" if artifact_state == "resumable" else artifact_state
    archived = logs / "old" / f"{run.name}.{reason}-001"
    archived_again = logs / "old" / f"{run.name}.{reason}-002"
    statuses = [line for line in result.stdout.splitlines() if line.isdigit()]
    assert statuses == ["1", "1"]
    assert not run.exists()
    assert (archived / "marker").read_text() == "preserved"
    assert (archived_again / "marker").read_text() == "newer"
