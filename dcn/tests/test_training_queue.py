import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


QUEUE = (
    Path(__file__).resolve().parents[2] / "utils/training_queue/queue.sh"
)
QUEUE_DEPTH = QUEUE.with_name("queue_depth.py")


def _executable(path: Path, source: str) -> None:
    path.write_text(source)
    path.chmod(0o755)


def test_queue_uses_only_idle_gpus_and_distributes_runs(tmp_path: Path) -> None:
    binaries = tmp_path / "bin"
    binaries.mkdir()
    _executable(
        binaries / "nvidia-smi",
        """#!/usr/bin/env bash
if [[ "$1" == --query-gpu=* ]]; then
    printf '0, GPU-a\n1, GPU-b\n2, GPU-c\n'
else
    printf 'GPU-b\n'
fi
""",
    )
    _executable(
        binaries / "python",
        """#!/usr/bin/env bash
touch "$DCN_PREPARED_MARKER"
while [ ! -e "$DCN_TRAINING_RELEASE" ]; do sleep 0.01; done
printf '%s:%s:%s\n' "$VARIANT" "$CUDA_VISIBLE_DEVICES" "$*" >> "$QUEUE_RECORD"
""",
    )
    record = tmp_path / "runs"
    environment = {
        **os.environ,
        "PATH": f"{binaries}:{os.environ['PATH']}",
        "QUEUE_RECORD": str(record),
        "TRAINING_QUEUE_IN_FLIGHT": "1",
        "TRAINING_QUEUE_SCRIPT": "custom_experiment.py",
    }

    subprocess.run(
        [
            "bash",
            "-c",
            f'set -u; source "{QUEUE}" && '
            "enqueue first VARIANT=dim_16 && "
            "enqueue second VARIANT=dim_32 && "
            "enqueue third VARIANT=dim_64 && drain",
        ],
        cwd=tmp_path,
        env=environment,
        check=True,
    )

    assignments = {
        variant: (gpu, arguments)
        for variant, gpu, arguments in (
            line.split(":") for line in record.read_text().splitlines()
        )
    }
    assert assignments["dim_16"][0] == "GPU-a"
    assert assignments["dim_32"][0] == "GPU-c"
    assert assignments["dim_64"][0] in {"GPU-a", "GPU-c"}
    assert all(
        arguments.endswith("-s custom_experiment.py")
        for _, arguments in assignments.values()
    )


def test_embedded_queue_records_compatible_timing_history(tmp_path: Path) -> None:
    binaries = tmp_path / "bin"
    binaries.mkdir()
    _executable(
        binaries / "nvidia-smi",
        """#!/usr/bin/env bash
if [[ "$1" == --query-gpu=* ]]; then printf '0, GPU-a\n'; fi
""",
    )
    _executable(
        binaries / "python",
        """#!/usr/bin/env bash
touch "$DCN_PREPARED_MARKER"
while [ ! -e "$DCN_TRAINING_RELEASE" ]; do sleep 0.01; done
printf "Prepared stage 'model' in 1.0s\n"
printf "Trained stage 'model' in 2.0s\n"
""",
    )
    state = tmp_path / "queue-state"
    environment = {
        **os.environ,
        "PATH": f"{binaries}:{os.environ['PATH']}",
        "TRAINING_QUEUE_CONTROL_PYTHON": sys.executable,
        "TRAINING_QUEUE_DATA_GROUP": "dataset-a",
        "TRAINING_QUEUE_IN_FLIGHT": "1",
        "TRAINING_QUEUE_SCRIPT": "experiment.py",
        "TRAINING_QUEUE_SERVICE_STATE_DIR": str(state),
    }

    subprocess.run(
        [
            "bash",
            "-c",
            f'source "{QUEUE}" && enqueue only VARIANT=only && drain',
        ],
        cwd=tmp_path,
        env=environment,
        check=True,
    )

    assert json.loads(state.joinpath("timing-history.json").read_text()) == {
        "entries": [
            {
                "data_group": "dataset-a",
                "ratio": 0.5,
                "script": "experiment.py",
            }
        ],
        "version": 1,
    }


def test_queue_rejects_multiple_training_slots_per_gpu(tmp_path: Path) -> None:
    binaries = tmp_path / "bin"
    binaries.mkdir()
    _executable(
        binaries / "nvidia-smi",
        """#!/usr/bin/env bash
if [[ "$1" == --query-gpu=* ]]; then printf '0, GPU-a\n'; fi
""",
    )
    environment = {
        **os.environ,
        "PATH": f"{binaries}:{os.environ['PATH']}",
        "TRAINING_QUEUE_RUNS_PER_GPU": "2",
        "TRAINING_QUEUE_SCRIPT": "custom_experiment.py",
    }

    result = subprocess.run(
        ["bash", "-c", f'source "{QUEUE}"'],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "one simultaneous training run per GPU" in result.stderr


def test_queue_defaults_cuda_allocator_and_preserves_override(tmp_path: Path) -> None:
    binaries = tmp_path / "bin"
    binaries.mkdir()
    _executable(
        binaries / "nvidia-smi",
        """#!/usr/bin/env bash
if [[ "$1" == --query-gpu=* ]]; then printf '0, GPU-a\n'; fi
""",
    )
    _executable(
        binaries / "python",
        """#!/usr/bin/env bash
touch "$DCN_PREPARED_MARKER"
while [ ! -e "$DCN_TRAINING_RELEASE" ]; do sleep 0.01; done
printf '%s\n' "$PYTORCH_CUDA_ALLOC_CONF" > "$QUEUE_RECORD"
""",
    )
    base_environment = {
        **os.environ,
        "PATH": f"{binaries}:{os.environ['PATH']}",
        "TRAINING_QUEUE_IN_FLIGHT": "1",
        "TRAINING_QUEUE_SCRIPT": "custom_experiment.py",
    }

    for case, (ambient_config, run_config, expected) in enumerate(
        (
            (None, None, "expandable_segments:True"),
            ("max_split_size_mb:256", None, "max_split_size_mb:256"),
            ("", None, ""),
            (
                "max_split_size_mb:256",
                "garbage_collection_threshold:0.8",
                "garbage_collection_threshold:0.8",
            ),
        )
    ):
        record = tmp_path / f"runs-{case}"
        environment = {
            key: value
            for key, value in base_environment.items()
            if key != "PYTORCH_CUDA_ALLOC_CONF"
        }
        environment["QUEUE_RECORD"] = str(record)
        if ambient_config is not None:
            environment["PYTORCH_CUDA_ALLOC_CONF"] = ambient_config

        assignment = (
            ""
            if run_config is None
            else f" PYTORCH_CUDA_ALLOC_CONF={run_config}"
        )

        subprocess.run(
            [
                "bash",
                "-c",
                f'source "{QUEUE}" && enqueue only VARIANT=only{assignment} && drain',
            ],
            cwd=tmp_path,
            env=environment,
            check=True,
        )

        assert record.read_text().strip() == expected


def test_queue_starts_training_while_next_run_is_preparing(tmp_path: Path) -> None:
    binaries = tmp_path / "bin"
    binaries.mkdir()
    _executable(
        binaries / "nvidia-smi",
        """#!/usr/bin/env bash
if [[ "$1" == --query-gpu=* ]]; then printf '0, GPU-a\n'; fi
""",
    )
    _executable(
        binaries / "python",
        """#!/usr/bin/env bash
if [ "$VARIANT" = second ]; then
    while [ ! -e "$FIRST_STARTED" ]; do sleep 0.01; done
fi
touch "$DCN_PREPARED_MARKER"
while [ ! -e "$DCN_TRAINING_RELEASE" ]; do sleep 0.01; done
printf '%s:%s\n' "$VARIANT" "$DCN_GPU_LOCK_SLOT" >> "$QUEUE_RECORD"
if [ "$VARIANT" = first ]; then touch "$FIRST_STARTED"; fi
""",
    )
    record = tmp_path / "runs"
    environment = {
        **os.environ,
        "PATH": f"{binaries}:{os.environ['PATH']}",
        "QUEUE_RECORD": str(record),
        "FIRST_STARTED": str(tmp_path / "first_started"),
        "TRAINING_QUEUE_IN_FLIGHT": "2",
        "TRAINING_QUEUE_SCRIPT": "custom_experiment.py",
    }

    subprocess.run(
        [
            "bash",
            "-c",
            f'set -u; source "{QUEUE}" && '
            "enqueue first VARIANT=first && "
            "enqueue second VARIANT=second && drain",
        ],
        cwd=tmp_path,
        env=environment,
        check=True,
        timeout=3,
    )

    assert set(record.read_text().splitlines()) == {"first:0", "second:0"}


def test_queue_starts_lookahead_without_waiting_for_every_gpu_to_prepare(
    tmp_path: Path,
) -> None:
    binaries = tmp_path / "bin"
    binaries.mkdir()
    _executable(
        binaries / "nvidia-smi",
        """#!/usr/bin/env bash
if [[ "$1" == --query-gpu=* ]]; then printf '0, GPU-a\n1, GPU-b\n'; fi
""",
    )
    _executable(
        binaries / "python",
        """#!/usr/bin/env bash
case "$VARIANT" in
    first)
        sleep 0.05
        touch "$FIRST_PREPARED"
        ;;
    second)
        sleep 0.2
        [ -e "$FIRST_STARTED" ] || touch "$FIRST_START_DELAYED"
        touch "$SECOND_PREPARED"
        ;;
    *)
        if [ ! -e "$FIRST_PREPARED" ] || [ ! -e "$SECOND_PREPARED" ]; then
            touch "$LOOKAHEAD_STARTED_EARLY"
        fi
        ;;
esac
touch "$DCN_PREPARED_MARKER"
while [ ! -e "$DCN_TRAINING_RELEASE" ]; do sleep 0.01; done
if [ "$VARIANT" = first ]; then touch "$FIRST_STARTED"; fi
""",
    )
    environment = {
        **os.environ,
        "PATH": f"{binaries}:{os.environ['PATH']}",
        "FIRST_PREPARED": str(tmp_path / "first_prepared"),
        "SECOND_PREPARED": str(tmp_path / "second_prepared"),
        "FIRST_STARTED": str(tmp_path / "first_started"),
        "FIRST_START_DELAYED": str(tmp_path / "first_start_delayed"),
        "LOOKAHEAD_STARTED_EARLY": str(tmp_path / "lookahead_started_early"),
        "TRAINING_QUEUE_IN_FLIGHT": "2",
        "TRAINING_QUEUE_SCRIPT": "custom_experiment.py",
    }

    subprocess.run(
        [
            "bash",
            "-c",
            f'set -u; source "{QUEUE}" && '
            "enqueue first VARIANT=first && "
            "enqueue second VARIANT=second && "
            "enqueue third VARIANT=third && "
            "enqueue fourth VARIANT=fourth && drain",
        ],
        cwd=tmp_path,
        env=environment,
        check=True,
    )

    assert (tmp_path / "lookahead_started_early").exists()
    assert not (tmp_path / "first_start_delayed").exists()


def test_failure_after_training_starts_makes_drain_fail(tmp_path: Path) -> None:
    binaries = tmp_path / "bin"
    binaries.mkdir()
    _executable(
        binaries / "nvidia-smi",
        """#!/usr/bin/env bash
if [[ "$1" == --query-gpu=* ]]; then printf '0, GPU-a\n'; fi
""",
    )
    _executable(
        binaries / "python",
        """#!/usr/bin/env bash
touch "$DCN_PREPARED_MARKER"
while [ ! -e "$DCN_TRAINING_RELEASE" ]; do sleep 0.01; done
[ "$VARIANT" != broken ]
""",
    )
    environment = {
        **os.environ,
        "PATH": f"{binaries}:{os.environ['PATH']}",
        "TRAINING_QUEUE_IN_FLIGHT": "2",
        "TRAINING_QUEUE_SCRIPT": "custom_experiment.py",
    }

    result = subprocess.run(
        [
            "bash",
            "-c",
            f'set -u; source "{QUEUE}" && '
            "enqueue first VARIANT=first && "
            "enqueue broken VARIANT=broken && drain",
        ],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "One or more queued runs failed" in result.stderr


def test_queue_uses_free_gpu_while_monitoring_lightly_used_gpu(
    tmp_path: Path,
) -> None:
    binaries = tmp_path / "bin"
    binaries.mkdir()
    _executable(
        binaries / "nvidia-smi",
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
    _executable(
        binaries / "python",
        """#!/usr/bin/env bash
touch "$DCN_PREPARED_MARKER"
while [ ! -e "$DCN_TRAINING_RELEASE" ]; do sleep 0.01; done
printf '%s:%s\n' "$VARIANT" "$CUDA_VISIBLE_DEVICES" >> "$QUEUE_RECORD"
sleep 1
""",
    )
    record = tmp_path / "runs"
    environment = {
        **os.environ,
        "PATH": f"{binaries}:{os.environ['PATH']}",
        "QUEUE_RECORD": str(record),
        "TRAINING_QUEUE_IN_FLIGHT": "1",
        "TRAINING_QUEUE_MONITOR_LIGHT_GPUS": "1",
        "TRAINING_QUEUE_GPU_CHECK_SECONDS": "0.05",
        "TRAINING_QUEUE_GPU_SAMPLE_SECONDS": "0.01",
        "TRAINING_QUEUE_CONTROL_PYTHON": sys.executable,
        "TRAINING_QUEUE_SCRIPT": "custom_experiment.py",
    }

    result = subprocess.run(
        [
            "bash",
            "-c",
            f'source "{QUEUE}" && '
            "enqueue first VARIANT=first && "
            "enqueue second VARIANT=second && drain",
        ],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )

    assignments = dict(line.split(":") for line in record.read_text().splitlines())
    assert assignments == {"first": "GPU-a", "second": "GPU-b"}
    assert "monitoring lightly used GPU 1" in result.stdout
    assert "admitted lightly used GPU 1" in result.stdout


def test_queue_waits_for_pending_gpu_check_before_stacking_lookahead(
    tmp_path: Path,
) -> None:
    binaries = tmp_path / "bin"
    binaries.mkdir()
    _executable(
        binaries / "nvidia-smi",
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
    _executable(
        binaries / "python",
        """#!/usr/bin/env bash
touch "$DCN_PREPARED_MARKER"
while [ ! -e "$DCN_TRAINING_RELEASE" ]; do sleep 0.01; done
printf '%s:%s\n' "$VARIANT" "$CUDA_VISIBLE_DEVICES" >> "$QUEUE_RECORD"
sleep 0.3
""",
    )
    record = tmp_path / "runs"
    environment = {
        **os.environ,
        "PATH": f"{binaries}:{os.environ['PATH']}",
        "QUEUE_RECORD": str(record),
        "TRAINING_QUEUE_IN_FLIGHT": "3",
        "TRAINING_QUEUE_MONITOR_LIGHT_GPUS": "1",
        "TRAINING_QUEUE_GPU_CHECK_SECONDS": "0.5",
        "TRAINING_QUEUE_GPU_SAMPLE_SECONDS": "0.05",
        "TRAINING_QUEUE_SCRIPT": "custom_experiment.py",
    }

    result = subprocess.run(
        [
            "bash",
            "-c",
            f'set -u; source "{QUEUE}" && '
            "enqueue first VARIANT=first && "
            "enqueue second VARIANT=second && drain",
        ],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
        timeout=3,
    )

    assignments = dict(line.split(":") for line in record.read_text().splitlines())
    assert assignments == {"first": "GPU-a", "second": "GPU-b"}
    assert "monitoring lightly used GPU 1" in result.stdout
    assert "admitted lightly used GPU 1" in result.stdout


def test_queue_keeps_changed_light_gpu_excluded(tmp_path: Path) -> None:
    binaries = tmp_path / "bin"
    binaries.mkdir()
    sample_count = tmp_path / "sample_count"
    _executable(
        binaries / "nvidia-smi",
        f"""#!/usr/bin/env bash
if [[ "$1" == --query-gpu=* ]]; then
    printf '0, GPU-a\n1, GPU-b\n'
elif [[ "$1" == --query-compute-apps=* ]]; then
    printf 'GPU-b\n'
elif [[ "$1" == --id=GPU-b ]]; then
    count=$(cat "{sample_count}" 2>/dev/null || printf 0)
    count=$((count + 1))
    printf '%s' "$count" > "{sample_count}"
    printf '%s, 100, 1000\n' "$((9 + count))"
fi
""",
    )
    _executable(
        binaries / "python",
        """#!/usr/bin/env bash
touch "$DCN_PREPARED_MARKER"
while [ ! -e "$DCN_TRAINING_RELEASE" ]; do sleep 0.01; done
printf '%s:%s\n' "$VARIANT" "$CUDA_VISIBLE_DEVICES" >> "$QUEUE_RECORD"
sleep 0.1
""",
    )
    record = tmp_path / "runs"
    evidence = tmp_path / "gpu-check-evidence"
    environment = {
        **os.environ,
        "PATH": f"{binaries}:{os.environ['PATH']}",
        "QUEUE_RECORD": str(record),
        "TRAINING_QUEUE_IN_FLIGHT": "1",
        "TRAINING_QUEUE_CONTROL_PYTHON": sys.executable,
        "TRAINING_QUEUE_MONITOR_LIGHT_GPUS": "1",
        "TRAINING_QUEUE_GPU_CHECK_SECONDS": "0.05",
        "TRAINING_QUEUE_GPU_SAMPLE_SECONDS": "0.01",
        "TRAINING_QUEUE_GPU_CHECK_EVIDENCE_DIR": str(evidence),
        "TRAINING_QUEUE_SCRIPT": "custom_experiment.py",
    }

    result = subprocess.run(
        [
            "bash",
            "-c",
            f'set -u; source "{QUEUE}" && '
            "enqueue first VARIANT=first && "
            "enqueue second VARIANT=second && drain",
        ],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )

    assignments = dict(line.split(":") for line in record.read_text().splitlines())
    assert assignments == {"first": "GPU-a", "second": "GPU-a"}
    assert "kept GPU 1 excluded" in result.stdout
    records = list(evidence.glob("gpu-check-GPU-b-*.json"))
    assert len(records) == 1
    decision = json.loads(records[0].read_text())
    assert decision["gpu"] == "GPU-b"
    assert decision["admitted"] is False
    assert len(decision["samples"]) > 1


def test_periodic_recheck_retains_each_rejected_attempt(tmp_path: Path) -> None:
    binaries = tmp_path / "bin"
    binaries.mkdir()
    sample_count = tmp_path / "sample_count"
    _executable(
        binaries / "nvidia-smi",
        f"""#!/usr/bin/env bash
if [[ "$1" == --query-gpu=* ]]; then
    printf '0, GPU-a\n'
elif [[ "$1" == --id=GPU-a ]]; then
    count=$(cat "{sample_count}" 2>/dev/null || printf 0)
    count=$((count + 1))
    printf '%s' "$count" > "{sample_count}"
    if [ "$count" -eq 1 ]; then
        printf '20, 100, 1000\n'
    else
        printf '0, 0, 1000\n'
    fi
fi
""",
    )
    evidence = tmp_path / "gpu-check-evidence"
    environment = {
        **os.environ,
        "PATH": f"{binaries}:{os.environ['PATH']}",
        "TRAINING_QUEUE_GPU_CHECK_EVIDENCE_DIR": str(evidence),
        "TRAINING_QUEUE_GPU_CHECK_SECONDS": "0",
        "TRAINING_QUEUE_GPU_RETRY_SECONDS": "0",
        "TRAINING_QUEUE_GPU_SETTLE_SECONDS": "0",
        "TRAINING_QUEUE_IN_FLIGHT": "1",
        "TRAINING_QUEUE_MONITOR_LIGHT_GPUS": "1",
        "TRAINING_QUEUE_SCRIPT": "custom_experiment.py",
    }

    subprocess.run(
        [
            "bash",
            "-c",
            f'set -u; source "{QUEUE}" && '
            "_start_gpu_check GPU-a rechecking && "
            "while _job_is_running \"${_gpu_check_pid[GPU-a]}\"; do sleep 0.01; done && "
            "_sync_gpu_checks && drain",
        ],
        cwd=tmp_path,
        env=environment,
        check=True,
    )

    records = list(evidence.glob("gpu-check-GPU-a-*.json"))
    assert len(records) == 1
    assert json.loads(records[0].read_text())["admitted"] is False


def test_queue_rejects_unavailable_gpu_check_evidence_storage(tmp_path: Path) -> None:
    binaries = tmp_path / "bin"
    binaries.mkdir()
    _executable(
        binaries / "nvidia-smi",
        """#!/usr/bin/env bash
if [[ "$1" == --query-gpu=* ]]; then printf '0, GPU-a\n'; fi
""",
    )
    evidence_file = tmp_path / "not-a-directory"
    evidence_file.write_text("preserve me")
    environment = {
        **os.environ,
        "PATH": f"{binaries}:{os.environ['PATH']}",
        "TRAINING_QUEUE_GPU_CHECK_EVIDENCE_DIR": str(evidence_file),
        "TRAINING_QUEUE_IN_FLIGHT": "1",
        "TRAINING_QUEUE_MONITOR_LIGHT_GPUS": "1",
        "TRAINING_QUEUE_SCRIPT": "custom_experiment.py",
    }

    result = subprocess.run(
        ["bash", "-c", f'source "{QUEUE}"'],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert evidence_file.read_text() == "preserve me"


def test_drain_stops_gpu_monitor(tmp_path: Path) -> None:
    binaries = tmp_path / "bin"
    binaries.mkdir()
    _executable(
        binaries / "nvidia-smi",
        """#!/usr/bin/env bash
if [[ "$1" == --query-gpu=* ]]; then
    printf '0, GPU-a\n1, GPU-cancellation-test\n'
elif [[ "$1" == --query-compute-apps=* ]]; then
    printf 'GPU-cancellation-test\n'
fi
""",
    )
    _executable(
        binaries / "python3",
        """#!/usr/bin/env bash
printf '%s' "$$" > "$GPU_CHECK_PID_RECORD"
trap 'exit 0' TERM
while true; do sleep 1; done
""",
    )
    _executable(
        binaries / "python",
        """#!/usr/bin/env bash
touch "$DCN_PREPARED_MARKER"
while [ ! -e "$DCN_TRAINING_RELEASE" ]; do sleep 0.01; done
""",
    )
    pid_record = tmp_path / "checker_pid"
    environment = {
        **os.environ,
        "PATH": f"{binaries}:{os.environ['PATH']}",
        "GPU_CHECK_PID_RECORD": str(pid_record),
        "TRAINING_QUEUE_CONTROL_PYTHON": sys.executable,
        "TRAINING_QUEUE_IN_FLIGHT": "1",
        "TRAINING_QUEUE_MONITOR_LIGHT_GPUS": "1",
        "TRAINING_QUEUE_GPU_CHECK_SECONDS": "30",
        "TRAINING_QUEUE_SCRIPT": "custom_experiment.py",
    }

    subprocess.run(
        [
            "bash",
            "-c",
            f'set -u; source "{QUEUE}" && enqueue only VARIANT=only && drain',
        ],
        cwd=tmp_path,
        env=environment,
        check=True,
    )

    checker_pid = int(pid_record.read_text())
    try:
        deadline = time.monotonic() + 1
        while Path(f"/proc/{checker_pid}").exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not Path(f"/proc/{checker_pid}").exists()
    finally:
        if Path(f"/proc/{checker_pid}").exists():
            os.kill(checker_pid, signal.SIGTERM)


def test_drain_stops_periodic_monitor_and_releases_gpu_gate(
    tmp_path: Path,
) -> None:
    binaries = tmp_path / "bin"
    binaries.mkdir()
    _executable(
        binaries / "nvidia-smi",
        """#!/usr/bin/env bash
if [[ "$1" == --query-gpu=* ]]; then printf '0, GPU-a\n'; fi
""",
    )
    _executable(
        binaries / "python3",
        """#!/usr/bin/env bash
printf '%s' "$$" > "$GPU_CHECK_PID_RECORD"
trap 'exit 0' TERM
while true; do sleep 1; done
""",
    )
    pid_record = tmp_path / "checker_pid"
    environment = {
        **os.environ,
        "PATH": f"{binaries}:{os.environ['PATH']}",
        "GPU_CHECK_PID_RECORD": str(pid_record),
        "TRAINING_QUEUE_IN_FLIGHT": "1",
        "TRAINING_QUEUE_GPU_SETTLE_SECONDS": "0",
        "TRAINING_QUEUE_SCRIPT": "custom_experiment.py",
    }

    subprocess.run(
        [
            "bash",
            "-c",
            f'set -u; source "{QUEUE}" && '
            "_start_gpu_check GPU-a rechecking && "
            'while [ ! -s "$GPU_CHECK_PID_RECORD" ]; do sleep 0.01; done && '
            "drain && flock -n generated/gpu-GPU-a.lock -c true",
        ],
        cwd=tmp_path,
        env=environment,
        check=True,
        timeout=3,
    )

    checker_pid = int(pid_record.read_text())
    try:
        deadline = time.monotonic() + 1
        while Path(f"/proc/{checker_pid}").exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not Path(f"/proc/{checker_pid}").exists()
    finally:
        if Path(f"/proc/{checker_pid}").exists():
            os.kill(checker_pid, signal.SIGTERM)


def test_queue_can_start_after_only_lightly_used_gpu_is_admitted(
    tmp_path: Path,
) -> None:
    binaries = tmp_path / "bin"
    binaries.mkdir()
    _executable(
        binaries / "nvidia-smi",
        """#!/usr/bin/env bash
if [[ "$1" == --query-gpu=* ]]; then
    printf '2, GPU-c\n'
elif [[ "$1" == --query-compute-apps=* ]]; then
    printf 'GPU-c\n'
elif [[ "$1" == --id=GPU-c ]]; then
    printf '10, 190, 1000\n'
fi
""",
    )
    _executable(
        binaries / "python",
        """#!/usr/bin/env bash
touch "$DCN_PREPARED_MARKER"
while [ ! -e "$DCN_TRAINING_RELEASE" ]; do sleep 0.01; done
printf '%s\n' "$CUDA_VISIBLE_DEVICES" > "$QUEUE_RECORD"
""",
    )
    record = tmp_path / "run"
    environment = {
        **os.environ,
        "PATH": f"{binaries}:{os.environ['PATH']}",
        "QUEUE_RECORD": str(record),
        "TRAINING_QUEUE_MONITOR_LIGHT_GPUS": "1",
        "TRAINING_QUEUE_GPU_CHECK_SECONDS": "0.02",
        "TRAINING_QUEUE_GPU_SAMPLE_SECONDS": "0.01",
        "TRAINING_QUEUE_SCRIPT": "custom_experiment.py",
    }

    subprocess.run(
        [
            "bash",
            "-c",
            f'set -u; source "{QUEUE}" && enqueue only VARIANT=only && drain',
        ],
        cwd=tmp_path,
        env=environment,
        check=True,
    )

    assert record.read_text().strip() == "GPU-c"


def test_queue_retries_when_all_monitored_gpus_are_initially_rejected(
    tmp_path: Path,
) -> None:
    binaries = tmp_path / "bin"
    binaries.mkdir()
    sample_count = tmp_path / "sample_count"
    checks = tmp_path / "checks"
    _executable(
        binaries / "nvidia-smi",
        f"""#!/usr/bin/env bash
if [[ "$1" == --query-gpu=* ]]; then
    printf '2, GPU-c\n'
elif [[ "$1" == --query-compute-apps=* ]]; then
    printf 'GPU-c\n'
elif [[ "$1" == --id=GPU-c ]]; then
    printf '%s\n' "$(date +%s%N)" >> "{checks}"
    count=$(cat "{sample_count}" 2>/dev/null || printf 0)
    count=$((count + 1))
    printf '%s' "$count" > "{sample_count}"
    if [ "$count" -eq 1 ]; then
        printf '20, 100, 1000\n'
    else
        printf '0, 0, 1000\n'
    fi
fi
""",
    )
    _executable(
        binaries / "python",
        """#!/usr/bin/env bash
touch "$DCN_PREPARED_MARKER"
while [ ! -e "$DCN_TRAINING_RELEASE" ]; do sleep 0.01; done
printf '%s\n' "$CUDA_VISIBLE_DEVICES" > "$QUEUE_RECORD"
""",
    )
    record = tmp_path / "run"
    environment = {
        **os.environ,
        "PATH": f"{binaries}:{os.environ['PATH']}",
        "QUEUE_RECORD": str(record),
        "TRAINING_QUEUE_IN_FLIGHT": "1",
        "TRAINING_QUEUE_MONITOR_LIGHT_GPUS": "1",
        "TRAINING_QUEUE_GPU_CHECK_SECONDS": "0",
        "TRAINING_QUEUE_GPU_RETRY_SECONDS": "2",
        "TRAINING_QUEUE_GPU_RECHECK_SECONDS": "600",
        "TRAINING_QUEUE_SCRIPT": "custom_experiment.py",
    }

    result = subprocess.run(
        [
            "bash",
            "-c",
            f'set -u; source "{QUEUE}" && enqueue only VARIANT=only && drain',
        ],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
        timeout=6,
    )

    assert record.read_text().strip() == "GPU-c"
    assert "kept GPU 2 excluded" in result.stdout
    assert result.stdout.count("monitoring lightly used GPU 2") == 2
    probe_times = [int(value) for value in checks.read_text().splitlines()]
    assert probe_times[1] - probe_times[0] >= 1_000_000_000


def test_periodic_check_only_pauses_the_gpu_being_checked(tmp_path: Path) -> None:
    binaries = tmp_path / "bin"
    binaries.mkdir()
    _executable(
        binaries / "nvidia-smi",
        """#!/usr/bin/env bash
if [[ "$1" == --query-gpu=* ]]; then
    printf '0, GPU-a\n1, GPU-b\n'
elif [[ "$1" == --id=* ]]; then
    printf '%s:%s\n' "${1#--id=}" "$(date +%s%N)" >> "$GPU_CHECK_RECORD"
    printf '0, 0, 1000\n'
fi
""",
    )
    _executable(
        binaries / "python",
        """#!/usr/bin/env bash
touch "$DCN_PREPARED_MARKER"
while [ ! -e "$DCN_TRAINING_RELEASE" ]; do sleep 0.01; done
printf 'start:%s:%s:%s\n' "$VARIANT" "$CUDA_VISIBLE_DEVICES" "$(date +%s%N)" >> "$QUEUE_RECORD"
if [ "$VARIANT" = first ]; then sleep 0.08; else sleep 0.35; fi
printf 'end:%s:%s:%s\n' "$VARIANT" "$CUDA_VISIBLE_DEVICES" "$(date +%s%N)" >> "$QUEUE_RECORD"
""",
    )
    record = tmp_path / "runs"
    checks = tmp_path / "checks"
    environment = {
        **os.environ,
        "PATH": f"{binaries}:{os.environ['PATH']}",
        "QUEUE_RECORD": str(record),
        "GPU_CHECK_RECORD": str(checks),
        "TRAINING_QUEUE_IN_FLIGHT": "1",
        "TRAINING_QUEUE_MONITOR_LIGHT_GPUS": "1",
        "TRAINING_QUEUE_GPU_CHECK_SECONDS": "0.05",
        "TRAINING_QUEUE_GPU_SAMPLE_SECONDS": "0.01",
        "TRAINING_QUEUE_GPU_RECHECK_SECONDS": "0",
        "TRAINING_QUEUE_GPU_SETTLE_SECONDS": "0",
        "TRAINING_QUEUE_SCRIPT": "custom_experiment.py",
    }

    result = subprocess.run(
        [
            "bash",
            "-c",
            f'set -u; source "{QUEUE}" && '
            "enqueue first VARIANT=first && "
            "_release_training && "
            "while ! grep -q '^end:first:' \"$QUEUE_RECORD\"; do sleep 0.01; done && "
            "enqueue second VARIANT=second && drain",
        ],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )

    events = [line.split(":") for line in record.read_text().splitlines()]
    second_start = int(next(row[3] for row in events if row[:2] == ["start", "second"]))
    second_end = int(next(row[3] for row in events if row[:2] == ["end", "second"]))
    gpu_a_checks = [
        int(timestamp)
        for gpu, timestamp in (
            line.split(":") for line in checks.read_text().splitlines()
        )
        if gpu == "GPU-a"
    ]
    assert any(second_start < timestamp < second_end for timestamp in gpu_a_checks)
    assert "rechecking GPU 0" in result.stdout


def test_gpu_check_does_not_block_lookahead_on_another_gpu(
    tmp_path: Path,
) -> None:
    binaries = tmp_path / "bin"
    binaries.mkdir()
    _executable(
        binaries / "nvidia-smi",
        """#!/usr/bin/env bash
if [[ "$1" == --query-gpu=* ]]; then printf '0, GPU-a\n1, GPU-b\n'; fi
""",
    )
    _executable(
        binaries / "python3",
        """#!/usr/bin/env bash
sleep 0.8
""",
    )
    _executable(
        binaries / "python",
        """#!/usr/bin/env bash
printf 'prepare:%s:%s\n' "$VARIANT" "$(date +%s%N)" >> "$QUEUE_RECORD"
sleep 0.05
touch "$DCN_PREPARED_MARKER"
while [ ! -e "$DCN_TRAINING_RELEASE" ]; do sleep 0.01; done
printf 'start:%s:%s\n' "$VARIANT" "$(date +%s%N)" >> "$QUEUE_RECORD"
if [ "$VARIANT" = first ]; then sleep 0.5; else sleep 0.1; fi
printf 'end:%s:%s\n' "$VARIANT" "$(date +%s%N)" >> "$QUEUE_RECORD"
""",
    )
    record = tmp_path / "runs"
    environment = {
        **os.environ,
        "PATH": f"{binaries}:{os.environ['PATH']}",
        "QUEUE_RECORD": str(record),
        "TRAINING_QUEUE_IN_FLIGHT": "2",
        "TRAINING_QUEUE_CONTROL_PYTHON": sys.executable,
        "TRAINING_QUEUE_GPU_SETTLE_SECONDS": "0",
        "TRAINING_QUEUE_SCRIPT": "custom_experiment.py",
    }

    subprocess.run(
        [
            "bash",
            "-c",
            f'set -u; source "{QUEUE}" && '
            "enqueue first VARIANT=first && "
            "_release_training && "
            'while ! grep -q \'^start:first:\' "$QUEUE_RECORD"; do sleep 0.01; done && '
            "_start_gpu_check GPU-b rechecking && "
            "enqueue second VARIANT=second && drain",
        ],
        cwd=tmp_path,
        env=environment,
        check=True,
        timeout=4,
    )

    events = {
        (event, variant): int(timestamp)
        for event, variant, timestamp in (
            line.split(":") for line in record.read_text().splitlines()
        )
    }
    assert events[("prepare", "second")] < events[("end", "first")]


def test_periodic_check_runs_before_an_already_prepared_successor(
    tmp_path: Path,
) -> None:
    binaries = tmp_path / "bin"
    binaries.mkdir()
    _executable(
        binaries / "nvidia-smi",
        """#!/usr/bin/env bash
if [[ "$1" == --query-gpu=* ]]; then
    printf '0, GPU-a\n'
elif [[ "$1" == --id=* ]]; then
    printf '%s\n' "$(date +%s%N)" >> "$GPU_CHECK_RECORD"
    printf '0, 0, 1000\n'
fi
""",
    )
    _executable(
        binaries / "python",
        """#!/usr/bin/env bash
touch "$DCN_PREPARED_MARKER"
while [ ! -e "$DCN_TRAINING_RELEASE" ]; do sleep 0.01; done
if [ "$VARIANT" = second ]; then
    while [ ! -e "$CHECK_REQUESTED" ]; do sleep 0.01; done
fi
exec 8>"generated/gpu-$CUDA_VISIBLE_DEVICES-slot-$DCN_GPU_LOCK_SLOT.lock"
flock -x 8
exec 9>"generated/gpu-$CUDA_VISIBLE_DEVICES.lock"
flock -s 9
printf 'start:%s:%s\n' "$VARIANT" "$(date +%s%N)" >> "$QUEUE_RECORD"
if [ "$VARIANT" = first ]; then sleep 1.2; else sleep 0.15; fi
printf 'end:%s:%s\n' "$VARIANT" "$(date +%s%N)" >> "$QUEUE_RECORD"
""",
    )
    record = tmp_path / "runs"
    checks = tmp_path / "checks"
    environment = {
        **os.environ,
        "PATH": f"{binaries}:{os.environ['PATH']}",
        "QUEUE_RECORD": str(record),
        "GPU_CHECK_RECORD": str(checks),
        "CHECK_REQUESTED": str(tmp_path / "check_requested"),
        "TRAINING_QUEUE_IN_FLIGHT": "2",
        "TRAINING_QUEUE_MONITOR_LIGHT_GPUS": "1",
        "TRAINING_QUEUE_GPU_CHECK_SECONDS": "0.05",
        "TRAINING_QUEUE_GPU_SAMPLE_SECONDS": "0.01",
        "TRAINING_QUEUE_GPU_RECHECK_SECONDS": "1",
        "TRAINING_QUEUE_GPU_SETTLE_SECONDS": "0",
        "TRAINING_QUEUE_SCRIPT": "custom_experiment.py",
    }

    subprocess.run(
        [
            "bash",
            "-c",
            f'set -u; source "{QUEUE}" && '
            "enqueue first VARIANT=first && "
            "enqueue second VARIANT=second && "
            "_release_training && "
            "while ! grep -q '^start:first:' \"$QUEUE_RECORD\"; do sleep 0.01; done && "
            "_gpu_next_check[GPU-a]=0 && "
            "_maybe_recheck_gpus && "
            "sleep 0.05 && touch \"$CHECK_REQUESTED\" && drain",
        ],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )

    events = {
        (event, variant): int(timestamp)
        for event, variant, timestamp in (
            line.split(":") for line in record.read_text().splitlines()
        )
    }
    check_times = [int(line) for line in checks.read_text().splitlines()]
    assert any(
        events[("end", "first")] < timestamp < events[("start", "second")]
        for timestamp in check_times
    )


def test_failed_periodic_check_keeps_gpu_excluded(tmp_path: Path) -> None:
    binaries = tmp_path / "bin"
    binaries.mkdir()
    sample_count = tmp_path / "gpu_a_samples"
    _executable(
        binaries / "nvidia-smi",
        f"""#!/usr/bin/env bash
if [[ "$1" == --query-gpu=* ]]; then
    printf '0, GPU-a\n1, GPU-b\n'
elif [[ "$1" == --id=GPU-a ]]; then
    count=$(cat "{sample_count}" 2>/dev/null || printf 0)
    count=$((count + 1))
    printf '%s' "$count" > "{sample_count}"
    printf '%s, 100, 1000\n' "$((9 + count))"
elif [[ "$1" == --id=GPU-b ]]; then
    printf '0, 0, 1000\n'
fi
""",
    )
    _executable(
        binaries / "python",
        """#!/usr/bin/env bash
touch "$DCN_PREPARED_MARKER"
while [ ! -e "$DCN_TRAINING_RELEASE" ]; do sleep 0.01; done
printf 'start:%s:%s\n' "$VARIANT" "$CUDA_VISIBLE_DEVICES" >> "$QUEUE_RECORD"
if [ "$VARIANT" = first ]; then sleep 1.1; else sleep 0.25; fi
printf 'end:%s:%s\n' "$VARIANT" "$CUDA_VISIBLE_DEVICES" >> "$QUEUE_RECORD"
""",
    )
    record = tmp_path / "runs"
    environment = {
        **os.environ,
        "PATH": f"{binaries}:{os.environ['PATH']}",
        "QUEUE_RECORD": str(record),
        "TRAINING_QUEUE_IN_FLIGHT": "1",
        "TRAINING_QUEUE_MONITOR_LIGHT_GPUS": "1",
        "TRAINING_QUEUE_GPU_CHECK_SECONDS": "0.05",
        "TRAINING_QUEUE_GPU_SAMPLE_SECONDS": "0.01",
        "TRAINING_QUEUE_GPU_RECHECK_SECONDS": "1",
        "TRAINING_QUEUE_GPU_SETTLE_SECONDS": "0",
        "TRAINING_QUEUE_SCRIPT": "custom_experiment.py",
    }

    result = subprocess.run(
        [
            "bash",
            "-c",
            f'set -u; source "{QUEUE}" && '
            "enqueue first VARIANT=first && "
            "_release_training && "
            "while ! grep -q '^end:first:' \"$QUEUE_RECORD\"; do sleep 0.01; done && "
            "enqueue second VARIANT=second && "
            "enqueue third VARIANT=third && drain",
        ],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )

    assignments = {
        variant: gpu
        for event, variant, gpu in (
            line.split(":") for line in record.read_text().splitlines()
        )
        if event == "start"
    }
    assert assignments == {"first": "GPU-a", "second": "GPU-b", "third": "GPU-b"}
    assert "rechecking GPU 0" in result.stdout


def test_queue_fails_before_launching_when_every_gpu_is_busy(tmp_path: Path) -> None:
    binaries = tmp_path / "bin"
    binaries.mkdir()
    _executable(
        binaries / "nvidia-smi",
        """#!/usr/bin/env bash
if [[ "$1" == --query-gpu=* ]]; then
    printf '0, GPU-a\n1, GPU-b\n'
else
    printf 'GPU-a\nGPU-b\n'
fi
""",
    )
    environment = {
        **os.environ,
        "PATH": f"{binaries}:{os.environ['PATH']}",
        "TRAINING_QUEUE_SCRIPT": "custom_experiment.py",
    }

    result = subprocess.run(
        ["bash", "-c", f'source "{QUEUE}"'],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "No idle GPUs" in result.stderr


def test_queue_depth_covers_the_slowest_observed_preprocessing(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.log"
    first.write_text(
        "Prepared stage 'one' in 3.0s\n"
        "Prepared stage 'two' in 2.0s\n"
        "Trained stage 'one' in 2.0s\n"
        "Trained stage 'two' in 3.0s\n"
    )
    second = tmp_path / "second.log"
    second.write_text(
        "Prepared stage 'model' in 7.5s\nTrained stage 'model' in 2.5s\n"
    )

    result = subprocess.run(
        ["python3", str(QUEUE_DEPTH), str(first), str(second)],
        text=True,
        capture_output=True,
        check=True,
    )

    assert result.stdout.strip() == "4"


def test_queue_depth_bootstraps_with_one_run_ahead(tmp_path: Path) -> None:
    result = subprocess.run(
        ["python3", str(QUEUE_DEPTH), str(tmp_path / "missing.log")],
        text=True,
        capture_output=True,
        check=True,
    )

    assert result.stdout.strip() == "2"


def test_queue_depth_is_capped(tmp_path: Path) -> None:
    log = tmp_path / "outlier.log"
    log.write_text(
        "Prepared stage 'model' in 1000.0s\nTrained stage 'model' in 1.0s\n"
    )

    result = subprocess.run(
        ["python3", str(QUEUE_DEPTH), str(log)],
        text=True,
        capture_output=True,
        check=True,
    )

    assert result.stdout.strip() == "4"


def test_queue_depth_is_calculated_per_gpu(
    tmp_path: Path,
) -> None:
    log = tmp_path / "timing.log"
    log.write_text(
        "Prepared stage 'model' in 3.0s\nTrained stage 'model' in 10.0s\n"
    )

    result = subprocess.run(
        ["python3", str(QUEUE_DEPTH), "--gpu-count", "4", str(log)],
        text=True,
        capture_output=True,
        check=True,
    )

    assert result.stdout.strip() == "2"
    assert "preprocessing is slower" not in result.stderr


def test_queue_depth_uses_only_compatible_historical_runs(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    state = tmp_path / "service"
    completed = state / "completed"
    completed.mkdir(parents=True)
    compatible = logs / "compatible"
    incompatible_script = logs / "incompatible-script"
    incompatible_data = logs / "incompatible-data"
    for directory in (compatible, incompatible_script, incompatible_data):
        directory.mkdir(parents=True)
    compatible.joinpath("sweep.log").write_text(
        "Prepared stage 'model' in 1.0s\nTrained stage 'model' in 10.0s\n"
    )
    incompatible_script.joinpath("sweep.log").write_text(
        "Prepared stage 'model' in 30.0s\nTrained stage 'model' in 10.0s\n"
    )
    incompatible_data.joinpath("sweep.log").write_text(
        "Prepared stage 'model' in 20.0s\nTrained stage 'model' in 10.0s\n"
    )
    for index, (run, script, data_group) in enumerate(
        (
            ("compatible", "experiment.py", "dataset-a"),
            ("incompatible-script", "other.py", "dataset-a"),
            ("incompatible-data", "experiment.py", "dataset-b"),
        )
    ):
        completed.joinpath(f"{index}.json").write_text(
            json.dumps({"data_group": data_group, "run": run, "script": script})
        )

    result = subprocess.run(
        [
            "python3",
            str(QUEUE_DEPTH),
            "--history-root",
            str(logs),
            "--service-state",
            str(state),
            "--script",
            "experiment.py",
            "--data-group",
            "dataset-a",
        ],
        text=True,
        capture_output=True,
        check=True,
    )

    assert result.stdout.strip() == "2"


def test_preprocessing_failure_aborts_the_initial_buffer(tmp_path: Path) -> None:
    binaries = tmp_path / "bin"
    binaries.mkdir()
    _executable(
        binaries / "nvidia-smi",
        """#!/usr/bin/env bash
if [[ "$1" == --query-gpu=* ]]; then printf '0, GPU-a\n'; fi
""",
    )
    _executable(binaries / "python", "#!/usr/bin/env bash\nexit 1\n")
    environment = {
        **os.environ,
        "PATH": f"{binaries}:{os.environ['PATH']}",
        "TRAINING_QUEUE_IN_FLIGHT": "1",
        "TRAINING_QUEUE_SCRIPT": "custom_experiment.py",
    }

    result = subprocess.run(
        [
            "bash",
            "-c",
            f'source "{QUEUE}" && enqueue broken VARIANT=dim_16',
        ],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "failed before the preprocessing buffer was ready" in result.stderr
