import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
QUEUE = ROOT / "utils/training_queue/queue.sh"
METRIC_SCRIPT = (
    ROOT / "experiments/g1_sasrec_item_ids_likes/checks/metric_regression_50m.py"
)
METRIC_FIXTURE = ROOT / "dcn/tests/fixtures/g1_baseline_50m_seed0.json"
UTILIZATION_SCRIPT = (
    ROOT / "experiments/g1_sasrec_item_ids_likes/checks/utilization_regression_50m.py"
)
RUN_SLOW_GPU = os.environ.get("RUN_SLOW_GPU_TESTS") == "1"


pytestmark = [
    pytest.mark.slow_gpu,
    pytest.mark.skipif(
        not RUN_SLOW_GPU,
        reason="set RUN_SLOW_GPU_TESTS=1 to run real-GPU integration tests",
    ),
]


def _dedicated_gpu() -> str:
    gpu = os.environ.get("SLOW_GPU_INDEX")
    if gpu is None:
        pytest.skip("set SLOW_GPU_INDEX to a dedicated physical GPU index")
    processes = subprocess.run(
        [
            "nvidia-smi",
            f"--id={gpu}",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if processes:
        pytest.skip(f"GPU {gpu} has an active compute process")
    return gpu


def _dedicated_gpus() -> list[str]:
    requested = os.environ.get("SLOW_GPU_INDICES")
    if requested is None:
        return [_dedicated_gpu()]
    gpus = requested.split()
    if not gpus:
        pytest.skip("SLOW_GPU_INDICES is empty")
    for gpu in gpus:
        processes = subprocess.run(
            [
                "nvidia-smi",
                f"--id={gpu}",
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if processes:
            pytest.skip(f"GPU {gpu} has an active compute process")
    return gpus


def _write_executable(path: Path, source: str) -> None:
    path.write_text(source)
    path.chmod(0o755)


def _events(path: Path) -> list[tuple[str, str, int]]:
    return [
        (event, run, int(timestamp))
        for event, run, timestamp in (
            line.split(":") for line in path.read_text().splitlines()
        )
    ]


def test_queue_has_no_idle_handoff_samples_on_real_gpu(tmp_path: Path) -> None:
    gpu = _dedicated_gpu()
    real_nvidia_smi = shutil.which("nvidia-smi")
    assert real_nvidia_smi is not None
    binaries = tmp_path / "bin"
    binaries.mkdir()
    event_path = tmp_path / "events"
    check_path = tmp_path / "checks"
    worker = tmp_path / "worker.py"
    worker.write_text(
        """import fcntl
import os
import time
from pathlib import Path

import torch


def record(event):
    with Path(os.environ["SYNTHETIC_EVENTS"]).open("a") as output:
        output.write(f"{event}:{os.environ['VARIANT']}:{time.monotonic_ns()}\\n")


record("prepare_start")
time.sleep(0.8)
torch.cuda.init()
left = torch.randn(512, 512, dtype=torch.bfloat16)
right = torch.randn(512, 512, dtype=torch.bfloat16)
record("prepare_end")
Path(os.environ["DCN_PREPARED_MARKER"]).touch()
while not Path(os.environ["DCN_TRAINING_RELEASE"]).exists():
    time.sleep(0.01)

base = Path("generated")
slot = base / f"gpu-{os.environ['CUDA_VISIBLE_DEVICES']}-slot-{os.environ['DCN_GPU_LOCK_SLOT']}.lock"
gate = base / f"gpu-{os.environ['CUDA_VISIBLE_DEVICES']}.lock"
with slot.open("w") as slot_file, gate.open("w") as gate_file:
    fcntl.flock(slot_file, fcntl.LOCK_EX)
    fcntl.flock(gate_file, fcntl.LOCK_SH)
    record("critical_start")
    left = left.cuda()
    right = right.cuda()
    torch.mm(left, right)
    torch.cuda.synchronize()
    record("training_start")
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        left = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
        right = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
        torch.mm(left, right)
    torch.cuda.synchronize()
    record("training_end")
    record("critical_end")
"""
    )
    _write_executable(
        binaries / "python",
        """#!/usr/bin/env bash
exec "$REAL_PYTHON" "$SYNTHETIC_WORKER"
""",
    )
    _write_executable(
        binaries / "nvidia-smi",
        """#!/usr/bin/env bash
if [[ "$1" == --id=* ]]; then
    printf '%s\n' "$(date +%s%N)" >> "$SYNTHETIC_CHECKS"
fi
exec "$REAL_NVIDIA_SMI" "$@"
""",
    )
    environment = {
        **os.environ,
        "PATH": f"{binaries}:{os.environ['PATH']}",
        "REAL_PYTHON": sys.executable,
        "REAL_NVIDIA_SMI": real_nvidia_smi,
        "SYNTHETIC_WORKER": str(worker),
        "SYNTHETIC_EVENTS": str(event_path),
        "SYNTHETIC_CHECKS": str(check_path),
        "TRAINING_QUEUE_GPUS": gpu,
        "TRAINING_QUEUE_IN_FLIGHT": "3",
        "TRAINING_QUEUE_MONITOR_LIGHT_GPUS": "1",
        "TRAINING_QUEUE_GPU_CHECK_SECONDS": "0.5",
        "TRAINING_QUEUE_GPU_SAMPLE_SECONDS": "0.1",
        "TRAINING_QUEUE_GPU_RECHECK_SECONDS": "5",
        "TRAINING_QUEUE_GPU_RETRY_SECONDS": "1",
        "TRAINING_QUEUE_GPU_SETTLE_SECONDS": "0",
        "TRAINING_QUEUE_SCRIPT": "synthetic.py",
        "WANDB_MODE": "disabled",
    }
    samples: list[tuple[int, int]] = []
    stop_sampling = threading.Event()

    def sample_gpu() -> None:
        while not stop_sampling.is_set():
            output = subprocess.run(
                [
                    real_nvidia_smi,
                    f"--id={gpu}",
                    "--query-gpu=utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            samples.append((time.monotonic_ns(), int(output.strip())))
            stop_sampling.wait(0.5)

    sampler = threading.Thread(target=sample_gpu)
    sampler.start()
    try:
        subprocess.run(
            [
                "bash",
                "-c",
                f'source "{QUEUE}" && '
                "enqueue first VARIANT=first && "
                "enqueue second VARIANT=second && "
                "enqueue third VARIANT=third && "
                "enqueue fourth VARIANT=fourth && drain",
            ],
            cwd=tmp_path,
            env=environment,
            check=True,
            timeout=45,
        )
    finally:
        stop_sampling.set()
        sampler.join(timeout=2)

    events = _events(event_path)
    intervals = {
        run: {
            event: timestamp
            for event, event_run, timestamp in events
            if event_run == run
        }
        for run in ("first", "second", "third", "fourth")
    }
    training = sorted(
        (intervals[run]["training_start"], intervals[run]["training_end"])
        for run in ("first", "second", "third", "fourth")
    )
    critical = sorted(
        (intervals[run]["critical_start"], intervals[run]["critical_end"])
        for run in ("first", "second", "third", "fourth")
    )
    assert all(left[1] <= right[0] for left, right in zip(critical, critical[1:]))
    for run in ("second", "third", "fourth"):
        preparation = (
            intervals[run]["prepare_start"],
            intervals[run]["prepare_end"],
        )
        assert any(
            preparation[0] < training_end and preparation[1] > training_start
            for training_start, training_end in training
        )

    check_samples = [int(value) for value in check_path.read_text().splitlines()]
    assert check_samples
    wall_to_monotonic = time.time_ns() - time.monotonic_ns()
    monotonic_checks = [value - wall_to_monotonic for value in check_samples]
    check_intervals = [
        [left[1], right[0]]
        for left, right in zip(training, training[1:])
        if any(left[1] <= sample <= right[0] for sample in monotonic_checks)
    ]
    active_samples = [
        (timestamp, utilization)
        for timestamp, utilization in samples
        if training[0][0] <= timestamp <= training[-1][1]
        and not any(
            start <= timestamp <= end for start, end in check_intervals
        )
    ]
    assert active_samples
    zero_offsets = [
        (timestamp - training[0][0]) / 1e9
        for timestamp, utilization in active_samples
        if utilization == 0
    ]
    assert not zero_offsets, (
        f"0% samples {zero_offsets}; training={training}; checks={check_intervals}"
    )


def test_g1_fixed_seed_50m_metric_regression() -> None:
    gpu = _dedicated_gpu()
    expected = json.loads(METRIC_FIXTURE.read_text())
    batch_size = int(os.environ.get("G1_REGRESSION_BATCH_SIZE", "128"))
    run_name = f"g1_metric_regression_50m_b{batch_size}_s0"
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": gpu,
        "G1_REGRESSION_BATCH_SIZE": str(batch_size),
        "G1_METRIC_REGRESSION_RUN_NAME": run_name,
        "WANDB_MODE": "disabled",
    }
    subprocess.run(
        [sys.executable, "-m", "dcn.main", "-s", str(METRIC_SCRIPT)],
        cwd=ROOT,
        env=environment,
        check=True,
        timeout=300,
    )
    actual = json.loads(
        (ROOT / "generated/logs" / run_name / "final_metrics.json").read_text()
    )
    for metric, reference in expected["metrics"].items():
        tolerance = expected["absolute_tolerance"][metric]
        assert actual[metric] == pytest.approx(reference, abs=tolerance)


def test_selected_g1_training_sustains_high_gpu_utilization(tmp_path: Path) -> None:
    gpus = _dedicated_gpus()
    if len(gpus) != 4:
        pytest.skip("the utilization regression requires four dedicated GPUs")
    inventory = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    gpu_info = {
        index.strip(): (uuid.strip(), name.strip())
        for index, uuid, name in (line.split(",", 2) for line in inventory)
    }
    for gpu in gpus:
        if "A100" not in gpu_info[gpu][1]:
            pytest.skip(
                f"utilization threshold is calibrated for A100, found {gpu_info[gpu][1]}"
            )
    markers = [
        tmp_path / f"training-window-{index}" for index in range(3 * len(gpus))
    ]
    binaries = tmp_path / "bin"
    binaries.mkdir()
    _write_executable(
        binaries / "python",
        """#!/usr/bin/env bash
exec "$REAL_PYTHON" "$@"
""",
    )
    environment = {
        **os.environ,
        "PATH": f"{binaries}:{os.environ['PATH']}",
        "REAL_PYTHON": sys.executable,
        "TRAINING_QUEUE_GPUS": " ".join(gpus),
        "TRAINING_QUEUE_IN_FLIGHT": "3",
        "TRAINING_QUEUE_DATA_GROUP": "g1-utilization-regression",
        "TRAINING_QUEUE_SCRIPT": str(UTILIZATION_SCRIPT),
        "G1_UTIL_BATCH_SIZE": "1280",
        "WANDB_MODE": "disabled",
    }
    samples: dict[str, list[tuple[int, int]]] = {gpu: [] for gpu in gpus}
    stop_sampling = threading.Event()

    def sample_gpu() -> None:
        while not stop_sampling.is_set():
            output = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.splitlines()
            timestamp = time.monotonic_ns()
            for index, utilization in (line.split(",", 1) for line in output):
                index = index.strip()
                if index in samples:
                    samples[index].append((timestamp, int(utilization.strip())))
            stop_sampling.wait(0.2)

    sampler = threading.Thread(target=sample_gpu)
    sampler.start()
    try:
        enqueues = " && ".join(
            f"enqueue g1_utilization_regression_50m_{index} "
            f"G1_UTIL_MARKER={marker} "
            f"G1_UTIL_RUN_NAME=g1_utilization_regression_50m_{index}"
            for index, marker in enumerate(markers)
        )
        subprocess.run(
            ["bash", "-c", f'source "{QUEUE}" && {enqueues} && drain'],
            cwd=ROOT,
            env=environment,
            check=True,
            timeout=300,
        )
    finally:
        stop_sampling.set()
        sampler.join(timeout=2)

    windows: dict[str, list[tuple[int, int]]] = {}
    for marker in markers:
        entries = [line.split() for line in marker.read_text().splitlines()]
        device = entries[0][2]
        bounds = {name: int(timestamp) for name, timestamp, _ in entries}
        windows.setdefault(device, []).append((bounds["start"], bounds["end"]))
    first_windows = [min(windows[gpu_info[gpu][0]]) for gpu in gpus]
    common_training_seconds = (
        min(end for _, end in first_windows)
        - max(start for start, _ in first_windows)
    ) / 1e9
    assert common_training_seconds >= 5.0
    mean_utilizations = {}
    elapsed_seconds = {}
    handoffs_by_gpu = {}
    for gpu in gpus:
        device_windows = sorted(windows[gpu_info[gpu][0]])
        assert len(device_windows) == 3
        start, end = device_windows[0][1], device_windows[-1][1]
        steady = [
            utilization
            for timestamp, utilization in samples[gpu]
            if start <= timestamp <= end
        ]
        assert len(steady) >= 10
        mean_utilizations[gpu] = sum(steady) / len(steady)
        elapsed_seconds[gpu] = (end - start) / 1e9
        handoffs_by_gpu[gpu] = [
            (next_start - previous_end) / 1e9
            for (_, previous_end), (next_start, _) in zip(
                device_windows, device_windows[1:]
            )
        ]
    print(
        {
            "mean_utilization": mean_utilizations,
            "elapsed_seconds": elapsed_seconds,
            "handoffs": handoffs_by_gpu,
        }
    )
    assert max(elapsed_seconds.values()) <= 140.0, elapsed_seconds
    assert max(map(max, handoffs_by_gpu.values())) <= 3.0, handoffs_by_gpu
    assert round(min(mean_utilizations.values())) >= 90, mean_utilizations
