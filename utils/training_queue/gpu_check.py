import argparse
import json
import logging
import os
import subprocess
from dataclasses import asdict, dataclass
from time import monotonic, sleep
from typing import Callable, Sequence


@dataclass(frozen=True)
class GpuSample:
    utilization: int
    used_memory: int
    total_memory: int

    @property
    def memory_percent(self) -> float:
        return 100 * self.used_memory / self.total_memory


@dataclass(frozen=True)
class ComputeProcess:
    pid: int
    used_memory: int


def foreign_memory_percent(
    processes: Sequence[ComputeProcess],
    *,
    excluded_pids: set[int],
    total_memory: int,
) -> float:
    used = sum(
        process.used_memory
        for process in processes
        if process.pid not in excluded_pids
    )
    return 100 * used / total_memory


def is_lightly_used(samples: Sequence[GpuSample]) -> bool:
    if not samples:
        return False
    first = samples[0]
    if first.utilization >= 20 or first.memory_percent >= 20:
        return False
    unchanged = all(
        (sample.utilization, sample.used_memory)
        == (first.utilization, first.used_memory)
        for sample in samples[1:]
    )
    always_very_light = all(
        sample.utilization <= 5 and sample.memory_percent <= 15
        for sample in samples
    )
    return unchanged or always_very_light


def _read_sample(gpu: str) -> GpuSample:
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={gpu}",
            "--query-gpu=utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    fields = [int(field.strip()) for field in result.stdout.strip().split(",")]
    if len(fields) != 3:
        raise ValueError(f"unexpected nvidia-smi output: {result.stdout!r}")
    return GpuSample(*fields)


def _read_compute_processes(gpu: str) -> list[ComputeProcess]:
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={gpu}",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return [
        ComputeProcess(*(int(field.strip()) for field in line.split(",")))
        for line in result.stdout.splitlines()
        if line.strip()
    ]


def wait_for_foreign_memory(
    gpu: str,
    *,
    max_percent: float = 20.0,
    interval: float = 5.0,
    on_wait: Callable[[], None] | None = None,
) -> bool:
    logger = logging.getLogger(__name__)
    released_preparation = False
    while True:
        sample = _read_sample(gpu)
        processes = _read_compute_processes(gpu)
        foreign = foreign_memory_percent(
            processes,
            excluded_pids={os.getpid()},
            total_memory=sample.total_memory,
        )
        if foreign < max_percent:
            return released_preparation
        if not released_preparation:
            if on_wait is not None:
                on_wait()
            released_preparation = True
        logger.warning(
            "GPU %s has %.1f%% foreign compute memory; waiting before training",
            gpu,
            foreign,
        )
        sleep(interval)


def monitor_gpu(gpu: str, duration: float, interval: float) -> list[GpuSample]:
    samples = [_read_sample(gpu)]
    if not is_lightly_used(samples):
        return samples
    deadline = monotonic() + duration
    while monotonic() < deadline:
        sleep(min(interval, max(0, deadline - monotonic())))
        samples.append(_read_sample(gpu))
    return samples


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--duration", type=float, default=30)
    parser.add_argument("--interval", type=float, default=1)
    arguments = parser.parse_args()
    if arguments.duration < 0 or arguments.interval <= 0:
        parser.error("duration must be non-negative and interval must be positive")
    try:
        samples = monitor_gpu(arguments.gpu, arguments.duration, arguments.interval)
    except Exception as error:
        print(
            json.dumps(
                {
                    "admitted": False,
                    "error": f"{type(error).__name__}: {error}",
                    "gpu": arguments.gpu,
                    "samples": [],
                },
                sort_keys=True,
            )
        )
        return 2
    admitted = is_lightly_used(samples)
    print(
        json.dumps(
            {
                "admitted": admitted,
                "gpu": arguments.gpu,
                "samples": [asdict(sample) for sample in samples],
            },
            sort_keys=True,
        )
    )
    return 0 if admitted else 1


if __name__ == "__main__":
    raise SystemExit(main())
