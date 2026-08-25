import pytest

from utils.training_queue.gpu_check import (
    ComputeProcess,
    GpuSample,
    foreign_memory_percent,
    is_lightly_used,
    wait_for_foreign_memory,
)


def _sample(utilization: int, memory_percent: float) -> GpuSample:
    return GpuSample(
        utilization=utilization,
        used_memory=int(memory_percent * 10),
        total_memory=1000,
    )


@pytest.mark.parametrize(
    ("samples", "expected"),
    [
        ([_sample(19, 19), _sample(19, 19)], True),
        ([_sample(1, 10), _sample(5, 15), _sample(3, 12)], True),
        ([_sample(1, 10), _sample(6, 10)], False),
        ([_sample(1, 10), _sample(1, 15.1)], False),
        ([_sample(20, 10), _sample(20, 10)], False),
        ([_sample(10, 20), _sample(10, 20)], False),
    ],
)
def test_light_gpu_policy(samples: list[GpuSample], expected: bool) -> None:
    assert is_lightly_used(samples) is expected


def test_foreign_memory_ignores_only_the_selected_process() -> None:
    processes = [ComputeProcess(1, 30), ComputeProcess(2, 50)]

    assert foreign_memory_percent(processes, excluded_pids={1}, total_memory=100) == 50


def test_foreign_memory_wait_discards_prepared_resources_once(monkeypatch) -> None:
    process_samples = iter(
        [
            [ComputeProcess(2, 30)],
            [ComputeProcess(2, 25)],
            [],
        ]
    )
    discarded: list[bool] = []
    monkeypatch.setattr(
        "utils.training_queue.gpu_check._read_sample",
        lambda gpu: GpuSample(0, 0, 100),
    )
    monkeypatch.setattr(
        "utils.training_queue.gpu_check._read_compute_processes",
        lambda gpu: next(process_samples),
    )
    monkeypatch.setattr("utils.training_queue.gpu_check.sleep", lambda seconds: None)

    wait_for_foreign_memory(
        "GPU-a",
        interval=0,
        on_wait=lambda: discarded.append(True),
    )

    assert discarded == [True]
