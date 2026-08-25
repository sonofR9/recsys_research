import fcntl
import logging
import threading
import time
from pathlib import Path
from typing import Callable

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from dcn.config.experiment import TrainedStage
from dcn.main import run_experiment
from neuralrec.run.train import TrainRunner


class _SquaredLoss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))

    def forward(self, batch: object) -> dict[str, torch.Tensor]:
        return {"loss": (self.weight**2).sum()}


class _CountingRunner(TrainRunner):
    def __init__(
        self,
        model: torch.nn.Module,
        log: list[str],
        name: str,
        note: Callable[[], str] = lambda: "",
    ) -> None:
        super().__init__(
            model=model,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
            state={},
        )
        self.log = log
        self.name = name
        self.note = note
        self.loader = DataLoader(TensorDataset(torch.zeros(4, 1)), batch_size=2)

    def train(self) -> None:
        self.log.append(f"train:{self.name}{self.note()}")
        self.train_epoch(0, self.loader)


class _RecordedStage(TrainedStage):
    def __init__(self, name: str, log: list[str], locks: Path | None = None) -> None:
        self._name = name
        self._locks = locks
        self.log = log
        self.model = _SquaredLoss()

    @property
    def name(self) -> str:
        return self._name

    @property
    def gpu_lock_path(self) -> Path | None:
        return None if self._locks is None else self._locks / "gpu.lock"

    @property
    def prep_lock_path(self) -> Path | None:
        return None if self._locks is None else self._locks / "prep.lock"

    def create_runner(self) -> TrainRunner:
        self.log.append(f"create:{self._name}{self._held()}")
        return _CountingRunner(self.model, self.log, self._name, self._held)

    def finish(self, runner: TrainRunner) -> None:
        self.log.append(f"finish:{self._name}{self._held()}")

    def _held(self) -> str:
        if self._locks is None:
            return ""
        gpu = _is_locked(self.gpu_lock_path)
        return f":gpu={gpu},prep={_is_locked(self.prep_lock_path)}"


class _ConcurrentRunner(_CountingRunner):
    def __init__(
        self,
        model: torch.nn.Module,
        active: list[int],
        maximum: list[int],
        guard: threading.Lock,
    ) -> None:
        super().__init__(model, [], "model")
        self.active = active
        self.maximum = maximum
        self.guard = guard

    def train(self) -> None:
        with self.guard:
            self.active[0] += 1
            self.maximum[0] = max(self.maximum[0], self.active[0])
        try:
            time.sleep(0.1)
        finally:
            with self.guard:
                self.active[0] -= 1


class _ConcurrentStage(_RecordedStage):
    def __init__(
        self,
        locks: Path,
        active: list[int],
        maximum: list[int],
        guard: threading.Lock,
    ) -> None:
        super().__init__("model", [], locks)
        self.active = active
        self.maximum = maximum
        self.guard = guard

    def create_runner(self) -> TrainRunner:
        return _ConcurrentRunner(self.model, self.active, self.maximum, self.guard)


class _PreparingRunner(_CountingRunner):
    def __init__(
        self,
        model: torch.nn.Module,
        log: list[str],
        name: str,
        note: Callable[[], str],
        prepare_note: Callable[[], str],
    ) -> None:
        super().__init__(model, log, name, note)
        self.prepare_note = prepare_note

    def prepare(self) -> None:
        self.log.append(f"prepare:{self.name}{self.prepare_note()}")


class _PreparingStage(_RecordedStage):
    def create_runner(self) -> TrainRunner:
        self.log.append(f"create:{self._name}{self._held()}")
        return _PreparingRunner(
            self.model,
            self.log,
            self._name,
            self._held,
            lambda: f"{self._held()},prep_mode={_lock_mode(self.prep_lock_path)}",
        )


class _PrebuiltDataStage(_PreparingStage):
    @property
    def prebuilds_runner_data(self) -> bool:
        return True

    def prebuild_runner_data(self) -> None:
        self.log.append(
            f"prebuild:{self._name}{self._held()},"
            f"prep_mode={_lock_mode(self.prep_lock_path)},"
            f"data={_is_locked(self.runner_data_lock_path)}"
        )

    def prebuild_runner_components(self) -> None:
        self.log.append(f"components:{self._name}{self._held()}")

    def activate_runner_device(self, runner: TrainRunner) -> None:
        self.log.append(f"activate:{self._name}{self._held()}")

    def release_runner_device_cache(self) -> None:
        self.log.append(f"release:{self._name}{self._held()}")


class _ConcurrentDataGroupStage(_PreparingStage):
    def __init__(
        self,
        name: str,
        ready: Path,
        active: list[int],
        maximum: list[int],
        guard: threading.Lock,
    ) -> None:
        super().__init__(name, [], ready.parent)
        self.ready = ready
        self.active = active
        self.maximum = maximum
        self.guard = guard

    @property
    def prebuilds_runner_data(self) -> bool:
        return True

    @property
    def runner_data_ready_path(self) -> Path:
        return self.ready

    def prebuild_runner_data(self) -> None:
        with self.guard:
            self.active[0] += 1
            self.maximum[0] = max(self.maximum[0], self.active[0])
        try:
            time.sleep(0.1)
        finally:
            with self.guard:
                self.active[0] -= 1


def _is_locked(path: Path) -> bool:
    """flock is per open file description, so a second handle on the same file
    contends with the first even from the same process."""
    if not path.exists():
        return False
    with open(path) as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(handle, fcntl.LOCK_UN)
        return False


def _lock_mode(path: Path) -> str:
    with open(path) as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return "exclusive"
        fcntl.flock(handle, fcntl.LOCK_UN)
    return "shared" if _is_locked(path) else "unlocked"


class _TwoStageExperiment:
    """Stands in for an Experiment: run_experiment only needs setup + stages."""

    def __init__(self, log: list[str]) -> None:
        self.log = log
        self.stages = [_RecordedStage("quantizer", log), _RecordedStage("model", log)]

    def setup(self) -> None:
        self.log.append("setup")


def test_stages_run_in_order(tmp_path: Path) -> None:
    log: list[str] = []

    run_experiment(_TwoStageExperiment(log))

    assert log == [
        "setup",
        "create:quantizer",
        "train:quantizer",
        "finish:quantizer",
        "create:model",
        "train:model",
        "finish:model",
    ]


def test_building_the_runner_and_training_take_a_lock_each_and_never_both(
    tmp_path: Path,
) -> None:
    log: list[str] = []

    _RecordedStage("model", log, tmp_path).run()

    assert log == [
        "create:model:gpu=False,prep=True",
        "train:model:gpu=True,prep=False",
        "finish:model:gpu=True,prep=False",
    ]
    assert not _is_locked(tmp_path / "gpu.lock")
    assert not _is_locked(tmp_path / "prep.lock")


def test_runner_prepares_before_taking_the_gpu_lock(tmp_path: Path) -> None:
    log: list[str] = []

    _PreparingStage("model", log, tmp_path).run()

    assert log == [
        "create:model:gpu=False,prep=True",
        "prepare:model:gpu=False,prep=True,prep_mode=shared",
        "train:model:gpu=True,prep=False",
        "finish:model:gpu=True,prep=False",
    ]


def test_prebuilt_data_keeps_safe_runner_work_outside_the_exclusive_prep_lock(
    tmp_path: Path,
) -> None:
    log: list[str] = []

    _PrebuiltDataStage("model", log, tmp_path).run()

    assert log == [
        "prebuild:model:gpu=False,prep=True,prep_mode=shared,data=True",
        "create:model:gpu=False,prep=True",
        "prepare:model:gpu=False,prep=True,prep_mode=shared",
        "train:model:gpu=True,prep=False",
        "finish:model:gpu=True,prep=False",
    ]


def test_prebuilt_data_loads_concurrently_after_queue_data_is_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ready = tmp_path / "runner-data.ready"
    ready.touch()
    monkeypatch.setenv("DCN_RUNNER_DATA_READY", str(ready))
    log: list[str] = []

    _PrebuiltDataStage("model", log, tmp_path).run()

    assert log[0] == (
        "prebuild:model:gpu=False,prep=True,prep_mode=shared,data=False"
    )


@pytest.mark.parametrize(
    ("groups", "expected_concurrency"),
    [(("shared", "shared"), 1), (("first", "second"), 2)],
)
def test_runner_data_groups_overlap_but_same_group_serializes(
    tmp_path: Path,
    groups: tuple[str, str],
    expected_concurrency: int,
) -> None:
    active = [0]
    maximum = [0]
    guard = threading.Lock()
    stages = [
        _ConcurrentDataGroupStage(
            name,
            tmp_path / f"runner-data-{group}.ready",
            active,
            maximum,
            guard,
        )
        for name, group in zip(("one", "two"), groups, strict=True)
    ]
    threads = [threading.Thread(target=stage.run) for stage in stages]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)

    assert all(not thread.is_alive() for thread in threads)
    assert maximum == [expected_concurrency]
    assert all(stage.ready.exists() for stage in stages)


def test_gpu_lock_allows_only_one_training_at_a_time(tmp_path: Path) -> None:
    active = [0]
    maximum = [0]
    guard = threading.Lock()
    stages = [
        _ConcurrentStage(tmp_path, active, maximum, guard),
        _ConcurrentStage(tmp_path, active, maximum, guard),
    ]
    threads = [threading.Thread(target=stage.run) for stage in stages]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)

    assert all(not thread.is_alive() for thread in threads)
    assert maximum == [1]


def test_stage_reports_preparation_and_training_durations(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)

    _RecordedStage("model", []).run()

    messages = [record.getMessage() for record in caplog.records]
    assert (
        sum(message.startswith("Prepared stage 'model' in ") for message in messages)
        == 1
    )
    assert (
        sum(message.startswith("Trained stage 'model' in ") for message in messages)
        == 1
    )


def test_stage_waits_for_preprocessing_buffer_before_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = tmp_path / "prepared"
    release = tmp_path / "start"
    monkeypatch.setenv("DCN_PREPARED_MARKER", str(marker))
    monkeypatch.setenv("DCN_TRAINING_RELEASE", str(release))
    log: list[str] = []
    thread = threading.Thread(target=_RecordedStage("model", log).run)

    thread.start()
    deadline = time.monotonic() + 2
    while not marker.exists() and time.monotonic() < deadline:
        time.sleep(0.01)

    assert marker.exists()
    assert log == ["create:model"]
    release.touch()
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert log == ["create:model", "train:model", "finish:model"]


def test_queued_stage_does_not_create_a_gpu_model_while_waiting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = tmp_path / "prepared"
    release = tmp_path / "start"
    monkeypatch.setenv("DCN_GPU_LOCK_SLOT", "0")
    monkeypatch.setenv("DCN_PREPARED_MARKER", str(marker))
    monkeypatch.setenv("DCN_TRAINING_RELEASE", str(release))
    log: list[str] = []
    thread = threading.Thread(target=_PrebuiltDataStage("model", log, tmp_path).run)

    thread.start()
    deadline = time.monotonic() + 2
    while not marker.exists() and time.monotonic() < deadline:
        time.sleep(0.01)

    assert marker.exists()
    assert log == [
        "prebuild:model:gpu=False,prep=True,prep_mode=shared,data=True",
        "components:model:gpu=False,prep=True",
        "create:model:gpu=False,prep=True",
        "prepare:model:gpu=False,prep=True,prep_mode=shared",
    ]
    release.touch()
    thread.join(timeout=2)

    assert not thread.is_alive()
    assert log[1:] == [
        "components:model:gpu=False,prep=True",
        "create:model:gpu=False,prep=True",
        "prepare:model:gpu=False,prep=True,prep_mode=shared",
        "activate:model:gpu=True,prep=False",
        "train:model:gpu=True,prep=False",
        "finish:model:gpu=True,prep=False",
        "release:model:gpu=True,prep=False",
    ]


def test_queued_stage_releases_cached_device_memory_before_gpu_handoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DCN_GPU_LOCK_SLOT", "0")
    log: list[str] = []

    _PrebuiltDataStage("model", log, tmp_path).run()

    assert log[-3:] == [
        "train:model:gpu=True,prep=False",
        "finish:model:gpu=True,prep=False",
        "release:model:gpu=True,prep=False",
    ]
