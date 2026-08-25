from pathlib import Path

import torch
import yaml
from torch import nn

from neuralrec.run.callbacks import CheckpointCallback
from neuralrec.run.train import TrainRunner


def _runner() -> TrainRunner:
    model = nn.Linear(2, 2)
    return TrainRunner(model=model, optimizer=torch.optim.SGD(model.parameters(), 0.1))


def _callback(tmp_path: Path, n_checkpoints: int) -> CheckpointCallback:
    return CheckpointCallback(
        checkpoint_dir=str(tmp_path),
        run_name="run",
        prefix="last",
        save_strategy="last_n",
        n_checkpoints=n_checkpoints,
    )


def _saved_epochs(tmp_path: Path) -> list[int]:
    files = (tmp_path / "run").glob("last_epoch_*.pt")
    return sorted(int(path.stem.rsplit("_", 1)[1]) for path in files)


def _recorded_epochs(tmp_path: Path) -> list[int]:
    metadata = yaml.safe_load((tmp_path / "run" / "last_metadata.yaml").read_text())
    return sorted(entry["epoch"] for entry in metadata["checkpoints"])


def test_keeps_the_latest_epoch(tmp_path: Path) -> None:
    state = _runner().state
    callback = _callback(tmp_path, n_checkpoints=1)

    for epoch in range(3):
        callback.save_checkpoint(state, epoch)

    assert _saved_epochs(tmp_path) == [2]
    assert _recorded_epochs(tmp_path) == [2]


def test_every_recorded_checkpoint_is_on_disk_when_a_run_is_repeated(
    tmp_path: Path,
) -> None:
    """A rerun under the same name meets the previous run's metadata."""
    state = _runner().state
    for _ in range(2):
        callback = _callback(tmp_path, n_checkpoints=2)
        for epoch in range(3):
            callback.save_checkpoint(state, epoch)

    assert _recorded_epochs(tmp_path) == _saved_epochs(tmp_path)


def test_resaving_an_epoch_records_it_once(tmp_path: Path) -> None:
    state = _runner().state
    callback = _callback(tmp_path, n_checkpoints=3)

    callback.save_checkpoint(state, 0)
    callback.save_checkpoint(state, 0)

    assert _recorded_epochs(tmp_path) == [0]
    assert _saved_epochs(tmp_path) == [0]


def test_keeps_the_latest_n_epochs(tmp_path: Path) -> None:
    state = _runner().state
    callback = _callback(tmp_path, n_checkpoints=2)

    for epoch in range(4):
        callback.save_checkpoint(state, epoch)

    assert _saved_epochs(tmp_path) == [2, 3]
    assert _recorded_epochs(tmp_path) == [2, 3]
