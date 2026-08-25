import math
from typing import Any

import polars as pl
import pytest
import torch
from torch.utils.data import DataLoader

from dcn.data import SequenceDataset, collate_sequence_batch
from dcn.models.two_tower import (
    TwoTowerLoss,
)
from dcn.tests.helpers import (
    CATEGORICAL,
    HISTORY_COUNTERS,
    ITEM_COUNTERS,
    two_tower_loss,
    two_tower_model,
)
from dcn.training import EpochTrainer
from neuralrec.run.callbacks import BestWeights, CheckpointCallback, EarlyStopping
from neuralrec.run.callbacks.base import Callback
from neuralrec.run.callbacks.validation import ValidationCallback
from neuralrec.utils import EXTRA_METRICS

CATALOG = 20  # item ids 1..20


pytestmark = pytest.mark.usefixtures("cpu_attention")


def _write_cyclic_parquet(path, *, num_users: int = 80, seq_len: int = 8) -> None:
    """Deterministic next-item chain: item_{t+1} = item_t + 1 (mod CATALOG).

    Every user walks the same cycle from a different phase, so the next item is
    a learnable function of the current one and the in-batch loss must fall.
    """
    rows = []
    for uid in range(num_users):
        for pos in range(seq_len):
            item = (uid + pos) % CATALOG + 1
            rows.append(
                {
                    "uid": uid,
                    "timestamp": pos,
                    "compact_item_id": item,
                    "artist_id": item % 5 + 1,
                    "history_count": float(pos),
                    "item_count": float(item) * 0.1,
                }
            )
    pl.DataFrame(rows).write_parquet(path)


def _make_loader(path, cache_dir, *, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = SequenceDataset(
        [path],
        [*CATEGORICAL, *HISTORY_COUNTERS, *ITEM_COUNTERS],
        cache_dir,
        user_column="uid",
        timestamp_column="timestamp",
        max_seq_len=16,
        min_seq_len=2,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_sequence_batch,
    )


def _make_training_model() -> TwoTowerLoss:
    return two_tower_loss(
        two_tower_model(num_hashes=2),
        hash_size=CATALOG + 1,
        num_in_batch_negatives=8,
    )


class _LossRecorder(Callback):
    """Captures the mean training loss of each epoch."""

    def __init__(self) -> None:
        super().__init__()
        self.epoch_losses: list[float] = []
        self.validated_by_epoch_end: list[bool] = []
        self._current: list[float] = []

    def on_step_end(
        self, state: dict[str, Any], batch: Any, out: dict[str, Any]
    ) -> None:
        self._current.append(float(out["loss"].detach()))

    def on_epoch_end(self, state: dict[str, Any]) -> None:
        self.epoch_losses.append(sum(self._current) / len(self._current))
        self.validated_by_epoch_end.append(
            "loss" in state.get(EXTRA_METRICS, {}).get("epoch/val", {})
        )
        self._current = []


class _ValidationScore(Callback):
    def __init__(
        self, scores: list[float], selected_weights: list[float] | None = None
    ) -> None:
        self.scores = scores
        self.selected_weights = selected_weights
        self.epochs_seen: list[int] = []

    def on_epoch_end(self, state: dict[str, Any]) -> None:
        epoch = state["train_runner"].current_epoch
        self.epochs_seen.append(epoch)
        if self.selected_weights is not None:
            state["model"].weight.data.fill_(self.selected_weights[epoch])
        state.setdefault(EXTRA_METRICS, {})["epoch/val_true"] = {
            "recall@100": self.scores[epoch]
        }


class _ScalarLoss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))

    def forward(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"loss": self.weight.square() * batch.mean()}


def test_epoch_trainer_runs_validates_and_learns(tmp_path):
    torch.manual_seed(0)
    parquet = tmp_path / "events.parquet"
    _write_cyclic_parquet(parquet)

    train_loader = _make_loader(
        parquet, tmp_path / "train_cache", batch_size=16, shuffle=True
    )
    val_loader = _make_loader(
        parquet, tmp_path / "val_cache", batch_size=16, shuffle=False
    )

    wrapper = _make_training_model()
    optimizer = torch.optim.Adam(wrapper.parameters(), lr=1e-2, fused=True)
    val_callback = ValidationCallback()
    recorder = _LossRecorder()

    trainer = EpochTrainer(
        model=wrapper,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=10,
        val_callback=val_callback,
        callbacks=[recorder],
    )
    trainer.train()

    assert len(recorder.epoch_losses) == 10
    assert all(math.isfinite(loss) for loss in recorder.epoch_losses)
    assert recorder.epoch_losses[-1] < recorder.epoch_losses[0]

    val_metrics = trainer.state[EXTRA_METRICS]["epoch/val"]
    assert "loss" in val_metrics and "hit_rate" in val_metrics
    assert math.isfinite(val_metrics["loss"])
    assert 0.0 <= val_metrics["hit_rate"] <= 1.0

    # Validation happens before the rest of the epoch-end callbacks, so a
    # logger or a checkpoint rule reads this epoch's metrics, not the last one's.
    assert all(recorder.validated_by_epoch_end)


def test_epoch_trainer_honors_early_stopping_request() -> None:
    model = _ScalarLoss()
    scores = _ValidationScore(
        [0.4, 0.3, 0.2, 0.1, 0.0], selected_weights=[1, 2, 3, 4, 5]
    )
    best_weights = BestWeights(
        metric_name="recall@100", metric_prefix="epoch/val_true"
    )
    stopping = EarlyStopping(
        metric_name="recall@100",
        metric_prefix="epoch/val_true",
        patience=2,
    )
    trainer = EpochTrainer(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        train_loader=DataLoader(torch.ones(1), batch_size=1),
        num_epochs=5,
        callbacks=[scores, best_weights, stopping],
    )

    trainer.train()

    assert scores.epochs_seen == [0, 1, 2]
    assert stopping.best_epoch == 0
    assert best_weights.restore(model)
    assert model.weight.item() == 1


def test_epoch_trainer_resumes_after_the_checkpointed_epoch() -> None:
    model = _ScalarLoss()
    scores = _ValidationScore([0.4, 0.3, 0.2, 0.1])
    trainer = EpochTrainer(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        train_loader=DataLoader(torch.ones(1), batch_size=1),
        num_epochs=4,
        callbacks=[scores],
    )
    trainer.load_state_dict({"current_epoch": 1, "step": 1, "global_step": 2})

    trainer.train()

    assert scores.epochs_seen == [2, 3]


def test_epoch_trainer_resume_restores_best_from_before_interruption(
    tmp_path,
) -> None:
    first_model = _ScalarLoss()
    first_scores = _ValidationScore(
        [0.9, 0.4], selected_weights=[1.0, 2.0]
    )
    first_best = BestWeights(
        metric_name="recall@100", metric_prefix="epoch/val_true"
    )
    first_checkpoint = CheckpointCallback(
        checkpoint_dir=str(tmp_path),
        run_name="resume_best",
        prefix="last",
        save_strategy="last_n",
        n_checkpoints=1,
    )
    first_trainer = EpochTrainer(
        model=first_model,
        optimizer=torch.optim.SGD(first_model.parameters(), lr=0.01),
        train_loader=DataLoader(torch.ones(1), batch_size=1),
        num_epochs=2,
        callbacks=[first_scores, first_best, first_checkpoint],
    )
    first_trainer.train()

    resumed_model = _ScalarLoss()
    resumed_scores = _ValidationScore(
        [0.0, 0.0, 0.8, 0.7], selected_weights=[0.0, 0.0, 3.0, 4.0]
    )
    resumed_best = BestWeights(
        metric_name="recall@100", metric_prefix="epoch/val_true"
    )
    resumed_checkpoint = CheckpointCallback(
        checkpoint_dir=str(tmp_path),
        run_name="resume_best",
        prefix="last",
        save_strategy="last_n",
        n_checkpoints=1,
    )
    resumed_trainer = EpochTrainer(
        model=resumed_model,
        optimizer=torch.optim.SGD(resumed_model.parameters(), lr=0.01),
        train_loader=DataLoader(torch.ones(1), batch_size=1),
        num_epochs=4,
        callbacks=[resumed_scores, resumed_best, resumed_checkpoint],
    )

    assert resumed_checkpoint.load_latest(resumed_trainer.state)
    resumed_trainer.train()

    assert resumed_scores.epochs_seen == [2, 3]
    assert resumed_best.best_score == 0.9
    assert resumed_best.best_epoch == 0
    assert resumed_best.restore(resumed_model)
    assert resumed_model.weight.item() == 1.0
