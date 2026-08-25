from typing import Any

from neuralrec.run.callbacks import CheckpointCallback, ValidationCallback


class PretrainAwareValidationCallback(ValidationCallback):
    def on_epoch_end(self, state: dict[str, Any]) -> None:
        if state["is_pretrain_epoch_end"]:
            super().on_epoch_end(state)


class PretrainAwareCheckpointCallback(CheckpointCallback):
    def on_epoch_end(self, state: dict[str, Any]) -> None:
        if state["is_pretrain_epoch_end"]:
            super().on_epoch_end(state)
