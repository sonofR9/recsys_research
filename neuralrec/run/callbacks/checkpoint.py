from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import torch
import yaml

from neuralrec.run.callbacks.base import Callback
from neuralrec.run.train import TrainRunner
from neuralrec.utils import to_float
from neuralrec.utils import EXTRA_METRICS


class CheckpointCallback(Callback):
    _state_fields: tuple[str, ...] = ()

    def __init__(
        self,
        checkpoint_dir: str = "./checkpoints",
        run_name: str = "",
        prefix: str = "checkpoint",
        save_strategy: Literal["last_n", "best_n"] = "last_n",
        n_checkpoints: int = 1,
        metric_name: str | None = None,
        metric_mode: Literal["min", "max"] = "min",
        metric_prefix: str = "epoch/val",
    ):
        self.checkpoint_dir = Path(checkpoint_dir) / run_name

        self.run_name = run_name
        self.prefix = prefix
        self.save_strategy = save_strategy
        self.n_checkpoints = n_checkpoints
        self.metric_name = metric_name
        self.metric_mode = metric_mode
        self.metric_prefix = metric_prefix

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._metadata_file = self.checkpoint_dir / f"{prefix}_metadata.yaml"

        if save_strategy == "best_n" and metric_name is None:
            raise ValueError(
                "metric_name must be provided when save_strategy is 'best_n'"
            )

    def _load_metadata(self) -> dict[str, Any]:
        if self._metadata_file.exists():
            with open(self._metadata_file, "r") as f:
                return yaml.safe_load(f) or {}  # type: ignore
        return {}

    def _collect_state_dicts(self, state: dict[str, Any]) -> dict[str, Any]:
        checkpoint_state = {}

        for key, value in state.items():
            if hasattr(value, "state_dict") and callable(value.state_dict):
                checkpoint_state[key] = value.state_dict()

        return checkpoint_state

    def _checkpoint_filename(self, epoch: int) -> str:
        return f"{self.prefix}_epoch_{epoch}.pt"

    def save_checkpoint(
        self,
        state: dict[str, Any],
        epoch: int,
        metric_value: float | None = None,
    ) -> None:
        checkpoint_state = self._collect_state_dicts(state)
        checkpoint_path = self.checkpoint_dir / self._checkpoint_filename(epoch)
        torch.save(checkpoint_state, checkpoint_path)

        metadata = self._load_metadata()
        metadata["checkpoints"] = metadata.get("checkpoints", [])

        checkpoint_info = {
            "epoch": epoch,
            "path": str(checkpoint_path),
            "global_step": checkpoint_state.get("global_step", 0),
        }

        if metric_value is not None:
            checkpoint_info["metric_value"] = to_float(metric_value)

        # A rerun under the same name inherits the previous run's metadata, and
        # writes over its files. Two entries for one path would let the pruned
        # one delete the file the kept one still names.
        metadata["checkpoints"] = [
            entry
            for entry in metadata["checkpoints"]
            if entry["path"] != checkpoint_info["path"]
        ]
        metadata["checkpoints"].append(checkpoint_info)

        if self.save_strategy == "best_n":
            reverse = self.metric_mode == "max"
            sorted_checkpoints = sorted(
                metadata["checkpoints"],
                key=lambda x: x.get(
                    "metric_value", float("inf") if not reverse else float("-inf")
                ),
                reverse=reverse,
            )
        else:
            sorted_checkpoints = sorted(
                metadata["checkpoints"], key=lambda x: x["epoch"], reverse=True
            )

        metadata["checkpoints"] = sorted_checkpoints[: self.n_checkpoints]
        kept_paths = {entry["path"] for entry in metadata["checkpoints"]}

        for ckpt in sorted_checkpoints[self.n_checkpoints :]:
            if ckpt["path"] in kept_paths:
                continue
            ckpt_path = Path(ckpt["path"])
            if ckpt_path.exists():
                ckpt_path.unlink()

        with open(self._metadata_file, "w") as f:
            yaml.dump(metadata, f, default_flow_style=False)
        print(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(
        self,
        state: dict[str, Any],
        checkpoint_path: str | Path,
        allow_missing: bool = False,
        allow_extra: bool = True,
    ) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        checkpoint_keys = set(checkpoint.keys())

        state_keys = set()
        for key in state.keys():
            if hasattr(state[key], "load_state_dict"):
                state_keys.add(key)

        missing_keys = state_keys - checkpoint_keys
        extra_keys = checkpoint_keys - state_keys

        if missing_keys and not allow_missing:
            raise ValueError(f"Missing keys in checkpoint: {missing_keys}")
        if extra_keys and not allow_extra:
            raise ValueError(f"Extra keys in checkpoint: {extra_keys}")

        for key in state_keys & checkpoint_keys:
            if key in state and hasattr(state[key], "load_state_dict"):
                state[key].load_state_dict(checkpoint[key])

        print(f"Checkpoint loaded from {checkpoint_path}")

    def _get_best_checkpoint_path(self) -> Path | None:
        metadata = self._load_metadata()
        checkpoints = metadata.get("checkpoints", [])
        if not checkpoints:
            return None

        if self.save_strategy == "best_n":
            if self.metric_mode == "max":
                best_checkpoint = max(checkpoints, key=lambda x: x["metric_value"])
            else:
                best_checkpoint = min(checkpoints, key=lambda x: x["metric_value"])
        else:
            best_checkpoint = max(checkpoints, key=lambda x: x["epoch"])

        return Path(best_checkpoint["path"])

    def get_latest_checkpoint(self) -> Path | None:
        metadata = self._load_metadata()
        checkpoints = metadata.get("checkpoints", [])
        if not checkpoints:
            return None
        latest = max(checkpoints, key=lambda x: x["epoch"])
        return Path(latest["path"])

    def save_best(self) -> None:
        best_path = self._get_best_checkpoint_path()
        if best_path is None:
            print("No checkpoints to link as best")
            return

        link_path = self.checkpoint_dir / f"{self.prefix}_best.pt"

        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()

        link_path.symlink_to(best_path.name)
        print(f"Best checkpoint linked: {link_path} -> {best_path.name}")

    def load_best(
        self,
        state: dict[str, Any],
        allow_missing: bool = True,
        allow_extra: bool = True,
    ) -> bool:
        checkpoint_path = self._get_best_checkpoint_path()
        if checkpoint_path is None:
            return False
        self.load_checkpoint(state, checkpoint_path, allow_missing, allow_extra)
        return True

    def load_latest(
        self,
        state: dict[str, Any],
        allow_missing: bool = True,
        allow_extra: bool = True,
    ) -> bool:
        checkpoint_path = self.get_latest_checkpoint()
        if checkpoint_path is None:
            return False
        self.load_checkpoint(state, checkpoint_path, allow_missing, allow_extra)
        return True

    def on_epoch_end(self, state: dict[str, Any]) -> None:
        runner: TrainRunner = state["train_runner"]

        if self.save_strategy != "best_n":
            self.save_checkpoint(state, runner.current_epoch)
            return

        metrics = state.get(EXTRA_METRICS, {}).get(self.metric_prefix, {})
        if self.metric_name not in metrics:
            # Nothing to rank this epoch's weights by, and the previous epoch's
            # score is not theirs to be saved under.
            print(
                f"No {self.metric_prefix}/{self.metric_name} this epoch; "
                "skipping the best-checkpoint save"
            )
            return
        self.save_checkpoint(state, runner.current_epoch, metrics[self.metric_name])
