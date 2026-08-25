from __future__ import annotations

import hashlib
import logging
import os
import subprocess
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from time import perf_counter, sleep
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn

from dcn.config.settings import (
    CheckpointConfig,
    DataloaderConfig,
    DayRangeConfig,
    LoggingConfig,
    LrScheduleConfig,
    PretrainConfig,
    RuntimeConfig,
)
from dcn.data import DatasetManager, EmaCounter, FieldConfig
from dcn.datasets.base import DatasetSource, DatasetSourceArtifacts
from dcn.models import LossWrapper, TargetExtractionWrapper
from dcn.nn import (
    MultiTaskEmbeddingLayer,
    MultiTaskEmbeddingLayerTorchRec,
    PrecomputedEmbeddingLookup,
)
from dcn.nn.transformer import LearnedPositionInput
from dcn.training import (
    DayByDayTrainer,
    PretrainAwareCheckpointCallback,
    PretrainAwareValidationCallback,
)
from neuralrec.nn.autocast import AutoCast
from neuralrec.run.callbacks import (
    BestWeights,
    CheckpointCallback,
    EarlyStopping,
    GradientNormClippingCallback,
    LoggingCallback,
    LrSchedule,
    ResourceUsageCallback,
    ValidationCallback,
    WandbCallback,
)
from neuralrec.run.callbacks.validation import PredictionConfig
from neuralrec.run.train import TrainRunner
from utils.global_config import config as global_config
from utils.locks import hold

logger = logging.getLogger(__name__)


def _optimizer_tree(optimizer: Any) -> list[Any]:
    result = []
    pending = [optimizer]
    seen = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        result.append(current)
        pending.extend(getattr(current, "optimizers", ()))
    return result


def _move_tensors(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_tensors(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_tensors(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_tensors(item, device) for item in value)
    return value


@dataclass
class TrainingCallbacks:
    all: list[Any]
    best_weights: BestWeights
    validation: ValidationCallback
    last_checkpoint: CheckpointCallback | None = None
    best_checkpoint: CheckpointCallback | None = None
    early_stopping: EarlyStopping | None = None
    lr_schedule: LrSchedule | None = None


class TrainingStage(ABC):
    """One step of a run, producing an artifact the next step may read."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def run(self) -> None: ...


class TrainedStage(TrainingStage):
    """A stage whose artifact comes out of the shared training loop."""

    def run(self) -> None:
        runner: TrainRunner | None = None
        defer_runner = os.environ.get("DCN_GPU_LOCK_SLOT") is not None
        if self.prebuilds_runner_data:
            with hold(self.prep_lock_path, "prep gate", shared=True):
                started = perf_counter()
                self._prebuild_runner_data()
                if defer_runner:
                    self.prebuild_runner_components()
                runner = self.create_runner()
                self.resume(runner)
                runner.prepare()
        else:
            started = perf_counter()
            with hold(self.prep_lock_path, "prep"):
                if defer_runner:
                    self.prebuild_runner_components()
                runner = self.create_runner()
                self.resume(runner)
            with hold(self.prep_lock_path, "prep warmup", shared=True):
                runner.prepare()
        logger.info(
            "Prepared stage %r in %.6fs", self.name, perf_counter() - started
        )
        self._wait_for_training_release()
        with hold(self.gpu_lock_path, "gpu"):
            with hold(
                self.gpu_gate_path, "gpu gate", shared=self.gpu_gate_shared
            ):
                prepare_again = self._wait_for_queue_gpu(runner)
                if defer_runner:
                    self.activate_runner_device(runner)
                if runner is None:
                    runner = self.create_runner()
                    self.resume(runner)
                    prepare_again = True
                if prepare_again:
                    runner.prepare()
                started = perf_counter()
                try:
                    runner.train()
                    self.finish(runner)
                    logger.info(
                        "Trained stage %r in %.6fs",
                        self.name,
                        perf_counter() - started,
                    )
                finally:
                    if defer_runner:
                        self.release_runner_device_cache()

    def _prebuild_runner_data(self) -> None:
        ready = self.runner_data_ready_path
        if ready is not None and ready.exists():
            self.prebuild_runner_data()
            return
        with hold(self.runner_data_lock_path, "prep data"):
            if ready is None or not ready.exists():
                self.prebuild_runner_data()
                if ready is not None:
                    ready.parent.mkdir(parents=True, exist_ok=True)
                    ready.touch()
                return
        self.prebuild_runner_data()

    def _wait_for_queue_gpu(self, runner: TrainRunner | None) -> bool:
        device = os.environ.get("DCN_GPU_LOCK_DEVICE")
        if device is None or os.environ.get("TRAINING_QUEUE_MONITOR_LIGHT_GPUS") != "1":
            return False
        from utils.training_queue.gpu_check import wait_for_foreign_memory

        return wait_for_foreign_memory(
            device,
            on_wait=(
                None if runner is None else runner.discard_prepared_resources
            ),
        )

    def _wait_for_training_release(self) -> None:
        marker = os.environ.get("DCN_PREPARED_MARKER")
        release = os.environ.get("DCN_TRAINING_RELEASE")
        if marker is None or release is None:
            return
        Path(marker).touch()
        while not Path(release).exists():
            sleep(0.1)

    def release_runner_device_cache(self) -> None:
        if not torch.cuda.is_available():
            return
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    @property
    def gpu_lock_path(self) -> Path | None:
        """Where to queue for the device against other runs, if anywhere.

        Only training and the final scoring are held under it, so a queued run
        reads its data and builds its model on CPU while this one trains. The
        model moves to the device only after this lock is held.
        """
        return None

    @property
    def gpu_gate_path(self) -> Path | None:
        return None

    @property
    def gpu_gate_shared(self) -> bool:
        return False

    @property
    def prep_lock_path(self) -> Path | None:
        """Where to queue against other runs for the caches under ``generated/``.

        Variants share those, and the builders are not written to be raced --
        they clear a directory and rebuild it in place. Warm, this lock is
        uncontended; cold, it is what stops two runs building the same thing
        over each other.
        """
        return None

    @property
    def prebuilds_runner_data(self) -> bool:
        return False

    def prebuild_runner_data(self) -> None: ...

    def prebuild_runner_components(self) -> None: ...

    def activate_runner_device(self, runner: TrainRunner) -> None: ...

    @property
    def runner_data_lock_path(self) -> Path | None:
        ready = self.runner_data_ready_path
        if ready is not None:
            return ready.with_suffix(".lock")
        path = self.prep_lock_path
        if path is None:
            return None
        return path.with_name(f"{path.stem}-data{path.suffix}")

    @property
    def runner_data_ready_path(self) -> Path | None:
        value = os.environ.get("DCN_RUNNER_DATA_READY")
        return Path(value) if value else None

    @abstractmethod
    def create_runner(self) -> TrainRunner: ...

    def resume(self, runner: TrainRunner) -> None: ...

    def finish(self, runner: TrainRunner) -> None: ...


@dataclass
class Experiment(TrainedStage):
    """Config-as-code description of a training run."""

    run_name: str = "experiment"
    base_path: str | Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[2] / "generated"
    )
    seed: int = 42
    invalidate_cache: bool = False

    runtime: RuntimeConfig = None  # type: ignore[assignment]
    day_range: DayRangeConfig = None  # type: ignore[assignment]
    dataloader: DataloaderConfig = None  # type: ignore[assignment]
    pretrain: PretrainConfig = None  # type: ignore[assignment]
    checkpointing: CheckpointConfig = None  # type: ignore[assignment]
    logging: LoggingConfig = None  # type: ignore[assignment]
    lr_schedule: LrScheduleConfig = None  # type: ignore[assignment]

    def settings_defaults(self) -> dict[str, Any]:
        """Defaults for the settings groups a script did not pass.

        A method, not a field default: dataclass fields resolve in reverse MRO,
        so a sibling base that merely inherits a group silently overrides the
        base that declared it. Methods resolve in normal MRO, so an override
        here wins wherever the class sits in the hierarchy.
        """
        return {
            "runtime": RuntimeConfig(),
            "day_range": DayRangeConfig(),
            "dataloader": DataloaderConfig(),
            "pretrain": PretrainConfig(),
            "checkpointing": CheckpointConfig(),
            "logging": LoggingConfig(),
            "lr_schedule": LrScheduleConfig(),
        }

    def __post_init__(self) -> None:
        for name, default in self.settings_defaults().items():
            if getattr(self, name) is None:
                setattr(self, name, default)

    @property
    def name(self) -> str:
        return self.run_name

    @property
    def stages(self) -> Sequence[TrainingStage]:
        return [self]

    @property
    def gpu_lock_path(self) -> Path | None:
        device_lock = self._gpu_device_lock_path()
        training_slot = os.environ.get("DCN_GPU_LOCK_SLOT")
        if training_slot is not None:
            return device_lock.with_name(
                f"{device_lock.stem}-slot-{training_slot}.lock"
            )
        return device_lock

    @property
    def gpu_gate_path(self) -> Path | None:
        if os.environ.get("DCN_GPU_LOCK_SLOT") is None:
            return None
        return self._gpu_device_lock_path()

    @property
    def gpu_gate_shared(self) -> bool:
        return os.environ.get("DCN_GPU_LOCK_SLOT") is not None

    def _gpu_device_lock_path(self) -> Path:
        device = os.environ.get("DCN_GPU_LOCK_DEVICE")
        if device is None:
            visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            device = visible_devices.split(",", 1)[0] if visible_devices else "0"
        if device.isdecimal():
            result = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={device}",
                    "--query-gpu=uuid",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            device = result.stdout.strip()
        return Path(self.base_path) / f"gpu-{device}.lock"

    @property
    def prep_lock_path(self) -> Path | None:
        return Path(self.base_path) / "prep.lock"

    def create_runner(self) -> TrainRunner:
        model = self.create_training_model()
        logger.info("Creating optimizers...")
        optimizer = self.create_optimizers()
        logger.info("Initializing trainer...")
        return self.create_trainer(model, optimizer)

    def prebuild_runner_components(self) -> None:
        _ = self.base_model
        _ = self.callbacks

    def activate_runner_device(self, runner: TrainRunner) -> None:
        optimizer_parameters = {
            id(parameter)
            for group in runner.optimizer.param_groups
            for parameter in group["params"]
        }
        runner.model.to(self.device)
        model_parameters = {id(parameter) for parameter in runner.model.parameters()}
        if not optimizer_parameters <= model_parameters:
            raise RuntimeError("moving the model invalidated optimizer parameters")
        fused = self.device.type == "cuda"
        for optimizer in _optimizer_tree(runner.optimizer):
            if "fused" in optimizer.defaults:
                optimizer.defaults["fused"] = fused
            for group in optimizer.param_groups:
                if "fused" in group:
                    group["fused"] = fused
            for parameter, state in list(optimizer.state.items()):
                optimizer.state[parameter] = _move_tensors(state, self.device)

    def resume(self, runner: TrainRunner) -> None:
        if not self.checkpointing.load_checkpoint:
            logger.info("Skipping checkpoint loading (load_checkpoint=False)")
            return
        assert self.callbacks.last_checkpoint is not None, (
            "load_checkpoint needs checkpointing.enabled"
        )
        if self.callbacks.last_checkpoint.load_latest(runner.state):
            logger.info("Resumed from latest checkpoint")

    def finish(self, runner: TrainRunner) -> None:
        if self.callbacks.best_checkpoint is not None:
            logger.info("Saving best checkpoint link...")
            self.callbacks.best_checkpoint.save_best()

    def setup(self) -> None:
        base_path = Path(self.base_path)
        base_path.mkdir(parents=True, exist_ok=True)
        global_config.initialize(base_path)

        torch.set_float32_matmul_precision("high")
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(False)

    @cached_property
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def runner_build_device(self) -> torch.device:
        if os.environ.get("DCN_GPU_LOCK_SLOT") is not None:
            return torch.device("cpu")
        return self.device

    @cached_property
    def artifacts(self) -> DatasetSourceArtifacts:
        return self.create_dataset_source().artifacts

    @cached_property
    def item_embeddings(self) -> PrecomputedEmbeddingLookup:
        """The catalog's pretrained item vectors, read from disk once.

        Reading them is seconds of parquet, so every caller that wants the
        table — or only its size — shares this one.
        """
        return PrecomputedEmbeddingLookup.from_parquet(
            self.artifacts.precomputed_embeddings[self.artifacts.item_id_column],
            learnable_default=False,
            strict=False,
        )

    @cached_property
    def dataset_key(self) -> str:
        """Short id of the prepared source data; every cache below it keys on this."""
        source = str(Path(self.artifacts.main_parquet).resolve())
        return hashlib.sha1(source.encode()).hexdigest()[:12]

    @cached_property
    def dataset_cache_dir(self) -> Path:
        return global_config.dataset_path / self.dataset_key

    @cached_property
    def counters_cache_dir(self) -> Path:
        return global_config.counters_path / self.dataset_key

    def counter(
        self,
        keys: list[str],
        fields: list[FieldConfig],
        aggregations: Sequence[str] = ("mean",),
    ) -> EmaCounter:
        return EmaCounter(
            keys=keys,
            fields=fields,
            cache_dir=self.counters_cache_dir,
            aggregations=aggregations,
        )

    @cached_property
    def counters(self) -> list[EmaCounter]:
        return self.create_counters()

    @cached_property
    def dataset_manager(self) -> DatasetManager:
        return self.create_dataset_manager()

    @cached_property
    def num_counters(self) -> int:
        return len(self.dataset_manager.counter_columns)

    @cached_property
    def training_day_bounds(self) -> tuple[int, int]:
        available_days = self.dataset_manager.get_available_days()
        training_days = [
            day
            for day in available_days
            if self.day_range.start_day <= day <= self.day_range.end_day
        ]
        start_day, end_day = min(training_days), max(training_days)
        logger.info(
            "Training on %s days (from %s to %s)",
            len(training_days),
            start_day,
            end_day,
        )
        return start_day, end_day

    @cached_property
    def callbacks(self) -> TrainingCallbacks:
        return self.create_callbacks()

    @abstractmethod
    def create_dataset_source(self) -> DatasetSource: ...

    @abstractmethod
    def create_counters(self) -> list[EmaCounter]: ...

    def create_dataset_manager(self) -> DatasetManager:
        counter_columns: list[str] = []
        for counter in self.counters:
            counter_columns.extend(counter.get_output_columns())

        return DatasetManager(
            main_parquet=self.artifacts.main_parquet,
            columns=self.artifacts.columns,
            counter_columns=counter_columns,
            cache_dir=self.dataset_cache_dir,
            counters=self.counters,
            invalidate_cache=self.invalidate_cache,
            timestamp_column=self.artifacts.timestamp_column,
        )

    @abstractmethod
    def _create_model(self) -> nn.Module: ...

    @cached_property
    def base_model(self) -> nn.Module:
        model: nn.Module = self._create_model().to(self.runner_build_device)
        logger.info("Model architecture:\n%s", model)
        logger.info(
            "Total parameters: %s",
            f"{sum(p.numel() for p in model.parameters()):,}",
        )
        return model

    def create_training_model(self) -> nn.Module:
        logger.info("Wrapping model with LossWrapper...")
        return self.apply_runtime_wrappers(
            LossWrapper(
                model=self.base_model,
                criterion=self.create_criterion(),
                metrics=self.create_metrics(),
            )
        )

    def apply_runtime_wrappers(self, model: nn.Module) -> nn.Module:
        dtype = self.runtime.dtype
        if dtype != torch.float32:
            logger.info("Wrapping model with AutoCast (%s)...", dtype)
            model = AutoCast(model, dtype=dtype)
        if self.runtime.compile:
            logger.info("Compiling training model...")
            model = torch.compile(model)
        return model

    @abstractmethod
    def create_criterion(self) -> nn.Module: ...

    def create_metrics(self) -> list[TargetExtractionWrapper]:
        return []

    @abstractmethod
    def create_optimizers(self) -> torch.optim.Optimizer: ...

    @property
    def embedding_types(self) -> Sequence[type[nn.Module]]:
        """Modules whose parameters take the embedding learning rate."""
        return [MultiTaskEmbeddingLayer]

    def split_parameters(
        self, model: nn.Module, embedding_types: Sequence[type[nn.Module]]
    ) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
        """(embedding, everything else), so the two can take different rates."""
        position_ids = {
            id(parameter)
            for module in model.modules()
            if isinstance(module, LearnedPositionInput)
            for parameter in module.parameters()
        }
        embedding_ids: set[int] = set()
        for module in model.modules():
            if isinstance(module, tuple(embedding_types)):
                embedding_ids.update(
                    id(p) for p in module.parameters() if id(p) not in position_ids
                )
        embedding = [p for p in model.parameters() if id(p) in embedding_ids]
        deep = [p for p in model.parameters() if id(p) not in embedding_ids]
        return embedding, deep

    def _create_embedding_optimizer(
        self,
        embedding: nn.Module,
        lr: float,
        sparse: bool,
        weight_decay: float,
    ) -> torch.optim.Optimizer:
        if isinstance(embedding, MultiTaskEmbeddingLayerTorchRec):
            from torchrec.optim import KeyedOptimizerWrapper

            return KeyedOptimizerWrapper(
                dict(embedding.embedding_bag_collection.named_parameters()),
                lambda params: torch.optim.Adam(
                    params,
                    lr=lr,
                    weight_decay=weight_decay,
                    fused=self.runner_build_device.type == "cuda",
                ),
            )

        embedding_params = list(embedding.parameters())
        if sparse:
            return torch.optim.SparseAdam(embedding_params, lr=lr)
        return torch.optim.Adam(
            embedding_params,
            lr=lr,
            weight_decay=weight_decay,
            fused=self.runner_build_device.type == "cuda",
        )

    def create_checkpoint_callback(
        self, prefix: str, **kwargs: Any
    ) -> CheckpointCallback:
        return PretrainAwareCheckpointCallback(
            checkpoint_dir=str(global_config.checkpoints_path),
            run_name=self.run_name,
            prefix=prefix,
            **kwargs,
        )

    def create_validation_callback(self) -> ValidationCallback:
        return PretrainAwareValidationCallback(
            prediction_config=self._create_prediction_config()
        )

    @property
    def trainer_runs_validation(self) -> bool:
        """Whether the trainer fires the validation callback itself."""
        return False

    def _create_checkpoint_callbacks(self) -> list[CheckpointCallback]:
        if not self.checkpointing.enabled:
            return []
        return [
            self.create_checkpoint_callback(
                "last",
                save_strategy="last_n",
                n_checkpoints=self.checkpointing.last_n_checkpoints,
            ),
            self.create_checkpoint_callback(
                "best",
                save_strategy=self.checkpointing.best_strategy,
                n_checkpoints=self.checkpointing.best_n_checkpoints,
                metric_name=self.checkpointing.best_metric_name,
                metric_mode=self.checkpointing.best_metric_mode,
                metric_prefix=self.checkpointing.best_metric_prefix,
            ),
        ]

    def create_callbacks(self) -> TrainingCallbacks:
        checkpoints = self._create_checkpoint_callbacks()
        best_weights = BestWeights(
            metric_name=self.checkpointing.best_metric_name,
            metric_mode=self.checkpointing.best_metric_mode,
            metric_prefix=self.checkpointing.best_metric_prefix,
        )
        validation = self.create_validation_callback()

        lr_schedule = LrSchedule(
            self.lr_schedule.shape,
            warmup_fraction=self.lr_schedule.warmup_fraction,
            min_lr_fraction=self.lr_schedule.min_lr_fraction,
            cycles=self.lr_schedule.cycles,
            timescale_steps=self.lr_schedule.timescale_steps,
            timescale_fraction=self.lr_schedule.timescale_fraction,
            power_exponent=self.lr_schedule.power_exponent,
            power_transition_tokens=self.lr_schedule.power_transition_tokens,
            optimizer_group_scope=self.lr_schedule.optimizer_group_scope,
            stop_at_horizon=not getattr(
                self, "adaptive_schedule_early_stopping", False
            ),
        )
        callbacks: list[Any] = [
            lr_schedule,
            *([] if self.trainer_runs_validation else [validation]),
            best_weights,
            *checkpoints,
            ResourceUsageCallback(
                model=self.base_model,
                embedding_parameters=self.split_parameters(
                    self.base_model, self.embedding_types
                )[0],
            ),
            LoggingCallback().every_n_steps(self.logging.log_interval),
            WandbCallback(
                run_name=self.run_name,
                project=self.logging.wandb_project,
            ),
        ]
        if self.runtime.gradient_clip_norm:
            callbacks.insert(
                0,
                GradientNormClippingCallback(max_norm=self.runtime.gradient_clip_norm),
            )

        return TrainingCallbacks(
            all=callbacks,
            best_weights=best_weights,
            validation=validation,
            lr_schedule=lr_schedule,
            last_checkpoint=checkpoints[0] if checkpoints else None,
            best_checkpoint=checkpoints[1] if checkpoints else None,
        )

    def _create_prediction_config(self) -> PredictionConfig | None:
        if not self.logging.enable_predictions:
            return None
        return PredictionConfig(
            predictions_dir=global_config.predictions_path / self.run_name,
            int_columns=dict(self.logging.prediction_int_columns),
            float_columns=dict(self.logging.prediction_float_columns),
        )

    def create_trainer(self, model: nn.Module, optimizer: Any) -> DayByDayTrainer:
        start_day, end_day = self.training_day_bounds
        return DayByDayTrainer(
            dataset_manager=self.dataset_manager,
            start_day=start_day,
            end_day=end_day,
            val_callback=self.callbacks.validation,
            train_batch_size=self.dataloader.batch_size,
            val_batch_size=self.dataloader.val_batch_size,
            num_workers=self.dataloader.num_workers,
            prefetch_factor=self.dataloader.prefetch_factor,
            pretrain_days=self.pretrain.days,
            pretrain_num_epochs=self.pretrain.num_epochs,
            pretrain_shuffle_days=self.pretrain.shuffle_days,
            model=model,
            optimizer=optimizer,
            callbacks=self.callbacks.all,
        )
