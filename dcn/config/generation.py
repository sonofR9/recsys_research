"""Generation variants: predict what a user interacts with next."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import pickle
from abc import abstractmethod
from contextlib import AbstractContextManager, nullcontext
from dataclasses import asdict, dataclass, field, replace
from functools import cached_property
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Literal, get_args

import mup
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dcn.config.experiment import TrainingCallbacks
from dcn.config.networks import build_causal_transformer, build_transformer_decoder
from dcn.config.retrieval import RetrievalExperiment, SampledSoftmaxExperiment
from dcn.config.semantic import SemanticExperiment, SemanticIdConfig
from dcn.config.settings import (
    TRANSFORMER,
    CheckpointConfig,
    DataloaderConfig,
    DayRangeConfig,
    RuntimeConfig,
    TransformerConfig,
    transformer_metadata,
)
from dcn.config.yambda_base import YambdaSourceExperiment
from dcn.data import EmaCounter
from dcn.datasets.yambda import EVENT_TYPE_IDS
from dcn.eval import (
    GenerationRecallCallback,
    TrueMetricCallback,
    build_catalog_batch,
    build_interaction_sets,
)
from dcn.models import (
    ActionTokenizer,
    BosTokenizer,
    CausalTokenDecoder,
    ClsTokenMode,
    EventTokenizer,
    ItemTokenizer,
    NextItemTargets,
    Seq2SeqTokenDecoder,
    SemanticIdConstraint,
    SemanticIdTokenizer,
    SequenceRetrievalModel,
    SequenceTargets,
    TimeWindowTargets,
    TimestampDeltaTokenizer,
    TokenDecoder,
    TokenPredictionLoss,
    TwoTowerLoss,
)
from dcn.nn import (
    ConcatenatedItemFeatureResidual,
    DirectAddItemFeature,
    GemmaItemFeatureResidual,
    LayerItemFeatureFusion,
    OfflineInBatchSoftmax,
    RandomCatalogNegatives,
    StreamingInBatchSoftmax,
)
from dcn.nn.sampled_softmax import Correction
from dcn.nn.semantic_embedding import (
    CombinedSemanticIdEmbedding,
    SemanticIdEmbedding,
)
from dcn.nn.types import ModuleWithDim
from dcn.training import register_stable_optimizer_groups
from dcn.training_metadata import (
    GENERATION_TRAINING_SEMANTICS_REVISION,
    NEGATIVE_SAMPLING_SEMANTICS_REVISION,
    TIMESTAMP_BIN_SEMANTICS_REVISION,
)
from neuralrec.run.callbacks import EarlyStopping
from neuralrec.run.callbacks.validation import ValidationCallback
from neuralrec.run.train import TrainRunner
from neuralrec.utils import EXTRA_METRICS, to_float
from utils.global_config import config as global_config
from utils.locks import hold

logger = logging.getLogger(__name__)

_EVENT_TYPE_COLUMN = "event_type_id"
_SECONDS_IN_DAY = 86_400

# Restated on every class a script instantiates: dataclass fields resolve in
# reverse MRO, so a sibling base inheriting the framework default would win.
_NUM_EPOCHS = 20

_TRUE_METRIC_PREFIX = "epoch/val_true"
_SELECTION_METRIC = "recall"
NegativeSampling = Literal[
    "online_logq",
    "offline_logq",
    "random",
    "random_offline_logq",
    "in_batch_no_logq",
    "mixed_online_logq",
    "mixed_offline_logq",
    "mixed_online_global_q",
    "mixed_online_global_q_negative_only",
]
PerLayerItemFeatures = Literal["none", "direct_add", "concat_residual", "gemma_ple"]


def _declared_initialization_parameter_ids(model: nn.Module) -> set[int]:
    return {
        id(module.position_embeddings.weight)
        for module in model.modules()
        if getattr(module, "preserve_declared_initialization", False)
    }


def _initializer_rng_nonadvancing_parameter_ids(model: nn.Module) -> set[int]:
    return {
        id(parameter)
        for module in model.modules()
        if getattr(module, "initializer_rng_nonadvancing", False)
        for parameter in module.parameters()
    }


def _initializer_parameter_runs(
    model: nn.Module,
) -> list[tuple[bool, list[tuple[str, nn.Parameter]]]]:
    marked = _initializer_rng_nonadvancing_parameter_ids(model)
    runs: list[tuple[bool, list[tuple[str, nn.Parameter]]]] = []
    for name, parameter in model.named_parameters():
        nonadvancing = id(parameter) in marked
        if not runs or runs[-1][0] != nonadvancing:
            runs.append((nonadvancing, []))
        runs[-1][1].append((name, parameter))
    return runs


def _initializer_rng_context(
    nonadvancing: bool, parameters: list[tuple[str, nn.Parameter]]
) -> AbstractContextManager[None]:
    if not nonadvancing:
        return nullcontext()
    devices = sorted(
        {
            (
                parameter.device.index
                if parameter.device.index is not None
                else torch.cuda.current_device()
            )
            for _, parameter in parameters
            if parameter.device.type == "cuda"
        }
    )
    return torch.random.fork_rng(devices=devices)


def _initialize_standard_parameters(model: nn.Module, initializer_std: float) -> None:
    preserved = _declared_initialization_parameter_ids(model)
    for nonadvancing, parameters in _initializer_parameter_runs(model):
        with _initializer_rng_context(nonadvancing, parameters):
            for name, parameter in parameters:
                if id(parameter) in preserved:
                    continue
                if "weight" in name:
                    nn.init.trunc_normal_(
                        parameter,
                        std=initializer_std,
                        a=-2 * initializer_std,
                        b=2 * initializer_std,
                    )
                elif "bias" in name:
                    nn.init.zeros_(parameter)


def _initialize_mup_parameters(model: nn.Module, initializer_std: float) -> None:
    preserved = _declared_initialization_parameter_ids(model)
    for nonadvancing, parameters in _initializer_parameter_runs(model):
        with _initializer_rng_context(nonadvancing, parameters):
            for name, parameter in parameters:
                if id(parameter) in preserved:
                    continue
                if name == "query_projection.weight" or name.endswith(".q_proj.weight"):
                    nn.init.zeros_(parameter)
                elif "weight" in name and parameter.ndim >= 2:
                    mup.init.trunc_normal_(
                        parameter,
                        std=initializer_std,
                        a=-2 * initializer_std,
                        b=2 * initializer_std,
                    )
                elif name.endswith("weight"):
                    nn.init.ones_(parameter)
                elif "bias" in name:
                    nn.init.zeros_(parameter)


class _PeriodicFinalValidationCallback(ValidationCallback):
    def __init__(self, every_n_epochs: int, total_epochs: int) -> None:
        super().__init__(prediction_config=None, cache_batches=True)
        self.every_n_epochs = every_n_epochs
        self.total_epochs = total_epochs

    def on_epoch_end(self, state: dict[str, Any]) -> None:
        epoch = state["train_runner"].current_epoch + 1
        if epoch != self.total_epochs and epoch % self.every_n_epochs:
            return
        super().on_epoch_end(state)


@dataclass
class HistoryGenerationExperiment(YambdaSourceExperiment, RetrievalExperiment):
    """A causal transformer over a user's likes."""

    transformer: TransformerConfig = replace(TRANSFORMER, learned_positions="forward")
    # A property of the token sequence rather than of the stack that reads it:
    # the model has to be able to predict the item the bos precedes.
    bos: bool = False
    cls_token: bool = False
    cls_token_mode: ClsTokenMode = "none"

    @property
    def prebuilds_runner_data(self) -> bool:
        return not self.invalidate_cache

    def prebuild_runner_data(self) -> None:
        _ = self.sequence_train_loader
        _ = self.sequence_val_loader

    def settings_defaults(self) -> dict[str, Any]:
        return {
            **super().settings_defaults(),
            "day_range": DayRangeConfig(start_day=0, end_day=299),
            "dataloader": DataloaderConfig(
                batch_size=128, val_batch_size=128, num_workers=4, prefetch_factor=4
            ),
            # Compiling costs ~10s of the first epoch against ~1s saved on every
            # epoch after it, and inductor's on-disk cache means only the first
            # run of a sweep pays even that.
            "runtime": RuntimeConfig(dtype=torch.bfloat16, compile=True),
        }

    def create_counters(self) -> list[EmaCounter]:
        return []

    @property
    def row_filter(self) -> pl.Expr | None:
        return pl.col(_EVENT_TYPE_COLUMN) == EVENT_TYPE_IDS["like"]

    @property
    def sequence_columns(self) -> list[str]:
        return [self.item_id_column]

    @abstractmethod
    def create_tokenizer(self) -> EventTokenizer: ...

    @property
    def model_dim(self) -> int:
        """Queries and items meet in a dot product, so the tower and the item
        table are one width, named once."""
        return self.transformer.dim

    @property
    def effective_cls_token_mode(self) -> ClsTokenMode:
        return "end_only" if self.cls_token else self.cls_token_mode

    def create_sequence_model(self, tokens_per_event: int) -> ModuleWithDim:
        events_per_sequence = self.max_seq_len + (self.window == "next_item")
        cls_tokens = (
            events_per_sequence
            if self.effective_cls_token_mode == "interleaved"
            else int(self.effective_cls_token_mode == "end_only")
        )
        return build_causal_transformer(
            self.transformer,
            max_seq_len=(
                events_per_sequence * tokens_per_event + self.bos + cls_tokens
            ),
        )

    def _with_bos(self, tokenizer: EventTokenizer) -> EventTokenizer:
        return BosTokenizer(tokenizer) if self.bos else tokenizer


@dataclass
class GenerationExperiment(SampledSoftmaxExperiment, HistoryGenerationExperiment):
    """SASRec over a user's likes: next-item prediction with in-batch negatives."""

    run_name: str = "sasrec_likes"
    num_epochs: int = _NUM_EPOCHS
    eval_ks: tuple[int, ...] = (10, 50, 100)
    eval_max_users: int | None = 20_000
    selection_k: int = 100
    evaluation_catalog: Literal["train", "all"] = "train"
    exclude_seen_from_evaluation: bool = True
    eval_every_n_epochs: int = 1
    restore_best_weights: bool = True
    early_stopping_patience: int | None = None
    early_stopping_min_delta: float = 0.0
    adaptive_schedule_early_stopping: bool = False
    per_layer_item_embeddings: bool = False
    per_layer_item_features: PerLayerItemFeatures = "none"
    per_layer_item_feature_dim: int | None = None
    item_embedding_dim: int | None = None
    timestamp_delta: Literal["plain", "log", "bins"] | None = None
    timestamp_combination: Literal["add", "concat"] = "add"
    timestamp_num_bins: int = 32
    negative_sampling: NegativeSampling = "online_logq"
    logq_correction: Correction = "yi2019"
    correct_positive_logq: bool = True
    mask_false_negatives: bool = True
    exclude_own_group_negatives: bool = True
    dense_random_negative_scores: bool = False
    random_negative_fraction: float = 0.5
    initializer_std: float | None = None

    def create_validation_callback(self) -> ValidationCallback:
        return _PeriodicFinalValidationCallback(
            every_n_epochs=self.eval_every_n_epochs,
            total_epochs=self.num_epochs,
        )

    def create_callbacks(self) -> TrainingCallbacks:
        callbacks = super().create_callbacks()
        # A schedule that anneals over its horizon declares its own length, and
        # patience would stop it on the plateau its decay is meant to leave.
        if self.early_stopping_patience is None or (
            self.lr_schedule.anneals_over_horizon
            and not self.adaptive_schedule_early_stopping
        ):
            return callbacks
        early_stopping = EarlyStopping(
            metric_name=self.checkpointing.best_metric_name,
            metric_prefix=self.checkpointing.best_metric_prefix,
            metric_mode=self.checkpointing.best_metric_mode,
            patience=self.early_stopping_patience,
            min_delta=self.early_stopping_min_delta,
        )
        best_weights_index = callbacks.all.index(callbacks.best_weights)
        callbacks.all.insert(best_weights_index + 1, early_stopping)
        callbacks.early_stopping = early_stopping
        return callbacks

    def prebuild_runner_data(self) -> None:
        super().prebuild_runner_data()
        _ = self.cutoff_query_loader
        _ = self.evaluation_interactions
        if self.negative_sampling in {
            "offline_logq",
            "random_offline_logq",
            "mixed_offline_logq",
        }:
            _ = self._offline_item_probabilities
        _ = self.training_targets_per_epoch
        _ = self.training_tokens_per_epoch

    def _shared_runner_data(
        self, name: str, key: tuple[Any, ...], build: Callable[[], Any]
    ) -> Any:
        ready = os.environ.get("DCN_RUNNER_DATA_READY")
        if not ready:
            return build()
        digest = hashlib.sha1(pickle.dumps(key)).hexdigest()[:12]
        cache_path = Path(ready).with_name(f"runner-data-{name}-{digest}.pickle")
        with hold(cache_path.with_suffix(".lock"), f"runner data {name}"):
            if cache_path.exists():
                started = perf_counter()
                payload = cache_path.read_bytes()
                value = pickle.loads(payload)
                logger.info(
                    "Loaded shared runner data %s (%s bytes) in %.3fs",
                    name,
                    len(payload),
                    perf_counter() - started,
                )
                return value
            value = build()
            payload = pickle.dumps(value)
            temporary = cache_path.with_name(f"{cache_path.name}.{os.getpid()}.tmp")
            temporary.write_bytes(payload)
            temporary.replace(cache_path)
            logger.info("Cached shared runner data %s (%s bytes)", name, len(payload))
            return value

    @cached_property
    def cutoff_query_loader(self) -> DataLoader:
        train_days, _ = self.train_and_validation_days
        return self.make_cutoff_query_loader(train_days)

    @cached_property
    def training_targets_per_epoch(self) -> int:
        return self.training_counts_per_epoch[0]

    @cached_property
    def training_tokens_per_epoch(self) -> int:
        return self.training_counts_per_epoch[1]

    @cached_property
    def training_counts_per_epoch(self) -> tuple[int, int]:
        train_days, _ = self.train_and_validation_days
        key = (
            self.dataset_key,
            tuple(train_days),
            self.max_seq_len,
            self.min_seq_len,
            self.window,
            self.stride,
            str(self.row_filter_for_split("train")),
            self.bos,
            self.effective_cls_token_mode,
            self.item_id_column,
        )
        return self._shared_runner_data("training-counts", key, self._training_counts)

    def _training_counts(self) -> tuple[int, int]:
        dataset = self.sequence_train_loader.dataset
        first_positive = 0 if self.bos else 1
        targets = 0
        tokens = 0
        for index in range(len(dataset)):
            length = len(dataset[index]["int_columns"][self.item_id_column])
            targets += max(0, length - first_positive)
            input_length = length + self.bos
            if self.effective_cls_token_mode == "interleaved":
                tokens += input_length + length
            else:
                tokens += input_length + int(
                    self.effective_cls_token_mode == "end_only" and input_length >= 2
                )
        return targets, tokens

    @cached_property
    def _offline_item_probabilities(self) -> torch.Tensor:
        positive_item_ids = []
        dataset = self.sequence_train_loader.dataset
        first_positive = 0 if self.bos else 1
        for sequence_index in range(len(dataset)):
            item_ids = dataset[sequence_index]["int_columns"][self.item_id_column]
            positive_item_ids.extend(item_ids[first_positive:])
        self.__dict__["training_targets_per_epoch"] = len(positive_item_ids)
        counts = torch.bincount(
            torch.tensor(positive_item_ids, dtype=torch.long),
            minlength=self.catalog_size,
        ).float()
        counts.clamp_(min=1e-12)
        return counts / counts.sum()

    def create_criterion(self) -> nn.Module:
        random_negatives = None
        random_proposal_probabilities = None
        in_batch_negatives = self.num_in_batch_negatives
        offline_probabilities = (
            self._offline_item_probabilities
            if self.negative_sampling
            in {"offline_logq", "random_offline_logq", "mixed_offline_logq"}
            else None
        )
        random_count = 0
        if self.negative_sampling in {"random", "random_offline_logq"}:
            in_batch_negatives = 0
            random_count = self.num_in_batch_negatives
        elif self.negative_sampling in {
            "mixed_online_logq",
            "mixed_offline_logq",
            "mixed_online_global_q",
            "mixed_online_global_q_negative_only",
        }:
            random_count = round(
                self.num_in_batch_negatives * self.random_negative_fraction
            )
            if not 1 <= random_count < self.num_in_batch_negatives:
                raise ValueError(
                    "mixed negative sampling must allocate at least one negative "
                    "to each source"
                )
            in_batch_negatives = self.num_in_batch_negatives - random_count

        if random_count:
            if offline_probabilities is not None:
                random_proposal_probabilities = offline_probabilities.clone()
                random_proposal_probabilities[0] = 0
                random_proposal_probabilities /= random_proposal_probabilities.sum()
            random_negatives = RandomCatalogNegatives(
                catalog_size=self.catalog_size,
                first_item_id=1,
                num_negatives=random_count,
                item_encoder=self.base_model.item_embedding,
                probabilities=(
                    random_proposal_probabilities
                    if self.negative_sampling == "random_offline_logq"
                    else None
                ),
                dense_scores=self.dense_random_negative_scores,
            )

        correction = (
            "none"
            if self.negative_sampling in {"random", "in_batch_no_logq"}
            else self.logq_correction
        )
        aggregate_streaming = self.negative_sampling in {
            "mixed_online_global_q",
            "mixed_online_global_q_negative_only",
        }
        correct_random_negatives = self.negative_sampling not in {
            "mixed_online_logq",
            "mixed_offline_logq",
        }
        if self.negative_sampling in {
            "online_logq",
            "mixed_online_logq",
            "mixed_online_global_q",
            "mixed_online_global_q_negative_only",
        }:
            loss = StreamingInBatchSoftmax(
                hash_size=self.catalog_size,
                num_in_batch_negatives=in_batch_negatives,
                alpha=self.logq_alpha,
                correction=correction,
                random_negatives=random_negatives,
                correct_random_negatives=correct_random_negatives,
                correct_positive=(
                    self.negative_sampling != "mixed_online_global_q_negative_only"
                    if aggregate_streaming
                    else self.correct_positive_logq
                ),
                mask_false_negatives=self.mask_false_negatives,
                exclude_own_group=self.exclude_own_group_negatives,
                uniform_mixture_fraction=(
                    random_count / self.num_in_batch_negatives
                    if aggregate_streaming
                    else (
                        self.random_negative_fraction
                        if self.negative_sampling == "mixed_online_logq"
                        else 0.0
                    )
                ),
                first_item_id=1,
                normalize_streaming_over_valid_ids=aggregate_streaming,
            )
        else:
            probabilities = (
                random_proposal_probabilities * random_count
                if self.negative_sampling == "random_offline_logq"
                else (
                    offline_probabilities
                    if offline_probabilities is not None
                    else torch.full((self.catalog_size,), 1 / self.catalog_size)
                )
            )
            if self.negative_sampling == "mixed_offline_logq":
                uniform_probabilities = torch.zeros_like(probabilities)
                uniform_probabilities[1:] = 1 / (self.catalog_size - 1)
                probabilities = (
                    (1 - self.random_negative_fraction) * probabilities
                    + self.random_negative_fraction * uniform_probabilities
                )
            loss = OfflineInBatchSoftmax(
                q=probabilities,
                num_in_batch_negatives=in_batch_negatives,
                correction=correction,
                random_negatives=random_negatives,
                correct_random_negatives=correct_random_negatives,
                correct_positive=self.correct_positive_logq,
                mask_false_negatives=self.mask_false_negatives,
                exclude_own_group=self.exclude_own_group_negatives,
            )
        return TwoTowerLoss(self.base_model, loss, targets=self.create_targets())

    def settings_defaults(self) -> dict[str, Any]:
        return {
            **super().settings_defaults(),
            "checkpointing": CheckpointConfig(
                best_strategy="best_n",
                best_metric_name=f"{_SELECTION_METRIC}@{self.selection_k}",
                best_metric_mode="max",
                best_metric_prefix=_TRUE_METRIC_PREFIX,
            ),
        }

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.per_layer_item_features not in get_args(PerLayerItemFeatures):
            raise ValueError(
                f"unknown per_layer_item_features {self.per_layer_item_features!r}; "
                f"expected one of {get_args(PerLayerItemFeatures)}"
            )
        if self.per_layer_item_embeddings and self.per_layer_item_features not in {
            "none",
            "direct_add",
        }:
            raise ValueError(
                "per_layer_item_embeddings is the legacy direct_add spelling and "
                "cannot select another item-feature family"
            )
        if self.effective_per_layer_item_features in {
            "concat_residual",
            "gemma_ple",
        }:
            if (
                not isinstance(self.per_layer_item_feature_dim, int)
                or isinstance(self.per_layer_item_feature_dim, bool)
                or self.per_layer_item_feature_dim < 1
            ):
                raise ValueError(
                    f"{self.effective_per_layer_item_features} requires a positive "
                    "per_layer_item_feature_dim"
                )
        elif self.per_layer_item_feature_dim is not None:
            raise ValueError(
                "per_layer_item_feature_dim applies only to concat_residual and "
                "gemma_ple"
            )
        if self.cls_token and self.cls_token_mode == "interleaved":
            raise ValueError(
                "cls_token=True selects end_only and conflicts with interleaved mode"
            )
        if self.cls_token_mode not in {"none", "end_only", "interleaved"}:
            raise ValueError(f"unknown CLS token mode {self.cls_token_mode!r}")
        if self.bos and self.effective_cls_token_mode == "interleaved":
            raise ValueError("interleaved CLS does not support BOS targets")
        if self.effective_cls_token_mode != "none" and self.window != "next_item":
            raise ValueError("CLS queries require next-item sequence windows")
        if self.effective_cls_token_mode != "none" and not isinstance(
            self.create_targets(), NextItemTargets
        ):
            raise ValueError("CLS queries require the next-item target strategy")
        if self.evaluation_catalog not in {"train", "all"}:
            raise ValueError(f"unknown evaluation_catalog {self.evaluation_catalog!r}")
        if self.negative_sampling not in get_args(NegativeSampling):
            raise ValueError(
                f"unknown negative_sampling {self.negative_sampling!r}; "
                f"expected one of {get_args(NegativeSampling)}"
            )
        if self.logq_correction not in get_args(Correction):
            raise ValueError(
                f"unknown logq_correction {self.logq_correction!r}; "
                f"expected one of {get_args(Correction)}"
            )
        if not 0 < self.random_negative_fraction < 1:
            raise ValueError("random_negative_fraction must be in (0, 1)")
        if self.early_stopping_patience is not None and (
            not isinstance(self.early_stopping_patience, int)
            or isinstance(self.early_stopping_patience, bool)
            or self.early_stopping_patience < 1
        ):
            raise ValueError("early_stopping_patience must be a positive integer")
        if (
            not math.isfinite(self.early_stopping_min_delta)
            or self.early_stopping_min_delta < 0
        ):
            raise ValueError("early_stopping_min_delta must be finite and non-negative")
        # The checkpoint rule names its metric by string, and a k the eval never
        # computes would only surface an epoch into the run.
        if self.selection_k not in self.eval_ks:
            raise ValueError(
                f"checkpoints are picked on {_SELECTION_METRIC}@{self.selection_k},"
                f" which eval_ks={self.eval_ks} does not compute"
            )

    @cached_property
    def item_embedding(self) -> nn.Embedding:
        return nn.Embedding(
            self.catalog_size,
            (
                self.model_dim
                if self.item_embedding_dim is None
                else self.item_embedding_dim
            ),
        )

    @property
    def effective_per_layer_item_features(self) -> PerLayerItemFeatures:
        if self.per_layer_item_embeddings:
            return "direct_add"
        return self.per_layer_item_features

    def _create_layer_item_features(
        self,
    ) -> tuple[list[nn.Embedding], list[LayerItemFeatureFusion]]:
        family = self.effective_per_layer_item_features
        if family == "none":
            return [], []
        feature_dim = (
            self.model_dim
            if family == "direct_add"
            else self.per_layer_item_feature_dim
        )
        assert feature_dim is not None
        fusion_factory: Callable[[], LayerItemFeatureFusion]
        if family == "direct_add":
            fusion_factory = lambda: DirectAddItemFeature(self.model_dim)
        elif family == "concat_residual":
            fusion_factory = lambda: ConcatenatedItemFeatureResidual(
                self.model_dim, feature_dim
            )
        else:
            fusion_factory = lambda: self.create_gemma_item_feature_fusion(feature_dim)
        with torch.random.fork_rng(devices=[]):
            embeddings = [
                nn.Embedding(self.catalog_size, feature_dim)
                for _ in range(self.transformer.num_layers)
            ]
            fusions = [fusion_factory() for _ in range(self.transformer.num_layers)]
        for embedding in embeddings:
            embedding.initializer_rng_nonadvancing = True
        return embeddings, fusions

    def create_gemma_item_feature_fusion(
        self, feature_dim: int
    ) -> GemmaItemFeatureResidual:
        return GemmaItemFeatureResidual(self.model_dim, feature_dim)

    def create_input_projection(self) -> nn.Linear | None:
        return (
            None
            if self.item_embedding.embedding_dim == self.model_dim
            else nn.Linear(
                self.item_embedding.embedding_dim, self.model_dim, bias=False
            )
        )

    def create_tokenizer(self) -> EventTokenizer:
        tokenizer: EventTokenizer = ItemTokenizer(
            self.item_embedding,
            item_id_column=self.item_id_column,
            projection=self.create_input_projection(),
        )
        if self.timestamp_delta is not None:
            tokenizer = TimestampDeltaTokenizer(
                tokenizer,
                kind=self.timestamp_delta,
                combination=self.timestamp_combination,
                num_bins=self.timestamp_num_bins,
            )
        return self._with_bos(tokenizer)

    def create_query_projection(self) -> nn.Linear | None:
        if self.item_embedding.embedding_dim == self.model_dim:
            return None
        return nn.Linear(self.model_dim, self.item_embedding.embedding_dim, bias=False)

    def _create_model(self) -> SequenceRetrievalModel:
        tokenizer = self.create_tokenizer()
        sequence_model = self.create_sequence_model(tokenizer.tokens_per_event)
        query_projection = self.create_query_projection()
        layer_item_embeddings, layer_item_feature_fusions = (
            self._create_layer_item_features()
        )
        model = SequenceRetrievalModel(
            tokenizer=tokenizer,
            sequence_model=sequence_model,
            item_embedding=self.item_embedding,
            item_id_column=self.item_id_column,
            query_projection=query_projection,
            cls_token_mode=self.effective_cls_token_mode,
            layer_item_embeddings=layer_item_embeddings,
            layer_item_feature_fusions=layer_item_feature_fusions,
        )
        if self.initializer_std is not None:
            _initialize_standard_parameters(model, self.initializer_std)
        return model

    @property
    def emit_user_column(self) -> bool:
        return True

    def _interactions(
        self, days: list[int], *, split: str = "all"
    ) -> dict[int, set[int]]:
        day_to_path = self.dataset_manager.day_to_path
        return build_interaction_sets(
            [day_to_path[day] for day in days],
            user_column=self.user_column,
            item_id_column=self.item_id_column,
            row_filter=self.row_filter_for_split(split),
        )

    @cached_property
    def evaluation_interactions(
        self,
    ) -> tuple[dict[int, set[int]], dict[int, set[int]], dict[int, set[int]]]:
        train_days, val_days = self.train_and_validation_days
        key = (
            self.dataset_key,
            tuple(train_days),
            tuple(val_days),
            self.evaluation_catalog,
            str(self.row_filter_for_split("train")),
            str(self.row_filter_for_split("validation")),
            str(self.row_filter_for_split("all")),
            self.user_column,
            self.item_id_column,
        )
        return self._shared_runner_data(
            "evaluation-interactions", key, self._evaluation_interactions
        )

    def _evaluation_interactions(
        self,
    ) -> tuple[dict[int, set[int]], dict[int, set[int]], dict[int, set[int]]]:
        train_days, val_days = self.train_and_validation_days
        train_seen = self._interactions(train_days, split="train")
        catalog_interactions = (
            train_seen
            if self.evaluation_catalog == "train"
            else self._interactions(sorted(set(train_days + val_days)))
        )
        relevance = self._interactions(val_days, split="validation")
        return train_seen, catalog_interactions, relevance

    @cached_property
    def true_metric(self) -> TrueMetricCallback:
        train_seen, catalog_interactions, relevance = self.evaluation_interactions

        return TrueMetricCallback(
            model=self.base_model,
            # Ranking the whole embedding table would mix in items training
            # never touched, which keep whatever their initializer wrote.
            item_batch=build_catalog_batch(
                set().union(*catalog_interactions.values()),
                item_id_column=self.item_id_column,
            ),
            query_loader=self.cutoff_query_loader,
            relevance=relevance,
            train_seen=train_seen,
            user_column=self.user_column,
            item_id_column=self.item_id_column,
            ks=self.eval_ks,
            prefix=_TRUE_METRIC_PREFIX,
            max_users=self.eval_max_users,
            seed=self.seed,
            dtype=self.runtime.dtype,
            user_chunk=1024,
            exclude_seen=self.exclude_seen_from_evaluation,
            every_n_epochs=self.eval_every_n_epochs,
        )

    def extra_callbacks(self, train_days: list[int], val_days: list[int]) -> list[Any]:
        return [self.true_metric]

    def finish(self, runner: TrainRunner) -> None:
        super().finish(runner)
        self._report_training_metadata(runner)
        self._report_final_metrics(runner)

    def _report_training_metadata(self, runner: TrainRunner) -> None:
        validation_loss = to_float(
            runner.state.get(EXTRA_METRICS, {}).get("epoch/val", {}).get("loss")
        )
        power_seen_tokens = max(
            (
                int(group.get("power_seen_tokens", 0))
                for group in runner.optimizer.param_groups
            ),
            default=0,
        )
        epochs_trained = runner.current_epoch + 1
        best_epoch_index = self.callbacks.best_weights.best_epoch
        stopped_epoch_index = runner.current_epoch
        best_epoch_at_cap = best_epoch_index == self.num_epochs - 1
        early_stopped = bool(
            self.callbacks.early_stopping is not None
            and self.callbacks.early_stopping.should_stop
        )
        lr_horizon_complete = (
            self.lr_schedule.anneals_over_horizon
            and epochs_trained >= (self.lr_schedule_horizon_epochs or self.num_epochs)
        )
        calibration_status, next_horizon = self._horizon_calibration(
            epochs_trained=epochs_trained,
            early_stopped=early_stopped,
        )
        schedule_steps = getattr(runner, "lr_schedule_total_steps", None)
        steps_per_epoch = getattr(runner, "steps_per_epoch", None)
        lr_schedule = self.callbacks.lr_schedule
        metadata = {
            "training_semantics_revision": GENERATION_TRAINING_SEMANTICS_REVISION,
            "dataset_size": getattr(self, "size", None),
            "seed": self.seed,
            "num_epochs": self.num_epochs,
            "max_epochs": self.num_epochs,
            "epochs_trained": epochs_trained,
            "best_epoch": (None if best_epoch_index is None else best_epoch_index + 1),
            "stopped_epoch": stopped_epoch_index + 1,
            "early_stopped": early_stopped,
            "best_epoch_at_cap": best_epoch_at_cap,
            "lr_horizon_complete": lr_horizon_complete,
            "selection_resolved": (
                best_epoch_index is not None
                and (
                    calibration_status == "calibrated"
                    if self.adaptive_schedule_early_stopping
                    else (
                        lr_horizon_complete
                        or (
                            early_stopped
                            and not best_epoch_at_cap
                            and stopped_epoch_index + 1 < self.num_epochs
                        )
                    )
                )
            ),
            "optimizer_steps_per_epoch": steps_per_epoch,
            "lr_schedule_horizon_steps": (
                schedule_steps if self.lr_schedule.requires_horizon else None
            ),
            "lr_schedule_timescale_steps": (
                None if lr_schedule is None else lr_schedule.resolved_timescale_steps
            ),
            "lr_group_traces": (
                {} if lr_schedule is None else lr_schedule.group_lr_trace
            ),
            "horizon_calibration_status": calibration_status,
            "next_lr_schedule_horizon_epochs": next_horizon,
            **(
                {"lr_schedule_horizon_epochs": (self.lr_schedule_horizon_epochs)}
                if self.lr_schedule.requires_horizon
                else {}
            ),
            "batch_size": self.dataloader.batch_size,
            "physical_batch_size": self.dataloader.batch_size,
            "gradient_accumulation_steps": (
                self.dataloader.gradient_accumulation_steps
            ),
            "effective_batch_size": self.dataloader.effective_batch_size,
            "val_batch_size": self.dataloader.val_batch_size,
            "num_workers": self.dataloader.num_workers,
            "prefetch_factor": self.dataloader.prefetch_factor,
            "model_dim": self.model_dim,
            "item_embedding_dim": self.item_embedding.embedding_dim,
            "embedding_learning_rate": self.embedding_learning_rate,
            "deep_learning_rate": self.deep_learning_rate,
            "weight_decay": self.weight_decay,
            "initializer_std": self.initializer_std,
            "runtime_dtype": str(self.runtime.dtype),
            "runtime_compile": self.runtime.compile,
            "gradient_clip_norm": self.runtime.gradient_clip_norm,
            "negative_sampling": self.negative_sampling,
            "cls_token": self.effective_cls_token_mode != "none",
            "cls_token_mode": self.effective_cls_token_mode,
            "targets_per_epoch": self.training_targets_per_epoch,
            "training_horizon": self.training_targets_per_epoch * epochs_trained,
            "tokens_per_epoch": self.training_tokens_per_epoch,
            "token_horizon": self.training_tokens_per_epoch * epochs_trained,
            "tokens_seen": (
                power_seen_tokens
                if self.lr_schedule.shape == "power"
                else self.training_tokens_per_epoch * epochs_trained
            ),
            "optimizer_steps": runner.global_step,
            "validation_loss": validation_loss,
            "transfer_invariants": {
                "experiment_class": type(self).__name__,
                "mup_base_dim": getattr(self, "mup_base_dim", None),
                "mup_delta_dim": getattr(self, "mup_delta_dim", None),
                "mup_base_ffn_dim": getattr(self, "mup_base_ffn_dim", None),
                "mup_delta_ffn_dim": getattr(self, "mup_delta_ffn_dim", None),
                "dataset_size": getattr(self, "size", None),
                "user_sample": (
                    None
                    if getattr(self, "user_sample", None) is None
                    else self.user_sample.name
                ),
                "event_type_filter": getattr(self, "event_type_filter", None),
                "min_item_interactions_per_item": getattr(
                    self, "min_item_interactions_per_item", None
                ),
                "drop_unmapped_items": getattr(self, "drop_unmapped_items", None),
                "validation_interval_seconds": self.validation_interval_seconds,
                "day_range": (
                    None if self.day_range is None else asdict(self.day_range)
                ),
                "batch_size": self.dataloader.batch_size,
                "physical_batch_size": self.dataloader.batch_size,
                "gradient_accumulation_steps": (
                    self.dataloader.gradient_accumulation_steps
                ),
                "effective_batch_size": self.dataloader.effective_batch_size,
                "model_dim": self.model_dim,
                "item_embedding_dim": self.item_embedding.embedding_dim,
                "max_seq_len": self.max_seq_len,
                "window": self.window,
                "bos": self.bos,
                "cls_token": self.effective_cls_token_mode != "none",
                "cls_token_mode": self.effective_cls_token_mode,
                "timestamp_delta": self.timestamp_delta,
                "timestamp_combination": self.timestamp_combination,
                "timestamp_num_bins": self.timestamp_num_bins,
                **(
                    {
                        "timestamp_bin_semantics_revision": (
                            TIMESTAMP_BIN_SEMANTICS_REVISION
                        )
                    }
                    if self.timestamp_delta == "bins"
                    else {}
                ),
                "per_layer_item_embeddings": self.per_layer_item_embeddings,
                "per_layer_item_features": self.effective_per_layer_item_features,
                "per_layer_item_feature_dim": self.per_layer_item_feature_dim,
                "negative_sampling": self.negative_sampling,
                **(
                    {
                        "negative_sampling_semantics_revision": (
                            NEGATIVE_SAMPLING_SEMANTICS_REVISION
                        )
                    }
                    if self.negative_sampling
                    in {
                        "online_logq",
                        "mixed_online_logq",
                        "mixed_offline_logq",
                        "mixed_online_global_q",
                        "mixed_online_global_q_negative_only",
                    }
                    else {}
                ),
                "num_in_batch_negatives": self.num_in_batch_negatives,
                "logq_correction": self.logq_correction,
                "random_negative_fraction": self.random_negative_fraction,
                "logq_alpha": self.logq_alpha,
                "correct_positive_logq": self.correct_positive_logq,
                "mask_false_negatives": self.mask_false_negatives,
                "exclude_own_group_negatives": self.exclude_own_group_negatives,
                "dense_random_negative_scores": self.dense_random_negative_scores,
                "eval_ks": self.eval_ks,
                "eval_max_users": self.eval_max_users,
                "eval_every_n_epochs": self.eval_every_n_epochs,
                "early_stopping_patience": self.early_stopping_patience,
                "early_stopping_min_delta": self.early_stopping_min_delta,
                "early_stopping_metric": self.checkpointing.best_metric_name,
                "early_stopping_metric_prefix": (self.checkpointing.best_metric_prefix),
                "selection_k": self.selection_k,
                "evaluation_catalog": self.evaluation_catalog,
                "exclude_seen_from_evaluation": self.exclude_seen_from_evaluation,
                "restore_best_weights": self.restore_best_weights,
                "adaptive_schedule_early_stopping": (
                    self.adaptive_schedule_early_stopping
                ),
                **(
                    {"lr_schedule_horizon_epochs": (self.lr_schedule_horizon_epochs)}
                    if self.lr_schedule.requires_horizon
                    else {}
                ),
                "transformer": transformer_metadata(self.transformer),
                "lr_schedule": asdict(self.lr_schedule),
            },
        }
        if self.timestamp_delta == "bins":
            metadata["transfer_invariants"][
                "timestamp_bin_semantics_revision"
            ] = TIMESTAMP_BIN_SEMANTICS_REVISION
        destination = global_config.logs_path / self.run_name / "training_metadata.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(metadata, indent=2, sort_keys=True))

    def _horizon_calibration(
        self, *, epochs_trained: int, early_stopped: bool
    ) -> tuple[str | None, int | None]:
        if not self.adaptive_schedule_early_stopping:
            return None, None
        if self.lr_schedule.shape == "constant":
            if early_stopped and epochs_trained < self.num_epochs:
                return "calibrated", None
            return "extend_cap", math.ceil(1.5 * self.num_epochs)
        horizon = self.lr_schedule_horizon_epochs
        if horizon is None:
            raise ValueError("adaptive schedules require a reference horizon")
        tolerance = max(3, round(0.1 * horizon))
        if self.lr_schedule.anneals_over_horizon:
            if early_stopped:
                if 0 <= horizon - epochs_trained <= tolerance:
                    return "calibrated", None
                return "shorten_horizon", epochs_trained
            return "extend_horizon", math.ceil(1.5 * horizon)
        if not early_stopped:
            return "extend_cap", math.ceil(1.5 * self.num_epochs)
        if abs(horizon - epochs_trained) <= tolerance:
            return "calibrated", None
        return "recalibrate_horizon", epochs_trained

    def _report_final_metrics(self, runner: TrainRunner) -> None:
        """What the experiment is judged on: the checkpoint that scored best,
        against every evaluable user rather than the tuning sample."""
        destination = global_config.logs_path / self.run_name / "final_metrics.json"
        destination.unlink(missing_ok=True)
        if self.restore_best_weights and not self.callbacks.best_weights.restore(
            runner.model
        ):
            logger.warning("No epoch scored; reporting the weights on hand")

        metrics = self.true_metric.score(max_users=None)
        if metrics is None:
            logger.warning("No user could be scored; skipping the final report")
            return

        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(metrics, indent=2, sort_keys=True))
        # Spelled out as well as filed: the run's wandb session logs per epoch
        # and is over by the time this is measured.
        logger.info("Final metrics (%s) -> %s", metrics, destination)


@dataclass
class MuTransferGenerationExperiment(GenerationExperiment):
    """Widthwise hyperparameter transfer using https://github.com/microsoft/mup."""

    mup_base_dim: int = 16
    mup_delta_dim: int = 32
    mup_base_ffn_dim: int | None = None
    mup_delta_ffn_dim: int | None = None
    item_embedding_dim: int | None = 64

    def __post_init__(self) -> None:
        super().__post_init__()
        for dim in (self.mup_base_dim, self.mup_delta_dim, self.transformer.dim):
            if dim % self.transformer.nhead:
                raise ValueError(
                    f"μP width {dim} must be divisible by {self.transformer.nhead} heads"
                )
        if (self.mup_base_ffn_dim is None) != (self.mup_delta_ffn_dim is None):
            raise ValueError(
                "mup_base_ffn_dim and mup_delta_ffn_dim are set together: an FFN "
                "width is transferable only if the base and delta models disagree "
                "about it"
            )
        if (
            self.mup_base_ffn_dim is not None
            and self.mup_base_ffn_dim == self.mup_delta_ffn_dim
        ):
            raise ValueError("mup_base_ffn_dim and mup_delta_ffn_dim must differ")

    def _ffn_dim_at_width(self, dim: int) -> int:
        ffn_ratio = self.transformer.ffn_intermediate_dim / self.transformer.dim
        return round(ffn_ratio * dim)

    def _model_at_width(
        self, dim: int, ffn_intermediate_dim: int | None = None
    ) -> SequenceRetrievalModel:
        narrower = replace(
            self,
            initializer_std=None,
            transformer=replace(
                self.transformer,
                dim=dim,
                ffn_intermediate_dim=(
                    ffn_intermediate_dim
                    if ffn_intermediate_dim is not None
                    else self._ffn_dim_at_width(dim)
                ),
            ),
        )
        narrower.__dict__["artifacts"] = self.artifacts
        narrower.__dict__["item_embeddings"] = self.item_embeddings
        return narrower._create_model()

    def create_input_projection(self) -> nn.Linear:
        return nn.Linear(self.item_embedding.embedding_dim, self.model_dim, bias=False)

    def create_query_projection(self) -> nn.Linear:
        return mup.MuReadout(
            self.model_dim,
            self.item_embedding.embedding_dim,
            bias=False,
            readout_zero_init=True,
        )

    def create_gemma_item_feature_fusion(
        self, feature_dim: int
    ) -> GemmaItemFeatureResidual:
        return GemmaItemFeatureResidual(
            self.model_dim,
            feature_dim,
            finite_readout_factory=lambda input_dim, output_dim: mup.MuReadout(
                input_dim, output_dim, bias=False
            ),
        )

    @cached_property
    def base_model(self) -> SequenceRetrievalModel:
        model = self._model_at_width(
            self.transformer.dim, self.transformer.ffn_intermediate_dim
        ).to(self.runner_build_device)
        base = self._model_at_width(self.mup_base_dim, self.mup_base_ffn_dim)
        delta = self._model_at_width(self.mup_delta_dim, self.mup_delta_ffn_dim)
        mup.set_base_shapes(model, base, delta=delta)
        if self.initializer_std is not None:
            _initialize_mup_parameters(model, self.initializer_std)
        head_dim = self.transformer.dim // self.transformer.nhead
        base_head_dim = self.mup_base_dim // self.transformer.nhead
        for layer in model.sequence_model.layers:
            layer.softmax_scale = base_head_dim**0.5 / head_dim
        logger.info("Model architecture:\n%s", model)
        logger.info(
            "Total parameters: %s",
            f"{sum(parameter.numel() for parameter in model.parameters()):,}",
        )
        return model

    def create_optimizers(self) -> torch.optim.Optimizer:
        embedding_params, deep_params = self.split_parameters(
            self.base_model, self.embedding_types
        )
        groups = [
            {
                "params": embedding_params,
                "lr": self.embedding_learning_rate,
                "schedule_group": "embedding",
            },
            {
                "params": deep_params,
                "lr": self.deep_learning_rate,
                "schedule_group": "deep",
            },
        ]
        optimizer = mup.MuAdam(
            [group for group in groups if group["params"]],
            weight_decay=self.weight_decay,
            fused=self.runner_build_device.type == "cuda",
        )
        return register_stable_optimizer_groups(optimizer)


@dataclass
class TimeWindowGenerationExperiment(GenerationExperiment):
    """Anything liked within a day of a token counts as that token's positive."""

    run_name: str = "sasrec_likes_24h"
    window_seconds: float = _SECONDS_IN_DAY
    lookahead: int = 32

    def create_targets(self) -> SequenceTargets:
        return TimeWindowTargets(
            window_seconds=self.window_seconds, lookahead=self.lookahead
        )


@dataclass
class ActionGenerationExperiment(GenerationExperiment):
    """Likes and listens both, with the action as a token of its own."""

    run_name: str = "sasrec_actions"

    def settings_defaults(self) -> dict[str, Any]:
        # Two tokens an event, so half as many sequences fit the activation budget.
        return {
            **super().settings_defaults(),
            "dataloader": DataloaderConfig(
                batch_size=64, val_batch_size=64, num_workers=8, prefetch_factor=4
            ),
        }

    @property
    def row_filter(self) -> pl.Expr | None:
        return pl.col(_EVENT_TYPE_COLUMN).is_in(
            [EVENT_TYPE_IDS["like"], EVENT_TYPE_IDS["listen"]]
        )

    @property
    def sequence_columns(self) -> list[str]:
        return [self.item_id_column, _EVENT_TYPE_COLUMN]

    def create_tokenizer(self) -> EventTokenizer:
        return ActionTokenizer(
            self.item_embedding,
            action_embedding=nn.Embedding(len(EVENT_TYPE_IDS), self.model_dim),
            item_id_column=self.item_id_column,
            action_column=_EVENT_TYPE_COLUMN,
        )


@dataclass
class SemanticGenerationExperiment(SemanticExperiment, HistoryGenerationExperiment):
    """G1 with the catalog replaced by code tuples."""

    run_name: str = "sasrec_semantic"
    num_epochs: int = _NUM_EPOCHS
    beam_width: int = 10

    @property
    def generated_levels(self) -> int:
        """Levels beam search decodes, leaving the collision suffix alone."""
        return self.semantic.num_levels

    def create_semantic_embedding(
        self,
    ) -> SemanticIdEmbedding | CombinedSemanticIdEmbedding:
        return SemanticIdEmbedding.learned(
            self.semantic_codes, num_items=self.num_items, embedding_dim=self.model_dim
        )

    def create_tokenizer(self) -> SemanticIdTokenizer:
        embedding = self.create_semantic_embedding()
        projection = (
            None
            if embedding.level_dim == self.model_dim
            else nn.Linear(embedding.level_dim, self.model_dim)
        )
        return SemanticIdTokenizer(
            embedding, item_id_column=self.item_id_column, projection=projection
        )

    @cached_property
    def code_constraint(self) -> SemanticIdConstraint:
        return SemanticIdConstraint(self.semantic_codes).to(self.runner_build_device)

    def _create_model(self) -> TokenDecoder:
        tokenizer = self.create_tokenizer()
        return CausalTokenDecoder(
            tokenizer=tokenizer,
            sequence_model=self.create_sequence_model(tokenizer.tokens_per_event),
            constraint=self.code_constraint,
        )

    def create_criterion(self) -> nn.Module:
        return TokenPredictionLoss(self.base_model)

    def extra_callbacks(self, train_days: list[int], val_days: list[int]) -> list[Any]:
        return [
            GenerationRecallCallback(
                model=self.base_model,
                # Shuffled: beam search reaches only the first few batches,
                # and a bucket-major loader would hand it the same users again.
                loader=self.make_sequence_loader(
                    val_days,
                    split="val",
                    batch_size=self.dataloader.val_batch_size,
                    shuffle=True,
                ),
                constraint=self.code_constraint,
                item_id_column=self.item_id_column,
                num_levels=self.generated_levels,
                beam_width=self.beam_width,
                dtype=self.runtime.dtype,
            )
        ]


@dataclass
class TigerExperiment(SemanticGenerationExperiment):
    """Encoder-decoder over semantic ids: the history is read, the next item written."""

    run_name: str = "tiger_semantic"

    def _create_model(self) -> TokenDecoder:
        tokenizer = self.create_tokenizer()
        return Seq2SeqTokenDecoder(
            tokenizer=tokenizer,
            encoder=self.create_sequence_model(tokenizer.tokens_per_event),
            decoder=build_transformer_decoder(
                self.transformer, max_seq_len=tokenizer.tokens_per_event
            ),
            constraint=self.code_constraint,
        )


@dataclass
class CombinedSemanticGenerationExperiment(SemanticGenerationExperiment):
    """G4's stack reading each code twice: as a trainable row, and as the
    centroid it stands for."""

    run_name: str = "sasrec_semantic_combined"

    def create_semantic_embedding(self) -> CombinedSemanticIdEmbedding:
        return CombinedSemanticIdEmbedding(
            [
                SemanticIdEmbedding.learned(
                    self.semantic_codes,
                    num_items=self.num_items,
                    embedding_dim=self.model_dim,
                ),
                SemanticIdEmbedding.from_codebooks(
                    self.semantic_codes,
                    self.semantic_codebooks,
                    num_items=self.num_items,
                ),
            ]
        )


@dataclass
class RqVaeGenerationExperiment(SemanticGenerationExperiment):
    """G4's stack over codes an RQ-VAE learned, rather than ones k-means assigned."""

    run_name: str = "sasrec_semantic_rqvae"
    semantic: SemanticIdConfig = field(
        default_factory=lambda: SemanticIdConfig(quantizer="rqvae")
    )
