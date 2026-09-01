from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import cached_property
import hashlib
import logging
import math
import os
from pathlib import Path
from typing import Literal, cast

import torch
from torch import nn

from dcn.config import (
    GenerationExperiment,
    MuTransferGenerationExperiment,
    SemanticHistoryExperiment,
    SemanticIdConfig,
)
from dcn.config.experiment import Experiment
from dcn.config.semantic import CollisionPolicy, SemanticExperiment
from dcn.config.semantic_history import SemanticHistoryRepresentation
from dcn.config.sequence import SequenceExperiment
from dcn.config.settings import DataloaderConfig, LrScheduleConfig, RuntimeConfig
from dcn.models import EventTokenizer, SequenceRetrievalModel
from dcn.nn.semantic_embedding import (
    CombinedSemanticIdEmbedding,
    SemanticIdEmbedding,
)
from experiments.g6_rqkmeans_history.native500m.protocol.design import (
    BATCH_SIZE,
    FIXED_HORIZON,
    KMEANS_MAX_ITERATIONS,
    KMEANS_TOLERANCE,
    REPRESENTATION_WIDTH,
    REPRESENTATIONS,
    SHARED_CODEBOOK_SIZES,
    TOKENIZER_LEVELS,
)
from experiments.g6_rqkmeans_history.protocol.collision_policy import (
    validate_collision_symbol_cap,
)
from experiments.generation_protocol import generation_protocol
from neuralrec.run.train import TrainRunner
from utils.global_config import config as global_config


logger = logging.getLogger(__name__)

Backbone = Literal["original_g1", "best_g1"]
SidLookupInitialization = Literal["random", "content_pca"]

APPLICABLE_INITIALIZATION_REPRESENTATIONS = (
    "learned_sid_event",
    "item_learned_frozen_sid_event",
    "learned_sid_tokens",
    "learned_frozen_sid_tokens",
    "interleaved_item_sid_tokens",
)

_CONTENT_PROJECTION = "per_level_centered_pca_v1"
_DATA_PROTOCOL = generation_protocol(
    event_type_filter="like",
    window="next_item",
    size="500m",
    user_sample=None,
)

_ORIGINAL_TRANSFORMER = replace(
    GenerationExperiment.transformer,
    dim=64,
    num_layers=2,
    nhead=2,
    num_kv_heads=2,
    ffn_intermediate_dim=256,
    ffn="gelu",
    norm="layer",
    norm_place="pre",
    input_norm=None,
    final_norm="layer",
    alibi=False,
    rope=None,
    learned_positions="forward",
    learned_position_fusion="add",
    learned_position_fusion_normalization=None,
    learned_position_fusion_residual=None,
    learned_position_initialization="default",
    learned_position_reverse_correction=None,
    learned_position_reverse_max_scale=0.1,
    learned_position_reverse_initializer_rng_nonadvancing=False,
    attention_window=None,
    input_dropout=0.1,
    ffn_dropout=0.1,
    gated_ffn_dropout=False,
)
_BEST_TRANSFORMER = replace(
    _ORIGINAL_TRANSFORMER,
    num_layers=4,
    num_kv_heads=1,
    ffn_intermediate_dim=192,
    ffn="swiglu",
    gated_ffn_dropout=True,
    ffn_dropout=0.1,
    norm_place="post",
    input_norm="rms",
    final_norm="rms",
    alibi=True,
    rope="timestamp_reverse",
    learned_positions=("forward", "reverse"),
    learned_position_fusion="concat",
    learned_position_fusion_residual="rezero",
    learned_position_reverse_correction="bounded_tanh",
    learned_position_reverse_max_scale=0.025,
    learned_position_reverse_initializer_rng_nonadvancing=True,
)


def _positive_rate(name: str, value: float) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive finite")
    return value


def _common(
    backbone: Backbone,
    *,
    embedding_learning_rate: float,
    deep_learning_rate: float,
    run_name: str,
    seed: int,
) -> dict[str, object]:
    if backbone not in {"original_g1", "best_g1"}:
        raise ValueError(f"unknown backbone {backbone!r}")
    best = backbone == "best_g1"
    return {
        "run_name": run_name,
        "seed": seed,
        **_DATA_PROTOCOL,
        "dataloader": DataloaderConfig(
            batch_size=BATCH_SIZE,
            val_batch_size=8192,
            num_workers=4,
            prefetch_factor=4,
            gradient_accumulation_steps=1,
        ),
        "num_epochs": FIXED_HORIZON,
        "lr_schedule_horizon_epochs": FIXED_HORIZON,
        "eval_every_n_epochs": 1,
        "eval_max_users": None,
        "restore_best_weights": True,
        "early_stopping_patience": None,
        "early_stopping_min_delta": 0.0,
        "adaptive_schedule_early_stopping": False,
        "transformer": _BEST_TRANSFORMER if best else _ORIGINAL_TRANSFORMER,
        "max_seq_len": 100,
        "bos": best,
        "cls_token": False,
        "cls_token_mode": "end_only" if best else "none",
        "lr_schedule": (
            LrScheduleConfig(
                "cosine",
                warmup_fraction=0.05,
                cycles=1,
                optimizer_group_scope="deep_only",
            )
            if best
            else LrScheduleConfig()
        ),
        "timestamp_delta": "bins" if best else None,
        "timestamp_combination": "add",
        "timestamp_num_bins": 32,
        "negative_sampling": "random_offline_logq" if best else "offline_logq",
        "num_in_batch_negatives": 2048 if best else 512,
        "logq_correction": "yi2019" if best else "baseline",
        "correct_positive_logq": best,
        "mask_false_negatives": False,
        "exclude_own_group_negatives": False,
        "dense_random_negative_scores": best,
        "random_negative_fraction": 0.5,
        "initializer_std": 0.02,
        "item_embedding_dim": 64 if best else None,
        "embedding_learning_rate": _positive_rate(
            "embedding_learning_rate", embedding_learning_rate
        ),
        "deep_learning_rate": _positive_rate(
            "deep_learning_rate", deep_learning_rate
        ),
        "weight_decay": 0.0,
        "runtime": RuntimeConfig(
            dtype=torch.bfloat16,
            compile=False,
            gradient_clip_norm=None,
        ),
        "final_ranking_evidence_group": None,
    }


class _WinnerDiagnosticsDeferred:
    def finish(self, runner: TrainRunner) -> None:
        Experiment.finish(self, runner)
        if not self.callbacks.best_weights.restore(runner.model):
            raise RuntimeError("validation-selected best weights are unavailable")
        artifact = self._persist_best_model_artifact()
        self._best_model_artifact = artifact
        self._write_training_metadata_atomically(runner)

    def generation_architecture_metadata(self) -> dict[str, object]:
        return self._with_best_model_artifact(
            super().generation_architecture_metadata()
        )

    def _with_best_model_artifact(
        self, metadata: dict[str, object]
    ) -> dict[str, object]:
        artifact = getattr(self, "_best_model_artifact", None)
        if artifact is not None:
            metadata["best_model_artifact"] = artifact
        return metadata

    def _persist_best_model_artifact(self) -> dict[str, str]:
        destination = (
            global_config.logs_path / self.run_name / "best_model_state.pt"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp")
        temporary.unlink(missing_ok=True)
        state = {
            name: tensor.detach().cpu()
            for name, tensor in self.base_model.state_dict().items()
        }
        torch.save(state, temporary)
        with temporary.open("rb") as file:
            os.fsync(file.fileno())
        digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
        if destination.exists():
            existing = hashlib.sha256(destination.read_bytes()).hexdigest()
            temporary.unlink()
            if existing != digest:
                raise RuntimeError("immutable best-model artifact changed")
            digest = existing
        else:
            temporary.replace(destination)
        destination.chmod(0o444)
        with destination.open("rb") as file:
            os.fsync(file.fileno())
        self._fsync_directory(destination.parent)
        return {
            "schema": "g6-best-model-state/v1",
            "path": destination.relative_to(global_config.base_path).as_posix(),
            "sha256": digest,
        }

    def _write_training_metadata_atomically(self, runner: TrainRunner) -> None:
        run_name = self.run_name
        staging_run_name = f"{run_name}/.training-metadata-stage"
        self.run_name = staging_run_name
        try:
            self._report_training_metadata(runner)
        finally:
            self.run_name = run_name
        staged = (
            global_config.logs_path
            / staging_run_name
            / "training_metadata.json"
        )
        destination = (
            global_config.logs_path / self.run_name / "training_metadata.json"
        )
        if not staged.is_file():
            raise RuntimeError("training metadata staging did not produce an artifact")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp")
        temporary.unlink(missing_ok=True)
        with temporary.open("wb") as file:
            file.write(staged.read_bytes())
            file.flush()
            os.fsync(file.fileno())
        temporary.replace(destination)
        self._fsync_directory(destination.parent)
        staged.unlink()
        staged.parent.rmdir()

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


class Native500MGenerationExperiment(_WinnerDiagnosticsDeferred, GenerationExperiment):
    pass


class Native500MMuTransferGenerationExperiment(
    _WinnerDiagnosticsDeferred, MuTransferGenerationExperiment
):
    pass


def _tensor_sha256(tensor: torch.Tensor) -> str:
    contiguous = tensor.detach().cpu().contiguous()
    payload = contiguous.view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def project_centroids_with_pca(
    centroids: torch.Tensor, output_dim: int
) -> torch.Tensor:
    if output_dim > centroids.shape[1]:
        raise ValueError(
            f"cannot project {centroids.shape[1]}-dimensional centroids to "
            f"{output_dim} dimensions"
        )
    if output_dim > min(centroids.shape):
        raise ValueError("not enough centroids for the requested PCA width")
    centered = centroids.detach().cpu().to(torch.float64)
    centered -= centered.mean(dim=0, keepdim=True)
    _, _, right_vectors = torch.linalg.svd(centered, full_matrices=False)
    components = right_vectors[:output_dim].clone()
    pivots = components.abs().argmax(dim=1)
    signs = components[torch.arange(output_dim), pivots].sign()
    signs[signs == 0] = 1
    components *= signs.unsqueeze(1)
    return centered @ components.T


def _match_rms(projected: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    target = reference.detach().cpu().to(torch.float64).square().mean().sqrt()
    current = projected.square().mean().sqrt()
    if not math.isfinite(float(current)) or current <= 0:
        raise ValueError("projected centroids have zero or non-finite RMS")
    return projected * (target / current)


def learned_sid_lookup(model: nn.Module) -> SemanticIdEmbedding:
    lookups = [
        module
        for module in model.modules()
        if isinstance(module, SemanticIdEmbedding)
        and isinstance(module.embedding, nn.Embedding)
        and module.embedding.weight.requires_grad
    ]
    if len(lookups) != 1:
        raise RuntimeError("requires exactly one trainable base SID lookup")
    return lookups[0]


def _validate_initialization(
    representation: SemanticHistoryRepresentation,
    initialization: SidLookupInitialization,
) -> None:
    if initialization not in {"random", "content_pca"}:
        raise ValueError("unknown SID lookup initialization")
    if (
        initialization == "content_pca"
        and representation not in APPLICABLE_INITIALIZATION_REPRESENTATIONS
    ):
        raise ValueError("content PCA requires a trainable SID lookup")


def _apply_sid_lookup_initialization(
    experiment: Native500MSemanticHistoryExperiment
    | ConventionalSemanticHistoryExperiment,
    model: nn.Module,
) -> dict[str, object] | None:
    if experiment.history_representation not in APPLICABLE_INITIALIZATION_REPRESENTATIONS:
        return None
    rng_before = torch.get_rng_state()
    learned = learned_sid_lookup(model)
    weight = learned.embedding.weight
    vocabulary = experiment.semantic_codes.vocabulary
    base_ranges = [
        vocabulary.level_range(level)
        for level in range(experiment.semantic_codebooks.num_levels)
    ]
    before = torch.cat(
        [weight[first:last].detach().cpu() for first, last in base_ranges]
    )
    level_diagnostics: list[dict[str, int | float]] = []
    if experiment.sid_lookup_initialization == "content_pca":
        with torch.no_grad():
            for level, (first, last) in enumerate(base_ranges):
                reference = weight[first:last]
                projected = project_centroids_with_pca(
                    experiment.semantic_codebooks.centroids[level], weight.shape[1]
                )
                initialized = _match_rms(projected, reference).to(
                    device=weight.device, dtype=weight.dtype
                )
                initialized = _match_rms(initialized, reference).to(
                    device=weight.device, dtype=weight.dtype
                )
                weight[first:last].copy_(initialized)
                level_diagnostics.append(
                    {
                        "level": level,
                        "random_rms": float(
                            reference.detach().float().square().mean().sqrt()
                        ),
                        "initialized_rms": float(
                            initialized.detach().float().square().mean().sqrt()
                        ),
                    }
                )
    else:
        level_diagnostics = [
            {
                "level": level,
                "random_rms": float(
                    weight[first:last].detach().float().square().mean().sqrt()
                ),
                "initialized_rms": float(
                    weight[first:last].detach().float().square().mean().sqrt()
                ),
            }
            for level, (first, last) in enumerate(base_ranges)
        ]
    base_mask = torch.zeros(len(weight), dtype=torch.bool, device=weight.device)
    for first, last in base_ranges:
        base_mask[first:last] = True
    after = torch.cat(
        [weight[first:last].detach().cpu() for first, last in base_ranges]
    )
    return {
        "version": 1,
        "mode": experiment.sid_lookup_initialization,
        "projection": (
            _CONTENT_PROJECTION
            if experiment.sid_lookup_initialization == "content_pca"
            else None
        ),
        "base_rows_before_sha256": _tensor_sha256(before),
        "base_rows_after_sha256": _tensor_sha256(after),
        "non_base_rows_sha256": _tensor_sha256(weight[~base_mask]),
        "codebook_centroids_sha256": _tensor_sha256(
            experiment.semantic_codebooks.centroids
        ),
        "rng_nonadvancing": torch.equal(rng_before, torch.get_rng_state()),
        "levels": level_diagnostics,
    }


def _validate_kmeans_convergence(experiment: SemanticExperiment) -> None:
    convergence = experiment.semantic_stage.convergence_diagnostics()
    if convergence is None:
        raise RuntimeError("native-500M RQ-KMeans convergence evidence is missing")
    if not isinstance(convergence, tuple) or len(convergence) != 2:
        raise RuntimeError("native-500M RQ-KMeans convergence evidence is malformed")
    document, _ = convergence
    if not isinstance(document, dict):
        raise RuntimeError("native-500M RQ-KMeans convergence evidence is malformed")
    levels = document.get("levels")
    if not isinstance(levels, list):
        raise RuntimeError("native-500M RQ-KMeans convergence levels are missing")
    records: list[tuple[int, str]] = []
    for record in levels:
        if not isinstance(record, dict):
            raise RuntimeError("RQ-KMeans convergence records must be well-formed")
        level = record.get("level")
        stop_reason = record.get("stop_reason")
        if (
            not isinstance(level, int)
            or isinstance(level, bool)
            or not isinstance(stop_reason, str)
        ):
            raise RuntimeError("RQ-KMeans convergence records must be well-formed")
        records.append((level, stop_reason))
    expected_levels = set(range(experiment.semantic.num_levels))
    actual_levels = [level for level, _ in records]
    if len(records) != len(expected_levels) or set(actual_levels) != expected_levels:
        raise RuntimeError(
            "RQ-KMeans convergence must contain exactly "
            f"{experiment.semantic.num_levels} unique expected levels"
        )
    cap_hits = [level for level, reason in records if reason == "max_iterations"]
    if cap_hits:
        raise RuntimeError(f"RQ-KMeans reached the iteration cap at levels {cap_hits}")
    successful_reasons = {"assignments_stable", "relative_inertia"}
    unknown = sorted({reason for _, reason in records} - successful_reasons)
    if unknown:
        raise RuntimeError(f"RQ-KMeans convergence has unknown stop reason {unknown}")


@dataclass
class Native500MSemanticHistoryExperiment(
    _WinnerDiagnosticsDeferred, SemanticHistoryExperiment
):
    sid_lookup_initialization: SidLookupInitialization = "random"
    _sid_initialization_diagnostics: dict[str, object] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        _validate_initialization(
            self.history_representation, self.sid_lookup_initialization
        )

    @property
    def sid_initialization_diagnostics(self) -> dict[str, object]:
        if self._sid_initialization_diagnostics is None:
            raise RuntimeError("SID initialization has not run or is inapplicable")
        return self._sid_initialization_diagnostics

    def apply_post_mup_initialization(self, model: nn.Module) -> None:
        self._sid_initialization_diagnostics = _apply_sid_lookup_initialization(
            self, model
        )

    def create_runner(self) -> TrainRunner:
        _validate_kmeans_convergence(self)
        validate_collision_symbol_cap(
            self.semantic_codes,
            policy=self.semantic.collision_policy,
            base_levels=self.semantic.num_levels,
            require_suffix_feasibility=True,
        )
        return super().create_runner()

    def true_metric_options(self) -> dict[str, object]:
        return GenerationExperiment.true_metric_options(self)

    def generation_architecture_metadata(self) -> dict[str, object]:
        metadata = super().generation_architecture_metadata()
        metadata |= {
            "history_representation": self.history_representation,
            "representation_width": self.representation_width,
            "sid_lookup_initialization": self.sid_lookup_initialization,
            "sid_lookup_projection": (
                _CONTENT_PROJECTION
                if self.sid_lookup_initialization == "content_pca"
                else None
            ),
            "g6_dataset_lineage": "native500m-v1",
            "g6_winner_diagnostics_deferred": True,
        }
        if self._sid_initialization_diagnostics is not None:
            metadata["sid_initialization_diagnostics"] = (
                self.sid_initialization_diagnostics
            )
        return self._with_best_model_artifact(metadata)


@dataclass
class ConventionalSemanticHistoryExperiment(
    _WinnerDiagnosticsDeferred, SemanticExperiment, GenerationExperiment
):
    history_representation: SemanticHistoryRepresentation = "learned_sid_event"
    representation_width: int = 128
    sid_lookup_initialization: SidLookupInitialization = "random"
    _sid_initialization_diagnostics: dict[str, object] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.history_representation not in REPRESENTATIONS:
            raise ValueError(
                f"unknown semantic history representation {self.history_representation!r}"
            )
        if self.representation_width != REPRESENTATION_WIDTH:
            raise ValueError(
                f"representation_width must be {REPRESENTATION_WIDTH}"
            )
        _validate_initialization(
            self.history_representation, self.sid_lookup_initialization
        )

    @property
    def history_tokens_per_event(self) -> int:
        if self.history_representation.endswith("_event"):
            return 1
        semantic_levels = self.semantic.num_levels + int(
            self.semantic.collision_policy == "suffix"
        )
        return semantic_levels + int(
            self.history_representation == "interleaved_item_sid_tokens"
        )

    def training_count_architecture_invariants(self) -> tuple[object, ...]:
        return (
            *super().training_count_architecture_invariants(),
            self.history_representation,
            self.semantic.cache_key,
        )

    def _learned_embedding(self) -> SemanticIdEmbedding:
        return SemanticIdEmbedding.learned(
            self.semantic_codes,
            num_items=self.num_items,
            embedding_dim=self.representation_width,
        )

    def _frozen_embedding(self) -> SemanticIdEmbedding:
        return SemanticIdEmbedding.from_codebooks(
            self.semantic_codes,
            self.semantic_codebooks,
            num_items=self.num_items,
            train_collision_suffix=self.semantic.collision_policy == "suffix",
        )

    def _combined_embedding(self) -> CombinedSemanticIdEmbedding:
        return CombinedSemanticIdEmbedding(
            [self._learned_embedding(), self._frozen_embedding()]
        )

    def create_tokenizer(self) -> EventTokenizer:
        return SemanticHistoryExperiment.create_tokenizer(self)

    @cached_property
    def base_model(self) -> SequenceRetrievalModel:
        model = self._create_model().to(self.runner_build_device)
        self._sid_initialization_diagnostics = _apply_sid_lookup_initialization(
            self, model
        )
        logger.info("Model architecture:\n%s", model)
        logger.info(
            "Total parameters: %s",
            f"{sum(parameter.numel() for parameter in model.parameters()):,}",
        )
        return model

    def create_optimizers(self) -> torch.optim.Optimizer:
        return SequenceExperiment.create_optimizers(self)

    @property
    def sid_initialization_diagnostics(self) -> dict[str, object]:
        if self._sid_initialization_diagnostics is None:
            raise RuntimeError("SID initialization has not run or is inapplicable")
        return self._sid_initialization_diagnostics

    def true_metric_options(self) -> dict[str, object]:
        return GenerationExperiment.true_metric_options(self)

    def create_runner(self) -> TrainRunner:
        _validate_kmeans_convergence(self)
        validate_collision_symbol_cap(
            self.semantic_codes,
            policy=self.semantic.collision_policy,
            base_levels=self.semantic.num_levels,
            require_suffix_feasibility=True,
        )
        return super().create_runner()

    def generation_architecture_metadata(self) -> dict[str, object]:
        metadata: dict[str, object] = {
            "history_representation": self.history_representation,
            "representation_width": self.representation_width,
            "sid_lookup_initialization": self.sid_lookup_initialization,
            "sid_lookup_projection": (
                _CONTENT_PROJECTION
                if self.sid_lookup_initialization == "content_pca"
                else None
            ),
            "g6_parameterization": "conventional",
            "g6_dataset_lineage": "native500m-v1",
            "g6_winner_diagnostics_deferred": True,
        }
        if self._sid_initialization_diagnostics is not None:
            metadata["sid_initialization_diagnostics"] = (
                self.sid_initialization_diagnostics
            )
        return self._with_best_model_artifact(metadata)


def build_control(
    *,
    backbone: Backbone,
    embedding_learning_rate: float,
    deep_learning_rate: float,
    run_name: str,
    seed: int = 42,
) -> GenerationExperiment:
    common = _common(
        backbone,
        embedding_learning_rate=embedding_learning_rate,
        deep_learning_rate=deep_learning_rate,
        run_name=run_name,
        seed=seed,
    )
    if backbone == "original_g1":
        return Native500MGenerationExperiment(**common)
    return Native500MMuTransferGenerationExperiment(
        **common,
        mup_base_dim=16,
        mup_delta_dim=32,
    )


def build_semantic_treatment(
    *,
    backbone: Backbone,
    representation: str,
    embedding_learning_rate: float,
    deep_learning_rate: float,
    num_levels: int,
    num_codes: int,
    run_name: str,
    seed: int = 42,
    representation_width: int = REPRESENTATION_WIDTH,
    collision_policy: CollisionPolicy = "suffix",
    sid_lookup_initialization: SidLookupInitialization = "random",
) -> Native500MSemanticHistoryExperiment | ConventionalSemanticHistoryExperiment:
    if representation not in REPRESENTATIONS:
        raise ValueError(f"unknown representation {representation!r}")
    if num_levels not in TOKENIZER_LEVELS:
        raise ValueError(f"num_levels must be one of {TOKENIZER_LEVELS}")
    if num_codes not in SHARED_CODEBOOK_SIZES:
        raise ValueError(f"num_codes must be one of {SHARED_CODEBOOK_SIZES}")
    if representation_width != REPRESENTATION_WIDTH:
        raise ValueError(f"representation_width must be {REPRESENTATION_WIDTH}")
    _validate_initialization(
        cast(SemanticHistoryRepresentation, representation),
        sid_lookup_initialization,
    )
    common = _common(
        backbone,
        embedding_learning_rate=embedding_learning_rate,
        deep_learning_rate=deep_learning_rate,
        run_name=run_name,
        seed=seed,
    )
    semantic = SemanticIdConfig(
        quantizer="kmeans",
        num_levels=num_levels,
        num_codes=num_codes,
        kmeans_iterations=KMEANS_MAX_ITERATIONS,
        kmeans_relative_inertia_tolerance=KMEANS_TOLERANCE,
        kmeans_assignment_early_stopping=True,
        collision_policy=collision_policy,
        seed=42,
    )
    experiment_type = (
        Native500MSemanticHistoryExperiment
        if backbone == "best_g1"
        else ConventionalSemanticHistoryExperiment
    )
    extra = (
        {"mup_base_dim": 16, "mup_delta_dim": 32}
        if backbone == "best_g1"
        else {}
    )
    return experiment_type(
        **common,
        **extra,
        history_representation=cast(SemanticHistoryRepresentation, representation),
        representation_width=representation_width,
        sid_lookup_initialization=sid_lookup_initialization,
        semantic=semantic,
    )


def build_collision_pair(
    *,
    backbone: Backbone,
    representation: str,
    embedding_learning_rate: float,
    deep_learning_rate: float,
    num_levels: int,
    num_codes: int,
    suffix_run_name: str,
    no_suffix_run_name: str,
    seed: int = 42,
    representation_width: int = REPRESENTATION_WIDTH,
    sid_lookup_initialization: SidLookupInitialization = "random",
) -> tuple[
    Native500MSemanticHistoryExperiment | ConventionalSemanticHistoryExperiment,
    Native500MSemanticHistoryExperiment | ConventionalSemanticHistoryExperiment,
]:
    common = {
        "backbone": backbone,
        "representation": representation,
        "embedding_learning_rate": embedding_learning_rate,
        "deep_learning_rate": deep_learning_rate,
        "num_levels": num_levels,
        "num_codes": num_codes,
        "seed": seed,
        "representation_width": representation_width,
        "sid_lookup_initialization": sid_lookup_initialization,
    }
    return (
        build_semantic_treatment(
            **common,
            run_name=suffix_run_name,
            collision_policy="suffix",
        ),
        build_semantic_treatment(
            **common,
            run_name=no_suffix_run_name,
            collision_policy="none",
        ),
    )
