from __future__ import annotations

from dataclasses import asdict
from functools import cache
import json
import math
import os
from pathlib import Path
import runpy
import sys
from typing import Any

from dcn.config.settings import transformer_metadata

from dcn.training_metadata import (
    NEGATIVE_SAMPLING_SEMANTICS_REVISION,
    TIMESTAMP_BIN_SEMANTICS_REVISION,
    has_current_generation_semantics,
)


TUNING_KEYS = {
    "G1_DATASET_SIZE",
    "G1_MAX_USERS",
    "G1_MAX_EPOCHS",
    "G1_SEED",
    "G1_TRAIN_BATCH_SIZE",
    "G1_VAL_BATCH_SIZE",
    "G1_TUNE_BATCH_SIZE",
    "G1_TUNE_CORRECT_POSITIVE_LOGQ",
    "G1_TUNE_DEEP_LR",
    "G1_TUNE_EMBEDDING_LR",
    "G1_TUNE_EPOCHS",
    "G1_TUNE_EXCLUDE_OWN_GROUP_NEGATIVES",
    "G1_TUNE_EXPERIMENT_FIELDS",
    "G1_TUNE_FFN_DIM",
    "G1_TUNE_GRADIENT_ACCUMULATION_STEPS",
    "G1_TUNE_LOGQ_ALPHA",
    "G1_TUNE_LOGQ_CORRECTION",
    "G1_TUNE_MASK_FALSE_NEGATIVES",
    "G1_TUNE_NUM_NEGATIVES",
    "G1_TUNE_NUM_LAYERS",
    "G1_TUNE_NUM_WORKERS",
    "G1_TUNE_RANDOM_FRACTION",
    "G1_TUNE_RUN",
    "G1_TUNE_RUN_REVISION",
    "G1_TUNE_SOURCE_VARIANT",
    "G1_TUNE_TRANSFORMER_FIELDS",
}

CONFIG_KEYS = TUNING_KEYS | {
    "G1_AGGREGATE_RUN",
    "G1_HOMEWORK_LOGQ_DATASET_SIZE",
    "G1_HOMEWORK_LOGQ_DEEP_LR",
    "G1_HOMEWORK_LOGQ_EMBEDDING_LR",
    "G1_HOMEWORK_LOGQ_EPOCHS",
    "G1_HOMEWORK_LOGQ_RUN",
    "G1_HOMEWORK_LOGQ_RUN_REVISION",
    "G1_HOMEWORK_RANDOM_DATASET_SIZE",
    "G1_HOMEWORK_RANDOM_DEEP_LR",
    "G1_HOMEWORK_RANDOM_EMBEDDING_LR",
    "G1_HOMEWORK_RANDOM_EPOCHS",
    "G1_HOMEWORK_RANDOM_RUN",
    "G1_HOMEWORK_RANDOM_RUN_REVISION",
    "G1_VARIANT",
    "G1_HOMEWORK_BATCH_SIZE",
    "G1_HOMEWORK_RUN_TAG",
    "G1_TRANSFER_BATCH_SIZE",
    "G1_TRANSFER_DEEP_LR",
    "G1_TRANSFER_DIM",
    "G1_TRANSFER_EMBEDDING_LR",
    "G1_TRANSFER_EPOCHS",
    "G1_TRANSFER_PARAMETERIZATION",
    "G1_TRANSFER_POWER_TOKENS",
    "G1_TRANSFER_RUN",
    "G1_TRANSFER_RUN_REVISION",
    "G1_TRANSFER_SOURCE_VARIANT",
    "G1_RQ5_RUN",
    "G1_RQ7_RUN",
    "G1_RQ8_RUN",
    "G1_RQ10_RUN",
    "G1_RQ11_RUN",
}

COMPLETE = "complete"
RESUMABLE = "resumable"
INCOMPATIBLE = "incompatible"


def _normalized(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _expected_metadata(experiment: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    item_embedding_dim = experiment.item_embedding_dim or experiment.model_dim
    top_level = {
        "dataset_size": experiment.size,
        "seed": experiment.seed,
        "num_epochs": experiment.num_epochs,
        "batch_size": experiment.dataloader.batch_size,
        "physical_batch_size": experiment.dataloader.batch_size,
        "gradient_accumulation_steps": (
            experiment.dataloader.gradient_accumulation_steps
        ),
        "effective_batch_size": experiment.dataloader.effective_batch_size,
        "val_batch_size": experiment.dataloader.val_batch_size,
        "num_workers": experiment.dataloader.num_workers,
        "prefetch_factor": experiment.dataloader.prefetch_factor,
        "model_dim": experiment.model_dim,
        "item_embedding_dim": item_embedding_dim,
        "embedding_learning_rate": experiment.embedding_learning_rate,
        "deep_learning_rate": experiment.deep_learning_rate,
        "weight_decay": experiment.weight_decay,
        "initializer_std": experiment.initializer_std,
        "runtime_dtype": str(experiment.runtime.dtype),
        "runtime_compile": experiment.runtime.compile,
        "gradient_clip_norm": experiment.runtime.gradient_clip_norm,
        "negative_sampling": experiment.negative_sampling,
        **(
            {
                "lr_schedule_horizon_epochs": (
                    experiment.lr_schedule_horizon_epochs
                )
            }
            if experiment.lr_schedule.requires_horizon
            else {}
        ),
    }
    invariants = {
        "experiment_class": type(experiment).__name__,
        "mup_base_dim": getattr(experiment, "mup_base_dim", None),
        "mup_delta_dim": getattr(experiment, "mup_delta_dim", None),
        "mup_base_ffn_dim": getattr(experiment, "mup_base_ffn_dim", None),
        "mup_delta_ffn_dim": getattr(experiment, "mup_delta_ffn_dim", None),
        "dataset_size": experiment.size,
        "user_sample": (
            None if experiment.user_sample is None else experiment.user_sample.name
        ),
        "event_type_filter": experiment.event_type_filter,
        "min_item_interactions_per_item": experiment.min_item_interactions_per_item,
        "drop_unmapped_items": experiment.drop_unmapped_items,
        "validation_interval_seconds": experiment.validation_interval_seconds,
        "day_range": asdict(experiment.day_range),
        "batch_size": experiment.dataloader.batch_size,
        "physical_batch_size": experiment.dataloader.batch_size,
        "gradient_accumulation_steps": (
            experiment.dataloader.gradient_accumulation_steps
        ),
        "effective_batch_size": experiment.dataloader.effective_batch_size,
        "model_dim": experiment.model_dim,
        "item_embedding_dim": item_embedding_dim,
        "max_seq_len": experiment.max_seq_len,
        "window": experiment.window,
        "bos": experiment.bos,
        "cls_token": experiment.effective_cls_token_mode != "none",
        "cls_token_mode": experiment.effective_cls_token_mode,
        "timestamp_delta": experiment.timestamp_delta,
        "timestamp_combination": experiment.timestamp_combination,
        "timestamp_num_bins": experiment.timestamp_num_bins,
        **(
            {"timestamp_bin_semantics_revision": TIMESTAMP_BIN_SEMANTICS_REVISION}
            if experiment.timestamp_delta == "bins"
            else {}
        ),
        "per_layer_item_embeddings": experiment.per_layer_item_embeddings,
        "per_layer_item_features": experiment.effective_per_layer_item_features,
        "per_layer_item_feature_dim": experiment.per_layer_item_feature_dim,
        "negative_sampling": experiment.negative_sampling,
        **(
            {
                "negative_sampling_semantics_revision": (
                    NEGATIVE_SAMPLING_SEMANTICS_REVISION
                )
            }
            if experiment.negative_sampling
            in {
                "online_logq",
                "mixed_online_logq",
                "mixed_offline_logq",
                "mixed_online_global_q",
                "mixed_online_global_q_negative_only",
            }
            else {}
        ),
        "num_in_batch_negatives": experiment.num_in_batch_negatives,
        "logq_correction": experiment.logq_correction,
        "random_negative_fraction": experiment.random_negative_fraction,
        "logq_alpha": experiment.logq_alpha,
        "correct_positive_logq": experiment.correct_positive_logq,
        "mask_false_negatives": experiment.mask_false_negatives,
        "exclude_own_group_negatives": experiment.exclude_own_group_negatives,
        "dense_random_negative_scores": experiment.dense_random_negative_scores,
        "eval_ks": experiment.eval_ks,
        "eval_max_users": experiment.eval_max_users,
        "eval_every_n_epochs": experiment.eval_every_n_epochs,
        "early_stopping_patience": experiment.early_stopping_patience,
        "early_stopping_min_delta": experiment.early_stopping_min_delta,
        "early_stopping_metric": experiment.checkpointing.best_metric_name,
        "early_stopping_metric_prefix": (
            experiment.checkpointing.best_metric_prefix
        ),
        "selection_k": experiment.selection_k,
        "evaluation_catalog": experiment.evaluation_catalog,
        "exclude_seen_from_evaluation": experiment.exclude_seen_from_evaluation,
        "restore_best_weights": experiment.restore_best_weights,
        "adaptive_schedule_early_stopping": (
            experiment.adaptive_schedule_early_stopping
        ),
        **(
            {
                "lr_schedule_horizon_epochs": (
                    experiment.lr_schedule_horizon_epochs
                )
            }
            if experiment.lr_schedule.requires_horizon
            else {}
        ),
        "transformer": transformer_metadata(experiment.transformer),
        "lr_schedule": asdict(experiment.lr_schedule),
    }
    if experiment.timestamp_delta == "bins":
        invariants["timestamp_bin_semantics_revision"] = (
            TIMESTAMP_BIN_SEMANTICS_REVISION
        )
    return _normalized(top_level), _normalized(invariants)


def _load_json(path: Path) -> Any:
    with path.open() as stream:
        return json.load(stream)


def _with_legacy_accumulation_defaults(metadata: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(metadata)
    normalized.setdefault(
        "cls_token_mode", "end_only" if normalized.get("cls_token") else "none"
    )
    normalized.setdefault("physical_batch_size", normalized.get("batch_size"))
    normalized.setdefault("gradient_accumulation_steps", 1)
    normalized.setdefault("effective_batch_size", normalized.get("batch_size"))
    invariants = normalized.get("transfer_invariants")
    if isinstance(invariants, dict):
        invariants = dict(invariants)
        invariants.setdefault("physical_batch_size", invariants.get("batch_size"))
        invariants.setdefault("gradient_accumulation_steps", 1)
        invariants.setdefault("effective_batch_size", invariants.get("batch_size"))
        invariants.setdefault("mup_base_ffn_dim", None)
        invariants.setdefault("mup_delta_ffn_dim", None)
        invariants.setdefault(
            "cls_token_mode",
            "end_only" if invariants.get("cls_token") else "none",
        )
        transformer = invariants.get("transformer")
        if isinstance(transformer, dict):
            transformer = dict(transformer)
            transformer.setdefault("gated_ffn_dropout", False)
            if transformer.get("learned_position_fusion") == "concat":
                transformer.setdefault("learned_position_fusion_semantics_revision", 1)
                if transformer["learned_position_fusion_semantics_revision"] < 3:
                    transformer.setdefault(
                        "learned_position_fusion_normalization", None
                    )
            invariants["transformer"] = transformer
        invariants.setdefault("adaptive_schedule_early_stopping", False)
        invariants.setdefault(
            "per_layer_item_features",
            "direct_add" if invariants.get("per_layer_item_embeddings") else "none",
        )
        invariants.setdefault("per_layer_item_feature_dim", None)
        schedule = invariants.get("lr_schedule")
        if isinstance(schedule, dict):
            schedule = dict(schedule)
            schedule.setdefault("optimizer_group_scope", "both")
            invariants["lr_schedule"] = schedule
        normalized["transfer_invariants"] = invariants
    return normalized


def _positive_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _valid_recipe_dynamic_metadata(metadata: dict[str, Any]) -> bool:
    max_epochs = metadata.get("num_epochs")
    epochs_trained = metadata.get("epochs_trained")
    targets = metadata.get("targets_per_epoch")
    tokens = metadata.get("tokens_per_epoch")
    steps = metadata.get("optimizer_steps")
    if not all(
        _positive_integer(value)
        for value in (max_epochs, epochs_trained, targets, tokens, steps)
    ):
        return False
    if metadata.get("max_epochs") != max_epochs or epochs_trained > max_epochs:
        return False
    stopped_epoch = metadata.get("stopped_epoch")
    best_epoch = metadata.get("best_epoch")
    if (
        stopped_epoch != epochs_trained
        or not _positive_integer(best_epoch)
        or best_epoch > stopped_epoch
    ):
        return False
    if metadata.get("training_horizon") != targets * epochs_trained:
        return False
    if metadata.get("token_horizon") != tokens * epochs_trained:
        return False
    tokens_seen = metadata.get("tokens_seen")
    if not _positive_integer(tokens_seen):
        return False
    schedule = metadata.get("transfer_invariants", {}).get("lr_schedule", {})
    if schedule.get("shape") != "power" and tokens_seen != tokens * epochs_trained:
        return False
    validation_loss = metadata.get("validation_loss")
    return validation_loss is None or (
        isinstance(validation_loss, (int, float))
        and not isinstance(validation_loss, bool)
        and math.isfinite(validation_loss)
        and validation_loss >= 0
    )


_HORIZON_FREE_SHAPES = frozenset({"constant", "inverse_sqrt", "power"})


def _spent_annealing_horizon(metadata: dict[str, Any]) -> int | None:
    """The horizon of a schedule that anneals over one and reached its end."""
    schedule = metadata.get("transfer_invariants", {}).get("lr_schedule", {})
    if schedule.get("shape") in _HORIZON_FREE_SHAPES:
        return None
    horizon = metadata.get("lr_schedule_horizon_epochs") or metadata.get("num_epochs")
    if not _positive_integer(horizon) or metadata["epochs_trained"] < horizon:
        return None
    return horizon


def _valid_dynamic_metadata(metadata: dict[str, Any]) -> bool:
    if not _valid_recipe_dynamic_metadata(metadata):
        return False
    invariants = metadata.get("transfer_invariants", {})
    if invariants.get("adaptive_schedule_early_stopping") is True:
        return _valid_adaptive_schedule_metadata(metadata)
    horizon = _spent_annealing_horizon(metadata)
    if horizon is not None:
        return metadata["best_epoch"] <= horizon
    return (
        metadata["stopped_epoch"] < metadata["max_epochs"]
        and metadata.get("early_stopped") is True
        and metadata.get("selection_resolved") is True
        and metadata.get("best_epoch_at_cap") is False
    )


def _valid_adaptive_schedule_metadata(metadata: dict[str, Any]) -> bool:
    if not _valid_adaptive_schedule_execution(metadata):
        return False
    calibration = metadata["horizon_calibration_status"]
    return calibration == "calibrated" and (
        metadata.get("early_stopped") is True
        and metadata.get("selection_resolved") is True
        and metadata.get("best_epoch_at_cap") is False
    )


def _valid_adaptive_schedule_execution(metadata: dict[str, Any]) -> bool:
    invariants = metadata.get("transfer_invariants", {})
    schedule = invariants.get("lr_schedule", {})
    steps_per_epoch = metadata.get("optimizer_steps_per_epoch")
    epochs_trained = metadata.get("epochs_trained")
    if not _positive_integer(steps_per_epoch):
        return False
    if metadata.get("optimizer_steps") != steps_per_epoch * epochs_trained:
        return False
    calibration = _adaptive_calibration(metadata, schedule)
    if calibration is None or (
        metadata.get("horizon_calibration_status"),
        metadata.get("next_lr_schedule_horizon_epochs"),
    ) != calibration:
        return False
    if metadata.get("selection_resolved") is not (calibration[0] == "calibrated"):
        return False
    horizon = metadata.get("lr_schedule_horizon_epochs")
    horizon_steps = metadata.get("lr_schedule_horizon_steps")
    if _schedule_requires_horizon(schedule):
        if not _positive_integer(horizon):
            return False
        if horizon_steps != steps_per_epoch * horizon:
            return False
    elif horizon_steps is not None:
        return False
    if schedule.get("shape") == "inverse_sqrt":
        fraction = schedule.get("timescale_fraction")
        timescale_steps = metadata.get("lr_schedule_timescale_steps")
        if (
            not isinstance(fraction, (int, float))
            or isinstance(fraction, bool)
            or not _positive_integer(timescale_steps)
            or timescale_steps != max(1, int(steps_per_epoch * horizon * fraction))
        ):
            return False
    elif metadata.get("lr_schedule_timescale_steps") is not None:
        return False
    return _valid_group_lr_traces(metadata, schedule)


def _schedule_requires_horizon(schedule: dict[str, Any]) -> bool:
    return (
        bool(schedule.get("warmup_fraction", 0.0))
        or schedule.get("timescale_fraction") is not None
        or schedule.get("shape") not in _HORIZON_FREE_SHAPES
    )


def _adaptive_calibration(
    metadata: dict[str, Any], schedule: dict[str, Any]
) -> tuple[str, int | None] | None:
    shape = schedule.get("shape")
    stopped = metadata.get("stopped_epoch")
    cap = metadata.get("max_epochs")
    early_stopped = metadata.get("early_stopped")
    if not _positive_integer(stopped) or not _positive_integer(cap):
        return None
    if shape == "constant":
        if early_stopped is True and stopped < cap:
            return "calibrated", None
        return "extend_cap", math.ceil(1.5 * cap)
    horizon = metadata.get("lr_schedule_horizon_epochs")
    if not _positive_integer(horizon):
        return None
    tolerance = max(3, round(0.1 * horizon))
    if shape not in _HORIZON_FREE_SHAPES:
        if early_stopped is True:
            if 0 <= horizon - stopped <= tolerance:
                return "calibrated", None
            return "shorten_horizon", stopped
        return "extend_horizon", math.ceil(1.5 * horizon)
    if early_stopped is not True:
        return "extend_cap", math.ceil(1.5 * cap)
    if abs(horizon - stopped) <= tolerance:
        return "calibrated", None
    return "recalibrate_horizon", stopped


def _valid_group_lr_traces(
    metadata: dict[str, Any], schedule: dict[str, Any]
) -> bool:
    traces = metadata.get("lr_group_traces")
    epochs_trained = metadata.get("epochs_trained")
    if not isinstance(traces, dict) or set(traces) != {"embedding", "deep"}:
        return False
    for trace in traces.values():
        if (
            not isinstance(trace, list)
            or len(trace) != epochs_trained
            or any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value < 0
                for value in trace
            )
        ):
            return False
    embedding_lr = metadata.get("embedding_learning_rate")
    deep_lr = metadata.get("deep_learning_rate")
    if any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value <= 0
        for value in (embedding_lr, deep_lr)
    ):
        return False
    factors = _expected_schedule_factors(metadata, schedule)
    if factors is None:
        return False
    scope = schedule.get("optimizer_group_scope")
    if scope not in {"both", "deep_only"}:
        return False
    expected_embedding = (
        [embedding_lr] * epochs_trained
        if scope == "deep_only"
        else [embedding_lr * factor for factor in factors]
    )
    expected_deep = [deep_lr * factor for factor in factors]
    return all(
        math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12)
        for actual, expected in zip(traces["embedding"], expected_embedding)
    ) and all(
        math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12)
        for actual, expected in zip(traces["deep"], expected_deep)
    )


def _expected_schedule_factors(
    metadata: dict[str, Any], schedule: dict[str, Any]
) -> list[float] | None:
    shape = schedule.get("shape")
    steps_per_epoch = metadata.get("optimizer_steps_per_epoch")
    epochs = metadata.get("epochs_trained")
    if shape == "constant":
        return [1.0] * epochs
    total = metadata.get("lr_schedule_horizon_steps")
    if not _positive_integer(total):
        return None
    warmup = int(total * schedule.get("warmup_fraction", 0.0))
    decay = max(1, total - warmup - 1)
    timescale = metadata.get("lr_schedule_timescale_steps")
    factors = []
    for epoch in range(1, epochs + 1):
        step = epoch * steps_per_epoch - 1
        if step < warmup:
            factors.append((step + 1) / warmup)
            continue
        decayed_step = step - warmup
        if shape == "inverse_sqrt":
            if not _positive_integer(timescale):
                return None
            decayed = math.sqrt(timescale / (timescale + decayed_step))
        else:
            progress = min(1.0, decayed_step / decay)
            if shape == "linear":
                decayed = 1 - progress
            elif shape == "cosine":
                cycles = schedule.get("cycles")
                if not _positive_integer(cycles):
                    return None
                decayed = (
                    0.0
                    if progress == 1
                    else 0.5
                    * (1 + math.cos(math.pi * ((cycles * progress) % 1.0)))
                )
            elif shape == "step":
                decayed = 1.0 if progress < 0.5 else 0.1 if progress < 0.75 else 0.01
            elif shape == "exponential":
                decayed = 0.01**progress
            elif shape == "polynomial":
                decayed = (1 - progress) ** 2
            elif shape == "warmup_stable_decay":
                decay_progress = max(0.0, (progress - 0.8) / 0.2)
                decayed = 0.5 * (1 + math.cos(math.pi * decay_progress))
            else:
                return None
        minimum = schedule.get("min_lr_fraction", 0.0)
        factors.append(minimum + (1 - minimum) * decayed)
    return factors


def _valid_metrics(metrics: Any) -> bool:
    if not isinstance(metrics, dict) or not metrics:
        return False
    for name, value in metrics.items():
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or (isinstance(value, float) and not math.isfinite(value))
            or value < 0
        ):
            return False
        if name.startswith(("recall@", "ndcg@", "coverage@")) and value > 1:
            return False
        if name == "num_users" and (
            value < 1
            or (isinstance(value, float) and not value.is_integer())
        ):
            return False
    return True


def _valid_config_metrics(metrics: Any, experiment: Any) -> bool:
    if not _valid_metrics(metrics):
        return False
    if not getattr(experiment, "adaptive_schedule_early_stopping", False):
        return True
    expected = {"num_users"}
    for k in experiment.eval_ks:
        expected.update(
            f"{name}@{k}"
            for name in ("ndcg", "recall", "capped_recall", "mrr", "coverage")
        )
    return set(metrics) == expected


def _tuning_assignments(raw_assignments: list[str]) -> dict[str, str]:
    assignments: dict[str, str] = {}
    for assignment in raw_assignments:
        name, separator, value = assignment.partition("=")
        if separator != "=" or name not in TUNING_KEYS:
            raise ValueError(f"unsupported verifier assignment: {assignment}")
        assignments[name] = value
    required = {
        "G1_TUNE_RUN",
        "G1_TUNE_SOURCE_VARIANT",
        "G1_TUNE_TRANSFORMER_FIELDS",
        "G1_TUNE_EXPERIMENT_FIELDS",
        "G1_TUNE_EMBEDDING_LR",
        "G1_TUNE_DEEP_LR",
    }
    if missing := required - assignments.keys():
        raise ValueError(f"missing verifier assignments: {sorted(missing)}")
    return assignments


def _tuning_experiment(dataset_size: str, assignments: dict[str, str]) -> Any:
    config_path = Path(__file__).parents[1] / "configs" / "rq_tuning_variant.py"
    return _isolated_experiment(
        config_path, {"G1_DATASET_SIZE": dataset_size, **assignments}
    )


def _verify_tuning(
    directory: Path,
    dataset_size: str,
    raw_assignments: list[str],
    expected_directory_name: str | None,
) -> bool:
    assignments = _tuning_assignments(raw_assignments)
    experiment = _tuning_experiment(dataset_size, assignments)
    expected_name = expected_directory_name or experiment.run_name
    if directory.name != expected_name:
        return False
    expected_top_level, expected_invariants = _expected_metadata(experiment)

    metrics_path = directory / "final_metrics.json"
    metadata_path = directory / "training_metadata.json"
    try:
        metrics = _load_json(metrics_path)
        metadata = _load_json(metadata_path)
        if isinstance(metadata, dict):
            metadata = _with_legacy_accumulation_defaults(metadata)
        metadata_mtime = metadata_path.stat().st_mtime_ns
        fresh = (
            has_current_generation_semantics(metadata)
            and metrics_path.stat().st_mtime_ns >= metadata_mtime
        )
    except (OSError, json.JSONDecodeError):
        return False
    if not fresh or not _valid_metrics(metrics) or not isinstance(metadata, dict):
        return False
    if any(metadata.get(name) != value for name, value in expected_top_level.items()):
        return False
    actual_invariants = metadata.get("transfer_invariants")
    if isinstance(actual_invariants, dict):
        actual_invariants = dict(actual_invariants)
        if actual_invariants.get("timestamp_bin_semantics_revision") is None:
            actual_invariants.pop("timestamp_bin_semantics_revision", None)
    if actual_invariants != expected_invariants:
        return False
    return _valid_dynamic_metadata(metadata)


def verify(directory: Path, dataset_size: str, raw_assignments: list[str]) -> bool:
    return _verify_tuning(directory, dataset_size, raw_assignments, None)


def verify_named_tuning(
    directory: Path,
    dataset_size: str,
    raw_assignments: list[str],
    expected_directory_name: str,
) -> bool:
    return _verify_tuning(
        directory,
        dataset_size,
        raw_assignments,
        expected_directory_name,
    )


def _config_assignments(raw_assignments: list[str]) -> dict[str, str]:
    assignments: dict[str, str] = {}
    for assignment in raw_assignments:
        name, separator, value = assignment.partition("=")
        if separator != "=" or name not in CONFIG_KEYS:
            raise ValueError(f"unsupported config-verifier assignment: {assignment}")
        assignments[name] = value
    return assignments


def _config_experiment(config_path: Path, assignments: dict[str, str]) -> Any:
    return _isolated_experiment(config_path, assignments)


@cache
def _canonical_path(path: str) -> Path:
    return Path(path).resolve()


def _modules_from(directory: Path) -> dict[str, Any]:
    directory = _canonical_path(os.fspath(directory))
    modules = {}
    for name, module in sys.modules.items():
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            continue
        try:
            if _canonical_path(os.fspath(module_file)).is_relative_to(directory):
                modules[name] = module
        except (OSError, RuntimeError, TypeError):
            continue
    return modules


def _unbind_modules(modules: dict[str, Any]) -> None:
    for name, module in modules.items():
        if sys.modules.get(name) is module:
            sys.modules.pop(name)
        parent_name, separator, child_name = name.rpartition(".")
        parent = sys.modules.get(parent_name) if separator else None
        if parent is not None and getattr(parent, child_name, None) is module:
            delattr(parent, child_name)


def _restore_modules(modules: dict[str, Any]) -> None:
    for name, module in modules.items():
        sys.modules[name] = module
        parent_name, separator, child_name = name.rpartition(".")
        parent = sys.modules.get(parent_name) if separator else None
        if parent is not None:
            setattr(parent, child_name, module)


def _isolated_experiment(
    config_path: Path, assignments: dict[str, str]
) -> Any:
    config_modules = _modules_from(config_path.parent)
    original = {
        name: value for name, value in os.environ.items() if name.startswith("G1_")
    }
    try:
        _unbind_modules(config_modules)
        for name in tuple(os.environ):
            if name.startswith("G1_"):
                os.environ.pop(name)
        os.environ.update(assignments)
        return runpy.run_path(str(config_path))["experiment"]
    finally:
        _unbind_modules(_modules_from(config_path.parent))
        _restore_modules(config_modules)
        for name in tuple(os.environ):
            if name.startswith("G1_"):
                os.environ.pop(name)
        os.environ.update(original)


def has_unaccumulated_batch_contract(
    metadata: dict[str, Any], batch_size: int
) -> bool:
    invariants = metadata.get("transfer_invariants", {})
    expected_batches = (batch_size, batch_size, batch_size)
    return (
        (
            metadata.get("batch_size"),
            metadata.get("physical_batch_size"),
            metadata.get("effective_batch_size"),
        )
        == expected_batches
        and metadata.get("gradient_accumulation_steps") == 1
        and (
            invariants.get("batch_size"),
            invariants.get("physical_batch_size"),
            invariants.get("effective_batch_size"),
        )
        == expected_batches
        and invariants.get("gradient_accumulation_steps") == 1
    )


def _config_artifact_metadata(
    directory: Path, experiment: Any, expected_run_name: str
) -> dict[str, Any] | None:
    if directory.name != expected_run_name:
        return None
    expected_top_level, expected_invariants = _expected_metadata(experiment)

    metrics_path = directory / "final_metrics.json"
    metadata_path = directory / "training_metadata.json"
    try:
        metrics = _load_json(metrics_path)
        metadata = _load_json(metadata_path)
        if isinstance(metadata, dict):
            metadata = _with_legacy_accumulation_defaults(metadata)
        fresh = (
            has_current_generation_semantics(metadata)
            and metrics_path.stat().st_mtime_ns >= metadata_path.stat().st_mtime_ns
        )
    except (OSError, json.JSONDecodeError):
        return None
    if (
        not fresh
        or not _valid_config_metrics(metrics, experiment)
        or not isinstance(metadata, dict)
    ):
        return None
    if any(metadata.get(name) != value for name, value in expected_top_level.items()):
        return None
    if metadata.get("transfer_invariants") != expected_invariants:
        return None
    return metadata


def _verify_config_experiment(
    directory: Path, experiment: Any, expected_run_name: str
) -> bool:
    metadata = _config_artifact_metadata(directory, experiment, expected_run_name)
    return metadata is not None and _valid_dynamic_metadata(metadata)


def _verify_config_recipe_experiment(
    directory: Path, experiment: Any, expected_run_name: str
) -> bool:
    metadata = _config_artifact_metadata(directory, experiment, expected_run_name)
    if metadata is None or not _valid_recipe_dynamic_metadata(metadata):
        return False
    invariants = metadata.get("transfer_invariants", {})
    return (
        _valid_adaptive_schedule_execution(metadata)
        if invariants.get("adaptive_schedule_early_stopping") is True
        else True
    )


def verify_config(
    directory: Path, config_path: Path, raw_assignments: list[str]
) -> bool:
    assignments = _config_assignments(raw_assignments)
    experiment = _config_experiment(config_path, assignments)
    return _verify_config_experiment(directory, experiment, experiment.run_name)


def verify_config_recipe(
    directory: Path, config_path: Path, raw_assignments: list[str]
) -> bool:
    assignments = _config_assignments(raw_assignments)
    experiment = _config_experiment(config_path, assignments)
    return _verify_config_recipe_experiment(
        directory, experiment, experiment.run_name
    )


def verify_config_alias(
    directory: Path,
    config_path: Path,
    raw_assignments: list[str],
    expected_run_name: str,
) -> bool:
    assignments = _config_assignments(raw_assignments)
    experiment = _config_experiment(config_path, assignments)
    return _verify_config_experiment(directory, experiment, expected_run_name)


def _partial_metadata_matches(
    metadata: Any,
    expected_top_level: dict[str, Any],
    expected_invariants: dict[str, Any],
) -> bool:
    if not isinstance(metadata, dict):
        return False
    metadata = _with_legacy_accumulation_defaults(metadata)
    if not has_current_generation_semantics(metadata):
        return False
    if any(
        name in metadata and metadata[name] != value
        for name, value in expected_top_level.items()
    ):
        return False
    invariants = metadata.get("transfer_invariants")
    return invariants is None or invariants == expected_invariants


def _classify(
    directory: Path,
    expected_top_level: dict[str, Any],
    expected_invariants: dict[str, Any],
    complete: bool,
) -> str:
    if complete:
        return COMPLETE
    if directory.is_symlink():
        return INCOMPATIBLE
    metrics_path = directory / "final_metrics.json"
    metadata_path = directory / "training_metadata.json"
    if not metadata_path.exists():
        return INCOMPATIBLE if metrics_path.exists() else RESUMABLE
    try:
        metadata = _load_json(metadata_path)
    except (OSError, json.JSONDecodeError):
        return RESUMABLE
    return (
        RESUMABLE
        if _partial_metadata_matches(
            metadata, expected_top_level, expected_invariants
        )
        else INCOMPATIBLE
    )


def classify_tuning(
    directory: Path, dataset_size: str, raw_assignments: list[str]
) -> str:
    assignments = _tuning_assignments(raw_assignments)
    experiment = _tuning_experiment(dataset_size, assignments)
    expected_top_level, expected_invariants = _expected_metadata(experiment)
    return _classify(
        directory,
        expected_top_level,
        expected_invariants,
        verify(directory, dataset_size, raw_assignments),
    )


def classify_config(
    directory: Path, config_path: Path, raw_assignments: list[str]
) -> str:
    assignments = _config_assignments(raw_assignments)
    experiment = _config_experiment(config_path, assignments)
    if directory.name != experiment.run_name:
        return INCOMPATIBLE
    expected_top_level, expected_invariants = _expected_metadata(experiment)
    return _classify(
        directory,
        expected_top_level,
        expected_invariants,
        verify_config(directory, config_path, raw_assignments),
    )


def classify_config_recipe(
    directory: Path, config_path: Path, raw_assignments: list[str]
) -> str:
    assignments = _config_assignments(raw_assignments)
    experiment = _config_experiment(config_path, assignments)
    if directory.name != experiment.run_name:
        return INCOMPATIBLE
    expected_top_level, expected_invariants = _expected_metadata(experiment)
    return _classify(
        directory,
        expected_top_level,
        expected_invariants,
        verify_config_recipe(directory, config_path, raw_assignments),
    )


def _serve() -> int:
    for line in sys.stdin:
        fields = line.rstrip("\n").split("\t")
        try:
            if len(fields) < 3:
                raise ValueError("verifier request has fewer than three fields")
            if fields[1] == "classify-config":
                result = classify_config(Path(fields[0]), Path(fields[2]), fields[3:])
            elif fields[1] == "classify-config-recipe":
                result = classify_config_recipe(
                    Path(fields[0]), Path(fields[2]), fields[3:]
                )
            elif fields[1] == "classify-tuning":
                result = classify_tuning(Path(fields[0]), fields[2], fields[3:])
            elif fields[1] == "config-recipe":
                valid = verify_config_recipe(
                    Path(fields[0]), Path(fields[2]), fields[3:]
                )
                result = "0" if valid else "1"
            elif fields[1] == "config":
                valid = verify_config(Path(fields[0]), Path(fields[2]), fields[3:])
                result = "0" if valid else "1"
            else:
                valid = verify(Path(fields[0]), fields[1], fields[2:])
                result = "0" if valid else "1"
            print(result, flush=True)
        except Exception as error:
            print(f"2\t{type(error).__name__}: {error}", flush=True)
    return 0


def main() -> int:
    if sys.argv[1:] == ["--server"]:
        return _serve()
    if len(sys.argv) >= 4 and sys.argv[1] == "--config":
        return 0 if verify_config(Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4:]) else 1
    if len(sys.argv) < 4:
        raise SystemExit(
            "usage: verify_artifact.py DIRECTORY DATASET_SIZE NAME=VALUE ..."
        )
    return 0 if verify(Path(sys.argv[1]), sys.argv[2], sys.argv[3:]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
