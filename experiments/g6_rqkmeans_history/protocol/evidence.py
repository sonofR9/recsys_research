from __future__ import annotations

from dataclasses import asdict, dataclass
import fcntl
import json
import math
from pathlib import Path
import statistics
from typing import Any, Sequence

from dcn.config import SemanticIdConfig
from dcn.config.settings import transformer_metadata
from dcn.eval.ranking_evidence import load_ranking_evidence
from dcn.training_metadata import (
    GENERATION_TRAINING_SEMANTICS_REVISION,
    TIMESTAMP_BIN_SEMANTICS_REVISION,
)
from experiments.g6_rqkmeans_history.launchers.compiled import build_experiment
from experiments.g6_rqkmeans_history.protocol.manifest import (
    CompiledJob,
    RANKING_EVIDENCE_GROUP,
    approved_manifest,
    validate_compiled_job,
)
from experiments.g6_rqkmeans_history.protocol.optuna_driver import Selection


REQUIRED_METRICS = tuple(
    f"{metric}@{cutoff}"
    for metric in ("recall", "ndcg", "mrr", "capped_recall", "coverage")
    for cutoff in (10, 50, 100)
)


@dataclass(frozen=True)
class VerifiedArtifact:
    compiled: CompiledJob
    path: Path
    metrics: dict[str, float]
    metadata: dict[str, Any]
    semantic_diagnostics: dict[str, Any] | None

    def selection(self) -> Selection:
        return Selection(
            compiled=self.compiled,
            objective=self.metrics["recall@100"],
            selection_resolved=True,
        )


class BoundaryApprovalRequired(RuntimeError):
    pass


class CapExtensionRequired(ValueError):
    pass


@dataclass(frozen=True)
class InferenceCostContract:
    version: int
    sequence_tokens: int
    transformer_multiply_accumulates: int
    tokenizer_multiply_accumulates: int
    embedding_scalar_reads: int

    @property
    def total_multiply_accumulates(self) -> int:
        return (
            self.transformer_multiply_accumulates + self.tokenizer_multiply_accumulates
        )

    @property
    def sort_key(self) -> tuple[int, int, int]:
        return (
            self.total_multiply_accumulates,
            self.embedding_scalar_reads,
            self.sequence_tokens,
        )


def load_verified_artifact(
    compiled: CompiledJob,
    logs_root: Path,
) -> VerifiedArtifact:
    validate_compiled_job(compiled)
    directory = logs_root / compiled.run_name
    contract = _read_json(directory / "g6_rq0_job.json")
    expected_contract = compiled.to_contract(approved_manifest())
    if contract != expected_contract:
        raise ValueError(f"{compiled.run_name}: job contract changed")
    metrics = _numeric_metrics(_read_json(directory / "final_metrics.json"))
    for name in REQUIRED_METRICS:
        if name not in metrics:
            raise ValueError(f"{compiled.run_name}: missing {name}")
    metadata = _read_json(directory / "training_metadata.json")
    _validate_metadata(compiled, metadata)
    load_ranking_evidence(
        logs_root / ".ranking-evidence" / RANKING_EVIDENCE_GROUP / "context.pt",
        directory / "ranking_evidence.pt",
    )
    diagnostics = None
    if _is_semantic(compiled):
        _validate_sid_metrics(compiled, metrics)
        diagnostics = _read_json(directory / "semantic_id_diagnostics.json")
        _validate_semantic_diagnostics(compiled, diagnostics)
    return VerifiedArtifact(compiled, directory, metrics, metadata, diagnostics)


def artifact_state(compiled: CompiledJob, logs_root: Path) -> str:
    directory = logs_root / compiled.run_name
    terminal = [
        directory / "g6_rq0_job.json",
        directory / "final_metrics.json",
        directory / "training_metadata.json",
        directory / "ranking_evidence.pt",
    ]
    if _is_semantic(compiled):
        terminal.append(directory / "semantic_id_diagnostics.json")
    if all(path.is_file() for path in terminal):
        try:
            load_verified_artifact(compiled, logs_root)
        except CapExtensionRequired:
            return "extend_cap"
        return "complete"
    if directory.exists() and any(directory.iterdir()):
        return "partial"
    return "missing"


def archive_run_artifact(
    compiled: CompiledJob,
    logs_root: Path,
    *,
    reason: str,
) -> Path:
    expected_state = {"incomplete": "partial", "cap-exhausted": "extend_cap"}
    if reason not in expected_state:
        raise ValueError(f"unknown G6 RQ0 archive reason {reason!r}")
    directory = logs_root / compiled.run_name
    lock_path = logs_root / ".run-locks" / f"{compiled.run_name}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"active training process owns {compiled.run_name}"
            ) from error
        state = artifact_state(compiled, logs_root)
        if state != expected_state[reason]:
            raise RuntimeError(
                f"{compiled.run_name}: cannot archive {state} artifact as {reason}"
            )
        archive_root = logs_root / "old"
        archive_root.mkdir(parents=True, exist_ok=True)
        attempt = 1
        while True:
            archive = archive_root / f"{compiled.run_name}.{reason}-{attempt:03d}"
            if not archive.exists() and not archive.is_symlink():
                break
            attempt += 1
        directory.rename(archive)
        return archive


def select_best(
    artifacts: Sequence[VerifiedArtifact],
    *,
    recall_band: float,
    ndcg_band: float,
) -> VerifiedArtifact:
    if not artifacts:
        raise ValueError("selection requires at least one artifact")
    if recall_band < 0 or ndcg_band < 0:
        raise ValueError("metric bands must be non-negative")
    best_recall = max(artifact.metrics["recall@100"] for artifact in artifacts)
    recall_tied = [
        artifact
        for artifact in artifacts
        if artifact.metrics["recall@100"] >= best_recall - recall_band
    ]
    best_ndcg = max(artifact.metrics["ndcg@100"] for artifact in recall_tied)
    ndcg_tied = [
        artifact
        for artifact in recall_tied
        if artifact.metrics["ndcg@100"] >= best_ndcg - ndcg_band
    ]
    order = {job.id: index for index, job in enumerate(approved_manifest().jobs)}
    return min(
        ndcg_tied,
        key=lambda artifact: (
            inference_cost_contract(artifact).sort_key,
            order[artifact.compiled.approved.id],
        ),
    )


def inference_cost_contract(artifact: VerifiedArtifact) -> InferenceCostContract:
    compiled = artifact.compiled
    validate_compiled_job(compiled)
    experiment = build_experiment(compiled)
    transformer = experiment.transformer
    events = experiment.max_seq_len
    tokens_per_event = experiment.history_tokens_per_event
    cls_tokens = (
        events
        if experiment.effective_cls_token_mode == "interleaved"
        else int(experiment.effective_cls_token_mode == "end_only")
    )
    sequence_tokens = events * tokens_per_event + int(experiment.bos) + cls_tokens
    model_dim = experiment.model_dim
    head_dim = model_dim // transformer.nhead
    attention_projection = (
        2 * model_dim * model_dim + 2 * model_dim * head_dim * transformer.num_kv_heads
    )
    attention = 2 * sequence_tokens * sequence_tokens * model_dim
    ffn_projections = (
        (3 if transformer.ffn in {"swiglu", "geglu", "reglu"} else 2)
        * model_dim
        * transformer.ffn_intermediate_dim
    )
    transformer_macs = transformer.num_layers * (
        sequence_tokens * (attention_projection + ffn_projections) + attention
    )
    content_width = _semantic_content_width(artifact)
    tokenizer_macs, embedding_reads = _tokenizer_cost_per_event(
        compiled, model_dim, content_width
    )
    return InferenceCostContract(
        version=1,
        sequence_tokens=sequence_tokens,
        transformer_multiply_accumulates=transformer_macs,
        tokenizer_multiply_accumulates=events * tokenizer_macs,
        embedding_scalar_reads=events * embedding_reads,
    )


def _tokenizer_cost_per_event(
    compiled: CompiledJob,
    model_dim: int,
    content_width: int | None,
) -> tuple[int, int]:
    representation = compiled.parameters.get("representation")
    if representation is None:
        return 0, model_dim
    levels = int(compiled.parameters["num_levels"]) + 1
    width = int(compiled.parameters["representation_width"])
    if content_width is None:
        raise ValueError("semantic serving cost requires semantic content width")
    if representation == "learned_sid_event":
        return 2 * levels * width * width + width * model_dim, 2 * levels * width
    if representation == "item_frozen_sid_event":
        return (model_dim + levels * content_width) * width + width * model_dim, (
            model_dim + levels * content_width
        )
    if representation == "item_learned_frozen_sid_event":
        combined = width + content_width
        return (model_dim + levels * combined) * width + width * model_dim, (
            model_dim + levels * combined
        )
    if representation == "learned_sid_tokens":
        return levels * width * model_dim, levels * width
    if representation == "learned_frozen_sid_tokens":
        combined = width + content_width
        return levels * (combined * width + width * model_dim), levels * combined
    if representation == "frozen_sid_tokens":
        return (
            levels * (content_width * width + width * model_dim),
            levels * content_width,
        )
    if representation == "interleaved_item_sid_tokens":
        return levels * width * model_dim + model_dim * model_dim, (
            levels * width + model_dim
        )
    raise ValueError(f"unknown representation {representation!r}")


def _semantic_content_width(artifact: VerifiedArtifact) -> int | None:
    if not _is_semantic(artifact.compiled):
        return None
    diagnostics = artifact.semantic_diagnostics
    if not isinstance(diagnostics, dict):
        raise ValueError("semantic serving cost requires verified diagnostics")
    width = diagnostics.get("semantic_content_width")
    if not isinstance(width, int) or isinstance(width, bool) or width < 1:
        raise ValueError("semantic serving cost has invalid semantic content width")
    return width


def require_resolved_boundary(winner: VerifiedArtifact) -> None:
    job = winner.compiled.approved
    if job.stage != "lr_boundary":
        return
    if job.forced_parameters.get("boundary_slot") == 3:
        raise BoundaryApprovalRequired(
            f"{job.run_name}: outermost LR extension won; new approval is required"
        )


def empirical_bands(
    control_artifacts: Sequence[VerifiedArtifact],
) -> dict[str, float]:
    if len(control_artifacts) != 10:
        raise ValueError("native-50M bands require exactly ten control artifacts")
    seeds = sorted(artifact.compiled.approved.seed for artifact in control_artifacts)
    if seeds != list(range(42, 52)):
        raise ValueError("native-50M bands require control seeds 42 through 51")
    common_metrics = set.intersection(
        *(set(artifact.metrics) for artifact in control_artifacts)
    )
    return {
        name: _round_up_one_significant_digit(
            statistics.stdev(artifact.metrics[name] for artifact in control_artifacts)
        )
        for name in sorted(common_metrics)
        if name != "num_users"
    }


def write_empirical_bands(path: Path, bands: dict[str, float]) -> None:
    document = {
        "dataset_size": "native-50m",
        "seeds": list(range(42, 52)),
        "rounding": "sample standard deviation rounded upward to one significant digit",
        "bands": bands,
    }
    content = json.dumps(document, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text() != content:
        raise RuntimeError(f"existing empirical bands differ: {path}")
    path.write_text(content)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {path}") from error
    if not isinstance(document, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return document


def _numeric_metrics(document: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for name, value in document.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"metric {name} must be numeric")
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"metric {name} must be finite and non-negative")
        metrics[name] = value
    return metrics


def _validate_metadata(
    compiled: CompiledJob,
    metadata: dict[str, Any],
    *,
    experiment_builder=build_experiment,
    builder_resolver=None,
) -> None:
    run_name = compiled.run_name
    if metadata.get("dataset_size") != "50m":
        raise ValueError(f"{run_name}: dataset size changed")
    if metadata.get("seed") != compiled.approved.seed:
        raise ValueError(f"{run_name}: seed changed")
    if metadata.get("training_semantics_revision") != (
        GENERATION_TRAINING_SEMANTICS_REVISION
    ):
        raise ValueError(f"{run_name}: training semantics changed")
    expected = compiled.parameters
    for parameter_name, metadata_name in (
        ("batch_size", "batch_size"),
        ("validation_batch_size", "val_batch_size"),
        ("embedding_learning_rate", "embedding_learning_rate"),
        ("deep_learning_rate", "deep_learning_rate"),
    ):
        if metadata.get(metadata_name) != expected[parameter_name]:
            raise ValueError(f"{run_name}: {parameter_name} changed")
    experiment = experiment_builder(compiled)
    for name in ("num_epochs", "max_epochs"):
        if metadata.get(name) != experiment.num_epochs:
            raise ValueError(f"{run_name}: {name} changed")
    transfer = metadata.get("transfer_invariants")
    if not isinstance(transfer, dict):
        raise ValueError(f"{run_name}: transfer invariants are absent")
    expected_transfer = {
        "experiment_class": type(experiment).__name__,
        "mup_base_dim": experiment.mup_base_dim,
        "mup_delta_dim": experiment.mup_delta_dim,
        "mup_base_ffn_dim": experiment.mup_base_ffn_dim,
        "mup_delta_ffn_dim": experiment.mup_delta_ffn_dim,
        "dataset_size": experiment.size,
        "user_sample": None,
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
        "item_embedding_dim": experiment.item_embedding_dim,
        "max_seq_len": experiment.max_seq_len,
        "window": experiment.window,
        "bos": experiment.bos,
        "cls_token": experiment.effective_cls_token_mode != "none",
        "cls_token_mode": experiment.effective_cls_token_mode,
        "timestamp_delta": experiment.timestamp_delta,
        "timestamp_combination": experiment.timestamp_combination,
        "timestamp_num_bins": experiment.timestamp_num_bins,
        "per_layer_item_embeddings": experiment.per_layer_item_embeddings,
        "per_layer_item_features": experiment.effective_per_layer_item_features,
        "per_layer_item_feature_dim": experiment.per_layer_item_feature_dim,
        "negative_sampling": experiment.negative_sampling,
        "num_in_batch_negatives": experiment.num_in_batch_negatives,
        "logq_correction": experiment.logq_correction,
        "random_negative_fraction": experiment.random_negative_fraction,
        "logq_alpha": experiment.logq_alpha,
        "correct_positive_logq": experiment.correct_positive_logq,
        "mask_false_negatives": experiment.mask_false_negatives,
        "exclude_own_group_negatives": experiment.exclude_own_group_negatives,
        "dense_random_negative_scores": experiment.dense_random_negative_scores,
        "eval_ks": list(experiment.eval_ks),
        "eval_max_users": experiment.eval_max_users,
        "eval_every_n_epochs": experiment.eval_every_n_epochs,
        "early_stopping_patience": experiment.early_stopping_patience,
        "early_stopping_min_delta": experiment.early_stopping_min_delta,
        "early_stopping_metric": experiment.checkpointing.best_metric_name,
        "early_stopping_metric_prefix": experiment.checkpointing.best_metric_prefix,
        "selection_k": experiment.selection_k,
        "evaluation_catalog": experiment.evaluation_catalog,
        "exclude_seen_from_evaluation": experiment.exclude_seen_from_evaluation,
        "restore_best_weights": experiment.restore_best_weights,
        "adaptive_schedule_early_stopping": (
            experiment.adaptive_schedule_early_stopping
        ),
        "transformer": transformer_metadata(experiment.transformer),
        "lr_schedule": asdict(experiment.lr_schedule),
    }
    if experiment.timestamp_delta == "bins":
        expected_transfer["timestamp_bin_semantics_revision"] = (
            TIMESTAMP_BIN_SEMANTICS_REVISION
        )
    if experiment.lr_schedule.requires_horizon:
        expected_transfer["lr_schedule_horizon_epochs"] = (
            experiment.lr_schedule_horizon_epochs
        )
    expected_transfer = json.loads(json.dumps(expected_transfer))
    for name, value in expected_transfer.items():
        if transfer.get(name) != value:
            raise ValueError(f"{run_name}: transfer invariant {name} changed")
    builder = (
        _builder(compiled) if builder_resolver is None else builder_resolver(compiled)
    )
    if builder in {"primary_control", "treatment"}:
        if metadata.get("selection_resolved") is not True:
            raise ValueError(f"{run_name}: training selection is unresolved")
        if metadata.get("epochs_trained") != 15:
            raise ValueError(f"{run_name}: cosine horizon did not finish")
    else:
        early_stopped = metadata.get("early_stopped")
        epochs_trained = metadata.get("epochs_trained")
        if early_stopped is not True and epochs_trained == experiment.num_epochs:
            raise CapExtensionRequired(
                f"{run_name}: original-backbone epoch cap must be extended"
            )
        if metadata.get("selection_resolved") is not True:
            raise ValueError(f"{run_name}: training selection is unresolved")
        if early_stopped is not True:
            raise ValueError(f"{run_name}: early stopping metadata is inconsistent")
        if metadata.get("best_epoch_at_cap") is not False:
            raise ValueError(f"{run_name}: best epoch reached the cap")


def _builder(compiled: CompiledJob) -> str:
    stage = compiled.approved.stage
    if stage in {"primary_control_tuning", "primary_control_repeats"}:
        return "primary_control"
    if stage == "original_control_tuning":
        return "original_control"
    if stage == "treatment_tuning":
        return "treatment"
    if stage == "bridge_tuning":
        return "bridge"
    return str(compiled.parameters["builder"])


def _is_semantic(compiled: CompiledJob) -> bool:
    return _builder(compiled) in {"treatment", "bridge"}


def _validate_sid_metrics(compiled: CompiledJob, metrics: dict[str, float]) -> None:
    levels = int(compiled.parameters["num_levels"])
    required = {f"sid_exact_recall@{cutoff}" for cutoff in (10, 50, 100)} | {
        f"sid_prefix_recall@{cutoff}_l{level}"
        for cutoff in (10, 50, 100)
        for level in range(1, levels + 1)
    }
    missing = required - metrics.keys()
    if missing:
        raise ValueError(
            f"{compiled.run_name}: missing SID metrics " + ", ".join(sorted(missing))
        )


def _validate_semantic_diagnostics(
    compiled: CompiledJob, diagnostics: dict[str, Any]
) -> None:
    levels = int(compiled.parameters["num_levels"])
    codes = int(compiled.parameters["num_codes"])
    expected_cache_key = SemanticIdConfig(
        quantizer="kmeans",
        num_levels=levels,
        num_codes=codes,
        kmeans_iterations=20,
        seed=42,
    ).cache_key
    expected_scalars = {
        "semantic_cache_key": expected_cache_key,
        "num_levels": levels,
        "shared_num_codes": codes,
    }
    for name, expected in expected_scalars.items():
        if diagnostics.get(name) != expected:
            raise ValueError(f"{compiled.run_name}: diagnostic {name} changed")
    content_width = diagnostics.get("semantic_content_width")
    if (
        not isinstance(content_width, int)
        or isinstance(content_width, bool)
        or content_width < 1
    ):
        raise ValueError(f"{compiled.run_name}: invalid semantic content width")
    for name in ("identifier_collision_rate", "collided_item_fraction"):
        value = diagnostics.get(name)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or not 0 <= float(value) <= 1
        ):
            raise ValueError(f"{compiled.run_name}: invalid diagnostic {name}")
    for name in (
        "p95_occupied_load",
        "p95_to_mean_occupied_load",
        "intra_code_cosine_similarity",
    ):
        values = diagnostics.get(name)
        if not isinstance(values, list) or len(values) != levels:
            raise ValueError(f"{compiled.run_name}: diagnostic {name} has wrong levels")
        for value in values:
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"{compiled.run_name}: invalid diagnostic {name}")
    suffix_symbols = diagnostics.get("collision_suffix_symbols")
    if (
        not isinstance(suffix_symbols, int)
        or isinstance(suffix_symbols, bool)
        or not 1 <= suffix_symbols <= 8192
    ):
        raise ValueError(
            f"{compiled.run_name}: collision suffix exceeds the symbol cap"
        )


def _round_up_one_significant_digit(value: float) -> float:
    if value == 0:
        return 0.0
    magnitude = 10 ** math.floor(math.log10(value))
    return math.ceil(value / magnitude) * magnitude
