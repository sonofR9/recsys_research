from __future__ import annotations

import argparse
import base64
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
from typing import Any

from dcn.config import Experiment, GenerationExperiment
from dcn.config.settings import transformer_metadata
from experiments.g2_esasrec.configs.local import (
    LocalG2Experiment,
    build_component,
    build_control,
    build_mixed_sampler,
)
from experiments.g2_esasrec.launchers.cost import attach_cost_evidence
from experiments.g2_esasrec.protocol.manifest import (
    CompiledJob,
    approved_manifest,
    load_compiled_jobs,
    validate_compiled_job,
)
from experiments.g2_esasrec.protocol.local_provenance import (
    LOCAL_IMPLEMENTATION_SOURCES,
    local_source_manifest as build_local_source_manifest,
)

JOB_ENVIRONMENT = "G2_COMPILED_JOB_B64"
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _builder_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    excluded = {
        "builder",
        "rectools_version",
        "selected_control",
        "selected_control_job_id",
        "source_job_id",
    }
    return {name: value for name, value in parameters.items() if name not in excluded}


def build_local_experiment(
    compiled: CompiledJob,
) -> GenerationExperiment | LocalG2Experiment:
    job = compiled.approved
    parameters = _builder_parameters(compiled.parameters)
    if job.stage in {"control_tuning", "control_repeats"}:
        experiment = build_control(run_name=job.run_name, **parameters)
    elif job.stage == "component_tuning":
        experiment = build_component(job.method, run_name=job.run_name, **parameters)
    elif job.stage == "mixed_tuning":
        experiment = build_mixed_sampler(run_name=job.run_name, **parameters)
    elif job.stage in {"lr_boundary", "reversal_confirmation"}:
        builder = compiled.parameters["builder"]
        if builder == "control":
            experiment = build_control(run_name=job.run_name, **parameters)
        elif builder == "mixed_sampler":
            experiment = build_mixed_sampler(run_name=job.run_name, **parameters)
        elif builder == "component":
            method = parameters.pop("method")
            experiment = build_component(method, run_name=job.run_name, **parameters)
        else:
            raise ValueError(f"{job.id}: unknown resolved builder {builder!r}")
    else:
        raise ValueError(f"{job.id} is not a local training job")
    experiment = replace(experiment, seed=job.seed)
    attach_cost_evidence(experiment)
    return experiment


def build_official_experiment(compiled: CompiledJob, interpreter: Path) -> Experiment:
    from experiments.g2_esasrec.official.queued import RecToolsExperiment

    if compiled.approved.stage != "official":
        raise ValueError("official queue adapter requires an official manifest job")
    if not interpreter.is_file():
        raise ValueError("G2_RECTOOLS_PYTHON must name the RecTools 0.19.0 interpreter")
    return RecToolsExperiment(
        run_name=compiled.approved.run_name,
        seed=compiled.approved.seed,
        rectools_python=interpreter,
        max_epochs=100,
    )


def _json_value(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def local_source_manifest() -> dict[str, str]:
    return build_local_source_manifest(PROJECT_ROOT)


def _local_transfer_invariants(
    experiment: GenerationExperiment | LocalG2Experiment,
) -> dict[str, Any]:
    return _json_value(
        {
            "experiment_class": type(experiment).__name__,
            "dataset_size": experiment.size,
            "user_sample": (
                None if experiment.user_sample is None else experiment.user_sample.name
            ),
            "event_type_filter": experiment.event_type_filter,
            "min_item_interactions_per_item": (
                experiment.min_item_interactions_per_item
            ),
            "drop_unmapped_items": experiment.drop_unmapped_items,
            "validation_interval_seconds": experiment.validation_interval_seconds,
            "day_range": (
                None if experiment.day_range is None else asdict(experiment.day_range)
            ),
            "batch_size": experiment.dataloader.batch_size,
            "physical_batch_size": experiment.dataloader.batch_size,
            "gradient_accumulation_steps": (
                experiment.dataloader.gradient_accumulation_steps
            ),
            "effective_batch_size": experiment.dataloader.effective_batch_size,
            "model_dim": experiment.model_dim,
            "item_embedding_dim": (
                experiment.model_dim
                if experiment.item_embedding_dim is None
                else experiment.item_embedding_dim
            ),
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
            "exclude_own_group_negatives": (experiment.exclude_own_group_negatives),
            "dense_random_negative_scores": (experiment.dense_random_negative_scores),
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
            "exclude_seen_from_evaluation": (experiment.exclude_seen_from_evaluation),
            "restore_best_weights": experiment.restore_best_weights,
            "adaptive_schedule_early_stopping": (
                experiment.adaptive_schedule_early_stopping
            ),
            "transformer": transformer_metadata(experiment.transformer),
            "lr_schedule": asdict(experiment.lr_schedule),
        }
    )


def persisted_job_contract(compiled: CompiledJob) -> dict[str, Any]:
    validate_compiled_job(compiled)
    contract = compiled.to_contract(approved_manifest())
    if compiled.approved.stage == "official":
        return contract
    experiment = build_local_experiment(compiled)
    return contract | {
        "local_implementation": {
            "sources": local_source_manifest(),
            "training": _json_value(
                {
                    "max_epochs": experiment.num_epochs,
                    "initializer_std": experiment.initializer_std,
                    "weight_decay": experiment.weight_decay,
                    "runtime_dtype": str(experiment.runtime.dtype),
                    "runtime_compile": experiment.runtime.compile,
                    "layer_family": getattr(experiment, "layer_family", "g1_control"),
                    "loss_kind": getattr(experiment, "loss_kind", "sampled_softmax"),
                    "gbce_t": getattr(experiment, "gbce_t", None),
                    "training_reverse_position_offset": getattr(
                        experiment, "training_reverse_position_offset", 0
                    ),
                }
            ),
            "transfer_invariants": _local_transfer_invariants(experiment),
        }
    }


def encode_compiled_job(compiled: CompiledJob) -> str:
    contract = compiled.to_contract(approved_manifest())
    payload = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(payload).decode()


def compiled_job_from_environment() -> CompiledJob:
    encoded = os.environ.get(JOB_ENVIRONMENT)
    if not encoded:
        raise RuntimeError(f"{JOB_ENVIRONMENT} is required")
    try:
        contract = json.loads(base64.urlsafe_b64decode(encoded).decode())
    except (ValueError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{JOB_ENVIRONMENT} is invalid") from error
    manifest = approved_manifest()
    if contract.get("manifest_sha256") != manifest.sha256:
        raise RuntimeError("compiled job references a different approved manifest")
    raw_job = contract.get("job")
    parameters = contract.get("parameters")
    if not isinstance(raw_job, dict) or not isinstance(parameters, dict):
        raise RuntimeError("compiled job contract is incomplete")
    matching = [job for job in manifest.jobs if job.to_dict() == raw_job]
    if len(matching) != 1:
        raise RuntimeError("compiled job identity is not approved")
    compiled = CompiledJob(matching[0], parameters)
    validate_compiled_job(compiled)
    return compiled


def write_job_contract(compiled: CompiledJob, root: Path) -> Path:
    destination = root / compiled.approved.run_name / "g2_job.json"
    content = (
        json.dumps(persisted_job_contract(compiled), indent=2, sort_keys=True) + "\n"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.read_text() != content:
        raise RuntimeError(f"existing G2 contract differs: {destination}")
    destination.write_text(content)
    return destination


def emit_rows(path: Path) -> list[str]:
    rows = []
    for compiled in load_compiled_jobs(path):
        kind = "official" if compiled.approved.stage == "official" else "local"
        rows.append(
            "\t".join((kind, compiled.approved.run_name, encode_compiled_job(compiled)))
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    arguments = parser.parse_args()
    print("\n".join(emit_rows(arguments.manifest)))


if __name__ == "__main__":
    main()
