from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import MethodType
from typing import Any

from experiments.g2_esasrec.analysis.benchmark import (
    CATALOG_SIZE,
    canonical_json_sha256,
    run_production_benchmark,
    write_benchmark,
)
from experiments.g2_esasrec.launchers.compiled import (
    build_local_experiment,
    persisted_job_contract,
)
from experiments.g2_esasrec.protocol.manifest import CompiledJob, validate_compiled_job


def build_selected_benchmark_experiment(
    compiled: CompiledJob, destination: Path
) -> Any:
    validate_compiled_job(compiled)
    if compiled.approved.stage == "official":
        raise ValueError("selected-model benchmark requires a local architecture")
    selected_run_name = compiled.approved.run_name
    experiment = build_local_experiment(compiled)
    experiment = replace(
        experiment,
        run_name=(
            f"g2_selected_benchmark_{selected_run_name}_"
            "deterministic_reproduction_offline"
        ),
        checkpointing=replace(
            experiment.checkpointing,
            enabled=False,
            load_checkpoint=False,
        ),
        logging=replace(experiment.logging, enable_predictions=False),
    )
    original_finish = experiment.finish

    def finish(owner: Any, runner: Any) -> None:
        original_finish(runner)
        selected_metrics = json.loads(
            (
                Path(owner.base_path)
                / "logs"
                / selected_run_name
                / "final_metrics.json"
            ).read_text()
        )
        diagnostic_metrics = json.loads(
            (
                Path(owner.base_path) / "logs" / owner.run_name / "final_metrics.json"
            ).read_text()
        )
        if diagnostic_metrics != selected_metrics:
            raise ValueError(
                "selected-model benchmark retraining did not reproduce selected metrics"
            )
        selected_metadata = json.loads(
            (
                Path(owner.base_path)
                / "logs"
                / selected_run_name
                / "training_metadata.json"
            ).read_text()
        )
        diagnostic_metadata = json.loads(
            (
                Path(owner.base_path)
                / "logs"
                / owner.run_name
                / "training_metadata.json"
            ).read_text()
        )
        if diagnostic_metadata != selected_metadata:
            raise ValueError(
                "selected-model benchmark retraining did not reproduce training metadata"
            )
        metrics_sha256 = canonical_json_sha256(selected_metrics)
        metadata_sha256 = canonical_json_sha256(selected_metadata)
        best_epoch = owner.callbacks.best_weights.best_epoch
        if best_epoch is None:
            raise ValueError("selected-model benchmark has no restored best epoch")
        result = run_production_benchmark(
            owner.base_model,
            query_batches=list(owner.cutoff_query_loader),
            item_batch=owner.true_metric.item_batch,
            user_column=owner.user_column,
            item_id_column=owner.item_id_column,
            device=runner.device,
            initialization_seed=compiled.approved.seed,
            initializer_std=owner.initializer_std,
            eligible_user_ids=set(owner.true_metric.relevance),
            max_seq_len=owner.max_seq_len,
            weight_source=(
                "validation_selected_recipe_reproduction_restored_weights"
            ),
            optimizer_steps=runner.global_step,
            best_epoch=best_epoch + 1,
        )
        result["basis"] = {
            "kind": "deterministic_selected_recipe_reproduction",
            "selected_run_name": selected_run_name,
            "selected_job_contract": persisted_job_contract(compiled),
            "selected_metrics_sha256": metrics_sha256,
            "diagnostic_metrics_sha256": metrics_sha256,
            "selected_training_metadata_sha256": metadata_sha256,
            "diagnostic_training_metadata_sha256": metadata_sha256,
        }
        result["diagnostic_run_name"] = owner.run_name
        write_benchmark(
            result,
            run_name=selected_run_name,
            catalog_size=CATALOG_SIZE,
            destination=destination,
        )

    experiment.finish = MethodType(finish, experiment)
    return experiment
