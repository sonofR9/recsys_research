import json
import hashlib
from pathlib import Path

import pytest

from experiments.g2_esasrec.protocol.manifest import (
    CompiledJob,
    MAX_RUNS,
    approved_manifest,
    load_compiled_jobs,
    validate_compiled_job,
    validate_approved_manifest,
)
from experiments.g2_esasrec.protocol.local_provenance import (
    LOCAL_EXECUTION_SOURCES,
    LOCAL_IMPLEMENTATION_SOURCES,
    local_source_manifest,
)


EXPECTED_LOCAL_EXECUTION_SOURCES = {
    "entry_and_protocol": {
        "dcn/__init__.py",
        "dcn/main.py",
        "experiments/g2_esasrec/__init__.py",
        "experiments/g2_esasrec/configs/__init__.py",
        "experiments/generation_protocol.py",
        "experiments/g2_esasrec/configs/local.py",
        "experiments/g2_esasrec/launchers/compiled.py",
        "experiments/g2_esasrec/launchers/cost.py",
        "experiments/g2_esasrec/launchers/run_local.py",
        "experiments/g2_esasrec/launchers/__init__.py",
        "experiments/g2_esasrec/protocol/__init__.py",
        "experiments/g2_esasrec/protocol/local_provenance.py",
        "experiments/g2_esasrec/protocol/manifest.py",
    },
    "configuration": {
        "dcn/config/__init__.py",
        "dcn/config/experiment.py",
        "dcn/config/generation.py",
        "dcn/config/networks.py",
        "dcn/config/ranking.py",
        "dcn/config/retrieval.py",
        "dcn/config/sasrec.py",
        "dcn/config/semantic.py",
        "dcn/config/sequence.py",
        "dcn/config/settings.py",
        "dcn/config/yambda.py",
        "dcn/config/yambda_base.py",
        "dcn/training_metadata.py",
    },
    "data_and_dataset": {
        "data/__init__.py",
        "data/counters/__init__.py",
        "data/counters/config.py",
        "data/counters/counter.py",
        "data/preprocessing.py",
        "data/split_by_day.py",
        "data/utils.py",
        "dcn/data/dataset.py",
        "dcn/data/__init__.py",
        "dcn/data/dataset_manager.py",
        "dcn/data/features.py",
        "dcn/data/packed.py",
        "dcn/data/sequence_dataset.py",
        "dcn/datasets/base.py",
        "dcn/datasets/__init__.py",
        "dcn/datasets/remap.py",
        "dcn/datasets/yambda.py",
    },
    "model_targets_and_loss": {
        "dcn/models/criterions.py",
        "dcn/models/__init__.py",
        "dcn/models/history_tokens.py",
        "dcn/models/loss_wrapper.py",
        "dcn/models/multi_head_network.py",
        "dcn/models/semantic_constraint.py",
        "dcn/models/sequence_retrieval.py",
        "dcn/models/sequence_targets.py",
        "dcn/models/two_tower.py",
        "dcn/models/token_generation.py",
        "dcn/nn/crossnet.py",
        "dcn/nn/dcnv2.py",
        "dcn/nn/densenet.py",
        "dcn/nn/esasrec.py",
        "dcn/nn/__init__.py",
        "dcn/nn/ffn.py",
        "dcn/nn/history_encoder.py",
        "dcn/nn/layer_item_features.py",
        "dcn/nn/layer_registry.py",
        "dcn/nn/multi_task_embedding.py",
        "dcn/nn/ple.py",
        "dcn/nn/precomputed_embeddings.py",
        "dcn/nn/resnet.py",
        "dcn/nn/sampled_softmax.py",
        "dcn/nn/semantic_embedding.py",
        "dcn/nn/transformer.py",
        "dcn/nn/types.py",
        "dcn/semantic/__init__.py",
        "dcn/semantic/artifacts.py",
        "dcn/semantic/codes.py",
        "dcn/semantic/residual_kmeans.py",
        "dcn/semantic/rq_vae.py",
        "dcn/semantic/trie.py",
    },
    "evaluation": {
        "dcn/eval/__init__.py",
        "dcn/eval/base.py",
        "dcn/eval/callback.py",
        "dcn/eval/generation.py",
        "dcn/eval/pairwise.py",
        "dcn/eval/ranking_metrics.py",
        "dcn/eval/true_metric.py",
    },
    "training_and_optimizer": {
        "dcn/training/__init__.py",
        "dcn/training/combined_optimizer.py",
        "dcn/training/epoch_trainer.py",
        "dcn/training/optimizer_groups.py",
        "dcn/training/pretrain_callback.py",
        "dcn/training/trainer.py",
        "neuralrec/__init__.py",
        "neuralrec/data/dataloader.py",
        "neuralrec/data/transforms.py",
        "neuralrec/data/__init__.py",
        "neuralrec/nn/autocast.py",
        "neuralrec/nn/__init__.py",
        "neuralrec/nn/metrics/__init__.py",
        "neuralrec/nn/metrics/base.py",
        "neuralrec/nn/metrics/classification.py",
        "neuralrec/nn/metrics/recall.py",
        "neuralrec/nn/metrics/regression.py",
        "neuralrec/run/callbacks/__init__.py",
        "neuralrec/run/callbacks/base.py",
        "neuralrec/run/callbacks/best_weights.py",
        "neuralrec/run/callbacks/checkpoint.py",
        "neuralrec/run/callbacks/clipping.py",
        "neuralrec/run/callbacks/early_stopping.py",
        "neuralrec/run/callbacks/logging.py",
        "neuralrec/run/callbacks/lr_schedule.py",
        "neuralrec/run/callbacks/resources.py",
        "neuralrec/run/callbacks/tensorboard.py",
        "neuralrec/run/callbacks/validation.py",
        "neuralrec/run/callbacks/wandb.py",
        "neuralrec/run/train.py",
        "neuralrec/run/__init__.py",
        "neuralrec/utils/__init__.py",
        "neuralrec/utils/stateful.py",
        "neuralrec/utils/utils.py",
    },
    "runtime_support": {
        "utils/__init__.py",
        "utils/global_config.py",
        "utils/locks.py",
    },
}


def test_local_provenance_exactly_covers_the_executed_g2_path() -> None:
    assert {
        category: set(sources) for category, sources in LOCAL_EXECUTION_SOURCES.items()
    } == EXPECTED_LOCAL_EXECUTION_SOURCES
    assert set(LOCAL_IMPLEMENTATION_SOURCES) == set().union(
        *EXPECTED_LOCAL_EXECUTION_SOURCES.values()
    )


def test_local_provenance_hashes_every_declared_source_exactly() -> None:
    project_root = Path(__file__).resolve().parents[3]
    manifest = local_source_manifest(project_root)

    assert list(manifest) == list(LOCAL_IMPLEMENTATION_SOURCES)
    assert manifest == {
        relative: hashlib.sha256((project_root / relative).read_bytes()).hexdigest()
        for relative in LOCAL_IMPLEMENTATION_SOURCES
    }


def test_approved_manifest_accounts_for_every_run_in_the_135_run_budget() -> None:
    manifest = approved_manifest()

    assert len(manifest.jobs) == MAX_RUNS == 135
    assert manifest.stage_counts == {
        "control_tuning": 20,
        "control_repeats": 10,
        "component_tuning": 72,
        "mixed_tuning": 12,
        "official": 3,
        "lr_boundary": 14,
        "reversal_confirmation": 4,
    }
    assert sum(job.conditional for job in manifest.jobs) == 19
    assert manifest.jobs_for_stage("closing") == []
    assert len({job.id for job in manifest.jobs}) == MAX_RUNS
    assert len({job.run_name for job in manifest.jobs}) == MAX_RUNS


def test_manifest_fixes_the_approved_seeds_and_forced_mixed_anchors() -> None:
    manifest = approved_manifest()

    repeats = manifest.jobs_for_stage("control_repeats")
    official = manifest.jobs_for_stage("official")
    mixed = manifest.jobs_for_stage("mixed_tuning")

    assert [job.seed for job in repeats] == list(range(42, 52))
    assert [job.seed for job in repeats if job.conditional] == [42]
    assert [job.seed for job in official] == [42, 43, 44]
    assert mixed[0].forced_parameters == {
        "uniform_fraction": 0.6,
        "logq_correction": "none",
    }
    assert mixed[1].forced_parameters == {
        "uniform_fraction": 0.6,
        "logq_correction": "yi2019",
    }


def test_manifest_rejects_changed_or_unknown_job_identity() -> None:
    document = approved_manifest().to_dict()
    document["jobs"][0]["run_name"] = "incidental-run"

    with pytest.raises(ValueError, match="approved manifest"):
        validate_approved_manifest(document)


def test_compiled_jobs_fail_closed_on_unknown_duplicate_or_unresolved_jobs(
    tmp_path,
) -> None:
    manifest = approved_manifest()
    job = manifest.jobs_for_stage("control_tuning")[0]
    valid = {
        "manifest_sha256": manifest.sha256,
        "jobs": [
            {
                "id": job.id,
                "run_name": job.run_name,
                "parameters": {
                    "batch_size": 512,
                    "embedding_learning_rate": 0.01,
                    "deep_learning_rate": 0.01,
                },
            }
        ],
    }
    path = tmp_path / "compiled.json"
    path.write_text(json.dumps(valid))
    assert load_compiled_jobs(path)[0].approved == job

    valid["jobs"].append(valid["jobs"][0])
    path.write_text(json.dumps(valid))
    with pytest.raises(ValueError, match="duplicate"):
        load_compiled_jobs(path)

    valid["jobs"] = [{"id": "unknown", "run_name": "unknown", "parameters": {}}]
    path.write_text(json.dumps(valid))
    with pytest.raises(ValueError, match="unknown"):
        load_compiled_jobs(path)

    valid["jobs"] = [{"id": job.id, "run_name": job.run_name}]
    path.write_text(json.dumps(valid))
    with pytest.raises(ValueError, match="parameters"):
        load_compiled_jobs(path)


def test_compiled_jobs_enforce_the_approved_parameter_domains(tmp_path) -> None:
    manifest = approved_manifest()
    job = manifest.jobs_for_stage("mixed_tuning")[0]
    document = {
        "manifest_sha256": manifest.sha256,
        "jobs": [
            {
                "id": job.id,
                "run_name": job.run_name,
                "parameters": {
                    "batch_size": 128,
                    "embedding_learning_rate": 0.001,
                    "deep_learning_rate": 0.001,
                    "ligr_multiplier": 4,
                    "uniform_fraction": 0.7,
                    "logq_correction": "none",
                },
            }
        ],
    }
    path = tmp_path / "compiled.json"
    path.write_text(json.dumps(document))

    with pytest.raises(ValueError, match="forced"):
        load_compiled_jobs(path)


def test_compiled_jobs_enforce_exact_stage_schemas_and_finite_ranges() -> None:
    manifest = approved_manifest()
    control = manifest.jobs_for_stage("control_tuning")[1]
    parameters = {
        "batch_size": 512,
        "embedding_learning_rate": 0.01,
        "deep_learning_rate": 0.02,
    }

    validate_compiled_job(CompiledJob(control, parameters))
    with pytest.raises(ValueError, match="unexpected parameters"):
        validate_compiled_job(CompiledJob(control, parameters | {"gbce_t": 0.75}))
    with pytest.raises(ValueError, match="approved search range"):
        validate_compiled_job(
            CompiledJob(control, parameters | {"deep_learning_rate": 0.384})
        )
    with pytest.raises(ValueError, match="batch_size"):
        validate_compiled_job(CompiledJob(control, parameters | {"batch_size": 129}))


def test_conditional_jobs_are_bound_to_the_verified_source_family() -> None:
    manifest = approved_manifest()
    boundary = next(
        job
        for job in manifest.jobs_for_stage("lr_boundary")
        if job.method == "ligr_sampled_softmax"
    )
    source = next(
        job
        for job in manifest.jobs_for_stage("component_tuning")
        if job.method == "ligr_sampled_softmax" and job.trial == 1
    )
    parameters = {
        "builder": "component",
        "method": "ligr_sampled_softmax",
        "source_job_id": source.id,
        "selected_control_job_id": "control_tuning:control_trial_01",
        "batch_size": 512,
        "embedding_learning_rate": 0.01,
        "deep_learning_rate": 0.384,
        "ligr_multiplier": 4,
    }

    validate_compiled_job(CompiledJob(boundary, parameters))
    wrong = parameters | {
        "method": "standard_sampled_softmax",
        "source_job_id": "component_tuning:standard_sampled_softmax_trial_01",
    }
    with pytest.raises(ValueError, match="conditional family"):
        validate_compiled_job(CompiledJob(boundary, wrong))


def test_capacity_dependent_components_require_an_approved_ligr_source() -> None:
    manifest = approved_manifest()
    job = next(
        job
        for job in manifest.jobs_for_stage("component_tuning")
        if job.method == "ligr_gbce" and job.trial == 1
    )
    source = next(
        job
        for job in manifest.jobs_for_stage("component_tuning")
        if job.method == "ligr_sampled_softmax" and job.trial == 1
    )
    parameters = {
        "batch_size": 512,
        "embedding_learning_rate": 0.01,
        "deep_learning_rate": 0.02,
        "selected_control_job_id": "control_tuning:control_trial_01",
        "source_job_id": source.id,
        "ligr_multiplier": 6,
        "gbce_t": 0.75,
    }

    validate_compiled_job(CompiledJob(job, parameters))
    with pytest.raises(ValueError, match="prerequisite job ID is missing"):
        validate_compiled_job(
            CompiledJob(
                job, {k: v for k, v in parameters.items() if k != "source_job_id"}
            )
        )
    with pytest.raises(ValueError, match="LiGR capacity"):
        validate_compiled_job(
            CompiledJob(
                job,
                parameters
                | {
                    "source_job_id": (
                        "component_tuning:standard_sampled_softmax_trial_01"
                    )
                },
            )
        )
