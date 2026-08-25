import json
from types import SimpleNamespace

import pytest

from experiments.g2_esasrec.analysis import benchmark
from experiments.g2_esasrec.analysis.evidence import METRICS

from experiments.g2_esasrec.launchers.optuna_workflow import (
    OptunaStudyWorkflow,
    compile_official_jobs,
)
from experiments.g2_esasrec.launchers.compiled import persisted_job_contract
from experiments.g2_esasrec.protocol.manifest import CompiledJob
from experiments.g2_esasrec.protocol.manifest import (
    approved_manifest,
    load_compiled_jobs,
)
from experiments.g2_esasrec.protocol.optuna_driver import G2OptunaDriver


def _write_artifact(logs, compiled, recall):
    job = compiled.approved
    directory = logs / job.run_name
    directory.mkdir(parents=True)
    metrics = {metric: 0.1 for metric in METRICS} | {
        "recall@100": recall,
        "num_users": 3414,
    }
    contract = persisted_job_contract(compiled)
    local = contract["local_implementation"]
    training = local["training"]
    (directory / "final_metrics.json").write_text(json.dumps(metrics))
    (directory / "training_metadata.json").write_text(
        json.dumps(
            {
                "dataset_size": "50m",
                "seed": job.seed,
                "selection_resolved": True,
                "best_epoch": 2,
                "batch_size": compiled.parameters["batch_size"],
                "embedding_learning_rate": compiled.parameters[
                    "embedding_learning_rate"
                ],
                "deep_learning_rate": compiled.parameters["deep_learning_rate"],
                "max_epochs": training["max_epochs"],
                "initializer_std": training["initializer_std"],
                "weight_decay": training["weight_decay"],
                "runtime_dtype": training["runtime_dtype"],
                "runtime_compile": training["runtime_compile"],
                "transfer_invariants": local["transfer_invariants"],
            }
        )
    )
    (directory / "cost_metrics.json").write_text(
        json.dumps(
            {
                "params_total": 1000,
                "params_deep": 500,
                "training_seconds": 2,
                "wall_seconds": 3,
                "median_train_epoch_seconds": 1,
                "peak_memory_gb": 1,
                "targets_per_second": 100,
                "best_epoch": 2,
            }
        )
    )
    (directory / "g2_job.json").write_text(json.dumps(contract))


def _write_benchmark(path, compiled, logs_root):
    user_ids = list(benchmark.QUERY_USER_IDS)
    metrics = {"recall@100": 0.2, "ndcg@100": 0.1}
    metadata = {
        "best_epoch": 6,
        "optimizer_steps": 1376,
        "selection_resolved": True,
    }
    diagnostic_run_name = (
        f"g2_selected_benchmark_{compiled.approved.run_name}_"
        "deterministic_reproduction_offline"
    )
    for run_name in (compiled.approved.run_name, diagnostic_run_name):
        directory = logs_root / run_name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "final_metrics.json").write_text(json.dumps(metrics))
        (directory / "training_metadata.json").write_text(json.dumps(metadata))
    metrics_sha256 = benchmark.canonical_json_sha256(metrics)
    metadata_sha256 = benchmark.canonical_json_sha256(metadata)
    document = {
        "run_name": compiled.approved.run_name,
        "device_name": "NVIDIA A100-SXM4-80GB",
        "catalog_size": 33148,
        "protocol": "one A100, fixed full catalog and query batch",
        "warmup_iterations": 20,
        "timed_iterations": 100,
        "query_batch_size": 256,
        "latency_p50_seconds": 0.01,
        "latency_p95_seconds": 0.02,
        "queries_per_second": 25600,
        "catalog_sha256": (
            "fa5acc91da974d077fb8c870ea4d4fc776efebd2ea374d8c3b0d23977ea1c831"
        ),
        "catalog_source": "full mapped pre-split catalog",
        "catalog_encoding_timed": False,
        "autocast_dtype": "torch.bfloat16",
        "item_representation_dtype": "torch.float32",
        "query_representation_dtypes": ["torch.bfloat16"],
        "ranking_dtype": "torch.float32",
        "top_k": 100,
        "query_selection": {
            "algorithm": "blake2b-64(seed:user_id)",
            "seed": 42,
            "population_size": 3414,
            "population_user_ids_sha256": benchmark.QUERY_POPULATION_SHA256,
            "user_ids": user_ids,
            "user_ids_sha256": benchmark.QUERY_USER_IDS_SHA256,
            "packed_query_payload_sha256": (
                benchmark.QUERY_PAYLOAD_SHA256_BY_MAX_SEQ_LEN[
                    persisted_job_contract(compiled)["local_implementation"][
                        "transfer_invariants"
                    ]["max_seq_len"]
                ]
            ),
        },
        "weights": {
            "source": "validation_selected_recipe_reproduction_restored_weights",
            "seed": compiled.approved.seed,
            "initializer_std": 0.02,
            "optimizer_steps": 1376,
            "best_epoch": 6,
            "state_sha256": "b" * 64,
        },
        "basis": {
            "kind": "deterministic_selected_recipe_reproduction",
            "selected_run_name": compiled.approved.run_name,
            "selected_job_contract": persisted_job_contract(compiled),
            "selected_metrics_sha256": metrics_sha256,
            "diagnostic_metrics_sha256": metrics_sha256,
            "selected_training_metadata_sha256": metadata_sha256,
            "diagnostic_training_metadata_sha256": metadata_sha256,
        },
        "diagnostic_run_name": diagnostic_run_name,
    }
    path.write_text(json.dumps(document))


def test_optuna_workflow_queues_observes_and_resumes_without_reemission(tmp_path):
    logs = tmp_path / "logs"
    driver = G2OptunaDriver(tmp_path / "study.sqlite3", seed=9)
    submissions = []

    def submit(compiled):
        submissions.append(compiled.approved.id)
        _write_artifact(logs, compiled, recall=0.2 + len(submissions) / 1000)

    compiled_path = tmp_path / "compiled.json"
    workflow = OptunaStudyWorkflow(
        driver,
        logs_root=logs,
        compiled_path=compiled_path,
        submit=submit,
    )

    assert workflow.advance(driver.next_control) == 20
    assert len(submissions) == 20
    assert len(load_compiled_jobs(compiled_path)) == 20
    assert workflow.advance(driver.next_control) == 0
    assert len(submissions) == 20


def test_fixed_jobs_submit_as_one_batch_and_partial_artifacts_fail_closed(tmp_path):
    logs = tmp_path / "logs"
    driver = G2OptunaDriver(tmp_path / "study.sqlite3")
    jobs = approved_manifest().jobs_for_stage("control_tuning")[:2]
    compiled = tuple(
        CompiledJob(
            job,
            {
                "batch_size": 512,
                "embedding_learning_rate": 0.01,
                "deep_learning_rate": 0.02,
            },
        )
        for job in jobs
    )
    batches = []

    def submit_batch(items):
        batches.append([item.approved.id for item in items])
        for index, item in enumerate(items):
            _write_artifact(logs, item, 0.2 + index / 100)

    workflow = OptunaStudyWorkflow(
        driver,
        logs_root=logs,
        compiled_path=tmp_path / "compiled.json",
        submit=lambda item: submit_batch((item,)),
        submit_batch=submit_batch,
    )

    assert len(workflow.run_compiled(compiled)) == 2
    assert batches == [[job.id for job in jobs]]

    partial = approved_manifest().jobs_for_stage("control_tuning")[2]
    partial_directory = logs / partial.run_name
    partial_directory.mkdir()
    (partial_directory / "sweep.log").write_text("failed")
    partial_compiled = CompiledJob(partial, compiled[0].parameters)
    with pytest.raises(ValueError, match="partial terminal artifact"):
        workflow.run_compiled((partial_compiled,))


def test_official_jobs_pin_rectools_version() -> None:
    jobs = compile_official_jobs()

    assert len(jobs) == 3
    assert {job.parameters["rectools_version"] for job in jobs} == {"0.19.0"}


def test_independent_studies_advance_on_one_shared_queue_runway(tmp_path):
    logs = tmp_path / "logs"
    jobs = approved_manifest().jobs_for_stage("control_tuning")[:2]
    compiled = [
        CompiledJob(
            job,
            {
                "batch_size": 512,
                "embedding_learning_rate": 0.01,
                "deep_learning_rate": 0.02,
            },
        )
        for job in jobs
    ]
    pending = {"first": [compiled[0]], "second": [compiled[1]]}
    records = []
    batches = []
    driver = SimpleNamespace(
        record_observation=lambda job, artifact: records.append(job.approved.id)
    )

    def submit_batch(items):
        batches.append([item.approved.id for item in items])
        for item in items:
            _write_artifact(logs, item, 0.2)

    workflow = OptunaStudyWorkflow(
        driver,
        logs_root=logs,
        compiled_path=tmp_path / "compiled.json",
        submit=lambda item: submit_batch((item,)),
        submit_batch=submit_batch,
    )
    asks = {
        name: (lambda name=name: pending[name].pop() if pending[name] else None)
        for name in pending
    }

    assert workflow.advance_parallel(asks) == 2
    assert batches == [[job.id for job in jobs]]
    assert records == [job.id for job in jobs]


def test_selected_benchmark_is_submitted_once_without_becoming_a_compiled_job(
    tmp_path,
):
    job = approved_manifest().jobs_for_stage("control_tuning")[0]
    compiled = CompiledJob(
        job,
        {
            "batch_size": 512,
            "embedding_learning_rate": 0.01,
            "deep_learning_rate": 0.02,
        },
    )
    compiled_path = tmp_path / "compiled.json"
    destination = tmp_path / "selected_benchmark_native50m.json"
    submissions = []
    workflow = OptunaStudyWorkflow(
        SimpleNamespace(),
        logs_root=tmp_path / "logs",
        compiled_path=compiled_path,
        submit=lambda item: None,
    )

    def submit_benchmark(item, path):
        submissions.append(item)
        _write_benchmark(path, item, workflow.logs_root)

    result = workflow.run_selected_benchmark(compiled, destination, submit_benchmark)
    resumed = workflow.run_selected_benchmark(compiled, destination, submit_benchmark)

    assert result == resumed
    assert submissions == [compiled]
    assert not compiled_path.exists()
