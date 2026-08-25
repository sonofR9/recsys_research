import json

import pytest
import torch
from torch import nn

from dcn.data.features import FeatureValues
from dcn.eval import build_catalog_batch
from experiments.g2_esasrec.analysis import benchmark
from experiments.g2_esasrec.launchers.compiled import persisted_job_contract
from experiments.g2_esasrec.launchers.selected_benchmark import (
    build_selected_benchmark_experiment,
)
from experiments.g2_esasrec.protocol.manifest import CompiledJob, approved_manifest


def _query_batch(user_ids: list[int]) -> dict:
    values = torch.tensor(user_ids, dtype=torch.long)
    offsets = torch.arange(len(user_ids) + 1, dtype=torch.long)
    return {
        "int_columns": {
            "uid": FeatureValues(values, offsets),
            "compact_item_id": FeatureValues(values.remainder(33148) + 1, offsets),
        },
        "float_columns": {},
        "timestamp": torch.arange(len(user_ids), dtype=torch.long),
        "cumulative_lens": offsets,
    }


class _FreshModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.01))

    def encode_cutoff_queries(self, batch: dict) -> torch.Tensor:
        last = batch["cumulative_lens"][1:] - 1
        users = batch["int_columns"]["uid"].dense()[last].float()
        return torch.stack((users * self.scale, users * self.scale + 1), dim=1)

    def encode_items(self, batch: dict) -> torch.Tensor:
        items = batch["int_columns"]["compact_item_id"].dense().float()
        return torch.stack((items * self.scale, items * self.scale + 1), dim=1)


def test_selected_model_benchmark_uses_warmups_and_reports_latency(monkeypatch):
    times = iter([0.0, 0.01, 0.01, 0.03, 0.03, 0.06])
    monkeypatch.setattr(benchmark, "perf_counter", lambda: next(times))
    calls = []

    result = benchmark.benchmark_selected_model(
        lambda: calls.append(1),
        query_batch_size=100,
        device=torch.device("cpu"),
        warmup_iterations=2,
        timed_iterations=3,
        require_a100=False,
    )

    assert len(calls) == 5
    assert result["latency_p50_seconds"] == pytest.approx(0.02)
    assert result["latency_p95_seconds"] == pytest.approx(0.029)
    assert result["queries_per_second"] == pytest.approx(5000.0)


def test_selected_model_benchmark_defaults_match_the_approved_protocol():
    assert benchmark.WARMUP_ITERATIONS == 20
    assert benchmark.TIMED_ITERATIONS == 100


def test_query_batch_selection_is_keyed_by_user_id_not_input_position():
    first = [_query_batch(list(range(1, 181))), _query_batch(list(range(181, 401)))]
    second = [
        _query_batch(list(range(400, 180, -1))),
        _query_batch(list(range(180, 0, -1))),
    ]

    _, first_users = benchmark.select_query_batch_by_user_hash(first, user_column="uid")
    _, second_users = benchmark.select_query_batch_by_user_hash(
        second, user_column="uid"
    )

    assert first_users.tolist() == second_users.tolist()
    assert len(first_users) == 256


def test_fresh_bf16_path_scores_the_full_catalog_and_records_the_contract(
    monkeypatch,
):
    model = _FreshModel()
    query_batches = [_query_batch(list(range(1, 301)))]
    eligible_user_ids = set(range(1, 301))
    query_batch, query_user_ids = benchmark.select_query_batch_by_user_hash(
        query_batches,
        user_column="uid",
        eligible_user_ids=eligible_user_ids,
    )
    monkeypatch.setattr(benchmark, "QUERY_POPULATION_SIZE", 300)
    monkeypatch.setattr(
        benchmark,
        "QUERY_POPULATION_SHA256",
        benchmark._tensor_sha256(torch.arange(1, 301)),
    )
    monkeypatch.setattr(
        benchmark, "QUERY_USER_IDS_SHA256", benchmark._tensor_sha256(query_user_ids)
    )
    monkeypatch.setattr(benchmark, "QUERY_USER_IDS", tuple(query_user_ids.tolist()))
    monkeypatch.setattr(
        benchmark,
        "QUERY_PAYLOAD_SHA256_BY_MAX_SEQ_LEN",
        {100: benchmark._query_payload_sha256(query_batch)},
    )
    result = benchmark.run_production_benchmark(
        model,
        query_batches=query_batches,
        item_batch=build_catalog_batch(
            range(1, 33149), item_id_column="compact_item_id"
        ),
        user_column="uid",
        item_id_column="compact_item_id",
        device=torch.device("cpu"),
        initialization_seed=42,
        initializer_std=0.02,
        eligible_user_ids=eligible_user_ids,
        max_seq_len=100,
        warmup_iterations=0,
        timed_iterations=1,
        require_a100=False,
    )

    assert result["query_batch_size"] == 256
    assert result["query_selection"] == {
        "algorithm": "blake2b-64(seed:user_id)",
        "seed": 42,
        "population_size": 300,
        "population_user_ids_sha256": result["query_selection"][
            "population_user_ids_sha256"
        ],
        "user_ids": result["query_selection"]["user_ids"],
        "user_ids_sha256": result["query_selection"]["user_ids_sha256"],
        "packed_query_payload_sha256": result["query_selection"][
            "packed_query_payload_sha256"
        ],
    }
    assert len(result["query_selection"]["user_ids"]) == 256
    assert result["catalog_sha256"] == (
        "fa5acc91da974d077fb8c870ea4d4fc776efebd2ea374d8c3b0d23977ea1c831"
    )
    assert result["weights"] == {
        "source": "fresh_initialized_no_checkpoint",
        "seed": 42,
        "initializer_std": 0.02,
        "optimizer_steps": 0,
        "state_sha256": result["weights"]["state_sha256"],
    }
    assert result["autocast_dtype"] == "torch.bfloat16"
    assert result["item_representation_dtype"] == "torch.float32"
    assert result["query_representation_dtypes"] == ["torch.float32"]
    assert result["ranking_dtype"] == "torch.float32"
    assert result["top_k"] == 100


def test_selected_benchmark_builds_a_deterministic_reproduction_identity(
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

    experiment = build_selected_benchmark_experiment(
        compiled, tmp_path / "selected_benchmark_native50m.json"
    )

    assert experiment.run_name == (
        f"g2_selected_benchmark_{job.run_name}_deterministic_reproduction_offline"
    )
    assert experiment.seed == job.seed
    assert experiment.num_epochs == 20
    assert experiment.checkpointing.enabled is False
    assert experiment.checkpointing.load_checkpoint is False
    assert experiment.run_name not in {
        approved.run_name for approved in approved_manifest().jobs
    }


def test_final_report_benchmark_evidence_is_exact_and_selected(tmp_path):
    path = tmp_path / "benchmark.json"
    logs_root = tmp_path / "logs"
    job = next(
        candidate
        for candidate in approved_manifest().jobs_for_stage("component_tuning")
        if candidate.method == "ligr_sampled_softmax" and candidate.trial == 8
    )
    compiled = CompiledJob(
        job,
        {
            "batch_size": 128,
            "embedding_learning_rate": 0.01,
            "deep_learning_rate": 0.02,
            "ligr_multiplier": 4,
            "selected_control_job_id": "control_tuning:control_trial_01",
        },
    )
    user_ids = list(benchmark.QUERY_USER_IDS)
    user_ids_sha256 = benchmark.QUERY_USER_IDS_SHA256
    selected_metrics = {"recall@100": 0.2, "ndcg@100": 0.1}
    selected_metadata = {
        "best_epoch": 6,
        "optimizer_steps": 1376,
        "selection_resolved": True,
    }
    selected_directory = logs_root / job.run_name
    selected_directory.mkdir(parents=True)
    (selected_directory / "final_metrics.json").write_text(
        json.dumps(selected_metrics)
    )
    (selected_directory / "training_metadata.json").write_text(
        json.dumps(selected_metadata)
    )
    diagnostic_run_name = (
        f"g2_selected_benchmark_{job.run_name}_deterministic_reproduction_offline"
    )
    diagnostic_directory = logs_root / diagnostic_run_name
    diagnostic_directory.mkdir(parents=True)
    (diagnostic_directory / "final_metrics.json").write_text(
        json.dumps(selected_metrics)
    )
    (diagnostic_directory / "training_metadata.json").write_text(
        json.dumps(selected_metadata)
    )
    metrics_sha256 = benchmark.canonical_json_sha256(selected_metrics)
    metadata_sha256 = benchmark.canonical_json_sha256(selected_metadata)
    path.write_text(
        json.dumps(
            {
                "run_name": job.run_name,
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
                    "population_user_ids_sha256": (benchmark.QUERY_POPULATION_SHA256),
                    "user_ids": user_ids,
                    "user_ids_sha256": user_ids_sha256,
                    "packed_query_payload_sha256": (
                        benchmark.QUERY_PAYLOAD_SHA256_BY_MAX_SEQ_LEN[100]
                    ),
                },
                "weights": {
                    "source": "validation_selected_recipe_reproduction_restored_weights",
                    "seed": 42,
                    "initializer_std": 0.02,
                    "optimizer_steps": 1376,
                    "best_epoch": 6,
                    "state_sha256": "b" * 64,
                },
                "basis": {
                    "kind": "deterministic_selected_recipe_reproduction",
                    "selected_run_name": job.run_name,
                    "selected_job_contract": persisted_job_contract(compiled),
                    "selected_metrics_sha256": metrics_sha256,
                    "diagnostic_metrics_sha256": metrics_sha256,
                    "selected_training_metadata_sha256": metadata_sha256,
                    "diagnostic_training_metadata_sha256": metadata_sha256,
                },
                "diagnostic_run_name": diagnostic_run_name,
            }
        )
    )

    evidence = benchmark.load_selected_benchmark(
        path,
        run_name=job.run_name,
        expected_compiled=compiled,
        logs_root=logs_root,
    )
    assert evidence["catalog_size"] == 33148

    stale_compiled = CompiledJob(
        job,
        {**compiled.parameters, "deep_learning_rate": 0.03},
    )
    with pytest.raises(ValueError, match="selected artifact changed"):
        benchmark.load_selected_benchmark(
            path,
            run_name=job.run_name,
            expected_compiled=stale_compiled,
            logs_root=logs_root,
        )

    selected_metrics["recall@100"] = 0.3
    (selected_directory / "final_metrics.json").write_text(json.dumps(selected_metrics))
    with pytest.raises(ValueError, match="selected metrics"):
        benchmark.load_selected_benchmark(
            path,
            run_name=job.run_name,
            expected_compiled=compiled,
            logs_root=logs_root,
        )
    (selected_directory / "final_metrics.json").write_text(
        json.dumps({"recall@100": 0.2, "ndcg@100": 0.1})
    )
    stale_metadata = selected_metadata | {"optimizer_steps": 99}
    (selected_directory / "training_metadata.json").write_text(
        json.dumps(stale_metadata)
    )
    with pytest.raises(ValueError, match="selected training metadata"):
        benchmark.load_selected_benchmark(
            path,
            run_name=job.run_name,
            expected_compiled=compiled,
            logs_root=logs_root,
        )
    (selected_directory / "training_metadata.json").write_text(
        json.dumps(selected_metadata)
    )

    document = json.loads(path.read_text())
    document["run_name"] = "unselected-run"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="selected run"):
        benchmark.load_selected_benchmark(
            path, run_name=job.run_name, logs_root=logs_root
        )
