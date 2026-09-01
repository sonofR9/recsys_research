from __future__ import annotations

from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import re

import pytest

from experiments.g1_sasrec_item_ids_likes.analysis import (
    rq8_reinvestigation_report,
    rq12_decoder_query_report,
)
from experiments.g1_sasrec_item_ids_likes.analysis.rq8_reinvestigation_candidates import (
    Rq8Candidate,
    make_confirmation_candidate,
    query_initial_candidates,
)


def _metadata(candidate: Rq8Candidate) -> dict[str, object]:
    cls_mode = (
        "none" if candidate.query_method == "standard" else candidate.query_method
    )
    targets = 70
    examples = 10
    tokens = {
        "standard": targets + examples,
        "end_only": targets + 2 * examples,
        "interleaved": 2 * (targets + examples),
    }[candidate.query_method]
    invariants = {
        "experiment_class": "MuTransferGenerationExperiment",
        "mup_base_dim": 16,
        "mup_delta_dim": 32,
        "mup_base_ffn_dim": None,
        "mup_delta_ffn_dim": None,
        "dataset_size": "500m",
        "user_sample": None,
        "event_type_filter": "like",
        "min_item_interactions_per_item": 5,
        "drop_unmapped_items": True,
        "validation_interval_seconds": 604800,
        "day_range": {"start_day": 0, "end_day": 300},
        "batch_size": 1280,
        "physical_batch_size": 1280,
        "gradient_accumulation_steps": 1,
        "effective_batch_size": 1280,
        "model_dim": 64,
        "item_embedding_dim": 64,
        "max_seq_len": 128,
        "window": "next_item",
        "bos": False,
        "cls_token": cls_mode != "none",
        "cls_token_mode": cls_mode,
        "timestamp_delta": "bins",
        "timestamp_combination": "add",
        "timestamp_num_bins": 16,
        "timestamp_bin_semantics_revision": 2,
        "per_layer_item_embeddings": False,
        "negative_sampling": "random",
        "num_in_batch_negatives": 512,
        "logq_correction": "yi2019",
        "random_negative_fraction": 0.5,
        "logq_alpha": 0.01,
        "correct_positive_logq": False,
        "mask_false_negatives": False,
        "exclude_own_group_negatives": False,
        "dense_random_negative_scores": False,
        "eval_ks": [10, 50, 100],
        "eval_max_users": 20000,
        "eval_every_n_epochs": 1,
        "early_stopping_patience": 3,
        "early_stopping_min_delta": 0.0,
        "early_stopping_metric": "recall@100",
        "early_stopping_metric_prefix": "epoch/val_true",
        "selection_k": 100,
        "evaluation_catalog": "all",
        "exclude_seen_from_evaluation": False,
        "restore_best_weights": True,
        "adaptive_schedule_early_stopping": False,
        "lr_schedule_horizon_epochs": 20,
        "transformer": rq8_reinvestigation_report._expected_transformer(candidate),
        "lr_schedule": rq8_reinvestigation_report._expected_schedule(),
    }
    return {
        "training_semantics_revision": 2,
        "dataset_size": "500m",
        "seed": candidate.seed,
        "num_epochs": 20,
        "max_epochs": 20,
        "epochs_trained": 20,
        "best_epoch": 2,
        "stopped_epoch": 20,
        "early_stopped": False,
        "best_epoch_at_cap": False,
        "lr_horizon_complete": True,
        "selection_resolved": True,
        "batch_size": 1280,
        "physical_batch_size": 1280,
        "gradient_accumulation_steps": 1,
        "effective_batch_size": 1280,
        "val_batch_size": 8192,
        "num_workers": 4,
        "prefetch_factor": 4,
        "model_dim": 64,
        "item_embedding_dim": 64,
        "embedding_learning_rate": 0.064,
        "deep_learning_rate": candidate.deep_lr,
        "weight_decay": 0.0,
        "initializer_std": 0.02,
        "runtime_dtype": "torch.bfloat16",
        "runtime_compile": False,
        "gradient_clip_norm": None,
        "negative_sampling": "random",
        "cls_token": cls_mode != "none",
        "cls_token_mode": cls_mode,
        "lr_schedule_horizon_epochs": 20,
        "targets_per_epoch": targets,
        "tokens_per_epoch": tokens,
        "transfer_invariants": invariants,
    }


def _write_artifact(
    logs: Path,
    candidate: Rq8Candidate,
    validation_recall: float,
    *,
    final_recall: float,
    observed_wall_seconds: int = 50,
) -> None:
    directory = logs / candidate.run_name
    directory.mkdir(parents=True, exist_ok=True)
    metadata = _metadata(candidate)
    metrics = {
        "recall@100": final_recall,
        "ndcg@100": final_recall / 2,
        "recall@10": final_recall / 4,
        "ndcg@10": final_recall / 5,
        "coverage@100": 0.5,
        "num_users": 37018,
    }
    start = datetime.now() + timedelta(days=1)
    dataset = logs.parent / "generated/datasets/yambda/500m_like_core5_knownitems"
    dataset.mkdir(parents=True, exist_ok=True)
    for filename in (
        "events.parquet",
        "events_remapped.parquet",
        "item_id_remap.parquet",
    ):
        (dataset / filename).write_text(f"fixture {filename}\n")
    cache_root = logs.parent / "generated/preprocessed/dataset/fixture/sequences"
    caches = {
        "train": cache_root / "train_fixture",
        "val": cache_root / "val_fixture",
        "true_metric_query": cache_root / "true_metric_query_fixture",
    }
    for cache in caches.values():
        cache.mkdir(parents=True, exist_ok=True)
        (cache / "metadata.json").write_text('{"fixture": true}\n')
        (cache / "buckets").mkdir(exist_ok=True)
        (cache / "buckets/bucket_00000.parquet").write_text("fixture bucket\n")
    lines = [
        f"{start:%Y-%m-%d %H:%M:%S},000 - INFO - Prepared stage 'test'",
        f"{start:%Y-%m-%d %H:%M:%S},000 - INFO - Preparing yambda "
        f"(size=500m, user_sample=None, listen_fraction=1.0) in {dataset}",
        *[
            f"{start:%Y-%m-%d %H:%M:%S},000 - INFO - "
            f"Loaded cached user sequences from {cache}"
            for cache in caches.values()
        ],
    ]
    for epoch in range(20):
        recall = validation_recall if epoch == 1 else validation_recall - 0.01
        lines.append(
            f"{start + timedelta(seconds=epoch + 1):%Y-%m-%d %H:%M:%S}.000 | "
            f"INFO - epoch {epoch} finished timing.train_epoch_time=1.000 "
            "timing.val_inference_time=0.100 timing.val_save_time=0.010 "
            f"epoch/val_true.recall@100={recall:.6f} "
            f"epoch/val_true.ndcg@100={recall / 2:.6f}"
        )
    end = start + timedelta(seconds=observed_wall_seconds)
    lines.append(f"{end:%Y-%m-%d %H:%M:%S},000 - INFO - Final metrics ({metrics!r})")
    (directory / "training_metadata.json").write_text(json.dumps(metadata))
    (directory / "final_metrics.json").write_text(json.dumps(metrics))
    (directory / "sweep.log").write_text("\n".join(lines) + "\n")


def _complete_query_surface(logs: Path) -> None:
    for candidate in query_initial_candidates():
        score = 0.20 - abs(candidate.deep_lr - 0.012)
        _write_artifact(logs, candidate, score, final_recall=score)
    for method_index, method in enumerate(("standard", "end_only", "interleaved")):
        winner = next(
            candidate
            for candidate in query_initial_candidates()
            if candidate.query_method == method and candidate.deep_lr == 0.012
        )
        for seed in (43, 44):
            candidate = make_confirmation_candidate(winner, seed)
            _write_artifact(
                logs,
                candidate,
                0.20,
                final_recall=0.10 + method_index / 100 + seed / 100000,
                observed_wall_seconds=50,
            )


def test_generates_separate_reader_tables_and_machine_evidence(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    _complete_query_surface(logs)
    anomaly = make_confirmation_candidate(
        next(
            candidate
            for candidate in query_initial_candidates()
            if candidate.query_method == "interleaved" and candidate.deep_lr == 0.012
        ),
        43,
    )
    _write_artifact(logs, anomaly, 0.20, final_recall=0.12, observed_wall_seconds=100)

    bundle = rq12_decoder_query_report.collect_report_bundle(
        logs, verify_recipe=lambda _directory, _candidate: True
    )

    assert bundle.reader_markdown.count("| query objective |") == 2
    assert bundle.reader_markdown.startswith(
        "## RQ12 — Which decoder-only query-token layout works best?"
    )
    assert "### Candidate-generation quality" in bundle.reader_markdown
    assert "### Training efficiency" in bundle.reader_markdown
    assert re.search(r"\b0\.\d{3}\b", bundle.reader_markdown)
    assert "runs" not in bundle.reader_markdown.lower()
    assert "deep LR" not in bundle.reader_markdown
    assert "g1_rq8_" not in bundle.reader_markdown
    assert "steady-state" in bundle.reader_markdown
    assert "Prepared stage" in bundle.reader_markdown
    assert "2 / 2 / 2" in bundle.reader_markdown
    assert all(
        not line or line.startswith(("#", "|"))
        for line in bundle.reader_markdown.splitlines()
    )
    efficiency_rows = bundle.reader_markdown.split("### Training efficiency\n\n")[
        1
    ].splitlines()
    assert len({line.count("|") for line in efficiency_rows if line}) == 1
    assert bundle.evidence["tuning_ledger"].endswith(
        "scratchpad/rq8_reinvestigation_tuning_500m.md"
    )
    assert len(bundle.evidence["methods"]) == 3
    assert all(len(method["artifacts"]) == 3 for method in bundle.evidence["methods"])
    assert all(
        len(method["all_required_artifacts"]) == 5
        for method in bundle.evidence["methods"]
    )
    assert bundle.evidence["total_required_cost"]["unique_artifact_count"] == 15
    assert bundle.evidence["total_required_cost"][
        "logged_train_validation_seconds"
    ] == pytest.approx(333.0)
    assert bundle.evidence["compatibility"]["config_recipe_verified"]
    assert len(bundle.evidence["compatibility"]["data_fingerprints"]) == 1
    assert len(bundle.evidence["compatibility"]["evaluator_fingerprints"]) == 1
    assert len(bundle.evidence["compatibility"]["base_architecture_fingerprints"]) == 1
    assert len(bundle.evidence["compatibility"]["common_objective_fingerprints"]) == 1
    assert bundle.evidence["compatibility"]["workload"] == {
        "auxiliary_ntp_targets_per_epoch": 0,
        "examples_per_epoch": 10,
        "next_item_targets_per_epoch": 70,
    }
    assert bundle.evidence["timing_anomalies"][0]["run_name"] == anomaly.run_name
    first = bundle.evidence["methods"][0]["artifacts"][0]
    assert first["deep_learning_rate"] == 0.012
    assert "selected_deep_learning_rate" not in first
    assert set(first["sha256"]) == {
        "training_metadata.json",
        "final_metrics.json",
        "sweep.log",
    }
    assert len(first["efficiency_inputs"]["epoch_timings"]) == 20
    assert first["efficiency_outputs"]["steady_state_targets_per_second"] == 70.0
    assert first["dataset_identity"]["dataset_directory"] == (
        "500m_like_core5_knownitems"
    )
    assert set(first["dataset_identity"]["dataset_content_sha256"]) == {
        "events.parquet",
        "events_remapped.parquet",
        "item_id_remap.parquet",
    }
    assert first["dataset_identity"]["sequence_cache_parent"] == "fixture"
    assert first["dataset_identity"]["content_mtime_verified_before_run_start"]
    assert set(first["dataset_identity"]["sequence_cache_content_sha256"]) == {
        "train_fixture",
        "val_fixture",
        "true_metric_query_fixture",
    }


def test_rejects_cross_method_workload_mismatch(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    _complete_query_surface(logs)
    for directory in logs.glob("g1_rq8_query_end_only_*"):
        path = directory / "training_metadata.json"
        metadata = json.loads(path.read_text())
        metadata["targets_per_epoch"] += 1
        metadata["tokens_per_epoch"] += 1
        path.write_text(json.dumps(metadata))

    with pytest.raises(rq12_decoder_query_report.Rq12ReportError, match="workload"):
        rq12_decoder_query_report.collect_report_bundle(
            logs, verify_recipe=lambda _directory, _candidate: True
        )


def test_rejects_dataset_content_that_postdates_training(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    _complete_query_surface(logs)
    content = (
        logs.parent
        / "generated/datasets/yambda/500m_like_core5_knownitems/events.parquet"
    )
    after_training_started = datetime.now() + timedelta(days=2)
    os.utime(content, (after_training_started.timestamp(),) * 2)

    with pytest.raises(
        rq12_decoder_query_report.Rq12ReportError,
        match="provenance file postdates run start",
    ):
        rq12_decoder_query_report.collect_report_bundle(
            logs, verify_recipe=lambda _directory, _candidate: True
        )


def test_rejects_an_artifact_that_fails_the_existing_recipe_verifier(
    tmp_path: Path,
) -> None:
    logs = tmp_path / "logs"
    _complete_query_surface(logs)

    with pytest.raises(
        rq12_decoder_query_report.Rq12ReportError,
        match="protocol-incompatible artifact",
    ):
        rq12_decoder_query_report.collect_report_bundle(
            logs, verify_recipe=lambda _directory, _candidate: False
        )


def test_writes_only_the_rq12_reader_and_evidence_files(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    _complete_query_surface(logs)
    bundle = rq12_decoder_query_report.collect_report_bundle(
        logs, verify_recipe=lambda _directory, _candidate: True
    )

    paths = rq12_decoder_query_report.write_report_bundle(
        bundle, tmp_path / "scratchpad", tmp_path / "evidence"
    )

    assert set(paths) == {"reader", "evidence"}
    assert paths["reader"].name == "rq12_decoder_query_reader_500m.md"
    assert paths["evidence"].name == "rq12_decoder_query_results.json"
    assert json.loads(paths["evidence"].read_text()) == bundle.evidence
