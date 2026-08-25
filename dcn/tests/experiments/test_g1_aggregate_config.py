from __future__ import annotations

import json
import os
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest
import torch

from dcn.config import MuTransferGenerationExperiment
from dcn.nn.precomputed_embeddings import PrecomputedEmbeddingLookup
from dcn.tests.helpers import packed_batch
from experiments.g1_sasrec_item_ids_likes.analysis.aggregate_candidates import (
    bridge_candidates,
    full_horizon_rerun_candidates,
    initial_candidates,
)
from experiments.g1_sasrec_item_ids_likes.launchers import verify_artifact


ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "experiments/g1_sasrec_item_ids_likes/configs/aggregate_variant.py"


def _load(run_name: str) -> MuTransferGenerationExperiment:
    previous = os.environ.get("G1_AGGREGATE_RUN")
    os.environ["G1_AGGREGATE_RUN"] = run_name
    try:
        experiment = runpy.run_path(str(CONFIG))["experiment"]
    finally:
        if previous is None:
            os.environ.pop("G1_AGGREGATE_RUN", None)
        else:
            os.environ["G1_AGGREGATE_RUN"] = previous
    assert isinstance(experiment, MuTransferGenerationExperiment)
    return experiment


def test_baseline_reconstructs_the_approved_original_model() -> None:
    candidate = initial_candidates()[1]
    experiment = _load(candidate.run_name)

    assert experiment.size == "500m"
    assert experiment.seed == 42
    assert experiment.dataloader.batch_size == 1280
    assert experiment.embedding_learning_rate == 0.064
    assert experiment.deep_learning_rate == 0.012
    assert experiment.num_epochs == 80
    assert not experiment.adaptive_schedule_early_stopping
    assert experiment.lr_schedule.shape == "constant"
    assert experiment.max_seq_len == 100
    assert not experiment.bos
    assert experiment.cls_token_mode == "none"
    assert experiment.timestamp_delta is None
    assert experiment.negative_sampling == "offline_logq"
    assert experiment.logq_correction == "baseline"
    assert not experiment.correct_positive_logq
    assert experiment.num_in_batch_negatives == 512
    assert experiment.transformer.dim == 64
    assert experiment.transformer.num_layers == 2
    assert experiment.transformer.nhead == 2
    assert experiment.transformer.num_kv_heads == 2
    assert experiment.transformer.ffn == "gelu"
    assert experiment.transformer.ffn_intermediate_dim == 256
    assert experiment.transformer.norm == "layer"
    assert experiment.transformer.norm_place == "pre"
    assert experiment.transformer.input_norm is None
    assert experiment.transformer.final_norm == "layer"
    assert experiment.transformer.learned_positions == "forward"
    assert experiment.transformer.learned_position_fusion == "add"
    assert not experiment.transformer.alibi
    assert experiment.transformer.rope is None
    assert experiment.transformer.attention_window is None


def test_aggregate_materializes_all_ten_fixed_members_at_each_selected_depth() -> None:
    candidates = [
        candidate
        for candidate in initial_candidates()
        if candidate.family == "aggregate"
        and candidate.embedding_lr == 0.064
        and candidate.deep_lr == 0.048
    ]

    for candidate in candidates:
        experiment = _load(candidate.run_name)
        transformer = experiment.transformer

        assert transformer.num_layers == candidate.num_layers
        assert transformer.ffn == "swiglu"
        assert transformer.ffn_intermediate_dim == 192
        assert transformer.gated_ffn_dropout
        assert transformer.ffn_dropout == 0.1
        assert transformer.nhead == 2
        assert transformer.num_kv_heads == 1
        assert transformer.norm == "layer"
        assert transformer.norm_place == "post"
        assert transformer.input_norm == "rms"
        assert transformer.final_norm == "rms"
        assert transformer.alibi
        assert transformer.rope == "timestamp_reverse"
        assert transformer.learned_positions == ("forward", "reverse")
        assert transformer.learned_position_fusion == "concat"
        assert transformer.learned_position_fusion_residual == "rezero"
        assert transformer.learned_position_reverse_correction == "bounded_tanh"
        assert transformer.learned_position_reverse_max_scale == 0.025
        assert transformer.learned_position_reverse_initializer_rng_nonadvancing
        assert transformer.attention_window is None
        assert experiment.max_seq_len == 100
        assert experiment.bos
        assert experiment.cls_token_mode == "end_only"
        assert experiment.timestamp_delta == "bins"
        assert experiment.timestamp_combination == "add"
        assert experiment.timestamp_num_bins == 32
        assert experiment.negative_sampling == "random_offline_logq"
        assert experiment.logq_correction == "yi2019"
        assert experiment.correct_positive_logq
        assert experiment.num_in_batch_negatives == 2048
        assert experiment.dense_random_negative_scores
        assert experiment.lr_schedule.shape == "cosine"
        assert experiment.lr_schedule.warmup_fraction == 0.05
        assert experiment.lr_schedule.cycles == 1
        assert experiment.lr_schedule.optimizer_group_scope == "deep_only"
        assert experiment.lr_schedule_horizon_epochs == 15
        assert experiment.num_epochs == 15
        assert not experiment.adaptive_schedule_early_stopping
        assert experiment.lr_schedule.anneals_over_horizon


def test_full_h15_reruns_disable_adaptive_and_patience_stopping() -> None:
    for candidate in full_horizon_rerun_candidates():
        experiment = _load(candidate.run_name)

        assert experiment.num_epochs == 15
        assert experiment.lr_schedule_horizon_epochs == 15
        assert not experiment.adaptive_schedule_early_stopping
        assert experiment.lr_schedule.anneals_over_horizon


def test_each_fixed_bridge_changes_only_its_named_member() -> None:
    experiments = {
        candidate.member: _load(candidate.run_name)
        for candidate in bridge_candidates(0.012)
    }

    assert experiments["swiglu"].transformer.ffn == "swiglu"
    assert experiments["swiglu"].transformer.ffn_intermediate_dim == 192
    assert experiments["scheduler"].lr_schedule.optimizer_group_scope == "deep_only"
    assert experiments["position"].transformer.learned_positions == (
        "forward",
        "reverse",
    )
    assert experiments["post_norm"].transformer.norm_place == "post"
    assert experiments["input_final_rms"].transformer.input_norm == "rms"
    assert experiments["input_final_rms"].transformer.final_norm == "rms"
    assert experiments["cls"].cls_token_mode == "end_only"
    assert experiments["time"].timestamp_num_bins == 32
    assert experiments["time"].transformer.rope == "timestamp_reverse"
    assert experiments["popularity"].negative_sampling == "random_offline_logq"
    assert experiments["gqa"].transformer.num_kv_heads == 1
    assert experiments["bos"].bos

    baseline = _load(initial_candidates()[1].run_name)
    for member, experiment in experiments.items():
        if member != "scheduler":
            assert experiment.lr_schedule == baseline.lr_schedule
            assert experiment.num_epochs == baseline.num_epochs
        if member != "popularity":
            assert experiment.negative_sampling == baseline.negative_sampling
        if member != "time":
            assert experiment.timestamp_delta == baseline.timestamp_delta
        if member != "cls":
            assert experiment.cls_token_mode == baseline.cls_token_mode


def test_generic_artifact_verifier_reconstructs_aggregate_config() -> None:
    candidate = next(
        candidate for candidate in initial_candidates() if candidate.family == "aggregate"
    )
    assignments = verify_artifact._config_assignments(
        [f"G1_AGGREGATE_RUN={candidate.run_name}"]
    )
    experiment = verify_artifact._config_experiment(CONFIG, assignments)
    _, invariants = verify_artifact._expected_metadata(experiment)

    assert experiment.run_name == candidate.run_name
    assert invariants["cls_token_mode"] == "end_only"
    assert invariants["transformer"]["rope"] == "timestamp_reverse"
    assert invariants["lr_schedule"]["optimizer_group_scope"] == "deep_only"


def test_deep_only_trace_accepts_an_exact_noncentral_embedding_lr() -> None:
    embedding_lr = 0.07764674795069047
    deep_lr = 0.02484672863178322
    factors = [10 / 19, 0.0]
    metadata = {
        "epochs_trained": 2,
        "optimizer_steps_per_epoch": 10,
        "lr_schedule_horizon_steps": 20,
        "embedding_learning_rate": embedding_lr,
        "deep_learning_rate": deep_lr,
        "lr_group_traces": {
            "embedding": [embedding_lr, embedding_lr],
            "deep": [deep_lr * factor for factor in factors],
        },
    }
    schedule = {
        "shape": "linear",
        "warmup_fraction": 0.0,
        "optimizer_group_scope": "deep_only",
    }

    assert verify_artifact._valid_group_lr_traces(metadata, schedule)


@pytest.mark.parametrize(
    ("stopped_epoch", "horizon_complete", "expected"),
    [(15, True, True), (14, False, False)],
)
def test_historical_h15_verifier_requires_the_completed_declared_horizon(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stopped_epoch: int,
    horizon_complete: bool,
    expected: bool,
) -> None:
    run_name = "historical_h15"
    directory = tmp_path / run_name
    directory.mkdir()
    schedule = {"shape": "cosine"}
    metadata = {
        "num_epochs": 15,
        "max_epochs": 15,
        "epochs_trained": stopped_epoch,
        "stopped_epoch": stopped_epoch,
        "best_epoch": min(10, stopped_epoch),
        "targets_per_epoch": 2,
        "tokens_per_epoch": 3,
        "optimizer_steps_per_epoch": 4,
        "optimizer_steps": 4 * stopped_epoch,
        "training_horizon": 2 * stopped_epoch,
        "token_horizon": 3 * stopped_epoch,
        "tokens_seen": 3 * stopped_epoch,
        "lr_schedule_horizon_epochs": 15,
        "lr_horizon_complete": horizon_complete,
        "transfer_invariants": {
            "adaptive_schedule_early_stopping": True,
            "lr_schedule": schedule,
        },
    }
    (directory / "training_metadata.json").write_text(json.dumps(metadata))
    (directory / "final_metrics.json").write_text(
        json.dumps({"recall@100": 0.1})
    )
    experiment = SimpleNamespace(
        run_name=run_name,
        lr_schedule_horizon_epochs=15,
        num_epochs=15,
        adaptive_schedule_early_stopping=False,
    )
    monkeypatch.setattr(
        verify_artifact, "_config_experiment", lambda *_args: experiment
    )
    monkeypatch.setattr(
        verify_artifact,
        "_expected_metadata",
        lambda _experiment: (
            {"num_epochs": 15},
            {
                "adaptive_schedule_early_stopping": False,
                "lr_schedule": schedule,
            },
        ),
    )
    monkeypatch.setattr(
        verify_artifact, "has_current_generation_semantics", lambda _metadata: True
    )
    monkeypatch.setattr(
        verify_artifact, "_with_legacy_accumulation_defaults", lambda metadata: metadata
    )
    monkeypatch.setattr(
        verify_artifact, "_valid_group_lr_traces", lambda *_args: True
    )

    assert (
        verify_artifact.verify_config_completed_historical_horizon(
            directory, CONFIG, []
        )
        is expected
    )


@pytest.mark.parametrize(
    ("stopped_epoch", "horizon_complete", "expected"),
    [(14, True, False), (15, False, False), (15, True, True)],
)
def test_public_config_verifier_requires_full_nonadaptive_horizon(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stopped_epoch: int,
    horizon_complete: bool,
    expected: bool,
) -> None:
    run_name = "current_h15"
    directory = tmp_path / run_name
    directory.mkdir()
    schedule = {"shape": "cosine"}
    invariants = {
        "adaptive_schedule_early_stopping": False,
        "lr_schedule": schedule,
    }
    metadata = {
        "num_epochs": 15,
        "max_epochs": 15,
        "epochs_trained": stopped_epoch,
        "stopped_epoch": stopped_epoch,
        "best_epoch": min(10, stopped_epoch),
        "early_stopped": stopped_epoch < 15,
        "best_epoch_at_cap": False,
        "selection_resolved": expected,
        "lr_schedule_horizon_epochs": 15,
        "lr_horizon_complete": horizon_complete,
        "targets_per_epoch": 2,
        "tokens_per_epoch": 3,
        "optimizer_steps_per_epoch": 4,
        "optimizer_steps": 4 * stopped_epoch,
        "training_horizon": 2 * stopped_epoch,
        "token_horizon": 3 * stopped_epoch,
        "tokens_seen": 3 * stopped_epoch,
        "transfer_invariants": invariants,
    }
    (directory / "training_metadata.json").write_text(json.dumps(metadata))
    (directory / "final_metrics.json").write_text(
        json.dumps({"recall@100": 0.1})
    )
    experiment = SimpleNamespace(
        run_name=run_name,
        adaptive_schedule_early_stopping=False,
    )
    monkeypatch.setattr(
        verify_artifact, "_config_experiment", lambda *_args: experiment
    )
    monkeypatch.setattr(
        verify_artifact,
        "_expected_metadata",
        lambda _experiment: ({"num_epochs": 15}, invariants),
    )
    monkeypatch.setattr(
        verify_artifact, "has_current_generation_semantics", lambda _metadata: True
    )
    monkeypatch.setattr(
        verify_artifact, "_with_legacy_accumulation_defaults", lambda value: value
    )

    assert verify_artifact.verify_config(directory, CONFIG, []) is expected


def test_aggregate_composition_materializes_and_runs_jointly_on_cpu(
    cpu_attention: None,
) -> None:
    candidate = next(
        candidate
        for candidate in initial_candidates()
        if candidate.family == "aggregate"
        and candidate.num_layers == 4
        and candidate.embedding_lr == 0.064
    )
    experiment = _load(candidate.run_name)
    experiment.__dict__["artifacts"] = SimpleNamespace(
        item_id_column="compact_item_id"
    )
    experiment.__dict__["item_embeddings"] = PrecomputedEmbeddingLookup(
        torch.randn(63, 8), learnable_default=False, strict=False
    )
    experiment.__dict__["device"] = torch.device("cpu")

    model = experiment.base_model.eval()
    with torch.no_grad():
        output = model(packed_batch([1, 2, 3, 4, 5], [3, 2]))

    assert output["query_repr"].shape == (9, 64)
    assert output["item_repr"].shape == (9, 64)
    assert output["lengths"].tolist() == [5, 4]
    assert output["timestamps"].tolist() == [0, 0, 1, 1, 2, 3, 3, 3, 4]
