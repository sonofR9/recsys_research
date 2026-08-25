from __future__ import annotations

import os
from pathlib import Path
import runpy

from dcn.config import MuTransferGenerationExperiment
from experiments.g1_sasrec_item_ids_likes.analysis.rq10_reinvestigation_candidates import (
    candidate_by_run,
    initial_candidates,
    make_lr_boundary_candidate,
    make_width_boundary_candidate,
    selected_width_lr_candidates,
)


ROOT = Path(__file__).resolve().parents[3]
CONFIG = (
    ROOT
    / "experiments/g1_sasrec_item_ids_likes/configs/rq10_reinvestigation_variant.py"
)
INITIAL_LAUNCHER = (
    ROOT
    / "experiments/g1_sasrec_item_ids_likes/launchers/architecture/"
    "rq10_reinvestigation_500m.sh"
)
FOLLOWUP_LAUNCHER = (
    ROOT
    / "experiments/g1_sasrec_item_ids_likes/launchers/architecture/"
    "rq10_reinvestigation_followups_500m.sh"
)


def test_initial_surface_is_the_approved_native_500m_budget() -> None:
    candidates = initial_candidates()

    assert len(candidates) == len({candidate.run_name for candidate in candidates}) == 12
    assert {(candidate.family, candidate.feature_width, candidate.deep_lr) for candidate in candidates} == {
        *(('input_output_only', None, rate) for rate in (0.006, 0.012, 0.024)),
        *(('direct_add', 64, rate) for rate in (0.006, 0.012, 0.024)),
        *(('concat_residual', width, 0.012) for width in (16, 32, 64)),
        *(('gemma_ple', width, 0.012) for width in (8, 16, 32)),
    }
    for candidate in candidates:
        assert candidate.dataset_size == "500m"
        assert candidate.embedding_lr == 0.064
        assert candidate.batch_size == 1280
        assert candidate.seed == 42
        assert candidate.horizon_epochs == 20
        assert candidate.num_layers == 4
        assert candidate_by_run(candidate.run_name) == candidate


def test_selected_width_lr_surface_adds_exactly_four_runs() -> None:
    candidates = selected_width_lr_candidates(
        concat_feature_width=32,
        gemma_feature_width=16,
    )

    assert len(candidates) == len({candidate.run_name for candidate in candidates}) == 4
    assert {(candidate.family, candidate.feature_width, candidate.deep_lr) for candidate in candidates} == {
        ('concat_residual', 32, 0.006),
        ('concat_residual', 32, 0.024),
        ('gemma_ple', 16, 0.006),
        ('gemma_ple', 16, 0.024),
    }
    assert all(candidate.stage == "selected_width_lr" for candidate in candidates)
    assert all(candidate_by_run(candidate.run_name) == candidate for candidate in candidates)


def test_boundary_candidates_extend_only_the_selected_axis() -> None:
    direct = next(
        candidate
        for candidate in initial_candidates()
        if candidate.family == "direct_add" and candidate.deep_lr == 0.024
    )
    concat = next(
        candidate
        for candidate in initial_candidates()
        if candidate.family == "concat_residual" and candidate.feature_width == 16
    )

    deep = make_lr_boundary_candidate(direct, "high")
    width = make_width_boundary_candidate(concat, "low")

    assert (deep.deep_lr, deep.feature_width, deep.stage) == (0.048, 64, "lr_boundary")
    assert (width.deep_lr, width.feature_width, width.stage) == (
        0.012,
        8,
        "width_boundary",
    )
    assert candidate_by_run(deep.run_name) == deep
    assert candidate_by_run(width.run_name) == width


def _load_experiment(run_name: str) -> MuTransferGenerationExperiment:
    previous = os.environ.get("G1_RQ10_RUN")
    os.environ["G1_RQ10_RUN"] = run_name
    try:
        experiment = runpy.run_path(str(CONFIG))["experiment"]
    finally:
        if previous is None:
            os.environ.pop("G1_RQ10_RUN", None)
        else:
            os.environ["G1_RQ10_RUN"] = previous
    assert isinstance(experiment, MuTransferGenerationExperiment)
    return experiment


def test_config_maps_candidates_to_the_fixed_four_layer_recipe() -> None:
    for candidate in initial_candidates():
        experiment = _load_experiment(candidate.run_name)

        assert experiment.run_name == candidate.run_name
        assert experiment.per_layer_item_features == candidate.family.replace(
            "input_output_only", "none"
        )
        assert experiment.per_layer_item_feature_dim == (
            candidate.feature_width
            if candidate.family in {"concat_residual", "gemma_ple"}
            else None
        )
        assert experiment.transformer.num_layers == 4
        assert experiment.transformer.dim == 64
        assert experiment.transformer.ffn == "swiglu"
        assert experiment.transformer.ffn_intermediate_dim == 171
        assert experiment.transformer.nhead == 2
        assert experiment.transformer.num_kv_heads == 1
        assert experiment.transformer.attention_window == 50
        assert experiment.max_seq_len == 128
        assert experiment.timestamp_delta == "bins"
        assert experiment.timestamp_num_bins == 16
        assert experiment.negative_sampling == "random"
        assert experiment.dataloader.effective_batch_size == 1280
        assert experiment.embedding_learning_rate == 0.064
        assert experiment.deep_learning_rate == candidate.deep_lr
        assert experiment.lr_schedule.shape == "linear"
        assert experiment.lr_schedule_horizon_epochs == 20
        assert experiment.num_epochs == 20
        assert not experiment.adaptive_schedule_early_stopping
        assert experiment.mup_base_dim == 16
        assert experiment.mup_delta_dim == 32


def test_initial_launcher_uses_the_persistent_queue() -> None:
    launcher = INITIAL_LAUNCHER.read_text()

    assert "initial_candidates" in launcher
    assert "expected 12" in launcher
    assert "utils/training_queue/queue.sh" in launcher
    assert 'enqueue "$run"' in launcher
    assert "drain" in launcher

    followup = FOLLOWUP_LAUNCHER.read_text()
    assert "G1_RQ10_FOLLOWUP_RUNS" in followup
    assert "candidate_by_run" in followup
    assert 'enqueue "$run"' in followup
    assert "drain" in followup
