from __future__ import annotations

import os
from pathlib import Path
import runpy
import subprocess

import pytest

from experiments.g1_sasrec_item_ids_likes.analysis.rq13_rq14_query_candidates import (
    DEEP_LRS,
    candidate_by_run,
    make_boundary_candidate,
    rq14_initial_candidates,
    validated_rq14_boundary_candidates,
)


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "experiments/g1_sasrec_item_ids_likes"
LAUNCHER = EXPERIMENT / "launchers/architecture/rq14_query_memory_500m.sh"
BOUNDARY_LAUNCHER = (
    EXPERIMENT / "launchers/architecture/rq14_query_memory_boundary_500m.sh"
)
CONFIG = EXPERIMENT / "configs/rq13_rq14_query_variant.py"


def test_rq14_manifest_is_the_exact_approved_twelve_run_grid() -> None:
    candidates = rq14_initial_candidates()

    assert len(candidates) == 12
    assert len({candidate.run_name for candidate in candidates}) == 12
    assert {candidate.study for candidate in candidates} == {"rq14"}
    assert {candidate.treatment for candidate in candidates} == {
        "shared_cls_only",
        "distinct_cls_only",
        "shared_history",
        "distinct_history",
    }
    assert {candidate.deep_lr for candidate in candidates} == set(DEEP_LRS)
    assert all(
        candidate_by_run(candidate.run_name) == candidate for candidate in candidates
    )


def test_rq14_launcher_lists_only_the_exact_manifest() -> None:
    result = subprocess.run(
        [str(LAUNCHER), "--list"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert result.stdout.splitlines() == [
        candidate.run_name for candidate in rq14_initial_candidates()
    ]


@pytest.mark.parametrize("arguments", [["--list", "extra"], ["--lst"], ["extra"]])
def test_rq14_launcher_rejects_unexpected_arguments(arguments: list[str]) -> None:
    result = subprocess.run(
        [str(LAUNCHER), *arguments],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 2
    assert "usage:" in result.stderr


@pytest.mark.parametrize("candidate", rq14_initial_candidates())
def test_rq14_config_has_the_approved_fixed_recipe(
    candidate, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("G1_QUERY_RUN", candidate.run_name)

    experiment = runpy.run_path(str(CONFIG))["experiment"]

    assert experiment.size == "500m"
    assert experiment.seed == 42
    assert experiment.dataloader.batch_size == 1280
    assert experiment.dataloader.effective_batch_size == 1280
    assert experiment.embedding_learning_rate == 0.064
    assert experiment.deep_learning_rate == candidate.deep_lr
    assert experiment.num_epochs == 20
    assert experiment.lr_schedule_horizon_epochs == 20
    assert experiment.lr_schedule.shape == "linear"
    assert experiment.adaptive_schedule_early_stopping is False
    assert experiment.early_stopping_patience is None
    assert experiment.restore_best_weights is True
    assert experiment.query_architecture == "decoder_decoder"
    assert experiment.num_query_slots == 4
    assert experiment.query_slots_shared is candidate.treatment.startswith("shared_")
    assert experiment.include_history_memory is candidate.treatment.endswith("_history")
    assert experiment.transformer.num_layers == 2
    assert experiment.transformer.attention_window == 54
    assert experiment.retrieval_decoder.num_layers == 1
    assert experiment.retrieval_decoder.ffn == "swiglu"
    assert experiment.retrieval_decoder.ffn_intermediate_dim == 128
    assert experiment.window == "bounded_prefix"
    assert experiment.prefix_cap == 1


def test_rq14_boundary_validation_fails_closed_and_roundtrips() -> None:
    initial = rq14_initial_candidates()[0]
    boundary = make_boundary_candidate(initial, "low", 1)
    evidence = {
        "research_question": "RQ14 decoder-decoder query memory",
        "dataset_size": "500m",
        "missing_initial_artifacts": [],
        "required_boundary_followups": [boundary.run_name],
        "required_followups": [boundary.run_name],
    }

    assert validated_rq14_boundary_candidates(
        evidence, [boundary.run_name]
    ) == (boundary,)
    assert candidate_by_run(boundary.run_name) == boundary

    with pytest.raises(ValueError, match="exact"):
        validated_rq14_boundary_candidates(evidence, [initial.run_name])
    with pytest.raises(ValueError, match="incomplete"):
        validated_rq14_boundary_candidates(
            {**evidence, "missing_initial_artifacts": [initial.run_name]},
            [boundary.run_name],
        )


def test_rq14_launchers_use_persistent_queue_and_recipe_verification() -> None:
    initial = LAUNCHER.read_text()
    boundary = BOUNDARY_LAUNCHER.read_text()

    assert "rq14_initial_candidates" in initial
    assert "exactly 12 runs" in initial
    assert "utils/training_queue/queue.sh" in initial
    assert "g1_require_config_compatible_or_absent" in initial
    assert 'enqueue "$run"' in initial
    assert "rq14_query_memory_report" in boundary
    assert "validated_rq14_boundary_candidates" in boundary
    assert "utils/training_queue/queue.sh" in boundary
    assert "g1_require_config_compatible_or_absent" in boundary


def test_rq14_boundary_launcher_rejects_non_boundary_run(tmp_path: Path) -> None:
    candidate = rq14_initial_candidates()[0]
    environment = os.environ.copy()
    environment["G1_QUERY_LOGS"] = str(tmp_path / "logs")
    environment["G1_QUERY_EVIDENCE_DIR"] = str(tmp_path / "evidence")
    environment["G1_QUERY_SCRATCHPAD_DIR"] = str(tmp_path / "scratchpad")

    result = subprocess.run(
        [str(BOUNDARY_LAUNCHER), "--list", candidate.run_name],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
