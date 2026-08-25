from __future__ import annotations

import os
from pathlib import Path
import runpy
import subprocess

import pytest

from dcn.training_metadata import GENERATION_TRAINING_SEMANTICS_REVISION
from experiments.g1_sasrec_item_ids_likes.analysis.rq8_reinvestigation_candidates import (
    candidate_by_run,
    initial_candidates,
    sequence_initial_candidates,
)
from experiments.g1_sasrec_item_ids_likes.launchers import verify_artifact


ROOT = Path(__file__).resolve().parents[3]
CONFIG = (
    ROOT
    / "experiments/g1_sasrec_item_ids_likes/configs/rq8_reinvestigation_variant.py"
)
LAUNCHER = (
    ROOT
    / "experiments/g1_sasrec_item_ids_likes/launchers/architecture/"
    "rq8_reinvestigation_500m.sh"
)


def test_initial_surface_contains_the_approved_query_and_corrected_sequence_runs() -> None:
    candidates = initial_candidates()
    query = [candidate for candidate in candidates if candidate.study == "query"]
    sequence = [
        candidate for candidate in candidates if candidate.study == "sequence"
    ]

    assert len(candidates) == len({candidate.run_name for candidate in candidates}) == 57
    assert {
        (candidate.query_method, candidate.deep_lr)
        for candidate in query
    } == {
        (method, deep_lr)
        for method in ("standard", "end_only", "interleaved")
        for deep_lr in (0.006, 0.012, 0.024)
    }
    assert {
        (candidate.position_method, candidate.max_seq_len, candidate.deep_lr)
        for candidate in sequence
    } == {
        (position, length, deep_lr)
        for position in ("alibi", "rope_reverse_alibi")
        for length in (12, 25, 50, 100, 128, 200, 256, 512)
        for deep_lr in (0.006, 0.012, 0.024)
    }
    assert tuple(sequence) == sequence_initial_candidates()
    assert len(sequence) == len({candidate.run_name for candidate in sequence}) == 48
    for candidate in candidates:
        assert candidate.dataset_size == "500m"
        assert candidate.embedding_lr == 0.064
        assert candidate.batch_size == 1280
        assert candidate.seed == 42
        assert candidate.cap_epochs == 20
        assert candidate_by_run(candidate.run_name) == candidate
        revision = "r1" if candidate.study == "query" else "r2"
        assert candidate.run_name.endswith(
            f"_cap20_ts{GENERATION_TRAINING_SEMANTICS_REVISION}_{revision}_500m"
        )
        if candidate.study == "sequence":
            assert "_sequence_fullcausal_" in candidate.run_name


def test_candidate_lookup_rejects_unknown_or_noncanonical_run_names() -> None:
    with pytest.raises(ValueError, match="unknown RQ8 candidate"):
        candidate_by_run("g1_rq8_query_standard_d0p012_500m")


def test_corrected_lookup_rejects_every_legacy_fixed_window_sequence_name() -> None:
    for candidate in sequence_initial_candidates():
        legacy_name = candidate.run_name.replace(
            "_sequence_fullcausal_", "_sequence_"
        ).replace("_r2_500m", "_r1_500m")

        with pytest.raises(ValueError, match="unknown RQ8 candidate"):
            candidate_by_run(legacy_name)


@pytest.mark.parametrize(
    ("study", "method", "expected_position", "expected_attention_window"),
    [
        ("query", "standard", (False, None, "forward"), 50),
        ("query", "end_only", (False, None, "forward"), 51),
        ("query", "interleaved", (False, None, "forward"), 100),
        ("sequence", "alibi", (True, None, None), None),
        ("sequence", "rope_reverse_alibi", (True, "reverse", None), None),
    ],
)
def test_config_materializes_each_treatment_without_changing_the_batch_contract(
    monkeypatch: pytest.MonkeyPatch,
    study: str,
    method: str,
    expected_position: tuple[bool, str | None, str | None],
    expected_attention_window: int | None,
) -> None:
    candidate = next(
        candidate
        for candidate in initial_candidates()
        if candidate.study == study
        and (
            candidate.query_method == method
            if study == "query"
            else candidate.position_method == method
        )
        and candidate.deep_lr == 0.012
        and candidate.max_seq_len == 128
    )
    monkeypatch.setenv("G1_RQ8_RUN", candidate.run_name)

    experiment = runpy.run_path(str(CONFIG))["experiment"]

    assert experiment.run_name == candidate.run_name
    assert experiment.size == "500m"
    assert experiment.seed == 42
    assert experiment.max_seq_len == 128
    assert experiment.cls_token_mode == (
        method if study == "query" and method != "standard" else "none"
    )
    assert experiment.transformer.alibi == expected_position[0]
    assert experiment.transformer.rope == expected_position[1]
    assert experiment.transformer.learned_positions == expected_position[2]
    assert experiment.transformer.attention_window == expected_attention_window
    assert experiment.embedding_learning_rate == 0.064
    assert experiment.deep_learning_rate == 0.012
    assert experiment.dataloader.batch_size == 1280
    assert experiment.dataloader.effective_batch_size == 1280
    assert experiment.dataloader.gradient_accumulation_steps == 1
    assert not experiment.dense_random_negative_scores
    assert experiment.lr_schedule.shape == "linear"
    assert experiment.num_epochs == 20
    assert experiment.lr_schedule_horizon_epochs == 20
    assert not experiment.adaptive_schedule_early_stopping
    assert experiment.eval_every_n_epochs == 1
    assert experiment.early_stopping_patience == 3
    assert experiment.early_stopping_min_delta == 0.0
    assert experiment.restore_best_weights


def test_every_initial_config_uses_direct_sampled_scoring_and_physical_batch_1280(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for candidate in initial_candidates():
        monkeypatch.setenv("G1_RQ8_RUN", candidate.run_name)
        experiment = runpy.run_path(str(CONFIG))["experiment"]

        assert not experiment.dense_random_negative_scores
        assert experiment.dataloader.batch_size == 1280
        assert experiment.dataloader.gradient_accumulation_steps == 1


def test_every_corrected_sequence_config_uses_full_causal_attention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for candidate in sequence_initial_candidates():
        monkeypatch.setenv("G1_RQ8_RUN", candidate.run_name)
        experiment = runpy.run_path(str(CONFIG))["experiment"]

        assert experiment.transformer.attention_window is None
        assert experiment.cls_token_mode == "none"


def test_config_recipe_verifier_reconstructs_the_candidate_identity() -> None:
    candidate = next(
        candidate
        for candidate in initial_candidates()
        if candidate.query_method == "interleaved" and candidate.deep_lr == 0.012
    )
    assignments = verify_artifact._config_assignments(
        [f"G1_RQ8_RUN={candidate.run_name}"]
    )

    experiment = verify_artifact._config_experiment(CONFIG, assignments)
    _, invariants = verify_artifact._expected_metadata(experiment)

    assert experiment.run_name == candidate.run_name
    assert invariants["cls_token_mode"] == "interleaved"
    assert invariants["dense_random_negative_scores"] is False


def test_launcher_submits_exactly_the_corrected_sequence_runs_once_with_length_scoped_cache_groups(
    tmp_path: Path,
) -> None:
    queue = tmp_path / "queue.sh"
    queue.write_text(
        "enqueue() { printf 'ENQUEUE %s GROUP=%s\\n' \"$1\" "
        "\"${TRAINING_QUEUE_DATA_GROUP-}\" >&2; return 0; }\n"
        "drain() { printf 'DRAIN\\n' >&2; return 0; }\n"
    )
    logs = tmp_path / "logs"
    logs.mkdir()

    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        capture_output=True,
        text=True,
        env=os.environ
        | {
            "G1_TRAINING_QUEUE_LIBRARY": str(queue),
            "G1_RQ8_LOGS": str(logs),
        },
    )

    assert result.returncode == 0, result.stderr
    lines = [line for line in result.stderr.splitlines() if line.startswith("ENQUEUE ")]
    assert len(lines) == 48
    assert len({line.split()[1] for line in lines}) == 48
    assert {line.split()[1] for line in lines} == {
        candidate.run_name for candidate in sequence_initial_candidates()
    }
    query_lines = [line for line in lines if "_query_" in line]
    sequence_lines = [line for line in lines if "_sequence_fullcausal_" in line]
    assert not query_lines
    assert len(sequence_lines) == 48
    for length in (12, 25, 50, 100, 128, 200, 256, 512):
        assert sum(
            line.split("GROUP=")[1] == f"g1-rq8-fullcausal-500m-seq{length}"
            for line in sequence_lines
        ) == 6
    assert result.stderr.count("DRAIN") == 1
