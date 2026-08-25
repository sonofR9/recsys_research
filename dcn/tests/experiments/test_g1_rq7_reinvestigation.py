from __future__ import annotations

import os
from pathlib import Path
import runpy
import subprocess

import pytest

from dcn.training_metadata import GENERATION_TRAINING_SEMANTICS_REVISION
from experiments.g1_sasrec_item_ids_likes.analysis.rq7_reinvestigation_candidates import (
    Rq7Candidate,
    bounded_reverse_diagnostic_candidates,
    bounded_reverse_r5_diagnostic_candidates,
    bounded_reverse_r6_diagnostic_candidates,
    candidate_by_run,
    current_implementation_revision,
    diagnostic_candidates,
    historical_combined_diagnostic_candidates,
    initial_candidates,
    legacy_concat_diagnostic_candidates,
    make_boundary_candidate,
    make_confirmation_candidate,
    rope_base_candidates,
    zero_reverse_diagnostic_candidates,
)
from experiments.g1_sasrec_item_ids_likes.launchers import verify_artifact


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "experiments/g1_sasrec_item_ids_likes"
CONFIG = EXPERIMENT / "configs/rq7_reinvestigation_variant.py"
DEBUG_LAUNCHER = EXPERIMENT / "launchers/architecture/rq7_reinvestigation_50m.sh"
INITIAL_LAUNCHER = EXPERIMENT / "launchers/architecture/rq7_reinvestigation_500m.sh"
ROPE_BASE_LAUNCHER = (
    EXPERIMENT / "launchers/architecture/rq7_reinvestigation_rope_bases_500m.sh"
)

PRIMARY_TREATMENTS = {
    "learned_forward_add",
    "learned_forward_concat",
    "learned_forward_reverse_add",
    "learned_forward_reverse_concat",
    "learned_forward_add_alibi",
    "learned_forward_concat_alibi",
    "learned_forward_reverse_add_alibi",
    "learned_forward_reverse_concat_alibi",
    "none",
    "alibi",
    "rope_forward_base10000",
    "rope_forward_base10000_alibi",
}


def test_candidate_surfaces_are_exact_and_disjoint() -> None:
    diagnostic = diagnostic_candidates()
    initial = initial_candidates()
    rope_bases = rope_base_candidates()

    assert (
        len(diagnostic) == len({candidate.run_name for candidate in diagnostic}) == 14
    )
    assert len(initial) == len({candidate.run_name for candidate in initial}) == 36
    assert len(rope_bases) == len({candidate.run_name for candidate in rope_bases}) == 6
    assert {candidate.treatment for candidate in initial} == PRIMARY_TREATMENTS
    assert {(candidate.treatment, candidate.deep_lr) for candidate in initial} == {
        (treatment, deep_lr)
        for treatment in PRIMARY_TREATMENTS
        for deep_lr in (0.006, 0.012, 0.024)
    }
    assert {(candidate.treatment, candidate.deep_lr) for candidate in rope_bases} == {
        (f"rope_forward_base{base}", deep_lr)
        for base in (100, 1000)
        for deep_lr in (0.006, 0.012, 0.024)
    }
    assert {candidate.treatment for candidate in diagnostic} == PRIMARY_TREATMENTS | {
        "rope_forward_base100",
        "rope_forward_base1000",
    }
    assert {candidate.deep_lr for candidate in diagnostic} == {0.012}
    assert {candidate.dataset_size for candidate in diagnostic} == {"50m"}
    assert {candidate.dataset_size for candidate in initial + rope_bases} == {"500m"}
    assert not {candidate.run_name for candidate in diagnostic} & {
        candidate.run_name for candidate in initial + rope_bases
    }
    for candidate in diagnostic + initial + rope_bases:
        assert candidate.embedding_lr == 0.064
        assert candidate.batch_size == 1280
        assert candidate.seed == 42
        assert candidate.horizon_epochs == 20
        assert candidate_by_run(candidate.run_name) == candidate
        revision = current_implementation_revision(candidate.treatment)
        assert candidate.implementation_revision == revision
        assert candidate.run_name.endswith(
            f"_ts{GENERATION_TRAINING_SEMANTICS_REVISION}_r{revision}_"
            f"{candidate.dataset_size}"
        )

    zero_reverse = zero_reverse_diagnostic_candidates()
    assert len(zero_reverse) == 4
    assert all(candidate.implementation_revision == 4 for candidate in zero_reverse)
    assert {candidate.treatment for candidate in zero_reverse} == {
        "learned_forward_reverse_add",
        "learned_forward_reverse_concat",
        "learned_forward_reverse_add_alibi",
        "learned_forward_reverse_concat_alibi",
    }
    bounded_r5 = bounded_reverse_r5_diagnostic_candidates()
    assert len(bounded_r5) == 4
    assert all(candidate.implementation_revision == 5 for candidate in bounded_r5)
    assert not {candidate.run_name for candidate in bounded_r5} & {
        candidate.run_name for candidate in diagnostic
    }
    bounded_r6 = bounded_reverse_r6_diagnostic_candidates()
    assert len(bounded_r6) == 4
    assert all(candidate.implementation_revision == 6 for candidate in bounded_r6)
    assert not {candidate.run_name for candidate in bounded_r6} & {
        candidate.run_name for candidate in diagnostic
    }


def test_completed_r1_and_r2_concat_diagnostics_remain_reconstructable() -> None:
    legacy_r1 = legacy_concat_diagnostic_candidates(1)
    legacy_r2 = legacy_concat_diagnostic_candidates(2)

    assert len(legacy_r1) == len(legacy_r2) == 4
    assert all(candidate.implementation_revision == 1 for candidate in legacy_r1)
    assert all(candidate.implementation_revision == 2 for candidate in legacy_r2)
    assert all(
        candidate.position.learned_position_fusion == "concat"
        for candidate in legacy_r1 + legacy_r2
    )
    assert all(
        candidate_by_run(candidate.run_name) == candidate
        for candidate in legacy_r1 + legacy_r2
    )
    assert not {candidate.run_name for candidate in legacy_r1 + legacy_r2} & {
        candidate.run_name for candidate in diagnostic_candidates()
    }


def test_historical_combined_r1_through_r6_diagnostics_remain_reconstructable() -> None:
    historical = historical_combined_diagnostic_candidates()

    assert len(historical) == 20
    assert {
        (candidate.position.learned_position_fusion, candidate.implementation_revision)
        for candidate in historical
    } == {
        ("add", 1),
        ("add", 4),
        ("add", 5),
        ("add", 6),
        ("concat", 1),
        ("concat", 2),
        ("concat", 3),
        ("concat", 4),
        ("concat", 5),
        ("concat", 6),
    }
    assert all(
        candidate_by_run(candidate.run_name) == candidate for candidate in historical
    )
    assert not {candidate.run_name for candidate in historical} & {
        candidate.run_name for candidate in diagnostic_candidates()
    }


def test_boundary_and_confirmation_identities_round_trip_without_preselection() -> None:
    surface = next(
        candidate
        for candidate in initial_candidates()
        if candidate.treatment == "rope_forward_base10000"
        and candidate.deep_lr == 0.024
    )

    low = make_boundary_candidate(surface, "low", 2)
    high = make_boundary_candidate(surface, "high", 2)
    confirmation = make_confirmation_candidate(surface, 44)

    assert (low.deep_lr, low.boundary_side, low.boundary_step) == (0.0015, "low", 2)
    assert (high.deep_lr, high.boundary_side, high.boundary_step) == (0.096, "high", 2)
    assert (confirmation.deep_lr, confirmation.seed) == (0.024, 44)
    assert candidate_by_run(low.run_name) == low
    assert candidate_by_run(high.run_name) == high
    assert candidate_by_run(confirmation.run_name) == confirmation

    with pytest.raises(ValueError, match="unknown RQ7 candidate"):
        candidate_by_run(surface.run_name.replace("_seed42_", "_seed43_"))
    with pytest.raises(ValueError, match="seed 43 or 44"):
        make_confirmation_candidate(surface, 42)  # type: ignore[arg-type]


def test_combined_followups_keep_the_bounded_reverse_r7_identity() -> None:
    surface = next(
        candidate
        for candidate in initial_candidates()
        if candidate.treatment == "learned_forward_reverse_concat"
        and candidate.deep_lr == 0.024
    )

    boundary = make_boundary_candidate(surface, "high", 1)
    confirmation = make_confirmation_candidate(surface, 43)

    assert surface.implementation_revision == 7
    assert boundary.implementation_revision == 7
    assert confirmation.implementation_revision == 7
    assert "_r7_500m" in boundary.run_name
    assert "_r7_500m" in confirmation.run_name
    assert candidate_by_run(boundary.run_name) == boundary
    assert candidate_by_run(confirmation.run_name) == confirmation


@pytest.mark.parametrize("implementation_revision", [1, 2])
@pytest.mark.parametrize(
    "overrides",
    [
        {"deep_lr": 0.012, "stage": "initial"},
        {
            "deep_lr": 0.048,
            "stage": "boundary",
            "boundary_side": "high",
            "boundary_step": 1,
        },
        {"deep_lr": 0.012, "stage": "confirmation", "seed": 43},
    ],
)
def test_legacy_concat_is_rejected_outside_the_diagnostic_surface(
    implementation_revision: int,
    overrides: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="historical RQ7 implementation revisions"):
        Rq7Candidate(
            treatment="learned_forward_concat",
            dataset_size="500m",
            implementation_revision=implementation_revision,  # type: ignore[arg-type]
            **overrides,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("legacy_revision", [1, 2])
def test_legacy_concat_native_names_are_not_parseable(
    legacy_revision: int,
) -> None:
    surface = next(
        candidate
        for candidate in initial_candidates()
        if candidate.treatment == "learned_forward_concat"
        and candidate.deep_lr == 0.024
    )
    corrected = (
        surface,
        make_boundary_candidate(surface, "high", 1),
        make_confirmation_candidate(surface, 43),
    )

    for candidate in corrected:
        with pytest.raises(
            ValueError,
            match="unknown RQ7 candidate|historical RQ7 implementation revisions",
        ):
            candidate_by_run(
                candidate.run_name.replace("_r3_", f"_r{legacy_revision}_")
            )


@pytest.mark.parametrize(
    ("treatment", "historical_revision"),
    [
        ("learned_forward_reverse_add", 1),
        ("learned_forward_reverse_add", 4),
        ("learned_forward_reverse_add", 5),
        ("learned_forward_reverse_add", 6),
        ("learned_forward_reverse_concat", 3),
        ("learned_forward_reverse_concat", 4),
        ("learned_forward_reverse_concat", 5),
        ("learned_forward_reverse_concat", 6),
        ("learned_forward_reverse_add_alibi", 1),
        ("learned_forward_reverse_add_alibi", 4),
        ("learned_forward_reverse_add_alibi", 5),
        ("learned_forward_reverse_add_alibi", 6),
        ("learned_forward_reverse_concat_alibi", 3),
        ("learned_forward_reverse_concat_alibi", 4),
        ("learned_forward_reverse_concat_alibi", 5),
        ("learned_forward_reverse_concat_alibi", 6),
    ],
)
def test_historical_combined_native_artifact_names_remain_reconstructable(
    treatment: str, historical_revision: int
) -> None:
    current = next(
        candidate
        for candidate in initial_candidates()
        if candidate.treatment == treatment and candidate.deep_lr == 0.012
    )
    historical_name = current.run_name.replace("_r7_", f"_r{historical_revision}_")

    historical = candidate_by_run(historical_name)

    assert historical.run_name == historical_name
    assert historical.implementation_revision == historical_revision
    assert historical.stage == "initial"


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_followup_identities_materialize_through_the_same_config(
    monkeypatch: pytest.MonkeyPatch, seed: int
) -> None:
    surface = next(
        candidate
        for candidate in initial_candidates()
        if candidate.treatment == "alibi" and candidate.deep_lr == 0.024
    )
    candidate = (
        make_boundary_candidate(surface, "high", 1)
        if seed == 42
        else make_confirmation_candidate(surface, seed)  # type: ignore[arg-type]
    )
    monkeypatch.setenv("G1_RQ7_RUN", candidate.run_name)

    experiment = runpy.run_path(str(CONFIG))["experiment"]

    assert experiment.run_name == candidate.run_name
    assert experiment.seed == seed
    assert experiment.deep_learning_rate == candidate.deep_lr
    assert experiment.transformer.alibi
    assert experiment.size == "500m"


def test_config_recipe_verifier_reconstructs_the_candidate_identity() -> None:
    candidate = next(
        candidate
        for candidate in diagnostic_candidates()
        if candidate.treatment == "learned_forward_reverse_concat_alibi"
    )
    assignments = verify_artifact._config_assignments(
        [f"G1_RQ7_RUN={candidate.run_name}"]
    )

    experiment = verify_artifact._config_experiment(CONFIG, assignments)
    _, invariants = verify_artifact._expected_metadata(experiment)

    assert experiment.run_name == candidate.run_name
    assert invariants["dataset_size"] == "50m"
    assert invariants["user_sample"] is None
    assert invariants["transformer"]["learned_positions"] == ["forward", "reverse"]
    assert invariants["transformer"]["learned_position_fusion"] == "concat"
    assert invariants["transformer"]["learned_position_fusion_residual"] == "rezero"
    assert invariants["transformer"]["learned_position_fusion_semantics_revision"] == 3
    assert "learned_position_initialization" not in invariants["transformer"]
    assert (
        invariants["transformer"]["learned_position_reverse_correction"]
        == "bounded_tanh"
    )
    assert invariants["transformer"]["learned_position_reverse_max_scale"] == 0.025
    assert (
        invariants["transformer"][
            "learned_position_reverse_initializer_rng_nonadvancing"
        ]
        is True
    )
    assert (
        invariants["transformer"][
            "learned_position_reverse_initializer_semantics_revision"
        ]
        == 1
    )
    assert (
        invariants["transformer"][
            "learned_position_reverse_correction_semantics_revision"
        ]
        == 1
    )
    assert "learned_position_fusion_normalization" not in invariants["transformer"]
    assert "rope_base" not in invariants["transformer"]
    assert invariants["transformer"]["alibi"] is True

    nondefault_rope = next(
        candidate
        for candidate in diagnostic_candidates()
        if candidate.treatment == "rope_forward_base100"
    )
    assignments = verify_artifact._config_assignments(
        [f"G1_RQ7_RUN={nondefault_rope.run_name}"]
    )
    experiment = verify_artifact._config_experiment(CONFIG, assignments)
    _, invariants = verify_artifact._expected_metadata(experiment)

    assert invariants["transformer"]["rope_base"] == 100.0


def test_legacy_metadata_normalization_preserves_r3_concat_identity() -> None:
    metadata = {
        "transfer_invariants": {
            "transformer": {
                "learned_position_fusion": "concat",
                "learned_position_fusion_residual": "rezero",
                "learned_position_fusion_semantics_revision": 3,
            }
        }
    }

    normalized = verify_artifact._with_legacy_accumulation_defaults(metadata)
    transformer = normalized["transfer_invariants"]["transformer"]

    assert transformer["learned_position_fusion_residual"] == "rezero"
    assert "learned_position_fusion_normalization" not in transformer


@pytest.mark.parametrize(
    ("revision", "normalization"),
    [(1, None), (2, "input_rms")],
)
def test_legacy_concat_recipes_reconstruct_their_exact_semantics(
    revision: int, normalization: str | None
) -> None:
    candidate = legacy_concat_diagnostic_candidates(revision)[0]  # type: ignore[arg-type]
    assignments = verify_artifact._config_assignments(
        [f"G1_RQ7_RUN={candidate.run_name}"]
    )

    experiment = verify_artifact._config_experiment(CONFIG, assignments)
    _, invariants = verify_artifact._expected_metadata(experiment)

    assert experiment.run_name == candidate.run_name
    assert experiment.transformer.learned_position_fusion_normalization == normalization
    assert experiment.transformer.learned_position_fusion_residual is None
    assert (
        invariants["transformer"]["learned_position_fusion_semantics_revision"]
        == revision
    )
    assert (
        invariants["transformer"]["learned_position_fusion_normalization"]
        == normalization
    )
    assert "learned_position_fusion_residual" not in invariants["transformer"]


@pytest.mark.parametrize(
    ("fusion", "revision"),
    [
        ("add", 1),
        ("add", 4),
        ("add", 5),
        ("add", 6),
        ("concat", 1),
        ("concat", 2),
        ("concat", 3),
        ("concat", 4),
        ("concat", 5),
        ("concat", 6),
    ],
)
def test_historical_combined_recipes_reconstruct_r1_through_r6(
    fusion: str, revision: int
) -> None:
    candidate = next(
        candidate
        for candidate in historical_combined_diagnostic_candidates()
        if candidate.position.learned_position_fusion == fusion
        and candidate.implementation_revision == revision
        and not candidate.position.alibi
    )
    assignments = verify_artifact._config_assignments(
        [f"G1_RQ7_RUN={candidate.run_name}"]
    )

    experiment = verify_artifact._config_experiment(CONFIG, assignments)
    _, invariants = verify_artifact._expected_metadata(experiment)
    transformer = invariants["transformer"]

    assert experiment.run_name == candidate.run_name
    assert experiment.transformer.learned_position_reverse_correction == (
        "bounded_tanh" if revision in (5, 6) else None
    )
    if revision in (5, 6):
        assert transformer["learned_position_reverse_max_scale"] == (
            0.1 if revision == 5 else 0.025
        )
        assert (
            experiment.transformer.learned_position_reverse_initializer_rng_nonadvancing
            is False
        )
        assert (
            "learned_position_reverse_initializer_rng_nonadvancing" not in transformer
        )
    else:
        assert "learned_position_reverse_correction" not in transformer
        assert "learned_position_reverse_max_scale" not in transformer
    assert experiment.transformer.learned_position_fusion_normalization == (
        "input_rms" if revision == 2 else None
    )
    assert experiment.transformer.learned_position_fusion_residual == (
        "rezero" if fusion == "concat" and revision in (3, 4, 5, 6) else None
    )
    assert experiment.transformer.learned_position_initialization == (
        "zero_reverse" if revision == 4 else "default"
    )


@pytest.mark.parametrize(
    ("treatment", "learned_positions", "fusion", "rope", "rope_base", "alibi"),
    [
        ("none", None, "add", None, 10000.0, False),
        ("alibi", None, "add", None, 10000.0, True),
        ("learned_forward_add", "forward", "add", None, 10000.0, False),
        (
            "learned_forward_reverse_concat_alibi",
            ("forward", "reverse"),
            "concat",
            None,
            10000.0,
            True,
        ),
        ("rope_forward_base100", None, "add", "forward", 100.0, False),
        (
            "rope_forward_base10000_alibi",
            None,
            "add",
            "forward",
            10000.0,
            True,
        ),
    ],
)
def test_config_materializes_treatments_and_fixed_native_protocol(
    monkeypatch: pytest.MonkeyPatch,
    treatment: str,
    learned_positions: str | tuple[str, str] | None,
    fusion: str,
    rope: str | None,
    rope_base: float,
    alibi: bool,
) -> None:
    candidates = (
        diagnostic_candidates()
        if treatment == "rope_forward_base100"
        else initial_candidates()
    )
    candidate = next(
        candidate
        for candidate in candidates
        if candidate.treatment == treatment and candidate.deep_lr == 0.012
    )
    monkeypatch.setenv("G1_RQ7_RUN", candidate.run_name)

    experiment = runpy.run_path(str(CONFIG))["experiment"]
    transformer = experiment.transformer

    assert experiment.run_name == candidate.run_name
    assert experiment.size == candidate.dataset_size
    assert experiment.seed == 42
    assert experiment.max_seq_len == 128
    assert experiment.embedding_learning_rate == 0.064
    assert experiment.deep_learning_rate == 0.012
    assert experiment.dataloader.batch_size == 1280
    assert experiment.dataloader.effective_batch_size == 1280
    assert experiment.dataloader.gradient_accumulation_steps == 1
    assert experiment.lr_schedule.shape == "linear"
    assert experiment.num_epochs == 20
    assert experiment.lr_schedule_horizon_epochs == 20
    assert not experiment.adaptive_schedule_early_stopping
    assert experiment.eval_every_n_epochs == 1
    assert experiment.restore_best_weights
    assert experiment.user_sample is None
    assert experiment.timestamp_delta == "bins"
    assert experiment.timestamp_combination == "add"
    assert experiment.timestamp_num_bins == 16
    assert experiment.negative_sampling == "random"
    assert experiment.dense_random_negative_scores
    assert not experiment.bos
    assert not experiment.cls_token
    assert experiment.cls_token_mode == "none"
    assert experiment.evaluation_catalog == "all"
    assert not experiment.exclude_seen_from_evaluation
    assert transformer.dim == 64
    assert transformer.num_layers == 2
    assert transformer.ffn == "swiglu"
    assert transformer.ffn_intermediate_dim == 171
    assert transformer.nhead == 2
    assert transformer.num_kv_heads == 1
    assert transformer.norm == "layer"
    assert transformer.norm_place == "pre"
    assert transformer.input_norm is None
    assert transformer.final_norm == "layer"
    assert transformer.dropout == 0.1
    assert transformer.input_dropout == 0.1
    assert transformer.ffn_dropout == 0.1
    assert transformer.attention_window == 50
    assert transformer.learned_positions == learned_positions
    assert transformer.learned_position_fusion == fusion
    assert transformer.learned_position_fusion_normalization is None
    assert transformer.learned_position_fusion_residual == (
        "rezero" if fusion == "concat" else None
    )
    assert transformer.learned_position_initialization == ("default")
    assert transformer.learned_position_reverse_correction == (
        "bounded_tanh" if learned_positions == ("forward", "reverse") else None
    )
    assert transformer.learned_position_reverse_max_scale == (
        0.025 if learned_positions == ("forward", "reverse") else 0.1
    )
    assert transformer.learned_position_reverse_initializer_rng_nonadvancing is (
        learned_positions == ("forward", "reverse")
    )
    assert transformer.rope == rope
    assert transformer.rope_base == rope_base
    assert transformer.alibi is alibi


@pytest.mark.parametrize(
    ("launcher", "expected"),
    [
        (DEBUG_LAUNCHER, bounded_reverse_diagnostic_candidates()),
        (ROPE_BASE_LAUNCHER, rope_base_candidates()),
    ],
)
def test_launchers_submit_each_exact_stage_once(
    tmp_path: Path,
    launcher: Path,
    expected: tuple[object, ...],
) -> None:
    queue = tmp_path / "queue.sh"
    queue.write_text(
        "enqueue() { printf 'ENQUEUE %s GROUP=%s\\n' \"$1\" "
        '"${TRAINING_QUEUE_DATA_GROUP-}" >&2; return 0; }\n'
        "drain() { printf 'DRAIN\\n' >&2; return 0; }\n"
    )
    logs = tmp_path / "logs"
    logs.mkdir()

    result = subprocess.run(
        ["bash", str(launcher)],
        capture_output=True,
        text=True,
        env=os.environ
        | {
            "G1_TRAINING_QUEUE_LIBRARY": str(queue),
            "G1_RQ7_LOGS": str(logs),
        },
    )

    assert result.returncode == 0, result.stderr
    lines = [line for line in result.stderr.splitlines() if line.startswith("ENQUEUE ")]
    expected_names = {candidate.run_name for candidate in expected}  # type: ignore[attr-defined]
    assert len(lines) == len(expected_names)
    assert {line.split()[1] for line in lines} == expected_names
    assert all(
        line.endswith(
            "GROUP=g1-rq7-50m-seq128"
            if launcher == DEBUG_LAUNCHER
            else "GROUP=g1-rq7-500m-seq128"
        )
        for line in lines
    )
    assert result.stderr.count("DRAIN") == 1


def test_native500_launcher_fails_closed_without_current_diagnostic_gate(
    tmp_path: Path,
) -> None:
    result = subprocess.run(
        ["bash", str(INITIAL_LAUNCHER)],
        capture_output=True,
        text=True,
        env=os.environ | {"G1_RQ7_LOGS": str(tmp_path)},
    )

    assert result.returncode != 0
    assert "diagnostic surface is incomplete" in result.stderr
