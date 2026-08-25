from __future__ import annotations

import json
import os
from pathlib import Path
import runpy
import subprocess

import pytest

from dcn.config import MuTransferGenerationExperiment
from experiments.g1_sasrec_item_ids_likes.analysis.rq11_mixed_streaming_candidates import (
    PRIMARY_FAMILIES,
    Rq11Candidate,
    candidate_by_run,
    diagnostic_candidates,
    initial_candidates,
    local_lr_candidates,
    manifest_payload,
    make_boundary_candidate,
)
from experiments.g1_sasrec_item_ids_likes.analysis.rq11_mixed_streaming_report import (
    Run,
    aggregate_mixture_outcome,
    build_report_bundle,
    collect_report_bundle,
    metric_cell,
    sync_readme,
)
from experiments.g1_sasrec_item_ids_likes.analysis.rq11_mixed_streaming_selection import (
    ArtifactEvidence,
    build_followup_plan,
    select_family_winner,
)
from experiments.g1_sasrec_item_ids_likes.launchers import verify_artifact


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "experiments/g1_sasrec_item_ids_likes"
CONFIG = EXPERIMENT / "configs/rq11_mixed_streaming_variant.py"
LAUNCHER = EXPERIMENT / "launchers/negatives/rq11_mixed_streaming_500m.sh"
MANIFEST = EXPERIMENT / "protocol/rq11_mixed_streaming_manifest.json"


def test_initial_manifest_is_the_exact_balanced_four_family_search() -> None:
    candidates = initial_candidates()

    assert len(candidates) == len({item.run_name for item in candidates}) == 24
    grouped = {
        family: [item for item in candidates if item.family == family]
        for family in {item.family for item in candidates}
    }
    assert set(grouped) == {
        "uniform_catalog",
        "streaming_global_q",
        "popularity_global_q",
        "aggregate_uniform_streaming_global_q",
    }
    assert {family: len(items) for family, items in grouped.items()} == {
        family: 6 for family in grouped
    }
    expected_pairs = {
        (0.006, 512),
        (0.012, 1024),
        (0.024, 2048),
        (0.006, 2048),
        (0.012, 512),
        (0.024, 1024),
    }
    assert {
        (item.deep_lr, item.negative_count) for item in grouped["uniform_catalog"]
    } == expected_pairs
    assert {
        (item.deep_lr, item.negative_count) for item in grouped["popularity_global_q"]
    } == expected_pairs
    assert {
        (item.deep_lr, item.negative_count, item.alpha)
        for item in grouped["streaming_global_q"]
    } == {
        (0.006, 512, 0.005),
        (0.012, 1024, 0.01),
        (0.024, 2048, 0.02),
        (0.006, 2048, 0.01),
        (0.012, 512, 0.02),
        (0.024, 1024, 0.005),
    }
    assert {
        (item.deep_lr, item.negative_count, item.alpha, item.uniform_fraction)
        for item in grouped["aggregate_uniform_streaming_global_q"]
    } == {
        (0.006, 512, 0.005, 0.25),
        (0.012, 1024, 0.02, 0.5),
        (0.024, 2048, 0.01, 0.75),
        (0.006, 2048, 0.02, 0.5),
        (0.012, 512, 0.005, 0.75),
        (0.024, 1024, 0.01, 0.25),
    }
    for candidate in candidates:
        assert candidate.dataset_size == "500m"
        assert candidate.embedding_lr == 0.064
        assert candidate.batch_size == 1280
        assert candidate.seed == 42
        assert candidate.horizon_epochs == 20
        assert candidate_by_run(candidate.run_name) == candidate


def test_checked_in_manifest_is_generated_from_the_public_candidates() -> None:
    assert json.loads(MANIFEST.read_text()) == manifest_payload()


def test_local_diagnostic_and_boundary_candidates_preserve_coordinates() -> None:
    winner = next(
        item
        for item in initial_candidates()
        if item.family == "aggregate_uniform_streaming_global_q"
        and item.deep_lr == 0.012
        and item.negative_count == 512
    )

    local = local_lr_candidates(winner)
    diagnostic = diagnostic_candidates(winner)
    negative_boundary = make_boundary_candidate(winner, "negative_count", "low", 1)
    alpha_boundary = make_boundary_candidate(winner, "alpha", "low", 1)
    fraction_boundary = make_boundary_candidate(winner, "uniform_fraction", "high", 1)
    lr_boundary = make_boundary_candidate(winner, "deep_lr", "high", 1)

    assert {item.deep_lr for item in local} == {0.006, 0.012, 0.024}
    assert winner in local
    assert len(diagnostic) == 3
    assert {item.deep_lr for item in diagnostic} == {0.006, 0.012, 0.024}
    assert {item.family for item in diagnostic} == {
        "aggregate_uniform_streaming_global_q_negative_only"
    }
    assert all(
        item.secondary_coordinates == winner.secondary_coordinates
        for item in diagnostic
    )
    assert negative_boundary.negative_count == 256
    assert alpha_boundary.alpha == 0.0025
    assert fraction_boundary.uniform_fraction == 0.875
    assert lr_boundary.deep_lr == 0.048
    assert all(
        candidate_by_run(item.run_name) == item
        for item in (
            *local,
            *diagnostic,
            negative_boundary,
            alpha_boundary,
            fraction_boundary,
            lr_boundary,
        )
    )


def test_config_locks_native_horizon_and_unconditional_global_q(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary = next(
        item
        for item in initial_candidates()
        if item.family == "aggregate_uniform_streaming_global_q"
    )
    monkeypatch.setenv("G1_RQ11_RUN", primary.run_name)

    experiment = runpy.run_path(str(CONFIG))["experiment"]

    assert isinstance(experiment, MuTransferGenerationExperiment)
    assert experiment.item_embedding_dim == 64
    assert experiment.mup_base_dim == 16
    assert experiment.mup_delta_dim == 32
    assert experiment.size == "500m"
    assert experiment.dataloader.batch_size == 1280
    assert experiment.embedding_learning_rate == 0.064
    assert experiment.deep_learning_rate == primary.deep_lr
    assert experiment.num_in_batch_negatives == primary.negative_count
    assert experiment.logq_alpha == primary.alpha
    assert experiment.random_negative_fraction == primary.uniform_fraction
    assert experiment.negative_sampling == "mixed_online_global_q"
    assert experiment.correct_positive_logq
    assert not experiment.mask_false_negatives
    assert not experiment.exclude_own_group_negatives
    assert experiment.num_epochs == experiment.lr_schedule_horizon_epochs == 20
    assert experiment.early_stopping_patience is None
    _, invariants = verify_artifact._expected_metadata(experiment)
    assert invariants["negative_sampling_semantics_revision"] == 2


def test_artifact_verifier_accepts_rq11_candidate_assignment() -> None:
    candidate = initial_candidates()[0]

    assert verify_artifact._config_assignments(
        [f"G1_RQ11_RUN={candidate.run_name}"]
    ) == {"G1_RQ11_RUN": candidate.run_name}


def _evidence(candidate: Rq11Candidate, recall: float, ndcg: float) -> ArtifactEvidence:
    return ArtifactEvidence(candidate, recall, ndcg, recall, ndcg)


def test_selection_uses_validation_recall_ndcg_then_negative_cost() -> None:
    candidates = [
        item for item in initial_candidates() if item.family == "uniform_catalog"
    ]
    evidence = [_evidence(item, 0.1, 0.04) for item in candidates]
    evidence[1] = _evidence(candidates[1], 0.2, 0.05)
    evidence[4] = _evidence(candidates[4], 0.2, 0.05)

    assert select_family_winner(evidence).candidate == candidates[4]


def test_followup_plan_requires_local_lr_before_diagnostics() -> None:
    initial = initial_candidates()
    available = {
        item.run_name: _evidence(item, 0.1 + index / 1000, 0.04)
        for index, item in enumerate(initial)
    }

    plan = build_followup_plan(lambda item: available.get(item.run_name))

    expected = []
    for family in PRIMARY_FAMILIES:
        winner = select_family_winner(
            available[item.run_name]
            for item in initial
            if item.family == family
        ).candidate
        expected.extend(
            item
            for item in local_lr_candidates(winner)
            if item.run_name not in available
        )
    assert plan.stage == "local_lr"
    assert plan.candidates == tuple(expected)
    assert all(item.stage == "local_lr" for item in plan.candidates)


def test_followup_plan_advances_each_unresolved_family_independently() -> None:
    available = {
        item.run_name: _evidence(item, 0.1, 0.04) for item in initial_candidates()
    }
    winners: dict[str, Rq11Candidate] = {}
    for family in PRIMARY_FAMILIES:
        winner = next(
            item
            for item in initial_candidates()
            if item.family == family
            and item.negative_count == 512
            and item.deep_lr == 0.012
        )
        winners[family] = winner
        available[winner.run_name] = _evidence(winner, 0.2, 0.05)
        for local in local_lr_candidates(winner):
            available.setdefault(local.run_name, _evidence(local, 0.19, 0.049))
    mixture = winners["aggregate_uniform_streaming_global_q"]
    for diagnostic in diagnostic_candidates(mixture):
        available[diagnostic.run_name] = _evidence(diagnostic, 0.18, 0.048)

    plan = build_followup_plan(lambda item: available.get(item.run_name))

    assert plan.stage == "boundary"
    assert tuple(item.family for item in plan.candidates) == PRIMARY_FAMILIES
    assert all(item.boundary_axis == "negative_count" for item in plan.candidates)
    assert all(item.negative_count == 256 for item in plan.candidates)

    boundary = plan.candidates[0]
    assert boundary.family == "uniform_catalog"
    available[boundary.run_name] = _evidence(boundary, 0.21, 0.051)
    continuation = build_followup_plan(lambda item: available.get(item.run_name))

    assert continuation.stage == "mixed"
    assert tuple(item.family for item in continuation.candidates) == (
        "uniform_catalog",
        "uniform_catalog",
        *PRIMARY_FAMILIES[1:],
    )
    assert all(item.stage == "local_lr" for item in continuation.candidates[:2])
    assert all(item.negative_count == 256 for item in continuation.candidates[:2])
    assert all(item.stage == "boundary" for item in continuation.candidates[2:])


def test_boundary_continuation_follows_axis_order_and_reopens_local_lr() -> None:
    available = {
        item.run_name: _evidence(item, 0.1, 0.04) for item in initial_candidates()
    }
    selected_coordinates = {
        "uniform_catalog": (0.012, 512, None, None),
        "streaming_global_q": (0.012, 1024, 0.01, None),
        "popularity_global_q": (0.012, 1024, None, None),
        "aggregate_uniform_streaming_global_q": (0.012, 1024, 0.02, 0.5),
    }
    for family, coordinates in selected_coordinates.items():
        winner = next(
            item
            for item in initial_candidates()
            if item.family == family
            and (
                item.deep_lr,
                item.negative_count,
                item.alpha,
                item.uniform_fraction,
            )
            == coordinates
        )
        available[winner.run_name] = _evidence(winner, 0.2, 0.05)
        for local in local_lr_candidates(winner):
            available.setdefault(local.run_name, _evidence(local, 0.19, 0.049))

    boundary_plan = build_followup_plan(lambda item: available.get(item.run_name))
    assert len(boundary_plan.candidates) == 2
    negative_boundary, alpha_boundary = boundary_plan.candidates
    assert negative_boundary.family == "uniform_catalog"
    assert negative_boundary.boundary_axis == "negative_count"
    assert alpha_boundary.family == "aggregate_uniform_streaming_global_q"
    assert alpha_boundary.boundary_axis == "alpha"

    available[negative_boundary.run_name] = _evidence(
        negative_boundary, 0.21, 0.051
    )
    available[alpha_boundary.run_name] = _evidence(alpha_boundary, 0.19, 0.049)
    continuation = build_followup_plan(lambda item: available.get(item.run_name))

    assert len(continuation.candidates) == 2
    assert all(item.stage == "local_lr" for item in continuation.candidates)
    assert {item.negative_count for item in continuation.candidates} == {256}

    for local in continuation.candidates:
        recall = 0.22 if local.deep_lr == 0.024 else 0.205
        available[local.run_name] = _evidence(local, recall, 0.052)

    negative_continuation = build_followup_plan(
        lambda item: available.get(item.run_name)
    )
    assert negative_continuation.stage == "boundary"
    assert negative_continuation.candidates[0].negative_count == 128

    outer_loser = negative_continuation.candidates[0]
    available[outer_loser.run_name] = _evidence(outer_loser, 0.19, 0.049)
    lr_continuation = build_followup_plan(lambda item: available.get(item.run_name))

    assert lr_continuation.stage == "boundary"
    assert lr_continuation.candidates[0].boundary_axis == "deep_lr"
    assert lr_continuation.candidates[0].deep_lr == 0.048
    assert lr_continuation.candidates[0].negative_count == 256


def test_diagnostics_wait_for_and_inherit_the_final_mixture_boundary_winner() -> None:
    available = {
        item.run_name: _evidence(item, 0.1, 0.04) for item in initial_candidates()
    }
    selected_coordinates = {
        "uniform_catalog": (0.012, 1024, None, None),
        "popularity_global_q": (0.012, 1024, None, None),
        "streaming_global_q": (0.012, 1024, 0.01, None),
        "aggregate_uniform_streaming_global_q": (0.012, 1024, 0.02, 0.5),
    }
    winners = {}
    for family, coordinates in selected_coordinates.items():
        winner = next(
            item
            for item in initial_candidates()
            if item.family == family
            and (
                item.deep_lr,
                item.negative_count,
                item.alpha,
                item.uniform_fraction,
            )
            == coordinates
        )
        winners[family] = winner
        available[winner.run_name] = _evidence(winner, 0.2, 0.05)
        for local in local_lr_candidates(winner):
            available.setdefault(local.run_name, _evidence(local, 0.19, 0.049))

    first = build_followup_plan(lambda item: available.get(item.run_name))
    assert first.stage == "boundary"
    alpha_004 = first.candidates[0]
    assert alpha_004.family == "aggregate_uniform_streaming_global_q"
    assert alpha_004.boundary_axis == "alpha"
    assert alpha_004.alpha == 0.04

    available[alpha_004.run_name] = _evidence(alpha_004, 0.21, 0.051)
    local_plan = build_followup_plan(lambda item: available.get(item.run_name))
    assert local_plan.stage == "local_lr"
    assert {item.alpha for item in local_plan.candidates} == {0.04}
    for local in local_plan.candidates:
        available[local.run_name] = _evidence(local, 0.205, 0.05)

    next_boundary = build_followup_plan(lambda item: available.get(item.run_name))
    assert next_boundary.stage == "boundary"
    alpha_008 = next_boundary.candidates[0]
    assert alpha_008.alpha == 0.08
    available[alpha_008.run_name] = _evidence(alpha_008, 0.19, 0.049)

    diagnostic_plan = build_followup_plan(lambda item: available.get(item.run_name))

    assert diagnostic_plan.stage == "diagnostic"
    assert len(diagnostic_plan.candidates) == 3
    assert {item.secondary_coordinates for item in diagnostic_plan.candidates} == {
        (1024, 0.04, 0.5)
    }
    assert not {
        item.run_name
        for item in diagnostic_candidates(
            winners["aggregate_uniform_streaming_global_q"]
        )
    } & {item.run_name for item in diagnostic_plan.candidates}


def test_diagnostics_remain_gated_while_any_primary_family_is_unresolved() -> None:
    available = {
        item.run_name: _evidence(item, 0.1, 0.04) for item in initial_candidates()
    }
    selected_coordinates = {
        "uniform_catalog": (0.012, 512, None, None),
        "streaming_global_q": (0.012, 1024, 0.01, None),
        "popularity_global_q": (0.012, 1024, None, None),
        "aggregate_uniform_streaming_global_q": (0.012, 1024, 0.02, 0.5),
    }
    winners = {}
    for family, coordinates in selected_coordinates.items():
        winner = next(
            item
            for item in initial_candidates()
            if item.family == family
            and (
                item.deep_lr,
                item.negative_count,
                item.alpha,
                item.uniform_fraction,
            )
            == coordinates
        )
        winners[family] = winner
        available[winner.run_name] = _evidence(winner, 0.2, 0.05)
        for local in local_lr_candidates(winner):
            available.setdefault(local.run_name, _evidence(local, 0.19, 0.049))
    mixture = winners["aggregate_uniform_streaming_global_q"]
    mixture_alpha_boundary = make_boundary_candidate(mixture, "alpha", "high", 1)
    available[mixture_alpha_boundary.run_name] = _evidence(
        mixture_alpha_boundary, 0.19, 0.049
    )
    for diagnostic in diagnostic_candidates(mixture):
        available[diagnostic.run_name] = _evidence(diagnostic, 0.18, 0.048)

    plan = build_followup_plan(lambda item: available.get(item.run_name))

    assert plan.stage == "boundary"
    assert len(plan.candidates) == 1
    assert plan.candidates[0].family == "uniform_catalog"
    assert plan.candidates[0].boundary_axis == "negative_count"


def test_launcher_inspection_never_enters_the_queue() -> None:
    result = subprocess.run(
        [str(LAUNCHER), "--list-initial"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        env={**os.environ, "G1_TRAINING_QUEUE_LIBRARY": "/does/not/exist"},
    )

    assert result.returncode == 0, result.stderr
    assert len(result.stdout.splitlines()) == 24


def test_launcher_requires_wandb_mode_in_persistent_queue_environment(
    tmp_path: Path,
) -> None:
    queue = tmp_path / "queue.sh"
    queue.write_text(
        "printf 'FORWARD=%s REQUIRED=%s WANDB=%s DATA=%s\\n' "
        '"${TRAINING_QUEUE_FORWARD_ENV:-}" '
        '"${TRAINING_QUEUE_REQUIRED_FORWARD_ENV:-}" '
        '"${WANDB_MODE:-}" "${G1_DATASET_SIZE:-}" >&2\n'
        "enqueue() { return 0; }\n"
        "drain() { return 0; }\n"
    )
    logs = tmp_path / "logs"
    logs.mkdir()

    result = subprocess.run(
        [str(LAUNCHER)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        env={
            **os.environ,
            "G1_TRAINING_QUEUE_LIBRARY": str(queue),
            "G1_RQ11_LOGS": str(logs),
        },
    )

    assert result.returncode == 0, result.stderr
    environment = next(
        line for line in result.stderr.splitlines() if line.startswith("FORWARD=")
    )
    assert "FORWARD=G1_DATASET_SIZE WANDB_MODE" in environment
    assert "REQUIRED=G1_DATASET_SIZE WANDB_MODE" in environment
    assert "WANDB=offline DATA=500m" in environment


def test_report_is_native_only_and_uses_reader_schema() -> None:
    selected = []
    for family in (
        "uniform_catalog",
        "streaming_global_q",
        "popularity_global_q",
        "aggregate_uniform_streaming_global_q",
    ):
        candidate = next(item for item in initial_candidates() if item.family == family)
        selected.append(
            Run(
                candidate=candidate,
                best_epoch=7,
                stopped_epoch=20,
                validation_recall=0.13,
                validation_ndcg=0.05,
                metrics={
                    "recall@100": 0.13,
                    "ndcg@100": 0.05,
                    "recall@10": 0.03,
                    "ndcg@10": 0.02,
                    "coverage@100": 0.6,
                },
            )
        )

    bundle = build_report_bundle(selected)

    assert bundle.evidence["dataset_size"] == "500m"
    assert bundle.evidence["claims_status"] == "pending"
    assert "deep LR" not in bundle.reader_markdown
    assert "run name" not in bundle.reader_markdown
    assert (
        "| negative sampling | negatives | logQ alpha | uniform fraction |"
        in bundle.reader_markdown
    )
    assert "aggregate uniform + streaming global-q" in bundle.tuning_markdown
    assert "negative-only" not in bundle.reader_markdown
    assert "deep LR" in bundle.tuning_markdown


def test_checked_in_reader_retains_broad_and_corrected_rq11_comparisons() -> None:
    report = ROOT / "experiments/g1_sasrec_item_ids_likes"
    readme = (report / "README.md").read_text()
    compact = (report / "scratchpad/research_questions_500m.md").read_text()

    for text in (readme, compact):
        assert "### Earlier broad negative-sampling comparison" in text
        assert "uniform random + fixed logQ on in-batch negatives" in text
        assert "### Corrected uniform/streaming mixture comparison" in text
        assert "aggregate uniform + streaming global-q" in text


def test_readme_sync_preserves_the_earlier_rq11_comparison(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(
        "before\n| historical comparison |\n"
        "<!-- rq11-mixed-streaming-generated:start -->\nold\n"
        "<!-- rq11-mixed-streaming-generated:end -->\nafter\n"
    )

    sync_readme(readme, "| corrected comparison |")

    text = readme.read_text()
    assert "| historical comparison |" in text
    assert "old" not in text
    assert "| corrected comparison |" in text


def test_report_skips_in_progress_candidate_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = initial_candidates()[0]
    (tmp_path / candidate.run_name).mkdir()
    monkeypatch.setattr(
        "experiments.g1_sasrec_item_ids_likes.analysis."
        "rq11_mixed_streaming_report.filesystem_inspector",
        lambda _: lambda __: None,
    )

    bundle = collect_report_bundle(tmp_path)

    assert bundle.evidence["claims_status"] == "pending"


def test_reader_cells_apply_native_500m_absolute_resolution_bands() -> None:
    assert metric_cell("recall@100", 0.134, 0.13) == (
        '<span style="color: green">+3% (0.134)</span>'
    )
    assert metric_cell("recall@100", 0.133, 0.13) == "+2% (0.133)"
    assert metric_cell("ndcg@100", 0.048, 0.05) == (
        '<span style="color: red">-4% (0.048)</span>'
    )
    assert metric_cell("coverage@100", 0.55, 0.6) == "-8% (0.550)"
    assert metric_cell("ndcg@100", 0.051825, 0.051902) == "+0% (0.052)"


@pytest.mark.parametrize(
    ("comparisons", "expected"),
    [
        ({"uniform": "better", "streaming": "better", "popularity": "better"}, "yes"),
        (
            {"uniform": "better", "streaming": "worse", "popularity": "unresolved"},
            "worse",
        ),
        (
            {"uniform": "better", "streaming": "trade-off", "popularity": "unresolved"},
            "trade-off",
        ),
        (
            {"uniform": "better", "streaming": "unresolved", "popularity": "better"},
            "unresolved",
        ),
    ],
)
def test_aggregate_outcome_preserves_non_win_semantics(
    comparisons: dict[str, str], expected: str
) -> None:
    assert aggregate_mixture_outcome(comparisons) == expected
