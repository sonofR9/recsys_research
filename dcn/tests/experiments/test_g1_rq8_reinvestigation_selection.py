from __future__ import annotations

import json
import os
from pathlib import Path
import runpy
import subprocess

import pytest

from experiments.g1_sasrec_item_ids_likes.analysis.rq8_reinvestigation_candidates import (
    candidate_by_run,
    initial_candidates,
    make_boundary_candidate,
    make_confirmation_candidate,
    sequence_initial_candidates,
)
from experiments.g1_sasrec_item_ids_likes.analysis.rq8_reinvestigation_selection import (
    ArtifactEvidence,
    SelectionEvidenceError,
    build_followup_plan,
    filesystem_inspector,
)


ROOT = Path(__file__).resolve().parents[3]
CONFIG = (
    ROOT
    / "experiments/g1_sasrec_item_ids_likes/configs/rq8_reinvestigation_variant.py"
)
LAUNCHER = (
    ROOT
    / "experiments/g1_sasrec_item_ids_likes/launchers/architecture/"
    "rq8_reinvestigation_followups_500m.sh"
)


def _initial_evidence() -> dict[str, ArtifactEvidence]:
    score = {0.006: 0.8, 0.012: 0.9, 0.024: 0.7}
    return {
        candidate.run_name: ArtifactEvidence(
            candidate=candidate,
            validation_recall=score[candidate.deep_lr],
            validation_ndcg=score[candidate.deep_lr] / 2,
        )
        for candidate in initial_candidates()
    }


def _inspect_from(
    evidence: dict[str, ArtifactEvidence],
):
    return lambda candidate: evidence.get(candidate.run_name)


def test_resolved_initial_surface_emits_only_two_repeats_per_query_method() -> None:
    evidence = _initial_evidence()

    plan = build_followup_plan(_inspect_from(evidence))

    assert not plan.boundary
    assert len(plan.confirmations) == 6
    assert {
        (candidate.query_method, candidate.deep_lr, candidate.seed)
        for candidate in plan.confirmations
    } == {
        (method, 0.012, seed)
        for method in ("standard", "end_only", "interleaved")
        for seed in (43, 44)
    }
    assert all(candidate.study == "query" for candidate in plan.confirmations)


def test_high_boundary_extends_one_log2_step_until_the_winner_is_interior() -> None:
    evidence = _initial_evidence()
    standard = [
        candidate
        for candidate in initial_candidates()
        if candidate.study == "query" and candidate.query_method == "standard"
    ]
    for candidate in standard:
        evidence[candidate.run_name] = ArtifactEvidence(
            candidate,
            validation_recall=candidate.deep_lr,
            validation_ndcg=candidate.deep_lr,
        )

    first = build_followup_plan(_inspect_from(evidence))

    assert len(first.boundary) == 1
    probe = first.boundary[0]
    assert (probe.deep_lr, probe.boundary_side, probe.boundary_step) == (
        0.048,
        "high",
        1,
    )
    assert not first.confirmations

    evidence[probe.run_name] = ArtifactEvidence(probe, 0.02, 0.02)
    resolved = build_followup_plan(_inspect_from(evidence))
    assert not resolved.boundary
    assert len(resolved.confirmations) == 6


def test_outer_boundary_winner_emits_the_next_step_and_never_repeats_sequence() -> None:
    evidence = _initial_evidence()
    surface = next(
        candidate
        for candidate in initial_candidates()
        if candidate.study == "sequence"
        and candidate.position_method == "alibi"
        and candidate.max_seq_len == 512
        and candidate.deep_lr == 0.024
    )
    for candidate in initial_candidates():
        if candidate.surface_key == surface.surface_key:
            evidence[candidate.run_name] = ArtifactEvidence(
                candidate,
                candidate.deep_lr,
                candidate.deep_lr,
            )

    first = build_followup_plan(_inspect_from(evidence))
    probe = first.boundary[0]
    assert probe.study == "sequence"
    assert probe.deep_lr == 0.048
    assert all(candidate.study == "query" for candidate in first.confirmations)

    evidence[probe.run_name] = ArtifactEvidence(probe, 0.06, 0.06)
    second = build_followup_plan(_inspect_from(evidence))
    assert [(candidate.deep_lr, candidate.boundary_step) for candidate in second.boundary] == [
        (0.096, 2)
    ]
    assert all(candidate.study == "query" for candidate in second.confirmations)


def test_low_boundary_halves_the_rate_and_candidate_names_round_trip() -> None:
    initial = next(
        candidate
        for candidate in initial_candidates()
        if candidate.study == "query"
        and candidate.query_method == "interleaved"
        and candidate.deep_lr == 0.006
    )
    boundary = make_boundary_candidate(initial, "low", 2)
    confirmation = make_confirmation_candidate(initial, 44)

    assert boundary.deep_lr == 0.0015
    assert candidate_by_run(boundary.run_name) == boundary
    assert candidate_by_run(confirmation.run_name) == confirmation
    assert confirmation.deep_lr == 0.006


def test_followup_config_preserves_the_approved_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial = next(
        candidate
        for candidate in initial_candidates()
        if candidate.study == "sequence"
        and candidate.position_method == "rope_reverse_alibi"
        and candidate.max_seq_len == 512
    )
    candidate = make_boundary_candidate(initial, "high", 1)
    monkeypatch.setenv("G1_RQ8_RUN", candidate.run_name)

    experiment = runpy.run_path(str(CONFIG))["experiment"]

    assert experiment.run_name == candidate.run_name
    assert experiment.size == "500m"
    assert experiment.seed == 42
    assert experiment.max_seq_len == 512
    assert experiment.embedding_learning_rate == 0.064
    assert experiment.deep_learning_rate == 0.048
    assert experiment.dataloader.batch_size == 1280
    assert experiment.dataloader.effective_batch_size == 1280
    assert experiment.dataloader.gradient_accumulation_steps == 1
    assert experiment.transformer.attention_window is None
    assert not experiment.dense_random_negative_scores
    assert experiment.lr_schedule.shape == "linear"
    assert experiment.lr_schedule_horizon_epochs == 20
    assert experiment.num_epochs == 20


def test_missing_initial_artifact_fails_closed() -> None:
    evidence = _initial_evidence()
    evidence.pop(next(iter(evidence)))

    with pytest.raises(SelectionEvidenceError, match="initial surface is incomplete"):
        build_followup_plan(_inspect_from(evidence))


def test_exact_primary_and_secondary_tie_fails_instead_of_adding_a_tiebreaker() -> None:
    evidence = _initial_evidence()
    surface = next(iter(initial_candidates())).surface_key
    candidates = [
        candidate for candidate in initial_candidates() if candidate.surface_key == surface
    ]
    for candidate in candidates[:2]:
        evidence[candidate.run_name] = ArtifactEvidence(candidate, 1.0, 1.0)

    with pytest.raises(SelectionEvidenceError, match="exact validation tie"):
        build_followup_plan(_inspect_from(evidence))


def test_filesystem_inspector_requires_config_verified_best_epoch_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = initial_candidates()[0]
    directory = tmp_path / candidate.run_name
    directory.mkdir()
    (directory / "training_metadata.json").write_text(
        json.dumps({"best_epoch": 2, "stopped_epoch": 20})
    )
    final_metrics = {"num_users": 37018}
    for k in (10, 50, 100):
        final_metrics.update(
            {
                f"{name}@{k}": 0.2
                for name in ("ndcg", "recall", "capped_recall", "mrr", "coverage")
            }
        )
    (directory / "final_metrics.json").write_text(json.dumps(final_metrics))
    (directory / "sweep.log").write_text(
        "epoch 1 finished epoch/val_true.recall@100=0.25 "
        "epoch/val_true.ndcg@100=0.125\n"
    )
    monkeypatch.setattr(
        "experiments.g1_sasrec_item_ids_likes.analysis."
        "rq8_reinvestigation_selection.verify_artifact.verify_config",
        lambda *args: True,
    )

    evidence = filesystem_inspector(tmp_path)(candidate)

    assert evidence == ArtifactEvidence(candidate, 0.25, 0.125)


def test_filesystem_inspector_rejects_truncated_annealing_horizon(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = initial_candidates()[0]
    directory = tmp_path / candidate.run_name
    directory.mkdir()
    (directory / "training_metadata.json").write_text(
        json.dumps({"best_epoch": 2, "stopped_epoch": 19})
    )
    final_metrics = {"num_users": 1}
    for k in (10, 50, 100):
        final_metrics.update(
            {
                f"{name}@{k}": 0.2
                for name in ("ndcg", "recall", "capped_recall", "mrr", "coverage")
            }
        )
    (directory / "final_metrics.json").write_text(json.dumps(final_metrics))
    (directory / "sweep.log").write_text(
        "epoch 1 finished epoch/val_true.recall@100=0.25 "
        "epoch/val_true.ndcg@100=0.125\n"
    )
    monkeypatch.setattr(
        "experiments.g1_sasrec_item_ids_likes.analysis."
        "rq8_reinvestigation_selection.verify_artifact.verify_config",
        lambda *args: True,
    )

    with pytest.raises(SelectionEvidenceError, match="did not finish.*20-epoch"):
        filesystem_inspector(tmp_path)(candidate)


def test_filesystem_inspector_rejects_incomplete_final_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = initial_candidates()[0]
    directory = tmp_path / candidate.run_name
    directory.mkdir()
    (directory / "training_metadata.json").write_text(
        json.dumps({"best_epoch": 2, "stopped_epoch": 20})
    )
    (directory / "final_metrics.json").write_text(json.dumps({"recall@100": 0.2}))
    (directory / "sweep.log").write_text(
        "epoch 1 finished epoch/val_true.recall@100=0.25 "
        "epoch/val_true.ndcg@100=0.125\n"
    )
    monkeypatch.setattr(
        "experiments.g1_sasrec_item_ids_likes.analysis."
        "rq8_reinvestigation_selection.verify_artifact.verify_config",
        lambda *args: True,
    )

    with pytest.raises(SelectionEvidenceError, match="incomplete final metrics"):
        filesystem_inspector(tmp_path)(candidate)


def test_filesystem_inspector_rejects_protocol_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = initial_candidates()[0]
    directory = tmp_path / candidate.run_name
    directory.mkdir()
    for name in ("training_metadata.json", "final_metrics.json", "sweep.log"):
        (directory / name).write_text("{}")
    monkeypatch.setattr(
        "experiments.g1_sasrec_item_ids_likes.analysis."
        "rq8_reinvestigation_selection.verify_artifact.verify_config",
        lambda *args: False,
    )

    with pytest.raises(SelectionEvidenceError, match="protocol-incompatible"):
        filesystem_inspector(tmp_path)(candidate)


def test_followup_launcher_replans_after_each_persistent_queue_wave(
    tmp_path: Path,
) -> None:
    candidates = (
        make_boundary_candidate(sequence_initial_candidates()[0], "high", 1),
        make_boundary_candidate(sequence_initial_candidates()[3], "low", 1),
    )
    selector = tmp_path / "selector.py"
    selector.write_text(
        "import os\n"
        "from pathlib import Path\n"
        "state = Path(os.environ['RQ8_TEST_STATE'])\n"
        "wave = int(state.read_text()) if state.exists() else 0\n"
        f"rows = {[(candidate.run_name, candidate.max_seq_len) for candidate in candidates]!r}\n"
        "if wave < len(rows):\n"
        "    state.write_text(str(wave + 1))\n"
        "    print(*rows[wave], sep='\\t')\n"
    )
    queue = tmp_path / "queue.sh"
    queue.write_text(
        "_test_sealed=0\n"
        "enqueue() { [ \"$_test_sealed\" -eq 0 ] || return 9; "
        "printf 'ENQUEUE %s GROUP=%s\\n' \"$1\" "
        "\"${TRAINING_QUEUE_DATA_GROUP-}\" >&2; }\n"
        "drain() { _test_sealed=1; printf 'DRAIN\\n' >&2; }\n"
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
            "G1_RQ8_FOLLOWUP_SELECTOR": str(selector),
            "RQ8_TEST_STATE": str(tmp_path / "state"),
        },
    )

    assert result.returncode == 0, result.stderr
    for candidate in candidates:
        assert f"ENQUEUE {candidate.run_name}" in result.stderr
        assert (
            f"GROUP=g1-rq8-fullcausal-500m-seq{candidate.max_seq_len}"
            in result.stderr
        )
    assert result.stderr.count("DRAIN") == 2


def test_completed_query_confirmations_are_not_relaunched() -> None:
    evidence = _initial_evidence()
    for method in ("standard", "end_only", "interleaved"):
        winner = next(
            candidate
            for candidate in initial_candidates()
            if candidate.study == "query"
            and candidate.query_method == method
            and candidate.deep_lr == 0.012
        )
        for seed in (43, 44):
            confirmation = make_confirmation_candidate(winner, seed)
            evidence[confirmation.run_name] = ArtifactEvidence(
                confirmation, 0.9, 0.45
            )

    plan = build_followup_plan(_inspect_from(evidence))

    assert not plan.confirmations
