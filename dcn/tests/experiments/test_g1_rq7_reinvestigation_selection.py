from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
import subprocess

import pytest

from experiments.g1_sasrec_item_ids_likes.analysis.rq7_reinvestigation_candidates import (
    candidate_by_run,
    diagnostic_candidates,
    initial_candidates,
    make_rope_base_extension_candidates,
    rope_base_candidates,
)
from experiments.g1_sasrec_item_ids_likes.analysis.rq7_reinvestigation_selection import (
    ArtifactEvidence,
    SelectionEvidenceError,
    build_followup_plan,
    filesystem_inspector,
    require_diagnostic_gate,
)


ROOT = Path(__file__).resolve().parents[3]
LAUNCHER = ROOT / (
    "experiments/g1_sasrec_item_ids_likes/launchers/architecture/"
    "rq7_reinvestigation_followups_500m.sh"
)


def _evidence(candidates=initial_candidates()) -> dict[str, ArtifactEvidence]:
    score = {0.006: 0.8, 0.012: 0.9, 0.024: 0.7}
    return {
        candidate.run_name: ArtifactEvidence(
            candidate,
            score[candidate.deep_lr],
            score[candidate.deep_lr] / 2,
            score[candidate.deep_lr],
            score[candidate.deep_lr] / 2,
        )
        for candidate in candidates
    }


def _inspect(evidence: dict[str, ArtifactEvidence]):
    return lambda candidate: evidence.get(candidate.run_name)


def _set_final(
    evidence: dict[str, ArtifactEvidence], treatment: str, recall: float, ndcg: float
) -> None:
    for name, item in tuple(evidence.items()):
        if item.candidate.treatment == treatment:
            evidence[name] = ArtifactEvidence(
                item.candidate,
                item.validation_recall,
                item.validation_ndcg,
                recall,
                ndcg,
            )


def test_outer_rope_base_extensions_are_predeclared_exact_surfaces() -> None:
    low = make_rope_base_extension_candidates("low")
    high = make_rope_base_extension_candidates("high")

    assert {(item.treatment, item.deep_lr) for item in low} == {
        ("rope_forward_base10", rate) for rate in (0.006, 0.012, 0.024)
    }
    assert {(item.treatment, item.deep_lr) for item in high} == {
        ("rope_forward_base100000", rate) for rate in (0.006, 0.012, 0.024)
    }
    assert all(item.stage == "rope_base_extension" for item in low + high)
    assert all(candidate_by_run(item.run_name) == item for item in low + high)


def test_missing_initial_artifact_fails_closed() -> None:
    evidence = _evidence()
    evidence.pop(next(iter(evidence)))

    with pytest.raises(SelectionEvidenceError, match="initial surface is incomplete"):
        build_followup_plan(_inspect(evidence))


def test_native50_gate_requires_every_exact_current_diagnostic() -> None:
    evidence = _evidence(diagnostic_candidates())
    evidence.pop(diagnostic_candidates()[0].run_name)

    with pytest.raises(
        SelectionEvidenceError, match="diagnostic surface is incomplete"
    ):
        require_diagnostic_gate(_inspect(evidence))


def test_native50_gate_compares_only_r7_combined_arms_to_matched_controls() -> None:
    evidence = _evidence(diagnostic_candidates())
    require_diagnostic_gate(_inspect(evidence))
    _set_final(evidence, "learned_forward_reverse_add", 0.896, 0.45)

    with pytest.raises(SelectionEvidenceError, match="learned_forward_reverse_add"):
        require_diagnostic_gate(_inspect(evidence))


def test_native50_gate_enforces_ndcg_band_independently_of_recall() -> None:
    evidence = _evidence(diagnostic_candidates())
    _set_final(evidence, "learned_forward_reverse_add", 0.904, 0.44)

    with pytest.raises(SelectionEvidenceError, match="learned_forward_reverse_add"):
        require_diagnostic_gate(_inspect(evidence))


def test_native50_gate_accepts_exact_noninferiority_boundaries() -> None:
    evidence = _evidence(diagnostic_candidates())
    _set_final(evidence, "learned_forward_add", 0.9, 0.45)
    _set_final(evidence, "learned_forward_reverse_add", 0.897, 0.449)

    require_diagnostic_gate(_inspect(evidence))


def test_selection_surface_uses_current_forward_and_combined_identities_only() -> None:
    forward_concat = [
        candidate
        for candidate in initial_candidates()
        if candidate.position.learned_positions == "forward"
        and candidate.position.learned_position_fusion == "concat"
    ]
    combined = [
        candidate
        for candidate in initial_candidates()
        if candidate.position.learned_positions == ("forward", "reverse")
    ]

    assert len(forward_concat) == 6
    assert all(candidate.implementation_revision == 3 for candidate in forward_concat)
    assert all("_r3_500m" in candidate.run_name for candidate in forward_concat)
    assert len(combined) == 12
    assert all(candidate.implementation_revision == 7 for candidate in combined)
    assert all("_r7_500m" in candidate.run_name for candidate in combined)


def test_historical_r6_cannot_fill_current_diagnostic_or_native_surfaces() -> None:
    diagnostic = _evidence(diagnostic_candidates())
    combined_diagnostic = next(
        candidate
        for candidate in diagnostic_candidates()
        if candidate.treatment == "learned_forward_reverse_add"
    )
    item = diagnostic.pop(combined_diagnostic.run_name)
    historical_name = combined_diagnostic.run_name.replace("_r7_", "_r6_")
    historical = candidate_by_run(historical_name)
    diagnostic[historical.run_name] = replace(item, candidate=historical)

    with pytest.raises(
        SelectionEvidenceError, match="diagnostic surface is incomplete"
    ):
        require_diagnostic_gate(_inspect(diagnostic))

    native = _evidence()
    combined_native = next(
        candidate
        for candidate in initial_candidates()
        if candidate.treatment == "learned_forward_reverse_add"
        and candidate.deep_lr == 0.012
    )
    item = native.pop(combined_native.run_name)
    historical_name = combined_native.run_name.replace("_r7_", "_r6_")
    historical = candidate_by_run(historical_name)
    native[historical.run_name] = replace(item, candidate=historical)

    with pytest.raises(SelectionEvidenceError, match="initial surface is incomplete"):
        build_followup_plan(_inspect(native))


@pytest.mark.parametrize("legacy_revision", [1, 2])
def test_legacy_concat_candidate_cannot_fill_the_corrected_surface(
    legacy_revision: int,
) -> None:
    corrected = next(
        candidate
        for candidate in initial_candidates()
        if candidate.position.learned_position_fusion == "concat"
    )

    with pytest.raises(ValueError, match="historical RQ7 implementation revisions"):
        replace(corrected, implementation_revision=legacy_revision)


def test_plain_rope_materially_below_alibi_emits_complete_lower_base_axis() -> None:
    evidence = _evidence()
    _set_final(evidence, "alibi", 0.9, 0.45)
    _set_final(evidence, "rope_forward_base10000", 0.89, 0.44)

    plan = build_followup_plan(_inspect(evidence))

    assert plan.rope_base == rope_base_candidates()
    assert not plan.boundary
    assert not plan.confirmations


@pytest.mark.parametrize(
    ("best_treatment", "outer_treatment"),
    [
        ("rope_forward_base100", "rope_forward_base10"),
        ("rope_forward_base10000", "rope_forward_base100000"),
    ],
)
def test_completed_base_axis_extends_once_only_at_an_outer_winner(
    best_treatment: str, outer_treatment: str
) -> None:
    evidence = _evidence()
    evidence.update(_evidence(rope_base_candidates()))
    _set_final(evidence, "alibi", 0.9, 0.45)
    _set_final(evidence, "rope_forward_base10000", 0.89, 0.44)
    for name, item in tuple(evidence.items()):
        if item.candidate.treatment in {
            "rope_forward_base100",
            "rope_forward_base1000",
            "rope_forward_base10000",
        }:
            bonus = 0.05 if item.candidate.treatment == best_treatment else 0.0
            evidence[name] = ArtifactEvidence(
                item.candidate,
                item.validation_recall + bonus,
                item.validation_ndcg,
                item.final_recall,
                item.final_ndcg,
            )

    plan = build_followup_plan(_inspect(evidence))

    assert {item.treatment for item in plan.rope_base} == {outer_treatment}
    assert len(plan.rope_base) == 3


def test_deep_lr_boundary_is_geometric_until_the_winner_is_interior() -> None:
    evidence = _evidence()
    target = "alibi"
    for name, item in tuple(evidence.items()):
        if item.candidate.treatment == target:
            evidence[name] = ArtifactEvidence(
                item.candidate,
                item.candidate.deep_lr,
                item.candidate.deep_lr,
                item.final_recall,
                item.final_ndcg,
            )

    first = build_followup_plan(_inspect(evidence))
    assert [(item.deep_lr, item.boundary_step) for item in first.boundary] == [
        (0.048, 1)
    ]

    probe = first.boundary[0]
    evidence[probe.run_name] = ArtifactEvidence(probe, 0.06, 0.06, 0.9, 0.45)
    second = build_followup_plan(_inspect(evidence))
    assert [(item.deep_lr, item.boundary_step) for item in second.boundary] == [
        (0.096, 2)
    ]


def test_close_plain_rope_emits_exact_three_method_seed_confirmations() -> None:
    evidence = _evidence()
    _set_final(evidence, "alibi", 0.9, 0.45)
    _set_final(evidence, "rope_forward_base10000", 0.898, 0.449)

    plan = build_followup_plan(_inspect(evidence))

    assert {(item.treatment, item.seed) for item in plan.confirmations} == {
        (treatment, seed)
        for treatment in (
            "alibi",
            "rope_forward_base10000",
            "rope_forward_base10000_alibi",
        )
        for seed in (43, 44)
    }


def test_selected_lower_base_is_the_plain_rope_confirmation_arm() -> None:
    evidence = _evidence()
    evidence.update(_evidence(rope_base_candidates()))
    _set_final(evidence, "alibi", 0.9, 0.45)
    _set_final(evidence, "rope_forward_base10000", 0.89, 0.44)
    for name, item in tuple(evidence.items()):
        if item.candidate.treatment == "rope_forward_base1000":
            evidence[name] = ArtifactEvidence(
                item.candidate,
                item.validation_recall + 0.05,
                item.validation_ndcg,
                0.899,
                0.4495,
            )

    plan = build_followup_plan(_inspect(evidence))

    assert not plan.rope_base
    assert {(item.treatment, item.seed) for item in plan.confirmations} == {
        (treatment, seed)
        for treatment in (
            "alibi",
            "rope_forward_base1000",
            "rope_forward_base10000_alibi",
        )
        for seed in (43, 44)
    }


def test_filesystem_inspector_uses_exact_best_epoch_and_final_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = initial_candidates()[0]
    directory = tmp_path / candidate.run_name
    directory.mkdir()
    (directory / "training_metadata.json").write_text(
        json.dumps({"best_epoch": 2, "stopped_epoch": 20})
    )
    metrics = {"num_users": 37018}
    for k in (10, 50, 100):
        metrics.update(
            {
                f"{name}@{k}": 0.2
                for name in ("ndcg", "recall", "capped_recall", "mrr", "coverage")
            }
        )
    metrics["recall@100"] = 0.3
    metrics["ndcg@100"] = 0.15
    (directory / "final_metrics.json").write_text(json.dumps(metrics))
    (directory / "sweep.log").write_text(
        "epoch 1 finished epoch/val_true.recall@100=0.25 "
        "epoch/val_true.ndcg@100=0.125\n"
    )
    monkeypatch.setattr(
        "experiments.g1_sasrec_item_ids_likes.analysis."
        "rq7_reinvestigation_selection.verify_artifact.verify_config",
        lambda *args: True,
    )

    assert filesystem_inspector(tmp_path)(candidate) == ArtifactEvidence(
        candidate, 0.25, 0.125, 0.3, 0.15
    )


def test_followup_launcher_replans_after_each_queue_wave(tmp_path: Path) -> None:
    candidates = (
        make_rope_base_extension_candidates("low")[0],
        make_rope_base_extension_candidates("high")[0],
    )
    selector = tmp_path / "selector.py"
    selector.write_text(
        "import os\nfrom pathlib import Path\n"
        "state = Path(os.environ['RQ7_TEST_STATE'])\n"
        "wave = int(state.read_text()) if state.exists() else 0\n"
        f"rows = {[item.run_name for item in candidates]!r}\n"
        "if wave < len(rows):\n"
        " state.write_text(str(wave + 1))\n print(rows[wave])\n"
    )
    queue = tmp_path / "queue.sh"
    queue.write_text(
        "_sealed=0\n"
        'enqueue() { [ "$_sealed" -eq 0 ] || return 9; '
        'printf \'ENQUEUE %s GROUP=%s\\n\' "$1" "${TRAINING_QUEUE_DATA_GROUP-}" >&2; }\n'
        "drain() { _sealed=1; printf 'DRAIN\\n' >&2; }\n"
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
            "G1_RQ7_LOGS": str(logs),
            "G1_RQ7_FOLLOWUP_SELECTOR": str(selector),
            "RQ7_TEST_STATE": str(tmp_path / "state"),
        },
    )

    assert result.returncode == 0, result.stderr
    assert result.stderr.count("DRAIN") == 2
    assert result.stderr.count("GROUP=g1-rq7-500m-seq128") == 2
