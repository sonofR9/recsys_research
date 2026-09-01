from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from experiments.g1_sasrec_item_ids_likes.analysis import (
    rq15_training_explanation as explanation_module,
)
from experiments.g1_sasrec_item_ids_likes.analysis.rq15_training_explanation import (
    Rq15ExplanationError,
    build_training_budget_explanation,
    validate_training_budget_explanation,
)


EVIDENCE = Path("experiments/g1_sasrec_item_ids_likes/evidence")


def _load(name: str) -> dict[str, object]:
    value = json.loads((EVIDENCE / name).read_text())
    assert isinstance(value, dict)
    return value


def test_explanation_recomputes_quality_and_three_training_budgets() -> None:
    results = _load("rq15_training_results.json")
    correctness = _load("rq15_training_correctness.json")

    explanation = build_training_budget_explanation(results, correctness)

    comparison = explanation["targeted_evidence"]["quality_compute_tradeoff"]
    budgets = comparison["budgets"]
    assert budgets["scratch_from_random_initialization_seconds"] == pytest.approx(
        12.4574
    )
    assert budgets["fine_tuning_stage_only_seconds"] == pytest.approx(55.9119)
    assert budgets["pretraining_plus_fine_tuning_seconds"] == pytest.approx(413.4405)
    assert comparison["full_recall@100_delta"] == pytest.approx(
        0.07912395842732294
    )
    assert comparison["full_ndcg@100_delta"] == pytest.approx(
        0.029792979704747043
    )
    assert comparison["quality_improved"] is True
    assert comparison["quality_resolution_bands"] == {
        "recall@100": 0.003,
        "ndcg@100": 0.001,
    }
    assert explanation["acceptance_protocol"]["criterion"] == (
        results["acceptance_criterion"]
    )
    assert comparison["cold_start_faster"] is False
    assert validate_training_budget_explanation(
        explanation, results, correctness
    )["status"] == "passed"


def test_explanation_proves_reuse_cannot_make_this_recipe_faster_than_scratch() -> None:
    explanation = build_training_budget_explanation(
        _load("rq15_training_results.json"),
        _load("rq15_training_correctness.json"),
    )

    amortization = explanation["targeted_evidence"]["amortization_bound"]
    assert amortization["fine_tuning_stage_exceeds_scratch"] is True
    assert amortization["asymptotic_seconds_per_fine_tune"] == pytest.approx(55.9119)
    assert amortization["finite_reuse_count_can_beat_scratch"] is False
    assert amortization["measured_reuse_count"] is None
    assert "does not establish an amortized serving or retraining win" in (
        amortization["claim"]
    )


def test_explanation_attributes_cost_gap_to_optimization_path_not_throughput() -> None:
    explanation = build_training_budget_explanation(
        _load("rq15_training_results.json"),
        _load("rq15_training_correctness.json"),
    )

    path = explanation["targeted_evidence"]["optimization_path"]
    assert path["scratch_best_epoch"] == 4
    assert path["pretrained_fine_tuning_best_epoch"] == 18
    assert path["scratch_lrs"] == {"embedding": 0.016, "deep": 0.003}
    assert path["pretrained_fine_tuning_lrs"] == {
        "embedding": 0.00025,
        "deep": 0.00075,
    }
    assert path["candidate_targets_per_epoch_identical"] is True
    assert path["input_tokens_per_epoch_identical"] is True
    assert path["throughput_comparable_within_5_percent"] is True
    assert path["checkpoint_pretraining_horizon_epochs"] == 20
    assert path["checkpoint_pretraining_best_epoch"] == 16
    assert path["pretrained_epoch_1_validation_recall@100"] == pytest.approx(0.0978)
    assert path["scratch_best_validation_recall@100"] == pytest.approx(0.0791)
    assert path["pretrained_epoch_1_already_above_scratch_best"] is True
    assert "does not establish learning-rate causality" in path["claim"]


def test_explanation_rejects_unresolved_or_artifact_tampered_results() -> None:
    results = _load("rq15_training_results.json")
    correctness = _load("rq15_training_correctness.json")
    unresolved = deepcopy(results)
    unresolved["required_followups"] = [{"run_name": "pending"}]

    with pytest.raises(Rq15ExplanationError, match="resolved"):
        build_training_budget_explanation(unresolved, correctness)

    tampered = deepcopy(results)
    tampered["surface_winners"]["pretrained_finetune"]["artifact_sha256"][
        "sweep.log"
    ] = "0" * 64
    with pytest.raises(Rq15ExplanationError, match="artifact"):
        build_training_budget_explanation(tampered, correctness)


def test_explanation_rejects_claim_values_not_replayed_from_raw_artifact() -> None:
    results = _load("rq15_training_results.json")
    correctness = _load("rq15_training_correctness.json")
    tampered = deepcopy(results)
    winner = tampered["surface_winners"]["pretrained_finetune"]
    treatment = tampered["treatments"]["pretrained_finetune"]["artifacts"]
    changed_recall = winner["full_user_metrics"]["recall@100"] + 0.01
    winner["full_user_metrics"]["recall@100"] = changed_recall
    matching = [item for item in treatment if item["run_name"] == winner["run_name"]]
    assert len(matching) == 1
    matching[0]["full_user_metrics"]["recall@100"] = changed_recall

    with pytest.raises(Rq15ExplanationError, match="raw metrics"):
        build_training_budget_explanation(tampered, correctness)


def test_explanation_rejects_tampered_acceptance_protocol() -> None:
    results = _load("rq15_training_results.json")
    results["acceptance_criterion"] = "Pretraining speed is not part of acceptance."

    with pytest.raises(Rq15ExplanationError, match="resolved"):
        build_training_budget_explanation(
            results, _load("rq15_training_correctness.json")
        )


def test_explanation_accepts_only_consistent_user_validated_completion() -> None:
    results = _load("rq15_training_results.json")
    correctness = _load("rq15_training_correctness.json")
    complete = deepcopy(results)
    complete["claims_status"] = "complete"
    complete["result_claims_user_validated"] = True

    build_training_budget_explanation(complete, correctness)

    inconsistent = deepcopy(complete)
    inconsistent["result_claims_user_validated"] = False
    with pytest.raises(Rq15ExplanationError, match="resolved"):
        build_training_budget_explanation(inconsistent, correctness)


def test_saved_explanation_binds_protocol_and_empirical_band_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    saved = _load("rq15_training_explanation.json")
    results = _load("rq15_training_results.json")
    correctness = _load("rq15_training_correctness.json")
    source = Path(
        "experiments/g1_sasrec_item_ids_likes/scratchpad/baseline_spread_500m.json"
    )
    changed_source = tmp_path / source.name
    changed_source.write_text(source.read_text() + "\n")
    monkeypatch.setattr(
        explanation_module, "_BASELINE_SPREAD_EVIDENCE", changed_source
    )

    with pytest.raises(Rq15ExplanationError, match="stale"):
        validate_training_budget_explanation(saved, results, correctness)


def test_saved_explanation_binds_authoritative_rq15_protocol_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    saved = _load("rq15_training_explanation.json")
    source = Path(
        "experiments/g1_sasrec_item_ids_likes/protocol/rq12_rq15_architecture_plan.md"
    )
    changed_source = tmp_path / source.name
    changed_source.write_text(source.read_text() + "\n")
    monkeypatch.setattr(explanation_module, "_ARCHITECTURE_PROTOCOL", changed_source)

    with pytest.raises(Rq15ExplanationError, match="stale"):
        validate_training_budget_explanation(
            saved,
            _load("rq15_training_results.json"),
            _load("rq15_training_correctness.json"),
        )


def test_explanation_validation_rejects_changed_claim_or_input() -> None:
    results = _load("rq15_training_results.json")
    correctness = _load("rq15_training_correctness.json")
    explanation = build_training_budget_explanation(results, correctness)
    changed = deepcopy(explanation)
    changed["conclusion"] = "unsupported"

    with pytest.raises(Rq15ExplanationError, match="stale"):
        validate_training_budget_explanation(changed, results, correctness)

    changed_results = deepcopy(results)
    changed_results["acceptance"]["scratch_time_to_checkpoint_seconds"] += 1
    with pytest.raises(Rq15ExplanationError):
        validate_training_budget_explanation(
            explanation, changed_results, correctness
        )


def test_saved_explanation_matches_current_bound_evidence() -> None:
    results = _load("rq15_training_results.json")
    correctness = _load("rq15_training_correctness.json")

    assert validate_training_budget_explanation(
        _load("rq15_training_explanation.json"), results, correctness
    )["status"] == "passed"
