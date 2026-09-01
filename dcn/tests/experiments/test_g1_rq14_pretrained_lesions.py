from __future__ import annotations

import json
import os
from pathlib import Path
import runpy

import pytest
import torch

from dcn.config.query_retrieval_training import (
    MuTransferRq14LesionDiagnosticExperiment,
    query_change_diagnostics,
)
from experiments.g1_sasrec_item_ids_likes.analysis.rq14_pretrained_lesion_candidates import (
    diagnostic_candidate_by_run,
    diagnostic_candidates,
)
from experiments.g1_sasrec_item_ids_likes.analysis.rq14_pretrained_lesion_evidence import (
    Rq14LesionEvidenceError,
    build_lesion_explanation,
    classify_lesion_effect,
    validate_selected_rerun_compatibility,
    validate_diagnostic_artifact,
)
from experiments.g1_sasrec_item_ids_likes.analysis.rq15_training_candidates import (
    source_candidates,
)


EXPERIMENT = Path("experiments/g1_sasrec_item_ids_likes")
CONFIG = EXPERIMENT / "configs/rq14_pretrained_lesion_variant.py"
LAUNCHER = EXPERIMENT / "launchers/architecture/rq14_pretrained_lesions_500m.sh"


def _experiment(candidate):
    keys = (
        "G1_RQ14_LESION_RUN",
        "G1_RQ15_SOURCE_RUN",
        "G1_RQ15_FIRST_STAGE_CHECKPOINT",
    )
    previous = {key: os.environ.get(key) for key in keys}
    try:
        os.environ["G1_RQ14_LESION_RUN"] = candidate.run_name
        os.environ["G1_RQ15_SOURCE_RUN"] = source_candidates()[1].run_name
        os.environ["G1_RQ15_FIRST_STAGE_CHECKPOINT"] = "/tmp/exact-rq15-source.pt"
        return runpy.run_path(str(CONFIG))["experiment"]
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_manifest_has_one_selected_diagnostic_cell_per_treatment() -> None:
    candidates = diagnostic_candidates()

    assert len(candidates) == 4
    assert len({candidate.run_name for candidate in candidates}) == 4
    assert {candidate.treatment for candidate in candidates} == {
        "shared_cls_only",
        "distinct_cls_only",
        "shared_history",
        "distinct_history",
    }
    assert {candidate.deep_lr for candidate in candidates} == {0.00075}
    assert all(
        diagnostic_candidate_by_run(item.run_name) == item for item in candidates
    )
    assert all("lesions" in item.run_name for item in candidates)


@pytest.mark.parametrize("candidate", diagnostic_candidates())
def test_diagnostic_recipe_preserves_the_selected_training_protocol(candidate) -> None:
    experiment = _experiment(candidate)

    assert isinstance(experiment, MuTransferRq14LesionDiagnosticExperiment)
    assert experiment.run_name == candidate.run_name
    assert experiment.size == "500m"
    assert experiment.seed == 42
    assert experiment.dataloader.effective_batch_size == 1280
    assert experiment.embedding_learning_rate == 0.00025
    assert experiment.deep_learning_rate == 0.00075
    assert experiment.num_epochs == 20
    assert experiment.lr_schedule.shape == "linear"
    assert experiment.lr_schedule.optimizer_group_scope == "both"
    assert experiment.training_method == "pretrained_finetune"
    assert experiment.auxiliary_ntp_weight == 0
    assert experiment.query_slots_shared is candidate.query_slots_shared
    assert experiment.include_history_memory is candidate.include_history_memory
    assert experiment.diagnostic_lesions == candidate.lesions


def test_query_change_diagnostics_fail_closed_and_measure_use() -> None:
    user_ids = torch.tensor([11, 22])
    normal = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    changed = torch.tensor([[1.0, 1.0], [0.0, 2.0]])

    result = query_change_diagnostics(user_ids, normal, user_ids.clone(), changed)

    assert result["num_users"] == 2
    assert result["changed_user_fraction"] == pytest.approx(0.5)
    assert result["mean_l2_change"] > 0
    assert result["max_l2_change"] == pytest.approx(1.0)

    with pytest.raises(ValueError, match="user identity"):
        query_change_diagnostics(user_ids, normal, user_ids.flip(0), changed)
    with pytest.raises(ValueError, match="finite"):
        query_change_diagnostics(
            user_ids, normal, user_ids, changed.fill_(float("nan"))
        )
    with pytest.raises(ValueError, match="at least one user"):
        query_change_diagnostics(
            torch.empty(0, dtype=torch.long),
            torch.empty(0, 2),
            torch.empty(0, dtype=torch.long),
            torch.empty(0, 2),
        )


def test_launcher_queues_exactly_four_diagnostic_runs_and_generates_evidence() -> None:
    text = LAUNCHER.read_text()

    assert "diagnostic_candidates" in text
    assert "exactly four" in text
    assert "utils/training_queue/queue.sh" in text
    assert 'enqueue "$run"' in text
    assert "rq14_pretrained_lesion_evidence" in text
    assert "G1_RQ14_LESION_RUN" in text


def _metrics(recall: float = 0.16, ndcg: float = 0.065) -> dict[str, float]:
    return {
        "recall@10": 0.04,
        "ndcg@10": 0.03,
        "recall@100": recall,
        "ndcg@100": ndcg,
        "coverage@100": 0.5,
    }


def _query_change(changed_fraction: float) -> dict[str, float | int]:
    changed = changed_fraction > 0
    return {
        "num_users": 100,
        "changed_user_fraction": changed_fraction,
        "mean_l2_change": 0.1 if changed else 0.0,
        "max_l2_change": 0.2 if changed else 0.0,
        "mean_relative_l2_change": 0.03 if changed else 0.0,
        "mean_cosine_distance": 0.001 if changed else 0.0,
    }


def _diagnostic_document(candidate, *, changed_fraction: float = 1.0):
    normal = _metrics()
    return {
        "schema_version": 1,
        "run_name": candidate.run_name,
        "dataset_size": "500m",
        "treatment": {
            "query_slots_shared": candidate.query_slots_shared,
            "include_history_memory": candidate.include_history_memory,
        },
        "training_protocol": {
            "seed": 42,
            "effective_batch_size": 1280,
            "embedding_learning_rate": 0.00025,
            "deep_learning_rate": 0.00075,
            "horizon_epochs": 20,
            "lr_schedule": "linear",
            "lr_schedule_optimizer_group_scope": "both",
            "training_method": "pretrained_finetune",
            "auxiliary_ntp_weight": 0.0,
            "source_checkpoint_sha256": "a" * 64,
            "best_checkpoint_restored": True,
        },
        "best_model_state_sha256": "b" * 64,
        "normal_metrics": normal,
        "lesions": {
            name: {
                "metrics": _metrics(recall=0.159, ndcg=0.0645),
                "query_change": _query_change(changed_fraction),
            }
            for name in candidate.lesions
        },
    }


def test_diagnostic_artifact_validation_fails_closed_on_training_or_lesion_changes() -> (
    None
):
    candidate = diagnostic_candidates()[2]
    document = _diagnostic_document(candidate)
    metadata = {
        "dataset_size": "500m",
        "seed": 42,
        "effective_batch_size": 1280,
        "embedding_learning_rate": 0.00025,
        "deep_learning_rate": 0.00075,
        "stopped_epoch": 20,
        "lr_horizon_complete": True,
        "selection_resolved": True,
        "training_method": "pretrained_finetune",
        "ntp_targets_per_epoch": 0,
        "auxiliary_ntp_weight": 0.0,
        "query_slots_shared": candidate.query_slots_shared,
        "include_history_memory": candidate.include_history_memory,
        "num_query_slots": 4,
        "diagnostic_protocol": "rq14_selected_cell_inference_lesions_v1",
        "diagnostic_lesions": list(candidate.lesions),
        "diagnostic_full_user_evaluation": True,
        "diagnostic_after_best_checkpoint_restore": True,
    }

    validated = validate_diagnostic_artifact(
        candidate, metadata, _metrics(), document, expected_checkpoint_sha256="a" * 64
    )
    assert validated["normal_metrics"] == _metrics()

    metadata.pop("lr_horizon_complete")
    with pytest.raises(Rq14LesionEvidenceError, match="training invariant"):
        validate_diagnostic_artifact(
            candidate,
            metadata,
            _metrics(),
            document,
            expected_checkpoint_sha256="a" * 64,
        )
    metadata["lr_horizon_complete"] = True
    document["lesions"].pop(candidate.lesions[-1])
    with pytest.raises(Rq14LesionEvidenceError, match="lesion set"):
        validate_diagnostic_artifact(
            candidate,
            metadata,
            _metrics(),
            document,
            expected_checkpoint_sha256="a" * 64,
        )


def test_explanation_distinguishes_used_redundant_states_from_ignored_states() -> None:
    normal = _metrics()
    used = classify_lesion_effect(normal, _metrics(0.159, 0.0645), _query_change(1.0))
    ignored = classify_lesion_effect(normal, normal, _query_change(0.0))

    assert used["state_use"] == "states_used"
    assert used["recommendation_effect"] == "within_noise_or_redundant"
    assert "gain" not in json.dumps(used).lower()
    assert ignored["state_use"] == "states_ignored"

    evidence = {
        "schema_version": 1,
        "status": "passed",
        "claims_status": "diagnostics_complete_claims_not_published",
        "runs": {
            candidate.treatment: {
                "normal_metrics": normal,
                "lesions": {
                    name: {
                        "metrics": normal,
                        "query_change": _query_change(0.0),
                    }
                    for name in candidate.lesions
                },
            }
            for candidate in diagnostic_candidates()
        },
    }
    explanation = build_lesion_explanation(evidence)
    assert explanation["status"] == "passed"
    assert explanation["summary"]["states_ignored"] == 18


def test_diagnostic_rerun_must_match_the_selected_source_within_shared_bands() -> None:
    source = {"full_user_metrics": {"recall@100": 0.16, "ndcg@100": 0.065}}

    assert validate_selected_rerun_compatibility(source, _metrics(), "selected") == {
        "diagnostic_minus_source_recall@100": 0.0,
        "diagnostic_minus_source_ndcg@100": 0.0,
    }
    with pytest.raises(Rq14LesionEvidenceError, match="beyond the band"):
        validate_selected_rerun_compatibility(
            source, _metrics(recall=0.156, ndcg=0.065), "selected"
        )
