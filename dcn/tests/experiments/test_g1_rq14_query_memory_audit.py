from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments.g1_sasrec_item_ids_likes.analysis.rq14_query_memory_audit import (
    Rq14AuditError,
    build_correctness_audit,
    validate_correctness_audit,
)
from experiments.g1_sasrec_item_ids_likes.analysis.rq14_query_memory_report import (
    current_implementation_sha256,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _results(logs: Path) -> dict[str, object]:
    treatments = (
        "shared_cls_only",
        "distinct_cls_only",
        "shared_history",
        "distinct_history",
    )
    selected = {}
    artifacts = {}
    for treatment in treatments:
        treatment_artifacts = []
        for lr_slug, deep_lr in (("0006", 0.006), ("0012", 0.012), ("0024", 0.024)):
            run = f"g1_rq14_{treatment}_d{lr_slug}_seed42_h20_r1_500m"
            directory = logs / run
            directory.mkdir(parents=True)
            for name in ("training_metadata.json", "final_metrics.json", "sweep.log"):
                content = f"{run}:{name}\n"
                if name == "training_metadata.json":
                    content = json.dumps(
                        {
                            "query_architecture": "decoder_decoder",
                            "candidate_targets_per_epoch": 100,
                            "expanded_examples_per_epoch": 100,
                            "ntp_targets_per_epoch": 0,
                            "transfer_invariants": {
                                "query_architecture": "decoder_decoder",
                                "candidate_targets_per_epoch": 100,
                                "expanded_examples_per_epoch": 100,
                                "ntp_targets_per_epoch": 0,
                            },
                        }
                    )
                (directory / name).write_text(content)
            hashes = {
                name: _sha(directory / name)
                for name in ("training_metadata.json", "final_metrics.json", "sweep.log")
            }
            artifacts[run] = hashes
            treatment_artifacts.append(
                {
                    "run_name": run,
                    "artifact_sha256": hashes,
                    "validation_curve": [
                        {
                            "epoch": epoch,
                            "recall@100": 0.1 + epoch / 1000,
                            "ndcg@100": 0.05,
                        }
                        for epoch in range(1, 21)
                    ],
                    "stage": "initial",
                    "deep_lr": deep_lr,
                    "best_epoch": 20,
                }
            )
        run = f"g1_rq14_{treatment}_d0012_seed42_h20_r1_500m"
        selected[treatment] = {
            "treatment": treatment,
            "run_name": run,
            "deep_lr": 0.012,
        }
        artifacts[treatment] = treatment_artifacts
    return {
        "research_question": "RQ14 decoder-decoder query memory",
        "dataset_size": "500m",
        "missing_initial_artifacts": [],
        "required_boundary_followups": [],
        "required_followups": [],
        "selected": selected,
        "treatments": {
            treatment: {
                "artifacts": artifacts[treatment]
            }
            for treatment in treatments
        },
    }


def _probe() -> dict[str, object]:
    return {
        "production_recipes_match": True,
        "shared_parameter_rows": 1,
        "distinct_parameter_rows": 4,
        "slot_order_preserved": True,
        "cls_only_memory_lengths": [4, 4],
        "history_memory_lengths": [6, 5],
        "history_precedes_slots": True,
        "target_only_query_max_delta": 0.0,
        "candidate_targets": 2,
        "candidate_targets_per_example": 1,
        "positive_ids": [3, 11],
        "changed_positive_ids": [4, 12],
        "production_target_class": "NextItemTargets",
        "distinct_slot_gradient_l1": [1.0, 1.0, 1.0, 1.0],
        "shared_slot_gradient_l1": [1.0],
        "history_embedding_gradient_l1": 1.0,
        "decoder_gradient_l1": 1.0,
    }


def test_audit_binds_exact_artifacts_implementation_and_model_checks(
    tmp_path: Path,
) -> None:
    logs = tmp_path / "logs"
    results = _results(logs)
    path = tmp_path / "results.json"
    path.write_text(json.dumps(results))

    audit = build_correctness_audit(
        logs,
        path,
        model_probe=_probe,
        implementation_hash={"implementation": "hash"},
    )

    assert audit["status"] == "passed"
    assert set(audit["checks"]) == {
        "artifact_and_recipe_integrity",
        "query_slot_identity_and_order",
        "memory_content_and_lengths",
        "target_exclusion_and_candidate_only_loss",
        "gradient_flow_to_every_slot_and_history",
        "learning_curves_and_lr_boundaries",
    }
    assert all(check["passed"] for check in audit["checks"].values())
    expected = {
        artifact["run_name"]: artifact["artifact_sha256"]
        for treatment in results["treatments"].values()
        for artifact in treatment["artifacts"]
    }
    assert validate_correctness_audit(
        audit,
        expected,
        implementation_hash={"implementation": "hash"},
    )["status"] == "passed"


def test_audit_fails_closed_for_unresolved_results_or_changed_artifact(
    tmp_path: Path,
) -> None:
    logs = tmp_path / "logs"
    results = _results(logs)
    path = tmp_path / "results.json"
    path.write_text(json.dumps({**results, "required_followups": ["pending"]}))

    with pytest.raises(Rq14AuditError, match="resolved"):
        build_correctness_audit(logs, path, model_probe=_probe)

    path.write_text(json.dumps(results))
    first = next(iter(results["selected"].values()))["run_name"]
    (logs / first / "sweep.log").write_text("changed")
    with pytest.raises(Rq14AuditError, match="hash"):
        build_correctness_audit(logs, path, model_probe=_probe)


def test_repository_implementation_hash_covers_rq14_report_and_core() -> None:
    hashes = current_implementation_sha256()

    assert "experiments/g1_sasrec_item_ids_likes/analysis/rq14_query_memory_report.py" in hashes
    assert "experiments/g1_sasrec_item_ids_likes/analysis/rq14_query_memory_explanation.py" in hashes
    assert "dcn/models/cross_attention_retrieval.py" in hashes
    assert all(len(value) == 64 for value in hashes.values())
