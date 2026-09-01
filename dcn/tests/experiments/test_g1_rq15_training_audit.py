from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments.g1_sasrec_item_ids_likes.analysis.rq15_training_audit import (
    REQUIRED_CHECKS,
    Rq15AuditError,
    build_correctness_audit,
    run_model_correctness_probe,
    validate_correctness_audit,
)
from experiments.g1_sasrec_item_ids_likes.analysis.rq15_training_candidates import (
    initial_candidates,
    source_candidates,
    source_checkpoint_metadata,
)
from experiments.g1_sasrec_item_ids_likes.analysis.rq15_training_report import (
    current_implementation_sha256,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_run(
    logs: Path,
    run_name: str,
    metadata: dict[str, object],
    *,
    checkpoint: bool = False,
) -> dict[str, str]:
    directory = logs / run_name
    directory.mkdir(parents=True)
    (directory / "training_metadata.json").write_text(json.dumps(metadata))
    (directory / "final_metrics.json").write_text('{"recall@100": 0.1}')
    (directory / "sweep.log").write_text("epoch 1 finished\n")
    names = ["training_metadata.json", "final_metrics.json", "sweep.log"]
    if checkpoint:
        (directory / "rq15_first_stage_checkpoint.pt").write_bytes(b"checkpoint")
        names.append("rq15_first_stage_checkpoint.pt")
    return {name: _sha256(directory / name) for name in names}


def _results(logs: Path) -> dict[str, object]:
    selected_source = source_candidates()[1]
    checkpoint_path = selected_source.checkpoint_path(logs).resolve()
    checkpoint_sha256 = hashlib.sha256(b"checkpoint").hexdigest()
    source_records = []
    run_artifacts: dict[str, dict[str, str]] = {}
    for source in source_candidates():
        hashes = _write_run(
            logs,
            source.run_name,
            {
                "dataset_size": "500m",
                "seed": 42,
                "num_epochs": 20,
                "embedding_learning_rate": source.embedding_lr,
                "deep_learning_rate": source.deep_lr,
                "batch_size": 1280,
                "candidate_targets_per_epoch": 0,
                "ntp_targets_per_epoch": 20,
                "expanded_examples_per_epoch": 4,
                "targets_per_epoch": 20,
            },
            checkpoint=True,
        )
        run_artifacts[source.run_name] = hashes
        source_records.append(
            {
                "run_name": source.run_name,
                "checkpoint_sha256": hashes["rq15_first_stage_checkpoint.pt"],
                "artifact_sha256": hashes,
            }
        )

    initialization = {
        "schema_version": 1,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "source_metadata": source_checkpoint_metadata(selected_source),
        "history_position_count": 128,
        "copied_modules": ["item_embedding", "memory_encoder", "tokenizer"],
        "newly_initialized_modules": [
            "decoder",
            "decoder_query",
            "query_projection",
            "query_slots",
        ],
    }
    treatment_records: dict[str, list[dict[str, object]]] = {
        "scratch_candidate_only": [],
        "pretrained_finetune": [],
        "auxiliary_ntp": [],
    }
    selected_records: dict[str, dict[str, object]] = {}
    for candidate in initial_candidates():
        method = candidate.training_method
        auxiliary = method == "auxiliary_ntp"
        pretrained = method == "pretrained_finetune"
        first_stage = initialization if pretrained else "scratch"
        metadata = {
            "dataset_size": "500m",
            "seed": candidate.seed,
            "num_epochs": candidate.horizon_epochs,
            "embedding_learning_rate": candidate.embedding_lr,
            "deep_learning_rate": candidate.deep_lr,
            "batch_size": candidate.batch_size,
            "query_architecture": "decoder_decoder",
            "query_slots_shared": False,
            "include_history_memory": False,
            "num_query_slots": 4,
            "training_method": method,
            "auxiliary_ntp_weight": 1.0 if auxiliary else 0.0,
            "loss_normalization": "candidate_and_ntp_separately_mean_normalized",
            "expanded_examples_per_epoch": 4,
            "candidate_targets_per_epoch": 4,
            "ntp_targets_per_epoch": 7 if auxiliary else 0,
            "targets_per_epoch": 11 if auxiliary else 4,
            "first_stage_initialization": first_stage,
        }
        metadata["transfer_invariants"] = dict(metadata)
        hashes = _write_run(logs, candidate.run_name, metadata)
        run_artifacts[candidate.run_name] = hashes
        record = {
            "run_name": candidate.run_name,
            "training_method": method,
            "checkpoint_sha256": checkpoint_sha256 if pretrained else None,
            "artifact_sha256": hashes,
        }
        treatment_records[method].append(record)
        selected_records.setdefault(method, record)

    return {
        "schema_version": 1,
        "research_question": "RQ15 decoder-decoder training method",
        "dataset_size": "500m",
        "claims_status": "correctness_audit_required",
        "missing_artifacts": [],
        "required_followups": [],
        "artifact_audit": {
            "status": "passed",
            "run_artifacts": run_artifacts,
        },
        "checkpoint_pretraining": source_records[1],
        "checkpoint_pretraining_surface": source_records,
        "scratch_control": selected_records["scratch_candidate_only"],
        "surface_winners": selected_records,
        "treatments": {
            method: {"artifacts": records}
            for method, records in treatment_records.items()
        },
    }


def _probe() -> dict[str, dict[str, object]]:
    return {
        "target_leakage": {
            "target_only_candidate_query_max_delta": 0.0,
            "target_only_ntp_query_max_delta": 0.0,
            "candidate_positive_ids": [4, 12],
            "changed_candidate_positive_ids": [5, 13],
            "ntp_positive_ids": [2, 3, 4, 11, 12],
        },
        "attention_masks": {
            "history_is_causal": True,
            "later_history_to_earlier_state_max_delta": 0.0,
            "later_history_to_later_state_l1": 1.0,
            "other_user_query_max_delta": 0.0,
            "query_slot_to_history_max_delta": 0.0,
            "cross_attention_other_user_max_delta": 0.0,
        },
        "gradient_flow": {
            "candidate_memory_encoder_gradient_l1": 1.0,
            "candidate_decoder_gradient_l1": 1.0,
            "candidate_slot_gradient_l1": [1.0, 1.0, 1.0, 1.0],
            "auxiliary_first_stage_projection_gradient_l1": 1.0,
            "auxiliary_memory_encoder_gradient_l1": 1.0,
        },
        "separate_loss_normalization_and_counts": {
            "candidate_targets": 2,
            "auxiliary_ntp_targets": 5,
            "candidate_accumulation_denominator": 2,
            "auxiliary_accumulation_denominator": 5,
            "combined_loss_delta": 0.0,
            "duplicated_batch_candidate_loss_delta": 0.0,
            "duplicated_batch_auxiliary_loss_delta": 0.0,
        },
        "checkpoint_copy_identity": {
            "copied_item_embedding": True,
            "copied_memory_encoder": True,
            "copied_tokenizer": True,
            "preserved_decoder": True,
            "preserved_decoder_query": True,
            "preserved_query_projection": True,
            "preserved_query_slots": True,
        },
    }


def test_audit_binds_model_checks_recipes_code_and_exact_run_artifacts(
    tmp_path: Path,
) -> None:
    logs = tmp_path / "logs"
    results = _results(logs)
    results_path = tmp_path / "results.json"
    results_path.write_text(json.dumps(results))

    audit = build_correctness_audit(
        logs,
        results_path,
        model_probe=_probe,
        recipe_probe=lambda: {"production_recipes_match": True},
        authoritative_results_probe=lambda: results,
        implementation_hash={"implementation": "a" * 64},
    )

    assert audit["status"] == "passed"
    assert set(audit["checks"]) == REQUIRED_CHECKS
    assert all(check["passed"] for check in audit["checks"].values())
    assert all(len(check["artifact_sha256"]) == 64 for check in audit["checks"].values())
    assert validate_correctness_audit(
        audit,
        results["artifact_audit"]["run_artifacts"],
        audit["result_binding"],
        implementation_hash={"implementation": "a" * 64},
    )["status"] == "passed"


def test_audit_fails_closed_before_boundaries_resolve_or_after_artifact_change(
    tmp_path: Path,
) -> None:
    logs = tmp_path / "logs"
    results = _results(logs)
    results_path = tmp_path / "results.json"
    results_path.write_text(
        json.dumps({**results, "required_followups": [{"run_name": "pending"}]})
    )

    with pytest.raises(Rq15AuditError, match="resolved"):
        build_correctness_audit(
            logs,
            results_path,
            model_probe=_probe,
            recipe_probe=lambda: {},
            authoritative_results_probe=lambda: results,
        )

    results_path.write_text(json.dumps(results))
    first_run = next(iter(results["artifact_audit"]["run_artifacts"]))
    (logs / first_run / "sweep.log").write_text("changed\n")
    with pytest.raises(Rq15AuditError, match="hash"):
        build_correctness_audit(
            logs,
            results_path,
            model_probe=_probe,
            recipe_probe=lambda: {},
            authoritative_results_probe=lambda: results,
        )


def test_audit_rejects_a_coherently_edited_incomplete_surface(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    authoritative = _results(logs)
    results = json.loads(json.dumps(authoritative))
    removed = results["treatments"]["auxiliary_ntp"]["artifacts"].pop()
    del results["artifact_audit"]["run_artifacts"][removed["run_name"]]
    results_path = tmp_path / "results.json"
    results_path.write_text(json.dumps(results))

    with pytest.raises(Rq15AuditError, match="authoritative report replay|incomplete"):
        build_correctness_audit(
            logs,
            results_path,
            model_probe=_probe,
            recipe_probe=lambda: {"production_recipes_match": True},
            authoritative_results_probe=lambda: authoritative,
            implementation_hash={"implementation": "a" * 64},
        )


def test_validator_rejects_tampered_check_payload(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    results = _results(logs)
    results_path = tmp_path / "results.json"
    results_path.write_text(json.dumps(results))
    audit = build_correctness_audit(
        logs,
        results_path,
        model_probe=_probe,
        recipe_probe=lambda: {"production_recipes_match": True},
        authoritative_results_probe=lambda: results,
        implementation_hash={"implementation": "a" * 64},
    )
    audit["checks"]["target_leakage"]["candidate_positive_ids"] = [999]

    with pytest.raises(Rq15AuditError, match="incomplete, stale, or failed"):
        validate_correctness_audit(
            audit,
            results["artifact_audit"]["run_artifacts"],
            audit["result_binding"],
            implementation_hash={"implementation": "a" * 64},
        )


def test_model_probe_covers_every_rq15_correctness_axis() -> None:
    probe = run_model_correctness_probe()

    assert probe["target_leakage"][
        "target_only_candidate_query_max_delta"
    ] == pytest.approx(0.0, abs=1e-6)
    assert probe["target_leakage"][
        "target_only_ntp_query_max_delta"
    ] == pytest.approx(0.0, abs=1e-6)
    assert probe["attention_masks"]["history_is_causal"] is True
    assert probe["attention_masks"][
        "later_history_to_earlier_state_max_delta"
    ] == pytest.approx(0.0, abs=1e-5)
    assert probe["attention_masks"]["later_history_to_later_state_l1"] > 0
    assert probe["attention_masks"]["other_user_query_max_delta"] == pytest.approx(0.0, abs=1e-5)
    assert probe["attention_masks"][
        "query_slot_to_history_max_delta"
    ] == pytest.approx(0.0, abs=1e-5)
    assert probe["attention_masks"][
        "cross_attention_other_user_max_delta"
    ] == pytest.approx(0.0, abs=1e-5)
    assert all(value > 0 for value in probe["gradient_flow"]["candidate_slot_gradient_l1"])
    assert probe["gradient_flow"]["candidate_memory_encoder_gradient_l1"] > 0
    assert probe["gradient_flow"]["candidate_decoder_gradient_l1"] > 0
    assert probe["gradient_flow"]["auxiliary_first_stage_projection_gradient_l1"] > 0
    assert probe["gradient_flow"]["auxiliary_memory_encoder_gradient_l1"] > 0
    assert probe["separate_loss_normalization_and_counts"]["candidate_targets"] == 2
    assert probe["separate_loss_normalization_and_counts"]["auxiliary_ntp_targets"] == 5
    assert probe["separate_loss_normalization_and_counts"][
        "combined_loss_delta"
    ] == pytest.approx(0.0, abs=1e-6)
    assert probe["checkpoint_copy_identity"]["copied_memory_encoder"] is True
    assert probe["checkpoint_copy_identity"]["preserved_decoder"] is True


def test_repository_implementation_hash_covers_the_audit_generator() -> None:
    hashes = current_implementation_sha256()

    assert (
        "experiments/g1_sasrec_item_ids_likes/analysis/rq15_training_audit.py"
        in hashes
    )
    assert "dcn/config/generation.py" in hashes
    assert "dcn/config/retrieval.py" in hashes
    assert "dcn/data/packed.py" in hashes
    assert "dcn/models/sequence_retrieval.py" in hashes
    assert (
        "experiments/g1_sasrec_item_ids_likes/configs/rq8_reinvestigation_variant.py"
        in hashes
    )
    assert "experiments/g1_sasrec_item_ids_likes/configs/variant.py" in hashes
    assert "neuralrec/run/train.py" in hashes
