from pathlib import Path

import polars as pl
import pytest

from experiments.g1_sasrec_item_ids_likes.analysis.rq13_prefix_expansion_audit import (
    REQUIRED_CHECKS,
    Rq13AuditError,
    audit_prefix_caches,
    current_implementation_sha256,
    eligible_target_counts_from_cache,
    run_model_correctness_probe,
    validate_audit_generation_stage,
    validate_bound_stage_one_audit,
    validate_correctness_audit,
)
from experiments.g1_sasrec_item_ids_likes.analysis.rq13_rq14_query_candidates import (
    make_selected_cap_candidates,
)


def _write_source(root: Path) -> Path:
    rows = []
    for user_id, length in ((11, 20), (29, 3)):
        rows.extend(
            {
                "uid": user_id,
                "timestamp": position,
                "compact_item_id": user_id * 100 + position,
                "event_type_id": 2,
            }
            for position in range(length)
        )
    source = root / "source.parquet"
    pl.DataFrame(rows).write_parquet(source)
    return source


def _write_cache(root: Path, cap: int, rows: list[dict], source: Path) -> Path:
    cache = root / f"train_cap{cap}"
    buckets = cache / "buckets"
    buckets.mkdir(parents=True)
    pl.DataFrame(rows).write_parquet(buckets / "bucket_00000.parquet")
    (cache / "metadata.json").write_text(
        "{"
        f'"params":{{"parquet_files":["{source}"],"columns":["compact_item_id"],'
        '"user_column":"uid","emit_user_column":true,"timestamp_column":"timestamp",'
        f'"max_seq_len":128,"min_seq_len":2,"window":"bounded_prefix","stride":1,'
        '"row_filter":"[([(col(\\"event_type_id\\")) == (dyn int: 2)]) & '
        '([(col(\\"timestamp\\")) < (dyn int: 1000)])]","n_buckets":null,'
        f'"prefix_length_rule":"truncated","prefix_cap":{cap}}},'
        f'"bucket_lengths":[{len(rows)}]'
        "}\n"
    )
    return cache


def _prefix_rows(cap: int) -> list[dict]:
    result = []
    for user_id, length in ((11, 20), (29, 3)):
        events = list(range(user_id * 100, user_id * 100 + length))
        for target in range(length - 1, max(0, length - 1 - cap), -1):
            start = max(0, target - 128)
            result.append(
                {
                    "uid": user_id,
                    "timestamp": list(range(start, target + 1)),
                    "compact_item_id": events[start : target + 1],
                }
            )
    return result


def test_prefix_cache_audit_proves_exact_counts_and_latest_slices(
    tmp_path: Path,
) -> None:
    source = _write_source(tmp_path)
    caches = {
        cap: _write_cache(tmp_path, cap, _prefix_rows(cap), source)
        for cap in (1, 8, 16)
    }

    result = audit_prefix_caches(caches)

    assert result["passed"] is True
    assert result["unique_users"] == 2
    assert result["expanded_examples"] == {
        "one_example": 2,
        "truncated_8": 10,
        "truncated_16": 18,
    }
    assert result["source_history_matches"] == {
        "one_example": True,
        "truncated_8": True,
        "truncated_16": True,
    }
    assert result["cap1_is_latest_cap8_slice"] is True
    assert result["cap8_is_latest_cap16_slice"] is True
    assert result["sequence_length_range"] == [2, 20]
    assert sorted(eligible_target_counts_from_cache(caches[16])) == [2, 19]


def test_prefix_cache_audit_covers_cap4_and_fitted_cap_source_slices(
    tmp_path: Path,
) -> None:
    source = _write_source(tmp_path)
    caches = {
        cap: _write_cache(tmp_path, cap, _prefix_rows(cap), source)
        for cap in (1, 4, 8, 16, 17)
    }

    result = audit_prefix_caches(caches)

    assert result["source_history_matches"] == {
        "one_example": True,
        "truncated_4": True,
        "truncated_8": True,
        "truncated_16": True,
        "selected_cap_17": True,
    }
    assert result["nested_latest_slices"] == {
        "1_in_4": True,
        "4_in_8": True,
        "8_in_16": True,
        "16_in_17": True,
    }


def test_prefix_cache_audit_rejects_a_nonlatest_cap8_slice(tmp_path: Path) -> None:
    source = _write_source(tmp_path)
    rows = _prefix_rows(8)
    rows[0] = {**rows[0], "compact_item_id": [999, *rows[0]["compact_item_id"][1:]]}
    caches = {
        1: _write_cache(tmp_path, 1, _prefix_rows(1), source),
        8: _write_cache(tmp_path, 8, rows, source),
        16: _write_cache(tmp_path, 16, _prefix_rows(16), source),
    }

    with pytest.raises(Rq13AuditError, match="latest cap-16 slices"):
        audit_prefix_caches(caches)


def test_prefix_cache_audit_rejects_mutually_nested_but_wrong_source_slices(
    tmp_path: Path,
) -> None:
    source = _write_source(tmp_path)
    rows_by_cap = {cap: _prefix_rows(cap) for cap in (1, 8, 16)}
    for rows in rows_by_cap.values():
        rows[0] = {
            **rows[0],
            "compact_item_id": [*rows[0]["compact_item_id"][:-1], 999],
        }
    caches = {
        cap: _write_cache(tmp_path, cap, rows_by_cap[cap], source) for cap in (1, 8, 16)
    }

    with pytest.raises(Rq13AuditError, match="source histories"):
        audit_prefix_caches(caches)


def test_prefix_cache_audit_proves_latest_128_history_events_for_long_users(
    tmp_path: Path,
) -> None:
    length = 150
    source = tmp_path / "source.parquet"
    pl.DataFrame(
        {
            "uid": [7] * length,
            "timestamp": list(range(length)),
            "compact_item_id": list(range(700, 700 + length)),
            "event_type_id": [2] * length,
        }
    ).write_parquet(source)

    def rows(cap: int) -> list[dict]:
        return [
            {
                "uid": 7,
                "timestamp": list(range(target - 128, target + 1)),
                "compact_item_id": list(range(700 + target - 128, 700 + target + 1)),
            }
            for target in range(length - 1, length - 1 - cap, -1)
        ]

    result = audit_prefix_caches(
        {cap: _write_cache(tmp_path, cap, rows(cap), source) for cap in (1, 8, 16)}
    )

    assert result["sequence_length_range"] == [129, 129]
    assert result["expanded_examples"]["truncated_16"] == 16


def test_model_probe_covers_masks_leakage_gradients_and_candidate_loss() -> None:
    result = run_model_correctness_probe()

    assert result["passed"] is True
    assert result["encoder_is_bidirectional"] is True
    assert result["future_history_changes_earlier_state"] is True
    assert result["other_user_history_max_delta"] == pytest.approx(0.0, abs=1e-6)
    assert result["target_only_query_max_delta"] == pytest.approx(0.0, abs=1e-6)
    assert result["candidate_targets"] == 2
    assert result["candidate_targets_per_example"] == 1
    assert result["duplicated_batch_loss_delta"] == pytest.approx(0.0, abs=1e-6)
    assert result["initial_readout_l1"] == 0
    assert result["bootstrap_readout_gradient_l1"] > 0
    assert result["bootstrap_memory_encoder_gradient_l1"] == 0
    assert result["bootstrap_cross_attention_gradient_l1"] == 0
    assert result["bootstrapped_readout_l1"] > 0
    assert result["memory_encoder_gradient_l1"] > 0
    assert result["cross_attention_gradient_l1"] > 0


def _audit(run_artifacts: dict[str, dict[str, str]]) -> dict:
    treatments = {
        treatment
        for treatment in (
            "one_example",
            "truncated_4",
            "truncated_8",
            "truncated_16",
            "required_8",
            "required_16",
        )
        if any(f"_{treatment}_" in run_name for run_name in run_artifacts)
    }
    checks = {name: {"passed": True} for name in REQUIRED_CHECKS}
    checks["prefix_counts_and_latest_slices"].update(
        {
            "source_history_matches": {name: True for name in treatments},
            "expanded_examples": {name: 1 for name in treatments},
            "cache_names": {name: name for name in treatments},
            "cache_files_sha256": {name: {"file": "hash"} for name in treatments},
            "cap1_is_latest_cap8_slice": True,
            "cap8_is_latest_cap16_slice": True,
            "source_file_count": 1,
            "source_files_manifest_sha256": "source",
        }
    )
    checks["target_exclusion_no_leakage"].update(
        {
            "target_only_query_max_delta": 0.0,
            "cached_target_position": "final event of each cached sequence",
        }
    )
    checks["encoder_attention_mask"].update(
        {
            "encoder_is_bidirectional": True,
            "future_history_changes_earlier_state": True,
            "other_user_history_max_delta": 0.0,
            "production_run_name": "run",
            "production_experiment_class": "MuTransferCrossAttentionGenerationExperiment",
            "production_query_architecture": "encoder_decoder",
            "production_window": "bounded_prefix",
            "production_prefix_length_rule": "truncated",
            "production_prefix_cap": 16,
            "production_targets_class": "NextItemTargets",
            "production_criterion_class": "TwoTowerLoss",
            "production_optimizer_class": "Adam",
        }
    )
    checks["gradient_flow"].update(
        {
            "initial_readout_l1": 0,
            "bootstrap_readout_gradient_l1": 1.0,
            "bootstrap_memory_encoder_gradient_l1": 0,
            "bootstrap_cross_attention_gradient_l1": 0,
            "bootstrapped_readout_l1": 1.0,
            "memory_encoder_gradient_l1": 1.0,
            "cross_attention_gradient_l1": 1.0,
        }
    )
    checks["candidate_only_loss"].update(
        {
            "all_candidate_targets_equal_examples": True,
            "all_ntp_targets_zero": True,
            "micro_candidate_targets_per_example": 1,
            "duplicated_batch_loss_delta": 0.0,
            "reduction": "proxy",
        }
    )
    curves = {
        run_name: [
            {
                "epoch": epoch,
                "recall@100": (0.118, 0.12, 0.119)[int(run_name[-1])]
                - (20 - epoch) / 1000,
                "ndcg@100": 0.05,
            }
            for epoch in range(1, 21)
        ]
        for run_name in run_artifacts
    }
    checks["learning_curves"].update(
        {
            "runs": curves,
            "selected_threshold_crossings": {
                name: {
                    "threshold_recall@100": 0.12,
                    "first_matching_epoch": 20,
                    "recall@100": 0.12,
                    "selected_epoch": 20,
                    "selected_recall@100": 0.12,
                }
                for name in (
                    "truncated_8_vs_one_example",
                    "truncated_16_vs_truncated_8",
                )
            },
        }
    )
    checks["lr_boundary"].update(
        {
            "required_followups": [],
            "surfaces": {
                treatment: [
                    {
                        "run_name": f"g1_rq13_{treatment}_d{index}",
                        "deep_learning_rate": rate,
                        "best_epoch": 20,
                        "validation_recall@100": (0.118, 0.12, 0.119)[index],
                    }
                    for index, rate in enumerate((0.006, 0.012, 0.024))
                ]
                for treatment in treatments
            },
        }
    )
    return {
        "schema_version": 1,
        "research_question": "RQ13 encoder-decoder prefix expansion",
        "dataset_size": "500m",
        "status": "passed",
        "checks": checks,
        "run_artifacts": run_artifacts,
        "implementation_sha256": current_implementation_sha256(),
    }


def _artifacts() -> dict[str, dict[str, str]]:
    return {
        f"g1_rq13_{treatment}_d{index}": {
            "training_metadata.json": "a",
            "final_metrics.json": "b",
            "sweep.log": "c",
        }
        for treatment in (
            "one_example",
            "truncated_8",
            "truncated_16",
            "required_8",
            "required_16",
        )
        for index in range(3)
    }


def test_audit_validation_binds_every_check_to_exact_run_artifacts() -> None:
    artifacts = _artifacts()

    validated = validate_correctness_audit(_audit(artifacts), artifacts)

    assert validated["status"] == "passed"
    assert validated["schema_version"] == 1


@pytest.mark.parametrize(
    "failure",
    ["missing_check", "missing_detail", "failed_check", "stale_run", "stale_code"],
)
def test_audit_validation_fails_closed(
    failure: str,
) -> None:
    artifacts = _artifacts()
    document = _audit(artifacts)
    if failure == "missing_check":
        document["checks"].pop(REQUIRED_CHECKS[0])
    elif failure == "missing_detail":
        document["checks"]["gradient_flow"].pop("memory_encoder_gradient_l1")
    elif failure == "failed_check":
        document["checks"][REQUIRED_CHECKS[0]]["passed"] = False
    elif failure == "stale_run":
        document["run_artifacts"] = {"run-a": {"sweep.log": "stale"}}
    else:
        document["implementation_sha256"] = {"stale.py": "stale"}

    with pytest.raises(Rq13AuditError):
        validate_correctness_audit(document, artifacts)


def _stage_one_results(
    artifacts: dict[str, dict[str, str]], status: str
) -> dict[str, object]:
    selected_cap = 32
    proposed = [
        candidate.run_name for candidate in make_selected_cap_candidates(selected_cap)
    ]
    treatments = {}
    for treatment in (
        "one_example",
        "truncated_4",
        "truncated_8",
        "truncated_16",
        "required_8",
        "required_16",
    ):
        treatments[treatment] = {
            "artifacts": [
                {"run_name": run_name, "artifact_sha256": hashes}
                for run_name, hashes in artifacts.items()
                if f"_{treatment}_" in run_name
            ]
        }
    cap_fit: dict[str, object] = {
        "status": status,
        "selected_cap": selected_cap,
        "practical_ceiling": {"selected": selected_cap},
    }
    required = proposed if status == "selected_cap_pending" else []
    missing = proposed if status == "selected_cap_pending" else []
    if status == "stage_one_audit_required":
        cap_fit["proposed_followups"] = proposed
    return {
        "research_question": "RQ13 encoder-decoder prefix expansion",
        "dataset_size": "500m",
        "missing_initial_artifacts": missing,
        "required_boundary_followups": [],
        "required_followups": required,
        "cap_fit": cap_fit,
        "treatments": treatments,
        "surface_winners": {name: {} for name in treatments},
    }


def _stage_one_artifacts() -> dict[str, dict[str, str]]:
    return {
        f"g1_rq13_{treatment}_d{index}": {
            "training_metadata.json": "a",
            "final_metrics.json": "b",
            "sweep.log": "c",
        }
        for treatment in (
            "one_example",
            "truncated_4",
            "truncated_8",
            "truncated_16",
            "required_8",
            "required_16",
        )
        for index in range(3)
    }


def test_audit_generator_accepts_exact_post_cap4_stage() -> None:
    results = _stage_one_results(_stage_one_artifacts(), "stage_one_audit_required")

    assert validate_audit_generation_stage(results) == "stage_one"

    results["cap_fit"]["proposed_followups"] = []
    with pytest.raises(Rq13AuditError, match="selected-cap followups"):
        validate_audit_generation_stage(results)

    results = _stage_one_results(_stage_one_artifacts(), "stage_one_audit_required")
    results["cap_fit"]["practical_ceiling"] = {"selected": 31}
    with pytest.raises(Rq13AuditError, match="practical ceiling"):
        validate_audit_generation_stage(results)


def test_selected_cap_stage_binds_fresh_stage_one_audit() -> None:
    artifacts = _stage_one_artifacts()
    audit = _audit(artifacts)
    results = _stage_one_results(artifacts, "selected_cap_pending")
    binding = validate_correctness_audit(audit, artifacts)
    results["cap_fit"]["input_bindings"] = {"stage_one_correctness_audit": binding}

    assert validate_bound_stage_one_audit(results, audit) == binding

    invalid_missing = _stage_one_results(artifacts, "selected_cap_pending")
    invalid_missing["missing_initial_artifacts"] = []
    with pytest.raises(Rq13AuditError, match="selected-cap followups"):
        validate_audit_generation_stage(invalid_missing)

    results["cap_fit"]["input_bindings"]["stage_one_correctness_audit"] = {
        **binding,
        "artifact_sha256": "0" * 64,
    }
    with pytest.raises(Rq13AuditError, match="binding"):
        validate_bound_stage_one_audit(results, audit)
