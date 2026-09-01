from __future__ import annotations

from pathlib import Path

import pytest

from experiments.g3_pretrained_item_embeddings.analysis import (
    rq2_unexpected_diagnostic_results as results,
)
from experiments.g3_pretrained_item_embeddings.analysis.rq2_unexpected_diagnostic_results import (
    _boundary_decision,
    _comparison,
    _resources,
    _validate_common_schedule,
)


def test_comparison_uses_baseline_scaled_native50m_bands() -> None:
    treatment = {
        "row_id": "treatment",
        "metrics": {"recall@100": 0.11, "ndcg@100": 0.031},
    }
    baseline = {
        "row_id": "baseline",
        "metrics": {"recall@100": 0.10, "ndcg@100": 0.030},
    }

    comparison = _comparison(treatment, baseline)

    assert comparison["recall_at_100_delta"] == pytest.approx(0.01)
    assert comparison["recall_at_100_operational_band"] == pytest.approx(0.019414)
    assert comparison["recall_at_100_change_exceeds_operational_band"] is False
    assert comparison["ndcg_at_100_operational_band"] == pytest.approx(0.0064281)


def test_width128_selected_rates_require_lower_deep_lr_boundary_only() -> None:
    selected = {
        "embedding_learning_rate": 0.3041556165944196,
        "deep_learning_rate": 0.014506684820055783,
        "best_epoch": 19,
        "horizon_epochs": 40,
    }

    decision = _boundary_decision(
        selected,
        prior_rows=[
            {
                "family_id": "rq2_content_concat",
                "capacity": 128,
                "deep_learning_rate": rate,
            }
            for rate in (0.1258268, 0.0324339, 0.0399306)
        ],
    )

    assert decision["embedding_boundary_triggered"] is False
    assert decision["deep_lower_boundary_triggered"] is True
    assert decision["selected_is_smallest_tested_width_128_deep_lr"] is True
    assert decision["capacity_boundary_triggered"] is False
    assert decision["horizon_extension_required"] is False
    assert decision["required_actions"] == [
        "three_width128_horizon40_lower_deep_learning_rate_probes"
    ]


def test_common_schedule_accepts_only_roundoff_level_difference() -> None:
    _validate_common_schedule(
        [
            {"normalized_lr_schedule": [1.0, 0.5, 0.0]},
            {"normalized_lr_schedule": [1.0, 0.5 + 5e-16, 0.0]},
        ]
    )
    with pytest.raises(ValueError, match="same normalized schedule"):
        _validate_common_schedule(
            [
                {"normalized_lr_schedule": [1.0, 0.5, 0.0]},
                {"normalized_lr_schedule": [1.0, 0.49, 0.0]},
            ]
        )


def test_efficiency_parser_requires_all_40_epochs(tmp_path: Path) -> None:
    log = tmp_path / "sweep.log"
    resources = " ".join(
        (
            "resources.params_total=100.0000",
            "resources.params_trainable=90.0000",
            "resources.params_embedding=60.0000",
            "resources.params_deep=40.0000",
            "resources.peak_memory_gb=1.0000",
        )
    )
    log.write_text("\n".join(f"{resources} timing.train_epoch_time=2.0000" for _ in range(40)))

    parsed, seconds = _resources(log)

    assert parsed == {
        "params_total": 100.0,
        "params_trainable": 90.0,
        "params_embedding": 60.0,
        "params_deep": 40.0,
        "peak_memory_gb": 1.0,
    }
    assert seconds == 80.0


def test_public_persistence_authenticates_before_exclusive_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    forged = results._document(
        {
            "schema_version": 1,
            "kind": "g3_rq2_unexpected_result_diagnostic_evidence",
            "protocol_sha256": results.APPROVED_PROTOCOL_SHA256,
            "diagnostic_ledger": {},
            "boundary_evidence": {},
            "rq1_evidence": {},
            "queue_batch": {},
            "ranking_context": {},
            "prior_tuning_ledger": [],
            "diagnostic_tuning_ledger": [],
            "all_tuning_and_diagnostic_ledger": [],
            "comparisons": {},
            "diagnostic_conclusion": {},
            "provisional_selection": {},
            "continuation": {},
            "opportunity_accounting": {},
        }
    )
    monkeypatch.setattr(
        results,
        "_authenticate_document",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("forged")),
    )

    with pytest.raises(ValueError, match="forged"):
        results.persist_rq2_unexpected_diagnostic_evidence(
            tmp_path / "evidence.json", forged, root=tmp_path
        )

    assert not (tmp_path / "evidence.json").exists()
