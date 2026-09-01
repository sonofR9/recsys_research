from __future__ import annotations

from pathlib import Path

import pytest

from experiments.g3_pretrained_item_embeddings.analysis import rq2_final_results as final


def _row(
    row_id: str,
    *,
    recall: float,
    ndcg: float = 0.03,
    deep_lr: float = 0.01,
    family: str = "rq2_content_concat",
    capacity: int = 128,
) -> dict[str, object]:
    return {
        "row_id": row_id,
        "run_name": row_id,
        "family_id": family,
        "capacity": capacity,
        "deep_learning_rate": deep_lr,
        "horizon_epochs": 40,
        "best_epoch": 19,
        "queue_wall_seconds": 10.0,
        "metrics": {"recall@100": recall, "ndcg@100": ndcg},
    }


def test_final_selection_uses_recall_then_ndcg() -> None:
    selected = final.select_final_content_candidate(
        [
            _row("lower", recall=0.09, ndcg=0.04),
            _row("winner", recall=0.10, ndcg=0.02),
            _row(
                "excluded",
                recall=0.20,
                family="rq2_id_only_densenet",
            ),
        ]
    )

    assert selected["row_id"] == "winner"


def test_exact_seven_reuse_rows_are_width128_content_concat() -> None:
    rows = [
        _row(row_id, recall=0.1 - index * 0.001)
        for index, row_id in enumerate(final._REUSABLE_ROW_IDS)
    ]
    rows.extend(
        [
            _row("width32", recall=0.2, capacity=32),
            _row("ablation", recall=0.2, family="rq2_content_zero_id"),
        ]
    )

    reusable = final.eligible_rq3_reuse_rows(rows)

    assert tuple(row["row_id"] for row in reusable) == final._REUSABLE_ROW_IDS


def test_reuse_surface_fails_closed_when_one_required_row_is_missing() -> None:
    rows = [_row(row_id, recall=0.1) for row_id in final._REUSABLE_ROW_IDS[:-1]]

    with pytest.raises(ValueError, match="exact seven"):
        final.eligible_rq3_reuse_rows(rows)


def test_resolved_boundary_requires_rates_on_both_sides() -> None:
    selected = _row("selected", recall=0.1, deep_lr=0.0145)
    rows = [
        _row("lower", recall=0.09, deep_lr=0.004),
        selected,
        _row("higher", recall=0.08, deep_lr=0.032),
    ]

    decision = final._resolved_boundary(selected, rows)

    assert decision["selected_is_interior"] is True
    assert decision["additional_runs_authorized"] is False

    with pytest.raises(ValueError, match="not interior"):
        final._resolved_boundary(selected, [selected, rows[2]])


def test_boundary_schedule_parity_allows_only_float_roundoff() -> None:
    final._validate_common_boundary_schedule(
        [
            {"normalized_lr_schedule": [1.0, 0.5, 0.0]},
            {"normalized_lr_schedule": [1.0, 0.5 + 5e-16, 0.0]},
        ]
    )

    with pytest.raises(ValueError, match="schedules differ"):
        final._validate_common_boundary_schedule(
            [
                {"normalized_lr_schedule": [1.0, 0.5, 0.0]},
                {"normalized_lr_schedule": [1.0, 0.4, 0.0]},
            ]
        )


def test_public_persistence_authenticates_before_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document = final._document(
        {
            "schema_version": 1,
            "kind": "g3_rq2_final_native50m_evidence",
            "protocol_sha256": final.APPROVED_PROTOCOL_SHA256,
            "diagnostic_evidence": {},
            "boundary_ledger": {},
            "queue_batch": {},
            "ranking_context": {},
            "boundary_tuning_ledger": [],
            "all_tuning_diagnostic_boundary_ledger": [],
            "final_content_selection": {},
            "final_rq2_comparison": {},
            "rq3_inputs": {},
            "opportunity_accounting": {},
        }
    )
    monkeypatch.setattr(
        final,
        "_authenticate_document",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("forged")),
    )

    with pytest.raises(ValueError, match="forged"):
        final.persist_rq2_final_evidence(
            tmp_path / "rq2.json", document, root=tmp_path
        )

    assert not (tmp_path / "rq2.json").exists()
