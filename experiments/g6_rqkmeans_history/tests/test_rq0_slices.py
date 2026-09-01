from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from dcn.eval.ranking_evidence import (
    RankingEvidence,
    load_ranking_evidence,
    write_ranking_evidence,
)
from dcn.semantic import SemanticCodes
from experiments.g6_rqkmeans_history.analysis.rq0_slices import (
    bucket_size_comparison,
    slice_comparison,
)


def _evidence(ranks: list[int]) -> RankingEvidence:
    return RankingEvidence(
        user_ids=torch.tensor([10, 20, 30]),
        history_item_ids=torch.tensor([1, 2, 3, 4, 5]),
        history_offsets=torch.tensor([0, 2, 4, 5]),
        relevant_item_ids=torch.tensor([6, 7, 8, 9]),
        relevance_offsets=torch.tensor([0, 2, 3, 4]),
        relevant_train_frequencies=torch.tensor([0, 2, 20, 200]),
        relevant_ranks=torch.tensor(ranks),
        max_k=100,
    )


def test_ranking_evidence_shares_context_and_is_immutable(tmp_path: Path) -> None:
    context_path = tmp_path / "context.pt"
    control_path = tmp_path / "control.pt"
    semantic_path = tmp_path / "semantic.pt"
    control = _evidence([1, 0, 5, 0])
    semantic = _evidence([0, 2, 1, 10])

    control_digest = write_ranking_evidence(
        control, context_path=context_path, ranking_path=control_path
    )
    semantic_digest = write_ranking_evidence(
        semantic, context_path=context_path, ranking_path=semantic_path
    )

    assert control_digest == semantic_digest
    assert load_ranking_evidence(context_path, control_path) == control
    assert load_ranking_evidence(context_path, semantic_path) == semantic

    changed_context = _evidence([1, 0, 5, 0])
    changed_context.history_item_ids[0] = 99
    with pytest.raises(RuntimeError, match="context changed"):
        write_ranking_evidence(
            changed_context,
            context_path=context_path,
            ranking_path=tmp_path / "changed.pt",
        )
    with pytest.raises(RuntimeError, match="ranking changed"):
        write_ranking_evidence(
            semantic,
            context_path=context_path,
            ranking_path=control_path,
        )


def test_ranking_evidence_rejects_negative_train_frequency() -> None:
    with pytest.raises(ValueError, match="frequencies must be nonnegative"):
        replace(
            _evidence([1, 0, 5, 0]),
            relevant_train_frequencies=torch.tensor([-1, 2, 20, 200]),
        )


def test_slice_comparison_uses_target_frequency_and_collided_history() -> None:
    control = _evidence([1, 0, 5, 0])
    semantic = _evidence([0, 2, 1, 10])
    codes = SemanticCodes(
        item_ids=torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9]),
        codes=torch.tensor(
            [
                [0, 0],
                [0, 1],
                [2, 0],
                [1, 0],
                [1, 1],
                [0, 0],
                [2, 1],
                [3, 0],
                [3, 1],
            ]
        ),
        codes_per_level=(4, 2),
    )

    result = slice_comparison(
        control,
        semantic,
        semantic_codes=codes,
        semantic_base_levels=2,
        control_run_name="control",
        semantic_run_name="semantic",
    )

    assert result["frequency_terciles"]["boundaries"] == [2, 20]
    assert result["slices"]["frequency_low"]["num_targets"] == 2
    assert result["slices"]["frequency_low"]["control"]["recall@100"] == 0.5
    assert result["slices"]["frequency_low"]["semantic"]["recall@100"] == 0.5
    assert result["slices"]["history_has_collided_base_sid"]["num_users"] == 1
    assert (
        result["slices"]["history_has_collided_base_sid"]["control"]["recall@100"]
        == 0.5
    )
    assert (
        result["slices"]["history_has_no_collided_base_sid"]["semantic"]["recall@100"]
        == 1.0
    )


def test_bucket_size_comparison_slices_relevant_targets() -> None:
    control = _evidence([1, 0, 5, 0])
    semantic = _evidence([0, 2, 1, 10])
    codes = SemanticCodes(
        item_ids=torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9]),
        codes=torch.tensor(
            [
                [0, 0],
                [0, 1],
                [2, 0],
                [1, 0],
                [1, 1],
                [0, 0],
                [2, 1],
                [3, 0],
                [3, 1],
            ]
        ),
        codes_per_level=(4, 2),
    )

    result = bucket_size_comparison(
        control,
        semantic,
        semantic_codes=codes,
        semantic_base_levels=2,
    )

    assert result["slices"]["bucket_size_1"]["num_targets"] == 3
    assert result["slices"]["bucket_size_2"]["num_targets"] == 1
    assert result["slices"]["bucket_size_2"]["control"]["recall@100"] == 1.0
    assert result["slices"]["bucket_size_2"]["semantic"]["recall@100"] == 0.0
