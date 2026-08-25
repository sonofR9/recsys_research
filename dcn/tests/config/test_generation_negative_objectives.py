from __future__ import annotations

from pathlib import Path

import pytest
import torch

from dcn.config import GenerationExperiment
from dcn.tests.miniature_yambda import configure


def _loss(base_path: Path, **overrides):
    num_negatives = overrides.pop("num_in_batch_negatives", 4)
    experiment = configure(
        GenerationExperiment,
        base_path,
        num_in_batch_negatives=num_negatives,
        **overrides,
    )
    experiment.setup()
    return experiment.create_criterion().loss


@pytest.mark.parametrize("correction", ["yi2019", "baseline"])
def test_offline_logq_exposes_the_selected_correction(
    base_path: Path, correction: str
) -> None:
    loss = _loss(
        base_path,
        negative_sampling="offline_logq",
        logq_correction=correction,
    )

    assert loss.correction == correction
    assert loss.q.sum().item() == pytest.approx(1.0)


def test_legacy_mixed_offline_logq_only_corrects_the_in_batch_source(
    base_path: Path,
) -> None:
    offline = _loss(base_path, negative_sampling="offline_logq")
    loss = _loss(
        base_path,
        negative_sampling="mixed_offline_logq",
        random_negative_fraction=0.5,
    )

    uniform = torch.zeros_like(offline.q)
    uniform[1:] = 1 / (uniform.numel() - 1)
    assert torch.allclose(loss.q, 0.5 * offline.q + 0.5 * uniform)
    assert loss.num_in_batch_negatives == 2
    assert loss.random_negatives.num_negatives == 2
    assert loss.random_negatives.probabilities is None
    assert not loss.correct_random_negatives


def test_aggregate_streaming_global_q_uses_realized_source_fraction(
    base_path: Path,
) -> None:
    loss = _loss(
        base_path,
        negative_sampling="mixed_online_global_q",
        num_in_batch_negatives=5,
        random_negative_fraction=0.5,
        correct_positive_logq=True,
        mask_false_negatives=False,
        exclude_own_group_negatives=False,
    )

    assert loss.num_in_batch_negatives == 3
    assert loss.random_negatives.num_negatives == 2
    assert loss.uniform_mixture_fraction == pytest.approx(2 / 5)
    assert loss.correct_positive
    assert loss.correct_random_negatives
    assert not loss.mask_false_negatives
    assert not loss.exclude_own_group


def test_negative_only_aggregate_streaming_is_a_distinct_diagnostic(
    base_path: Path,
) -> None:
    loss = _loss(
        base_path,
        negative_sampling="mixed_online_global_q_negative_only",
        mask_false_negatives=False,
        exclude_own_group_negatives=False,
    )

    assert not loss.correct_positive
    assert loss.correct_random_negatives


def test_random_offline_logq_uses_expected_negative_multiplicity(
    base_path: Path,
) -> None:
    loss = _loss(base_path, negative_sampling="random_offline_logq")

    assert loss.num_in_batch_negatives == 0
    assert loss.random_negatives.num_negatives == 4
    assert loss.random_negatives.probabilities.sum().item() == pytest.approx(1.0)
    assert loss.q.sum().item() == pytest.approx(4.0)
    assert loss.correct_random_negatives


@pytest.mark.parametrize("correction", ["yi2019", "baseline"])
def test_fixed_logq_logits_match_the_selected_reference_objective(
    base_path: Path, correction: str
) -> None:
    loss = _loss(
        base_path,
        negative_sampling="offline_logq",
        logq_correction=correction,
        mask_false_negatives=False,
    )
    query = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    positives = torch.tensor([[0.5, 1.0], [1.5, 2.0]])
    positive_ids = torch.tensor([1, 2])
    negative_representations = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0]], [[0.5, 0.5], [1.0, 1.0]]]
    )
    negative_ids = torch.tensor([[2, 3], [1, 3]])

    logits = loss.logits(
        query,
        positives,
        positive_ids,
        torch.tensor([1, 1]),
        negatives=(negative_representations, negative_ids),
    )

    positive_scores = (query * positives).sum(-1)
    negative_scores = torch.einsum(
        "qd,qnd->qn", query, negative_representations
    )
    effective_q = loss.q.clamp_min(loss.eps)
    if correction == "yi2019":
        expected_positive = positive_scores - effective_q[positive_ids].log()
        expected_negative = negative_scores - effective_q[negative_ids].log()
    else:
        expected_positive = positive_scores
        leave_one_out = effective_q[negative_ids] / (
            loss.q.sum() - effective_q[positive_ids]
        ).unsqueeze(1)
        expected_negative = negative_scores - leave_one_out.log()
    expected = torch.cat(
        [expected_positive.unsqueeze(1), expected_negative], dim=1
    )

    assert torch.allclose(logits, expected)


def test_global_q_sampling_path_matches_unconditional_draw_reference(
    base_path: Path,
) -> None:
    loss = _loss(
        base_path,
        negative_sampling="offline_logq",
        logq_correction="yi2019",
        correct_positive_logq=True,
        mask_false_negatives=False,
        exclude_own_group_negatives=False,
    ).eval()
    query = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, -1.0]]
    )
    positives = torch.tensor(
        [[0.5, 1.0], [1.5, 0.5], [0.5, 1.0], [2.0, 1.0]]
    )
    positive_ids = torch.tensor([1, 2, 1, 3])
    group_sizes = torch.tensor([2, 2])

    torch.manual_seed(4)
    negative_indices = torch.randint(4, (4, 4))
    assert any(
        index in own_group
        for row, own_group in enumerate(({0, 1}, {0, 1}, {2, 3}, {2, 3}))
        for index in negative_indices[row].tolist()
    )
    assert torch.any(positive_ids[negative_indices] == positive_ids.unsqueeze(1))
    negative_ids = positive_ids[negative_indices]
    negative_scores = torch.einsum(
        "qd,qnd->qn", query, positives[negative_indices]
    )
    expected = torch.cat(
        [
            (
                (query * positives).sum(-1)
                - loss.q[positive_ids].clamp_min(loss.eps).log()
            ).unsqueeze(1),
            negative_scores - loss.q[negative_ids].clamp_min(loss.eps).log(),
        ],
        dim=1,
    )

    torch.manual_seed(4)
    actual = loss.logits(query, positives, positive_ids, group_sizes)

    assert torch.isfinite(actual).all()
    assert torch.allclose(actual, expected)
