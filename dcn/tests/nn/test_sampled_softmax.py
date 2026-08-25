import polars as pl
import pytest
import torch
from torch.nn import functional as F

from dcn.nn.sampled_softmax import (
    Correction,
    OfflineInBatchSoftmax,
    RandomCatalogNegatives,
    StreamingInBatchSoftmax,
)


def _make_loss(
    hash_size: int = 64,
    k: int = 4,
    alpha: float = 0.05,
    correction: Correction = "yi2019",
) -> StreamingInBatchSoftmax:
    return StreamingInBatchSoftmax(
        hash_size=hash_size,
        num_in_batch_negatives=k,
        alpha=alpha,
        correction=correction,
    )


def _make_batch(dim: int = 4):
    group_sizes = torch.tensor([3, 2, 4])  # N = 9, three groups
    total = int(group_sizes.sum())
    positive_item_ids = torch.arange(total)
    query_repr = torch.randn(total, dim, requires_grad=True)
    positive_item_repr = torch.randn(total, dim, requires_grad=True)
    return query_repr, positive_item_repr, positive_item_ids, group_sizes


@pytest.mark.parametrize("correction", ["yi2019", "baseline"])
def test_a_negative_that_is_the_label_is_not_scored_against_it(
    correction: Correction,
) -> None:
    loss = _make_loss(k=2, correction=correction)
    query_repr = torch.randn(2, 4)
    positive_item_repr = torch.randn(2, 4)
    positive_item_ids = torch.tensor([7, 7])
    # Every negative is drawn from the other example, which liked the same item.
    negatives = (
        positive_item_repr.flip(0).unsqueeze(1).expand(-1, 2, -1),
        positive_item_ids.unsqueeze(1).expand(-1, 2),
    )

    logits = loss.logits(
        query_repr,
        positive_item_repr,
        positive_item_ids,
        torch.tensor([1, 1]),
        negatives,
    )

    assert torch.isfinite(logits[:, 0]).all()
    assert (logits[:, 1:] == -torch.inf).all()


def test_forward_finite_and_grads_flow() -> None:
    torch.manual_seed(0)
    loss = _make_loss()
    query_repr, positive_item_repr, positive_item_ids, group_sizes = _make_batch()

    out = loss(query_repr, positive_item_repr, positive_item_ids, group_sizes)
    assert out.ndim == 0 and torch.isfinite(out)

    out.backward()
    assert query_repr.grad is not None and positive_item_repr.grad is not None
    assert query_repr.grad.abs().sum() > 0
    assert positive_item_repr.grad.abs().sum() > 0


@pytest.mark.parametrize("correction", ["none", "yi2019", "baseline"])
def test_chunked_in_batch_loss_matches_unchunked_loss_and_gradients(
    correction: Correction,
) -> None:
    torch.manual_seed(47)
    reference = StreamingInBatchSoftmax(
        hash_size=64,
        num_in_batch_negatives=7,
        correction=correction,
        score_chunk_size=None,
    ).eval()
    chunked = StreamingInBatchSoftmax(
        hash_size=64,
        num_in_batch_negatives=7,
        correction=correction,
        score_chunk_size=3,
    ).eval()
    gaps = torch.linspace(0.5, 2.0, 64)
    for criterion in (reference, chunked):
        criterion.smoothed_gap.copy_(gaps)
        criterion.last_seen_step.fill_(1)
    reference_query, reference_positive, ids, group_sizes = _make_batch()
    chunked_query = reference_query.detach().clone().requires_grad_()
    chunked_positive = reference_positive.detach().clone().requires_grad_()

    torch.manual_seed(53)
    reference_value = reference(
        reference_query, reference_positive, ids, group_sizes
    )
    torch.manual_seed(53)
    chunked_value = chunked(chunked_query, chunked_positive, ids, group_sizes)

    assert torch.allclose(chunked_value, reference_value, atol=1e-6)

    reference_value.backward()
    chunked_value.backward()
    assert torch.allclose(chunked_query.grad, reference_query.grad, atol=1e-6)
    assert torch.allclose(chunked_positive.grad, reference_positive.grad, atol=1e-6)


def test_chunked_in_batch_scoring_does_not_save_full_negative_representations() -> None:
    num_examples, num_negatives, dim, chunk_size = 9, 11, 5, 3
    loss = StreamingInBatchSoftmax(
        hash_size=64,
        num_in_batch_negatives=num_negatives,
        score_chunk_size=chunk_size,
    ).eval()
    loss.smoothed_gap.fill_(1.0)
    loss.last_seen_step.fill_(1)
    query_repr = torch.randn(num_examples, dim, requires_grad=True)
    positive_item_repr = torch.randn(num_examples, dim, requires_grad=True)
    positive_item_ids = torch.arange(num_examples)
    saved_shapes: list[torch.Size] = []

    def save(tensor: torch.Tensor) -> torch.Tensor:
        saved_shapes.append(tensor.shape)
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(save, lambda tensor: tensor):
        value = loss(
            query_repr,
            positive_item_repr,
            positive_item_ids,
            torch.ones(num_examples, dtype=torch.long),
        )
    value.backward()

    assert torch.Size((num_examples, num_negatives, dim)) not in saved_shapes
    assert not any(
        len(shape) == 3 and shape[0] == num_examples and shape[1] > chunk_size
        for shape in saved_shapes
    )


def test_negatives_avoid_own_group() -> None:
    torch.manual_seed(1)
    loss = _make_loss(k=8)
    group_sizes = torch.tensor([3, 2, 4])
    total = int(group_sizes.sum())
    ends = torch.cumsum(group_sizes, dim=0)
    starts = ends - group_sizes
    group_of_example = torch.repeat_interleave(
        torch.arange(group_sizes.size(0)), group_sizes
    )

    for _ in range(200):
        neg_idx = loss._sample_negatives(group_sizes, total)
        assert neg_idx.shape == (total, 8)
        for example in range(total):
            group = int(group_of_example[example])
            own_block = set(range(int(starts[group]), int(ends[group])))
            assert all(n not in own_block for n in neg_idx[example].tolist())


def test_streaming_prefers_frequent_item() -> None:
    loss = _make_loss(hash_size=64, k=2, alpha=0.05)
    common, rare = 1, 2

    for step in range(2000):
        loss._observe(torch.tensor([common]))
        if step % 50 == 0:
            loss._observe(torch.tensor([rare]))

    # smaller smoothed gap B => larger p̂ = 1/B for the frequently-seen item
    assert loss.smoothed_gap[common] < loss.smoothed_gap[rare]
    assert loss._log_q(torch.tensor([common])) > loss._log_q(torch.tensor([rare]))


def test_streaming_estimator_counts_target_occurrences() -> None:
    loss = _make_loss(hash_size=8, alpha=0.25)

    loss._observe(torch.tensor([1, 2, 1, 1]))

    assert loss.step.item() == 4
    assert loss.last_seen_step[1].item() == 4
    assert loss.last_seen_step[2].item() == 2
    assert loss.smoothed_gap[1].item() == pytest.approx(1.1875)
    assert loss.smoothed_gap[2].item() == pytest.approx(2.0)


def test_batched_streaming_update_matches_the_same_event_stream() -> None:
    ids = torch.tensor([1, 2, 1, 3, 2, 1, 3, 3])
    batched = _make_loss(hash_size=8, alpha=0.2)
    sequential = _make_loss(hash_size=8, alpha=0.2)

    batched._observe(ids)
    for item_id in ids:
        sequential._observe(item_id.unsqueeze(0))

    assert torch.equal(batched.step, sequential.step)
    assert torch.equal(batched.last_seen_step, sequential.last_seen_step)
    assert torch.allclose(batched.smoothed_gap, sequential.smoothed_gap)


def test_an_item_the_run_never_sampled_is_rare_not_ubiquitous() -> None:
    loss = _make_loss(hash_size=64, k=2, alpha=0.05)
    seen, never_seen = 1, 2

    for _ in range(100):
        loss._observe(torch.tensor([seen]))

    # Never sampled in 100 steps is evidence of a *small* sampling probability;
    # left uninitialized it would read as the most frequent item there is.
    assert loss._log_q(torch.tensor([never_seen])) < loss._log_q(torch.tensor([seen]))
    assert float(loss._log_q(torch.tensor([never_seen]))) == pytest.approx(
        -torch.log(torch.tensor(100.0)).item(), abs=1e-5
    )


def test_the_first_sighting_of_an_item_seeds_its_estimate() -> None:
    loss = _make_loss(hash_size=64, k=2, alpha=0.05)
    other, late = 1, 2

    for _ in range(29):
        loss._observe(torch.tensor([other]))
    loss._observe(torch.tensor([late]))

    # Seen once, on step 30: one sighting in 30 steps, not one in 30 * alpha.
    assert float(loss.smoothed_gap[late]) == pytest.approx(30.0)


def test_before_any_step_no_item_is_corrected() -> None:
    loss = _make_loss()

    assert float(loss._log_q(torch.tensor([3]))) == pytest.approx(0.0)


def test_only_the_colliding_negatives_are_dropped() -> None:
    loss = _make_loss(k=3).eval()
    loss.smoothed_gap.fill_(1.0)  # -log(1) == 0 -> uncorrected logits
    query_repr = torch.randn(2, 4)
    positive_item_repr = torch.randn(2, 4)
    positive_item_ids = torch.tensor([5, 6])
    group_sizes = torch.tensor([1, 1])

    neg_repr = torch.randn(2, 3, 4)
    # row 0 col 0 collides with positive 5; row 1 col 2 collides with positive 6
    neg_ids = torch.tensor([[5, 1, 2], [3, 4, 6]])

    logits = loss.logits(
        query_repr,
        positive_item_repr,
        positive_item_ids,
        group_sizes,
        negatives=(neg_repr, neg_ids),
    )

    assert logits.shape == (2, 4)
    assert logits[0, 1] == -torch.inf and logits[1, 3] == -torch.inf
    assert torch.allclose(logits[0, 2], (query_repr[0] * neg_repr[0, 1]).sum())
    assert torch.allclose(logits[1, 1], (query_repr[1] * neg_repr[1, 0]).sum())


class _IdOnlyItemEncoder(torch.nn.Module):
    def __init__(self, catalog_size: int, dim: int) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(catalog_size, dim)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(ids)


def test_random_catalog_negatives_sample_only_known_item_ids() -> None:
    negatives = RandomCatalogNegatives(
        catalog_size=8,
        first_item_id=1,
        num_negatives=100,
        item_encoder=_IdOnlyItemEncoder(8, 4),
    )

    _, ids = negatives(num_examples=20, device=torch.device("cpu"))

    assert ids.min() >= 1
    assert ids.max() < 8


def test_dense_random_scores_match_direct_sampled_scores() -> None:
    catalog_size = 17
    embedding = torch.nn.Embedding(catalog_size, 5)
    query = torch.randn(7, 5, requires_grad=True)
    direct = RandomCatalogNegatives(
        catalog_size=catalog_size,
        num_negatives=6,
        item_encoder=embedding,
        first_item_id=1,
    )
    dense = RandomCatalogNegatives(
        catalog_size=catalog_size,
        num_negatives=6,
        item_encoder=embedding,
        first_item_id=1,
        dense_scores=True,
    )

    torch.manual_seed(4)
    direct_scores, direct_ids = direct.logits(query)
    torch.manual_seed(4)
    dense_scores, dense_ids = dense.logits(query)

    assert torch.equal(direct_ids, dense_ids)
    assert torch.allclose(direct_scores, dense_scores, atol=1e-6)

    direct_scores.sum().backward()
    direct_query_grad = query.grad.detach().clone()
    direct_weight_grad = embedding.weight.grad.detach().clone()
    query.grad = None
    embedding.weight.grad = None
    dense_scores.sum().backward()

    assert torch.allclose(query.grad, direct_query_grad, atol=1e-6)
    assert torch.allclose(embedding.weight.grad, direct_weight_grad, atol=1e-6)


def test_chunked_random_scores_match_unchunked_scores_and_gradients() -> None:
    catalog_size, dim = 17, 5
    reference_embedding = torch.nn.Embedding(catalog_size, dim)
    chunked_embedding = torch.nn.Embedding(catalog_size, dim)
    chunked_embedding.load_state_dict(reference_embedding.state_dict())
    reference_query = torch.randn(7, dim, requires_grad=True)
    chunked_query = reference_query.detach().clone().requires_grad_()
    reference = RandomCatalogNegatives(
        catalog_size=catalog_size,
        num_negatives=7,
        item_encoder=reference_embedding,
        first_item_id=1,
        score_chunk_size=None,
    )
    chunked = RandomCatalogNegatives(
        catalog_size=catalog_size,
        num_negatives=7,
        item_encoder=chunked_embedding,
        first_item_id=1,
        score_chunk_size=3,
    )

    torch.manual_seed(43)
    reference_scores, reference_ids = reference.logits(reference_query)
    torch.manual_seed(43)
    chunked_scores, chunked_ids = chunked.logits(chunked_query)

    assert torch.equal(chunked_ids, reference_ids)
    assert torch.allclose(chunked_scores, reference_scores, atol=1e-6)

    reference_scores.square().sum().backward()
    chunked_scores.square().sum().backward()
    assert torch.allclose(chunked_query.grad, reference_query.grad, atol=1e-6)
    assert torch.allclose(
        chunked_embedding.weight.grad, reference_embedding.weight.grad, atol=1e-6
    )


@pytest.mark.parametrize(
    "embedding",
    [
        torch.nn.Embedding(17, 5, padding_idx=0),
        torch.nn.Embedding(17, 5, max_norm=1.0),
        torch.nn.Embedding(17, 5, scale_grad_by_freq=True),
        torch.nn.Embedding(17, 5, sparse=True),
    ],
)
def test_dense_random_scores_reject_nondefault_embedding_semantics(
    embedding: torch.nn.Embedding,
) -> None:
    with pytest.raises(ValueError, match="default embedding semantics"):
        RandomCatalogNegatives(
            catalog_size=17,
            num_negatives=6,
            item_encoder=embedding,
            dense_scores=True,
        )


def test_dense_random_scores_reject_embedding_subclasses() -> None:
    class CustomEmbedding(torch.nn.Embedding):
        pass

    with pytest.raises(TypeError, match="plain nn.Embedding"):
        RandomCatalogNegatives(
            catalog_size=17,
            num_negatives=6,
            item_encoder=CustomEmbedding(17, 5),
            dense_scores=True,
        )


def test_random_catalog_negatives_can_follow_an_exact_proposal() -> None:
    negatives = RandomCatalogNegatives(
        catalog_size=4,
        first_item_id=1,
        num_negatives=20,
        item_encoder=_IdOnlyItemEncoder(4, 4),
        probabilities=torch.tensor([0.0, 0.0, 1.0, 0.0]),
    )

    _, ids = negatives(num_examples=10, device=torch.device("cpu"))

    assert ids.unique().tolist() == [2]


def test_random_catalog_negatives_extend_the_in_batch_ones() -> None:
    torch.manual_seed(19)
    catalog_size, dim, in_batch, random_count = 32, 4, 2, 3
    loss = StreamingInBatchSoftmax(
        hash_size=64,
        num_in_batch_negatives=in_batch,
        random_negatives=RandomCatalogNegatives(
            catalog_size=catalog_size,
            num_negatives=random_count,
            item_encoder=_IdOnlyItemEncoder(catalog_size, dim),
        ),
    ).eval()
    loss.smoothed_gap.fill_(1.0)
    query_repr, positive_item_repr, positive_item_ids, group_sizes = _make_batch(dim)

    logits = loss.logits(query_repr, positive_item_repr, positive_item_ids, group_sizes)

    assert logits.shape == (query_repr.shape[0], 1 + in_batch + random_count)
    assert torch.isfinite(logits[:, 0]).all()


def test_random_negatives_alone_train_without_in_batch_ones() -> None:
    torch.manual_seed(23)
    catalog_size, dim = 32, 4
    encoder = _IdOnlyItemEncoder(catalog_size, dim)
    loss = StreamingInBatchSoftmax(
        hash_size=64,
        num_in_batch_negatives=0,
        random_negatives=RandomCatalogNegatives(
            catalog_size=catalog_size, num_negatives=5, item_encoder=encoder
        ),
    )
    query_repr, positive_item_repr, positive_item_ids, group_sizes = _make_batch(dim)

    out = loss(query_repr, positive_item_repr, positive_item_ids, group_sizes)
    out.backward()

    assert out.ndim == 0 and torch.isfinite(out)
    assert encoder.embedding.weight.grad.abs().sum() > 0


def test_predefined_negatives_are_used() -> None:
    loss = _make_loss(k=2)
    loss.smoothed_gap.fill_(1.0)
    query_repr = torch.randn(3, 4, requires_grad=True)
    positive_item_repr = torch.randn(3, 4, requires_grad=True)
    positive_item_ids = torch.tensor([0, 1, 2])
    group_sizes = torch.tensor([1, 1, 1])

    neg_repr = torch.randn(3, 2, 4, requires_grad=True)
    neg_ids = torch.tensor([[7, 8], [9, 10], [11, 12]])

    out = loss(
        query_repr,
        positive_item_repr,
        positive_item_ids,
        group_sizes,
        negatives=(neg_repr, neg_ids),
    )
    out.backward()

    assert out.ndim == 0 and torch.isfinite(out)
    assert neg_repr.grad is not None and neg_repr.grad.abs().sum() > 0


def test_yi2019_correction_shifts_positive_logit() -> None:
    torch.manual_seed(7)
    loss = _make_loss(hash_size=16, k=4).eval()
    query_repr, positive_item_repr, positive_item_ids, group_sizes = _make_batch()
    query_repr = query_repr.detach()
    positive_item_repr = positive_item_repr.detach()

    loss.smoothed_gap.fill_(1.0)  # -log(1) == 0 -> uncorrected logits
    loss.last_seen_step.fill_(1)  # every bucket has an estimate of its own
    torch.manual_seed(3)
    raw = loss.logits(query_repr, positive_item_repr, positive_item_ids, group_sizes)

    skew = torch.linspace(0.1, 0.9, loss.hash_size)
    loss.smoothed_gap.copy_(skew)
    torch.manual_seed(3)  # identical negative sampling
    corrected = loss.logits(
        query_repr, positive_item_repr, positive_item_ids, group_sizes
    )

    assert not torch.allclose(raw, corrected)
    expected_pos_shift = -torch.log(skew[positive_item_ids].clamp_min(loss.eps))
    assert torch.allclose(raw[:, 0] - corrected[:, 0], expected_pos_shift, atol=1e-5)


def test_offline_q_correction_matches_exact_softmax() -> None:
    torch.manual_seed(11)
    q = torch.tensor([0.5, 0.25, 0.125, 0.0625, 0.0625])
    loss = OfflineInBatchSoftmax(q=q, num_in_batch_negatives=2)
    query_repr = torch.randn(2, 4)
    positive_item_repr = torch.randn(2, 4)
    positive_item_ids = torch.tensor([0, 3])
    group_sizes = torch.tensor([1, 1])
    neg_repr = torch.randn(2, 2, 4)
    neg_ids = torch.tensor([[1, 2], [4, 1]])

    out = loss(
        query_repr,
        positive_item_repr,
        positive_item_ids,
        group_sizes,
        negatives=(neg_repr, neg_ids),
    )

    pos_score = (query_repr * positive_item_repr).sum(-1) - q[positive_item_ids].log()
    neg_score = (query_repr.unsqueeze(1) * neg_repr).sum(-1) - q[neg_ids].log()
    expected = F.cross_entropy(
        torch.cat([pos_score.unsqueeze(1), neg_score], dim=1),
        torch.zeros(2, dtype=torch.long),
    )
    assert torch.allclose(out, expected, atol=1e-6)


def test_homework_logq_corrects_only_negative_logits() -> None:
    q = torch.tensor([0.5, 0.25, 0.125, 0.0625, 0.0625])
    loss = OfflineInBatchSoftmax(
        q=q,
        num_in_batch_negatives=2,
        correct_positive=False,
        mask_false_negatives=False,
    )
    query_repr = torch.randn(2, 4)
    positive_item_repr = torch.randn(2, 4)
    positive_item_ids = torch.tensor([0, 3])
    neg_repr = torch.randn(2, 2, 4)
    neg_ids = torch.tensor([[0, 2], [4, 3]])

    logits = loss.logits(
        query_repr,
        positive_item_repr,
        positive_item_ids,
        torch.tensor([1, 1]),
        negatives=(neg_repr, neg_ids),
    )

    expected_positive = (query_repr * positive_item_repr).sum(-1)
    expected_negative = (
        torch.einsum("qd,qnd->qn", query_repr, neg_repr) - q[neg_ids].log()
    )
    assert torch.allclose(logits[:, 0], expected_positive)
    assert torch.allclose(logits[:, 1:], expected_negative)


def test_no_correction_leaves_all_logits_raw() -> None:
    q = torch.tensor([0.5, 0.25, 0.125, 0.0625, 0.0625])
    loss = OfflineInBatchSoftmax(
        q=q,
        num_in_batch_negatives=2,
        correction="none",
        mask_false_negatives=False,
    )
    query_repr = torch.randn(2, 4)
    positive_item_repr = torch.randn(2, 4)
    positive_item_ids = torch.tensor([0, 3])
    neg_repr = torch.randn(2, 2, 4)
    neg_ids = torch.tensor([[0, 2], [4, 3]])

    logits = loss.logits(
        query_repr,
        positive_item_repr,
        positive_item_ids,
        torch.tensor([1, 1]),
        negatives=(neg_repr, neg_ids),
    )

    expected = torch.cat(
        [
            (query_repr * positive_item_repr).sum(-1, keepdim=True),
            torch.einsum("qd,qnd->qn", query_repr, neg_repr),
        ],
        dim=1,
    )
    assert torch.allclose(logits, expected)


def test_mixed_negatives_share_one_proposal_correction() -> None:
    torch.manual_seed(41)
    q = torch.tensor([0.5, 0.25, 0.125, 0.0625, 0.0625])
    encoder = _IdOnlyItemEncoder(5, 4)
    loss = OfflineInBatchSoftmax(
        q=q,
        num_in_batch_negatives=2,
        random_negatives=RandomCatalogNegatives(
            catalog_size=5,
            first_item_id=1,
            num_negatives=1,
            item_encoder=encoder,
            probabilities=torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0]),
        ),
        correct_random_negatives=True,
        correct_positive=False,
        mask_false_negatives=False,
    )
    query_repr = torch.randn(2, 4)
    positive_item_repr = torch.randn(2, 4)
    positive_item_ids = torch.tensor([0, 3])
    neg_repr = torch.randn(2, 2, 4)
    neg_ids = torch.tensor([[1, 2], [1, 2]])

    logits = loss.logits(
        query_repr,
        positive_item_repr,
        positive_item_ids,
        torch.tensor([1, 1]),
        negatives=(neg_repr, neg_ids),
    )

    expected_in_batch = (
        torch.einsum("qd,qnd->qn", query_repr, neg_repr) - q[neg_ids].log()
    )
    expected_random = (
        query_repr @ encoder.embedding.weight[4].unsqueeze(1) - q[4].log()
    )
    assert torch.allclose(logits[:, 1:3], expected_in_batch)
    assert torch.allclose(logits[:, 3:], expected_random)


def test_streaming_mixture_uses_normalized_data_and_uniform_proposals() -> None:
    loss = StreamingInBatchSoftmax(
        hash_size=4,
        num_in_batch_negatives=2,
        uniform_mixture_fraction=0.25,
        first_item_id=1,
        normalize_streaming_over_valid_ids=True,
    )
    loss.step.fill_(8)
    loss.last_seen_step.fill_(1)
    loss.smoothed_gap.copy_(torch.tensor([8.0, 2.0, 4.0, 8.0]))

    actual = loss._log_q(torch.arange(4)).exp()

    data_q = torch.tensor([0.0, 1 / 2, 1 / 4, 1 / 8])
    data_q /= data_q.sum()
    uniform_q = torch.tensor([0.0, 1 / 3, 1 / 3, 1 / 3])
    assert torch.allclose(actual, 0.75 * data_q + 0.25 * uniform_q)


def test_aggregate_streaming_proposal_corrects_every_logit() -> None:
    encoder = _IdOnlyItemEncoder(5, 2)
    encoder.embedding.weight.data.zero_()
    loss = StreamingInBatchSoftmax(
        hash_size=5,
        num_in_batch_negatives=1,
        random_negatives=RandomCatalogNegatives(
            catalog_size=5,
            first_item_id=1,
            num_negatives=1,
            item_encoder=encoder,
            probabilities=torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0]),
        ),
        uniform_mixture_fraction=0.4,
        first_item_id=1,
        normalize_streaming_over_valid_ids=True,
        correct_positive=True,
        correct_random_negatives=True,
        mask_false_negatives=False,
        exclude_own_group=False,
    ).eval()
    loss.step.fill_(8)
    loss.last_seen_step.fill_(1)
    loss.smoothed_gap.copy_(torch.tensor([1.0, 2.0, 4.0, 8.0, 8.0]))
    query = torch.zeros(2, 2)
    positives = torch.zeros(2, 2)
    positive_ids = torch.tensor([1, 2])
    in_batch_ids = torch.tensor([[3], [1]])
    in_batch = (torch.zeros(2, 1, 2), in_batch_ids)

    logits = loss.logits(
        query,
        positives,
        positive_ids,
        torch.tensor([1, 1]),
        negatives=in_batch,
    )

    streaming = torch.tensor([1 / 2, 1 / 4, 1 / 8, 1 / 8])
    streaming /= streaming.sum()
    proposal = 0.6 * streaming + 0.4 * torch.full((4,), 1 / 4)
    expected = -torch.log(
        torch.tensor(
            [
                [proposal[0], proposal[2], proposal[3]],
                [proposal[1], proposal[0], proposal[3]],
            ]
        )
    )
    assert torch.allclose(logits, expected)


def test_homework_sampling_draws_from_every_target_position() -> None:
    loss = OfflineInBatchSoftmax(
        q=torch.full((16,), 1 / 16),
        num_in_batch_negatives=64,
        exclude_own_group=False,
    )

    torch.manual_seed(0)
    indices = loss._sample_negatives(torch.tensor([4]), total=4)

    assert set(indices.flatten().tolist()) == {0, 1, 2, 3}


def test_offline_q_stays_frozen_in_training() -> None:
    torch.manual_seed(13)
    loss = OfflineInBatchSoftmax(q=torch.full((16,), 1 / 16), num_in_batch_negatives=4)
    loss.train()
    query_repr, positive_item_repr, positive_item_ids, group_sizes = _make_batch()

    before = loss.q.clone()
    loss(query_repr, positive_item_repr, positive_item_ids, group_sizes)

    assert torch.equal(loss.q, before)


def test_baseline_matches_hand_computed_formula() -> None:
    torch.manual_seed(5)
    loss = _make_loss(hash_size=16, k=2, correction="baseline").eval()
    gaps = torch.linspace(0.5, 2.0, 16)
    loss.smoothed_gap.copy_(gaps)
    loss.last_seen_step.fill_(1)

    query_repr = torch.randn(3, 4)
    positive_item_repr = torch.randn(3, 4)
    positive_item_ids = torch.tensor([0, 1, 2])
    group_sizes = torch.tensor([1, 1, 1])
    neg_repr = torch.randn(3, 2, 4)
    neg_ids = torch.tensor([[7, 8], [9, 10], [11, 12]])

    out = loss(
        query_repr,
        positive_item_repr,
        positive_item_ids,
        group_sizes,
        negatives=(neg_repr, neg_ids),
    )

    q = 1.0 / gaps
    pos_score = (query_repr * positive_item_repr).sum(-1)
    neg_score = (query_repr.unsqueeze(1) * neg_repr).sum(-1)
    leave_one_out_q = q[neg_ids] / (q.sum() - q[positive_item_ids]).unsqueeze(1)
    corrected_neg = neg_score - leave_one_out_q.log()
    scores = torch.cat([pos_score.unsqueeze(1), corrected_neg], dim=1)
    positive_weight = 1.0 - torch.softmax(scores, dim=1)[:, 0]
    expected = -(
        positive_weight * (pos_score - torch.logsumexp(corrected_neg, dim=1))
    ).mean()

    assert torch.allclose(out, expected, atol=1e-6)


def test_baseline_grads_flow() -> None:
    torch.manual_seed(6)
    loss = _make_loss(hash_size=16, correction="baseline")
    loss.smoothed_gap.copy_(torch.linspace(0.5, 2.0, 16))
    loss.last_seen_step.fill_(1)
    query_repr, positive_item_repr, positive_item_ids, group_sizes = _make_batch()

    out = loss(query_repr, positive_item_repr, positive_item_ids, group_sizes)
    assert out.ndim == 0 and torch.isfinite(out)

    out.backward()
    assert query_repr.grad is not None and positive_item_repr.grad is not None
    assert query_repr.grad.abs().sum() > 0
    assert positive_item_repr.grad.abs().sum() > 0


@pytest.mark.parametrize("correction", ["yi2019", "baseline"])
def test_hit_rate_reads_the_same_logits_the_loss_does(correction: Correction) -> None:
    torch.manual_seed(29)
    loss = _make_loss(hash_size=16, k=3, correction=correction).eval()
    loss.smoothed_gap.copy_(torch.linspace(0.5, 2.0, 16))
    loss.last_seen_step.fill_(1)
    query_repr, positive_item_repr, positive_item_ids, group_sizes = _make_batch()

    torch.manual_seed(31)
    logits = loss.logits(query_repr, positive_item_repr, positive_item_ids, group_sizes)
    torch.manual_seed(31)  # identical negative sampling
    out = loss(query_repr, positive_item_repr, positive_item_ids, group_sizes)

    assert torch.allclose(loss.loss_from_logits(logits), out, atol=1e-6)


def test_unknown_correction_rejected() -> None:
    with pytest.raises(ValueError, match="correction"):
        StreamingInBatchSoftmax(
            hash_size=8,
            num_in_batch_negatives=2,
            correction="typo",  # type: ignore[arg-type]
        )


def test_from_train_interactions_builds_normalized_q() -> None:
    torch.manual_seed(17)
    train_data = pl.DataFrame({"item_id": [0, 0, 0, 1, 2, 2]})
    loss = OfflineInBatchSoftmax.from_train_interactions(
        train_data, catalog_size=4, num_in_batch_negatives=4, item_id_column="item_id"
    )

    assert torch.allclose(loss.q[:3], torch.tensor([3 / 6, 1 / 6, 2 / 6]))
    assert 0 < loss.q[3] < 1e-9
    assert torch.allclose(loss.q.sum(), torch.tensor(1.0))

    unseen_positive = torch.tensor([3, 0, 1])
    out = loss(
        torch.randn(3, 4),
        torch.randn(3, 4),
        unseen_positive,
        torch.tensor([1, 1, 1]),
    )
    assert torch.isfinite(out)
