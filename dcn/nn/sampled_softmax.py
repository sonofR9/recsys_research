from abc import ABC, abstractmethod
from collections.abc import Callable
import math
from typing import Literal, Self, get_args

import polars as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from dcn.data.packed import ragged_positions

Correction = Literal["none", "yi2019", "baseline"]
DEFAULT_NEGATIVE_SCORE_CHUNK_SIZE = 256


def _validated_chunk_size(chunk_size: int | None) -> int | None:
    if chunk_size is not None and chunk_size <= 0:
        raise ValueError("score_chunk_size must be positive or None")
    return chunk_size


def _checkpointed(
    function: Callable[..., torch.Tensor], *args: torch.Tensor
) -> torch.Tensor:
    if torch.is_grad_enabled() and any(argument.requires_grad for argument in args):
        return checkpoint(function, *args, use_reentrant=False)
    return function(*args)


class RandomCatalogNegatives(nn.Module):
    """Uniform negatives drawn from the whole catalog and encoded on the fly."""

    def __init__(
        self,
        catalog_size: int,
        num_negatives: int,
        item_encoder: nn.Module | Callable[[torch.Tensor], torch.Tensor],
        first_item_id: int = 0,
        probabilities: torch.Tensor | None = None,
        dense_scores: bool = False,
        score_chunk_size: int | None = DEFAULT_NEGATIVE_SCORE_CHUNK_SIZE,
    ) -> None:
        super().__init__()
        if not 0 <= first_item_id < catalog_size:
            raise ValueError(
                f"first_item_id must be in [0, {catalog_size}), got {first_item_id}"
            )
        if dense_scores and isinstance(item_encoder, nn.Embedding):
            if (
                item_encoder.padding_idx is not None
                or item_encoder.max_norm is not None
                or item_encoder.scale_grad_by_freq
                or item_encoder.sparse
            ):
                raise ValueError("dense scores require default embedding semantics")
        self.catalog_size = catalog_size
        self.num_negatives = num_negatives
        self.item_encoder = item_encoder
        self.first_item_id = first_item_id
        self.dense_scores = dense_scores
        self.score_chunk_size = _validated_chunk_size(score_chunk_size)
        if probabilities is not None:
            if probabilities.shape != (catalog_size,):
                raise ValueError("probabilities must have one value per catalog item")
            probabilities = probabilities.detach().float().clone()
            probabilities[:first_item_id] = 0
            if probabilities.sum() <= 0:
                raise ValueError("probabilities must assign mass to a known item")
            probabilities /= probabilities.sum()
        self.register_buffer("probabilities", probabilities)

    def _sample_ids(self, num_examples: int, device: torch.device) -> torch.Tensor:
        shape = (num_examples, self.num_negatives)
        return (
            torch.randint(
                self.first_item_id,
                self.catalog_size,
                shape,
                device=device,
            )
            if self.probabilities is None
            else torch.multinomial(
                self.probabilities,
                num_examples * self.num_negatives,
                replacement=True,
            ).view(shape)
        )

    def forward(
        self, num_examples: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ids = self._sample_ids(num_examples, device)
        return self.item_encoder(ids), ids

    def logits(self, query_repr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ids = self._sample_ids(query_repr.shape[0], query_repr.device)
        if self.training and self.dense_scores:
            catalog_ids = torch.arange(self.catalog_size, device=query_repr.device)
            catalog_repr = self.item_encoder(catalog_ids)
            scores = F.linear(query_repr, catalog_repr).gather(1, ids)
        else:
            scores = self._direct_scores(query_repr, ids)
        return scores, ids

    def _direct_scores(
        self, query_repr: torch.Tensor, ids: torch.Tensor
    ) -> torch.Tensor:
        chunk_size = self.score_chunk_size
        if chunk_size is None or ids.shape[1] <= chunk_size:
            return torch.einsum("qd,qnd->qn", query_repr, self.item_encoder(ids))
        return torch.cat(
            [
                _checkpointed(
                    lambda query, chunk_ids: torch.einsum(
                        "qd,qnd->qn", query, self.item_encoder(chunk_ids)
                    ),
                    query_repr,
                    ids[:, start : start + chunk_size],
                )
                for start in range(0, ids.shape[1], chunk_size)
            ],
            dim=1,
        )


class InBatchSampledSoftmaxLoss(nn.Module, ABC):
    def __init__(
        self,
        num_in_batch_negatives: int,
        correction: Correction = "yi2019",
        eps: float = 1e-12,
        random_negatives: RandomCatalogNegatives | None = None,
        correct_positive: bool = True,
        mask_false_negatives: bool = True,
        exclude_own_group: bool = True,
        correct_random_negatives: bool = True,
        score_chunk_size: int | None = DEFAULT_NEGATIVE_SCORE_CHUNK_SIZE,
    ) -> None:
        super().__init__()
        if correction not in get_args(Correction):
            raise ValueError(f"unknown correction {correction!r}")
        self.num_in_batch_negatives = num_in_batch_negatives
        self.correction = correction
        self.eps = eps
        self.random_negatives = random_negatives
        self.correct_random_negatives = correct_random_negatives
        self.correct_positive = correct_positive
        self.mask_false_negatives = mask_false_negatives
        self.exclude_own_group = exclude_own_group
        self.score_chunk_size = _validated_chunk_size(score_chunk_size)

    @abstractmethod
    def _log_q(self, ids: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def _q_total(self) -> torch.Tensor: ...

    def _q(self, ids: torch.Tensor) -> torch.Tensor:
        return self._log_q(ids).exp()

    def _observe(self, ids: torch.Tensor) -> None: ...

    def _sample_negatives(self, group_sizes: torch.Tensor, total: int) -> torch.Tensor:
        device = group_sizes.device
        if not self.exclude_own_group:
            return torch.randint(
                total,
                (total, self.num_in_batch_negatives),
                device=device,
            )
        group_of_example, _ = ragged_positions(group_sizes, total)
        group_ends = torch.cumsum(group_sizes, dim=0)
        example_ends = group_ends[group_of_example].unsqueeze(1)
        example_sizes = group_sizes[group_of_example].unsqueeze(1)

        max_shift = total - example_sizes
        shift = (
            torch.rand((total, self.num_in_batch_negatives), device=device) * max_shift
        ).long()
        return (example_ends + shift) % total

    def logits(
        self,
        query_repr: torch.Tensor,
        positive_item_repr: torch.Tensor,
        positive_item_ids: torch.Tensor,
        group_sizes: torch.Tensor,
        negatives: tuple[torch.Tensor, torch.Tensor] | None = None,
        acceptable_positive_ids: torch.Tensor | None = None,
        acceptable_positive_offsets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._validate_acceptable_positives(
            query_repr,
            acceptable_positive_ids,
            acceptable_positive_offsets,
        )
        if self.training:
            self._observe(positive_item_ids)

        if negatives is None:
            neg_idx = self._sample_negatives(group_sizes, query_repr.shape[0])
            neg_ids = positive_item_ids[neg_idx]
            in_batch_scores = self._in_batch_scores(
                query_repr, positive_item_repr, neg_idx
            )
        else:
            neg_repr, neg_ids = negatives
            in_batch_scores = torch.einsum("qd,qnd->qn", query_repr, neg_repr)

        pos_score = (query_repr * positive_item_repr).sum(-1)
        if self.correction == "yi2019":
            if self.correct_positive:
                pos_score = pos_score - self._log_q(positive_item_ids)
        elif self.correction == "baseline":
            leave_one_out_total = (
                self._q_total() - self._q(positive_item_ids)
            ).clamp_min(self.eps)

        def adjust_scores(
            scores: torch.Tensor,
            ids: torch.Tensor,
            correct: bool,
        ) -> torch.Tensor:
            if correct and self.correction == "yi2019":
                scores -= self._log_q(ids)
            elif correct and self.correction == "baseline":
                leave_one_out_q = (
                    self._q(ids) / leave_one_out_total.unsqueeze(1)
                ).clamp_min(self.eps)
                scores -= leave_one_out_q.log()
            if self.mask_false_negatives:
                scores = scores.masked_fill(
                    ids == positive_item_ids.unsqueeze(1), -torch.inf
                )
            if acceptable_positive_ids is not None:
                assert acceptable_positive_offsets is not None
                scores = scores.masked_fill(
                    self._acceptable_positive_mask(
                        ids,
                        acceptable_positive_ids,
                        acceptable_positive_offsets,
                    ),
                    -torch.inf,
                )
            return scores

        negative_scores = [adjust_scores(in_batch_scores, neg_ids, True)]
        if self.random_negatives is not None:
            random_scores, random_ids = self.random_negatives.logits(query_repr)
            negative_scores.append(
                adjust_scores(random_scores, random_ids, self.correct_random_negatives)
            )
        return torch.cat([pos_score.unsqueeze(1), *negative_scores], dim=1)

    @staticmethod
    def _validate_acceptable_positives(
        query_repr: torch.Tensor,
        ids: torch.Tensor | None,
        offsets: torch.Tensor | None,
    ) -> None:
        if (ids is None) != (offsets is None):
            raise ValueError("acceptable positive ids and offsets must be set together")
        if ids is None:
            return
        assert offsets is not None
        if ids.ndim != 1 or offsets.ndim != 1:
            raise ValueError(
                "acceptable positive ids and offsets must be one-dimensional"
            )
        if offsets.shape[0] != query_repr.shape[0] + 1:
            raise ValueError("acceptable positive offsets must delimit every query")
        if offsets.device != ids.device or ids.device != query_repr.device:
            raise ValueError("acceptable positives must share the query device")
        if int(offsets[0]) != 0 or int(offsets[-1]) != ids.shape[0]:
            raise ValueError("acceptable positive offsets must span every id")
        if bool((offsets[1:] < offsets[:-1]).any()):
            raise ValueError("acceptable positive offsets must be nondecreasing")

    @staticmethod
    def _acceptable_positive_mask(
        candidate_ids: torch.Tensor,
        acceptable_ids: torch.Tensor,
        offsets: torch.Tensor,
    ) -> torch.Tensor:
        if candidate_ids.numel() == 0 or acceptable_ids.numel() == 0:
            return torch.zeros_like(candidate_ids, dtype=torch.bool)
        if bool((candidate_ids < 0).any()) or bool((acceptable_ids < 0).any()):
            raise ValueError("acceptable-positive masking requires non-negative ids")
        counts = offsets.diff()
        acceptable_rows = torch.arange(
            candidate_ids.shape[0], device=candidate_ids.device
        ).repeat_interleave(counts)
        stride = torch.maximum(candidate_ids.max(), acceptable_ids.max()) + 1
        acceptable_keys = torch.sort(acceptable_rows * stride + acceptable_ids).values
        candidate_rows = torch.arange(
            candidate_ids.shape[0], device=candidate_ids.device
        ).unsqueeze(1)
        candidate_keys = candidate_rows * stride + candidate_ids
        positions = torch.searchsorted(acceptable_keys, candidate_keys.flatten())
        bounded = positions.clamp_max(acceptable_keys.shape[0] - 1)
        return (acceptable_keys[bounded] == candidate_keys.flatten()).view_as(
            candidate_ids
        )

    def _in_batch_scores(
        self,
        query_repr: torch.Tensor,
        positive_item_repr: torch.Tensor,
        negative_indices: torch.Tensor,
    ) -> torch.Tensor:
        chunk_size = self.score_chunk_size
        if chunk_size is None or negative_indices.shape[1] <= chunk_size:
            return torch.einsum(
                "qd,qnd->qn", query_repr, positive_item_repr[negative_indices]
            )
        return torch.cat(
            [
                _checkpointed(
                    lambda query, positives, indices: torch.einsum(
                        "qd,qnd->qn", query, positives[indices]
                    ),
                    query_repr,
                    positive_item_repr,
                    negative_indices[:, start : start + chunk_size],
                )
                for start in range(0, negative_indices.shape[1], chunk_size)
            ],
            dim=1,
        )

    def loss_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Split out so a caller needing the logits anyway does not score twice."""
        if self.correction in {"none", "yi2019"}:
            labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
            return F.cross_entropy(logits, labels)

        positive_weight = 1.0 - torch.softmax(logits.detach(), dim=1)[:, 0]
        return -(
            positive_weight * (logits[:, 0] - torch.logsumexp(logits[:, 1:], dim=1))
        ).mean()

    def forward(
        self,
        query_repr: torch.Tensor,
        positive_item_repr: torch.Tensor,
        positive_item_ids: torch.Tensor,
        group_sizes: torch.Tensor,
        negatives: tuple[torch.Tensor, torch.Tensor] | None = None,
        acceptable_positive_ids: torch.Tensor | None = None,
        acceptable_positive_offsets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.loss_from_logits(
            self.logits(
                query_repr,
                positive_item_repr,
                positive_item_ids,
                group_sizes,
                negatives,
                acceptable_positive_ids,
                acceptable_positive_offsets,
            )
        )


class GeneralizedBCELoss(InBatchSampledSoftmaxLoss):
    def __init__(
        self,
        catalog_size: int,
        num_in_batch_negatives: int = 0,
        t: float = 0.75,
        random_negatives: RandomCatalogNegatives | None = None,
        mask_false_negatives: bool = False,
        exclude_own_group: bool = False,
        score_chunk_size: int | None = DEFAULT_NEGATIVE_SCORE_CHUNK_SIZE,
    ) -> None:
        if (
            not isinstance(catalog_size, int)
            or isinstance(catalog_size, bool)
            or catalog_size <= 1
        ):
            raise ValueError("catalog_size must be greater than one")
        if not math.isfinite(t) or not 0 < t <= 1:
            raise ValueError("gBCE t must be in (0, 1]")
        super().__init__(
            num_in_batch_negatives=num_in_batch_negatives,
            correction="none",
            random_negatives=random_negatives,
            mask_false_negatives=mask_false_negatives,
            exclude_own_group=exclude_own_group,
            score_chunk_size=score_chunk_size,
        )
        self.catalog_size = catalog_size
        self.t = t

    def _log_q(self, ids: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(ids, dtype=torch.float32)

    def _q_total(self) -> torch.Tensor:
        return torch.tensor(float(self.catalog_size))

    def transform_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 2:
            raise ValueError("gBCE logits must have shape [examples, candidates]")
        num_negatives = logits.shape[-1] - 1
        if num_negatives < 1:
            raise ValueError("gBCE requires at least one negative")
        alpha = num_negatives / (self.catalog_size - 1)
        beta = alpha * (self.t * (1 - 1 / alpha) + 1 / alpha)
        positive = logits[:, :1].to(torch.float64)
        negatives = logits[:, 1:].to(torch.float64)
        epsilon = 1e-10
        probability = torch.sigmoid(positive).clamp(epsilon, 1 - epsilon)
        adjusted = probability.pow(-beta).clamp(
            1 + epsilon, torch.finfo(torch.float64).max
        )
        adjusted = (1 / (adjusted - 1)).clamp(epsilon, torch.finfo(torch.float64).max)
        return torch.cat([adjusted.log(), negatives], dim=1)

    def loss_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        transformed = self.transform_logits(logits)
        targets = torch.zeros_like(transformed)
        targets[:, 0] = 1
        return F.binary_cross_entropy_with_logits(transformed, targets)


class StreamingInBatchSoftmax(InBatchSampledSoftmaxLoss):
    def __init__(
        self,
        hash_size: int,
        num_in_batch_negatives: int,
        alpha: float = 0.01,
        correction: Correction = "yi2019",
        eps: float = 1e-12,
        random_negatives: RandomCatalogNegatives | None = None,
        correct_positive: bool = True,
        mask_false_negatives: bool = True,
        exclude_own_group: bool = True,
        correct_random_negatives: bool = True,
        uniform_mixture_fraction: float = 0.0,
        first_item_id: int = 0,
        normalize_streaming_over_valid_ids: bool = False,
        score_chunk_size: int | None = DEFAULT_NEGATIVE_SCORE_CHUNK_SIZE,
    ) -> None:
        super().__init__(
            num_in_batch_negatives,
            correction,
            eps,
            random_negatives,
            correct_positive,
            mask_false_negatives,
            exclude_own_group,
            correct_random_negatives,
            score_chunk_size,
        )
        if not 0 <= uniform_mixture_fraction < 1:
            raise ValueError("uniform_mixture_fraction must be in [0, 1)")
        if not 0 <= first_item_id < hash_size:
            raise ValueError("first_item_id must be in [0, hash_size)")
        self.hash_size = hash_size
        self.alpha = alpha
        self.uniform_mixture_fraction = uniform_mixture_fraction
        self.first_item_id = first_item_id
        self.normalize_streaming_over_valid_ids = normalize_streaming_over_valid_ids
        self.step = nn.Buffer(torch.zeros((), dtype=torch.long))
        # A and B of Yi et al. 2019, Algorithm 2
        self.last_seen_step = nn.Buffer(torch.zeros(hash_size, dtype=torch.long))
        self.smoothed_gap = nn.Buffer(torch.zeros(hash_size, dtype=torch.float32))

    def _hash(self, ids: torch.Tensor) -> torch.Tensor:
        return ids % self.hash_size

    @torch.no_grad()
    def _observe(self, ids: torch.Tensor) -> None:
        buckets = self._hash(ids.flatten())
        num_observations = buckets.numel()
        if num_observations == 0:
            return
        positions = self.step + torch.arange(
            1, num_observations + 1, device=buckets.device
        )
        order = torch.argsort(buckets, stable=True)
        sorted_buckets = buckets[order]
        sorted_positions = positions[order]
        starts_group = torch.ones(
            num_observations, dtype=torch.bool, device=buckets.device
        )
        starts_group[1:] = sorted_buckets[1:] != sorted_buckets[:-1]
        starts = starts_group.nonzero().flatten()
        ends = torch.cat([starts[1:], starts.new_tensor([num_observations])])
        counts = ends - starts
        group = starts_group.cumsum(0) - 1
        unique_buckets = sorted_buckets[starts]
        ranks = torch.arange(num_observations, device=buckets.device) - starts[group]

        previous_positions = sorted_positions.roll(1)
        previous_positions[starts] = self.last_seen_step[unique_buckets]
        gaps = (sorted_positions - previous_positions).clamp_min(1).float()
        previously_seen = self.last_seen_step[unique_buckets] != 0
        included = previously_seen[group] | ranks.ne(0)
        remaining_updates = counts[group] - ranks - 1
        weights = (1.0 - self.alpha) ** remaining_updates
        weighted_gaps = torch.zeros(
            unique_buckets.numel(), dtype=torch.float32, device=buckets.device
        )
        weighted_gaps.scatter_add_(0, group, torch.where(included, weights * gaps, 0.0))
        initial = torch.where(
            previously_seen,
            self.smoothed_gap[unique_buckets],
            gaps[starts],
        )
        update_counts = counts - (~previously_seen).long()
        self.smoothed_gap[unique_buckets] = (
            1.0 - self.alpha
        ) ** update_counts * initial + self.alpha * weighted_gaps
        self.last_seen_step[unique_buckets] = sorted_positions[ends - 1]
        self.step += num_observations

    def _gap(self, buckets: torch.Tensor) -> torch.Tensor:
        """Steps between two hits of a bucket, as far as the run can tell."""
        never_hit = self.last_seen_step[buckets] == 0
        return torch.where(
            never_hit, self.step.clamp_min(1).float(), self.smoothed_gap[buckets]
        )

    def _log_q(self, ids: torch.Tensor) -> torch.Tensor:
        log_q = -torch.log(self._gap(self._hash(ids)).clamp_min(self.eps))
        if self.uniform_mixture_fraction == 0:
            return log_q
        normalization_start = (
            self.first_item_id if self.normalize_streaming_over_valid_ids else 0
        )
        buckets = torch.arange(
            normalization_start,
            self.hash_size,
            device=self.smoothed_gap.device,
        )
        log_q -= torch.logsumexp(
            -torch.log(self._gap(buckets).clamp_min(self.eps)), dim=0
        )
        uniform_log_q = torch.full_like(log_q, -torch.inf)
        known = ids >= self.first_item_id
        if self.normalize_streaming_over_valid_ids:
            log_q = log_q.masked_fill(~known, -torch.inf)
        uniform_log_q[known] = -torch.log(
            log_q.new_tensor(self.hash_size - self.first_item_id)
        )
        fraction = log_q.new_tensor(self.uniform_mixture_fraction)
        return torch.logaddexp(
            log_q + torch.log1p(-fraction),
            uniform_log_q + fraction.log(),
        )

    def _q_total(self) -> torch.Tensor:
        if self.uniform_mixture_fraction:
            return self.smoothed_gap.new_ones(())
        buckets = torch.arange(self.hash_size, device=self.smoothed_gap.device)
        return (1.0 / self._gap(buckets).clamp_min(self.eps)).sum()


class OfflineInBatchSoftmax(InBatchSampledSoftmaxLoss):
    def __init__(
        self,
        q: torch.Tensor,
        num_in_batch_negatives: int,
        correction: Correction = "yi2019",
        eps: float = 1e-12,
        random_negatives: RandomCatalogNegatives | None = None,
        correct_positive: bool = True,
        mask_false_negatives: bool = True,
        exclude_own_group: bool = True,
        correct_random_negatives: bool = True,
        score_chunk_size: int | None = DEFAULT_NEGATIVE_SCORE_CHUNK_SIZE,
    ) -> None:
        super().__init__(
            num_in_batch_negatives,
            correction,
            eps,
            random_negatives,
            correct_positive,
            mask_false_negatives,
            exclude_own_group,
            correct_random_negatives,
            score_chunk_size,
        )
        self.q = nn.Buffer(q.detach().float())

    @classmethod
    def from_train_interactions(
        cls,
        train_data: pl.DataFrame,
        catalog_size: int,
        num_in_batch_negatives: int,
        item_id_column: str,
        correction: Correction = "yi2019",
        eps: float = 1e-12,
    ) -> Self:
        q = _item_frequencies(train_data, catalog_size, item_id_column, eps)
        return cls(q, num_in_batch_negatives, correction, eps)

    def _log_q(self, ids: torch.Tensor) -> torch.Tensor:
        return torch.log(self.q[ids].clamp_min(self.eps))

    def _q_total(self) -> torch.Tensor:
        return self.q.sum()


def _item_frequencies(
    train_data: pl.DataFrame,
    catalog_size: int,
    item_id_column: str,
    eps: float,
) -> torch.Tensor:
    grouped = train_data.group_by(item_id_column).agg(pl.len().alias("count"))
    item_ids = torch.from_numpy(grouped[item_id_column].to_numpy().astype(int))
    counts = torch.zeros(catalog_size, dtype=torch.float32)
    counts[item_ids] = torch.from_numpy(grouped["count"].to_numpy().astype("float32"))

    counts = counts.clamp(min=eps)
    return counts / counts.sum()
