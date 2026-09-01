from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional

from .codes import SemanticCodes
from .residual_kmeans import ResidualCodebooks


@dataclass(frozen=True)
class SemanticIdDiagnostics:
    identifier_collision_rate: float
    collided_item_fraction: float
    unique_base_tuples: int
    collision_bucket_size_p50: float
    collision_bucket_size_p95: float
    collision_bucket_size_p99: float
    collision_bucket_size_max: int
    occupied_codes: tuple[int, ...]
    dead_code_fraction: tuple[float, ...]
    reconstruction_mse_by_depth: tuple[float, ...]
    p95_occupied_load: tuple[float, ...]
    p95_to_mean_occupied_load: tuple[float, ...]
    intra_code_cosine_similarity: tuple[float, ...]


def _level_statistics(
    codes: torch.Tensor, embeddings: torch.Tensor
) -> tuple[float, float, float]:
    _, inverse, counts = torch.unique(
        codes, sorted=True, return_inverse=True, return_counts=True
    )
    loads = counts.to(torch.float64)
    p95_load = torch.quantile(loads, 0.95, interpolation="higher")
    mean_load = loads.mean()

    normalized = functional.normalize(embeddings.to(torch.float64), dim=1)
    cluster_sums = normalized.new_zeros((len(counts), normalized.shape[1]))
    cluster_sums.index_add_(0, inverse, normalized)
    pair_sums = (cluster_sums.square().sum(1) - loads) / 2
    pair_counts = loads * (loads - 1) / 2
    total_pairs = pair_counts.sum()
    similarity = (
        pair_sums.sum() / total_pairs
        if bool(total_pairs > 0)
        else normalized.new_zeros(())
    )
    return float(p95_load), float(p95_load / mean_load), float(similarity)


def semantic_id_diagnostics(
    semantic_codes: SemanticCodes,
    embeddings: torch.Tensor,
    *,
    num_base_levels: int,
    codebooks: ResidualCodebooks | None = None,
) -> SemanticIdDiagnostics:
    if embeddings.ndim != 2 or embeddings.shape[0] != len(semantic_codes.item_ids):
        raise ValueError("expected one embedding per semantic id")
    if not 1 <= num_base_levels <= semantic_codes.num_levels:
        raise ValueError("num_base_levels must select existing code levels")
    if codebooks is not None and (
        codebooks.num_levels != num_base_levels or codebooks.dim != embeddings.shape[1]
    ):
        raise ValueError("codebooks must match the base levels and embedding width")

    base_codes = semantic_codes.codes[:, :num_base_levels]
    _, inverse, counts = torch.unique(
        base_codes, dim=0, return_inverse=True, return_counts=True
    )
    num_items = len(base_codes)
    identifier_collision_rate = 1 - len(counts) / num_items
    collided_item_fraction = float((counts[inverse] > 1).to(torch.float64).mean())
    bucket_sizes = counts.to(torch.float64)
    bucket_quantiles = torch.quantile(
        bucket_sizes,
        torch.tensor([0.5, 0.95, 0.99], dtype=torch.float64),
        interpolation="higher",
    )

    occupied_codes = tuple(
        int(torch.unique(base_codes[:, level]).numel())
        for level in range(num_base_levels)
    )
    dead_code_fraction = tuple(
        1 - occupied / semantic_codes.codes_per_level[level]
        for level, occupied in enumerate(occupied_codes)
    )
    reconstruction_mse_by_depth: tuple[float, ...] = ()
    if codebooks is not None:
        reconstruction = torch.zeros_like(embeddings, dtype=codebooks.centroids.dtype)
        errors = []
        for level in range(num_base_levels):
            reconstruction += codebooks.centroids[level][base_codes[:, level]]
            residual = embeddings.to(reconstruction.dtype) - reconstruction
            errors.append(float(residual.square().sum(1).mean()))
        reconstruction_mse_by_depth = tuple(errors)

    level_statistics = [
        _level_statistics(base_codes[:, level], embeddings)
        for level in range(num_base_levels)
    ]
    return SemanticIdDiagnostics(
        identifier_collision_rate=identifier_collision_rate,
        collided_item_fraction=collided_item_fraction,
        unique_base_tuples=len(counts),
        collision_bucket_size_p50=float(bucket_quantiles[0]),
        collision_bucket_size_p95=float(bucket_quantiles[1]),
        collision_bucket_size_p99=float(bucket_quantiles[2]),
        collision_bucket_size_max=int(counts.max()),
        occupied_codes=occupied_codes,
        dead_code_fraction=dead_code_fraction,
        reconstruction_mse_by_depth=reconstruction_mse_by_depth,
        p95_occupied_load=tuple(statistic[0] for statistic in level_statistics),
        p95_to_mean_occupied_load=tuple(
            statistic[1] for statistic in level_statistics
        ),
        intra_code_cosine_similarity=tuple(
            statistic[2] for statistic in level_statistics
        ),
    )
