from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

_ASSIGNMENT_CHUNK_SIZE = 4096


def _nearest_centroid(points: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    """Index of the closest centroid per point, chunked over points."""
    assignments = torch.empty(len(points), dtype=torch.int64, device=points.device)
    for start in range(0, len(points), _ASSIGNMENT_CHUNK_SIZE):
        chunk = points[start : start + _ASSIGNMENT_CHUNK_SIZE]
        assignments[start : start + len(chunk)] = torch.cdist(chunk, centroids).argmin(
            1
        )
    return assignments


def _initial_centroids(
    points: torch.Tensor, num_clusters: int, generator: torch.Generator
) -> torch.Tensor:
    sample = torch.randperm(len(points), generator=generator, device=points.device)
    repeats = -(-num_clusters // len(points))
    return points[sample.repeat(repeats)[:num_clusters]].clone()


def _fit_kmeans(
    points: torch.Tensor,
    num_clusters: int,
    num_iterations: int,
    generator: torch.Generator,
) -> torch.Tensor:
    centroids = _initial_centroids(points, num_clusters, generator)
    for _ in range(num_iterations):
        assignments = _nearest_centroid(points, centroids)
        totals = torch.zeros_like(centroids)
        totals.index_add_(0, assignments, points)
        counts = torch.bincount(assignments, minlength=num_clusters).unsqueeze(1)
        moved = counts > 0
        centroids = torch.where(moved, totals / counts.clamp(min=1), centroids)
    return centroids


@dataclass(frozen=True)
class ResidualCodebooks:
    """One codebook per quantization level: ``[levels, codes, dim]``."""

    centroids: torch.Tensor

    def __post_init__(self) -> None:
        assert self.centroids.ndim == 3, (
            f"expected [levels, codes, dim] centroids, got {tuple(self.centroids.shape)}"
        )

    @property
    def num_levels(self) -> int:
        return self.centroids.shape[0]

    @property
    def num_codes(self) -> int:
        return self.centroids.shape[1]

    @property
    def dim(self) -> int:
        return self.centroids.shape[2]

    def to(self, device: torch.device | str) -> ResidualCodebooks:
        return ResidualCodebooks(self.centroids.to(device))

    def encode(self, embeddings: torch.Tensor) -> torch.Tensor:
        residual = embeddings.to(self.centroids.dtype)
        codes = []
        for level in range(self.num_levels):
            level_codes = _nearest_centroid(residual, self.centroids[level])
            codes.append(level_codes)
            residual = residual - self.centroids[level][level_codes]
        return torch.stack(codes, dim=1)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        levels = [
            self.centroids[level][codes[:, level]] for level in range(self.num_levels)
        ]
        return torch.stack(levels).sum(0)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.centroids.cpu(), path)

    @classmethod
    def load(cls, path: Path) -> ResidualCodebooks:
        return cls(torch.load(path, weights_only=True))


def fit_residual_kmeans(
    embeddings: torch.Tensor,
    *,
    num_levels: int,
    num_codes: int,
    num_iterations: int = 20,
    seed: int = 0,
) -> ResidualCodebooks:
    """Fit one k-means codebook per level, each on the previous level's residual."""
    generator = torch.Generator(device=embeddings.device).manual_seed(seed)
    residual = embeddings.float()
    codebooks = []
    for level in range(num_levels):
        centroids = _fit_kmeans(residual, num_codes, num_iterations, generator)
        codebooks.append(centroids)
        residual = residual - centroids[_nearest_centroid(residual, centroids)]
        logger.info(
            "Residual k-means level %s/%s: mean squared residual %.6f",
            level + 1,
            num_levels,
            float(residual.pow(2).sum(1).mean()),
        )
    return ResidualCodebooks(torch.stack(codebooks))
