from __future__ import annotations

import logging
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import torch

logger = logging.getLogger(__name__)

_ASSIGNMENT_CHUNK_SIZE = 4096
KMEANS_FITTER_REVISION = "convergent-lloyd-v1"


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


@dataclass(frozen=True)
class KMeansLevelDiagnostics:
    level: int
    iterations_run: int
    stop_reason: Literal["assignments_stable", "relative_inertia", "max_iterations"]
    initial_inertia: float
    final_inertia: float
    final_relative_inertia_improvement: float | None
    final_assignment_changes: int | None
    inertia_increase_count: int
    wall_time_seconds: float = 0.0
    peak_allocated_bytes: int = 0
    peak_reserved_bytes: int = 0
    occupied_codes: int = 0
    dead_code_fraction: float = 0.0
    residual_mse: float = 0.0


@dataclass(frozen=True)
class ResidualKMeansDiagnostics:
    max_iterations: int
    relative_inertia_tolerance: float | None
    assignment_early_stopping: bool
    levels: tuple[KMeansLevelDiagnostics, ...]


@dataclass(frozen=True)
class ResidualKMeansFit:
    codebooks: ResidualCodebooks
    codes: torch.Tensor
    diagnostics: ResidualKMeansDiagnostics


def _fit_kmeans(
    points: torch.Tensor,
    num_clusters: int,
    max_iterations: int,
    generator: torch.Generator,
    *,
    level: int,
    relative_inertia_tolerance: float | None,
    assignment_early_stopping: bool,
) -> tuple[torch.Tensor, KMeansLevelDiagnostics, torch.Tensor]:
    centroids = _initial_centroids(points, num_clusters, generator)
    assignments = _nearest_centroid(points, centroids)
    inertia = float((points - centroids[assignments]).square().sum())
    initial_inertia = inertia
    final_relative_improvement: float | None = None
    final_assignment_changes: int | None = None
    inertia_increase_count = 0
    stop_reason: Literal["assignments_stable", "relative_inertia", "max_iterations"] = (
        "max_iterations"
    )
    iterations_run = 0
    for iteration in range(1, max_iterations + 1):
        totals = torch.zeros_like(centroids)
        totals.index_add_(0, assignments, points)
        counts = torch.bincount(assignments, minlength=num_clusters).unsqueeze(1)
        moved = counts > 0
        centroids = torch.where(moved, totals / counts.clamp(min=1), centroids)
        updated_assignments = _nearest_centroid(points, centroids)
        updated_inertia = float(
            (points - centroids[updated_assignments]).square().sum()
        )
        final_assignment_changes = int((updated_assignments != assignments).sum())
        final_relative_improvement = (inertia - updated_inertia) / max(
            abs(inertia), torch.finfo(points.dtype).eps
        )
        inertia_increase_count += int(final_relative_improvement < 0)
        iterations_run = iteration
        assignments = updated_assignments
        inertia = updated_inertia
        if assignment_early_stopping and final_assignment_changes == 0:
            stop_reason = "assignments_stable"
            break
        if (
            relative_inertia_tolerance is not None
            and final_relative_improvement is not None
            and final_relative_improvement >= 0
            and final_relative_improvement <= relative_inertia_tolerance
        ):
            stop_reason = "relative_inertia"
            break
    return (
        centroids,
        KMeansLevelDiagnostics(
            level=level,
            iterations_run=iterations_run,
            stop_reason=stop_reason,
            initial_inertia=initial_inertia,
            final_inertia=inertia,
            final_relative_inertia_improvement=final_relative_improvement,
            final_assignment_changes=final_assignment_changes,
            inertia_increase_count=inertia_increase_count,
        ),
        assignments,
    )


def _fit_kmeans_fixed(
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
        assert (
            self.centroids.ndim == 3
        ), f"expected [levels, codes, dim] centroids, got {tuple(self.centroids.shape)}"

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
    for _ in range(num_levels):
        centroids = _fit_kmeans_fixed(residual, num_codes, num_iterations, generator)
        codebooks.append(centroids)
        residual = residual - centroids[_nearest_centroid(residual, centroids)]
    return ResidualCodebooks(torch.stack(codebooks))


def fit_residual_kmeans_with_diagnostics(
    embeddings: torch.Tensor,
    *,
    num_levels: int,
    num_codes: int,
    max_iterations: int = 300,
    relative_inertia_tolerance: float | None = 1e-4,
    assignment_early_stopping: bool = True,
    seed: int = 0,
) -> ResidualKMeansFit:
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    if relative_inertia_tolerance is not None and relative_inertia_tolerance < 0:
        raise ValueError("relative_inertia_tolerance must be nonnegative")
    generator = torch.Generator(device=embeddings.device).manual_seed(seed)
    residual = embeddings.float()
    codebooks = []
    codes = []
    diagnostics = []
    for level in range(num_levels):
        if residual.is_cuda:
            torch.cuda.reset_peak_memory_stats(residual.device)
        started = time.perf_counter()
        centroids, level_diagnostics, level_codes = _fit_kmeans(
            residual,
            num_codes,
            max_iterations,
            generator,
            level=level,
            relative_inertia_tolerance=relative_inertia_tolerance,
            assignment_early_stopping=assignment_early_stopping,
        )
        codebooks.append(centroids)
        codes.append(level_codes)
        residual = residual - centroids[level_codes]
        occupied_codes = int(torch.unique(level_codes).numel())
        diagnostics.append(
            replace(
                level_diagnostics,
                wall_time_seconds=time.perf_counter() - started,
                peak_allocated_bytes=(
                    torch.cuda.max_memory_allocated(residual.device)
                    if residual.is_cuda
                    else 0
                ),
                peak_reserved_bytes=(
                    torch.cuda.max_memory_reserved(residual.device)
                    if residual.is_cuda
                    else 0
                ),
                occupied_codes=occupied_codes,
                dead_code_fraction=1 - occupied_codes / num_codes,
                residual_mse=float(residual.square().sum(1).mean()),
            )
        )
        logger.info(
            "Residual k-means level %s/%s: mean squared residual %.6f",
            level + 1,
            num_levels,
            float(residual.pow(2).sum(1).mean()),
        )
    return ResidualKMeansFit(
        codebooks=ResidualCodebooks(torch.stack(codebooks)),
        codes=torch.stack(codes, dim=1),
        diagnostics=ResidualKMeansDiagnostics(
            max_iterations=max_iterations,
            relative_inertia_tolerance=relative_inertia_tolerance,
            assignment_early_stopping=assignment_early_stopping,
            levels=tuple(diagnostics),
        ),
    )
