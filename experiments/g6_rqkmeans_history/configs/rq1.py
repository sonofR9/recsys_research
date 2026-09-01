from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math

import torch
from torch import nn

from dcn.config import SemanticHistoryExperiment, SemanticIdConfig
from dcn.models import SequenceRetrievalModel
from dcn.nn.semantic_embedding import (
    CombinedSemanticIdEmbedding,
    SemanticIdEmbedding,
)
from experiments.g6_rqkmeans_history.configs.rq0 import _common
from experiments.g6_rqkmeans_history.protocol.rq1_manifest import (
    Rq1SearchJob,
    SidLookupInitialization,
    validate_rq1_search_job,
)


_CONTENT_PROJECTION = "per_level_centered_pca_v1"


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    contiguous = tensor.detach().cpu().contiguous()
    return contiguous.view(torch.uint8).numpy().tobytes()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(_tensor_bytes(tensor)).hexdigest()


def project_centroids_with_pca(
    centroids: torch.Tensor, output_dim: int
) -> torch.Tensor:
    if output_dim > centroids.shape[1]:
        raise ValueError(
            f"cannot project {centroids.shape[1]}-dimensional centroids to "
            f"{output_dim} dimensions"
        )
    if output_dim > min(centroids.shape):
        raise ValueError("not enough centroids for the requested PCA width")
    centered = centroids.detach().cpu().to(torch.float64)
    centered = centered - centered.mean(dim=0, keepdim=True)
    _, _, right_vectors = torch.linalg.svd(centered, full_matrices=False)
    components = right_vectors[:output_dim].clone()
    pivot_columns = components.abs().argmax(dim=1)
    signs = components[
        torch.arange(output_dim), pivot_columns
    ].sign().clamp(min=-1, max=1)
    signs[signs == 0] = 1
    components *= signs.unsqueeze(1)
    return centered @ components.T


def _match_rms(projected: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    target = reference.detach().cpu().to(torch.float64).square().mean().sqrt()
    current = projected.square().mean().sqrt()
    if not math.isfinite(float(current)) or current <= 0:
        raise ValueError("projected centroids have zero or non-finite RMS")
    return projected * (target / current)


def rq1_learned_sid_embedding(model: nn.Module) -> SemanticIdEmbedding:
    combined = [
        module
        for module in model.modules()
        if isinstance(module, CombinedSemanticIdEmbedding)
    ]
    if len(combined) != 1:
        raise RuntimeError("RQ1 requires exactly one combined SID embedding")
    learned = [
        embedding
        for embedding in combined[0].embeddings
        if isinstance(embedding.embedding, nn.Embedding)
        and embedding.embedding.weight.requires_grad
    ]
    if len(learned) != 1:
        raise RuntimeError("RQ1 requires exactly one trainable SID lookup")
    return learned[0]


@dataclass
class Rq1InitializationExperiment(SemanticHistoryExperiment):
    sid_lookup_initialization: SidLookupInitialization = "random"
    _sid_initialization_diagnostics: dict[str, object] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.sid_lookup_initialization not in {"random", "content_pca"}:
            raise ValueError("unknown SID lookup initialization")
        if self.history_representation != "item_learned_frozen_sid_event":
            raise ValueError(
                "RQ1 initialization requires item_learned_frozen_sid_event"
            )

    @property
    def sid_initialization_diagnostics(self) -> dict[str, object]:
        if self._sid_initialization_diagnostics is None:
            raise RuntimeError("RQ1 SID initialization has not run")
        return self._sid_initialization_diagnostics

    def apply_post_mup_initialization(self, model: nn.Module) -> None:
        rng_before = torch.get_rng_state()
        learned = rq1_learned_sid_embedding(model)
        weight = learned.embedding.weight
        vocabulary = self.semantic_codes.vocabulary
        base_ranges = [
            vocabulary.level_range(level)
            for level in range(self.semantic_codebooks.num_levels)
        ]
        before = torch.cat(
            [weight[first:last].detach().cpu() for first, last in base_ranges]
        )
        level_diagnostics = []
        if self.sid_lookup_initialization == "content_pca":
            with torch.no_grad():
                for level, (first, last) in enumerate(base_ranges):
                    reference = weight[first:last]
                    projected = project_centroids_with_pca(
                        self.semantic_codebooks.centroids[level], weight.shape[1]
                    )
                    projected = _match_rms(projected, reference)
                    initialized = projected.to(
                        device=weight.device, dtype=weight.dtype
                    )
                    initialized = _match_rms(initialized, reference).to(
                        device=weight.device, dtype=weight.dtype
                    )
                    weight[first:last].copy_(initialized)
                    level_diagnostics.append(
                        {
                            "level": level,
                            "random_rms": float(
                                reference.detach().float().square().mean().sqrt()
                            ),
                            "initialized_rms": float(
                                initialized.detach().float().square().mean().sqrt()
                            ),
                        }
                    )
        else:
            level_diagnostics = [
                {
                    "level": level,
                    "random_rms": float(
                        weight[first:last].detach().float().square().mean().sqrt()
                    ),
                    "initialized_rms": float(
                        weight[first:last].detach().float().square().mean().sqrt()
                    ),
                }
                for level, (first, last) in enumerate(base_ranges)
            ]
        base_mask = torch.zeros(len(weight), dtype=torch.bool, device=weight.device)
        for first, last in base_ranges:
            base_mask[first:last] = True
        after = torch.cat(
            [weight[first:last].detach().cpu() for first, last in base_ranges]
        )
        self._sid_initialization_diagnostics = {
            "version": 1,
            "mode": self.sid_lookup_initialization,
            "projection": (
                _CONTENT_PROJECTION
                if self.sid_lookup_initialization == "content_pca"
                else None
            ),
            "base_rows_before_sha256": _tensor_sha256(before),
            "base_rows_after_sha256": _tensor_sha256(after),
            "non_base_rows_sha256": _tensor_sha256(weight[~base_mask]),
            "codebook_centroids_sha256": _tensor_sha256(
                self.semantic_codebooks.centroids
            ),
            "rng_nonadvancing": torch.equal(rng_before, torch.get_rng_state()),
            "levels": level_diagnostics,
        }

    def generation_architecture_metadata(self) -> dict[str, object]:
        metadata = super().generation_architecture_metadata()
        metadata |= {
            "history_representation": self.history_representation,
            "representation_width": self.representation_width,
            "sid_lookup_initialization": self.sid_lookup_initialization,
            "sid_lookup_projection": (
                _CONTENT_PROJECTION
                if self.sid_lookup_initialization == "content_pca"
                else None
            ),
        }
        if self._sid_initialization_diagnostics is not None:
            metadata["sid_initialization_diagnostics"] = (
                self.sid_initialization_diagnostics
            )
        return metadata


def build_rq1_initialization(
    initialization: SidLookupInitialization,
    *,
    embedding_learning_rate: float,
    deep_learning_rate: float,
    run_name: str,
    training_seed: int = 42,
) -> Rq1InitializationExperiment:
    common = _common(
            "best_g1",
            batch_size=256,
            validation_batch_size=8192,
            embedding_learning_rate=embedding_learning_rate,
            deep_learning_rate=deep_learning_rate,
            run_name=run_name,
        )
    common["seed"] = training_seed
    return Rq1InitializationExperiment(
        **common,
        history_representation="item_learned_frozen_sid_event",
        representation_width=32,
        sid_lookup_initialization=initialization,
        semantic=SemanticIdConfig(
            quantizer="kmeans",
            num_levels=4,
            num_codes=512,
            kmeans_iterations=20,
            seed=42,
        ),
    )


def build_rq1_search_experiment(
    job: Rq1SearchJob,
) -> Rq1InitializationExperiment:
    validate_rq1_search_job(job)
    if job.reused:
        raise ValueError("RQ1 random carryover must not be rebuilt")
    return build_rq1_initialization(
        job.initialization,
        embedding_learning_rate=job.coordinate.embedding_learning_rate,
        deep_learning_rate=job.coordinate.deep_learning_rate,
        run_name=job.physical_run_name,
    )
