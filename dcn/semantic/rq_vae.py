from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .residual_kmeans import ResidualCodebooks, fit_residual_kmeans

logger = logging.getLogger(__name__)

_KMEANS_ITERATIONS = 20


@dataclass
class QuantizationOutput:
    quantized: torch.Tensor
    codes: torch.Tensor
    codebook_loss: torch.Tensor
    commitment_loss: torch.Tensor


class ResidualQuantizer(nn.Module):
    """Learned residual codebooks with straight-through gradients."""

    def __init__(self, num_levels: int, num_codes: int, dim: int):
        super().__init__()
        self.num_levels = num_levels
        self.num_codes = num_codes
        self.centroids = nn.Parameter(torch.randn(num_levels, num_codes, dim) * 0.1)

    def initialize_from(self, codebooks: ResidualCodebooks) -> None:
        assert codebooks.centroids.shape == self.centroids.shape, (
            f"cannot seed {tuple(self.centroids.shape)} codebooks from"
            f" {tuple(codebooks.centroids.shape)}"
        )
        with torch.no_grad():
            self.centroids.copy_(codebooks.centroids.to(self.centroids))

    def codebooks(self) -> ResidualCodebooks:
        return ResidualCodebooks(self.centroids.detach())

    def forward(self, latents: torch.Tensor) -> QuantizationOutput:
        residual = latents
        quantized = torch.zeros_like(latents)
        codes = []
        codebook_loss = latents.new_zeros(())
        commitment_loss = latents.new_zeros(())
        for level in range(self.num_levels):
            centroids = self.centroids[level]
            level_codes = torch.cdist(residual.detach(), centroids).argmin(1)
            codes.append(level_codes)
            chosen = centroids[level_codes]
            # Per level, not on the summed quantization: one loss would hand
            # every level the same gradient and none would go coarse to fine.
            codebook_loss = codebook_loss + F.mse_loss(chosen, residual.detach())
            commitment_loss = commitment_loss + F.mse_loss(residual, chosen.detach())
            quantized = quantized + chosen
            residual = residual - chosen.detach()

        return QuantizationOutput(
            quantized=latents + (quantized - latents).detach(),
            codes=torch.stack(codes, dim=1),
            codebook_loss=codebook_loss / self.num_levels,
            commitment_loss=commitment_loss / self.num_levels,
        )


class RqVae(nn.Module):
    """Autoencoder whose bottleneck is a residual quantizer."""

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        quantizer: ResidualQuantizer,
        commitment_weight: float = 0.25,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.quantizer = quantizer
        self.commitment_weight = commitment_weight

    def forward(self, embeddings: torch.Tensor) -> dict[str, torch.Tensor]:
        quantization = self.quantizer(self.encoder(embeddings))
        reconstruction_loss = F.mse_loss(
            self.decoder(quantization.quantized), embeddings
        )
        return {
            "loss": (
                reconstruction_loss
                + quantization.codebook_loss
                + self.commitment_weight * quantization.commitment_loss
            ),
            "reconstruction_loss": reconstruction_loss,
            "codebook_loss": quantization.codebook_loss,
            "commitment_loss": quantization.commitment_loss,
        }

    @torch.no_grad()
    def initialize_codebooks(self, embeddings: torch.Tensor, seed: int = 0) -> None:
        """Seed the codebooks with residual k-means over the current latents."""
        latents = self.encoder(embeddings)
        self.quantizer.initialize_from(
            fit_residual_kmeans(
                latents,
                num_levels=self.quantizer.num_levels,
                num_codes=self.quantizer.num_codes,
                num_iterations=_KMEANS_ITERATIONS,
                seed=seed,
            )
        )

    @torch.no_grad()
    def codes(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.quantizer(self.encoder(embeddings)).codes
