"""Semantic ids: the stage that assigns them and the experiment that reads them."""

from __future__ import annotations

import dataclasses
import hashlib
import logging
from abc import abstractmethod
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from dcn.config.experiment import TrainingStage
from dcn.config.sequence import SequenceExperiment
from dcn.semantic import (
    ResidualCodebooks,
    ResidualQuantizer,
    RqVae,
    SemanticCodes,
    fit_residual_kmeans,
)
from dcn.semantic.artifacts import load_item_embeddings
from dcn.training import EpochTrainer
from neuralrec.run.callbacks import LoggingCallback

logger = logging.getLogger(__name__)

Quantizer = Literal["kmeans", "rqvae"]


@dataclass(frozen=True)
class SemanticIdConfig:
    num_levels: int = 3
    num_codes: int = 1024
    quantizer: Quantizer = "kmeans"
    kmeans_iterations: int = 20

    latent_dim: int = 128
    hidden_dim: int = 256
    num_epochs: int = 5
    batch_size: int = 1024
    learning_rate: float = 1e-3
    seed: int = 42

    @property
    def cache_key(self) -> str:
        """Readable prefix plus a digest of *every* field: a field left out
        would hand a run the codes some other setting produced."""
        digest = hashlib.sha1(repr(dataclasses.astuple(self)).encode()).hexdigest()[:10]
        return f"{self.quantizer}_{self.num_levels}x{self.num_codes}_{digest}"


class SemanticIdStage(TrainingStage):
    """Assigns every item a code tuple and writes it next to the dataset."""

    def __init__(
        self,
        *,
        embeddings_parquet: Path,
        codes_path: Path,
        codebooks_path: Path,
        config: SemanticIdConfig,
        device: torch.device,
        invalidate_cache: bool = False,
    ):
        self.embeddings_parquet = embeddings_parquet
        self.codes_path = codes_path
        self.codebooks_path = codebooks_path
        self.config = config
        self.device = device
        self.invalidate_cache = invalidate_cache

    @property
    def name(self) -> str:
        return f"semantic_ids_{self.config.quantizer}"

    def run(self) -> None:
        if not self.invalidate_cache and self.codes_path.exists():
            logger.info("Semantic ids already assigned at %s", self.codes_path)
            return

        item_ids, embeddings = load_item_embeddings(self.embeddings_parquet)
        logger.info(
            "Quantizing %s item embeddings into %s levels of %s codes",
            len(item_ids),
            self.config.num_levels,
            self.config.num_codes,
        )
        codebooks, codes = self.quantize(embeddings.to(self.device))
        semantic_codes = SemanticCodes.with_collision_suffix(
            item_ids, codes.cpu(), self.config.num_codes
        )
        logger.info(
            "%s items share a code tuple with another; widest bucket holds %s",
            int((semantic_codes.codes[:, -1] > 1).sum()),
            semantic_codes.codes_per_level[-1],
        )
        codebooks.save(self.codebooks_path)
        semantic_codes.save(self.codes_path)

    @abstractmethod
    def quantize(
        self, embeddings: torch.Tensor
    ) -> tuple[ResidualCodebooks, torch.Tensor]: ...


class KMeansIdStage(SemanticIdStage):
    def quantize(
        self, embeddings: torch.Tensor
    ) -> tuple[ResidualCodebooks, torch.Tensor]:
        codebooks = fit_residual_kmeans(
            embeddings,
            num_levels=self.config.num_levels,
            num_codes=self.config.num_codes,
            num_iterations=self.config.kmeans_iterations,
            seed=self.config.seed,
        )
        return codebooks, codebooks.encode(embeddings)


class RqVaeIdStage(SemanticIdStage):
    """Codes from an autoencoder whose bottleneck is a residual quantizer."""

    def quantize(
        self, embeddings: torch.Tensor
    ) -> tuple[ResidualCodebooks, torch.Tensor]:
        model = self._build_model(embeddings.shape[1]).to(self.device)
        model.initialize_codebooks(embeddings, seed=self.config.seed)

        EpochTrainer(
            model=model,
            optimizer=torch.optim.Adam(
                model.parameters(), lr=self.config.learning_rate, fused=True
            ),
            train_loader=DataLoader(
                TensorDataset(embeddings.cpu()),
                batch_size=self.config.batch_size,
                shuffle=True,
                collate_fn=lambda rows: torch.stack([row[0] for row in rows]),
            ),
            num_epochs=self.config.num_epochs,
            callbacks=[LoggingCallback()],
        ).train()

        return model.quantizer.codebooks().to("cpu"), model.codes(embeddings)

    def _build_model(self, embedding_dim: int) -> RqVae:
        return RqVae(
            encoder=nn.Sequential(
                nn.Linear(embedding_dim, self.config.hidden_dim),
                nn.GELU(),
                nn.Linear(self.config.hidden_dim, self.config.latent_dim),
            ),
            decoder=nn.Sequential(
                nn.Linear(self.config.latent_dim, self.config.hidden_dim),
                nn.GELU(),
                nn.Linear(self.config.hidden_dim, embedding_dim),
            ),
            quantizer=ResidualQuantizer(
                num_levels=self.config.num_levels,
                num_codes=self.config.num_codes,
                dim=self.config.latent_dim,
            ),
        )


@dataclass
class SemanticExperiment(SequenceExperiment):
    """A sequence experiment whose items carry semantic ids."""

    semantic: SemanticIdConfig = field(default_factory=SemanticIdConfig)

    @property
    def stages(self) -> list[TrainingStage]:
        return [self.semantic_stage, self]

    @cached_property
    def semantic_stage(self) -> SemanticIdStage:
        stage_types: dict[Quantizer, type[SemanticIdStage]] = {
            "kmeans": KMeansIdStage,
            "rqvae": RqVaeIdStage,
        }
        return stage_types[self.semantic.quantizer](
            embeddings_parquet=Path(
                self.artifacts.precomputed_embeddings[self.item_id_column]
            ),
            codes_path=self._semantic_dir / "codes.pt",
            codebooks_path=self._semantic_dir / "codebooks.pt",
            config=self.semantic,
            device=self.device,
            invalidate_cache=self.invalidate_cache,
        )

    @property
    def _semantic_dir(self) -> Path:
        return self.dataset_cache_dir / "semantic" / self.semantic.cache_key

    @cached_property
    def semantic_codes(self) -> SemanticCodes:
        return SemanticCodes.load(self.semantic_stage.codes_path)

    @cached_property
    def semantic_codebooks(self) -> ResidualCodebooks:
        return ResidualCodebooks.load(self.semantic_stage.codebooks_path)
