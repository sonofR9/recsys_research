"""Semantic ids: the stage that assigns them and the experiment that reads them."""

from __future__ import annotations

import hashlib
import fcntl
import json
import logging
from abc import abstractmethod
from dataclasses import asdict, dataclass, field
from functools import cached_property
import math
from pathlib import Path
from typing import Callable, Literal, get_args

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from dcn.config.experiment import TrainingStage
from dcn.config.sequence import SequenceExperiment
from dcn.semantic import (
    KMEANS_FITTER_REVISION,
    ResidualCodebooks,
    ResidualQuantizer,
    RqVae,
    SemanticCodes,
    fit_residual_kmeans,
    fit_residual_kmeans_with_diagnostics,
)
from dcn.semantic.artifacts import load_item_embeddings
from dcn.training import EpochTrainer
from neuralrec.run.callbacks import LoggingCallback

logger = logging.getLogger(__name__)

Quantizer = Literal["kmeans", "rqvae"]
CollisionPolicy = Literal["suffix", "none"]
KMEANS_MATERIALIZATION_REVISION = "shared-base-v2"


@dataclass(frozen=True)
class SemanticIdConfig:
    num_levels: int = 3
    num_codes: int = 1024
    quantizer: Quantizer = "kmeans"
    kmeans_iterations: int = 20
    kmeans_relative_inertia_tolerance: float | None = None
    kmeans_assignment_early_stopping: bool = False
    collision_policy: CollisionPolicy = "suffix"

    latent_dim: int = 128
    hidden_dim: int = 256
    num_epochs: int = 5
    batch_size: int = 1024
    learning_rate: float = 1e-3
    seed: int = 42

    def __post_init__(self) -> None:
        if self.collision_policy not in get_args(CollisionPolicy):
            raise ValueError(f"unknown collision policy {self.collision_policy!r}")
        if self.kmeans_relative_inertia_tolerance is not None and (
            not math.isfinite(self.kmeans_relative_inertia_tolerance)
            or self.kmeans_relative_inertia_tolerance < 0
        ):
            raise ValueError(
                "kmeans_relative_inertia_tolerance must be nonnegative finite"
            )

    @property
    def convergence_enabled(self) -> bool:
        return (
            self.kmeans_relative_inertia_tolerance is not None
            or self.kmeans_assignment_early_stopping
        )

    @property
    def base_cache_key(self) -> str:
        values = (
            self.num_levels,
            self.num_codes,
            self.quantizer,
            self.kmeans_iterations,
            self.latent_dim,
            self.hidden_dim,
            self.num_epochs,
            self.batch_size,
            self.learning_rate,
            self.seed,
        )
        if self.convergence_enabled:
            values += (
                KMEANS_FITTER_REVISION,
                KMEANS_MATERIALIZATION_REVISION,
                self.kmeans_relative_inertia_tolerance,
                self.kmeans_assignment_early_stopping,
            )
        digest = hashlib.sha1(repr(values).encode()).hexdigest()[:10]
        return f"{self.quantizer}_{self.num_levels}x{self.num_codes}_{digest}"

    @property
    def cache_key(self) -> str:
        """Readable prefix plus a digest of *every* field: a field left out
        would hand a run the codes some other setting produced."""
        if self.collision_policy == "suffix":
            return self.base_cache_key
        return f"{self.base_cache_key}_collision_{self.collision_policy}"


class SemanticIdStage(TrainingStage):
    """Assigns every item a code tuple and writes it next to the dataset."""

    def __init__(
        self,
        *,
        embeddings_parquet: Path,
        codes_path: Path,
        codebooks_path: Path,
        fit_diagnostics_path: Path,
        config: SemanticIdConfig,
        device: torch.device,
        invalidate_cache: bool = False,
    ):
        self.embeddings_parquet = embeddings_parquet
        self.codes_path = codes_path
        self.codebooks_path = codebooks_path
        self.fit_diagnostics_path = fit_diagnostics_path
        self.config = config
        self.device = device
        self.invalidate_cache = invalidate_cache

    @property
    def name(self) -> str:
        return f"semantic_ids_{self.config.quantizer}"

    @property
    def materialization_lock_path(self) -> Path:
        return self.codebooks_path.parent / ".materialization.lock"

    def run(self) -> None:
        self.materialization_lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.materialization_lock_path.open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            self._run_locked()

    def _run_locked(self) -> None:
        if not self.invalidate_cache and self.cache_complete:
            logger.info("Semantic ids already assigned at %s", self.codes_path)
            return

        item_ids, embeddings = load_item_embeddings(self.embeddings_parquet)
        logger.info(
            "Quantizing %s item embeddings into %s levels of %s codes",
            len(item_ids),
            self.config.num_levels,
            self.config.num_codes,
        )
        loaded_codebooks = (
            self.config.quantizer == "kmeans"
            and not self.invalidate_cache
            and self.codebooks_path.exists()
            and self.can_reuse_codebooks
        )
        if loaded_codebooks:
            codebooks, codes = self.reuse_codebooks(item_ids, embeddings)
        else:
            codebooks, codes = self.quantize(embeddings.to(self.device))
        codes = codes.cpu()
        semantic_codes = (
            SemanticCodes.with_collision_suffix(item_ids, codes, self.config.num_codes)
            if self.config.collision_policy == "suffix"
            else SemanticCodes.without_collision_suffix(
                item_ids, codes, self.config.num_codes
            )
        )
        _, inverse, counts = torch.unique(
            codes, dim=0, return_inverse=True, return_counts=True
        )
        logger.info(
            "%s items share a code tuple with another; widest bucket holds %s",
            int((counts[inverse] > 1).sum()),
            int(counts.max()),
        )
        self.materialize(codebooks, semantic_codes, loaded_codebooks=loaded_codebooks)

    def reuse_codebooks(
        self, item_ids: torch.Tensor, embeddings: torch.Tensor
    ) -> tuple[ResidualCodebooks, torch.Tensor]:
        codebooks = ResidualCodebooks.load(self.codebooks_path).to(self.device)
        return codebooks, codebooks.encode(embeddings.to(self.device))

    def materialize(
        self,
        codebooks: ResidualCodebooks,
        semantic_codes: SemanticCodes,
        *,
        loaded_codebooks: bool,
    ) -> None:
        if not loaded_codebooks:
            codebooks.save(self.codebooks_path)
        semantic_codes.save(self.codes_path)

    @property
    def cache_complete(self) -> bool:
        return self.codes_path.exists()

    @property
    def can_reuse_codebooks(self) -> bool:
        return True

    @abstractmethod
    def quantize(
        self, embeddings: torch.Tensor
    ) -> tuple[ResidualCodebooks, torch.Tensor]: ...


class KMeansIdStage(SemanticIdStage):
    @property
    def base_codes_path(self) -> Path:
        return self.codebooks_path.parent / "base_codes.pt"

    @property
    def fit_materialization_marker_path(self) -> Path:
        return self.codebooks_path.parent / "kmeans_fit_materialization.json"

    @property
    def materialization_marker_path(self) -> Path:
        return self.codes_path.parent / "kmeans_materialization.json"

    def _run_locked(self) -> None:
        if not self.config.convergence_enabled:
            super()._run_locked()
            return
        base_paths = (
            self.codebooks_path,
            self.base_codes_path,
            self.fit_diagnostics_path,
            self.fit_materialization_marker_path,
        )
        policy_paths = (self.codes_path, self.materialization_marker_path)
        if self.invalidate_cache and any(
            path.exists() for path in (*base_paths, *policy_paths)
        ):
            raise RuntimeError("authenticated KMeans materialization is immutable")
        if any(path.exists() for path in base_paths) and not self.can_reuse_codebooks:
            raise RuntimeError("authenticated KMeans fit is incomplete or corrupted")
        if any(path.exists() for path in policy_paths) and not self.cache_complete:
            raise RuntimeError(
                "authenticated KMeans collision materialization is incomplete or corrupted"
            )
        super()._run_locked()

    @property
    def cache_complete(self) -> bool:
        if not self.config.convergence_enabled:
            return super().cache_complete
        artifacts_exist = (
            self.codes_path.exists() and self.materialization_marker_path.exists()
        )
        return (
            artifacts_exist
            and self.can_reuse_codebooks
            and self._collision_materialization_marker_is_valid()
        )

    @property
    def can_reuse_codebooks(self) -> bool:
        if not self.config.convergence_enabled:
            return True
        try:
            return (
                self.convergence_diagnostics() is not None
                and self._fit_materialization_marker_is_valid()
            )
        except RuntimeError:
            logger.warning(
                "Refitting KMeans because cached convergence diagnostics are invalid"
            )
            return False

    def materialize(
        self,
        codebooks: ResidualCodebooks,
        semantic_codes: SemanticCodes,
        *,
        loaded_codebooks: bool,
    ) -> None:
        if not self.config.convergence_enabled:
            super().materialize(
                codebooks, semantic_codes, loaded_codebooks=loaded_codebooks
            )
            return
        if not loaded_codebooks:
            base_codes = SemanticCodes.without_collision_suffix(
                semantic_codes.item_ids,
                semantic_codes.codes[:, : self.config.num_levels],
                self.config.num_codes,
            )
            self._atomic_save(codebooks.save, self.codebooks_path)
            self._atomic_save(base_codes.save, self.base_codes_path)
            fit_artifacts = {
                "codebooks": self.codebooks_path,
                "base_codes": self.base_codes_path,
                "fit_diagnostics": self.fit_diagnostics_path,
            }
            fit_document = {
                "schema": "residual-kmeans-fit-materialization/v1",
                "fitter_revision": KMEANS_FITTER_REVISION,
                "materialization_revision": KMEANS_MATERIALIZATION_REVISION,
                "semantic_base_cache_key": self.config.base_cache_key,
                "artifact_sha256": self._artifact_hashes(fit_artifacts),
            }
            self._atomic_write_text(
                self.fit_materialization_marker_path,
                json.dumps(fit_document, indent=2, sort_keys=True) + "\n",
            )
        self._atomic_save(semantic_codes.save, self.codes_path)
        document = {
            "schema": "residual-kmeans-collision-materialization/v1",
            "fitter_revision": KMEANS_FITTER_REVISION,
            "materialization_revision": KMEANS_MATERIALIZATION_REVISION,
            "semantic_base_cache_key": self.config.base_cache_key,
            "semantic_cache_key": self.config.cache_key,
            "fit_materialization_sha256": self._file_hash(
                self.fit_materialization_marker_path
            ),
            "codes_sha256": self._file_hash(self.codes_path),
        }
        self._atomic_write_text(
            self.materialization_marker_path,
            json.dumps(document, indent=2, sort_keys=True) + "\n",
        )

    @staticmethod
    def _atomic_save(save: Callable[[Path], None], path: Path) -> None:
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.unlink(missing_ok=True)
        save(temporary)
        temporary.replace(path)

    @staticmethod
    def _atomic_write_text(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.unlink(missing_ok=True)
        temporary.write_text(content)
        temporary.replace(path)

    @staticmethod
    def _file_hash(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @classmethod
    def _artifact_hashes(cls, artifacts: dict[str, Path]) -> dict[str, str]:
        return {name: cls._file_hash(path) for name, path in artifacts.items()}

    def _fit_materialization_marker_is_valid(self) -> bool:
        try:
            document = json.loads(self.fit_materialization_marker_path.read_text())
        except (OSError, json.JSONDecodeError):
            return False
        expected = {
            "schema": "residual-kmeans-fit-materialization/v1",
            "fitter_revision": KMEANS_FITTER_REVISION,
            "materialization_revision": KMEANS_MATERIALIZATION_REVISION,
            "semantic_base_cache_key": self.config.base_cache_key,
        }
        if (
            not isinstance(document, dict)
            or set(document) != {*expected, "artifact_sha256"}
            or any(document.get(name) != value for name, value in expected.items())
        ):
            return False
        hashes = document.get("artifact_sha256")
        artifacts = {
            "codebooks": self.codebooks_path,
            "base_codes": self.base_codes_path,
            "fit_diagnostics": self.fit_diagnostics_path,
        }
        if not isinstance(hashes, dict) or set(hashes) != set(artifacts):
            return False
        try:
            actual = self._artifact_hashes(artifacts)
        except OSError:
            return False
        return hashes == actual

    def _collision_materialization_marker_is_valid(self) -> bool:
        try:
            document = json.loads(self.materialization_marker_path.read_text())
            expected = {
                "schema": "residual-kmeans-collision-materialization/v1",
                "fitter_revision": KMEANS_FITTER_REVISION,
                "materialization_revision": KMEANS_MATERIALIZATION_REVISION,
                "semantic_base_cache_key": self.config.base_cache_key,
                "semantic_cache_key": self.config.cache_key,
                "fit_materialization_sha256": self._file_hash(
                    self.fit_materialization_marker_path
                ),
                "codes_sha256": self._file_hash(self.codes_path),
            }
        except (OSError, json.JSONDecodeError):
            return False
        return document == expected

    def reuse_codebooks(
        self, item_ids: torch.Tensor, embeddings: torch.Tensor
    ) -> tuple[ResidualCodebooks, torch.Tensor]:
        if not self.config.convergence_enabled:
            return super().reuse_codebooks(item_ids, embeddings)
        codebooks = ResidualCodebooks.load(self.codebooks_path).to(self.device)
        base_codes = SemanticCodes.load(self.base_codes_path)
        if not item_ids.equal(base_codes.item_ids):
            raise RuntimeError("cached KMeans item ids do not match content embeddings")
        return codebooks, base_codes.codes

    def quantize(
        self, embeddings: torch.Tensor
    ) -> tuple[ResidualCodebooks, torch.Tensor]:
        if self.config.convergence_enabled:
            fit = fit_residual_kmeans_with_diagnostics(
                embeddings,
                num_levels=self.config.num_levels,
                num_codes=self.config.num_codes,
                max_iterations=self.config.kmeans_iterations,
                relative_inertia_tolerance=(
                    self.config.kmeans_relative_inertia_tolerance
                ),
                assignment_early_stopping=(
                    self.config.kmeans_assignment_early_stopping
                ),
                seed=self.config.seed,
            )
            document = {
                "schema": "residual-kmeans-convergence/v1",
                "fitter_revision": KMEANS_FITTER_REVISION,
                "semantic_base_cache_key": self.config.base_cache_key,
                **asdict(fit.diagnostics),
            }
            content = json.dumps(document, indent=2, sort_keys=True) + "\n"
            self.fit_diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.fit_diagnostics_path.with_suffix(".json.tmp")
            temporary.write_text(content)
            temporary.replace(self.fit_diagnostics_path)
            return fit.codebooks, fit.codes
        codebooks = fit_residual_kmeans(
            embeddings,
            num_levels=self.config.num_levels,
            num_codes=self.config.num_codes,
            num_iterations=self.config.kmeans_iterations,
            seed=self.config.seed,
        )
        return codebooks, codebooks.encode(embeddings)

    def convergence_diagnostics(self) -> tuple[dict[str, object], str] | None:
        if not self.config.convergence_enabled:
            return None
        try:
            payload = self.fit_diagnostics_path.read_bytes()
            document = json.loads(payload)
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError("invalid KMeans convergence diagnostics") from error
        expected = {
            "schema": "residual-kmeans-convergence/v1",
            "fitter_revision": KMEANS_FITTER_REVISION,
            "semantic_base_cache_key": self.config.base_cache_key,
            "max_iterations": self.config.kmeans_iterations,
            "relative_inertia_tolerance": (
                self.config.kmeans_relative_inertia_tolerance
            ),
            "assignment_early_stopping": (self.config.kmeans_assignment_early_stopping),
        }
        if not isinstance(document, dict) or any(
            document.get(name) != value for name, value in expected.items()
        ):
            raise RuntimeError("KMeans convergence diagnostics do not match config")
        levels = document.get("levels")
        if not isinstance(levels, list) or len(levels) != self.config.num_levels:
            raise RuntimeError("KMeans convergence levels do not match config")
        return document, hashlib.sha256(payload).hexdigest()


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
            codebooks_path=self._semantic_base_dir / "codebooks.pt",
            fit_diagnostics_path=(
                self._semantic_base_dir / "kmeans_fit_diagnostics.json"
            ),
            config=self.semantic,
            device=self.device,
            invalidate_cache=self.invalidate_cache,
        )

    @property
    def _semantic_dir(self) -> Path:
        return self.dataset_cache_dir / "semantic" / self.semantic.cache_key

    @property
    def _semantic_base_dir(self) -> Path:
        if self.semantic.quantizer == "kmeans":
            cache_key = self.semantic.base_cache_key
        else:
            cache_key = self.semantic.cache_key
        return self.dataset_cache_dir / "semantic" / cache_key

    @cached_property
    def semantic_codes(self) -> SemanticCodes:
        return SemanticCodes.load(self.semantic_stage.codes_path)

    @cached_property
    def semantic_codebooks(self) -> ResidualCodebooks:
        return ResidualCodebooks.load(self.semantic_stage.codebooks_path)
