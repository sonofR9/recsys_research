from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import fcntl
import json
import math
from pathlib import Path
from typing import Any, Literal, get_args

import torch

from dcn.config.generation import MuTransferGenerationExperiment
from dcn.config.semantic import KMeansIdStage, SemanticExperiment
from dcn.models import EventTokenizer, SemanticHistoryTokenizer, TimestampDeltaTokenizer
from dcn.nn import ConcatenatedItemFeatureResidual
from dcn.nn.semantic_embedding import (
    CombinedSemanticIdEmbedding,
    SemanticIdEmbedding,
)
from dcn.semantic import SemanticIdDiagnostics, semantic_id_diagnostics
from dcn.semantic.artifacts import load_item_embeddings
from neuralrec.run.train import TrainRunner

SemanticHistoryRepresentation = Literal[
    "learned_sid_event",
    "item_frozen_sid_event",
    "item_learned_frozen_sid_event",
    "item_frozen_sid_learned_residual_event",
    "learned_sid_tokens",
    "learned_frozen_sid_tokens",
    "frozen_sid_tokens",
    "interleaved_item_sid_tokens",
]


@dataclass
class SemanticHistoryExperiment(SemanticExperiment, MuTransferGenerationExperiment):
    run_name: str = "rqkmeans_semantic_history"
    history_representation: SemanticHistoryRepresentation = "learned_sid_event"
    representation_width: int = 64
    frozen_event_width: int = 128
    learned_residual_max_scale: float | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.history_representation not in get_args(SemanticHistoryRepresentation):
            raise ValueError(
                "unknown semantic history representation "
                f"{self.history_representation!r}"
            )
        if (
            not isinstance(self.representation_width, int)
            or isinstance(self.representation_width, bool)
            or self.representation_width < 1
        ):
            raise ValueError("representation_width must be a positive integer")
        if (
            not isinstance(self.frozen_event_width, int)
            or isinstance(self.frozen_event_width, bool)
            or self.frozen_event_width < 1
        ):
            raise ValueError("frozen_event_width must be a positive integer")
        if self.learned_residual_max_scale is not None:
            if self.history_representation != "item_frozen_sid_learned_residual_event":
                raise ValueError(
                    "bounded learned residual requires the learned residual representation"
                )
            if (
                not isinstance(self.learned_residual_max_scale, (int, float))
                or isinstance(self.learned_residual_max_scale, bool)
                or not math.isfinite(self.learned_residual_max_scale)
                or self.learned_residual_max_scale < 0
            ):
                raise ValueError(
                    "learned_residual_max_scale must be nonnegative finite"
                )

    @property
    def history_tokens_per_event(self) -> int:
        if self.history_representation.endswith("_event"):
            return 1
        semantic_levels = self.semantic.num_levels + int(
            self.semantic.collision_policy == "suffix"
        )
        return semantic_levels + int(
            self.history_representation == "interleaved_item_sid_tokens"
        )

    def training_count_architecture_invariants(self) -> tuple[object, ...]:
        invariants = (
            *super().training_count_architecture_invariants(),
            self.history_representation,
            self.semantic.cache_key,
        )
        if self.history_representation == "item_frozen_sid_learned_residual_event":
            return (
                *invariants,
                self.representation_width,
                self.frozen_event_width,
                *(
                    ()
                    if self.learned_residual_max_scale is None
                    else (self.learned_residual_max_scale,)
                ),
            )
        return invariants

    def _learned_embedding(self) -> SemanticIdEmbedding:
        return SemanticIdEmbedding.learned(
            self.semantic_codes,
            num_items=self.num_items,
            embedding_dim=self.representation_width,
        )

    def _frozen_embedding(self) -> SemanticIdEmbedding:
        return SemanticIdEmbedding.from_codebooks(
            self.semantic_codes,
            self.semantic_codebooks,
            num_items=self.num_items,
            train_collision_suffix=self.semantic.collision_policy == "suffix",
        )

    def _combined_embedding(self) -> CombinedSemanticIdEmbedding:
        return CombinedSemanticIdEmbedding(
            [self._learned_embedding(), self._frozen_embedding()]
        )

    def create_tokenizer(self) -> EventTokenizer:
        name = self.history_representation
        common = {
            "item_id_column": self.item_id_column,
            "model_dim": self.model_dim,
        }
        if name == "learned_sid_event":
            tokenizer = SemanticHistoryTokenizer.learned_sid_event(
                self._learned_embedding(),
                encoder_hidden_dim=self.representation_width,
                **common,
            )
        elif name == "item_frozen_sid_event":
            tokenizer = SemanticHistoryTokenizer.item_frozen_sid_event(
                self.item_embedding,
                self._frozen_embedding(),
                encoder_hidden_dim=self.representation_width,
                **common,
            )
        elif name == "item_learned_frozen_sid_event":
            tokenizer = SemanticHistoryTokenizer.item_learned_frozen_sid_event(
                self.item_embedding,
                self._combined_embedding(),
                encoder_hidden_dim=self.representation_width,
                **common,
            )
        elif name == "item_frozen_sid_learned_residual_event":
            frozen_embedding = self._frozen_embedding()
            with torch.random.fork_rng(devices=[]):
                learned_embedding = self._learned_embedding()
            tokenizer = SemanticHistoryTokenizer.item_frozen_sid_learned_residual_event(
                self.item_embedding,
                frozen_embedding,
                learned_embedding,
                encoder_hidden_dim=self.frozen_event_width,
                residual_max_scale=self.learned_residual_max_scale,
                **common,
            )
        elif name == "learned_sid_tokens":
            tokenizer = SemanticHistoryTokenizer.learned_sid_tokens(
                self._learned_embedding(), **common
            )
        elif name == "learned_frozen_sid_tokens":
            tokenizer = SemanticHistoryTokenizer.learned_frozen_sid_tokens(
                self._combined_embedding(),
                encoder_hidden_dim=self.representation_width,
                **common,
            )
        elif name == "frozen_sid_tokens":
            tokenizer = SemanticHistoryTokenizer.frozen_sid_tokens(
                self._frozen_embedding(),
                encoder_hidden_dim=self.representation_width,
                **common,
            )
        else:
            tokenizer = SemanticHistoryTokenizer.interleaved_item_sid_tokens(
                self.item_embedding,
                self._learned_embedding(),
                **common,
            )
        if self.timestamp_delta is not None:
            tokenizer = TimestampDeltaTokenizer(
                tokenizer,
                kind=self.timestamp_delta,
                combination=self.timestamp_combination,
                num_bins=self.timestamp_num_bins,
            )
        return self._with_bos(tokenizer)

    def true_metric_options(self) -> dict[str, Any]:
        return {
            **super().true_metric_options(),
            "semantic_codes": self.semantic_codes,
            "semantic_base_levels": self.semantic.num_levels,
        }

    def generation_architecture_metadata(self) -> dict[str, object]:
        metadata = super().generation_architecture_metadata()
        if self.learned_residual_max_scale is not None:
            metadata |= {
                "history_representation": self.history_representation,
                "representation_width": self.representation_width,
                "frozen_event_width": self.frozen_event_width,
                "learned_residual_max_scale": self.learned_residual_max_scale,
            }
        return metadata

    def finish(self, runner: TrainRunner) -> None:
        super().finish(runner)
        document = self.semantic_diagnostics_document()
        content = json.dumps(document, indent=2, sort_keys=True) + "\n"
        destination = (
            Path(self.base_path)
            / "logs"
            / self.run_name
            / "semantic_id_diagnostics.json"
        )
        if destination.exists():
            if destination.read_text() != content:
                raise RuntimeError(f"semantic diagnostics changed: {destination}")
            return
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(content)
        temporary.replace(destination)

    def semantic_diagnostics_document(self) -> dict[str, object]:
        item_ids, embeddings = load_item_embeddings(
            Path(self.artifacts.precomputed_embeddings[self.item_id_column])
        )
        if not item_ids.equal(self.semantic_codes.item_ids):
            raise RuntimeError("semantic codes and content embeddings are misaligned")
        diagnostics = self._cached_semantic_diagnostics(embeddings)
        document = {
            "semantic_cache_key": self.semantic.cache_key,
            "num_levels": self.semantic.num_levels,
            "shared_num_codes": self.semantic.num_codes,
            "semantic_content_width": self.semantic_codebooks.dim,
            "collision_policy": self.semantic.collision_policy,
            "collision_suffix_symbols": (
                self.semantic_codes.codes_per_level[-1]
                if self.semantic.collision_policy == "suffix"
                else 0
            ),
            **diagnostics,
        }
        if isinstance(self.semantic_stage, KMeansIdStage):
            convergence = self.semantic_stage.convergence_diagnostics()
            if convergence is not None:
                convergence_document, convergence_sha256 = convergence
                document |= {
                    "kmeans_convergence": convergence_document,
                    "kmeans_convergence_sha256": convergence_sha256,
                }
        if self.learned_residual_max_scale is not None:
            residuals = [
                module
                for module in self.base_model.modules()
                if isinstance(module, ConcatenatedItemFeatureResidual)
                and module.max_scale is not None
            ]
            if len(residuals) != 1:
                raise RuntimeError("bounded learned residual module is not unique")
            residual = residuals[0]
            document |= {
                "learned_residual_raw_scale": residual.residual_scale.item(),
                "learned_residual_effective_scale": (
                    residual.effective_residual_scale().item()
                ),
            }
        return document

    def _cached_semantic_diagnostics(
        self, embeddings: torch.Tensor
    ) -> dict[str, object]:
        cache_path = self._semantic_base_dir / "semantic_id_diagnostics_v2.json"
        lock_path = self._semantic_base_dir / ".materialization.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if cache_path.is_file() and not self.invalidate_cache:
                cached = json.loads(cache_path.read_text())
                if isinstance(cached, dict) and set(cached) == {
                    field.name for field in fields(SemanticIdDiagnostics)
                }:
                    return cached
                if (
                    not isinstance(cached, dict)
                    or set(cached) != {"schema", "diagnostics"}
                    or cached["schema"] != "semantic-id-diagnostics/v2"
                    or not isinstance(cached["diagnostics"], dict)
                ):
                    raise RuntimeError(f"invalid semantic diagnostics: {cache_path}")
                return cached["diagnostics"]
            diagnostics = asdict(
                semantic_id_diagnostics(
                    self.semantic_codes,
                    embeddings,
                    num_base_levels=self.semantic.num_levels,
                    codebooks=self.semantic_codebooks,
                )
            )
            content = (
                json.dumps(
                    {
                        "schema": "semantic-id-diagnostics/v2",
                        "diagnostics": diagnostics,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
            temporary = cache_path.with_suffix(".json.tmp")
            temporary.write_text(content)
            temporary.replace(cache_path)
            return diagnostics
