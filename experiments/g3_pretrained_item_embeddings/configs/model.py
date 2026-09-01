from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from functools import cached_property
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

import torch
from torch import nn

from dcn.models import SequenceRetrievalModel, TwoTowerLoss
from dcn.config.settings import (
    DataloaderConfig,
    LrScheduleConfig,
    RuntimeConfig,
    TransformerConfig,
)
from dcn.nn import (
    ContentProjection,
    FrequencyContentGate,
    GlobalContentGate,
    ItemContentCatalogEncoder,
    ItemContentDenseNetEncoder,
    ItemMetadataDenseNetEncoder,
    ItemMetadataEmbedding,
    PrecomputedEmbeddingLookup,
    PretrainedCatalogEncoder,
    SafeItemEmbedding,
)
from dcn.nn.densenet import DenseNet
from dcn.nn.types import ModuleWithDim
from experiments.g3_pretrained_item_embeddings.data import (
    LoadedFeatureData,
    load_feature_data,
)
from experiments.g3_pretrained_item_embeddings.diagnostics import (
    G3DiagnosticsCallback,
    G3GateDiagnosticsCallback,
    G3DiagnosticTwoTowerLoss,
)
from experiments.g3_pretrained_item_embeddings.protocol.constants import (
    APPROVED_PROTOCOL,
    APPROVED_PROTOCOL_SHA256,
)
from experiments.g4_future_items.configs.control import (
    G4GenerationExperiment,
    build_control,
)
from experiments.generation_protocol import generation_protocol
from utils.global_config import config as global_config


HistoryRepresentation = Literal[
    "learned_id",
    "content",
    "id_content",
    "id_content_zero_id",
    "id_only_densenet",
]
CatalogRepresentation = Literal[
    "learned_id",
    "frozen_content",
    "trainable_content",
    "id_frozen_content",
    "id_trainable_content",
]
ContentGate = Literal["fixed", "global", "frequency"]
FrequencyGateSemantics = Literal["fp32_p09_v2"]
MetadataFeature = Literal["artist", "album"]
ItemIdTying = Literal["tied", "untied"]

RQ3_CATALOG_REPRESENTATIONS: dict[str, CatalogRepresentation] = {
    "rq3_output_learned": "learned_id",
    "rq3_output_frozen_content": "frozen_content",
    "rq3_output_trainable_content": "trainable_content",
    "rq3_output_learned_frozen_content": "id_frozen_content",
    "rq3_output_learned_trainable_content": "id_trainable_content",
}

_RQ2_WIDTHS = {32, 64, 128, 256, 512}
_RQ4_WIDTHS = {16, 32, 64, 128}
_RQ5_WIDTHS = {16, 32, 64, 96, 128}


@dataclass(frozen=True)
class G3Representation:
    history_representation: HistoryRepresentation = "learned_id"
    catalog_representation: CatalogRepresentation = "learned_id"
    history_hidden_dim: int | None = None
    content_gate: ContentGate = "fixed"
    gate_hidden_dim: int | None = None
    frequency_gate_semantics: FrequencyGateSemantics | None = None
    metadata: tuple[MetadataFeature, ...] = ()
    metadata_dim: int | None = None
    extra_item_id_dim: int | None = None
    item_id_tying: ItemIdTying | None = None

    def __post_init__(self) -> None:
        if self.history_representation not in {
            "learned_id",
            "content",
            "id_content",
            "id_content_zero_id",
            "id_only_densenet",
        }:
            raise ValueError("unknown history representation")
        if self.catalog_representation not in {
            "learned_id",
            "frozen_content",
            "trainable_content",
            "id_frozen_content",
            "id_trainable_content",
        }:
            raise ValueError("unknown catalog representation")
        if self.content_gate not in {"fixed", "global", "frequency"}:
            raise ValueError("unknown content gate")
        if self.item_id_tying not in {None, "tied", "untied"}:
            raise ValueError("unknown item-ID tying")
        if self.item_id_tying == "tied" and (
            self.history_representation not in {"learned_id", "id_content"}
            or self.catalog_representation != "learned_id"
        ):
            raise ValueError(
                "item-ID tying requires learned-ID or ID/content history and a "
                "learned-ID catalog"
            )
        if self.history_representation in {
            "id_content",
            "id_content_zero_id",
            "id_only_densenet",
        }:
            _positive_integer("history_hidden_dim", self.history_hidden_dim)
        elif self.history_hidden_dim is not None:
            raise ValueError("history_hidden_dim requires a DenseNet history encoder")
        if self.content_gate != "fixed" and self.history_representation != "id_content":
            raise ValueError("content gates require id_content history input")
        if self.content_gate == "frequency":
            _positive_integer("gate_hidden_dim", self.gate_hidden_dim)
            if self.frequency_gate_semantics not in {None, "fp32_p09_v2"}:
                raise ValueError("unknown frequency gate semantics")
        elif self.gate_hidden_dim is not None:
            raise ValueError("gate_hidden_dim requires the frequency gate")
        elif self.frequency_gate_semantics is not None:
            raise ValueError("frequency gate semantics require the frequency gate")
        if len(set(self.metadata)) != len(self.metadata):
            raise ValueError("metadata features must be unique")
        if any(value not in {"artist", "album"} for value in self.metadata):
            raise ValueError("unknown metadata feature")
        if self.metadata:
            _positive_integer("metadata_dim", self.metadata_dim)
        elif self.metadata_dim is not None:
            raise ValueError("metadata_dim requires metadata features")
        if self.extra_item_id_dim is not None:
            _positive_integer("extra_item_id_dim", self.extra_item_id_dim)
            if self.metadata:
                raise ValueError("extra item-ID capacity control excludes metadata")

    @property
    def needs_feature_data(self) -> bool:
        return (
            bool(self.metadata)
            or self.extra_item_id_dim is not None
            or (self.content_gate == "frequency")
        )

    def to_dict(self) -> dict[str, object]:
        representation = {
            "history_representation": self.history_representation,
            "catalog_representation": self.catalog_representation,
            "history_hidden_dim": self.history_hidden_dim,
            "content_gate": self.content_gate,
            "gate_hidden_dim": self.gate_hidden_dim,
            "metadata": list(self.metadata),
            "metadata_dim": self.metadata_dim,
            "extra_item_id_dim": self.extra_item_id_dim,
        }
        if self.frequency_gate_semantics is not None:
            representation["frequency_gate_semantics"] = self.frequency_gate_semantics
        if self.item_id_tying is not None:
            representation["item_id_tying"] = self.item_id_tying
        return representation

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> G3Representation:
        if not isinstance(value, Mapping):
            raise ValueError("representation payload must be an object")
        allowed = {field.name for field in fields(cls)}
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unknown representation fields: {sorted(unknown)}")
        required = {
            "history_representation",
            "catalog_representation",
            "history_hidden_dim",
            "content_gate",
            "gate_hidden_dim",
            "metadata",
            "metadata_dim",
            "extra_item_id_dim",
        }
        missing = required - set(value)
        if missing:
            raise ValueError(f"missing representation fields: {sorted(missing)}")
        metadata = value["metadata"]
        if not isinstance(metadata, list) or any(
            not isinstance(name, str) for name in metadata
        ):
            raise ValueError("representation metadata must be a JSON string list")
        arguments = dict(value)
        arguments["metadata"] = tuple(metadata)
        return cls(**arguments)


class _ItemIdDenseNetEncoder(ModuleWithDim):
    def __init__(
        self,
        num_items: int,
        item_dim: int,
        output_dim: int,
        hidden_dim: int,
        *,
        mask_unknown_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_items = num_items
        self.item_embedding = nn.Embedding(num_items + 1, item_dim, padding_idx=0)
        self.encoder = DenseNet(item_dim, output_dim, hidden_dim=hidden_dim)
        self.mask_unknown_output = mask_unknown_output

    @property
    def out_dim(self) -> int:
        return self.encoder.out_dim

    def forward(self, item_ids: torch.Tensor) -> torch.Tensor:
        known = (item_ids >= 1) & (item_ids <= self.num_items)
        safe = torch.where(
            known,
            item_ids,
            torch.zeros_like(item_ids),
        )
        embedded = self.item_embedding(safe)
        if self.mask_unknown_output:
            embedded = embedded * known.unsqueeze(-1)
        output = self.encoder(embedded)
        return output * known.unsqueeze(-1) if self.mask_unknown_output else output


@dataclass
class G3GenerationExperiment(G4GenerationExperiment):
    representation: G3Representation = G3Representation()
    feature_data_path: Path | None = None
    g3_dataset_size: Literal["native-50m", "native-500m"] = "native-50m"
    final_ranking_evidence_group: str | None = "g3-native50m"
    gate_mechanism_diagnostics: bool = False
    g3_execution_identity: dict[str, object] | None = None
    g3_evaluation_population: dict[str, object] | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        expected_size = {
            "native-50m": "50m",
            "native-500m": "500m",
        }[self.g3_dataset_size]
        if self.size != expected_size:
            raise ValueError("G3 dataset label differs from the configured data size")
        if self.dataloader.batch_size != APPROVED_PROTOCOL.batch_size:
            raise ValueError("G3 uses the approved batch size 512")
        if self.feature_data_path is None:
            raise ValueError("G3 requires the approved training-only feature data")
        if self.g3_dataset_size == "native-500m":
            if self.representation.extra_item_id_dim is not None:
                raise ValueError("native-500M G3 does not declare extra_item_id_dim")
            history_has_item_ids = self.representation.history_representation in {
                "learned_id",
                "id_content",
                "id_content_zero_id",
                "id_only_densenet",
            }
            catalog_has_item_ids = self.representation.catalog_representation in {
                "learned_id",
                "id_frozen_content",
                "id_trainable_content",
            }
            if (
                history_has_item_ids
                and catalog_has_item_ids
                and self.representation.item_id_tying is None
            ):
                raise ValueError(
                    "native-500M G3 item-ID tying must be declared explicitly"
                )

    @cached_property
    def item_embeddings(self) -> PrecomputedEmbeddingLookup:
        path = self.artifacts.precomputed_embeddings[self.artifacts.item_id_column]
        if _sha256(path) != APPROVED_PROTOCOL.content_hash(self.g3_dataset_size):
            raise ValueError("compact content table differs from the approved hash")
        content = PrecomputedEmbeddingLookup.from_parquet(
            path,
            learnable_default=False,
            strict=False,
        )
        if content.out_dim != 128:
            raise ValueError("G3 content embeddings must have width 128")
        table = content.embedding.weight
        if not bool(torch.isfinite(table).all()):
            raise ValueError("G3 content embeddings must be finite")
        if not torch.allclose(
            table.norm(dim=-1),
            torch.ones(table.shape[0], dtype=table.dtype),
            atol=2e-5,
            rtol=2e-5,
        ):
            raise ValueError("G3 content embeddings must be unit-normalized")
        return content

    @cached_property
    def g3_feature_data(self) -> LoadedFeatureData:
        if self.feature_data_path is None:
            raise ValueError("feature_data_path is not configured")
        data = load_feature_data(self.feature_data_path)
        if len(data.training_counts) != self.catalog_size:
            raise ValueError("G3 feature data differs from the item catalog")
        return data

    @cached_property
    def item_embedding(self) -> nn.Module:
        encoder = _build_isolated(self._history_encoder)
        if self.representation.metadata or self.representation.extra_item_id_dim:
            encoder = _build_isolated(lambda: self._with_metadata(encoder))
        return encoder

    @cached_property
    def catalog_item_encoder(self) -> nn.Module:
        if self._shares_complete_encoder:
            return self.item_embedding
        encoder = _build_isolated(self._catalog_encoder)
        if self.representation.metadata or self.representation.extra_item_id_dim:
            encoder = _build_isolated(lambda: self._with_metadata(encoder))
        return encoder

    @cached_property
    def shared_item_id_embedding(self) -> nn.Embedding:
        if self.representation.item_id_tying != "tied":
            raise ValueError("shared item table requires tied item-ID semantics")
        embedding = _build_isolated(
            lambda: (
                SafeItemEmbedding(self.num_items, self.model_dim)
                if self.g3_dataset_size == "native-500m"
                else nn.Embedding(
                    self.catalog_size,
                    self.model_dim,
                    padding_idx=0,
                )
            )
        )
        if not isinstance(embedding, nn.Embedding):
            raise TypeError("shared item table must be an embedding")
        return embedding

    @property
    def _shares_complete_encoder(self) -> bool:
        return (
            self.representation.item_id_tying == "tied"
            and self.representation.history_representation == "learned_id"
            and self.representation.catalog_representation == "learned_id"
            and bool(
                self.representation.metadata
                or self.representation.extra_item_id_dim is not None
            )
        )

    @property
    def _normalize_content(self) -> bool:
        return self.g3_dataset_size == "native-500m"

    def _history_encoder(self) -> nn.Module:
        kind = self.representation.history_representation
        if kind == "learned_id":
            if self.representation.item_id_tying == "tied":
                return self.shared_item_id_embedding
            if self.g3_dataset_size == "native-500m":
                return SafeItemEmbedding(self.num_items, self.model_dim)
            return nn.Embedding(self.catalog_size, self.model_dim)
        if kind == "content":
            return ContentProjection(
                self.item_embeddings,
                self.model_dim,
                normalize_content=self._normalize_content,
            )
        hidden = self.representation.history_hidden_dim
        assert hidden is not None
        if kind == "id_only_densenet":
            return _ItemIdDenseNetEncoder(
                self.num_items,
                self.model_dim,
                self.model_dim,
                hidden,
                mask_unknown_output=self.g3_dataset_size == "native-500m",
            )
        gate: ModuleWithDim | None = None
        if self.representation.content_gate == "global":
            gate = GlobalContentGate()
        elif self.representation.content_gate == "frequency":
            assert self.representation.gate_hidden_dim is not None
            gate = FrequencyContentGate(
                self.g3_feature_data.training_counts,
                self.representation.gate_hidden_dim,
                initial_probability=(
                    0.9
                    if self.representation.frequency_gate_semantics == "fp32_p09_v2"
                    else 0.9999
                ),
                fp32_math=(
                    self.representation.frequency_gate_semantics == "fp32_p09_v2"
                ),
            )
        encoder = ItemContentDenseNetEncoder(
            self.num_items,
            self.model_dim,
            self.item_embeddings,
            self.model_dim,
            hidden,
            content_gate=gate,
            item_embedding=(
                self.shared_item_id_embedding
                if self.representation.item_id_tying == "tied"
                else None
            ),
            normalize_content=self._normalize_content,
        )
        if kind == "id_content_zero_id":
            encoder.item_embedding.weight.requires_grad_(False)
            with torch.no_grad():
                encoder.item_embedding.weight.zero_()
        return encoder

    def _create_model(self) -> SequenceRetrievalModel:
        model = super()._create_model()
        if self.g3_dataset_size == "native-500m":
            with torch.no_grad():
                for module in model.modules():
                    if (
                        isinstance(module, nn.Embedding)
                        and module.padding_idx is not None
                    ):
                        module.weight[module.padding_idx].zero_()
        elif self.representation.item_id_tying == "tied":
            with torch.no_grad():
                self.shared_item_id_embedding.weight[0].zero_()
        if self.representation.history_representation != "id_content_zero_id":
            return model
        encoder = model.item_embedding
        if not isinstance(encoder, ItemContentDenseNetEncoder):
            raise TypeError("zero-ID diagnostic requires the concat history encoder")
        with torch.no_grad():
            encoder.item_embedding.weight.zero_()
        return model

    def split_parameters(
        self, model: nn.Module, embedding_types: Sequence[type[nn.Module]]
    ) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
        embedding, deep = super().split_parameters(model, embedding_types)
        return (
            [parameter for parameter in embedding if parameter.requires_grad],
            [parameter for parameter in deep if parameter.requires_grad],
        )

    def _catalog_encoder(self) -> nn.Module:
        kind = self.representation.catalog_representation
        if kind == "learned_id":
            if self.representation.item_id_tying == "tied":
                return self.shared_item_id_embedding
            if self.g3_dataset_size == "native-500m":
                return SafeItemEmbedding(self.num_items, self.model_dim)
            return nn.Embedding(self.catalog_size, self.model_dim)
        if kind in {"frozen_content", "trainable_content"}:
            return PretrainedCatalogEncoder(
                self.item_embeddings,
                self.model_dim,
                trainable=kind == "trainable_content",
                normalize_content=self._normalize_content,
            )
        return ItemContentCatalogEncoder(
            self.num_items,
            self.model_dim,
            self.item_embeddings,
            self.model_dim,
            trainable_content=kind == "id_trainable_content",
            normalize_content=self._normalize_content,
        )

    def _with_metadata(self, item_encoder: nn.Module) -> nn.Module:
        branches = []
        if self.representation.extra_item_id_dim is not None:
            offsets = torch.arange(self.catalog_size + 1, dtype=torch.long)
            feature_ids = torch.arange(self.catalog_size, dtype=torch.long)
            branches.append(
                ItemMetadataEmbedding(
                    offsets,
                    feature_ids,
                    self.num_items,
                    self.representation.extra_item_id_dim,
                )
            )
        else:
            assert self.representation.metadata_dim is not None
            data = self.g3_feature_data
            for name in self.representation.metadata:
                rows = data.artist_rows if name == "artist" else data.album_rows
                vocab_size = (
                    data.artist_vocab_size
                    if name == "artist"
                    else data.album_vocab_size
                )
                offsets, values = _rows_to_csr(rows)
                branches.append(
                    ItemMetadataEmbedding(
                        offsets,
                        values,
                        vocab_size,
                        self.representation.metadata_dim,
                    )
                )
        return ItemMetadataDenseNetEncoder(
            item_encoder,
            branches,
            self.model_dim,
            hidden_dim=self.model_dim,
        )

    def generation_architecture_metadata(self) -> dict[str, object]:
        return {
            **super().generation_architecture_metadata(),
            "g3_protocol_sha256": APPROVED_PROTOCOL_SHA256,
            "g3_dataset_size": self.g3_dataset_size,
            "g3_representation": self.representation.to_dict(),
            "g3_execution_identity": self.g3_execution_identity,
            "g3_evaluation_population": self.g3_evaluation_population,
        }

    def _report_final_metrics(self, runner: Any) -> None:
        if self.g3_dataset_size != "native-500m":
            super()._report_final_metrics(runner)
            return
        if self.g3_execution_identity is None or self.g3_evaluation_population is None:
            raise RuntimeError("native-500M execution identity is absent")
        best_weights = self.callbacks.best_weights
        if not self.restore_best_weights or not best_weights.restore(runner.model):
            raise RuntimeError("native-500M final evaluation has no best checkpoint")
        if best_weights.best_epoch is None:
            raise RuntimeError("native-500M best checkpoint has no epoch")
        run_directory = global_config.logs_path / self.run_name
        run_directory.mkdir(parents=True, exist_ok=True)
        state = {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in runner.model.state_dict().items()
        }
        state_sha256 = _state_dict_sha256(state)
        _publish_final_evaluation_bundle(
            run_directory=run_directory,
            best_epoch=best_weights.best_epoch + 1,
            state=state,
            state_sha256=state_sha256,
            execution_identity=self.g3_execution_identity,
            evaluation_population=self.g3_evaluation_population,
            evaluate=lambda: super(G3GenerationExperiment, self)._report_final_metrics(
                runner
            ),
        )
        if _state_dict_sha256(runner.model.state_dict()) != state_sha256:
            raise RuntimeError("final evaluation changed the restored checkpoint")

    @cached_property
    def g3_diagnostic_criterion(self) -> G3DiagnosticTwoTowerLoss:
        base = super().create_criterion()
        if not isinstance(base, TwoTowerLoss):
            raise TypeError("G3 diagnostics require the two-tower criterion")
        return G3DiagnosticTwoTowerLoss(
            base.model,
            base.loss,
            targets=base.targets,
            training_counts=self.g3_feature_data.training_counts,
        )

    def create_criterion(self) -> nn.Module:
        return self.g3_diagnostic_criterion

    def extra_callbacks(self, train_days: list[int], val_days: list[int]) -> list[Any]:
        components = {
            "history_encoder": self.item_embedding,
            "catalog_encoder": self.catalog_item_encoder,
            "sequence_model": self.base_model.sequence_model,
            **self._catalog_diagnostic_components(),
        }
        if self.base_model.query_projection is not None:
            components["query_projection"] = self.base_model.query_projection
        diagnostics = G3DiagnosticsCallback(
            criterion=self.g3_diagnostic_criterion,
            catalog_encoder=self.catalog_item_encoder,
            components=components,
            run_log_directory=global_config.logs_path / self.run_name,
        )
        callbacks = [*super().extra_callbacks(train_days, val_days), diagnostics]
        history_encoder = self.item_embedding
        if (
            self.gate_mechanism_diagnostics
            and isinstance(history_encoder, ItemContentDenseNetEncoder)
            and history_encoder.content_gate is not None
        ):
            callbacks.append(
                G3GateDiagnosticsCallback(
                    gate=history_encoder.content_gate,
                    content_provider=history_encoder.content,
                    training_counts=self.g3_feature_data.training_counts,
                    run_log_directory=global_config.logs_path / self.run_name,
                )
            )
        return callbacks

    def _catalog_diagnostic_components(self) -> dict[str, nn.Module]:
        encoder = self.catalog_item_encoder
        if isinstance(encoder, PretrainedCatalogEncoder):
            return {
                "catalog_content_table": encoder.content,
                "catalog_projection": encoder.projection,
            }
        if isinstance(encoder, ItemContentCatalogEncoder):
            return {
                "catalog_item_table": encoder.item_embedding,
                "catalog_content_table": encoder.content,
                "catalog_projection": encoder.projection,
            }
        return {}


def build_g3_experiment(
    *,
    run_name: str,
    dataset_size: Literal["native-50m", "native-500m"],
    embedding_learning_rate: float,
    deep_learning_rate: float,
    lr_schedule_horizon_epochs: int,
    representation: G3Representation,
    feature_data_path: Path | None = None,
    seed: int = 42,
    gate_mechanism_diagnostics: bool = False,
) -> G3GenerationExperiment:
    source = (
        _build_native500m_control(
            run_name=run_name,
            embedding_learning_rate=embedding_learning_rate,
            deep_learning_rate=deep_learning_rate,
            lr_schedule_horizon_epochs=lr_schedule_horizon_epochs,
            seed=seed,
        )
        if dataset_size == "native-500m"
        else build_control(
            run_name=run_name,
            batch_size=APPROVED_PROTOCOL.batch_size,
            embedding_learning_rate=embedding_learning_rate,
            deep_learning_rate=deep_learning_rate,
            lr_schedule_horizon_epochs=lr_schedule_horizon_epochs,
            seed=seed,
        )
    )
    values = {
        field.name: getattr(source, field.name)
        for field in fields(source)
        if field.init
    }
    values.update(
        size={"native-50m": "50m", "native-500m": "500m"}[dataset_size],
        representation=representation,
        feature_data_path=feature_data_path,
        g3_dataset_size=dataset_size,
        final_ranking_evidence_group=(
            "g3-native500m-likes"
            if dataset_size == "native-500m"
            else dataset_size.replace("native-", "g3-native")
        ),
        gate_mechanism_diagnostics=gate_mechanism_diagnostics,
    )
    return G3GenerationExperiment(**values)


def build_native500m_job(
    job: Mapping[str, object],
    *,
    feature_data_path: Path,
) -> G3GenerationExperiment:
    required = {
        "run_name",
        "family_id",
        "batch_size",
        "seed",
        "horizon_epochs",
        "embedding_learning_rate",
        "deep_learning_rate",
        "capacity",
        "resolved_representation",
    }
    missing = required - set(job)
    if missing:
        raise ValueError(f"native-500M job is missing fields: {sorted(missing)}")
    run_name = job["run_name"]
    family_id = job["family_id"]
    if not isinstance(run_name, str) or not run_name:
        raise ValueError("run_name must be a non-empty string")
    if not isinstance(family_id, str):
        raise ValueError("family_id must be a string")
    if job["batch_size"] != 512 or isinstance(job["batch_size"], bool):
        raise ValueError("native-500M G3 jobs use batch size 512")
    seed = job["seed"]
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    horizon = job["horizon_epochs"]
    if not isinstance(horizon, int) or isinstance(horizon, bool):
        raise ValueError("horizon_epochs must be an integer")
    representation_payload = job["resolved_representation"]
    if not isinstance(representation_payload, Mapping):
        raise ValueError("resolved_representation must be an explicit object")
    representation = G3Representation.from_dict(representation_payload)
    _validate_native500m_family(
        family_id,
        job["capacity"],
        representation,
    )
    return build_g3_experiment(
        run_name=run_name,
        dataset_size="native-500m",
        embedding_learning_rate=_canonical_rate(
            "embedding_learning_rate", job["embedding_learning_rate"]
        ),
        deep_learning_rate=_canonical_rate(
            "deep_learning_rate", job["deep_learning_rate"]
        ),
        lr_schedule_horizon_epochs=horizon,
        representation=representation,
        feature_data_path=feature_data_path,
        seed=seed,
        gate_mechanism_diagnostics=family_id
        in {
            "rq5_global_gate",
            "rq5_frequency_gate",
        },
    )


def _validate_native500m_family(
    family_id: str,
    capacity: object,
    representation: G3Representation,
) -> None:
    if capacity is not None and (
        not isinstance(capacity, int) or isinstance(capacity, bool)
    ):
        raise ValueError("capacity must be an exact integer or null")
    fixed: dict[str, G3Representation] = {
        "baseline": G3Representation(item_id_tying="tied"),
        "untied_control": G3Representation(item_id_tying="untied"),
        "rq1_content_input": G3Representation(history_representation="content"),
    }
    if family_id in fixed:
        if capacity is not None or representation != fixed[family_id]:
            raise ValueError("resolved representation does not match family")
        return
    if family_id == "rq2_content_concat":
        if capacity not in _RQ2_WIDTHS or representation != G3Representation(
            history_representation="id_content",
            history_hidden_dim=capacity,
            item_id_tying="tied",
        ):
            raise ValueError("resolved representation does not match family")
        return
    if family_id in RQ3_CATALOG_REPRESENTATIONS:
        expected_catalog = RQ3_CATALOG_REPRESENTATIONS[family_id]
        if (
            capacity is not None
            or representation.history_representation != "id_content"
            or representation.history_hidden_dim not in _RQ2_WIDTHS
            or representation.catalog_representation != expected_catalog
            or representation.item_id_tying != "untied"
            or representation.content_gate != "fixed"
            or representation.metadata
            or representation.extra_item_id_dim is not None
        ):
            raise ValueError("resolved representation does not match family")
        return
    metadata = {
        "rq4_artist": ("artist",),
        "rq4_album": ("album",),
        "rq4_artist_album": ("artist", "album"),
    }
    if family_id in metadata:
        if capacity not in _RQ4_WIDTHS or representation != G3Representation(
            metadata=metadata[family_id],
            metadata_dim=capacity,
            item_id_tying="tied",
        ):
            raise ValueError("resolved representation does not match family")
        return
    if family_id == "rq5_global_gate":
        if (
            capacity is not None
            or representation.history_representation != "id_content"
            or representation.history_hidden_dim not in _RQ2_WIDTHS
            or representation.catalog_representation != "learned_id"
            or representation.item_id_tying != "tied"
            or representation.content_gate != "global"
            or representation.gate_hidden_dim is not None
            or representation.metadata
            or representation.extra_item_id_dim is not None
        ):
            raise ValueError("resolved representation does not match family")
        return
    if family_id == "rq5_frequency_gate":
        if (
            capacity not in _RQ5_WIDTHS
            or representation.history_representation != "id_content"
            or representation.history_hidden_dim not in _RQ2_WIDTHS
            or representation.catalog_representation != "learned_id"
            or representation.item_id_tying != "tied"
            or representation.content_gate != "frequency"
            or representation.gate_hidden_dim != capacity
            or representation.frequency_gate_semantics != "fp32_p09_v2"
            or representation.metadata
            or representation.extra_item_id_dim is not None
        ):
            raise ValueError("resolved representation does not match family")
        return
    if family_id in {"bridge_rq3_output", "bridge_rq4_metadata", "aggregate"}:
        if capacity is not None:
            raise ValueError("conditional families do not have a capacity coordinate")
        _validate_conditional_representation(family_id, representation)
        return
    raise ValueError(f"unknown native-500M G3 family {family_id!r}")


def _validate_conditional_representation(
    family_id: str,
    representation: G3Representation,
) -> None:
    if (
        representation.history_representation
        not in {"learned_id", "content", "id_content"}
        or representation.catalog_representation
        not in {
            "learned_id",
            "frozen_content",
            "trainable_content",
            "id_frozen_content",
            "id_trainable_content",
        }
        or representation.extra_item_id_dim is not None
    ):
        raise ValueError("resolved representation is outside the composition envelope")
    if representation.history_representation == "id_content":
        if representation.history_hidden_dim not in _RQ2_WIDTHS:
            raise ValueError(
                "resolved representation is outside the composition envelope"
            )
    elif representation.history_hidden_dim is not None:
        raise ValueError("resolved representation is outside the composition envelope")
    if representation.content_gate == "frequency":
        if (
            representation.gate_hidden_dim not in _RQ5_WIDTHS
            or representation.frequency_gate_semantics != "fp32_p09_v2"
        ):
            raise ValueError(
                "resolved representation is outside the composition envelope"
            )
    elif (
        representation.gate_hidden_dim is not None
        or representation.frequency_gate_semantics is not None
    ):
        raise ValueError("resolved representation is outside the composition envelope")
    if representation.metadata:
        if (
            representation.metadata
            not in {("artist",), ("album",), ("artist", "album")}
            or representation.metadata_dim not in _RQ4_WIDTHS
        ):
            raise ValueError(
                "resolved representation is outside the composition envelope"
            )
    elif representation.metadata_dim is not None:
        raise ValueError("resolved representation is outside the composition envelope")
    if representation.item_id_tying == "tied":
        if (
            representation.history_representation not in {"learned_id", "id_content"}
            or representation.catalog_representation != "learned_id"
        ):
            raise ValueError(
                "resolved representation is outside the composition envelope"
            )
    elif (
        representation.history_representation == "learned_id"
        and representation.catalog_representation == "learned_id"
    ):
        raise ValueError("untied learned-ID control cannot enter a composition")
    if (
        representation.history_representation == "id_content"
        and representation.catalog_representation == "learned_id"
        and representation.item_id_tying != "tied"
    ):
        raise ValueError("RQ2-derived composition must preserve item-ID tying")
    if (
        representation.history_representation == "id_content"
        and representation.catalog_representation != "learned_id"
        and representation.item_id_tying != "untied"
    ):
        raise ValueError("RQ3 target composition requires independent item-ID tables")
    if family_id == "bridge_rq3_output" and (
        representation.catalog_representation == "learned_id" or representation.metadata
    ):
        raise ValueError("resolved representation does not match RQ3 bridge")
    if family_id == "bridge_rq4_metadata" and not representation.metadata:
        raise ValueError("resolved representation does not match RQ4 bridge")
    if family_id == "aggregate" and representation == G3Representation(
        item_id_tying="tied"
    ):
        raise ValueError("baseline-only aggregate must reuse the baseline artifact")


def _build_native500m_control(
    *,
    run_name: str,
    embedding_learning_rate: float,
    deep_learning_rate: float,
    lr_schedule_horizon_epochs: int,
    seed: int,
) -> G4GenerationExperiment:
    transformer = TransformerConfig(
        dim=64,
        num_layers=2,
        nhead=2,
        num_kv_heads=1,
        ffn_intermediate_dim=192,
        dropout=0.1,
        input_dropout=0.1,
        ffn_dropout=0.1,
        gated_ffn_dropout=True,
        ffn="swiglu",
        norm="layer",
        norm_place="post",
        input_norm="rms",
        final_norm="rms",
        alibi=True,
        rope="timestamp_reverse",
        learned_positions=("forward", "reverse"),
        learned_position_fusion="concat",
        learned_position_fusion_residual="rezero",
        learned_position_reverse_correction="bounded_tanh",
        learned_position_reverse_max_scale=0.025,
        learned_position_reverse_initializer_rng_nonadvancing=True,
        attention_window=None,
    )
    horizon = _positive_integer_value(
        "lr_schedule_horizon_epochs", lr_schedule_horizon_epochs
    )
    return G4GenerationExperiment(
        run_name=run_name,
        seed=seed,
        **generation_protocol(
            event_type_filter="like",
            window="next_item",
            size="500m",
        ),
        dataloader=DataloaderConfig(
            batch_size=512,
            val_batch_size=8192,
            num_workers=4,
            prefetch_factor=4,
        ),
        runtime=RuntimeConfig(
            dtype=torch.bfloat16,
            compile=False,
            gradient_clip_norm=None,
        ),
        num_epochs=horizon,
        lr_schedule_horizon_epochs=horizon,
        eval_every_n_epochs=1,
        restore_best_weights=True,
        early_stopping_patience=3,
        early_stopping_min_delta=0.0,
        adaptive_schedule_early_stopping=False,
        transformer=transformer,
        max_seq_len=100,
        bos=True,
        cls_token=False,
        cls_token_mode="end_only",
        lr_schedule=LrScheduleConfig(
            "cosine",
            warmup_fraction=0.05,
            cycles=1,
            optimizer_group_scope="deep_only",
        ),
        timestamp_delta="bins",
        timestamp_combination="add",
        timestamp_num_bins=32,
        negative_sampling="random_offline_logq",
        logq_correction="yi2019",
        correct_positive_logq=True,
        mask_false_negatives=False,
        exclude_own_group_negatives=False,
        num_in_batch_negatives=2048,
        dense_random_negative_scores=True,
        initializer_std=0.02,
        item_embedding_dim=64,
        embedding_learning_rate=_positive_rate(
            "embedding_learning_rate", embedding_learning_rate
        ),
        deep_learning_rate=_positive_rate("deep_learning_rate", deep_learning_rate),
        weight_decay=0.0,
    )


def build_rq3_representation(
    family_id: str,
    *,
    history_hidden_dim: int,
    item_id_tying: ItemIdTying | None = None,
) -> G3Representation:
    try:
        catalog_representation = RQ3_CATALOG_REPRESENTATIONS[family_id]
    except KeyError as error:
        raise ValueError(f"unknown RQ3 output family {family_id!r}") from error
    return G3Representation(
        history_representation="id_content",
        history_hidden_dim=history_hidden_dim,
        catalog_representation=catalog_representation,
        item_id_tying=item_id_tying,
    )


def _build_isolated(factory: Callable[[], nn.Module]) -> nn.Module:
    with torch.random.fork_rng(devices=[]):
        module = factory()
    module.initializer_rng_nonadvancing = True
    return module


def _rows_to_csr(
    rows: tuple[tuple[int, ...], ...]
) -> tuple[torch.Tensor, torch.Tensor]:
    offsets = [0]
    values = []
    for row in rows:
        values.extend(row)
        offsets.append(len(values))
    return torch.tensor(offsets), torch.tensor(values, dtype=torch.long)


def _positive_integer(name: str, value: int | None) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _positive_integer_value(name: str, value: int) -> int:
    _positive_integer(name, value)
    return value


def _positive_rate(name: str, value: float) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive finite number")
    return float(value)


def _state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _write_checkpoint(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        existing = torch.load(path, map_location="cpu", weights_only=True)
        if (
            not isinstance(existing, dict)
            or existing.get("schema_version") != 1
            or existing.get("best_epoch") != payload["best_epoch"]
            or existing.get("state_sha256") != payload["state_sha256"]
            or not isinstance(existing.get("state_dict"), dict)
            or _state_dict_sha256(existing["state_dict"]) != payload["state_sha256"]
        ):
            raise RuntimeError(f"restored checkpoint changed: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    try:
        try:
            os.link(temporary, path)
        except FileExistsError:
            _write_checkpoint(path, payload)
    finally:
        temporary.unlink(missing_ok=True)


def _publish_final_evaluation_bundle(
    *,
    run_directory: Path,
    best_epoch: int,
    state: Mapping[str, torch.Tensor],
    state_sha256: str,
    execution_identity: Mapping[str, object],
    evaluation_population: Mapping[str, object],
    evaluate: Callable[[], None],
) -> None:
    from experiments.g3_pretrained_item_embeddings.launchers.native500m import (
        validate_current_source_ledger,
    )

    lock_path = run_directory / ".final_evaluation.lock"
    lock_fd = os.open(
        lock_path,
        os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        validate_current_source_ledger(execution_identity)
        proof_path = run_directory / "final_evaluation_proof.json"
        expected = {
            "best_epoch": best_epoch,
            "checkpoint_state_sha256": state_sha256,
            "execution_identity_sha256": str(execution_identity["sha256"]),
            "evaluation_population": dict(evaluation_population),
        }
        if proof_path.exists():
            _validate_final_evaluation_bundle(run_directory, expected)
            return
        checkpoint_path = run_directory / "restored_best_checkpoint.pt"
        _write_checkpoint(
            checkpoint_path,
            {
                "schema_version": 1,
                "best_epoch": best_epoch,
                "state_sha256": state_sha256,
                "state_dict": dict(state),
            },
        )
        evaluate()
        validate_current_source_ledger(execution_identity)
        proof_body = {
            "schema_version": 1,
            **expected,
            "checkpoint": _artifact_identity(checkpoint_path),
            "final_metrics": _artifact_identity(run_directory / "final_metrics.json"),
            "ranking_evidence": _artifact_identity(
                run_directory / "ranking_evidence.pt"
            ),
            "top_item_rankings": _artifact_identity(
                run_directory / "top_item_rankings.json"
            ),
        }
        _write_json_immutable(
            proof_path,
            {**proof_body, "sha256": _canonical_sha256(proof_body)},
        )
    finally:
        os.close(lock_fd)


def _validate_final_evaluation_bundle(
    run_directory: Path,
    expected: Mapping[str, object],
) -> None:
    proof_path = run_directory / "final_evaluation_proof.json"
    try:
        proof = json.loads(proof_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("existing final evaluation proof is invalid") from error
    body = {key: value for key, value in proof.items() if key != "sha256"}
    required = {
        "schema_version",
        "best_epoch",
        "checkpoint",
        "checkpoint_state_sha256",
        "execution_identity_sha256",
        "evaluation_population",
        "final_metrics",
        "ranking_evidence",
        "top_item_rankings",
    }
    if (
        set(body) != required
        or body.get("schema_version") != 1
        or proof.get("sha256") != _canonical_sha256(body)
        or any(body.get(key) != value for key, value in expected.items())
    ):
        raise RuntimeError("existing final evaluation proof differs")
    artifact_names = {
        "checkpoint": "restored_best_checkpoint.pt",
        "final_metrics": "final_metrics.json",
        "ranking_evidence": "ranking_evidence.pt",
        "top_item_rankings": "top_item_rankings.json",
    }
    for key, filename in artifact_names.items():
        identity = body[key]
        if (
            not isinstance(identity, dict)
            or set(identity)
            != {
                "path",
                "size_bytes",
                "sha256",
            }
            or identity.get("path") != filename
        ):
            raise RuntimeError("existing final evaluation artifact identity differs")
        path = run_directory / filename
        if _artifact_identity(path) != identity:
            raise RuntimeError("existing final evaluation artifact differs")
    checkpoint = torch.load(
        run_directory / str(body["checkpoint"]["path"]),
        map_location="cpu",
        weights_only=True,
    )
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("schema_version") != 1
        or checkpoint.get("best_epoch") != body["best_epoch"]
        or checkpoint.get("state_sha256") != body["checkpoint_state_sha256"]
        or not isinstance(checkpoint.get("state_dict"), dict)
        or _state_dict_sha256(checkpoint["state_dict"])
        != body["checkpoint_state_sha256"]
    ):
        raise RuntimeError("existing restored checkpoint differs")


def _artifact_identity(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"final evaluation artifact is absent: {path}")
    return {
        "path": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _write_json_immutable(path: Path, value: dict[str, object]) -> None:
    content = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"final evaluation proof changed: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(content)
    try:
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != content:
                raise RuntimeError(f"final evaluation proof changed: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def _canonical_rate(name: str, value: object) -> float:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a canonical float64 string")
    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a canonical float64 string") from error
    if format(parsed, ".17g") != value:
        raise ValueError(f"{name} must be serialized with format(value, '.17g')")
    return _positive_rate(name, parsed)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
