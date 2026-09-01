from collections.abc import Iterator, Sequence
import math

import torch
from torch import nn
import torch.nn.functional as F

from dcn.data.features import FeatureValues

from .densenet import DenseNet
from .precomputed_embeddings import PrecomputedEmbeddingLookup, segment_sum
from .types import ModuleWithDim


def _safe_ids(
    ids: torch.Tensor, num_known_ids: int
) -> tuple[torch.Tensor, torch.Tensor]:
    known = (ids >= 1) & (ids <= num_known_ids)
    return torch.where(known, ids, torch.zeros_like(ids)), known


class SafeItemEmbedding(nn.Embedding):
    def __init__(self, num_known_ids: int, embedding_dim: int) -> None:
        if num_known_ids < 1 or embedding_dim < 1:
            raise ValueError("embedding dimensions must be positive")
        super().__init__(num_known_ids + 1, embedding_dim, padding_idx=0)
        self.num_known_ids = num_known_ids

    def forward(self, item_ids: torch.Tensor) -> torch.Tensor:
        safe_ids, known = _safe_ids(item_ids, self.num_known_ids)
        return super().forward(safe_ids) * known.unsqueeze(-1)


class ContentProjection(ModuleWithDim):
    def __init__(
        self,
        content: PrecomputedEmbeddingLookup,
        output_dim: int,
        *,
        normalize_content: bool = False,
    ) -> None:
        super().__init__()
        if output_dim < 1:
            raise ValueError("output_dim must be positive")
        self.content = content
        self.normalize_content = normalize_content
        self.projection = nn.Linear(content.out_dim, output_dim, bias=False)
        self._out_dim = output_dim

    @property
    def out_dim(self) -> int:
        return self._out_dim

    def forward(self, compact_ids: torch.Tensor) -> torch.Tensor:
        content = self.content.lookup(compact_ids)
        if self.normalize_content:
            content = F.normalize(content, dim=-1)
        return self.projection(content)


class _PretrainedContentTable(ModuleWithDim):
    def __init__(
        self,
        content: PrecomputedEmbeddingLookup,
        trainable: bool,
        *,
        normalize_content: bool = False,
    ) -> None:
        super().__init__()
        table = content.dense_table()
        if trainable:
            self.embedding = nn.Embedding.from_pretrained(
                table, freeze=False, padding_idx=0
            )
            self.embedding.preserve_declared_initialization = True
        else:
            self.embedding = None
            self.register_buffer("frozen", table)
        self.num_known_ids = content.num_known_ids
        self._out_dim = content.out_dim
        self.normalize_content = normalize_content

    @property
    def out_dim(self) -> int:
        return self._out_dim

    def forward(self, compact_ids: torch.Tensor) -> torch.Tensor:
        safe_ids, known = _safe_ids(compact_ids, self.num_known_ids)
        embeddings = (
            F.embedding(safe_ids, self.frozen)
            if self.embedding is None
            else self.embedding(safe_ids)
        )
        if self.normalize_content:
            embeddings = F.normalize(embeddings, dim=-1)
        return embeddings * known.unsqueeze(-1)


class PretrainedCatalogEncoder(ModuleWithDim):
    def __init__(
        self,
        content: PrecomputedEmbeddingLookup,
        output_dim: int,
        *,
        trainable: bool,
        normalize_content: bool = False,
    ) -> None:
        super().__init__()
        if output_dim < 1:
            raise ValueError("output_dim must be positive")
        self.content = _PretrainedContentTable(
            content,
            trainable,
            normalize_content=normalize_content,
        )
        self.projection = nn.Linear(content.out_dim, output_dim, bias=False)
        self._out_dim = output_dim

    @property
    def out_dim(self) -> int:
        return self._out_dim

    def content_embeddings(self, compact_ids: torch.Tensor) -> torch.Tensor:
        return self.content(compact_ids)

    def content_parameters(self) -> Iterator[nn.Parameter]:
        return self.content.parameters()

    def forward(self, compact_ids: torch.Tensor) -> torch.Tensor:
        return self.projection(self.content_embeddings(compact_ids))


class GlobalContentGate(ModuleWithDim):
    def __init__(self, initial_probability: float = 0.9999) -> None:
        super().__init__()
        if not math.isfinite(initial_probability) or not 0 < initial_probability < 1:
            raise ValueError("initial_probability must be strictly between 0 and 1")
        initial_logit = math.log(initial_probability / (1 - initial_probability))
        self.logit = nn.Parameter(torch.tensor(initial_logit))

    @property
    def out_dim(self) -> int:
        return 1

    def forward(self, compact_ids: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logit).expand(*compact_ids.shape, 1)


class FrequencyContentGate(ModuleWithDim):
    def __init__(
        self,
        training_counts: torch.Tensor,
        hidden_dim: int,
        initial_probability: float = 0.9999,
        fp32_math: bool = False,
    ) -> None:
        super().__init__()
        if training_counts.ndim != 1 or len(training_counts) < 2:
            raise ValueError("training_counts must contain unknown 0 and known IDs")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        if not math.isfinite(initial_probability) or not 0 < initial_probability < 1:
            raise ValueError("initial_probability must be strictly between 0 and 1")
        counts = training_counts.float()
        if not bool(torch.isfinite(counts).all()) or bool((counts < 0).any()):
            raise ValueError("training_counts must be nonnegative finite")
        logged = torch.log1p(counts[1:])
        deviation = logged.std(unbiased=False)
        standardized = (
            logged - logged.mean()
            if deviation == 0
            else (logged - logged.mean()) / deviation
        )
        frequencies = torch.cat([standardized.new_zeros(1), standardized])
        self.register_buffer("standardized_log_counts", frequencies)
        self.fp32_math = fp32_math
        output = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(output.weight)
        nn.init.constant_(
            output.bias,
            math.log(initial_probability / (1 - initial_probability)),
        )
        output.preserve_declared_initialization = True
        self.network = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            output,
            nn.Sigmoid(),
        )

    @property
    def out_dim(self) -> int:
        return 1

    def forward(self, compact_ids: torch.Tensor) -> torch.Tensor:
        safe_ids, _ = _safe_ids(compact_ids, len(self.standardized_log_counts) - 1)
        frequencies = F.embedding(safe_ids, self.standardized_log_counts.unsqueeze(-1))
        if self.fp32_math:
            with torch.autocast(device_type=frequencies.device.type, enabled=False):
                return self.network(frequencies.float())
        return self.network(frequencies)


class ItemContentDenseNetEncoder(ModuleWithDim):
    def __init__(
        self,
        num_items: int,
        item_dim: int,
        content: PrecomputedEmbeddingLookup,
        output_dim: int,
        hidden_dim: int,
        *,
        content_gate: ModuleWithDim | None = None,
        item_embedding: nn.Embedding | None = None,
        normalize_content: bool = False,
    ) -> None:
        super().__init__()
        if num_items < 1 or item_dim < 1:
            raise ValueError("item catalog and embedding dimensions must be positive")
        if num_items != content.num_known_ids:
            raise ValueError(
                f"item catalog has {num_items} IDs but content has "
                f"{content.num_known_ids}"
            )
        self.num_items = num_items
        if item_embedding is not None and (
            item_embedding.num_embeddings != num_items + 1
            or item_embedding.embedding_dim != item_dim
        ):
            raise ValueError("injected item embedding has incompatible dimensions")
        self.item_embedding = (
            item_embedding
            if item_embedding is not None
            else nn.Embedding(num_items + 1, item_dim, padding_idx=0)
        )
        self.content = content
        self.content_gate = content_gate
        self.normalize_content = normalize_content
        self.encoder = DenseNet(
            item_dim + content.out_dim,
            output_dim,
            hidden_dim=hidden_dim,
        )

    @property
    def out_dim(self) -> int:
        return self.encoder.out_dim

    def composed_features(self, compact_ids: torch.Tensor) -> torch.Tensor:
        safe_ids, known = _safe_ids(compact_ids, self.num_items)
        learned = self.item_embedding(safe_ids) * known.unsqueeze(-1)
        content = self.content.lookup(compact_ids)
        if self.normalize_content:
            content = F.normalize(content, dim=-1)
        if self.content_gate is not None:
            content = self.content_gate(compact_ids) * content
        return torch.cat([learned, content], dim=-1)

    def forward(self, compact_ids: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.composed_features(compact_ids))


class ItemContentCatalogEncoder(ModuleWithDim):
    def __init__(
        self,
        num_items: int,
        item_dim: int,
        content: PrecomputedEmbeddingLookup,
        output_dim: int,
        *,
        trainable_content: bool,
        item_embedding: nn.Embedding | None = None,
        normalize_content: bool = False,
    ) -> None:
        super().__init__()
        if num_items < 1 or item_dim < 1 or output_dim < 1:
            raise ValueError("catalog encoder dimensions must be positive")
        if num_items != content.num_known_ids:
            raise ValueError(
                f"item catalog has {num_items} IDs but content has "
                f"{content.num_known_ids}"
            )
        self.num_items = num_items
        if item_embedding is not None and (
            item_embedding.num_embeddings != num_items + 1
            or item_embedding.embedding_dim != item_dim
        ):
            raise ValueError("injected item embedding has incompatible dimensions")
        self.item_embedding = (
            item_embedding
            if item_embedding is not None
            else nn.Embedding(num_items + 1, item_dim, padding_idx=0)
        )
        self.content = _PretrainedContentTable(
            content,
            trainable_content,
            normalize_content=normalize_content,
        )
        self.projection = nn.Linear(item_dim + content.out_dim, output_dim, bias=False)
        self._out_dim = output_dim

    @property
    def out_dim(self) -> int:
        return self._out_dim

    def content_embeddings(self, compact_ids: torch.Tensor) -> torch.Tensor:
        return self.content(compact_ids)

    def content_parameters(self) -> Iterator[nn.Parameter]:
        return self.content.parameters()

    def forward(self, compact_ids: torch.Tensor) -> torch.Tensor:
        safe_ids, known = _safe_ids(compact_ids, self.num_items)
        features = torch.cat(
            [
                self.item_embedding(safe_ids) * known.unsqueeze(-1),
                self.content_embeddings(compact_ids),
            ],
            dim=-1,
        )
        return self.projection(features)


class MeanPooledIdEmbedding(ModuleWithDim):
    def __init__(self, num_known_ids: int, embedding_dim: int) -> None:
        super().__init__()
        if num_known_ids < 1 or embedding_dim < 1:
            raise ValueError("embedding dimensions must be positive")
        self.num_known_ids = num_known_ids
        self.embedding = nn.Embedding(num_known_ids + 1, embedding_dim, padding_idx=0)
        self._out_dim = embedding_dim

    @property
    def out_dim(self) -> int:
        return self._out_dim

    def forward(self, ids: FeatureValues) -> torch.Tensor:
        safe_ids, known = _safe_ids(ids.values, self.num_known_ids)
        sums = segment_sum(self.embedding(safe_ids) * known.unsqueeze(-1), ids.offsets)
        counts = segment_sum(known.to(sums.dtype).unsqueeze(-1), ids.offsets).clamp_min(
            1
        )
        return sums / counts


class ItemMetadataEmbedding(ModuleWithDim):
    def __init__(
        self,
        item_offsets: torch.Tensor,
        feature_ids: torch.Tensor,
        num_known_feature_ids: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()
        if num_known_feature_ids < 1 or embedding_dim < 1:
            raise ValueError("metadata embedding dimensions must be positive")
        if item_offsets.ndim != 1 or len(item_offsets) < 3:
            raise ValueError("item offsets must contain unknown and known item rows")
        if feature_ids.ndim != 1:
            raise ValueError("feature IDs must be one-dimensional")
        if item_offsets.is_floating_point() or feature_ids.is_floating_point():
            raise ValueError("item offsets and feature IDs must be integer tensors")
        offsets = item_offsets.detach().clone().long()
        features = feature_ids.detach().clone().long()
        if offsets[0].item() != 0:
            raise ValueError("item offsets must start at zero")
        if bool((offsets[1:] < offsets[:-1]).any()):
            raise ValueError("item offsets must be nondecreasing")
        if offsets[-1].item() != len(features):
            raise ValueError("final item offset must equal the feature ID count")
        if bool((features < 0).any()) or bool((features > num_known_feature_ids).any()):
            raise ValueError(f"feature ID must be in 0..{num_known_feature_ids}")
        unknown_end = offsets[1].item()
        if bool((features[:unknown_end] != 0).any()):
            raise ValueError("unknown item 0 cannot map to known metadata")
        self.register_buffer("item_offsets", offsets)
        self.register_buffer("feature_ids", features)
        self.num_items = len(offsets) - 2
        self.embedding = nn.Embedding(
            num_known_feature_ids + 1, embedding_dim, padding_idx=0
        )
        self._out_dim = embedding_dim

    @property
    def out_dim(self) -> int:
        return self._out_dim

    def forward(self, compact_item_ids: torch.Tensor) -> torch.Tensor:
        original_shape = compact_item_ids.shape
        flat_ids = compact_item_ids.reshape(-1)
        safe_ids, _ = _safe_ids(flat_ids, self.num_items)
        starts = self.item_offsets[safe_ids]
        counts = self.item_offsets[safe_ids + 1] - starts
        item_rows = torch.repeat_interleave(
            torch.arange(len(flat_ids), device=flat_ids.device), counts
        )
        row_starts = torch.repeat_interleave(
            torch.cat([counts.new_zeros(1), counts.cumsum(0)[:-1]]), counts
        )
        ranks = torch.arange(len(item_rows), device=flat_ids.device) - row_starts
        mapped_feature_ids = self.feature_ids[starts[item_rows] + ranks]
        known_features = mapped_feature_ids != 0
        values = self.embedding(mapped_feature_ids) * known_features.unsqueeze(-1)
        sums = values.new_zeros(len(flat_ids), self.out_dim)
        sums.index_add_(0, item_rows, values)
        denominators = values.new_zeros(len(flat_ids), 1)
        denominators.index_add_(
            0, item_rows, known_features.to(values.dtype).unsqueeze(-1)
        )
        pooled = sums / denominators.clamp_min(1)
        return pooled.reshape(*original_shape, self.out_dim)


def _encoder_width(encoder: nn.Module) -> int:
    for attribute in ("out_dim", "embedding_dim"):
        width = getattr(encoder, attribute, None)
        if isinstance(width, int) and width > 0:
            return width
    raise ValueError("item and metadata encoders must declare their output width")


class ItemMetadataDenseNetEncoder(ModuleWithDim):
    def __init__(
        self,
        item_encoder: nn.Module,
        metadata_branches: Sequence[ItemMetadataEmbedding],
        output_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        if not metadata_branches:
            raise ValueError("at least one metadata branch is required")
        catalog_sizes = {branch.num_items for branch in metadata_branches}
        if len(catalog_sizes) != 1:
            raise ValueError("metadata branches must share one item catalog")
        metadata_num_items = next(iter(catalog_sizes))
        item_num_items = getattr(item_encoder, "num_items", None)
        if isinstance(item_encoder, nn.Embedding):
            item_num_items = item_encoder.num_embeddings - 1
        if isinstance(item_num_items, int) and item_num_items != metadata_num_items:
            raise ValueError(
                f"item catalog has {item_num_items} IDs but metadata has "
                f"{metadata_num_items}"
            )
        self.item_encoder = item_encoder
        self.metadata_branches = nn.ModuleList(metadata_branches)
        input_dim = _encoder_width(item_encoder) + sum(
            _encoder_width(branch) for branch in metadata_branches
        )
        self.encoder = DenseNet(input_dim, output_dim, hidden_dim=hidden_dim)

    @property
    def out_dim(self) -> int:
        return self.encoder.out_dim

    def composed_features(self, compact_item_ids: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                self.item_encoder(compact_item_ids),
                *(branch(compact_item_ids) for branch in self.metadata_branches),
            ],
            dim=-1,
        )

    def forward(self, compact_item_ids: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.composed_features(compact_item_ids))
