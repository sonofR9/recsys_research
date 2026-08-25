from abc import abstractmethod
from typing import Literal

import torch
import torch.nn as nn

from dcn.data.features import FeatureValues

from .types import ModuleWithDim, OutputDims


HashedFeature = tuple[torch.Tensor, torch.Tensor]


class BaseMultiTaskEmbeddingLayer(ModuleWithDim):
    """Hash-based shared embedding table split per task."""

    def __init__(
        self,
        feature_configs: dict[str, int],
        num_embeddings: int,
        embedding_dim: int,
        split_ratios: dict[str, float],
        mode: Literal["sum", "concat"] = "sum",
    ):
        super().__init__()

        assert abs(sum(split_ratios.values()) - 1.0) < 1e-6, (
            "split_ratios must sum to 1.0"
        )
        assert all(r >= 0 for r in split_ratios.values()), (
            "split_ratios must be non-negative"
        )
        assert (num_embeddings & (num_embeddings - 1)) == 0, (
            "num_embeddings must be a power of 2"
        )

        self.split_ratios = split_ratios
        self.split_names = list(split_ratios.keys())
        self.mode = mode
        self.feature_configs = feature_configs
        self.feature_names = list(feature_configs.keys())
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.split_dims = {}
        remaining = embedding_dim
        for i, (name, ratio) in enumerate(split_ratios.items()):
            if i == len(split_ratios) - 1:
                self.split_dims[name] = remaining
            else:
                dim = int(embedding_dim * ratio)
                self.split_dims[name] = dim
                remaining -= dim

        max_num_hashes = max(feature_configs.values())
        self.hash_a = nn.Buffer(
            torch.randint(1, 2**63 - 1, (max_num_hashes,), dtype=torch.int64) | 1,
            persistent=True,
        )

        self._log2_num_embeddings = (num_embeddings - 1).bit_length()

    def _multiply_shift_hash(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        shift = 64 - self._log2_num_embeddings
        hashed = (a.unsqueeze(-1) * x.long()) >> shift
        mask = self.num_embeddings - 1
        return hashed & mask

    def _hash_features(
        self, features: dict[str, FeatureValues]
    ) -> dict[str, HashedFeature]:
        result = {}
        for name in self.feature_names:
            assert name in features, f"feature '{name}' missing from input"
            feat = features[name]
            num_hashes = self.feature_configs[name]
            hashed_values = self._multiply_shift_hash(
                feat.values, self.hash_a[:num_hashes]
            )
            result[name] = (hashed_values, feat.offsets)
        return result

    @abstractmethod
    def _lookup_embeddings(
        self, hashed: dict[str, HashedFeature]
    ) -> dict[str, list[torch.Tensor]]:
        """For each feature, one ``[N, embedding_dim]`` tensor per hash."""
        ...

    def _split_and_aggregate(
        self, hash_embeddings: dict[str, list[torch.Tensor]]
    ) -> dict[str, dict[str, torch.Tensor]]:
        result = {split_name: {} for split_name in self.split_names}

        for name, embeddings_list in hash_embeddings.items():
            split_parts = {split_name: [] for split_name in self.split_names}

            for embedding in embeddings_list:
                offset = 0
                for split_name in self.split_names:
                    dim = self.split_dims[split_name]
                    split_parts[split_name].append(embedding[:, offset : offset + dim])
                    offset += dim

            for split_name in self.split_names:
                parts = split_parts[split_name]
                if self.mode == "sum":
                    result[split_name][name] = torch.stack(parts, dim=0).sum(dim=0)
                else:
                    result[split_name][name] = torch.cat(parts, dim=-1)

        return result

    def forward(self, features: dict[str, FeatureValues]) -> dict[str, torch.Tensor]:
        hashed = self._hash_features(features)
        hash_embeddings = self._lookup_embeddings(hashed)
        split_embeddings = self._split_and_aggregate(hash_embeddings)

        result = {}
        for split_name in self.split_names:
            parts = [split_embeddings[split_name][name] for name in self.feature_names]
            result[split_name] = torch.cat(parts, dim=1) if parts else torch.empty(0)

        return result

    @property
    def out_dim(self) -> OutputDims:
        num_features = len(self.feature_names)

        if self.mode == "sum":
            dims = {name: dim * num_features for name, dim in self.split_dims.items()}
        else:
            total_hashes = sum(self.feature_configs.values())
            dims = {name: dim * total_hashes for name, dim in self.split_dims.items()}

        return OutputDims(dims=dims)


class MultiTaskEmbeddingLayer(BaseMultiTaskEmbeddingLayer):
    def __init__(
        self,
        feature_configs: dict[str, int],
        num_embeddings: int,
        embedding_dim: int,
        split_ratios: dict[str, float],
        sparse: bool = True,
        mode: Literal["sum", "concat"] = "sum",
    ):
        super().__init__(
            feature_configs, num_embeddings, embedding_dim, split_ratios, mode
        )

        self.unified_embedding = nn.EmbeddingBag(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            mode="sum",
            sparse=sparse,
            include_last_offset=True,
        )

    # FIXME: nested Python loop (per feature, per hash) issuing one EmbeddingBag call each, and
    # the whole method is @torch.compiler.disable'd — so this is a graph break and likely the
    # embedding hot path's bottleneck. The `list[Tensor]-per-hash` abstraction in the base class
    # is what forces the loop; a single batched lookup (stack hashes into one bag call) would
    # remove both the loop and the compiler disable. Reconsider the _lookup_embeddings contract.
    @torch.compiler.disable
    def _lookup_embeddings(
        self, hashed: dict[str, HashedFeature]
    ) -> dict[str, list[torch.Tensor]]:
        result = {}
        for name in self.feature_names:
            hashed_values, offsets = hashed[name]
            result[name] = [
                self.unified_embedding(
                    hashed_values[i],
                    offsets,
                )
                for i in range(hashed_values.shape[0])
            ]
        return result


class MultiTaskEmbeddingLayerTorchRec(BaseMultiTaskEmbeddingLayer):
    def __init__(
        self,
        feature_configs: dict[str, int],
        num_embeddings: int,
        embedding_dim: int,
        split_ratios: dict[str, float],
        mode: Literal["sum", "concat"] = "sum",
    ):
        super().__init__(
            feature_configs, num_embeddings, embedding_dim, split_ratios, mode
        )

        from torchrec import EmbeddingBagCollection
        from torchrec.modules.embedding_configs import EmbeddingBagConfig

        self.embedding_bag_collection = EmbeddingBagCollection(
            tables=[
                EmbeddingBagConfig(
                    name="unified",
                    embedding_dim=embedding_dim,
                    num_embeddings=num_embeddings,
                    feature_names=["unified"],
                )
            ],
        )

    def _lookup_embeddings(
        self, hashed: dict[str, HashedFeature]
    ) -> dict[str, list[torch.Tensor]]:
        from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

        all_values: list[torch.Tensor] = []
        all_lengths: list[torch.Tensor] = []
        feature_hash_counts: list[int] = []
        num_rows = next(iter(hashed.values()))[1].shape[0] - 1

        for name in self.feature_names:
            hashed_values, offsets = hashed[name]
            lengths = (offsets[1:] - offsets[:-1]).to(dtype=torch.int32)
            num_hashes = hashed_values.shape[0]
            for i in range(num_hashes):
                all_values.append(hashed_values[i])
                all_lengths.append(lengths)
            feature_hash_counts.append(num_hashes)

        kjt = KeyedJaggedTensor(
            keys=["unified"],
            values=torch.cat(all_values),
            lengths=torch.cat(all_lengths),
        )

        embeddings_output = self.embedding_bag_collection(kjt)
        if hasattr(embeddings_output, "_awaitable"):
            embeddings_output = embeddings_output.wait()
        flat_embeddings = embeddings_output["unified"]

        result = {}
        bag_index = 0
        for name, num_hashes in zip(self.feature_names, feature_hash_counts):
            per_hash = []
            for _ in range(num_hashes):
                per_hash.append(
                    flat_embeddings[bag_index * num_rows : (bag_index + 1) * num_rows]
                )
                bag_index += 1
            result[name] = per_hash
        return result
