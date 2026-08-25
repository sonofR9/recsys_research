from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn

from dcn.data.features import FeatureValues

from dcn.nn.multi_task_embedding import BaseMultiTaskEmbeddingLayer
from dcn.nn.types import ModuleWithDim, OutputDims

SHARED_KEY = "shared"


class MultiHeadNetwork(ModuleWithDim):
    def __init__(
        self,
        multi_task_embedding: BaseMultiTaskEmbeddingLayer,
        shared_network: ModuleWithDim,
        task_networks: Mapping[str, ModuleWithDim],
        feature_encoders: Sequence[tuple[str, ModuleWithDim]],
        dense_feature_names: Sequence[str] = (),
        dense_encoder: ModuleWithDim | None = None,
        history_encoder: ModuleWithDim | None = None,
    ):
        super().__init__()

        assert task_networks, "task_networks must be non-empty"

        embedding_split_keys = set(multi_task_embedding.split_ratios.keys())
        expected_keys = {SHARED_KEY} | set(task_networks.keys())
        assert embedding_split_keys == expected_keys, (
            f"Embedding split_ratios keys {embedding_split_keys} must equal "
            f"{{'shared'}} ∪ task_networks keys {expected_keys}"
        )

        self.multi_task_embedding = multi_task_embedding
        self.shared_network = shared_network
        self.task_names = list(task_networks.keys())
        self.task_networks = nn.ModuleDict(task_networks)
        self.encoder_columns = [column for column, _ in feature_encoders]
        # A list, not a mapping: a column may feed more than one encoder.
        self.feature_encoders = nn.ModuleList(
            [encoder for _, encoder in feature_encoders]
        )
        self.dense_feature_names = list(dense_feature_names)
        self.dense_encoder = dense_encoder
        self.history_encoder = history_encoder

    def forward(self, batch: dict[str, Any]) -> dict[str, FeatureValues]:
        int_columns = batch["int_columns"]

        embeddings = self.multi_task_embedding(int_columns)

        shared_parts = [embeddings[SHARED_KEY]]
        for column, encoder in zip(self.encoder_columns, self.feature_encoders):
            shared_parts.append(encoder(int_columns[column]))
        if self.dense_feature_names:
            float_columns = batch["float_columns"]
            # matrix(), not dense(): counters arrive as one fixed-width
            # column rather than one column each.
            dense = torch.cat(
                [float_columns[name].matrix() for name in self.dense_feature_names],
                dim=1,
            )
            shared_parts.append(
                dense if self.dense_encoder is None else self.dense_encoder(dense)
            )

        token_features = torch.cat(shared_parts, dim=1)

        offsets = batch.get("cumulative_lens")
        if offsets is None:
            offsets = torch.arange(
                token_features.shape[0] + 1,
                dtype=torch.int64,
                device=token_features.device,
            )

        if self.history_encoder is not None:
            token_features = torch.cat(
                [token_features, self.history_encoder(token_features, offsets)], dim=1
            )
        shared_out = self.shared_network(token_features)

        outputs: dict[str, FeatureValues] = {}
        for name in self.task_names:
            task_in = torch.cat([shared_out, embeddings[name]], dim=1)
            outputs[name] = FeatureValues(self.task_networks[name](task_in), offsets)
        return outputs

    @property
    def out_dim(self) -> OutputDims:
        dims: dict[str, int] = {}
        for name in self.task_names:
            net_out = self.task_networks[name].out_dim
            assert isinstance(net_out, int)
            dims[name] = net_out
        return OutputDims(dims=dims)
