import logging

import torch
import polars as pl
from torch import nn as nn

logger = logging.getLogger(__name__)


# FIXME: module name `ple.py` reads as PLE (Progressive Layered Extraction, the standard
# multi-task recsys architecture) but this file is a PiecewiseLinearEncoder for numeric
# features — unrelated. Rename the module to avoid the collision.
class PiecewiseLinearEncoder(nn.Module):
    @staticmethod
    def compute_bins(
        X: torch.Tensor,
        n_bins: int,
    ) -> list[torch.Tensor]:
        bin_edges = torch.quantile(
            X, torch.linspace(0, 1, n_bins + 1, device=X.device), dim=0
        )

        bins_per_feature = []
        for feature_idx in range(bin_edges.shape[1]):
            unique_edges = torch.unique(bin_edges[:, feature_idx])
            bins_per_feature.append(unique_edges)

        return bins_per_feature

    @classmethod
    def from_dataset(
        cls,
        dense_train_df: pl.DataFrame,
        n_bins: int = 32,
    ) -> "PiecewiseLinearEncoder":
        dense_trian_tensor = torch.tensor(
            dense_train_df.to_numpy(), dtype=torch.float32
        )

        bins_per_feature = cls.compute_bins(dense_trian_tensor, n_bins)
        cls._warn_about_constant_features(bins_per_feature, dense_train_df.columns)

        n_features = len(bins_per_feature)
        # A constant feature has a single edge and so no bin at all. It still
        # needs a column, which the zero weight and bias below make constant.
        max_n_bins = max(1, *(len(b) - 1 for b in bins_per_feature))

        weight = torch.zeros(n_features, max_n_bins, dtype=torch.float32)
        bias = torch.zeros(n_features, max_n_bins, dtype=torch.float32)

        mask = torch.zeros(n_features, max_n_bins, dtype=torch.bool)
        single_bin_mask = torch.zeros(n_features, dtype=torch.bool)

        n_bins_list = []

        for feat_idx, edges in enumerate(bins_per_feature):
            n_bins = max(len(edges) - 1, 1)
            n_bins_list.append(n_bins)

            for bin_idx in range(len(edges) - 1):
                a, b = edges[bin_idx], edges[bin_idx + 1]
                width = (b - a).item()
                weight[feat_idx, bin_idx] = 1.0 / max(width, 1e-8)
                bias[feat_idx, bin_idx] = -a / max(width, 1e-8)

            mask[feat_idx, :n_bins] = True
            single_bin_mask[feat_idx] = n_bins == 1

        return cls(weight, bias, mask, n_bins_list, single_bin_mask)

    @staticmethod
    def _warn_about_constant_features(
        bins_per_feature: list[torch.Tensor], names: list[str]
    ) -> None:
        constant = [
            names[index]
            for index, edges in enumerate(bins_per_feature)
            if len(edges) < 2
        ]
        if constant:
            logger.warning(
                "Constant in the fitting sample, so encoded as 0 and carrying"
                " nothing into the model: %s",
                ", ".join(constant),
            )

    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor,
        mask: torch.Tensor,
        n_bins: int,
        single_bin_mask,
    ):
        super().__init__()

        self.weight = nn.Buffer(weight)
        self.bias = nn.Buffer(bias)
        self.single_bin_mask = nn.Buffer(single_bin_mask)
        self.mask = nn.Buffer(mask)
        self._n_bins = n_bins

    @property
    def n_bins(self):
        return self._n_bins

    @property
    def out_dim(self) -> int:
        """One column per live bin: features differ in how many they got."""
        return int(self.mask.sum())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.bias + self.weight * x.unsqueeze(-1)

        if encoded.size(-1) >= 1:
            encoded[..., 0] = torch.clamp(encoded[..., 0], max=1.0)

        if encoded.size(-1) >= 2:
            if encoded.size(-1) > 2:
                encoded[..., 1:-1] = torch.clamp(encoded[..., 1:-1], min=0.0, max=1.0)
            encoded[..., -1] = torch.clamp(encoded[..., -1], min=0.0)

        encoded[:, self.single_bin_mask, 0] = encoded[:, self.single_bin_mask, 0].clamp(
            0.0, 1.0
        )

        flat_encoded = encoded.flatten(start_dim=1)

        flat_mask = self.mask.flatten()
        result = flat_encoded[:, flat_mask]

        return result
