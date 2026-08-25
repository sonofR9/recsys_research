from typing import Callable, Literal

import torch
from torch import nn

from .densenet import DenseNet
from .types import ModuleWithDim

LayerItemFeaturePlacement = Literal["before", "after"]


class LayerItemFeatureFusion(ModuleWithDim):
    initializer_rng_nonadvancing = True

    def __init__(
        self,
        model_dim: int,
        feature_dim: int,
        placement: LayerItemFeaturePlacement,
    ) -> None:
        super().__init__()
        if model_dim < 1 or feature_dim < 1:
            raise ValueError("item-feature dimensions must be positive")
        self.model_dim = model_dim
        self.feature_dim = feature_dim
        self.placement = placement

    @property
    def out_dim(self) -> int:
        return self.model_dim

    def _combine(
        self,
        hidden: torch.Tensor,
        update: torch.Tensor,
        active: torch.Tensor | None,
    ) -> torch.Tensor:
        combined = hidden + update
        if active is None:
            return combined
        if active.shape != hidden.shape[:-1]:
            raise ValueError(
                f"expected item-feature mask {hidden.shape[:-1]}, got {active.shape}"
            )
        return torch.where(active.unsqueeze(-1), combined, hidden)


class DirectAddItemFeature(LayerItemFeatureFusion):
    def __init__(self, model_dim: int) -> None:
        super().__init__(model_dim, model_dim, "before")

    def forward(
        self,
        hidden: torch.Tensor,
        original: torch.Tensor,
        feature: torch.Tensor,
        active: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._combine(hidden, feature, active)


class ConcatenatedItemFeatureResidual(LayerItemFeatureFusion):
    def __init__(self, model_dim: int, feature_dim: int) -> None:
        super().__init__(model_dim, feature_dim, "before")
        self.hidden_norm = nn.RMSNorm(model_dim, eps=1e-12)
        self.feature_projection = nn.Linear(feature_dim, model_dim, bias=False)
        self.feature_norm = nn.RMSNorm(model_dim, eps=1e-12)
        self.encoder = DenseNet(2 * model_dim, model_dim)
        self.residual_scale = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        hidden: torch.Tensor,
        original: torch.Tensor,
        feature: torch.Tensor,
        active: torch.Tensor | None = None,
    ) -> torch.Tensor:
        encoded = self.encoder(
            torch.cat(
                [
                    self.hidden_norm(hidden),
                    self.feature_norm(self.feature_projection(feature)),
                ],
                dim=-1,
            )
        )
        return self._combine(hidden, self.residual_scale * encoded, active)


class GemmaItemFeatureResidual(LayerItemFeatureFusion):
    def __init__(
        self,
        model_dim: int,
        feature_dim: int,
        finite_readout_factory: Callable[[int, int], nn.Linear] | None = None,
    ) -> None:
        super().__init__(model_dim, feature_dim, "after")
        finite_readout_factory = finite_readout_factory or (
            lambda input_dim, output_dim: nn.Linear(
                input_dim, output_dim, bias=False
            )
        )
        self.original_projection = finite_readout_factory(model_dim, feature_dim)
        self.original_norm = nn.RMSNorm(feature_dim, eps=1e-12)
        self.hidden_gate = finite_readout_factory(model_dim, feature_dim)
        self.output_projection = nn.Linear(feature_dim, model_dim, bias=False)
        self.output_norm = nn.RMSNorm(model_dim, eps=1e-12)
        self.activation = nn.GELU()
        self.residual_scale = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        hidden: torch.Tensor,
        original: torch.Tensor,
        feature: torch.Tensor,
        active: torch.Tensor | None = None,
    ) -> torch.Tensor:
        original_feature = self.original_norm(self.original_projection(original))
        per_layer_feature = (
            original_feature + self.feature_dim**0.5 * feature
        ) / 2**0.5
        gate = self.activation(self.hidden_gate(hidden))
        update = self.output_norm(self.output_projection(gate * per_layer_feature))
        return self._combine(hidden, self.residual_scale * update, active)
