from typing import Literal

import torch
from torch import nn

from .types import ActivationFactory, ModuleWithDim, NormalizationFactory


class FlexibleResBlock(ModuleWithDim):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        activation_factory: ActivationFactory,
        norm_factory: NormalizationFactory,
        norm_type: Literal["pre", "post"] = "pre",
        dropout: float = 0.0,
    ):
        super().__init__()
        self._out_dim = out_dim

        self.shortcut = (
            nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim)
        )

        if norm_type == "pre":
            self.main = nn.Sequential(
                norm_factory(in_dim),
                nn.Linear(in_dim, out_dim),
                activation_factory(),
                nn.Dropout(dropout),
            )
            self.post_block_op = nn.Identity()
        else:
            self.main = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                activation_factory(),
                nn.Dropout(dropout),
            )
            self.post_block_op = norm_factory(out_dim)

    @property
    def out_dim(self) -> int:
        return self._out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.post_block_op(self.main(x) + self.shortcut(x))


class ResNet1D(ModuleWithDim):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        activation_factory: ActivationFactory = nn.GELU,
        norm_factory: NormalizationFactory = nn.BatchNorm1d,
        norm_type: Literal["pre", "post"] = "pre",
        dropout: float = 0.0,
    ):
        super().__init__()

        layers = []
        current_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(
                FlexibleResBlock(
                    in_dim=current_dim,
                    out_dim=h_dim,
                    activation_factory=activation_factory,
                    norm_factory=norm_factory,
                    norm_type=norm_type,
                    dropout=dropout,
                )
            )
            current_dim = h_dim

        self.network = nn.Sequential(*layers)
        self._out_dim = hidden_dims[-1] if hidden_dims else input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    @property
    def out_dim(self) -> int:
        return self._out_dim
