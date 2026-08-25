from typing import Literal

import torch
from torch import nn

from .layer_registry import layer_registry
from .types import ModuleWithDim

ActivationKind = Literal["relu", "gelu", "silu"]

_ACTIVATIONS: dict[ActivationKind, type[nn.Module]] = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
}


class RegularMLP(ModuleWithDim):
    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        dropout: float = 0.0,
        activation: ActivationKind = "gelu",
    ) -> None:
        super().__init__()
        self._dim = dim
        self.linear1 = nn.Linear(dim, intermediate_dim)
        self.linear2 = nn.Linear(intermediate_dim, dim)
        self.activation = _ACTIVATIONS[activation]()
        self.dropout = nn.Dropout(dropout)

    @property
    def input_projection(self) -> nn.Linear:
        return self.linear1

    @property
    def output_projection(self) -> nn.Linear:
        return self.linear2

    @property
    def out_dim(self) -> int:
        return self._dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))

    def init_weights(self, base_std: float, res_std: float) -> None:
        nn.init.trunc_normal_(
            self.linear1.weight, mean=0.0, std=base_std, a=-2 * base_std, b=2 * base_std
        )
        nn.init.zeros_(self.linear1.bias)
        nn.init.trunc_normal_(
            self.linear2.weight, mean=0.0, std=res_std, a=-2 * res_std, b=2 * res_std
        )
        nn.init.zeros_(self.linear2.bias)


class GatedMLP(ModuleWithDim):
    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        activation: ActivationKind,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self._dim = dim
        self.w1 = nn.Linear(dim, intermediate_dim, bias=bias)
        self.w2 = nn.Linear(dim, intermediate_dim, bias=bias)
        self.w3 = nn.Linear(intermediate_dim, dim, bias=bias)
        self.activation = _ACTIVATIONS[activation]()
        self.dropout = nn.Dropout(dropout)

    @property
    def gate_projection(self) -> nn.Linear:
        return self.w1

    @property
    def value_projection(self) -> nn.Linear:
        return self.w2

    @property
    def output_projection(self) -> nn.Linear:
        return self.w3

    @property
    def out_dim(self) -> int:
        return self._dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.activation(self.w1(x)) * self.w2(x)
        return self.w3(self.dropout(hidden))

    def init_weights(self, base_std: float, res_std: float) -> None:
        for layer in (self.w1, self.w2):
            nn.init.trunc_normal_(
                layer.weight,
                mean=0.0,
                std=base_std,
                a=-2 * base_std,
                b=2 * base_std,
            )
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        nn.init.trunc_normal_(
            self.w3.weight, mean=0.0, std=res_std, a=-2 * res_std, b=2 * res_std
        )
        if self.w3.bias is not None:
            nn.init.zeros_(self.w3.bias)


class ReGLU(GatedMLP):
    def __init__(self, dim: int, intermediate_dim: int, dropout: float = 0.0) -> None:
        super().__init__(dim, intermediate_dim, "relu", dropout)


class GEGLU(GatedMLP):
    def __init__(self, dim: int, intermediate_dim: int, dropout: float = 0.0) -> None:
        super().__init__(dim, intermediate_dim, "gelu", dropout)


@layer_registry.register
class SwiGLU(GatedMLP):
    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__(dim, intermediate_dim, "silu", dropout, bias)

    @classmethod
    def from_config(cls, config: dict) -> "SwiGLU":
        dropout = config.get("dropout", 0.0) if config.get("gated_ffn_dropout") else 0.0
        return cls(
            dim=config["dim"],
            intermediate_dim=config["intermediate_dim"],
            dropout=dropout,
        )
