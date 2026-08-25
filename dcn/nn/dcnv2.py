from typing import Literal

import torch
from torch import nn

from .crossnet import CrossNetwork
from .types import ModuleWithDim


class DcnV2(ModuleWithDim):
    def __init__(
        self,
        cross: CrossNetwork,
        deep_parallel: ModuleWithDim | None = None,
        mode: Literal["sequential", "parallel", "combined"] = "sequential",
        output_dim: int | None = None,
        compression: ModuleWithDim | None = None,
        deep_sequential: ModuleWithDim | None = None,
    ):
        super().__init__()
        assert isinstance(cross.out_dim, int)
        self.compression = compression
        self.cross = cross
        self.deep_parallel = deep_parallel
        self.deep_sequential = deep_sequential
        self.mode = mode

        if self.mode == "parallel":
            assert output_dim is not None, "output_dim required for parallel mode"
            assert deep_parallel is not None, "deep_parallel required for parallel mode"
            assert isinstance(deep_parallel.out_dim, int)

            combined_dim = cross.out_dim + deep_parallel.out_dim
            self.final_layer = nn.Linear(combined_dim, output_dim)
            self._out_dim = output_dim
        elif self.mode == "combined":
            assert deep_sequential is not None, (
                "deep_sequential required for combined mode"
            )
            assert isinstance(deep_sequential.out_dim, int)
            self._out_dim = deep_sequential.out_dim
        elif self.mode == "sequential":
            assert deep_sequential is not None, (
                "deep_sequential required for sequential mode"
            )
            assert isinstance(deep_sequential.out_dim, int)
            self._out_dim = deep_sequential.out_dim
        else:
            raise ValueError(
                f"Unknown mode: {mode}. Must be 'sequential', 'parallel', or 'combined'"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.compression is not None:
            x = self.compression(x)

        if self.mode == "sequential":
            return self.deep_sequential(self.cross(x))  # type: ignore

        if self.mode == "parallel":
            cross_out = self.cross(x)
            deep_out = self.deep_parallel(x)  # type: ignore
            combined = torch.cat([cross_out, deep_out], dim=-1)
            return self.final_layer(combined)

        if self.mode == "combined":
            cross_out = self.cross(x)
            deep_parallel = self.deep_parallel(x)  # type: ignore
            combined = torch.cat([cross_out, deep_parallel], dim=-1)
            deep_out = self.deep_sequential(combined)  # type: ignore
            return deep_out

        raise ValueError(f"Unknown mode: {self.mode}")

    @property
    def out_dim(self) -> int:
        return self._out_dim
