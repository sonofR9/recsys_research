import torch
from torch import nn

from .types import ModuleWithDim


class DenseNet(ModuleWithDim):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        preserve_input_rms: bool = False,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        hidden_dim = output_dim if hidden_dim is None else hidden_dim
        self.input_dim = input_dim
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.GELU()
        self.output_projection = nn.Linear(hidden_dim, output_dim)
        self.output_norm = (
            nn.RMSNorm(output_dim, eps=1e-12, elementwise_affine=False)
            if preserve_input_rms
            else nn.Identity()
        )
        self.preserve_input_rms = preserve_input_rms
        self._out_dim = output_dim

    @property
    def out_dim(self) -> int:
        return self._out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.output_projection(self.activation(self.input_projection(x)))
        if not self.preserve_input_rms:
            return output
        input_rms = x.square().mean(dim=-1, keepdim=True).sqrt()
        return self.output_norm(output) * input_rms
