import torch

from .types import ModuleWithDim


class HistoryEncoder(ModuleWithDim):
    """Summarises what a user has done up to and including the current event."""

    def __init__(self, projection: ModuleWithDim, sequence_model: ModuleWithDim):
        super().__init__()
        assert projection.out_dim == sequence_model.out_dim, (
            f"projection dim {projection.out_dim} != "
            f"sequence model dim {sequence_model.out_dim}"
        )
        self.projection = projection
        self.sequence_model = sequence_model

    @property
    def out_dim(self) -> int:
        return self.sequence_model.out_dim

    def forward(
        self, token_features: torch.Tensor, cumulative_lens: torch.Tensor
    ) -> torch.Tensor:
        return self.sequence_model(self.projection(token_features), cumulative_lens)
