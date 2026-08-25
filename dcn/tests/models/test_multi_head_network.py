import pytest
import torch
import torch.nn as nn

from dcn.data.features import FeatureValues
from dcn.models.multi_head_network import MultiHeadNetwork
from dcn.nn.types import ModuleWithDim, OutputDims


class FakeMultiTaskEmbedding(nn.Module):
    def __init__(
        self,
        split_ratios: dict[str, float],
        embed_dim: int = 8,
        num_features: int = 2,
    ) -> None:
        super().__init__()
        self.split_ratios = split_ratios
        per_split = embed_dim
        self._dims = {name: per_split * num_features for name in split_ratios}

    @property
    def out_dim(self) -> OutputDims:
        return OutputDims(dims=self._dims)

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        batch = next(iter(features.values())).shape[0]
        return {name: torch.ones(batch, dim) for name, dim in self._dims.items()}


class LinearHead(ModuleWithDim):
    def __init__(self, in_dim: int, out_dim: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self._out_dim = out_dim

    @property
    def out_dim(self) -> int:
        return self._out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class IdentityShared(ModuleWithDim):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self._dim = dim
        self.proj = nn.Linear(dim, dim)

    @property
    def out_dim(self) -> int:
        return self._dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def _build(task_split: dict[str, float], embed_dim: int = 8) -> MultiHeadNetwork:
    split_ratios = {"shared": 0.5, **task_split}
    embedding = FakeMultiTaskEmbedding(
        split_ratios=split_ratios, embed_dim=embed_dim, num_features=2
    )
    shared_dim = embedding.out_dim.dims["shared"]
    shared = IdentityShared(shared_dim)
    task_networks = {
        name: LinearHead(in_dim=shared_dim + embedding.out_dim.dims[name])
        for name in task_split
    }
    return MultiHeadNetwork(embedding, shared, task_networks, feature_encoders=[])


def test_two_tasks_returns_dict() -> None:
    model = _build({"like": 0.25, "listen": 0.25})
    batch_size = 4
    categorical_features = {
        "a": torch.zeros(batch_size, dtype=torch.int64),
        "b": torch.zeros(batch_size, dtype=torch.int64),
    }

    output = model({"int_columns": categorical_features})

    assert set(output.keys()) == {"like", "listen"}
    assert output["like"].values.shape == (batch_size, 1)
    assert output["listen"].values.shape == (batch_size, 1)


def test_three_tasks_dynamic() -> None:
    model = _build({"a": 0.25, "b": 0.15, "c": 0.10})
    categorical_features = {
        "x": torch.zeros(3, dtype=torch.int64),
        "y": torch.zeros(3, dtype=torch.int64),
    }

    output = model({"int_columns": categorical_features})

    assert set(output.keys()) == {"a", "b", "c"}


def test_single_task() -> None:
    model = _build({"rating": 0.5})
    categorical_features = {
        "u": torch.zeros(2, dtype=torch.int64),
        "i": torch.zeros(2, dtype=torch.int64),
    }

    output = model({"int_columns": categorical_features})

    assert set(output.keys()) == {"rating"}


def test_split_ratios_must_match_tasks() -> None:
    embedding = FakeMultiTaskEmbedding(split_ratios={"shared": 0.5, "like": 0.5})
    shared = IdentityShared(embedding.out_dim.dims["shared"])
    task_networks = {
        "listen": LinearHead(in_dim=shared.out_dim + embedding.out_dim.dims["like"])
    }

    with pytest.raises(AssertionError, match="split_ratios"):
        MultiHeadNetwork(embedding, shared, task_networks, feature_encoders=[])


def test_a_packed_dense_column_feeds_the_shared_network() -> None:
    batch_size, width = 4, 3
    split_ratios = {"shared": 0.5, "like": 0.5}
    embedding = FakeMultiTaskEmbedding(split_ratios=split_ratios, num_features=2)
    shared_dim = embedding.out_dim.dims["shared"]
    shared = IdentityShared(shared_dim + width)
    model = MultiHeadNetwork(
        embedding,
        shared,
        {"like": LinearHead(in_dim=shared.out_dim + embedding.out_dim.dims["like"])},
        feature_encoders=[],
        dense_feature_names=["counters"],
    )
    packed = FeatureValues(
        values=torch.arange(batch_size * width, dtype=torch.float32),
        offsets=torch.arange(batch_size + 1, dtype=torch.int64) * width,
    )

    output = model(
        {
            "int_columns": {"a": torch.zeros(batch_size, dtype=torch.int64)},
            "float_columns": {"counters": packed},
        }
    )

    assert output["like"].values.shape == (batch_size, 1)


def test_out_dim_aggregates_task_dims() -> None:
    model = _build({"like": 0.25, "listen": 0.25})

    out_dim = model.out_dim

    assert out_dim.dims == {"like": 1, "listen": 1}


def test_a_history_encoder_widens_the_shared_input() -> None:
    history_dim = 5
    tokens = 6

    class _ConstantHistory(ModuleWithDim):
        @property
        def out_dim(self) -> int:
            return history_dim

        def forward(
            self, token_features: torch.Tensor, cumulative_lens: torch.Tensor
        ) -> torch.Tensor:
            self.seen_lens = cumulative_lens
            return torch.ones(token_features.shape[0], history_dim)

    embedding = FakeMultiTaskEmbedding(split_ratios={"shared": 0.5, "like": 0.5})
    shared = IdentityShared(embedding.out_dim.dims["shared"] + history_dim)
    history = _ConstantHistory()
    model = MultiHeadNetwork(
        embedding,
        shared,
        {"like": LinearHead(in_dim=shared.out_dim + embedding.out_dim.dims["like"])},
        feature_encoders=[],
        history_encoder=history,
    )
    cumulative_lens = torch.tensor([0, 4, tokens])

    output = model(
        {
            "int_columns": {"a": torch.zeros(tokens, dtype=torch.int64)},
            "cumulative_lens": cumulative_lens,
        }
    )

    assert output["like"].values.shape == (tokens, 1)
    torch.testing.assert_close(history.seen_lens, cumulative_lens)
