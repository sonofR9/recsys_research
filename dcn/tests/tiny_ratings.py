"""Four users rating four movies, and the one-task network that reads them."""

from functools import partial
from pathlib import Path

import polars as pl
import torch

from dcn.models.criterions import CriterionSpec, MultiCriterion, TargetExtractionWrapper
from dcn.models.multi_head_network import MultiHeadNetwork
from dcn.nn.multi_task_embedding import MultiTaskEmbeddingLayer
from dcn.nn.precomputed_embeddings import PrecomputedEmbeddingLookup
from dcn.nn.resnet import ResNet1D

USER_COLUMN = "user_id"
ITEM_COLUMN = "movie_id"
TARGET = "rating"

RATINGS = {
    USER_COLUMN: [1, 2, 3, 4],
    ITEM_COLUMN: [1, 2, 3, 4],
    TARGET: [5.0, 4.0, 3.0, 2.0],
    "timestamp": [100, 200, 300, 400],
}

# One-hot, so a lookup's output says which row it read.
EMBEDDINGS = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]

extract_rating = partial(
    TargetExtractionWrapper, prediction_column=TARGET, target_column=TARGET
)


def write_ratings(path: Path, **extra_columns: object) -> Path:
    pl.DataFrame({**RATINGS, **extra_columns}).write_parquet(path)
    return path


def write_embeddings(path: Path, id_column: str = ITEM_COLUMN) -> Path:
    pl.DataFrame(
        {id_column: [1, 2, 3, 4], "normalized_embed": EMBEDDINGS}
    ).write_parquet(path)
    return path


def rating_criterion() -> MultiCriterion:
    return MultiCriterion(
        [CriterionSpec(TARGET, extract_rating(torch.nn.MSELoss()), 1.0)]
    )


def one_task_network(
    precomputed: PrecomputedEmbeddingLookup, *, extra_shared_dim: int = 0
) -> MultiHeadNetwork:
    """Hashed user and movie ids plus the precomputed lookup, into one head."""
    embedding = MultiTaskEmbeddingLayer(
        feature_configs={USER_COLUMN: 2, ITEM_COLUMN: 2},
        num_embeddings=64,
        embedding_dim=8,
        split_ratios={"shared": 0.5, TARGET: 0.5},
        sparse=False,
    )
    shared_in = (
        embedding.out_dim.dims["shared"] + precomputed.embedding_dim + extra_shared_dim
    )
    shared = ResNet1D(
        input_dim=shared_in, hidden_dims=[shared_in], norm_factory=torch.nn.LayerNorm
    )
    task_in = shared.out_dim + embedding.out_dim.dims[TARGET]
    return MultiHeadNetwork(
        multi_task_embedding=embedding,
        shared_network=shared,
        task_networks={
            TARGET: ResNet1D(
                input_dim=task_in,
                hidden_dims=[task_in, 1],
                norm_factory=torch.nn.LayerNorm,
            )
        },
        feature_encoders=[(ITEM_COLUMN, precomputed)],
    )
