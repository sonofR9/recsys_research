from pathlib import Path
from typing import Any

import pytest
import torch

from dcn.data.dataset import EventDataset, collate_event_batch
from dcn.data.features import FeatureValues
from dcn.datasets.base import DatasetSource, DatasetSourceArtifacts
from dcn.datasets.remap import (
    apply_id_remap_to_parquet,
    build_id_remap_and_remapped_embeddings,
)
from dcn.models.loss_wrapper import LossWrapper
from dcn.nn.precomputed_embeddings import PrecomputedEmbeddingLookup
from dcn.tests.helpers import scalar_feature
from dcn.tests.tiny_ratings import (
    EMBEDDINGS,
    ITEM_COLUMN,
    TARGET,
    USER_COLUMN,
    one_task_network,
    rating_criterion,
    write_embeddings,
    write_ratings,
)


class _OneTaskFakeSource(DatasetSource):
    def __init__(
        self, parquet_path: Path, embeddings_path: Path, tmp_path: Path
    ) -> None:
        self.parquet_path = parquet_path
        self.embeddings_path = embeddings_path
        self.tmp_path = tmp_path
        self._artifacts = self._prepare()

    @property
    def artifacts(self) -> DatasetSourceArtifacts:
        return self._artifacts

    def _prepare(self) -> DatasetSourceArtifacts:
        remap_parquet = self.tmp_path / "remap.parquet"
        remapped_embeddings_parquet = self.tmp_path / "remapped_embeddings.parquet"
        remapped_main_parquet = self.tmp_path / "remapped_main.parquet"

        build_id_remap_and_remapped_embeddings(
            main_parquet=self.parquet_path,
            embeddings_parquet=self.embeddings_path,
            remap_parquet=remap_parquet,
            remapped_embeddings_parquet=remapped_embeddings_parquet,
            raw_id_column=ITEM_COLUMN,
            embedding_column="normalized_embed",
        )

        apply_id_remap_to_parquet(
            main_parquet=self.parquet_path,
            remap_parquet=remap_parquet,
            output_parquet=remapped_main_parquet,
            id_column=ITEM_COLUMN,
            compact_column=ITEM_COLUMN,
        )

        return DatasetSourceArtifacts(
            main_parquet=remapped_main_parquet,
            columns=[USER_COLUMN, ITEM_COLUMN, TARGET],
            precomputed_embeddings={ITEM_COLUMN: remapped_embeddings_parquet},
            timestamp_column="timestamp",
            user_column=USER_COLUMN,
            item_id_column=ITEM_COLUMN,
        )


@pytest.fixture
def one_task_parquet(tmp_path: Path) -> Path:
    return write_ratings(tmp_path / "movielens_like.parquet")


@pytest.fixture
def one_task_embeddings(tmp_path: Path) -> Path:
    return write_embeddings(tmp_path / "embeddings.parquet")


def test_end_to_end_basic(
    one_task_parquet: Path, one_task_embeddings: Path, tmp_path: Path
) -> None:
    source = _OneTaskFakeSource(one_task_parquet, one_task_embeddings, tmp_path)
    artifacts = source.artifacts

    dataset = EventDataset(
        parquet_files=[artifacts.main_parquet],
        columns=artifacts.columns,
        timestamp_column=artifacts.timestamp_column,
    )

    batch = collate_event_batch([dataset[index] for index in range(len(dataset))])

    assert set(batch["int_columns"].keys()) == {USER_COLUMN, ITEM_COLUMN}
    for feature_values in batch["int_columns"].values():
        assert feature_values.num_rows() == 4
        assert feature_values.values.shape == (4,)
        assert feature_values.offsets.tolist() == [0, 1, 2, 3, 4]
    assert torch.allclose(
        batch["float_columns"][TARGET].dense(), torch.tensor([5.0, 4.0, 3.0, 2.0])
    )

    precomputed = PrecomputedEmbeddingLookup.from_parquet(
        artifacts.precomputed_embeddings[ITEM_COLUMN],
        learnable_default=False,
        strict=False,
    )
    wrapper = LossWrapper(
        model=one_task_network(precomputed), criterion=rating_criterion()
    )
    result = wrapper(batch)

    assert result[f"{TARGET}_pred"].values.shape == (4, 1)
    assert torch.isfinite(result["loss"])
    result["loss"].backward()


def test_end_to_end_variable_length() -> None:
    movie_history = [[1, 2, 3], [2], [], [4, 1]]
    flat_movies = [movie_id for row in movie_history for movie_id in row]
    offsets = [0]
    for row in movie_history:
        offsets.append(offsets[-1] + len(row))

    categorical_features: dict[str, FeatureValues] = {
        USER_COLUMN: scalar_feature(torch.tensor([1, 2, 3, 4], dtype=torch.int64)),
        ITEM_COLUMN: FeatureValues(
            values=torch.tensor(flat_movies, dtype=torch.int64),
            offsets=torch.tensor(offsets, dtype=torch.int64),
        ),
    }

    precomputed_embeddings = torch.tensor(EMBEDDINGS)
    precomputed = PrecomputedEmbeddingLookup(
        embeddings=precomputed_embeddings,
        learnable_default=False,
        strict=False,
    )

    wrapper = LossWrapper(
        model=one_task_network(precomputed), criterion=rating_criterion()
    )
    batch: dict[str, Any] = {
        "int_columns": categorical_features,
        "float_columns": {TARGET: scalar_feature(torch.tensor([1.0, 0.8, 0.6, 0.4]))},
        "timestamp": torch.tensor([100, 200, 300, 400], dtype=torch.int64),
    }
    result = wrapper(batch)

    assert result[f"{TARGET}_pred"].values.shape == (4, 1)
    assert torch.isfinite(result["loss"])
    result["loss"].backward()

    movie_pooled_expected = torch.stack(
        [
            precomputed_embeddings[[0, 1, 2]].sum(dim=0),
            precomputed_embeddings[1],
            torch.zeros(4),
            precomputed_embeddings[[3, 0]].sum(dim=0),
        ]
    )
    movie_pooled_actual = precomputed(categorical_features[ITEM_COLUMN])
    assert torch.allclose(movie_pooled_actual, movie_pooled_expected)
