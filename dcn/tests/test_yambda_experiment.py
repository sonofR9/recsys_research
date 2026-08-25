from pathlib import Path

import polars as pl
import torch

from data.preprocessing import COUNTERS_COLUMN
from dcn.config import EmbeddingConfig, YambdaExperiment
from dcn.data.features import FeatureValues
from dcn.datasets.base import DatasetSourceArtifacts
from dcn.models import MultiHeadNetwork
from dcn.tests.helpers import scalar_feature

_PRECOMPUTED_DIM = 4
_PRECOMPUTED_NUM_IDS = 6


def _write_precomputed_embeddings(tmp_path: Path) -> Path:
    embeddings = [
        [float(column == row % _PRECOMPUTED_DIM) for column in range(_PRECOMPUTED_DIM)]
        for row in range(_PRECOMPUTED_NUM_IDS)
    ]
    frame = pl.DataFrame(
        {
            "compact_id": list(range(1, _PRECOMPUTED_NUM_IDS + 1)),
            "normalized_embed": embeddings,
        }
    )
    path = tmp_path / "compact_item_id.parquet"
    frame.write_parquet(path)
    return path


def _fake_artifacts(tmp_path: Path) -> DatasetSourceArtifacts:
    return DatasetSourceArtifacts(
        main_parquet=tmp_path / "main.parquet",
        columns=[
            "item_id",
            "compact_item_id",
            "uid",
            "album_id",
            "artist_id",
            "target_like",
            "target_listen",
            "listen_mask",
        ],
        precomputed_embeddings={
            "compact_item_id": _write_precomputed_embeddings(tmp_path)
        },
        timestamp_column="timestamp",
        user_column="uid",
        item_id_column="compact_item_id",
    )


class _FakeManager:
    def __init__(self, counter_columns: list[str]) -> None:
        self.counter_columns = counter_columns
        self.dense_columns = [COUNTERS_COLUMN]


def _small_experiment() -> YambdaExperiment:
    return YambdaExperiment(embedding=EmbeddingConfig(num_embeddings=64, dim=8))


def test_create_model_forward_backward(tmp_path: Path) -> None:
    experiment = _small_experiment()
    counter_columns = ["counter_0", "counter_1", "counter_2"]

    # Pre-seeded so _create_model never touches the real dataset pipeline:
    # cached_property is non-data, so the instance attribute shadows it.
    experiment.artifacts = _fake_artifacts(tmp_path)
    experiment.dataset_manager = _FakeManager(counter_columns)
    experiment.num_counters = len(counter_columns)
    experiment.device = torch.device("cpu")

    model = experiment._create_model()

    assert isinstance(model, MultiHeadNetwork)

    batch_size = 4
    ids = torch.arange(1, batch_size + 1, dtype=torch.int64)
    int_columns = {
        name: scalar_feature(ids)
        for name in [
            "item_id",
            "compact_item_id",
            "uid",
            "album_id",
            "artist_id",
        ]
    }
    float_columns = {
        COUNTERS_COLUMN: FeatureValues(
            values=torch.randn(batch_size * len(counter_columns)),
            offsets=torch.arange(batch_size + 1, dtype=torch.int64)
            * len(counter_columns),
        )
    }

    model.train()
    outputs = model({"int_columns": int_columns, "float_columns": float_columns})

    assert set(outputs.keys()) == {"like", "listen"}
    assert outputs["like"].values.shape == (batch_size, 1)
    assert outputs["listen"].values.shape == (batch_size, 1)

    loss = sum(output.values.sum() for output in outputs.values())
    loss.backward()

    assert any(param.grad is not None for param in model.parameters())
