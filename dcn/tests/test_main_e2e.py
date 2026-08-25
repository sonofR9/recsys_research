from datetime import date
from pathlib import Path

import pytest
import torch

from dcn.config import (
    CheckpointConfig,
    DataloaderConfig,
    DayRangeConfig,
    Experiment,
    LoggingConfig,
    PretrainConfig,
    RuntimeConfig,
)
from dcn.data import EmaCounter
from dcn.datasets.base import DatasetSource, DatasetSourceArtifacts
from dcn.main import load_experiment, run_experiment
from dcn.models import MultiHeadNetwork, TargetExtractionWrapper
from dcn.nn import PrecomputedEmbeddingLookup
from dcn.tests.tiny_ratings import (
    ITEM_COLUMN,
    TARGET,
    USER_COLUMN,
    extract_rating,
    one_task_network,
    rating_criterion,
    write_embeddings,
    write_ratings,
)
from neuralrec.nn.metrics import RMSE


class _FakeDatasetSource(DatasetSource):
    def __init__(self, base_path: Path) -> None:
        self.base_path = Path(base_path)
        self._artifacts = self._prepare()

    @property
    def artifacts(self) -> DatasetSourceArtifacts:
        return self._artifacts

    def _prepare(self) -> DatasetSourceArtifacts:
        main_parquet = self.base_path / "main.parquet"
        embeddings_parquet = self.base_path / "embeddings.parquet"

        if not main_parquet.exists():
            write_ratings(main_parquet, day=[date(2023, 1, 1)] * 4)
        if not embeddings_parquet.exists():
            write_embeddings(embeddings_parquet, id_column="compact_id")

        return DatasetSourceArtifacts(
            main_parquet=main_parquet,
            columns=[USER_COLUMN, ITEM_COLUMN, TARGET],
            precomputed_embeddings={ITEM_COLUMN: embeddings_parquet},
            timestamp_column="timestamp",
            user_column=USER_COLUMN,
            item_id_column=ITEM_COLUMN,
        )


class _SmallExperiment(Experiment):
    def create_dataset_source(self) -> DatasetSource:
        return _FakeDatasetSource(self.base_path)

    def create_counters(self) -> list[EmaCounter]:
        return []

    def create_criterion(self) -> torch.nn.Module:
        return rating_criterion()

    def create_metrics(self) -> list[TargetExtractionWrapper]:
        return [extract_rating(RMSE())]

    def create_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.base_model.parameters(), lr=0.001, fused=True)

    def _create_model(self) -> MultiHeadNetwork:
        return one_task_network(
            PrecomputedEmbeddingLookup.from_parquet(
                self.artifacts.precomputed_embeddings[ITEM_COLUMN],
                learnable_default=False,
                strict=False,
            ),
            extra_shared_dim=self.num_counters,
        )


def _build_experiment(tmp_path: Path) -> _SmallExperiment:
    return _SmallExperiment(
        run_name="test_run",
        base_path=tmp_path,
        invalidate_cache=False,
        runtime=RuntimeConfig(dtype=torch.float32, compile=False),
        day_range=DayRangeConfig(start_day=0, end_day=0),
        dataloader=DataloaderConfig(
            batch_size=2, val_batch_size=2, num_workers=0, prefetch_factor=None
        ),
        pretrain=PretrainConfig(days=0, num_epochs=0, shuffle_days=False),
        logging=LoggingConfig(enable_predictions=False),
        checkpointing=CheckpointConfig(load_checkpoint=False),
    )


@pytest.mark.training_e2e
def test_run_experiment_trains_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WANDB_MODE", "disabled")
    experiment = _build_experiment(tmp_path)
    run_experiment(experiment)


def test_load_experiment_discovers_instance(tmp_path: Path) -> None:
    script_path = tmp_path / "script.py"
    script_path.write_text(
        "from dcn.config import YambdaExperiment\n\nexperiment = YambdaExperiment()\n"
    )

    experiment = load_experiment(script_path)

    assert isinstance(experiment, Experiment)
