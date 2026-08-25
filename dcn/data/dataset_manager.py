import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

import yaml
from torch.utils.data import DataLoader, Dataset, Sampler

from data.counters import EmaCounter
from data.preprocessing import COUNTERS_COLUMN, preprocess_counters
from data.split_by_day import split_main_parquet_by_day
from neuralrec.utils.stateful import Stateful
from utils.locks import hold

from .dataset import EventDataset, collate_event_batch

logger = logging.getLogger(__name__)


def _counters_key(counter_columns: Sequence[str]) -> str:
    """Short id of a counter set."""
    if not counter_columns:
        return "none"
    digest = hashlib.sha1("\n".join(counter_columns).encode()).hexdigest()
    return f"{len(counter_columns)}x{digest[:8]}"


class DatasetManager(Stateful):
    _state_fields = ("metadata_file",)

    def __init__(
        self,
        main_parquet: str | Path,
        columns: list[str],
        counter_columns: list[str],
        cache_dir: Path,
        counters: Sequence[EmaCounter] = (),
        invalidate_cache: bool = False,
        timestamp_column: str = "timestamp",
    ):
        self.data_dir = Path(cache_dir)
        self.columns = list(columns)
        self.counter_columns = list(counter_columns)
        self.counters_column = COUNTERS_COLUMN if counter_columns else None
        self.timestamp_column = timestamp_column
        self.counters = list(counters)

        self._days_dir = self.data_dir / "days"
        self._enriched_dir = (
            self.data_dir / f"enriched_{_counters_key(counter_columns)}"
        )
        self.metadata_file = self._enriched_dir / "metadata.yaml"

        with hold(self.data_dir / "dataset-manager.lock", "dataset manager"):
            if not invalidate_cache and self.metadata_file.exists():
                self._load_metadata()
                logger.info("Metadata loaded from cache")
                return

            main_parquet = Path(main_parquet).resolve()
            self._prepare_data(main_parquet, invalidate_cache)
            if self.counters:
                self._run_preprocessing(invalidate_cache)
            self._save_metadata(main_parquet)
            logger.info("Dataset preparation completed")

    def _prepare_data(self, main_parquet: Path, invalidate_cache: bool = False):
        self._enriched_dir.mkdir(parents=True, exist_ok=True)

        self.original_day_to_path = split_main_parquet_by_day(
            main_parquet, self._days_dir, invalidate_cache
        )
        self.day_to_path = dict(self.original_day_to_path)
        logger.info(f"Split parquet into {len(self.original_day_to_path)} days")

    def _save_metadata(self, source_main_parquet: Path):
        metadata = {
            "data_dir": str(self.data_dir.resolve()),
            "columns": self.columns,
            "timestamp_column": self.timestamp_column,
            "counter_columns": self.counter_columns,
            "original_day_to_path": {
                day: str(path) for day, path in self.original_day_to_path.items()
            },
            "day_to_path": {day: str(path) for day, path in self.day_to_path.items()},
            "source_main_parquet": str(source_main_parquet),
            "preparation_timestamp": datetime.now().isoformat(),
        }

        with open(self.metadata_file, "w", encoding="utf-8") as f:
            yaml.dump(metadata, f)

    def _load_metadata(self, validate_schema: bool = True):
        assert self.metadata_file.exists(), (
            f"Metadata file not found: {self.metadata_file}"
        )

        with open(self.metadata_file, "r", encoding="utf-8") as f:
            metadata: Any = yaml.safe_load(f)

        assert metadata is not None, "Metadata file is empty"
        for key in (
            "day_to_path",
            "original_day_to_path",
            "counter_columns",
            "data_dir",
        ):
            assert key in metadata, f"Metadata missing '{key}'"

        loaded_columns = metadata.get("columns", self.columns)
        loaded_counter_columns = metadata["counter_columns"]

        # FIXME: schema validation is all-or-nothing exact list equality on column names. Any
        # reorder or single added/removed column (e.g. one extra counter) raises and forces a
        # full rebuild of every per-day parquet — there is no per-counter / per-column
        # incremental invalidation. Consider keying cached artifacts by a content hash so only
        # the affected days rebuild. Also note load_state_dict restores with
        # validate_schema=False, silently trusting whatever is on disk.
        if validate_schema:
            mismatches = []
            if loaded_counter_columns != self.counter_columns:
                mismatches.append(
                    f"counter_columns: {loaded_counter_columns} != {self.counter_columns}"
                )
            if loaded_columns != self.columns:
                mismatches.append(f"columns: {loaded_columns} != {self.columns}")
            if mismatches:
                raise ValueError(
                    "Cached dataset metadata mismatch. "
                    "Set invalidate_cache=True to regenerate. Details: "
                    + "; ".join(mismatches)
                )

        self.columns = loaded_columns
        self.counter_columns = loaded_counter_columns
        self.timestamp_column = metadata.get("timestamp_column", self.timestamp_column)

        self.original_day_to_path = {
            int(day): Path(path)
            for day, path in metadata["original_day_to_path"].items()
        }
        self.day_to_path = {
            int(day): Path(path) for day, path in metadata["day_to_path"].items()
        }

    def get_available_days(self) -> list[int]:
        return sorted(self.day_to_path.keys())

    @property
    def dense_columns(self) -> list[str]:
        """Float columns a model reads as a dense block."""
        return [self.counters_column] if self.counters_column else []

    def _run_preprocessing(self, invalidate_cache: bool = False) -> None:
        days = sorted(self.original_day_to_path.keys())

        enriched_paths = preprocess_counters(
            counters=self.counters,
            counter_columns=self.counter_columns,
            day_to_path=self.original_day_to_path,
            days=days,
            output_dir=self._enriched_dir / "packed",
            output_column=self.counters_column,
            invalidate_cache=invalidate_cache,
        )

        self.day_to_path.update(enriched_paths)
        logger.info("Counter preprocessing completed")

    def create_dataset(self, days: int | list[int]) -> EventDataset:
        if isinstance(days, int):
            days = [days]

        day_files = []
        for day in days:
            assert day in self.day_to_path, f"Day {day} not found in available days"
            day_files.append(self.day_to_path[day])

        return EventDataset(
            parquet_files=day_files,
            columns=self.columns + self.dense_columns,
            timestamp_column=self.timestamp_column,
        )

    def create_dataloader(
        self,
        days: int | list[int] | None = None,
        batch_size: int = 256,
        shuffle: bool = False,
        num_workers: int = 4,
        prefetch_factor: int | None = 4,
        collate_fn: Callable | None = collate_event_batch,
        dataset: Dataset | None = None,
        sampler: Sampler | None = None,
        pin_memory: bool = True,
    ) -> DataLoader:
        if dataset is None:
            assert days is not None, "Either days or dataset must be provided"
            dataset = self.create_dataset(days)
        assert sampler is None or not shuffle, (
            "a sampler already decides the order; shuffle=True cannot be honoured"
        )

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            # A worker re-reads its bucket parquet on every spawn, so letting
            # them die between epochs costs that read once per epoch per loader.
            persistent_workers=num_workers > 0,
            # Torch rejects a prefetch factor without workers to prefetch with.
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            # Polars' thread pool does not survive a fork, and torch's
            # default start method is fork.
            multiprocessing_context="forkserver" if num_workers > 0 else None,
        )

    def load_state_dict(self, state_dict: dict[str, Any]):
        super().load_state_dict(state_dict)
        self.metadata_file = Path(self.metadata_file)
        self._load_metadata(validate_schema=False)
