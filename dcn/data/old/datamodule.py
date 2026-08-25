"""
PyTorch Lightning DataModule for binary format data.
"""

from pathlib import Path
from typing import Optional, List
import pickle

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from .binary_converter import convert_datasets
from .embedding_cache import EmbeddingCache
from .dataset import RecommenderDataset, collate_batch


class RecommenderDataModule(LightningDataModule):
    def __init__(
        self,
        embeddings_parquet: str | Path,
        main_parquet: str | Path,
        counter_columns: List[str],
        binary_data_dir: str | Path = "binary_data",
        batch_size: int = 256,
        num_workers: int = 4,
        val_days: int = 7,
        invalidate_cache: bool = False,
    ):
        super().__init__()
        self.embeddings_parquet = Path(embeddings_parquet)
        self.main_parquet = Path(main_parquet)
        self.counter_columns = counter_columns
        self.binary_data_dir = Path(binary_data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_days = val_days
        self.invalidate_cache = invalidate_cache

    def prepare_data(self):
        convert_datasets(
            embeddings_parquet=self.embeddings_parquet,
            main_parquet=self.main_parquet,
            output_dir=self.binary_data_dir,
            counter_columns=self.counter_columns,
            invalidate_cache=self.invalidate_cache,
        )

    def setup(self, stage: Optional[str] = None):
        embeddings_dir = self.binary_data_dir / "embeddings"
        main_dir = self.binary_data_dir / "main"

        day_offsets_pkl = main_dir / "day_offsets.pkl"
        with open(day_offsets_pkl, "rb") as f:
            day_offsets = pickle.load(f)

        days_sorted = sorted(day_offsets.keys())

        if len(days_sorted) <= self.val_days:
            raise ValueError(
                f"Not enough days ({len(days_sorted)}) for validation "
                f"(val_days={self.val_days}). Need at least {self.val_days + 1} days."
            )

        train_start_day = days_sorted[0]
        train_end_day = days_sorted[-(self.val_days + 1)]
        val_start_day = days_sorted[-self.val_days]
        val_end_day = days_sorted[-1]

        embedding_cache = EmbeddingCache(embeddings_dir)

        if stage == "fit" or stage is None:
            self.train_dataset = RecommenderDataset(
                binary_dir=main_dir,
                embedding_cache=embedding_cache,
                start_day=train_start_day,
                end_day=train_end_day,
            )

            self.val_dataset = RecommenderDataset(
                binary_dir=main_dir,
                embedding_cache=embedding_cache,
                start_day=val_start_day,
                end_day=val_end_day,
            )

            print(
                f"Train: {train_start_day} to {train_end_day} ({len(self.train_dataset)} samples)"
            )
            print(
                f"Val: {val_start_day} to {val_end_day} ({len(self.val_dataset)} samples)"
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_batch,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_batch,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )
