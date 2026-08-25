"""
PyTorch Dataset for binary format with O(1) offset-based access.
"""

import mmap
import pickle
import struct
from pathlib import Path
from typing import Dict, List, Optional
import torch
from torch.utils.data import Dataset

from .embedding_cache import EmbeddingCache


class RecommenderDataset(Dataset):
    def __init__(
        self,
        binary_dir: str | Path,
        embedding_cache: EmbeddingCache,
        start_timestamp: Optional[int] = None,
        end_timestamp: Optional[int] = None,
        start_day: Optional[str] = None,
        end_day: Optional[str] = None,
    ):
        binary_dir = Path(binary_dir)
        self.embedding_cache = embedding_cache

        data_bin = binary_dir / "main_data.bin"
        day_offsets_pkl = binary_dir / "day_offsets.pkl"
        metadata_pkl = binary_dir / "main_metadata.pkl"

        with open(metadata_pkl, "rb") as f:
            metadata = pickle.load(f)

        self.row_size = metadata["row_size"]
        schema = metadata["schema"]
        counter_columns = metadata.get("counter_columns", [])

        self.schema_dict = {
            name: (idx, dtype) for idx, (name, dtype) in enumerate(schema)
        }
        self.counter_columns = counter_columns

        unpack_format = "<"
        for _, dtype in schema:
            if dtype == "int64":
                unpack_format += "q"
            elif dtype == "int32":
                unpack_format += "i"
            elif dtype == "int8":
                unpack_format += "b"
            elif dtype == "float32":
                unpack_format += "f"
            elif dtype == "float64":
                unpack_format += "d"
        self.unpack_format = unpack_format
        self.schema = schema

        with open(day_offsets_pkl, "rb") as f:
            day_offsets = pickle.load(f)

        file_handle = open(data_bin, "rb")
        self.mmap_handle = mmap.mmap(file_handle.fileno(), 0, access=mmap.ACCESS_READ)
        self.file_handle = file_handle

        self.valid_indices = self._build_index(
            data_bin, day_offsets, start_timestamp, end_timestamp, start_day, end_day
        )

    def _build_index(
        self,
        data_bin: Path,
        day_offsets: Dict,
        start_timestamp: Optional[int],
        end_timestamp: Optional[int],
        start_day: Optional[str],
        end_day: Optional[str],
    ) -> List[int]:
        if start_day is not None or end_day is not None:
            days_sorted = sorted(day_offsets.keys())
            start_offset = 0
            end_offset = data_bin.stat().st_size // self.row_size

            if start_day is not None and start_day in day_offsets:
                start_offset = day_offsets[start_day]["start_offset"]

            if end_day is not None and end_day in day_offsets:
                end_offset = day_offsets[end_day]["end_offset"]

            return list(range(start_offset, end_offset))

        elif start_timestamp is not None or end_timestamp is not None:
            valid_indices = []
            total_rows = data_bin.stat().st_size // self.row_size

            for idx in range(total_rows):
                ts = self._read_timestamp(idx)

                if start_timestamp is not None and ts < start_timestamp:
                    continue
                if end_timestamp is not None and ts >= end_timestamp:
                    continue

                valid_indices.append(idx)

            return valid_indices
        else:
            total_rows = data_bin.stat().st_size // self.row_size
            return list(range(total_rows))

    def _read_timestamp(self, idx: int) -> int:
        ts_field_idx, ts_dtype = self.schema_dict["timestamp"]

        offset = 0
        for i in range(ts_field_idx):
            _, dtype = self.schema[i]
            offset += self._dtype_size(dtype)

        byte_offset = idx * self.row_size + offset
        self.mmap_handle.seek(byte_offset)
        ts_bytes = self.mmap_handle.read(8)
        return struct.unpack("<q", ts_bytes)[0]

    def _dtype_size(self, dtype: str) -> int:
        sizes = {
            "int64": 8,
            "int32": 4,
            "int8": 1,
            "float32": 4,
            "float64": 8,
        }
        return sizes[dtype]

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Dict:
        actual_idx = self.valid_indices[idx]

        byte_offset = actual_idx * self.row_size
        self.mmap_handle.seek(byte_offset)
        row_bytes = self.mmap_handle.read(self.row_size)

        values = struct.unpack(self.unpack_format, row_bytes)

        row = {}
        for i, (col_name, dtype) in enumerate(self.schema):
            value = values[i]

            if dtype.startswith("int"):
                row[col_name] = int(value)
            elif dtype.startswith("float"):
                row[col_name] = float(value)
            else:
                row[col_name] = value

        item_embedding = self.embedding_cache.get_embedding(row["item_id"])

        counters = (
            torch.tensor(
                [row[col] for col in self.counter_columns], dtype=torch.float32
            )
            if self.counter_columns
            else torch.tensor([], dtype=torch.float32)
        )

        target_like = 1 if row["event_type"] == 1 else 0
        target_listen = row["listen_share"]

        return {
            "item_id": row["item_id"],
            "user_id": row["user_id"],
            "album_id": row["album_id"],
            "artist_id": row["artist_id"],
            "item_embedding": item_embedding,
            "counters": counters,
            "target_like": target_like,
            "target_listen": target_listen,
            "timestamp": row["timestamp"],
        }

    def __del__(self):
        if hasattr(self, "mmap_handle"):
            self.mmap_handle.close()
        if hasattr(self, "file_handle"):
            self.file_handle.close()


def collate_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    return {
        "item_id": torch.tensor([x["item_id"] for x in batch], dtype=torch.long),
        "user_id": torch.tensor([x["user_id"] for x in batch], dtype=torch.long),
        "album_id": torch.tensor([x["album_id"] for x in batch], dtype=torch.long),
        "artist_id": torch.tensor([x["artist_id"] for x in batch], dtype=torch.long),
        "item_embedding": torch.stack([x["item_embedding"] for x in batch]),
        "counters": torch.stack([x["counters"] for x in batch])
        if batch[0]["counters"].numel() > 0
        else torch.empty(len(batch), 0),
        "target_like": torch.tensor(
            [x["target_like"] for x in batch], dtype=torch.float32
        ),
        "target_listen": torch.tensor(
            [x["target_listen"] for x in batch], dtype=torch.float32
        ),
        "timestamp": torch.tensor([x["timestamp"] for x in batch], dtype=torch.long),
    }
