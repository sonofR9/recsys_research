import mmap
import pickle
from pathlib import Path

import numpy as np
import torch


class EmbeddingCache:
    def __init__(self, binary_dir: str | Path):
        binary_dir = Path(binary_dir)

        with open(binary_dir / "embeddings_metadata.pkl", "rb") as f:
            metadata = pickle.load(f)

            self._embedding_dim = metadata["embedding_dim"]
            dtype_str = metadata["dtype"]

        if dtype_str == "float32":
            self.dtype = torch.float32
            self.np_dtype = np.float32
        elif dtype_str == "float64":
            self.dtype = torch.float64
            self.np_dtype = np.float64
        else:
            self.dtype = torch.float32
            self.np_dtype = np.float32

        self.bytes_per_embedding = (
            self._embedding_dim * np.dtype(self.np_dtype).itemsize
        )

        with open(binary_dir / "id_to_offset.pkl", "rb") as f:
            self.id_to_offset: dict[int, int] = pickle.load(f)

        self.file_handle = open(binary_dir / "embeddings.bin", "rb")
        self.mmap_handle = mmap.mmap(
            self.file_handle.fileno(), 0, access=mmap.ACCESS_READ
        )

        self.default_embedding = torch.zeros(self._embedding_dim, dtype=self.dtype)

    @property
    def embedding_dim(self):
        return self._embedding_dim

    def get_embedding(self, item_id: int) -> torch.Tensor:
        offset = self.id_to_offset.get(item_id)
        if offset is None:
            return self.default_embedding

        self.mmap_handle.seek(offset)
        emb_bytes = self.mmap_handle.read(self.bytes_per_embedding)
        emb_np = np.frombuffer(emb_bytes, dtype=self.np_dtype)

        return torch.from_numpy(emb_np).clone()

    def __del__(self):
        if hasattr(self, "mmap_handle"):
            self.mmap_handle.close()
        if hasattr(self, "file_handle"):
            self.file_handle.close()
