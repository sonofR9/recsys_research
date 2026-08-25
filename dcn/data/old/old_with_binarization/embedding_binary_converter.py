import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


def convert_embeddings_to_binary(
    embeddings_parquet: str | Path,
    output_dir: str | Path,
    id_column: str = "item_id",
    embedding_column: str = "item_embedding",
    invalidate_cache: bool = False,
) -> dict[str, Any]:
    embeddings_parquet = Path(embeddings_parquet)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    embeddings_bin = output_dir / "embeddings.bin"
    id_to_offset_pkl = output_dir / "id_to_offset.pkl"
    metadata_pkl = output_dir / "embeddings_metadata.pkl"

    if not invalidate_cache and metadata_pkl.exists():
        print("Binary embeddings cache exists, loading metadata...")
        with open(metadata_pkl, "rb") as f:
            return pickle.load(f)

    print(f"Converting {embeddings_parquet.name} to binary format...")

    table = pq.read_table(embeddings_parquet)

    item_ids: np.ndarray = table[id_column].to_numpy()
    embeddings: np.ndarray = table[embedding_column].to_numpy()
    # embeddings = np.vstack(embeddings)

    embedding_dim = embeddings.shape[1]
    dtype = embeddings.dtype

    id_to_offset = {}

    with open(embeddings_bin, "wb") as f:
        for item_id, emb in zip(item_ids, embeddings):
            offset = f.tell()
            id_to_offset[int(item_id)] = offset
            f.write(emb.tobytes())

    with open(id_to_offset_pkl, "wb") as f:
        pickle.dump(id_to_offset, f)

    metadata = {
        "embedding_dim": int(embedding_dim),
        "dtype": str(dtype),
        "num_embeddings": len(item_ids),
        "created_at": datetime.now().isoformat(),
    }

    with open(metadata_pkl, "wb") as f:
        pickle.dump(metadata, f)

    print(f"Converted {len(item_ids)} embeddings (dim={embedding_dim})")
    return metadata
