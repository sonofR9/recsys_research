"""
Binary format converter for efficient multi-process data loading.
"""

import struct
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pyarrow.parquet as pq
import pickle
from datetime import datetime
import duckdb


def _infer_schema_from_parquet(
    parquet_path: Path, columns: Optional[List[str]] = None
) -> List[Tuple[str, str]]:
    """Infer schema from parquet file."""
    table = pq.read_table(parquet_path)
    schema = []

    cols_to_process = columns if columns else table.column_names

    for col_name in cols_to_process:
        if col_name not in table.column_names:
            raise ValueError(f"Column {col_name} not found in parquet file")

        pa_type = table.schema.field(col_name).type

        if pa_type == pq.lib.int64():
            dtype = "int64"
        elif pa_type == pq.lib.int32():
            dtype = "int32"
        elif pa_type == pq.lib.int8():
            dtype = "int8"
        elif pa_type == pq.lib.float32():
            dtype = "float32"
        elif pa_type == pq.lib.float64():
            dtype = "float64"
        elif str(pa_type).startswith("list"):
            dtype = "list_float32"
        else:
            raise ValueError(f"Unsupported type {pa_type} for column {col_name}")

        schema.append((col_name, dtype))

    return schema


def _dtype_size(dtype: str) -> int:
    """Return size in bytes for a dtype."""
    sizes = {
        "int64": 8,
        "int32": 4,
        "int8": 1,
        "float32": 4,
        "float64": 8,
    }
    if dtype.startswith("list"):
        raise ValueError(f"Cannot get size for list dtype: {dtype}")
    return sizes[dtype]


def _pack_value(value, dtype: str) -> bytes:
    """Pack a single value into bytes based on dtype."""
    if value is None:
        value = 0

    if dtype == "int64":
        return struct.pack("<q", int(value))
    elif dtype == "int32":
        return struct.pack("<i", int(value))
    elif dtype == "int8":
        return struct.pack("<b", int(value))
    elif dtype == "float32":
        return struct.pack("<f", float(value))
    elif dtype == "float64":
        return struct.pack("<d", float(value))
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def convert_to_binary(
    parquet_path: str | Path,
    output_dir: str | Path,
    mode: str = "main",
    id_column: Optional[str] = None,
    embedding_column: Optional[str] = None,
    sort_column: Optional[str] = None,
    day_column: Optional[str] = None,
    columns: Optional[List[str]] = None,
    invalidate_cache: bool = False,
) -> Dict:
    """
    Universal converter for parquet to binary format.

    Args:
        parquet_path: Path to input parquet file
        output_dir: Directory to store binary files
        mode: 'embeddings' or 'main'
        id_column: Column name for ID mapping (embeddings mode)
        embedding_column: Column name for embeddings (embeddings mode)
        sort_column: Column to sort by (main mode)
        day_column: Column to extract day from (main mode)
        columns: List of columns to include (main mode, None = all)
        invalidate_cache: Force rebuild

    Returns:
        Metadata dict
    """
    parquet_path = Path(parquet_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_pkl = output_dir / f"{mode}_metadata.pkl"

    if not invalidate_cache and metadata_pkl.exists():
        print(f"Binary {mode} cache exists, loading metadata...")
        with open(metadata_pkl, "rb") as f:
            return pickle.load(f)

    print(f"Converting {parquet_path.name} to binary format (mode={mode})...")

    if mode == "embeddings":
        return _convert_embeddings(
            parquet_path, output_dir, id_column, embedding_column
        )
    elif mode == "main":
        return _convert_main(parquet_path, output_dir, sort_column, day_column, columns)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def _convert_embeddings(
    parquet_path: Path,
    output_dir: Path,
    id_column: str,
    embedding_column: str,
) -> Dict:
    """Convert embeddings parquet to binary format."""
    embeddings_bin = output_dir / "embeddings.bin"
    id_to_offset_pkl = output_dir / "id_to_offset.pkl"
    metadata_pkl = output_dir / "embeddings_metadata.pkl"

    table = pq.read_table(parquet_path)

    item_ids = table[id_column].to_numpy()
    embeddings = np.stack([row.as_py() for row in table[embedding_column]])

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


def _convert_main(
    parquet_path: Path,
    output_dir: Path,
    sort_column: Optional[str],
    day_column: Optional[str],
    columns: Optional[List[str]],
) -> Dict:
    """Convert main data parquet to binary format with DuckDB sorting."""
    data_bin = output_dir / "main_data.bin"
    day_offsets_pkl = output_dir / "day_offsets.pkl"
    metadata_pkl = output_dir / "main_metadata.pkl"

    schema = _infer_schema_from_parquet(parquet_path, columns)

    row_size = sum(
        _dtype_size(dtype) for _, dtype in schema if not dtype.startswith("list")
    )

    con = duckdb.connect(":memory:")

    if sort_column:
        query = f"SELECT * FROM read_parquet('{parquet_path}') ORDER BY {sort_column}"
    else:
        query = f"SELECT * FROM read_parquet('{parquet_path}')"

    result = con.execute(query)

    day_offsets = {}
    current_day = None
    row_offset = 0

    with open(data_bin, "wb") as f:
        while True:
            batch = result.fetchmany(10000)
            if not batch:
                break

            for row in batch:
                row_data = _pack_row(row, schema)
                f.write(row_data)

                if day_column:
                    day_value = row[
                        schema.index(
                            (
                                day_column,
                                next(dt for col, dt in schema if col == day_column),
                            )
                        )
                    ]
                    day = _timestamp_to_day(day_value)

                    if day != current_day:
                        if current_day is not None:
                            day_offsets[current_day]["end_offset"] = row_offset
                        day_offsets[day] = {
                            "start_offset": row_offset,
                            "end_offset": None,
                            "timestamp": int(day_value),
                        }
                        current_day = day

                row_offset += 1

        if current_day is not None:
            day_offsets[current_day]["end_offset"] = row_offset

    con.close()

    if day_offsets:
        with open(day_offsets_pkl, "wb") as f:
            pickle.dump(day_offsets, f)

    metadata = {
        "row_size": row_size,
        "num_rows": row_offset,
        "schema": schema,
        "num_days": len(day_offsets) if day_offsets else 0,
        "created_at": datetime.now().isoformat(),
    }

    with open(metadata_pkl, "wb") as f:
        pickle.dump(metadata, f)

    print(
        f"Converted {row_offset} rows across {len(day_offsets)} days (row_size={row_size} bytes)"
    )
    return metadata


def _pack_row(row: tuple, schema: List[Tuple[str, str]]) -> bytes:
    """Pack a row into bytes."""
    row_bytes = b""
    for i, (col_name, dtype) in enumerate(schema):
        if dtype.startswith("list"):
            continue
        value = row[i]
        row_bytes += _pack_value(value, dtype)
    return row_bytes


def _timestamp_to_day(timestamp: int) -> str:
    """Convert timestamp to day string (YYYY-MM-DD)."""
    dt = datetime.fromtimestamp(timestamp)
    return dt.strftime("%Y-%m-%d")


def convert_datasets(
    embeddings_parquet: str | Path,
    main_parquet: str | Path,
    output_dir: str | Path,
    counter_columns: Optional[List[str]] = None,
    invalidate_cache: bool = False,
) -> Tuple[Dict, Dict]:
    """
    Convert both parquet files to binary format.

    Returns:
        (embeddings_metadata, main_metadata)
    """
    output_dir = Path(output_dir)

    emb_metadata = convert_to_binary(
        parquet_path=embeddings_parquet,
        output_dir=output_dir / "embeddings",
        mode="embeddings",
        id_column="item_id",
        embedding_column="item_embedding",
        invalidate_cache=invalidate_cache,
    )

    main_columns = [
        "item_id",
        "user_id",
        "album_id",
        "artist_id",
        "event_type",
        "listen_share",
        "target",
        "timestamp",
    ]
    if counter_columns:
        main_columns.extend(counter_columns)

    main_metadata = convert_to_binary(
        parquet_path=main_parquet,
        output_dir=output_dir / "main",
        mode="main",
        sort_column="timestamp",
        day_column="timestamp",
        columns=main_columns,
        invalidate_cache=invalidate_cache,
    )

    main_metadata["counter_columns"] = counter_columns or []

    return emb_metadata, main_metadata
