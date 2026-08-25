# Binary Format Documentation

## Overview

This document describes the custom binary format used for efficient multi-process data loading in the recommender system. The format is optimized for:

1. **Multi-process friendly**: Uses memory-mapped files that can be shared across processes
2. **O(1) random access**: Fixed-width rows allow direct offset calculation
3. **Day-based indexing**: Pre-computed offsets for each day enable efficient validation splits
4. **Persistent caching**: Binary files are created once and reused across runs

## Architecture

### Embeddings Format

**Files:**
- `embeddings.bin` - Binary file containing all embeddings
- `id_to_offset.pkl` - Pickle file mapping item_id → byte offset
- `embeddings_metadata.pkl` - Metadata (embedding_dim, dtype, etc.)

**Structure:**
```
embeddings.bin:
[embedding_0][embedding_1][embedding_2]...

Each embedding is embedding_dim * sizeof(dtype) bytes
Offset for item_id is stored in id_to_offset mapping
```

**Access Pattern:**
1. Load `id_to_offset` mapping into memory (small, ~8 bytes per item)
2. Memory-map `embeddings.bin` file
3. For item_id: `offset = id_to_offset[item_id]`
4. Seek to offset and read `embedding_dim * sizeof(dtype)` bytes

### Main Data Format

**Files:**
- `main_data.bin` - Binary file with fixed-width rows
- `day_offsets.pkl` - Pickle file with day → (start_offset, end_offset) mapping
- `main_metadata.pkl` - Metadata (row_size, schema, etc.)

**Schema:**
```python
[
    ('item_id', 'int64'),      # 8 bytes
    ('user_id', 'int64'),      # 8 bytes
    ('album_id', 'int64'),     # 8 bytes
    ('artist_id', 'int64'),    # 8 bytes
    ('event_type', 'int8'),    # 1 byte
    ('listen_share', 'float32'), # 4 bytes
    ('target', 'int8'),        # 1 byte
    ('timestamp', 'int64'),    # 8 bytes
    ('counter_1', 'float32'),  # 4 bytes (if present)
    ('counter_2', 'float32'),  # 4 bytes (if present)
    ...
]
```

**Structure:**
```
main_data.bin:
[row_0][row_1][row_2]...

Each row is row_size bytes (sum of all field sizes)
Rows are sorted by timestamp
Offset for row idx: idx * row_size
```

**Day Offsets:**
```python
{
    '2024-01-01': {
        'start_offset': 0,
        'end_offset': 1000,
        'timestamp': 1704067200
    },
    '2024-01-02': {
        'start_offset': 1000,
        'end_offset': 2500,
        'timestamp': 1704153600
    },
    ...
}
```

## Conversion Process

### One-Time Setup

```python
from data import convert_datasets

# Convert parquet to binary format (only once)
emb_metadata, main_metadata = convert_datasets(
    embeddings_parquet="embeddings.parquet",
    main_parquet="main.parquet",
    output_dir="binary_data",
    counter_columns=['counter_1', 'counter_2'],
    invalidate_cache=False,  # Set to True to rebuild
)
```

### Subsequent Runs

Binary files are automatically detected and reused. No conversion happens unless:
- Files don't exist
- `invalidate_cache=True` is passed

## Usage

### EmbeddingCache

```python
from data import EmbeddingCache

# Each process creates its own instance
# But they all share the same memory-mapped file
cache = EmbeddingCache("binary_data/embeddings")

# O(1) lookup
emb = cache.get_embedding(item_id=42)

# Batch lookup
embs = cache.get_embeddings_batch([1, 2, 3, 999999])
# Missing IDs return default zero embedding
```

### RecommenderDataset

```python
from data import RecommenderDataset, EmbeddingCache

cache = EmbeddingCache("binary_data/embeddings")

# Full dataset
dataset = RecommenderDataset(
    binary_dir="binary_data/main",
    embedding_cache=cache,
)

# Filter by day range
dataset = RecommenderDataset(
    binary_dir="binary_data/main",
    embedding_cache=cache,
    start_day="2024-01-01",
    end_day="2024-01-10",
)

# O(1) access
sample = dataset[0]
```

### RecommenderDataModule

```python
from data import RecommenderDataModule

# Standard train/val split
datamodule = RecommenderDataModule(
    binary_data_dir="binary_data",
    batch_size=256,
    num_workers=4,
    val_split_ratio=0.1,
)

datamodule.setup()
train_loader = datamodule.train_dataloader()
val_loader = datamodule.val_dataloader()

# Day-by-day iteration
for day, dataset in datamodule.create_day_datasets():
    loader = datamodule.get_day_dataloader(day)
    # Train on this day
```

## Multi-Process Behavior

### Memory Mapping

Each DataLoader worker process:
1. Creates its own `EmbeddingCache` instance
2. Opens its own file handle to `embeddings.bin`
3. Creates its own memory mapping

**Important:** While each process has its own mapping, the OS shares the underlying physical memory pages. This means:
- First process loads pages into RAM
- Subsequent processes reuse the same physical pages
- Total RAM usage ≈ file size (not file_size × num_workers)

### Dataset Instances

Each worker also creates its own:
- `RecommenderDataset` instance
- Memory mapping of `main_data.bin`
- File handle

Again, OS shares physical pages across processes.

## Performance Characteristics

### Embeddings

- **Lookup time**: O(1) - dictionary lookup + memory read
- **Memory per process**: ~8 bytes per item (id_to_offset mapping)
- **Shared memory**: embedding file size (e.g., 10GB for 1M items × 128 dims × float32)

### Main Data

- **Access time**: O(1) - offset calculation + memory read
- **Memory per process**: Minimal (just day_offsets mapping)
- **Shared memory**: main data file size

### Conversion Time

- **Embeddings**: ~1-2 minutes for 1M embeddings
- **Main data**: ~5-10 minutes for 100M rows (depends on sorting)
- **Subsequent runs**: 0 seconds (files are cached)

## Comparison with Parquet

| Aspect | Parquet | Binary Format |
|--------|---------|---------------|
| Random access | Slow (row group based) | O(1) with offset |
| Multi-process | Each worker loads data | Shared memory mapping |
| Memory usage | High (decompression) | Low (direct mapping) |
| Startup time | Fast (no conversion) | Slow first time, instant after |
| Flexibility | High (schema evolution) | Low (fixed schema) |

## Best Practices

1. **First run**: Set `invalidate_cache=False`, let conversion happen once
2. **Schema changes**: Set `invalidate_cache=True` to rebuild
3. **num_workers**: Use 4-8 workers for best throughput
4. **persistent_workers**: Set to `True` to avoid recreation overhead
5. **Day-by-day training**: Use `get_day_dataloader()` for sequential training

## Troubleshooting

### "File not found" errors
- Run conversion first: `convert_datasets(...)`
- Check `binary_data/` directory exists

### High memory usage
- Check if multiple processes are loading id_to_offset
- Verify memory mapping is working (should see shared pages in `top`)

### Slow first access
- Normal - OS loads pages on first access
- Subsequent accesses are fast (pages cached in RAM)

### Incorrect data
- Set `invalidate_cache=True` to rebuild from parquet
- Verify parquet files are correct