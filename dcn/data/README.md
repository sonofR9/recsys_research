# Data Module

Data loading and preprocessing for the multi-task recommender system.

## Quick Start

```python
from data import DatasetManager

manager = DatasetManager(
    main_parquet="data/main.parquet",
    embeddings_parquet="data/embeddings.parquet",
    data_dir="data/processed",
    counter_columns=['user_likes_count', 'item_popularity'],
)

dataset = manager.create_dataset(days=0)
dataloader = manager.create_dataloader(dataset, batch_size=256)

for batch in dataloader:
    # batch contains: item_id, uid, album_id, artist_id,
    # item_embedding, counters, target_like, target_listen, timestamp
    pass
```

## API Reference

### DatasetManager

Main interface for data operations.

**Constructor:**
```python
DatasetManager(
    main_parquet: str | Path,
    embeddings_parquet: str | Path,
    data_dir: str | Path,
    counter_columns: List[str],
    invalidate_cache: bool = False,
)
```

Data preparation happens automatically in `__init__`.

**Methods:**

- `get_available_days() -> List[int]` - Get list of available day IDs
- `create_dataset(days: int | List[int]) -> DayDataset` - Create dataset for specific day(s)
- `create_dataloader(dataset, batch_size=256, shuffle=False, num_workers=4) -> DataLoader` - Create DataLoader

### Batch Format

Each batch is a dictionary with:
- `item_id`: torch.Tensor (int64)
- `uid`: torch.Tensor (int64)
- `album_id`: torch.Tensor (int64)
- `artist_id`: torch.Tensor (int64)
- `item_embedding`: torch.Tensor (float32)
- `counters`: torch.Tensor (float32)
- `target_like`: torch.Tensor (float32)
- `target_listen`: torch.Tensor (float32)
- `timestamp`: torch.Tensor (int64)

## Day-by-Day Training

```python
manager = DatasetManager(...)

for day in manager.get_available_days():
    dataset = manager.create_dataset(days=day)
    dataloader = manager.create_dataloader(dataset, shuffle=False)
    
    for batch in dataloader:
        # train on this day
        pass
```

## File Structure

After initialization, `data_dir` contains:
```
data_dir/
├── days/
│   ├── day_0000.parquet
│   ├── day_0001.parquet
│   └── ...
├── embeddings/
│   ├── embeddings.bin
│   ├── id_to_offset.pkl
│   └── embeddings_metadata.pkl
└── metadata.yaml
```

## See Also

- `example_usage.py` - Usage examples
- `README_DEVELOPMENT.md` - Implementation details