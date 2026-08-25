"""A yambda_data directory small enough to train on inside a test."""

from pathlib import Path
from typing import Any

import polars as pl
import torch

from dcn.config import (
    CheckpointConfig,
    DataloaderConfig,
    DayRangeConfig,
    LoggingConfig,
    RuntimeConfig,
    SemanticIdConfig,
)
from dcn.datasets.yambda import UserSample

SECONDS_IN_DAY = 86_400
NUM_USERS = 2
NUM_ITEMS = 5
NUM_DAYS = 2

# One day of one user, as (event_type, item offset, played_ratio_pct). Two likes
# a day, so a likes-only variant has sequences to learn from; the first listen
# repeats the track just liked and is played to the end, the second is a
# different track and a skip, so the two disagree on both ranking targets.
DAY_PATTERN = [
    ("like", 0, 0),
    ("listen", 0, 100),
    ("like", 1, 0),
    ("listen", 2, 20),
]
EVENTS_PER_USER_PER_DAY = len(DAY_PATTERN)


def write_miniature_yambda(base_path: Path) -> Path:
    data_path = base_path / "yambda_data"
    (data_path / "flat" / "50m").mkdir(parents=True)

    rows = []
    for user in range(1, NUM_USERS + 1):
        for day in range(NUM_DAYS):
            for position, (event_type, item_offset, played) in enumerate(DAY_PATTERN):
                rows.append(
                    {
                        "uid": user,
                        "timestamp": day * SECONDS_IN_DAY + position * 60 + user,
                        "item_id": 1 + (user + day + item_offset) % NUM_ITEMS,
                        "is_organic": 0,
                        "played_ratio_pct": played,
                        "track_length_seconds": 180,
                        "event_type": event_type,
                    }
                )
    pl.DataFrame(rows).write_parquet(data_path / "flat" / "50m" / "multi_event.parquet")

    items = list(range(1, NUM_ITEMS + 1))
    pl.DataFrame({"item_id": items, "artist_id": [i % 3 for i in items]}).write_parquet(
        data_path / "artist_item_mapping.parquet"
    )
    pl.DataFrame({"item_id": items, "album_id": [i % 4 for i in items]}).write_parquet(
        data_path / "album_item_mapping.parquet"
    )
    pl.DataFrame(
        {
            "item_id": items,
            "normalized_embed": [[float(i), float(i % 2), 0.5, -0.5] for i in items],
        }
    ).write_parquet(data_path / "embeddings.parquet")

    return base_path


def configure(experiment_class: type, base_path: Path, **overrides: Any):
    """A variant shrunk to whatever still exercises its whole stack."""
    settings = {
        "run_name": f"test_{experiment_class.__name__}",
        "base_path": base_path,
        "user_sample": UserSample(max_users=NUM_USERS),
        "num_epochs": 1,
        "max_seq_len": 4,
        "min_seq_len": 2,
        "validation_days": 1,
        "day_range": DayRangeConfig(start_day=0, end_day=NUM_DAYS),
        "dataloader": DataloaderConfig(
            batch_size=NUM_USERS,
            val_batch_size=NUM_USERS,
            num_workers=0,
            prefetch_factor=None,
        ),
        "runtime": RuntimeConfig(dtype=torch.bfloat16, compile=False),
        "logging": LoggingConfig(enable_predictions=False, log_interval=1),
        "checkpointing": CheckpointConfig(load_checkpoint=False),
    }
    return experiment_class(**{**settings, **overrides})


def semantic_overrides(**overrides: Any) -> dict:
    return {"semantic": SemanticIdConfig(num_levels=2, num_codes=3, **overrides)}
