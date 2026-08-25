from pathlib import Path

import polars as pl
from tqdm import tqdm

from .counters import EmaCounter

COUNTERS_COLUMN = "counters"


def preprocess_counters(
    counters: list[EmaCounter],
    counter_columns: list[str],
    day_to_path: dict[int, Path],
    days: list[int],
    output_dir: Path,
    output_column: str = COUNTERS_COLUMN,
    invalidate_cache: bool = False,
) -> dict[int, Path]:
    if not counters:
        return {}

    output_dir.mkdir(parents=True, exist_ok=True)
    packed = (
        pl.concat_list(pl.col(column).cast(pl.Float32) for column in counter_columns)
        .list.to_array(len(counter_columns))
        .alias(output_column)
    )

    enriched_day_to_path: dict[int, Path] = {}
    total_iterations = len(days) * len(counters)

    with tqdm(total=total_iterations, desc="Processing counters") as pbar:
        for day in sorted(days):
            day_df = pl.read_parquet(day_to_path[day])

            for counter_idx, counter in enumerate(counters, 1):
                pbar.set_postfix(day=day, counter=f"{counter_idx}/{len(counters)}")
                day_df = counter.process_day(day, day_df, invalidate_cache)
                pbar.update(1)

            output_path = output_dir / f"day_{day:04d}.parquet"
            enriched_day_to_path[day] = output_path
            if invalidate_cache or not output_path.exists():
                day_df.with_columns(packed).write_parquet(output_path)

    return enriched_day_to_path
