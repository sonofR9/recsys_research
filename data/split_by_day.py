import os
from pathlib import Path

from tqdm import tqdm

from utils.global_config import config

from .utils import setup_duckdb_connection


def split_main_parquet_by_day(
    main_parquet: Path,
    output_dir: Path,
    invalidate_cache: bool,
) -> dict[int, Path]:
    """Split main parquet file by day into separate files.

    The input parquet must have a 'day' column (DATE type).

    Args:
        main_parquet: Path to the input parquet file
        output_dir: Directory to write day-split files
        invalidate_cache: If True, reprocess all files. If False, skip existing files.

    Returns:
        Mapping of day_index -> parquet_path
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = config.tmp_path
    os.makedirs(temp_dir, exist_ok=True)

    conn = setup_duckdb_connection(temp_dir, "split_by_day.duckdb")
    try:
        days_result = conn.execute(
            """
            SELECT DISTINCT day
            FROM read_parquet(?)
            ORDER BY day
        """,
            [str(main_parquet)],
        ).fetchall()

        unique_days = [row[0] for row in days_result]

        day_to_path = {}
        skipped = 0
        processed = 0

        for day_idx, day_value in tqdm(
            enumerate(unique_days), total=len(unique_days), desc="Splitting by day"
        ):
            output_file = output_dir / f"day_{day_idx:04d}.parquet"

            if not invalidate_cache and output_file.exists():
                day_to_path[day_idx] = output_file
                skipped += 1
                continue

            conn.execute(
                f"""
                COPY (
                    SELECT * EXCLUDE (day)
                    FROM read_parquet('{main_parquet}')
                    WHERE day = ?
                ) TO '{output_file}' (FORMAT PARQUET)
            """,
                [day_value],
            )

            day_to_path[day_idx] = output_file
            processed += 1

        print(f"Processed: {processed}, Skipped: {skipped}")
    finally:
        conn.close()

    return day_to_path
