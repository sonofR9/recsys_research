import logging
import os
from datetime import datetime
from pathlib import Path

import duckdb
import psutil


def log_memory(msg: str, level: int = logging.INFO) -> None:
    process = psutil.Process(os.getpid())
    mem_gb = process.memory_info().rss / 1024**3
    curr_time = f"{datetime.now().strftime('%H:%M:%S')}"
    full_msg = f"{curr_time} [MEM {mem_gb:.2f} GB] {msg}"
    logging.log(level, full_msg)


def setup_duckdb_connection(
    temp_dir_path: str, db_name: str, read_only: bool = False
) -> duckdb.DuckDBPyConnection:
    db_name = os.path.join(temp_dir_path, db_name)
    con = duckdb.connect(db_name, read_only=read_only)

    log_memory("Connecting to duckdb")
    con.execute(f"PRAGMA temp_directory='{temp_dir_path}';")
    con.execute("SET preserve_insertion_order = false;")
    con.execute("SET enable_progress_bar=false;")
    con.execute("SET memory_limit = '24GB';")
    con.execute("PRAGMA max_temp_directory_size='300GiB'")

    return con


def to_day(timestamp_column: str = "timestamp") -> str:
    """DuckDB expression turning a unix-seconds column into a UTC date."""
    return f"CAST(to_timestamp({timestamp_column}::BIGINT) AT TIME ZONE 'UTC' AS DATE)"


TO_DAY = to_day()


def merge_parquets_duckdb(
    parquet_paths: list[Path],
    output_path: Path,
    temp_dir: str = ".tmp",
) -> None:
    os.makedirs(temp_dir, exist_ok=True)

    with duckdb.connect() as conn:
        conn.execute(f"PRAGMA temp_directory='{temp_dir}';")
        conn.execute("SET memory_limit = '24GB';")

        paths_str = ", ".join(f"'{p}'" for p in parquet_paths)
        conn.execute(f"""
            COPY (
                SELECT * FROM read_parquet([{paths_str}])
            ) TO '{output_path}' (FORMAT PARQUET)
        """)
