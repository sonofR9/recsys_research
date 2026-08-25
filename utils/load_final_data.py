import duckdb
from pathlib import Path
from typing import Union, Optional
import polars as pl
import tempfile
import multiprocessing as mp
import os
from .utils import log_memory


def _subprocess_worker(
    data_source_path: str,
    test_users_path: str,
    output_path: str,
    fstr_dict: dict[str, float],
    timestamp_filter: tuple[Optional[int], Optional[int]],
    user_fraction: float,
    top_n_features: Optional[int],
    user_id_col: str,
    target_col: str,
    timestamp_col: str,
    temp_dir: str,
) -> None:
    """Worker function that runs in subprocess to execute DuckDB query."""

    sorted_features = sorted(fstr_dict.items(), key=lambda x: x[1], reverse=True)

    if top_n_features is not None:
        sorted_features = sorted_features[:top_n_features]

    feature_names = [f[0] for f in sorted_features]

    # Ensure required columns are included
    item_col = "item_id"
    day_column = "day"
    required_cols = [
        user_id_col,
        target_col,
        timestamp_col,
        item_col,
        day_column,
    ]
    for col in required_cols:
        if col not in feature_names:
            feature_names.insert(0, col)

    # Build timestamp filter
    min_ts, max_ts = timestamp_filter
    conditions = []
    if min_ts is not None:
        conditions.append(f"{timestamp_col} >= {min_ts}")
    if max_ts is not None:
        conditions.append(f"{timestamp_col} < {max_ts}")

    ts_conditions = ""
    if conditions:
        ts_conditions = "WHERE " + " AND ".join(conditions)

    log_memory("Starting duckdb query")

    duckdb_temp_dir = os.path.join(temp_dir, f"duckdb_tmp_{os.getpid()}")
    os.makedirs(duckdb_temp_dir, exist_ok=True)

    try:
        with duckdb.connect() as con:
            con.execute("SET enable_progress_bar=false;")
            con.execute("SET memory_limit = '10GB';")
            con.execute("SET preserve_insertion_order = false;")
            con.execute("SET force_compression = 'auto';")

            con.execute(f"SET temp_directory = '{duckdb_temp_dir}';")

            con.execute(
                f"CREATE VIEW source_data AS SELECT * FROM read_parquet('{data_source_path}')"
            )
            con.execute(
                f"CREATE VIEW test_users_tbl AS SELECT * FROM read_parquet('{test_users_path}')"
            )

            source_columns = set(
                row[0] for row in con.execute("DESCRIBE source_data").fetchall()
            )

            calculate_day = day_column not in source_columns
            if calculate_day:
                feature_names.remove(day_column)
            calculate_target = target_col not in source_columns
            if calculate_target:
                feature_names.remove(target_col)

            cols_str = ", ".join(f"d.{col}" for col in feature_names)

            if calculate_day:
                cols_str += f", CAST(EPOCH(date_trunc('day', to_timestamp(d.{timestamp_col}::BIGINT))) / 86400 AS INTEGER) AS {day_column}"
            if calculate_target:
                cols_str += (
                    ", CASE WHEN event_type = 'like' THEN 1.0 "
                    f"ELSE 0.0 END::FLOAT AS {target_col}"
                )

            query = f"""
                WITH filtered_data AS (
                    SELECT *
                    FROM source_data
                    {ts_conditions}
                ),
                
                -- Get distinct non-test users and sample them
                all_non_test_users AS (
                    SELECT DISTINCT d.{user_id_col}
                    FROM filtered_data d
                    ANTI JOIN test_users_tbl t ON d.{user_id_col} = t.{user_id_col}
                ),
                
                sampled_non_test_users AS (
                    SELECT {user_id_col}
                    FROM all_non_test_users
                    USING SAMPLE {user_fraction * 100:.4f}% (bernoulli)
                ),
                
                -- Combine: all test user rows + sampled non-test user rows
                final_data AS (
                    -- All rows for test users
                    SELECT {cols_str}
                    FROM filtered_data d
                    SEMI JOIN test_users_tbl t ON d.{user_id_col} = t.{user_id_col}
                    
                    UNION ALL
                    
                    -- All rows for sampled non-test users
                    SELECT {cols_str}
                    FROM filtered_data d
                    SEMI JOIN sampled_non_test_users s ON d.{user_id_col} = s.{user_id_col}
                )
                
                SELECT * FROM final_data
            """

            con.execute(f"COPY ({query}) TO '{output_path}' (FORMAT PARQUET)")
    finally:
        import shutil

        if os.path.exists(duckdb_temp_dir):
            shutil.rmtree(duckdb_temp_dir, ignore_errors=True)

    log_memory("end duckdb")


def load_filtered_data(
    data_source: Union[str, Path, pl.DataFrame],
    test_users: pl.DataFrame,
    fstr_dict: dict[str, float],
    timestamp_filter: tuple[Optional[int], Optional[int]] = (None, None),
    user_fraction: float = 0.1,
    top_n_features: Optional[int] = None,
    user_id_col: str = "uid",
    target_col: str = "target",
    timestamp_col: str = "timestamp",
    random_seed: int = 42,
    temp_dir: Optional[str] = None,
) -> pl.DataFrame:
    """
    Load and filter data using DuckDB in a subprocess.
    Result is written to a temporary parquet file and loaded in main process.

    Parameters
    ----------
    data_source : Union[str, Path, pl.DataFrame]
        Path to parquet file or Polars DataFrame
    test_users : pl.DataFrame
        DataFrame containing test user IDs to always include
    fstr_dict : dict[str, float]
        Feature importance dictionary {feature_name: importance_score}
    timestamp_filter : tuple[Optional[int], Optional[int]]
        (min_timestamp, max_timestamp) - None means no bound
    user_fraction : float
        Fraction of non-test users to sample (0.0 to 1.0)
    top_n_features : Optional[int]
        Number of top features to select based on fstr_dict. None = all features
    user_id_col : str
        Name of user ID column
    target_col : str
        Name of target column (always included)
    timestamp_col : str
        Name of timestamp column
    random_seed : int
        Random seed for reproducible user sampling
    temp_dir : Optional[str]
        Directory for temporary files. If None, uses current working directory.

    Returns
    -------
    pl.DataFrame
        Filtered Polars DataFrame
    """
    if temp_dir is None:
        temp_dir = os.getcwd()

    os.makedirs(temp_dir, exist_ok=True)

    data_source_path = None
    test_users_path = None
    cleanup_data_source = False
    cleanup_test_users = False

    log_memory("before everything")
    try:
        if isinstance(data_source, pl.DataFrame):
            with tempfile.NamedTemporaryFile(
                suffix=".parquet", dir=temp_dir, delete=False
            ) as f:
                data_source_path = f.name
            data_source.write_parquet(data_source_path)
            cleanup_data_source = True
        else:
            data_source_path = str(data_source)

        with tempfile.NamedTemporaryFile(
            suffix=".parquet", dir=temp_dir, delete=False
        ) as f:
            test_users_path = f.name
        test_users.write_parquet(test_users_path)
        cleanup_test_users = True

        log_memory("after wrote test users etc.")
        with tempfile.NamedTemporaryFile(
            suffix=".parquet", dir=temp_dir, delete=False
        ) as f:
            output_path = f.name

        ctx = mp.get_context("spawn")

        process = ctx.Process(
            target=_subprocess_worker,
            args=(
                data_source_path,
                test_users_path,
                output_path,
                fstr_dict,
                timestamp_filter,
                user_fraction,
                top_n_features,
                user_id_col,
                target_col,
                timestamp_col,
                temp_dir,
            ),
        )

        process.start()
        process.join()

        if process.exitcode != 0:
            raise RuntimeError(f"Subprocess failed with exit code: {process.exitcode}")

        log_memory("sub process end")
        df = pl.read_parquet(output_path)
        log_memory("loaded parquet")
        return df

    finally:
        if (
            cleanup_data_source
            and data_source_path
            and os.path.exists(data_source_path)
        ):
            os.unlink(data_source_path)
        if cleanup_test_users and test_users_path and os.path.exists(test_users_path):
            os.unlink(test_users_path)
        if "output_path" in locals() and os.path.exists(output_path):
            os.unlink(output_path)


def load_train_val_data(
    data_source: Union[str, Path, pl.DataFrame],
    test_users: pl.DataFrame,
    fstr_dict: dict[str, float],
    start_day: int = 0,
    train_end: int = 299,
    val_days: int = 1,
    user_fraction: float = 0.1,
    top_n_features: Optional[int] = None,
    random_seed: int = 42,
    skip_train_data_crutch: bool = False,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Convenience function to load both train and validation sets.

    Parameters
    ----------
    data_source : Union[str, Path, pl.DataFrame]
        Path to parquet file or Polars DataFrame
    test_users : pl.DataFrame
        DataFrame containing test user IDs to always include
    fstr_dict : dict[str, float]
        Feature importance dictionary {feature_name: importance_score}
    start_day : int
        Starting day for training data (inclusive)
    train_end : int
        End day for training data (exclusive), start of validation
    val_days : int
        Number of days for validation data (starts after train_end)
    user_fraction : float
        Fraction of non-test users to sample (0.0 to 1.0)
    top_n_features : Optional[int]
        Number of top features to select based on fstr_dict. None = all features
    random_seed : int
        Random seed for reproducible user sampling
    temp_dir : Optional[str]
        Directory for temporary files. If None, uses current working directory.

    Returns
    -------
    tuple[pl.DataFrame, pl.DataFrame]
        (train_df, val_df) as Polars DataFrames
    """
    user_id_col = "uid"
    target_col = "target"
    timestamp_col = "timestamp"
    temp_dir = ".tmp"
    SECONDS_IN_DAY = 60 * 60 * 24

    start_cutoff = SECONDS_IN_DAY * start_day
    train_cutoff = SECONDS_IN_DAY * train_end
    val_cutoff = SECONDS_IN_DAY * (train_end + val_days)

    common_params = {
        "data_source": data_source,
        "test_users": test_users,
        "fstr_dict": fstr_dict,
        "user_fraction": user_fraction,
        "top_n_features": top_n_features,
        "user_id_col": user_id_col,
        "target_col": target_col,
        "timestamp_col": timestamp_col,
        "random_seed": random_seed,
        "temp_dir": temp_dir,
    }

    train_df = None
    if not skip_train_data_crutch:
        log_memory("before loading train")
        train_df = load_filtered_data(
            **common_params,
            timestamp_filter=(start_cutoff, train_cutoff),
        )

    log_memory("before loading val")
    val_df = load_filtered_data(
        **common_params,
        timestamp_filter=(train_cutoff, val_cutoff),
    )
    log_memory("loaded everything")

    return train_df, val_df
