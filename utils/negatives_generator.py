import functools
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, ParamSpec, TypeVar

import duckdb
import polars as pl

from utils.global_config import config as global_config
from data.utils import TO_DAY, log_memory, setup_duckdb_connection

SECONDS_IN_DAY = 86_400

P = ParamSpec("P")
R = TypeVar("R")


@dataclass
class NegativeSamplingConfig:
    min_k: int
    max_k: int
    multiplier: float
    top_n: int | None  # None when weighted=True
    seed: int
    num_seeds: int
    window_days: int
    margin_multiplier: int
    from_whole_history: bool = False
    weighted: bool = False


@dataclass
class BucketingParams:
    seeds: list[int]
    target_per_join: int
    num_buckets: int


def accept_dataframe_as_path(param_name: str = "input_path") -> Callable:
    """
    Decorator that allows a function expecting a file path to also accept a pl.DataFrame.

    If a DataFrame is passed, it writes it to a temporary parquet file,
    passes the path to the function, and cleans up afterwards.

    Args:
        param_name: Name of the parameter that accepts the path/DataFrame

    Example:
        @accept_dataframe_as_path("input_path")
        def process_file(input_path: str | pl.DataFrame, output_path: str) -> pl.DataFrame:
            ...
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            import inspect

            sig = inspect.signature(func)
            param_names = list(sig.parameters.keys())

            temp_file = None
            modified_kwargs = dict(kwargs)
            modified_args = args

            if param_name in kwargs:
                value = kwargs[param_name]
                if isinstance(value, pl.DataFrame):
                    temp_file = tempfile.NamedTemporaryFile(
                        suffix=".parquet", delete=False
                    )
                    value.write_parquet(temp_file.name)
                    modified_kwargs[param_name] = temp_file.name
            elif param_name in param_names:
                param_index = param_names.index(param_name)
                if param_index < len(args) and isinstance(
                    args[param_index], pl.DataFrame
                ):
                    temp_file = tempfile.NamedTemporaryFile(
                        suffix=".parquet", delete=False
                    )
                    args[param_index].write_parquet(temp_file.name)
                    modified_args = (
                        *args[:param_index],
                        temp_file.name,
                        *args[param_index + 1 :],
                    )

            try:
                return func(*modified_args, **modified_kwargs)
            finally:
                if temp_file is not None:
                    Path(temp_file.name).unlink(missing_ok=True)

        return wrapper

    return decorator


def _get_max_day_epoch(input_path: str) -> tuple[int, int]:
    """Get maximum timestamp and corresponding day epoch from input data."""
    max_ts = int(
        pl.scan_parquet(input_path).select(pl.col("timestamp").max()).collect().item()
    )
    max_day_epoch_s = (max_ts // SECONDS_IN_DAY) * SECONDS_IN_DAY
    log_memory(f"Max timestamp: {max_ts}, max_day_epoch_s: {max_day_epoch_s}")
    return max_ts, max_day_epoch_s


def _get_timestamp_range(input_path: str) -> tuple[int, int]:
    """Get min and max timestamps from input data."""
    result = (
        pl.scan_parquet(input_path)
        .select(
            pl.col("timestamp").min().alias("min_ts"),
            pl.col("timestamp").max().alias("max_ts"),
        )
        .collect()
    )
    min_ts = int(result["min_ts"][0])
    max_ts = int(result["max_ts"][0])
    log_memory(f"Timestamp range: {min_ts} to {max_ts}")
    return min_ts, max_ts


def _load_base_data(con: duckdb.DuckDBPyConnection, input_path: str) -> int:
    """Load input data and create base table with computed columns."""
    con.execute(
        "CREATE TABLE base_df AS SELECT * FROM read_parquet(?);",
        [input_path],
    )
    base_count = con.execute("SELECT COUNT(*) FROM base_df").fetchone()[0]
    log_memory(f"After loading base_df into DuckDB (rows: {base_count})")

    log_memory("Before creating base table")
    con.execute(
        f"""
        CREATE TABLE base AS
        SELECT 
            *,
            {TO_DAY} as day
        FROM base_df;
    """
    )
    con.execute("DROP TABLE base_df;")
    log_memory("After creating base table")
    return base_count


def _create_item_meta_table(con: duckdb.DuckDBPyConnection) -> None:
    """Create item metadata table with unique item_id entries."""
    log_memory("Before creating item_meta table")
    con.execute(
        """
        CREATE TABLE item_meta AS
        WITH ranked_meta AS (
            SELECT 
                item_id, 
                artist_id, 
                album_id,
                ROW_NUMBER() OVER (PARTITION BY item_id ORDER BY artist_id, album_id) as rn
            FROM base
            WHERE item_id IS NOT NULL
        )
        SELECT item_id, artist_id, album_id
        FROM ranked_meta
        WHERE rn = 1;
    """
    )
    item_meta_count = con.execute("SELECT COUNT(*) FROM item_meta").fetchone()[0]
    log_memory(f"After creating item_meta table (rows: {item_meta_count})")


def _create_user_targets_table(
    con: duckdb.DuckDBPyConnection, config: NegativeSamplingConfig
) -> None:
    """Create user targets table with target_k values."""
    log_memory("Before creating user_targets table")
    con.execute(
        f"""
        CREATE TABLE user_targets AS
        SELECT 
            uid,
            SUM(CASE WHEN event_type = 'like' THEN 1 ELSE 0 END) AS likes_cnt,
            SUM(CASE WHEN event_type = 'dislike' THEN 1 ELSE 0 END) AS dislikes_cnt,
            CAST(
                LEAST(
                    GREATEST(
                        CAST({config.multiplier} AS DOUBLE) * CAST(
                            SUM(CASE WHEN event_type = 'like' THEN 1 ELSE 0 END) -
                            SUM(CASE WHEN event_type = 'dislike' THEN 1 ELSE 0 END)
                        AS DOUBLE),
                        CAST({config.min_k} AS DOUBLE)
                    ),
                    CAST({config.max_k} AS DOUBLE)
                ) AS BIGINT
            ) as target_k
        FROM base
        GROUP BY uid;
    """
    )
    log_memory("After creating user_targets table")

    stats = con.execute(
        """
        SELECT 
            COUNT(*) as num_users,
            SUM(target_k) as total_target_k,
            AVG(target_k) as avg_target_k,
            MIN(target_k) as min_target_k,
            MAX(target_k) as max_target_k,
            SUM(likes_cnt) as total_likes_cnt,
            SUM(dislikes_cnt) as total_dislikes_cnt
        FROM user_targets
    """
    ).fetchone()
    log_memory(
        f"user_targets stats: users={stats[0]}, total_k={stats[1]}, "
        f"avg_k={stats[2]:.2f}, min={stats[3]}, max={stats[4]}, "
        f"total_likes={stats[5]}, total_dislikes={stats[6]}"
    )


def _create_seen_table(con: duckdb.DuckDBPyConnection) -> None:
    """Create table of seen user-item pairs."""
    log_memory("Before creating seen table")
    con.execute(
        """
        CREATE TABLE seen AS
        SELECT DISTINCT uid, item_id
        FROM base
        WHERE uid IS NOT NULL AND item_id IS NOT NULL;
    """
    )
    log_memory("After creating seen table")


def _create_daily_item_counts_table(
    con: duckdb.DuckDBPyConnection, weighted: bool
) -> int:
    """Create daily item counts table.

    When weighted=True, keeps all rows (no aggregation) for weighted sampling.
    When weighted=False, aggregates by day and item_id.

    Both modes include row_id and org_rate for consistent processing downstream.
    """
    log_memory("Before creating daily_item_counts table")

    if weighted:
        # No aggregation - each row is a separate entry for weighted sampling
        # org_rate is 0.0 or 1.0 based on is_organic
        con.execute(
            f"""
            CREATE TABLE daily_item_counts AS
            SELECT 
                {TO_DAY} as day,
                item_id,
                CAST(is_organic AS DOUBLE) as org_rate,
                (random() * 100000)::INTEGER as row_id
            FROM base
            WHERE item_id IS NOT NULL;
        """
        )
    else:
        con.execute(
            f"""
            CREATE TABLE daily_item_counts AS
            SELECT 
                {TO_DAY} as day,
                item_id,
                COUNT(*) as daily_count,
                SUM(CAST(is_organic AS INTEGER)) as organic_count,
                (random() * 100000)::INTEGER as row_id
            FROM base
            WHERE item_id IS NOT NULL
            GROUP BY 1, 2;
        """
        )

    daily_counts_rows = con.execute(
        "SELECT COUNT(*) FROM daily_item_counts"
    ).fetchone()[0]
    log_memory(
        f"After creating daily_item_counts table (rows: {daily_counts_rows}, weighted: {weighted})"
    )
    return daily_counts_rows


def _create_valid_days_table(
    con: duckdb.DuckDBPyConnection, window_days: int, max_day_epoch_s: int
) -> int:
    """Create table of valid days for popularity calculation."""
    log_memory("Before creating valid_days table")
    con.execute(
        f"""
        CREATE TABLE valid_days AS
        SELECT unnest(generate_series(
            (SELECT min(day) + INTERVAL {window_days} DAY FROM daily_item_counts),
            to_timestamp({max_day_epoch_s}),
            INTERVAL 1 DAY
        )) as day;
    """
    )
    valid_days_count = con.execute("SELECT COUNT(*) FROM valid_days").fetchone()[0]
    log_memory(f"After creating valid_days table (rows: {valid_days_count})")
    return valid_days_count


def _create_daily_top_table(
    con: duckdb.DuckDBPyConnection,
    window_days: int,
    top_n: int | None,
    weighted: bool,
) -> int:
    """Create daily top items table with windowed popularity.

    When weighted=True, keeps all rows without aggregation or top_n filtering.
    When weighted=False, aggregates and filters to top_n items per day.

    Both modes output: day, item_id, org_rate, row_id
    """
    log_memory("Before creating daily_top table")

    if weighted:
        # Weighted mode: no aggregation, no top_n, keep all rows with day mapping
        con.execute(
            f"""
            CREATE TABLE daily_top AS
            SELECT 
                v.day as day,
                d.item_id,
                d.org_rate,
                d.row_id
            FROM valid_days v
            JOIN daily_item_counts d
              ON d.day >= v.day - INTERVAL {window_days} DAY 
             AND d.day < v.day;
        """
        )
    else:
        con.execute(
            f"""
            CREATE TABLE daily_top AS
            WITH windowed_counts AS (
                SELECT 
                    v.day as target_day,
                    d.item_id,
                    SUM(d.daily_count) as total_count,
                    SUM(d.organic_count) as total_organic,
                    SUM(d.daily_count) as total_events,
                    (random() * 100000)::INTEGER as row_id
                FROM valid_days v
                JOIN daily_item_counts d
                  ON d.day >= v.day - INTERVAL {window_days} DAY 
                 AND d.day < v.day
                GROUP BY v.day, d.item_id
            ),
            ranked AS (
                SELECT 
                    target_day as day,
                    item_id,
                    CAST(total_organic AS DOUBLE) / NULLIF(total_events, 0) as org_rate,
                    total_count,
                    row_id,
                    row_number() OVER (
                        PARTITION BY target_day 
                        ORDER BY total_count DESC, item_id ASC
                    ) as rn
                FROM windowed_counts
            )
            SELECT day, item_id, org_rate, row_id
            FROM ranked 
            WHERE rn <= {top_n};
        """
        )

    daily_top_count = con.execute("SELECT COUNT(*) FROM daily_top").fetchone()[0]
    log_memory(
        f"After creating daily_top table (rows: {daily_top_count}, weighted: {weighted})"
    )

    con.execute("DROP TABLE daily_item_counts;")
    con.execute("DROP TABLE valid_days;")
    log_memory("After dropping intermediate daily tables")

    return daily_top_count


def _create_global_items_table(
    con: duckdb.DuckDBPyConnection, top_n: int | None, weighted: bool
) -> int:
    """Create table of globally popular items across all history.

    When weighted=True, keeps all rows without aggregation or top_n filtering.
    When weighted=False, aggregates and filters to top_n items.

    Both modes output: item_id, org_rate, row_id
    Note: day is NOT included here for non-weighted mode; it will be generated
    per-row at sampling time to avoid data leakage.
    """
    log_memory("Before creating global_items table")

    if weighted:
        # Weighted mode: no aggregation, keep all rows
        # org_rate is 0.0 or 1.0 based on is_organic
        con.execute(
            """
            CREATE TABLE global_items AS
            SELECT 
                item_id,
                CAST(is_organic AS DOUBLE) as org_rate,
                day,
                (random() * 100000)::INTEGER as row_id
            FROM base
            WHERE item_id IS NOT NULL;
        """
        )
    else:
        con.execute(
            f"""
            CREATE TABLE global_items AS
            WITH item_stats AS (
                SELECT 
                    item_id,
                    COUNT(*) as total_count,
                    SUM(CAST(is_organic AS INTEGER)) as organic_count
                FROM base
                WHERE item_id IS NOT NULL
                GROUP BY item_id
            ),
            ranked AS (
                SELECT 
                    item_id,
                    total_count,
                    CAST(organic_count AS DOUBLE) / NULLIF(total_count, 0) as org_rate,
                    (random() * 100000)::INTEGER as row_id,
                    row_number() OVER (ORDER BY total_count DESC, item_id ASC) as rn
                FROM item_stats
            )
            SELECT item_id, org_rate, row_id
            FROM ranked 
            WHERE rn <= {top_n};
        """
        )

    global_count = con.execute("SELECT COUNT(*) FROM global_items").fetchone()[0]
    log_memory(
        f"After creating global_items table (rows: {global_count}, weighted: {weighted})"
    )
    return global_count


def _compute_bucketing_params(
    con: duckdb.DuckDBPyConnection,
    item_pool_count: int,
    config: NegativeSamplingConfig,
) -> BucketingParams:
    """Compute bucketing parameters for the sampling strategy."""
    user_count = con.execute("SELECT COUNT(*) FROM user_targets").fetchone()[0]
    log_memory(f"item_pool rows: {item_pool_count}, users: {user_count}")

    seeds = [config.seed + i for i in range(config.num_seeds)]
    target_per_join = max(1, (config.margin_multiplier * config.max_k) // len(seeds))
    num_buckets = int(max(1, item_pool_count // target_per_join))
    log_memory(
        f"num_buckets: {num_buckets}, target_per_join: {target_per_join}, "
        f"num_seeds: {config.num_seeds}"
    )

    return BucketingParams(
        seeds=seeds,
        target_per_join=target_per_join,
        num_buckets=num_buckets,
    )


def _create_user_buckets_table(
    con: duckdb.DuckDBPyConnection, params: BucketingParams
) -> None:
    """Create user buckets table for hash-based sampling."""
    log_memory("Before creating user_buckets table")
    seed_cases_user = " ".join(
        [
            f"WHEN {i} THEN CAST(hash(CAST(uid AS VARCHAR) || '_{s}') AS UBIGINT) % {params.num_buckets}"
            for i, s in enumerate(params.seeds)
        ]
    )
    con.execute(
        f"""
        CREATE TABLE user_buckets AS
        SELECT 
            u.uid,
            u.target_k,
            s.seed_idx,
            CASE s.seed_idx {seed_cases_user} END as bucket
        FROM user_targets u
        CROSS JOIN (SELECT unnest(generate_series(0, {len(params.seeds) - 1})) as seed_idx) s;
    """
    )
    log_memory("After creating user_buckets table")


def _create_item_buckets_table(
    con: duckdb.DuckDBPyConnection,
    params: BucketingParams,
    from_whole_history: bool,
    weighted: bool,
) -> None:
    """Create item buckets table for hash-based sampling.

    For windowed mode and weighted whole history: hash includes item_id, day, row_id
    For non-weighted whole history: hash includes item_id, row_id (no day)
    """
    log_memory("Before creating item_buckets table")

    if from_whole_history and not weighted:
        # Global non-weighted: no day column, hash on item_id + row_id
        seed_cases_item = " ".join(
            [
                f"WHEN {i} THEN CAST(hash(CAST(item_id AS VARCHAR) || '_' || "
                f"CAST(row_id AS VARCHAR) || '_{s}') AS UBIGINT) % {params.num_buckets}"
                for i, s in enumerate(params.seeds)
            ]
        )
        con.execute(
            f"""
            CREATE TABLE item_buckets AS
            SELECT 
                g.item_id,
                g.org_rate,
                g.row_id,
                s.seed_idx,
                CASE s.seed_idx {seed_cases_item} END as bucket
            FROM global_items g
            CROSS JOIN (SELECT unnest(generate_series(0, {len(params.seeds) - 1})) as seed_idx) s;
        """
        )
    else:
        # Windowed mode or weighted whole history: has day column
        seed_cases_item = " ".join(
            [
                f"WHEN {i} THEN CAST(hash(CAST(item_id AS VARCHAR) || '_' || "
                f"CAST(day AS VARCHAR) || '_' || CAST(row_id AS VARCHAR) || '_{s}') AS UBIGINT) % {params.num_buckets}"
                for i, s in enumerate(params.seeds)
            ]
        )
        source_table = "global_items" if from_whole_history else "daily_top"
        con.execute(
            f"""
            CREATE TABLE item_buckets AS
            SELECT 
                t.item_id,
                t.day,
                t.org_rate,
                t.row_id,
                s.seed_idx,
                CASE s.seed_idx {seed_cases_item} END as bucket
            FROM {source_table} t
            CROSS JOIN (SELECT unnest(generate_series(0, {len(params.seeds) - 1})) as seed_idx) s;
        """
        )
    log_memory("After creating item_buckets table")


def _create_bucket_indexes(con: duckdb.DuckDBPyConnection) -> None:
    """Create indexes for efficient bucket joins."""
    log_memory("Before creating indexes")
    con.execute("CREATE INDEX idx_user_buckets ON user_buckets(seed_idx, bucket);")
    con.execute("CREATE INDEX idx_item_buckets ON item_buckets(seed_idx, bucket);")
    con.execute("CREATE INDEX idx_seen ON seen(uid, item_id);")
    log_memory("After creating indexes")


def _create_unfiltered_candidates_table(
    con: duckdb.DuckDBPyConnection,
    from_whole_history: bool,
    weighted: bool,
) -> int:
    """Create unfiltered candidates by joining buckets and filtering seen items.

    For non-weighted whole history: no day column (will be generated at sampling)
    For all other modes: includes day column
    """
    log_memory("Before creating unfiltered_candidates table")

    if from_whole_history and not weighted:
        # Global non-weighted: no day column
        con.execute(
            """
            CREATE TABLE unfiltered_candidates AS
            SELECT u.uid, i.item_id, i.org_rate, i.row_id, u.target_k
            FROM user_buckets u
            JOIN item_buckets i 
              ON u.seed_idx = i.seed_idx 
             AND u.bucket = i.bucket
            WHERE NOT EXISTS (
                SELECT 1 FROM seen s 
                WHERE s.uid = u.uid AND s.item_id = i.item_id
            );
        """
        )
    else:
        # Windowed mode or weighted whole history: has day column
        con.execute(
            """
            CREATE TABLE unfiltered_candidates AS
            SELECT u.uid, i.item_id, i.day, i.org_rate, i.row_id, u.target_k
            FROM user_buckets u
            JOIN item_buckets i 
              ON u.seed_idx = i.seed_idx 
             AND u.bucket = i.bucket
            WHERE NOT EXISTS (
                SELECT 1 FROM seen s 
                WHERE s.uid = u.uid AND s.item_id = i.item_id
            );
        """
        )

    candidate_count = con.execute(
        "SELECT COUNT(*) FROM unfiltered_candidates"
    ).fetchone()[0]
    log_memory(f"After creating unfiltered_candidates table (rows: {candidate_count})")

    con.execute("DROP TABLE user_buckets;")
    con.execute("DROP TABLE item_buckets;")
    con.execute("DROP TABLE seen;")
    log_memory("After dropping intermediate tables")

    return candidate_count


def _create_sampled_table(
    con: duckdb.DuckDBPyConnection,
    seed: int,
    from_whole_history: bool,
    weighted: bool,
    min_ts: int | None = None,
    max_ts: int | None = None,
) -> int:
    """Sample candidates per user up to target_k.

    Uses row_id in hash for unique ordering in all modes.
    For non-weighted whole history mode, generates random day per row.

    Candidates are deduplicated to one row per (uid, item_id) *before* the
    top-target_k cut: the seed cross-join emits the same candidate once per
    seed (and a windowed candidate can recur on several days), so cutting
    first would spend a user's quota on repeats of one item.

    Output always has: uid, item_id, day, org_rate
    """
    log_memory("Before creating sampled table")
    hash_seed_444 = seed + 444
    hash_seed_888 = seed + 888

    if from_whole_history and not weighted:
        # Global non-weighted: generate day per row from timestamp range
        min_day_epoch = (min_ts // SECONDS_IN_DAY) * SECONDS_IN_DAY
        max_day_epoch = (max_ts // SECONDS_IN_DAY) * SECONDS_IN_DAY
        num_days = (max_day_epoch - min_day_epoch) // SECONDS_IN_DAY + 1

        con.execute(
            f"""
            CREATE TABLE sampled AS
            WITH deduplicated AS (
                SELECT uid, item_id, org_rate, target_k, row_id
                FROM (
                    SELECT
                        uid, item_id, org_rate, target_k, row_id,
                        row_number() OVER (
                            PARTITION BY uid, item_id ORDER BY row_id
                        ) as dedup_rn
                    FROM unfiltered_candidates
                )
                WHERE dedup_rn = 1
            ),
            ranked AS (
                SELECT
                    uid,
                    item_id,
                    org_rate,
                    target_k,
                    row_id,
                    row_number() OVER (
                        PARTITION BY uid
                        ORDER BY hash(
                            CAST(uid AS VARCHAR) || '_' ||
                            CAST(item_id AS VARCHAR) || '_' ||
                            CAST(row_id AS VARCHAR) || '_{hash_seed_444}'
                        )
                    ) as rn
                FROM deduplicated
            )
            SELECT
                uid,
                item_id,
                org_rate,
                to_timestamp(
                    {min_day_epoch} + (
                        CAST(hash(
                            CAST(uid AS VARCHAR) || '_' || 
                            CAST(item_id AS VARCHAR) || '_' ||
                            CAST(row_id AS VARCHAR) || '_{hash_seed_888}'
                        ) AS UBIGINT) % {num_days}
                    ) * {SECONDS_IN_DAY}
                ) as day
            FROM ranked
            WHERE rn <= target_k;
        """
        )
    else:
        # Windowed mode or weighted whole history: has day column
        con.execute(
            f"""
            CREATE TABLE sampled AS
            WITH deduplicated AS (
                SELECT uid, item_id, day, org_rate, target_k, row_id
                FROM (
                    SELECT
                        uid, item_id, day, org_rate, target_k, row_id,
                        -- row_id breaks ties on day: without it the surviving
                        -- duplicate, and with it the org_rate the row carries,
                        -- depends on the order duckdb happened to scan in.
                        row_number() OVER (
                            PARTITION BY uid, item_id ORDER BY day, row_id
                        ) as dedup_rn
                    FROM unfiltered_candidates
                )
                WHERE dedup_rn = 1
            ),
            ranked AS (
                SELECT
                    uid,
                    item_id,
                    day,
                    org_rate,
                    target_k,
                    row_number() OVER (
                        PARTITION BY uid
                        ORDER BY hash(
                            CAST(uid AS VARCHAR) || '_' ||
                            CAST(item_id AS VARCHAR) || '_' ||
                            CAST(day AS VARCHAR) || '_' ||
                            CAST(row_id AS VARCHAR) || '_{hash_seed_444}'
                        )
                    ) as rn
                FROM deduplicated
            )
            SELECT uid, item_id, day, org_rate
            FROM ranked
            WHERE rn <= target_k;
        """
        )

    sampled_count = con.execute("SELECT COUNT(*) FROM sampled").fetchone()[0]
    log_memory(f"After creating sampled table (rows: {sampled_count})")

    con.execute("DROP TABLE unfiltered_candidates;")
    log_memory("After dropping unfiltered_candidates")

    return sampled_count


def _create_negatives_table(
    con: duckdb.DuckDBPyConnection,
    seed: int,
) -> int:
    """Create negatives table with all computed columns.

    All modes now have day column, so we use unified logic:
    - timestamp = day + random offset within day
    - is_organic = probabilistic based on org_rate
    """
    log_memory("Before creating negatives table")
    hash_seed_555 = seed + 555
    hash_seed_777 = seed + 777

    con.execute(
        f"""
        CREATE TABLE negatives AS
        SELECT 
            s.uid,
            s.item_id,
            m.artist_id,
            m.album_id,
            CAST(
                epoch(s.day) + 
                CAST(hash(
                    CAST(s.uid AS VARCHAR) || '_' || 
                    CAST(s.item_id AS VARCHAR) || '_{hash_seed_555}'
                ) AS UBIGINT) % {SECONDS_IN_DAY}
            AS BIGINT) as timestamp,
            (
                CAST(hash(
                    CAST(s.uid AS VARCHAR) || '_' || 
                    CAST(s.item_id AS VARCHAR) || '_' || 
                    CAST(s.day AS VARCHAR) || '_{hash_seed_777}'
                ) AS UBIGINT) / 18446744073709551615.0
            ) < COALESCE(s.org_rate, 0.0) as is_organic,
            'random_negative' as event_type
        FROM sampled s
        LEFT JOIN item_meta m ON s.item_id = m.item_id;
    """
    )

    neg_count = con.execute("SELECT COUNT(*) FROM negatives").fetchone()[0]
    log_memory(f"After creating negatives table (rows: {neg_count})")

    con.execute("DROP TABLE sampled;")
    con.execute("DROP TABLE item_meta;")
    log_memory("After dropping sampled and item_meta")

    return neg_count


def _get_output_columns() -> list[str]:
    """Return the ordered list of output columns."""
    return [
        "uid",
        "item_id",
        "artist_id",
        "album_id",
        "timestamp",
        "is_organic",
        "event_type",
    ]


def _create_final_output_table(
    con: duckdb.DuckDBPyConnection,
) -> tuple[int, int, int]:
    """Create final output table combining base data and negatives."""
    output_columns = _get_output_columns()
    output_columns_sql = ", ".join(output_columns)

    log_memory("Before creating base_output table")
    con.execute(
        f"""
        CREATE TABLE base_output AS
        SELECT {output_columns_sql}
        FROM base;
    """
    )
    base_count = con.execute("SELECT COUNT(*) FROM base_output").fetchone()[0]
    log_memory(f"After creating base_output table (rows: {base_count})")

    con.execute("DROP TABLE base;")
    log_memory("After dropping base")

    neg_count = con.execute("SELECT COUNT(*) FROM negatives").fetchone()[0]

    log_memory("Before creating final_output table")
    con.execute(
        f"""
        CREATE TABLE final_output AS
        SELECT {output_columns_sql} FROM base_output
        UNION ALL
        SELECT {output_columns_sql} FROM negatives;
    """
    )
    final_count = con.execute("SELECT COUNT(*) FROM final_output").fetchone()[0]
    log_memory(f"After creating final_output table (rows: {final_count})")

    con.execute("DROP TABLE base_output;")
    con.execute("DROP TABLE negatives;")
    log_memory("After dropping base_output and negatives")

    return base_count, neg_count, final_count


def _export_and_cleanup(
    con: duckdb.DuckDBPyConnection,
    output_path: str,
    base_count: int,
    neg_count: int,
    final_count: int,
) -> None:
    """Export final output to parquet and cleanup."""
    log_memory("Before exporting to parquet")
    con.execute(
        "COPY final_output TO ? (FORMAT PARQUET, COMPRESSION 'zstd');",
        [output_path],
    )
    log_memory("After exporting to parquet")

    log_memory(
        f"Final counts - base: {base_count}, negatives: {neg_count}, total: {final_count}"
    )
    if base_count > 0:
        log_memory(f"Negatives ratio: {neg_count / base_count:.4f}")

    con.execute("DROP TABLE final_output;")
    con.close()
    log_memory("After closing DuckDB connection")


def _resolve_output_path(
    output_path: str | None, return_dataframe: bool
) -> tuple[str, Any]:
    """Resolve the output path, creating a temp file if needed."""
    temp_output_file = None
    if output_path is None:
        if return_dataframe:
            temp_output_file = tempfile.NamedTemporaryFile(
                suffix=".parquet", delete=False
            )
            output_path = temp_output_file.name
            temp_output_file.close()
        else:
            raise ValueError("output_path must be provided when return_dataframe=False")
    return output_path, temp_output_file


def _load_result_dataframe(output_path: str, temp_output_file: Any) -> pl.DataFrame:
    """Load result DataFrame and cleanup temp file if needed."""
    log_memory("Loading result DataFrame")
    result = pl.read_parquet(output_path)
    log_memory(f"Loaded result DataFrame (rows: {result.height})")

    if temp_output_file is not None:
        try:
            os.unlink(output_path)
        except OSError:
            pass

    return result


def _drop_item_pool_table(
    con: duckdb.DuckDBPyConnection, from_whole_history: bool
) -> None:
    """Drop the appropriate item pool table based on mode."""
    if from_whole_history:
        con.execute("DROP TABLE IF EXISTS global_items;")
    else:
        con.execute("DROP TABLE IF EXISTS daily_top;")
    con.execute("DROP TABLE user_targets;")
    log_memory("After dropping item pool and user_targets tables")


def _process_with_duckdb(
    input_path: str,
    output_path: str,
    config: NegativeSamplingConfig,
    max_day_epoch_s: int | None,
    min_ts: int | None,
    max_ts: int | None,
) -> None:
    """Main DuckDB processing pipeline for both windowed and whole history modes."""
    tmp_dir = global_config.tmp_path
    tmp_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=str(tmp_dir)) as tmpdir:
        con = setup_duckdb_connection(tmpdir, "processing.duckdb")

        try:
            # Load and prepare base data
            _load_base_data(con, input_path)
            _create_item_meta_table(con)
            _create_user_targets_table(con, config)
            _create_seen_table(con)

            # Create item pool based on mode
            if config.from_whole_history:
                item_pool_count = _create_global_items_table(
                    con, config.top_n, config.weighted
                )
            else:
                _create_daily_item_counts_table(con, config.weighted)
                _create_valid_days_table(con, config.window_days, max_day_epoch_s)
                item_pool_count = _create_daily_top_table(
                    con, config.window_days, config.top_n, config.weighted
                )

            # Compute bucketing and create buckets
            bucketing_params = _compute_bucketing_params(con, item_pool_count, config)
            _create_user_buckets_table(con, bucketing_params)
            _create_item_buckets_table(
                con,
                bucketing_params,
                config.from_whole_history,
                config.weighted,
            )

            _drop_item_pool_table(con, config.from_whole_history)

            # Create indexes and candidates
            _create_bucket_indexes(con)
            _create_unfiltered_candidates_table(
                con, config.from_whole_history, config.weighted
            )

            # Sample and create negatives
            _create_sampled_table(
                con,
                config.seed,
                config.from_whole_history,
                config.weighted,
                min_ts,
                max_ts,
            )
            _create_negatives_table(con, config.seed)

            # Create final output and export
            base_count, neg_count, final_count = _create_final_output_table(con)
            _export_and_cleanup(con, output_path, base_count, neg_count, final_count)

        except Exception:
            con.close()
            raise


@accept_dataframe_as_path("input_path")
def add_popular_random_negatives(
    input_path: str | pl.DataFrame,
    output_path: str | None = None,
    *,
    min_k: int,
    max_k: int,
    multiplier: float,
    top_n: int | None = None,
    seed: int,
    num_seeds: int = 10,
    window_days: int = 7,
    margin_multiplier: int = 10,
    from_whole_history: bool = False,
    weighted: bool = False,
    return_dataframe: bool = True,
) -> pl.DataFrame | None:
    """
    Process negatives and save directly to parquet without materializing in Python.

    This function generates random negative samples for users by selecting from
    popular items they haven't interacted with. It supports multiple modes:

    1. Windowed mode (default): Samples from items popular within a rolling window
       before each day. This gives time-aware negatives.

    2. Whole history mode: Samples from globally popular items across all history.
       This produces more negatives and is better for datasets with many events.

    3. Weighted mode: Instead of selecting from top_n items, samples from all items
       with probability proportional to their occurrence count. Popular items are
       more likely to be selected as negatives. Can be combined with either windowed
       or whole history mode.

    Args:
        input_path: Path to input parquet file or DataFrame
        output_path: Path to output parquet file. If None and return_dataframe=True,
                     uses a temporary file.
        min_k: Minimum number of negatives per user
        max_k: Maximum number of negatives per user
        multiplier: Multiplier for target_k calculation based on user's net likes
        top_n: Number of top popular items to sample from. Required when weighted=False,
               ignored when weighted=True.
        seed: Random seed for reproducibility
        num_seeds: Number of seeds for bucketing strategy (default: 10)
        window_days: Window size in days for popularity calculation in windowed mode
                     (default: 7, ignored when from_whole_history=True)
        margin_multiplier: Margin multiplier for bucketing to ensure enough candidates
                           (default: 10)
        from_whole_history: If True, uses global popularity instead of windowed.
                            Better for datasets with many events. (default: False)
        weighted: If True, samples items with probability proportional to their
                  occurrence count (popularity). top_n is ignored. (default: False)
        return_dataframe: If True, returns DataFrame. If False, returns None.

    Returns:
        DataFrame if return_dataframe=True, otherwise None

    Example:
        # Windowed mode (time-aware negatives from top items)
        result = add_popular_random_negatives(
            "events.parquet",
            min_k=5, max_k=100, multiplier=2.0,
            top_n=10000, seed=42
        )

        # Whole history mode (more negatives for large datasets)
        result = add_popular_random_negatives(
            "events.parquet",
            min_k=5, max_k=100, multiplier=2.0,
            top_n=10000, seed=42,
            from_whole_history=True
        )

        # Weighted mode (popularity-proportional sampling)
        result = add_popular_random_negatives(
            "events.parquet",
            min_k=5, max_k=100, multiplier=2.0,
            seed=42,
            weighted=True  # top_n not needed
        )

        # Weighted + whole history mode
        result = add_popular_random_negatives(
            "events.parquet",
            min_k=5, max_k=100, multiplier=2.0,
            seed=42,
            from_whole_history=True,
            weighted=True
        )
    """
    log_memory("Start add_popular_random_negatives")

    # Validate arguments
    if not weighted and top_n is None:
        raise ValueError("top_n is required when weighted=False")

    config = NegativeSamplingConfig(
        min_k=min_k,
        max_k=max_k,
        multiplier=multiplier,
        top_n=top_n,
        seed=seed,
        num_seeds=num_seeds,
        window_days=window_days,
        margin_multiplier=margin_multiplier,
        from_whole_history=from_whole_history,
        weighted=weighted,
    )

    output_path, temp_output_file = _resolve_output_path(output_path, return_dataframe)

    # Get timestamp information based on mode
    min_ts, max_ts, max_day_epoch_s = None, None, None

    if from_whole_history and not weighted:
        # Need full timestamp range for random day generation
        min_ts, max_ts = _get_timestamp_range(input_path)
    elif not from_whole_history:
        # Need max_day_epoch_s for windowed mode
        _, max_day_epoch_s = _get_max_day_epoch(input_path)

    log_memory(f"Mode: from_whole_history={from_whole_history}, weighted={weighted}")

    _process_with_duckdb(
        input_path, output_path, config, max_day_epoch_s, min_ts, max_ts
    )

    log_memory("End add_popular_random_negatives")

    if return_dataframe:
        return _load_result_dataframe(output_path, temp_output_file)

    return None
