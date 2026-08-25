import tempfile
import os
import polars as pl
import duckdb
import psutil

SECONDS_IN_DAY = 86_400


def log_memory(msg: str):
    process = psutil.Process(os.getpid())
    mem_gb = process.memory_info().rss / 1024**3
    print(f"[MEM {mem_gb:.2f} GB] {msg}", flush=True)


def add_popular_random_negatives(
    df: pl.DataFrame | str,
    min_k: int,
    max_k: int,
    multiplier: float,
    top_n: int,
    seed: int,
    num_seeds: int,
    window_days: int,
    margin_multiplier: int,
) -> pl.DataFrame:

    log_memory("add_popular_random_negatives: start")

    # Handle path input
    if isinstance(df, str):
        input_path = df
        # Need to get max_ts from file
        temp_df = pl.scan_parquet(df).select("timestamp").collect()
        max_ts = int(temp_df.get_column("timestamp").max())
        del temp_df
    else:
        input_path = None
        max_ts = int(df.get_column("timestamp").max())

    max_day_epoch_s = (max_ts // SECONDS_IN_DAY) * SECONDS_IN_DAY

    log_memory("add_popular_random_negatives: calculated max_ts")

    # Create duckdb instance explicitly to offload all heavy components
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "processing.duckdb")
        con = duckdb.connect(database=db_path)
        con.execute("SET enable_progress_bar=false;")
        con.execute("SET memory_limit = '20GB';")
        con.execute("SET preserve_insertion_order = false;")

        log_memory("add_popular_random_negatives: duckdb connected")

        # Load data into DuckDB
        if input_path is not None:
            con.execute(
                f"CREATE TABLE base_df AS SELECT * FROM read_parquet('{input_path}')"
            )
        else:
            con.register("base_df_temp", df)
            con.execute("CREATE TABLE base_df AS SELECT * FROM base_df_temp")
            con.unregister("base_df_temp")
            del df

        log_memory("add_popular_random_negatives: data loaded into duckdb")

        # Create base events table with day truncation
        con.execute(
            """
            CREATE TABLE base_events AS
            SELECT 
                item_id, 
                date_trunc('day', to_timestamp(CAST(timestamp AS BIGINT))) as day,
                is_organic
            FROM base_df
            WHERE item_id IS NOT NULL;
        """
        )

        log_memory("add_popular_random_negatives: base_events created")

        # Create item_meta table
        con.execute(
            """
            CREATE TABLE item_meta AS
            SELECT DISTINCT item_id, artist_id, album_id
            FROM base_df
            WHERE item_id IS NOT NULL;
        """
        )

        log_memory("add_popular_random_negatives: item_meta created")

        # Create user_targets table
        con.execute(
            f"""
            CREATE TABLE user_targets AS
            SELECT 
                uid,
                CAST(
                    LEAST(
                        {max_k},
                        GREATEST(
                            {min_k},
                            CAST({multiplier} * (
                                SUM(CASE WHEN event_type = 'like' THEN 1 ELSE 0 END) -
                                SUM(CASE WHEN event_type = 'dislike' THEN 1 ELSE 0 END)
                            ) AS BIGINT)
                        )
                    ) AS BIGINT
                ) as target_k
            FROM base_df
            GROUP BY uid;
        """
        )

        log_memory("add_popular_random_negatives: user_targets created")

        # Create users table
        con.execute(
            """
            CREATE TABLE users AS
            SELECT DISTINCT uid
            FROM base_df;
        """
        )

        log_memory("add_popular_random_negatives: users created")

        # Create seen table
        con.execute(
            """
            CREATE TABLE seen AS
            SELECT DISTINCT uid, item_id
            FROM base_df
            WHERE item_id IS NOT NULL;
        """
        )

        log_memory("add_popular_random_negatives: seen created")

        # Create daily_top table
        con.execute(
            f"""
            CREATE TABLE daily_top AS 
            WITH valid_days AS (
                SELECT unnest(generate_series(
                    (SELECT min(day) FROM base_events),
                    to_timestamp({max_day_epoch_s}),
                    INTERVAL 1 DAY
                )) as day
            ),
            windowed AS (
                SELECT 
                    d.day,
                    b.item_id,
                    b.is_organic
                FROM valid_days d
                JOIN base_events b 
                  ON b.day >= d.day - INTERVAL {window_days} DAY 
                 AND b.day < d.day
            ),
            aggregated AS (
                SELECT 
                    day,
                    item_id,
                    COUNT(*) as count,
                    AVG(CAST(is_organic AS DOUBLE)) as org_rate
                FROM windowed
                GROUP BY day, item_id
            ),
            ranked AS (
                SELECT 
                    day,
                    item_id,
                    org_rate,
                    count,
                    row_number() OVER (PARTITION BY day ORDER BY count DESC, item_id ASC) as rn
                FROM aggregated
            )
            SELECT day, item_id, org_rate
            FROM ranked 
            WHERE rn <= {top_n};
        """
        )

        log_memory("add_popular_random_negatives: daily_top created")

        # Get counts for bucket calculation - ensure integer types
        total_candidate_rows = int(
            con.execute("SELECT COUNT(*) FROM daily_top").fetchone()[0]
        )
        target_per_join = max(1, (margin_multiplier * max_k) // num_seeds)
        num_buckets = int(max(1, total_candidate_rows // target_per_join))

        log_memory(f"add_popular_random_negatives: num_buckets={num_buckets}")

        # Create bucketed user tables for each seed
        seeds = [seed + i for i in range(num_seeds)]

        # Build UNION ALL for user buckets
        # Use XOR (^) instead of addition to avoid overflow
        user_bucket_unions = []
        for i, s in enumerate(seeds):
            user_bucket_unions.append(
                f"""
                SELECT 
                    uid,
                    target_k,
                    (hash(uid) ^ CAST({s} AS UBIGINT)) % CAST({num_buckets} AS UBIGINT) as bucket,
                    {i} as seed_idx
                FROM users u
                JOIN user_targets ut USING (uid)
                """
            )

        con.execute(
            f"""
            CREATE TABLE u_all AS
            {" UNION ALL ".join(user_bucket_unions)};
        """
        )

        log_memory("add_popular_random_negatives: u_all created")

        # Build UNION ALL for item buckets
        # Use XOR (^) instead of addition to avoid overflow
        item_bucket_unions = []
        for i, s in enumerate(seeds):
            item_bucket_unions.append(
                f"""
                SELECT 
                    item_id,
                    day,
                    org_rate,
                    (hash(item_id) ^ hash(CAST(epoch(day) AS BIGINT)) ^ CAST({s} AS UBIGINT)) % CAST({num_buckets} AS UBIGINT) as bucket,
                    {i} as seed_idx
                FROM daily_top
                """
            )

        con.execute(
            f"""
            CREATE TABLE i_all AS
            {" UNION ALL ".join(item_bucket_unions)};
        """
        )

        log_memory("add_popular_random_negatives: i_all created")

        # Create unfiltered candidates with anti-join
        con.execute(
            """
            CREATE TABLE unfiltered_candidates AS
            WITH joined AS (
                SELECT u.uid, u.target_k, i.item_id, i.day, i.org_rate
                FROM u_all u
                JOIN i_all i 
                  ON u.seed_idx = i.seed_idx 
                 AND u.bucket = i.bucket
            )
            SELECT j.uid, j.item_id, j.day, j.org_rate, j.target_k
            FROM joined j
            LEFT JOIN seen s ON j.uid = s.uid AND j.item_id = s.item_id
            WHERE s.uid IS NULL;
        """
        )

        log_memory(
            "add_popular_random_negatives: unfiltered_candidates created"
        )

        # Drop intermediate tables to free memory
        con.execute("DROP TABLE u_all;")
        con.execute("DROP TABLE i_all;")
        con.execute("DROP TABLE base_events;")

        log_memory("add_popular_random_negatives: dropped intermediate tables")

        # Sample candidates using ranking
        # Use XOR (^) instead of addition to avoid overflow
        con.execute(
            f"""
            CREATE TABLE sampled AS
            WITH ranked AS (
                SELECT 
                    uid,
                    item_id,
                    day,
                    org_rate,
                    target_k,
                    ROW_NUMBER() OVER (
                        PARTITION BY uid 
                        ORDER BY hash(uid) ^ hash(item_id) ^ hash(CAST(epoch(day) AS BIGINT)) ^ CAST({seed + 444} AS UBIGINT)
                    ) as rn
                FROM unfiltered_candidates
            )
            SELECT DISTINCT uid, item_id, day, org_rate
            FROM ranked
            WHERE rn <= target_k;
        """
        )

        log_memory("add_popular_random_negatives: sampled created")

        # Drop unfiltered_candidates
        con.execute("DROP TABLE unfiltered_candidates;")

        log_memory(
            "add_popular_random_negatives: dropped unfiltered_candidates"
        )

        # Create negatives with timestamp and is_organic
        # Use XOR (^) instead of addition to avoid overflow
        con.execute(
            f"""
            CREATE TABLE negatives AS
            SELECT 
                s.uid,
                s.item_id,
                im.artist_id,
                im.album_id,
                CAST(
                    epoch(s.day) + 
                    CAST((hash(s.uid) ^ hash(s.item_id) ^ CAST({seed + 555} AS UBIGINT)) % CAST({SECONDS_IN_DAY} AS UBIGINT) AS BIGINT)
                AS BIGINT) as timestamp,
                CAST(
                    (CAST(hash(s.uid) ^ hash(s.item_id) ^ hash(CAST(epoch(s.day) AS BIGINT)) ^ CAST({seed + 777} AS UBIGINT) AS DOUBLE) / 18446744073709551616.0)
                    < COALESCE(s.org_rate, 0.0)
                AS BOOLEAN) as is_organic,
                'random_negative' as event_type
            FROM sampled s
            LEFT JOIN item_meta im ON s.item_id = im.item_id;
        """
        )

        log_memory("add_popular_random_negatives: negatives created")

        # Drop sampled and item_meta
        con.execute("DROP TABLE sampled;")
        con.execute("DROP TABLE item_meta;")
        con.execute("DROP TABLE seen;")
        con.execute("DROP TABLE users;")
        con.execute("DROP TABLE user_targets;")
        con.execute("DROP TABLE daily_top;")

        log_memory("add_popular_random_negatives: dropped more tables")

        # Combine base_df with negatives using UNION ALL BY NAME
        con.execute(
            """
            CREATE TABLE result AS
            SELECT * FROM base_df
            UNION ALL BY NAME
            SELECT * FROM negatives;
        """
        )

        log_memory("add_popular_random_negatives: result created")

        # Drop base_df and negatives
        con.execute("DROP TABLE base_df;")
        con.execute("DROP TABLE negatives;")

        log_memory(
            "add_popular_random_negatives: dropped base_df and negatives"
        )

        # Collect final result
        result = con.execute("SELECT * FROM result").pl()

        log_memory("add_popular_random_negatives: collected result")

        con.close()

    log_memory("add_popular_random_negatives: duckdb closed")

    # Cast event_type to Categorical for consistency with original
    with pl.StringCache():
        result = result.with_columns(pl.col("event_type").cast(pl.Categorical))

    log_memory("add_popular_random_negatives: done")

    return result
