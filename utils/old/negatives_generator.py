import polars as pl


SECONDS_IN_DAY = 86_400


def prepare_base(df: pl.DataFrame) -> pl.LazyFrame:
    return df.lazy().with_columns(
        pl.col("timestamp").cast(pl.Int64),
        pl.from_epoch(pl.col("timestamp"), time_unit="s").alias("ts"),
        pl.from_epoch(pl.col("timestamp"), time_unit="s")
        .dt.truncate("1d")
        .alias("day"),
    )


def rolling_item_stats(
    base: pl.LazyFrame,
    *,
    window_days: int,
    max_day_epoch_s: int,
) -> pl.LazyFrame:
    max_day = pl.from_epoch(
        pl.lit(max_day_epoch_s), time_unit="s"
    ).dt.truncate("1d")

    return (
        base.sort(["day", "item_id"])
        .group_by_dynamic(
            index_column="day",
            every="1d",
            period=f"{window_days}d",
            offset=f"-{window_days}d",  # makes label=right correspond to the window end day
            closed="left",  # [day-window, day)
            label="right",
            start_by="window",  # generate windows on the regular grid (don’t depend on datapoints)
            group_by="item_id",
        )
        .agg(
            pl.len().alias("count"),
            pl.col("is_organic").mean().alias("org_rate"),
        )
        .filter(pl.col("day") <= max_day)
        .select(["day", "item_id", "count", "org_rate"])
    )


def top_items_with_buckets(
    roll: pl.LazyFrame,
    *,
    top_n: int,
    k: int,
    seed: int,
) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    top = (
        roll.group_by("day")
        .agg(
            pl.col("item_id")
            .sort_by("count", descending=True)
            .head(top_n)
            .alias("item_id"),
            pl.col("count")
            .sort_by("count", descending=True)
            .head(top_n)
            .alias("count"),
            pl.col("org_rate")
            .sort_by("count", descending=True)
            .head(top_n)
            .alias("org_rate"),
        )
        .explode(["item_id", "count", "org_rate"])
        .with_columns(
            (pl.len().over("day") // k)
            .clip(lower_bound=1)
            .alias("bucket_count")
        )
        .with_columns(
            (
                pl.struct(["item_id", "day"]).hash(seed)
                % pl.col("bucket_count")
            ).alias("bucket")
        )
    )

    day_bucket = top.select(["day", "bucket_count"]).unique()
    return top, day_bucket


def users_with_buckets(
    base: pl.LazyFrame,
    day_bucket: pl.LazyFrame,
    *,
    seed: int,
) -> pl.LazyFrame:
    return (
        base.select(["uid", "day"])
        .unique()
        .join(day_bucket, on="day", how="inner")
        .with_columns(
            (
                pl.struct(["uid", "day"]).hash(seed) % pl.col("bucket_count")
            ).alias("bucket")
        )
    )


def generate_candidates(
    *,
    users: pl.LazyFrame,
    top: pl.LazyFrame,
    seen: pl.LazyFrame,
    item_meta: pl.LazyFrame,
    k: int,
    seed: int,
) -> pl.LazyFrame:
    return (
        users.join(
            top.select(["day", "item_id", "bucket", "org_rate"]),
            on=["day", "bucket"],
            how="inner",
        )
        .select(["uid", "item_id", "day", "org_rate"])
        .join(seen, on=["uid", "item_id"], how="anti")
        .with_columns(
            pl.struct(["uid", "item_id", "day"]).hash(seed + 123).alias("rand")
        )
        .group_by("uid")
        .agg(
            pl.col("item_id").sort_by("rand").head(k).alias("item_id"),
            pl.col("day").sort_by("rand").head(k).alias("day"),
            pl.col("org_rate").sort_by("rand").head(k).alias("org_rate"),
        )
        .explode(["item_id", "day", "org_rate"])
        .unique(["uid", "item_id"])
        .with_columns(
            (
                pl.col("day").dt.epoch("s")
                + (
                    pl.struct(["uid", "item_id"]).hash(seed + 555)
                    % SECONDS_IN_DAY
                )
            )
            .cast(pl.Int64)
            .alias("timestamp")
        )
        .join(item_meta, on="item_id", how="left")
        .with_columns(
            (
                pl.struct(["uid", "item_id", "day"])
                .hash(seed + 777)
                .cast(pl.Float64)
                / (2**64)
                < pl.col("org_rate").fill_null(0.0)
            ).alias("is_organic"),
            pl.lit("random_negative").cast(pl.Categorical).alias("event_type"),
        )
        .drop(["org_rate"])
    )


def add_popular_random_negatives(
    df: pl.DataFrame,
    k: int,
    top_n: int,
    seed: int,
    window_days: int,
) -> pl.DataFrame:
    max_ts = int(df.get_column("timestamp").max())
    max_day_epoch_s = (max_ts // SECONDS_IN_DAY) * SECONDS_IN_DAY

    base = prepare_base(df)

    seen = base.select(["uid", "item_id"]).unique()
    item_meta = base.select(["item_id", "artist_id", "album_id"]).unique()

    rolling_stats = rolling_item_stats(
        base,
        window_days=window_days,
        max_day_epoch_s=max_day_epoch_s,
    )
    top, day_bucket = top_items_with_buckets(
        rolling_stats, top_n=top_n, k=k, seed=seed
    )

    users = users_with_buckets(base, day_bucket, seed=seed + 9999)

    candidates = generate_candidates(
        users=users,
        top=top,
        seen=seen,
        item_meta=item_meta,
        k=k,
        seed=seed,
    )

    out = pl.concat([base.drop(["ts"]), candidates], how="diagonal_relaxed")

    with pl.StringCache():
        return out.collect(engine="streaming")
