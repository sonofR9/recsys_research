import polars as pl
from typing import Sequence


def add_running_event_feature(
    lf: pl.LazyFrame,
    group_cols: Sequence[str],
    event_type: str,
    window_days: int,
) -> pl.LazyFrame:
    group_name = "_".join(group_cols)
    feature_name = f"{group_name}_{event_type}s_{window_days}d"

    lf = lf.with_columns(
        pl.from_epoch("timestamp", time_unit="s")
        .dt.replace_time_zone("UTC")
        .dt.truncate("1d")
        .alias("day")
    )

    spine = lf.select([*group_cols, "day"]).unique()

    daily_counts = (
        lf.filter(pl.col("event_type") == event_type)
        .group_by([*group_cols, "day"])
        .agg(pl.len().alias("cnt"))
    )

    daily = (
        spine.join(
            daily_counts,
            on=[*group_cols, "day"],
            how="left",
        )
        .with_columns(pl.col("cnt").fill_null(0))
        .sort([*group_cols, "day"])
    )

    rolling = daily.rolling(
        index_column="day",
        period=f"{window_days}d",
        closed="left",
        group_by=group_cols,
    ).agg(pl.col("cnt").sum().alias(feature_name))

    lf = lf.join(
        rolling.select([*group_cols, "day", feature_name]),
        on=[*group_cols, "day"],
        how="left",
    )

    return lf
