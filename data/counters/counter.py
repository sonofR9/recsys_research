import hashlib
import math
from collections.abc import Callable, Sequence
from pathlib import Path

import polars as pl

from .config import (
    CounterAggregation,
    FieldConfig,
    get_aggregated_counter_columns,
    get_base_counter_columns,
)

_AGGREGATION_EXPRS: dict[CounterAggregation, Callable[[str], pl.Expr]] = {
    "min": lambda col: pl.col(col).min(),
    "max": lambda col: pl.col(col).max(),
    "mean": lambda col: pl.col(col).mean(),
    "sum": lambda col: pl.col(col).sum(),
    "std": lambda col: pl.col(col).std(ddof=0),
}

_ROW_INDEX = "__row"
_STATE_EPSILON = 1e-6


class EmaCounter:
    """EMA counter keyed on one or more columns, one column per (field, decay)."""

    def __init__(
        self,
        keys: list[str],
        fields: list[FieldConfig],
        cache_dir: Path,
        aggregations: Sequence[CounterAggregation] = ("mean",),
    ):
        self.keys = keys
        self.fields = fields
        self.aggregations = list(aggregations)

        self.ema_cols = get_base_counter_columns(self.keys, self.fields)
        self.output_cols = get_aggregated_counter_columns(
            self.keys, self.fields, self.aggregations
        )

        base_dir = Path(cache_dir) / self._cache_name()
        self.states_dir = base_dir / "states"
        self.results_dir = base_dir / "results"
        self.states_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.alphas = {
            decay.out_column_suffix: math.exp(-math.log(2) / decay.half_life_days)
            for field in self.fields
            for decay in field.decays
        }

    def _cache_name(self) -> str:
        shape = [*self.output_cols, *(str(field.condition) for field in self.fields)]
        digest = hashlib.sha1("\n".join(shape).encode()).hexdigest()
        return f"{'_'.join(self.keys)}_{digest[:8]}"

    def get_output_columns(self) -> list[str]:
        return self.output_cols

    def state_path(self, day: int | None = None) -> Path:
        if day is None:
            return self.states_dir
        return self.states_dir / f"day_{day:04d}.parquet"

    def result_path(self, day: int | None = None) -> Path:
        if day is None:
            return self.results_dir
        return self.results_dir / f"day_{day:04d}.parquet"

    def _load_or_create_state(self, day: int) -> pl.DataFrame:
        state_path = self.state_path(day)
        if state_path.exists():
            return pl.read_parquet(state_path)

        schema = {key: pl.Int64 for key in self.keys} | {
            col: pl.Float64 for col in self.ema_cols
        }
        return pl.DataFrame(schema=schema)

    def _explode_list_keys(self, df: pl.DataFrame) -> pl.DataFrame:
        for key in self.keys:
            dtype = df.schema[key]
            if isinstance(dtype, pl.List):
                df = df.explode(key).with_columns(pl.col(key).cast(dtype.inner))
        return df

    def _enrich(self, day_df: pl.DataFrame, state_df: pl.DataFrame) -> pl.DataFrame:
        indexed = day_df.with_row_index(_ROW_INDEX)
        per_key = (
            self._explode_list_keys(indexed)
            .join(state_df, on=self.keys, how="left")
            .with_columns([pl.col(col).fill_null(0.0) for col in self.ema_cols])
        )
        agg_exprs = [
            _AGGREGATION_EXPRS[aggregation](ema).alias(f"{ema}_{aggregation}")
            for ema in self.ema_cols
            for aggregation in self.aggregations
        ]
        per_row = per_key.group_by(_ROW_INDEX).agg(agg_exprs)
        return indexed.join(per_row, on=_ROW_INDEX, how="left").drop(_ROW_INDEX)

    def _compute_new_state(
        self, day_df: pl.DataFrame, state_df: pl.DataFrame
    ) -> pl.DataFrame:
        keys_str = "_".join(self.keys)
        agg_exprs = []
        update_exprs = []
        for field in self.fields:
            # A literal cannot be aggregated -- polars refuses `pl.lit(1.0).sum()`.
            increment = (
                pl.len().cast(pl.Float64)
                if field.condition is None
                else field.condition.cast(pl.Float64).sum()
            )
            sum_col = f"_sum_{field.name}"
            agg_exprs.append(increment.alias(sum_col))

            for decay in field.decays:
                col = f"{keys_str}_{field.name}_{decay.out_column_suffix}"
                update_exprs.append(
                    (
                        (pl.col(col) + pl.col(sum_col))
                        * self.alphas[decay.out_column_suffix]
                    ).alias(col)
                )

        expanded = self._explode_list_keys(day_df).drop_nulls(subset=self.keys)
        sum_cols = [f"_sum_{field.name}" for field in self.fields]
        # Full join, not left: a key with no event today still has a history, and
        # starting from today's keys alone would restart its EMA from zero.
        carried = (
            state_df.join(
                expanded.group_by(self.keys).agg(agg_exprs),
                on=self.keys,
                how="full",
                coalesce=True,
            )
            .with_columns(
                [pl.col(col).fill_null(0.0) for col in self.ema_cols + sum_cols]
            )
            .with_columns(update_exprs)
            .select(self.keys + self.ema_cols)
        )
        alive = pl.any_horizontal(
            [pl.col(col).abs() > _STATE_EPSILON for col in self.ema_cols]
        )
        return carried.filter(alive)

    def process_day(
        self,
        day: int,
        day_df: pl.DataFrame,
        invalidate_cache: bool = False,
    ) -> pl.DataFrame:
        cache_path = self.result_path(day)
        if not invalidate_cache and cache_path.exists():
            return pl.read_parquet(cache_path)

        state_df = self._load_or_create_state(day)

        result_df = self._enrich(day_df, state_df)
        result_df.write_parquet(cache_path)

        new_state = self._compute_new_state(day_df, state_df)
        new_state.write_parquet(self.state_path(day + 1))

        return result_df
