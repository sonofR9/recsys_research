import math
from pathlib import Path

import polars as pl

from .config import FieldConfig, get_counter_columns


class EmaCounter:
    def __init__(
        self,
        keys: list[str],
        fields: list[FieldConfig],
        output_dir: Path,
        name: str | None = None,
    ):
        self.keys = keys
        self.fields = fields
        self.name = name or "_".join(keys)

        self.counter_dir = Path(output_dir) / self.name
        self.counter_dir.mkdir(parents=True, exist_ok=True)

        self.ema_cols = get_counter_columns(self.fields)
        self.alphas = {
            decay.out_column_suffix: math.exp(-math.log(2) / decay.half_life_days)
            for field in self.fields
            for decay in field.decays
        }

    def get_output_columns(self) -> list[str]:
        return self.ema_cols

    def _state_path(self, day: int) -> Path:
        return self.counter_dir / f"day_{day:04d}.parquet"

    def _cache_path(self, day: int) -> Path:
        return self.counter_dir / f"cache_day_{day:04d}.parquet"

    def _load_or_create_state(self, day: int) -> pl.DataFrame:
        state_path = self._state_path(day)
        if state_path.exists():
            return pl.read_parquet(state_path)

        schema = {key: pl.Int64 for key in self.keys} | {
            col: pl.Float64 for col in self.ema_cols
        }
        return pl.DataFrame(schema=schema)

    def _compute_new_state(
        self, day_df: pl.DataFrame, state_df: pl.DataFrame
    ) -> pl.DataFrame:
        agg_exprs = []
        update_exprs = []
        for field in self.fields:
            inc_expr = (
                pl.lit(1.0)
                if field.condition is None
                else eval(field.condition).cast(pl.Float64)
            )
            sum_col = f"_sum_{field.out_column_prefix}"
            agg_exprs.append(inc_expr.sum().alias(sum_col))

            for decay in field.decays:
                col = f"{field.out_column_prefix}_{decay.out_column_suffix}_ema"
                update_exprs.append(
                    (
                        pl.col(col) * self.alphas[decay.out_column_suffix]
                        + pl.col(sum_col)
                    ).alias(col)
                )

        return (
            day_df.group_by(self.keys)
            .agg(agg_exprs)
            .join(state_df, on=self.keys, how="left")
            .with_columns([pl.col(col).fill_null(0.0) for col in self.ema_cols])
            .with_columns(update_exprs)
            .select(self.keys + self.ema_cols)
        )

    def process_day(
        self,
        day: int,
        day_df: pl.DataFrame,
        invalidate_cache: bool = False,
    ) -> pl.DataFrame:
        cache_path = self._cache_path(day)
        if not invalidate_cache and cache_path.exists():
            return pl.read_parquet(cache_path)

        state_df = self._load_or_create_state(day)

        result_df = day_df.join(state_df, on=self.keys, how="left").with_columns(
            [pl.col(col).fill_null(0.0) for col in self.ema_cols]
        )
        result_df.write_parquet(cache_path)

        new_state = self._compute_new_state(day_df, state_df)
        new_state.write_parquet(self._state_path(day + 1))

        return result_df
