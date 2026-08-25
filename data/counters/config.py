from dataclasses import dataclass, field
from typing import Literal

import polars as pl

CounterAggregation = Literal["min", "max", "mean", "sum", "std"]


@dataclass
class DecayConfig:
    half_life_days: float
    out_column_suffix: str = field(default="")

    def __post_init__(self):
        if not self.out_column_suffix:
            self.out_column_suffix = f"{self.half_life_days:g}d_ema"


@dataclass
class FieldConfig:
    """Which rows a counter counts, under which name.

    A bare ``condition`` counts every row. Prefer :meth:`matching` when the
    rows are picked by one column's value: it derives the name and the filter
    from that value together, so they cannot drift apart.
    """

    name: str
    decays: list[DecayConfig]
    condition: pl.Expr | None = None

    @classmethod
    def matching(
        cls, column: str, value: str | None, decays: list[DecayConfig]
    ) -> "FieldConfig":
        """Count the rows whose ``column`` holds ``value``, named after it.

        ``value=None`` counts every row, under the name ``all``. The column is
        the caller's to name: a counter knows nothing about the dataset it runs
        on.
        """
        if value is None:
            return cls(name="all", decays=decays)
        return cls(name=value, decays=decays, condition=pl.col(column) == value)


def get_base_counter_columns(keys: list[str], fields: list[FieldConfig]) -> list[str]:
    keys_str = "_".join(keys)
    return [
        f"{keys_str}_{f.name}_{d.out_column_suffix}" for f in fields for d in f.decays
    ]


def get_aggregated_counter_columns(
    keys: list[str],
    fields: list[FieldConfig],
    aggregations: list[CounterAggregation],
) -> list[str]:
    return [
        f"{base}_{aggregation}"
        for base in get_base_counter_columns(keys, fields)
        for aggregation in aggregations
    ]
