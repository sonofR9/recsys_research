from dataclasses import dataclass, field


@dataclass
class DecayConfig:
    half_life_days: float
    out_column_suffix: str = field(default="")

    def __post_init__(self):
        if not self.out_column_suffix:
            self.out_column_suffix = f"{int(self.half_life_days)}d"


@dataclass
class FieldConfig:
    condition: (
        str | None
    )  # Polars expression string, e.g., "pl.col('event_type') == 'like'". None = count all rows
    out_column_prefix: str
    decays: list[DecayConfig]


def get_counter_columns(fields: list[FieldConfig]) -> list[str]:
    return [
        f"{f.out_column_prefix}_{d.out_column_suffix}_ema"
        for f in fields
        for d in f.decays
    ]
