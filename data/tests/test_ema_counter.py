import math
import shutil
import tempfile
import time
from collections.abc import Generator
from pathlib import Path

import polars as pl
import pytest

from data.counters.config import DecayConfig, FieldConfig
from data.counters.counter import EmaCounter

_LIST_KEY_SCHEMA = {
    "uid": pl.Int64,
    "artist_id": pl.List(pl.Int64),
    "event_type": pl.Utf8,
}


def _field(event_type: str, *half_lives: float) -> FieldConfig:
    return FieldConfig.matching(
        "event_type", event_type, [DecayConfig(days) for days in half_lives or (7,)]
    )


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    tmp = tempfile.mkdtemp()
    yield Path(tmp)
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def cache_dir(temp_dir: Path) -> Path:
    return temp_dir / "counters"


@pytest.fixture
def simple_counter(cache_dir: Path) -> EmaCounter:
    return EmaCounter(
        cache_dir=cache_dir,
        keys=["item_id"],
        fields=[_field("like", 7)],
    )


@pytest.fixture
def multi_field_counter(cache_dir: Path) -> EmaCounter:
    return EmaCounter(
        cache_dir=cache_dir,
        keys=["item_id"],
        fields=[
            _field("like", 7),
            _field("dislike", 7),
        ],
    )


class TestEmaCounter:
    def test_get_output_columns(self, simple_counter: EmaCounter) -> None:
        columns = simple_counter.get_output_columns()
        assert len(columns) == 1
        assert "item_id_like_7d_ema_mean" in columns

    def test_get_output_columns_multi_field(
        self, multi_field_counter: EmaCounter
    ) -> None:
        columns = multi_field_counter.get_output_columns()
        assert len(columns) == 2
        assert "item_id_like_7d_ema_mean" in columns
        assert "item_id_dislike_7d_ema_mean" in columns

    def test_ema_decay_single_day(self, simple_counter: EmaCounter) -> None:
        day_df = pl.DataFrame(
            {
                "item_id": [1, 1, 2, 3],
                "event_type": ["like", "like", "like", "dislike"],
            }
        )

        result = simple_counter.process_day(0, day_df)

        assert "item_id_like_7d_ema_mean" in result.columns

        item1_ema = result.filter(pl.col("item_id") == 1)["item_id_like_7d_ema_mean"]
        assert all(value == 0.0 for value in item1_ema.to_list())

    def test_ema_decay_multiple_days(self, simple_counter: EmaCounter) -> None:
        day0_df = pl.DataFrame(
            {
                "item_id": [1, 1],
                "event_type": ["like", "like"],
            }
        )
        simple_counter.process_day(0, day0_df)

        day1_df = pl.DataFrame(
            {
                "item_id": [1],
                "event_type": ["like"],
            }
        )
        result = simple_counter.process_day(1, day1_df)

        alpha = math.exp(-math.log(2) / 7)
        expected_ema = 2.0 * alpha

        actual_ema = result.filter(pl.col("item_id") == 1)["item_id_like_7d_ema_mean"][
            0
        ]
        assert actual_ema == pytest.approx(expected_ema, rel=1e-6)

    def test_each_day_of_age_costs_one_more_decay(
        self, simple_counter: EmaCounter
    ) -> None:
        # One like per day for three days; on day 3 they are 3, 2 and 1 days old.
        for day in range(3):
            simple_counter.process_day(
                day, pl.DataFrame({"item_id": [1], "event_type": ["like"]})
            )

        result = simple_counter.process_day(
            3, pl.DataFrame({"item_id": [1], "event_type": ["like"]})
        )

        alpha = math.exp(-math.log(2) / 7)
        expected = alpha**3 + alpha**2 + alpha
        actual = result.filter(pl.col("item_id") == 1)["item_id_like_7d_ema_mean"][0]
        assert actual == pytest.approx(expected, rel=1e-6)

    def test_a_field_without_an_event_type_counts_every_row(
        self, cache_dir: Path
    ) -> None:
        counter = EmaCounter(
            cache_dir=cache_dir,
            keys=["item_id"],
            fields=[FieldConfig(name="all", decays=[DecayConfig(7.0)])],
        )
        day0_df = pl.DataFrame(
            {
                "item_id": [1, 1, 2],
                "event_type": ["like", "dislike", "like"],
            }
        )
        counter.process_day(0, day0_df)

        result = counter.process_day(
            1, pl.DataFrame({"item_id": [1, 2], "event_type": ["like", "like"]})
        )

        alpha = math.exp(-math.log(2) / 7)
        emas = result.sort("item_id")["item_id_all_7d_ema_mean"].to_list()
        assert emas == pytest.approx([2.0 * alpha, 1.0 * alpha], rel=1e-6)

    def test_an_idle_day_decays_a_key_instead_of_forgetting_it(
        self, simple_counter: EmaCounter
    ) -> None:
        simple_counter.process_day(
            0, pl.DataFrame({"item_id": [1, 2], "event_type": ["like", "like"]})
        )
        simple_counter.process_day(
            1, pl.DataFrame({"item_id": [1], "event_type": ["like"]})
        )

        result = simple_counter.process_day(
            2, pl.DataFrame({"item_id": [1, 2], "event_type": ["like", "like"]})
        )

        alpha = math.exp(-math.log(2) / 7)
        item2_ema = result.filter(pl.col("item_id") == 2)["item_id_like_7d_ema_mean"][0]
        assert item2_ema == pytest.approx(alpha**2, rel=1e-6)

    def test_a_key_decayed_into_noise_is_dropped_from_the_state(
        self, cache_dir: Path
    ) -> None:
        counter = EmaCounter(
            cache_dir=cache_dir,
            keys=["item_id"],
            fields=[_field("like", 1)],
        )
        counter.process_day(0, pl.DataFrame({"item_id": [1], "event_type": ["like"]}))

        for day in range(1, 22):
            counter.process_day(
                day, pl.DataFrame({"item_id": [2], "event_type": ["like"]})
            )

        state = pl.read_parquet(counter.state_path(22))
        assert state["item_id"].to_list() == [2]

    def test_state_persistence(self, simple_counter: EmaCounter) -> None:
        day0_df = pl.DataFrame(
            {
                "item_id": [1, 1, 1],
                "event_type": ["like", "like", "like"],
            }
        )
        simple_counter.process_day(0, day0_df)

        state_path = simple_counter.state_path(1)
        assert state_path.exists()

        state_df = pl.read_parquet(state_path)
        assert "item_id" in state_df.columns
        assert "item_id_like_7d_ema" in state_df.columns

    def test_new_keys_get_zero_initial(self, simple_counter: EmaCounter) -> None:
        day0_df = pl.DataFrame(
            {
                "item_id": [1],
                "event_type": ["like"],
            }
        )
        simple_counter.process_day(0, day0_df)

        day1_df = pl.DataFrame(
            {
                "item_id": [2],
                "event_type": ["like"],
            }
        )
        result = simple_counter.process_day(1, day1_df)

        item2_ema = result.filter(pl.col("item_id") == 2)["item_id_like_7d_ema_mean"][0]
        assert item2_ema == 0.0

    def test_multiple_fields_work_together(
        self, multi_field_counter: EmaCounter
    ) -> None:
        day_df = pl.DataFrame(
            {
                "item_id": [1, 1, 1],
                "event_type": ["like", "like", "dislike"],
            }
        )

        result = multi_field_counter.process_day(0, day_df)

        assert "item_id_like_7d_ema_mean" in result.columns
        assert "item_id_dislike_7d_ema_mean" in result.columns

    def test_cache_hit_returns_cached_result(self, simple_counter: EmaCounter) -> None:
        day_df = pl.DataFrame(
            {
                "item_id": [1],
                "event_type": ["like"],
            }
        )

        result1 = simple_counter.process_day(0, day_df)
        result2 = simple_counter.process_day(0, day_df, invalidate_cache=False)

        assert result1.equals(result2)

    def test_invalidate_cache_forces_recomputation(
        self, simple_counter: EmaCounter
    ) -> None:
        day_df = pl.DataFrame(
            {
                "item_id": [1],
                "event_type": ["like"],
            }
        )

        simple_counter.process_day(0, day_df)

        cache_path = simple_counter.result_path(0)
        original_mtime = cache_path.stat().st_mtime

        time.sleep(0.01)

        simple_counter.process_day(0, day_df, invalidate_cache=True)
        new_mtime = cache_path.stat().st_mtime

        assert new_mtime > original_mtime

    def test_multi_key_counter(self, cache_dir: Path) -> None:
        counter = EmaCounter(
            cache_dir=cache_dir,
            keys=["user_id", "item_id"],
            fields=[_field("like", 7)],
        )

        day_df = pl.DataFrame(
            {
                "user_id": [1, 1, 2],
                "item_id": [10, 10, 10],
                "event_type": ["like", "like", "like"],
            }
        )

        result = counter.process_day(0, day_df)

        assert "user_id_item_id_like_7d_ema_mean" in result.columns

    def test_multiple_decay_rates(self, cache_dir: Path) -> None:
        counter = EmaCounter(
            cache_dir=cache_dir,
            keys=["item_id"],
            fields=[
                FieldConfig(
                    name="like",
                    condition=pl.col("event_type") == "like",
                    decays=[
                        DecayConfig(half_life_days=1),
                        DecayConfig(half_life_days=7),
                        DecayConfig(half_life_days=30),
                    ],
                )
            ],
        )

        columns = counter.get_output_columns()
        assert len(columns) == 3
        assert "item_id_like_1d_ema_mean" in columns
        assert "item_id_like_7d_ema_mean" in columns
        assert "item_id_like_30d_ema_mean" in columns

    def test_aggregations_produce_a_column_per_aggregation(
        self, cache_dir: Path
    ) -> None:
        counter = EmaCounter(
            cache_dir=cache_dir,
            keys=["uid", "artist_id"],
            fields=[_field("like", 7)],
            aggregations=("min", "max", "mean", "sum", "std"),
        )

        assert counter.get_output_columns() == [
            "uid_artist_id_like_7d_ema_min",
            "uid_artist_id_like_7d_ema_max",
            "uid_artist_id_like_7d_ema_mean",
            "uid_artist_id_like_7d_ema_sum",
            "uid_artist_id_like_7d_ema_std",
        ]

    def test_list_valued_key_updates_each_expanded_entry(self, cache_dir: Path) -> None:
        counter = EmaCounter(
            cache_dir=cache_dir,
            keys=["uid", "artist_id"],
            fields=[_field("like", 7)],
        )

        day0 = pl.DataFrame(
            {"uid": [1], "artist_id": [[1, 2]], "event_type": ["like"]},
            schema=_LIST_KEY_SCHEMA,
        )
        counter.process_day(0, day0)

        state = pl.read_parquet(counter.state_path(1)).sort("artist_id")
        assert state["uid"].to_list() == [1, 1]
        assert state["artist_id"].to_list() == [1, 2]
        alpha = math.exp(-math.log(2) / 7)
        assert state["uid_artist_id_like_7d_ema"].to_list() == pytest.approx(
            [alpha, alpha]
        )

    def test_list_valued_key_aggregates_per_event(self, cache_dir: Path) -> None:
        counter = EmaCounter(
            cache_dir=cache_dir,
            keys=["uid", "artist_id"],
            fields=[_field("like", 7)],
            aggregations=("min", "max", "mean", "sum", "std"),
        )

        day0 = pl.DataFrame(
            {
                "uid": [1, 1, 1],
                "artist_id": [[1], [1], [2]],
                "event_type": ["like"] * 3,
            },
            schema=_LIST_KEY_SCHEMA,
        )
        counter.process_day(0, day0)

        day1 = pl.DataFrame(
            {"uid": [1], "artist_id": [[1, 2]], "event_type": ["like"]},
            schema=_LIST_KEY_SCHEMA,
        )
        row = counter.process_day(1, day1).row(0, named=True)

        alpha = math.exp(-math.log(2) / 7)
        assert row["uid_artist_id_like_7d_ema_min"] == pytest.approx(1.0 * alpha)
        assert row["uid_artist_id_like_7d_ema_max"] == pytest.approx(2.0 * alpha)
        assert row["uid_artist_id_like_7d_ema_mean"] == pytest.approx(1.5 * alpha)
        assert row["uid_artist_id_like_7d_ema_sum"] == pytest.approx(3.0 * alpha)
        assert row["uid_artist_id_like_7d_ema_std"] == pytest.approx(0.5 * alpha)

    def test_list_valued_enrichment_preserves_rows(self, cache_dir: Path) -> None:
        counter = EmaCounter(
            cache_dir=cache_dir,
            keys=["uid", "artist_id"],
            fields=[_field("like", 7)],
        )

        day = pl.DataFrame(
            {"uid": [1, 2], "artist_id": [[1, 2], [3]], "event_type": ["like", "like"]},
            schema=_LIST_KEY_SCHEMA,
        )
        result = counter.process_day(0, day)

        assert result.height == 2
        assert result["artist_id"].to_list() == [[1, 2], [3]]
