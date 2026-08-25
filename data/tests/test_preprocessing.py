import shutil
import tempfile
from collections.abc import Callable, Generator
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import polars as pl
import pytest

from data.preprocessing import preprocess_counters


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    tmp = tempfile.mkdtemp()
    yield Path(tmp)
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def sample_day_files(temp_dir: Path) -> dict[int, Path]:
    day0_df = pl.DataFrame(
        {
            "item_id": [1, 1, 2],
            "uid": [100, 100, 101],
            "event_type": ["like", "like", "dislike"],
            "timestamp": [100, 200, 300],
        }
    )

    day1_df = pl.DataFrame(
        {
            "item_id": [1, 2, 3],
            "uid": [100, 101, 102],
            "event_type": ["like", "like", "like"],
            "timestamp": [86_500, 86_600, 86_700],
        }
    )

    day2_df = pl.DataFrame(
        {
            "item_id": [2, 3, 4],
            "uid": [101, 102, 103],
            "event_type": ["dislike", "like", "listen"],
            "timestamp": [173_000, 173_100, 173_200],
        }
    )

    day0_path = temp_dir / "day_0000.parquet"
    day1_path = temp_dir / "day_0001.parquet"
    day2_path = temp_dir / "day_0002.parquet"

    day0_df.write_parquet(day0_path)
    day1_df.write_parquet(day1_path)
    day2_df.write_parquet(day2_path)

    return {
        0: day0_path,
        1: day1_path,
        2: day2_path,
    }


COUNTER_COLUMNS = ["uid_like_7d_ema", "uid_like_30d_ema", "item_id_like_7d_ema"]


def _counter(
    *columns: str, on_call: Callable[[int, bool], None] | None = None
) -> MagicMock:
    """A counter that appends one column per name, with row-dependent values."""

    def process_day(
        day: int,
        day_df: pl.DataFrame,
        invalidate_cache: bool = False,
    ) -> pl.DataFrame:
        if on_call is not None:
            on_call(day, invalidate_cache)
        return day_df.with_columns(
            [
                pl.Series(
                    name, [index + row / 10 for row in range(len(day_df))], pl.Float64
                )
                for index, name in enumerate(columns, 1)
            ]
        )

    counter = MagicMock()
    counter.get_output_columns.return_value = list(columns)
    counter.process_day.side_effect = process_day
    return counter


class TestPreprocessCounters:
    def test_preprocess_counters_enriches_data(
        self,
        temp_dir: Path,
        sample_day_files: dict[int, Path],
    ) -> None:
        result = preprocess_counters(
            counters=[_counter("item_id_like_7d_ema")],
            counter_columns=["item_id_like_7d_ema"],
            day_to_path=sample_day_files,
            days=[0, 1, 2],
            output_dir=temp_dir / "packed",
        )

        assert sorted(result) == [0, 1, 2]
        assert all(path.exists() for path in result.values())
        assert "item_id_like_7d_ema" in pl.read_parquet(result[0]).columns

    def test_preprocess_counters_all_days_processed(
        self,
        temp_dir: Path,
        sample_day_files: dict[int, Path],
    ) -> None:
        processed_days: list[int] = []

        preprocess_counters(
            counters=[
                _counter(
                    "counter_col", on_call=lambda day, _: processed_days.append(day)
                )
            ],
            counter_columns=["counter_col"],
            day_to_path=sample_day_files,
            days=[0, 1, 2],
            output_dir=temp_dir / "packed",
        )

        assert sorted(processed_days) == [0, 1, 2]

    def test_preprocess_counters_multiple_counters(
        self,
        temp_dir: Path,
        sample_day_files: dict[int, Path],
    ) -> None:
        counter1_calls: list[int] = []
        counter2_calls: list[int] = []

        preprocess_counters(
            counters=[
                _counter(
                    "counter1_col", on_call=lambda day, _: counter1_calls.append(day)
                ),
                _counter(
                    "counter2_col", on_call=lambda day, _: counter2_calls.append(day)
                ),
            ],
            counter_columns=["counter1_col", "counter2_col"],
            day_to_path=sample_day_files,
            days=[0, 1],
            output_dir=temp_dir / "packed",
        )

        assert sorted(counter1_calls) == [0, 1]
        assert sorted(counter2_calls) == [0, 1]

    def test_preprocess_counters_invalidate_cache_passed(
        self,
        temp_dir: Path,
        sample_day_files: dict[int, Path],
    ) -> None:
        invalidate_values: list[bool] = []

        preprocess_counters(
            counters=[
                _counter(
                    "counter_col",
                    on_call=lambda _, invalidate: invalidate_values.append(invalidate),
                )
            ],
            counter_columns=["counter_col"],
            day_to_path=sample_day_files,
            days=[0, 1],
            output_dir=temp_dir / "packed",
            invalidate_cache=True,
        )

        assert all(value is True for value in invalidate_values)

    def test_preprocess_counters_empty_counters_list(
        self,
        temp_dir: Path,
        sample_day_files: dict[int, Path],
    ) -> None:
        result = preprocess_counters(
            counters=[],
            counter_columns=[],
            day_to_path=sample_day_files,
            days=[0, 1, 2],
            output_dir=temp_dir / "packed",
        )

        assert result == {}

    def test_preprocess_counters_subset_of_days(
        self,
        temp_dir: Path,
        sample_day_files: dict[int, Path],
    ) -> None:
        processed_days: list[int] = []

        result = preprocess_counters(
            counters=[
                _counter(
                    "counter_col", on_call=lambda day, _: processed_days.append(day)
                )
            ],
            counter_columns=["counter_col"],
            day_to_path=sample_day_files,
            days=[1],
            output_dir=temp_dir / "packed",
        )

        assert processed_days == [1]
        assert list(result.keys()) == [1]

    def test_preprocess_counters_days_processed_in_order(
        self,
        temp_dir: Path,
        sample_day_files: dict[int, Path],
    ) -> None:
        processed_days: list[int] = []

        preprocess_counters(
            counters=[
                _counter(
                    "counter_col", on_call=lambda day, _: processed_days.append(day)
                )
            ],
            counter_columns=["counter_col"],
            day_to_path=sample_day_files,
            days=[2, 0, 1],
            output_dir=temp_dir / "packed",
        )

        assert processed_days == [0, 1, 2]


class TestPackedCounterColumn:
    def _run(
        self,
        temp_dir: Path,
        day_files: dict[int, Path],
        days: list[int],
        counter_columns: list[str] | None = None,
        invalidate_cache: bool = False,
    ) -> dict[int, Path]:
        return preprocess_counters(
            counters=[_counter(*COUNTER_COLUMNS)],
            counter_columns=counter_columns or COUNTER_COLUMNS,
            day_to_path=day_files,
            days=days,
            output_dir=temp_dir / "packed",
            invalidate_cache=invalidate_cache,
        )

    def test_packs_counters_into_one_array_column(
        self, temp_dir: Path, sample_day_files: dict[int, Path]
    ) -> None:
        result = self._run(temp_dir, sample_day_files, [0, 1])

        packed = pl.read_parquet(result[0])
        assert packed.schema["counters"] == pl.Array(pl.Float32, len(COUNTER_COLUMNS))
        np.testing.assert_allclose(
            packed["counters"].to_numpy(),
            [[1.0, 2.0, 3.0], [1.1, 2.1, 3.1], [1.2, 2.2, 3.2]],
            rtol=1e-6,
        )

    def test_keeps_the_individual_counter_columns(
        self, temp_dir: Path, sample_day_files: dict[int, Path]
    ) -> None:
        result = self._run(temp_dir, sample_day_files, [0])

        packed = pl.read_parquet(result[0])
        for column in COUNTER_COLUMNS:
            assert column in packed.columns
        assert packed.columns[-1] == "counters"

    def test_preserves_counter_column_order(
        self, temp_dir: Path, sample_day_files: dict[int, Path]
    ) -> None:
        result = self._run(
            temp_dir,
            sample_day_files,
            [0],
            counter_columns=list(reversed(COUNTER_COLUMNS)),
        )

        np.testing.assert_allclose(
            pl.read_parquet(result[0])["counters"].to_numpy(),
            [[3.0, 2.0, 1.0], [3.1, 2.1, 1.1], [3.2, 2.2, 1.2]],
            rtol=1e-6,
        )

    def test_reuses_cache_when_not_invalidated(
        self, temp_dir: Path, sample_day_files: dict[int, Path]
    ) -> None:
        result = self._run(temp_dir, sample_day_files, [0])
        pl.DataFrame({"sentinel": [1]}).write_parquet(result[0])

        self._run(temp_dir, sample_day_files, [0])

        assert pl.read_parquet(result[0]).columns == ["sentinel"]

    def test_invalidate_cache_rebuilds(
        self, temp_dir: Path, sample_day_files: dict[int, Path]
    ) -> None:
        result = self._run(temp_dir, sample_day_files, [0])
        pl.DataFrame({"sentinel": [1]}).write_parquet(result[0])

        self._run(temp_dir, sample_day_files, [0], invalidate_cache=True)

        assert "counters" in pl.read_parquet(result[0]).columns
