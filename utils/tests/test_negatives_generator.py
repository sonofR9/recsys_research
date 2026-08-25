import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest

from utils.negatives_generator import add_popular_random_negatives


@pytest.fixture
def temp_dir():
    tmp = tempfile.mkdtemp()
    yield Path(tmp)
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def mock_config(temp_dir):
    class MockConfig:
        preprocessed_path = temp_dir / "preprocessed"
        tmp_path = temp_dir / "tmp"

    MockConfig.preprocessed_path.mkdir(parents=True, exist_ok=True)
    MockConfig.tmp_path.mkdir(parents=True, exist_ok=True)

    with patch("utils.negatives_generator.global_config", MockConfig()):
        yield MockConfig()


@pytest.fixture
def sample_data(temp_dir):
    SECONDS_IN_DAY = 86400

    df = pl.DataFrame(
        {
            "uid": [1, 1, 1, 2, 2, 3, 3, 3, 3],
            "item_id": [10, 11, 12, 20, 21, 30, 31, 32, 33],
            "artist_id": [100, 100, 101, 200, 200, 300, 300, 301, 301],
            "album_id": [1000, 1000, 1001, 2000, 2000, 3000, 3000, 3001, 3001],
            "timestamp": [
                1 * SECONDS_IN_DAY + 100,
                1 * SECONDS_IN_DAY + 200,
                2 * SECONDS_IN_DAY + 100,
                1 * SECONDS_IN_DAY + 100,
                2 * SECONDS_IN_DAY + 100,
                1 * SECONDS_IN_DAY + 100,
                1 * SECONDS_IN_DAY + 200,
                2 * SECONDS_IN_DAY + 100,
                2 * SECONDS_IN_DAY + 200,
            ],
            "event_type": [
                "like",
                "like",
                "like",
                "like",
                "dislike",
                "like",
                "like",
                "like",
                "like",
            ],
            "is_organic": [True, True, False, True, True, True, False, True, True],
        }
    )

    path = temp_dir / "sample_data.parquet"
    df.write_parquet(path)
    return path, df


class TestAddPopularRandomNegatives:
    def test_negatives_count_within_bounds(self, mock_config, sample_data):
        input_path, original_df = sample_data
        min_k, max_k = 2, 5

        result = add_popular_random_negatives(
            input_path=str(input_path),
            min_k=min_k,
            max_k=max_k,
            multiplier=1.0,
            top_n=100,
            seed=42,
            window_days=1,
            return_dataframe=True,
        )

        negatives = result.filter(pl.col("event_type") == "random_negative")
        neg_counts = negatives.group_by("uid").agg(pl.count().alias("count"))

        for row in neg_counts.iter_rows(named=True):
            assert row["count"] >= min_k, (
                f"User {row['uid']} has {row['count']} negatives, expected >= {min_k}"
            )
            assert row["count"] <= max_k, (
                f"User {row['uid']} has {row['count']} negatives, expected <= {max_k}"
            )

    def test_no_seen_items_in_negatives(self, mock_config, sample_data):
        input_path, original_df = sample_data

        result = add_popular_random_negatives(
            input_path=str(input_path),
            min_k=2,
            max_k=10,
            multiplier=1.0,
            top_n=100,
            seed=42,
            window_days=1,
            return_dataframe=True,
        )

        negatives = result.filter(pl.col("event_type") == "random_negative")

        for uid in original_df["uid"].unique().to_list():
            seen_items = set(
                original_df.filter(pl.col("uid") == uid)["item_id"].to_list()
            )
            negative_items = set(
                negatives.filter(pl.col("uid") == uid)["item_id"].to_list()
            )

            overlap = seen_items & negative_items
            assert len(overlap) == 0, (
                f"User {uid} has seen items in negatives: {overlap}"
            )

    def test_output_schema(self, mock_config, sample_data):
        input_path, _ = sample_data

        result = add_popular_random_negatives(
            input_path=str(input_path),
            min_k=1,
            max_k=5,
            multiplier=1.0,
            top_n=100,
            seed=42,
            window_days=1,
            return_dataframe=True,
        )

        required_columns = [
            "uid",
            "item_id",
            "artist_id",
            "album_id",
            "timestamp",
            "is_organic",
            "event_type",
        ]
        for col in required_columns:
            assert col in result.columns, f"Missing column: {col}"

    def test_original_data_preserved(self, mock_config, sample_data):
        input_path, original_df = sample_data

        result = add_popular_random_negatives(
            input_path=str(input_path),
            min_k=1,
            max_k=5,
            multiplier=1.0,
            top_n=100,
            seed=42,
            window_days=1,
            return_dataframe=True,
        )

        non_negatives = result.filter(pl.col("event_type") != "random_negative")

        assert non_negatives.height == original_df.height, (
            "Original data rows were modified"
        )

    def test_weighted_mode_works_without_top_n(self, mock_config, sample_data):
        input_path, _ = sample_data

        result = add_popular_random_negatives(
            input_path=str(input_path),
            min_k=1,
            max_k=5,
            multiplier=1.0,
            top_n=None,
            seed=42,
            weighted=True,
            window_days=1,
            return_dataframe=True,
        )

        negatives = result.filter(pl.col("event_type") == "random_negative")
        assert negatives.height > 0, "No negatives generated in weighted mode"

    def test_from_whole_history_mode(self, mock_config, sample_data):
        input_path, _ = sample_data

        result = add_popular_random_negatives(
            input_path=str(input_path),
            min_k=1,
            max_k=5,
            multiplier=1.0,
            top_n=100,
            seed=42,
            from_whole_history=True,
            return_dataframe=True,
        )

        negatives = result.filter(pl.col("event_type") == "random_negative")
        assert negatives.height > 0, "No negatives generated in whole history mode"

    def test_top_n_required_when_not_weighted(self, mock_config, sample_data):
        input_path, _ = sample_data

        with pytest.raises(ValueError, match="top_n is required when weighted=False"):
            add_popular_random_negatives(
                input_path=str(input_path),
                min_k=1,
                max_k=5,
                multiplier=1.0,
                top_n=None,
                seed=42,
                weighted=False,
                return_dataframe=True,
            )

    def test_multiplier_affects_target_k(self, mock_config, sample_data):
        input_path, _ = sample_data

        result_low = add_popular_random_negatives(
            input_path=str(input_path),
            min_k=1,
            max_k=100,
            multiplier=0.5,
            top_n=100,
            seed=42,
            window_days=1,
            return_dataframe=True,
        )

        result_high = add_popular_random_negatives(
            input_path=str(input_path),
            min_k=1,
            max_k=100,
            multiplier=5.0,
            top_n=100,
            seed=42,
            window_days=1,
            return_dataframe=True,
        )

        neg_low = result_low.filter(pl.col("event_type") == "random_negative").height
        neg_high = result_high.filter(pl.col("event_type") == "random_negative").height

        assert neg_high >= neg_low, (
            "Higher multiplier should produce more or equal negatives"
        )

    @pytest.mark.parametrize("from_whole_history", [False, True])
    def test_no_duplicate_negatives_per_user(
        self, mock_config, sample_data, from_whole_history
    ):
        # The seed cross-join offers each candidate once per seed, so the
        # target_k cut has to come after deduplication or a user's quota gets
        # spent on repeats of a single item.
        input_path, _ = sample_data

        result = add_popular_random_negatives(
            input_path=str(input_path),
            min_k=3,
            max_k=5,
            multiplier=1.0,
            top_n=100,
            seed=42,
            window_days=1,
            from_whole_history=from_whole_history,
            return_dataframe=True,
        )

        negatives = result.filter(pl.col("event_type") == "random_negative")
        assert negatives.height > 0
        pairs = negatives.select("uid", "item_id")
        assert pairs.height == pairs.unique().height, (
            "duplicate (uid, item_id) negatives"
        )

    def test_users_reach_target_k_when_candidates_allow(self, mock_config, sample_data):
        # uid 2 saw only items 20 and 21, so the day-1 pool leaves it four
        # unseen candidates -- comfortably more than target_k.
        input_path, _ = sample_data

        result = add_popular_random_negatives(
            input_path=str(input_path),
            min_k=3,
            max_k=3,
            multiplier=1.0,
            top_n=100,
            seed=42,
            window_days=1,
            return_dataframe=True,
        )

        negatives = result.filter(
            (pl.col("event_type") == "random_negative") & (pl.col("uid") == 2)
        )
        assert negatives.height == 3
