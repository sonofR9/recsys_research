import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from utils.candidate_generators import (
    CounterStateSpec,
    DecayedPopularityGenerator,
    EnsembleCandidateGenerator,
    BaseCandidateGenerator,
)


@pytest.fixture
def temp_dir():
    tmp = tempfile.mkdtemp()
    yield Path(tmp)
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def mock_config(temp_dir):
    class MockConfig:
        candgen_path = temp_dir / "candgen"
        counters_path = temp_dir / "counters"

    MockConfig.candgen_path.mkdir(parents=True, exist_ok=True)
    MockConfig.counters_path.mkdir(parents=True, exist_ok=True)

    with patch("utils.candidate_generators.config", MockConfig()):
        yield MockConfig()


class TestCounterStateSpec:
    def test_out_column_single_key(self):
        spec = CounterStateSpec(
            key_columns=("item_id",),
            interaction_type="like",
            half_life_days=7,
        )

        out_col = spec.out_column
        assert "item_id" in out_col
        assert "like" in out_col
        assert "7d" in out_col

    def test_out_column_multi_key(self):
        spec = CounterStateSpec(
            key_columns=("user_id", "item_id"),
            interaction_type="like",
            half_life_days=7,
        )

        out_col = spec.out_column
        assert "user_id_item_id" in out_col

    def test_field_config_property(self):
        spec = CounterStateSpec(
            key_columns=("item_id",),
            interaction_type="like",
            half_life_days=14,
        )

        field_config = spec.field_config
        assert field_config.name == "like"
        assert str(field_config.condition) == str(pl.col("event_type") == "like")
        assert len(field_config.decays) == 1
        assert field_config.decays[0].half_life_days == 14

    def test_none_interaction_type(self):
        spec = CounterStateSpec(
            key_columns=("item_id",),
            interaction_type=None,
            half_life_days=7,
        )

        field_config = spec.field_config
        assert field_config.name == "all"
        assert field_config.condition is None


class TestDecayedPopularityGenerator:
    def test_fitting_ranks_items_by_decayed_likes(self, mock_config, temp_dir):
        days_dir = temp_dir / "days"
        days_dir.mkdir()
        for day in range(3):
            pl.DataFrame(
                {
                    "uid": [1, 2, 3, 4],
                    "item_id": [10, 10, 10, 20],
                    "event_type": ["like"] * 4,
                }
            ).write_parquet(days_dir / f"day_{day:04d}.parquet")

        generator = DecayedPopularityGenerator(decay_days=7).fit(days_dir)
        candidates = generator.generate_batch(
            pl.DataFrame({"uid": [1]}), day=2, n_candidates=2
        )

        assert candidates["item_id"].to_list() == [10, 20]


class TestEnsembleCandidateGenerator:
    def test_compute_n_candidates_per_generator_equal_weights(self, mock_config):
        mock_gen1 = MagicMock(spec=BaseCandidateGenerator)
        mock_gen1.is_fitted = True
        mock_gen2 = MagicMock(spec=BaseCandidateGenerator)
        mock_gen2.is_fitted = True

        ensemble = EnsembleCandidateGenerator(generators=[])
        ensemble.generators = [(mock_gen1, 1.0), (mock_gen2, 1.0)]

        allocations = ensemble._compute_n_candidates_per_generator(100)

        assert len(allocations) == 2
        assert sum(allocations) == 100

    def test_compute_n_candidates_per_generator_unequal_weights(self, mock_config):
        mock_gen1 = MagicMock(spec=BaseCandidateGenerator)
        mock_gen1.is_fitted = True
        mock_gen2 = MagicMock(spec=BaseCandidateGenerator)
        mock_gen2.is_fitted = True

        ensemble = EnsembleCandidateGenerator(generators=[])
        ensemble.generators = [(mock_gen1, 3.0), (mock_gen2, 1.0)]

        allocations = ensemble._compute_n_candidates_per_generator(100)

        assert len(allocations) == 2
        assert sum(allocations) == 100
        assert allocations[0] > allocations[1]

    def test_allocations_sum_to_n_candidates(self, mock_config):
        mock_gen1 = MagicMock(spec=BaseCandidateGenerator)
        mock_gen1.is_fitted = True
        mock_gen2 = MagicMock(spec=BaseCandidateGenerator)
        mock_gen2.is_fitted = True
        mock_gen3 = MagicMock(spec=BaseCandidateGenerator)
        mock_gen3.is_fitted = True

        ensemble = EnsembleCandidateGenerator(generators=[])
        ensemble.generators = [(mock_gen1, 1.0), (mock_gen2, 2.0), (mock_gen3, 3.0)]

        for n in [10, 50, 100, 137, 500]:
            allocations = ensemble._compute_n_candidates_per_generator(n)
            assert sum(allocations) == n, f"Allocations don't sum to {n}: {allocations}"

    def test_total_weight_property(self, mock_config):
        mock_gen1 = MagicMock(spec=BaseCandidateGenerator)
        mock_gen2 = MagicMock(spec=BaseCandidateGenerator)

        ensemble = EnsembleCandidateGenerator(generators=[])
        ensemble.generators = [(mock_gen1, 2.5), (mock_gen2, 1.5)]

        assert ensemble.total_weight == 4.0

    def test_is_fitted_all_fitted(self, mock_config):
        mock_gen1 = MagicMock(spec=BaseCandidateGenerator)
        mock_gen1.is_fitted = True
        mock_gen2 = MagicMock(spec=BaseCandidateGenerator)
        mock_gen2.is_fitted = True

        ensemble = EnsembleCandidateGenerator(generators=[])
        ensemble.generators = [(mock_gen1, 1.0), (mock_gen2, 1.0)]

        assert ensemble.is_fitted is True

    def test_is_fitted_some_not_fitted(self, mock_config):
        mock_gen1 = MagicMock(spec=BaseCandidateGenerator)
        mock_gen1.is_fitted = True
        mock_gen2 = MagicMock(spec=BaseCandidateGenerator)
        mock_gen2.is_fitted = False

        ensemble = EnsembleCandidateGenerator(generators=[])
        ensemble.generators = [(mock_gen1, 1.0), (mock_gen2, 1.0)]

        assert ensemble.is_fitted is False


class TestBaseCandidateGenerator:
    def test_generator_not_fitted_raises(self, mock_config):
        class TestGenerator(BaseCandidateGenerator):
            def _fit_impl(self, data_path, force):
                pass

            def _generate_batch_impl(self, users_df, day, n_candidates):
                return pl.DataFrame({"uid": [], "item_id": []})

        generator = TestGenerator()
        users_df = pl.DataFrame({"uid": [1, 2, 3]})

        with pytest.raises(RuntimeError, match="Generator not fitted"):
            generator.generate_batch(users_df, day=0, n_candidates=10)

    def test_deduplication_removes_duplicates(self, mock_config):
        class TestGenerator(BaseCandidateGenerator):
            def _fit_impl(self, data_path, force):
                pass

            def _generate_batch_impl(self, users_df, day, n_candidates):
                return pl.DataFrame(
                    {
                        "uid": [1, 1, 1, 2, 2],
                        "item_id": [10, 10, 20, 30, 30],
                    }
                )

        generator = TestGenerator()
        generator._is_fitted = True

        users_df = pl.DataFrame({"uid": [1, 2]})
        result = generator.generate_batch(
            users_df, day=0, n_candidates=100, deduplicate=True
        )

        unique_pairs = result.select(["uid", "item_id"]).unique()
        assert unique_pairs.height == result.height

    def test_head_n_candidates_limits_output(self, mock_config):
        class TestGenerator(BaseCandidateGenerator):
            def _fit_impl(self, data_path, force):
                pass

            def _generate_batch_impl(self, users_df, day, n_candidates):
                return pl.DataFrame(
                    {
                        "uid": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
                        "item_id": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                    }
                )

        generator = TestGenerator()
        generator._is_fitted = True

        users_df = pl.DataFrame({"uid": [1, 2]})
        result = generator.generate_batch(users_df, day=0, n_candidates=3)

        for uid in [1, 2]:
            user_candidates = result.filter(pl.col("uid") == uid)
            assert user_candidates.height <= 3, f"User {uid} has more than 3 candidates"

    def test_generator_name_property(self, mock_config):
        class MyCustomGenerator(BaseCandidateGenerator):
            def _fit_impl(self, data_path, force):
                pass

            def _generate_batch_impl(self, users_df, day, n_candidates):
                return pl.DataFrame({"uid": [], "item_id": []})

        generator = MyCustomGenerator()
        assert generator.generator_name == "mycustomgenerator"

    def test_cache_dir_created(self, mock_config):
        class TestGenerator(BaseCandidateGenerator):
            def _fit_impl(self, data_path, force):
                pass

            def _generate_batch_impl(self, users_df, day, n_candidates):
                return pl.DataFrame({"uid": [], "item_id": []})

        generator = TestGenerator()
        assert generator.cache_dir.exists()
