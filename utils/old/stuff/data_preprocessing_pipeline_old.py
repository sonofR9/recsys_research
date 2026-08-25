from dataclasses import dataclass
from typing import Callable
import os

import polars as pl

from .feature_generator import FeatureGenerator
from .negatives_generator import add_popular_random_negatives


@dataclass
class RandomNegativesConfig:
    min_k: int
    max_k: int
    multiplier: float

    top_n: int
    num_seeds: int
    window_days: int

    margin_multiplier: float


class DataPreprocessingPipeline:
    def __init__(
        self,
        *,
        random_negatives_configs: list[RandomNegativesConfig],
        feature_windows: list[str],
        feature_importances: dict[str, float] | None,
        feature_importance_threshold: float,
        target_for_fake: float,
        use_cache: bool,
        invalidate_cache: bool,
        preprocess_dir: str,
        preprocess_prefix: str,
        seed: int = 42,
    ):
        self.random_negatives_configs = random_negatives_configs
        self.feature_windows = feature_windows
        self.feature_importances = feature_importances
        self.feature_importance_threshold = feature_importance_threshold
        self.target_for_fake = target_for_fake

        self.preprocess_dir = preprocess_dir
        self.preprocess_prefix = preprocess_prefix
        self.use_cache = use_cache
        self.invalidate_cache = invalidate_cache
        self.seed = seed

    def generate_train_features(self, train: pl.DataFrame) -> pl.DataFrame:
        train = self._cache_or_compute(
            train,
            cache_names=["train_with_negatives"],
            compute_fn=self._generate_negatives,
        )

        train = self._cache_or_compute(
            history=train.lazy(),
            # to_enrich=train.lazy(),
            to_enrich=None,
            cache_names=["train_with_features"],
            compute_fn=self._generate_features,
        )

        return train

    def generate_test_features(
        self, history: pl.DataFrame, test: pl.DataFrame
    ) -> pl.DataFrame:
        return self._generate_features(
            history=history.lazy(), to_enrich=test.lazy()
        )

    def _generate_features(
        self,
        history: pl.LazyFrame,
        to_enrich: pl.LazyFrame | None,
    ) -> pl.DataFrame:
        generator = FeatureGenerator(
            windows=self.feature_windows,
            target_for_fake=self.target_for_fake,
            feature_importances=self.feature_importances,
            feature_importance_threshold=self.feature_importance_threshold,
        )

        eval_enriched = generator.create_features(
            target_df=to_enrich,
            event_history_df=history,
        )

        return eval_enriched

    def _generate_negatives(self, train: pl.DataFrame) -> pl.DataFrame:
        for config in self.random_negatives_configs:
            train = add_popular_random_negatives(
                train,
                min_k=config.min_k,
                max_k=config.max_k,
                multiplier=config.multiplier,
                top_n=config.top_n,
                seed=self.seed,
                num_seeds=config.num_seeds,
                window_days=config.window_days,
                margin_multiplier=config.margin_multiplier,
            )
        return train

    def _cache_or_compute(
        self, *args, cache_names: list[str], compute_fn: Callable, **kwargs
    ) -> tuple[pl.DataFrame, ...] | pl.DataFrame:
        cache_prefix = f"{self.preprocess_dir}/{self.preprocess_prefix}"
        cache_paths = [
            f"{cache_prefix}_{name}.parquet" for name in cache_names
        ]

        all_cached = all(os.path.exists(path) for path in cache_paths)

        if self.use_cache and not self.invalidate_cache and all_cached:
            results = [pl.read_parquet(path) for path in cache_paths]
        else:
            result = compute_fn(*args, **kwargs)
            results = result if isinstance(result, tuple) else (result,)

            if self.use_cache:
                for data, path in zip(results, cache_paths):
                    data.write_parquet(path)

        return results[0] if len(results) == 1 else tuple(results)
