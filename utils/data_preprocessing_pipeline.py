from dataclasses import dataclass
from pathlib import Path
import gc

import polars as pl

from .utils import log_memory
from .negatives_generator import add_popular_random_negatives
from data.counters import EmaCounter
from data.preprocessing import preprocess_counters
from data.split_by_day import split_main_parquet_by_day
from data.utils import merge_parquets_duckdb


@dataclass
class RandomNegativesConfig:
    min_k: int
    max_k: int
    multiplier: float

    top_n: int | None
    num_seeds: int
    window_days: int

    margin_multiplier: float
    from_whole_history: bool
    weighted: bool


DataFrameOrPath = pl.DataFrame | str | Path


def _load_if_path(data: DataFrameOrPath) -> pl.DataFrame:
    if isinstance(data, (str, Path)):
        return pl.read_parquet(data)
    return data


def _ensure_on_disk(data: DataFrameOrPath, output_path: Path) -> Path:
    if isinstance(data, (str, Path)):
        return Path(data)
    data.write_parquet(output_path)
    return output_path


class DataPreprocessingPipeline:
    def __init__(
        self,
        *,
        counters: list[EmaCounter],
        random_negatives_configs: list[RandomNegativesConfig] | None = None,
        feature_importances: dict[str, float] | None = None,
        feature_importance_threshold: float = 0.0,
        use_cache: bool = True,
        invalidate_cache: bool = False,
        preprocess_dir: str,
        preprocess_prefix: str,
        seed: int = 42,
    ):
        self.counters = counters
        self.random_negatives_configs = random_negatives_configs or []
        self.feature_importances = feature_importances
        self.feature_importance_threshold = feature_importance_threshold

        self.preprocess_dir = Path(preprocess_dir)
        self.preprocess_prefix = preprocess_prefix
        self.use_cache = use_cache
        self.invalidate_cache = invalidate_cache
        self.seed = seed

        self.preprocess_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, name: str) -> Path:
        return self.preprocess_dir / f"{self.preprocess_prefix}_{name}.parquet"

    def get_counter_columns(self) -> list[str]:
        columns = []
        for counter in self.counters:
            columns.extend(counter.get_output_columns())

        if self.feature_importances is None:
            return columns
        return [
            col
            for col in columns
            if col in self.feature_importances
            and self.feature_importances[col] > self.feature_importance_threshold
        ]

    def generate_train_features(
        self, train: DataFrameOrPath, return_path: bool = False
    ) -> pl.DataFrame | Path:
        log_memory("Starting train feature generation")

        train_with_negatives_path = self._generate_negatives(train)
        del train
        gc.collect()
        log_memory("After generating negatives")

        train_with_features_path = self._apply_counters(
            train_with_negatives_path,
            cache_name="train_with_features",
        )
        log_memory("After applying counters")

        if return_path:
            return train_with_features_path

        result = pl.read_parquet(train_with_features_path)
        log_memory("After loading result")
        return result

    def generate_test_features(
        self,
        history: DataFrameOrPath,
        test: DataFrameOrPath,
    ) -> pl.DataFrame:
        log_memory("generate_test_features: start")

        history_path = _ensure_on_disk(history, self._get_cache_path("temp_history"))
        del history
        gc.collect()

        test_path = _ensure_on_disk(test, self._get_cache_path("temp_test"))
        del test
        gc.collect()

        log_memory("generate_test_features: data saved to disk")

        features_path = self._apply_counters_for_test(
            history_path=history_path,
            test_path=test_path,
        )

        log_memory("generate_test_features: features generated")

        result = pl.read_parquet(features_path)

        for temp_name in ["temp_history", "temp_test", "temp_test_features"]:
            temp_path = self._get_cache_path(temp_name)
            if temp_path.exists():
                temp_path.unlink()

        log_memory("generate_test_features: done")
        return result

    def _generate_negatives(self, train: DataFrameOrPath) -> Path:
        final_cache_path = self._get_cache_path("train_with_negatives")

        if self.use_cache and not self.invalidate_cache and final_cache_path.exists():
            log_memory("Using cached train_with_negatives")
            return final_cache_path

        current_path = _ensure_on_disk(train, self._get_cache_path("temp_train_input"))
        log_memory(f"Input saved to {current_path}")

        if not self.random_negatives_configs:
            log_memory("No negatives configs, using input as-is")
            if current_path != final_cache_path:
                import shutil

                shutil.copy2(current_path, final_cache_path)
                temp_input_path = self._get_cache_path("temp_train_input")
                if temp_input_path.exists():
                    temp_input_path.unlink()
            return final_cache_path

        for i, config in enumerate(self.random_negatives_configs):
            log_memory(
                f"Processing negatives config {i + 1}/{len(self.random_negatives_configs)}"
            )

            if i == len(self.random_negatives_configs) - 1:
                output_path = final_cache_path
            else:
                output_path = self._get_cache_path(f"temp_negatives_step_{i}")

            add_popular_random_negatives(
                input_path=str(current_path),
                output_path=str(output_path),
                min_k=config.min_k,
                max_k=config.max_k,
                multiplier=config.multiplier,
                top_n=config.top_n,
                seed=self.seed,
                num_seeds=config.num_seeds,
                window_days=config.window_days,
                margin_multiplier=int(config.margin_multiplier),
                return_dataframe=False,
                from_whole_history=config.from_whole_history,
                weighted=config.weighted,
            )

            if i > 0:
                prev_temp_path = self._get_cache_path(f"temp_negatives_step_{i - 1}")
                if prev_temp_path.exists():
                    prev_temp_path.unlink()

            current_path = output_path
            gc.collect()

        temp_input_path = self._get_cache_path("temp_train_input")
        if temp_input_path.exists() and temp_input_path != current_path:
            temp_input_path.unlink()

        return final_cache_path

    def _apply_counters(
        self,
        data_path: Path,
        cache_name: str,
    ) -> Path:
        output_path = self._get_cache_path(cache_name)

        if self.use_cache and not self.invalidate_cache and output_path.exists():
            log_memory(f"Using cached {cache_name}")
            return output_path

        if not self.counters:
            log_memory("No counters configured, returning input as-is")
            import shutil

            shutil.copy2(data_path, output_path)
            return output_path

        days_dir = self._get_cache_path(f"{cache_name}_days")
        day_to_path = split_main_parquet_by_day(
            data_path, days_dir, self.invalidate_cache
        )
        days = sorted(day_to_path.keys())
        log_memory(f"Split data into {len(days)} day files")

        enriched_dir = self._get_cache_path(f"{cache_name}_enriched")
        enriched_paths = preprocess_counters(
            counters=self.counters,
            day_to_path=day_to_path,
            enriched_dir=enriched_dir,
            days=days,
            invalidate_cache=self.invalidate_cache,
        )
        log_memory(f"Processed counters for {len(enriched_paths)} days")

        parquet_files = [enriched_paths[day] for day in days]
        merge_parquets_duckdb(parquet_files, output_path)
        log_memory(f"Merged enriched data to {output_path}")

        return output_path

    def _apply_counters_for_test(
        self,
        history_path: Path,
        test_path: Path,
    ) -> Path:
        output_path = self._get_cache_path("temp_test_features")

        if not self.counters:
            log_memory("No counters configured, returning test as-is")
            import shutil

            shutil.copy2(test_path, output_path)
            return output_path

        history_days_dir = self._get_cache_path("test_history_days")
        history_day_to_path = split_main_parquet_by_day(
            history_path, history_days_dir, self.invalidate_cache
        )
        history_days = sorted(history_day_to_path.keys())
        log_memory(f"Split history into {len(history_days)} day files")

        history_enriched_dir = self._get_cache_path("test_history_enriched")
        preprocess_counters(
            counters=self.counters,
            day_to_path=history_day_to_path,
            enriched_dir=history_enriched_dir,
            days=history_days,
            invalidate_cache=self.invalidate_cache,
        )
        log_memory("Counter state built from history")

        last_history_day_idx = max(history_days)

        test_days_dir = self._get_cache_path("test_days")
        test_day_to_path = split_main_parquet_by_day(
            test_path, test_days_dir, self.invalidate_cache
        )

        test_day_to_path_shifted = {
            last_history_day_idx + 1 + day_idx: path
            for day_idx, path in enumerate(sorted(test_day_to_path.values()))
        }
        test_days_shifted = sorted(test_day_to_path_shifted.keys())
        log_memory(f"Split test into {len(test_days_shifted)} day files (shifted)")

        test_enriched_dir = self._get_cache_path("test_enriched")
        enriched_paths = preprocess_counters(
            counters=self.counters,
            day_to_path=test_day_to_path_shifted,
            enriched_dir=test_enriched_dir,
            days=test_days_shifted,
            invalidate_cache=False,
        )
        log_memory(f"Processed counters for {len(enriched_paths)} test days")

        parquet_files = [enriched_paths[day] for day in test_days_shifted]
        merge_parquets_duckdb(parquet_files, output_path)
        log_memory(f"Merged test features to {output_path}")

        return output_path
