from dataclasses import dataclass
from typing import Callable
from pathlib import Path
import gc
import multiprocessing as mp

import polars as pl

from .utils import log_memory
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
    from_whole_history: bool
    weighted: bool


DataFrameOrPath = pl.DataFrame | str | Path


def _load_if_path(data: DataFrameOrPath) -> pl.DataFrame:
    """Load DataFrame from path if needed."""
    if isinstance(data, (str, Path)):
        return pl.read_parquet(data)
    return data


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

        self.preprocess_dir = Path(preprocess_dir)
        self.preprocess_prefix = preprocess_prefix
        self.use_cache = use_cache
        self.invalidate_cache = invalidate_cache
        self.seed = seed

        self.preprocess_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, name: str) -> Path:
        """Get the cache path for a given name."""
        return self.preprocess_dir / f"{self.preprocess_prefix}_{name}.parquet"

    def generate_train_features(
        self, train: DataFrameOrPath, return_path: bool = False
    ) -> pl.DataFrame | Path:
        """Generate training features with negatives."""

        log_memory("Before generating negatives")

        train_with_negatives_path = self._generate_negatives_subprocess(train)

        log_memory("After generating negatives (subprocess completed)")

        del train
        gc.collect()
        log_memory("After del train")

        train_with_features_path = self._generate_features_subprocess(
            history_path=train_with_negatives_path,
            to_enrich_path=None,
            cache_name="train_with_features",
        )

        log_memory("After generating features (subprocess completed)")

        if return_path:
            return train_with_features_path

        result = pl.read_parquet(train_with_features_path)
        log_memory("After loading result")

        return result

    def _generate_negatives_subprocess(self, train: DataFrameOrPath) -> Path:
        """Generate negatives in separate subprocesses, one config at a time to reduce RAM."""

        final_cache_path = self._get_cache_path("train_with_negatives")

        if self.use_cache and not self.invalidate_cache and final_cache_path.exists():
            log_memory("Using cached train_with_negatives")
            return final_cache_path

        current_path = self._ensure_on_disk(train, "temp_train_input")
        log_memory(f"Input saved to {current_path}")

        if not self.random_negatives_configs:
            log_memory("No negatives configs to process, using input as-is")
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

            config_dict = {
                "min_k": config.min_k,
                "max_k": config.max_k,
                "multiplier": config.multiplier,
                "top_n": config.top_n,
                "num_seeds": config.num_seeds,
                "window_days": config.window_days,
                "margin_multiplier": config.margin_multiplier,
                "from_whole_history": config.from_whole_history,
                "weighted": config.weighted,
            }

            ctx = mp.get_context("spawn")

            status_path = self._get_cache_path(f"_subprocess_status_neg_{i}")
            if status_path.exists():
                status_path.unlink()

            process = ctx.Process(
                target=self._single_negatives_worker_static,
                args=(
                    str(current_path),
                    str(output_path),
                    str(status_path),
                    config_dict,
                    self.seed,
                    i,
                    len(self.random_negatives_configs),
                ),
            )

            log_memory(f"Starting subprocess for negatives config {i + 1}")
            process.start()
            process.join()

            log_memory(
                f"Subprocess for config {i + 1} exited with code {process.exitcode}"
            )

            if process.exitcode != 0:
                error_msg = "Unknown error"
                if status_path.exists():
                    error_msg = status_path.read_text()
                raise RuntimeError(
                    f"Subprocess failed with exit code {process.exitcode}: {error_msg}"
                )

            if not output_path.exists():
                error_msg = ""
                if status_path.exists():
                    error_msg = status_path.read_text()
                raise RuntimeError(
                    f"Subprocess completed but output file not found: {error_msg}"
                )

            if status_path.exists():
                status_path.unlink()

            if i > 0:
                prev_temp_path = self._get_cache_path(f"temp_negatives_step_{i - 1}")
                if prev_temp_path.exists():
                    prev_temp_path.unlink()
                    log_memory(f"Cleaned up temp file from step {i}")

            current_path = output_path

            gc.collect()

        temp_input_path = self._get_cache_path("temp_train_input")
        if temp_input_path.exists() and temp_input_path != current_path:
            temp_input_path.unlink()

        return final_cache_path

    @staticmethod
    def _single_negatives_worker_static(
        input_path: str,
        output_path: str,
        status_path: str,
        config_dict: dict,
        seed: int,
        config_index: int,
        total_configs: int,
    ):
        """
        Static worker function to generate negatives for a single config in a subprocess.
        Now uses DuckDB-based implementation that writes directly to parquet.
        """
        import logging

        logging.basicConfig(level=logging.DEBUG)  # WARN

        log_memory("_single_negatives_worker_static")
        from pathlib import Path

        from .negatives_generator import add_popular_random_negatives

        status_file = Path(status_path)

        try:
            log_memory(f"[Subprocess {config_index + 1}/{total_configs}] Starting")
            log_memory(
                f"[Subprocess {config_index + 1}/{total_configs}] Input: {input_path}"
            )
            log_memory(
                f"[Subprocess {config_index + 1}/{total_configs}] Output: {output_path}"
            )

            # Call the updated function that handles everything internally
            # and writes directly to parquet
            # FIXME(sashanovak): make dict dict from config_dict and just add necessary values to it
            add_popular_random_negatives(
                input_path=input_path,
                output_path=output_path,
                min_k=config_dict["min_k"],
                max_k=config_dict["max_k"],
                multiplier=config_dict["multiplier"],
                top_n=config_dict["top_n"],
                seed=seed,
                num_seeds=config_dict["num_seeds"],
                window_days=config_dict["window_days"],
                margin_multiplier=config_dict["margin_multiplier"],
                return_dataframe=False,
                from_whole_history=config_dict["from_whole_history"],
                weighted=config_dict["weighted"],
            )

            log_memory(f"[Subprocess {config_index + 1}/{total_configs}] Done!")

            status_file.write_text("SUCCESS")

        except Exception as e:
            import traceback

            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            print(
                f"[Subprocess {config_index + 1}/{total_configs}] ERROR: {error_msg}",
                flush=True,
            )
            status_file.write_text(f"ERROR: {error_msg}")
            raise

    def _generate_features_subprocess(
        self,
        history_path: Path,
        to_enrich_path: Path | None,
        cache_name: str | None,
    ) -> Path:
        """Generate features in a subprocess to ensure memory release."""

        if cache_name is not None:
            output_path = self._get_cache_path(cache_name)

            if self.use_cache and not self.invalidate_cache and output_path.exists():
                log_memory(f"Using cached {cache_name}")
                return output_path
        else:
            output_path = self._get_cache_path("temp_features_output")

        ctx = mp.get_context("spawn")

        status_path = self._get_cache_path("_subprocess_status_features")
        if status_path.exists():
            status_path.unlink()

        process = ctx.Process(
            target=self._features_worker_static,
            args=(
                str(history_path),
                str(to_enrich_path) if to_enrich_path else None,
                str(output_path),
                str(status_path),
                self.feature_windows,
                self.target_for_fake,
                self.feature_importances,
                self.feature_importance_threshold,
            ),
        )

        log_memory("Starting subprocess for feature generation")
        process.start()
        process.join()

        log_memory(f"Subprocess exited with code {process.exitcode}")

        if process.exitcode != 0:
            error_msg = ""
            if status_path.exists():
                error_msg = status_path.read_text()
            raise RuntimeError(
                f"Subprocess failed with exit code {process.exitcode}: {error_msg}"
            )

        if not output_path.exists():
            error_msg = ""
            if status_path.exists():
                error_msg = status_path.read_text()
            raise RuntimeError(
                f"Subprocess completed but output file not found: {error_msg}"
            )

        if status_path.exists():
            status_path.unlink()

        return output_path

    @staticmethod
    def _features_worker_static(
        history_path: str,
        to_enrich_path: str | None,
        output_path: str,
        status_path: str,
        feature_windows: list[str],
        target_for_fake: float,
        feature_importances: dict[str, float] | None,
        feature_importance_threshold: float,
    ):
        """
        Static worker function to generate features in a subprocess.
        """
        import polars as pl
        from pathlib import Path

        try:
            from .feature_generator import FeatureGenerator
        except ImportError:
            from model.feature_generator import FeatureGenerator

        status_file = Path(status_path)

        try:
            print(f"[Subprocess] Starting feature generation", flush=True)
            print(f"[Subprocess] History path: {history_path}", flush=True)
            print(f"[Subprocess] To enrich path: {to_enrich_path}", flush=True)

            generator = FeatureGenerator(
                windows=feature_windows,
                target_for_fake=target_for_fake,
                feature_importances=feature_importances,
                feature_importance_threshold=feature_importance_threshold,
            )

            print("[Subprocess] Created FeatureGenerator", flush=True)

            result = generator.create_features(
                target_df=to_enrich_path,
                event_history_df=history_path,
            )

            print(
                f"[Subprocess] Generated features, shape {result.shape}",
                flush=True,
            )

            if isinstance(result, pl.LazyFrame):
                result = result.collect()

            result.write_parquet(output_path)
            print("[Subprocess] Saved result", flush=True)

            status_file.write_text("SUCCESS")

        except Exception as e:
            import traceback

            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            print(f"[Subprocess] ERROR: {error_msg}", flush=True)
            status_file.write_text(f"ERROR: {error_msg}")
            raise

    def generate_test_features(
        self,
        history: DataFrameOrPath,
        test: DataFrameOrPath,
    ) -> pl.DataFrame:
        """
        Generate test features using history.
        """
        log_memory("generate_test_features: start")

        history_path = self._ensure_on_disk(history, "temp_history")
        del history
        gc.collect()

        test_path = self._ensure_on_disk(test, "temp_test")
        del test
        gc.collect()

        log_memory("generate_test_features: data saved to disk")

        features_path = self._generate_features_subprocess(
            history_path=history_path,
            to_enrich_path=test_path,
            cache_name=None,
        )

        log_memory("generate_test_features: features generated")

        result = pl.read_parquet(features_path)

        # Clean up temp files
        for temp_name in ["temp_history", "temp_test", "temp_features_output"]:
            temp_path = self._get_cache_path(temp_name)
            if temp_path.exists():
                temp_path.unlink()

        log_memory("generate_test_features: done")

        return result

    def _ensure_on_disk(self, data: DataFrameOrPath, temp_name: str) -> Path:
        """Ensure data is on disk, saving if necessary."""
        if isinstance(data, (str, Path)):
            return Path(data)

        temp_path = self._get_cache_path(temp_name)
        data.write_parquet(temp_path)
        return temp_path

    def _cache_or_compute_to_path(
        self,
        data: DataFrameOrPath,
        cache_name: str,
        compute_fn: Callable[[pl.DataFrame], pl.DataFrame],
    ) -> Path:
        """
        Check cache or compute result, always returning a path.
        """
        cache_path = self._get_cache_path(cache_name)

        if self.use_cache and not self.invalidate_cache and cache_path.exists():
            return cache_path

        df = _load_if_path(data)
        result = compute_fn(df)
        del df
        result.write_parquet(cache_path)
        del result

        return cache_path

    def _cache_or_compute(
        self, *args, cache_names: list[str], compute_fn: Callable, **kwargs
    ) -> tuple[pl.DataFrame, ...] | pl.DataFrame:
        """Legacy method for backward compatibility."""
        cache_paths = [self._get_cache_path(name) for name in cache_names]

        all_cached = all(path.exists() for path in cache_paths)

        if self.use_cache and not self.invalidate_cache and all_cached:
            results = [pl.read_parquet(path) for path in cache_paths]
        else:
            result = compute_fn(*args, **kwargs)
            results = result if isinstance(result, tuple) else (result,)

            if self.use_cache:
                for data, path in zip(results, cache_paths):
                    data.write_parquet(path)

        return results[0] if len(results) == 1 else tuple(results)
