import os
import tempfile
from pathlib import Path
from typing import Optional, Dict, Callable
import multiprocessing as mp
import traceback
import gc

import polars as pl

from .utils import log_memory


SECONDS_IN_DAY = 60 * 60 * 24


def _truncate_history_worker(
    result_queue,
    history_path: str,
    output_path: str,
    max_timestamp: int,
):
    """
    Worker process that truncates history using DuckDB.
    """
    try:
        import duckdb

        log_memory(
            f"[Truncate] Starting history truncation to timestamp < {max_timestamp}"
        )

        conn = duckdb.connect()

        if history_path.endswith(".parquet") or os.path.isfile(history_path):
            source_pattern = f"'{history_path}'"
        else:
            source_pattern = f"'{history_path}/*.parquet'"

        query = f"""
            COPY (
                SELECT * FROM read_parquet({source_pattern})
                WHERE timestamp < {max_timestamp}
            ) TO '{output_path}' (FORMAT PARQUET, COMPRESSION 'zstd')
        """

        conn.execute(query)

        count_query = f"""
            SELECT COUNT(*) FROM read_parquet('{output_path}')
        """
        row_count = conn.execute(count_query).fetchone()[0]

        conn.close()

        log_memory(f"[Truncate] Truncated history saved: {row_count} rows")

        result_queue.put(
            {
                "status": "success",
                "row_count": row_count,
                "output_path": output_path,
            }
        )

    except Exception as e:
        error_msg = f"[Truncate] Error: {str(e)}\n{traceback.format_exc()}"
        log_memory(error_msg)
        result_queue.put(
            {
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
        )


def _worker_process(
    result_queue,
    candidate_generator_factory: Callable,
    catboost_model_path: str,
    pipeline_params: Dict,
    feature_columns: list,
    batch_users_path: str,
    history_path: str,
    n_candidates_generate: int,
    n_candidates_final: int,
    generator_kwargs: Optional[Dict],
    next_day_timestamp: int,
    next_day: int,
    output_path: str,
    batch_idx: int,
):
    """
    Worker process that generates candidates, creates features, reranks, and saves results.
    """
    import sys

    def log_flush(msg):
        log_memory(msg)
        sys.stdout.flush()
        sys.stderr.flush()

    try:
        log_flush(f"[Batch {batch_idx}] Worker started")

        # Import inside subprocess (spawn creates fresh process)
        import polars as pl
        from catboost import CatBoostClassifier, CatBoostRanker
        from .data_preprocessing_pipeline import DataPreprocessingPipeline

        log_flush(f"[Batch {batch_idx}] Imports complete")

        # Load batch users
        log_flush(f"[Batch {batch_idx}] Reading batch users...")
        batch_users = pl.read_parquet(batch_users_path)
        log_flush(f"[Batch {batch_idx}] Loaded {len(batch_users)} users")

        # Create candidate generator
        log_flush(f"[Batch {batch_idx}] Creating candidate generator...")
        candidate_generator = candidate_generator_factory()
        log_flush(f"[Batch {batch_idx}] Candidate generator created")

        # Generate candidates
        log_flush(f"[Batch {batch_idx}] Generating candidates...")
        candidates_df = candidate_generator.generate_batch(
            users_df=batch_users,
            n_candidates=n_candidates_generate,
            generator_kwargs=generator_kwargs,
            deduplicate=True,
        )
        log_flush(
            f"[Batch {batch_idx}] Generated {len(candidates_df)} candidates"
        )

        del candidate_generator
        gc.collect()

        # Create test DataFrame
        test_df = candidates_df.select(["uid", "item_id"]).with_columns(
            [
                pl.lit(next_day_timestamp).alias("timestamp"),
                pl.lit(next_day).alias("day"),
            ]
        )
        del candidates_df
        gc.collect()
        log_flush(
            f"[Batch {batch_idx}] Created test DataFrame with {len(test_df)} rows"
        )

        # Create pipeline and generate features
        log_flush(f"[Batch {batch_idx}] Creating pipeline...")
        pipeline = DataPreprocessingPipeline(**pipeline_params)

        log_flush(f"[Batch {batch_idx}] Generating features...")
        test_with_features = pipeline.generate_test_features(
            history=history_path,
            test=test_df,
        )
        del test_df
        del pipeline
        gc.collect()
        log_flush(
            f"[Batch {batch_idx}] Features generated, shape: {test_with_features.shape}"
        )

        # Load CatBoost model
        log_flush(f"[Batch {batch_idx}] Loading model...")
        try:
            model = CatBoostClassifier()
            model.load_model(catboost_model_path)
        except Exception:
            model = CatBoostRanker()
            model.load_model(catboost_model_path)
        log_flush(f"[Batch {batch_idx}] Model loaded")

        # Prepare features
        available_features = [
            col for col in feature_columns if col in test_with_features.columns
        ]
        if len(available_features) != len(feature_columns):
            missing = set(feature_columns) - set(available_features)
            log_flush(
                f"[Batch {batch_idx}] Warning: Missing features: {missing}"
            )

        X = test_with_features.select(available_features).to_pandas()

        # Predictions
        log_flush(f"[Batch {batch_idx}] Running predictions...")
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(X)[:, 1]
        else:
            scores = model.predict(X)

        del X
        gc.collect()
        log_flush(f"[Batch {batch_idx}] Predictions complete")

        # Build result with ranking
        result_df = (
            test_with_features.select(["uid", "item_id"])
            .with_columns(pl.Series("score", scores))
            .sort(["uid", "score"], descending=[False, True])
            .with_columns(
                [
                    pl.col("score")
                    .rank(method="ordinal", descending=True)
                    .over("uid")
                    .alias("rank")
                ]
            )
            .filter(pl.col("rank") <= n_candidates_final)
            .sort(["uid", "rank"])
        )

        del test_with_features
        del scores
        gc.collect()
        log_flush(
            f"[Batch {batch_idx}] Ranking complete, {len(result_df)} rows"
        )

        # Save results
        result_df.write_parquet(output_path)
        log_flush(f"[Batch {batch_idx}] Results saved to {output_path}")

        result_queue.put(
            {
                "status": "success",
                "batch_idx": batch_idx,
                "n_users": len(batch_users),
                "n_results": len(result_df),
            }
        )

    except Exception as e:
        error_msg = (
            f"[Batch {batch_idx}] Error: {str(e)}\n{traceback.format_exc()}"
        )
        log_flush(error_msg)
        result_queue.put(
            {
                "status": "error",
                "batch_idx": batch_idx,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
        )


def generate_and_rerank_candidates(
    candidate_generator_factory: Callable,
    test_users: pl.DataFrame,
    history_path: str,
    catboost_model_path: str,
    pipeline_params: Dict,
    feature_columns: list,
    output_path: str,
    n_candidates_generate: int = 500,
    n_candidates_final: int = 100,
    batch_size: int = 100,
    generator_kwargs: Optional[Dict[str, Dict]] = None,
    temp_dir: Optional[str] = None,
    cleanup_temp: bool = True,
    end_day: Optional[int] = None,
) -> pl.DataFrame:
    """
    Generate candidates, create features, rerank with CatBoost, and return top candidates.
    """
    # Use spawn context to avoid fork issues with Polars
    ctx = mp.get_context("spawn")

    log_memory("Starting generate_and_rerank_candidates")

    # Setup temp directory
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp(prefix="candidate_gen_")
    else:
        os.makedirs(temp_dir, exist_ok=True)

    temp_dir = Path(temp_dir)
    log_memory(f"Using temp directory: {temp_dir}")

    truncated_history_path = None

    # Truncate history if end_day is specified
    if end_day is not None:
        log_memory(f"Truncating history to first {end_day} days...")

        max_timestamp_for_truncation = end_day * SECONDS_IN_DAY
        truncated_history_path = str(temp_dir / "truncated_history.parquet")

        result_queue = ctx.Queue()

        process = ctx.Process(
            target=_truncate_history_worker,
            kwargs={
                "result_queue": result_queue,
                "history_path": history_path,
                "output_path": truncated_history_path,
                "max_timestamp": max_timestamp_for_truncation,
            },
        )

        process.start()
        log_memory(f"[Truncate] Process started (PID: {process.pid})")

        process.join()

        if not result_queue.empty():
            result = result_queue.get()
            if result["status"] == "success":
                log_memory(
                    f"[Truncate] Completed successfully: {result['row_count']} rows"
                )
                history_path = truncated_history_path
            else:
                raise RuntimeError(
                    f"History truncation failed: {result['error']}\n{result['traceback']}"
                )
        else:
            raise RuntimeError(
                "History truncation process ended without result"
            )

        gc.collect()

    # Calculate next day timestamp
    log_memory("Calculating next day timestamp from history...")

    if history_path.endswith(".parquet") or os.path.isfile(history_path):
        history_timestamps = (
            pl.scan_parquet(history_path).select("timestamp").collect()
        )
    else:
        history_timestamps = (
            pl.scan_parquet(f"{history_path}/*.parquet")
            .select("timestamp")
            .collect()
        )

    max_timestamp = history_timestamps.select(pl.col("timestamp").max()).item()
    del history_timestamps
    gc.collect()

    current_day = int(max_timestamp // SECONDS_IN_DAY)
    next_day = current_day + 1
    next_day_timestamp = (next_day * SECONDS_IN_DAY) + (SECONDS_IN_DAY // 2)

    log_memory(f"History max timestamp: {max_timestamp}, next_day: {next_day}")

    # Split users into batches
    n_users = len(test_users)
    n_batches = (n_users + batch_size - 1) // batch_size

    log_memory(
        f"Processing {n_users} users in {n_batches} batches of {batch_size}"
    )

    batch_output_paths = []
    failed_batches = []

    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_users)

        log_memory(
            f"=== Starting batch {batch_idx + 1}/{n_batches} (users {start_idx}-{end_idx}) ==="
        )

        # Save batch users to temp file
        batch_users = test_users.slice(start_idx, end_idx - start_idx)
        batch_users_path = str(temp_dir / f"batch_users_{batch_idx}.parquet")
        batch_users.write_parquet(batch_users_path)

        batch_output_path = str(
            temp_dir / f"batch_results_{batch_idx}.parquet"
        )
        batch_output_paths.append(batch_output_path)

        # Use spawn context queues and process
        result_queue = ctx.Queue()

        process = ctx.Process(
            target=_worker_process,
            kwargs={
                "result_queue": result_queue,
                "candidate_generator_factory": candidate_generator_factory,
                "catboost_model_path": catboost_model_path,
                "pipeline_params": pipeline_params,
                "feature_columns": feature_columns,
                "batch_users_path": batch_users_path,
                "history_path": history_path,
                "n_candidates_generate": n_candidates_generate,
                "n_candidates_final": n_candidates_final,
                "generator_kwargs": generator_kwargs,
                "next_day_timestamp": next_day_timestamp,
                "next_day": next_day,
                "output_path": batch_output_path,
                "batch_idx": batch_idx,
            },
        )

        process.start()
        log_memory(f"[Batch {batch_idx}] Process started (PID: {process.pid})")

        process.join()

        if not result_queue.empty():
            result = result_queue.get()
            if result["status"] == "success":
                log_memory(
                    f"[Batch {batch_idx}] Completed successfully: "
                    f"{result['n_users']} users -> {result['n_results']} results"
                )
            else:
                log_memory(f"[Batch {batch_idx}] Failed: {result['error']}")
                failed_batches.append(batch_idx)
        else:
            log_memory(
                f"[Batch {batch_idx}] Process ended without result (exit code: {process.exitcode})"
            )
            failed_batches.append(batch_idx)

        if cleanup_temp and os.path.exists(batch_users_path):
            os.remove(batch_users_path)

        gc.collect()
        log_memory(f"[Batch {batch_idx}] Main process cleanup done")

    # Combine results
    log_memory("Combining batch results...")

    existing_batch_paths = [p for p in batch_output_paths if os.path.exists(p)]

    if not existing_batch_paths:
        raise RuntimeError("No batch results were generated successfully")

    if len(existing_batch_paths) < len(batch_output_paths):
        log_memory(
            f"Warning: {len(batch_output_paths) - len(existing_batch_paths)} batches failed"
        )

    final_result = pl.concat(
        [pl.read_parquet(path) for path in existing_batch_paths]
    )

    log_memory(
        f"Combined results: {len(final_result)} rows, {final_result['uid'].n_unique()} users"
    )

    final_result.write_parquet(output_path)
    log_memory(f"Final results saved to {output_path}")

    if cleanup_temp:
        for path in existing_batch_paths:
            if os.path.exists(path):
                os.remove(path)

        if truncated_history_path and os.path.exists(truncated_history_path):
            os.remove(truncated_history_path)
            log_memory("Truncated history file cleaned up")

        try:
            os.rmdir(temp_dir)
        except OSError:
            pass

        log_memory("Temp files cleaned up")

    if failed_batches:
        log_memory(f"Warning: Failed batches: {failed_batches}")

    return final_result


def get_feature_columns_from_model(model_path: str) -> list:
    """Extract feature column names from a saved CatBoost model."""
    from catboost import CatBoostClassifier, CatBoostRanker

    try:
        model = CatBoostClassifier()
        model.load_model(model_path)
    except Exception:
        model = CatBoostRanker()
        model.load_model(model_path)

    return model.feature_names_
