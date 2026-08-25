import os
import tempfile
from pathlib import Path
from typing import Optional, Dict
import gc
import logging

import polars as pl

from .utils import log_memory


SECONDS_IN_DAY = 60 * 60 * 24


def _truncate_history(
    history_path: str,
    output_path: str,
    max_timestamp: int,
) -> int:
    """
    Truncates history using DuckDB.

    Returns:
        Number of rows in truncated history.
    """
    import duckdb

    log_memory(f"[Truncate] Starting history truncation to timestamp < {max_timestamp}")

    conn = duckdb.connect()
    conn.execute("SET enable_progress_bar=false;")

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

    return row_count


def _enrich_with_item_metadata(
    df: pl.DataFrame,
    item_metadata: pl.DataFrame,
) -> pl.DataFrame:
    """
    Join item metadata (artist_id, album_id) to DataFrame.
    """
    cols_to_join = ["item_id"]
    cols_to_join.extend(["artist_id", "album_id"])

    metadata_subset = item_metadata.select(cols_to_join).unique(subset=["item_id"])

    result = df.join(
        metadata_subset,
        on="item_id",
        how="left",
    )

    log_memory("Enriched DataFrame with item metadata")

    return result


def _inject_debug_additional_candidates(
    candidates_df: pl.DataFrame,
    debug_additional_candidates: pl.DataFrame,
    batch_users: pl.DataFrame,
    batch_idx: int,
) -> pl.DataFrame:
    """
    Inject true positive interactions into candidates for debugging purposes.
    """
    batch_user_ids = batch_users.select("uid")

    candidates_schema = dict(candidates_df.schema)
    uid_dtype = candidates_schema.get("uid", pl.UInt32)
    item_id_dtype = candidates_schema.get("item_id", pl.UInt32)

    batch_positives = (
        debug_additional_candidates.with_columns(
            [
                pl.col("uid").cast(uid_dtype),
                pl.col("item_id").cast(item_id_dtype),
            ]
        )
        .join(
            batch_user_ids.with_columns(pl.col("uid").cast(uid_dtype)),
            on="uid",
            how="inner",
        )
        .filter(pl.col("target") > 0)
        .select(["uid", "item_id"])
        .unique()
    )

    n_positives = len(batch_positives)
    n_users_with_positives = batch_positives["uid"].n_unique()

    log_memory(
        f"[Batch {batch_idx}][DEBUG] Injecting {n_positives} positive interactions "
        f"for {n_users_with_positives} users"
    )

    if n_positives == 0:
        log_memory(f"[Batch {batch_idx}][DEBUG] No positive interactions to inject")
        return candidates_df

    combined = pl.concat([candidates_df.select(["uid", "item_id"]), batch_positives])

    deduplicated = combined.unique(subset=["uid", "item_id"])

    n_original = len(candidates_df)
    n_after = len(deduplicated)
    n_new = n_after - n_original

    log_memory(
        f"[Batch {batch_idx}][DEBUG] Candidates: {n_original} -> {n_after} "
        f"(+{n_new} new from positives, {n_positives - n_new} were already present)"
    )

    return deduplicated


def _process_batch(
    candidate_generator,
    catboost_model_path: str,
    pipeline_params: Dict,
    feature_columns: list,
    batch_users: pl.DataFrame,
    history_path: str,
    item_metadata: pl.DataFrame,
    n_candidates_generate: int,
    n_candidates_final: int,
    generator_kwargs: Optional[Dict],
    next_day_timestamp: int,
    batch_idx: int,
    debug_additional_candidates: Optional[pl.DataFrame] = None,
) -> pl.DataFrame:
    """
    Process a single batch: generate candidates, create features, rerank, and return results.
    """
    from catboost import CatBoostClassifier, CatBoostRanker
    from .data_preprocessing_pipeline import DataPreprocessingPipeline

    log_memory(f"[Batch {batch_idx}] Processing {len(batch_users)} users")
    log_memory(f"[Batch {batch_idx}] Generating candidates...")
    candidates_df = candidate_generator.generate_batch_ensemble(
        users_df=batch_users,
        n_candidates=n_candidates_generate,
        generator_kwargs=generator_kwargs,
        deduplicate=True,
    )
    log_memory(f"[Batch {batch_idx}] Generated {len(candidates_df)} candidates")

    del candidate_generator
    gc.collect()

    if debug_additional_candidates is not None:
        candidates_df = _inject_debug_additional_candidates(
            candidates_df=candidates_df,
            debug_additional_candidates=debug_additional_candidates,
            batch_users=batch_users,
            batch_idx=batch_idx,
        )

    test_df = candidates_df.select(["uid", "item_id"]).with_columns(
        [
            pl.lit(next_day_timestamp).cast(pl.UInt32).alias("timestamp"),
        ]
    )
    del candidates_df
    gc.collect()

    log_memory(f"[Batch {batch_idx}] Enriching with item metadata...")
    test_df = _enrich_with_item_metadata(test_df, item_metadata)

    log_memory(f"[Batch {batch_idx}] Creating pipeline...")
    pipeline = DataPreprocessingPipeline(**pipeline_params)

    log_memory(f"[Batch {batch_idx}] Generating features...")

    test_with_features = pipeline.generate_test_features(
        history=history_path,
        test=test_df,
    )
    del test_df
    del pipeline
    gc.collect()
    log_memory(
        f"[Batch {batch_idx}] Features generated, shape: {test_with_features.shape}"
    )

    log_memory(f"[Batch {batch_idx}] Loading model...")
    try:
        model = CatBoostClassifier()
        model.load_model(catboost_model_path)
    except Exception:
        model = CatBoostRanker()
        model.load_model(catboost_model_path)
    log_memory(f"[Batch {batch_idx}] Model loaded")

    available_features = [
        col for col in feature_columns if col in test_with_features.columns
    ]
    if len(available_features) != len(feature_columns):
        missing = set(feature_columns) - set(available_features)
        log_memory(
            f"[Batch {batch_idx}] Warning: Missing features: {missing}", logging.WARNING
        )

    X = test_with_features.select(available_features)

    log_memory(f"[Batch {batch_idx}] Running predictions...")
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X)[:, 1]
    else:
        scores = model.predict(X)

    del X
    del model
    gc.collect()
    log_memory(f"[Batch {batch_idx}] Predictions complete")

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
    log_memory(f"[Batch {batch_idx}] Ranking complete, {len(result_df)} rows")

    return result_df


def generate_and_rerank_candidates(
    candidate_generator,
    test_users: pl.DataFrame,
    history_path: str,
    item_metadata: pl.DataFrame,
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
    debug_additional_candidates: Optional[pl.DataFrame] = None,
) -> pl.DataFrame:
    """
    Generate candidates, create features, rerank with CatBoost, and return top candidates.
    """
    log_memory("Starting generate_and_rerank_candidates")

    if debug_additional_candidates is not None:
        n_debug_interactions = len(debug_additional_candidates)
        n_debug_users = debug_additional_candidates["uid"].n_unique()
        n_positive = debug_additional_candidates.filter(pl.col("target") > 0).shape[0]
        log_memory(
            f"[DEBUG MODE] Positive interactions injection enabled: "
            f"{n_debug_interactions} total interactions, {n_positive} positive, "
            f"{n_debug_users} users",
            logging.DEBUG,
        )

    if temp_dir is None:
        temp_dir = tempfile.mkdtemp(prefix="candidate_gen_")
    else:
        os.makedirs(temp_dir, exist_ok=True)

    temp_dir = Path(temp_dir)
    log_memory(f"Using temp directory: {temp_dir}")

    truncated_history_path = None

    if end_day is not None:
        log_memory(f"Truncating history to first {end_day} days...")

        max_timestamp_for_truncation = end_day * SECONDS_IN_DAY
        truncated_history_path = str(temp_dir / "truncated_history.parquet")

        row_count = _truncate_history(
            history_path=history_path,
            output_path=truncated_history_path,
            max_timestamp=max_timestamp_for_truncation,
        )

        log_memory(f"[Truncate] Completed successfully: {row_count} rows")
        history_path = truncated_history_path

        gc.collect()

    log_memory("Calculating next day timestamp from history...")

    if history_path.endswith(".parquet") or os.path.isfile(history_path):
        history_timestamps = pl.scan_parquet(history_path).select("timestamp").collect()
    else:
        history_timestamps = (
            pl.scan_parquet(f"{history_path}/*.parquet").select("timestamp").collect()
        )

    max_timestamp = history_timestamps.select(pl.col("timestamp").max()).item()
    del history_timestamps
    gc.collect()

    current_day = int(max_timestamp // SECONDS_IN_DAY)
    next_day = current_day + 1
    next_day_timestamp = (next_day * SECONDS_IN_DAY) + (SECONDS_IN_DAY // 2)

    log_memory(f"History max timestamp: {max_timestamp}, next_day: {next_day}")

    n_users = len(test_users)
    n_batches = (n_users + batch_size - 1) // batch_size

    log_memory(f"Processing {n_users} users in {n_batches} batches of {batch_size}")

    batch_results = []
    failed_batches = []

    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_users)

        log_memory(
            f"=== Starting batch {batch_idx + 1}/{n_batches} (users {start_idx}-{end_idx}) ==="
        )

        batch_users = test_users.slice(start_idx, end_idx - start_idx)

        try:
            result_df = _process_batch(
                candidate_generator=candidate_generator,
                catboost_model_path=catboost_model_path,
                pipeline_params=pipeline_params,
                feature_columns=feature_columns,
                batch_users=batch_users,
                history_path=history_path,
                item_metadata=item_metadata,
                n_candidates_generate=n_candidates_generate,
                n_candidates_final=n_candidates_final,
                generator_kwargs=generator_kwargs,
                next_day_timestamp=next_day_timestamp,
                batch_idx=batch_idx,
                debug_additional_candidates=debug_additional_candidates,
            )

            batch_results.append(result_df)
            log_memory(
                f"[Batch {batch_idx}] Completed successfully: "
                f"{len(batch_users)} users -> {len(result_df)} results"
            )

        except Exception as e:
            import traceback

            log_memory(
                f"[Batch {batch_idx}] Failed: {str(e)}\n{traceback.format_exc()}"
            )
            failed_batches.append(batch_idx)

        gc.collect()
        log_memory(f"[Batch {batch_idx}] Cleanup done")

    log_memory("Combining batch results...")

    if not batch_results:
        raise RuntimeError("No batch results were generated successfully")

    if len(batch_results) < n_batches:
        log_memory(
            f"Warning: {n_batches - len(batch_results)} batches failed",
            logging.WARNING,
        )

    final_result = pl.concat(batch_results)
    del batch_results
    gc.collect()

    log_memory(
        f"Combined results: {len(final_result)} rows, {final_result['uid'].n_unique()} users"
    )

    final_result.write_parquet(output_path)
    log_memory(f"Final results saved to {output_path}")

    if cleanup_temp:
        if truncated_history_path and os.path.exists(truncated_history_path):
            os.remove(truncated_history_path)
            log_memory("Truncated history file cleaned up")

        try:
            os.rmdir(temp_dir)
        except OSError:
            pass

        log_memory("Temp files cleaned up")

    if failed_batches:
        log_memory(
            f"Warning: Failed batches: {failed_batches}",
            logging.WARNING,
        )

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
