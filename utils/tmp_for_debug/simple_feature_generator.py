"""
Simple, leak-free feature generator for debugging purposes.

This module provides a minimal feature generator that is guaranteed to be
leak-free by design. It uses strict temporal filtering to ensure that
features for any row are computed ONLY from events that occurred BEFORE
that row's timestamp.

Key differences from the original feature_generator.py:
1. Explicit timestamp-based filtering (not day-based window functions)
2. Features are computed per-row with strict "< timestamp" filtering
3. No spine table that could mix future and past data
4. Clear separation between history and target data
"""

import polars as pl
import duckdb
import tempfile
from pathlib import Path
import os
import gc

from utils.global_config import config as global_config
from .utils import log_memory, setup_duckdb_connection


class SimpleFeatureGenerator:
    """
    A simple, leak-free feature generator.

    This generator computes features for each (uid, item_id) pair in the target
    using ONLY events from the history that occurred STRICTLY BEFORE the target
    row's timestamp.

    Features generated:
    - user_likes_before: Number of likes by user before this timestamp
    - user_dislikes_before: Number of dislikes by user before this timestamp
    - user_total_before: Total interactions by user before this timestamp
    - item_likes_before: Number of likes on item before this timestamp
    - item_dislikes_before: Number of dislikes on item before this timestamp
    - item_total_before: Total interactions on item before this timestamp
    - user_item_likes_before: Likes by this user on this item before
    - user_item_total_before: Total interactions by this user on this item before
    - artist_likes_before: Likes on this artist before this timestamp
    - artist_total_before: Total interactions on this artist before this timestamp
    - user_artist_likes_before: Likes by this user on this artist before
    - user_artist_total_before: Total interactions by this user on this artist before
    """

    def __init__(
        self,
        windows_days: list[int] | None = None,
        target_for_fake: float = 0.05,
    ):
        """
        Initialize the feature generator.

        Args:
            windows_days: List of window sizes in days for windowed features.
                         If None, uses [1, 7, 30].
            target_for_fake: Target value for synthetic negative samples.
        """
        self.windows_days = windows_days or [1, 7, 30]
        self.target_for_fake = target_for_fake
        self.SECONDS_IN_DAY = 86400

    def create_features(
        self,
        history_df: pl.DataFrame | str | Path,
        target_df: pl.DataFrame | str | Path | None = None,
    ) -> pl.DataFrame:
        """
        Create features for target rows using history data.

        IMPORTANT: This method ensures NO DATA LEAKAGE by:
        1. Using strict "timestamp < target_timestamp" filtering
        2. Computing features independently for each target row
        3. Never using any data from the future

        Args:
            history_df: Historical events (DataFrame or path to parquet).
                       Must have columns: uid, item_id, timestamp, event_type
                       Optional: artist_id, album_id, is_organic
            target_df: Target rows to generate features for.
                      If None, uses history_df (same-source mode).
                      Must have columns: uid, item_id, timestamp

        Returns:
            DataFrame with target rows enriched with features.
        """
        log_memory("SimpleFeatureGenerator: Starting create_features")
        if target_df is None:
            target_df = history_df
            same_source = True
        else:
            same_source = False

        if isinstance(history_df, (str, Path)):
            history_df = pl.read_parquet(history_df)
        if isinstance(target_df, (str, Path)):
            target_df = pl.read_parquet(target_df)

        log_memory(
            f"SimpleFeatureGenerator: Loaded data. History: {history_df.shape}, Target: {target_df.shape}"
        )
        tmp_dir = global_config.tmp_path
        tmp_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(dir=str(tmp_dir)) as tmpdir:
            history_path = os.path.join(tmpdir, "history.parquet")
            target_path = os.path.join(tmpdir, "target.parquet")

            history_df.write_parquet(history_path)
            target_df.write_parquet(target_path)

            del history_df
            if not same_source:
                del target_df
            gc.collect()

            result = self._process_with_duckdb(
                history_path, target_path, tmpdir, same_source
            )

        log_memory(f"SimpleFeatureGenerator: Done. Result shape: {result.shape}")
        return result

    def _process_with_duckdb(
        self,
        history_path: str,
        target_path: str,
        tmpdir: str,
        same_source: bool,
    ) -> pl.DataFrame:
        """Process features using DuckDB with strict temporal filtering."""

        con = setup_duckdb_connection(tmpdir, "simple_features.db")

        try:
            con.execute(f"""
                CREATE TABLE history AS
                SELECT 
                    uid,
                    item_id,
                    COALESCE(artist_id, 0) as artist_id,
                    COALESCE(album_id, 0) as album_id,
                    timestamp,
                    event_type,
                    COALESCE(is_organic, true) as is_organic,
                    (event_type = 'like')::INTEGER as is_like,
                    (event_type = 'dislike')::INTEGER as is_dislike
                FROM read_parquet('{history_path}')
                WHERE event_type IN ('like', 'dislike')
            """)

            history_count = con.execute("SELECT COUNT(*) FROM history").fetchone()[0]
            log_memory(f"SimpleFeatureGenerator: Loaded {history_count} history rows")

            con.execute(f"""
                CREATE TABLE target AS
                SELECT 
                    uid,
                    item_id,
                    COALESCE(artist_id, 0) as artist_id,
                    COALESCE(album_id, 0) as album_id,
                    timestamp,
                    COALESCE(event_type, 'unknown') as event_type
                FROM read_parquet('{target_path}')
            """)

            target_count = con.execute("SELECT COUNT(*) FROM target").fetchone()[0]
            log_memory(f"SimpleFeatureGenerator: Loaded {target_count} target rows")

            feature_queries = self._build_feature_queries()

            result_query = self._build_result_query(feature_queries, same_source)

            con.execute(f"""
                CREATE TABLE result AS
                {result_query}
            """)

            result_path = os.path.join(tmpdir, "result.parquet")
            con.execute(f"COPY result TO '{result_path}' (FORMAT PARQUET)")

            con.close()

            return pl.read_parquet(result_path)

        except Exception as e:
            con.close()
            raise e

    def _build_feature_queries(self) -> dict[str, str]:
        """Build SQL subqueries for each feature type."""

        queries = {}

        queries["user_life"] = """
            SELECT 
                t.uid,
                t.item_id,
                t.timestamp as t_timestamp,
                COALESCE(SUM(h.is_like), 0) as user_likes_life,
                COALESCE(SUM(h.is_dislike), 0) as user_dislikes_life,
                COALESCE(COUNT(*), 0) as user_total_life
            FROM target t
            LEFT JOIN history h 
                ON t.uid = h.uid 
                AND h.timestamp < t.timestamp
            GROUP BY t.uid, t.item_id, t.timestamp
        """

        queries["item_life"] = """
            SELECT 
                t.uid,
                t.item_id,
                t.timestamp as t_timestamp,
                COALESCE(SUM(h.is_like), 0) as item_likes_life,
                COALESCE(SUM(h.is_dislike), 0) as item_dislikes_life,
                COALESCE(COUNT(*), 0) as item_total_life
            FROM target t
            LEFT JOIN history h 
                ON t.item_id = h.item_id 
                AND h.timestamp < t.timestamp
            GROUP BY t.uid, t.item_id, t.timestamp
        """

        queries["user_item_life"] = """
            SELECT 
                t.uid,
                t.item_id,
                t.timestamp as t_timestamp,
                COALESCE(SUM(h.is_like), 0) as user_item_likes_life,
                COALESCE(COUNT(*), 0) as user_item_total_life
            FROM target t
            LEFT JOIN history h 
                ON t.uid = h.uid 
                AND t.item_id = h.item_id
                AND h.timestamp < t.timestamp
            GROUP BY t.uid, t.item_id, t.timestamp
        """

        queries["artist_life"] = """
            SELECT 
                t.uid,
                t.item_id,
                t.timestamp as t_timestamp,
                COALESCE(SUM(h.is_like), 0) as artist_likes_life,
                COALESCE(COUNT(*), 0) as artist_total_life
            FROM target t
            LEFT JOIN history h 
                ON t.artist_id = h.artist_id 
                AND t.artist_id != 0
                AND h.timestamp < t.timestamp
            GROUP BY t.uid, t.item_id, t.timestamp
        """

        queries["user_artist_life"] = """
            SELECT 
                t.uid,
                t.item_id,
                t.timestamp as t_timestamp,
                COALESCE(SUM(h.is_like), 0) as user_artist_likes_life,
                COALESCE(COUNT(*), 0) as user_artist_total_life
            FROM target t
            LEFT JOIN history h 
                ON t.uid = h.uid 
                AND t.artist_id = h.artist_id
                AND t.artist_id != 0
                AND h.timestamp < t.timestamp
            GROUP BY t.uid, t.item_id, t.timestamp
        """

        for window_days in self.windows_days:
            window_seconds = window_days * self.SECONDS_IN_DAY
            suffix = f"{window_days}d"

            queries[f"user_{suffix}"] = f"""
                SELECT 
                    t.uid,
                    t.item_id,
                    t.timestamp as t_timestamp,
                    COALESCE(SUM(h.is_like), 0) as user_likes_{suffix},
                    COALESCE(SUM(h.is_dislike), 0) as user_dislikes_{suffix},
                    COALESCE(COUNT(*), 0) as user_total_{suffix}
                FROM target t
                LEFT JOIN history h 
                    ON t.uid = h.uid 
                    AND h.timestamp < t.timestamp
                    AND h.timestamp >= t.timestamp - {window_seconds}
                GROUP BY t.uid, t.item_id, t.timestamp
            """

            queries[f"item_{suffix}"] = f"""
                SELECT 
                    t.uid,
                    t.item_id,
                    t.timestamp as t_timestamp,
                    COALESCE(SUM(h.is_like), 0) as item_likes_{suffix},
                    COALESCE(SUM(h.is_dislike), 0) as item_dislikes_{suffix},
                    COALESCE(COUNT(*), 0) as item_total_{suffix}
                FROM target t
                LEFT JOIN history h 
                    ON t.item_id = h.item_id 
                    AND h.timestamp < t.timestamp
                    AND h.timestamp >= t.timestamp - {window_seconds}
                GROUP BY t.uid, t.item_id, t.timestamp
            """

        return queries

    def _build_result_query(
        self,
        feature_queries: dict[str, str],
        same_source: bool,
    ) -> str:
        """Build the final result query joining all features."""

        cte_parts = []
        for name, query in feature_queries.items():
            cte_parts.append(f"{name} AS ({query})")

        cte_clause = "WITH " + ",\n".join(cte_parts)

        join_parts = []
        select_parts = [
            "t.uid",
            "t.item_id",
            "t.artist_id",
            "t.album_id",
            "t.timestamp",
        ]

        if same_source:
            select_parts.append(f"""
                CASE 
                    WHEN t.event_type = 'like' THEN 1.0 
                    WHEN t.event_type = 'dislike' THEN 0.0 
                    ELSE {self.target_for_fake} 
                END as target
            """)
        else:
            select_parts.append(f"{self.target_for_fake} as target")

        for name in feature_queries.keys():
            if "user_life" == name:
                select_parts.extend(
                    [
                        f"{name}.user_likes_life",
                        f"{name}.user_dislikes_life",
                        f"{name}.user_total_life",
                    ]
                )
            elif "item_life" == name:
                select_parts.extend(
                    [
                        f"{name}.item_likes_life",
                        f"{name}.item_dislikes_life",
                        f"{name}.item_total_life",
                    ]
                )
            elif "user_item_life" == name:
                select_parts.extend(
                    [f"{name}.user_item_likes_life", f"{name}.user_item_total_life"]
                )
            elif "artist_life" == name:
                select_parts.extend(
                    [f"{name}.artist_likes_life", f"{name}.artist_total_life"]
                )
            elif "user_artist_life" == name:
                select_parts.extend(
                    [f"{name}.user_artist_likes_life", f"{name}.user_artist_total_life"]
                )
            elif name.startswith("user_") and name.endswith("d"):
                suffix = name.replace("user_", "")
                select_parts.extend(
                    [
                        f"{name}.user_likes_{suffix}",
                        f"{name}.user_dislikes_{suffix}",
                        f"{name}.user_total_{suffix}",
                    ]
                )
            elif name.startswith("item_") and name.endswith("d"):
                suffix = name.replace("item_", "")
                select_parts.extend(
                    [
                        f"{name}.item_likes_{suffix}",
                        f"{name}.item_dislikes_{suffix}",
                        f"{name}.item_total_{suffix}",
                    ]
                )

            join_parts.append(f"""
                LEFT JOIN {name} 
                    ON t.uid = {name}.uid 
                    AND t.item_id = {name}.item_id 
                    AND t.timestamp = {name}.t_timestamp
            """)

        select_clause = ",\n    ".join(select_parts)
        join_clause = "\n".join(join_parts)

        return f"""
            {cte_clause}
            SELECT 
                {select_clause}
            FROM target t
            {join_clause}
        """


def create_simple_features(
    history_path: str | Path,
    target_path: str | Path | None = None,
    windows_days: list[int] | None = None,
    target_for_fake: float = 0.05,
) -> pl.DataFrame:
    """
    Convenience function to create features using SimpleFeatureGenerator.

    Args:
        history_path: Path to historical events parquet file.
        target_path: Path to target rows parquet file. If None, uses history_path.
        windows_days: List of window sizes in days.
        target_for_fake: Target value for synthetic samples.

    Returns:
        DataFrame with features.
    """
    generator = SimpleFeatureGenerator(
        windows_days=windows_days,
        target_for_fake=target_for_fake,
    )
    return generator.create_features(history_path, target_path)
