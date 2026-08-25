import duckdb
import os
from abc import ABC, abstractmethod
from typing import List, Union, Optional, Dict, Any
from pathlib import Path
from multiprocessing import Process, Queue
from dataclasses import dataclass
import logging
import polars as pl

from .utils import setup_duckdb_connection, log_memory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_STORAGE_DIR = ".cache/candidates"
SECONDS_IN_DAY = 86400


@dataclass
class GeneratorConfig:
    storage_dir: str = DEFAULT_STORAGE_DIR
    end_day: int = 0

    @property
    def db_name(self) -> str:
        return f"candidates_day_{self.end_day}.duckdb"

    @property
    def db_path(self) -> str:
        return os.path.join(self.storage_dir, self.db_name)

    @property
    def truncated_data_path(self) -> str:
        return os.path.join(
            self.storage_dir, f"data_day_{self.end_day}.parquet"
        )


class CandidateGeneratorStore:
    def __init__(self, config: GeneratorConfig):
        self.config = config
        self._conn: Optional[duckdb.DuckDBPyConnection] = None
        os.makedirs(self.config.storage_dir, exist_ok=True)

    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        if self._conn is None:
            log_memory("CandidateGeneratorStore: opening connection")
            self._conn = setup_duckdb_connection(
                self.config.storage_dir, self.config.db_name
            )
        return self._conn

    def reconnect(self):
        log_memory("CandidateGeneratorStore: reconnecting")
        if self._conn is not None:
            self._conn.close()
            self._conn = None
        _ = self.conn

    def table_exists(self, table_name: str) -> bool:
        result = self.conn.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
            (table_name,),
        ).fetchone()
        return result[0] > 0

    def drop_tables(self, table_names: List[str]):
        for table in table_names:
            try:
                self.conn.execute(f"DROP TABLE IF EXISTS {table}")
            except Exception as e:
                logger.warning(f"Failed to drop table {table}: {e}")

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    def cleanup(self):
        self.close()
        if os.path.exists(self.config.db_path):
            os.remove(self.config.db_path)


class BaseCandidateGenerator(ABC):
    def __init__(self, store: CandidateGeneratorStore):
        self.store = store
        self._is_fitted = False

    @property
    def generator_name(self) -> str:
        return self.__class__.__name__.lower()

    @property
    def table_prefix(self) -> str:
        return self.generator_name

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def _get_fit_params(self) -> Dict[str, Any]:
        return {}

    @abstractmethod
    def _get_table_names(self) -> List[str]:
        pass

    def fit(
        self, data_path: Union[str, Path], force: bool = False
    ) -> "BaseCandidateGenerator":
        generator_name = self.generator_name
        log_memory(f"{generator_name}.fit start force={force}")

        data_path = str(data_path)
        params = self._get_fit_params()

        source = f"read_parquet('{data_path}')"

        log_memory(f"{generator_name}.fit subprocess start")
        self._run_fit_in_subprocess(
            fit_func=self._fit_impl_static,
            source=source,
            db_path=self.store.config.db_path,
            storage_dir=self.store.config.storage_dir,
            table_prefix=self.table_prefix,
            force=force,
            **params,
        )
        log_memory(f"{generator_name}.fit subprocess end")

        self._is_fitted = True
        logger.info(f"{generator_name} fitted")
        return self

    @staticmethod
    def _fit_impl_static(
        source: str,
        db_path: str,
        storage_dir: str,
        table_prefix: str,
        force: bool,
        **kwargs,
    ):
        raise NotImplementedError()

    @staticmethod
    def _run_fit_in_subprocess(fit_func, **kwargs):
        queue = Queue()

        def target(q, func, **kw):
            try:
                func(**kw)
                q.put(("success", None))
            except Exception as e:
                import traceback

                q.put(("error", f"{str(e)}\n{traceback.format_exc()}"))

        process = Process(target=target, args=(queue, fit_func), kwargs=kwargs)
        process.start()
        process.join()

        status, result = queue.get()
        if status == "error":
            raise RuntimeError(f"Fitting failed in subprocess: {result}")

    @abstractmethod
    def generate_batch(
        self, users_df: pl.DataFrame, n_candidates: int = 100
    ) -> pl.DataFrame:
        pass

    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError("Generator not fitted. Call fit() first.")

    @staticmethod
    def _empty_result_df() -> pl.DataFrame:
        return pl.DataFrame(
            schema={"uid": pl.UInt64, "item_id": pl.UInt64, "source": pl.Utf8}
        )


def _table_exists_in_db(conn, table_name: str) -> bool:
    result = conn.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
        (table_name,),
    ).fetchone()
    return result[0] > 0


class PopularityBasedGenerator(BaseCandidateGenerator):
    def __init__(
        self,
        store: CandidateGeneratorStore,
        windows_days: Optional[List[int]] = None,
    ):
        super().__init__(store)
        self.windows_days = windows_days or [7, 30, 90]

    def _get_fit_params(self) -> Dict[str, Any]:
        return {
            "windows_days": self.windows_days,
            "end_day": self.store.config.end_day,
        }

    def _get_table_names(self) -> List[str]:
        return [f"{self.table_prefix}_window_{w}d" for w in self.windows_days]

    @staticmethod
    def _fit_impl_static(
        source: str,
        db_path: str,
        storage_dir: str,
        table_prefix: str,
        force: bool,
        windows_days: List[int],
        end_day: int,
    ):
        log_memory("PopularityBasedGenerator._fit_impl_static start")
        conn = setup_duckdb_connection(storage_dir, os.path.basename(db_path))

        try:
            ref_ts = end_day * SECONDS_IN_DAY

            for window_days in windows_days:
                table_name = f"{table_prefix}_window_{window_days}d"

                if not force and _table_exists_in_db(conn, table_name):
                    log_memory(
                        f"PopularityBasedGenerator: {table_name} exists, skipping"
                    )
                    continue

                log_memory(f"PopularityBasedGenerator: creating {table_name}")
                window_start = ref_ts - (window_days * SECONDS_IN_DAY)

                conn.execute(f"DROP TABLE IF EXISTS {table_name}")
                conn.execute(
                    f"""
                    CREATE TABLE {table_name} AS
                    SELECT 
                        CAST(item_id AS UBIGINT) as item_id,
                        COUNT(*) as like_count
                    FROM {source}
                    WHERE event_type = 'like'
                      AND timestamp >= {window_start}
                      AND timestamp <= {ref_ts}
                    GROUP BY item_id
                    ORDER BY like_count DESC
                """
                )
                conn.execute(
                    f"CREATE INDEX IF NOT EXISTS idx_{table_name}_likes ON {table_name}(like_count DESC)"
                )
                log_memory(f"PopularityBasedGenerator: {table_name} done")

        finally:
            conn.close()
        log_memory("PopularityBasedGenerator._fit_impl_static end")

    def generate_batch(
        self,
        users_df: pl.DataFrame,
        n_candidates: int = 100,
        window_days: Optional[int] = None,
    ) -> pl.DataFrame:
        log_memory("PopularityBasedGenerator.generate_batch start")
        self._check_fitted()

        if window_days is None:
            window_days = min(self.windows_days)

        if window_days not in self.windows_days:
            raise ValueError(
                f"Window {window_days} not available. Available: {self.windows_days}"
            )

        table_name = f"{self.table_prefix}_window_{window_days}d"
        source_name = f"popularity_{window_days}d"

        user_ids = users_df.select("uid").to_series().to_list()

        candidates_result = self.store.conn.execute(
            f"SELECT item_id FROM {table_name} ORDER BY like_count DESC LIMIT ?",
            (n_candidates,),
        ).fetchall()

        if not candidates_result:
            return self._empty_result_df()

        item_ids = [row[0] for row in candidates_result]
        items_df = pl.DataFrame(
            {
                "item_id": pl.Series(item_ids, dtype=pl.UInt64),
                "source": [source_name] * len(item_ids),
            }
        )

        users_select = users_df.select(pl.col("uid").cast(pl.UInt64))
        result_df = users_select.join(items_df, how="cross")

        log_memory(
            f"PopularityBasedGenerator.generate_batch end shape={result_df.shape}"
        )
        return result_df


class UserHistoryBasedGenerator(BaseCandidateGenerator):
    def __init__(
        self, store: CandidateGeneratorStore, min_cooccurrence: int = 2
    ):
        super().__init__(store)
        self.min_cooccurrence = min_cooccurrence

    def _get_fit_params(self) -> Dict[str, Any]:
        return {"min_cooccurrence": self.min_cooccurrence}

    def _get_table_names(self) -> List[str]:
        return [
            f"{self.table_prefix}_user_items",
            f"{self.table_prefix}_cooccurrence",
        ]

    @staticmethod
    def _fit_impl_static(
        source: str,
        db_path: str,
        storage_dir: str,
        table_prefix: str,
        force: bool,
        min_cooccurrence: int,
    ):
        log_memory("UserHistoryBasedGenerator._fit_impl_static start")
        conn = setup_duckdb_connection(storage_dir, os.path.basename(db_path))

        try:
            user_items_table = f"{table_prefix}_user_items"
            cooccurrence_table = f"{table_prefix}_cooccurrence"

            if not force and _table_exists_in_db(conn, cooccurrence_table):
                log_memory("UserHistoryBasedGenerator: tables exist, skipping")
                conn.close()
                return

            conn.execute(f"DROP TABLE IF EXISTS {cooccurrence_table}")
            conn.execute(f"DROP TABLE IF EXISTS {user_items_table}")

            log_memory("UserHistoryBasedGenerator: creating user_items")
            conn.execute(
                f"""
                CREATE TABLE {user_items_table} AS
                SELECT DISTINCT CAST(uid AS UBIGINT) as uid, CAST(item_id AS UBIGINT) as item_id
                FROM {source}
                WHERE event_type = 'like'
            """
            )

            log_memory("UserHistoryBasedGenerator: creating cooccurrence")
            conn.execute(
                f"""
                CREATE TABLE {cooccurrence_table} AS
                WITH pairs AS (
                    SELECT a.item_id as item_a, b.item_id as item_b, COUNT(DISTINCT a.uid) as cooccur_count
                    FROM {user_items_table} a
                    JOIN {user_items_table} b ON a.uid = b.uid
                    WHERE a.item_id < b.item_id
                    GROUP BY a.item_id, b.item_id
                    HAVING COUNT(DISTINCT a.uid) >= {min_cooccurrence}
                )
                SELECT item_a, item_b, cooccur_count FROM pairs
                UNION ALL
                SELECT item_b, item_a, cooccur_count FROM pairs
            """
            )

            log_memory("UserHistoryBasedGenerator: creating indexes")
            conn.execute(
                f"CREATE INDEX idx_{user_items_table}_user ON {user_items_table}(uid)"
            )
            conn.execute(
                f"CREATE INDEX idx_{cooccurrence_table}_item ON {cooccurrence_table}(item_a)"
            )

        finally:
            conn.close()
        log_memory("UserHistoryBasedGenerator._fit_impl_static end")

    def generate_batch(
        self, users_df: pl.DataFrame, n_candidates: int = 100
    ) -> pl.DataFrame:
        log_memory("UserHistoryBasedGenerator.generate_batch start")
        self._check_fitted()

        user_ids = users_df.select("uid").to_series().to_list()
        all_rows = []
        source_name = "cooccurrence"

        user_items_table = f"{self.table_prefix}_user_items"
        cooccurrence_table = f"{self.table_prefix}_cooccurrence"

        for i, user_id in enumerate(user_ids):
            if i % 1000 == 0 and i > 0:
                log_memory(
                    f"UserHistoryBasedGenerator.generate_batch user {i}/{len(user_ids)}"
                )

            candidates = self.store.conn.execute(
                f"""
                WITH user_items AS (SELECT item_id FROM {user_items_table} WHERE uid = ?),
                candidates AS (
                    SELECT c.item_b as item_id, SUM(c.cooccur_count) as score
                    FROM {cooccurrence_table} c
                    WHERE c.item_a IN (SELECT item_id FROM user_items)
                      AND c.item_b NOT IN (SELECT item_id FROM user_items)
                    GROUP BY c.item_b
                )
                SELECT item_id FROM candidates ORDER BY score DESC LIMIT ?
            """,
                (user_id, n_candidates),
            ).fetchall()

            for (item_id,) in candidates:
                all_rows.append((user_id, item_id, source_name))

        if not all_rows:
            return self._empty_result_df()

        result_df = pl.DataFrame(
            all_rows,
            schema={"uid": pl.UInt64, "item_id": pl.UInt64, "source": pl.Utf8},
            orient="row",
        )
        log_memory(
            f"UserHistoryBasedGenerator.generate_batch end shape={result_df.shape}"
        )
        return result_df


class ArtistBasedGenerator(BaseCandidateGenerator):
    def _get_fit_params(self) -> Dict[str, Any]:
        return {}

    def _get_table_names(self) -> List[str]:
        return [
            f"{self.table_prefix}_artist_items",
            f"{self.table_prefix}_user_artists",
            f"{self.table_prefix}_user_seen",
        ]

    @staticmethod
    def _fit_impl_static(
        source: str,
        db_path: str,
        storage_dir: str,
        table_prefix: str,
        force: bool,
    ):
        log_memory("ArtistBasedGenerator._fit_impl_static start")
        conn = setup_duckdb_connection(storage_dir, os.path.basename(db_path))

        try:
            artist_items = f"{table_prefix}_artist_items"
            user_artists = f"{table_prefix}_user_artists"
            user_seen = f"{table_prefix}_user_seen"

            if not force and _table_exists_in_db(conn, user_artists):
                log_memory("ArtistBasedGenerator: tables exist, skipping")
                conn.close()
                return

            conn.execute(f"DROP TABLE IF EXISTS {artist_items}")
            conn.execute(f"DROP TABLE IF EXISTS {user_artists}")
            conn.execute(f"DROP TABLE IF EXISTS {user_seen}")

            log_memory("ArtistBasedGenerator: creating artist_items")
            conn.execute(
                f"""
                CREATE TABLE {artist_items} AS
                SELECT CAST(artist_id AS UBIGINT) as artist_id, CAST(item_id AS UBIGINT) as item_id, COUNT(*) as popularity
                FROM {source} WHERE event_type = 'like'
                GROUP BY artist_id, item_id
            """
            )

            log_memory("ArtistBasedGenerator: creating user_artists")
            conn.execute(
                f"""
                CREATE TABLE {user_artists} AS
                SELECT CAST(uid AS UBIGINT) as uid, CAST(artist_id AS UBIGINT) as artist_id, COUNT(*) as like_count
                FROM {source} WHERE event_type = 'like'
                GROUP BY uid, artist_id
            """
            )

            log_memory("ArtistBasedGenerator: creating user_seen")
            conn.execute(
                f"""
                CREATE TABLE {user_seen} AS
                SELECT DISTINCT CAST(uid AS UBIGINT) as uid, CAST(item_id AS UBIGINT) as item_id
                FROM {source}
            """
            )

            log_memory("ArtistBasedGenerator: creating indexes")
            conn.execute(
                f"CREATE INDEX idx_{artist_items}_artist ON {artist_items}(artist_id)"
            )
            conn.execute(
                f"CREATE INDEX idx_{user_artists}_user ON {user_artists}(uid)"
            )
            conn.execute(
                f"CREATE INDEX idx_{user_seen}_user ON {user_seen}(uid)"
            )

        finally:
            conn.close()
        log_memory("ArtistBasedGenerator._fit_impl_static end")

    def generate_batch(
        self, users_df: pl.DataFrame, n_candidates: int = 100
    ) -> pl.DataFrame:
        log_memory("ArtistBasedGenerator.generate_batch start")
        self._check_fitted()

        user_ids = users_df.select("uid").to_series().to_list()
        all_rows = []
        source_name = "artist"

        artist_items = f"{self.table_prefix}_artist_items"
        user_artists = f"{self.table_prefix}_user_artists"
        user_seen = f"{self.table_prefix}_user_seen"

        for i, user_id in enumerate(user_ids):
            if i % 1000 == 0 and i > 0:
                log_memory(
                    f"ArtistBasedGenerator.generate_batch user {i}/{len(user_ids)}"
                )

            candidates = self.store.conn.execute(
                f"""
                WITH ua AS (SELECT artist_id, like_count FROM {user_artists} WHERE uid = ?),
                seen AS (SELECT item_id FROM {user_seen} WHERE uid = ?),
                cand AS (
                    SELECT ai.item_id, SUM(ai.popularity * ua.like_count) as score
                    FROM {artist_items} ai
                    JOIN ua ON ai.artist_id = ua.artist_id
                    WHERE ai.item_id NOT IN (SELECT item_id FROM seen)
                    GROUP BY ai.item_id
                )
                SELECT item_id FROM cand ORDER BY score DESC LIMIT ?
            """,
                (user_id, user_id, n_candidates),
            ).fetchall()

            for (item_id,) in candidates:
                all_rows.append((user_id, item_id, source_name))

        if not all_rows:
            return self._empty_result_df()

        result_df = pl.DataFrame(
            all_rows,
            schema={"uid": pl.UInt64, "item_id": pl.UInt64, "source": pl.Utf8},
            orient="row",
        )
        log_memory(
            f"ArtistBasedGenerator.generate_batch end shape={result_df.shape}"
        )
        return result_df


class AlbumBasedGenerator(BaseCandidateGenerator):
    def _get_fit_params(self) -> Dict[str, Any]:
        return {}

    def _get_table_names(self) -> List[str]:
        return [
            f"{self.table_prefix}_album_items",
            f"{self.table_prefix}_user_albums",
            f"{self.table_prefix}_user_seen",
        ]

    @staticmethod
    def _fit_impl_static(
        source: str,
        db_path: str,
        storage_dir: str,
        table_prefix: str,
        force: bool,
    ):
        log_memory("AlbumBasedGenerator._fit_impl_static start")
        conn = setup_duckdb_connection(storage_dir, os.path.basename(db_path))

        try:
            album_items = f"{table_prefix}_album_items"
            user_albums = f"{table_prefix}_user_albums"
            user_seen = f"{table_prefix}_user_seen"

            if not force and _table_exists_in_db(conn, user_albums):
                log_memory("AlbumBasedGenerator: tables exist, skipping")
                conn.close()
                return

            conn.execute(f"DROP TABLE IF EXISTS {album_items}")
            conn.execute(f"DROP TABLE IF EXISTS {user_albums}")
            conn.execute(f"DROP TABLE IF EXISTS {user_seen}")

            log_memory("AlbumBasedGenerator: creating album_items")
            conn.execute(
                f"""
                CREATE TABLE {album_items} AS
                SELECT CAST(album_id AS UBIGINT) as album_id, CAST(item_id AS UBIGINT) as item_id, COUNT(*) as popularity
                FROM {source} WHERE event_type = 'like'
                GROUP BY album_id, item_id
            """
            )

            log_memory("AlbumBasedGenerator: creating user_albums")
            conn.execute(
                f"""
                CREATE TABLE {user_albums} AS
                SELECT CAST(uid AS UBIGINT) as uid, CAST(album_id AS UBIGINT) as album_id, COUNT(*) as like_count
                FROM {source} WHERE event_type = 'like'
                GROUP BY uid, album_id
            """
            )

            log_memory("AlbumBasedGenerator: creating user_seen")
            conn.execute(
                f"""
                CREATE TABLE {user_seen} AS
                SELECT DISTINCT CAST(uid AS UBIGINT) as uid, CAST(item_id AS UBIGINT) as item_id
                FROM {source}
            """
            )

            log_memory("AlbumBasedGenerator: creating indexes")
            conn.execute(
                f"CREATE INDEX idx_{album_items}_album ON {album_items}(album_id)"
            )
            conn.execute(
                f"CREATE INDEX idx_{user_albums}_user ON {user_albums}(uid)"
            )
            conn.execute(
                f"CREATE INDEX idx_{user_seen}_user ON {user_seen}(uid)"
            )

        finally:
            conn.close()
        log_memory("AlbumBasedGenerator._fit_impl_static end")

    def generate_batch(
        self, users_df: pl.DataFrame, n_candidates: int = 100
    ) -> pl.DataFrame:
        log_memory("AlbumBasedGenerator.generate_batch start")
        self._check_fitted()

        user_ids = users_df.select("uid").to_series().to_list()
        all_rows = []
        source_name = "album"

        album_items = f"{self.table_prefix}_album_items"
        user_albums = f"{self.table_prefix}_user_albums"
        user_seen = f"{self.table_prefix}_user_seen"

        for i, user_id in enumerate(user_ids):
            if i % 1000 == 0 and i > 0:
                log_memory(
                    f"AlbumBasedGenerator.generate_batch user {i}/{len(user_ids)}"
                )

            candidates = self.store.conn.execute(
                f"""
                WITH ua AS (SELECT album_id, like_count FROM {user_albums} WHERE uid = ?),
                seen AS (SELECT item_id FROM {user_seen} WHERE uid = ?),
                cand AS (
                    SELECT ai.item_id, SUM(ai.popularity * ua.like_count) as score
                    FROM {album_items} ai
                    JOIN ua ON ai.album_id = ua.album_id
                    WHERE ai.item_id NOT IN (SELECT item_id FROM seen)
                    GROUP BY ai.item_id
                )
                SELECT item_id FROM cand ORDER BY score DESC LIMIT ?
            """,
                (user_id, user_id, n_candidates),
            ).fetchall()

            for (item_id,) in candidates:
                all_rows.append((user_id, item_id, source_name))

        if not all_rows:
            return self._empty_result_df()

        result_df = pl.DataFrame(
            all_rows,
            schema={"uid": pl.UInt64, "item_id": pl.UInt64, "source": pl.Utf8},
            orient="row",
        )
        log_memory(
            f"AlbumBasedGenerator.generate_batch end shape={result_df.shape}"
        )
        return result_df


class EnsembleCandidateGenerator:
    def __init__(self, generators: List[tuple], config: GeneratorConfig):
        self.config = config
        self.store = CandidateGeneratorStore(config)
        self.generators: List[tuple] = []

        for gen_tuple in generators:
            if len(gen_tuple) == 2:
                gen_class, weight = gen_tuple
                kwargs = {}
            else:
                gen_class, weight, kwargs = gen_tuple
            generator = gen_class(store=self.store, **kwargs)
            self.generators.append((generator, weight))

    @property
    def is_fitted(self) -> bool:
        return all(gen.is_fitted for gen, _ in self.generators)

    @property
    def total_weight(self) -> float:
        return sum(weight for _, weight in self.generators)

    def _compute_n_candidates_per_generator(
        self, n_candidates: int
    ) -> List[int]:
        total_weight = self.total_weight
        allocations = [
            int(n_candidates * w / total_weight) for _, w in self.generators
        ]
        remainder = n_candidates - sum(allocations)
        if remainder > 0:
            sorted_indices = sorted(
                range(len(self.generators)),
                key=lambda i: self.generators[i][1],
                reverse=True,
            )
            for i in range(remainder):
                allocations[sorted_indices[i % len(sorted_indices)]] += 1
        return allocations

    def fit(
        self, data_path: Union[str, Path], force: bool = False
    ) -> "EnsembleCandidateGenerator":
        log_memory("EnsembleCandidateGenerator.fit start")

        for generator, _ in self.generators:
            generator.fit(data_path, force=force)

        # IMPORTANT: reconnect after all subprocesses have written to DB
        self.store.reconnect()

        logger.info("EnsembleCandidateGenerator ready")
        log_memory("EnsembleCandidateGenerator.fit end")
        return self

    @staticmethod
    def _empty_result_df() -> pl.DataFrame:
        return pl.DataFrame(
            schema={"uid": pl.UInt64, "item_id": pl.UInt64, "source": pl.Utf8}
        )

    def generate_batch(
        self,
        users_df: pl.DataFrame,
        n_candidates: int = 100,
        generator_kwargs: Optional[Dict[str, Dict]] = None,
        deduplicate: bool = True,
    ) -> pl.DataFrame:
        log_memory("EnsembleCandidateGenerator.generate_batch start")
        if not self.is_fitted:
            raise RuntimeError("Ensemble not fitted. Call fit() first.")

        generator_kwargs = generator_kwargs or {}
        allocations = self._compute_n_candidates_per_generator(n_candidates)
        all_results = []

        for (generator, _), n_alloc in zip(self.generators, allocations):
            if n_alloc <= 0:
                continue

            gen_name = generator.__class__.__name__
            try:
                extra_kwargs = generator_kwargs.get(gen_name, {})
                request_n = n_alloc * 2 if deduplicate else n_alloc
                candidates_df = generator.generate_batch(
                    users_df, request_n, **extra_kwargs
                )
                if candidates_df.height > 0:
                    all_results.append(candidates_df)
            except Exception as e:
                logger.warning(f"Generator {gen_name} failed: {e}")
                continue

        if not all_results:
            return self._empty_result_df()

        log_memory("EnsembleCandidateGenerator.generate_batch combining")
        combined_df = pl.concat(all_results)

        if deduplicate:
            combined_df = combined_df.unique(
                subset=["uid", "item_id"], keep="first"
            )

        result_df = combined_df.group_by("uid", maintain_order=True).head(
            n_candidates
        )
        log_memory(
            f"EnsembleCandidateGenerator.generate_batch end shape={result_df.shape}"
        )
        return result_df

    def get_allocation_info(self, n_candidates: int = 100) -> Dict[str, int]:
        allocations = self._compute_n_candidates_per_generator(n_candidates)
        return {
            gen.__class__.__name__: n
            for (gen, _), n in zip(self.generators, allocations)
        }

    def cleanup(self):
        self.store.cleanup()


def _truncate_data_impl(
    full_data_path: str,
    truncated_data_path: str,
    end_timestamp: int,
    storage_dir: str,
):
    log_memory("_truncate_data_impl start")
    conn = setup_duckdb_connection(storage_dir, "temp_truncate.duckdb")
    try:
        conn.execute(
            f"""
            COPY (
                SELECT * FROM read_parquet('{full_data_path}')
                WHERE timestamp <= {end_timestamp}
            ) TO '{truncated_data_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
        )
    finally:
        conn.close()
    log_memory("_truncate_data_impl end")


def _run_truncate_in_subprocess(
    full_data_path: str,
    truncated_data_path: str,
    end_timestamp: int,
    storage_dir: str,
):
    queue = Queue()

    def target(q, **kw):
        try:
            _truncate_data_impl(**kw)
            q.put(("success", None))
        except Exception as e:
            import traceback

            q.put(("error", f"{str(e)}\n{traceback.format_exc()}"))

    process = Process(
        target=target,
        args=(queue,),
        kwargs={
            "full_data_path": full_data_path,
            "truncated_data_path": truncated_data_path,
            "end_timestamp": end_timestamp,
            "storage_dir": storage_dir,
        },
    )
    process.start()
    process.join()

    status, result = queue.get()
    if status == "error":
        raise RuntimeError(f"Data truncation failed: {result}")


def create_ensemble(
    generators_config: List[tuple],
    full_data_path: Union[str, Path],
    end_day: int,
    storage_dir: str = DEFAULT_STORAGE_DIR,
    force: bool = False,
) -> EnsembleCandidateGenerator:
    log_memory(f"create_ensemble start end_day={end_day} force={force}")

    full_data_path = str(full_data_path)
    end_timestamp = end_day * SECONDS_IN_DAY

    os.makedirs(storage_dir, exist_ok=True)
    config = GeneratorConfig(storage_dir=storage_dir, end_day=end_day)
    truncated_data_path = config.truncated_data_path

    if force or not os.path.exists(truncated_data_path):
        log_memory("create_ensemble: truncating data")
        _run_truncate_in_subprocess(
            full_data_path, truncated_data_path, end_timestamp, storage_dir
        )
        log_memory("create_ensemble: truncation done")

    ensemble = EnsembleCandidateGenerator(generators_config, config)
    ensemble.fit(truncated_data_path, force=force)

    log_memory("create_ensemble end")
    return ensemble
