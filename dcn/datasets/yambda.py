import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import kagglehub
import polars as pl
from kagglehub import KaggleDatasetAdapter

from data.utils import setup_duckdb_connection, to_day
from utils.global_config import config as global_config
from utils.locks import hold

from .base import DatasetSource, DatasetSourceArtifacts
from .remap import apply_id_remap_to_parquet, build_id_remap_and_remapped_embeddings

logger = logging.getLogger(__name__)

YambdaSize = Literal["50m", "500m", "5b"]

YAMBDA_USER_COLUMN = "uid"
YAMBDA_ITEM_ID_COLUMN = "compact_item_id"

YAMBDA_ID_COLUMNS = [
    "item_id",
    YAMBDA_ITEM_ID_COLUMN,
    YAMBDA_USER_COLUMN,
    "album_id",
    "artist_id",
]

_HASH_BUCKETS = 1_000_000

# The ranking homework's thresholds, kept so its numbers are comparable.
FULL_PLAY_PERCENT = 95
SKIP_PERCENT = 50
LIKE_ATTRIBUTION_SECONDS = 24 * 60 * 60

# The dataset's own enum order.
EVENT_TYPE_IDS: dict[str, int] = {
    "listen": 0,
    "dislike": 1,
    "like": 2,
    "undislike": 3,
    "unlike": 4,
}


@dataclass(frozen=True)
class UserSample:
    """How much of the user base to keep. Samples *users*, never events."""

    max_users: int | None = None
    fraction: float | None = None
    seed: int = 42

    def __post_init__(self) -> None:
        if (self.max_users is None) == (self.fraction is None):
            raise ValueError("set exactly one of max_users / fraction")
        if self.fraction is not None and not 0 < self.fraction <= 1:
            raise ValueError(f"fraction must be in (0, 1], got {self.fraction}")

    @property
    def name(self) -> str:
        chosen = (
            f"{self.max_users}users"
            if self.max_users is not None
            else f"{self.fraction:g}frac"
        )
        return f"{chosen}_seed{self.seed}"

    def duckdb_query(self, uid_source: str) -> str:
        """SQL picking the sampled uids out of ``uid_source``.

        By hash rather than ``USING SAMPLE``, which depends on an input order the
        preparation connection does not preserve.
        """
        bucket = f"hash(uid || '_{self.seed}')"
        if self.max_users is not None:
            return (
                f"SELECT uid FROM (SELECT DISTINCT uid FROM {uid_source}) "
                f"ORDER BY {bucket}, uid LIMIT {self.max_users}"
            )
        keep = round(self.fraction * _HASH_BUCKETS)
        return (
            f"SELECT DISTINCT uid FROM {uid_source} "
            f"WHERE {bucket} % {_HASH_BUCKETS} < {keep}"
        )


def _get_test_users() -> pl.DataFrame:
    test_users = kagglehub.dataset_load(
        KaggleDatasetAdapter.POLARS,
        "thekabeton/ysda-recsys-2026-yambda-dataset/versions/3",
        "test_users.csv",
    ).collect()
    return test_users.select("uid")


class YambdaDatasetSource(DatasetSource):
    """Yambda multi-event interactions, prepared for the training pipeline."""

    event_columns = [
        *YAMBDA_ID_COLUMNS,
        "event_type_id",
        "target_like",
        "target_listen",
        "listen_mask",
    ]

    def __init__(
        self,
        data_path: Path,
        size: YambdaSize,
        *,
        listen_sample_fraction: float = 1.0,
        user_sample: UserSample | None = None,
        approx_test_users_only: bool = False,
        event_type_filter: str | None = None,
        min_item_interactions_per_item: int = 0,
        drop_unmapped_items: bool = False,
        invalidate_cache: bool = False,
    ):
        if user_sample is not None and approx_test_users_only:
            raise ValueError(
                "user_sample and approx_test_users_only both pick the user base; "
                "set at most one"
            )
        if not 0 < listen_sample_fraction <= 1:
            raise ValueError(
                f"listen_sample_fraction must be in (0, 1], got {listen_sample_fraction}"
            )
        if event_type_filter is not None and event_type_filter not in EVENT_TYPE_IDS:
            raise ValueError(f"unknown event_type_filter {event_type_filter!r}")
        if min_item_interactions_per_item < 0:
            raise ValueError("min_item_interactions_per_item must be non-negative")

        self.data_path = Path(data_path)
        self.size = size
        self.listen_sample_fraction = listen_sample_fraction
        self.user_sample = user_sample
        self.approx_test_users_only = approx_test_users_only
        self.event_type_filter = event_type_filter
        self.min_item_interactions_per_item = min_item_interactions_per_item
        self.drop_unmapped_items = drop_unmapped_items

        self._artifacts = self._prepare(invalidate_cache)

    @property
    def artifacts(self) -> DatasetSourceArtifacts:
        return self._artifacts

    @property
    def _multi_event_path(self) -> Path:
        return self.data_path / "flat" / self.size / "multi_event.parquet"

    @property
    def _work_dir_name(self) -> str:
        """Every knob that changes the prepared parquet, in the directory name.

        ``Experiment.dataset_key`` hashes this path, so a knob left out hands the
        whole cached chain below it the previous run's events.
        """
        parts = [self.size]
        if self.user_sample is not None:
            parts.append(self.user_sample.name)
        if self.approx_test_users_only:
            parts.append("testusers")
        if self.listen_sample_fraction < 1.0:
            parts.append(f"listen{self.listen_sample_fraction:g}")
        if self.event_type_filter is not None:
            parts.append(self.event_type_filter)
        if self.min_item_interactions_per_item:
            parts.append(f"core{self.min_item_interactions_per_item}")
        if self.drop_unmapped_items:
            parts.append("knownitems")
        return "_".join(parts)

    def _work_dir(self) -> Path:
        work_dir = global_config.datasets_path("yambda") / self._work_dir_name
        work_dir.mkdir(parents=True, exist_ok=True)
        return work_dir

    def _prepare(self, invalidate_cache: bool) -> DatasetSourceArtifacts:
        work_dir = self._work_dir()
        with hold(work_dir / "source.lock", "yambda source"):
            return self._prepare_locked(work_dir, invalidate_cache)

    def _prepare_locked(
        self, work_dir: Path, invalidate_cache: bool
    ) -> DatasetSourceArtifacts:
        logger.info(
            "Preparing yambda (size=%s, user_sample=%s, listen_fraction=%s) in %s",
            self.size,
            self.user_sample,
            self.listen_sample_fraction,
            work_dir,
        )

        events_parquet = work_dir / "events.parquet"
        if invalidate_cache or not events_parquet.exists():
            self._build_events_parquet(events_parquet)
        else:
            logger.info("Reusing existing yambda events parquet: %s", events_parquet)

        remap_parquet = work_dir / "item_id_remap.parquet"
        compact_embeddings_parquet = work_dir / "embeddings_compact.parquet"
        main_parquet = work_dir / "events_remapped.parquet"

        if (
            invalidate_cache
            or not remap_parquet.exists()
            or not compact_embeddings_parquet.exists()
        ):
            build_id_remap_and_remapped_embeddings(
                main_parquet=events_parquet,
                embeddings_parquet=self.data_path / "embeddings.parquet",
                remap_parquet=remap_parquet,
                remapped_embeddings_parquet=compact_embeddings_parquet,
                raw_id_column="item_id",
                embedding_column="normalized_embed",
            )
        else:
            logger.info("Reusing existing item_id remap: %s", remap_parquet)

        if invalidate_cache or not main_parquet.exists():
            apply_id_remap_to_parquet(
                main_parquet=events_parquet,
                remap_parquet=remap_parquet,
                output_parquet=main_parquet,
                id_column="item_id",
                compact_column=YAMBDA_ITEM_ID_COLUMN,
                drop_unmapped=self.drop_unmapped_items,
            )
        else:
            logger.info("Reusing existing remapped main parquet: %s", main_parquet)

        return DatasetSourceArtifacts(
            main_parquet=main_parquet,
            columns=list(self.event_columns),
            precomputed_embeddings={YAMBDA_ITEM_ID_COLUMN: compact_embeddings_parquet},
            timestamp_column="timestamp",
            user_column=YAMBDA_USER_COLUMN,
            item_id_column=YAMBDA_ITEM_ID_COLUMN,
        )

    def _selected_users_query(self) -> str | None:
        if self.approx_test_users_only:
            return "SELECT DISTINCT uid FROM test_users"
        if self.user_sample is not None:
            return self.user_sample.duckdb_query("events")
        return None

    def _sampled_events_ctes(self) -> str:
        """CTEs ending in ``sampled``: the events this run trains on.

        Listens outnumber the rest by two orders of magnitude, so only they are
        thinned, and by an event hash so the CTE names the same events on re-run.
        """
        if (selected := self._selected_users_query()) is None:
            user_events = "SELECT * FROM events"
        else:
            user_events = (
                f"WITH selected_users AS MATERIALIZED ({selected}) "
                f"SELECT * FROM events WHERE uid IN (SELECT uid FROM selected_users)"
            )

        ctes = [f"user_events AS ({user_events})"]
        if self.listen_sample_fraction >= 1.0:
            sampled_source = "SELECT * FROM user_events"
        else:
            keep = round(self.listen_sample_fraction * _HASH_BUCKETS)
            event_hash = "hash(uid || '_' || item_id || '_' || timestamp)"
            sampled_source = f"""
                SELECT * FROM user_events
                WHERE event_type <> 'listen'
                   OR {event_hash} % {_HASH_BUCKETS} < {keep}
            """
        ctes.append(f"sampled_events AS ({sampled_source})")

        filtered_source = "SELECT * FROM sampled_events"
        if self.event_type_filter is not None:
            filtered_source += f" WHERE event_type = '{self.event_type_filter}'"
        ctes.append(f"filtered_events AS ({filtered_source})")

        if self.min_item_interactions_per_item:
            ctes.append(
                "core_items AS ("
                "SELECT item_id FROM filtered_events GROUP BY item_id "
                f"HAVING count(*) >= {self.min_item_interactions_per_item}"
                ")"
            )
            ctes.append(
                "sampled AS (SELECT * FROM filtered_events "
                "WHERE item_id IN (SELECT item_id FROM core_items))"
            )
        else:
            ctes.append("sampled AS (SELECT * FROM filtered_events)")
        return ",\n".join(ctes)

    def _side_feature_ctes(self) -> str:
        return f"""
            item_artists AS (
                SELECT item_id, LIST(DISTINCT artist_id) AS artist_id
                FROM '{self.data_path / "artist_item_mapping.parquet"}'
                GROUP BY item_id
            ),
            item_albums AS (
                SELECT item_id, LIST(DISTINCT album_id) AS album_id
                FROM '{self.data_path / "album_item_mapping.parquet"}'
                GROUP BY item_id
            )
        """

    def _events_query(self) -> str:
        """The rows and targets this source trains on, as one SELECT.

        Reads the ``sampled`` CTE and the item side-feature ones. A subclass
        that wants different rows or different targets overrides this rather
        than the copy machinery around it.
        """
        event_type_case = " ".join(
            f"WHEN '{name}' THEN {value}" for name, value in EVENT_TYPE_IDS.items()
        )
        return f"""
            WITH {self._sampled_events_ctes()}, {self._side_feature_ctes()}
            SELECT
                sampled.uid,
                sampled.item_id,
                sampled.event_type,
                CASE sampled.event_type {event_type_case}
                     ELSE error('unknown event_type: '
                                || sampled.event_type) END AS event_type_id,
                sampled.timestamp,
                sampled.is_organic,
                sampled.played_ratio_pct AS listen_share,
                sampled.track_length_seconds AS track_length,
                COALESCE(artists.artist_id, [0]::BIGINT[]) AS artist_id,
                COALESCE(albums.album_id, [0]::BIGINT[]) AS album_id,
                (sampled.event_type = 'like')::FLOAT AS target_like,
                (CASE WHEN sampled.event_type = 'listen'
                      THEN sampled.played_ratio_pct / 100.0
                      ELSE 0.0 END)::FLOAT AS target_listen,
                (sampled.event_type = 'listen') AS listen_mask,
                {to_day("sampled.timestamp")} AS day
            FROM sampled
            LEFT JOIN item_artists artists ON sampled.item_id = artists.item_id
            LEFT JOIN item_albums albums ON sampled.item_id = albums.item_id
        """

    def _build_events_parquet(self, output_path: Path) -> None:
        tmp_dir = global_config.tmp_path
        tmp_dir.mkdir(parents=True, exist_ok=True)
        connection = setup_duckdb_connection(str(tmp_dir), "yambda_prep")
        # Under the final name only once complete: an interrupted COPY leaves a
        # truncated file, which the caller's existence check would call a hit.
        partial_path = output_path.with_suffix(".partial")
        try:
            connection.execute(
                f"CREATE OR REPLACE VIEW events AS "
                f"SELECT * FROM '{self._multi_event_path}'"
            )
            if self.approx_test_users_only:
                connection.register("test_users", _get_test_users())

            connection.execute(
                f"COPY ({self._events_query()}) TO '{partial_path}' (FORMAT PARQUET)"
            )
            partial_path.rename(output_path)
        finally:
            connection.close()
            partial_path.unlink(missing_ok=True)
        logger.info("Wrote yambda events parquet: %s", output_path)


class HomeworkYambdaDatasetSource(YambdaDatasetSource):
    """Yambda as the ranking homework derives it.

    One row per recommended listen, labelled with two binary targets: was the
    track played to the end, and was it liked. Yambda records a like as its own
    event, so a like is attributed to the listen of the same track it most
    likely refers to. ``is_preference_pair`` marks the listens a ranker learns
    from -- those whose feedback differs from a time-adjacent neighbour's.

    Construction and rationale: https://huggingface.co/datasets/matfu21/yambda-50m-lag-features
    """

    event_columns = [*YAMBDA_ID_COLUMNS, "target_like", "target_full_play"]

    @property
    def _work_dir_name(self) -> str:
        return f"homework_{super()._work_dir_name}"

    def _events_query(self) -> str:
        return f"""
            WITH {self._sampled_events_ctes()}, {self._side_feature_ctes()},
            recommended AS (
                SELECT * FROM sampled WHERE NOT CAST(is_organic AS BOOLEAN)
            ),
            listens AS (
                SELECT * REPLACE (timestamp::BIGINT AS timestamp),
                       row_number() OVER () AS listen_id
                FROM recommended
                WHERE event_type = 'listen'
            ),
            liked AS (
                SELECT DISTINCT listens.listen_id
                FROM recommended likes
                JOIN listens
                  ON listens.uid = likes.uid
                 AND listens.item_id = likes.item_id
                 AND abs(listens.timestamp - likes.timestamp::BIGINT)
                     <= {LIKE_ATTRIBUTION_SECONDS}
                WHERE likes.event_type = 'like'
                QUALIFY row_number() OVER (
                    PARTITION BY likes.uid, likes.item_id, likes.timestamp
                    ORDER BY abs(listens.timestamp - likes.timestamp::BIGINT),
                             listens.timestamp
                ) = 1
            ),
            labelled AS (
                SELECT
                    listens.uid,
                    listens.item_id,
                    listens.timestamp,
                    listens.played_ratio_pct AS listen_share,
                    listens.track_length_seconds AS track_length,
                    (listens.listen_id IN (SELECT listen_id FROM liked))::FLOAT
                        AS target_like,
                    (listens.played_ratio_pct > {FULL_PLAY_PERCENT})::FLOAT
                        AS target_full_play,
                    listens.played_ratio_pct < {SKIP_PERCENT} AS is_skip
                FROM listens
            )
            SELECT
                labelled.*,
                COALESCE(artists.artist_id, [0]::BIGINT[]) AS artist_id,
                COALESCE(albums.album_id, [0]::BIGINT[]) AS album_id,
                COALESCE(target_like <> lag(target_like) OVER neighbours, FALSE)
                    OR COALESCE(target_like <> lead(target_like) OVER neighbours, FALSE)
                    OR COALESCE(
                        target_full_play <> lag(target_full_play) OVER neighbours, FALSE
                    )
                    OR COALESCE(
                        target_full_play <> lead(target_full_play) OVER neighbours, FALSE
                    ) AS is_preference_pair,
                {to_day("labelled.timestamp")} AS day
            FROM labelled
            LEFT JOIN item_artists artists ON labelled.item_id = artists.item_id
            LEFT JOIN item_albums albums ON labelled.item_id = albums.item_id
            WINDOW neighbours AS (PARTITION BY labelled.uid ORDER BY labelled.timestamp)
        """
