from pathlib import Path

import polars as pl
import pytest

from dcn.datasets.yambda import HomeworkYambdaDatasetSource
from utils.global_config import config as global_config

SECONDS_IN_DAY = 86_400

ORGANIC, RECOMMENDED = 1, 0

# (uid, timestamp, item_id, event_type, is_organic, played_ratio_pct)
_EVENTS = [
    (1, 0, 10, "listen", RECOMMENDED, 100),
    (1, 60, 11, "listen", RECOMMENDED, 10),
    (1, 120, 12, "listen", ORGANIC, 100),
    (1, 180, 10, "like", RECOMMENDED, 0),
    (2, 0, 10, "listen", RECOMMENDED, 100),
    (2, 60, 11, "listen", RECOMMENDED, 100),
    (3, 0, 10, "listen", RECOMMENDED, 100),
    (3, 100, 10, "like", RECOMMENDED, 0),
    (3, 200, 10, "listen", RECOMMENDED, 100),
    (4, 0, 10, "listen", RECOMMENDED, 100),
    (4, 2 * SECONDS_IN_DAY, 10, "like", RECOMMENDED, 0),
]

_ITEMS = [10, 11, 12]


@pytest.fixture
def yambda_dir(tmp_path: Path) -> Path:
    data_path = tmp_path / "yambda_data"
    (data_path / "flat" / "50m").mkdir(parents=True)

    pl.DataFrame(
        _EVENTS,
        schema=[
            "uid",
            "timestamp",
            "item_id",
            "event_type",
            "is_organic",
            "played_ratio_pct",
        ],
        orient="row",
    ).with_columns(track_length_seconds=pl.lit(200)).write_parquet(
        data_path / "flat" / "50m" / "multi_event.parquet"
    )

    pl.DataFrame({"item_id": _ITEMS, "artist_id": [100, 101, 101]}).write_parquet(
        data_path / "artist_item_mapping.parquet"
    )
    pl.DataFrame({"item_id": _ITEMS, "album_id": [200, 200, 201]}).write_parquet(
        data_path / "album_item_mapping.parquet"
    )
    pl.DataFrame(
        {"item_id": _ITEMS, "normalized_embed": [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]}
    ).write_parquet(data_path / "embeddings.parquet")

    global_config.initialize(tmp_path)
    return data_path


@pytest.fixture
def listens(yambda_dir: Path) -> pl.DataFrame:
    source = HomeworkYambdaDatasetSource(data_path=yambda_dir, size="50m")
    return pl.read_parquet(source.artifacts.main_parquet).sort("uid", "timestamp")


def _of_user(listens: pl.DataFrame, uid: int) -> pl.DataFrame:
    return listens.filter(pl.col("uid") == uid)


def test_only_recommended_listens_become_rows(listens: pl.DataFrame) -> None:
    assert _of_user(listens, 1)["timestamp"].to_list() == [0, 60]


def test_a_full_play_is_more_than_95_percent(listens: pl.DataFrame) -> None:
    assert _of_user(listens, 1)["target_full_play"].to_list() == [1.0, 0.0]
    assert _of_user(listens, 1)["is_skip"].to_list() == [False, True]


def test_a_like_labels_the_listen_of_the_same_track(listens: pl.DataFrame) -> None:
    assert _of_user(listens, 1)["target_like"].to_list() == [1.0, 0.0]


def test_an_equidistant_like_labels_the_earlier_listen(listens: pl.DataFrame) -> None:
    assert _of_user(listens, 3)["target_like"].to_list() == [1.0, 0.0]


def test_a_like_beyond_the_attribution_window_labels_nothing(
    listens: pl.DataFrame,
) -> None:
    assert _of_user(listens, 4)["target_like"].to_list() == [0.0]


def test_a_listen_whose_neighbours_agree_is_not_a_preference_pair(
    listens: pl.DataFrame,
) -> None:
    assert _of_user(listens, 1)["is_preference_pair"].to_list() == [True, True]
    assert _of_user(listens, 2)["is_preference_pair"].to_list() == [False, False]
