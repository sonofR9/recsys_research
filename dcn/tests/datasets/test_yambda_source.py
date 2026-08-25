import threading
import time
from pathlib import Path

import polars as pl
import pytest

from dcn.datasets.yambda import EVENT_TYPE_IDS, UserSample, YambdaDatasetSource
from utils.global_config import config as global_config

SECONDS_IN_DAY = 86_400
USERS = [1, 2, 3, 4, 5, 6]
ITEMS = [10, 11, 12]
EVENTS_PER_USER = 4


@pytest.fixture
def yambda_dir(tmp_path: Path) -> Path:
    """A miniature yambda layout: every user has the same number of events, so
    an event-level sample is distinguishable from a user-level one."""
    data_path = tmp_path / "yambda_data"
    (data_path / "flat" / "50m").mkdir(parents=True)

    rows = []
    for user in USERS:
        for position in range(EVENTS_PER_USER):
            rows.append(
                {
                    "uid": user,
                    "timestamp": position * SECONDS_IN_DAY + user,
                    "item_id": ITEMS[position % len(ITEMS)],
                    "is_organic": 1,
                    "played_ratio_pct": 50 + position,
                    "track_length_seconds": 200,
                    "event_type": "like" if position == 0 else "listen",
                }
            )
    pl.DataFrame(rows).write_parquet(data_path / "flat" / "50m" / "multi_event.parquet")

    pl.DataFrame({"item_id": ITEMS, "artist_id": [100, 101, 101]}).write_parquet(
        data_path / "artist_item_mapping.parquet"
    )
    pl.DataFrame({"item_id": ITEMS, "album_id": [200, 200, 201]}).write_parquet(
        data_path / "album_item_mapping.parquet"
    )
    pl.DataFrame(
        {
            "item_id": ITEMS,
            "normalized_embed": [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
        }
    ).write_parquet(data_path / "embeddings.parquet")

    global_config.initialize(tmp_path)
    return data_path


def _source(data_path: Path, **kwargs) -> YambdaDatasetSource:
    return YambdaDatasetSource(data_path=data_path, size="50m", **kwargs)


def _main_frame(source: YambdaDatasetSource) -> pl.DataFrame:
    return pl.read_parquet(source.artifacts.main_parquet)


def test_concurrent_sources_build_a_shared_cold_cache_once(
    yambda_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = YambdaDatasetSource._build_events_parquet
    active = 0
    maximum = 0
    builds = 0
    guard = threading.Lock()

    def build_events(source: YambdaDatasetSource, output: Path) -> None:
        nonlocal active, maximum, builds
        with guard:
            active += 1
            maximum = max(maximum, active)
            builds += 1
        try:
            time.sleep(0.05)
            original(source, output)
        finally:
            with guard:
                active -= 1

    monkeypatch.setattr(YambdaDatasetSource, "_build_events_parquet", build_events)
    sources: list[YambdaDatasetSource] = []
    errors: list[BaseException] = []

    def build() -> None:
        try:
            sources.append(_source(yambda_dir))
        except BaseException as error:
            errors.append(error)

    threads = [threading.Thread(target=build) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert all(not thread.is_alive() for thread in threads)
    assert errors == []
    assert len(sources) == 2
    assert builds == 1
    assert maximum == 1


def test_user_sample_keeps_whole_histories(yambda_dir: Path) -> None:
    frame = _main_frame(_source(yambda_dir, user_sample=UserSample(max_users=2)))

    per_user = frame.group_by("uid").agg(pl.len().alias("events"))
    assert per_user.height == 2
    assert per_user["events"].to_list() == [EVENTS_PER_USER] * 2


def test_user_sample_is_reproducible(yambda_dir: Path) -> None:
    sample = UserSample(max_users=3, seed=7)
    first = _main_frame(_source(yambda_dir, user_sample=sample, invalidate_cache=True))
    second = _main_frame(_source(yambda_dir, user_sample=sample, invalidate_cache=True))

    assert sorted(first["uid"].unique()) == sorted(second["uid"].unique())


def test_without_a_sample_every_user_is_kept(yambda_dir: Path) -> None:
    frame = _main_frame(_source(yambda_dir))

    assert sorted(frame["uid"].unique().to_list()) == USERS


def test_event_core_is_built_after_filtering_to_the_requested_action(
    yambda_dir: Path,
) -> None:
    source = _source(
        yambda_dir,
        event_type_filter="like",
        min_item_interactions_per_item=5,
    )
    frame = _main_frame(source)

    assert frame["event_type"].unique().to_list() == ["like"]
    assert frame.height == len(USERS)
    assert frame["compact_item_id"].unique().to_list() == [1]
    embeddings = pl.read_parquet(
        source.artifacts.precomputed_embeddings["compact_item_id"]
    )
    assert embeddings.height == 1


def test_thinning_listens_leaves_the_other_actions_whole(yambda_dir: Path) -> None:
    frame = _main_frame(
        _source(
            yambda_dir, user_sample=UserSample(max_users=3), listen_sample_fraction=0.01
        )
    )

    assert frame.filter(pl.col("event_type") == "like").height == 3
    assert set(frame["uid"].unique()) == set(
        frame.filter(pl.col("event_type") == "like")["uid"]
    )


def test_event_type_is_exposed_as_an_id(yambda_dir: Path) -> None:
    frame = _main_frame(_source(yambda_dir))

    by_type = dict(frame.select("event_type", "event_type_id").unique().iter_rows())
    assert by_type == {name: EVENT_TYPE_IDS[name] for name in by_type}


def test_multivalent_metadata_survives_as_lists(yambda_dir: Path) -> None:
    frame = _main_frame(_source(yambda_dir))

    artists = dict(
        frame.select("item_id", "artist_id").unique(subset="item_id").iter_rows()
    )
    assert artists[10] == [100]
    assert artists[11] == [101]


def test_targets_and_compact_ids_are_materialized(yambda_dir: Path) -> None:
    source = _source(yambda_dir)
    frame = _main_frame(source)

    assert set(source.artifacts.columns) <= set(frame.columns)
    likes = frame.filter(pl.col("event_type") == "like")
    assert likes["target_like"].to_list() == [1.0] * likes.height
    assert likes["listen_mask"].to_list() == [False] * likes.height
    assert sorted(frame["compact_item_id"].unique().to_list()) == [1, 2, 3]
