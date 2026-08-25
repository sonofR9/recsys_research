import pytest
import polars as pl
import numpy as np
import io

from ..feature_generator import FeatureGenerator
from ..data_preprocessing_pipeline import DataPreprocessingPipeline
from ..train_utils import (
    generate_features_and_negatives_local,
    generate_test_features_local,
)
from ..simple_feature_generator_polars import add_running_event_feature
from .utils import FSTR_DICT, generate_data

DAY = 86400 + 1  # to be sure about correct rounding


ITEM_ARTIST_CSV = """item_id,artist_id
10,Artist_A
20,Artist_B
"""

RAW_EVENTS_CSV = f"""uid,item_id,timestamp,event_type,is_organic,uid_likes_life,artist_cnt_life,uid_likes_7d,uid_artist_likes_7d
1,10,{0 * DAY + 0},like,true,0,0,0,0
1,40,{0 * DAY + 0},like,true,0,0,0,0
1,30,{1 * DAY + 0},like,true,2,0,2,0
1,10,{1 * DAY + 1},like,true,2,1,2,1
1,20,{1 * DAY + 2},dislike,true,2,0,2,0
1,10,{1 * DAY + 3},listen,true,2,1,2,1
2,10,{1 * DAY + 10},like,true,0,1,0,0
1,10,{2 * DAY + 0},dislike,true,4,3,4,2
2,30,{3 * DAY + 0},like,true,1,0,1,0
1,30,{3 * DAY + 0},listen,true,4,0,4,0
"""

TARGETS_CSV = f"""uid,item_id,timestamp,uid_likes_life,artist_cnt_life,uid_artist_like_ratio_7d,uid_artist_likes_7d,event_type
1,30,{4 * DAY},4,0,0,0,like
1,10,{4 * DAY},4,4,0.5,2,like
2,30,{4 * DAY},2,0,0,0,dislike
3,10,{4 * DAY},0,4,0,0,listen
"""


@pytest.fixture
def item_artist_map():
    """Consistent mapping: Item 10->A, 20->B, 30->None."""
    return pl.read_csv(io.StringIO(ITEM_ARTIST_CSV))


@pytest.fixture
def raw_event_history(item_artist_map):
    """History with high density for User 1 on Day 1."""
    events = pl.read_csv(io.StringIO(RAW_EVENTS_CSV))
    events = events.with_columns(pl.col("is_organic").cast(pl.Boolean))
    return events.join(item_artist_map, on="item_id", how="left")


@pytest.fixture
def target_df(item_artist_map):
    """Day 5 Targets: Includes a cold-start user (User 3)."""
    targets = pl.read_csv(io.StringIO(TARGETS_CSV))
    return targets.join(item_artist_map, on="item_id", how="left")


def test_e2e_same_source(raw_event_history):
    """
    Validates that the burst of events on Day 1 is handled correctly:
    1. They all see only Day 0 history.
    2. Event types (like vs listen vs dislike) are weighted correctly for features.
    """
    RAW_COLUMNS = [
        "uid",
        "item_id",
        "timestamp",
        "event_type",
        "is_organic",
        "artist_id",
    ]

    FEATURE_COLUMNS = [
        "uid_likes_life",
        "artist_cnt_life",
        "uid_likes_7d",
        "uid_artist_likes_7d",
    ]
    raw_df = raw_event_history.select(RAW_COLUMNS)
    expected = raw_event_history.select(RAW_COLUMNS + FEATURE_COLUMNS)

    fg = FeatureGenerator(
        feature_importances={
            "uid_likes_life": 1.0,
            "artist_cnt_life": 1.0,
            "uid_likes_7d": 1.0,
            "uid_artist_likes_7d": 1.0,
        },
        windows=["7d"],
        smooth_alpha=0,
        smooth_beta=0,
    )

    generated = fg.create_features(
        event_history_df=raw_df,
        target_df=raw_df,
        return_lazy=False,
    )

    joined = generated.join(
        expected,
        on=["uid", "item_id", "timestamp"],
        suffix="_expected",
    ).sort("timestamp")

    # with open("tmp_same_source.csv", "w") as f:
    #     f.write(joined.to_pandas().to_csv())

    for f in FEATURE_COLUMNS:
        diff = joined.filter(pl.col(f) != pl.col(f"{f}_expected")).select(
            ["uid", "item_id", "timestamp", f, f"{f}_expected"]
        )

        assert diff.height == 0, f"Feature {f} mismatch:\n{diff}"


def test_e2e_different_sources(raw_event_history, target_df):
    """
    Tests the target join on Day 5, ensuring 'listen' and 'dislike'
    contribute to 'cnt' but not 'likes' in the smoothing ratios.
    """
    targets_raw = target_df.select(["uid", "item_id", "timestamp", "artist_id"])
    expected = target_df

    feature_importances = {
        "uid_likes_life": 1.0,
        "artist_cnt_life": 1.0,
        "uid_artist_like_ratio_7d": 1.0,
        "uid_artist_likes_7d": 1.0,
    }

    fg = FeatureGenerator(
        feature_importances=feature_importances,
        windows=["7d"],
        smooth_alpha=0,
        smooth_beta=1,
    )

    generated = fg.create_features(
        event_history_df=raw_event_history,
        target_df=targets_raw,
        return_lazy=False,
    )

    joined = generated.join(
        expected,
        on=["uid", "item_id", "timestamp"],
        suffix="_expected",
    )

    # with open("tmp_different_sources.csv", "w") as f:
    #     f.write(
    #         joined.drop("artist_id_expected", "artist_id", "timestamp", "target")
    #         .to_pandas()
    #         .to_csv()
    #     )

    for f in feature_importances.keys():
        assert joined.filter(pl.col(f) != pl.col(f"{f}_expected")).height == 0


SEED = 42

NUM_USERS = 20
NUM_ITEMS = 50
NUM_EVENTS = 2000
NUM_DAYS = 40

HISTORY_DAYS = 38
TARGET_DAYS = 1


def run_consistency_check(train_gen_fn, test_gen_fn, feature_columns):
    """
    Generic harness to verify that features generated from a full dataset
    match features generated from split history/target datasets.
    """
    np.random.seed(SEED)

    split_ts = HISTORY_DAYS * 86400 - 1
    end_ts = (HISTORY_DAYS + TARGET_DAYS) * 86400 - 1

    df = generate_data(
        num_days=NUM_DAYS,
        num_users=NUM_USERS,
        num_items=NUM_ITEMS,
        num_events=NUM_EVENTS,
    ).sort("timestamp")

    history_df = df.filter(pl.col("timestamp") < split_ts)
    target_df = df.filter(
        (pl.col("timestamp") >= split_ts) & (pl.col("timestamp") < end_ts)
    )

    full_generated = train_gen_fn(df)
    split_generated = test_gen_fn(history_df, target_df)

    comparison_from_full = full_generated.filter(
        (pl.col("timestamp") >= split_ts) & (pl.col("timestamp") < end_ts)
    ).sort(["uid", "item_id", "timestamp"])

    comparison_from_split = split_generated.sort(["uid", "item_id", "timestamp"])

    assert comparison_from_full.height == comparison_from_split.height, (
        f"Row count mismatch! Full: {comparison_from_full.height}, Split: {comparison_from_split.height}"
    )

    # with open("tmp_cross_full.csv", "w", encoding="utf-8") as f:
    #     joined = full_generated.join(
    #         comparison_from_split,
    #         on=["uid", "item_id", "timestamp"],
    #         how="left",
    #         suffix="_expected",
    #     ).drop(
    #         "album_id_expected",
    #         "artist_id_expected",
    #         # "timestamp",
    #         "event_type_expected",
    #         "day_expected",
    #         "is_organic_expected",
    #         "target",
    #         "target_expected",
    #     )
    #     f.write(
    #         joined.select(reversed(sorted(joined.columns))).to_pandas().to_csv(sep="\t")
    #     )

    for col in feature_columns:
        joined = comparison_from_full.join(
            comparison_from_split, on=["uid", "item_id", "timestamp"], suffix="_split"
        )
        diff = joined.filter(pl.col(col) != pl.col(f"{col}_split"))

        assert diff.height == 0, (
            f"Consistency mismatch in feature '{col}' for {diff.height} rows."
        )


def test_consistency_same_vs_different_source():
    feature_columns = list(FSTR_DICT.keys())
    fg = FeatureGenerator(
        feature_importances={f: 1.0 for f in feature_columns},
        windows=["7d"],
        smooth_alpha=1,
        smooth_beta=1,
    )

    run_consistency_check(
        train_gen_fn=lambda df: fg.create_features(
            event_history_df=df, target_df=df, return_lazy=False
        ),
        test_gen_fn=lambda hist, target: fg.create_features(
            event_history_df=hist, target_df=target, return_lazy=False
        ),
        feature_columns=feature_columns,
    )


def test_consistency_same_vs_polars():
    np.random.seed(SEED)
    df = generate_data(
        num_days=NUM_DAYS,
        num_users=NUM_USERS,
        num_items=NUM_ITEMS,
        num_events=NUM_EVENTS,
    ).sort("timestamp")

    fstr_dict = {
        "uid_likes_7d": 1.0,
        "item_likes_7d": 1.0,
        "artist_likes_7d": 1.0,
        "uid_dislikes_7d": 1.0,
        "item_dislikes_7d": 1.0,
        "artist_dislikes_7d": 1.0,
    }
    feature_columns = list(fstr_dict.keys())

    pipeline = DataPreprocessingPipeline(
        random_negatives_configs=[],
        feature_windows=["7d"],
        feature_importances={f: 1.0 for f in feature_columns},
        feature_importance_threshold=0.0,
        target_for_fake=0.05,
        preprocess_dir=".tmp",
        preprocess_prefix="test_pipeline_consistency",
        use_cache=False,
        invalidate_cache=True,
        seed=42,
    )
    pipeline_df = pipeline.generate_train_features(df)

    rolling_data = df.lazy()
    for group_cols in [
        "item_id",
        "uid",
        "artist_id",
    ]:
        if not isinstance(group_cols, list):
            group_cols = [group_cols]
        rolling_data = add_running_event_feature(
            rolling_data, group_cols=group_cols, event_type="like", window_days=7
        )
        rolling_data = add_running_event_feature(
            rolling_data, group_cols=group_cols, event_type="dislike", window_days=7
        )
    rolling_data = rolling_data.collect(engine="streaming")

    assert pipeline_df.height == rolling_data.height

    with open("tmp_vs_polars.csv", "w", encoding="utf-8") as f:
        joined = pipeline_df.join(
            rolling_data,
            on=["uid", "item_id", "timestamp"],
            how="left",
            suffix="_polars",
        )
        f.write(joined.select(reversed(sorted(joined.columns))).to_pandas().to_csv())

    for col in feature_columns:
        joined = pipeline_df.join(
            rolling_data, on=["uid", "item_id", "timestamp"], suffix="_polars"
        )
        rolling_name = f"{col}_polars"
        if not col.startswith("uid_"):
            parts = col.split("_", 1)
            rolling_name = f"{parts[0]}_id_{parts[1]}"
        diff = joined.filter(pl.col(col) != pl.col(f"{rolling_name}"))

        assert diff.height == 0, (
            f"Consistency mismatch in feature '{col}' for {diff.height} rows."
        )


def test_pipeline_consistency_train_vs_test():
    feature_columns = list(FSTR_DICT.keys())
    pipeline = DataPreprocessingPipeline(
        random_negatives_configs=[],
        feature_windows=["7d"],
        feature_importances={f: 1.0 for f in feature_columns},
        feature_importance_threshold=0.0,
        target_for_fake=0.05,
        preprocess_dir=".tmp",
        preprocess_prefix="test_pipeline_consistency",
        use_cache=False,
        invalidate_cache=True,
        seed=42,
    )

    run_consistency_check(
        train_gen_fn=lambda df: pipeline.generate_train_features(df),
        test_gen_fn=lambda hist, target: pipeline.generate_test_features(
            history=hist, test=target
        ),
        feature_columns=feature_columns,
    )


def test_generate_wrappers_consistency():
    feature_columns = list(FSTR_DICT.keys())
    params = dict(
        random_negatives_configs=[],
        feature_windows=["7d"],
        feature_importances={f: 1.0 for f in feature_columns},
        feature_importance_threshold=0.0,
        target_for_fake=0.05,
        preprocess_dir=".tmp",
        preprocess_prefix="test_pipeline_consistency",
        use_cache=False,
        invalidate_cache=True,
        seed=42,
    )

    def generate_test_features_wrapper(
        history: pl.DataFrame, target_df: pl.DataFrame, **kwargs
    ):
        merged_df = pl.concat([history, target_df], how="vertical").sort(
            ["uid", "item_id", "timestamp"]
        )
        return generate_test_features_local(train_data=merged_df, **kwargs)

    run_consistency_check(
        train_gen_fn=lambda df: generate_features_and_negatives_local(
            random_negatives_configs=[],
            prefix="test_consistency_test_features_local",
            train_data=df,
            override_pipeline_params=params,
            use_cache=False,
        ),
        test_gen_fn=lambda hist, target: generate_test_features_wrapper(
            history=hist,
            target_df=target,
            random_negatives_configs=[],
            prefix="test_consistency_test_features_local",
            override_pipeline_params=params,
            train_end_day=HISTORY_DAYS,
            val_days=TARGET_DAYS,
            use_cache=False,
        ),
        feature_columns=feature_columns,
    )
