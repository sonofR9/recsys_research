from copy import deepcopy
from pathlib import Path

import polars as pl
from catboost import CatBoostClassifier, CatBoostRanker, Pool

from data.counters import EmaCounter
from utils.global_config import config as global_config

from .data_preprocessing_pipeline import (
    DataPreprocessingPipeline,
    RandomNegativesConfig,
)
from data.utils import log_memory

TO_DROP = [
    "target",
    "item_id",
    "uid",
    "event_type",
    "timestamp",
    "day",
    "is_organic",
]
SECONDS_IN_DAY = 60 * 60 * 24

METRICS = [
    "AUC",
    "CrossEntropy",
    "Precision",
    "Recall",
    "Accuracy",
    "QueryAUC",
    "RecallAt",
    "LogLikelihoodOfPrediction",
]

MAX_GROUP_SIZE = 1023


def train_and_test_catboost(
    df: pl.DataFrame,
    exclude_generated_in_eval: bool = False,
    catboost_model_dir: str | None = None,
    end_day: int = 299,
    val_days: int = 1,
) -> CatBoostClassifier:
    to_drop = [feature for feature in TO_DROP if feature in df.columns]

    df = df.sort("uid")
    train = df.filter(pl.col("timestamp") < SECONDS_IN_DAY * end_day)
    train = train.group_by("uid").head(MAX_GROUP_SIZE)

    val = df.filter(
        (pl.col("timestamp") >= SECONDS_IN_DAY * end_day)
        & (pl.col("timestamp") < SECONDS_IN_DAY * (end_day + val_days))
    )
    if exclude_generated_in_eval:
        val = val.filter(pl.col("event_type").is_in(["like", "dislike"]))
    val = val.group_by("uid").head(MAX_GROUP_SIZE)

    train_pool = Pool(
        data=train.drop(to_drop),
        label=train["target"],
        group_id=train["uid"],
    )

    val_pool = Pool(
        data=val.drop(to_drop),
        label=val["target"],
        group_id=val["uid"],
    )

    model = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.1,
        depth=4,
        l2_leaf_reg=10,
        loss_function="CrossEntropy",
        # eval_metric="AUC",
        eval_metric="CrossEntropy",
        custom_metric=METRICS,
        early_stopping_rounds=100,
        verbose=10,
        use_best_model=True,
        task_type="GPU",
        metric_period=10,
    )
    # model = CatBoostRanker(
    #     iterations=2000,
    #     learning_rate=0.1,
    #     depth=6,
    #     l2_leaf_reg=10,
    #     loss_function="YetiRank:mode=NDCG",
    #     eval_metric="CrossEntropy",
    #     custom_metric=METRICS,
    #     early_stopping_rounds=200,
    #     verbose=100,
    #     use_best_model=True,
    #     task_type="GPU",
    #     metric_period=10,
    # )

    model.fit(train_pool, eval_set=val_pool, plot=True)

    if catboost_model_dir is not None:
        model.save_model(catboost_model_dir)

    # metrics = model.eval_metrics(
    #     val_pool, metrics=["AUC", "Logloss", "Accuracy"], plot=True
    # )

    # print fstr
    fstr = model.get_feature_importance(type="PredictionValuesChange")
    fstr_pairs = sorted(
        zip(model.feature_names_, fstr), key=lambda x: x[1], reverse=True
    )

    print("fstr")
    for name, val in fstr_pairs:
        print(f"{name}: {val}")

    return model


def check_on_dataset(
    df: pl.DataFrame,
    model: CatBoostClassifier | CatBoostRanker,
    end_day: int = 299,
    val_days: int = 1,
    eval_generated: bool = True,
    eval_no_generated: bool = True,
):
    to_drop = [feature for feature in TO_DROP if feature in df.columns]

    df = df.sort("uid")
    only_val_days = df.filter(
        (pl.col("timestamp") >= SECONDS_IN_DAY * end_day)
        & (pl.col("timestamp") < SECONDS_IN_DAY * (end_day + val_days))
    )
    all_events = only_val_days

    no_generated = only_val_days.filter(pl.col("event_type").is_in(["like", "dislike"]))

    all_pool = Pool(
        data=all_events.drop(to_drop),
        label=all_events["target"].to_list(),
        group_id=all_events["uid"],
    )

    no_generated_pool = Pool(
        data=no_generated.drop(to_drop),
        label=no_generated["target"].to_list(),
        group_id=no_generated["uid"],
    )

    if eval_generated:
        print("with generated")
        _ = model.eval_metrics(
            all_pool,
            metrics=METRICS,
            plot=True,
        )

    if eval_no_generated:
        print("no generated")
        _ = model.eval_metrics(
            no_generated_pool,
            metrics=METRICS,
            plot=True,
        )


def check_on_sampled(
    df: pl.DataFrame,
    model: CatBoostClassifier | CatBoostRanker,
    end_day: int = 299,
    val_days: int = 1,
):
    df = df.sort("uid")
    to_drop = [feature for feature in TO_DROP if feature in df.columns]

    train = df.filter(pl.col("timestamp") < SECONDS_IN_DAY * end_day)
    train = train.filter(pl.col("event_type").is_in(["random_negative"]))

    val = df.filter(
        (pl.col("timestamp") >= SECONDS_IN_DAY * end_day)
        & (pl.col("timestamp") < SECONDS_IN_DAY * (end_day + val_days))
    )
    val = val.filter(pl.col("event_type").is_in(["random_negative"]))

    train_pool = Pool(
        data=train.drop(to_drop),
        label=train["target"].to_list(),
        group_id=train["uid"],
    )

    val_pool = Pool(
        data=val.drop(to_drop),
        label=val["target"].to_list(),
        group_id=val["uid"],
    )

    print("random_negative on train pool")
    _ = model.eval_metrics(
        train_pool, metrics=["AUC", "Logloss", "Accuracy"], plot=True
    )

    print("random_negative on val pool")
    _ = model.eval_metrics(val_pool, metrics=["AUC", "Logloss", "Accuracy"], plot=True)


def generate_features_and_negatives_local(
    random_negatives_configs: list[RandomNegativesConfig],
    prefix: str,
    train_data: pl.DataFrame | str | Path,
    counters: list[EmaCounter] | None = None,
    override_pipeline_params: dict = {},
    use_cache: bool = False,
) -> pl.DataFrame:
    common_pipeline_params = dict(
        counters=counters or [],
        random_negatives_configs=random_negatives_configs,
        use_cache=use_cache,
        invalidate_cache=(not use_cache),
        preprocess_dir=global_config.preprocessed_path,
        seed=42,
        feature_importances=None,
        feature_importance_threshold=0.0,
    )
    common_pipeline_params.update(override_pipeline_params)

    params = deepcopy(common_pipeline_params)
    params["preprocess_prefix"] = prefix

    data_preprocessing = DataPreprocessingPipeline(**params)

    return data_preprocessing.generate_train_features(train=train_data)


def generate_test_features_local(
    random_negatives_configs: list[RandomNegativesConfig],
    prefix: str,
    train_data: pl.DataFrame | str | Path,
    counters: list[EmaCounter] | None = None,
    override_pipeline_params: dict = {},
    train_end_day: int = 299,
    val_days: int = 1,
    use_cache: bool = False,
    target_for_fake: float = 0.05,
) -> pl.DataFrame:
    log_memory("generate_test_features_local: start")
    print(train_data.shape)

    train_truncated = train_data.filter(
        (pl.col("timestamp") < SECONDS_IN_DAY * train_end_day)
    )
    log_memory("generate_test_features_local: train truncated")
    val_data = train_data.filter(
        (pl.col("timestamp") >= SECONDS_IN_DAY * train_end_day)
        & (pl.col("timestamp") < SECONDS_IN_DAY * (train_end_day + val_days))
    )
    print("unique in test")
    print(val_data.shape)
    log_memory("val data")

    common_pipeline_params = dict(
        counters=counters or [],
        random_negatives_configs=random_negatives_configs,
        use_cache=use_cache,
        invalidate_cache=(not use_cache),
        preprocess_dir=global_config.preprocessed_path,
        seed=42,
        feature_importances=None,
        feature_importance_threshold=0.0,
    )
    common_pipeline_params.update(override_pipeline_params)

    params = deepcopy(common_pipeline_params)
    params["preprocess_prefix"] = prefix

    data_preprocessing = DataPreprocessingPipeline(**params)

    result = data_preprocessing.generate_test_features(
        history=train_truncated, test=val_data
    )
    print("stats before join:")
    print(result.shape)
    result = result.join(
        val_data, on=["uid", "item_id", "timestamp"], how="left"
    ).with_columns(
        pl.when(pl.col("event_type").eq("like"))
        .then(1)
        .when(pl.col("event_type").eq("dislike"))
        .then(0)
        .otherwise(target_for_fake)
        .alias("target")
    )
    right_extra_columns = [
        column for column in result.columns if column.endswith("_right")
    ]
    result = result.drop(right_extra_columns)
    print("stats after join:")
    print(result.shape)
    log_memory("generate_test_features_local: generated test features")

    return result
