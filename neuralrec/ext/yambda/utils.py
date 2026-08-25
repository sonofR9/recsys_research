import random
from typing import Literal

import polars as pl

from neuralrec.data.dataset import Dataset
from neuralrec.ext.yambda.huggingface import YambdaDataset
from neuralrec.ext.yambda.transforms import ItemIdLast, RemapItemIds
from neuralrec.data.transforms import ToTorch


def make_item_id_mapping(
    df: pl.DataFrame, item_ids_column: str = "item_id"
) -> dict[int, int]:
    unique = (
        df.select(pl.col(item_ids_column).list.explode())
        .unique()
        .to_series()
        .sort()
        .to_list()
    )
    return {int(old_id): new_id for new_id, old_id in enumerate(unique, start=1)}


def load_yambda_user_split(
    valid_ratio: float = 0.2,
    dataset_size: Literal["50m", "500m", "5b"] = "50m",
    seed: int = 0,
    num_train_samples: int = 500,
    num_valid_samples: int = 100,
):
    yambda = YambdaDataset(
        dataset_type="sequential",
        dataset_size=dataset_size,
    )
    listens = yambda.interaction("listens")
    table = getattr(listens.data, "table", listens.data)
    df = pl.from_arrow(table)

    item_id_mapping = make_item_id_mapping(df)
    max_item_id = max(item_id_mapping.values())

    users = df["uid"].unique().to_list()
    random.seed(seed)
    random.shuffle(users)

    n_valid = int(len(users) * valid_ratio)
    valid_users = frozenset(users[:n_valid])
    train_users = frozenset(users[n_valid:])

    df_train = df.filter(pl.col("uid").is_in(train_users))
    df_valid = df.filter(pl.col("uid").is_in(valid_users))

    transform = [ItemIdLast(max_len=128), RemapItemIds(item_id_mapping), ToTorch()]
    return (
        Dataset(df_train, transform=transform).take(num_train_samples),
        Dataset(df_valid, transform=transform).take(num_valid_samples),
        max_item_id,
    )
