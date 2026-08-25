import hashlib
import json
import logging
import pathlib
from dataclasses import dataclass, field
from functools import cached_property
from typing import Dict, List

import numpy as np
import polars as pl
import torch

from yambda.constants import Constants
from yambda.processing import timesplit


logger = logging.getLogger(__name__)

CHECKPOINT_DIR = "./checkpoints"


@dataclass
class Data:
    train: pl.LazyFrame
    validation: pl.LazyFrame
    test: pl.LazyFrame
    pretrained_embeddings: torch.Tensor | None
    item_id_to_idx: dict[int, int]

    _train_user_ids: torch.Tensor | None = field(init=False, default=None)

    @property
    def num_items(self):
        return len(self.item_id_to_idx)

    @cached_property
    def num_train_users(self):
        return self.train.select(pl.len()).collect(engine="streaming").item()

    def train_user_ids(self, device):
        if (
            self._train_user_ids is None
            or self._train_user_ids.device != device
        ):
            self._train_user_ids = (
                self.train.select("uid")
                .collect(engine="streaming")["uid"]
                .to_torch()
                .to(device)
            )
        return self._train_user_ids


def preprocess(
    df: pl.LazyFrame,
    pretrained_embeddings: pl.LazyFrame | None,
    interaction: str,
    max_seq_len: int = 200,
    train_days: int | None = None,
    full_train: bool = False,
) -> Data:
    """
    Preprocesses raw interaction data for recommendation system modeling.

    Args:
        df: Raw input data containing user interaction sequences
        interaction: Type of interaction to process. Must be either 'likes' or 'listens'.
        val_size: Size of validation period in seconds (default: from Constants)
        max_seq_len: Maximum sequence length
        train_days: If set, train on first N days and test on day N+1 (for local validation)
        full_train: If True, use entire dataset for training (for final submission)

    Returns:
        Data: Named tuple containing:
            - train: Training data
            - validation: Validation data (None if not applicable)
            - test: Test data
            - item_id_to_idx: Mapping from original item IDs to model indices

    Note:
        - For 'listens' interactions, uses strict engagement threshold
        - Item indices start at 1 to reserve 0 for padding/masking
    """
    if interaction == "listens":
        df = df.select(
            "uid",
            pl.col("item_id", "timestamp").list.gather(
                pl.col("played_ratio_pct").list.eval(
                    pl.arg_where(
                        pl.element() >= Constants.TRACK_LISTEN_THRESHOLD
                    )
                )
            ),
        ).filter(pl.col("item_id").list.len() > 0)

    unique_item_ids = (
        df.select(pl.col("item_id").explode().unique().sort())
        .collect(engine="streaming")["item_id"]
        .to_list()
    )
    item_id_to_idx = {
        int(item_id): i for i, item_id in enumerate(unique_item_ids)
    }

    def replace_strict(df):
        return (
            df.select(
                pl.col("item_id").list.eval(
                    pl.element().replace_strict(item_id_to_idx)
                ),
                pl.all().exclude("item_id"),
            )
            .collect(engine="streaming")
            .lazy()
        )

    reindexed_embeddings = _reindex_pretrained_embeddings(
        pretrained_embeddings, item_id_to_idx
    )

    # Determine test_timestamp and val_size based on mode
    if full_train:
        # Use entire dataset for training - set test_timestamp beyond all data
        logger.info("Using full dataset for training (no time split)")
        test_timestamp = Constants.LAST_TIMESTAMP + Constants.DAY_SECONDS
        val_size = 0
        gap_size = 0
    elif train_days is not None:
        # Train on first N days, test on day N+1
        test_timestamp = (train_days + 1) * Constants.DAY_SECONDS
        val_size = Constants.DAY_SECONDS
        gap_size = 0
        logger.info(
            f"Training on first {train_days} days, testing on day {train_days + 1}"
        )
    else:
        assert False, "specify training, validation etc. days"
        # Default: use Constants.TEST_TIMESTAMP
        test_timestamp = Constants.TEST_TIMESTAMP
        gap_size = Constants.GAP_SIZE

    train, val, test = timesplit.sequential_split_train_val_test(
        df,
        val_size=val_size,
        test_timestamp=test_timestamp,
        gap_size=gap_size,
        drop_non_train_items=False,
    )

    train = train.select(
        "uid", pl.all().exclude("uid").list.slice(-max_seq_len, max_seq_len)
    )
    train = replace_strict(train)
    val = replace_strict(val)
    test = replace_strict(test)

    return Data(train, val, test, reindexed_embeddings, item_id_to_idx)


EMBEDDINGS_CACHE_DIR = pathlib.Path(CHECKPOINT_DIR) / "embeddings_cache"


def _hash_item_id_to_idx(item_id_to_idx: dict[int, int]) -> str:
    serialized = json.dumps(sorted(item_id_to_idx.items()), separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


def _reindex_pretrained_embeddings(
    pretrained_embeddings: pl.LazyFrame | None,
    item_id_to_idx: dict[int, int],
) -> torch.Tensor | None:
    if pretrained_embeddings is None:
        return None

    cache_hash = _hash_item_id_to_idx(item_id_to_idx)
    cache_path = EMBEDDINGS_CACHE_DIR / f"pretrained_{cache_hash}.pt"

    if cache_path.exists():
        logger.info(f"Loading cached pretrained embeddings from {cache_path}")
        return torch.load(cache_path, weights_only=True)

    num_items = len(item_id_to_idx)
    embedding_dim = pretrained_embeddings.select(
        pl.col("normalized_embed").list.len()
    ).first().collect(engine="streaming").item()

    idx_df = pl.DataFrame({
        "item_id": list(item_id_to_idx.keys()),
        "idx": list(item_id_to_idx.values()),
    })

    joined_df = (
        idx_df.lazy()
        .join(pretrained_embeddings, on="item_id", how="left")
        .sort("idx")
        .collect(engine="streaming")
    )

    matched = num_items - joined_df["normalized_embed"].null_count()
    reindexed_np = (
        joined_df
        .with_columns(
            pl.col("normalized_embed")
            .fill_null([0.0] * embedding_dim)
            .list.to_array(embedding_dim)
        )["normalized_embed"]
        .to_numpy()
        .astype(np.float32)
    )
    reindexed = torch.from_numpy(reindexed_np)

    logger.info(
        f"Reindexed pretrained embeddings: {matched}/{num_items} items matched, dim={embedding_dim}"
    )

    EMBEDDINGS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(reindexed, cache_path)
    logger.info(f"Cached pretrained embeddings to {cache_path}")

    return reindexed


class TrainDataset:
    def __init__(self, dataset: pl.DataFrame, num_items: int, max_seq_len: int):
        self._dataset = dataset
        self._num_items = num_items
        self._max_seq_len = max_seq_len

    @property
    def dataset(self) -> pl.DataFrame:
        return self._dataset

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> Dict[str, List[int] | int]:
        sample = self._dataset.row(index, named=True)

        item_sequence = sample["item_id"][:-1][-self._max_seq_len :]
        positive_sequence = sample["item_id"][1:][-self._max_seq_len :]
        negative_sequence = np.random.randint(
            0, self._num_items, size=(len(item_sequence),)
        ).tolist()  # FIXME: фу. Фу вдвойне: и в даталоадере (не зная об in batch/ mixed), и не исключает самих себя

        return {
            "user.ids": [sample["uid"]],
            "user.length": 1,
            "item.ids": item_sequence,
            "item.length": len(item_sequence),
            "positive.ids": positive_sequence,
            "positive.length": len(positive_sequence),
            "negative.ids": negative_sequence,
            "negative.length": len(negative_sequence),
        }


class EvalDataset:
    def __init__(self, dataset: pl.DataFrame, max_seq_len: int):
        self._dataset = dataset
        self._max_seq_len = max_seq_len

    @property
    def dataset(self) -> pl.DataFrame:
        return self._dataset

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> Dict[str, List[int] | int]:
        sample = self._dataset.row(index, named=True)

        item_sequence = sample["item_id_train"][-self._max_seq_len :]
        next_items = sample["item_id_valid"]

        return {
            "user.ids": [sample["uid"]],
            "user.length": 1,
            "item.ids": item_sequence,
            "item.length": len(item_sequence),
            "labels.ids": next_items,
            "labels.length": len(next_items),
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    processed_batch = {}
    for key in batch[0].keys():
        if key.endswith(".ids"):
            prefix = key.split(".")[0]
            assert "{}.length".format(prefix) in batch[0]

            processed_batch[f"{prefix}.ids"] = []
            processed_batch[f"{prefix}.length"] = []

            for sample in batch:
                processed_batch[f"{prefix}.ids"].extend(sample[f"{prefix}.ids"])
                processed_batch[f"{prefix}.length"].append(
                    sample[f"{prefix}.length"]
                )

    for part, values in processed_batch.items():
        processed_batch[part] = torch.tensor(values, dtype=torch.long)

    return processed_batch


def infer_users(eval_dataloader, model: torch.nn.Module, device: str):
    user_ids = []
    user_embeddings = []

    model.eval()
    for batch in eval_dataloader:
        for key in batch.keys():
            batch[key] = batch[key].to(device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            user_ids.append(batch["user.ids"])
            user_embeddings.append(model(batch))

    return torch.cat(user_ids, dim=0), torch.cat(user_embeddings, dim=0).float()


def infer_items(model):
    orig = model._orig_mod if hasattr(model, "_orig_mod") else model
    return orig.get_all_item_embeddings()
