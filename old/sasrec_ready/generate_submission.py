#!/usr/bin/env python3
"""
Generate submission file for SASRec model.

This script:
1. Loads the full dataset (likes) to build item2ind mapping
2. Loads test users from Kaggle
3. Builds user history from the full dataset using Polars (memory efficient)
4. Loads trained SASRec model
5. Generates top-100 recommendations for each test user
6. Outputs submission in required format: uid,item_ids
"""

import argparse
from pathlib import Path

import kagglehub
import polars as pl
import torch
from kagglehub import KaggleDatasetAdapter
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

# Import local modules
from sasrec import SASRec

MAX_SEQ_LEN = 265
SPECIAL_TOKENS = ["PAD", "BOS", "UNK"]


class TestDataset(Dataset):
    """Dataset for test users inference using Polars DataFrame."""

    def __init__(self, user_histories_df: pl.DataFrame):
        """
        Args:
            user_histories_df: Polars DataFrame with columns:
                - uid: user id
                - item_inds: list of item indices (already converted, chronologically sorted)
        """
        self.user_histories_df = user_histories_df

    def __len__(self):
        return len(self.user_histories_df)

    def __getitem__(self, idx):
        row = self.user_histories_df.row(idx, named=True)
        uid = row["uid"]
        history_inds = row["item_inds"]

        # Keep only last MAX_SEQ_LEN - 1 items (to leave room for BOS)
        if len(history_inds) > MAX_SEQ_LEN - 1:
            history_inds = history_inds[-(MAX_SEQ_LEN - 1) :]

        # Convert to tensor
        history = torch.tensor(history_inds, dtype=torch.long)

        return {"uid": uid, "history": history}


def collate_fn(batch):
    """Collate function for test dataloader."""
    uids = [x["uid"] for x in batch]
    histories = [x["history"] for x in batch]

    # Pad histories
    histories_padded = pad_sequence(
        histories, batch_first=True, padding_value=0
    )

    return {"uids": uids, "history": histories_padded}


def load_likes_data():
    """Load likes data from Kaggle."""
    print("Loading likes data from Kaggle...")
    file_path = "likes.parquet"
    data = kagglehub.dataset_load(
        KaggleDatasetAdapter.POLARS,
        "thekabeton/ysda-recsys-2026-yambda-dataset/versions/3",
        file_path,
    ).collect()
    print(f"Loaded {len(data)} likes")
    return data


def load_test_users():
    """Load test users from Kaggle."""
    print("Loading test users from Kaggle...")
    file_path = "test_users.csv"
    test_users = kagglehub.dataset_load(
        KaggleDatasetAdapter.POLARS,
        "thekabeton/ysda-recsys-2026-yambda-dataset/versions/3",
        file_path,
    ).collect()
    print(f"Loaded {len(test_users)} test users")
    return test_users


def build_item2ind(data: pl.DataFrame) -> dict:
    """Build item2ind mapping from data."""
    print("Building item2ind mapping...")
    val_size = 60 * 60 * 24
    gap_size = 60 * 30
    item2ind = dict(
        data.filter(
            pl.col("timestamp") < data["timestamp"].max() - val_size - gap_size
        )
        .select("item_id")
        .unique()
        .sort("item_id")
        .with_row_index("ind", offset=len(SPECIAL_TOKENS))
        .select(["item_id", "ind"])
        .iter_rows()
    )
    print(f"Built mapping for {len(item2ind)} items")
    return item2ind


def build_ind2item(item2ind: dict) -> dict:
    """Build reverse mapping from item index to item_id."""
    return {v: k for k, v in item2ind.items()}


def build_user_histories_df(
    data: pl.DataFrame, item2ind: dict, test_uids: pl.DataFrame
) -> pl.DataFrame:
    """Build user histories from likes data using Polars (memory efficient).

    Args:
        data: Full likes DataFrame
        item2ind: Dict mapping item_id -> item_ind
        test_uids: DataFrame with test user IDs

    Returns:
        DataFrame with columns: uid, item_inds (list of indices)
    """
    print("Building user histories with Polars...")

    # Filter to only test users first to reduce memory
    data_filtered = data.join(test_uids.select("uid"), on="uid", how="semi")
    print(f"Filtered to {len(data_filtered)} likes for test users")

    # Sort by timestamp and group by user
    user_histories_df = (
        data_filtered.sort(["uid", "timestamp"])
        .group_by("uid")
        .agg(pl.col("item_id").alias("item_ids"))
    )

    # Remove duplicates while maintaining order
    user_histories_df = user_histories_df.with_columns(
        pl.col("item_ids").list.unique(maintain_order=True).alias("item_ids")
    )

    # Convert item_ids to item_inds using replace_strict
    # UNK index is 2 for unknown items
    user_histories_df = user_histories_df.with_columns(
        pl.col("item_ids")
        .list.eval(pl.element().replace_strict(item2ind, default=2))
        .alias("item_inds")
    ).drop("item_ids")

    # Ensure all test users are present (even those without history)
    user_histories_df = (
        test_uids.select("uid")
        .join(user_histories_df, on="uid", how="left")
        .with_columns(pl.col("item_inds").fill_null([]))
    )

    print(f"Built histories for {len(user_histories_df)} test users")
    return user_histories_df


def load_model(checkpoint_path: str, item2ind: dict, device: str) -> SASRec:
    """Load trained SASRec model from checkpoint."""
    print(f"Loading model from {checkpoint_path}...")

    model = SASRec(item2ind, SPECIAL_TOKENS)
    model.to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        if "metric" in checkpoint and checkpoint["metric"] is not None:
            print(f"Model metric from checkpoint: {checkpoint['metric']:.4f}")
        if "epoch" in checkpoint:
            print(f"Model epoch: {checkpoint['epoch']}")
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print("Model loaded successfully")
    return model


@torch.no_grad()
def generate_predictions(
    model: SASRec,
    dataloader: DataLoader,
    ind2item: dict,
    device: str,
    top_k: int = 100,
) -> dict:
    """Generate top-k predictions for all test users."""
    print(f"Generating top-{top_k} predictions...")

    predictions = {}

    for batch in tqdm(dataloader, desc="Generating predictions"):
        uids = batch["uids"]
        history = batch["history"].to(device, non_blocking=True)

        # Get top-k item indices
        top_k_indices = model.top_k(history, top_k)

        # Convert indices to item_ids
        for i, uid in enumerate(uids):
            item_indices = top_k_indices[i].cpu().tolist()
            # Convert indices to item_ids, skip special tokens
            item_ids = []
            for idx in item_indices:
                if idx in ind2item:
                    item_ids.append(ind2item[idx])
            predictions[uid] = item_ids

    print(f"Generated predictions for {len(predictions)} users")
    return predictions


def save_submission(predictions: dict, output_path: str):
    """Save predictions to submission file."""
    print(f"Saving submission to {output_path}...")

    with open(output_path, "w") as f:
        f.write("uid,item_ids\n")
        for uid, item_ids in predictions.items():
            item_ids_str = " ".join(map(str, item_ids))
            f.write(f"{uid},{item_ids_str}\n")

    print(f"Submission saved with {len(predictions)} users")


def main():
    parser = argparse.ArgumentParser(description="Generate SASRec submission")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/baseline_best.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="submission.csv",
        help="Output submission file path",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for inference",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Number of recommendations per user",
    )
    args = parser.parse_args()

    print(f"Using device: {args.device}")

    # Load data
    likes_data = load_likes_data()
    test_users = load_test_users()

    # Build mappings
    item2ind = build_item2ind(likes_data)
    ind2item = build_ind2item(item2ind)

    # Build user histories DataFrame (memory efficient)
    user_histories_df = build_user_histories_df(
        likes_data, item2ind, test_users
    )

    # Check how many test users have history
    users_with_history = user_histories_df.filter(
        pl.col("item_inds").list.len() > 0
    ).height
    print(f"Test users with history: {users_with_history}/{len(test_users)}")

    # Create dataset and dataloader
    test_dataset = TestDataset(user_histories_df)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    # Load model
    model = load_model(args.checkpoint, item2ind, args.device)

    # Generate predictions
    predictions = generate_predictions(
        model, test_dataloader, ind2item, args.device, args.top_k
    )

    # Save submission
    save_submission(predictions, args.output)

    print("Done!")


if __name__ == "__main__":
    main()
