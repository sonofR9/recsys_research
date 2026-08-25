#!/usr/bin/env python3

import logging

import click
import kagglehub
import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader

from yambda.evaluation.metrics import calc_metrics
from yambda.evaluation.ranking import Embeddings, Targets, rank_items

from .data import (
    CHECKPOINT_DIR,
    EvalDataset,
    collate_fn,
    infer_items,
    infer_users,
)
from .utils import (
    common_options,
    load_data,
    prepare_eval_df,
    setup_environment,
)


logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] [%(levelname)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_test_users_from_kagglehub() -> pl.DataFrame:
    logger.debug("Loading test users from kagglehub...")
    file_path = "test_users.csv"
    test_users = kagglehub.dataset_load(
        kagglehub.KaggleDatasetAdapter.POLARS,
        "thekabeton/ysda-recsys-2026-yambda-dataset/versions/3",
        file_path,
    ).collect()

    logger.debug(f"Loaded {len(test_users)} test users from kagglehub")
    return test_users


def load_test_users(
    use_kagglehub: bool, test_users_path: str | None
) -> pl.DataFrame | None:
    if use_kagglehub:
        return load_test_users_from_kagglehub()
    elif test_users_path is not None:
        logger.debug(f"Loading test users from {test_users_path}...")
        return pl.read_parquet(test_users_path)
    return None


def filter_eval_df_by_test_users(
    eval_df: pl.DataFrame, test_users_df: pl.DataFrame
) -> pl.DataFrame:
    return eval_df.join(test_users_df.select("uid"), on="uid", how="inner")


def get_most_popular_items_last_day(
    train_df: pl.DataFrame, max_seq_len: int
) -> list[int]:
    from yambda.constants import Constants

    max_timestamp = train_df.select(pl.col("timestamp").list.max()).max().item()
    last_day_start = max_timestamp - Constants.DAY_SECONDS

    last_day_items = (
        train_df.select(
            pl.col("item_id").list.gather(
                pl.col("timestamp").list.eval(
                    pl.arg_where(pl.element() >= last_day_start)
                )
            )
        )
        .select(pl.col("item_id").explode())
        .filter(pl.col("item_id").is_not_null())
    )

    return (
        last_day_items.group_by("item_id")
        .len()
        .sort("len", descending=True)
        .head(max_seq_len)["item_id"]
        .to_list()
    )


def prepare_eval_df_from_test_users(
    train_df: pl.DataFrame, test_users_df: pl.DataFrame
) -> pl.DataFrame:
    eval_df = train_df.join(
        test_users_df.select("uid"), on="uid", how="inner"
    ).select(pl.col("uid"), pl.col("item_id").alias("item_id_train"))
    return eval_df.with_columns(pl.col("item_id_train").alias("item_id_valid"))


def get_users_without_history(
    train_df: pl.DataFrame, test_users_df: pl.DataFrame
) -> pl.DataFrame:
    return test_users_df.select("uid").join(
        train_df.select("uid"), on="uid", how="anti"
    )


def rank_users(
    model: torch.nn.Module,
    eval_df: pl.DataFrame,
    max_seq_len: int,
    batch_size: int,
    num_candidates: int,
    device: str,
):
    eval_dataset = EvalDataset(dataset=eval_df, max_seq_len=max_seq_len)
    eval_dataloader = DataLoader(
        dataset=eval_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        drop_last=False,
        shuffle=False,
    )

    model.eval()
    with torch.inference_mode():
        user_ids, user_embeddings = infer_users(
            eval_dataloader=eval_dataloader, model=model, device=device
        )
        item_embeddings = infer_items(model=model)

    item_embeddings_obj = Embeddings(
        ids=torch.arange(start=0, end=item_embeddings.shape[0], device=device),
        embeddings=item_embeddings,
    )
    user_embeddings_obj = Embeddings(ids=user_ids, embeddings=user_embeddings)

    with torch.inference_mode():
        return rank_items(
            users=user_embeddings_obj,
            items=item_embeddings_obj,
            num_items=num_candidates,
        )


def evaluate_model(
    model: torch.nn.Module,
    eval_df: pl.DataFrame,
    max_seq_len: int,
    batch_size: int,
    num_candidates: int,
    device: str,
):
    ranked = rank_users(
        model, eval_df, max_seq_len, batch_size, num_candidates, device
    )

    df_user_ids = torch.tensor(
        eval_df["uid"].to_list(), dtype=torch.long, device=device
    )
    df_target_ids = [
        torch.tensor(item_ids, dtype=torch.long, device=device)
        for item_ids in eval_df["item_id_valid"].to_list()
    ]
    targets = Targets(user_ids=df_user_ids, item_ids=df_target_ids)

    metric_names = [
        f"{name}@{k}"
        for name in ["recall", "ndcg", "coverage"]
        for k in [10, 50, 100]
    ]
    return calc_metrics(ranked, targets, metrics=metric_names)


def generate_candidates(ranked, idx_to_item_id: dict[int, int]) -> pl.DataFrame:
    user_ids_np = ranked.user_ids.cpu().numpy()
    item_ids_np = ranked.item_ids.cpu().numpy()

    idx_to_item_arr = np.array(
        [idx_to_item_id.get(i, 0) for i in range(item_ids_np.max() + 1)]
    )
    candidate_item_ids = idx_to_item_arr[item_ids_np]

    item_ids_str = [" ".join(map(str, row)) for row in candidate_item_ids]

    return pl.DataFrame(
        {
            "uid": user_ids_np.tolist(),
            "item_ids": item_ids_str,
        }
    )


@click.command()
@click.option("--exp_name", required=True, type=str)
@common_options
@click.option(
    "--num_candidates",
    required=False,
    type=int,
    default=100,
    show_default=True,
)
@click.option(
    "--use_kagglehub",
    is_flag=True,
    default=False,
    help="Load test users from kagglehub",
)
@click.option(
    "--test_users_path",
    required=False,
    type=str,
    default=None,
    help="Path to test_users.parquet",
)
@click.option(
    "--output_path",
    required=False,
    type=str,
    default=None,
    help="Path to save submission CSV (enables candidate generation mode)",
)
@click.option(
    "--checkpoint_path",
    required=False,
    type=str,
    default=None,
    help="Path to checkpoint file (overrides exp_name-based path)",
)
def main(
    exp_name: str,
    data_dir: str,
    size: str,
    batch_size: int,
    max_seq_len: int,
    num_candidates: int,
    train_days: int | None,
    full_train: bool,
    use_kagglehub: bool,
    test_users_path: str | None,
    output_path: str | None,
    checkpoint_path: str | None,
):
    seed = 42
    device = "cuda"
    interaction = "likes"
    generate_candidates_mode = output_path is not None

    if (
        generate_candidates_mode
        and not use_kagglehub
        and test_users_path is None
    ):
        raise click.UsageError(
            "For candidate generation, provide --use_kagglehub or --test_users_path"
        )

    setup_environment(seed)

    logger.debug("Preprocessing data...")
    data, train_df = load_data(
        data_dir, size, interaction, max_seq_len, train_days, full_train
    )
    logger.debug(
        f"Preprocessing data has finished! Training users: {len(train_df)}"
    )

    logger.debug("Preparing validation data...")
    val_df = data.validation.collect(engine="streaming")
    logger.debug(f"Validation users: {len(val_df)}")

    eval_df = prepare_eval_df(train_df, val_df)
    users_without_history = None

    test_users_df = load_test_users(use_kagglehub, test_users_path)
    if test_users_df is not None:
        eval_df = prepare_eval_df_from_test_users(train_df, test_users_df)
        users_without_history = get_users_without_history(
            train_df, test_users_df
        )
        logger.debug(
            f"Users with history: {len(eval_df)}, users without history: {len(users_without_history)}"
        )

    logger.debug(f"Will evaluate on {len(eval_df)} users")

    if len(eval_df) == 0:
        raise ValueError("No matching users found")

    logger.debug(f"Evaluating on {len(eval_df)} users")

    ckpt_path = (
        checkpoint_path
        if checkpoint_path
        else f"{CHECKPOINT_DIR}/{exp_name}_best_state.pth"
    )
    model = torch.load(ckpt_path, weights_only=False).to(device)
    model.eval()

    if generate_candidates_mode:
        ranked = rank_users(
            model, eval_df, max_seq_len, batch_size, num_candidates, device
        )

        logger.debug("Converting item indices back to original IDs...")
        idx_to_item_id = {
            idx: item_id for item_id, idx in data.item_id_to_idx.items()
        }
        output_df = generate_candidates(ranked, idx_to_item_id)

        if users_without_history is not None and len(users_without_history) > 0:
            fallback_items = get_most_popular_items_last_day(
                train_df, num_candidates
            )
            fallback_item_ids = [
                idx_to_item_id.get(idx, 0) for idx in fallback_items
            ]
            fallback_str = " ".join(
                map(str, fallback_item_ids[:num_candidates])
            )

            fallback_df = pl.DataFrame(
                {
                    "uid": users_without_history["uid"].to_list(),
                    "item_ids": [fallback_str] * len(users_without_history),
                }
            )
            output_df = pl.concat([output_df, fallback_df])
            logger.debug(
                f"Added {len(users_without_history)} users with fallback items"
            )

        output_df.write_csv(output_path)

        logger.info(f"\n{'=' * 60}")
        logger.info(f"Candidate Generation Summary:")
        logger.info(f"{'=' * 60}")
        logger.info(f"Total test users processed: {len(output_df)}")
        logger.info(f"Candidates per user: {num_candidates}")
        logger.info(f"Output saved to: {output_path}")
        logger.info(f"{'=' * 60}\n")
    else:
        metrics = evaluate_model(
            model=model,
            eval_df=eval_df,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            num_candidates=num_candidates,
            device=device,
        )

        logger.info(f"\n{'=' * 60}")
        logger.info(f"Evaluation Results:")
        logger.info(f"{'=' * 60}")
        for metric_name, metric_values in metrics.items():
            for k, value in metric_values.items():
                logger.info(f"eval/{metric_name}@{k} {value:.4f}")
        logger.info(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
