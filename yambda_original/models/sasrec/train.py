#!/usr/bin/env python3

import logging
import math
import os

import click
import polars as pl
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import CHECKPOINT_DIR, TrainDataset, collate_fn
from .eval import evaluate_model
from .ml_logger import MlLogger
from .model import SASRecEncoder
from .utils import (
    TopKCheckpoints,
    common_options,
    load_data,
    prepare_eval_df,
    setup_environment,
)


def create_cosine_scheduler_with_warmup(
    optimizer,
    total_steps: int,
    warmup_ratio: float,
    min_lr_ratio: float = 0.01,
    **kwargs,
):
    warmup_steps = int(warmup_ratio * total_steps)

    def lr_lambda(step: int):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (
                1 + math.cos(math.pi * progress)
            )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] [%(levelname)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def train(
    train_dataloader: DataLoader,
    model: SASRecEncoder,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    writer: MlLogger,
    checkpoints: TopKCheckpoints | object,
    device: str = "cuda:0",
    num_epochs: int = 100,
    eval_df: pl.DataFrame | None = None,
    max_seq_len: int = 200,
    batch_size: int = 256,
    num_candidates: int = 100,
    **kwargs,
) -> dict | None:
    logger.debug("Start training...")

    global_step = 0
    final_metrics = None
    for epoch_num in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        train_bar = tqdm(
            train_dataloader, desc=f"Epoch {epoch_num + 1}/{num_epochs}"
        )
        for batch in train_bar:
            for key in batch.keys():
                batch[key] = batch[key].to(device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_value = loss.float().item()
            epoch_loss += loss_value
            num_batches += 1
            global_step += 1
            if global_step % 100 == 0:
                writer.log({"train_step/loss": loss_value}, global_step)
                writer.log(
                    {"train_step/lr": optimizer.param_groups[0]["lr"]},
                    global_step,
                )
            scheduler.step()

            if global_step % 100 == 0:
                train_bar.set_description(
                    f"Epoch {epoch_num + 1}/{num_epochs} loss: {loss_value:.4f}"
                )

        avg_epoch_loss = epoch_loss / num_batches
        writer.log(
            {
                "train_epoch/epoch": epoch_num + 1,
                "train_epoch/loss": avg_epoch_loss,
            },
            global_step,
        )
        logger.debug(
            f"Epoch {epoch_num + 1} average loss: {avg_epoch_loss:.4f}"
        )

        if eval_df is not None:
            metrics = evaluate_model(
                model=model,
                eval_df=eval_df,
                max_seq_len=max_seq_len,
                batch_size=batch_size,
                num_candidates=num_candidates,
                device=device,
            )
            for metric_name, metric_values in metrics.items():
                for k, value in metric_values.items():
                    writer.log(
                        {f"eval_epoch/{metric_name}@{k}": value}, global_step
                    )
            logger.debug(f"Epoch {epoch_num + 1} metrics: {metrics}")

            recall_100 = metrics["recall"][100]
            checkpoints.update(recall_100, epoch_num + 1, model.state_dict())

            final_metrics = metrics

    logger.debug("Training procedure has been finished!")

    return final_metrics


def create_training_environment(
    data_dir: str,
    size: str,
    max_seq_len: int,
    train_days: int | None,
    full_train: bool,
    validate: bool,
    batch_size: int,
    seed: int,
    interaction: str = "likes",
    **kwargs,
):
    """Create and return the training environment (data, dataloader, eval_df)."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    if validate and train_days is None:
        raise click.UsageError("--validate requires --train_days to be set")

    if full_train and train_days is not None:
        raise click.UsageError("Cannot use both --full_train and --train_days")

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
    logger.debug(f"Will evaluate on {len(eval_df)} users after each epoch")

    if not validate or len(eval_df) == 0:
        eval_df = None

    train_dataset = TrainDataset(
        dataset=train_df, num_items=data.num_items, max_seq_len=max_seq_len
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        drop_last=True,
        shuffle=True,
        num_workers=5,
        prefetch_factor=10,
        pin_memory_device="cuda",
        pin_memory=True,
    )

    return data, train_dataloader, eval_df


def create_model_and_optimizer(
    data,
    max_seq_len: int,
    embedding_dim: int,
    num_heads: int,
    num_layers: int,
    dropout: float,
    use_bos_tokens: bool,
    use_alibi: bool,
    use_positional_embedding: bool,
    use_swiglu: bool,
    num_kv_heads: int | None,
    output_projection: str,
    learning_rate: float,
    num_epochs: int,
    train_dataloader: DataLoader,
    warmup_ratio: float,
    min_lr_ratio: float,
    device: str = "cuda",
    **kwargs,
) -> tuple[
    SASRecEncoder, torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR
]:
    """Create and return a SASRecEncoder model, optimizer, and scheduler with the given configuration."""
    model = SASRecEncoder(
        num_items=data.num_items,
        max_sequence_length=max_seq_len,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        use_bos_tokens=use_bos_tokens,
        use_alibi=use_alibi,
        use_positional_embedding=use_positional_embedding,
        use_swiglu=use_swiglu,
        num_kv_heads=num_kv_heads,
        pretrained_embeddings=data.pretrained_embeddings,
        output_projection=output_projection,
    ).to(device)

    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, fused=True
    )

    # Create scheduler
    steps_per_epoch = len(train_dataloader)
    total_steps = num_epochs * steps_per_epoch

    scheduler = create_cosine_scheduler_with_warmup(
        optimizer=optimizer,
        total_steps=total_steps,
        warmup_ratio=warmup_ratio,
        min_lr_ratio=min_lr_ratio,
    )

    return model, optimizer, scheduler


@click.command()
@click.option("--exp_name", required=True, type=str)
@common_options
@click.option(
    "--embedding_dim", required=False, type=int, default=64, show_default=True
)
@click.option(
    "--num_heads", required=False, type=int, default=2, show_default=True
)
@click.option(
    "--num_layers", required=False, type=int, default=2, show_default=True
)
@click.option(
    "--learning_rate",
    required=False,
    type=float,
    default=1e-3,
    show_default=True,
)
@click.option(
    "--dropout", required=False, type=float, default=0.0, show_default=True
)
@click.option(
    "--num_epochs", required=True, type=int, default=100, show_default=True
)
@click.option(
    "--validate",
    is_flag=True,
    default=False,
    help="Run validation after each epoch (requires --train_days)",
)
@click.option(
    "--num_candidates",
    required=False,
    type=int,
    default=100,
    show_default=True,
    help="Number of candidates for validation",
)
@click.option(
    "--additional_info",
    required=False,
    type=str,
    default="",
    show_default=True,
    help="additional run info",
)
@click.option(
    "--tensorboard_enabled",
    required=False,
    type=bool,
    default=True,
    show_default=True,
    help="enable logging to tensorboard and wandb",
)
@click.option(
    "--use_bos_tokens",
    required=False,
    type=bool,
    default=False,
)
@click.option(
    "--use_positional_embedding",
    required=False,
    type=bool,
    default=True,
)
@click.option(
    "--use_alibi",
    required=False,
    type=bool,
    default=False,
)
@click.option(
    "--use_swiglu",
    required=False,
    type=bool,
    default=False,
)
@click.option(
    "--num_kv_heads",
    required=False,
    type=int,
    default=None,
    show_default=True,
)
@click.option(
    "--seed",
    required=False,
    type=int,
    default=42,
    show_default=True,
)
@click.option(
    "--warmup_ratio",
    required=False,
    type=float,
    default=0.0,
    show_default=True,
    help="Warmup ratio for cosine scheduler (0.0 = no warmup)",
)
@click.option(
    "--min_lr_ratio",
    required=False,
    type=float,
    default=1.0,
    show_default=True,
    help="min lr ratio",
)
@click.option(
    "--output_projection",
    required=False,
    type=click.Choice(["none", "embedding_dim", "pretrained_dim"]),
    default="none",
    show_default=True,
    help="How to project transformer output when pretrained embeddings are used: "
    "none=keep concatenated dim, embedding_dim=project to embedding_dim, "
    "pretrained_dim=project to pretrained_embedding_dim",
)
def main(
    exp_name: str,
    data_dir: str,
    size: str,
    batch_size: int,
    max_seq_len: int,
    embedding_dim: int,
    num_heads: int,
    num_layers: int,
    learning_rate: float,
    dropout: float,
    num_epochs: int,
    train_days: int | None,
    full_train: bool,
    validate: bool,
    num_candidates: int,
    additional_info: str,
    tensorboard_enabled: bool,
    use_bos_tokens: bool,
    use_alibi: bool,
    use_swiglu: bool,
    num_kv_heads: int | None,
    seed: int,
    use_positional_embedding: bool,
    warmup_ratio: float,
    min_lr_ratio: float,
    output_projection: str,
):
    writer = MlLogger(
        run_name=exp_name,
        log_dir=f"{CHECKPOINT_DIR}/runs/{exp_name}",
        config={
            key: value
            for key, value in locals().items()
            if key not in {"data_dir"}
        },
        enabled=tensorboard_enabled,
    )

    device = "cuda"
    interaction = "likes"

    checkpoints = TopKCheckpoints(
        k=2, checkpoint_dir=CHECKPOINT_DIR, exp_name=exp_name
    )

    data, train_dataloader, eval_df = create_training_environment(**locals())

    # Use centralized model and optimizer creation
    model, optimizer, scheduler = create_model_and_optimizer(**locals())

    # model = torch.compile(model)

    train(**locals())

    writer.finish()

    best_state_path = f"{CHECKPOINT_DIR}/{exp_name}_best_state.pth"
    best = checkpoints.best()
    if best is not None:
        torch.save(best.state_dict, best_state_path)
        logger.debug(
            f"Saved best checkpoint (epoch {best.epoch}, recall@100={best.score:.4f}) to {best_state_path}"
        )
    else:
        torch.save(model, best_state_path)
        logger.debug(f"Saved final model to {best_state_path}")


if __name__ == "__main__":
    main()
