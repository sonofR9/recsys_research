#!/usr/bin/env python3

import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from torch.utils.data import DataLoader

import wandb


sys.path.insert(0, str(Path(__file__).parent.parent))

from models.sasrec.data import CHECKPOINT_DIR
from models.sasrec.ml_logger import MlLogger
from models.sasrec.train import (
    create_model_and_optimizer,
    create_training_environment,
    train,
)
from models.sasrec.utils import setup_environment


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class DummyCheckpoints:
    def update(self, *args, **kwargs):
        pass

    def best(self):
        return None


def train_with_config(
    seed: int,
    max_seq_len: int,
    prepared_datasets: Dict[int, Tuple[Any, DataLoader, Any]],
    config: Dict[str, Any],
    device: str,
    **kwargs,
) -> Dict[str, Any] | None:
    """Train a model with the given configuration and return metrics."""
    setup_environment(seed)

    data, train_dataloader, eval_df = prepared_datasets[max_seq_len]

    model, optimizer, scheduler = create_model_and_optimizer(
        **locals(), **kwargs
    )

    exp_name = f"sweep_{seed}_{wandb.run.id if wandb.run else 'local'}"
    writer = MlLogger(
        run_name=exp_name,
        log_dir=f"{CHECKPOINT_DIR}/runs/{exp_name}",
        config=config,
        enabled=False,
    )

    checkpoints = DummyCheckpoints()

    metrics = train(
        train_dataloader=train_dataloader,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        writer=writer,
        checkpoints=checkpoints,
        device=device,
        num_epochs=config["num_epochs"],
        eval_df=eval_df,
        max_seq_len=config["max_seq_len"],
        batch_size=config["batch_size"],
        num_candidates=config["num_candidates"],
    )

    return metrics


def main() -> None:
    """Main function to run the hyperparameter sweep."""
    data_dir = (
        "/home/sonofr/repos/shad/ysda_recsys/competition/generated/yambda_data"
    )
    size = "5b"
    interaction = "likes"
    train_days = 299
    batch_size = 256
    num_candidates = 100

    max_seq_len_values = [128, 200, 256, 512]
    prepared_datasets = {}

    logger.info("Preparing datasets for different max_seq_len values...")
    for seq_len in max_seq_len_values:
        data, train_dataloader, eval_df = create_training_environment(
            data_dir=data_dir,
            size=size,
            max_seq_len=seq_len,
            train_days=train_days,
            full_train=False,
            validate=True,
            batch_size=batch_size,
            seed=42,  # Base seed for environment setup
            interaction=interaction,
        )
        prepared_datasets[seq_len] = (data, train_dataloader, eval_df)
        logger.info(
            f"Prepared dataset for max_seq_len={seq_len}: {len(train_dataloader)} batches, {len(eval_df) if eval_df is not None else 0} validation users"
        )

    device = "cuda"
    logger.info(f"Using device: {device}")

    sweep_config = {
        "method": "random",
        "metric": {"name": "recall@100", "goal": "maximize"},
        "parameters": {
            "learning_rate": {
                "distribution": "log_uniform",
                "min": math.log(3e-5),
                "max": math.log(1e-2),
            },
            "embedding_dim": {"value": 256},
            "num_kv_heads": {"values": [2, 4]},
            "num_heads": {"value": 4},
            "num_layers": {"value": 6},
            "dropout": {"distribution": "uniform", "min": 0.0, "max": 0.5},
            "warmup_ratio": {"distribution": "uniform", "min": 0.0, "max": 0.3},
            "min_lr_ratio": {
                "distribution": "uniform",
                "min": 0.01,
                "max": 0.5,
            },
            "use_swiglu": {"value": True},
            "max_seq_len": {"value": 512},
            "num_epochs": {
                "distribution": "int_uniform",
                "min": 10,
                "max": 100,
            },
            # Fixed parameters for all runs
            "batch_size": {"value": batch_size},
            "num_candidates": {"value": num_candidates},
            "use_bos_tokens": {"value": True},
            "use_alibi": {"value": True},
            "use_positional_embedding": {"value": True},
            "output_projection": {"value": "none"},
        },
    }

    def sweep_train() -> None:
        """Function called by W&B for each hyperparameter configuration."""
        wandb.init(group="sweep_04_19", job_type="hyperparams_search")
        config = wandb.config

        seeds = [42]
        all_recall_100 = []
        all_metrics = []
        try:
            for seed in seeds:
                metrics = train_with_config(
                    seed=seed,
                    prepared_datasets=prepared_datasets,
                    device=device,
                    config=config,
                    **config,
                )

                assert metrics is not None
                recall_100 = metrics["recall"][100]
                all_recall_100.append(recall_100)
                all_metrics.append(metrics)

                wandb.log(
                    {
                        f"seed_{seed}/recall@100": recall_100,
                    }
                )
        except Exception as e:
            logger.error(f"Training run failed with error: {e}")
            wandb.log({"recall@100": 0.0, "num_successful_runs": 0})
            return

        avg_recall_100 = sum(all_recall_100) / len(all_recall_100)

        wandb.log(
            {
                "recall@100": avg_recall_100,
                # "num_successful_runs": len(all_recall_100),
                # "std_recall@100": (
                #     torch.std(torch.tensor(all_recall_100)).item()
                #     if len(all_recall_100) > 1
                #     else 0.0
                # ),
            }
        )

        logger.info(
            f"Average recall@100: {avg_recall_100:.4f} (from {len(all_recall_100)} runs)"
        )

    sweep_id = wandb.sweep(
        sweep_config, project="yambda_hw", entity="sasha7tdd7-bmstu"
    )
    logger.info(f"Starting sweep with ID: {sweep_id}")

    wandb.agent(sweep_id, function=sweep_train, count=20)


if __name__ == "__main__":
    main()
