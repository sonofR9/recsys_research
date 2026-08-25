from contextlib import contextmanager
from typing import Any

from torch.utils.tensorboard import SummaryWriter

import wandb


class MlLogger:
    def __init__(
        self,
        run_name: str,
        log_dir: str,
        config: dict[str, Any],
        enabled: bool,
    ):
        self.enabled = enabled
        if not self.enabled:
            return

        self.run = wandb.init(
            entity="sasha7tdd7-bmstu",
            project="yambda_hw",
            name=run_name,
            config=config,
        )
        self.writer = SummaryWriter(log_dir=log_dir)

    def log(self, metrics: dict[str, Any], step: int):
        if not self.enabled:
            return

        self.run.log(metrics, step=step)
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, step)

    def finish(self):
        if not self.enabled:
            return

        self.writer.close()
        self.run.finish()

    # def log_final_metrics(self, metrics: dict[str, Any]):
    #     self.run.log(metrics)
    #     self.run.finish()


@contextmanager
def ml_logger(run_name: str, log_dir: str, config: dict[str, Any]):
    logger = MlLogger(run_name, log_dir, config)
    try:
        yield logger
    finally:
        logger.finish()
