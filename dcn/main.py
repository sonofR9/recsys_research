#!/usr/bin/env python3

import argparse
from contextlib import contextmanager
import fcntl
import importlib.util
import logging
import multiprocessing
import os
import sys
from pathlib import Path
from typing import Iterator

from .config import Experiment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger(__name__)


@contextmanager
def _run_ownership(run_name: str) -> Iterator[None]:
    lock_path = Path("generated/logs/.run-locks") / f"{run_name}.lock"
    inherited_path = os.environ.get("DCN_RUN_LOCK_PATH")
    inherited_fd = os.environ.get("DCN_RUN_LOCK_FD")
    if (inherited_path is None) != (inherited_fd is None):
        raise RuntimeError("queue run ownership requires both lock path and FD")
    if inherited_path is not None and inherited_fd is not None:
        if Path(inherited_path).resolve() != lock_path.resolve():
            raise RuntimeError("queue run-ownership lock does not match run name")
        try:
            descriptor = int(inherited_fd)
            descriptor_stat = os.fstat(descriptor)
            path_stat = lock_path.stat()
        except (OSError, ValueError) as error:
            raise RuntimeError("queue run-ownership FD is invalid") from error
        if (descriptor_stat.st_dev, descriptor_stat.st_ino) != (
            path_stat.st_dev,
            path_stat.st_ino,
        ):
            raise RuntimeError("queue run-ownership FD does not match lock path")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("queue did not transfer run ownership") from error
        yield
        return

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"run {run_name!r} is already active") from error
        handle.seek(0)
        handle.truncate()
        handle.write(str(os.getpid()))
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train multi-task recommender system")
    parser.add_argument(
        "-s",
        "--script",
        type=str,
        required=True,
        help="Path to a Python file exposing a module-level `experiment` (an Experiment instance).",
    )
    return parser.parse_args()


def load_experiment(script_path: str | Path) -> Experiment:
    script_path = Path(script_path)
    spec = importlib.util.spec_from_file_location("dcn_experiment_script", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load experiment script: {script_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        del sys.modules[spec.name]
        raise

    experiment = getattr(module, "experiment", None)
    if experiment is None:
        raise AttributeError(
            f"Script {script_path} must define a module-level `experiment` attribute."
        )
    if not isinstance(experiment, Experiment):
        raise TypeError(
            f"`experiment` in {script_path} must be an Experiment instance, "
            f"got {type(experiment).__name__}."
        )
    return experiment


def run_experiment(experiment: Experiment) -> None:
    experiment.setup()

    for stage in experiment.stages:
        logger.info("Starting stage %r...", stage.name)
        stage.run()
        logger.info("Stage %r completed", stage.name)

    logger.info("Training completed!")


def _kill_child_processes() -> None:
    """Every child this process has is a dataloader worker or the forkserver
    behind them, and torch registers an atexit join per persistent pinned-memory
    worker. A worker parked on a full prefetch queue takes the whole five-second
    timeout; across three loaders that was a minute of a finished run sitting on
    the GPU. Killed first, those joins return at once, and the run leaves the way
    it otherwise would -- exiting outright instead orphans the forkserver and
    every worker under it, a gigabyte apiece."""
    for worker in multiprocessing.active_children():
        worker.kill()


def main() -> None:
    args = parse_args()
    experiment = load_experiment(args.script)
    with _run_ownership(experiment.run_name):
        try:
            run_experiment(experiment)
        finally:
            # A crashed run is exactly when the queue behind it wants the device
            # back quickly, so this is not only the happy path's business.
            _kill_child_processes()


if __name__ == "__main__":
    main()
