from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import logging
import platform
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import rectools
import torch
from pytorch_lightning import seed_everything
from rectools import Columns
from rectools.dataset import Dataset
from rectools.models import SASRecModel
from rectools.models.nn.item_net import IdEmbeddingsItemNet
from rectools.models.nn.transformers.ligr import LiGRLayers

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.g2_esasrec.official import catalog_data, protocol, provenance

GENERATED = PROJECT_ROOT / "generated"
EVAL_KS = (10, 50, 100)
SELECTION_METRIC = "recall@100"
RECTOOLS_VERSION = provenance.RECTOOLS_VERSION
EXPECTED_CATALOG_SIZE = 33_148
DEFAULTS = {
    "n_factors": 256,
    "n_blocks": 2,
    "n_heads": 4,
    "dropout_rate": 0.2,
    "session_max_len": 100,
    "n_negatives": 256,
    "lr": 1e-3,
    "batch_size": 128,
    "max_epochs": 100,
    "patience": 10,
}

logger = logging.getLogger("g2_official_esasrec")


def build_interactions(split: protocol.Split) -> pd.DataFrame:
    train = split.train.sort(
        [protocol.USER_COLUMN, protocol.TIMESTAMP_COLUMN], maintain_order=True
    )
    return pd.DataFrame(
        {
            Columns.User: train.get_column(protocol.USER_COLUMN).to_numpy(),
            Columns.Item: train.get_column(protocol.ITEM_COLUMN).to_numpy(),
            Columns.Weight: 1.0,
            Columns.Datetime: pd.to_datetime(
                train.get_column(protocol.TIMESTAMP_COLUMN).to_numpy(), unit="s"
            ),
        }
    )


def rank(
    model: SASRecModel, dataset: Dataset, users: list[int], k: int
) -> dict[int, list[int]]:
    recommendations = model.recommend(
        users=users, dataset=dataset, k=k, filter_viewed=False
    ).sort_values([Columns.User, Columns.Rank])
    return {
        int(user): [int(item) for item in items]
        for user, items in recommendations.groupby(Columns.User)[Columns.Item]
    }


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _environment() -> dict[str, object]:
    package_lines = sorted(
        f"{distribution.metadata['Name']}=={distribution.version}"
        for distribution in importlib.metadata.distributions()
        if distribution.metadata["Name"]
    )
    selected = {
        package: importlib.metadata.version(package)
        for package in provenance.OFFICIAL_PACKAGE_VERSIONS
    }
    python_version = platform.python_version()
    if python_version != provenance.OFFICIAL_PYTHON_VERSION:
        raise RuntimeError(f"official Python version changed: {python_version}")
    if selected != provenance.OFFICIAL_PACKAGE_VERSIONS:
        raise RuntimeError(f"official package versions changed: {selected}")
    return {
        "python": python_version,
        "packages": selected,
        "installed_packages": package_lines,
        "installed_packages_count": len(package_lines),
        "installed_packages_sha256": hashlib.sha256(
            "\n".join(package_lines).encode()
        ).hexdigest(),
    }


def _source_provenance() -> dict[str, dict[str, str]]:
    return provenance.source_manifest(
        provenance.rectools_source_paths()
        | {
            "catalog_data": Path(catalog_data.__file__),
            "runner": Path(__file__),
            "protocol": Path(protocol.__file__),
            "provenance": Path(provenance.__file__),
        }
    )


def train_and_score(arguments: argparse.Namespace) -> dict[str, object]:
    if rectools.__version__ != RECTOOLS_VERSION:
        raise RuntimeError(
            f"RecTools {RECTOOLS_VERSION} required, found {rectools.__version__}"
        )
    seed_everything(arguments.seed, workers=True)
    split = protocol.load_split(GENERATED)
    if split.catalog_size != EXPECTED_CATALOG_SIZE:
        raise RuntimeError(
            f"expected {EXPECTED_CATALOG_SIZE} mapped items, got {split.catalog_size}"
        )
    relevant = protocol.relevance(split)
    histories = protocol.query_histories(split, arguments.session_max_len)
    users = protocol.evaluable_users(histories, relevant)
    dataset = Dataset.construct(build_interactions(split))
    model = SASRecModel(
        n_blocks=arguments.n_blocks,
        n_heads=arguments.n_heads,
        n_factors=arguments.n_factors,
        dropout_rate=arguments.dropout_rate,
        session_max_len=arguments.session_max_len,
        loss="sampled_softmax",
        n_negatives=arguments.n_negatives,
        lr=arguments.lr,
        batch_size=arguments.batch_size,
        epochs=1,
        deterministic=True,
        verbose=0,
        dataloader_num_workers=arguments.num_workers,
        item_net_block_types=(IdEmbeddingsItemNet,),
        transformer_layers_type=LiGRLayers,
        data_preparator_type=catalog_data.CatalogCompleteSASRecDataPreparator,
        data_preparator_kwargs={
            "candidate_item_ids": split.catalog.tolist(),
            "expected_candidate_count": EXPECTED_CATALOG_SIZE,
        },
        recommend_batch_size=arguments.recommend_batch_size,
        recommend_torch_device="cuda" if torch.cuda.is_available() else "cpu",
    )

    history: list[dict[str, float]] = []
    best: dict[str, float | int] = {"epoch": 0, SELECTION_METRIC: -1.0}
    best_state = None
    started = time.perf_counter()
    for epoch in range(1, arguments.max_epochs + 1):
        epoch_started = time.perf_counter()
        model.fit_partial(dataset, min_epochs=1, max_epochs=1)
        train_seconds = time.perf_counter() - epoch_started
        metrics = protocol.score_rankings(
            rank(model, dataset, users, max(EVAL_KS)),
            relevant,
            users,
            EVAL_KS,
            split.catalog_size,
        )
        history.append({"epoch": epoch, "train_seconds": train_seconds, **metrics})
        if metrics[SELECTION_METRIC] > best[SELECTION_METRIC]:
            best = {"epoch": epoch, **metrics}
            best_state = copy.deepcopy(model.lightning_model.state_dict())
        elif epoch - int(best["epoch"]) >= arguments.patience:
            break
        logger.info(
            "epoch=%s recall@100=%.6f ndcg@100=%.6f",
            epoch,
            metrics["recall@100"],
            metrics["ndcg@100"],
        )

    if best_state is not None:
        model.lightning_model.load_state_dict(best_state)
    epochs_trained = int(history[-1]["epoch"])
    candidate_evidence = protocol.candidate_catalog_evidence(
        split, model.data_preparator.get_known_item_ids()
    )
    return {
        "run_name": arguments.run_name,
        "experiment": "g2_esasrec",
        "implementation": "RecTools SASRecModel with LiGRLayers",
        "dataset_size": "native-50m",
        "seed": arguments.seed,
        "best_epoch": best["epoch"],
        "epochs_trained": epochs_trained,
        "max_epochs": arguments.max_epochs,
        "patience": arguments.patience,
        "early_stopped": epochs_trained < arguments.max_epochs,
        "best_epoch_at_cap": best["epoch"] == arguments.max_epochs,
        "wall_seconds": time.perf_counter() - started,
        "hyperparameters": {
            key: getattr(arguments, key)
            for key in (
                "n_blocks",
                "n_heads",
                "n_factors",
                "dropout_rate",
                "session_max_len",
                "n_negatives",
                "lr",
                "batch_size",
            )
        },
        "protocol": {
            "cutoff": split.cutoff,
            **candidate_evidence,
            "candidate_catalog_source": "full mapped pre-split catalog",
            "train_events": split.train.height,
            "model_train_events_after_session_truncation": (
                model.data_preparator.train_dataset.interactions.df.shape[0]
            ),
            "validation_events": split.validation.height,
            "evaluable_users": len(users),
            "eval_ks": list(EVAL_KS),
            "selection_metric": SELECTION_METRIC,
            "exclude_seen": False,
        },
        "provenance": {
            "git_head": _git_head(),
            "environment": _environment(),
            "sources": _source_provenance(),
        },
        "metrics": {key: value for key, value in best.items() if key != "epoch"},
        "history": history,
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--recommend-batch-size", type=int, default=1024)
    for key, value in DEFAULTS.items():
        parser.add_argument(
            f"--{key.replace('_', '-')}", type=type(value), default=value
        )
    arguments = parser.parse_args()
    if arguments.run_name is None:
        arguments.run_name = f"g2_official_esasrec_50m_s{arguments.seed}"
    return arguments


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    arguments = parse_arguments()
    report = train_and_score(arguments)
    destination = GENERATED / "logs" / arguments.run_name
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "final_metrics.json").write_text(
        json.dumps(report["metrics"], indent=2, sort_keys=True)
    )
    (destination / "training_metadata.json").write_text(
        json.dumps(
            {key: value for key, value in report.items() if key != "history"},
            indent=2,
            sort_keys=True,
        )
    )
    (destination / "epoch_history.json").write_text(
        json.dumps(report["history"], indent=2)
    )


if __name__ == "__main__":
    main()
