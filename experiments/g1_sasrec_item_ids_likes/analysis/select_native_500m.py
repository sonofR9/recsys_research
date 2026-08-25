from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from decimal import Decimal
from pathlib import Path
from typing import Callable


EXPERIMENT = Path(__file__).resolve().parent.parent
REPO_ROOT = EXPERIMENT.parents[1]
TRANSFER_CONFIG = EXPERIMENT / "configs/transfer_variant.py"
VERIFY_ARTIFACT = EXPERIMENT / "launchers/verify_artifact.py"
EXPECTED_WINNER = ("0.001", "0.002")
APPROVED_STAGES = {
    20: tuple(
        (embedding, deep)
        for embedding in ("0.008", "0.016", "0.032", "0.064")
        for deep in ("0.002", "0.004", "0.008", "0.016", "0.032")
    ),
    40: (
        *tuple(
            (embedding, deep)
            for embedding in ("0.002", "0.004", "0.008")
            for deep in ("0.0005", "0.001", "0.002", "0.004")
        ),
        ("0.016", "0.032"),
    ),
    60: (
        ("0.0005", "0.001"),
        ("0.0005", "0.002"),
        ("0.0005", "0.004"),
        ("0.001", "0.001"),
        ("0.001", "0.002"),
        ("0.001", "0.004"),
        ("0.002", "0.002"),
        ("0.004", "0.001"),
    ),
    80: (("0.0005", "0.002"),),
}


def _slug(value: str) -> str:
    return value.replace(".", "p").replace("-", "m")


def _run(epochs: int, embedding_lr: str, deep_lr: str) -> str:
    cap = "" if epochs == 20 else f"_cap{epochs}"
    return (
        f"batchscale_b1280_e{_slug(embedding_lr)}_d{_slug(deep_lr)}"
        f"{cap}_ts2_r2"
    )


def _assignments(run: str, epochs: int, embedding_lr: str, deep_lr: str) -> list[str]:
    return [
        "G1_DATASET_SIZE=50m",
        f"G1_TRANSFER_RUN={run}",
        f"G1_TRANSFER_EPOCHS={epochs}",
        f"G1_TRANSFER_EMBEDDING_LR={embedding_lr}",
        f"G1_TRANSFER_DEEP_LR={deep_lr}",
        "G1_TRANSFER_PARAMETERIZATION=conventional",
        "G1_TRANSFER_BATCH_SIZE=1280",
        "G1_TRANSFER_DIM=64",
        "G1_TRANSFER_SOURCE_VARIANT=homework_fixed_leave_one_out",
    ]


def _verifiers() -> tuple[
    Callable[[Path, Path, list[str]], bool],
    Callable[[Path, Path, list[str]], bool],
]:
    spec = importlib.util.spec_from_file_location("g1_verify_artifact", VERIFY_ARTIFACT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load G1 artifact verifier")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.verify_config, module.verify_config_recipe


def _load_json(path: Path) -> dict:
    with path.open() as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def select_native_500m(
    generated: Path,
    verify: Callable[[Path, Path, list[str]], bool] | None = None,
    verify_recipe: Callable[[Path, Path, list[str]], bool] | None = None,
) -> dict:
    if verify is None or verify_recipe is None:
        default_verify, default_verify_recipe = _verifiers()
        verify = verify or default_verify
        verify_recipe = verify_recipe or default_verify_recipe
    approved = [
        (epochs, embedding_lr, deep_lr)
        for epochs, points in APPROVED_STAGES.items()
        for embedding_lr, deep_lr in points
    ]
    records = []
    for epochs, embedding_lr, deep_lr in approved:
        run = _run(epochs, embedding_lr, deep_lr)
        directory = generated / "logs" / f"g1_transfer_{run}_50m"
        assignments = _assignments(run, epochs, embedding_lr, deep_lr)
        selectable = verify(directory, TRANSFER_CONFIG, assignments)
        if not selectable:
            if not verify_recipe(directory, TRANSFER_CONFIG, assignments):
                raise ValueError(
                    f"missing or incompatible approved artifact: {directory.name}"
                )
        metadata = _load_json(directory / "training_metadata.json")
        if not selectable and not (
            metadata.get("selection_resolved") is False
            and metadata.get("stopped_epoch") == epochs
            and metadata.get("max_epochs") == epochs
        ):
            raise ValueError(
                f"approved artifact is not a cap-limited continuation: "
                f"{directory.name}"
            )
        metrics = _load_json(directory / "final_metrics.json")
        recall = metrics.get("recall@100")
        if (
            not isinstance(recall, (int, float))
            or isinstance(recall, bool)
            or not math.isfinite(recall)
        ):
            raise ValueError(f"{directory.name} has invalid recall@100")
        records.append(
            {
                "run": directory.name,
                "epochs": epochs,
                "embedding_lr": embedding_lr,
                "deep_lr": deep_lr,
                "selection_resolved": (
                    selectable and metadata.get("selection_resolved") is True
                ),
                "recall@100": float(recall),
                "metrics": metrics,
                "metadata": metadata,
            }
        )

    expected = [
        record
        for record in records
        if (record["embedding_lr"], record["deep_lr"]) == EXPECTED_WINNER
        and record["epochs"] == 60
    ]
    if len(expected) != 1 or not expected[0]["selection_resolved"]:
        raise ValueError("expected native-50M winner is not selection_resolved")
    for record in records:
        has_later_continuation = any(
            later["epochs"] > record["epochs"]
            and later["embedding_lr"] == record["embedding_lr"]
            and later["deep_lr"] == record["deep_lr"]
            for later in records
        )
        if not record["selection_resolved"] and not has_later_continuation:
            raise ValueError(
                f"terminal artifact is not selection_resolved: {record['run']}"
            )
    eligible = [record for record in records if record["selection_resolved"]]
    if not eligible:
        raise ValueError("approved grid has no validation-resolved candidate")
    best_recall = max(record["recall@100"] for record in eligible)
    winners = [record for record in eligible if record["recall@100"] == best_recall]
    if len(winners) != 1:
        raise ValueError("approved grid does not have a unique resolved winner")
    winner = winners[0]
    winner_rates = (winner["embedding_lr"], winner["deep_lr"])
    if winner_rates != EXPECTED_WINNER:
        raise ValueError(f"unexpected native-50M winner: {winner_rates}")

    final_stage = [record for record in records if record["epochs"] == 60]
    embedding_rates = sorted(
        {Decimal(record["embedding_lr"]) for record in final_stage}
    )
    deep_rates = sorted({Decimal(record["deep_lr"]) for record in final_stage})
    embedding_winner = Decimal(winner["embedding_lr"])
    deep_winner = Decimal(winner["deep_lr"])
    if not (
        embedding_rates[0] < embedding_winner < embedding_rates[-1]
        and deep_rates[0] < deep_winner < deep_rates[-1]
    ):
        raise ValueError("native-50M winner is not interior to the final LR screen")

    digest_records = [
        {
            "run": record["run"],
            "metrics": record["metrics"],
            "metadata": record["metadata"],
        }
        for record in records
    ]
    encoded = json.dumps(
        digest_records,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    source_digest = hashlib.sha256(encoded).hexdigest()
    return {
        "source_digest": source_digest,
        "source_id": source_digest[:12],
        "source_artifacts": len(records),
        "embedding_lr": winner["embedding_lr"],
        "deep_lr": winner["deep_lr"],
        "recall@100": winner["recall@100"],
        "winner_run": winner["run"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated", type=Path, default=REPO_ROOT / "generated")
    parser.add_argument(
        "--format", choices=("json", "tsv", "provenance-tsv"), default="json"
    )
    args = parser.parse_args()
    selection = select_native_500m(args.generated)
    if args.format == "tsv":
        print(
            "\t".join(
                str(selection[key])
                for key in (
                    "source_id",
                    "embedding_lr",
                    "deep_lr",
                    "source_artifacts",
                )
            )
        )
    elif args.format == "provenance-tsv":
        print(
            "\t".join(
                str(selection[key])
                for key in (
                    "source_digest",
                    "source_id",
                    "embedding_lr",
                    "deep_lr",
                    "source_artifacts",
                    "winner_run",
                )
            )
        )
    else:
        print(json.dumps(selection, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
