"""Render RQ1's width-transfer tables in reader-facing report format.

Each table asks what it costs to use the control's rate at a width that was
tuned for a different one, so every row's reference is that width's own best
point on the same one-dimensional sweep.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import re

from experiments.g1_sasrec_item_ids_likes.analysis import reporting

CONTROL_WIDTH = 64
SHARED_EMBEDDING_LR = 0.032
SHARED_DEEP_LR = 0.012

PROXY = re.compile(
    r"g1_rqtune_(?:architecture_control|dimension_(?P<width>\d+))"
    r"_e\w+_hz1_ts2_r2_50m"
)
CONFIRMATION = re.compile(
    r"g1_rqtune_rqfinal_(?:architecture_control|dimension_(?P<width>\d+))"
    r"_e\w+_d\w+_b1280_ts2_r2_500m"
)


@dataclass(frozen=True)
class Run:
    width: int
    embedding_lr: float
    deep_lr: float
    recall: float
    ndcg: float
    epochs_trained: int | None
    horizon_epochs: int | None
    name: str

    @property
    def short_of_horizon(self) -> bool:
        return (
            self.epochs_trained is not None
            and self.horizon_epochs is not None
            and self.epochs_trained < self.horizon_epochs
        )


def load_runs(generated: Path, pattern: re.Pattern[str]) -> list[Run]:
    runs = []
    for directory in sorted((generated / "logs").iterdir()):
        match = pattern.fullmatch(directory.name)
        if match is None:
            continue
        metadata_path = directory / "training_metadata.json"
        metrics_path = directory / "final_metrics.json"
        if not metadata_path.exists() or not metrics_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text())
        metrics = json.loads(metrics_path.read_text())
        if metrics.get("recall@100") is None:
            continue
        width = match.group("width")
        runs.append(
            Run(
                width=int(width) if width else CONTROL_WIDTH,
                embedding_lr=metadata["embedding_learning_rate"],
                deep_lr=metadata["deep_learning_rate"],
                recall=metrics["recall@100"],
                ndcg=metrics["ndcg@100"],
                epochs_trained=metadata.get("epochs_trained"),
                horizon_epochs=metadata.get("lr_schedule_horizon_epochs"),
                name=directory.name,
            )
        )
    return runs


def sweep_table(runs: list[Run], swept: str) -> str:
    held, shared = (
        ("embedding_lr", SHARED_DEEP_LR)
        if swept == "deep"
        else ("deep_lr", SHARED_EMBEDDING_LR)
    )
    held_value = SHARED_EMBEDDING_LR if swept == "deep" else SHARED_DEEP_LR
    rate_of = (lambda run: run.deep_lr) if swept == "deep" else (
        lambda run: run.embedding_lr
    )
    lines = [
        f"| transformer width | best {swept} LR | recall@100 at {shared:g} "
        f"| ndcg@100 at {shared:g} | reference: this width's best |",
        "| ---: | :---: | :---: | :---: | :---: |",
    ]
    for width in sorted({run.width for run in runs}):
        on_curve = [
            run
            for run in runs
            if run.width == width and getattr(run, held) == held_value
        ]
        if not on_curve:
            continue
        best = max(on_curve, key=lambda run: run.recall)
        transferred = next(
            (run for run in on_curve if rate_of(run) == shared), None
        )
        if transferred is None:
            continue
        label = f"{width}" + (" (control)" if width == CONTROL_WIDTH else "")
        recall_cell = reporting.change_cell(
            transferred.recall, best.recall, "recall@100"
        )
        ndcg_cell = reporting.change_cell(transferred.ndcg, best.ndcg, "ndcg@100")
        lines.append(
            f"| {label} | {rate_of(best):g} | {recall_cell} | {ndcg_cell} "
            f"| {reporting.absolute(best.recall)} / "
            f"{reporting.absolute(best.ndcg)} |"
        )
    return "\n".join(lines)


def confirmation_table(runs: list[Run]) -> str:
    lines = [
        "| transformer width | 50M-local rate | recall@100 at the shared rate "
        "| ndcg@100 at the shared rate | reference: the local rate | epochs |",
        "| ---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for width in sorted({run.width for run in runs}):
        at_width = [run for run in runs if run.width == width]
        shared = next(
            (
                run
                for run in at_width
                if run.embedding_lr == SHARED_EMBEDDING_LR
                and run.deep_lr == SHARED_DEEP_LR
            ),
            None,
        )
        if shared is None:
            continue
        local = [run for run in at_width if run is not shared]
        label = f"{width}" + (" (control)" if width == CONTROL_WIDTH else "")
        epochs = f"{shared.epochs_trained}/{shared.horizon_epochs}"
        if not local:
            lines.append(
                f"| {label} | same rate | {reporting.absolute(shared.recall)} "
                f"| {reporting.absolute(shared.ndcg)} | — | {epochs} |"
            )
            continue
        best_local = max(local, key=lambda run: run.recall)
        lines.append(
            f"| {label} | {best_local.embedding_lr:g}/{best_local.deep_lr:g} "
            f"| {reporting.change_cell(shared.recall, best_local.recall, 'recall@100')} "
            f"| {reporting.change_cell(shared.ndcg, best_local.ndcg, 'ndcg@100')} "
            f"| {reporting.absolute(best_local.recall)} / "
            f"{reporting.absolute(best_local.ndcg)} | {epochs} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated", type=Path, default=Path("generated"))
    arguments = parser.parse_args()

    proxy = load_runs(arguments.generated, PROXY)
    print("### deep-LR sweep, embedding held at 0.032 (native 50M)\n")
    print(sweep_table(proxy, "deep") + "\n")
    print("### embedding-LR sweep, deep held at 0.012 (native 50M)\n")
    print(sweep_table(proxy, "embedding") + "\n")

    confirmation = load_runs(arguments.generated, CONFIRMATION)
    print("### native-500M confirmation\n")
    print(confirmation_table(confirmation) + "\n")
    short = sorted(run.name for run in confirmation if run.short_of_horizon)
    if short:
        print(f"Stopped short of the annealing horizon: {', '.join(short)}")


if __name__ == "__main__":
    main()
