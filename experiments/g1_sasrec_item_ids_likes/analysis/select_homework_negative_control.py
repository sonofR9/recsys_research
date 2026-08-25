from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import runpy

from dcn.training_metadata import GENERATION_TRAINING_SEMANTICS_REVISION


INITIAL_EMBEDDING_LRS = {0.0005, 0.001, 0.002}
INITIAL_DEEP_LRS = {0.001, 0.002, 0.004}


@dataclass(frozen=True)
class ControlRun:
    embedding_lr: float
    deep_lr: float
    recall_at_100: float
    resolved: bool
    path: Path


def canonicalize_learning_rates(raw_values: list[str]) -> list[str]:
    canonical = []
    seen: set[float] = set()
    for raw_value in raw_values:
        try:
            value = float(raw_value)
        except ValueError as error:
            raise ValueError(
                f"LR must be a positive number with finite value: {raw_value}"
            ) from error
        if not math.isfinite(value) or value <= 0:
            raise ValueError(
                f"LR must be a positive number with finite value: {raw_value}"
            )
        if value in seen:
            raise ValueError(f"LR values contain duplicate value: {raw_value}")
        seen.add(value)
        canonical.append(str(value))
    return canonical


def _normalized_lineage(lineage: str) -> str:
    match = re.fullmatch(
        r"(?P<prefix>g1_homework_(?:random|logq))_"
        r"(?P<tag>[a-z0-9_]+)(?P<rates>_e[^_]+_d[^_]+)",
        lineage,
    )
    if match is None or match.group("tag") != "capcontinue":
        return lineage
    return f"{match.group('prefix')}_initial{match.group('rates')}"


def _continuation(run: ControlRun) -> tuple[int, int, str] | None:
    match = re.fullmatch(
        r"(?P<lineage>.+?)(?:_cap(?P<epochs>[1-9]\d*))?"
        rf"_ts{GENERATION_TRAINING_SEMANTICS_REVISION}"
        r"_r(?P<revision>[1-9]\d*)_50m",
        run.path.name,
    )
    if match is None:
        return None
    epochs = int(match.group("epochs") or 20)
    if epochs < 20 or (epochs == 20 and match.group("epochs") is not None):
        return None
    lineage = _normalized_lineage(match.group("lineage"))
    return epochs, int(match.group("revision")), lineage


def _latest_resolved(
    pair: tuple[float, float], candidates: list[ControlRun]
) -> ControlRun:
    resolved = [run for run in candidates if run.resolved]
    continuations = [(run, _continuation(run)) for run in resolved]
    if any(continuation is None for _, continuation in continuations):
        if len(resolved) == 1 and not re.search(
            r"(?:_cap\d+_|_capcontinue_)", resolved[0].path.name
        ):
            return resolved[0]
        raise ValueError(f"ambiguous resolved artifacts for LR pair {pair}")
    lineages = {continuation[2] for _, continuation in continuations}
    if len(lineages) != 1:
        raise ValueError(f"ambiguous resolved artifacts for LR pair {pair}")
    lineage = next(iter(lineages))
    chain = [
        (run, continuation)
        for run in candidates
        if (continuation := _continuation(run)) is not None
        and continuation[2] == lineage
    ]
    anchors = [item for item in chain if item[1][0] == 20]
    if len(anchors) != 1:
        raise ValueError(f"cap continuation has no unique base for LR pair {pair}")
    ordered_chain = sorted(chain, key=lambda item: item[1][:2])
    epochs = [continuation[0] for _, continuation in ordered_chain]
    revisions = [continuation[1] for _, continuation in ordered_chain]
    expected_epochs = [20]
    while expected_epochs[-1] < epochs[-1]:
        expected_epochs.append(expected_epochs[-1] * 2)
    expected_revisions = list(range(1, len(expected_epochs) + 1))
    if epochs != expected_epochs or revisions != expected_revisions:
        raise ValueError(f"ambiguous resolved artifacts for LR pair {pair}")
    return max(continuations, key=lambda item: item[1][:2])[0]


def validate_selection(
    runs: list[ControlRun], embedding_lr: float, deep_lr: float
) -> ControlRun:
    by_pair: dict[tuple[float, float], list[ControlRun]] = {}
    for run in runs:
        by_pair.setdefault((run.embedding_lr, run.deep_lr), []).append(run)
    required = {
        (embedding, deep)
        for embedding in INITIAL_EMBEDDING_LRS
        for deep in INITIAL_DEEP_LRS
    }
    missing = required - by_pair.keys()
    if missing:
        raise ValueError(f"initial Cartesian grid is incomplete: {sorted(missing)}")
    unresolved = [
        pair for pair, candidates in by_pair.items() if not any(run.resolved for run in candidates)
    ]
    if unresolved:
        raise ValueError(f"cap-unresolved LR pairs: {sorted(unresolved)}")
    resolved = [
        _latest_resolved(pair, candidates)
        for pair, candidates in by_pair.items()
    ]
    best_recall = max(run.recall_at_100 for run in resolved)
    winners = [run for run in resolved if run.recall_at_100 == best_recall]
    if len(winners) != 1:
        raise ValueError("50M recall winner is not unique")
    winner = winners[0]
    if (winner.embedding_lr, winner.deep_lr) != (embedding_lr, deep_lr):
        raise ValueError(
            "requested selection is not the unique 50M recall winner: "
            f"expected {winner.embedding_lr:g}:{winner.deep_lr:g}"
        )
    embedding_neighbors = {
        run.embedding_lr for run in resolved if run.deep_lr == winner.deep_lr
    }
    deep_neighbors = {
        run.deep_lr for run in resolved if run.embedding_lr == winner.embedding_lr
    }
    if not (
        any(value < winner.embedding_lr for value in embedding_neighbors)
        and any(value > winner.embedding_lr for value in embedding_neighbors)
    ):
        raise ValueError("embedding-LR winner is not closed on both boundaries")
    if not (
        any(value < winner.deep_lr for value in deep_neighbors)
        and any(value > winner.deep_lr for value in deep_neighbors)
    ):
        raise ValueError("deep-LR winner is not closed on both boundaries")
    return winner


def _load_runs(root: Path, family: str) -> list[ControlRun]:
    prefix = f"g1_homework_{family}_"
    config = root / f"experiments/g1_sasrec_item_ids_likes/configs/homework_{family}_control.py"
    verifier = runpy.run_path(
        str(root / "experiments/g1_sasrec_item_ids_likes/launchers/verify_artifact.py")
    )
    classify = verifier["classify_config"]
    environment_prefix = f"G1_HOMEWORK_{family.upper()}"
    runs = []
    for directory in sorted((root / "generated/logs").glob(f"{prefix}*_50m")):
        run = directory.name.removeprefix(prefix).removesuffix("_50m")
        if run.startswith("selected_"):
            continue
        metadata_path = directory / "training_metadata.json"
        metrics_path = directory / "final_metrics.json"
        if not metadata_path.exists() or not metrics_path.exists():
            raise ValueError(f"incomplete 50M artifact: {directory.name}")
        metadata = json.loads(metadata_path.read_text())
        metrics = json.loads(metrics_path.read_text())
        revision = run.rsplit("_r", 1)[-1]
        assignments = [
            f"{environment_prefix}_RUN={run}",
            f"{environment_prefix}_EPOCHS={metadata['num_epochs']}",
            f"{environment_prefix}_RUN_REVISION={revision}",
            f"{environment_prefix}_EMBEDDING_LR={metadata['embedding_learning_rate']}",
            f"{environment_prefix}_DEEP_LR={metadata['deep_learning_rate']}",
            f"{environment_prefix}_DATASET_SIZE=50m",
        ]
        state = classify(directory, config, assignments)
        if state == "incompatible":
            raise ValueError(f"incompatible 50M artifact: {directory.name}")
        runs.append(
            ControlRun(
                embedding_lr=float(metadata["embedding_learning_rate"]),
                deep_lr=float(metadata["deep_learning_rate"]),
                recall_at_100=float(metrics["recall@100"]),
                resolved=state == "complete",
                path=directory,
            )
        )
    return runs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonicalize-lrs", nargs="+")
    parser.add_argument("--family", choices=("random", "logq"))
    parser.add_argument("--embedding-lr", type=float)
    parser.add_argument("--deep-lr", type=float)
    arguments = parser.parse_args()
    if arguments.canonicalize_lrs is not None:
        if any(
            value is not None
            for value in (
                arguments.family,
                arguments.embedding_lr,
                arguments.deep_lr,
            )
        ):
            parser.error("--canonicalize-lrs cannot be combined with selection")
        try:
            canonical = canonicalize_learning_rates(arguments.canonicalize_lrs)
        except ValueError as error:
            parser.error(str(error))
        print(*canonical, sep="\n")
        return
    if any(
        value is None
        for value in (
            arguments.family,
            arguments.embedding_lr,
            arguments.deep_lr,
        )
    ):
        parser.error("selection requires --family, --embedding-lr, and --deep-lr")
    root = Path(__file__).resolve().parents[3]
    winner = validate_selection(
        _load_runs(root, arguments.family),
        arguments.embedding_lr,
        arguments.deep_lr,
    )
    print(f"{winner.embedding_lr}:{winner.deep_lr}")


if __name__ == "__main__":
    main()
