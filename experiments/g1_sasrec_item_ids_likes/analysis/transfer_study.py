"""Read the transfer-study sweeps and report where each optimum sits.

The study asks whether the learning-rate optimum moves along an axis, so the
readout is the argmax of a one-dimensional curve, not a winning configuration.
A curve whose argmax sits on an endpoint has not been bracketed and says so.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import re

METRIC = "recall@100"
EMBEDDING = "embedding_learning_rate"
DEEP = "deep_learning_rate"
HORIZON_FREE_SHAPES = frozenset({"constant", "inverse_sqrt", "power"})


@dataclass(frozen=True)
class Point:
    curve: str
    embedding_lr: float
    deep_lr: float
    metric: float
    best_epoch: int | None
    epochs_trained: int | None
    horizon_complete: bool | None
    schedule_shape: str | None
    name: str

    @property
    def truncated(self) -> bool:
        """A horizon-free shape has no horizon to leave unspent."""
        return (
            self.schedule_shape not in HORIZON_FREE_SHAPES
            and self.horizon_complete is False
        )


def _collect(generated: Path, pattern: re.Pattern[str]) -> list[Point]:
    points = []
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
        if metrics.get(METRIC) is None:
            continue
        points.append(
            Point(
                curve=match.group("curve"),
                embedding_lr=metadata[EMBEDDING],
                deep_lr=metadata[DEEP],
                metric=metrics[METRIC],
                best_epoch=metadata.get("best_epoch"),
                epochs_trained=metadata.get("epochs_trained"),
                horizon_complete=metadata.get("lr_horizon_complete"),
                schedule_shape=(
                    (metadata.get("transfer_invariants") or {})
                    .get("lr_schedule", {})
                    .get("shape")
                ),
                name=directory.name,
            )
        )
    return points


def _curve_order(curve: str) -> tuple[str, int]:
    digits = re.search(r"\d+", curve)
    return (re.sub(r"\d+", "", curve), int(digits.group()) if digits else 0)


def _table(title: str, note: str, points: list[Point], swept: str) -> str:
    rate_of = (lambda point: point.deep_lr) if swept == DEEP else (
        lambda point: point.embedding_lr
    )
    rates = sorted({rate_of(point) for point in points})
    curves = sorted({point.curve for point in points}, key=_curve_order)
    lines = [
        f"### {title}",
        "",
        note + "; **bold** marks each curve's argmax.",
        "",
        "| curve | "
        + " | ".join(f"{rate:g}" for rate in rates)
        + " | argmax | best/trained |",
        "|" + " --- |" * (len(rates) + 3),
    ]
    for curve in curves:
        on_curve = [point for point in points if point.curve == curve]
        by_rate = {rate_of(point): point for point in on_curve}
        best = max(on_curve, key=lambda point: point.metric)
        cells = []
        for rate in rates:
            point = by_rate.get(rate)
            if point is None:
                cells.append("—")
            elif point is best:
                cells.append(f"**{point.metric:.5f}**")
            else:
                cells.append(f"{point.metric:.5f}")
        bracketed = rates.index(rate_of(best)) not in (0, len(rates) - 1)
        argmax = f"{rate_of(best):g}" + ("" if bracketed else " (endpoint)")
        epochs = f"{best.best_epoch}/{best.epochs_trained}"
        lines.append(
            f"| {curve} | " + " | ".join(cells) + f" | {argmax} | {epochs} |"
        )
    truncated = sorted(point.name for point in points if point.truncated)
    lines.append("")
    if truncated:
        lines.append(f"Not horizon-complete: {', '.join(truncated)}")
        lines.append("")
    return "\n".join(lines)


WIDTH = re.compile(
    r"g1_rqtune_(?P<curve>dimension_\d+|architecture_control)"
    r"_e\w+_hz1_ts2_r2_50m"
)
FFN = re.compile(
    r"g1_transfer_(?P<curve>ffn(?:ratio|base)_ffn\d+)_e\w+_ts2_r2_50m"
)
SIZE = re.compile(
    r"g1_transfer_sizesweep_(?P<curve>linear|power)_e\w+_ts2_r2_SIZE"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated", type=Path, default=Path("generated"))
    arguments = parser.parse_args()

    width = _collect(arguments.generated, WIDTH)
    if width:
        print(
            _table(
                "Arm A — model width, deep-LR sweep (native 50M)",
                "embedding LR held at 0.032",
                [point for point in width if point.embedding_lr == 0.032],
                DEEP,
            )
        )
        print(
            _table(
                "Arm A — model width, embedding-LR sweep (native 50M)",
                "deep LR held at 0.012",
                [point for point in width if point.deep_lr == 0.012],
                EMBEDDING,
            )
        )
    ffn = _collect(arguments.generated, FFN)
    if ffn:
        print(
            _table(
                "Arm B — FFN width, deep-LR sweep (native 50M)",
                "embedding LR held at 0.032",
                ffn,
                DEEP,
            )
        )
    for size in ("50m", "500m"):
        sized = _collect(
            arguments.generated,
            re.compile(SIZE.pattern.replace("SIZE", size)),
        )
        deep_sweep = [point for point in sized if point.embedding_lr == 0.032]
        if deep_sweep:
            print(
                _table(
                    f"Arm C — dataset size, deep-LR sweep (native {size})",
                    "embedding LR held at 0.032",
                    deep_sweep,
                    DEEP,
                )
            )
        embedding_sweep = [point for point in sized if point.deep_lr == 0.012]
        if embedding_sweep:
            print(
                _table(
                    f"Arm C — dataset size, embedding-LR sweep (native {size})",
                    "deep LR held at 0.012",
                    embedding_sweep,
                    EMBEDDING,
                )
            )


if __name__ == "__main__":
    main()
