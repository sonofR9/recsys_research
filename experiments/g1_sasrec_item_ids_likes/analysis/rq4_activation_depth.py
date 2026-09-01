from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from experiments.g1_sasrec_item_ids_likes.analysis import reporting

if TYPE_CHECKING:
    from utils.report_file_facts import ReportFileFacts


EMBEDDING_LR = 0.064
BATCH_SIZE = 1280
METRICS = ("recall@100", "ndcg@100")
WIDTHS = {
    "ReLU": 171,
    "GELU": 171,
    "SiLU": 171,
    "ReGLU": 114,
    "GEGLU": 114,
    "SwiGLU": 114,
}
GATED_FAMILIES = {"ReGLU", "GEGLU", "SwiGLU"}
VALIDATION_RECALL_PATTERN = re.compile(
    r"epoch/val_true\.recall@100=(?P<recall>[+-]?(?:\d+(?:\.\d*)?|\.\d+))"
)
SURFACES = (
    ("ReLU", 2, (0.006, 0.012, 0.024, 0.048, 0.096)),
    ("GELU", 2, (0.006, 0.012, 0.024, 0.048, 0.096)),
    ("SiLU", 2, (0.006, 0.012, 0.024, 0.048, 0.096)),
    ("ReGLU", 2, (0.006, 0.012, 0.024, 0.048, 0.096)),
    ("GEGLU", 2, (0.006, 0.012, 0.024)),
    ("SwiGLU", 2, (0.006, 0.012, 0.024)),
    ("GELU", 4, (0.006, 0.012, 0.024, 0.048, 0.096)),
    ("SwiGLU", 4, (0.006, 0.012, 0.024)),
    ("GELU", 8, (0.006, 0.012, 0.024, 0.048, 0.096)),
    ("SwiGLU", 8, (0.006, 0.012, 0.024)),
)
PINNED_GELU_DEPTH2 = tuple(
    "g1_rqtune_rqfinal_ffn_gelu171_e0p064_"
    f"d{str(rate).replace('.', 'p')}_b1280_cap40_ts2_r3_500m"
    for rate in (0.006, 0.012, 0.024)
)


@dataclass(frozen=True)
class RunSpec:
    family: str
    layers: int
    width: int
    deep_lr: float
    name: str
    reused: bool

    @property
    def gated(self) -> bool:
        return self.family in GATED_FAMILIES


@dataclass(frozen=True)
class Run:
    family: str
    layers: int
    width: int
    embedding_lr: float
    deep_lr: float
    batch_size: int
    best_epoch: int
    stopped_epoch: int
    horizon_epochs: int
    validation_recall: float
    metrics: dict[str, float]
    name: str
    reused: bool


def expected_specs() -> tuple[RunSpec, ...]:
    specs = []
    pinned = iter(PINNED_GELU_DEPTH2)
    for family, layers, rates in SURFACES:
        for rate in rates:
            reused = family == "GELU" and layers == 2 and rate in (0.006, 0.012, 0.024)
            if reused:
                name = next(pinned)
            else:
                rate_slug = str(rate).replace(".", "p")
                family_slug = family.lower()
                name = (
                    f"g1_rqtune_rqffnact_{family_slug}_l{layers}_w{WIDTHS[family]}_"
                    f"e0p064_d{rate_slug}_b1280_cap40_ts2_r4_500m"
                )
            specs.append(
                RunSpec(
                    family=family,
                    layers=layers,
                    width=WIDTHS[family],
                    deep_lr=rate,
                    name=name,
                    reused=reused,
                )
            )
    return tuple(specs)


def _require_equal(value: object, expected: object, context: str) -> None:
    if value != expected:
        raise ValueError(f"{context}: expected {expected!r}, got {value!r}")


def _load_validation_recall(path: Path, context: str) -> float:
    recalls = [
        float(match.group("recall"))
        for match in VALIDATION_RECALL_PATTERN.finditer(path.read_text())
    ]
    if not recalls:
        raise ValueError(f"{context}: missing epoch/val_true.recall@100 in sweep.log")
    return max(recalls)


def _load_run(directory: Path, spec: RunSpec) -> Run:
    metadata = json.loads((directory / "training_metadata.json").read_text())
    metrics = json.loads((directory / "final_metrics.json").read_text())
    validation_recall = _load_validation_recall(directory / "sweep.log", spec.name)
    transformer = metadata["transfer_invariants"]["transformer"]
    context = spec.name
    _require_equal(metadata["embedding_learning_rate"], EMBEDDING_LR, context)
    _require_equal(metadata["deep_learning_rate"], spec.deep_lr, context)
    _require_equal(metadata["batch_size"], BATCH_SIZE, context)
    _require_equal(metadata["selection_resolved"], True, context)
    _require_equal(
        metadata["transfer_invariants"]["restore_best_weights"], True, context
    )
    _require_equal(metadata["lr_schedule_horizon_epochs"], 20, context)
    _require_equal(metadata["stopped_epoch"], 20, context)
    _require_equal(metadata["transfer_invariants"]["model_dim"], 64, context)
    _require_equal(metadata["transfer_invariants"]["item_embedding_dim"], 64, context)
    _require_equal(metadata["transfer_invariants"]["mup_base_dim"], 16, context)
    _require_equal(metadata["transfer_invariants"]["mup_delta_dim"], 32, context)
    _require_equal(transformer["ffn"], spec.family.lower(), context)
    _require_equal(transformer["ffn_intermediate_dim"], spec.width, context)
    _require_equal(transformer["ffn_dropout"], 0.1, context)
    _require_equal(transformer["num_layers"], spec.layers, context)
    if spec.gated:
        _require_equal(transformer["gated_ffn_dropout"], True, context)
    for metric in METRICS:
        if metrics.get(metric) is None:
            raise ValueError(f"{context}: missing {metric}")
    _require_equal(metrics["num_users"], 37018, context)
    return Run(
        family=spec.family,
        layers=spec.layers,
        width=spec.width,
        embedding_lr=metadata["embedding_learning_rate"],
        deep_lr=metadata["deep_learning_rate"],
        batch_size=metadata["batch_size"],
        best_epoch=metadata["best_epoch"],
        stopped_epoch=metadata["stopped_epoch"],
        horizon_epochs=metadata["lr_schedule_horizon_epochs"],
        validation_recall=validation_recall,
        metrics={metric: metrics[metric] for metric in METRICS},
        name=spec.name,
        reused=spec.reused,
    )


def _artifact_paths(generated: Path, spec: RunSpec) -> tuple[Path, Path, Path]:
    directory = generated / "logs" / spec.name
    return (
        directory / "training_metadata.json",
        directory / "final_metrics.json",
        directory / "sweep.log",
    )


def load_runs(generated: Path, facts: ReportFileFacts | None = None) -> list[Run]:
    specs = expected_specs()
    present = [
        spec
        for spec in specs
        if all(map(Path.exists, _artifact_paths(generated, spec)))
    ]
    if not present:
        return []
    if len(present) != len(specs):
        missing = [spec.name for spec in specs if spec not in present]
        raise ValueError(
            f"RQ4 activation/depth surface is incomplete: {len(present)}/"
            f"{len(specs)} artifacts; missing {', '.join(missing)}"
        )
    if facts is None:
        return [_load_run(generated / "logs" / spec.name, spec) for spec in specs]
    paths = tuple(path for spec in specs for path in _artifact_paths(generated, spec))
    serialized = facts.load_or_compute(
        "g1_rq4_activation_depth_surface_v1",
        paths,
        lambda: [
            asdict(_load_run(generated / "logs" / spec.name, spec)) for spec in specs
        ],
    )
    return [Run(**values) for values in serialized]


def selected_runs(runs: list[Run]) -> dict[tuple[str, int], Run]:
    selected: dict[tuple[str, int], Run] = {}
    tested_rates: dict[tuple[str, int], set[float]] = {}
    for run in runs:
        key = (run.family, run.layers)
        tested_rates.setdefault(key, set()).add(run.deep_lr)
        current = selected.get(key)
        if current is None or run.validation_recall > current.validation_recall:
            selected[key] = run
    for key, run in selected.items():
        rates = sorted(tested_rates[key])
        if run.deep_lr in (rates[0], rates[-1]):
            family, layers = key
            raise ValueError(
                f"{family} at {layers} layers selects deep LR {run.deep_lr:g}, "
                "a tested boundary; extend the surface before rendering"
            )
    return selected


def paired_table(runs: list[Run]) -> str:
    selected = selected_runs(runs)
    lines = [
        "| activation | plain FFN recall@100 | gated FFN recall@100 | "
        "plain FFN ndcg@100 | gated FFN ndcg@100 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for plain, gated in (("ReLU", "ReGLU"), ("GELU", "GEGLU"), ("SiLU", "SwiGLU")):
        control = selected[(plain, 2)]
        treatment = selected[(gated, 2)]
        lines.append(
            "| "
            + " | ".join(
                (
                    f"{plain} → {gated}",
                    reporting.absolute(control.metrics["recall@100"]),
                    reporting.change_cell(
                        treatment.metrics["recall@100"],
                        control.metrics["recall@100"],
                        "recall@100",
                    ),
                    reporting.absolute(control.metrics["ndcg@100"]),
                    reporting.change_cell(
                        treatment.metrics["ndcg@100"],
                        control.metrics["ndcg@100"],
                        "ndcg@100",
                    ),
                )
            )
            + " |"
        )
    return "\n".join(lines)


def depth_table(runs: list[Run]) -> str:
    selected = selected_runs(runs)
    lines = [
        "| layers | GELU recall@100 | SwiGLU recall@100 | GELU ndcg@100 | "
        "SwiGLU ndcg@100 |",
        "| ---: | --- | --- | --- | --- |",
    ]
    for layers in (2, 4, 8):
        control = selected[("GELU", layers)]
        treatment = selected[("SwiGLU", layers)]
        lines.append(
            "| "
            + " | ".join(
                (
                    str(layers),
                    reporting.absolute(control.metrics["recall@100"]),
                    reporting.change_cell(
                        treatment.metrics["recall@100"],
                        control.metrics["recall@100"],
                        "recall@100",
                    ),
                    reporting.absolute(control.metrics["ndcg@100"]),
                    reporting.change_cell(
                        treatment.metrics["ndcg@100"],
                        control.metrics["ndcg@100"],
                        "ndcg@100",
                    ),
                )
            )
            + " |"
        )
    return "\n".join(lines)


def reader_tables(runs: list[Run]) -> str:
    return paired_table(runs) + "\n\n" + depth_table(runs)


def _bold(cells: tuple[str, ...], selected: bool) -> tuple[str, ...]:
    if not selected:
        return cells
    return tuple(f"**{cell}**" for cell in cells)


def tuning_report(runs: list[Run]) -> str:
    selected = selected_runs(runs)
    grouped = {(family, layers): [] for family, layers, _ in SURFACES}
    for run in runs:
        grouped[(run.family, run.layers)].append(run)
    lines = [
        "# G1 RQ4 — FFN activation, gating, and depth tuning",
        "",
        "Native Yambda-500M, batch 1280, embedding LR 0.064, linear 20-epoch "
        "horizon under cap 40. Each table bolds its validation-selected "
        "recall@100 row. Plain FFNs use width 171; gated FFNs use width 114 "
        "and enable the same 0.1 internal FFN dropout as the plain arms. All "
        "arms use μP target/base/delta widths 64/16/32.",
    ]
    for family, layers, _ in SURFACES:
        width = WIDTHS[family]
        kind = "gated" if family in GATED_FAMILIES else "plain"
        lines.extend(
            (
                "",
                f"### {family}, {layers} layers ({kind}, width {width})",
                "",
                "| embedding LR | deep LR | batch size | best/stopped/horizon "
                "epoch | validation recall@100 | final recall@100 | final ndcg@100 |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            )
        )
        for run in grouped[(family, layers)]:
            cells = _bold(
                (
                    f"{run.embedding_lr:.3f}",
                    f"{run.deep_lr:.3f}",
                    str(run.batch_size),
                    f"{run.best_epoch}/{run.stopped_epoch}/{run.horizon_epochs}",
                    reporting.absolute(run.validation_recall),
                    reporting.absolute(run.metrics["recall@100"]),
                    reporting.absolute(run.metrics["ndcg@100"]),
                ),
                run == selected[(family, layers)],
            )
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated", type=Path, default=Path("generated"))
    parser.add_argument("--tuning", action="store_true")
    arguments = parser.parse_args()
    runs = load_runs(arguments.generated)
    if len(runs) != len(expected_specs()):
        raise ValueError("RQ4 activation/depth surface is absent")
    print(tuning_report(runs) if arguments.tuning else reader_tables(runs))


if __name__ == "__main__":
    main()
