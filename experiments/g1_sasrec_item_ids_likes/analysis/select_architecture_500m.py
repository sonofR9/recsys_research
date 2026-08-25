from __future__ import annotations

import argparse
import math
from pathlib import Path

from experiments.g1_sasrec_item_ids_likes.analysis import collect


def _selection(raw: str) -> tuple[str, tuple[float, float]]:
    fields = raw.split(":")
    if len(fields) != 3 or not fields[0]:
        raise ValueError(f"invalid selection {raw!r}")
    try:
        rates = (float(fields[1]), float(fields[2]))
    except ValueError as error:
        raise ValueError(f"invalid selection {raw!r}") from error
    if any(not math.isfinite(rate) or rate <= 0 for rate in rates):
        raise ValueError(f"invalid selection {raw!r}")
    return fields[0], rates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--selection", action="append", default=[])
    parser.add_argument(
        "--exploratory-selection",
        action="append",
        default=[],
        help="confirm this base as evidence beside RQ4 rather than as its selection",
    )
    arguments = parser.parse_args()
    if not arguments.selection and not arguments.exploratory_selection:
        parser.error("at least one --selection is required")
    requested: dict[str, tuple[float, float]] = {}
    exploratory: set[str] = set()
    try:
        for raw in arguments.selection + arguments.exploratory_selection:
            base, rates = _selection(raw)
            if base in requested and requested[base] != rates:
                raise ValueError(f"conflicting selections for {base}")
            requested[base] = rates
        for raw in arguments.exploratory_selection:
            exploratory.add(_selection(raw)[0])
        collect.GENERATED = arguments.generated
        requested_bases = set(requested)
        includes_ffn = any(
            base.startswith(("ffn_gelu", "ffn_swiglu"))
            for base in requested_bases
        )

        def relevant_base(base: str) -> bool:
            if base in {"architecture_control", "control_control", "sequence_128"}:
                return True
            if base in requested_bases:
                return True
            return includes_ffn and base.startswith(("ffn_gelu", "ffn_swiglu"))

        runs = collect.load_report_runs(
            "50m", configuration_base_filter=relevant_base
        )
        selected = collect.validate_architecture_final_selections(
            runs, requested, exploratory_bases=frozenset(exploratory)
        )
    except ValueError as error:
        parser.error(str(error))
    for base, winner in sorted(selected.items()):
        embedding_lr, deep_lr = collect._run_rates(winner)
        print(
            "\t".join(
                (
                    base,
                    winner.name,
                    f"{embedding_lr:g}",
                    f"{deep_lr:g}",
                    str(collect._run_batch_size(winner)),
                )
            )
        )


if __name__ == "__main__":
    main()
