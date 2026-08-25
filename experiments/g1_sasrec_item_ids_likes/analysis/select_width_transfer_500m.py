from __future__ import annotations

import argparse
from pathlib import Path

from experiments.g1_sasrec_item_ids_likes.analysis import collect


APPROVED_WIDTHS = (16, 256)
APPROVED_RATES = (0.032, 0.012)
APPROVED_BATCH_SIZE = 1280


def select_width_transfer_confirmations(
    proxy_runs: list[collect.ReportRun],
) -> list[tuple[int, str, float, float, int]]:
    rows = []
    for width in APPROVED_WIDTHS:
        candidates = {
            run.name: run
            for run in proxy_runs
            if run.status == "completed"
            and run.research_question == 8
            and collect._manifest_base(run.configuration) == f"dimension_{width}"
            and collect._run_rates(run) == APPROVED_RATES
            and collect._run_batch_size(run) == APPROVED_BATCH_SIZE
        }
        if len(candidates) != 1:
            raise ValueError(
                f"RQ1 width {width} requires exactly one compatible native-50M "
                "RQ8 dimension artifact at approved rates 0.032/0.012 and "
                "batch 1280"
            )
        run = next(iter(candidates.values()))
        expected_metadata = {
            ("transfer_invariants", "experiment_class"): (
                "MuTransferGenerationExperiment"
            ),
            ("transfer_invariants", "mup_base_dim"): 16,
            ("transfer_invariants", "mup_delta_dim"): 32,
            ("item_embedding_dim",): 64,
            ("model_dim",): width,
        }
        mismatches = [
            ".".join(path)
            for path, expected in expected_metadata.items()
            if collect._nested_value(run.metadata, path) != expected
        ]
        if mismatches:
            raise ValueError(
                f"{run.name}: invalid RQ1 width-transfer metadata: "
                + ", ".join(mismatches)
            )
        rows.append(
            (width, run.name, *APPROVED_RATES, APPROVED_BATCH_SIZE)
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        collect.GENERATED = arguments.generated
        approved_bases = {f"dimension_{width}" for width in APPROVED_WIDTHS}
        proxy_runs = collect.load_report_runs(
            "50m",
            research_question=8,
            configuration_base_filter=approved_bases.__contains__,
        )
        rows = select_width_transfer_confirmations(proxy_runs)
    except ValueError as error:
        parser.error(str(error))
    for row in rows:
        print("\t".join(str(value) for value in row))


if __name__ == "__main__":
    main()
