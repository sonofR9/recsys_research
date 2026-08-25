import argparse
import runpy
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-lr", type=float, required=True)
    parser.add_argument("--deep-lr", type=float, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    arguments = parser.parse_args()

    collect_path = Path(__file__).with_name("collect.py")
    namespace = runpy.run_path(str(collect_path))
    logs = namespace["GENERATED"] / "logs"
    control_directories = [
        directory
        for pattern in (
            "g1_rqtune_architecture_control_*_50m",
            "g1_rqtune_control_control_*_50m",
        )
        for directory in logs.glob(pattern)
    ]
    runs = namespace["load_report_runs"](
        "50m", research_question=8, directories=control_directories
    )
    control_runs = [
        run
        for run in runs
        if namespace["_manifest_base"](run.configuration) == "sequence_128"
    ]
    try:
        winner = namespace["_control_proxy_winner"](
            "global architecture control", control_runs
        )
    except ValueError as error:
        raise SystemExit(str(error)) from None
    selected = (
        namespace["_run_rates"](winner),
        namespace["_run_batch_size"](winner),
    )
    expected = (
        (round(arguments.embedding_lr, 12), round(arguments.deep_lr, 12)),
        arguments.batch_size,
    )
    if selected != expected:
        raise SystemExit(
            f"global batch selection mismatch: selected={selected}, expected={expected}"
        )


if __name__ == "__main__":
    main()
