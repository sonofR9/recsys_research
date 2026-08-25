import os
from pathlib import Path

from experiments.g2_esasrec.launchers.compiled import compiled_job_from_environment
from experiments.g2_esasrec.launchers.selected_benchmark import (
    build_selected_benchmark_experiment,
)

compiled_job = compiled_job_from_environment()
raw_destination = os.environ.get("G2_BENCHMARK_OUTPUT")
if raw_destination is None:
    raise RuntimeError("G2_BENCHMARK_OUTPUT is required")
experiment = build_selected_benchmark_experiment(
    compiled_job,
    Path(raw_destination),
)
