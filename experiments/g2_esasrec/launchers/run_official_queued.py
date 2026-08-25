import os
from pathlib import Path

from experiments.g2_esasrec.launchers.compiled import (
    build_official_experiment,
    compiled_job_from_environment,
    write_job_contract,
)

compiled_job = compiled_job_from_environment()
interpreter = os.environ.get("G2_RECTOOLS_PYTHON")
if interpreter is None:
    raise RuntimeError("G2_RECTOOLS_PYTHON must name the RecTools 0.19.0 interpreter")
experiment = build_official_experiment(compiled_job, Path(interpreter))
write_job_contract(compiled_job, Path(experiment.base_path) / "logs")
