from pathlib import Path

from experiments.g2_esasrec.launchers.compiled import (
    build_local_experiment,
    compiled_job_from_environment,
    write_job_contract,
)

compiled_job = compiled_job_from_environment()
experiment = build_local_experiment(compiled_job)
write_job_contract(compiled_job, Path(experiment.base_path) / "logs")
