from pathlib import Path

from experiments.g6_rqkmeans_history.launchers.compiled import (
    build_experiment,
    compiled_job_from_environment,
    write_job_contract,
)


compiled_job = compiled_job_from_environment()
experiment = build_experiment(compiled_job)
write_job_contract(compiled_job, Path(experiment.base_path) / "logs")
