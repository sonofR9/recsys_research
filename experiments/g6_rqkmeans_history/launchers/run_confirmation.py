from pathlib import Path

from experiments.g6_rqkmeans_history.configs.confirmation import (
    build_confirmation_experiment,
)
from experiments.g6_rqkmeans_history.launchers.confirmation_runtime import (
    job_from_environment,
    write_contract,
)


job, manifest = job_from_environment()
experiment = build_confirmation_experiment(job)
write_contract(job, manifest, Path(experiment.base_path) / "logs")
