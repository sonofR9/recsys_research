from pathlib import Path

from experiments.g6_rqkmeans_history.configs.collision_recovery import (
    build_collision_recovery_experiment,
)
from experiments.g6_rqkmeans_history.launchers.collision_recovery_runtime import (
    recovery_job_from_environment,
    write_recovery_contract,
)


job = recovery_job_from_environment()
experiment = build_collision_recovery_experiment(job)
write_recovery_contract(job, Path(experiment.base_path) / "logs")
