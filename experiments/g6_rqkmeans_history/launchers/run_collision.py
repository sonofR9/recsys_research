from pathlib import Path

from experiments.g6_rqkmeans_history.configs.collision_policy import (
    build_collision_policy_experiment,
)
from experiments.g6_rqkmeans_history.launchers.collision_runtime import (
    collision_job_from_environment,
    write_collision_contract,
)


job = collision_job_from_environment()
experiment = build_collision_policy_experiment(job)
write_collision_contract(job, Path(experiment.base_path) / "logs")
