from pathlib import Path

from experiments.g6_rqkmeans_history.configs.rq1 import (
    build_rq1_search_experiment,
)
from experiments.g6_rqkmeans_history.launchers.rq1_runtime import (
    rq1_job_from_environment,
    write_rq1_contract,
)


job = rq1_job_from_environment()
experiment = build_rq1_search_experiment(job)
write_rq1_contract(job, Path(experiment.base_path) / "logs")
