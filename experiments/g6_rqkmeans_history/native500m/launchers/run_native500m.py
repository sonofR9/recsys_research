from pathlib import Path

from experiments.g6_rqkmeans_history.native500m.launchers.runtime import (
    build_experiment,
    load_runtime_job,
    write_run_contract,
)


manifest, job = load_runtime_job()
experiment = build_experiment(job)
write_run_contract(Path(experiment.base_path) / "logs", manifest, job)
