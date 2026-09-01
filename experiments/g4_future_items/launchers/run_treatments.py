from pathlib import Path

from experiments.g4_future_items.launchers.compiled import (
    build_training_experiment,
    compiled_job_from_environment,
    write_job_contract,
)


compiled_job, ledger_path = compiled_job_from_environment()
experiment = build_training_experiment(compiled_job)
write_job_contract(compiled_job, ledger_path, Path(experiment.base_path) / "logs")
