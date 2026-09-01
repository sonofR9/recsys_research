from pathlib import Path

from experiments.g3_pretrained_item_embeddings.launchers.rq3_boundary import (
    build_rq3_boundary_training_experiment,
    compiled_rq3_boundary_job_from_environment,
    write_rq3_boundary_job_contract,
)


compiled_job, ledger, ledger_path, feature_data_path = (
    compiled_rq3_boundary_job_from_environment()
)
experiment = build_rq3_boundary_training_experiment(
    compiled_job,
    ledger=ledger,
    feature_data_path=feature_data_path,
)
write_rq3_boundary_job_contract(
    compiled_job,
    ledger,
    ledger_path,
    Path(experiment.base_path) / "logs",
)
