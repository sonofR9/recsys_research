from pathlib import Path

from experiments.g3_pretrained_item_embeddings.launchers.rq2_id_boundary import (
    build_training_experiment,
    compiled_rq2_id_boundary_job_from_environment,
    write_job_contract,
)


compiled_job, ledger_path, feature_data_path = (
    compiled_rq2_id_boundary_job_from_environment()
)
experiment = build_training_experiment(
    compiled_job,
    feature_data_path=feature_data_path,
)
write_job_contract(compiled_job, ledger_path, Path(experiment.base_path) / "logs")
