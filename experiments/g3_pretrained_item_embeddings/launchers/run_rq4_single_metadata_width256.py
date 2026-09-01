from pathlib import Path

from experiments.g3_pretrained_item_embeddings.launchers.rq4_single_metadata_width256 import (
    build_training_experiment,
    compiled_job_from_environment,
    write_job_contract,
)


compiled_job, ledger, ledger_path, feature_data_path = compiled_job_from_environment()
experiment = build_training_experiment(
    compiled_job,
    ledger=ledger,
    feature_data_path=feature_data_path,
)
write_job_contract(
    compiled_job,
    ledger,
    ledger_path,
    Path(experiment.base_path) / "logs",
)
