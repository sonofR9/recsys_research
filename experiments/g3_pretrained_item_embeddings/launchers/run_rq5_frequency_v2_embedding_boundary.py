from pathlib import Path

from experiments.g3_pretrained_item_embeddings.launchers.rq5_frequency_v2_embedding_boundary import (
    build_training_experiment,
    compiled_job_from_environment,
    write_job_contract,
)


compiled, ledger, ledger_path, feature_path = compiled_job_from_environment()
experiment = build_training_experiment(
    compiled,
    ledger=ledger,
    feature_data_path=feature_path,
)
write_job_contract(
    compiled,
    ledger,
    ledger_path,
    Path(experiment.base_path) / "logs",
)
