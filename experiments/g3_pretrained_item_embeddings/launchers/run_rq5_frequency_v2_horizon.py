from pathlib import Path

from experiments.g3_pretrained_item_embeddings.launchers.rq5_frequency_v2_horizon import (
    build_rq5_frequency_v2_horizon_training_experiment,
    compiled_rq5_frequency_v2_horizon_job_from_environment,
    write_rq5_frequency_v2_horizon_job_contract,
)


compiled, ledger, ledger_path, feature_path = (
    compiled_rq5_frequency_v2_horizon_job_from_environment()
)
experiment = build_rq5_frequency_v2_horizon_training_experiment(
    compiled,
    ledger=ledger,
    feature_data_path=feature_path,
)
write_rq5_frequency_v2_horizon_job_contract(
    compiled,
    ledger,
    ledger_path,
    Path(experiment.base_path) / "logs",
)
