import os
from pathlib import Path

from experiments.g1_aggregate_dataset_size.configs.fixed26_calibration import (
    build_fixed26_experiment,
)
from experiments.g1_aggregate_dataset_size.launchers.fixed26_calibration import (
    verify_experiment_contract,
    verify_worker_contract,
    write_job_contract,
)
from experiments.g1_aggregate_dataset_size.protocol.fixed26_calibration import (
    load_fixed26_manifest,
)


manifest = load_fixed26_manifest()
verify_worker_contract(Path("generated/logs").resolve(), manifest, Path(__file__))
job = manifest.job_by_run(os.environ["G1_FIXED26_CALIBRATION_RUN"])
experiment = build_fixed26_experiment(job, manifest)
verify_experiment_contract(experiment, manifest)
write_job_contract(Path(experiment.base_path) / "logs", job, manifest)
