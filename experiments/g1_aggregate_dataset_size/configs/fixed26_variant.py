import os

from experiments.g1_aggregate_dataset_size.configs.fixed26_calibration import (
    build_fixed26_experiment,
)
from experiments.g1_aggregate_dataset_size.protocol.fixed26_calibration import (
    load_fixed26_manifest,
)


manifest = load_fixed26_manifest()
job = manifest.job_by_run(os.environ["G1_FIXED26_CALIBRATION_RUN"])
experiment = build_fixed26_experiment(job, manifest)
