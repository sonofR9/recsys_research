import os
from pathlib import Path

from experiments.g6_rqkmeans_history.launchers.remediation_bounded import (
    JOB_ENVIRONMENT,
    build_bounded_gate_experiment,
    decode_bounded_gate_job,
    write_bounded_gate_contract,
)


encoded = os.environ.get(JOB_ENVIRONMENT)
if not encoded:
    raise RuntimeError(f"{JOB_ENVIRONMENT} is required")
compiled_job = decode_bounded_gate_job(encoded)
experiment = build_bounded_gate_experiment(compiled_job)
write_bounded_gate_contract(compiled_job, Path(experiment.base_path) / "logs")
