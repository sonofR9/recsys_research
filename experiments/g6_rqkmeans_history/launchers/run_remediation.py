import os
from pathlib import Path

from experiments.g6_rqkmeans_history.launchers.remediation_compiled import (
    JOB_ENVIRONMENT,
    build_remediation_experiment,
    decode_remediation_job,
    write_remediation_contract,
)


encoded = os.environ.get(JOB_ENVIRONMENT)
if not encoded:
    raise RuntimeError(f"{JOB_ENVIRONMENT} is required")
compiled_job = decode_remediation_job(encoded)
experiment = build_remediation_experiment(compiled_job)
write_remediation_contract(compiled_job, Path(experiment.base_path) / "logs")
