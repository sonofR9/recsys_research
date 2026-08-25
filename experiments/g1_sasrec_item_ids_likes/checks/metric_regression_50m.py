import os
from dataclasses import replace

os.environ["G1_VARIANT"] = "baseline"
os.environ["G1_DATASET_SIZE"] = "50m"
os.environ.pop("G1_MAX_USERS", None)
os.environ.pop("G1_VAL_BATCH_SIZE", None)

from experiments.g1_sasrec_item_ids_likes.configs.variant import VARIANTS  # noqa: E402


baseline = VARIANTS["baseline"]
batch_size = int(os.environ.get("G1_REGRESSION_BATCH_SIZE", "128"))
run_name = os.environ.get(
    "G1_METRIC_REGRESSION_RUN_NAME",
    f"g1_metric_regression_50m_b{batch_size}_s0",
)
experiment = replace(
    baseline,
    run_name=run_name,
    seed=0,
    dataloader=replace(
        baseline.dataloader,
        batch_size=batch_size,
    ),
)
