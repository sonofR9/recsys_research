import os
from dataclasses import replace
from importlib import reload

os.environ["G1_VARIANT"] = "baseline"
os.environ.setdefault("G1_DATASET_SIZE", "500m")
os.environ.pop("G1_MAX_USERS", None)
os.environ.pop("G1_VAL_BATCH_SIZE", None)

from experiments.g1_sasrec_item_ids_likes.configs import variant as variant_module  # noqa: E402


baseline = reload(variant_module).VARIANTS["baseline"]
seed = os.environ.get("G1_SEED")
experiment = (
    replace(
        baseline,
        seed=int(seed),
        run_name=f"{baseline.run_name}_s{seed}",
    )
    if seed is not None
    else baseline
)
