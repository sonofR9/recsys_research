import os
from typing import cast

from experiments.g2_esasrec.configs.local import ComponentMethod
from experiments.g2_esasrec.configs.smoke import SMOKE_METHODS, build_smoke

method = os.environ.get("G2_SMOKE_METHOD")
if method not in SMOKE_METHODS:
    raise ValueError(f"G2_SMOKE_METHOD must be one of {SMOKE_METHODS}")

experiment = build_smoke(cast(ComponentMethod, method))
