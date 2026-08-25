from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from dcn.config import Experiment
from utils.locks import hold

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _canonical_positive(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default))
    if not raw.isdigit() or int(raw) < 1 or raw != str(int(raw)):
        raise ValueError(f"{name} must be a canonical positive integer")
    return int(raw)


@dataclass
class RecToolsExperiment(Experiment):
    rectools_python: Path = Path()
    max_epochs: int = 100

    @property
    def runner_path(self) -> Path:
        return PROJECT_ROOT / "experiments/g2_esasrec/official/run_official.py"

    def create_dataset_source(self):
        raise NotImplementedError

    def create_counters(self):
        raise NotImplementedError

    def _create_model(self):
        raise NotImplementedError

    def create_criterion(self):
        raise NotImplementedError

    def create_optimizers(self):
        raise NotImplementedError

    @classmethod
    def from_environment(cls) -> "RecToolsExperiment":
        interpreter = os.environ.get("G2_RECTOOLS_PYTHON")
        if interpreter is None:
            raise ValueError("G2_RECTOOLS_PYTHON must name the RecTools interpreter")
        path = Path(interpreter)
        if not path.is_file():
            raise ValueError(f"G2_RECTOOLS_PYTHON is not a file: {path}")
        seed = _canonical_positive("G2_OFFICIAL_SEED", 42)
        max_epochs = _canonical_positive("G2_OFFICIAL_MAX_EPOCHS", 100)
        return cls(
            run_name=f"g2_official_esasrec_50m_s{seed}",
            seed=seed,
            rectools_python=path,
            max_epochs=max_epochs,
        )

    def run(self) -> None:
        self._wait_for_training_release()
        with hold(self.gpu_lock_path, "gpu"):
            with hold(self.gpu_gate_path, "gpu gate", shared=True):
                self._wait_for_queue_gpu(None)
                subprocess.run(
                    [
                        str(self.rectools_python),
                        str(self.runner_path),
                        "--run-name",
                        self.run_name,
                        "--seed",
                        str(self.seed),
                        "--max-epochs",
                        str(self.max_epochs),
                    ],
                    cwd=PROJECT_ROOT,
                    check=True,
                )
