import os
from dataclasses import dataclass, fields, replace
from pathlib import Path
from time import monotonic_ns

os.environ["G1_VARIANT"] = "selected_quality_b1280"
os.environ["G1_DATASET_SIZE"] = "50m"
os.environ.pop("G1_MAX_USERS", None)

from experiments.g1_sasrec_item_ids_likes.configs.variant import VARIANTS  # noqa: E402
from dcn.config import GenerationExperiment, TrainingCallbacks  # noqa: E402
from neuralrec.run.callbacks.base import Callback  # noqa: E402


class UtilizationWindow(Callback):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.start: int | None = None

    def on_step_end(self, state, batch, out) -> None:
        runner = state["train_runner"]
        device = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if self.start is None:
            self.start = monotonic_ns()
        if runner.step + 1 == len(runner.train_loader):
            self.path.write_text(
                f"start {self.start} {device}\nend {monotonic_ns()} {device}\n"
            )


@dataclass
class UtilizationExperiment(GenerationExperiment):
    def create_callbacks(self) -> TrainingCallbacks:
        callbacks = super().create_callbacks()
        marker = os.environ.get("G1_UTIL_MARKER")
        if marker is not None:
            callbacks.all.insert(0, UtilizationWindow(Path(marker)))
        return callbacks

selected = VARIANTS["selected_quality_b1280"]
settings = {
    field.name: getattr(selected, field.name)
    for field in fields(selected)
    if field.init
}
settings.update(
    run_name=os.environ.get(
        "G1_UTIL_RUN_NAME", "g1_utilization_regression_50m"
    ),
    num_epochs=1,
    eval_every_n_epochs=1,
    dataloader=replace(
        selected.dataloader,
        batch_size=int(os.environ.get("G1_UTIL_BATCH_SIZE", "1280")),
    ),
)
experiment = UtilizationExperiment(**settings)
