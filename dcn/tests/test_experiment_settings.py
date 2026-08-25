import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch

from dcn.config.experiment import Experiment
from dcn.config.generation import (
    ActionGenerationExperiment,
    CombinedSemanticGenerationExperiment,
    GenerationExperiment,
    RqVaeGenerationExperiment,
    SemanticGenerationExperiment,
    TigerExperiment,
    TimeWindowGenerationExperiment,
)
from dcn.config.ranking import (
    HomeworkRankingExperiment,
    RankingExperiment,
    RankingWithHistoryExperiment,
    SemanticRankingExperiment,
)
from dcn.config.settings import DataloaderConfig

RANKING = (
    RankingExperiment,
    RankingWithHistoryExperiment,
    SemanticRankingExperiment,
    HomeworkRankingExperiment,
)
GENERATION = (
    GenerationExperiment,
    TimeWindowGenerationExperiment,
    ActionGenerationExperiment,
    SemanticGenerationExperiment,
    TigerExperiment,
    CombinedSemanticGenerationExperiment,
    RqVaeGenerationExperiment,
)


@dataclass
class _Concrete(Experiment):
    def create_dataset_source(self) -> Any: ...
    def create_counters(self) -> list: ...
    def create_criterion(self) -> Any: ...
    def create_optimizers(self) -> Any: ...
    def _create_model(self) -> Any: ...


@dataclass
class _Declares(_Concrete):
    def settings_defaults(self) -> dict[str, Any]:
        return {**super().settings_defaults(), "dataloader": DataloaderConfig(7, 7)}


@dataclass
class _InheritsOnly(_Concrete):
    pass


class TestSettingsResolution:
    @pytest.mark.parametrize(
        "bases", [(_Declares, _InheritsOnly), (_InheritsOnly, _Declares)]
    )
    def test_a_sibling_that_only_inherits_never_wins(self, bases: tuple) -> None:
        """The hazard this mechanism exists for: dataclass *fields* resolve in
        reverse MRO, so a base inheriting the framework default would win by
        sitting earlier. Method resolution does not have that ordering."""
        variant = dataclass(type("Variant", bases, {"__annotations__": {}}))

        assert variant().dataloader.batch_size == 7

    def test_a_script_can_still_override_a_group(self) -> None:
        assert (
            RankingExperiment(dataloader=DataloaderConfig(3, 3)).dataloader.batch_size
            == 3
        )

    def test_every_ranking_variant_trains_under_the_same_loader(self) -> None:
        loaders = {
            (variant().dataloader.batch_size, variant().dataloader.val_batch_size)
            for variant in RANKING
        }

        assert loaders == {(64, 64)}

    def test_the_two_token_variant_halves_the_batch_and_the_rest_do_not(self) -> None:
        batches = {
            variant.__name__: variant().dataloader.batch_size for variant in GENERATION
        }

        assert batches.pop("ActionGenerationExperiment") == 64
        assert set(batches.values()) == {128}

    @pytest.mark.parametrize("variant", GENERATION)
    def test_generation_prebuilds_cached_data_before_parallel_setup(
        self, variant: type
    ) -> None:
        assert variant().prebuilds_runner_data
        assert not variant(invalidate_cache=True).prebuilds_runner_data

    @pytest.mark.parametrize("variant", RANKING + GENERATION)
    def test_no_variant_falls_back_to_the_framework_day_range(
        self, variant: type
    ) -> None:
        assert variant().day_range.end_day == 299


def test_visible_gpu_has_its_own_training_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-c")

    experiment = _Concrete(base_path=tmp_path)

    assert experiment.gpu_lock_path == tmp_path / "gpu-GPU-c.lock"
    assert experiment.gpu_gate_path is None


def test_queue_training_slot_has_its_own_training_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a")
    monkeypatch.setenv("DCN_GPU_LOCK_DEVICE", "GPU-a")
    monkeypatch.setenv("DCN_GPU_LOCK_SLOT", "2")

    experiment = _Concrete(base_path=tmp_path)

    assert experiment.gpu_lock_path == tmp_path / "gpu-GPU-a-slot-2.lock"
    assert experiment.gpu_gate_path == tmp_path / "gpu-GPU-a.lock"
    assert experiment.gpu_gate_shared


def test_queue_builds_runner_components_on_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setenv("DCN_GPU_LOCK_SLOT", "0")

    assert _Concrete().runner_build_device == torch.device("cpu")

    monkeypatch.delenv("DCN_GPU_LOCK_SLOT")

    assert _Concrete().runner_build_device == torch.device("cuda")


def test_queue_prebuild_warms_cuda_context_without_moving_the_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initialized = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "init", lambda: initialized.append(True))
    monkeypatch.setenv("DCN_GPU_LOCK_SLOT", "0")
    experiment = _Concrete()
    experiment.__dict__["base_model"] = object()
    experiment.__dict__["callbacks"] = object()

    experiment.prebuild_runner_components()

    assert initialized == [True]


def test_default_cuda_device_uses_the_physical_gpu_zero_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binaries = tmp_path / "bin"
    binaries.mkdir()
    nvidia_smi = binaries / "nvidia-smi"
    nvidia_smi.write_text("#!/usr/bin/env bash\nprintf 'GPU-zero\\n'\n")
    nvidia_smi.chmod(0o755)
    monkeypatch.setenv("PATH", f"{binaries}:{os.environ['PATH']}")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    experiment = _Concrete(base_path=tmp_path)

    assert experiment.gpu_lock_path == tmp_path / "gpu-GPU-zero.lock"
