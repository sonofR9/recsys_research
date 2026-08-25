import os
from pathlib import Path
import subprocess

import pytest

from experiments.g2_esasrec.configs.smoke import SMOKE_METHODS, build_smoke


ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("method", SMOKE_METHODS)
def test_smoke_covers_each_layer_loss_pair_with_user_id_sampling(method: str) -> None:
    experiment = build_smoke(method)

    assert experiment.user_sample is not None
    assert experiment.user_sample.max_users == 2_000
    assert experiment.user_sample.seed == 42
    assert experiment.user_sample.fraction is None
    assert experiment.num_epochs == 1
    assert experiment.run_name == f"g2_smoke_{method}_2000users_seed42"
    assert experiment.transformer.ffn_intermediate_dim % 32 == 0


def test_smoke_rejects_unapproved_component() -> None:
    with pytest.raises(ValueError, match="approved G2 smoke"):
        build_smoke("matched_standard_gbce")


def test_smoke_queue_checks_persistent_service_before_loading_queue(tmp_path) -> None:
    project = tmp_path / "project"
    source = ROOT / "experiments/g2_esasrec/launchers/queue_smokes.sh"
    launcher = project / "experiments/g2_esasrec/launchers/queue_smokes.sh"
    launcher.parent.mkdir(parents=True)
    launcher.write_text(source.read_text())
    queue = project / "utils/training_queue/queue.sh"
    queue.parent.mkdir(parents=True)
    marker = tmp_path / "queue-sourced"
    queue.write_text(
        f"printf sourced > {marker!s}\n" "enqueue() { :; }\n" "drain() { :; }\n"
    )
    (queue.parent / "service.py").touch()
    executable_directory = tmp_path / "bin"
    executable_directory.mkdir()
    python = executable_directory / "python"
    python.write_text("#!/usr/bin/env bash\nexit 17\n")
    python.chmod(0o755)

    result = subprocess.run(
        ["bash", str(launcher)],
        env=os.environ | {"PATH": f"{executable_directory}:{os.environ['PATH']}"},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 17
    assert not marker.exists()


def test_smoke_queue_pins_offline_wandb_mode() -> None:
    launcher = ROOT / "experiments/g2_esasrec/launchers/queue_smokes.sh"

    assert '"WANDB_MODE=offline"' in launcher.read_text()
