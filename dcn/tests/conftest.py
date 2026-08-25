import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

# torchrec is imported eagerly by the embedding layer and is not
# installed here.
sys.modules.setdefault("torchrec", MagicMock())
sys.modules.setdefault("torchrec.optim", MagicMock())

from dcn.tests.miniature_yambda import write_miniature_yambda  # noqa: E402
from utils.global_config import config  # noqa: E402


@pytest.fixture
def base_path(tmp_path: Path) -> Path:
    return write_miniature_yambda(tmp_path)


@pytest.fixture(autouse=True)
def verify_training_e2e_device(request: pytest.FixtureRequest) -> None:
    expected = os.environ.get("TEST_EXPECTED_E2E_DEVICE")
    if expected is None or request.node.get_closest_marker("training_e2e") is None:
        return
    actual = "cuda" if torch.cuda.is_available() else "cpu"
    assert actual == expected


@pytest.fixture
def cpu_attention():
    """Run attention through the CPU reference kernel.

    flash-attn is CUDA-only and needs half precision, so any test that wants to
    exercise an attention stack in plain float32 asks for this.
    """
    previous = config.cpu_attention
    config.set_cpu_attention(True)
    yield
    config.set_cpu_attention(previous)
