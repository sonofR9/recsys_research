import fcntl
from pathlib import Path

import pytest

from utils.locks import hold


def test_shared_holders_block_an_exclusive_holder(tmp_path: Path) -> None:
    path = tmp_path / "gpu.lock"

    with hold(path, "shared gpu", shared=True):
        with hold(path, "second shared gpu", shared=True):
            with open(path) as handle:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
