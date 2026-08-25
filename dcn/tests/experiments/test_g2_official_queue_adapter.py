import fcntl
from pathlib import Path
import subprocess
import sys

from experiments.g2_esasrec.official.queued import RecToolsExperiment


def test_queue_adapter_runs_the_pinned_official_recipe_under_the_gpu_lock(
    monkeypatch, tmp_path: Path
) -> None:
    marker = tmp_path / "prepared"
    release = tmp_path / "release"
    release.touch()
    monkeypatch.setenv("DCN_PREPARED_MARKER", str(marker))
    monkeypatch.setenv("DCN_TRAINING_RELEASE", str(release))
    monkeypatch.setenv("DCN_GPU_LOCK_DEVICE", "GPU-test")
    monkeypatch.setenv("DCN_GPU_LOCK_SLOT", "0")
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        with (tmp_path / "gpu-GPU-test-slot-0.lock").open("a+") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                pass
            else:
                raise AssertionError("official runner executed outside the GPU lock")
        calls.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    experiment = RecToolsExperiment(
        run_name="g2_official_esasrec_50m_s43",
        seed=43,
        base_path=tmp_path,
        rectools_python=Path(sys.executable),
    )

    experiment.run()

    assert marker.is_file()
    assert calls == [
        [
            sys.executable,
            str(experiment.runner_path),
            "--run-name",
            experiment.run_name,
            "--seed",
            "43",
            "--max-epochs",
            "100",
        ]
    ]
    assert (tmp_path / "gpu-GPU-test-slot-0.lock").is_file()


def test_queue_adapter_environment_requires_the_rectools_interpreter(
    monkeypatch,
) -> None:
    monkeypatch.delenv("G2_RECTOOLS_PYTHON", raising=False)

    try:
        RecToolsExperiment.from_environment()
    except ValueError as error:
        assert "G2_RECTOOLS_PYTHON" in str(error)
    else:
        raise AssertionError("missing RecTools interpreter was accepted")
