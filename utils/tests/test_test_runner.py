import os
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "test.sh"


def _fake_pytest(tmp_path: Path) -> tuple[Path, Path]:
    calls = tmp_path / "calls"
    executable = tmp_path / "pytest"
    executable.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ " $* " == *" --collect-only "* ]]; then\n'
        '    if [[ " $* " == *" -m training_e2e "* ]]; then\n'
        '        printf \'%s\\n\' "${TEST_E2E_ITEMS:-dcn/tests/test_main_e2e.py::test_trains}"\n'
        "    else\n"
        '        printf \'%s\\n\' "${TEST_ITEMS:-}"\n'
        "    fi\n"
        '    exit "${TEST_COLLECTION_STATUS:-0}"\n'
        "fi\n"
        'printf \'%s\\n\' "$*" >> "${TEST_INVOCATIONS:?}"\n'
        'if [[ -n "${TEST_DEVICE_INVOCATIONS:-}" ]]; then\n'
        '    printf \'%s|%s|%s\\n\' "${CUDA_VISIBLE_DEVICES-}" '
        '"${TEST_EXPECTED_E2E_DEVICE-}" "$*" >> "$TEST_DEVICE_INVOCATIONS"\n'
        "fi\n"
        'if [[ -n "${TEST_EXPECTED_GPU_GATE:-}" && '
        '-n "${CUDA_VISIBLE_DEVICES:-}" ]]; then\n'
        '    if flock -n "$TEST_EXPECTED_GPU_GATE" -c true; then exit 9; fi\n'
        "fi\n"
        '[[ -z "${TEST_FAIL_ON:-}" || " $* " != *" ${TEST_FAIL_ON} "* ]]\n'
    )
    executable.chmod(0o755)
    return executable, calls


def test_default_run_splits_non_gpu_suite_between_workers(tmp_path: Path) -> None:
    pytest, calls = _fake_pytest(tmp_path)
    items = [
        "dcn/tests/test_generation_e2e.py::test_generation_variant_trains[next_like]",
        "dcn/tests/test_generation_e2e.py::test_generation_variant_trains[semantic]",
        "dcn/tests/test_locks.py::test_lock_is_released",
        "utils/tests/test_negatives_generator.py::test_negative_count",
    ]
    environment = os.environ | {
        "PYTEST": str(pytest),
        "TEST_INVOCATIONS": str(calls),
        "TEST_ITEMS": "\n".join(items),
        "TEST_JOBS": "2",
    }

    result = subprocess.run(
        [RUNNER], cwd=ROOT, env=environment, text=True, capture_output=True
    )

    assert result.returncode == 0, result.stdout + result.stderr
    invocations = [
        call
        for call in calls.read_text().splitlines()
        if "-m training_e2e" not in call
    ]
    assert len(invocations) == 2
    assert all("-m not slow_gpu" in invocation for invocation in invocations)
    assert all(sum(item in invocation for invocation in invocations) == 1 for item in items)


def test_run_uses_up_to_eight_configured_workers(tmp_path: Path) -> None:
    pytest, calls = _fake_pytest(tmp_path)
    items = [f"dcn/tests/test_example.py::test_case_{index}" for index in range(10)]
    environment = os.environ | {
        "PYTEST": str(pytest),
        "TEST_INVOCATIONS": str(calls),
        "TEST_ITEMS": "\n".join(items),
        "TEST_JOBS": "8",
    }

    result = subprocess.run(
        [RUNNER], cwd=ROOT, env=environment, text=True, capture_output=True
    )

    assert result.returncode == 0, result.stdout + result.stderr
    invocations = [
        call
        for call in calls.read_text().splitlines()
        if "-m training_e2e" not in call
    ]
    assert len(invocations) == 8


def test_default_uses_sixteen_workers_capped_by_item_count(tmp_path: Path) -> None:
    pytest, calls = _fake_pytest(tmp_path)
    items = [f"dcn/tests/test_example.py::test_case_{index}" for index in range(10)]
    environment = os.environ | {
        "PYTEST": str(pytest),
        "TEST_INVOCATIONS": str(calls),
        "TEST_ITEMS": "\n".join(items),
    }

    result = subprocess.run(
        [RUNNER], cwd=ROOT, env=environment, text=True, capture_output=True
    )

    assert result.returncode == 0, result.stdout + result.stderr
    cpu_invocations = [
        call
        for call in calls.read_text().splitlines()
        if "-m training_e2e" not in call
    ]
    assert len(cpu_invocations) == len(items)


def test_gpu_override_runs_training_items_serially_beside_cpu_workers(
    tmp_path: Path,
) -> None:
    pytest, calls = _fake_pytest(tmp_path)
    devices = tmp_path / "devices"
    nvidia_smi = tmp_path / "nvidia-smi"
    gpu_uuid = f"GPU-{tmp_path.name}"
    gpu_gate = ROOT / "generated" / f"gpu-{gpu_uuid}.lock"
    nvidia_smi.write_text(
        "#!/usr/bin/env bash\n"
        f'if [[ " $* " == *" --query-gpu=uuid "* ]]; then echo {gpu_uuid}; fi\n'
        'if [[ " $* " == *" --query-gpu=utilization.gpu,"* ]]; then '
        'echo "0, 4, 81920"; fi\n'
    )
    nvidia_smi.chmod(0o755)
    environment = os.environ | {
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "PYTEST": str(pytest),
        "TEST_INVOCATIONS": str(calls),
        "TEST_DEVICE_INVOCATIONS": str(devices),
        "TEST_EXPECTED_GPU_GATE": str(gpu_gate),
        "TEST_ITEMS": "dcn/tests/test_locks.py::test_lock",
        "TEST_E2E_ITEMS": "\n".join(
            [
                "dcn/tests/test_generation_e2e.py::test_trains[a]",
                "dcn/tests/test_generation_e2e.py::test_trains[b]",
            ]
        ),
        "TEST_E2E_DEVICE": "gpu",
        "TEST_GPU": "4",
        "TEST_GPU_CHECK_SECONDS": "0",
        "TEST_JOBS": "2",
    }

    try:
        result = subprocess.run(
            [RUNNER], cwd=ROOT, env=environment, text=True, capture_output=True
        )
    finally:
        gpu_gate.unlink(missing_ok=True)

    assert result.returncode == 0, result.stdout + result.stderr
    invocations = devices.read_text().splitlines()
    gpu = [line for line in invocations if "-m training_e2e" in line]
    cpu = [line for line in invocations if "-m not slow_gpu and not training_e2e" in line]
    assert len(gpu) == 1, result.stdout + result.stderr
    assert gpu[0].startswith("4|cuda|")
    assert len(cpu) == 1
    assert cpu[0].startswith("|cpu|")


def test_focused_run_forwards_arguments_to_one_non_gpu_pytest(tmp_path: Path) -> None:
    pytest, calls = _fake_pytest(tmp_path)
    environment = os.environ | {
        "PYTEST": str(pytest),
        "TEST_INVOCATIONS": str(calls),
    }

    result = subprocess.run(
        [RUNNER, "dcn/tests/test_locks.py", "-x"],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    invocation = calls.read_text().strip()
    assert invocation == "-m not slow_gpu dcn/tests/test_locks.py -x"


def test_default_run_fails_after_a_worker_fails(tmp_path: Path) -> None:
    pytest, calls = _fake_pytest(tmp_path)
    failed_item = "dcn/tests/test_example.py::test_fails"
    environment = os.environ | {
        "PYTEST": str(pytest),
        "TEST_INVOCATIONS": str(calls),
        "TEST_ITEMS": f"{failed_item}\ndcn/tests/test_example.py::test_passes",
        "TEST_FAIL_ON": failed_item,
        "TEST_JOBS": "2",
    }

    result = subprocess.run(
        [RUNNER], cwd=ROOT, env=environment, text=True, capture_output=True
    )

    assert result.returncode == 1
    worker_calls = [
        call
        for call in calls.read_text().splitlines()
        if "-m training_e2e" not in call
    ]
    assert len(worker_calls) == 2


@pytest.mark.parametrize("collection_exit_status", [0, 7])
def test_default_run_fails_when_test_collection_does_not_produce_items(
    tmp_path: Path, collection_exit_status: int
) -> None:
    pytest_executable, calls = _fake_pytest(tmp_path)
    environment = os.environ | {
        "PYTEST": str(pytest_executable),
        "TEST_INVOCATIONS": str(calls),
        "TEST_COLLECTION_STATUS": str(collection_exit_status),
    }

    result = subprocess.run(
        [RUNNER], cwd=ROOT, env=environment, text=True, capture_output=True
    )

    assert result.returncode != 0
    assert not calls.exists()
