import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "experiments/g1_sasrec_item_ids_likes"


def _queue_stub(tmp_path: Path) -> Path:
    queue = tmp_path / "queue.sh"
    queue.write_text(
        "enqueue() { printf 'ENQUEUE %s\\n' \"$*\" >&2; return 0; }\n"
        "drain() { return 0; }\n"
    )
    return queue


def _run(tmp_path: Path, **environment: str) -> subprocess.CompletedProcess:
    launchers = tmp_path / "experiments/g1_sasrec_item_ids_likes/launchers"
    shutil.copytree(EXPERIMENT / "launchers", launchers)
    return subprocess.run(
        ["bash", str(launchers / "ffn/tuning_500m.sh")],
        capture_output=True,
        text=True,
        env=os.environ | {"G1_TRAINING_QUEUE_LIBRARY": str(_queue_stub(tmp_path))} | environment,
    )


def test_launcher_sweeps_the_deep_rate_for_both_parameter_matched_arms(
    tmp_path: Path,
) -> None:
    result = _run(tmp_path)

    assert result.returncode == 0
    assert result.stderr.count("ENQUEUE ") == 6
    assert (
        "ENQUEUE g1_rqtune_rqfinal_ffn_swiglu114_e0p064_d0p012_b1280_cap40_ts2_r3_500m"
        in result.stderr
    )
    assert (
        "ENQUEUE g1_rqtune_rqfinal_ffn_gelu171_e0p064_d0p024_b1280_cap40_ts2_r3_500m"
        in result.stderr
    )
    assert "G1_TUNE_SOURCE_VARIANT=ffn_swiglu" in result.stderr
    assert "G1_TUNE_FFN_DIM=114" in result.stderr
    assert "G1_TUNE_FFN_DIM=171" in result.stderr


def test_launcher_holds_the_embedding_rate_fixed_across_every_run(
    tmp_path: Path,
) -> None:
    result = _run(tmp_path)

    enqueued = [
        line for line in result.stderr.splitlines() if line.startswith("ENQUEUE ")
    ]
    assert len(enqueued) == 6
    assert all("G1_TUNE_EMBEDDING_LR=0.064" in line for line in enqueued)
    assert {
        line.split("G1_TUNE_DEEP_LR=")[1].split()[0] for line in enqueued
    } == {"0.006", "0.012", "0.024"}


def test_launcher_rejects_an_arm_the_manifest_does_not_define(tmp_path: Path) -> None:
    result = _run(tmp_path, G1_FFN_TUNING_ARMS="swiglu113")

    assert result.returncode == 2
    assert "Missing compatible ffn/swiglu113 manifest treatment" in result.stderr
    assert "ENQUEUE " not in result.stderr


def test_launcher_rejects_a_duplicate_rate(tmp_path: Path) -> None:
    result = _run(tmp_path, G1_FFN_TUNING_DEEP_LRS="0.012 0.012")

    assert result.returncode == 2
    assert "Duplicate G1_FFN_TUNING_DEEP_LRS value: 0.012" in result.stderr
