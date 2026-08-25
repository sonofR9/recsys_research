import os
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "experiments/g1_sasrec_item_ids_likes"
GLOBAL_BATCH_STUB = (
    Path(__file__).resolve().parents[1]
    / "fixtures/global_batch_verifier_stub.sh"
)


def _width_grid(report_run, width: int) -> list[object]:
    embedding_lrs = (0.008, 0.016, 0.032, 0.064)
    deep_lrs = (0.003, 0.006, 0.012, 0.024)
    oracle = (0.032, 0.012) if width == 32 else (0.016, 0.006)
    return [
        report_run(
            name=f"width_{width}_{embedding_lr}_{deep_lr}",
            configuration=(
                f"dimension_{width}_e{str(embedding_lr).replace('.', 'p')}_"
                f"d{str(deep_lr).replace('.', 'p')}_b1280"
            ),
            dataset_size="50m",
            research_question=8,
            method="model dimension",
            status="completed",
            metrics={
                "recall@100": (
                    1.0 if (embedding_lr, deep_lr) == oracle else 0.5
                )
            },
            metadata={
                "batch_size": 1280,
                "embedding_learning_rate": embedding_lr,
                "deep_learning_rate": deep_lr,
                "item_embedding_dim": 64,
                "model_dim": width,
                "transfer_invariants": {
                    "experiment_class": "MuTransferGenerationExperiment",
                    "mup_base_dim": 16,
                    "mup_delta_dim": 32,
                },
            },
        )
        for embedding_lr in embedding_lrs
        for deep_lr in deep_lrs
    ]


def test_selector_explicitly_reuses_exact_rq8_dimension_artifacts() -> None:
    from experiments.g1_sasrec_item_ids_likes.analysis import collect
    from experiments.g1_sasrec_item_ids_likes.analysis.select_width_transfer_500m import (
        select_width_transfer_confirmations,
    )

    runs = [
        run
        for width in (16, 32, 64, 128, 256)
        for run in _width_grid(collect.ReportRun, width)
    ]

    selected = select_width_transfer_confirmations(runs)

    actual = [
        (width, embedding_lr, deep_lr, batch)
        for width, _, embedding_lr, deep_lr, batch in selected
    ]
    assert actual == [
        (16, 0.032, 0.012, 1280),
        (256, 0.032, 0.012, 1280),
    ]


def test_selector_does_not_require_a_new_cartesian_lr_sweep() -> None:
    from experiments.g1_sasrec_item_ids_likes.analysis import collect
    from experiments.g1_sasrec_item_ids_likes.analysis.select_width_transfer_500m import (
        select_width_transfer_confirmations,
    )

    runs = [
        run
        for width in (16, 256)
        for run in _width_grid(collect.ReportRun, width)
        if collect._run_rates(run) == (0.032, 0.012)
    ]

    selected = select_width_transfer_confirmations(runs)

    assert [width for width, *_ in selected] == [16, 256]


def test_selector_cli_emits_exact_rq8_provenance(
    monkeypatch, capsys, tmp_path: Path
) -> None:
    from experiments.g1_sasrec_item_ids_likes.analysis import collect
    from experiments.g1_sasrec_item_ids_likes.analysis import (
        select_width_transfer_500m,
    )

    runs = [
        run
        for width in (16, 256)
        for run in _width_grid(collect.ReportRun, width)
        if collect._run_rates(run) == (0.032, 0.012)
    ]
    monkeypatch.setattr(collect, "load_report_runs", lambda *_args, **_kwargs: runs)
    monkeypatch.setattr(
        "sys.argv",
        ["select_width_transfer_500m.py", "--generated", str(tmp_path)],
    )

    select_width_transfer_500m.main()

    assert capsys.readouterr().out.splitlines() == [
        "16\twidth_16_0.032_0.012\t0.032\t0.012\t1280",
        "256\twidth_256_0.032_0.012\t0.032\t0.012\t1280",
    ]


def _selector_stub(tmp_path: Path, rows: str) -> Path:
    selector = tmp_path / "select_width_transfer_500m.py"
    selector.write_text(f"print({rows!r}, end='')\n")
    return selector


def _queue_stub(tmp_path: Path, *, fail_enqueue: bool = False) -> Path:
    queue = tmp_path / "queue.sh"
    enqueue_result = "return 97;" if fail_enqueue else "return 0;"
    queue.write_text(
        "printf 'FORWARD %s WANDB=%s DATA=%s\\n' "
        "\"${TRAINING_QUEUE_FORWARD_ENV:-}\" \"${WANDB_MODE:-}\" "
        "\"${G1_DATASET_SIZE:-}\" >&2\n"
        f"enqueue() {{ printf 'ENQUEUE %s\\n' \"$*\" >&2; {enqueue_result} }}\n"
        "drain() { return 0; }\n"
    )
    return queue


def _launcher_environment(
    tmp_path: Path, selector_rows: str, *, fail_enqueue: bool = False
) -> dict[str, str]:
    return os.environ | {
        "G1_TRAINING_QUEUE_LIBRARY": str(
            _queue_stub(tmp_path, fail_enqueue=fail_enqueue)
        ),
        "G1_TEST_GLOBAL_BATCH_VERIFIER": str(GLOBAL_BATCH_STUB),
        "G1_GLOBAL_BATCH_SELECTION": "control/control:0.032:0.012:1280",
        "G1_WIDTH_TRANSFER_SELECTOR": str(_selector_stub(tmp_path, selector_rows)),
    }


def test_launcher_enqueues_exact_two_width_confirmations(tmp_path: Path) -> None:
    launcher_directory = (
        tmp_path / "experiments/g1_sasrec_item_ids_likes/launchers"
    )
    shutil.copytree(EXPERIMENT / "launchers", launcher_directory)
    selector_rows = (
        "16\tproxy_width_16\t0.032\t0.012\t1280\n"
        "256\tproxy_width_256\t0.032\t0.012\t1280\n"
    )

    result = subprocess.run(
        ["bash", str(launcher_directory / "transfer/width_transfer_500m.sh")],
        capture_output=True,
        text=True,
        env=_launcher_environment(tmp_path, selector_rows),
    )

    assert result.returncode == 0
    assert result.stderr.count("ENQUEUE ") == 2
    assert (
        "ENQUEUE g1_rqtune_rqfinal_dimension_16_e0p032_d0p012_b1280_ts2_r2_500m"
        in result.stderr
    )
    assert (
        "G1_TUNE_SOURCE_VARIANT=dim_16 G1_TUNE_TRANSFORMER_FIELDS=dim "
        "G1_TUNE_EXPERIMENT_FIELDS= G1_TUNE_EMBEDDING_LR=0.032 "
        "G1_TUNE_DEEP_LR=0.012 G1_TUNE_BATCH_SIZE=1280 "
        "G1_TUNE_GRADIENT_ACCUMULATION_STEPS=1 G1_TUNE_FFN_DIM=43"
        in result.stderr
    )
    assert (
        "ENQUEUE g1_rqtune_rqfinal_dimension_256_e0p032_d0p012_b1280_ts2_r2_500m"
        in result.stderr
    )
    assert "G1_TUNE_FFN_DIM=684" in result.stderr


def test_launcher_propagates_enqueue_failure_without_counting_it(
    tmp_path: Path,
) -> None:
    launcher_directory = (
        tmp_path / "experiments/g1_sasrec_item_ids_likes/launchers"
    )
    shutil.copytree(EXPERIMENT / "launchers", launcher_directory)
    selector_rows = (
        "16\tproxy_width_16\t0.032\t0.012\t1280\n"
        "256\tproxy_width_256\t0.032\t0.012\t1280\n"
    )

    result = subprocess.run(
        ["bash", str(launcher_directory / "transfer/width_transfer_500m.sh")],
        capture_output=True,
        text=True,
        env=_launcher_environment(
            tmp_path, selector_rows, fail_enqueue=True
        ),
    )

    assert result.returncode == 1
    assert result.stderr.count("ENQUEUE ") == 1
    assert "enqueued=1" not in result.stdout


def test_launcher_submits_single_width_cap_continuation_with_offline_provenance(
    tmp_path: Path,
) -> None:
    launcher_directory = (
        tmp_path / "experiments/g1_sasrec_item_ids_likes/launchers"
    )
    shutil.copytree(EXPERIMENT / "launchers", launcher_directory)
    selector_rows = (
        "16\tproxy_width_16\t0.032\t0.012\t1280\n"
        "256\tproxy_width_256\t0.032\t0.012\t1280\n"
    )
    environment = _launcher_environment(tmp_path, selector_rows) | {
        "G1_WIDTH_TRANSFER_WIDTHS": "16",
        "G1_WIDTH_TRANSFER_EPOCHS": "40",
        "G1_WIDTH_TRANSFER_RUN_REVISION": "2",
    }

    result = subprocess.run(
        ["bash", str(launcher_directory / "transfer/width_transfer_500m.sh")],
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0
    assert result.stderr.count("ENQUEUE ") == 1
    assert "dimension_16_e0p032_d0p012_b1280_cap40_ts2_r2_500m" in result.stderr
    assert "dimension_256" not in result.stderr
    assert "FORWARD G1_DATASET_SIZE WANDB_MODE WANDB=offline DATA=500m" in result.stderr


@pytest.mark.parametrize(
    "selector_rows",
    [
        "16\tproxy_width_16\t0.016\t0.012\t1280\n"
        "256\tproxy_width_256\t0.032\t0.012\t1280\n",
        "16\tproxy_width_16\t0.032\t0.012\t1280\n",
    ],
)
def test_launcher_rejects_unapproved_selector_output(
    tmp_path: Path, selector_rows: str
) -> None:
    launcher_directory = (
        tmp_path / "experiments/g1_sasrec_item_ids_likes/launchers"
    )
    shutil.copytree(EXPERIMENT / "launchers", launcher_directory)

    result = subprocess.run(
        ["bash", str(launcher_directory / "transfer/width_transfer_500m.sh")],
        capture_output=True,
        text=True,
        env=_launcher_environment(tmp_path, selector_rows),
    )

    assert result.returncode == 2
    assert "ENQUEUE " not in result.stderr
