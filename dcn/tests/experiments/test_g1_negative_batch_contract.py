import os
import runpy
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest


EXPERIMENT = (
    Path(__file__).resolve().parents[3]
    / "experiments/g1_sasrec_item_ids_likes"
)
TUNING_LAUNCHER = EXPERIMENT / "launchers/negatives/tuning_50m.sh"
FINAL_LAUNCHER = EXPERIMENT / "launchers/negatives/selected_500m.sh"
QUEUE_STUB = Path(__file__).resolve().parents[1] / "fixtures/training_queue_stub.sh"
GLOBAL_BATCH_STUB = (
    Path(__file__).resolve().parents[1]
    / "fixtures/global_batch_verifier_stub.sh"
)
COLLECT = EXPERIMENT / "analysis/collect.py"
EMBEDDING_LRS = (0.008, 0.016, 0.032)
DEEP_LRS = (0.003, 0.006, 0.012)
NEGATIVE_FAMILIES = (
    "fixed_inbatch_global_q_yi2019",
    "fixed_inbatch_leave_one_out",
    "streaming_inbatch_global_q_yi2019",
    "uniform_random",
    "popularity_random_global_q_yi2019",
    "uncorrected_inbatch",
    "uniform_random_plus_streaming_logq_negative_only",
    "uniform_random_plus_fixed_logq_negative_only",
)


def _negative_run(
    report_run: Any,
    name: str,
    embedding: float,
    deep: float,
    score: float,
    *,
    batch: int = 1280,
    count: int = 512,
) -> Any:
    return report_run(
        name=name,
        configuration=name,
        dataset_size="50m",
        research_question=11,
        method="fixed in-batch global-q Yi-2019",
        status="completed",
        metrics={"recall@100": score},
        metadata={
            "batch_size": batch,
            "embedding_learning_rate": embedding,
            "deep_learning_rate": deep,
            "transfer_invariants": {
                "num_in_batch_negatives": count,
                "logq_alpha": 0.01,
                "random_negative_fraction": 0.5,
            },
        },
    )


def _canonical_negative_grid(
    report_run: Any,
    scores: dict[tuple[float, float], float] | None = None,
) -> list[Any]:
    scores = scores or {}
    return [
        _negative_run(
            report_run,
            f"initial_e{embedding}_d{deep}",
            embedding,
            deep,
            scores.get((embedding, deep), 0.5),
        )
        for embedding in EMBEDDING_LRS
        for deep in DEEP_LRS
    ]


def _run(path: Path, **environment: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(path)],
        capture_output=True,
        text=True,
        env=os.environ
        | {"G1_TRAINING_QUEUE_LIBRARY": str(QUEUE_STUB)}
        | environment,
    )


def _stage_contract(stage: str) -> tuple[dict[str, str], str]:
    common = {
        "G1_GLOBAL_BATCH_SIZE": "1536",
        "G1_GLOBAL_BATCH_SELECTION": "control/control:0.016:0.006:1536",
        "G1_TEST_GLOBAL_BATCH_VERIFIER": str(GLOBAL_BATCH_STUB),
        "G1_TUNE_NEGATIVE_FAMILIES": "fixed_inbatch_global_q_yi2019",
        "G1_TUNE_NEGATIVE_STAGE": stage,
        "G1_TUNE_RUN_TAG": "accumulationcontract",
    }
    if stage == "lr":
        common.update(
            G1_TUNE_EMBEDDING_LRS="0.013",
            G1_TUNE_DEEP_LRS="0.007",
        )
        suffix = "accumulationcontract_e0p013_d0p007_b1536_ts2_r2"
    elif stage == "secondary":
        common.update(
            G1_TUNE_SELECTED_LRS=(
                "fixed_inbatch_global_q_yi2019:0.016:0.006"
            ),
            G1_TUNE_NEGATIVE_COUNTS="777",
        )
        suffix = "accumulationcontract_e0p016_d0p006_b1536_n777_ts2_r2"
    else:
        common.update(
            G1_TUNE_SELECTED_LRS=(
                "fixed_inbatch_global_q_yi2019:0.016:0.006"
            ),
            G1_TUNE_SELECTED_SECONDARY=(
                "fixed_inbatch_global_q_yi2019:777:0.01:0.5"
            ),
            G1_TUNE_EMBEDDING_LRS="0.013",
            G1_TUNE_DEEP_LRS="0.007",
        )
        suffix = (
            "accumulationcontract_"
            "e0p013_d0p007_b1536_n777_a0p01_r0p5_ts2_r2"
        )
    return common, f"neg_fixed_inbatch_global_q_yi2019_{suffix}"


@pytest.mark.parametrize("stage", ["lr", "secondary", "local_lr"])
def test_negative_tuning_forces_unaccumulated_enqueue_contract(
    tmp_path: Path, stage: str
) -> None:
    queue_stub = tmp_path / "queue.sh"
    queue_stub.write_text(
        "enqueue() { printf 'ENQUEUE %s\\n' \"$*\" >&2; }\n"
        "drain() { return 97; }\n"
    )
    environment, _ = _stage_contract(stage)
    result = _run(
        TUNING_LAUNCHER,
        **environment,
        G1_TRAINING_QUEUE_LIBRARY=str(queue_stub),
        G1_TUNE_GRADIENT_ACCUMULATION_STEPS="7",
    )

    enqueues = [
        line for line in result.stderr.splitlines() if line.startswith("ENQUEUE")
    ]
    assert enqueues
    assert all(
        "G1_TUNE_GRADIENT_ACCUMULATION_STEPS=1" in line for line in enqueues
    )
    assert all(
        "G1_TUNE_GRADIENT_ACCUMULATION_STEPS=7" not in line for line in enqueues
    )


@pytest.mark.parametrize("stage", ["lr", "secondary", "local_lr"])
def test_negative_tuning_verifies_unaccumulated_artifact_contract(
    tmp_path: Path, stage: str
) -> None:
    launcher_directory = (
        tmp_path / "experiments/g1_sasrec_item_ids_likes/launchers"
    )
    shutil.copytree(EXPERIMENT / "launchers", launcher_directory)
    environment, run = _stage_contract(stage)
    artifact = tmp_path / "generated/logs" / f"g1_rqtune_{run}_50m"
    artifact.mkdir(parents=True)
    (artifact / "marker").write_text("complete")
    verifier_log = tmp_path / "verifier.log"
    (launcher_directory / "verify_artifact.py").write_text(
        "import os\n"
        "import sys\n"
        "from pathlib import Path\n"
        "log = Path(os.environ['G1_TEST_VERIFIER_LOG'])\n"
        "for line in sys.stdin:\n"
        "    with log.open('a') as output:\n"
        "        output.write(line)\n"
        "    valid = 'G1_TUNE_GRADIENT_ACCUMULATION_STEPS=1' in line\n"
        "    print('complete' if valid else '2\\tmissing accumulation', flush=True)\n"
    )
    result = _run(
        launcher_directory / "negatives/tuning_50m.sh",
        **environment,
        G1_TEST_VERIFIER_LOG=str(verifier_log),
        G1_TUNE_GRADIENT_ACCUMULATION_STEPS="7",
    )

    assert result.returncode == 1
    verifier_input = verifier_log.read_text()
    assert "G1_TUNE_GRADIENT_ACCUMULATION_STEPS=1" in verifier_input
    assert "G1_TUNE_GRADIENT_ACCUMULATION_STEPS=7" not in verifier_input
    assert "G1_TEST_QUEUE_STUB_ENQUEUE" not in result.stderr


def test_negative_selection_ignores_off_global_batch() -> None:
    namespace = runpy.run_path(str(COLLECT))
    report_run = namespace["ReportRun"]

    def make_run(batch: int, count: int, embedding: float, deep: float, score: float):
        return report_run(
            name=f"neg_b{batch}_n{count}_e{embedding}_d{deep}",
            configuration=f"neg_b{batch}_n{count}",
            dataset_size="50m",
            research_question=11,
            method="fixed in-batch global-q Yi-2019",
            status="completed",
            metrics={"recall@100": score},
            metadata={
                "batch_size": batch,
                "embedding_learning_rate": embedding,
                "deep_learning_rate": deep,
                "transfer_invariants": {
                    "num_in_batch_negatives": count,
                    "logq_alpha": 0.01,
                    "random_negative_fraction": 0.5,
                },
            },
        )

    runs = [
        make_run(
            1536,
            512,
            embedding,
            deep,
            0.8 if (embedding, deep) == (0.016, 0.006) else 0.6,
        )
        for embedding in EMBEDDING_LRS
        for deep in DEEP_LRS
    ]
    runs.append(make_run(1536, 2048, 0.016, 0.006, 0.7))
    runs.extend(
        make_run(
            1536,
            1024,
            embedding,
            deep,
            0.9 if (embedding, deep) == (0.016, 0.006) else 0.7,
        )
        for embedding in EMBEDDING_LRS
        for deep in DEEP_LRS
    )
    runs.extend(
        make_run(1280, 1024, embedding, deep, 0.99)
        for embedding in EMBEDDING_LRS
        for deep in DEEP_LRS
    )

    winner = namespace["_negative_proxy_winner"]("fixed", runs, 1536)

    assert winner.metadata["batch_size"] == 1536
    assert winner.metadata["transfer_invariants"]["num_in_batch_negatives"] == 1024


def test_negative_initial_lr_accepts_closed_sparse_boundary_chain() -> None:
    namespace = runpy.run_path(str(COLLECT))
    report_run = namespace["ReportRun"]
    runs = _canonical_negative_grid(report_run, {(0.032, 0.003): 0.8})
    runs.extend(
        (
            _negative_run(report_run, "boundary_e0.064_d0.003", 0.064, 0.003, 0.9),
            _negative_run(report_run, "boundary_e0.128_d0.003", 0.128, 0.003, 0.85),
            _negative_run(report_run, "boundary_e0.032_d0.0015", 0.032, 0.0015, 0.4),
        )
    )

    winner = namespace["_negative_initial_lr_winner"]("fixed", runs, 1280)

    assert winner.name == "boundary_e0.064_d0.003"


@pytest.mark.parametrize("winner_rates", [(0.008, 0.006), (0.032, 0.006)])
def test_negative_initial_lr_rejects_min_or_max_boundary_winner(
    winner_rates: tuple[float, float],
) -> None:
    namespace = runpy.run_path(str(COLLECT))
    runs = _canonical_negative_grid(namespace["ReportRun"], {winner_rates: 0.9})

    with pytest.raises(ValueError, match="remains on a tested boundary"):
        namespace["_negative_initial_lr_winner"]("fixed", runs, 1280)


def test_negative_initial_lr_requires_canonical_grid() -> None:
    namespace = runpy.run_path(str(COLLECT))
    runs = _canonical_negative_grid(
        namespace["ReportRun"], {(0.016, 0.006): 0.9}
    )
    runs = [
        run
        for run in runs
        if (
            run.metadata["embedding_learning_rate"],
            run.metadata["deep_learning_rate"],
        )
        != (0.008, 0.003)
    ]

    with pytest.raises(ValueError, match="missing batch-1280 LR points"):
        namespace["_negative_initial_lr_winner"]("fixed", runs, 1280)


def test_negative_initial_lr_collapses_exact_semantic_duplicates() -> None:
    namespace = runpy.run_path(str(COLLECT))
    report_run = namespace["ReportRun"]
    runs = [
        run
        for run in _canonical_negative_grid(report_run)
        if (
            run.metadata["embedding_learning_rate"],
            run.metadata["deep_learning_rate"],
        )
        != (0.016, 0.006)
    ]
    runs.extend(
        (
            _negative_run(report_run, "a_original", 0.016, 0.006, 0.8),
            _negative_run(report_run, "z_duplicate", 0.016, 0.006, 0.99),
        )
    )

    winner = namespace["_negative_initial_lr_winner"]("fixed", runs, 1280)

    assert winner.name == "a_original"


def test_negative_tuning_requires_global_batch() -> None:
    result = _run(TUNING_LAUNCHER, G1_GLOBAL_BATCH_SIZE="")

    assert result.returncode == 2
    assert "G1_GLOBAL_BATCH_SIZE" in result.stderr


def test_negative_secondary_rejects_batch_axis() -> None:
    result = _run(
        TUNING_LAUNCHER,
        G1_GLOBAL_BATCH_SIZE="1536",
        G1_TUNE_NEGATIVE_STAGE="secondary",
        G1_TUNE_BATCH_SIZES="1024 1536",
        G1_TUNE_SELECTED_LRS="fixed_inbatch_global_q_yi2019:0.016:0.006",
        G1_TUNE_NEGATIVE_FAMILIES="fixed_inbatch_global_q_yi2019",
    )

    assert result.returncode == 2
    assert "training batch is global" in result.stderr


def test_negative_local_selection_has_no_family_batch() -> None:
    result = _run(
        TUNING_LAUNCHER,
        G1_GLOBAL_BATCH_SIZE="1536",
        G1_TUNE_NEGATIVE_STAGE="local_lr",
        G1_TUNE_SELECTED_LRS="fixed_inbatch_global_q_yi2019:0.016:0.006",
        G1_TUNE_SELECTED_SECONDARY=(
            "fixed_inbatch_global_q_yi2019:1280:1024:0.01:0.5"
        ),
        G1_TUNE_NEGATIVE_FAMILIES="fixed_inbatch_global_q_yi2019",
    )

    assert result.returncode == 2
    assert "Invalid G1_TUNE_SELECTED_SECONDARY" in result.stderr


def test_negative_final_requires_global_batch_outside_winners() -> None:
    result = _run(
        FINAL_LAUNCHER,
        G1_GLOBAL_BATCH_SIZE="1536",
        G1_SELECTED_NEGATIVE_WINNERS=(
            "fixed_inbatch_global_q_yi2019:0.016:0.006:1280:512:0.01:0.5"
        ),
    )

    assert result.returncode == 2
    assert "Invalid G1_SELECTED_NEGATIVE_WINNERS" in result.stderr


def test_negative_launcher_uses_injected_queue_stub() -> None:
    result = _run(
        TUNING_LAUNCHER,
        G1_GLOBAL_BATCH_SIZE="1536",
        G1_GLOBAL_BATCH_SELECTION="control/control:0.016:0.006:1536",
        G1_TEST_GLOBAL_BATCH_VERIFIER=str(GLOBAL_BATCH_STUB),
        G1_TUNE_NEGATIVE_FAMILIES="fixed_inbatch_global_q_yi2019",
        G1_TUNE_EMBEDDING_LRS="0.013",
        G1_TUNE_DEEP_LRS="0.007",
        G1_TUNE_RUN_TAG="queueguard",
    )

    assert result.returncode == 1
    assert "G1_TEST_QUEUE_STUB_SOURCED" in result.stderr
    assert "G1_TEST_QUEUE_STUB_ENQUEUE" in result.stderr
    assert "dcn.main" not in result.stdout + result.stderr


def test_negative_extended_cap_has_run_and_assignment_provenance(
    tmp_path: Path,
) -> None:
    queue_stub = tmp_path / "queue.sh"
    queue_stub.write_text(
        "enqueue() { printf 'ENQUEUE %s\\n' \"$*\" >&2; return 97; }\n"
        "drain() { return 97; }\n"
    )
    result = _run(
        TUNING_LAUNCHER,
        G1_TRAINING_QUEUE_LIBRARY=str(queue_stub),
        G1_GLOBAL_BATCH_SIZE="1536",
        G1_GLOBAL_BATCH_SELECTION="control/control:0.016:0.006:1536",
        G1_TEST_GLOBAL_BATCH_VERIFIER=str(GLOBAL_BATCH_STUB),
        G1_TUNE_NEGATIVE_FAMILIES="fixed_inbatch_global_q_yi2019",
        G1_TUNE_EMBEDDING_LRS="0.013",
        G1_TUNE_DEEP_LRS="0.007",
        G1_TUNE_RUN_TAG="capcheck",
        G1_TUNE_EPOCHS="30",
        G1_TUNE_RUN_REVISION="3",
    )

    assert result.returncode == 1
    assert "_cap30_ts2_r3_50m" in result.stderr
    assert "G1_TUNE_EPOCHS=30" in result.stderr
    assert "G1_TUNE_RUN_REVISION=3" in result.stderr


@pytest.mark.parametrize("run_tag", ["cap30", "extension_cap30"])
def test_negative_rejects_reserved_cap_suffix(run_tag: str) -> None:
    result = _run(
        TUNING_LAUNCHER,
        G1_GLOBAL_BATCH_SIZE="1280",
        G1_GLOBAL_BATCH_SELECTION="control/control:0.016:0.006:1280",
        G1_TEST_GLOBAL_BATCH_VERIFIER=str(GLOBAL_BATCH_STUB),
        G1_TUNE_NEGATIVE_FAMILIES="fixed_inbatch_global_q_yi2019",
        G1_TUNE_RUN_TAG=run_tag,
    )

    assert result.returncode == 2
    assert "reserved tag" in result.stderr
    assert "ENQUEUE" not in result.stderr


def test_negative_tuning_rejects_unselected_batch_before_queue() -> None:
    result = _run(
        TUNING_LAUNCHER,
        G1_GLOBAL_BATCH_SIZE="999",
        G1_GLOBAL_BATCH_SELECTION="control/control:0.016:0.006:999",
        G1_TEST_GLOBAL_BATCH_VERIFIER=str(GLOBAL_BATCH_STUB),
        G1_TUNE_NEGATIVE_FAMILIES="fixed_inbatch_global_q_yi2019",
        G1_TUNE_EMBEDDING_LRS="0.013",
        G1_TUNE_DEEP_LRS="0.007",
        G1_TUNE_RUN_TAG="invalidglobalbatch",
    )

    assert result.returncode == 2
    assert "G1_TEST_GLOBAL_BATCH_VERIFIER" in result.stderr
    assert "G1_TEST_QUEUE_STUB_SOURCED" not in result.stderr


def test_negative_final_rejects_unselected_batch_before_queue() -> None:
    winners = " ".join(
        f"{family}:0.016:0.006:512:0.01:0.5"
        for family in NEGATIVE_FAMILIES
    )
    result = _run(
        FINAL_LAUNCHER,
        G1_GLOBAL_BATCH_SIZE="999",
        G1_GLOBAL_BATCH_SELECTION="control/control:0.016:0.006:999",
        G1_TEST_GLOBAL_BATCH_VERIFIER=str(GLOBAL_BATCH_STUB),
        G1_SELECTED_NEGATIVE_WINNERS=winners,
    )

    assert result.returncode == 2
    assert "G1_TEST_GLOBAL_BATCH_VERIFIER" in result.stderr
    assert "G1_TEST_QUEUE_STUB_SOURCED" not in result.stderr


def test_negative_final_extended_cap_has_collision_safe_provenance(
    tmp_path: Path,
) -> None:
    queue_stub = tmp_path / "queue.sh"
    queue_stub.write_text(
        "enqueue() { printf 'ENQUEUE %s\\n' \"$*\" >&2; return 97; }\n"
        "drain() { return 97; }\n"
    )
    winners = " ".join(
        f"{family}:0.016:0.006:512:0.01:0.5"
        for family in NEGATIVE_FAMILIES
    )

    result = _run(
        FINAL_LAUNCHER,
        G1_TRAINING_QUEUE_LIBRARY=str(queue_stub),
        G1_GLOBAL_BATCH_SIZE="1280",
        G1_GLOBAL_BATCH_SELECTION="control/control:0.016:0.006:1280",
        G1_TEST_GLOBAL_BATCH_VERIFIER=str(GLOBAL_BATCH_STUB),
        G1_SELECTED_NEGATIVE_WINNERS=winners,
        G1_FINAL_EPOCHS="30",
        G1_FINAL_RUN_REVISION="3",
    )

    assert result.returncode == 1
    assert "_cap30_ts2_r3_500m" in result.stderr
    assert "G1_TUNE_EPOCHS=30" in result.stderr
    assert "G1_TUNE_RUN_REVISION=3" in result.stderr
