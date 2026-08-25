import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
CHECK = (
    ROOT
    / "experiments/g1_sasrec_item_ids_likes/checks/utilization_regression_50m.py"
)


def test_utilization_gate_uses_the_current_selected_architecture(monkeypatch) -> None:
    monkeypatch.setenv("G1_VARIANT", "test_restore")
    monkeypatch.setenv("G1_DATASET_SIZE", "500m")
    monkeypatch.delenv("G1_MAX_USERS", raising=False)
    monkeypatch.delenv("G1_UTIL_BATCH_SIZE", raising=False)

    namespace = runpy.run_path(str(CHECK))
    selected = namespace["selected"]

    assert "selected_quality_b1280" in selected.run_name
    assert selected.dataloader.batch_size == 1280
    assert selected.embedding_learning_rate == 0.001
    assert selected.deep_learning_rate == 0.003
