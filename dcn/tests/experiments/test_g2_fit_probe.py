import os
from pathlib import Path
import subprocess

from dcn.main import load_experiment


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "experiments/g2_esasrec/launchers/run_fit_probe.py"


def test_fit_probe_is_max_ligr_and_samples_users_by_hash(monkeypatch):
    monkeypatch.setenv("G2_FIT_BATCH_SIZE", "512")

    experiment = load_experiment(SCRIPT)

    assert experiment.transformer.ffn_intermediate_dim == 1536
    assert experiment.transformer.ffn_intermediate_dim % 32 == 0
    assert experiment.loss_kind == "gbce"
    assert experiment.gbce_t == 0.75
    assert experiment.dataloader.batch_size == 512
    assert experiment.user_sample.max_users == 2_000
    query = experiment.user_sample.duckdb_query("users")
    assert "SELECT DISTINCT uid" in query
    assert "ORDER BY hash(uid || '_42'), uid LIMIT 2000" in query


def test_fit_probe_launcher_queues_every_control_batch():
    launcher = ROOT / "experiments/g2_esasrec/launchers/queue_fit_probes.sh"
    text = launcher.read_text()

    assert "for batch_size in 128 256 512 1024 1280" in text
    assert "utils/training_queue/queue.sh" in text
    assert '"WANDB_MODE=offline"' in text
    subprocess.run(["bash", "-n", str(launcher)], check=True, env=os.environ)
