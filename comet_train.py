"""Minimal PyTorch training loop with Comet logging.

One loss (cross-entropy), several metrics (accuracy / precision / recall / f1).
Synthetic data, so it runs standalone — just to test Comet.

Comet setup (pick one):
  export COMET_API_KEY=...            # online logging to comet.com
  COMET_MODE=offline python comet_train.py   # writes a .zip you upload later

Run:
  source /home/sonofr/python_venvs/.venv/bin/activate
  python comet_train.py
"""

# comet_ml must be imported before torch for its auto-logging hooks.
import os

import comet_ml
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, TensorDataset

CONFIG = {
    "n_samples": 4000,
    "n_features": 20,
    "n_classes": 3,
    "hidden": 64,
    "batch_size": 128,
    "lr": 1e-3,
    "epochs": 15,
    "seed": 0,
}


def make_experiment():
    """Online if COMET_API_KEY is set, else offline (writes a local .zip)."""
    params = dict(project_name="test project", auto_metric_logging=False)
    if os.environ.get("COMET_MODE") == "offline" or not os.environ.get(
        "COMET_API_KEY"
    ):
        return comet_ml.OfflineExperiment(
            offline_directory="./comet_offline", **params
        )
    return comet_ml.Experiment(**params)


def make_data(cfg, device):
    g = torch.Generator().manual_seed(cfg["seed"])
    # Linearly-separable-ish synthetic problem: y from a random linear map + noise.
    X = torch.randn(cfg["n_samples"], cfg["n_features"], generator=g)
    W = torch.randn(cfg["n_features"], cfg["n_classes"], generator=g)
    y = (
        X @ W
        + 0.5 * torch.randn(cfg["n_samples"], cfg["n_classes"], generator=g)
    ).argmax(1)

    n_val = cfg["n_samples"] // 5
    train = TensorDataset(X[n_val:].to(device), y[n_val:].to(device))
    val = TensorDataset(X[:n_val].to(device), y[:n_val].to(device))
    return (
        DataLoader(train, batch_size=cfg["batch_size"], shuffle=True),
        DataLoader(val, batch_size=cfg["batch_size"]),
    )


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg["n_features"], cfg["hidden"]),
            nn.ReLU(),
            nn.Linear(cfg["hidden"], cfg["n_classes"]),
        )

    def forward(self, x):
        return self.net(x)


def compute_metrics(y_true, y_pred):
    kw = dict(average="macro", zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, **kw),
        "recall": recall_score(y_true, y_pred, **kw),
        "f1": f1_score(y_true, y_pred, **kw),
    }


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    losses, preds, targets = [], [], []
    for xb, yb in loader:
        logits = model(xb)
        losses.append(criterion(logits, yb).item())
        preds.append(logits.argmax(1).cpu())
        targets.append(yb.cpu())
    y_pred = torch.cat(preds).numpy()
    y_true = torch.cat(targets).numpy()
    metrics = compute_metrics(y_true, y_pred)
    metrics["loss"] = sum(losses) / len(losses)
    return metrics


def main():
    cfg = CONFIG
    torch.manual_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    exp = make_experiment()
    exp.log_parameters(cfg)

    train_loader, val_loader = make_data(cfg, device)
    model = MLP(cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], fused=True)

    for epoch in range(cfg["epochs"]):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)  # the single loss
            loss.backward()
            optimizer.step()
            exp.log_metric("train_loss", loss.item())

        val = evaluate(model, val_loader, criterion)
        exp.log_metrics({f"val_{k}": v for k, v in val.items()}, epoch=epoch)
        print(
            f"epoch {epoch:02d} | "
            + " | ".join(f"{k}={v:.4f}" for k, v in val.items())
        )

    exp.end()


if __name__ == "__main__":
    main()
