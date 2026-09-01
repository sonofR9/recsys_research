from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import re


_EPOCH_RECALL = re.compile(
    r"\bepoch (?P<epoch>\d+) finished\b.*?"
    r"\bepoch/val_true\.recall@100=(?P<recall>[-+0-9.eE]+)"
)


@dataclass(frozen=True)
class ValidationCurve:
    recall_at_100: tuple[float, ...]
    normalized_auc: float
    first_epoch_at_95_percent: int
    source_sha256: str


def load_validation_curve(
    path: Path,
    *,
    expected_epochs: int = 15,
) -> ValidationCurve:
    if expected_epochs < 1:
        raise ValueError("expected_epochs must be positive")
    content = path.read_bytes()
    observations: dict[int, float] = {}
    for match in _EPOCH_RECALL.finditer(content.decode()):
        epoch = int(match.group("epoch"))
        recall = float(match.group("recall"))
        if epoch in observations:
            raise ValueError(f"duplicate validation epoch {epoch} in {path}")
        if not math.isfinite(recall) or not 0 <= recall <= 1:
            raise ValueError(f"invalid Recall@100 at epoch {epoch} in {path}")
        observations[epoch] = recall
    expected = list(range(expected_epochs))
    if sorted(observations) != expected:
        raise ValueError(
            f"{path}: expected epochs 0..{expected_epochs - 1}, got "
            f"{sorted(observations)}"
        )
    values = tuple(observations[epoch] for epoch in expected)
    best = max(values)
    if best <= 0:
        raise ValueError(f"{path}: validation Recall@100 never became positive")
    threshold = 0.95 * best
    first_epoch = next(
        epoch + 1 for epoch, value in enumerate(values) if value >= threshold
    )
    return ValidationCurve(
        recall_at_100=values,
        normalized_auc=sum(value / best for value in values) / len(values),
        first_epoch_at_95_percent=first_epoch,
        source_sha256=hashlib.sha256(content).hexdigest(),
    )
