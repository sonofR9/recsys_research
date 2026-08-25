from __future__ import annotations

from typing import Any


GENERATION_TRAINING_SEMANTICS_REVISION = 2
TIMESTAMP_BIN_SEMANTICS_REVISION = 2
NEGATIVE_SAMPLING_SEMANTICS_REVISION = 2


def has_current_generation_semantics(metadata: dict[str, Any]) -> bool:
    revision = metadata.get("training_semantics_revision")
    return (
        type(revision) is int
        and revision == GENERATION_TRAINING_SEMANTICS_REVISION
    )
