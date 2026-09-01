from __future__ import annotations

from pathlib import Path

from .rq4_followup_gate import require_rq4_further_capacity_width_approval


def compile_rq4_next_capacity_ledger(*, root: Path) -> None:
    require_rq4_further_capacity_width_approval(root)
    raise RuntimeError(
        "RQ4 next-capacity plan has not been approved and implemented"
    )
