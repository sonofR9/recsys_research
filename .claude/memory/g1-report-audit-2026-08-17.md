---
name: g1-report-audit-2026-08-17
description: When a G1 result contradicts established theory, suspect the protocol before believing the number
metadata:
  type: feedback
---

The research lead reads results against known theory first. Two G1 findings were
challenged on those grounds and both turned out to be protocol artifacts, not
measurements: a smaller SwiGLU transformer "tying" a much wider GELU one, and
uniform random negatives "beating" offline logQ correction.

**Why:** a result that contradicts a well-established prior is evidence about
the harness until proven otherwise. In G1 the shared cause was
`lr_schedule_horizon_epochs` staying at 20 while the epoch cap moved to 40, so
runs trained frozen weights past the horizon and early stopping reported the
schedule ending as convergence.

**How to apply:** before writing a surprising finding into a report, check the
stopping reason, the schedule state at the reported epoch, and whether the two
arms are matched. Open question still unanswered: popularity-random with offline
logQ should beat uniform random negatives, and the current RQ11 arms are not
matched well enough to test that. Full write-up in
`experiments/g1_sasrec_item_ids_likes/notes/report_audit_2026-08-17.md`.
See [[tuning-search-method]].
