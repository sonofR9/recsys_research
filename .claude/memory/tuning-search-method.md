---
name: tuning-search-method
description: Grid search only for one parameter; use random search or Optuna as soon as two or more move
metadata:
  type: feedback
---

Never use grid search for multi-parameter hyperparameter tuning. Random search
or Optuna is always better; a grid is only acceptable when exactly one
parameter moves.

**Why:** a Cartesian grid over k parameters spends its budget on k
one-dimensional projections, so it cannot afford a useful resolution in any of
them — which is how G1 ended up with 9-point LR grids that still left winners
sitting on a tested boundary.

**How to apply:** for the embedding/deep LR pair, or a rate crossed with a
negative count, mixture share, or alpha, propose random search or Optuna at the
same budget. The existing Cartesian grids in
`experiments/g1_sasrec_item_ids_likes/protocol/tuning.md` stay only so
already-completed comparisons remain interpretable against each other. Also
recorded in `CLAUDE.md`. See [[g1-report-audit-2026-08-17]].
