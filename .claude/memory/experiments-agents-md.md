---
name: experiments-agents-md
description: experiments/AGENTS.md is a second binding rules file, separate from agents/AGENTS.md, and is easy to miss
metadata:
  type: feedback
---

There are two AGENTS.md files and they say different things. `agents/AGENTS.md`
is the research-team workflow (roles, approval steps, work tracker).
`experiments/AGENTS.md` is the experiment protocol: data rules, tuning and
transfer rules, and the exact reader-facing report and table format. Read the
second one before planning, changing, or launching any experiment.

**Why:** `CLAUDE.md` used to point only at `agents/AGENTS.md` and
`experiments/list.md`, so `experiments/AGENTS.md` went unread through the whole
G1 transfer study. It already prescribed the method for the question being
worked on — fit optimal LR at three or more proxy horizons and extrapolate each
optimizer group as `lr*(D) = A D^-beta` — and that method is already
implemented in `experiments/hyperparameter_transfer.py`. A two-size comparison
with a cross-shaped LR design was run instead, which cannot even be fed to that
fitter: the interaction column of its quadratic response surface is degenerate
on a cross.

**How to apply:** `experiments/CLAUDE.md` is now a symlink to it so it loads
with the directory, and `CLAUDE.md` names it explicitly. Before proposing any
LR-transfer or tuning work, check what that file already specifies and whether
a helper for it already exists under `experiments/`. See
[[tuning-search-method]] and [[g1-report-audit-2026-08-17]].
