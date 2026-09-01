# Project context

A personal RecSys research framework. The goal is to test hypotheses across datasets and architectures from a single configurable entry point.

`competition/` is the main working folder — run from here and use the `.claude/` (skills, memory) inside it, not any home-directory or parent copy.

See [docs/architecture.md](docs/architecture.md) for the data flow, module layout, caching/paths, tests, and legacy code.

## General

Be concise.

## Running

The intended entry point is one command driven by a script:

```bash
# important to source!
source ~/.bash_aliases
source /home/sonofr/python_venvs/.venv/bin/activate
python -m dcn.main -s dcn/scripts/yambda_dcn.py
# or
./run_dcn.sh
```

Set `data.invalidate_cache: true` to force a rebuild of cached artifacts.

When work requires multiple full training runs — a sweep, seed repeats, or
several variants — ensure the persistent
[`utils/training_queue`](utils/training_queue/README.md) service is running,
then run the approved launcher. Its granular jobs join the shared schedule and
survive the submitting shell. Do not start competing queues. Direct invocation
is for one run or debugging only.

## Workflow

Use write-code skill when you want to write code.

### Research team

The primary agent acts as lead researcher and orchestrator. Read
[`agents/AGENTS.md`](agents/AGENTS.md) before planning research, delegating
work, or running experiments. Keep assignments current in
[`agents/STATUS.md`](agents/STATUS.md).

### Review

Review your own work first, then have a **subagent** review it that knows only
the short task description — not the plan, the reasoning, or the conversation.
It sees the code the way the next reader will, which is the point: anything it
has to ask about is not explained by the code. Give it the one-line task and
the diff, nothing else.

### Testing

During development, run only the tests that can be affected by the current
change with `pytest -q path/to/test_file.py`. After implementation and review
are complete, run `./test.sh` once. It runs the non-training CPU tests in 16
parallel pytest groups by default, then runs the training end-to-end group
serially; set `TEST_JOBS` to override the parallel group count. GPU tests remain
opt-in and must be run directly with pytest on a dedicated GPU.
The runner does not impose machine-specific PyTorch or BLAS thread limits.
The selected-model A100 utilization regression is an optional `slow_gpu` test,
not a prerequisite for launching a sweep.
Run and time the complete non-training suite only when external CPU load is
low. Postpone it while the shared host is busy; a loaded-host duration is not
performance evidence and cannot satisfy test-optimization acceptance.

## Conventions

- Be concise — terse, direct answers. Skip preamble, restating the question, and summaries of what you just did unless asked.
- Coverage is intentionally narrow — don't add tests just to add tests.
- Don't read or modify legacy code (`old/` subfolders, `week02_version.ipynb`) or third-party code (`yambda_original/`) unless explicitly asked.

### Sampling

**Always sample by user id, never by position.** That way distributions are preserved.

### Comments

Comments are a cost. They drift out of sync with the code, they push the code
apart, and they are read as if they were true. The default is none.

- **Never comment what the code does.** Names, types and structure say that. If
  a comment is needed to follow the code, rename or split the code instead.
- **Do comment what the code cannot say**: a link to the paper behind an
  uncommon algorithm, a compatibility quirk of a library, an invariant enforced
  somewhere else.
- **Intent, only when the choice is surprising.** If a reader would ask "why
  this and not the obvious thing?", answer it. A choice that already looks
  right needs nothing.
- **Three lines is a smell.** A comment that long usually means the code under
  it wants a name — extract a function or a value object and delete the
  comment.
- Same for docstrings: public API that is not obvious from its signature. No
  docstring on a one-line property.
- Never delete a `FIXME`/`TODO` you have not actually fixed.

## Project-scoped storage

Everything project-related — including memory, notes, scratch files — must live **inside the repo**. The user works across multiple machines and keeps state in git.

- Memory goes in `.claude/memory/` at the repo root (also `.agents/memory/`; `MEMORY.md` index + per-topic files).
- Same rule for any other Claude artifacts you might be tempted to write to a home-directory location.

## Experiments

Every agent-chosen learned neural width must be divisible by 16; prefer a
multiple of 32. This includes model, attention, FFN/*GLU, MLP/DenseNet,
projection, and learned-feature widths. Fixed external input widths are exempt,
but their learned projection target is not. Never round a parameter-count
formula to an irregular width such as 171.

All newly launched research training uses batch size `512`; do not tune batch
size. Spend the tuning budget on the experiment's relevant non-batch
hyperparameters, including the training epoch/schedule horizon. Preserve
completed runs at other batch sizes as historical evidence, but do not launch
new non-512 runs without explicit user approval.

Measure seed variability once for the canonical unchanged control at each
dataset size and reuse it across all future experiments at that size. Reuse only the
relative dispersion for each metric (sample standard deviation divided by the
baseline mean), applied to the current experiment's own baseline value; never
copy an old absolute metric or absolute band. Do not launch another calibration
unless the user explicitly requests it.

Before planning, changing, or launching any experiment, read
[`experiments/AGENTS.md`](experiments/AGENTS.md) — it holds the binding
protocol, tuning, transfer, and report-format rules — then `experiments/list.md`
and the experiment's own report. `experiments/CLAUDE.md` symlinks to it so it
loads with the directory.

For work originating from [`experiments/ideas.md`](experiments/ideas.md), also
read [`experiments/ideas_understanding.md`](experiments/ideas_understanding.md).
It records the current interpretation, concrete implementation variants, and
acceptance criteria for unfinished ideas, but it is not an approved run plan.
Keep it synchronized when the user clarifies an idea.

Raw experiment artifacts and valid results are an immutable historical record.
Never delete them because an agent believes they are invalid, incomplete,
confounded, stale, incomparable, or superseded; preserve them in audit storage
and raise the concern separately. Before changing a result's status,
interpretation, conclusion, selection eligibility, or reader-facing table,
present the supporting evidence to the user and obtain explicit validation of
the claim. A deterministic boundary extension already specified in an approved
plan is an operational continuation, not a new result claim: when the approved
compiler authenticates the exact triggering result, frozen edge rule, and
prescribed boundary points, launch those points without requesting another
approval. The eventual selected result,
interpretation, conclusion, and reader-facing table still require validation.
Once the user approves a correction, replace incorrect tables and
claims in the active report instead of retaining or labeling them as
historical. Version control and raw evidence preserve the audit trail; deleting
raw results remains prohibited.

Choose exactly one dataset size for a research question and use it for every
tuning run and final result in that question. Present the proposed size and
obtain explicit user validation before implementation or training.
Smaller samples are allowed only for correctness/debugging checks and cannot
select treatments or support claims. If dataset size is itself the approved
research axis, keep it as a separate research question within its experiment
group and train every size in its native regime without repeating smaller data
to match a larger run's targets, tokens, or optimizer steps.

Treat the training epoch or schedule horizon as a tunable hyperparameter.
Validate every epoch, restore the best validation epoch, and report both the
declared horizon and restored epoch. For schedules without a declared horizon,
use early stopping; if it has not triggered before the selected cap, including
when the best epoch equals the cap, extend and rerun before using the result.

A schedule that anneals over a horizon must complete every tuned candidate
horizon and report the best validation epoch within it; early stopping is not
applied because patience would fire on the plateau the decay is meant to
produce. A run that stops short of its candidate horizon is unusable.

For hyperparameter tuning, use grid search only when a single parameter moves.
With two or more, use random search or Optuna at the same budget.

Before designing a new hyperparameter calibration, inspect the selected result
for the same baseline or the closest protocol-matched predecessor. Use its best
known values as the central starting point instead of recalibrating from
scratch. When transfer is imperfect because batch size, depth, dataset, or
another relevant field changed, keep the known winner as an anchor and retune
locally; do not treat its metric as evidence for the new protocol. By default,
probe three candidates around that anchor and extend only in a direction where
the selected value reaches the tested boundary. Do not fit an LR-versus-horizon
law merely to rediscover a baseline learning rate that is already known.
