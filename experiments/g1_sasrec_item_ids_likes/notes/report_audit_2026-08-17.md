# G1 reader-report audit — 2026-08-17

Audit of [`README.md`](../README.md) and the generator behind it
([`analysis/collect.py`](../analysis/collect.py)), triggered by two reader
objections: the logQ ordering and the SwiGLU result both looked wrong.

The colouring and percentages in the report are generated, not hand-written:
`_absolute_metric_cell`, `_relative_metric_cell` and `_relative_percent_cell`
render a delta green or red only when it exceeds the shared empirical band, and
`_percent_precision` picks the digit count from that band. Nothing below is a
formatting complaint; every finding is about what the numbers mean.

## 1. The LR schedule horizon was decoupled from the epoch cap (root cause)

`lr_schedule_horizon_epochs` is hardcoded to 20 in
[`configs/variant.py`](../configs/variant.py), while `num_epochs` — the safety
cap — was raised to 40 (and 60) for cap continuations.
For `linear`, `cosine`, `polynomial` and `warmup_stable_decay` with
`min_lr_fraction = 0`, the rate is exactly zero once the horizon is spent, so
every epoch past the horizon trained frozen weights. Early stopping then
observed three flat epochs, fired, and the run recorded itself as
`selection_resolved`.

Concretely, on
`g1_rqtune_rqfinal_neg_fixed_inbatch_leave_one_out_..._cap40_ts2_r3_500m`:
epochs 19–22 all show `lr=0.0000`, recall pinned at 0.1168, `best_epoch=20`,
`stopped_epoch=23`, `best_epoch_at_cap=False`. The stop was forced by the
schedule reaching zero, not by convergence — and raising the cap could never
have changed the answer, because the model cannot move after the horizon.

This inverts the intended reading of the protocol: extending the cap is supposed
to give a run more training, but with a fixed horizon it only buys frozen
epochs and a fake early stop.

**Decision taken (research lead):** a schedule that anneals over its horizon
declares its own length, so spending that horizon and reporting the best
validation epoch within it *is* an accepted result. Such a run trains the
horizon exactly and gets no early stopping at all, because patience would fire
on the plateau the decay is meant to produce. The cap and early stopping stay
for the step-by-step shapes (`constant`, `inverse_sqrt`, `power`), which have
no horizon of their own. A run that stops short of its annealing horizon
remains unresolved.

**Implemented in:**

- `neuralrec/run/callbacks/lr_schedule.py` — the callback sets `should_stop`
  once an annealing shape has spent its decay.
- `dcn/config/settings.py` — `LrScheduleConfig.anneals_over_horizon`.
- `dcn/config/generation.py` — no early stopping for an annealing schedule;
  metadata gains `lr_horizon_complete`, which resolves `selection_resolved`.
- `analysis/collect.py` — `_completed_an_annealed_horizon` is an accepted
  resolution path in `_uses_validation_selected_training`.

**Blast radius:** the frozen tails past the horizon are cosmetic — the reported
epoch is the annealed one either way — so no artifact is invalidated by this
finding. Only finding 2 (18 artifacts) still costs runs.

## 2. `step` and warmup-stable-decay never reached their decay

`step` holds its opening rate for the first half of the decay phase and WSD for
the first 80%. Runs that early-stopped before that point are numerically the
constant schedule, so they are no evidence for their own shape. In the 50M
ledger this showed up as 8 (`step`) and 11 (WSD) bit-identical rows against the
constant arm; measured decay progress was 0.4/0.45/0.55 for 50M `step` and 0.211
for 50M WSD.

`_declared_decay_engaged` in `analysis/collect.py` now rejects them.

## 3. RQ1's width tables compared runs with themselves

At widths 32, 64 and 128 the 50M-local optimum *is* the common rate, so the
"common" and "local" columns are the same artifact. The report rendered that as
`0%` regret and as a bare metric in the 500M "vs local" column — both read as
measured agreement. They now render `same run`.

## 4. RQ11 printed a logQ alpha for families that have none

[Yi et al. 2019](https://doi.org/10.1145/3298689.3346996) give the step size
alpha to the *streaming* frequency estimator (Algorithm 2) only. A cached
proposal distribution has no such parameter, so the `0.01` shown against every
fixed/offline row was the config default, not a tuned correction strength. The
column is now populated only for the streaming families.

No tuning compute was wasted on it: across all 6,886 artifacts, no fixed-logQ
run varies alpha.

## 5. RQ11 cannot answer whether offline logQ beats uniform random

The expectation is sound: popularity-proportional sampling with a correctly
estimated offline logQ should beat uniform random negatives. The report says the
opposite, but the arms are not comparable — the offline-logQ arm stopped five
epochs earlier than uniform random with about 75% of its LR schedule unused, and
its control was not matched (admitted in
[`evidence/rq11_negative_sampling.md`](../evidence/rq11_negative_sampling.md)).
The 500M leave-one-out arm is one of the artifacts finding 1 invalidates.

This question needs a rerun with matched horizons before any ordering is
reported.

## 6. RQ4's GELU-vs-SwiGLU tie is an early-stopping artifact

The selected widths are 192 (GELU) and 32 (SwiGLU) — 5.3× width and 3.4× FFN
parameters, which is the reader's objection and it is directionally right. The
more important number is that the gap is 31,702 parameters, 0.31% of a
10.13M-parameter model, so "SwiGLU needs a narrower FFN" is not a
capacity story at this scale.

SwiGLU led from epoch 7, peaked at epoch 9, and patience 3 killed it at epoch 12
of a 20-epoch linear horizon on a ±0.0006 wobble; GELU kept improving to epoch
15 and ran to epoch 18. The tie was as much a stopping-rule outcome as a measured
equivalence.

**Closed.** Both confirmations were relaunched under the horizon rule and now
train all 20 epochs (`..._cap40_ts2_r3_500m`). SwiGLU's best epoch moves from 9
to 13 and its recall@100 from 0.12985 to 0.13059, against GELU's unchanged
0.13095: the recall gap falls from 0.00110 to 0.00036, six times inside the
0.00215 band, and the coverage gap from 0.204 to 0.088. The families are a
genuine tie on ranking quality; GELU's coverage advantage survives.

## 7. RQ5's cosine-restart table is not a restart-only axis

The restart family carries a 5% warmup, and its one-cycle row is the same arm
RQ6 reports as "cosine warmup". Comparing 1 vs 2 vs 4 cycles inside that family
is fine; reading it against the main schedule table is not. The generator now
labels the axis `cosine cycles, warmup 5%`.

RQ5's headline — constant beats every decay treatment — is also the claim most
exposed to finding 1, because the frozen-tail stop penalises exactly the decay
shapes and not the constant arm.

## 8. RQ6's constant-warmup NDCG cell drifted from its source

`scratchpad/research_questions_500m.md` renders `+1% (0.0542)`; `README.md`
carries `0% (0.0542)` for the same cell. A hand edit diverged from the generated
table.

## 9. RQ8 was reported from incomplete evidence

[`evidence/rq8_rq9_current_confirmations.md`](../evidence/rq8_rq9_current_confirmations.md)
records RQ8 as incomplete and `agents/STATUS.md` has it blocked, yet the report
carried a sequence-length table with no caveat.

**Closed.** The MHA-4 and MHA-8 continuations landed, so all thirteen axes now
have their native-500M confirmation and `README.md` carries the full set. The
provisional sequence-only renderer that existed for the blocked period
(`_provisional_rq8_sequence_table` and the `--rq8-sequence-only` mode) is
deleted: it rendered the same table without control-relative deltas and
overwrote the complete one.

## 10. The report generator could not run at all

An LR-boundary point was launched twice under different round labels
(`..._lrboundary2_ts2_r2_50m` and `..._lrboundary4_ts2_r2_50m`): same config,
same seed, byte-identical metrics, 22 minutes apart. The ambiguity guard treated
the pair as a broken lineage and refused to render the ledger, so the committed
reports were no longer reproducible from the artifacts. The guard now collapses
launches that agree on everything except wall clock and peak memory.

## 11. Tuning method

Hyperparameter tuning here uses a grid whenever more than one parameter moves.
For two or more parameters, random search or Optuna dominates grid search at
equal budget; grid search is only appropriate for a single parameter. Recorded
in [`protocol/tuning.md`](../protocol/tuning.md).

Following from that, the architecture treatments now hold the embedding rate at
0.032 — the value that wins across the already-tuned arms — and search the deep
rate alone. That is one moving parameter, which is the case a grid still
answers, and it stops a treatment differing from its control in how far its own
search reached. Existing arms keep the wider surface they already have; nothing
is discarded.

## 12. Validation is the reported evaluation window

Validation is the last seven days of the stream and is also what the report
scores. A recommender must be trained up to the moment it predicts, so no later
held-out period exists to select on. The epoch is therefore selected on the
window it is scored on. Accepted by the research lead as unavoidable and
documented in [`protocol/tuning.md`](../protocol/tuning.md); every arm carries
the same optimism, so the between-arm comparisons remain the readable part.

## What has to be rerun

Under the amended horizon rule the only binding gaps are RQ5's two inert
schedule arms (finding 2), at the fixed embedding rate 0.032, batch 1280, 50M:

1. `schedule_step` — deep 0.006 and 0.012.
2. `schedule_wsd_warmup5` — deep 0.003, 0.006 and 0.012.

Each trains its full 20-epoch horizon, so its decay engages and the run resolves
its own selection. If a winner lands on the end of that deep line, the line is
extended by one point before selection.

**Done.** Both arms were rerun with the 500M confirmations they implied, and
every report path regenerates. Findings 6 and 9 are closed. Finding 5 (RQ11
matched horizons) remains open as a reader caveat rather than a blocked
selection: it needs a matched-horizon, matched-pool rerun before an ordering
between offline logQ and uniform random can be reported, which `README.md` now
says on that question.

The general form of that caveat — the native-500M confirmations span 9 to 41
trained epochs because arms launched before and after the horizon rule stop
differently — is stated once in the report header.
