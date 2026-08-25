# Experiment plan — hyperparameter transfer

## Question and decision

- Research question: does a learning rate selected on a small, cheap run
  transfer to a large one, across three axes — model width, FFN width, and
  dataset size?
- Falsifiable hypothesis: the location of the learning-rate optimum is
  invariant along each axis. Under μP it is invariant in model width; with an
  FFN base width of its own it is invariant in FFN width; under a
  token-indexed horizon-free schedule it is invariant in dataset size.
- Decision this evidence will support: whether G1 can keep tuning on the 50M
  proxy and transfer, and whether RQ1's current negative result stands.

## Why the existing evidence cannot answer it

The 50M proxy surface is measured under a linear schedule that anneals to zero
over 20 epochs, but 1180 of its 1733 revision-2 runs stopped short of that
horizon on early-stopping patience. Only selected winners were continued to the
horizon, so at every unselected grid point the recorded metric is the value the
run held when patience fired mid-decay, not the value its rate would reach.

The bias is not symmetric: the surviving points are systematically the ones that
spent more of their schedule. RQ1 reads a 6% and a 20.5% transfer regret off
this surface, and in both cases the transferred point is the truncated one — at
width 256 it peaked at epoch 5 and was killed at 8, against an oracle that ran
to 20. The same defect produced RQ4's wrong SwiGLU width. No conclusion about
transfer can be drawn until the compared points spend the same schedule.

## Controlled comparison

- Unchanged control: `control/control`, dim 64, SwiGLU 171, batch 1280,
  linear 20-epoch horizon, μP base 16 / delta 32.
- Treatments and exact values:
  - **A. model width** — transformer dim ∈ {16, 32, 64, 128, 256}, item table
    fixed at 64, FFN width ratio-scaled as today.
  - **B. FFN width** — FFN width ∈ {32, 64, 128, 224} at dim 64, each run twice:
    once with today's ratio-derived μP base, once with
    `mup_base_ffn_dim=32, mup_delta_ffn_dim=64`.
  - **C. dataset size** — control architecture on 50M and 500M, each under the
    linear 20-epoch schedule and under the token-indexed `power` schedule
    (exponent −0.51, transition at the 50M control's token count).
- Factors held fixed: seed 42, batch 1280, no gradient accumulation, item table
  64 dims, validation every epoch, best weights restored, all other
  architecture and negative-sampling settings at the control.
- Treatment-specific parameters that must be tuned: none. This study measures
  where an optimum sits; it does not select a configuration.

## Data and evaluation

- Proxy dataset and native training recipe: Yambda-50M, native regime, the
  control recipe above.
- Full dataset and native training recipe: Yambda-500M, native regime, same
  recipe. Used only in arm C.
- Split, eligible cohort, target, and evaluation metrics: unchanged from G1 —
  last 7 days validation, likes only, recall@100 reported as the primary metric
  with ndcg@100, recall@10 and coverage@100 alongside.
- Early stopping, validation cadence, and epoch-cap extension rule: every arm
  trains its full declared horizon and reports the best epoch inside it. The
  linear arms are horizon-complete by the annealing rule and carry
  `lr_horizon_complete=True`. The `power` arms are horizon-free, so they keep
  early stopping with patience 3 and a cap of 40; a best epoch at the cap
  extends the cap and reruns.

## Tuning and promotion

- Fixed global batch size and any approved feasibility exception: 1280
  everywhere, no exception.
- Learning-rate grid and boundary-extension rule: two one-dimensional sweeps
  per arm, since only one parameter moves in each:
  - deep LR ∈ {0.003, 0.006, 0.012, 0.024, 0.048, 0.096} at embedding LR 0.032
  - embedding LR ∈ {0.008, 0.016, 0.032, 0.064, 0.128, 0.256} at deep LR 0.012

  A sweep whose argmax lands on an endpoint is extended by one point in that
  direction and rerun before it is read.
- Other approved tuning axes: none.
- Proxy winner rule: none. The readout is the argmax position of each curve and
  whether it moves along the axis, not a configuration to promote.
- Full-data confirmations, repeats, and shared empirical noise bands: single
  seed. Curves are compared against the existing empirical bands (recall@100
  0.00215019 on 500M); a shift of the argmax by one grid step is reported as
  unresolved unless the two neighbouring points differ by more than the band.

## Execution and verification

- Implementation and focused checks: `mup_base_ffn_dim`/`mup_delta_ffn_dim` on
  `MuTransferGenerationExperiment` (landed, covered by tests in
  `dcn/tests/test_generation_e2e.py`). Arm C needs the `power` schedule wired
  to a transition token count; the schedule itself is already implemented.
- Queue batches and dependencies: three independent batches, no ordering
  between them. Arm A 60 runs at 50M, arm B 48 at 50M, arm C 12 at 50M and 12
  at 500M. Roughly 3–4 GPU-hours in total; all submitted to the persistent
  `utils/training_queue` service.
- Evidence and report tables to produce: one curve table per arm
  (`evidence/transfer_study.md`), plus a corrected RQ1 section if arm A
  overturns it.
- Expected result and conditions that require debugging or another sweep:
  arm A is expected to show aligned optima once every point spends its horizon,
  which would retract RQ1's 6%/20.5% regret. Arm B is expected to show a moving
  optimum under the ratio-derived base and a fixed one under the FFN base. If
  arm B's optimum still moves, the FFN base is not the whole story and the
  attention and readout scaling need the same audit. Arm C has no expected
  answer; a moving optimum under both schedules means dataset-size transfer
  needs a different instrument than the schedule.
- Independent review criteria: every reported point carries
  `lr_horizon_complete=True` or is a horizon-free shape; no curve is read with
  its argmax on an unextended endpoint; the argmax claim is stated against the
  empirical band.

## Approval

- Exact plan approved by the user: yes, as a directive rather than a reviewed
  plan — "do a better testing then. Also mu p/ mu transfer should work with
  swiglu too. It would be ideal for you to find the way to transfer
  hyperparameters across dataset sizes." The three arms were chosen by the lead
  to answer it; the user did not see this document before the runs.
- Approval reference/date: session message, 2026-08-17
