# G6 native-500M rerun plan

## Question and decision

- Research question: rerun G6 RQ0-RQ3 natively on Yambda-500M and replace
  their native-50M reader evidence with the native-500M selections.
- Falsifiable hypothesis: a natively fitted and tuned semantic-history bundle
  can improve Recall@100 over the native-500M best-G1 item-ID history control
  without a beyond-band NDCG@100 regression, while the complete bundle remains
  better than the original G1/SASRec baseline.
- Decision: select the native-500M representation, SID initialization,
  tokenizer, and collision policy; use that atomic bundle in G6's aggregate
  comparison and report native-500M results instead of the old RQ0-RQ3
  native-50M results.

## Controlled comparison

- G6 primary baseline: a newly tuned batch-512 native-500M reconstruction of
  the original two-layer G1/SASRec item-ID baseline. It is the first row and
  percentage-delta reference in every primary reader comparison.
- Cumulative best-G1 control: a newly tuned batch-512 native-500M
  reconstruction of the final best-G1 item-ID combination. It is the local
  control that isolates the SID-history marginal and appears in a separate
  diagnostic table; it never replaces the primary baseline.
- Historical batch-1280 G1 artifacts and the retired batch-640 size surface provide
  configuration anchors and immutable audit evidence only. They are not
  selection-eligible in the batch-512 rerun.
- No parameter-count, width-matched, or compute-matched control is added.

## Data and evaluation

- Main dataset: native Yambda-500M for every RQ0-RQ3 tuning and final result.
- User validation: the user requested that all G6 experiments be rerun on
  500M and reported instead of their 50M results.
- Use every eligible user, sampled only by user ID, with likes, core at least
  five, the final-seven-day temporal holdout, train-mapped items, full-catalog
  scoring, and no seen-item exclusion.
- Recall@100 is primary. Also report NDCG@100, MRR@100, capped Recall and
  Coverage at 10/50/100, exact and every-depth prefix SID Recall, ICR,
  occupied load, intra-code cosine, reconstruction residuals, collision
  buckets, and eligible target-frequency and collision slices.
- Reuse the reviewed native-500M relative dispersions: Recall@100 1.685%,
  NDCG@100 1.966%, MRR@100 2.085%, and Coverage@100 13.429%. For every
  comparison and metric, define its operational absolute band as that metric's
  relative dispersion times the comparison's own unrounded reference value:
  original G1 for primary tables, best-G1 for SID marginals, random
  initialization for RQ1, and the frozen RQ0 anchor for RQ2/RQ3. Launch no new
  repeat calibration.
- Every newly launched run uses physical batch 512 without accumulation,
  validates every epoch, trains exactly 26 epochs, and restores its best
  validation checkpoint. Horizon is not a search axis. Best-G1 and SID runs
  complete their annealed schedules at epoch 26; the conventional original-G1
  control and its bridge preserve their tuned constant-LR shape without early
  stopping.

## Shared tokenizer and selection contract

- Fit RQ-KMeans on normalized 128-dimensional content vectors with seed 42.
- Use one shared codebook size at every residual level. Search only levels
  `{3, 4}` and shared codes `{512, 2048, 8192}`.
- Use convergent Lloyd fitting with a 300-iteration safety cap, relative
  inertia tolerance `1e-4`, and assignment early stopping. Reject cap hits.
  These replace the old fixed-iteration quality axis: all candidates must
  converge, so cap and tolerance are fixed algorithm semantics rather than
  tunable downstream-quality parameters. Levels and shared code count are the
  remaining applicable RQ-KMeans fitting axes. Fixing tolerance is an explicit
  protocol exception requested for approval; it is not inferred from intrinsic
  tokenizer quality.
- Count the collision suffix vocabulary against the 8192-symbol per-level cap.
- Every scrambled-Sobol design uses
  `torch.quasirandom.SobolEngine(scramble=True, seed=42)`, draws from index zero
  without fast-forwarding, and assigns axes in the order written below.
  Manifest anchors precede Sobol rows; resolved float coordinates and the full
  manifest hash are frozen before submission.
- Map a Sobol value `u` on an ordered discrete axis of length `n` to
  `min(floor(u*n), n-1)`. Map a continuous LR to
  `low * (high/low)^u`.
- Center embedding and deep LR coordinates on the already tuned native-500M
  controls and sample multiplicative offsets log-uniformly over the equivalent
  absolute ranges `[0.008, 0.512]` and `[0.002, 0.128]`.
- Select by validation Recall@100, then NDCG@100 when Recall is within the
  scaled band, then deterministic manifest order. Runtime is not a tie-break.
- When a selected LR is in the outer 10% of its interval, freeze every other
  value and add four outward points at factors `sqrt(2)`, `2`, `2sqrt(2)`,
  and `4`. A second boundary win returns for approval.

## Control calibration

- Best-G1 control: at the fixed 26-epoch horizon, run twelve deterministic
  joint LR coordinates: anchor embedding/deep rates
  `0.0468526465053628/0.032703745675187676` from
  `g1_aggregate_aggregate_none_l4_e0p0468526465053628_d0p032703745675187676_h15_c0_initial_ts2_r1_500m`,
  followed by Sobol indices 0-10 over `(embedding LR, deep LR)`. The horizon is
  reused rather than retuned. Budget: 12 runs, with at most eight approved
  LR-boundary cells.
- Original-G1/SASRec control: constant LR for exactly 26 epochs without early
  stopping and twelve deterministic joint LR coordinates: anchor embedding/deep
  rates `0.001/0.002` from
  `g1_transfer_selected_native50_af8b8a8133c7_e0p001_d0p002_cap40_ts2_r2_500m`,
  followed by Sobol indices 0-10 over `(embedding LR, deep LR)`. Budget: 12
  runs, with at most eight approved LR-boundary cells.

## RQ0: history representation

- Compare exactly the seven required representations on the best-G1 control:
  trainable SID event; item ID plus frozen SID event; item ID plus trainable
  and frozen SID event; trainable SID tokens; trainable and frozen SID tokens;
  frozen SID tokens; and interleaved item-ID/SID tokens.
- Keep concrete-item targets and output table, the last 100 history items,
  item-count truncation, deterministic collision suffix, backbone, sampler,
  and evaluation fixed. Use DenseNet for every concatenated representation.
- Fix representation width at 128. Tune the historically strongest item-ID plus
  frozen-SID-event family first with twelve cells: every one of the six
  `(levels, shared codes)` pairs appears twice, while the embedding/deep LR
  coordinates use deterministic scrambled Sobol points around the control
  rates. Freeze its selected tokenizer and LR pair as the starting anchor for
  the other six representation families.
- Give each remaining family eight cells: the inherited exact anchor, one cell
  at each of the other five tokenizer pairs with inherited LRs, and two local
  Sobol LR refinements at the inherited tokenizer. The anchor stays the first
  family's winner for all six searches, so family order cannot change the
  design. Each family still selects independently from its own cells.
- The primary table compares every representation with original G1/SASRec. A
  separate local table compares it with best-G1 item IDs to isolate the SID
  marginal. Promote a representation into the atomic bundle only if it beats
  the primary baseline beyond that comparison's Recall band without a
  beyond-band NDCG regression and also beats the best-G1 local control beyond
  its Recall band without a beyond-band NDCG regression. Otherwise retain the
  no-SID best-G1 aggregate explicitly.
- Bridge only the selected representation to the original-G1/SASRec control
  with eight LR cells: the selected representation's resolved LR pair as the
  first coordinate, followed by Sobol indices 0-6 over
  `(embedding LR, deep LR)`.
- Expected budget: `12 + 12 + 12 + 6*8 + 8 = 92` runs. Two controls, seven
  representation surfaces, and the selected bridge each permit at most eight
  LR-boundary cells, so the maximum is 172 runs.
- The old controlled learned-SID remediation is historical. If the native-500M
  item-plus-trainable/frozen family is unexpectedly worse than the matched
  frozen-only family after tuning, the required correctness investigation
  reopens that controlled residual comparison before RQ0 is finalized.

## RQ1: SID lookup initialization

- Depend on RQ0. Use its winner when that representation has a trainable base
  SID lookup; otherwise use the strongest applicable RQ0 family as a diagnostic
  without replacing the RQ0 winner.
- Compare random truncated-normal initialization with deterministic
  content-PCA initialization. The content arm centers float64 centroids,
  projects them to the selected lookup width with deterministic SVD signs,
  matches each random block's RMS, overwrites only base-code rows after global
  initialization, leaves special/suffix rows identical, and does not advance
  RNG state.
- Freeze representation, tokenizer, capacity, collision policy, data, and
  backbone. Run six paired LR coordinates at the fixed horizon: the exact RQ0
  anchor plus Sobol indices 0-4 over `(embedding-LR multiplier, deep-LR
  multiplier)`. Reuse the exact compatible RQ0 random anchor. Confirm both
  frozen winners at seeds 43-45. Choose coordinates on seed 42; decide promotion
  and report final quality from the paired four-seed means.
- Precedence proposed for approval: a beyond-band Recall improvement without a
  beyond-band NDCG regression wins even if convergence is slower. When final
  quality is within the bands, content initialization wins only if it
  converges faster; otherwise retain random. Define faster convergence as both
  a lower paired mean epoch-to-95%-of-final and a higher paired mean normalized
  Recall AUC. A tie or disagreement between those indicators is not faster.
- Expected new budget: 17 runs: eleven new seed-42 cells after exact-anchor
  reuse, then six confirmations. Boundary extension remains paired across both
  initialization arms; the maximum is 33 runs.

## RQ2 and RQ3: collision policy and tokenizer

- Freeze the RQ0 representation and the RQ1 initialization only when RQ1
  applies to that representation. Treat representation, initialization,
  tokenizer, and collision policy as one atomic terminal G6 bundle.
- If no SID representation passes RQ0's promotion rule, use the highest-Recall
  eligible SID family as a frozen diagnostic base for RQ1-RQ3. It does not
  replace best-G1 item IDs in the terminal aggregate unless a later atomic SID
  bundle passes the promotion rule.
- Pair every coordinate across the deterministic suffix policy and no-suffix
  policy. Only the collision policy differs.
- Start from the selected RQ0/RQ1 tokenizer and LR pair. At seed 42, evaluate
  every one of the six `(levels, shared codes)` pairs at the inherited LRs for
  both collision policies, then give each policy winner four paired local-LR
  refinements. This is 20 seed-42 cells, one of which is the exact reusable RQ0
  suffix anchor, so 19 are new. Policy comparisons remain paired.
- Select each policy independently. RQ2 retains the RQ0 setting whenever its
  selected suffix system has any unrounded Recall regression. Promote a changed
  tokenizer or policy only when Recall improves beyond the band with no
  beyond-band NDCG regression. If both policies qualify, use Recall, NDCG
  within the band, then manifest order. The policies are mutually exclusive.
- Freeze seed-42 winners and confirm both policy winners and the RQ0 anchor at
  seeds 43 and 44, reusing exact overlaps. Choose coordinates on seed 42;
  decide promotion and report final quality from the paired three-seed means.
- The seed-42 suffix anchor is reused from RQ0 or RQ1. Expected new budget is
  21, 23, or 25 runs. Starting from 19 new seed-42 cells, confirmations add six
  when neither overlap applies, four when either the suffix winner equals the
  RQ0 anchor or RQ1 already confirmed that anchor, and two when both apply.
  Boundary extension remains paired across both policies; the maximum is 41
  runs.
- If the terminal bundle differs from RQ0's original-G1 bridge, run a new
  eight-cell bridge: its source anchor followed by Sobol indices 0-6 over
  `(embedding LR, deep LR)`; maximum 16 with its LR boundary. The selected terminal
  best-G1 run is already the aggregate and is not duplicated.

## Execution and reporting

- Dependency order: controls, RQ0, RQ1, paired RQ2/RQ3, conditional terminal
  bridge, aggregate/report.
- The feasible expected totals are `{130, 132, 134, 138, 140, 142}` runs: RQ0
  contributes 92, RQ1 contributes 17, RQ2/RQ3 contributes one of
  `{21, 23, 25}`, and a changed terminal bridge contributes either zero or
  eight. Including every approved boundary branch, the maximum is
  `172 + 33 + 41 + 16 = 262` runs. Any
  second boundary win, cap failure, or reopened
  correctness investigation returns for approval and is outside this bound.
- Submit each approved multi-run stage through the existing persistent queue.
  Reuse only exact dataset/configuration/cache identities; preserve every old
  50M and batch-640 artifact in audit storage.
- Replace RQ0-RQ3 reader, compact, and tuning tables with native-500M evidence.
  Use separate tables for original-G1/SASRec and best-G1 controls, keeping
  `Method`, `Recall@100`, and `Delta Recall@100` adjacent. Omit performance
  columns unless a material anomaly is needed to establish validity.
- The aggregate table compares original G1/SASRec with the trained terminal
  best-G1-plus-SID bundle. Decompose that gain into the cumulative best-G1 gain
  and terminal SID marginal, and report the unrounded interaction gap for
  Recall@100, NDCG@100, MRR@100, and Coverage@100. For each metric, set
  interaction resolution to that metric's approved native-500M relative
  dispersion times the unrounded original-G1 baseline. Label the gap positive
  or negative only beyond that resolution; otherwise label it unresolved. If
  no SID bundle passes both promotion comparisons, the aggregate is best-G1
  without SID and no duplicate no-SID run is launched.
- Require focused implementation tests and a blind protocol/code review before
  launch, then independent artifact/result/report review after completion.

## Acceptance criteria

- RQ0: the final model must not be much worse than SASRec. Ideally it should
  be better.
- RQ1: codebook initialization should converge faster than random
  initialization, and faster convergence counts only when final recall is
  non-inferior.
- RQ2: the selected collision-token configuration must not worsen downstream
  Recall@100 versus the RQ0 setting.
- RQ3: compare independently tuned systems with and without collision
  resolution.
- Replace the old RQ0-RQ3 native-50M reader results with the completed
  native-500M results.

## Approval

- Material choices: original-G1/SASRec as G6's primary baseline with best-G1
  as the local SID-marginal control; batch-512 reruns; fixed 26-epoch horizon;
  exact reduced axes and sequential budgets; RQ1 quality-versus-convergence
  precedence; and terminal conflict rule. Fixing K-Means convergence at a
  300-iteration cap, `1e-4` tolerance, and assignment early stopping instead of
  tuning tolerance is an explicit requested exception.
- Exact requested scope: one of `{130, 132, 134, 138, 140, 142}` expected runs
  and 262
  maximum new full runs plus focused non-selection smoke checks.
- User approval: approved on 2026-08-31.
