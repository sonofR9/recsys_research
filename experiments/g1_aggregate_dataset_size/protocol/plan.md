# G1 aggregate dataset-size study

## Question and hypothesis

- Research question and status: a separate experiment whose explicit research
  axis is the G1 aggregate on native 50M versus native 500M; approved for
  implementation and training.
- Current understanding: the native-500M aggregate is complete. Native 50M has
  no compatible aggregate, exact baseline, bridges, or uncertainty band.
- Falsifiable hypothesis: the selected eleven-member aggregate improves
  Recall@100 and NDCG@100 over the exact original baseline in both native data
  regimes.
- Why the result matters: this closes G1 with measured aggregated improvement,
  rather than adding deltas from separately trained treatments.

## Comparison

- At each size, compare the current-protocol original G1 baseline against one
  jointly trained aggregate with the same eleven approved members:
  1. dropout-matched SwiGLU width 192;
  2. deep-only one-cycle cosine with 5% warmup;
  3. ALiBi plus learned forward/reverse positional embeddings concatenated to
     the item representation;
  4. post-LayerNorm blocks;
  5. input and final RMSNorm;
  6. end-only CLS querying;
  7. 32-bin additive time plus raw reverse-timestamp RoPE;
  8. popularity-catalog global-q with 2,048 negatives and positive-and-negative
     Yi-2019 correction;
  9. grouped-query attention with 2 query heads and 1 key/value head;
  10. a learned BOS token;
  11. depth selected independently from 4, 6, and 8 layers.
- Keep dimension 64, sequence length 100, full causal attention, dropout 0.1,
  seed 42 for selection, and the existing likes/split/catalog protocol fixed.
  Native 500M retains its frozen global batch 1,280. Native 50M uses the batch
  calibrated and then frozen under this plan.
- Preserve the existing 500M selection and raw evidence exactly. Reuse its
  selected four-layer aggregate and all eleven bridges after artifact replay.
- RQ12 is already represented by end-only CLS. Exclude RQ13 because its
  encoder-decoder winner is worse than decoder-only. Exclude the mutually
  exclusive RQ14/RQ15 decoder-decoder family: the user validated its internal
  changes as statistically insignificant, it lacks a matched aggregate bridge,
  and it is not an additive member of the decoder-only aggregate.

## Data and evaluation

- Dataset size is the explicit axis of this experiment: native Yambda-50M and
  native Yambda-500M are separate regimes. This experiment has its own
  identity, report, evidence, and work item; it is not another RQ inside the
  single-size native-500M G1 experiment. Never repeat 50M data to match 500M
  steps, targets, tokens, or wall time.
- Sample by user ID only. Use likes, core items with at least five interactions,
  the final seven-day timestamp holdout, mapped training items, full
  mapped-catalog scoring, and no seen-item exclusion.
- Recall@100 is primary and NDCG@100 breaks unresolved Recall@100 ties. Also
  report Recall@10, NDCG@10, and Coverage@100.
- Reuse the approved native-500M operational bands. For native 50M, run ten
  exact selected-baseline repeats, seeds 42 through 51. The unrounded sample
  standard deviation of each metric is its single-run resolution band; render
  the band rounded upward to three decimals. These are practical resolution
  diagnostics, not confidence intervals or treatment-specific significance
  tests, and are never transferred between sizes.

## Hyperparameter selection

- Use the fixed-width `MuTransferGenerationExperiment` family. Tune embedding
  and deep learning rates independently for the 50M baseline and for each 50M
  aggregate depth because neither optimizer group has compatible 50M evidence.
- Calibrate the 50M experiment-global batch and baseline rates together on the
  unchanged original G1 baseline. Test batches `512` and `1280`, each with the
  same three `(embedding LR, deep LR)` candidates:
  `(0.003261002414691765, 0.025343654763668278)`,
  `(0.0011832644052772452, 0.06640811442971185)`, and
  `(0.006775906584815153, 0.012851178723155708)`. Select within each batch and
  then between batch winners by validation Recall@100, NDCG@100, and stable
  run name. Require all six verified runs, then freeze the winning batch and
  its own LR pair for later 50M baseline repeats and bridges.
- The fixed-LR `160/320/480/640/1280/2560` artifacts are immutable audit-only
  evidence and cannot participate in corrected selection or its tuning table.
- Each aggregate depth starts with the deterministic three-trial joint random
  search:
  `(0.032, 0.012)`,
  `(0.019275929014542306, 0.01942482874826853)`, and
  `(0.046127309413540894, 0.00854511881682427)` for
  `(embedding LR, deep LR)`. The latter two are log-uniform draws from
  `[0.016, 0.064] x [0.006, 0.024]` using `random.Random(1)`.
- If a noncentral draw wins, add exactly three local log-uniform pairs in the
  factor-two box around it, clipped to the original bounds. Use seed 101 after
  the first draw and seed 102 after the second. The generated pairs are:
  - first draw:
    `(0.02667329716331745, 0.011583588892357813)`,
    `(0.03739158126074027, 0.022404888315289213)`, and
    `(0.024128597404038644, 0.017700854385094125)`;
  - second draw:
    `(0.0268290495956486, 0.011433182352002583)`,
    `(0.027424687121848843, 0.012648955509984123)`, and
    `(0.042889501869815085, 0.008971599022863638)`.
- Re-select once. If a selected coordinate is in the outer quarter of its
  original log interval, add three probes beyond that boundary while holding
  the other coordinate fixed:
  - embedding low: `0.004/0.006349604207872798/0.010079368399158985`;
  - embedding high: `0.10159366732596477/0.16126989438654377/0.256`;
  - deep low: `0.0015/0.0023811015779522994/0.00377976314968462`;
  - deep high: `0.03809762524723678/0.06047621039495391/0.096`.
  If both coordinates qualify, run all six. Allow one boundary round and
  return for approval if its outer point still wins.
- Select within a family by validation Recall@100, then same-epoch NDCG@100,
  then stable run name. Select aggregate depth by the same rule.
- Baseline, repeats, and horizon-free bridges use an 80-epoch safety cap,
  patience three, validation every epoch, and restore the best checkpoint. If
  stopping has not triggered strictly before epoch 80, return for approval;
  the result is not usable.
- Aggregate candidates and the scheduler bridge complete the inherited H15
  one-cycle horizon without early stopping and restore the best validation
  epoch within it. A result is usable only when its metadata proves the full
  horizon completed. A best epoch at 15 triggers an H24 rerun from the start
  for the affected three-rate depth surface; a best epoch at 24 triggers H36.
  Return for approval if H36 still ends at its best epoch.

## Run stages and compute

- First parameterize the existing aggregate config, manifest, verifier,
  launcher, collector, and report by dataset size without changing any frozen
  500M identity or output.
- Focused tests cover cross-size candidate isolation, the exact 50M composition,
  completed schedule traces, stopping-cap rejection, repeat identity, the
  3,414-user evaluation count, and unrounded bridge arithmetic.
- Native 50M stage 1 jointly calibrates batch and baseline rates with exactly
  six configurations: the same three LR pairs at batch 512 and batch 1280.
- Stage 2 adds nine baseline seeds to make ten exact repeats and freezes the
  native-50M bands before any treatment is selected or interpreted.
- Stage 3 runs thirteen matched bridges at the frozen baseline rates before
  aggregate training: ten fixed-member bridges plus depth-4, depth-6, and
  depth-8 bridges. The two unselected depth bridges remain diagnostic and are
  excluded from the eleven-member sum.
- Stage 4 runs three LR trials at each of 4, 6, and 8 aggregate layers and then
  selects the aggregate. The selected depth already has its required bridge.
  The initial total is 37 native-50M configurations.
- Aggregate local-LR and optimizer-boundary refinement can increase the total
  to 64. For an endpoint optimum, each affected depth repeats the same bounded
  LR procedure at H24 and, if needed, H36; these corrections add at most 72,
  and a scheduler bridge can add at most one H24 and one H36 rerun. The exact
  pre-approved maximum is therefore 138 configurations. An outer optimizer
  boundary, unresolved epoch-80 cap, H36 endpoint optimum, new
  treatment, or changed metric rule returns for approval.
- Submit granular jobs through the existing persistent training queue with no
  GPU exclusions, allowing all admitted GPUs to work concurrently.
- Run affected tests during implementation, obtain blind code review, then run
  `./test.sh` once when shared-host CPU load is low. Verify every selected raw
  artifact independently before reporting.

## Interpretation and reporting

- Generate this experiment's own README, separate 50M and 500M tuning ledgers,
  and evidence files. Its reader-facing `## Aggregated improvement` section has
  one table for each size. Link it from G1 without copying a second-size RQ into
  G1, and preserve every existing G1 RQ table and 500M value.
- For every metric and size, compute from unrounded values: baseline and
  aggregate absolute scores, aggregate gain in points and percent, the sum of
  eleven disjoint bridge gains, and the interaction gap `aggregate gain -
  summed bridge gains`.
- Preserve the already approved 500M interaction interpretation. At 50M,
  report the interaction gap as descriptive and leave its direction unresolved:
  a baseline single-run band does not calibrate an estimator containing eleven
  single-run bridges. Do not claim a cross-size difference without separately
  approved cross-size uncertainty.
- If the result contradicts the expected improvement, first verify the
  composition, schedule trace, restored checkpoint, full-catalog evaluation,
  timestamp handling, and proposal correction. If correct, investigate and
  explain the cause with experimental evidence.

## Acceptance criteria

- Run the best combination of all changes and report the metrics and uplift on
  both the 50M and 500M datasets.

## Approval

- Material choices: treat dataset size as a separate explicit research axis;
  calibrate and freeze the 50M global batch; independently select 4/6/8-layer
  depth; tune both optimizer groups with the exact bounded policy above; create
  native-50M bands from ten exact repeats before aggregate selection; reuse all
  verified 500M artifacts; leave 50M interaction direction descriptive; and
  exclude the unresolved mutually exclusive decoder-decoder challenger.
- Exact scope requested for approval: implementation, review, and the corrected
  37-run native-50M initial path, with a 138-configuration maximum including
  every declared LR and H24/H36 correction.
- User approval: approved on 2026-08-28.
