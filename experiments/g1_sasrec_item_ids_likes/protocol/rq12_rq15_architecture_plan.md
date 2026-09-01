# RQ12-RQ15 query-architecture plan

## Question and hypothesis

- Research questions and status: RQ12 is `complete`, RQ13 is `review`, and
  RQ14-RQ15 are `wip` while RQ15 tests whether dense NTP supervision closes
  RQ14's architecture-level quality gap.
- Current understanding: native-500M RQ8 already resolves standard item-state,
  end-only CLS, and interleaved autoregressive CLS querying on one frozen
  decoder-only surface. The repository has causal history retrieval and generic
  cross-attention blocks, but no bidirectional item-history encoder, bounded
  prefix-expansion dataset, four-slot decoder memory, or two-stage retrieval
  objective.
- Falsifiable hypotheses:
  1. end-only CLS remains the best of the three existing decoder-only layouts;
  2. bounded prefix expansion improves an encoder-decoder because its encoder
     otherwise receives only one downstream target per user;
  3. four distinct first-decoder query slots outperform four repetitions of one
     token, and retaining history states in cross-attention helps when four
     summary states are insufficient; and
  4. NTP-pretrained or auxiliary-NTP first decoders improve the final
     decoder-decoder candidate query over downstream-only joint training.
- Why the result matters: these RQs separate query construction, amount of
  downstream supervision, memory representation, and representation
  pretraining instead of attributing all effects to one architecture label.

## Comparison

### Common architecture surface

- Reuse the exact native-500M RQ8 query recipe: history length 128, model width
  64, two history-stack layers, the existing item table and sampled-softmax
  objective, and all other frozen RQ8 query fields.
- Decoder-only arms retain their already matched physical windows: 50 for
  standard, 51 for end-only CLS, and 100 for interleaved CLS.
- The encoder-decoder history encoder has two bidirectional full-attention
  layers. The decoder has one learned query token and one causal-self-attention
  plus cross-attention layer with SwiGLU intermediate width 128. Its only output
  state is the retrieval query.
- The decoder-decoder first stack has two causal layers and appends four query
  slots after the history. Its physical causal window is 54, so the fourth slot
  can still see 50 history items after the three preceding CLS slots. Its second
  decoder has one learned query token and one causal-self-attention plus
  cross-attention layer with SwiGLU intermediate width 128. It consumes all
  four CLS states directly; no averaging, concatenation, or projection fuses
  them first.
- Both cross-attention architectures score candidates with the same item table,
  negative proposal, correction, and sampled-softmax definition as the frozen
  RQ8 recipe. The predicted next item is not present in encoder or first-decoder
  memory.

### RQ12: decoder-only query layout

- Standard item-state query.
- One end-only CLS query.
- Interleaved `[item1, CLS, item2, CLS, ...]`; every CLS predicts the following
  item.
- The abandoned decoder-only four-end-CLS arm is not implemented or reported.
- All three rows are reused only after exact artifact compatibility checks.

### RQ13: encoder-decoder prefix expansion

- One example per user: the latest available history, up to 128 items, predicts
  its next item.
- Truncated expansion, latest 8 prefixes per user: every example retains up to
  its latest 128 preceding items and may be shorter.
- Truncated expansion, latest 16 prefixes per user: identical except for cap 16.
- Required-length expansion, latest 8 prefixes per user: eligible prefixes have
  at least 128 preceding items and retain exactly the latest 128. A user whose
  complete training history has fewer than 128 preceding items contributes
  exactly one whole-history example.
- Required-length expansion, latest 16 prefixes per user: identical except for
  cap 16.
- Caps 8 and 16 are independent reported treatments, not a selection search.
  Prefix endpoints are the latest eligible ones, not random or evenly spaced.

#### Approved RQ13 cap extension

- Preserve every existing RQ13 row and raw artifact. Add latest-4 truncated
  prefixes at deep LRs 0.006, 0.012, and 0.024 as a fourth cap-response anchor.
- Select each cap's LR by validation Recall@100, then validation NDCG@100 and
  training time. Complete geometric LR-boundary follow-ups before fitting.
- Fit validation Recall@100 at caps 1, 4, 8, and 16 with the predeclared
  unweighted constrained curve `A - B * cap^(-p)`, where
  `max(y) <= A <= 1`, `0 <= B <= A`, and `0.05 <= p <= 2`.
- The selection target is `1.10 * mean([0.1367, 0.1343, 0.1363])`, the frozen
  RQ12 standard decoder validation Recall@100. The separate reader success
  target is `1.10 * 0.13468336146286186` full-user Recall@100. Full-user
  metrics never select a cap or learning rate.
- The practical ceiling is the minimum of: the largest cap supported by at
  least half the users; the largest cap whose source-exact input-token count
  is at most twice cap 16; and cap 32, one doubling past the largest observed
  anchor. Expanded-example count is diagnostic only. The audited native-500M
  values are support cap 45, token-compute cap 35, and practical cap 32.
  The source calculation must reproduce the logged cap-16 input-token count
  `66,404,954` before it is trusted; it then gives `122,550,944` tokens and
  `1,772,396` expanded examples at cap 32.
- Select the first integer cap at least 17 whose fitted validation curve reaches
  the validation target, capped by the practical ceiling. If it does not cross
  within the ceiling, run cap 32 as a bounded boundary probe and make no
  target-attainment claim.
- Report all 16 independent `y_i +/- 0.003` sensitivity fits and four
  leave-one-cap-out fits as diagnostic envelopes, not confidence intervals.
  Also fit the non-selecting comparator `A - B * exp(-k * (cap - 1))`, with
  `0.0001 <= k <= 2`. A cap-32 prediction disagreement above 0.003 or an
  asymptote-target disagreement marks extrapolation as model-dependent but
  does not change the bounded cap-32 selection.
- Bind the fit to the exact four contributing artifact hashes and the
  source-history manifest. Missing points, unresolved LR boundaries, changed
  frozen controls, invalid source counts, optimizer failure, or nonunique
  primary fit fail closed and cannot launch stage two.
- Train the selected practical cap at deep LRs 0.006, 0.012, and 0.024, then
  continue geometric LR-boundary follow-ups until its winner is interior.

### RQ14: decoder-decoder query memory

- Four repetitions of one shared CLS token; second decoder attends CLS only.
- Four distinct tokens `CLS_0` through `CLS_3`; second decoder attends CLS only.
- Four repetitions of one shared CLS token; second decoder attends history plus
  CLS.
- Four distinct tokens `CLS_0` through `CLS_3`; second decoder attends history
  plus CLS.
- Every arm trains both decoders jointly from scratch with downstream candidate
  loss only. The four CLS states are all passed separately to cross-attention.

#### Approved pretrained reinvestigation

- Re-run the same four architecture treatments with the RQ15-selected training
  method: load the exact selected first-decoder NTP checkpoint, newly initialize
  the CLS slots and second decoder, and jointly fine-tune both decoders with
  candidate loss only. No auxiliary NTP loss is used during fine-tuning.
- Every treatment uses the same source-checkpoint digest, native Yambda-500M,
  seed 42, batch 1280, the frozen model capacities, and the complete 20-epoch
  linear schedule applied to both optimizer groups. Validate every epoch and
  restore the best validation epoch within the completed schedule horizon.
- Fix embedding LR at the RQ15-selected `0.00025`. Tune deep LR independently
  for every architecture at `0.000375`, `0.00075`, and `0.0015`. If an outer
  deep LR wins, continue geometrically in that direction until the winner is
  interior.
- The initial surface contains 12 cells. Reuse the three exact compatible
  distinct-CLS/CLS-only RQ15 cells after an artifact-bound compatibility audit;
  submit the remaining nine cells through the persistent shared queue.
- A pretrained treatment near the old candidate-only Recall@100 regime is a
  correctness failure until checkpoint loading, slot initialization, gradients,
  memory construction, targets, schedule, and evaluation have been verified.
- Preserve the usable candidate-only comparison as a separately labelled
  training-regime result. The pretrained comparison supplies the current RQ14
  architecture decision. If it selects an architecture other than the one used
  for RQ15, revalidating RQ15 requires a separate approved plan.

### RQ15: decoder-decoder training method

- Use four distinct CLS tokens and the better CLS-only or history-plus-CLS
  memory selected within the distinct-token RQ14 rows.
- Joint scratch: both decoders train from scratch with downstream candidate loss
  only. Reuse the eligible selected RQ14 row as the compatible centre point and
  train the remaining joint-LR surface.
- Pretrain then fine-tune: initialize the first decoder from the compatible
  selected native-500M RQ8 standard NTP checkpoint, newly initialize the four
  CLS tokens and second decoder, then jointly fine-tune both decoders with
  downstream loss only. If the compatibility gate fails, run a dedicated
  first-decoder NTP pretraining stage instead.
- Simultaneous auxiliary NTP: train both decoders from scratch with downstream
  candidate loss plus first-decoder NTP loss. Average each loss over its own
  targets and use `loss = candidate_loss + 1.0 * ntp_loss`.
- If the auxiliary arm regresses, run weights 0.1 and 0.3 at its selected
  embedding/deep-LR pair as pre-approved corrective follow-ups.

## Data and evaluation

- Single dataset size: native Yambda-500M for every tuning and final run.
- User validation: G1's native-500M size was previously validated, and the user
  approved retaining the frozen RQ8 surface in this planning session.
- No user sampling. Likes only, core items with at least five interactions,
  final seven days held out, mapped items only, full mapped catalog, and the
  existing G1 seen-item evaluation rule.
- Primary metric: Recall@100. Secondary metrics: NDCG@100, Recall@10, NDCG@10,
  and coverage@100.
- Native-500M operational bands remain 0.003 recall, 0.001 NDCG/MRR, and 0.1
  coverage, derived from the shared ten-run unchanged-control evidence.
- No new seed repeats are required. Existing repeated RQ8 evidence remains
  intact; new selected runs use the shared empirical bands.

## Hyperparameter selection

- Physical and effective batch stay 1280. A memory failure does not authorize a
  silent batch, accumulation, or negative-pool change; it returns for a material
  plan correction.
- Every treatment checks three deep LRs for every embedding LR. Joint scratch
  and pretrain/fine-tune use the full Cartesian product of embedding LR
  `0.032, 0.064, 0.128` and deep LR `0.00075, 0.0015, 0.003`. Simultaneous
  auxiliary NTP uses the full Cartesian product of the same embedding LRs and
  deep LR `0.003, 0.012, 0.048`.
- The dedicated NTP source independently selects deep LR `0.024, 0.048, 0.096`
  at embedding LR `0.064` before downstream pretrained runs are submitted.
- Follow the global validation winner iteratively. When its embedding LR is an
  outer boundary, extend the embedding grid by a factor of two and train the
  method's full three-deep-LR row at the new embedding LR. When its deep LR is
  an outer boundary within that embedding row, extend one point at the same
  embedding LR using the method's geometric deep-grid ratio: two for joint
  scratch and pretrain/fine-tune, four for auxiliary NTP. Request both frontier
  additions when both coordinates are on boundaries, then recompute the global
  winner and continue until neither boundary wins.
- For pretrain/fine-tune only, replace the next geometric embedding point after
  low-boundary step 7 (`0.00025`) with the terminal frozen-embedding endpoint
  `0`. Train all three deep LRs there. Do not continue the embedding axis below
  zero; continue any independently required local deep-LR probe at either the
  last positive embedding LR or the frozen endpoint.
- Width, history-stack depth, second-decoder depth, prefix cap, CLS count, and
  cross-attention-decoder SwiGLU width 128 are user-fixed treatment definitions,
  not tuning axes. The report does not claim that 128 is the capacity optimum.
- The auxiliary NTP weight is user-fixed initially at 1.0. Weights 0.1 and 0.3
  are conditional corrections at the selected rate, not a joint LR/weight
  optimum search. Conclusions are limited to these defined recipes.
- All runs use the frozen RQ8 20-epoch linear annealing horizon, validate every
  epoch, complete the horizon without patience stopping, and restore the best
  validation epoch within it. A dedicated NTP pretraining fallback uses the
  same horizon and selects its validation-best checkpoint.
- μP base/delta construction is extended to the two new architecture families;
  each family owns one unchanged μP parameterization. No rate transfers between
  decoder-only, encoder-decoder, and decoder-decoder families are claimed.

## Run stages and compute

### Correctness and implementation

1. Add a generic bounded-prefix sequence policy whose cache identity includes
   length rule and cap. Tests prove latest-prefix selection, caps 8/16, exact
   128-event truncation, the one-example short-history fallback, chronology,
   and target exclusion.
2. Add bidirectional self-attention as a reusable transformer option while
   preserving the existing causal path. Tests prove within-history
   bidirectionality and prove that the held-out target never enters memory.
3. Add reusable shared/distinct end-query slots and extraction of either CLS or
   history-plus-CLS packed memory. Tests prove token identity, memory lengths,
   ordering, masks, and gradients to every CLS slot.
4. Add a cross-attention retrieval model with one learned decoder query and the
   existing item encoder/sampled-softmax interface. Tests prove one candidate
   target per expanded example and full-catalog query/item compatibility.
5. Add the auxiliary first-decoder NTP objective and checkpoint initialization.
   Tests prove NTP covers only item-to-next-item transitions, losses use separate
   denominators, downstream gradients update the first decoder without NTP,
   and the compatibility loader copies only the intended first-stage weights.
6. Add metadata for original users, expanded examples, candidate targets, NTP
   targets, and input tokens. Extend the focused generation E2E tests across all
   treatment modes, then run the complete non-GPU suite once after blind review.

### Full native-500M runs

- RQ12: zero new runs if all three frozen RQ8 selections pass compatibility.
- RQ13 original surface: 5 treatments by 3 deep rates = 15 preserved runs.
  Extension: 3 latest-4 runs, then 3 runs at the fitted practical cap, plus
  deterministic geometric LR-boundary follow-ups when an outer LR wins.
- Original RQ14 candidate-only surface: 4 treatments by 3 deep rates = 12
  preserved runs. Approved pretrained reinvestigation: 4 treatments by 3 deep
  rates = 12 cells, with 3 exact distinct-CLS/CLS-only RQ15 cells reused and 9
  new native-500M runs, plus deterministic geometric deep-LR boundaries.
- RQ15: 9 joint-scratch LR pairs, 9 pretrain/fine-tune pairs, and 9
  auxiliary-NTP pairs. Reuse the compatible selected RQ14 scratch centre, so 26
  downstream runs are new. Add 3 dedicated NTP-source runs before fine-tuning:
  29 new native-500M runs.
- Conditional additions: deterministic LR-boundary continuations; two auxiliary
  weights only after an auxiliary regression. Each RQ12 arm that
  fails compatibility receives its own three-rate seed-42 rerun; no new
  confirmation seeds are required because native-500M shared noise bands govern
  the comparison.
- All multi-run work is submitted as granular jobs to the one persistent
  training queue, without GPU exclusions. Execute and validate one RQ at a
  time. The user explicitly authorized starting RQ14 after the finalized RQ13
  report regardless of RQ13's result. RQ15 is not submitted until the user
  validates RQ14 and its distinct-token memory selection.

## Interpretation and reporting

- RQ12 retains the existing quality conclusion only if artifact configuration,
  dataset, objective, and evaluator digests match. Otherwise the mismatched arm
  is rerun under the frozen surface; historical evidence is preserved.
- Every treatment selects its LR using validation Recall@100, then validation
  NDCG@100 and lower training time as tie-breakers. Final full-user metrics are
  never used for treatment or hyperparameter selection.
- RQ13 compares each of four expansion rows with no expansion. Its best-method
  decision uses the selected validation Recall@100; rows within 0.003 choose the
  lower-cost method, then no expansion, cap 8, and cap 16 in that simplicity
  order. The caps remain separate reader conclusions. A final-metric gain must
  exceed the applicable band; an unresolved row does not justify its extra
  training cost.
- RQ14 reports paired shared-versus-distinct effects within each memory and
  CLS-only-versus-history-plus-CLS effects within each token identity. Its
  overall winner is validation-selected; a difference within 0.003 selects
  CLS-only memory, then shared tokens. RQ15 separately uses the higher
  validation-Recall@100 distinct-token memory; a difference within 0.003
  selects CLS-only.
- The pretrained RQ14 reinvestigation uses the same paired comparisons and
  simplicity rule. The reader report keeps the earlier candidate-only table
  separately and makes the pretrained table the current architecture decision.
  Full-user metrics do not select treatments. If an expected architectural
  effect remains unresolved after correctness checks, saved inference lesions
  remove history memory and individual CLS contributions to test whether the
  additional states influence candidate scores and metrics.
- RQ15 compares both NTP strategies with downstream-only scratch. Its winner is
  selected by validation Recall@100; a difference within 0.003 selects lower
  total training wall time and then downstream-only scratch. Pretraining cost is
  never omitted from its efficiency conclusion.
- Unexpected regressions require targeted checks of prefix counts and history
  slices, target leakage, attention masks, per-slot gradients, memory use,
  learning curves, loss scales, pretraining load digests, and LR boundaries.
- Every RQ receives two separate reader tables:
  1. candidate-generation quality with Recall/NDCG at 10 and 100 and
     coverage@100; and
  2. training efficiency with examples and candidate/NTP targets per epoch,
     input tokens per epoch, steady-state throughput, time through the selected
     checkpoint, and total required training wall time.
- Tuning ledgers contain every selected field and raw run. Reader tables contain
  only usable completed results. Existing RQ8 results and all rejected or
  superseded artifacts remain preserved in audit storage.
- Aggregate replacement is outside this approval at the user's explicit
  direction. No new aggregate is launched or claimed from these RQs; after the
  results resolve, aggregate compatibility and conflict precedence will be
  discussed with the user before a separate manifest or run is created.

## Acceptance criteria

### RQ12: decoder-only query layout


- everything is copied from already existing runs (since everything in this question has already been investigated in enough depth)

### RQ13: encoder-decoder prefix expansion

- 16 truncated prefixes should train in less epochs than 8 truncated prefixes. And probably have higher quality. And 8 truncated prefixes should train in less epochs then no extensions. Final quality can vary but if some of the methods give a much worse quality then the baseline, you should explain why.
- Everything is compared with similar regular decoder-only sasrec too.
- The cap-response fit must use validation Recall@100 only. It should choose a
  practical cap that reaches 110% of the regular decoder validation quality;
  if the target is outside the practical range, run cap 32 as a boundary probe
  and explain that the target was not demonstrated.
- The final selected-cap result should reach 110% of the regular decoder's
  mean full-user Recall@100. If it does not, first verify prefix slices, LR
  resolution, fit inputs, and source-derived ceilings; if they are correct,
  explain the gap with the saved curve, sensitivity, and model-form evidence.


### RQ14: decoder-decoder query memory

- decoder which cross attends both cls tokens and history probably should be better.
- 4 separate class tokens should probably have better metrics then the the same cls tokens.
- If statements above do not hold true, first debug and if everything is correct explain why is that so.

### RQ15: decoder-decoder training method

- Adding a pretraining stage should at minimum decrease training time without
  losing quality, and will most probably improve the main metrics.

## Approval

- Resolved user choices: caps 8 and 16 reported separately; latest eligible
  prefixes; bidirectional encoder; no decoder-only four-CLS arm; joint
  downstream-only RQ14 control; frozen RQ8 surface; two history layers plus one
  cross-attention decoder layer; required-length short histories retained once;
  initial auxiliary NTP weight 1.0 with conditional 0.1/0.3; compatible RQ8 NTP
  checkpoint reuse; cross-attention-decoder SwiGLU width fixed at 128 without a
  capacity search; aggregate discussion deferred until after these results.
- Exact scope requested for approval: implementation and focused tests, the 33
  initial native-500M runs, deterministic LR-boundary continuations, the two
  conditional auxiliary weights, dedicated pretraining fallback only if
  checkpoint compatibility fails, and three-rate reruns for any RQ12 arm that
  fails compatibility. This approves the full staged scope, not out-of-order
  submission: only the current RQ may be launched. Aggregate replacement is
  excluded.
- RQ15 joint-LR correction: the user approved checking multiple deep LRs for
  every embedding LR on 2026-08-27.
- RQ14 pretrained reinvestigation: the user approved the exact four treatments,
  fixed embedding LR `0.00025`, three-deep-LR surface, compatible reuse, native
  Yambda-500M dataset, and nine new initial runs on 2026-08-27.
