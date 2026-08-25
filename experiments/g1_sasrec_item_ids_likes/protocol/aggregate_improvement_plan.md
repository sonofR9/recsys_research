# G1 aggregated-improvement plan

## Question and hypothesis

- Research question and status: what ranking improvement remains when every
  selected G1 change is trained together? Approved for implementation and full
  native-500M runs.
- Current understanding: the component RQs used different local controls, so
  their published deltas cannot be added directly. Fresh one-factor bridges
  against one frozen original-model baseline are required.
- Falsifiable hypothesis: the aggregate improves Recall@100 and NDCG@100 over
  the baseline by more than the native-500M operational bands.
- Why the result matters: this model becomes G1's maximum-quality result and
  the candidate inherited by later experiments.

## Comparison

- Original-model baseline: current-protocol reconstruction of the homework G1
  model: dimension 64, two transformer layers, MHA 2Q/2KV, GELU width 256,
  pre-LayerNorm blocks, no input norm, final LayerNorm, learned-forward
  additive positions, full causal attention, sequence length 100, no time
  feature, no CLS, and 512 fixed leave-one-out logQ negatives.
- Aggregate members:
  1. dropout-matched SwiGLU width 192 with internal FFN dropout 0.1, so both
     the model width 64 and FFN width are divisible by 32;
  2. deep-only one-cycle cosine with 5% warmup;
  3. ALiBi plus revision-7 learned-forward-and-reverse item concatenation;
  4. post-LayerNorm blocks;
  5. input RMSNorm plus final RMSNorm;
  6. end-only CLS querying;
  7. the atomic 32-bin additive time feature plus raw reverse timestamp RoPE;
  8. popularity-catalog global-q with 2,048 negatives and positive-and-negative
     Yi-2019 correction;
  9. GQA with 2 query heads and 1 key/value head;
  10. a learned BOS token;
  11. aggregate depth selected from 4, 6, and 8 layers.
- The two normalization treatments operate at different locations and compose.
  Popularity global-q supersedes rather than combines with uniform negatives.
  Learned forward/reverse concatenation, ALiBi, timestamp RoPE, BOS, and CLS
  are jointly representable; BOS receives the oldest positional anchor, while
  the CLS query copies the latest timestamp and therefore has reverse timestamp
  position zero.
- The time representation is an atomic bundle because RQ9 selected the pair,
  not either member independently. Time features are a resolved family-level
  win against no time feature; the exact leading variants are mutually
  unresolved. The proposed conflict rule chooses the numerical Recall@100
  leader as the family's aggregate representative without claiming that it is
  better than the other co-leaders. The RQ11 rule selects popularity global-q
  for materially higher NDCG with non-inferior Recall@100. SwiGLU enters under
  the user's explicit RQ4 decision: Recall@100 is non-inferior at every tested
  depth and NDCG@100 resolves at two and eight layers. The user explicitly
  promotes GQA as a no-quality-loss efficiency change and BOS as a consistent
  weak positive, and requests direct aggregate selection across 4/6/8 layers.
- Factors held fixed: native Yambda-500M, seed 42, global batch 1,280, width 64,
  sequence length 100, full causal attention, dropout 0.1, the existing
  likes/split/catalog protocol, and current training semantics. Learning rates
  and aggregate depth follow the explicit rules below.
- RQ1 contributes no quality treatment. RQ6 is subsumed by RQ5. Sequence
  length 512, attention-window changes, and RQ10 PLE are omitted because they
  were unresolved or merely non-inferior. Losing FFN, query, position, time,
  scheduler, and negative-sampling alternatives are omitted.

## Data and evaluation

- All selection, bridge, aggregate, and final evidence uses native
  Yambda-500M, previously validated by the user for G1 continuation work.
- Sample by user ID only. Use likes, core items with at least five
  interactions, the final seven-day timestamp holdout, mapped training items,
  full mapped-catalog scoring, and no seen-item exclusion.
- Recall@100 is primary. NDCG@100 breaks unresolved Recall@100 ties. Also
  report Recall@10, NDCG@10, and Coverage@100.
- Reuse the experiment's single-run operational bands for both treatment and
  interaction interpretation: 0.003 Recall, 0.001 NDCG/MRR, and 0.1 Coverage.
  These bands came from the ten unchanged native-500M control repeats and stay
  a practical diagnostic rather than a significance test.
- This aggregation does not study dataset scale, so it has no 50M companion.

## Hyperparameter selection

- Use the fixed-width μP experiment class used by the current G1 treatment
  evidence. No width transfer occurs.
- Reconstruct and tune the original-model baseline first with constant LR,
  embedding LR fixed at 0.064, and deep LR 0.006/0.012/0.024. Freeze its best
  artifact by validation Recall@100, then NDCG@100, before any bridge or
  aggregate result is selected.
- A baseline winner at deep LR 0.006 adds 0.003/0.0015/0.00075; a winner at
  0.024 adds 0.048/0.096/0.192. Hold every other field fixed, allow one such
  three-probe round, and return for user review if its outer point still wins.
- Each bridge changes exactly one aggregate member from the frozen baseline
  and fixes the baseline's selected learning rates. Bridges measure matched
  standalone gains for aggregate arithmetic; the already accepted RQs remain
  the treatment-selection evidence.
- Composition invalidates both optimizer-group rates. At each of 4, 6, and 8
  layers, give the aggregate the same three-candidate, seed-42 joint
  random-search screen: the central
  `(embedding LR, deep LR) = (0.064, 0.048)` plus two log-uniform draws from
  `[0.032, 0.128] x [0.024, 0.096]` using Python `random.Random(42)`. The exact
  generated pairs are `(0.07764674795069047, 0.02484672863178322)` and
  `(0.0468526465053628, 0.032703745675187676)`.
- If the first random draw wins, add local pairs
  `(0.06957712293357378, 0.04045869601192933)`,
  `(0.058511920889791694, 0.03301809071022853)`, and
  `(0.08595109115349953, 0.03835282700655922)`. If the second draw wins, add
  `(0.06547725593418215, 0.027941927666112344)`,
  `(0.06295219371532267, 0.04651981979406204)`, and
  `(0.046467420627589774, 0.0404566728379668)`. These are log-uniform draws
  from the original bounds clipped to a factor-two box around the winner,
  using seeds 4201 and 4202 respectively.
- Re-select after that one local round. For a winning coordinate in the outer
  25% of its original log interval, add exactly three probes while holding the
  other coordinate fixed: embedding-low
  `0.008/0.012699208415745596/0.02015873679831797`, embedding-high
  `0.20318733465192954/0.32253978877308753/0.512`, deep-low
  `0.006/0.009524406311809197/0.01511905259873848`, or deep-high
  `0.15239050098894713/0.24190484157981565/0.384`. If both coordinates are
  boundary-adjacent, add all six runs. Allow one boundary round and return for
  user review if an outer point still wins.
- Every aggregate candidate and the scheduler bridge use the declared
  one-cycle H15 horizon, train all 15 epochs without an early-stopping
  callback, restore the best validation epoch within that completed horizon,
  and never trigger adaptive horizon correction. A historical H15 artifact is
  selection-eligible only when its metadata proves
  `stopped_epoch == epochs_trained == lr_schedule_horizon_epochs == 15` and
  `lr_horizon_complete == true`; the dormant historical adaptive flag may be
  true. Horizon-free baseline and bridge runs retain their 80-epoch safety cap
  and patience-three stopping rule.
- Before any further selection, rerun exactly two incomplete H15 recipes under
  the corrected full-horizon method: 6 layers at embedding/deep LR
  `0.064/0.048`, and 8 layers at
  `0.0468526465053628/0.032703745675187676`. Their run names explicitly contain
  `full_horizon_rerun`. Only these two runs are in the pre-selection launcher
  and report requirement.
- Historical audit only: earlier adaptive work produced H13, H18, H27, and
  other H12/H17/H19/H23 corrections. All raw directories and names remain
  immutable evidence, but corrections are excluded from current selection and
  cannot supersede, block, or request another run.
- Select the aggregate by validation Recall@100, then same-epoch NDCG@100,
  then stable run name. Report the restored best checkpoint's full-user
  metrics.

## Run stages and compute

- Implement one dedicated aggregate configuration builder,
  manifest, verifier, report generator, and persistent-queue launcher.
- Focused tests cover exact composition, normalization placement, learned/time
  position coexistence, end-only CLS timestamps, popularity correction, and a
  constant embedding-LR trace under deep-only scheduling.
- Use one seed-42 run per exact configuration and reuse the existing
  native-500M single-run noise bands. Run ten fixed-member one-factor bridges
  against the frozen baseline. After aggregate selection freezes one of 4, 6,
  or 8 layers, run that selected depth as the eleventh bridge.
- Twenty-nine historical configurations have already been launched. The
  corrected pre-selection wave adds exactly two, for 31 launched artifacts
  after it completes. The maximum conditional envelope is 66: the 11 immutable
  historical corrections already present, plus at most 55 current-protocol
  configurations (six baseline, nine original aggregate, two full-H15 reruns,
  up to nine local, up to eighteen optimizer-boundary, and eleven bridge
  configurations). No adaptive horizon extension is part of this envelope.
  An unresolved horizon-free safety cap or extended optimizer boundary returns
  for approval. Any new treatment, dataset, metric rule, or tuning axis also
  does.
- Submit granular jobs to the existing persistent training queue without GPU
  exclusions. The service may fill all admitted GPUs in parallel.
- Existing sequence caches may be reused only after their metadata matches the
  exact sequence length, time fields, and query-token requirements.

## Interpretation and reporting

- A positive quality result requires aggregate Recall@100 more than 0.003
  above baseline and NDCG@100 no more than 0.001 below it. Otherwise report a
  trade-off, unresolved result, or regression under the existing rules.
- Do not reinterpret a surprising aggregate result until configuration,
  schedule traces, restored checkpoint, full-catalog evaluation, timestamp
  handling, and negative-proposal correction have been independently checked.
- Add `## Aggregated improvement` to the reader README. For every reader
  metric, show the baseline and aggregate values, aggregate gain in metric
  points and percent, sum of the eleven one-factor bridge gains, and interaction
  gap.
- Reuse the report's single-run operational band for the interaction gap.
  Label an interaction positive or negative only when its absolute value
  exceeds that band; otherwise label it unresolved.
- Preserve the component RQ tables. The aggregate section adds evidence and
  does not replace historical valid comparisons.

## Approval

- Material choices requiring approval: the exact eleven-member manifest, the
  RQ9 numerical-Recall conflict rule, the original-model baseline
  reconstruction, the per-depth aggregate joint-LR surface, reuse of the
  single-run bands, the 23-run initial budget, and the current 66-run maximum
  conditional envelope.
- Historical audit: the initial plan was approved with corrections on
  2026-08-25; the recovery wave and then H27 continuation were approved on
  2026-08-26 under the former adaptive method. Those approvals explain the
  preserved artifacts but are superseded for current selection by the
  methodology correction below.
- Methodology correction approved on 2026-08-26: completed H15 replaces
  adaptive horizon calibration for current selection; exactly two full-H15
  reruns precede selection; historical correction artifacts remain audit-only;
  the current actual/planned count is 29/31 and the revised maximum is 66.
