# G3 native-500M-only rerun plan

## Question and hypothesis

- Replace the rejected native-50M analysis with native Yambda-500M evidence for
  every G3 research question, hyperparameter selection, bridge, and aggregate.
- Test whether pretrained item content, catalog content, artist/album metadata,
  and frequency-adaptive mixing improve next-item retrieval over the two-layer
  version of the best G1 combination.
- Native-50M metrics, selected hyperparameters, checkpoints, and conclusions
  are historical audit evidence only. They cannot select or support a 500M
  claim.

## Comparison

- Freeze a newly tuned native-500M baseline with one tied learned item-ID table
  for history and catalog scoring. It is the authenticated best G1 combination
  with only its depth member removed: model width 64, two post-norm Transformer
  layers, two query heads, one KV head, SwiGLU width 192 with internal dropout
  0.1, input and final RMSNorm, full causal attention, and retained history 100.
- Preserve the other selected G1 members exactly: end-only CLS query, BOS,
  ALiBi, revision-7 full-width learned forward and reverse positions
  concatenated to the item representation, 32-bin additive time deltas, raw
  reverse timestamp RoPE, and dropout 0.1. The position fusion uses ReZero,
  bounded-tanh reverse correction with maximum scale 0.025, and non-advancing
  reverse-position initialization.
- Fix the selected G1 retrieval objective: 2,048 popularity-catalog global-Q
  negatives, dense random-negative scoring, Yi-2019 correction with alpha 0.01
  on both positive and negative logits, no false-negative mask, dot-product
  scoring, and no output normalization.
- Use the selected G1 one-cycle cosine schedule with 5% warmup on the deep
  optimizer group only; the embedding LR remains constant. G3 retunes both
  optimizer-group rates and the schedule horizon at batch 512.
- Do not use MuTransfer or μP. Do not reuse G1 metrics or its batch-1280
  optimizer coordinates as G3 evidence. Every G3 arm, bridge, and aggregate
  remains two layers; depth is not a G3 axis.
- Put that tied baseline first in every primary overall and slice table and use
  it for every primary percentage delta and baseline-facing promotion gate.
  Direct-predecessor marginal gates are additional and never replace that
  reader reference.
- Run one untied learned-ID secondary control for RQ1 because replacing the
  history table necessarily removes baseline weight tying. It receives the same
  nine-cell tuning budget as RQ1, but never replaces the tied G1-best baseline
  in a primary table or promotion rule. Do not run a parameter-count-matched,
  width-matched, or compute-matched control.
- Hold the G1-derived transformer, loss, negative sampling, sequence
  construction, and evaluation fixed. Tune only both learning rates, the
  one-cycle schedule horizon, and the parameter investigated by the RQ.

## Data and evaluation

- Sole evidence dataset: native Yambda-500M likes. Batch size is 512 and seed is
  42. No 50M run may enter tuning, selection, acceptance, aggregate arithmetic,
  or the reader report.
- Native dataset: 157,357 mapped items, 8,013,866 remapped like events,
  7,755,722 training interactions, and 81,020 training users. Use the final
  seven days for validation, train only mapped items, score the full mapped
  catalog, and do not exclude seen items.
- Frozen compact content table: width 128, IDs `1..157357`, SHA-256
  `647b62ccc6cb214181e6aa44768fe94abd69e840b7758824f8e521dfe040043c`.
- Every content branch consumes an L2-normalized content vector. Frozen rows
  are verified unit-normalized and normalized again at lookup as a fail-closed
  invariant. A trainable content copy is normalized after lookup on every
  forward pass, so optimization may change its direction but cannot expose
  vector norm as a score shortcut. Padding remains the zero vector. This
  normalization is before concatenation or projection; the learned model-width
  output is not normalized.
- Build versioned native-500M content and feature manifests around the existing
  source-consistent feature files. Fail closed on dataset-size mismatch, file
  identity, compact-ID order, width, duplicates, missing IDs, non-finite values,
  or train-split drift.
- Primary metric: Recall@100. Secondary metrics: Recall@10/50,
  NDCG@10/50/100, MRR@10/50/100, capped Recall@10/50/100, and catalog
  coverage@10/50/100.
- Epoch-level selection may use the fixed deterministic user-ID sample capped at
  20,000 users, but every restored candidate writes final metrics and ranking
  evidence over all evaluable native-500M users. Final reader and slice tables
  use only those full-user metrics.
- Keep the approved head/mid/tail item-frequency and low/mid/high-history
  slices. Slice metrics retain the unchanged global ranking and are mechanism
  evidence, not separate significance tests.
- Reuse only the native-500M relative noise dispersions from the ten accepted
  unchanged-control repeats, scaling them to the newly selected G3 baseline.
  Do not reuse their absolute means or launch another repeat batch. For a tail
  Recall@100 decision, scale the canonical Recall@100 relative dispersion to
  the G3 baseline's tail Recall@100 and label it an operational proxy rather
  than a slice-calibrated significance band.

## Hyperparameter selection

- One-cycle cosine scheduling with 5% warmup applies only to the deep optimizer
  group; the embedding LR stays constant. Every candidate completes its
  declared horizon, validates every epoch, and restores the best validation
  epoch within that horizon.
- Start the tied baseline search from the selected G1 aggregate coordinate:
  embedding LR `0.0468526465053628` and deep LR
  `0.032703745675187676`. That result used four layers, batch 1280, fixed-width
  MuTransfer, and horizon 15, so it is a tuning anchor rather than G3 evidence;
  G3 still measures its two-layer, batch-512, non-MuP baseline directly.
- At each horizon in `{10, 20, 40}`, run the exact G1 anchor plus the first two
  scrambled-Sobol joint LR pairs in a factor-two log-space box around it. This
  gives nine baseline cells. Select and freeze the best of all nine by the
  common metric order after resolving any allowed boundary extension. Do not
  fit an LR-versus-horizon law and do not run the former horizon-30 calibration.

  | horizon | order | embedding LR | deep LR |
  | ---: | ---: | ---: | ---: |
  | 10 | 0 | 0.046852646505362798 | 0.032703745675187676 |
  | 10 | 1 | 0.06375957559078467 | 0.033592533248942007 |
  | 10 | 2 | 0.03996662827497631 | 0.025850902250806791 |
  | 20 | 0 | 0.046852646505362798 | 0.032703745675187676 |
  | 20 | 1 | 0.093265638113829 | 0.052869980163324198 |
  | 20 | 2 | 0.037244925692432665 | 0.031983368359407911 |
  | 40 | 0 | 0.046852646505362798 | 0.032703745675187676 |
  | 40 | 1 | 0.09152246809261437 | 0.03128309208399048 |
  | 40 | 2 | 0.027995819103053991 | 0.034516357983428329 |
- Center each later family on the selected LR pair of its compatible direct
  predecessor. Its hard LR domain is a factor-sixteen range around that
  predecessor, while its initial probes stay inside a factor-two box. The hard
  domain exists only to bound the approved one-round boundary extensions; it is
  not an initial-search range. RQ5 is the exception: it fixes embedding LR to
  the selected RQ2 value and searches only deep LR and horizon.
- Anchor families exactly as follows: the untied control, RQ1, and RQ2 use the
  selected tied baseline; all five RQ3 targets use the selected RQ2 input; all
  three standalone RQ4 arms use the selected tied baseline; and both RQ5 gates
  use selected fixed RQ2 concatenation. An RQ3 bridge uses the
  aggregate-selected input, and an RQ4 bridge uses that input plus the selected
  bridged output, or the aggregate-selected input/output chain when no RQ3
  bridge is needed.
  A distinct aggregate family uses the most specific authenticated compatible
  predecessor in the order input, output, metadata. If that predecessor already
  contains every selected member, reuse it as the aggregate instead of
  launching a duplicate family.
- Family codes are fixed as: baseline `0`, untied control `10`, RQ1 `11`, RQ2
  `20`, RQ3 variants `31..35`, RQ4 artist/album/both `41..43`, RQ5
  global/frequency `51..52`, RQ3 bridge `61`, RQ4 bridge `62`, and aggregate
  `63`. For `n` Sobol rows in addition to the anchor, use
  `scipy.stats.qmc.Sobol(d=2, scramble=True,
  seed=300000 + 100*family_code + horizon)`, generate the smallest power-of-two
  block containing `n`, retain its first `n` rows, and log-map both coordinates
  to the local factor-two box. The exact predecessor anchor is manifest order
  zero; the Sobol rows follow it. Serialize every LR with Python
  `format(value, '.17g')` and parse it as float64.
- A local factor-two box uses
  `[max(family_low,anchor/2), min(family_high,2*anchor)]` independently for
  each LR.
- A family without capacity receives nine cells: at each horizon 10/20/40, the
  predecessor anchor plus the first two local Sobol pairs.
- A three-capacity family receives three H20 LR pairs per capacity: the
  predecessor anchor plus the first two local Sobol pairs. After selecting
  capacity, add exactly three cells: the selected H20 LR pair at H10 and H40,
  plus the third local H20 Sobol pair not used in the capacity stage. Manifest
  order is capacity low/center/high, then within capacity
  anchor/Sobol-1/Sobol-2, followed by H10/H40/H20. The total remains twelve.
- RQ5 reuses the selected RQ2 embedding LR in every run. Its global gate gets
  deep LR `{d/2, d, 2d}` at each horizon 10/20/40, where `d` is selected RQ2
  deep LR: nine cells. Its frequency gate gets that three-value deep-LR grid at
  H20 for each width 32/64/96, then the selected width and deep LR at H10 and
  H40: eleven cells. These are one-dimensional grids, not Sobol searches.
- Select by validation Recall@100, then same-epoch NDCG@100, then deterministic
  manifest order.
- Primary reader deltas always use the tied G1-best baseline. Aggregate
  eligibility additionally requires every dependent component to improve its
  direct predecessor: RQ3 versus variant 1, RQ4 versus the common tied
  baseline, and a learned gate versus its fixed/global gate predecessor.
  Use the same baseline-scaled absolute Recall@100 band for these marginal
  comparisons. RQ1, RQ4, and RQ5 may use their declared tail trade-off only
  when overall Recall@100 is non-inferior to both the tied baseline and their
  direct predecessor and the required tail improvement is present.
- A family winner is on an LR boundary when its selected coordinate is the
  minimum or maximum tested value for that optimizer group within the family.
  Such a winner adds three one-axis probes: multiply the boundary LR by
  `{1/2,1/4,1/8}` on the low side or `{2,4,8}` on the high side while holding
  the other LR, horizon, treatment, and capacity fixed. If both optimizer groups
  are boundary winners, add both three-run sets. A capacity boundary adds three
  LR cells at the winner's horizon: the selected pair plus the first two local
  factor-two Sobol pairs. RQ2 adds width 32 or 512, RQ4 adds width 128 on its
  high side (16 is the smallest permitted learned width), and RQ5 adds width 16
  or 128. If horizon 40 wins and its restored best epoch is 40, run horizon 60
  at the selected rates. A second unresolved LR/capacity boundary or a
  horizon-60 endpoint returns for approval.
- RQ5 never extends or retunes embedding LR. Its deep-LR boundary adds the same
  three directional probes, its frequency-width boundary adds the three-value
  deep-LR grid at the new width, and its horizon endpoint rule is unchanged.

## Treatments

### RQ1: What happens when pretrained embeddings replace item IDs?

- Secondary mechanism control: separate learned history and catalog item-ID
  tables with the complete G1-derived backbone. Tune it for nine cells. It
  isolates content from the unavoidable tying change but is never the primary
  reader or promotion reference.
- Frozen content-only history: normalized content width 128 projected to model
  width 64; learned item-ID catalog target.
- The projection is the existing bias-free linear 128-to-64 map with no output
  normalization. Its capacity is deliberately fixed because this rerun changes
  only the evidence dataset, not the approved RQ1 treatment.
- Apply the unchanged G1 learned-position concatenation and time features after
  constructing the width-64 content item representation; keep BOS and end-only
  CLS behavior identical to the baseline.
- Nine tuning opportunities.

### RQ2: Does concatenating content and item ID help?

- Concatenate learned item ID width 64 and normalized frozen content width 128,
  then use a DenseNet to return width 64.
- Tie the learned-ID history branch to the learned catalog target table, as in
  the G1-best baseline, so RQ2 does not confound content with weight untying.
  DenseNet is the existing two-layer encoder and receives no output
  normalization.
- DenseNet hidden widths: 64, 128, and 256.
- Apply the unchanged G1 learned-position concatenation and time features after
  the content/item DenseNet returns the width-64 item representation.
- Twelve tuning opportunities.

### RQ3: Which prediction embedding is best?

- Recreate the selected RQ2 history encoder with one independent learned
  history-ID table shared by all five RQ3 arms. This is required to keep the
  history input identical when the catalog target is changed. Compare exactly:
  1. learned item ID;
  2. frozen pretrained content;
  3. trainable content initialized from pretrained content;
  4. learned item ID plus frozen pretrained content;
  5. learned item ID plus trainable content initialized from pretrained content.
- Nine opportunities per target, 45 total. Variant 1 is the local RQ3 control;
  no tied RQ2 artifact is reused and trained weights never transfer.
- Save the existing norm, logit, gradient, representation-drift, and
  frequency-slice diagnostics needed to explain an unexpected ordering.
- Every learned-ID catalog table is separate from the RQ2 history table.
  Content-only targets use a bias-free linear 128-to-64 projection; concatenated
  targets use a bias-free linear 192-to-64 projection. All five use raw
  dot-product scores without output normalization. Frozen and trainable content
  rows are L2-normalized before either projection; trainable rows are normalized
  on every forward pass.
- A target is aggregate-eligible only when its marginal full-user Recall@100
  gain over variant 1 exceeds the band. Otherwise retain variant 1 regardless
  of the target's absolute gain over the G1-best baseline.

### RQ4: Do artist and album features help?

- Start independently from the common two-layer G1-best tied learned-ID
  baseline. Compare artist only, album only, and artist plus album. Do not add
  RQ1, RQ2, RQ3, or RQ5 treatments in this RQ.
- Reuse the selected common baseline as the exact no-metadata control; it adds
  no run. A metadata arm is aggregate-eligible only when it improves that
  baseline under the RQ4 decision rule.
- Use train-only compact vocabularies, unknown index zero, multi-value mean
  pooling, and concatenation. Preserve the baseline's learned-ID tying by using
  one shared metadata DenseNet encoder for both history and catalog scoring;
  project the concatenated representation back to width 64.
- Shared metadata widths: 16, 32, and 64. Twelve opportunities per family, 36
  total.

### RQ5: Does a frequency-adaptive content gate help?

- Start from the selected RQ2 input. Reuse fixed concatenation (`gate = 1`) as
  the mechanism control and compare a learned global scalar gate with the
  corrected FP32, initial-probability-0.9 frequency gate.
- Reuse the selected RQ2 embedding LR for fixed, global, and frequency-gated
  concatenation. Tune only deep LR, horizon, and the frequency-gate width.
- Frequency-gate hidden widths: 32, 64, and 96. Give the global family nine
  opportunities and the frequency family eleven, 20 total.
- The global gate is eligible only if it improves fixed concatenation. The
  frequency gate is eligible only if it satisfies its acceptance comparison
  against the G1-best baseline, fixed concatenation, and the global gate.
- The earlier BF16-saturated gate is raw audit evidence only.

## Run stages and budget

Before submitting evidence runs, the affected focused tests and artifact
verifiers must prove:

- the baseline contains exactly the ten fixed G1 aggregate members, remains two
  layers, and uses neither MuTransfer nor μP;
- the tied baseline and RQ2 share the intended table object and gradients, the
  RQ1 diagnostic is untied, standalone RQ4 shares one metadata encoder between
  history and catalog, and every RQ3 arm has identical history-input structure
  and initialization;
- adding frozen or optional modules does not advance RNG state for common
  learned parameters, frozen/trainable tables have the intended optimizer
  membership, and every expected component receives gradients;
- every frozen and trainable content lookup is L2-normalized before composition
  or projection, trainable normalization preserves gradients, padding remains
  zero, and model-width outputs remain unnormalized;
- direct item scoring and dense full-catalog scoring have value and gradient
  parity for every catalog encoder, while popularity sampling and positive and
  negative Yi-2019 correction match the G1 implementation;
- BOS, end-only CLS, ALiBi, learned-position concatenation, timestamp RoPE, time
  bins, causal masks, and content/metadata composition coexist with the exact
  intended token and timestamp semantics;
- native-500M dataset, split, remap, content, metadata, and train-frequency
  manifests fail closed on identity or ordering drift; and
- the embedding LR is constant, every RQ5 row reuses the selected RQ2 embedding
  LR exactly, the deep one-cycle trace completes its declared horizon, the
  validation metric and slice denominators are correct, and the restored
  checkpoint is the recorded best epoch.

Run the affected focused tests, obtain a blind pre-run review with no unresolved
correctness or scientific-integrity findings, then run `./test.sh` once before
the first evidence batch. Synthetic or short debugging checks have distinct
non-evidence names and cannot select treatments.

1. Tune and freeze the tied native-500M baseline: nine opportunities.
2. Run the untied diagnostic, RQ1, RQ2, and independent RQ4 concurrently after
   baseline calibration: 66 opportunities.
3. Run RQ3 and RQ5 concurrently after RQ2 selection: 65 opportunities.
4. Resolve compatible promoted treatments, then run at most nine RQ3 bridge,
   nine RQ4 bridge, and nine exact aggregate opportunities: at most 27.

- Each conditional bridge or aggregate family uses the same nine-cell
  no-capacity recipe and the same LR/horizon boundary rules as a component
  family.

- Initial RQ budget: `9 + 9 + 9 + 12 + 45 + 36 + 20 = 140`
  opportunities. The second `9` is the untied RQ1 mechanism control.
- Initial plus conditional bridge/aggregate budget is at most 167
  opportunities. Eight two-LR no-capacity families can add at most seven
  extensions each; four two-LR capacity families can add at most ten each;
  the RQ5 global and frequency families can add at most four and seven; and the
  three conditional bridge/aggregate families can add at most seven each.
  Therefore the complete approved worst-case envelope is
  `167 + 8*7 + 4*10 + 4 + 7 + 3*7 = 295` opportunities. A second unresolved
  boundary is outside this envelope and returns for approval. Exact compatible
  reuse reduces physical launches; it cannot be spent on extra search.
- If no treatment qualifies, the aggregate equals the baseline and no duplicate
  aggregate run is launched.
- All multi-run batches use the existing persistent shared queue, explicit data
  group `g3-native500m-likes`, one process per admitted GPU, and enough granular
  jobs to use all eight GPUs without a competing queue.

## Compatibility and aggregate arithmetic

- RQ1 content-only, RQ2 fixed concatenation, RQ5 global-gated concatenation,
  and RQ5 frequency-gated concatenation are mutually exclusive input choices.
  RQ5 candidates include RQ2 as a prerequisite and supersede it rather than
  contributing a second input gain.
- An RQ2-derived gate may qualify as one atomic input bundle even if fixed RQ2
  alone does not, but it must still satisfy its marginal gate comparison. Among
  eligible input choices, select a Recall@100 leader when its advantage exceeds
  the band. If candidates are unresolved on Recall@100, select higher tail
  Recall@100, then NDCG@100, then manifest order. If none qualifies, retain the
  G1-best tied input. This resolves RQ1 versus every RQ2/RQ5 branch.
- RQ3 is scientifically measured on the independent-input RQ2 chain. Only a
  target that improves variant 1 marginally may enter aggregation. Bridge that
  target onto the aggregate-selected input, re-evaluate its marginal gain there,
  and include it only if the bridge remains eligible.
- RQ4 is scientifically measured independently against the common tied
  baseline. Only metadata that qualifies in that standalone comparison may
  enter final aggregation. Do not combine it with RQ1/RQ2/RQ3/RQ5 during RQ4
  selection. At final closure, bridge it once onto the aggregate-selected
  input/output chain, re-evaluate its marginal gain there, and include it only
  if the bridge remains eligible.
- Run an input-plus-output bridge whenever the aggregate input differs from the
  RQ3 scientific chain. If standalone RQ4 qualifies and another aggregate
  treatment changes input or output, add metadata only at final closure through
  an input-plus-output-plus-metadata bridge. The final bridge is the aggregate
  when it contains every qualifying compatible component.
- Before bridge submission, freeze a resolved manifest listing every included
  treatment, dependency, predecessor artifact, omission, and exact marginal
  comparison. Any conflict not resolved above returns for approval.
- The reported standalone-plus-bridge gain counts the baseline-to-input term
  once and every later component only through its marginal bridge. The
  interaction gap is the trained aggregate gain minus that non-overlapping sum.

## Unexpected results

If a result contradicts the expectation, first verify dataset/remap identity,
initialization, masks, losses, optimizer membership, gradients, direct/dense
score parity, evaluation denominators, and schedule completion. If the
implementation is correct, test and explain the mechanism with the saved
diagnostics and a targeted controlled rerun or ablation. Prose alone is not
enough.

## Reporting

- Replace the active reader, compact, and tuning tables with native-500M-only
  RQ1-RQ5 evidence. Remove RQ6 because dataset size is no longer a research
  axis.
- Preserve every existing 50M ledger, evidence JSON, and raw run artifact.
  Remove the obsolete active 50M reader/compact/tuning tables after the 500M
  report is authenticated; do not present historical incorrect tables beside
  the corrected report.
- Use `RQ1: ...` through `RQ5: ...`, then `## Aggregated improvement`.
- Aggregate arithmetic always includes Recall@100 and NDCG@100. If any
  treatment is included through a tail trade-off, it also includes the exact
  tail Recall@100 metric used for that decision. For every included metric,
  report baseline, trained aggregate, gain, non-overlapping standalone-plus-
  bridge sum, interaction gap, and interaction classification.
- Classify an interaction as positive when its gap exceeds the metric's
  baseline-scaled absolute band, negative when it is below the negative band,
  and unresolved otherwise. Tail Recall@100 uses the operational proxy band
  defined above and is explicitly labeled as such.
- Synchronize `protocol/plan.md`, `experiments/ideas_understanding.md`, the work
  tracker, protocol constants, report tests, and generated reports so no active
  specification still selects on 50M or defines the removed size RQ.
- Omit runtime, throughput, GPU-memory, latency, parameter-count, and efficiency
  tables or columns. Automatically emitted telemetry remains raw validity
  evidence only.

## Acceptance criteria

### RQ1

Compare content-only input with the two-layer G1-best tied learned item IDs.
Report overall and head/mid/tail results; a slice-only win requires aggregate
non-inferiority to that baseline.

### RQ2

Concatenation must beat the two-layer G1-best tied learned item-ID baseline to
establish content complementarity.

### RQ3

Compare every output target with the learned item-output table. Use paired
contrasts to separate target type, pretrained initialization, and freezing. I
think variant 4 will be the best, but I may be wrong. If results differ from
this expectation, explain why with experimental proof such as plots or gradient
norms, not only words. Variant 4 should be better than variants 1 and 2.
Variant 3 should not be much worse than variants 1 and 2 and will most probably
be better.

### RQ4

Artist and album features should improve tail metrics and should not make
overall Recall@100 worse than the two-layer G1-best tied learned-ID baseline.

### RQ5

The frequency-adaptive gate should improve tail Recall@100 and should not make
overall Recall@100 worse than the two-layer G1-best tied learned-ID baseline,
the fixed concatenation, or the learned global gate.

### Aggregated improvement

Run the best compatible combination of all qualifying changes. Report its
Recall@100 and NDCG@100 against the two-layer G1-best tied baseline, the
non-overlapping standalone-plus-bridge gain, and the interaction gap.

## Approval

- Dataset choice validated by the user on 2026-08-30: native Yambda-500M only;
  native-50M is too unstable for analysis.
- G1's selected aggregate learning rates are the starting point for the G3
  baseline search; the user rejected a fresh LR-versus-horizon fit on
  2026-08-31.
- The user clarified on 2026-08-31 that every content representation is
  normalized, RQ4 starts independently from the common baseline, and RQ5 fixes
  embedding LR to the selected RQ2 value.
- Exact two-layer G1-best baseline, fixed linear RQ1 projection, normalized
  content, independent RQ4, fixed RQ5 embedding LR, tuning surface, 140 initial
  opportunities, 167 including conditional bridges/aggregate, 295 worst-case
  bounded envelope, RQ6 removal, and report replacement: approved by the user
  on 2026-08-31.
