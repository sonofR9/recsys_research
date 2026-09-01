# G3 pretrained item embeddings: approved run plan

## Question and hypothesis

- Research questions and status:
  - RQ1: replace learned history item IDs with pretrained content embeddings;
  - RQ2: concatenate learned item IDs and pretrained content embeddings;
  - RQ3: compare learned-ID, pretrained, and concatenated retrieval targets;
  - RQ4: add artist and album features;
  - RQ5: condition the item-ID/content mixture on training frequency;
  - RQ6: compare the selected treatment's uplift on native
    Yambda-50M and native Yambda-500M.
- Current understanding: learned IDs memorize collaborative identity, while the
  provided content/audio vectors can generalize across related and infrequent
  items. Input and output representations answer different questions and must
  be independently configurable. Concatenation requires a DenseNet encoder.
- Falsifiable hypotheses:
  - content alone helps tail items but may lose collaborative identity;
  - learned ID plus content is complementary and beats the original learned-ID
    baseline;
  - a learned-ID plus frozen-content output target is the best RQ3 target;
  - artist and album features help tail retrieval without hurting aggregate
    Recall@100;
  - a frequency-conditioned mixture helps tail items without reducing
    aggregate quality;
  - content uplift may shrink at 500M as collaborative ID estimates improve.
- Why the result matters: G3 determines whether later candidate generators
  should use content and metadata in their history encoder, catalog encoder, or
  both, and whether the conclusion transfers across dataset sizes.

## Comparison

### Shared control and invariants

- Original baseline: the selected conventional native-50M G4 control, which is
  the batch-512 G1-derived SASRec model with one tied learned item-ID table for
  input and output. Reuse its completed metrics and ranking evidence rather
  than retraining it.
- Every RQ1-RQ5 primary reader table, promotion decision, percentage delta, and
  item- or user-slice table uses this tied original learned-ID baseline as its
  first row and reference. RQ6 uses the size-matched tied original baseline at
  each size.
- Local representation diagnostic: the same model and objective with separate
  learned 64-dimensional history-input and catalog-output item-ID tables. This
  is tuned once and may remain in explicitly requested secondary mechanism
  contrasts. It never replaces the tied original baseline in a primary table
  or promotion rule.
- The tied original baseline remains the aggregate comparator. The untied
  control isolates G3 representation changes from weight tying, but untying is
  not promoted as an independent treatment. For aggregate selection, every G3
  representation that requires separate input/output encoders is an atomic
  bundle with untying and must have a matched direct comparison against the tied
  baseline. Its baseline-to-bundle gain includes the tying change exactly once.
- Fixed backbone and objective: model width 64, two transformer layers, two
  query heads, one key/value head, SwiGLU width 192, pre-layer normalization,
  RMS input normalization, forward learned positions, 128 stored history
  events, attention window 50, 16 timestamp-delta bins, 512 dense random
  catalog negatives, and the existing shifted next-liked-item loss.
- Every catalog representation has output width 64. Scoring is an unnormalized
  dot product for every arm; no arm receives an otherwise absent final
  normalization or temperature.
- Frozen source content vectors remain unit-normalized. Learned projections and
  DenseNet encoders use the deep optimizer group. Learned item, artist, album,
  and trainable content tables use the embedding optimizer group. Frozen tables
  are excluded from the optimizer.
- Input and catalog tables are separate in all local G3 arms, including when
  both are learned item IDs. RQ3 therefore changes only the catalog target.

### RQ1: pretrained content replaces the history item ID

- Primary control: the tied original learned item-ID baseline.
- Secondary diagnostic: the untied learned item-ID input/output control, used
  only to isolate the unavoidable tying change inside the treatment bundle.
- Treatment: frozen 128-dimensional content lookup followed by a learned
  128-to-64 projection as the history representation; the learned item-ID
  output is unchanged.
- Report aggregate and deterministic train-frequency head/mid/tail slices.

### RQ2: pretrained content is concatenated with the history item ID

- Treatment: concatenate learned item-ID width 64 and frozen content width 128,
  then use DenseNet to return width 64 before the transformer.
- Primary control: the tied original learned item-ID baseline. The untied
  learned-ID control may remain as a secondary mechanism diagnostic.
- DenseNet hidden-width candidates for the content treatment are 64, 128, and
  256. No parameter-matched ID-only DenseNet is required or promoted by the
  active protocol.
- The learned item-ID catalog output remains fixed.

### RQ3: catalog prediction representation

- Fix the selected RQ2 input definition and reinitialize it independently for
  each run. Do not transfer selected weights.
- Compare five width-64 catalog encoders:
  1. a randomly initialized learned item-ID table;
  2. the frozen 128-dimensional content table plus a learned 128-to-64
     projection;
  3. a trainable exact copy initialized from the content table plus the same
     projection;
  4. concatenated learned item ID and frozen content plus a learned
     192-to-64 projection;
  5. concatenated learned item ID and a trainable content-initialized copy plus
     the same projection.
- Variant 1 is the selected RQ2 treatment and is reused rather than duplicated.
- The primary reader and promotion comparison for every variant is still the
  tied original learned-ID baseline. The explicitly requested variant-to-
  variant contrasts remain secondary scientific diagnostics and acceptance
  evidence.
- Save per-epoch query/catalog norms, positive and negative logit distributions,
  gradient norms by component, content-initialization drift/cosine similarity,
  and the same statistics by item-frequency slice. These diagnostics are
  mandatory mechanism evidence if the ordering contradicts the expectation.

### RQ4: artist and album features

- Start from the selected RQ2 concatenated-input and RQ3 catalog definitions.
  This preserves the requested scientific chain even if a different input later
  wins aggregate selection.
- Compare artist only, album only, and artist plus album.
- Build train-only compact feature vocabularies with unknown index zero. An item
  may have several artists or albums; mean-pool their learned embeddings so
  items with more metadata values do not receive a larger representation only
  because of cardinality.
- Concatenate metadata to the existing event and catalog representations and
  use separate DenseNet encoders to return width 64.
- Tune one shared metadata width from 16, 32, and 64 for each family.
- The primary comparison and promotion reference is the tied original
  learned-ID baseline. No parameter-matched extra item-ID control is required
  by the active protocol.

### RQ5: frequency-adaptive item-ID/content mixture

- Start from the selected RQ2 input. Apply a gate to the frozen content branch
  before the exact same DenseNet:
  `DenseNet(concat(item_id, gate * content))`.
- Compare:
  - the tied original learned-ID baseline as the first primary reader row and
    promotion reference;
  - the selected RQ2 input, which is exactly the fixed `gate = 1` control;
  - one learned global scalar gate;
  - a per-item gate conditioned only on standardized `log1p(train_count)`.
- The frequency gate is a sigmoid MLP with hidden width 32, 64, or 96. Counts and
  standardization statistics use training data only.
- This RQ changes the input only. If selected, it supersedes the fixed RQ2
  input and requires bridges to selected RQ3/RQ4 components before aggregation.

## Data and evaluation

### Main G3 experiment

- Proposed and requested-for-validation dataset: native Yambda-50M for RQ1-RQ5,
  tuning, bridges, and the main aggregate.
- Protocol: likes, core threshold five, final seven days held out by timestamp,
  train only mapped items, score the full mapped catalog, do not exclude seen
  items, and evaluate all 3,414 eligible users.
- No position-based sampling is permitted. Any debugging sample is selected by
  user ID and has a distinct non-evidence artifact name.
- Content source: `generated/yambda_data/embeddings.parquet`, column
  `normalized_embed`, exactly 128 finite values per item. Before launch, freeze
  its SHA-256, compaction implementation hash, dataset-specific compact remap
  hash, and compact output hash in a manifest.
- Current native-50M compact content table contains 33,148 mapped catalog items
  and has complete compact-ID coverage. Its current SHA-256 is
  `aa14c76ea36d5a9b8730bd856ba0f0e90bc7230a7179e04650b22d5a9572dd64`.
  Loading fails closed on hash, row count, compact-ID order, width, duplicates,
  missing IDs, or non-finite values.
- Artist and album mappings are compacted from training data only and their
  hashes, vocabulary sizes, unknown rates, and per-item cardinalities are saved.
- Primary metric: Recall@100.
- Secondary ranking metrics: Recall@10/50, NDCG@10/50/100, MRR@10/50/100, and
  capped Recall@10/50/100. Report catalog coverage@10/50/100 as a trade-off,
  not as a substitute for Recall@100.
- Diagnostic slices:
  - head/mid/tail are equal catalog-count terciles ordered deterministically by
    `(training interaction count, compact item ID)`;
  - low/mid/high-history are equal user-count terciles ordered by
    `(training history length, user ID)`.
- Item-frequency slice Recall@K keeps each user's unchanged global top-K
  ranking. For a slice, restrict only that user's relevant targets to the slice,
  compute hits divided by relevant targets in the slice, exclude users with no
  targets in that slice, and macro-average the remaining users. A user with
  targets in several slices contributes independently to each applicable slice.
  Save slice user and target denominators.
- Retain automatically emitted epoch, schedule, runtime, and resource telemetry
  for operational sanity; do not run dedicated performance benchmarks. Omit
  performance and efficiency tables or columns from reader,
  compact, and tuning reports unless the user explicitly asks about performance
  or training budget, or a material anomaly is needed to establish validity.
- Main-treatment decisions use the repository's approved native-50M relative
  resolution calibration, scaled by the current control's metric. Slice
  results are mechanism evidence and are not promoted as independent wins.

### Separate dataset-size companion

- Dataset size is the explicit research axis, so this companion uses native
  Yambda-50M and native Yambda-500M. Neither dataset is repeated or resampled to
  match the other's examples, tokens, epochs, or optimizer steps.
- Select the complete treatment definition on 50M, then freeze input, output,
  metadata, gate, and capacity choices before any 500M treatment run.
- At 500M, independently tune the exact learned-ID baseline and the frozen G3
  treatment under the native 500M protocol. Do not reselect a different G3
  treatment at 500M.
- Rebuild native content and metadata remaps at each size and bind them by hash.
- The current native-500M compact content table SHA-256 is
  `647b62ccc6cb214181e6aa44768fe94abd69e840b7758824f8e521dfe040043c`.
- Use size-matched empirical resolution only. A scale effect is claimed only
  when the absolute unrounded difference between the 500M and 50M treatment
  gains exceeds the sum of the two size-specific absolute bands. Report its
  sign so a smaller or larger uplift at 500M is detectable.

## Hyperparameter selection

- Batch size is fixed at 512 and is not tuned.
- No MuTransfer or μP rule is used.
- Selection objective: validation Recall@100, then validation NDCG@100, then
  deterministic manifest order. Performance is not a tie-breaker.
- Every run validates every epoch, completes its declared linear-schedule
  horizon, restores the best validation epoch, and records both horizon and
  restored epoch.
- Initial rate bounds are centered on the reusable native-50M control:
  - embedding LR: `[0.0368614745, 0.5897835914]`;
  - deep LR: `[0.0081084848, 0.1297357573]`;
  - candidate schedule horizons: 15, 25, and 40 epochs.
- Because two learning rates and the horizon move together, use a deterministic
  scrambled Sobol/random-search design, not a Cartesian grid. The exact control
  coordinate at embedding LR `0.1474458978`, deep LR `0.0324339393`, horizon 25
  is included where compatible.
- First calibrate the untied control with three joint LR probes at each of the
  three horizons, fit the two optimizer-group horizon relations, and validate
  the fit on one held-out control coordinate. Reject the transfer if the held-out
  prediction does not select an LR inside the same resolution-equivalent
  region; in that case every family uses direct nine-coordinate tuning.
- A family without a capacity axis receives nine direct coordinates: three
  joint LR probes at each horizon.
- A family with three capacity values receives twelve coordinates: three joint
  LR probes per capacity at the transferred/selected horizon, followed by one
  horizon probe at each of 15, 25, and 40 for the selected capacity. If horizon
  transfer is rejected, use a balanced 12-coordinate Sobol design over both
  LRs, horizon, and capacity instead.
- Give directly competing families in the same RQ identical selection
  opportunities. Reused compatible cells count toward that budget and are not
  rerun; add only the missing coordinates needed to complete the shared budget.
- Boundary extension:
  - if either selected LR is in the outer 10% of its allowed interval, add three
    joint outward probes in the selected direction while holding the selected
    treatment/capacity fixed;
  - if the outward winner is again on the new boundary, return for approval;
  - if horizon 40 is selected and its restored best epoch is 40, complete a
    horizon-60 run at the selected rates; return for approval if its best epoch
    is again the endpoint;
  - if a method-specific capacity boundary wins, extend once with three LR
    probes at the next capacity in that direction.

## Run stages and compute

### Correctness gates before evidence runs

- Exact frozen-table preservation before and after global initialization.
- Dense content lookup equality against compact-table rows.
- Frozen versus trainable content gradient and optimizer-membership checks.
- Separate input/output table identity and gradient-isolation checks.
- Direct per-item scoring versus dense full-catalog scoring value and gradient
  parity for every catalog encoder.
- Dense random-negative sampling parity after replacing the current
  `nn.Embedding`-specific fast path with the generic catalog encoder path.
- RNG-isolation check proving that adding a frozen or optional module does not
  change common learned-parameter initialization.
- Artist/album unknown, multi-value mean-pooling, and train-only-vocabulary
  checks.
- Tiny end-to-end training/evaluation checks for every representation family.
- Focused metric regression checks for aggregate and slice denominators.

### Initial evidence budget

| stage | families | runs |
| --- | ---: | ---: |
| untied learned-ID control plus held-out horizon-transfer check | 1 | 10 |
| RQ1 content-only input | 1 | 9 |
| RQ2 content concat | 1 | 12 |
| RQ3 output variants 1-5; compatible RQ2 cells are reused | 5 | 45 |
| RQ4 artist, album, and both | 3 | 36 |
| RQ5 global and frequency-conditioned gates | 2 | 24 |
| native-500M learned-ID baseline and frozen selected G3 treatment | 2 | 18 |
| **selection opportunities** | **15** | **154** |

- Completed parameter-matched RQ2 DenseNet and RQ4 extra-ID artifacts are
  preserved unchanged as raw audit evidence. They are historical unsolicited
  controls, excluded from the active protocol, budgets, selection, promotion,
  and reader/tuning/compact tables; they are never deleted.
- Conditional bridges and aggregate:
  - selected RQ3 output over the aggregate-selected input, when that input is
    not the RQ2 input used by RQ3: up to nine runs;
  - selected RQ4 metadata over the aggregate-selected input/output bridge when
    that exact chain was not already trained by RQ4: up to nine runs;
  - if the last bridge is not already the exact complete aggregate, tune the
    exact aggregate for up to nine runs;
  - conditional total: at most 27 runs.
- Compatible reused cells reduce physical launches; the corrected active plan
  covers at most 154 initial physical runs so reuse cannot silently be spent on
  extra search.
- Initial plus conditional budget: at most 181 physical runs before boundary
  extensions.
- Each family may receive at most three LR-boundary runs, one horizon-endpoint
  run, and three capacity-boundary runs when applicable. A second unresolved
  boundary requires renewed approval rather than silently expanding the sweep.
- All evidence batches use the persistent shared training queue, one process per
  GPU, no default GPU exclusions, light utilization monitoring, and enough
  granular jobs to use all available GPUs. Sequence/content/metadata caches are
  built once per native dataset size and shared read-only by runs.

## Interpretation and reporting

- Reader and compact questions are ordered and named `RQ1: ...` through
  `RQ6: ...`. Every RQ1-RQ5 primary table starts with the tied original
  learned-ID baseline and computes percentages from it; RQ6 uses its
  size-matched original baseline. Explicitly requested local or predecessor
  contrasts may remain secondary diagnostics but never replace that reference.
- Runtime telemetry remains available for sanity checking but is omitted from
  reader, compact, and tuning tables unless the user explicitly asks about
  performance or training budget, or a material anomaly must explain validity.
- The active tuning ledger includes every usable completed coordinate belonging
  to the corrected active protocol. Failed, interrupted, boundary-rejected,
  incompatible, and historical unsolicited-control artifacts remain in raw
  audit storage and never appear in reader or tuning tables.
- Reuse exact completed control or cumulative-treatment artifacts when their
  data, architecture, objective, representation, tuning, and evaluation
  manifests match. Do not rerun solely to reorganize the report.
- If a result contradicts an expectation, first test data/remap identity,
  initialization, masks, losses, gradients, direct/dense score parity,
  evaluation denominators, and schedule completion. If implementation is
  correct, test the proposed mechanism with saved learning curves, frequency
  slices, norms, gradients, representation drift, and a targeted controlled
  rerun or ablation. Prose alone does not close the question.
- Promotion rule: a treatment enters the future baseline only when Recall@100
  improves beyond the applicable band, or the predeclared tail trade-off below
  selects it while aggregate Recall@100 is within its band. An unresolved or
  merely non-inferior treatment without the required tail improvement is
  omitted.
- Tail trade-off:
  - RQ1 may select content-only when aggregate Recall@100 is within the original
    baseline band and measured tail Recall@100 is higher than the original
    baseline;
  - RQ4 may select metadata when aggregate Recall@100 is within the original
    baseline band and measured tail Recall@100 is higher than the original
    baseline;
  - RQ5 may select the frequency gate when aggregate Recall@100 is within the
    original baseline band and measured tail Recall@100 is higher than the
    original baseline. The fixed-concatenation and global-gate comparisons
    remain required secondary mechanism checks.
  Slice deltas remain descriptive because no slice-specific repeat calibration
  exists; the report must say so rather than call them significant.

### Compatibility and aggregate arithmetic

- RQ1 content-only and RQ2 content-plus-ID are mutually exclusive input
  representations. Only a treatment satisfying its RQ-specific acceptance and
  promotion rules is eligible. If both qualify, select the larger Recall@100
  improvement beyond the band; if their difference is unresolved, use
  deterministic manifest order. If neither qualifies, retain learned
  item IDs.
- RQ5 supersedes RQ2 when it qualifies; it is not added as a second independent
  gain. If RQ2 alone does not qualify but the RQ2-plus-gate bundle does, treat
  them as one atomic input bundle and require its matched baseline comparison.
- RQ3 is scientifically measured on the concatenated RQ2 input regardless of
  which input wins aggregate selection. A qualifying catalog-output treatment
  is bridged onto the aggregate-selected input before inclusion. Count the
  baseline-to-input gain once, then count the output bridge's marginal gain over
  that input.
- RQ4 is scientifically measured on the selected RQ2-plus-RQ3 chain. A
  qualifying metadata treatment is bridged onto the aggregate-selected
  input/output chain before inclusion. Count only its marginal bridge gain over
  those prerequisites.
- If an RQ1 input or RQ5 gate wins after RQ3/RQ4 were selected, run the required
  input-plus-output and input-plus-output-plus-metadata bridges. The final bridge
  is the aggregate when it exactly contains every qualifying compatible
  component.
- The first baseline-to-selected-representation term is always measured against
  the tied original baseline and therefore includes untying once. All later
  output, metadata, and gate terms are marginal bridge gains over that atomic
  prerequisite. The untied local control is never inserted as an uncounted
  arithmetic baseline.
- Before launching the aggregate, write a resolved manifest listing all
  included treatments, dependencies, exact predecessor artifacts, and every
  omission. Any conflict not resolved by the rules above returns for approval.
- Aggregate tuning uses nine Sobol coordinates over embedding LR, deep LR, and
  horizon, with the same boundary rules. Batch remains 512.
- For both 50M and 500M, compare a size-matched original learned-ID baseline
  with tied input/output weights against a jointly trained best compatible G3
  combination. The 500M frozen treatment is that size's aggregate candidate and
  includes any required untying inside the atomic G3 bundle.
- `## Aggregated improvement` reports baseline and aggregate Recall@100 and
  NDCG@100, absolute and percentage gain, the sum of non-overlapping standalone
  plus bridge gains, and the interaction gap from unrounded metrics. It also
  reports any other metric used to include a component. Interaction is called
  positive or negative only beyond the size-matched empirical band; otherwise
  it is unresolved.
- If no treatment qualifies, the aggregate candidate equals the original
  baseline and no duplicate aggregate run is launched.

## Acceptance criteria

### RQ1

Compare content-only input with the original tied learned item IDs. Report
overall and head/mid/tail results; a slice-only win requires aggregate
non-inferiority to that original baseline.

### RQ2

Concatenation must beat the original tied learned item-ID baseline to establish
content complementarity.

### RQ3

Compare every output target with the learned item-output table. Use paired
contrasts to separate target type, pretrained initialization, and freezing. I
think the expected result is that variant 4 will be the best, but I may be very
wrong. You must explain why the results are expected if they differ from my
expectations. Not just with words: include experimental proof such as plots or
gradient norms. Variant 4 should be better than variants 1 and 2. Variant 3
should not be much worse than variants 1 and 2, and will most probably be
better.

### RQ4

Artist and album features should improve tail metrics and should not make
overall Recall@100 worse than the original tied learned-ID baseline.

### RQ5

The frequency-adaptive gate should improve tail Recall@100 and should not make
overall Recall@100 worse than the original tied learned-ID baseline, the fixed
concatenation, or the learned global gate.

### RQ6

Compare the selected treatment's Recall@100 gain over item IDs at both sizes.
Claim a scale effect only when the two gains differ beyond their combined
resolution.

## Approval

- Material choices requiring approval:
  - native Yambda-50M for all main G3 questions;
  - native 50M and 500M only in the separate dataset-size companion;
  - tied original baseline plus an untied local representation control;
  - the exact five RQ3 output variants and no output normalization;
  - the proposed RQ4 metadata design and acceptance wording;
  - inclusion of RQ5 frequency-adaptive gating;
  - Sobol/random-search tuning, batch 512, and the corrected 181-run active
    initial-plus-conditional budget before bounded extensions;
  - the dependency-aware aggregate and dataset-size aggregate rules.
- User approval: approved on 2026-08-29 (`sounds good`).
