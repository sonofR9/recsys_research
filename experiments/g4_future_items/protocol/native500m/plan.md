# G4 native-500M rerun plan: future-item supervision

## Question and hypothesis

- RQ1: does a 24-hour future window improve Recall@100 over next-liked-item
  training?
- RQ2: is uniform sampling among the next ten liked events non-inferior to the
  baseline and competitive with the 24-hour rule?
- RQ3: can a learned behavior-similarity selector identify better future
  periods than deterministic selectors and improve downstream Recall@100?
- Every native-500M arm uses the two-layer form of G1's
  validation-selected aggregate. It transfers all ten non-depth members of
  that recipe, including SwiGLU with intermediate width 192, while fixing
  model width 64, item-embedding width 64, two attention heads, and two
  transformer layers. No scaling-only depth, width, or per-layer-embedding
  increase is allowed. Relative to that fresh baseline, each treatment changes
  training-positive construction only; architecture, inference, and evaluation
  remain fixed.

The completed native-50M runs are immutable historical evidence. They do not
select a native-500M configuration, supply a metric, or satisfy any native-500M
gate.

## Comparison

- Frozen original baseline: a freshly tuned native-500M next-liked-item G4
  control at batch 512 and seed 42. It is G1's selected aggregate recipe with
  the depth member removed: two layers; model/item width 64; SwiGLU width 192;
  deep-only one-cycle cosine with 5% warmup; ALiBi plus concatenated learned
  forward/reverse positions; post-LayerNorm; input/final RMSNorm; end-only CLS;
  binned time plus reverse timestamp RoPE; popularity global-q negatives; GQA;
  and BOS. It retains G1's max sequence length 100 and full causal attention.
- The exact fixed model identity is `MuTransferGenerationExperiment` with μP
  base/delta widths 16/32, target width 64, and the single tied 64-dimensional
  input/output item table. It has two query heads and one KV head; input,
  attention, and FFN dropout 0.1; gated-FFN dropout enabled; ReZero position
  fusion; bounded-tanh reverse-position correction with maximum scale 0.025
  and RNG-nonadvancing initialization; additive 32-bin time encoding;
  `random_offline_logq` sampling with 2,048 negatives, Yi-2019 correction,
  positive-logit correction, dense random-negative scores, and random fraction
  0.5. These fields are fixed in every arm.
- G1 selected the complete four-layer aggregate at batch 1280. Its metrics,
  selected rates, and four-layer result are not reused as G4 evidence. G4
  removes only the depth member from aggregate membership, fixes batch 512,
  and retunes the two-layer control on native 500M before freezing it.
- RQ1: sample one strictly later liked-event occurrence uniformly from the next
  24 hours, falling back to the next liked item when the window is empty.
- RQ2: sample one occurrence uniformly from the next ten liked events.
- RQ3 deterministic families: UTC hour-of-week, content-centroid cosine, and
  item/artist/album weighted Jaccard.
- RQ3 learned family: the existing fixed feature vector and
  `HistGradientBoostingClassifier` definition, followed by hard top-k and
  proportional period sampling downstream.
- The exact causal bounds, common selector universe, chronological 70/15/15
  selector split, relevance label, five user-id hash folds, masks, fallback,
  and deterministic RNG definitions remain those in the approved native-50M
  plan. They are implementation semantics only and are re-frozen against fresh
  native-500M data and manifests.
- Every primary table starts with the fresh original native-500M baseline and
  computes percentage changes from it. RQ2 also includes RQ1 as a secondary
  comparison; RQ3 also includes the best fixed-window treatment.
- G4 treatments are mutually exclusive positive-selection rules. The winning
  treatment cannot be composed with another G4 treatment.

## Data and evaluation

- Single dataset size: native Yambda-500M for every tuning run, selector,
  feasibility measurement, final result, and aggregate decision.
- No final-run user sampling. Diagnostics may sample only by hashed user id and
  cannot select a method or support a quality claim.
- Recommender histories and targets use likes only. RQ3 may use listens inside
  the training interval only for its independent selector label.
- Retain core items with at least five likes, drop unmapped items, hold out the
  final seven days by timestamp, train only mapped items, score the full mapped
  catalog, and do not exclude seen items.
- Primary metric: full-catalog Recall@100. Secondary metrics are Recall, NDCG,
  MRR, capped Recall, and coverage at 10, 50, and 100. Report the existing
  target-distance, event-rank, and user-activity slices with fresh denominators.
- Scale only the current G4 native-500M baseline by the reviewed shared
  native-500M relative dispersions:
  - Recall: 3.152286%, 2.115923%, 1.684781% at 10, 50, 100;
  - NDCG: 2.679530%, 2.272053%, 1.966145%;
  - MRR: 2.392931%, 2.156615%, 2.084706%;
  - capped Recall: 2.954665%, 2.106629%, 1.683308%;
  - coverage: 16.765104%, 15.102334%, 13.429391%.
- The dispersion source is
  `experiments/g1_sasrec_item_ids_likes/scratchpad/baseline_spread_500m.json`.
  Never reuse its absolute means or bands and never import native-50M
  dispersions.

## Hyperparameter selection

- Batch size is fixed at 512 and is not tuned.
- Embedding LR is fixed in every arm to G1's selected aggregate value
  `0.0468526465053628`; it is not tuned. The one-cycle cosine horizon is fixed
  to G1's selected 15 epochs, with 5% warmup and deep-only scheduling. Every
  run completes that horizon, validates each epoch, and restores the best
  validation epoch.
- Deep LR is the only tuned recommender hyperparameter. Every recommender study
  starts from G1's selected deep LR `x = 0.032703745675187676` and runs the
  three-point logarithmic grid `{x/2, x, 2x}`:
  `{0.016351872837593838, 0.032703745675187676, 0.06540749135037535}`.
- Select cumulatively by validation Recall@100, then validation loss, then
  deterministic manifest order. If `x/2` wins, add `{x/8, x/4}`; if `2x` wins,
  add `{4x, 8x}`. If the newly added outward point `x/8` or `8x` wins, add one
  final pair `{x/32, x/16}` or `{16x, 32x}`. A final outward winner `x/32` or
  `32x` requires a newly approved expansion. Thus each recommender study has
  three base runs and at most four extensions.
- Every RQ3 downstream arm fixes `objective_period_count = 1`, matching the
  existing native-50M G4 runtime projection; capacity is not retuned.
- Apply that one-dimensional rule independently to every recommender study
  before freezing its winner. No selected deep LR transfers between studies.
  Final all-user metrics and performance telemetry never select a run.
- The selector stage retains 12 trials for each of its four families, 48 total.
  Only the learned family has a continuous selector learning rate; only that
  family receives its existing four-trial learning-rate boundary round,
  repeated at most once for eight selector extensions maximum.

## Run stages and compute

1. Freeze a new `native500m` control manifest and source/data identity. Existing
   native-50M manifests, ledgers, artifacts, and evidence remain untouched.
2. Run three selection-eligible control cells, boundary-resolve the winner with
   at most four cells, then freeze the selected native-500M control and a new
   treatment-semantics manifest.
3. Run three RQ1 and three RQ2 cells, then any triggered bounded extensions.
4. Before RQ3 selector search, run the two deterministic hashed-user,
   non-selection feasibility preflights defined below on the memory-bounded
   native-500M materializer. They may stop RQ3 but cannot pass the native gate
   or support selector/recommender claims.
5. If the feasibility preflight is below the declared limits, run 48 fresh
   selector trials, any triggered learned-selector boundary round, and the
   untouched-test paired user-bootstrap gate.
6. If learned selector quality resolves above the strongest deterministic
   selector, run the complete five-fold native-500M materialization gate.
7. If both gates pass, run three cells for each of deterministic hard, learned
   hard, and learned proportional downstream objectives, then any triggered
   bounded extensions.

Expected research work is at most 66 runs: 3 control, 6 fixed-window, 48
selector, and 9 RQ3 downstream runs. The bounded maximum is 98: up to four
extensions for each of six recommender studies and eight learned-selector
extensions. A pre-selector feasibility stop leaves 9 base recommender runs and
9–21 including their possible extensions. A failed selector or materialization
gate omits 9 base downstream runs and 9–21 including possible extensions.

All multi-run batches use the existing persistent `utils/training_queue`
service with no manual GPU exclusions and no competing queue.

### Native-500M RQ3 feasibility and materialization gate

- The preflight selectors are fixed independently of every quality result:
  1-hour periods, 7-day lookahead, minimum one liked event, content cosine for
  the deterministic arm, and 31 leaves, classifier LR 0.05, and L2 `1e-5` for
  the learned arm. They run the complete five-fold deterministic/learned
  artifact path on two nested samples and cannot enter any selection table.
- Assign a user by the first eight SHA-256 bytes of canonical compact JSON
  `["g4-feasibility-v1", uid, 42]`, interpreted unsigned big-endian. The 5%
  sample has value modulo 20 equal to zero; the 10% sample has value modulo 10
  equal to zero. Each run starts from an empty directory, uses 16 workers, and
  records wall seconds, 10-Hz descendant-tree peak RSS, and logical bytes.
- For wall time and logical bytes, let `M5` and `M10` be the two measurements
  and define the conservative full projection as
  `max(20*M5, 10*M10, M10 + 18*max(0, M10-M5))`. Each projection must be at
  most 80% of its full gate limit. Peak RSS must be at most 200 GiB on the 10%
  run and may grow by no more than 25% from the 5% run. Focused tests must also
  prove that the implementation has no population-sized in-memory accumulator.
  Any failed condition stops RQ3 before selector search.
- Retain the limits of 12 wall-clock hours, 250 GiB peak descendant-tree RSS,
  and 250 GiB logical output/scratch bytes.
- Admit the full measurement only after ten consecutive one-minute load samples
  at most 16, with no active training-queue or other G4 materialization job.
- Continue recording timed global load after launch, but do not invalidate an
  otherwise passing run because unrelated host work raises it. Descendant-tree
  RSS and logical output remain binding.
- Unrelated contention means one-minute global load exceeds 16 for two
  consecutive post-launch samples. If wall time exceeds 12 hours and this exact
  condition occurred, classify the result as inconclusive and permit one fresh
  quiet rerun. A wall-time failure without that condition, a second wall-time
  failure, any RSS failure, or any output-size failure stops RQ3 before
  downstream recommender training.
- The complete native-500M measurement must start from an empty size-scoped
  directory, cover all folds and artifact writes, and use the existing 10-Hz
  descendant-tree RSS measurement. A sample or projection cannot pass it.

The native-50M retry-17 cost is feasibility context only: 2.875 hours,
77.25 GiB peak descendant RSS, and 1.272 GiB logical output. Native-500M raw
events are about 10 times larger and retained remapped likes about 28.5 times
larger, so the unchanged in-memory implementation must not enter the full gate.

## Interpretation and reporting

- RQ1 support requires a Recall@100 gain over the fresh native-500M baseline
  greater than its scaled 1.684781% dispersion.
- RQ2 is non-inferior when its Recall@100 is no more than that band below the
  baseline; promotion still requires an improvement beyond the band.
- RQ3 selector support requires the learned-minus-strongest-deterministic paired
  user-bootstrap interval on untouched-test NDCG@10 to lie above zero.
  Downstream support additionally requires Recall@100 gains beyond the band
  over both the original baseline and best fixed-window treatment.
- Unexpected results require saved target, mask, gradient, configuration,
  completion, restored-epoch, and slice checks plus a targeted experimental
  test of the proposed explanation.
- Generate native-500M tuning and compact artifacts from verified metadata.
  Reader and compact headings use consecutive `RQ{i}: <question>` names.
- Replace the active `README.md` tables and conclusions only with usable
  native-500M evidence. Preserve all native-50M artifacts and the old report in
  audit storage.
- Promote the qualifying target rule with highest Recall@100. Ties within the
  Recall@100 band prefer next-10, 24-hour, deterministic period, learned hard,
  then learned proportional. The selected standalone run is the aggregate; no
  duplicate aggregate run is launched.
- Aggregated improvement reports Recall@100, NDCG@100, and Coverage@100. Because
  the axis is mutually exclusive, the standalone sum equals aggregate gain and
  the interaction gap is zero and descriptive.

## Acceptance criteria

- RQ1: compare the 24-hour objective with next-item training. Report results by
  target distance and user activity. It most probably should be better because
  it aligns better with the evaluation.
- RQ2: compare next-10-events with both next-item and 24-hour training. Report
  results by target distance and user activity. It should not be much worse
  than the baseline.
- RQ3: the learned selector must beat the best deterministic selector, and its
  downstream model must beat next-item and the best fixed-window control.
  Report selector quality and materialization cost. It must not be worse than
  the baseline and will most probably be better. If implementation or runtime
  is too difficult, apply the predeclared selector/runtime gate rather than
  launching downstream recommender runs.

## Approval

- User selected native Yambda-500M on 2026-08-30.
- User selected SwiGLU intermediate width 192 for every native-500M arm on
  2026-08-30. Historical native-50M SwiGLU-171 artifacts remain unchanged.
- User selected G1's best aggregate recipe as the G4 baseline on 2026-08-30,
  with its depth increase removed. The G4 control therefore transfers the ten
  non-depth members, stays at two layers and width 64, fixes batch 512, and is
  tuned afresh on native 500M.
- User simplified recommender tuning to deep LR only on 2026-08-31. Embedding
  LR and the 15-epoch horizon stay fixed to G1, and the search starts from G1's
  deep LR.
- User approved the complete proposal on 2026-08-31: exact treatments,
  one-dimensional deep-LR grids, 66 expected and 98 maximum runs, conditional
  RQ3 feasibility stage, retained 12-hour/250-GiB/250-GiB limits, and the
  post-launch load interpretation.
