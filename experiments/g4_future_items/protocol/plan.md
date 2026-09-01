# G4 research plan: future-item supervision

## Question and hypothesis

- Research questions and status:
  - RQ1, not started: does a 24-hour future window help?
  - RQ2, not started: does a next-10-liked-events window help?
  - RQ3, not started: can behavior-similar future periods define better positives?
- Current understanding: next-item supervision may overemphasize the arbitrary
  first future event. Broader, strictly future positive sets may align training
  better with the final-seven-day retrieval evaluation.
- Falsifiable hypotheses:
  - RQ1: uniform future-event sampling inside 24 hours improves Recall@100 over
    next-item training.
  - RQ2: uniform sampling among the next ten liked events is non-inferior to
    next-item training and competitive with the 24-hour rule.
  - RQ3: a learned period-similarity predictor ranks useful future
    periods better than any individual deterministic signal, and its selected
    positives improve downstream Recall@100 over the fixed-window objectives.
- Why the result matters: the winning rule changes only construction of
  training positives, so it can improve the existing serving model without
  changing inference.

## Comparison

- Unchanged control: preserve the fixed native-50M G1 recipe currently recorded
  in `control_manifest.json`; every prefix predicts its next liked item. Its old
  canonical hash
  `052b6491ded0295b2a14b322f1a744dd2ad3a9624e31ca461f7f694e463db594`
  identifies only the pre-revision recipe and is not a launch manifest. After
  approval, freeze a new canonical manifest/hash that fixes batch `512` and
  declares embedding rate, deep rate, and schedule horizon as the three tuned
  fields. G4 owns its control recipe and source closure.
- RQ1 treatment: every eligible prefix samples one future liked event uniformly
  from `(prefix timestamp, prefix timestamp + 24 hours]`.
- RQ2 treatment: every eligible prefix samples one future liked event uniformly
  from ranks 1 through 10 after that prefix.
- RQ3 selector families:
  - matching UTC hour-of-week;
  - content-centroid cosine similarity;
  - weighted-Jaccard item, artist, and album frequency similarities;
  - a learned classifier using the same liked-event similarity features as the
    deterministic selectors plus time gap, prefix/candidate hour and day
    features, period activity, and past user activity;
  - hard top-k and score-proportional positive sampling for the selected learned
    classifier.
- RQ3 period widths: UTC-aligned 1-hour, 6-hour, and calendar-day periods.
  Candidate lookahead is bounded to 3/7 days for 1-hour and 6-hour periods and
  14/28 days for calendar-day periods.
- Factors held fixed: final data protocol, control architecture, sequence
  length, attention window, timestamp tokens, negative proposal, optimizer,
  linear schedule shape, initialization, evaluation, seed policy, and batch
  size `512`. The schedule horizon is tuned.
- Method-specific capacity: none; all recommender arms use the control model.
- Sanity control: reproduce target counts and next-item outputs from the
  unchanged target builder before enabling broader positives.
- Future-item and period artifacts alter training supervision only. Validation
  keeps the control's held-out next-item loss and ranking metrics; validation
  batches never query train-only future-event or selector artifacts.

For RQ1 and RQ2, each causal prefix contributes one sampled positive per epoch.
Event occurrences, rather than deduplicated item ids, define the sampling
distribution. Every unique valid positive item for that query is masked from
its negatives. If a prefix has no 24-hour candidate, RQ1 falls back to that
prefix's next liked item; RQ3 applies the same fallback when its selector has no
candidate. This preserves the control's prefix/user distribution and its frozen
`P_control` prefix-positive pairs and `S_control` optimizer steps per epoch
without duplicating eligible prefixes. Report those counts, eligibility, and
fallback rates. No positive crosses the training cutoff.

RQ3 creates all selector labels and candidates strictly inside the training
interval. The common selector universe contains, for every causal liked-event
prefix, every later liked-event occurrence from that user in `(prefix,
prefix + 28d]`. The independent relevance outcome for an occurrence is
weighted-Jaccard similarity between listened-artist frequencies in the trailing
24 hours `(prefix timestamp - 24h, prefix timestamp]` and in the strictly
post-prefix portion of the
UTC calendar day containing the candidate occurrence: `(max(prefix timestamp,
day start), day end)`. The binary threshold uses the nearest-rank empirical
80th percentile on selector-train: sort `n` outcomes and take index
`ceil(0.8*n)-1`. The label is `similarity > threshold`; therefore, if the
threshold is zero, only positive similarities are relevant. An
empty listen-artist union has similarity zero; each listen occurrence contributes
unit mass split evenly across its nonzero mapped artist ids. Liked-event similarities are
selector inputs, never labels. Every
selector ranks this identical event universe; events outside its lookahead or
inside an ineligible period receive score zero. Thus query sets, relevance,
NDCG denominators, and candidate counts are common across structural choices.

For a trial with width `w`, its causal liked-event past is exactly
`(prefix timestamp - w, prefix timestamp]`. Candidate periods are UTC-aligned,
use half-open `[start, end)` bounds, start strictly after the prefix, and end at
or before the applicable partition cutoff. Let the
pre-evaluation span be integer Unix seconds `[T0, Tcut)`, let
`T0` be the minimum timestamp among retained mapped training likes and `Tcut`
be the exact final-evaluation cutoff frozen in the control semantics manifest;
let
`D = Tcut - T0`, `T70 = T0 + floor(70*D/100)`, and
`T85 = T0 + floor(85*D/100)`. Selector train is `[T0, T70)`, validation is
`[T70, T85)`, and test is `[T85, Tcut)`. A query and all candidate/label events
used for that row must lie wholly inside the same half-open partition; the
documented future and trailing windows retain their own open/closed endpoints.
Tune on validation and apply the gate once on untouched test. No selector may
use the final-seven-day events.
Keep every common-universe query with at least one candidate; no-positive
queries contribute zero to NDCG@10. AUROC pools candidate pairs and is reported
only when both classes exist. For selector tuning, gate point estimates, and
bootstrap inputs, first average query NDCG@10 within each user and then average
users equally. This user-balanced NDCG is the sole selector selection objective.

Deterministic scores are fully specified as follows. The time score compares
the exact prefix timestamp with the candidate-period start and is one when
their circular UTC hour-of-week distance is at most the tuned tolerance and
zero otherwise. Define continuous UTC hour-of-week as
`24*((floor(t/86400)+3) mod 7) + (t mod 86400)/3600`, where Monday is zero and
Unix epoch day is Thursday (`3`); circular distance is the shorter distance on
the 168-hour ring. Content cosine uses the occurrence-weighted
arithmetic mean of normalized Yambda item-content embeddings in that causal
past and candidate period, mapped from `[-1, 1]` to `[0, 1]`. An occurrence
without content is omitted; an empty or zero-norm mean centroid scores zero.
Weighted Jaccard is
`sum(min(a_i, b_i)) / sum(max(a_i, b_i))`; each liked event contributes unit
mass for item id and unit mass split evenly over its nonzero mapped artist or
album ids; missing/unknown id zero is excluded. An empty union scores zero.
Candidate periods with fewer than the tuned minimum liked events are
ineligible. Scores tie by earlier period start, then compact item id.

The learned feature vector is exactly: continuous circular time similarity
`1 - distance_hours/84` using the distance above, the content score, and the
three liked-event Jaccard scores;
`log1p(candidate_period_start - prefix_timestamp)` in integer seconds; log1p
liked-event counts in the causal past and
candidate period; log1p user like counts in trailing 7d and 28d before the
prefix; log1p active UTC days in that trailing 28d; and sine/cosine encodings of
prefix and candidate-start UTC hour-of-day and day-of-week. For timestamp `t`,
hour phase is `2*pi*((t mod 86400)/86400)` and weekday phase is
`2*pi*(((floor(t/86400)+3) mod 7)/7)`; the prefix uses its exact timestamp and
the candidate uses its period start. Count/gap features
are standardized by fit-row mean and population standard deviation, replacing
a zero standard deviation by one. Inverse-frequency fit weights are
`N/(2*N_class)`, which gives mean weight one. The classifier is
`HistGradientBoostingClassifier(loss="log_loss", max_iter=200,
min_samples_leaf=20, max_bins=255, early_stopping=False, random_state=42)` with
only maximum leaves, learning rate, and L2 varied as declared below. It emits a
probability in `[0, 1]`. If selector-train has no rows or lacks either label
class, the learned selector is unfit and fails the RQ3 gate without downstream
training.

Cross-fitting has three explicit stages:

| stage | fitted artifacts | fit rows | selection/evaluation | output |
| :--- | :--- | :--- | :--- | :--- |
| selector search | relevance threshold, normalizers, class weights, classifier | first 70% selector-train | choose hyperparameters on next 15% validation | one locked configuration |
| selector gate | none | none | compare locked learned and strongest locked deterministic pipelines once on final 15% test | frozen selector evidence |
| downstream materialization | five fold-local thresholds, normalizers, class weights, classifiers | after the gate, all pre-evaluation rows from four user folds | no reselection or selector claim | scores for the fifth fold |

Downstream folds use the first eight SHA-256 bytes of canonical UTF-8 JSON
`["g4-fold-v1",uid,42]`, interpreted as an unsigned big-endian integer modulo
five. JSON integers are base-10 values with no whitespace, and `uid` is the
stored integer user id. Every
fold-fitted artifact excludes the scored user fold; global hyperparameters were
selected chronologically before this refit and are not claimed user-nested.
Selector-test evidence remains the pre-refit gate result and is never rescored.
Freeze all selector settings before recommender training.

## Data and evaluation

- Single dataset size: native Yambda-50M for every RQ, tuning run, selector,
  and final result.
- User validation reference: approved on 2026-08-28.
- Sampling: no final-run user sampling. Any bounded diagnostic sample is
  selected by hashed user id and cannot select a treatment.
- Protocol: recommender histories and targets use likes only; RQ3 may use
  training-interval listens solely for its independent selector label. Core
  items have at least five like interactions; drop
  unmapped items; hold out the final seven days by timestamp; train only mapped
  items; score the complete mapped catalog; do not exclude seen items.
- Primary metric: full-catalog Recall@100 on all 3,414 evaluable users.
- Secondary metrics: Recall/NDCG/MRR at 10, 50, and 100; capped Recall at 10,
  50, and 100; catalog coverage at 10, 50, and 100; targets/s, wall time, and
  materialization cost.
- Slices:
  - target temporal distance: `(0, 6h]`, `(6h, 24h]`, `(1d, 3d]`, `(3d, 7d]`;
  - target event rank: 1, 2-5, 6-10, and 11+;
  - user activity quartiles computed from training-period like counts.
- Slice construction reuses each model's one full-catalog ranking. Order every
  post-cutoff relevance event by `(timestamp, compact_item_id)` per user, assign
  each occurrence to one temporal and one rank bin, and deduplicate item ids
  within each user/bin. A repeated item may appear in multiple bins when it has
  occurrences there; a user may contribute to multiple bins and contributes to
  a bin only when it has at least one mapped relevant item there. Standard
  recall divides hits by unique relevant items; capped recall divides by
  `min(k, relevant items)`. Report the contributing-user denominator for every
  slice. Activity-quartile cut points come from training-like counts of the
  fixed evaluation users: sort by `(like_count, uid)` and assign quartile
  `floor(4 * zero_based_rank / num_users)`. This deterministically splits count
  ties and gives each user exactly one quartile. Report Recall, capped Recall,
  NDCG, and MRR at 10/50/100 plus coverage and user count for every temporal,
  rank, and activity slice.
- Shared empirical bands: reuse the one-time canonical native-50M calibration
  from the G1 aggregate experiment's selected batch-512 unchanged control at
  seeds 42 through 51 (repeat queue batch
  `5c6250d40aa84ad5925f8389069c2b47`). For each metric, use only its reviewed
  relative dispersion (`sample stddev with ddof=1 / unrounded calibration
  mean`) and multiply it by G4's own unrounded baseline metric. G4 launches no
  control repeats and never reuses the calibration's absolute means or absolute
  bands. Never import the native-500M calibration.
- Baseline identity has two immutable resolution stages. Before control tuning,
  validate `control_manifest.json` against the launcher/config dataclass schema,
  instantiate its anchor, and emit `control_semantics_manifest.json` containing
  the family-manifest hash, complete resolved configuration, dataset key,
  split/catalog identities, target-builder identity fixture, training-semantic
  revisions, and hashes of every control-affecting source. Hash and freeze that
  manifest before the first tuning job. After tuning, emit and hash
  `selected_control_manifest.json` containing the semantics-manifest hash,
  fixed batch, selected rates/horizon, selection evidence, and exact seed-42 configuration.
  Every treatment must verify the selected manifest before launch. After
  implementation and independent review, emit and freeze a
  `treatment_semantics_manifest.json` containing the selected-control hash,
  reviewed G4 source closure and hashes, selector/materializer schemas,
  objective and mask fixtures, and artifact formats. Every selector and
  treatment job verifies that manifest. Implementation of this approved plan
  is the only permitted pre-freeze source delta; any unapproved semantic/data
  change or any source change after its applicable freeze requires a revised
  plan. Repeats cannot absorb it. `manifest_contract.md` is the binding byte,
  schema, runtime-mapping, identity, and permitted-delta specification.

## Hyperparameter selection

- Current-code control tuning: 20 Optuna random-search trials with sampler seed
  42 at fixed batch `512`, over embedding LR `[1e-4, 0.256]`, deep LR
  `[1e-4, 0.128]`, and integer linear-schedule horizon `[5, 30]` epochs. Both
  rates are log-uniform and the horizon is uniform in five-epoch steps. Include
  the old selected rates with a 20-epoch horizon as one anchor.
- Each changed objective tunes both optimizer groups because the objective
  changes: 12 seeded Optuna random-search trials at batch `512` over embedding
  LR `[1e-4, 0.256]`, deep LR `[1e-4, 0.128]`, and integer linear-schedule
  horizon `[5, 30]` epochs. Both rates are log-uniform and the horizon is
  uniform in five-epoch steps. The control rates and horizon are included as
  one anchor trial.
- RQ3 gives equal 12-trial recommender budgets to the best deterministic
  selector, learned hard top-k, and learned proportional-sampling arms. Hard
  selection tunes `k in {1, 2, 4}`; proportional sampling tunes the number of
  periods drawn in `{1, 2, 4}` inside the same 12-trial budget.
- Every recommender base study uses Optuna sampler seed 42, training seed 42,
  identical data-order seeding, and an objective-specific positive RNG derived
  from `(42, epoch, objective)`. Select the horizon-complete run with highest
  validation Recall@100. Values whose Recall differs by at most `1e-6` tie and
  go to lower validation loss, then shorter schedule horizon, smaller `k`/draw
  count for treatments, lower sum of learning rates, lower embedding rate,
  lower deep rate, lexicographically smaller canonical parameter JSON, and
  finally run name. A nonfinite Recall or loss makes the run unusable.
  Final all-user metrics never select a run.
- Rate boundary rule: for interval `[lo, hi]`, a selected rate `x` triggers when
  `z=(ln(x)-ln(lo))/(ln(hi)-ln(lo))` satisfies `z <= 0.1` or `z >= 0.9`.
  Extend it fourfold in the triggered direction. Freeze every categorical and
  non-triggered continuous value from the base-study winner; if both rates
  trigger, expand and resample both simultaneously. Run one four-trial joint
  round with sampler seed 43 and select cumulatively from the entering winner
  plus every trial in that round. If its cumulative winner again triggers,
  repeat once with seed 44 and the same freeze rule. Thus every study has at
  most eight boundary trials. The same total tie-break above applies to every
  cumulative selection. If the second extension still selects a boundary,
  selection stops and a newly approved expanded plan is required; the bounded
  plan does not claim that unresolved surface.
- Horizon boundary rule: the base domain is exactly
  `{5, 10, 15, 20, 25, 30}`. A base winner of 5 triggers the inclusive integer
  domain `[2, 30]` with step 1 in round one; a base winner of 30 triggers
  `[5, 40]` with step 1. After the lower expansion, round two triggers only when
  the cumulative winner is in `{2, 3, 4}` and uses `[1, 30]` with step 1. After
  the upper expansion, round two triggers only when the cumulative winner is in
  `{37, 38, 39, 40}` and uses `[5, 50]` with step 1. After round two, a winner
  in `{1, 2, 3}` for the lower domain or `{46, 47, 48, 49, 50}` for the upper
  domain remains unresolved and requires a newly approved expansion. A winner
  at the old 5/30 endpoint that is interior after expansion does not itself
  trigger round two. Boundary trials sample every triggered rate/horizon
  dimension jointly in the same four-trial round and freeze the rest. Model
  capacity and token horizon do not change; no model-size or token-horizon
  transfer is used.

## Run stages and compute

- Focused checks:
  - exact handcrafted target sets for next-item, 24-hour, next-10, and period
    selectors;
  - strict cutoff, causal-prefix, duplicate-item, pair-budget, false-negative
    mask, and deterministic-seed checks;
  - miniature end-to-end training and sliced metric regression;
  - target-builder parity for the unchanged next-item path.
- Control: 20 tuning runs plus up to eight joint rate/horizon boundary runs.
  The selected seed-42 tuning run is G4's baseline. Reuse the external shared
  native-50M relative calibration; launch no G4 control repeats.
- RQ1: 12 tuning runs, plus at most eight boundary runs.
- RQ2: 12 tuning runs, plus at most eight boundary runs.
- RQ3 selector stage uses sampler seed 42 and 12 validation-selected trials for
  each of four families, for 48 behaviorally distinct trials total. Shared
  categorical axes are period width `{1h, 6h, 24h}` and minimum liked events
  per candidate period `{1, 2, 4}`. Content, frequency, and learned lookahead is
  `{3d, 7d}` for 1h/6h periods and `{14d, 28d}` for 24h periods. To avoid
  degenerate weekday matches, the time family uses 7d for 1h/6h and `{14d,
  28d}` for 24h. All intervals are `(prefix timestamp, prefix + lookahead]`.
  The time matcher additionally searches hour-of-week tolerance `{0, 1h, 2h}`;
  content cosine has no extra axis; weighted Jaccard searches liked-event entity
  `{item, artist, album}`;
  and the learned `HistGradientBoostingClassifier` searches maximum leaves
  `{7, 15, 31}`, learning rate log-uniform in `[0.01, 0.2]`, and L2
  regularization log-uniform in `[1e-5, 1]`. Finite categorical combinations
  are sampled without replacement. Fit feature normalization on selector-train
  only: map cosine from `[-1, 1]` to `[0, 1]`, keep Jaccard in `[0, 1]`, log1p
  count/gap features then standardize them by train mean/std, and encode cyclic
  hour/day values with sine/cosine. Select each family on validation NDCG@10,
  treating differences at most `1e-12` as ties and breaking them by AUROC,
  shorter lookahead, wider period, fewer minimum events, then family-specific
  simplicity: lower time tolerance; item before artist before album; or fewer
  maximum leaves, lower learning rate, and lower L2. Remaining ties use
  lexicographically smaller canonical parameter JSON and then run name. A
  nonfinite NDCG makes a trial unusable; undefined AUROC sorts below every
  defined AUROC and ties another undefined AUROC. Report
  validation and untouched-test
  NDCG@10, AUROC, query/pair/class counts, positive rate, and materialization
  cost.
- The strongest deterministic pipeline is the validation winner across all 36
  time/content/frequency trials, not a scorer conditional on the learned
  structure. Because every pipeline scores the common 28-day event universe,
  compare that locked deterministic winner with the locked learned winner on
  identical test queries, candidates, relevance labels, and denominators.
- The learned selector's continuous learning-rate surface follows the same
  inclusive outer-10%, four-trial, one-repeat rule. A boundary round freezes
  width, lookahead, minimum events, maximum leaves, and L2 at the learned base
  winner, varies only learning rate, uses seeds 43 then 44, and selects
  cumulatively with the selector tie-break above. If the second extension
  remains at the boundary, RQ3 selection stops for a newly approved expansion.
- RQ3 gate: stop before recommender training if the learned classifier does not
  resolve above the strongest deterministic selector on untouched-test
  NDCG@10, or if full native-50M materialization exceeds 12 hours, 250 GiB peak
  aggregate RSS, or 250 GiB logical output/scratch size.
  Resolution uses the distinct selector-test users with at least one common
  query. Precompute each user's mean query-level learned-minus-deterministic
  NDCG@10 difference. `Generator(PCG64(42))` makes 10,000 replicates by drawing
  exactly the number of eligible users uniformly with replacement and taking
  the sampled multiset mean. Sort replicate means; the lower nearest-rank 2.5%
  endpoint is zero-based element `ceil(0.025*10000)-1 = 249`, without
  interpolation, and must exceed zero. The upper endpoint analogously uses
  `ceil(0.975*10000)-1`. Report the unsampled user mean and this 95% interval.
  Preserve and report failed or
  unresolved selector evidence.
- The materialization gate has one reference fixture: host
  `a100-1.vla.yp-c.yandex.net`, one AMD EPYC 7702 socket, CPUs 0–15, one process
  tree, 16 materialization workers, 929,980,153,856 bytes of host RAM, Python
  3.12.13, `POLARS_MAX_THREADS=16`, and
  BLAS/OpenMP thread counts one. Reuse the verified immutable base
  Yambda/remap/content caches with `data.invalidate_cache=false`, but start from
  an empty dedicated G4 output/scratch directory. Immediately before the quiet
  load window, sequentially read and SHA-256-verify every input file; at launch,
  require `mincore` to report every input page resident, otherwise repeat the
  prewarm. The timed region begins
  before constructing the common universe and ends only after five-fold fit,
  scoring, artifact serialization, `fsync`, and hash verification. It includes
  input reads and all output writes. No intermediate is deleted before the
  measurement, so logical bytes are the sum of regular-file `st_size` values in
  that directory; symlinks are forbidden. A supervisor samples `/proc` every
  100 ms from immediately before child creation until every descendant is
  reaped. At each sample it discovers the leader's complete descendant tree by
  PPID, reads every distinct PID's resident pages from `/proc/PID/statm`,
  multiplies their sum by `SC_PAGE_SIZE`, and records the maximum sampled sum as
  peak aggregate RSS. A vanished PID contributes zero at that sample. This
  10-Hz process-tree statistic, rather than unavailable cgroup accounting, is
  the binding memory measurement. GiB means `2^30` bytes.
  Start only after ten minutes with one-minute load average at most 16 and no
  active training-queue or other G4 materialization job; monitor load every
  minute. A run with load above 16 for two consecutive samples is invalid and
  postponed rather than used for the gate. Record host, CPU, RAM, source/cache
  hashes, environment, commands, load samples, wall time, memory peak, and
  logical bytes. The gate applies to the selected learned configuration's
  complete fold-scored candidate artifact that supports every downstream
  epoch; a sample or partial build cannot pass it.
- RQ3 downstream stage after the gate: 12 recommender tuning runs for each of
  best deterministic, learned hard top-k, and learned proportional sampling,
  plus at most eight boundary runs per arm.
- RQ3 target materialization ranks eligible positive-score periods by the
  scorer definitions above. The deterministic arm and learned hard arm tune
  `k`, take the first `k` periods (or all when fewer exist), and uniformly sample
  one liked-event occurrence from their union. Learned proportional sampling
  first checks unmodified predicted probabilities; when their sum is positive,
  clip to `[1e-6, 1]`, normalize across eligible periods, draw the tuned number
  without replacement (or all when fewer exist), then uniformly sample one
  liked-event occurrence from their union. When no eligible positive-score
  period exists, use the next liked item. Keep
  all candidate pairs for selector evaluation and classifier fitting; use
  inverse-frequency sample weights rather than pair/class subsampling. The
  12-hour, 250-GiB RSS, and 250-GiB logical-size gate applies to this complete
  native-50M construction.
- False-negative masks follow each objective's complete acceptable set. RQ1
  masks unique item ids from the next 24 hours; RQ2 masks unique ids from the
  next ten events. Deterministic and learned-hard RQ3 mask unique ids in their
  selected top-k period union. Learned-proportional RQ3 masks unique ids from
  every eligible positive-probability period because any may be drawn. A
  fallback query masks only its next-item target.
- Every stochastic target draw is traversal- and worker-independent. RQ1/RQ2
  sort candidate occurrences by `(timestamp, compact_item_id)`. RQ3 sorts
  eligible periods by `(period_start, period_end)` and, after selecting periods,
  sorts their candidate occurrences by `(period_start, timestamp,
  compact_item_id)`. Exact duplicate keys retain their occurrence multiplicity;
  their internal order cannot change the selected period or item. Initialize
  one generator per query by deriving
  a uint64 seed from the first eight bytes of SHA-256 over canonical compact
  UTF-8 JSON `["g4-target-v1", training_seed, epoch, objective_id, uid,
  prefix_timestamp, prefix_compact_item_id]`, interpreted unsigned big-endian,
  and use
  NumPy `Generator(PCG64(seed))`. Learned-proportional uses that generator first
  for weighted period choice without replacement and then for uniform
  occurrence choice; every other stochastic objective uses its first uniform
  occurrence choice. All numeric fields are stored base-10 integers; timestamps are
  signed Unix seconds exactly as stored. Objective ids are exactly
  `control_next_item`, `rq1_24h`, `rq2_next10`,
  `rq3_deterministic_hard`, `rq3_learned_hard`, and
  `rq3_learned_proportional`. Exactly duplicated prefix keys intentionally
  receive the same draw. Save the objective id and seed derivation revision in
  run metadata.
- All recommender runs use their tuned linear-annealing horizon, validate every
  epoch on Recall@100, finish the declared horizon, restore the best validation
  epoch, and report both. A run shorter than its candidate horizon is unusable.
- One horizon-complete selected tuning run is the final single-seed result for
  each arm. The shared control calibration sets the relative resolution bands;
  treatments are not mislabeled as repeats.
- Expected work is 128 trials: 20 control-tuning runs, 60 treatment recommender
  runs, and 48 selector trials. The maximum approved scope is 184 after up to
  48 joint rate/horizon boundary trials across six
  recommender studies and eight learned-selector boundary trials.
  A failed RQ3 selector gate removes 36 downstream recommender runs and their
  possible 24 boundary trials.
- Queue: after implementation and independent protocol review, submit every
  full run through the existing persistent `utils/training_queue` service in
  granular jobs. No competing queue and no manual GPU exclusions.

## Aggregation

- Frozen original baseline: the selected seed-42 current-code unchanged
  batch-512 control from G4 tuning. Scale its metrics by the shared native-50M
  relative dispersions to obtain G4's operational bands.
- G4 treatments are mutually exclusive positive-selection rules; they cannot
  be composed.
- A treatment qualifies when its Recall@100 exceeds the active baseline by
  more than the frozen `b_recall`. Select the qualifying treatment with highest
  Recall@100. Ties inside `b_recall` go to the simpler rule in this order: next-10, 24-hour,
  deterministic period, learned hard top-k, learned proportional sampling.
- The selected standalone run is also the aggregate because G4 contains one
  mutually exclusive axis. Do not launch a duplicate aggregate. If no
  treatment qualifies, the aggregate equals the baseline and no duplicate is
  trained.
- Aggregate arithmetic includes Recall@100, NDCG@100, and Coverage@100. The
  standalone sum equals the aggregate gain and the interaction gap is zero;
  label it descriptively unresolved.

## Interpretation and reporting

- RQ1 support: Recall@100 gain greater than `b_recall`. A result inside the band is
  null; a decrease greater than `b_recall` is a regression.
- RQ2 support: non-inferior to next-item when Recall@100 is no more than
  `b_recall` lower. Promotion still requires an improvement greater than
  `b_recall`.
- RQ3 selector support: the paired user-cluster bootstrap interval for learned
  minus strongest deterministic untouched-test NDCG@10 lies above zero.
  Downstream support additionally requires
  Recall@100 gains greater than `b_recall` over next-item and the best
  fixed-window treatment.
- Unexpected results require saved checks of target construction, budgets,
  masks, gradients, configurations, run completion, restored epochs, and slice
  behavior, followed by targeted reruns or ablations that test the proposed
  mechanism.
- Generate one tuning table per method, one compact G4 table artifact, sliced
  evaluation tables, selector-quality and materialization tables, and the
  reader-facing `README.md` with one section per RQ and `Aggregated
  improvement`.
- Promote only the selected mutually exclusive target rule to the G4 future
  baseline.

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

## Additional research question proposed but excluded

- RQ4: does temporal-decay weighting of valid future positives improve over
  uniform sampling? This is a distinct objective axis and is excluded from the
  present 128-trial scope unless separately approved.

## Approval

- Material assumptions:
  - native Yambda-50M is the single experiment size;
  - batch size is fixed at `512` and is never tuned;
  - G4 tunes embedding rate, deep rate, and linear-schedule horizon, then reuses
    the G1 aggregate experiment's one-time native-50M relative dispersions and
    launches no repeat batch;
  - dataset timestamps are UTC because no user timezone is available;
  - RQ3 uses the exact selector definition and bounded stop gate above; "best
    deterministic" means the validation winner across every deterministic
    pipeline, evaluated against learned on the common event universe;
  - RQ2's "not much worse" means within the measured `b_recall`; RQ3's selector
    "beat" means a paired user-bootstrap interval above zero, while its
    downstream "beat" means a Recall@100 gain beyond `b_recall`;
  - RQ4 is not part of the current experiment.
- Revised scope requested for approval: the fixed batch-512 policy, joint
  rate/horizon surfaces and boundary rule, reuse of the one-time relative 50M
  calibration, all three G4 RQs, 128 expected trials and 184 maximum after
  bounded extensions, the RQ3 selector gate, the mutual-exclusion aggregate
  rule, and the final report.
- Original scope approved on 2026-08-28; the revised exact horizon surface and
  run scope were approved for training on 2026-08-28.
