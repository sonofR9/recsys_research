# G2 eSASRec experiment plan

## Question and hypothesis

- Research questions and status:
  - RQ1, `not_started`: what are official RecTools eSASRec's metrics on
    Yambda-50M, how do they compare with the selected local recipe, and does the
    local implementation preserve the official LiGR and loss behavior on fixed
    tensors?
  - RQ2, `not_started`: what does each pluggable change buy under the repository
    protocol: standard versus LiGR blocks and tuned gBCE versus sampled softmax?
  - RQ3, `not_started`: does mixed uniform/in-batch sampling improve coverage
    without a resolved recall@100 loss, and should logQ be applied?
- Current understanding: eSASRec is SASRec shifted-sequence training with LiGR
  blocks and sampled-softmax loss. The paper treats mixed uniform/in-batch
  negatives as optional. Existing branch evidence is useful audit material but
  used 500M, full softmax instead of gBCE, no mixed-sampler arm, and no
  size-matched uncertainty band, so it cannot select the G2 result.
- Falsifiable hypotheses:
  - the local LiGR, gBCE, and sampled-softmax modules match RecTools on fixed
    tensors within dtype-appropriate tolerance;
  - LiGR plus sampled softmax improves recall@100 or NDCG@100 over the
    independently tuned standard/gBCE recipe by more than the native-50M band;
  - a mixed sampler can improve coverage@100 beyond its band while keeping
    recall@100 within its band of uniform eSASRec.
- Why the result matters: G2 decides whether eSASRec or one of its component
  recipes should replace the size-calibrated G1 structure as the next
  experiment group's baseline.

## Comparison

- Unchanged primary control: the G1 `future_baseline` structure on native 50M:
  dimension 64, two layers, two query heads and one KV head, SwiGLU width 192,
  pre-LayerNorm with input RMSNorm and final LayerNorm, learned forward
  positions, attention window 50, sequence length 128, 16 additive timestamp
  bins, linear 20-epoch schedule, and uniform sampled softmax with 512
  negatives. Its batch and two learning rates are recalibrated in G2 and then
  frozen.
  Width 192 is retained only because this control must remain structurally
  unchanged; every new G2 SwiGLU intermediate width is divisible by 32.
- Official sanity control: RecTools 0.19.0 `SASRecModel` with `LiGRLayers`,
  `loss="sampled_softmax"`, dimension 256, two blocks, four heads, dropout 0.2,
  sequence length 100, 256 uniform negatives, batch 128, and learning rate
  0.001. It is scored with the repository split, catalog, seen-item policy, and
  evaluable-user denominator. This is an official-recipe reference, not a
  treatment selector or a metric-reproduction claim unless the compared local
  recipe is identical.
  The previous diagnostic runner derived the RecTools item map from training
  interactions and therefore exposed only 33,112 candidates, omitting 36
  validation-only items from the 33,148-item mapped catalog. The approved rerun
  extends only the RecTools item map and embedding table to all 33,148 items;
  its training input remains exactly the original 614,244 pre-cutoff events,
  followed by RecTools' unchanged per-user session-length truncation.
- Local component matrix:

  | layer family | loss | capacity |
  | --- | --- | --- |
  | official-style SASRec block | sampled softmax | ReLU FFN width 256 |
  | official-style SASRec block | gBCE | ReLU FFN width 256 |
  | parameter-matched SASRec diagnostic | sampled softmax | ReLU FFN width paired to selected LiGR capacity |
  | parameter-matched SASRec diagnostic | gBCE | ReLU FFN width paired to selected LiGR capacity |
  | LiGR | sampled softmax | selected SwiGLU multiplier |
  | LiGR | gBCE | the same selected SwiGLU multiplier |

  LiGR uses pre-norm and gates both attention and FFN residuals. Its SwiGLU
  multiplier is selected from `{2, 4, 6}` under sampled softmax, yielding
  widths `{512, 1024, 1536}`, and is then held fixed for gBCE. All are divisible
  by 32. The parameter-matched standard widths are respectively `{1024, 1792,
  2560}`; the exact parameter counts must be within 2% of the paired LiGR stack
  and are recorded before training. All component models keep dimension 256,
  two blocks, four heads, dropout 0.2, sequence length 100, and project
  initialization standard deviation 0.02 with norm gains left at one.
- Loss treatments: sampled softmax and gBCE see the same 256 uniformly sampled
  catalog negatives and shifted-sequence targets. Sampled softmax has no logQ
  correction in this factorial. gBCE tunes `t` in `[0.25, 1.0]` and includes
  the paper's `0.75` anchor.
- Mixed-sampler treatment: the selected LiGR/sampled-softmax model keeps 256
  total negatives and varies the uniform fraction continuously in `[0.2, 0.8]`
  plus logQ `{none, fully corrected Yi-2019}`. Both paper anchors at uniform
  fraction 0.6 are forced into the search.
- Factors held fixed: native data, mapped catalog, targets, evaluation,
  selected global batch, optimizer type, initialization, epoch validation,
  checkpoint restoration, and total negative count within the component and
  mixed-sampler comparisons.
- Deterministic parity gate: with identical fixed weights, inputs, masks,
  negative item IDs, and logits, compare local and RecTools LiGR forward output,
  standard-block forward output, sampled-softmax loss, gBCE transform/loss, and
  gradients. Once this gate passes, every treatment, tuning, and confirmation
  run uses the local implementations. RecTools is used only as the parity oracle
  and for the three official-recipe reference runs. A parity failure blocks full
  training until the local implementation is corrected; it does not trigger a
  fallback to RecTools training.

## Data and evaluation

- Single dataset size for every RQ, tuning run, repeat, and final
  result: **native Yambda-50M**. No 500M result will select or appear in the
  active G2 report; existing 500M artifacts remain immutable audit evidence.
- User validation reference for dataset size: approved on 2026-08-25.
- Sampling: no final-run user sampling. Correctness smoke tests may use at most
  2,000 users selected by user ID and have distinct diagnostic artifact names.
- Events and split: likes only; core items with at least five interactions;
  final seven days held out by timestamp; train only mapped items; score the
  full mapped catalog; do not exclude previously seen items.
- Primary metric: recall@100. Secondary metrics: recall, NDCG, MRR, capped
  recall, and coverage at 10/50/100; parameter count; epoch and wall time; peak
  GPU memory; training throughput; full-catalog inference throughput and p50/p95
  latency.
- Uncertainty: after control selection, run the exact unchanged control for ten
  seeds `42..51`. For every metric, its absolute band is the sample standard
  deviation across the ten validation-selected restored checkpoints. The
  reader threshold is that value rounded upward to one significant digit.
  Bands are fixed before interpreting any treatment and are not confidence
  intervals. The seed-42 control tuning winner may be reused as the first repeat
  only if its complete configuration is identical. In that exact-reuse case,
  the compiled repeat batch contains only seeds `43..51`; the conditional
  seed-42 repeat identity remains reserved in the 135-slot maximum.
- Decision order: recall@100 first. Values within its band are tied; break ties
  by NDCG@100 and then lower cost. The mixed sampler is selected only if recall
  is non-inferior within the recall band and coverage improves beyond its band.

## Hyperparameter selection

- Control search: 20 Optuna trials jointly over batch
  `{128, 256, 512, 1024, 1280}`, embedding LR `[1e-4, 0.256]` log-uniform, and
  deep LR `[1e-4, 0.128]` log-uniform. The selected batch is promoted to the G2
  global batch and reused by every local treatment. A candidate is eligible
  only if every planned LiGR configuration fits at that batch; infeasible
  candidates are recorded, not silently replaced.
- Each of the six component methods receives 12 Optuna trials. Sampled-softmax
  methods tune the two log-uniform rates; the LiGR sampled-softmax method also
  tunes its multiplier. gBCE methods tune both rates and `t`, keeping the layer
  family's selected capacity fixed. The paper anchors are forced trials.
- The mixed sampler receives 12 Optuna trials: two forced 0.6 anchors followed
  by ten TPE trials over uniform fraction and logQ choice. It reuses the selected
  LiGR/sampled-softmax rates and capacity.
- μP is not used: no hyperparameter is transferred across width, architecture,
  or dataset size.
- LR boundary rule: if a selected learning rate is within 5% of its search
  bound, add one otherwise-identical point threefold beyond that bound and one
  midpoint point, subject to finite stable rates. At most two extensions per
  method are pre-approved; selection waits for them. Standalone report
  generation recomputes this trigger from every initial winner and rejects a
  compiled manifest that omits either required boundary slot.
- Training: constant-schedule component methods validate every epoch, stop with
  patience 10, restore the best recall@100 checkpoint, and start with a
  100-epoch safety cap. A run that reaches the cap or selects its cap epoch is
  extended and rerun before use. The unchanged G1 control completes its declared
  20-epoch linear horizon and restores the best epoch within it.
- RecTools reverse positions are defined on `ses[:-1]` during training. Local
  component runs therefore offset the terminal target before assigning reverse
  positions, so every scored training history has exactly the same positions
  as cutoff inference on that history.

## Run stages and compute

- Correctness stage: tensor parity tests plus four at-most-2,000-user smoke runs
  covering standard/LiGR and gBCE/sampled softmax. Smoke metrics cannot select a
  treatment or support a claim.
- Full stages:
  1. 20 control Optuna trials and up to two LR-boundary extensions.
  2. Nine additional unchanged-control repeats after reusing the exact seed-42
     winner, or ten if it cannot be reused.
  3. Six component methods times 12 trials = 72 runs, plus at most 12 LR-boundary
     extensions.
  4. Twelve mixed-sampler trials.
  5. Aggregated improvement reuses one already-trained atomic bundle. The
     candidates are the six component winners and the eligible mixed-sampler
     winner, with the recalibrated control as fallback. A candidate qualifies
     only under the fixed recall-band or tied-recall NDCG/cost Pareto promotion
     rule; final selection among qualifying candidates uses the same band-aware
     recall, NDCG, and wall-time order. No duplicate closing run is launched.
  6. Rerun the three official RecTools 50M seeds. The existing artifacts lack
     the RecTools version and source hashes required to prove exact provenance;
     they remain diagnostic audit evidence. Eligibility pins both the RecTools
     sources and the local catalog adapter, runner, split/scoring protocol, and
     provenance implementation recorded by the runner. The RecTools contract
     covers dataset mapping, `ModelBase` and transformer fitting, negative
     sampling, item/position/backbone/similarity modules, and the Torch ranking
     path used by recommendation. The same contract pins Python 3.12.13,
     RecTools 0.19.0, torch 2.7.1, PyTorch Lightning 2.5.2, NumPy 1.26.4,
     pandas 2.2.3, and Polars 1.43.2.
     Local eligibility likewise hashes the centrally enumerated entry,
     configuration, data preparation, sequence dataset, model, target, loss,
     evaluation, training-loop, optimizer, and runtime-support sources. The
     verifier rejects a missing, extra, or changed source entry.
  7. If a result reverses both the paper and every existing G2 diagnostic,
     pre-approved confirmation is limited to the two implicated existing tuned
     configurations at seeds 43 and 44 (four runs) after implementation checks
     pass. Each confirmation directly records its tuned source artifact. Exact
     source IDs, seeds, and metrics are persisted, and normal final selection
     and reporting stop until the user validates their interpretation.
- Maximum approved full-run budget: 135 runs, including all conditional
  extensions, official reruns, and confirmations. No unlisted treatment may use
  that budget.
- Submit all full runs as granular jobs through the one persistent
  `utils/training_queue` service without GPU exclusions. Group jobs by shared
  sequence/preprocessing shape; the queue may overlap CPU preparation but keeps
  one training process per admitted GPU.

## Interpretation and reporting

- RQ1 table: official RecTools eSASRec, local selected eSASRec, and the selected
  G1-derived control. Official/local metric differences are descriptive unless
  their complete recipes match; they do not establish reproduction or attribute
  a gain to sampled softmax. Fixed-tensor forward/loss/gradient parity is the
  implementation-equivalence gate.
- RQ2 tables: the six independently tuned component methods, with actual layer,
  loss, gBCE `t`, capacity, parameter count, recall/NDCG/coverage, time, and peak
  memory. Pairwise claims are made only where factors are held fixed; the
  parameter-matched rows guard against explaining LiGR effects by extra capacity.
- RQ3 table: uniform eSASRec and mixed variants, reporting the selected uniform
  fraction and logQ behavior plus the accuracy/coverage trade-off.
- Latency protocol for selected models: one A100, fixed full catalog and query
  batch, 20 warm-up iterations followed by 100 synchronized timed iterations;
  report throughput and p50/p95 latency from the same benchmark harness.
- Unexpected results require inspection of negative IDs/proposal probabilities,
  gBCE calibration, masks, initialization, norm gains, best-epoch restoration,
  parameter counts, and tensor parity before confirmation runs.
- Generated artifacts: a complete tuning ledger with every run, a compact
  native-50M RQ table draft, machine-readable selection, composition, and band
  evidence, and a reader README containing only usable native-50M results.
- Aggregated-improvement section: compare the recalibrated G1 control with the
  selected existing atomic bundle. Record all six component candidates, the
  optional eligible mixed candidate, candidate qualification separately from
  final selection, the included bundle, and every omission. For recall@100,
  NDCG@100, and coverage@100, report unrounded baseline-to-bundle point and
  percent gains, the identical standalone atomic-bundle gain, interaction gap
  zero, and an unresolved label against the native-50M size-matched band.
- Promotion: promote the selected G2 atomic bundle to the future baseline only if it
  improves recall@100 beyond the band, or ties recall and improves NDCG@100
  beyond its band without a cost regression that removes it from the
  quality/performance Pareto frontier.

## Approval

- Material assumptions: native Yambda-50M is the sole research size; RecTools
  0.19.0 is the official reference; the current G1 `future_baseline` structure
  is the primary control; the 135-run maximum includes conditional work rather
  than guaranteeing every conditional run will launch.
- Exact scope requested for approval: dataset size, RQs, controls, six-method
  component matrix, mixed-sampler surface, uncertainty repeats, selection rule,
  training semantics, and maximum run budget above.
- User approval: approved on 2026-08-25, including the later amendments that
  every new G2 SwiGLU width is divisible by 32 and parity-matched local
  implementations are the mandatory research-training path, and that the five
  duplicate closing identities are removed in favor of exact tuned-artifact
  reuse with machine-readable zero-interaction composition evidence.
