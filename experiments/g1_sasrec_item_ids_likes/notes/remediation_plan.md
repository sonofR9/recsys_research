# G1 remediation plan

## Question and hypothesis

- Research question and status: RQ1, RQ4–RQ7, and RQ9–RQ10 are in human
  review. RQ8 awaits two head continuations and RQ11 awaits five canonical
  cap-lineage repairs; combined RQ2/RQ3 remain outside this approved
  component-remediation plan.
- Current understanding: the native 50M tuning and exact 500M component
  confirmations replaced the invalid fixed-endpoint evidence. Reader and
  scratchpad reports use only current-protocol artifacts.
- Falsifiable hypothesis: protocol-correct tuning preserves the main selected
  components, but any current conclusion may change when its exact treatment
  receives an equal tuning budget.
- Why the result matters: G1 defines the baseline inherited by later groups.

## Comparison

- Reproduction control: the exact homework model, objective, and training
  recipe on full Yambda-500M, with recall@100 required to fall in 0.1235-0.13.
- Architecture control: one tuned selected-architecture configuration, reused
  byte-for-byte within an axis.
- Each table changes only its named treatment. Competing methods receive the
  same treatment-specific LR budget at one batch selected from the unchanged
  architecture control and shared globally through RQ11.
- RQ4 covers GELU widths 128/171/256/384 and SwiGLU widths
  16/32/64/96/128/171/224.
- RQ5/RQ6 cover 13 unique schedule/warmup configurations: constant, linear,
  cosine, polynomial, exponential, WSD, step, inverse-sqrt, the controlled 5%
  warmup pairs, and cosine with one, two, and four cycles.
- RQ7 covers 19 treatments: none; learned forward, reverse, and both; RoPE
  forward and reverse; ALiBi; each RoPE direction with ALiBi; each learned
  direction with ALiBi; every RoPE/learned direction pair; and each
  RoPE/learned direction pair with ALiBi. Every row is compared with none.
- RQ8 covers dimensions 16/32/64/128/256; depths 1/2/4; sequence lengths
  12/25/50/100/128/200/256/512; MHA 1/2/4/8 and GQA 2Q/1KV; windows
  none/10/25/50/75/100; dropout 0/.05/.1/.2/.3/.5; the eight specified norm
  configurations spanning pre/post, LayerNorm/RMSNorm/BatchNorm, input norm,
  final norm, and no final norm; BOS on/off; CLS-query off/on; and the RQ4
  FFN-capacity evidence. The CLS hidden state replaces the final history-state
  query for the same held-out next-item target.
- RQ9 covers exactly 15 configurations: no time feature; forward/reverse raw
  timestamp RoPE; forward/reverse log-time RoPE; plain/log delta addition;
  8/16/32/64-bin addition; bins plus reverse timestamp RoPE; log-delta
  concatenation; bin concatenation; and bins plus log-time RoPE.
- RQ10 compares one shared item table with per-layer tables.
- RQ11 covers fixed global-q Yi-2019, homework fixed leave-one-out, streaming
  global-q Yi-2019, uniform random, popularity random with exact correction,
  uncorrected in-batch, and uniform-random mixtures with fixed and streaming
  logQ in-batch components.
- The homework-compatible fixed in-batch logQ arm must beat its otherwise
  identical uncorrected control as it does in the homework. Failure of this
  reproduction blocks RQ11 and triggers an effective-negative-pool and
  correction-semantics audit; it is not reported as a random-negative win.

## Data and evaluation

- Hyperparameter tuning uses full Yambda-50M. Final conclusions use full
  Yambda-500M. Smoke samples, if needed, are selected by user ID.
- All runs use likes, core-item interactions at least five, the final seven-day
  timestamp holdout, mapped training items, full mapped-catalog scoring, and no
  seen-item exclusion.
- Recall@100 is primary. NDCG@100, recall@10, NDCG@10, and coverage@100 are
  secondary. Steady epoch time and peak memory are reader-facing only in RQ3.
- Ten identical homework-baseline 500M repeats define each metric's sample
  standard deviation, absolute band, and standard deviation as a percentage of
  its mean. These are shared empirical resolution bands, not confidence
  intervals or formal significance tests.

## Hyperparameter selection

- The unchanged architecture control first receives embedding LR 0.008/0.016/0.032 crossed with
  deep LR 0.003/0.006/0.012 at batch 1280. Its selected LR pair screens batch
  1024/1536/2048 against 1280. A 2048 winner extends to 2560, then to 3072
  only if 2560 still wins. The selected batch receives a local LR recheck.
- Every treatment then receives its own Cartesian LR grid at that one selected
  batch across RQ4-RQ11. Batch is not selected per treatment.
- Every downstream launcher recomputes and verifies that global selection from
  completed control evidence before it can set up the queue.
- RQ11 additionally tunes negative counts 512/1024/2048 independently of batch
  size, streaming
  alpha 0.0025/0.005/0.01/0.02/0.04, and mixed random shares
  .125/.25/.5/.75/.875. The exact secondary winner receives a local LR grid.
- If 2048 negatives wins, extend the negative-count grid to 4096.
- μP uses a fixed 64-dimensional item table and learned projections for width
  transfer. It does not transfer changed architectures or dataset size.
- Extend every boundary winner beyond that boundary. The first extension uses
  embedding LR 0.064 and/or deep LR 0.024. If the extension boundary wins
  again, continue until the optimum is interior or a documented
  feasibility/stability limit is reached.

## Run stages and compute

The user explicitly approved the component-remediation program through RQ11,
including all new runs needed to close those research questions. RQ2, RQ3, and
the final combined run require a separate exact plan and explicit approval.

The homework baseline/recalibration uses training-semantics revision 1. The
corrected tuning and RQ1 transfer launchers use manifest-revision-2 names and
exact compatibility checks. The report generator accepts only current-revision
homework repeats, selected-batch calibrations, and RQ1 width/data-transfer
evidence; no legacy result is accepted. These paths and their selected
artifacts passed independent review.

1. Audit every existing artifact by exact configuration and code/protocol
   semantics. Reuse exact matches only; seed repeats do not count as tuning.
2. Implement missing manifest, RQ11 local-retuning, report-generation,
   uncertainty, precision, and color support. A reviewer verifies the protocol
   and frozen manifest. Non-training tests are omitted by explicit user request.
3. Run 50M tuning through the single shared queue:
   - Cartesian manifest entries across 86 non-alias architecture, schedule,
     time, and item-table treatments at their control-selected batch;
   - three batch-screen runs for the shared architecture control, plus one or
     two explicitly tagged boundary extensions and its local LR recheck;
   - 168–176 corrected RQ11 runs: 72 initial, 32 secondary, and 64–72
     exact-local runs depending on how many winners change at most one
     secondary axis.
   The fixed-batch design removes 252 treatment-specific batch screens from the
   earlier estimate; recount exact reuse before submission. The stopped
   manifest-revision-1 RQ1 artifacts are incompatible because they use batch
   512. Older legacy artifacts do not record training-semantics revision 1;
   the 108 legacy three-pair architecture artifacts additionally use
   conventional `GenerationExperiment`, not the required μP proxy.
   Conditional control-batch LR rechecks and the specified outward boundary
   extension are additional. Track and report the realized count against the
   approved planning envelope before it is exceeded.
4. Run ten fresh unchanged homework-baseline repeats under training-semantics
   revision 1. After the shared batch selection, rerun the homework-compatible
   calibration at the actually selected batch size and require recall@100 in
   0.1235-0.13 before promotion. The unchanged homework baseline, not the
   selected-batch check, supplies the shared empirical bands.
5. Run one 500M confirmation per exact method/family treatment selected from
   its proxy grid; this means one configuration for each table row, not several
   finalists inside one method and not seed repeats.
   No legacy 500M evidence is reusable. The planning envelope is 110 new 500M
   runs: ten baseline repeats, 86 non-alias
   architecture/RQ4-RQ10 treatments, eight RQ11 treatments, and the selected-
   batch calibration. Exact reuse should make the realized count lower. No
   treatment gets seed repeats.
6. Report every candidate with both 500M recall and steady epoch time, mark the
   Pareto-nondominated set, and do not attribute comparisons that change
   several factors to batch size alone. The balance choice is the fastest configuration
   whose recall@100 is within one shared empirical band of the maximum-quality
   configuration; memory breaks remaining ties.
7. After every component choice is closed, propose the exact RQ2/RQ3 recipes
   and final 500M run for user approval. Do not launch them before approval.

Every multi-run stage uses one shared training queue, one simultaneous training
per GPU, immediate admission of idle GPUs, light-GPU monitoring, overlapping
preprocessing, and no default exclusions.

## Interpretation and reporting

- A metric improvement larger than that metric's baseline band is green; a
  decline larger than the band is red; otherwise it is neutral.
- Unexpected results are checked against raw metrics, user count, convergence,
  proposal/correction invariants, and search-boundary status before acceptance.
- Generate a complete usable-run 50M tuning ledger and README-draft 50M/500M
  reports from metadata. The drafts use the README's RQ-specific result-table
  schemas and ordering and contain only the title, RQ headings, and tables.
  The tuning ledger includes usable completed controls, absolute metrics,
  percentage differences, and tuned fields, and bolds one best valid row per
  method. Raw audit storage retains failures and unusable runs; reports omit
  them together with artifact links, encoded identifiers, role/status fields,
  and provenance-only columns.
- Store rejected, interrupted, incompatible, and legacy run artifacts under
  `generated/logs/old/`. Launchers archive an invalidated canonical artifact
  there under its per-run lock before submitting its replacement; selection
  and report generation ignore the archive.
- Before accepting RQ6, verify the warmup and inverse-sqrt curves against the
  configured optimizer-step horizon in focused tests and raw LR traces. Classify
  every observed delta using the per-metric empirical band.
- Rewrite the reader README in RQ-number order. It contains only the required
  question tables, method descriptions/details,
  necessary analysis, and short conclusions. It omits `runs`, uses justified
  precision and colors, and fully enumerates both selected configurations.
- Move an RQ only to `review` after its raw evidence, generated reports,
  reader-facing answer, and independent review pass. Only the user may mark it
  `complete`.
- Remove or correct every stale completion claim in the protocol, detailed
  checklist, and agent status before moving any RQ to `review`.

## Verification

- The reviewer checks the manifest before training and independently checks
  raw logs, arithmetic, reuse decisions, generated tables, and claims after
  training.
- Do not run non-training tests for this remediation. Validate the training
  paths through the approved training program and raw artifacts.
- Generate every automated report twice and require an empty diff on the
  second generation. Trace every reader-facing value to run metadata and
  `final_metrics.json`.
- Mark each numbered item in `report_revision_checklist.md` complete only after
  its evidence exists and this independent verification passes.

## Approval

- Material assumptions: the treatment sets, grids, first boundary extension,
  RQ3 selection rule, and compute envelope above.
- Approved scope: manifest/report plumbing, the stated proxy program, required
  boundary and local retuning, baseline repeats, and one 500M confirmation per
  component treatment. RQ2/RQ3 and the final combined run are not yet approved.
- User approval: explicitly granted on 2026-08-14.
