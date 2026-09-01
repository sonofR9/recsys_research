# RQ10 reinvestigation plan

## Question and hypothesis

- Research question: can layer-specific item features improve a four-layer
  SASRec when their fusion preserves the input/output-only control?
- Current evidence: the older matched native-500M four-seed comparison is flat
  at recall 0.13950 versus 0.13920. The current reader comparison, 0.135 versus
  0.117, is invalid because neither arm completed its declared linear-schedule
  horizon and the treatment learning rate was transferred from 50M.
- Falsifiable hypothesis: a baseline-containing concatenated residual or
  Gemma-style multiplicative mapping is non-inferior to the independently
  tuned four-layer control and most likely improves recall or NDCG.
- A plain random full-width addition is a reproduction treatment, not the
  expected winner.

## Comparison

- Control: four transformer layers with the tied input/output item embedding
  only. The implementation is unchanged; the reader-facing name replaces the
  ambiguous `shared table` name.
- Direct-add reproduction: before block `l`, add an independent full-width
  lookup `E_l[item]` to the residual stream.
- Concatenated residual: before block `l`, add
  `alpha_l * DenseNet([RMSNorm(h_l); RMSNorm(P_l(E_l[item]))])`, where
  `P_l` maps the fixed feature width to model width and `alpha_l` starts at
  zero. Keeping the finite-width projection separate gives μP the correct
  finite-to-infinite axis; DenseNet then receives exactly two scalable
  model-width inputs. The feature width is independently tuned at 16, 32,
  and 64.
- Gemma-style PLE: form a compact per-item/per-layer feature from a projection
  of the original item-token representation and an independent lookup, gate it
  multiplicatively with the post-FFN hidden state, project it back to model
  width, normalize it, and add it after block `l`. Its residual scale starts at
  zero so the seeded treatment initially equals the control. PLE width is
  independently tuned at 8, 16, and 32.
- Every treatment uses exactly four transformer layers. Width 64, SwiGLU
  intermediate width 192, two query heads, one KV head, attention window 50,
  learned-forward position, 16-bin timestamp input, standard item-state
  autoregression, random negatives, sequence length 128, no BOS/CLS, batch
  1280, and all other selected-G1 fields are held fixed.

## Data and evaluation

- All tuning and selection use native Yambda-500M, as approved by the user for
  G1. There is no 50M preselection.
- Validation runs every epoch. Primary metric is recall@100; NDCG@100 is the
  selection tie-break. Recall@10, NDCG@10, coverage@100, parameter count, and
  epoch time are reported.
- Non-inferiority requires recall loss no larger than 0.003 and NDCG loss no
  larger than 0.001 against the independently tuned four-layer control.

## Hyperparameter selection

- Every initial run uses embedding LR 0.064 for the tied and auxiliary
  embedding tables and independently searches deep LR at 0.006, 0.012, and
  0.024 per family.
- Concatenated feature width and PLE width are selected at deep LR 0.012, then
  the selected width is run at 0.006 and 0.024. Each stage moves one parameter.
- A learning-rate or feature-width boundary winner extends geometrically in
  that direction until the selected point is interior or reaches the hard
  valid width bound of one. These deterministic follow-ups close the approved
  single-axis stage; they do not introduce another jointly moving parameter.
- Every run uses `MuTransferGenerationExperiment`; the base and delta models
  include the same four-layer treatment structure and new projections.
- Every run completes the declared 20-epoch linear schedule without adaptive
  early stopping, restores the best validation checkpoint within the horizon,
  and records the best epoch.

## Run stages and compute

- Focused tests prove layer count, layer-specific lookup identity, query-token
  masking, output/gradient shapes, shared-parameter RNG identity, exact
  zero-start control equality, Gemma placement, and optimizer grouping.
- Initial native-500M budget is 16 runs: three control, three direct-add, five
  concatenated-residual, and five Gemma-style PLE runs.
- Runs are submitted as granular jobs to the persistent shared training queue,
  allowing all available GPUs to schedule them concurrently.
- If both baseline-containing treatments are materially worse, load the best
  control checkpoint, add the zero-start branch, train the branch alone, then
  optionally unfreeze. This distinguishes absent signal from joint-optimization
  failure and is diagnostic rather than selection evidence.

## Interpretation and reporting

- The corrected reader table contains only matched, horizon-complete
  native-500M evidence. The invalid 0.135/0.117 table is removed rather than
  retained as historical evidence.
- The valid older two-layer four-seed comparison remains explicitly labelled
  historical context and cannot select the four-layer treatment.
- A treatment is selected only if it is non-inferior. Material improvement is
  recall gain above 0.003, or NDCG gain above 0.001 while recall is
  non-inferior.
- No architectural mechanism claim is made for an unexpectedly degraded path
  without gate trajectories, branch/hidden RMS, gradients, training/validation
  curves, item-frequency strata, and exact initial control equality. A path can
  still be excluded by the pre-approved metric decision rule while its cause is
  reported as unresolved.

## Approval

- Dataset, four-layer depth, treatments, tuning surface, run budget, selection
  rule, and conditional diagnostic were approved in the conversation on
  2026-08-24.
