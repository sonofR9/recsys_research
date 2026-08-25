# Report revision checklist

User review from 2026-08-14. Keep this file after every item is completed.

Protocol reset from 2026-08-15: every existing 50M result used a fixed endpoint
or target-count matching, and every existing 500M result used a fixed endpoint.
None is reusable as current evidence, including the old ten-run empirical band.
Invalid artifacts live in `generated/logs/old/`; reader and tuning reports omit
them. A new native early-stopped calibration and a new ten-repeat baseline band
now replace those two invalid evidence sets. In
addition to the numbered requirements below,
final verification must confirm that performance columns occur only in RQ3,
GELU is RQ4's absolute control, method paper links are present where
applicable, stale evidence files do not duplicate contradictory conclusions,
and reports contain only usable completed runs.

## Current remediation audit

Mark an item complete only after its required evidence exists and an
independent reviewer verifies the result against the stated requirement.

1. [x] Do not call the future baseline final while any selected component is
   supported only by an unresolved RQ7 or RQ11 result. Verify that every
   promoted component has an accepted 500M comparison.
   Verified: the reader report contains no final/future-baseline promotion;
   RQ2 and RQ3 remain explicitly pending approval.
2. [ ] Replace the obsolete RQ11 screen with the corrected objective families,
   including the homework-style fixed leave-one-out logQ method. Verify
   one control-selected batch shared across methods, family-specific LR and
   negative-count/secondary tuning, and one 500M result for each of the eight
   reader-table treatments.
   Current blocker: five canonical cap-40 predecessors for the homework logQ
   surface are queued after strict lineage review rejected gapped cap-80 chains.
3. [x] Retune every RQ7 treatment with the approved Cartesian μP proxy grid,
   extend boundary optima, and report learned-forward plus learned-backward.
   Verify all final comparisons use no positional encoding as the baseline.
   Verified: all 19 treatments use no encoding as control, include both learned
   directions, and have reviewed current-protocol proxy and final evidence.
4. [ ] Complete fair RQ8 tuning for dimension, depth, sequence length, heads,
   GQA, attention windows, normalization kind/place, BOS, FFN capacity, and
   dropout. Verify every dependence has its own 500M table and supported
   decision. Give CLS a separately specified causal objective or explicitly
   reject it as outside the existing next-item causal target.
   Current blocker: the exact MHA-4 and MHA-8 cap continuations are queued;
   every other axis has proxy and final evidence.
5. [x] Restore the full requested RQ9 method space: plain/log deltas,
   addition/concatenation, timestamp RoPE directions, bins, and useful
   combinations. Verify proxy exclusions and final selections are explicit.
   Verified: all 15 treatments have reviewed 50M tuning, exact 500M
   confirmations, generated tables, and dataset-size ranking analysis.
6. [ ] Add complete, fair tuning evidence for every method in RQ4-RQ10,
   especially RQ5, RQ6, RQ9, and RQ10. Verify the LR grid is extended whenever
   a winner is on a boundary and useful batch sizes are handled consistently.
   Current blocker: RQ8 awaits two head cap continuations; RQ4-RQ7 and RQ9-RQ10
   are in human review with current-protocol tuning and final evidence.
7. [x] Use and label the actual homework-compatible baseline, reproduce
   recall@100 in 0.1235-0.13, and reconfirm calibration after the selected batch-
   size change. Verify every percentage comparison names the correct control.
   Verified evidence: the native batch-1280 500M calibration restored best
   epoch 20, stopped at epoch 23, and reached recall@100 0.12736188.
8. [x] Run ten unchanged 500M baseline repeats and record each metric's sample
   standard deviation as a percentage of its mean. Verify the shared band is
   not described as a formal or treatment-specific confidence interval.
   Verified evidence: recall@100 mean 0.12762411, sample SD 0.00215019
   (1.685%); ndcg@100 mean 0.04837984, sample SD 0.00095122 (1.966%).
9. [x] Remove `runs` from reader-facing tables after item 8, and round every
   metric and percentage to precision justified by its shared baseline band.
   Verified: current reader and compact tables omit `runs` and use the shared
   band-compatible display precision.
10. [ ] Move RQ2 after the component RQs and enumerate every parameter for both
    the maximum-quality and quality/performance configurations. Verify neither
    configuration depends on an unresolved selection.
   Blocker: RQ2 now follows RQ11 and no longer promotes the legacy combination,
   but current maximum-quality and quality/performance configurations cannot be
   enumerated before every component and the combined recipe are accepted.
11. [ ] Define RQ3's balance criterion, report the relevant Pareto candidates
    including useful batch sizes, and avoid attributing a multi-factor change
    to batch size alone.
    Blocker: `analysis/collect.py` implements the one-band fastest-candidate
    rule and Pareto validation, but accepted current candidates and the
    reader-facing Pareto output are absent. Lead: obtain and accept the current
    candidates. Report owner: then render RQ3 and verify the choice does not
    isolate batch.
12. [ ] Make the generated tuning, compact 50M, and compact 500M reports
    complete for every RQ and method. Verify controls, absolute metrics,
    percentage changes, all usable completed proxy runs, and one bold winner
    per method are present where required. Failed, incomplete, invalid, and
    legacy runs must not appear.
   Current state: report generation is corrected; the 50M ledger must be
   regenerated after the active tuning wave completes.
13. [x] Correct RQ6's interpretation so every claim agrees with the shared
    empirical band, including any resolved harm from warmup.
   Verified: constant warmup is unresolved, cosine warmup is harmful, and
   inverse-square-root warmup helps but remains below constant.
14. [ ] Correct stale checklist, protocol, and agent-status claims. Verify no
    artifact says a run, review, or RQ is complete before its evidence exists.
   Current state: premature reader metrics and promotions are removed; final
   status verification remains.
15. [x] Color improvements beyond the applicable shared baseline spread green,
    declines beyond it red, and changes within it neutral. Verify the report
    calls this an approximate empirical resolution rule, not formal statistical
    significance.
   Verified: generated reader tables use per-metric absolute bands and describe
   them as an approximate empirical resolution rule, not significance.

## Common experiment rules

- [x] Use μP for proxy-to-target model-size transfer during hyperparameter
  tuning; document where μP cannot transfer a changed architecture or dataset.
  Verified: current proxy grids use the fixed-item-table μP family; reports
  explicitly do not claim transfer across architecture or dataset-size changes.
- [ ] When the best tested learning rate is a search-boundary value, extend the
  grid beyond that boundary before selecting it. Continue extending until the
  optimum is interior or a documented feasibility/stability limit is reached.
  Current blocker: the two RQ8 MHA continuations must resolve before the final
  architecture selection is closed.
- [x] Run ten identical full-data baseline repeats once, estimate shared metric
  spread in percentages, and reuse that noise estimate for single-run variants.
  Verified shared empirical bands: recall@100 1.685% and ndcg@100 1.966%.
- [x] Remove `runs` from reader-facing tables after the ten-run baseline spread
  is available.
- [x] Round metrics to a precision justified by the shared noise estimate.
- [x] Color improvements green and regressions red only when they exceed the
  shared empirical noise band; keep unresolved changes neutral.
  Verified in the current reader and compact report generators.
- [x] Name an axis column by the changed parameter, or omit a configuration
  column when the variant value already fully describes the comparison.
  Verified in the current reader and compact tables.
- [x] Keep unrelated fixed-recipe details out of axis-table configuration cells.
- [ ] Require a separate full proxy-tuning report for every method, with every
  usable completed run and hyperparameter setting shown and the best
  configuration bolded. Failed, incomplete, invalid, and legacy runs remain
  available only in raw audit storage.
  Current blocker: RQ8 awaits its final two head continuations; RQ2/RQ3 are not
  approved yet.
- [ ] Make the generated 50M and 500M reports README drafts: use the same
  RQ-specific result-table schemas and ordering, with only the title, research
  questions, and tables.
  Current blocker: RQ8 awaits final generation; RQ2/RQ3 are not approved yet.
- [x] Partially automate report tables and auxiliary artifacts from run
  metadata while keeping conclusions hand-written.

## RQ7 — position encoding

- [x] Add a treatment combining learned forward and learned reverse positions.
- [x] Verify and document whether ALiBi uses FlashAttention's built-in support.
- [x] Compare every treatment against no positional encoding, not learned
  forward positions.
  Verified in the reader, compact reports, and raw selection evidence.
- [x] Replace the inadequate three-rate proxy with μP-compatible tuning and
  extend any learning-rate grid whose winner lies on its boundary.
  Verified: every promoted RQ7 row has a closed treatment-specific LR surface.

## RQ8 — scaling and architecture

- [ ] Add and report maximum sequence lengths 256 and 512 under the current
  proxy-to-final protocol.
  Current state: both native-50M grids are queued; sequence-512 uses physical
  batch 640 with accumulation 2 for effective batch 1280.
- [x] Add a sequence-length distribution plot and report its median for the
  explicitly labeled training-eligible cohort.
- [ ] Add and report local-attention windows full/10/25/50/75/100 under the
  current proxy-to-final protocol.
  Current state: all non-control native-50M grids are queued; window 50 is the
  shared control.
- [ ] Add and report dropout 0/.05/.1/.2/.3/.5 under the current
  proxy-to-final protocol.
  Current state: all non-control native-50M grids are queued; dropout .1 is the
  shared control.
- [ ] Tune each new treatment fairly under the common proxy-to-final protocol.

## RQ11 — negative sampling

- [ ] Explain why tuned logQ underperforms uniform random negatives, or correct
  the premise with valid evidence.
  Current blocker: the homework-matched interpretation must be regenerated
  after the five strict-lineage repairs complete.
- [x] Expose and test the homework-style fixed-logQ correction instead of
  routing every logQ family through the Yi-2019 correction.
- [x] Resolve the correction-scale mismatch between offline in-batch and
  random-plus-offline proposal probabilities.
- [x] Check whether in-batch negative count is insufficient under the corrected
  objective and tuning protocol.
- [x] Check whether online/offline logQ learning-rate and batch grids stopped at
  a boundary or otherwise received inadequate tuning.
- [x] Split family learning-rate and secondary-axis proxy stages; require the
  selected family-specific rates for every secondary run and use collision-safe
  stage tags in run names.
- [x] Keep exact global-q arms unconditional, and name leave-one-out and mixed
  negative-only objectives separately instead of calling them exact corrections.
- [ ] Rerun any insufficiently tuned negative family before keeping the current
  conclusion.
  Current state: the eight main families are closed; five canonical cap-40
  homework-logQ predecessors are queued for strict selector/report validation.
