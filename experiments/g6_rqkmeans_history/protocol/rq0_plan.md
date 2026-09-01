# G6 RQ0: RQ-KMeans semantic IDs in the history

## Question and hypothesis

- Research question and status: which of the seven approved semantic-history representations is best for next-item retrieval; pending approval.
- Current understanding: RQ-KMeans is fitted to the mapped catalog's normalized content embeddings. The target and retrieval catalog remain concrete item IDs.
- Falsifiable hypothesis: at least one semantic-history representation improves Recall@100 beyond the size-matched noise band of the best G1-combination item-ID control without a beyond-band NDCG@100 regression.
- Why the result matters: RQ0 selects the history representation inherited by G6 RQ1-RQ3.
- Additional diagnostic question: are gains concentrated on tail targets or histories containing base-SID collisions? This reuses the selected runs and adds no training.

## Comparison

- Primary unchanged control: the best combination at the end of G1, reconstructed at native 50M and recalibrated there: 4 layers, GQA 2Q/1KV, SwiGLU width 192, gated dropout 0.1, deep-only cosine with 5% warmup and horizon 15, ALiBi plus learned forward/reverse positions, post-LN with input/final RMSNorm, end-only CLS, 32-bin additive time plus raw reverse-time RoPE, global-popularity catalog negatives (2048) with the full Yi et al. correction, and BOS.
- Secondary unchanged control: the original G1 item-ID baseline, reconstructed and recalibrated at native 50M: 2 layers, MHA 2Q/2KV, GELU width 256, pre-LayerNorm, learned forward positions, no time/CLS/BOS, and fixed leave-one-out logQ negatives (512).
- Primary treatments on the best-combination backbone:
  1. One event token from trainable SID embeddings, level tags, and a DenseNet projection.
  2. One event token from item ID plus frozen RQ-KMeans centroids and a DenseNet projection.
  3. One event token from item ID plus trainable SID embeddings plus frozen centroids and a DenseNet projection.
  4. One trainable SID token per level.
  5. One token per level from trainable SID embeddings plus frozen centroids and a DenseNet projection.
  6. One frozen centroid token per level.
  7. Interleaved learned item-ID and trainable SID tokens for every history item.
- Secondary bridge: only the primary-surface winning semantic representation is reconstructed on the original G1 baseline. The seven-by-two full crossing is out of scope.
- Factors held fixed: native 50M data, mapped catalog, split, histories, targets, model width 64, optimizer family, validation protocol, output item table, negative sampler within each backbone, and a single physical batch size that fits every treatment.
- Every RQ-KMeans residual level uses the same codebook size. There is no per-level size parameter.
- Every concatenated event or token representation uses DenseNet and emits width 64.
- All arms use the same last 100 history items. Expanded representations truncate by item count, not token count. The final token of an item's event is the query state for its next-item target.
- Collision policy: append a deterministic collision-rank suffix to every base SID. Because no centroid exists for this suffix, it has a separately trainable embedding even in otherwise frozen-centroid treatments. RQ2/RQ3 may later replace or remove this policy.

## Data and evaluation

- Single dataset size proposed for all G6 questions, tuning, and final evidence: native Yambda-50M.
- User validation reference for the dataset size: user approved the plan on 2026-08-27.
- Sampling unit and sample size: all eligible users; no positional sampling.
- Event filters, split, and catalog: likes, core at least 5, seven-day temporal holdout, mapped items only, full mapped-item catalog, and no seen-item exclusion.
- Primary metric: Recall@100. Secondary retrieval metrics: NDCG@100, MRR@100, capped Recall@100, and catalog coverage at 10/50/100.
- SID metrics: exact and prefix SID Recall at 10/50/100 after mapping concrete recommendations to base SID tuples; identifier collision rate `1 - unique base tuples / mapped items`; fraction of items in non-singleton collision buckets; per-level raw p95 occupied-bucket load and p95/mean load; and mean within-code cosine similarity of normalized content vectors.
- Slice diagnostic: item metrics by train-frequency tercile and by whether the history contains a collided base tuple.
- Shared empirical noise bands: ten repeats of the tuned primary item-ID control, seeds 42-51, with per-metric sample standard deviation rounded upward to one significant digit. No treatment-specific confirmation seeds.

## Hyperparameter selection

- RQ-KMeans input: `normalized_embed`; fixed seed 42 and fixed fitting implementation/settings for RQ0.
- Shared tokenizer search for every semantic treatment: number of residual levels in `{2, 3, 4}` and one shared per-level codebook size in `{32, 64, 128, 256, 512}`.
- Method-specific representation width: `{32, 64, 128}` for trainable SID/tag embeddings or DenseNet hidden width. The transformer and item embedding widths stay 64.
- Trainable lookup tables use the embedding optimizer group; DenseNet, adapters, and transformer weights use the deep group.
- Primary-control calibration: Optuna TPE, seed 42, over feasible common physical batch sizes from `{128, 256, 512, 1024, 1280}`, embedding LR log-uniform in `[1e-4, 0.256]`, and deep LR log-uniform in `[1e-4, 0.128]`.
- Original-control calibration: fixed selected common batch and independent embedding/deep LR tuning over the same ranges.
- Each semantic treatment is tuned independently with Optuna TPE over levels, shared codebook size, representation width, embedding LR, and deep LR. Enqueued tokenizer anchors are `(levels, codes) = (2, 256), (3, 64), (3, 128), (4, 32)`.
- The original-backbone bridge freezes the winning tokenizer and representation definition, then retunes its embedding/deep LRs.
- Boundary rule: if a selected LR lies in the outer 10% of its log interval, add four trials beyond that boundary with all non-LR settings frozen. One such extension is allowed per LR; another boundary win returns for approval.
- The best-combination backbone completes its declared 15-epoch schedule horizon and restores the best validation epoch. The original backbone uses early stopping with patience 3 and a generous 40-epoch cap; a non-triggered stop extends and reruns.

## Run stages and compute

- Focused checks: tokenizer determinism and cache identity; collision suffix uniqueness; token ordering and item-count truncation; event/query alignment; target leakage; DenseNet shape/parameter grouping; exact/prefix metrics; and one tiny smoke run per serialization family.
- Memory preflight uses the largest initial expanded sequence. Only batch sizes fitting every arm are eligible.
- Initial full-run budget: 20 primary-control Optuna trials, 12 original-control trials, 9 additional primary-control repeats, 16 trials for each of seven treatments, and 12 original-backbone bridge trials: 165 runs.
- Conditional LR-boundary budget: at most 8 extra trials for each of ten tuned surfaces: 80 runs. Maximum approved RQ0 budget: 245 full runs.
- Full runs use the persistent shared training queue and no GPU exclusion. Tokenizer artifacts are fitted on demand and cached by their full configuration.
- Frozen original baseline for later G6 aggregation: the selected native-50M original G1 control from this plan.
- Dependency graph: primary control calibration -> empirical bands and seven treatment sweeps -> primary winner -> original-backbone bridge. RQ1-RQ3 inherit the RQ0 winner; mutually exclusive representation variants never combine.
- The G6 closing aggregate will use the best-G1-combination backbone plus the terminal selected G6 bundle. RQ0 does not launch that aggregate.

## Interpretation and reporting

- Select the highest validation Recall@100 treatment; ties within the Recall band break by NDCG@100, then lower serving cost. The item-ID control remains the selected method if no semantic treatment improves beyond the Recall band.
- Promotion requires Recall@100 above the primary control by more than its band and no NDCG@100 regression beyond its band.
- Unexpected regressions require checks of target/query alignment, truncation parity, collision slices, code utilization, parameter counts, negative sampling, convergence, and selected learning-rate boundaries before interpretation.
- Reuse the selected seed-42 artifacts as final cells; only the primary control receives repeats for bands.
- Reader-facing tables are separate by baseline. In each quality table, `Method`, `Recall@100`, and `Delta Recall@100` are adjacent, followed by other quality metrics. SID diagnostics and efficiency appear in separate tables. Tuning parameters and run counts do not appear in reader tables.
- The full tuning table and compact selected-artifact manifest live under `evidence/`; the top-level report links them.
- Aggregate arithmetic later uses matched baseline-to-treatment bridges only. It never sums overlapping tokenizer, initialization, or collision-policy deltas.

## Acceptance criteria

- The final model must not be much worse than SASRec. Ideally it should be better.

## Approval

- Material assumptions: native 50M; tune all seven representations only on the best G1-combination surface; bridge only the selected winner to the original G1 baseline; and use a trainable collision-rank suffix component even for frozen-centroid representations.
- Exact scope requested for approval: comparison, tuning ranges, 165 initial and 245 maximum full-run budget, metrics, selection rule, and the additional no-run slice diagnostic above.
- User approval: approved on 2026-08-27 after confirming that all seven
  representations use the best G1-combination backbone and only the selected
  winner is bridged to the original G1 baseline.

## Approved learned-SID remediation

- Motivation: the original comparison between `item_frozen_sid_event` and
  `item_learned_frozen_sid_event` changed the SID depth and DenseNet width as
  well as adding learned SID embeddings, so it cannot determine the learned
  embeddings' incremental effect.
- Control reuse: the selected native-50M `item_frozen_sid_event` artifact on
  the best-G1 backbone: batch 256, 3 residual levels, 512 codes at every level,
  DenseNet width 128, Recall@100 0.13018.
- Corrected treatment: construct the identical frozen-SID event token and add
  the learned per-level SID embeddings through the local
  `ConcatenatedItemFeatureResidual`. Its scalar residual gate starts at zero,
  so the treatment initially equals the frozen-only representation.
- Fixed factors: best-G1 backbone, native Yambda-50M, batch 256, 3 residual
  levels, 512 codes at every level, frozen-event DenseNet width 128, collision
  suffix policy, sampler, schedule, validation, and all other model factors.
- Tuned factors: learned SID width in `{32, 64, 128}`, embedding learning rate,
  and deep learning rate. Use 16 Optuna trials and at most 8 approved LR
  boundary runs.
- Promotion: Recall@100 must exceed the reused control by more than 0.002 with
  no NDCG@100 regression greater than 0.002. If promoted, run a 12-trial
  original-G1 bridge plus at most 8 LR boundary runs.
- Run budget: 24 new native-50M runs unless promoted; 44 maximum if promoted.
  Raw artifacts from the original comparison remain immutable.
- Acceptance: `concat(embed(item_id), frozen_sid, embed(sid_i))` must be better
  than or not worse than `concat(embed(item_id), frozen_sid)` under this
  controlled comparison.
- User approval: approved on 2026-08-27.
- Execution binding: v3 carries the two completed v2 trials into the same
  16-trial surface and schedules trials 2-15 under new identities. Trials 2-4
  are the three width values at the frozen control's selected learning rates;
  the remaining eleven are adaptive TPE points. The two v2 artifact contracts,
  all three anchors, and a restart-stable per-trial TPE seed rule are part of
  the v3 manifest hash. This preserves every raw artifact and the approved
  physical-run budget after the v2 launcher was found to repeat suggestions
  when reconstructed.
- Bounded-gate follow-up: approved on 2026-08-27 after v3 showed that the
  zero-initialized scalar inherits deep LR 0.034636 and Adam moves it by about
  0.0346 on its first update. Keep the selected width 32, embedding LR 0.256,
  deep LR 0.034636, and every other v3 factor fixed. Grid-search only the local
  residual bound in `{0, 0.01, 0.025, 0.05, 0.1}` using
  `bound * tanh(raw_scale)`. Bound zero is the frozen-path parity diagnostic;
  positive bounds are the learned-SID candidates. Run exactly five additional
  native-50M cells, with no automatic extension.
