# G6 RQ1-RQ3: initialization and collision policy

## Question and hypothesis

- Research questions and status: RQ1 compares random and content-informed SID
  lookup initialization; RQ2 tunes RQ-KMeans with the deterministic collision
  suffix; RQ3 repeats the same search without collision resolution. Pending
  approved.
- RQ1 hypothesis: centroid initialization converges faster than random
  initialization without worse final retrieval quality.
- RQ2 hypothesis: an independently tuned tokenizer with the suffix is at least
  as good as the RQ0 tokenizer.
- RQ3 hypothesis: removing the suffix changes the best tokenizer and usually
  hurts concrete-item retrieval when collision buckets are large.
- RQ1 is an initialization diagnostic. RQ2 and RQ3 determine the terminal G6
  tokenizer and collision policy.

## Comparison

### RQ1

- Use the strongest RQ0 family that has a trainable base-code lookup:
  `item_learned_frozen_sid_event` at four levels, 512 codes at every level,
  learned width 32, and DenseNet width 32. Its RQ0 result was Recall@100
  0.127049 and NDCG@100 0.048725.
- Random initialization is the current truncated-normal lookup initialization.
- Content initialization first performs the identical global muP
  initialization, then overwrites only the four base-code row ranges. For each
  level, center its float64 codebook centroids, project them from 128 to 32
  dimensions using SVD/PCA with a deterministic sign convention, and rescale
  the projected block by one scalar to exactly match the corresponding random
  block's RMS. Special-token and collision-suffix rows remain byte-identical
  to the random arm. The overwrite does not advance RNG state.
- Both arms have identical architecture, parameter count, codebooks, suffix
  rule, seed, and optimizer grouping. No trainable projection is added.
- RQ1 does not replace the RQ0 winner and is not an aggregate component.

### RQ2 and RQ3

- Freeze the selected RQ0 representation: item ID plus frozen SID event,
  DenseNet width 128, on the best-G1 backbone.
- RQ2 appends the current deterministic within-base-tuple collision-rank
  suffix. RQ3 uses only the base RQ-KMeans tuple. Item IDs remain in history
  and the output is always a concrete item, so only the SID feature is
  ambiguous in RQ3.
- Search one common set of base-tokenizer and training coordinates. Every
  coordinate uses the same fitted codebooks, seed, data, learning rates, and
  model settings in the two policies; only the suffix policy changes. Each
  policy is selected independently from the paired surface.
- The comparison is between two complete tokenizer systems. Because the
  suffix also adds a learned input and parameters, no isolated causal claim
  about collision information will be made.
- Sanity controls are the immutable RQ0 winner at three levels, 512 codes,
  20 iterations, embedding LR 0.3620386719675124, and deep LR
  0.03463626154088337, plus the best-G1 item-ID-only control.

## Data and evaluation

- All RQ1-RQ3 evidence uses native Yambda-50M, validated by the user on
  2026-08-27. Use every eligible user and sample only by user ID.
- Keep the RQ0 likes, core-at-least-five, final-seven-day holdout, mapped-train
  catalog, full-catalog scoring, and no-seen-item-exclusion protocol.
- Keep physical batch 256, seed 42 for selection, the completed 15-epoch cosine
  horizon, and best-validation checkpoint restoration.
- Recall@100 is primary. NDCG@100, MRR@100, capped Recall at 10/50/100, and
  Coverage at 10/50/100 are secondary. Native-50M resolution bands are 0.002
  Recall and NDCG, 0.003 MRR, and 0.03 Coverage.
- Report base-SID exact Recall and prefix Recall at every depth after stripping
  the suffix. Also report ICR, collided-item fraction, unique-tuple count,
  bucket-size p50/p95/p99/max, target/history collision slices, per-level
  occupied p95 load and p95/mean, dead-code fraction, intra-code cosine,
  residual/reconstruction MSE, parameter and artifact sizes, fit and epoch
  time, peak GPU memory, embedding reads, MACs, and full-catalog latency.

## Hyperparameter selection

### RQ1

- Tune only embedding and deep learning rates on one paired 16-coordinate
  panel. Reuse the ten authenticated exact-architecture random-initialization
  cells from RQ0: trials 10-15 and their four embedding-LR boundary cells.
  Mirror all ten for content initialization, then add six manifest-fixed
  scrambled-Sobol log-uniform coordinates from embedding LR
  `[1e-4, 0.256]` and deep LR `[1e-4, 0.128]` to both arms.
- Select each arm by validation Recall@100. Values within 0.002 of the best
  Recall use NDCG@100, then stable manifest order; serving cost is identical.
- If either selected arm is at an LR boundary, add four outward points for
  that optimizer group with its other settings frozen and mirror those points
  into the other initialization arm. At most two such groups, or 16 physical
  runs, are approved. A further boundary winner returns for approval.

### RQ2 and RQ3

- Use one shared size at every residual level. Search levels
  `{2, 3, 4, 5}`, shared codes `{32, 64, 128, 256, 512, 1024, 2048, 4096,
  8192}`, RQ-KMeans iterations `{10, 20, 40}`, embedding LR log-uniform in
  `[0.008, 0.512]`, and deep LR log-uniform in `[0.002, 0.128]`. Fix
  RQ-KMeans and training seed 42 and representation width 128.
- Reject a coordinate before training if any level, including the suffix
  vocabulary, exceeds 8192 symbols.
- Use 40 manifest-deterministic paired coordinates: four anchors and 36
  scrambled-Sobol coordinates. The anchors are the exact RQ0 coordinate and
  `(2, 64, 20)`, `(4, 1024, 20)`, and `(5, 4096, 40)` at the RQ0 learning
  rates.
- Select each policy independently by validation Recall@100. Values within
  0.002 of the best Recall use NDCG@100, then measured latency, then stable
  manifest order. Intrinsic SID metrics never select a tokenizer.
- If either selected policy lies in the outer 10% of an LR interval, add four
  outward points at its frozen tokenizer and other LR and mirror each point
  into the other policy. At most eight paired coordinates, or 16 physical
  runs, are approved. More triggered edges return for approval. Discrete axes
  are bounded research domains; a winner at their cap is reported as
  boundary-selected rather than silently extended.

## Run stages and compute

- Focused checks cover deterministic projection and sign handling, post-muP
  overwrite order, RNG non-advancement, equality of all non-base-code
  parameters and lookup rows, suffix on/off serialization, paired codebook
  hashes, symbol caps, collision mapping, query alignment, and metrics.
- RQ1 has 32 logical surface cells. Ten random cells are reused, so 22 are new.
  After selection, rerun both frozen settings at seeds 43-45: six new cells.
  Expected new budget is 28; maximum is 44 with mirrored LR boundaries.
- RQ2/RQ3 have 80 logical surface cells. The exact RQ0 suffix-on anchor is
  reused, so 79 are new. Rerun each selected policy and the RQ0 setting at
  seeds 43 and 44, reusing overlaps: at most six more cells. Expected new
  budget is at most 85; maximum is 101 with paired LR boundaries.
- Freeze the seed-42 selections before confirmation. Use seed 42-44 means for
  final RQ2/RQ3 comparisons without selecting again on the repeats.
- Submit all multi-run stages to the existing persistent queue. Fit each base
  tokenizer once and reuse it across the paired suffix policies only after its
  full cache identity matches.
- RQ2 must retain the RQ0 tokenizer if its selected suffix-on system worsens
  Recall by more than 0.002. Then compare the selected RQ2 and RQ3 systems by
  the same Recall-band/NDCG rule. A changed terminal tokenizer or collision
  policy is promoted only if it improves Recall over RQ0 by more than 0.002
  without an NDCG regression greater than 0.002; otherwise retain the RQ0
  terminal bundle.
- RQ1 is incompatible with the aggregate because it studies a different
  representation. RQ2 and RQ3 are mutually exclusive; the preceding terminal
  rule resolves them.
- The native-50M original G1 baseline and best-G1 item-only artifacts are
  frozen. If the terminal bundle is unchanged, reuse the existing RQ0
  original-backbone bridge. If it changes, retune the fixed terminal bundle on
  the original backbone with 12 LR trials and at most eight LR-boundary cells.
  The terminal best-G1 artifact is already the G6 aggregate and is not rerun.

## Interpretation and reporting

- RQ1 final quality uses the four-seed mean for each selected initialization.
  Content initialization is non-inferior only when mean Recall and NDCG are
  each no worse by more than 0.002. Convergence uses all 15 validation points:
  report normalized Recall AUC and the first epoch reaching 95% of each run's
  own best. Call content faster only when final quality is non-inferior and the
  paired direction agrees at all four seeds; otherwise call speed descriptive
  or unresolved.
- For unexpected RQ1 behavior, verify gradients, initialization hashes and
  scale, projection reconstruction, duplicated frozen-centroid information,
  and LR-dependent warm-start erasure. For unexpected RQ2/RQ3 behavior, verify
  paired identities, suffix capacity, bucket and frequency slices, utilization,
  convergence, and selected LR boundaries with saved tests or targeted runs.
- Give RQ1, RQ2, and RQ3 separate reader tables. In every quality table put
  `Method`, `Recall@100`, and `Delta Recall@100` next to each other. Put SID,
  collision, and efficiency diagnostics in separate tables; keep tuning rows
  under `evidence/`.
- The closing aggregate compares the frozen original G1 baseline with the
  terminal best-G1-plus-SID bundle. Arithmetic uses the matched G1 best-bundle
  gain plus the terminal SID marginal and never adds RQ1 or both collision
  policies.

## Acceptance criteria

- Codebook initialization should converge faster than random initialization,
  and faster convergence counts only when final recall is non-inferior.
- The selected collision-token setup must not worsen downstream Recall@100
  versus the RQ0 setting.
- Compare independently tuned systems with and without collision resolution.

## Approval

- Material choices are the deterministic PCA initialization, 16 paired RQ1 LR
  coordinates, 40 paired RQ2/RQ3 coordinates, shared per-level code size, the
  terminal promotion rule, confirmation seeds, and conditional bridge.
- Exact scope requested for approval: treatments, native-50M protocol, search
  domains, up to 113 expected new RQ1-RQ3 runs, 145 maximum with LR boundaries,
  and up to 12 expected/20 maximum conditional original-backbone bridge runs.
- User approval: validated in conversation before implementation.
