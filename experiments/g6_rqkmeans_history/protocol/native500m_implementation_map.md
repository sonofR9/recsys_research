# G6 native-500M implementation map

The approved protocol is `native500m_rerun_plan.md`. Implementation is
authorized; queue submission remains gated on focused tests and blind review.

## Reuse boundary

- Reuse the seven existing semantic-history model implementations from
  `dcn/models/history_tokens.py` and the current local G6 builders. No new model
  representation is required.
- Reuse the validated initialization semantics from `Rq1InitializationExperiment`,
  but not its representation-specific class or builder. Add a task-local generic
  initializer that resolves exactly one trainable base-SID lookup in every
  applicable RQ0 representation: trainable SID event, item plus
  trainable/frozen SID event, trainable SID tokens, trainable/frozen SID tokens,
  and interleaved item/SID tokens. Frozen-only families remain explicitly
  inapplicable to RQ1.
- Preserve the reviewed float64 SVD projection and canonical sign rule at the
  fixed width 128. Every allowed codebook has at least 512 centroids. Preserve
  special/suffix rows and RNG state exactly.
- Reuse `CollisionPolicyExperiment`, collision-symbol validation, semantic fit
  diagnostics, and convergence validation. Add one native500M composite
  experiment that combines the generic lookup initializer with collision-cap
  validation for RQ2/RQ3.
- Reuse the native-500M G1 source loaders only as authenticated configuration
  anchors. Every selection-eligible control and treatment is rebuilt at batch
  512 under a new G6 manifest.
- Keep the original-G1 control and selected SID bridge on the conventional
  original backbone with constant LR for 26 epochs. Keep best-G1 and its SID
  treatments on the MuTransfer aggregate backbone with schedules annealed over
  26 epochs.
- Preserve the current native-50M modules, manifests, reports, and raw evidence
  unchanged as historical audit material.

## New package boundary

Add `experiments/g6_rqkmeans_history/native500m/` with these responsibilities:

- `protocol/design.py`: ordered domains, exact anchors, Sobol mapping, LR
  coordinates, sequential inherited anchors, boundary factors, and run-count
  accounting.
- `protocol/contracts.py`: immutable stage/job dataclasses, canonical JSON,
  logical and physical SHA-256 identities, source-selection hashes, and exact
  reuse declarations. Every manifest binds the exact approved-plan SHA-256 and
  an approval artifact that records the user-approved expected/max budget and
  fixed-tolerance exception. That artifact does not exist while approval is
  pending.
- `protocol/selection.py`: per-reference bands, seed-mean confirmation
  decisions, two-baseline promotion, no-SID fallback, and atomic terminal-bundle
  decisions.
- `configs/runtime.py`: batch-512 native-500M controls plus all semantic
  treatments, using one dataset-explicit builder and convergent RQ-KMeans
  semantics.
- `launchers/runtime.py`: authenticated environment contracts and cache-safe
  waves grouped by dataset and fully authenticated semantic-cache identity.
- `launchers/queue_*.sh`: versioned JSON job specifications submitted with
  `find-batch` followed by `submit-batch` to the existing persistent queue.
  Exact retry recovers the already committed batch, and no ordinary `enqueue`
  loop, direct multi-run path, or second queue service is allowed.
- `analysis/collect.py`: strict metadata/artifact verification, incremental
  tuning ledgers, final selections, confirmation summaries, and aggregate
  decomposition. It recomputes every selection/report metric from ranking
  evidence and verifies agreement with saved final metrics.
- `analysis/topk_evidence.py`: immutable context-bound top-100 concrete-item
  rankings keyed by evaluation query/target. Bind dataset, split, ordered
  catalog, checkpoint, evaluator configuration, stage/job/manifest identity,
  and file SHA-256. This supplies the recommended item IDs absent from the
  compact rank-only evidence and permits independent Coverage and SID-metric
  recomputation.
- `analysis/report.py`: generated RQ0-RQ3 tables and aggregate section with the
  original baseline first, separate best-G1 local tables, and adjacent
  `Method`, `Recall@100`, and `Delta Recall@100` columns. It writes
  `evidence/rq0_reader_native500m.md`,
  `evidence/rq0_tuning_native500m.md`,
  `evidence/rq1_rq3_reader_native500m.md`, and
  `evidence/rq1_rq3_tuning_native500m.md` before composing the active README.

## Dependency stages

1. Materialize and review the native-500M original-G1 and best-G1 control
   manifests at the fixed 26-epoch horizon before any treatment manifest is
   materialized.
2. Materialize the first RQ0 surface, freeze its selected tokenizer/LR anchor,
   then materialize the six smaller inherited-anchor surfaces. Resolve
   boundaries and make a provisional best-G1-local selection, then
   materialize its original-G1 bridge and freeze the final two-baseline RQ0
   promotion document.
3. Materialize the paired RQ1 surface only from the authenticated, resolved RQ0
   promotion document.
   Resolve boundaries and paired seed confirmations.
4. Materialize paired RQ2/RQ3 surfaces from the authenticated RQ0/RQ1 bundle.
   Resolve boundaries and paired seed confirmations, then freeze a provisional
   `diagnostic_sid_bundle` for scientific reporting.
5. Materialize and authenticate a conditional terminal bridge when the
   provisional SID bundle differs from the RQ0 bridge. Only then freeze
   `aggregate_bundle`, which is best-G1 without SID when final promotion fails.
6. Freeze the aggregate manifest, collect ranking evidence and diagnostics,
   then regenerate the active G6 report from authenticated evidence only.

Every stage manifest binds its predecessor document SHA-256 and refuses to
materialize until the predecessor is selection-resolved. Boundary manifests
are finite and separate from initial surfaces. A second boundary win, K-Means
cap hit, or reopened correctness investigation
stops materialization and returns for approval.

Every semantic-cache contract additionally binds dataset identity, ordered
fitted item-ID/catalog SHA-256, normalized content-embedding SHA-256, fitter
revision and full configuration, plus the produced codes, codebooks,
fit-diagnostics, and materialization-marker hashes. Parameter equality alone
never authorizes cache reuse, and collision policies may share only a base fit
whose complete input/output identity matches.

Every accepted run binds a successful queue job and returned batch identity to
its stage manifest and verifies checkpoint and best/restored epoch consistency.
The collector recomputes Recall/NDCG/MRR from compact rank evidence and
Coverage plus exact/prefix SID and collision metrics from the authenticated
top-100 item-ranking artifact. It rejects disagreement with
`final_metrics.json` outside the evaluator's declared numeric tolerance and
writes immutable, content-hashed evidence documents.

## Focused test matrix

- `test_native500m_design.py`: exact anchor order, inherited-anchor handoff,
  Sobol coordinates, restricted discrete domains, width 128, code cap, stage
  counts, all feasible expected totals, and the 262-run maximum.
- `test_native500m_contracts.py`: canonical hashes, source authentication,
  dataset isolation, batch 512, cross-stage dependency rejection, exact reuse,
  and round-trip environment contracts.
- `test_native500m_configs.py`: native size, no user sample, final-seven-day
  split, mapped full catalog, no seen-item exclusion, schedule completion,
  representation parameters, initialization behavior for every applicable
  RQ0 family at width 128 and all three code counts, composite initialized
  collision policies, and convergent K-Means configuration.
- `test_native500m_selection.py`: comparison-local bands, Recall/NDCG ordering,
  two-baseline SID promotion, RQ1 four-seed means, RQ2/RQ3 three-seed means,
  nonnegative RQ2 rule, separate diagnostic/aggregate bundles, no duplicate
  aggregate run after failed SID promotion, and boundary triggers. Test RQ1's
  convergence precedence with epoch-to-95% and normalized-AUC agreement and
  disagreement cases using the existing learning-curve computation; content is
  faster only when epoch-to-95% is lower and normalized AUC is higher, and a
  tie or disagreement retains random when quality is within band.
- `test_native500m_evidence.py`: run-contract equality, checkpoint restoration,
  complete 26-epoch horizons, cache hashes,
  convergence diagnostics, queue batch/job success, metric recomputation from
  rank and top-100 item evidence, saved-metric agreement, top-K context binding,
  and immutable evidence hashes.
- `test_native500m_report.py`: consecutive RQ0-RQ3 headings, original baseline
  first, separate local controls, adjacent Recall columns, 500M RQ0-RQ3 tables,
  aggregate arithmetic, and omitted efficiency columns.
- `test_native500m_queue.py`: versioned JSON specifications, manifest-bound
  batch membership, `find-batch`/`submit-batch` idempotent retry, committed-batch
  recovery, and rejection of partial or foreign batches.
- Retain the existing RQ1 initialization, collision, semantic-embedding, and
  generation-protocol tests as regression coverage.

## Launch gate

Before any manifest materialization, require the exact plan SHA and explicit
approval artifact. Before queue submission, the implementation diff receives a
blind review and focused tests pass. The complete non-training suite runs once
after that review when shared CPU load is low. Each launcher first emits and
verifies the versioned JSON specification without submission, resolves an exact
prior batch with `find-batch`, otherwise uses `submit-batch`, and records the
returned batch identity plus stage-manifest SHA under the experiment scratchpad.
