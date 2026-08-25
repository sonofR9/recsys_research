# RQ5 scheduler remediation plan

## Question and decision

- Research question: Which learning-rate scheduler and optimizer-group scope
  is best once each scheduler's planned step count is calibrated to its own
  observed stopping point?
- Primary falsifiable hypothesis: for both inverse square root and one-cycle
  cosine annealing with tuned warmup, at least one of the two optimizer-group
  scopes satisfies `delta recall@100 >= -0.003` and
  `delta NDCG@100 >= -0.001` against constant learning rate.
- Secondary hypothesis: keeping the embedding LR constant while scheduling
  only the deep LR improves final ranking quality relative to applying the
  same schedule multiplier to both optimizer groups.
- Decision: replace every invalid RQ5 row and decide separately whether each
  priority scheduler is not materially worse than constant.
- Overall leader among the tested candidates is the usable treatment with
  highest recall@100. Treatments
  no more than 0.003 lower in recall@100 and 0.001 lower in NDCG@100 are
  co-leaders. If constant is a co-leader, it remains the recommended default;
  otherwise choose the co-leader with highest recall@100, then NDCG@100.
- Within a scheduler, `deep_only` is better only when it gains more than 0.003
  recall@100 and is not more than 0.001 worse in NDCG@100. The symmetric rule
  applies to `both`; smaller differences leave optimizer-group scope unresolved.

## Controlled comparison

- Scheduler treatments: constant, linear, cosine, polynomial, exponential,
  step, WSD, inverse square root, fixed-5%-warmup cosine with 1, 2, or 4
  cycles, and a separately warmup-tuned one-cycle cosine treatment.
- Every non-constant scheduler is crossed with two explicit scopes:
  `both`, where its multiplier is applied to the embedding and deep optimizer
  groups, and `deep_only`, where embedding LR remains constant at 0.064 and
  only the deep group follows the curve. Constant has one scope because its
  two scope labels would be mathematically identical. This gives 23 treatments.
- All 23 treatments are retuned. Historic rows are not reused because their
  horizon handling, embedding LR, or optimizer-group scope does not match this
  comparison.
- Every treatment keeps the selected architecture, data protocol, objective,
  batch 1280, evaluation catalog, and recall@100 selection metric.
- Linear receives its own schedule variant and tuning surface; it is no longer
  an alias of the architecture control.
- Scope comparisons are made only within the same scheduler. Both scopes use
  the same candidate values and equal budgets, but select and freeze winners
  independently; a winner is never reused across scopes.
- The historic constant result remains a diagnostic. The comparison control is
  the newly tuned constant arm under the same embedding rate, target regime,
  and seed as every treatment.

## Data and evaluation

- All tuning and reported evidence uses full native Yambda-500M with seed 42.
  There is no Yambda-50M stage and no separate seed-43 confirmation stage.
- Every completed candidate remains in the tuning ledger. The best usable
  seed-42 candidate for each treatment enters the reader table directly.
- All runs use likes, core items with at least five interactions, the final
  seven-day timestamp holdout, mapped training items, full mapped-catalog
  scoring, and no seen-item exclusion.
- Recall@100 is primary. NDCG@100, recall@10, NDCG@10, and coverage@100 are
  secondary. Validation runs every epoch and the reported checkpoint is the
  best recall@100 epoch.
- Validation loss is audited as a convergence diagnostic. Recall@100 remains
  the selection and patience metric; loss is not substituted for the ranking
  objective.

## Tuning

- Global batch stays 1280. Every RQ5 run uses exactly one embedding LR: 0.064.
  It is never tuned or changed between schedulers or scopes.
- Every treatment receives three initial candidate slots. Constant and each
  one-parameter treatment use deep LR 0.003, 0.006, and 0.012. The fixed-5%
  warmup rows isolate cycle count within each scope.
- A lower-bound winner extends to 0.0015, 0.00075, and 0.000375. An upper-bound
  winner extends to 0.024, 0.048, and 0.096. This is the single conditional
  boundary round described below.
- Within each scope, inverse square root jointly tunes deep LR and timescale
  fraction with three initial candidate pairs: the central point plus two
  reproducible log-uniform random draws, seed 42, over `[0.0015, 0.024]` and
  `[0.0125, 0.20]`. The absolute timescale is derived from that run's
  calibrated reference horizon and persisted as steps.
- Within each scope, the separate warmup-tuned one-cycle cosine treatment
  jointly tunes deep LR and warmup fraction with the same three-candidate
  design and bounds.
- The central pair is `(0.006, 0.05)`, and both scopes use the same seeded
  candidate pairs. If a non-central pair wins, add three seed-42 log-uniform
  local candidates from the original bounds clipped to a factor-two box around
  that pair before selection. A winner is boundary-adjacent
  when either parameter lies in the outer 25% of that parameter's log search
  interval. For each affected side, hold the other parameter at its winning
  value and add three log-spaced probes in the adjacent interval produced by
  extending that bound fourfold. Those three probes are strictly new and do
  not repeat the old boundary.
- Per the three-candidate default, a central winner does not trigger extra
  local trials. The joint surfaces are bounded screens rather than exhaustive
  two-dimensional optimization, and the report states this limit.
- Boundary extension is limited to one three-probe round per affected
  parameter. A treatment that still wins on the new boundary is unresolved
  and returns for approval instead of being reported as selected.
- The fixed-5%-warmup one-cycle candidate at deep LR 0.006 is the same model as
  the joint search's central `(0.006, 0.05)` candidate. Each scope trains that
  configuration once and reuses its artifact in both tuning surfaces.
- Rank usable candidates by best-epoch validation recall@100, then same-epoch
  validation NDCG@100, then stable surface run name. Because the existing
  validation logs persist four decimal places, normalize both ranking metrics
  to four decimals before comparison. Freeze that exact seed-42 artifact and
  candidate-set digest for the report.

## Frozen schedule definitions

- Linear decays from 1 to 0 over its calibrated horizon.
- Cosine uses one half-cosine from 1 to 0; restart rows use 1, 2, or 4 complete
  half-cosine cycles after their warmup.
- Polynomial uses quadratic decay from 1 to 0.
- Exponential decays from 1 to 0.01.
- Inverse square root uses `sqrt(T / (T + step))`, where `T` is its calibrated
  absolute timescale.
- Step stays at 1 through 50% of the horizon, then uses 0.1 through 75%, then
  0.01 through the end.
- WSD warms up for 5%, stays flat through 80% of the post-warmup decay phase,
  then cosine-decays to 0 over its final 20%.
- Fixed-warmup cosine uses 5%; the tuned one-cycle treatment searches the
  declared warmup fraction. All unspecified minimum-LR fractions are 0.

## Training horizons

- This section is an explicit RQ5-only exception to the current G1 rule that
  annealed schedules must always finish their declared horizon. Approval of
  this plan authorizes patience stopping plus from-scratch horizon calibration
  for RQ5; other experiments retain the existing rule.
- Validation and recall@100 early stopping with patience three remain enabled
  for every scheduler. Different schedulers, scopes, and LR candidates may use
  different calibrated horizons.
- Historical native-500M stopping epochs are initial horizon estimates only:
  linear 17, cosine 21, polynomial 20, exponential 18, step 20, WSD 20,
  inverse square root 23, and fixed-warmup cosine 19/22/22 epochs for 1/2/4
  cycles. The tuned-warmup one-cycle estimate starts at 19. `deep_only` starts
  from the corresponding `both` estimate.
- For an annealed run with declared horizon `H`, the epoch cap is `H`. If
  patience stops at epoch `E` and `H - E` is at most
  `max(3, round(0.1 * H))`, the horizon is close enough to the actual end. If
  it stops earlier, rerun from scratch with `H = E`. If it reaches `H` before
  patience fires, rerun from scratch with `H = ceil(1.5 * H)`.
- Inverse square root uses the same calibration rule for the reference horizon
  that determines its absolute timescale, while retaining a generous safety
  cap of `max(80, 2 * H)`. If its actual stop differs from `H` beyond the same
  tolerance, derive a new timescale from the observed stop and rerun from
  scratch. Constant has no schedule horizon and uses an 80-epoch safety cap.
- A candidate gets at most two heuristic corrective horizon reruns. The 28
  initial surfaces explicitly named in `rq5_scheduler_approval.json` may then
  receive at most two additional, stagewise bracket-midpoint reruns. Every
  other unresolved surface returns for approval and cannot enter selection or
  reports.
- Eleven explicitly named initial surfaces may receive one final `a5` run at
  the approved untried lower midpoint in their remaining bracket; the
  canonical initial-run-to-horizon mapping is fixed in
  `rq5_scheduler_approval.json`. This does not change the acceptance tolerance
  or authorize an `a6`. Thirteen fully exhausted surfaces are retained as
  auditable evidence and listed there as approved ineligible exclusions. They
  are not calibrated and cannot enter winner selection or reader-facing result
  tables.
- Hitting the constant or inverse-square-root safety cap without patience
  firing extends the cap by 50% and reruns from scratch.
- Every artifact persists optimizer steps per epoch, absolute planned schedule
  steps, early-stop state, best epoch, stopped epoch, horizon-calibration
  status, and separate embedding/deep LR traces. A `deep_only` artifact is
  usable only if the embedding trace is constant; a `both` artifact is usable
  only if both groups follow the declared multiplier.

## Execution and verification

- Add explicit schedule-horizon configuration, collision-safe horizon names,
  horizon-step metadata, exact verifier/report gates, dedicated linear
  provenance, optimizer-group scope and trace verification, seeded
  joint-search support, and frozen winner manifests by TDD. Optimizer groups
  receive stable identities; scope masking and resume never depend on order.
- Add an RQ5-specific adaptive-horizon policy. The default callback behavior
  for every other experiment remains full-horizon annealing without patience.
- Align the schedule launcher and selector with the single fixed 0.064
  embedding rate; their old 0.008/0.016/0.032 surface is not used by RQ5.
- Initial tuning has 69 treatment-candidate slots and 67 unique runs because
  the two fixed/tuned 5%-warmup central configurations are shared. One
  conditional boundary round can add at most 81 unique configurations, and
  non-central joint winners can add at most 12 local configurations, for a
  ceiling of 160 candidate configurations. Two heuristic corrective horizon
  reruns per configuration give the prior ceiling of 480 queued jobs. The 12
  initially approved and 16 deterministic probe-follow-up surfaces may receive
  at most 56 additional bracket-midpoint reruns. Up to 20 exact named surfaces
  may receive one final exact bracket run, so the hard ceiling is 556 queued
  jobs, not a generic five-attempt ceiling over every configuration. The 16
  probe identities were frozen only after their a2 evidence exhausted under
  the initial allowlist, using the user's standing approval for small
  protocol-preserving corrections. There is no confirmation batch. These are
  ceilings, not a batch submitted upfront.
- Submit stagewise through the existing persistent queue: initial candidates,
  horizon corrections, then boundary probes. Never enqueue later stages before
  their evidence determines them.
- Regenerate a complete target-regime tuning ledger, compact 500M result table,
  raw RQ5 evidence, and reader report. Incorrect active tables are replaced,
  not retained.
- The fail-closed report pipeline is
  [`analysis/rq5_scheduler_report.py`](../analysis/rq5_scheduler_report.py).
  After every candidate and required probe resolves, `analysis/collect.py
  --rq5-only` writes the dedicated native-500M tuning ledger, reader-table
  draft, and machine-readable evidence. It does not generate a 50M RQ5 table
  or edit the reader README.
- Independent review verifies both optimizer-group traces, calibrated schedule
  steps versus actual stopping steps, search boundaries, frozen digests,
  metric arithmetic, and claims.

## Approval

- Material decision: scheduler hyperparameters are tuned and reported directly
  on native 500M seed 42. There is no 50M proxy or seed-43 confirmation.
- Material assumption: “validation loss stopped improving” is a diagnostic;
  recall@100 remains the stopping and selection metric. Changing stopping to
  cross-entropy loss could select worse recommenders.
- Material scope addition: every non-constant scheduler is tested both on all
  optimizer groups and on the deep group only.
- Material horizon rule: early stopping is allowed. Every schedule is
  calibrated and, when necessary, rerun from scratch so its planned step count
  approximately matches its own actual stopping point; insufficient horizons
  are extended rather than accepted.
- Material protocol exception: this adaptive early-stopping rule replaces
  full-horizon annealing for RQ5 only and requires an RQ5-specific code path.
- Exact plan approved by the user: yes.
- Exact probe and selection details approved by the user: clipped factor-two
  seed-42 local boxes, three strictly new points per fourfold boundary
  extension, and four-decimal validation ranking.
- Approval reference/date: chat approval, 2026-08-22.
