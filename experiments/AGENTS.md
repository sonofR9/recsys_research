# Experiment rules

Read `list.md`, its referenced requirements, and the experiment's local report
before changing or launching an experiment.

Before implementing or training any research question whose current plan has
not already received explicit verbal approval, write a concrete plan from
[`plan_template.md`](plan_template.md), explain the intended comparison and
your understanding to the user, and obtain that approval. Planning, read-only
investigation, and protocol review may precede approval; code changes, smoke
training, sweeps, and final runs may not. Approval applies only to the presented
scope. Return for approval when the hypothesis, treatment, control, data
protocol, tuning strategy, or run budget changes materially.

After the user approves the initial plan and initial correction strategy, the
lead may execute small, protocol-preserving refinements without requesting
approval again. This includes deterministic follow-up points, bounded reruns,
and implementation fixes needed to finish the approved comparison. In
particular, when the approved compiler authenticates the exact triggering
result, frozen edge rule, and prescribed follow-up points, launch those points
automatically; do not ask the user to approve that operational continuation.
Return for approval only when the hypothesis, treatment, control, data
protocol, tuning strategy, or run budget changes materially. Final result
interpretations and claims still require the validation defined in the
repository-level rules.

- Choose exactly one dataset size for a research question and use it for every
  hyperparameter-selection run, repeat, and final result in that question.
  Present the proposed size in the plan and obtain explicit user validation
  before implementation or training. Tests, smoke
  tests, and debugging may use a smaller sample with distinct artifact names,
  but those results cannot select treatments or support claims. An explicitly
  approved dataset-size research question may compare multiple sizes within
  its experiment group; each size then trains in its native regime.
- In every semantic-ID experiment, select tokenizer hyperparameters on the
  exact downstream task rather than on reconstruction or intrinsic SID metrics
  alone. Tune number of levels, per-level vocabulary/codebook sizes, and all
  method-specific tokenizer parameters. Bound the search: for about `2^20`
  catalog items, each level has at most `2^13` symbols including any
  collision-resolution symbols assigned to that level.
- Include intrinsic SID metrics and SID-level recall, including prefix recall
  where applicable, in every semantic-ID report. They are mandatory diagnostics
  and useful proxies, not treatment-selection or primary target metrics. Map
  each final top-`K` concrete-item ranking to base SID tuples without collision
  suffixes: exact SID recall@`K` checks for the target tuple and prefix
  recall@`K,d` checks its first `d` levels, for every depth. Keep the item cutoff
  `K`, do not let duplicate tuples create extra ranks, and count a wrong item
  with the target tuple as a proxy hit; therefore also report resolved item
  recall and collision diagnostics. Direct SID generators additionally report
  exact/prefix recall on raw SID beams before collision resolution. For
  variable-length SIDs, append `<eos_sid>` and then `<pad_sid>` to the declared
  family maximum before computing every prefix depth, keeping every evaluation
  example in every depth's denominator.
- All generation experiments use the homework-compatible final protocol in
  `generation_protocol.py`: no final-run user sampling; retain the relevant
  event types and core items with at least five interactions; hold out the final
  seven days by timestamp; train only mapped items; score the full mapped
  catalog; and do not exclude previously seen items at evaluation. Launchers
  default to this protocol; the experiment's approved dataset size is an
  explicit configuration value.
- Sample only by user id, never by position.
- Recall@100 is the primary metric.
- Freeze the experiment's original baseline before composing any RQ report.
  Every RQ's primary reader comparison uses that same original baseline as its
  first row and as the reference for every percentage delta, including slice
  tables. A predecessor, cumulative treatment, or local control may appear in
  an explicitly requested secondary diagnostic comparison, but it never
  replaces the original baseline in the primary comparison or promotion rule.
- Do not create or require a parameter-count-matched, width-matched, or
  compute-matched control unless the user explicitly requests it. By default,
  hold the transformer/backbone and evaluation protocol fixed, tune learning
  rate and horizon plus only the parameter investigated by the RQ, and compare
  with the frozen original baseline. A "matched standalone" aggregate check
  means a protocol-matched direct comparison with that baseline, not a
  parameter-matched control.
- For an architecture ablation, change only that architecture value. Tune its
  learning rate, epoch/schedule horizon, and family-specific capacity parameter;
  keep every other setting fixed and give competing families equal search
  budgets. Tune the unchanged control once and reuse it.
- Every agent-chosen learned neural width must be divisible by 16; prefer a
  multiple of 32. This includes model, attention, FFN/*GLU, MLP/DenseNet,
  projection, and learned-feature widths. Fixed external input widths are
  exempt, but every learned target or hidden width must comply. Never use an
  irregular width such as 171 to approximate a parameter count.
- Fix training batch size at `512` for all newly launched research runs. Do not
  spend tuning budget on batch size. A different batch requires explicit user
  approval; completed runs at other batches remain historical evidence.
- Tune the training epoch or schedule horizon with the other relevant
  non-batch hyperparameters. Validate every epoch, restore the best validation
  checkpoint, and record both the candidate horizon and restored epoch.
- Before proposing a fresh hyperparameter calibration, inspect the selected
  result for the same baseline or the closest protocol-matched predecessor.
  Use its best known values as the central starting point. If batch size,
  depth, dataset, objective, or another relevant field differs, treat the old
  winner only as the center of a local search, not as result evidence. Use
  three candidates around that center by default. Whenever the selected
  learning rate is the smallest or largest tested value, add probes farther in
  that direction before selecting it. Do not fit an LR-versus-horizon law or
  launch a broad baseline recalibration when a suitable best-known starting
  point already exists.
- For G1's remaining native-500M architecture ablations, fix the embedding
  learning rate at 0.064 and tune only the deep rate. On that horizon-complete
  surface, 0.064 is indistinguishable from the optimum of 0.032, so the search
  budget goes to the rate that carries the treatment. G1's already-settled
  comparisons stay at 0.032 so they remain interpretable against each other.
  A new 50M experiment calibrates the unchanged control's embedding and deep
  rates at 50M once; it never imports the 500M rate. Keep that embedding rate
  for treatments that leave the item table, objective, and model family
  unchanged. Otherwise tune both groups with equal budgets and record the
  exception.
- Submit every multi-run batch through `utils/training_queue`. Use one training
  process per GPU, overlap CPU preprocessing, admit idle GPUs immediately, and
  enable light-GPU monitoring without excluding devices by default.
- Update the report after every completed run, including failures or unusable
  results. Every comparison table must show the
  reference parameter value, absolute metrics, and percentage differences.
- Use automatically available runtime and resource telemetry for operational
  sanity; do not add dedicated performance benchmarks by default. Omit
  efficiency tables and all
  timing, throughput, GPU-memory, latency, parameter-count, and efficiency
  columns from reader, compact, and tuning reports unless the user explicitly
  asks about performance or training budget, or a material anomaly must be
  shown to explain whether the scientific result is valid.
- Performance is not a default selection or tie-break criterion. Apply the
  requested quality metrics in their declared order, then use deterministic
  manifest order. Use a performance criterion only when the user explicitly
  made performance or training budget part of the RQ.
- Calibrate repeated-seed variability once per dataset size on its canonical
  unchanged control. All future experiments at that size reuse each metric's saved
  relative dispersion (`sample stddev / baseline mean`) by scaling their own
  baseline; they never reuse the calibration's absolute metric or absolute
  band. Do not launch a new calibration without explicit user approval.
- A reported run count means repeated seeds of exactly one configuration;
  hyperparameter settings are separate rows, not repeats.
- Before beginning an experiment, propose additional relevant research
  questions. Every RQ needs a metric table, a short hypothesis for the observed
  behavior, and a best-method decision.
- Put acceptance criteria in their own plan section. When the user supplied
  them, copy their wording almost verbatim. Fix obvious spelling or grammar only
  when that cannot change the meaning. Preserve the user's comparisons,
  uncertainty, and strength of expectation: `should`, `probably`, `at minimum`,
  and `must` are different and must not be silently strengthened or weakened.
  If the wording is materially ambiguous, ask instead of replacing it with the
  lead researcher's interpretation.

  Acceptance criteria usually say what scientific result is expected. Keep
  them short and direct. A user may instead explicitly make compatible evidence
  reuse or another operational condition the acceptance criterion; preserve
  that too. Agents must not add generic execution requirements—finish the LR
  search, save metadata, produce tables, restore the best checkpoint, or
  complete review—as acceptance criteria on their own. Those belong in the
  protocol unless the user explicitly made one a criterion.

  Do not translate the user's wording into formal gates, statistical language,
  or a different selection rule. Do not add invented thresholds. Do not replace
  a concrete expectation with phrases such as "a defensible result." If the
  user gave no criteria, propose criteria in the same direct style and include
  them in the plan approval.

  Examples:

  - User: "everything is copied from already existing runs, since everything
    in this question has already been investigated in enough depth."
    Write: "Everything is copied from already existing runs, since everything
    in this question has already been investigated in enough depth."
    Do not write: "Run a compatibility audit and accept if all artifact hashes
    match." The rewrite replaces the user's criterion with a new formal gate.
  - User: "16 truncated prefixes should train in less epochs than 8 ... And
    probably have higher quality."
    Write: "Sixteen truncated prefixes should train in fewer epochs than eight,
    and probably have higher quality."
    Do not write: "Select 16 prefixes if Recall@100 improves significantly and
    training cost is Pareto-optimal." The rewrite invents a decision rule.
  - User: "adding pretraining stage should at minimum decrease training time
    without losing quality and most probably improve main metrics."
    Write: "Adding a pretraining stage should at minimum decrease training time
    without losing quality, and will most probably improve the main metrics."
    Do not write: "Any quality/time trade-off is acceptable if it is
    defensible." The rewrite removes the expected direction.
- If a result contradicts the predeclared hypothesis or a user expectation,
  record the expectation and treat an implementation or protocol defect as the
  leading possibility. Before interpreting the result, verify with focused
  tests and saved evidence that the treatment and control differ only on the
  intended axis; data and target construction, masks, losses, gradient flow,
  artifact/configuration identity, evaluation, and run completion are correct.
  There is a high prior probability that a surprising result is caused by an
  implementation error, so a code reading or an assertion that the code looks
  correct is not sufficient. If correctness is established, investigate the
  scientific cause and support the explanation experimentally with targeted
  ablations, controlled reruns, learning curves, data slices, or
  gradient/activation evidence. Prose alone is insufficient, and the RQ is not
  review-ready until the unexpected result is either corrected or explained by
  saved experimental evidence that tests the proposed mechanism or rules out
  plausible confounds.
- Prefix every research question in `ideas.md` with one status:
  `not_started`, `wip`, `review`, or `complete`. Agents may assign only the
  first three and move a question to `review` only when its required runs,
  generated evidence, analysis, and reader-facing report are ready for user
  review. Only the user may mark an RQ `complete`.
- When an event representation concatenates several inputs before the
  transformer and needs an encoder, use DenseNet.

## Report format

Each experiment directory keeps its reader-facing report in its top-level
`README.md`. The report is the only file at that level; implementation,
launchers, report details, archived results, and bookkeeping belong in
meaningfully named subdirectories. Do not leave a flat collection of files in
an experiment directory.

The reader-facing `README.md` contains only:

1. one H1 with the experiment-group name;
2. a two-to-three-line description;
3. one H2 per research question, in order, named exactly
   `RQ{i}: <question>` with consecutive numbering;
4. for each question: one or more result tables, one line per method explaining it,
   essential method or paper details, analysis only when the result is not
   obvious or expected, and a three-to-five-line conclusion;
5. a closing `## Aggregated improvement` section that measures the trained best
   combination of all changes selected by the experiment against its original
   baseline.

This rule was adopted on 2026-08-24. It applies to every experiment added on or
after that date, every earlier experiment with at least one canonical `work/`
task not in `done` at adoption, and every grandfathered experiment whose task is
later moved out of `done`. Earlier experiments whose tasks were all in `done`
at adoption are grandfathered until reopened. A governed experiment is not
report-ready until its aggregate model has been trained and this section has
been generated.

For each dataset size reported, freeze one exact original-baseline
configuration and artifact before composing the aggregate. Include a treatment
only when its RQ explicitly promotes it to the future baseline under its
pre-approved decision rule: normally this means a
Recall@100 improvement past the empirical band, or an RQ-specific multi-metric
trade-off that selects it while Recall@100 remains non-inferior. Mere
non-inferiority or an unresolved result does not qualify. Record the exact
members and every omission before launching the aggregate.

The plan must define treatment compatibility, dependencies, and conflict
precedence before the component RQs run. Treat a method and any prerequisite
that cannot be isolated meaningfully as one atomic bundle, and require matched
evidence for that bundle. Components whose standalone gains will be summed must
be disjoint. For a dependency chain, count the prerequisite bundle's
baseline-to-bundle gain once and obtain a bridge run for each dependent addition
to measure its marginal gain over the already-counted prerequisite; never sum
overlapping baseline-to-treatment gains. Mutually exclusive or superseding
winners cannot coexist in the aggregate; choose among them using the
pre-approved conflict rule. If the observed winners expose a conflict the rule
does not resolve, obtain user approval for the corrected aggregate plan before
launch.
For an experiment already in progress when this rule was adopted, document and
obtain approval for the compatibility/precedence rule before any remaining
component, bridge, or aggregate run; completed component runs need not have
predated a rule that did not yet exist.

Each included treatment must also have protocol-matched standalone evidence
that changes only that treatment relative to the frozen original baseline under
the same dataset and evaluation protocol. Run a bridge cell first when an RQ
used a local or sequential control and therefore lacks that comparison. Put
every qualifying treatment into one model, train it under that same protocol,
and report it against the original baseline as its own row: time features and
SwiGLU and the CLS query together, not one at a time. For an approved
dataset-size companion, train a size-matched aggregate and baseline at every
reported size.

Retune every hyperparameter invalidated by the composition under the existing
family-specific rules, including deep rate and, when applicable, embedding
rate, epoch/schedule horizon, or capacity. Batch remains `512`; any exception
requires explicit user approval. The plan must state the exact aggregate tuning
surface, method, boundary rule, and run budget.

Call the measured baseline-to-combination change the **aggregated improvement**.
For Recall@100, NDCG@100, and every metric used by an included RQ's decision,
report the baseline and aggregate absolute scores, the aggregate gain in metric
points and percent, the sum of disjoint standalone gains plus non-overlapping
marginal bridge gains in metric points, and the interaction gap
`aggregate gain - summed non-overlapping component gains`. Compute all
arithmetic from unrounded metrics before rendering three-decimal reader values.
The gap is descriptive unless its magnitude exceeds a pre-approved interaction
resolution derived from size-matched repeat evidence; only then call it positive
or negative interaction, otherwise label it unresolved. If no treatment
qualifies, state that the aggregate candidate equals the baseline and do not
launch a duplicate training run.

Use this comparison-table shape from the original experiment requirements.
Cost columns remain omitted unless the user explicitly asked about performance
or training budget, or a material anomaly is necessary to establish validity:

| variant | recall@100 | ndcg@100 | other metrics |
| :--- | :---: | :---: | ---: |
| baseline | 0.1 | 1 | ... |
| descriptive short name | +30% (0.13) | −2% (0.98) | ... |
| descriptive short name | +20% (0.12) | +1% (1.01) | ... |

Name an axis column after the parameter being changed; omit a configuration
column when that value fully identifies the treatment. Never give a selected
learning rate a column of its own. Tuning it is still required, but the chosen
value explains nothing about a treatment and crowds out the metrics; report a
rate only where it is the parameter under study, as in a transfer or schedule
question, and leave every other selected rate in the tuning ledger. If a configuration
column is needed for a composite comparison, call it `configuration` and list
only parameters needed to distinguish those rows. The baseline row must state
the baseline value of every changed parameter. Once an experiment's shared
empirical noise bands and single-run policy are recorded here, omit `runs` from
its reader-facing tables. Treatment metric cells show percentage change and the absolute metric.

Round every metric to three decimals. For each metric, multiply the saved
same-size relative dispersion by the current experiment's own unrounded
baseline to obtain its operational threshold. Never reuse an absolute mean or
band, and never transfer relative dispersions across dataset sizes. This is a
deliberate practical approximation, not a significance test. Render a change
green when it improves by more than its derived threshold and red when it
worsens by more than that threshold, using
`<span style="color: green">+5% (0.134)</span>` and
`<span style="color: red">-5% (0.121)</span>`; leave every smaller change
uncolored and unremarked. State the approved dataset size, band provenance, and
thresholds once in the report's opening lines. Use `—` only for genuinely
unavailable fields; do not silently drop a baseline or merge unrelated controls.

Put implementation details, exact code links, and raw-log extracts under an
`evidence/` directory; put report-generation code under `analysis/`. Keep
launchers, archived results, operational instructions, and bookkeeping in
their own named directories outside the main report.

For every research method, keep a separate generated hyperparameter-tuning
report with every tuning run as one row, including the tuned learning rates,
epoch/schedule horizon, family-specific hyperparameters, and metrics. Group tables by
research question and method, and bold the best recall@100 configuration in
each method table. Also keep one generated compact report for the experiment's
approved dataset size containing only research-question headings and result
tables. Every compact research-question heading uses the same consecutive
`RQ{i}: <question>` form as the reader report. The telemetry-omission rule
applies to both tuning and compact reports. Report scripts must generate these
tables and reusable auxiliary artifacts from run metadata; conclusions and
interpretation remain hand-written. Generated intermediate artifacts may live
under the experiment's `scratchpad/` directory.
