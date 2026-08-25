# Research-question plan

## Question and hypothesis

- Research question and status:
- Current understanding:
- Falsifiable hypothesis:
- Why the result matters:

## Comparison

- Unchanged control, including the value of the tested parameter:
- Exact treatment values or methods:
- Factors held fixed:
- Method-specific capacity parameter, if any:
- Sanity or reproduction control:

## Data and evaluation

- Single dataset size proposed for all RQs, tuning, and final evidence:
- User validation reference for the dataset size:
- Sampling unit and any sample size:
- Event filters, core threshold, split, and catalog:
- Primary and secondary metrics:
- Shared empirical noise bands and decision threshold:

## Hyperparameter selection

- Parameters tuned for the control:
- Parameters tuned independently for each treatment:
- Initial learning-rate and batch-size grids:
- Family-specific grids:
- μP or other model-size transfer method:
- Token-horizon transfer method and held-out validation:
- Boundary-extension rule:

## Run stages and compute

- Focused correctness and metric-regression checks:
- Tuning stages and expected run count:
- Boundary or secondary-axis stages:
- Final selections/repeats on the same dataset and expected run count:
- Frozen original-baseline configuration and artifact for aggregation:
- Pre-approved conditional inclusion rule and candidate treatment set:
- Compatibility/dependency graph, atomic bundles, and conflict precedence:
- Resolved aggregate manifest and omissions, recorded after RQ decisions and
  before aggregate launch:
- Matched standalone or bridge evidence for every included treatment:
- Aggregate tuning method, surface, boundary rule, and expected run count:
- Size-matched aggregate cells for a dataset-size companion, if applicable:
- Queue grouping, sequence-cache boundaries, and GPU requirements:

## Interpretation and reporting

- Evidence that would support or reject the hypothesis:
- Checks required for an unexpected result:
- Reuse policy for controls and final runs:
- Reader-facing table(s):
- Full tuning table and compact artifact for the approved dataset size:
- Parameters promoted to the future baseline if selected:
- `## Aggregated improvement` table: original baseline, trained aggregate,
  per-metric aggregated improvement, summed individual gains, and interaction gap:
- Metrics included in aggregate arithmetic and the interaction-resolution rule:
- Treatments omitted from the aggregate and why:

## Approval

- Material assumptions or open choices:
- Exact scope requested for approval:
- User approval: pending
