# G1 tuning protocol

Detailed grids live here; conclusions and selected configurations belong in
the reader-facing [README](../README.md). The prior-semantics 500M report is a
[historical snapshot](../archive/500m/research_questions_500m.md).

## Dataset-size protocol

Yambda-50M is the fast tuning and analysis dataset. It uses its native full
training cohort; it is never repeated to match the number of targets, optimizer
steps, or training time of Yambda-500M. Yambda-500M is the final comparison
dataset and likewise uses its native full cohort.

Both sizes use the same baseline architecture, objective, split, batch size,
schedule family, and evaluation metric unless one of those fields is the
research axis. Evaluate validation every epoch, apply the same predeclared
early-stopping rule, restore the best validation checkpoint, and report that
checkpoint. A maximum epoch count is only a safety cap, not the selected
training horizon.

A schedule that anneals over its horizon — every shape except `constant`,
`inverse_sqrt` and `power` — declares how long training lasts. Spending that
horizon and reporting the best validation epoch within it is an accepted
result: the schedule has finished, and there is no later epoch it could have
chosen. Such a run trains the horizon exactly, without early stopping, because
patience would otherwise fire on the plateau the decay is meant to produce.
The safety cap remains for step-by-step shapes, which have no horizon of their
own and must early-stop strictly inside it.

A run that stops short of its annealing horizon is not selection-resolved: it
measures a schedule it never finished applying.

A decaying shape that stopped before its decay engaged is likewise unusable:
`step` holds its opening rate through the first half of the decay phase and
warmup-stable-decay through the first 80%, so a short run under either is
numerically the constant schedule and is no evidence for its own shape.

Validation is the last seven days of the interaction stream, which is also the
reported evaluation window. A recommender has to be trained up to the moment
it is asked to predict, so there is no later held-out period to select on. The
selected epoch is therefore chosen on the window it is scored on; treat every
epoch-selection difference between arms as optimistic, and compare arms only
against each other.

Tune the approved learning-rate grid on 50M and select the treatment there.
Run the selected treatment once on 500M with the transferred learning rates;
do not use its result to revise the 50M selection. Record both datasets so the
50M comparison remains independently interpretable.

All G1 artifacts selected at a fixed final epoch, including the former
127-epoch target-count proxies and fixed-10-epoch 500M confirmations, are
historical and unusable for research-question answers. The target-count
matching protocol and its launchers have been removed.

Model-width transfer may still use μP from Yang et al., [Tensor Programs V:
Tuning Large Neural Networks via Zero-Shot Hyperparameter
Transfer](https://arxiv.org/abs/2203.03466). It transfers widthwise
learning-rate behavior only; it does not transfer dataset size or validate a
changed position encoding, normalization, depth, negative objective, or other
family field.

## Final-run uncertainty policy

Each selected 500M treatment and the final combined model run once. One set of
unchanged 500M control repeats supplies a shared approximate empirical noise
band. Only artifacts with the current explicit training-semantics revision may
be reused; no legacy artifact qualifies. The report does not call this a
treatment-specific confidence interval.

## Search method

Grid search is only for a single parameter. As soon as two or more parameters
move together — the embedding/deep LR pair, a rate with a negative count, a
mixture share with an alpha — use random search or Optuna at the same budget.
A Cartesian grid over k parameters spends its budget on k one-dimensional
projections and cannot afford a useful resolution in any of them.

The Cartesian LR grids described below predate this rule and are retained only
to keep already-completed comparisons interpretable against each other. New
tuning stages use random search or Optuna.

The architecture treatments now hold the embedding learning rate at 0.032, the
rate that wins across the already-tuned arms, and search the deep rate alone
over `0.003/0.006/0.012`. One moving parameter is exactly the case a grid still
answers, and it keeps a treatment from differing from its control in how far
its own search happened to reach. An axis held fixed by protocol has no
boundary to extend; the deep rate still does.

## Family tuning rules

- Reuse an unchanged tuned control across comparisons.
- Select batch once from the unchanged comparison control, then tune embedding
  and deep LR per treatment at that fixed batch.
- Tune capacity parameters intrinsic to a family, such as SwiGLU intermediate
  width, under the same search budget as its competing family.
- Negative-sampling families are in-batch without logQ, online logQ, fixed
  offline logQ, uniform random, random with offline correction, mixed random +
  online logQ, and mixed random + fixed logQ. Mixed families additionally tune
  random/in-batch share and total negative count.
- Run proxy configurations once. Run only each selected 500M treatment once.

## Exploratory 500M confirmations

A treatment the proxy did not select can still be confirmed on 500M, to test
whether the proxy ranked a family correctly rather than to change what the
family transfers. Name it in `G1_FINAL_EXPLORATORY_TREATMENTS` alongside
`G1_FINAL_TREATMENTS`, give the launch its own `G1_FINAL_RUN_TAG`, and the
preflight passes it as `--exploratory-selection`. Such a run is exempt only from
the rule that an FFN transfer carries a family winner; it must still name the
closed native-50M learning-rate winner of its own configuration, and it is
reported beside its research question rather than inside it.

## Superseded RQ11 negative/logQ proxy grid

This 50M-proxy/500M-transfer workflow is historical and must not be used for
selection or reporting. The active native-500M four-family protocol is
[rq11_mixed_streaming_plan.md](rq11_mixed_streaming_plan.md); its generated
reader table and tuning ledger replace the results described below.

Eight explicitly named objectives use the selected architecture and global
control-selected training batch:

- fixed in-batch global q with fully corrected Yi-2019 logits;
- fixed in-batch q with the leave-one-out-weighted reference loss;
- streaming in-batch global q with fully corrected Yi-2019 logits;
- uniform random without logQ;
- popularity random with fully corrected global-q Yi-2019 logits;
- in-batch without logQ;
- uniform random plus streaming-logQ in-batch negatives, correcting only the
  logQ negative component;
- uniform random plus fixed-logQ in-batch negatives, correcting only the logQ
  negative component.

RQ11 is not accepted unless the homework-compatible fixed in-batch logQ arm
reproduces the homework ordering against the otherwise identical uncorrected
in-batch arm. If it does not, treat the experiment as an implementation or
protocol failure rather than evidence that random negatives are better. Audit
the effective in-batch pool and correction semantics before further final runs.

The fully corrected global-q arms draw target positions unconditionally. They
do not exclude the query's sequence or mask same-item draws, because either
operation would make the proposal query-dependent while the correction still
uses global q. The mixed arms are named as negative-only component corrections,
not as exact Yi-2019 sampled softmax objectives. The leave-one-out-weighted arm
is likewise a separately named reference loss rather than a global-q arm.

`G1_TUNE_NEGATIVE_STAGE=lr` runs only the Cartesian embedding-LR
`0.008/0.016/0.032` by deep-LR `0.003/0.006/0.012` grid. A boundary extension
overrides `G1_TUNE_EMBEDDING_LRS` and/or `G1_TUNE_DEEP_LRS` and must set a
unique `G1_TUNE_RUN_TAG`; the tag and both rates are included in every run
name, so an extension cannot silently overwrite the initial logs.

After selecting one interior LR pair per family,
`G1_TUNE_NEGATIVE_STAGE=secondary` requires
`G1_TUNE_SELECTED_LRS='family:embedding_lr:deep_lr ...'`. It passes that
family-specific pair to every negative-count, streaming-alpha, and
mixture-share run. The logical count grid is `512/1024/2048`: count 512 is
already supplied by the LR winner, while the secondary launcher enqueues only
1024 and 2048. A count-2048 winner adds 4096 before selection. Defaults also cover alpha
`0.0025/0.005/0.01/0.02/0.04` and random shares
`0.125/0.25/0.5/0.75/0.875`; alpha 0.01 and share 0.5 are likewise reused from
the LR stage. Custom secondary grids also require a unique run tag. A secondary-axis
winner must receive one local LR-grid iteration as that exact configuration
before 500M selection.

Negative count is independent of training batch size. The global training batch
is unchanged in initial LR, secondary, exact-local LR, and final runs. It is
recorded in metadata and encoded in non-default-batch and exact run names.

The exact local iteration uses `G1_TUNE_NEGATIVE_STAGE=local_lr` and requires
the selected LR pair plus one
`family:negative_count:alpha:random_share` entry per selected family
in `G1_TUNE_SELECTED_SECONDARY`. It crosses that unchanged secondary
configuration with the LR grid; LR boundary overrides use the same LR-list and
unique-tag mechanism as the initial stage. When the exact winner changes zero
or one secondary axis, its already-completed selected-LR point is omitted from
the local grid; it remains when two or more axes changed because no earlier
stage contains that combination. The 72-run local grid is therefore reduced by
one per zero/one-axis family, to 64 runs when this applies to all eight. Every
local run name contains its family, both rates, batch size, negative count,
alpha, and random share.

After review of the local winners, `selected_500m.sh` requires exactly one
`family:embedding_lr:deep_lr:negative_count:alpha:random_share`
entry for each of the eight table treatments in
`G1_SELECTED_NEGATIVE_WINNERS`. It rejects missing and duplicate treatments
and encodes the complete selected configuration in each final run name.

Fixed in-batch q is a normalized per-draw item probability. Popularity-sampled
random uses `number_of_draws × proposal_q`, the expected negative multiplicity,
and draws from normalized `proposal_q`. Focused reference-logit and internal
sampling-path tests lock these meanings. None of these historical selections is
eligible for the current RQ11 claim.

## Suspicious architecture-result proxy grid

The shared architecture control receives its Cartesian LR grid at the accepted
global effective batch 1280. The selected control LR pair and batch 1280 are
then reused only by unchanged control aliases; batch 1280 is held fixed for
every RQ4-RQ11 treatment, including timestamp and negative-sampling
treatments. Obsolete 1024/1536/2048 batch screens are not required selection
evidence. Each exact treatment uses
`MuTransferGenerationExperiment` with a
fixed 64-dimensional item table and receives its own Cartesian embedding/deep
LR grid at the fixed comparison batch. Sequence-512 uses physical batch 640
with gradient accumulation 2 for the same effective batch 1280 in both proxy
and final training; every other treatment uses physical batch 1280 with
accumulation 1. An LR-boundary winner extends only that LR axis before
selection; treatment batch is never optimized independently.

Use `G1_TUNE_AXES` to stage architecture families independently, for example
`G1_TUNE_AXES=position`; sequence length and time each run alone so unrelated
treatments keep their preprocessing groups. Use
`G1_TUNE_NEGATIVE_FAMILIES` with the descriptive family names in the negative
launcher to stage one or more objectives; its default is all. A partial-axis
invocation may reuse the control only after the launcher verifies the exact
stage control artifact and `G1_TUNE_BATCH_CONTROL`; otherwise it fails before
queue setup.
The manifest-driven `selected_500m.sh` likewise requires exact control
provenance when the shared control is not part of that final invocation.
Ordinary LR stages set `G1_GLOBAL_BATCH_SIZE` and identify the verified
control as `control_key:embedding_lr:deep_lr:batch_size` in
`G1_TUNE_BATCH_CONTROL`. Every downstream architecture, timestamp, RQ11, and
final stage also supplies the same value as
`G1_GLOBAL_BATCH_SELECTION=control/control:embedding_lr:deep_lr:batch_size`;
the launcher recomputes the winner from the completed batch-1280 control LR
surface before queue setup.
Final selections reuse `G1_GLOBAL_BATCH_SIZE`;
treatment entries contain only treatment and the two
selected learning rates. The unchanged control is tuned exactly once on its
approved batch surface. No treatment-specific batch extension is allowed.

- GELU widths 128/171/256/384 compete with SwiGLU widths
  16/32/64/96/128/171/224.
  RQ4 compares the independently tuned best GELU and SwiGLU families;
  SwiGLU-192 is the shared selected-architecture control for the other axes.
- The dimension axis is one μP width family with a fixed 64-dimensional item
  table and learned projections. It preserves the selected SwiGLU capacity
  ratio 192/64: dimensions 16/32/64/128/256 use rounded intermediate widths
  43/86/171/342/684.
- Position treatments cover none, learned forward/reverse, RoPE
  forward/reverse, ALiBi, and every existing two- and three-way combination.
- Attention-head count compares MHA with 1/2/4/8 heads against MHA-2. The
  grouping table compares MHA-2 with 2-query/1-KV GQA; GQA remains the shared
  selected-architecture control for unrelated axes.
- Normalization compares the existing block/input/final LayerNorm, RMSNorm,
  BatchNorm, no-final-norm, and post-norm treatments against the shared
  pre-LayerNorm control.

Every proxy configuration runs once on Yambda-50M. The selected configuration
for each exact treatment runs once on 500M; uncertainty comes from the one
shared unchanged-control repeat set, never from treatment-specific repeats.

The RQ8 grid covers sequence lengths 12/25/50/100/128/200/256/512,
local-attention windows none/10/25/50/75/100, dropout
0/0.05/0.1/0.2/0.3/0.5, and CLS-query off/on.
The CLS treatment replaces the final history-state query with a learned token
appended after that history. The control query and CLS query predict the same
held-out next item, so the number and identity of supervised targets stay
unchanged. Its launcher uses the shared μP proxy experiment. The sequence
analysis includes the 75,725
users with at least two pre-holdout events and finds a median training history
of 48 events; its reproducible table and plot are in `../evidence/`.

The earlier conventional proxy shortlist and 29-run 500M batch are archived
historical evidence. They do not satisfy this μP treatment-specific tuning
protocol and are not current RQ conclusions. Current conclusions use only the
corrected proxy and exact final runs whose raw artifacts passed verification.
