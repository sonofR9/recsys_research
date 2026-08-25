# Understanding of the experiment ideas

This interprets [`ideas.md`](ideas.md); it is not an approved collection of run
plans. Completed research questions are omitted because their meaning and
conclusions belong in the experiment reports. Status remains in `ideas.md` and
the work tracker.

## Program-wide structure

### Main experiment

Each experiment's common setup declares its proposed dataset size. That size
must be approved before implementation or training and is then used for all of
the experiment's RQs, tuning, and final evidence unless dataset size is itself
the explicitly approved research axis. Each experiment calibrates its unchanged
control at that size, selects one global batch size, measures size-matched
empirical bands, and freezes common preprocessing, split, catalog, evaluation
users, and non-treatment settings. A changed architecture or objective family
receives an equal tuning budget; an unchanged family reuses the control
settings.

Implementation follows the repository's existing stage model. A run is a
config-as-code `Experiment`; preprocessing or learned artifacts such as a
tokenizer or teacher are preceding stages, not hidden work inside the model
training loop. Sequence-pair construction belongs in `sequence_targets.py`,
event serialization in `history_tokens.py`, reusable layers in `dcn/nn/`,
experiment assembly in `dcn/config/`, and full-catalog or generation metrics in
`dcn/eval/`. Each RQ below states the changed component and comparison arms;
the eventual approved plan must additionally enumerate its exact configuration
grid, run count, and promotion rule.

### Semantic-ID tuning rule

Every group that constructs or consumes semantic IDs tunes the tokenizer on
that group's downstream validation task. Number of levels, every level's
vocabulary/codebook size, and method-specific tokenizer parameters are part of
the search; reconstruction, ICR, load, and semantic cohesion are diagnostics,
not substitutes for downstream selection. The search must stay reasonably
bounded: for a catalog of about `2^20` items, every level is limited to at most
`2^13` symbols, including collision-resolution symbols assigned to that level.
Every semantic-ID report still includes those intrinsic metrics plus SID-level
recall and prefix recall where applicable. They are mandatory diagnostics and
useful proxies, even though the downstream target metric selects treatments.

### Common acceptance contract

Acceptance means that an RQ has produced a defensible comparative answer, not
merely that its arms ran. Each RQ below names its control and decision rule.
Unless it says otherwise:

- promote a quality treatment only when its primary downstream metric improves
  over the named control beyond the dataset-size empirical band;
- accept an efficiency treatment only when primary quality is not worse beyond
  the band and at least one declared resource metric improves measurably;
- treat a delta inside the band as a resolved null result, not a gain; and
- reject a treatment that is worse beyond the band, while retaining the result
  as a valid negative finding once the regression is explained.

A negative or surprising null result is acceptance-ready only after focused
invariants prove that the treatment was active, the comparison was matched,
the hyperparameter optimum was not left on a search boundary, training reached
its valid stopping condition and restored its best checkpoint, and learning
curves plus relevant data/model slices identify a plausible mechanism. Rerun
matched controls or seeds when those checks cannot distinguish a real effect
from instability. Every report states the comparative decision—promote,
trade-off, no detectable effect, reject, or blocked—and includes the evidence
required by its RQ-specific acceptance paragraph.

## 1. SASRec over item IDs and likes

### Common setup

**Main dataset: native Yambda-500M.** The calibration range, global batch, and
open G1 comparisons already live on 500M. Moving them would break comparability
with completed G1 evidence. The common control is the homework-compatible
item-ID SASRec trained on likes under the final full-catalog protocol.
Implement variants as `GenerationExperiment` subclasses/configurations;
transformer components come from `dcn/config/networks.py`, token changes from
`dcn/models/history_tokens.py`, and objectives from
`dcn/nn/sampled_softmax.py`.

### RQ2 — What is the maximum-quality combination?

**Understanding.** Individual wins may interact; adding their reported gains is
not evidence that their combination wins.

**Implementation.** Include only treatments already resolved as improvements
on 500M. Compose them in one model, retune the combined model, and compare it
with the unchanged control and strongest individual treatment. Report the sum
of isolated gains beside the measured joint gain and analyze destructive or
synergistic interactions.

**Acceptance.** Select the combination only if Recall@100 exceeds the strongest
qualifying individual treatment beyond the native-500M band; otherwise retain
that individual and report a null or regression. Compare its measured gain with
the approved sum of non-overlapping component gains. A shortfall beyond the
interaction-resolution band requires component-removal runs plus compatibility,
convergence, and tuning-boundary analysis.

### RQ3 — What is the best quality/performance balance?

**Understanding.** This asks for a Pareto choice, not the configuration with
the second-highest recall.

**Implementation.** Benchmark measured G1 candidates under one hardware
procedure. Record recall@100, epoch time, full-catalog inference latency, peak
accelerator memory, and parameter count. Remove dominated points, then select a
cheaper configuration under an explicit maximum recall loss. The selected point
must be a complete configuration and is retuned as such.

**Acceptance.** The selected point must be Pareto-nondominated against all
eligible complete configurations. A cheaper point qualifies only when
Recall@100 is non-inferior to the maximum-quality configuration within the band
and at least one of epoch time, full-catalog latency, peak memory, or parameters
improves beyond repeated-measurement dispersion. If none qualifies, retain the
maximum-quality configuration and report the absence of a useful trade-off.

### RQ10 — Do per-layer item embeddings help?

**Understanding.** The intended mechanism is Gemma-style Per-Layer Embeddings,
not an invented concatenation inside the FFN.

**Implementation.** First pin lookup, projection, sharing, and injection
semantics from the released Gemma implementation. Compare the ordinary shared
input table with layer-specific item lookups and a proposed parameter-matched
wider/deeper control. Tune the treatment family fairly and report quality,
total parameters, active accelerator parameters, lookup bandwidth, memory, and
latency. Reference: [Gemma PLE](https://ai.google.dev/gemma/docs/gemma-3n).

**Acceptance.** Attribute a quality gain to per-layer embeddings only if
Recall@100 exceeds both the shared-table control and parameter-matched
wider/deeper control beyond the band; beating only the smaller control is a
capacity result. A trade-off qualifies only with recall non-inferiority and a
cost improvement beyond measurement dispersion. Nulls or regressions require
tests of layer-table independence/gradient flow and analysis of lookup use,
bandwidth, memory, convergence, and tuning boundaries. Until the
parameter-matched control and those diagnostics exist, leave the mechanism
unresolved even if a smaller-control comparison has completed.

### Dataset-size RQ — Does scale change the maximum-quality G1 gain?

**Understanding.** Test whether the maximum-quality G1 configuration improves
over the homework-compatible baseline by the same amount at 50M and 500M. The
quality/performance Pareto configuration is not a second selection target in
this companion RQ.

**Implementation.** Apply the shared companion protocol to the exact final G1
configuration and homework baseline. Recalibrate both optimizer groups at each
size, retain the same architectural treatment definition, and compare
`delta_50M` with `delta_500M`. This is distinct from RQ1's μP transfer question.

**Acceptance.** Compute treatment-minus-baseline Recall@100 deltas independently
at 50M and 500M and compare their difference with a predeclared combined
resolution from the two size-specific bands. Claim a scale interaction only
beyond that resolution; otherwise conclude that the gain is stable within
measurement resolution. A sign reversal or boundary case requires seed,
tuning-boundary, catalog-slice, and protocol-equivalence analysis.

## 2. eSASRec

### Common setup

**Main dataset: native Yambda-50M.** Use the size-calibrated G1 structural
baseline, repository preprocessing/evaluation, and official RecTools behavior
for the modules under test.
Add an eSASRec experiment configuration beside the existing generation
configuration and isolate reusable LiGR/loss components under `dcn/nn/` rather
than importing an external training loop.

### RQ1 — What are eSASRec's metrics and component effects?

**Understanding.** The core eSASRec recipe combines the SASRec shifted-sequence
objective, LiGR blocks, and sampled-softmax loss. The question also asks which
pluggable changes cause any improvement.

**Implementation.** Verify local modules against the official implementation on
deterministic tiny tensors; this is a parity gate, not an experiment arm. Run a
paper-style factorial of standard versus LiGR blocks and tuned gBCE versus
sampled softmax while holding shifted-sequence targets and the negative
proposal fixed. The primary repository comparison remains selected eSASRec
versus the G1-derived control; because that control already uses sampled
softmax, do not claim sampled softmax as an additional gain there. Test the
paper's optional mixed uniform/in-batch sampler as a separate arm on the full
eSASRec model, tuning mixture and logQ rather than confounding the factorial.
Match capacity and report quality, coverage, time, memory, and latency.
References: [paper](https://arxiv.org/abs/2508.06450) and [RecTools](https://github.com/MTSWebServices/RecTools).

**Acceptance.** Official and local forward, loss, and gradient tensors must
agree within predeclared dtype-specific numerical tolerance, and reproduced
official metrics must lie inside their repeat-derived band before local results
are eligible. Claim eSASRec improvement only when selected eSASRec Recall@100
beats the G1-derived control beyond the band. Resolve LiGR versus standard
blocks at the same loss and matched capacity, and gBCE versus sampled softmax at
the same block family. Resolve mixture logQ only against an otherwise identical
same-fraction uncorrected arm. Promote each component only for a Recall@100 gain
beyond its paired band; a tied model needs non-inferior recall and a measurable
resource gain. Nulls or regressions require convergence, loss-scale,
proposal/logQ, coverage, capacity, and interaction analysis without
double-counting sampled softmax.

## 3. Pretrained item embeddings

### Common setup

**Main dataset: native Yambda-50M.** Compute, normalize, version, and freeze the
provided content/audio vector for every train-mapped item. The control has a
learned item-ID input table and learned item-output table. Input and target are
studied in RQ1/RQ2; RQ3 fixes the concatenated RQ2 input while changing the
prediction space.
Use `PrecomputedEmbeddingLookup` for frozen vectors, add composition encoders
under `dcn/nn/`, and keep query/catalog encoders exposed through the existing
full-catalog evaluator.

### RQ1 — What happens when pretrained embeddings replace item IDs?

**Understanding.** This isolates content-only history representation from
collaborative item memorization.

**Implementation.** Replace the learned input lookup with the fixed pretrained
vector followed by a learned projection to model width. Keep the learned
item-ID output target fixed. Tune the projection and model rates, match active
capacity where possible, and report overall plus item-frequency slices.

**Acceptance.** Classify content-only input as improvement, null, or regression
from its Recall@100 delta against the learned item-ID control and the empirical
band. Select it overall only after a band-exceeding gain; a slice-specialist
claim additionally requires aggregate non-inferiority and a predeclared
frequency-slice gain beyond its control-derived band. Nulls or regressions
require coverage, missing-vector, norm, head/mid/tail, convergence, and resource
analysis after frozen-vector and projection-gradient tests pass.

### RQ2 — Does concatenating content and item ID help?

**Understanding.** Test whether content generalization and item-specific
collaborative information are complementary.

**Implementation.** Concatenate learned item-ID and frozen content vectors and
encode them with DenseNet before the transformer. Compare with the unchanged
item-ID input and a proposed parameter-matched item-ID-only DenseNet control.
Keep the learned item-output target fixed. Tune encoder capacity and relevant
rates without changing the negative objective.

**Acceptance.** Attribute complementarity to content only when concatenation
beats the parameter-matched item-ID-only DenseNet control beyond the Recall@100
band and does not regress against the unchanged item-ID control. Beating only
the unchanged control is an encoder/capacity result. A tied treatment qualifies
only with recall non-inferiority and a resource or predeclared slice improvement
beyond its measurement band; otherwise retain the simpler control after
projection-use, slice, convergence, and boundary analysis.

### RQ3 — Which prediction embedding is best?

**Understanding.** The alternatives are learned item-ID embedding, pretrained
embedding, and their concatenation as retrieval targets. Concatenation is one
joint target space, not automatically multi-task learning.

**Implementation.** Feed every history item as
`DenseNet(concat(learned_item_id_embedding, pretrained_embedding))`, using the
selected RQ2 encoder. Keep that input identical and try these full-catalog
output tables:

1. a learned item-ID embedding table initialized randomly;
2. the frozen pretrained table followed by a learned projection;
3. a trainable copy initialized from the pretrained table, followed by the same
   projection;
4. `concat(learned_item_id, frozen_pretrained)` followed by a learned shared
   projection and normalization;
5. the same concatenation with a trainable pretrained copy.

Match final output dimension and normalization, and give each family the same
tuning budget. This separates target identity, initialization, and whether the
content component remains frozen.

**Acceptance.** Replace the randomly initialized learned item-output table only
when a candidate's Recall@100 gain exceeds the band; a cheaper target may
qualify with recall non-inferiority and resource savings beyond dispersion.
Paired contrasts must separately resolve target type, pretrained
initialization, and freezing. Nulls or regressions require dimension/freeze and
full-catalog tests plus score-scale, normalization, trainable-copy drift,
frequency-slice, convergence, and tuning-boundary analysis.

### Dataset-size RQ — Does scale change the content benefit?

**Understanding.** More interactions may strengthen collaborative item-ID
learning relative to fixed pretrained information, so the winning content
treatment's effect may differ between 50M and 500M.

**Implementation.** Select the best G3 treatment on 50M, freeze its input and
prediction definitions, then train it and the learned item-ID baseline natively
at both sizes. Rebuild mapped content tables and tune both families fairly
inside each size. Compare overall, head/mid/tail, and low-history treatment
deltas; do not reselect a different G3 treatment on 500M.

**Acceptance.** Compare the frozen treatment-minus-item-ID Recall@100 deltas
between 50M and 500M using combined size-specific resolution. Claim a scale
effect only when that delta difference exceeds the combined resolution;
otherwise report no detectable interaction. Overall and head/mid/tail or
low-history conclusions must agree or the report must explain their
disagreement through content coverage, catalog composition, slice weighting,
convergence, and tuning boundaries without reselecting the treatment at 500M.

## 4. Predicting future items during training

### Common setup

**Main dataset: native Yambda-50M.** The control is standard next-liked-item
training. Construct all broader positives strictly after each prefix and within
the training interval. Keep final-seven-day evaluation unchanged and mask all
valid positives for a query from its negatives.
Extend `SequenceTargets`/`TimeWindowTargets` for target construction and expose
each rule through a `TimeWindowGenerationExperiment` configuration; the model
architecture itself stays unchanged.

### RQ1 — Does a 24-hour future window help?

**Understanding.** The target treats any positive engagement in the next day as
acceptable instead of privileging the immediately next event.

**Implementation.** For each selected prefix, sample eligible liked-item
positives from its next 24 hours using dense multi-positive supervision. Match
the number of prefix-positive
pairs and optimizer budget to the next-item control. Report aggregate metrics
and recall by temporal distance. Reference: [PinnerFormer](https://arxiv.org/abs/2205.04507).

**Acceptance.** The 24-hour objective helps only if final Recall@100 exceeds the
matched next-item control beyond the band; a within-band delta is a null and a
negative beyond-band delta rejects it. A temporal-slice benefit is reportable
only with aggregate non-inferiority and a slice gain beyond its control-derived
band. Nulls or regressions require strict-future/window and negative-mask tests
plus temporal-distance, activity, multiplicity, coverage, and convergence
analysis.

### RQ2 — Does a next-event-count window help?

**Understanding.** A next-10-events target removes user activity rate from the
window definition.

**Implementation.** Use positives from the next 10 eligible events while
keeping prefix sampling and total positive-pair budget matched. The source asks
for 10; smaller `k` values are optional diagnostics, not required treatments.
Compare with next-item and 24-hour objectives under the same evaluator.

**Acceptance.** Select next-10-events only when Recall@100 exceeds both
next-item and 24-hour controls beyond the band; otherwise retain the strongest
non-regressing control. A specialist conclusion requires aggregate
non-inferiority and a predeclared activity-rate slice gain beyond its band.
Nulls or regressions require exact-next-events and negative-mask tests plus
activity-rate, temporal-distance, target-overlap, multiplicity, and convergence
analysis.

### RQ3 — Can behavior-similar periods define better positives?

**Understanding.** This proposes learning which later hours/days represent the
same user state, then using only their items as positives. It is conditional
future supervision, not simply a longer window.

**Implementation.** First define period boundaries and training-only similarity
labels. Try the following selectors, all using only information available at
the prefix and training-only future labels:

1. deterministic matching to the same local hour and day-of-week in later
   weeks;
2. cosine similarity between content centroids of the prefix period and each
   candidate future period;
3. cosine or weighted-Jaccard similarity between their item, artist, or album
   frequency vectors;
4. a binary period-pair classifier whose inputs are the past-period summary,
   time gap, hour/day features, and user-activity counters, and whose label is
   whether future-period behavior exceeds a fixed similarity threshold;
5. for the best learned score, hard top-k period selection versus sampling
   positives proportionally to the predicted similarity.

Use one-hour, six-hour, and calendar-day periods as the initial bounded choices.
For hourly periods search later periods within the next seven days; for daily
periods search the next 28 days. With several moving choices, use one bounded
random search over period width, lookahead, similarity threshold/top-k, and
classifier settings rather than a Cartesian grid.

Freeze every selector before recommender training and compare it with matched
next-item and fixed-window controls using the same prefix-positive budget. Stop
before recommender training if the classifier does not beat the deterministic
selectors on held-out period-pair ranking or if materializing its positives is
outside the approved runtime budget.

**Acceptance.** Advance the classifier only if held-out period ranking beats
the best deterministic selector beyond its repeat-derived band and
materialization stays within the approved relative runtime budget. The selected
period treatment wins only if downstream Recall@100 beats next-item and the
best fixed-window control beyond the recommendation band. Otherwise reject it
after leakage tests and selector-rank/recall, positive-overlap/diversity,
distance/activity, and runtime analysis.

## 5. Likes and listens with action tokens

### Common setup

**Main dataset: native Yambda-50M.** Preserve chronological likes and listens.
Maintain matched likes-only and listens-only controls and report each target
separately; macro metrics cannot hide negative transfer. Count comparable
history in events rather than serialized tokens.
Build on `ActionGenerationExperiment`: serialization changes belong in
`ActionTokenizer` or a new event tokenizer, target/mask changes in
`SequenceTargets`, and multi-output losses in the experiment criterion rather
than special cases in the trainer.

### RQ1 — Does one action type in history help the other?

**Understanding.** This changes history information while keeping the predicted
task fixed.

**Implementation.** For the likes task, compare likes-only history with joint
likes+listens history. Repeat symmetrically for listens. Keep the target head,
loss, and evaluation fixed within each pair. This produces two transfer deltas:
listens→likes and likes→listens.

**Acceptance.** Resolve listens→likes and likes→listens independently against
their target-only controls; joint history is promoted for a direction only
after a target Recall@100 gain beyond its band. A trade-off requires target
non-inferiority and a resource benefit beyond dispersion; macro averaging may
not hide negative transfer. Nulls or regressions require chronology/merge tests
and history-length, activity, action-overlap, attention-use, convergence, and
cost analysis.

### RQ2 — Does jointly predicting likes and listens help?

**Understanding.** This changes target supervision while holding joint history
fixed.

**Implementation.** Compare two independently trained single-target models with
one shared encoder and two task heads. Tune the loss weight, compare shared and
task-specific output projections under matched budgets, and report both target
deltas against their corresponding single-task control.

**Acceptance.** A joint model is a quality winner only when neither action's
Recall@100 regresses beyond its band and at least one improves beyond its band.
If both are merely non-inferior, selection requires a serving or training cost
saving beyond dispersion. Otherwise keep separate models after head-routing and
mask tests plus loss-boundary, gradient-conflict, calibration, per-action, and
convergence analysis.

### RQ3 — Are explicit `<action>` and `<item>` tokens needed?

**Understanding.** Determine whether the model benefits from an explicit typed
grammar or whether action information can be attached directly to an item
event.

**Implementation.** Compare two-token `<action> <item>` serialization with a
single item event carrying an action embedding and, if desired, action-specific
item tokens. Match visible event history and transformer capacity. Report raw
token length and latency in addition to per-action quality.

**Acceptance.** Prefer compact single-event serialization when like and listen
Recall@100 are both non-inferior and token count or latency improves beyond
dispersion. Prefer explicit typing for quality only when at least one action
improves beyond its band without beyond-band harm to the other. If both quality
and cost are indistinguishable, report a null; regressions require equal-history
and causal tests plus truncation, action-confusion, position, and representation
use analysis.

### RQ4 — Does request-conditioned interleaving help?

**Understanding.** Train one model on interleaved task blocks. A request token
selects which action-specific item should follow: `<want like>` asks for the
next liked item after the current prefix and `<want listen>` asks for the next
listened item. Ordinary chronological items remain in the same causal stream;
after such an item the model outputs its like/listen probabilities. A stream
can therefore look exactly like `<want like>, item_i, <want listen>, item_j,
item_k, [p_like, p_listen], <want like>, item_l`. At serving, appending
`<want like>` requests a like recommendation.

**Implementation.** For every eligible chronological prefix derive three
different blocks: `<want like>, next_like_after_prefix`, `<want listen>,
next_listen_after_prefix`, and `immediate_next_item, [like/listen output]`.
Thus a requested target may skip intervening events, whereas the ordinary block
never does. Sample and pack these blocks into the interleaved causal training
stream with segment-aware attention: every block sees its own chronological
prefix and its request, but cannot attend to targets from adjacent blocks. A
requested block applies item loss at the request position. An ordinary block
applies chronological next-item loss before the item, then a two-logit action
loss after the item using that item's hidden state; the probability pair is an
output, not ground-truth input. Loss masks must also guarantee that an item
prediction cannot attend to its action target. Tune the block mixture while
keeping the number of supervised items fixed. Compare the all-ordinary stream,
interleaving without action loss, and the full interleaved stream. At evaluation
append each request separately and measure recall for its held-out action.
RQ4.2 isolates whether the post-item action loss is necessary.

Also compare an offline-only baseline without request tokens. Serialize each
history event as one vector built from the item embedding, `is_like`, and
listen percentage; encode listen percentage both as a normalized scalar and as
a learned bin embedding. From the prefix, score every candidate item and make
the auxiliary like-probability and expected-listen-percentage heads conditional
on that candidate, so those predictions can actually rerank items. Train the
three outputs jointly and tune their weights. Report ordinary next-item recall,
like-oriented ranking from item score plus calibrated like probability, and
listen-oriented ranking from item score plus calibrated expected listen. This
is a deliberately non-production control: it tests whether enriched item events
and multi-task prediction beat the special-token interface, not whether they
support a serving request protocol.

**Acceptance.** A request-conditioned system qualifies only if requested-like,
requested-listen, and ordinary next-item recall do not regress beyond their
bands and at least one improves beyond its band; alternatively all must be
non-inferior with a cost saving beyond dispersion. The offline enriched-event
model is selected only as an offline baseline under the same metric rule, not
as a production interface. Nulls or regressions require mask/target tests and
block-mixture, gradient-conflict, calibration, oracle, slice, and latency
analysis.

### RQ4.1 — How should action and item tokens be aggregated?

**Understanding.** Rich events should not necessarily double sequence length.

**Implementation.** Compare the expanded serialization with one event vector
from `DenseNet(concat(action_embedding, item_embedding))`. Keep the same event
history and request/target grammar. BPE is not assumed to solve this comparison;
it can be studied later as a separate serializer.

**Acceptance.** Select the aggregated event vector only when both action recalls
are non-inferior and sequence length, memory, or latency improves beyond
dispersion; select expanded tokens for quality only after a band-exceeding gain
on one action without harm to the other. If quality and costs stay within their
bands, report no detectable difference. Regressions require equal-access tests
and truncation, representation-use, action-confusion, and latency profiling.

### RQ4.2 — Which auxiliary losses are needed?

**Understanding.** Separate the value of predicting the next item from the
value of predicting interaction type.

**Implementation.** With representation fixed, compare next-item loss only,
and next-item plus interaction-type loss. Tune the joint weight and report both
item retrieval and action classification. Do not add an interaction-only arm:
without an item objective it cannot answer the recommendation question.

**Acceptance.** The no-auxiliary-loss arm is the control. The auxiliary action
loss helps recommendation only when item Recall@100 improves beyond its band
without beyond-band harm to either action. It may be retained as a multi-task
trade-off when recall is non-inferior and action classification or calibration
improves beyond its repeat-derived band. Otherwise reject it after loss-mask
tests and weight-boundary, gradient-conflict, calibration, per-action, and
convergence analysis.

### RQ4.3 — Additional ablation

**Understanding.** The source leaves the factor open. A concrete proposed RQ is
whether request information is needed only at the final query or throughout
training target construction.

**Implementation.** With the selected representation and losses fixed, compare
one request token appended only before the serving/training query with request
tokens inserted before every supervised item target. Keep the same visible
history events and number of target pairs. Report per-action quality, token
length, and whether intermediate request tokens alter the opposite action's
transfer. This is a proposed completion of the blank RQ, not an interpretation
of a treatment already specified in `ideas.md`.

**Acceptance.** Per-target request placement helps only if one action recall
improves beyond its band and the other does not regress; query-only placement is
preferred when both are non-inferior and its token or latency saving exceeds
dispersion. If quality and cost deltas are all within their bands, report no
detectable placement effect. Regressions require position/mask tests plus
opposite-action transfer, truncation, attention-to-request, calibration, and
leakage analysis.

## 6. RQ-KMeans semantic IDs in history

### Common setup

**Main dataset: native Yambda-50M.** Fit RQ-KMeans only on content vectors for
the train-mapped catalog. Version normalization, codebooks, assignments,
centroids, level vocabulary, and collision map. The downstream task remains
item retrieval. Report item metrics plus ICR, p95 load at every level,
intra-code similarity, and collision distribution.
Tune levels, per-level codebook sizes, and other RQ-KMeans parameters on this
downstream item-retrieval task under the program-wide per-level limit.
Create a `KMeansIdStage` before the downstream experiment, represent codes with
`SemanticIdTokenizer`/`dcn/nn/semantic_embedding.py`, and store all fitted
artifacts under the experiment's dataset-keyed paths.

### RQ0 — How should SIDs describe history?

**Understanding.** Compare seven sequence representations without changing the
prediction target.

**Implementation.** Test exactly: one event token made from trainable SID-level
embeddings plus level tags; item ID concatenated with frozen codebook vectors;
item ID concatenated with trainable SID embeddings and frozen codebook vectors;
one sequence token per trainable SID level; the same per-level tokens with
trainable and frozen codebook vectors concatenated; frozen codebook vectors as
the per-level tokens; and an interleaved item-ID/SID token stream for every
history item. Use DenseNet for every concatenated event representation. For all
expanded forms truncate by history-item count and keep the same items visible.
Tune SID parameters and representation capacity separately for every family
under one fixed collision-resolution policy; RQ2/RQ3 later reopen that policy.

**Acceptance.** Compare every independently tuned SID representation with the
learned item-ID history control. Promote one only through a Recall@100 gain
beyond the band, or through non-inferior recall plus a sequence-length, memory,
or latency improvement beyond dispersion; otherwise retain item IDs. Nulls or
regressions require alignment/truncation/mask tests and analysis by history
length, collision bucket, and item popularity with full SID diagnostics.

### RQ1 — How should SID embeddings be initialized?

**Understanding.** Compare pure task learning with content-informed
initialization.

**Implementation.** Apply this only to RQ0 representations that contain a
trainable SID lookup. Initialize that lookup randomly or from the corresponding
codebook/centroid vectors after an explicit projection, then train end to end
under equal budgets and report convergence plus final quality. If the RQ0
winner has no trainable SID lookup, use the strongest applicable representation
for this initialization RQ without replacing the RQ0 winner.

**Acceptance.** Random initialization is the control. Select content/codebook
initialization if final Recall@100 improves beyond the band, or if recall is
non-inferior and time-to-best validation improves beyond run-time dispersion;
early convergence alone is otherwise diagnostic. Nulls or regressions require
artifact/scale/trainability tests plus learning-curve, update-norm, collision,
and popularity analysis.

### RQ2 — What is the best collision-resolution-token setup?

**Understanding.** Find the best RQ-KMeans tokenizer hyperparameters when a
collision-resolution token is included. This is primarily a search over
codebook size, number of levels, and related tokenizer parameters—not over
different collision-token designs.

**Implementation.** Use one fixed collision-resolution rule and random-search
the number of residual levels, every level's codebook size, and relevant
RQ-KMeans fitting parameters with the selected RQ0 representation. Select on
downstream item recall, with ICR, p95 load, intra-code similarity, collisions,
memory, and latency as diagnostics. Apply the `2^13` maximum to every level,
including the collision-resolution level/symbols, for an approximately
`2^20`-item catalog.

**Acceptance.** Replace the fixed collision-token reference only when a tuned
configuration improves Recall@100 beyond the band, or is non-inferior while
using measurably less vocabulary, memory, or latency. If none qualifies, retain
the reference. Nulls or regressions require deterministic uniqueness and symbol-
limit tests plus load, collision-bucket, tail-recall, memory, and latency
analysis.

### RQ3 — What happens without collision resolution?

**Understanding.** Repeat RQ2's tokenizer-hyperparameter search without adding
a collision-resolution token. This is not a single ablation that removes the
token from RQ2's winner; the optimum number of levels and codebook sizes may
change when collisions remain unresolved.

**Implementation.** Run the same bounded random-search axes and budget as RQ2,
with the same selected RQ0 representation, but omit collision resolution.
Define an ambiguous tuple as the shared history representation of every item in
its bucket. Select on downstream item recall and report the same intrinsic,
collision-bucket, memory, and latency diagnostics so the independently selected
with/without-collision setups can be compared.

**Acceptance.** Compare the independently tuned no-collision system with the
independently tuned RQ2 collision winner. Prefer no collision only after a
Recall@100 gain beyond the band, or non-inferior recall with a measurable
vocabulary, memory, or latency benefit; otherwise retain collision resolution.
Nulls or regressions require ambiguous-tuple/full-catalog tests and bucket-size,
tail, load, and latency analysis.

### Dataset-size RQ — Does scale change the SID-history gain?

**Understanding.** Larger catalogs may increase semantic sharing and collisions
simultaneously.

**Implementation.** Freeze the selected representation mechanism, then refit
RQ-KMeans natively and retune baseline/treatment at each size. Compare against
the learned item-ID history baseline, including changes in ICR, load, collision
distribution, tail recall, and treatment delta.

**Acceptance.** Resolve the SID-history gain separately at each size against its
size-local item-ID control. Claim a scale interaction only when the difference
between treatment deltas exceeds the preapproved combined resolution; otherwise
report no detectable scale effect. Nulls, reversals, or slice disagreement
require artifact-isolation tests and catalog, load, collision, truncation,
frequency-slice, and cost analysis at both sizes.

## 7. Semantic-ID generation

### Common setup

**Main dataset: native Yambda-50M.** By default, fit and freeze one approved
RQ-KMeans tokenizer for comparisons that do not make tokenizer design part of
the treatment. RQ2 and RQ7 are explicit exceptions: they first use a shared
tokenizer for isolation, then give every representation/direction an equal
independent search over levels, per-level sizes, and RQ-KMeans parameters; the
independently tuned systems determine their final answers. Build a trie of valid
tuples and hold constrained-beam policy, collision mapping, parameter budget,
and candidate budget fixed inside each comparison. Report exact/prefix SID
recall, resolved item recall, ICR, per-level p95 load, intra-code similarity,
collision distribution, and trie/beam diagnostics. Select tokenizer settings
on G7 item recall under the program-wide per-level limit; SID recall and
intrinsic SID metrics are reported proxies, not selection targets.
Use `SemanticGenerationExperiment` with `CausalTokenDecoder` or
`Seq2SeqTokenDecoder`; implement valid-prefix constraints through
`SemanticIdConstraint`/the semantic trie and generation metrics in `dcn/eval/`.

### RQ1 — Decoder-only or encoder-decoder?

**Understanding.** Compare two ways to condition autoregressive SID generation
on history.

**Implementation.** In the decoder-only model, serialize history followed by
target SID tokens. In the encoder-decoder model, encode history once and decode
the SID autoregressively. Match active parameters, history items, target tokens,
beam width, and training budget. Use item IDs in encoder history as the initial
explicit control.

**Acceptance.** Decoder-only is the control. Select encoder-decoder only after a
Recall@100 gain beyond the band, or non-inferior recall with a measurable
training/serving resource advantage; otherwise retain decoder-only. Nulls or
regressions require causal-boundary, leakage, parameter, and decoding-equivalence
tests plus beam-survival, branching, history-length, and cost analysis with SID
diagnostics.

### RQ2 — How should the encoder describe history?

**Understanding.** Compare the three history representations listed explicitly
for G7's encoder-decoder model; G6 now has a broader seven-way representation
study.

**Implementation.** Compare one projected SID event token, SID+item-ID event
token, and one token per level. Truncate all by history-item count and keep the
decoder/output unchanged. First use one shared tokenizer to isolate the history
representation. Then give each representation an equal independent tokenizer
search over levels, per-level codebook sizes, and RQ-KMeans parameters; these
tuned-system runs determine the final answer.

**Acceptance.** The projected SID event is the reference. Select SID+item-ID or
per-level tokens only if the independently tuned system beats it beyond the
Recall@100 band, or is non-inferior with a measurable resource benefit. Shared-
tokenizer and independently tuned conclusions must agree or their reversal must
be explained through sequence length, representation–tokenizer interaction,
collision/load, and target-survival analysis.

### RQ3 — Causal or bidirectional encoder?

**Understanding.** At serving time the entire past history is available, so a
bidirectional history encoder may use context that a causal pretraining setup
forbids.

**Implementation.** Change only the encoder attention mask in the selected
architecture. Keep query extraction, positions, decoder, and history identical.
Verify that neither mask can see target SID tokens.

**Acceptance.** The causal encoder is the control. Select bidirectional history
attention only if Recall@100 improves beyond the band, or is non-inferior with a
measurable resource benefit; otherwise retain causal attention. Nulls or
regressions require exact-mask/leakage tests and history-length, attention-
distance, prefix-depth, and beam-survival analysis.

### RQ4 — Does next-item encoder pretraining help?

**Understanding.** Test whether a causal SASRec representation is a better
initialization for SID generation.

**Implementation.** Pretrain the selected causal encoder on item-ID next-item
prediction using training data only, then fine-tune the whole generator. Compare
with scratch under matched final-stage budgets. A frozen-encoder arm is a
proposed diagnostic, not a source requirement.

**Acceptance.** Scratch training is the control. Promote pretraining only when
Recall@100 improves beyond the band; an efficiency claim requires non-inferior
recall and lower amortized total pretraining-plus-fine-tuning cost beyond
dispersion. Nulls or regressions require prefix/checkpoint/unfreezing tests and
learning-curve, representation-drift, history, and popularity analysis.

### RQ5 — Does SID-level logQ help?

**Understanding.** Compare marginal code popularity with popularity conditional
on the preceding SID prefix. With a full code softmax this is a popularity
adjustment, not a sampled-softmax proposal correction; call it logQ only when
the codes were actually sampled from that proposal.

**Implementation.** From training targets cache smoothed counts for
`q_l(c) = count_l(c) / sum_c count_l(c)` and
`q_l(c | prefix) = count(prefix, c) / count(prefix)`. Try:

1. no adjustment;
2. marginal per-level adjustment at every SID position;
3. prefix-conditional adjustment at every position;
4. marginal adjustment at level one and prefix-conditional adjustment later;
5. the best definition with probability flooring/clipping.

For a valid code `c`, use `adjusted_logit = logit - beta * log(q)` and tune
`beta`, including zero. Compute conditional probabilities only over children
valid under that prefix. Apply the adjustment during training loss computation
only; constrained-beam inference uses raw model logits, with `beta = 0` as the
unadjusted train/eval control. Unit tests must compare cached counts, support,
and normalization with hand-computed fixtures; this is unit validation rather
than a separate experiment. Keep concrete-item popularity out of this RQ.

**Acceptance.** `beta = 0` is the control. Select a nonzero marginal,
conditional, hybrid, or clipped adjustment only when item Recall@100—not SID
token accuracy—improves beyond the band; otherwise keep raw training logits.
Nulls or regressions require count/support/normalization fixtures and level-,
prefix-frequency-, load-, collision-, and calibration analysis with SID
diagnostics.

### RQ6 — Does item-popularity correction applied to the SID help?

**Understanding.** The full item proposal cannot simply be subtracted at every
SID level, which would count it repeatedly. Adding `-log q(item)` to a loss
containing only the observed positive is also ineffective: it is constant with
respect to model parameters and therefore does not change gradients. A
positive-only version must instead weight the complete sequence loss.

**Implementation.** Compare these concrete arms:

1. no item-popularity correction;
2. no negatives: compute the full positive SID loss
   `L_item = sum_l CE(s_l | s_<l, history)` and multiply it by a normalized,
   clipped weight `(q(item) + epsilon)^(-alpha)`; tune `alpha`, including zero,
   and clipping, and normalize weights to mean one;
3. uncorrected sequence-level sampled softmax: sample complete negative items,
   map them to collision-resolved SID tuples, and score every complete tuple;
4. the identical sequence-level sampled-softmax arm, subtracting the sampled
   item's training-only `log q(item)` exactly once;
5. an inference-only diagnostic that reranks complete item sequences by
   `sum_l log p(s_l | s_<l, history) - beta log q(item)`.

Arm 2 is inverse-popularity loss weighting, not logQ correction. For collisions,
proposal mass belongs to concrete items after resolution rather than being
duplicated across an unresolved tuple. Fix the selected RQ5 treatment across
all arms. Treat arms 1/2 and arms 3/4 as paired contrasts; arm 5 is paired with
raw beam scoring from the identical arm-1 checkpoint and candidate beam. Do not
attribute differences between the positive-only and sampled-tuple objective
families to popularity correction. Verify item-proposal normalization and
one-time correction in deterministic unit tests.

**Acceptance.** Resolve the three matched pairs independently. A correction
qualifies after a Recall@100 gain beyond the paired empirical band; an in-band
delta is no detectable effect and a beyond-band loss rejects it. A predeclared
tail/coverage trade-off requires recall non-inferiority and a secondary gain
beyond its band. Nulls or regressions require weight/proposal/pair-identity tests
and popularity-decile, collision, candidate-oracle, head/tail, and SID analysis;
cross-objective differences are not correction evidence.

### RQ7 — Does reverse SID generation order help?

**Understanding.** The ordinary order predicts coarse-to-fine levels. Reversing
it predicts fine-to-coarse and may expose more discriminative information
earlier, at the cost of a harder first decision and a differently shaped trie.

**Implementation.** First isolate order by training forward and reversed
decoders on the same SID assignment, parameter budget, and beam/candidate
budget, with separately tuned decoder settings. Then give each direction an
equal independent tokenizer search over levels, per-level codebook sizes, and
RQ-KMeans parameters; these tuned-system runs determine the final answer.
Reverse the level vocabulary, target order, prefix constraints, and trie
traversal together; do not merely reverse positional embeddings. Report prefix
and item recall by generated depth, valid-branch counts, beam survival of the
target item, and latency.

**Acceptance.** Forward order is the control. Select reverse order only when the
independently tuned system improves Recall@100 beyond the band, or is
non-inferior with a measurable latency benefit; otherwise retain forward order.
A null, regression, or shared-versus-tuned reversal requires complete reversal/
trie tests and depth-wise branching and target-survival analysis.

### Decoder-only RQ2 — How should history be serialized?

**Understanding.** The decoder-only model needs the same history-representation
comparison as the encoder-decoder model, but the history and target SID share
one causal stream.

**Implementation.** Before a target-boundary token, serialize each history item
as (1) one projected event from concatenated trainable SID embeddings and level
tags, (2) one projected event from concatenated item-ID and SID embeddings, or
(3) one token per SID level. Append the target SID after the boundary and use a
strict causal mask, so no history state can see any target token. Truncate all
arms by history-item count. First compare them with one shared tokenizer, then
give every representation an equal independent search over SID levels,
per-level sizes, RQ-KMeans settings, and its representation-specific model
settings; the independently tuned systems determine the final answer.

**Acceptance.** The projected SID event is the decoder-only reference. Select
another serialization only if its independently tuned Recall@100 improves
beyond the band, or is non-inferior with a measurable sequence-length, memory,
or latency benefit. Nulls, regressions, or shared-versus-tuned reversals require
causal/equal-history tests and truncation, interaction, target-survival,
collision/load, and SID analysis.

### Decoder-only RQ4 — Does next-item pretraining help?

**Understanding.** Test whether item-ID next-item pretraining initializes the
decoder-only history model better than training SID generation from scratch.

**Implementation.** Use the selected decoder-only history serialization and
pretrain the same causal backbone on item-ID next-item prediction using only
training prefixes. Initialize the SID generator from that backbone and
fine-tune every parameter. Compare with a scratch model under the same final
training and tuning budget; retain a frozen-backbone arm only as a diagnostic.

**Acceptance.** Scratch training is the control. Promote decoder-only
pretraining only after a Recall@100 gain beyond the band; an efficiency claim
requires non-inferior recall and lower amortized total cost beyond dispersion.
Nulls or regressions require checkpoint/causal-prefix/unfreezing tests and
learning-curve, representation-drift, history, popularity, and SID analysis.

### Decoder-only RQ5 — Does SID-level logQ help?

**Understanding.** The SID popularity definitions do not depend on having a
separate encoder, but corrections must be applied only at target-SID positions
in the combined causal stream.

**Implementation.** Reuse the exact training-only marginal and
prefix-conditional count definitions from encoder-decoder RQ5. Compare no
adjustment, marginal adjustment at every level, prefix-conditional adjustment,
marginal first level plus conditional later levels, and the selected definition
with probability flooring/clipping. Tune `beta`, including zero. History-token
losses and logits are excluded from the correction, and conditional support is
restricted by the decoder-only trie prefix. Apply the adjustment only while
computing the training loss; constrained-beam inference always uses the raw
model logits. The `beta = 0` arm is the unadjusted train/eval control. An
inference-time SID adjustment would be a separate reranking RQ, not another
interpretation of these runs.

**Acceptance.** `beta = 0` is the decoder-only control. Select a nonzero
adjustment only when item Recall@100 improves beyond the band; otherwise retain
unadjusted training and inference. Nulls or regressions require count/support,
target-mask, and raw-inference tests plus level-, prefix-frequency-, load-,
collision-, calibration-, and SID analysis.

### Decoder-only RQ6 — Does item-popularity correction help?

**Understanding.** Apply concrete-item popularity to the complete target SID
sequence, not once per SID level. As in the encoder-decoder model, an additive
positive-only `-log q(item)` constant cannot change gradients.

**Implementation.** Fix the selected decoder-only RQ5 treatment—including
leaving it off if RQ5 found no benefit—and the history representation across
three paired contrasts:

1. unweighted positive SID loss versus normalized clipped inverse-popularity
   weighting of the identical positive loss;
2. uncorrected complete-item tuple sampled softmax versus the identical sampled
   tuples with item `log q` subtracted exactly once;
3. raw complete-sequence beam scores versus item-popularity reranking of the
   identical checkpoint and generated candidate beams.

Every negative tuple is scored after the same causal history boundary. Do not
attribute differences between positive-only and sampled-tuple objective
families to popularity correction.

**Acceptance.** Resolve each weighted/unweighted, corrected/uncorrected, and
reranked/raw pair independently. Select a correction after a Recall@100 gain
beyond the paired empirical band; an in-band delta is no detectable effect and
a beyond-band loss rejects it. A tail/coverage trade-off requires
non-inferiority and a secondary gain beyond its band. Nulls or regressions
require pair-identity/proposal tests and popularity, collision, oracle,
beam-survival, and SID analysis.

### Decoder-only RQ7 — Does reverse SID order help?

**Understanding.** Reversing coarse-to-fine generation may change the
decoder-only beam in the same way as for encoder-decoder generation, while the
causal history prefix stays unchanged.

**Implementation.** Compare forward and reversed target SIDs after the same
history boundary. Reverse level vocabulary, target positions, trie traversal,
and prefix constraints together. First isolate order with one shared tokenizer;
then independently tune tokenizer and decoder settings for each direction under
equal budgets. Match final beam and candidate budgets.

**Acceptance.** Forward order is the control. Select reversed decoder-only order
only when independently tuned Recall@100 improves beyond the band, or is
non-inferior with a measurable latency benefit. Nulls, regressions, or shared-
versus-tuned reversals require identical-history/reversal/trie tests and depth-
wise branching, target-survival, and SID analysis.

### Dataset-size RQ — Does scale change generative retrieval's gain?

**Understanding.** Larger catalogs change code utilization, collisions, trie
branching, and beam-search difficulty.

**Implementation.** Treat this RQ as its own explicitly approved four-cell
study: item-ID baseline and the frozen selected generator mechanism, each on
native 50M and native 500M. Refit RQ-KMeans and every data-derived artifact on
each size's training catalog. Give baseline and treatment equal size-local model
tuning budgets; do not transfer rates, checkpoints, tokenizer assignments, or
empirical bands across sizes. At each size, select the batch on that size's
unchanged item-ID control with explicit approval, then reuse that batch for its
baseline and treatment; do not transfer the 50M batch to 500M automatically.
Keep the treatment definition and candidate/beam budget rule fixed, and report
within-size treatment deltas plus the change in delta, using separately measured
size-specific bands.

**Acceptance.** Resolve generator-minus-item-ID Recall@100 separately at 50M
and 500M with their size-local bands. Claim scale dependence only when the
difference in those deltas exceeds the preapproved combined resolution;
otherwise report no detectable interaction. Nulls, reversals, or unequal slice
effects require cross-size isolation tests and catalog-load, collision,
branching, oracle, latency, and SID analysis.

## 8. Item-ID and SID outputs in an encoder-decoder model

### Common setup

**Main dataset: native Yambda-50M.** Reuse the selected G7-style encoder and SID
tokenizer as architecture definitions, but fit/train all artifacts on this
group's data. The primary control is that encoder-decoder with its SID decoder
only. Keep the candidate and beam budgets explicit.
Tune levels, per-level codebook sizes, and tokenizer-specific parameters on the
joint G8 downstream validation task under the program-wide per-level limit.
Add a joint model under `dcn/models/` that returns named SID and item outputs so
`LossWrapper` can attach separate criteria and metrics without coupling the
training loop to either head.

### RQ1 — What happens with separate item-ID and SID decoder branches?

**Understanding.** Test both parallel prediction and a genuinely sequential
architecture in which the concrete item is predicted only after its SID has
been generated.

**Implementation.** Compare:

1. SID-only and item-only attribution controls;
2. a shared history encoder with parallel autoregressive-SID and one-token
   full-catalog item-ID decoders;
3. one sequential decoder whose target is
   `[sid_1, ..., sid_L, item_id]`, so the item-ID loss is applied after the SID;
4. a cascaded variant where a one-token item decoder cross-attends to the
   completed SID-decoder states instead of placing item ID in the same token
   stream.

Tune SID/item loss weights for joint models. During training, report sequential
item accuracy both with teacher-forced SID prefixes and with generated SID
prefixes; at inference only the generated path counts. Report each output's
recall, their oracle union, fused ranking, latency, and error recovery when an
earlier SID level is wrong.

**Acceptance.** Compare every joint architecture with the stronger of the
SID-only and item-only controls. Promote it only for a fused Recall@100 gain
beyond the band, or non-inferior recall with a measurable latency or memory
gain. Oracle union is diagnostic and cannot establish a win. Nulls or
regressions require per-head recall, branch complementarity, gradient/loss
balance, teacher-forced versus generated-SID gaps, error recovery, and cost
analysis, plus tests that each loss reaches only its intended head.

### RQ2 — Which logQ definition is correct for the joint model?

**Understanding.** Item-head and SID corrections are separate factors. SID
correction enters this experiment only if G7 showed a benefit outside its
applicable empirical band.

**Implementation.** With architecture fixed, test:

1. neither correction;
2. concrete-item logQ on the item head only;
3. the selected G7 SID correction only, if it passed the prerequisite;
4. both corrections, if arm 3 is eligible.

If G7 selects sequence-level item-popularity weighting rather than SID-level
adjustment, use that exact method on the SID loss; do not rename it SID logQ.
Tune correction strength for every enabled head and the joint loss weight. Do
not spend runs on marginal/prefix SID variants that G7 already rejected.

**Acceptance.** No correction is the control and item-only correction is always
eligible. SID-only and combined correction are eligible only after G7 accepts
SID correction. Promote a combination only when fused Recall@100 improves
beyond its matched no-correction control; claim an interaction only when the
combined effect differs from the two isolated effects beyond the predeclared
interaction resolution. Nulls or regressions require per-head, popularity,
collision, calibration, loss-balance, and gradient analysis. The report must
record the G7 gate and cannot claim an SID-correction result if that gate fails.

### RQ3 — Which head should produce the final ranking?

**Understanding.** Separate candidate generation from ranking: SID likelihood
and item score can order the same reachable items differently.

**Implementation.** Resolve the SID beam to concrete items and rank the exact
same candidate set by accumulated SID likelihood or the item branch. Also
report the candidate-set oracle. An independent union of item-branch candidates
is a separate treatment because it changes recall opportunity.

**Acceptance.** SID-likelihood ranking is the control. Select item-head ranking
only when it improves Recall@100 on the identical resolved candidate set beyond
the band; otherwise retain SID likelihood. A candidate union is judged as a
separate treatment at a fixed final candidate budget. Candidate-set oracle is
diagnostic only. Nulls or regressions require reachability, score calibration,
collision-bucket, popularity, and candidate-survival analysis.

### RQ4 — Does an encoder-decoder help regular item-ID SASRec?

**Understanding.** Replace the current decoder-only causal SASRec—not an
“encoder-only SASRec”—with an encoder-decoder whose decoder predicts exactly
one token: the next item ID.

**Implementation.** The control is the current causal transformer that consumes
history and predicts the next item from its final history state. The treatment
encodes the same history, feeds one learned query/BOS token to a one-position
decoder that cross-attends to all encoder states, and applies the same
full-catalog item loss at that single decoder position. Match active parameters,
history, objective, candidates, and tuning budget. Do not generate a sequence
of future items in this RQ.

**Acceptance.** Causal SASRec is the control. Select the one-token
encoder-decoder only for a Recall@100 gain beyond the band, or non-inferior
recall with a measurable training-time, serving-latency, or memory advantage.
Nulls or regressions require leakage and one-output-position tests plus matched
parameter, optimization, attention-use, history-length, and cost analysis.

### RQ5 — Do parallel forward/reverse SID branches help?

**Understanding.** Forward and reversed SIDs factorize the same item probability
in different orders and may make complementary beam errors. An item-ID head may
add a third, non-factorized signal.

**Implementation.** Using the G7-selected settings for each direction, compare
forward only, reverse only, shared-encoder forward+reverse decoders, and
forward+reverse plus the one-token item-ID head. Tune branch loss weights. At
inference compare each branch, score fusion over the intersection, candidate
union followed by normalized score fusion, and the union oracle. Keep the final
candidate budget fixed so a larger union is not a free recall advantage; report
branch overlap, unique hits, latency, and memory.

**Acceptance.** The stronger forward-only or reverse-only branch is the
control. Promote parallel branches, with or without the item head, only when
realized fused Recall@100 improves beyond the band under the same final
candidate budget, or is non-inferior with a measurable resource advantage.
Oracle union is diagnostic only. Nulls or regressions require branch overlap,
unique-hit survival, reachability, score calibration, loss balance, latency,
memory, and SID diagnostics.

## 9. Improving semantic IDs

### Common setup

**Main dataset: native Yambda-50M.** Use the same input content vectors and
train catalog for all tokenizer families. Frozen tokenizer families feed the
same downstream generator; coupled methods are compared as systems and then
ablated internally. Report reconstruction, ICR, load/entropy, collisions,
semantic cohesion, exact/prefix SID recall, item recall, token count, and beam
latency.
Give every tokenizer family an equal bounded search over levels, per-level
codebook sizes, and its method-specific parameters, selecting on downstream
item recall rather than reconstruction. Enforce the program-wide per-level
limit.
Implement each learned tokenizer as a preceding experiment stage analogous to
`KMeansIdStage`/`RqVaeIdStage`; only explicitly coupled methods may keep the
tokenizer trainable inside the downstream model.

### RQ1 — RQ-KMeans baseline

**Understanding.** Establish the simple residual-clustering reference for SID
quality and downstream generation.

**Implementation.** Random-search number of levels, per-level codebook sizes,
and RQ-KMeans fitting parameters under the shared bound and budget. Every trial
trains/evaluates the common downstream generator; select by item recall, then
freeze its assignments. This is the control for tokenizer improvements.

**Acceptance.** The predeclared default RQ-KMeans configuration is the search
control. Replace it only when a tuned configuration improves downstream item
Recall@100 beyond the band; if quality is non-inferior, prefer a configuration
only for a measurable tokenizer or decoding resource advantage. Deterministic,
training-only fitting, parameter-bound, and frozen-assignment tests are
required. A null or boundary optimum requires load, collision,
reconstruction, cohesion, and beam analysis before the default is retained.

### RQ2 — Does RQ-VAE improve over RQ-KMeans?

**Understanding.** Learn the reconstruction space and residual codebooks jointly
instead of clustering a fixed representation.

**Implementation.** Under an equal search budget, tune levels, per-level
codebook sizes, latent width, reconstruction weight, and commitment settings.
Train/evaluate the same downstream generator for selection, freeze the winning
tokenizer, and compare both intrinsic SID metrics and downstream item recall
with RQ-KMeans.

**Acceptance.** Selected RQ-KMeans is the control. Promote independently tuned
RQ-VAE only when downstream item Recall@100 improves beyond the band, or is
non-inferior with a measurable end-to-end resource advantage; reconstruction
or intrinsic SID gains alone cannot select it. Nulls or regressions require
code utilization, residual/reconstruction, collision, cohesion, target beam
survival, and training-stability analysis.

### RQ3 — Does PLUM or a PLUM modification help?

**Understanding.** PLUM is a complete pretrained-language-model adaptation
pipeline, but the requested modification is specifically its multi-resolution
SID tokenizer. Original PLUM starts with a large codebook and decreases its
cardinality at later residual levels.

**Implementation.** Use one declared pretrained checkpoint, recommendation
fine-tuning corpus, token budget, and downstream evaluator. First run two
attribution controls: standard equal-size RQ-KMeans/RQ-VAE SIDs with direct
recommendation fine-tuning, and the same standard SIDs after PLUM-style
continued domain pretraining. Then reproduce complete PLUM with its tokenizer,
continued pretraining, and recommendation fine-tuning. Within that fixed PLUM
system, compare these codebook-size schedules:

1. **front-heavy/original:** largest first level, monotonically smaller later
   levels;
2. **back-heavy/reverse:** smallest first level, monotonically larger later
   levels;
3. **middle-heavy:** small first and last levels, with the largest codebook in
   the middle.

For front-heavy and back-heavy schedules, use the same multiset of per-level
sizes in opposite order. For middle-heavy schedules, match total codebook
parameters and approximate log-capacity. Tune number of levels, the size
multiset, and schedule steepness on downstream recall under the per-level
limit. Compare the winning tokenizer with an equal-checkpoint PLUM control and
report level utilization and collision structure; do not attribute checkpoint
or continued-pretraining gains to the schedule.

**Acceptance.** Use matched contrasts: standard SIDs plus continued pretraining
versus standard SIDs with direct fine-tuning isolates pretraining; front-heavy
PLUM versus the equal-checkpoint standard-SID arm isolates its tokenizer; and
back- or middle-heavy schedules versus front-heavy isolate schedule order.
Accept each claim only for a downstream item Recall@100 gain beyond its matched
control; intrinsic SID gains alone are insufficient. Nulls or regressions
require checkpoint/corpus equivalence plus per-level attribution, utilization,
collision, reconstruction, cohesion, and beam analysis.

### RQ4 — Does R3-VAE help?

**Understanding.** Test reference-vector guidance, rating-based stabilization,
and SID-quality regularization as an RQ-VAE family.

**Implementation.** Reproduce the full R3-VAE tokenizer, then ablate its three
added mechanisms against matched RQ-VAE. Freeze each selected tokenizer before
the common generator and report stability/collapse as well as final metrics.
Reference: [R3-VAE](https://arxiv.org/abs/2604.11440).

**Acceptance.** Matched RQ-VAE is the control. Promote the tuned full R3-VAE
only for an item Recall@100 gain beyond the band, or non-inferior recall with a
measurable stability or resource benefit. Attribute a component only when its
removal worsens recall beyond the band relative to the full model. Nulls or
regressions require collapse, utilization, optimization stability,
reconstruction, collision, and beam diagnostics.

### RQ5 — Do variable-length BPE SIDs help?

**Understanding.** The source proposes BPE as the mechanism for variable-length
identifiers; a separate non-BPE controller is only a proposed addition.

**Implementation.** Start from fixed base SID tuples, learn BPE merges on the
training catalog only, and map every item to its merged token sequence. Compare
under fixed decoding-latency and candidate budgets, reporting vocabulary size,
length distribution, collisions, and quality.

**Acceptance.** Fixed-length base SIDs are the control. Promote BPE only for an
item Recall@100 gain beyond the band, or non-inferior recall with a measurable
sequence-length or decoding-latency benefit under the fixed candidate budget.
Tests must verify training-only merges, reversible mapping, EOS/PAD handling,
valid generation, and collision preservation. Nulls or regressions require
length-by-frequency, collision, target-survival, and beam-cost analysis.

### RQ6 — Does DIGER's differentiable tokenizer help?

**Understanding.** Let recommendation gradients update SID assignments while
preventing early codebook collapse.

**Implementation.** Jointly train tokenizer and generator with Gumbel
exploration and a declared uncertainty-decay schedule. Compare with the same
architecture using a frozen tokenizer and ablate exploration/decay. Report code
utilization throughout training. Reference: [DIGER](https://arxiv.org/abs/2601.19711).

**Acceptance.** The identical generator with a frozen tokenizer is the control.
Promote full DIGER only for an item Recall@100 gain beyond the band, or
non-inferior recall with a measurable resource benefit. An exploration or
decay mechanism is supported only when its ablation worsens recall beyond the
band. Tests must verify tokenizer gradients, Gumbel behavior, schedule
completion, and frozen inference. Nulls or regressions require utilization,
assignment churn, collapse, uncertainty, stability, and beam analysis.

### RQ7 — Does a collision token help?

**Understanding.** Append exact disambiguation only to semantic tuples shared by
multiple items.

**Implementation.** Hold the base tokenizer fixed and compare unresolved tuples
with collision suffixes. Report exact item resolution, suffix vocabulary/load,
dynamic-catalog implications, decoding cost, and item metrics.

**Acceptance.** Unresolved base tuples are the control. Promote collision
suffixes for a Recall@100 gain beyond the band; alternatively accept them as a
trade-off when recall is non-inferior and concrete-item ambiguity is eliminated
without a measurable latency or memory regression. Nulls or regressions require
collision-bucket, tail-item, suffix-load, reachability, dynamic-catalog, and
decoding-cost analysis plus exact-resolution tests.

### RQ8 — Does Purely Semantic Indexing improve collision resolution?

**Understanding.** *Purely Semantic Indexing for LLM-based Generative
Recommendation and Retrieval* makes item IDs unique without appending a
non-semantic collision token. When a nearest-centroid tuple is already used, it
allows another nearby semantic tuple instead. The paper proposes exhaustive
candidate matching (ECM) and the cheaper recursive residual searching (RRS).

**Implementation.** Starting from one selected residual tokenizer, compare its
ordinary nearest-centroid assignments plus collision suffix with PSI-ECM and
PSI-RRS assignments without a suffix. For ECM, enumerate the top-k centroid
choices at each level for one item, score and sort its complete Cartesian-product
tuples by the paper's cumulative residual criterion, and assign that item's
first unused tuple. For RRS, search centroid candidates recursively using the
updated residual and backtrack until that item's first unused complete tuple is
found. Both algorithms mutate the shared used-ID set and are item-order
dependent: use one declared deterministic order for the main comparison and
report sensitivity across several fixed training-item permutations. Restrict
ECM to candidate widths whose product is tractable. Tune the candidate-width
vector and shared tokenizer parameters on downstream item recall.
Report uniqueness, displacement from the nearest tuple, reconstruction,
semantic cohesion, assignment runtime/memory, and generator item recall under
the same beam budget. This is 50M-only by explicit scope decision; its result
must not be generalized to 500M from the separate RQ-KMeans scale study.
Reference: [Purely Semantic Indexing](https://arxiv.org/abs/2509.16446).

**Acceptance.** The nearest-tuple tokenizer with collision suffixes is the
control. Promote ECM or RRS only for an item Recall@100 gain beyond the band;
accept RRS as an efficiency trade-off only with non-inferior recall and a
measurable assignment-time or memory gain. Both must reproduce paper-style
allocation on fixed fixtures and produce unique suffix-free assignments.
Nulls or regressions require displacement, reconstruction, cohesion,
order-sensitivity, failure-rate, assignment-cost, and beam analysis.

## 10. Semantic IDs with collaborative information

### Common setup

**Main dataset: native Yambda-50M.** The control is the selected content-only
tokenizer trained with the same downstream generator. All collaborative pairs,
transitions, and user states use training data only.
Tune each content/collaborative tokenizer's levels, per-level codebook sizes,
and method-specific parameters on G10 downstream recall with equal bounded
budgets and the program-wide per-level limit.
Add collaborative artifact-building as a preceding stage and feed its frozen
item representation into the existing semantic-ID stages.

### RQ1 — Does QARM-like alignment help?

**Understanding.** Adapt content representations toward the interaction
objective before quantization so codes reflect both semantics and behavior.

**Implementation.** Pin the intended QARM component from the reference before
coding. A proposed local realization projects frozen content toward a
recommendation-trained item space with contrastive alignment. Compare content
only, collaborative only, and aligned content+collaborative inputs before the
same quantizer. Reference direction: [QARM V2](https://arxiv.org/abs/2602.08559).

**Acceptance.** Content-only and collaborative-only tokenizers are both
controls. Promote aligned content+collaborative input only when downstream item
Recall@100 beats both beyond their bands. A cheaper representation is acceptable
only with recall non-inferior to the stronger control and a measurable pipeline
resource gain. Nulls or regressions require alignment-quality, neighbor,
utilization, collision, leakage, popularity, and tail-slice analysis.

### RQ2 — Do interaction pairs improve RQ-VAE?

**Understanding.** Add pairwise behavioral proximity to the reconstruction
objective, but `pairs` is underspecified in the source.

**Implementation.** Obtain approval for one pair definition. A proposed first
version uses next-item transitions and co-listens within a fixed training-only
window with popularity-matched negatives. Add a contrastive pair term to
RQ-VAE, tune its weight, and compare with matched RQ-VAE.

**Acceptance.** Matched RQ-VAE without the pair term is the control. Promote a
pair definition and loss weight only when downstream item Recall@100 improves
beyond the band; pair separation or intrinsic SID gains alone are insufficient.
Nulls or regressions require positive/negative-pair fixtures and loss-weight,
pair-separation, collision/load, reconstruction, tail-item, and training-
feasibility analysis.

## 10A. User-aware semantic-ID tokenization

### Common setup

**Main dataset: native Yambda-50M.** This is a separate experiment because a
user-aware code is a representation of a user-item pair, not one globally
cacheable item identifier. Construct supervision and all aggregate features
from training data only. Use a frozen upstream retriever to materialize one
candidate list for every training prefix and the fixed evaluation candidate
lists; all arms see the same lists. Fit representation learners and tokenizer
codebooks only on training-prefix pairs (positives plus the shared retrieved
negatives), freeze them, and only then transform validation/test pairs. The
control tokenizes the candidate's global content vector. Tune tokenizer
parameters separately for each pair representation on downstream reranking
recall under the program-wide limit.

Every user/item counter, listen/action aggregate, and temporal feature is
computed as of that prefix timestamp; static catalog attributes are the only
features allowed to use their full training-catalog value. Build training-pair
vectors out of fold: fit the representation learner on the other folds, encode
the held-out fold for tokenizer fitting, then refit the selected learner on all
training data only to transform validation/test pairs. This prevents the pair's
own target or later events from entering its tokenizer vector.

The downstream scorer receives the encoded history plus the candidate's SID
level embeddings, combines them with the same DenseNet, and outputs one
candidate relevance logit. Train it with the same listwise/sampled-softmax
objective on the training candidate lists; compare user-aware versus global
codes with matched scorer capacity and candidates. Report both the fixed-list
oracle and reranked recall. Implement each representation learner and tokenizer
as preceding stages and freeze them before scorer training. Report the cost of
computing and tokenizing every user-candidate pair. A later generative-index
study needs a separate design because there is no single global trie.

### RQ1 — Which user-item representation produces the best user-aware SIDs?

**Understanding.** Compare a simple non-trained user summary with two
like-supervised pair encoders. The monolithic network may learn strong crosses;
the split network forces user/context and item vectors to remain independently
meaningful and may therefore yield a more representative concatenation for
tokenization.

**Implementation.** Compare these representations:

1. **Summed history.** Define the user vector as the masked sum of history-item
   embeddings, with a mean-normalized diagnostic to control for history length.
   Concatenate it with the candidate item embedding and project to the tokenizer
   input dimension. No future event may enter the summary.
2. **Monolithic pair network.** Train DCNv2 with a DenseNet deep branch on
   binary like probability. Inputs include the user/history representation,
   candidate item and album IDs, artist/category and other available categorical
   features, user/item counters, listen/action aggregates, and applicable
   temporal/context features. Extract the last shared hidden layer before the
   like head as the pair vector. Include a capacity-matched DenseNet without
   explicit DCN crosses as an attribution control.
3. **Split user/item network.** Train
   `sigmoid(dot(f_user(user, context_without_candidate_item), f_item(item)))`
   on the same like target. The DCNv2 user tower receives history and context
   features but neither candidate item ID nor candidate content. The DCNv2 item
   tower receives item ID/content, album/artist/category, and item counters.
   Normalize equal-width tower outputs and tokenize their concatenation. Also
   evaluate each tower alone to determine which half carries the gain.

Fit every vocabulary, counter, and label transform on training data. Tune the
two supervised encoders on held-out like log loss/AUC, freeze their selected
checkpoints, then independently tune each semantic tokenizer on downstream
recall. Compare all pair representations and the global content-only code on
the identical frozen candidates and matched final vector dimensions.

**Acceptance.** The global content-code scorer is the control. Promote a
user-aware representation only when fixed-list reranked Recall@100 improves
beyond the band on identical candidates; like AUC, SID metrics, and across-user
code variation are diagnostics, not selection metrics. A simpler pair
representation may win with non-inferior recall and a measurable materialization
or serving-cost advantage. Tests must prove prefix-causal aggregates,
out-of-fold training-pair vectors, and train-only tokenizer fitting. Nulls or
regressions require missingness, feature/tower attribution, candidate slices,
code variation, collision/load, and latency/cache analysis. If materializing
user-candidate codes exceeds the approved resource budget, report the measured
bottleneck and mark the RQ blocked rather than claiming a quality result.

## 11. Gryphon

### Common setup

**Main dataset: native Yambda-50M.** Fit a fresh residual-K-Means tokenizer on
the 50M training catalog. Match active parameters and inference budget between
vanilla generative retrieval and Gryphon. The companion RQ below is where its
large-catalog claim is tested on 500M.
Tune the shared tokenizer's levels, per-level codebook sizes, and fitting
parameters on downstream item recall under the program-wide limit, then freeze
the same selected assignments for vanilla GR and Gryphon.
Implement Gryphon as a `SemanticGenerationExperiment` variant with a shared
history encoder, existing constrained decoder, and a new item-scoring module;
the generation callback must retain resolved beam candidates for rescoring.

### RQ1 — Does Gryphon improve item-level retrieval?

**Understanding.** Gryphon shares one history encoder between SID generation
and an Item-Level Scoring Module (ILSM). Beam search supplies reachable items;
ILSM reranks those same concrete items and separates collisions.

**Implementation.** Reproduce vanilla GR, collision-resolved GR, and Gryphon.
Use the same RQ-KMeans setup for vanilla GR and Gryphon, replace comparable
decoder capacity with the lightweight item scorer, and train generation plus
sampled-softmax item loss with a tuned weight. Retain each beam candidate set
and score it both by SID likelihood and ILSM. The faithful comparison does not
add independently retrieved item candidates. Report candidate oracle, item
recall, collision ordering, encoding/beam/reranking latency, memory, and
parameters. Reference: [Gryphon](https://arxiv.org/abs/2606.08604).

**Acceptance.** The stronger matched vanilla or collision-resolved GR arm is
the control. Promote Gryphon only when item Recall@100 improves beyond the band,
or is non-inferior with a measurable serving-latency or memory advantage;
candidate oracle alone cannot establish a win. Nulls or regressions require
candidate-oracle, target beam-survival, collision-ordering, item-loss-weight,
capacity, and end-to-end cost analysis.

### Dataset-size RQ — Does scale change Gryphon's gain?

**Understanding.** This is especially important for Gryphon because trie
branching, SID collisions, and beam-score accumulation grow with catalog scale.

**Implementation.** Refit and tune fresh residual-K-Means tokenizers and matched
vanilla-GR/Gryphon models independently at 50M and 500M. Freeze the Gryphon
mechanism and candidate budget definition; do not reuse 50M assignments or
rates. Compare within-size Gryphon-minus-vanilla deltas plus collision and beam
diagnostics. This replaces the earlier assumption that Gryphon should simply
start on 500M.

**Acceptance.** Resolve Gryphon-minus-vanilla Recall@100 separately at 50M and
500M using size-local bands. Claim a scale effect only when the difference
between those two deltas exceeds the preapproved combined resolution; otherwise
report no detectable interaction. Nulls, reversals, or unequal slice effects
require size-local tokenizer and training checks plus collision, trie-branching,
target-survival, candidate-oracle, latency, and memory analysis.

## 12. Diffusion over item content embeddings

### Common setup

**Main dataset: native Yambda-50M.** Freeze normalized content/audio targets for
the train-mapped catalog. All arms use the same history encoder, nearest-neighbor
catalog index, candidate count, and inference accounting.
Add continuous target heads under `dcn/models/` and a catalog-nearest-neighbor
evaluation path under `dcn/eval/`; do not force continuous outputs through the
semantic-token trie.

### RQ1 — Can diffusion generate a useful item embedding?

**Understanding.** Generate a continuous vector conditioned on history, then
retrieve the nearest catalog items instead of predicting a discrete ID/SID.

**Implementation.** Train latent diffusion to denoise corrupted positive-item
embeddings conditioned on the user representation. Compare with deterministic
direct embedding regression at matched capacity. Sweep a bounded number of
denoising steps and samples, normalize outputs before retrieval, and report
recall, target distance, duplicate rate, coverage, and end-to-end latency.

**Acceptance.** Matched one-shot embedding regression is the control. Promote
diffusion only for item Recall@100 improvement beyond the band, or non-inferior
recall with a measurable coverage, duplicate-rate, or latency advantage.
Target distance alone cannot select it. Nulls or regressions require denoising-
trajectory, sample-collapse, popularity, target-distance, candidate-oracle,
coverage, and end-to-end compute analysis; an over-budget method is reported as
infeasible at the measured step/sample setting.

### RQ2 — Can a transformer predict continuous residuals step by step?

**Understanding.** This is a continuous analogue of SID decoding without a
fixed quantizer.

**Implementation.** Fit PCA on training-item embeddings, partition its ordered
components into variance-balanced blocks, and inverse-transform each block into
one additive continuous residual. In addition to that baseline, try:

1. **learned stagewise residuals:** separate heads predict the remaining target
   error after each previous head, with deep supervision on every partial sum;
2. **shared recurrent refinement:** one weight-shared block repeatedly predicts
   a correction to the current vector, with the step embedding indicating
   refinement depth;
3. **masked coordinate blocks:** partition the continuous target dimensions,
   randomize block order during training, and autoregressively fill masked
   blocks conditioned on observed/predicted blocks;
4. **continuous latent pyramid:** train a non-quantized multi-resolution
   autoencoder and predict its coarse latent followed by finer additive latent
   corrections;
5. **rectified-flow/flow-matching head:** learn a continuous trajectory from
   noise or the history-conditioned initial estimate to the target embedding,
   using a small fixed number of solver steps.

Tune step count and loss weights for each family. Compare them with diffusion
and one-shot regression under equal parameter, candidate, and declared
inference-step budgets; normalize the final vector before nearest-neighbor
retrieval and include every refinement/solver step in latency.

**Acceptance.** The stronger of matched one-shot regression and diffusion is
the control. Promote a residual/refinement family only for an item Recall@100
gain beyond the band, or non-inferior recall with a measurable latency, memory,
coverage, or duplicate-rate advantage. Partial-step target distance is a
diagnostic, not the decision metric. Nulls or regressions require family-
specific stability, partial-step error, collapse/divergence, candidate-oracle,
and complete inference-cost analysis.

## 13. Reinforcement learning for multi-step ranking

### Common setup

**Main dataset: native Yambda-50M.** Data scale is secondary to evaluation
validity. Restrict actions to reranking a fixed candidate set. No training claim
is accepted without logged propensities or a separately validated simulator.
Build this on `RankingExperiment`/`RankingWithHistoryExperiment`, with policy
objectives and off-policy estimators as separate reusable modules. Do not alter
the logged dataset in place to simulate counterfactual feedback.

### RQ1 — Does multi-step RL improve ranking?

**Understanding.** Optimize a trajectory reward rather than imitate the next
logged action, while accounting for policy-induced state changes.

**Implementation.** Define state as history plus prior slate feedback, action as
a reranked fixed slate, reward as the next-step like/listen value, and horizon
as five decisions. Compare behavior cloning, a supervised listwise ranker, and
a conservative offline-RL objective. The current Yambda events contain neither
logged slates nor behavior-policy propensities, so they cannot train or evaluate
this treatment causally. Implementation therefore starts only after obtaining
logged slate/propensity data or a separately validated action-conditional user
simulator; use IPS/DR and effective sample size for the former, and label the
latter as simulator-only evidence. Until then this group is scientifically
blocked, not merely waiting for a model class.

**Acceptance.** This RQ remains blocked until logged slates and propensities or
a validated action-conditional simulator satisfy the evaluation gate. After
that gate, promote offline RL only when estimated multi-step value exceeds both
behavior cloning and the supervised ranker beyond estimator uncertainty while
one-step ranking quality is non-inferior. Report overlap, effective sample size,
clipping sensitivity, IPS/DR disagreement, and policy shift; simulator results
remain explicitly simulator-only and cannot establish an offline causal win.

## 14. Thinking over history

### Common setup

**Main dataset: native Yambda-50M.** Keep the final item objective fixed and
require equal-parameter and equal-latency controls so extra compute is not
misnamed reasoning.
Implement the approved mechanism as a history-encoder variant and expose its
intermediate state explicitly for utilization diagnostics; keep retrieval and
full-catalog evaluation unchanged.

### RQ1 — Does thinking based on history help?

**Understanding.** The source does not define “thinking.” The proposed concrete
reading is iterative latent refinement: a small set of latent intent states
repeatedly reads the fixed history before producing the retrieval query.

**Implementation.** Append four learned latent states after the encoded history
and refine them for four steps with one weight-shared cross-attention/FFN block;
pool the final states into the ordinary item-retrieval query and train only with
the unchanged next-item objective. Compare with the baseline, an
equal-parameter unshared deeper transformer, and an equal-latency repeated
block without latent states. Report recall, state diversity/collapse, stepwise
query change, and latency. This proposed definition requires plan approval, but
the treatment itself is no longer unspecified.

**Acceptance.** The latent-refinement model must beat both equal-parameter and
equal-latency controls in Recall@100 beyond the band to support a “thinking”
mechanism claim; non-inferiority alone is insufficient. An efficiency claim is
allowed only with non-inferior recall and a measurable resource gain. Nulls or
regressions require latent-state diversity/collapse, stepwise query-change,
gradient/use, history-length, and matched-control analysis.

## Remaining ideas

### Additional features for SID construction

**Understanding.** Determine which content, collaborative, temporal, or action
features improve both code quality and retrieval.

**Implementation.** On 50M, add one feature family at a time before
quantization, and give every feature family the same bounded downstream search
over levels, per-level codebook sizes, and method-specific parameters. Select
on item recall and report intrinsic SID metrics as diagnostics.

**Acceptance.** The feature-free or content-only tokenizer is the control for
each added feature family. A correct feature integration must be non-inferior
in downstream item Recall@100; promote it only when recall improves beyond the
band, because intrinsic SID improvements alone cannot select it. An in-band
result resolves the RQ as no detectable benefit but does not justify retaining
the extra feature by default. A regression is acceptance-ready only after
missingness, normalization, leakage, feature-use, utilization, collision,
subgroup, and materialization-cost analysis verifies that the feature was
correctly active and explains the loss.

### Muon or another optimizer

**Understanding.** This is an optimizer RQ, not a semantic-ID method.

**Implementation.** After fixing a model/objective, tune each optimizer with an
equal budget and compare convergence, final quality, time, and memory on that
experiment's main size.

**Acceptance.** The current tuned optimizer is the control. Replace it only for
a final Recall@100 gain beyond the band, or non-inferior recall with a measurable
time-to-best, total training-time, or peak-memory improvement, including tuning
cost. Nulls or regressions require learning curves, search-boundary, seed
stability, failure-rate, best-checkpoint, and step-versus-wall-clock analysis.

### DCNv2 with DenseNet deep part

**Understanding.** Establish a non-sequential ranking/reranking control over a
fixed candidate set.

**Implementation.** Use DenseNet for the deep branch, keep explicit cross
features in DCNv2, and compare with a capacity-matched MLP ranker on 50M.

**Acceptance.** The capacity-matched MLP is the control. Retain explicit DCNv2
crosses only for a Recall@100 gain beyond the band, or non-inferior recall with
a measurable latency or memory advantage. Nulls or regressions require
candidate-oracle, calibration, feature-slice, cross-contribution, parameter-
capacity, and cost analysis.

### DCNv2 plus a history transformer

**Understanding.** Test whether sequential context adds information beyond
explicit crosses.

**Implementation.** Feed the selected history-transformer representation into
the fixed DCNv2 ranker and compare with DCNv2 alone under the same candidates.

**Acceptance.** DCNv2 without sequential history is the control. Add the
history transformer only when Recall@100 improves beyond the band; additional
compute without a quality gain is rejected. Nulls or regressions require
history-length and order-shuffle slices, candidate oracle, representation-use,
capacity, training-time, latency, and memory analysis.

### Transformer pretraining for ranking

**Understanding.** Separate useful initialization from a permanently frozen
representation.

**Implementation.** Compare scratch, pretrained-and-fine-tuned, and
pretrained-frozen history encoders with the same downstream ranker on 50M.

**Acceptance.** Training the same encoder from scratch is the control. Promote
pretrained-and-fine-tuned initialization only for a Recall@100 gain beyond the
band. A pretrained variant may qualify as an efficiency trade-off with
non-inferior recall and lower amortized total cost, including pretraining;
frozen pretraining is an attribution arm, not the default winner. Nulls or
regressions require representation-drift, domain-mismatch, optimization,
convergence, checkpoint, and total-cost analysis.

## Ambiguities that still block plans

PSI is now resolved as *Purely Semantic Indexing for LLM-based Generative
Recommendation and Retrieval*. Proposed completions remain explicitly marked
for the empty G5 RQ4.3, decoder-only G7 carry-over, continuous-residual G12 RQ,
and G14 thinking prompt. Before implementation or training, each experiment
still needs an approved plan with exact configurations, run counts, and
selection rules.
