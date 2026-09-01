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

An RQ is accepted when every planned arm is implemented, trained to the valid
stopping condition, evaluated, and reported with a decision. Unless stated
otherwise, Recall@100 delta `d` and band `b` decide quality: improvement if
`d > b`, null if `|d| <= b`, and regression if `d < -b`. A cross-size effect
must exceed `b_small + b_large`. “Best” and “stronger” mean validation-selected,
with ties going to the simpler model. Non-inferiority selects a treatment only
for an explicitly stated efficiency trade-off. Regressions and surprising
nulls require focused correctness checks and a short analysis. Unexpected-result
explanations must satisfy the empirical-evidence rule in
`experiments/AGENTS.md`; prose alone is insufficient. All standard rules there
still apply.

## 1. SASRec over item IDs and likes

### Common setup

**Main dataset: Yambda-500M.** The calibration range, global batch, and
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

**Acceptance.** Compare the combination with both the original baseline and the
strongest included treatment. Also report its interaction gap against summed
component gains.

### RQ3 — What is the best quality/performance balance?

**Understanding.** This asks for a Pareto choice, not the configuration with
the second-highest recall.

**Implementation.** Benchmark measured G1 candidates under one hardware
procedure. Record recall@100, epoch time, full-catalog inference latency, peak
accelerator memory, and parameter count. Remove dominated points, then select a
cheaper configuration under an explicit maximum recall loss. The selected point
must be a complete configuration and is retuned as such.

**Acceptance.** Select a Pareto-nondominated point. A cheaper point may trade up
to the predeclared maximum Recall@100 loss for improvement in at least one
declared resource metric.

### RQ10 — Do per-layer item embeddings help?

**Understanding.** The intended mechanism is Gemma-style Per-Layer Embeddings,
not an invented concatenation inside the FFN.

**Implementation.** First pin lookup, projection, sharing, and injection
semantics from the released Gemma implementation. Compare the ordinary shared
input table with layer-specific item lookups and a proposed parameter-matched
wider/deeper control. Tune the treatment family fairly and report quality,
total parameters, active accelerator parameters, lookup bandwidth, memory, and
latency. Reference: [Gemma PLE](https://ai.google.dev/gemma/docs/gemma-3n).

**Acceptance.** A PLE gain must beat both the shared-table and
parameter-matched capacity controls. Beating only the smaller control is a
capacity result, not evidence for PLE.
s
### RQ12 — Which decoder-only query-token layout works best?

**Understanding.** This isolates whether candidate generation benefits from a
learned query state or repeated autoregressive query supervision. The
interleaved arm uses one shared CLS token between items, before each following
item.

**Implementation.** Compare standard item-state querying, one end-only CLS,
and the autoregressive interleaved layout `[item1, CLS, item2, CLS, ...]`, where
each CLS state predicts the following item. Preserve eligible native-500M
results for the
standard, one-CLS, and interleaved arms after verifying that their data,
architecture, objective, and evaluation protocol match the new comparison.
Give genuinely new architecture families equal tuning budgets.

**Acceptance.** Compare every layout by final candidate-generation quality,
with Recall@100 primary. In a separate efficiency table report training
throughput, processed examples and supervised targets to the best checkpoint,
and wall-clock time to that checkpoint.

### RQ13 — Does bounded prefix expansion help an encoder-decoder model?

**Understanding.** Prefix expansion turns several transitions from one user
history into separate encoder-decoder examples. Fixed caps of 8 and 16 prevent
highly active users from dominating the training set. The comparison also
separates allowing short truncated prefixes from requiring a full-length
encoder input.

**Implementation.** The required retained-history length is 128. Compare:

1. no expansion: one final history-to-next-item example per user;
2. truncated-prefix expansion: the latest 8 or 16 eligible chronological
   prefixes, retaining at most the last 128 events and allowing shorter
   prefixes; and
3. required-length expansion: the latest 8 or 16 prefixes with at least 128
   preceding events, each retaining the last 128; when the user's complete
   history is shorter than 128, emit exactly one whole-history example instead.

Keep the encoder-decoder architecture and downstream objective identical.
The two caps are reported independently and are not a search axis.

**Acceptance.** Compare final candidate-generation metrics separately from
training efficiency. Because the arms create different example counts, report
both original users and generated prefix examples, histories and targets per
second, processed examples and targets to the best checkpoint, and wall-clock
time to that checkpoint.

### RQ14 — Which decoder-decoder query memory works best?

**Understanding.** The first causal decoder appends four query slots after the
history. Distinct tokens may learn specialized summaries; exposing the history
states as well as the four query states lets the second decoder recover details
that the summaries omit.

**Implementation.** Run a two-by-two comparison. The four appended tokens are
either four copies of one shared CLS token or distinct `CLS_0` through `CLS_3`.
The second decoder cross-attends either only the four resulting CLS states or
the CLS states concatenated with the first decoder's history states. Hold the
training method at joint downstream-only training from scratch so this RQ
changes only token identity and cross-attention memory.

**Acceptance.** Resolve the shared-versus-distinct and CLS-only-versus-CLS-plus-
history effects using final candidate-generation metrics. Report training
throughput, processed examples and targets to the best checkpoint, and
wall-clock time to that checkpoint in a separate efficiency table.

### RQ15 — How should the distinct-CLS decoder-decoder model be trained?

**Understanding.** This asks whether next-item supervision makes the first
decoder a better history representation before or during downstream training.
It does not repeat the shared-CLS arms because the intended contrast is the
incremental benefit for four specialized query slots.

**Implementation.** Use four distinct CLS tokens and the better CLS-only or
CLS-plus-history memory selected in RQ14. Compare:

1. joint training from scratch using only the second decoder's final
   candidate-generation loss;
2. first-decoder NTP pretraining followed by joint fine-tuning of both decoders
   using only the final candidate-generation loss; and
3. joint training from scratch using the final candidate-generation loss plus
   an auxiliary NTP loss on the first decoder.

In the second arm, downstream gradients continue to update the first decoder;
only its NTP loss is absent during fine-tuning. Pretraining cost is included in
the arm's total training cost. The two losses in the third arm are averaged
separately and initially combined with NTP weight 1.0; weights 0.1 and 0.3 are
conditional corrections only if the initial auxiliary arm regresses.

**Acceptance.** Compare final candidate-generation metrics in one table and
training efficiency in another. The efficiency accounting includes every
pretraining and fine-tuning stage and reports throughput, processed examples
and targets to the selected checkpoint, and total wall-clock time.

## 2. eSASRec

### Common setup

**Main dataset: Yambda-500M.** Use the size-calibrated G1 structural
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

**Acceptance.** Pass forward/loss/gradient parity against the official
implementation. Report matched effects for LiGR, loss type, sampler mixture,
and logQ, then compare the selected system with G1.

## 3. Pretrained item embeddings

### Common setup

**Main dataset: Yambda-500M.** Compute, normalize, version, and freeze the
provided content/audio vector for every train-mapped item. The control is the
best G1 combination with its depth increase removed: two layers, tied learned
item IDs, SwiGLU-192, deep-only one-cycle cosine with 5% warmup, ALiBi plus forward/reverse
learned positions concatenated to the item representation, post-norm blocks,
input/final RMSNorm, end-only CLS, 32-bin additive time plus raw reverse
timestamp RoPE, popularity global-Q negatives, GQA, and BOS. G3 retunes it at
batch 512 without MuTransfer or μP and does not reuse its G1 metrics. Input and
target are studied in RQ1/RQ2; RQ3 fixes the concatenated RQ2 input while
changing the prediction space.
Use `PrecomputedEmbeddingLookup` for frozen vectors, add composition encoders
under `dcn/nn/`, and keep query/catalog encoders exposed through the existing
full-catalog evaluator.

Every content-consuming branch uses an L2-normalized content representation
before concatenation or projection. Frozen rows are checked and normalized at
lookup; trainable copies are normalized on every forward pass. Learned
model-width outputs remain unnormalized unless an RQ explicitly changes that
axis.

The frozen two-layer G1-best tied learned-ID experiment baseline is the first
row and percentage reference for every RQ1-RQ5 primary reader comparison,
promotion decision, and overall or slice table. Untied or predecessor controls
may remain only as explicitly requested secondary mechanism diagnostics; they
never replace that baseline.

Retain automatically available runtime and resource telemetry for sanity; do
not add dedicated performance benchmarks. Omit it from reader, compact, and tuning tables unless the user
explicitly asks about performance or training budget, or a material anomaly is
needed to explain validity. Selection uses the requested quality metrics and
then deterministic order, never performance by default.

### RQ1 — What happens when pretrained embeddings replace item IDs?

**Understanding.** This isolates content-only history representation from
collaborative item memorization.

**Implementation.** Replace the learned input lookup with the normalized fixed
pretrained vector followed by a learned projection to model width. Keep the learned
item-ID output target fixed. Use the existing bias-free linear 128-to-64
projection; projection-family capacity is not an RQ1 axis. Tune both learning
rates and the schedule horizon while holding the transformer/backbone fixed,
and report overall plus item-frequency slices against the G1-best baseline.
Retain one tuned untied learned-ID input/output arm only as the secondary
mechanism control needed to separate content replacement from lost weight
tying; it never replaces the G1-best primary baseline.

**Acceptance.** Compare content-only input with the two-layer G1-best tied
learned item IDs. Report overall and head/mid/tail results; a slice-only win
requires aggregate non-inferiority to that baseline.

### RQ2 — Does concatenating content and item ID help?

**Understanding.** Test whether content generalization and item-specific
collaborative information are complementary.

**Implementation.** Concatenate learned item-ID and normalized frozen content
vectors and encode them with DenseNet before the transformer. Compare with the two-layer
G1-best tied learned-ID baseline. Tie the history learned-ID branch to the
learned catalog target so the treatment does not confound content with weight
untying. Tune encoder capacity and relevant rates without changing the negative
objective. No parameter-matched item-ID-only DenseNet is active or required.

**Acceptance.** Concatenation must beat the two-layer G1-best tied learned
item-ID baseline to establish content complementarity.

### RQ3 — Which prediction embedding is best?

**Understanding.** The alternatives are learned item-ID embedding, pretrained
embedding, and their concatenation as retrieval targets. Concatenation is one
joint target space, not automatically multi-task learning.

**Implementation.** Feed every history item as
`DenseNet(concat(learned_item_id_embedding, pretrained_embedding))`, using the
selected RQ2 encoder. Give this history encoder one independent learned-ID table
shared by all five arms so its input remains identical while the catalog target
changes. Try these full-catalog output tables:

1. a learned item-ID embedding table initialized randomly;
2. the frozen pretrained table followed by a learned projection;
3. a trainable copy initialized from the pretrained table, followed by the same
   projection;
4. `concat(learned_item_id, frozen_pretrained)` followed by a learned shared
   projection;
5. the same concatenation with a trainable pretrained copy.

Match final output dimension, use raw dot-product scoring without output
normalization, and give each family the same tuning budget. This separates
target identity, initialization, and whether the content component remains
frozen. Normalize frozen and trainable content rows before their learned
projection, including after every trainable lookup. Variant 1 is the local control; an output treatment is aggregate-eligible
only when it improves variant 1 marginally.

Every primary reader metric and promotion decision references the two-layer
G1-best tied learned-ID baseline. The explicitly requested comparisons among
variants 1-5 remain secondary scientific acceptance evidence and are not
replaced by the baseline comparison.

**Acceptance.** Compare every output target with the learned item-output table.
Use paired contrasts to separate target type, pretrained initialization, and
freezing. I think the expected result is that variant 4 will be the best, but I mey be very wrong. You must explain why your results are expected (if results differs from my expectations). Not just by words, but include experimental proof: plots/ gradient norms etc. And variant 4 should be better then 1 or 2. Also variant 3 should not be much worse then 1 and 2 and most probably better.

### RQ4 — Do artist and album features help?

**Understanding.** Artist and album identities may share evidence across
related and tail items beyond content and collaborative item IDs.

**Implementation.** Start independently from the common two-layer G1-best tied
learned-ID baseline. Add artist only, album only, and artist plus album through
train-only compact vocabularies, mean pooling, concatenation, and DenseNet
projection. Use one shared metadata encoder for history and catalog so the
baseline's item-ID tying is preserved. Tune metadata width and relevant learning
rates/horizon while holding the backbone fixed. Do not include RQ1, RQ2, RQ3,
or RQ5 treatments during RQ4 selection and do not create a parameter-matched
extra item-ID control. Combine selected metadata with other methods only during
the final aggregate stage.

**Acceptance.** Artist and album features should improve tail metrics and
should not make overall Recall@100 worse than the two-layer G1-best tied
learned-ID baseline.

### RQ5 — Does a frequency-adaptive content gate help?

**Understanding.** A learned gate may rely more on content for sparse items and
more on collaborative identity for frequent items.

**Implementation.** Compare fixed concatenation, a learned global gate, and a
train-frequency-conditioned gate. Put the two-layer G1-best tied learned-ID
baseline first in the primary reader comparison and use it for promotion and
percentage deltas. Retain fixed-versus-global-versus-frequency comparisons as
the explicit mechanism diagnostic. Reuse the selected RQ2 embedding learning
rate for every RQ5 run; tune only deep learning rate, horizon, and frequency-gate
width. A learned gate must improve its fixed or global predecessor marginally
before it can enter the aggregate.

**Acceptance.** The frequency-adaptive gate should improve tail Recall@100 and
should not make overall Recall@100 worse than the two-layer G1-best tied
learned-ID baseline, the fixed concatenation, or the learned global gate.

Completed parameter-matched RQ2 DenseNet and RQ4 extra-ID artifacts remain
preserved in raw audit storage. They are excluded from the active protocol,
budgets, selection, promotion, and reader/compact/tuning tables; they are never
deleted.

## 4. Predicting future items during training

### Common setup

**Main dataset: Yambda-500M.** The control is standard next-liked-item
training. Construct all broader positives strictly after each prefix and within
the training interval. Keep final-seven-day evaluation unchanged and mask all
valid positives for a query from its negatives.
Extend `SequenceTargets`/`TimeWindowTargets` for target construction and expose
each rule through a `TimeWindowGenerationExperiment` configuration. Every
native-500M arm uses the two-layer form of G1's selected
aggregate: transfer its ten non-depth members, including SwiGLU width 192, but
keep model/item width 64, two attention heads, and two transformer layers. Do
not add scaling-only depth, width, or per-layer-embedding increases. Retain
G1's fixed-width μP class with base/delta widths 16/32, its tied item table,
two query heads and one KV head, and the exact selected aggregate dropout,
position, time, negative-sampling, norm, CLS, GQA, and BOS semantics.
Historical native-50M SwiGLU-171 artifacts remain unchanged.

Batch size is fixed at 512 and is not tuned. Tune the relevant non-batch
parameter: fix embedding LR and the 15-epoch deep-only one-cycle-cosine horizon
to G1's selected aggregate values, and tune only deep learning rate. Start with
half, equal to, and twice G1's deep LR; add two points farther left or right only
when an edge wins. Fix every RQ3 recommender arm to one selected period rather
than retuning capacity. Reuse only the reviewed relative dispersions from the
one-time unchanged-control native-500M calibration, scaling G4's own
native-500M baseline values; never reuse the calibration's absolute scores or
absolute bands. G4 launches no repeat batch.

### RQ1 — Does a 24-hour future window help?

**Understanding.** The target treats any positive engagement in the next day as
acceptable instead of privileging the immediately next event.

**Implementation.** For every causal prefix, sample one eligible liked-event
positive per epoch from its next 24 hours. This is PinnerFormer-style dense
all-action supervision: prefix coverage is dense, while one future positive is
sampled for each prefix rather than scoring every future positive in one loss
row. Fall back to the next liked item when a prefix has no 24-hour candidate,
so the prefix/user distribution, number of prefix-positive pairs, and optimizer
budget match the next-item control. Mask every valid 24-hour positive item from
that query's negatives and report aggregate metrics, recall by temporal
distance, eligibility, and fallback rates. Reference: [PinnerFormer](https://arxiv.org/abs/2205.04507).

**Acceptance.** Compare the 24-hour objective with next-item training. Report
results by target distance and user activity. It most probably should be better since it better alines with the evaluation.

### RQ2 — Does a next-event-count window help?

**Understanding.** A next-10-events target removes user activity rate from the
window definition.

**Implementation.** Use positives from the next 10 eligible events while
keeping prefix sampling and total positive-pair budget matched. The source asks
for 10; smaller `k` values are optional diagnostics, not required treatments.
Compare with next-item and 24-hour objectives under the same evaluator.

**Acceptance.** Compare next-10-events with both next-item and 24-hour training.
Report results by target distance and user activity. It should not be much worse then the baseline

### RQ3 — Can behavior-similar periods define better positives?

**Understanding.** This proposes learning which later hours/days represent the
same user state, then using only their items as positives. It is conditional
future supervision, not simply a longer window.

**Implementation.** First define period boundaries and training-only similarity
labels. The offline selector may inspect candidate-period liked events while
constructing training targets, but neither it nor the recommender may inspect
the final seven evaluation days. Try the following selectors:

1. deterministic matching to the same UTC hour and day-of-week in later weeks,
   because Yambda has no user-timezone field;
2. cosine similarity between content centroids of the prefix period and each
   candidate future period;
3. weighted-Jaccard similarity between their item, artist, or album frequency
   vectors;
4. a binary period-pair classifier whose inputs combine the same liked-event
   content/frequency similarities as the deterministic selectors with time
   gap, continuous circular hour-of-week similarity, prefix/candidate hour/day
   features, period activity, and past
   user-activity counters. Its independent label is whether weighted-Jaccard
   similarity between the user's listened-artist frequency vectors in the two
   periods strictly exceeds the selector-training partition's fixed
   nearest-rank 80th-percentile threshold (`label = similarity > threshold`);
5. for the best learned score, hard top-k period selection versus sampling
   positives proportionally to the predicted similarity.

Use one-hour, six-hour, and calendar-day periods as the bounded choices. For
hourly periods search 3/7-day lookaheads; for daily periods search 14/28-day
lookaheads. Give the time, content, frequency, and learned
families equal random-search budgets over their predeclared conditional spaces.
Tune on the chronological selector-validation partition and evaluate the gate
once on an untouched selector-test partition; neither may use the final seven
evaluation days.

Evaluate every selector on one common universe: every later liked-event
occurrence in the next 28 training-only days for each causal prefix. Define an
event's independent relevance from listened-artist similarity between the
prefix's trailing 24 hours and only the strictly post-prefix part of the UTC
day containing that event. Structural
period choices change scores, not candidates, labels, queries, or NDCG
denominators. The time family excludes a three-day lookahead because matching
UTC day-of-week inside it is degenerate.

Every width-`w` past-period summary is exactly `(prefix-w, prefix]` and every
UTC-aligned candidate period starts strictly after the causal prefix. For
downstream target construction,
generate learned-selector scores with five user-id hash folds: train on four
folds and predict the fifth, so no classifier scores a user or pair it trained
on. Recommender histories and targets remain likes-only; listens are consumed
only inside the training interval to define the independent selector label.

Freeze every selector before recommender training and compare it with matched
next-item and fixed-window controls using the same prefix-positive budget. Fall
back to the next liked item when no selected period is available, preserving
the control prefix/user distribution, and report the fallback rate. Stop before
recommender training if the classifier does not beat the deterministic
selectors on untouched-test period-pair ranking or if materializing its
positives is outside the approved runtime budget.

For the selector gate, compare the learned winner with the validation winner
across every deterministic pipeline on the common event universe. Evaluate both
once on selector test and resolve their paired query difference by user-cluster
bootstrap before calling the learned selector better.

**Acceptance.** The learned selector must beat the best deterministic selector,
and its downstream model must beat next-item and the best fixed-window control.
Report selector quality and materialization cost. It must be not worse then baseline. Most probably better.

## 5. Likes and listens with action tokens

### Common setup

**Main dataset: Yambda-500M.** Preserve chronological likes and listens.
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

**Acceptance.** Compare each cross-action history direction with its target-only
control. Report likes and listens separately; averaging may not hide negative
transfer. Also include some analysis: popularity biased attention map and some dipper analysis too. To check your results you also need the following comparison: likes + listens history model with history length limited by likes (with all listens in between) must be better then likes only model with the same history length in likes (but without listens). Or at least not worse. But most probably better.

### RQ2 — Does jointly predicting likes and listens help?

**Understanding.** This changes target supervision while holding joint history
fixed.

**Implementation.** Compare two independently trained single-target models with
one shared encoder and two task heads. Tune the loss weight, compare shared and
task-specific output projections under matched budgets, and report both target
deltas against their corresponding single-task control.

**Acceptance.** The joint model must improve at least one action without
regressing the other. Compare it with both single-task controls.

### RQ3 — Are explicit `<action>` and `<item>` tokens needed?

**Understanding.** Determine whether the model benefits from an explicit typed
grammar or whether action information can be attached directly to an item
event.

**Implementation.** Compare two-token `<action> <item>` serialization with a
single item event carrying an action embedding and, if desired, action-specific
item tokens. Match visible event history and transformer capacity. Report raw
token length and latency in addition to per-action quality.

**Acceptance.** Compare explicit tokens with compact event encoding. Prefer the
compact form when quality is non-inferior and sequence length or latency is
better.

### RQ4 — Does request-conditioned interleaving help?

**Understanding.** Train one model on interleaved task blocks. `<want like>`
asks for the next future item with `is_like = 1`. `<want listen>` uses an
explicit attributed-listen target: the earliest future raw listen or like event.
Its target may therefore be listen-only or liked, because every like is treated
as a listen for this request. Ordinary
chronological items remain in the same causal stream; after such an item the
model outputs its like/listen probabilities. A stream can therefore look like
`<want like>, item_i, <want listen>, item_j, item_k,
[p_like, p_listen], <want like>, item_l`. At serving, appending `<want like>`
requests a like recommendation.

**Implementation.** For every eligible chronological prefix derive three
blocks: `<want like>, next_liked_item`, `<want listen>, next_listened_item`, and
`immediate_next_item, [like/listen output]`. `next_listened_item` is the first
future event in the union of raw listen and like events; record which event type
supplied it. This attribution is used by every `<want listen>` control and
treatment. A requested target may skip intervening events, whereas the ordinary
block never does. Sample and pack these blocks
into the interleaved causal training stream with segment-aware attention: every
block sees its own chronological prefix and its request, but cannot attend to
targets from adjacent blocks. A requested block applies item loss at the
request position. An ordinary block applies chronological next-item loss before
the item, then a two-logit action loss after the item using that item's hidden
state; the probability pair is an output, not ground-truth input. Loss masks
must also guarantee that an item prediction cannot attend to its action target.
Tune the block mixture while keeping the number of supervised items fixed.
Compare the all-ordinary stream, interleaving without action loss, and the full
interleaved stream. At evaluation append each request separately; for
`<want listen>`, also report recall split by listen-only versus liked targets.
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

**Acceptance.** Compare each requested output with its single-task control; one
must improve and neither may regress. Use the same controls for the offline
model and label it non-production.

### RQ4.1 — How should action and item tokens be aggregated?

**Understanding.** Rich events should not necessarily double sequence length.

**Implementation.** Compare the expanded serialization with one event vector
from `DenseNet(concat(action_embedding, item_embedding))`. Keep the same event
history and request/target grammar. BPE is not assumed to solve this comparison;
it can be studied later as a separate serializer.

**Acceptance.** Prefer aggregation when both action metrics are non-inferior and
sequence length, memory, or latency improves; otherwise select by Recall@100.

### RQ4.2 — Which auxiliary losses are needed?

**Understanding.** Separate the value of predicting the next item from the
value of predicting interaction type.

**Implementation.** With representation fixed, compare next-item loss only,
and next-item plus interaction-type loss. Tune the joint weight and report both
item retrieval and action classification. Do not add an interaction-only arm:
without an item objective it cannot answer the recommendation question.

**Acceptance.** Compare with no auxiliary loss. Keep the auxiliary loss only if
item retrieval improves, or stays non-inferior while action prediction improves.

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

**Acceptance.** Compare per-target with query-only request placement. Per-target
placement must improve one action without regressing the other; otherwise
prefer the cheaper query-only form.

### RQ5 — Which model is best for likes prediction?

**Understanding.** Choose the best way to predict the next liked item when both
likes and listens are available in history. This closes the separate G5
comparisons into one like-focused decision.

**Implementation.** Compare on identical like-evaluation prefixes and candidate
sets:

1. likes-only history with a next-like objective;
2. chronological likes+listens history with the same next-like objective;
3. the selected joint like/listen model from RQ2;
4. the selected request-conditioned model from RQ4 evaluated with
   `<want like>`; and
5. the offline enriched-event model from RQ4.

For the first two arms, expose the same last `L` liked items; arm 2 additionally
includes every listen between those likes. Match capacity and tuning budgets.
Reuse earlier evidence only when its evaluation prefixes, history construction,
and candidates are identical; otherwise run a matched bridge.

**Acceptance.** Select by like Recall@100; ties go to lower serving cost. The
likes+listens next-like model is expected to beat likes-only. A null or
regression requires the same popularity-attention and mechanism analysis as
RQ1.

## 6. RQ-KMeans semantic IDs in history

### Common setup

**Main dataset: Yambda-500M.** Fit RQ-KMeans only on content vectors for the
train-mapped catalog. Version normalization, codebooks, assignments,
centroids, level vocabulary, and collision map. The downstream task remains
item retrieval. Report item metrics plus ICR, p95 load at every level,
intra-code similarity, and collision distribution.
Search only three or four levels and one shared codebook size from 512, 2048,
or 8192 at every residual level. G6 never assigns independently tunable sizes
to different levels. Use the fixed 26-epoch horizon and width 128 throughout.
Create a `KMeansIdStage` before the downstream experiment, represent codes with
`SemanticHistoryTokenizer`/`dcn/nn/semantic_embedding.py`, and store all fitted
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
Tune the historically strongest item-ID plus frozen-SID-event family first,
then use its selected SID and learning-rate setup as the starting anchor for
smaller independent searches of the other six families. RQ2/RQ3 later reopen
the collision policy.
Use the final best G1 combination as the primary backbone for all seven
representations. Also reconstruct the original G1 item-ID baseline at 500M, and
bridge only the winning representation to that original backbone rather than
running the full seven-by-two crossing.

**Acceptance.** Compare every tuned SID representation with learned item-ID
history. The final model must not be much worse then sasrec. And ideally it should be better (I think it can be).

### RQ1 — How should SID embeddings be initialized?

**Understanding.** Compare pure task learning with content-informed
initialization.

**Implementation.** Apply this only to RQ0 representations that contain a
trainable SID lookup. Initialize that lookup randomly or from the corresponding
codebook/centroid vectors after an explicit projection, then train end to end
under equal budgets and report convergence plus final quality. If the RQ0
winner has no trainable SID lookup, use the strongest applicable representation
for this initialization RQ without replacing the RQ0 winner.

**Acceptance.** Compare content/codebook initialization with random
initialization. Faster convergence counts only when final recall is
non-inferior. Codebook initialization should converge faster.

### RQ2 — What is the best collision-resolution-token setup?

**Understanding.** Find the best RQ-KMeans tokenizer hyperparameters when a
collision-resolution token is included. This is primarily a search over
codebook size, number of levels, and related tokenizer parameters—not over
different collision-token designs.

**Implementation.** Use one fixed collision-resolution rule and search the
restricted three/four-level by 512/2048/8192 shared-codebook surface from the
selected RQ0/RQ1 setup, with local learning-rate refinement. Select on
downstream item recall, with ICR, p95 load, intra-code similarity, collisions,
memory, and latency as diagnostics. Apply the `2^13` maximum to every level,
including the collision-resolution level/symbols, for an approximately
`2^20`-item catalog.

**Acceptance.** Compare every tuned collision-token configuration with the RQ0
setting. The selected collision-token configuration must not worsen downstream
Recall@100 versus RQ0.

### RQ3 — What happens without collision resolution?

**Understanding.** Repeat RQ2's tokenizer-hyperparameter search without adding
a collision-resolution token. This is not a single ablation that removes the
token from RQ2's winner; the optimum number of levels and shared codebook size may
change when collisions remain unresolved.

**Implementation.** Run the same restricted paired axes and budget as RQ2,
starting from the same selected RQ0/RQ1 setup, but omit collision resolution.
Define an ambiguous tuple as the shared history representation of every item in
its bucket. Select on downstream item recall and report the same intrinsic,
collision-bucket, memory, and latency diagnostics so the independently selected
with/without-collision setups can be compared.

**Acceptance.** Compare independently tuned systems with and without collision
resolution.

## 7. Semantic-ID generation

### Common setup

**Main dataset: Yambda-500M.** By default, fit and freeze one approved
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

**Acceptance.** Compare encoder-decoder with matched decoder-only generation.
An efficiency win requires non-inferior recall.

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

**Acceptance.** Compare all three history representations. Independently tuned
systems decide; the shared-tokenizer comparison only explains the result.

### RQ3 — Causal or bidirectional encoder?

**Understanding.** At serving time the entire past history is available, so a
bidirectional history encoder may use context that a causal pretraining setup
forbids.

**Implementation.** Change only the encoder attention mask in the selected
architecture. Keep query extraction, positions, decoder, and history identical.
Verify that neither mask can see target SID tokens.

**Acceptance.** Compare bidirectional with causal history attention under the
same encoder-decoder setup.

### RQ4 — Does next-item encoder pretraining help?

**Understanding.** Test whether a causal SASRec representation is a better
initialization for SID generation.

**Implementation.** Pretrain the selected causal encoder on item-ID next-item
prediction using training data only, then fine-tune the whole generator. Compare
with scratch under matched final-stage budgets. A frozen-encoder arm is a
proposed diagnostic, not a source requirement.

**Acceptance.** Compare pretrained and scratch generators. Any efficiency claim
must include pretraining plus fine-tuning cost. Model with pretraining will be most probably better. You should consider other results unexpected.

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

**Acceptance.** Compare every adjustment with `beta = 0`. Select by item
Recall@100, not SID-token accuracy. In my previous tests sids logq has improved the metrics quite a bit.

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

**Acceptance.** Separately compare weighted vs unweighted loss, corrected vs
uncorrected sampled softmax, and reranked vs raw beams. Do not compare across
objective families.

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

**Acceptance.** Compare independently tuned forward and reverse systems. Use
the shared-tokenizer comparison only to isolate generation order.

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

**Acceptance.** Compare all three decoder-only serializations. Independently
tuned systems decide; the shared-tokenizer comparison only explains the result.

### Decoder-only RQ4 — Does next-item pretraining help?

**Understanding.** Test whether item-ID next-item pretraining initializes the
decoder-only history model better than training SID generation from scratch.

**Implementation.** Use the selected decoder-only history serialization and
pretrain the same causal backbone on item-ID next-item prediction using only
training prefixes. Initialize the SID generator from that backbone and
fine-tune every parameter. Compare with a scratch model under the same final
training and tuning budget; retain a frozen-backbone arm only as a diagnostic.

**Acceptance.** Compare pretrained and scratch decoder-only generators. Any
efficiency claim must include pretraining plus fine-tuning cost. Pretraining is
expected to improve metrics; a non-improvement is unexpected.

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

**Acceptance.** Compare every decoder-only adjustment with `beta = 0`. Select by
item Recall@100, not SID-token accuracy. Previous SID-logQ experiments improved
metrics substantially, so a non-improvement is unexpected.

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

**Acceptance.** Separately compare weighted vs unweighted loss, corrected vs
uncorrected sampled softmax, and reranked vs raw beams. Do not compare across
objective families.

### Decoder-only RQ7 — Does reverse SID order help?

**Understanding.** Reversing coarse-to-fine generation may change the
decoder-only beam in the same way as for encoder-decoder generation, while the
causal history prefix stays unchanged.

**Implementation.** Compare forward and reversed target SIDs after the same
history boundary. Reverse level vocabulary, target positions, trie traversal,
and prefix constraints together. First isolate order with one shared tokenizer;
then independently tune tokenizer and decoder settings for each direction under
equal budgets. Match final beam and candidate budgets.

**Acceptance.** Compare independently tuned forward and reverse decoder-only
systems. Use the shared-tokenizer comparison only to isolate generation order.

## 8. Item-ID and SID outputs in an encoder-decoder model

### Common setup

**Main dataset: Yambda-500M.** Reuse the selected G7-style encoder and SID
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
   full-catalog item-ID decoders. Main head - sid decoder;
3. one sequential decoder whose target is
   `[sid_1, ..., sid_L, item_id]`, so the item-ID loss is applied after the SID;
4. the same sequential decoder with training-time corruption of teacher-forced
   SID inputs before `item_id`: replace selected SID tokens with a level-specific
   mask token, or replace a suffix with a different valid catalog continuation;
   and
5. a cascaded variant where a one-token item decoder cross-attends to the
   completed SID-decoder states instead of placing item ID in the same token
   stream.

Arm 4 uses two passes over the same encoded history. The clean pass computes all
SID losses. The corrupted pass recomputes only the final item-ID prediction and
item loss; it contributes no SID-token loss. Tune corruption probability
including zero and the clean/corrupted item-loss mixture. Exclude the true SID
continuation from random replacement. Report controlled corruption curves by
probability and corrupted level. Tune SID/item loss weights for joint models.
During training, report sequential item accuracy both with teacher-forced SID
prefixes and with generated SID prefixes; at inference only the generated path
counts. Report each output's recall, their oracle union, fused ranking, latency,
and error recovery when an earlier SID level is wrong.

**Acceptance.** Variant 2 should beat the SID-only control in variant 1 by item
Recall@100. Variant 3 probably will not work well. Variants 4 and 5 should
probably beat variant 3 under generated-SID inference.

### RQ2 — Which logQ definition is correct for the joint model?

**Understanding.** Item-head and SID corrections are separate factors.

**Implementation.** With architecture fixed, test:

1. neither correction;
2. concrete-item logQ on the item head only;
3. the selected G7 SID correction only, if it passed the prerequisite;
4. both corrections, if arm 3 is eligible.

If G7 selects sequence-level item-popularity weighting rather than SID-level
adjustment, use that exact method on the SID loss; do not rename it SID logQ.
Tune correction strength for every enabled head and the joint loss weight. Do
not spend runs on marginal/prefix SID variants that G7 already rejected.

**Acceptance.** Compare item-only correction with no correction. If SID
correction is eligible, compare SID-only with no correction and compare both
corrections with no correction and with each single correction.

### RQ3 — Which head should produce the final generation?

**Understanding.** In the model with shared history between sid generator and item id generator which head is better?

**Implementation.** Reuse results from rq1.

**Acceptance.** Compare SID-head and item-head Recall@100 from the same joint
model. Also report fusion and use it only if it beats both heads.

### RQ4 — Do parallel forward/reverse SID branches help?

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

**Acceptance.** Compare parallel branches with the stronger single branch under
one final candidate budget. Select by realized fusion; oracle union is
diagnostic only.

## 9. Improving semantic IDs

### Common setup

**Main dataset: Yambda-500M.** Use the same input content vectors and
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

**Acceptance.** Select the tuned RQ-KMeans configuration by downstream item
recall. Report tokenizer and decoding cost for non-inferior alternatives.

### RQ2 — Does RQ-VAE improve over RQ-KMeans?

**Understanding.** Learn the reconstruction space and residual codebooks jointly
instead of clustering a fixed representation.

**Implementation.** Under an equal search budget, tune levels, per-level
codebook sizes, latent width, reconstruction weight, and commitment settings.
Train/evaluate the same downstream generator for selection, freeze the winning
tokenizer, and compare both intrinsic SID metrics and downstream item recall
with RQ-KMeans.

Do not forget to inialize rqvae with rqkmeans!

**Acceptance.** Downstream Recall@100 must not be worse than RQ-KMeans and will
probably be slightly better.

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

**Acceptance.** Full PLUM should have downstream Recall@100 at least not much
worse than matched RQ-VAE. Separately compare continued pretraining with direct
fine-tuning and compare all three codebook schedules inside the fixed PLUM
system.

### RQ4 — Does R3-VAE help?

**Understanding.** Test reference-vector guidance, rating-based stabilization,
and SID-quality regularization as an RQ-VAE family.

**Implementation.** Reproduce the full R3-VAE tokenizer, then ablate its three
added mechanisms against matched RQ-VAE. Freeze each selected tokenizer before
the common generator and report stability/collapse as well as final metrics.
Reference: [R3-VAE](https://arxiv.org/abs/2604.11440).

**Acceptance.** R3-VAE should most probably improve downstream Recall@100 over
matched RQ-VAE.

### RQ5 — Do variable-length BPE SIDs help?

**Understanding.** The source proposes BPE as the mechanism for variable-length
identifiers; a separate non-BPE controller is only a proposed addition.

**Implementation.** Start from fixed base SID tuples, learn BPE merges on the
training catalog only, and map every item to its merged token sequence. Compare
under fixed decoding-latency and candidate budgets, reporting vocabulary size,
length distribution, collisions, and quality.

**Acceptance.** BPE most probably should have non-inferior recall and shorter or faster decoding.

### RQ6 — Does DIGER's differentiable tokenizer help?

**Understanding.** Let recommendation gradients update SID assignments while
preventing early codebook collapse.

**Implementation.** Jointly train tokenizer and generator with Gumbel
exploration and a declared uncertainty-decay schedule. Compare with the same
architecture using the selected frozen RQ-VAE tokenizer and ablate
exploration/decay. Report code utilization throughout training. Reference:
[DIGER](https://arxiv.org/abs/2601.19711).

**Acceptance.** Compare DIGER with the same generator using the frozen RQ-VAE
tokenizer. DIGER should be better than this matched RQ-VAE control.

### RQ7 — Does a collision token help?

**Understanding.** Append exact disambiguation only to semantic tuples shared by
multiple items.

**Implementation.** Hold the base tokenizer fixed and compare unresolved tuples
with collision suffixes. Report exact item resolution, suffix vocabulary/load,
dynamic-catalog implications, decoding cost, and item metrics.

**Acceptance.** Compare collision suffixes with unresolved tuples. Suffixes may
win with non-inferior recall if they remove ambiguity without worsening memory
or decoding latency beyond its band.

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

**Acceptance.** Compare ECM and RRS with collision suffixes. Both must produce
unique assignments; RRS may win on assignment cost with non-inferior recall.

## 10. Semantic IDs with collaborative information

### Common setup

**Main dataset: Yambda-500M.** The control is the selected content-only
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

**Acceptance.** Aligned content+collaborative input must beat both content-only
and collaborative-only tokenizers.

### RQ2 — Do interaction pairs improve RQ-VAE?

**Understanding.** Add pairwise behavioral proximity to the reconstruction
objective, but `pairs` is underspecified in the source.

**Implementation.** Obtain approval for one pair definition. A proposed first
version uses next-item transitions and co-listens within a fixed training-only
window with popularity-matched negatives. Add a contrastive pair term to
RQ-VAE, tune its weight, and compare with matched RQ-VAE.

**Acceptance.** Compare every pair definition with matched RQ-VAE without the
pair term. Select by item recall, not pair separation or intrinsic SID metrics. It must not be worse then the original rqvae. And most probably it should be better.

## 10A. User-aware semantic-ID tokenization

### Common setup

**Main dataset: Yambda-500M.** This is a separate experiment because a
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

**Acceptance.** Compare every user-aware representation with global content
codes on identical candidate lists and select by reranked Recall@100.
Query-specific SID recall should beat global-code SID recall on those lists.
Like AUC and other SID metrics are diagnostics. Report materialization cost and
mark infeasible variants as blocked.

## 11. Gryphon

### Common setup

**Main dataset: Yambda-500M.** Fit a fresh residual-K-Means tokenizer on
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

**Acceptance.** Gryphon should beat the stronger of matched vanilla GR and
collision-resolved GR by Recall@100.

## 12. Diffusion over item content embeddings

### Common setup

**Main dataset: Yambda-500M.** Freeze normalized content/audio targets for
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

**Acceptance.** Compare diffusion with matched one-shot regression. Select by
item recall; coverage, duplicates, target distance, and latency describe the
trade-off. Mark over-budget settings infeasible.

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

**Acceptance.** Compare every residual family with the stronger of one-shot
regression and diffusion. Select by item recall; partial-step error and cost are
diagnostics and trade-off metrics.

## 13. Reinforcement learning for multi-step ranking

### Common setup

**Main dataset: Yambda-500M.** Data scale is secondary to evaluation
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

**Acceptance.** Block until logged propensities or a validated simulator exist.
Then DR value must beat both supervised controls beyond its uncertainty band,
IPS must agree in direction, and one-step quality must not regress. Label
simulator evidence as simulator-only.

## 14. Thinking over history

### Common setup

**Main dataset: Yambda-500M.** Keep the final item objective fixed and
require equal-parameter and equal-latency controls so extra compute is not
misnamed reasoning. Keep retrieval and full-catalog evaluation unchanged and
expose every intermediate trace for diagnostics.

### RQ1 — Does thinking based on history help?

**Understanding.** Treat “thinking” as observable intermediate context before
the final prediction. Test action events between like targets and an explicit
SID draft-revision trace.

**Implementation.** Compare:

1. **Listens between likes.** Train for next-like prediction, but retain every
   chronological listen between the visible liked items as intermediate context.
   Match the number of visible likes to a likes-only control; this is also a
   bridge to G5 RQ5.
2. **SID revision.** Train a revision decoder on
   `[history, corrupted_draft, SID_END] -> clean_sid`. Create the draft by
   masking SID tokens or replacing a suffix with a different valid SID
   continuation; artificial draft values receive no prediction loss. Tune the
   corruption probability including zero. At inference, first generate a draft
   with the matched one-pass generator, then pass that actual draft to the
   revision decoder and rank by the revision.
3. **Direction variants.** For SID revision, compare forward/forward,
   reverse/forward, and forward/reverse first/second attempts.

For SID corruption, exclude the true continuation from random replacement.
Compare every family with the unchanged one-pass baseline and its
equal-parameter and equal-latency controls. Report draft and revised SID/item
recall, correction rate after a wrong draft, and latency.

**Acceptance.** A thinking claim requires beating both equal-parameter and
equal-latency controls. SID revision must also improve revised item recall over
both its draft and the matched one-pass generator.

## Remaining ideas

### RQ1 — Do additional features improve SID construction?

**Understanding.** Determine which content, collaborative, temporal, or action
features improve both code quality and retrieval.

**Implementation.** On 50M, add one feature family at a time before
quantization, and give every feature family the same bounded downstream search
over levels, per-level codebook sizes, and method-specific parameters. Select
on item recall and report intrinsic SID metrics as diagnostics.

**Acceptance.** Compare each feature with the selected tokenizer without that
feature family. It is usable only if non-inferior and selected only for an
item-recall gain; SID metrics are diagnostic.

### RQ2 — Does Muon or another optimizer help?

**Understanding.** This is an optimizer RQ, not a semantic-ID method.

**Implementation.** After fixing a model/objective, tune each optimizer with an
equal budget and compare convergence, final quality, time, and memory on that
experiment's main size.

**Acceptance.** Compare with the current tuned optimizer. A challenger may win
with non-inferior recall and lower total time or memory, including tuning cost.

### RQ3 — Do auxiliary prediction tasks improve semantic-ID recommendation?

**Understanding.** Test whether predicting item properties or counters provides
useful supervision beyond next-item loss.

**Implementation.** Fix one SID tokenizer and downstream model. Compare
next-item loss alone with auxiliary heads for available categorical targets and
training-only item-counter bins, first separately and then in the best
combination. Fit every bin and counter from training data; dynamic counters must
be prefix-causal. Tune auxiliary loss weights and report auxiliary accuracy as
a diagnostic.

**Acceptance.** Compare every auxiliary task with next-item loss alone. Select
only by downstream item recall; auxiliary accuracy is diagnostic.

### RQ4 — Does DCNv2 with a DenseNet deep part help?

**Understanding.** Establish a non-sequential ranking/reranking control over a
fixed candidate set.

**Implementation.** Use DenseNet for the deep branch, keep explicit cross
features in DCNv2, and compare with a capacity-matched MLP ranker on 50M.

**Acceptance.** Compare DCNv2 with a capacity-matched MLP. Retain explicit
crosses only for better recall, or non-inferior recall with lower latency or
memory.

### RQ5 — Does adding a history transformer to DCNv2 help?

**Understanding.** Test whether sequential context adds information beyond
explicit crosses.

**Implementation.** Feed the selected history-transformer representation into
the fixed DCNv2 ranker and compare with DCNv2 alone under the same candidates.

**Acceptance.** Compare with DCNv2 alone. Keep the history transformer only if
Recall@100 improves; extra compute cannot be justified by non-inferiority.

### RQ6 — Does transformer pretraining improve ranking?

**Understanding.** Separate useful initialization from a permanently frozen
representation.

**Implementation.** Compare scratch, pretrained-and-fine-tuned, and
pretrained-frozen history encoders with the same downstream ranker on 50M.

**Acceptance.** Compare fine-tuned and frozen pretraining with scratch. Frozen
pretraining is an attribution arm; any efficiency claim includes pretraining
cost.

## Ambiguities that still block plans

PSI is now resolved as *Purely Semantic Indexing for LLM-based Generative
Recommendation and Retrieval*. Proposed completions remain explicitly marked
for the empty G5 RQ4.3 and continuous-residual G12 RQ. Before implementation or
training, each experiment still needs an approved plan with exact
configurations, run counts, and selection rules.
