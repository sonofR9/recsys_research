# Additional experiment ideas

These ideas fill gaps in [`ideas.md`](ideas.md); they are not approved run plans.
Diagnostics and the history-length repair are inputs to closing G1. The later
ideas assume G1 is complete and reuse its homework-compatible model and data
protocol. Main experiments default to native Yambda-50M and use resolution
bands measured there. G1 has already shown that training settings, evidence,
and bands never transfer across sizes.

## Recommended order

The ratings below are judgment calls, based on proximity to an observed G1
failure mode, ability to explain later experiments, implementation cost, and
strength of the cited prior evidence. They are not measured Yambda results.

| priority | idea | expected value | implementation risk | closest existing group |
| --- | --- | --- | --- | --- |
| 0 | diagnostic slices and simple baselines | very high | low | all groups |
| 1 | history length × recency-anchored positions | high | low | G1 RQ7/RQ8 |
| 2 | repeat/explore retrieval | high | medium | G1/G4 |
| 3 | multiple interest queries | high | medium | G1 |
| 4 | transition-graph distillation | high | medium | G1/G3 |
| 5 | false-negative-aware hard negatives | medium-high | medium | G1 RQ11 |
| 6 | short/long-timescale encoder | medium-high | medium | G1 RQ8/RQ9 |
| 7 | self-supervised sequence regularization | medium | medium | G1/G7 |
| 8 | retrieval geometry and calibration | medium | low | G1 RQ11 |
| 9 | HSTU backbone | potentially high | high | G1/G2 |

## 0. Diagnostic matrix before more architecture work

**Dataset: native Yambda-500M.** This diagnoses the established G1 control and
must use the same dataset regime and evaluation population as the evidence it
is intended to explain.

**Question.** Where does the selected G1 model fail, and are aggregate gains
coming from the same users and items?

Create fixed evaluation slices by user activity, history length, target item
frequency, repeat-versus-novel target, time since last event, and number of
relevant future items. Add non-neural popularity, recency, last-item
nearest-neighbor, and item-transition baselines. Sampling, if needed for a
diagnostic, must be by user ID.

This is not a new model. I rank it first because one inexpensive artifact can
decide whether to invest in repeat modeling, content/cold-start, multiple
interests, or transition structure. Keep aggregate recall/NDCG primary and
treat slices as mechanism evidence, not as independent opportunities to
cherry-pick wins.

## 1. History length × recency-anchored positions

**Dataset: native Yambda-500M.** This is an interaction inside the established
G1 RQ7/RQ8 surface; moving it to 50M would make it incomparable with the
position and history-length findings it repairs.

**Question.** Does longer usable history become non-degrading when learned
positions are anchored at the newest event rather than the oldest event?

**Hypothesis.** Extra past events should be useful or ignorable, but the current
forward table moves the newest event to a different learned index whenever the
history length changes. Reverse indexing keeps the newest event at index zero
and may therefore remove a position-distribution shift that masks the value of
long histories. The existing position experiment at length 128 does not test
this interaction. The existing length sweep also keeps attention window 50 in
a two-layer model, so caps above roughly 100 do not expose the cutoff query to
older events.

**Minimal comparison.** First repair the existing length-200 learning-rate
confound under learned forward positions; the length-512 arm already used 512
uniform catalog negatives per query despite gradient accumulation. Then compare
forward versus corrected reverse learned positions at lengths 128 and 512 with
full causal attention in all four cells, independently selecting the permitted
learning rates. Before training, report actual next-item training-window fill,
the number of query targets exposed to every learned position, and evaluation
metrics sliced by user-history length. Because the loader is packed, this is a
truncation and position-exposure question, not a padding-mask question. Treat a
decrease at longer length as a debugging signal: verify identical target users
and events, target count, negative pool, schedule completion, position-index
semantics, and optimization before accepting it as a model result.

The current reverse table is indexed from the end of each complete training
chunk. Under all-next-item training, position zero belongs to the chunk's final
token, which has no next-item loss, while the cutoff query uses position zero at
evaluation. This train/evaluation exposure mismatch must be measured; a
shift-invariant relative treatment such as RoPE or ALiBi is the fallback if the
existing reverse implementation does not test the intended mechanism.

This should remain a controlled G1 RQ7/RQ8 interaction, not become a new
top-level experiment. It is not approved for implementation or training yet.

## 2. Repeat/explore mixture

**Question.** Does explicitly separating repeat consumption from exploration
improve full-catalog retrieval?

**Hypothesis.** A single dot-product head compromises between retrieving items
already present in history and discovering unseen items. A gate can first
predict repeat versus explore, then mix a pointer-like score over history items
with the normal catalog score.

**Minimal comparison.** Selected G1 control; the same model with only the
repeat/explore gate; the gate plus separate repeat and explore heads. Tune the
gate-loss weight and mixture temperature, not a large architecture grid. Report
overall metrics, repeat-target recall, novel-target recall, gate calibration,
and the natural repeat rate. If likes are effectively unique, reject this idea
after the diagnostic rather than implementing it.

This is motivated by [RepeatNet](https://arxiv.org/abs/1812.02646), but the
treatment should preserve this repository's transformer and full-catalog
protocol.

## 3. Multiple interest queries

**Question.** Is the single CLS retrieval vector—the largest current G1 win—a
bottleneck for users with several active tastes?

**Hypothesis.** Several learned query tokens can attend to different parts of
the history. Scoring an item by a calibrated maximum or log-sum-exp across
queries should improve recall and coverage. Item encoding stays unchanged, but
catalog scoring, ANN queries, and candidate merging generally grow with the
number of user queries and must be included in the cost comparison.

**Minimal comparison.** One CLS query against 2, 4, and 8 queries; max versus
log-sum-exp aggregation; optional diversity regularization between query
vectors. Keep total transformer capacity matched where practical. Report
quality, coverage, query utilization/collapse, latency, and gains by user
history diversity. Select the number of queries jointly with aggregation using
a random search if more than one parameter moves.

This is the retrieval-stage idea behind
[MIND](https://arxiv.org/abs/1904.08030) and
[ComiRec](https://arxiv.org/abs/2005.09347), applied directly to the already
successful CLS-query design.

## 4. Global transition-graph distillation

**Question.** Does teaching item embeddings global item-to-item transitions add
information that a finite per-user history transformer misses?

**Hypothesis.** A transition teacher captures reliable catalog-wide local
structure, especially for sparse histories, while SASRec captures personalized
context. An auxiliary distillation loss can combine them without a graph model
at serving time.

**Minimal comparison.** Build a directed adjacent-item (`k=1`) transition-count
matrix from training data only. For each item with outgoing transitions, form a
sparse teacher distribution as
`p(j|i) ∝ count(i,j)^(1 / temperature)` over observed outgoing edges; assign
zero target probability to non-edges and skip zero-outdegree source items. Train
the full-catalog item-embedding softmax to match that distribution with
cross-entropy. Compare the G1 control, joint
recommendation-plus-distillation training, and initialization from a separately
distilled item table without continued distillation. Tune the distillation
weight and teacher temperature. Report overall and activity/frequency slices,
embedding uniformity, and item-to-item transition recall. Wider transition
windows are a later one-parameter axis, not part of this first budget.

The closest reference is
[MQSA-TED](https://arxiv.org/abs/2311.01056). This differs from G10 because the
output remains ordinary item-ID retrieval; semantic IDs are not required.

## 5. False-negative-aware hard negatives

**Dataset: native Yambda-500M.** This extends the established G1 RQ11 objective,
so its control, negative distribution, and empirical bands remain native-500M.

**Question.** Can informative hard negatives beat uniform random without
mistaking plausible future positives for negatives?

**Hypothesis.** RQ11 shows that diversity and effective pool size matter, but it
does not test model-mined hard negatives. Pure top-score mining will over-sample
false negatives; filtering items that occur later within the same user's
training partition, then down-weighting suspiciously hard samples, may retain
useful gradients. Validation and test events must never affect training masks.

**Minimal comparison.** Use the final selected G1 negative objective as control,
with uniform random retained as a secondary mining baseline if another family
wins RQ11. At matched negative count and schedule, compare the control, mixed
control plus mined, and mixed plus false-negative masking/weighting. Mine from a
lagged checkpoint or momentum item index so the proposal is stable. Tune hard
fraction and one weight/temperature with random search. Report training-partition
collision rate, gradient concentration, popularity distribution, and ranking
metrics. Collision with held-out future positives is an after-training
diagnostic only.

Relevant starting points are
[HDCCF](https://arxiv.org/abs/2204.11752) and
[UFNRec](https://arxiv.org/abs/2208.04116). This should extend RQ11 rather than
become an unrelated top-level group.

## 6. Explicit short- and long-timescale encoder

**Question.** Does separating the current session from long-term taste improve
over generic multiple-query pooling?

**Hypothesis.** G1 does not establish a benefit from extending history beyond
100 events and finds that time features help, suggesting that relevance may be
structured by recency rather than raw length. One query can attend inside the
current session while another summarizes older history; a learned gate can
combine them.

**Minimal comparison.** Define session boundaries using a training-only gap
threshold selected from the timestamp distribution. Run this only after idea 3
establishes a two-query control. At equal query count and aggregation, compare
two generic queries, two queries restricted to short/long masks, and masked
queries plus a learned gate; test session-boundary embeddings as a separate
one-parameter ablation. Hold total history and model dimension fixed. Report
metrics by session age and history length, gate calibration, and latency. This
isolates temporal structure from the generic multi-query gain and is distinct
from RQ9, which changes time representation rather than pooling.

## 7. Self-supervised sequence regularization

**Question.** Can the same histories provide useful supervision beyond
next-item loss, particularly for sparse users and tail items?

**Hypothesis.** Dropout-consistent views or semantically valid positive pairs
can reduce representation degeneration without the label corruption caused by
aggressive crop/reorder augmentation.

**Minimal comparison.** Start with a joint next-item plus contrastive loss,
using two dropout views of the same prefix. Then test same-target prefixes as
positives. Only if both are stable, compare masking/cropping. Tune the auxiliary
loss weight and temperature; report item/query embedding isotropy, slice
metrics, and training cost. Compare joint training with pretrain-then-finetune
so an apparent gain is not just extra optimizer steps.

[DuoRec](https://arxiv.org/abs/2110.05730) motivates the conservative dropout
and same-target variants; [CL4SRec](https://arxiv.org/abs/2010.14395) is the
augmentation baseline.

## 8. Retrieval geometry and score calibration

**Question.** Are uncontrolled embedding norms or a fixed softmax temperature
limiting sampled-softmax training and full-catalog ranking?

**Hypothesis.** L2-normalized item/query embeddings with a learned or tuned
temperature may improve optimization, reduce popularity encoded only through
norm, and make negative-family comparisons more stable. It may also hurt if
norm carries useful confidence, so this is a cheap falsifiable axis.

**Minimal comparison.** Dot product control; cosine with fixed temperature;
cosine with learned global temperature; optionally separate query and item norm
penalties. Because scoring geometry changes the objective seen by both towers,
independently tune embedding and deep learning rates for each family but keep
negative sampling fixed. Report recall/NDCG/coverage, norm distributions by
popularity, temperature, score calibration, latency, and convergence. Do not
combine this with a new sampler until its own effect is resolved.

## 9. HSTU as a backbone family

**Dataset: native Yambda-50M.** This architecture-family comparison does not
need a separate 500M study.

**Question.** Does a recommendation-specific sequential transducer outperform
the optimized SASRec transformer at matched quality/cost budgets?

**Hypothesis.** HSTU's pointwise aggregated attention and recommendation-specific
block may model heterogeneous, non-stationary event streams more efficiently
than a language-style transformer. The advantage may disappear at Yambda's
short sequence lengths, which makes a controlled reproduction valuable.

**Minimal comparison.** First reproduce a public/reference HSTU unit and verify
causal masking and packed sequences. Calibrate the unchanged SASRec control's
global batch and negative count on native 50M, reuse those values for HSTU,
then tune embedding and deep learning rates plus family-specific width/depth
with equal budgets. Report two distinct views: an equal-parameter comparison
and a quality-versus-measured-latency frontier. If that 50M global batch is
infeasible for HSTU, document why and obtain approval for recalibration rather
than tuning a family-specific batch silently.

Reference: [Actions Speak Louder than Words / HSTU](https://arxiv.org/abs/2402.17152).
This is a better-defined external architecture comparison than adding more
isolated transformer knobs to G1.

## Extensions to existing groups

These should be folded into existing groups rather than opened as independent
programs:

- **G3 — frequency-adaptive ID/content gate.** Learn how much to trust item ID
  versus pretrained content from item frequency or embedding uncertainty.
  Compare against fixed replace/concatenate variants and report cold/tail/head
  slices.
- **G4 — horizon-conditioned future-set loss.** Supply a horizon token (next
  event, next hour, next day) and train one model to retrieve the corresponding
  future set. Compare with separate models to test whether horizons share useful
  structure.
- **G5 — missing-action denoising.** Randomly hide action type or item within a
  multi-behavior event and reconstruct it as an auxiliary loss. This directly
  tests whether action/item fusion learns cross-behavior structure.
- **G7/G8/G11 — candidate recall decomposition.** Always separate SID beam
  recall, collision-resolution recall, item-head recall, union recall, and final
  reranking recall. Without this decomposition a hybrid generator can improve
  for the wrong claimed reason.
- **G9/G10 — recommendation-aligned SID evaluation.** In addition to ICR, load
  balance, and intra-code similarity, measure prefix transition entropy,
  frequency-conditioned collision rate, cold/tail item recall, and how often
  content neighbors are behavioral substitutes. No single intrinsic SID metric
  should select a tokenizer without a downstream confirmation.

## Ideas to defer

Diffusion, RL, and explicit thinking are worth retaining, but they are lower
priority than the experiments above. Diffusion lacks a proven catalog-grounding
advantage over SIDs; offline RL lacks a trustworthy reward/evaluation design;
and thinking lacks a concrete supervision signal and equal-compute control.
They should advance only after the supervised item/SID generator, direct item
scorer, and diagnostic slices are complete.
