# Implementation details

## Shared setup

All RQs use the shared Yambda likes/final-seven-day
[protocol](../../generation_protocol.py#L7-L30) and the same baseline
[configuration factory](../configs/variant.py#L28-L140). Corrected follow-ups
compose one selected treatment with the shared batch-512 base
([composition](../configs/rq_tuning_variant.py#L47-L144)).

## RQ1 — μTransfer and dataset-size protocol

Current usable results and the model-width versus dataset-size conclusion are
in the [RQ1 transfer evidence](rq1_transfer.md).

- μP fixes the 64-dimensional item table, projects into and out of transformer
  width, registers base/delta shapes before initialization, uses width-aware
  attention scaling, and optimizes with MuAdam
  ([implementation](../../../dcn/config/generation.py#L545-L631)).
- Width and LR treatments use the revision-1 Cartesian μP dimension
  [manifest](../launchers/architecture/manifest.sh); the report aliases those
  exact artifacts into RQ1 without rerunning them.
- The dataset-size protocol trains 50M and 500M on their native cohorts,
  evaluates validation every epoch, applies early stopping, and restores the
  best validation checkpoint. The detailed comparison contract is in the
  [tuning protocol](../protocol/tuning.md).
- The collector marks fixed-terminal-epoch artifacts unusable. This includes
  the former target-count-matched 50M proxies and fixed-10-epoch 500M
  confirmations; none can answer an RQ
  ([collector](../analysis/collect.py)).

## RQ2 — Best transformer combination

- `selected_quality` defines sequence 128, linear decay, 16 additive time bins,
  uniform negatives, SwiGLU-192, GQA, and window 50
  ([configuration](../configs/variant.py#L440-L454)).
- `future_baseline` adds batch 512, input RMSNorm, and embedding/deep LRs
  0.032/0.012 ([final configuration](../configs/variant.py#L546-L561)).
- `rqfinal_normalization_input_rms` is dynamically composed
  ([composition](../configs/rq_tuning_variant.py#L94-L144),
  [500M invocation](../launchers/architecture/selected_500m.sh#L79-L90)).
- Architecture fields become concrete modules in the transformer
  [builder](../../../dcn/config/networks.py#L179-L242).

## RQ3 — Metrics/performance balance

- Batch and separate embedding/deep-LR candidates come from the selected model
  ([candidates](../configs/variant.py#L540-L608)).
- Embedding and deep parameters remain separate optimizer groups
  ([split](../../../dcn/config/experiment.py#L525-L548),
  [optimizer](../../../dcn/config/sequence.py#L362-L377)).
- Parameter counts and peak memory come from the resource
  [callback](../../../neuralrec/run/callbacks/resources.py#L14-L43); epoch
  costs are parsed by the [collector](../analysis/collect.py#L511-L574).

## RQ4 — SwiGLU

- GELU and SwiGLU variants are declared in the
  [variant grid](../configs/variant.py#L237-L245).
- SwiGLU is implemented in the FFN [layer](../../../dcn/nn/ffn.py#L9-L43)
  and selected by the transformer [factory](../../../dcn/config/networks.py#L179-L202).
- Families receive equal width/LR tuning
  ([proxy](../launchers/architecture/tuning_50m.sh#L52-L61),
  [500M GELU selection](../launchers/architecture/selected_500m.sh#L49-L49)); the
  SwiGLU control reuses `rqfinal_neg_random`
  ([500M invocation](../launchers/negatives/selected_500m.sh#L50-L50)).

## RQ5 — Learning-rate schedules

- The native-500M treatment surface, fixed embedding rate, scheduler scope,
  and adaptive horizon settings are declared by the dedicated
  [candidate builder](../analysis/rq5_scheduler_candidates.py) and
  [configuration](../configs/rq5_scheduler_variant.py).
- Candidate correction chains and strict artifact eligibility are resolved by
  the [correction planner](../analysis/rq5_scheduler_corrections.py). Validation
  selection, shared-central provenance, conditional probes, and the frozen
  candidate digest are enforced by the
  [selector](../analysis/rq5_scheduler_selection.py).
- The [report pipeline](../analysis/rq5_scheduler_report.py) accepts only the
  completed frozen native-500M surface. It writes a dedicated tuning ledger,
  reader-table draft, and JSON evidence while leaving conclusions and the
  reader README untouched. The ordinary compact 500M writer consumes the same
  fail-closed reader draft; the 50M writer has no RQ5 stage. JSON evidence keeps
  every surface's full correction chain, optimizer-group LR traces, horizon
  state, and selection mapping. Report acceptance recomputes optimizer-group
  trace validity from those published traces and schedule metadata rather than
  trusting the cached verifier flag.
- Schedule curves and optimizer-group scope are implemented in the
  [callback](../../../neuralrec/run/callbacks/lr_schedule.py) and
  [optimizer grouping](../../../dcn/training/optimizer_groups.py).

## RQ6 — Warmup

- Controlled schedule pairs vary only warmup
  ([variants](../configs/variant.py#L339-L362)).
- Warmup steps, inverse-square-root timescale, and the ramp are derived from the
  training horizon ([implementation](../../../neuralrec/run/callbacks/lr_schedule.py#L73-L110),
  [ramp](../../../neuralrec/run/callbacks/lr_schedule.py#L159-L190)).

## RQ7 — Position encodings

- The complete position grid is in
  [variants](../configs/variant.py#L246-L313), with equal-budget
  [proxy tuning](../launchers/architecture/tuning_50m.sh#L63-L83) and
  [final runs](../launchers/architecture/selected_500m.sh#L51-L72).
- Config maps position choices to modules in the transformer
  [builder](../../../dcn/config/networks.py#L181-L242).
- RoPE supports forward and reverse indices
  ([RoPE](../../../dcn/nn/transformer.py#L130-L227)); learned tables support
  both directions ([positions](../../../dcn/nn/transformer.py#L472-L510));
  GPU attention passes the ALiBi slopes to FlashAttention's built-in
  `alibi_slopes` argument
  ([attention](../../../dcn/nn/transformer.py#L328-L413)). The CPU integration
  fallback ignores ALiBi and therefore verifies wiring and shapes, not its
  numerical effect.

## RQ8 — Scaling and architecture

- Dimension, depth, dropout, heads/GQA, FFN width, sequence length,
  normalization, BOS, and windows are declared in the
  [main grid](../configs/variant.py#L220-L245),
  [normalization/BOS grid](../configs/variant.py#L314-L321), and
  [window grid](../configs/variant.py#L468-L469).
- The immutable architecture [fields](../../../dcn/config/settings.py#L53-L74)
  feed the causal transformer [builder](../../../dcn/config/networks.py#L205-L242).
- GQA and local windows are implemented in
  [attention](../../../dcn/nn/transformer.py#L261-L413); BOS is a non-target
  tokenizer [decorator](../../../dcn/models/history_tokens.py#L178-L230).
- The optional CLS [query token](../../../dcn/models/history_tokens.py) is
  inserted after the training history and replaces its final-state query for
  the same held-out target; cutoff inference appends it after the full history
  in the [retrieval model](../../../dcn/models/sequence_retrieval.py).
- Tuned head and normalization treatments are instantiated in the
  [final grid](../launchers/architecture/selected_500m.sh#L74-L90).
- The expanded sequence-length, local-window, and dropout treatments are in
  the μP [proxy launcher](../launchers/architecture/tuning_50m.sh). The
  final-protocol history distribution is generated
  by the [analysis script](../analysis/sequence_length_distribution.py) and
  recorded with its [plot and median](sequence_length_distribution.md).
- The sequence-512 final preserves effective batch 1280 as physical batch 640
  × two accumulation steps in the [final launcher](../launchers/architecture/selected_500m.sh).
  Target-weighted accumulation is implemented in the
  [trainer](../../../neuralrec/run/train.py) and records physical, accumulation,
  and effective batch sizes in training metadata.

## RQ9 — Timestamp-delta embeddings

- Plain/log/binned add/concat deltas, timestamp RoPE, bin counts, and combined
  treatments are in [variants](../configs/variant.py#L475-L537).
- Sequence-aware deltas, clipping, normalization, buckets, and fusion are in
  the history [tokenizer](../../../dcn/models/history_tokens.py#L107-L175).
- Linear and `log1p` gaps are clipped at 30 days. Concatenation uses a
  `2d → 2d → d` projection rather than increasing transformer width; binned
  treatments use normalized `log1p` gaps with dedicated zero-gap and clipped
  boundary buckets. Binned artifacts must record timestamp-bin semantics
  revision 2; earlier artifacts with an unreachable final bucket are retained
  as unusable provenance.
- Continuous timestamp positions map through the network
  [configuration](../../../dcn/config/networks.py#L186-L193) to rotary
  [positions](../../../dcn/nn/transformer.py#L160-L192). Raw RoPE uses elapsed
  days from the first or last valid event; log RoPE uses `log1p` elapsed seconds
  from that anchor, not pairwise gaps.

## RQ10 — Per-layer item embeddings

- The treatment is enabled by its [variant](../configs/variant.py#L470-L474) and
  constructs one table per layer
  ([tables](../../../dcn/config/generation.py#L382-L398)).
- Each lookup is added before its corresponding transformer layer
  ([lookup](../../../dcn/models/sequence_retrieval.py#L43-L54),
  [injection](../../../dcn/nn/transformer.py#L579-L598)).

## RQ11 — Negative sampling and logQ

- The global-q and streaming-frequency objectives follow
  [Yi et al. (2019)](https://doi.org/10.1145/3298689.3346996). The active
  comparison is the native-500M four-family
  [protocol](../protocol/rq11_mixed_streaming_plan.md): uniform catalog,
  streaming in-batch global-q, popularity catalog global-q, and the fully
  corrected aggregate uniform/streaming proposal. Global-q arms keep
  unconditional target-position draws: false-negative masking and own-sequence
  exclusion are disabled.
- The persistent-queue [launcher](../launchers/negatives/rq11_mixed_streaming_500m.sh)
  executes the deterministic joint search, local deep-LR completion, boundary
  continuation, and the final negative-only diagnostic. The
  [candidate manifest](../protocol/rq11_mixed_streaming_manifest.json) fixes
  every candidate and the selector releases later stages only from compatible,
  horizon-complete artifacts.
- The experiment dispatches each criterion in
  [configuration](../../../dcn/config/generation.py#L211-L308).
- Catalog sampling and dense scoring live in
  [random negatives](../../../dcn/nn/sampled_softmax.py#L14-L85); correction,
  masking, and online/offline probabilities live in
  [sampled softmax](../../../dcn/nn/sampled_softmax.py#L142-L331).
