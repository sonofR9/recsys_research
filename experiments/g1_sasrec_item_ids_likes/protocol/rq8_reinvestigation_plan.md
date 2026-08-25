# RQ8 reinvestigation plan

## Question and hypothesis

- Research question and status: RQ8 is `wip`; recheck query-token training and
  the sequence-length axis.
- Current understanding: the reported end-only CLS gain is a single-seed
  native-500M confirmation selected by an obsolete 50M search whose embedding
  rate reached 0.256. Sequence length 200 used a mismatched LR search and 512
  used physical batch 640, so neither row isolates history length.
- Falsifiable hypothesis: a learned query state improves next-item retrieval;
  supervising the same learned query after every prefix may improve it further.
  Longer retained histories should be non-inferior when positional anchoring,
  batch composition, and tuning are controlled, with reverse-index RoPE
  potentially helping the longest histories.
- Why the result matters: query construction and retained history change every
  downstream architecture result and the future combined model.

## Comparison

- Unchanged query control: item states predict the next item.
- Query treatments: the existing end-only CLS query and an interleaved layout
  `[item1, CLS, item2, CLS, ...]` in which CLS states alone predict subsequent
  items. Every method has exactly one query for each next-item target.
- Query factors held fixed: sequence length 128, learned forward positions,
  a 50-item receptive field, and the existing selected RQ8 architecture.
  Because the local-attention implementation counts physical tokens, the
  corresponding physical windows are 50 for standard, 51 for end-only CLS,
  and 100 for interleaved CLS.
- Sequence treatments: retained history lengths 12, 25, 50, 100, 128, 200,
  256, and 512, each under causal ALiBi and reverse-index RoPE plus ALiBi.
- Sequence factors held fixed: standard item-state autoregression, no CLS,
  full causal attention (`attention_window=None`), physical and effective batch
  1280, and every non-position architecture field. The usable receptive field
  therefore grows with retained history length.
- Sanity control: the standard query at length 128 is regenerated under the
  same native-500M protocol as each comparison it controls.

## Data and evaluation

- Single dataset size: native Yambda-500M for every tuning run, repeat, and
  final result.
- User validation: `yes, use 500m`, 2026-08-23.
- Sampling: no user sampling. Likes only, core items with at least five
  interactions, final seven days held out, mapped items only, full mapped
  catalog, and seen items retained at evaluation.
- Primary metric: recall@100. Secondary metrics: NDCG@100, recall@10,
  NDCG@10, and coverage@100.
- Decision and reader thresholds: 0.003 recall, 0.001 NDCG/MRR, and 0.1
  coverage, the fixed rounded operational form of the native-500M empirical
  bands in `agents/AGENTS.md`.

## Hyperparameter selection

- Embedding LR is 0.064 for every run.
- Deep LR is tuned independently for every query method and every
  position-by-length treatment at 0.006, 0.012, and 0.024.
- If the best result is on a grid boundary, extend once in that direction and
  continue until the selected point is interior or the boundary is resolved.
- Global and physical batch remain 1280. Long-history memory work must preserve
  the physical negative pool; gradient accumulation with a smaller physical
  batch is not admissible evidence.
- Every treatment uses direct sampled-negative scoring instead of materializing
  scores for the full catalog. The two paths have matched logits and gradients
  in the focused regression test; the direct path removes the old sequence-512
  38.75-GiB backward allocation without changing sampled items or the loss.
- The existing linear schedule remains fixed. Every run completes its declared
  20-epoch annealing horizon, validates each epoch, and restores the best epoch
  within that horizon; patience is configured but inactive for the annealed
  schedule.

## Run stages and compute

- Focused tests prove the interleaved training and inference layouts, causal
  target isolation, exactly one target per transition, and correct token-count
  metadata. A distinct local smoke run may check execution but cannot select a
  treatment.
- Query tuning: 3 methods by 3 deep rates, 9 native-500M runs at seed 42.
- Query confirmation: each frozen winner at seeds 43 and 44, 6 additional
  native-500M runs, giving three total seeds per method.
- Corrected sequence tuning: 2 positional treatments by 8 lengths by 3 deep
  rates, 48 native-500M full-causal runs at seed 42, plus deterministic
  boundary continuations. Corrected run identities contain `sequence_fullcausal`
  and protocol revision `r2`; fixed-window `g1_rq8_sequence_*` artifacts cannot
  satisfy corrected candidate lookup.
- All multi-run work enters the persistent training queue as granular jobs,
  without manual GPU exclusions. The corrected initial launcher submits only
  these 48 sequence runs, and sequence caches are shared by history length.
  Completed query runs and confirmations are consumed by selection but are not
  relaunched.

## Interpretation and reporting

- The end-only CLS claim is retained only if its repeated native-500M result
  clears the fixed thresholds against the repeated standard-query control.
- Interleaved CLS is compared with both standard and end-only CLS using the
  same targets and seed pairing.
- Longer-history regressions trigger checks of LR boundaries, physical batch,
  token truncation, masks, position indices, target counts, stopping behavior,
  and history-population coverage. Results are reported as observed; a desired
  monotone conclusion is not imposed on unresolved evidence.
- RQ8 receives one query table and two separate sequence-length tables: causal
  ALiBi and reverse-index RoPE plus ALiBi.
- The obsolete FFN-family table is removed from RQ8. FFN evidence remains in
  RQ4 and its raw artifacts are preserved.
- Full native-500M tuning ledgers and a compact generated RQ8 table accompany
  the reader report. Only a resolved query winner is promoted to the future
  combined model.

## Approval

- Approved scope: three query objectives, all eight history lengths under the
  two named positional treatments, fixed embedding LR 0.064, three-point deep
  LR tuning with boundary continuation, and native Yambda-500M throughout.
- User approval: approved on 2026-08-23; dataset confirmation `yes, use 500m`.
- Corrective approval: on 2026-08-23 the user explicitly approved rerunning the
  48 sequence candidates with full causal attention while preserving the
  native-500M data, positional treatments, lengths, rates, batch, schedule,
  scoring, and deterministic boundary rule above.

## Post-run evidence eligibility

- The query-token surface is complete and reader-eligible: every selected
  method has native-500M confirmations at seeds 42, 43, and 44.
- The completed fixed-window sequence-length surface is blocked from tuning,
  corrected evidence, and reader tables. It used `attention_window=50` with
  two transformer layers, so
  the maximum usable receptive field was 99 physical tokens and retained
  lengths at or above 100 did not expose progressively longer history.
- Sequence length remains `wip`. The separately approved corrected surface uses
  full causal attention and is pending execution; no corrected sequence metric
  is published before selection resolves. All completed raw fixed-window
  artifacts remain preserved for audit and are never reused as corrected
  evidence.
