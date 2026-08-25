# RQ11 mixed uniform/streaming-global-q plan

## Question and hypothesis

- Question: does a properly corrected mixture of uniform catalog negatives and
  streaming global-q in-batch negatives improve native Yambda-500M metrics over
  either pure source?
- Hypothesis: uniform negatives improve catalog coverage while streaming
  in-batch negatives concentrate training on currently frequent items, so an
  interior mixture can improve recall@100 beyond both endpoints.
- Existing evidence is diagnostic only. Its hyperparameters were selected on
  50M, and its mixed objective corrected only negative logits.

## Comparison

The primary comparison contains four matched families:

1. uniform catalog negatives without logQ;
2. streaming in-batch global-q Yi-2019, correcting the positive and negatives;
3. popularity-sampled catalog negatives with global-q Yi-2019 correction, the
   third method in the current unresolved raw-leader cluster;
4. the aggregate mixture: `f*K` uniform catalog draws and `(1-f)*K` streaming
   in-batch draws, with `q_mix = f*U + (1-f)*q_stream` applied to the positive
   and both negative sources.

One negative-only aggregate-mixture arm is retained as a diagnostic ablation to
explain why the old mixed row is not comparable to the new treatment.

Everything else is held fixed: selected G1 architecture, muP parameterization,
global batch 1280, embedding LR 0.064, seed 42, and the same validation and
stopping discipline. Negative count is independent of training batch.
Every global-q treatment draws targets unconditionally:
`mask_false_negatives=false` and `exclude_own_group_negatives=false`.

## Data and evaluation

- Dataset for tuning and conclusions: native Yambda-500M only.
- Protocol: likes, core items with at least five interactions, final seven-day
  timestamp holdout, mapped training items, full mapped-catalog scoring, and no
  seen-item exclusion.
- Recall@100 is primary. NDCG@100, recall@10, NDCG@10, and coverage@100 are
  secondary.
- Native-500M resolution bands are 0.003 recall, 0.001 NDCG/MRR, and 0.1
  coverage.

## Tuning and selection

The three controls and primary mixture receive equal six-run random-search
budgets. Candidate generation is deterministic and balanced over:

- deep LR: 0.006, 0.012, 0.024;
- total negatives: 512, 1024, 2048;
- streaming alpha where applicable: 0.005, 0.01, 0.02;
- uniform fraction for the mixture: 0.25, 0.5, 0.75.

The exact uniform and popularity-random candidates are `(deep LR, negatives)`:

| candidate | deep LR | negatives |
| --- | ---: | ---: |
| 1 | 0.006 | 512 |
| 2 | 0.012 | 1024 |
| 3 | 0.024 | 2048 |
| 4 | 0.006 | 2048 |
| 5 | 0.012 | 512 |
| 6 | 0.024 | 1024 |

The exact streaming candidates are `(deep LR, negatives, alpha)`:

| candidate | deep LR | negatives | alpha |
| --- | ---: | ---: | ---: |
| 1 | 0.006 | 512 | 0.005 |
| 2 | 0.012 | 1024 | 0.010 |
| 3 | 0.024 | 2048 | 0.020 |
| 4 | 0.006 | 2048 | 0.010 |
| 5 | 0.012 | 512 | 0.020 |
| 6 | 0.024 | 1024 | 0.005 |

The exact aggregate-mixture candidates add uniform fraction:

| candidate | deep LR | negatives | alpha | uniform fraction |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.006 | 512 | 0.005 | 0.25 |
| 2 | 0.012 | 1024 | 0.020 | 0.50 |
| 3 | 0.024 | 2048 | 0.010 | 0.75 |
| 4 | 0.006 | 2048 | 0.020 | 0.50 |
| 5 | 0.012 | 512 | 0.005 | 0.75 |
| 6 | 0.024 | 1024 | 0.010 | 0.25 |

After each family's six joint-search runs, freeze its best secondary setting
(negative count, plus alpha/fraction where applicable) and complete the local
deep-LR trio 0.006/0.012/0.024 at that exact setting. One point already exists,
so this adds at most two runs per family. The diagnostic negative-only mixture
then receives the same deep-LR trio while holding the selected primary
mixture's secondary setting. The complete pre-boundary surface is therefore at
most 35 native-500M runs: 24 joint-search, at most 8 local-LR, and 3 diagnostic.

Boundary resolution is deterministic. Re-select the family winner, then extend
one winning boundary coordinate in this order: negative count, alpha, mixture
fraction, deep LR. The first outward values are 256/4096 negatives,
0.0025/0.04 alpha, 0.125/0.875 fraction, and 0.003/0.048 deep LR. Hold every
other coordinate fixed. After any secondary-axis extension wins, rerun the
local deep-LR trio at the new secondary setting before inspecting the next
coordinate. Repeat the same order geometrically while an outer point wins.

Every run completes the declared 20-epoch linear annealing horizon, validates
each epoch, and restores the best epoch within the horizon. Adaptive early
stopping is inactive for this horizon-bearing schedule. No default extra seed
is used.

Within each family, select by validation recall@100, with NDCG@100 then the
lower-cost configuration breaking unresolved ties. Against each control:

- better requires recall higher by more than 0.003 and NDCG no lower by more
  than 0.001;
- worse means recall lower by more than 0.003, or NDCG lower by more than 0.001
  when recall is within its resolution band;
- a recall gain beyond 0.003 accompanied by an NDCG loss beyond 0.001 is a
  trade-off, not a win;
- every other outcome is unresolved.

The mixture answers yes only if it is better than all three controls.

## Implementation and verification

- Normalize streaming probabilities over valid catalog IDs only.
- Use the realized source fraction after integer allocation.
- Correct the positive and every sampled negative with the aggregate proposal.
- Lock unconditional target draws for global-q arms in focused configuration
  and sampling-path tests.
- Rename the new primary objective separately from the historical
  negative-only ablation and fix stale tests/report labels without deleting raw
  historical artifacts.
- Run focused sampler/config/report tests, then the full non-GPU suite once
  after blind review and only when host CPU load permits.
- Submit all full runs through the persistent shared training queue, with one
  process per admitted GPU and no manual GPU exclusions.

## Additional follow-up idea

If an interior static mixture wins, test an annealed mixture fraction that
starts more uniform and ends more streaming-focused. This is a separate RQ and
is not part of the present budget.

## Approval

- Approved by the user on 2026-08-23: native Yambda-500M, the four primary
  families, the negative-only diagnostic, the up-to-35-run pre-boundary
  surface, and deterministic boundary continuation.
