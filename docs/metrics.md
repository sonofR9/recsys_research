# What every experiment reports

Machinery in [eval_true_ndcg.md](eval_true_ndcg.md); the protocol it runs under
in [protocol.md](protocol.md).

## Quality

| Metric | k | Read it as |
| --- | --- | --- |
| `recall@k` | 10, 50, 100 | share of the user's held-out positives retrieved. **`recall@100` is the main metric**: it selects checkpoints and ranks variants |
| `ndcg@k` | 10, 50, 100 | ordering quality inside the cut. `ndcg@100` is reported alongside recall |
| `capped_recall@k` | 10, 50, 100 | recall over the positives a k-long list can hold — the [Yambda benchmark's](../yambda_original/yambda/evaluation/metrics.py) definition |
| `mrr@k` | 10, 50, 100 | where the first hit lands; read `@10` |
| `coverage@k` | 10, 50, 100 | share of the catalog the top-k lists touch between them. Catches a model answering everyone with the head of the catalog |
| `num_users` | — | how many users the averages are over. It rescales everything else silently |

`val_loss` stays in the logs as a training-health signal and selects nothing:
sampled softmax over in-batch negatives measures a different question than
ranking the catalog does.

## Reporting format

Per [../experiments/list.md](../experiments/list.md), every variant is reported
as a percentage of the baseline *and* an absolute:

| variant | recall@100 | ndcg@100 |
| --- | --- | --- |
| baseline | 0.1 | 1 |
| descriptive short name | +30% (0.13) | -2% (0.98) |

The experiment's `analysis/collect.py` generates this from what the runs left on
disk. Configurations run under several seeds carry `±` the standard deviation.

## Two traps

**`coverage@k` is a property of the scored population, not of a user** — more
users touch more of the catalog. Compare it only across runs with the same
`num_users`, and not at all between single runs: measured seed-to-seed spread
on it reaches 23% of its own mean, far wider than any other column.

**Yambda's published numbers are not ours.** Aligning `capped_recall@k` aligns
the *definition* only: they score users with no positives as 0 where we drop
them, and they rank a differently built catalog. Their NDCG is further off —
the implementation in this checkout computes its ideal DCG from the actual hits
(`metrics.py:167`), which makes it ≈1 for any user with a hit.

## Cost

Quality alone does not pick a winner, so every run also reports:

| Number | Where |
| --- | --- |
| `resources/params_total`, `params_trainable` | `ResourceUsageCallback` |
| `resources/peak_memory_gb` | same, CUDA only |
| `timing/train_epoch_time` | the trainer |
| epochs to the best checkpoint | the best-checkpoint metadata |

Wall clock to the best checkpoint is the product of the last two.
