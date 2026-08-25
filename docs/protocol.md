# How a run is scored

One protocol behind every number in [metrics.md](metrics.md): hold out the last
day, rank the whole catalog for each user, mask what they already saw. The
machinery is [eval_true_ndcg.md](eval_true_ndcg.md).

- **Split.** The last day of the configured range is held out
  (`validation_days=1`); everything before it trains. Global temporal split, as
  Yambda does it.
- **Query.** One per user: their state at the cutoff, from a `whole`-window
  loader. The training loader slides, and scoring a mid-history state as if it
  were the cutoff measures something else.
- **Catalog.** The items training touched — ranking the whole embedding table
  would mix in rows no gradient ever reached, which are noise with a random
  seed attached.
- **Relevance.** The run's own `row_filter` on the held-out day, so a
  likes-only run is judged on likes. Items already seen in training are masked
  to `-inf` and never scored.
- **Evaluable users.** Those with at least one held-out positive that is in the
  catalog and not train-seen. Everyone else is dropped, not counted as zero. A
  user with fewer than `min_seq_len` training events never reaches the query
  loader at all, which is a second and independent reason not to be scored.
- **Sample vs. final.** Epoch evals score a fixed `eval_max_users` sample keyed
  on a hash of the user id, so the same users are scored whatever order they
  arrive in and a shifting population moves the sample only at its boundary.
  After training, the best checkpoint is re-scored against **every** evaluable
  user, and that report is what an experiment reports. It lands in
  `generated/logs/<run_name>/final_metrics.json`.

## Reading the results

**No difference is claimed without size-matched uncertainty evidence.**
Measured spread on `recall@100` runs from ±0.0006 to ±0.0032 across seeds
depending on the configuration, and two runs of one config under one seed still
differ by ~0.001 from GPU nondeterminism alone. An approved experiment may use
single treatment runs against empirical bands from repeated unchanged controls,
but those bands apply only at the dataset size that produced them. Otherwise,
the compared configurations need repeats of their own.

**Changing how the run is executed re-rolls the dice.** Persistent dataloader
workers move the RNG stream (torch draws a worker seed from the global
generator per iterator created), and fused optimizer kernels, `torch.compile`
and contracted reductions all change float reduction order. None of them is a
correctness change, but results before and after are different draws, so a
results table must come from one code version throughout.
