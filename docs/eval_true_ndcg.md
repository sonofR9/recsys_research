# The true metric: full-catalog future-day ranking

In-batch training metrics (`hit_rate`, sampled-softmax loss) score a positive
against a handful of in-batch negatives. They move for reasons that have little
to do with retrieval quality — a bigger batch makes the task harder, a skewed
batch makes it easier — and a model can improve on them while getting worse at
the only thing that matters: putting the items a user will actually touch at the
top of the *whole catalog*.

`dcn/eval/` computes that directly. Rank every catalog item for every user, then
average NDCG@k / Recall@k / MRR@k over the items each user touches on days the
model never saw.

## The pieces

Everything in `dcn/eval/` is model-agnostic and never imports a model.

- **`ranking_metrics.py`** — `ndcg_at_k` / `recall_at_k` / `capped_recall_at_k`
  / `mrr_at_k` over a ranked id list and a relevant-id set. Pure, binary
  relevance, no tensors. The two recalls differ only in the denominator:
  `recall_at_k` divides by the user's positives, `capped_recall_at_k` by
  `min(positives, k)`, which is how the Yambda benchmark defines it.
- **`build_interaction_sets(files, *, user_column, item_id_column, row_filter)`**
  — `{user_id: {item_id, ...}}` read from per-day parquets. Called twice: on the
  *future* days for relevance, on the *training* days for the seen-mask.
  `row_filter` is a polars expression, normally the experiment's own, so a
  likes-only run is judged on likes; the columns it names are read for it.
- **`build_item_snapshot(files, *, item_id_column, columns, timestamp_column)`**
  — one collated row per catalog item, carrying each item's **latest** row in
  the given days. Item side features (counters especially) drift, so the
  snapshot is the state a model would see if it scored the catalog at the
  cutoff. The result is an ordinary event batch.
- **`build_catalog_batch(item_ids, *, item_id_column)`** — the same batch for a
  model whose item tower is a plain embedding table and reads no features, so
  there is nothing to snapshot.
- **`evaluate_true_ndcg(...)`** — takes caller-encoded `query_repr` /
  `item_repr`, ranks by raw dot product, and returns the mean per-user metrics,
  `coverage@k` (the share of the catalog the top-k lists cover between them,
  which is what catches a model answering everyone with the head of the
  catalog) and `num_users`. `max_users` scores a sample instead of everyone,
  drawn by user id so the sample does not move between epochs or runs.
- **`TrueMetricCallback`** — runs the whole thing at epoch end against a live
  model and logs under `epoch/val_true`. Encoding runs under its own `dtype`
  autocast, because the training model's `AutoCast` wrapper is not in the way
  here and a flash-attention sequence model refuses fp32. `score(max_users=...)`
  runs it outside the epoch cadence: a run tunes against a sample and reports
  the whole population once, after training.

## Masking and who gets evaluated

Train-seen catalog items are masked to `-inf` before ranking: re-recommending
something a user already consumed is not retrieval.

That mask forces a matching decision about relevance. A relevant id that was
*also* seen in training is unrecoverable post-mask — it can never appear in the
top-k — so it is dropped from that user's relevance rather than counted as a
guaranteed miss. A user is evaluated iff at least one in-catalog,
not-train-seen relevant id survives; `num_users` reports how many did. Watch
it: a metric that improves while `num_users` moves is not comparable across
runs.

## No leakage

The query and the label come from disjoint day ranges by construction:

- The query loader is built from the **training** days, and a user's query is
  the **last token** of their history — their state at the cutoff.
- `relevance` is built from **strictly later** days.

So nothing the query encoder saw can appear in what it is scored against.

## The model contract

Two methods, which is what retrieval eval needs and what a joint `forward`
cannot provide — the catalog batch has no user column, and the query batch is
not one row per item:

```python
model.encode_queries(batch) -> Tensor  # (num_tokens, dim)
model.encode_items(batch)   -> Tensor  # (num_rows, dim)
```

`TwoTowerModel` and `SimpleTwoTowerModel` both implement them, and their
`forward` is composed from the two, so training and eval cannot drift apart.

## Wiring it into an experiment

`GenerationExperiment` (G1 and the variants under it) always runs it: the
catalog is the items training touched, relevance is the run's own event filter
on the held-out day, and `checkpointing.best_metric_prefix` points the
best-checkpoint rule at `epoch/val_true` so `recall@100` selects. Its `finish()`
re-scores the best checkpoint over every evaluable user and writes
`generated/logs/<run_name>/final_metrics.json`.

`SasRecExperiment` has `enable_true_metric: bool` (on by default in
`SimpleTwoTowerExperiment`). When set, `create_trainer` appends
`_build_true_metric_callback(...)` to the callback list.

Two things such an experiment owns:

- **`item_snapshot_columns`** — the features the *item tower* reads. Get this
  wrong and the snapshot batch is missing a column the tower indexes.
- **`emit_user_column`** on the query loader — `SequenceDataset` groups by user
  but does not emit the id as a token column, and the metric has to attribute
  each query to a user. The experiment turns it on when the model needs it *or*
  when the true metric is enabled.

```python
experiment = SimpleTwoTowerExperiment(enable_true_metric=True)
```

Cost is one full catalog encode plus one pass over the training days per eval;
`every_n_epochs` on the callback throttles it, `user_chunk` bounds the peak
`users x catalog` score matrix.
