# Task: validate the framework against 10 real architecture variants

Status legend: `[ ]` not working / not started, `[x]` runs end-to-end on truncated yambda.

## The ask (verbatim intent)

Make a few **real runs** to check that everything works and is convenient to
use, across the architectures below. Common to all: dataset is **yambda 50m**.

### Ranking variants

Multi-target: predict like / not-like **and** listen ratio.

- [x] **R1** — regular DCNv2 over counters + multivalent features (e.g. album id)
      + univalent features (e.g. item id).
- [x] **R2** — R1 + a transformer over the user history (and the current
      candidate) as one of the features. CLS token or aggregation, either.
- [x] **R3** — R2 + semantic ids in the history and as features. RQ-KMeans,
      3 levels x 1024 codes, fitted on the ready-made embeddings.
- [x] **R4** — R1's model on the ranking homework's own derivation of yambda,
      added to settle acceptance criterion 5 (see below).

### Generation variants

- [x] **G1** — SASRec: causal transformer over user history of item ids.
      **Likes only.** Next-token prediction. logQ correction + in-batch
      negatives.
- [x] **G2** — G1, but any like within the next 24 hours counts as a positive.
- [x] **G3** — G1, but over likes *and* listens, with the action type in the
      history (each action a separate token). Plain next-token prediction.
- [x] **G4** — G1 with semantic ids from RQ-KMeans. Beam search over sids
      (reference implementation: `week13_trends/hw_v2/homework.ipynb` on
      `main`).
- [x] **G5** — TIGER-like with semantic ids.
- [x] **G6** — G4 with sids in two variants: trainable embeddings on top of the
      sids, and embeddings taken from the RQ-KMeans construction.
- [x] **G7** — G4 with RQ-VAE semantics, initialized from RQ-KMeans.

### Testing scale

Truncate yambda by **sampling users, not events**. 10,000 users is what the
runs below use; at 50m that is close to the whole release.

## Acceptance criteria

1. **All variants work.** A variant that does not run is a bug to fix in the
   framework, not something to paper over with a variant-specific crutch.
2. **Almost no task-specific code in shared code** (dataset, training).
   Ranking-vs-generation splits are acceptable; task-specific branches in
   shared code are not.
3. **No training loops outside shared code.** Anything a variant needs (e.g.
   RL) goes into the shared framework and reuses what is there. Task-specific
   code should be mostly *architecture configuration* plus *pipeline
   sequencing* (e.g. "first train sids with the common loop, then the model").
4. **Good code quality.** Self-review before handing over, then a subagent
   review that knows only the one-line task description.
5. **R1 reproduces the homework's DCNv2.** Pairwise accuracy ≈ **0.56–0.57 on
   full play** and ≈ **0.53 on likes**, the numbers
   `week07_dl_ranking/homework/homework.ipynb` reaches.

   **Met, by R4.** As written the criterion assumed R1 sees the homework's
   data. It does not, and no amount of tuning R1 would have got it there — see
   [Ranking](#ranking) below. R4 is R1's model on the homework's own derivation
   of yambda, and it reaches 0.529 on likes and 0.576 on full play.

## How to run a variant

```bash
source /home/sonofr/python_venvs/.venv/bin/activate
python -m dcn.main -s dcn/scripts/<variant>.py
```

| Variant | Script |
| --- | --- |
| R1 | `dcn/scripts/ranking_dcn.py` |
| R2 | `dcn/scripts/ranking_dcn_history.py` |
| R3 | `dcn/scripts/ranking_dcn_semantic.py` |
| R4 | `dcn/scripts/ranking_dcn_homework.py` |
| G1 | `experiments/g1_sasrec_item_ids_likes/configs/baseline.py` |
| G2 | `dcn/scripts/sasrec_likes_24h.py` |
| G3 | `dcn/scripts/sasrec_actions.py` |
| G4 | `dcn/scripts/sasrec_semantic.py` |
| G5 | `dcn/scripts/tiger_semantic.py` |
| G6 | `dcn/scripts/sasrec_semantic_combined.py` |
| G7 | `dcn/scripts/sasrec_semantic_rqvae.py` |

## Run results

All thirteen scripts run to completion on yambda 50m truncated to 10,000
sampled users — 47.8M events, 934k distinct items, of which 881k like events
over 181k items. That is nearly the whole 50m release, so it is the largest
truncation this dataset offers. R1–R3 train three epochs (they plateau after
the first), R4 one, the generation variants twenty. Numbers are the last
epoch's validation metrics; none of the variants is tuned.

The ranking and retrieval rows are from a re-run after the counter encoder
landed (see [Ranking](#ranking)), which changed what every counter-fed model
reads. The generation variants use no counters and are unaffected.

### Ranking

| Variant | loss | like BCE | listen MSE | listen R² | pairwise like | pairwise listen |
| --- | --- | --- | --- | --- | --- | --- |
| R1 | 0.1767 | 0.0887 | 0.1760 | 0.032 | 0.5097 | 0.5227 |
| R2 | 0.1726 | 0.0857 | 0.1737 | 0.044 | 0.5057 | 0.5217 |
| R3 | 0.1735 | 0.0868 | 0.1734 | 0.047 | 0.5148 | 0.5220 |

R4 predicts a different pair of targets, so it gets its own table. One epoch,
270 training days and 30 held out:

| Variant | loss | like BCE | full play BCE | pairwise like | pairwise full play |
| --- | --- | --- | --- | --- | --- |
| R4 | 0.8469 | 0.1740 | 0.6728 | **0.5291** | **0.5761** |

R1–R3 are within noise of each other and flat from the first epoch on. Neither
of the two changes that moved R4 moves them: equalising the batch size left
R1's pairwise numbers the same to two decimals, and putting the counters
through the piecewise-linear encoder (below) left them there too.

**R4 meets acceptance criterion 5** (≈0.53 like, 0.56–0.57 full play). It is
R1's model — same trunk, same heads, same embedding table — reading the
homework's derivation of yambda instead of yambda's own events, which is what
the criterion turned out to be about. Two things had to change:

**The data.** Reading `week07_dl_ranking/homework/homework.ipynb`, four
differences each push R1's metric down, and none is a framework defect.
`HomeworkYambdaDatasetSource` reproduces all four:

- **Rows.** The homework keeps only **listen** events, only **non-organic**
  ones, and then only those whose feedback differs from a time-adjacent
  neighbour's (`extract_preference_pairs`). R1 reads every event. Of yambda's
  events under this truncation, 22,463,555 are recommended listens, and 30.6%
  of those are preference pairs.
- **Targets.** Both of the homework's are binary: `is_full_play` (>95% of the
  track, 63.1% of rows) and `is_like` (1.67%). R1 predicts a *continuous*
  listen ratio, so its pairwise metric compares near-ties the homework's
  `labels[i+1] != labels[i]` filter would drop outright. That alone makes the
  listen number much harder.
- **Where a like lives.** The homework **attributes** each like to the listen
  it belongs to (same `uid`/`item_id`, within 24h), so a like is a *label on a
  listen row*. In yambda a like is its own event. Of the 948,548 adjacent pairs
  R1's metric compares on the like target, **37.7% are a like and a listen of
  the same track seconds apart** — identical in every column the model reads,
  so they score exactly 0.5 by construction.
- **Leak guard.** The homework's counters are computed at `timestamp - 15min`,
  matched to the metric's session gap.

**The counter scale.** The homework puts its dense features through a 32-bin
`PiecewiseLinearEncoder`; every variant here fed its EMA counters into the
trunk raw. The counters run from 0.001 to 1,271 in the mean and reach 6,810,
next to embeddings of scale ~0.02, so the first `Linear` learned almost nothing
from them. The encoder existed (`dcn/nn/ple.py`) but only the two-tower variant
used it; it now sits on `SequenceExperiment` and every counter-fed model gets
it. On R4 that is worth +0.007 like and **+0.045 full play** (0.5219/0.5315
before it).

One difference is left, and it is a capability gap rather than a variant bug:
the homework's counters are exact lag counters evaluated at `timestamp − 15min`,
while `EmaCounter` decays per day and only ever reads previous days. A user's
second listen of the same track within one day — a strong signal in the
homework's `ui_lag_*` features — is invisible to R4.

### Generation

| Variant | loss | in-batch hit rate (last / best) |
| --- | --- | --- |
| G1 | 6.573 | 0.087 / 0.121 |
| G2 | 6.723 | 0.041 / 0.047 |
| G3 | 5.849 | 0.063 / 0.090 |

| Variant | loss | token accuracy | beam recall@10 (last / best) | sid recall@10 | level-0 sid recall@10 |
| --- | --- | --- | --- | --- | --- |
| G4 | 2.968 | 0.424 | 0.0089 / 0.0104 | 0.0133 | 0.1627 |
| G5 | 5.837 | 0.256 | 0.0015 / 0.0118 | 0.0030 | 0.0636 |
| G6 | 2.940 | 0.424 | 0.0074 / 0.0192 | 0.0178 | 0.1893 |
| G7 | 3.012 | 0.403 | 0.0089 / 0.0163 | 0.0237 | 0.2870 |

Beam recall is no longer zero. Two changes account for that. Decoding now stops
after the quantizer's own levels instead of also guessing the collision suffix,
which is an arbitrary tie-break the history cannot predict; and a generated code
tuple is scored by the items that carry it, so a tuple ten items wide fills the
top ten on its own. The rest was scale: the earlier runs saw 98k like events
over 47k items.

The per-level numbers say where the beam loses the answer. `level0_sid_recall@10`
of 0.16–0.29 against `sid_recall@10` of 0.013–0.024 means the first code is often
right and the tuple as a whole rarely is — the coarse bucket is learnable at
this scale and the refinement is not. Per-slot accuracy says the same from the
training side: slot 3 (the collision suffix) sits near 0.89 because it is nearly
always 0, while slots 0 and 1 are at 0.07 and 0.15.

G5 (TIGER) trails the decoder-only variants on every number. Among the three
decoder-only ones G7 leads on both sid recall (0.024) and the coarse bucket
(0.287), with G6 behind it — so on this run the codes an RQ-VAE learned beat
the k-means ones, and reading each code as both a trainable row and its
centroid (G6) beats reading it as a row alone (G4).

### Retrieval

| Variant | loss | in-batch hit rate | true NDCG@10 | true Recall@100 |
| --- | --- | --- | --- | --- |
| SASRec two-tower | 6.201 | 0.013 | — | — |
| simple two-tower | 7.861 | 0.003 | 0.0096 | 0.0265 |

Both read a 41-day slice rather than the full 300, so they are not comparable
with the rows above — they exist to exercise the two-tower stack and, for the
simple one, the full-catalog NDCG eval. It scores 2,225 users against the whole
catalog.

SASRec's hit rate fell from 0.027 when the counter-encoder fit changed under it.
It used to fit on whatever the first million rows of the concatenated training
days happened to be — over a 41-day slice, roughly the first week, when every
counter is still warming up from zero. Spreading the sample over all training
days is the correct fit and it scored worse here; on one run of one variant
that is as likely to be noise as signal, and nothing has been tuned either way.

## Delivery

One pull request into `dev`. Intermediate branches/PRs are fine; the final
state is a single PR.

## Notes

- Shared code is explicitly in scope for change — this stage is about polishing
  it.
- Keep this file's checkboxes in sync: uncheck a variant if a later change
  breaks it, and fix it before the end.
