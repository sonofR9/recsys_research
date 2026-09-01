# G1 — SASRec over item ids, likes only

> Historical snapshot only. No run in this file records the current
> training-semantics revision 2, so none is current evidence or a current
> research conclusion.

Experiment 1 of [../../../list.md](../../../list.md). A causal transformer reads a user's
likes as one token per item and is trained to name the next one, scored by dot
product against the item table. Negatives come from the batch, corrected by
logQ. Nothing but the item id enters the model.

- **[README.md](../../README.md)** — current reader-facing report
- **[research_questions_500m.md](research_questions_500m.md)** — the measured answer,
  hypothesis and tables for every rq1–rq11 question
- **[automated report scratchpad](../../scratchpad/research_questions_500m.md)** —
  generated 50M/500M result tables and the complete metadata-backed
  [50M tuning ledger](../../scratchpad/hyperparameter_tuning_50m.md)
- **[results.md](../50m/results.md)** — archived Yambda-50M results and
  [research-question answers](../50m/research_questions.md)
- Shared background: [metrics](../../../../docs/metrics.md) ·
  [protocol](../../../../docs/protocol.md) · [dataset](../../../../docs/dataset.md)

## Running

```bash
source /home/sonofr/python_venvs/.venv/bin/activate
python -m dcn.main -s experiments/g1_sasrec_item_ids_likes/configs/baseline.py
./experiments/g1_sasrec_item_ids_likes/launchers/core/final_sweep.sh
G1_SEED=1 ./experiments/g1_sasrec_item_ids_likes/launchers/core/final_sweep.sh dim_32
G1_DATASET_SIZE=50m ./experiments/g1_sasrec_item_ids_likes/launchers/core/sweep.sh dim_32
python experiments/g1_sasrec_item_ids_likes/analysis/collect.py --write
```

Config is `dcn/config/generation.py::GenerationExperiment`; `configs/baseline.py` only
states what this experiment pins. Final launchers default to the shared 500M
protocol in `experiments/generation_protocol.py`; smaller runs get dataset and
user-sample suffixes so they cannot overwrite final artifacts.

## Results

Full Yambda-500M likes, items with at least five interactions, and the final
seven days held out by timestamp. Training uses all earlier events and only
items with a trainable embedding; evaluation has 37,018 users and a
157,357-item catalog. The corrected baseline is the homework setup itself:
dim 64, 2 layers, 2-head MHA, GELU FFN 256, LayerNorm, learned forward
positions, batch 128, offline logQ, constant 0.001 learning rates, and 10
epochs. Its four fixed seeds reach recall@100 **0.1271 ±0.0018** and ndcg@100
**0.0483 ±0.0009**, inside the required 0.1235–0.13 calibration range.

The final tuned combination reaches recall@100 **0.14589** and NDCG@100
**0.05578** in its one 500M run. It trains in 13.2 steady seconds/epoch at
23.9 GB. It uses the selected architecture plus input RMSNorm, batch 512,
LR 0.032/0.012, and uniform random negatives. The unchanged selected-control
repeats give the shared practical recall noise band **±0.00049**; this is not a
treatment-specific confidence interval. The lower-memory `selected_balanced`
arm reaches **0.1407 ±0.0012** recall at 13.3 GB.

`runs=4` in the historical tables means the exact same configuration trained
with seeds 0, 1, 2, and 3. New tuning uses one run per configuration. Run counts
never combine nearby hyperparameters; every tuned value has its own row. Each
new table includes the reference parameter value in the reference row.

`capped_recall@100` remains in each raw `final_metrics.json`; the tables show
the required primary metric, `recall@100`.

## Training throughput

The opt-in A100 regression runs three real selected-model trainings on each of
four GPUs. It requires overlapping training across all four devices, exclusive
training per GPU, mean utilization rounding to at least 90% per GPU after the
first run through validation and two process handoffs, and completion within
140 seconds per device. GPU
checks pause only the GPU being observed; other workers continue preparing.
The queue fills its preprocessing lookahead before waiting and shares an exact-
configuration data-ready marker, removing the former 40–50-second handoff gap.

The original fixed-LR batch comparison is not applicable after retuning: its
larger batches reused batch-128 rates. Validation batch 8192 has the same
seed-0 500M recall as 4096 within 0.06%, halves validation time relative to
4096, and avoids the memory cost of 16384.

The batch-128 control averages 64.3% steady-state SM utilization on the
selected-quality model. An Nsight trace attributes the loss to per-query
negative-embedding gathers, dtype copies, and 335,509 eager kernel launches.
Random-negative training now computes one dense catalog score matrix and
gathers the same sampled logits from it; the sampled objective and ids are
unchanged. On the 30,000-user 50M check this preserves recall@100 (0.0636 versus
0.0640), cuts steady epoch time by about 20%, and reduces peak memory from 5.4
GB to 1.35 GB at batch 128. Batched sequence reads remove the next input-
pipeline bottleneck. The small-sample GPU regression uses batch 1280 to provide
a representative A100 workload and begins accounting when the first run ends,
including validation and both subsequent process handoffs.

Retuning embedding LR and deep LR recovers the quality lost by larger batches.
The 50M ranking does not transfer exactly to 500M, so each shortlisted setting
was confirmed once on the full dataset.

| batch | embedding LR | deep LR | 50M recall@100 | 500M recall@100 | 500M ndcg@100 | steady epoch | peak memory |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 512 | 0.004 | 0.012 | 0.06204 | **0.14358** | **0.05484** | 13.0s | 23.9 GB |
| 512 | 0.008 | 0.012 | **0.06941** | 0.14115 | 0.05401 | 13.0s | 23.9 GB |
| 512 | 0.008 | 0.006 | 0.06629 | 0.13932 | 0.05296 | 13.0s | 23.9 GB |
| 1024 | 0.012 | 0.012 | 0.05719 | 0.13578 | 0.05157 | 12–13s | 45.7 GB |
| 1024 | 0.008 | 0.012 | 0.04250 | 0.13484 | 0.05136 | 12–13s | 45.7 GB |
| 1024 | 0.012 | 0.024 | 0.04162 | 0.12680 | 0.04738 | 12–13s | 45.7 GB |

The original batch-512 shortlist was 0.22% below the historical four-seed
quality mean. Subsequent family tuning selected LR 0.032/0.012, and input
RMSNorm lifted the final run to 0.14589 recall in 13.2 seconds/epoch. Full-data
utilization was 91–93%; this is now the future throughput/quality baseline.
Whole-model compilation remains rejected because its first selected-model step
spent over a minute compiling at 0% GPU utilization.

The post-fix μP width-32 LR grid also checks whether tuning transfers across
dataset size. It does not: on 50M, LR 0.1 beats 0.05 (0.07542 versus 0.07021
recall@100), while on 500M LR 0.05 beats 0.1 (0.12567 versus 0.11803). Applying
the 50M winner on 500M loses 6.1% relative to the 500M winner. μP is therefore
used for width transfer; 50M only shortlists LR/batch candidates, which are
then compared once each on 500M.

<!-- RESULTS TABLES -->

<!-- run-prefix: g1_calibrated_ -->

### Learning-rate schedule and warmup

| variant | reference configuration | runs | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 | epoch_time | peak_memory_gb | params_deep | params_embedding | best_epoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | constant; warmup=0%; LR=0.001 | 4 | 0.1271 ±0.0018 | 0.0483 ±0.0009 | 0.0251 ±0.0007 | 0.0204 ±0.0006 | 0.5428 ±0.0132 | 19.9 ±0.6 | 13.2 ±0.0 | 0.107M ±0.000M | 10.1M ±0.0M | 9 ±0 |
| lr_cosine_warmup | — | 4 | -2% (0.1246 ±0.0007) | -2% (0.0473 ±0.0002) | -3% (0.0243 ±0.0002) | -2% (0.0199 ±0.0002) | -32% (0.3700 ±0.0020) | 21.5 ±0.1 | 13.2 ±0.0 | 0.107M ±0.000M | 10.1M ±0.0M | 9 ±0 |
| lr_cosine_cycles2 | — | 4 | -2% (0.1243 ±0.0009) | -2% (0.0472 ±0.0002) | -4% (0.0242 ±0.0002) | -3% (0.0198 ±0.0002) | -32% (0.3673 ±0.0034) | 19.6 ±0.7 | 13.2 ±0.0 | 0.107M ±0.000M | 10.1M ±0.0M | 9 ±0 |
| lr_cosine_cycles4 | — | 4 | -2% (0.1242 ±0.0008) | -2% (0.0472 ±0.0003) | -4% (0.0241 ±0.0001) | -3% (0.0198 ±0.0001) | -32% (0.3702 ±0.0028) | 19.7 ±0.6 | 13.2 ±0.0 | 0.107M ±0.000M | 10.1M ±0.0M | 9 ±0 |

### Embedding learning rate under cosine warmup (baseline: embedding LR=0.001; deep LR fixed at 0.001)

| variant | reference configuration | runs | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 | epoch_time | peak_memory_gb | params_deep | params_embedding | best_epoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lr_cosine_warmup | embedding LR=0.001; deep LR=0.001 | 4 | 0.1246 ±0.0007 | 0.0473 ±0.0002 | 0.0243 ±0.0002 | 0.0199 ±0.0002 | 0.3700 ±0.0020 | 21.5 ±0.1 | 13.2 ±0.0 | 0.107M ±0.000M | 10.1M ±0.0M | 9 ±0 |
| embedding_lr_5e4 | — | 4 | -4% (0.1202 ±0.0013) | -3% (0.0457 ±0.0004) | -3% (0.0235 ±0.0003) | -3% (0.0193 ±0.0002) | -25% (0.2760 ±0.0059) | 19.3 ±0.9 | 13.2 ±0.0 | 0.107M ±0.000M | 10.1M ±0.0M | 9 ±0 |
| embedding_lr_2e4 | — | 4 | -14% (0.1067 ±0.0010) | -15% (0.0403 ±0.0005) | -16% (0.0204 ±0.0004) | -16% (0.0167 ±0.0003) | -54% (0.1714 ±0.0051) | 19.2 ±0.4 | 13.2 ±0.0 | 0.107M ±0.000M | 10.1M ±0.0M | 9 ±0 |
| embedding_lr_1e4 | — | 4 | -31% (0.0865 ±0.0043) | -31% (0.0326 ±0.0017) | -34% (0.0161 ±0.0012) | -32% (0.0136 ±0.0009) | -71% (0.1077 ±0.0096) | 20.5 ±1.8 | 13.2 ±0.0 | 0.107M ±0.000M | 10.1M ±0.0M | 9 ±0 |

### Deep learning rate under cosine warmup (baseline: deep LR=0.001; embedding LR fixed at 0.001)

| variant | reference configuration | runs | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 | epoch_time | peak_memory_gb | params_deep | params_embedding | best_epoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| deep_lr_3e3 | — | 4 | +4% (0.1302 ±0.0008) | +5% (0.0498 ±0.0004) | +6% (0.0258 ±0.0004) | +7% (0.0213 ±0.0003) | +16% (0.4298 ±0.0084) | 19.2 ±0.7 | 13.2 ±0.0 | 0.107M ±0.000M | 10.1M ±0.0M | 9 ±0 |
| deep_lr_5e3 | — | 4 | +4% (0.1299 ±0.0014) | +5% (0.0497 ±0.0005) | +8% (0.0262 ±0.0006) | +8% (0.0214 ±0.0004) | +18% (0.4380 ±0.0043) | 19.6 ±0.5 | 13.2 ±0.0 | 0.107M ±0.000M | 10.1M ±0.0M | 9 ±0 |
| deep_lr_2e3 | — | 4 | +3% (0.1286 ±0.0010) | +4% (0.0491 ±0.0004) | +6% (0.0258 ±0.0005) | +6% (0.0211 ±0.0002) | +14% (0.4230 ±0.0092) | 19.0 ±0.3 | 13.2 ±0.0 | 0.107M ±0.000M | 10.1M ±0.0M | 9 ±0 |
| lr_cosine_warmup | deep LR=0.001; embedding LR=0.001 | 4 | 0.1246 ±0.0007 | 0.0473 ±0.0002 | 0.0243 ±0.0002 | 0.0199 ±0.0002 | 0.3700 ±0.0020 | 21.5 ±0.1 | 13.2 ±0.0 | 0.107M ±0.000M | 10.1M ±0.0M | 9 ±0 |

### Negative sampling and logQ (baseline: 512 in-batch negatives with offline logQ)

| variant | reference configuration | runs | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 | epoch_time | peak_memory_gb | params_deep | params_embedding | best_epoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| neg_random | — | 4 | +5% (0.1332 ±0.0018) | +5% (0.0505 ±0.0005) | +5% (0.0263 ±0.0004) | +5% (0.0214 ±0.0001) | -10% (0.4864 ±0.0194) | 19.7 ±0.4 | 21.5 ±0.0 | 0.107M ±0.000M | 10.1M ±0.0M | 9 ±0 |
| baseline | offline logQ; 512 in-batch negatives | 4 | 0.1271 ±0.0018 | 0.0483 ±0.0009 | 0.0251 ±0.0007 | 0.0204 ±0.0006 | 0.5428 ±0.0132 | 19.9 ±0.6 | 13.2 ±0.0 | 0.107M ±0.000M | 10.1M ±0.0M | 9 ±0 |
| neg_random_offline_logq | — | 4 | -0% (0.1266 ±0.0004) | -1% (0.0479 ±0.0003) | -1% (0.0248 ±0.0003) | -2% (0.0199 ±0.0003) | +21% (0.6563 ±0.0143) | 20.3 ±0.4 | 21.5 ±0.0 | 0.107M ±0.000M | 10.1M ±0.0M | 9 ±0 |
| neg_online_logq | — | 4 | -3% (0.1231 ±0.0009) | -3% (0.0470 ±0.0003) | -1% (0.0248 ±0.0003) | -2% (0.0200 ±0.0002) | +11% (0.6026 ±0.0122) | 19.8 ±0.7 | 13.2 ±0.0 | 0.107M ±0.000M | 10.1M ±0.0M | 9 ±0 |
| neg_in_batch_no_logq | — | 4 | -49% (0.0643 ±0.0011) | -51% (0.0239 ±0.0004) | -51% (0.0123 ±0.0003) | -52% (0.0098 ±0.0001) | +69% (0.9161 ±0.0036) | 19.9 ±0.3 | 13.2 ±0.0 | 0.107M ±0.000M | 10.1M ±0.0M | 9 ±0 |

### Final combinations (reference: cosine warmup, embedding LR=0.001, deep LR=0.001)

| variant | reference configuration | runs | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 | epoch_time | peak_memory_gb | params_deep | params_embedding | best_epoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| selected_quality | — | 4 | +15% (0.1439 ±0.0005) | +18% (0.0556 ±0.0004) | +27% (0.0309 ±0.0005) | +25% (0.0249 ±0.0005) | +24% (0.4589 ±0.0046) | 20.8 ±0.1 | 21.5 ±0.0 | 0.100M ±0.000M | 10.1M ±0.0M | 9 ±0 |
| selected_balanced | — | 4 | +13% (0.1407 ±0.0012) | +15% (0.0546 ±0.0006) | +26% (0.0306 ±0.0005) | +23% (0.0245 ±0.0004) | +26% (0.4648 ±0.0116) | 18.0 ±0.1 | 13.3 ±0.0 | 0.100M ±0.000M | 10.1M ±0.0M | 9 ±0 |
| lr_cosine_warmup | embedding LR=0.001; deep LR=0.001; cosine, warmup=5% | 4 | 0.1246 ±0.0007 | 0.0473 ±0.0002 | 0.0243 ±0.0002 | 0.0199 ±0.0002 | 0.3700 ±0.0020 | 21.5 ±0.1 | 13.2 ±0.0 | 0.107M ±0.000M | 10.1M ±0.0M | 9 ±0 |

### Shared attention window under cosine warmup (baseline: full attention)

| variant | reference configuration | runs | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 | epoch_time | peak_memory_gb | params_deep | params_embedding | best_epoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| window_50 | — | 4 | +1% (0.1260 ±0.0009) | +1% (0.0479 ±0.0005) | +0% (0.0244 ±0.0006) | +2% (0.0202 ±0.0005) | +5% (0.3888 ±0.0057) | 19.3 ±0.2 | 13.2 ±0.0 | 0.107M ±0.000M | 10.1M ±0.0M | 9 ±0 |
| lr_cosine_warmup | full attention; cosine, warmup=5% | 4 | 0.1246 ±0.0007 | 0.0473 ±0.0002 | 0.0243 ±0.0002 | 0.0199 ±0.0002 | 0.3700 ±0.0020 | 21.5 ±0.1 | 13.2 ±0.0 | 0.107M ±0.000M | 10.1M ±0.0M | 9 ±0 |

### Timestamp delta and timestamp RoPE under cosine warmup (baseline: no timestamp-delta feature)

| variant | reference configuration | runs | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 | epoch_time | peak_memory_gb | params_deep | params_embedding | best_epoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| time_bins_16 | — | 4 | +6% (0.1322 ±0.0012) | +7% (0.0505 ±0.0008) | +12% (0.0271 ±0.0007) | +11% (0.0220 ±0.0008) | -7% (0.3456 ±0.0159) | 20.7 ±0.2 | 13.2 ±0.0 | 0.107M ±0.000M | 10.1M ±0.0M | 9 ±0 |
| time_bins_64 | — | 4 | +6% (0.1318 ±0.0009) | +7% (0.0505 ±0.0004) | +11% (0.0270 ±0.0002) | +11% (0.0220 ±0.0004) | -7% (0.3435 ±0.0133) | 20.5 ±0.3 | 13.2 ±0.0 | 0.107M ±0.000M | 10.1M ±0.0M | 9 ±0 |
| time_bins_add | — | 4 | +6% (0.1315 ±0.0007) | +7% (0.0505 ±0.0003) | +13% (0.0275 ±0.0006) | +12% (0.0222 ±0.0004) | -8% (0.3387 ±0.0108) | 20.3 ±0.3 | 13.2 ±0.0 | 0.107M ±0.000M | 10.1M ±0.0M | 9 ±0 |
| time_bins_reverse_rope | — | 4 | +5% (0.1311 ±0.0013) | +6% (0.0502 ±0.0006) | +12% (0.0272 ±0.0003) | +10% (0.0219 ±0.0003) | -7% (0.3455 ±0.0125) | 23.6 ±0.4 | 13.2 ±0.0 | 0.107M ±0.000M | 10.1M ±0.0M | 9 ±0 |
| time_bins_8 | — | 4 | +5% (0.1304 ±0.0009) | +5% (0.0499 ±0.0006) | +11% (0.0269 ±0.0007) | +9% (0.0217 ±0.0006) | -7% (0.3425 ±0.0058) | 20.6 ±0.3 | 13.2 ±0.0 | 0.107M ±0.000M | 10.1M ±0.0M | 9 ±0 |
| lr_cosine_warmup | no time feature; cosine, warmup=5% | 4 | 0.1246 ±0.0007 | 0.0473 ±0.0002 | 0.0243 ±0.0002 | 0.0199 ±0.0002 | 0.3700 ±0.0020 | 21.5 ±0.1 | 13.2 ±0.0 | 0.107M ±0.000M | 10.1M ±0.0M | 9 ±0 |

### muTransfer rate transfer across width (reference: standard width=64)

| variant | reference configuration | runs | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 | epoch_time | peak_memory_gb | params_deep | params_embedding | best_epoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mup_dim128_lr5e2 | — | 4 | +5% (0.1303 ±0.0006) | +6% (0.0502 ±0.0003) | +10% (0.0268 ±0.0001) | +10% (0.0218 ±0.0001) | +43% (0.5284 ±0.0067) | 20.3 ±1.5 | 13.2 ±0.0 | 0.426M ±0.000M | 10.1M ±0.0M | 9 ±0 |
| cosine_dim_128 | — | 4 | +2% (0.1266 ±0.0012) | +3% (0.0486 ±0.0007) | +5% (0.0255 ±0.0005) | +5% (0.0209 ±0.0007) | +36% (0.5031 ±0.0142) | 27.1 ±0.8 | 25.9 ±0.0 | 0.410M ±0.000M | 20.1M ±0.0M | 9 ±0 |
| mup_dim32_lr5e2 | — | 4 | +1% (0.1256 ±0.0008) | +2% (0.0482 ±0.0004) | +3% (0.0250 ±0.0007) | +4% (0.0207 ±0.0005) | +10% (0.4061 ±0.0277) | 21.2 ±0.7 | 13.2 ±0.0 | 0.033M ±0.000M | 10.1M ±0.0M | 9 ±0 |
| lr_cosine_warmup | standard width=64; item embedding dim=64; cosine, warmup=5% | 4 | 0.1246 ±0.0007 | 0.0473 ±0.0002 | 0.0243 ±0.0002 | 0.0199 ±0.0002 | 0.3700 ±0.0020 | 21.5 ±0.1 | 13.2 ±0.0 | 0.107M ±0.000M | 10.1M ±0.0M | 9 ±0 |
| mup_dim32_lr1e1 | — | 4 | -3% (0.1206 ±0.0040) | -3% (0.0460 ±0.0018) | -3% (0.0235 ±0.0013) | -3% (0.0194 ±0.0011) | +2% (0.3760 ±0.0260) | 23.5 ±1.1 | 13.2 ±0.0 | 0.033M ±0.000M | 10.1M ±0.0M | 9 ±0 |
| cosine_dim_32 | — | 4 | -8% (0.1146 ±0.0015) | -9% (0.0432 ±0.0005) | -9% (0.0220 ±0.0003) | -10% (0.0179 ±0.0002) | -31% (0.2536 ±0.0081) | 21.9 ±1.4 | 6.9 ±0.0 | 0.029M ±0.000M | 5.0M ±0.0M | 9 ±0 |
| mup_dim128_lr1e1 | — | 4 | -39% (0.0763 ±0.0581) | -39% (0.0290 ±0.0224) | -37% (0.0154 ±0.0119) | -37% (0.0125 ±0.0097) | -40% (0.2232 ±0.2565) | 22.7 ±0.8 | 13.2 ±0.0 | 0.426M ±0.000M | 10.1M ±0.0M | 9 ±0 |

### Embedding and model dimension under cosine warmup (baseline: dim=64)

| variant | reference configuration | runs | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 | epoch_time | peak_memory_gb | params_deep | params_embedding | best_epoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cosine_dim_128 | — | 4 | +2% (0.1266 ±0.0012) | +3% (0.0486 ±0.0007) | +5% (0.0255 ±0.0005) | +5% (0.0209 ±0.0007) | +36% (0.5031 ±0.0142) | 27.1 ±0.8 | 25.9 ±0.0 | 0.410M ±0.000M | 20.1M ±0.0M | 9 ±0 |
| lr_cosine_warmup | dim=64; cosine, warmup=5% | 4 | 0.1246 ±0.0007 | 0.0473 ±0.0002 | 0.0243 ±0.0002 | 0.0199 ±0.0002 | 0.3700 ±0.0020 | 21.5 ±0.1 | 13.2 ±0.0 | 0.107M ±0.000M | 10.1M ±0.0M | 9 ±0 |
| cosine_dim_32 | — | 4 | -8% (0.1146 ±0.0015) | -9% (0.0432 ±0.0005) | -9% (0.0220 ±0.0003) | -10% (0.0179 ±0.0002) | -31% (0.2536 ±0.0081) | 21.9 ±1.4 | 6.9 ±0.0 | 0.029M ±0.000M | 5.0M ±0.0M | 9 ±0 |

### Grouped-query attention under cosine warmup (baseline: MHA: heads=2, kv_heads=2)

| variant | reference configuration | runs | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 | epoch_time | peak_memory_gb | params_deep | params_embedding | best_epoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lr_cosine_warmup | MHA: heads=2, kv_heads=2; cosine, warmup=5% | 4 | 0.1246 ±0.0007 | 0.0473 ±0.0002 | 0.0243 ±0.0002 | 0.0199 ±0.0002 | 0.3700 ±0.0020 | 21.5 ±0.1 | 13.2 ±0.0 | 0.107M ±0.000M | 10.1M ±0.0M | 9 ±0 |
| cosine_heads_gqa | — | 4 | -0% (0.1244 ±0.0007) | -0% (0.0471 ±0.0002) | -0% (0.0242 ±0.0005) | -1% (0.0198 ±0.0003) | -2% (0.3616 ±0.0058) | 19.9 ±0.3 | 13.2 ±0.0 | 0.098M ±0.000M | 10.1M ±0.0M | 9 ±0 |

### Sequence length under cosine warmup (baseline: max_seq_len=100)

| variant | reference configuration | runs | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 | epoch_time | peak_memory_gb | params_deep | params_embedding | best_epoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lr_cosine_warmup | max_seq_len=100; cosine, warmup=5% | 4 | 0.1246 ±0.0007 | 0.0473 ±0.0002 | 0.0243 ±0.0002 | 0.0199 ±0.0002 | 0.3700 ±0.0020 | 21.5 ±0.1 | 13.2 ±0.0 | 0.107M ±0.000M | 10.1M ±0.0M | 9 ±0 |
| cosine_seq_128 | — | 4 | -0% (0.1241 ±0.0008) | -0% (0.0472 ±0.0006) | -1% (0.0241 ±0.0002) | +0% (0.0199 ±0.0004) | -5% (0.3501 ±0.0127) | 17.9 ±0.6 | 13.3 ±0.0 | 0.108M ±0.000M | 10.1M ±0.0M | 9 ±0 |

### Dropout under cosine warmup (baseline: dropout=0.1)

| variant | reference configuration | runs | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 | epoch_time | peak_memory_gb | params_deep | params_embedding | best_epoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lr_cosine_warmup | dropout=0.1; cosine, warmup=5% | 4 | 0.1246 ±0.0007 | 0.0473 ±0.0002 | 0.0243 ±0.0002 | 0.0199 ±0.0002 | 0.3700 ±0.0020 | 21.5 ±0.1 | 13.2 ±0.0 | 0.107M ±0.000M | 10.1M ±0.0M | 9 ±0 |
| cosine_dropout_0 | — | 4 | -4% (0.1200 ±0.0007) | -3% (0.0459 ±0.0003) | -2% (0.0237 ±0.0005) | -2% (0.0196 ±0.0004) | +23% (0.4567 ±0.0075) | 19.5 ±0.5 | 13.2 ±0.0 | 0.107M ±0.000M | 10.1M ±0.0M | 9 ±0 |

### Feedforward kind under cosine warmup (baseline: GELU, ffn_dim=256)

| variant | reference configuration | runs | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 | epoch_time | peak_memory_gb | params_deep | params_embedding | best_epoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cosine_ffn_swiglu_matched | — | 4 | +2% (0.1267 ±0.0027) | +3% (0.0485 ±0.0010) | +4% (0.0253 ±0.0007) | +5% (0.0208 ±0.0006) | +7% (0.3962 ±0.0154) | 20.1 ±0.6 | 13.2 ±0.0 | 0.107M ±0.000M | 10.1M ±0.0M | 9 ±0 |
| cosine_ffn_swiglu | — | 4 | +1% (0.1261 ±0.0006) | +2% (0.0481 ±0.0003) | +3% (0.0251 ±0.0002) | +3% (0.0205 ±0.0003) | +9% (0.4019 ±0.0181) | 19.8 ±0.4 | 13.2 ±0.0 | 0.140M ±0.000M | 10.1M ±0.0M | 9 ±0 |
| lr_cosine_warmup | GELU, ffn_dim=256; cosine, warmup=5% | 4 | 0.1246 ±0.0007 | 0.0473 ±0.0002 | 0.0243 ±0.0002 | 0.0199 ±0.0002 | 0.3700 ±0.0020 | 21.5 ±0.1 | 13.2 ±0.0 | 0.107M ±0.000M | 10.1M ±0.0M | 9 ±0 |

### Position encoding under cosine warmup (baseline: learned forward positions)

| variant | reference configuration | runs | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 | epoch_time | peak_memory_gb | params_deep | params_embedding | best_epoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lr_cosine_warmup | learned forward positions; cosine, warmup=5% | 4 | 0.1246 ±0.0007 | 0.0473 ±0.0002 | 0.0243 ±0.0002 | 0.0199 ±0.0002 | 0.3700 ±0.0020 | 21.5 ±0.1 | 13.2 ±0.0 | 0.107M ±0.000M | 10.1M ±0.0M | 9 ±0 |
| cosine_pos_rope | — | 4 | -3% (0.1213 ±0.0031) | -3% (0.0461 ±0.0011) | -2% (0.0238 ±0.0006) | -2% (0.0195 ±0.0005) | -8% (0.3399 ±0.0111) | 20.9 ±0.6 | 13.2 ±0.0 | 0.100M ±0.000M | 10.1M ±0.0M | 9 ±0 |
| cosine_pos_rope_reverse | — | 4 | -3% (0.1207 ±0.0016) | -3% (0.0459 ±0.0008) | -2% (0.0238 ±0.0005) | -3% (0.0194 ±0.0004) | -9% (0.3350 ±0.0117) | 20.6 ±0.3 | 13.2 ±0.0 | 0.100M ±0.000M | 10.1M ±0.0M | 9 ±0 |
| cosine_pos_learned_reverse | — | 4 | -8% (0.1144 ±0.0025) | -10% (0.0428 ±0.0008) | -14% (0.0210 ±0.0007) | -13% (0.0173 ±0.0004) | -19% (0.2989 ±0.0120) | 19.6 ±0.3 | 13.2 ±0.0 | 0.107M ±0.000M | 10.1M ±0.0M | 9 ±0 |

### Residual normalization place under cosine warmup (baseline: pre-norm)

| variant | reference configuration | runs | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 | epoch_time | peak_memory_gb | params_deep | params_embedding | best_epoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lr_cosine_warmup | pre-norm; cosine, warmup=5% | 4 | 0.1246 ±0.0007 | 0.0473 ±0.0002 | 0.0243 ±0.0002 | 0.0199 ±0.0002 | 0.3700 ±0.0020 | 21.5 ±0.1 | 13.2 ±0.0 | 0.107M ±0.000M | 10.1M ±0.0M | 9 ±0 |
| cosine_norm_post | — | 4 | -1% (0.1238 ±0.0019) | -1% (0.0467 ±0.0007) | -3% (0.0236 ±0.0006) | -3% (0.0194 ±0.0003) | +12% (0.4149 ±0.0047) | 19.4 ±0.5 | 13.2 ±0.0 | 0.107M ±0.000M | 10.1M ±0.0M | 9 ±0 |

## Findings

Historical results are four unchanged seeds on full Yambda-500M; new tuned
treatments run once and use the shared ±0.00049 practical recall noise band.
The fixed-width μP setup transfers deep LR 0.05: recall rises from 0.1256 at
width 32 to 0.1303 at width 128.

The final combination reaches 0.14589 recall@100. Independently tuned random
negatives beat tuned offline logQ by 2.9%; SwiGLU beats tuned GELU by 2.3%;
learned-forward positions remain best; and input RMSNorm adds 0.8%. Use
`future_baseline` as the runnable future default and
`selected_balanced` only when its 13.3 GB footprint is required.

The accepted 500M scheduler family still selects linear decay; two and four
cosine cycles add no benefit. Targeted corrected-baseline runs select
embedding/deep LRs 0.032/0.012, SwiGLU-192, learned forward positions, 16 additive time bins,
sequence length 128, 2-query/1-KV-head GQA, pre-norm plus input RMSNorm,
dropout 0.1, and attention window 50.

The exact complete future configuration and every rq1–rq11 conclusion are in
[research_questions_500m.md](research_questions_500m.md). Changes within a table are
always relative to the explicit `reference configuration` row; results from
the older 128-dimensional reference family are not numerically mixed with the
corrected 64-dimensional tables.
