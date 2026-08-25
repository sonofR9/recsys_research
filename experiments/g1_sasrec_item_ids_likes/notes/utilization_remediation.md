# Utilization remediation

Measured on 2026-08-14 with real Yambda-50M selected-model runs, one training
per GPU, no repeated sampler, and validation plus handoffs included. Preparation
before the first completed run is excluded.

| profile | batch | mean utilization | zero samples | normal handoff | run time |
| --- | ---: | ---: | ---: | ---: | ---: |
| stopped original queue, GPUs 4–7 | 512 | 24–32% | 34–55% | long gaps | 8–10s training |
| shared preparation, depth 4 | 512 | 61–68% | 1–5% | 0.04–1.55s | 7–9s |
| real larger-batch queue | 1280 | 80% weighted | 2.3% | 0.06–0.11s | 6–8s |
| validation-cadenced queue, GPU 4 | 1280 | 86.7% | 0% | 0.06s normally | 5.7–6.2s |

The cadence profile produced five bitwise-identical metric artifacts and five
bitwise-identical metadata artifacts. The ordinary validation callback now
uses the configured ten-epoch cadence and still runs on the final epoch. A
batch-2048 boundary check reached only 82–83%, used 17.9 GB, and reduced
same-rate recall@100 by 9.84% relative to batch 1280, so it was rejected.

The remaining short-proxy loss is per-process CUDA context and allocator cold
start. A persistent GPU process was rejected for this remediation because safe
lookahead would require new RNG, module, callback, loader-worker, failure, and
foreign-GPU isolation machinery. The full 500M selected training is long enough
to amortize cold start and previously measured 91–93% utilization.

The opt-in long-window gate was rerun cleanly on four A100s after the queue
changes. Mean utilization was 92.57–92.83%, the measured three-run windows were
116.6–117.3 seconds, and all validation/teardown/activation handoffs were
1.59–2.00 seconds. This gate uses batch 1280 and a repeated sampler, so it does
not certify short batch-512 proxies.

The revision-1 RQ1 batch confirmed that distinction. Nineteen of 27 runs
finished before it was stopped: preparation took 18–45 seconds while training
took 3–7 seconds. Each run rescanned 301 daily parquets and started four
forkserver loader workers; up to 24 lookahead processes contended at once. A
separate long steady batch-512 check averaged 82.61% utilization. The existing
real short-run measurement remains 61–68% including handoffs.

Batch 512 was not a scientific constraint. Those manifest-revision-1 RQ1
artifacts remain preserved but are excluded from the corrected evidence. The
active manifest-revision-2 RQ1, architecture, and RQ11 proxy controls use batch
1280; RQ11 keeps its default negative count independently fixed at 512.

The opt-in gate now instantiates `selected_quality_b1280`, the current approved
architecture control, rather than the retired batch-512 LR candidate. Its real
A100 acceptance run will be recorded separately from the historical
measurements above.

Queue-depth refresh previously reread every sweep log in a median 1.28 seconds.
The script/data-group index built once in 1.06 seconds and refreshed in a
median 0.12 seconds on the current artifact set. Commands, repetitions,
environment, and raw timings are in
[`utilization_queue_benchmark.json`](../evidence/utilization_queue_benchmark.json).

Focused queue, GPU-admission, service-history, and gate-configuration tests
passed. The real A100 gate is deferred while every device has a foreign compute
process; no occupied GPU was disturbed.
