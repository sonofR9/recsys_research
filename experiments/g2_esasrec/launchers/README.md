# G2 queue launchers

`protocol/approved_manifest.json` fixes all 135 approved job identities. Training
uses a separate compiled manifest containing exact hyperparameters for the jobs
that are ready to run. Adaptive Optuna jobs are compiled only after their trial
is asked; conditional jobs are included only when their approved trigger fires.
The conditional seed-42 repeat keeps its maximum-budget identity, but an exact
selected seed-42 control artifact replaces it, so the repeat batch submits only
seeds 43 through 51.

Run and durably resume a complete sequential TPE study with one command:

```bash
python -m experiments.g2_esasrec.launchers.optuna_workflow control \
  --database experiments/g2_esasrec/scratchpad/optuna.sqlite3 \
  --compiled experiments/g2_esasrec/scratchpad/compiled_native50m.json
```

The `component` and `mixed` study commands additionally take their verified
control/LiGR run names. Each command asks one approved slot, atomically appends
it to the compiled ledger, submits and waits through the persistent queue,
verifies the artifact, tells Optuna, and only then asks the next slot.
Control and component commands require the queued fit evidence below. It uses
the maximum LiGR width (1536), worst-memory gBCE loss, one A100 optimizer step,
and a user-ID sample capped at 2,000. Control candidates at batches without a
successful persisted probe are ineligible.

Produce all five fit probes through the persistent queue before control
selection:

```bash
bash experiments/g2_esasrec/launchers/queue_fit_probes.sh
```

The exact evidence index is `generated/logs/g2_fit_probes_native50m.json`.
Each job uses the worst-memory LiGR/gBCE recipe at multiplier 6, width 1536,
`t=0.75`, one optimizer step, and
`UserSample(max_users=2000, seed=42)`, whose DuckDB query orders user IDs by
their seeded hash.

After those probes, the complete approved program is one resumable command:

```bash
G2_RECTOOLS_PYTHON=/home/sashanovak/envs/esasrec/bin/python \
python -m experiments.g2_esasrec.launchers.optuna_workflow program \
  --database experiments/g2_esasrec/scratchpad/optuna.sqlite3 \
  --compiled experiments/g2_esasrec/scratchpad/compiled_native50m.json \
  --fit-evidence generated/logs/g2_fit_probes_native50m.json \
  --bands experiments/g2_esasrec/evidence/bands_native50m.json
```

It resumes the control study and boundary points, control repeats and bands,
all six dependency-ordered component studies and boundaries, mixed sampling,
artifact-reuse aggregate selection, three official seeds, the exact selected
artifact benchmark, and generated reports. Unexpected reversal confirmation
remains explicit and requires exactly two verified existing
`--implicated-run` values; confirmations directly source those artifacts. Once
confirmation artifacts exist, the workflow writes exact source/seed/metric
evidence and a human-readable review, then stops final selection and reporting
until the user validates their interpretation.
Ready fixed stages are submitted as one queue batch. The independent standard
sampled-softmax, standard gBCE, and LiGR sampled-softmax studies advance in
parallel rounds; after LiGR capacity resolves, the three capacity-dependent
studies do the same. Adaptive trials within one study remain sequential.

Validate a compiled manifest without submitting work:

```bash
G2_QUEUE_DRY_RUN=1 \
bash experiments/g2_esasrec/launchers/queue_compiled.sh compiled.json
```

Submit through the already-running persistent queue:

```bash
source ~/.bash_aliases
source /home/sonofr/python_venvs/.venv/bin/activate
bash experiments/g2_esasrec/launchers/queue_compiled.sh compiled.json
```

Set `G2_RECTOOLS_PYTHON` to the RecTools 0.19.0 interpreter when the compiled
batch contains official jobs. Every queued run writes `g2_job.json`,
`training_metadata.json`, and `final_metrics.json` below its exact
`generated/logs/<run_name>/` directory. Existing `g2_job.json` files are never
overwritten with a different contract.

Generate the fail-closed native-50M bands, selection evidence, compact RQ
tables, composition evidence, and complete tuning ledger after every required
artifact resolves:

```bash
python -m experiments.g2_esasrec.analysis.generate compiled_complete.json \
  --fit-evidence generated/logs/g2_fit_probes_native50m.json \
  --benchmark experiments/g2_esasrec/evidence/selected_benchmark_native50m.json \
  --composition experiments/g2_esasrec/evidence/composition_native50m.json
```

The report generator recomputes LR-boundary triggers from every initial winner
and rejects omitted triggered slots. It also rejects contract/hash changes,
including any change to the pinned local official catalog adapter, runner,
split/scoring protocol, or provenance source. The RecTools pin covers all 23
executed dataset, fit, sampler, network, similarity, and Torch-ranking source
modules. Local contracts hash the complete enumerated entry, configuration,
data/dataset, model/target/loss, evaluation, training/optimizer, and runtime
path, and reject omitted or changed entries. It also rejects wrong data sizes or seeds,
unresolved best epochs, incomplete metric sets, and missing benchmark evidence
for the selected existing aggregate artifact. The composition document records
all six component candidates, optional eligible mixed candidate, qualification
and final-selection states, baseline fallback, exact point/percent gains,
identical standalone sums, zero interaction gaps, and size-matched unresolved
labels.
The compact aggregate section reports the selected recipe's p50/p95 latency and
throughput. Reader tables use recipe labels and human-readable qualification and
omission prose; exact run identities and encoded states remain only in machine
evidence.
After recall-band ties and exact NDCG ties, selection uses the persisted
end-to-end training-loop `wall_seconds`; epoch-only training time is reported
separately and never substituted for that tie-break.

After aggregate-artifact selection, `program` automatically queues a diagnostic and
waits for `experiments/g2_esasrec/evidence/selected_benchmark_native50m.json`
before report generation. `--benchmark` overrides that destination. The
diagnostic is outside the 135 approved training identities: it rebuilds the
selected local architecture from its verified compiled job, uses fresh weights
with the selected seed and initializer (no checkpoint and zero optimizer
steps), and records that basis plus a state hash.

The latency workload selects exactly 256 queries by the seeded hash of user ID,
independent of input position, and scores the exact 33,148-item mapped catalog.
It runs the production bf16 encoders and float32 full-catalog ranking/top-100
path on one A100, with catalog encoding outside the measured call, 20 warmups,
and 100 synchronized timed iterations. The artifact records the selected user
IDs and hash, catalog identity and hash, dtypes, timing contract, throughput,
and p50/p95 latency. The approved 3,414-user population, selected 256 users,
and packed selected histories (IDs, items, timestamps, offsets, and cumulative
lengths) are pinned by exact hashes. Existing evidence is validated and reused;
it is never silently overwritten.
