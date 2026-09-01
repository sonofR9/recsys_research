# Training queue

## Persistent service

Start one scheduler from `competition/`. It stays alive after the starting
shell exits. Existing launchers automatically append their individual
`enqueue` calls to the service while it is running:

```bash
QUEUE_SERVICE="python utils/training_queue/service.py"
$QUEUE_SERVICE start
bash experiments/g1_sasrec_item_ids_likes/launchers/negatives/tuning_50m.sh
$QUEUE_SERVICE status
```

Each sourced `queue.sh` creates a durable batch. Its `enqueue` calls enter the
one shared granular scheduler, and its `drain` seals and waits only for that
batch. Launchers may submit concurrently: a small batch cannot reserve or
underfill the machine while another batch has runnable work.

Run-specific assignments passed to `enqueue` are preserved. If a child also
needs an ambient launcher variable, name it in `TRAINING_QUEUE_FORWARD_ENV`:

```bash
TRAINING_QUEUE_FORWARD_ENV="CUSTOM_DATA_ROOT" \
CUSTOM_DATA_ROOT=/datasets/yambda \
bash experiments/example/sweep.sh
```

Set `TRAINING_QUEUE_REQUIRED_FORWARD_ENV` to the subset whose absence must
abort before a batch is created. Queue-internal, GPU-assignment, and
secret-like names cannot be persisted. G1 launchers register
`G1_DATASET_SIZE` and `WANDB_MODE` as required provenance through their shared
launcher helper.

Submitted runs, batches, and results survive submitting shells under
`generated/training-queue-service/`. `status --json` is available for
monitoring. A launcher exits nonzero if one of its own runs fails; failures do
not fail unrelated launcher batches.

Launchers that require all-or-none submission may pass a versioned JSON job
specification to `service.py submit-batch`. The service publishes and seals the
whole batch under one dispatch lock; an exact retry recovers an interrupted
unsealed submission before publishing the replacement.

Only one service can own scheduling. While it is running, directly sourcing
`queue.sh` submits to it instead of creating a competing embedded queue.

For an exclusive massive GPU test:

```bash
$QUEUE_SERVICE drain
# GPUs are clear after drain returns; queued submissions remain durable.
# Run the explicitly approved GPU test.
$QUEUE_SERVICE resume
```

`pause` prevents undispatched runs from entering preprocessing or training.
Already dispatched runs finish normally. `drain` pauses and waits for all
dispatched preprocessing and training to finish. `resume` continues pending
work. `stop` also preserves pending submissions and exits after dispatched
runs finish; `start` resumes them later.

The persistent scheduler retains the existing GPU admission, periodic light-use
checks, preprocessing lookahead, and per-GPU training locks. Each launcher
still configures and sources the same public shell interface:

```bash
TRAINING_QUEUE_SCRIPT=experiments/example/variant.py
source utils/training_queue/queue.sh

enqueue example_small VARIANT=small
enqueue example_large VARIANT=large
drain
```

The embedded form is also available for one-run debugging when no service is
active.

The queue uses every idle GPU, excludes devices with active compute processes,
and preserves its adaptive preprocessing buffer across submissions. The depth
per GPU is `1 + ceil(preparation time / training-lock time)`, capped at four.
Historical timings are restricted to completed jobs with the same experiment
script and data group; unrelated workloads cannot inflate the buffer.

Optional controls:

The persistent scheduler reads queue-wide controls when `start` runs. Restart
it with `stop` then `start` to change them. Run assignments and
`TRAINING_QUEUE_DATA_GROUP` remain per launcher batch.

- `TRAINING_QUEUE_GPUS="1 3"` restricts eligible physical GPU indices.
- `TRAINING_QUEUE_IN_FLIGHT=2` overrides the calculated depth per training slot.
- `TRAINING_QUEUE_MAX_IN_FLIGHT=4` caps the calculated depth.
- `TRAINING_QUEUE_CPU_THREADS_PER_RUN=N` caps CPU-library threads in each
  queued process. Dataloader workers remain controlled by the experiment.
- `TRAINING_QUEUE_DATA_GROUP=name` lets one exact data/cache configuration
  prebuild under the exclusive data lock once, then allows its queued model
  variants to load the warm data concurrently. Do not set it for a mixed-data
  or mixed-sequence-cache batch.
- `TRAINING_QUEUE_MONITOR_LIGHT_GPUS=1` admits initially busy GPUs after a
  30-second usage check. Idle GPUs start immediately while these checks run.
  Lookahead on other GPUs continues while a check is in progress; once
  admitted, the checked GPU is preferred while it has no queued work.
- `TRAINING_QUEUE_GPU_CHECK_SECONDS=30` and
  `TRAINING_QUEUE_GPU_SAMPLE_SECONDS=1` control the monitoring window and
  sample interval.
- `TRAINING_QUEUE_GPU_RETRY_SECONDS=60` controls how soon an excluded GPU is
  checked again while the queue is waiting for capacity. A failed periodic
  check retains that GPU's exclusive gate across retries.
- `TRAINING_QUEUE_GPU_RECHECK_SECONDS=600` controls periodic per-GPU checks.
  A due GPU stops receiving new runs, finishes its current runs, and is checked
  while training continues on the other GPUs.
- `TRAINING_QUEUE_GPU_SETTLE_SECONDS=2` lets the completed queue process release
  its CUDA context before a periodic check starts sampling.
- `TRAINING_QUEUE_GPU_CHECK_EVIDENCE_DIR=path` changes where rejected light-GPU
  probe samples are retained. The default is
  `generated/training-queue-gpu-checks/`.
- `PYTORCH_CUDA_ALLOC_CONF` defaults to `expandable_segments:True` for queued
  processes. Set it explicitly to use a different PyTorch allocator policy.

A monitored GPU is eligible when utilization and memory stay unchanged below
20%, or when every sample stays at or below 5% utilization and 15% memory.
Immediately before entering its training lock, a queued run also waits while
foreign compute processes hold at least 20% of that GPU's memory. Queue-owned
processes waiting for the GPU lock are included. Queued jobs build the CPU
model, optimizer and callbacks, warm evaluation loaders, start the pinned
training loader, and fetch its first batch before waiting. CUDA context creation,
model activation and all training compute remain inside the GPU lock. This
overlaps the expensive preparation while keeping one training process per GPU.

The queue allows one simultaneous training run per GPU. Additional processes
may prepare ahead, but they share that GPU's single training lock.

## Slow GPU integration tests

Both real-GPU tests are opt-in and require an explicitly dedicated device:

```bash
RUN_SLOW_GPU_TESTS=1 SLOW_GPU_INDEX=1 \
pytest -q dcn/tests/test_slow_gpu_integration.py
```

The launcher test takes about 20 seconds. The fixed-seed G1 metric regression
uses full Yambda-50M, batch size 128, and the shared final-seven-day protocol;
it takes about 1.2 minutes. Two pre-change repetitions were bitwise identical,
so the versioned fixture uses an absolute tolerance of `1e-6`. The selected G1
utilization regression queues three separate one-epoch runs of the current
selected architecture over native Yambda-50M at batch 1280 and requires at
least 90% mean utilization from the end of the first run's first step through
the last run. This includes
validation and queue handoff dips. Its A100 window must also finish within 140
seconds, so slower artificial work cannot satisfy the utilization check.
