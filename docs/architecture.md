# Architecture

Data flows: **DatasetSource → DatasetManager → EventDataset / SequenceDataset + DataLoader → model (LossWrapper / AutoCast) → trainer**.

## Entry point

`dcn/main.py` walks an experiment's **stages** in order and runs each one. A
plain single-model run is one stage (the experiment itself); a run that has to
fit something before the model that consumes it — a quantizer, a teacher —
prepends a stage rather than a training loop of its own. Stages are built
lazily so a later one can read what an earlier one wrote.

Configuration is **config-as-code**, not YAML. A run is an `Experiment`
instance (`dcn/config/experiment.py`): top-level scalars (`run_name`,
`base_path`, `seed`, `invalidate_cache`) plus nested settings groups from
`dcn/config/settings.py` (`runtime`, `day_range`, `dataloader`, `pretrain`,
`checkpointing`, `logging`), each a dataclass owning its defaults.
`Experiment` also exposes the factory hooks (`create_dataset_source`,
`_create_model`, `create_criterion`, `create_optimizers`, `create_trainer`, …);
the generic ones read the grouped settings, the experiment-specific ones are
overridden by a subclass. The hooks are parameterless: the class **owns its
derived state** as `functools.cached_property` values (`artifacts`, `counters`,
`dataset_manager`, `num_counters`, `device`, `training_day_bounds`,
`callbacks`, `base_model`), built once on first access. Launch with
`python -m dcn.main -s dcn/scripts/<variant>.py`, where the script exposes a
module-level `experiment`. A variant that has become a *run experiment* — a
question being answered, with a protocol and results — moves its script under
`experiments/<id>_<name>/` next to that write-up; `experiments/list.md` is the
queue. Override a single knob by replacing the group, e.g.
`RankingExperiment(dataloader=DataloaderConfig(batch_size=32))`; unset fields
in the replaced group keep their defaults.

### The experiment hierarchy

```
Experiment                     day-by-day event training (dcn/config/yambda.py)
└── SequenceExperiment         user sequences, temporal split, EpochTrainer
    ├── RankingExperiment      multi-target DCNv2 over counters + features
    │   └── …WithHistory       + a causal transformer over the history
    ├── RetrievalExperiment    the criterion owns the model; catalog scoring
    │   ├── SampledSoftmax…    in-batch negatives with a logQ correction
    │   │   └── TwoTowerRetrieval…     a query tower and an item tower
    │   │       ├── SasRec…    hashed features, counters, history transformer
    │   │       └── SimpleTwoTower…    one embedding per tower
    │   └── HistoryGeneration…     a causal transformer over the history
    │       ├── GenerationExperiment   SASRec over item ids, likes only
    │       │                          (+ SampledSoftmax…)
    │       │   ├── TimeWindow…    any like inside a window is the answer
    │       │   └── Action…        likes and listens, the action its own token
    │       └── Semantic…      the catalog replaced by code tuples, and the
    │           └── Tiger…     softmax by next-code prediction
    └── SemanticExperiment     mixed in wherever semantic ids are needed
```

`dcn/config/networks.py` holds the architecture *builders* more than one
experiment wants, so a variant reads as configuration rather than as a second
copy of the same stack.

## Modules

- `dcn/datasets/` — pluggable dataset sources. `DatasetSource.artifacts` returns `DatasetSourceArtifacts` (a main parquet + the flat list of `columns` to load + precomputed embedding parquets + the timestamp, user and compact-item-id column names). Columns carry no semantic role: the dataset buckets each by its parquet dtype into generic int/float columns, and roles (feature/target/mask/counter) are applied downstream by name. `remap.py` builds compact id mappings for features that have precomputed embeddings (id 0 = unknown).
- `data/` (top-level) — generic preprocessing reused across dataset sources: `split_by_day.py` (writes per-day parquets keyed by `timestamp_column`), `counters/` (EMA counters with multiple half-lives), `preprocessing.py` (`preprocess_counters`). A counter key may be list-valued (e.g. `[uid, artist_id]` where an event has several artists): the row expands to the cartesian product of its key values, every expanded entry is updated and read back, and the per-row values are reduced by each configured aggregation (`min`/`max`/`mean`/`sum`/`std`), yielding one output column per (field, decay, aggregation), suffixed with the aggregation name.
- `dcn/data/` — `DatasetManager` orchestrates per-day caching, counter materialization, and produces `EventDataset` + collator. `EventDataset` exposes rows as generic `int_columns`/`float_columns`; the collators turn every column into a ragged `FeatureValues` (`values` + `offsets`), so a single row can carry a variable-length bag per feature (e.g. several artists for an item). `SequenceDataset` groups the same days into per-user histories (`whole` or `sliding` windows, optional `row_filter`), spilling them into RAM-bounded buckets on disk and reading a bucket at a time; `BucketShuffleSampler` keeps that read pattern sequential. `packed.py` holds the index arithmetic for variable-length batches (repeat, append, split off the tail).
- `dcn/nn/` — reusable layers, in the spirit of `torch.nn`: nothing here knows what a run predicts or which columns a batch carries. `MultiHeadNetwork` = embeddings (`MultiTaskEmbeddingLayer` nn or torchrec backend, with split ratios per task) + per-column `feature_encoders` + `shared_network` + per-task heads. `transformer.py` (varlen attention, pluggable positions), `crossnet.py` / `dcnv2.py`, `resnet.py`, `ffn.py`, `ple.py`, `precomputed_embeddings.py`, `semantic_embedding.py` (an item described by its code tuple), `sampled_softmax.py` (in-batch negatives with a logQ correction).
- `dcn/models/` — the architectures this project trains, assembled from those blocks; everything here reads a batch of user events. `loss_wrapper.py` attaches per-task criterions and metrics to a model, and `criterions.py` pulls each one's prediction, target and mask out of the batch by column name. `two_tower.py` / `simple_two_tower.py` / `sequence_retrieval.py` are the retrieval models; `sequence_targets.py` decides which (query, positive) pairs a packed batch is trained on. `history_tokens.py` turns events into the token sequence a causal model reads — one token per item, or per action, or per semantic-id level — which is where most of the generation variants differ from each other. `token_generation.py` writes an event out as one token per slot, decoder-only or encoder-decoder, with beam search over whatever a `TokenConstraint` allows; `semantic_constraint.py` is the only piece that knows those tokens are semantic ids (a slot may hold codes of its own level, narrowed to the prefixes the trie continues).
- `dcn/semantic/` — semantic ids. `residual_kmeans.py` fits residual k-means over the content embeddings; `rq_vae.py` is an autoencoder whose bottleneck is a residual quantizer (codebooks initialized from k-means). `codes.py` holds the assignment plus the flat `(level, code)` token vocabulary and a collision suffix level, so a generated tuple names an item rather than a bucket; `trie.py` answers which codes may follow a prefix and which item a full tuple names.
- `dcn/eval/` — offline eval, model-agnostic. `base.py` holds the epoch-end shell they share: gate on the epoch count, score a loader batch by batch under the run's precision, pool the per-row scores and log the averages under a prefix. `ranking_metrics.py`: pure binary-relevance NDCG@k / Recall@k / MRR@k, plus the Yambda benchmark's recall, whose denominator stops at k. `true_metric.py`: full-catalog future-day eval — `build_interaction_sets` extracts `{user: item ids}` from per-day parquets (future days → relevance, training days → seen-mask) through the experiment's own row filter, `build_item_snapshot` collates one row per catalog item from its latest training row and `build_catalog_batch` does the same from ids alone; `evaluate_true_ndcg` takes caller-encoded `query_repr`/`item_repr`, ranks the whole catalog per user by dot product with train-seen items masked to `-inf`, and averages the metrics over users with at least one in-catalog unseen relevant id — over a fixed sample of them when `max_users` is set — reporting catalog coverage alongside. `callback.py`: `TrueMetricCallback` runs it at epoch end on a live model (logged under `epoch/val_true`), calling only `model.encode_queries` / `model.encode_items`; `score()` runs it off that cadence, which is how a run reports its final numbers over the whole population. `pairwise.py`: `PairwiseAccuracyCallback` scores whether the model orders *adjacent* events of one user the way the target does, skipping pairs further apart than a session gap or tied in the target, and pooling pairs across batches (logged under `epoch/val_pairwise`). `generation.py`: `GenerationRecallCallback` instead measures the model's own decoding path — each validation sequence gives up its last event, the model beam-searches from what remains, and recall@k of the answer is logged under `epoch/val_beam`. See [eval_true_ndcg.md](eval_true_ndcg.md).
- `dcn/training/` — `DayByDayTrainer` extends `neuralrec.run.train.TrainRunner` to walk days in order, with optional pretraining phase (multiple epochs, optionally shuffled days). `EpochTrainer` instead replays one loader for N epochs, which is what the sequence experiments use, resuming from the epoch a restored checkpoint left off at. `CombinedOptimizer` runs deep + (sparse or torchrec) embedding optimizers together; it can disable one of them (`set_enabled`), but nothing wires that to a freeze schedule yet. `PretrainAware*` callbacks gate validation/checkpointing during pretraining.
- `neuralrec/` — generic, project-agnostic training framework: `TrainRunner`, callbacks (logging, validation, TensorBoard, checkpointing on any logged metric, gradient clipping, parameter counts and peak memory), `AutoCast` (bf16/fp16), `DataLoader` with transforms/GPU prefetch, distributed helpers, metrics. `dcn/` is a consumer of `neuralrec/`.
- `utils/` — shared helpers (`global_config`, dataset download, candidate generators, etc.). Imported as `from utils.X import ...` (note: top-level package, not `dcn.utils`).

## Caching and paths

All generated artifacts (preprocessed parquets, counters, candgen, checkpoints, logs, predictions, per-dataset working dirs) live under `Experiment.base_path` (defaults to `<repo>/competition/generated`, derived from the repo root) and are accessed via `utils.global_config.config` (a singleton initialized in `Experiment.setup`). Use these accessors instead of hard-coding paths. Set `invalidate_cache=True` on the experiment to force rebuild.

Everything per-dataset hangs off `dataset_key`, a hash of the resolved source
parquet, so a run over a different truncation of the same dataset can never be
handed the other one's cached days, counters, sequences or semantic ids.
Sequence caches additionally key on the columns, filter and window they were
built for; semantic ids key on the quantizer that produced them.

## Tests

Tests live under `dcn/tests/` (framework / model unit tests), `neuralrec/tests/` and `data/tests/` (counters, preprocessing). Coverage is intentionally narrow — only logic where unit tests provide real value; a lot of the surrounding code is integration-shaped and not worth unit-testing. `dcn/tests/test_ranking_e2e.py` and `test_generation_e2e.py` do run every variant end to end on a miniature yambda layout, because the wiring between the parts is what breaks.

```bash
pytest -q path/to/test_file.py::test_name   # affected test during development
./test.sh                                  # complete parallel non-GPU suite
TEST_JOBS=4 ./test.sh                      # override adaptive CPU worker count
TEST_E2E_DEVICE=gpu TEST_GPU=4 ./test.sh   # checked GPU for training E2E tests
```

`test.sh` hides GPUs and deselects `slow_gpu` tests. Run those tests directly
with pytest only when a dedicated GPU is available. Miniature training E2E
tests stay on CPU by default. The override rejects a GPU with a foreign process,
monitors it for light use, takes the training queue's GPU gate, and runs those
tests serially while other tests keep running on CPU. The runner imposes no
PyTorch or BLAS thread limit.

## Legacy and third-party

`old/` subfolders throughout the tree (e.g. `dcn/data/old/`, `utils/old/`) and the top-level `week02_version.ipynb` are legacy. `yambda_original/` is the upstream Yambda code (third-party reference).
