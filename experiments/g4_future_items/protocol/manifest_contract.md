# G4 control-manifest contract

Status: approved on 2026-08-28. Submission remains blocked until independent
implementation review passes and the control semantics manifest freezes these
exact reviewed source bytes and the native-50M data identity.

## Canonical bytes and schema

Manifest hashes use the exact bytes emitted by CPython 3.12
`json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
allow_nan=False).encode("utf-8")`, with no appended newline. In particular,
integer and floating-point JSON types remain distinct, so `0` and `0.0` have
different bytes. Load with duplicate-key rejection; missing and unknown keys
are rejected. `control_manifest.json` version 2 is a closed schema: its current
key tree and JSON value types are required exactly, except values under
`tunable` describe allowed search values rather than runtime fields.
All G4 ledgers and the selected-control manifest use schema version 2.

The canonical SHA-256 of the approved family manifest is
`ceccb6d6e73d082dea9502fa64f1e2af88c3460788dffdb4656e5aaf6aebd459`.

## Runtime mapping

| manifest path | runtime field |
| :--- | :--- |
| `fixed.data.day_range` | `DayRangeConfig(start_day, end_day)` |
| `fixed.data.*` except `day_range` | same-named `GenerationExperiment` constructor field |
| `fixed.evaluation.catalog` | `evaluation_catalog` |
| `fixed.evaluation.exclude_seen` | `exclude_seen_from_evaluation` |
| `fixed.evaluation.ks` | `eval_ks` |
| `fixed.evaluation.selection_max_users` | `eval_max_users` |
| other `fixed.evaluation.*` | same-named experiment field |
| `fixed.loss.*` | same-named experiment field |
| `fixed.model.transformer` | `TransformerConfig(**value)` |
| other `fixed.model.*` | same-named experiment field |
| `fixed.training.compile`, `dtype`, `gradient_clip_norm` | `RuntimeConfig(compile, dtype=torch.bfloat16, gradient_clip_norm)` |
| `fixed.training.num_workers`, `prefetch_factor`, `val_batch_size`, `gradient_accumulation_steps` plus batch `512` | `DataloaderConfig(num_workers, prefetch_factor, val_batch_size, gradient_accumulation_steps, batch_size)` |
| `fixed.training.lr_schedule` | `LrScheduleConfig(**value)` |
| `fixed.training.training_semantics_revision` | required emitted metadata value; not a constructor argument |
| other `fixed.training.*` | same-named experiment field |
| anchor/fixed `batch_size` | `dataloader.batch_size`; always `512` |
| selected `embedding_learning_rate`, `deep_learning_rate` | same-named experiment field |
| selected `lr_schedule_horizon_epochs` | both `lr_schedule_horizon_epochs` and `num_epochs` |

The verifier constructs these grouped dataclasses, serializes their resolved
values back through the inverse mapping, and requires exact equality with the
manifest plus the selected tunables. This round trip is a focused test and a
pre-launch check.

## Source and data identity

Before control tuning, `control_semantics_manifest.json` records:

- the approved canonical family-manifest hash;
- the exact sorted G4-owned source-path allowlist frozen in
  `experiments/g4_future_items/protocol/manifest.py`, measured from the G4
  control entry point, control-semantics construction, and their transitive
  local imports;
- SHA-256 for every allowlisted source and rejection of any imported local
  source outside that list;
- resolved main/remap/content-embedding parquet paths, sizes, mtimes, SHA-256,
  and the framework dataset key;
- exact resolved runtime configuration, semantic revision values, split
  cutoff, mapped-catalog hash, and next-item target fixture hash.

It uses the same canonical JSON bytes and is hashed and write-protected before
the first tuning job. No other experiment supplies G4 source paths, hashes, or
anchor configuration.

Preview the derived control freeze without writing:

```bash
python -m experiments.g4_future_items.launchers.freeze_control
```

After review, freeze both the semantics manifest and control-tuning ledger:

```bash
python -m experiments.g4_future_items.launchers.freeze_control --write
```

After tuning, `selected_control_manifest.json` records the semantics-manifest
hash, fixed batch, selected rates/horizon, selection row and tie-break fields,
completed epochs, restored best epoch, exact seed-42 configuration, and its
canonical hash. Its builder consumes the complete frozen base ledger, every
triggered boundary ledger, and exactly one immutable completed artifact set for
every row. It reproduces cumulative selection and rejects incomplete, extra,
nonwinning, or still-boundary-unresolved evidence. G4 launches no control
repeat stage; it reuses only the reviewed native-50M relative calibration.

After implementation and independent review,
`treatment_semantics_manifest.json` records the selected-control hash; the
exact sorted transitive local-source path list imported by each G4 entry point
and every source SHA-256; selector, fold, RNG, target, mask, and artifact schema
revisions; focused-fixture hashes; and exact before/after SHA-256 values for
every changed approved runtime path. A preimplementation source manifest freezes
each approved path's SHA-256, or JSON null when absent, before the first code
edit. The treatment manifest records the post-review hash map and derives its
changed-path list by unequal values; unrelated worktree paths never enter either
map. Its source list may differ from the control list only by actual imported
paths from that changed-path list. It rejects an imported local path absent from
the recorded list and is
frozen before any full selector or treatment job. A later source or schema hash
change is not a run delta.

If an unrelated change later touches a source imported only through the shared
closure, preserve the version-1 control, selected-control, and treatment
manifests and add a version-2 treatment compatibility manifest. The version-2
manifest must bind that complete historical lineage, the unchanged control-data
identity, every current source and fixture hash, the exact old/new hashes for
the reviewed compatibility paths, and reproducible equivalence evidence. That
evidence binds all frozen control/RQ1/RQ2 ledgers and their reconstructed runtime
projections plus every planned RQ3 objective form. The verifier must regenerate
the evidence and reject any mismatch. A version-2 manifest is scoped only to
those frozen G4 native-50M configurations; it does not assert generic behavioral
equivalence for the changed modules. New selector and RQ3 ledgers bind the
version-2 hash, while historical RQ1/RQ2 ledgers retain their original hashes.
Historical-ledger reporting supplies the version-2 manifest as a compatibility
anchor; the verifier validates its current closure before accepting any bound
version-1 lineage document as historical.

The approved implementation may add or modify only this runtime source-path set
before that treatment freeze:

- `dcn/config/generation.py`;
- `dcn/data/packed.py`;
- `dcn/data/sequence_dataset.py`;
- `dcn/eval/ranking_metrics.py`;
- `dcn/models/history_tokens.py`;
- `dcn/models/loss_wrapper.py`;
- `dcn/models/sequence_targets.py`;
- `dcn/nn/sampled_softmax.py`;
- `dcn/training/epoch_trainer.py`;
- `experiments/g4_future_items/__init__.py`;
- `experiments/g4_future_items/configs/__init__.py`;
- `experiments/g4_future_items/configs/control.py`;
- `experiments/g4_future_items/configs/selectors.py`;
- `experiments/g4_future_items/configs/treatments.py`;
- `experiments/g4_future_items/launchers/__init__.py`;
- `experiments/g4_future_items/launchers/compiled.py`;
- `experiments/g4_future_items/launchers/freeze_control.py`;
- `experiments/g4_future_items/launchers/run_control.py`;
- `experiments/g4_future_items/launchers/run_selectors.py`;
- `experiments/g4_future_items/launchers/run_treatments.py`;
- `experiments/g4_future_items/selectors.py`;
- `experiments/g4_future_items/targets.py`;
- `experiments/g4_future_items/protocol/__init__.py`;
- `experiments/g4_future_items/protocol/manifest.py`;
- `experiments/g4_future_items/protocol/materialization.py`;
- `experiments/g4_future_items/protocol/metrics.py`.

The treatment manifest contains the complete control closure plus the actual
changed/imported subset of this set and their hashes; an unused approved path
need not appear. Focused tests, ledgers, evidence, and reports are non-runtime
artifacts and are excluded from runtime source comparison. The approved
non-runtime locations are `dcn/tests/experiments/g4_future_items/`,
`experiments/g4_future_items/protocol/ledgers/`,
`experiments/g4_future_items/evidence/`,
`experiments/g4_future_items/report/`, and the reader-facing
`experiments/g4_future_items/README.md`. Any runtime implementation change
outside the source set requires a revised plan before it is made.

## Permitted deltas

Every submitted job has a closed-schema canonical row in a stage ledger under
`experiments/g4_future_items/protocol/ledgers/`. Base ledgers are compiled and
frozen before their first job; a boundary ledger is compiled and frozen after
the preceding result triggers it and before its first job. The launcher accepts
only a ledger row id. The verifier first requires exact equality with that row,
then compares the resolved job projection with the applicable frozen semantics
manifest. These are the only removable JSON Pointer paths:

| job class | removable paths | exact allowed values |
| :--- | :--- | :--- |
| control tuning | `/run_name`, `/protocol/stage`, `/protocol/trial_id`, `/embedding_learning_rate`, `/deep_learning_rate`, `/lr_schedule_horizon_epochs`, `/seed` | stage `control_tuning`; trial ids 1–20; batch exactly `512`; rates in the manifest intervals; horizon in `{5,10,15,20,25,30}`; seed `42`; every exact value equals its frozen ledger row |
| RQ1 | `/run_name`, `/protocol/stage`, `/protocol/trial_id`, `/objective/id`, `/objective/window_seconds`, `/loss/valid_positive_mask_mode`, `/embedding_learning_rate`, `/deep_learning_rate`, `/lr_schedule_horizon_epochs`, `/seed` | stage `rq1_tuning`; trials 1–12; id `rq1_24h`; window `86400`; mask `next_24h_unique`; batch `512`; rates and horizon use the base surface; seed `42` |
| RQ2 | `/run_name`, `/protocol/stage`, `/protocol/trial_id`, `/objective/id`, `/objective/event_lookahead`, `/loss/valid_positive_mask_mode`, `/embedding_learning_rate`, `/deep_learning_rate`, `/lr_schedule_horizon_epochs`, `/seed` | stage `rq2_tuning`; trials 1–12; id `rq2_next10`; lookahead `10`; mask `next_10_unique`; batch `512`; rates and horizon use the base surface; seed `42` |
| RQ3 deterministic hard | `/run_name`, `/protocol/stage`, `/protocol/trial_id`, `/objective/id`, `/objective/selector_artifact_sha256`, `/objective/period_count`, `/loss/valid_positive_mask_mode`, `/embedding_learning_rate`, `/deep_learning_rate`, `/lr_schedule_horizon_epochs`, `/seed` | stage `rq3_deterministic_tuning`; trials 1–12; id `rq3_deterministic_hard`; selected deterministic artifact hash; period count in `{1,2,4}`; mask `selected_period_union_unique`; batch `512`; rates and horizon use the base surface; seed `42` |
| RQ3 learned hard | `/run_name`, `/protocol/stage`, `/protocol/trial_id`, `/objective/id`, `/objective/selector_artifact_sha256`, `/objective/period_count`, `/loss/valid_positive_mask_mode`, `/embedding_learning_rate`, `/deep_learning_rate`, `/lr_schedule_horizon_epochs`, `/seed` | stage `rq3_learned_hard_tuning`; trials 1–12; id `rq3_learned_hard`; selected learned artifact hash; period count in `{1,2,4}`; mask `selected_period_union_unique`; batch `512`; rates and horizon use the base surface; seed `42` |
| RQ3 learned proportional | `/run_name`, `/protocol/stage`, `/protocol/trial_id`, `/objective/id`, `/objective/selector_artifact_sha256`, `/objective/period_count`, `/loss/valid_positive_mask_mode`, `/embedding_learning_rate`, `/deep_learning_rate`, `/lr_schedule_horizon_epochs`, `/seed` | stage `rq3_learned_proportional_tuning`; trials 1–12; id `rq3_learned_proportional`; selected learned artifact hash; period count in `{1,2,4}`; mask `all_positive_probability_periods_unique`; batch `512`; rates and horizon use the base surface; seed `42` |

Selector search, gate, and fold materialization use a separate closed selector
job schema. Its only per-row fields are `/stage`, `/trial_id` or `/fold_id`,
`/boundary_round`,
`/family`, `/period_width_seconds`, `/lookahead_seconds`,
`/minimum_liked_events`, `/time_tolerance_seconds`, `/frequency_entity`,
`/max_leaf_nodes`, `/learning_rate`, `/l2_regularization`, `/seed`, and declared
input/output artifact slot SHA-256 values plus downstream-frozen result or gate
payload SHA-256 values. Search stage is `selector_search`, with
family in `{time,content,frequency,learned}` and trial id 1–12 within family;
period width is in `{3600,21600,86400}`, minimum liked events in `{1,2,4}`, and
lookahead is `{259200,604800}` for hourly widths or `{1209600,2419200}` for the
daily width, except time uses only `604800` for hourly widths. Time alone has
tolerance in `{0,3600,7200}`; frequency alone has entity in
`{item,artist,album}`; learned alone has maximum leaves in `{7,15,31}`, learning
rate in `[0.01,0.2]`, and L2 in `[0.00001,1]`; every inapplicable field is JSON
null. Every concrete value, null, and artifact hash is frozen in the applicable
ledger row before execution. Search metrics are read from the hashed search
artifact, never its runtime sidecar. Gate rows bind both selected search slots
and payload hashes. Gate stage `selector_gate` has no tunable fields and emits a
hashed decision payload whose digest is frozen in every materialization row.
Materialization stage `selector_materialization` has fold ids 0–4 and uses the
selected configuration only. Base rows have null boundary round; boundary rows
use 1 or 2.

The five materialization rows are executed together by the native cost-gate
launcher inside its single supervised process tree, not as queue jobs. A passing
measurement promotes byte-identical digest-addressed deterministic and learned
artifacts to the runtime artifact root. Every RQ3 tuning or boundary ledger
records the canonical `materialization_cost_evidence_sha256`; compilation and
launch reject a failed resource/load gate, an artifact hash that is not the
measured arm, or a promoted artifact whose manifest/arrays do not verify.
The public native launcher holds one nonblocking project-scoped advisory lock
from before input prewarm through child reap and evidence serialization. It
also recognizes both module and script native-materialization command forms
when checking for foreign work, so concurrent measured launches fail closed.

Recommender boundary ledgers record both `rate_bounds` and `horizon_values`.
They also embed the exact entering row and entering rate/horizon domains; the
entering-row digest is checked against that content. Validation recompiles the
seed-43/44 rows from those frozen inputs and requires canonical equality,
including every non-triggered value. Before compilation and again on load,
recommender boundaries verify the complete predecessor-ledger sequence and one
immutable job-contract/training-metadata/log set per predecessor row, reproduce
the cumulative validation winner, and require the embedded entering row to be
that ledger row. Selector boundaries analogously consume the complete frozen
selector-ledger sequence and each row's hash-verified `artifact.json`,
`artifact.sha256`, and bound `result.json`, then reproduce the canonical learned
family winner. Both boundary schemas store this as `predecessor_evidence` with
exact file identities. Base recommender and selector-search
ledgers are likewise recompiled from their seed-42 anchors and require exact
canonical rows rather than accepting any self-rehashed in-domain coordinate.
Rows add `/protocol/boundary_round` with value `1` or `2`, replace stage by the
corresponding base stage plus `_boundary`, replace trial id by 1–4, and sample
every triggered rate and horizon jointly. Rate intervals use the exact
fourfold extension from `plan.md`. Horizon domains use its exact base,
lower/upper round-one, and lower/upper round-two sets; a non-triggered horizon
is frozen as a singleton. Selector boundary rows analogously use
`/boundary_round`, stage `selector_search_boundary`, and trial id 1–4. Batch,
period count, selector structure, and every non-triggered rate/horizon equal the
entering winner. Every sampled rate and horizon is frozen in the boundary
ledger before launch. No other path, value, source/data identity, objective id,
mask mode, or semantic change is allowed.
