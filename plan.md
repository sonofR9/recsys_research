# Plan for End-to-End Transformer Support

## 1. Goals

Support two transformer roles out of the box:

- transformer as a feature extractor inside a larger graph
- transformer as the final sequence model for next-item prediction

The design should also support mixed inputs in one model:

- sequential categorical features
- non-sequential categorical features
- dense or real-valued features
- outputs of other subnetworks

The key constraint is that transformer support must be graph-composable, not a special case bolted onto [`MultiHeadNetwork`](dcn/nn/multi_head_network.py).

## 2. Main architectural direction

### 2.1 Separate graph composition from concrete blocks

Instead of teaching [`MultiHeadNetwork`](dcn/nn/multi_head_network.py) transformer-specific behavior, introduce a higher-level model composition layer that can assemble networks from reusable nodes:

- feature encoders
- sequence encoders
- pooling adapters
- token projection adapters
- task heads
- final sequence models

[`MultiHeadNetwork`](dcn/nn/multi_head_network.py) should remain a vector model:

- it consumes vector-like inputs
- it produces vector outputs per task
- it does not become sequence-aware internally

For sequence-aware topologies, we should build a separate top-level model that composes:

- vector subnetworks such as [`MultiHeadNetwork`](dcn/nn/multi_head_network.py)
- sequence subnetworks such as [`TransformerEncoder`](dcn/nn/transformer.py:252)
- adapters between vector space and token space

### 2.2 Two valid transformer usage modes

#### A. Transformer as one feature branch

A transformer consumes one or several sequential features, produces a sequence, then that sequence is reduced to a vector via pooling or CLS extraction. That vector can then be concatenated with other vector features and fed into a shared part or task head.

This is a feature-extractor use case.

#### B. Transformer as the final model

A vector model such as [`MultiHeadNetwork`](dcn/nn/multi_head_network.py) produces vector outputs. These outputs are projected into one or more tokens. Those projected tokens are combined with true sequence branches and fed into a final transformer. The final transformer emits sequence outputs for next-item prediction or another sequence loss.

This is a sequence-decoder or sequence-head use case.

## 3. Data pipeline plan

## 3.1 Mixed sequential and non-sequential features

The batch format must distinguish between:

- scalar categorical features
- multivalue but non-temporal bag features
- temporal sequence features
- dense per-example features
- dense per-token features if needed later

The current dataset code already has a partial sequence collator in [`collate_sequence_batch()`](dcn/data/dataset.py:127), but it is not enough as the primary abstraction.

We need a unified batch contract that can represent both:

- standard per-example vector models
- sequence models with mixed inputs

### Proposed batch structure

- `categorical_features`
  - per-example features stored as [`FeatureValues`](dcn/data/features.py)
- `sequence_categorical_features`
  - flattened token features stored as [`FeatureValues`](dcn/data/features.py)
- `dense_features`
  - optional per-example dense tensor
- `sequence_dense_features`
  - optional flattened per-token dense tensor
- `targets`
  - either per-example or per-token depending on task type
- `masks`
  - same rule as targets
- `timestamp`
- `cumulative_lens`
  - only present when sequence features exist
- `sequence_lengths`
  - optional convenience metadata

### Dataset changes

[`EventDataset`](dcn/data/dataset.py:12) should be extended so config can declare, per feature:

- whether it is scalar
- whether it is bag-like
- whether it is temporal sequence
- whether it is an input feature or a target-side feature

This should not hardcode item history or artist history. The dataset source should expose enough metadata for batching and model assembly.

## 3.2 Sequence preprocessing for next-item prediction

For autoregressive training, preprocessing should support history-style construction generically:

- input sequence features built from prefix tokens
- target sequence built from shifted future tokens
- optional aligned side features such as artist history for each item token

This logic should live in preprocessing or dataset preparation, not inside the model.

The general abstraction is:

- build one or more aligned token streams
- optionally shift one stream into targets
- preserve shared `cumulative_lens`

## 4. Model composition plan

## 4.1 Keep vector and sequence models separate

We should define two families of modules:

- vector modules returning `[batch, dimension]`
- sequence modules returning `[total_tokens, dimension]` plus relying on `cumulative_lens`

Then add explicit adapters between them.

### New module categories

#### Vector-producing modules

Examples:

- [`MultiHeadNetwork`](dcn/nn/multi_head_network.py)
- pooled transformer branch
- embedding plus MLP feature tower
- pooled history encoder

#### Sequence-producing modules

Examples:

- [`TransformerEncoder`](dcn/nn/transformer.py:252)
- token embedding branch over aligned histories
- token-level subnetwork on top of item or artist histories

#### Adapters

Examples:

- `TakeClsToken`
- `AveragePoolSequence`
- `MaxPoolSequence`
- `ProjectVectorToTokens`
- `ConcatenateVectors`
- `ConcatenateTokenFeatures`
- `MakeSequence`
- `BroadcastVectorToSequence` if ever needed explicitly

## 4.2 Graph builder instead of hardcoded model shape

[`dcn/main.py`](dcn/main.py) currently builds a fixed shape centered on [`MultiHeadNetwork`](dcn/nn/multi_head_network.py). That is too restrictive.

We should introduce a graph builder with config nodes such as:

- `input_feature`
- `embedding`
- `precomputed_embedding`
- `mlp`
- `dcnv2`
- `transformer`
- `pool`
- `concat`
- `project_to_tokens`
- `make_sequence`
- `multihead_network`
- `task_head`

The builder should validate whether each node is vector-shaped or sequence-shaped.

This will allow expressing both simple and complex models without special-casing transformer logic in [`MultiHeadNetwork`](dcn/nn/multi_head_network.py).

## 5. Pseudo-code examples of supported architectures

## 5.1 Transformer branch inside a vector multi-task model

```text
item_history_tokens = ConcatTokenFeatures(
    Dcnv2(ItemEmbedding(history_item_id)),
    ArtistEmbedding(history_artist_id),
    dim=sequence_dim
)

history_vector_1 = TakeClsToken(
    Transformer(item_history_tokens)
)

artist_history_vector = AveragePoolSequence(
    Transformer(ArtistEmbedding(history_artist_id))
)

shared_input = ConcatVectors(
    history_vector_1,
    artist_history_vector,
    DenseFeatures(some_realvalue_features),
    MultiTaskEmbedding(feature_x).shared
)

multihead_network_1 = MultiHeadNetwork(
    shared_part = SharedSubnetwork(shared_input),
    like_out = Identity(MultiTaskEmbedding(feature_x).shared.like),
    listen_out = Network(
        ConcatVectors(
            Subnetwork(feature_3),
            TakeClsToken(Transformer(history_feature_3_tokens))
        )
    )
)

like_out = Network(
    ConcatVectors(
        multihead_network_1.like_out,
        multihead_network_1.listen_out,
        multihead_network_1.shared_part
    )
)

listen_out = Network(
    ConcatVectors(
        multihead_network_1.listen_out,
        multihead_network_1.shared_part
    )
)
```

Key property:

- transformer outputs are pooled before entering vector-only parts
- [`MultiHeadNetwork`](dcn/nn/multi_head_network.py) remains vector-only

## 5.2 Final transformer on top of vector outputs plus true sequence branch

```text
multihead_network = MultiHeadNetwork(
    shared_part = SharedSubnetwork(base_features),
    like_out = LikeTower(base_features),
    listen_out = ListenTower(base_features)
)

sequence_input = MakeSequence(
    ProjectToTokens(count = 2, source = multihead_network.like_out),
    ProjectToTokens(count = 1, source = multihead_network.listen_out),
    HistoryItemSequenceSubnetwork(history_items, history_artists)
)

final_sequence = Transformer(sequence_input)

next_item_logits = VocabularyProjection(final_sequence)
```

Key property:

- the final transformer is not embedded inside [`MultiHeadNetwork`](dcn/nn/multi_head_network.py)
- vector outputs are adapted into tokens explicitly
- real sequence branches and synthetic projected tokens coexist in one token stream

## 5.3 Pure next-item-prediction transformer setup

```text
history_tokens = ConcatTokenFeatures(
    ItemEmbedding(history_item_id),
    ArtistEmbedding(history_artist_id),
    DenseTokenFeatures(history_realvalue_features)
)

encoded_history = Transformer(history_tokens)
next_item_logits = VocabularyProjection(encoded_history)
loss = NextTokenCrossEntropy(next_item_logits, target_item_id_sequence)
```

Key property:

- no [`MultiHeadNetwork`](dcn/nn/multi_head_network.py) required
- this is the main autoregressive recommendation use case

## 6. Losses plan

The only place where we should add explicit transformer-training options is losses, exactly as requested.

### New loss capabilities

- next-token cross entropy over item vocabulary
- optional masked-token style sequence loss later
- optional loss over only selected positions
- optional pooling-aware loss for pooled transformer branches if needed later

[`LossWrapper`](dcn/nn/losses/loss_wrapper.py) should become shape-aware at the task level:

- vector task
- sequence task

But model composition should stay outside the loss layer. The loss only decides how to compare predictions and targets once the model graph already exists.

## 7. Configuration plan

## 7.1 Feature schema

Config should describe feature roles explicitly:

- scalar categorical
- sequence categorical
- scalar dense
- sequence dense
- precomputed embeddings
- target sequence definitions for autoregressive tasks

## 7.2 Model graph schema

Config should define a graph of named nodes. Each node should specify:

- type
- inputs
- output kind
  - vector or sequence
- output dimension if static
- task ownership if relevant

Example shape of config concepts:

```text
model:
  nodes:
    history_item_embedding: ...
    history_artist_embedding: ...
    history_token_concat: ...
    history_transformer: ...
    history_cls: ...
    base_multihead: ...
    like_to_tokens: ...
    final_sequence: ...
  outputs:
    like: ...
    listen: ...
    next_item: ...
```

This allows one config system to express both current vector models and future sequence models.

## 8. Implementation order

1. Remove the current transformer hack from [`MultiHeadNetwork`](dcn/nn/multi_head_network.py).
2. Define explicit batch metadata for sequence and non-sequence features.
3. Extend dataset and collators to emit mixed-feature batches.
4. Introduce vector modules, sequence modules, and adapters as separate abstractions.
5. Introduce a graph builder in [`dcn/main.py`](dcn/main.py) or extracted builder modules.
6. Keep [`MultiHeadNetwork`](dcn/nn/multi_head_network.py) as a reusable vector node in that graph.
7. Add sequence-task support to [`LossWrapper`](dcn/nn/losses/loss_wrapper.py).
8. Add next-token criterion in [`dcn/nn/losses/criterions.py`](dcn/nn/losses/criterions.py) or adjacent loss modules.
9. Add end-to-end tests for:
   - pooled transformer branch inside vector model
   - final transformer over projected vector outputs plus history sequence
   - pure next-item-prediction transformer
10. Add canonical example configs for each pattern.

## 9. Review notes

The most important design rule is:

- transformers are not a mode of [`MultiHeadNetwork`](dcn/nn/multi_head_network.py)
- transformers are reusable graph nodes that may feed into vector models or become the final sequence model

That keeps the architecture clean and matches the examples you gave.
