# Experiment ideas

## Reading the numbers

Report every metric to three decimals. Each experiment uses empirical resolution
bands measured from unchanged control repeats at that experiment's approved
dataset size. Anything inside the applicable band is one result, whichever way
it points, and must not be described as a win, a gain, or a regression.

These bands are practical approximations rather than significance tests. Never
reuse a band across dataset sizes. Native Yambda-500M currently uses 0.003 for
recall, 0.001 for NDCG and MRR, and 0.1 for coverage; other sizes must record
their own bands before making treatment decisions.

## 1 SASRec over item ids, likes only, logQ + in-batch negatives.

   → [g1_sasrec_item_ids_likes/](g1_sasrec_item_ids_likes/)

Purpose and general direction:

1.1 Here you should optimize the transformer (check different variants etc.).
1.2 Report 2 things: the best metrics and the best balance between metrics and performance.
1.3 use utransfer or a better solution if exists for hyperparameters tuning
1.4 At native Yambda-500M, the baseline itself must use the
homework-compatible model and training setup and reproduce recall@100 in the
0.1235–0.13 range. A separate calibration control does not satisfy this
requirement. Ablations from a native-500M baseline outside that range are
intermediate and cannot answer final research questions at that size.
1.5 After changing batch size, reconfirm that the native-500M final baseline
remains in the recall@100 calibration range.
1.6 For RQ11, treat every negative-sampling method as its own family. Tune its
learning rate and useful batch sizes; for mixed random/in-batch methods also
tune mixture and negative count.
1.7 Finalize both the maximum-quality and quality/performance configurations,
list every selected parameter, and record the chosen complete configuration as
the baseline for later experiment groups.
1.8 Treat the 50M-versus-500M comparison as a separately approved dataset-size
study. Train both sizes in the same baseline-style native-dataset regime, but
never repeat 50M to match 500M targets, tokens, steps, or epochs and never reuse
one size's empirical bands for the other. Validate every epoch, stop early when
validation stops improving, restore the best epoch, and use that best epoch for
every reported metric. Epoch count is a safety cap, not a fixed selected
endpoint.

Research questions:

- [done] rq1 Does utransfer work? <!-- work:g1-rq1 -->
- [wip] rq2 What is the best combination for the transformer in terms of metrics (take a look at rqs below)? <!-- work:g1-rq2 -->
- [wip] rq3 What is the best combination for the transformer in terms of balance between metrics and performance (take a look at rqs below)? <!-- work:g1-rq3 -->
- [done] rq4 Does swiglu help? <!-- work:g1-rq4 -->
- [done] rq5 Which lr scheduler works the best? <!-- work:g1-rq5 -->
- [done] rq6 Does lr warmup help? <!-- work:g1-rq6 -->
- [done] rq7 Which one is better: rope, alibi, position embeddings or their combinations? Also you must include variants with position embeddings calculated from the end (so the last event will be always first). It sounds like a good idea to include the same thing for the rope too. <!-- work:g1-rq7 -->
- [done] rq8 How dim affects metrics? And what about num layers, max sequence length, number of heads, groupped query attention, shared window length, normalization kind and place, existance of bos token, may be cls token, ffn ratio? Each dependance should be in separate table. If scaling does not work, it needs a very thorough analysis of why. <!-- work:g1-rq8 -->
- [done] rq9 Is there a variant of timestamp delta embedding which improves metrics? Here interesting to check plain timestamp delta, log timestamp delta and other variants. Some of them you can try to apply in rope. Also it can be a great idea to make bins with timestamp deltas. Please come up with more variants which can help. May be combinations of this variants are better. May be some of the timestamps embeddings should be concatenated. And so on. <!-- work:g1-rq9 -->
- [done] rq10 Does adding additional separate embeddings on each layer help (like in latest versions of gemma)? <!-- work:g1-rq10 -->
- [done] rq11 Does a corrected uniform/streaming global-q mixture beat its independently tuned source families? <!-- work:g1-rq11 -->

## 2 esasrec

- [complete] what is the metrics for esasrec on yambda? Use official implementation from the https://github.com/MTSWebServices/RecTools. If it is possible, report improvement from each pluggable change. <!-- work:g2-esasrec -->

## 3 as 1 but adding pretained embedding

- [not_started] rq1 how replacing item id with pretrained embedding affects metrics?
- [not_started] rq2 how concatenating pretrained embedding to item id embedding affects metrics?
- [not_started] rq3 With concatenated item-ID and pretrained embeddings as the input, what is better to predict: item-ID embedding, pretrained embedding, or their concatenation?
- [not_started] rq4 Does adding other features (album ids, artist ids) improve the metrics?
- [not_started] rq_size Does dataset size change the selected pretrained-embedding treatment's improvement over the learned item-ID baseline? Train both on native 50M and native 500M.

## 4 as 1 but predicting future items during training

Allow predicting future items during training too

- [not_started] rq1 Does predicting any item from the future (in 24 hours window) help? Pinner former style.
- [not_started] rq2 What if predicting any item in some other window? Like within next 10 items?
- [not_started] rq3 What if predict similar hours-days for the user (train separate model to classify if users behaviour will be similar during some other hours/days and use only them as future items)? If it is too hard to implement/ it will run too long, don't try it.

## 5 as 1 over likes and listens, the action its own token

Report both likes and listens.

- [not_started] rq1 Does likes in history help to listens and otherwise (compared to the model which only trained on likes/ listens)?
- [not_started] rq2 Does predicting both likes and listens help to listens or likes (compared to the model which only trained on likes/ listens)?
- [not_started] rq3 Do you need separate token like `<action>` before action and `<item>` before item?
- [not_started] rq4 How metrics will be affected if you will interleave the actions in the following manner: sometimes will be `<want like>item_id<want listen>item_id item_id <like prob, listen_prob><want like>item_id`. The idea is that during serving we want likes => we need to tell the model to predict next like. But it may be beneficial for the model to sometimes predict like prob etc. And it must not if it is like or not.
  Also compare an offline-only sequence of item events without `<want ...>`
  tokens: include `is_like` and listen percentage in each event and predict the
  next item together with its like probability and listen percentage.
- [not_started] rq4.1 How should we aggregate those action tokens? May be we can make some sort of encoder on top of concatenate(action token, item_id) so that history want become longer? Bpe would do it anyways.
- [not_started] rq4.2 Do we need to include loss on predicting both next item and interaction type or not?
- [not_started] rq4.3 may be something else here needs ablation?

## 6 as 1 RQ-KMeans semantic ids in the history

Tune semantic-ID parameters on the downstream recommendation task, including
number of levels and every level's codebook size. For a catalog of about `2^20`
items, no level may exceed `2^13` symbols including collision-resolution
symbols.

Here you should report metrics in both sids (recalls etc.), items as well as sids metrics themself. For sid metrics report:

- exact SID recall and prefix recall by level
- icr
- load balance on each level p95
- intra code similarity

- [not_started] rq0 What is the best way to describe history:
  1. each history item as one token in sequence from concatenation of sids trainable embeddings + emeddings for "sid_0,sid_1" etc. and some projection to model dim on top of that?
  2. concatenation of item id + sids embeddings (codebooks vectors)?
  3. concatenation of item id + sids trainable embedding as in 1 and codebook vector?
  4. each sid a separate token in the sequence (sequence length must be truncated by the number of items, not number of sids)
  5. as in 4 but concatenation of the trainable embedding and the codebook?
  6. as in 4 but sids are untrainable (uses codebook)?
  7. history will have interleaved item id and sid (item_id_0,sid_1_item_id_0, sid_2_item_id_0, ...)?
   (you should tune sids hyperparameters for each variant separately)
- [not_started] rq1 What is the best variant of initalizing sid embeddings in the model? With content from quantization/ random?
- [not_started] rq2 What number of levels, per-level codebook sizes, and other RQ-KMeans parameters work best with a collision-resolution token?
- [not_started] rq3 What number of levels, per-level codebook sizes, and other RQ-KMeans parameters work best without a collision-resolution token?
- [not_started] rq_size Does dataset size change the selected SID-history treatment's improvement over item-ID history? Fit native tokenizers and train both models on native 50M and native 500M.

## 7 semantic id generator

Use rqkmeans here.

Tune semantic-ID parameters on this downstream generation task under the same
`2^13` per-level limit for a catalog of about `2^20` items.

Here you should report metrics in both sids (recalls etc.), items as well as sids metrics themself. For sid metrics report:

- exact SID recall and prefix recall by level
- icr
- load balance on each level p95
- intra code similarity

Use constrained beam search for sids generation.

- [not_started] rq1 lets use sids in the history and generation. Which architecture is better:
1. decoder only model which generates sids?
2. encoder-decoder model?

Research questions for the encoder-decoder model (use **item ids CHECK THIS ONE** in the history):

- [not_started] rq2 What is the best way to describe history in the best variant for encoder-decoder:
  1. each history item as one token in sequence from concatenation of sids trainable embeddings + emeddings for "sid_0,sid_1" etc. and some projection to model dim on top of that?
  2. concatenation of item id + sids embeddings as in 1?
  3. each sid a separate token in the sequence (sequence length must be truncated by the number of items, not number of sids)
  (you should tune sids hyperparameters for each variant separately)
- [not_started] rq3 does usage of the causal transformer vs non causal in encoder part affect metrics?
- [not_started] rq4 lets use causal transformer in the encoder and pretrain it on the next item prediction task. Does it affect the final metrics after the final training?
- [not_started] rq5 Does adding logq correction for sids help? You sohuld try different variants: popularity based on the sid itself without taking into account previous sids of the item and popularity within the same sid prefix.
- [not_started] rq6 Does adding logq correction based on the item popularity (to all sids simultaneously) help?
- [not_started] rq7 Does generating the SID in reverse level order help compared with the normal forward order?

For the decoder only:

- [not_started] rq2_decoder_only What is the best way to serialize history in
  the decoder-only model: one projected SID event token, concatenated item-ID
  and SID event token, or one token per SID level? Truncate by history-item
  count and tune SID parameters separately for every variant.
- [not_started] rq4_decoder_only Does pretraining the decoder-only backbone on
  next-item prediction improve SID generation after full fine-tuning?
- [not_started] rq5_decoder_only Does SID-level logQ/popularity adjustment help
  the decoder-only model? Compare marginal SID popularity, popularity within
  the generated SID prefix, and their combinations.
- [not_started] rq6_decoder_only Does applying concrete-item popularity to the
  complete generated SID loss help the decoder-only model? Compare positive
  loss weighting, sequence-level sampled-softmax correction, and inference
  reranking.
- [not_started] rq7_decoder_only Does reverse SID generation order help the
  decoder-only model?

Encoder RQ3 is not copied: a standard decoder-only model must remain causal
over its combined history-and-target stream. Bidirectional attention over only
the history would be a separate prefix-LM architecture, not this decoder-only
control.

Dataset-size question:

- [not_started] rq_size Does dataset size change the selected semantic-ID generator's improvement over its item-ID retrieval baseline? Fit native tokenizers and train both models on native 50M and native 500M.


## 8 item id in the encoder-decoder sid model

Tune semantic-ID parameters on this joint downstream task under the same
`2^13` per-level limit for a catalog of about `2^20` items.

- [not_started] rq1 What if we add separate item-ID and SID decoders? Also test a separate sequential architecture which first generates the SID and then predicts the item ID.
- [not_started] rq2 Test combinations of item-head logQ and the best SID correction from G7; include SID correction only if it was beneficial in G7.
- [not_started] rq3 Which head is better for the final ranking?
- [not_started] rq4 What if we replace regular decoder-only causal SASRec with an encoder-decoder whose decoder predicts exactly one token, the next item ID?
- [not_started] rq5 What if forward-order and reverse-order SID decoders are trained in parallel, with and without an additional item-ID head?

## 9 semantic ids improvement

Tune every tokenizer family on the downstream recommendation task, not only on
reconstruction or intrinsic SID metrics. Keep every level at or below `2^13`
symbols, including collision-resolution symbols, for a catalog of about `2^20`
items.

rqkmeans
rqvae
plum and modifications: codebooks larger at the beginning as in original PLUM;
the reverse schedule with smaller codebooks at the beginning; and a schedule
which is small at the start and end and largest in the middle
r3vae
variable lengh (bpe) and modifications
built-in into the model (diger)
collision token?
Purely Semantic Indexing for LLM-based Generative Recommendation and Retrieval

## 10 semantic ids with collaborative info

Tune semantic-ID parameters on this downstream recommendation task under the
same `2^13` per-level limit for a catalog of about `2^20` items.

qarm-like (with encoder and pairs)?
pairs and rqvae?

## 10A user-aware semantic-ID tokenization

Tune semantic-ID parameters on this downstream recommendation task under the
same `2^13` per-level limit for a catalog of about `2^20` items.

- [not_started] rq1 Which user-item representation works best for user-aware
  semantic-ID tokenization: summed history plus item; the last hidden layer of
  a monolithic DCNv2/DenseNet like predictor with all available features; or
  concatenated user/item tower vectors from a dot-product like predictor?


## 11 gryphon

Tune semantic-ID parameters on Gryphon's downstream item-retrieval metric under
the same `2^13` per-level limit for a catalog of about `2^20` items.

...

- [not_started] rq_size Does dataset size change Gryphon's improvement over matched vanilla generative retrieval? Fit native tokenizers and train both systems on native 50M and native 500M.

## 12 diffusion

generating audio embedding as in diffusion?
generate audio embedding step by step in transformer. Somewhat like sid but give the model more degrease of freedom? And without necessary quentization?


## 13 rl

- rl with ranking (multi step)

## 14 thinking

- thinking based on the history. For example like this: train likes only model and add listens between like as "thinking stage"
- encoder-decoder sids thinking: during training some times drop some of the sids/ change them to the random ones, but increase the length of sids by repeating the same thing. For example like so: history -> sid_0, "flipped sid_1", sid_2, SID_END, sid_0, sid_1, sid_2, SID_END, END
May be also include traces with reversed sids etc. (I mean first predict sid_2, sid_1 and so on) as an additional variant

## TODO

sids construction:

- additional features: first categorical only, then add counters etc.
- muon etc?
- additional training tasks (predicting counter etc.)

ranking:

- dcnv2 (with densenet as deep part)
- add transformer on top of the history to the dcnv2
- transformer with pretraining
