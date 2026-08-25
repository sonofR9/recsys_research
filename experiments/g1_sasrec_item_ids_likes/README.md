# G1 — SASRec over item IDs, likes only

A causal transformer predicts the next liked item from item-ID history on native Yambda-50M and Yambda-500M cohorts.
Each settled comparison follows its declared tuning or transfer rule; every usable run validates each epoch and restores its best checkpoint.
A schedule that anneals over a horizon trains that horizon and reports the best epoch inside it; a horizon-free schedule stops early on patience. RQ5 is the approved exception: it uses patience plus from-scratch horizon calibration and accepts only schedules whose planned horizon matches their observed stopping point within its declared tolerance.
Arms are therefore not epoch-matched — the native-500M confirmations span 9 to 41 trained epochs — so a small gap between two treatments can reflect how much of its schedule each one received.
The approved evidence dataset for settled treatment RQs is native Yambda-500M.
Its empirical bands come from ten validation-selected repeats of the unchanged
batch-1280 control: 0.003 for recall, 0.001 for NDCG or MRR, and 0.1 for
coverage. Every metric is reported to three decimals; green and red mark
differences beyond those bands and anything smaller is one result whichever
way it points. These are practical thresholds, not significance tests.

## RQ1 — Does μTransfer work?

Transfer here means that the configured rate stays optimal as the model widens.
μP fixes a base width and MuAdam divides each tensor's rate by its own width
multiplier, so holding 0.032/0.012 across widths already shrinks the effective
step size in proportion to width — that rescaling is what is under test, not a
constant step size.

Each 50M table sweeps one rate and holds the other at the control's value, so a
row's reference is that width's own best point on the same sweep and the metric
cells are what the control's rate costs at that width.

| transformer width | best deep LR | recall@100 at 0.012 | ndcg@100 at 0.012 | reference: this width's best |
| ---: | :---: | :---: | :---: | :---: |
| 16 | 0.024 | <span style="color: red">-6% (0.072)</span> | <span style="color: red">-7% (0.027)</span> | 0.076 / 0.029 |
| 32 | 0.012 | 0% (0.076) | 0% (0.028) | 0.076 / 0.028 |
| 64 (control) | 0.012 | 0% (0.068) | 0% (0.024) | 0.068 / 0.024 |
| 128 | 0.012 | 0% (0.065) | 0% (0.025) | 0.065 / 0.025 |
| 256 | 0.024 | -2% (0.065) | +1% (0.024) | 0.066 / 0.024 |

| transformer width | best embedding LR | recall@100 at 0.032 | ndcg@100 at 0.032 | reference: this width's best |
| ---: | :---: | :---: | :---: | :---: |
| 16 | 0.064 | <span style="color: red">-6% (0.072)</span> | -3% (0.027) | 0.076 / 0.027 |
| 32 | 0.128 | <span style="color: red">-7% (0.076)</span> | <span style="color: red">-7% (0.028)</span> | 0.082 / 0.031 |
| 64 (control) | 0.128 | <span style="color: red">-11% (0.068)</span> | <span style="color: red">-14% (0.024)</span> | 0.076 / 0.028 |
| 128 | 0.128 | <span style="color: red">-14% (0.065)</span> | <span style="color: red">-12% (0.025)</span> | 0.076 / 0.028 |
| 256 | 0.128 | <span style="color: red">-18% (0.065)</span> | <span style="color: red">-19% (0.024)</span> | 0.079 / 0.030 |

| transformer width | 50M-local rate | recall@100 at the shared rate | ndcg@100 at the shared rate | reference: the local rate | epochs |
| ---: | :---: | :---: | :---: | :---: | :---: |
| 16 | 0.032/0.024 | 0% (0.120) | 0% (0.045) | 0.120 / 0.045 | 20/20 |
| 32 | same rate | 0.131 | 0.050 | — | 17/20 |
| 64 (control) | same rate | 0.135 | 0.052 | — | 17/20 |
| 128 | same rate | 0.134 | 0.051 | — | 17/20 |
| 256 | 0.008/0.012 | <span style="color: green">+5% (0.134)</span> | <span style="color: green">+6% (0.051)</span> | 0.128 / 0.048 | 13/20 |

| dataset | batch size | embedding LR | deep LR | best/stopped epoch | epoch cap | recall@100 | ndcg@100 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 50M | 1280 | 0.001 | 0.002 | 56/59 | 60 | 0.100 | 0.037 |
| 500M | 1280 | 0.001 | 0.002 | 20/23 | 40 | 0.127 | 0.048 |

Treatment descriptions:

- μP width transfer fixes the item table at 64 dimensions, varies transformer width from 16 to 256, and applies the control's 0.032/0.012 to every width.
- The shared rate is the control's own selected rate at width 64; the 50M-local rate is the alternative each width selected on the superseded proxy surface.
- Native-size LR reuse is the separate conventional check: keep the width-64 homework recipe and batch 1280 fixed and apply the 50M-selected 0.001/0.002 once on native 500M. Its table reports two data sizes rather than a treatment and a control, so its rows carry no percentage change.

Implementation: [μP and native-size protocol](evidence/implementation.md#rq1--μtransfer-and-dataset-size-protocol), [model-width implementation](../../dcn/config/generation.py#L545-L631), [table generator](analysis/rq1_width_transfer.py), and [artifact-level evidence](evidence/rq1_transfer.md). The parameterization follows [Tensor Programs V](https://arxiv.org/abs/2203.03466).

Analysis: Only the deep-LR table tests μP. The item table keeps a fixed 64
dimensions at every model width, so μP never rescales its rate; the embedding
table shows that the interaction does not move the optimum, which is a weaker
claim.

The deep optimum moves by at most one 2× grid step over a 16× width range,
without direction. Standard parameterization would put that drift near four
steps, so the sweep could have detected a failure. It has no negative control,
though: every width-varying run in this repository is μP, so nothing here rules
out the model simply being width-insensitive in any parameterization.

The 500M table is not yet readable. Four of its five shared-rate runs stopped
short of the 20-epoch annealing horizon while both local-rate comparators
trained it in full, which biases the comparison against the shared rate — the
width-256 win is conservative rather than inflated. Its local rates also come
from the truncated proxy surface; the corrected surface picks 0.064/0.024 at
width 16 and 0.128/0.012 at width 256, and neither has been run on 500M.

Conclusion: On a horizon-complete 50M surface μP transfers the deep rate across
a 16× width range — the optimum stays at 0.012–0.024 with no drift — and the
fixed-width item table's rate does not drift either. Conventional 0.001/0.002
reuse separately succeeds at native size. The 500M half is unsettled: its
confirmations are horizon-truncated and test alternatives the proxy no longer
selects. What clearly does not transfer is dataset size, where the embedding
optimum moves 4× between 50M and 500M; see [the transfer
study](evidence/transfer_study.md).

## RQ4 — Does SwiGLU help?

| activation | plain FFN recall@100 | gated FFN recall@100 | plain FFN ndcg@100 | gated FFN ndcg@100 |
| --- | --- | --- | --- | --- |
| ReLU → ReGLU | 0.132 | +2% (0.135) | 0.050 | <span style="color: green">+4% (0.052)</span> |
| GELU → GEGLU | 0.134 | +2% (0.137) | 0.051 | <span style="color: green">+3% (0.053)</span> |
| SiLU → SwiGLU | 0.132 | <span style="color: green">+3% (0.137)</span> | 0.050 | <span style="color: green">+3% (0.052)</span> |

| layers | GELU recall@100 | SwiGLU recall@100 | GELU ndcg@100 | SwiGLU ndcg@100 |
| ---: | --- | --- | --- | --- |
| 2 | 0.134 | +2% (0.137) | 0.051 | <span style="color: green">+2% (0.052)</span> |
| 4 | 0.135 | +1% (0.136) | 0.052 | 0% (0.052) |
| 8 | 0.137 | +1% (0.139) | 0.053 | <span style="color: green">+3% (0.054)</span> |

Treatment descriptions:

- ReLU, GELU and SiLU use the same plain two-matrix FFN at width 171; ReGLU, GEGLU and SwiGLU use the activation-matched gated three-matrix FFN at parameter-matched width 114.
- The new study matches internal FFN dropout at 0.1, fixes the embedding rate at 0.064, tunes the deep rate independently for every family and depth, and compares GELU with dropout-matched SwiGLU at 2, 4 and 8 layers.

Implementation and evidence: [FFN implementations](../../dcn/nn/ffn.py), [activation/depth launcher](launchers/ffn/activation_depth_500m.sh), [generated table and ledger code](analysis/rq4_activation_depth.py), [complete tuning ledger](scratchpad/rq4_activation_depth_tuning_500m.md), and [protocol and exact evidence](evidence/rq4_activation_depth.md). SwiGLU follows [Shazeer (2020)](https://arxiv.org/abs/2002.05202).

Analysis: On the validation-selected 42-run surface, every gated form is directionally better than its activation-matched plain counterpart. ReGLU over ReLU and GEGLU over GELU are unresolved on recall but resolve on NDCG; SwiGLU over SiLU resolves on both. GELU is numerically the strongest plain activation, but its gaps to ReLU and SiLU remain inside the recall threshold. GEGLU has 0.136632 recall against SwiGLU's 0.136517 at two layers, a 0.000114 gap, so this study does not distinguish the two gated variants.

Conclusion: Dropout-matched SwiGLU is directionally above GELU at every tested depth: +1.67%, +0.74%, and +1.40% recall at 2, 4, and 8 layers. Each recall gap remains below the 0.003 resolution threshold, while the two-layer and eight-layer NDCG gains resolve. The local evidence supports gating as the consistent mechanism, but not a unique SwiGLU win over GEGLU. This aligns with modern LLM practice, where gated FFNs—most commonly SwiGLU—are a standard strong choice, without treating that external consensus as local proof.

## RQ5 — Which learning-rate scheduler works best?

| scheduler | optimizer groups scheduled | schedule parameter | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| constant | both | — | 0.141 | 0.054 | 0.031 | 0.023 | 0.718 |
| linear | both | — | <span style="color: green">+3% (0.145)</span> | <span style="color: green">+3% (0.056)</span> | +1% (0.031) | <span style="color: green">+5% (0.024)</span> | -12% (0.635) |
| linear | deep only | — | +1% (0.143) | +1% (0.054) | +2% (0.031) | +4% (0.024) | -6% (0.672) |
| cosine | both | — | +1% (0.144) | +1% (0.054) | 0% (0.031) | +2% (0.024) | -8% (0.660) |
| cosine | deep only | — | +1% (0.143) | +1% (0.055) | 0% (0.031) | +3% (0.024) | -7% (0.668) |
| polynomial | both | — | -1% (0.140) | -1% (0.053) | -4% (0.030) | -1% (0.023) | -10% (0.646) |
| polynomial | deep only | — | <span style="color: red">-3% (0.137)</span> | <span style="color: red">-4% (0.052)</span> | -7% (0.028) | <span style="color: red">-6% (0.022)</span> | -1% (0.707) |
| exponential | both | — | -1% (0.140) | -1% (0.053) | -3% (0.030) | 0% (0.023) | -5% (0.681) |
| exponential | deep only | — | -2% (0.138) | <span style="color: red">-2% (0.053)</span> | -3% (0.030) | -1% (0.023) | -7% (0.669) |
| step | both | — | +1% (0.143) | +2% (0.055) | +1% (0.031) | +3% (0.024) | -3% (0.697) |
| step | deep only | — | +1% (0.142) | +1% (0.054) | 0% (0.031) | +2% (0.024) | -6% (0.675) |
| WSD | both | warmup=0.05, cycles=1 | +2% (0.144) | <span style="color: green">+2% (0.055)</span> | -1% (0.030) | +2% (0.024) | -7% (0.666) |
| WSD | deep only | warmup=0.05, cycles=1 | +2% (0.144) | <span style="color: green">+2% (0.055)</span> | +1% (0.031) | +4% (0.024) | -8% (0.657) |
| inverse sqrt | both | timescale=0.05 | -1% (0.139) | <span style="color: red">-2% (0.053)</span> | -4% (0.029) | -3% (0.022) | +4% (0.749) |
| inverse sqrt | deep only | timescale=0.05 | <span style="color: red">-2% (0.138)</span> | <span style="color: red">-3% (0.052)</span> | -3% (0.030) | -4% (0.022) | +2% (0.730) |
| cosine, warmup 5%, 1 cycle | both | warmup=0.05, cycles=1 | +1% (0.143) | +1% (0.054) | -4% (0.030) | +2% (0.023) | -13% (0.623) |
| **cosine, warmup 5%, 1 cycle** | **deep only** | **warmup=0.05, cycles=1** | **<span style="color: green">+3% (0.146)</span>** | **<span style="color: green">+3% (0.055)</span>** | **+4% (0.032)** | **<span style="color: green">+5% (0.024)</span>** | **-10% (0.647)** |
| cosine, warmup 5%, 2 cycles | both | warmup=0.05, cycles=2 | -2% (0.139) | <span style="color: red">-2% (0.052)</span> | -7% (0.028) | -4% (0.022) | -8% (0.658) |
| cosine, warmup 5%, 2 cycles | deep only | warmup=0.05, cycles=2 | 0% (0.141) | 0% (0.054) | -1% (0.030) | +1% (0.023) | -4% (0.686) |
| cosine, warmup 5%, 4 cycles | both | warmup=0.05, cycles=4 | -2% (0.139) | <span style="color: red">-2% (0.053)</span> | -7% (0.029) | -2% (0.023) | -5% (0.680) |
| cosine, warmup 5%, 4 cycles | deep only | warmup=0.05, cycles=4 | -1% (0.139) | <span style="color: red">-2% (0.053)</span> | -3% (0.030) | -2% (0.023) | 0% (0.720) |
| cosine, tuned warmup | both | warmup=0.05, cycles=1 | <span style="color: red">-2% (0.138)</span> | <span style="color: red">-3% (0.052)</span> | -6% (0.029) | -3% (0.022) | -6% (0.673) |
| cosine, tuned warmup | deep only | warmup=0.0209409463814, cycles=1 | +1% (0.143) | 0% (0.054) | -5% (0.029) | -1% (0.023) | <span style="color: red">-21% (0.569)</span> |

Treatment descriptions:

- Every row uses native Yambda-500M, seed 42, batch 1280, and one fixed embedding LR of 0.064. The selected deep LR is tuned separately for every treatment and scope.
- `both` applies the schedule multiplier to the embedding and deep optimizer groups; `deep only` keeps the embedding rate constant and schedules only the deep group. Constant needs one row because its two scope labels are mathematically identical.
- WSD uses 5% warmup, a stable plateau, and final cosine decay. The fixed-warmup cosine rows compare one, two, and four cycles; the tuned-warmup treatment is a bounded two-parameter screen rather than exhaustive optimization.

Implementation and evidence: [scheduler variants](configs/rq5_scheduler_variant.py), [schedule equations](../../neuralrec/run/callbacks/lr_schedule.py), [approved protocol](protocol/rq5_scheduler_remediation_plan.md), [complete tuning ledger](scratchpad/rq5_scheduler_tuning_500m.md), and [machine-readable evidence](evidence/rq5_scheduler_results.json).

Analysis: Candidate selection uses best-epoch validation recall@100, then same-epoch NDCG@100; the table reports full-user metrics from the frozen winner. One-cycle cosine with 5% warmup and deep-only scheduling leads. Linear on both groups is the closest co-leader; step on both groups and both WSD scopes also remain inside the co-leader thresholds, while constant is outside them. Deep-only scheduling resolves as better only for the fixed- and tuned-warmup one-cycle cosine treatments; scope remains unresolved for every other scheduler. Inverse square root misses the approved non-inferiority rule. One non-selected WSD/deep-only candidate at LR 0.048 stopped at epoch 12 of horizon 15 before its decay engaged; it satisfies the approved tolerance, remains in the tuning ledger as auditable evidence, and does not determine the WSD winner.

Conclusion: Schedule only the deep group with one-cycle cosine and 5% warmup. It improves recall@100 by 3.01% and NDCG@100 by 3.13% over tuned constant. Coverage is numerically 9.83% lower but remains inside the report's 0.1 absolute threshold. Linear decay on both groups is the closest alternative; WSD remains a co-leader but is not the selected default, and inverse square root does not reach the approved non-inferiority threshold.

## RQ6 — Does learning-rate warmup help?

Obsolete: done in rq5

Implementation: [controlled variants](configs/variant.py), [warmup and timescale derivation](../../neuralrec/run/callbacks/lr_schedule.py#L73-L110), and [ramp implementation](../../neuralrec/run/callbacks/lr_schedule.py#L159-L190).

## RQ7 — Which position encoding works best: RoPE, ALiBi, learned positions, or combinations?

### Earlier broad position-encoding comparison

| encoding | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 |
| --- | --- | --- | --- | --- | --- |
| position none | 0.124 | 0.047 | 0.025 | 0.020 | 0.650 |
| **position alibi** | <span style="color: green">+10% (0.135)</span> | <span style="color: green">+10% (0.052)</span> | <span style="color: green">+13% (0.029)</span> | <span style="color: green">+12% (0.023)</span> | -5% (0.620) |
| position all | <span style="color: green">+6% (0.131)</span> | <span style="color: green">+7% (0.050)</span> | +8% (0.027) | <span style="color: green">+8% (0.022)</span> | -13% (0.567) |
| position learned alibi | <span style="color: green">+9% (0.135)</span> | <span style="color: green">+10% (0.052)</span> | +10% (0.028) | <span style="color: green">+11% (0.023)</span> | -4% (0.621) |
| position learned forward | <span style="color: green">+9% (0.135)</span> | <span style="color: green">+9% (0.052)</span> | +10% (0.028) | <span style="color: green">+11% (0.023)</span> | -11% (0.577) |
| position learned forward reverse | <span style="color: red">-3% (0.120)</span> | <span style="color: red">-5% (0.045)</span> | -10% (0.023) | <span style="color: red">-10% (0.018)</span> | +4% (0.678) |
| position learned reverse | <span style="color: green">+5% (0.130)</span> | <span style="color: green">+4% (0.049)</span> | 0% (0.025) | +1% (0.020) | +3% (0.670) |
| position learned reverse alibi | +1% (0.125) | +1% (0.048) | -3% (0.025) | -1% (0.020) | <span style="color: red">-27% (0.475)</span> |
| position reverse all | <span style="color: red">-4% (0.119)</span> | <span style="color: red">-5% (0.045)</span> | -9% (0.023) | <span style="color: red">-8% (0.019)</span> | <span style="color: red">-31% (0.446)</span> |
| position rope | +1% (0.125) | +1% (0.048) | 0% (0.025) | +1% (0.020) | -4% (0.625) |
| position rope alibi | <span style="color: green">+9% (0.135)</span> | <span style="color: green">+9% (0.052)</span> | +11% (0.028) | <span style="color: green">+10% (0.022)</span> | +1% (0.659) |
| position rope learned | <span style="color: green">+8% (0.134)</span> | <span style="color: green">+9% (0.052)</span> | +11% (0.028) | <span style="color: green">+11% (0.023)</span> | -3% (0.631) |
| position rope learned reverse | <span style="color: red">-4% (0.118)</span> | <span style="color: red">-5% (0.045)</span> | -9% (0.023) | <span style="color: red">-6% (0.019)</span> | <span style="color: red">-41% (0.385)</span> |
| position rope learned reverse alibi | <span style="color: red">-5% (0.117)</span> | <span style="color: red">-6% (0.044)</span> | -12% (0.022) | <span style="color: red">-8% (0.019)</span> | <span style="color: red">-49% (0.332)</span> |
| position rope reverse | +1% (0.125) | +1% (0.048) | 0% (0.025) | +1% (0.021) | -4% (0.625) |
| position rope reverse alibi | <span style="color: green">+8% (0.133)</span> | <span style="color: green">+8% (0.051)</span> | +10% (0.028) | <span style="color: green">+9% (0.022)</span> | -9% (0.591) |
| position rope reverse learned | <span style="color: green">+9% (0.134)</span> | <span style="color: green">+9% (0.052)</span> | +11% (0.028) | <span style="color: green">+11% (0.023)</span> | -3% (0.628) |
| position rope reverse learned alibi | <span style="color: green">+7% (0.132)</span> | <span style="color: green">+7% (0.050)</span> | +8% (0.028) | <span style="color: green">+8% (0.022)</span> | -13% (0.567) |
| position rope reverse learned reverse | <span style="color: green">+6% (0.131)</span> | <span style="color: green">+6% (0.050)</span> | +7% (0.027) | <span style="color: green">+6% (0.021)</span> | -4% (0.623) |

<!-- rq7-reinvestigation-generated:start -->
### Learned-position fusion comparisons

| learned-position treatment | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| forward additive | 0.134 | 0.051 | 0.028 | 0.022 | 0.647 |
| forward concatenated to item | 0% (0.134) | +1% (0.052) | +2% (0.029) | +3% (0.023) | +3% (0.668) |
| forward + reverse additive | 0% (0.133) | 0% (0.051) | 0% (0.028) | 0% (0.022) | -2% (0.634) |
| forward + reverse concatenated to item | +2% (0.137) | <span style="color: green">+3% (0.053)</span> | +5% (0.029) | <span style="color: green">+6% (0.024)</span> | +1% (0.657) |
| ALiBi + forward additive | 0.136 | 0.052 | 0.028 | 0.022 | 0.662 |
| **ALiBi + forward concatenated to item** | +1% (0.138) | <span style="color: green">+3% (0.053)</span> | +7% (0.030) | <span style="color: green">+8% (0.024)</span> | +3% (0.684) |
| ALiBi + forward + reverse additive | -1% (0.135) | -1% (0.051) | -1% (0.028) | -2% (0.022) | +1% (0.670) |
| ALiBi + forward + reverse concatenated to item | +1% (0.138) | <span style="color: green">+3% (0.053)</span> | +4% (0.029) | <span style="color: green">+6% (0.023)</span> | +4% (0.691) |

### RoPE / ALiBi comparison

| RoPE / ALiBi treatment | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| no position encoding | -2% (0.133) | <span style="color: red">-3% (0.050)</span> | -5% (0.027) | <span style="color: red">-6% (0.022)</span> | +8% (0.671) |
| **ALiBi** | 0.135 | 0.052 | 0.029 | 0.023 | 0.624 |
| plain forward RoPE (base 10000) | -1% (0.135) | -1% (0.051) | -2% (0.028) | -3% (0.022) | +5% (0.653) |
| forward RoPE + ALiBi | 0% (0.135) | -1% (0.052) | -2% (0.028) | -3% (0.022) | +5% (0.657) |
<!-- rq7-reinvestigation-generated:end -->

The broad table preserves the original independently tuned comparison. The focused tables below it are the later implementation-correctness reinvestigation and do not replace those additional treatment results.

Treatment semantics:

- Additive forward learned position is added to the item representation. Combined r7 additive nests that exact position-input function and adds `max_scale * tanh(gate) * reverse_embedding`, with a zero-initialized gate and `max_scale = 0.025`.
- Forward concat revision 3 is `item + gate_zero * variance_preserved(DenseNet([item; forward]))`. Combined r7 concat nests that exact position-input function and adds `0.025 * tanh(reverse_gate) * variance_preserved(DenseNet([item; forward; reverse]))`; the second branch receives the original item representation and both full-width position embeddings. Correction construction and its normal broad-initializer run are RNG-isolated, so every subsequent shared parameter retains the forward-only control seed.
- Reverse learned absolute positions anchor the last valid token at index zero. A constant sequence offset cancels from RoPE's relative phases, so reverse RoPE is instead a sign-reversed pairwise phase parameterization, not an absolute end anchor.
- ALiBi applies the same head-specific monotone attention bias in every named ALiBi treatment.

Implementation and evidence: [current treatment identities](analysis/rq7_reinvestigation_candidates.py), [generated report code](analysis/rq7_reinvestigation_report.py), [approved protocol](protocol/rq7_reinvestigation_plan.md), [eligible native-500M tuning ledger](scratchpad/rq7_reinvestigation_tuning_500m.md), and [machine-readable evidence](evidence/rq7_reinvestigation_results.json).

Analysis: Selection uses best-epoch validation recall@100, then same-epoch
NDCG@100, after each native-500M LR boundary is resolved; the table reports
restored-checkpoint full-user metrics. The RoPE comparison uses the selected
seed-42 run plus seed-43/44 confirmations. Plain base-10,000 RoPE trails ALiBi
by 0.000847 recall@100 and 0.000703 NDCG@100 on the three-seed means, both
inside the approved comparability bands. The four r7 forward+reverse rows are
also resolved. Relative to their approved forward controls, their
recall@100/NDCG@100 deltas are -0.000562/-0.000204 for additive,
+0.002710/+0.001666 for concat, -0.000831/-0.000573 for additive with ALiBi,
and +0.001768/+0.001419 for concat with ALiBi. No pair falls below its control
by more than the 0.003/0.001 non-inferiority margins. Combined r1–r6 and concat
r1/r2 artifacts remain excluded from active report selection.

Conclusion: Plain forward RoPE is comparable to ALiBi under repeated
native-500M evidence; adding ALiBi to RoPE is also inside the same ranking
bands. All requested forward+reverse variants satisfy the acceptance
criterion. Concatenated forward+reverse positions are the useful form: they
improve the approved additive-forward controls with and without ALiBi, whereas
the additive reverse correction is slightly lower but non-inferior. The
numerically best learned-position row remains the simpler ALiBi + forward
concat r3 at
0.137816 recall@100 and 0.053320 NDCG@100; ALiBi + forward/reverse concat r7 is
effectively tied at 0.137769 and 0.052975.

## RQ8 — How do scaling and architecture choices affect metrics?

The architecture tables retain the existing native-500M confirmations, with
the first row as the axis control. The FFN axis is RQ4 and is not repeated
here.

| dimension | recall@100 | ndcg@100 |
| --- | --- | --- |
| **64** | 0.135 | 0.052 |
| 128 | 0% (0.134) | 0% (0.051) |
| 16 | <span style="color: red">-11% (0.120)</span> | <span style="color: red">-13% (0.045)</span> |
| 256 | <span style="color: red">-5% (0.128)</span> | <span style="color: red">-6% (0.048)</span> |
| 32 | <span style="color: red">-3% (0.131)</span> | <span style="color: red">-3% (0.050)</span> |

| depth | recall@100 | ndcg@100 |
| --- | --- | --- |
| 2 layers | 0.135 | 0.052 |
| 1 layers | -1% (0.134) | -2% (0.051) |
| **4 layers** | +2% (0.137) | <span style="color: green">+2% (0.053)</span> |

| mha head count | recall@100 | ndcg@100 |
| --- | --- | --- |
| **2Q/2KV** | 0.134 | 0.052 |
| 1Q/1KV | <span style="color: red">-3% (0.130)</span> | <span style="color: red">-4% (0.049)</span> |
| 4Q/4KV | <span style="color: red">-5% (0.127)</span> | <span style="color: red">-6% (0.048)</span> |
| 8Q/8KV | <span style="color: red">-16% (0.113)</span> | <span style="color: red">-18% (0.042)</span> |

| attention grouping | recall@100 | ndcg@100 |
| --- | --- | --- |
| MHA 2Q/2KV | 0.134 | 0.052 |
| **GQA 2Q/1KV** | 0% (0.135) | 0% (0.052) |

| block normalization kind | recall@100 | ndcg@100 |
| --- | --- | --- |
| **LayerNorm** | 0.135 | 0.052 |
| BatchNorm | <span style="color: red">-3% (0.131)</span> | <span style="color: red">-4% (0.050)</span> |
| RMSNorm | -1% (0.133) | -1% (0.051) |

| residual normalization placement | recall@100 | ndcg@100 |
| --- | --- | --- |
| pre-LayerNorm | 0.135 | 0.052 |
| **post-LayerNorm** | <span style="color: green">+2% (0.138)</span> | <span style="color: green">+2% (0.053)</span> |

| input and final normalization | recall@100 | ndcg@100 |
| --- | --- | --- |
| no input + final LayerNorm | 0.135 | 0.052 |
| **input + final RMSNorm** | <span style="color: green">+3% (0.138)</span> | <span style="color: green">+3% (0.053)</span> |
| input LayerNorm + final LayerNorm | <span style="color: red">-6% (0.127)</span> | <span style="color: red">-7% (0.048)</span> |
| input RMSNorm + final LayerNorm | <span style="color: red">-8% (0.124)</span> | <span style="color: red">-10% (0.047)</span> |
| no input or final norm | -1% (0.134) | <span style="color: red">-2% (0.051)</span> |

| attention window | recall@100 | ndcg@100 |
| --- | --- | --- |
| 50 | 0.135 | 0.052 |
| 10 | <span style="color: red">-17% (0.112)</span> | <span style="color: red">-19% (0.042)</span> |
| **100** | +1% (0.136) | +1% (0.052) |
| 25 | <span style="color: red">-2% (0.132)</span> | <span style="color: red">-3% (0.050)</span> |
| 75 | -1% (0.133) | -1% (0.051) |
| full | 0% (0.134) | 0% (0.052) |

| dropout | recall@100 | ndcg@100 |
| --- | --- | --- |
| **0.1** | 0.135 | 0.052 |
| 0.0 | <span style="color: red">-3% (0.131)</span> | <span style="color: red">-3% (0.050)</span> |
| 0.2 | -1% (0.133) | -1% (0.051) |
| 0.3 | -2% (0.132) | <span style="color: red">-3% (0.050)</span> |
| 0.05 | -1% (0.133) | -2% (0.051) |
| 0.5 | <span style="color: red">-6% (0.127)</span> | <span style="color: red">-7% (0.048)</span> |

| bos | recall@100 | ndcg@100 |
| --- | --- | --- |
| disabled | 0.135 | 0.052 |
| **enabled** | +1% (0.135) | +2% (0.052) |

The query-token comparison uses independently tuned native-500M treatments and
reports three-seed mean full-user metrics. Standard item-state querying is the
control.

| query objective | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| standard item-state | 0.135 | 0.051 | 0.028 | 0.022 | 0.728 |
| **end-only CLS** | <span style="color: green">+11% (0.149)</span> | <span style="color: green">+19% (0.061)</span> | <span style="color: green">+34% (0.038)</span> | <span style="color: green">+38% (0.031)</span> | <span style="color: red">-39% (0.441)</span> |
| interleaved CLS | -1% (0.133) | <span style="color: red">-2% (0.050)</span> | -6% (0.027) | <span style="color: red">-5% (0.021)</span> | <span style="color: red">-27% (0.532)</span> |

| causal ALiBi retained history length | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 12 | <span style="color: red">-7% (0.124)</span> | <span style="color: red">-8% (0.047)</span> | <span style="color: red">-11% (0.025)</span> | <span style="color: red">-11% (0.020)</span> | <span style="color: red">-37% (0.444)</span> |
| 25 | -2% (0.132) | <span style="color: red">-2% (0.050)</span> | -3% (0.027) | -4% (0.021) | -10% (0.636) |
| 50 | 0% (0.135) | 0% (0.051) | +1% (0.028) | 0% (0.022) | -11% (0.633) |
| 100 | +1% (0.135) | 0% (0.051) | -2% (0.028) | -2% (0.022) | +5% (0.747) |
| 128 | 0.134 | 0.051 | 0.028 | 0.022 | 0.709 |
| 200 | 0% (0.134) | 0% (0.051) | -1% (0.028) | -1% (0.022) | -10% (0.639) |
| 256 | +1% (0.135) | +1% (0.052) | +1% (0.028) | +1% (0.023) | 0% (0.712) |
| **512** | **+1% (0.136)** | **+1% (0.052)** | **+1% (0.028)** | **0% (0.023)** | **-4% (0.679)** |

| reverse-RoPE + ALiBi retained history length | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 12 | <span style="color: red">-7% (0.124)</span> | <span style="color: red">-8% (0.047)</span> | -7% (0.026) | <span style="color: red">-7% (0.020)</span> | <span style="color: red">-27% (0.467)</span> |
| 25 | -2% (0.132) | -2% (0.050) | 0% (0.027) | 0% (0.022) | -2% (0.627) |
| 50 | 0% (0.135) | 0% (0.051) | 0% (0.027) | -1% (0.022) | -3% (0.621) |
| 100 | 0% (0.134) | 0% (0.051) | -1% (0.027) | 0% (0.022) | 0% (0.645) |
| 128 | 0.134 | 0.051 | 0.027 | 0.022 | 0.642 |
| 200 | 0% (0.134) | -1% (0.051) | -1% (0.027) | -2% (0.021) | +13% (0.724) |
| 256 | 0% (0.134) | -1% (0.050) | -1% (0.027) | -3% (0.021) | +11% (0.715) |
| **512** | **0% (0.135)** | **+1% (0.051)** | **+2% (0.028)** | **+2% (0.022)** | **+13% (0.725)** |

Treatment descriptions:

- Model dimension scales the transformer width while the item table stays at 64 dimensions; depth varies the number of blocks. The FFN width scales with model dimension at the control's ratio, which puts four dimension arms on widths that are not multiples of 16 — measured and left alone in [notes/ffn_width_alignment.md](notes/ffn_width_alignment.md).
- Head-count treatments split the same width into 1, 2, 4, or 8 query and key/value heads; grouped-query attention halves the key/value heads only.
- Block normalization swaps LayerNorm for RMSNorm or BatchNorm inside each block; placement moves it from before to after the residual branch.
- Input and final normalization vary the norms outside the blocks: at the tokenizer output and on the final sequence state.
- Sequence length is the history the tokenizer keeps. The corrected study uses full causal attention at every length, so the usable receptive field grows from 12 to 512, and compares causal ALiBi with reverse-index RoPE plus ALiBi. Each table uses its own length-128 arm as the percentage reference.
- BOS prepends a learned start token.
- Standard item-state querying trains item states autoregressively and uses the final item state to retrieve the next item.
- End-only CLS keeps autoregressive item-state training but replaces the final next-item query with one learned token appended after the observed history.
- Interleaved CLS trains `[item1, CLS, item2, CLS, ...]` autoregressively, with each learned query state predicting the following item.

Implementation: [architecture axes](configs/variant.py), [query and corrected sequence treatments](configs/rq8_reinvestigation_variant.py), [transformer construction](../../dcn/config/networks.py#L181-L242), [attention and windowing](../../dcn/nn/transformer.py#L228-L411), and [native-500M RQ8 evidence](evidence/rq8_reinvestigation_results.json).

Analysis: The native-500M query result resolves the CLS claim under same-dataset
tuning and repeated seeds. On the corrected full-causal surface, length 12 is
materially worse than 128 under both positional methods. From length 50 through
512, every recall difference from the same-method length-128 reference remains
inside the 0.003 threshold. The small rises and dips are non-monotonic, but none
is a resolved longer-history regression; causal ALiBi at 512 is numerically
highest at 0.135543 recall and 0.051821 NDCG. Reverse RoPE does not become better
at long histories: its length-512 recall is 0.000624 lower, an unresolved gap.
The only threshold-clearing reversal beyond length 50 is ALiBi coverage from
100 to 200 (0.747 to 0.639). Both lengths expose the same 7,674,702 targets per
epoch, but fixed batches of 1,280 sequences reduce optimizer updates from 99 to
73 per epoch at length 200. Ranking quality does not share the regression, so
this is a coverage/update-dynamics finding rather than evidence that longer
usable history harms retrieval accuracy.

Conclusion: End-only CLS is a resolved ranking win: recall@100 improves by 11%
and NDCG@100 by 19%, but coverage@100 materially regresses by 39%. Interleaved
CLS does not improve recall and its NDCG regression exceeds the reporting
threshold. Select causal ALiBi with length 512 as the nominal history setting:
it has the best observed recall and NDCG, although lengths 50–512 and the
length-512 reverse-RoPE variant remain operationally unresolved by the fixed
operational thresholds.

## RQ9 — Does a timestamp-delta representation improve metrics?

| time representation | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 |
| --- | --- | --- | --- | --- | --- |
| no time feature | 0.121 | 0.045 | 0.023 | 0.019 | 0.733 |
| 8 log-spaced bins, add | <span style="color: green">+13% (0.137)</span> | <span style="color: green">+16% (0.052)</span> | <span style="color: green">+23% (0.029)</span> | <span style="color: green">+25% (0.023)</span> | -9% (0.663) |
| 16 log-spaced bins, add | <span style="color: green">+11% (0.135)</span> | <span style="color: green">+14% (0.052)</span> | <span style="color: green">+20% (0.028)</span> | <span style="color: green">+22% (0.023)</span> | <span style="color: red">-21% (0.577)</span> |
| 32 log-spaced bins, add | <span style="color: green">+12% (0.136)</span> | <span style="color: green">+14% (0.052)</span> | <span style="color: green">+22% (0.028)</span> | <span style="color: green">+22% (0.023)</span> | <span style="color: red">-32% (0.497)</span> |
| 64 log-spaced bins, add | <span style="color: green">+11% (0.134)</span> | <span style="color: green">+12% (0.051)</span> | <span style="color: green">+17% (0.027)</span> | <span style="color: green">+17% (0.022)</span> | <span style="color: red">-15% (0.622)</span> |
| **32 bins + raw reverse RoPE** | <span style="color: green">+14% (0.137)</span> | <span style="color: green">+16% (0.052)</span> | <span style="color: green">+23% (0.029)</span> | <span style="color: green">+22% (0.023)</span> | -10% (0.658) |
| 32 bins + log forward RoPE | <span style="color: green">+13% (0.137)</span> | <span style="color: green">+16% (0.053)</span> | <span style="color: green">+25% (0.029)</span> | <span style="color: green">+26% (0.023)</span> | -12% (0.647) |
| 32 bins, concatenate-and-project | +1% (0.122) | +1% (0.046) | +4% (0.024) | +5% (0.019) | <span style="color: red">-44% (0.408)</span> |
| clipped linear delta, add | <span style="color: green">+9% (0.131)</span> | <span style="color: green">+9% (0.049)</span> | <span style="color: green">+13% (0.026)</span> | <span style="color: green">+12% (0.021)</span> | -2% (0.717) |
| log delta, add | <span style="color: green">+7% (0.129)</span> | <span style="color: green">+9% (0.049)</span> | <span style="color: green">+14% (0.027)</span> | <span style="color: green">+14% (0.021)</span> | +3% (0.754) |
| log delta, concatenate-and-project | <span style="color: green">+12% (0.135)</span> | <span style="color: green">+16% (0.052)</span> | <span style="color: green">+23% (0.029)</span> | <span style="color: green">+26% (0.023)</span> | <span style="color: red">-25% (0.547)</span> |
| raw elapsed-time RoPE, forward | <span style="color: green">+5% (0.127)</span> | <span style="color: green">+6% (0.048)</span> | +10% (0.026) | <span style="color: green">+10% (0.020)</span> | +7% (0.782) |
| raw elapsed-time RoPE, reverse | <span style="color: green">+5% (0.126)</span> | <span style="color: green">+6% (0.048)</span> | +10% (0.026) | <span style="color: green">+10% (0.020)</span> | +5% (0.767) |
| log elapsed-time RoPE, forward | <span style="color: red">-3% (0.117)</span> | <span style="color: red">-3% (0.044)</span> | -4% (0.022) | -3% (0.018) | <span style="color: red">-42% (0.424)</span> |
| log elapsed-time RoPE, reverse | <span style="color: red">-11% (0.107)</span> | <span style="color: red">-10% (0.041)</span> | -9% (0.021) | <span style="color: red">-7% (0.017)</span> | -8% (0.677) |

Treatment descriptions:

- No time feature is the item-only control.
- Clipped linear delta and log delta add one continuous pairwise-gap embedding to each item token.
- The 8-, 16-, 32-, and 64-bin additive variants discretize normalized log gaps; bin count is the mechanical axis.
- 32 bins plus raw reverse RoPE combines pairwise gap bins with elapsed time anchored at the last valid event.
- 32 bins plus log forward RoPE combines gap bins with log elapsed time anchored at the first valid event.
- Log delta and 32-bin concatenate-and-project variants fuse time through a learned projection instead of addition.
- Raw elapsed-time RoPE forward anchors continuous rotary positions at the first valid event.
- Raw elapsed-time RoPE reverse anchors continuous rotary positions at the last valid event.
- Log elapsed-time RoPE forward applies `log1p` before first-event rotary positioning.
- Log elapsed-time RoPE reverse applies `log1p` before last-event rotary positioning.

Implementation: [time variants](configs/variant.py#L475-L537), the [history tokenizer](../../dcn/models/history_tokens.py#L107-L175), timestamp [network configuration](../../dcn/config/networks.py#L186-L193), and rotary [position inputs](../../dcn/nn/transformer.py#L160-L192). Timestamp rotation follows [RoPE](https://arxiv.org/abs/2104.09864). Binned variants use timestamp semantics revision 2 with dedicated zero-gap and clipped-boundary buckets; concatenation projects `2d → 2d → d`.

Analysis: Treatment ranking does not transfer reliably across dataset size. Log elapsed-time RoPE forward leads the 50M proxy at recall `0.084` but regresses to `0.117` on 500M, while 32 bins plus raw reverse RoPE is weak on 50M (`0.072`) yet leads 500M at `0.137`.

Conclusion: On native 500M, several additive and binned time features materially improve ranking quality. The 32-bin plus raw-reverse combination is the numerical recall leader, but every binned additive variant, both bin-plus-RoPE combinations, and log-delta concatenation sit inside the recall threshold of it, and 32-bin plus log-forward RoPE has the highest NDCG — the leading group is mutually unresolved. Pure log RoPE hurts. Time representations should therefore be selected in the target dataset regime rather than promoted from the 50M ranking.

## RQ10 — Do separate item embeddings at every transformer layer help?

The earlier matched native-500M sanity comparison is retained as two-layer context; it cannot select the four-layer treatment. The reinvestigation independently tunes the four-layer input/output-only control, direct addition, a zero-start concatenated DenseNet residual, and zero-start Gemma-style PLE.

### Earlier valid two-layer comparison

| item-feature path | recall@100 | ndcg@100 |
| --- | ---: | ---: |
| input/output item embedding only | 0.140 | 0.053 |
| direct full-width addition before every layer | 0.139 | 0.053 |

### Four-layer reinvestigation

| item-feature path | feature width | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 | parameters (M) | median epoch (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Input/output item embedding only** | **—** | **0.138** | **0.053** | **0.029** | **0.023** | **0.648** | **10.3** | **12.70** |
| Direct full-width addition | 64 | <span style="color: red">-11% (0.123)</span> | <span style="color: red">-12% (0.047)</span> | <span style="color: red">-14% (0.025)</span> | <span style="color: red">-14% (0.020)</span> | -3% (0.632) | 50.6 | 12.94 |
| Zero-start concatenated DenseNet residual | 64 | <span style="color: red">-3% (0.133)</span> | <span style="color: red">-3% (0.051)</span> | -2% (0.028) | -3% (0.023) | +4% (0.676) | 50.6 | 13.64 |
| Zero-start Gemma-style PLE | 2 | -1% (0.136) | -1% (0.053) | +0% (0.029) | +1% (0.023) | -8% (0.596) | 11.5 | 13.47 |

Method details:

- Input/output-only uses the tied item table only at token input and sampled-softmax output.
- Direct addition injects an independent full-width item lookup before every transformer block.
- Concatenated residual fuses normalized hidden and projected item features through a zero-start DenseNet residual before each block.
- Gemma-style PLE follows [Gemma 3n per-layer embeddings](https://ai.google.dev/gemma/docs/gemma-3n): a compact lookup combines with the original token, is gated by the post-block hidden state, projected to model width, normalized, and added through a zero-start residual.

Selection uses validation recall@100 and then same-epoch NDCG@100. Zero-start Gemma-style PLE selects width 2 and deep LR 0.012. Against the tuned control, its exact final recall/NDCG losses are 0.002021729 and 0.000431982; both are inside the approved 0.003/0.001 non-inferiority bands. Not selected by the decision rule: Direct full-width addition, Zero-start concatenated DenseNet residual. Their internal degradation mechanisms remain unresolved, so no architectural-harm claim is made.

Conclusion: Select Zero-start Gemma-style PLE with width 2 and deep LR 0.012 for the added-feature treatment. It satisfies the not-worse-than-baseline acceptance gate. The run does not establish a metric improvement over the control. The other tested fusions are not selected on this surface, without attributing their losses to the architectures themselves.

## RQ11 — How do online logQ, offline logQ, random, mixed, and uncorrected negatives compare?

### Earlier broad negative-sampling comparison

| negative sampling | negatives | logQ alpha | random fraction | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed in-batch leave-one-out logQ | 2048 | — | — | 0.116 | 0.044 | 0.023 | 0.019 | 0.461 |
| fixed in-batch global-q Yi-2019 | 2048 | — | — | <span style="color: green">+5% (0.122)</span> | <span style="color: green">+5% (0.046)</span> | +8% (0.024) | <span style="color: green">+7% (0.020)</span> | <span style="color: green">+34% (0.617)</span> |
| popularity random global-q Yi-2019 | 512 | — | — | <span style="color: green">+15% (0.133)</span> | <span style="color: green">+16% (0.051)</span> | <span style="color: green">+23% (0.028)</span> | <span style="color: green">+22% (0.023)</span> | <span style="color: green">+29% (0.593)</span> |
| streaming in-batch global-q Yi-2019 | 512 | 0.005 | — | <span style="color: green">+14% (0.133)</span> | <span style="color: green">+15% (0.051)</span> | <span style="color: green">+22% (0.028)</span> | <span style="color: green">+20% (0.022)</span> | +21% (0.556) |
| uncorrected in-batch | 512 | — | — | <span style="color: red">-45% (0.064)</span> | <span style="color: red">-48% (0.023)</span> | <span style="color: red">-53% (0.011)</span> | <span style="color: red">-55% (0.008)</span> | <span style="color: green">+101% (0.925)</span> |
| **uniform random** | 512 | — | — | <span style="color: green">+16% (0.135)</span> | <span style="color: green">+17% (0.052)</span> | <span style="color: green">+23% (0.028)</span> | <span style="color: green">+21% (0.023)</span> | <span style="color: green">+25% (0.577)</span> |
| uniform random + fixed logQ on in-batch negatives | 512 | — | 0.25 | <span style="color: green">+11% (0.129)</span> | <span style="color: green">+12% (0.049)</span> | <span style="color: green">+20% (0.027)</span> | <span style="color: green">+16% (0.021)</span> | <span style="color: green">+36% (0.626)</span> |
| uniform random + streaming logQ on in-batch negatives | 512 | 0.01 | 0.875 | <span style="color: green">+6% (0.123)</span> | <span style="color: green">+6% (0.047)</span> | +11% (0.025) | <span style="color: green">+9% (0.020)</span> | +1% (0.466) |

| homework-matched objective | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 |
| --- | --- | --- | --- | --- | --- |
| fixed leave-one-out logQ | 0.127 | 0.048 | 0.024 | 0.019 | 0.511 |
| **uniform random** | <span style="color: green">+3% (0.132)</span> | <span style="color: green">+4% (0.050)</span> | +7% (0.026) | <span style="color: green">+6% (0.020)</span> | <span style="color: green">+25% (0.641)</span> |

### Corrected uniform/streaming mixture comparison

<!-- rq11-mixed-streaming-generated:start -->
| negative sampling | negatives | logQ alpha | uniform fraction | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| uniform catalog | 2048 | — | — | 0.136 | 0.052 | 0.028 | 0.023 | 0.560 |
| streaming in-batch global-q | 1024 | 0.005 | — | -1% (0.134) | +0% (0.052) | +3% (0.029) | +3% (0.023) | +12% (0.628) |
| **popularity catalog global-q** | 2048 | — | — | +1% (0.137) | <span style="color: green">+2% (0.053)</span> | +6% (0.030) | <span style="color: green">+5% (0.024)</span> | <span style="color: green">+18% (0.662)</span> |
| aggregate uniform + streaming global-q | 256 | 0.0025 | 0.75 | +0% (0.136) | +0% (0.052) | +1% (0.028) | +1% (0.023) | <span style="color: green">+21% (0.674)</span> |
<!-- rq11-mixed-streaming-generated:end -->

The earlier broad tables remain useful native-500M comparisons and are retained above. The later matched native-500M search selected every family independently by validation recall@100, with validation NDCG@100 and lower cost as tie-breakers. All selected and boundary runs completed their declared 20-epoch linear horizon and restored the best validation checkpoint. Percentages should be interpreted within each table because the controls and tuning protocols differ.

Treatment descriptions:

- Fixed leave-one-out logQ uses cached item frequencies and excludes the current positive from its proposal denominator.
- Fixed global-q uses cached global proposal probabilities and corrects both positive and negative logits.
- Uniform catalog draws every negative uniformly and applies no logQ correction.
- Streaming global-q corrects the positive and sampled in-batch negatives with a valid-catalog-normalized online proposal.
- Popularity global-q draws catalog negatives from the cached training distribution and corrects the positive and negatives.
- Uncorrected in-batch treats other batch positives as negatives without proposal correction.
- The earlier fixed/streaming mixtures combine uniform catalog negatives with corrected in-batch negatives.
- Aggregate uniform + streaming global-q allocates the integer negative budget between both sources and corrects every logit with the realized aggregate proposal.

Implementation and protocol: [approved plan](protocol/rq11_mixed_streaming_plan.md), [candidate manifest](protocol/rq11_mixed_streaming_manifest.json), [sampled-softmax implementation](../../dcn/nn/sampled_softmax.py), and [historical diagnostic evidence](evidence/rq11_negative_sampling.md).

Conclusion: The corrected mixture does not beat the other tuned families. Its 0.136 recall and 0.052 NDCG are unresolved against uniform and streaming global-q; popularity catalog global-q has numerically highest recall and materially higher NDCG at 0.137 / 0.053, so the approved decision rule selects it over the mixture. The mixture does provide the highest coverage, 0.674, but that secondary gain does not offset the ranking result. At the mixture's selected secondary configuration, removing positive-logit correction reduces the best diagnostic to 0.128 recall / 0.049 NDCG, supporting full correction for that configuration.
