# Transfer study — where the learning-rate optimum sits

All 137 runs below were launched under the annealing-horizon rule, so every
linear-schedule point trains its full 20-epoch horizon and reports the best
epoch inside it. The `power` points are horizon-free and keep early stopping.
Regenerate with `python -m experiments.g1_sasrec_item_ids_likes.analysis.transfer_study`.

Each table sweeps one rate and holds the other fixed, so the readout is where
the curve peaks, not which configuration wins. The native-500M recall@100
resolution band is 0.00215019; native 50M has no repeat band, and its
curves show non-monotone bumps of roughly 0.005, which is the scale below
which a 50M difference should not be read.

## Why these runs exist

The pre-existing 50M surface was measured under a linear schedule that anneals
to zero over 20 epochs, but 1180 of its 1733 revision-2 runs stopped short of
that horizon on early-stopping patience. Only selected winners were continued.
Every unselected grid point therefore recorded the value it held when patience
fired mid-decay, and the comparisons built on those points — including RQ1's
transfer regret and RQ4's SwiGLU width — compared runs that had spent different
fractions of their schedule.

### Arm A — model width, deep-LR sweep (native 50M)

embedding LR held at 0.032; **bold** marks each curve's argmax.

| curve | 0.003 | 0.006 | 0.012 | 0.024 | 0.048 | 0.096 | argmax | best/trained |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| architecture_control | 0.05114 | 0.05731 | **0.06771** | 0.06404 | 0.05815 | 0.04284 | 0.012 | 19/20 |
| dimension_16 | 0.05210 | 0.06687 | 0.07152 | **0.07620** | 0.06328 | 0.04662 | 0.024 | 19/20 |
| dimension_32 | 0.04875 | 0.05798 | **0.07608** | 0.07131 | 0.07063 | 0.04142 | 0.012 | 20/20 |
| dimension_128 | 0.05073 | 0.05776 | **0.06537** | 0.06498 | 0.05161 | 0.05752 | 0.012 | 20/20 |
| dimension_256 | 0.05103 | 0.06075 | 0.06519 | **0.06648** | 0.05460 | 0.06442 | 0.024 | 20/20 |

### Arm A — model width, embedding-LR sweep (native 50M)

deep LR held at 0.012; **bold** marks each curve's argmax.

| curve | 0.008 | 0.016 | 0.032 | 0.064 | 0.128 | 0.256 | argmax | best/trained |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| architecture_control | 0.05284 | 0.05335 | 0.06771 | 0.07289 | **0.07635** | 0.07045 | 0.128 | 20/20 |
| dimension_16 | 0.05636 | 0.05548 | 0.07152 | **0.07604** | 0.07500 | 0.06808 | 0.064 | 20/20 |
| dimension_32 | 0.05157 | 0.05554 | 0.07608 | 0.07650 | **0.08162** | 0.07266 | 0.128 | 19/20 |
| dimension_128 | 0.05475 | 0.05514 | 0.06537 | 0.07592 | **0.07642** | 0.06801 | 0.128 | 19/20 |
| dimension_256 | 0.05394 | 0.05862 | 0.06519 | 0.07804 | **0.07928** | 0.06978 | 0.128 | 20/20 |

### Arm B — FFN width, deep-LR sweep (native 50M)

embedding LR held at 0.032; **bold** marks each curve's argmax.

| curve | 0.003 | 0.006 | 0.012 | 0.024 | 0.048 | 0.096 | argmax | best/trained |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ffnbase_ffn32 | 0.05146 | 0.05851 | 0.06607 | **0.07176** | 0.06356 | 0.04972 | 0.024 | 16/20 |
| ffnbase_ffn64 | 0.05067 | 0.05979 | 0.06825 | **0.06867** | 0.06065 | 0.06315 | 0.024 | 17/20 |
| ffnbase_ffn128 | 0.05073 | 0.05675 | 0.06126 | **0.06663** | 0.05637 | 0.06407 | 0.024 | 19/20 |
| ffnbase_ffn224 | 0.05119 | 0.05774 | **0.06737** | 0.06177 | 0.05926 | 0.05692 | 0.012 | 20/20 |
| ffnratio_ffn32 | 0.05021 | 0.05728 | 0.06480 | **0.06566** | 0.06381 | 0.05504 | 0.024 | 18/20 |
| ffnratio_ffn64 | 0.04953 | 0.05811 | 0.06209 | **0.06961** | 0.06389 | 0.04521 | 0.024 | 20/20 |
| ffnratio_ffn128 | 0.05073 | 0.05675 | 0.06126 | **0.06663** | 0.05637 | 0.06407 | 0.024 | 19/20 |
| ffnratio_ffn224 | 0.04954 | 0.05738 | **0.06741** | 0.06484 | 0.06052 | 0.04474 | 0.012 | 18/20 |

### Arm C — dataset size, deep-LR sweep (native 50m)

embedding LR held at 0.032; **bold** marks each curve's argmax.

| curve | 0.003 | 0.006 | 0.012 | 0.024 | 0.048 | 0.096 | argmax | best/trained |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| linear | 0.05114 | 0.05731 | **0.06771** | 0.06404 | 0.05815 | 0.04284 | 0.012 | 19/20 |
| power | 0.07895 | 0.04349 | 0.08355 | **0.08747** | 0.03565 | 0.07488 | 0.024 | 35/38 |

### Arm C — dataset size, embedding-LR sweep (native 50m)

deep LR held at 0.012; **bold** marks each curve's argmax.

| curve | 0.008 | 0.016 | 0.032 | 0.064 | 0.128 | 0.256 | argmax | best/trained |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| linear | 0.05284 | 0.05335 | 0.06771 | 0.07289 | **0.07635** | 0.07045 | 0.128 | 20/20 |
| power | — | — | **0.08355** | — | — | — | 0.032 | 27/30 |

### Arm C — dataset size, deep-LR sweep (native 500m)

embedding LR held at 0.032; **bold** marks each curve's argmax.

| curve | 0.003 | 0.006 | 0.012 | 0.024 | 0.048 | 0.096 | argmax | best/trained |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| linear | 0.12285 | 0.12960 | 0.13500 | **0.13588** | 0.13344 | 0.13259 | 0.024 | 17/20 |
| power | 0.10779 | 0.12733 | 0.12749 | 0.13353 | **0.13423** | 0.13331 | 0.048 | 32/35 |

### Arm C — dataset size, embedding-LR sweep (native 500m)

deep LR held at 0.012; **bold** marks each curve's argmax.

| curve | 0.008 | 0.016 | 0.032 | 0.064 | 0.128 | 0.256 | argmax | best/trained |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| linear | 0.12761 | 0.13240 | **0.13500** | 0.13395 | 0.13136 | 0.13222 | 0.032 | 19/20 |
| power | — | — | **0.12749** | — | — | — | 0.032 | 24/27 |

## What each arm answers

### Arm A — μTransfer across model width: it works

The deep-LR optimum sits at 0.012 or 0.024 at every width from 16 to 256, and
the embedding-LR optimum at 0.064 or 0.128. Neither drifts with width; the
one-step jitter has no direction. On this surface μP transfers both rates
across a 16× width range.

This retracts RQ1's larger negative result. At width 256 the shared rate
0.032/0.012 was recorded at 0.042882 against a 0.008/0.012 "local optimum" of
0.053937, a 20.5% regret. Horizon-complete, the same shared rate reaches
0.06519 and now *beats* that local point by 0.011: the regret does not shrink,
it reverses. The old shared-rate run had peaked at epoch 5 and been killed at 8
of 20. The width-16 regret is unaffected — its transferred run was only two
epochs short — and stands at 0.07152 against 0.07620.

The surface also shows that 0.032/0.012 is not the 50M optimum at any width.
The 50M optimum is 0.128/0.012, worth 0.07635 against the control's 0.06771,
13% more recall@100 on the proxy.

### Arm B — μTransfer across FFN width: no correction was needed

`mup_base_ffn_dim` gives the FFN output projection a fan-in base of its own, so
MuAdam divides its rate by the FFN width rather than by the model dimension.
The prediction was that the optimum would stop moving with FFN width once it
was applied. It did not need to: the optimum is at deep LR 0.024 for widths 32,
64 and 128 and at 0.012 for 224 — the same in both parameterizations. Widths 32
and 128 differ by 4× in FFN width and share an optimum without the correction.

The base changes absolute quality at the narrow end, 0.07176 against 0.06566 at
width 32, but that difference is at the 50M noise scale and the argmax is
unmoved. Widths 128 are identical between the two modes by construction: the
ratio-derived base at FFN 128 is 32, the same absolute base the fixed mode uses.

The conclusion is that RQ4's SwiGLU problem was never a parameterization
problem. It was a learning-rate search that stopped below the rate width 128
wanted. The `mup_base_ffn_dim` option is kept because it is the correct μP
treatment of an independently varying FFN width, but it buys nothing at this
model size and is off by default.

### Arm C — across dataset size: the deep rate transfers, the embedding rate does not

| rate | 50M optimum | 500M optimum | cost of transferring the 50M choice |
| --- | --- | --- | --- |
| deep, embedding held at 0.032 | 0.012 | 0.024 | 0.00088, inside the band |
| embedding, deep held at 0.012 | 0.128 | 0.032 | 0.00364, through the band |

The deep rate transfers: 0.012 is the 50M optimum and is unresolved from the
500M optimum. The embedding rate does not: the 50M optimum is 4× higher than
the 500M optimum, and carrying it across costs a resolved 0.0036 recall@100.

This is the shape the split learning rate was for. The item table sees an order
of magnitude more items and updates at 500M, and its rate has to come down
accordingly, while the transformer's rate is invariant to the data it is
trained on. A proxy-tuned recipe should transfer its deep rate and retune only
the embedding rate at the target size — one rate on a single axis, which is the
cheapest possible confirmation.

The current control's 0.032/0.012 is the 500M optimum on this surface. It is
badly off on the 50M proxy, where 0.128/0.012 is 13% better, so the proxy has
been ranking treatments at an embedding rate 4× below its own optimum.

### The power schedule does not help

A token-indexed `power` schedule was the candidate instrument for making the
optimum budget-agnostic. It is not one here. At 500M its curve is flat across
0.024–0.096, all inside the band, and its best point of 0.13423 is below the
linear schedule's 0.13588. At 50M it is unreadable: being horizon-free it keeps
early stopping, and patience 3 fires at epoch 5 for deep LR 0.048 and epoch 9
for 0.006 while other points run to 40, producing a curve that swings between
0.036 and 0.087 with no relation to the rate. A horizon-free schedule needs a
stopping rule that is not patience on a noisy validation curve before it can be
swept at all.
