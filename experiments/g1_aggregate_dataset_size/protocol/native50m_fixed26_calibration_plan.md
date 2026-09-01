# Native-50M fixed-26 shared calibration

## Question and hypothesis

- Re-estimate native-50M run-to-run variability without seed-dependent early stopping.
- The large existing dispersion is primarily caused by different seeds stopping at different training horizons.
- Training every repeat for 26 epochs should produce a more stable shared calibration.

## Comparison

- Reuse the selected batch-512 MuTransfer item-ID control with embedding learning rate
  `0.003261002414691765` and deep learning rate `0.025343654763668278`.
- Repeat seeds 42–51 as ten distinct new runs.
- Train every run for exactly 26 epochs, validate every epoch, disable early stopping,
  and restore the best validation checkpoint within the 26-epoch horizon.
- Preserve the previous seed-42–51 artifacts and calibration as immutable audit evidence.

## Data and evaluation

- Native Yambda-50M with the existing full-user, likes-only, final-seven-day,
  mapped-train-catalog, full-catalog, no-seen-item-exclusion protocol.
- Keep the selected control's architecture, optimizer, batch 512, negative sampling,
  sequence length, data seed semantics, and evaluation unchanged.
- Report every existing Recall, NDCG, MRR, capped-Recall, and Coverage metric.

## Run and analysis plan

- Submit one sealed ten-job batch through the existing persistent training queue.
- Require authenticated configuration and metric artifacts for every seed.
- Record per-seed best epoch and final trained epoch.
- Compute the mean, sample standard deviation with `ddof=1`, and
  `sample standard deviation / mean` from unrounded restored-checkpoint metrics.
- Present the candidate calibration for user validation before replacing the shared
  native-50M table or changing any experiment conclusion.

## Acceptance criteria

- Remove early stopping and train every calibration repeat for 26 epochs.
- All ten runs differ only by seed and complete all 26 epochs.
- The old calibration and raw runs remain preserved.

## Approval

- The user approved seeds 42–51, no early stopping, and exactly 26 epochs on
  2026-08-30.
