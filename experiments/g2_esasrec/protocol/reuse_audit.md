# Existing eSASRec branch reuse audit

## Verified state

- Audited branch: `exp/esasrec` at `8e92db8` against the current
  `exp/g1-sasrec-likes-metrics` worktree.
- Focused branch tests:

  ```text
  pytest -q dcn/tests/nn/test_transformer.py \
    dcn/tests/config/test_networks.py \
    dcn/tests/config/test_generation_initialization.py \
    dcn/tests/experiments/test_g2_report_generation.py
  65 passed, 2 warnings in 3.19s
  ```

- `official/check_protocol.py` still reproduces the expected native-50M split:
  cutoff 25394930, catalog 33148, 614244 training events, 20398 validation
  events, and 3414 evaluable users.
- The three official eSASRec 50M artifacts are validation-resolved, but their
  metadata does not record the RecTools version, source hashes, negative-sampler
  implementation, or environment lock. They cannot satisfy the new provenance
  verifier and must be rerun under the approved plan.

## Reuse

- Reuse the `GatedResidual` arithmetic and its two-gates-per-block wiring as a
  reference while adapting it to the current transformer API.
- Reuse the official protocol restatement and runner structure after removing
  500M behavior and adding exact version/source/configuration provenance.
- Reuse the existing raw 50M and 500M artifacts only as immutable diagnostics.
- Reuse the current mainline mixed uniform/in-batch and fully corrected global-q
  sampled-softmax implementation; it is newer than the eSASRec branch.
- Use local implementations for all G2 treatment, tuning, and confirmation runs
  after deterministic RecTools forward/loss/gradient parity passes. RecTools
  remains the fixed oracle and official-recipe reference, not a full-run
  fallback.

## Replace

- Do not cherry-pick `b8902ca`: the current transformer, FFN, initialization,
  negative-sampling, metadata, and test APIs have diverged substantially.
- Do not port the branch's SwiGLU change. Current mainline already supports
  explicitly enabled gated-FFN dropout and tests it across ReGLU, GEGLU, and
  SwiGLU. New G2 SwiGLU widths are `{512, 1024, 1536}`, all divisible by 32;
  width 171 remains only in the unchanged G1 control.
- Replace the full-softmax ablation with the approved gBCE/sampled-softmax
  matrix.
- Replace every 500M launcher, tuning table, rate-tail task, reader table, and
  conclusion. G2 research evidence is native 50M only.
- Replace the branch report collector with fail-closed selectors for the exact
  approved manifest, native-50M band evidence, complete tuning ledger, compact
  tables, cost metrics, and exact atomic-bundle aggregate reuse.
- Rework the norm-initialization exception against the current initializer
  rather than copying the old boolean patch. G2 needs project matrices at
  standard deviation 0.02 while preserving norm gains at one, without changing
  G1 defaults or initializer RNG behavior.

## Missing implementation

- Public gBCE loss with catalog-size calibration and fixed sampled-negative
  inputs.
- RecTools-parity fixtures for LiGR, official SASRec blocks, sampled softmax,
  gBCE, and gradients.
- An official-style SASRec block. The generic current pre-norm block cannot
  reproduce RecTools SASRec's normalized-query/raw-key-value residual path.
- G2 configuration builders for the six component methods, capacity-matched
  standard diagnostics, primary G1 control, and mixed sampler.
- Optuna manifests, boundary extensions, queue launcher, artifact verifier,
  ten-seed band generator, performance/latency benchmark, and report pipeline.

## Post-approval TDD order

1. Add deterministic RecTools reference fixtures and failing public-interface
   tests for LiGR and official SASRec blocks.
2. Implement the two block families in the current transformer stack and pass
   forward/gradient parity.
3. Add failing sampled-loss parity tests, then implement gBCE and the shared
   fixed-negative interface.
4. Add configuration-invariant tests, then implement G2 variants and
   initialization behavior.
5. Add manifest/selector/report tests, then implement launchers, verification,
   uncertainty, cost, latency, and reporting.
6. Run focused tests, self-review, blind review, `./test.sh`, four approved
   smoke runs, then submit the approved native-50M batch.
