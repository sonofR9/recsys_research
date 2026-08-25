# RQ7 reinvestigation plan

## Question and decision

- Research question: how should forward and reverse learned positions be fused,
  does adding ALiBi change that answer, and can a correctly tuned forward RoPE
  match ALiBi on native Yambda-500M?
- Current evidence problem: the reader's native-500M rows are single
  confirmations whose treatment-specific learning rates were selected on 50M.
  The 50M ordering does not transfer, so those rows are not a valid native-500M
  selection surface.
- Falsifiable hypotheses: concatenating forward and reverse learned embeddings
  avoids the destructive interference possible when two full-width tables are
  added; adding the reverse embedding is non-inferior to forward-only both with
  and without ALiBi; and plain RoPE is non-inferior to ALiBi after matched
  native-500M tuning.
- Decision: select a learned-position fusion only if it is non-inferior to its
  stated control. Select plain RoPE as comparable to ALiBi only if their
  repeated native-500M ranking metrics are within the fixed operational bands.
  Otherwise retain ALiBi and publish the implementation and inductive-bias
  evidence explaining the gap.

## Controlled comparison

- Learned-position treatments:
  1. learned forward;
  2. learned forward, concatenated to the item representation;
  3. learned forward + reverse, additive;
  4. learned forward + reverse, concatenated to the item representation;
  5. ALiBi + learned forward;
  6. ALiBi + learned forward, concatenated to the item representation;
  7. ALiBi + learned forward + reverse, additive;
  8. ALiBi + learned forward + reverse, concatenated to the item
     representation.
- Forward-only concat is the revision-3 control: the item representation plus
  a zero-gated, input-RMS-preserving DenseNet over `[item; forward]`. Combined
  revision 7 nests that exact control and adds a second correction
  `0.025 * tanh(reverse_gate) * DenseNet([item; forward; reverse])`. Both learned
  positions are full width, the second DenseNet receives the original item
  representation, and its gate starts at zero. Additive revision 7 similarly
  nests exact forward-add and adds
  `0.025 * tanh(reverse_gate) * reverse_embedding`. The ALiBi arms additionally
  apply the same attention bias as their controls.
- RoPE diagnostic treatments: no positional encoding, ALiBi, forward RoPE at
  base 10,000, and forward RoPE at base 10,000 plus ALiBi.
- If tuned base-10,000 RoPE remains materially below ALiBi, extend the RoPE
  architecture axis to bases 100 and 1,000. Together with 10,000 these are the
  three predeclared short-history bases. Extend once beyond a boundary only if
  the best base is an outer point.
- Held fixed: standard item-state autoregression, sequence length 128,
  attention window 50, two-layer width-64 selected architecture, batch 1280,
  16-bin timestamp input, random negatives, no BOS/CLS, and every non-position
  field.

## Data and evaluation

- Evidence dataset: native Yambda-500M, already validated by the user for G1.
- Debug dataset: native Yambda-50M with distinct diagnostic identities, no user
  sampling, and no reuse for selection, promotion, or reader claims.
- Both sizes use their native examples; 50M is never repeated or matched to the
  500M token, target, step, or epoch count.
- Primary metric: recall@100. Secondary ranking metrics: NDCG@100, recall@10,
  and NDCG@10. Coverage@100 is reported as a trade-off.
- Non-inferiority bands: no loss larger than 0.003 recall or 0.001 NDCG. The
  corresponding coverage diagnostic band is 0.1.
- The learned-forward concatenation arm compares with additive learned
  forward. The additive and concatenated learned-forward+reverse arms compare
  with additive learned forward. Their ALiBi counterparts compare with ALiBi
  + additive learned forward. Plain RoPE compares with ALiBi.

## Tuning and selection

- Every run uses embedding LR 0.064, physical/effective batch 1280, and μP.
- Each native-500M treatment independently tunes deep LR at 0.006, 0.012, and
  0.024. A boundary winner extends geometrically by 2x in the required
  direction until the selection is interior or otherwise resolved.
- Selection uses best-epoch validation recall@100, then same-epoch NDCG@100.
  Reader metrics come from the restored selected checkpoint on the full user
  set.
- Every run completes the declared 20-epoch linear schedule horizon without
  adaptive early stopping, validates every epoch, and restores the best epoch
  within the horizon.
- The selected ALiBi, selected plain-RoPE, and RoPE+ALiBi arms receive seed-43
  and seed-44 confirmations if plain RoPE remains close enough to require a
  repeated comparability decision or remains worse and requires a proof-grade
  gap.
- A learned-fusion comparison that is materially worse returns to fusion-scale,
  normalization, indexing, and optimization debugging. Any corrected pair may
  receive two additional confirmation seeds before acceptance.

## Concat implementation revision

- Revision 1 concat artifacts are diagnostic-only: the global standard-0.02
  initialization makes the two-layer DenseNet output about 100 times smaller
  than the additive transformer input.
- Revision 2 keeps the declared direct concatenation and two-layer DenseNet,
  then non-affinely normalizes its output and rescales it to the concatenated
  input RMS. Its native-50M diagnostics remain collapsed against the additive
  controls, so revision 2 is diagnostic-only.
- Revision 3 keeps the item representation as a residual path and adds a
  variance-preserved DenseNet branch behind a learned scalar gate initialized
  at zero. This starts from the item-only scale while allowing learned position
  contributions to enter during optimization.
- Revision 4 applies only to the four forward+reverse treatments. Their forward
  table retains the standard initialization and their added reverse table is
  initialized exactly to zero. Additive revision 4 therefore starts as its
  matched forward-additive position input and gives the reverse table a
  first-step gradient. Concat revision 4 retains revision 3's item residual and
  zero-gated variance-preserved DenseNet over `[item; forward; reverse]`; only
  the gate receives a first-step branch gradient, and the branch tables begin
  learning after the gate moves.
- Revision 5 applies only to the four forward+reverse treatments. Its additive
  form nests the exact forward-add control before a separately registered
  bounded reverse correction. Its concat form nests the exact forward-concat
  revision-3 branch before a separately registered, variance-preserved
  DenseNet over `[original item; forward; reverse]`. Both correction gates are
  zero initialized and their scalar coefficients are bounded in magnitude by
  0.1. Forward-control parameters are constructed and registered before every
  reverse-correction parameter so their position-input function matches the
  corresponding control at initialization; this does not claim equality of
  unrelated transformer-layer parameters.
- Revision 6 keeps the exact revision-5 architecture and initialization but
  reduces both reverse-correction coefficient bounds from 0.1 to 0.025.
- Revision 7 keeps the revision-6 architecture and 0.025 bound. It marks the
  reverse-correction parameter subtree for RNG-isolated broad initialization.
  Correction module construction is also RNG-isolated. The correction still
  receives the normal project or μP initializer, while restoring the RNG state
  after both construction and that contiguous parameter run makes every
  subsequent shared parameter and the zero-gate model output exactly match its
  seeded forward-only control.
- Existing revision 1 through revision 6 artifacts remain immutable and
  reconstructable. Forward-only concat remains revision 3. Only revision 7
  combined treatments are eligible for future native-500M selection and
  reporting; older combined treatments remain audit history only.

## Execution and verification

- Add independent expected-value tests for RoPE and an opt-in real-A100 packed
  varlen/GQA forward-backward check. The current implementation already matched
  FlashAttention numerically in a read-only audit; the tests make that proof
  persistent.
- Focused tests must prove two distinct learned tables, additive versus
  concatenated fusion, correct forward/reverse indices, ALiBi composition,
  unchanged causal masking, and exact treatment identities.
- Before any 500M submission, run one native-50M diagnostic at deep LR 0.012
  for each of the twelve primary treatments and the two lower-base RoPE
  diagnostics: 14 correctness/debug runs. Their metrics cannot select or
  remove a 500M treatment.
- The current diagnostic surface reuses its ten unchanged exact artifacts and
  schedules only the four new revision-7 forward+reverse identities. The
  native-500M launcher fails closed until all fourteen exact diagnostics are
  complete and no revision-7 arm is materially below its matched forward-only
  control under the diagnostic implementation gate. This gate catches broken
  implementations; it is not treatment selection or reader evidence.
- Initial native-500M batch: twelve primary treatments by three deep rates, 36
  runs. Conditional lower-base RoPE batch: two bases by three deep rates, 6
  runs. Boundary continuations and the predeclared confirmations follow only
  as required.
- All multi-run work uses the persistent shared training queue without manual
  GPU exclusions. Code/protocol receives a blind review before training; raw
  evidence, selection, tables, and claims receive an independent final review.

## Interpretation and reporting

- Reverse learned embeddings genuinely anchor the last token at index zero.
  Reverse RoPE does not: its sequence-length offset cancels from relative
  phases, so it is a sign-reversed parameterization rather than an absolute
  end anchor. The corrected report must state this distinction.
- If plain RoPE is still worse after reference-correct implementation,
  base-aware native tuning, boundary resolution, and paired repeats, compare
  it with no position and RoPE+ALiBi. Recovery from adding ALiBi supports the
  explanation that this next-item task benefits from ALiBi's explicit monotone
  recency prior, which plain RoPE represents only implicitly.
- Replace the historical RQ7 reader table with generated corrected tables for
  learned-position fusion and RoPE/ALiBi. Do not retain invalid historical
  rows in the reader or tuning tables; preserve all raw artifacts.

## Approval

- Exact treatment list, corrected item-plus-position concatenation semantics,
  50M diagnostic stage, native-500M tuning surface, conditional RoPE-base
  branch, confirmation rule, and report replacement: approved by the user.
