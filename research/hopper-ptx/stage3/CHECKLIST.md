# Stage 3 audit and frozen component contract (2026-10-13)

Entry versions: outer `63d6e475d381308f1afd18cc1bdd3b163d07f4ef`, inner
`f4fe783ac7fc4f1cdaa9efa72a3575537d4fba36`, both `research/hopper-ptx`.
Existing untracked `experiments/` and inner `.serena/` are preserved.

| Entry category | Capability | Work required |
| --- | --- | --- |
| Implemented with saved evidence | sm_90a, BF16 MMA parser, 88-register WGMMA tuple | Recheck on rebuilt runtime; inspect metadata rather than PASS alone |
| Implemented without valid numeric evidence | ldmatrix m8n8 x1/x2/x4, normal/trans, shared/shared::cta | Execute all 12 forms; all lanes/registers; nonuniform bit patterns; independently distribute matrix elements to lanes |
| Implemented with later evidence | parameter alignment/copy guards, WGMMA SS/RS 64/88 tuples | Reuse stage8 parameter tests and stage6 regression |
| Missing validation | ldmatrix address type, full row bounds, 64-bit truncation | Add explicit validation and illegal address tests; retain supported 32/64-bit and symbol forms |
| Deferred, not accepted here | ldmatrix detailed timing/layout cost | Stage6/12 resource integration; legal detailed execution must continue to reject |
| Deferred, not accepted here | mma.sync arithmetic, generic-address ldmatrix, other shapes/types, predicated forms | Dense inventory stage10; required execution stage11/12; original FA3 stage15 |
| Deferred, not accepted here | complete Dense/Python/H100 | Stages11/13/14 |

Frozen positive matrix: 12 opcode forms × (compact b32, padded/permuted b64
with address offset, end-boundary/permuted b32) = 36 cases, each 2 CTAs × 2
warps. Extra symbol, address/destination alias and u32/s32/u64/s64 address cases
(42 numeric cases total); a short final row is an additional range negative.
Output is copied from
simulator global memory and saved in full; reference is comparison only.
16-bit loads require exact bit equality (absolute/RMS error 0).
Negative cases cover modifier, arity, tuple count/type/duplicate, address
type/space, alignment/range/high-bit, and incomplete active warps. New MMA type
and WGMMA forms require parser positives and modifier/operand negatives.
Ordinary stage2 numeric and stage8 parameter/label regressions are mandatory.

No stage10 dependency is needed for this component contract; generated test
PTX and CUDA12.8 ptxas validation are local preparation, not Dense completion.
No later implementation stage begins during this session.

Final result: all four current stage3 criteria pass on inner
`eca83b30959381ee14bb26384a0c4f01c3d35c01`; outer is the commit containing this
checklist/report. See `evidence/acceptance-20261013/frozen/gate.json` and the
Chinese stage3 report for actual runtime, commands, failures and boundaries.
The original unsupported timing/arithmetic/app entries remain deferred.
