# Live pilot diagnostics, 14 October 2026

Exact model: `qwen/qwen3.6-27b`; requested route: `chutes/fp8` only,
fallbacks disabled. Returned provider fields were null. This report contains
execution diagnostics, not a measured memory effect or a MAGE comparison.

- The small authenticated non-reasoning probe returned `READY`, using 19 input
  tokens and 2 output tokens, with reported billing of 0.00125065 RUB.
- `qwen36-g6-E0-E3-20261014` reached reasoning-enabled inference preflight, but
  its first long extraction exceeded a 30-second response timeout. Delivery and
  billing are ambiguous; the run has no predictions or scores.
- The attempted second replicate stopped during its public connection/catalog
  check, before paid preflight or benchmark dispatch.
- `qwen36-g6-E0-E3-20261014-r3` passed reasoning-enabled fixed-route preflight.
  E0's base extraction completed in 166.27 seconds, used 5,814 tokens, and cost
  1.22125595 RUB. Its seven extracted constraints had malformed object shapes
  and were rejected by the ledger. The extraction adapter had incorrectly
  reported parser success. The first added-traveler extraction then exceeded
  its 180-second response timeout. Its usage and billing remain unknown. The
  caller confirmed terminal failure; no traveler predictions or scores exist.

No ambiguous paid request was automatically repeated. Different run IDs are
explicit development attempts with recorded protocol changes, not pooled or
completed benchmark replicates. Unknown billing is not represented as zero.

The next version requests schema-constrained extraction and validates the
constraint object structure independently. It stores the immutable public base
query and exact supplied itinerary deterministically, removing a redundant
model interpretation step from every condition. Streaming transport is implemented
using the official Polza SSE contract; complete final usage is required
before accounting can be considered complete. These are development fixes and
must be frozen before the next matched comparison.

The reasoning-enabled streamed writer in R7 used its entire 8,192 output tokens
without emitting JSON. The native planner subsequently completed Benjamin's
actual itinerary in two model steps. The development run was stopped after this
readiness failure while Andrew's writer request was streaming. That request's
ID is retained and its billing is explicitly unknown; the caller's process was
confirmed terminal. Offline scoring of the one actual completed itinerary found
12 of 13 changed slots passed, with full-plan pass false. This partial diagnostic
does not measure group success or a memory effect.

A separate non-reasoning writer diagnostic emitted JSON but still hit the
8,192-token cap and returned a truncated object. The next diagnostic uses a
16,384-token extraction cap, shared across ledger conditions, while retaining
the planner's settings. Completed raw writer text is now retained separately
for diagnostics. There are no automatic paid length retries. A development-only
readiness gate stops future pilots at parser failure rather than spending a full
group on an invalid writer; it is forbidden during validation/holdout scoring.

The 16,384-token non-reasoning writer diagnostic completed: 45 constraints,
zero unresolved items, 1,776 input tokens, 8,217 output tokens, and 2.00841449 RUB.
The ledger audit initially found all 45 E3 records overflowed the 4,096-token
budget because their audit identity/version fields were repeated. The compact
render now retains each reference once and preserves every binding/evidence
field: all 45 records fit at 3,286 tokens, with full required-record coverage.
A real-response regression fixture preserves this counterexample without gold
or future data. E0 uses 2,714 tokens for the same extracted records.

The audit also found four valid numeric bounds falsely rejected when followed
by sentence periods; that literal parser bug is fixed. E4 still rejects four
machine policy labels not literally supported by the request. Those semantic
limitations remain explicit, and E4/E5/E6 are not promoted above E3 based on
their feature count. The writer prompt now forbids invented labels and asks it
to mark unrepresentable/ambiguous policies unresolved.

The new R9 paired development pilot uses a saved catalog snapshot with recorded
hash/timestamp plus live exact-model preflight. R8 failed its public catalog
transfer before any paid request. R9's complete-group metrics remain pending.

Regression checks also exposed two memory defects: missing prior properties
could suppress the actual entity needed for a tool lookup, and missing required
choice slots did not produce repair targets. The corrected ledger retains only
the valid actual entity as an unresolved-property anchor, and derives omitted
slot repair subjects from causal constraints without inventing choices. Newly
visible tool rows can establish properties of previously submitted/base choices
through exact entity/city joins, without re-submitting those choices or making
extra searches.
