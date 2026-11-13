# Development trace findings and proposed next candidate

14 October 2026. These are causal retrieval diagnostics on the recorded E0
development trace in `qwen36-g6-E0-E3-20261014-r9`. They make no model calls, open
no answer dataset, and do not measure alternate planner performance. The live
E0/E3 pilot retains its frozen implementation throughout execution.

The first real trace falsifies a useful assumption in the initial candidate:
protecting the entire historical dependency closure does not ensure useful
planner context under a fixed 4,096-token cap. E3 can replace all binding context
with an overflow warning. Merely packing the same rows into reversible tables
reduces size, but does not solve the larger late-traveler cases.

| Arrival | E3 compact full closure | Lossless table encoding | Proposed materialized frontier, ordinary rows |
| --- | ---: | ---: | ---: |
| Benjamin | 3,241 | 2,886 | 3,241 |
| Andrew | 2,242 | 1,756 | 2,351 |
| Thomas | 4,229 | 3,320 | 2,949 |
| Brenda | 10,343 | 8,037 | 3,276 |
| Violet | 11,887 | 9,070 | 1,273 |

These numbers are exact tokens from the pinned Qwen tokenizer. The table and
frontier scripts record their captured log-prefix hashes. Different rows describe
different information policies; required-record recall is not directly comparable
after defining a narrower frontier. The original E3 trace additionally contains
a value-reference defect described below. This is development evidence for an
implementation change, not an efficacy result or clean ablation.

Three mechanisms explain the next proposed candidate:

1. Treat an actual submitted choice as a materialized fact. Later requirements
   consume that exact choice and independently grounded properties, while the
   historical requirements that led to it remain in the append-only archive.
   An observed fact does not assert that the earlier traveler satisfied their
   requirements. Revision and invalidation must still inspect the full graph.
2. Avoid retrieving complete itineraries solely because their names appear in a
   companion-list introduction. The current prototype removes only complete
   list lines made entirely of already known traveler names. All other literal
   mentions keep the existing lexical fallback. A successful JSON parse does
   not prove semantic extraction coverage.
3. Resolve the explicit `:value` attribute as an alias of the stored choice ID.
   The writer uses `decision:Benjamin:day1:accommodation:value`, while the ledger
   stores `decision:Benjamin:day1:accommodation`. Central normalization must
   preserve versions, causal boundaries, withdrawals, and property identities.

A separate live-data defect concerns city identity. Submitted entities use
`San Antonio(Texas)` and `Houston(Texas)`, while returned rows use bare city
names. Exact string comparison currently fails to establish their meal and
accommodation properties. The official annotation tools strip the parenthesized
state suffix. A conservative common matcher must prefer exact identity, accept
the equivalent bare city, reject differing annotated states and ambiguous rows,
and leave names exact. No extra tool query or hidden lookup is needed.

Prototype files:

- `scripts/audit_recorded_contexts.py`: alternate E0/E2/E3 reads on identical
  actual observations and submissions.
- `scripts/prototype_binding_tables.py`: reversible column tables and reference
  symbols, with exact row-content/multiplicity round trips.
- `scripts/prototype_materialized_frontier.py`: an isolated retrieval hypothesis,
  with a local value-alias correction and narrower companion-list filtering.

Do not promote this prototype directly to the recovery controller. Its alias
handling is confined to resolution and does not normalize every invalidation
path. Archive graph traversal and planner frontier traversal need explicit
separation. The next runnable candidate should have component ablations,
regressions against this trace, and fresh complete-group planning under the same
model, environment, step allowance, visible-memory cap, and accounting policy.

R9 source has been preserved in its `source_archive.zip` and sidecar. All 32
available original source pins match. The manifest sanitizer redacted the
`secrets.py` pin; its current bytes are captured, but the missing original hash
remains unverified. Archive integrity verification and deterministic replay must
retain that partial provenance label.

An authentic MAGE comparison still requires the authors' source/configuration
or confirmed equivalence and a shared execution protocol. The existing public
CSV planner differs from the paper's stateful booking environment and per-action
memory reads. These diagnoses do not support a claim of beating MAGE.
