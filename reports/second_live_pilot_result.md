# Completed E0/E3F development run

Run `qwen36-E0-E3F-vpn-off-20261014T104907393354Z` completed on 14 October 2026. This is one development family with six added travelers. The original artifacts are preserved.

| Condition | Group success SR | Full-plan PS | Changed-slot SPS | Recorded main billing RUB | Total model tokens |
|---|---:|---:|---:|---:|---:|
| E0 | 0 | 0 | 0.621439 | 23.23014926 | 228227 |
| E3F | 0 | 0 | 0.722222 | 18.87134955 | 175343 |

SPS improves by 10.08 percentage points, while the primary complete-group endpoint remains zero. Neither condition produced a fully correct added-traveler plan. E3F's lower cost includes a skipped Brenda turn, so it does not establish greater efficiency at equal completed work. A single family cannot estimate population uncertainty or establish superiority over MAGE.

The 39 completed main requests cost 42.10149881 RUB. The separately billed preflight cost 0.03087946 RUB, for 42.13237827 RUB in this successful suite. Earlier attempts and probes are excluded from this figure.

Every completed main/preflight response reports the requested exact `qwen/qwen3.6-27b` model and the configured routing. Provider-family metadata reports Chutes; the precision, underlying weights and runtime revision remain unverified. Both conditions share temperature 0, seed 20261013, a 16,384-token planner completion cap, a 4,096-token visible-memory cap, and 30 model dispatches per current arrival.

All 35 source pins and 28 upstream source pins verify. Both archived-source memory replays match. A fresh isolated invocation of the pinned official scorer reproduces each complete saved score object exactly. The source archive SHA256 is `2d29217eea1fb7f4001d53555add51dfdab2ac88419efea29a61a4279872a192`. Detailed model, billing, hash and scoring evidence is in [the verification record](fresh_frontier_verification/identity_billing_scoring_verification.json).

## Failures that drive the next implementation

Benjamin, Andrew and Thomas each achieve E3F SPS 1.0 but PS 0. Each changes one slot outside the current requested scope: Benjamin changes day-2 lunch; Andrew and Thomas repeat their first-day hotel into day 2. The fixed public base is already supplied in every common planner prompt, but the existing prompt does not explicitly declare preservation of unmodified slots. A new experiment declares that task contract for every compared condition, and the constraint candidate checks only conservatively verified unchanged slots. A requested slot is an exception even if its valid eventual choice equals the base; no scorer or gold-difference set identifies exceptions.

Brenda's 48 protected records require 4,915 tokens. E3F therefore produces an explicit memory-overflow result and makes zero planner calls for her. Her actual plan is absent, which causes unresolved references for Violet and Marge. Memory must retain that absence rather than invent intended choices. No E3F completion ended with `finish_reason=length` in this run.

Benjamin's first-day hotel has one space after its colon, while the already observed official row has two. A new shared identity join permits unique ASCII whitespace normalization after literal matching, preserving the exact actual choice, original row and event evidence. Ambiguous names remain unresolved. It recovers Benjamin's previously unavailable hotel price and rating without any new database or model call.

The full failure analysis is [the causal failure audit](fresh_frontier_failure_audit/audit.md).

## Offline feasibility of the new representation

E3T keeps the E3F selection policy and uses reversible tables and a visible reference dictionary. On the exact chronological E3F actions, Brenda's identical 48 selected records require **3,515 tokens**. All six recorded arrival reads fit the same 4,096-token cap, including the additional hotel properties established by the shared whitespace fix. Ordered row multiplicity, nested keys, exact values, units, versions, references and sources survive pack/unpack and rendered JSON readback. Corrupted or missing records fail the equality audit.

The downstream missing Brenda references remain absent on this fixed trace. These are context-feasibility measurements; there are no E3T live scores or counterfactual corrected plans. [The table audit](fresh_frontier_table_audit/audit.md) includes a separate E0 fixed-action sensitivity check and reproducible evidence.

The next candidate E3I combines these tables with declared unchanged-slot obligations and a narrow model repair loop. On the same original actions, the additional default/exception representation raises Brenda's context to 3,701 tokens, still within 4,096. The corresponding E0 fixed-action sensitivity trace peaks at 3,706. Scope coverage is independently checked from the raw visible request and does not certify cuisine, numeric relation or house-rule interpretation. Only actual model submissions can repair a plan, charged to the same remaining arrival allowance. Generic semantic hard validation remains disabled. A fresh live comparison is required before attributing improved complete-plan success to this implementation. The final offline suite passes 550 tests and 53 subtests; the [candidate review](fresh_candidate_review/audit.json) records causal budgets and known limits.

Authentic MAGE is still absent from the runnable comparison. Its published model matches the selected backbone, but the public planner's CSV tools and session-start memory schedule differ from the paper protocol; [the protocol audit](mage_protocol_audit.md) documents that limit.
