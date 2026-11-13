# Verified free-form result

Run `freeform-transfer-20261015T182535Z` completed all ten paired development cases. **Adding model-written advisory constraint IR did not improve the competent fact-ledger control.** FACT_LEDGER passes 9/10, ADVISORY_IR 8/10: eight shared passes, one shared failure, one ledger-only pass, no IR-only wins. This is a small disclosed development result, not a population estimate.

| Condition | Passes | Model calls, including maintenance | Tokens | Exact provider RUB |
|---|---:|---:|---:|---:|
| FACT_LEDGER | 9/10 | 10 | 23,514 | 3.03146738 |
| ADVISORY_IR | 8/10 | 20 | 48,328 | 4.84335275 |
| Total | 17/20 | 30 | 71,842 | 7.87482013 |

The advisory condition costs 59.77% more, including 1.19643518 RUB for ten writer calls. All 30 accepted requests have distinct recorded IDs and the exact requested Qwen alias. There are no recorded failed requests, unknown usage/cost or length finishes. The requested route is Chutes/fp8 with no fallbacks; provider precision/weights/runtime are not independently attested.

**Context pressure does not explain this result.** Ledger memory uses 511–875 Qwen tokens and advisory memory 855–2,526, under the shared 4,096-token ceiling. Every pair has identical raw user requirements, current facts, conflict markers and factual revision history. Offers, task and output bindings are also identical. No padding, policy eviction, overflow or actor-format rejection occurs. The memory ceiling excludes shared offers and other prompt tokens.

Both actors choose a floor despite simultaneous contradictory check-ins and an explicit wait-for-reconciliation rule. Conflict markers are present; the actors incorrectly treat a higher insertion version as newer causal evidence. In the only discordant case, all deployment offers violate either time or coverage requirements. The ledger correctly abstains. The advisory actor acknowledges the unmodeled coverage requirement but treats it as optional and chooses a violating offer. These are evidence/obligation-authority failures, not missing-memory failures.

All ten writer responses parse, but only 20 of 25 proposed predicates are structurally accepted; five are rejected and 16 clauses remain unmodeled. Several accepted predicates use the wrong literal/type or strengthen a conditional into an unconditional rule. Successful actors often recover from the preserved raw instructions. The license case has no accepted predicates, so it gives no live evidence for successful model-extracted snapshot binding. See [the failure audit](failure_audit.md) for per-case evidence.

**Decision:** keep the factual ledger as the competent baseline and do not promote the advisory writer. Constraint memory still needs a benefit beyond that control. Do not patch these disclosed IDs and count a rerun as independent evidence. The next already-frozen experiment tests current E1/E2 closure policy on seven other development families with a generous 32,768-token primary memory ceiling; it is a different mechanism test, not a retest of advisory semantic extraction. Validation/holdout and an authentic matched MAGE comparison remain outstanding.

Verification hashed all 68 archived members against the prepared source/input pins. A network/credential-denied subprocess imported archived code and exactly reproduced all 10 writer prompts, 20 actor prompts, full contexts, grounded actions, full grades and result records. An independent standard-library oracle re-grounded all 35 offers and 114 distinct offer/criterion verdicts, agreeing with all 20 actual choices and saved statuses. The independent oracle validates annotation execution, not blind annotation meaning. Original run files remain byte-for-byte unchanged.

Reproducible evidence: [archive replay](verification.json), [billing audit](accounting.md), [independent oracle](direct_oracle.md), [failure analysis](failure_audit.md). These checks made no new model calls. The run remains ten project-authored, single-decision development workflows with one seed; it proves neither broad agent generality nor superiority over MAGE.
