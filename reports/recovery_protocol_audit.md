# Planning privileges, full submissions, and scoped recovery audit

Audit date: 14 October 2026. Official source pin:
`6cd9de14b71915e39ac742a20dc33785e14b6aab`.
This is a code audit and regression specification, not an inference experiment.

## Authoritative submission semantics

The official runner accepts later revisions to earlier added travelers. It does
not limit an action to the traveler whose request initiated that round.

Evidence in the pinned source:

| Source | Behavior |
| --- | --- |
| `external/MemoryArena/run_travel.py:39` | `parse_all_plans` splits every named section and assigns `name_plan_pairs[name] = plan`; the last full section for a name wins. |
| `external/MemoryArena/run_travel.py:281` | Every raw action is appended to the cumulative plan text. |
| `external/MemoryArena/run_travel.py:327` | The cumulative text is parsed for every added traveler revealed through the current round. |
| `external/MemoryArena/run_travel.py:376` | The final saved submission is the last cumulative result, rather than independent initial per-person plans. |
| `external/MemoryArena/env/env_systems/travel_planner_env/combination.py:54` | Every final person's latest section is parsed into the submitted plan. |
| `external/MemoryArena/env/env_systems/travel_planner_env/prompts.py:12` | The direct-planning system prompt explicitly allows changing previous added travelers, while fixing the base traveler. |
| `external/MemoryArena/env/env_systems/travel_planner_env/prompts.py:168` | The agent's base-person template says to generate plans for all travelers except the finalized base. |

The current-only worker/runner filtering found at the start of this audit
discarded legal revisions, reducing planning privileges in every condition.
Its statement that upstream submits only the current person's section was
incorrect. The accepted contract is a group-level ledger of the latest full
named section for each causally revealed added traveler.

An executable check used AST-extracted copies of the pinned
`parse_all_plans` and `parse_plan_text` functions, with no dataset imports:

1. Bob first submits Day 1 = `Old` and Day 2 = `Old second day`.
2. During Carol's round, the action includes Bob's new full section containing
   only Day 1 = `New`, followed by Carol's own section.
3. Final Bob has only the new Day 1. Day 2 is absent, rather than inherited.
4. Appending an explicit empty Bob section changes his final parsed plan to
   `None`. An earlier valid Bob plan is not silently resurrected.

Absence of a name in a later action preserves its previous full section.
Presence of a name replaces its whole section, including omissions and malformed
or empty replacements. Multiple sections for the same name in one action also
use the last occurrence. A per-slot merge is therefore insufficient.

The official parser mechanically retains arbitrary headers in cumulative text;
an accidentally guessed future name could become matchable later. The research
adapter's chosen causal guard rejects unknown/future headers when emitted and
never promotes them after that person's arrival. It also ignores base-person
revisions. Those guards enforce the task's visibility and fixed-base protocol
and must be identical for every matched baseline, including real MAGE. Preserve
the rejected raw output and record the guard in the run manifest; do not call
this sanitization byte-for-byte reproduction of the runner's unrestricted text
accumulation.

## Authoritative action budget

`TravelPlannerAgent.run_single_person` uses
`for step_idx in range(self.max_steps)` at `agent/travel_planner.py:191` and
issues one model request at line 201. Any number of tool calls returned in that
response are executed within that one step at lines 228-237. A final response
also consumes a step. Exhaustion at lines 257-260 produces no final plan.

With the matched configuration, each newly arriving traveler gets at most 30
planner/model steps. Emitting revisions to Bob during Carol's normal response
spends Carol's current allowance; it does not create an extra allowance for Bob.
For a research controller that performs additional repair calls, initial
planning plus every repair attempt must share the same current-arrival envelope
of 30. Each repair worker receives only the remaining allowance. A fresh process
must never imply a fresh budget. Keep the existing total-group token and RUB
caps across initial planning, extraction, validation, and recovery.

These limits must be checked at dispatch, using the parent-side count of
planner requests, including a request that fails after dispatch. The worker's
reported `total_steps` can corroborate that count but cannot replenish it.
Multiple tool calls are logged separately for latency and coverage, and do not
subtract extra planner steps. Offline scoring calls consume no agent steps.

## Separate actual submissions from memory validity

The external submission ledger and the memory dependency graph have different
roles. The submission ledger records what the agent actually emitted. A memory
validator may mark an emitted choice unsafe for subsequent retrieval, but that
does not retract the externally submitted plan or substitute a null prediction.
The final scorer must receive the latest actual named sections, even when a
validator believes one is wrong. Otherwise a false validator rejection could
artificially change official success.

The final scorer receives one row per expected added traveler, with the latest
`plan`, `submission_version`, `source_round`, and `source_attempt`. Unsubmitted
members receive null plans. Keep the append-only action/revision history in a
separate file; passing that history directly to a scorer that rejects duplicate
traveler rows is incorrect.

Memory must replace whole-plan state consistently with that ledger:

- New full sections append new decision versions and exact action evidence.
- Omitted old days/slots become unavailable; an explicit empty replacement
  withdraws every old plan decision from active retrieval.
- Changing an entity invalidates attributes grounded in its previous entity
  version, until new causal tool evidence establishes attributes of the new
  choice. Old Hotel A prices cannot describe newly submitted Hotel B.
- Dependency closures use the latest actual, valid source version or explicitly
  report a missing/invalid reference. Historical versions remain auditable.
- Replacing Bob and correctly updating Carol in the same final action should
  validate that final state. It should not invalidate the newly repaired Carol
  merely because her ancestor changed earlier in the same batch.

## Gap found in E5 and implemented repair

The audited `LedgerMemory._invalidate` appends invalidation versions and records
`REPLAN_REQUIRED`, affected decisions, and the earliest affected step.
`commit_subtask` can later notice replacements and mark a recovery
`REPLANNED_UNVERIFIED`, but it does not call a planner. The audited runner did
not schedule repair: it committed the failed output, advanced the chronological
adapter, and subsequently abstained when a retrieval returned
`RECOVERY_REQUIRED`. Consequently E5 provided invalidation bookkeeping, not
end-to-end recovery. Reporting successful replanning from that path would be
unsupported.

The implementation now includes a group-level submission ledger and a causal
`GroupSession` repair controller. Parent-side dispatch counts enforce the shared
current-arrival allowance. The controller selects already revealed affected
travelers, rereads bounded diagnostic context, commits full actual sections,
and acknowledges only new action evidence and current decision versions.
Omitted slots and attributes of replaced entities are retired from active
memory. Regression tests cover full-section revisions, exhausted allowances,
actual submissions surviving validator rejection, and ancestor-first partial
repair. Live recovery efficacy remains unmeasured.

The existing `completion_verified` flag must not be populated from hidden
official scores during execution. Under `judgement_mode=none`, deterministic
memory checks can establish internal consistency only. Actual recovery success
is assessed offline after the group; runtime status remains unverified.

## Budgeted causal recovery contract

1. Cache each request only when the chronological adapter reveals it. A repair
   target must come from this cache; never call a data loader for a future
   request or pass the full request list into a worker. The fixed base is an
   immutable anchor and is never a repair target.
2. Start a current-round envelope before the initial planner call. Track
   `current_round`, `steps_used`, `steps_remaining`, group token/RUB usage, and a
   predeclared maximum repair-attempt count. All repairs charge this envelope.
3. Commit the raw actual action to the submission ledger, then validate the
   resulting memory state. Recover only from trace-backed deterministic
   violations or invalidated dependency scopes. Malformed extraction,
   contradictory/unsupported requirements, unknown formats, budget overflow,
   and cycles require explicit diagnosis; do not invent corrective facts.
4. Derive the affected visible added travelers from the invalidation graph.
   Repair ancestors before dependent travelers, with stable traveler-index
   ordering for ties. Recompute the unresolved scope after each actual repair;
   a single response may repair several visible named plans.
5. A repair packet uses the target's already-visible original query, the current
   causal time, the immutable base, a bounded memory context, and a compact
   diagnostic containing causal requirement/decision IDs and observed
   comparisons. It contains no gold plan, gold slot, official score, judge hint,
   oracle annotation, or future request. `subject_traveler_index` can be older
   than `charged_round`; those fields must remain distinct.
6. Provide an explicit recovery-context path. Ordinary `RECOVERY_REQUIRED`
   remains a guard on unrelated planning, but it must not block the scoped
   repair itself. Invalidated decisions may appear only as labeled failure
   evidence, never as active binding facts. Required valid anchors and
   constraints must still obey the same exact memory-token budget. Batches may
   not be concatenated to bypass that limit.
7. Invoke a fresh official planner worker with `max_steps=steps_remaining` and
   the current visible-name map. Count its actual parent-side planner requests.
   Accept every actual legal named section, replacing whole-plan versions in
   both the submission ledger and memory. Additional model extraction/repair
   calls are charged and categorized, never hidden as free maintenance.
8. Stop on resolved deterministic checks, exhausted step/token/RUB budget,
   attempt cap, a repeated unchanged scope, or a nonrepairable condition.
   Record the terminal reason. Exhaustion does not erase the latest actual
   externally submitted plan. Commit the current chronological arrival exactly
   once before the next request becomes visible.
9. Log recovery ID, trigger/source spans, original and replaced decision refs,
   target subject, charged round, causal step, attempt, actual model/tool calls,
   input/output tokens, provider RUB, latency, valid/invalidated scope, and
   terminal reason. Replay uses those accepted actions and backend transitions
   without inference, future requests, or hidden data.

An authentic MAGE comparison must receive the same visible inputs, immutable
base policy, accepted full-section revision privileges, memory budget, tools,
30-step current-round envelope, and group resource caps. If MAGE has no repair
controller, the E5 controller is a declared treatment with charged overhead,
rather than an unreported increase in planning budget.

## Required regression examples

| Case | Expected outcome |
| --- | --- |
| Bob v1 old dinner; Carol emits Bob v2 and Carol's plan | Both revisions accepted; final scorer Bob = v2, with source round Carol. B0, B1, and E3 share this privilege. |
| Dave later JOINs Bob's dinner | Retrieval delivers Bob v2, never v1; Dave can use the corrected choice. |
| Bob v2 omits old Day 2 | Old Day 2 is absent from final submission and active memory. |
| Bob v2 is explicitly empty/malformed | Bob final plan is null; v1 remains history only. |
| Carol action omits Bob entirely | Bob v1 stays the latest actual submission. |
| Two Bob sections in one action | Last complete section wins. |
| Attempted base or future-name revision | Ignored and logged; no effect on later accepted state. |
| Main planner uses 28 steps; repair needs 3 | Repair receives 2, consumes at most 2, and records exhaustion; never gets 30. |
| One planner step calls three tools | Charge one planner step and three tool latencies. |
| Two affected travelers are repaired | Both share the same remaining current-round budget. |
| Main planner uses all 30 | No repair/model dispatch occurs. |
| Bad Bob invalidates Carol; unrelated Erin exists | Only Bob/Carol scope is repaired; Erin's accepted version stays unchanged. |
| False memory validator rejection | Offline scoring still sees the latest actual emitted plan; no synthetic null or oracle repair. |
| Recovery query contains past data only | Guarded worker cannot read hidden/future files; the next request is not revealed early. |
| Group reset and deterministic replay after repair | No previous-group records; replay reproduces exact submission and memory version history. |

The helper and scripted runner tests should demonstrate these protocol rules
without treating scripted outcomes as research evidence. Any benchmark success
assertion must come solely from the isolated pinned official scorer.
