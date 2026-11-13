# Entity-grounded memory implementation

This version implements the 30 October 2026 design in a separate, versioned workflow. Archived legacy experiments and their official scores are preserved. The selected workspace is **Entity-grounded · MemoryArena 120** (`run_fc0552fd6bbc474a`). Its 38 original execution traces were copied with event boundaries and hashes, not regenerated or relabeled as facts.

## Workflow

1. **Traces** previews parser/deduplication decisions and exact ordered source events. Commit is separate.
2. **Episodes** proposes chronological episodes with unknown outcomes retained. A real Polza smoke proposal is ready under Operations → propose episodes; inspect its original event span and independent validation, then accept/reject with a reason.
3. **Entities** preserves namespaces and explicit identity candidates. Unresolved identities cannot be used as resolved fact subjects. Review can explicitly choose a distinct identity or an existing scoped identity.
4. **Facts / State** extract literal values, units, required/preferred modality, validity, scoped subjects, tasks, branches, decisions, pending work and dependencies. They remain independent of concepts.
5. **Concepts** use accepted episodes/facts/states. Embedding uses the real pinned MiniLM adapter; the exact canonical input/hash/model/vector are recorded. PCA refits explicitly and is never retrieval input. Structural layouts are labeled separately.
6. **Hyperedges** proposes overlapping groups of 3–6 accepted concepts/rules. All pairwise cosine scores, embedding IDs, thresholds and bounded-search near misses are retained. No source-diversity bonus or hidden diversity gate is used for associations.
7. **Rules** are separately generated and independently checked; acceptance remains an experimental hypothesis, not transferable truth. Required inputs, applicability, exceptions, counterevidence and source roots are stored. Unsupported and stale rules are excluded. Free-form applicability is conservatively unbound unless supported by the explicit binding vocabulary. Promotion is not inferred from similarity or a human click.
8. **Prompt** resolves scoped aliases, traverses typed links under explicit hop/node limits, checks time/branch/status, exposes conflicts and unknowns, and prioritizes required facts/current commitments. It records the exact serialized role/message JSON, hash, pinned tokenizer count, included/excluded reasons and paths. Missing required coverage produces `insufficient_context_budget`; an unpopulated fact graph produces `no_relevant_memory`.
9. Context dispatch is disabled unless the frozen run configuration explicitly enables the Polza agent (`agent_enabled: true`). Dispatch checks the reviewed message hash and graph/config snapshots. The controlled messages are identical to preview. Provider-owned framing/system instructions are not claimed to be visible.
10. Challenges preview the complete dependency closure, then invalidate it atomically. Edited proposals create new records and preserve predecessors. Full exports restore with checksum/provenance checks. Redacted manifests explicitly cannot replay omitted source/model/prompt payloads.

## MemoryArena

The existing isolated planner, chronological request release, source pins and official scorer are reused. A–D are selectable in the case picker: episodic; entity/state; entity/direct-rule; full hypergraph. Each starts with fresh memory. Unlike legacy automatic experiments, A–D pause for researcher review at proposed memory mutations. Use **Review current memory step**, inspect and commit/cancel the pending proposal, then Resume. The full controlled planner prompt is available at each decision boundary and in call receipts. Variant B stops at entities/facts/state; C and D additionally validate and embed accepted rules. D visits every accepted hyperedge from the current consolidation, rather than only the last one.

## Durable step-by-step execution

In an entity-grounded workspace, choose **Configure step-by-step pipeline**, select the imported evidence scope and A–D variant, and optionally enter a task query. Creating the plan performs no inference. **Execute one step** executes one algorithm/model operation. **Continue to next review** continues through read-only stages and stops at every proposed graph mutation. Review actual source spans, independent validation, request/response receipts and billing, select proposals, and enter your decision reason. Committing in continuation mode advances to the next review; in single-step mode it remains paused.

The persistent plan includes source inspection/import, episodes, entities, facts, state, concepts, real embeddings, explicit PCA, full-dimensional hyperedge discovery, a rule proposal for each accepted association, separate rule validation, rule embeddings and projection. Query execution has separately executed and recorded entity resolution, typed traversal, ranked facts/rule-slot binding and exact token-budgeted message assembly. Accepted rules remain hypotheses. Inapplicable rules, absent prerequisites, abstentions and no qualifying associations produce visible exclusions or `no_op`, never fabricated objects. Context dispatch remains a separate explicit operation, disabled without a configured agent. A completed memory pipeline is not an official benchmark result.

Extraction covers all selected evidence in bounded batches (default eight traces); each batch records its exact trace IDs and has a separate review boundary. The graph/config pins, cursor, decisions, outputs, failures and costs persist in SQLite. Status polling is read-only. A restart preserves completed previews and flags interrupted work for explicit retry; it never silently repeats paid generation. Pause takes effect at the current operation boundary. Cancelling retains receipts and accepted memory. Changes outside the plan cause a stale-version failure requiring a new pass. Full exports include plans and their per-stage results; restored plans are archived, never automatically executed.

API: `GET/POST /api/runs/{id}/pipeline`, `POST /api/runs/{id}/pipeline/control` (`step`, `run`, `pause`, `retry`, `skip`, `cancel`), and `GET /api/runs/{id}/operations/{operation_id}/calls`. Individual operations remain available under advanced controls.

A–D context comparison in the Prompt pane does **not** execute or score benchmark tasks. No new A–D performance result is claimed by this implementation. Historical case-120 scores remain attached to their original runs. The pinned protocol is used rather than silently updating upstream files.

Primary materials checked on 30 October 2026: [official project](https://memoryarena.github.io/), [official repository README](https://github.com/ZexueHe/MemoryArena/blob/main/README.md), plus the locally pinned worker/scorer/readiness manifests.

## Verification

- Backend regression suite covers legacy behavior and new provenance, identity, temporal conflict, typed traversal, missing values, exact prompt dispatch, budgets, all-pairs cohesion, overlapping groups, source invalidation, revision immutability, API isolation, export/restore and benchmark review pauses.
- A real Polza extraction initially failed the exact-span check and did not mutate the graph. The constrained-span follow-up succeeded and independently validated one episode. Both attempts are retained; total reported cost was approximately RUB 0.6990. The successful proposal remains uncommitted for researcher review.
- The pinned `sentence-transformers/all-MiniLM-L6-v2@1110a243fdf4706b3f48f1d95db1a4f5529b4d41` produced a real, normalized 384-dimensional embedding from its local cache. No local text generation is used.
- Session/type interpretation uses named labels and shapes as well as color. Keyboard-accessible object/relation buttons supplement the canvas.

Before changes, the existing database was backed up to `mnemos/state/pre_entity_grounded_backup.sqlite`. New command schemas are available at `/api/grounded/schema`; API transport schemas are also in `/openapi.json`.
