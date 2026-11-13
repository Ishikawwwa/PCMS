# Mnemos Prototype 01

A local concept-formation workbench with a React/TypeScript/React Three Fiber observatory and a FastAPI/SQLite engine. This app is independent of the existing constraint-memory experiments.

## Start

From the repository root in PowerShell:

```powershell
# First installation (Python 3.11+, uv, Node 20.19+ or 22.12+):
.\mnemos\start.ps1 -Install

# Subsequent starts:
.\mnemos\start.ps1
```

Open http://127.0.0.1:8000. The launcher serves the compiled frontend and API together. Use one server process: the single-user worker owns job recovery. The database lives at `mnemos/state/mnemos.sqlite`; override its path with `MNEMOS_DB`. Stop with Ctrl+C. Keep the server bound to loopback.

For development, start the API from the root, then the frontend in a second terminal:

```powershell
.\.venv\Scripts\python.exe -m uvicorn mnemos.backend.app:app --host 127.0.0.1 --port 8000
npm --prefix mnemos/frontend run dev
```

## Polza generation and real embeddings

Text generation uses **Polza**, defaulting to `qwen/qwen3.6-27b` with structured JSON outputs, temperature 0, reasoning disabled and an 8,192-token completion allowance. There is no Ollama or local-generation path. The same chosen model handles extraction and separate validation calls. Change the model ID and frozen advanced settings when creating a new experiment. Polza manages the served revision: exact returned model, provider, request ID, usage, finish reason and RUB cost are recorded for every completed or failed request. This does not pin the provider's physical weights/runtime.

The key is read only on the backend, from `MNEMOS_POLZA_API_KEY`, ignored `mnemos/state/polza.env`, or `POLZA_API_KEY`, in that order. The browser, graph settings and exports never receive it. The configured key is already saved for this machine. `/api/provider` exposes only connection readiness and nonsecret metadata. Paid calls are initiated through explicit operation buttons. Requests with ambiguous delivery are not automatically repeated. Invalid model output remains a failed operation with its accounting intact.

Embeddings use the pinned local pretrained MiniLM encoder; this is vector encoding, not local text generation. Demo remains an optional, conspicuously labeled scripted test mode and is never substituted for Polza. Install the optional embedding dependencies with:

```powershell
.\mnemos\start.ps1 -Install -Embeddings
```

Download the frozen encoder once if it is not already cached:

```powershell
.\mnemos\.venv\Scripts\python.exe -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',revision='1110a243fdf4706b3f48f1d95db1a4f5529b4d41',trust_remote_code=False)"
```

The app loads the cached encoder without model downloads. Its tokenizer truncates long inputs to its own sequence limit (256 tokens for this MiniLM revision); the limit is recorded in embedding receipts and node metadata. Full source text is preserved separately. All variants use the same compatible encoder/revision. Real local compute cost is unmeasured; provider costs are reported in RUB. The current machine's runtime selection is saved in ignored `mnemos/state/runtime.json` for the launcher.

## MemoryArena experimental workspace

The corpus browser registers all **270 public input groups**, labeled development, validation or holdout. Selecting a group loads its original base request, supplied public base itinerary and chronological traveler requests. File checksums are checked against the pinned split manifest; the adapter cannot open hidden answers. Inspecting held-out material makes that run exploratory; this UI does not claim a frozen benchmark result.

A separate dataset contains **94 actual recorded events** from group 120, E1: requests, explicit task contracts, tool calls/results and actual agent submissions. It retains archive path, source file/line SHA-256, original event ID, order and session boundaries. An actual submission closes its causal episode. Repeated actions remain separate occurrences even when their text matches; reimporting the same archived occurrence is deduplicated.

In Inputs & controls, search for a group and select a partition. **Import next source** advances one chronological record; **Import selected corpus** loads the whole chosen group. Import does not make model calls. Extract granular factual episodes, inspect their exact cited excerpts and accept/reject them; then form concepts from related facts and continue the existing formation pipeline. Every click creates a job/event and commit-driven graph update. The Model usage area and Log show prompt/output tokens, actual provider charges, cache hits and failures. Save queries, rate A/C/E retrieval, export JSON/CSV, replay versions, and create a separate run to compare settings.

Two prepared corpora are available for group 120: public inputs and archived execution. The active experimental workspace contains the recorded source observations and the first real Polza episode proposal, awaiting researcher review. An earlier rejected quote-format attempt remains in a separate audit run with its real charge. No historical failure is silently replaced.

The manual graph workspace performs **memory-formation and retrieval experiments over benchmark inputs/observed executions**. For fresh planning and a final official result, use the separate full-case execution workflow below. Hidden gold answers remain separate in both workflows.

## Run a complete MemoryArena case

Use **MemoryArena cases** in the header. Select any of the 270 cases by group number and partition. The case view shows the fixed public base request, a day-by-day itinerary table and each added traveler's raw request. A case is the full group, not a single isolated traveler. Preview is kept outside execution memory.

Choose a memory condition (Mnemos, bounded history, or no persisted memory), a per-traveler decision allowance (up to the official 30-step allowance), and a visible-memory token budget. **Open fresh execution paused** waits before the first paid model decision. **Step action** allows one planner decision and its resulting official tool batch, or one complete memory operation. **Run full case to final score** / **Resume full run** advances through all travelers and the isolated official scorer. Pause takes effect at the next operation boundary; Stop preserves already committed observations and accounting. A failed transport/preflight never becomes a zero task score. Fully accounted model outputs that fail the memory schema are rejected and recorded; planning continues from original evidence without that update. Known failures after a complete traveler commit can resume from that checkpoint without repeating earlier paid planning. Partial travelers and ambiguous server interruptions cannot be automatically repeated. For a failed request with an exact provider generation ID, checkpoint recovery first reads its final Polza history accounting. A separate correction receipt preserves the original failure, resolves the actual charge and never retries inference; unresolved or mismatched accounting still blocks continuation. Recoveries and source revisions remain in the execution bundle.

Each execution starts with a fresh graph, a chronological adapter and a latest-submission ledger. The unchanged pinned public `TravelPlannerAgent` runs in a fresh worker per traveler with official tools. The worker receives only the public base, the current revealed request and bounded earlier memory. It cannot read hidden/future dataset files or credentials. Original tool arguments are used; the optional travel-specific city/price overlays are not enabled by this workspace condition. Prior-traveler full-section revisions follow the existing adapter's actual-submission semantics.

Mnemos execution bundles each planner decision and its ordered tool calls/results into a single immutable execution trace. Individual tool calls/results remain inside that trace and the audit log, never separate graph nodes. After each traveler, extraction batches of up to six traces produce individual contextualized factual episodes (L1), not one broad traveler summary. Concepts (L2) must cite at least two supporting factual episodes; derivation links follow traces → facts → concepts. Independent validation checks each fact and concept, embeds accepted material, discovers cohesive hyperedges and optionally attempts one eligible abstraction. Failed semantic checks remain visible as `needs_review`. Automatic promotions are **machine_checked**, never `human_reviewed`. Similarity never promotes a claim on its own. The planner sees machine-reviewed concepts as fallible interpretations, latest actual submissions as unverified actions, and a budgeted set of original requests/tool observations. Every supplied memory context and omitted record ID is saved. Retrieval is at traveler/session start, matching this local adapter; this is not the MAGE paper's per-action booking runtime.

The Benchmark inspector shows live phases, action history, exact original submissions/revisions and supplied memory contexts. On completion it displays **SR** (all expected travelers fully pass), **PS** (full-plan traveler pass fraction), **SPS** (upstream changed-slot pass fraction), and per-traveler pass counts. Missing/unparsed submissions are materialized by the official scorer as failures. Hidden answers are opened only by the separate scorer after planning; scores never feed back into the graph or planner. The execution bundle includes settings, official source/data/tokenizer pins, captured backend/adapter source, actual predictions, scoring metadata, worker audits, events and paid call receipts. Source archives exclude credentials and preserve continuation revisions. Completed scores can also be exported as CSV. The displayed final plans use the latest accepted submission version, including later revisions of earlier travelers.

These single-case executions are exploratory experiments, including cases that have been previewed or inspected before. A final official score is an actual measured case result, not evidence of held-out generalization or superiority. A new execution is required to compare memory conditions; no graph state is carried across executions.

## Walk through a run

1. Create a clearly labeled Demo run or configure Polza.
2. Import a fixture, paste text, or load JSONL. JSONL records need `content`; optional `session`, `provenance`, and arbitrary source metadata are retained. Exact normalized-content duplicates are skipped.
3. Extract granular facts as episodes. Session boundaries are preserved; each batch contains up to six unused traces and returns 1–32 separately cited factual episodes. Batch N processes up to N such batches (maximum 20). Inspect candidates and original quotes, then accept/reject them with a review reason.
4. Form concepts from at least two related validated factual episodes in a session. Review candidates individually or explicitly accept the pending batch through Inputs & controls. Unaccepted concepts cannot be embedded or used for discovery. Edit a concept proposal to create a new candidate with a preserved predecessor.
5. Embed accepted objects, then deliberately **Recompute PCA**. New embeddings never silently refit the projection. Objects without PCA coordinates use a labeled display layout.
6. Associate accepted embedded concepts. Membership threshold, minimum pairwise cosine and top K are adjustable; overrides are retained with the operation. Run settings control cardinality (2–6), Jaccard filtering and depth. Discovery permits mixed concept levels and overlapping groups; layer visibility filters only the scene. The initial neighborhood algorithm is pair-seeded and greedily preserves minimum pairwise similarity, avoiding single-link components.
7. Select a hyperedge and crystallize. The generator can return no abstraction. Exact duplicates, missing cross-member evidence, invalid output, and failed independent validation prevent promotion. A passing result is still **needs_review** until a researcher accepts it. Edited abstractions are validated again.
8. Embed the accepted meta-concept and associate again to form recursive candidate groups. Inputs & controls can run the bounded embed → project → associate sequence; pause stops future steps, and cancellation suppresses pending graph output. In-flight provider inference may still finish and is audited.
9. Ask memory and compare A/C/E, or run the held-out queries frozen before concept formation. Inspect IDs, original cosine scores, edge centroid scores, expanded source evidence and latency. Human review buttons save groundedness/correctness judgments and reasons. Export the entire run as JSON or retrieval rows as CSV.
10. Challenge a trace after inspecting its dependency impact. Dependent episodes/concepts become `needs_revalidation`; affected hyperedges become stale. Revalidation is a separate operation. A disputed source cannot be silently approved; corrective information is imported as a new source. History is preserved.

Pipeline tabs and Next only change focus; labeled Run buttons execute operations. Use the Objects panel for keyboard-accessible selection. Fullscreen includes both scene and pipeline; inspector and controls remain available as overlays. Escape exits fullscreen. Pause rotation, orbit by dragging, and zoom with the wheel. Frame timings are measured from the browser render loop; Configuration can record the latest sample as an event.

## Fixtures and evidence limits

- Agent task observations: nine **authored synthetic** examples, useful for exercising the flow.
- Recorded agent tool interactions: six actual planner tool events copied from this repository's MemoryArena run. Original source paths, line numbers and SHA-256 hashes are retained. These are synthetic benchmark tool data; call/result pairing is not independently verified and no task-success claim is attached.
- Star Wars: six paraphrased film observations labeled canon, not agent traces.
- 500-node fixture: synthetic rendering samples only. Use it to record actual FPS/mean/p95 frame times on the target machine; no performance target is asserted.

Schema validation and exact quotes establish structural grounding, not semantic truth. The separate LLM validator assists the human reviewer; no automated scientific correctness guarantee is claimed. Demo acceptance must not be described as a validated scientific abstraction. No benchmark superiority is asserted.

Retrieval modes are A raw traces, B episodes, C ordinary concepts, D concepts plus association expansion, and E concepts/meta-concepts plus association expansion. The UI defaults to A/C/E; the API supports all five. Edge scores use normalized centroids of persisted member vectors. The answer is an extractive citation list, not an additional LLM answer. Character counts are measured; retrieval token counts are `null` because an LLM tokenizer has not been configured. Human judgments remain unrated until entered. Held-out membership is recorded, but avoiding semantic leakage remains the researcher's responsibility.

## API and persistence

All documented endpoints are under `/api`; interactive schemas are at `/docs`. Actions require `expected_graph_version` and an `idempotency_key`. Reusing a key with the same request returns its job; a changed request or stale version receives HTTP 409. SQLite reserves one mutating job per run. Mutations commit a full snapshot, materialized nodes/links/hyperedges, graph version and graph.patch event in one transaction. SSE resumes using `Last-Event-ID` and carries only committed snapshots. This prototype sends whole snapshots rather than minimal field deltas, appropriate for the small initial corpus.

The append-only event log is protected from UPDATE/DELETE by SQLite triggers. Replay uses historical committed snapshots without repeating inference; inspection supports a historical graph version. Raw content is never edited. New proposals/edits preserve predecessor and source lineage. Complete model prompts, structured/raw responses, model identity, usage and errors are retained in call receipts, including unsuccessful jobs. Complete outputs are cached by input/settings/prompt hash; schema gates run before caching. Embeddings retain canonical input, frozen revision and dimensions.

Do not open the same database with multiple server processes. On restart, previously active jobs are marked failed with an explicit ambiguous-execution notice, preserving receipts and graph state. Cancellation does not terminate an already-running paid request; it suppresses its commit. The app is a loopback research tool, not an authenticated multi-user service.

## Verification

```powershell
.\.venv\Scripts\python.exe -m pytest mnemos/tests -q --basetemp=tmp/pytest-mnemos-local
npm --prefix mnemos/frontend run build
```

Tests exercise recursive formation, source provenance, invalidation, numerical cosine agreement, non-chaining overlapping groups, projection independence, atomic failure, idempotency, stale versions, cancellation, validator rejection, model budgets, export and replay. Browser verification exercises the actual buttons. A full source bundle is the `mnemos` directory plus its package lock and fixtures; generated dependencies, database and build products are ignored.

Implementation references: [Sentence Transformer revision and normalization](https://www.sbert.net/docs/package_reference/sentence_transformer/model.html), [FastAPI streaming responses](https://fastapi.tiangolo.com/advanced/custom-response/), [React Three Fiber Canvas](https://r3f.docs.pmnd.rs/api/canvas).

### Rebuild legacy execution memory

On a completed benchmark, **Rebuild layered memory** creates a separate research workspace from the saved source events, with each decision and its tool results grouped into an execution trace. It preserves original event/node identities inside the trace, imports no old summaries or concepts, and makes no planner calls. Use Episode to extract facts, review/validate them, then Concept to form concepts before association and abstraction. The original scored execution and all its immutable history remain intact; the rebuilt workspace is an offline memory-structure experiment and has no new benchmark score.
