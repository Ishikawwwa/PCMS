# PCMS

Per-concept memory subspaces (PCMS) indexes source-backed evidence by concept, tracks confirmed state, and retrieves relevant prior decisions within a token budget. The Travel Planning agent runs on MemoryArena with the released travel tools and scorer. The local lab console supports PCMS-only runs and paired runs with the authors’ MAGE implementation.

See the [lab setup and run instructions](reports/concept_subspaces/lab_console.md). The PCMS implementation, runner, and dashboard are in [experiments/concept_subspaces](experiments/concept_subspaces). The repository also includes the supporting constraint-memory experiments and the separate [Mnemos workbench](mnemos/README.md).

## Constraint Memory Lab

A runnable research implementation aimed at improving MemoryArena Group Travel Planning beyond MAGE with the **same Qwen3.6-27B backbone**, served by Polza.ai. It is not yet a demonstrated performance improvement.

The portable `ConstraintMemory` API stores opaque facts, explicit contracts, exact evidence and causal versions across domains. It reports unresolved semantics, stale or conflicting observations and mixed units as UNVERIFIED. The travel-specific E3R development adapter adds individually verified request checks and fresh repair evidence to the tested E3I frontier. Historical requirements and actual imperfect choices remain archived. See [the generalization contract](reports/generalization_contract.md) for the release criteria and evidence limits.

## Mnemos experimental workspace

The new [Mnemos workbench](mnemos/README.md) provides a local 3D research interface for evidence-backed concept formation. Generation runs through Polza, with real pretrained embeddings, chronological MemoryArena corpora, review gates, persistent hyperedges, retrieval comparisons, costs and replay. Start it with `.\mnemos\start.ps1` and open `http://127.0.0.1:8000`. This graph-formation experiment is separate from the historical travel-planning experiments below.

## Current evidence

- Official MemoryArena source, dataset, tool databases, Qwen tokenizer, and base-family splits are downloaded and pinned.
- Synthetic causal fixtures, real tool/parser integration, isolated official scoring, and deterministic replay are covered by tests.
- Local tests cover official-worker integration, independent scope checks, exact tables, inherited-slot repair and replay, plus independently designed workflows in scheduling, procurement, deployment and inventory. Tests make no model calls; the latest counts and source/test hashes are in `reports/verification.json`.
- The first paired development group is complete: both E0 and E3 have SR 0 and PS 0; SPS is 0.3483 and 0.3333 respectively. The verified charge is 33.0048 RUB, excluding earlier probes. Both memory snapshots replay exactly. See `reports/first_live_pilot_result.md` for failure mechanisms and the original source-pin limitation.
- The second paired E0/E3F run is complete and independently verified: both have SR 0 and PS 0; SPS is 0.6214 and 0.7222. Successful-suite billing is 42.13237827 RUB including preflight. See `reports/second_live_pilot_result.md`. Three E3F travelers pass every scored changed-slot constraint but fail an unchanged base slot. Brenda's memory overflow skips her planner entirely.
- The third paired E0/E3I run is independently verified: full-plan passes are 1/6 versus 2/6, SPS 0.3935 versus 0.7511, and both SR values remain zero. Total billing including preflight is 50.10287685 RUB. All contexts fit 4096 tokens. See `reports/third_live_pilot_result.md`; this repeatedly inspected development family does not establish generalization.
- Independent portable API workflows test unrelated domains, opaque names, alternate layouts, numeric representations, revisions, conflicting observations and unknown semantics. The structured model diagnostic stopped at its spending threshold after 59/72 conditions: on 29 complete pairs CONSTRAINT passes 29 and capped HISTORY passes 20. Every difference is policy loss under pressure; this is not broad generalization. Archived replay and an independent oracle verify the saved results in `reports/portable_transfer_partial_verification/verification.md`.
- The completed free-form diagnostic passes 9/10 with FACT_LEDGER and 8/10 with ADVISORY_IR. The extra interpretation costs 59.77% more and is not promoted. These ten development workflows do not establish broad generality. See `reports/freeform_live_verification/summary.md`.
- The seven-family E1/E2 development screen is complete: E1 passes 18/49 full traveler plans and E2 passes 16/49; neither arm passes a complete group. E2 adds the current dependency-closure retrieval policy to the same versioned ledger. This result does not support promoting closure. Original runs and the disclosed infrastructure recovery remain preserved. See `reports/cross_family_completion/analysis.md` and `reports/cross_family_completion/accounting.md`.
- The authors' MAGE release has now been located and pinned locally at `ace693139dc43e8bdb3076da4127483b41a5060f`. Environment and scheduler audits are in `reports/MAGE_author_runtime_audit/`. Its stateful tools, restoration, verification, and scoring differ from this CSV planner. A matched run of the authentic runtime remains outstanding; no claim of beating MAGE is supported.
- The paper's stateful booking tools and per-action memory reads differ from this public CSV planner and session-start read schedule. `reports/mage_protocol_audit.md` documents the comparison limits and a declared path to a local paper-inspired comparator.

See `reports/implementation_status.json` and `reports/research_strategy.md` for the observed state and next experiments.

## Run locally

Requires Python 3.12 and uv. In PowerShell from this directory:

```powershell
uv sync --frozen --python 3.12 --extra dev --extra reports
uv run --no-sync python scripts/prepare_memoryarena.py
uv run --no-sync python scripts/prepare_tokenizer.py
uv run --no-sync constraint-memory check
uv run --no-sync python -m pytest -q --basetemp=tmp/pytest-local
uv run --no-sync constraint-memory probe
uv run --no-sync constraint-memory run --variants E0 E2 E3 B1 B2 --limit 2 --run-id first-qwen-pilot
uv run --no-sync constraint-memory replay runs/first-qwen-pilot
uv run --no-sync constraint-memory report runs/first-qwen-pilot
```

The data and tool database are already prepared in this workspace. Preparation commands make a fresh checkout reproducible. Run directories must be new; the runner refuses to mix cached predictions across variants. The first model command makes a tiny paid completion; `run` makes paid agent and extraction calls.

The supplied key is stored only in ignored `.env`. Use `.env.example` to configure another machine. Credentials are excluded from model workers, manifests, errors, and reports. API traffic uses `https://polza.ai/api/v1`, with the exact model ID `qwen/qwen3.6-27b`.

## Conditions

| ID | Condition |
|---|---|
| B0 | No persisted memory; the fixed public base itinerary remains available to every condition |
| B1 | Complete prior allowed traces, subject to the configured visible-memory cap |
| B2 | BM25 over prior allowed traces |
| E0 | Constraint/outcome lists, scoped definitions, bounded compaction and optional destructive eviction |
| E1 | Exact versioned constraint and decision ledger |
| E2 | E1 plus deterministic dependency closure |
| E3 | E2 plus protected compression; initial candidate with observed overflow |
| E3F | Materialized current constraint frontier; exact consumed facts, full historical archive, and narrower companion-list retrieval; second live development pilot complete |
| E3T | E3F selection policy with reversible exact tables and a visible reference dictionary; presentation ablation |
| E3I | E3T plus conservatively scoped declared unchanged-slot obligations and budgeted current-traveler model repair; verified third development pilot |
| E3R | E3I plus individually evidenced numeric/shared-choice predicates, observed complete-name checks, current-binding repair evidence and shared tool/schema factors; unmeasured development candidate |
| E4 | E3 plus deterministic structural, evidence, and consistency checks |
| E5 | E4 plus targeted invalidation and scoped planner repair within the current arrival's remaining step allowance |
| E6 | E5 plus bounded multi-hop glossary resolution |
| B3 | Authentic MAGE adapter required; unavailable rather than silently approximated |

E5 schedules causal, scoped repair for already revealed travelers, with ancestors before dependents. Original planning and every repair share the same 30-step current-arrival allowance and group token/RUB caps. Repairs stop on an unchanged scope, attempt cap, or exhausted resources. Internal consistency is recorded as `REPLANNED_UNVERIFIED`; only offline scoring assesses task success. This remains experimental and should not be selected as the best architecture without evidence. E4 literal checks do not prove semantic extraction correctness. The rules-only parser is a limited offline diagnostic; live research configs use the same Qwen model for constraint extraction.

The E0/E3I overlay declares the same base-preservation contract in both planner prompts. E3I represents the default plus exact requested-slot exceptions compactly, using the already shared public base, and archives exact leaf values, versions and evidence. Independent raw-request scope coverage permits only unchanged-slot hard checks; it does not certify cuisine or policy interpretation. Unknown or ambiguous scope disables those checks. E3I requests actual model repair of the current traveler only; inherited drift in a separately revised prior traveler remains unvalidated by this narrow controller. No coordinator fills a base/gold plan into a failed submission. The shared contract is a disclosed protocol factor, so cross-run score differences cannot isolate its effect.

E0 `merge` compacts record representations and evicts context when necessary; `drop` also removes evicted records from the retrieval index while preserving the audit archive. It is a deterministic first implementation, not an LLM summary algorithm. E3 keeps binding records exact and reports overflow instead of silently losing required dependencies. Overflow batches are logged but are not concatenated beyond the common budget.

The main E0/E1/E2 controls use one glossary pass to isolate graph closure. `configs/variants/E0_original.yaml` enables the original bounded iterative glossary. Run a single overlay with `--variant-config configs/variants/E0_original.yaml`; separate run manifests retain these micro-ablation settings.

The pinned Qwen tokenizer enforces visible-memory budgets, while API-reported usage accounts for planning and maintenance separately. B1 under 4096 tokens is a capped history control: it can overflow, and must not be described as unrestricted long context. An unrestricted-history diagnostic requires a separately declared budget condition.

The immutable base query and itinerary are supplied in the shared planner prompt. They are stored deterministically in ledger conditions, without a model extraction call, and excluded from lexical whole-plan duplication in the memory context. Explicit typed edges still retrieve required base slots/properties. History baselines store added-traveler traces without duplicating that fixed prompt. Missing properties remain unresolved while the exact submitted entity stays available for a legal tool query.

## Causal isolation and provenance

The chronological adapter releases one current traveler request, requiring a commitment before advancing. The planner runs in a fresh subprocess that receives only that request, the fixed public base itinerary, and bounded memory. It cannot read the hidden dataset, future-request directory, audit annotations, or run logs; its environment excludes API keys. The six official tools expose no arbitrary filesystem or code execution. The model client lives in the trusted coordinator.

Every condition can revise previously revealed added travelers. Each named section replaces that traveler's full submitted plan; omitted old slots are withdrawn, while an absent name preserves the previous submission. Base and future-name revisions are ignored and logged. The scorer receives the latest actual submissions even if memory validation rejects them. See `reports/recovery_protocol_audit.md` for the upstream contract and regression coverage.

This is an audited data-flow boundary for the supplied official code, not an operating-system sandbox for malicious Python. The trusted coordinator loads public requests chronologically and does not pass hidden data to memory. The official scorer is a separate process, invoked after planning; only it opens the hidden answer dataset and verifies its checksum. Memory is reset for each group and each treatment.

Pinned artifacts:

- MemoryArena: `6cd9de14b71915e39ac742a20dc33785e14b6aab`.
- Dataset: `da1a37c8b19280e18627ca01cf368195a5e1d92e`, 270 groups, 45 base-itinerary families.
- Splits: 8 development families, 7 validation families, 30 holdout families; 10 smoke groups drawn only from development.
- Official flight CSV: SHA-256 `8dafdb0e3f8b79ce599a1e612a772865295bc226b46e5fb278368f7255b11cee`.
- Qwen tokenizer: revision `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`; checksum recorded in `data/manifests/qwen_tokenizer.json`.

Polza exposes a provider-managed model alias. The model catalog, returned provider/model IDs, inference settings, and local source hashes are recorded per run. Routing requests allow only `chutes/fp8` with fallbacks disabled. Exact weight/runtime revision cannot be inferred from the alias. The paper used vLLM 0.20.0 on A100 GPUs; this implementation uses the public MemoryArena tool-calling planner. Matching the model name alone does not reproduce that setup.

Streamed responses identify the provider family as `Chutes`; earlier non-streamed responses returned a null provider field. That corroborates the family, while exact served weights/runtime remain provider-managed. The default planner keeps reasoning enabled with an 8,192-token completion cap; the E3F development overlay gives all compared conditions 16,384. The same Qwen model is used for schema-constrained extraction with reasoning disabled and a separately charged 16,384-token cap, shared across ledger conditions. Development traces showed that the previous writer could consume its entire allowance on reasoning or truncate the verbose JSON output.

## Config and outputs

Edit `configs/travel.yaml` for partitions and budgets. Credentials cannot be put in config. `--partition holdout` fails unless methodology is explicitly frozen in config. All treatment switches, parser settings, steps, tools, prompts, retries, and reasoning parameters are logged. The base config retains its original E0/E2 primary contrast; `configs/variants/E3F.yaml` explicitly declares E0/E3F on SR for current development. The ultimate comparator remains authentic MAGE under a shared protocol.

Each `runs/<id>/` contains the manifest, events, model usage, extractions, retrieval contexts, predictions, complete memory operations, snapshots, official scores, and reports. Replay needs no model calls and verifies memory snapshot hashes. Failed preflight/inference produces an explicit blocked/failed manifest; missing metrics never become zero. The scorer materializes every expected traveler, so missing submissions count as official failures.

The HTTPS transport warms a verified persistent connection without credentials before inference. It retries safe connection failures, but never automatically repeats a paid request whose delivery is ambiguous. Such failures retain unknown usage/billing and stop the run; they are not scored as model failures.

Streaming progress contains request IDs and numeric counts only, in `transport_events.jsonl`. Complete streams require final provider tokens, RUB cost, finish reasons, and `[DONE]`; incomplete streams never become partial predictions. Extraction requests a strict JSON schema and independently rejects malformed object structures. `reports/live_pilot_diagnostics.md` records the observations that prompted these changes.

Completed writer text is retained in `writer_responses.jsonl` for coordinator-side diagnostics, without injecting failed text into memory. `scripts/probe_writer.py` validates one development request before full planning. The development pilot stops on parser failure; this readiness gate is prohibited on validation and holdout, where actual model failures must remain in the evaluation.

`constraint_memory.reference.MageReferenceKernel` is a pure implementation of the paper's Grow/Compress/Maintain/Revise tree operations, with explicit implementation choices and injected callbacks. Its six deterministic tests verify prefix retention, exact boundary restoration, branch reuse, and preserved audit history. It is not a registered comparator, an author implementation, or an environment rollback mechanism. A matched runtime controller/read schedule is still required.

For this machine's intermittent catalog transfers, `scripts/run_first_pilot.py --catalog-snapshot reports/polza_catalog_verified_connection.json` can reuse the saved catalog. The run records its hash and timestamp, and still requires live preflight to return the exact configured model. This flag does not claim the catalog was freshly fetched.

`reports/development_memory_stress_audit.md` records the first real-trace counterexamples. Value aliases, city/property joins, and source-hash sanitization are now fixed. The original pilot is preserved. E3F is a separate runnable candidate; `reports/r9_grounded_frontier_diagnostic.md` audits its causal feasibility on recorded actions. Fresh live planning is required to test efficacy.

Prepare the next free-form diagnostic without model calls:

```powershell
uv run --no-sync python experiments/freeform_memory/run.py
```

This prepares ten distinct disclosed raw-language workflows and compares FACT_LEDGER with ADVISORY_IR over the same requirements and factual state. There is no artificial padding, no default monetary stop and no aggregate-token stop. The fixed cases allow at most 20 actor and 10 writer requests with declared per-call completion allowances. Add `--live` to execute the frozen model diagnostic; `reports/run_with_vpn_off.md` describes standalone execution. Optional `--budget-rub` is used only if explicitly requested. This remains development evaluation, with no authentic MAGE or broad held-out transfer claim.

`memory/inheritance.py` separately compiles exact unchanged-slot obligations from an explicitly declared TASK_CONTRACT and the public base. It never treats that policy as a quoted user request. Hard checks require independently verified request coverage; the current LLM writer does not establish that, so its interpretations yield advisory unresolved obligations. This module is not integrated into E3F, never edits actual submissions, and makes no task-success claim.

Run source can be preserved with `scripts/archive_run_source.py` and replayed with `scripts/replay_pinned.py`. R9 required explicit `--allow-redacted-pins` because its sanitizer hid the `secrets.py` hash. Its archive has checked integrity and 32 matching available source pins, while that missing original pin stays unverified. `--check-only` does not replay a live run; actual replay of this partial capture requires `--allow-partial-provenance` and never upgrades that limitation.

SR and PS are reported alongside upstream **SPS**. SPS is retained under its actual name, rather than equated to the paper's sPS. Reports distinguish official person-weighted PS from group-mean PS, pair identical group/replicate results, bootstrap by base-itinerary family, include exact McNemar tests, and account for latency, storage, tokens, and RUB billing. Secondary comparisons are exploratory. Scripted test scores are diagnostic only and are never evidence about Qwen or memory superiority.

## Research sources

- [MAGE paper, model and protocol](https://arxiv.org/html/2606.06090v1#S4.SS1).
- [Microsoft Research MAGE publication](https://www.microsoft.com/en-us/research/publication/beyond-semantic-organization-memory-as-execution-state-management-for-long-horizon-agents/).
- [Official MemoryArena source](https://github.com/ZexueHe/MemoryArena).
- [Official travel setup](https://github.com/ZexueHe/MemoryArena/blob/main/setup_travel.md).
- [Polza Qwen3.6-27B model](https://polza.ai/models/qwen/qwen3.6-27b).
- [Polza Chat Completions API](https://github.com/polza-ai/docs/blob/main/api-reference/chat/completions.mdx).

The supplied experimentation PDF is design guidance. Constraint correctness and real development failures drive the implementation choices; MAGE remains a comparison baseline, rather than the candidate architecture.

The E3F retrieval fallback now scopes literal person mentions by day and slot, independently of extracted edges. A single parsed relation cannot suppress another relation involving the same person. Possessive phrases use their referenced day/slot; ambiguous mentions retain exact actual entity choices. These are retrieval hints, not new inferred constraints or completeness proofs. The updated fixed-trace audit is `reports/r9_grounded_frontier_literal_fallback_diagnostic.json`; its maximum remains 3,794 tokens with zero overflows.

For running while Codex is disconnected, use `scripts/run_benchmark_suite.py`. It runs local verification, one bounded paired development group, automatic source archival, and offline pinned replay. See [VPN-off instructions](reports/run_with_vpn_off.md). The new default pair is E0/E3I; the candidate-specific source and test freeze must verify before any model request. A local suite lock prevents overlapping standalone jobs. Abrupt process or machine termination can still leave provider billing unresolved; the staged durable request journal is not integrated.
