The adapter uses official MemoryArena commit `6cd9de14b71915e39ac742a20dc33785e14b6aab`.
`scripts/prepare_memoryarena.py` downloads the exact pinned public dataset, official
Drive flight CSV, and Qwen3.6-27B tokenizer; it records source and database hashes.
The 270 groups cluster into 45 identical fixed-base-itinerary families. Development,
validation, and holdout contain 8, 7, and 30 disjoint families (48, 42, and 180 groups).
The ten smoke groups belong exclusively to development.

Use a fresh `ChronologicalAdapter(root, group_id)` and reset the memory backend at
every group. Each `iter_requests()` yield requires `commit_result()` before the next
request can be read. Both the fixed public base and each current request have
immutable checksums. The adapter does not load answer keys.

`PlannerWorker(...).run_turn(request, memory_context, base_person)` starts a fresh
process for one traveler. That process receives the fixed public base, current
request, and bounded memory text. It runs the unchanged official
`TravelPlannerAgent.run_single_person` loop with native tool schemas and local CSV
tools. Its only model connection is JSON RPC to the parent-owned Polza client.
Credentials are not inherited. Hidden/future dataset paths and evaluator imports
are denied by an audit hook. The source imports avoid broad upstream package
initializers, which would otherwise import unrelated agents or dataset loaders.
This visibility boundary assumes audited trusted Python code; it is not an
adversarial operating-system sandbox. No official planner tool accepts arbitrary
filesystem operations or code execution.

Every treatment receives the same fixed-base prompt. Prior added-traveler plans
enter only through its memory backend: automatic accumulated-history prompts and
step-memory reads are disabled. All evaluator feedback is absent. This is the
official-public-source fallback protocol, with the official agent, tools, prompt,
parser, step semantics, and scorer. It does not reproduce MAGE's custom controller
(public implementation not located) or provider-independent vLLM inference stack.

The returned `submitted_text` and `plan` describe the target traveler;
`accepted_submissions` additionally contains every emitted section for previously
revealed added travelers. The group submission ledger keeps the latest full named
section, including explicit empty/malformed replacements and omitted slots. Base
and future-name revisions are ignored and logged consistently across treatments.
Every accepted section is parsed with official `combination.parse_plan_text`.
`structured_tool_results`
contain the exact rows and columns already returned by an official tool, aligned
with scratchpad tool-call IDs and the unchanged formatted observation. Capturing
them issues no additional search or database lookup.

The paper's Appendix A describes per-action retrieval and stateful booking/session
tools that are absent from this public CSV planner. The current protocol is a
declared local experimental harness; its scores cannot reproduce or directly
beat published MAGE scores. See `reports/mage_protocol_audit.md`.

`score_official(upstream_root, dataset_path, predictions, group_ids,
expected_dataset_sha=...)` sends hidden-data access to a separate scorer process.
It imports the official scorer without the planner, verifies the hidden checksum,
and disables unpinned Hugging Face loading. Every expected traveler is materialized
in the submission; missing/unparsed plans remain null and fail. This closes the
upstream runner's missing-member omission without changing its scoring functions.

Official metrics retain their actual names:

- SR: groups where every expected added traveler passes all evaluated slots/days.
- PS: global person-weighted full-plan pass rate. `group_mean_PS` is separately
  reported for paired group analyses.
- SPS: average over groups of average over travelers with nonempty gold-versus-base
  changed-slot sets of the fraction of changed slots passed. It evaluates six
  slots, excluding `current_city`, using official similarity threshold 0.7.
  Zero-change travelers are excluded from SPS. Equivalence to the paper's sPS has
  not been established, so the adapter never silently relabels it.

Adapter tests exercise real official CSV tools and parser using a scripted model
client, causal read/import rejection, split isolation, and exact official scoring
on synthetic hidden fixtures. These are harness tests, not model-performance results.
