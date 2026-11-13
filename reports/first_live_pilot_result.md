# First live development pilot: E0 versus E3

Run `qwen36-g6-E0-E3-20261014-r9` completed on 14 October 2026 using
Polza `qwen/qwen3.6-27b`, temperature 0, one development group (6), and six added
travelers. Limits were 30 planner steps per arriving traveler, 8,192 completion
tokens, and 4,096 memory-context tokens. This pilot establishes execution and
diagnostic evidence; it supports no superiority claim.

| Condition | Official group SR | Official PS | Official SPS | Agent tokens | Maintenance tokens | Total tokens | Provider RUB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E0 | 0% | 0% | 34.8291% | 181,817 | 34,109 | 215,926 | 22.15028434 |
| E3 | 0% | 0% | 33.3333% | 57,587 | 34,081 | 91,668 | 10.85454738 |

E3 minus E0 is 0 percentage points for SR/PS and -1.4957 points for SPS
(-4.29% relative). E3 used 57.55% fewer tokens and cost 51.00% less, largely
because four travelers never reached the planner. This is not evidence of an
efficient memory system achieving comparable task success. R9's total recorded
charge was **33.00483172 RUB**; earlier probes and aborted runs are separate.

## Independent verification

The archived scorer was rerun offline against the saved final predictions and
pinned hidden dataset, without model/API calls. Its global scores and per-group
records exactly matched both saved scorer outputs. Results, report totals, and
the usage ledger agree: E0 has 21 unique provider request IDs, E3 has 11, and no
missing token or RUB values. Deterministic memory/tool operations have explicitly
zero model tokens and provider charges. Their latencies are recorded separately;
request-latency sums are not end-to-end wall-clock time.

Both recorded memory replays match their snapshots. Independently checked
archive/extracted-file hashes match all 32 verifiable original source pins, and
the ZIP digest matches its archive manifest. The original `secrets.py` pin was
redacted, leaving **partial original-pin verification**; the archive's observed
hash does not establish that missing original pin. Provider-managed model
weights/runtime and the originally installed dependency environment are not
fully captured by these source checks.

Evidence: [saved results](C:/Users/Administrator/Desktop/MemoryResearch/runs/qwen36-g6-E0-E3-20261014-r9/results.jsonl),
[usage ledger](C:/Users/Administrator/Desktop/MemoryResearch/runs/qwen36-g6-E0-E3-20261014-r9/usage.jsonl),
[official E0 scores](C:/Users/Administrator/Desktop/MemoryResearch/runs/qwen36-g6-E0-E3-20261014-r9/official_scores_E0.json),
[official E3 scores](C:/Users/Administrator/Desktop/MemoryResearch/runs/qwen36-g6-E0-E3-20261014-r9/official_scores_E3.json),
[memory replay](C:/Users/Administrator/Desktop/MemoryResearch/runs/qwen36-g6-E0-E3-20261014-r9/replay.json), and
[source archive manifest](C:/Users/Administrator/Desktop/MemoryResearch/runs/qwen36-g6-E0-E3-20261014-r9/source_archive.json).

## Observed failure mechanisms

E3 refused Thomas, Brenda, Violet, and Marge with `BUDGET_OVERFLOW`: protected
contexts required respectively 4,229, 10,344, 11,888, and 6,903 pinned-tokenizer
tokens, exceeding 4,096. All four had zero planner steps and null final plans.
This is a concrete memory-budget limitation (X12), rather than a completed
planning failure or evidence that protected constraints were successfully used.

E0 reached Thomas and Brenda, but each final provider call had
`finish_reason=length` and 8,192 output tokens. Both produced no named final plan
and were scored as failures. They used only 2 and 3 of the available 30 planner
steps: the completion cap, rather than the step cap, was the recorded immediate
limit (X12, with missing output representation as a secondary consequence).

Even the early E3 plans failed full-person success despite SPS=1.0. Offline
slot checks identified changes to fields whose official gold value was **exactly
the fixed public-base value**:

| Condition / traveler | Slot | Public-base value | Actually submitted value |
| --- | --- | --- | --- |
| E3 Benjamin | Day 2 lunch | Barbeque Nation, San Antonio(Texas) | Burger Queen Drive In, San Antonio(Texas) |
| E3 Andrew | Day 1 lunch | Burger Queen Drive In, San Antonio(Texas) | Pita Pit, San Antonio(Texas) |
| E3 Andrew | Day 2 accommodation | Charming Bedroom in Gramercy, San Antonio(Texas) | Airy Brooklyn Limestone: 2 bedroom suite, San Antonio(Texas) |
| E0 Andrew | Day 2 lunch | Barbeque Nation, San Antonio(Texas) | Big Jack's, San Antonio(Texas) |
| E0 Andrew | Day 2 dinner | Pita Pit, San Antonio(Texas) | Minerva's Food & Cocktails, San Antonio(Texas) |

These slots are excluded from the scorer's gold-versus-base changed-slot set,
so SPS can be perfect while full-plan PS/SR remain zero. The worker supplies the
complete fixed public-base plan. The observed mechanism is unnecessary changes
to inherited choices, consistent with a planner/inheritance error (X8), rather
than missing access to the public base. E0 Benjamin additionally changed the
unchanged Day 2 lunch and Day 3 accommodation and missed one scored changed
breakfast slot. This does not establish that every required memory constraint
was correct, or explain every later non-null plan; those need a separate audit.
No gold facts or hand repairs were inserted into runtime memory.

Detailed evidence is retained in [planner traces](C:/Users/Administrator/Desktop/MemoryResearch/runs/qwen36-g6-E0-E3-20261014-r9/traces.jsonl)
and [memory retrievals](C:/Users/Administrator/Desktop/MemoryResearch/runs/qwen36-g6-E0-E3-20261014-r9/retrievals.jsonl).

## Metadata correction and limits

The frozen manifest declares E0/E3 with primary endpoint SR. The original
summary accidentally retained hardcoded E0/E2 primary metadata, while its
Markdown named E3/E0 and labeled every endpoint secondary. The original files
remain unchanged. A separate [corrected reanalysis](C:/Users/Administrator/Desktop/MemoryResearch/runs/qwen36-g6-E0-E3-20261014-r9/reports/corrected_primary_metadata/report.md)
reads the saved declaration, labels only SR primary, and preserves **all scores,
interval endpoints, paired differences, and resource values**. Its
[provenance](C:/Users/Administrator/Desktop/MemoryResearch/runs/qwen36-g6-E0-E3-20261014-r9/reports/corrected_primary_metadata/analysis_provenance.json)
records updated analysis hashes, before/after metadata, unchanged-original-file
hash checks, and zero new model calls. PS/SPS remain secondary and unadjusted.

There is one group and one base family. Resampling that single family 10,000
times produces identical observations; zero-width 95% bootstrap intervals
cannot estimate population uncertainty. Both conditions fail, giving zero
discordant pairs and exact McNemar p=1. The effective sample does not establish
equivalence, superiority, or generalization. The implementation status and
corrected reanalysis both retain `superiority_claim_supported=false`.

This was development, with no frozen holdout and no authentic MAGE comparison.
Post-pilot alias/city fixes and the separate E3F candidate are new source
conditions. Their diagnostic promise cannot be attributed retroactively to
this frozen E3 run. The next evidence must come from new paired development
runs, followed by a predeclared held-out comparison; primary SR remains the
target even when changed-slot SPS looks favorable.
