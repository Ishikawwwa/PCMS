# Per-concept memory on Polza

## Model and runtime

Every development model call used Polza `qwen/qwen3.6-27b`, with `chutes/fp8` requested and provider fallback disabled. The client checks the returned model ID and records prompt, completion, reasoning and total tokens when reported, plus provider billing and latency. Writes and routing use Qwen with reasoning disabled. The development decision actor uses the same model with reasoning enabled.

The memory writes from raw user messages and tool observations. It keeps the original text and exact supporting spans, indexes a record under all linked concepts, tracks aliases with evidence, and materializes scoped current and historical rows. Considered items and proposed changes have their own status. A writer validation failure permits one repair. If inference is interrupted, the pending event remains on disk and can resume before later observations are accepted. For shopping, budget arithmetic requires one explicit budget and a confirmed priced receipt for every prior slot. A read can follow either a recalled table reference or the model's direct scope resolution. The latter is measured independently as a flat-table control.

## Live development results

| Run | Outcome | Polza calls | Total tokens | Billed RUB |
| --- | --- | ---: | ---: | ---: |
| First shopping and travel pass | Shopping retrieved the predecessor, but older rows crowded out the balance. Travel stopped when the model emitted a confirmation and withdrawal for the same replacement. | 11 | 9,054 | 0.5976 |
| Task 24 with the exact released instruction | After prioritizing the predecessor and balance, all table modes delivered both in 158 tokens at a 192-token ceiling. Concept recall alone delivered the predecessor and omitted the computed balance. | 6 | 5,206 | 0.2885 |
| Travel after correcting replacement handling | The live actor selected the $75 accessible bus while the $100 increase was pending, and the faster $95 accessible rail offer after confirmation. All four reads succeeded at both cutoffs. The direct table used 73 and 87 memory tokens, versus 107 and 121 with concept recall. | 15 | 14,376 | 1.5373 |
| Fresh Task 24 pass with alias extraction and budget-row deduplication | All table modes delivered the three purchase receipts and the $34.74 balance in 186 tokens. Qwen wrote four source-linked aliases. | 6 | 7,689 | 0.4620 |

Repacking the same live Task 24 writes and route without repeating inference recovered **all three purchase receipts and the $34.74 balance in 186 tokens**. The change drops the separate budget row when the derived balance is present and cites the budget together with the purchase receipts. The earlier 158-token block omitted Product 2. The considered Mermaid cake cup never contributed to spend.

Task 24 uses the [exact released Product 4 instruction](https://huggingface.co/datasets/ZexueHe/memoryarena/blob/da1a37c8b19280e18627ca01cf368195a5e1d92e/bundled_shopping/data.jsonl) and a reconstruction from the [published MAGE summaries](https://arxiv.org/pdf/2606.06090v1#page=15). The reconstruction includes the cross-system distractor disclosed in the first iteration. This is a live retrieval result from a partial trace. The travel offers above are a disclosed development fixture. They test time-sensitive decisions with actual inference.

## Direct comparison

The paired runner now compares official MAGE with a standalone PCMS agent. Both use the same released travel environment and domain tools, Polza configuration, and official evaluator. PCMS controls its own actions and retrieves across travelers from its evidence journal. MAGE uses its official tree, verification, and replay. The configured action ceiling is per attempt, so report actual calls and total tokens alongside success. The default harness uses official MAGE first. `--first-arm pcms` reverses order for a development check.

## Earlier augmentation pilot

The earlier harness added a source-linked memory block at traveler start while keeping the released MAGE action loop, verification, and restoration in both arms. Those logs measure augmentation. The console labels them as augmentation pilots. [Released MAGE source](https://github.com/Komorebi660/MAGE/tree/ace693139dc43e8bdb3076da4127483b41a5060f)

Run manifests distinguish shortened development prefixes from full tasks. The full-length claims require paired, complete development and held-out bundles with costs reported for all memory and actor calls.

Two author-runtime runs were started. Group 6 uses a two-traveler prefix, a 40-step limit, and actor reasoning enabled. Group 152 uses a two-traveler prefix, a 12-step limit, and actor reasoning disabled in both arms for faster development feedback. Group 152 was selected for its short opening queries. The author verifier in its baseline retried a selection after giving feedback that also described the selection as satisfying its stated cuisine and rating constraints. The released verifier behavior was retained in both arms.

## Reproduce

```powershell
.\.venv\Scripts\python.exe -m experiments.concept_subspaces.run_live --case shopping --shopping-release tmp/concept-subspaces-shopping-release.jsonl --out runs/concept-subspaces-shopping-repeat
.\.venv\Scripts\python.exe -m experiments.concept_subspaces.run_live --case travel --out runs/concept-subspaces-travel-repeat
.\.venv\Scripts\python.exe -X utf8 -m experiments.concept_subspaces.run_mage_travel --travel-release data/hidden/memoryarena_travel.jsonl --train-csv tmp/concept-subspaces-travel-train.csv --group 6 --turns 2 --out runs/concept-subspaces-mage-repeat
```
