# Partial portable-transfer verification

Audited run: `portable-transfer-20261014T191914Z`. **PASS_WITH_SCOPE_LIMITATIONS**: no audit consistency failures. This is an archived, terminal `failed` run with a recorded monetary-cap stop. No network/model requests or credential reads were made; no original artifact was modified. Present process state was not inspected or inferred.

## What is saved

There are 59 result records, 59 committed responses and 59 completed usage records with identical ordered case/condition keys and unique request IDs. Every committed response text, embedded usage record, model ID and request ID matches its result/usage counterpart. All 286 transport events map to those same 59 request IDs and agree on model/provider identity.

There are 60 request-intent records: the unmatched final intent is `inventory-short-messages:HISTORY`. The archived runner appends an intent before calling the client, whose budget check precedes its HTTP request. The missing usage/transport/response and exact cap crossing support pre-network budget rejection for this intent; the intent is not evidence of an extra completion or extra billed request. There are 13 uncompleted conditions total: that intent plus 12 never reached conditions (inventory pressure/infeasible, both conditions across three formats). These are missing observations, not failed model answers.

## Hashes and offline replay

All 105 archived members match the freeze: 48 source files, 47 tests and 10 other inputs. The archive SHA, source hash map and runner SHA match the run manifest. The **original archived task SHA** is `9c03964ed67a67228bfbe6eb62825893c15eec3cc65f763f47c83ab2192a3f04`, matching the original task anchor in both freeze and manifest. The run's `task_inputs.json` SHA is `d27d98438ce9d14362c34a61b83186874996589b0a0412da20f99d8656b02db5`. Its parsed contents are identical; its 600,000 bytes differ from the original 600,002 bytes **only by removal of the final CRLF**, consistent with the archived writer. Current corresponding files also match every freeze hash.

The freeze report itself was not archived/pinned by the run. Test/input digest comparison therefore uses the current external freeze; source/script/task/ZIP also have run-manifest anchors. The claimed earlier test counts are metadata, not a rerun performed by this audit.

The verifier imports scoring code directly from the verified ZIP. **All 59 entire saved score records and all 59 entire result records reproduce exactly**, including checks, statuses, bindings, versions, source hashes, contexts and messages. All 60 request messages reconstruct; all 72 planned contexts recompute using the local tokenizer whose SHA matches the archived tokenizer pin (the tokenizer bytes were not archived).

A separate `direct_oracle.py` uses four explicit task formulas with `Fraction` arithmetic, set membership/inclusion and string equality. It imports no memory/kernel code. It checks 36 tasks, 108 offered options, and reproduces **all 59 entire score records**, including exact saved check provenance, not just aggregate pass totals. `direct_oracle.json` contains the per-option feasibility and per-result comparisons.

## Cost and identity

Exact Decimal ledger sum: **20.12386276 RUB**, using the saved JSON numeric lexemes. The first 58 completions cost 19.79230304 RUB; completion 59 cost 0.33155972 RUB, crossing the effective 20 RUB client cap by 0.12386276 RUB. The archived cap check runs before a request and permits this single-request overshoot. The final logged intent then hits the monetary check. Total usage is 91,166 input + 73,072 output = 164,238 tokens (72,110 reasoning tokens included in output). Total plus the next conservative reserve is 174,946, within the effective 200,000 token cap.

**Configuration provenance caveat:** the manifest's agent config snapshot says 100 RUB and 1,500,000 tokens per group; the archived runner actually constructs the client with **20 RUB and 200,000 tokens**. The observed stop matches the runner values. Saved costs are provider-reported ledger evidence, not independently reconciled invoices or price estimates.

Requested, committed and returned model IDs all equal `qwen/qwen3.6-27b`; all usage records report accepted exact matches. Routing requests consistently specify only `chutes/fp8` with fallbacks disabled; returned provider labels all say `Chutes`. These labels are consistent, but do not independently establish fp8 precision, server weight/revision or authenticity of provider reports.

## Actual observations and repeated units

| Observation set | HISTORY | CONSTRAINT |
|---|---:|---:|
| All saved conditions | 20/29 | 30/30 |
| 29 complete case pairs only | 20/29 | 29/29 |
| One unpaired case, inventory-short-messages | missing | 1/1 |

Among the 29 complete pairs, 20 have both conditions pass and nine have CONSTRAINT pass/HISTORY fail; no reverse discordance was observed. Every HISTORY failure is an abstention on a feasible pressure task; all nine pressure histories omitted initial requirements, the policy and revision. The nine rows comprise **three domain/load templates repeated in three formats**, not nine independent new domains.

The planned 36 case rows are four domain templates × three load variants × three serialization/order variants. There are 12 domain/load templates (10 have any completion); ignoring format and irrelevant pressure padding leaves eight distinct domain decision scenarios. Short and pressure reuse decision facts/options. Complete-pair coverage is nine full format triplets plus two inventory-short formats. No uncertainty interval or significance claim that treats these repeats as independent is made.

| Domain | Load | HISTORY | CONSTRAINT | Complete case pairs |
|---|---|---:|---:|---:|
| deployment | short | 3/3 | 3/3 | 3 |
| deployment | pressure | 0/3 | 3/3 | 3 |
| deployment | infeasible | 3/3 | 3/3 | 3 |
| inventory | short | 2/2 | 3/3 | 2 |
| inventory | pressure | 0/0 | 0/0 | 0 |
| inventory | infeasible | 0/0 | 0/0 | 0 |
| procurement | short | 3/3 | 3/3 | 3 |
| procurement | pressure | 0/3 | 3/3 | 3 |
| procurement | infeasible | 3/3 | 3/3 | 3 |
| scheduling | short | 3/3 | 3/3 | 3 |
| scheduling | pressure | 0/3 | 3/3 | 3 |
| scheduling | infeasible | 3/3 | 3/3 | 3 |

## Scope and reproduction

This verifies consistency of a partial, structured synthetic development diagnostic. It establishes no free-form extraction result, cross-model transfer, held-out-domain generalization, completed 72-condition score or MAGE comparison. The mechanism receives explicit observed fact/property bindings and JSON contracts. Missing conditions remain missing.

Run from the repository root using the existing offline environment:

```powershell
.venv/Scripts/python.exe -I -B reports/portable_transfer_partial_verification/verify.py
.venv/Scripts/python.exe -I -B reports/portable_transfer_partial_verification/direct_oracle.py
```

The main verifier has a Python audit hook blocking network activity, subprocess execution, credential-file reads and writes outside this report directory. Its only nonstandard dependency is the already installed `tokenizers` package for exact context reconstruction. `verification.json` contains the complete file hash inventory, accounting prefixes, missing keys and archived replay records.
