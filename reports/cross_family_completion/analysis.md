# Independent completed-run analysis: cross-family-continuation-20261017T134637385730Z

E2 did not outperform E1 in this seven-family development mechanism screen. Both scored 0/7 on the primary group success endpoint. E1 passed 18/49 traveler plans and E2 passed 16/49. Every saved official group and traveler metric matched a fresh run of the pinned official scorer.

All 14 sessions and 98 added-traveler commitments are present. The run and its archived native/controller source were not modified by this audit. No model or network requests were made; scoring used the already-pinned local answer data after planning.

| Metric | E1 | E2 | E2 minus E1 (pp) | 95% family bootstrap interval (pp) | E2 family win/tie/loss |
|---|---:|---:|---:|---:|---:|
| SR | 0.000% | 0.000% | +0.000 | [+0.000, +0.000] | 0/7/0 |
| PS | 36.735% | 32.653% | -4.082 | [-14.286, +6.122] | 2/2/3 |
| SPS | 73.956% | 71.098% | -2.858 | [-18.152, +12.757] | 2/0/5 |

SR requires every expected traveler to pass every evaluated day/slot. PS is full-plan traveler pass rate. SPS averages correctness on gold-versus-base changed slots; equivalence to the MAGE paper's sPS is unverified. Only SR was predeclared primary; PS and SPS are secondary and unadjusted. The zero SR interval is a floor result, not evidence of equivalence. Seven original families, with one group and one seed each, are the inference units; travelers are repeated observations.

| Group | E1 passed / 7 | E2 passed / 7 | PS delta (pp) | E1 SPS | E2 SPS | SPS delta (pp) |
|---|---:|---:|---:|---:|---:|---:|
| 120 | 1 | 2 | +14.29 | 70.04% | 67.05% | -3.00 |
| 197 | 6 | 5 | -14.29 | 97.62% | 80.95% | -16.67 |
| 254 | 0 | 0 | +0.00 | 42.66% | 72.68% | +30.02 |
| 26 | 2 | 2 | +0.00 | 69.25% | 69.11% | -0.15 |
| 64 | 4 | 5 | +14.29 | 57.14% | 76.19% | +19.05 |
| 69 | 3 | 1 | -28.57 | 87.93% | 74.49% | -13.44 |
| 89 | 2 | 1 | -14.29 | 93.06% | 57.23% | -35.83 |

The JSON contains all 98 traveler-level hard-pass decisions, changed-slot numerator/denominator, memory status and charged actor steps. Fresh scoring was also checked against those saved fields, including the zero-based result versus one-based scorer indexing.

## Context pressure and mechanism

| Quantity | E1 | E2 |
|---|---:|---:|
| Arrival reads / base reads | 49 / 7 | 49 / 7 |
| Arrival PARTIAL_CONTEXT | 15 | 16 |
| Overflow / skipped arrivals | 0 / 0 | 0 / 0 |
| Mean / maximum memory tokens | 23185 / 32765 | 21211 / 32760 |
| Arrival reads at ≥95% of memory cap | 16 | 16 |
| Omitted record retrieval occurrences | 3073 | 2479 |
| Distinct omitted group/record references | 1750 | 1651 |
| Native required / returned record occurrences | 13656 / 10583 | 8988 / 6509 |
| Missing-reference arrivals / occurrences | 0 / 0 | 6 / 29 |
| Returned records with resolved target | 0 | 449 |

The shared 32,768-token cap is serialized memory, not the whole actor prompt. Native pinned tokenizer counts stayed within that cap. Both arms experienced lossy retrieval truncation; no arrival was skipped and neither used the protected overflow branch. Omission occurrences can count the same record on several reads. No destructive record deletion was enabled. Each policy defines its own required-record set, so its recorded recall is not gold semantic recall and has no common cross-arm denominator.

E2 enables dependency closure within the same typed ledger. The integrated policy also changes named-traveler seed suppression, live dependency binding, missing-property entity fallback, selected outcomes, and which records survive truncation. Glossary hopping, protected compression, deterministic validation and targeted repair are disabled in both arms. The run therefore tests this bundled policy, rather than graph traversal alone. The observations cannot attribute the accuracy difference to capacity or to one closure submechanism.

## Submission and resource effects

Native named-submission success was 42/49 for E1 and 45/49 for E2; this is distinct from official correctness (18/49 and 16/49). All 11 missing named submissions coincide with finish_reason=length and 16,384 output tokens entirely reported as reasoning. Observed actor step counts are 2–4, so the 30-request actor allowance was not exhausted. This association identifies a separate completion-budget problem; it does not establish a memory failure cause.

| Accepted-call resource | E1 | E2 |
|---|---:|---:|
| Writer / actor responses | 49 / 106 | 49 / 103 |
| Known input tokens | 3042676 | 2807545 |
| Known output tokens | 787985 | 711346 |
| Known receipt cost (RUB) | 266.08740784 | 247.06082498 |
| Maximum actor prompt tokens | 44596 | 45159 |

Across the lineage there are 307 unique accepted responses (98 writer, 209 actor), four sent failures with unknown bills, and one known not-sent preflight failure at zero cost. Known accepted receipt cost is 513.14823282 RUB; the complete total remains unknown. Reused accepted responses were retained without another purchase. Mixed interrupted/reconstructed session timings do not support a clean latency comparison.

The run is explicitly an infrastructure-recovered diagnostic. Three sent actor failures consume their original per-arrival actor budgets. One sent writer failure required a physical writer allowance of 99 rather than 98 (total bound 3,039 rather than 3,038); exactly one accepted writer output per arrival was retained. It must not be represented as satisfying the original physical dispatch budget.

## Causal scope and next experiment

All 98 saved worker audits report a fresh current-query-only process, earlier plans available through memory, no judge feedback, no hidden scorer/data-loader modules and no credentials. All accepted writer payloads contain only the current raw request event and the causal prior-traveler frontier. The pinned runner gates official scoring after all 14 sessions and the isolated scorer reports no planner module loaded. This source/log audit supports causal separation; it is not an independent operating-system penetration test.

This development sample uses shared declared task/tool contracts and a local typed-ledger comparison. It is not authentic MAGE, has no held-out result, uses one seed, and cannot establish general memory transfer or same-model MAGE superiority. Provider-managed runtime identity, sequential execution over several days, and transport changes remain limits even though model/routing settings were pinned.

Freeze this negative result. Audit source-grounded constraint fidelity, correct binding/entity evidence, and output completion separately before changing memory policy. Use independent multi-action development workflows to examine format-independent behavior; any newly added dataset families must be explicitly designated for development, without taking iteration data from validation or holdout. All eight existing development families have already been inspected, so a capacity/completion-budget factorial within that partition could diagnose pressure and output truncation but cannot provide fresh-family confirmation. Such a diagnostic should predeclare settings, freeze adapters, include a sufficient-memory condition and independent seeds; it is an optional next experiment, not a prerequisite to preserving this result. Freeze any selected portable change before authentic same-model MAGE and held-out/transfer comparisons. Do not select per-family retrieval rules or tune against these saved answers.

Reproduce with ` .\.venv\Scripts\python.exe .\reports\cross_family_completion\analysis.py `. Input hashes, native source pins, all traveler scores, per-arrival pressure, bootstrap output and integrity checks are in analysis.json. The script writes only its two sibling artifacts.
