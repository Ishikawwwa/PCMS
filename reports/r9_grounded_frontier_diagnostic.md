# R9 causal context-feasibility diagnostic

The actual E3F development candidate fit every arrival of the recorded E0 trace
within 4096 Qwen3.6-27B tokenizer tokens. Its largest required context was 3794
tokens. This measures context feasibility on fixed actions, **not planner efficacy
or a counterfactual benchmark score**.

| Arrival | Archived E3 tokens | Alias/city-fixed E3 tokens | E3F tokens | E3F status | E3F missing references |
|---|---:|---:|---:|---|---:|
| Benjamin | 3241 | 3241 | 3281 | UNRESOLVED | 0 |
| Andrew | 2242 | 2769 | 1332 | OK | 0 |
| Thomas | 4229 | 5756 | 3794 | UNRESOLVED | 2 |
| Brenda | 10343 | 22190 | 3316 | UNRESOLVED | 16 |
| Violet | 11887 | 30861 | 1313 | UNRESOLVED | 9 |
| Marge | 7774 | 15354 | 1918 | UNRESOLVED | 5 |

Numbers are the required complete contexts, rather than the small overflow notice
returned when a condition fails. Archived E3 and corrected E3 each overflowed on
four arrivals. E3F had no overflows. Remaining unresolved references and writer
uncertainty stay explicit; fitting the cap does not make a requirement verified.

The replay feeds all conditions the same accepted actions, observations, and
writer extractions from the completed R9 E0 run. Tool rows activate only when the
corresponding recorded `observe` operation is reached. Structured observations
must match both the scratchpad's literal text and its SHA-256, and a genuine
tool-result event must precede the same commit boundary. No database, hidden
answer, or future observation is supplied to a memory state.

Each commit mirrors `GroupSession`: current choices replace only accepted full
traveler sections; earlier valid choices and public-base choices remain available
for grounding; changed or newly observed attributes are inserted while unchanged
prior facts are deduplicated. Both corrected E3 and E3F insert 312 property
records, including 66 on earlier/public-base choices. Archived E3 inserts 15,
including three earlier/base flight prices. The increase is real observed
evidence recovered by the city fix, so correcting grounding increases the memory
load rather than merely reducing diagnostic text.

The source archive is verified before its four pure memory/extraction modules
are loaded under private names. Current modules are separately captured under
private names, including the actual `FrontierMemory` class. No imported runtime
global is replaced, and source changes during the audit cause failure. Original
R9 source provenance remains **partial** for the redacted `secrets.py` pin; that
file is not executed by this diagnostic. The loaded archived module pins match.
Exact log, module, tokenizer, archive, and script hashes are in the JSON report.

The E3F retrieval policy stops at submitted choices and property/entity proofs,
retains full archive dependencies for invalidation, omits duplicate raw outcome
text, and filters only complete known-name companion-list lines from lexical
retrieval. Literal names in other request lines retain fallback. Current binding
requirements remain visible; historical requirements that led to an actual past
choice are archived rather than recursively copied into the planner context.

The key limitation is deliberate: when archived/fixed E3 overflows, this
diagnostic still feeds the later E0 actions. A real runtime would block planner
dispatch on those failures. The alternate actions, extraction responses, costs,
and scores remain unobserved. Old versus corrected E3 also combines reference
normalization with improved grounding, and E3F adds a separate retrieval policy;
these are not clean one-mechanism ablations. A new live matched development run
is needed before making a performance claim.

Four synthetic causal checks additionally passed: adding an observed property to
an earlier public-base choice, preserving same-value deduplication, rejecting a
tool row at/after the commit boundary, and rejecting a choice absent from the
recorded accepted action.

Artifacts:

- `scripts/audit_grounded_frontier.py`
- `reports/r9_grounded_frontier_diagnostic.json`

Reproduce with:

```powershell
python scripts/audit_grounded_frontier.py runs/qwen36-g6-E0-E3-20261014-r9
```
