# MAGE protocol and reimplementation audit

Audit: 14 October 2026. This is a primary-source/code assessment, not a measured
MAGE result. PDF page numbers below are one-based. No runtime source,
configuration, credentials, or running pilot was changed.

A local paper reimplementation can be a useful declared experimental baseline.
It cannot presently establish reproduction of the authors' implementation or
make the published MAGE score a matched control. Author source has not been
located in the primary sources reviewed, and the paper describes materially
different planning affordances.

| Evidence | Consequence for this repository |
| --- | --- |
| PDF pp. 5–7, sections 3.2–3.3 and Algorithm 1: raw and summary trees; context consists of active-path summaries, recent raw trace, and branch hints. Grow merges identical siblings; Compress summarizes a contiguous covered segment; Maintain judges the segment and summary; Revise removes the failed summary from the active path and resumes before it. | These mechanisms are sufficiently explicit to implement and test. They are not semantic retrieval or merely a traveler-summary list. [Method and algorithm](https://arxiv.org/html/2606.06090v1#S3). |
| PDF p. 13, Appendix A.2: baseline storage starts empty per task; full traces are written after subtasks; retrieval occurs before each action using the current trace. | Session-start retrieval alone is a different operating condition. [PDF Appendix A.2](https://arxiv.org/pdf/2606.06090v1#page=13). |
| PDF pp. 12–13, Appendix A.1: travel uses returned option IDs, select/book operations, session IDs, cost/itinerary snapshots, and Finish[done]. | Public CSV-search-plus-final-text planning does not implement that stateful environment. The HTML view omits these boxed prompts; consult the PDF. [Travel prompt](https://arxiv.org/pdf/2606.06090v1#page=12). |
| PDF p. 7, section 4.1: Qwen3.6-27B/ReAct; A100, vLLM 0.20.0, Python 3.12. | The exact Polza model alias matches the backbone name, while provider precision, serving runtime, and model revision remain distinct. [Experimental setup](https://arxiv.org/html/2606.06090v1#S4.SS1). |

Implementation choices still requiring explicit declarations include root sentinels,
node identity versus boundary-step IDs, content-equality normalization, compression
thresholds, summary/judge prompts, hint packing, judge-output handling, stopping
limits, and restoration of external planning state. In particular, memory-pointer
rollback must not silently retract actual submitted plans. The existing submission
ledger and scoped revision protocol remain authoritative in the public adapter.

The next controlled iteration should freeze the completed pilot, then introduce a
separate **MAGE-paper-inspired/public-adapter** condition. Implement the four
operations and active-path state, with charged summarization/judging and explicit
bounded revision. Add live action-level observation/read hooks as a separately
declared factor shared with comparison baselines. Freeze every unspecified choice
before validation/holdout scoring, and test branch exclusion, correct restoration,
deduplication, and retention of valid prefix state. Report results as a comparison
with this local baseline; do not claim an authentic MAGE victory.

A stronger **MAGE paper reimplementation** label additionally requires recreating
the paper's stateful travel environment and action interface, then running all
compared memories under that same protocol. Authentic reproduction requires a
reviewed author implementation/configuration, or author-confirmed equivalence;
source availability and matched protocol should be demonstrated rather than
assumed.

The adapter README now describes `accepted_submissions` for visible added
travelers and the official cumulative replacement behavior. Its session-start
protocol remains different from the paper's per-action retrieval. The supported
source-availability finding is that the reviewed primary sources did not identify
public implementation code.
