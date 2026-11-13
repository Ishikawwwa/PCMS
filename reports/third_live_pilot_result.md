# Verified E0 / E3I development result

Run: `qwen36-E0-E3I-vpn-off-20261014T135633482517Z`.
Both variants completed on one repeatedly inspected development family, using
Qwen3.6-27B through Polza with matched budgets and a shared declared inheritance
contract. Archived-source rescoring and memory replay reproduce both results.

| Metric | E0 | E3I |
|---|---:|---:|
| Complete-group success (SR) | 0 | 0 |
| Full-plan passes (PS) | 1/6 | 2/6 |
| Changed-slot score (SPS) | 0.3935185 | 0.7510684 |
| Total agent + maintenance tokens | 220075 | 215975 |
| Main-run billed RUB | 25.59472332 | 24.47727407 |

Total billing including preflight was 50.10287685 RUB. All 40 archived source
pins, 28 upstream hashes and 47 request identities verified. Contexts fit the
4096-token cap and all 34 planner dispatches respected the common step budget.
Original artifacts were not edited.

E3I's remaining errors include a truncated observed hotel name, failed searches
using annotated city names, choices violating available numeric bounds, and
punctuation-only inheritance repair. Incorrect prior actual choices must stay
actual history; they cannot be repaired by inserting offline gold choices.

E3R is a new, unmeasured development candidate for individually verified request
predicates, observed full-name checks, freshly revalidated repair cards and a
shared city/schema protocol. It is not evidence of a score improvement. The
portable constraint API and separate transfer diagnostic address generality;
see `generalization_contract.md` for their evidence limits and promotion rule.

Independent verification: `E3I_live_verification/verification.md`.
Failure evidence: `E3I_live_failure_audit/audit.md`.
No authentic matched MAGE result or broad model-transfer result is available.
