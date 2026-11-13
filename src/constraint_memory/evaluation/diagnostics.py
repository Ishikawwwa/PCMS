"""Offline memory-fidelity diagnostics against manually audited causal IDs.

The benchmark agent must never receive these audit sets. They score recorded
memory events after execution; they do not modify traces or insert oracle facts.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

ERROR_TAXONOMY = {
    "X1": "Constraint not extracted",
    "X2": "Dependency edge missing/wrong",
    "X3": "Correct memory, failed retrieval",
    "X4": "Outdated/contaminated retrieval",
    "X5": "Compression loss",
    "X6": "Incorrect entity/term grounding",
    "X7": "Inconsistent committed state",
    "X8": "Planner error with correct memory",
    "X9": "Search/tool coverage error",
    "X10": "Output formatting / evaluator mismatch",
    "X11": "Validation/rollback error",
    "X12": "Budget/timeout limitation",
}


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def extraction_metrics(expected_ids: Iterable[str], extracted_ids: Iterable[str]) -> dict[str, Any]:
    expected, actual = set(expected_ids), set(extracted_ids)
    correct = expected & actual
    precision, recall = _ratio(len(correct), len(actual)), _ratio(len(correct), len(expected))
    f1 = _ratio(2 * len(correct), len(expected) + len(actual))
    return {"precision": precision, "recall": recall, "f1": f1,
            "true_positive": len(correct), "false_positive": len(actual - expected),
            "false_negative": len(expected - actual), "expected_count": len(expected),
            "missing_ids": sorted(expected - actual), "unexpected_ids": sorted(actual - expected)}


def retrieval_metrics(
    required_causal_ids: Iterable[str],
    delivered_ids: Iterable[str],
    *,
    context_budget_tokens: int,
    delivered_tokens: int,
    dependency_chains: Mapping[str, Iterable[str]] | None = None,
    allowed_ids: Iterable[str] | None = None,
    stale_ids: Iterable[str] = (),
    wrong_version_ids: Iterable[str] = (),
) -> dict[str, Any]:
    """Measure what was actually delivered, including unresolved seed nodes.

    Each ``dependency_chains`` key is a seed constraint ID and its values are all
    required dependencies. The seed itself must also be delivered for that
    closure to count as complete. Required sets must include only causally
    available constraints, as annotated by the offline audit.
    """
    if context_budget_tokens < 0 or delivered_tokens < 0:
        raise ValueError("Token counts/budgets cannot be negative")
    if delivered_tokens > context_budget_tokens:
        raise ValueError("Recorded delivered context exceeds its declared budget")
    required, delivered = set(required_causal_ids), set(delivered_ids)
    stale = delivered & set(stale_ids)
    wrong_version = delivered & set(wrong_version_ids)
    valid_delivered = delivered - stale - wrong_version
    allowed = set(allowed_ids) if allowed_ids is not None else required
    irrelevant = delivered - allowed
    complete: list[str] = []
    incomplete: list[str] = []
    for seed, dependencies in (dependency_chains or {}).items():
        needed = {seed, *dependencies}
        (complete if needed <= valid_delivered else incomplete).append(seed)
    return {
        "required_constraint_recall_at_budget": _ratio(len(required & valid_delivered), len(required)),
        "context_budget_tokens": context_budget_tokens, "delivered_tokens": delivered_tokens,
        "required_count": len(required), "recovered_count": len(required & valid_delivered),
        "missing_required_ids": sorted(required - valid_delivered),
        "closure_completeness": _ratio(len(complete), len(complete) + len(incomplete)),
        "complete_seed_ids": sorted(complete), "incomplete_seed_ids": sorted(incomplete),
        "false_inclusion_rate": _ratio(len(irrelevant), len(delivered)),
        "false_inclusion_ids": sorted(irrelevant),
        "stale_reference_rate": _ratio(len(stale), len(delivered)), "stale_reference_ids": sorted(stale),
        "wrong_version_rate": _ratio(len(wrong_version), len(delivered)), "wrong_version_ids": sorted(wrong_version),
    }


def compression_metrics(active_requirement_ids: Iterable[str], query_accessible_ids_after: Iterable[str]) -> dict[str, Any]:
    active, accessible = set(active_requirement_ids), set(query_accessible_ids_after)
    lost = active - accessible
    return {"active_requirements": len(active), "lost_count": len(lost),
            "constraint_loss_rate": _ratio(len(lost), len(active)), "lost_ids": sorted(lost)}


def reliability_metrics(*, writes: int, parser_errors: int, contradictory_writes: int) -> dict[str, Any]:
    if any(value < 0 or not isinstance(value, int) for value in (writes, parser_errors, contradictory_writes)):
        raise ValueError("Reliability counts must be nonnegative integers")
    if parser_errors > writes or contradictory_writes > writes:
        raise ValueError("Error counts cannot exceed observed writes")
    return {"writes": writes, "parser_errors": parser_errors, "contradictory_writes": contradictory_writes,
            "parser_error_rate": _ratio(parser_errors, writes),
            "successful_schema_parsing_rate": _ratio(writes - parser_errors, writes),
            "contradiction_rate": _ratio(contradictory_writes, writes)}


def recovery_metrics(expected_invalidated_ids: Iterable[str], actual_invalidated_ids: Iterable[str]) -> dict[str, Any]:
    expected, actual = set(expected_invalidated_ids), set(actual_invalidated_ids)
    return {"correctly_isolated_ids": sorted(expected & actual),
            "false_rollback_ids": sorted(actual - expected),
            "missed_downstream_invalidation_ids": sorted(expected - actual),
            "isolation_precision": _ratio(len(expected & actual), len(actual)),
            "downstream_invalidation_recall": _ratio(len(expected & actual), len(expected))}


def validate_failure_report(case: Mapping[str, Any]) -> dict[str, Any]:
    """Require a trace-backed primary diagnosis, rather than guessing causality."""
    required = ("group_id", "traveler_index", "raw_user_request", "required_causal_source_spans",
                "extracted_constraints", "dependency_closure", "retrieved_prompt", "committed_plan",
                "tool_results", "scorer_outcome", "primary_cause", "diagnosis")
    missing = [field for field in required if field not in case]
    if missing:
        raise ValueError(f"Incomplete failure report: {missing}")
    causes = [case["primary_cause"], *case.get("secondary_causes", [])]
    if any(cause not in ERROR_TAXONOMY for cause in causes):
        raise ValueError("Failure causes must be one of X1 through X12")
    if not str(case["diagnosis"]).strip():
        raise ValueError("A trace-backed diagnosis is required")
    return dict(case)
