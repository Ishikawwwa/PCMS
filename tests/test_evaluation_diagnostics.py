import pytest

from constraint_memory.evaluation import (
    aggregate_ledger, compression_metrics, extraction_metrics, recovery_metrics,
    reliability_metrics, retrieval_metrics, summarize_resources,
)


def test_extraction_precision_recall_counts_and_empty_denominators():
    result = extraction_metrics(["a", "b"], ["a", "c"])
    assert result["precision"] == result["recall"] == result["f1"] == .5
    assert result["missing_ids"] == ["b"]
    assert extraction_metrics(["a"], [])["f1"] == 0
    assert extraction_metrics([], [])["recall"] is None


def test_retrieval_requires_complete_delivered_nonstale_chain():
    result = retrieval_metrics(["a", "b", "c"], ["a", "b", "c", "junk"],
        context_budget_tokens=100, delivered_tokens=80,
        dependency_chains={"a": ["b", "c"]}, stale_ids=["c"])
    assert result["required_constraint_recall_at_budget"] == pytest.approx(2 / 3)
    assert result["closure_completeness"] == 0
    assert result["false_inclusion_rate"] == .25
    assert result["stale_reference_rate"] == .25
    with pytest.raises(ValueError, match="exceeds"):
        retrieval_metrics(["a"], ["a"], context_budget_tokens=10, delivered_tokens=11)


def test_closure_dependencies_without_seed_do_not_count_as_complete():
    result = retrieval_metrics(["seed", "prior"], ["prior"], context_budget_tokens=10,
        delivered_tokens=5, dependency_chains={"seed": ["prior"]})
    assert result["closure_completeness"] == 0
    assert retrieval_metrics([], [], context_budget_tokens=0, delivered_tokens=0)["closure_completeness"] is None


def test_compression_and_scoped_recovery_quantify_missing_and_false_rollbacks():
    assert compression_metrics(["a", "b"], ["b", "other"])["constraint_loss_rate"] == .5
    result = recovery_metrics(["failed", "dependent"], ["failed", "unrelated"])
    assert result["false_rollback_ids"] == ["unrelated"]
    assert result["missed_downstream_invalidation_ids"] == ["dependent"]
    assert result["isolation_precision"] == result["downstream_invalidation_recall"] == .5


def test_reliability_does_not_turn_no_observations_into_perfect_parsing():
    assert reliability_metrics(writes=0, parser_errors=0, contradictory_writes=0)["successful_schema_parsing_rate"] is None
    assert reliability_metrics(writes=100, parser_errors=2, contradictory_writes=1)["successful_schema_parsing_rate"] == .98
    with pytest.raises(ValueError):
        reliability_metrics(writes=1, parser_errors=2, contradictory_writes=0)


def test_ledger_separates_all_maintenance_calls_and_provider_rubles():
    ledger = [
        {"operation": "agent", "prompt_tokens": 10, "completion_tokens": 5, "cost_rub": .3, "latency_seconds": 2},
        {"operation": "extraction", "input_tokens": 20, "output_tokens": 3, "cost_rub": .4, "latency_seconds": 1},
        {"operation": "glossary", "input_tokens": 7, "output_tokens": 4, "cost_rub": .2, "latency_seconds": 3},
        {"operation": "flight_search", "category": "tool", "input_tokens": 0, "output_tokens": 0, "cost_rub": 0, "latency_seconds": .5},
    ]
    result = aggregate_ledger(ledger)
    assert result["categories"]["agent"]["total_tokens"] == 15
    assert result["categories"]["maintenance"]["total_tokens"] == 34
    assert result["total"]["cost_rub"] == pytest.approx(.9)
    assert result["external_tools"]["latency_seconds"] == .5
    assert result["operations"]["glossary"]["calls"] == 1


def test_unknown_billing_and_missing_tokens_remain_null_not_zero():
    result = aggregate_ledger([
        {"operation": "agent", "input_tokens": 3, "output_tokens": 2, "cost_rub": .2},
        {"operation": "validation", "input_tokens": None, "output_tokens": 4},
    ])
    assert result["total"]["total_tokens"] is None
    assert result["total"]["known_total_tokens"] == 5
    assert result["total"]["cost_rub"] is None
    assert result["total"]["known_cost_rub"] == .2
    assert result["total"]["missing_cost_rub_calls"] == 1
    assert aggregate_ledger([])["total"]["total_tokens"] is None


def test_wall_clock_is_separate_from_summed_request_latency():
    result = summarize_resources([{"wall_clock_seconds": 1, "usage": [
        {"operation": "agent", "latency_seconds": 1},
        {"operation": "extraction", "latency_seconds": 1}]}])
    assert result["wall_clock_seconds"]["total"] == 1
    assert result["total"]["latency_seconds"] == 2
    assert result["per_group"]["total"]["total_tokens"] is None


def test_resource_per_group_counts_include_maintenance_and_preserve_completeness():
    result = summarize_resources([
        {"usage": [{"operation": "agent", "input_tokens": 10, "output_tokens": 2, "cost_rub": .1}]},
        {"usage": [{"operation": "extraction", "input_tokens": 20, "output_tokens": 3, "cost_rub": .2}]},
    ])
    assert result["per_group"]["agent"]["total_tokens"] == 6
    assert result["per_group"]["maintenance"]["total_tokens"] == 11.5
    assert result["per_group"]["total"]["cost_rub"] == pytest.approx(.15)
