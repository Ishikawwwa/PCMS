"""E4 source numeric validation must preserve signs and decimal values."""

import json

import pytest

from constraint_memory.extraction import parse_submission
from constraint_memory.memory import create_backend


def validate_literal(value, source):
    backend = create_backend("E4", {"token_budget": 10000})
    backend.reset("review")
    backend.observe({"event_id": "request", "group_id": "review", "step": 0,
                     "type": "request", "content": source})
    constraint = {"kind": "LOCAL", "scope": {"group": "review", "traveler": "Bob", "day": 1},
        "subject": {"traveler": "Bob", "day": 1, "slot": "accommodation", "attribute": "price"},
        "operator": "LTE", "target": {"value": value}, "source_refs": ["event:request"],
        "evidence_quote": source, "confidence": 1}
    context = backend.begin_subtask({"group_id": "review", "traveler": "Bob", "step": 0,
        "query": source, "round_idx": 1, "extraction": {"constraints": [constraint]}})
    return backend.snapshot(), context


def test_equivalent_decimal_source_spelling_is_valid_literal_evidence():
    snapshot, context = validate_literal(80.0, "Bob's day-one accommodation must cost at most 80.00 USD.")
    assert snapshot["constraints"]
    assert not any(issue["code"] == "EVIDENCE_MISMATCH" for issue in context["diagnostics"])


def test_numeric_sign_mismatch_is_rejected_even_when_digits_match():
    snapshot, context = validate_literal(20, "Bob's accommodation must cost at most -20 USD.")
    assert not snapshot["constraints"]
    assert any(issue["code"] == "EVIDENCE_MISMATCH" for issue in context["diagnostics"])


@pytest.mark.parametrize("price, should_commit", [(80, True), (20, False)])
def test_property_evidence_binds_value_to_selected_entity_row(price, should_commit):
    backend = create_backend("E4", {"token_budget": 10000})
    backend.reset("review")
    backend.observe({"event_id": "request", "group_id": "review", "step": 0,
        "type": "request", "content": "Bob requests accommodation in Boston."})
    backend.begin_subtask({"group_id": "review", "traveler": "Bob", "step": 0,
                          "round_idx": 1, "query": "Select accommodation"})
    tool_rows = [{"NAME": "Hotel A", "city": "Boston", "price": 80},
                 {"NAME": "Hotel B", "city": "Boston", "price": 20}]
    source = json.dumps(tool_rows, sort_keys=True)
    backend.observe({"event_id": "tool", "group_id": "review", "step": 1,
                     "type": "tool_result", "content": source})
    action = "=== Bob's Plan ===\nDay 1:\nAccommodation: Hotel A, Boston"
    decisions = parse_submission(action)
    decisions.append({"traveler": "Bob", "day": 1, "slot": "accommodation",
        "attribute": "price", "value": price, "selected_value": "Hotel A, Boston",
        "grounded_entity": {"name": "Hotel A", "city": "Boston"},
        "grounding_method": "exact_subject_entity_city_observed_row",
        "source_refs": ["event:tool", "event:action"], "evidence_quote": source,
        "status": "COMMITTED"})
    backend.commit_subtask({"group_id": "review", "traveler": "Bob", "step": 2,
                            "action": action, "decisions": decisions})
    properties = [decision for decision in backend.snapshot()["decisions"]
                  if decision.get("attribute") == "price"]
    assert bool(properties) is should_commit
