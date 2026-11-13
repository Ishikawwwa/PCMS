"""Declared inheritance checks actual current choices without repairing them."""
import copy
import json

import pytest

from constraint_memory.memory.binding_tables import unpack_sections
from constraint_memory.memory.inheritance import declare_inheritance_contract
from constraint_memory.memory.inherited_frontier import InheritedFrontierMemory


def observe(memory, eid, content, step, event_type="request"):
    memory.observe({"group_id": "g", "event_id": eid, "step": step, "type": event_type, "content": content})


def choice(name, slot, value, day=1):
    return {"traveler": name, "day": day, "slot": slot, "value": value, "source_refs": ["event:action"]}


def seeded(budget=10000, configure=True):
    memory = InheritedFrontierMemory({"token_budget": budget})
    memory.reset("g")
    text = "Explicitly preserve unrequested slots from the public base."
    observe(memory, "contract", text, 0, "task_contract")
    contract = declare_inheritance_contract("base-v1", "g", text, ["event:contract"], enabled=True, created_at=0)
    observe(memory, "base-request", "I am Base.\nBuild my trip.", 1)
    memory.begin_subtask({"group_id": "g", "traveler": "Base", "step": 1, "query": "I am Base.\nBuild my trip."})
    rows = [choice("Base", "breakfast", "Base Cafe, Boston"), choice("Base", "dinner", "Base Grill, Boston")]
    memory.commit_subtask({"group_id": "g", "traveler": "Base", "step": 1,
                          "action": "Base Cafe, Boston; Base Grill, Boston", "decisions": rows})
    if configure:
        memory.configure_inheritance(contract, "Base")
    return memory, contract


def begin(memory, query="For dinner on day1, I want a restaurant that bans smoking.", name="Current", step=2, prior=None, extraction=None, eid=None):
    eid = eid or f"request-{name}-{step}"
    observe(memory, eid, query, step)
    request = {"group_id": "g", "traveler": name, "step": step, "query": query,
               "source_refs": ["event:" + eid], "prior_travelers": prior or ["Base"], "extraction": extraction or {}}
    return request, memory.begin_subtask(request)


def commit(memory, rows, name="Current", step=2, **extra):
    return memory.commit_subtask({"group_id": "g", "traveler": name, "step": step,
        "action": "; ".join(str(row["value"]) for row in rows), "decisions": rows,
        "replace_travelers": sorted({row["traveler"] for row in rows}), **extra})


def sections(context):
    return unpack_sections(json.loads(context["context"].split("\n", 1)[1]))


def test_compact_row_discloses_contract_and_exact_exceptions_and_archives_full_provenance():
    memory, contract = seeded()
    request, context = begin(memory)
    assert context["inheritance_hard_checks_enabled"]
    row = sections(context)["DECLARED INHERITANCE"][0]
    assert row["base"] == "Base"
    assert row["exceptions"] == {"1": ["dinner"]}
    assert row["source_refs"] == ["event:contract", *request["source_refs"]]
    assert row["contract_ref"] == "task-contract:base-v1@v1"
    assert "Base Cafe, Boston" not in context["context"]
    archived = memory.snapshot()["inheritance"]
    assert len(archived["base_decisions"]) == 2
    assert archived["base_fingerprint"] == row["base_sha256"]
    compiled = archived["compiled"]["Current"]
    assert not compiled["semantic_interpretation_verified"]
    assert compiled["causal_request_verified"]
    assert compiled["constraints"][0]["target"]["value"] == "Base Cafe, Boston"
    assert compiled["constraints"][0]["base_decision_ref"] == "decision:Base:day1:breakfast@v1"
    assert compiled["exceptions"][0]["scope_evidence"][0]["evidence_quote"] == request["query"]
    assert archived["archive"][0]["certificate"]["source_refs"] == request["source_refs"]


@pytest.mark.parametrize("alter", ["missing_source", "wrong_text", "wrong_event_type", "future_source", "foreign_group"])
def test_configuration_requires_exact_observed_causal_current_group_contract(alter):
    memory, contract = seeded(configure=False)
    if alter == "missing_source":
        contract["source_refs"] = ["event:absent"]
    elif alter == "wrong_text":
        contract["text"] = "Invented policy."
    elif alter == "wrong_event_type":
        contract["source_refs"] = ["event:base-request"]
    elif alter == "future_source":
        observe(memory, "future-contract", contract["text"], 10, "task_contract")
        contract["created_at"] = 11
    else:
        contract["group_id"] = "other"
    with pytest.raises(ValueError):
        memory.configure_inheritance(contract, "Base")


def test_configuration_requires_already_committed_actual_base_values():
    memory = InheritedFrontierMemory()
    memory.reset("g")
    observe(memory, "contract", "Explicit policy", 0, "task_contract")
    contract = declare_inheritance_contract("c", "g", "Explicit policy", ["event:contract"], enabled=True)
    with pytest.raises(ValueError, match="committed"):
        memory.configure_inheritance(contract, "Base")
    memory.begin_subtask({"group_id": "g", "traveler": "Base", "step": 0, "query": "base"})
    memory.commit_subtask({"group_id": "g", "traveler": "Base", "step": 0,
        "action": "Different literal", "decisions": [choice("Base", "breakfast", "Fabricated Cafe")]})
    with pytest.raises(ValueError, match="actual action"):
        memory.configure_inheritance(contract, "Base")


@pytest.mark.parametrize("alter", ["raw_query", "source", "source_type", "future_source", "unknown_prior", "foreign_prior"])
def test_current_hard_scope_is_bound_to_actual_request_evidence_and_visible_people(alter):
    memory, _ = seeded()
    request, _ = begin(memory)
    if alter == "raw_query":
        request["query"] = "For breakfast on day1, I want a cheap place."
    elif alter == "source":
        request["source_refs"] = ["event:absent"]
    elif alter == "source_type":
        request["source_refs"] = ["event:contract"]
    elif alter == "future_source":
        observe(memory, "future-request", request["query"], 9)
        request["source_refs"] = ["event:future-request"]
    elif alter == "unknown_prior":
        request["prior_travelers"] = ["Future"]
    else:
        request["prior_travelers"] = [{"group_id": "other", "traveler": "Base", "created_at": 0}]
    context = memory.begin_subtask(request)
    assert not context["inheritance_hard_checks_enabled"]
    result = commit(memory, [choice("Current", "breakfast", "Changed Cafe, Boston")])
    assert result["inheritance_check"]["status"] == "UNVERIFIED"
    assert result["inheritance_check"]["violations"] == []


def test_ambiguous_missing_current_scope_keeps_advisory_obligations_without_freezing_slots():
    memory, _ = seeded()
    _, context = begin(memory, "I want to join Base and need every meal cheaper.")
    assert not context["inheritance_hard_checks_enabled"]
    check = commit(memory, [choice("Current", "breakfast", "Changed Cafe, Boston")])["inheritance_check"]
    assert check["status"] == "UNVERIFIED"
    assert check["task_success"] is None


def test_violation_preserves_failed_actual_choice_and_only_current_repair_is_rendered():
    memory, _ = seeded()
    request, _ = begin(memory)
    rows = [choice("Current", "breakfast", "Changed Cafe, Boston"), choice("Current", "dinner", "Requested Grill, Boston")]
    check = commit(memory, rows, status="FAILED", branch_status="FAILED")["inheritance_check"]
    assert check["status"] == "VIOLATED"
    assert check["task_success"] is None
    latest = memory._current_values("Current", 2)
    assert {row["value"] for row in latest} == {row["value"] for row in rows}
    assert all(row["status"] == "COMMITTED" for row in latest)
    request["step"] = 3
    repair = memory.begin_subtask(request)
    repair_rows = sections(repair)["INHERITANCE REPAIR"]
    assert len(repair_rows) == 1
    assert repair_rows[0]["traveler"] == "Current"
    assert repair_rows[0]["expected"] == "Base Cafe, Boston"
    assert repair_rows[0]["actual_ref"] == "decision:Current:day1:breakfast@v1"
    assert any(ref.startswith("event:action:g:Current:") for ref in repair_rows[0]["source_refs"])
    assert not any(record["constraint_id"].startswith("constraint:task-contract:") for record in memory.snapshot()["constraints"])
    _, next_context = begin(memory, "For dinner on day1, I want a cheap place.", name="Next", step=4, prior=["Base", "Current"])
    assert "INHERITANCE REPAIR" not in sections(next_context)


def test_actual_revision_resolves_inheritance_without_autofill_and_retains_history():
    memory, _ = seeded()
    request, _ = begin(memory)
    first = commit(memory, [choice("Current", "breakfast", "Changed Cafe, Boston"), choice("Current", "dinner", "Requested Grill, Boston")])
    assert first["inheritance_check"]["status"] == "VIOLATED"
    request["step"] = 3
    memory.begin_subtask(request)
    second = commit(memory, [choice("Current", "breakfast", "Base Cafe, Boston"), choice("Current", "dinner", "Requested Grill, Boston")], step=3)
    assert second["inheritance_check"]["status"] == "PASSED"
    assert second["inheritance_check"]["checked_slot_count"] == 1
    snapshot = memory.snapshot()
    assert any(row["value"] == "Changed Cafe, Boston" for row in snapshot["decisions"])
    assert len(snapshot["inheritance"]["check_history"]) == 2
    assert snapshot["inheritance"]["checks"]["Current"]["check"]["status"] == "PASSED"


def test_missing_current_submission_preserves_previous_actual_and_extra_person_choices():
    memory, _ = seeded()
    request, _ = begin(memory)
    commit(memory, [choice("Current", "breakfast", "Base Cafe, Boston"), choice("Current", "dinner", "Requested Grill, Boston")])
    request["step"] = 3
    memory.begin_subtask(request)
    before = memory._current_values("Current", 3)
    result = commit(memory, [choice("Other", "breakfast", "Other Cafe, Boston")], step=3, replace_travelers=["Other"])
    assert result["inheritance_check"]["status"] == "PASSED"
    assert result["inheritance_check"]["current_submission_present"] is False
    assert memory._current_values("Current", 3) == before
    assert memory._current_values("Other", 3)[0]["value"] == "Other Cafe, Boston"
    assert all(row["subject"]["traveler"] == "Current" for row in result["inheritance_check"]["violations"])


def test_explicit_empty_full_current_section_withdraws_previous_actual_instead_of_passing_stale_choices():
    memory, _ = seeded()
    request, _ = begin(memory)
    accepted = commit(memory, [choice("Current", "breakfast", "Base Cafe, Boston"), choice("Current", "dinner", "Requested Grill, Boston")])
    assert accepted["inheritance_check"]["status"] == "PASSED"
    request["step"] = 3
    memory.begin_subtask(request)
    empty = memory.commit_subtask({"group_id": "g", "traveler": "Current", "step": 3,
        "action": "=== Current's Plan ===", "decisions": [], "replace_travelers": ["Current"]})
    assert empty["inheritance_check"]["status"] == "VIOLATED"
    assert empty["inheritance_check"]["violations"][0]["code"] == "INHERITED_SLOT_MISSING"
    assert memory._current_values("Current", 3) == []
    assert any(row["value"] == "Base Cafe, Boston" for row in memory.snapshot()["decisions"])


def test_full_current_revision_missing_unchanged_slot_reports_missing_without_autofill():
    memory, _ = seeded()
    begin(memory)
    checked = commit(memory, [choice("Current", "dinner", "Requested Grill, Boston")])["inheritance_check"]
    assert checked["status"] == "VIOLATED"
    assert checked["violations"][0]["code"] == "INHERITED_SLOT_MISSING"
    assert not any(row["slot"] == "breakfast" for row in memory._current_values("Current", 2))


def test_immutable_base_snapshot_does_not_follow_later_base_revision_and_reconfigure_is_idempotent():
    memory, contract = seeded()
    fingerprint = memory.snapshot()["inheritance"]["base_fingerprint"]
    commit(memory, [choice("Base", "breakfast", "Revised Base Cafe, Boston")], name="Base", step=2)
    memory.configure_inheritance(contract, "Base")
    begin(memory, step=3)
    checked = commit(memory, [choice("Current", "breakfast", "Revised Base Cafe, Boston")], step=3)["inheritance_check"]
    assert checked["status"] == "VIOLATED"
    assert checked["violations"][0]["expected"] == "Base Cafe, Boston"
    assert memory.snapshot()["inheritance"]["base_fingerprint"] == fingerprint


def test_custom_inheritance_and_repair_rows_participate_in_protected_overflow_batches():
    memory, _ = seeded(budget=30)
    request, context = begin(memory)
    assert context["status"] == "BUDGET_OVERFLOW"
    assert context["token_count"] <= 30
    assert any("DECLARED INHERITANCE" in batch["context"] for batch in context["batches"])
    commit(memory, [choice("Current", "breakfast", "Changed Cafe, Boston")])
    request["step"] = 3
    repair = memory.begin_subtask(request)
    assert repair["status"] == "BUDGET_OVERFLOW"
    assert repair["token_count"] <= 30
    assert any("INHERITANCE REPAIR" in batch["context"] for batch in repair["batches"])
    for batch in repair["batches"]:
        if batch["status"] == "OK":
            assert batch["token_estimate"] <= 30


def test_replay_of_explicit_calls_is_deterministic_and_reset_removes_group_state():
    first, contract = seeded()
    second, _ = seeded()
    for memory in (first, second):
        request, _ = begin(memory)
        commit(memory, [choice("Current", "breakfast", "Base Cafe, Boston"), choice("Current", "dinner", "Requested Grill, Boston")])
    assert first.snapshot() == second.snapshot()
    copied = first.snapshot()
    copied["inheritance"]["base_decisions"][0]["value"] = "mutated"
    assert first.snapshot() != copied
    first.reset("other")
    assert first.snapshot()["inheritance"]["contract"] is None
    assert first.snapshot()["inheritance"]["archive"] == []
    with pytest.raises(ValueError, match="Group"):
        first.configure_inheritance(contract, "Base")


def test_fixed_public_base_is_required_outside_the_memory_cap():
    with pytest.raises(ValueError, match="common planner prompt"):
        InheritedFrontierMemory({"public_base_outside_memory": False})
