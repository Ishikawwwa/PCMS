"""Synthetic integration of partial source checks, actual choices, and repair.

These fixtures declare their own property units and tool observations. They do
not read benchmark answers, recorded research runs, databases, or model APIs.
"""
import copy
import json

import pytest

from constraint_memory.memory.binding_tables import unpack_sections
from constraint_memory.memory.inheritance import declare_inheritance_contract
from constraint_memory.memory.request_checks import validate_request_checks
from constraint_memory.memory.verified_frontier import VerifiedFrontierMemory


BASE = {
    (1, "breakfast"): "Base Cafe, Wellington",
    (1, "dinner"): "Base Grill, Wellington",
    (1, "accommodation"): "Immutable Base House, Wellington",
    (4, "accommodation"): "Immutable Base House, Wellington",
}
ROWS = [
    {"NAME": "Reference House", "city": "Wellington", "price": 900, "review rate number": 3},
    {"NAME": "Cheap House", "city": "Wellington", "price": 500, "review rate number": 3},
    {"NAME": "Suitable House", "city": "Wellington", "price": 800, "review rate number": 3},
    {"NAME": "Harbor House: private suite", "city": "Wellington", "price": 900, "review rate number": 3},
    {"Name": "Base Grill", "City": "Wellington", "Average Cost": 40, "Aggregate Rating": 3},
]
QUERY = (
    "I am Noah.\nI'm traveling with Atlas and Lea.\n"
    "For my first-day accommodation, I want a place priced within $100 of Lea's day4 accommodation "
    "and rated at least as high.\n\nFor dinner on day1, I'd like to join Lea."
)


def observe(memory, eid, content, step, kind="request"):
    memory.observe({"group_id": "g", "event_id": eid, "step": step, "type": kind, "content": content})


def request(memory, name, query, step, *, eid=None, prior=None, observe_new=True):
    eid = eid or f"request-{name}-{step}"
    if observe_new: observe(memory, eid, query, step)
    raw = {"group_id": "g", "traveler": name, "step": step, "query": query,
        "source_refs": ["event:" + eid], "prior_travelers": ["Atlas"] if prior is None else prior}
    return raw, memory.begin_subtask(raw)


def full_plan(name, accommodation, *, rows=ROWS, tool_ref="event:tool-rows"):
    values = {**BASE, (1, "accommodation"): accommodation} if name == "Noah" else {**BASE, (4, "accommodation"): accommodation}
    decisions = []
    for (day, slot), value in values.items():
        choice = {"traveler": name, "day": day, "slot": slot, "attribute": "value", "value": value,
            "source_refs": ["event:action"]}
        decisions.append(choice)
        selected_name = value.rsplit(",", 1)[0]
        matching = [row for row in rows if row.get("NAME", row.get("Name")) == selected_name]
        if slot != "accommodation" or len(matching) != 1: continue
        row = matching[0]
        for attribute, column in (("price", "price"), ("rating", "review rate number")):
            fact = {**choice, "attribute": attribute, "value": row[column],
                "source_refs": [tool_ref, "event:declared-schema", "event:action"]}
            if attribute == "price": fact["unit"] = "USD"
            decisions.append(fact)
    return decisions


def commit(memory, name, decisions, step):
    return memory.commit_subtask({"group_id": "g", "traveler": name, "step": step,
        "action": json.dumps(decisions, ensure_ascii=False), "decisions": decisions, "replace_travelers": [name]})


def seeded(budget=10000):
    memory = VerifiedFrontierMemory({"token_budget": budget})
    memory.reset("g")
    schema = {"source_kind": "DECLARED_PROPERTY_SCHEMA", "fields": {"price": {"unit": "USD"}}}
    observe(memory, "declared-schema", schema, 0, "property_schema")
    text = "Explicit synthetic policy: keep unrequested immutable public-base slots."
    observe(memory, "contract", text, 0, "task_contract")
    contract = declare_inheritance_contract("synthetic-base", "g", text, ["event:contract"], enabled=True)
    request(memory, "Atlas", "I am Atlas.\nBuild my fixed itinerary.", 1)
    base = [{"traveler": "Atlas", "day": day, "slot": slot, "value": value,
        "source_refs": ["event:action"]} for (day, slot), value in BASE.items()]
    commit(memory, "Atlas", base, 1)
    memory.configure_inheritance(contract, "Atlas")
    observe(memory, "tool-rows", json.dumps(ROWS), 2, "tool_result")
    request(memory, "Lea", "For accommodation on day4, I want a private place.", 2)
    commit(memory, "Lea", full_plan("Lea", "Reference House, Wellington"), 2)
    return memory


def decoded(context):
    return unpack_sections(json.loads(context["context"].split("\n", 1)[1]))


def test_initial_read_certifies_only_supported_individual_predicates_and_no_full_semantics():
    memory = seeded()
    _, context = request(memory, "Noah", QUERY, 3, prior=["Atlas", "Lea"])
    assert context["status"] == "OK"
    assert context["request_predicates_certified"] == 3
    assert "VERIFIED REQUEST REPAIR" not in decoded(context)
    compiled = memory.snapshot()["request_checks"]["compiled"]["Noah"]
    assert not compiled["full_request_semantics_verified"]
    assert {row["subject"]["traveler"] for row in compiled["predicates"]} == {"Noah"}
    assert all(row["source_refs"] == ["event:request-Noah-3"] for row in compiled["predicates"])
    assert all(row["binding"]["source_refs"] for row in compiled["predicates"])


def test_proven_price_failure_preserves_actual_choice_and_shows_all_requested_choices_for_repair():
    memory = seeded()
    raw, _ = request(memory, "Noah", QUERY, 3, prior=["Atlas", "Lea"])
    result = commit(memory, "Noah", full_plan("Noah", "Cheap House, Wellington"), 3)
    check = result["request_check"]
    assert check["status"] == "VIOLATED" and check["task_success"] is None
    assert result["inheritance_check"]["status"] == "PASSED"
    assert len(check["violations"]) == 1
    violation = check["violations"][0]
    assert violation["check_kind"] == "request_predicate"
    assert violation["actual"] == 500 and violation["expected_value"] == "900"
    assert violation["current_entity_ref"] == "decision:Noah:day1:accommodation@v1"
    raw["step"] = 4
    context = memory.begin_subtask(raw)
    sections = decoded(context)
    assert len(sections["VERIFIED REQUEST REPAIR"]) == 1
    assert {row["value"] for row in sections["CURRENT REQUEST CHOICES"]} == {
        "Cheap House, Wellington", "Base Grill, Wellington"}
    assert memory._current_values("Noah", 4)[0]["status"] == "COMMITTED"


def test_full_model_repair_preserves_correct_requested_slot_and_archives_failed_submission():
    memory = seeded()
    raw, _ = request(memory, "Noah", QUERY, 3, prior=["Atlas", "Lea"])
    commit(memory, "Noah", full_plan("Noah", "Cheap House, Wellington"), 3)
    raw["step"] = 4
    memory.begin_subtask(raw)
    result = commit(memory, "Noah", full_plan("Noah", "Suitable House, Wellington"), 4)
    assert result["request_check"]["status"] == "CHECKED_UNVERIFIED_TASK"
    assert result["request_check"]["predicate_check"]["status"] == "PASSED"
    assert result["request_check"]["task_success"] is None
    latest = {(row["day"], row["slot"]): row for row in memory._current_values("Noah", 4)}
    assert latest[(1, "dinner")]["value"] == "Base Grill, Wellington"
    assert latest[(1, "accommodation")]["value"] == "Suitable House, Wellington"
    archived = memory.snapshot()
    assert any(row["value"] == "Cheap House, Wellington" for row in archived["decisions"])
    assert len(archived["request_checks"]["check_history"]) >= 3
    json.dumps(archived, allow_nan=False)


def test_unique_observed_truncated_title_is_reported_but_never_property_grounded_or_autofilled():
    memory = seeded()
    raw, _ = request(memory, "Noah", QUERY, 3, prior=["Atlas", "Lea"])
    result = commit(memory, "Noah", full_plan("Noah", "Harbor House, Wellington"), 3)
    violations = result["request_check"]["violations"]
    assert len(violations) == 1
    assert violations[0]["check_kind"] == "observed_full_name"
    assert violations[0]["observed_full_name"] == "Harbor House: private suite"
    current = memory.committed_decisions(3)
    assert not any(row["traveler"] == "Noah" and row["slot"] == "accommodation" and row.get("attribute") == "price" for row in current)
    assert any(row["value"] == "Harbor House, Wellington" for row in memory._current_values("Noah", 3))
    raw["step"] = 4
    memory.begin_subtask(raw)
    repaired = commit(memory, "Noah", full_plan("Noah", "Harbor House: private suite, Wellington"), 4)
    assert repaired["request_check"]["violations"] == []
    assert repaired["request_check"]["predicate_check"]["status"] == "PASSED"


def test_changed_prior_anchor_disables_old_compiled_comparison_instead_of_false_violation():
    memory = seeded()
    request(memory, "Noah", QUERY, 3, prior=["Atlas", "Lea"])
    compiled = copy.deepcopy(memory.snapshot()["request_checks"]["compiled"]["Noah"])
    commit(memory, "Noah", full_plan("Noah", "Cheap House, Wellington"), 3)
    updated_rows = [{"NAME": "New Reference House", "city": "Wellington", "price": 600, "review rate number": 3}]
    observe(memory, "new-rows", json.dumps(updated_rows), 4, "tool_result")
    request(memory, "Lea", "For accommodation on day4, I want a private place.", 4)
    commit(memory, "Lea", full_plan("Lea", "New Reference House, Wellington", rows=updated_rows, tool_ref="event:new-rows"), 4)
    pool = [{**row, "group_id": "g"} for row in memory.committed_decisions(4)]
    check = validate_request_checks(compiled, pool, step=4)
    assert check["status"] == "UNVERIFIED"
    assert check["violations"] == []
    assert all(row["code"] == "DEPENDENCY_CHANGED" for row in check["unverified"])


def test_changed_anchor_rebegin_revalidates_and_drops_stale_repair_card():
    memory = seeded()
    raw, _ = request(memory, "Noah", QUERY, 3, prior=["Atlas", "Lea"])
    commit(memory, "Noah", full_plan("Noah", "Cheap House, Wellington"), 3)
    updated_rows = [{"NAME": "New Reference House", "city": "Wellington", "price": 600, "review rate number": 3}]
    observe(memory, "new-rows", json.dumps(updated_rows), 4, "tool_result")
    request(memory, "Lea", "For accommodation on day4, I want a private place.", 4)
    commit(memory, "Lea", full_plan("Lea", "New Reference House, Wellington", rows=updated_rows, tool_ref="event:new-rows"), 4)
    raw["step"] = 5
    context = memory.begin_subtask(raw)
    assert "VERIFIED REQUEST REPAIR" not in decoded(context)
    assert context["request_predicates_certified"] == 3


def test_changed_anchor_remaining_failure_card_uses_new_version_and_bound():
    memory = seeded()
    raw, _ = request(memory, "Noah", QUERY, 3, prior=["Atlas", "Lea"])
    commit(memory, "Noah", full_plan("Noah", "Cheap House, Wellington"), 3)
    updated_rows = [{"NAME": "New Reference House", "city": "Wellington", "price": 1000, "review rate number": 3}]
    observe(memory, "new-rows", json.dumps(updated_rows), 4, "tool_result")
    request(memory, "Lea", "For accommodation on day4, I want a private place.", 4)
    commit(memory, "Lea", full_plan("Lea", "New Reference House, Wellington", rows=updated_rows, tool_ref="event:new-rows"), 4)
    raw["step"] = 5
    context = memory.begin_subtask(raw)
    repair = decoded(context)["VERIFIED REQUEST REPAIR"]
    assert len(repair) == 1
    assert repair[0]["expected_value"] == "1000"
    current_price = next(row for row in memory.committed_decisions(5)
        if row["decision_id"] == "decision:Lea:day4:accommodation:price")
    assert current_price["version"] > 1
    assert repair[0]["target_ref"] == f"{current_price['decision_id']}@v{current_price['version']}"
    assert "event:new-rows" in repair[0]["target_source_refs"]
    # Earlier check evidence remains historical rather than being overwritten.
    history = memory.snapshot()["request_checks"]["check_history"]
    old = next(row for row in history if row["traveler"] == "Noah")
    assert old["check"]["violations"][0]["expected_value"] == "900"


def test_unsupported_semantics_and_unobserved_names_remain_unverified_not_task_success():
    memory = seeded()
    query = "For accommodation on day1, I want a quiet vegan place with a flexible booking policy."
    _, context = request(memory, "Noah", query, 3, prior=["Atlas", "Lea"])
    assert context["request_predicates_certified"] == 0
    result = commit(memory, "Noah", full_plan("Noah", "Unobserved Place, Wellington"), 3)
    assert result["request_check"]["predicate_check"]["status"] == "UNVERIFIED"
    assert result["request_check"]["identity_check"]["status"] == "UNVERIFIED"
    assert result["request_check"]["violations"] == []
    assert result["request_check"]["task_success"] is None


def test_forged_raw_request_or_future_tool_rows_never_authorize_a_current_repair():
    memory = seeded()
    raw, _ = request(memory, "Noah", QUERY, 3, prior=["Atlas", "Lea"])
    raw["query"] = QUERY.replace("$100", "$400")
    context = memory.begin_subtask(raw)
    assert context["request_predicates_certified"] == 0
    result = commit(memory, "Noah", full_plan("Noah", "Cheap House, Wellington"), 3)
    assert result["request_check"]["violations"] == []
    raw, _ = request(memory, "Noah", "For accommodation on day1, I want a quiet place.", 4, prior=["Atlas", "Lea"])
    observe(memory, "future-only-row", json.dumps([{"NAME": "Future House: private suite", "city": "Wellington", "price": 900}]), 9, "tool_result")
    result = commit(memory, "Noah", full_plan("Noah", "Future House, Wellington"), 4)
    assert result["request_check"]["violations"] == []
    assert result["request_check"]["identity_check"]["status"] == "UNVERIFIED"


def test_other_person_does_not_receive_current_repair_rows_and_snapshot_replay_is_exact():
    memories = [seeded(), seeded()]
    for memory in memories:
        request(memory, "Noah", QUERY, 3, prior=["Atlas", "Lea"])
        commit(memory, "Noah", full_plan("Noah", "Cheap House, Wellington"), 3)
        _, context = request(memory, "Iris", "For dinner on day1, I want a calm restaurant.", 4, prior=["Atlas", "Lea", "Noah"])
        assert "VERIFIED REQUEST REPAIR" not in decoded(context)
        assert "CURRENT REQUEST CHOICES" not in decoded(context)
    assert memories[0].snapshot() == memories[1].snapshot()
    assert memories[0].snapshot()["inheritance"]["contract"]["source_refs"] == ["event:contract"]
    copied = memories[0].snapshot()
    copied["request_checks"]["latest_checks"]["Noah"]["check"]["violations"].clear()
    assert copied != memories[0].snapshot()


def test_tiny_budget_does_not_hide_concatenated_repair_rows_outside_the_cap():
    memory = seeded(budget=30)
    raw, _ = request(memory, "Noah", QUERY, 3, prior=["Atlas", "Lea"])
    commit(memory, "Noah", full_plan("Noah", "Cheap House, Wellington"), 3)
    raw["step"] = 4
    context = memory.begin_subtask(raw)
    assert context["status"] == "BUDGET_OVERFLOW"
    assert context["token_count"] <= 30
    assert any("VERIFIED REQUEST REPAIR" in batch["context"] for batch in context["batches"])
    assert any("CURRENT REQUEST CHOICES" in batch["context"] for batch in context["batches"])


def test_large_exact_decimal_property_does_not_round_away_a_real_price_violation():
    memory = seeded()
    reference = "10000000000000000000000000000.01"
    current = "9999999999999999999999999999.99"
    rows = [
        {"NAME": "New Reference House", "city": "Wellington", "price": reference, "review rate number": 3},
        {"NAME": "Near Boundary House", "city": "Wellington", "price": current, "review rate number": 3},
    ]
    observe(memory, "decimal-rows", json.dumps(rows), 4, "tool_result")
    request(memory, "Lea", "For accommodation on day4, I want a private place.", 4)
    commit(memory, "Lea", full_plan("Lea", "New Reference House, Wellington", rows=rows, tool_ref="event:decimal-rows"), 4)
    request(memory, "Noah", QUERY.replace("$100", "$0.01"), 5, prior=["Atlas", "Lea"])
    result = commit(memory, "Noah", full_plan("Noah", "Near Boundary House, Wellington", rows=rows, tool_ref="event:decimal-rows"), 5)
    check = result["request_check"]["predicate_check"]
    assert check["status"] == "VIOLATED"
    assert len(check["violations"]) == 1
    assert check["violations"][0]["expected_value"] == reference
    assert check["violations"][0]["actual"] == current
