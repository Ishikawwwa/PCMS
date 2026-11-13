"""Evidence tests for conservative city identity and exact observed properties."""
import copy
import json

import pytest

from constraint_memory import extraction as original_extraction
from constraint_memory.memory import grounding as original_grounding



@pytest.fixture(scope="module")
def proposed():
    return original_grounding, original_extraction


def selection(city="San Antonio(Texas)", name="Hotel A", slot="accommodation"):
    return {"traveler": "Benjamin", "day": 1, "slot": slot, "attribute": "value",
            "value": f"{name}, {city}", "source_refs": ["event:action"]}


def hotel(city="San Antonio", name="Hotel A", price=80, eid="tool"):
    return {"event_id": eid, "tool_name": "AccommodationSearch",
            "row": {"NAME": name, "city": city, "price": price, "review rate number": 4.5,
                    "room type": "Private room", "house_rules": "No smoking",
                    "minimum nights": 1, "maximum occupancy": 2}}


def extract_price(extraction, choice, rows):
    result = extraction.grounded_attributes([choice], rows)
    price = [r for r in result["decisions"] if r["attribute"] == "price"]
    return result, price


@pytest.mark.parametrize("selected,observed,rank", [
    ("San Antonio", "San Antonio", 2), ("San Antonio(Texas)", "San Antonio(Texas)", 2),
    ("San Antonio(Texas)", "San Antonio", 1), ("San Antonio (Texas)", "San Antonio", 1),
    ("San Antonio(TX)", "San Antonio", 1), ("San Antonio", "San Antonio(Texas)", 1),
    ("New York(New York)", "New York", 1), ("Charleston(West Virginia)", "Charleston", 1),
    ("San Antonio(Texas)", "San Antonio(California)", 0),
    ("San Antonio(Texas)", "Houston", 0), ("San Antonio", "san antonio", 0),
    ("San Antonio(Texas)", "San Antonio(TX)", 0),
    ("San Antonio(downtown)", "San Antonio", 0),
    ("San Antonio(Texas) airport", "San Antonio", 0),
    ("San Antonio(Texas)(downtown)", "San Antonio", 0),
    ("San Antonio()", "San Antonio", 0), ("(Texas)", "", 0),
    (None, "San Antonio", 0), ("San Antonio", None, 0), ("", "", 0),
])
def test_city_alias_is_narrow_and_state_safe(proposed, selected, observed, rank):
    assert proposed[0].city_match_rank(selected, observed) == rank


def test_original_bug_and_proposed_matching_from_same_visible_row(proposed):
    choice, observed = selection(), [hotel()]
    result, price = extract_price(proposed[1], choice, observed)
    assert len(result["decisions"]) == 6
    assert price[0]["value"] == 80
    assert price[0]["selected_value"] == choice["value"]
    assert price[0]["grounded_entity"] == {"name": "Hotel A", "city": "San Antonio(Texas)"}
    assert price[0]["grounding_method"] == "exact_subject_entity_city_state_annotation_observed_row"
    assert proposed[0].property_row_proof(choice["value"], "accommodation", "price", 80,
                                          [observed[0]["row"]]) == (True, "PROPERTY_ROW_CHECKED")


@pytest.mark.parametrize("rows", [
    [hotel("San Antonio", price=80), hotel("San Antonio", price=90)],
    [hotel("San Antonio(Texas)", price=80), hotel("San Antonio(California)", price=90)],
])
def test_ambiguous_bare_city_fails_closed_in_both_paths(proposed, rows):
    choice = selection("San Antonio")
    result, price = extract_price(proposed[1], choice, rows)
    assert price == []
    assert result["unresolved"][0]["reason"] == "AMBIGUOUS_TOOL_ENTITY"
    assert proposed[0].property_row_proof(choice["value"], "accommodation", "price", 80,
        [[r["row"] for r in rows]]) == (False, "PROPERTY_AMBIGUOUS_ENTITY")


@pytest.mark.parametrize("reverse", [False, True])
def test_exact_city_row_is_preferred_independent_of_observation_order(proposed, reverse):
    rows = [hotel("San Antonio", price=90, eid="bare"),
            hotel("San Antonio(Texas)", price=80, eid="exact")]
    if reverse:
        rows.reverse()
    choice = selection()
    _, price = extract_price(proposed[1], choice, rows)
    assert price[0]["value"] == 80 and price[0]["source_refs"][0] == "event:exact"
    assert price[0]["grounding_method"] == "exact_subject_entity_city_observed_row"
    contents = [[r["row"] for r in rows]]
    assert proposed[0].property_row_proof(choice["value"], "accommodation", "price", 80, contents)[0]
    assert not proposed[0].property_row_proof(choice["value"], "accommodation", "price", 90, contents)[0]


@pytest.mark.parametrize("name,city", [("Hotel B", "San Antonio"), ("hotel a", "San Antonio"),
                                      ("Hotel A annex", "San Antonio"), ("Hotel A", "Houston"),
                                      ("Hotel A", "San Antonio(California)")])
def test_name_or_annotated_state_mismatch_cannot_supply_evidence(proposed, name, city):
    choice, rows = selection(), [hotel(city, name)]
    assert extract_price(proposed[1], choice, rows)[1] == []
    assert proposed[0].property_row_proof(choice["value"], "accommodation", "price", 80,
                                          [rows[0]["row"]]) == (False, "PROPERTY_ENTITY_MISMATCH")


def test_identical_rows_repeated_in_observations_are_not_false_ambiguity(proposed):
    row = hotel()
    rows = [row, {**copy.deepcopy(row), "event_id": "repeat"}]
    assert extract_price(proposed[1], selection(), rows)[1][0]["value"] == 80
    assert proposed[0].property_row_proof(selection()["value"], "accommodation", "price", 80,
                                          [[r["row"] for r in rows]])[0]


def test_same_city_conflicting_exact_rows_fail_closed_even_with_bare_fallback(proposed):
    rows = [hotel(price=100), hotel("San Antonio(Texas)", price=80), hotel("San Antonio(Texas)", price=90)]
    assert extract_price(proposed[1], selection(), rows)[1] == []
    assert proposed[0].property_row_proof(selection()["value"], "accommodation", "price", 80,
                                          [[r["row"] for r in rows]]) == (False, "PROPERTY_AMBIGUOUS_ENTITY")


def test_restaurant_properties_and_flight_identity_remain_literal(proposed):
    choice = selection(name="Pita Pit", slot="lunch")
    row = {"event_id": "meal", "tool_name": "RestaurantSearch",
           "row": {"Name": "Pita Pit", "City": "San Antonio", "Average Cost": 38,
                   "Aggregate Rating": 3.8, "Cuisines": "Indian, Chinese"}}
    result = proposed[1].grounded_attributes([choice], [row])
    assert {r["attribute"]: r["value"] for r in result["decisions"]} == {
        "price": 38, "rating": 3.8, "cuisine": "Indian, Chinese"}
    for r in result["decisions"]:
        assert proposed[0].property_row_proof(choice["value"], choice["slot"], r["attribute"], r["value"], [row["row"]])[0]
    flight = {**choice, "slot": "transportation", "value": "Flight Number: F123, from X to Y"}
    observation = {"event_id": "flight", "tool_name": "FlightSearch", "row": {"Flight Number": "F123", "Price": 416}}
    new = proposed[1].grounded_attributes([flight], [observation])
    assert len(new["decisions"]) == 1 and new["decisions"][0]["value"] == 416


def test_wrong_value_and_unrecognized_source_format_still_fail_closed(proposed):
    choice = selection()
    assert proposed[0].property_row_proof(choice["value"], "accommodation", "price", 20,
                                          [hotel()["row"]]) == (False, "PROPERTY_VALUE_MISMATCH")
    assert proposed[0].property_row_proof(choice["value"], "accommodation", "price", 80,
                                          ["Hotel A in San Antonio has some price somewhere."]) == (False, "PROPERTY_UNVERIFIABLE_ROW")


def test_visible_dataframe_source_works_without_structured_row_shortcut(proposed):
    import pandas as pd
    row = hotel()["row"]
    source = pd.DataFrame([row]).to_string(index=False)
    assert proposed[0].property_row_proof(selection()["value"], "accommodation", "price", 80, [source])[0]


def test_proposal_does_not_mutate_evidence_choices_or_live_modules(proposed):
    choice, rows = selection(), [hotel()]
    rows[0]["source_content"] = json.dumps(rows[0]["row"])
    before = copy.deepcopy((choice, rows))
    result = proposed[1].grounded_attributes([choice], rows)
    assert (choice, rows) == before
    assert all(r["evidence_quote"] == rows[0]["source_content"] for r in result["decisions"])


def test_e4_can_commit_alias_grounded_properties_from_literal_action_and_tool_text(proposed):
    from constraint_memory.memory.ledger import LedgerMemory
    m = LedgerMemory("E4", {"token_budget": 10000})
    m.reset("g")
    choice = selection()
    rows = [hotel()]
    rows[0]["source_content"] = json.dumps(rows[0]["row"], sort_keys=True)
    m.observe({"event_id": "tool", "group_id": "g", "step": 0, "type": "tool_result",
               "content": rows[0]["source_content"]})
    m.begin_subtask({"group_id": "g", "traveler": "Benjamin", "step": 0, "query": ""})
    attributes = proposed[1].grounded_attributes([choice], rows)["decisions"]
    result = m.commit_subtask({"group_id": "g", "traveler": "Benjamin", "step": 1,
        "action": "Accommodation: " + choice["value"], "decisions": [choice] + attributes})
    assert result["status"] == "OK"
    committed = m.committed_decisions(1)
    assert len(committed) == 7
    assert {r["attribute"] for r in committed} == {"value", "price", "rating", "room_type",
                                                     "house_rules", "minimum_nights", "maximum_occupancy"}
