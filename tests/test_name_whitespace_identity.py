"""Conservative name aliases ground only unique causally visible complete rows."""
import copy
import json

import pytest

from constraint_memory.extraction import grounded_attributes, parse_submission
from constraint_memory.memory.grounding import entity_name_match_rank, property_row_proof
from constraint_memory.memory.ledger import LedgerMemory


SELECTED_NAME = "Airy Brooklyn Limestone: 2 bedroom suite"
VISIBLE_NAME = "Airy Brooklyn Limestone:  2 bedroom suite"


def choice(name=SELECTED_NAME, city="San Antonio(Texas)"):
    return {"traveler": "Benjamin", "day": 1, "slot": "accommodation", "attribute": "value",
            "value": f"{name}, {city}", "source_refs": ["event:action"]}


def observation(name=VISIBLE_NAME, city="San Antonio", price=603.0, event_id="visible-tool", **extra):
    # This complete row was actually displayed by the development pilot's
    # AccommodationSearch, before a submitted single-space spelling. No answer
    # key or later observation is needed to identify its 603 price/3 rating.
    row = {"NAME": name, "price": price, "room type": "Private room",
           "house_rules": "No visitors & No children under 10", "minimum nights": 3.0,
           "maximum occupancy": 2, "review rate number": 3.0, "city": city, **extra}
    return {"event_id": event_id, "tool_name": "AccommodationSearch", "row": row,
            "source_content": json.dumps(row, sort_keys=True)}


def properties(selected, rows):
    return grounded_attributes([selected], rows)


def proof(selected, rows, value=603.0, attribute="price"):
    return property_row_proof(selected["value"], selected["slot"], attribute, value,
                              [row["source_content"] for row in rows])


@pytest.mark.parametrize("selected,observed,rank", [
    (SELECTED_NAME, SELECTED_NAME, 2), (SELECTED_NAME, VISIBLE_NAME, 1),
    (SELECTED_NAME, "Airy Brooklyn Limestone:\t 2 bedroom suite", 1),
    (SELECTED_NAME, "  Airy Brooklyn Limestone: 2 bedroom suite\t", 1),
    (SELECTED_NAME, "Airy Brooklyn Limestone : 2 bedroom suite", 0),
    (SELECTED_NAME, "Airy Brooklyn Limestone; 2 bedroom suite", 0),
    (SELECTED_NAME, "airy Brooklyn Limestone: 2 bedroom suite", 0),
    (SELECTED_NAME, "Airy Brooklyn Limestone: 2 Bedroom suite", 0),
    (SELECTED_NAME, "Airy Brooklyn Limestone: 2 bedroom suites", 0),
    (SELECTED_NAME, "Airy Brooklyn Limestone:\n2 bedroom suite", 0),
    (SELECTED_NAME, "Airy Brooklyn Limestone:\v2 bedroom suite", 0),
    (SELECTED_NAME, "Airy Brooklyn Limestone:\u00a02 bedroom suite", 0),
    ("", "", 0), (" \t", " \t", 0), (None, SELECTED_NAME, 0), (SELECTED_NAME, None, 0),
])
def test_alias_changes_only_ascii_horizontal_whitespace(selected, observed, rank):
    assert entity_name_match_rank(selected, observed) == rank


def test_visible_pilot_row_recovers_properties_without_rewriting_choice_or_evidence():
    selected, rows = choice(), [observation()]
    before = copy.deepcopy((selected, rows))
    result = properties(selected, rows)
    facts = {record["attribute"]: record for record in result["decisions"]}
    assert facts["price"]["value"] == 603.0 and facts["rating"]["value"] == 3.0
    assert len(facts) == 6
    for record in facts.values():
        assert record["selected_value"] == selected["value"]
        assert record["grounded_entity"] == {"name": SELECTED_NAME, "city": "San Antonio(Texas)"}
        assert record["grounding_method"] == "subject_entity_horizontal_whitespace_alias_city_state_annotation_observed_row"
        assert record["source_refs"] == ["event:visible-tool", "event:action"]
        assert record["evidence_quote"] == rows[0]["source_content"]
        assert proof(selected, rows, record["value"], record["attribute"]) == (True, "PROPERTY_ROW_CHECKED")
    assert (selected, rows) == before


@pytest.mark.parametrize("reverse", [False, True])
def test_literal_name_precedes_whitespace_alias_independent_of_row_order(reverse):
    selected = choice()
    rows = [observation(price=603.0, event_id="alias"),
            observation(name=SELECTED_NAME, price=620.0, event_id="literal")]
    if reverse:
        rows.reverse()
    price = next(record for record in properties(selected, rows)["decisions"] if record["attribute"] == "price")
    assert price["value"] == 620.0 and price["source_refs"][0] == "event:literal"
    assert price["grounding_method"] == "exact_subject_entity_city_state_annotation_observed_row"
    assert proof(selected, rows, 620.0)[0]
    assert proof(selected, rows, 603.0) == (False, "PROPERTY_VALUE_MISMATCH")


@pytest.mark.parametrize("reverse", [False, True])
def test_exact_city_precedes_state_alias_within_name_alias_tier(reverse):
    selected = choice()
    rows = [observation(price=603.0, event_id="bare-city"),
            observation(city="San Antonio(Texas)", price=620.0, event_id="literal-city")]
    if reverse:
        rows.reverse()
    price = next(record for record in properties(selected, rows)["decisions"] if record["attribute"] == "price")
    assert price["value"] == 620.0 and price["source_refs"][0] == "event:literal-city"
    assert price["grounding_method"] == "subject_entity_horizontal_whitespace_alias_city_observed_row"
    assert proof(selected, rows, 620.0)[0]
    assert not proof(selected, rows, 603.0)[0]


def test_exact_name_precedence_retains_existing_safe_city_alias_rules():
    selected = choice()
    rows = [observation(name=SELECTED_NAME, price=620.0, event_id="exact-name"),
            observation(city="San Antonio(Texas)", price=603.0, event_id="alias-name")]
    price = next(record for record in properties(selected, rows)["decisions"] if record["attribute"] == "price")
    assert price["value"] == 620.0 and price["source_refs"][0] == "event:exact-name"
    assert proof(selected, rows, 620.0)[0]


@pytest.mark.parametrize("other", [
    "Airy Brooklyn Limestone:   2 bedroom suite",
    "Airy Brooklyn Limestone:\t2 bedroom suite",
])
def test_distinct_raw_name_collisions_fail_even_when_both_properties_agree(other):
    selected = choice()
    rows = [observation(), observation(name=other, event_id="collision")]
    result = properties(selected, rows)
    assert result["decisions"] == []
    assert result["unresolved"] == [{"reason": "AMBIGUOUS_TOOL_ENTITY", "entity": SELECTED_NAME, "city": "San Antonio(Texas)"}]
    assert proof(selected, rows) == (False, "PROPERTY_AMBIGUOUS_ENTITY")
    assert proof(selected, rows, 3.0, "rating") == (False, "PROPERTY_AMBIGUOUS_ENTITY")


def test_distinct_complete_rows_fail_even_when_name_price_and_rating_agree():
    selected = choice()
    rows = [observation(), observation(event_id="conflicting-row", house_rules="No smoking")]
    assert properties(selected, rows)["decisions"] == []
    assert proof(selected, rows) == (False, "PROPERTY_AMBIGUOUS_ENTITY")


def test_identical_complete_rows_can_deduplicate_without_mutating_sources():
    selected = choice()
    rows = [observation(), observation(event_id="repeat")]
    before = copy.deepcopy(rows)
    result = properties(selected, rows)
    assert len(result["decisions"]) == 6
    assert all(record["source_refs"][0] == "event:repeat" for record in result["decisions"])
    assert proof(selected, rows)[0]
    assert rows == before


@pytest.mark.parametrize("name,city", [
    ("Airy Brooklyn Limestone :  2 bedroom suite", "San Antonio"),
    ("Airy Brooklyn Limestone:  2 bedroom Suite", "San Antonio"),
    (VISIBLE_NAME, "Houston"), (VISIBLE_NAME, "san antonio"),
    (VISIBLE_NAME, "San Antonio(California)"), (VISIBLE_NAME, "San Antonio(downtown)"),
])
def test_whitespace_alias_does_not_relax_punctuation_case_or_city(name, city):
    selected, rows = choice(), [observation(name=name, city=city)]
    assert properties(selected, rows)["decisions"] == []
    assert proof(selected, rows) == (False, "PROPERTY_ENTITY_MISMATCH")


def test_visible_dataframe_proof_preserves_internal_double_space_name():
    import pandas as pd
    selected, rows = choice(), [observation()]
    contents = [pd.DataFrame([rows[0]["row"]]).to_string(index=False)]
    assert property_row_proof(selected["value"], "accommodation", "price", 603.0, contents)[0]
    assert property_row_proof(selected["value"], "accommodation", "rating", 3.0, contents)[0]


def test_whitespace_matching_does_not_relax_flight_identifiers():
    selected = {**choice(), "slot": "transportation", "value": "Flight Number: F123, from X to Y"}
    row = {"event_id": "flight", "tool_name": "FlightSearch", "row": {"Flight Number": "F 123", "Price": 603.0},
           "source_content": json.dumps({"Flight Number": "F 123", "Price": 603.0})}
    assert properties(selected, [row])["decisions"] == []
    assert proof(selected, [row]) == (False, "PROPERTY_ENTITY_MISMATCH")


def test_e4_alias_facts_anchor_the_literal_revised_entity_and_preserve_archive():
    memory = LedgerMemory("E4", {"token_budget": 10000})
    memory.reset("g")
    rows = [observation()]
    memory.observe({"event_id": "visible-tool", "group_id": "g", "step": 0,
                    "type": "tool_result", "content": rows[0]["source_content"]})
    for step, name in [(1, VISIBLE_NAME), (3, SELECTED_NAME)]:
        selected = choice(name=name)
        action = f"=== Benjamin's Plan ===\nDay 1:\nAccommodation: {selected['value']}"
        memory.begin_subtask({"group_id": "g", "traveler": "Benjamin", "step": step - 1, "query": ""})
        attributes = properties(selected, rows)["decisions"]
        result = memory.commit_subtask({"group_id": "g", "traveler": "Benjamin", "step": step,
            "action": action, "decisions": parse_submission(action) + attributes,
            "replace_travelers": ["Benjamin"]})
        assert result["status"] == "OK"
    current = memory.committed_decisions(3)
    entities = [record for record in current if record["attribute"] == "value"]
    assert len(entities) == 1 and entities[0]["value"] == choice()["value"] and entities[0]["version"] == 2
    facts = [record for record in current if record["attribute"] != "value"]
    assert len(facts) == 6
    for record in facts:
        assert record["entity_ref"] == "decision:Benjamin:day1:accommodation@v2"
        assert record["selected_value"] == choice()["value"]
        assert record["evidence_quote"] == rows[0]["source_content"]
        assert "event:visible-tool" in record["source_refs"]
        assert record["grounding_method"] == "subject_entity_horizontal_whitespace_alias_city_state_annotation_observed_row"
    assert any(record["value"] == choice(name=VISIBLE_NAME)["value"] and record["version"] == 1
               for record in memory.snapshot()["decisions"] if record["attribute"] == "value")


def test_future_visible_row_metadata_cannot_ground_an_earlier_e4_commit():
    memory = LedgerMemory("E4", {"token_budget": 10000})
    memory.reset("g")
    rows = [observation()]
    memory.observe({"event_id": "visible-tool", "group_id": "g", "step": 5,
                    "type": "tool_result", "content": rows[0]["source_content"]})
    selected = choice()
    action = f"=== Benjamin's Plan ===\nDay 1:\nAccommodation: {selected['value']}"
    memory.begin_subtask({"group_id": "g", "traveler": "Benjamin", "step": 0, "query": ""})
    memory.commit_subtask({"group_id": "g", "traveler": "Benjamin", "step": 1,
        "action": action, "decisions": parse_submission(action) + properties(selected, rows)["decisions"]})
    assert not any(record["attribute"] != "value" for record in memory.committed_decisions(1))
