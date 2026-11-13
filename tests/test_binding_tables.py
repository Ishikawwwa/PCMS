"""Exact roundtrips for bindings; no models, gold plans, or extra tool rows."""
import copy
import math

import pytest

from constraint_memory.memory import create_backend
from constraint_memory.memory.binding_tables import pack_sections, unpack_sections, serialize_sections


def test_order_multiplicity_null_missing_and_nested_symbol_keys_are_exact():
    ref = "decision:Alice:day2:dinner@v3"
    first = {"record_ref": ref, "value": "~literal", "unit": None, "source_refs": ["event:g:1"]}
    sections = {"KNOWN CONTEXT": [first, {"record_ref": ref, "value": "Actual Cafe", "depends_on": {ref: "~~literal"}},
        {"record_ref": "decision:Bob:day1:lunch@v1", "value": "Actual Lunch"}, copy.deepcopy(first)],
        "ACTIVE CONSTRAINTS": [{"subject": {"day": 2, "slot": "dinner"}, "target": {"ref": ref, "offset": -15.5}, "source_refs": ["event:g:1"]}],
        "OPEN QUESTIONS": [{"detail": {"~0": "~literal", ref: [None, False, 0, "event:unique"]}}]}
    before = copy.deepcopy(sections)
    packed = pack_sections(sections)
    assert unpack_sections(packed) == before
    assert sections == before
    assert len(packed["tables"]["KNOWN CONTEXT"]["rows"]) == 4
    assert serialize_sections(sections).startswith("EXACT BINDING TABLES")


@pytest.mark.parametrize("bad", [None, {"version": True}, {"version": 2}, {"version": 1, "symbols": [], "tables": {}}])
def test_malformed_table_is_never_silently_decoded(bad):
    with pytest.raises(ValueError): unpack_sections(bad)


@pytest.mark.parametrize("row", [[True, []], [1, []], [0, []], [0, [1, 2]], [0]])
def test_invalid_row_width_and_schema_references_fail(row):
    with pytest.raises(ValueError):
        unpack_sections({"version": 1, "symbols": {}, "tables": {"x": {"columns": [["value"]], "rows": [row]}}})


def test_unknown_symbol_and_duplicate_columns_fail():
    with pytest.raises(ValueError):
        unpack_sections({"version": 1, "symbols": {}, "tables": {"x": {"columns": [["value"]], "rows": [[0, ["~9"]]]}}})
    with pytest.raises(ValueError):
        unpack_sections({"version": 1, "symbols": {}, "tables": {"x": {"columns": [["value", "value"]], "rows": []}}})


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, object(), (1, 2), {1: "numeric key"}])
def test_unsupported_values_are_not_stringified_or_dropped(value):
    with pytest.raises((ValueError, TypeError)): pack_sections({"x": [{"value": value}]})


def test_table_frontier_retains_causal_choice_version_and_archive():
    memory = create_backend("E3T", {"token_budget": 100000})
    memory.reset("g")
    request = {"group_id": "g", "traveler": "Alice", "step": 0}
    memory.begin_subtask(request)
    memory.commit_subtask({**request, "action": "Actual Cafe, Boston", "decisions": [
        {"traveler": "Alice", "day": 1, "slot": "dinner", "value": "Actual Cafe, Boston"}]})
    result = memory.begin_subtask({"group_id": "g", "traveler": "Current", "step": 1,
                                   "dependency_refs": ["decision:Alice:day1:dinner"]})
    assert result["status"] == "OK"
    assert result["closure_ids"] == ["decision:Alice:day1:dinner@v1"]
    assert result["records"][0]["value"] == "Actual Cafe, Boston"
    assert memory.snapshot()["variant"] == "E3T"
    assert memory.features["lossless_binding_tables"]


def test_opaque_reference_symbols_do_not_require_a_benchmark_id_vocabulary():
    sections={"facts":[{"reference":"meeting/room-A/availability","value":True}],
              "rules":[{"subject":"meeting/room-A/availability","operator":"EQUALS","target":True}]}
    packed=pack_sections(sections,reference_values=["meeting/room-A/availability"])
    assert "meeting/room-A/availability" in packed["symbols"].values()
    assert unpack_sections(packed)==sections
