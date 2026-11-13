"""Attraction list punctuation never consumes a model repair step."""
import copy

import pytest

from constraint_memory.memory.inheritance import _same_slot_value
from test_inheritance_contract import inputs
from constraint_memory.memory.inheritance import compile_inheritance_obligations,validate_inheritance


@pytest.mark.parametrize("actual,expected",[("Zoo;Park","Zoo;Park;"),("Zoo;","Zoo"),("Zoo;Park;","Zoo;Park")])
def test_only_one_terminal_attraction_delimiter_is_immaterial(actual,expected):
    assert _same_slot_value("attraction",actual,expected)
    assert not _same_slot_value("accommodation",actual,expected)


@pytest.mark.parametrize("actual,expected",[("Park;Zoo","Zoo;Park;"),("Zoo","Zoo;Park;"),
    ("Zoo;Other","Zoo;Park;"),("Zoo;Park;;","Zoo;Park;"),("Zoo;Park;Park","Zoo;Park;"),
    ("Zoo;park","Zoo;Park;"),("Zoo; Park","Zoo;Park;"),("Zo","Zoo;")])
def test_names_order_multiplicity_and_internal_separators_remain_exact(actual,expected):
    assert not _same_slot_value("attraction",actual,expected)


def test_equivalent_list_passes_without_changing_actual_archived_values():
    contract,base,request,parsed=inputs()
    base=[{**base[0],"slot":"attraction","value":"Zoo;Park;"}]
    compiled=compile_inheritance_obligations(contract,base,request,parsed)
    actual=[{"traveler":"Current","day":1,"slot":"attraction","value":"Zoo;Park"}]
    before=copy.deepcopy((compiled,actual))
    result=validate_inheritance(compiled,actual)
    assert result["status"]=="PASSED" and result["task_success"] is None
    assert (compiled,actual)==before
