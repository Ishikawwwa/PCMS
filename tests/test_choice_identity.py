"""Observed-name repair never supplies an unobserved or ambiguous identifier."""
import copy
import json

import pytest

from constraint_memory.memory.choice_identity import validate_choice_identity


REQUEST={"group_id":"g","traveler":"Current","step":1}
EXCEPTIONS=[{"day":1,"slot":"accommodation"}]


def choices(value="City Limestone, Boston"):
    return [{"decision_id":"decision:Current:day1:accommodation","group_id":"g","traveler":"Current",
             "day":1,"slot":"accommodation","attribute":"value","value":value,"status":"COMMITTED",
             "version":2,"created_at":3,"source_refs":["event:actual"]}]


def events(names=("City Limestone: 2 bedroom suite",),step=2,group="g"):
    return [{"event_id":"rows","step":step,"group_id":group,"type":"tool_result",
             "content":json.dumps([{"Name":name,"City":"Boston","Price":80} for name in names])}]


def check(records=None,observed=None,exceptions=None):
    return validate_choice_identity(REQUEST,records or choices(),observed or events(),
                                    EXCEPTIONS if exceptions is None else exceptions,3)


def test_unique_literal_title_hint_retains_actual_version_and_observed_evidence():
    actual,observed=choices(),events()
    before=copy.deepcopy((actual,observed))
    result=check(actual,observed)
    assert result["status"]=="VIOLATED" and result["task_success"] is None
    row=result["violations"][0]
    assert row["actual"]=="City Limestone, Boston"
    assert row["observed_full_name"]=="City Limestone: 2 bedroom suite"
    assert row["actual_ref"].endswith("@v2")
    assert row["observed_source_refs"]==["event:rows"]
    assert (actual,observed)==before


@pytest.mark.parametrize("names",[("City Limestone: room A","City Limestone: room B"),("Different Limestone: room A",)])
def test_distinct_suffixes_or_arbitrary_prefixes_remain_unverified(names):
    result=check(observed=events(names))
    assert result["status"]=="UNVERIFIED" and not result["violations"]


def test_exact_observed_name_or_whitespace_identity_is_accepted_without_change():
    assert check(choices("City Limestone: 2 bedroom suite, Boston"))["status"]=="PASSED"
    assert check(choices("City Limestone: 2 bedroom suite, Boston"),events(("City Limestone:  2 bedroom suite",)))["status"]=="PASSED"


@pytest.mark.parametrize("observed",[events(step=4),events(group="other")])
def test_future_or_other_group_rows_never_identify_a_name(observed):
    assert not check(observed=observed)["violations"]


def test_unchanged_public_base_slots_are_outside_new_choice_name_checks():
    assert not check(exceptions=[])["violations"]


def test_identical_duplicate_observations_do_not_create_false_ambiguity():
    assert check(observed=events()+events())["status"]=="VIOLATED"


def test_differing_complete_rows_with_same_title_are_ambiguous():
    observed=events()
    altered=copy.deepcopy(observed[0])
    altered["event_id"]="second"
    altered["content"]=json.dumps([{"Name":"City Limestone: 2 bedroom suite","City":"Boston","Price":90}])
    assert check(observed=observed+[altered])["status"]=="UNVERIFIED"
