"""Transfer fixtures use no travel schema, benchmark corpus, or answer keys."""
import copy
import json

import pytest

from constraint_memory.memory.portable import ConstraintMemory
from constraint_memory.memory.predicate_kernel import compare
from constraint_memory.memory.binding_tables import TABLE_GUIDANCE,unpack_sections


def configured(scope,values,predicates):
    memory=ConstraintMemory(scope)
    memory.observe("records",0,values)
    for ref,value in values.items(): memory.write_fact(ref,"records","/"+ref,0)
    memory.observe("requirements",1,predicates,kind="task_contract")
    for index,predicate in enumerate(predicates): memory.write_constraint("requirement-"+str(index),"requirements","/"+str(index),1)
    return memory


@pytest.mark.parametrize("scope,values,predicate,expected",[
    ("meeting",{"earlier.finish":75,"next.start":90},{"subject_ref":"next.start","operator":"GTE","target":{"ref":"earlier.finish","offset":10}},"SATISFIED"),
    ("meeting",{"earlier.finish":75,"next.start":80},{"subject_ref":"next.start","operator":"GTE","target":{"ref":"earlier.finish","offset":10}},"VIOLATED"),
    ("procurement",{"quote.total":121,"purchase.limit":120},{"subject_ref":"quote.total","operator":"LTE","target":{"ref":"purchase.limit"}},"VIOLATED"),
    ("deployment",{"build.flags":["audit","tls"],"approved.flags":["tls"]},{"subject_ref":"build.flags","operator":"SUPERSET_OF","target":{"ref":"approved.flags"}},"SATISFIED"),
    ("lab-inventory",{"container.location":"cabinet-B","required.location":"cabinet-A"},{"subject_ref":"container.location","operator":"EQUALS","target":{"ref":"required.location"}},"VIOLATED"),
    ("configuration",{"package.enabled":True},{"subject_ref":"package.enabled","operator":"EQUALS","target":{"value":1}},"VIOLATED"),
])
def test_same_kernel_handles_unrelated_domains_without_schema_switch(scope,values,predicate,expected):
    memory=configured(scope,values,[predicate])
    result=memory.check("requirement-0",1)
    assert result["status"]==expected and result["task_success"] is None
    assert result["bindings"] and result["source"]["event_id"]=="requirements"


def test_live_revision_changes_check_and_snapshot_preserves_exact_target_version():
    predicates=[{"subject_ref":"chosen.cost","operator":"LTE","target":{"ref":"budget","binding_mode":"LIVE"}},
        {"subject_ref":"chosen.cost","operator":"LTE","target":{"ref":"budget","binding_mode":"SNAPSHOT","version":1}}]
    memory=configured("purchase",{"chosen.cost":90,"budget":100},predicates)
    assert memory.check("requirement-0",1)["status"]=="SATISFIED"
    memory.observe("budget-change",2,{"budget":80})
    memory.write_fact("budget","budget-change","/budget",2)
    assert memory.check("requirement-0",2)["status"]=="VIOLATED"
    assert memory.check("requirement-1",2)["status"]=="SATISFIED"
    assert memory.check("requirement-0",1)["status"]=="SATISFIED"


def test_model_json_is_advisory_without_explicit_structured_contract():
    memory=ConstraintMemory("configuration")
    memory.observe("fact",0,{"enabled":False})
    memory.write_fact("enabled","fact","/enabled",0)
    memory.observe("model-output",1,{"subject_ref":"enabled","operator":"EQUALS","target":{"value":True}},kind="model_output")
    memory.write_constraint("requested","model-output","",1)
    assert memory.check("requested",1)["reason"]=="SEMANTICS_NOT_VERIFIED"


def test_unknown_future_and_cross_scope_inputs_never_satisfy_requirements():
    memory=configured("schedule",{"slot.start":5},[{"subject_ref":"slot.start","operator":"LTE","target":{"ref":"deadline"}}])
    memory.observe("future",5,{"deadline":9})
    memory.write_fact("deadline","future","/deadline",5)
    assert memory.check("requirement-0",1)["reason"]=="TARGET_UNKNOWN"
    assert memory.check("requirement-0",5)["status"]=="SATISFIED"
    with pytest.raises(ValueError,match="scope"): memory.observe("cross",6,{},scope_id="other")
    with pytest.raises(ValueError,match="future"): memory.write_fact("bad","future","/deadline",1)


def test_stale_selected_entity_properties_require_fresh_evidence():
    memory=ConstraintMemory("procurement")
    memory.observe("supplier-A",0,{"selected":"A","cost":20})
    memory.write_fact("supplier","supplier-A","/selected",0)
    memory.write_fact("selected.cost","supplier-A","/cost",0,entity_ref={"ref":"supplier","version":1})
    memory.observe("supplier-B",2,{"selected":"B"})
    memory.write_fact("supplier","supplier-B","/selected",2)
    assert memory.resolve("selected.cost",2) is None
    assert memory.resolve("selected.cost",1)["value"]==20
    assert len(memory.snapshot()["facts"]["selected.cost"])==1


def test_explicit_withdrawal_never_resurrects_an_old_value():
    memory=configured("scheduling",{"reservation":"Room-B"},[])
    memory.observe("cancellation",2,{"withdraw":"reservation"},kind="action")
    memory.withdraw_fact("reservation","cancellation","",2)
    assert memory.resolve("reservation",2) is None
    assert memory.resolve("reservation",1)["value"]=="Room-B"
    assert len(memory.snapshot()["facts"]["reservation"])==2


def test_structured_provenance_pointer_and_immutable_event_are_checked():
    memory=ConstraintMemory("config")
    memory.observe("source",0,{"a/b":{"~field":False},"items":[3,4]})
    assert memory.write_fact("flag","source","/a~1b/~0field",0)["value"] is False
    assert memory.write_fact("second","source","/items/1",0)["value"]==4
    with pytest.raises(ValueError): memory.write_fact("bad","source","/items/01",0)
    with pytest.raises(ValueError): memory.write_fact("bad","source","/a~2b",0)
    with pytest.raises(ValueError,match="immutable"): memory.observe("source",0,{})


def test_exact_frontier_and_overflow_use_same_envelope_in_every_domain():
    memory=configured("deployment",{"revision":7,"expected":7},[{"subject_ref":"revision","operator":"EQUALS","target":{"ref":"expected"}}])
    before=copy.deepcopy(memory.snapshot())
    context=memory.frontier(["requirement-0"],1,len,100000)
    assert unpack_sections(json.loads(context["context"][len(TABLE_GUIDANCE):]))==context["sections"]
    small=memory.frontier(["requirement-0"],1,len,5)
    assert small["status"]=="BUDGET_OVERFLOW" and len(small["context"])<=5
    assert memory.snapshot()==before


@pytest.mark.parametrize("actual,operator,target,expected",[(True,"EQUALS",1,False),
    ("abc","IN","abcdef",None),(["safe"],"SUPERSET_OF",["safe","audit"],False),
    ("0.3","ABS_DIFF_LTE",{"value":"0.2","tolerance":"0.1"},True),
    ("0.3","ABS_DIFF_LTE",{"value":"0.2","tolerance":"-0.1"},None),
    (float("nan"),"LTE",4,None),(False,"LT",1,None),
    ("1.2.3","LT","1.2.4",None)])
def test_types_decimal_boundaries_and_unsupported_domain_comparators_are_honest(actual,operator,target,expected):
    assert compare(actual,operator,target) is expected


@pytest.mark.parametrize("namespace",["meeting","budget","release","opaque-namespace-947","unseen-unicode-Ω"])
def test_renaming_every_resource_and_field_does_not_change_reasoning(namespace):
    import hashlib
    references=[hashlib.sha256((namespace+str(index)).encode()).hexdigest() for index in range(2)]
    memory=configured(namespace,{references[0]:17,references[1]:20},[
        {"subject_ref":references[0],"operator":"LTE","target":{"ref":references[1],"offset":-2}}])
    assert memory.check("requirement-0",1)["status"]=="SATISFIED"
    memory.observe("updated",2,{references[0]:19})
    memory.write_fact(references[0],"updated","/"+references[0],2)
    assert memory.check("requirement-0",2)["status"]=="VIOLATED"


@pytest.mark.parametrize("wording",["Please keep a ten minute gap after Alpha.",
    "Hi team,\nBeta needs to follow Alpha by 10 minutes.\nThanks!",
    "Scheduling rule: Beta.start >= Alpha.end + 10 min."])
def test_free_form_and_changed_presentation_remain_available_without_fake_certification(wording):
    memory=ConstraintMemory("meeting")
    memory.observe("user-message",0,wording,kind="request")
    memory.observe("writer",1,{"subject_ref":"Beta.start","operator":"GTE",
        "target":{"ref":"Alpha.end","offset":10}},kind="model_output")
    interpreted=memory.write_advisory_constraint("gap","writer","","user-message",0,len(wording),1)
    assert interpreted["source"]["quote"]==wording
    assert interpreted["semantic_status"]=="UNVERIFIED"
    assert memory.check("gap",1)["reason"]=="SEMANTICS_NOT_VERIFIED"
    context=memory.frontier(["gap"],1,len,100000)
    decoded=unpack_sections(json.loads(context["context"][len(TABLE_GUIDANCE):]))
    assert decoded["CONSTRAINTS"][0]["source"]["quote"]==wording


def test_fact_constraint_namespace_collisions_are_rejected():
    memory=configured("namespace",{"value":7},[])
    memory.observe("constraint",1,{"subject_ref":"value","operator":"EQUALS","target":{"value":7}},kind="task_contract")
    with pytest.raises(ValueError,match="collide"): memory.write_constraint("value","constraint","",1)


def test_simultaneous_conflicting_observations_are_unknown_until_later_revision():
    memory=configured("inventory",{"count":4},[{"subject_ref":"count","operator":"GTE","target":{"value":3}}])
    memory.observe("sensor-A",2,{"count":2})
    memory.observe("sensor-B",2,{"count":5})
    memory.write_fact("count","sensor-A","/count",2)
    memory.write_fact("count","sensor-B","/count",2)
    assert memory.resolve("count",2) is None
    assert memory.resolve("count",2,3) is None
    assert memory.check("requirement-0",2)["status"]=="UNVERIFIED"
    memory.observe("reconciled",3,{"count":5})
    memory.write_fact("count","reconciled","/count",3)
    assert memory.check("requirement-0",3)["status"]=="SATISFIED"
    assert memory.resolve("count",1)["value"]==4


def test_unit_mismatch_is_unknown_even_if_comparator_omits_unit():
    memory=configured("purchase",{},[{"subject_ref":"cost","operator":"LTE","target":{"ref":"limit"}}])
    memory.observe("prices",2,{"cost":10,"limit":20})
    memory.write_fact("cost","prices","/cost",2,unit="EUR")
    memory.write_fact("limit","prices","/limit",2,unit="USD")
    assert memory.check("requirement-0",2)["reason"]=="UNIT_UNKNOWN_OR_MISMATCH"


@pytest.mark.parametrize("actual,operator,target,expected",[(True,"IN",[1],False),
    ([True],"INTERSECTS",[1],False),({"x":True},"EQUALS",{"x":1},False),
    ([1],"SUPERSET_OF",[True],False),([{"x":False}],"EQUALS",[{"x":0}],False),
    ("1000000000000000000000000000001","ABS_DIFF_LTE",{"value":"1000000000000000000000000000000","tolerance":0},False)])
def test_nested_equality_preserves_json_types_and_arithmetic_is_exact(actual,operator,target,expected):
    assert compare(actual,operator,target) is expected


def test_large_integer_offset_does_not_round_away_constraint_violation():
    amount=10**30
    memory=configured("large-numbers",{"actual":amount,"base":amount},[
        {"subject_ref":"actual","operator":"GTE","target":{"ref":"base","offset":1}}])
    assert memory.check("requirement-0",1)["status"]=="VIOLATED"


def test_non_json_keys_and_contradictory_live_snapshot_targets_are_rejected():
    memory=ConstraintMemory("typed")
    with pytest.raises(ValueError,match="keys"): memory.observe("same",0,{1:"setting"})
    assert "same" not in memory.events
    memory.observe("rule",1,{"subject_ref":"x","operator":"LTE","target":{"ref":"limit","version":1}},kind="task_contract")
    with pytest.raises(ValueError,match="LIVE"): memory.write_constraint("rule","rule","",1)
