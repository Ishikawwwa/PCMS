import json
from constraint_memory.memory import create_backend


def request(name,step,query="",extraction=None):
    return {"group_id":"g","traveler":name,"round_idx":1,"step":step,"query":query,"extraction":extraction or {}}


def choice(name,value,day=1,slot="dinner"):
    return {"traveler":name,"day":day,"slot":slot,"value":value,"source_refs":["event:action"]}


def test_full_replacement_retires_omitted_slots_and_entity_attributes():
    memory=create_backend("E3",{"token_budget":10000}); memory.reset("g")
    memory.begin_subtask(request("Bob",0))
    memory.observe({"event_id":"tool","group_id":"g","step":0,"type":"tool_result","content":json.dumps({"NAME":"Hotel A","city":"Boston","price":80})})
    memory.commit_subtask({**request("Bob",1),"action":"Day1 Hotel A, Boston; Day2 Old dinner",
        "decisions":[choice("Bob","Hotel A, Boston",slot="accommodation"),choice("Bob","Old dinner",day=2),
                     {**choice("Bob",80,slot="accommodation"),"attribute":"price","source_refs":["event:tool","event:action"]}]})
    memory.commit_subtask({**request("Bob",2),"action":"Day1 Hotel B, Boston","replace_travelers":["Bob"],
                          "decisions":[choice("Bob","Hotel B, Boston",slot="accommodation")]})
    context=memory.begin_subtask(request("Carol",3,"Bob accommodation dinner"))
    active=[r for r in context["records"] if r.get("traveler")=="Bob"]
    assert any(r.get("value")=="Hotel B, Boston" for r in active)
    assert not any(r.get("value") in (80,"Old dinner") for r in active)
    assert any(r.get("status")=="INVALIDATED" and r.get("value")==80 for r in memory.snapshot()["decisions"])


def test_checked_actual_repair_clears_pending_but_never_claims_task_success():
    memory=create_backend("E5",{"token_budget":10000}); memory.reset("g")
    memory.observe({"event_id":"req","group_id":"g","step":0,"type":"request","content":"Bob must choose Good dinner."})
    constraint={"constraint_id":"c","kind":"LOCAL","subject":{"traveler":"Bob","day":1,"slot":"dinner"},
                "operator":"EQUALS","target":{"value":"Good dinner"},"source_refs":["event:req"]}
    memory.begin_subtask(request("Bob",0,extraction={"constraints":[constraint]}))
    bad=memory.commit_subtask({**request("Bob",1),"action":"Bad dinner","decisions":[choice("Bob","Bad dinner")],"remaining_action_budget":4})
    recovery=bad["recovery"][0]
    assert memory.pending_recoveries()
    fixed=memory.commit_subtask({**request("Bob",2),"action":"Good dinner","decisions":[choice("Bob","Good dinner")],
                                "replace_travelers":["Bob"],"remaining_action_budget":3})
    latest=max((r for r in memory.snapshot()["decisions"] if r["decision_id"]=="decision:Bob:day1:dinner"),key=lambda r:r["version"])
    ack=memory.acknowledge_recovery(recovery["recovery_id"],{"group_id":"g","step":2,
        "repaired_action_event_ids":[fixed["action_event_id"]],"repaired_decision_refs":[f"{latest['decision_id']}@v{latest['version']}"],"revision_action_cost":1})
    assert ack["status"]=="REPLANNED_UNVERIFIED" and not memory.pending_recoveries()
    assert ack["recovery"]["success"] is None
    assert any(r["value"]=="Bad dinner" for r in memory.snapshot()["decisions"])


def test_recovery_cannot_be_acknowledged_with_old_evidence_or_extra_action_budget():
    memory=create_backend("E5",{"token_budget":10000}); memory.reset("g")
    memory.begin_subtask(request("Bob",0))
    original=memory.commit_subtask({**request("Bob",0),"action":"Old","decisions":[choice("Bob","Old")]})
    failed=memory.commit_subtask({**request("Bob",1),"action":"","decisions":[],
        "invalidate_decision_refs":["decision:Bob:day1:dinner"],"remaining_action_budget":1})
    recovery=failed["recovery"][0]
    ack=memory.acknowledge_recovery(recovery["recovery_id"],{"group_id":"g","step":1,
        "repaired_action_event_ids":[original["action_event_id"]],"repaired_decision_refs":["decision:Bob:day1:dinner@v1"],"revision_action_cost":2})
    assert ack["status"]=="REVISION_BUDGET_EXCEEDED" and memory.pending_recoveries()


def test_partial_chain_repair_advances_to_the_unrepaired_descendant():
    memory=create_backend("E5",{"token_budget":10000}); memory.reset("g")
    memory.begin_subtask(request("Alice",0))
    memory.commit_subtask({**request("Alice",0),"action":"Good dinner","decisions":[choice("Alice","Good dinner")]})
    for step,name,prior in [(1,"Bob","Alice"),(2,"Carol","Bob")]:
        eid="req-"+name
        memory.observe({"event_id":eid,"group_id":"g","step":step,"type":"request","content":f"{name} joins {prior} for dinner"})
        constraint={"constraint_id":"c-"+name,"kind":"JOIN","subject":{"traveler":name,"day":1,"slot":"dinner"},
                    "operator":"EQUALS","target":{"ref":f"decision:{prior}:day1:dinner"},"source_refs":["event:"+eid]}
        memory.begin_subtask(request(name,step,extraction={"constraints":[constraint]}))
        memory.commit_subtask({**request(name,step),"action":"Good dinner","decisions":[choice(name,"Good dinner")]})
    trigger=memory.commit_subtask({**request("Bob",3),"action":"","decisions":[],
        "invalidate_decision_refs":["decision:Bob:day1:dinner"],"remaining_action_budget":4})
    recovery=trigger["recovery"][0]
    assert memory.required_repair_travelers()==["Bob","Carol"]
    fixed=memory.commit_subtask({**request("Bob",4),"action":"Good dinner","replace_travelers":["Bob"],"decisions":[choice("Bob","Good dinner")]})
    current=max((r for r in memory.snapshot()["decisions"] if r["traveler"]=="Bob"),key=lambda r:r["version"])
    partial=memory.acknowledge_recovery(recovery["recovery_id"],{"group_id":"g","step":4,
        "repaired_action_event_ids":[fixed["action_event_id"]],"repaired_decision_refs":[f"{current['decision_id']}@v{current['version']}"],"revision_action_cost":1})
    assert partial["status"]=="RECOVERY_REQUIRED"
    assert memory.required_repair_travelers()==["Carol"]


def test_required_slot_omission_creates_a_repairable_scope_without_inventing_a_choice():
    memory=create_backend("E5",{"token_budget":10000}); memory.reset("g")
    memory.observe({"event_id":"req","group_id":"g","step":0,"type":"request","content":"Bob must choose Good dinner."})
    constraint={"constraint_id":"dinner","kind":"LOCAL","subject":{"traveler":"Bob","day":1,"slot":"dinner"},
                "operator":"EQUALS","target":{"value":"Good dinner"},"source_refs":["event:req"]}
    memory.begin_subtask(request("Bob",0,extraction={"constraints":[constraint]}))
    missing=memory.commit_subtask({**request("Bob",1),"action":"Only lunch","replace_travelers":["Bob"],
                                  "decisions":[choice("Bob","Only lunch",slot="lunch")],"remaining_action_budget":4})
    assert missing["status"]=="RECOVERY_REQUIRED"
    assert memory.required_repair_travelers()==["Bob"]
    assert not any(d["slot"]=="dinner" for d in memory.snapshot()["decisions"])
    fixed=memory.commit_subtask({**request("Bob",2),"action":"Good dinner","replace_travelers":["Bob"],"decisions":[choice("Bob","Good dinner")]})
    latest=max((d for d in memory.snapshot()["decisions"] if d["slot"]=="dinner"),key=lambda d:d["version"])
    recovery=missing["recovery"][0]
    ack=memory.acknowledge_recovery(recovery["recovery_id"],{"group_id":"g","step":2,
        "repaired_action_event_ids":[fixed["action_event_id"]],"repaired_decision_refs":[f"{latest['decision_id']}@v{latest['version']}"],"revision_action_cost":1})
    assert ack["status"]=="REPLANNED_UNVERIFIED"


def test_unobserved_property_is_unverified_rather_than_a_missing_itinerary_choice():
    memory=create_backend("E5",{"token_budget":10000}); memory.reset("g")
    memory.observe({"event_id":"req","group_id":"g","step":0,"type":"request","content":"Bob dinner costs less than 80."})
    constraint={"constraint_id":"price","kind":"LOCAL","subject":{"traveler":"Bob","day":1,"slot":"dinner","attribute":"price"},
                "operator":"LT","target":{"value":80},"source_refs":["event:req"]}
    memory.begin_subtask(request("Bob",0,extraction={"constraints":[constraint]}))
    result=memory.commit_subtask({**request("Bob",1),"action":"Exact Cafe, Boston","replace_travelers":["Bob"],
                                 "decisions":[choice("Bob","Exact Cafe, Boston")],"remaining_action_budget":4})
    assert not memory.pending_recoveries()
    assert not result["recovery"]
