"""Narrow contract repair consumes real planner steps and preserves actions."""
import copy
from pathlib import Path

import pytest
from tokenizers import Tokenizer

from constraint_memory.config import load_config
from constraint_memory.memory import create_backend
from constraint_memory.orchestration import group_session as module
from constraint_memory.orchestration.contracts import INHERITANCE_CONTRACT_TEXT
from constraint_memory.orchestration.group_session import GroupSession
from constraint_memory.orchestration.logging import read_jsonl
from constraint_memory.orchestration.replay import canonical_hash
from constraint_memory.orchestration.runner import format_base


ROOT=Path(__file__).resolve().parents[1]
QUERY="I am Bob.\nI'm traveling with Base.\nFor day1 dinner, I want a different place."
BASE={"name":"Base","query":"Fixed public trip", "daily_plans":[
    {"day":day,"current_city":"Boston","transportation":"-","breakfast":f"Cafe {day}",
     "attraction":f"Park {day}","lunch":f"Lunch {day}","dinner":f"Dinner {day}",
     "accommodation":f"Hotel {day}"} for day in (1,2)]}


class Adapter:
    query=QUERY
    def __init__(self,*args):
        self.base={"base_template_id":"family","base_person":copy.deepcopy(BASE)}
        self.metadata={"traveler_count":1}
        self.committed=False
    def iter_requests(self):
        yield {"group_id":"g","traveler_index":1,"name":"Bob","query":self.query}
        assert self.committed
    def commit_result(self,result):
        assert result["name"]=="Bob"
        self.committed=True


class NoModel:
    def __init__(self): self.records=[]
    def set_scope(self,*args): pass


def session(tmp_path,monkeypatch,planner,variant="E3I",steps=30,budget=4096,query=QUERY):
    class CurrentAdapter(Adapter): pass
    CurrentAdapter.query=query
    monkeypatch.setattr(module,"ChronologicalAdapter",CurrentAdapter)
    # Independent slot coverage can be complete with unparsed policy semantics.
    monkeypatch.setattr(module,"extract",lambda *args: {"parse_success":False,"constraints":[],
        "unresolved":[{"reason":"Unparsed policy"}]})
    config=load_config(ROOT/"configs/travel.yaml",ROOT/"configs/variants/E3I.yaml")
    config["agent"]["max_steps_per_traveler"]=steps
    config["agent"]["max_agent_visible_memory_tokens"]=budget
    tokenizer=Tokenizer.from_file(str(ROOT/"external/qwen-tokenizer/tokenizer.json"))
    return GroupSession(ROOT,tmp_path,variant,"g",config,
        {"database":{"database_path":"unused"}},tokenizer,NoModel(),planner,format_base)


def planned(bad=True,steps=1):
    base=copy.deepcopy(BASE)
    base["name"]="Bob"
    base["daily_plans"][0]["dinner"]="Requested Dinner"
    if bad: base["daily_plans"][1]["accommodation"]="Unrequested Hotel"
    action=format_base(base)
    return {"final_plan":action,"submitted_text":action.split("\n",1)[1],"plan":base["daily_plans"],
        "scratchpad":[],"total_steps":steps,"parent_model_requests":steps,"success":True}


def test_current_slot_repair_spends_remaining_budget_and_retains_bad_action(tmp_path,monkeypatch):
    budgets=[]
    class Planner:
        def __init__(self,*args,**kwargs):
            assert kwargs["task_contract_text"]==INHERITANCE_CONTRACT_TEXT
            budgets.append(kwargs["max_steps"])
        def run_turn(self,packet,context,base):
            assert packet["query"]==QUERY and base==BASE
            assert len(group.tokenizer.encode(context).ids)<=4096
            if len(budgets)>1:
                assert "INHERITANCE REPAIR" in context
                assert "Unrequested Hotel" in context and "Hotel 2" in context
            return planned(len(budgets)==1,steps=2 if len(budgets)==1 else 1)
    group=session(tmp_path,monkeypatch,Planner,steps=5)
    predictions,row=group.run()
    assert budgets==[5,3]
    assert predictions[0]["plan"][1]["accommodation"]=="Hotel 2"
    assert predictions[0]["submission_version"]==2
    assert row["travelers"][0]["steps"]==3 and row["travelers"][0]["repair_attempts"]==1
    checks=read_jsonl(tmp_path/"inheritance_checks.jsonl")
    assert [row["check"]["status"] for row in checks]==["VIOLATED","PASSED"]
    assert all(row["check"]["task_success"] is None for row in checks)
    assert any(record["value"]=="Unrequested Hotel" for record in group.backend.snapshot()["decisions"])
    # Reproduce every backend operation with the identical token counter.
    replay=None
    for operation in read_jsonl(tmp_path/"memory_ops.jsonl"):
        if operation["op"]=="reset":
            replay=create_backend("E3I",operation["config"])
            replay.set_token_counter(group.count_tokens,name="Qwen3.6-27B pinned tokenizer")
        if "args" in operation: getattr(replay,operation["op"])(*operation["args"])
        else: getattr(replay,operation["op"])(operation["arg"])
    assert canonical_hash(replay.snapshot())==canonical_hash(group.backend.snapshot())


def test_control_receives_same_contract_without_candidate_repair(tmp_path,monkeypatch):
    seen=[]
    class Planner:
        def __init__(self,*args,**kwargs): seen.append(kwargs)
        def run_turn(self,*args): return planned()
    group=session(tmp_path,monkeypatch,Planner,variant="E0")
    predictions,_=group.run()
    assert len(seen)==1 and seen[0]["task_contract_text"]==INHERITANCE_CONTRACT_TEXT
    assert predictions[0]["plan"][1]["accommodation"]=="Unrequested Hotel"
    assert not (tmp_path/"inheritance_checks.jsonl").exists()
    assert read_jsonl(tmp_path/"events.jsonl")[0]["type"]=="task_contract"


@pytest.mark.parametrize("first_steps,mode,expected_budgets",[(30,"bad",[30]),(28,"missing",[30,2]),(1,"bad",[30,29])])
def test_exhausted_or_unchanged_repair_keeps_actual_latest_plan(tmp_path,monkeypatch,first_steps,mode,expected_budgets):
    budgets=[]
    class Planner:
        def __init__(self,*args,**kwargs): budgets.append(kwargs["max_steps"])
        def run_turn(self,*args):
            if len(budgets)==1: return planned(steps=first_steps)
            if mode=="missing":
                return {"final_plan":"","plan":None,"scratchpad":[],"total_steps":2,
                        "parent_model_requests":2,"success":False}
            return planned()
    group=session(tmp_path,monkeypatch,Planner)
    predictions,row=group.run()
    assert budgets==expected_budgets
    assert predictions[0]["plan"][1]["accommodation"]=="Unrequested Hotel"
    assert row["travelers"][0]["steps"]<=30
    assert row["travelers"][0]["recovery_terminal"] in {"step_budget_exhausted","unchanged_inheritance_failure"}


def test_uncertain_request_scope_never_freezes_extra_legitimate_changes(tmp_path,monkeypatch):
    seen=[]
    class Planner:
        def __init__(self,*args,**kwargs): seen.append(kwargs)
        def run_turn(self,*args): return planned()
    group=session(tmp_path,monkeypatch,Planner,query="I want different dinners and hotels.")
    predictions,_=group.run()
    assert len(seen)==1 and predictions[0]["plan"][1]["accommodation"]=="Unrequested Hotel"
    assert read_jsonl(tmp_path/"inheritance_checks.jsonl")[0]["check"]["status"]=="UNVERIFIED"


def test_memory_overflow_never_dispatches_or_synthesizes_a_plan(tmp_path,monkeypatch):
    class ForbiddenPlanner:
        def __init__(self,*args,**kwargs): raise AssertionError("Planner must not dispatch")
    group=session(tmp_path,monkeypatch,ForbiddenPlanner,budget=10)
    predictions,row=group.run()
    assert predictions[0]["plan"] is None and row["travelers"][0]["steps"]==0
    assert row["travelers"][0]["status"]=="BUDGET_OVERFLOW"


def test_explicit_empty_full_section_replaces_prior_submission_and_withdraws_slots(tmp_path,monkeypatch):
    budgets=[]
    class Planner:
        def __init__(self,*args,**kwargs): budgets.append(kwargs["max_steps"])
        def run_turn(self,*args):
            if len(budgets)==1: return planned(steps=28)
            return {"final_plan":"=== Bob's Plan ===\n", "submitted_text":"", "plan":None,
                    "scratchpad":[],"total_steps":2,"parent_model_requests":2,"success":False}
    group=session(tmp_path,monkeypatch,Planner)
    predictions,row=group.run()
    assert budgets==[30,2]
    assert predictions[0]["plan"] is None and predictions[0]["submission_version"]==2
    assert not [record for record in group.backend.committed_decisions(group.step) if record["traveler"]=="Bob"]
    check=read_jsonl(tmp_path/"inheritance_checks.jsonl")[-1]["check"]
    assert check["status"]=="VIOLATED"
    assert {violation["code"] for violation in check["violations"]}=={"INHERITED_SLOT_MISSING"}
    assert row["travelers"][0]["steps"]==30
