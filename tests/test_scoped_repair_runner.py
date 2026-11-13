from pathlib import Path
from types import SimpleNamespace
import pytest
from tokenizers import Tokenizer
from constraint_memory.config import load_config
from constraint_memory.orchestration import group_session as module
from constraint_memory.orchestration.group_session import GroupSession
from constraint_memory.orchestration.runner import format_base
from constraint_memory.orchestration.logging import read_jsonl

ROOT=Path(__file__).resolve().parents[1]


class CurrentOnlyAdapter:
    def __init__(self,*args):
        self.base={"base_template_id":"family","base_person":{"name":"Base","query":"fixed public base",
            "daily_plans":[{"day":1,"dinner":"Base dinner"}]}}
        self.metadata={"traveler_count":1}
        self.committed=False
    def iter_requests(self):
        yield {"group_id":"g","traveler_index":1,"name":"Bob","query":"Bob must choose Good dinner."}
        assert self.committed
    def commit_result(self,result):
        assert result["group_id"]=="g" and result["traveler_index"]==1
        assert not self.committed
        self.committed=True


class NoInference:
    def __init__(self): self.records=[]
    def set_scope(self,*args): pass


def extracted_constraint(client,request,events,*args):
    if request["traveler"]!="Bob": return {}
    return {"constraints":[{"constraint_id":"require-good","kind":"LOCAL",
        "subject":{"traveler":"Bob","day":1,"slot":"dinner"},"operator":"EQUALS",
        "target":{"value":"Good dinner"},"source_refs":["event:"+events[0]["event_id"]]}]}


def session(tmp_path,monkeypatch,planner,steps=30):
    monkeypatch.setattr(module,"ChronologicalAdapter",CurrentOnlyAdapter)
    monkeypatch.setattr(module,"extract",extracted_constraint)
    config=load_config(ROOT/"configs/travel.yaml")
    config["agent"]["max_steps_per_traveler"]=steps
    config["memory"]["backend"]="E5"
    tokenizer=Tokenizer.from_file(str(ROOT/"external/qwen-tokenizer/tokenizer.json"))
    return GroupSession(ROOT,tmp_path,"E5","g",config,{"database":{"database_path":"external/MemoryArena/env/env_systems/travel_planner_env/database"}},
        tokenizer,NoInference(),planner,format_base)


def planned(value,steps):
    text=f"Day 1:\nDinner: {value}"
    return {"final_plan":f"=== Bob's Plan ===\n{text}","submitted_text":text,
        "plan":[{"day":1,"dinner":value}],"success":True,"scratchpad":[],"total_steps":steps,
        "parent_model_requests":steps,"error_message":None}


def test_actual_repair_shares_remaining_steps_and_clears_only_checked_memory(tmp_path,monkeypatch):
    budgets=[]
    class Planner:
        def __init__(self,*args,**kwargs): budgets.append(kwargs["max_steps"])
        def run_turn(self,packet,context,base):
            assert "gold" not in packet and "answers" not in packet
            if len(budgets)==1: return planned("Bad dinner",2)
            assert "RECOVERY_REQUIRED" in context and "Good dinner" in context
            return planned("Good dinner",1)
    group=session(tmp_path,monkeypatch,Planner,steps=5)
    predictions,row=group.run()
    assert budgets==[5,3]
    assert predictions[0]["plan"][0]["dinner"]=="Good dinner"
    assert predictions[0]["submission_version"]==2
    assert row["travelers"][0]["steps"]==3 and row["travelers"][0]["repair_attempts"]==1
    assert not group.pending()
    recovery=group.backend.snapshot()["recovery_log"][0]
    assert recovery["status"]=="REPLANNED_UNVERIFIED" and recovery["success"] is None
    assert any(r["value"]=="Bad dinner" for r in group.backend.snapshot()["decisions"])
    assert any(op["op"]=="acknowledge_recovery" for op in read_jsonl(tmp_path/"memory_ops.jsonl"))


def test_exhausted_repair_keeps_latest_actual_submission_instead_of_synthetic_null(tmp_path,monkeypatch):
    budgets=[]
    class Planner:
        def __init__(self,*args,**kwargs): budgets.append(kwargs["max_steps"])
        def run_turn(self,*args):
            if len(budgets)==1: return planned("Bad dinner",28)
            return {"final_plan":"","plan":None,"success":False,"scratchpad":[],"total_steps":2,
                    "parent_model_requests":2,"error_message":"step cap"}
    group=session(tmp_path,monkeypatch,Planner)
    predictions,row=group.run()
    assert budgets==[30,2]
    assert predictions[0]["plan"][0]["dinner"]=="Bad dinner"
    assert row["travelers"][0]["steps"]==30
    assert row["travelers"][0]["recovery_terminal"]=="step_budget_exhausted"
    assert group.pending()


def test_no_extra_dispatch_when_initial_planning_uses_all_thirty_steps(tmp_path,monkeypatch):
    budgets=[]
    class Planner:
        def __init__(self,*args,**kwargs): budgets.append(kwargs["max_steps"])
        def run_turn(self,*args): return planned("Bad dinner",30)
    group=session(tmp_path,monkeypatch,Planner)
    predictions,row=group.run()
    assert budgets==[30] and row["travelers"][0]["steps"]==30
    assert predictions[0]["plan"] is not None
