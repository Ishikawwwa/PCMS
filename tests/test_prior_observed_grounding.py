from pathlib import Path
from tokenizers import Tokenizer
from constraint_memory.config import load_config
from constraint_memory.orchestration import group_session as module
from constraint_memory.orchestration.group_session import GroupSession
from constraint_memory.orchestration.runner import format_base

ROOT=Path(__file__).resolve().parents[1]


def test_new_visible_tool_rows_ground_an_actual_base_choice_without_rewriting_it(tmp_path,monkeypatch):
    class Adapter:
        def __init__(self,*args):
            self.base={"base_template_id":"family","base_person":{"name":"Base","query":"fixed public base",
                "daily_plans":[{"day":1,"dinner":"Exact Cafe, Boston"}]}}
            self.metadata={"traveler_count":2}
        def iter_requests(self):
            yield {"group_id":"g","traveler_index":1,"name":"Bob","query":"Choose my dinner"}
            yield {"group_id":"g","traveler_index":2,"name":"Carol","query":"Consider Base dinner price"}
        def commit_result(self,*args): pass
    class Client:
        records=[]
        def set_scope(self,*args): pass
    class Planner:
        def __init__(self,*args,**kwargs): pass
        def run_turn(self,packet,context,base):
            if packet["name"]=="Carol":
                assert '"attribute":"price"' in context and '"value":30' in context
            text="Day 1:\nDinner: Other Cafe, Boston"
            observation='[{"Name":"Exact Cafe","City":"Boston","Average Cost":30}]'
            return {"final_plan":f"=== {packet['name']}'s Plan ===\n{text}","submitted_text":text,
                "plan":[{"day":1,"dinner":"Other Cafe, Boston"}],"success":True,"total_steps":1,"parent_model_requests":1,
                "scratchpad":[{"tool_results":[{"tool_call_id":"t","result":observation}]}],
                "structured_tool_results":[{"tool_call_id":"t","tool_name":"RestaurantSearch","formatted_result":observation,
                    "displayed_rows":[{"Name":"Exact Cafe","City":"Boston","Average Cost":30}]}]}
    monkeypatch.setattr(module,"ChronologicalAdapter",Adapter)
    def extracted(client,request,events,*args):
        if request["traveler"]!="Carol": return {}
        return {"constraints":[{"kind":"RELATION","subject":{"traveler":"Carol","day":1,"slot":"dinner","attribute":"price"},
            "operator":"LTE","target":{"ref":"decision:Base:day1:dinner:price"},"source_refs":["event:"+events[0]["event_id"]]}]}
    monkeypatch.setattr(module,"extract",extracted)
    config=load_config(ROOT/"configs/travel.yaml")
    tokenizer=Tokenizer.from_file(str(ROOT/"external/qwen-tokenizer/tokenizer.json"))
    group=GroupSession(ROOT,tmp_path,"E3","g",config,{"database":{"database_path":"unused"}},tokenizer,Client(),Planner,format_base)
    group.run()
    base_choices=[r for r in group.backend.snapshot()["decisions"] if r["traveler"]=="Base" and r["slot"]=="dinner"]
    assert len([r for r in base_choices if r["attribute"]=="value"])==1
    facts=[r for r in base_choices if r["attribute"]=="price"]
    assert len(facts)==1 and facts[0]["value"]==30
    assert any(f"event:{row['event_id']}" in facts[0]["source_refs"] for row in group.backend.snapshot()["evidence"] if row["type"]=="tool_result")
