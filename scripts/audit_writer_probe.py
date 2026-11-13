"""Validate a completed causal writer response against each ledger boundary."""
import argparse
import collections
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from tokenizers import Tokenizer
from constraint_memory.config import load_config,verify_tokenizer
from constraint_memory.benchmark_adapter import load_base,load_turn
from constraint_memory.memory import create_backend
from constraint_memory.extraction import parse_submission
from constraint_memory.orchestration.runner import format_base
from constraint_memory.orchestration.logging import write_json

parser=argparse.ArgumentParser(); parser.add_argument("probe_directory"); args=parser.parse_args()
root=Path(__file__).resolve().parents[1]
probe=Path(args.probe_directory)
extraction=json.loads((probe/"result.json").read_text(encoding="utf-8"))["extraction"]
base=load_base(root,"6")["base_person"]
packet=load_turn(root,"6",1)
config=load_config(root/"configs/travel.yaml")
tokenizer=Tokenizer.from_file(str(verify_tokenizer(root)[0]))
output={"diagnostic_only":True,"constraints":len(extraction["constraints"]),"conditions":{}}
for variant in ("E0","E3","E4"):
    memory=create_backend(variant,{**config["memory"],"token_budget":config["agent"]["max_agent_visible_memory_tokens"]})
    memory.reset("6"); memory.set_token_counter(lambda text:len(tokenizer.encode(text).ids),name="Qwen3.6-27B pinned tokenizer")
    memory.observe({"group_id":"6","event_id":"6:0","step":0,"type":"request","content":base["query"]})
    memory.begin_subtask({"group_id":"6","traveler":base["name"],"step":0,"query":base["query"],"public_anchor_travelers":[base["name"]]})
    action=format_base(base)
    memory.commit_subtask({"group_id":"6","traveler":base["name"],"step":0,"action":action,"decisions":parse_submission(action)})
    memory.observe({"group_id":"6","event_id":"6:1","step":1,"type":"request","content":packet["query"]})
    context=memory.begin_subtask({"group_id":"6","traveler":packet["name"],"step":1,"query":packet["query"],
        "extraction":extraction,"public_anchor_travelers":[base["name"]]})
    output["conditions"][variant]={"status":context["status"],"context_tokens":context["token_count"],
        "stored_constraints":len(memory.snapshot()["constraints"]),"required_records":len(context["required_closure_ids"]),
        "returned_records":len(context["closure_ids"]),"diagnostic_codes":dict(collections.Counter(d["code"] for d in context["diagnostics"]))}
    if variant=="E4":
        output["conditions"][variant]["diagnostic_details"]=context["diagnostics"]
write_json(probe/"ledger_audit.json",output)
print(json.dumps(output,indent=2))
