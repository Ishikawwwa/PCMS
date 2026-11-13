"""One causally visible development request; no planner, future questions or scores."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from constraint_memory.config import load_config
from constraint_memory.benchmark_adapter import load_turn
from constraint_memory.extraction import extract
from constraint_memory.polza import PolzaClient
from constraint_memory.secrets import load_env,redact
from constraint_memory.orchestration.logging import write_json

parser=argparse.ArgumentParser()
parser.add_argument("--run-id",default="qwen36-writer-nonthinking-g6-16k-20261014")
args=parser.parse_args()
load_env()
config=load_config("configs/travel.yaml")
agent=config["agent"]
directory=Path("runs/writer_probes")/args.run_id
if directory.exists(): raise SystemExit("Use a new diagnostic run ID")
directory.mkdir(parents=True)
client=PolzaClient(model=agent["model"],provider=agent["routing"],temperature=agent["temperature"],seed=config["experiment"]["seed"],
    max_tokens=agent["max_completion_tokens"],timeout=agent["request_timeout_seconds"],retries=2,stream=agent["stream"],
    reasoning=agent["reasoning"],extraction_reasoning=agent["extraction_reasoning"],extraction_max_tokens=agent["max_extraction_tokens"],usage_path=directory/"usage.jsonl")
try:
    packet=load_turn(Path.cwd(),"6",1)
    client.set_scope("writer-probe:6",packet["name"])
    event={"event_id":"6:1","group_id":"6","step":1,"type":"request","content":packet["query"]}
    result=extract(client,{"group_id":"6","traveler":packet["name"],"round_idx":1,"step":1,"query":packet["query"]},[event])
    output={"status":"complete","diagnostic_only":True,"model":agent["model"],"extraction_reasoning":agent["extraction_reasoning"],
            "extraction":result,"usage":client.get_usage_stats()}
    print({"parse_success":result["parse_success"],"constraints":len(result["constraints"]),"unresolved":len(result["unresolved"]),"usage":client.get_usage_stats()},flush=True)
except Exception as error:
    output={"status":"failed","diagnostic_only":True,"error":redact(str(error)),"usage":client.get_usage_stats(),"records":client.records}
    print(output,flush=True)
finally:
    client.close()
write_json(directory/"result.json",output)
