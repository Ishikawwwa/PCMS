"""Verify a fixed upstream route without substituting the required model."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from constraint_memory.polza import PolzaClient
from constraint_memory.secrets import load_env,redact
from constraint_memory.orchestration.logging import write_json
load_env()
routing={"only":["chutes/fp8"],"allow_fallbacks":False}
client=PolzaClient(max_tokens=128,timeout=20,retries=2,reasoning={"enabled":False},provider=routing,
    usage_path="runs/provider_route_probe/usage.jsonl")
try:
    response=client.complete([{"role":"user","content":"Reply READY only."}])
    result={"status":"ready","model":client.model_name,"routing_requested":routing,
        "routing_response":response.get("provider"),"content":response["choices"][0]["message"].get("content"),
        "usage":client.get_usage_stats(),"returned_top_level_fields":list(response)}
except Exception as error:
    result={"status":"blocked","model":client.model_name,"routing_requested":routing,"error":redact(str(error))}
finally:
    client.close()
write_json("reports/polza_route_probe.json",result)
print(json.dumps(result,ensure_ascii=False))
raise SystemExit(0 if result["status"]=="ready" else 2)
