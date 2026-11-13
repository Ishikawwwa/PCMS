"""Bounded connectivity + authenticated Qwen completion, never prints credentials."""
import json
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from constraint_memory.polza import PolzaClient
from constraint_memory.secrets import load_env, redact
from constraint_memory.orchestration.logging import write_json

load_env()
parser=argparse.ArgumentParser()
parser.add_argument("--skip-catalog",action="store_true",help="Directly test the exact configured model; successful inference verifies availability")
args=parser.parse_args()
client = PolzaClient(timeout=15, retries=2, max_tokens=128,
                     usage_path="runs/provider_probe/usage.jsonl", reasoning={"enabled":False})
try:
    catalog = client.verify_model() if not args.skip_catalog else {"verification":"exact model inference probe"}
    response = client.complete([{"role":"user","content":"Reply with the word READY only."}])
    outcome={"status":"ready","model":client.model_name,"catalog":catalog,
             "response":response["choices"][0]["message"]["content"],"usage":client.get_usage_stats()}
except Exception as error:
    outcome={"status":"blocked","model":client.model_name,"error":redact(str(error))}
finally:
    client.close()
outcome["transport_diagnostics"]=client.transport_diagnostics
write_json("runs/provider_probe/result.json", outcome)
print(json.dumps(outcome,ensure_ascii=False))
sys.exit(0 if outcome["status"]=="ready" else 2)
