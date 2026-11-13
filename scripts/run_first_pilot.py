"""Bounded development pilot with a verified live fixed Polza route."""
import json
import sys
import argparse
import datetime as dt
import re
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from constraint_memory.config import load_config
from constraint_memory.polza import PolzaClient
from constraint_memory.secrets import load_env,redact
from constraint_memory.orchestration.runner import run_experiment
from constraint_memory.orchestration.logging import write_json
from constraint_memory.memory import create_backend

load_env()
parser=argparse.ArgumentParser()
parser.add_argument("--run-id")
parser.add_argument("--variants",nargs="+",default=["E0","E3"])
parser.add_argument("--variant-config",help="Explicit candidate overlay; shared agent settings apply to every compared variant")
parser.add_argument("--limit",type=int,default=1)
parser.add_argument("--catalog-snapshot",help="Reuse a saved verified catalog; live exact-model inference still verifies availability")
args=parser.parse_args()
args.run_id=args.run_id or "qwen36-"+"-".join(args.variants)+"-"+dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
if not re.fullmatch(r"[A-Za-z0-9_.-]+",args.run_id) or args.run_id in {".",".."}:
    parser.error("Run ID must be a single safe directory name")
if args.limit<1:
    parser.error("Pilot group limit must be positive")
if len(set(args.variants)) != len(args.variants):
    parser.error("Each variant must appear once")
if (Path("runs")/args.run_id).exists() or (Path("runs/provider_preflights")/args.run_id).exists():
    raise SystemExit("Run ID already exists; use a fresh run ID before paid preflight")
config=load_config("configs/travel.yaml",overlay=args.variant_config)
if config["experiment"]["partition"] not in {"smoke","dev"}:
    parser.error("This writer-readiness pilot is development-only")
for variant in args.variants:
    create_backend(variant,config["memory"])
agent=config["agent"]
client=PolzaClient(model=agent["model"],provider=agent["routing"],reasoning=agent["reasoning"],
    temperature=agent["temperature"],seed=config["experiment"]["seed"],max_tokens=agent["max_completion_tokens"],
    timeout=agent["request_timeout_seconds"],retries=2,max_total_tokens=agent["max_total_tokens_per_group"],max_cost_rub=agent["max_cost_rub_per_group"],
    usage_path=Path("runs/provider_preflights")/args.run_id/"usage.jsonl")
client.stream=agent.get("stream",False)
client.extraction_reasoning=agent.get("extraction_reasoning")
client.extraction_max_tokens=agent.get("max_extraction_tokens",client.max_tokens)
try:
    print("Verifying exact Qwen catalog and fixed provider route...",flush=True)
    model=client.verify_model(snapshot_path=args.catalog_snapshot)
    probe=client.complete([{"role":"user","content":"Reply READY only."}],max_tokens=128,operation="preflight")
    if probe.get("model")!=agent["model"]:
        raise RuntimeError("Live preflight did not return the exact configured model")
    write_json(f"reports/{args.run_id}_preflight.json",{"model":model,"routing_requested":agent["routing"],
        "catalog_provenance":client.catalog_provenance,
        "returned_provider":probe.get("provider"),"model_loaded":probe.get("model"),"usage":client.get_usage_stats(),
        "request_id":probe.get("id"),
        "finish_reason":probe["choices"][0].get("finish_reason"),"transport_diagnostics":client.transport_diagnostics})
    print("Starting matched complete development groups for "+", ".join(args.variants)+" (common per-arrival step cap).",flush=True)
    if not args.variant_config:
        if len(args.variants)>1:
            config["experiment"]["primary_contrast"]=args.variants[:2]
        config["experiment"]["id"]="first_live_"+"_".join(args.variants)+"_development_pilot"
    config["experiment"]["stop_on_extraction_failure"]=True
    run=run_experiment(config,variants=args.variants,limit=args.limit,run_id=args.run_id,client=client)
    print("Completed: "+str(run),flush=True)
except Exception as error:
    write_json(f"reports/{args.run_id}_failure.json",{"status":"failed","error":redact(str(error)),
        "usage":client.get_usage_stats(),"records":client.records,"transport_diagnostics":client.transport_diagnostics})
    print(type(error).__name__+": "+redact(str(error)),flush=True)
    raise SystemExit(2)
finally:
    client.close()
