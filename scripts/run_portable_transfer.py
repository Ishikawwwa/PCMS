"""Credential-free preparation by default; --live runs the separate Polza diagnostic."""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from constraint_memory.config import load_config,verify_tokenizer,sha256
from constraint_memory.orchestration.logging import write_json,append_jsonl
from constraint_memory.polza import PolzaClient
from constraint_memory.secrets import load_env,redact
from constraint_memory.transfer import context_for,messages_for,evaluate_selection
from run_benchmark_suite import SuiteLock


def verify_freeze(cases_path):
    pin=json.loads((ROOT/"reports/portable_transfer_development_freeze.json").read_text(encoding="utf-8"))
    if pin.get("cases_sha256")!=sha256(cases_path): raise ValueError("Task input differs from the tested transfer freeze")
    current={path.relative_to(ROOT).as_posix():sha256(path) for folder in ("src","tests") for path in (ROOT/folder).rglob("*.py")}
    if current!=pin.get("source_and_tests_sha256"): raise ValueError("Source/tests differ from tested transfer freeze")
    for path,digest in pin.get("inputs_sha256",{}).items():
        target=(ROOT/path).resolve()
        if not target.is_relative_to(ROOT) or sha256(target)!=digest: raise ValueError("Transfer input changed: "+path)
    if not pin.get("tests",{}).get("passed"): raise ValueError("Transfer freeze lacks passing tests")
    return pin


def _main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases",default="data/transfer/development.json")
    parser.add_argument("--live",action="store_true",help="Make paid model requests; default is preparation only")
    parser.add_argument("--run-id")
    args=parser.parse_args(argv)
    raw=(ROOT/args.cases).read_bytes()
    cases=json.loads(raw)
    if not isinstance(cases,list) or not cases: parser.error("Provide a nonempty task input list")
    if len({case["case_id"] for case in cases})!=len(cases): parser.error("Cases must have unique IDs")
    tokenizer_path,_=verify_tokenizer(ROOT)
    from tokenizers import Tokenizer
    tokenizer=Tokenizer.from_file(str(tokenizer_path))
    count=lambda value:len(tokenizer.encode(value).ids)
    budget=4096
    prepared=[{"case_id":case["case_id"],"domain":case["domain"],"format":case["format"],
               "contexts":{condition:context_for(case,condition,count,budget) for condition in ("HISTORY","CONSTRAINT")}} for case in cases]
    if not args.live:
        summary={"status":"prepared_no_model_calls","cases":len(cases),
            "domains":sorted({case["domain"] for case in cases}),"formats":sorted({case["format"] for case in cases}),
            "cases_sha256":hashlib.sha256(raw).hexdigest(),"model_transfer_measured":False,
            "contexts":[{**row,"contexts":{key:{field:value for field,value in item.items() if field!="context"}
                         for key,item in row["contexts"].items()}} for row in prepared]}
        write_json(ROOT/"reports/portable_transfer_preparation.json",summary)
        print(json.dumps({key:value for key,value in summary.items() if key!="contexts"},indent=2))
        return 0
    freeze=verify_freeze(ROOT/args.cases)
    run_id=args.run_id or "portable-transfer-"+dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+",run_id) or run_id in {".",".."}: parser.error("Invalid run ID")
    destination=ROOT/"runs"/run_id
    if destination.exists(): parser.error("Run exists; use a fresh ID")
    destination.mkdir(parents=True)
    config=load_config(ROOT/"configs/travel.yaml")
    agent=config["agent"]
    manifest={"status":"prepared","run_id":run_id,"cases_sha256":hashlib.sha256(raw).hexdigest(),
              "model":agent["model"],"routing":agent["routing"],"memory_budget":budget,
              "agent_settings":agent,"seed":config["experiment"]["seed"],
              "max_model_calls_per_case_condition":1,"model_transfer_measured":False,
              "source_sha256":{path.relative_to(ROOT).as_posix():sha256(path) for path in (ROOT/"src").rglob("*.py")},
              "script_sha256":sha256(Path(__file__)),"limitations":["development synthetic tasks",
                  "structured input contracts; no natural-language extraction measurement",
                  "shared-core scoring needs independent review","not a MAGE comparison"]}
    write_json(destination/"manifest.json",manifest)
    write_json(destination/"task_inputs.json",cases)
    with zipfile.ZipFile(destination/"frozen_inputs.zip","w",compression=zipfile.ZIP_DEFLATED) as archive:
        for path in [*freeze["source_and_tests_sha256"],*freeze["inputs_sha256"]]:
            archive.write(ROOT/path,path)
    manifest["frozen_inputs_sha256"]=sha256(destination/"frozen_inputs.zip")
    write_json(destination/"manifest.json",manifest)
    load_env(ROOT/".env")
    rows=[]
    try:
        with PolzaClient(model=agent["model"],provider=agent["routing"],reasoning=agent["reasoning"],
             temperature=0,seed=config["experiment"]["seed"],max_tokens=8192,stream=agent["stream"],
             max_total_tokens=200000,max_cost_rub=20,timeout=agent["request_timeout_seconds"],
             usage_path=destination/"usage.jsonl") as client:
            client.verify_model(snapshot_path=ROOT/"reports/polza_catalog_verified_connection.json")
            manifest["status"]="running"
            write_json(destination/"manifest.json",manifest)
            for index,(case,ready) in enumerate(zip(cases,prepared)):
                conditions=("HISTORY","CONSTRAINT") if index%2==0 else ("CONSTRAINT","HISTORY")
                for condition in conditions:
                    if verify_freeze(ROOT/args.cases)!=freeze:
                        raise ValueError("Transfer freeze changed after preparation")
                    client.set_scope("transfer-suite",case["case_id"]+":"+condition)
                    context=ready["contexts"][condition]
                    row={"case_id":case["case_id"],"domain":case["domain"],"format":case["format"],"condition":condition,"context":context}
                    if context["status"]=="BUDGET_OVERFLOW": row.update(status="BUDGET_OVERFLOW",passed=False)
                    else:
                        messages=messages_for(case,context)
                        append_jsonl(destination/"requests.jsonl",{"case_id":case["case_id"],"condition":condition,"messages":messages})
                        raw_response=client.complete(messages,json_output=True,operation="agent")
                        content=raw_response["choices"][0]["message"]["content"]
                        append_jsonl(destination/"committed_responses.jsonl",{"case_id":case["case_id"],"condition":condition,
                            "response_text":content,"request_id":raw_response.get("id"),"model":raw_response.get("model"),
                            "usage":client.records[-1] if client.records else None})
                        try: response=json.loads(content)
                        except (TypeError,ValueError): response=None
                        row.update(messages=messages,response_text=content,**evaluate_selection(case,response))
                    rows.append(row)
                    append_jsonl(destination/"results.jsonl",row)
                    print(f"{case['case_id']} {condition}: {row['status']} passed={row['passed']}",flush=True)
            if verify_freeze(ROOT/args.cases)!=freeze:
                raise ValueError("Transfer freeze changed before completion")
            manifest.update(status="complete",model_transfer_measured=True,usage=client.get_usage_stats(),
                passed={condition:sum(row["passed"] for row in rows if row["condition"]==condition) for condition in ("HISTORY","CONSTRAINT")})
    except Exception as error:
        manifest.update(status="failed",error=redact(str(error)),completed_conditions=len(rows))
        print("STOPPED "+run_id+": "+redact(str(error)),flush=True)
        return 1
    finally: write_json(destination/"manifest.json",manifest)
    print("COMPLETED "+run_id,flush=True)
    return 0


def main(argv=None):
    arguments=list(sys.argv[1:] if argv is None else argv)
    if "--live" not in arguments: return _main(arguments)
    with SuiteLock(ROOT): return _main(arguments)


if __name__=="__main__": raise SystemExit(main())
