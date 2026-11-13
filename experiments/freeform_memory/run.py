"""Prepare or execute the fixed-size free-form diagnostic; monetary stop is optional.

No annotations are opened during preparation or planning. The evaluator reads
them only after a durable actual-action record. Current frozen travel/structured
transfer sources stay unchanged; this experiment records its own source pins.
"""
from __future__ import annotations

import argparse
import datetime as dt
from decimal import Decimal
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import re
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"))
sys.path.insert(0,str(ROOT/"scripts"))
sys.path.insert(0,str(ROOT))

from constraint_memory.config import load_config,sha256,verify_tokenizer
from constraint_memory.polza import PolzaClient
from constraint_memory.secrets import load_env,redact
from run_benchmark_suite import SuiteLock
from experiments.freeform_memory.runtime import (
    prepare_context,writer_messages,planner_messages,commit_option,encoded,
)


def durable(path,value,*,append=False):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    data=json.dumps(redact(value),ensure_ascii=False,allow_nan=False)+"\n"
    if append:
        with path.open("a",encoding="utf-8",newline="\n") as stream:
            stream.write(data);stream.flush();os.fsync(stream.fileno())
    else:
        temporary=path.with_suffix(path.suffix+".writing")
        with temporary.open("w",encoding="utf-8",newline="\n") as stream:
            stream.write(data);stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,path)


def decode(content):
    def invalid(value): raise ValueError("Nonfinite JSON literal")
    value=json.loads(content,parse_constant=invalid)
    encoded(value)
    return value


def source_map(folder):
    return {path.relative_to(ROOT).as_posix():sha256(path) for path in sorted((ROOT/folder).rglob("*.py"))}


def verify_freeze():
    pin=json.loads((ROOT/"reports/freeform_transfer_development_freeze.json").read_text(encoding="utf-8"))
    if source_map("src")!=pin["source_sha256"] or source_map("experiments/freeform_memory")!=pin["prototype_sha256"]:
        raise ValueError("Free-form source differs from the tested freeze")
    for relative,digest in pin["runtime_inputs_sha256"].items():
        target=(ROOT/relative).resolve()
        if not target.is_relative_to(ROOT) or sha256(target)!=digest:
            raise ValueError("Free-form runtime input changed: "+relative)
    if pin.get("verification",{}).get("passed",0)<1: raise ValueError("Passing prototype verification is required")
    return pin


def load_after_commit(destination,pin,case,response):
    # The action artifact is required independently of the caller's intention.
    actions=destination/"actions.jsonl"
    rows=[json.loads(line) for line in actions.read_text(encoding="utf-8").splitlines()]
    if not rows or rows[-1]["case_id"]!=case["case_id"] or rows[-1]["response"]!=response:
        raise ValueError("Offline annotations require an already durable current action")
    path=ROOT/"data/transfer/freeform_annotations.json"
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=pin["annotations_sha256"]:
        raise ValueError("Offline annotation changed after the development freeze")
    annotations=json.loads(raw)
    annotation=next(row for row in annotations["cases"] if row["case_id"]==case["case_id"])
    from experiments.freeform_memory.evaluator import grade_case
    return grade_case(case,annotation,response),raw


def request(client,destination,case,condition,phase,messages,*,freeze=None):
    if freeze is not None and verify_freeze()!=freeze:
        raise ValueError("Free-form freeze changed before model dispatch")
    attempt_id=f"{case['case_id']}:{condition}:{phase}"
    durable(destination/"request_journal.jsonl",{"event":"INTENT_ONLY_NOT_DISPATCH_PROOF",
        "attempt_id":attempt_id,"case_id":case["case_id"],"condition":condition,"phase":phase,"messages":messages},append=True)
    before=len(client.records)
    try:
        raw=client.complete(messages,json_output=True,operation="extraction" if phase=="writer" else "agent")
    except Exception as error:
        durable(destination/"request_journal.jsonl",{"event":"REQUEST_FAILED","attempt_id":attempt_id,
            "request_sent":getattr(error,"request_sent",None),"error":redact(str(error)),
            "new_usage":client.records[before:]},append=True)
        raise
    durable(destination/"committed_responses.jsonl",{"attempt_id":attempt_id,"case_id":case["case_id"],
        "condition":condition,"phase":phase,"raw_response":raw,
        "request_id":raw.get("id") if isinstance(raw,dict) else None,
        "model":raw.get("model") if isinstance(raw,dict) else None,"usage":client.records[before:]},append=True)
    durable(destination/"request_journal.jsonl",{"event":"RESPONSE_COMMITTED","attempt_id":attempt_id,
        "request_id":raw.get("id") if isinstance(raw,dict) else None,
        "model":raw.get("model") if isinstance(raw,dict) else None,"usage":client.records[before:]},append=True)
    return raw["choices"][0]["message"]["content"]


def _main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live",action="store_true")
    parser.add_argument("--budget-rub",type=Decimal,help="Optional user-chosen monetary stop; disabled by default")
    parser.add_argument("--run-id")
    args=parser.parse_args(argv)
    if args.budget_rub is not None and (not args.budget_rub.is_finite() or args.budget_rub<=0
            or not math.isfinite(float(args.budget_rub)) or float(args.budget_rub)<=0):
        parser.error("If supplied, --budget-rub must be finite, positive and representable by the provider client")
    cases_path=ROOT/"data/transfer/freeform_development.json"
    cases=json.loads(cases_path.read_text(encoding="utf-8"))
    if not isinstance(cases,list) or not cases or len({row["case_id"] for row in cases})!=len(cases):
        raise ValueError("Distinct nonempty free-form task cases required")
    tokenizer_path,_=verify_tokenizer(ROOT)
    from tokenizers import Tokenizer
    tokenizer=Tokenizer.from_file(str(tokenizer_path))
    count=lambda value:len(tokenizer.encode(value).ids)
    prepared=[]
    for case in cases:
        context=prepare_context(case,"FACT_LEDGER",count)
        writer=writer_messages(case)
        planner=planner_messages(case,context) if context["status"]!="BUDGET_OVERFLOW" else None
        prepared.append({"case_id":case["case_id"],"domain":case["domain"],"ledger_status":context["status"],
            "ledger_memory_tokens":context["required_tokens"],"writer_message_tokens":count(encoded(writer)),
            "ledger_planner_message_tokens":count(encoded(planner)) if planner else None})
    if not args.live:
        report={"status":"prepared_no_model_calls","cases":len(cases),"source_task_sha256":sha256(cases_path),
                "conditions":["FACT_LEDGER","ADVISORY_IR"],"annotations_loaded":False,
                "model_transfer_measured":False,"cases_prepared":prepared}
        durable(ROOT/"reports/freeform_transfer_preparation.json",report)
        print(json.dumps({key:value for key,value in report.items() if key!="cases_prepared"},indent=2))
        return 0
    pin=verify_freeze()
    run_id=args.run_id or "freeform-transfer-"+dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+",run_id) or run_id in {".",".."}: parser.error("Invalid run ID")
    destination=ROOT/"runs"/run_id
    if destination.exists(): parser.error("Use a fresh run ID; paid runs are never silently repeated")
    destination.mkdir(parents=True)
    config=load_config(ROOT/"configs/travel.yaml")
    agent=config["agent"]
    manifest={"run_id":run_id,"status":"prepared","model":agent["model"],"routing":agent["routing"],
        "python":sys.version,"platform":platform.platform(),
        "created_utc":dt.datetime.now(dt.timezone.utc).isoformat(),"conditions":["FACT_LEDGER","ADVISORY_IR"],
        "task_sha256":sha256(cases_path),"source_sha256":pin["source_sha256"],"prototype_sha256":pin["prototype_sha256"],
        "memory_tokens":4096,"planner_calls_per_case_condition":1,"writer_calls_per_advisory_condition":1,
        "effective_inference":{"temperature":0,"seed":config["experiment"]["seed"],"planner_max_tokens":8192,
            "writer_max_tokens":8192,"planner_reasoning":{"enabled":True},"writer_reasoning":{"enabled":False},
            "stream":agent["stream"],"timeout_seconds":agent["request_timeout_seconds"]},
        "suite_budget_rub":str(args.budget_rub) if args.budget_rub is not None else None,
        "suite_token_cap":None,"max_model_requests":3*len(cases),"model_transfer_measured":False,
        "annotations_visibility":"offline only after durable actual action; never writer/planner/recovery input",
        "limitations":["10 disclosed development workflows, not independent heldout data",
            "model predicate interpretations stay advisory","memory cap excludes shared current offers and other prompt tokens",
            "an optional user monetary threshold can be crossed by one in-flight response","not an authentic MAGE comparison"]}
    durable(destination/"manifest.json",manifest)
    durable(destination/"task_inputs.json",cases)
    with zipfile.ZipFile(destination/"frozen_inputs.zip","w",compression=zipfile.ZIP_DEFLATED) as archive:
        for path in [*pin["source_sha256"],*pin["prototype_sha256"],*pin["runtime_inputs_sha256"]]:
            archive.write(ROOT/path,path)
    load_env(ROOT/".env")
    results=[]
    client=None
    annotations_archived=False
    try:
        with PolzaClient(model=agent["model"],provider=agent["routing"],temperature=0,
            seed=config["experiment"]["seed"],max_tokens=8192,extraction_max_tokens=8192,
            reasoning={"enabled":True},extraction_reasoning={"enabled":False},stream=agent["stream"],
            # The existing client requires a numeric total-token threshold.
            # Infinity disables that aggregate stop; fixed cases/call counts
            # and per-call completion allowances bound the experiment.
            max_total_tokens=math.inf,max_cost_rub=float(args.budget_rub) if args.budget_rub is not None else None,
            timeout=agent["request_timeout_seconds"],
            usage_path=destination/"usage.jsonl") as client:
            client.verify_model(snapshot_path=ROOT/"reports/polza_catalog_verified_connection.json")
            manifest["status"]="running"
            durable(destination/"manifest.json",manifest)
            for index,case in enumerate(cases):
                order=("FACT_LEDGER","ADVISORY_IR") if index%2==0 else ("ADVISORY_IR","FACT_LEDGER")
                for condition in order:
                    if verify_freeze()!=pin: raise ValueError("Free-form freeze changed during execution")
                    client.set_scope("freeform-suite",case["case_id"]+":"+condition)
                    interpreted=None
                    parse_error=None
                    if condition=="ADVISORY_IR":
                        text=request(client,destination,case,condition,"writer",writer_messages(case),freeze=pin)
                        try: interpreted=decode(text)
                        except (TypeError,ValueError) as error: parse_error=str(error)
                    context=prepare_context(case,condition,count,interpretation=interpreted)
                    durable(destination/"contexts.jsonl",{"case_id":case["case_id"],"condition":condition,
                        "writer_parse_error":parse_error,"context":context},append=True)
                    if context["status"]=="BUDGET_OVERFLOW":
                        row={"case_id":case["case_id"],"condition":condition,"status":"BUDGET_OVERFLOW",
                             "passed":False,"source":"runtime boundary; no action or model failure inferred"}
                    else:
                        text=request(client,destination,case,condition,"planner",planner_messages(case,context),freeze=pin)
                        try: response=decode(text)
                        except (TypeError,ValueError): response=None
                        action=commit_option(case,response)
                        durable(destination/"actions.jsonl",{"case_id":case["case_id"],"condition":condition,
                            "response":response,"action":action},append=True)
                        grade,annotation_bytes=load_after_commit(destination,pin,case,response)
                        if not annotations_archived:
                            with zipfile.ZipFile(destination/"frozen_inputs.zip","a",compression=zipfile.ZIP_DEFLATED) as archive:
                                archive.writestr("data/transfer/freeform_annotations.json",annotation_bytes)
                            annotations_archived=True
                        if not grade["grading_valid"]: raise ValueError("Independent offline grading inputs are invalid")
                        if not grade["expected_labels_check"]["consistent"]:
                            raise ValueError("Offline annotation labels disagree with independently derived verdicts")
                        row={"case_id":case["case_id"],"condition":condition,"domain":case["domain"],
                             "status":grade["status"],"passed":grade["passed"],"grade":grade}
                    durable(destination/"results.jsonl",row,append=True)
                    results.append(row)
                    manifest.update(completed_conditions=len(results),frozen_inputs_sha256=sha256(destination/"frozen_inputs.zip"))
                    durable(destination/"manifest.json",manifest)
                    print(f"{case['case_id']} {condition}: {row['status']} passed={row['passed']}",flush=True)
            if verify_freeze()!=pin: raise ValueError("Free-form freeze changed before completion")
            manifest.update(status="complete",model_transfer_measured=True,usage=client.get_usage_stats(),
                passed={condition:sum(row["passed"] for row in results if row["condition"]==condition)
                        for condition in ("FACT_LEDGER","ADVISORY_IR")})
    except Exception as error:
        manifest.update(status="failed",error=redact(str(error)),completed_conditions=len(results),
                        frozen_inputs_sha256=sha256(destination/"frozen_inputs.zip"))
        if client is not None: manifest["usage"]=client.get_usage_stats()
        print("STOPPED "+run_id+": "+redact(str(error)),flush=True)
        return 1
    finally: durable(destination/"manifest.json",manifest)
    print("COMPLETED "+run_id,flush=True)
    return 0


def main(argv=None):
    arguments=list(sys.argv[1:] if argv is None else argv)
    if "--live" not in arguments: return _main(arguments)
    with SuiteLock(ROOT): return _main(arguments)


if __name__=="__main__": raise SystemExit(main())
