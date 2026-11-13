from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

from .config import load_config
from .secrets import load_env, redact


def main(argv=None):
    parser=argparse.ArgumentParser(description="Constraint memory experiments on pinned MemoryArena through Polza.ai")
    sub=parser.add_subparsers(dest="command",required=True)
    check=sub.add_parser("check",help="Validate pinned sources, data, tools and config without paid calls")
    check.add_argument("--config",default="configs/travel.yaml")
    probe=sub.add_parser("probe",help="Verify exact model access with one tiny paid completion")
    probe.add_argument("--config",default="configs/travel.yaml")
    run=sub.add_parser("run",help="Run matched treatments on the same frozen groups")
    run.add_argument("--config",default="configs/travel.yaml")
    run.add_argument("--variants",nargs="+",default=None)
    run.add_argument("--limit",type=int)
    run.add_argument("--run-id")
    run.add_argument("--variant-config",help="Merge one variant overlay into the shared config")
    run.add_argument("--partition",choices=["smoke","dev","validation","holdout","all"])
    replay=sub.add_parser("replay",help="Verify causal memory replay without model calls")
    replay.add_argument("run_dir")
    report=sub.add_parser("report",help="Produce paired statistics and cost reports from official scores")
    report.add_argument("run_dir")
    args=parser.parse_args(argv)
    load_env()
    try:
        if args.command in {"run","check","probe"}:
            config=load_config(args.config,getattr(args,"variant_config",None))
            if getattr(args,"variant_config",None) and args.variants and len(args.variants)>1:
                raise ValueError("A variant overlay applies to one condition; compare separate runs to avoid confounded switches")
        if args.command=="check":
            from .benchmark_adapter import verify_readiness
            readiness=verify_readiness(Path.cwd())
            from .config import verify_tokenizer
            _,manifest=verify_tokenizer(Path.cwd())
            print(json.dumps({"status":"ready","model":config["agent"]["model"],
                              "groups":readiness["provenance"]["group_count"],
                              "base_families":readiness["provenance"]["base_family_count"],
                              "database_complete":readiness["database"]["complete"],
                              "tokenizer_manifest":manifest,"api_connectivity":"not checked"},indent=2))
        elif args.command=="probe":
            from .polza import PolzaClient
            client=PolzaClient(model=config["agent"]["model"],max_tokens=128,timeout=20,retries=0,reasoning={"enabled":False})
            catalog=client.verify_model()
            raw=client.complete([{"role":"user","content":"Reply READY only."}])
            print(json.dumps({"model":client.model_name,"catalog":catalog,"usage":client.get_usage_stats(),
                              "content":raw["choices"][0]["message"].get("content")},ensure_ascii=False))
        elif args.command=="run":
            if args.partition:
                config["experiment"]["partition"]=args.partition
            if args.limit is not None and args.limit<=0:
                raise ValueError("--limit must be positive")
            from .orchestration.runner import run_experiment
            print(run_experiment(config,variants=args.variants,limit=args.limit,run_id=args.run_id))
        elif args.command=="replay":
            from .orchestration.replay import replay_run
            print(json.dumps(replay_run(args.run_dir),indent=2))
        elif args.command=="report":
            from .evaluation import report_run
            print(json.dumps(report_run(args.run_dir),default=str,indent=2))
    except Exception as error:
        print(f"{type(error).__name__}: {redact(str(error))}",file=sys.stderr)
        return 2
    return 0


if __name__=="__main__":
    raise SystemExit(main())
