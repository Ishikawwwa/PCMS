"""Matched causal benchmark execution with isolated official scoring."""
from __future__ import annotations
import copy
import datetime as dt
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

from constraint_memory.benchmark_adapter import (PlannerWorker, load_base, load_splits, load_turn,
    selected_group_ids, verify_readiness, score_official)
from constraint_memory.benchmark_adapter.data import ChronologicalAdapter
from constraint_memory.benchmark_adapter.submissions import SubmissionLedger
from constraint_memory.config import validate_config, sha256, verify_tokenizer
from constraint_memory.extraction import extract, parse_submission, grounded_attributes
from constraint_memory.memory import create_backend
from constraint_memory.polza import PolzaClient, BudgetExceeded
from .logging import append_jsonl, read_jsonl, write_json
from .group_session import GroupSession


def format_base(base):
    lines = [f"=== {base['name']}'s Plan ==="]
    for day in base["daily_plans"]:
        lines.append(f"Day {day.get('days', day.get('day'))}:")
        for key in ("current_city", "transportation", "breakfast", "attraction", "lunch", "dinner", "accommodation"):
            lines.append(f"{key.replace('_', ' ').title()}: {day.get(key, '-')}")
    return "\n".join(lines)


def run_experiment(config, root=Path.cwd(), variants=None, limit=None, run_id=None, client=None, planner_factory=None):
    """Creates a fresh run; never mixes cached predictions across treatments."""
    validate_config(config)
    root = Path(root).resolve()
    readiness = verify_readiness(root)
    if readiness["provenance"]["dataset_revision"] != config["experiment"]["dataset_revision"]:
        raise ValueError("Configured dataset differs from prepared dataset")
    if readiness["provenance"]["upstream_sha"] != config["experiment"]["upstream_repo_sha"]:
        raise ValueError("Configured upstream differs from prepared source")
    variants = variants or [config["memory"]["backend"]]
    for variant in variants:
        create_backend(variant)  # Fail unsupported/authentic MAGE before network calls.
    if {"E3I","E3R"}&set(variants) and not config["experiment"].get("declared_inheritance_contract",False):
        raise ValueError("E3I/E3R requires a shared declared inheritance contract before model calls")
    if "E3R" in variants and config["experiment"].get("tool_city_state_annotation_normalization") is not True:
        raise ValueError("E3R requires a shared declared city tool adapter before model calls")
    if "E3R" in variants and config["experiment"].get("tool_price_unit")!="USD":
        raise ValueError("E3R requires a shared declared price schema before model calls")
    groups = selected_group_ids(root, config["experiment"]["partition"])
    if limit:
        groups = groups[:limit]
    if not groups:
        raise ValueError("No groups selected")
    run_id = run_id or dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = root / "runs" / run_id
    if run_dir.exists():
        raise ValueError("Run directory already exists; use a fresh run ID")
    run_dir.mkdir(parents=True)
    cfg_agent = config["agent"]
    client = client or PolzaClient(model=cfg_agent["model"], base_url=cfg_agent["base_url"],
        temperature=cfg_agent["temperature"], max_tokens=cfg_agent["max_completion_tokens"],
        seed=config["experiment"]["seed"], max_total_tokens=cfg_agent["max_total_tokens_per_group"],
        max_cost_rub=cfg_agent["max_cost_rub_per_group"], timeout=cfg_agent["request_timeout_seconds"],
        retries=cfg_agent["retries"], reasoning=cfg_agent.get("reasoning"), provider=cfg_agent.get("routing"),
        stream=cfg_agent.get("stream",False),
        extraction_reasoning=cfg_agent.get("extraction_reasoning"),
        extraction_max_tokens=cfg_agent.get("max_extraction_tokens"),
        usage_path=run_dir / "usage.jsonl")
    manifest = {"run_id":run_id,"status":"preflight","config":copy.deepcopy(config),
                "variants":variants,"group_ids":groups,"created_utc":dt.datetime.now(dt.timezone.utc).isoformat(),
                "python":sys.version,"platform":platform.platform(),"pins":readiness["provenance"],
                "splits_sha256":sha256(root/config["experiment"]["splits_manifest"]),
                "source_sha256":{str(p.relative_to(root)):sha256(p) for p in sorted((root/"src").rglob("*.py"))},
                "inference_limitations":"Polza model alias and catalog pinned by snapshot; weight precision/runtime/revision are provider-managed. Not the paper's vLLM setup."}
    if hasattr(client,"usage_path"):
        client.usage_path=run_dir/"usage.jsonl"
    write_json(run_dir/"manifest.json",manifest)
    try:
        manifest["model_catalog_snapshot"] = client.verify_model()
        manifest["model_catalog_provenance"] = getattr(client,"catalog_provenance",{"mode":"not_reported"})
        tokenizer_path,tokenizer_manifest = verify_tokenizer(root)
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        manifest["tokenizer_sha256"] = sha256(tokenizer_path)
        manifest["tokenizer_manifest"] = tokenizer_manifest
        # Provider routing is kept fixed by config. A model alias alone cannot prove identical weights.
        manifest["status"]="running"
        write_json(run_dir/"manifest.json",manifest)
        rows=[]
        for variant in variants:
            predictions=[]
            variant_rows=[]
            for group_id in groups:
                group_predictions,row=GroupSession(root,run_dir,variant,group_id,config,readiness,tokenizer,
                    client,planner_factory,format_base).run()
                predictions.extend(group_predictions)
                variant_rows.append(row)
            scores=score_official(root/config["experiment"]["upstream_root"],root/"data/hidden/memoryarena_travel.jsonl",predictions,groups,
                                  expected_dataset_sha=readiness["provenance"]["dataset_sha256"])
            write_json(run_dir/f"official_scores_{variant}.json",scores)
            # Adapter's normalized per-group results are authoritative.
            for row in variant_rows:
                group_score=next(g for g in scores["per_group"] if str(g["group_id"])==row["group_id"])
                if group_score["base_template_id"] != row["base_template_id"]:
                    raise ValueError("Scorer base family differs from causally visible data")
                row["official_metrics"]=group_score["official_metrics"]
                row["official_metric_metadata"]=scores.get("official_metric_metadata",scores.get("metric_metadata",{}))
                scored_travelers={int(person["traveler_index"])-1:person for person in group_score.get("traveler_metrics",[])}
                row["travelers"]=[{**person,**scored_travelers.get(person["traveler_index"],{}),
                                   "traveler_index":person["traveler_index"]} for person in row["travelers"]]
                row["official_metric_name"]=scores.get("metric_definition",{}).get("SPS","upstream SPS")
                row["official_metric_version"]=config["experiment"]["upstream_repo_sha"]
                rows.append(row)
                append_jsonl(run_dir/"results.jsonl",row)
        manifest["status"]="complete"
        write_json(run_dir/"manifest.json",manifest)
        from constraint_memory.evaluation import report_run
        primary_baseline,primary_treatment=config["experiment"].get("primary_contrast",["E0","E2"])
        report_run(run_dir,requested_metrics=config["evaluation"]["metrics"],bootstrap_samples=config["evaluation"]["bootstrap_iterations"],
                   baseline=primary_baseline,treatment=primary_treatment,seed=config["experiment"]["seed"])
        return run_dir
    except Exception as error:
        manifest["status"]="blocked" if not read_jsonl(run_dir/"predictions.jsonl") else "failed"
        manifest["error"]={"type":type(error).__name__,"message":str(error)}
        write_json(run_dir/"manifest.json",manifest)
        raise
