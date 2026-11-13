"""Prepare or run the current-source E1/E2 comparison on seven dev families.

Preparation is offline and opens no request turns, hidden answers or credentials.
Live execution uses native causal sessions. Scoring starts after all planning.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from constraint_memory.benchmark_adapter import score_official, verify_readiness
from constraint_memory.benchmark_adapter.data import load_splits
from constraint_memory.config import load_config, sha256, validate_config, verify_tokenizer
from constraint_memory.memory import create_backend
from constraint_memory.orchestration.group_session import GroupSession
from constraint_memory.orchestration.runner import format_base
from constraint_memory.polza import PolzaClient, BudgetExceeded
from constraint_memory.secrets import load_env, redact
from archive_run_source import archive_run_source
from run_benchmark_suite import SuiteLock

VARIANTS = ("E1", "E2")
FREEZE = "reports/cross_family_mechanism/source_freeze.json"
SELECTION = "reports/cross_family_mechanism/selection.json"
CATALOG = "reports/polza_catalog_verified_connection.json"


def durable(path, value, *, append=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(redact(value), ensure_ascii=False, allow_nan=False) + "\n"
    if append:
        with path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    else:
        temporary = path.with_suffix(path.suffix + ".writing")
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)


def source_map(root, directory):
    return {p.relative_to(root).as_posix(): sha256(p) for p in sorted((root / directory).rglob("*.py"))}


def feature_contrast(config):
    arms = {v: create_backend(v, config["memory"]) for v in VARIANTS}
    if any(type(b).__name__ != "LedgerMemory" for b in arms.values()):
        raise ValueError("Both arms must use plain LedgerMemory")
    features = {v: arms[v].features for v in VARIANTS}
    differences = [key for key in features["E1"] if features["E1"][key] != features["E2"][key]]
    if differences != ["dependency_closure"] or features["E1"]["dependency_closure"] is not False:
        raise ValueError("Only dependency_closure may differ between E1 and E2")
    if not features["E2"]["dependency_closure"]:
        raise ValueError("E2 dependency closure was disabled")
    for key in ("protected_compression", "glossary_hops", "deterministic_validation", "targeted_recovery"):
        if any(f[key] for f in features.values()):
            raise ValueError("This screen excludes additional mechanisms: " + key)
    return features


def validate_effective(config, memory_tokens=32768):
    # The unchanged production validator requires a positive aggregate token cap.
    # A validation-only sentinel preserves its other structural/causal checks.
    # The actual effective config and client use no aggregate stop.
    shadow = copy.deepcopy(config)
    shadow["agent"]["max_total_tokens_per_group"] = 1
    validate_config(shadow)
    agent, experiment, memory = config["agent"], config["experiment"], config["memory"]
    exact = {"model": "qwen/qwen3.6-27b", "provider": "polza", "base_url": "https://polza.ai/api/v1",
             "temperature": 0, "max_steps_per_traveler": 30, "max_completion_tokens": 16384,
             "max_extraction_tokens": 16384, "max_agent_visible_memory_tokens": memory_tokens,
             "max_total_tokens_per_group": None, "max_cost_rub_per_group": None,
             "reasoning": {"enabled": True}, "extraction_reasoning": {"enabled": False},
             "routing": {"only": ["chutes/fp8"], "allow_fallbacks": False}, "stream": True}
    if type(memory_tokens) is not int or memory_tokens not in (8192, 32768):
        raise ValueError("Only the primary 32768 or separately declared 8192 sensitivity is supported")
    for key, value in exact.items():
        if agent.get(key) != value:
            raise ValueError("Effective agent setting differs from frozen protocol: " + key)
    if (experiment.get("partition") != "dev" or experiment.get("seed") != 20261013
            or experiment.get("primary_contrast") != list(VARIANTS) or experiment.get("primary_endpoint") != "SR"
            or experiment.get("stop_on_extraction_failure") is not False
            or experiment.get("declared_inheritance_contract") is not True
            or experiment.get("tool_city_state_annotation_normalization") is not True
            or experiment.get("tool_price_unit") != "USD"):
        raise ValueError("Effective experiment protocol changed")
    if (memory.get("backend") != "E1" or memory.get("parser") != "llm"
            or memory.get("public_base_outside_memory") is not True
            or memory.get("compression_mode") != "merge" or memory.get("destructive_removal", False)):
        raise ValueError("Effective shared memory protocol changed")
    return feature_contrast(config)


def build_protocol(root: Path, memory_tokens: int = 32768) -> dict:
    root = Path(root).resolve()
    selection = json.loads((root / SELECTION).read_text(encoding="utf-8"))
    splits = load_splits(root)
    if sha256(root / "data/manifests/splits.json") != selection["split"]["sha256"]:
        raise ValueError("Frozen selection and current splits differ")
    groups = selection["group_ids"]
    if len(groups) != 7 or len(set(groups)) != 7 or selection["conditions"] != list(VARIANTS):
        raise ValueError("Exactly seven distinct frozen development groups and E1/E2 are required")
    families = []
    for item in selection["selected_groups"]:
        group = item["group_id"]
        meta = splits["groups"][group]
        if (group not in groups or int(group) not in splits["partitions"]["dev"]
                or meta["traveler_count"] != 7 or item["traveler_count"] != 7
                or meta["base_template_id"] != item["base_template_id"]
                or meta["base_template_id"] in selection["exposure_audit"]["conservatively_excluded_family_ids"]):
            raise ValueError("Frozen group metadata or exposure differs")
        families.append(meta["base_template_id"])
    if len(families) != 7 or len(set(families)) != 7:
        raise ValueError("Seven distinct development families are required")
    schedule = []
    for index, group in enumerate(groups):
        order = VARIANTS if index % 2 == 0 else VARIANTS[::-1]
        for variant in order:
            schedule.append({"group_id": group, "variant": variant})
    selected_schedule = [{"group_id": item["group_id"], "variant": item["condition"]} for item in selection["schedule"]]
    if schedule != selected_schedule:
        raise ValueError("Frozen balanced schedule differs")
    config = load_config(root / "configs/travel.yaml")
    config["experiment"].update(id="cross_family_closure", partition="dev", candidate="E2",
        primary_contrast=list(VARIANTS), primary_endpoint="SR", declared_inheritance_contract=True,
        tool_city_state_annotation_normalization=True, tool_price_unit="USD", stop_on_extraction_failure=False)
    config["agent"].update(max_completion_tokens=16384, max_extraction_tokens=16384,
        max_agent_visible_memory_tokens=memory_tokens, max_total_tokens_per_group=None, max_cost_rub_per_group=None)
    config["memory"].update(backend="E1", destructive_removal=False)
    features = validate_effective(config, memory_tokens)
    commitments = sum(splits["groups"][group]["traveler_count"] for group in groups) * len(VARIANTS)
    return {"schema_version": 1, "label": "primary" if memory_tokens == 32768 else "memory_8192_sensitivity",
        "config": config, "group_ids": groups, "family_ids": families, "schedule": schedule,
        "selection_sha256": sha256(root / SELECTION), "features": features,
        "bounds": {"group_conditions": len(schedule), "added_traveler_commitments": commitments,
                   "max_actor_requests": 30 * commitments, "max_writer_requests": commitments,
                   "max_total_model_requests": 31 * commitments},
        "effective_policy": {"monetary_stop_enabled": False, "aggregate_token_stop_enabled": False,
            "max_cost_rub_per_group": None, "max_total_tokens_per_group": None,
            "validation_only_aggregate_token_sentinel": 1, "artificial_padding": False,
            "memory_payload_tokens": memory_tokens, "public_base_outside_memory": True,
            "memory_ceiling_is_not_total_model_context_ceiling": True,
            "scoring": "Separate official scorer only after all planned sessions finish; no actor feedback"},
        "estimand": "Total effect of the current closure retrieval policy, including its seed/binding behavior",
        "limitations": ["Development screen, one group per family and one seed; no heldout inference",
            "Same writer settings do not guarantee identical outputs or histories after actions diverge",
            "32768 is a generous memory ceiling chosen from earlier trace feasibility, not a tuned score",
            "E1/E2 share unprotected compaction; required records can still be omitted at either ceiling",
            "No prior local live evidence in selected families; public smoke exposure is disclosed in selection",
            "USD is a shared caller declaration, not independent verification of database currency",
            "Not an authentic MAGE comparison; provider weights and runtime remain unverified"]}


def verify_freeze(root: Path):
    root = Path(root).resolve()
    pin = json.loads((root / FREEZE).read_text(encoding="utf-8"))
    for field, directory in (("source_sha256", "src"), ("experiment_sha256", "experiments/cross_family")):
        if source_map(root, directory) != pin[field]:
            raise ValueError("Source differs from tested cross-family freeze: " + directory)
    for relative, expected in pin["runtime_inputs_sha256"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or sha256(path) != expected:
            raise ValueError("Cross-family frozen input changed: " + relative)
    if pin.get("verification", {}).get("passed", 0) < 1:
        raise ValueError("Passing experimental verification is required before inference")
    return pin


class JournalClient(PolzaClient):
    """Native transport/accounting with a durable intent and raw-response journal."""
    def __init__(self, *args, destination, freeze_check=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.destination = Path(destination)
        self.freeze_check = freeze_check
        self.attempt_number = 0
        self.current_attempt = None
        self.call_counts = {}
        self.unaccounted_attempts = 0

    def complete(self, messages, tools=None, operation=None, max_tokens=None, json_output=False, response_schema=None):
        if self.freeze_check is not None:
            self.freeze_check()
        op = operation or self.operation
        # Native GroupSession/PlannerWorker enforce 30 actor calls per arrival.
        # The outer guard uses the fixed seven-arrival session allowance so
        # repeated traveler names do not share an accidental one-call limit.
        key = (self.group_id, op)
        limit = 7 if op == "extraction" else 210
        if self.group_id is None or self.traveler is None or op not in ("agent", "extraction"):
            raise ValueError("Every model request requires an explicit planned session/traveler scope")
        if self.call_counts.get(key, 0) >= limit:
            raise ValueError("Fixed native seven-arrival session request allowance exceeded")
        self.call_counts[key] = self.call_counts.get(key, 0) + 1
        self.attempt_number += 1
        self.current_attempt = f"request-{self.attempt_number:05d}"
        common = {"attempt_id": self.current_attempt, "group_id": self.group_id, "traveler": self.traveler, "operation": op}
        durable(self.destination / "request_journal.jsonl", {**common, "event": "INTENT_ONLY_NOT_DISPATCH_PROOF",
            "messages": messages, "tools": tools, "max_tokens_override": max_tokens,
            "json_output": json_output, "response_schema": response_schema}, append=True)
        before = len(self.records)
        try:
            raw = super().complete(messages, tools, operation, max_tokens, json_output, response_schema)
            durable(self.destination / "request_journal.jsonl", {**common, "event": "RESPONSE_ACCEPTED",
                "request_id": raw.get("id"), "usage": self.records[before:]}, append=True)
            return raw
        except BaseException as error:
            sent = False if isinstance(error, BudgetExceeded) else getattr(error, "request_sent", None)
            if len(self.records) == before and sent is not False:
                self.unaccounted_attempts += 1
            durable(self.destination / "request_journal.jsonl", {**common, "event": "REQUEST_FAILED",
                "error_type": type(error).__name__, "error": str(error),
                "request_sent": sent, "usage": self.records[before:]}, append=True)
            raise
        finally:
            self.current_attempt = None

    def _request(self, method, path, payload=None, authenticated=True):
        raw = super()._request(method, path, payload, authenticated)
        if method == "POST":
            # Persist before native identity/schema validation or tool-call parsing.
            durable(self.destination / "committed_responses.jsonl", {"attempt_id": self.current_attempt,
                "group_id": self.group_id, "traveler": self.traveler, "operation": self.operation,
                "raw_response": raw}, append=True)
        return raw

    def get_usage_stats(self):
        result = super().get_usage_stats()
        result.update(journal_attempts=self.attempt_number, unaccounted_request_attempts=self.unaccounted_attempts)
        if self.unaccounted_attempts:
            # An interrupt can escape before native accounting has a receipt.
            # Retain known sums, but never report unknown spend/tokens as zero.
            result.update(total_cost_rub=None, total_input_tokens=None, total_output_tokens=None)
        return result


def make_client(config, destination):
    agent = config["agent"]
    return JournalClient(destination=destination, model=agent["model"], base_url=agent["base_url"],
        temperature=agent["temperature"], max_tokens=agent["max_completion_tokens"],
        extraction_max_tokens=agent["max_extraction_tokens"], seed=config["experiment"]["seed"],
        reasoning=agent["reasoning"], extraction_reasoning=agent["extraction_reasoning"],
        provider=agent["routing"], stream=agent["stream"], timeout=agent["request_timeout_seconds"],
        retries=agent["retries"], max_total_tokens=math.inf, max_cost_rub=None,
        usage_path=Path(destination) / "usage.jsonl")


def run_study(root, destination, protocol, readiness, tokenizer, client, *, session_factory=GroupSession, scorer=score_official):
    root, destination = Path(root), Path(destination)
    config, groups = protocol["config"], protocol["group_ids"]
    validate_effective(config, config["agent"]["max_agent_visible_memory_tokens"])
    predictions = {v: [] for v in VARIANTS}
    session_rows = {v: [] for v in VARIANTS}
    for index, item in enumerate(protocol["schedule"], 1):
        variant, group = item["variant"], item["group_id"]
        print(f"SESSION {index}/{len(protocol['schedule'])} group={group} {variant}", flush=True)
        current, row = session_factory(root, destination, variant, group, config, readiness,
            tokenizer, client, None, format_base).run()
        if row["group_id"] != group or row["variant"] != variant:
            raise ValueError("Completed session identity differs from schedule")
        durable(destination / "session_completions.jsonl", {"execution_index": index,
            "group_id": group, "variant": variant, "predictions": current, "row": row}, append=True)
        predictions[variant].extend(current)
        session_rows[variant].append(row)
    # No hidden-score access before the complete fixed planning grid is durable.
    durable(destination / "planning_complete.json", {"sessions": len(protocol["schedule"]), "scoring_started": False})
    rows = []
    for variant in VARIANTS:
        scores = scorer(root / config["experiment"]["upstream_root"], root / "data/hidden/memoryarena_travel.jsonl",
            predictions[variant], groups, expected_dataset_sha=readiness["provenance"]["dataset_sha256"])
        durable(destination / f"official_scores_{variant}.json", scores)
        for row in session_rows[variant]:
            matches = [g for g in scores["per_group"] if str(g["group_id"]) == row["group_id"]]
            if len(matches) != 1 or matches[0]["base_template_id"] != row["base_template_id"]:
                raise ValueError("Official scorer group or family differs from completed causal session")
            group_score = matches[0]
            row["official_metrics"] = group_score["official_metrics"]
            row["official_metric_metadata"] = scores.get("official_metric_metadata", scores.get("metric_metadata", {}))
            people = {int(p["traveler_index"]) - 1: p for p in group_score.get("traveler_metrics", [])}
            row["travelers"] = [{**person, **people.get(person["traveler_index"], {}),
                "traveler_index": person["traveler_index"]} for person in row["travelers"]]
            row["official_metric_name"] = scores.get("metric_definition", {}).get("SPS", "upstream SPS")
            row["official_metric_version"] = config["experiment"]["upstream_repo_sha"]
            durable(destination / "results.jsonl", row, append=True)
            rows.append(row)
    return rows


def archive_protocol(root, destination, pin, protocol):
    files = {**pin["experiment_sha256"], **pin["runtime_inputs_sha256"], FREEZE: sha256(root / FREEZE)}
    bodies = {}
    for relative, expected in files.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Protocol archive path leaves workspace: " + relative)
        body = path.read_bytes()
        if hashlib.sha256(body).hexdigest() != expected:
            raise ValueError("Protocol archive differs from frozen input: " + relative)
        bodies[relative] = body
    with zipfile.ZipFile(destination / "experiment_inputs.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative, body in bodies.items():
            archive.writestr(relative, body)
    durable(destination / "experiment_inputs.json", {"members_sha256": files,
        "archive_sha256": sha256(destination / "experiment_inputs.zip")})
    durable(destination / "protocol.json", protocol)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--memory-tokens", type=int, choices=(32768, 8192), default=32768,
                        help="32768 primary; 8192 is a separately declared sensitivity, never run automatically")
    parser.add_argument("--run-id")
    args = parser.parse_args(argv)
    protocol = build_protocol(ROOT, args.memory_tokens)
    if not args.live:
        durable(ROOT / "reports/cross_family_mechanism/preparation.json", protocol)
        print(json.dumps({"status": "prepared_no_model_calls", "label": protocol["label"],
            "group_ids": protocol["group_ids"], "bounds": protocol["bounds"], "policy": protocol["effective_policy"]}, indent=2))
        return 0
    with SuiteLock(ROOT):
        pin = verify_freeze(ROOT)
        readiness = verify_readiness(ROOT)
        experiment = protocol["config"]["experiment"]
        for actual, configured in (("dataset_revision", "dataset_revision"), ("upstream_sha", "upstream_repo_sha")):
            if readiness["provenance"][actual] != experiment[configured]:
                raise ValueError("Protocol provenance differs from prepared source/data")
        tokenizer_path, tokenizer_manifest = verify_tokenizer(ROOT)
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        run_id = args.run_id or f"cross-family-{protocol['label']}-" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id) or run_id in {".", ".."}:
            parser.error("Invalid run ID")
        destination = ROOT / "runs" / run_id
        if destination.exists():
            parser.error("Use a fresh run ID; paid runs are never silently repeated")
        destination.mkdir(parents=True)
        manifest = {"run_id": run_id, "status": "prepared", "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "config": protocol["config"], "variants": list(VARIANTS), "group_ids": protocol["group_ids"],
            "schedule": protocol["schedule"], "bounds": protocol["bounds"], "effective_policy": protocol["effective_policy"],
            "source_sha256": pin["source_sha256"], "splits_sha256": sha256(ROOT / "data/manifests/splits.json"),
            "python": sys.version, "platform": platform.platform(), "pins": readiness["provenance"],
            "tokenizer_sha256": sha256(tokenizer_path), "tokenizer_manifest": tokenizer_manifest,
            "protocol_label": protocol["label"], "limitations": protocol["limitations"],
            "live_comparison_validated": False, "authentic_MAGE_comparison": False}
        durable(destination / "manifest.json", manifest)
        client = None
        try:
            archive_run_source(ROOT, destination)
            archive_protocol(ROOT, destination, pin, protocol)
            load_env(ROOT / ".env")
            with make_client(protocol["config"], destination) as client:
                def check():
                    if verify_freeze(ROOT) != pin:
                        raise ValueError("Cross-family freeze changed during execution")
                client.freeze_check = check
                manifest["model_catalog_snapshot"] = client.verify_model(snapshot_path=ROOT / CATALOG)
                manifest["model_catalog_provenance"] = client.catalog_provenance
                manifest["status"] = "running"
                durable(destination / "manifest.json", manifest)
                rows = run_study(ROOT, destination, protocol, readiness, tokenizer, client)
                check()
                from constraint_memory.orchestration.replay import replay_run
                replay = replay_run(destination)
                if len(replay) != protocol["bounds"]["group_conditions"]:
                    raise ValueError("Replay does not cover every planned group-condition")
                from constraint_memory.evaluation import report_run
                report_run(destination, requested_metrics=("SR", "PS", "SPS"), primary_contrast=VARIANTS,
                    primary_endpoint="SR", bootstrap_samples=10000, seed=experiment["seed"])
                manifest.update(status="complete", completed_conditions=len(rows), live_comparison_validated=True,
                    replay_all_match=True, usage=client.get_usage_stats())
        except BaseException as error:
            interrupted = isinstance(error, KeyboardInterrupt)
            manifest.update(status="interrupted" if interrupted else "failed",
                error={"type": type(error).__name__, "message": redact(str(error))})
            if client is not None:
                manifest["usage"] = client.get_usage_stats()
            print("STOPPED " + run_id + ": " + redact(str(error)), flush=True)
            return 130 if interrupted else 1
        finally:
            durable(destination / "manifest.json", manifest)
        print("COMPLETED " + run_id, flush=True)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
