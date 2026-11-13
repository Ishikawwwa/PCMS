"""Explicit, audited continuation of a stopped cross-family lineage.

--prepare runs tests and network-denied reconstruction, then pins that parent.
--live uses its pin, retains closed isolated sessions, replays the active prefix
and makes new requests only after the complete semantic boundary matches.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from experiments.cross_family_continuation import core
from experiments.cross_family_continuation.pacing import FreshConnectionTransport, PacedTransport
from experiments.cross_family_continuation.progress import ObservedClient, ObservedTransport, ProgressReporter, observe_session
from experiments.cross_family_recovery_second.transport import RecordingChatTransport

base = core.base
REPORTS = ROOT / "reports/cross_family_continuation"
DEFAULT_PARENT = "cross-family-continuation-20261017T124525520115Z"


def preparation(parent_id):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", parent_id) or parent_id in (".", ".."):
        raise ValueError("Invalid parent run ID")
    return REPORTS / "preparations" / parent_id


def source():
    return base.frozen.source_map(ROOT, "experiments/cross_family_continuation")


def verify_pin(parent):
    directory = preparation(parent["run_id"])
    pin = base.read(directory / "source_freeze.json")
    if (pin["parent_run_id"] != parent["run_id"] or source() != pin["source_sha256"]
            or parent["lineage_files_sha256"] != pin["lineage_files_sha256"]
            or parent.get("claim_files_sha256", {}) != pin.get("claim_files_sha256", {})
            or base.sha256(directory / "replay_verified.json") != pin["replay_verified_sha256"]
            or base.sha256(directory / "verification.json") != pin["verification_sha256"]
            or pin["unit_tests"]["exit_code"] != 0 or pin["unit_tests"]["passed"] < 1):
        raise base.ReplayMismatch("Prepared source, tests, lineage or replay evidence changed")
    return pin


def inherited_rows(parent, variant, group_id):
    scope = f"{variant}:{group_id}"
    return [{**copy.deepcopy(row), "inherited_failed_dispatch": True, "new_billable": False}
            for row in parent["all_failures"] if row["group_id"] == scope]


def recovery_disclosure(parent):
    failed_writers = sum(row.get("request_sent") is True and row.get("operation") == "extraction"
                         for row in parent["all_failures"])
    bounds = copy.deepcopy(parent["protocol"]["bounds"])
    effective = copy.deepcopy(bounds)
    effective["max_writer_requests"] += failed_writers
    effective["max_total_model_requests"] += failed_writers
    return {"classification": "infrastructure_recovered_diagnostic" if failed_writers else "original_dispatch_bounds",
        "original_dispatch_bound_satisfied": failed_writers == 0,
        "original_bounds": bounds, "effective_physical_dispatch_bounds": effective,
        "additional_sent_writer_allowance": failed_writers,
        "accepted_writers_per_arrival": 1,
        "policy": "Recover a writer with no accepted output; preserve every sent failure and unknown charge; disclose extra physical writer dispatches rather than fabricate missing memory"}


def execute_grid(destination, parent, tokenizer, readiness, client, *, session_factory=None, scorer=None, progress=None):
    session_factory = session_factory or base.frozen.GroupSession
    if progress is not None:
        progress.phase(f"Validating and retaining {parent['imported_model_responses']} closed-session receipts locally")
    completed, retained = core.import_completed(parent, destination, client)
    base.frozen.durable(destination / "closed_sessions_import.json", retained)
    protocol = copy.deepcopy(parent["protocol"])
    config = protocol["config"]
    config["agent"].update(stream=True, request_timeout_seconds=1800, retries=0)
    all_rows = {variant: [] for variant in base.frozen.VARIANTS}
    predictions = {variant: [] for variant in base.frozen.VARIANTS}
    for item in completed:
        print(f"RETAINED SESSION {item['execution_index']}/14 group={item['group_id']} {item['variant']} (closed isolated session)", flush=True)
        all_rows[item["variant"]].append(copy.deepcopy(item["row"]))
        predictions[item["variant"]].extend(copy.deepcopy(item["predictions"]))
    for index, item in enumerate(protocol["schedule"][len(completed):], len(completed) + 1):
        mode = "replay saved active prefix, then continue" if index == len(completed) + 1 else "new planning"
        print(f"SESSION {index}/14 group={item['group_id']} {item['variant']} ({mode})", flush=True)
        if progress is not None:
            progress.session(index, item["variant"], item["group_id"])
        session = session_factory(ROOT, destination, item["variant"], item["group_id"], config,
            readiness, tokenizer, client, core.ContinuationPlanner, base.frozen.format_base)
        if progress is not None:
            observe_session(session, progress, client)
        current, row = base.install_historical_seed_order(session, parent, client).run()
        row["usage"].extend(inherited_rows(parent, item["variant"], item["group_id"]))
        base.frozen.durable(destination / "session_completions.jsonl", {"execution_index": index,
            "group_id": item["group_id"], "variant": item["variant"], "predictions": current, "row": row}, append=True)
        all_rows[item["variant"]].append(row)
        predictions[item["variant"]].extend(current)
        if progress is not None:
            progress.session_done(item["variant"], item["group_id"])
    if not client.live:
        raise base.ReplayMismatch("Offline continuation crossed its failed-request boundary")
    base.frozen.durable(destination / "planning_complete.json", {"sessions": 14, "parent_run_id": parent["run_id"]})
    return finish_scores(destination, parent, config, readiness, predictions, all_rows, scorer=scorer, progress=progress)


def finish_scores(destination, parent, config, readiness, predictions, all_rows, *, scorer=None, progress=None):
    if not (destination / "planning_complete.json").exists():
        raise base.ReplayMismatch("Scoring requires the complete fixed planning grid")
    scorer = scorer or base.score_official
    results = []
    active_scope = parent["failed_intent"]["group_id"]
    for variant in base.frozen.VARIANTS:
        if progress is not None:
            progress.phase(f"SCORING {variant}: all fourteen planning sessions are complete")
        scores = scorer(ROOT / config["experiment"]["upstream_root"], ROOT / "data/hidden/memoryarena_travel.jsonl",
            predictions[variant], parent["protocol"]["group_ids"],
            expected_dataset_sha=readiness["provenance"]["dataset_sha256"])
        base.frozen.durable(destination / f"official_scores_{variant}.json", scores)
        for original in all_rows[variant]:
            row = copy.deepcopy(original)
            row["recovery_disclosure"] = recovery_disclosure(parent)
            scope = f"{variant}:{row['group_id']}"
            timing = parent.get("timing_sources", {}).get(scope)
            mixed = scope == active_scope or (timing is not None and timing["mixed"])
            if timing is not None:
                row["wall_clock_seconds"] = timing["row"]["wall_clock_seconds"]
                row["timing_origin_run_id"] = timing["run_id"]
                if not mixed:
                    row["usage"] = copy.deepcopy(timing["row"]["usage"])
                row["usage"] = [{**usage, "reused": True, "new_billable": False,
                    "origin_run_id": usage.get("origin_run_id", timing["run_id"])} for usage in row["usage"]]
            if mixed:
                row["reconstruction_plus_new_wall_clock_seconds"] = row["wall_clock_seconds"]
                row["wall_clock_seconds"] = None
                for usage in row["usage"]:
                    if not usage.get("request_status"):
                        usage["reconstruction_latency_seconds"] = usage.get("latency_seconds")
                        usage["latency_seconds"] = None
                row["timing_scope"] = "Mixed reconstruction/new session; continuous wall and local/tool timings unavailable"
            else:
                row["timing_scope"] = "Retained original closed-session timing" if timing else "New continuation session; request latency includes declared pacing"
            score = next(item for item in scores["per_group"] if str(item["group_id"]) == row["group_id"])
            if score["base_template_id"] != row["base_template_id"]:
                raise ValueError("Official scorer family differs")
            row["official_metrics"] = score["official_metrics"]
            row["official_metric_metadata"] = scores.get("official_metric_metadata", scores.get("metric_metadata", {}))
            people = {int(item["traveler_index"]) - 1: item for item in score.get("traveler_metrics", [])}
            row["travelers"] = [{**person, **people.get(person["traveler_index"], {}), "traveler_index": person["traveler_index"]}
                                for person in row["travelers"]]
            row["official_metric_name"] = scores.get("metric_definition", {}).get("SPS", "upstream SPS")
            row["official_metric_version"] = config["experiment"]["upstream_repo_sha"]
            base.frozen.durable(destination / "results.jsonl", row, append=True)
            results.append(row)
    return results


def archive_continuation(destination, pin, pin_path):
    members = {**pin["source_sha256"], pin_path.relative_to(ROOT).as_posix(): base.sha256(pin_path)}
    with zipfile.ZipFile(destination / "continuation_source.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative in members:
            archive.write(ROOT / relative, relative)
    base.frozen.durable(destination / "continuation_source.json", {
        "members_sha256": members, "archive_sha256": base.sha256(destination / "continuation_source.zip")})


def execute(parent_id, live):
    with base.SuiteLock(ROOT), ProgressReporter() as progress:
        progress.phase("Validating ancestor runs, archived source, receipts and lineage claims")
        parent = core.load_parent(ROOT, parent_id)
        progress.configure(parent)
        starting_source = source()
        pin = verify_pin(parent) if live else None
        if live:
            core.verify_unused(ROOT, parent_id)
        readiness = base.verify_readiness(ROOT)
        progress.phase("Loading the pinned tokenizer and verifying local benchmark data")
        tokenizer_path, _ = base.verify_tokenizer(ROOT)
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        run_id = "cross-family-continuation-" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        destination = ROOT / "runs" / run_id if live else ROOT / "tmp/cross-family-continuation" / uuid.uuid4().hex
        destination.mkdir(parents=True, exist_ok=False)
        progress.phase(f"RUN {run_id}; {'LIVE' if live else 'OFFLINE RECONSTRUCTION; network denied'}")
        config = copy.deepcopy(parent["manifest"]["config"])
        config["agent"].update(stream=True, request_timeout_seconds=1800, retries=0)
        disclosure = recovery_disclosure(parent)
        manifest = {**copy.deepcopy(parent["manifest"]), "run_id": run_id, "status": "retaining_closed_and_replaying_active",
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "config": config,
            "parent_run_id": parent_id, "original_run_id": base.PARENT_ID,
            "continuation_schema_version": 1, "parent_files_sha256": parent["file_sha256"],
            "lineage_files_sha256": parent["lineage_files_sha256"],
            "inherited_accepted_responses": len(parent["accepted"]), "inherited_failures": parent["all_failures"],
            "inherited_failure_debits": parent["failure_debits"],
            "inherited_writer_failure_debits": parent.get("writer_failure_debits", {}),
            "recovery_disclosure": disclosure, "effective_bounds": disclosure["effective_physical_dispatch_bounds"],
            "original_protocol_label": parent["manifest"].get("original_protocol_label", parent["manifest"].get("protocol_label")),
            "protocol_label": (disclosure["classification"] if not disclosure["original_dispatch_bound_satisfied"]
                               else parent["manifest"].get("protocol_label")),
            "predecessor_directories_required": list(parent["lineage_files_sha256"]),
            "source_freeze": str((preparation(parent_id) / "source_freeze.json").relative_to(ROOT)),
            "transport_changes": {"wire_stream": True, "native_serializer_stream": False,
                "recorded_raw_stream": True, "minimum_request_start_interval_seconds": 10,
                "initial_cooldown_seconds": 10, "automatic_paid_retries": 0,
                "fresh_connection_before_every_post": True, "max_safe_preflight_attempts": 3},
            "prefix_policy": "Retain closed isolated sessions and validate their receipts; natively reconstruct only the unfinished session; compare complete combined causal logs and snapshots before dispatch",
            "live_comparison_validated": False}
        manifest.pop("error", None)
        manifest.pop("usage", None)
        manifest.pop("failed_dispatch_budget", None)
        base.frozen.durable(destination / "manifest.json", manifest)
        if live:
            base.archive_run_source(ROOT, destination)
            archive_continuation(destination, pin, preparation(parent_id) / "source_freeze.json")
            base.frozen.durable(destination / "inherited_failed_usage.json", parent["all_failures"])
            base.load_env(ROOT / ".env")
        client = None
        try:
            agent = config["agent"]
            with ObservedClient(parent, destination, live=live, progress=progress,
                    api_key=None if live else "offline-continuation-network-denied", model=agent["model"], base_url=agent["base_url"],
                    temperature=agent["temperature"], max_tokens=agent["max_completion_tokens"],
                    extraction_max_tokens=agent["max_extraction_tokens"], seed=config["experiment"]["seed"],
                    provider=agent["routing"], reasoning=agent["reasoning"], extraction_reasoning=agent["extraction_reasoning"],
                    stream=False, timeout=1800, retries=0, max_total_tokens=math.inf, max_cost_rub=None) as client:
                def gate():
                    progress.phase("Verifying the complete causal boundary: all semantic logs, snapshots and exact next request")
                    proof = core.compare_prefix(parent["directory"], destination)
                    if client.cursor != len(parent["accepted"]) or source() != starting_source or not core.verify_all_predecessors(parent):
                        raise base.ReplayMismatch("Cached coverage, source or predecessor evidence changed")
                    proof.update(parent_run_id=parent_id, accepted_responses_not_repurchased=len(parent["accepted"]),
                        retained_closed_session_responses=parent["imported_model_responses"],
                        native_active_session_responses_replayed=parent["active_model_responses"],
                        inherited_sent_failures=sum(row.get("request_sent") is True for row in parent["all_failures"]),
                        failure_debits=parent["failure_debits"], writer_failure_debits=parent.get("writer_failure_debits", {}),
                        recovery_disclosure=recovery_disclosure(parent), model_calls=0, source_sha256=starting_source,
                        lineage_files_sha256=parent["lineage_files_sha256"],
                        failed_intent_sha256=hashlib.sha256(base.canonical(parent["failed_intent"]).encode()).hexdigest())
                    base.frozen.durable(destination / "boundary_verified.json", proof)
                    if live:
                        verify_pin(parent)
                        core.verify_unused(ROOT, parent_id, exclude_child=destination)
                        core.claim(ROOT, parent_id, destination)
                        manifest["status"] = "continuing_after_verified_boundary"
                        base.frozen.durable(destination / "manifest.json", manifest)
                        progress.phase("BOUNDARY VERIFIED: saved responses retained; new requests are now enabled")
                    else:
                        base.frozen.durable(preparation(parent_id) / "replay_verified.json", proof)
                client.gate_callback = gate
                if live:
                    recorded = RecordingChatTransport(client, destination)
                    observed = ObservedTransport(recorded, progress)
                    fresh = FreshConnectionTransport(observed, client, destination, progress=progress)
                    client.network_transport = PacedTransport(fresh, destination, progress=progress)
                    client.freeze_check = lambda: verify_pin(parent)
                execute_grid(destination, parent, tokenizer, readiness, client, progress=progress)
                if not core.verify_all_predecessors(parent):
                    raise base.ReplayMismatch("A predecessor changed during continuation")
                from constraint_memory.orchestration.replay import replay_run
                progress.phase("FINAL VERIFICATION: replaying all fourteen memory stores")
                if len(replay_run(destination)) != 14:
                    raise base.ReplayMismatch("Completed memory replay does not cover fourteen sessions")
                from constraint_memory.evaluation import report_run
                progress.phase("REPORTING: computing paired metrics and bootstrap intervals")
                report_run(destination, requested_metrics=("SR", "PS", "SPS"), primary_contrast=base.frozen.VARIANTS,
                    primary_endpoint="SR", bootstrap_samples=10000, seed=config["experiment"]["seed"])
                manifest.update(status="complete", completed_conditions=14, replay_all_match=True,
                    live_comparison_validated=True, usage=client.get_usage_stats())
        except base.ReplayBoundaryReached:
            if live:
                raise
            manifest.update(status="offline_prefix_verified", usage=client.get_usage_stats())
            print("OFFLINE CONTINUATION VERIFIED: closed sessions retained, active state reconstructed, no model calls", flush=True)
            return 0
        except BaseException as error:
            manifest.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                            error={"type": type(error).__name__, "message": base.redact(str(error))})
            if client is not None:
                manifest["usage"] = client.get_usage_stats()
            try:
                progress.heartbeat()
            except Exception:
                pass
            print("STOPPED " + run_id + ": " + base.redact(str(error)), flush=True)
            return 1
        finally:
            base.frozen.durable(destination / "manifest.json", manifest)
        print("COMPLETED " + run_id, flush=True)
        progress.phase("COMPLETE: planning, scoring, memory verification and reports finished")
        return 0


def prepare(parent_id):
    directory = preparation(parent_id)
    directory.mkdir(parents=True, exist_ok=True)
    starting_source = source()
    started = time.perf_counter()
    command = [sys.executable, "-m", "pytest", "experiments/cross_family_continuation", "-q", "-p", "no:cacheprovider",
               "--basetemp=" + str(ROOT / "tmp" / ("continuation-tests-" + uuid.uuid4().hex))]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    match = re.search(r"(\d+) passed", result.stdout)
    tests = {"command": command, "exit_code": result.returncode, "passed": int(match[1]) if match else 0,
             "stdout": result.stdout, "stderr": result.stderr, "elapsed_seconds": time.perf_counter() - started, "model_calls": 0}
    base.frozen.durable(directory / "verification.json", tests)
    print(result.stdout, end="", flush=True)
    if result.returncode or not match:
        raise RuntimeError("Offline continuation tests failed: " + result.stderr)
    if source() != starting_source:
        raise ValueError("Source changed during verification")
    started = time.perf_counter()
    code = execute(parent_id, False)
    base.frozen.durable(directory / "replay_execution.json", {"exit_code": code, "model_calls": 0,
        "elapsed_seconds": time.perf_counter() - started})
    if code:
        return code
    parent = core.load_parent(ROOT, parent_id)
    if source() != starting_source or not core.verify_all_predecessors(parent):
        raise ValueError("Source or predecessor evidence changed during reconstruction")
    pin = {"parent_run_id": parent_id, "source_sha256": starting_source,
        "lineage_files_sha256": parent["lineage_files_sha256"], "unit_tests": tests,
        "claim_files_sha256": parent.get("claim_files_sha256", {}),
        "verification_sha256": base.sha256(directory / "verification.json"),
        "replay_verified_sha256": base.sha256(directory / "replay_verified.json"),
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "model_calls": 0}
    base.frozen.durable(directory / "source_freeze.json", pin)
    verify_pin(parent)
    core.verify_unused(ROOT, parent_id)
    print("PREPARED " + parent_id + ": tested source pinned; no paid calls", flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--live", action="store_true")
    mode.add_argument("--prepare", action="store_true")
    parser.add_argument("--parent", default=DEFAULT_PARENT)
    args = parser.parse_args(argv)
    preparation(args.parent)
    if not args.live:
        def deny_network(event, _args):
            if event.startswith("socket.") or event == "http.client.connect":
                raise PermissionError("Offline continuation cannot access the network")
        sys.addaudithook(deny_network)
    return prepare(args.parent) if args.prepare else execute(args.parent, args.live)


if __name__ == "__main__":
    raise SystemExit(main())
