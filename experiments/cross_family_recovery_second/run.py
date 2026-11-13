"""Continue the audited two-failure lineage with recorded Chat SSE transport.

Default execution is network-denied prefix reconstruction, never inference.
Both sent failures remain charged to the arrival allowance and have unknown bills.
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
import re
import sys
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from experiments.cross_family_recovery import run as base

PARENT_ID = base.PARENT_ID
CHILD_ID = "cross-family-recovery-20261016T144804072163Z"
REPORTS = ROOT / "reports/cross_family_recovery_second"
FREEZE = REPORTS / "source_freeze.json"
ReplayMismatch = base.ReplayMismatch
ReplayBoundaryReached = base.ReplayBoundaryReached
CLAIM_RELATIVE = "reports/cross_family_recovery_second/live_lineage_claim.json"


def read_lineage(root):
    root = Path(root).resolve()
    parent = base.read_parent(root)
    base.verify_recovery_freeze(parent)
    child_dir = root / "runs" / CHILD_ID
    child = base.read(child_dir / "manifest.json")
    audit = base.read(root / "reports/cross_family_recovery_second/accounting.json")
    claim = base.read(root / "reports/cross_family_recovery/live_lineage_claim.json")
    if (child.get("status") != "failed" or child.get("parent_run_id") != PARENT_ID
            or claim.get("parent_run_id") != PARENT_ID or claim.get("child_run_id") != CHILD_ID
            or Path(claim["child_directory"]).resolve() != child_dir.resolve()
            or audit.get("status") != "verified_second_stop_no_new_accepted_completion"
            or audit.get("parent_run_id") != PARENT_ID or audit.get("child_run_id") != CHILD_ID):
        raise ReplayMismatch("The audited two-failure lineage or original claim differs")
    child_hashes = base.tree_hashes(child_dir)
    if (parent["file_sha256"] != audit["file_sha256"]["parent"]
            or child_hashes != audit["file_sha256"]["child"]):
        raise ReplayMismatch("A predecessor artifact changed after its independent audit")
    base.verify_archive(root, child_dir)
    with zipfile.ZipFile(child_dir / "recovery_source.zip") as archive:
        expected = {**base.frozen.source_map(root, "experiments/cross_family_recovery"),
                    base.FREEZE.relative_to(root).as_posix(): base.sha256(base.FREEZE)}
        if len(archive.namelist()) != len(set(archive.namelist())) or set(archive.namelist()) != set(expected):
            raise ReplayMismatch("Prior recovery archive members differ")
        if any(hashlib.sha256(archive.read(name)).hexdigest() != digest for name, digest in expected.items()):
            raise ReplayMismatch("Prior recovery archive source differs")
    journal = base.rows(child_dir / "request_journal.jsonl")
    intents = [row for row in journal if row["event"] == "INTENT_ONLY_NOT_DISPATCH_PROOF"]
    accepted = [row for row in journal if row["event"] == "RESPONSE_ACCEPTED"]
    failures = [row for row in journal if row["event"] == "REQUEST_FAILED"]
    usage = [row for row in base.rows(child_dir / "usage.jsonl") if row.get("request_status")]
    if (len(intents), len(accepted), len(failures), len(usage)) != (156, 155, 1, 156):
        raise ReplayMismatch("The prior child has work beyond the audited first replacement failure")
    expected_intents = [entry["intent"] for entry in parent["accepted"]] + [parent["failed_intent"]]
    if base.canonical(intents) != base.canonical(expected_intents):
        raise ReplayMismatch("Prior recovery request intents differ from the original")
    if base.sha256(child_dir / "committed_responses.jsonl") != base.sha256(parent["directory"] / "committed_responses.jsonl"):
        raise ReplayMismatch("Prior recovery has different or additional accepted responses")
    failed = usage[-1]
    if (failed.get("request_sent") is not True or failed.get("request_status") != "failed"
            or failed.get("reused") is not False or failed.get("new_billable") is not True
            or failed.get("cost_rub") is not None or failed.get("total_tokens") is not None
            or any(row.get("reused") is not True or row.get("new_billable") is not False for row in usage[:-1])):
        raise ReplayMismatch("Prior replacement dispatch or cache accounting differs")
    proof = base.compare_prefix(parent["directory"], child_dir)
    if proof["status"] != "exact_semantic_prefix" or proof["snapshots"] != 7:
        raise ReplayMismatch("The paid child's actual causal prefix differs")
    parent.update(child_run_id=CHILD_ID, child={"run_id": CHILD_ID, "directory": child_dir,
                  "manifest": child, "file_sha256": child_hashes}, extra_failed_usage=copy.deepcopy(failed))
    return parent


def verify_second_lineage(root, child_id, exclude_child=None):
    root = Path(root).resolve()
    if (root / CLAIM_RELATIVE).exists():
        raise ReplayMismatch("Second continuation already claimed; inspect its saved run before another paid attempt")
    excluded = Path(exclude_child).resolve() if exclude_child is not None else None
    for directory in sorted((root / "runs").glob("*")):
        if not directory.is_dir():
            continue
        markers = [directory / name for name in ("second_replacement_attempt.json", "replacement_attempt.json")]
        try:
            manifest = base.read(directory / "manifest.json") if (directory / "manifest.json").exists() else None
            if manifest is not None and not isinstance(manifest, dict):
                raise ValueError("Invalid manifest")
        except (ValueError, OSError):
            if any(marker.exists() for marker in markers) or directory.name.startswith("cross-family-recovery-second-"):
                raise ReplayMismatch("Ambiguous previous second continuation requires review")
            continue
        for marker in markers:
            if not marker.exists():
                continue
            try:
                previous_id = base.read(marker)["parent_run_id"]
            except (ValueError, OSError, KeyError, TypeError):
                raise ReplayMismatch("Ambiguous prior replacement marker requires review")
            if previous_id == child_id or (manifest or {}).get("parent_run_id") == child_id:
                raise ReplayMismatch("A second continuation already reached the new dispatch boundary")
        if manifest is None:
            if directory.name.startswith("cross-family-recovery-second-"):
                raise ReplayMismatch("A previous second continuation has no manifest")
            continue
        if manifest.get("parent_run_id") != child_id:
            continue
        if directory.resolve() == excluded:
            continue
        stats = manifest.get("usage")
        if (manifest.get("status") not in ("failed", "interrupted") or not isinstance(stats, dict)
                or type(stats.get("new_request_attempts")) is not int or stats["new_request_attempts"] != 0):
            raise ReplayMismatch("Previous second continuation is not a proven terminal cache-only failure")
        try:
            journal = base.rows(directory / "request_journal.jsonl")
            intents = [row for row in journal if row.get("event") == "INTENT_ONLY_NOT_DISPATCH_PROOF"]
            ids = {row["attempt_id"] for row in intents}
            if (len(intents) > 155
                    or any(row.get("attempt_id") != f"request-{i:05d}" for i, row in enumerate(intents, 1))
                    or any(row.get("attempt_id") not in ids or row.get("event") not in
                           ("INTENT_ONLY_NOT_DISPATCH_PROOF", "RESPONSE_ACCEPTED", "REQUEST_FAILED") for row in journal)
                    or any(row.get("event") == "REQUEST_FAILED" and row.get("request_sent") is not False for row in journal)):
                raise ValueError("Ambiguous dispatch journal")
            for row in base.rows(directory / "usage.jsonl"):
                evidence = any(key in row for key in ("model", "request_id", "request_status", "reused", "new_billable"))
                local = (row.get("model_call") is False and not evidence and all(row.get(key) == 0
                         for key in ("input_tokens", "output_tokens", "total_tokens", "cost_rub")))
                if not local and (row.get("request_status") not in ("completed", "failed")
                                  or row.get("reused") is not True or row.get("new_billable") is not False):
                    raise ValueError("Ambiguous model usage")
        except (ValueError, OSError, TypeError, KeyError, AttributeError):
            raise ReplayMismatch("Prior second-continuation receipts require a new audit")


def claim_second_lineage(root, child_id, destination):
    path = Path(root) / CLAIM_RELATIVE
    path.parent.mkdir(parents=True, exist_ok=True)
    value = {"parent_run_id": child_id, "child_run_id": Path(destination).name,
             "child_directory": str(Path(destination).resolve()),
             "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
             "policy": "One explicit new continuation; no automatic paid retry"}
    try:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(base.canonical(value) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        raise ReplayMismatch("Second continuation already claimed; no new request sent")


class SecondRecoveryClient(base.RecoveryClient):
    def _annotate_replacement_marker(self):
        path = self.destination / "replacement_attempt.json"
        if not self.boundary_verified or not path.exists():
            return
        marker = base.read(path)
        marker.update(parent_run_id=self.parent["child_run_id"], original_run_id=self.parent["run_id"],
                      parent_request_id=self.parent["extra_failed_usage"].get("request_id"),
                      original_request_id=self.parent["failed_usage"].get("request_id"),
                      stream=True, wire_stream=True, native_serializer_stream=False,
                      inherited_sent_failures=2, remaining_new_actor_calls=27,
                      policy="One deliberate continuation after exact replay; both sent failures count inside the original 30-step arrival allowance")
        base.frozen.durable(path, marker)

    def complete(self, *args, **kwargs):
        try:
            return super().complete(*args, **kwargs)
        finally:
            # Also label a pre-dispatch failure correctly if the frozen base
            # wrote its compatibility marker but never reached transport.
            self._annotate_replacement_marker()

    def _dispatch(self, method, path, payload):
        if self.replay_entry is None and self.live and self.boundary_verified:
            # Correct the inherited compatibility marker before any network
            # transport may run, including a crash during a paid request.
            self._annotate_replacement_marker()
        return super()._dispatch(method, path, payload)

    def get_usage_stats(self):
        result = super().get_usage_stats()
        failures = [{**copy.deepcopy(self.parent["failed_usage"]), "origin_run_id": self.parent["run_id"]},
                    {**copy.deepcopy(self.parent["extra_failed_usage"]), "origin_run_id": self.parent["child_run_id"]}]
        result.update(inherited_failed_attempts=2, inherited_failures=failures,
                      total_cost_rub=None, total_input_tokens=None, total_output_tokens=None,
                      accounting_note="155 reused receipts are not new charges. Both sent predecessor failures remain unknown and count inside the original action allowance.")
        return result


class SecondRecoveryPlanner(base.RecoveryPlanner):
    def __init__(self, upstream_root, db_path, client, **kwargs):
        self.second_failed_dispatch = int(client.group_id == "E2:69" and kwargs.get("causal_round") == 7)
        if self.second_failed_dispatch:
            kwargs["max_steps"] -= 1
        super().__init__(upstream_root, db_path, client, **kwargs)

    def run_turn(self, request, memory_context, base_person):
        result = super().run_turn(request, memory_context, base_person)
        if self.second_failed_dispatch:
            result["parent_model_requests"] += 1
            result["total_steps"] += 1
            result["inherited_failed_model_dispatches"] = 2
        return result


def append_predecessor_failures(row, parent):
    row["usage"].extend([
        {**copy.deepcopy(parent["failed_usage"]), "origin_run_id": parent["run_id"],
         "inherited_failed_dispatch": True, "new_billable": False},
        {**copy.deepcopy(parent["extra_failed_usage"]), "origin_run_id": parent["child_run_id"],
         "inherited_failed_dispatch": True, "new_billable": False},
    ])


def execute_grid(root, destination, parent, tokenizer, readiness, client):
    protocol = copy.deepcopy(parent["protocol"])
    config = protocol["config"]
    config["agent"].update(stream=True, request_timeout_seconds=1800)
    predictions = {variant: [] for variant in base.frozen.VARIANTS}
    all_rows = {variant: [] for variant in base.frozen.VARIANTS}
    for index, item in enumerate(protocol["schedule"], 1):
        print(f"SESSION {index}/14 group={item['group_id']} {item['variant']} (saved replies are not repurchased)", flush=True)
        session = base.frozen.GroupSession(root, destination, item["variant"], item["group_id"], config,
            readiness, tokenizer, client, SecondRecoveryPlanner, base.frozen.format_base)
        current, row = base.install_historical_seed_order(session, parent, client).run()
        if (item["group_id"], item["variant"]) == ("69", "E2"):
            append_predecessor_failures(row, parent)
        base.frozen.durable(destination / "session_completions.jsonl", {"execution_index": index,
            "group_id": item["group_id"], "variant": item["variant"], "predictions": current, "row": row}, append=True)
        predictions[item["variant"]].extend(current)
        all_rows[item["variant"]].append(row)
    if not client.live:
        raise ReplayMismatch("Offline execution crossed the required failed-request boundary")
    base.frozen.durable(destination / "planning_complete.json", {"sessions": 14, "parent_run_id": CHILD_ID})
    result_rows = []
    parent_rows = {(item["group_id"], item["variant"]): item["row"] for item in base.rows(parent["directory"] / "session_completions.jsonl")}
    for variant in base.frozen.VARIANTS:
        scores = base.score_official(root / config["experiment"]["upstream_root"], root / "data/hidden/memoryarena_travel.jsonl",
            predictions[variant], protocol["group_ids"], expected_dataset_sha=readiness["provenance"]["dataset_sha256"])
        base.frozen.durable(destination / f"official_scores_{variant}.json", scores)
        for row in all_rows[variant]:
            prior = parent_rows.get((row["group_id"], row["variant"]))
            if prior is not None:
                row["reconstruction_wall_clock_seconds"] = row["wall_clock_seconds"]
                row["wall_clock_seconds"] = prior["wall_clock_seconds"]
                row["usage"] = [{**copy.deepcopy(item), "reused": True, "new_billable": False,
                    "origin_run_id": PARENT_ID} for item in prior["usage"]]
                row["timing_scope"] = "Original completed session; reconstruction time is separate"
            elif (row["group_id"], row["variant"]) == ("69", "E2"):
                row["reconstruction_plus_new_wall_clock_seconds"] = row["wall_clock_seconds"]
                row["wall_clock_seconds"] = None
                for item in row["usage"]:
                    if not item.get("request_status"):
                        item["reconstruction_latency_seconds"] = item["latency_seconds"]
                        item["latency_seconds"] = None
                row["timing_scope"] = "Mixed reconstruction/new work; continuous wall and local/tool timings unavailable"
            else:
                row["timing_scope"] = "New live continuation session"
            score = next(item for item in scores["per_group"] if str(item["group_id"]) == row["group_id"])
            if score["base_template_id"] != row["base_template_id"]:
                raise ValueError("Official scorer family differs")
            row["official_metrics"] = score["official_metrics"]
            row["official_metric_metadata"] = scores.get("official_metric_metadata", scores.get("metric_metadata", {}))
            people = {int(person["traveler_index"]) - 1: person for person in score.get("traveler_metrics", [])}
            row["travelers"] = [{**person, **people.get(person["traveler_index"], {}), "traveler_index": person["traveler_index"]} for person in row["travelers"]]
            row["official_metric_name"] = scores.get("metric_definition", {}).get("SPS", "upstream SPS")
            row["official_metric_version"] = config["experiment"]["upstream_repo_sha"]
            base.frozen.durable(destination / "results.jsonl", row, append=True)
            result_rows.append(row)
    return result_rows


def verify_freeze(parent):
    pin = base.read(FREEZE)
    if (base.frozen.source_map(ROOT, "experiments/cross_family_recovery_second") != pin["source_sha256"]
            or parent["file_sha256"] != pin["parent_files_sha256"]
            or parent["child"]["file_sha256"] != pin["child_files_sha256"]
            or base.sha256(REPORTS / "replay_verified.json") != pin["replay_verified_sha256"]
            or base.sha256(base.FREEZE) != pin["previous_recovery_freeze_sha256"]
            or base.sha256(ROOT / "reports/cross_family_recovery/live_lineage_claim.json") != pin["previous_claim_sha256"]
            or base.sha256(REPORTS / "accounting.json") != pin["accounting_sha256"]
            or not pin["unit_tests"].get("passed")):
        raise ReplayMismatch("Second-continuation source, evidence, or actual offline replay pin changed")
    base.verify_recovery_freeze(parent)
    return pin


def predecessor_hashes_match(parent):
    return (base.tree_hashes(parent["directory"]) == parent["file_sha256"]
            and base.tree_hashes(parent["child"]["directory"]) == parent["child"]["file_sha256"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--run-id")
    args = parser.parse_args(argv)
    if not args.live:
        def deny_network(event, _args):
            if event.startswith("socket.") or event == "http.client.connect":
                raise PermissionError("Offline second recovery cannot access the network")
        sys.addaudithook(deny_network)
    with base.SuiteLock(ROOT):
        parent = read_lineage(ROOT)
        starting_source = base.frozen.source_map(ROOT, "experiments/cross_family_recovery_second")
        pin = verify_freeze(parent) if args.live else None
        if args.live:
            verify_second_lineage(ROOT, CHILD_ID)
        readiness = base.verify_readiness(ROOT)
        tokenizer_path, _ = base.verify_tokenizer(ROOT)
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        run_id = args.run_id or "cross-family-recovery-second-" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id) or run_id in (".", ".."):
            parser.error("Invalid run ID")
        destination = ROOT / "runs" / run_id if args.live else ROOT / "tmp/cross-family-second-prefix" / uuid.uuid4().hex
        if destination.exists():
            parser.error("Use a new run directory; previous runs are never overwritten")
        destination.mkdir(parents=True)
        config = copy.deepcopy(parent["manifest"]["config"])
        config["agent"].update(stream=True, request_timeout_seconds=1800)
        failures = [{**copy.deepcopy(parent["failed_usage"]), "origin_run_id": PARENT_ID},
                    {**copy.deepcopy(parent["extra_failed_usage"]), "origin_run_id": CHILD_ID}]
        manifest = {**copy.deepcopy(parent["manifest"]), "run_id": run_id, "status": "replaying_prefix",
                    "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "config": config,
                    "parent_run_id": CHILD_ID, "original_run_id": PARENT_ID,
                    "parent_files_sha256": parent["child"]["file_sha256"],
                    "original_files_sha256": parent["file_sha256"],
                    "inherited_failures": failures, "inherited_accepted_responses": 155,
                    "failed_dispatch_budget": "Olivia E2:69: 1 accepted +2 sent failures +at most27new actor calls =30",
                    "transport_changes": {"wire_stream": True, "native_serializer_stream": False,
                        "serializer_note": "The injected recording transport adds only stream=true; all inference fields are checked against the original contract",
                        "raw_sse_capture": True, "non_error_identity_checks": "unchanged native strict parser",
                        "terminal_provider_error": "record and reject before reporting metadata inconsistencies",
                        "automatic_paid_retries": False, "read_timeout_seconds": 1800},
                    "timing_policy": "Original complete sessions retain original timings; mixed cell continuous timings unavailable",
                    "predecessor_directories_required": [PARENT_ID, CHILD_ID],
                    "live_comparison_validated": False}
        manifest.pop("error", None)
        manifest.pop("usage", None)
        base.frozen.durable(destination / "manifest.json", manifest)
        if args.live:
            base.archive_run_source(ROOT, destination)
            with zipfile.ZipFile(destination / "continuation_source.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
                for relative in pin["source_sha256"]:
                    archive.write(ROOT / relative, relative)
                archive.write(FREEZE, FREEZE.relative_to(ROOT).as_posix())
            base.frozen.durable(destination / "inherited_failed_usage.json", failures)
            base.load_env(ROOT / ".env")
        client = None
        try:
            agent = config["agent"]
            with SecondRecoveryClient(parent, destination, live=args.live,
                api_key=None if args.live else "offline-second-recovery-no-network", model=agent["model"], base_url=agent["base_url"],
                temperature=agent["temperature"], max_tokens=agent["max_completion_tokens"], extraction_max_tokens=agent["max_extraction_tokens"],
                seed=config["experiment"]["seed"], provider=agent["routing"], reasoning=agent["reasoning"],
                extraction_reasoning=agent["extraction_reasoning"], stream=False, timeout=1800, retries=0,
                max_total_tokens=math.inf, max_cost_rub=None) as client:
                def gate():
                    proof = base.compare_prefix(parent["directory"], destination)
                    if (client.cursor != 155 or not predecessor_hashes_match(parent)
                            or base.frozen.source_map(ROOT, "experiments/cross_family_recovery_second") != starting_source):
                        raise ReplayMismatch("Cached coverage, predecessor artifacts, or source changed")
                    proof.update(original_run_id=PARENT_ID, parent_run_id=CHILD_ID, accepted_responses_reused=155,
                        inherited_sent_failures=2, remaining_new_olivia_actor_calls=27, model_calls=0,
                        source_sha256=starting_source, parent_files_sha256=parent["file_sha256"],
                        child_files_sha256=parent["child"]["file_sha256"],
                        failed_intent_sha256=hashlib.sha256(base.canonical(parent["failed_intent"]).encode()).hexdigest())
                    base.frozen.durable(destination / "boundary_verified.json", proof)
                    if args.live:
                        verify_freeze(parent)
                        verify_second_lineage(ROOT, CHILD_ID, exclude_child=destination)
                        claim_second_lineage(ROOT, CHILD_ID, destination)
                        base.frozen.durable(destination / "second_replacement_attempt.json", {
                            "parent_run_id": CHILD_ID, "original_run_id": PARENT_ID,
                            "inherited_failures": failures, "remaining_new_actor_calls": 27,
                            "wire_stream": True, "automatic_paid_retry": False})
                        manifest["status"] = "continuing_after_verified_boundary"
                        base.frozen.durable(destination / "manifest.json", manifest)
                    else:
                        base.frozen.durable(REPORTS / "replay_verified.json", proof)
                client.gate_callback = gate
                if args.live:
                    from experiments.cross_family_recovery_second.transport import RecordingChatTransport
                    client.network_transport = RecordingChatTransport(client, destination)
                    client.freeze_check = lambda: verify_freeze(parent)
                result_rows = execute_grid(ROOT, destination, parent, tokenizer, readiness, client)
                if not predecessor_hashes_match(parent):
                    raise ReplayMismatch("A predecessor changed during continuation")
                from constraint_memory.orchestration.replay import replay_run
                replay = replay_run(destination)
                if len(replay) != 14:
                    raise ReplayMismatch("Completed memory replay does not cover all 14 sessions")
                from constraint_memory.evaluation import report_run
                report_run(destination, requested_metrics=("SR", "PS", "SPS"), primary_contrast=base.frozen.VARIANTS,
                    primary_endpoint="SR", bootstrap_samples=10000, seed=config["experiment"]["seed"])
                manifest.update(status="complete", completed_conditions=len(result_rows), replay_all_match=True,
                    live_comparison_validated=True, usage=client.get_usage_stats())
        except ReplayBoundaryReached:
            if args.live:
                raise
            manifest.update(status="offline_prefix_verified", usage=client.get_usage_stats())
            print("OFFLINE SECOND PREFIX VERIFIED: 155 saved responses; two failed dispatches retained; no model calls", flush=True)
            return 0
        except BaseException as error:
            manifest.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                            error={"type": type(error).__name__, "message": base.redact(str(error))})
            if client is not None:
                manifest["usage"] = client.get_usage_stats()
            print("STOPPED " + run_id + ": " + base.redact(str(error)), flush=True)
            return 1
        finally:
            base.frozen.durable(destination / "manifest.json", manifest)
        print("COMPLETED " + run_id, flush=True)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
