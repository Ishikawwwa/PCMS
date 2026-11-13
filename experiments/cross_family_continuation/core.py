"""Lineage validation, retained closed sessions, and native active-session replay.

This controller never retries an ambiguous completion automatically. Closed,
isolated GroupSessions are imported as historical observations; their native
model receipts are validated again, without claiming to rerun their planners.
"""
from __future__ import annotations

import copy
import datetime as dt
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from experiments.cross_family_recovery import run as base
from experiments.cross_family_recovery_second import run as second
from constraint_memory.benchmark_adapter import PlannerWorker
from constraint_memory.polza import PolzaClient

ReplayMismatch = base.ReplayMismatch
ReplayBoundaryReached = base.ReplayBoundaryReached
CLAIMS_RELATIVE = "reports/cross_family_continuation/claims"
SEMANTIC_LOGS = ("events.jsonl", "memory_ops.jsonl", "extractions.jsonl", "retrievals.jsonl",
    "recoveries.jsonl", "submission_events.jsonl", "predictions.jsonl", "traces.jsonl",
    "session_completions.jsonl")
TRANSPORT_FIELDS = {"stream", "request_timeout_seconds", "retries"}


def _run_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", value) or value in {".", ".."}:
        raise ReplayMismatch("Invalid lineage run ID")
    return value


def _scope(row):
    if row.get("variant") is not None:
        return f"{row['variant']}:{row['group_id']}"
    return str(row.get("group_id", ""))


def _semantic_intent(intent):
    return {key: intent[key] for key in base.PAYLOAD_FIELDS}


def _verify_zip(path, members, archive_digest=None):
    if archive_digest is not None and base.sha256(path) != archive_digest:
        raise ReplayMismatch("Continuation archive bytes changed")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != set(members):
            raise ReplayMismatch("Continuation archive members differ")
        for name, expected in members.items():
            member = Path(name)
            if member.is_absolute() or ".." in member.parts or "\\" in name:
                raise ReplayMismatch("Unsafe continuation archive member")
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise ReplayMismatch("Continuation archive member hash differs: " + name)


def _runtime_archive(root, directory, manifest, legacy_parent):
    """Validate past runtime archives, without changing or executing them."""
    run_id = manifest["run_id"]
    if run_id == base.PARENT_ID:
        return
    if run_id == second.CHILD_ID:
        pin = base.read(base.FREEZE)
        members = {**pin["recovery_source_sha256"], base.FREEZE.relative_to(root).as_posix(): base.sha256(base.FREEZE)}
        _verify_zip(directory / "recovery_source.zip", members)
    elif (directory / "continuation_source.json").exists():
        sidecar = base.read(directory / "continuation_source.json")
        _verify_zip(directory / "continuation_source.zip", sidecar["members_sha256"], sidecar["archive_sha256"])
        if manifest.get("continuation_schema_version") != 1:
            raise ReplayMismatch("Unsupported reusable continuation schema")
    elif (directory / "continuation_source.zip").exists():
        # The frozen second-stage runtime predates reusable sidecars.
        pin = second.verify_freeze(legacy_parent)
        members = {**pin["source_sha256"], second.FREEZE.relative_to(root).as_posix(): base.sha256(second.FREEZE)}
        _verify_zip(directory / "continuation_source.zip", members)
    else:
        raise ReplayMismatch("A predecessor has no verifiable controller archive")


def _ancestor_claim(root, directory, manifest):
    """Validate the one-way claim that permitted this predecessor to send."""
    if manifest["run_id"] == base.PARENT_ID:
        return None
    if manifest["run_id"] == second.CHILD_ID:
        path = root / "reports/cross_family_recovery/live_lineage_claim.json"
    elif (directory / "continuation_source.json").exists():
        path = _claim_path(root, manifest["parent_run_id"])
    else:
        path = root / "reports/cross_family_recovery_second/live_lineage_claim.json"
    try:
        receipt = base.read(path)
        if (not isinstance(receipt, dict) or receipt.get("parent_run_id") != manifest["parent_run_id"]
                or receipt.get("child_run_id") != manifest["run_id"]
                or Path(receipt["child_directory"]).resolve() != directory.resolve()):
            raise ValueError("Ancestor claim links another parent or child")
    except (ValueError, OSError, KeyError, TypeError):
        raise ReplayMismatch("A predecessor's exclusive dispatch claim is missing or conflicting") from None
    return path.relative_to(root).as_posix(), base.sha256(path)


def _read_receipts(directory, manifest):
    journal = base.rows(directory / "request_journal.jsonl")
    intents = [row for row in journal if row.get("event") == "INTENT_ONLY_NOT_DISPATCH_PROOF"]
    accepted_events = [row for row in journal if row.get("event") == "RESPONSE_ACCEPTED"]
    failed_events = [row for row in journal if row.get("event") == "REQUEST_FAILED"]
    committed = base.rows(directory / "committed_responses.jsonl")
    model_usage = [row for row in base.rows(directory / "usage.jsonl") if row.get("request_status")]
    if (len(failed_events) != 1 or len(intents) != len(committed) + 1
            or len(accepted_events) != len(committed) or len(model_usage) != len(intents)
            or not journal or journal[-1].get("event") != "REQUEST_FAILED"):
        raise ReplayMismatch("Only one terminal unaccepted model-request boundary is supported")
    if len(journal) != 2 * len(intents):
        raise ReplayMismatch("Unexpected or incomplete model journal events")
    accepted, request_ids = [], set()
    for index, (intent, saved, accepted_event, usage) in enumerate(zip(intents, committed, accepted_events, model_usage), 1):
        attempt = f"request-{index:05d}"
        raw = saved.get("raw_response")
        if (intent.get("attempt_id") != attempt or saved.get("attempt_id") != attempt
                or accepted_event.get("attempt_id") != attempt or journal[2 * index - 2] != intent
                or journal[2 * index - 1] != accepted_event or not isinstance(raw, dict)
                or not isinstance(raw.get("id"), str) or not raw["id"] or raw["id"] in request_ids
                or usage.get("request_id") != raw["id"] or accepted_event.get("request_id") != raw["id"]
                or raw.get("model") != "qwen/qwen3.6-27b" or usage.get("request_status") != "completed"
                or usage.get("response_accepted") is not True or usage.get("model_identity_validation") != "EXACT_MATCH"):
            raise ReplayMismatch("Accepted response identity/order differs from its original receipt")
        for key in ("group_id", "traveler", "operation"):
            if saved.get(key) != intent.get(key) or usage.get(key) != intent.get(key):
                raise ReplayMismatch("Accepted model receipt scope differs")
        receipt = raw.get("usage", {})
        if tuple(usage.get(key) for key in ("input_tokens", "output_tokens", "total_tokens", "cost_rub")) != tuple(
                receipt.get(key) for key in ("prompt_tokens", "completion_tokens", "total_tokens", "cost_rub")):
            raise ReplayMismatch("Accepted usage differs from the saved provider response")
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            if type(receipt.get(key)) is not int or receipt[key] < 0:
                raise ReplayMismatch("Invalid accepted provider token usage")
        if receipt["total_tokens"] < receipt["prompt_tokens"] + receipt["completion_tokens"]:
            raise ReplayMismatch("Accepted token total understates usage")
        cost = receipt.get("cost_rub")
        if cost is not None and (type(cost) not in (int, float) or not math.isfinite(cost) or cost < 0):
            raise ReplayMismatch("Invalid accepted provider billing")
        request_ids.add(raw["id"])
        accepted.append({"intent": intent, "raw_response": raw, "usage": usage,
            "origin_run_id": usage.get("origin_run_id", manifest["run_id"]),
            "origin_attempt_id": usage.get("parent_attempt_id", attempt)})
    failed_intent, failed_usage = intents[-1], model_usage[-1]
    sent = failed_usage.get("request_sent")
    if (failed_intent.get("attempt_id") != f"request-{len(intents):05d}"
            or failed_events[0].get("attempt_id") != failed_intent["attempt_id"]
            or failed_usage.get("request_status") != "failed" or type(sent) is not bool
            or failed_events[0].get("request_sent") is not sent
            or any(failed_usage.get(key) != failed_intent.get(key) for key in ("group_id", "traveler", "operation"))):
        raise ReplayMismatch("The terminal failure is not an audited sent request without accepted output")
    if sent is False:
        if (failed_usage.get("billing_status") != "not_sent" or failed_usage.get("request_id") is not None
                or any(type(failed_usage.get(key)) not in (int, float) or failed_usage[key] != 0
                       for key in ("input_tokens", "output_tokens", "total_tokens", "cost_rub"))):
            raise ReplayMismatch("A pre-dispatch failure lacks strict native zero-usage proof")
        for path in (directory / "new_requests").glob("*/dispatch.json"):
            dispatch = base.read(path)
            if dispatch.get("logical_attempt_id") == failed_intent["attempt_id"] and dispatch.get("request_sent") is not False:
                raise ReplayMismatch("A supposedly unsent failure has conflicting POST dispatch evidence")
    elif (failed_usage.get("billing_status") == "not_sent"
          or (failed_usage.get("billing_status") == "unknown_after_failed_request"
              and any(failed_usage.get(key) is not None for key in ("input_tokens", "output_tokens", "total_tokens", "cost_rub")))):
        raise ReplayMismatch("Sent-request billing status contradicts its recorded usage")
    if len(journal[-1].get("usage", [])) != 1 or base.canonical(journal[-1]["usage"][0]) != base.canonical(failed_usage):
        raise ReplayMismatch("Terminal failure accounting differs between journal and usage ledger")
    return accepted, failed_intent, failed_usage


def _validate_config(config, original):
    current = copy.deepcopy(config)
    expected = copy.deepcopy(original)
    for key in TRANSPORT_FIELDS:
        current["agent"].pop(key, None)
        expected["agent"].pop(key, None)
    if base.canonical(current) != base.canonical(expected):
        raise ReplayMismatch("A predecessor changed an inference, memory, tool, or causal protocol setting")


def _failure_arrival(accepted, failed_intent):
    # The frozen E1/E2 protocol extracts once per arrival and has no repairs.
    # An ordinal, rather than a traveler name, also handles repeated names.
    arrival = sum(entry["intent"]["group_id"] == failed_intent["group_id"]
                  and entry["intent"]["operation"] == "extraction" for entry in accepted)
    if failed_intent["operation"] == "extraction":
        arrival += 1
        try:
            payload = json.loads(failed_intent["messages"][-1]["content"])
            current = payload["current_request"]
        except (ValueError, KeyError, TypeError, IndexError):
            raise ReplayMismatch("Failed writer boundary lacks its original causal request") from None
        if (type(current.get("round_idx")) is not int or current["round_idx"] != arrival
                or current.get("traveler") != failed_intent["traveler"]
                or str(current.get("group_id")) != failed_intent["group_id"].split(":", 1)[-1]):
            raise ReplayMismatch("Failed writer request does not match its released arrival")
        if not 1 <= arrival <= 7:
            raise ReplayMismatch("Failed writer arrival exceeds the frozen session")
        return arrival
    if failed_intent["operation"] != "agent":
        raise ReplayMismatch("Unsupported failed operation in the fixed travel protocol")
    if not 1 <= arrival <= 7:
        raise ReplayMismatch("Cannot bind the failed actor request to a released arrival")
    latest = next(entry["intent"] for entry in reversed(accepted)
                  if entry["intent"]["group_id"] == failed_intent["group_id"] and entry["intent"]["operation"] == "extraction")
    if latest["traveler"] != failed_intent["traveler"]:
        raise ReplayMismatch("Failed actor request differs from the current released traveler")
    return arrival


def _completed(directory, protocol):
    completed = base.rows(directory / "session_completions.jsonl")
    if len(completed) >= len(protocol["schedule"]):
        raise ReplayMismatch("Planning already completed; run offline finalization rather than new inference")
    for index, row in enumerate(completed, 1):
        expected = protocol["schedule"][index - 1]
        if (row.get("execution_index") != index or row.get("group_id") != expected["group_id"]
                or row.get("variant") != expected["variant"] or row.get("row", {}).get("group_id") != expected["group_id"]
                or row["row"].get("variant") != expected["variant"]):
            raise ReplayMismatch("Completed sessions are not a contiguous original schedule prefix")
    return completed


def load_parent(root, parent_id):
    """Load a terminal leaf and validate every predecessor without inference."""
    root = Path(root).resolve()
    _run_id(parent_id)
    chain = []
    seen = set()
    cursor = parent_id
    while True:
        _run_id(cursor)
        if cursor in seen or len(seen) >= 128:
            raise ReplayMismatch("Continuation ancestry is cyclic or exceeds the supported bound")
        seen.add(cursor)
        directory = root / "runs" / cursor
        manifest = base.read(directory / "manifest.json")
        if (not isinstance(manifest, dict) or manifest.get("run_id") != cursor
                or manifest.get("status") not in ("failed", "interrupted")):
            raise ReplayMismatch("Select a terminal stopped run, not a complete or active run")
        chain.append({"run_id": cursor, "directory": directory, "manifest": manifest})
        if cursor == base.PARENT_ID:
            break
        cursor = manifest.get("parent_run_id")
    chain.reverse()
    original = base.read_parent(root)
    legacy_parent = None
    if any(node["run_id"] == second.CHILD_ID for node in chain):
        legacy_parent = second.read_lineage(root)
    protocol = original["protocol"]
    original_config = original["manifest"]["config"]
    all_failures, prior_entries, timing_sources, claim_files = [], None, {}, {}
    for node in chain:
        directory, manifest = node["directory"], node["manifest"]
        _validate_config(manifest["config"], original_config)
        if manifest.get("source_sha256") != original["manifest"]["source_sha256"]:
            raise ReplayMismatch("Native memory/planner source differs within the lineage")
        base.verify_archive(root, directory)
        _runtime_archive(root, directory, manifest, legacy_parent)
        claimed = _ancestor_claim(root, directory, manifest)
        if claimed is not None:
            claim_files[claimed[0]] = claimed[1]
        node["file_sha256"] = base.tree_hashes(directory)
        entries, failed_intent, failed_usage = _read_receipts(directory, manifest)
        node.update(accepted=entries, failed_intent=failed_intent, failed_usage=failed_usage,
                    completed=_completed(directory, protocol))
        if prior_entries is not None:
            predecessor = chain[chain.index(node) - 1]
            if manifest.get("parent_files_sha256") != predecessor["file_sha256"]:
                raise ReplayMismatch("A predecessor changed after this child was created")
            if len(entries) < len(prior_entries):
                raise ReplayMismatch("A child dropped inherited accepted responses")
            for previous, current in zip(prior_entries, entries):
                if (base.canonical(previous["raw_response"]) != base.canonical(current["raw_response"])
                        or base.canonical(_semantic_intent(previous["intent"])) != base.canonical(_semantic_intent(current["intent"]))):
                    raise ReplayMismatch("A child changed or resampled a saved accepted response")
                current["origin_run_id"] = previous["origin_run_id"]
                current["origin_attempt_id"] = previous["origin_attempt_id"]
            next_intent = entries[len(prior_entries)]["intent"] if len(entries) > len(prior_entries) else failed_intent
            if base.canonical(_semantic_intent(next_intent)) != base.canonical(_semantic_intent(predecessor["failed_intent"])):
                raise ReplayMismatch("A child did not continue its predecessor's exact unaccepted request")
        failure = {**copy.deepcopy(failed_usage), "origin_run_id": node["run_id"],
                   "origin_attempt_id": failed_intent["attempt_id"], "arrival_index": _failure_arrival(entries, failed_intent)}
        all_failures.append(failure)
        for completed in node["completed"]:
            scope = _scope(completed)
            if scope not in timing_sources:
                mixed = (any(usage.get("reused") is True or usage.get("inherited_failed_dispatch") is True
                             for usage in completed["row"].get("usage", []))
                         or any(f["group_id"] == scope and f["origin_run_id"] != node["run_id"] for f in all_failures))
                timing_sources[scope] = {"row": copy.deepcopy(completed["row"]), "run_id": node["run_id"], "mixed": mixed}
        prior_entries = entries
    leaf = chain[-1]
    # The transition from the frozen second-stage controller is independently
    # audited once. Future reusable children carry archive sidecars and parent
    # byte pins, so no new stage-specific source edit is needed after each stop.
    if (leaf["directory"] / "continuation_source.json").exists() is False and leaf["run_id"] not in (base.PARENT_ID, second.CHILD_ID):
        audit_path = root / "reports/cross_family_continuation/accounting.json"
        audit = base.read(audit_path)
        if (audit.get("status") != "verified_third_stop_with_saved_provider_rate_limit"
                or audit.get("parent_run_id") != leaf["run_id"]
                or audit.get("file_sha256", {}).get("parent") != leaf["file_sha256"]):
            raise ReplayMismatch("The legacy-to-reusable migration lacks its independent stop audit")
    completed = leaf["completed"]
    closed = {_scope(row) for row in completed}
    imported = 0
    for entry_index, entry in enumerate(leaf["accepted"]):
        if entry["intent"]["group_id"] in closed:
            if imported != entry_index:
                raise ReplayMismatch("Closed-session model receipts are not a contiguous prefix")
            imported += 1
    if any(entry["intent"]["group_id"] in closed for entry in leaf["accepted"][imported:]):
        raise ReplayMismatch("Closed-session receipts reappear inside the active session")
    failure_debits, writer_failure_debits = {}, {}
    for failure in all_failures:
        if failure["request_sent"] is not True:
            continue
        destination = writer_failure_debits if failure["operation"] == "extraction" else failure_debits
        by_arrival = destination.setdefault(failure["group_id"], {})
        arrival = failure["arrival_index"]
        by_arrival[arrival] = by_arrival.get(arrival, 0) + 1
    writer_failures = sum(sum(by_arrival.values()) for by_arrival in writer_failure_debits.values())
    original_writer_bound = protocol["bounds"]["max_writer_requests"]
    return {"root": root, "run_id": leaf["run_id"], "directory": leaf["directory"], "manifest": leaf["manifest"],
            "protocol": protocol, "accepted": leaf["accepted"], "failed_intent": leaf["failed_intent"],
            "failed_usage": leaf["failed_usage"], "retrievals": base.rows(leaf["directory"] / "retrievals.jsonl"),
            "all_failures": all_failures, "failure_debits": failure_debits,
            "writer_failure_debits": writer_failure_debits, "lineage": chain,
            "lineage_files_sha256": {node["run_id"]: node["file_sha256"] for node in chain},
            "claim_files_sha256": claim_files,
            "file_sha256": leaf["file_sha256"], "completed": completed,
            "imported_model_responses": imported, "active_model_responses": len(leaf["accepted"]) - imported,
            "timing_sources": timing_sources,
            "writer_attempt_policy": {"accepted_writers_per_arrival": 1,
                "original_max_writer_requests": original_writer_bound,
                "inherited_sent_writer_failures": writer_failures,
                "effective_max_writer_requests": original_writer_bound + writer_failures,
                "classification": "infra_recovered_diagnostic" if writer_failures else "original_writer_attempt_bound",
                "note": "A failed sent writer has no accepted semantic output. Recovered writer attempts are declared extra physical requests, not silently granted original-budget calls."}}


def verify_all_predecessors(parent):
    try:
        return (all(base.tree_hashes(node["directory"]) == node["file_sha256"] for node in parent["lineage"])
                and all(base.sha256(Path(parent.get("root", ROOT)) / relative) == expected
                        for relative, expected in parent.get("claim_files_sha256", {}).items()))
    except OSError:
        return False


class ContinuationClient(base.RecoveryClient):
    def _annotate_replacement_marker(self):
        path = self.destination / "replacement_attempt.json"
        if not self.boundary_verified or not path.exists():
            return
        scope = self.parent["failed_intent"]["group_id"]
        debits = self.parent.get("failure_debits", {}).get(scope, {})
        arrival = next((failure.get("arrival_index") for failure in reversed(self.parent.get("all_failures", []))
                        if failure.get("origin_run_id") == self.parent["run_id"]), None)
        if arrival is None:
            arrival = max((int(index) for index in debits), default=0)
        failed = debits.get(arrival, debits.get(str(arrival), 0))
        current_arrival, accepted_actor = 0, 0
        for entry in self.parent["accepted"]:
            intent = entry["intent"]
            if intent["group_id"] != scope:
                continue
            if intent["operation"] == "extraction":
                current_arrival += 1
            elif intent["operation"] == "agent" and current_arrival == arrival:
                accepted_actor += 1
        marker = base.read(path)
        marker.update(parent_run_id=self.parent["run_id"], original_run_id=base.PARENT_ID,
                      stream=True, wire_stream=True, native_serializer_stream=False,
                      inherited_sent_failures=sum(failure.get("request_sent") is True for failure in self.parent.get("all_failures", [])),
                      current_arrival_index=arrival, current_arrival_failed_dispatches=failed,
                      failed_operation=self.parent["failed_intent"]["operation"],
                      writer_attempt_policy=self.parent.get("writer_attempt_policy"),
                      remaining_new_actor_calls=30 - failed - accepted_actor,
                      policy="One deliberate continuation of the exact failed request; every inherited sent failure remains inside its own original arrival allowance")
        base.frozen.durable(path, marker)

    def complete(self, *args, **kwargs):
        try:
            return super().complete(*args, **kwargs)
        finally:
            self._annotate_replacement_marker()

    def _dispatch(self, method, path, payload):
        if self.replay_entry is None and self.live and self.boundary_verified:
            self._annotate_replacement_marker()
        return super()._dispatch(method, path, payload)

    def _record_usage(self, record):
        row = copy.deepcopy(record)
        entry = self.replay_entry
        row.update(reused=entry is not None, new_billable=entry is None,
                   origin_run_id=(entry.get("origin_run_id", self.parent["run_id"]) if entry else self.destination.name))
        if entry is not None:
            row.update(parent_attempt_id=entry.get("origin_attempt_id", entry["intent"]["attempt_id"]),
                       replay_latency_seconds=row["latency_seconds"], latency_seconds=entry["usage"]["latency_seconds"])
        PolzaClient._record_usage(self, row)

    def get_usage_stats(self):
        result = base.frozen.JournalClient.get_usage_stats(self)
        failures = copy.deepcopy(self.parent.get("all_failures", []))
        failed_known = sum((Decimal(str(row["cost_rub"])) for row in failures if row.get("cost_rub") is not None), Decimal(0))
        accepted_known = sum((Decimal(str(entry["usage"]["cost_rub"])) for entry in self.parent["accepted"]
                              if entry["usage"].get("cost_rub") is not None), Decimal(0))
        new_known = sum((Decimal(str(row["cost_rub"])) for row in self.records
                         if not row.get("reused") and row.get("cost_rub") is not None), Decimal(0))
        inherited = accepted_known + failed_known
        unknown = (any(row.get("cost_rub") is None for row in failures)
                   or any(entry["usage"].get("cost_rub") is None for entry in self.parent["accepted"])
                   or any(row.get("cost_rub") is None for row in self.records if not row.get("reused")))
        result.update(inherited_accepted_responses=self.cursor, new_request_attempts=self.new_attempts,
                      inherited_failed_attempts=len(failures), inherited_failures=failures,
                      inherited_known_accepted_cost_rub_exact=str(accepted_known),
                      inherited_known_failed_cost_rub_exact=str(failed_known), inherited_known_cost_rub_exact=str(inherited),
                      new_known_cost_rub_exact=str(new_known), combined_known_cost_rub_exact=str(inherited + new_known),
                      total_cost_rub=None if unknown else float(inherited + new_known),
                      accounting_note="Each accepted receipt is retained once. Distinct sent predecessor failures are separate from active execution records and never inferred free.")
        if any(row.get("total_tokens") is None for row in failures):
            result.update(total_input_tokens=None, total_output_tokens=None)
        return result


class ContinuationPlanner(PlannerWorker):
    def __init__(self, upstream_root, db_path, client, **kwargs):
        by_arrival = client.parent.get("failure_debits", {}).get(client.group_id, {})
        arrival = kwargs.get("causal_round")
        self.inherited_failed_dispatches = by_arrival.get(arrival, by_arrival.get(str(arrival), 0))
        if type(self.inherited_failed_dispatches) is not int or not 0 <= self.inherited_failed_dispatches < kwargs["max_steps"]:
            raise ReplayMismatch("No actor dispatch allowance remains at this inherited failure boundary")
        kwargs["max_steps"] -= self.inherited_failed_dispatches
        super().__init__(upstream_root, db_path, client, **kwargs)

    def run_turn(self, request, memory_context, base_person):
        result = super().run_turn(request, memory_context, base_person)
        if self.inherited_failed_dispatches:
            result["total_steps"] += self.inherited_failed_dispatches
            result["parent_model_requests"] += self.inherited_failed_dispatches
            result["inherited_failed_model_dispatches"] = self.inherited_failed_dispatches
        return result


def append_failures(row, parent):
    scope = _scope(row)
    row["usage"].extend({**copy.deepcopy(failure), "inherited_failed_dispatch": True, "new_billable": False}
                        for failure in parent.get("all_failures", []) if failure["group_id"] == scope)


def import_completed(parent, destination, client):
    """Retain isolated completed sessions and validate their native receipts."""
    destination = Path(destination)
    completed = copy.deepcopy(parent["completed"])
    closed = {_scope(row) for row in completed}
    count = parent["imported_model_responses"]
    if client.cursor or client.records or client.attempt_number:
        raise ReplayMismatch("Closed-session import requires a fresh replay client")
    if any(entry["intent"]["group_id"] not in closed for entry in parent["accepted"][:count]):
        raise ReplayMismatch("Imported receipt scope is not a completed isolated session")
    if any(entry["intent"]["group_id"] in closed for entry in parent["accepted"][count:]):
        raise ReplayMismatch("Completed receipt prefix is not contiguous")
    # Native validation advances the same request counters and scope guards.
    # Its temporary usage sink avoids duplicating model receipts when the
    # historical usage stream is retained in its original causal order below.
    usage_path = client.usage_path
    client.usage_path = destination / "import_validation/usage.jsonl"
    try:
        for entry in parent["accepted"][:count]:
            intent = entry["intent"]
            client.set_scope(intent["group_id"], intent["traveler"])
            raw = client.complete(copy.deepcopy(intent["messages"]), copy.deepcopy(intent["tools"]), intent["operation"],
                                  intent["max_tokens_override"], intent["json_output"], copy.deepcopy(intent["response_schema"]))
            if base.canonical(raw) != base.canonical(entry["raw_response"]):
                raise ReplayMismatch("Imported native validation returned another response")
    finally:
        client.usage_path = usage_path
    if client.cursor != count or len(client.records) != count or client.attempt_number != count:
        raise ReplayMismatch("Closed-session native receipt validation has incomplete coverage")
    imported_logs = {}
    for name in SEMANTIC_LOGS:
        selected = [row for row in base.rows(parent["directory"] / name) if _scope(row) in closed]
        for row in selected:
            base.frozen.durable(destination / name, row, append=True)
        imported_logs[name] = len(selected)
    for row in completed:
        relative = Path("snapshots") / row["variant"] / f"{row['group_id']}.json"
        source = parent["directory"] / relative
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as output:
            output.write(source.read_bytes())
            output.flush()
            os.fsync(output.fileno())
    validated = {row["request_id"]: row for row in client.records}
    retained_model_count = 0
    for row in base.rows(parent["directory"] / "usage.jsonl"):
        if _scope(row) not in closed:
            continue
        retained = copy.deepcopy(row)
        if row.get("request_status"):
            receipt = validated.get(row.get("request_id"))
            if receipt is None:
                raise ReplayMismatch("Historical usage has no validated accepted receipt")
            for key in ("reused", "new_billable", "origin_run_id", "parent_attempt_id", "replay_latency_seconds"):
                retained[key] = receipt[key]
            retained_model_count += 1
        base.frozen.durable(destination / "usage.jsonl", retained, append=True)
    if retained_model_count != count:
        raise ReplayMismatch("Retained usage count differs from imported accepted receipts")
    for row in base.rows(parent["directory"] / "writer_responses.jsonl"):
        if _scope(row) in closed:
            base.frozen.durable(destination / "writer_responses.jsonl", row, append=True)
    closed_retrievals = [row for row in parent["retrievals"] if _scope(row) in closed]
    if parent["retrievals"][:len(closed_retrievals)] != closed_retrievals:
        raise ReplayMismatch("Closed retrievals do not form a chronological prefix")
    client.retrieval_cursor = len(closed_retrievals)
    proof = {"retained_completed_sessions": len(completed), "retained_model_responses": count,
             "native_active_replayed": 0, "native_receipt_validation": True, "full_native_execution": False,
             "imported_semantic_log_rows": imported_logs, "retained_snapshots": len(completed),
             "historical_retrieval_cursor": client.retrieval_cursor,
             "note": "Closed isolated sessions are retained, not rerun. Only active-session prompts and tools are reconstructed natively."}
    base.frozen.durable(destination / "closed_session_import.json", proof)
    return completed, proof


def compare_prefix(parent, destination):
    directory = parent["directory"] if isinstance(parent, dict) else parent
    return base.compare_prefix(directory, destination)


def _claim_path(root, parent_id):
    return Path(root) / CLAIMS_RELATIVE / (_run_id(parent_id) + ".json")


def verify_unused(root, parent_id, exclude_child=None):
    root = Path(root).resolve()
    if _claim_path(root, parent_id).exists():
        raise ReplayMismatch("This stopped parent already has a claimed continuation; inspect that child")
    excluded = Path(exclude_child).resolve() if exclude_child is not None else None
    for directory in sorted((root / "runs").glob("*")):
        if not directory.is_dir():
            continue
        markers = [directory / name for name in ("continuation_attempt.json", "replacement_attempt.json", "second_replacement_attempt.json")]
        try:
            manifest = base.read(directory / "manifest.json") if (directory / "manifest.json").exists() else None
            if manifest is not None and not isinstance(manifest, dict):
                raise ValueError("Invalid child manifest")
        except (ValueError, OSError):
            if any(path.exists() for path in markers) or directory.name.startswith("cross-family-continuation-"):
                raise ReplayMismatch("An unreadable continuation requires an explicit audit")
            continue
        for marker in markers:
            if marker.exists():
                try:
                    marked_parent = base.read(marker)["parent_run_id"]
                except (ValueError, OSError, KeyError, TypeError):
                    raise ReplayMismatch("An unreadable dispatch marker requires an explicit audit")
                if marked_parent == parent_id or (manifest or {}).get("parent_run_id") == parent_id:
                    raise ReplayMismatch("A child already reached this parent's new-dispatch boundary")
        if manifest is None:
            if directory.name.startswith("cross-family-continuation-"):
                raise ReplayMismatch("A previous continuation has no readable manifest")
            continue
        if manifest.get("parent_run_id") != parent_id:
            continue
        if directory.resolve() == excluded:
            continue
        stats = manifest.get("usage")
        if (manifest.get("status") not in ("failed", "interrupted") or not isinstance(stats, dict)
                or type(stats.get("new_request_attempts")) is not int or stats["new_request_attempts"] != 0):
            raise ReplayMismatch("A previous child may contain new paid work")
        try:
            journal = base.rows(directory / "request_journal.jsonl")
            intents = [row for row in journal if row.get("event") == "INTENT_ONLY_NOT_DISPATCH_PROOF"]
            limit = manifest.get("inherited_accepted_responses")
            if type(limit) is not int or not 0 <= len(intents) <= limit:
                raise ValueError("Unknown imported cache allowance")
            ids = {row["attempt_id"] for row in intents}
            if (any(row.get("attempt_id") != f"request-{i:05d}" for i, row in enumerate(intents, 1))
                    or any(row.get("event") not in ("INTENT_ONLY_NOT_DISPATCH_PROOF", "RESPONSE_ACCEPTED", "REQUEST_FAILED")
                           or row.get("attempt_id") not in ids for row in journal)
                    or any(row.get("event") == "REQUEST_FAILED" and row.get("request_sent") is not False for row in journal)):
                raise ValueError("Ambiguous child model journal")
            for row in base.rows(directory / "usage.jsonl"):
                evidence = any(key in row for key in ("model", "request_id", "request_status", "reused", "new_billable"))
                local = row.get("model_call") is False and not evidence and all(row.get(key) == 0 for key in
                                                                                 ("input_tokens", "output_tokens", "total_tokens", "cost_rub"))
                if not local and (row.get("request_status") not in ("completed", "failed")
                                  or row.get("reused") is not True or row.get("new_billable") is not False):
                    raise ValueError("Unknown or new model receipt")
        except (ValueError, OSError, AttributeError, TypeError, KeyError):
            raise ReplayMismatch("A previous child's dispatch evidence requires an audit")


def claim(root, parent_id, destination):
    path = _claim_path(root, parent_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    value = {"parent_run_id": parent_id, "child_run_id": Path(destination).name,
             "child_directory": str(Path(destination).resolve()), "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
             "policy": "One deliberate continuation of this exact stopped leaf; no automatic paid retry"}
    try:
        with path.open("x", encoding="utf-8", newline="\n") as output:
            output.write(base.canonical(value) + "\n")
            output.flush()
            os.fsync(output.fileno())
    except FileExistsError:
        raise ReplayMismatch("This parent was already claimed; no new request sent")
