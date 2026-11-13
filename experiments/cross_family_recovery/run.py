"""Replay a paid prefix offline, or continue it with non-streaming transport.

The parent is immutable. Cache hits traverse native identity/accounting/tool
parsers, and every regenerated request must match its original intent exactly.
No original failed response is fabricated or accepted as a task action.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
from decimal import Decimal
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import re
import sys
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from experiments.cross_family import run as frozen
from constraint_memory.benchmark_adapter import PlannerWorker, verify_readiness, score_official
from constraint_memory.config import sha256, verify_tokenizer
from constraint_memory.polza import PolzaClient
from constraint_memory.secrets import load_env, redact
from archive_run_source import archive_run_source, verify_archive
from run_benchmark_suite import SuiteLock

PARENT_ID = "cross-family-primary-20261015T192052114101Z"
REPORTS = ROOT / "reports/cross_family_recovery"
FREEZE = REPORTS / "source_freeze.json"
PAYLOAD_FIELDS = ("group_id", "traveler", "operation", "messages", "tools", "max_tokens_override", "json_output", "response_schema")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    path = Path(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []


def tree_hashes(directory):
    return {path.relative_to(directory).as_posix(): sha256(path) for path in sorted(Path(directory).rglob("*")) if path.is_file()}


def verify_unused_lineage(root, parent_id, exclude_child=None):
    """Called under SuiteLock; an earlier possible paid continuation is final.

    Only a demonstrably cache-only terminal failure may be reconstructed again.
    The exclusive claim also survives a missing or renamed child directory.
    """
    root = Path(root).resolve()
    claim = root / "reports/cross_family_recovery/live_lineage_claim.json"
    if claim.exists():
        raise ReplayMismatch("Recovery lineage has already been claimed; inspect the existing child, do not repurchase its continuation")
    excluded = Path(exclude_child).resolve() if exclude_child is not None else None
    for directory in sorted((root / "runs").glob("*")):
        if not directory.is_dir():
            continue
        marker = directory / "replacement_attempt.json"
        manifest_path = directory / "manifest.json"
        try:
            manifest = read(manifest_path) if manifest_path.exists() else None
            if manifest is not None and not isinstance(manifest, dict):
                raise ValueError("Manifest is not an object")
        except (ValueError, OSError):
            if marker.exists() or directory.name.startswith("cross-family-recovery-"):
                raise ReplayMismatch("Ambiguous previous recovery manifest requires an audit: " + directory.name)
            continue
        if marker.exists():
            try:
                marked_parent = read(marker)["parent_run_id"]
            except (ValueError, OSError, KeyError, TypeError):
                raise ReplayMismatch("Ambiguous previous replacement marker requires an audit: " + directory.name)
            if marked_parent == parent_id or (manifest or {}).get("parent_run_id") == parent_id:
                raise ReplayMismatch("A previous child already reached the replacement boundary: " + directory.name)
        if manifest is None:
            if directory.name.startswith("cross-family-recovery-"):
                raise ReplayMismatch("Previous recovery has no manifest: " + directory.name)
            continue
        if manifest.get("parent_run_id") != parent_id:
            continue
        if excluded is not None and directory.resolve() == excluded:
            continue
        stats = manifest.get("usage")
        if (manifest.get("status") not in ("failed", "interrupted") or not isinstance(stats, dict)
                or type(stats.get("new_request_attempts")) is not int or stats["new_request_attempts"] != 0):
            raise ReplayMismatch("Previous recovery is not a proven terminal cache-only failure: " + directory.name)
        try:
            journal = rows(directory / "request_journal.jsonl")
            usage = rows(directory / "usage.jsonl")
            intents = [row for row in journal if row.get("event") == "INTENT_ONLY_NOT_DISPATCH_PROOF"]
            for row in usage:
                model_evidence = any(key in row for key in ("request_status", "request_id", "model", "reused", "new_billable"))
                local = (row.get("model_call") is False and not model_evidence
                         and all(row.get(key) == 0 for key in ("input_tokens", "output_tokens", "total_tokens", "cost_rub")))
                if not local and (row.get("request_status") not in ("completed", "failed")
                                  or row.get("reused") is not True or row.get("new_billable") is not False):
                    raise ValueError("Previous child has a new or ambiguous model usage row")
            if (len(intents) > 155
                    or any(row.get("attempt_id") != f"request-{index:05d}" for index, row in enumerate(intents, 1))
                    or any(row.get("event") not in ("INTENT_ONLY_NOT_DISPATCH_PROOF", "RESPONSE_ACCEPTED", "REQUEST_FAILED")
                           or row.get("attempt_id") not in {item["attempt_id"] for item in intents} for row in journal)
                    or any(row.get("event") == "REQUEST_FAILED" and row.get("request_sent") is not False for row in journal)):
                raise ValueError("Previous child has new or ambiguous dispatch evidence")
        except (ValueError, OSError, AttributeError, TypeError):
            raise ReplayMismatch("Previous recovery dispatch evidence requires an audit: " + directory.name)


def claim_lineage(root, parent_id, destination):
    """Durable one-way boundary before the first possible new model dispatch."""
    claim = Path(root) / "reports/cross_family_recovery/live_lineage_claim.json"
    claim.parent.mkdir(parents=True, exist_ok=True)
    value = {"parent_run_id": parent_id, "child_run_id": Path(destination).name,
             "child_directory": str(Path(destination).resolve()),
             "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
             "policy": "No second paid continuation from this original parent without a new audit"}
    try:
        with claim.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(canonical(value) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        raise ReplayMismatch("Recovery lineage was already claimed; no new request sent")


class ReplayMismatch(ValueError):
    request_sent = False


class ReplayBoundaryReached(RuntimeError):
    request_sent = False


def read_parent(root, parent_id=PARENT_ID):
    root = Path(root).resolve()
    if parent_id != PARENT_ID:
        raise ValueError("This recovery supports only its audited original parent; later failures require a new audit")
    directory = root / "runs" / parent_id
    manifest, protocol = read(directory / "manifest.json"), read(directory / "protocol.json")
    if manifest["status"] != "failed" or manifest["config"] != protocol["config"]:
        raise ValueError("Parent must be the terminal original protocol")
    frozen.verify_freeze(root)
    if frozen.source_map(root, "src") != manifest["source_sha256"]:
        raise ValueError("Actual native source differs from parent")
    verify_archive(root, directory)
    sidecar = read(directory / "experiment_inputs.json")
    if sha256(directory / "experiment_inputs.zip") != sidecar["archive_sha256"]:
        raise ValueError("Parent experiment archive changed")
    with zipfile.ZipFile(directory / "experiment_inputs.zip") as archive:
        members = sidecar["members_sha256"]
        if len(archive.namelist()) != len(set(archive.namelist())) or set(archive.namelist()) != set(members):
            raise ValueError("Parent experiment archive members differ")
        for name, expected in members.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected or sha256(root / name) != expected:
                raise ValueError("Parent frozen input changed: " + name)
    journal = rows(directory / "request_journal.jsonl")
    intents = [r for r in journal if r["event"] == "INTENT_ONLY_NOT_DISPATCH_PROOF"]
    committed = rows(directory / "committed_responses.jsonl")
    usage = [r for r in rows(directory / "usage.jsonl") if r.get("request_status")]
    accepted_events = [r for r in journal if r["event"] == "RESPONSE_ACCEPTED"]
    failed_events = [r for r in journal if r["event"] == "REQUEST_FAILED"]
    if (len(intents), len(committed), len(usage), len(accepted_events), len(failed_events)) != (156, 155, 156, 155, 1):
        raise ValueError("Parent no longer has its audited single-failure boundary")
    accepted = []
    ids = set()
    for index, (intent, response, bill, accepted_event) in enumerate(zip(intents, committed, usage, accepted_events), 1):
        raw = response["raw_response"]
        if (intent["attempt_id"] != f"request-{index:05d}" or response["attempt_id"] != intent["attempt_id"]
                or accepted_event["attempt_id"] != intent["attempt_id"] or bill["request_id"] != raw["id"]
                or raw["id"] in ids or raw["model"] != "qwen/qwen3.6-27b"
                or not bill.get("response_accepted") or bill["model_identity_validation"] != "EXACT_MATCH"):
            raise ValueError("Parent accepted receipt sequence or identity changed")
        for field in ("group_id", "traveler", "operation"):
            if response[field] != intent[field] or bill[field] != intent[field]:
                raise ValueError("Parent receipt scope differs")
        if (bill["input_tokens"], bill["output_tokens"], bill["total_tokens"], bill["cost_rub"]) != (
                raw["usage"]["prompt_tokens"], raw["usage"]["completion_tokens"], raw["usage"]["total_tokens"], raw["usage"]["cost_rub"]):
            raise ValueError("Parent receipt usage differs")
        ids.add(raw["id"])
        accepted.append({"intent": intent, "raw_response": raw, "usage": bill})
    failed_intent, failed_usage = intents[-1], usage[-1]
    if (failed_intent["attempt_id"] != "request-00156" or failed_intent["group_id"] != "E2:69"
            or failed_intent["traveler"] != "Olivia" or failed_intent["operation"] != "agent"
            or failed_usage.get("request_sent") is not True or failed_usage["cost_rub"] is not None
            or failed_usage["total_tokens"] is not None or failed_events[0]["attempt_id"] != failed_intent["attempt_id"]):
        raise ValueError("Parent failed boundary differs from audited request")
    if len(rows(directory / "session_completions.jsonl")) != 6:
        raise ValueError("Parent complete-session prefix changed")
    return {"run_id": parent_id, "directory": directory, "manifest": manifest, "protocol": protocol,
        "accepted": accepted, "failed_intent": failed_intent, "failed_usage": failed_usage,
        "retrievals": rows(directory / "retrievals.jsonl"),
        "file_sha256": tree_hashes(directory)}


def expected_payload(config, intent):
    agent = config["agent"]
    extraction = intent["operation"] == "extraction"
    cap = agent["max_extraction_tokens"] if extraction else agent["max_completion_tokens"]
    payload = {"model": agent["model"], "messages": intent["messages"], "temperature": agent["temperature"],
        "max_tokens": min(intent["max_tokens_override"] or cap, cap), "seed": config["experiment"]["seed"]}
    if intent["tools"]:
        payload.update(tools=intent["tools"], tool_choice="auto")
    if intent["json_output"]:
        payload["response_format"] = {"type": "json_object"}
    if intent["response_schema"] is not None:
        payload["response_format"] = {"type": "json_schema", "json_schema": {
            "name": "constraint_extraction", "schema": intent["response_schema"], "strict": True}}
    payload["provider"] = agent["routing"]
    payload["reasoning"] = agent["extraction_reasoning"] if extraction else agent["reasoning"]
    return payload


class RecoveryClient(frozen.JournalClient):
    def __init__(self, parent, destination, *, live=False, gate_callback=None, **kwargs):
        self.parent = parent
        self.cursor = 0
        self.live = live
        self.gate_callback = gate_callback
        self.boundary_verified = False
        self.new_attempts = 0
        self.replay_entry = None
        self.active_intent = None
        self.retrieval_cursor = 0
        seen_ids = set()
        for index, entry in enumerate(parent["accepted"], 1):
            raw, bill, intent = entry["raw_response"], entry["usage"], entry["intent"]
            if (intent["attempt_id"] != f"request-{index:05d}" or not raw.get("id")
                    or raw["id"] != bill["request_id"] or raw["id"] in seen_ids):
                raise ReplayMismatch("Cached receipt identity, uniqueness or order differs from original usage")
            seen_ids.add(raw["id"])
            if (raw["usage"]["prompt_tokens"], raw["usage"]["completion_tokens"], raw["usage"]["total_tokens"], raw["usage"]["cost_rub"]) != (
                    bill["input_tokens"], bill["output_tokens"], bill["total_tokens"], bill["cost_rub"]):
                raise ReplayMismatch("Cached provider receipt usage differs from saved accounting")
        self.network_transport = kwargs.pop("transport", None)
        kwargs.setdefault("usage_path", Path(destination) / "usage.jsonl")
        super().__init__(destination=destination, transport=self._dispatch, **kwargs)
        config, agent = parent["manifest"]["config"], parent["manifest"]["config"]["agent"]
        for attribute, value in (("model_name", agent["model"]), ("temperature", agent["temperature"]),
                ("max_tokens", agent["max_completion_tokens"]), ("extraction_max_tokens", agent["max_extraction_tokens"]),
                ("seed", config["experiment"]["seed"]), ("provider", agent["routing"]),
                ("reasoning", agent["reasoning"]), ("extraction_reasoning", agent["extraction_reasoning"]), ("stream", False)):
            if canonical(getattr(self, attribute)) != canonical(value):
                raise ReplayMismatch("Recovery inference setting changed: " + attribute)
        if self.max_total_tokens != math.inf or self.max_cost_rub is not None:
            raise ReplayMismatch("Recovery adds an undeclared resource stop")

    def complete(self, messages, tools=None, operation=None, max_tokens=None, json_output=False, response_schema=None):
        actual = {"group_id": self.group_id, "traveler": self.traveler, "operation": operation or self.operation,
            "messages": messages, "tools": tools, "max_tokens_override": max_tokens,
            "json_output": json_output, "response_schema": response_schema}
        cached = self.cursor < len(self.parent["accepted"])
        expected = self.parent["accepted"][self.cursor]["intent"] if cached else (
            self.parent["failed_intent"] if not self.boundary_verified else None)
        if expected is not None and canonical(actual) != canonical({key: expected[key] for key in PAYLOAD_FIELDS}):
            frozen.durable(self.destination / "replay_mismatch.json", {"cache_position": self.cursor,
                "actual": actual, "expected": {key: expected[key] for key in PAYLOAD_FIELDS}})
            raise ReplayMismatch("Regenerated request differs from parent at cache position " + str(self.cursor))
        self.active_intent = actual
        self.replay_entry = self.parent["accepted"][self.cursor] if cached else None
        if not cached and not self.boundary_verified:
            if self.gate_callback is None:
                if self.live:
                    raise ReplayMismatch("Live continuation requires a full prefix-verification gate")
            else:
                self.gate_callback()
            if not self.live:
                raise ReplayBoundaryReached("All cached responses and the failed-request boundary match; no request sent")
            self.boundary_verified = True
            frozen.durable(self.destination / "replacement_attempt.json", {
                "parent_run_id": self.parent["run_id"], "parent_attempt_id": self.parent["failed_intent"]["attempt_id"],
                "parent_request_id": self.parent["failed_usage"].get("request_id"),
                "policy": "One deliberate new attempt after exact replay, charged inside original 30-step arrival allowance",
                "original_bill_known": False, "stream": False})
        try:
            raw = super().complete(messages, tools, operation, max_tokens, json_output, response_schema)
            if cached:
                self.cursor += 1
            return raw
        finally:
            self.replay_entry = None
            self.active_intent = None

    def _dispatch(self, method, path, payload):
        if method != "POST" or path != "/chat/completions":
            raise ReplayMismatch("Recovery transport accepts only the native completion endpoint")
        if canonical(payload) != canonical(expected_payload(self.parent["manifest"]["config"], self.active_intent)):
            raise ReplayMismatch("Generated completion payload differs from frozen inference contract")
        if self.replay_entry is not None:
            return copy.deepcopy(self.replay_entry["raw_response"])
        if not self.live or not self.boundary_verified:
            raise ReplayBoundaryReached("New requests are disabled until full replay verification")
        self.new_attempts += 1
        if self.network_transport is not None:
            return self.network_transport(method, path, payload)
        # The native authenticated transport retains its no-ambiguous-POST-retry policy.
        self.transport = None
        try:
            return PolzaClient._request(self, method, path, payload)
        finally:
            self.transport = self._dispatch

    def _record_usage(self, record):
        record = copy.deepcopy(record)
        record.update(reused=self.replay_entry is not None, new_billable=self.replay_entry is None,
            origin_run_id=self.parent["run_id"] if self.replay_entry is not None else self.destination.name)
        if self.replay_entry is not None:
            record.update(parent_attempt_id=self.replay_entry["intent"]["attempt_id"],
                replay_latency_seconds=record["latency_seconds"], latency_seconds=self.replay_entry["usage"]["latency_seconds"])
        super()._record_usage(record)

    def get_usage_stats(self):
        result = super().get_usage_stats()
        inherited = sum((Decimal(str(entry["usage"]["cost_rub"])) for entry in self.parent["accepted"]), Decimal(0))
        new = [row for row in self.records if not row.get("reused")]
        new_known = sum((Decimal(str(row["cost_rub"])) for row in new if row["cost_rub"] is not None), Decimal(0))
        result.update(inherited_accepted_responses=self.cursor, new_request_attempts=self.new_attempts,
            inherited_known_cost_rub_exact=str(inherited), new_known_cost_rub_exact=str(new_known),
            combined_known_cost_rub_exact=str(inherited + new_known), total_cost_rub=None,
            total_input_tokens=None, total_output_tokens=None, inherited_failed_attempts=1,
            inherited_failed_usage=self.parent["failed_usage"],
            accounting_note="Reused receipts are not new charges. Original sent failed request remains unknown and separate from active execution records.")
        return result


class RecoveryPlanner(PlannerWorker):
    def __init__(self, upstream_root, db_path, client, **kwargs):
        self.inherited_failed_dispatch = int(client.group_id == "E2:69" and kwargs.get("causal_round") == 7)
        if self.inherited_failed_dispatch:
            kwargs["max_steps"] -= 1
        super().__init__(upstream_root, db_path, client, **kwargs)

    def run_turn(self, request, memory_context, base_person):
        result = super().run_turn(request, memory_context, base_person)
        if self.inherited_failed_dispatch:
            result["parent_model_requests"] += 1
            result["total_steps"] += 1
            result["inherited_failed_model_dispatches"] = 1
        return result


def semantic_row(name, original):
    row = copy.deepcopy(original)
    if name == "traces.jsonl":
        result = row["result"]
        result.pop("latency_seconds", None)
        for tool in result.get("structured_tool_results", []):
            tool.pop("latency_seconds", None)
    if name == "session_completions.jsonl":
        item = row["row"]
        item.pop("wall_clock_seconds", None)
        for usage in item["usage"]:
            for key in ("latency_seconds", "replay_latency_seconds", "reused", "new_billable", "origin_run_id", "parent_attempt_id"):
                usage.pop(key, None)
    return row


def compare_prefix(parent_dir, child_dir):
    parent_dir, child_dir = Path(parent_dir), Path(child_dir)
    compared = {}
    names = ("events.jsonl", "memory_ops.jsonl", "extractions.jsonl", "retrievals.jsonl", "recoveries.jsonl",
        "submission_events.jsonl", "predictions.jsonl", "traces.jsonl", "session_completions.jsonl")
    for name in names:
        before = [semantic_row(name, row) for row in rows(parent_dir / name)]
        after = [semantic_row(name, row) for row in rows(child_dir / name)]
        if canonical(before) != canonical(after):
            first = next((i for i, (a, b) in enumerate(zip(before, after)) if canonical(a) != canonical(b)), min(len(before), len(after)))
            raise ReplayMismatch(f"Reconstructed {name} differs at row {first}; parent={len(before)}, child={len(after)}")
        compared[name] = {"rows": len(before), "semantic_sha256": hashlib.sha256(canonical(before).encode()).hexdigest()}
    expected = {path.relative_to(parent_dir / "snapshots").as_posix(): read(path) for path in (parent_dir / "snapshots").rglob("*.json")}
    actual = {path.relative_to(child_dir / "snapshots").as_posix(): read(path) for path in (child_dir / "snapshots").rglob("*.json")}
    if canonical(expected) != canonical(actual):
        raise ReplayMismatch("Reconstructed saved snapshots differ")
    return {"status": "exact_semantic_prefix", "logs": compared, "snapshots": len(expected),
        "timing_exclusions": ["traces.result.latency_seconds", "traces.result.structured_tool_results[].latency_seconds",
            "session_completions.row.wall_clock_seconds", "session_completions.row.usage[].latency_seconds"],
        "replay_annotations_removed_from_usage": ["replay_latency_seconds", "reused", "new_billable", "origin_run_id", "parent_attempt_id"]}


def install_historical_seed_order(session, parent, client):
    """Restore a recorded set-enumeration order, never change seed membership.

    _named_travelers returns a Python set; its process-random ordering was not
    pinned in the original study. Historical required IDs supply only an order
    for seeds independently recomputed from the same current causal store.
    Exact full context/prompt/snapshot verification is still mandatory.
    """
    if not parent.get("retrievals") or not hasattr(session, "backend"):
        return session
    native_begin = session.begin

    def begin(request):
        if client.retrieval_cursor >= len(parent["retrievals"]):
            return native_begin(request)
        expected = parent["retrievals"][client.retrieval_cursor]
        if (expected["variant"], str(expected["group_id"]), expected["traveler"]) != (session.variant, session.group_id, request["traveler"]):
            raise ReplayMismatch("Historical retrieval sequence differs")
        original_seeds = session.backend._seed_ids
        def ordered_seeds(current_request, step):
            seeds = original_seeds(current_request, step)
            target = expected["required_closure_ids"]

            def required(candidate):
                # Evaluate read-only copies; trial traversal diagnostics must
                # never leak into the actual native begin_subtask operation.
                diagnostics = session.backend._diagnostics
                metrics = session.backend._metrics
                session.backend._diagnostics = copy.deepcopy(diagnostics)
                session.backend._metrics = copy.deepcopy(metrics)
                try:
                    if session.backend.features["dependency_closure"]:
                        records, _ = session.backend._closure(candidate, step, retrieval=True)
                        return [record["record_ref"] for record in records]
                    result = []
                    for ref in candidate:
                        record = session.backend._resolve(ref, step, diagnostics=False)
                        if record:
                            identity = record.get("constraint_id", record.get("decision_id", record.get("term_id")))
                            result.append(f"{identity}@v{record['version']}")
                    return result
                finally:
                    session.backend._diagnostics = diagnostics
                    session.backend._metrics = metrics

            ordered, chosen_names, attempts = seeds, None, 0
            if required(seeds) != target:
                native_named = session.backend._named_travelers
                query = str(current_request.get("query", current_request.get("request", "")))
                names = sorted(native_named(query))
                if len(names) > 8:
                    raise ReplayMismatch("Historical name enumeration exceeds this study's eight-person bound")
                # E1 exposes the original named blocks directly. Try that
                # order first; for E2 it is only a candidate and must pass the
                # same native closure check. This avoids thousands of costly
                # native seed recomputations when seven people are mentioned.
                def first_record(name):
                    return next((index for index, ref in enumerate(target)
                                 if ref.startswith(f"decision:{name}:")), len(target))
                hinted_names = sorted(names, key=first_record)
                matched = False
                seen = set()
                for permutation in itertools.chain((hinted_names,), itertools.permutations(names)):
                    try:
                        session.backend._named_travelers = lambda _query: list(permutation)
                        candidate = original_seeds(current_request, step)
                    finally:
                        session.backend._named_travelers = native_named
                    if sorted(candidate) != sorted(seeds):
                        raise ReplayMismatch("Historical name ordering changed seed membership")
                    key = tuple(candidate)
                    if key in seen:
                        continue
                    seen.add(key)
                    attempts += 1
                    if required(candidate) == target:
                        ordered, chosen_names, matched = candidate, list(permutation), True
                        break
                if not matched:
                    raise ReplayMismatch("No native named-set enumeration reproduces the historical required closure order")
            frozen.durable(client.destination / "historical_seed_order.jsonl", {"variant": session.variant,
                "group_id": session.group_id, "traveler": request["traveler"],
                "same_seed_multiset": True, "native_seeds": seeds, "replayed_seeds": ordered,
                "named_set_order": chosen_names, "trial_unique_permutations": attempts,
                "required_closure_order_verified": True}, append=True)
            return ordered

        session.backend._seed_ids = ordered_seeds
        try:
            context = native_begin(request)
        finally:
            session.backend._seed_ids = original_seeds
        client.retrieval_cursor += 1
        return context

    session.begin = begin
    return session


def execute_grid(root, destination, parent, tokenizer, readiness, client):
    protocol = copy.deepcopy(parent["protocol"])
    config = protocol["config"]
    config["agent"].update(stream=False, request_timeout_seconds=1800)
    predictions = {variant: [] for variant in frozen.VARIANTS}
    all_rows = {variant: [] for variant in frozen.VARIANTS}
    for index, item in enumerate(protocol["schedule"], 1):
        print(f"SESSION {index}/14 group={item['group_id']} {item['variant']} (cached prefix is not rebilled)", flush=True)
        session = frozen.GroupSession(root, destination, item["variant"], item["group_id"], config,
            readiness, tokenizer, client, RecoveryPlanner, frozen.format_base)
        current, row = install_historical_seed_order(session, parent, client).run()
        if item == {"group_id": "69", "variant": "E2"}:
            row["usage"].append({**parent["failed_usage"], "origin_run_id": parent["run_id"],
                "inherited_failed_dispatch": True, "new_billable": False})
        frozen.durable(destination / "session_completions.jsonl", {"execution_index": index,
            "group_id": item["group_id"], "variant": item["variant"], "predictions": current, "row": row}, append=True)
        predictions[item["variant"]].extend(current)
        all_rows[item["variant"]].append(row)
    if not client.live:
        raise ReplayMismatch("Offline replay crossed its required failed-request boundary")
    frozen.durable(destination / "planning_complete.json", {"sessions": 14, "parent_run_id": parent["run_id"]})
    result_rows = []
    for variant in frozen.VARIANTS:
        scores = score_official(root / config["experiment"]["upstream_root"], root / "data/hidden/memoryarena_travel.jsonl",
            predictions[variant], protocol["group_ids"], expected_dataset_sha=readiness["provenance"]["dataset_sha256"])
        frozen.durable(destination / f"official_scores_{variant}.json", scores)
        for row in all_rows[variant]:
            parent_rows = {(item["group_id"], item["variant"]): item["row"] for item in rows(parent["directory"] / "session_completions.jsonl")}
            prior = parent_rows.get((row["group_id"], row["variant"]))
            if prior is not None:
                row["reconstruction_wall_clock_seconds"] = row["wall_clock_seconds"]
                row["wall_clock_seconds"] = prior["wall_clock_seconds"]
                row["usage"] = [{**copy.deepcopy(item), "reused": True, "new_billable": False,
                    "origin_run_id": parent["run_id"]} for item in prior["usage"]]
                row["timing_scope"] = "Original complete parent session; reconstruction time reported separately"
            elif (row["group_id"], row["variant"]) == ("69", "E2"):
                row["reconstruction_plus_new_wall_clock_seconds"] = row["wall_clock_seconds"]
                row["wall_clock_seconds"] = None
                for item in row["usage"]:
                    if not item.get("request_status"):
                        item["reconstruction_latency_seconds"] = item["latency_seconds"]
                        item["latency_seconds"] = None
                row["timing_scope"] = "Mixed replay/new cell; wall clock and local/tool timing unavailable for comparison; received model latencies retain original/new provenance"
            else:
                row["timing_scope"] = "New live continuation session"
            score = next(item for item in scores["per_group"] if str(item["group_id"]) == row["group_id"])
            if score["base_template_id"] != row["base_template_id"]:
                raise ValueError("Official scorer family differs")
            row["official_metrics"] = score["official_metrics"]
            row["official_metric_metadata"] = scores.get("official_metric_metadata", scores.get("metric_metadata", {}))
            people = {int(p["traveler_index"]) - 1: p for p in score.get("traveler_metrics", [])}
            row["travelers"] = [{**person, **people.get(person["traveler_index"], {}), "traveler_index": person["traveler_index"]} for person in row["travelers"]]
            row["official_metric_name"] = scores.get("metric_definition", {}).get("SPS", "upstream SPS")
            row["official_metric_version"] = config["experiment"]["upstream_repo_sha"]
            frozen.durable(destination / "results.jsonl", row, append=True)
            result_rows.append(row)
    return result_rows


def verify_recovery_freeze(parent):
    pin = read(FREEZE)
    if frozen.source_map(ROOT, "experiments/cross_family_recovery") != pin["recovery_source_sha256"]:
        raise ValueError("Recovery source differs from tested freeze")
    if parent["file_sha256"] != pin["parent_files_sha256"] or pin["unit_tests"]["passed"] < 1:
        raise ValueError("Parent evidence or tested recovery pin differs")
    if sha256(REPORTS / "replay_verified.json") != pin["replay_verified_sha256"]:
        raise ValueError("Required actual-prefix replay verification changed")
    frozen.verify_freeze(ROOT)
    return pin


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-run", default=PARENT_ID)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--run-id")
    args = parser.parse_args(argv)
    if not args.live:
        def deny_network(event, _args):
            if event.startswith("socket.") or event == "http.client.connect":
                raise PermissionError("Offline recovery cannot access the network")
        sys.addaudithook(deny_network)
    with SuiteLock(ROOT):
        parent = read_parent(ROOT, args.parent_run)
        starting_recovery_source = frozen.source_map(ROOT, "experiments/cross_family_recovery")
        recovery_pin = verify_recovery_freeze(parent) if args.live else None
        if args.live:
            verify_unused_lineage(ROOT, parent["run_id"])
        readiness = verify_readiness(ROOT)
        tokenizer_path, tokenizer_manifest = verify_tokenizer(ROOT)
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        run_id = args.run_id or "cross-family-recovery-" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id) or run_id in (".", ".."):
            parser.error("Invalid run ID")
        destination = ROOT / "runs" / run_id if args.live else ROOT / "tmp/cross-family-prefix-replay" / uuid.uuid4().hex
        if destination.exists():
            parser.error("Use a new run directory; recovery never overwrites a run")
        destination.mkdir(parents=True)
        config = copy.deepcopy(parent["manifest"]["config"])
        config["agent"].update(stream=False, request_timeout_seconds=1800)
        manifest = {**copy.deepcopy(parent["manifest"]), "run_id": run_id, "status": "replaying_prefix",
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "config": config,
            "parent_run_id": parent["run_id"], "parent_files_sha256": parent["file_sha256"],
            "transport_changes": {"stream": {"parent": True, "continuation": False},
                "read_timeout_seconds": {"parent": 180, "continuation": 1800}},
            "inherited_accepted_responses": len(parent["accepted"]), "inherited_failed_usage": parent["failed_usage"],
            "failed_dispatch_budget": "Olivia E2:69: 1 accepted +1 sent failed +at most28new actor calls =30",
            "timing_policy": "Retain original timings for six completed parent sessions; mixed partial session wall-clock/local/tool timings unavailable; remaining sessions newly timed",
            "parent_directory_required_for_full_evidence": True,
            "live_comparison_validated": False}
        manifest.pop("error", None)
        manifest.pop("usage", None)
        frozen.durable(destination / "manifest.json", manifest)
        if args.live:
            archive_run_source(ROOT, destination)
            with zipfile.ZipFile(destination / "recovery_source.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
                for relative in recovery_pin["recovery_source_sha256"]:
                    archive.write(ROOT / relative, relative)
                archive.write(FREEZE, FREEZE.relative_to(ROOT).as_posix())
            frozen.durable(destination / "inherited_failed_usage.json", parent["failed_usage"])
            load_env(ROOT / ".env")
        client = None
        try:
            agent = config["agent"]
            with RecoveryClient(parent, destination, live=args.live,
                api_key=None if args.live else "offline-replay-no-network", model=agent["model"], base_url=agent["base_url"],
                temperature=agent["temperature"], max_tokens=agent["max_completion_tokens"], extraction_max_tokens=agent["max_extraction_tokens"],
                seed=config["experiment"]["seed"], provider=agent["routing"], reasoning=agent["reasoning"],
                extraction_reasoning=agent["extraction_reasoning"], stream=False, timeout=1800, retries=agent["retries"],
                max_total_tokens=math.inf, max_cost_rub=None) as client:
                def gate():
                    proof = compare_prefix(parent["directory"], destination)
                    if client.cursor != 155 or tree_hashes(parent["directory"]) != parent["file_sha256"]:
                        raise ReplayMismatch("Parent artifacts or cache coverage changed before continuation")
                    if frozen.source_map(ROOT, "experiments/cross_family_recovery") != starting_recovery_source:
                        raise ReplayMismatch("Recovery code changed during reconstruction")
                    proof.update(parent_run_id=parent["run_id"], accepted_responses_reused=155,
                        failed_request_intent_sha256=hashlib.sha256(canonical(parent["failed_intent"]).encode()).hexdigest(),
                        model_calls=0, parent_files_sha256=parent["file_sha256"], transport_changes=manifest["transport_changes"],
                        recovery_source_sha256=starting_recovery_source,
                        historical_seed_order="Only permutation of freshly recomputed native seeds; original retrieval order verified by full logs/prompts/snapshots")
                    frozen.durable(destination / "boundary_verified.json", proof)
                    if args.live:
                        verify_recovery_freeze(parent)
                        verify_unused_lineage(ROOT, parent["run_id"], exclude_child=destination)
                        claim_lineage(ROOT, parent["run_id"], destination)
                        manifest["status"] = "continuing_after_verified_boundary"
                        frozen.durable(destination / "manifest.json", manifest)
                    else:
                        frozen.durable(REPORTS / "replay_verified.json", proof)
                client.gate_callback = gate
                if args.live:
                    client.freeze_check = lambda: verify_recovery_freeze(parent)
                result_rows = execute_grid(ROOT, destination, parent, tokenizer, readiness, client)
                if tree_hashes(parent["directory"]) != parent["file_sha256"]:
                    raise ReplayMismatch("Parent artifacts changed during continuation")
                from constraint_memory.orchestration.replay import replay_run
                replay = replay_run(destination)
                if len(replay) != 14:
                    raise ReplayMismatch("Completed memory replay does not cover the full grid")
                from constraint_memory.evaluation import report_run
                report_run(destination, requested_metrics=("SR", "PS", "SPS"), primary_contrast=frozen.VARIANTS,
                    primary_endpoint="SR", bootstrap_samples=10000, seed=config["experiment"]["seed"])
                manifest.update(status="complete", completed_conditions=len(result_rows), replay_all_match=True,
                    live_comparison_validated=True, usage=client.get_usage_stats())
        except ReplayBoundaryReached:
            if args.live:
                raise
            manifest.update(status="offline_prefix_verified", usage=client.get_usage_stats())
            print("OFFLINE PREFIX VERIFIED: 155 responses reused, exact failed-request boundary; no model calls", flush=True)
            return 0
        except BaseException as error:
            manifest.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                error={"type": type(error).__name__, "message": redact(str(error))})
            if client is not None:
                manifest["usage"] = client.get_usage_stats()
            print("STOPPED " + run_id + ": " + redact(str(error)), flush=True)
            return 1
        finally:
            frozen.durable(destination / "manifest.json", manifest)
        print("COMPLETED " + run_id, flush=True)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
