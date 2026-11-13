"""Live MemoryArena execution using the pinned official worker and scorer.

Case preview is public UI data. Executions always start from a new graph and
release one current request; neither preview nor an archived graph is reused.
"""
import copy
import hashlib
import json
import time

import numpy as np

from src.constraint_memory.benchmark_adapter.data import ChronologicalAdapter, load_base, load_splits, load_turn, verify_readiness
from src.constraint_memory.benchmark_adapter.planner import PlannerWorker
from src.constraint_memory.benchmark_adapter.scoring import score_official
from src.constraint_memory.benchmark_adapter.submissions import SubmissionLedger
from src.constraint_memory.config import verify_tokenizer

from .adapters import canonical
from .engine import Engine, source_ids
from .memoryarena import ROOT
from .models import Action, BenchmarkInput, Settings
from .provider import client as make_client, safe_error, accounted_receipts, reconcile_failed_billing
from .store import Conflict, encoded, identity, now
from .snapshot import archive_source
from .traces import EXECUTION_KINDS, trace_record


class Stopped(Exception):
    pass


class MemoryOutputRejected(ValueError):
    """A fully billed model response failed the memory output contract."""


def case_detail(group_id):
    manifest = load_splits(ROOT)
    if group_id not in manifest["groups"]:
        raise KeyError("MemoryArena case not found")
    meta = manifest["groups"][group_id]
    partition = next(p for p in ("dev", "validation", "holdout") if int(group_id) in manifest["partitions"][p])
    return {"group_id": group_id, "partition": partition, "traveler_count": meta["traveler_count"], "family_id": meta["base_template_id"], "base": load_base(ROOT, group_id), "travelers": [load_turn(ROOT, group_id, i) for i in range(1, meta["traveler_count"] + 1)], "protocol": "Pinned public MemoryArena CSV tool planner; fresh worker per traveler, chronological requests, final isolated official scoring.", "metrics": {"SR": "All expected added travelers pass their full plan", "PS": "Fraction of expected travelers passing their full plan", "SPS": "Mean fraction of changed slots passed; upstream SPS, not assumed paper sPS"}, "preview_is_not_planner_input": True}


def session(store, run_id):
    with store.connection() as db:
        row = db.execute("SELECT data,control FROM benchmark_sessions WHERE run_id=?", (run_id,)).fetchone()
    if not row:
        raise KeyError("This run is not a benchmark execution")
    value = json.loads(row["data"])
    value["control"] = json.loads(row["control"])
    if value.get("job_id"):
        job = store.job(run_id, value["job_id"])
        value["job_status"] = job["status"]
        if job["status"] in {"failed", "cancelled"}:
            value["status"] = job["status"]
            value["error"] = job["error"]
    return value


def control(store, run_id, command):
    current = session(store, run_id)
    if current["status"] in {"completed", "failed", "cancelled"}:
        raise Conflict("Execution is finished; create a new execution to run again")
    with store.connection() as db:
        state = current["control"]
        if command == "run":
            state.update(mode="running", permits=0)
        elif command == "pause":
            state.update(mode="paused", permits=0)
        elif command == "step":
            state.update(mode="paused", permits=state.get("permits", 0) + 1)
        elif command == "stop":
            state["stop"] = True
        else:
            raise ValueError("Unknown execution control")
        db.execute("UPDATE benchmark_sessions SET control=? WHERE run_id=?", (encoded(state), run_id))
        store.event(db, run_id, "benchmark.control", {"command": command, "control": state})
    return session(store, run_id)


def prepare(store, body: BenchmarkInput):
    detail = case_detail(body.group_id)
    if body.settings.mode != "polza":
        raise ValueError("Live benchmark execution requires Polza, never Demo")
    if body.memory_mode in {"A", "B", "C", "D"} and body.settings.memory_design != "entity_grounded_v1":
        raise ValueError("A–D require the entity-grounded design")
    if body.memory_mode not in {"A", "B", "C", "D"} and body.settings.memory_design != "legacy":
        raise ValueError("Legacy MemoryArena execution cannot be labeled entity-grounded. Use a frozen entity-grounded experiment adapter")
    settings = body.settings.model_copy(update={"dataset_id": "memoryarena-" + body.group_id})
    run = store.create(f"MemoryArena {body.group_id} · live {body.memory_mode}", settings.model_dump())
    archive_source(store, run["id"])
    config = body.model_dump()
    state = {"run_id": run["id"], "group_id": body.group_id, "status": "queued", "phase": "preflight", "current_traveler": 0, "traveler_count": detail["traveler_count"], "partition": detail["partition"], "config": config, "actions": [], "predictions": [], "travelers": [], "memory_reads": [], "scores": None, "created_at": now(), "protocol": detail["protocol"], "evaluation_scope": "Single-case exploratory execution; not a generalization or superiority claim", "memory_acceptance": "Independent machine validation during automatic execution; never labeled human approval"}
    with store.connection() as db:
        db.execute("INSERT INTO benchmark_sessions VALUES(?,?,?)", (run["id"], encoded(state), encoded({"mode": "paused" if body.start_paused else "running", "permits": 0, "stop": False})))
    request = Action(action="benchmark", parameters={"group_id": body.group_id}, expected_graph_version=0, idempotency_key="benchmark-execution")
    job, _ = store.reserve(run["id"], request)
    state["job_id"] = job["id"]
    with store.connection() as db:
        db.execute("UPDATE benchmark_sessions SET data=? WHERE run_id=?", (encoded(state), run["id"]))
    return run, job


def resume_checkpoint(store, run_id):
    state = session(store, run_id)
    if state["status"] != "failed" or not state["travelers"] or state["current_traveler"] != len(state["travelers"]):
        raise Conflict("Recovery requires a failed execution at a fully committed traveler boundary; partial/in-flight travelers cannot be repeated automatically")
    if "Server restarted before completion" in (state.get("error") or ""):
        raise Conflict("Server interruption left possible in-flight inference. Automatic continuation is unavailable until that request is reconciled; no paid request will be repeated automatically")
    reconcile_failed_billing(store, run_id)
    if any(r["data"].get("cost_unit") == "RUB" and r["data"].get("cost") is None for r in accounted_receipts(store.receipts(run_id))):
        raise Conflict("A paid request has unresolved billing; reconcile it before continuing this execution")
    previous = {"job_id": state["job_id"], "error": state.get("error"), "at": now(), "completed_travelers": len(state["travelers"]), "policy": "Restore actual committed submissions; skip failed maintenance at that boundary; do not repeat paid planning."}
    version = store.run(run_id)["version"]
    job, _ = store.reserve(run_id, Action(action="benchmark", parameters={"group_id": state["group_id"], "resume": True}, expected_graph_version=version, idempotency_key="resume-after-" + state["job_id"]))
    state.setdefault("recoveries", []).append(previous)
    state.setdefault("memory_failures", []).append({"traveler": state["current_traveler"], "reason": previous["error"], "policy": previous["policy"]})
    state.update(job_id=job["id"], status="queued", error=None, scores=None, phase="restoring_checkpoint")
    state.pop("job_status", None)
    state.pop("control", None)
    archive_source(store, run_id, suffix=job["id"])
    with store.connection() as db:
        db.execute("UPDATE benchmark_sessions SET data=?,control=? WHERE run_id=?", (encoded(state), encoded({"mode": "running", "permits": 0, "stop": False}), run_id))
        store.event(db, run_id, "benchmark.checkpoint_resumed", previous)
    return job


class Execution:
    def __init__(self, store, engine, run_id, job_id, *, planner_factory=PlannerWorker, client_factory=make_client, scorer=score_official, readiness=verify_readiness):
        self.store, self.engine, self.run_id, self.job_id = store, engine, run_id, job_id
        self.planner_factory, self.client_factory, self.scorer, self.readiness = planner_factory, client_factory, scorer, readiness
        self.state = session(store, run_id)
        self.settings = Settings(**store.run(run_id)["settings"])
        self.config = self.state["config"]
        self.seen = set()
        self.seen.update(n.get("source_identity") for n in store.run(run_id)["graph"]["nodes"] if n["type"] == "trace")
        self.turn_ids = []
        self.turn = 0
        self.current_step = 0
        self.client = None
        self.worker_audits = []
        self.pending_trace = []

    def save(self, phase=None, **fields):
        if phase:
            self.state["phase"] = phase
        self.state.update(fields, updated_at=now())
        with self.store.connection() as db:
            db.execute("UPDATE benchmark_sessions SET data=? WHERE run_id=?", (encoded(self.state), self.run_id))
            self.store.event(db, self.run_id, "benchmark.progress", {"status": self.state["status"], "phase": self.state["phase"], "traveler": self.turn, "step": self.current_step, "completed_travelers": len(self.state["travelers"]), "action_count": len(self.state["actions"])})

    def gate(self, label):
        waiting = False
        while True:
            with self.store.connection() as db:
                row = db.execute("SELECT control FROM benchmark_sessions WHERE run_id=?", (self.run_id,)).fetchone()
                policy = json.loads(row["control"])
                if policy.get("stop") or self.store.job(self.run_id, self.job_id)["status"] != "running":
                    raise Stopped("Execution stopped at an action boundary; committed observations and paid receipts retained")
                allowed = policy["mode"] == "running" or policy.get("permits", 0) > 0
                if allowed and policy["mode"] == "paused":
                    policy["permits"] -= 1
                    db.execute("UPDATE benchmark_sessions SET control=? WHERE run_id=?", (encoded(policy), self.run_id))
            if allowed:
                self.save(label, status="running", waiting_for_step=False)
                return
            if not waiting:
                self.save(label, status="paused", waiting_for_step=True)
                waiting = True
            time.sleep(.2)

    def patch(self, graph, result, event_id, final=False):
        current = self.store.run(self.run_id)
        if not self.store.commit(self.run_id, self.job_id, current["version"], graph, result, event_id, final=final):
            raise Stopped("Execution output was cancelled or invalidated before commit")

    def observe(self, kind, content, key, **metadata):
        if key in self.seen:
            return
        self.seen.add(key)
        if not isinstance(content, str):
            content = encoded(content)
        if not content.strip():
            return
        if kind in EXECUTION_KINDS:
            record = {"content": content, "event_kind": kind, "source_event_id": key,
                      "session": f"live-{self.run_id}-traveler-{self.turn}",
                      "step": self.current_step, "traveler_index": self.turn, **metadata}
            self.pending_trace.append(record)
            with self.store.connection() as db:
                self.store.event(db, self.run_id, "benchmark.execution_event", record)
            return
        graph = self.store.run(self.run_id)["graph"]
        eid, nid = identity("event"), identity("trace")
        graph["nodes"].append({"id": nid, "run_id": self.run_id, "type": "trace", "level": 0, "title": f"T{self.turn} · {kind} · {content[:65]}", "content": content, "content_hash": hashlib.sha256(content.encode()).hexdigest(), "source_identity": key, "status": "recorded", "validation_state": "source_record", "version": 1, "source_version": graph["graph_version"], "created_at": now(), "created_by_event": eid, "session": f"live-{self.run_id}-traveler-{self.turn}", "provenance": "Live observed MemoryArena public input / actual agent or official tool execution", "source_metadata": {"group_id": self.state["group_id"], "traveler_index": self.turn, "kind": kind, "step": self.current_step, **metadata}, "explanation": "Observed by the live benchmark bridge; not a gold answer or inferred success"})
        self.turn_ids.append(nid)
        self.patch(graph, {"action": "benchmark.observe", "kind": kind, "created_ids": [nid], "updated_ids": []}, eid)

    def flush_trace(self, incomplete=False):
        if not self.pending_trace:
            return
        record = trace_record(self.pending_trace)
        self.pending_trace = []
        self.observe("execution_trace", record["content"], record["source_event_id"],
                     events=record["events"], trace_schema=record["trace_schema"], incomplete=incomplete)

    def operation(self, action, targets=(), parameters=None):
        # One permit covers a complete memory operation, including a separate
        # verifier when crystallizing. Cached operations obey the same boundary.
        self.gate("memory." + action)
        current = self.store.run(self.run_id)
        graph = copy.deepcopy(current["graph"])
        eid = identity("event")
        request = Action(action=action, targets=list(targets), parameters=parameters or {}, expected_graph_version=current["version"], idempotency_key=identity("auto"))
        started = time.perf_counter()
        receipt_start = len(self.store.receipts(self.run_id))
        try:
            result = self.memory_engine.operation(current, graph, request, self.job_id, eid)
        except ValueError as error:
            fresh = self.store.receipts(self.run_id)[receipt_start:]
            last = fresh[-1]["data"] if fresh else {}
            accounting = last.get("provider_accounting", [])
            known_response = accounting and accounting[-1].get("request_status") == "completed" and accounting[-1].get("response_accepted") is True and last.get("cost") is not None
            if action in {"episode", "concept", "validate", "crystallize"} and known_response:
                raise MemoryOutputRejected(f"{action}: {safe_error(error)}") from None
            raise
        result.update(action=action, wall_seconds=time.perf_counter() - started, acceptance_policy="independent_machine_validation" if action == "validate" else None)
        self.patch(graph, result, eid)
        self.state["actions"].append({"kind": "memory." + action, "traveler": self.turn, "event_id": eid, "wall_seconds": result["wall_seconds"], "created_ids": result["created_ids"], "updated_ids": result["updated_ids"]})
        self.save()
        return result

    def memory_context(self, request, ledger, tokenizer):
        mode = self.config["memory_mode"]
        if mode == "none":
            return ""
        graph = self.store.run(self.run_id)["graph"]
        nodes = graph["nodes"]
        candidates = []
        if mode == "mnemos":
            for n in nodes:
                if n["type"] == "episode" and n["status"] == "validated" and n.get("episode_schema") == "mnemos/granular-fact-v1":
                    candidates.append((1, n["id"], "Observed fact [" + n["id"] + "]: " + n["content"] + " Context: " + n["episode"]["context"] + " Evidence: " + ",".join(n["evidence_ids"])))
            for n in nodes:
                if n["type"] in {"concept", "meta_concept"} and n["status"] == "validated":
                    candidates.append((0, n["id"], "Machine-reviewed, fallible concept " + n["id"] + ": " + canonical(n) + " Evidence: " + ",".join(sorted(source_ids(graph, n["id"])))))
        # Latest ACTUAL submissions remain distinct from induced concepts.
        for index, submission in ledger.latest.items():
            priority = 1 if submission["name"].lower() in request["query"].lower() else 2
            candidates.append((priority, f"submission:{index}:v{submission['submission_version']}", f"Latest actual submission for {submission['name']} (not verified task success):\n{submission['final_plan']}"))
        for n in reversed(nodes):
            if n["type"] == "trace" and n["status"] == "recorded" and n.get("source_metadata", {}).get("kind") == "request" and n["source_metadata"]["traveler_index"] < self.turn:
                candidates.append((3, n["id"], "Original earlier request [" + n["id"] + "]:\n" + n["content"]))
        if mode == "mnemos":
            embedded = [n for n in nodes if n["type"] == "trace" and "vector" in n and n["status"] == "recorded" and n.get("source_metadata", {}).get("kind") == "execution_trace"]
            if embedded:
                q, _ = self.engine.models.embed([request["query"]], self.settings)
                for n in sorted(embedded, key=lambda n: -float(np.dot(q[0], n["vector"])))[:8]:
                    candidates.append((4, n["id"], "Observed execution trace [" + n["id"] + "]:\n" + n["content"]))
        header = "Memory from causally earlier execution only. Concepts are fallible machine interpretations; actual submissions do not prove correctness. Verify against original observations and tools.\n"
        text, included, omitted = header, [], []
        for _, reference, content in sorted(candidates, key=lambda item: item[0]):
            proposed = text + "\n" + content + "\n"
            if len(tokenizer.encode(proposed).ids) <= self.config["memory_token_budget"]:
                text = proposed
                included.append(reference)
            else:
                omitted.append(reference)
        read = {"traveler": self.turn, "mode": mode, "included_ids": included, "omitted_ids": omitted, "tokens": len(tokenizer.encode(text).ids), "budget": self.config["memory_token_budget"], "context": text}
        self.state["memory_reads"].append(read)
        self.save("memory.read")
        return text

    def consolidate(self):
        if self.config["memory_mode"] != "mnemos":
            return
        self.embed_pending()
        episode = self.operation("episode", self.turn_ids, {"count": 20})
        eids = episode["created_ids"]
        self.operation("validate", eids)
        graph = self.store.run(self.run_id)["graph"]
        accepted_facts = [n["id"] for n in graph["nodes"] if n["id"] in eids and n["status"] == "validated"]
        if len(accepted_facts) < 2:
            self.save("memory.facts_need_review")
            return
        concepts = self.operation("concept", accepted_facts)["created_ids"]
        self.operation("validate", concepts)
        self.embed_pending()
        graph = self.store.run(self.run_id)["graph"]
        if sum(n["type"] in {"concept", "meta_concept"} and n["status"] == "validated" for n in graph["nodes"]) >= 2:
            discovered = self.operation("associate")
            if self.config["auto_crystallize"] and discovered["created_ids"]:
                graph = self.store.run(self.run_id)["graph"]
                eligible = [h["id"] for h in graph["hyperedges"] if h["id"] in discovered["created_ids"] and h["level"] < self.settings.max_depth]
                if not eligible:
                    return
                edge = eligible[0]
                crystallized = self.operation("crystallize", [edge])
                if crystallized["created_ids"]:
                    self.operation("validate", crystallized["created_ids"])
                    self.embed_pending()
                    self.operation("associate")

    def embed_pending(self):
        graph = self.store.run(self.run_id)["graph"]
        if any("vector" not in n and n["status"] in {"recorded", "validated"} for n in graph["nodes"]):
            self.operation("embed")
            self.operation("project")

    def run(self):
        if not self.store.transition(self.run_id, self.job_id, "running"):
            return
        execution = self
        delegate = self.engine.models

        class GatedModels:
            def generate(self, operation, data, settings, receipt):
                execution.save("memory." + operation)
                return delegate.generate(operation, data, settings, receipt)
            def embed(self, texts, settings):
                return delegate.embed(texts, settings)
            def embedding_metadata(self, settings):
                return delegate.embedding_metadata(settings)

        self.memory_engine = Engine(self.store, GatedModels())
        try:
            self.save("preflight", status="running")
            ready = self.readiness(ROOT)
            from tokenizers import Tokenizer
            tokenizer_path, tokenizer_manifest = verify_tokenizer(ROOT)
            tokenizer = Tokenizer.from_file(str(tokenizer_path))
            causal = ChronologicalAdapter(ROOT, self.state["group_id"])
            ledger = SubmissionLedger(self.state["group_id"], causal.base["base_person"]["name"])
            self.state["pins"] = {"upstream_sha": ready["provenance"]["upstream_sha"], "dataset_sha256": ready["provenance"]["dataset_sha256"], "tokenizer": tokenizer_manifest}
            self.observe("base_request", causal.base["base_person"]["query"], "base-request")
            self.observe("public_base", causal.base["base_person"]["daily_plans"], "public-base")
            self.client = self.client_factory(self.settings)
            self.client.set_scope(self.run_id, operation="agent")

            class LivePlannerClient:
                def chat_with_tools(inner, messages, tools):
                    # Tool outputs are already observed by the worker before the NEXT paid decision.
                    for message in messages:
                        if message.get("role") == "tool":
                            execution.observe("tool_result", str(message.get("content", "")), f"tool-result:{execution.turn}:{message.get('tool_call_id')}", tool_call_id=message.get("tool_call_id"), tool_name=message.get("name"))
                    execution.flush_trace()
                    execution.state["agent_prompt_preview"] = {"messages": copy.deepcopy(messages), "sha256": hashlib.sha256(encoded(messages).encode()).hexdigest(), "traveler": execution.turn, "next_decision": execution.current_step + 1, "dispatch_status": "preview"}
                    execution.gate("planner.decision")
                    execution.current_step += 1
                    execution.save("planner.inference", waiting_for_step=False)
                    before = len(execution.client.records)
                    started = time.perf_counter()
                    raw_response, response = None, None
                    try:
                        response = execution.client.chat_with_tools(messages, tools)
                        raw_response = response.raw_response
                        if response.content:
                            execution.observe("agent_output", response.content, f"agent:{execution.turn}:{execution.current_step}")
                        for call in response.tool_calls or []:
                            execution.observe("tool_call", {"id": call.id, "name": call.name, "arguments": call.arguments}, f"tool-call:{execution.turn}:{call.id}")
                        return response
                    finally:
                        rows = execution.client.records[before:]
                        row = rows[-1] if rows else {}
                        execution.store.receipt(execution.run_id, execution.job_id, {"operation": "planner", "model": execution.settings.llm_model, "messages": messages, "tools": tools, "raw_output": raw_response, "provider_accounting": rows, "cost": row.get("cost_rub"), "cost_unit": "RUB", "usage": {"input_tokens": row.get("input_tokens"), "output_tokens": row.get("output_tokens"), "total_tokens": row.get("total_tokens")}, "traveler": execution.turn, "step": execution.current_step, "wall_seconds": time.perf_counter() - started})
                        execution.state["actions"].append({"kind": "planner.decision", "traveler": execution.turn, "step": execution.current_step, "cost_rub": row.get("cost_rub"), "tool_calls": [{"id": t.id, "name": t.name, "arguments": t.arguments} for t in response.tool_calls or []] if response else [], "completed": response is not None})
                        execution.save("planner.action_completed")

            for request in causal.iter_requests():
                self.turn = request["traveler_index"]
                checkpoint = next((t for t in self.state["travelers"] if t["traveler_index"] == self.turn), None)
                if checkpoint:
                    ledger.register_request(request)
                    restored = {"group_id": self.state["group_id"], "traveler_index": self.turn, "final_plan": checkpoint["actual_final_plan"], "accepted_submissions": checkpoint["accepted_revisions"]["accepted"], "total_steps": checkpoint["steps"], "error_message": checkpoint.get("error")}
                    ledger.accept(restored)
                    causal.commit_result(restored)
                    self.worker_audits.append(checkpoint["worker_audit"])
                    if self.turn == len(self.state["travelers"]):
                        reconstructed = ledger.predictions(variant="MNEMOS_" + self.config["memory_mode"].upper(), family=causal.base["base_template_id"])
                        for actual, saved in zip(reconstructed, self.state["predictions"]):
                            if any(actual.get(k) != saved.get(k) for k in ("traveler_index", "submitted_text", "plan", "submission_version", "source_round")):
                                raise ValueError("Committed checkpoint predictions do not replay exactly")
                        self.save("checkpoint.restored")
                    continue
                self.turn_ids, self.current_step = [], 0
                ledger.register_request(request)
                self.save("traveler.revealed", current_traveler=self.turn, current_name=request["name"])
                self.observe("request", request["query"], f"request:{self.turn}")
                context = self.memory_context(request, ledger, tokenizer)
                self.client.set_scope(self.run_id, traveler=request["name"], operation="agent")
                worker = self.planner_factory(ROOT / "external/MemoryArena", ROOT / ready["database"]["database_path"], LivePlannerClient(), model=self.settings.llm_model, temperature=self.settings.temperature, max_steps=self.config["max_steps_per_traveler"], timeout_seconds=86400, denied_paths=[ROOT / "mnemos/state", ROOT / "runs"], visible_travelers=ledger.visible_travelers, causal_round=self.turn)
                result = worker.run_turn(request, context, causal.base["base_person"])
                audit = result.get("worker_audit", {})
                if audit.get("hidden_modules_loaded") or audit.get("model_credentials_present"):
                    raise ValueError("Planner isolation audit failed")
                self.worker_audits.append(audit)
                for tool in result.get("structured_tool_results", []):
                    self.observe("tool_result", tool["formatted_result"], f"tool-result:{self.turn}:{tool['tool_call_id']}", tool_name=tool["tool_name"], tool_call_id=tool["tool_call_id"])
                accepted = ledger.accept(result)
                self.observe("submission", result.get("final_plan") or "No final plan was submitted within the step allowance.", f"submission:{self.turn}", name=request["name"], accepted_submissions=accepted)
                self.flush_trace()
                causal.commit_result(result)
                self.state["travelers"].append({"traveler_index": self.turn, "name": request["name"], "actual_final_plan": result.get("final_plan", ""), "steps": result.get("total_steps"), "planner_success": result.get("success"), "error": result.get("error_message"), "accepted_revisions": accepted, "worker_audit": audit})
                self.state["predictions"] = ledger.predictions(variant="MNEMOS_" + self.config["memory_mode"].upper(), family=causal.base["base_template_id"])
                self.save("traveler.committed")
                try:
                    self.consolidate()
                except MemoryOutputRejected as error:
                    failure = {"traveler": self.turn, "reason": str(error), "policy": "Reject invalid memory output; preserve original evidence and actual submissions; continue planning without this failed update."}
                    self.state.setdefault("memory_failures", []).append(failure)
                    self.state["actions"].append({"kind": "memory.output_rejected", **failure})
                    with self.store.connection() as db:
                        self.store.event(db, self.run_id, "benchmark.memory_output_rejected", failure)
                    self.save("memory.output_rejected")
            self.gate("official.scoring")
            scores = self.scorer(ROOT / "external/MemoryArena", ROOT / "data/hidden/memoryarena_travel.jsonl", self.state["predictions"], [self.state["group_id"]], expected_dataset_sha=ready["provenance"]["dataset_sha256"])
            self.save("completed", status="completed", scores=scores, completed_at=now(), worker_audits=self.worker_audits)
            graph = self.store.run(self.run_id)["graph"]
            self.patch(graph, {"action": "benchmark.completed", "created_ids": [], "updated_ids": [], "official_metrics": scores["official_metrics"]}, identity("event"), final=True)
        except Stopped as error:
            self.save("cancelled", status="cancelled", error=str(error))
            self.store.transition(self.run_id, self.job_id, "cancelled", str(error))
        except Exception as error:
            self.flush_trace(incomplete=True)
            message = safe_error(error)
            self.save("failed", status="failed", error=message, scores=None)
            self.store.transition(self.run_id, self.job_id, "failed", message)
        finally:
            if self.client:
                self.client.close()
