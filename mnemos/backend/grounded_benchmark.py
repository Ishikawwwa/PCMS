"""Causal A–D MemoryArena adapter using the existing isolated official worker.

Generation proposes memory. Every mutation is held for explicit researcher
review; no rule is silently machine-promoted during a benchmark.
"""
import copy
import hashlib
import time
from .grounded_pipeline import memory_stages

from .benchmark import Execution, Stopped
from .grounded import GroundedEngine, config_id, envelope
from .grounded_context import assemble
from .grounded_schema import Command, ContextQuery
from .store import encoded, identity, now
from .traces import EXECUTION_KINDS


class GroundedExecution(Execution):
    def observe(self, kind, content, key, **metadata):
        if kind in EXECUTION_KINDS:
            return super().observe(kind, content, key, **metadata)
        if key in self.seen:
            return
        self.seen.add(key)
        if not isinstance(content, str):
            content = encoded(content)
        if not content.strip():
            return
        run = self.store.run(self.run_id)
        cmd = Command(operation_id=identity("observe"), graph_version=run["version"], config_id=config_id(run), type="import_traces")
        raw = metadata.get("events") or [{"kind": kind, "content": content}]
        events = [{"id": "event:" + str(i), "kind": e["kind"], "role": None, "content": e["content"], "observed_at": None,
                   "source_ref": key, "content_hash": hashlib.sha256(e["content"].encode()).hexdigest(), "start": 0, "end": len(e["content"])} for i, e in enumerate(raw)]
        n = envelope("trace", f"T{self.turn} · {kind} · {content[:65]}", cmd, run["version"], content=content, events=events,
                     status="recorded", source_identity=key, run_id=self.run_id, session=f"traveler:{self.turn}", session_id=f"traveler:{self.turn}",
                     scope="case:" + self.state["group_id"], content_hash=hashlib.sha256(content.encode()).hexdigest(),
                     source_metadata={"group_id": self.state["group_id"], "traveler_index": self.turn, "kind": kind, **metadata})
        graph = run["graph"]
        graph["nodes"].append(n)
        graph["schema_version"] = "mnemos/entity-grounded-v1"
        self.turn_ids.append(n["id"])
        self.patch(graph, {"action": "benchmark.observe", "created_ids": [n["id"]], "updated_ids": []}, cmd.operation_id)

    def memory_context(self, request, ledger, tokenizer):
        query = ContextQuery(query=request["query"], scope="case:" + self.state["group_id"], branch_id="main", variant=self.config["memory_mode"], token_budget=self.config["memory_token_budget"], max_hops=self.settings.retrieval_hops, max_nodes=self.settings.retrieval_nodes)
        run = self.store.run(self.run_id)
        value = assemble(run, query, tokenizer, {"model": "Qwen/Qwen3.6-27B", "revision": "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"})
        if value["status"] == "insufficient_context_budget":
            raise ValueError("insufficient_context_budget: required facts cannot fit; no silent incomplete context")
        # These are memory-context bytes. The worker's full controlled prompt,
        # including its permitted public base itinerary, is previewed at every
        # planner decision by LivePlannerClient and recorded verbatim in calls.
        text = value["messages"][1]["content"]
        self.state["memory_reads"].append({"traveler": self.turn, "mode": query.variant, "included_ids": [c["id"] for c in value["included"]], "omitted_ids": [c["id"] for c in value["excluded"]], "tokens": value["token_count"], "budget": query.token_budget, "context": text, "retrieval": value, "prompt_scope": "memory context only; actual complete planner messages are in agent_prompt_preview/call receipts"})
        self.save("memory.read")
        return text

    def consolidate(self):
        grounded = GroundedEngine(self.store, self.engine.models)
        variant = self.config["memory_mode"]
        stages = [{"type": s} for s in memory_stages(variant)]
        completed, index = [], 0
        while index < len(stages):
            entry = stages[index]
            stage = entry["type"]
            if stage == "rules":
                graph = self.store.run(self.run_id)["graph"]
                edges = [h for h in graph["hyperedges"] if h["status"] == "accepted" and h.get("created_operation_id") in completed]
                stages[index:index+1] = [{"type": "propose_rule", "inputs": [h["id"]]} for h in edges]
                continue
            self.save("memory.pipeline", memory_pipeline={"traveler": self.turn, "index": index, "stages": stages})
            self.gate("memory." + stage)
            run = self.store.run(self.run_id)
            inputs = self.turn_ids if stage == "propose_episodes" else []
            allowed = {"resolve_entities": {"episode"}, "propose_facts": {"episode", "entity"}, "propose_task_state": {"episode", "entity", "fact"},
                       "propose_concepts": {"episode", "fact", "state"}, "embed": {"concept"}, "project": {"concept", "rule"},
                       "propose_hyperedges": {"concept", "rule"}, "propose_direct_rule": {"episode", "fact", "state"}, "project_rules": {"concept", "rule"}}
            if stage in allowed:
                inputs = [n["id"] for n in run["graph"]["nodes"] if n["type"] in allowed[stage] and n["status"] == "accepted" and n.get("scope") == "case:" + self.state["group_id"]]
                if not inputs:
                    entry.update(status="no_op", reason="No accepted inputs for this stage")
                    index += 1
                    continue
            if stage == "propose_rule":
                inputs = entry["inputs"]
            if stage in {"validate_rule", "embed_rules"}:
                inputs = [n["id"] for n in run["graph"]["nodes"] if n["type"] == "rule" and n["status"] == "accepted" and n.get("created_operation_id") in completed]
                if not inputs:
                    entry.update(status="no_op", reason="No accepted rules from this traveler")
                    index += 1
                    continue
            actual = {"embed_rules": "embed", "project_rules": "project"}.get(stage, stage)
            command = Command(operation_id=identity("op"), graph_version=run["version"], config_id=config_id(run), type=actual, input_ids=inputs,
                              parameters={"scope": "case:" + self.state["group_id"], "task_id": "case:" + self.state["group_id"], "branch_id": "main", "source_trace_ids": self.turn_ids})
            value = {"operation_id": command.operation_id, "command": command.model_dump(), "job_id": self.job_id, "base_graph_version": run["version"], "config_id": config_id(run), "status": "running", "proposals": [], "diagnostics": {}, "usage": {}, "started_at": now()}
            grounded.save(self.run_id, value)
            entry.update(operation_id=command.operation_id, status="running")
            receipt_ids = {r["id"] for r in self.store.receipts(self.run_id)}
            try:
                proposals, diagnostics = grounded.propose(run, command, self.job_id)
                value.update(status="success" if proposals else "no_op", proposals=proposals, diagnostics=diagnostics)
            except Exception as error:
                value.update(status="failed", error=str(error))
                entry.update(status="failed", error=str(error))
                raise
            finally:
                calls = [r for r in self.store.receipts(self.run_id) if r["id"] not in receipt_ids]
                value.update(ended_at=now(), usage={"call_ids": [r["id"] for r in calls], "cost_rub": sum(r["data"].get("cost") or 0 for r in calls if r["data"].get("cost_unit") == "RUB")})
                grounded.save(self.run_id, value)
            if not proposals:
                entry["status"] = "no_op"
                index += 1
                continue
            self.save("memory.awaiting_researcher_review", status="paused", pending_operation_id=command.operation_id, waiting_for_step=True)
            with self.store.connection() as db:
                policy = {"mode": "paused", "permits": 0, "stop": False}
                db.execute("UPDATE benchmark_sessions SET control=? WHERE run_id=?", (encoded(policy), self.run_id))
            while True:
                reviewed = grounded.get(self.run_id, command.operation_id)
                from .benchmark import session
                state = session(self.store, self.run_id)
                if state["control"].get("stop"):
                    raise Stopped("Researcher stopped benchmark during memory review")
                if reviewed.get("commit") or reviewed["status"] == "cancelled":
                    entry["status"] = "committed" if reviewed.get("commit") else "cancelled"
                    completed.append(command.operation_id)
                    self.state.pop("pending_operation_id", None)
                    self.save("memory.review_completed", pending_operation_id=None)
                    break
                time.sleep(.3)
            index += 1
        self.save("memory.pipeline_completed", memory_pipeline={"traveler": self.turn, "index": index, "stages": stages})


def execution_class(body_or_state):
    mode = body_or_state.memory_mode if hasattr(body_or_state, "memory_mode") else body_or_state["config"]["memory_mode"]
    return GroundedExecution if mode in {"A", "B", "C", "D"} else Execution
