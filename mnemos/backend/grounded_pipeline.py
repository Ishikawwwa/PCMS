"""Durable, researcher-gated execution of the actual memory algorithms.

Each worker executes one stage. Paid calls are never replayed by a status read
or a server restart; continuing a failed stage requires an explicit retry.
"""
import json
import threading

from pydantic import BaseModel, ConfigDict, Field
from typing import Literal

from .grounded import ELIGIBLE, config_id
from .grounded_context import assemble
from .grounded_schema import Command, ContextQuery
from .models import Action
from .store import Conflict, encoded, identity, now


class PipelineStart(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scope: str = Field(min_length=1, max_length=300)
    variant: Literal["A", "B", "C", "D"] = "D"
    input_ids: list[str] = Field(default_factory=list, max_length=500)
    records: list[dict] = Field(default_factory=list, max_length=500)
    source_run_id: str | None = None
    task_id: str | None = None
    branch_id: str = "main"
    query: str = Field(default="", max_length=16000)
    token_budget: int = Field(default=4096, ge=16, le=32768)
    allow_hypotheses: bool = False
    trace_batch_size: int = Field(default=8, ge=1, le=16)


def memory_stages(variant):
    result = ["propose_episodes"]
    if variant != "A":
        result += ["resolve_entities", "propose_facts", "propose_task_state"]
    if variant in {"C", "D"}:
        result += ["propose_concepts", "embed", "project"]
    if variant == "C":
        result += ["propose_direct_rule"]
    if variant == "D":
        result += ["propose_hyperedges", "rules"]
    if variant in {"C", "D"}:
        result += ["validate_rule", "embed_rules", "project_rules"]
    return result


class PipelineRunner:
    def __init__(self, store, engine, pool, context):
        self.store, self.engine, self.pool, self.context = store, engine, pool, context
        self.lock = threading.RLock()
        with store.connection() as db:
            db.execute("CREATE TABLE IF NOT EXISTS grounded_pipelines(id TEXT PRIMARY KEY,run_id TEXT,data TEXT)")

    def recover(self):
        with self.store.connection() as db:
            values = [json.loads(r["data"]) for r in db.execute("SELECT data FROM grounded_pipelines")]
        for value in values:
            if value["status"] == "running":
                step = value["steps"][value["cursor"]]
                if step.get("operation_id"):
                    op = self.engine.get(value["run_id"], step["operation_id"])
                    if op["status"] in {"queued", "running"}:
                        calls = [r for r in self.store.receipts(value["run_id"]) if r["job_id"] == op["job_id"]]
                        op.update(status="failed", error="Server interrupted operation; inspect paid receipts before explicit retry", ended_at=now(),
                                  usage={"call_ids": [r["id"] for r in calls], "cost_rub": sum(r["data"].get("cost") or 0 for r in calls if r["data"].get("cost_unit") == "RUB"),
                                         "unknown_billing_calls": sum(r["data"].get("cost_unit") == "RUB" and r["data"].get("cost") is None for r in calls)})
                        self.engine.save(value["run_id"], op)
                self.sync(value)
                if value["status"] == "running":
                    value.update(status="failed", mode="paused", error="Server interrupted this stage. Inspect its receipts before explicitly retrying; no automatic replay")
                    value["steps"][value["cursor"]]["status"] = "failed"
                self.save(value)

    def save(self, value):
        value["updated_at"] = now()
        with self.store.connection() as db:
            db.execute("INSERT OR REPLACE INTO grounded_pipelines VALUES(?,?,?)", (value["id"], value["run_id"], encoded(value)))
            self.store.event(db, value["run_id"], "pipeline.updated", {"pipeline_id": value["id"], "status": value["status"], "mode": value["mode"],
                             "cursor": value["cursor"], "config_id": value["config_id"], "graph_version": value["expected_graph_version"],
                             "step": value["steps"][value["cursor"]] if value["cursor"] < len(value["steps"]) else None})

    def get(self, run_id, pipeline_id=None):
        with self.store.connection() as db:
            row = db.execute("SELECT data FROM grounded_pipelines WHERE run_id=? " + ("AND id=?" if pipeline_id else "ORDER BY rowid DESC LIMIT 1"),
                             (run_id, pipeline_id) if pipeline_id else (run_id,)).fetchone()
        return json.loads(row["data"]) if row else None

    def start(self, run_id, body):
        with self.lock:
            previous = self.get(run_id)
            if previous and previous["status"] not in {"completed", "cancelled"}:
                raise Conflict("An unfinished pipeline exists; continue or cancel it first")
            run = self.store.run(run_id)
            if run["settings"].get("memory_design") != "entity_grounded_v1":
                raise Conflict("Pipeline requires an entity-grounded workspace")
            with self.store.connection() as db:
                if db.execute("SELECT 1 FROM jobs WHERE run_id=? AND status IN ('queued','running')", (run_id,)).fetchone():
                    raise Conflict("Finish or stop the active workspace job before creating a standalone pipeline")
            nodes = {n["id"]: n for n in run["graph"]["nodes"]}
            if body.input_ids and any(i not in nodes or nodes[i]["type"] != "trace" or nodes[i].get("scope") != body.scope or nodes[i]["status"] not in ELIGIBLE for i in body.input_ids):
                raise ValueError("Choose accepted traces in the pipeline scope")
            if body.records and body.source_run_id:
                raise ValueError("Choose one import source")
            if any(record.get("scope", record.get("session_id", record.get("session", "imported"))) != body.scope for record in body.records):
                raise ValueError("Imported records must use the pipeline scope")
            stages = ["import_traces" if body.records or body.source_run_id else "inspect_traces", *memory_stages(body.variant)]
            if body.query.strip():
                stages += ["context.resolve", "context.traverse", "context.select", "context.pack"]
            value = {"id": identity("pipeline"), "run_id": run_id, "request": body.model_dump(), "config_id": config_id(run),
                     "created_at": now(), "as_of": now(), "expected_graph_version": run["version"], "cursor": 0,
                     "status": "ready", "mode": "paused", "steps": [{"type": s, "status": "pending"} for s in stages]}
            self.save(value)
            return value

    def sync(self, value):
        if value["cursor"] >= len(value["steps"]):
            value["status"] = "completed"
            return
        step = value["steps"][value["cursor"]]
        if step.get("operation_id"):
            operation = self.engine.get(value["run_id"], step["operation_id"])
            step.update(usage=operation.get("usage", {}), diagnostics=operation.get("diagnostics", {}), ended_at=operation.get("ended_at"))
            if operation.get("commit"):
                step.update(status="committed", decision=operation["commit"], output_ids=operation["commit"]["accepted_ids"],
                            graph_version=operation.get("committed_graph_version", self.store.run(value["run_id"])["version"]))
                # Engine records the exact committed version, never adopt an unrelated write.
                value["expected_graph_version"] = operation["base_graph_version"] + 1
                value["cursor"] += 1
                value["status"] = "ready"
            elif operation["status"] in {"success", "no_op"}:
                value["status"] = "awaiting_review"
                step["status"] = "awaiting_review"
            elif operation["status"] in {"failed", "cancelled"}:
                step.update(status="failed", error=operation.get("error", "Proposal cancelled; explicitly retry or skip"))
                value.update(status="failed", error=step["error"])
        if value["cursor"] == len(value["steps"]):
            value["status"] = "completed"

    def control(self, run_id, command):
        with self.lock:
            value = self.get(run_id)
            if not value:
                raise KeyError("No pipeline")
            if command == "cancel":
                value.update(status="cancelled", mode="paused")
                if value["cursor"] < len(value["steps"]):
                    step = value["steps"][value["cursor"]]
                    if step.get("operation_id"):
                        op = self.engine.get(run_id, step["operation_id"])
                        if not op.get("commit"):
                            self.store.transition(run_id, op["job_id"], "cancelled", "Pipeline cancelled; billing retained")
                            op.update(status="cancelled", proposals=[])
                            self.engine.save(run_id, op)
                self.save(value)
                return value
            if value["status"] in {"cancelled", "completed"}:
                raise Conflict("Pipeline is finished; create another pass explicitly")
            self.sync(value)
            if command == "pause":
                value["mode"] = "paused"
            elif command in {"step", "run", "retry", "skip"}:
                if command == "skip":
                    if value["status"] != "failed":
                        raise Conflict("Only a failed step can be skipped; review proposals explicitly")
                    step = value["steps"][value["cursor"]]
                    step.update(status="skipped", reason="Explicit researcher skip after failure")
                    value["cursor"] += 1
                    value.update(status="ready", error=None)
                elif command == "retry":
                    if value["status"] != "failed":
                        raise Conflict("Only failed steps can be retried")
                    step = value["steps"][value["cursor"]]
                    value["steps"][value["cursor"]] = {**{k: step[k] for k in ("type", "input_ids", "source_trace_ids", "batch", "batches") if k in step}, "status": "pending",
                                                        "previous_attempts": [*step.get("previous_attempts", []), step]}
                    value.update(status="ready", error=None)
                value["mode"] = "run" if command == "run" else "paused"
            else:
                raise ValueError("Unknown pipeline control")
            self.save(value)
            if command != "pause" and value["status"] == "ready":
                self.launch(value)
            return self.get(run_id)

    def reviewed(self, run_id):
        with self.lock:
            value = self.get(run_id)
            if not value or value["status"] in {"cancelled", "completed"}:
                return
            self.sync(value)
            self.save(value)
            if value["mode"] == "run" and value["status"] == "ready":
                self.launch(value)

    def launch(self, value):
        if value["cursor"] == len(value["steps"]):
            value["status"] = "completed"
            self.save(value)
            return
        value["status"] = "running"
        value["steps"][value["cursor"]]["status"] = "running"
        self.save(value)
        self.pool.submit(self.execute, value["run_id"], value["id"])

    def execute(self, run_id, pipeline_id):
        try:
            value = self.get(run_id, pipeline_id)
            if value["status"] == "cancelled":
                return
            step = value["steps"][value["cursor"]]
            run = self.store.run(run_id)
            if run["version"] != value["expected_graph_version"] or config_id(run) != value["config_id"]:
                raise Conflict("Graph/config changed outside this pipeline. Cancel and create a new pass; paid operations are not replayed")
            request, graph = value["request"], run["graph"]
            stage = step["type"]
            scoped = [n for n in graph["nodes"] if n.get("scope") == request["scope"] and n["status"] in ELIGIBLE]
            batch = [s.get("operation_id") for s in value["steps"][:value["cursor"]]]
            traces = [n["id"] for n in scoped if n["type"] == "trace" and (not request["input_ids"] or n["id"] in request["input_ids"])]
            extraction = stage in {"propose_episodes", "resolve_entities", "propose_facts", "propose_task_state", "propose_concepts", "propose_direct_rule"}
            batch_size = request.get("trace_batch_size", 8)
            if extraction and "source_trace_ids" not in step and len(traces) > batch_size:
                groups = [traces[i:i+batch_size] for i in range(0, len(traces), batch_size)]
                with self.lock:
                    latest = self.get(run_id, pipeline_id)
                    if latest["status"] == "cancelled":
                        return
                    value["mode"] = latest["mode"]
                    value["steps"][value["cursor"]:value["cursor"]+1] = [{"type": stage, "source_trace_ids": group, "batch": index+1, "batches": len(groups), "status": "pending"} for index, group in enumerate(groups)]
                    self.save(value)
                    self.launch(value)
                return
            traces = step.get("source_trace_ids", traces)
            step["started_at"] = now()
            if stage == "inspect_traces":
                if not traces:
                    raise ValueError("No accepted traces in this scope. Import evidence before extraction")
                step.update(status="success", input_ids=traces, result={"trace_ids": traces, "event_count": sum(len(n.get("events", [])) for n in scoped if n["id"] in traces), "hashes": {n["id"]: n.get("content_hash") for n in scoped if n["id"] in traces}})
            elif stage.startswith("context."):
                query = ContextQuery(query=request["query"], scope=request["scope"], task_id=request["task_id"], branch_id=request["branch_id"],
                                     variant=request["variant"], token_budget=request["token_budget"], as_of=value["as_of"], allow_hypotheses=request["allow_hypotheses"])
                result = self.context(run, query) if stage == "context.pack" else assemble(run, query, None, {}, stop_after=stage.split(".")[1])
                step.update(status="success", result=result, graph_version=run["version"])
            elif stage == "rules":
                edges = [h for h in graph["hyperedges"] if h["status"] in ELIGIBLE and h.get("created_operation_id") in batch]
                if edges:
                    value["steps"][value["cursor"]:value["cursor"]+1] = [{"type": "propose_rule", "input_ids": [h["id"]], "status": "pending"} for h in edges]
                    value["status"] = "ready"
                    self.save(value)
                    # Expansion is scheduling, not an inference operation.
                    self.launch(value)
                    return
                step.update(status="no_op", result={"reason": "No accepted hyperedge from this pass; no rule invented"})
            else:
                actual = {"embed_rules": "embed", "project_rules": "project"}.get(stage, stage)
                by_type = {"propose_episodes": {"trace"}, "resolve_entities": {"episode"}, "propose_facts": {"episode", "entity"},
                           "propose_task_state": {"episode", "entity", "fact"}, "propose_concepts": {"episode", "fact", "state"},
                           "embed": {"concept"}, "project": {"concept", "rule"}, "propose_hyperedges": {"concept", "rule"},
                           "propose_direct_rule": {"episode", "fact", "state"}, "validate_rule": {"rule"}, "embed_rules": {"rule"}, "project_rules": {"concept", "rule"}}
                inputs = step.get("input_ids") or [n["id"] for n in scoped if n["type"] in by_type.get(stage, set()) and
                          (stage not in {"validate_rule", "embed_rules"} or n.get("created_operation_id") in batch) and
                          (not extraction or n["type"] == "entity" or any(span["trace_id"] in traces for span in n.get("provenance", [])))]
                if stage == "propose_episodes":
                    inputs = traces
                parameters = {"scope": request["scope"], "task_id": request["task_id"], "branch_id": request["branch_id"], "source_trace_ids": traces}
                if stage == "import_traces":
                    parameters.update(records=request["records"], source_run_id=request["source_run_id"])
                if stage not in {"import_traces"} and not inputs:
                    step.update(status="no_op", result={"reason": "No accepted inputs for " + stage + "; upstream proposals may have abstained or been rejected"})
                else:
                    command = Command(operation_id=identity("op"), graph_version=run["version"], config_id=value["config_id"], type=actual, input_ids=inputs, parameters=parameters)
                    with self.lock:
                        latest = self.get(run_id, pipeline_id)
                        if latest["status"] == "cancelled":
                            return
                        value["mode"] = latest["mode"]
                        job, _ = self.store.reserve(run_id, Action(action="attribute", parameters={"pipeline_id": pipeline_id, "grounded_preview": command.model_dump()},
                                                  expected_graph_version=run["version"], idempotency_key="pipeline:" + command.operation_id))
                        self.engine.save(run_id, {"operation_id": command.operation_id, "command": command.model_dump(), "job_id": job["id"], "base_graph_version": run["version"], "status": "queued", "proposals": []})
                        step.update(operation_id=command.operation_id, input_ids=inputs)
                        self.save(value)
                    from .snapshot import archive_source
                    archive_source(self.store, run_id, suffix=command.operation_id)
                    self.engine.preview(run_id, command, job["id"])
                    with self.lock:
                        current = self.get(run_id, pipeline_id)
                        if current["status"] == "cancelled":
                            return
                        self.sync(current)
                        self.save(current)
                    return
            step["ended_at"] = now()
            value["cursor"] += 1
            value["status"] = "completed" if value["cursor"] == len(value["steps"]) else "ready"
            with self.lock:
                if self.get(run_id, pipeline_id)["status"] == "cancelled":
                    return
                # Honour pause received while the stage was executing.
                value["mode"] = self.get(run_id, pipeline_id)["mode"]
                self.save(value)
                if value["mode"] == "run" and value["status"] == "ready":
                    self.launch(value)
        except Exception as error:
            with self.lock:
                value = self.get(run_id, pipeline_id)
                if value["status"] != "cancelled":
                    step = value["steps"][value["cursor"]]
                    if step.get("operation_id"):
                        op = self.engine.get(run_id, step["operation_id"])
                        if op["status"] in {"queued", "running"}:
                            self.store.transition(run_id, op["job_id"], "failed", str(error))
                            op.update(status="failed", error=str(error))
                            self.engine.save(run_id, op)
                    value.update(status="failed", mode="paused", error=str(error))
                    value["steps"][value["cursor"]].update(status="failed", error=str(error), ended_at=now())
                    self.save(value)
