"""Versioned workbench endpoints; archived legacy experiments are preserved."""
import copy
import hashlib
import json
from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel, Field

from .grounded import GroundedEngine, config_id, DESIGN
from .grounded_context import assemble
from .grounded_schema import Command, Commit, ContextQuery, SCHEMAS
from .models import Action
from .provider import client as provider_client, safe_error
from .store import Conflict, encoded, identity, now


def dispatch_context(run, preview, messages_hash, client):
    if preview["graph_version"] != run["version"] or preview["config_id"] != config_id(run):
        raise Conflict("Context is stale; preview again")
    if preview["status"] != "success":
        raise Conflict("Context has unresolved ambiguity or insufficient budget")
    raw = encoded(preview["messages"])
    if hashlib.sha256(raw.encode()).hexdigest() != messages_hash or preview["messages_sha256"] != messages_hash:
        raise Conflict("Messages differ from reviewed preview")
    if not run["settings"].get("agent_enabled"):
        raise ValueError("Agent adapter not configured; preview only")
    return client.complete(copy.deepcopy(preview["messages"]), operation="mnemos_grounded_agent", json_output=False)


def install(app, store, models, pool):
    engine = GroundedEngine(store, models)
    app.state.grounded = engine
    router = APIRouter(prefix="/api")

    def ensure(run_id):
        run = store.run(run_id)
        if run["settings"].get("memory_design") != "entity_grounded_v1":
            raise Conflict("Legacy experiment is preserved. Create an entity-grounded run and import its traces explicitly")
        return run

    @router.get("/grounded/schema")
    def schema():
        return {"version": DESIGN, "commands": Command.model_json_schema(), "commit": Commit.model_json_schema(), "context": ContextQuery.model_json_schema(),
                "objects": {k: v.model_json_schema() for k, v in SCHEMAS.items()}, "legacy_results": "preserved; not reclassified as entity-grounded evidence"}

    @router.get("/runs/{run_id}/operations")
    def operations(run_id: str):
        store.run(run_id)
        with store.connection() as db:
            return [json.loads(r["data"]) for r in db.execute("SELECT data FROM grounded_operations WHERE run_id=? ORDER BY rowid DESC", (run_id,))]

    @router.get("/runs/{run_id}/grounded/config")
    def configuration(run_id: str):
        run = ensure(run_id)
        return {"config_id": config_id(run), "settings": run["settings"], "schema": DESIGN,
                "agent": "configured" if run["settings"].get("agent_enabled") else "not configured; context preview only",
                "tokenizer": "pinned Qwen3.6-27B" if run["settings"]["llm_model"] == "qwen/qwen3.6-27b" else "not configured"}

    @router.get("/runs/{run_id}/operations/{operation_id}")
    def operation(run_id: str, operation_id: str):
        return engine.get(run_id, operation_id)

    @router.get("/runs/{run_id}/operations/{operation_id}/calls")
    def operation_calls(run_id: str, operation_id: str):
        op = engine.get(run_id, operation_id)
        ids = set(op.get("usage", {}).get("call_ids", []))
        return [r for r in store.receipts(run_id) if r["id"] in ids]

    @router.post("/runs/{run_id}/operations", status_code=202)
    def propose(run_id: str, body: Command):
        run = ensure(run_id)
        try:
            old = engine.get(run_id, body.operation_id)
        except KeyError:
            old = None
        if old:
            if old["command"] != body.model_dump():
                raise Conflict("Operation ID belongs to different inputs")
            return {"operation": old, "job": store.job(run_id, old["job_id"])}
        if body.config_id != config_id(run):
            raise Conflict("Configuration ID mismatch")
        action = Action(action="attribute", parameters={"grounded_preview": body.model_dump()}, expected_graph_version=body.graph_version, idempotency_key="grounded-preview:" + body.operation_id)
        job, fresh = store.reserve(run_id, action)
        initial = {"operation_id": body.operation_id, "command": body.model_dump(), "job_id": job["id"], "base_graph_version": run["version"], "status": "queued", "proposals": []}
        engine.save(run_id, initial)
        if fresh:
            from .snapshot import archive_source
            try:
                archive_source(store, run_id, suffix=hashlib.sha256(body.operation_id.encode()).hexdigest()[:16])
            except Exception as error:
                store.transition(run_id, job["id"], "failed", str(error))
                raise
            pool.submit(engine.preview, run_id, body, job["id"])
        return {"operation": initial, "job": job}

    @router.post("/runs/{run_id}/operations/{operation_id}/commit")
    def commit(run_id: str, operation_id: str, body: Commit):
        ensure(run_id)
        result = engine.commit(run_id, operation_id, body)
        pipeline.reviewed(run_id)
        return result

    @router.post("/runs/{run_id}/operations/{operation_id}/cancel")
    def cancel(run_id: str, operation_id: str):
        value = engine.get(run_id, operation_id)
        if value.get("commit"):
            raise Conflict("Committed operation cannot be cancelled; branch or challenge evidence")
        if value.get("job_id"):
            if store.job(run_id, value["job_id"])["request"]["action"] != "benchmark":
                store.transition(run_id, value["job_id"], "cancelled", "Researcher cancelled preview; any in-flight billing retained")
        value.update(status="cancelled", proposals=[])
        engine.save(run_id, value)
        return value

    @router.post("/runs/{run_id}/branch")
    def branch(run_id: str, body: dict):
        original = store.run(run_id)
        version = body.get("graph_version", original["version"])
        graph = original["graph"]
        if version != original["version"]:
            graph = next((e["data"]["graph"] for e in store.events(run_id) if e["data"].get("graph_version") == version), None)
            if graph is None:
                raise KeyError("Historical graph version unavailable")
        from .models import Settings
        settings = Settings(**{**original["settings"], **body.get("settings", {})}).model_dump()
        run = store.create(body.get("name", original["name"][:70] + " · branch"), settings)
        with store.connection() as db:
            snapshot = copy.deepcopy(graph)
            snapshot["graph_version"] = 0
            db.execute("UPDATE runs SET graph=? WHERE id=?", (encoded(snapshot), run["id"]))
            store.event(db, run["id"], "branch.created", {"parent_run_id": run_id, "parent_graph_version": version, "graph": snapshot})
        return store.run(run["id"])

    def tokenize(run):
        from src.constraint_memory.config import verify_tokenizer
        from tokenizers import Tokenizer
        if run["settings"]["llm_model"] != "qwen/qwen3.6-27b":
            raise ValueError("Tokenizer not configured for this model; context assembly disabled")
        path, manifest = verify_tokenizer(Path(__file__).resolve().parents[2])
        return Tokenizer.from_file(str(path)), manifest

    def context(run, body):
        tokenizer, manifest = tokenize(run)
        result = assemble(run, body, tokenizer, manifest)
        with store.connection() as db:
            db.execute("INSERT INTO grounded_retrievals VALUES(?,?,?)", (result["id"], run["id"], encoded(result)))
            store.event(db, run["id"], "context.previewed", {"retrieval_id": result["id"], "graph_version": run["version"], "messages_sha256": result["messages_sha256"], "status": result["status"]})
        return result

    @router.post("/runs/{run_id}/context")
    def preview_context(run_id: str, body: ContextQuery):
        return context(ensure(run_id), body)

    @router.post("/runs/{run_id}/context/compare")
    def compare(run_id: str, body: ContextQuery):
        run = ensure(run_id)
        outputs = [context(run, body.model_copy(update={"variant": variant})) for variant in "ABCD"]
        return {"id": identity("comparison"), "run_id": run_id, "graph_version": run["version"], "config_id": config_id(run), "results": outputs,
                "outcome_metrics": None, "evaluation_status": "prompt comparison only; no benchmark execution or outcome scores", "shared_budget": body.token_budget}

    @router.get("/runs/{run_id}/contexts")
    def contexts(run_id: str):
        store.run(run_id)
        with store.connection() as db:
            return [json.loads(r["data"]) for r in db.execute("SELECT data FROM grounded_retrievals WHERE run_id=? ORDER BY rowid DESC", (run_id,))]

    class Dispatch(BaseModel):
        messages_sha256: str

    @router.post("/runs/{run_id}/contexts/{retrieval_id}/dispatch")
    def dispatch(run_id: str, retrieval_id: str, body: Dispatch):
        run = ensure(run_id)
        if run["settings"].get("agent_enabled") and len([r for r in store.receipts(run_id) if r["data"].get("cost_unit") == "RUB"]) >= run["settings"]["max_model_calls"]:
            raise ValueError("Run model-call budget exhausted")
        with store.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT data FROM grounded_retrievals WHERE id=? AND run_id=?", (retrieval_id, run_id)).fetchone()
            if not row:
                raise KeyError("Context preview not found")
            value = json.loads(row["data"])
            if value["dispatch_status"] != "preview_only":
                raise Conflict("This preview was already dispatched or attempted; explicit new preview required")
            # Perform all checks before claiming dispatch.
            if not run["settings"].get("agent_enabled") or value["status"] != "success":
                raise ValueError("Agent not configured or context unresolved")
            if value["graph_version"] != run["version"] or body.messages_sha256 != value["messages_sha256"]:
                raise Conflict("Context or hash changed")
            value.update(dispatch_status="in_flight", dispatched_at=now())
            db.execute("UPDATE grounded_retrievals SET data=? WHERE id=?", (encoded(value), retrieval_id))
        from .models import Settings
        connection = provider_client(Settings(**run["settings"]))
        try:
            value["agent_response"] = dispatch_context(run, value, body.messages_sha256, connection)
            value["dispatch_status"] = "completed"
        except Exception as error:
            value.update(dispatch_status="failed", agent_error=safe_error(error))
        finally:
            value["provider_accounting"] = connection.records
            for receipt in connection.records:
                store.receipt(run_id, retrieval_id, {"operation": "agent", "messages": value["messages"], "messages_sha256": body.messages_sha256, "cost": receipt.get("cost_rub"), "cost_unit": "RUB", "provider_accounting": [receipt]})
            connection.close()
            with store.connection() as db:
                db.execute("UPDATE grounded_retrievals SET data=? WHERE id=?", (encoded(value), retrieval_id))
                store.event(db, run_id, "agent.dispatched", {"retrieval_id": retrieval_id, "status": value["dispatch_status"], "messages_sha256": body.messages_sha256})
        return value

    @router.get("/runs/{run_id}/grounded/export")
    def export(run_id: str, include_sources: bool = False):
        run = copy.deepcopy(store.run(run_id))
        with store.connection() as db:
            ops = [json.loads(r["data"]) for r in db.execute("SELECT data FROM grounded_operations WHERE run_id=?", (run_id,))]
            queries = [json.loads(r["data"]) for r in db.execute("SELECT data FROM grounded_retrievals WHERE run_id=?", (run_id,))]
            pipelines = [json.loads(r["data"]) for r in db.execute("SELECT data FROM grounded_pipelines WHERE run_id=?", (run_id,))]
        result = {"schema": DESIGN, "run": run, "operations": ops, "contexts": queries, "pipelines": pipelines, "events": store.events(run_id), "model_calls": store.receipts(run_id), "source_text_included": include_sources, "omissions": []}
        if not include_sources:
            # Raw model inputs, snapshots and previews also contain source text.
            result["events"] = [{"id": e["id"], "seq": e["seq"], "kind": e["kind"], "created_at": e["created_at"], "data_sha256": hashlib.sha256(encoded(e["data"]).encode()).hexdigest()} for e in result["events"]]
            result["model_calls"] = [{"id": c["id"], "job_id": c["job_id"], "data_sha256": hashlib.sha256(encoded(c["data"]).encode()).hexdigest(), "cost": c["data"].get("cost")} for c in result["model_calls"]]
            for n in run["graph"]["nodes"]:
                if n["type"] == "trace":
                    n.pop("content", None)
                    n["title"] = "Source omitted"
                    n["events"] = [{k: v for k, v in e.items() if k != "content"} for e in n.get("events", [])]
                for span in n.get("provenance", []):
                    span.pop("quote", None)
            result["operations"] = [{"operation_id": o["operation_id"], "status": o["status"], "sha256": hashlib.sha256(encoded(o).encode()).hexdigest()} for o in ops]
            result["contexts"] = [{"id": q["id"], "status": q["status"], "messages_sha256": q["messages_sha256"]} for q in queries]
            result["pipelines"] = [{"id": p["id"], "status": p["status"], "sha256": hashlib.sha256(encoded(p).encode()).hexdigest()} for p in pipelines]
            result["omissions"] = ["source text, model payloads, prompt bytes, proposal bodies and replay snapshots omitted; this redacted manifest cannot fully replay"]
        result["manifest_sha256"] = hashlib.sha256(encoded(result).encode()).hexdigest()
        return result

    @router.post("/grounded/restore")
    def restore(body: dict):
        expected = body.get("manifest_sha256")
        payload = {k: v for k, v in body.items() if k != "manifest_sha256"}
        if expected != hashlib.sha256(encoded(payload).encode()).hexdigest():
            raise ValueError("Archive checksum mismatch")
        if not payload.get("source_text_included") or payload.get("omissions"):
            raise ValueError("Redacted archive cannot restore evidence/replay; use a full source-inclusive export")
        from .models import Settings
        from .grounded import source_spans
        settings = Settings(**payload["run"]["settings"]).model_dump()
        graph = payload["run"]["graph"]
        ids = [n["id"] for n in graph["nodes"]] + [h["id"] for h in graph["hyperedges"]]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate archive object IDs")
        for n in graph["nodes"]:
            if n["type"] in {"fact", "state", "concept", "rule", "episode"}:
                source_spans(graph, n.get("provenance", []))
        for link in graph["links"]:
            if link["source"] not in ids or link["target"] not in ids:
                raise ValueError("Dangling archive relation")
        ancestry = {}
        for link in graph["links"]:
            if link["type"] == "DERIVED_FROM":
                ancestry.setdefault(link["source"], []).append(link["target"])
        visiting, visited = set(), set()
        def acyclic(nid):
            if nid in visiting:
                raise ValueError("Circular provenance in archive")
            if nid in visited:
                return
            visiting.add(nid)
            for parent in ancestry.get(nid, []):
                acyclic(parent)
            visiting.remove(nid)
            visited.add(nid)
        for nid in ids:
            acyclic(nid)
        from .grounded import compatible_embeddings
        compatible_embeddings(graph, [n for n in graph["nodes"] if n.get("embedding_ref")])
        for q in payload["contexts"]:
            if hashlib.sha256(encoded(q["messages"]).encode()).hexdigest() != q["messages_sha256"]:
                raise ValueError("Archived prompt checksum mismatch")
        run = store.create(payload["run"]["name"][:80] + " · restored", settings)
        with store.connection() as db:
            db.execute("UPDATE runs SET graph=?,version=? WHERE id=?", (encoded(graph), graph["graph_version"], run["id"]))
            for event in payload["events"]:
                if event["kind"] == "graph.patch":
                    store.event(db, run["id"], "graph.patch", event["data"])
            for op in payload["operations"]:
                db.execute("INSERT INTO grounded_operations VALUES(?,?,?)", (op["operation_id"], run["id"], encoded(op)))
            for plan in payload.get("pipelines", []):
                plan = {**plan, "id": identity("pipeline"), "run_id": run["id"], "status": "cancelled", "mode": "paused", "restored_from": plan["id"], "replay_only": True}
                db.execute("INSERT INTO grounded_pipelines VALUES(?,?,?)", (plan["id"], run["id"], encoded(plan)))
            for q in payload["contexts"]:
                q = {**q, "id": identity("retrieval"), "run_id": run["id"], "dispatch_status": "archived_replay_only"}
                db.execute("INSERT INTO grounded_retrievals VALUES(?,?,?)", (q["id"], run["id"], encoded(q)))
            store.event(db, run["id"], "archive.restored", {"manifest_sha256": expected, "original_run_id": payload["run"]["id"]})
        return store.run(run["id"])

    from .grounded_pipeline import PipelineRunner, PipelineStart
    pipeline = PipelineRunner(store, engine, pool, context)
    app.state.pipeline = pipeline

    @router.get("/runs/{run_id}/pipeline")
    def get_pipeline(run_id: str):
        ensure(run_id)
        return pipeline.get(run_id)

    @router.post("/runs/{run_id}/pipeline")
    def start_pipeline(run_id: str, body: PipelineStart):
        ensure(run_id)
        return pipeline.start(run_id, body)

    @router.post("/runs/{run_id}/pipeline/control")
    def control_pipeline(run_id: str, body: dict):
        ensure(run_id)
        return pipeline.control(run_id, body.get("command"))

    app.include_router(router)
    return engine
