import asyncio
import csv
import io
import json
import os
from contextlib import asynccontextmanager
from typing import Literal
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .engine import Engine, health, impact, source_ids
from .fixtures import datasets
from .models import Action, Query, RunInput, BenchmarkInput
from .benchmark import Execution, case_detail, prepare as prepare_benchmark, session as benchmark_session, control as benchmark_control, resume_checkpoint
from .memoryarena import descriptors as memoryarena_descriptors, load as load_memoryarena
from .provider import status as provider_status, accounted_receipts
from .store import Conflict, Store, encoded, identity
from .snapshot import read_archive
from .traces import legacy_records


def create_app(database=None, models=None):
    store = Store(database or os.environ.get("MNEMOS_DB", str(Path(__file__).parents[1] / "state" / "mnemos.sqlite")))
    engine = Engine(store, models)
    pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix="mnemos")
    @asynccontextmanager
    async def lifespan(app):
        # Recovery belongs to a running server, never a module import/test factory.
        store.recover()
        app.state.pipeline.recover()
        yield
        pool.shutdown(wait=False, cancel_futures=True)
    app = FastAPI(title="Mnemos Prototype 01", version="0.1.0", lifespan=lifespan)
    app.state.store, app.state.engine = store, engine
    app.add_middleware(CORSMiddleware, allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"], allow_methods=["GET", "POST"], allow_headers=["Content-Type", "Last-Event-ID"])

    @app.middleware("http")
    async def local_mutations(request, call_next):
        # A random website must not drive this unauthenticated local workbench.
        origin = request.headers.get("origin")
        if request.method == "POST" and origin and origin not in {"http://127.0.0.1:5173", "http://localhost:5173", "http://127.0.0.1:8000", "http://localhost:8000"}:
            return Response("Foreign origin rejected", status_code=403)
        return await call_next(request)

    @app.exception_handler(Conflict)
    async def conflict(request, error):
        return Response(encoded({"detail": str(error)}), status_code=409, media_type="application/json")

    @app.exception_handler(ValueError)
    async def invalid(request, error):
        return Response(encoded({"detail": str(error)}), status_code=422, media_type="application/json")

    @app.exception_handler(KeyError)
    async def missing(request, error):
        return Response(encoded({"detail": str(error)}), status_code=404, media_type="application/json")

    with store.connection() as db:
        for dataset in [*datasets(), *memoryarena_descriptors()]:
            db.execute("INSERT OR IGNORE INTO datasets VALUES(?,?)", (dataset["id"], encoded(dataset)))

    @app.get("/api/datasets")
    def list_datasets():
        with store.connection() as db:
            records = [json.loads(row["data"]) for row in db.execute("SELECT data FROM datasets")]
        return [{**{k: v for k, v in d.items() if k != "records"}, "record_count": d.get("record_count", len(d["records"]))} for d in records]

    @app.get("/api/provider")
    def provider():
        return provider_status()

    @app.get("/api/memoryarena/cases/{group_id}")
    def inspect_case(group_id: str):
        return case_detail(group_id)

    @app.post("/api/memoryarena/executions", status_code=202)
    def execute_case(body: BenchmarkInput):
        run, job = prepare_benchmark(store, body)
        from .grounded_benchmark import execution_class
        pool.submit(execution_class(body)(store, engine, run["id"], job["id"]).run)
        return {"run": run, "job": job, "benchmark": benchmark_session(store, run["id"])}

    @app.get("/api/runs/{run_id}/benchmark")
    def benchmark(run_id: str):
        return benchmark_session(store, run_id)

    @app.post("/api/runs/{run_id}/trace-workspace", status_code=202)
    def rebuild_trace_workspace(run_id: str):
        original = store.run(run_id)
        if any(j["status"] in {"queued", "running"} for j in store.jobs(run_id)):
            raise Conflict("Wait for execution to finish before rebuilding its traces")
        records = legacy_records(original["graph"])
        if not records:
            raise ValueError("No observed execution traces to rebuild")
        run = store.create((original["name"][:65] + " · layered traces"), original["settings"])
        with store.connection() as db:
            store.event(db, run["id"], "workspace.rebuilt_from", {"original_run_id": run_id, "original_graph_version": original["version"], "policy": "Regroup original events into execution traces. Derive facts and concepts afresh; original graph, receipts and official result preserved. No planning repeated."})
        job, _ = store.reserve(run["id"], Action(action="import", parameters={"records": records}, expected_graph_version=0, idempotency_key="rebuild-traces"))
        pool.submit(engine.execute, run["id"], job["id"])
        return {"run": run, "job": job}

    class BenchmarkCommand(BaseModel):
        command: Literal["run", "pause", "step", "stop"]

    @app.post("/api/runs/{run_id}/benchmark/control")
    def control_benchmark(run_id: str, body: BenchmarkCommand):
        return benchmark_control(store, run_id, body.command)

    @app.post("/api/runs/{run_id}/benchmark/resume", status_code=202)
    def resume_benchmark(run_id: str):
        job = resume_checkpoint(store, run_id)
        from .grounded_benchmark import execution_class
        pool.submit(execution_class(benchmark_session(store, run_id))(store, engine, run_id, job["id"]).run)
        return {"job": job, "benchmark": benchmark_session(store, run_id)}

    @app.get("/api/runs/{run_id}/benchmark/scores.csv")
    def benchmark_csv(run_id: str):
        state = benchmark_session(store, run_id)
        if not state.get("scores"):
            raise Conflict("Official scores are available only after all travelers finish")
        output = io.StringIO(newline="")
        writer = csv.writer(output)
        writer.writerow(["run_id", "case", "memory_mode", "traveler_index", "name", "full_pass", "changed_slots", "changed_slots_passed", "SPS", "submission_version", "SR_group", "PS_group", "SPS_group"])
        metrics = state["scores"]["official_metrics"]
        for person in state["scores"]["per_group"][0]["traveler_metrics"]:
            prediction = next(p for p in state["predictions"] if p["traveler_index"] == person["traveler_index"])
            writer.writerow([run_id, state["group_id"], state["config"]["memory_mode"], person["traveler_index"], prediction["name"], person["full_pass"], person["changed_slots"], person["changed_slots_passed"], person["SPS"], prediction["submission_version"], metrics["SR"], metrics["PS"], metrics["SPS"]])
        return Response(output.getvalue(), media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="memoryarena-{state["group_id"]}-{run_id}-scores.csv"'})

    @app.get("/api/datasets/{dataset_id}")
    def dataset_detail(dataset_id: str):
        with store.connection() as db:
            row = db.execute("SELECT data FROM datasets WHERE id=?", (dataset_id,)).fetchone()
        if not row:
            raise KeyError("Dataset not found")
        dataset = json.loads(row["data"])
        if dataset.get("source") == "memoryarena":
            dataset["records"] = load_memoryarena(dataset)
        return {**dataset, "record_count": len(dataset["records"])}

    class DatasetInput(BaseModel):
        name: str = Field(min_length=1, max_length=100)
        records: list[dict] = Field(min_length=1, max_length=5000)

    @app.post("/api/datasets")
    def import_dataset(body: DatasetInput):
        if any(not isinstance(r.get("content"), str) or not r["content"].strip() for r in body.records):
            raise ValueError("Each record needs nonempty content")
        dataset = {"id": identity("dataset"), "name": body.name, "records": body.records, "description": "Imported by researcher"}
        with store.connection() as db:
            db.execute("INSERT INTO datasets VALUES(?,?)", (dataset["id"], encoded(dataset)))
        return dataset

    @app.get("/api/runs")
    def list_runs():
        return store.list_runs()

    @app.post("/api/runs")
    def create_run(body: RunInput):
        return store.create(body.name, body.settings.model_dump())

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str):
        return store.run(run_id)

    @app.get("/api/runs/{run_id}/graph")
    def graph_snapshot(run_id: str, version: int | None = None):
        run = store.run(run_id)
        if version is None or version == run["version"]:
            return run["graph"]
        for event in store.events(run_id):
            snapshot = event["data"].get("graph")
            if snapshot and snapshot["graph_version"] == version:
                return snapshot
        raise KeyError("Graph version not found")

    @app.post("/api/runs/{run_id}/actions", status_code=202)
    def act(run_id: str, body: Action):
        if store.run(run_id)["settings"].get("memory_design") == "entity_grounded_v1":
            raise Conflict("Entity-grounded runs use the versioned operations preview/commit API")
        if body.action == "benchmark":
            raise ValueError("Start a fresh benchmark through the MemoryArena case picker")
        job, fresh = store.reserve(run_id, body)
        if fresh:
            pool.submit(engine.execute, run_id, job["id"])
        return job

    @app.get("/api/runs/{run_id}/jobs/{job_id}")
    def job(run_id: str, job_id: str):
        return store.job(run_id, job_id)

    @app.get("/api/runs/{run_id}/jobs")
    def jobs(run_id: str):
        return store.jobs(run_id)

    @app.get("/api/runs/{run_id}/calls")
    def calls(run_id: str):
        store.run(run_id)
        return store.receipts(run_id)

    @app.post("/api/runs/{run_id}/jobs/{job_id}/cancel")
    def cancel(run_id: str, job_id: str):
        store.transition(run_id, job_id, "cancelled", "Cancellation requested. In-flight local inference may finish; its graph output will not commit.")
        return store.job(run_id, job_id)

    @app.get("/api/runs/{run_id}/events")
    async def events(run_id: str, request: Request, after: int = 0):
        store.run(run_id)
        header = request.headers.get("last-event-id", "0")
        if not header.isdigit():
            raise ValueError("Last-Event-ID must be an integer")
        async def stream():
            cursor = max(after, int(header))
            while not await request.is_disconnected():
                for event in store.events(run_id, cursor):
                    cursor = event["seq"]
                    yield f'id: {cursor}\nevent: {event["kind"]}\ndata: {encoded(event)}\n\n'
                yield ": heartbeat\n\n"
                await asyncio.sleep(.5)
        return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/api/runs/{run_id}/log")
    def log(run_id: str, include_snapshots: bool = False):
        store.run(run_id)
        events = store.events(run_id)
        if include_snapshots:
            return events
        return [{**event, "data": {**{k: v for k, v in event["data"].items() if k != "graph"}, **({"snapshot_ref": event["data"]["graph"]["graph_version"]} if "graph" in event["data"] else {})}} for event in events]

    @app.get("/api/runs/{run_id}/objects/{object_id}")
    def inspect(run_id: str, object_id: str, version: int | None = None):
        graph = graph_snapshot(run_id, version)
        obj = next((n for n in [*graph["nodes"], *graph["hyperedges"]] if n["id"] == object_id), None)
        if not obj:
            raise KeyError("Object not found")
        sources = source_ids(graph, object_id)
        if obj["type"] == "hyperedge":
            sources = set().union(*(source_ids(graph, n) for n in obj["member_ids"]))
        neighbors = []
        if "vector" in obj:
            import numpy as np
            neighbors = sorted([{"id": n["id"], "title": n["title"], "cosine": float(np.dot(obj["vector"], n["vector"]))} for n in graph["nodes"] if "vector" in n and n["id"] != object_id], key=lambda n: -n["cosine"])[:10]
        event = next((e for e in store.events(run_id) if e["id"] == obj["created_by_event"]), None)
        if event is None and obj.get("created_operation_id"):
            try:
                operation = app.state.grounded.get(run_id, obj["created_operation_id"])
                event = {"id": operation["operation_id"], "kind": "grounded.operation", "data": operation}
            except KeyError:
                pass
        return {"object": obj, "source_ids": sorted(sources), "sources": [n for n in graph["nodes"] if n["id"] in sources], "neighbors": neighbors, "impact": impact(graph, object_id), "generating_event": event, "model_calls": [r for r in store.receipts(run_id) if event and r["job_id"] == event["data"].get("job_id")]}

    @app.post("/api/runs/{run_id}/retrieve")
    def retrieve(run_id: str, body: Query):
        if store.run(run_id)["settings"].get("memory_design") == "entity_grounded_v1":
            raise Conflict("Use the entity-grounded context API; legacy retrieval is not a substitute")
        return engine.retrieve(run_id, body.query, body.variants, body.top_k)

    @app.post("/api/runs/{run_id}/evaluate")
    def evaluate(run_id: str):
        queries = store.run(run_id)["settings"]["held_out_queries"]
        return [engine.retrieve(run_id, q, ["A", "C", "E"], 5) for q in queries]

    @app.get("/api/runs/{run_id}/evaluations")
    def evaluations(run_id: str):
        store.run(run_id)
        with store.connection() as db:
            values = [json.loads(r["data"]) for r in db.execute("SELECT data FROM evaluations WHERE run_id=? ORDER BY rowid DESC", (run_id,))]
        return values

    class RetrievalReview(BaseModel):
        variant: Literal["A", "B", "C", "D", "E"]
        groundedness: Literal["supported", "unsupported", "unrated"]
        correctness: Literal["pass", "fail", "unrated"]
        reason: str = Field(min_length=1, max_length=4000)

    @app.post("/api/runs/{run_id}/evaluations/{evaluation_id}/review")
    def review_evaluation(run_id: str, evaluation_id: str, body: RetrievalReview):
        store.run(run_id)
        with store.connection() as db:
            row = db.execute("SELECT data FROM evaluations WHERE id=? AND run_id=?", (evaluation_id, run_id)).fetchone()
            if not row:
                raise KeyError("Evaluation not found")
            evaluation = json.loads(row["data"])
            result = next((r for r in evaluation["results"] if r["variant"] == body.variant), None)
            if not result:
                raise KeyError("Evaluation variant not found")
            result.update(groundedness_review=body.groundedness, answer_correctness=body.correctness, review_reason=body.reason, review_source="human_researcher")
            db.execute("UPDATE evaluations SET data=? WHERE id=?", (encoded(evaluation), evaluation_id))
            store.event(db, run_id, "evaluation.reviewed", {"evaluation_id": evaluation_id, **body.model_dump()})
        return evaluation

    @app.get("/api/runs/{run_id}/health")
    def metrics(run_id: str):
        result = health(store.run(run_id)["graph"])
        receipts = accounted_receipts(store.receipts(run_id))
        paid = [r["data"] for r in receipts if r["data"].get("cost_unit") == "RUB"]
        unknown = [r for r in paid if r.get("cost") is None]
        result["model_usage"] = {"calls": len(paid), "cache_hits": sum(bool(r["data"].get("cache_hit")) for r in receipts), "reported_cost_rub": sum(r.get("cost") or 0 for r in paid), "unknown_billing_calls": len(unknown), "input_tokens": sum(r.get("usage", {}).get("input_tokens") or 0 for r in paid), "output_tokens": sum(r.get("usage", {}).get("output_tokens") or 0 for r in paid)}
        return result

    class FrameSample(BaseModel):
        fps: float = Field(gt=0, le=10000)
        mean_ms: float = Field(gt=0, le=100000)
        p95_ms: float = Field(gt=0, le=100000)
        samples: int = Field(ge=1, le=100000)
        user_agent: str = Field(max_length=1000)

    @app.post("/api/runs/{run_id}/performance")
    def performance(run_id: str, body: FrameSample):
        run = store.run(run_id)
        data = {**body.model_dump(), "graph_version": run["version"], "node_count": len(run["graph"]["nodes"]), "note": "Observed requestAnimationFrame timing in this browser; not a benchmark guarantee."}
        with store.connection() as db:
            eid = store.event(db, run_id, "performance.measured", data)
        return {"event_id": eid, **data}

    @app.get("/api/runs/{run_id}/export")
    def export(run_id: str, format: str = "json"):
        run = store.run(run_id)
        with store.connection() as db:
            evaluations = [json.loads(r["data"]) for r in db.execute("SELECT data FROM evaluations WHERE run_id=?", (run_id,))]
        if format == "csv":
            output = io.StringIO(newline="")
            writer = csv.writer(output)
            writer.writerow(["evaluation_id", "graph_version", "query", "variant", "rank", "object_id", "cosine", "source_ids", "groundedness_review", "answer_correctness", "retrieval_characters", "latency_seconds"])
            for evaluation in evaluations:
                for result in evaluation["results"]:
                    for rank, hit in enumerate(result["hits"], 1):
                        writer.writerow([evaluation["id"], evaluation["graph_version"], evaluation["query"], result["variant"], rank, hit["id"], hit["score"], "|".join(hit["source_ids"]), result["groundedness_review"], result["answer_correctness"], result["retrieval_characters"], result["wall_seconds"]])
            return Response(output.getvalue(), media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="{run_id}.csv"'})
        if format != "json":
            raise ValueError("Export format must be json or csv")
        try:
            execution = benchmark_session(store, run_id)
        except KeyError:
            execution = None
        return Response(encoded({"schema_version": "mnemos-01/v1", "run": run, "benchmark": execution, "source_archive": read_archive(store, run_id), "events": store.events(run_id), "model_calls": store.receipts(run_id), "evaluations": evaluations, "health": health(run["graph"])}), media_type="application/json", headers={"Content-Disposition": f'attachment; filename="{run_id}.json"'})

    from .grounded_api import install as install_grounded
    install_grounded(app, store, engine.models, pool)
    build = Path(__file__).parents[1] / "frontend" / "dist"
    if build.exists():
        app.mount("/", StaticFiles(directory=build, html=True), name="workbench")
    return app


app = create_app()
