"""Synthetic fixtures exercise real orchestration, graph mutations and retrieval.

These tests do not represent model quality or benchmark results.
"""
import copy
from types import SimpleNamespace

import pytest

from mnemos.backend.grounded import GroundedEngine
from mnemos.backend.grounded_context import assemble
from mnemos.backend.grounded_pipeline import PipelineRunner, PipelineStart, memory_stages
from mnemos.backend.grounded_schema import Commit
from mnemos.backend.models import Settings
from mnemos.backend.store import Store, Conflict


class ImmediatePool:
    def submit(self, fn, *args):
        fn(*args)


class FixtureModels:
    test_only = True

    def __init__(self):
        self.calls = []

    def embed(self, texts, settings):
        return [[1., .01 * i, .01] for i in range(len(texts))], "synthetic-test-only"

    def grounded_generate(self, kind, data, settings, receipt, schema=None):
        self.calls.append(kind)
        receipt({"operation": kind, "cost": 0, "synthetic_test_only": True})
        if kind == "grounded_validation":
            return {"decisions": [{"id": n["id"], "passed": True, "reason": "Synthetic fixture validator", "novelty": True, "unsupported": [], "contradictions": []} for n in data["candidates"]]}
        base = {"title": "Synthetic fixture", "scope": data["scope"], "provenance": [data["allowed_provenance_spans"][0]]}
        entities = data["entities"]
        selected = data["selected_objects"]
        if kind == "propose_episodes":
            records = [{**base, "task": "transport", "goal": "Accessible transport", "summary": "Eric requires step-free transport", "events": ["Request observed"], "decisions": [], "outcome": "unknown", "uncertainty": []}]
        elif kind == "resolve_entities":
            records = [{**base, "title": "Eric", "entity_type": "person", "canonical_name": "Eric", "aliases": [], "identity_candidates": [], "resolution": "resolved", "confidence": 1.}]
        elif kind == "propose_facts":
            records = [{**base, "subject_id": entities[0]["id"], "predicate": "accessibility", "value": "step-free", "value_type": "string", "target_id": None, "units": None, "modality": "required", "valid_from": None, "valid_to": None, "assertion_time": None, "confidence": 1.}]
        elif kind == "propose_task_state":
            records = [{**base, "task_id": data.get("task_id") or "trip", "branch_id": "main", "state_type": "goal", "execution_status": "pending", "content": "Arrange accessible transport", "dependencies": [], "entity_ids": [entities[0]["id"]], "valid_from": None, "valid_to": None, "uncertainty": []}]
        elif kind == "propose_concepts":
            records = [{**base, "title": f"Synthetic concept {i}", "proposition": f"Synthetic scoped transport principle {i}", "applicability": ["same_scope"], "exceptions": [], "supporting_ids": [n["id"] for n in selected], "contradicting_ids": []} for i in range(3)]
        elif kind in {"propose_rule", "propose_direct_rule"}:
            records = [{**base, "instruction": "Check known accessibility before choosing transport", "applicability": ["same_scope"], "target_entity_types": ["person"], "task_types": ["transport"], "required_inputs": [{"name": "access", "object_type": "fact", "predicate": "accessibility", "scope": "same_scope", "entity_type": None}], "exceptions": ["Unknown constraints require clarification"], "supporting_ids": [n["id"] for n in data.get("members", selected)], "contradicting_ids": []}]
        else:
            raise AssertionError(kind)
        return {"records": records, "abstention": None}


def make_lab(tmp_path):
    store = Store(tmp_path / "pipeline.sqlite")
    run = store.create("Synthetic pipeline regression", Settings(mode="demo").model_dump())
    models = FixtureModels()
    engine = GroundedEngine(store, models)
    tokenizer = SimpleNamespace(encode=lambda text: SimpleNamespace(ids=text.split()))
    runner = PipelineRunner(store, engine, ImmediatePool(), lambda run, query: assemble(run, query, tokenizer, {"model": "synthetic-tokenizer"}))
    return store, engine, runner, models, run["id"]


def review(lab):
    store, engine, runner, models, rid = lab
    value = runner.get(rid)
    step = value["steps"][value["cursor"]]
    op = engine.get(rid, step["operation_id"])
    before = store.run(rid)["version"]
    engine.commit(rid, op["operation_id"], Commit(graph_version=before, accepted_ids=[p["id"] for p in op["proposals"]], reviewer="Fixture human", reason="Synthetic regression review"))
    runner.reviewed(rid)
    assert store.run(rid)["version"] >= before + 1


def test_complete_d_pipeline_review_boundaries_and_exact_context(tmp_path):
    lab = make_lab(tmp_path)
    store, engine, runner, models, rid = lab
    runner.start(rid, PipelineStart(scope="trip", task_id="trip", query="Arrange transport for Eric", allow_hypotheses=True,
                 records=[{"scope": "trip", "content": "Eric requires step-free transport."}]))
    assert store.run(rid)["version"] == 0
    runner.control(rid, "step")
    assert runner.get(rid)["status"] == "awaiting_review"
    assert not store.run(rid)["graph"]["nodes"]
    review(lab)
    assert runner.get(rid)["status"] == "ready"
    assert models.calls == []  # Step executed import only, not extraction.
    runner.control(rid, "run")
    for _ in range(30):
        value = runner.get(rid)
        if value["status"] == "completed":
            break
        assert value["status"] == "awaiting_review", value
        before = copy.deepcopy(store.run(rid)["graph"])
        calls = len(models.calls)
        runner.control(rid, "run")
        assert len(models.calls) == calls
        assert store.run(rid)["graph"] == before
        review(lab)
    value = runner.get(rid)
    assert value["status"] == "completed", value
    graph = store.run(rid)["graph"]
    assert {n["type"] for n in graph["nodes"]} == {"trace", "episode", "entity", "fact", "state", "concept", "rule"}
    assert len(graph["hyperedges"]) == 1
    assert len(graph["embeddings"]) == 4
    assert graph["projection"]
    steps = {s["type"]: s for s in value["steps"]}
    assert steps["validate_rule"]["status"] == "committed"
    assert steps["context.resolve"]["result"]["entity_resolutions"][0]["name"] == "Eric"
    assert steps["context.traverse"]["result"]["paths"]
    assert steps["context.select"]["result"]["rule_checks"][0]["eligible"]
    context = steps["context.pack"]["result"]
    assert context["status"] == "success"
    assert "step-free" in context["serialized_messages"]
    assert context["dispatch_status"] == "preview_only"
    assert context["outcome_score"] is None
    # State persists across controller recreation; status reads do not invoke models.
    calls = len(models.calls)
    again = PipelineRunner(store, engine, ImmediatePool(), None)
    assert again.get(rid) == runner.get(rid)
    assert len(models.calls) == calls


def test_failure_does_not_replay_and_explicit_skip_is_audited(tmp_path):
    lab = make_lab(tmp_path)
    store, engine, runner, models, rid = lab
    runner.start(rid, PipelineStart(scope="empty"))
    runner.control(rid, "step")
    assert runner.get(rid)["status"] == "failed"
    assert models.calls == []
    runner.recover()
    assert runner.get(rid)["status"] == "failed"
    runner.control(rid, "skip")
    value = runner.get(rid)
    assert value["steps"][0]["status"] == "skipped"
    assert value["steps"][1]["status"] == "no_op"
    runner.control(rid, "cancel")
    assert runner.get(rid)["status"] == "cancelled"


def test_stale_graph_rejected_and_b_variant_has_no_semantic_induction(tmp_path):
    lab = make_lab(tmp_path)
    store, engine, runner, models, rid = lab
    runner.start(rid, PipelineStart(scope="trip", records=[{"scope": "trip", "content": "Eric requires step-free transport."}]))
    with pytest.raises(Conflict):
        runner.start(rid, PipelineStart(scope="trip"))
    with store.connection() as db:
        db.execute("UPDATE runs SET version=1 WHERE id=?", (rid,))
    runner.control(rid, "run")
    assert runner.get(rid)["status"] == "failed"
    assert "changed outside" in runner.get(rid)["error"]
    assert models.calls == []
    assert memory_stages("B") == ["propose_episodes", "resolve_entities", "propose_facts", "propose_task_state"]


def test_extraction_batches_cover_every_trace_without_repeating_paid_calls(tmp_path):
    lab = make_lab(tmp_path)
    store, engine, runner, models, rid = lab
    runner.start(rid, PipelineStart(scope="trip", variant="A", trace_batch_size=1,
                 records=[{"scope": "trip", "content": f"Eric requires step-free transport. Event {i}."} for i in range(3)]))
    runner.control(rid, "run")
    review(lab)  # Import, then pause at first extraction proposal.
    for _ in range(3):
        value = runner.get(rid)
        assert value["status"] == "awaiting_review"
        step = value["steps"][value["cursor"]]
        op = engine.get(rid, step["operation_id"])
        assert len(op["command"]["parameters"]["source_trace_ids"]) == 1
        assert len(op["command"]["input_ids"]) == 1
        review(lab)
    value = runner.get(rid)
    assert value["status"] == "completed"
    batches = [s for s in value["steps"] if s["type"] == "propose_episodes"]
    covered = [trace for s in batches for trace in s["source_trace_ids"]]
    assert len(covered) == len(set(covered)) == 3
    assert models.calls.count("propose_episodes") == 3


def test_restart_preserves_completed_preview_and_never_auto_replays(tmp_path):
    lab = make_lab(tmp_path)
    store, engine, runner, models, rid = lab
    runner.start(rid, PipelineStart(scope="trip", records=[{"scope": "trip", "content": "Eric requires step-free transport."}]))
    runner.control(rid, "step")
    # Simulate interruption after the preview was saved but before coordinator sync.
    value = runner.get(rid)
    value["status"] = "running"
    runner.save(value)
    recovered = PipelineRunner(store, engine, ImmediatePool(), None)
    recovered.recover()
    assert recovered.get(rid)["status"] == "awaiting_review"
    assert models.calls == []
    # Simulate interruption of a read stage before a worker executed it.
    recovered.control(rid, "cancel")
    recovered.start(rid, PipelineStart(scope="trip"))
    value = recovered.get(rid)
    value["status"] = "running"
    value["steps"][0]["status"] = "running"
    recovered.save(value)
    recovered.recover()
    assert recovered.get(rid)["status"] == "failed"
    assert models.calls == []


def test_memoryarena_d_consolidation_runs_every_stage_with_review(tmp_path):
    import threading
    import time
    from mnemos.backend.benchmark import prepare, session, control
    from mnemos.backend.grounded_benchmark import GroundedExecution
    from mnemos.backend.engine import Engine
    from mnemos.backend.models import BenchmarkInput

    store = Store(tmp_path / "benchmark-d.sqlite")
    models = FixtureModels()
    run, job = prepare(store, BenchmarkInput(group_id="17", memory_mode="D", settings=Settings()))
    rid = run["id"]
    store.transition(rid, job["id"], "running")
    execution = GroundedExecution(store, Engine(store, models), rid, job["id"])
    execution.turn = 1
    execution.observe("request", "Eric requires step-free transport.", "request:1")
    errors, reviewed = [], set()

    def work():
        try:
            execution.consolidate()
        except Exception as error:
            errors.append(error)

    thread = threading.Thread(target=work, daemon=True)
    thread.start()
    grounded = GroundedEngine(store, models)
    try:
        for _ in range(1000):
            if not thread.is_alive():
                break
            state = session(store, rid)
            pending = state.get("pending_operation_id")
            if pending and pending not in reviewed:
                op = grounded.get(rid, pending)
                assert op["status"] == "success"
                grounded.commit(rid, pending, Commit(graph_version=store.run(rid)["version"], accepted_ids=[p["id"] for p in op["proposals"]], reviewer="Synthetic test human", reason="Regression fixture only"))
                reviewed.add(pending)
                control(store, rid, "run")
            time.sleep(.02)
        assert not thread.is_alive(), session(store, rid)
        assert not errors, errors
        state = session(store, rid)
        assert state["phase"] == "memory.pipeline_completed"
        stages = state["memory_pipeline"]["stages"]
        assert all(s["status"] == "committed" for s in stages)
        assert {s["type"] for s in stages} >= {"propose_rule", "validate_rule", "embed_rules", "project_rules"}
        graph = store.run(rid)["graph"]
        assert len(graph["hyperedges"]) == 1 and len(graph["embeddings"]) == 4
        assert len(reviewed) == len(stages)
    finally:
        if thread.is_alive():
            control(store, rid, "stop")
            thread.join(2)
