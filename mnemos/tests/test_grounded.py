"""Scientific contracts: actual accepted objects and exact messages, not UI labels."""
import copy
import json
from types import SimpleNamespace

import pytest

from mnemos.backend.grounded import GroundedEngine, config_id, discover_grounded, source_spans
from mnemos.backend.grounded_api import dispatch_context
from mnemos.backend.grounded_context import assemble
from mnemos.backend.grounded_schema import Command, Commit, ContextQuery
from mnemos.backend.models import Settings, Action
from mnemos.backend.store import Store, Conflict, identity


class FakeModels:
    test_only = True
    output = {"records": [], "abstention": "test abstention"}
    passed = True

    def grounded_generate(self, kind, data, settings, receipt, schema=None):
        receipt({"operation": kind, "cost": 0, "prompt_version": "test-only", "messages": data})
        if kind == "grounded_validation":
            return {"decisions": [{"id": n["id"], "passed": self.passed, "reason": "Test validator", "novelty": True, "unsupported": [], "contradictions": []} for n in data["candidates"]]}
        return copy.deepcopy(self.output)


@pytest.fixture
def lab(tmp_path):
    store = Store(tmp_path / "grounded.sqlite")
    run = store.create("Synthetic regression only", Settings(mode="demo").model_dump())
    models = FakeModels()
    return store, GroundedEngine(store, models), models, run["id"]


def preview(lab, kind, parameters=None, inputs=()):
    store, engine, models, rid = lab
    run = store.run(rid)
    command = Command(operation_id=identity("op"), graph_version=run["version"], config_id=config_id(run), type=kind, input_ids=list(inputs), parameters=parameters or {})
    job, _ = store.reserve(rid, Action(action="attribute", parameters={"test": command.model_dump()}, expected_graph_version=run["version"], idempotency_key=command.operation_id))
    engine.preview(rid, command, job["id"])
    return engine.get(rid, command.operation_id)


def commit(lab, value):
    store, engine, _, rid = lab
    return engine.commit(rid, value["operation_id"], Commit(graph_version=store.run(rid)["version"], accepted_ids=[p["id"] for p in value["proposals"]], reviewer="test human", reason="Synthetic regression only"))


def imported(lab):
    value = preview(lab, "import_traces", {"records": [{"session": "S1", "scope": "trip-1", "content": "Eric requires step-free transport. Emma budget is unknown.", "events": [{"id": "ev1", "kind": "request", "content": "Eric requires step-free transport. Emma budget is unknown."}]}]})
    assert value["status"] == "success"
    commit(lab, value)
    trace = lab[0].run(lab[3])["graph"]["nodes"][0]
    return trace, {"trace_id": trace["id"], "event_id": "ev1", "start": 0, "end": 33, "quote": trace["events"][0]["content"][:33]}


def test_preview_commit_idempotency_and_exact_spans(lab):
    t, span = imported(lab)
    graph = lab[0].run(lab[3])["graph"]
    source_spans(graph, [span])
    with pytest.raises(ValueError):
        source_spans(graph, [{**span, "quote": "invented"}])
    duplicate = preview(lab, "import_traces", {"records": [{"session": "S1", "scope": "trip-1", "content": t["content"], "events": [{"id": "ev1", "kind": "request", "content": t["content"]}]}]})
    assert duplicate["status"] == "no_op"
    assert duplicate["diagnostics"]["duplicates"] == [0]
    before = lab[0].run(lab[3])["version"]
    result = commit(lab, duplicate)
    assert lab[1].commit(lab[3], duplicate["operation_id"], Commit(**result["commit"])) == result
    assert lab[0].run(lab[3])["version"] == before + 1


def test_unsupported_output_cannot_commit_and_stale_writes_fail(lab):
    t, span = imported(lab)
    lab[2].output = {"records": [{"title": "Eric", "entity_type": "person", "canonical_name": "Eric", "aliases": [], "scope": "trip-1", "identity_candidates": [], "resolution": "resolved", "confidence": .9, "provenance": [span]}], "abstention": None}
    lab[2].passed = False
    value = preview(lab, "resolve_entities")
    assert value["status"] == "success"
    with pytest.raises(ValueError, match="Unsupported"):
        commit(lab, value)
    assert len(lab[0].run(lab[3])["graph"]["nodes"]) == 1
    lab[2].passed = True
    value = preview(lab, "resolve_entities")
    another = preview(lab, "import_traces", {"records": [{"content": "Other evidence", "scope": "other"}]})
    commit(lab, another)
    with pytest.raises(Conflict):
        commit(lab, value)


def test_abstention_and_ambiguous_identity_are_not_facts(lab):
    t, span = imported(lab)
    empty = preview(lab, "resolve_entities")
    assert empty["status"] == "no_op"
    record = {"title": "Emma", "entity_type": "person", "canonical_name": "Emma", "aliases": [], "scope": "trip-1", "identity_candidates": ["uncertain"], "resolution": "ambiguous", "confidence": .5, "provenance": [span]}
    lab[2].output = {"records": [record], "abstention": None}
    value = preview(lab, "resolve_entities")
    commit(lab, value)
    entity = lab[0].run(lab[3])["graph"]["nodes"][-1]
    assert entity["status"] == "ambiguous"
    decision = preview(lab, "decide_identity", {"same_as": None, "reason": "Reviewed original event; distinct scoped Emma"}, [entity["id"]])
    commit(lab, decision)
    assert lab[0].run(lab[3])["graph"]["nodes"][-1]["resolution"] == "resolved"


def test_edit_creates_reviewed_revision_without_overwriting_content(lab):
    t, span = imported(lab)
    lab[2].output = {'records': [{'title': 'Original episode', 'task': None, 'goal': None, 'summary': 'Eric requests transport', 'events': ['Request observed'], 'decisions': [], 'outcome': 'unknown', 'uncertainty': [], 'scope': 'trip-1', 'provenance': [span]}], 'abstention': None}
    first = preview(lab, 'propose_episodes', inputs=[t['id']])
    assert first['status'] == 'success', first
    commit(lab, first)
    original = copy.deepcopy(lab[0].run(lab[3])['graph']['nodes'][-1])
    revised = preview(lab, 'revise_proposal', {'changes': {'title': 'Reviewed episode title'}}, [original['id']])
    assert revised['status'] == 'success', revised
    commit(lab, revised)
    graph = lab[0].run(lab[3])['graph']
    old = next(n for n in graph['nodes'] if n['id'] == original['id'])
    assert old['title'] == original['title'] and old['status'] == 'retired'
    new = graph['nodes'][-1]
    assert new['id'] != old['id'] and new['supersedes_id'] == old['id']
    assert any(r['type'] == 'SUPERSEDES' and r['source'] == new['id'] for r in graph['links'])


class Tokenizer:
    def encode(self, text):
        return SimpleNamespace(ids=list(text.encode()))  # Exact bytes, declared test tokenizer.


def node(id, type, scope="trip-1", **extra):
    return {"id": id, "type": type, "title": id, "status": "accepted", "scope": scope, "level": 1, **extra}


def scenario():
    ns = [node("eric", "entity", canonical_name="Eric", aliases=[], resolution="resolved", entity_type="person"),
          node("emma", "entity", canonical_name="Emma", aliases=[], resolution="resolved", entity_type="person"),
          node("trip", "entity", canonical_name="Trip", aliases=[], resolution="resolved", entity_type="trip"),
          node("other-emma", "entity", "trip-2", canonical_name="Emma", aliases=[], resolution="resolved", entity_type="person"),
          node("access", "fact", subject_id="eric", predicate="transport_accessibility", value="step-free", value_type="string", units=None, target_id=None, modality="required"),
          node("transfers", "fact", subject_id="eric", predicate="transfers", value="few", value_type="string", units=None, target_id=None, modality="preferred"),
          node("budget", "fact", subject_id="emma", predicate="budget", value=None, value_type="unknown", units=None, target_id=None, modality="required"),
          node("other-budget", "fact", "trip-2", subject_id="other-emma", predicate="budget", value=999, value_type="number", modality="required", units="USD"),
          node("pending", "state", task_id="revise", branch_id="main", state_type="pending", execution_status="pending", content="Replacement route pending", entity_ids=["trip"])]
    links = []
    def link(s, t, kind):
        links.append({"id": f"{s}:{t}", "source": s, "target": t, "type": kind, "scope": "trip-1", "status": "accepted"})
    link("eric", "trip", "PARTICIPATES_IN")
    link("emma", "trip", "PARTICIPATES_IN")
    link("eric", "access", "HAS_FACT")
    link("eric", "transfers", "HAS_FACT")
    link("emma", "budget", "HAS_FACT")
    link("trip", "pending", "HAS_STATE")
    return {"id": "test", "version": 1, "settings": Settings(mode="demo").model_dump(), "graph": {"nodes": ns, "links": links, "hyperedges": [], "projection": None}}


def query(run, **values):
    return assemble(run, ContextQuery(query="Revise Eric's transport", token_budget=20000, **values), Tokenizer(), {"model": "test-only-byte-tokenizer"})


def test_indirect_participants_modality_unknown_values_and_state():
    result = query(scenario())
    ids = {v["id"] for v in result["included"]}
    assert {"access", "budget", "pending", "transfers"} <= ids
    assert "other-budget" not in ids
    assert result["scope"] == "trip-1"
    assert "unknown" in result["serialized_messages"]
    assert "preferred" in result["serialized_messages"] and "required" in result["serialized_messages"]
    assert "999" not in result["serialized_messages"]
    assert result["missing_slots"][0]["id"] == "budget"
    assert len(result["paths"]["budget"]) == 3
    assert "budget" not in {c["id"] for c in query(scenario(),max_hops=1)["included"]}


def test_same_name_across_scopes_requires_clarification():
    run = scenario()
    result = assemble(run, ContextQuery(query="Emma's budget", token_budget=20000), Tokenizer(), {})
    assert result["status"] == "clarification_required"
    assert result["included"] == []
    assert result["ambiguities"] == ["emma"]


def test_temporal_conflict_branch_and_insufficient_budget():
    run = scenario()
    first = next(n for n in run["graph"]["nodes"] if n["id"] == "budget")
    first.update(value=40, value_type="number", units="USD", valid_to="2026-02-01T00:00:00Z")
    run["graph"]["nodes"].append({**first, "id": "budget-new", "value": 60, "valid_to": None, "valid_from": "2026-02-01T00:00:00Z"})
    run["graph"]["links"].append({"id":"new-budget-link","source":"emma","target":"budget-new","type":"HAS_FACT","scope":"trip-1","status":"accepted"})
    past = query(run, as_of="2026-01-01T00:00:00Z")
    assert "budget-new" not in {c["id"] for c in past["included"]}
    present = query(run, as_of="2026-03-01T00:00:00Z")
    assert "budget" not in {c["id"] for c in present["included"]}
    run["graph"]["nodes"].append({**run["graph"]["nodes"][-1], "id": "budget-conflict", "value": 70})
    run["graph"]["links"].append({"id":"conflict-budget-link","source":"emma","target":"budget-conflict","type":"HAS_FACT","scope":"trip-1","status":"accepted"})
    assert query(run, as_of="2026-03-01T00:00:00Z")["conflicts"]
    tiny = assemble(run, ContextQuery(query="Eric transport", scope="trip-1", token_budget=32), Tokenizer(), {})
    assert tiny["status"] == "insufficient_context_budget" and tiny["omitted_requirements"]
    assert "pending" not in {c["id"] for c in query(run, branch_id="alternate")["included"]}


def test_prompt_fidelity_no_agent_and_hash_guard():
    run = scenario()
    preview = query(run)
    class Agent:
        def complete(self, messages, **kwargs):
            self.messages = messages
            return {"actual": "response"}
    agent = Agent()
    with pytest.raises(ValueError, match="not configured"):
        dispatch_context(run, preview, preview["messages_sha256"], agent)
    run["settings"]["agent_enabled"] = True
    preview = query(run)
    dispatch_context(run, preview, preview["messages_sha256"], agent)
    assert agent.messages == preview["messages"]
    assert json.loads(preview["serialized_messages"]) == agent.messages
    with pytest.raises(Conflict):
        dispatch_context(run, preview, "wrong hash", agent)


def test_hypotheses_and_missing_inputs_excluded():
    run = scenario()
    rule = node("rule", "rule", instruction="Check all participant constraints", hypothesis_status="hypothesis", origin_method="hypergraph", applicability=["same_scope"], task_types=["revise"], contradicting_ids=[], required_inputs=[{"name": "budget", "object_type": "fact", "predicate": "budget", "scope": "same_scope"}])
    run["graph"]["nodes"].append(rule)
    result = query(run)
    assert not result["rule_checks"][0]["eligible"]
    assert "rule" not in {c["id"] for c in result["included"]}
    budget = next(n for n in run["graph"]["nodes"] if n["id"] == "budget")
    budget.update(value=50, value_type="number", units="USD")
    ready = query(run, allow_hypotheses=True)
    assert ready["rule_checks"][0]["eligible"]
    assert "rule" in {c["id"] for c in ready["included"]}


def embedding_graph(vectors):
    ns, es = [], []
    for i, vector in enumerate(vectors):
        ns.append(node(str(i), "concept", level=2, embedding_ref="e" + str(i)))
        es.append({"id": "e" + str(i), "model": "real-test-fixture", "revision": "pinned", "dimension": len(vector), "status": "accepted", "vector": vector})
    return {"nodes": ns, "links": [], "hyperedges": [], "embeddings": es}


def test_all_pairs_no_chain_and_overlapping_triplets():
    import math
    graph = embedding_graph([[1, 0], [math.cos(.6), math.sin(.6)], [math.cos(1.2), math.sin(1.2)]])
    groups, why = discover_grounded(graph, Settings(threshold=.7, min_pairwise=.7))
    assert groups == [] and why["near_misses"]
    graph = embedding_graph([[1, .1], [1, .2], [1, .3], [1, .4]])
    groups, why = discover_grounded(graph, Settings(threshold=.7, min_pairwise=.7, max_size=3))
    assert len(groups) == 4
    assert all(len(g["member_ids"]) == 3 and len(g["pairwise_similarity"]) == 3 for g in groups)
    graph["embeddings"][0]["revision"] = "other"
    with pytest.raises(ValueError, match="Incompatible"):
        discover_grounded(graph, Settings())


def test_source_invalidation_closure_is_atomic(lab):
    t, span = imported(lab)
    store, engine, models, rid = lab
    run = store.run(rid)
    graph = run["graph"]
    graph["nodes"].append(node("f", "fact", provenance=[span]))
    graph["nodes"].append(node("r", "rule"))
    graph["links"] += [{"id": "f:t", "type": "DERIVED_FROM", "source": "f", "target": t["id"]}, {"id": "r:f", "type": "DERIVED_FROM", "source": "r", "target": "f"}]
    from mnemos.backend.store import encoded
    with store.connection() as db:
        db.execute("UPDATE runs SET graph=? WHERE id=?", (encoded(graph), rid))
    value = preview(lab, "challenge_source", {"reason": "contradictory original evidence"}, [t["id"]])
    with pytest.raises(ValueError, match="closure"):
        engine.commit(rid, value["operation_id"], Commit(graph_version=1, accepted_ids=[value["proposals"][0]["id"]], reviewer="test", reason="test"))
    commit(lab, value)
    statuses = {n["id"]: n["status"] for n in store.run(rid)["graph"]["nodes"]}
    assert statuses[t["id"]] == "disputed" and statuses["f"] == statuses["r"] == "needs_revalidation"


def test_api_preview_commit_legacy_guard_and_export_roundtrip(tmp_path):
    import time
    from fastapi.testclient import TestClient
    from mnemos.backend.app import create_app
    app = create_app(tmp_path / "api.sqlite", FakeModels())
    with TestClient(app) as client:
        run = client.post('/api/runs', json={'name': 'Explicit synthetic API regression', 'settings': {'mode': 'demo'}}).json()
        base = '/api/runs/' + run['id']
        cfg = client.get(base + '/grounded/config').json()
        command = {'operation_id': 'import-once', 'type': 'import_traces', 'graph_version': 0, 'config_id': cfg['config_id'], 'input_ids': [], 'parameters': {'records': [{'content': 'Exact source\nwith newline', 'scope': 'test'}]}}
        first = client.post(base + '/operations', json=command)
        assert first.status_code == 202, first.text
        for _ in range(50):
            op = client.get(base + '/operations/import-once').json()
            if op['status'] not in {'queued', 'running'}:
                break
            time.sleep(.02)
        assert op['status'] == 'success'
        assert client.get(base + '/graph').json()['nodes'] == []
        assert client.post(base + '/operations', json=command).json()['job']['id'] == first.json()['job']['id']
        decision = {'graph_version': 0, 'accepted_ids': [p['id'] for p in op['proposals']], 'reviewer': 'test human', 'reason': 'reviewed exact original'}
        assert client.post(base + '/operations/import-once/commit', json=decision).status_code == 200
        assert client.post(base + '/actions', json={'action': 'import', 'parameters': {}, 'targets': [], 'expected_graph_version': 1, 'idempotency_key': 'bypass'}).status_code == 409
        exported = client.get(base + '/grounded/export?include_sources=true').json()
        restored = client.post('/api/grounded/restore', json=exported)
        assert restored.status_code == 200, restored.text
        assert restored.json()['graph'] == exported['run']['graph']
        redacted = client.get(base + '/grounded/export').json()
        assert 'Exact source' not in json.dumps(redacted)
        assert client.post('/api/grounded/restore', json=redacted).status_code == 422


def test_memoryarena_review_pauses_before_memory_commit(tmp_path):
    import threading
    import time
    from mnemos.backend.benchmark import prepare, session, control
    from mnemos.backend.grounded_benchmark import GroundedExecution
    from mnemos.backend.engine import Engine
    from mnemos.backend.models import BenchmarkInput
    store = Store(tmp_path / 'benchmark.sqlite')
    fake = FakeModels()
    run, job = prepare(store, BenchmarkInput(group_id='17', memory_mode='A', settings=Settings()))
    store.transition(run['id'], job['id'], 'running')
    runner = GroundedExecution(store, Engine(store, fake), run['id'], job['id'])
    runner.turn = 1
    runner.state['current_traveler'] = 1
    runner.observe('request', 'Eric requires step-free transport.', 'request:1')
    trace = store.run(run['id'])['graph']['nodes'][0]
    span = {'trace_id': trace['id'], 'event_id': 'event:0', 'start': 0, 'end': len(trace['content']), 'quote': trace['content']}
    fake.output = {'records': [{'title': 'Transport request episode', 'task': None, 'goal': 'Arrange transport', 'summary': 'Eric requests accessible transport', 'events': ['Eric states requirement'], 'decisions': [], 'outcome': 'unknown', 'uncertainty': ['No observed outcome'], 'scope': 'case:17', 'provenance': [span]}], 'abstention': None}
    errors = []
    def worker():
        try:
            runner.consolidate()
        except Exception as e:
            errors.append(e)
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    try:
        for _ in range(100):
            state = session(store, run['id'])
            if state.get('pending_operation_id'):
                break
            time.sleep(.03)
        assert state.get('pending_operation_id'), errors
        assert not any(n['type'] == 'episode' for n in store.run(run['id'])['graph']['nodes'])
        grounded = GroundedEngine(store, fake)
        proposal = grounded.get(run['id'], state['pending_operation_id'])
        assert proposal['status'] == 'success'
        grounded.commit(run['id'], proposal['operation_id'], Commit(graph_version=store.run(run['id'])['version'], accepted_ids=[p['id'] for p in proposal['proposals']], reviewer='test human', reason='Reviewed full original source'))
        thread.join(3)
        assert not thread.is_alive() and not errors
        assert any(n['type'] == 'episode' and n['status'] == 'accepted' for n in store.run(run['id'])['graph']['nodes'])
        assert store.job(run['id'], job['id'])['status'] == 'running'
    finally:
        if thread.is_alive():
            control(store, run['id'], 'stop')
            thread.join(2)
