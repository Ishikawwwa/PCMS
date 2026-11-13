import copy
import json
import sqlite3

import numpy as np
import pytest
from fastapi.testclient import TestClient

from mnemos.backend.adapters import Models
from mnemos.backend.algorithms import discover, projection
from mnemos.backend.app import create_app
from mnemos.backend.engine import Engine, impact, source_ids, attributed_fact
from mnemos.backend.fixtures import datasets
from mnemos.backend.models import Action, Settings
from mnemos.backend.store import Conflict, Store, identity


@pytest.fixture
def lab(tmp_path):
    store = Store(tmp_path / "lab.sqlite")
    run = store.create("test", Settings(mode="demo",memory_design="legacy").model_dump())
    return store, Engine(store), run["id"]


def act(lab, action, targets=None, **parameters):
    store, engine, rid = lab
    request = Action(action=action, targets=targets or [], parameters=parameters, expected_graph_version=store.run(rid)["version"], idempotency_key=identity("test"))
    job, fresh = store.reserve(rid, request)
    engine.execute(rid, job["id"])
    return store.job(rid, job["id"])


def graph(lab):
    return lab[0].run(lab[2])["graph"]


def good(lab, action, targets=None, **parameters):
    job = act(lab, action, targets, **parameters)
    assert job["status"] == "succeeded", job
    return job


def approve(lab):
    ids = [n["id"] for n in graph(lab)["nodes"] if n["status"] in {"candidate", "needs_review"}]
    good(lab, "review", ids, decision="accept", reason="Test reviewer checked original sources; demo acceptance only")


def concepts(lab):
    good(lab, "import", records=datasets()[0]["records"])
    good(lab, "episode", count=3)
    approve(lab)
    good(lab, "concept")
    approve(lab)


def test_facts_are_individual_episodes_and_concepts_cite_the_fact_layer(lab):
    concepts(lab)
    g = graph(lab)
    episodes = [n for n in g["nodes"] if n["type"] == "episode"]
    assert len(episodes) == len(datasets()[0]["records"])
    assert all(n["episode_schema"] == "mnemos/granular-fact-v1" and n["episode"]["fact"] in n["content"] and n["evidence_scope"] for n in episodes)
    for concept in [n for n in g["nodes"] if n["type"] == "concept"]:
        parents = [link["target"] for link in g["links"] if link["source"] == concept["id"] and link["type"] == "DERIVED_FROM"]
        assert len(parents) >= 2
        assert set(parents) == set(concept["episode_ids"])
        assert all(next(n for n in g["nodes"] if n["id"] == pid)["type"] == "episode" for pid in parents)
    before = len(episodes)
    assert act(lab, "episode", count=20)["status"] == "failed"  # no unused source batch
    assert sum(n["type"] == "episode" for n in graph(lab)["nodes"]) == before


def test_agent_claims_and_tool_results_keep_their_distinct_evidence_scope():
    fact = {"fact":"Hotel costs $120 per night", "evidence_quotes":[{"source_id":"trace-1","quote":"$120 per night"}]}
    for kind, expected in [("agent_output", "Agent-reported claim"), ("tool_call", "Attempted tool action"), ("tool_result", "Observed tool result")]:
        source = {"id":"trace-1", "source_metadata":{"events":[{"kind":kind,"content":"Hotel costs $120 per night"}]}}
        fields = attributed_fact(fact, [source])
        assert fields["content"].startswith(expected)
        assert fields["evidence_roles"] == [kind]


def test_full_recursive_pipeline_provenance_challenge_retrieval(lab):
    concepts(lab)
    good(lab, "embed")
    assert graph(lab)["projection"] is None  # no silent refits
    good(lab, "project")
    projected = copy.deepcopy(graph(lab)["projection"])
    good(lab, "associate", settings={"threshold": 0, "min_pairwise": 0})
    edge = graph(lab)["hyperedges"][0]
    assert edge["status"] == "candidate"
    good(lab, "crystallize", [edge["id"]])
    meta = next(n for n in graph(lab)["nodes"] if n["type"] == "meta_concept")
    assert meta["status"] == "needs_review"
    assert source_ids(graph(lab), meta["id"])
    approve(lab)
    good(lab, "embed", [meta["id"]])
    assert graph(lab)["projection"] == projected
    good(lab, "associate", settings={"threshold": 0, "min_pairwise": 0})
    assert any(meta["id"] in h["member_ids"] for h in graph(lab)["hyperedges"])
    result = lab[1].retrieve(lab[2], "capacity overload concurrent requests", ["A", "C", "E"], 5)
    assert [r["variant"] for r in result["results"]] == ["A", "C", "E"]
    assert all(r["hits"] for r in result["results"])
    hit = result["results"][1]["hits"][0]
    node = next(n for n in graph(lab)["nodes"] if n["id"] == hit["id"])
    q, _ = lab[1].models.embed([result["query"]], Settings(mode="demo",memory_design="legacy"))
    assert hit["score"] == pytest.approx(np.dot(q[0], node["vector"]))
    source = next(iter(source_ids(graph(lab), meta["id"])))
    before = {n["id"]: n.get("content") for n in graph(lab)["nodes"]}
    affected = impact(graph(lab), source)
    good(lab, "challenge", [source], reason="Source observation disputed")
    after = graph(lab)
    assert {n["id"]: n.get("content") for n in after["nodes"]} == before
    assert next(n for n in after["nodes"] if n["id"] == meta["id"])["status"] == "needs_revalidation"
    assert all(h["status"] == "stale" for h in after["hyperedges"] if h["id"] in affected["hyperedge_ids"])
    good(lab, "revalidate", [meta["id"]])
    assert next(n for n in graph(lab)["nodes"] if n["id"] == meta["id"])["status"] == "needs_revalidation"


def test_atomic_import_and_exact_deduplication(lab):
    good(lab, "import", records=[{"content": "one"}, {"content": "one"}])
    assert len(graph(lab)["nodes"]) == 1
    before = graph(lab)
    job = act(lab, "import", records=[{"content": "two"}, {"oops": "invalid"}])
    assert job["status"] == "failed"
    assert graph(lab) == before


def test_single_job_idempotence_stale_version_cancel(lab):
    store, engine, rid = lab
    request = Action(action="import", parameters={"records": [{"content": "x"}]}, expected_graph_version=0, idempotency_key="same")
    first, fresh = store.reserve(rid, request)
    assert fresh
    assert store.reserve(rid, request)[0]["id"] == first["id"]
    with pytest.raises(Conflict):
        store.reserve(rid, request.model_copy(update={"idempotency_key": "other"}))
    with pytest.raises(Conflict):
        store.reserve(rid, request.model_copy(update={"parameters": {"records": [{"content": "changed"}]}}))
    store.transition(rid, first["id"], "cancelled")
    engine.execute(rid, first["id"])
    assert graph(lab)["nodes"] == []
    good(lab, "import", records=[{"content": "new"}])
    with pytest.raises(Conflict):
        store.reserve(rid, request.model_copy(update={"idempotency_key": "stale"}))
    assert store.reserve(rid, request)[0]["status"] == "cancelled"


def test_cohesive_neighborhood_avoids_single_link_chaining_and_overlaps():
    ids = ["a", "b", "c"]
    # a-b and b-c pass, a-c fails. A connected-components implementation would fail this test.
    vectors = [[1, 0], [.8, .6], [.28, .96]]
    settings = Settings(threshold=.7, min_pairwise=.7, jaccard=.9)
    proposals = discover(ids, vectors, settings, {i: {i} for i in ids})
    assert {tuple(p["member_ids"]) for p in proposals} == {("a", "b"), ("b", "c")}
    assert all(p["min_pairwise_cosine"] >= .7 for p in proposals)
    assert discover(ids, vectors, settings.model_copy(update={"threshold": .99}), {i: {i} for i in ids}) == []


def test_projection_does_not_influence_membership(lab):
    concepts(lab)
    good(lab, "embed")
    good(lab, "associate", settings={"threshold":0,"min_pairwise":0})
    before = [h["member_ids"] for h in graph(lab)["hyperedges"]]
    good(lab, "project")
    good(lab, "associate", settings={"threshold":0,"min_pairwise":0})
    after = [h["member_ids"] for h in graph(lab)["hyperedges"]]
    assert after == before + before
    fit = projection(["x","y"], [[1,0],[0,1]])
    assert fit["explained_variance"] == pytest.approx(1)
    assert len(fit["coordinates"]["x"]) == 3


class InvalidModels(Models):
    def generate(self, operation, data, settings, receipt):
        output = super().generate(operation, data, settings, receipt)
        if operation == "concept":
            output["evidence_quotes"][0]["quote"] = "fabricated source text"
        return output


def test_invalid_model_output_audited_no_phantom_nodes(lab):
    good(lab, "import", records=datasets()[0]["records"])
    good(lab, "episode", count=3)
    approve(lab)
    lab[1].models = InvalidModels()
    before = graph(lab)
    job = act(lab, "concept")
    assert job["status"] == "failed"
    assert "quote" in job["error"]
    assert graph(lab) == before
    assert lab[0].receipts(lab[2])


class RejectingModels(Models):
    def generate(self, operation, data, settings, receipt):
        output = super().generate(operation, data, settings, receipt)
        if operation == "validation":
            output.update(passed=False, reason="unsupported scope", unsupported=["broader than source observations"])
        return output


def test_validator_rejection_persists_edge_without_meta(lab):
    concepts(lab)
    good(lab, "embed")
    good(lab, "associate", settings={"threshold":0,"min_pairwise":0})
    edge = graph(lab)["hyperedges"][0]
    lab[1].models = RejectingModels()
    good(lab, "crystallize", [edge["id"]])
    assert not any(n["type"] == "meta_concept" for n in graph(lab)["nodes"])
    assert graph(lab)["hyperedges"][0]["rejection_reason"] == "unsupported scope"


def test_immutable_events_and_recovery(lab):
    store, _, rid = lab
    job, _ = store.reserve(rid, Action(action="embed", expected_graph_version=0, idempotency_key="recover"))
    with pytest.raises(sqlite3.IntegrityError):
        with store.connection() as db:
            db.execute("DELETE FROM events")
    store.recover()
    assert store.job(rid, job["id"])["status"] == "failed"
    assert store.run(rid)["status"] == "action_failed"


def test_api_snapshot_replay_export_and_cross_origin(tmp_path):
    app = create_app(tmp_path / "api.sqlite")
    with TestClient(app) as client:
        run = client.post("/api/runs", json={"name":"api","settings":{"memory_design":"legacy"}}).json()
        rid = run["id"]
        assert client.get("/api/datasets").status_code == 200
        body = {"action":"import","parameters":{"records":[{"content":"Evidence"}]},"expected_graph_version":0,"idempotency_key":"api-job"}
        # Reserve and execute deterministically rather than polling the test executor.
        job, _ = app.state.store.reserve(rid, Action(**body))
        app.state.engine.execute(rid, job["id"])
        assert client.get(f"/api/runs/{rid}/graph").json()["graph_version"] == 1
        assert client.get(f"/api/runs/{rid}/graph?version=0").json()["nodes"] == []
        assert client.post(f"/api/runs/{rid}/actions",json=body).status_code == 202
        assert client.post(f"/api/runs/{rid}/actions",json={**body,"idempotency_key":"stale"}).status_code == 409
        export = client.get(f"/api/runs/{rid}/export").json()
        assert export["run"]["graph"] == client.get(f"/api/runs/{rid}/graph").json()
        oid = export["run"]["graph"]["nodes"][0]["id"]
        assert client.get(f"/api/runs/{rid}/objects/{oid}").json()["generating_event"]["kind"] == "graph.patch"
        assert "evaluation_id" in client.get(f"/api/runs/{rid}/export?format=csv").text
        assert client.post("/api/runs",json={"name":"bad"},headers={"Origin":"https://untrusted.example"}).status_code == 403
        assert client.get("/api/runs/missing/graph").status_code == 404


def test_local_mode_never_falls_back_to_demo():
    settings = Settings(mode="polza", llm_revision="provider-managed", embedding_model="mnemos/nonexistent-test-model", embedding_revision="0000000000000000000000000000000000000000")
    with pytest.raises(Exception):
        Models().embed(["test"], settings)


def test_call_budget_failure_does_not_commit_partial_batch(lab):
    store, engine, _ = lab
    rid = store.create("budget", Settings(mode="demo",max_model_calls=1).model_dump())["id"]
    limited = (store, engine, rid)
    good(limited, "import", records=datasets()[0]["records"])
    job = act(limited, "episode", count=3)
    assert job["status"] == "failed"
    assert "budget" in job["error"]
    assert not any(n["type"] == "episode" for n in graph(limited)["nodes"])
    assert len(store.receipts(rid)) == 1


def test_edit_preserves_predecessor_and_requires_new_review(lab):
    good(lab, "import", records=datasets()[0]["records"])
    good(lab, "episode", count=3)
    approve(lab)
    good(lab, "concept")
    previous = next(n for n in graph(lab)["nodes"] if n["type"] == "concept")
    evidence = source_ids(graph(lab), previous["id"])
    good(lab, "edit", [previous["id"]], changes={"title": "Scoped concurrency limit", "proposition": "Bound concurrent work when the cited service reports overload."})
    after = graph(lab)
    original = next(n for n in after["nodes"] if n["id"] == previous["id"])
    revised = next(n for n in after["nodes"] if n.get("supersedes_id") == previous["id"])
    assert original["proposition"] == previous["proposition"]
    assert original["status"] == "superseded"
    assert revised["status"] == "needs_review"
    assert source_ids(after, revised["id"]) == evidence
    assert "vector" not in revised


def test_invalid_output_not_cached_and_can_retry(lab):
    good(lab, "import", records=datasets()[0]["records"])
    good(lab, "episode", count=3)
    approve(lab)
    lab[1].models = InvalidModels()
    assert act(lab, "concept")["status"] == "failed"
    lab[1].models = Models()
    good(lab, "concept")
    assert sum(n["type"] == "concept" for n in graph(lab)["nodes"]) == 3


def test_historical_inspection_and_evaluation_review_export(tmp_path):
    app = create_app(tmp_path / "reviews.sqlite")
    store, engine = app.state.store, app.state.engine
    rid = store.create("reviews", Settings(mode="demo",memory_design="legacy").model_dump())["id"]
    lab = (store, engine, rid)
    concepts(lab)
    good(lab, "embed")
    node = next(n for n in graph(lab)["nodes"] if n["type"] == "concept")
    historical = graph(lab)["graph_version"]
    source = next(iter(source_ids(graph(lab), node["id"])))
    good(lab, "challenge", [source], reason="Disputed for a versioned inspection test")
    with TestClient(app) as client:
        current = client.get(f"/api/runs/{rid}/objects/{node['id']}").json()
        old = client.get(f"/api/runs/{rid}/objects/{node['id']}?version={historical}").json()
        assert current["object"]["status"] == "needs_revalidation"
        assert old["object"]["status"] == "validated"
        result = client.post(f"/api/runs/{rid}/retrieve",json={"query":"recovery evidence", "variants":["A","C","E"]}).json()
        review = client.post(f"/api/runs/{rid}/evaluations/{result['id']}/review",json={"variant":"A","groundedness":"supported","correctness":"pass","reason":"Human reviewed the original evidence."})
        assert review.status_code == 200
        assert review.json()["results"][0]["answer_correctness"] == "pass"
        exported = client.get(f"/api/runs/{rid}/export").json()
        assert exported["evaluations"][0]["results"][0]["review_source"] == "human_researcher"
        perf = client.post(f"/api/runs/{rid}/performance",json={"fps":60,"mean_ms":16.7,"p95_ms":20,"samples":120,"user_agent":"test browser"})
        assert perf.status_code == 200
        assert perf.json()["node_count"] == len(graph(lab)["nodes"])


def test_recorded_fixture_source_hashes_are_reproducible():
    import hashlib
    from pathlib import Path
    fixture = next(d for d in datasets() if d["id"] == "recorded-agent-tools")
    assert len(fixture["records"]) == 6
    root = Path(__file__).parents[2]
    for record in fixture["records"]:
        path = root / record["source_path"]
        if not path.exists():
            continue  # Bundled event content/provenance stays portable without the larger run archive.
        original = path.read_text(encoding="utf-8").splitlines()[record["source_line"]-1]
        assert hashlib.sha256(original.encode()).hexdigest() == record["source_line_sha256"]
        assert json.loads(original) == json.loads(record["content"])
