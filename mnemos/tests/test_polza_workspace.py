import json

import pytest
from fastapi.testclient import TestClient

from mnemos.backend import adapters
from mnemos.backend.app import create_app
from mnemos.backend.engine import Engine
from mnemos.backend.memoryarena import checked_file, descriptors, load
from mnemos.backend.models import Action, Settings
from mnemos.backend.store import Store


class FakePolza:
    def __init__(self, finish="stop", error=False):
        self.records = []
        self.finish = finish
        self.error = error
        self.closed = False

    def complete(self, messages, **kwargs):
        self.messages, self.kwargs = messages, kwargs
        self.records.append({"input_tokens": 100, "output_tokens": 30, "total_tokens": 130, "cost_rub": .0123, "model": "qwen/qwen3.6-27b", "provider": "Chutes"})
        if self.error:
            raise ValueError("Provider error mentioning pza_fake_sensitive_key")
        return {"id": "request-123", "model": "qwen/qwen3.6-27b", "provider": "Chutes", "choices": [{"finish_reason": self.finish, "message": {"content": json.dumps({"passed": True, "reason": "Sources support the claim", "unsupported": [], "contradictions": [], "novelty": True})}}]}

    def close(self):
        self.closed = True


def test_remote_generation_audits_schema_usage_and_strips_vectors(monkeypatch):
    remote = FakePolza()
    monkeypatch.setattr(adapters, "polza_client", lambda settings: remote)
    receipts = []
    output = adapters.Models().generate("validation", {"candidate": {"proposition": "Supported claim", "vector": [1, 0], "embedding_input": "unnecessary"}}, Settings(), receipts.append)
    assert output["passed"]
    assert remote.kwargs["response_schema"]["additionalProperties"] is False
    assert "vector" not in remote.messages[1]["content"]
    assert receipts[0]["cost"] == .0123
    assert receipts[0]["cost_unit"] == "RUB"
    assert receipts[0]["request_id"] == "request-123"
    assert receipts[0]["usage"]["total_tokens"] == 130
    assert remote.closed


@pytest.mark.parametrize("remote", [FakePolza(finish="length"), FakePolza(error=True)])
def test_paid_failed_calls_keep_usage_and_redact_errors(monkeypatch, remote):
    monkeypatch.setattr(adapters, "polza_client", lambda settings: remote)
    receipts = []
    with pytest.raises(ValueError):
        adapters.Models().generate("validation", {}, Settings(), receipts.append)
    assert receipts[0]["cost"] == .0123
    assert receipts[0]["error"]
    assert "pza_fake_sensitive_key" not in json.dumps(receipts)


def test_ollama_is_not_an_execution_mode():
    with pytest.raises(ValueError):
        Settings(mode="local")
    assert Settings().mode == "polza"
    assert Settings().llm_url == "https://polza.ai/api/v1"


def test_memoryarena_import_is_chronological_public_and_hash_checked():
    catalog = descriptors()
    if not catalog:
        pytest.skip("MemoryArena corpus is not installed")
    groups = [d for d in catalog if d["corpus_kind"] == "public_inputs"]
    assert len(groups) == 270
    dataset = next(d for d in groups if d["id"] == "memoryarena-120")
    records = load(dataset)
    assert [r["sequence"] for r in records] == list(range(len(records)))
    assert records[0]["event_kind"] == "base_request"
    assert records[1]["event_kind"] == "public_base_itinerary"
    assert all("data/visible/120/" in r["source_path"] for r in records)
    assert all(len(r["source_file_sha256"]) == 64 for r in records)
    with pytest.raises(ValueError):
        checked_file("data/hidden/anything.json")
    with pytest.raises(ValueError, match="checksum"):
        checked_file(records[0]["source_path"], "0" * 64)


def test_recorded_actions_end_their_causal_episode():
    catalog = descriptors()
    dataset = next((d for d in catalog if d["corpus_kind"] == "recorded_execution"), None)
    if not dataset:
        pytest.skip("Recorded execution archive not installed")
    records = load(dataset)
    assert len(records) < 94
    sessions = list(dict.fromkeys(r["session"] for r in records))
    first = [r for r in records if r["session"] == sessions[0]]
    assert len(first) == 1
    assert first[0]["event_kind"] == "execution_trace"
    events = first[0]["events"]
    assert events[-1]["kind"] == "agent_action"
    assert {e["kind"] for e in events} >= {"request", "tool_call", "tool_result"}
    assert sum(len(r.get("events", [])) for r in records) == 94
    assert all(e["metadata"]["source_event_id"].startswith("120:") for e in events)
    assert json.loads(events[0]["content"])["source_kind"] == "DECLARED_PROPERTY_SCHEMA"


def test_next_source_ingestion_and_catalog_metadata(tmp_path):
    app = create_app(tmp_path / "workspace.sqlite")
    with TestClient(app) as client:
        catalog = client.get("/api/datasets").json()
        assert all("records" not in d for d in catalog)
        assert client.get("/api/provider").json()["generation"] == "remote"
        assert "api_key" not in json.dumps(client.get("/api/provider").json())
        store, engine = app.state.store, app.state.engine
        rid = store.create("incremental", Settings(mode="demo").model_dump())["id"]
        for version in (0, 1):
            request = Action(action="import", parameters={"dataset_id": "agent-task-observations", "count": 1}, expected_graph_version=version, idempotency_key=f"next-{version}")
            job, _ = store.reserve(rid, request)
            engine.execute(rid, job["id"])
            assert store.job(rid, job["id"])["status"] == "succeeded"
        nodes = store.run(rid)["graph"]["nodes"]
        assert len(nodes) == 2
        assert nodes[0]["content"] != nodes[1]["content"]


def test_factory_import_does_not_recover_a_live_servers_jobs(tmp_path):
    path = tmp_path / "live.sqlite"
    first = create_app(path)
    store = first.state.store
    rid = store.create("live", Settings(mode="demo").model_dump())["id"]
    job, _ = store.reserve(rid, Action(action="embed", expected_graph_version=0, idempotency_key="live-job"))
    second = create_app(path)
    assert second.state.store.job(rid, job["id"])["status"] == "queued"
    store.transition(rid, job["id"], "running")
    store.transition(rid, job["id"], "failed", "Server recovery")
    store.commit(rid, job["id"], 0, store.run(rid)["graph"], {}, "event_not_committed")
    assert store.run(rid)["version"] == 0
    assert store.job(rid, job["id"])["status"] == "failed"


def test_repeated_actions_are_distinct_occurrences_but_reimport_is_deduplicated(tmp_path):
    store = Store(tmp_path / "repeats.sqlite")
    engine = Engine(store)
    rid = store.create("repeat", Settings(mode="demo").model_dump())["id"]
    records = [{"content": "Same repeated tool request", "source_path": "archive.jsonl", "source_file_sha256": "abc", "source_event_id": f"event-{i}"} for i in range(2)]
    job, _ = store.reserve(rid, Action(action="import", parameters={"records": [*records, records[0]]}, expected_graph_version=0, idempotency_key="repeats"))
    engine.execute(rid, job["id"])
    assert len(store.run(rid)["graph"]["nodes"]) == 2
    assert len({n["source_identity"] for n in store.run(rid)["graph"]["nodes"]}) == 2


def test_quote_schema_pins_source_and_literal_excerpts():
    sources = [{"id": "trace-1", "content": "Original first line.\nExact second line."}]
    schema = adapters.output_schema("episode", sources)
    choices = schema["properties"]["facts"]["items"]["properties"]["evidence_quotes"]["items"]["anyOf"]
    assert choices[0]["properties"]["source_id"]["enum"] == ["trace-1"]
    assert all(q in sources[0]["content"] for q in choices[0]["properties"]["quote"]["enum"])
