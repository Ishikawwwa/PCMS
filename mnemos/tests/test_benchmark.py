import copy
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from mnemos.backend.adapters import Models
from mnemos.backend.app import create_app
from mnemos.backend.benchmark import Execution, Stopped, case_detail, control, prepare, session, resume_checkpoint
from mnemos.backend.engine import Engine
from mnemos.backend.models import BenchmarkInput, Settings
from mnemos.backend.store import Store
from mnemos.backend.provider import reconcile_failed_billing, accounted_receipts
from src.constraint_memory.benchmark_adapter.planner_worker import format_person_plan
from src.constraint_memory.benchmark_adapter.submissions import named_sections
from src.constraint_memory.benchmark_adapter.scoring import score_official


ROOT = Path(__file__).parents[2]


def test_failed_request_accounting_is_reconciled_without_retry_or_rewriting(tmp_path):
    store = Store(tmp_path / "billing.sqlite")
    run, job = prepare(store, BenchmarkInput(group_id="17"))
    store.receipt(run["id"], job["id"], {"operation":"episode", "cost":None, "cost_unit":"RUB", "provider_accounting":[{"request_id":"gen_123"}]})
    original = store.receipts(run["id"])[0]
    calls = []
    def fetch(request_id):
        calls.append(request_id)
        return {"id":request_id,"status":"failed","clientCost":"0","usage":None}
    reconcile_failed_billing(store, run["id"], fetch=fetch)
    reconcile_failed_billing(store, run["id"], fetch=fetch)
    assert calls == ["gen_123"]
    assert store.receipts(run["id"])[0] == original
    effective = accounted_receipts(store.receipts(run["id"]))
    assert len(effective) == 1 and effective[0]["data"]["cost"] == 0
    assert effective[0]["data"]["billing_reconciliation"]["original_receipt_id"] == original["id"]
    store.receipt(run["id"], job["id"], {"cost":None,"cost_unit":"RUB","provider_accounting":[{"request_id":"gen_456"}]})
    with pytest.raises(ValueError, match="not final"):
        reconcile_failed_billing(store, run["id"], fetch=lambda _: {"id":"gen_other","status":"completed","clientCost":"0"})


def test_execution_events_form_one_trace_with_exact_call_result_pairing(tmp_path):
    store = Store(tmp_path / "traces.sqlite")
    run, job = prepare(store, BenchmarkInput(group_id="17"))
    store.transition(run["id"], job["id"], "running")
    execution = Execution(store, Engine(store), run["id"], job["id"])
    execution.turn, execution.current_step = 1, 1
    execution.observe("agent_output", "Find an available hotel", "agent:1")
    execution.observe("tool_call", {"id":"call1","name":"HotelSearch","arguments":{"city":"Portland"}}, "call:1", tool_call_id="call1")
    assert not store.run(run["id"])["graph"]["nodes"]
    execution.observe("tool_result", "Riverside Hotel: $120/night", "result:1", tool_call_id="call1")
    execution.flush_trace()
    execution.observe("tool_result", "Riverside Hotel: $120/night", "result:1", tool_call_id="call1")
    execution.flush_trace()
    nodes = store.run(run["id"])["graph"]["nodes"]
    assert len(nodes) == 1
    trace = nodes[0]
    assert trace["source_metadata"]["kind"] == "execution_trace"
    events = trace["source_metadata"]["events"]
    assert [e["kind"] for e in events] == ["agent_output","tool_call","tool_result"]
    assert events[1]["metadata"]["tool_call_id"] == events[2]["metadata"]["tool_call_id"] == "call1"
    assert "$120/night" in trace["content"]
    assert json.loads(events[1]["content"])["arguments"] == {"city":"Portland"}
    assert len([e for e in store.events(run["id"]) if e["kind"] == "benchmark.execution_event"]) == 3
pytestmark = pytest.mark.skipif(not (ROOT / "data/manifests/splits.json").exists(), reason="MemoryArena not installed")


class FakeClient:
    def __init__(self, settings):
        self.records = []
    def set_scope(self, *args, **kwargs):
        pass
    def close(self):
        pass


class BasePlanPlanner:
    """Scripted plumbing diagnostic: copies public base, makes no model calls."""
    contexts = []
    requests = []
    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs
    def run_turn(self, request, context, base):
        self.contexts.append(context)
        self.requests.append(request)
        text = format_person_plan(request["name"], base["daily_plans"])
        section = named_sections(text)[request["name"]]
        return {"group_id": request["group_id"], "traveler_index": request["traveler_index"], "name": request["name"], "final_plan": text, "submitted_text": section, "plan": copy.deepcopy(base["daily_plans"]), "accepted_submissions": [{"name": request["name"], "submitted_text": section, "plan": copy.deepcopy(base["daily_plans"])}], "scratchpad": [], "structured_tool_results": [], "total_steps": 0, "success": True, "error_message": None, "worker_audit": {"hidden_modules_loaded": [], "model_credentials_present": False, "scripted_test_only": True}}


class ScriptedModels(Models):
    def generate(self, operation, data, settings, receipt):
        return super().generate(operation, data, settings.model_copy(update={"mode": "demo"}), receipt)
    def embed(self, texts, settings):
        return super().embed(texts, settings.model_copy(update={"mode": "demo"}))


def ready(root):
    return {"provenance": {"upstream_sha": "test-pin", "dataset_sha256": "test-sha"}, "database": {"database_path": "external/MemoryArena/env/env_systems/travel_planner_env/database"}}


@pytest.mark.parametrize("mode", ["none", "history", "mnemos"])
def test_full_case_execution_commits_every_traveler_then_scores(tmp_path, mode):
    store = Store(tmp_path / "case.sqlite")
    engine = Engine(store, ScriptedModels())
    run, job = prepare(store, BenchmarkInput(group_id="17", memory_mode=mode, settings=Settings(memory_design="legacy")))
    assert store.run(run["id"])["graph"]["nodes"] == []
    assert "travelers" not in session(store, run["id"])["config"]
    scored = []
    def scorer(upstream, hidden_path, predictions, gids, **kwargs):
        assert gids == ["17"]
        assert len(predictions) == 5
        assert len(session(store, run["id"])["travelers"]) == 5
        scored.extend(predictions)
        return {"official_metrics": {"SR": 0, "PS": 0, "SPS": 0}, "per_group": []}
    Execution(store, engine, run["id"], job["id"], planner_factory=BasePlanPlanner, client_factory=FakeClient, scorer=scorer, readiness=ready).run()
    result = session(store, run["id"])
    assert result["status"] == "completed", result.get("error")
    assert store.job(run["id"], job["id"])["status"] == "succeeded"
    assert len(scored) == 5
    assert [r["traveler_index"] for r in scored] == [1,2,3,4,5]
    assert result["scores"]["official_metrics"]["SR"] == 0
    assert all("hidden" not in n["provenance"].lower() or "not a gold" in n.get("explanation", "") for n in store.run(run["id"])["graph"]["nodes"] if n["type"] == "trace")
    if mode == "mnemos":
        nodes = store.run(run["id"])["graph"]["nodes"]
        assert any(n["type"] == "concept" for n in nodes)
        assert not any(n.get("validation_state") == "human_reviewed" for n in nodes)
    for read in result["memory_reads"]:
        assert read["tokens"] <= read["budget"]


def test_pause_single_step_and_stop_at_action_boundary(tmp_path):
    store = Store(tmp_path / "step.sqlite")
    run, job = prepare(store, BenchmarkInput(group_id="17", start_paused=True))
    store.transition(run["id"], job["id"], "running")
    worker = Execution(store, Engine(store), run["id"], job["id"])
    entered = threading.Event()
    def gated():
        worker.gate("test.decision")
        entered.set()
    thread = threading.Thread(target=gated)
    thread.start()
    assert not entered.wait(.1)
    control(store, run["id"], "step")
    assert entered.wait(2)
    thread.join(2)
    assert session(store, run["id"])["control"]["permits"] == 0
    assert session(store, run["id"])["control"]["mode"] == "paused"
    control(store, run["id"], "stop")
    with pytest.raises(Stopped):
        worker.gate("next.decision")


def test_infrastructure_failure_never_becomes_a_zero_official_score(tmp_path):
    store = Store(tmp_path / "failed.sqlite")
    run, job = prepare(store, BenchmarkInput(group_id="17"))
    def bad_readiness(root):
        raise ValueError("Pinned tool database mismatch")
    Execution(store, Engine(store), run["id"], job["id"], readiness=bad_readiness).run()
    result = session(store, run["id"])
    assert result["status"] == "failed"
    assert result["scores"] is None
    assert "mismatch" in result["error"]


def test_case_preview_and_benchmark_export_are_separate_from_execution_memory(tmp_path):
    app = create_app(tmp_path / "api.sqlite")
    with TestClient(app) as client:
        detail = client.get("/api/memoryarena/cases/17").json()
        assert detail["traveler_count"] == 5
        assert len(detail["travelers"]) == 5
        assert "answers" not in detail
        run, job = prepare(app.state.store, BenchmarkInput(group_id="17", start_paused=True))
        exported = client.get(f"/api/runs/{run['id']}/export").json()
        assert exported["benchmark"]["group_id"] == "17"
        assert exported["run"]["graph"]["nodes"] == []
        assert exported["source_archive"]["timing"] == "Before execution job dispatch"
        assert "mnemos/backend/benchmark.py" in exported["source_archive"]["sha256"]
        assert client.get(f"/api/runs/{run['id']}/benchmark/scores.csv").status_code == 409
        assert client.get("/api/memoryarena/cases/999").status_code == 404


def test_complete_case_reaches_real_official_scorer_with_scripted_public_base_plans(tmp_path):
    """Real official scoring plumbing, never an LLM efficacy measurement."""
    store = Store(tmp_path / "official.sqlite")
    run, job = prepare(store, BenchmarkInput(group_id="17", memory_mode="none"))
    Execution(store, Engine(store), run["id"], job["id"], planner_factory=BasePlanPlanner, client_factory=FakeClient, scorer=score_official).run()
    result = session(store, run["id"])
    assert result["status"] == "completed", result.get("error")
    assert result["scores"]["metric_metadata"]["dataset_checksum_verified"]
    assert result["scores"]["metric_metadata"]["scorer_isolated"]
    assert result["scores"]["per_group"][0]["traveler_count"] == 5
    assert set(result["scores"]["official_metrics"]) == {"SR", "PS", "SPS"}


def test_checkpoint_recovery_does_not_repeat_committed_travelers(tmp_path):
    store = Store(tmp_path / "recover.sqlite")
    run, job = prepare(store, BenchmarkInput(group_id="17", memory_mode="none"))
    class FailAfterTwo(Execution):
        def consolidate(self):
            if self.turn == 2:
                raise ValueError("Recorded memory schema failure after the second commit")
    kwargs = dict(planner_factory=BasePlanPlanner, client_factory=FakeClient, readiness=ready, scorer=lambda *a, **k: {"official_metrics":{"SR":0,"PS":0,"SPS":0},"per_group":[]})
    BasePlanPlanner.requests = []
    FailAfterTwo(store, Engine(store), run["id"], job["id"], **kwargs).run()
    before = session(store, run["id"])
    assert before["status"] == "failed"
    assert len(before["travelers"]) == 2
    assert [r["traveler_index"] for r in BasePlanPlanner.requests] == [1,2]
    resumed = resume_checkpoint(store, run["id"])
    Execution(store, Engine(store), run["id"], resumed["id"], **kwargs).run()
    after = session(store, run["id"])
    assert after["status"] == "completed", after.get("error")
    assert [r["traveler_index"] for r in BasePlanPlanner.requests] == [1,2,3,4,5]
    assert len(after["recoveries"]) == 1
    assert after["predictions"][:2] == before["predictions"]


def test_invalid_paid_memory_output_is_rejected_without_stopping_planning(tmp_path):
    store = Store(tmp_path / "rejected.sqlite")
    run, job = prepare(store, BenchmarkInput(group_id="17", memory_mode="mnemos"))
    class InvalidPaidMemory(ScriptedModels):
        def generate(self, operation, data, settings, receipt):
            receipt({"operation":operation,"cost":.01,"cost_unit":"RUB","provider_accounting":[{"request_status":"completed","response_accepted":True}]})
            return {}
    execution = Execution(store, Engine(store, InvalidPaidMemory()), run["id"], job["id"], planner_factory=BasePlanPlanner, client_factory=FakeClient, readiness=ready, scorer=lambda *a, **k: {"official_metrics":{"SR":0,"PS":0,"SPS":0},"per_group":[]})
    execution.run()
    result = session(store, run["id"])
    assert result["status"] == "completed", result.get("error")
    assert len(result["memory_failures"]) == 5
    assert len(result["travelers"]) == 5
    assert not any(n["type"] != "trace" for n in store.run(run["id"])["graph"]["nodes"])
