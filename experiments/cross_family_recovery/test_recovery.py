"""Independent recovery tests using invented receipts and a no-network transport.

These checks exercise recovery integrity and accounting, not task performance.
The real parent run is read only by the explicit provenance checks below.
"""
from __future__ import annotations

import builtins
from collections import Counter
import copy
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import socket
from types import SimpleNamespace
import uuid

import pytest

from constraint_memory.benchmark_adapter.planner import PlannerWorker
from constraint_memory.memory import create_backend
from constraint_memory.polza import ProviderError
from experiments.cross_family_recovery import run


ROOT = Path(__file__).resolve().parents[2]
PARENT_ID = "cross-family-primary-20261015T192052114101Z"


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_lines(path):
    path = Path(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def write_lines(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n" for value in values), encoding="utf-8")


@pytest.fixture
def destination():
    # Workspace-inherited ACLs avoid restricted host temporary-directory ACLs.
    result = ROOT / "tmp/cross-family-recovery-tests" / uuid.uuid4().hex
    result.mkdir(parents=True)
    return result


@pytest.fixture
def fake_parent():
    config = json.loads((ROOT / "runs" / PARENT_ID / "manifest.json").read_text(encoding="utf-8"))["config"]
    accepted = []
    for index, operation in enumerate(("agent", "extraction"), 1):
        intent = {"attempt_id": f"request-{index:05d}", "event": "INTENT_ONLY_NOT_DISPATCH_PROOF",
                  "group_id": "E1:120", "traveler": "OfflineTraveler", "operation": operation,
                  "messages": [{"role": "user", "content": f"Invented offline {operation} request."}],
                  "tools": None, "max_tokens_override": None, "json_output": False, "response_schema": None}
        raw = {"id": f"offline-response-{index}", "model": config["agent"]["model"], "provider": "offline-provider",
               "choices": [{"message": {"role": "assistant", "content": "{}"}, "finish_reason": "stop"}],
               "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5, "cost_rub": .25}}
        usage = {"group_id": intent["group_id"], "traveler": intent["traveler"], "operation": operation,
                 "request_id": raw["id"], "request_status": "completed", "response_accepted": True,
                 "model_identity_validation": "EXACT_MATCH", "model": raw["model"],
                 "input_tokens": 2, "output_tokens": 3, "total_tokens": 5, "cost_rub": .25,
                 "currency": "RUB", "latency_seconds": 23.0}
        accepted.append({"intent": intent, "raw_response": raw, "usage": usage})
    failed_intent = {"attempt_id": "request-00003", "event": "INTENT_ONLY_NOT_DISPATCH_PROOF",
                     "group_id": "E2:69", "traveler": "Olivia", "operation": "agent",
                     "messages": [{"role": "user", "content": "Invented continuation boundary."}],
                     "tools": None, "max_tokens_override": None, "json_output": False, "response_schema": None}
    failed_usage = {"group_id": "E2:69", "traveler": "Olivia", "operation": "agent", "request_sent": True,
                    "request_id": "offline-unknown", "request_status": "failed", "input_tokens": None,
                    "output_tokens": None, "total_tokens": None, "cost_rub": None,
                    "billing_status": "unknown_after_failed_request"}
    return {"run_id": PARENT_ID, "directory": ROOT / "runs" / PARENT_ID,
            "manifest": {"run_id": PARENT_ID, "config": config, "status": "failed"},
            "protocol": {"config": config}, "accepted": accepted,
            "failed_intent": failed_intent, "failed_usage": failed_usage}


def make_client(parent, destination, transport, *, live=False, gate_callback=None, **overrides):
    agent = parent["manifest"]["config"]["agent"]
    kwargs = {"api_key": "offline-only-not-a-credential", "model": agent["model"],
              "temperature": agent["temperature"], "max_tokens": agent["max_completion_tokens"],
              "extraction_max_tokens": agent["max_extraction_tokens"], "seed": 20261013,
              "provider": copy.deepcopy(agent["routing"]), "reasoning": copy.deepcopy(agent["reasoning"]),
              "extraction_reasoning": copy.deepcopy(agent["extraction_reasoning"]), "stream": False,
              "max_total_tokens": math.inf, "max_cost_rub": None, "transport": transport,
              "usage_path": destination / "usage.jsonl"}
    kwargs.update(overrides)
    return run.RecoveryClient(parent, destination, live=live, gate_callback=gate_callback, **kwargs)


def issue(client, intent):
    client.set_scope(intent["group_id"], intent["traveler"])
    return client.complete(copy.deepcopy(intent["messages"]), tools=copy.deepcopy(intent["tools"]),
                           operation=intent["operation"], max_tokens=intent["max_tokens_override"],
                           json_output=intent["json_output"], response_schema=copy.deepcopy(intent["response_schema"]))


def unexpected_transport(*args, **kwargs):
    pytest.fail("Cached or rejected request reached the live transport")


def test_canonical_ignores_object_key_order_but_preserves_types_and_sequence():
    assert run.canonical({"b": 2, "a": [1, 2]}) == run.canonical({"a": [1, 2], "b": 2})
    assert run.canonical([1, 2]) != run.canonical([2, 1])
    assert run.canonical(True) != run.canonical(1)
    assert run.canonical(1) != run.canonical("1")
    assert run.canonical({"a": None}) != run.canonical({})


def test_all_cached_receipts_reuse_native_validation_without_transport(fake_parent, destination):
    original = copy.deepcopy(fake_parent)
    with make_client(fake_parent, destination, unexpected_transport) as client:
        for entry in fake_parent["accepted"]:
            assert issue(client, entry["intent"]) == entry["raw_response"]
        assert len(client.records) == 2
        assert all(record["model_identity_validation"] == "EXACT_MATCH" for record in client.records)
        assert all(record["response_accepted"] for record in client.records)
    assert fake_parent == original


@pytest.mark.parametrize("field,value", [
    ("group_id", "E2:120"), ("traveler", "OtherTraveler"), ("operation", "extraction"),
    ("messages", [{"role": "user", "content": "Changed request"}]),
    ("tools", [{"type": "function", "function": {"name": "changed"}}]),
    ("max_tokens_override", 4), ("json_output", True),
    ("response_schema", {"type": "object", "properties": {}}),
])
def test_every_semantic_intent_difference_stops_before_transport(fake_parent, destination, field, value):
    changed = copy.deepcopy(fake_parent["accepted"][0]["intent"])
    changed[field] = value
    with make_client(fake_parent, destination, unexpected_transport) as client:
        with pytest.raises((ValueError, ProviderError, RuntimeError, AssertionError)):
            issue(client, changed)
        assert client.records == []


@pytest.mark.parametrize("field,value", [
    ("model", "other/model"), ("temperature", .75), ("seed", 1),
    ("max_tokens", 12), ("provider", {"only": ["another-provider"], "allow_fallbacks": False}),
    ("reasoning", {"enabled": False}),
])
def test_payload_semantic_drift_stops_before_cache_or_network(fake_parent, destination, field, value):
    with pytest.raises((ValueError, ProviderError, RuntimeError, AssertionError)):
        with make_client(fake_parent, destination, unexpected_transport, **{field: value}) as client:
            issue(client, fake_parent["accepted"][0]["intent"])


def test_extractor_payload_configuration_is_also_checked(fake_parent, destination):
    with pytest.raises((ValueError, ProviderError, RuntimeError, AssertionError)):
        with make_client(fake_parent, destination, unexpected_transport,
                         extraction_reasoning={"enabled": True}) as client:
            for entry in fake_parent["accepted"]:
                issue(client, entry["intent"])


def test_cached_wrong_model_is_rejected_by_native_identity_guard(fake_parent, destination):
    fake_parent["accepted"][0]["raw_response"]["model"] = "wrong/model"
    with make_client(fake_parent, destination, unexpected_transport) as client:
        with pytest.raises((ProviderError, ValueError, RuntimeError)):
            issue(client, fake_parent["accepted"][0]["intent"])


def test_cached_invalid_usage_cannot_bypass_native_accounting(fake_parent, destination):
    fake_parent["accepted"][0]["raw_response"]["usage"]["prompt_tokens"] = -1
    # Keep the two saved receipts mutually consistent so this specifically
    # exercises the native provider-usage type/range guard after cache lookup.
    fake_parent["accepted"][0]["usage"]["input_tokens"] = -1
    with make_client(fake_parent, destination, unexpected_transport) as client:
        with pytest.raises((ProviderError, ValueError, RuntimeError)):
            issue(client, fake_parent["accepted"][0]["intent"])


def test_cached_receipt_request_id_must_match_its_original_usage(fake_parent, destination):
    intent = copy.deepcopy(fake_parent["accepted"][0]["intent"])
    fake_parent["accepted"][0]["raw_response"]["id"] = "different-original-request"
    with pytest.raises((ProviderError, ValueError, RuntimeError, AssertionError)):
        with make_client(fake_parent, destination, unexpected_transport) as client:
            issue(client, intent)


@pytest.mark.parametrize("corruption", ["reordered", "missing", "duplicated"])
def test_cache_shape_cannot_silently_skip_or_search_for_another_receipt(fake_parent, destination, corruption):
    intents = [copy.deepcopy(entry["intent"]) for entry in fake_parent["accepted"]]
    if corruption == "reordered":
        fake_parent["accepted"].reverse()
    elif corruption == "missing":
        fake_parent["accepted"].pop()
    else:
        fake_parent["accepted"].insert(1, copy.deepcopy(fake_parent["accepted"][0]))
    with pytest.raises((ProviderError, ValueError, RuntimeError, AssertionError)):
        with make_client(fake_parent, destination, unexpected_transport) as client:
            for intent in intents:
                issue(client, intent)


def test_offline_cache_exhaustion_raises_unambiguously_unsent_boundary(fake_parent, destination):
    with make_client(fake_parent, destination, unexpected_transport) as client:
        for entry in fake_parent["accepted"]:
            issue(client, entry["intent"])
        with pytest.raises(run.ReplayBoundaryReached) as failure:
            issue(client, fake_parent["failed_intent"])
        assert failure.value.request_sent is False


def test_one_explicit_replacement_dispatch_occurs_only_after_gate(fake_parent, destination):
    events = []
    new_raw = copy.deepcopy(fake_parent["accepted"][0]["raw_response"])
    new_raw["id"] = "offline-new-response"

    def gate():
        events.append("gate")

    def transport(method, path, payload):
        assert events == ["gate"]
        events.append("network")
        assert method == "POST" and path == "/chat/completions"
        assert not payload.get("stream", False)
        assert payload["model"] == "qwen/qwen3.6-27b"
        return copy.deepcopy(new_raw)

    with make_client(fake_parent, destination, transport, live=True, gate_callback=gate) as client:
        for entry in fake_parent["accepted"]:
            issue(client, entry["intent"])
        assert events == []
        assert issue(client, fake_parent["failed_intent"])["id"] == "offline-new-response"
        stats = client.get_usage_stats()
        assert stats["inherited_accepted_responses"] == 2
        assert stats["new_request_attempts"] == 1
        assert isinstance(stats["inherited_known_cost_rub_exact"], str)
        assert isinstance(stats["new_known_cost_rub_exact"], str)
        assert isinstance(stats["combined_known_cost_rub_exact"], str)
        assert Decimal(stats["inherited_known_cost_rub_exact"]) == Decimal("0.50")
        assert Decimal(stats["new_known_cost_rub_exact"]) == Decimal("0.25")
        assert Decimal(stats["combined_known_cost_rub_exact"]) == Decimal("0.75")
        assert stats["total_cost_rub"] is None
        assert stats["inherited_failed_attempts"] == 1
        assert stats["inherited_failed_usage"] == fake_parent["failed_usage"]
        assert sum(bool(record["reused"]) for record in client.records) == 2
        assert sum(bool(record["new_billable"]) for record in client.records) == 1
    assert events == ["gate", "network"]
    replacement = json.loads((destination / "replacement_attempt.json").read_text(encoding="utf-8"))
    assert replacement["parent_run_id"] == PARENT_ID
    assert replacement["parent_attempt_id"] == fake_parent["failed_intent"]["attempt_id"]
    assert replacement["original_bill_known"] is False


def test_bad_boundary_never_runs_gate_or_network(fake_parent, destination):
    events = []
    with make_client(fake_parent, destination, unexpected_transport, live=True,
                     gate_callback=lambda: events.append("gate")) as client:
        for entry in fake_parent["accepted"]:
            issue(client, entry["intent"])
        changed = copy.deepcopy(fake_parent["failed_intent"])
        changed["messages"][0]["content"] = "Different boundary"
        with pytest.raises((ValueError, ProviderError, RuntimeError, AssertionError)):
            issue(client, changed)
    assert events == []


def test_failed_boundary_proof_prevents_new_request(fake_parent, destination):
    def bad_gate():
        raise ValueError("Independent replay prefix differs")

    with make_client(fake_parent, destination, unexpected_transport, live=True, gate_callback=bad_gate) as client:
        for entry in fake_parent["accepted"]:
            issue(client, entry["intent"])
        with pytest.raises(ValueError, match="prefix differs"):
            issue(client, fake_parent["failed_intent"])


def test_future_ambiguous_failure_is_not_retried(fake_parent, destination):
    calls = []

    def transport(*args):
        calls.append(args)
        raise ProviderError("Invented sent interruption", request_sent=True, request_id="offline-failure")

    with make_client(fake_parent, destination, transport, live=True, gate_callback=lambda: None) as client:
        for entry in fake_parent["accepted"]:
            issue(client, entry["intent"])
        with pytest.raises(ProviderError, match="sent interruption"):
            issue(client, fake_parent["failed_intent"])
        assert len(calls) == 1
        assert client.records[-1]["total_tokens"] is None
        assert client.records[-1]["cost_rub"] is None
        stats = client.get_usage_stats()
        assert stats["total_cost_rub"] is None
        assert stats["inherited_failed_attempts"] == 1
        assert stats["new_request_attempts"] == 1
        assert Decimal(stats["combined_known_cost_rub_exact"]) == Decimal(".50")


def test_failed_parent_bill_stays_unknown_separate_from_execution_records(fake_parent, destination):
    with make_client(fake_parent, destination, unexpected_transport) as client:
        for entry in fake_parent["accepted"]:
            issue(client, entry["intent"])
        # Uncertain prior dispatches must not enter the native active ledger:
        # they would block _check_budget despite the unlimited aggregate cap.
        assert all(record["total_tokens"] is not None for record in client.records)
        stats = client.get_usage_stats()
        assert stats["inherited_accepted_responses"] == 2
        assert stats["new_request_attempts"] == 0
        assert Decimal(stats["inherited_known_cost_rub_exact"]) == Decimal(".50")
        assert Decimal(stats["new_known_cost_rub_exact"]) == Decimal(0)
        assert Decimal(stats["combined_known_cost_rub_exact"]) == Decimal(".50")
        assert stats["total_cost_rub"] is None
        assert stats["inherited_failed_usage"]["cost_rub"] is None
        assert stats["inherited_failed_usage"]["total_tokens"] is None
    assert fake_parent["failed_usage"]["cost_rub"] is None
    assert fake_parent["failed_usage"]["total_tokens"] is None


def test_actual_parent_provenance_validation_reads_no_secret_or_hidden_oracle(monkeypatch):
    original_open, original_path_open = builtins.open, Path.open

    def check(path):
        if isinstance(path, (str, Path)):
            normalized = str(path).replace("\\", "/").lower()
            assert "/data/hidden/" not in normalized
            assert "/data/audit_annotations/" not in normalized
            assert Path(path).name.lower() != ".env"

    def guarded_open(path, *args, **kwargs):
        check(path)
        return original_open(path, *args, **kwargs)

    def guarded_path_open(path, *args, **kwargs):
        check(path)
        return original_path_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(Path, "open", guarded_path_open)
    monkeypatch.setattr(socket.socket, "connect", unexpected_transport)
    actual = run.read_parent(ROOT, PARENT_ID)
    assert len(actual["accepted"]) == 155
    assert actual["failed_intent"]["attempt_id"] == "request-00156"
    assert actual["failed_usage"]["request_sent"] is True
    assert actual["failed_usage"]["cost_rub"] is None


def test_read_parent_validation_does_not_mutate_original_artifacts():
    parent = ROOT / "runs" / PARENT_ID
    before = {path.relative_to(parent).as_posix(): file_hash(path) for path in parent.rglob("*") if path.is_file()}
    run.read_parent(ROOT, PARENT_ID)
    after = {path.relative_to(parent).as_posix(): file_hash(path) for path in parent.rglob("*") if path.is_file()}
    assert before == after


@pytest.fixture
def prefix_pair(destination):
    parent, child = destination / "parent-prefix", destination / "child-prefix"
    scope = {"variant": "E2", "group_id": "69"}
    ordinary = {
        "events.jsonl": [{**scope, "event_id": "69:1", "step": 1, "type": "request", "content": "offline"}],
        "memory_ops.jsonl": [{**scope, "op": "reset", "arg": "69", "config": {"token_budget": 32768}}],
        "extractions.jsonl": [{**scope, "traveler": "Olivia", "extraction": {"constraints": []}}],
        "retrievals.jsonl": [{**scope, "traveler": "Olivia", "context": "All exact memory contents", "records": []}],
        "recoveries.jsonl": [{**scope, "charged_round": 6, "terminal": "no_recovery_required"}],
        "submission_events.jsonl": [{**scope, "charged_round": 6, "accepted": {"accepted": []}, "steps_used": 2}],
        "predictions.jsonl": [{**scope, "traveler_index": 6, "final_plan": "Exact actual action", "plan": []}],
        "traces.jsonl": [{**scope, "traveler_index": 6, "result": {"final_plan": "Exact actual action",
                             "latency_seconds": 12.0, "scratchpad": [{"thought": "Exact thought"}],
                             "structured_tool_results": [{"tool_name": "lookup", "latency_seconds": 2.0,
                                                           "displayed_rows": [{"Price": 5}],
                                                           "formatted_result": "Exact displayed result"}]},
                           "commit": {"stored_ids": {"decisions": ["d1"]}}}],
        "session_completions.jsonl": [{"execution_index": 1, **scope, "predictions": [{"plan": []}],
                                        "row": {**scope, "wall_clock_seconds": 25.0,
                                                "usage": [{"operation": "agent", "latency_seconds": 20.0,
                                                           "input_tokens": 2, "output_tokens": 3, "cost_rub": .25}],
                                                "memory_stats": {"record_count": 1}}}],
    }
    for directory in (parent, child):
        for filename, rows in ordinary.items():
            write_lines(directory / filename, rows)
        for variant, group in (("E1", "120"), ("E2", "69")):
            write_json(directory / "snapshots" / variant / f"{group}.json",
                       {"variant": variant, "group_id": group, "evidence": [], "decisions": [{"version": 1}]})
    return parent, child


def test_prefix_comparison_accepts_exact_semantics(prefix_pair):
    parent, child = prefix_pair
    assert isinstance(run.compare_prefix(parent, child), dict)


def test_prefix_comparison_excludes_only_reconstructed_timing(prefix_pair):
    parent, child = prefix_pair
    trace = json_lines(child / "traces.jsonl")
    trace[0]["result"]["latency_seconds"] = .05
    trace[0]["result"]["structured_tool_results"][0]["latency_seconds"] = .01
    write_lines(child / "traces.jsonl", trace)
    sessions = json_lines(child / "session_completions.jsonl")
    sessions[0]["row"]["wall_clock_seconds"] = .15
    sessions[0]["row"]["usage"][0]["latency_seconds"] = .1
    write_lines(child / "session_completions.jsonl", sessions)
    assert isinstance(run.compare_prefix(parent, child), dict)


@pytest.mark.parametrize("filename,mutate", [
    ("events.jsonl", lambda rows: rows[0].update(content="Different observation")),
    ("memory_ops.jsonl", lambda rows: rows[0].update(arg="197")),
    ("extractions.jsonl", lambda rows: rows[0]["extraction"].update(constraints=[{"id": "different"}])),
    ("retrievals.jsonl", lambda rows: rows[0].update(context="Different retrieval")),
    ("submission_events.jsonl", lambda rows: rows[0].update(steps_used=3)),
    ("predictions.jsonl", lambda rows: rows[0].update(final_plan="Different actual action")),
    ("traces.jsonl", lambda rows: rows[0]["result"]["scratchpad"][0].update(thought="Different thought")),
    ("traces.jsonl", lambda rows: rows[0]["result"]["structured_tool_results"][0].update(formatted_result="Different tool result")),
    ("session_completions.jsonl", lambda rows: rows[0]["row"]["memory_stats"].update(record_count=2)),
])
def test_prefix_comparison_rejects_any_causal_state_or_action_difference(prefix_pair, filename, mutate):
    parent, child = prefix_pair
    rows = json_lines(child / filename)
    mutate(rows)
    write_lines(child / filename, rows)
    with pytest.raises((ValueError, RuntimeError, AssertionError)):
        run.compare_prefix(parent, child)


def test_prefix_comparison_does_not_drop_a_tool_row_field_named_latency_seconds(prefix_pair):
    parent, child = prefix_pair
    for directory, value in ((parent, 1), (child, 2)):
        rows = json_lines(directory / "traces.jsonl")
        rows[0]["result"]["structured_tool_results"][0]["displayed_rows"][0]["latency_seconds"] = value
        write_lines(directory / "traces.jsonl", rows)
    with pytest.raises((ValueError, RuntimeError, AssertionError)):
        run.compare_prefix(parent, child)


def test_prefix_comparison_includes_partial_session_snapshot(prefix_pair):
    parent, child = prefix_pair
    snapshot_path = child / "snapshots/E2/69.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    snapshot["decisions"][0]["version"] = 2
    write_json(snapshot_path, snapshot)
    with pytest.raises((ValueError, RuntimeError, AssertionError)):
        run.compare_prefix(parent, child)


def test_prefix_comparison_rejects_missing_or_extra_prefix_rows(prefix_pair):
    parent, child = prefix_pair
    rows = json_lines(child / "retrievals.jsonl")
    rows.append(copy.deepcopy(rows[0]))
    write_lines(child / "retrievals.jsonl", rows)
    with pytest.raises((ValueError, RuntimeError, AssertionError)):
        run.compare_prefix(parent, child)


@pytest.mark.parametrize("scope,round_idx,expected", [("E2:69", 7, 29), ("E2:69", 6, 30),
                                                    ("E1:69", 7, 30), ("E2:120", 7, 30)])
def test_recovery_planner_preserves_original_failed_action_allowance(monkeypatch, scope, round_idx, expected):
    def fake_turn(self, request, memory_context, base_person):
        return {"total_steps": 2, "parent_model_requests": 2, "final_plan": "offline plan"}

    monkeypatch.setattr(PlannerWorker, "run_turn", fake_turn)
    planner = run.RecoveryPlanner(ROOT / "external/MemoryArena", ROOT / "data/tool_db", SimpleNamespace(group_id=scope),
                                  model="qwen/qwen3.6-27b", temperature=0, max_steps=30, causal_round=round_idx)
    assert planner.max_steps == expected
    result = planner.run_turn({}, "offline", {})
    charged = 1 if expected == 29 else 0
    assert result["total_steps"] == 2 + charged
    assert result["parent_model_requests"] == 2 + charged


def test_official_scoring_starts_only_after_complete_original_grid(monkeypatch, fake_parent, destination):
    protocol = json.loads((ROOT / "runs" / PARENT_ID / "protocol.json").read_text(encoding="utf-8"))
    fake_parent["protocol"] = protocol
    fake_parent["directory"] = destination / "invented-parent-sessions"
    prior_timings = {(item["group_id"], item["variant"]): 100.0 + index
                     for index, item in enumerate(protocol["schedule"][:6])}
    write_lines(fake_parent["directory"] / "session_completions.jsonl", [
        {**item, "row": {"wall_clock_seconds": prior_timings[(item["group_id"], item["variant"])],
                          "usage": [{"request_status": "completed", "latency_seconds": 20., "cost_rub": .25}]}}
        for item in protocol["schedule"][:6]])
    order = []
    scored = []

    class ScriptedSession:
        def __init__(self, root, run_dir, variant, group_id, config, *args):
            self.variant, self.group_id = variant, group_id
            assert config["agent"]["stream"] is False
            assert config["agent"]["max_agent_visible_memory_tokens"] == 32768
            assert config["agent"]["max_completion_tokens"] == 16384

        def run(self):
            order.append({"group_id": self.group_id, "variant": self.variant})
            return [{"group_id": self.group_id, "variant": self.variant}], {
                "group_id": self.group_id, "variant": self.variant,
                "base_template_id": "offline-family-" + self.group_id, "travelers": [], "usage": [],
                "wall_clock_seconds": .25}

    def scorer(upstream, hidden, predictions, groups, **kwargs):
        assert len(order) == 14
        assert len(json_lines(destination / "session_completions.jsonl")) == 14
        assert (destination / "planning_complete.json").exists()
        assert len(predictions) == 7
        scored.append(True)
        return {"per_group": [{"group_id": group, "base_template_id": "offline-family-" + group,
                               "official_metrics": {"SR": 0, "PS": 0, "SPS": 0}} for group in groups]}

    monkeypatch.setattr(run.frozen, "GroupSession", ScriptedSession)
    monkeypatch.setattr(run, "score_official", scorer)
    readiness = {"provenance": {"dataset_sha256": "offline-fixture-sha"}}
    result = run.execute_grid(ROOT, destination, fake_parent, None, readiness, SimpleNamespace(live=True))
    assert order == protocol["schedule"]
    assert len(result) == 14
    assert len(scored) == 2
    original_failed = [usage for row in result for usage in row["usage"] if usage.get("inherited_failed_dispatch")]
    assert len(original_failed) == 1
    assert original_failed[0]["cost_rub"] is None
    assert original_failed[0]["new_billable"] is False
    for row in result:
        key = (row["group_id"], row["variant"])
        if key in prior_timings:
            assert row["wall_clock_seconds"] == prior_timings[key]
            assert row["reconstruction_wall_clock_seconds"] == .25
            assert all(item["reused"] for item in row["usage"])
        elif key == ("69", "E2"):
            assert row["wall_clock_seconds"] is None
            assert row["reconstruction_plus_new_wall_clock_seconds"] == .25


def test_partial_or_failed_grid_cannot_open_the_scorer(monkeypatch, fake_parent, destination):
    protocol = json.loads((ROOT / "runs" / PARENT_ID / "protocol.json").read_text(encoding="utf-8"))
    fake_parent["protocol"] = protocol
    seen = []

    class ScriptedSession:
        def __init__(self, root, run_dir, variant, group_id, config, *args):
            self.variant, self.group_id = variant, group_id

        def run(self):
            seen.append((self.group_id, self.variant))
            if len(seen) == 7:
                raise ProviderError("Invented ambiguous seventh-session interruption", request_sent=True)
            return [], {"group_id": self.group_id, "variant": self.variant,
                        "base_template_id": "offline-family-" + self.group_id, "travelers": [], "usage": []}

    monkeypatch.setattr(run.frozen, "GroupSession", ScriptedSession)
    monkeypatch.setattr(run, "score_official", unexpected_transport)
    with pytest.raises(ProviderError, match="seventh-session"):
        run.execute_grid(ROOT, destination, fake_parent, None, {}, SimpleNamespace(live=True))
    assert len(seen) == 7
    assert len(json_lines(destination / "session_completions.jsonl")) == 6
    assert not (destination / "planning_complete.json").exists()
    assert not (destination / "results.jsonl").exists()


def build_historical_session(destination, variant):
    """Use native seed inference, fixed constraint prefixes, and real closures."""
    backend = create_backend(variant, {"token_budget": 32768})
    backend.reset("120")
    for name in ("Leslie", "Olivia"):
        for slot in ("lunch", "dinner"):
            ref = f"decision:{name}:day1:{slot}"
            backend.decisions[ref] = [{"decision_id": ref, "version": 1, "created_at": 0,
                                       "status": "COMMITTED", "traveler": name, "day": 1,
                                       "slot": slot, "value": name + " Cafe", "attribute": "value"}]
            backend.visible_index.add(ref)
    for ref, dependencies in (("constraint:a", ["constraint:z"]), ("constraint:z", [])):
        backend.constraints[ref] = [{"constraint_id": ref, "version": 1, "created_at": 0,
                                     "status": "ACTIVE", "scope": {"group": "120", "traveler": "NewTraveler"},
                                     "subject": {"traveler": "NewTraveler", "day": 1, "slot": "lunch"},
                                     "operator": "EQUALS", "target": {"value": "Cafe"},
                                     "kind": "LOCAL", "depends_on": dependencies, "source_refs": []}]
        backend.visible_index.add(ref)
    actual_request = {"group_id": "120", "traveler": "NewTraveler", "step": 0,
                      "query": "Keep the choices of Leslie and Olivia."}
    native_names = list(backend._named_travelers(actual_request["query"]))
    desired_names = list(reversed(native_names))
    expected_backend = copy.deepcopy(backend)
    expected_backend._named_travelers = lambda query: desired_names.copy()
    expected_context = expected_backend.begin_subtask(actual_request)
    original_seeds = backend._seed_ids(actual_request, 0)

    class Session:
        group_id = "120"

        def __init__(self):
            self.variant = variant
            self.backend = backend
            self.begin_calls = []
            self.actual_seeds = None
            self.failure = None

        def begin(self, current_request):
            self.begin_calls.append(copy.deepcopy(current_request))
            if self.failure == "before_seeds":
                raise ValueError("Invented begin failure before seed computation")
            current_seed_method = self.backend._seed_ids
            def capture_seeds(seed_request, step):
                seeds = current_seed_method(seed_request, step)
                self.actual_seeds = seeds.copy()
                return seeds
            self.backend._seed_ids = capture_seeds
            try:
                result = self.backend.begin_subtask(current_request)
            finally:
                self.backend._seed_ids = current_seed_method
            if self.failure == "after_seeds":
                raise ValueError("Invented begin failure after seed computation")
            return result

    session = Session()
    client = SimpleNamespace(destination=destination, retrieval_cursor=0)
    parent = {"retrievals": [{"variant": variant, "group_id": "120", "traveler": "NewTraveler",
                               "required_closure_ids": expected_context["required_closure_ids"]}]}
    return session, parent, client, actual_request, original_seeds, expected_backend, expected_context


@pytest.fixture(params=["E1", "E2"])
def historical_session(destination, request):
    return build_historical_session(destination, request.param)


def test_historical_names_only_permute_native_seed_multiset(historical_session):
    session, parent, client, request, original, expected_backend, expected_context = historical_session
    before_parent = copy.deepcopy(parent)
    native_seed_method = session.backend._seed_ids
    native_named_method = session.backend._named_travelers
    assert run.install_historical_seed_order(session, parent, client) is session
    result = session.begin(request)
    assert Counter(session.actual_seeds) == Counter(original)
    assert session.actual_seeds[:2] == ["constraint:a", "constraint:z"]
    assert result["required_closure_ids"] == expected_context["required_closure_ids"]
    assert result["context"] == expected_context["context"]
    assert client.retrieval_cursor == 1
    assert session.backend._seed_ids == native_seed_method
    assert session.backend._named_travelers == native_named_method
    assert parent == before_parent
    assert len(json_lines(client.destination / "historical_seed_order.jsonl")) == 1


def test_historical_ordering_stops_when_parent_retrieval_prefix_is_exhausted(historical_session):
    session, parent, client, request, original, expected_backend, expected_context = historical_session
    native_begin_backend = copy.deepcopy(session.backend)
    run.install_historical_seed_order(session, parent, client)
    session.begin(request)
    native_result = native_begin_backend.begin_subtask(request)
    after_prefix = session.begin(request)
    assert after_prefix["required_closure_ids"] == native_result["required_closure_ids"]
    assert session.actual_seeds == original
    assert client.retrieval_cursor == 1
    assert len(json_lines(client.destination / "historical_seed_order.jsonl")) == 1


@pytest.mark.parametrize("field,value", [("variant", "other-variant"), ("group_id", "197"), ("traveler", "Leslie")])
def test_wrong_historical_traversal_halts_before_seed_computation(historical_session, field, value):
    session, parent, client, request, original, expected_backend, expected_context = historical_session
    parent["retrievals"][0][field] = value
    native_seed_method = session.backend._seed_ids
    native_named_method = session.backend._named_travelers
    run.install_historical_seed_order(session, parent, client)
    with pytest.raises(run.ReplayMismatch, match="sequence differs"):
        session.begin(request)
    assert session.begin_calls == []
    assert session.actual_seeds is None
    assert client.retrieval_cursor == 0
    assert session.backend._seed_ids == native_seed_method
    assert session.backend._named_travelers == native_named_method
    assert not (client.destination / "historical_seed_order.jsonl").exists()


@pytest.mark.parametrize("failure", ["before_seeds", "after_seeds", "resolver"])
def test_historical_seed_and_named_methods_restored_in_finally(historical_session, failure):
    session, parent, client, request, original, expected_backend, expected_context = historical_session
    native_seed_method = session.backend._seed_ids
    native_named_method = session.backend._named_travelers
    if failure == "resolver":
        def broken_resolve(*args, **kwargs):
            raise ValueError("Invented resolver failure")
        session.backend._resolve = broken_resolve
    else:
        session.failure = failure
    run.install_historical_seed_order(session, parent, client)
    with pytest.raises(ValueError, match="Invented"):
        session.begin(request)
    assert session.backend._seed_ids == native_seed_method
    assert session.backend._named_travelers == native_named_method
    assert client.retrieval_cursor == 0


def test_historical_candidate_search_has_only_native_read_effects(historical_session):
    session, parent, client, request, original, expected_backend, expected_context = historical_session
    run.install_historical_seed_order(session, parent, client)
    result = session.begin(request)
    assert result == expected_context
    assert session.backend.snapshot() == expected_backend.snapshot()
    assert session.backend._metrics == expected_backend._metrics
    assert session.backend._diagnostics == expected_backend._diagnostics


def test_historical_trial_closures_restore_diagnostic_metric_counts(historical_session):
    session, parent, client, request, original, expected_backend, expected_context = historical_session
    # Real closure errors update both diagnostics and the metrics Counter. An
    # order-search trial must leave neither extra warnings nor extra counts.
    session.backend.constraints["constraint:a"][0]["depends_on"].append("decision:Missing:day1:lunch:price")
    expected_backend = copy.deepcopy(session.backend)
    desired_names = list(reversed(list(session.backend._named_travelers(request["query"]))))
    expected_backend._named_travelers = lambda query: desired_names.copy()
    expected_context = expected_backend.begin_subtask(request)
    parent["retrievals"][0]["required_closure_ids"] = expected_context["required_closure_ids"]
    run.install_historical_seed_order(session, parent, client)
    result = session.begin(request)
    assert result == expected_context
    assert session.backend._diagnostics == expected_backend._diagnostics
    assert session.backend._metrics == expected_backend._metrics


def test_impossible_historical_hint_is_rejected_instead_of_added_or_ignored(historical_session):
    session, parent, client, request, original, expected_backend, expected_context = historical_session
    parent["retrievals"][0]["required_closure_ids"].append("never-a-seed@v77")
    native_seed_method = session.backend._seed_ids
    native_named_method = session.backend._named_travelers
    before = copy.deepcopy(session.backend.snapshot())
    run.install_historical_seed_order(session, parent, client)
    with pytest.raises(run.ReplayMismatch):
        session.begin(request)
    assert client.retrieval_cursor == 0
    assert session.backend._seed_ids == native_seed_method
    assert session.backend._named_travelers == native_named_method
    assert session.backend.decisions == {row["decision_id"]: [row] for row in before["decisions"]}
    assert all("never-a-seed" not in key for key in session.backend.decisions)


def test_historical_fixed_constraint_prefix_is_never_sorted_by_closure_order(historical_session):
    session, parent, client, request, original, expected_backend, expected_context = historical_session
    assert original[:2] == ["constraint:a", "constraint:z"]
    if session.variant == "E2":
        assert expected_context["required_closure_ids"][:2] == ["constraint:z@v1", "constraint:a@v1"]
    run.install_historical_seed_order(session, parent, client)
    session.begin(request)
    assert session.actual_seeds[:2] == original[:2]
    assert Counter(session.actual_seeds) == Counter(original)


def test_no_historical_retrievals_means_no_wrapper_or_hint_journal(historical_session):
    session, parent, client, request, original, expected_backend, expected_context = historical_session
    parent["retrievals"] = []
    original_begin = session.begin
    assert run.install_historical_seed_order(session, parent, client) is session
    assert session.begin == original_begin
    session.begin(request)
    assert session.actual_seeds == original
    assert client.retrieval_cursor == 0
    assert not (client.destination / "historical_seed_order.jsonl").exists()


@pytest.fixture
def lineage_root(destination):
    root = destination / "isolated-lineage-workspace"
    (root / "runs").mkdir(parents=True)
    return root


def make_lineage_child(root, name="custom-unprefixed-child", *, status="failed", attempts=0):
    child = root / "runs" / name
    child.mkdir(parents=True)
    write_json(child / "manifest.json", {"run_id": name, "parent_run_id": PARENT_ID,
                                        "status": status, "usage": {"new_request_attempts": attempts}})
    write_lines(child / "usage.jsonl", [
        {"group_id": "E1:120", "operation": "memory_reset", "model_call": False,
         "input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "cost_rub": 0},
        {"group_id": "E1:120", "traveler": "OfflineTraveler", "operation": "agent",
         "request_status": "completed", "model": "qwen/qwen3.6-27b", "request_id": "offline-reused",
         "reused": True, "new_billable": False, "input_tokens": 2, "output_tokens": 3, "cost_rub": .25}])
    write_lines(child / "request_journal.jsonl", [
        {"event": "INTENT_ONLY_NOT_DISPATCH_PROOF", "attempt_id": "request-00001", "group_id": "E1:120"},
        {"event": "RESPONSE_ACCEPTED", "attempt_id": "request-00001", "group_id": "E1:120"}])
    return child


def test_unused_lineage_accepts_no_child_and_ignores_unrelated_runs(lineage_root):
    unrelated = make_lineage_child(lineage_root, "unrelated-paid-research", status="complete", attempts=9)
    manifest = json.loads((unrelated / "manifest.json").read_text(encoding="utf-8"))
    manifest["parent_run_id"] = "different-parent-study"
    write_json(unrelated / "manifest.json", manifest)
    run.verify_unused_lineage(lineage_root, PARENT_ID)


@pytest.mark.parametrize("status", ["failed", "interrupted"])
def test_lineage_allows_only_terminal_verified_cache_only_children(lineage_root, status):
    child = make_lineage_child(lineage_root, status=status)
    before = {path.name: file_hash(path) for path in child.iterdir() if path.is_file()}
    run.verify_unused_lineage(lineage_root, PARENT_ID)
    assert before == {path.name: file_hash(path) for path in child.iterdir() if path.is_file()}


@pytest.mark.parametrize("status", ["complete", "running", "replaying_prefix", "continuing_after_verified_boundary", "unknown"])
def test_terminal_or_active_live_lineage_cannot_be_replayed_even_with_zero_attempt_claim(lineage_root, status):
    make_lineage_child(lineage_root, status=status, attempts=0)
    with pytest.raises((ValueError, RuntimeError)):
        run.verify_unused_lineage(lineage_root, PARENT_ID)


@pytest.mark.parametrize("attempts", [1, None, "0", True, -1])
def test_lineage_rejects_positive_missing_or_invalid_new_attempt_accounting(lineage_root, attempts):
    child = make_lineage_child(lineage_root, attempts=attempts)
    if attempts is None:
        manifest = json.loads((child / "manifest.json").read_text(encoding="utf-8"))
        manifest["usage"].pop("new_request_attempts")
        write_json(child / "manifest.json", manifest)
    with pytest.raises((ValueError, RuntimeError)):
        run.verify_unused_lineage(lineage_root, PARENT_ID)


@pytest.mark.parametrize("flags", [
    {"reused": False, "new_billable": True}, {"reused": True, "new_billable": True},
    {"reused": False, "new_billable": False}, {"reused": True}, {"new_billable": False}, {},
])
def test_lineage_scans_all_child_model_usage_not_only_summary_or_directory_prefix(lineage_root, flags):
    child = make_lineage_child(lineage_root, "a-user-chosen-run-name")
    usage = json_lines(child / "usage.jsonl")
    for key in ("reused", "new_billable"):
        usage[-1].pop(key)
    usage[-1].update(flags)
    write_lines(child / "usage.jsonl", usage)
    with pytest.raises((ValueError, RuntimeError)):
        run.verify_unused_lineage(lineage_root, PARENT_ID)


@pytest.mark.parametrize("receipt", [
    {"model": "qwen/qwen3.6-27b", "operation": "agent", "new_billable": True, "reused": False},
    {"operation": "agent", "model_call": True, "cost_rub": .25},
])
def test_lineage_missing_request_status_does_not_hide_an_explicit_model_receipt(lineage_root, receipt):
    child = make_lineage_child(lineage_root)
    usage = json_lines(child / "usage.jsonl")
    usage.append(receipt)
    write_lines(child / "usage.jsonl", usage)
    with pytest.raises((ValueError, RuntimeError)):
        run.verify_unused_lineage(lineage_root, PARENT_ID)


def test_replacement_marker_blocks_even_when_zero_attempt_manifest_is_stale(lineage_root):
    child = make_lineage_child(lineage_root)
    write_json(child / "replacement_attempt.json", {"parent_run_id": PARENT_ID,
                                                    "parent_attempt_id": "request-00156"})
    with pytest.raises((ValueError, RuntimeError)):
        run.verify_unused_lineage(lineage_root, PARENT_ID)


@pytest.mark.parametrize("sent", [True, None])
def test_sent_or_ambiguous_failed_request_blocks_another_paid_continuation(lineage_root, sent):
    child = make_lineage_child(lineage_root)
    journal = json_lines(child / "request_journal.jsonl")
    journal.append({"event": "REQUEST_FAILED", "attempt_id": "request-00156", "request_sent": sent})
    write_lines(child / "request_journal.jsonl", journal)
    with pytest.raises((ValueError, RuntimeError)):
        run.verify_unused_lineage(lineage_root, PARENT_ID)


def test_new_dispatch_intent_after_inherited_prefix_blocks_repurchase_even_without_receipt(lineage_root):
    child = make_lineage_child(lineage_root)
    write_lines(child / "request_journal.jsonl", [
        {"event": "INTENT_ONLY_NOT_DISPATCH_PROOF", "attempt_id": f"request-{index:05d}"}
        for index in range(1, 157)])
    with pytest.raises((ValueError, RuntimeError)):
        run.verify_unused_lineage(lineage_root, PARENT_ID)


@pytest.mark.parametrize("bad_body", ["", "{truncated", "null", "[]"])
def test_known_recovery_directory_with_unreadable_manifest_fails_closed(lineage_root, bad_body):
    child = lineage_root / "runs/cross-family-recovery-unreadable"
    child.mkdir(parents=True)
    (child / "manifest.json").write_text(bad_body, encoding="utf-8")
    with pytest.raises((ValueError, RuntimeError, TypeError)):
        run.verify_unused_lineage(lineage_root, PARENT_ID)


def test_known_recovery_directory_with_missing_manifest_fails_closed(lineage_root):
    (lineage_root / "runs/cross-family-recovery-without-manifest").mkdir()
    with pytest.raises((ValueError, RuntimeError)):
        run.verify_unused_lineage(lineage_root, PARENT_ID)


def test_current_child_can_be_excluded_only_for_its_verified_cache_boundary(lineage_root):
    current = make_lineage_child(lineage_root, "new-current-child", status="replaying_prefix")
    run.verify_unused_lineage(lineage_root, PARENT_ID, exclude_child=current)
    make_lineage_child(lineage_root, "previous-paid-child", attempts=1)
    with pytest.raises((ValueError, RuntimeError)):
        run.verify_unused_lineage(lineage_root, PARENT_ID, exclude_child=current)


def test_excluding_current_child_cannot_hide_an_existing_replacement_marker(lineage_root):
    current = make_lineage_child(lineage_root, "new-current-child", status="replaying_prefix")
    write_json(current / "replacement_attempt.json", {"parent_run_id": PARENT_ID})
    with pytest.raises((ValueError, RuntimeError)):
        run.verify_unused_lineage(lineage_root, PARENT_ID, exclude_child=current)


def test_existing_lineage_claim_blocks_regardless_of_child_status_or_exclusion(lineage_root):
    current = make_lineage_child(lineage_root, "current-child", status="replaying_prefix")
    write_json(lineage_root / "reports/cross_family_recovery/live_lineage_claim.json",
               {"parent_run_id": PARENT_ID, "child_run_id": current.name})
    with pytest.raises((ValueError, RuntimeError)):
        run.verify_unused_lineage(lineage_root, PARENT_ID, exclude_child=current)


@pytest.mark.parametrize("body", ["{partial", "null", "[]", '{"parent_run_id":"a-different-parent","child_run_id":"other"}'])
def test_corrupt_or_wrong_parent_claim_cannot_silently_be_replaced(lineage_root, body):
    claim = lineage_root / "reports/cross_family_recovery/live_lineage_claim.json"
    claim.parent.mkdir(parents=True)
    claim.write_text(body, encoding="utf-8")
    before = file_hash(claim)
    with pytest.raises((ValueError, RuntimeError, TypeError)):
        run.verify_unused_lineage(lineage_root, PARENT_ID)
    assert file_hash(claim) == before


def test_lineage_claim_creation_is_exclusive_and_preserves_existing_receipt(lineage_root):
    first = lineage_root / "runs/first-deliberate-continuation"
    second = lineage_root / "runs/second-duplicate-continuation"
    first.mkdir()
    second.mkdir()
    run.claim_lineage(lineage_root, PARENT_ID, first)
    claim = lineage_root / "reports/cross_family_recovery/live_lineage_claim.json"
    receipt = json.loads(claim.read_text(encoding="utf-8"))
    assert receipt["parent_run_id"] == PARENT_ID
    assert receipt["child_run_id"] == first.name
    before = file_hash(claim)
    with pytest.raises((FileExistsError, ValueError, RuntimeError)):
        run.claim_lineage(lineage_root, PARENT_ID, second)
    assert file_hash(claim) == before


def test_seven_named_travelers_use_one_verified_hint_without_reordering_constraint_prefix(destination):
    session, parent, client, request, _, _, _ = build_historical_session(destination, "E1")
    for name in ("Anchor", "Betty", "Mark", "Jasmine", "Audrey"):
        for slot in ("lunch", "dinner"):
            ref = f"decision:{name}:day1:{slot}"
            session.backend.decisions[ref] = [{"decision_id": ref, "version": 1, "created_at": 0,
                                               "status": "COMMITTED", "traveler": name, "day": 1,
                                               "slot": slot, "value": name + " Cafe", "attribute": "value"}]
            session.backend.visible_index.add(ref)
    request["query"] = "Keep Leslie, Olivia, Anchor, Betty, Mark, Jasmine, and Audrey choices."
    request["public_anchor_travelers"] = ["Anchor"]
    names = list(session.backend._named_travelers(request["query"]))
    assert len(names) == 7
    effective_names = [name for name in names if name != "Anchor"]
    desired_names = list(reversed(effective_names)) + ["Anchor"]
    expected_backend = copy.deepcopy(session.backend)
    expected_backend._named_travelers = lambda query: desired_names.copy()
    expected_context = expected_backend.begin_subtask(request)
    parent["retrievals"][0]["required_closure_ids"] = expected_context["required_closure_ids"]
    original = session.backend._seed_ids(request, 0)
    run.install_historical_seed_order(session, parent, client)
    result = session.begin(request)
    assert result == expected_context
    assert session.actual_seeds[:2] == ["constraint:a", "constraint:z"]
    assert Counter(session.actual_seeds) == Counter(original)
    assert not any(ref.startswith("decision:Anchor:") for ref in session.actual_seeds)
    receipt = json_lines(client.destination / "historical_seed_order.jsonl")[-1]
    assert receipt["trial_unique_permutations"] == 1
    assert receipt["required_closure_order_verified"] is True
    assert session.backend.snapshot() == expected_backend.snapshot()


def test_misleading_e2_first_appearance_hint_falls_back_to_exact_native_closure(destination):
    session, parent, client, request, _, _, _ = build_historical_session(destination, "E2")
    native_names = list(session.backend._named_travelers(request["query"]))
    first, second = list(reversed(native_names))
    # The first traveler's dinner depends on the second traveler's lunch.
    # Closure therefore names the second person first, while reproducing its
    # exact sequence still requires the first person's seed block first.
    session.backend.decisions[f"decision:{first}:day1:dinner"][0]["depends_on"] = [f"decision:{second}:day1:lunch"]
    expected_backend = copy.deepcopy(session.backend)
    expected_backend._named_travelers = lambda query: [first, second]
    expected_context = expected_backend.begin_subtask(request)
    parent["retrievals"][0]["required_closure_ids"] = expected_context["required_closure_ids"]
    actual_first_person_ref = next(ref for ref in expected_context["required_closure_ids"] if ref.startswith("decision:"))
    assert actual_first_person_ref.startswith(f"decision:{second}:")
    original = session.backend._seed_ids(request, 0)
    run.install_historical_seed_order(session, parent, client)
    result = session.begin(request)
    assert result == expected_context
    assert session.actual_seeds[:2] == ["constraint:a", "constraint:z"]
    assert Counter(session.actual_seeds) == Counter(original)
    receipt = json_lines(client.destination / "historical_seed_order.jsonl")[-1]
    assert receipt["trial_unique_permutations"] == 2
    assert receipt["named_set_order"] == [first, second]
    assert session.backend.snapshot() == expected_backend.snapshot()
    assert session.backend._metrics == expected_backend._metrics
    assert session.backend._diagnostics == expected_backend._diagnostics
