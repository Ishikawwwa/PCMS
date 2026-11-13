"""Independent checks for the second audited continuation.

Invented model receipts exercise accounting and recovery gates without network,
credentials, task scoring, or hidden answers. Real lineage evidence is read only.
"""
from __future__ import annotations

import builtins
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
from constraint_memory.polza import ProviderError
from experiments.cross_family_recovery_second import run


ROOT = Path(__file__).resolve().parents[2]
PARENT_ID = "cross-family-primary-20261015T192052114101Z"
CHILD_ID = "cross-family-recovery-20261016T144804072163Z"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def write_lines(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(value, allow_nan=False) + "\n" for value in values), encoding="utf-8")


@pytest.fixture
def destination():
    result = ROOT / "tmp/cross-family-second-independent-tests" / uuid.uuid4().hex
    result.mkdir(parents=True)
    return result


@pytest.fixture
def fake_lineage():
    config = json.loads((ROOT / "runs" / PARENT_ID / "manifest.json").read_text(encoding="utf-8"))["config"]
    accepted = []
    for index, operation in enumerate(("agent", "extraction"), 1):
        intent = {"attempt_id": f"request-{index:05d}", "event": "INTENT_ONLY_NOT_DISPATCH_PROOF",
                  "group_id": "E1:120", "traveler": "OfflineTraveler", "operation": operation,
                  "messages": [{"role": "user", "content": f"Invented offline {operation}."}],
                  "tools": None, "max_tokens_override": None, "json_output": False, "response_schema": None}
        raw = {"id": f"offline-old-{index}", "model": config["agent"]["model"], "provider": "offline-provider",
               "choices": [{"message": {"role": "assistant", "content": "{}"}, "finish_reason": "stop"}],
               "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5, "cost_rub": .25}}
        usage = {"group_id": intent["group_id"], "traveler": intent["traveler"], "operation": operation,
                 "request_id": raw["id"], "request_status": "completed", "response_accepted": True,
                 "model_identity_validation": "EXACT_MATCH", "model": raw["model"], "input_tokens": 2,
                 "output_tokens": 3, "total_tokens": 5, "cost_rub": .25, "currency": "RUB", "latency_seconds": 23.}
        accepted.append({"intent": intent, "raw_response": raw, "usage": usage})
    failed_intent = {"attempt_id": "request-00003", "event": "INTENT_ONLY_NOT_DISPATCH_PROOF",
                     "group_id": "E2:69", "traveler": "Olivia", "operation": "agent",
                     "messages": [{"role": "user", "content": "Invented exact continuation boundary."}],
                     "tools": None, "max_tokens_override": None, "json_output": False, "response_schema": None}
    failure = {"group_id": "E2:69", "traveler": "Olivia", "operation": "agent", "request_sent": True,
               "request_status": "failed", "input_tokens": None, "output_tokens": None,
               "total_tokens": None, "cost_rub": None, "billing_status": "unknown_after_failed_request"}
    first_failure = {**failure, "request_id": "offline-stream-rejected"}
    second_failure = {**failure, "request_id": None, "error": "Invented RemoteDisconnected"}
    return {"run_id": PARENT_ID, "directory": ROOT / "runs" / PARENT_ID,
            "manifest": {"run_id": PARENT_ID, "config": config, "status": "failed"},
            "protocol": {"config": config}, "accepted": accepted, "failed_intent": failed_intent,
            "failed_usage": first_failure, "extra_failed_usage": second_failure,
            "child_run_id": CHILD_ID,
            "child": {"run_id": CHILD_ID, "directory": ROOT / "runs" / CHILD_ID,
                      "manifest": {"run_id": CHILD_ID, "parent_run_id": PARENT_ID, "status": "failed"}}}


def no_network(*args, **kwargs):
    pytest.fail("No new request is permitted by this recovery test")


def client_for(parent, destination, transport=no_network, *, live=False, gate_callback=None, **overrides):
    agent = parent["manifest"]["config"]["agent"]
    kwargs = {"api_key": "offline-only-not-a-credential", "model": agent["model"],
              "temperature": agent["temperature"], "max_tokens": agent["max_completion_tokens"],
              "extraction_max_tokens": agent["max_extraction_tokens"], "seed": 20261013,
              "provider": copy.deepcopy(agent["routing"]), "reasoning": copy.deepcopy(agent["reasoning"]),
              "extraction_reasoning": copy.deepcopy(agent["extraction_reasoning"]), "stream": False,
              "max_total_tokens": math.inf, "max_cost_rub": None, "transport": transport,
              "usage_path": destination / "usage.jsonl"}
    kwargs.update(overrides)
    return run.SecondRecoveryClient(parent, destination, live=live, gate_callback=gate_callback, **kwargs)


def issue(client, intent):
    client.set_scope(intent["group_id"], intent["traveler"])
    return client.complete(copy.deepcopy(intent["messages"]), tools=copy.deepcopy(intent["tools"]),
                           operation=intent["operation"], max_tokens=intent["max_tokens_override"],
                           json_output=intent["json_output"], response_schema=copy.deepcopy(intent["response_schema"]))


def inherited_failure_rows(stats):
    assert isinstance(stats["inherited_failures"], list)
    return stats["inherited_failures"]


def test_second_replay_reuses_each_accepted_receipt_once_with_no_transport(fake_lineage, destination):
    before = copy.deepcopy(fake_lineage)
    with client_for(fake_lineage, destination) as client:
        for entry in fake_lineage["accepted"]:
            assert issue(client, entry["intent"]) == entry["raw_response"]
        assert len(client.records) == 2
        assert all(record["response_accepted"] and record["reused"] and not record["new_billable"] for record in client.records)
        assert all(record["total_tokens"] is not None for record in client.records)
    assert fake_lineage == before


def test_two_inherited_unknown_bills_remain_separate_and_cache_is_not_double_charged(fake_lineage, destination):
    with client_for(fake_lineage, destination) as client:
        for entry in fake_lineage["accepted"]:
            issue(client, entry["intent"])
        stats = client.get_usage_stats()
        assert stats["inherited_failed_attempts"] == 2
        assert stats["inherited_accepted_responses"] == 2
        assert stats["new_request_attempts"] == 0
        assert Decimal(stats["inherited_known_cost_rub_exact"]) == Decimal(".50")
        assert Decimal(stats["new_known_cost_rub_exact"]) == 0
        assert Decimal(stats["combined_known_cost_rub_exact"]) == Decimal(".50")
        assert stats["total_cost_rub"] is None
        failures = inherited_failure_rows(stats)
        assert len(failures) == 2
        assert {row["origin_run_id"] for row in failures} == {PARENT_ID, CHILD_ID}
        for row in failures:
            usage = row.get("usage", row)
            assert usage["cost_rub"] is None
            assert usage["total_tokens"] is None


@pytest.mark.parametrize("scope,round_idx,steps,charged", [
    ("E2:69", 7, 28, 2), ("E2:69", 6, 30, 0), ("E1:69", 7, 30, 0), ("E2:120", 7, 30, 0),
])
def test_second_planner_keeps_both_failed_dispatches_inside_original_thirty(monkeypatch, scope, round_idx, steps, charged):
    def native_turn(self, request, memory_context, base_person):
        return {"total_steps": 2, "parent_model_requests": 2, "final_plan": "offline"}
    monkeypatch.setattr(PlannerWorker, "run_turn", native_turn)
    planner = run.SecondRecoveryPlanner(ROOT / "external/MemoryArena", ROOT / "data/tool_db",
                                        SimpleNamespace(group_id=scope), model="qwen/qwen3.6-27b",
                                        temperature=0, max_steps=30, causal_round=round_idx)
    assert planner.max_steps == steps
    result = planner.run_turn({}, "offline memory", {})
    assert result["total_steps"] == 2 + charged
    assert result["parent_model_requests"] == 2 + charged


def test_second_recovery_boundary_matches_before_gate_and_never_dispatches_in_offline_mode(fake_lineage, destination):
    events = []
    with client_for(fake_lineage, destination, gate_callback=lambda: events.append("gate")) as client:
        for entry in fake_lineage["accepted"]:
            issue(client, entry["intent"])
        assert events == []
        with pytest.raises(run.ReplayBoundaryReached) as failure:
            issue(client, fake_lineage["failed_intent"])
        assert failure.value.request_sent is False
        assert events == ["gate"]
        assert client.get_usage_stats()["new_request_attempts"] == 0


def test_second_recovery_wrong_boundary_halts_before_gate_or_api(fake_lineage, destination):
    events = []
    with client_for(fake_lineage, destination, live=True, gate_callback=lambda: events.append("gate")) as client:
        for entry in fake_lineage["accepted"]:
            issue(client, entry["intent"])
        bad = copy.deepcopy(fake_lineage["failed_intent"])
        bad["messages"][0]["content"] = "Different boundary"
        with pytest.raises((ValueError, RuntimeError)):
            issue(client, bad)
        assert events == []


def test_second_recovery_failed_proof_prevents_any_new_dispatch(fake_lineage, destination):
    def failed_gate():
        raise ValueError("Independent second-lineage proof failed")
    with client_for(fake_lineage, destination, live=True, gate_callback=failed_gate) as client:
        for entry in fake_lineage["accepted"]:
            issue(client, entry["intent"])
        with pytest.raises(ValueError, match="proof failed"):
            issue(client, fake_lineage["failed_intent"])


def test_one_new_second_attempt_follows_gate_and_adds_only_its_cost(fake_lineage, destination):
    events = []
    new = copy.deepcopy(fake_lineage["accepted"][0]["raw_response"])
    new["id"] = "offline-v2-new"
    def gate():
        events.append("gate")
    def transport(method, path, payload):
        assert events == ["gate"]
        events.append("api")
        assert method == "POST" and path == "/chat/completions"
        return copy.deepcopy(new)
    with client_for(fake_lineage, destination, transport, live=True, gate_callback=gate) as client:
        for entry in fake_lineage["accepted"]:
            issue(client, entry["intent"])
        assert issue(client, fake_lineage["failed_intent"])["id"] == "offline-v2-new"
        stats = client.get_usage_stats()
        assert stats["new_request_attempts"] == 1
        assert Decimal(stats["new_known_cost_rub_exact"]) == Decimal(".25")
        assert Decimal(stats["combined_known_cost_rub_exact"]) == Decimal(".75")
        assert stats["inherited_failed_attempts"] == 2
        assert stats["total_cost_rub"] is None
    assert events == ["gate", "api"]


def test_a_third_ambiguous_failure_is_not_retried_or_fabricated_as_zero(fake_lineage, destination):
    calls = []
    def transport(*args):
        calls.append(args)
        raise ProviderError("Invented third ambiguous failure", request_sent=True)
    with client_for(fake_lineage, destination, transport, live=True, gate_callback=lambda: None) as client:
        for entry in fake_lineage["accepted"]:
            issue(client, entry["intent"])
        with pytest.raises(ProviderError, match="third ambiguous"):
            issue(client, fake_lineage["failed_intent"])
        assert len(calls) == 1
        assert client.records[-1]["cost_rub"] is None
        assert client.records[-1]["total_tokens"] is None
        assert client.get_usage_stats()["total_cost_rub"] is None


@pytest.fixture
def lineage_root(destination):
    root = destination / "second-lineage-root"
    (root / "runs").mkdir(parents=True)
    write_json(root / "reports/cross_family_recovery/live_lineage_claim.json",
               {"parent_run_id": PARENT_ID, "child_run_id": CHILD_ID})
    return root


def second_child(root, name="custom-v2-child", *, status="failed", new_attempts=0):
    child = root / "runs" / name
    child.mkdir()
    write_json(child / "manifest.json", {"run_id": name, "parent_run_id": CHILD_ID,
                                        "status": status, "usage": {"new_request_attempts": new_attempts}})
    write_lines(child / "request_journal.jsonl", [{"event": "INTENT_ONLY_NOT_DISPATCH_PROOF", "attempt_id": "request-00001"},
                                                  {"event": "RESPONSE_ACCEPTED", "attempt_id": "request-00001"}])
    write_lines(child / "usage.jsonl", [{"operation": "agent", "request_status": "completed", "reused": True,
                                        "new_billable": False, "model": "qwen/qwen3.6-27b", "cost_rub": .25}])
    return child


def test_old_claim_stays_immutable_and_does_not_block_audited_second_lineage(lineage_root):
    old_claim = lineage_root / "reports/cross_family_recovery/live_lineage_claim.json"
    before = digest(old_claim)
    run.verify_second_lineage(lineage_root, CHILD_ID)
    assert digest(old_claim) == before


def test_cache_only_second_failure_can_be_reconstructed_without_removing_old_claim(lineage_root):
    second_child(lineage_root)
    old_claim = lineage_root / "reports/cross_family_recovery/live_lineage_claim.json"
    before = digest(old_claim)
    run.verify_second_lineage(lineage_root, CHILD_ID)
    assert digest(old_claim) == before


@pytest.mark.parametrize("status,attempts", [("complete", 0), ("running", 0), ("failed", 1), ("interrupted", None)])
def test_second_lineage_refuses_paid_or_ambiguous_custom_child(lineage_root, status, attempts):
    second_child(lineage_root, "not-a-standard-recovery-prefix", status=status, new_attempts=attempts)
    with pytest.raises((ValueError, RuntimeError)):
        run.verify_second_lineage(lineage_root, CHILD_ID)


@pytest.mark.parametrize("marker", ["replacement_attempt.json", "second_replacement_attempt.json"])
def test_second_replacement_marker_blocks_relaunch_despite_stale_zero_summary(lineage_root, marker):
    child = second_child(lineage_root)
    write_json(child / marker, {"parent_run_id": CHILD_ID})
    with pytest.raises((ValueError, RuntimeError)):
        run.verify_second_lineage(lineage_root, CHILD_ID)


def test_second_new_model_receipt_cannot_hide_in_zero_attempt_summary(lineage_root):
    child = second_child(lineage_root)
    write_lines(child / "usage.jsonl", [{"operation": "agent", "model_call": True, "cost_rub": .25}])
    with pytest.raises((ValueError, RuntimeError)):
        run.verify_second_lineage(lineage_root, CHILD_ID)


def test_current_second_child_exclusion_does_not_hide_previous_paid_second_child(lineage_root):
    current = second_child(lineage_root, "current-v2-child", status="replaying_prefix")
    run.verify_second_lineage(lineage_root, CHILD_ID, exclude_child=current)
    second_child(lineage_root, "previous-v2-paid-child", new_attempts=1)
    with pytest.raises((ValueError, RuntimeError)):
        run.verify_second_lineage(lineage_root, CHILD_ID, exclude_child=current)


def test_separate_second_claim_is_exclusive_and_preserves_first_claim(lineage_root):
    old_claim = lineage_root / "reports/cross_family_recovery/live_lineage_claim.json"
    old_before = digest(old_claim)
    first, duplicate = lineage_root / "runs/second-first", lineage_root / "runs/second-duplicate"
    first.mkdir()
    duplicate.mkdir()
    run.claim_second_lineage(lineage_root, CHILD_ID, first)
    claim = lineage_root / "reports/cross_family_recovery_second/live_lineage_claim.json"
    receipt = json.loads(claim.read_text(encoding="utf-8"))
    assert receipt["parent_run_id"] == CHILD_ID
    assert receipt["child_run_id"] == first.name
    before = digest(claim)
    with pytest.raises((ValueError, RuntimeError, FileExistsError)):
        run.claim_second_lineage(lineage_root, CHILD_ID, duplicate)
    assert digest(claim) == before
    assert digest(old_claim) == old_before
    with pytest.raises((ValueError, RuntimeError)):
        run.verify_second_lineage(lineage_root, CHILD_ID)


@pytest.mark.parametrize("body", ["{partial", "null", "[]", '{"parent_run_id":"wrong","child_run_id":"other"}'])
def test_corrupt_or_wrong_second_claim_fails_closed(lineage_root, body):
    claim = lineage_root / "reports/cross_family_recovery_second/live_lineage_claim.json"
    claim.parent.mkdir(parents=True)
    claim.write_text(body, encoding="utf-8")
    with pytest.raises((ValueError, RuntimeError, TypeError)):
        run.verify_second_lineage(lineage_root, CHILD_ID)


def test_actual_lineage_validation_is_offline_and_never_opens_credentials_or_hidden_answers(monkeypatch):
    original_open, original_path_open = builtins.open, Path.open
    def check(path):
        if isinstance(path, (str, Path)):
            value = str(path).replace("\\", "/").lower()
            assert "/data/hidden/" not in value
            assert "/data/audit_annotations/" not in value
            assert Path(path).name.lower() != ".env"
    def guarded_open(path, *args, **kwargs):
        check(path)
        return original_open(path, *args, **kwargs)
    def guarded_path_open(path, *args, **kwargs):
        check(path)
        return original_path_open(path, *args, **kwargs)
    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(Path, "open", guarded_path_open)
    monkeypatch.setattr(socket.socket, "connect", no_network)
    parent = run.read_lineage(ROOT)
    assert len(parent["accepted"]) == 155
    assert parent["failed_usage"]["cost_rub"] is None
    assert parent["extra_failed_usage"]["cost_rub"] is None
    assert parent["extra_failed_usage"]["request_id"] is None
    assert len({entry["raw_response"]["id"] for entry in parent["accepted"]}) == 155
    assert sum((Decimal(str(entry["usage"]["cost_rub"])) for entry in parent["accepted"]), Decimal(0)) == Decimal("294.18187364")


def test_actual_lineage_validation_preserves_original_child_and_first_claim():
    directories = [ROOT / "runs" / PARENT_ID, ROOT / "runs" / CHILD_ID]
    claim = ROOT / "reports/cross_family_recovery/live_lineage_claim.json"
    before = {(str(directory), str(path.relative_to(directory))): digest(path)
              for directory in directories for path in directory.rglob("*") if path.is_file()}
    old_claim = digest(claim)
    run.read_lineage(ROOT)
    after = {(str(directory), str(path.relative_to(directory))): digest(path)
             for directory in directories for path in directory.rglob("*") if path.is_file()}
    assert before == after
    assert digest(claim) == old_claim


def test_second_grid_scores_only_after_fourteen_sessions_and_inherits_both_failures_once(monkeypatch, fake_lineage, destination):
    protocol = json.loads((ROOT / "runs" / PARENT_ID / "protocol.json").read_text(encoding="utf-8"))
    fake_lineage["protocol"] = protocol
    fake_lineage["directory"] = destination / "invented-original-parent"
    prior_timings = {(item["group_id"], item["variant"]): 100. + index
                     for index, item in enumerate(protocol["schedule"][:6])}
    write_lines(fake_lineage["directory"] / "session_completions.jsonl", [
        {**item, "row": {"wall_clock_seconds": prior_timings[(item["group_id"], item["variant"])],
                          "usage": [{"request_status": "completed", "latency_seconds": 20., "cost_rub": .25}]}}
        for item in protocol["schedule"][:6]])
    before = copy.deepcopy(fake_lineage)
    order, scored = [], []

    class ScriptedSession:
        def __init__(self, root, run_dir, variant, group_id, config, *args):
            self.variant, self.group_id = variant, group_id
            assert config["agent"]["stream"] is True
            assert config["agent"]["max_agent_visible_memory_tokens"] == 32768
            assert config["agent"]["max_completion_tokens"] == 16384
            assert config["agent"]["max_steps_per_traveler"] == 30
        def run(self):
            order.append({"group_id": self.group_id, "variant": self.variant})
            return [{"group_id": self.group_id, "variant": self.variant}], {
                "group_id": self.group_id, "variant": self.variant, "base_template_id": "offline-family-" + self.group_id,
                "travelers": [], "wall_clock_seconds": .25,
                "usage": [{"operation": "memory_reset", "model_call": False, "latency_seconds": .01,
                           "input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "cost_rub": 0}]}

    def scorer(upstream, hidden, predictions, groups, **kwargs):
        assert len(order) == 14
        completions = [json.loads(line) for line in (destination / "session_completions.jsonl").read_text(encoding="utf-8").splitlines()]
        assert len(completions) == 14
        proof = json.loads((destination / "planning_complete.json").read_text(encoding="utf-8"))
        assert proof["parent_run_id"] == CHILD_ID
        assert len(predictions) == 7
        scored.append(True)
        return {"per_group": [{"group_id": group, "base_template_id": "offline-family-" + group,
                               "official_metrics": {"SR": 0, "PS": 0, "SPS": 0}} for group in groups]}

    monkeypatch.setattr(run.base.frozen, "GroupSession", ScriptedSession)
    monkeypatch.setattr(run.base, "score_official", scorer)
    rows = run.execute_grid(ROOT, destination, fake_lineage, None, {"provenance": {"dataset_sha256": "offline"}}, SimpleNamespace(live=True))
    assert order == protocol["schedule"]
    assert len(rows) == 14 and len(scored) == 2
    inherited = [(row["group_id"], row["variant"], usage) for row in rows for usage in row["usage"] if usage.get("inherited_failed_dispatch")]
    assert len(inherited) == 2
    assert all((group, variant) == ("69", "E2") for group, variant, usage in inherited)
    assert {usage["origin_run_id"] for group, variant, usage in inherited} == {PARENT_ID, CHILD_ID}
    assert all(usage["cost_rub"] is None and usage["new_billable"] is False for group, variant, usage in inherited)
    for row in rows:
        key = (row["group_id"], row["variant"])
        if key in prior_timings:
            assert row["wall_clock_seconds"] == prior_timings[key]
            assert row["reconstruction_wall_clock_seconds"] == .25
        elif key == ("69", "E2"):
            assert row["wall_clock_seconds"] is None
            assert row["reconstruction_plus_new_wall_clock_seconds"] == .25
    assert fake_lineage == before


def test_second_grid_stops_at_ambiguous_failure_before_scoring(monkeypatch, fake_lineage, destination):
    fake_lineage["protocol"] = json.loads((ROOT / "runs" / PARENT_ID / "protocol.json").read_text(encoding="utf-8"))
    seen = []
    class ScriptedSession:
        def __init__(self, root, run_dir, variant, group_id, config, *args):
            self.variant, self.group_id = variant, group_id
        def run(self):
            seen.append((self.group_id, self.variant))
            if len(seen) == 7:
                raise ProviderError("Invented second-grid sent interruption", request_sent=True)
            return [], {"group_id": self.group_id, "variant": self.variant,
                        "base_template_id": "offline-family-" + self.group_id, "travelers": [], "usage": []}
    monkeypatch.setattr(run.base.frozen, "GroupSession", ScriptedSession)
    monkeypatch.setattr(run.base, "score_official", no_network)
    with pytest.raises(ProviderError, match="second-grid"):
        run.execute_grid(ROOT, destination, fake_lineage, None, {}, SimpleNamespace(live=True))
    assert len(seen) == 7
    assert not (destination / "planning_complete.json").exists()
    assert not (destination / "results.jsonl").exists()


def test_second_replacement_marker_is_correct_before_dispatch_and_after_ambiguous_failure(fake_lineage, destination):
    observed = []
    marker_path = destination / "replacement_attempt.json"
    def assert_marker():
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        assert marker["parent_run_id"] == CHILD_ID
        assert marker["original_run_id"] == PARENT_ID
        assert marker["parent_request_id"] is None
        assert marker["original_request_id"] == fake_lineage["failed_usage"]["request_id"]
        assert marker["stream"] is True
        assert marker["wire_stream"] is True
        assert marker["native_serializer_stream"] is False
        assert marker["inherited_sent_failures"] == 2
        assert marker["remaining_new_actor_calls"] == 27
        return marker
    def transport(*args):
        # This callback is the first possible external dispatch: provenance
        # must already describe the second continuation at this instant.
        observed.append(assert_marker())
        raise ProviderError("Invented interruption after correct marker", request_sent=True)
    with client_for(fake_lineage, destination, transport, live=True, gate_callback=lambda: None) as client:
        for entry in fake_lineage["accepted"]:
            issue(client, entry["intent"])
        with pytest.raises(ProviderError, match="correct marker"):
            issue(client, fake_lineage["failed_intent"])
        assert len(observed) == 1
        assert assert_marker() == observed[0]
        assert client.get_usage_stats()["new_request_attempts"] == 1
        assert client.get_usage_stats()["total_cost_rub"] is None


def test_cached_and_offline_boundary_replay_never_create_replacement_markers(fake_lineage, destination):
    markers = [destination / name for name in ("replacement_attempt.json", "second_replacement_attempt.json")]
    with client_for(fake_lineage, destination, gate_callback=lambda: None) as client:
        for entry in fake_lineage["accepted"]:
            issue(client, entry["intent"])
            assert all(not path.exists() for path in markers)
        with pytest.raises(run.ReplayBoundaryReached):
            issue(client, fake_lineage["failed_intent"])
        assert all(not path.exists() for path in markers)
        assert client.get_usage_stats()["new_request_attempts"] == 0
