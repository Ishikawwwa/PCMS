"""Independent continuation invariants using invented offline receipts only."""
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
import zipfile

import pytest

from constraint_memory.benchmark_adapter.planner import PlannerWorker
from constraint_memory.polza import ProviderError
from experiments.cross_family_continuation import core


ROOT = Path(__file__).resolve().parents[2]
PARENT_ID = "cross-family-recovery-second-20261016T183331333825Z"
ORIGINAL_ID = "cross-family-primary-20261015T192052114101Z"
FIRST_ID = "cross-family-recovery-20261016T144804072163Z"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def write_rows(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(value, allow_nan=False) + "\n" for value in values), encoding="utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def forbidden(*args, **kwargs):
    pytest.fail("Network, hidden-answer or scoring access is forbidden in these tests")


@pytest.fixture(autouse=True)
def deny_network(monkeypatch):
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


@pytest.fixture
def fake_parent(tmp_path):
    # The inference contract is public configuration; all messages and receipts
    # below are invented and contain no task answers or real user prompts.
    config = json.loads((ROOT / "runs" / ORIGINAL_ID / "manifest.json").read_text(encoding="utf-8"))["config"]
    entries = []
    for index, operation in enumerate(("extraction", "agent"), 1):
        intent = {"attempt_id": f"request-{index:05d}", "event": "INTENT_ONLY_NOT_DISPATCH_PROOF",
            "group_id": "E1:10", "traveler": "InventedTraveler", "operation": operation,
            "messages": [{"role": "user", "content": f"Invented offline {operation}."}],
            "tools": None, "max_tokens_override": None, "json_output": False, "response_schema": None}
        raw = {"id": f"invented-accepted-{index}", "model": config["agent"]["model"], "provider": "Chutes",
            "choices": [{"message": {"role": "assistant", "content": "{}"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5, "cost_rub": .25}}
        usage = {"group_id": intent["group_id"], "traveler": intent["traveler"], "operation": operation,
            "request_id": raw["id"], "request_status": "completed", "response_accepted": True,
            "model_identity_validation": "EXACT_MATCH", "model": raw["model"], "input_tokens": 2,
            "output_tokens": 3, "total_tokens": 5, "cost_rub": .25, "currency": "RUB", "latency_seconds": 23.}
        entries.append({"intent": intent, "raw_response": raw, "usage": usage,
            "origin_run_id": ORIGINAL_ID if index == 1 else PARENT_ID})
    failed_intent = {"attempt_id": "request-00003", "event": "INTENT_ONLY_NOT_DISPATCH_PROOF",
        "group_id": "E2:197", "traveler": "Violet", "operation": "agent",
        "messages": [{"role": "user", "content": "Invented exact continuation boundary."}],
        "tools": None, "max_tokens_override": None, "json_output": False, "response_schema": None}
    failures = []
    for index, origin in enumerate((ORIGINAL_ID, FIRST_ID, PARENT_ID)):
        failures.append({"group_id": "E2:69" if index < 2 else "E2:197", "traveler": "Olivia" if index < 2 else "Violet",
            "operation": "agent", "request_sent": True, "request_status": "failed", "request_id": None,
            "input_tokens": None, "output_tokens": None, "total_tokens": None, "cost_rub": None,
            "billing_status": "unknown_after_failed_request", "origin_run_id": origin,
            "arrival_index": 7 if index < 2 else 2})
    directory = tmp_path / "invented-parent"
    directory.mkdir()
    return {"run_id": PARENT_ID, "directory": directory,
        "manifest": {"run_id": PARENT_ID, "config": config, "status": "failed"},
        "protocol": {"config": config, "schedule": [{"variant": "E1", "group_id": "10"}, {"variant": "E2", "group_id": "197"}]},
        "accepted": entries, "failed_intent": failed_intent, "failed_usage": failures[-1],
        "all_failures": failures, "failure_debits": {"E2:69": {7: 2}, "E2:197": {2: 1}},
        "lineage": [], "lineage_files_sha256": {}, "completed": [], "retrievals": [],
        "imported_model_responses": 0, "active_model_responses": 2}


def make_client(parent, destination, transport=forbidden, live=False, gate_callback=None):
    destination.mkdir(parents=True, exist_ok=True)
    agent = parent["manifest"]["config"]["agent"]
    return core.ContinuationClient(parent, destination, live=live, gate_callback=gate_callback,
        api_key="offline-invented-not-a-credential", model=agent["model"], temperature=agent["temperature"],
        max_tokens=agent["max_completion_tokens"], extraction_max_tokens=agent["max_extraction_tokens"],
        seed=20261013, provider=copy.deepcopy(agent["routing"]), reasoning=copy.deepcopy(agent["reasoning"]),
        extraction_reasoning=copy.deepcopy(agent["extraction_reasoning"]), stream=False,
        max_total_tokens=math.inf, max_cost_rub=None, transport=transport, usage_path=destination / "usage.jsonl")


def issue(client, intent):
    client.set_scope(intent["group_id"], intent["traveler"])
    return client.complete(copy.deepcopy(intent["messages"]), tools=copy.deepcopy(intent["tools"]),
        operation=intent["operation"], max_tokens=intent["max_tokens_override"],
        json_output=intent["json_output"], response_schema=copy.deepcopy(intent["response_schema"]))


def test_cached_receipts_keep_distinct_original_provenance_and_are_not_purchased(fake_parent, tmp_path):
    before = copy.deepcopy(fake_parent)
    with make_client(fake_parent, tmp_path / "replay") as client:
        for entry in fake_parent["accepted"]:
            assert issue(client, entry["intent"]) == entry["raw_response"]
        assert len(client.records) == 2
        assert all(row["reused"] is True and row["new_billable"] is False for row in client.records)
        assert [row["origin_run_id"] for row in client.records] == [ORIGINAL_ID, PARENT_ID]
        stats = client.get_usage_stats()
        assert stats["new_request_attempts"] == 0
        assert Decimal(stats["combined_known_cost_rub_exact"]) == Decimal(".50")
    assert fake_parent == before


def test_all_three_inherited_unknown_bills_remain_unknown_and_separate(fake_parent, tmp_path):
    with make_client(fake_parent, tmp_path / "replay") as client:
        for entry in fake_parent["accepted"]:
            issue(client, entry["intent"])
        stats = client.get_usage_stats()
        assert stats["inherited_failed_attempts"] == 3
        assert len(stats["inherited_failures"]) == 3
        assert {row["origin_run_id"] for row in stats["inherited_failures"]} == {ORIGINAL_ID, FIRST_ID, PARENT_ID}
        assert all(row["cost_rub"] is None and row["total_tokens"] is None for row in stats["inherited_failures"])
        assert stats["total_cost_rub"] is None
        assert stats["total_input_tokens"] is None and stats["total_output_tokens"] is None
        assert Decimal(stats["new_known_cost_rub_exact"]) == 0


@pytest.mark.parametrize("scope,arrival,allowance,debit", [
    ("E2:69", 7, 28, 2), ("E2:197", 2, 29, 1), ("E2:197", 1, 30, 0),
    ("E1:197", 2, 30, 0), ("E2:69", 6, 30, 0), ("E2:254", 2, 30, 0),
])
def test_failed_dispatch_debits_are_per_scope_and_arrival(monkeypatch, fake_parent, scope, arrival, allowance, debit):
    def fake_turn(self, request, memory_context, base_person):
        return {"total_steps": 2, "parent_model_requests": 2, "final_plan": "invented"}
    monkeypatch.setattr(PlannerWorker, "run_turn", fake_turn)
    planner = core.ContinuationPlanner(ROOT / "external/MemoryArena", ROOT / "data/tool_db",
        SimpleNamespace(group_id=scope, parent=fake_parent), model="qwen/qwen3.6-27b",
        temperature=0, max_steps=30, causal_round=arrival)
    assert planner.max_steps == allowance
    result = planner.run_turn({}, "invented memory", {})
    assert result["total_steps"] == result["parent_model_requests"] == 2 + debit


def test_offline_boundary_is_checked_before_gate_and_never_dispatches(fake_parent, tmp_path):
    calls = []
    with make_client(fake_parent, tmp_path / "replay", gate_callback=lambda: calls.append("gate")) as client:
        for entry in fake_parent["accepted"]:
            issue(client, entry["intent"])
        assert calls == []
        with pytest.raises(core.base.ReplayBoundaryReached) as stopped:
            issue(client, fake_parent["failed_intent"])
        assert stopped.value.request_sent is False
        assert calls == ["gate"]
        assert client.get_usage_stats()["new_request_attempts"] == 0


def test_changed_failed_prompt_prevents_gate_or_new_request(fake_parent, tmp_path):
    calls = []
    with make_client(fake_parent, tmp_path / "replay", live=True, gate_callback=lambda: calls.append("gate")) as client:
        for entry in fake_parent["accepted"]:
            issue(client, entry["intent"])
        altered = copy.deepcopy(fake_parent["failed_intent"])
        altered["messages"][0]["content"] += " altered"
        with pytest.raises((ValueError, RuntimeError)):
            issue(client, altered)
        assert calls == []


def test_new_sent_failure_is_not_retried_or_added_as_zero(fake_parent, tmp_path):
    calls = []
    def failed_transport(*args):
        calls.append(args)
        raise ProviderError("Invented next sent failure", request_sent=True)
    with make_client(fake_parent, tmp_path / "replay", failed_transport, live=True, gate_callback=lambda: None) as client:
        for entry in fake_parent["accepted"]:
            issue(client, entry["intent"])
        with pytest.raises(ProviderError, match="Invented next"):
            issue(client, fake_parent["failed_intent"])
        assert len(calls) == 1 and client.records[-1]["cost_rub"] is None
        assert client.get_usage_stats()["total_cost_rub"] is None


def test_closed_session_import_validates_receipts_and_keeps_causal_logs_without_native_replanning(fake_parent, tmp_path):
    parent = copy.deepcopy(fake_parent)
    parent["imported_model_responses"] = 2
    parent["active_model_responses"] = 0
    local = {"group_id": "E1:10", "traveler": "InventedTraveler", "operation": "memory_observe", "model_call": False,
        "input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "cost_rub": 0, "latency_seconds": .01}
    row = {"group_id": "10", "variant": "E1", "usage": [copy.deepcopy(local)] + [copy.deepcopy(entry["usage"]) for entry in parent["accepted"]],
        "wall_clock_seconds": 18., "travelers": [{"name": "InventedTraveler"}]}
    checkpoint = {"execution_index": 1, "group_id": "10", "variant": "E1", "predictions": [{"name": "InventedTraveler"}], "row": row}
    parent["completed"] = [checkpoint]
    closed_retrieval = {"variant": "E1", "group_id": "10", "traveler": "InventedTraveler", "text": "invented memory", "required_closure_ids": []}
    parent["retrievals"] = [closed_retrieval]
    logs = {name: [] for name in ("events.jsonl", "memory_ops.jsonl", "extractions.jsonl", "retrievals.jsonl", "recoveries.jsonl",
        "submission_events.jsonl", "predictions.jsonl", "traces.jsonl", "session_completions.jsonl")}
    logs["events.jsonl"] = [{"variant": "E1", "group_id": "10", "content": "invented observed fact"}]
    logs["memory_ops.jsonl"] = [{"variant": "E1", "group_id": "10", "op": "observe", "arg": {"invented": True}}]
    logs["retrievals.jsonl"] = [closed_retrieval]
    logs["session_completions.jsonl"] = [checkpoint]
    logs["predictions.jsonl"] = [{"variant": "E1", "group_id": "10", "name": "InventedTraveler"}]
    for name, values in logs.items():
        write_rows(parent["directory"] / name, values)
    write_json(parent["directory"] / "snapshots/E1/10.json", {"invented_causal_state": {"version": 1}})
    write_rows(parent["directory"] / "usage.jsonl", [local] + [entry["usage"] for entry in parent["accepted"]])
    before = {path.relative_to(parent["directory"]).as_posix(): sha(path) for path in parent["directory"].rglob("*") if path.is_file()}
    destination = tmp_path / "imported"
    with make_client(parent, destination) as client:
        imported, proof = core.import_completed(parent, destination, client)
        assert imported == [checkpoint]
        assert proof["retained_completed_sessions"] == 1
        assert proof["retained_model_responses"] == 2
        assert proof["native_receipt_validation"] is True
        assert proof["full_native_execution"] is False
        assert proof["native_active_replayed"] == 0
        assert client.cursor == 2
        assert client.get_usage_stats()["new_request_attempts"] == 0
        assert client.retrieval_cursor == 1
        assert [receipt["origin_run_id"] for receipt in client.records] == [ORIGINAL_ID, PARENT_ID]
    for name, values in logs.items():
        path = destination / name
        actual = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
        assert actual == values
    assert sha(destination / "snapshots/E1/10.json") == before["snapshots/E1/10.json"]
    imported_usage = [json.loads(line) for line in (destination / "usage.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(imported_usage) == 3 and imported_usage[0] == local
    assert all(receipt["reused"] is True and receipt["new_billable"] is False for receipt in imported_usage[1:])
    assert {path.relative_to(parent["directory"]).as_posix(): sha(path) for path in parent["directory"].rglob("*") if path.is_file()} == before


@pytest.fixture
def lineage_root(tmp_path):
    root = tmp_path / "lineage"
    (root / "runs").mkdir(parents=True)
    write_json(root / "reports/cross_family_recovery/live_lineage_claim.json", {"parent_run_id": ORIGINAL_ID, "child_run_id": FIRST_ID})
    write_json(root / "reports/cross_family_recovery_second/live_lineage_claim.json", {"parent_run_id": FIRST_ID, "child_run_id": PARENT_ID})
    return root


def invented_child(root, name="arbitrary-child", status="failed", attempts=0):
    child = root / "runs" / name
    child.mkdir()
    write_json(child / "manifest.json", {"run_id": name, "parent_run_id": PARENT_ID, "inherited_accepted_responses": 202,
        "status": status, "usage": {"new_request_attempts": attempts}})
    write_rows(child / "request_journal.jsonl", [
        {"event": "INTENT_ONLY_NOT_DISPATCH_PROOF", "attempt_id": "request-00001"},
        {"event": "RESPONSE_ACCEPTED", "attempt_id": "request-00001"}])
    write_rows(child / "usage.jsonl", [{"operation": "agent", "request_status": "completed", "reused": True,
        "new_billable": False, "model": "qwen/qwen3.6-27b", "cost_rub": .25}])
    return child


def test_existing_legacy_claims_are_preserved_and_allow_new_audited_lineage(lineage_root):
    old = {str(path): sha(path) for path in (lineage_root / "reports").rglob("*.json")}
    core.verify_unused(lineage_root, PARENT_ID)
    assert {str(path): sha(path) for path in (lineage_root / "reports").rglob("*.json")} == old


@pytest.mark.parametrize("status,attempts", [("complete", 0), ("running", 0), ("failed", 1), ("interrupted", None), ("failed", False)])
def test_paid_or_ambiguous_arbitrary_named_children_block(lineage_root, status, attempts):
    invented_child(lineage_root, status=status, attempts=attempts)
    with pytest.raises((ValueError, RuntimeError)):
        core.verify_unused(lineage_root, PARENT_ID)


def test_terminal_proven_cache_only_child_allows_new_reconstruction(lineage_root):
    invented_child(lineage_root)
    core.verify_unused(lineage_root, PARENT_ID)


@pytest.mark.parametrize("marker", ["replacement_attempt.json", "second_replacement_attempt.json", "continuation_attempt.json"])
def test_durable_dispatch_marker_blocks_even_with_zero_summary(lineage_root, marker):
    child = invented_child(lineage_root)
    write_json(child / marker, {"parent_run_id": PARENT_ID})
    with pytest.raises((ValueError, RuntimeError)):
        core.verify_unused(lineage_root, PARENT_ID)


def test_excluding_current_child_never_hides_another_paid_child(lineage_root):
    current = invented_child(lineage_root, "current", status="replaying_prefix")
    core.verify_unused(lineage_root, PARENT_ID, exclude_child=current)
    invented_child(lineage_root, "previous-paid", attempts=1)
    with pytest.raises((ValueError, RuntimeError)):
        core.verify_unused(lineage_root, PARENT_ID, exclude_child=current)


def test_unknown_model_receipt_cannot_hide_behind_zero_summary(lineage_root):
    child = invented_child(lineage_root)
    write_rows(child / "usage.jsonl", [{"model_call": True, "operation": "agent", "cost_rub": .25}])
    with pytest.raises((ValueError, RuntimeError)):
        core.verify_unused(lineage_root, PARENT_ID)


def test_claim_is_exclusive_durable_and_preserves_both_old_claims(lineage_root):
    old = {str(path): sha(path) for path in (lineage_root / "reports").rglob("*.json")}
    child, duplicate = lineage_root / "runs/first", lineage_root / "runs/duplicate"
    child.mkdir()
    duplicate.mkdir()
    core.claim(lineage_root, PARENT_ID, child)
    claims = list((lineage_root / "reports/cross_family_continuation/claims").glob("*.json"))
    assert len(claims) == 1
    receipt = json.loads(claims[0].read_text(encoding="utf-8"))
    assert receipt["parent_run_id"] == PARENT_ID and receipt["child_run_id"] == child.name
    after_first = sha(claims[0])
    with pytest.raises((ValueError, RuntimeError, FileExistsError)):
        core.claim(lineage_root, PARENT_ID, duplicate)
    assert sha(claims[0]) == after_first
    assert all(sha(path) == checksum for path, checksum in old.items())
    with pytest.raises((ValueError, RuntimeError)):
        core.verify_unused(lineage_root, PARENT_ID)


@pytest.mark.parametrize("body", ["{partial", "null", "[]", '{"parent_run_id":"other"}'])
def test_malformed_claim_fails_closed(lineage_root, body):
    path = lineage_root / "reports/cross_family_continuation/claims" / f"{PARENT_ID}.json"
    path.parent.mkdir(parents=True)
    path.write_text(body, encoding="utf-8")
    with pytest.raises((ValueError, RuntimeError, TypeError)):
        core.verify_unused(lineage_root, PARENT_ID)


@pytest.fixture(scope="module")
def actual_parent():
    return core.load_parent(ROOT, PARENT_ID)


def test_actual_parent_has_202_saved_completions_and_three_failures(actual_parent):
    assert actual_parent["run_id"] == PARENT_ID
    assert len(actual_parent["accepted"]) == 202
    assert len(actual_parent["all_failures"]) == 3
    assert len(actual_parent["completed"]) == 9
    assert actual_parent["imported_model_responses"] == 198
    assert actual_parent["active_model_responses"] == 4
    assert actual_parent["failure_debits"] == {"E2:69": {7: 2}, "E2:197": {2: 1}}
    assert all(row["cost_rub"] is None and row["total_tokens"] is None for row in actual_parent["all_failures"])
    assert Decimal(str(sum(Decimal(str(entry["usage"]["cost_rub"])) for entry in actual_parent["accepted"]))) == Decimal("354.73968545")


def test_actual_load_never_reads_credentials_hidden_answers_or_scores(monkeypatch):
    original_open, original_path_open = builtins.open, Path.open
    def checked(path):
        if isinstance(path, (str, Path)):
            value = str(path).replace("\\", "/").lower()
            assert "/data/hidden/" not in value and "/data/audit_annotations/" not in value
            assert Path(path).name.lower() != ".env"
    def guarded_open(path, *args, **kwargs):
        checked(path)
        return original_open(path, *args, **kwargs)
    def guarded_path_open(path, *args, **kwargs):
        checked(path)
        return original_path_open(path, *args, **kwargs)
    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(Path, "open", guarded_path_open)
    value = core.load_parent(ROOT, PARENT_ID)
    assert len(value["accepted"]) == 202


@pytest.mark.parametrize("change", ["raw_response", "request_intent"])
def test_actual_load_rejects_an_inherited_cache_change_without_modifying_any_run(monkeypatch, change):
    original_read = core._read_receipts
    def altered(directory, manifest):
        accepted, failed_intent, failed_usage = original_read(directory, manifest)
        if Path(directory).name == PARENT_ID:
            if change == "raw_response":
                accepted[0]["raw_response"]["choices"][0]["message"]["content"] = "invented cache corruption"
            else:
                accepted[0]["intent"]["messages"][0]["content"] = "invented prompt corruption"
        return accepted, failed_intent, failed_usage
    monkeypatch.setattr(core, "_read_receipts", altered)
    with pytest.raises(core.ReplayMismatch, match="changed or resampled"):
        core.load_parent(ROOT, PARENT_ID)


@pytest.mark.parametrize("change", ["member_content", "unexpected_member", "archive_digest"])
def test_runtime_archive_integrity_rejects_changed_or_extra_members(tmp_path, change):
    archive = tmp_path / "invented_source.zip"
    body = b"# invented frozen source\n"
    expected = {"invented.py": hashlib.sha256(body).hexdigest()}
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("invented.py", body)
    original_digest = sha(archive)
    core._verify_zip(archive, expected, original_digest)
    if change == "member_content":
        with zipfile.ZipFile(archive, "w") as zipped:
            zipped.writestr("invented.py", body + b"# changed\n")
    elif change == "unexpected_member":
        with zipfile.ZipFile(archive, "a") as zipped:
            zipped.writestr("unexpected.py", b"# extra\n")
    with pytest.raises(core.ReplayMismatch):
        core._verify_zip(archive, expected, "0" * 64 if change == "archive_digest" else None)


def test_predecessor_pin_verification_rejects_mutation_without_touching_real_runs(tmp_path):
    directory = tmp_path / "invented-previous"
    write_json(directory / "receipt.json", {"stable": True})
    before = {"receipt.json": sha(directory / "receipt.json")}
    parent = {"lineage": [{"run_id": "invented", "directory": directory, "file_sha256": before}],
        "lineage_files_sha256": {"invented": before}}
    assert core.verify_all_predecessors(parent) is True
    write_json(directory / "receipt.json", {"stable": False})
    assert core.verify_all_predecessors(parent) is False


@pytest.fixture
def semantic_pair(tmp_path):
    old, new = tmp_path / "semantic-old", tmp_path / "semantic-new"
    names = ("events.jsonl", "memory_ops.jsonl", "extractions.jsonl", "retrievals.jsonl", "recoveries.jsonl",
        "submission_events.jsonl", "predictions.jsonl", "traces.jsonl", "session_completions.jsonl")
    payloads = {name: [] for name in names}
    payloads["events.jsonl"] = [{"variant": "E1", "group_id": "10", "event_id": "invented:1", "content": "invented exact observation"}]
    payloads["retrievals.jsonl"] = [{"variant": "E1", "group_id": "10", "text": "invented exact memory", "closure_ids": ["a", "b"]}]
    payloads["traces.jsonl"] = [{"variant": "E1", "group_id": "10", "result": {"final_plan": "invented chosen action",
        "latency_seconds": 1., "structured_tool_results": [{"output": "invented tool result", "latency_seconds": 2.}]}}]
    payloads["session_completions.jsonl"] = [{"variant": "E1", "group_id": "10", "row": {
        "wall_clock_seconds": 3., "usage": [{"latency_seconds": 1., "cost_rub": .25, "input_tokens": 2}]}}]
    for directory in (old, new):
        for name, content in payloads.items():
            write_rows(directory / name, content)
        write_json(directory / "snapshots/E1/10.json", {"decisions": [{"version": 1, "value": "invented choice"}]})
    return {"directory": old}, new, payloads


def test_semantic_comparison_excludes_only_documented_timing_and_replay_annotations(semantic_pair):
    parent, target, payloads = semantic_pair
    altered = copy.deepcopy(payloads["traces.jsonl"])
    altered[0]["result"]["latency_seconds"] = 100.
    altered[0]["result"]["structured_tool_results"][0]["latency_seconds"] = 200.
    write_rows(target / "traces.jsonl", altered)
    altered = copy.deepcopy(payloads["session_completions.jsonl"])
    altered[0]["row"]["wall_clock_seconds"] = 300.
    altered[0]["row"]["usage"][0].update(latency_seconds=100., replay_latency_seconds=.1,
        reused=True, new_billable=False, origin_run_id="invented-origin", parent_attempt_id="invented-attempt")
    write_rows(target / "session_completions.jsonl", altered)
    proof = core.compare_prefix(parent, target)
    assert proof["status"] == "exact_semantic_prefix" and proof["snapshots"] == 1


@pytest.mark.parametrize("kind", ["observation", "memory", "closure_order", "action", "tool_output", "receipt", "snapshot", "extra_row"])
def test_semantic_comparison_never_weakens_prompt_action_state_or_receipt_equality(semantic_pair, kind):
    parent, target, payloads = semantic_pair
    if kind == "snapshot":
        write_json(target / "snapshots/E1/10.json", {"decisions": [{"version": 2, "value": "invented choice"}]})
    else:
        name = {"observation": "events.jsonl", "memory": "retrievals.jsonl", "closure_order": "retrievals.jsonl",
            "action": "traces.jsonl", "tool_output": "traces.jsonl", "receipt": "session_completions.jsonl", "extra_row": "events.jsonl"}[kind]
        altered = copy.deepcopy(payloads[name])
        if kind == "observation":
            altered[0]["content"] += " changed"
        elif kind == "memory":
            altered[0]["text"] += " changed"
        elif kind == "closure_order":
            altered[0]["closure_ids"].reverse()
        elif kind == "action":
            altered[0]["result"]["final_plan"] += " changed"
        elif kind == "tool_output":
            altered[0]["result"]["structured_tool_results"][0]["output"] += " changed"
        elif kind == "receipt":
            altered[0]["row"]["usage"][0]["cost_rub"] = .0
        elif kind == "extra_row":
            altered.append(copy.deepcopy(altered[0]))
        write_rows(target / name, altered)
    with pytest.raises((ValueError, RuntimeError)):
        core.compare_prefix(parent, target)


FOURTH_ID = "cross-family-continuation-20261016T234548697206Z"


def writer_intent(round_idx=1, traveler="InventedWriter", scope="E2:10"):
    return {"attempt_id": "request-00001", "event": "INTENT_ONLY_NOT_DISPATCH_PROOF",
        "group_id": scope, "traveler": traveler, "operation": "extraction",
        "messages": [{"role": "user", "content": json.dumps({"current_request": {
            "group_id": scope.split(":", 1)[-1], "traveler": traveler, "round_idx": round_idx}})}],
        "tools": None, "max_tokens_override": None, "json_output": True, "response_schema": None}


@pytest.mark.parametrize("prior_writers,expected", [(0, 1), (1, 2), (6, 7)])
def test_failed_writer_binds_next_released_arrival_without_using_traveler_name_as_ordinal(prior_writers, expected):
    accepted = [{"intent": {"group_id": "E2:10", "operation": "extraction", "traveler": "RepeatedInventedName"}}
        for _ in range(prior_writers)]
    accepted += [{"intent": {"group_id": "E1:10", "operation": "extraction", "traveler": "OtherScope"}},
        {"intent": {"group_id": "E2:10", "operation": "agent", "traveler": "RepeatedInventedName"}}]
    assert core._failure_arrival(accepted, writer_intent(expected, "RepeatedInventedName")) == expected


@pytest.mark.parametrize("changed", ["round", "round_bool", "traveler", "group", "missing_payload", "beyond_seven"])
def test_failed_writer_rejects_unreleased_or_mismatched_causal_request(changed):
    intent = writer_intent()
    accepted = []
    payload = json.loads(intent["messages"][-1]["content"])
    if changed == "round":
        payload["current_request"]["round_idx"] = 2
    elif changed == "round_bool":
        payload["current_request"]["round_idx"] = True
    elif changed == "traveler":
        payload["current_request"]["traveler"] = "DifferentInventedWriter"
    elif changed == "group":
        payload["current_request"]["group_id"] = "11"
    elif changed == "missing_payload":
        payload = {"irrelevant": "invented"}
    elif changed == "beyond_seven":
        accepted = [{"intent": {"group_id": "E2:10", "operation": "extraction", "traveler": "InventedWriter"}} for _ in range(7)]
        payload["current_request"]["round_idx"] = 8
    intent["messages"][-1]["content"] = json.dumps(payload)
    with pytest.raises(core.ReplayMismatch):
        core._failure_arrival(accepted, intent)


@pytest.fixture
def unsent_writer_journal(tmp_path):
    directory = tmp_path / "invented-unsent-writer"
    directory.mkdir()
    intent = writer_intent()
    failure = {"group_id": intent["group_id"], "traveler": intent["traveler"], "operation": "extraction",
        "request_status": "failed", "request_sent": False, "request_id": None, "billing_status": "not_sent",
        "input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "cost_rub": 0}
    event = {"event": "REQUEST_FAILED", "attempt_id": intent["attempt_id"], "request_sent": False, "usage": [copy.deepcopy(failure)]}
    write_rows(directory / "request_journal.jsonl", [intent, event])
    write_rows(directory / "usage.jsonl", [failure])
    write_rows(directory / "committed_responses.jsonl", [])
    return directory, {"run_id": "invented-unsent-writer"}, intent, failure, event


def save_failure_fixture(directory, intent, failure, event):
    write_rows(directory / "request_journal.jsonl", [intent, event])
    write_rows(directory / "usage.jsonl", [failure])


def test_definitely_unsent_writer_requires_and_preserves_explicit_zero_usage(unsent_writer_journal):
    directory, manifest, intent, failure, event = unsent_writer_journal
    accepted, actual_intent, actual_failure = core._read_receipts(directory, manifest)
    assert accepted == [] and actual_intent == intent and actual_failure == failure
    assert actual_failure["request_sent"] is False
    assert all(actual_failure[key] == 0 for key in ("input_tokens", "output_tokens", "total_tokens", "cost_rub"))


@pytest.mark.parametrize("field,value", [
    ("request_sent", None), ("request_sent", "false"), ("request_sent", 0),
    ("billing_status", "unknown_after_failed_request"), ("request_id", "invented-remote-id"),
    ("input_tokens", None), ("output_tokens", "0"), ("total_tokens", False), ("cost_rub", .01),
])
def test_unsent_writer_rejects_ambiguous_flag_or_nonzero_unverified_billing(unsent_writer_journal, field, value):
    directory, manifest, intent, failure, event = unsent_writer_journal
    failure[field] = value
    event["usage"] = [copy.deepcopy(failure)]
    if field == "request_sent":
        event["request_sent"] = value
    save_failure_fixture(directory, intent, failure, event)
    with pytest.raises(core.ReplayMismatch):
        core._read_receipts(directory, manifest)


def test_unsent_journal_and_native_usage_must_report_identical_bill(unsent_writer_journal):
    directory, manifest, intent, failure, event = unsent_writer_journal
    event["usage"][0]["cost_rub"] = .25
    save_failure_fixture(directory, intent, failure, event)
    with pytest.raises(core.ReplayMismatch, match="accounting differs"):
        core._read_receipts(directory, manifest)


@pytest.mark.parametrize("change", ["zero_unknown_bill", "sent_but_labeled_not_sent"])
def test_sent_failure_cannot_be_labeled_free_without_a_verified_provider_receipt(unsent_writer_journal, change):
    directory, manifest, intent, failure, event = unsent_writer_journal
    failure.update(request_sent=True, billing_status="unknown_after_failed_request",
        input_tokens=None, output_tokens=None, total_tokens=None, cost_rub=None)
    event["request_sent"] = True
    if change == "zero_unknown_bill":
        failure["cost_rub"] = 0
    else:
        failure["billing_status"] = "not_sent"
        failure.update(input_tokens=0, output_tokens=0, total_tokens=0, cost_rub=0)
    event["usage"] = [copy.deepcopy(failure)]
    save_failure_fixture(directory, intent, failure, event)
    with pytest.raises(core.ReplayMismatch):
        core._read_receipts(directory, manifest)


@pytest.mark.parametrize("sent", [True, None, "unknown"])
def test_unsent_proof_cannot_override_a_conflicting_post_dispatch_marker(unsent_writer_journal, sent):
    directory, manifest, intent, failure, event = unsent_writer_journal
    write_json(directory / "new_requests/attempt00001/dispatch.json",
        {"logical_attempt_id": intent["attempt_id"], "request_sent": sent})
    with pytest.raises(core.ReplayMismatch, match="conflicting POST"):
        core._read_receipts(directory, manifest)


def test_real_fourth_stop_writer_is_separate_from_all_actor_budgets(monkeypatch):
    parent = core.load_parent(ROOT, FOURTH_ID)
    assert len(parent["accepted"]) == 219 and len(parent["completed"]) == 10
    assert parent["imported_model_responses"] == 219 and parent["active_model_responses"] == 0
    assert parent["failure_debits"] == {"E2:69": {7: 2}, "E2:197": {2: 1}}
    assert parent["writer_failure_debits"] == {"E2:64": {1: 1}}
    policy = parent["writer_attempt_policy"]
    assert policy["accepted_writers_per_arrival"] == 1
    assert policy["original_max_writer_requests"] == 98 and policy["inherited_sent_writer_failures"] == 1
    assert policy["effective_max_writer_requests"] == 99 and policy["classification"] == "infra_recovered_diagnostic"
    assert len(parent["all_failures"]) == 4 and all(row["cost_rub"] is None for row in parent["all_failures"])
    monkeypatch.setattr(PlannerWorker, "run_turn", lambda *args: {"total_steps": 2, "parent_model_requests": 2})
    planner = core.ContinuationPlanner(ROOT / "external/MemoryArena", ROOT / "data/tool_db",
        SimpleNamespace(group_id="E2:64", parent=parent), model="qwen/qwen3.6-27b", temperature=0, max_steps=30, causal_round=1)
    assert planner.max_steps == 30
    result = planner.run_turn({}, "invented memory", {})
    assert result["total_steps"] == result["parent_model_requests"] == 2


def test_definitely_unsent_writer_adds_neither_actor_nor_writer_debit(monkeypatch):
    original_read = core._read_receipts
    def unsent(directory, manifest):
        accepted, intent, failure = original_read(directory, manifest)
        if Path(directory).name == FOURTH_ID:
            failure = {**failure, "request_sent": False, "request_id": None, "billing_status": "not_sent",
                "input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "cost_rub": 0}
        return accepted, intent, failure
    # Receipt parser's zero-proof cases are tested independently above. This
    # isolated injection tests only how a validated unsent receipt affects the
    # lineage budget assembler; no real receipt or run file is changed.
    monkeypatch.setattr(core, "_read_receipts", unsent)
    parent = core.load_parent(ROOT, FOURTH_ID)
    assert parent["failure_debits"] == {"E2:69": {7: 2}, "E2:197": {2: 1}}
    assert parent["writer_failure_debits"] == {}
    assert parent["writer_attempt_policy"]["effective_max_writer_requests"] == 98
    assert parent["writer_attempt_policy"]["inherited_sent_writer_failures"] == 0
    assert parent["all_failures"][-1]["cost_rub"] == 0 and parent["all_failures"][-1]["request_sent"] is False


@pytest.mark.parametrize("bad", ["parent", "child", "directory", "missing"])
def test_ancestor_exclusive_claim_must_link_exact_parent_child_and_directory(tmp_path, bad):
    directory = tmp_path / "runs/invented-child"
    directory.mkdir(parents=True)
    write_json(directory / "continuation_source.json", {"invented_sidecar": True})
    manifest = {"run_id": "invented-child", "parent_run_id": "invented-parent"}
    claim_path = tmp_path / core.CLAIMS_RELATIVE / "invented-parent.json"
    receipt = {"parent_run_id": "invented-parent", "child_run_id": "invented-child", "child_directory": str(directory)}
    write_json(claim_path, receipt)
    relative, expected = core._ancestor_claim(tmp_path, directory, manifest)
    assert relative == claim_path.relative_to(tmp_path).as_posix() and expected == sha(claim_path)
    if bad == "parent":
        receipt["parent_run_id"] = "another-parent"
    elif bad == "child":
        receipt["child_run_id"] = "another-child"
    elif bad == "directory":
        receipt["child_directory"] = str(tmp_path / "different-directory")
    if bad == "missing":
        claim_path.unlink()
    else:
        write_json(claim_path, receipt)
    with pytest.raises(core.ReplayMismatch, match="claim is missing or conflicting"):
        core._ancestor_claim(tmp_path, directory, manifest)


def test_predecessor_verification_detects_claim_tampering_and_removal(tmp_path):
    directory = tmp_path / "runs/invented-child"
    write_json(directory / "receipt.json", {"stable": True})
    claim_path = tmp_path / core.CLAIMS_RELATIVE / "invented-parent.json"
    write_json(claim_path, {"parent_run_id": "invented-parent", "child_run_id": "invented-child"})
    parent = {"root": tmp_path, "lineage": [{"directory": directory, "file_sha256": {"receipt.json": sha(directory / "receipt.json")}}],
        "claim_files_sha256": {claim_path.relative_to(tmp_path).as_posix(): sha(claim_path)}}
    assert core.verify_all_predecessors(parent) is True
    write_json(claim_path, {"parent_run_id": "another-parent", "child_run_id": "invented-child"})
    assert core.verify_all_predecessors(parent) is False
    claim_path.unlink()
    assert core.verify_all_predecessors(parent) is False
