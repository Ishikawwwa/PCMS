"""Independent console-only progress checks with invented offline data."""
from __future__ import annotations

import copy
from decimal import Decimal
import json
from pathlib import Path
import socket
from types import SimpleNamespace

import pytest

from constraint_memory.polza import ProviderError
from . import progress


class Clock:
    def __init__(self):
        self.value = 0.

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_rows(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(value) + "\n" for value in values), encoding="utf-8")


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Progress tests must never access the network")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


@pytest.fixture
def parent(tmp_path):
    schedule = [{"variant": variant, "group_id": str(group)} for group in range(1, 8) for variant in ("E1", "E2")]
    completed = [{**item, "execution_index": index, "row": {"usage": [], "travelers": []}, "predictions": []}
        for index, item in enumerate(schedule[:10], 1)]
    traces, accepted = [], []
    for item in schedule[:11]:
        count = 7 if item in schedule[:10] else 4
        for arrival in range(1, count + 1):
            name = f"Invented{arrival}"
            scope = f"{item['variant']}:{item['group_id']}"
            traces.append({**item, "charged_round": arrival, "result": {"name": name, "success": arrival % 3 != 0}})
            for operation in ("extraction", "agent", "agent"):
                accepted.append({"intent": {"group_id": scope, "traveler": name, "operation": operation},
                    "usage": {"cost_rub": .01, "latency_seconds": 3., "operation": operation}})
    directory = tmp_path / "invented-parent"
    write_rows(directory / "traces.jsonl", traces)
    return {"directory": directory, "protocol": {"schedule": schedule, "bounds": {"added_traveler_commitments": 98}},
        "accepted": accepted, "completed": completed, "failure_debits": {"E1:6": {5: 0}},
        "all_failures": [{"cost_rub": None, "request_sent": True} for _ in range(4)]}


@pytest.fixture
def reporter(parent):
    clock, lines = Clock(), []
    value = progress.ProgressReporter(parent, sink=lines.append, clock=clock, background=False)
    try:
        yield value, clock, lines
    finally:
        value.close()


def test_reporter_starts_at_74_processed_travelers_and_ten_closed_sessions(reporter):
    value, clock, lines = reporter
    assert len(value.completed) == 74 and len(value.closed_sessions) == 10
    assert "travelers 74/98" in value.summary() and "sessions 10/14" in value.summary()
    assert "24 left" in value.summary()
    assert "including recorded unsuccessful outcomes" in "\n".join(lines)
    assert value.unknown == 4
    assert value.thread is None


def test_startup_does_not_present_unloaded_totals_as_zero():
    lines = []
    with progress.ProgressReporter(sink=lines.append, clock=Clock(), background=False) as value:
        value.phase("Validating saved ancestry")
        value.heartbeat()
        text = "\n".join(lines)
        assert "loading" in text
        assert "0/98" not in text and "0 RUB" not in text


def test_heartbeat_does_not_repeat_overall_counts_twice(reporter):
    value, clock, lines = reporter
    value.arrival_start("E1:6", 5, "Invented5")
    value.heartbeat()
    assert lines[-1].count("travelers 74/98") == 1
    assert lines[-1].count("sessions 10/14") == 1


def test_replayed_active_arrivals_do_not_inflate_completed_counts_or_eta_history(reporter):
    value, clock, lines = reporter
    before = list(value.history)
    for arrival in range(1, 5):
        value.arrival_start("E1:6", arrival, f"Invented{arrival}", replay=True)
        clock.advance(1)
        value.arrival_done("E1:6", arrival, f"Invented{arrival}", 1, replayed=True)
    assert len(value.completed) == 74 and value.history == before
    assert "travelers 74/98" in value.summary()
    value.arrival_start("E1:6", 5, "Invented5")
    value.request_result({"operation": "agent", "request_status": "completed", "response_accepted": True,
        "cost_rub": .01, "latency_seconds": 12, "input_tokens": 2, "output_tokens": 3}, finish_reason="stop")
    value.arrival_done("E1:6", 5, "Invented5", 12)
    value.arrival_done("E1:6", 5, "Invented5", 12)
    assert len(value.completed) == 75
    assert len(value.history) == len(before) + 1


def test_cached_actor_receipts_count_against_current_arrival_without_new_charge(reporter):
    value, clock, lines = reporter
    value.parent["failure_debits"] = {"E1:6": {1: 2}}
    value.arrival_start("E1:6", 1, "Invented1", replay=True)
    assert value.actor_used == 2
    before = value.new_posts, value.new_results, value.new_known
    for _ in range(2):
        value.request_result({"operation": "agent", "request_status": "completed", "response_accepted": True,
            "cost_rub": .25}, cached=True)
    assert value.actor_used == 4
    assert (value.new_posts, value.new_results, value.new_known) == before


def test_no_eta_is_invented_without_history_or_exact_request_count():
    lines, clock = [], Clock()
    with progress.ProgressReporter(sink=lines.append, clock=clock, background=False) as value:
        value.phase("waiting for connection")
        clock.advance(20)
        value.heartbeat()
        assert "ETA" not in value.summary()
        assert "64 calls" not in "\n".join(lines)
        assert "ALIVE" in lines[-1] and "20s" in lines[-1]


def test_historical_eta_is_explicitly_rough_and_never_a_token_percentage(reporter):
    value, clock, lines = reporter
    text = value.summary()
    assert "rough" in text and "ETA" in text and "--" in text
    assert "%" not in text and "64 calls" not in text


def test_preflight_progress_says_no_post_and_unsent_failure_adds_no_charge(reporter):
    value, clock, lines = reporter
    before = value.new_posts, value.new_results, value.new_known, value.unknown
    value("preflight_connect", attempt=1, max_attempts=3, timeout_seconds=10)
    clock.advance(10)
    value("preflight_failed", attempt=1, max_attempts=3, elapsed_seconds=10, error_type="TimeoutError", stage="HEAD")
    value("preflight_exhausted", attempt=3, max_attempts=3)
    value.request_result({"operation": "extraction", "request_status": "failed", "request_sent": False,
        "billing_status": "not_sent", "cost_rub": 0, "input_tokens": 0, "output_tokens": 0, "latency_seconds": 33})
    assert (value.new_posts, value.new_results, value.new_known, value.unknown) == before
    output = "\n".join(lines)
    assert "no model POST sent" in output and "no model charge" in output
    assert "model POST not sent" in output


def test_console_callback_ignores_prompt_reasoning_credentials_and_unknown_fields(reporter):
    value, clock, lines = reporter
    secret = "pza_" + "A" * 40
    marker = "INVENTED-SENSITIVE-PROMPT-DO-NOT-PRINT"
    value("preflight_failed", attempt=1, max_attempts=3, elapsed_seconds=10,
        error_type="TimeoutError", stage="HEAD", prompt=marker, reasoning=marker,
        api_key=secret, Authorization="Bearer " + secret, arbitrary_body={"content": marker})
    value("unrecognized_event", prompt=marker, reasoning=marker, api_key=secret)
    output = "\n".join(lines)
    assert secret not in output and marker not in output and "Bearer " not in output


def test_stream_heartbeat_reads_only_safe_counters_without_response_content(reporter, tmp_path):
    value, clock, lines = reporter
    directory = tmp_path / "invented-stream"
    directory.mkdir()
    value.watch_stream(directory)
    clock.advance(10)
    value.heartbeat()
    assert "waiting for HTTP headers" in lines[-1]
    secret = "INVENTED-PRIVATE-REASONING-DO-NOT-PRINT"
    (directory / "response.sse").write_text(secret, encoding="utf-8")
    write_json(directory / "response_headers.json", {"status": 200, "headers": {"Set-Cookie": secret}})
    write_rows(directory / "progress.jsonl", [{"chunks": 40, "reasoning_chars": 1234, "content_chars": 20,
        "tool_argument_chars": 9, "content": secret, "reasoning": secret, "api_key": "pza_" + "B" * 40}])
    clock.advance(10)
    value.heartbeat()
    assert "HTTP 200" in lines[-1] and "reasoning chars 1,234" in lines[-1]
    assert "answer chars 20" in lines[-1] and "tool-argument chars 9" in lines[-1]
    assert "chunks 40" in lines[-1] and secret not in "\n".join(lines)
    value.stop_stream()
    assert value.stream_directory is None


def test_stream_heartbeat_ignores_malformed_and_nonnumeric_counter_values(reporter, tmp_path):
    value, clock, lines = reporter
    directory = tmp_path / "invented-stream"
    directory.mkdir()
    write_json(directory / "response_headers.json", {"status": 200})
    marker = "INVENTED-CONTENT-MUST-NOT-BE-LOGGED"
    (directory / "progress.jsonl").write_text(json.dumps({"chunks": marker, "content_chars": -2,
        "reasoning_chars": True, "tool_argument_chars": None}) + "\n{partial", encoding="utf-8")
    value.watch_stream(directory)
    value.heartbeat()
    assert marker not in "\n".join(lines)
    assert "reasoning chars True" not in lines[-1] and "answer chars -2" not in lines[-1]


class Spy:
    def __init__(self, clock=None, broken=None):
        self.clock = clock or Clock()
        self.calls = []
        self.broken = broken

    def __getattr__(self, name):
        def observe(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            if self.broken == name or self.broken == "all":
                raise RuntimeError("Invented observer failure")
        return observe


class Recorded:
    def __init__(self, destination, result=None, failure=None):
        self.destination = destination
        self.attempts = 0
        self.result, self.failure = result, failure
        self.calls = []

    def __call__(self, method, path, payload):
        self.calls.append((method, path, payload))
        self.attempts += 1
        if self.failure is not None:
            raise self.failure
        return self.result


def test_transport_observer_preserves_payload_response_and_one_dispatch(tmp_path):
    payload = {"messages": [{"role": "user", "content": "invented private task"}], "model": "invented-model"}
    before = copy.deepcopy(payload)
    result = {"choices": [{"message": {"content": "invented private response"}}]}
    recorded, observer = Recorded(tmp_path, result), Spy()
    actual = progress.ObservedTransport(recorded, observer)("POST", "/chat/completions", payload)
    assert actual is result and payload == before
    assert len(recorded.calls) == 1 and recorded.calls[0][2] is payload
    assert [call[0] for call in observer.calls] == ["watch_stream", "stop_stream"]
    assert observer.calls[0][1][0] == tmp_path / "new_requests/attempt00001"


@pytest.mark.parametrize("broken", ["watch_stream", "stop_stream"])
def test_console_failure_cannot_prevent_or_duplicate_a_transport_request(tmp_path, broken):
    result = {"invented": "unchanged result"}
    recorded = Recorded(tmp_path, result)
    actual = progress.ObservedTransport(recorded, Spy(broken=broken))("POST", "/chat/completions", {"invented": True})
    assert actual is result and len(recorded.calls) == 1


def test_transport_cleanup_observer_cannot_replace_the_original_provider_failure(tmp_path):
    failure = ProviderError("Invented original ambiguous POST failure", request_sent=True)
    recorded = Recorded(tmp_path, failure=failure)
    with pytest.raises(ProviderError) as observed:
        progress.ObservedTransport(recorded, Spy(broken="stop_stream"))("POST", "/chat/completions", {"invented": True})
    assert observed.value is failure and len(recorded.calls) == 1


def client_without_constructor(observer, cached):
    client = object.__new__(progress.ObservedClient)
    client.progress, client.cursor, client.records = observer, 0, []
    client.parent = {"accepted": [{}] if cached else []}
    client.group_id, client.traveler, client.operation = "E1:10", "InventedTraveler", "agent"
    return client


@pytest.mark.parametrize("broken", [None, "request_start", "request_result", "cache_progress"])
def test_client_observation_keeps_native_arguments_result_and_cache_semantics(monkeypatch, broken):
    observer, calls = Spy(broken=broken), []
    client = client_without_constructor(observer, cached=True)
    messages = [{"role": "user", "content": "invented private prompt"}]
    tools = [{"type": "function", "function": {"name": "invented_tool", "parameters": {}}}]
    raw = {"choices": [{"finish_reason": "stop", "message": {"content": "invented private response"}}]}
    def native(self, *args):
        calls.append(args)
        self.records.append({"operation": "agent", "request_status": "completed", "response_accepted": True, "cost_rub": .25})
        self.cursor += 1
        return raw
    monkeypatch.setattr(progress.core.ContinuationClient, "complete", native)
    actual = client.complete(messages, tools=tools, operation="agent", max_tokens=30, json_output=True, response_schema={"invented": True})
    assert actual is raw and len(calls) == 1
    assert calls[0] == (messages, tools, "agent", 30, True, {"invented": True})
    assert calls[0][0] is messages and calls[0][1] is tools


def test_failed_client_observation_cannot_replace_native_exception_or_retry(monkeypatch):
    observer, calls = Spy(broken="request_result"), []
    client = client_without_constructor(observer, cached=False)
    failure = ProviderError("Invented original native failure", request_sent=True)
    def native(self, *args):
        calls.append(args)
        self.records.append({"operation": "agent", "request_status": "failed", "request_sent": True, "cost_rub": None})
        raise failure
    monkeypatch.setattr(progress.core.ContinuationClient, "complete", native)
    with pytest.raises(ProviderError) as observed:
        client.complete([{"role": "user", "content": "invented private prompt"}])
    assert observed.value is failure and len(calls) == 1


@pytest.mark.parametrize("broken", [None, "arrival_start", "arrival_done"])
def test_session_observation_preserves_native_packet_result_and_single_execution(broken):
    clock, calls = Clock(), []
    observer = Spy(clock, broken=broken)
    result = {"invented": "unchanged native result"}
    def native(packet):
        calls.append(packet)
        clock.advance(12)
        return result
    session = SimpleNamespace(scope="E1:10", run_arrival=native)
    client = SimpleNamespace(cursor=0, parent={"accepted": [{}]})
    assert progress.observe_session(session, observer, client) is session
    packet = {"traveler_index": 1, "name": "InventedTraveler", "query": "invented private task"}
    actual = session.run_arrival(packet)
    assert actual is result and calls == [packet] and calls[0] is packet


def test_session_observation_preserves_native_failure_and_does_not_mark_completed():
    observer, calls = Spy(), []
    failure = ProviderError("Invented original arrival failure", request_sent=False)
    def native(packet):
        calls.append(packet)
        raise failure
    session = SimpleNamespace(scope="E1:10", run_arrival=native)
    progress.observe_session(session, observer, SimpleNamespace(cursor=0, parent={"accepted": []}))
    with pytest.raises(ProviderError) as observed:
        session.run_arrival({"traveler_index": 1, "name": "InventedTraveler"})
    assert observed.value is failure and len(calls) == 1
    assert not any(call[0] == "arrival_done" for call in observer.calls)


def test_observer_clock_failure_cannot_prevent_native_arrival():
    def broken_clock():
        raise RuntimeError("Invented observer clock failure")
    observer, calls = Spy(clock=broken_clock), []
    session = SimpleNamespace(scope="E1:10", run_arrival=lambda packet: calls.append(packet) or "native result")
    progress.observe_session(session, observer, SimpleNamespace(cursor=0, parent={"accepted": []}))
    assert session.run_arrival({"traveler_index": 1, "name": "InventedTraveler"}) == "native result"
    assert len(calls) == 1
